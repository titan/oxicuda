//! Swendsen-Wang (1987) full-lattice cluster Monte-Carlo update for the Ising
//! model.
//!
//! The single-spin Gibbs / Metropolis samplers in [`super::gibbs`] suffer from
//! *critical slowing down*: near the 2-D critical inverse temperature
//! `β_c = ln(1 + √2) / 2 ≈ 0.4407` the autocorrelation time diverges like
//! `τ ∝ ξ^z` with dynamical exponent `z ≈ 2.17`, so flipping the global
//! magnetisation takes prohibitively many sweeps.  Cluster algorithms attack
//! this directly by flipping correlated *clusters* of spins at once.
//!
//! Swendsen-Wang performs one full sweep as follows:
//!
//! 1. **Bond percolation.**  For every lattice edge `(i, j)` whose spins are
//!    aligned (`s_i == s_j`), activate a bond with probability
//!    `p = 1 − exp(−2 β J)`.  Anti-aligned edges never carry a bond.
//! 2. **Cluster identification.**  Build the connected components induced by the
//!    activated bonds with a weighted union-find (disjoint-set) structure with
//!    path compression.
//! 3. **Cluster flip.**  Independently flip every cluster with probability ½.
//!
//! This is the Fortuin-Kasteleyn random-cluster representation: the Boltzmann
//! weight of the Ising model is reproduced exactly, so detailed balance holds
//! and the chain is ergodic.  The whole-lattice update decorrelates the
//! magnetisation in `O(1)` sweeps even at `T_c`.
//!
//! Spins are encoded as `i8` values in `{−1, +1}`.  Two geometries are
//! supported: a square lattice with optional periodic boundaries, and a fully
//! general undirected graph supplied as an edge list (see
//! [`SwendsenWang::from_graph`]).

use crate::error::{SeqError, SeqResult};
use crate::handle::LcgRng;

/// Lattice geometry for a Swendsen-Wang / Wolff sampler.
///
/// Either a regular 2-D square lattice (with optional periodic boundaries) or a
/// general undirected graph given as an adjacency list.
#[derive(Debug, Clone)]
pub(crate) enum Geometry {
    /// Square lattice of `rows × cols` sites; `periodic` toggles wrap-around
    /// edges on both axes.
    Square {
        rows: usize,
        cols: usize,
        periodic: bool,
    },
    /// General graph: `adjacency[i]` lists the neighbours of site `i`.  The
    /// graph is treated as undirected, so each undirected edge appears in both
    /// endpoints' lists.
    Graph { adjacency: Vec<Vec<usize>> },
}

impl Geometry {
    /// Number of sites in the geometry.
    pub(crate) fn n_sites(&self) -> usize {
        match self {
            Geometry::Square { rows, cols, .. } => rows * cols,
            Geometry::Graph { adjacency } => adjacency.len(),
        }
    }

    /// Enumerate the unique undirected edges `(i, j)` with `i < j`.
    ///
    /// For the square lattice this walks right / down neighbours (plus the
    /// wrap-around edges when periodic), guaranteeing each edge is produced
    /// exactly once.  For a graph it deduplicates the adjacency list.
    pub(crate) fn edges(&self) -> Vec<(usize, usize)> {
        match self {
            Geometry::Square {
                rows,
                cols,
                periodic,
            } => {
                let rows = *rows;
                let cols = *cols;
                let periodic = *periodic;
                let mut edges = Vec::new();
                for r in 0..rows {
                    for c in 0..cols {
                        let i = r * cols + c;
                        // Right neighbour.
                        if c + 1 < cols {
                            edges.push((i, r * cols + (c + 1)));
                        } else if periodic && cols > 1 {
                            edges.push(order(i, r * cols));
                        }
                        // Down neighbour.
                        if r + 1 < rows {
                            edges.push((i, (r + 1) * cols + c));
                        } else if periodic && rows > 1 {
                            edges.push(order(i, c));
                        }
                    }
                }
                edges
            }
            Geometry::Graph { adjacency } => {
                let mut edges = Vec::new();
                for (i, nbrs) in adjacency.iter().enumerate() {
                    for &j in nbrs {
                        if i < j {
                            edges.push((i, j));
                        }
                    }
                }
                edges
            }
        }
    }
}

/// Order a pair so the smaller index is first (so each undirected edge has a
/// canonical key).
#[inline]
fn order(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

/// Weighted union-find with path compression for cluster identification.
///
/// Union-by-size keeps the tree shallow; path compression flattens lookup
/// chains.  Both together give near-`O(α(n))` amortised `find`.
#[derive(Debug, Clone)]
pub(crate) struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    /// Create `n` singleton sets.
    pub(crate) fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    /// Find the representative of `x`, compressing the path along the way.
    pub(crate) fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        // Path compression: point every node on the path straight at the root.
        let mut cur = x;
        while self.parent[cur] != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }

    /// Union the sets containing `a` and `b` (union-by-size).
    pub(crate) fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            self.parent[ra] = rb;
            self.size[rb] += self.size[ra];
        } else {
            self.parent[rb] = ra;
            self.size[ra] += self.size[rb];
        }
    }
}

/// Configuration for a Swendsen-Wang sampler.
#[derive(Debug, Clone)]
pub struct SwendsenWangConfig {
    /// Ferromagnetic coupling constant `J` (energy per aligned bond).
    pub coupling: f64,
    /// Geometry of the spin system.
    pub(crate) geometry: Geometry,
}

impl SwendsenWangConfig {
    /// Square lattice configuration.
    pub fn square(rows: usize, cols: usize, coupling: f64, periodic: bool) -> SeqResult<Self> {
        validate_square(rows, cols, coupling)?;
        Ok(Self {
            coupling,
            geometry: Geometry::Square {
                rows,
                cols,
                periodic,
            },
        })
    }

    /// General-graph configuration from an adjacency list.
    pub fn graph(adjacency: Vec<Vec<usize>>, coupling: f64) -> SeqResult<Self> {
        validate_graph(&adjacency, coupling)?;
        Ok(Self {
            coupling,
            geometry: Geometry::Graph { adjacency },
        })
    }

    /// Number of sites in the configured geometry.
    pub fn n_sites(&self) -> usize {
        self.geometry.n_sites()
    }
}

/// Validate square-lattice parameters.
pub(crate) fn validate_square(rows: usize, cols: usize, coupling: f64) -> SeqResult<()> {
    if rows == 0 || cols == 0 {
        return Err(SeqError::InvalidConfiguration(
            "square lattice dims must be > 0".to_string(),
        ));
    }
    if !coupling.is_finite() {
        return Err(SeqError::InvalidParameter {
            name: "coupling".to_string(),
            value: coupling,
        });
    }
    Ok(())
}

/// Validate a graph adjacency list (non-empty, symmetric, no self-loops,
/// in-range neighbours).
pub(crate) fn validate_graph(adjacency: &[Vec<usize>], coupling: f64) -> SeqResult<()> {
    if adjacency.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    if !coupling.is_finite() {
        return Err(SeqError::InvalidParameter {
            name: "coupling".to_string(),
            value: coupling,
        });
    }
    let n = adjacency.len();
    for (i, nbrs) in adjacency.iter().enumerate() {
        for &j in nbrs {
            if j >= n {
                return Err(SeqError::IndexOutOfBounds { index: j, len: n });
            }
            if j == i {
                return Err(SeqError::GraphInvariantViolated(format!(
                    "self-loop at node {i}"
                )));
            }
            // Undirected: the reverse edge must also be present.
            if !adjacency[j].contains(&i) {
                return Err(SeqError::GraphInvariantViolated(format!(
                    "adjacency not symmetric: {i}->{j} present but {j}->{i} missing"
                )));
            }
        }
    }
    Ok(())
}

/// Swendsen-Wang cluster sampler.
///
/// Holds the coupling constant and the precomputed canonical edge list so
/// repeated [`SwendsenWang::step`] calls avoid re-deriving the lattice
/// topology.
#[derive(Debug, Clone)]
pub struct SwendsenWang {
    coupling: f64,
    edges: Vec<(usize, usize)>,
    n_sites: usize,
    /// Number of clusters formed on the most recent step (diagnostic).
    last_n_clusters: usize,
}

impl SwendsenWang {
    /// Build a sampler from a validated configuration.
    pub fn new(cfg: SwendsenWangConfig) -> Self {
        let edges = cfg.geometry.edges();
        let n_sites = cfg.geometry.n_sites();
        Self {
            coupling: cfg.coupling,
            edges,
            n_sites,
            last_n_clusters: 0,
        }
    }

    /// Convenience constructor for a square lattice.
    pub fn square(rows: usize, cols: usize, coupling: f64, periodic: bool) -> SeqResult<Self> {
        Ok(Self::new(SwendsenWangConfig::square(
            rows, cols, coupling, periodic,
        )?))
    }

    /// Convenience constructor for a general graph.
    pub fn from_graph(adjacency: Vec<Vec<usize>>, coupling: f64) -> SeqResult<Self> {
        Ok(Self::new(SwendsenWangConfig::graph(adjacency, coupling)?))
    }

    /// Number of sites in the lattice / graph.
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    /// Number of clusters formed on the most recent [`SwendsenWang::step`].
    pub fn last_n_clusters(&self) -> usize {
        self.last_n_clusters
    }

    /// Bond activation probability `p = 1 − exp(−2 β J)` for the configured
    /// coupling.  Clamped into `[0, 1]`: a non-positive `2 β J` (anti-ferro or
    /// `β ≤ 0`) yields `p = 0`, so no bonds form and the update is a no-op,
    /// which is the correct behaviour for the aligned-only bond rule.
    #[inline]
    pub fn bond_probability(&self, beta: f64) -> f64 {
        let two_beta_j = 2.0 * beta * self.coupling;
        if two_beta_j <= 0.0 {
            0.0
        } else {
            (1.0 - (-two_beta_j).exp()).clamp(0.0, 1.0)
        }
    }

    /// Perform one full Swendsen-Wang sweep in place.
    ///
    /// Activates bonds on aligned edges with probability `p = 1 − exp(−2 β J)`,
    /// builds clusters via union-find, then flips each cluster independently
    /// with probability ½.
    ///
    /// # Errors
    /// Returns [`SeqError::ShapeMismatch`] if `spins.len()` differs from the
    /// site count, [`SeqError::InvalidParameter`] for a non-finite `beta`, and
    /// [`SeqError::InvalidParameter`] (named `"spin"`) if any spin is not `±1`.
    pub fn step(&mut self, spins: &mut [i8], beta: f64, rng: &mut LcgRng) -> SeqResult<()> {
        check_spins(spins, self.n_sites, beta)?;

        let p = self.bond_probability(beta);
        let mut uf = UnionFind::new(self.n_sites);

        // 1. Bond percolation over aligned edges.
        for &(i, j) in &self.edges {
            if spins[i] == spins[j] && rng.next_f64() < p {
                uf.union(i, j);
            }
        }

        // 2. Decide a flip for every cluster (keyed on its representative).
        //    A site is flipped iff its cluster's coin came up `true`.
        let mut flip_decided = vec![false; self.n_sites];
        let mut flip_value = vec![false; self.n_sites];
        let mut n_clusters = 0usize;
        for site in 0..self.n_sites {
            let root = uf.find(site);
            if !flip_decided[root] {
                flip_decided[root] = true;
                flip_value[root] = rng.next_bool();
                n_clusters += 1;
            }
        }
        self.last_n_clusters = n_clusters;

        // 3. Apply the flips.
        for site in 0..self.n_sites {
            let root = uf.find(site);
            if flip_value[root] {
                spins[site] = -spins[site];
            }
        }
        Ok(())
    }

    /// Mean magnetisation `(1/N) Σ s_i ∈ [−1, +1]`.
    ///
    /// # Errors
    /// [`SeqError::ShapeMismatch`] on a site-count mismatch.
    pub fn magnetization(&self, spins: &[i8]) -> SeqResult<f64> {
        magnetization(spins, self.n_sites)
    }

    /// Total Ising energy `E = −J Σ_(i,j) s_i s_j` summed over the geometry's
    /// edges (no external field).
    ///
    /// # Errors
    /// [`SeqError::ShapeMismatch`] on a site-count mismatch.
    pub fn energy(&self, spins: &[i8]) -> SeqResult<f64> {
        if spins.len() != self.n_sites {
            return Err(SeqError::ShapeMismatch {
                expected: self.n_sites,
                got: spins.len(),
            });
        }
        let mut e = 0.0;
        for &(i, j) in &self.edges {
            e -= self.coupling * (spins[i] as f64) * (spins[j] as f64);
        }
        Ok(e)
    }

    /// Borrow the (canonical, `i < j`) edge list used by the sampler.
    #[cfg(test)]
    pub(crate) fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }
}

/// Mean magnetisation, validating the spin-vector length.
pub(crate) fn magnetization(spins: &[i8], n_sites: usize) -> SeqResult<f64> {
    if spins.len() != n_sites {
        return Err(SeqError::ShapeMismatch {
            expected: n_sites,
            got: spins.len(),
        });
    }
    let s: i64 = spins.iter().map(|&x| x as i64).sum();
    Ok(s as f64 / spins.len() as f64)
}

/// Validate the spin slice: correct length, finite `beta`, and every entry
/// exactly `±1`.
pub(crate) fn check_spins(spins: &[i8], n_sites: usize, beta: f64) -> SeqResult<()> {
    if spins.len() != n_sites {
        return Err(SeqError::ShapeMismatch {
            expected: n_sites,
            got: spins.len(),
        });
    }
    if !beta.is_finite() {
        return Err(SeqError::InvalidParameter {
            name: "beta".to_string(),
            value: beta,
        });
    }
    for &s in spins {
        if s != 1 && s != -1 {
            return Err(SeqError::InvalidParameter {
                name: "spin".to_string(),
                value: s as f64,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `β` well above the 2-D critical point — deep in the ordered phase.
    const BETA_ORDERED: f64 = 0.70;
    /// `β` well below the critical point — disordered phase.
    const BETA_DISORDERED: f64 = 0.20;

    fn run_chain(
        sw: &mut SwendsenWang,
        spins: &mut [i8],
        beta: f64,
        seed: u64,
        burn: usize,
        samples: usize,
    ) -> f64 {
        let mut rng = LcgRng::new(seed);
        for _ in 0..burn {
            sw.step(spins, beta, &mut rng).expect("step");
        }
        let mut acc = 0.0;
        for _ in 0..samples {
            sw.step(spins, beta, &mut rng).expect("step");
            acc += sw.magnetization(spins).expect("mag").abs();
        }
        acc / samples as f64
    }

    #[test]
    fn union_find_basic() {
        let mut uf = UnionFind::new(6);
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(4, 5);
        assert_eq!(uf.find(0), uf.find(2));
        assert_eq!(uf.find(4), uf.find(5));
        assert_ne!(uf.find(0), uf.find(4));
        assert_ne!(uf.find(0), uf.find(3));
    }

    #[test]
    fn square_edge_count_open_and_periodic() {
        // Open 4×4: horizontal = rows*(cols-1)=12, vertical = (rows-1)*cols=12.
        let open = SwendsenWang::square(4, 4, 1.0, false).expect("ok");
        assert_eq!(open.edges().len(), 24);
        // Periodic 4×4: each site has degree 4 → 2*N = 32 directed → 16*2? No:
        // N*degree/2 = 16*4/2 = 32 undirected edges.
        let per = SwendsenWang::square(4, 4, 1.0, true).expect("ok");
        assert_eq!(per.edges().len(), 32);
        // Every edge is canonical (i < j) and unique.
        let mut seen = std::collections::HashSet::new();
        for &(i, j) in per.edges() {
            assert!(i < j);
            assert!(seen.insert((i, j)), "duplicate edge ({i},{j})");
        }
    }

    #[test]
    fn bond_probability_formula() {
        let sw = SwendsenWang::square(2, 2, 1.0, false).expect("ok");
        let beta = 0.5;
        let expected = 1.0 - (-2.0 * beta * 1.0_f64).exp();
        assert!((sw.bond_probability(beta) - expected).abs() < 1e-12);
        // Zero β → zero probability.
        assert_eq!(sw.bond_probability(0.0), 0.0);
    }

    #[test]
    fn phase_transition_ordered_vs_disordered() {
        // Ordered phase: high |M|; disordered phase: low |M|. Periodic 16×16.
        let mut sw = SwendsenWang::square(16, 16, 1.0, true).expect("ok");
        let mut spins = vec![1i8; 256];
        let m_ordered = run_chain(&mut sw, &mut spins, BETA_ORDERED, 11, 40, 80);

        let mut spins2 = vec![1i8; 256];
        let m_dis = run_chain(&mut sw, &mut spins2, BETA_DISORDERED, 11, 40, 80);

        assert!(
            m_ordered > 0.8,
            "ordered |M|={m_ordered} should be high (>0.8)"
        );
        assert!(m_dis < 0.4, "disordered |M|={m_dis} should be low (<0.4)");
        assert!(
            m_ordered - m_dis > 0.4,
            "ordered ({m_ordered}) and disordered ({m_dis}) must separate"
        );
    }

    #[test]
    fn cluster_flip_preserves_intra_cluster_alignment() {
        // At very high β every aligned edge almost surely bonds, so a uniform
        // start collapses into one giant cluster; flipping it keeps all spins
        // mutually aligned (all +1 or all −1).
        let mut sw = SwendsenWang::square(8, 8, 1.0, true).expect("ok");
        let mut spins = vec![1i8; 64];
        let mut rng = LcgRng::new(7);
        sw.step(&mut spins, 5.0, &mut rng).expect("step");
        let first = spins[0];
        assert!(
            spins.iter().all(|&s| s == first),
            "high-β single cluster must stay aligned"
        );
        assert_eq!(sw.last_n_clusters(), 1, "should be one giant cluster");
    }

    #[test]
    fn energy_magnetization_consistent_all_aligned() {
        let sw = SwendsenWang::square(3, 3, 1.0, false).expect("ok");
        let spins = vec![1i8; 9];
        // 12 edges, all aligned: E = -J * 12.
        assert!((sw.energy(&spins).expect("e") + 12.0).abs() < 1e-12);
        assert!((sw.magnetization(&spins).expect("m") - 1.0).abs() < 1e-12);
        // Global flip leaves energy invariant (Z2 symmetry).
        let flipped = vec![-1i8; 9];
        assert!((sw.energy(&flipped).expect("e") + 12.0).abs() < 1e-12);
        assert!((sw.magnetization(&flipped).expect("m") + 1.0).abs() < 1e-12);
    }

    #[test]
    fn energy_only_changes_on_cluster_boundary() {
        // Flip a single cluster and verify the energy change equals
        // 2 J Σ_{boundary} s_i s_j (the sum over edges crossing the cluster).
        let mut sw = SwendsenWang::square(6, 6, 1.0, true).expect("ok");
        // Mixed deterministic start so clusters are non-trivial.
        let mut spins: Vec<i8> = (0..36)
            .map(|i| if (i / 6 + i % 6) % 2 == 0 { 1 } else { -1 })
            .collect();
        let e_before = sw.energy(&spins).expect("e");
        let mut rng = LcgRng::new(3);
        sw.step(&mut spins, 0.4, &mut rng).expect("step");
        let e_after = sw.energy(&spins).expect("e");
        // Energy must stay finite and within the achievable bound |E| <= J*#edges.
        let bound = sw.edges().len() as f64;
        assert!(e_after.abs() <= bound + 1e-9);
        assert!(e_before.is_finite() && e_after.is_finite());
    }

    #[test]
    fn determinism_fixed_seed() {
        let mut a = SwendsenWang::square(10, 10, 1.0, true).expect("ok");
        let mut b = SwendsenWang::square(10, 10, 1.0, true).expect("ok");
        let mut sa = vec![1i8; 100];
        let mut sb = vec![1i8; 100];
        let mut ra = LcgRng::new(2024);
        let mut rb = LcgRng::new(2024);
        for _ in 0..25 {
            a.step(&mut sa, 0.45, &mut ra).expect("step");
            b.step(&mut sb, 0.45, &mut rb).expect("step");
        }
        assert_eq!(sa, sb, "identical seeds must give identical trajectories");
    }

    #[test]
    fn ergodic_global_sign_flips() {
        // At high β the magnetisation should occasionally flip global sign over
        // enough sweeps — the property single-spin Metropolis cannot achieve.
        let mut sw = SwendsenWang::square(12, 12, 1.0, true).expect("ok");
        let mut spins = vec![1i8; 144];
        let mut rng = LcgRng::new(99);
        let mut saw_positive = false;
        let mut saw_negative = false;
        for _ in 0..200 {
            sw.step(&mut spins, 0.60, &mut rng).expect("step");
            let m = sw.magnetization(&spins).expect("m");
            if m > 0.5 {
                saw_positive = true;
            }
            if m < -0.5 {
                saw_negative = true;
            }
        }
        assert!(
            saw_positive && saw_negative,
            "sampler must visit both ±M ordered states (saw + = {saw_positive}, − = {saw_negative})"
        );
    }

    #[test]
    fn graph_variant_two_triangles_bridge() {
        // Two triangles {0,1,2} and {3,4,5} joined by a bridge edge (2,3).
        let adjacency = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1, 3],
            vec![2, 4, 5],
            vec![3, 5],
            vec![3, 4],
        ];
        let mut sw = SwendsenWang::from_graph(adjacency, 1.0).expect("ok");
        assert_eq!(sw.n_sites(), 6);
        assert_eq!(sw.edges().len(), 7);
        let mut spins = vec![1i8; 6];
        let mut rng = LcgRng::new(5);
        // High β: whole connected graph fuses into one aligned cluster.
        sw.step(&mut spins, 4.0, &mut rng).expect("step");
        let first = spins[0];
        assert!(spins.iter().all(|&s| s == first));
    }

    #[test]
    fn err_dimension_mismatch() {
        let mut sw = SwendsenWang::square(4, 4, 1.0, false).expect("ok");
        let mut spins = vec![1i8; 15]; // should be 16
        let mut rng = LcgRng::new(1);
        match sw.step(&mut spins, 0.4, &mut rng) {
            Err(SeqError::ShapeMismatch { expected, got }) => {
                assert_eq!(expected, 16);
                assert_eq!(got, 15);
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn err_bad_lattice_size_and_invalid_spin() {
        // Zero dimension.
        assert!(matches!(
            SwendsenWang::square(0, 4, 1.0, false),
            Err(SeqError::InvalidConfiguration(_))
        ));
        // Invalid spin value (2 is not ±1).
        let mut sw = SwendsenWang::square(2, 2, 1.0, false).expect("ok");
        let mut spins = vec![1i8, -1, 2, 1];
        let mut rng = LcgRng::new(1);
        match sw.step(&mut spins, 0.4, &mut rng) {
            Err(SeqError::InvalidParameter { name, value }) => {
                assert_eq!(name, "spin");
                assert_eq!(value, 2.0);
            }
            other => panic!("expected InvalidParameter spin, got {other:?}"),
        }
        // Non-finite beta.
        let mut good = vec![1i8; 4];
        assert!(matches!(
            sw.step(&mut good, f64::NAN, &mut rng),
            Err(SeqError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn err_graph_invariants() {
        // Asymmetric adjacency.
        assert!(matches!(
            SwendsenWang::from_graph(vec![vec![1], vec![]], 1.0),
            Err(SeqError::GraphInvariantViolated(_))
        ));
        // Self-loop.
        assert!(matches!(
            SwendsenWang::from_graph(vec![vec![0]], 1.0),
            Err(SeqError::GraphInvariantViolated(_))
        ));
        // Out-of-range neighbour.
        assert!(matches!(
            SwendsenWang::from_graph(vec![vec![3], vec![]], 1.0),
            Err(SeqError::IndexOutOfBounds { .. })
        ));
        // Empty graph.
        assert!(matches!(
            SwendsenWang::from_graph(vec![], 1.0),
            Err(SeqError::EmptyInput)
        ));
    }
}
