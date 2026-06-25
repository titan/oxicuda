//! Wolff (1989) single-cluster Monte-Carlo update for the ferromagnetic Ising
//! model.
//!
//! Where Swendsen-Wang (see [`super::swendsen_wang`]) partitions the *entire*
//! lattice into clusters every sweep, Wolff grows and flips a **single** cluster
//! seeded at a random site.  This is often more efficient per unit work near
//! criticality because the algorithm preferentially builds large clusters (the
//! probability of seeding inside a big cluster is proportional to its size),
//! and a single flood-fill touches only the spins it actually flips.
//!
//! One update proceeds as:
//!
//! 1. **Seed.**  Pick a site `k` uniformly at random; remember its spin value
//!    `s_seed`.
//! 2. **Grow.**  Flood-fill (stack / BFS) from `k`.  For each aligned neighbour
//!    `j` (`s_j == s_seed`) not yet in the cluster, add it with probability
//!    `p = 1 − exp(−2 β J)`.
//! 3. **Flip.**  Flip the whole grown cluster unconditionally.
//!
//! Because the seed cluster is *always* flipped (not with probability ½), the
//! single-cluster move still satisfies detailed balance — the asymmetry in the
//! proposal is exactly cancelled by the Fortuin-Kasteleyn bond weights.  The
//! chain is ergodic: over enough updates it visits both `+M` and `−M` ordered
//! states, decorrelating the global magnetisation far faster than single-spin
//! Metropolis near `T_c`.
//!
//! Spins are `i8` values in `{−1, +1}`.  Square lattices (open or periodic) and
//! general graphs are both supported; the geometry and validation helpers are
//! shared with [`super::swendsen_wang`].

use super::swendsen_wang::{Geometry, check_spins, magnetization, validate_graph, validate_square};
use crate::error::{SeqError, SeqResult};
use crate::handle::LcgRng;

/// Configuration for a Wolff sampler.
#[derive(Debug, Clone)]
pub struct WolffConfig {
    /// Ferromagnetic coupling constant `J`.
    pub coupling: f64,
    /// Geometry of the spin system.
    pub(crate) geometry: Geometry,
}

impl WolffConfig {
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

/// Wolff single-cluster sampler.
///
/// Precomputes a per-site neighbour table from the geometry so the flood-fill
/// can enumerate neighbours in `O(degree)` without re-deriving lattice indices.
#[derive(Debug, Clone)]
pub struct Wolff {
    coupling: f64,
    /// `neighbours[i]` = adjacency of site `i` (square lattice expanded once).
    neighbours: Vec<Vec<usize>>,
    n_sites: usize,
    /// Size of the cluster flipped on the most recent step (diagnostic).
    last_cluster_size: usize,
}

/// Expand a [`Geometry`] into an explicit per-site neighbour table.
fn build_neighbours(geometry: &Geometry) -> Vec<Vec<usize>> {
    match geometry {
        Geometry::Square {
            rows,
            cols,
            periodic,
        } => {
            let rows = *rows;
            let cols = *cols;
            let periodic = *periodic;
            let mut nbrs = vec![Vec::new(); rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    let i = r * cols + c;
                    // Left / right.
                    if c + 1 < cols {
                        nbrs[i].push(r * cols + (c + 1));
                        nbrs[r * cols + (c + 1)].push(i);
                    } else if periodic && cols > 1 {
                        let j = r * cols; // wrap to column 0
                        nbrs[i].push(j);
                        nbrs[j].push(i);
                    }
                    // Up / down.
                    if r + 1 < rows {
                        nbrs[i].push((r + 1) * cols + c);
                        nbrs[(r + 1) * cols + c].push(i);
                    } else if periodic && rows > 1 {
                        let j = c; // wrap to row 0
                        nbrs[i].push(j);
                        nbrs[j].push(i);
                    }
                }
            }
            nbrs
        }
        Geometry::Graph { adjacency } => adjacency.clone(),
    }
}

impl Wolff {
    /// Build a sampler from a validated configuration.
    pub fn new(cfg: WolffConfig) -> Self {
        let neighbours = build_neighbours(&cfg.geometry);
        let n_sites = cfg.geometry.n_sites();
        Self {
            coupling: cfg.coupling,
            neighbours,
            n_sites,
            last_cluster_size: 0,
        }
    }

    /// Convenience constructor for a square lattice.
    pub fn square(rows: usize, cols: usize, coupling: f64, periodic: bool) -> SeqResult<Self> {
        Ok(Self::new(WolffConfig::square(
            rows, cols, coupling, periodic,
        )?))
    }

    /// Convenience constructor for a general graph.
    pub fn from_graph(adjacency: Vec<Vec<usize>>, coupling: f64) -> SeqResult<Self> {
        Ok(Self::new(WolffConfig::graph(adjacency, coupling)?))
    }

    /// Number of sites in the lattice / graph.
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    /// Size of the cluster flipped on the most recent [`Wolff::step`].
    pub fn last_cluster_size(&self) -> usize {
        self.last_cluster_size
    }

    /// Bond / addition probability `p = 1 − exp(−2 β J)`.  See
    /// [`super::swendsen_wang::SwendsenWang::bond_probability`] for the
    /// clamping rationale.
    #[inline]
    pub fn bond_probability(&self, beta: f64) -> f64 {
        let two_beta_j = 2.0 * beta * self.coupling;
        if two_beta_j <= 0.0 {
            0.0
        } else {
            (1.0 - (-two_beta_j).exp()).clamp(0.0, 1.0)
        }
    }

    /// Perform one Wolff single-cluster update in place.
    ///
    /// Seeds at a uniformly random site, grows the aligned cluster by stack
    /// flood-fill (each aligned neighbour joins with probability
    /// `p = 1 − exp(−2 β J)`), then flips the whole cluster.
    ///
    /// # Errors
    /// [`SeqError::ShapeMismatch`] on a site-count mismatch,
    /// [`SeqError::InvalidParameter`] for non-finite `beta` or non-`±1` spins.
    pub fn step(&mut self, spins: &mut [i8], beta: f64, rng: &mut LcgRng) -> SeqResult<()> {
        check_spins(spins, self.n_sites, beta)?;
        if self.n_sites == 0 {
            return Err(SeqError::EmptyInput);
        }

        let p = self.bond_probability(beta);
        let seed = rng.next_usize(self.n_sites);
        let s_seed = spins[seed];

        // Flood-fill cluster growth using an explicit stack.  `in_cluster`
        // marks membership; a site is added to the stack only once.
        let mut in_cluster = vec![false; self.n_sites];
        let mut stack: Vec<usize> = Vec::new();
        in_cluster[seed] = true;
        stack.push(seed);
        let mut cluster: Vec<usize> = vec![seed];

        while let Some(site) = stack.pop() {
            for &nbr in &self.neighbours[site] {
                if !in_cluster[nbr] && spins[nbr] == s_seed && rng.next_f64() < p {
                    in_cluster[nbr] = true;
                    stack.push(nbr);
                    cluster.push(nbr);
                }
            }
        }

        // Flip the whole cluster unconditionally.
        for &site in &cluster {
            spins[site] = -spins[site];
        }
        self.last_cluster_size = cluster.len();
        Ok(())
    }

    /// Mean magnetisation `(1/N) Σ s_i ∈ [−1, +1]`.
    ///
    /// # Errors
    /// [`SeqError::ShapeMismatch`] on a site-count mismatch.
    pub fn magnetization(&self, spins: &[i8]) -> SeqResult<f64> {
        magnetization(spins, self.n_sites)
    }

    /// Total Ising energy `E = −J Σ_(i,j) s_i s_j` over the geometry's edges
    /// (no external field).  Each undirected edge is counted once.
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
        // Sum over canonical edges (i < j) so every bond is counted exactly once.
        for (i, nbrs) in self.neighbours.iter().enumerate() {
            for &j in nbrs {
                if i < j {
                    e -= self.coupling * (spins[i] as f64) * (spins[j] as f64);
                }
            }
        }
        Ok(e)
    }

    /// Borrow the per-site neighbour table.
    #[cfg(test)]
    pub(crate) fn neighbours(&self) -> &[Vec<usize>] {
        &self.neighbours
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BETA_ORDERED: f64 = 0.70;
    const BETA_DISORDERED: f64 = 0.20;

    fn run_chain(
        w: &mut Wolff,
        spins: &mut [i8],
        beta: f64,
        seed: u64,
        burn: usize,
        samples: usize,
    ) -> f64 {
        let mut rng = LcgRng::new(seed);
        for _ in 0..burn {
            w.step(spins, beta, &mut rng).expect("step");
        }
        let mut acc = 0.0;
        for _ in 0..samples {
            w.step(spins, beta, &mut rng).expect("step");
            acc += w.magnetization(spins).expect("mag").abs();
        }
        acc / samples as f64
    }

    #[test]
    fn neighbour_table_square_periodic_degree_four() {
        let w = Wolff::square(5, 5, 1.0, true).expect("ok");
        // Every site on a periodic lattice has exactly 4 neighbours.
        for nbrs in w.neighbours() {
            assert_eq!(nbrs.len(), 4);
        }
        assert_eq!(w.n_sites(), 25);
    }

    #[test]
    fn neighbour_table_square_open_corner_and_centre() {
        let w = Wolff::square(3, 3, 1.0, false).expect("ok");
        // Corner site 0 has 2 neighbours; centre site 4 has 4.
        assert_eq!(w.neighbours()[0].len(), 2);
        assert_eq!(w.neighbours()[4].len(), 4);
    }

    #[test]
    fn bond_probability_formula() {
        let w = Wolff::square(2, 2, 1.0, false).expect("ok");
        let beta = 0.33_f64;
        let expected = 1.0 - (-2.0 * beta).exp();
        assert!((w.bond_probability(beta) - expected).abs() < 1e-12);
        assert_eq!(w.bond_probability(0.0), 0.0);
    }

    #[test]
    fn cluster_contains_only_aligned_spins() {
        // Build a checkerboard: no two neighbours are aligned, so any seeded
        // cluster is exactly the singleton seed — and flipping a single spin
        // changes |M| by 2/N.
        let w_dims = (6usize, 6usize);
        let mut w = Wolff::square(w_dims.0, w_dims.1, 1.0, true).expect("ok");
        let mut spins: Vec<i8> = (0..36)
            .map(|i| if (i / 6 + i % 6) % 2 == 0 { 1 } else { -1 })
            .collect();
        let mut rng = LcgRng::new(1);
        w.step(&mut spins, 5.0, &mut rng).expect("step");
        // On a checkerboard the neighbours are anti-aligned, so the seed cluster
        // is a single site regardless of β.
        assert_eq!(
            w.last_cluster_size(),
            1,
            "checkerboard cluster is singleton"
        );
    }

    #[test]
    fn high_beta_uniform_flips_whole_lattice() {
        // From a uniform state at high β the cluster spans the connected
        // lattice and the global flip yields the opposite uniform state.
        let mut w = Wolff::square(8, 8, 1.0, true).expect("ok");
        let mut spins = vec![1i8; 64];
        let mut rng = LcgRng::new(42);
        w.step(&mut spins, 5.0, &mut rng).expect("step");
        assert_eq!(w.last_cluster_size(), 64, "should engulf whole lattice");
        assert!(spins.iter().all(|&s| s == -1), "uniform −1 after flip");
    }

    #[test]
    fn phase_transition_ordered_vs_disordered() {
        let mut w = Wolff::square(16, 16, 1.0, true).expect("ok");
        let mut spins = vec![1i8; 256];
        let m_ordered = run_chain(&mut w, &mut spins, BETA_ORDERED, 17, 60, 120);

        let mut spins2 = vec![1i8; 256];
        let m_dis = run_chain(&mut w, &mut spins2, BETA_DISORDERED, 17, 60, 120);

        assert!(
            m_ordered > 0.8,
            "ordered |M|={m_ordered} should be high (>0.8)"
        );
        assert!(m_dis < 0.4, "disordered |M|={m_dis} should be low (<0.4)");
        assert!(
            m_ordered - m_dis > 0.4,
            "ordered ({m_ordered}) vs disordered ({m_dis}) must separate"
        );
    }

    #[test]
    fn energy_change_matches_boundary_sum() {
        // A whole-cluster flip changes energy only on the cluster boundary.
        // ΔE = 2 J Σ_{(i in C, j not in C)} s_i s_j (pre-flip spins).
        let mut w = Wolff::square(7, 7, 1.0, true).expect("ok");
        let mut spins: Vec<i8> = (0..49)
            .map(|i| if (i * 7 + 3) % 5 < 2 { 1 } else { -1 })
            .collect();

        // Reproduce the exact cluster the next step will build by replaying the
        // RNG, so we can compute the analytic boundary energy change.
        let beta = 0.45;
        let p = w.bond_probability(beta);
        let mut rng = LcgRng::new(123);
        let seed = rng.next_usize(w.n_sites());
        let s_seed = spins[seed];
        let mut in_cluster = vec![false; w.n_sites()];
        let mut stack = vec![seed];
        in_cluster[seed] = true;
        while let Some(site) = stack.pop() {
            for &nbr in &w.neighbours()[site] {
                if !in_cluster[nbr] && spins[nbr] == s_seed && rng.next_f64() < p {
                    in_cluster[nbr] = true;
                    stack.push(nbr);
                }
            }
        }
        // Analytic boundary energy change (pre-flip spins).
        let mut delta = 0.0;
        for (i, nbrs) in w.neighbours().iter().enumerate() {
            for &j in nbrs {
                if i < j && in_cluster[i] != in_cluster[j] {
                    delta += 2.0 * 1.0 * (spins[i] as f64) * (spins[j] as f64);
                }
            }
        }

        let e_before = w.energy(&spins).expect("e");
        // Now actually run the step with a *fresh* RNG at the same seed so the
        // same cluster is grown and flipped.
        let mut rng2 = LcgRng::new(123);
        w.step(&mut spins, beta, &mut rng2).expect("step");
        let e_after = w.energy(&spins).expect("e");
        assert!(
            (e_after - e_before - delta).abs() < 1e-9,
            "ΔE={} should equal boundary sum {}",
            e_after - e_before,
            delta
        );
    }

    #[test]
    fn determinism_fixed_seed() {
        let mut a = Wolff::square(10, 10, 1.0, true).expect("ok");
        let mut b = Wolff::square(10, 10, 1.0, true).expect("ok");
        let mut sa = vec![1i8; 100];
        let mut sb = vec![1i8; 100];
        let mut ra = LcgRng::new(7777);
        let mut rb = LcgRng::new(7777);
        for _ in 0..50 {
            a.step(&mut sa, 0.45, &mut ra).expect("step");
            b.step(&mut sb, 0.45, &mut rb).expect("step");
        }
        assert_eq!(sa, sb);
        assert_eq!(a.last_cluster_size(), b.last_cluster_size());
    }

    #[test]
    fn ergodic_global_sign_flips() {
        // Single-cluster updates flip the global magnetisation sign readily at
        // high β — the decorrelation property single-spin Metropolis lacks.
        let mut w = Wolff::square(12, 12, 1.0, true).expect("ok");
        let mut spins = vec![1i8; 144];
        let mut rng = LcgRng::new(2025);
        let mut saw_positive = false;
        let mut saw_negative = false;
        for _ in 0..150 {
            w.step(&mut spins, 0.60, &mut rng).expect("step");
            let m = w.magnetization(&spins).expect("m");
            if m > 0.5 {
                saw_positive = true;
            }
            if m < -0.5 {
                saw_negative = true;
            }
        }
        assert!(
            saw_positive && saw_negative,
            "must explore both ±M states (+ = {saw_positive}, − = {saw_negative})"
        );
    }

    #[test]
    fn graph_variant_path_three() {
        // Path graph 0—1—2; at high β a seed at an end can engulf the path.
        let adjacency = vec![vec![1], vec![0, 2], vec![1]];
        let mut w = Wolff::from_graph(adjacency, 1.0).expect("ok");
        assert_eq!(w.n_sites(), 3);
        let mut spins = vec![1i8; 3];
        let mut rng = LcgRng::new(9);
        // Repeatedly stepping should at some point flip a multi-site cluster.
        let mut saw_multi = false;
        for _ in 0..30 {
            w.step(&mut spins, 4.0, &mut rng).expect("step");
            if w.last_cluster_size() > 1 {
                saw_multi = true;
            }
        }
        assert!(saw_multi, "high-β path should grow a multi-site cluster");
    }

    #[test]
    fn err_dimension_mismatch() {
        let mut w = Wolff::square(4, 4, 1.0, false).expect("ok");
        let mut spins = vec![1i8; 17]; // should be 16
        let mut rng = LcgRng::new(1);
        match w.step(&mut spins, 0.4, &mut rng) {
            Err(SeqError::ShapeMismatch { expected, got }) => {
                assert_eq!(expected, 16);
                assert_eq!(got, 17);
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn err_bad_lattice_and_invalid_spin() {
        assert!(matches!(
            Wolff::square(3, 0, 1.0, false),
            Err(SeqError::InvalidConfiguration(_))
        ));
        let mut w = Wolff::square(2, 2, 1.0, false).expect("ok");
        let mut spins = vec![1i8, -1, 0, 1]; // 0 is not ±1
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            w.step(&mut spins, 0.4, &mut rng),
            Err(SeqError::InvalidParameter { .. })
        ));
        // Infinite beta.
        let mut good = vec![1i8; 4];
        assert!(matches!(
            w.step(&mut good, f64::INFINITY, &mut rng),
            Err(SeqError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn err_graph_invariants() {
        assert!(matches!(
            Wolff::from_graph(vec![vec![1], vec![]], 1.0),
            Err(SeqError::GraphInvariantViolated(_))
        ));
        assert!(matches!(
            Wolff::from_graph(vec![], 1.0),
            Err(SeqError::EmptyInput)
        ));
        assert!(matches!(
            Wolff::from_graph(vec![vec![5], vec![]], 1.0),
            Err(SeqError::IndexOutOfBounds { .. })
        ));
    }
}
