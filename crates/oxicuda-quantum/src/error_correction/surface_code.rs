//! Rotated surface code with a minimum-weight perfect-matching (MWPM) decoder.
//!
//! This module implements a distance-`d` **rotated** surface code (Fowler et al.,
//! *"Surface codes: Towards practical large-scale quantum computation"*, Phys.
//! Rev. A 86, 032324, 2012) together with a syndrome decoder, in the **classical
//! Pauli-frame** picture used by virtually every surface-code threshold study.
//!
//! ## Scope and conventions
//!
//! Rather than carrying a 2^{d²}-amplitude state vector through the stabilizer
//! measurement schedule (which is unnecessary for tracking how Pauli errors map
//! to logical errors), we track an error as an independent pair of `X`/`Z` bit
//! masks over the `d²` data qubits — a [`PauliError`]. This is exact for Pauli
//! noise: a stabilizer's syndrome bit is just the parity of the overlap between
//! the stabilizer support and the offending error component, and a residual
//! error commutes with every stabilizer iff it lies in the stabilizer group
//! (no logical error). All correctness statements below are made in this frame.
//!
//! * **Layout.** `d²` data qubits sit on a `d × d` grid (index `r·d + c`).
//!   The `d²−1` stabilizers form the rotated checkerboard: `(d−1)²` weight-4
//!   bulk plaquettes plus `2(d−1)` weight-2 boundary plaquettes. `Z`-type
//!   plaquettes detect `X` errors; `X`-type plaquettes detect `Z` errors. There
//!   are exactly `(d²−1)/2` of each.
//! * **Decoder.** For each error species we build the matching graph whose
//!   nodes are the stabilizers of the detecting type plus one virtual boundary
//!   node, and whose edges are data qubits. Decoding is an *exact* minimum-weight
//!   perfect matching solved by subset (bitmask) dynamic programming over the
//!   **defect set** — not the full stabilizer set — so the cost is `O(2^{k}·k)`
//!   in the number of fired stabilizers `k`, which is tiny for the low-weight
//!   errors a distance-`d` code is meant to handle. (A Blossom-V matching would
//!   be the drop-in replacement for high-weight syndromes at large `d`; the DP
//!   here is the same minimum-weight matching, just enumerated exactly.)
//!
//! For a distance-`d` code this corrects every error of weight `≤ ⌊(d−1)/2⌋`:
//! the residual `error ∘ correction` is a cycle of weight below `d`, hence a
//! product of stabilizers, hence logically trivial.

use std::collections::VecDeque;

use crate::error::{QuantumError, QuantumResult};

/// Stabilizer species of a rotated-surface-code plaquette.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabKind {
    /// `X`-type plaquette (product of `X` on its support); detects `Z` errors.
    X,
    /// `Z`-type plaquette (product of `Z` on its support); detects `X` errors.
    Z,
}

/// A single stabilizer generator: its species and the data qubits it acts on.
#[derive(Debug, Clone)]
pub struct Stabilizer {
    /// `X`- or `Z`-type.
    pub kind: StabKind,
    /// Indices (`r·d + c`) of the data qubits in the plaquette support.
    pub support: Vec<usize>,
}

/// A Pauli error over the data qubits: independent `X` (bit-flip) and `Z`
/// (phase-flip) components. `y = x · z` on a qubit is represented by both bits
/// being set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PauliError {
    /// `x[q] = true` ⇔ an `X` acts on data qubit `q`.
    pub x: Vec<bool>,
    /// `z[q] = true` ⇔ a `Z` acts on data qubit `q`.
    pub z: Vec<bool>,
}

impl PauliError {
    /// The trivial (identity) error on `num_qubits` data qubits.
    #[must_use]
    pub fn identity(num_qubits: usize) -> Self {
        Self {
            x: vec![false; num_qubits],
            z: vec![false; num_qubits],
        }
    }

    /// A single `X` error on data qubit `q`.
    ///
    /// # Errors
    /// [`QuantumError::QubitIndexOutOfRange`] if `q >= num_qubits`.
    pub fn single_x(num_qubits: usize, q: usize) -> QuantumResult<Self> {
        if q >= num_qubits {
            return Err(QuantumError::QubitIndexOutOfRange {
                index: q,
                n_qubits: num_qubits,
            });
        }
        let mut e = Self::identity(num_qubits);
        e.x[q] = true;
        Ok(e)
    }

    /// A single `Z` error on data qubit `q`.
    ///
    /// # Errors
    /// [`QuantumError::QubitIndexOutOfRange`] if `q >= num_qubits`.
    pub fn single_z(num_qubits: usize, q: usize) -> QuantumResult<Self> {
        if q >= num_qubits {
            return Err(QuantumError::QubitIndexOutOfRange {
                index: q,
                n_qubits: num_qubits,
            });
        }
        let mut e = Self::identity(num_qubits);
        e.z[q] = true;
        Ok(e)
    }

    /// Number of data qubits carrying a non-identity Pauli.
    #[must_use]
    pub fn weight(&self) -> usize {
        self.x
            .iter()
            .zip(self.z.iter())
            .filter(|(a, b)| **a || **b)
            .count()
    }

    /// Compose two errors (qubit-wise XOR of `X` and `Z` parts).
    ///
    /// # Errors
    /// [`QuantumError::DimensionMismatch`] if the operands span different qubit
    /// counts.
    pub fn compose(&self, other: &Self) -> QuantumResult<Self> {
        if self.x.len() != other.x.len() {
            return Err(QuantumError::DimensionMismatch {
                expected: self.x.len(),
                got: other.x.len(),
            });
        }
        let x = self
            .x
            .iter()
            .zip(other.x.iter())
            .map(|(a, b)| a ^ b)
            .collect();
        let z = self
            .z
            .iter()
            .zip(other.z.iter())
            .map(|(a, b)| a ^ b)
            .collect();
        Ok(Self { x, z })
    }
}

/// The measured syndrome: one bit per stabilizer (`true` = defect / fired).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Syndrome {
    /// Indexed by stabilizer index (matching [`SurfaceCode::stabilizers`]).
    pub bits: Vec<bool>,
}

impl Syndrome {
    /// `true` when no stabilizer fired.
    #[must_use]
    pub fn is_trivial(&self) -> bool {
        self.bits.iter().all(|b| !b)
    }

    /// Number of fired stabilizers (defects).
    #[must_use]
    pub fn weight(&self) -> usize {
        self.bits.iter().filter(|b| **b).count()
    }
}

/// Configuration for a [`SurfaceCode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceCodeConfig {
    /// Code distance `d` (must be a positive odd integer).
    pub distance: usize,
}

/// Matching graph for one error species: stabilizer nodes `0..num`, a single
/// virtual boundary node at index `num`, and qubit-labelled edges.
#[derive(Debug, Clone)]
struct MatchingGraph {
    num_nodes: usize,
    boundary: usize,
    /// `adj[node] = [(neighbour, data_qubit_label), …]`.
    adj: Vec<Vec<(usize, usize)>>,
}

/// A distance-`d` rotated surface code with an MWPM decoder.
#[derive(Debug, Clone)]
pub struct SurfaceCode {
    distance: usize,
    num_data: usize,
    stabilizers: Vec<Stabilizer>,
    /// Stabilizer indices (into `stabilizers`) of each species, in order.
    x_stab_indices: Vec<usize>,
    z_stab_indices: Vec<usize>,
    /// Support of a logical-`X` operator representative (column 0).
    logical_x_support: Vec<usize>,
    /// Support of a logical-`Z` operator representative (row 0).
    logical_z_support: Vec<usize>,
    /// Matching graph over `Z`-stabilizers (decodes `X` errors).
    graph_z: MatchingGraph,
    /// Matching graph over `X`-stabilizers (decodes `Z` errors).
    graph_x: MatchingGraph,
}

impl SurfaceCode {
    /// Build a distance-`d` rotated surface code.
    ///
    /// # Errors
    /// [`QuantumError::InvalidParameter`] if `distance` is `0` or even.
    pub fn new(distance: usize) -> QuantumResult<Self> {
        if distance == 0 || distance.is_multiple_of(2) {
            return Err(QuantumError::InvalidParameter {
                name: "distance must be a positive odd integer".into(),
            });
        }
        let d = distance;
        let num_data = d * d;
        let stabilizers = build_stabilizers(d);

        let x_stab_indices: Vec<usize> = stabilizers
            .iter()
            .enumerate()
            .filter(|(_, s)| s.kind == StabKind::X)
            .map(|(i, _)| i)
            .collect();
        let z_stab_indices: Vec<usize> = stabilizers
            .iter()
            .enumerate()
            .filter(|(_, s)| s.kind == StabKind::Z)
            .map(|(i, _)| i)
            .collect();

        // Logical-Z = Z along the top row (qubits 0..d).
        let logical_z_support: Vec<usize> = (0..d).collect();
        // Logical-X = X along the left column (qubits 0, d, 2d, …).
        let logical_x_support: Vec<usize> = (0..d).map(|r| r * d).collect();

        let graph_z = build_matching_graph(&stabilizers, &z_stab_indices, num_data);
        let graph_x = build_matching_graph(&stabilizers, &x_stab_indices, num_data);

        Ok(Self {
            distance: d,
            num_data,
            stabilizers,
            x_stab_indices,
            z_stab_indices,
            logical_x_support,
            logical_z_support,
            graph_z,
            graph_x,
        })
    }

    /// Build from a [`SurfaceCodeConfig`].
    ///
    /// # Errors
    /// As [`SurfaceCode::new`].
    pub fn from_config(cfg: SurfaceCodeConfig) -> QuantumResult<Self> {
        Self::new(cfg.distance)
    }

    /// Code distance `d`.
    #[must_use]
    pub fn distance(&self) -> usize {
        self.distance
    }

    /// Number of physical data qubits (`d²`).
    #[must_use]
    pub fn num_data_qubits(&self) -> usize {
        self.num_data
    }

    /// Total number of stabilizer generators (`d²−1`).
    #[must_use]
    pub fn num_stabilizers(&self) -> usize {
        self.stabilizers.len()
    }

    /// Number of `X`-type stabilizers.
    #[must_use]
    pub fn num_x_stabilizers(&self) -> usize {
        self.x_stab_indices.len()
    }

    /// Number of `Z`-type stabilizers.
    #[must_use]
    pub fn num_z_stabilizers(&self) -> usize {
        self.z_stab_indices.len()
    }

    /// Read-only view of the stabilizer generators.
    #[must_use]
    pub fn stabilizers(&self) -> &[Stabilizer] {
        &self.stabilizers
    }

    /// Support of the logical-`X` representative used by [`Self::logical_error`].
    #[must_use]
    pub fn logical_x_support(&self) -> &[usize] {
        &self.logical_x_support
    }

    /// Support of the logical-`Z` representative used by [`Self::logical_error`].
    #[must_use]
    pub fn logical_z_support(&self) -> &[usize] {
        &self.logical_z_support
    }

    /// Extract the stabilizer syndrome produced by a Pauli error.
    ///
    /// A `Z`-type stabilizer fires when its support overlaps the `X` component
    /// of the error in an odd number of qubits; an `X`-type stabilizer fires on
    /// odd overlap with the `Z` component.
    ///
    /// # Errors
    /// [`QuantumError::DimensionMismatch`] if `error` spans the wrong number of
    /// data qubits.
    pub fn syndrome(&self, error: &PauliError) -> QuantumResult<Syndrome> {
        if error.x.len() != self.num_data || error.z.len() != self.num_data {
            return Err(QuantumError::DimensionMismatch {
                expected: self.num_data,
                got: error.x.len(),
            });
        }
        let bits = self
            .stabilizers
            .iter()
            .map(|s| {
                let component = match s.kind {
                    StabKind::Z => &error.x,
                    StabKind::X => &error.z,
                };
                let overlap = s.support.iter().filter(|&&q| component[q]).count();
                !overlap.is_multiple_of(2)
            })
            .collect();
        Ok(Syndrome { bits })
    }

    /// Decode a syndrome into a correction [`PauliError`].
    ///
    /// `X` corrections come from matching the `Z`-stabilizer defects; `Z`
    /// corrections from the `X`-stabilizer defects.
    ///
    /// # Errors
    /// [`QuantumError::DimensionMismatch`] if `syndrome` has the wrong length.
    pub fn decode(&self, syndrome: &Syndrome) -> QuantumResult<PauliError> {
        if syndrome.bits.len() != self.stabilizers.len() {
            return Err(QuantumError::DimensionMismatch {
                expected: self.stabilizers.len(),
                got: syndrome.bits.len(),
            });
        }

        // Local (graph) indices of the fired stabilizers of each species.
        let z_defects: Vec<usize> = self
            .z_stab_indices
            .iter()
            .enumerate()
            .filter(|&(_, &g)| syndrome.bits[g])
            .map(|(l, _)| l)
            .collect();
        let x_defects: Vec<usize> = self
            .x_stab_indices
            .iter()
            .enumerate()
            .filter(|&(_, &g)| syndrome.bits[g])
            .map(|(l, _)| l)
            .collect();

        let x_correction = match_defects(&self.graph_z, &z_defects, self.num_data);
        let z_correction = match_defects(&self.graph_x, &x_defects, self.num_data);

        Ok(PauliError {
            x: x_correction,
            z: z_correction,
        })
    }

    /// Decode `error`'s syndrome and return the residual `error ∘ correction`.
    ///
    /// In the Pauli frame, the decode succeeded iff the residual has no logical
    /// component (see [`Self::logical_error`]).
    ///
    /// # Errors
    /// As [`Self::syndrome`].
    pub fn residual_after_correction(&self, error: &PauliError) -> QuantumResult<PauliError> {
        let syndrome = self.syndrome(error)?;
        let correction = self.decode(&syndrome)?;
        error.compose(&correction)
    }

    /// Whether a (residual) error implements a non-trivial logical operation,
    /// returned as `(logical_x_flipped, logical_z_flipped)`.
    ///
    /// A residual `X`-string flips the logical state by logical-`X` iff it
    /// anticommutes with the logical-`Z` operator, i.e. it overlaps the
    /// logical-`Z` support in an odd number of qubits (and symmetrically for
    /// the `Z` component against the logical-`X` support).
    ///
    /// # Errors
    /// [`QuantumError::DimensionMismatch`] if `error` spans the wrong number of
    /// data qubits.
    pub fn logical_error(&self, error: &PauliError) -> QuantumResult<(bool, bool)> {
        if error.x.len() != self.num_data || error.z.len() != self.num_data {
            return Err(QuantumError::DimensionMismatch {
                expected: self.num_data,
                got: error.x.len(),
            });
        }
        let x_flip = !self
            .logical_z_support
            .iter()
            .filter(|&&q| error.x[q])
            .count()
            .is_multiple_of(2);
        let z_flip = !self
            .logical_x_support
            .iter()
            .filter(|&&q| error.z[q])
            .count()
            .is_multiple_of(2);
        Ok((x_flip, z_flip))
    }
}

/// Construct the rotated-code stabilizer list for distance `d`.
fn build_stabilizers(d: usize) -> Vec<Stabilizer> {
    let mut stabs = Vec::new();
    // Plaquettes are anchored at dual-lattice corners (a, b) ∈ {0..=d}².
    for a in 0..=d {
        for b in 0..=d {
            let mut support = Vec::new();
            for (dr, dc) in [(-1i64, -1i64), (-1, 0), (0, -1), (0, 0)] {
                let r = a as i64 + dr;
                let c = b as i64 + dc;
                if r >= 0 && r < d as i64 && c >= 0 && c < d as i64 {
                    support.push((r as usize) * d + (c as usize));
                }
            }
            let kind = if (a + b).is_multiple_of(2) {
                StabKind::Z
            } else {
                StabKind::X
            };
            let keep = match support.len() {
                // Weight-4 bulk plaquette: always kept.
                4 => true,
                // Weight-2 boundary plaquette: X on top/bottom, Z on left/right.
                2 => {
                    let on_top_or_bottom = a == 0 || a == d;
                    let on_left_or_right = b == 0 || b == d;
                    match kind {
                        StabKind::X => on_top_or_bottom,
                        StabKind::Z => on_left_or_right,
                    }
                }
                // Weight-1 corners / empty: discarded.
                _ => false,
            };
            if keep {
                stabs.push(Stabilizer { kind, support });
            }
        }
    }
    stabs
}

/// Build the matching graph for one stabilizer species.
fn build_matching_graph(
    stabilizers: &[Stabilizer],
    local_indices: &[usize],
    num_data: usize,
) -> MatchingGraph {
    let num = local_indices.len();
    let boundary = num;
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num + 1];

    for q in 0..num_data {
        // Local indices of same-species stabilizers whose support contains q.
        let touching: Vec<usize> = local_indices
            .iter()
            .enumerate()
            .filter(|&(_, &g)| stabilizers[g].support.contains(&q))
            .map(|(l, _)| l)
            .collect();
        match touching.as_slice() {
            [s, t] => {
                adj[*s].push((*t, q));
                adj[*t].push((*s, q));
            }
            [s] => {
                adj[*s].push((boundary, q));
                adj[boundary].push((*s, q));
            }
            // A qubit untouched (or, impossibly, touched 3+ times) by this
            // species contributes no edge to this graph.
            _ => {}
        }
    }

    MatchingGraph {
        num_nodes: num + 1,
        boundary,
        adj,
    }
}

/// Result of [`bfs`]: per-node distances and predecessor `(node, qubit_label)`
/// links used to reconstruct chains.
type BfsResult = (Vec<usize>, Vec<Option<(usize, usize)>>);

/// Breadth-first search from `source` that never expands *out of* the boundary
/// node, so boundary-to-boundary chains are not produced. Returns distances and
/// predecessor `(node, qubit_label)` links for path reconstruction.
fn bfs(graph: &MatchingGraph, source: usize) -> BfsResult {
    let n = graph.num_nodes;
    let mut dist = vec![usize::MAX; n];
    let mut pred: Vec<Option<(usize, usize)>> = vec![None; n];
    let mut queue = VecDeque::new();
    dist[source] = 0;
    queue.push_back(source);
    while let Some(u) = queue.pop_front() {
        if u == graph.boundary {
            continue; // absorbing: never route a chain *through* the boundary
        }
        for &(v, q) in &graph.adj[u] {
            if dist[v] == usize::MAX {
                dist[v] = dist[u] + 1;
                pred[v] = Some((u, q));
                queue.push_back(v);
            }
        }
    }
    (dist, pred)
}

/// Reconstruct the data-qubit labels along the BFS path `source → target`.
fn path_qubits(pred: &[Option<(usize, usize)>], source: usize, target: usize) -> Vec<usize> {
    let mut qubits = Vec::new();
    let mut cur = target;
    while cur != source {
        match pred[cur] {
            Some((prev, q)) => {
                qubits.push(q);
                cur = prev;
            }
            None => break, // unreachable; leave partial (won't occur for valid codes)
        }
    }
    qubits
}

/// Exact minimum-weight perfect matching of `defects` (local node indices) to
/// each other or to the boundary, returning the `num_data`-length correction
/// mask (qubits to flip).
fn match_defects(graph: &MatchingGraph, defects: &[usize], num_data: usize) -> Vec<bool> {
    let mut correction = vec![false; num_data];
    if defects.is_empty() {
        return correction;
    }
    let k = defects.len();

    // One BFS per defect gives all pairwise distances + boundary distance.
    let searches: Vec<BfsResult> = defects.iter().map(|&d| bfs(graph, d)).collect();

    // Safety valve: an enormous syndrome would make the 2^k DP explode. Such
    // syndromes never arise from the low-weight errors this code corrects, but
    // guard anyway by greedily matching every defect to the boundary (always a
    // valid, if non-minimal, correction) — this cannot loop or hang.
    if k > 20 {
        for (i, &_d) in defects.iter().enumerate() {
            let path = path_qubits(&searches[i].1, defects[i], graph.boundary);
            for q in path {
                correction[q] ^= true;
            }
        }
        return correction;
    }

    let pair_dist = |i: usize, j: usize| searches[i].0[defects[j]];
    let bnd_dist = |i: usize| searches[i].0[graph.boundary];

    let size = 1usize << k;
    let mut dp = vec![usize::MAX; size];
    // partner[mask]: what the lowest set bit of `mask` matched to — usize::MAX
    // for the boundary, else the defect index it paired with.
    let mut partner = vec![usize::MAX; size];
    dp[0] = 0;

    for mask in 1..size {
        let i = mask.trailing_zeros() as usize;
        let rest = mask & !(1 << i);

        // Option 1: send defect i to the boundary.
        let bd = bnd_dist(i);
        if bd != usize::MAX && dp[rest] != usize::MAX {
            let cost = dp[rest] + bd;
            if cost < dp[mask] {
                dp[mask] = cost;
                partner[mask] = usize::MAX;
            }
        }

        // Option 2: pair defect i with another defect j still in the mask.
        let mut jbits = rest;
        while jbits != 0 {
            let j = jbits.trailing_zeros() as usize;
            jbits &= jbits - 1;
            let pd = pair_dist(i, j);
            let sub = rest & !(1 << j);
            if pd != usize::MAX && dp[sub] != usize::MAX {
                let cost = dp[sub] + pd;
                if cost < dp[mask] {
                    dp[mask] = cost;
                    partner[mask] = j;
                }
            }
        }
    }

    // Reconstruct the matching and accumulate the (XOR of) chains.
    let mut mask = size - 1;
    while mask != 0 {
        let i = mask.trailing_zeros() as usize;
        let p = partner[mask];
        if p == usize::MAX {
            let path = path_qubits(&searches[i].1, defects[i], graph.boundary);
            for q in path {
                correction[q] ^= true;
            }
            mask &= !(1 << i);
        } else {
            let path = path_qubits(&searches[i].1, defects[i], defects[p]);
            for q in path {
                correction[q] ^= true;
            }
            mask &= !((1 << i) | (1 << p));
        }
    }

    correction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_three_layout_counts() {
        let code = SurfaceCode::new(3).expect("valid distance-3 surface code");
        assert_eq!(code.distance(), 3);
        assert_eq!(code.num_data_qubits(), 9); // d²
        assert_eq!(code.num_stabilizers(), 8); // d² − 1
        assert_eq!(code.num_x_stabilizers(), 4); // (d²−1)/2
        assert_eq!(code.num_z_stabilizers(), 4);
        // Bulk plaquettes have weight 4, boundary plaquettes weight 2.
        let weight4 = code
            .stabilizers()
            .iter()
            .filter(|s| s.support.len() == 4)
            .count();
        let weight2 = code
            .stabilizers()
            .iter()
            .filter(|s| s.support.len() == 2)
            .count();
        assert_eq!(weight4, 4); // (d−1)²
        assert_eq!(weight2, 4); // 2(d−1)
    }

    #[test]
    fn css_stabilizers_commute() {
        // Every X-type and Z-type stabilizer must share an even number of qubits.
        let code = SurfaceCode::new(3).expect("valid distance-3 surface code");
        for sx in code.stabilizers().iter().filter(|s| s.kind == StabKind::X) {
            for sz in code.stabilizers().iter().filter(|s| s.kind == StabKind::Z) {
                let overlap = sx.support.iter().filter(|q| sz.support.contains(q)).count();
                assert!(overlap.is_multiple_of(2), "overlap={overlap}");
            }
        }
    }

    #[test]
    fn logical_operators_anticommute() {
        let code = SurfaceCode::new(3).expect("valid distance-3 surface code");
        // Logical X (col 0) and logical Z (row 0) share exactly one qubit.
        let overlap = code
            .logical_x_support()
            .iter()
            .filter(|q| code.logical_z_support().contains(q))
            .count();
        assert_eq!(overlap, 1);
        // Both have weight equal to the distance.
        assert_eq!(code.logical_x_support().len(), 3);
        assert_eq!(code.logical_z_support().len(), 3);
    }

    #[test]
    fn invalid_distance_rejected() {
        assert!(SurfaceCode::new(0).is_err()); // zero
        assert!(SurfaceCode::new(2).is_err()); // even
        assert!(SurfaceCode::new(4).is_err()); // even
        assert!(SurfaceCode::new(3).is_ok()); // odd
    }

    #[test]
    fn zero_error_gives_empty_syndrome_and_no_correction() {
        let code = SurfaceCode::new(3).expect("valid distance-3 surface code");
        let err = PauliError::identity(code.num_data_qubits());
        let syn = code.syndrome(&err).expect("syndrome for identity error");
        assert!(syn.is_trivial());
        let corr = code.decode(&syn).expect("decode trivial syndrome");
        assert_eq!(corr.weight(), 0);
    }

    #[test]
    fn single_x_on_bulk_qubit_fires_two_z_stabilizers() {
        let code = SurfaceCode::new(3).expect("valid distance-3 surface code");
        // Qubit 4 is the central qubit, in two Z-type plaquettes.
        let err = PauliError::single_x(code.num_data_qubits(), 4)
            .expect("valid single-X error on qubit 4");
        let syn = code
            .syndrome(&err)
            .expect("syndrome for single-X error on qubit 4");
        assert_eq!(syn.weight(), 2, "central X error should fire two defects");
        // The two defects must both be Z-type stabilizers (they detect X).
        for (i, &fired) in syn.bits.iter().enumerate() {
            if fired {
                assert_eq!(code.stabilizers()[i].kind, StabKind::Z);
            }
        }
        // …and the decoder corrects it with no logical error.
        let residual = code
            .residual_after_correction(&err)
            .expect("residual after correcting X error on qubit 4");
        assert_eq!(
            code.logical_error(&residual)
                .expect("logical error check on residual after X correction"),
            (false, false)
        );
    }

    #[test]
    fn single_z_on_bulk_qubit_fires_two_x_stabilizers() {
        let code = SurfaceCode::new(3).expect("valid distance-3 surface code");
        let err = PauliError::single_z(code.num_data_qubits(), 4)
            .expect("valid single-Z error on qubit 4");
        let syn = code
            .syndrome(&err)
            .expect("syndrome for single-Z error on qubit 4");
        assert_eq!(syn.weight(), 2);
        for (i, &fired) in syn.bits.iter().enumerate() {
            if fired {
                assert_eq!(code.stabilizers()[i].kind, StabKind::X);
            }
        }
        let residual = code
            .residual_after_correction(&err)
            .expect("residual after correcting Z error on qubit 4");
        assert_eq!(
            code.logical_error(&residual)
                .expect("logical error check on residual after Z correction"),
            (false, false)
        );
    }

    #[test]
    fn every_weight_one_error_is_corrected() {
        let code = SurfaceCode::new(3).expect("valid distance-3 surface code");
        let n = code.num_data_qubits();
        for q in 0..n {
            // Single X.
            let ex = PauliError::single_x(n, q).expect("valid single-X error on qubit q");
            let rx = code
                .residual_after_correction(&ex)
                .expect("residual after correcting single-X error");
            assert_eq!(
                code.logical_error(&rx)
                    .expect("logical error check on single-X residual"),
                (false, false),
                "X error on qubit {q} not corrected"
            );
            // Single Z.
            let ez = PauliError::single_z(n, q).expect("valid single-Z error on qubit q");
            let rz = code
                .residual_after_correction(&ez)
                .expect("residual after correcting single-Z error");
            assert_eq!(
                code.logical_error(&rz)
                    .expect("logical error check on single-Z residual"),
                (false, false),
                "Z error on qubit {q} not corrected"
            );
            // Single Y (= X·Z) is also weight one and must be corrected.
            let mut ey =
                PauliError::single_x(n, q).expect("valid single-X component of Y error on qubit q");
            ey.z[q] = true;
            let ry = code
                .residual_after_correction(&ey)
                .expect("residual after correcting single-Y error");
            assert_eq!(
                code.logical_error(&ry)
                    .expect("logical error check on single-Y residual"),
                (false, false),
                "Y error on qubit {q} not corrected"
            );
        }
    }

    #[test]
    fn corrected_residual_has_trivial_syndrome() {
        // After correction the residual must commute with every stabilizer.
        let code = SurfaceCode::new(3).expect("valid distance-3 surface code");
        let n = code.num_data_qubits();
        for q in 0..n {
            let err = PauliError::single_x(n, q).expect("valid single-X error on qubit q");
            let residual = code
                .residual_after_correction(&err)
                .expect("residual after correcting error on qubit q");
            let syn = code
                .syndrome(&residual)
                .expect("syndrome for residual error");
            assert!(syn.is_trivial(), "residual syndrome non-trivial for q={q}");
        }
    }

    #[test]
    fn boundary_x_error_fires_single_defect() {
        // Corner qubit 0 lies in a single Z-type (weight-2) boundary plaquette,
        // so a single X there produces exactly one defect, matched to the
        // boundary by the decoder.
        let code = SurfaceCode::new(3).expect("valid distance-3 surface code");
        let err = PauliError::single_x(code.num_data_qubits(), 0)
            .expect("valid single-X error on corner qubit 0");
        let syn = code
            .syndrome(&err)
            .expect("syndrome for boundary X error on qubit 0");
        assert_eq!(syn.weight(), 1);
        let residual = code
            .residual_after_correction(&err)
            .expect("residual after correcting boundary X error");
        assert_eq!(
            code.logical_error(&residual)
                .expect("logical error check on boundary residual"),
            (false, false)
        );
    }

    #[test]
    fn distance_five_layout_counts() {
        // The construction generalises: d=5 ⇒ 25 data, 24 stabilizers (12/12).
        let code = SurfaceCode::new(5).expect("valid distance-5 surface code");
        assert_eq!(code.num_data_qubits(), 25);
        assert_eq!(code.num_stabilizers(), 24);
        assert_eq!(code.num_x_stabilizers(), 12);
        assert_eq!(code.num_z_stabilizers(), 12);
    }
}
