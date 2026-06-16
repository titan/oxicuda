//! Exact optimal contraction-path finder via dynamic programming.
//!
//! Given N tensors (N ≤ 20) with labelled indices and associated dimensions, this module
//! finds the contraction order that minimises the total number of floating-point operations.
//!
//! # Algorithm
//!
//! Each tensor is described by a [`TensorSpec`] (a list of index IDs and their dimensions).
//! Contracting tensors T_A (index set S_A) and T_B (index set S_B) produces a tensor T_C
//! with index set S_C = (S_A ∪ S_B) \ (S_A ∩ S_B)  (symmetric difference) and costs
//!   flops = ∏_{i ∈ S_A ∪ S_B} dim(i).
//!
//! The DP iterates over all 2^N bitmask subsets in order of increasing popcount:
//!   `DP[mask]` = (optimal cumulative flops, optimal split mask pair)
//!
//! For N ≤ 15 this is very fast; N = 20 takes ≲ seconds.

use std::collections::HashMap;

use crate::{TnError, TnResult};

// ---------------------------------------------------------------------------
// Public data structures
// ---------------------------------------------------------------------------

/// Specification for a single tensor in the network: index IDs and their dimensions.
#[derive(Debug, Clone)]
pub struct TensorSpec {
    /// Index IDs assigned to each axis of this tensor.
    pub indices: Vec<usize>,
    /// Corresponding dimension for each index (same length as `indices`).
    pub shape: Vec<usize>,
}

impl TensorSpec {
    /// Construct a new [`TensorSpec`], validating that `indices` and `shape` have equal length.
    pub fn new(indices: Vec<usize>, shape: Vec<usize>) -> TnResult<Self> {
        if indices.len() != shape.len() {
            return Err(TnError::InvalidConfiguration(format!(
                "TensorSpec indices length {} ≠ shape length {}",
                indices.len(),
                shape.len()
            )));
        }
        Ok(Self { indices, shape })
    }
}

/// Configuration for the optimal contraction path search.
#[derive(Debug, Clone)]
pub struct ContractionPathConfig {
    /// Maximum number of tensors accepted. Networks larger than this return an error.
    /// Default: 20.
    pub max_n_tensors: usize,
    /// When two splits produce the same cumulative flops, prefer the one with smaller peak
    /// intermediate tensor size. Default: false.
    pub prefer_memory: bool,
}

impl Default for ContractionPathConfig {
    fn default() -> Self {
        Self {
            max_n_tensors: 20,
            prefer_memory: false,
        }
    }
}

/// The result of an optimal contraction path search.
#[derive(Debug, Clone)]
pub struct OptimalPath {
    /// Sequence of `(i, j)` pairs (i < j) into the *remaining* tensor list at each step.
    /// After each contraction, the result replaces index `i` and index `j` is removed.
    pub pairs: Vec<(usize, usize)>,
    /// Total number of FLOPs for the optimal order.
    pub total_flops: u64,
    /// Maximum number of elements in any intermediate tensor produced along the path.
    pub peak_intermediate_size: u64,
    /// Number of input tensors.
    pub n_tensors: usize,
}

/// Internal DP table entry keyed by subset bitmask.
#[derive(Debug, Clone)]
pub struct DpEntry {
    /// Cumulative FLOPs to contract all tensors in this subset optimally.
    pub flops: u64,
    /// Index IDs present in the contracted result of this subset.
    pub intermediate_indices: Vec<usize>,
    /// Corresponding dimensions of the contracted result.
    pub intermediate_dims: Vec<usize>,
    /// Bitmasks of the two sub-subsets whose combination achieves this optimum.
    /// `None` for singleton subsets (base case).
    pub split: Option<(u64, u64)>,
    /// Peak intermediate size across all contractions in this subtree.
    pub peak_size: u64,
}

// ---------------------------------------------------------------------------
// Core helpers
// ---------------------------------------------------------------------------

/// Build a [`HashMap`] from index-ID to dimension from the flat slice of pairs.
pub fn build_index_dims(index_dims: &[(usize, usize)]) -> HashMap<usize, usize> {
    index_dims.iter().copied().collect()
}

/// Compute the number of FLOPs for contracting two tensors with index sets `indices_a` and
/// `indices_b`: the cost is the product of *all* involved dimensions (union).
pub fn contraction_flops(
    indices_a: &[usize],
    indices_b: &[usize],
    index_dims: &HashMap<usize, usize>,
) -> u64 {
    let mut union: Vec<usize> = indices_a.to_vec();
    for &idx in indices_b {
        if !union.contains(&idx) {
            union.push(idx);
        }
    }
    union.iter().fold(1u64, |acc, &id| {
        acc.saturating_mul(*index_dims.get(&id).unwrap_or(&1) as u64)
    })
}

/// Compute the resulting index set after contracting A and B: symmetric difference
/// (indices that appear in exactly one of A, B).
pub fn contraction_result_indices(indices_a: &[usize], indices_b: &[usize]) -> Vec<usize> {
    let mut result: Vec<usize> = indices_a
        .iter()
        .filter(|&&x| !indices_b.contains(&x))
        .copied()
        .collect();
    result.extend(
        indices_b
            .iter()
            .filter(|&&x| !indices_a.contains(&x))
            .copied(),
    );
    result
}

/// Compute the dimensions of the result tensor from contracting A and B.
/// Returns (result_indices, result_dims).
fn contraction_result_spec(
    indices_a: &[usize],
    indices_b: &[usize],
    index_dims: &HashMap<usize, usize>,
) -> (Vec<usize>, Vec<usize>) {
    let result_indices = contraction_result_indices(indices_a, indices_b);
    let result_dims: Vec<usize> = result_indices
        .iter()
        .map(|id| *index_dims.get(id).unwrap_or(&1))
        .collect();
    (result_indices, result_dims)
}

/// Product of dimensions → number of elements (saturating).
fn dim_product(dims: &[usize]) -> u64 {
    dims.iter()
        .fold(1u64, |acc, &d| acc.saturating_mul(d as u64))
}

// ---------------------------------------------------------------------------
// DP over subsets
// ---------------------------------------------------------------------------

/// Core DP: fill a table of 2^N entries indexed by bitmask.
///
/// Returns a [`HashMap<u64, DpEntry>`] (sparse for N > 15) containing, for every non-empty
/// subset, the optimal contraction cost and split.
fn dp_over_subsets(
    tensors: &[TensorSpec],
    index_dims: &HashMap<usize, usize>,
    prefer_memory: bool,
) -> TnResult<HashMap<u64, DpEntry>> {
    let n = tensors.len();
    // Pre-compute base case for singleton subsets
    let mut table: HashMap<u64, DpEntry> = HashMap::new();
    for (i, tensor) in tensors.iter().enumerate() {
        let mask = 1u64 << i;
        table.insert(
            mask,
            DpEntry {
                flops: 0,
                intermediate_indices: tensor.indices.clone(),
                intermediate_dims: tensor.shape.clone(),
                split: None,
                peak_size: dim_product(&tensor.shape),
            },
        );
    }

    // Collect all non-singleton subsets sorted by popcount (ascending).
    // For N ≤ 20 we enumerate all 2^N masks.
    let total_masks = 1u64 << n;
    // Group masks by popcount to process in order
    let mut by_popcount: Vec<Vec<u64>> = vec![Vec::new(); n + 1];
    for mask in 1..total_masks {
        let pc = mask.count_ones() as usize;
        by_popcount[pc].push(mask);
    }

    for masks in by_popcount.iter().skip(2) {
        for &mask in masks {
            let mut best: Option<DpEntry> = None;
            // Enumerate all proper non-empty sub-subsets s1 of mask (bit trick).
            // We restrict s1 < mask - s1 to avoid double-counting.
            let mut s1 = mask & mask.wrapping_neg(); // lowest bit of mask
            loop {
                let s2 = mask ^ s1;
                // Avoid s1 == 0 and s1 == mask (both always ensured by loop structure),
                // and canonicalise by requiring s1 < s2.
                if s1 < s2 {
                    if let (Some(e1), Some(e2)) = (table.get(&s1), table.get(&s2)) {
                        let f = contraction_flops(
                            &e1.intermediate_indices,
                            &e2.intermediate_indices,
                            index_dims,
                        );
                        let cum_flops = e1.flops.saturating_add(e2.flops).saturating_add(f);
                        let (res_idx, res_dims) = contraction_result_spec(
                            &e1.intermediate_indices,
                            &e2.intermediate_indices,
                            index_dims,
                        );
                        let step_size = dim_product(&res_dims);
                        let peak = e1.peak_size.max(e2.peak_size).max(step_size);
                        let is_better = match &best {
                            None => true,
                            Some(b) => {
                                if cum_flops != b.flops {
                                    cum_flops < b.flops
                                } else if prefer_memory {
                                    peak < b.peak_size
                                } else {
                                    false
                                }
                            }
                        };
                        if is_better {
                            best = Some(DpEntry {
                                flops: cum_flops,
                                intermediate_indices: res_idx,
                                intermediate_dims: res_dims,
                                split: Some((s1, s2)),
                                peak_size: peak,
                            });
                        }
                    }
                }
                // Next proper subset
                s1 = (s1.wrapping_sub(mask)) & mask;
                if s1 == 0 {
                    break;
                }
            }
            if let Some(entry) = best {
                table.insert(mask, entry);
            } else {
                return Err(TnError::ContractionPathInvalid(format!(
                    "DP failed for subset mask {mask:#b}: could not find valid split"
                )));
            }
        }
    }
    Ok(table)
}

// ---------------------------------------------------------------------------
// Path reconstruction
// ---------------------------------------------------------------------------

/// Reconstruct the sequence of (i, j) contraction pairs from the DP table.
///
/// This translates bitmask splits into pairs of indices into the *working list* of tensors,
/// matching the convention used by [`path::ContractionPath`].
fn reconstruct_path(table: &HashMap<u64, DpEntry>, n: usize) -> TnResult<Vec<(usize, usize)>> {
    if n == 1 {
        return Ok(Vec::new());
    }
    let full_mask = (1u64 << n) - 1;
    let mut steps: Vec<(usize, usize)> = Vec::with_capacity(n - 1);

    // We'll simulate the working list: each entry holds the original bitmasks of the tensors
    // that have been merged into it. We process contractions in the order dictated by the DP.
    // Collect all contractions in order (leaf-to-root of the binary tree).
    let mut order: Vec<(u64, u64)> = Vec::new();
    collect_contractions(table, full_mask, &mut order)?;

    // `order[0]` is the last (root) contraction; we reverse for leaf-first ordering.
    order.reverse();

    // Working list: maps "slot index in remaining list" to a bitmask of original tensors.
    let mut working: Vec<u64> = (0..n as u64).map(|i| 1u64 << i).collect();

    for (s1, s2) in order {
        // Find positions of s1 and s2 in the working list.
        // A working slot matches sub-bitmask m if slot's bitmask == m (exact match at leaves)
        // or is a merged superset produced by a prior step.
        let pos1 = working.iter().position(|&m| m == s1).ok_or_else(|| {
            TnError::ContractionPathInvalid(format!("could not find s1={s1:#b} in working list"))
        })?;
        let pos2 = working.iter().position(|&m| m == s2).ok_or_else(|| {
            TnError::ContractionPathInvalid(format!("could not find s2={s2:#b} in working list"))
        })?;
        let (i, j) = if pos1 < pos2 {
            (pos1, pos2)
        } else {
            (pos2, pos1)
        };
        steps.push((i, j));
        // Merge: replace slot i with union, remove slot j.
        let merged = working[i] | working[j];
        working[i] = merged;
        working.remove(j);
    }
    Ok(steps)
}

/// Recursively collect (s1, s2) pairs in depth-first order (root first).
fn collect_contractions(
    table: &HashMap<u64, DpEntry>,
    mask: u64,
    order: &mut Vec<(u64, u64)>,
) -> TnResult<()> {
    let entry = table.get(&mask).ok_or_else(|| {
        TnError::ContractionPathInvalid(format!("missing DP entry for mask {mask:#b}"))
    })?;
    if let Some((s1, s2)) = entry.split {
        order.push((s1, s2));
        collect_contractions(table, s1, order)?;
        collect_contractions(table, s2, order)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Find the exact optimal contraction path for a tensor network via DP over subsets.
///
/// `tensors`: each tensor described by its index IDs and corresponding dimensions.
/// `index_dims`: slice of `(index_id, dimension)` pairs covering all indices used.
/// `config`: tuning parameters (max network size, memory tie-breaking).
///
/// Returns an [`OptimalPath`] containing the sequence of pairwise contractions as
/// `(i, j)` pairs (i < j) into the remaining working list at each step.
pub fn optimal_contraction_path(
    tensors: &[TensorSpec],
    index_dims: &[(usize, usize)],
    config: &ContractionPathConfig,
) -> TnResult<OptimalPath> {
    let n = tensors.len();
    if n == 0 {
        return Err(TnError::EmptyInput);
    }
    if n > config.max_n_tensors {
        return Err(TnError::InvalidParameter {
            name: "n_tensors".into(),
            reason: format!(
                "{n} exceeds max_n_tensors={} (increase config.max_n_tensors if intended)",
                config.max_n_tensors
            ),
        });
    }
    if n == 1 {
        let peak = dim_product(&tensors[0].shape);
        return Ok(OptimalPath {
            pairs: Vec::new(),
            total_flops: 0,
            peak_intermediate_size: peak,
            n_tensors: 1,
        });
    }

    let idx_map = build_index_dims(index_dims);
    let table = dp_over_subsets(tensors, &idx_map, config.prefer_memory)?;

    let full_mask = (1u64 << n) - 1;
    let root = table
        .get(&full_mask)
        .ok_or_else(|| TnError::ContractionPathInvalid("DP root entry missing".into()))?;

    let total_flops = root.flops;
    let peak_intermediate_size = root.peak_size;
    let pairs = reconstruct_path(&table, n)?;

    Ok(OptimalPath {
        pairs,
        total_flops,
        peak_intermediate_size,
        n_tensors: n,
    })
}

/// Compute a simple greedy contraction order (at the specification level, without tensor data)
/// and return its total FLOPs.  Used for comparison in [`compare_with_greedy`].
pub fn greedy_flops(tensors: &[TensorSpec], index_dims: &HashMap<usize, usize>) -> u64 {
    let mut working: Vec<(Vec<usize>, Vec<usize>)> = tensors
        .iter()
        .map(|t| (t.indices.clone(), t.shape.clone()))
        .collect();
    let mut total = 0u64;
    while working.len() > 1 {
        let mut best_cost = u64::MAX;
        let mut best_i = 0usize;
        let mut best_j = 1usize;
        let mut best_indices: Vec<usize> = Vec::new();
        let mut best_dims: Vec<usize> = Vec::new();
        for i in 0..working.len() {
            for j in i + 1..working.len() {
                let cost = contraction_flops(&working[i].0, &working[j].0, index_dims);
                if cost < best_cost {
                    best_cost = cost;
                    best_i = i;
                    best_j = j;
                    let (ri, rd) =
                        contraction_result_spec(&working[i].0, &working[j].0, index_dims);
                    best_indices = ri;
                    best_dims = rd;
                }
            }
        }
        total = total.saturating_add(best_cost);
        working[best_i] = (best_indices, best_dims);
        working.remove(best_j);
    }
    total
}

/// Compare the optimal DP path with a greedy path on the same network.
///
/// Returns `(optimal_path, greedy_flops)`.  The ratio `greedy_flops / optimal.total_flops`
/// gives the speedup achievable by using the optimal order.
pub fn compare_with_greedy(
    tensors: &[TensorSpec],
    index_dims: &[(usize, usize)],
) -> TnResult<(OptimalPath, u64)> {
    let config = ContractionPathConfig::default();
    let optimal = optimal_contraction_path(tensors, index_dims, &config)?;
    let idx_map = build_index_dims(index_dims);
    let gf = greedy_flops(tensors, &idx_map);
    Ok((optimal, gf))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build index_dims slice from a flat map.
    fn make_idx(pairs: &[(usize, usize)]) -> Vec<(usize, usize)> {
        pairs.to_vec()
    }

    // -----------------------------------------------------------------------
    // Test 1: single tensor — no contractions, flops = 0
    // -----------------------------------------------------------------------
    #[test]
    fn single_tensor_no_contractions() {
        let spec = TensorSpec::new(vec![0, 1], vec![3, 4]).expect("new should succeed");
        let index_dims = make_idx(&[(0, 3), (1, 4)]);
        let config = ContractionPathConfig::default();
        let path = optimal_contraction_path(&[spec], &index_dims, &config)
            .expect("optimal_contraction_path should succeed");
        assert_eq!(path.n_tensors, 1);
        assert_eq!(path.pairs.len(), 0);
        assert_eq!(path.total_flops, 0);
    }

    // -----------------------------------------------------------------------
    // Test 2: two tensors — exactly one contraction with correct flops
    // Matrix multiply: [2,3] × [3,4] → cost = 2*3*4 = 24
    // -----------------------------------------------------------------------
    #[test]
    fn two_tensors_matrix_multiply_flops() {
        // A[i,j], B[j,k]; shared index j
        let a = TensorSpec::new(vec![0, 1], vec![2, 3]).expect("new should succeed"); // i=0,j=1
        let b = TensorSpec::new(vec![1, 2], vec![3, 4]).expect("new should succeed"); // j=1,k=2
        let index_dims = make_idx(&[(0, 2), (1, 3), (2, 4)]);
        let config = ContractionPathConfig::default();
        let path = optimal_contraction_path(&[a, b], &index_dims, &config)
            .expect("optimal_contraction_path should succeed");
        assert_eq!(path.pairs.len(), 1);
        assert_eq!(path.total_flops, 24); // 2*3*4
        assert_eq!(path.pairs[0], (0, 1));
    }

    // -----------------------------------------------------------------------
    // Test 3: contraction_flops helper — matrix multiply
    // -----------------------------------------------------------------------
    #[test]
    fn contraction_flops_matrix_multiply() {
        let idx_map: HashMap<usize, usize> = [(0, 2), (1, 3), (2, 4)].iter().copied().collect();
        let f = contraction_flops(&[0, 1], &[1, 2], &idx_map);
        assert_eq!(f, 24); // 2*3*4
    }

    // -----------------------------------------------------------------------
    // Test 4: contraction_result_indices — symmetric difference
    // -----------------------------------------------------------------------
    #[test]
    fn result_indices_symmetric_difference() {
        // A has {0,1,2}, B has {1,2,3} → result = {0,3}
        let res = contraction_result_indices(&[0, 1, 2], &[1, 2, 3]);
        assert!(res.contains(&0));
        assert!(res.contains(&3));
        assert!(!res.contains(&1));
        assert!(!res.contains(&2));
        assert_eq!(res.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Test 5: three tensors — known example where order matters
    // A[i,j] × B[j,k] × C[k,l]  with dims i=2,j=10,k=10,l=2
    // Order (AB)C costs: 2*10*10 + 2*10*2 = 200+40 = 240
    // Order A(BC) costs: 10*10*2 + 2*10*2 = 200+40 = 240  (same by symmetry here)
    // Try asymmetric: i=2,j=100,k=2,l=2
    // (AB)C: 2*100*2 + 2*2*2 = 400+8 = 408
    // A(BC): 100*2*2 + 2*100*2 = 400+400 = 800
    // → optimal is (AB)C
    // -----------------------------------------------------------------------
    #[test]
    fn three_tensors_optimal_order() {
        // A[0,1], B[1,2], C[2,3]; dims: 0=2,1=100,2=2,3=2
        let a = TensorSpec::new(vec![0, 1], vec![2, 100]).expect("new should succeed");
        let b = TensorSpec::new(vec![1, 2], vec![100, 2]).expect("new should succeed");
        let c = TensorSpec::new(vec![2, 3], vec![2, 2]).expect("new should succeed");
        let index_dims = make_idx(&[(0, 2), (1, 100), (2, 2), (3, 2)]);
        let config = ContractionPathConfig::default();
        let path = optimal_contraction_path(&[a, b, c], &index_dims, &config)
            .expect("optimal_contraction_path should succeed");
        assert_eq!(path.pairs.len(), 2);
        // Optimal: contract A and B first (tensors 0,1), then with C
        // This means pairs[0] = (0,1), pairs[1] = (0,1) or similar
        // Verify flops: optimal = 408
        assert_eq!(path.total_flops, 408);
    }

    // -----------------------------------------------------------------------
    // Test 6: N > max_n_tensors → error
    // -----------------------------------------------------------------------
    #[test]
    fn too_many_tensors_returns_error() {
        let config = ContractionPathConfig {
            max_n_tensors: 5,
            prefer_memory: false,
        };
        let tensors: Vec<TensorSpec> = (0..6)
            .map(|i| TensorSpec::new(vec![i], vec![2]).expect("new should succeed"))
            .collect();
        let index_dims: Vec<(usize, usize)> = (0..6).map(|i| (i, 2)).collect();
        let result = optimal_contraction_path(&tensors, &index_dims, &config);
        assert!(result.is_err());
        match result.unwrap_err() {
            TnError::InvalidParameter { name, .. } => assert_eq!(name, "n_tensors"),
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 7: empty input → error
    // -----------------------------------------------------------------------
    #[test]
    fn empty_input_returns_error() {
        let config = ContractionPathConfig::default();
        let result = optimal_contraction_path(&[], &[], &config);
        assert!(matches!(result.unwrap_err(), TnError::EmptyInput));
    }

    // -----------------------------------------------------------------------
    // Test 8: all indices disjoint (outer product)
    // Four tensors, no shared indices. Each step costs product of all involved dims.
    // -----------------------------------------------------------------------
    #[test]
    fn outer_product_no_shared_indices() {
        // A[0],B[1],C[2],D[3], all dim=3 → full outer product
        let tensors: Vec<TensorSpec> = (0..4)
            .map(|i| TensorSpec::new(vec![i], vec![3]).expect("new should succeed"))
            .collect();
        let index_dims: Vec<(usize, usize)> = (0..4).map(|i| (i, 3)).collect();
        let config = ContractionPathConfig::default();
        let path = optimal_contraction_path(&tensors, &index_dims, &config)
            .expect("optimal_contraction_path should succeed");
        assert_eq!(path.pairs.len(), 3);
        // All contractions are outer products; DP finds the minimum-cost order.
        // Optimal: pair up (A×B) and (C×D) first, then combine:
        // 3×3 + 3×3 + 9×9 = 9 + 9 + 81 = 99
        // (Sequential order 9+27+81=117 is strictly worse.)
        assert_eq!(path.total_flops, 99);
    }

    // -----------------------------------------------------------------------
    // Test 9: full contraction (all indices shared) → scalar result
    // -----------------------------------------------------------------------
    #[test]
    fn full_shared_indices_scalar_result() {
        // A[0,1], B[0,1] — both share all indices → result is scalar
        let a = TensorSpec::new(vec![0, 1], vec![4, 5]).expect("new should succeed");
        let b = TensorSpec::new(vec![0, 1], vec![4, 5]).expect("new should succeed");
        let index_dims = make_idx(&[(0, 4), (1, 5)]);
        let config = ContractionPathConfig::default();
        let path = optimal_contraction_path(&[a, b], &index_dims, &config)
            .expect("optimal_contraction_path should succeed");
        assert_eq!(path.total_flops, 20); // 4*5
        // result has 0 indices → scalar
        let idx_map = build_index_dims(&index_dims);
        let result_idxs = contraction_result_indices(&[0, 1], &[0, 1]);
        assert!(result_idxs.is_empty());
        let result_dims: Vec<usize> = result_idxs.iter().map(|id| idx_map[id]).collect();
        assert_eq!(dim_product(&result_dims), 1); // scalar → 1 element
    }

    // -----------------------------------------------------------------------
    // Test 10: matrix chain multiplication (classic DP example)
    // Matrices: A(10×30), B(30×5), C(5×60)
    // (AB)C: 10*30*5 + 10*5*60 = 1500 + 3000 = 4500
    // A(BC): 30*5*60 + 10*30*60 = 9000 + 18000 = 27000
    // Optimal: (AB)C with 4500 flops
    // -----------------------------------------------------------------------
    #[test]
    fn matrix_chain_optimal_matches_known_result() {
        // Index 0:row_A=10, 1:col_A=row_B=30, 2:col_B=row_C=5, 3:col_C=60
        let a = TensorSpec::new(vec![0, 1], vec![10, 30]).expect("new should succeed");
        let b = TensorSpec::new(vec![1, 2], vec![30, 5]).expect("new should succeed");
        let c = TensorSpec::new(vec![2, 3], vec![5, 60]).expect("new should succeed");
        let index_dims = make_idx(&[(0, 10), (1, 30), (2, 5), (3, 60)]);
        let config = ContractionPathConfig::default();
        let path = optimal_contraction_path(&[a, b, c], &index_dims, &config)
            .expect("optimal_contraction_path should succeed");
        // Optimal: (AB)C costs 10*30*5 + 10*5*60 = 1500+3000 = 4500
        assert_eq!(path.total_flops, 4500);
        assert_eq!(path.pairs.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Test 11: optimal path always ≤ greedy path in flops
    // -----------------------------------------------------------------------
    #[test]
    fn optimal_le_greedy_for_small_networks() {
        // Asymmetric network where greedy may not be optimal
        // A[0,1] dim 2,50; B[1,2] dim 50,3; C[0,2] dim 2,3; D[2,3] dim 3,4
        let a = TensorSpec::new(vec![0, 1], vec![2, 50]).expect("new should succeed");
        let b = TensorSpec::new(vec![1, 2], vec![50, 3]).expect("new should succeed");
        let c = TensorSpec::new(vec![0, 2], vec![2, 3]).expect("new should succeed");
        let d = TensorSpec::new(vec![2, 3], vec![3, 4]).expect("new should succeed");
        let index_dims = make_idx(&[(0, 2), (1, 50), (2, 3), (3, 4)]);
        let (opt, gf) = compare_with_greedy(&[a, b, c, d], &index_dims)
            .expect("compare_with_greedy should succeed");
        assert!(
            opt.total_flops <= gf,
            "optimal ({}) should be ≤ greedy ({})",
            opt.total_flops,
            gf
        );
    }

    // -----------------------------------------------------------------------
    // Test 12: 8-tensor ring computes in reasonable time
    // Tensors T_i[i, (i+1)%8], all dim=4 — a ring of 8 tensors
    // -----------------------------------------------------------------------
    #[test]
    fn ring_of_8_tensors_completes() {
        let n = 8usize;
        let dim = 4usize;
        // Each tensor i has indices (i, (i+1)%n) with dimension dim
        let tensors: Vec<TensorSpec> = (0..n)
            .map(|i| {
                TensorSpec::new(vec![i, (i + 1) % n], vec![dim, dim])
                    .expect("value should be present")
            })
            .collect();
        let index_dims: Vec<(usize, usize)> = (0..n).map(|i| (i, dim)).collect();
        let config = ContractionPathConfig::default();
        let path = optimal_contraction_path(&tensors, &index_dims, &config)
            .expect("optimal_contraction_path should succeed");
        assert_eq!(path.pairs.len(), n - 1);
        // Sanity: total flops should be positive
        assert!(path.total_flops > 0);
    }

    // -----------------------------------------------------------------------
    // Test 13: prefer_memory tie-breaking flag does not affect correctness
    // -----------------------------------------------------------------------
    #[test]
    fn prefer_memory_produces_valid_path() {
        let a = TensorSpec::new(vec![0, 1], vec![3, 3]).expect("new should succeed");
        let b = TensorSpec::new(vec![1, 2], vec![3, 3]).expect("new should succeed");
        let c = TensorSpec::new(vec![2, 0], vec![3, 3]).expect("new should succeed");
        let index_dims = make_idx(&[(0, 3), (1, 3), (2, 3)]);
        let config_mem = ContractionPathConfig {
            max_n_tensors: 20,
            prefer_memory: true,
        };
        let config_flops = ContractionPathConfig::default();
        let path_mem =
            optimal_contraction_path(&[a.clone(), b.clone(), c.clone()], &index_dims, &config_mem)
                .expect("value should be present");
        let path_flops = optimal_contraction_path(&[a, b, c], &index_dims, &config_flops)
            .expect("optimal_contraction_path should succeed");
        // Both must produce a valid 2-step path
        assert_eq!(path_mem.pairs.len(), 2);
        assert_eq!(path_flops.pairs.len(), 2);
        // Memory mode should have flops ≥ optimal (can't be strictly better)
        assert!(path_mem.total_flops >= path_flops.total_flops);
    }

    // -----------------------------------------------------------------------
    // Test 14: contraction_result_indices is idempotent for disjoint sets
    // -----------------------------------------------------------------------
    #[test]
    fn result_indices_disjoint_is_union() {
        let a = vec![0usize, 1, 2];
        let b = vec![3usize, 4, 5];
        let res = contraction_result_indices(&a, &b);
        assert_eq!(res.len(), 6);
        for &i in a.iter().chain(b.iter()) {
            assert!(res.contains(&i));
        }
    }

    // -----------------------------------------------------------------------
    // Test 15: build_index_dims builds correct map
    // -----------------------------------------------------------------------
    #[test]
    fn build_index_dims_correct() {
        let pairs = vec![(0usize, 10usize), (1, 20), (2, 30)];
        let map = build_index_dims(&pairs);
        assert_eq!(map[&0], 10);
        assert_eq!(map[&1], 20);
        assert_eq!(map[&2], 30);
    }

    // -----------------------------------------------------------------------
    // Test 16: chain of 4 tensors optimal matches expected
    // A(2×3), B(3×4), C(4×5), D(5×6)
    // All orders produce different totals; DP finds minimum.
    // Best: ((AB)C)D: 2*3*4 + 2*4*5 + 2*5*6 = 24+40+60 = 124
    // A((BC)D): 3*4*5 + 3*5*6 + 2*3*6 = 60+90+36 = 186
    // (AB)(CD): 2*3*4 + 4*5*6 + 2*4*6 = 24+120+48 = 192
    // -----------------------------------------------------------------------
    #[test]
    fn chain_of_4_optimal_flops() {
        let a = TensorSpec::new(vec![0, 1], vec![2, 3]).expect("new should succeed");
        let b = TensorSpec::new(vec![1, 2], vec![3, 4]).expect("new should succeed");
        let c = TensorSpec::new(vec![2, 3], vec![4, 5]).expect("new should succeed");
        let d = TensorSpec::new(vec![3, 4], vec![5, 6]).expect("new should succeed");
        let index_dims = make_idx(&[(0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]);
        let config = ContractionPathConfig::default();
        let path = optimal_contraction_path(&[a, b, c, d], &index_dims, &config)
            .expect("optimal_contraction_path should succeed");
        assert_eq!(path.pairs.len(), 3);
        // Optimal ((AB)C)D = 124
        assert_eq!(path.total_flops, 124);
    }
}
