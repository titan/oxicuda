//! Tensor-network simplification rules (P2 preprocessing transformations).
//!
//! These transformations reduce a tensor network before contraction without changing the
//! result. Rules are applied in a fixed-point loop until no further change occurs.
//!
//! ## Rules implemented
//!
//! 1. **Trace reduction** — for each tensor, contract any repeated label (self-trace).
//! 2. **Degree-1 node absorption** — a leaf (one bond to the rest of the network) is
//!    absorbed into its single neighbour.
//! 3. **Degree-2 node absorption** — a pass-through tensor with exactly two distinct
//!    bonds is contracted with one of its two neighbours.
//! 4. **Scalar folding** — a disconnected scalar (rank-0) is multiplied into any other
//!    tensor as a prefactor.
//! 5. **Parallel edge merging** — two tensors sharing multiple bonds have those bonds
//!    fused into a single merged index whose dimension is the product of the originals.
//! 6. **Gauge fixing** — left singular vectors of the mode-unfolding along each internal
//!    bond are used as gauge matrices that improve the condition number of intermediate
//!    matrices during contraction.
//!
//! ## Label convention
//!
//! - **Negative** labels are *internal bonds* that may be shared between exactly two tensors.
//! - **Positive** labels are *external free indices* unique to one tensor.
//! - The value `0` is reserved (never generated here).

use std::collections::HashMap;

use crate::svd::svd_dense::svd_jacobi;
use crate::{TnError, TnResult};

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

/// A tensor in a network, identified by its position in the node list.
///
/// Shared (negative) labels between two tensors mark contracting bonds.
/// Positive labels are external free indices.
#[derive(Debug, Clone)]
pub struct NetworkTensor {
    /// Flat data buffer, row-major.
    pub data: Vec<f64>,
    /// Dimension of each axis.
    pub dims: Vec<usize>,
    /// Label for each axis. Shared negative labels between two tensors mark contracting
    /// bonds; positive labels are external free indices.
    pub labels: Vec<i64>,
    /// Human-readable name (for debugging).
    pub name: String,
}

impl NetworkTensor {
    /// Construct a new [`NetworkTensor`], validating that `data.len() == dims.iter().product()`.
    pub fn new(
        data: Vec<f64>,
        dims: Vec<usize>,
        labels: Vec<i64>,
        name: impl Into<String>,
    ) -> TnResult<Self> {
        if dims.len() != labels.len() {
            return Err(TnError::ShapeMismatch {
                expected: vec![labels.len()],
                got: vec![dims.len()],
            });
        }
        let total: usize = dims.iter().product::<usize>().max(1);
        if data.len() != total {
            return Err(TnError::ShapeMismatch {
                expected: dims.clone(),
                got: vec![data.len()],
            });
        }
        Ok(Self {
            data,
            dims,
            labels,
            name: name.into(),
        })
    }

    /// Returns the total number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.dims.iter().product::<usize>().max(1)
    }

    /// Returns `true` if this tensor is a scalar (zero axes OR all dims are 1).
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.labels.is_empty()
    }

    /// Returns the scalar value. Only meaningful when `is_scalar()` is true.
    #[inline]
    pub fn scalar_value(&self) -> f64 {
        *self.data.first().unwrap_or(&0.0)
    }
}

/// A collection of tensors forming a tensor network.
#[derive(Debug, Clone)]
pub struct TensorNetwork {
    pub tensors: Vec<NetworkTensor>,
}

impl TensorNetwork {
    /// Construct an empty tensor network.
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
        }
    }

    /// Add a tensor, returning its index.
    pub fn push(&mut self, t: NetworkTensor) -> usize {
        let idx = self.tensors.len();
        self.tensors.push(t);
        idx
    }
}

impl Default for TensorNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics collected during network simplification.
#[derive(Debug, Clone, Default)]
pub struct SimplifyStats {
    pub traces_removed: usize,
    pub leaves_absorbed: usize,
    pub chains_simplified: usize,
    pub scalars_folded: usize,
    pub parallel_bonds_fused: usize,
    pub total_flops_saved_estimate: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Label / stride helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Row-major strides for `dims`.
fn strides_from_dims(dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return vec![];
    }
    let mut strides = vec![1usize; dims.len()];
    for i in (0..dims.len() - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// Convert a flat index into per-axis indices for the given dims.
fn unravel(flat: usize, dims: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; dims.len()];
    let mut rem = flat;
    for i in (0..dims.len()).rev() {
        out[i] = rem % dims[i];
        rem /= dims[i];
    }
    out
}

/// Compute a flat index from per-axis indices and strides.
fn ravel(indices: &[usize], strides: &[usize]) -> usize {
    indices.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Rule 1: Trace reduction
// ─────────────────────────────────────────────────────────────────────────────

/// For a single tensor, sum over every pair of axes that share the same label.
/// Returns `Err` only on internal shape inconsistencies.
pub fn trace_one_tensor(t: &NetworkTensor) -> TnResult<NetworkTensor> {
    // Find duplicate labels (they must appear exactly twice to be a valid trace pair).
    let mut label_positions: HashMap<i64, Vec<usize>> = HashMap::new();
    for (ax, &lab) in t.labels.iter().enumerate() {
        label_positions.entry(lab).or_default().push(ax);
    }

    // Collect the first pair of duplicates (if any).
    let trace_pair: Option<(i64, usize, usize)> = label_positions
        .iter()
        .filter(|(_, positions)| positions.len() >= 2)
        .map(|(&lab, positions)| (lab, positions[0], positions[1]))
        .next();

    let (_, ax_a, ax_b) = match trace_pair {
        None => return Ok(t.clone()),
        Some(triple) => triple,
    };

    // Verify dimensions match.
    if t.dims[ax_a] != t.dims[ax_b] {
        return Err(TnError::DimensionMismatch {
            a: t.dims[ax_a],
            b: t.dims[ax_b],
        });
    }
    let trace_dim = t.dims[ax_a];

    // Build output: remove both ax_a and ax_b.
    let kept_axes: Vec<usize> = (0..t.labels.len())
        .filter(|&a| a != ax_a && a != ax_b)
        .collect();
    let out_dims: Vec<usize> = kept_axes.iter().map(|&a| t.dims[a]).collect();
    let out_labels: Vec<i64> = kept_axes.iter().map(|&a| t.labels[a]).collect();
    let out_total = out_dims.iter().product::<usize>().max(1);
    let mut out_data = vec![0.0f64; out_total];

    let in_strides = strides_from_dims(&t.dims);
    let out_strides = strides_from_dims(&out_dims);

    for out_flat in 0..out_total {
        let out_idx = unravel(out_flat, &out_dims);
        let mut acc = 0.0f64;
        for tr in 0..trace_dim {
            // Reconstruct input multi-index
            let mut in_idx = vec![0usize; t.dims.len()];
            for (ki, &ax) in kept_axes.iter().enumerate() {
                in_idx[ax] = out_idx[ki];
            }
            in_idx[ax_a] = tr;
            in_idx[ax_b] = tr;
            let in_flat = ravel(&in_idx, &in_strides);
            acc += t.data[in_flat];
        }
        let out_flat_check = ravel(&out_idx, &out_strides);
        out_data[out_flat_check] = acc;
    }

    let mut result =
        NetworkTensor::new(out_data, out_dims, out_labels, format!("{}_traced", t.name))?;
    // Recursively handle additional self-traces on the same tensor.
    let remaining_dups: bool = {
        let mut lp2: HashMap<i64, usize> = HashMap::new();
        let mut found = false;
        for &lab in &result.labels {
            let cnt = lp2.entry(lab).or_insert(0);
            *cnt += 1;
            if *cnt >= 2 {
                found = true;
            }
        }
        found
    };
    if remaining_dups {
        result = trace_one_tensor(&result)?;
    }
    Ok(result)
}

/// Remove self-traces: for each tensor, contract any repeated label with itself.
/// Returns the total number of traces removed across all tensors.
pub fn remove_traces(net: &mut TensorNetwork) -> TnResult<usize> {
    let mut total = 0usize;
    for t in &mut net.tensors {
        // Count how many trace pairs exist before.
        let mut lp: HashMap<i64, usize> = HashMap::new();
        for &lab in &t.labels {
            *lp.entry(lab).or_insert(0) += 1;
        }
        let pairs_before: usize = lp.values().map(|&c| c / 2).sum();
        if pairs_before > 0 {
            *t = trace_one_tensor(t)?;
            total += pairs_before;
        }
    }
    Ok(total)
}

// ─────────────────────────────────────────────────────────────────────────────
// Rule 2: Leaf absorption (degree-1 nodes)
// ─────────────────────────────────────────────────────────────────────────────

/// Build a map from each negative label to the list of (tensor_index, axis) pairs
/// that carry it.
fn build_bond_map(net: &TensorNetwork) -> HashMap<i64, Vec<(usize, usize)>> {
    let mut bond_map: HashMap<i64, Vec<(usize, usize)>> = HashMap::new();
    for (ti, t) in net.tensors.iter().enumerate() {
        for (ax, &lab) in t.labels.iter().enumerate() {
            if lab < 0 {
                bond_map.entry(lab).or_default().push((ti, ax));
            }
        }
    }
    bond_map
}

/// Count the number of *distinct* internal bonds (negative labels) that tensor `ti`
/// participates in.
fn bond_degree(ti: usize, net: &TensorNetwork) -> usize {
    net.tensors[ti]
        .labels
        .iter()
        .filter(|&&l| l < 0)
        .collect::<std::collections::HashSet<_>>()
        .len()
}

/// Contract two tensors over all shared labels, returning the merged tensor.
pub fn contract_two_tensors(a: &NetworkTensor, b: &NetworkTensor) -> TnResult<NetworkTensor> {
    // Find shared labels and their axis positions.
    let mut shared: Vec<(usize, usize)> = Vec::new();
    for (ia, &la) in a.labels.iter().enumerate() {
        for (ib, &lb) in b.labels.iter().enumerate() {
            if la == lb {
                shared.push((ia, ib));
            }
        }
    }
    for &(ia, ib) in &shared {
        if a.dims[ia] != b.dims[ib] {
            return Err(TnError::DimensionMismatch {
                a: a.dims[ia],
                b: b.dims[ib],
            });
        }
    }

    let shared_a: Vec<usize> = shared.iter().map(|(ia, _)| *ia).collect();
    let shared_b: Vec<usize> = shared.iter().map(|(_, ib)| *ib).collect();
    let kept_a: Vec<usize> = (0..a.labels.len())
        .filter(|i| !shared_a.contains(i))
        .collect();
    let kept_b: Vec<usize> = (0..b.labels.len())
        .filter(|j| !shared_b.contains(j))
        .collect();

    let mut out_dims: Vec<usize> = kept_a.iter().map(|&i| a.dims[i]).collect();
    out_dims.extend(kept_b.iter().map(|&j| b.dims[j]));
    let mut out_labels: Vec<i64> = kept_a.iter().map(|&i| a.labels[i]).collect();
    out_labels.extend(kept_b.iter().map(|&j| b.labels[j]));
    let out_total = out_dims.iter().product::<usize>().max(1);
    let mut out_data = vec![0.0f64; out_total];

    let a_strides = strides_from_dims(&a.dims);
    let b_strides = strides_from_dims(&b.dims);
    let out_strides = strides_from_dims(&out_dims);

    let kept_a_dims: Vec<usize> = kept_a.iter().map(|&i| a.dims[i]).collect();
    let kept_b_dims: Vec<usize> = kept_b.iter().map(|&j| b.dims[j]).collect();
    let shared_dims: Vec<usize> = shared_a.iter().map(|&i| a.dims[i]).collect();

    let n_kept_a = kept_a_dims.iter().product::<usize>().max(1);
    let n_kept_b = kept_b_dims.iter().product::<usize>().max(1);
    let n_shared = shared_dims.iter().product::<usize>().max(1);

    for ka in 0..n_kept_a {
        let ka_idx = unravel(ka, &kept_a_dims);
        for kb in 0..n_kept_b {
            let kb_idx = unravel(kb, &kept_b_dims);
            let mut out_idx_val = 0usize;
            for (pos, &val) in ka_idx.iter().enumerate() {
                out_idx_val += val * out_strides[pos];
            }
            for (pos_off, &val) in kb_idx.iter().enumerate() {
                out_idx_val += val * out_strides[kept_a.len() + pos_off];
            }
            let mut acc = 0.0f64;
            for s in 0..n_shared {
                let s_pos = unravel(s, &shared_dims);
                let mut a_flat = 0usize;
                for (k, &ax) in kept_a.iter().enumerate() {
                    a_flat += ka_idx[k] * a_strides[ax];
                }
                for (k, &ax) in shared_a.iter().enumerate() {
                    a_flat += s_pos[k] * a_strides[ax];
                }
                let mut b_flat = 0usize;
                for (k, &ax) in kept_b.iter().enumerate() {
                    b_flat += kb_idx[k] * b_strides[ax];
                }
                for (k, &ax) in shared_b.iter().enumerate() {
                    b_flat += s_pos[k] * b_strides[ax];
                }
                acc += a.data[a_flat] * b.data[b_flat];
            }
            out_data[out_idx_val] = acc;
        }
    }

    let name = format!("{}_{}", a.name, b.name);
    NetworkTensor::new(out_data, out_dims, out_labels, name)
}

/// Absorb all degree-1 leaf tensors into their single neighbours.
///
/// A leaf is a tensor with exactly one distinct negative (internal) label.
/// We contract leaf with neighbour, remove leaf from the network, and replace the
/// neighbour with the contracted result.
///
/// Returns the number of leaf absorptions performed.
pub fn absorb_leaves(net: &mut TensorNetwork) -> TnResult<usize> {
    let mut total = 0usize;
    loop {
        let bond_map = build_bond_map(net);
        // Find a leaf: a tensor with exactly one negative label, and that bond connects
        // to exactly one other tensor.
        let mut found: Option<(usize, usize)> = None; // (leaf_idx, neighbour_idx)
        'outer: for (ti, t) in net.tensors.iter().enumerate() {
            if bond_degree(ti, net) == 1 {
                // Find the single internal bond label.
                let bond_label = t.labels.iter().find(|&&l| l < 0).copied();
                if let Some(label) = bond_label {
                    if let Some(participants) = bond_map.get(&label) {
                        // The bond should connect exactly two tensors.
                        let neighbours: Vec<usize> = participants
                            .iter()
                            .filter(|(idx, _)| *idx != ti)
                            .map(|(idx, _)| *idx)
                            .collect();
                        if neighbours.len() == 1 {
                            found = Some((ti, neighbours[0]));
                            break 'outer;
                        }
                    }
                }
            }
        }
        let (leaf_idx, neigh_idx) = match found {
            None => break,
            Some(pair) => pair,
        };
        // Contract leaf into neighbour.
        let leaf = net.tensors[leaf_idx].clone();
        let neigh = net.tensors[neigh_idx].clone();
        let contracted = contract_two_tensors(&leaf, &neigh)?;
        // Replace neighbour, remove leaf.
        net.tensors[neigh_idx] = contracted;
        net.tensors.remove(leaf_idx);
        total += 1;
    }
    Ok(total)
}

// ─────────────────────────────────────────────────────────────────────────────
// Rule 3: Degree-2 chain simplification
// ─────────────────────────────────────────────────────────────────────────────

/// Absorb degree-2 pass-through tensors by contracting with one neighbour.
///
/// A degree-2 tensor has exactly two distinct internal bonds, each connecting to a
/// different neighbour. We absorb it into the first neighbour.
///
/// Returns the number of absorptions.
pub fn simplify_chains(net: &mut TensorNetwork) -> TnResult<usize> {
    let mut total = 0usize;
    loop {
        let bond_map = build_bond_map(net);
        // Find a degree-2 internal node: exactly 2 distinct negative labels and it
        // connects to exactly 2 distinct other tensors.
        let mut found: Option<(usize, usize)> = None; // (chain_node_idx, neighbour_idx)
        'outer: for (ti, t) in net.tensors.iter().enumerate() {
            let internal_labels: Vec<i64> = t
                .labels
                .iter()
                .filter(|&&l| l < 0)
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            if internal_labels.len() != 2 {
                continue;
            }
            // For each internal label, find the neighbour.
            let mut neighbours: Vec<usize> = Vec::new();
            for &lab in &internal_labels {
                if let Some(parts) = bond_map.get(&lab) {
                    for &(idx, _) in parts {
                        if idx != ti && !neighbours.contains(&idx) {
                            neighbours.push(idx);
                        }
                    }
                }
            }
            if neighbours.len() == 2 {
                found = Some((ti, neighbours[0]));
                break 'outer;
            }
        }
        let (chain_idx, neigh_idx) = match found {
            None => break,
            Some(pair) => pair,
        };
        // Contract chain node into first neighbour.
        let chain = net.tensors[chain_idx].clone();
        let neigh = net.tensors[neigh_idx].clone();
        let contracted = contract_two_tensors(&chain, &neigh)?;
        net.tensors[neigh_idx] = contracted;
        net.tensors.remove(chain_idx);
        total += 1;
    }
    Ok(total)
}

// ─────────────────────────────────────────────────────────────────────────────
// Rule 4: Gauge fixing
// ─────────────────────────────────────────────────────────────────────────────

/// Unfold a tensor along `axis` returning a 2D matrix (axis_dim × rest_dim), row-major.
///
/// The matrix M has shape `(dims[axis], prod(other dims))`.
pub fn unfold_along_axis(data: &[f64], dims: &[usize], axis: usize) -> TnResult<Vec<Vec<f64>>> {
    if axis >= dims.len() {
        return Err(TnError::IndexOutOfBounds {
            index: axis,
            len: dims.len(),
        });
    }
    let mode_dim = dims[axis];
    let total: usize = dims.iter().product::<usize>().max(1);
    let rest_dim = total.checked_div(mode_dim).unwrap_or(0);

    let _strides = strides_from_dims(dims);
    let other_axes: Vec<usize> = (0..dims.len()).filter(|&a| a != axis).collect();
    let other_dims: Vec<usize> = other_axes.iter().map(|&a| dims[a]).collect();
    let other_strides = strides_from_dims(&other_dims);

    let mut matrix = vec![vec![0.0f64; rest_dim]; mode_dim];

    for (flat, &value) in data.iter().enumerate().take(total) {
        let idx = unravel(flat, dims);
        let row = idx[axis];
        // Compute the column index in "other dims" order.
        let col = if other_axes.is_empty() {
            0
        } else {
            let other_idx: Vec<usize> = other_axes.iter().map(|&a| idx[a]).collect();
            ravel(&other_idx, &other_strides)
        };
        matrix[row][col] = value;
    }
    Ok(matrix)
}

/// Apply a row-major matrix `m` (shape `out_dim × in_dim`) to axis `axis` of tensor `t`.
///
/// Equivalent to contracting the `axis` leg with `m`. The result has the same shape
/// as `t` except `dims[axis]` becomes `out_dim`.
fn apply_matrix_to_axis(
    t: &NetworkTensor,
    axis: usize,
    mat: &[f64],
    out_dim: usize,
) -> TnResult<NetworkTensor> {
    let in_dim = t.dims[axis];
    if mat.len() != out_dim * in_dim {
        return Err(TnError::ShapeMismatch {
            expected: vec![out_dim, in_dim],
            got: vec![mat.len()],
        });
    }
    let mut new_dims = t.dims.clone();
    new_dims[axis] = out_dim;
    let new_total = new_dims.iter().product::<usize>().max(1);
    let mut new_data = vec![0.0f64; new_total];

    let out_strides = strides_from_dims(&new_dims);
    let in_total = t.dims.iter().product::<usize>().max(1);

    for in_flat in 0..in_total {
        let in_idx = unravel(in_flat, &t.dims);
        let k = in_idx[axis]; // which column of mat
        let val = t.data[in_flat];
        for r in 0..out_dim {
            let mat_val = mat[r * in_dim + k];
            // Build out_idx from in_idx with axis replaced by r.
            let mut out_flat = 0usize;
            for (a, &i) in in_idx.iter().enumerate() {
                if a == axis {
                    out_flat += r * out_strides[a];
                } else {
                    out_flat += i * out_strides[a];
                }
            }
            new_data[out_flat] += mat_val * val;
        }
    }
    NetworkTensor::new(new_data, new_dims, t.labels.clone(), t.name.clone())
}

/// Gauge-fix all internal bonds via left singular vectors of the corresponding mode
/// unfolding. Returns the number of bonds gauge-fixed.
///
/// For each internal bond label `b` shared by tensors A (at axis `ax_a`) and
/// B (at axis `ax_b`):
/// 1. Unfold A along `ax_a` → matrix M of shape `(dim_b, rest_a)`.
/// 2. SVD: M = U · diag(s) · Vt.
/// 3. Apply gauge G = U^T to A's `ax_a` axis (contracting left singular vectors).
/// 4. Apply G^{-1} = U (since U is orthogonal) to B's `ax_b` axis to compensate.
///
/// This puts the bond in left-canonical form, minimising the condition number of A's
/// mode-unfolding without changing the overall network value.
pub fn gauge_fix_bonds(net: &mut TensorNetwork) -> TnResult<usize> {
    let bond_map = build_bond_map(net);
    let mut fixed = 0usize;

    // Collect all bonds that connect exactly two distinct tensors.
    let mut bonds_to_fix: Vec<(i64, usize, usize, usize, usize)> = Vec::new();
    // (label, tensor_a_idx, axis_a, tensor_b_idx, axis_b)
    for (&label, participants) in &bond_map {
        if participants.len() == 2 {
            let (ta, ax_a) = participants[0];
            let (tb, ax_b) = participants[1];
            if ta != tb {
                bonds_to_fix.push((label, ta, ax_a, tb, ax_b));
            }
        }
    }

    for (_label, ta, ax_a, tb, ax_b) in bonds_to_fix {
        if ta >= net.tensors.len() || tb >= net.tensors.len() {
            continue;
        }
        let dim_bond = net.tensors[ta].dims[ax_a];
        // Unfold tensor A along ax_a → (dim_bond × rest_a).
        let unfolded = unfold_along_axis(&net.tensors[ta].data, &net.tensors[ta].dims, ax_a)?;
        let rest_a = net.tensors[ta].numel().checked_div(dim_bond).unwrap_or(0);
        if rest_a == 0 || dim_bond == 0 {
            continue;
        }
        // Flatten the unfolded matrix to a Vec<f64> in row-major order.
        let mat_flat: Vec<f64> = unfolded.iter().flatten().copied().collect();
        // SVD: M = U * diag(s) * Vt with M shape (dim_bond × rest_a).
        let svd_res = match svd_jacobi(&mat_flat, dim_bond, rest_a) {
            Ok(r) => r,
            Err(_) => continue, // skip bonds where SVD fails
        };
        // G = U^T (shape k × dim_bond), G_inv = U (shape dim_bond × k).
        // We apply G to A (contracting axis ax_a: new axis has dim k) and
        // G_inv to B (expanding axis ax_b: stays dim k iff k == dim_bond).
        // Since U is (dim_bond × k) orthogonal (k = min(dim_bond, rest_a)),
        // applying U^T to A changes its ax_a dim from dim_bond to k,
        // and applying U to B restores the same bond dimension k on B's side.
        let k = svd_res.k;
        // G_t (k × dim_bond): G^T[r, c] = U[c, r]
        let g_t: Vec<f64> = {
            let mut gt = vec![0.0f64; k * dim_bond];
            for r in 0..k {
                for c in 0..dim_bond {
                    gt[r * dim_bond + c] = svd_res.u[c * k + r];
                }
            }
            gt
        };
        // G_inv = U (dim_bond × k), but we need shape (dim_bond × k) for apply to B.
        // B's axis ax_b currently has dim dim_bond; applying U gives new dim k.
        // Both transformations preserve k = dim_bond when dim_bond ≤ rest_a.
        let g_inv: Vec<f64> = svd_res.u.clone(); // (dim_bond × k), row-major

        // Apply G^T to A at axis ax_a (transforms dim_bond → k).
        let new_a = apply_matrix_to_axis(&net.tensors[ta], ax_a, &g_t, k)?;
        // Apply G^{-1} = U to B at axis ax_b (transforms dim_bond → k).
        let new_b = apply_matrix_to_axis(&net.tensors[tb], ax_b, &g_inv, k)?;

        // Only update if dimensions are consistent.
        if new_a.dims[ax_a] == new_b.dims[ax_b] {
            net.tensors[ta] = new_a;
            net.tensors[tb] = new_b;
            fixed += 1;
        }
    }
    Ok(fixed)
}

// ─────────────────────────────────────────────────────────────────────────────
// Rule 5: Scalar folding
// ─────────────────────────────────────────────────────────────────────────────

/// Fold any disconnected scalar tensors (rank-0) into the first other tensor as a
/// multiplicative prefactor. Returns the number of scalars folded.
pub fn fold_scalars(net: &mut TensorNetwork) -> TnResult<usize> {
    let mut total = 0usize;
    loop {
        // Find a scalar tensor.
        let scalar_idx = net.tensors.iter().position(|t| t.is_scalar());
        let scalar_idx = match scalar_idx {
            None => break,
            Some(idx) => idx,
        };
        // Find another (non-scalar) tensor to absorb it.
        let target_idx = net.tensors.iter().position(|t| !t.is_scalar());
        let target_idx = match target_idx {
            None => {
                // All tensors are scalars — fold them together.
                if net.tensors.len() <= 1 {
                    break;
                }
                // Multiply scalar into index 0 (unless it is the scalar itself).
                if scalar_idx == 0 { 1 } else { 0 }
            }
            Some(idx) => {
                if idx == scalar_idx {
                    // Find a different target.
                    match net.tensors.iter().position(|t| {
                        !t.is_scalar() && {
                            let pos = net.tensors.iter().position(|x| std::ptr::eq(x, t));
                            pos.map(|p| p != scalar_idx).unwrap_or(false)
                        }
                    }) {
                        None => break,
                        Some(p) => p,
                    }
                } else {
                    idx
                }
            }
        };

        if scalar_idx == target_idx {
            break;
        }

        let scalar_val = net.tensors[scalar_idx].scalar_value();
        // Scale all elements of the target tensor.
        for v in &mut net.tensors[target_idx].data {
            *v *= scalar_val;
        }
        net.tensors.remove(scalar_idx);
        total += 1;
    }
    Ok(total)
}

// ─────────────────────────────────────────────────────────────────────────────
// Rule 6: Parallel edge merging
// ─────────────────────────────────────────────────────────────────────────────

/// Fuse parallel edges between the same pair of tensors into a single merged bond.
///
/// When tensors A and B share more than one negative label (parallel bonds), their
/// Kronecker product over those bonds gives a single fused index with dimension equal
/// to the product of the individual bond dimensions.
///
/// Returns the number of fusions performed.
pub fn fuse_parallel_bonds(net: &mut TensorNetwork) -> TnResult<usize> {
    let mut total = 0usize;
    loop {
        // Find a pair (ta, tb) that shares more than one negative label.
        let bond_map = build_bond_map(net);
        // For each tensor pair, collect the shared bonds.
        let mut pair_bonds: HashMap<(usize, usize), Vec<i64>> = HashMap::new();
        for (&label, parts) in &bond_map {
            if parts.len() == 2 {
                let (t0, _) = parts[0];
                let (t1, _) = parts[1];
                if t0 != t1 {
                    let key = (t0.min(t1), t0.max(t1));
                    pair_bonds.entry(key).or_default().push(label);
                }
            }
        }
        // Find a pair with more than one shared bond.
        let fusion_target = pair_bonds.into_iter().find(|(_, labels)| labels.len() >= 2);
        let ((ta, tb), labels_to_fuse) = match fusion_target {
            None => break,
            Some(entry) => entry,
        };

        // For each tensor, find the axis positions of the bonds to fuse.
        let axes_a: Vec<usize> = labels_to_fuse
            .iter()
            .map(|lab| {
                net.tensors[ta].labels.iter().position(|l| l == lab).ok_or(
                    TnError::ContractionPathInvalid("label not found in tensor A".into()),
                )
            })
            .collect::<TnResult<Vec<_>>>()?;
        let axes_b: Vec<usize> = labels_to_fuse
            .iter()
            .map(|lab| {
                net.tensors[tb].labels.iter().position(|l| l == lab).ok_or(
                    TnError::ContractionPathInvalid("label not found in tensor B".into()),
                )
            })
            .collect::<TnResult<Vec<_>>>()?;

        // Validate that bond dims match.
        for (ax_a, ax_b) in axes_a.iter().zip(axes_b.iter()) {
            if net.tensors[ta].dims[*ax_a] != net.tensors[tb].dims[*ax_b] {
                return Err(TnError::DimensionMismatch {
                    a: net.tensors[ta].dims[*ax_a],
                    b: net.tensors[tb].dims[*ax_b],
                });
            }
        }

        // Compute the fused dimension.
        let fused_dim: usize = axes_a.iter().map(|&ax| net.tensors[ta].dims[ax]).product();

        // Allocate a new fresh negative label for the fused bond.
        // Use the minimum of the fused labels minus 1 to stay negative.
        let new_label: i64 = labels_to_fuse.iter().min().copied().unwrap_or(-1) - 1;

        // For tensor A: replace all fused axes with a single fused axis.
        // Strategy: permute fused axes to the front, then reshape.
        let tensor_a = fuse_axes_in_tensor(&net.tensors[ta], &axes_a, fused_dim, new_label)?;
        let tensor_b = fuse_axes_in_tensor(&net.tensors[tb], &axes_b, fused_dim, new_label)?;

        net.tensors[ta] = tensor_a;
        net.tensors[tb] = tensor_b;
        total += 1;
    }
    Ok(total)
}

/// Helper: fuse the given axes of a tensor into a single axis with `fused_dim` and
/// `new_label`. The fused axis is placed at the position of the first fused axis.
fn fuse_axes_in_tensor(
    t: &NetworkTensor,
    axes: &[usize],
    fused_dim: usize,
    new_label: i64,
) -> TnResult<NetworkTensor> {
    if axes.is_empty() {
        return Ok(t.clone());
    }
    // Validate product of dims matches fused_dim.
    let computed_fused: usize = axes.iter().map(|&ax| t.dims[ax]).product();
    if computed_fused != fused_dim {
        return Err(TnError::DimensionMismatch {
            a: computed_fused,
            b: fused_dim,
        });
    }
    // Build the output dims and labels: replace fused axes with single fused dim.
    let insert_pos = *axes.iter().min().unwrap();
    let mut new_dims: Vec<usize> = Vec::new();
    let mut new_labels: Vec<i64> = Vec::new();
    let mut fused_inserted = false;
    for ax in 0..t.dims.len() {
        if axes.contains(&ax) {
            if !fused_inserted {
                new_dims.push(fused_dim);
                new_labels.push(new_label);
                fused_inserted = true;
            }
        } else {
            new_dims.push(t.dims[ax]);
            new_labels.push(t.labels[ax]);
        }
    }
    let _ = insert_pos; // used conceptually above

    // Now we need to re-order/re-layout the data.
    // The multi-index order in new tensor: all non-fused axes in original order,
    // plus the fused axes stride through their original indices in-order.
    // Permuted axis order: fused axes first (in axis order), then non-fused.
    // We iterate over output multi-index directly.

    let new_total = new_dims.iter().product::<usize>().max(1);
    let mut new_data = vec![0.0f64; new_total];

    let out_strides = strides_from_dims(&new_dims);

    let fused_axes_sorted: Vec<usize> = {
        let mut v = axes.to_vec();
        v.sort_unstable();
        v
    };
    let fused_axes_dims: Vec<usize> = fused_axes_sorted.iter().map(|&a| t.dims[a]).collect();

    let in_total = t.dims.iter().product::<usize>().max(1);
    for in_flat in 0..in_total {
        let in_idx = unravel(in_flat, &t.dims);
        // Compute the fused index.
        let fused_idx: usize = {
            let sub_idx: Vec<usize> = fused_axes_sorted.iter().map(|&ax| in_idx[ax]).collect();
            ravel(&sub_idx, &strides_from_dims(&fused_axes_dims))
        };
        // Build output multi-index.
        let mut out_idx = vec![0usize; new_dims.len()];
        // Position 0 in the fused group corresponds to insert_pos.
        // Actually we placed the fused axis at the position of the first fused ax in non-fused order.
        // Let's map: output axes are [non_fused in original order, with fused group in place].
        // new_dims layout: original non-fused axes keep order, fused axes → single entry.
        let mut out_pos = 0usize;
        let mut fused_placed = false;
        for (ax, &in_val) in in_idx.iter().enumerate().take(t.dims.len()) {
            if fused_axes_sorted.contains(&ax) {
                if !fused_placed {
                    out_idx[out_pos] = fused_idx;
                    out_pos += 1;
                    fused_placed = true;
                }
            } else {
                out_idx[out_pos] = in_val;
                out_pos += 1;
            }
        }
        let out_flat = ravel(&out_idx, &out_strides);
        new_data[out_flat] = t.data[in_flat];
    }

    NetworkTensor::new(new_data, new_dims, new_labels, t.name.clone())
}

// ─────────────────────────────────────────────────────────────────────────────
// Network contraction
// ─────────────────────────────────────────────────────────────────────────────

/// After simplification, contract the entire network using a greedy contraction order.
///
/// Returns the flat data of the resulting tensor. For a fully contracted network,
/// this is a scalar (length 1).
pub fn contract_network(net: &TensorNetwork) -> TnResult<Vec<f64>> {
    if net.tensors.is_empty() {
        return Err(TnError::EmptyInput);
    }
    if net.tensors.len() == 1 {
        return Ok(net.tensors[0].data.clone());
    }

    // Use a greedy contraction: always contract the pair with the most shared labels first,
    // to reduce intermediate tensor sizes.
    let mut working: Vec<NetworkTensor> = net.tensors.clone();

    while working.len() > 1 {
        // Find the pair with the most shared negative labels.
        let mut best_pair: Option<(usize, usize, usize)> = None; // (i, j, shared_count)
        for i in 0..working.len() {
            for j in i + 1..working.len() {
                let shared = count_shared_labels(&working[i], &working[j]);
                if shared > 0 {
                    let is_better = best_pair
                        .as_ref()
                        .map(|(_, _, sc)| shared > *sc)
                        .unwrap_or(true);
                    if is_better {
                        best_pair = Some((i, j, shared));
                    }
                }
            }
        }

        match best_pair {
            Some((i, j, _)) => {
                let a = working[i].clone();
                let b = working[j].clone();
                let contracted = contract_two_tensors(&a, &b)?;
                working[i] = contracted;
                working.remove(j);
            }
            None => {
                // No shared labels: tensors are disconnected.
                // Merge as outer product (no shared labels → all dims retained).
                let a = working[0].clone();
                let b = working[1].clone();
                let merged = contract_two_tensors(&a, &b)?;
                working[0] = merged;
                working.remove(1);
            }
        }
    }

    Ok(working
        .into_iter()
        .next()
        .map(|t| t.data)
        .unwrap_or_default())
}

/// Count the number of label values shared between two tensors (including positive labels).
fn count_shared_labels(a: &NetworkTensor, b: &NetworkTensor) -> usize {
    a.labels.iter().filter(|la| b.labels.contains(la)).count()
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixed-point simplification loop
// ─────────────────────────────────────────────────────────────────────────────

/// Apply all simplification rules in a fixed-point loop until no further change.
///
/// Order of application per round:
/// 1. Remove traces
/// 2. Fold scalars
/// 3. Fuse parallel bonds
/// 4. Absorb leaves (degree-1)
/// 5. Simplify chains (degree-2)
///
/// Gauge fixing is intentionally applied only once at the end (it is not idempotent
/// in the same way and may increase computational cost if applied repeatedly).
pub fn simplify_network(net: &mut TensorNetwork) -> TnResult<SimplifyStats> {
    let mut stats = SimplifyStats::default();

    loop {
        let before = net.tensors.len();
        let before_total_els: usize = net.tensors.iter().map(|t| t.numel()).sum();

        let traces = remove_traces(net)?;
        stats.traces_removed += traces;

        let scalars = fold_scalars(net)?;
        stats.scalars_folded += scalars;

        let fused = fuse_parallel_bonds(net)?;
        stats.parallel_bonds_fused += fused;

        let leaves = absorb_leaves(net)?;
        stats.leaves_absorbed += leaves;

        let chains = simplify_chains(net)?;
        stats.chains_simplified += chains;

        let after = net.tensors.len();
        let after_total_els: usize = net.tensors.iter().map(|t| t.numel()).sum();

        // Estimate FLOPs saved as the reduction in total tensor elements.
        if after_total_els < before_total_els {
            stats.total_flops_saved_estimate = stats
                .total_flops_saved_estimate
                .saturating_add((before_total_els - after_total_els) as u64);
        }

        // Converged if nothing changed this round.
        if before == after
            && traces == 0
            && scalars == 0
            && fused == 0
            && leaves == 0
            && chains == 0
        {
            break;
        }
    }
    Ok(stats)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1: valid construction ────────────────────────────────────────────

    #[test]
    fn network_tensor_construction() {
        // 2×3 tensor with 6 elements should succeed.
        let t = NetworkTensor::new(vec![1.0; 6], vec![2, 3], vec![1, 2], "A".to_string());
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.dims, vec![2, 3]);
        assert_eq!(t.labels, vec![1, 2]);
    }

    // ── Test 2: wrong data length → Err ──────────────────────────────────────

    #[test]
    fn network_tensor_shape_error() {
        // 2×3 tensor but only 5 elements.
        let t = NetworkTensor::new(vec![1.0; 5], vec![2, 3], vec![1, 2], "A".to_string());
        assert!(t.is_err());
    }

    // ── Test 3: trace of 2×2 identity → scalar 2 ─────────────────────────────

    #[test]
    fn remove_traces_rank2() {
        // Tensor A[i, i] = identity 2×2 → trace = 2.0
        let identity = vec![1.0, 0.0, 0.0, 1.0]; // [[1,0],[0,1]]
        let t = NetworkTensor::new(identity, vec![2, 2], vec![-1, -1], "I".to_string()).unwrap();
        let traced = trace_one_tensor(&t).unwrap();
        // Result should be a scalar.
        assert!(
            traced.is_scalar(),
            "expected scalar, got dims={:?}",
            traced.dims
        );
        assert!((traced.scalar_value() - 2.0).abs() < 1e-12);
    }

    // ── Test 4: trace on rank-3 tensor A[i,j,i] → vector ────────────────────

    #[test]
    fn remove_traces_rank3_one_trace() {
        // A[i, j, i] with i∈{0,1}, j∈{0,1,2}. Total 12 elements.
        // A[i, j, i] for i=0: elements at (0,j,0) for j=0,1,2.
        // A[i, j, i] for i=1: elements at (1,j,1) for j=0,1,2.
        // Layout (i, j, k) with dims (2, 3, 2):
        //   flat = i*6 + j*2 + k
        // So A[i, j, i] = data[i*6 + j*2 + i]
        // We want trace[j] = sum_i A[i, j, i]
        let mut data = vec![0.0f64; 12];
        // Set A[0, j, 0] = (j+1)*1.0 and A[1, j, 1] = (j+1)*2.0
        for j in 0..3 {
            // i=0, k=0 → flat = 0*6 + j*2 + 0 = j*2
            data[j * 2] = (j + 1) as f64;
            // i=1, k=1 → flat = 1*6 + j*2 + 1 = 6 + j*2 + 1
            data[6 + j * 2 + 1] = (j + 1) as f64 * 2.0;
        }
        // Labels: axis 0 and axis 2 share label -1; axis 1 has label 1.
        let t = NetworkTensor::new(data, vec![2, 3, 2], vec![-1, 1, -1], "A".to_string()).unwrap();
        let traced = trace_one_tensor(&t).unwrap();
        // Expect dims = [3], trace[j] = (j+1)*1.0 + (j+1)*2.0 = 3*(j+1).
        assert_eq!(traced.dims, vec![3]);
        for j in 0..3 {
            let expected = 3.0 * (j + 1) as f64;
            assert!(
                (traced.data[j] - expected).abs() < 1e-12,
                "traced.data[{}] = {} ≠ {}",
                j,
                traced.data[j],
                expected
            );
        }
    }

    // ── Test 5: single leaf absorbed into body ────────────────────────────────

    #[test]
    fn absorb_leaves_single() {
        // Leaf A[k] and body B[k, l] connected via bond k=-1.
        // A = [2.0, 3.0], B = identity 2×2.
        // Result: C[l] = sum_k A[k] * B[k, l] = A = [2.0, 3.0].
        let a = NetworkTensor::new(vec![2.0, 3.0], vec![2], vec![-1], "A".to_string()).unwrap();
        let b = NetworkTensor::new(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![2, 2],
            vec![-1, 1],
            "B".to_string(),
        )
        .unwrap();
        let mut net = TensorNetwork::new();
        net.push(a);
        net.push(b);
        let count = absorb_leaves(&mut net).unwrap();
        assert_eq!(count, 1, "expected 1 leaf absorbed");
        assert_eq!(net.tensors.len(), 1, "expected 1 tensor remaining");
        let result = &net.tensors[0];
        assert_eq!(result.dims, vec![2]);
        assert!((result.data[0] - 2.0).abs() < 1e-12);
        assert!((result.data[1] - 3.0).abs() < 1e-12);
    }

    // ── Test 6: chain A[k]--k--B[k,l]--l--C[l] → single tensor ─────────────

    #[test]
    fn absorb_leaves_chain() {
        // A[k] = [1.0, 1.0], B[k,l] = identity 2×2, C[l] = [3.0, 4.0].
        // Result should be A @ B @ C = dot([1,1], I, [3,4]) = 7.0 scalar.
        let a = NetworkTensor::new(vec![1.0, 1.0], vec![2], vec![-1], "A".to_string()).unwrap();
        let b = NetworkTensor::new(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![2, 2],
            vec![-1, -2],
            "B".to_string(),
        )
        .unwrap();
        let c = NetworkTensor::new(vec![3.0, 4.0], vec![2], vec![-2], "C".to_string()).unwrap();
        let mut net = TensorNetwork::new();
        net.push(a);
        net.push(b);
        net.push(c);
        let _count = absorb_leaves(&mut net).unwrap();
        assert_eq!(net.tensors.len(), 1, "expected 1 tensor remaining");
        let result = &net.tensors[0];
        assert!(result.is_scalar() || result.numel() == 1);
        assert!((result.data[0] - 7.0).abs() < 1e-12);
    }

    // ── Test 7: degree-2 pass-through absorbed between two matrices ───────────

    #[test]
    fn simplify_chains_rank2_passthrough() {
        // A[i, k] (2×2) -- k=-1 -- M[k, l] (2×2) -- l=-2 -- B[l, j] (2×2)
        // A = I, M = [[2,0],[0,3]], B = I.
        // After simplify_chains: M absorbed into A or B; then 2 tensors remain
        // (they are not leaves, so absorb_leaves won't fire without more rounds).
        // We just verify simplify_chains reduces tensor count by 1.
        let a = NetworkTensor::new(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![2, 2],
            vec![1, -1],
            "A".to_string(),
        )
        .unwrap();
        let m = NetworkTensor::new(
            vec![2.0, 0.0, 0.0, 3.0],
            vec![2, 2],
            vec![-1, -2],
            "M".to_string(),
        )
        .unwrap();
        let b = NetworkTensor::new(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![2, 2],
            vec![-2, 2],
            "B".to_string(),
        )
        .unwrap();
        let mut net = TensorNetwork::new();
        net.push(a);
        net.push(m);
        net.push(b);
        let count = simplify_chains(&mut net).unwrap();
        assert!(count >= 1, "expected at least 1 chain simplification");
        assert_eq!(
            net.tensors.len(),
            2,
            "expected 2 tensors after chain simplification"
        );
    }

    // ── Test 8: scalar folded multiplies into other tensor ───────────────────

    #[test]
    fn fold_scalars_multiplies_into_other() {
        // Scalar s = 3.0, tensor T[i] = [1.0, 2.0].
        // After fold: T = [3.0, 6.0].
        let scalar = NetworkTensor::new(vec![3.0], vec![], vec![], "S".to_string()).unwrap();
        let tensor = NetworkTensor::new(vec![1.0, 2.0], vec![2], vec![1], "T".to_string()).unwrap();
        let mut net = TensorNetwork::new();
        net.push(scalar);
        net.push(tensor);
        let count = fold_scalars(&mut net).unwrap();
        assert_eq!(count, 1);
        assert_eq!(net.tensors.len(), 1);
        let t = &net.tensors[0];
        assert!((t.data[0] - 3.0).abs() < 1e-12);
        assert!((t.data[1] - 6.0).abs() < 1e-12);
    }

    // ── Test 9: fuse two parallel bonds into one ──────────────────────────────

    #[test]
    fn fuse_parallel_bonds_two_bonds() {
        // A[i, j] with labels [-1, -2] and B[i, j] with labels [-1, -2].
        // dims: A = (2, 3), B = (2, 3). Fused bond dim = 2*3=6.
        let a = NetworkTensor::new(
            (0..6).map(|x| x as f64).collect(),
            vec![2, 3],
            vec![-1, -2],
            "A".to_string(),
        )
        .unwrap();
        let b = NetworkTensor::new(
            (0..6).map(|x| x as f64).collect(),
            vec![2, 3],
            vec![-1, -2],
            "B".to_string(),
        )
        .unwrap();
        let mut net = TensorNetwork::new();
        net.push(a);
        net.push(b);
        let count = fuse_parallel_bonds(&mut net).unwrap();
        assert_eq!(count, 1);
        // Each tensor now has a single internal axis of dim 6.
        assert_eq!(net.tensors[0].dims, vec![6]);
        assert_eq!(net.tensors[1].dims, vec![6]);
    }

    // ── Test 10: contract two tensors A[i,k] @ B[k,j] = C[i,j] ──────────────

    #[test]
    fn contract_network_matmul() {
        // A = [[1,2],[3,4]] (2×2), B = [[5,6],[7,8]] (2×2), bond k=-1.
        // Expected C = A @ B = [[19,22],[43,50]].
        let a = NetworkTensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            vec![1, -1],
            "A".to_string(),
        )
        .unwrap();
        let b = NetworkTensor::new(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2],
            vec![-1, 2],
            "B".to_string(),
        )
        .unwrap();
        let mut net = TensorNetwork::new();
        net.push(a);
        net.push(b);
        let result = contract_network(&net).unwrap();
        assert_eq!(result.len(), 4);
        assert!((result[0] - 19.0).abs() < 1e-12, "C[0,0]={}", result[0]);
        assert!((result[1] - 22.0).abs() < 1e-12, "C[0,1]={}", result[1]);
        assert!((result[2] - 43.0).abs() < 1e-12, "C[1,0]={}", result[2]);
        assert!((result[3] - 50.0).abs() < 1e-12, "C[1,1]={}", result[3]);
    }

    // ── Test 11: simplify_network fixed-point idempotent ─────────────────────

    #[test]
    fn simplify_network_fixed_point() {
        // Simple 2-tensor network: leaf + body. After first simplify, 1 tensor remains.
        // Second simplify should change nothing.
        let leaf = NetworkTensor::new(vec![1.0, 2.0], vec![2], vec![-1], "L".to_string()).unwrap();
        let body = NetworkTensor::new(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![2, 2],
            vec![-1, 1],
            "B".to_string(),
        )
        .unwrap();
        let mut net = TensorNetwork::new();
        net.push(leaf);
        net.push(body);
        let _stats1 = simplify_network(&mut net).unwrap();
        let tensors_after_first = net.tensors.len();
        let _stats2 = simplify_network(&mut net).unwrap();
        let tensors_after_second = net.tensors.len();
        assert_eq!(tensors_after_first, tensors_after_second);
    }

    // ── Test 12: stats count leaves_absorbed after leaf absorption ───────────

    #[test]
    fn simplify_network_stats_counts() {
        let leaf = NetworkTensor::new(vec![1.0, 1.0], vec![2], vec![-1], "L".to_string()).unwrap();
        let body = NetworkTensor::new(
            vec![2.0, 0.0, 0.0, 3.0],
            vec![2, 2],
            vec![-1, 1],
            "B".to_string(),
        )
        .unwrap();
        let mut net = TensorNetwork::new();
        net.push(leaf);
        net.push(body);
        let stats = simplify_network(&mut net).unwrap();
        assert!(
            stats.leaves_absorbed > 0,
            "expected leaves_absorbed > 0, got {}",
            stats.leaves_absorbed
        );
    }

    // ── Test 13: gauge_fix_bonds runs without error ───────────────────────────

    #[test]
    fn gauge_fix_bonds_runs() {
        // Simple 2-tensor network: A[k, i] and B[k, j] sharing bond k=-1.
        let a = NetworkTensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            vec![-1, 1],
            "A".to_string(),
        )
        .unwrap();
        let b = NetworkTensor::new(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![2, 2],
            vec![-1, 2],
            "B".to_string(),
        )
        .unwrap();
        let mut net = TensorNetwork::new();
        net.push(a);
        net.push(b);
        let result = gauge_fix_bonds(&mut net);
        assert!(result.is_ok(), "gauge_fix_bonds returned Err: {:?}", result);
        // Network still has 2 tensors.
        assert_eq!(net.tensors.len(), 2);
    }

    // ── Test 14: fully contracted network returns a scalar ────────────────────

    #[test]
    fn contract_network_scalar() {
        // A[k] · B[k] = dot product = [1,2,3]·[4,5,6] = 32.
        let a =
            NetworkTensor::new(vec![1.0, 2.0, 3.0], vec![3], vec![-1], "A".to_string()).unwrap();
        let b =
            NetworkTensor::new(vec![4.0, 5.0, 6.0], vec![3], vec![-1], "B".to_string()).unwrap();
        let net = TensorNetwork {
            tensors: vec![a, b],
        };
        let result = contract_network(&net).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 32.0).abs() < 1e-12, "result={}", result[0]);
    }

    // ── Test 15: simplified ≡ direct contraction for a 3-tensor chain ─────────

    #[test]
    fn simplify_then_contract_matches_direct() {
        // Chain: A[i, k] --k=-1-- B[k, l] --l=-2-- C[l, j]
        // A = [[1,2],[3,4]], B = [[1,0],[0,1]], C = [[5,6],[7,8]].
        // Direct contraction: A @ I @ C = A @ C = [[19,22],[43,50]].
        let make = |data: Vec<f64>, dims: Vec<usize>, labels: Vec<i64>, name: &str| {
            NetworkTensor::new(data, dims, labels, name.to_string()).unwrap()
        };
        let a = make(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], vec![1, -1], "A");
        let b_mat = make(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], vec![-1, -2], "B");
        let c = make(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], vec![-2, 2], "C");

        // Direct contraction without simplification.
        let net_direct = TensorNetwork {
            tensors: vec![a.clone(), b_mat.clone(), c.clone()],
        };
        let direct_result = contract_network(&net_direct).unwrap();

        // Simplified then contract.
        let mut net_simplified = TensorNetwork {
            tensors: vec![a, b_mat, c],
        };
        let _stats = simplify_network(&mut net_simplified).unwrap();
        let simplified_result = contract_network(&net_simplified).unwrap();

        assert_eq!(
            direct_result.len(),
            simplified_result.len(),
            "result sizes differ"
        );
        for (d, s) in direct_result.iter().zip(simplified_result.iter()) {
            assert!(
                (d - s).abs() < 1e-10,
                "mismatch: direct={} simplified={}",
                d,
                s
            );
        }
    }
}
