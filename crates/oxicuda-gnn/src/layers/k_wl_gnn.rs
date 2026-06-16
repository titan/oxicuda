//! k-WL expressive GNN (Maron et al., "Invariant and Equivariant Graph
//! Networks", NeurIPS 2019), specialised to `k = 2`.
//!
//! A 2-WL / order-2 invariant graph network operates on **node-pair**
//! representations: a third-order tensor `X ∈ ℝ^{n × n × d}` assigning a feature
//! vector to every ordered pair `(i, j)`. Its layers are the *permutation-
//! equivariant linear maps* on order-2 tensors, whose space is spanned by the
//! `Bell(4) = 15` basis operations (one per partition of the four index slots
//! `{i, j, i′, j′}`). A pointwise nonlinearity follows each equivariant linear
//! map; stacking these yields a network strictly more expressive than the
//! 1-WL message-passing GNNs (it matches the 2-WL test).
//!
//! The 15 equivariant basis operations on a single channel `X ∈ ℝ^{n × n}`
//! (with `diag[i] = X[i,i]`, `row[i] = Σ_k X[i,k]`, `col[j] = Σ_k X[k,j]`,
//! `trace = Σ_k X[k,k]`, `total = Σ_{a,b} X[a,b]`) are:
//!
//! | op | output `Y[i,j]` | description |
//! |----|-----------------|-------------|
//! | 1  | `X[i,j]`        | identity |
//! | 2  | `X[j,i]`        | transpose |
//! | 3  | `δij·diag[i]`   | diagonal → diagonal |
//! | 4  | `δij·row[i]`    | row-sum → diagonal |
//! | 5  | `δij·col[i]`    | col-sum → diagonal |
//! | 6  | `δij·trace`     | trace → diagonal |
//! | 7  | `δij·total`     | total → diagonal |
//! | 8  | `diag[i]`       | diagonal → rows |
//! | 9  | `diag[j]`       | diagonal → cols |
//! | 10 | `row[i]`        | row-sum → rows |
//! | 11 | `col[j]`        | col-sum → cols |
//! | 12 | `row[j]`        | row-sum → cols |
//! | 13 | `col[i]`        | col-sum → rows |
//! | 14 | `trace`         | trace broadcast |
//! | 15 | `total`         | total broadcast |
//!
//! Every operation commutes with the simultaneous row/column permutation of the
//! index set, so the whole layer is permutation-equivariant; a permutation of
//! the input node labels permutes the output node-pair tensor identically. A
//! permutation-invariant pooling over all pairs produces a graph-level readout.

use crate::error::{GnnError, GnnResult};
use crate::handle::LcgRng;

/// The 15 permutation-equivariant linear basis operations on order-2 tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairOp {
    /// `Y[i,j] = X[i,j]`.
    Identity,
    /// `Y[i,j] = X[j,i]`.
    Transpose,
    /// `Y[i,j] = δij · X[i,i]`.
    DiagToDiag,
    /// `Y[i,j] = δij · Σ_k X[i,k]`.
    RowSumToDiag,
    /// `Y[i,j] = δij · Σ_k X[k,i]`.
    ColSumToDiag,
    /// `Y[i,j] = δij · Σ_k X[k,k]`.
    TraceToDiag,
    /// `Y[i,j] = δij · Σ_{a,b} X[a,b]`.
    TotalToDiag,
    /// `Y[i,j] = X[i,i]` (diagonal broadcast along rows).
    DiagToRows,
    /// `Y[i,j] = X[j,j]` (diagonal broadcast along columns).
    DiagToCols,
    /// `Y[i,j] = Σ_k X[i,k]` (row-sum broadcast along rows).
    RowSumToRows,
    /// `Y[i,j] = Σ_k X[k,j]` (col-sum broadcast along columns).
    ColSumToCols,
    /// `Y[i,j] = Σ_k X[j,k]` (row-sum of `j` broadcast along columns).
    RowSumToCols,
    /// `Y[i,j] = Σ_k X[k,i]` (col-sum of `i` broadcast along rows).
    ColSumToRows,
    /// `Y[i,j] = Σ_k X[k,k]` (trace broadcast everywhere).
    TraceBroadcast,
    /// `Y[i,j] = Σ_{a,b} X[a,b]` (total sum broadcast everywhere).
    TotalBroadcast,
}

impl PairOp {
    /// All 15 basis operations in canonical order.
    pub const ALL: [PairOp; 15] = [
        PairOp::Identity,
        PairOp::Transpose,
        PairOp::DiagToDiag,
        PairOp::RowSumToDiag,
        PairOp::ColSumToDiag,
        PairOp::TraceToDiag,
        PairOp::TotalToDiag,
        PairOp::DiagToRows,
        PairOp::DiagToCols,
        PairOp::RowSumToRows,
        PairOp::ColSumToCols,
        PairOp::RowSumToCols,
        PairOp::ColSumToRows,
        PairOp::TraceBroadcast,
        PairOp::TotalBroadcast,
    ];
}

/// Per-channel reductions of an `n × n` matrix used by every basis operation.
struct Reductions {
    diag: Vec<f32>,
    row: Vec<f32>,
    col: Vec<f32>,
    trace: f32,
    total: f32,
}

impl Reductions {
    /// Compute the reductions for channel `c` of `x` (`[n × n × dim]`).
    fn compute(x: &[f32], n: usize, dim: usize, c: usize) -> Self {
        let mut diag = vec![0.0_f32; n];
        let mut row = vec![0.0_f32; n];
        let mut col = vec![0.0_f32; n];
        let mut trace = 0.0_f32;
        let mut total = 0.0_f32;
        for i in 0..n {
            for j in 0..n {
                let v = x[(i * n + j) * dim + c];
                row[i] += v;
                col[j] += v;
                total += v;
                if i == j {
                    diag[i] = v;
                    trace += v;
                }
            }
        }
        Self {
            diag,
            row,
            col,
            trace,
            total,
        }
    }

    /// Value of basis operation `op` at output entry `(i, j)`.
    #[inline]
    fn op_value(
        &self,
        op: PairOp,
        x: &[f32],
        n: usize,
        dim: usize,
        c: usize,
        i: usize,
        j: usize,
    ) -> f32 {
        let on_diag = i == j;
        match op {
            PairOp::Identity => x[(i * n + j) * dim + c],
            PairOp::Transpose => x[(j * n + i) * dim + c],
            PairOp::DiagToDiag => {
                if on_diag {
                    self.diag[i]
                } else {
                    0.0
                }
            }
            PairOp::RowSumToDiag => {
                if on_diag {
                    self.row[i]
                } else {
                    0.0
                }
            }
            PairOp::ColSumToDiag => {
                if on_diag {
                    self.col[i]
                } else {
                    0.0
                }
            }
            PairOp::TraceToDiag => {
                if on_diag {
                    self.trace
                } else {
                    0.0
                }
            }
            PairOp::TotalToDiag => {
                if on_diag {
                    self.total
                } else {
                    0.0
                }
            }
            PairOp::DiagToRows => self.diag[i],
            PairOp::DiagToCols => self.diag[j],
            PairOp::RowSumToRows => self.row[i],
            PairOp::ColSumToCols => self.col[j],
            PairOp::RowSumToCols => self.row[j],
            PairOp::ColSumToRows => self.col[i],
            PairOp::TraceBroadcast => self.trace,
            PairOp::TotalBroadcast => self.total,
        }
    }
}

/// Apply a single equivariant basis operation to every channel of a pair tensor.
///
/// `x` is `[n × n × dim]` (row-major, entry `(i,j,c)` at `(i*n+j)*dim + c`);
/// the result has the same shape.
///
/// # Errors
///
/// * [`GnnError::EmptyGraph`] if `n == 0`.
/// * [`GnnError::InvalidLayerConfig`] if `dim == 0`.
/// * [`GnnError::DimensionMismatch`] if `x.len() != n * n * dim`.
pub fn apply_pair_op(op: PairOp, x: &[f32], n: usize, dim: usize) -> GnnResult<Vec<f32>> {
    if n == 0 {
        return Err(GnnError::EmptyGraph);
    }
    if dim == 0 {
        return Err(GnnError::InvalidLayerConfig(
            "k-WL: dim must be > 0".to_string(),
        ));
    }
    if x.len() != n * n * dim {
        return Err(GnnError::DimensionMismatch {
            expected: n * n * dim,
            got: x.len(),
        });
    }
    let mut out = vec![0.0_f32; n * n * dim];
    for c in 0..dim {
        let red = Reductions::compute(x, n, dim, c);
        for i in 0..n {
            for j in 0..n {
                out[(i * n + j) * dim + c] = red.op_value(op, x, n, dim, c, i, j);
            }
        }
    }
    Ok(out)
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a [`KWlGnn`] layer.
#[derive(Debug, Clone, Copy)]
pub struct KWlConfig {
    /// Input pair-feature channels `d`.
    pub in_features: usize,
    /// Output pair-feature channels `d′`.
    pub out_features: usize,
    /// Seed for deterministic equivariant weight initialisation.
    pub seed: u64,
}

// ─── Layer ────────────────────────────────────────────────────────────────────

/// A single 2-WL (order-2) equivariant layer with a ReLU nonlinearity.
///
/// Holds a weight tensor `W ∈ ℝ^{15 × d × d′}` mixing the 15 equivariant basis
/// operations across channels, plus a 2-parameter equivariant bias per output
/// channel (`bias_all` everywhere and `bias_diag` additionally on the diagonal).
/// Weights are initialised deterministically from `config.seed`.
pub struct KWlGnn {
    config: KWlConfig,
    /// `[15 × in × out]` flattened as `weight[(op * in + c) * out + c']`.
    weight: Vec<f32>,
    /// `[out]` constant bias applied to every pair entry.
    bias_all: Vec<f32>,
    /// `[out]` extra bias applied only to diagonal entries.
    bias_diag: Vec<f32>,
}

impl KWlGnn {
    /// Construct a 2-WL layer from configuration with deterministic weights.
    ///
    /// # Errors
    ///
    /// Returns [`GnnError::InvalidLayerConfig`] if `in_features == 0` or
    /// `out_features == 0`.
    pub fn new(config: KWlConfig) -> GnnResult<Self> {
        if config.in_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "k-WL: in_features must be > 0".to_string(),
            ));
        }
        if config.out_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "k-WL: out_features must be > 0".to_string(),
            ));
        }
        let n_ops = PairOp::ALL.len();
        let n_w = n_ops * config.in_features * config.out_features;
        let mut rng = LcgRng::new(config.seed);
        let weight: Vec<f32> = (0..n_w).map(|_| centered_unit(&mut rng) * 0.3).collect();
        let bias_all: Vec<f32> = (0..config.out_features)
            .map(|_| centered_unit(&mut rng) * 0.1)
            .collect();
        let bias_diag: Vec<f32> = (0..config.out_features)
            .map(|_| centered_unit(&mut rng) * 0.1)
            .collect();
        Ok(Self {
            config,
            weight,
            bias_all,
            bias_diag,
        })
    }

    /// Output channel count `d′`.
    #[inline]
    pub fn output_dim(&self) -> usize {
        self.config.out_features
    }

    /// Equivariant forward pass over a node-pair tensor.
    ///
    /// # Arguments
    ///
    /// * `pair_features` — `[n × n × dim]` input tensor (`dim == in_features`).
    /// * `n_nodes` — number of graph nodes `n`.
    /// * `dim` — input channel count (must equal `in_features`).
    ///
    /// # Returns
    ///
    /// `[n × n × out_features]` pair tensor after the equivariant linear map,
    /// equivariant bias, and a pointwise ReLU.
    ///
    /// # Errors
    ///
    /// * [`GnnError::EmptyGraph`] if `n_nodes == 0`.
    /// * [`GnnError::DimensionMismatch`] if `dim != in_features` or
    ///   `pair_features.len() != n * n * dim`.
    /// * [`GnnError::NonFiniteOutput`] if any output value is non-finite.
    pub fn forward(
        &self,
        pair_features: &[f32],
        n_nodes: usize,
        dim: usize,
    ) -> GnnResult<Vec<f32>> {
        if n_nodes == 0 {
            return Err(GnnError::EmptyGraph);
        }
        if dim != self.config.in_features {
            return Err(GnnError::DimensionMismatch {
                expected: self.config.in_features,
                got: dim,
            });
        }
        if pair_features.len() != n_nodes * n_nodes * dim {
            return Err(GnnError::DimensionMismatch {
                expected: n_nodes * n_nodes * dim,
                got: pair_features.len(),
            });
        }

        let n = n_nodes;
        let in_f = self.config.in_features;
        let out_f = self.config.out_features;

        // Pre-compute per-channel reductions once.
        let reductions: Vec<Reductions> = (0..in_f)
            .map(|c| Reductions::compute(pair_features, n, in_f, c))
            .collect();

        let mut out = vec![0.0_f32; n * n * out_f];
        for i in 0..n {
            for j in 0..n {
                let on_diag = i == j;
                // Gather the 15 basis values for each input channel.
                for cp in 0..out_f {
                    let mut acc = self.bias_all[cp];
                    if on_diag {
                        acc += self.bias_diag[cp];
                    }
                    for (op_idx, &op) in PairOp::ALL.iter().enumerate() {
                        for (c, red) in reductions.iter().enumerate() {
                            let w = self.weight[(op_idx * in_f + c) * out_f + cp];
                            if w != 0.0 {
                                acc += w * red.op_value(op, pair_features, n, in_f, c, i, j);
                            }
                        }
                    }
                    // Pointwise ReLU preserves equivariance.
                    out[(i * n + j) * out_f + cp] = acc.max(0.0);
                }
            }
        }

        if out.iter().any(|v| !v.is_finite()) {
            return Err(GnnError::NonFiniteOutput("KWlGnn::forward"));
        }
        Ok(out)
    }

    /// Permutation-invariant graph-level readout: sum-pool over all node pairs.
    ///
    /// Returns a `[dim]` vector `g[c] = Σ_{i,j} pair_features[i,j,c]`, which is
    /// invariant to any relabelling of the nodes.
    ///
    /// # Errors
    ///
    /// * [`GnnError::EmptyGraph`] if `n_nodes == 0`.
    /// * [`GnnError::InvalidLayerConfig`] if `dim == 0`.
    /// * [`GnnError::DimensionMismatch`] if
    ///   `pair_features.len() != n * n * dim`.
    pub fn graph_readout(
        &self,
        pair_features: &[f32],
        n_nodes: usize,
        dim: usize,
    ) -> GnnResult<Vec<f32>> {
        graph_readout_sum(pair_features, n_nodes, dim)
    }
}

/// Standalone permutation-invariant sum-pool readout over node pairs.
///
/// `g[c] = Σ_{i,j} pair_features[i,j,c]`.
///
/// # Errors
///
/// * [`GnnError::EmptyGraph`] if `n_nodes == 0`.
/// * [`GnnError::InvalidLayerConfig`] if `dim == 0`.
/// * [`GnnError::DimensionMismatch`] if `pair_features.len() != n * n * dim`.
pub fn graph_readout_sum(pair_features: &[f32], n_nodes: usize, dim: usize) -> GnnResult<Vec<f32>> {
    if n_nodes == 0 {
        return Err(GnnError::EmptyGraph);
    }
    if dim == 0 {
        return Err(GnnError::InvalidLayerConfig(
            "k-WL readout: dim must be > 0".to_string(),
        ));
    }
    if pair_features.len() != n_nodes * n_nodes * dim {
        return Err(GnnError::DimensionMismatch {
            expected: n_nodes * n_nodes * dim,
            got: pair_features.len(),
        });
    }
    let mut g = vec![0.0_f32; dim];
    for i in 0..n_nodes {
        for j in 0..n_nodes {
            for c in 0..dim {
                g[c] += pair_features[(i * n_nodes + j) * dim + c];
            }
        }
    }
    Ok(g)
}

/// Map the LCG generator to a value in roughly `[-1, 1)`.
///
/// `LcgRng::next_u32` returns the top 31 bits of the state in `[0, 2³¹)`, so we
/// normalise by `2³¹` to obtain a unit value and re-centre it about zero.
#[inline]
fn centered_unit(rng: &mut LcgRng) -> f32 {
    let unit = (rng.next_u32() as f32) / 4_294_967_296.0_f32; // [0, 1)
    unit * 2.0 - 1.0
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Permute a pair tensor `[n×n×dim]` by `perm` (output node `i` reads from
    /// original node `perm[i]`).
    fn permute_pairs(x: &[f32], n: usize, dim: usize, perm: &[usize]) -> Vec<f32> {
        let mut out = vec![0.0_f32; n * n * dim];
        for i in 0..n {
            for j in 0..n {
                for c in 0..dim {
                    out[(i * n + j) * dim + c] = x[(perm[i] * n + perm[j]) * dim + c];
                }
            }
        }
        out
    }

    fn arange_pairs(n: usize, dim: usize) -> Vec<f32> {
        (0..n * n * dim).map(|i| (i as f32) * 0.07 - 0.5).collect()
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn new_valid() {
        let layer = KWlGnn::new(KWlConfig {
            in_features: 2,
            out_features: 3,
            seed: 7,
        })
        .expect("test invariant: value must be valid");
        assert_eq!(layer.output_dim(), 3);
    }

    #[test]
    fn new_invalid_zero_in() {
        assert!(
            KWlGnn::new(KWlConfig {
                in_features: 0,
                out_features: 3,
                seed: 1,
            })
            .is_err()
        );
    }

    #[test]
    fn new_invalid_zero_out() {
        assert!(
            KWlGnn::new(KWlConfig {
                in_features: 2,
                out_features: 0,
                seed: 1,
            })
            .is_err()
        );
    }

    #[test]
    fn fifteen_basis_ops() {
        assert_eq!(PairOp::ALL.len(), 15);
    }

    // ── (a) Permutation equivariance — the defining property ─────────────────

    #[test]
    fn forward_permutation_equivariant() {
        let n = 4;
        let din = 2;
        let dout = 3;
        let layer = KWlGnn::new(KWlConfig {
            in_features: din,
            out_features: dout,
            seed: 2024,
        })
        .expect("test invariant: value must be valid");
        let x = arange_pairs(n, din);
        let perm = [2usize, 0, 3, 1];

        let y = layer
            .forward(&x, n, din)
            .expect("test invariant: value must be valid");
        let y_permuted = permute_pairs(&y, n, dout, &perm);

        let x_permuted = permute_pairs(&x, n, din, &perm);
        let y_from_permuted = layer
            .forward(&x_permuted, n, din)
            .expect("test invariant: value must be valid");

        for (a, b) in y_permuted.iter().zip(y_from_permuted.iter()) {
            assert!((a - b).abs() < 1e-4, "equivariance broken: {a} vs {b}");
        }
    }

    // ── (b) Shapes & finiteness ──────────────────────────────────────────────

    #[test]
    fn forward_shape_and_finite() {
        let n = 5;
        let din = 2;
        let dout = 2;
        let layer = KWlGnn::new(KWlConfig {
            in_features: din,
            out_features: dout,
            seed: 11,
        })
        .expect("test invariant: value must be valid");
        let x = arange_pairs(n, din);
        let y = layer
            .forward(&x, n, din)
            .expect("test invariant: value must be valid");
        assert_eq!(y.len(), n * n * dout);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    // ── (c) Transpose basis op: symmetric input → symmetric output ───────────

    #[test]
    fn transpose_op_symmetric_input_symmetric_output() {
        let n = 3;
        let dim = 1;
        // Symmetric matrix X[i,j] = X[j,i].
        let mut x = vec![0.0_f32; n * n * dim];
        let vals = [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]];
        for i in 0..n {
            for j in 0..n {
                x[(i * n + j) * dim] = vals[i][j];
            }
        }
        let y = apply_pair_op(PairOp::Transpose, &x, n, dim)
            .expect("test invariant: value must be valid");
        // Output must be symmetric.
        for i in 0..n {
            for j in 0..n {
                let a = y[(i * n + j) * dim];
                let b = y[(j * n + i) * dim];
                assert!((a - b).abs() < 1e-6, "not symmetric at ({i},{j})");
            }
        }
        // For a symmetric input the transpose op returns the input unchanged.
        for (a, b) in y.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn symmetrization_always_symmetric() {
        // Identity + Transpose is a symmetrising equivariant combination.
        let n = 3;
        let dim = 1;
        let x = arange_pairs(n, dim); // generally non-symmetric
        let id = apply_pair_op(PairOp::Identity, &x, n, dim)
            .expect("test invariant: value must be valid");
        let tr = apply_pair_op(PairOp::Transpose, &x, n, dim)
            .expect("test invariant: value must be valid");
        let sym: Vec<f32> = id.iter().zip(tr.iter()).map(|(a, b)| a + b).collect();
        for i in 0..n {
            for j in 0..n {
                let a = sym[(i * n + j) * dim];
                let b = sym[(j * n + i) * dim];
                assert!((a - b).abs() < 1e-6, "symmetrised output not symmetric");
            }
        }
    }

    // ── Basis op correctness on a tiny known matrix ──────────────────────────

    #[test]
    fn basis_ops_match_definition() {
        // 2x2 matrix [[1, 2], [3, 4]], single channel.
        let n = 2;
        let dim = 1;
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let get = |op| apply_pair_op(op, &x, n, dim).expect("op");

        // diag = [1, 4], row = [3, 7], col = [4, 6], trace = 5, total = 10.
        assert_eq!(get(PairOp::Identity), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(get(PairOp::Transpose), vec![1.0, 3.0, 2.0, 4.0]);
        assert_eq!(get(PairOp::DiagToDiag), vec![1.0, 0.0, 0.0, 4.0]);
        assert_eq!(get(PairOp::RowSumToDiag), vec![3.0, 0.0, 0.0, 7.0]);
        assert_eq!(get(PairOp::ColSumToDiag), vec![4.0, 0.0, 0.0, 6.0]);
        assert_eq!(get(PairOp::TraceToDiag), vec![5.0, 0.0, 0.0, 5.0]);
        assert_eq!(get(PairOp::TotalToDiag), vec![10.0, 0.0, 0.0, 10.0]);
        assert_eq!(get(PairOp::DiagToRows), vec![1.0, 1.0, 4.0, 4.0]);
        assert_eq!(get(PairOp::DiagToCols), vec![1.0, 4.0, 1.0, 4.0]);
        assert_eq!(get(PairOp::RowSumToRows), vec![3.0, 3.0, 7.0, 7.0]);
        assert_eq!(get(PairOp::ColSumToCols), vec![4.0, 6.0, 4.0, 6.0]);
        assert_eq!(get(PairOp::RowSumToCols), vec![3.0, 7.0, 3.0, 7.0]);
        assert_eq!(get(PairOp::ColSumToRows), vec![4.0, 4.0, 6.0, 6.0]);
        assert_eq!(get(PairOp::TraceBroadcast), vec![5.0, 5.0, 5.0, 5.0]);
        assert_eq!(get(PairOp::TotalBroadcast), vec![10.0, 10.0, 10.0, 10.0]);
    }

    #[test]
    fn basis_ops_individually_equivariant() {
        let n = 4;
        let dim = 2;
        let x = arange_pairs(n, dim);
        let perm = [3usize, 1, 0, 2];
        for &op in PairOp::ALL.iter() {
            let y = apply_pair_op(op, &x, n, dim).expect("op");
            let y_perm = permute_pairs(&y, n, dim, &perm);
            let xp = permute_pairs(&x, n, dim, &perm);
            let y2 = apply_pair_op(op, &xp, n, dim).expect("op");
            for (a, b) in y_perm.iter().zip(y2.iter()) {
                assert!((a - b).abs() < 1e-4, "op {op:?} not equivariant");
            }
        }
    }

    // ── (d) Invalid dims → error ─────────────────────────────────────────────

    #[test]
    fn forward_dim_mismatch_errors() {
        let layer = KWlGnn::new(KWlConfig {
            in_features: 2,
            out_features: 2,
            seed: 1,
        })
        .expect("test invariant: value must be valid");
        let x = vec![0.0_f32; 3 * 3 * 3]; // dim 3 != in_features 2
        let err = layer.forward(&x, 3, 3);
        assert!(matches!(err, Err(GnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn forward_length_mismatch_errors() {
        let layer = KWlGnn::new(KWlConfig {
            in_features: 2,
            out_features: 2,
            seed: 1,
        })
        .expect("test invariant: value must be valid");
        let x = vec![0.0_f32; 10]; // not n*n*dim
        let err = layer.forward(&x, 3, 2);
        assert!(matches!(err, Err(GnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn forward_empty_graph_errors() {
        let layer = KWlGnn::new(KWlConfig {
            in_features: 2,
            out_features: 2,
            seed: 1,
        })
        .expect("test invariant: value must be valid");
        let err = layer.forward(&[], 0, 2);
        assert!(matches!(err, Err(GnnError::EmptyGraph)));
    }

    #[test]
    fn apply_op_bad_len_errors() {
        let err = apply_pair_op(PairOp::Identity, &[1.0, 2.0], 2, 1);
        assert!(matches!(err, Err(GnnError::DimensionMismatch { .. })));
    }

    // ── Graph-level readout invariance ────────────────────────────────────────

    #[test]
    fn readout_permutation_invariant() {
        let n = 4;
        let dim = 2;
        let layer = KWlGnn::new(KWlConfig {
            in_features: dim,
            out_features: dim,
            seed: 5,
        })
        .expect("test invariant: value must be valid");
        let x = arange_pairs(n, dim);
        let perm = [1usize, 3, 0, 2];
        let g1 = layer
            .graph_readout(&x, n, dim)
            .expect("test invariant: value must be valid");
        let xp = permute_pairs(&x, n, dim, &perm);
        let g2 = layer
            .graph_readout(&xp, n, dim)
            .expect("test invariant: value must be valid");
        for (a, b) in g1.iter().zip(g2.iter()) {
            assert!((a - b).abs() < 1e-4, "readout not invariant: {a} vs {b}");
        }
    }

    #[test]
    fn readout_sums_all_pairs() {
        let n = 2;
        let dim = 1;
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let g = graph_readout_sum(&x, n, dim).expect("test invariant: value must be valid");
        assert_eq!(g.len(), 1);
        assert!((g[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn readout_bad_len_errors() {
        let err = graph_readout_sum(&[1.0, 2.0, 3.0], 2, 1);
        assert!(matches!(err, Err(GnnError::DimensionMismatch { .. })));
    }

    // ── Determinism of weight init ────────────────────────────────────────────

    #[test]
    fn forward_deterministic_with_seed() {
        let n = 3;
        let dim = 2;
        let x = arange_pairs(n, dim);
        let make = || {
            KWlGnn::new(KWlConfig {
                in_features: dim,
                out_features: dim,
                seed: 99,
            })
            .expect("test invariant: value must be valid")
            .forward(&x, n, dim)
            .expect("test invariant: value must be valid")
        };
        assert_eq!(make(), make());
    }
}
