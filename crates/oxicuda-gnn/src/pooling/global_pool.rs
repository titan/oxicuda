//! Global graph pooling: reduce all node features to a single graph-level representation.

use crate::error::{GnnError, GnnResult};

/// Pooling strategy for global graph pooling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalPoolType {
    /// Average of node features.
    Mean,
    /// Element-wise maximum over node features.
    Max,
    /// Sum of node features.
    Sum,
    /// Soft-attention-weighted pooling.
    Attention,
}

fn validate_input(x: &[f32], n_nodes: usize, feat_dim: usize) -> GnnResult<()> {
    if feat_dim == 0 {
        return Err(GnnError::InvalidLayerConfig(
            "feat_dim must be > 0".to_string(),
        ));
    }
    if n_nodes == 0 {
        return Err(GnnError::EmptyGraph);
    }
    if x.len() != n_nodes * feat_dim {
        return Err(GnnError::DimensionMismatch {
            expected: n_nodes * feat_dim,
            got: x.len(),
        });
    }
    Ok(())
}

/// Global mean pooling: `g[k] = (1/n) * Σ_i x[i, k]`.
///
/// Returns `[feat_dim]`.
pub fn global_mean_pool(x: &[f32], n_nodes: usize, feat_dim: usize) -> GnnResult<Vec<f32>> {
    validate_input(x, n_nodes, feat_dim)?;
    let inv_n = 1.0 / n_nodes as f32;
    let mut out = vec![0.0_f32; feat_dim];
    for i in 0..n_nodes {
        for k in 0..feat_dim {
            out[k] += x[i * feat_dim + k];
        }
    }
    for v in &mut out {
        *v *= inv_n;
    }
    Ok(out)
}

/// Global max pooling: `g[k] = max_i x[i, k]`.
///
/// Returns `[feat_dim]`.
pub fn global_max_pool(x: &[f32], n_nodes: usize, feat_dim: usize) -> GnnResult<Vec<f32>> {
    validate_input(x, n_nodes, feat_dim)?;
    let mut out = vec![f32::NEG_INFINITY; feat_dim];
    for i in 0..n_nodes {
        for k in 0..feat_dim {
            if x[i * feat_dim + k] > out[k] {
                out[k] = x[i * feat_dim + k];
            }
        }
    }
    Ok(out)
}

/// Global sum pooling: `g[k] = Σ_i x[i, k]`.
///
/// Returns `[feat_dim]`.
pub fn global_sum_pool(x: &[f32], n_nodes: usize, feat_dim: usize) -> GnnResult<Vec<f32>> {
    validate_input(x, n_nodes, feat_dim)?;
    let mut out = vec![0.0_f32; feat_dim];
    for i in 0..n_nodes {
        for k in 0..feat_dim {
            out[k] += x[i * feat_dim + k];
        }
    }
    Ok(out)
}

/// Global attention pooling.
///
/// Gate: `score_i = a^T tanh(W * h_i + b)`, normalized by softmax.
/// Output: `g = Σ_i softmax(score)_i * h_i`.
///
/// # Arguments
///
/// - `x`: `[n_nodes × feat_dim]`
/// - `gate_weight`: `[feat_dim × feat_dim]`
/// - `gate_bias`: `[feat_dim]`
///
/// Returns `[feat_dim]`.
pub fn global_attention_pool(
    x: &[f32],
    n_nodes: usize,
    feat_dim: usize,
    gate_weight: &[f32],
    gate_bias: &[f32],
) -> GnnResult<Vec<f32>> {
    validate_input(x, n_nodes, feat_dim)?;
    if gate_weight.len() != feat_dim * feat_dim {
        return Err(GnnError::WeightShapeMismatch {
            r: feat_dim,
            c: feat_dim,
            d: feat_dim,
        });
    }
    if gate_bias.len() != feat_dim {
        return Err(GnnError::DimensionMismatch {
            expected: feat_dim,
            got: gate_bias.len(),
        });
    }

    // Compute gate scores: score_i = sum_k tanh(Σ_j W[k,j]*x[i,j] + b[k])
    let mut scores = Vec::with_capacity(n_nodes);
    for i in 0..n_nodes {
        let mut score = 0.0_f32;
        for k in 0..feat_dim {
            let mut lin = gate_bias[k];
            for j in 0..feat_dim {
                lin += gate_weight[k * feat_dim + j] * x[i * feat_dim + j];
            }
            score += lin.tanh();
        }
        scores.push(score);
    }

    // Softmax over scores
    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let alphas: Vec<f32> = if sum_exp > 0.0 {
        exps.iter().map(|&e| e / sum_exp).collect()
    } else {
        vec![1.0 / n_nodes as f32; n_nodes]
    };

    // Weighted sum
    let mut out = vec![0.0_f32; feat_dim];
    for i in 0..n_nodes {
        for k in 0..feat_dim {
            out[k] += alphas[i] * x[i * feat_dim + k];
        }
    }
    Ok(out)
}

/// Pool across a batch of graphs with varying sizes.
///
/// `x`: `[total_nodes × feat_dim]`
/// `batch_ids`: `[total_nodes]` — graph index for each node
/// Returns `[n_graphs × feat_dim]`.
pub fn batched_global_pool(
    x: &[f32],
    batch_ids: &[usize],
    n_graphs: usize,
    feat_dim: usize,
    pool: GlobalPoolType,
) -> GnnResult<Vec<f32>> {
    if feat_dim == 0 {
        return Err(GnnError::InvalidLayerConfig(
            "feat_dim must be > 0".to_string(),
        ));
    }
    let total = batch_ids.len();
    if x.len() != total * feat_dim {
        return Err(GnnError::DimensionMismatch {
            expected: total * feat_dim,
            got: x.len(),
        });
    }
    for &g in batch_ids {
        if g >= n_graphs {
            return Err(GnnError::NodeIndexOutOfRange {
                idx: g,
                n_nodes: n_graphs,
            });
        }
    }

    match pool {
        GlobalPoolType::Sum | GlobalPoolType::Mean | GlobalPoolType::Attention => {
            let mut out = vec![0.0_f32; n_graphs * feat_dim];
            let mut counts = vec![0usize; n_graphs];
            for i in 0..total {
                let g = batch_ids[i];
                counts[g] += 1;
                for k in 0..feat_dim {
                    out[g * feat_dim + k] += x[i * feat_dim + k];
                }
            }
            if matches!(pool, GlobalPoolType::Mean) {
                for g in 0..n_graphs {
                    if counts[g] > 0 {
                        let inv = 1.0 / counts[g] as f32;
                        for k in 0..feat_dim {
                            out[g * feat_dim + k] *= inv;
                        }
                    }
                }
            }
            Ok(out)
        }
        GlobalPoolType::Max => {
            let mut out = vec![f32::NEG_INFINITY; n_graphs * feat_dim];
            let mut has_nodes = vec![false; n_graphs];
            for i in 0..total {
                let g = batch_ids[i];
                has_nodes[g] = true;
                for k in 0..feat_dim {
                    if x[i * feat_dim + k] > out[g * feat_dim + k] {
                        out[g * feat_dim + k] = x[i * feat_dim + k];
                    }
                }
            }
            for g in 0..n_graphs {
                if !has_nodes[g] {
                    for k in 0..feat_dim {
                        out[g * feat_dim + k] = 0.0;
                    }
                }
            }
            Ok(out)
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_feats() -> (Vec<f32>, usize, usize) {
        // 4 nodes, feat_dim=2
        // node 0 = [1,2], node 1 = [3,4], node 2 = [5,6], node 3 = [7,8]
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        (x, 4, 2)
    }

    #[test]
    fn mean_pool_equals_sum_over_n() {
        let (x, n, d) = simple_feats();
        let mean = global_mean_pool(&x, n, d).expect("test invariant: value must be valid");
        let sum = global_sum_pool(&x, n, d).expect("test invariant: value must be valid");
        for k in 0..d {
            assert!((mean[k] - sum[k] / n as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn sum_pool_correct() {
        let (x, n, d) = simple_feats();
        let out = global_sum_pool(&x, n, d).expect("test invariant: value must be valid");
        assert!((out[0] - 16.0).abs() < 1e-6); // 1+3+5+7
        assert!((out[1] - 20.0).abs() < 1e-6); // 2+4+6+8
    }

    #[test]
    fn max_pool_selects_correct_features() {
        let (x, n, d) = simple_feats();
        let out = global_max_pool(&x, n, d).expect("test invariant: value must be valid");
        assert!((out[0] - 7.0).abs() < 1e-6); // max of [1,3,5,7]
        assert!((out[1] - 8.0).abs() < 1e-6); // max of [2,4,6,8]
    }

    #[test]
    fn attention_pool_sums_to_one_weight() {
        // With all-zero gate weights and bias, all scores equal → uniform alpha
        let (x, n, d) = simple_feats();
        let gw = vec![0.0_f32; d * d];
        let gb = vec![0.0_f32; d];
        let out =
            global_attention_pool(&x, n, d, &gw, &gb).expect("test invariant: value must be valid");
        // With uniform alpha = 1/4, out = mean
        let mean = global_mean_pool(&x, n, d).expect("test invariant: value must be valid");
        for k in 0..d {
            assert!((out[k] - mean[k]).abs() < 1e-5);
        }
    }

    #[test]
    fn empty_graph_error() {
        let err = global_mean_pool(&[], 0, 2);
        assert!(matches!(err, Err(GnnError::EmptyGraph)));
    }

    #[test]
    fn batched_mean_pool_consistency() {
        // 4 nodes split into 2 graphs: [0,1] → graph 0, [2,3] → graph 1
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let batch_ids = vec![0usize, 0, 1, 1];
        let out = batched_global_pool(&x, &batch_ids, 2, 2, GlobalPoolType::Mean)
            .expect("test invariant: value must be valid");
        // Graph 0 mean = ([1,2]+[3,4])/2 = [2,3]
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
        // Graph 1 mean = ([5,6]+[7,8])/2 = [6,7]
        assert!((out[2] - 6.0).abs() < 1e-6);
        assert!((out[3] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn batched_max_pool() {
        let x = vec![1.0_f32, 10.0, 3.0, 2.0, 5.0, 0.0, 7.0, 9.0];
        let batch_ids = vec![0usize, 0, 1, 1];
        let out = batched_global_pool(&x, &batch_ids, 2, 2, GlobalPoolType::Max)
            .expect("test invariant: value must be valid");
        // Graph 0: max([1,10],[3,2]) = [3,10]
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 10.0).abs() < 1e-6);
        // Graph 1: max([5,0],[7,9]) = [7,9]
        assert!((out[2] - 7.0).abs() < 1e-6);
        assert!((out[3] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn batched_sum_pool() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let batch_ids = vec![0usize, 0];
        let out = batched_global_pool(&x, &batch_ids, 1, 2, GlobalPoolType::Sum)
            .expect("test invariant: value must be valid");
        assert!((out[0] - 4.0).abs() < 1e-6);
        assert!((out[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn dimension_mismatch_error() {
        let err = global_mean_pool(&[1.0_f32, 2.0, 3.0], 4, 2);
        assert!(matches!(err, Err(GnnError::DimensionMismatch { .. })));
    }

    #[test]
    fn attention_pool_output_length() {
        let (x, n, d) = simple_feats();
        let gw = vec![0.1_f32; d * d];
        let gb = vec![0.0_f32; d];
        let out =
            global_attention_pool(&x, n, d, &gw, &gb).expect("test invariant: value must be valid");
        assert_eq!(out.len(), d);
    }
}
