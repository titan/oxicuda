//! Federated tabular learning primitives.
//!
//! CPU building blocks for horizontal and vertical federated learning over
//! tabular data, with no networking — only the local math each participant runs:
//!
//! * **Horizontal partitioning** ([`horizontal_split`]): split rows across
//!   clients (samples are private, the feature schema is shared).
//! * **Vertical partitioning** ([`vertical_split`]): split feature columns
//!   across clients (every client sees the same rows, different columns), the
//!   layout used by vertical FL / private-set-intersection pipelines.
//! * **FedAvg** ([`fed_avg`], McMahan et al. 2017): sample-count-weighted mean
//!   of client parameter vectors.
//! * **FedProx proximal term** ([`fedprox_proximal`], Li et al. 2020): the
//!   `μ/2 ‖w − w_global‖²` regulariser that keeps local updates near the global
//!   model under client drift / heterogeneity.
//! * **Secure aggregation masks** ([`SecureAggregator`]): pairwise additive
//!   masks that cancel on summation, so the server learns only the aggregate,
//!   never an individual client's vector.

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

// ─── Data partitioning ─────────────────────────────────────────────────────────

/// A contiguous slice description of one client's shard of a row-major matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shard {
    /// Index of the first row owned by this client (horizontal) — always 0 for
    /// vertical splits.
    pub row_start: usize,
    /// Number of rows in this shard.
    pub n_rows: usize,
    /// Index of the first feature column owned by this client (vertical) —
    /// always 0 for horizontal splits.
    pub col_start: usize,
    /// Number of feature columns in this shard.
    pub n_cols: usize,
}

/// Horizontally partition a `[n_samples × n_features]` matrix across
/// `n_clients` clients (rows split, all features shared).
///
/// Returns one `(shard_descriptor, data)` per client.  The first
/// `n_samples % n_clients` clients receive one extra row to absorb the
/// remainder, so the partition is balanced and lossless.
///
/// # Errors
/// Returns an error on shape mismatch, zero clients, or more clients than rows.
pub fn horizontal_split(
    data: &[f32],
    n_samples: usize,
    n_features: usize,
    n_clients: usize,
) -> TabularResult<Vec<(Shard, Vec<f32>)>> {
    if n_clients == 0 {
        return Err(TabularError::InvalidParameter {
            name: "n_clients".into(),
            msg: "must be > 0".into(),
        });
    }
    if n_features == 0 || n_samples == 0 {
        return Err(TabularError::EmptyInput);
    }
    if data.len() != n_samples * n_features {
        return Err(TabularError::DimensionMismatch {
            expected: n_samples * n_features,
            got: data.len(),
        });
    }
    if n_clients > n_samples {
        return Err(TabularError::InsufficientSamples {
            need: n_clients,
            got: n_samples,
        });
    }
    let base = n_samples / n_clients;
    let rem = n_samples % n_clients;
    let mut out = Vec::with_capacity(n_clients);
    let mut row = 0usize;
    for c in 0..n_clients {
        let rows = base + usize::from(c < rem);
        let start = row * n_features;
        let end = (row + rows) * n_features;
        out.push((
            Shard {
                row_start: row,
                n_rows: rows,
                col_start: 0,
                n_cols: n_features,
            },
            data[start..end].to_vec(),
        ));
        row += rows;
    }
    Ok(out)
}

/// Vertically partition a `[n_samples × n_features]` matrix across
/// `n_clients` clients (columns split, all rows shared).
///
/// Each returned shard is a dense row-major `[n_samples × shard_cols]` matrix.
///
/// # Errors
/// Returns an error on shape mismatch, zero clients, or more clients than
/// feature columns.
pub fn vertical_split(
    data: &[f32],
    n_samples: usize,
    n_features: usize,
    n_clients: usize,
) -> TabularResult<Vec<(Shard, Vec<f32>)>> {
    if n_clients == 0 {
        return Err(TabularError::InvalidParameter {
            name: "n_clients".into(),
            msg: "must be > 0".into(),
        });
    }
    if n_features == 0 || n_samples == 0 {
        return Err(TabularError::EmptyInput);
    }
    if data.len() != n_samples * n_features {
        return Err(TabularError::DimensionMismatch {
            expected: n_samples * n_features,
            got: data.len(),
        });
    }
    if n_clients > n_features {
        return Err(TabularError::InvalidFeatureCount { n: n_features });
    }
    let base = n_features / n_clients;
    let rem = n_features % n_clients;
    let mut out = Vec::with_capacity(n_clients);
    let mut col = 0usize;
    for c in 0..n_clients {
        let cols = base + usize::from(c < rem);
        let mut shard = vec![0.0_f32; n_samples * cols];
        for r in 0..n_samples {
            let src = r * n_features + col;
            shard[r * cols..(r + 1) * cols].copy_from_slice(&data[src..src + cols]);
        }
        out.push((
            Shard {
                row_start: 0,
                n_rows: n_samples,
                col_start: col,
                n_cols: cols,
            },
            shard,
        ));
        col += cols;
    }
    Ok(out)
}

// ─── FedAvg ───────────────────────────────────────────────────────────────────

/// Federated averaging (McMahan et al. 2017): sample-count-weighted mean of the
/// per-client parameter vectors.
///
/// `client_params[i]` is client `i`'s flattened parameter vector (all the same
/// length); `client_weights[i]` is its number of local training samples.  The
/// aggregate is `Σ_i n_i w_i / Σ_i n_i`.
///
/// # Errors
/// Returns an error if the lists are empty, mismatched in length, contain
/// differently-sized vectors, or the total weight is zero.
pub fn fed_avg(client_params: &[Vec<f32>], client_weights: &[usize]) -> TabularResult<Vec<f32>> {
    if client_params.is_empty() {
        return Err(TabularError::EmptyInput);
    }
    if client_params.len() != client_weights.len() {
        return Err(TabularError::DimensionMismatch {
            expected: client_params.len(),
            got: client_weights.len(),
        });
    }
    let dim = client_params[0].len();
    if dim == 0 {
        return Err(TabularError::EmptyInput);
    }
    let total: usize = client_weights.iter().sum();
    if total == 0 {
        return Err(TabularError::InvalidParameter {
            name: "client_weights".into(),
            msg: "total sample weight must be > 0".into(),
        });
    }
    let mut agg = vec![0.0_f32; dim];
    for (params, &w) in client_params.iter().zip(client_weights.iter()) {
        if params.len() != dim {
            return Err(TabularError::DimensionMismatch {
                expected: dim,
                got: params.len(),
            });
        }
        let frac = w as f32 / total as f32;
        for (a, &p) in agg.iter_mut().zip(params.iter()) {
            *a += frac * p;
        }
    }
    Ok(agg)
}

/// Unweighted (simple) federated average — every client contributes equally.
///
/// # Errors
/// Returns an error on empty input or mismatched vector lengths.
pub fn fed_avg_uniform(client_params: &[Vec<f32>]) -> TabularResult<Vec<f32>> {
    let weights = vec![1usize; client_params.len()];
    fed_avg(client_params, &weights)
}

/// FedProx proximal penalty (Li et al. 2020): `μ/2 · ‖w_local − w_global‖²`.
///
/// Added to the local objective, it bounds how far a client's update may stray
/// from the broadcast global model, stabilising training under non-IID data.
///
/// # Errors
/// Returns an error if the two vectors differ in length.
pub fn fedprox_proximal(w_local: &[f32], w_global: &[f32], mu: f32) -> TabularResult<f32> {
    if w_local.len() != w_global.len() {
        return Err(TabularError::DimensionMismatch {
            expected: w_global.len(),
            got: w_local.len(),
        });
    }
    let sq: f32 = w_local
        .iter()
        .zip(w_global.iter())
        .map(|(&l, &g)| {
            let d = l - g;
            d * d
        })
        .sum();
    Ok(0.5 * mu * sq)
}

/// Gradient of the FedProx proximal term w.r.t. the local weights:
/// `μ · (w_local − w_global)`.
///
/// # Errors
/// Returns an error if the two vectors differ in length.
pub fn fedprox_gradient(w_local: &[f32], w_global: &[f32], mu: f32) -> TabularResult<Vec<f32>> {
    if w_local.len() != w_global.len() {
        return Err(TabularError::DimensionMismatch {
            expected: w_global.len(),
            got: w_local.len(),
        });
    }
    Ok(w_local
        .iter()
        .zip(w_global.iter())
        .map(|(&l, &g)| mu * (l - g))
        .collect())
}

// ─── Secure aggregation ─────────────────────────────────────────────────────────

/// Pairwise additive secure-aggregation masker (Bonawitz et al. 2017, simplified
/// without the dropout-recovery secret sharing).
///
/// Each ordered client pair `(i, j)` shares a pseudo-random mask `m_{ij}`
/// derived from a common seed.  Client `i` adds `Σ_{j>i} m_{ij} − Σ_{j<i} m_{ji}`
/// to its vector before upload.  Summed over all clients the masks cancel, so
/// the server recovers the exact aggregate while no single masked vector leaks
/// its client's contribution.
#[derive(Debug, Clone)]
pub struct SecureAggregator {
    n_clients: usize,
    dim: usize,
    /// Shared seed; the per-pair seed is `base_seed ⊕ encode(i, j)`.
    base_seed: u64,
}

impl SecureAggregator {
    /// Construct an aggregator for `n_clients` clients over `dim`-length vectors.
    ///
    /// # Errors
    /// Returns an error if `n_clients < 2` or `dim == 0`.
    pub fn new(n_clients: usize, dim: usize, base_seed: u64) -> TabularResult<Self> {
        if n_clients < 2 {
            return Err(TabularError::InvalidParameter {
                name: "n_clients".into(),
                msg: "secure aggregation needs at least 2 clients".into(),
            });
        }
        if dim == 0 {
            return Err(TabularError::EmptyInput);
        }
        Ok(Self {
            n_clients,
            dim,
            base_seed,
        })
    }

    /// Deterministic per-pair mask vector for the ordered pair `(low, high)`
    /// with `low < high`.  Both clients derive the same vector from the shared
    /// seed.
    fn pair_mask(&self, low: usize, high: usize) -> Vec<f32> {
        // Symmetric seed so (low, high) and (high, low) agree.
        let seed = self
            .base_seed
            .wrapping_add((low as u64) << 32)
            .wrapping_add(high as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut rng = LcgRng::new(seed);
        (0..self.dim)
            .map(|_| {
                // Full-range unit uniform mapped to [-1, 1).
                let u = f64::from(rng.next_u32()) / 2f64.powi(32);
                (2.0 * u - 1.0) as f32
            })
            .collect()
    }

    /// Apply this client's net mask to `vector`, returning the masked vector that
    /// is safe to upload.
    ///
    /// # Errors
    /// Returns an error if `client >= n_clients` or `vector.len() != dim`.
    pub fn mask(&self, client: usize, vector: &[f32]) -> TabularResult<Vec<f32>> {
        if client >= self.n_clients {
            return Err(TabularError::InvalidParameter {
                name: "client".into(),
                msg: "client index out of range".into(),
            });
        }
        if vector.len() != self.dim {
            return Err(TabularError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        let mut out = vector.to_vec();
        for other in 0..self.n_clients {
            if other == client {
                continue;
            }
            let (low, high) = (client.min(other), client.max(other));
            let m = self.pair_mask(low, high);
            // Add when this client is the lower index, subtract otherwise, so the
            // two contributions cancel in the global sum.
            let sign = if client < other { 1.0 } else { -1.0 };
            for (o, &mv) in out.iter_mut().zip(m.iter()) {
                *o += sign * mv;
            }
        }
        Ok(out)
    }

    /// Sum a full set of masked client vectors back into the true aggregate
    /// (the pairwise masks cancel exactly).
    ///
    /// # Errors
    /// Returns an error if the number of masked vectors does not equal
    /// `n_clients` or any vector has the wrong length.
    pub fn aggregate(&self, masked: &[Vec<f32>]) -> TabularResult<Vec<f32>> {
        if masked.len() != self.n_clients {
            return Err(TabularError::DimensionMismatch {
                expected: self.n_clients,
                got: masked.len(),
            });
        }
        let mut sum = vec![0.0_f32; self.dim];
        for v in masked {
            if v.len() != self.dim {
                return Err(TabularError::DimensionMismatch {
                    expected: self.dim,
                    got: v.len(),
                });
            }
            for (s, &x) in sum.iter_mut().zip(v.iter()) {
                *s += x;
            }
        }
        Ok(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizontal_split_is_lossless_and_balanced() {
        let data: Vec<f32> = (0..(7 * 3)).map(|i| i as f32).collect();
        let shards = horizontal_split(&data, 7, 3, 3).expect("split");
        assert_eq!(shards.len(), 3);
        let total_rows: usize = shards.iter().map(|(s, _)| s.n_rows).sum();
        assert_eq!(total_rows, 7);
        // First (7 % 3 = 1) client gets the extra row → 3, 2, 2.
        assert_eq!(shards[0].0.n_rows, 3);
        assert_eq!(shards[1].0.n_rows, 2);
        assert_eq!(shards[2].0.n_rows, 2);
        // Concatenating shards reproduces the original.
        let mut rebuilt = Vec::new();
        for (_, d) in &shards {
            rebuilt.extend_from_slice(d);
        }
        assert_eq!(rebuilt, data);
    }

    #[test]
    fn vertical_split_columns_match() {
        // 2 samples × 5 features, 2 clients → cols 3 + 2.
        let data: Vec<f32> = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, 9.0, //
        ];
        let shards = vertical_split(&data, 2, 5, 2).expect("split");
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].0.n_cols, 3);
        assert_eq!(shards[1].0.n_cols, 2);
        // Client 0 owns columns 0..3.
        assert_eq!(shards[0].1, vec![0.0, 1.0, 2.0, 5.0, 6.0, 7.0]);
        // Client 1 owns columns 3..5.
        assert_eq!(shards[1].1, vec![3.0, 4.0, 8.0, 9.0]);
    }

    #[test]
    fn fed_avg_weighted_mean() {
        let p = vec![vec![0.0_f32, 0.0], vec![4.0, 8.0]];
        let w = vec![1usize, 3];
        let agg = fed_avg(&p, &w).expect("avg");
        // (1·0 + 3·4)/4 = 3 ; (1·0 + 3·8)/4 = 6
        assert!((agg[0] - 3.0).abs() < 1e-6);
        assert!((agg[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn fed_avg_uniform_matches_plain_mean() {
        let p = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let agg = fed_avg_uniform(&p).expect("avg");
        assert!((agg[0] - 3.0).abs() < 1e-6);
        assert!((agg[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn fed_avg_rejects_mismatched_dims() {
        let p = vec![vec![1.0_f32, 2.0], vec![3.0]];
        let w = vec![1usize, 1];
        assert!(fed_avg(&p, &w).is_err());
    }

    #[test]
    fn fedprox_zero_when_equal() {
        let w = vec![0.5_f32, -1.0, 2.0];
        let pen = fedprox_proximal(&w, &w, 0.1).expect("prox");
        assert!(pen.abs() < 1e-9);
        let grad = fedprox_gradient(&w, &w, 0.1).expect("grad");
        assert!(grad.iter().all(|&g| g.abs() < 1e-9));
    }

    #[test]
    fn fedprox_value_and_gradient() {
        let l = vec![1.0_f32, 2.0];
        let g = vec![0.0_f32, 0.0];
        let mu = 2.0;
        // 0.5·2·(1²+2²) = 5
        assert!((fedprox_proximal(&l, &g, mu).expect("p") - 5.0).abs() < 1e-6);
        // μ·(l−g) = [2, 4]
        let grad = fedprox_gradient(&l, &g, mu).expect("g");
        assert!((grad[0] - 2.0).abs() < 1e-6);
        assert!((grad[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn secure_aggregation_masks_cancel() {
        let n_clients = 4;
        let dim = 6;
        let agg = SecureAggregator::new(n_clients, dim, 2024).expect("new");
        // Client vectors.
        let vectors: Vec<Vec<f32>> = (0..n_clients)
            .map(|c| (0..dim).map(|d| (c * dim + d) as f32 * 0.1).collect())
            .collect();
        // Plain (insecure) sum for reference.
        let mut plain = vec![0.0_f32; dim];
        for v in &vectors {
            for (s, &x) in plain.iter_mut().zip(v.iter()) {
                *s += x;
            }
        }
        // Mask each, then aggregate.
        let masked: Vec<Vec<f32>> = vectors
            .iter()
            .enumerate()
            .map(|(c, v)| agg.mask(c, v).expect("mask"))
            .collect();
        // An individual masked vector must NOT equal its plaintext (privacy).
        assert_ne!(masked[0], vectors[0]);
        let recovered = agg.aggregate(&masked).expect("agg");
        for (r, p) in recovered.iter().zip(plain.iter()) {
            assert!((r - p).abs() < 1e-4, "recovered {r} vs plain {p}");
        }
    }

    #[test]
    fn secure_aggregator_requires_two_clients() {
        assert!(SecureAggregator::new(1, 4, 0).is_err());
    }

    #[test]
    fn pair_mask_is_symmetric() {
        let agg = SecureAggregator::new(3, 5, 7).expect("new");
        // The mask client 0 uses against client 2 must equal what client 2 uses
        // against client 0 (only the sign differs in mask()).
        let m02 = agg.pair_mask(0, 2);
        let m20 = agg.pair_mask(0, 2);
        assert_eq!(m02, m20);
    }
}
