//! SwAV — Caron et al. 2020 — Swapping Assignments between Views.
//!
//! Each view is assigned to one of `K` learnable prototypes via a softmax
//! over `Z·C^⊤` (the "code"). To prevent representation collapse, the codes
//! are normalised to be balanced across the batch with a fast Sinkhorn-Knopp
//! iteration (3 iterations in the paper).
//!
//! The training objective swaps assignments between views:
//! ```text
//!     L = (1/2) [ CE(p_a, q_b) + CE(p_b, q_a) ]
//!     p = softmax(Z·C^⊤ / τ)        (predictions)
//!     q = sinkhorn(softmax(Z·C^⊤))  (codes — stop-gradient)
//! ```

use crate::error::{SslError, SslResult};

/// SwAV configuration.
#[derive(Debug, Clone)]
pub struct SwavConfig {
    /// Temperature τ used inside the prediction softmax. Default 0.1.
    pub temperature: f32,
    /// Sinkhorn entropy regulariser ε. Default 0.05.
    pub epsilon: f32,
    /// Number of Sinkhorn-Knopp iterations. Default 3.
    pub sinkhorn_iters: usize,
}

impl Default for SwavConfig {
    fn default() -> Self {
        Self {
            temperature: 0.1,
            epsilon: 0.05,
            sinkhorn_iters: 3,
        }
    }
}

impl SwavConfig {
    /// Validated config.
    ///
    /// # Errors
    /// - [`SslError::InvalidTemperature`] for τ ≤ 0 or non-finite.
    /// - [`SslError::InvalidLossWeight`] for ε ≤ 0 or non-finite.
    pub fn new(temperature: f32, epsilon: f32, sinkhorn_iters: usize) -> SslResult<Self> {
        if !(temperature.is_finite() && temperature > 0.0) {
            return Err(SslError::InvalidTemperature { temp: temperature });
        }
        if !(epsilon.is_finite() && epsilon > 0.0) {
            return Err(SslError::InvalidLossWeight { weight: epsilon });
        }
        Ok(Self {
            temperature,
            epsilon,
            sinkhorn_iters,
        })
    }
}

/// Sinkhorn-Knopp normalisation of a `[N × K]` non-negative matrix
/// to make rows sum to 1/N and columns sum to 1/K.
///
/// Operates on `Q = exp(scores / ε)` (caller passes the exponentiated matrix).
///
/// # Errors
/// - [`SslError::EmptyInput`] if `n == 0` or `k < 2`.
/// - [`SslError::DimensionMismatch`] if `q.len() != n*k`.
/// - [`SslError::SinkhornDiverged`] if a row/col sum becomes zero.
pub fn sinkhorn_knopp(q: &mut [f32], n: usize, k: usize, iters: usize) -> SslResult<()> {
    if n == 0 {
        return Err(SslError::EmptyInput);
    }
    if k < 2 {
        return Err(SslError::NumPrototypesTooSmall);
    }
    if q.len() != n * k {
        return Err(SslError::DimensionMismatch {
            expected: n * k,
            got: q.len(),
        });
    }
    // Initial total normalisation so the matrix sums to 1.
    let total: f32 = q.iter().sum();
    if total <= 0.0 {
        return Err(SslError::SinkhornDiverged { iters: 0 });
    }
    let inv_total = 1.0 / total;
    for v in q.iter_mut() {
        *v *= inv_total;
    }
    // Alternating row/column normalisation.
    for it in 0..iters {
        // Column sums → divide each col by sum * K.
        let mut col_sum = vec![0.0_f32; k];
        for i in 0..n {
            for j in 0..k {
                col_sum[j] += q[i * k + j];
            }
        }
        for c in &col_sum {
            if !(c.is_finite() && *c > 0.0) {
                return Err(SslError::SinkhornDiverged { iters: it });
            }
        }
        for i in 0..n {
            for j in 0..k {
                q[i * k + j] /= col_sum[j] * k as f32;
            }
        }
        // Row sums → divide each row by sum * N.
        for i in 0..n {
            let mut row_sum = 0.0_f32;
            for j in 0..k {
                row_sum += q[i * k + j];
            }
            if !(row_sum.is_finite() && row_sum > 0.0) {
                return Err(SslError::SinkhornDiverged { iters: it });
            }
            for j in 0..k {
                q[i * k + j] /= row_sum * n as f32;
            }
        }
    }
    // Rescale so each row sums to 1 (probability distribution).
    for i in 0..n {
        let mut row_sum = 0.0_f32;
        for j in 0..k {
            row_sum += q[i * k + j];
        }
        if row_sum > 0.0 {
            for j in 0..k {
                q[i * k + j] /= row_sum;
            }
        }
    }
    Ok(())
}

/// Softmax over the `K` columns of a `[N × K]` matrix at temperature `τ`.
fn row_softmax(scores: &[f32], n: usize, k: usize, t: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * k);
    for i in 0..n {
        let row = &scores[i * k..(i + 1) * k];
        let mut max_v = f32::NEG_INFINITY;
        for &v in row {
            if v / t > max_v {
                max_v = v / t;
            }
        }
        let mut s = 0.0_f64;
        let mut tmp = Vec::with_capacity(k);
        for &v in row {
            let e = ((v / t - max_v) as f64).exp();
            tmp.push(e);
            s += e;
        }
        let inv = 1.0_f64 / s.max(1e-30);
        for v in &tmp {
            out.push((*v * inv) as f32);
        }
    }
    out
}

/// Cross-entropy `−Σ_n Σ_k q[n,k]·log p[n,k]` averaged across batch.
fn cross_entropy(q: &[f32], p: &[f32], n: usize, k: usize) -> f32 {
    let mut total = 0.0_f64;
    for i in 0..n {
        for j in 0..k {
            let pj = p[i * k + j].max(1e-12);
            total += -(q[i * k + j] as f64) * (pj as f64).ln();
        }
    }
    (total / n as f64) as f32
}

/// Compute the SwAV swapped-assignment loss.
///
/// `z_a`, `z_b` are `[N × D]` projections; `protos` is `[K × D]` row-major.
/// Returns the average loss over the two directions.
///
/// # Errors
/// - [`SslError::DimensionMismatch`] when shapes disagree.
/// - [`SslError::EmptyInput`] when `n == 0` or `d == 0`.
/// - [`SslError::NumPrototypesTooSmall`] when `k < 2`.
pub fn swav_loss(
    z_a: &[f32],
    z_b: &[f32],
    protos: &[f32],
    n: usize,
    d: usize,
    k: usize,
    cfg: &SwavConfig,
) -> SslResult<f32> {
    if n == 0 || d == 0 {
        return Err(SslError::EmptyInput);
    }
    if k < 2 {
        return Err(SslError::NumPrototypesTooSmall);
    }
    if z_a.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: z_a.len(),
        });
    }
    if z_b.len() != n * d {
        return Err(SslError::DimensionMismatch {
            expected: n * d,
            got: z_b.len(),
        });
    }
    if protos.len() != k * d {
        return Err(SslError::DimensionMismatch {
            expected: k * d,
            got: protos.len(),
        });
    }
    // Score matrices Z · Cᵀ.
    let scores_a = matmul_zct(z_a, protos, n, d, k);
    let scores_b = matmul_zct(z_b, protos, n, d, k);
    // Codes Q = Sinkhorn(exp(score / ε))
    let mut q_a: Vec<f32> = scores_a.iter().map(|s| (s / cfg.epsilon).exp()).collect();
    let mut q_b: Vec<f32> = scores_b.iter().map(|s| (s / cfg.epsilon).exp()).collect();
    sinkhorn_knopp(&mut q_a, n, k, cfg.sinkhorn_iters)?;
    sinkhorn_knopp(&mut q_b, n, k, cfg.sinkhorn_iters)?;
    // Predictions p = softmax(score / τ)
    let p_a = row_softmax(&scores_a, n, k, cfg.temperature);
    let p_b = row_softmax(&scores_b, n, k, cfg.temperature);
    // Swapped CE
    let l1 = cross_entropy(&q_b, &p_a, n, k);
    let l2 = cross_entropy(&q_a, &p_b, n, k);
    Ok(0.5 * (l1 + l2))
}

/// `Z · Cᵀ` where `Z ∈ ℝ^{N×D}` and `C ∈ ℝ^{K×D}`. Returns `[N × K]`.
fn matmul_zct(z: &[f32], protos: &[f32], n: usize, d: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n * k];
    for i in 0..n {
        for j in 0..k {
            let mut acc = 0.0_f32;
            let row = &z[i * d..(i + 1) * d];
            let proto = &protos[j * d..(j + 1) * d];
            for (a, b) in row.iter().zip(proto.iter()) {
                acc += a * b;
            }
            out[i * k + j] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swav_default_config() {
        let cfg = SwavConfig::default();
        assert!((cfg.temperature - 0.1).abs() < 1e-7);
        assert!((cfg.epsilon - 0.05).abs() < 1e-7);
        assert_eq!(cfg.sinkhorn_iters, 3);
    }

    #[test]
    fn swav_rejects_invalid_temperature() {
        assert!(SwavConfig::new(0.0, 0.05, 3).is_err());
        assert!(SwavConfig::new(-1.0, 0.05, 3).is_err());
    }

    #[test]
    fn swav_rejects_invalid_epsilon() {
        assert!(SwavConfig::new(0.1, 0.0, 3).is_err());
    }

    #[test]
    fn sinkhorn_normalises_uniform_input() {
        // Uniform input → output is also uniform.
        let n = 4;
        let k = 3;
        let mut q = vec![1.0_f32; n * k];
        sinkhorn_knopp(&mut q, n, k, 5).expect("sinkhorn_knopp should succeed");
        for v in &q {
            assert!((*v - 1.0 / k as f32).abs() < 1e-4, "v = {v}");
        }
    }

    #[test]
    fn sinkhorn_rows_sum_to_one() {
        let n = 4;
        let k = 3;
        let mut q = vec![0.0_f32; n * k];
        for (i, v) in q.iter_mut().enumerate() {
            *v = (i as f32 + 1.0) * 0.1;
        }
        sinkhorn_knopp(&mut q, n, k, 5).expect("sinkhorn_knopp should succeed");
        for i in 0..n {
            let s: f32 = q[i * k..(i + 1) * k].iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "row sum = {s}");
        }
    }

    #[test]
    fn sinkhorn_rejects_zero_total() {
        let n = 2;
        let k = 2;
        let mut q = vec![0.0_f32; n * k];
        assert!(sinkhorn_knopp(&mut q, n, k, 3).is_err());
    }

    #[test]
    fn sinkhorn_rejects_too_few_prototypes() {
        let mut q = vec![1.0_f32];
        assert!(sinkhorn_knopp(&mut q, 1, 1, 3).is_err());
    }

    #[test]
    fn swav_loss_finite_on_random_data() {
        let n = 8;
        let d = 4;
        let k = 6;
        let z_a: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.013).sin()).collect();
        let z_b: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.027).cos()).collect();
        let protos: Vec<f32> = (0..k * d).map(|i| (i as f32 * 0.041).sin()).collect();
        let cfg = SwavConfig::default();
        let l = swav_loss(&z_a, &z_b, &protos, n, d, k, &cfg).expect("swav_loss should succeed");
        assert!(l.is_finite() && l > 0.0);
    }

    #[test]
    fn swav_rejects_dim_mismatch_protos() {
        let n = 2;
        let d = 4;
        let k = 3;
        let z = vec![1.0_f32; n * d];
        let protos = vec![1.0_f32; k * 2]; // wrong d
        let cfg = SwavConfig::default();
        assert!(swav_loss(&z, &z, &protos, n, d, k, &cfg).is_err());
    }
}
