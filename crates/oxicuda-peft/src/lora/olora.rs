//! OLoRA — Orthonormal Low-Rank Adaptation.
//!
//! Reference: Büyükakyüz, K. (2024). *OLoRA: Orthonormal Low-Rank Adaptation of Large
//! Language Models*. <https://arxiv.org/abs/2406.01775>
//!
//! OLoRA replaces the standard LoRA random-Gaussian initialisation of `A` with a matrix
//! whose rows are *orthonormal in* `ℝ^{in_features}`. The orthonormal basis is obtained by
//! drawing a Gaussian matrix and running modified Gram-Schmidt across its rows. The
//! resulting `A` satisfies `A · Aᵀ = I_rank`, which empirically accelerates convergence
//! relative to vanilla LoRA. Both `A` and `B` are trainable.
//!
//! Forward: `y = s · B · (A · x)`, with `s = α / rank`.
//!
//! Closed-form gradients (loss `L`, upstream `g = ∂L/∂y`, `t = A · x`):
//!
//! ```text
//!   ∂L/∂B  = s · g · tᵀ            (out × rank, outer product)
//!   ∂L/∂A  = s · (Bᵀ · g) · xᵀ     (rank × in, outer product)
//! ```
//!
//! Orthonormality is not maintained after the first SGD step; users wanting an explicit
//! orthogonality regulariser can add `||A · Aᵀ - I||_F` to the loss.

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

/// Hyper-parameter bundle for a single OLoRA adapter.
#[derive(Debug, Clone)]
pub struct OloraConfig {
    /// Input feature count.
    pub in_features: usize,
    /// Output feature count.
    pub out_features: usize,
    /// Low-rank dimension. Must satisfy `rank ≤ min(in_features, out_features)` and
    /// `rank ≤ in_features` for orthonormality to exist.
    pub rank: usize,
    /// Global scaling factor `α`; effective scale is `s = α / rank`.
    pub alpha: f64,
}

impl OloraConfig {
    /// Effective scale `s = α / rank`.
    #[must_use]
    pub fn scale(&self) -> f64 {
        if self.rank == 0 {
            0.0
        } else {
            self.alpha / self.rank as f64
        }
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// - [`PeftError::EmptyInput`] if any dimension is zero.
    /// - [`PeftError::RankTooLarge`] if `rank > min(in_features, out_features)`.
    ///   Orthonormality of row-rank `r` requires `r ≤ in_features`, which is implied.
    pub fn validate(&self) -> PeftResult<()> {
        if self.in_features == 0 || self.out_features == 0 || self.rank == 0 {
            return Err(PeftError::EmptyInput);
        }
        let dim = self.in_features.min(self.out_features);
        if self.rank > dim {
            return Err(PeftError::RankTooLarge {
                rank: self.rank,
                dim,
            });
        }
        Ok(())
    }
}

/// OLoRA adapter with orthonormally-initialised `A` and zero-initialised `B`.
///
/// `a` has shape `[rank × in_features]` (row-major) with row-orthonormal contents at
/// construction time. `b` has shape `[out_features × rank]` (row-major), zero-initialised.
#[derive(Debug, Clone)]
pub struct OloraAdapter {
    /// Down-projection, row-major `[rank × in_features]`.
    pub a: Vec<f64>,
    /// Up-projection, row-major `[out_features × rank]`.
    pub b: Vec<f64>,
    /// Captured configuration.
    pub cfg: OloraConfig,
}

impl OloraAdapter {
    /// Build a fresh adapter.
    ///
    /// 1. Draw `M ∈ ℝ^{rank × in_features}` from `N(0, 1)`.
    /// 2. Run modified Gram-Schmidt across rows of `M`.
    /// 3. Store the orthonormalised matrix as `A`.
    ///
    /// `B` is zero-initialised so the adapter starts as a no-op.
    ///
    /// # Errors
    ///
    /// Forwards [`OloraConfig::validate`] errors.
    pub fn new(cfg: OloraConfig, rng_seed: u64) -> PeftResult<Self> {
        cfg.validate()?;
        let r = cfg.rank;
        let in_f = cfg.in_features;
        let mut rng = LcgRng::new(rng_seed);
        let n = r * in_f;
        let mut a = vec![0.0_f64; n];
        let mut i = 0;
        while i + 1 < n {
            let (u, v) = rng.next_normal_pair();
            a[i] = u as f64;
            a[i + 1] = v as f64;
            i += 2;
        }
        if i < n {
            a[i] = rng.next_normal() as f64;
        }
        modified_gram_schmidt_rows(&mut a, r, in_f);
        let b = vec![0.0_f64; cfg.out_features * r];
        Ok(Self { a, b, cfg })
    }

    /// Number of trainable parameters, `rank · (in_features + out_features)`.
    #[must_use]
    pub fn n_trainable(&self) -> usize {
        self.cfg.rank * (self.cfg.in_features + self.cfg.out_features)
    }

    /// Effective scale `s = α / rank`.
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.cfg.scale()
    }

    /// Check whether `A · Aᵀ ≈ I_rank` within `tol`.
    ///
    /// Returns `true` immediately after construction (within machine precision); after any
    /// gradient update `A` generally drifts away from orthonormality.
    #[must_use]
    pub fn is_a_orthonormal(&self, tol: f64) -> bool {
        let r = self.cfg.rank;
        let in_f = self.cfg.in_features;
        for i in 0..r {
            for j in 0..r {
                let mut dot = 0.0_f64;
                for k in 0..in_f {
                    dot += self.a[i * in_f + k] * self.a[j * in_f + k];
                }
                let target = if i == j { 1.0 } else { 0.0 };
                if (dot - target).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Compute `y = s · B · (A · x)`.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] when `x.len() != in_features`.
    pub fn forward(&self, x: &[f64]) -> PeftResult<Vec<f64>> {
        if x.len() != self.cfg.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.cfg.in_features,
                got: x.len(),
            });
        }
        let t = self.compute_ax(x);
        Ok(self.compute_bt_scaled(&t))
    }

    /// Closed-form gradients with respect to both `A` and `B`.
    ///
    /// Returns `(grad_a, grad_b)`:
    /// - `grad_a` row-major `[rank × in_features]` with `grad_a = s · (Bᵀ · grad_y) · xᵀ`.
    /// - `grad_b` row-major `[out_features × rank]` with `grad_b = s · grad_y · tᵀ`,
    ///   `t = A · x`.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] when input lengths disagree with the
    /// adapter's expected shapes.
    pub fn backward(&self, x: &[f64], grad_y: &[f64]) -> PeftResult<(Vec<f64>, Vec<f64>)> {
        if x.len() != self.cfg.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.cfg.in_features,
                got: x.len(),
            });
        }
        if grad_y.len() != self.cfg.out_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.cfg.out_features,
                got: grad_y.len(),
            });
        }
        let r = self.cfg.rank;
        let in_f = self.cfg.in_features;
        let out = self.cfg.out_features;
        let s = self.scale();
        let t = self.compute_ax(x);
        // grad_b[i,k] = s * grad_y[i] * t[k]
        let mut grad_b = vec![0.0_f64; out * r];
        for (i, g_i) in grad_y.iter().enumerate() {
            let row = i * r;
            let scaled = s * g_i;
            for (k, t_k) in t.iter().enumerate() {
                grad_b[row + k] = scaled * t_k;
            }
        }
        // u[k] = Σ_i Bᵀ[k,i] · grad_y[i] = Σ_i B[i,k] · grad_y[i]
        let mut u = vec![0.0_f64; r];
        for (k, u_k) in u.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            for (i, g_i) in grad_y.iter().enumerate() {
                acc += self.b[i * r + k] * g_i;
            }
            *u_k = acc;
        }
        // grad_a[k,j] = s * u[k] * x[j]
        let mut grad_a = vec![0.0_f64; r * in_f];
        for (k, u_k) in u.iter().enumerate() {
            let row = k * in_f;
            let scaled = s * u_k;
            for (j, x_j) in x.iter().enumerate() {
                grad_a[row + j] = scaled * x_j;
            }
        }
        Ok((grad_a, grad_b))
    }

    /// SGD update for both factors: `A ← A − lr · grad_a`, `B ← B − lr · grad_b`.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] when gradient lengths disagree with the
    /// adapter's expected shapes.
    pub fn apply_grads(&mut self, grad_a: &[f64], grad_b: &[f64], lr: f64) -> PeftResult<()> {
        let exp_a = self.cfg.rank * self.cfg.in_features;
        let exp_b = self.cfg.out_features * self.cfg.rank;
        if grad_a.len() != exp_a {
            return Err(PeftError::DimensionMismatch {
                expected: exp_a,
                got: grad_a.len(),
            });
        }
        if grad_b.len() != exp_b {
            return Err(PeftError::DimensionMismatch {
                expected: exp_b,
                got: grad_b.len(),
            });
        }
        for (a, g) in self.a.iter_mut().zip(grad_a.iter()) {
            *a -= lr * g;
        }
        for (b, g) in self.b.iter_mut().zip(grad_b.iter()) {
            *b -= lr * g;
        }
        Ok(())
    }

    fn compute_ax(&self, x: &[f64]) -> Vec<f64> {
        let r = self.cfg.rank;
        let in_f = self.cfg.in_features;
        let mut t = vec![0.0_f64; r];
        for (k, t_k) in t.iter_mut().enumerate() {
            let row_start = k * in_f;
            let mut acc = 0.0_f64;
            for (j, x_j) in x.iter().enumerate() {
                acc += self.a[row_start + j] * x_j;
            }
            *t_k = acc;
        }
        t
    }

    fn compute_bt_scaled(&self, t: &[f64]) -> Vec<f64> {
        let r = self.cfg.rank;
        let out = self.cfg.out_features;
        let s = self.scale();
        let mut y = vec![0.0_f64; out];
        for (i, y_i) in y.iter_mut().enumerate() {
            let row_start = i * r;
            let mut acc = 0.0_f64;
            for (k, t_k) in t.iter().enumerate() {
                acc += self.b[row_start + k] * t_k;
            }
            *y_i = s * acc;
        }
        y
    }
}

/// Modified Gram-Schmidt across rows of `a` (row-major `[rows × cols]`).
///
/// Each row is normalised against all previously processed rows. If a row's residual norm
/// is below `1e-12` (i.e. the row is numerically in the span of earlier rows) it is
/// replaced by a deterministic canonical basis vector so the output is still well-defined.
fn modified_gram_schmidt_rows(a: &mut [f64], rows: usize, cols: usize) {
    for i in 0..rows {
        // Orthogonalise row i against all earlier (unit-norm, mutually orthogonal) rows.
        for j in 0..i {
            let dot = row_dot(a, i, j, cols);
            row_axpy(a, i, j, -dot, cols);
        }
        let norm = row_norm(a, i, cols);
        if norm >= 1e-12 {
            row_scale(a, i, 1.0 / norm, cols);
            continue;
        }
        // Fallback: replace row i by a canonical basis vector orthogonalised against all
        // earlier rows. With cols ≥ rows such a vector always exists.
        let mut chosen = false;
        for p in 0..cols {
            let mut e_p = vec![0.0_f64; cols];
            e_p[p] = 1.0;
            for j in 0..i {
                let mut dot = 0.0_f64;
                for (k, v) in e_p.iter().enumerate() {
                    dot += v * a[j * cols + k];
                }
                for k in 0..cols {
                    e_p[k] -= dot * a[j * cols + k];
                }
            }
            let res = e_p.iter().map(|v| v * v).sum::<f64>().sqrt();
            if res > 1e-12 {
                let inv = 1.0 / res;
                for (k, v) in e_p.iter().enumerate() {
                    a[i * cols + k] = v * inv;
                }
                chosen = true;
                break;
            }
        }
        if !chosen {
            for k in 0..cols {
                a[i * cols + k] = 0.0;
            }
        }
    }
}

fn row_dot(a: &[f64], i: usize, j: usize, cols: usize) -> f64 {
    (0..cols).map(|k| a[i * cols + k] * a[j * cols + k]).sum()
}

fn row_axpy(a: &mut [f64], i: usize, j: usize, alpha: f64, cols: usize) {
    for k in 0..cols {
        a[i * cols + k] += alpha * a[j * cols + k];
    }
}

fn row_norm(a: &[f64], i: usize, cols: usize) -> f64 {
    (0..cols)
        .map(|k| a[i * cols + k].powi(2))
        .sum::<f64>()
        .sqrt()
}

fn row_scale(a: &mut [f64], i: usize, s: f64, cols: usize) {
    for k in 0..cols {
        a[i * cols + k] *= s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg(in_f: usize, out_f: usize, rank: usize, alpha: f64) -> OloraConfig {
        OloraConfig {
            in_features: in_f,
            out_features: out_f,
            rank,
            alpha,
        }
    }

    #[test]
    fn a_orthonormal_after_init() {
        let cfg = default_cfg(8, 5, 3, 6.0);
        let adapter = OloraAdapter::new(cfg, 7).unwrap();
        assert!(
            adapter.is_a_orthonormal(1e-9),
            "rows of A must be orthonormal after Gram-Schmidt"
        );
    }

    #[test]
    fn a_reproducible_across_seeds() {
        let cfg = default_cfg(7, 5, 3, 4.0);
        let a1 = OloraAdapter::new(cfg.clone(), 42).unwrap();
        let a2 = OloraAdapter::new(cfg, 42).unwrap();
        assert_eq!(a1.a, a2.a);
    }

    #[test]
    fn initial_forward_is_zero_with_zero_b() {
        let cfg = default_cfg(6, 4, 2, 4.0);
        let adapter = OloraAdapter::new(cfg, 11).unwrap();
        let x: Vec<f64> = (0..6).map(|i| i as f64 - 2.5).collect();
        let y = adapter.forward(&x).unwrap();
        assert_eq!(y.len(), 4);
        for &v in &y {
            assert!(v.abs() < 1e-15, "expected zero output, got {v}");
        }
    }

    #[test]
    fn forward_dimensions_correct() {
        let cfg = default_cfg(7, 9, 3, 6.0);
        let mut adapter = OloraAdapter::new(cfg, 11).unwrap();
        for (i, b) in adapter.b.iter_mut().enumerate() {
            *b = (i as f64 + 1.0) * 0.05;
        }
        let x = vec![1.0_f64; 7];
        let y = adapter.forward(&x).unwrap();
        assert_eq!(y.len(), 9);
    }

    #[test]
    fn backward_shapes_correct() {
        let cfg = default_cfg(5, 4, 2, 4.0);
        let adapter = OloraAdapter::new(cfg, 3).unwrap();
        let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3];
        let grad_y = vec![0.1_f64, -0.2, 0.3, 0.4];
        let (grad_a, grad_b) = adapter.backward(&x, &grad_y).unwrap();
        assert_eq!(grad_a.len(), 2 * 5);
        assert_eq!(grad_b.len(), 4 * 2);
    }

    fn loss_at(a: &OloraAdapter, x: &[f64], gy: &[f64]) -> f64 {
        gy.iter()
            .zip(a.forward(x).unwrap().iter())
            .map(|(g, y)| g * y)
            .sum()
    }

    #[test]
    fn backward_matches_finite_differences() {
        let cfg = default_cfg(4, 3, 2, 4.0);
        let mut adapter = OloraAdapter::new(cfg, 99).unwrap();
        for (i, b) in adapter.b.iter_mut().enumerate() {
            *b = 0.1 * (i as f64 + 1.0);
        }
        adapter.a[0] += 0.05;
        adapter.a[3] -= 0.07;
        let x = vec![0.5_f64, -1.0, 0.25, 0.75];
        let gy = vec![1.0_f64, -0.5, 0.25];
        let (grad_a, grad_b) = adapter.backward(&x, &gy).unwrap();
        let eps = 1e-6_f64;
        for (k, &g_k) in grad_b.iter().enumerate() {
            let s = adapter.b[k];
            adapter.b[k] = s + eps;
            let lp = loss_at(&adapter, &x, &gy);
            adapter.b[k] = s - eps;
            let lm = loss_at(&adapter, &x, &gy);
            adapter.b[k] = s;
            assert!(((lp - lm) / (2.0 * eps) - g_k).abs() < 1e-5, "B[{k}]");
        }
        for (k, &g_k) in grad_a.iter().enumerate() {
            let s = adapter.a[k];
            adapter.a[k] = s + eps;
            let lp = loss_at(&adapter, &x, &gy);
            adapter.a[k] = s - eps;
            let lm = loss_at(&adapter, &x, &gy);
            adapter.a[k] = s;
            assert!(((lp - lm) / (2.0 * eps) - g_k).abs() < 1e-5, "A[{k}]");
        }
    }

    #[test]
    fn sgd_reduces_loss() {
        let mut adapter = OloraAdapter::new(default_cfg(6, 4, 2, 4.0), 21).unwrap();
        let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3, 0.75];
        let target = {
            let mut probe = adapter.clone();
            for (i, b) in probe.b.iter_mut().enumerate() {
                *b = 0.4 * (i as f64 + 1.0);
            }
            probe.forward(&x).unwrap()
        };
        let mse = |a: &OloraAdapter| -> f64 {
            a.forward(&x)
                .unwrap()
                .iter()
                .zip(target.iter())
                .map(|(p, q)| (p - q).powi(2))
                .sum()
        };
        let initial = mse(&adapter);
        for _ in 0..200 {
            let y = adapter.forward(&x).unwrap();
            let gy: Vec<f64> = y.iter().zip(target.iter()).map(|(p, q)| p - q).collect();
            let (ga, gb) = adapter.backward(&x, &gy).unwrap();
            adapter.apply_grads(&ga, &gb, 0.02).unwrap();
        }
        let final_loss = mse(&adapter);
        assert!(
            final_loss * 10.0 < initial,
            "loss {final_loss} should drop >10x from {initial}"
        );
    }

    #[test]
    fn a_loses_orthonormality_after_updates() {
        let cfg = default_cfg(6, 5, 3, 4.0);
        let mut adapter = OloraAdapter::new(cfg, 33).unwrap();
        assert!(adapter.is_a_orthonormal(1e-9));
        // Apply a single hand-crafted grad_a that we know is non-trivial — this directly
        // demonstrates that the adapter is "actually trainable" (the update path mutates
        // A) and that orthonormality is not preserved by a generic SGD step.
        let grad_a: Vec<f64> = (0..adapter.a.len())
            .map(|i| 0.1 * (i as f64 + 1.0))
            .collect();
        let grad_b = vec![0.0_f64; adapter.b.len()];
        adapter.apply_grads(&grad_a, &grad_b, 0.5).unwrap();
        assert!(
            !adapter.is_a_orthonormal(1e-6),
            "A should drift from orthonormality once it is updated"
        );
    }

    #[test]
    fn invalid_configs_rejected() {
        for (i, o, r) in [(0, 4, 2), (4, 0, 2), (4, 4, 0)] {
            assert!(matches!(
                OloraAdapter::new(default_cfg(i, o, r, 1.0), 0),
                Err(PeftError::EmptyInput)
            ));
        }
        // rank > in_features (orthonormality impossible) and rank > min(in,out)
        for (i, o, r) in [(3, 8, 5), (8, 3, 5)] {
            assert!(matches!(
                OloraAdapter::new(default_cfg(i, o, r, 1.0), 0),
                Err(PeftError::RankTooLarge { .. })
            ));
        }
    }

    #[test]
    fn alpha_zero_produces_zero_forward() {
        let mut adapter = OloraAdapter::new(default_cfg(5, 4, 2, 0.0), 77).unwrap();
        for (i, b) in adapter.b.iter_mut().enumerate() {
            *b = 0.1 * (i as f64 + 1.0);
        }
        let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3];
        let y = adapter.forward(&x).unwrap();
        for &v in &y {
            assert!(v.abs() < 1e-15, "α=0 must zero out adapter, got {v}");
        }
    }

    #[test]
    fn dim_mismatch_rejected() {
        let mut adapter = OloraAdapter::new(default_cfg(5, 3, 2, 2.0), 0).unwrap();
        let dm = |r: PeftResult<Vec<f64>>| matches!(r, Err(PeftError::DimensionMismatch { .. }));
        assert!(dm(adapter.forward(&[1.0, 2.0, 3.0])));
        assert!(matches!(
            adapter.backward(&[0.1_f64; 5], &[0.1_f64; 2]),
            Err(PeftError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            adapter.backward(&[0.1_f64; 4], &[0.1_f64; 3]),
            Err(PeftError::DimensionMismatch { .. })
        ));
        for (ga, gb) in [
            (&[0.0_f64; 5][..], &[0.0_f64; 6][..]),
            (&[0.0; 10], &[0.0; 5]),
        ] {
            assert!(matches!(
                adapter.apply_grads(ga, gb, 0.1),
                Err(PeftError::DimensionMismatch { .. })
            ));
        }
    }

    #[test]
    fn gram_schmidt_handles_near_zero_rows() {
        // Third row is a near-zero multiple of e_0. MGS must fall back to a basis
        // replacement (deterministic, no NaN, no panic) and still produce orthonormal rows.
        let mut a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1e-300, 0.0, 0.0];
        modified_gram_schmidt_rows(&mut a, 3, 3);
        for i in 0..3 {
            let n: f64 = (0..3).map(|k| a[i * 3 + k].powi(2)).sum::<f64>().sqrt();
            assert!((n - 1.0).abs() < 1e-9, "row {i} norm {n}");
        }
        for i in 0..3 {
            for j in (i + 1)..3 {
                let dot: f64 = (0..3).map(|k| a[i * 3 + k] * a[j * 3 + k]).sum();
                assert!(dot.abs() < 1e-9, "rows {i},{j} dot={dot}");
            }
        }
    }

    #[test]
    fn n_trainable_counts_a_plus_b() {
        let adapter = OloraAdapter::new(default_cfg(8, 12, 4, 8.0), 0).unwrap();
        assert_eq!(adapter.n_trainable(), 4 * (8 + 12));
    }

    #[test]
    fn scale_alpha_over_rank_applied() {
        let mut a1 = OloraAdapter::new(default_cfg(5, 3, 2, 4.0), 33).unwrap();
        let mut a2 = OloraAdapter::new(default_cfg(5, 3, 2, 8.0), 33).unwrap();
        let b_seed: Vec<f64> = (0..a1.b.len()).map(|i| 0.1 * (i as f64 + 1.0)).collect();
        a1.b.copy_from_slice(&b_seed);
        a2.b.copy_from_slice(&b_seed);
        let x = vec![0.5_f64, -0.25, 1.0, -1.5, 0.3];
        let y1 = a1.forward(&x).unwrap();
        let y2 = a2.forward(&x).unwrap();
        for (v1, v2) in y1.iter().zip(y2.iter()) {
            assert!((2.0 * v1 - v2).abs() < 1e-12);
        }
        assert!((a1.scale() - 2.0).abs() < 1e-15);
        assert!((a2.scale() - 4.0).abs() < 1e-15);
    }
}
