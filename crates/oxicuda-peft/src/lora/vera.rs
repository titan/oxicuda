//! VeRA — Vector-based Random Adaptation
//!
//! Reference: Kopiczko, D., Blankevoort, T., & Asano, Y. M. (2024).
//! *VeRA: Vector-based Random Adaptation*. International Conference on Learning
//! Representations (ICLR). <https://arxiv.org/abs/2310.11454>
//!
//! VeRA freezes a pair of shared random projection matrices `A ∈ ℝ^{r×in}` and
//! `B ∈ ℝ^{out×r}` (identical across every layer in a network) and only learns two
//! tiny scaling vectors per layer:
//!
//! ```text
//!   y = α · diag(d_b) · B · diag(d_d) · A · x
//! ```
//!
//! where `d_d ∈ ℝ^{r}` and `d_b ∈ ℝ^{out}` are the only trainable parameters.
//! Trainable parameter count per layer is `r + out`, which is roughly an order of
//! magnitude smaller than vanilla LoRA's `r · (in + out)`.
//!
//! ## Initialisation
//!
//! - `A` and `B` are drawn with Kaiming-uniform from a *fixed* seed and frozen for the
//!   entire training run.
//! - `d_d` is initialised to `init_scale_d · 1` so that the random projection chain has
//!   a non-trivial starting magnitude.
//! - `d_b` is initialised to `init_scale_b · 1` (default `0`) so that the adapter
//!   starts as an identity perturbation of the base layer, matching the LoRA pattern.
//!
//! ## Closed-form gradients
//!
//! Letting `t1 = A · x`, `t2 = d_d ⊙ t1`, `t3 = B · t2` and `y = α · (d_b ⊙ t3)`:
//!
//! - `∂L / ∂d_b[i]   = α · grad_y[i] · t3[i]`
//! - `∂L / ∂d_d[j]   = α · t1[j] · Σ_i grad_y[i] · d_b[i] · B[i,j]`
//!
//! These are computed without storing any intermediate matrices beyond `t1` and `t3`.

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

/// Hyper-parameter bundle for a single VeRA adapter.
#[derive(Debug, Clone)]
pub struct VeraConfig {
    /// Input feature count.
    pub in_dim: usize,
    /// Output feature count.
    pub out_dim: usize,
    /// Rank of the shared random projection.
    pub rank: usize,
    /// Global scale multiplier `α` (analogous to LoRA's `α`).
    pub alpha: f64,
    /// Initial value of every entry of `d_d`.
    pub init_scale_d: f64,
    /// Initial value of every entry of `d_b`.
    pub init_scale_b: f64,
    /// Seed used to draw the *frozen* `A` and `B` (when calling
    /// [`VeraSharedRandom::new`]).
    pub seed: u64,
}

/// Frozen, layer-independent random projection pair `(A, B)` shared across all VeRA
/// adapters.
///
/// A is row-major `[rank × in_dim]`. B is row-major `[out_dim × rank]`. Once
/// constructed, neither field is ever mutated by the crate.
#[derive(Debug, Clone)]
pub struct VeraSharedRandom {
    /// `rank × in_dim` random matrix (row-major).
    pub a: Vec<f64>,
    /// `out_dim × rank` random matrix (row-major).
    pub b: Vec<f64>,
    /// Cached `in_dim` (for shape checks).
    pub in_dim: usize,
    /// Cached `out_dim` (for shape checks).
    pub out_dim: usize,
    /// Cached `rank`.
    pub rank: usize,
}

impl VeraSharedRandom {
    /// Build the shared random pair `(A, B)` from a fixed `seed` using a Kaiming-uniform
    /// distribution.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::EmptyInput`] if any dimension is zero and
    /// [`PeftError::RankTooLarge`] if `rank > min(in_dim, out_dim)`.
    pub fn new(in_dim: usize, out_dim: usize, rank: usize, seed: u64) -> PeftResult<Self> {
        if in_dim == 0 || out_dim == 0 || rank == 0 {
            return Err(PeftError::EmptyInput);
        }
        if rank > in_dim.min(out_dim) {
            return Err(PeftError::RankTooLarge {
                rank,
                dim: in_dim.min(out_dim),
            });
        }
        let mut rng = LcgRng::new(seed);
        // Kaiming-uniform bounds for an `(out, in)` weight slab:
        //   bound = sqrt(6 / fan_in)
        let bound_a = (6.0_f64 / in_dim as f64).sqrt();
        let bound_b = (6.0_f64 / rank as f64).sqrt();

        let mut a = vec![0.0_f64; rank * in_dim];
        for v in a.iter_mut() {
            let u = rng.next_f32() as f64; // [0, 1)
            *v = (u * 2.0 - 1.0) * bound_a;
        }
        let mut b = vec![0.0_f64; out_dim * rank];
        for v in b.iter_mut() {
            let u = rng.next_f32() as f64;
            *v = (u * 2.0 - 1.0) * bound_b;
        }

        Ok(Self {
            a,
            b,
            in_dim,
            out_dim,
            rank,
        })
    }
}

/// A single per-layer VeRA adapter.
///
/// Only [`VeraAdapter::d_d`] and [`VeraAdapter::d_b`] are trainable. The frozen random
/// projection `A`, `B` lives in a separately-owned [`VeraSharedRandom`] that is passed
/// in at forward/backward time so a whole network of VeRA layers can share one copy.
#[derive(Debug, Clone)]
pub struct VeraAdapter {
    /// Configuration captured at construction time.
    pub config: VeraConfig,
    /// Trainable rank-vector, length `rank`.
    pub d_d: Vec<f64>,
    /// Trainable output-side vector, length `out_dim`.
    pub d_b: Vec<f64>,
}

impl VeraAdapter {
    /// Build a fresh adapter with `d_d = init_scale_d · 1` and `d_b = init_scale_b · 1`.
    ///
    /// # Errors
    ///
    /// Same validation as [`VeraSharedRandom::new`].
    pub fn new(cfg: VeraConfig) -> PeftResult<Self> {
        if cfg.in_dim == 0 || cfg.out_dim == 0 || cfg.rank == 0 {
            return Err(PeftError::EmptyInput);
        }
        if cfg.rank > cfg.in_dim.min(cfg.out_dim) {
            return Err(PeftError::RankTooLarge {
                rank: cfg.rank,
                dim: cfg.in_dim.min(cfg.out_dim),
            });
        }
        let d_d = vec![cfg.init_scale_d; cfg.rank];
        let d_b = vec![cfg.init_scale_b; cfg.out_dim];
        Ok(Self {
            config: cfg,
            d_d,
            d_b,
        })
    }

    /// Number of trainable parameters per layer, `r + out_dim`.
    #[must_use]
    pub fn n_trainable(&self) -> usize {
        self.config.rank + self.config.out_dim
    }

    /// Compute `y = α · diag(d_b) · B · diag(d_d) · A · x`.
    ///
    /// `x` must have length `in_dim`.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] if `x.len() != in_dim` or if
    /// `shared` shapes disagree with `self.config`.
    pub fn forward(&self, shared: &VeraSharedRandom, x: &[f64]) -> PeftResult<Vec<f64>> {
        self.check_shapes(shared, x)?;
        let (_, t3) = self.forward_internal(shared, x);
        let alpha = self.config.alpha;
        let y: Vec<f64> = self
            .d_b
            .iter()
            .zip(t3.iter())
            .map(|(db, t3i)| alpha * db * t3i)
            .collect();
        Ok(y)
    }

    /// Closed-form gradients with respect to `(d_d, d_b)` given upstream gradient
    /// `grad_y` of the loss w.r.t. the adapter output.
    ///
    /// Returns `(grad_d_d, grad_d_b)`. The frozen shared `A` and `B` are *not* updated
    /// in any way: they are only read.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] when any input has the wrong length.
    pub fn backward(
        &self,
        shared: &VeraSharedRandom,
        x: &[f64],
        grad_y: &[f64],
    ) -> PeftResult<(Vec<f64>, Vec<f64>)> {
        self.check_shapes(shared, x)?;
        if grad_y.len() != self.config.out_dim {
            return Err(PeftError::DimensionMismatch {
                expected: self.config.out_dim,
                got: grad_y.len(),
            });
        }
        let (t1, t3) = self.forward_internal(shared, x);
        let alpha = self.config.alpha;
        // ∂L/∂d_b[i] = α · grad_y[i] · t3[i]
        let grad_d_b: Vec<f64> = grad_y
            .iter()
            .zip(t3.iter())
            .map(|(g, t3i)| alpha * g * t3i)
            .collect();
        // ∂L/∂d_d[j] = α · t1[j] · Σ_i grad_y[i] · d_b[i] · B[i,j]
        // Pre-compute h[i] = grad_y[i] · d_b[i].
        let h: Vec<f64> = grad_y
            .iter()
            .zip(self.d_b.iter())
            .map(|(g, db)| g * db)
            .collect();
        let r = self.config.rank;
        let mut grad_d_d = vec![0.0_f64; r];
        for (j, (t1j, g_dd_j)) in t1.iter().zip(grad_d_d.iter_mut()).enumerate() {
            let mut acc = 0.0_f64;
            for (i, hi) in h.iter().enumerate() {
                acc += hi * shared.b[i * r + j];
            }
            *g_dd_j = alpha * acc * t1j;
        }
        Ok((grad_d_d, grad_d_b))
    }

    /// Run only the linear-algebra portion of the forward pass and return the
    /// intermediate tensors `(t1, t3)` used by both [`VeraAdapter::forward`] and
    /// [`VeraAdapter::backward`].
    fn forward_internal(&self, shared: &VeraSharedRandom, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let r = self.config.rank;
        let in_dim = self.config.in_dim;
        let out_dim = self.config.out_dim;
        // t1 = A · x, length r
        let mut t1 = vec![0.0_f64; r];
        for (j, t1j) in t1.iter_mut().enumerate() {
            let row_start = j * in_dim;
            let mut acc = 0.0_f64;
            for (k, xk) in x.iter().enumerate().take(in_dim) {
                acc += shared.a[row_start + k] * xk;
            }
            *t1j = acc;
        }
        // t2 = d_d ⊙ t1, length r (stored separately for clarity)
        let t2: Vec<f64> = self.d_d.iter().zip(t1.iter()).map(|(d, v)| d * v).collect();
        // t3 = B · t2, length out_dim
        let mut t3 = vec![0.0_f64; out_dim];
        for (i, t3i) in t3.iter_mut().enumerate() {
            let row_start = i * r;
            let mut acc = 0.0_f64;
            for (j, t2j) in t2.iter().enumerate() {
                acc += shared.b[row_start + j] * t2j;
            }
            *t3i = acc;
        }
        (t1, t3)
    }

    /// Cross-check `shared` and `x` against `self.config`.
    fn check_shapes(&self, shared: &VeraSharedRandom, x: &[f64]) -> PeftResult<()> {
        if shared.in_dim != self.config.in_dim
            || shared.out_dim != self.config.out_dim
            || shared.rank != self.config.rank
        {
            return Err(PeftError::DimensionMismatch {
                expected: self.config.in_dim + self.config.out_dim + self.config.rank,
                got: shared.in_dim + shared.out_dim + shared.rank,
            });
        }
        if shared.a.len() != self.config.rank * self.config.in_dim {
            return Err(PeftError::DimensionMismatch {
                expected: self.config.rank * self.config.in_dim,
                got: shared.a.len(),
            });
        }
        if shared.b.len() != self.config.out_dim * self.config.rank {
            return Err(PeftError::DimensionMismatch {
                expected: self.config.out_dim * self.config.rank,
                got: shared.b.len(),
            });
        }
        if x.len() != self.config.in_dim {
            return Err(PeftError::DimensionMismatch {
                expected: self.config.in_dim,
                got: x.len(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config(in_dim: usize, out_dim: usize, rank: usize) -> VeraConfig {
        VeraConfig {
            in_dim,
            out_dim,
            rank,
            alpha: 1.0,
            init_scale_d: 0.1,
            init_scale_b: 0.0,
            seed: 7,
        }
    }

    #[test]
    fn rejects_zero_in_dim() {
        let cfg = default_config(0, 4, 2);
        assert!(matches!(VeraAdapter::new(cfg), Err(PeftError::EmptyInput)));
    }

    #[test]
    fn rejects_zero_out_dim() {
        let cfg = default_config(4, 0, 2);
        assert!(matches!(VeraAdapter::new(cfg), Err(PeftError::EmptyInput)));
    }

    #[test]
    fn rejects_zero_rank() {
        let cfg = default_config(4, 4, 0);
        assert!(matches!(VeraAdapter::new(cfg), Err(PeftError::EmptyInput)));
    }

    #[test]
    fn rejects_rank_too_large() {
        let cfg = default_config(3, 4, 5);
        let err = VeraAdapter::new(cfg).expect_err("rank > min should fail");
        assert!(matches!(err, PeftError::RankTooLarge { .. }));
    }

    #[test]
    fn seed_zero_is_valid() {
        // Seeds are unconstrained; LcgRng's internal addition handles the zero case.
        let shared = VeraSharedRandom::new(4, 5, 2, 0).expect("seed=0 should be fine");
        assert_eq!(shared.a.len(), 2 * 4);
        assert_eq!(shared.b.len(), 5 * 2);
    }

    #[test]
    fn init_shapes_match_dims() {
        let cfg = default_config(5, 7, 3);
        let ad = VeraAdapter::new(cfg).unwrap();
        assert_eq!(ad.d_d.len(), 3);
        assert_eq!(ad.d_b.len(), 7);
        assert_eq!(ad.n_trainable(), 3 + 7);
    }

    #[test]
    fn deterministic_shared_for_fixed_seed() {
        let s1 = VeraSharedRandom::new(6, 8, 3, 42).unwrap();
        let s2 = VeraSharedRandom::new(6, 8, 3, 42).unwrap();
        assert_eq!(s1.a, s2.a);
        assert_eq!(s1.b, s2.b);
    }

    #[test]
    fn forward_output_dim_equals_out_dim() {
        let cfg = default_config(6, 9, 4);
        let shared = VeraSharedRandom::new(6, 9, 4, 11).unwrap();
        let ad = VeraAdapter::new(cfg).unwrap();
        let x: Vec<f64> = (0..6).map(|i| i as f64 * 0.1).collect();
        let y = ad.forward(&shared, &x).unwrap();
        assert_eq!(y.len(), 9);
    }

    #[test]
    fn zero_d_b_gives_zero_output() {
        let cfg = VeraConfig {
            in_dim: 4,
            out_dim: 5,
            rank: 2,
            alpha: 1.5,
            init_scale_d: 0.2,
            init_scale_b: 0.0, // zero d_b
            seed: 3,
        };
        let shared = VeraSharedRandom::new(4, 5, 2, 3).unwrap();
        let ad = VeraAdapter::new(cfg).unwrap();
        let x = vec![1.0_f64, -0.5, 0.25, 2.0];
        let y = ad.forward(&shared, &x).unwrap();
        for v in y {
            assert!(v.abs() < 1e-15);
        }
    }

    #[test]
    fn zero_d_d_gives_zero_output() {
        let cfg = VeraConfig {
            in_dim: 4,
            out_dim: 5,
            rank: 2,
            alpha: 1.5,
            init_scale_d: 0.0,
            init_scale_b: 0.5,
            seed: 13,
        };
        let shared = VeraSharedRandom::new(4, 5, 2, 13).unwrap();
        let ad = VeraAdapter::new(cfg).unwrap();
        let x = vec![1.0, -1.0, 2.0, 3.0];
        let y = ad.forward(&shared, &x).unwrap();
        for v in y {
            assert!(v.abs() < 1e-15);
        }
    }

    #[test]
    fn alpha_scales_output_linearly() {
        let make_cfg = |alpha: f64| VeraConfig {
            in_dim: 5,
            out_dim: 4,
            rank: 2,
            alpha,
            init_scale_d: 0.3,
            init_scale_b: 0.7,
            seed: 21,
        };
        let shared = VeraSharedRandom::new(5, 4, 2, 21).unwrap();
        let x = vec![0.1, -0.2, 0.3, -0.4, 0.5];
        let y1 = VeraAdapter::new(make_cfg(1.0))
            .unwrap()
            .forward(&shared, &x)
            .unwrap();
        let y2 = VeraAdapter::new(make_cfg(2.0))
            .unwrap()
            .forward(&shared, &x)
            .unwrap();
        for (a, b) in y1.iter().zip(y2.iter()) {
            assert!((2.0 * a - b).abs() < 1e-12, "α=2 must double y");
        }
    }

    #[test]
    fn backward_matches_finite_differences() {
        let cfg = VeraConfig {
            in_dim: 4,
            out_dim: 3,
            rank: 2,
            alpha: 0.5,
            init_scale_d: 0.6,
            init_scale_b: 0.4,
            seed: 99,
        };
        let shared = VeraSharedRandom::new(4, 3, 2, 99).unwrap();
        let mut ad = VeraAdapter::new(cfg.clone()).unwrap();
        // perturb d_d / d_b so we don't sit on degenerate values
        ad.d_d[0] = 0.7;
        ad.d_d[1] = -0.2;
        ad.d_b[0] = 0.3;
        ad.d_b[1] = -0.9;
        ad.d_b[2] = 1.2;
        let x = vec![0.5, -0.25, 1.0, -1.5];
        // Choose a fixed grad_y; treat L(d_d, d_b) = grad_y · y(d_d, d_b)
        let grad_y = vec![1.3, -0.7, 0.45];
        let (g_dd, g_db) = ad.backward(&shared, &x, &grad_y).unwrap();

        // Finite-difference w.r.t. d_d
        let eps = 1e-6_f64;
        for (j, &g) in g_dd.iter().enumerate() {
            let saved = ad.d_d[j];
            ad.d_d[j] = saved + eps;
            let yp = ad.forward(&shared, &x).unwrap();
            ad.d_d[j] = saved - eps;
            let ym = ad.forward(&shared, &x).unwrap();
            ad.d_d[j] = saved;
            let lp: f64 = grad_y.iter().zip(yp.iter()).map(|(a, b)| a * b).sum();
            let lm: f64 = grad_y.iter().zip(ym.iter()).map(|(a, b)| a * b).sum();
            let fd = (lp - lm) / (2.0 * eps);
            assert!((fd - g).abs() < 1e-7, "d_d[{j}] FD={fd} analytical={g}");
        }
        // Finite-difference w.r.t. d_b
        for (i, &g) in g_db.iter().enumerate() {
            let saved = ad.d_b[i];
            ad.d_b[i] = saved + eps;
            let yp = ad.forward(&shared, &x).unwrap();
            ad.d_b[i] = saved - eps;
            let ym = ad.forward(&shared, &x).unwrap();
            ad.d_b[i] = saved;
            let lp: f64 = grad_y.iter().zip(yp.iter()).map(|(a, b)| a * b).sum();
            let lm: f64 = grad_y.iter().zip(ym.iter()).map(|(a, b)| a * b).sum();
            let fd = (lp - lm) / (2.0 * eps);
            assert!((fd - g).abs() < 1e-7, "d_b[{i}] FD={fd} analytical={g}");
        }
    }

    #[test]
    fn shared_unchanged_after_forward_and_backward() {
        let cfg = default_config(4, 5, 2);
        let shared = VeraSharedRandom::new(4, 5, 2, 99).unwrap();
        let a_snap = shared.a.clone();
        let b_snap = shared.b.clone();
        let ad = VeraAdapter::new(cfg).unwrap();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let _ = ad.forward(&shared, &x).unwrap();
        let _ = ad
            .backward(&shared, &x, &[0.1, 0.2, 0.3, 0.4, 0.5])
            .unwrap();
        assert_eq!(shared.a, a_snap);
        assert_eq!(shared.b, b_snap);
    }

    #[test]
    fn dim_mismatch_in_forward_raises() {
        let cfg = default_config(4, 5, 2);
        let shared = VeraSharedRandom::new(4, 5, 2, 1).unwrap();
        let ad = VeraAdapter::new(cfg).unwrap();
        let bad_x = vec![1.0, 2.0, 3.0]; // length 3 ≠ 4
        assert!(matches!(
            ad.forward(&shared, &bad_x),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn dim_mismatch_in_backward_raises() {
        let cfg = default_config(4, 5, 2);
        let shared = VeraSharedRandom::new(4, 5, 2, 1).unwrap();
        let ad = VeraAdapter::new(cfg).unwrap();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let bad_gy = vec![1.0, 2.0]; // wrong length
        assert!(matches!(
            ad.backward(&shared, &x, &bad_gy),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }
}
