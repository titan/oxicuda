//! LoHa — Low-rank Hadamard product adapter.
//!
//! Reference: Hyeon-Woo, N., Ye-Bin, M., & Oh, T.-H. (2022). *FedPara: Low-rank Hadamard
//! Product for Communication-Efficient Federated Learning*.
//! <https://arxiv.org/abs/2108.06098>. The same construction was popularised as a LoRA
//! variant by Kohaku-Blueleaf (2023) in the LyCORIS package.
//!
//! LoHa builds the weight delta as the element-wise (Hadamard) product of two
//! independent rank-`r` factor pairs:
//!
//! ```text
//!   ΔW = s · ((B₁ · A₁) ⊙ (B₂ · A₂)),     s = α / rank
//! ```
//!
//! where `A_i ∈ ℝ^{r × in}`, `B_i ∈ ℝ^{out × r}`, and `⊙` is the elementwise product on
//! the resulting `out × in` matrices. Although `B₁A₁` and `B₂A₂` are each rank-`r`, their
//! Hadamard product can have rank up to `r²`, which gives LoHa strictly higher expressivity
//! than vanilla LoRA at the same `rank` budget while keeping a `4 · rank · (in + out)`
//! parameter cost.
//!
//! ## Closed-form gradient
//!
//! With `P₁ = B₁A₁`, `P₂ = B₂A₂`, `M = P₁ ⊙ P₂`, and upstream `grad_y = ∂L/∂y`,
//!
//! ```text
//!   dM[o,i]   = s · grad_y[o] · x[i]
//!   dP₁[o,i]  = dM[o,i] · P₂[o,i]      (Hadamard chain rule)
//!   dP₂[o,i]  = dM[o,i] · P₁[o,i]
//!   dB₁       = dP₁ · A₁ᵀ              (shape out × rank)
//!   dA₁       = B₁ᵀ · dP₁              (shape rank × in)
//!   dB₂       = dP₂ · A₂ᵀ
//!   dA₂       = B₂ᵀ · dP₂
//! ```
//!
//! Both factor pairs are trainable. Like vanilla LoRA, `B₁ = B₂ = 0` at init so the adapter
//! contributes zero to the base model at training start.

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

/// Gradient bundle returned by [`LoHaAdapter::backward`]: `(dA1, dB1, dA2, dB2)`.
pub type LoHaGrads = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// Hyper-parameter bundle for a single LoHa adapter.
#[derive(Debug, Clone)]
pub struct LoHaConfig {
    /// Input feature count (column count of each `A_i`).
    pub in_features: usize,
    /// Output feature count (row count of each `B_i`).
    pub out_features: usize,
    /// Low-rank dimension `r` shared by all four factors.
    pub rank: usize,
    /// Global scaling factor `α`. The effective multiplier is `s = α / rank`.
    pub alpha: f64,
}

impl LoHaConfig {
    /// Effective scale `s = α / rank`.
    #[must_use]
    pub fn scale(&self) -> f64 {
        if self.rank == 0 {
            0.0
        } else {
            self.alpha / self.rank as f64
        }
    }

    /// Validate the configuration without constructing an adapter.
    ///
    /// # Errors
    ///
    /// - [`PeftError::EmptyInput`] if any dimension is zero.
    /// - [`PeftError::RankTooLarge`] if `rank > min(in_features, out_features)`.
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

/// LoHa adapter holding two rank-`r` factor pairs.
///
/// All matrices are stored row-major:
/// - `a1`, `a2`: `[rank × in_features]`
/// - `b1`, `b2`: `[out_features × rank]`
#[derive(Debug, Clone)]
pub struct LoHaAdapter {
    /// First down-projection, row-major `[rank × in_features]`.
    pub a1: Vec<f64>,
    /// First up-projection, row-major `[out_features × rank]`. Zero-initialised.
    pub b1: Vec<f64>,
    /// Second down-projection, row-major `[rank × in_features]`.
    pub a2: Vec<f64>,
    /// Second up-projection, row-major `[out_features × rank]`. Zero-initialised.
    pub b2: Vec<f64>,
    /// Captured configuration.
    pub cfg: LoHaConfig,
}

impl LoHaAdapter {
    /// Build a fresh adapter.
    ///
    /// `A₁`, `A₂` are drawn from `N(0, 1/√in_features)` via paired Box-Muller
    /// ([`LcgRng::next_normal_pair`]). `B₁`, `B₂` are zero-initialised so the adapter
    /// contributes zero at the start of training (because `(B₁A₁) ⊙ (B₂A₂) = 0`).
    ///
    /// # Errors
    ///
    /// Forwards [`LoHaConfig::validate`] errors.
    pub fn new(cfg: LoHaConfig, rng_seed: u64) -> PeftResult<Self> {
        cfg.validate()?;
        let mut rng = LcgRng::new(rng_seed);
        let std = 1.0_f64 / (cfg.in_features as f64).sqrt();
        let n_a = cfg.rank * cfg.in_features;
        let a1 = sample_normal_vec(&mut rng, n_a, std);
        let a2 = sample_normal_vec(&mut rng, n_a, std);
        let b1 = vec![0.0_f64; cfg.out_features * cfg.rank];
        let b2 = vec![0.0_f64; cfg.out_features * cfg.rank];
        Ok(Self {
            a1,
            b1,
            a2,
            b2,
            cfg,
        })
    }

    /// Number of trainable parameters: `2 · (rank · in + out · rank)`.
    #[must_use]
    pub fn n_trainable(&self) -> usize {
        2 * self.cfg.rank * (self.cfg.in_features + self.cfg.out_features)
    }

    /// Effective scale `s = α / rank`.
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.cfg.scale()
    }

    /// Compute `y = s · M · x` with `M = (B₁A₁) ⊙ (B₂A₂)`.
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
        let p1 = ba_product(
            &self.b1,
            &self.a1,
            self.cfg.out_features,
            self.cfg.rank,
            self.cfg.in_features,
        );
        let p2 = ba_product(
            &self.b2,
            &self.a2,
            self.cfg.out_features,
            self.cfg.rank,
            self.cfg.in_features,
        );
        let s = self.scale();
        let out = self.cfg.out_features;
        let in_f = self.cfg.in_features;
        let mut y = vec![0.0_f64; out];
        for (o, y_o) in y.iter_mut().enumerate() {
            let row = o * in_f;
            let mut acc = 0.0_f64;
            for (i, x_i) in x.iter().enumerate() {
                acc += p1[row + i] * p2[row + i] * x_i;
            }
            *y_o = s * acc;
        }
        Ok(y)
    }

    /// Closed-form gradients with respect to `(a1, b1, a2, b2)`.
    ///
    /// Returns the four row-major gradient buffers in matching shapes:
    /// - `da1`: `[rank × in]`
    /// - `db1`: `[out × rank]`
    /// - `da2`: `[rank × in]`
    /// - `db2`: `[out × rank]`
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] when `x.len() != in_features` or
    /// `grad_y.len() != out_features`.
    pub fn backward(&self, x: &[f64], grad_y: &[f64]) -> PeftResult<LoHaGrads> {
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
        let s = self.scale();
        let r = self.cfg.rank;
        let in_f = self.cfg.in_features;
        let out = self.cfg.out_features;
        let p1 = ba_product(&self.b1, &self.a1, out, r, in_f);
        let p2 = ba_product(&self.b2, &self.a2, out, r, in_f);
        // dP₁[o,i] = s · grad_y[o] · x[i] · P₂[o,i],
        // dP₂[o,i] = s · grad_y[o] · x[i] · P₁[o,i].
        let mut dp1 = vec![0.0_f64; out * in_f];
        let mut dp2 = vec![0.0_f64; out * in_f];
        for (o, g_o) in grad_y.iter().enumerate() {
            let row = o * in_f;
            let scaled = s * g_o;
            for (i, x_i) in x.iter().enumerate() {
                let base = scaled * x_i;
                let idx = row + i;
                dp1[idx] = base * p2[idx];
                dp2[idx] = base * p1[idx];
            }
        }
        let (da1, db1) = grads_from_dp(&dp1, &self.a1, &self.b1, out, r, in_f);
        let (da2, db2) = grads_from_dp(&dp2, &self.a2, &self.b2, out, r, in_f);
        Ok((da1, db1, da2, db2))
    }

    /// SGD update across all four factors.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] when any gradient buffer disagrees with the
    /// adapter's expected shapes.
    pub fn apply_grads(
        &mut self,
        da1: &[f64],
        db1: &[f64],
        da2: &[f64],
        db2: &[f64],
        lr: f64,
    ) -> PeftResult<()> {
        let exp_a = self.cfg.rank * self.cfg.in_features;
        let exp_b = self.cfg.out_features * self.cfg.rank;
        if da1.len() != exp_a {
            return Err(PeftError::DimensionMismatch {
                expected: exp_a,
                got: da1.len(),
            });
        }
        if da2.len() != exp_a {
            return Err(PeftError::DimensionMismatch {
                expected: exp_a,
                got: da2.len(),
            });
        }
        if db1.len() != exp_b {
            return Err(PeftError::DimensionMismatch {
                expected: exp_b,
                got: db1.len(),
            });
        }
        if db2.len() != exp_b {
            return Err(PeftError::DimensionMismatch {
                expected: exp_b,
                got: db2.len(),
            });
        }
        for (p, g) in self.a1.iter_mut().zip(da1.iter()) {
            *p -= lr * g;
        }
        for (p, g) in self.b1.iter_mut().zip(db1.iter()) {
            *p -= lr * g;
        }
        for (p, g) in self.a2.iter_mut().zip(da2.iter()) {
            *p -= lr * g;
        }
        for (p, g) in self.b2.iter_mut().zip(db2.iter()) {
            *p -= lr * g;
        }
        Ok(())
    }
}

/// Sample `n` Gaussian draws scaled by `std`.
fn sample_normal_vec(rng: &mut LcgRng, n: usize, std: f64) -> Vec<f64> {
    let mut out = vec![0.0_f64; n];
    let mut i = 0;
    while i + 1 < n {
        let (u, v) = rng.next_normal_pair();
        out[i] = (u as f64) * std;
        out[i + 1] = (v as f64) * std;
        i += 2;
    }
    if i < n {
        out[i] = (rng.next_normal() as f64) * std;
    }
    out
}

/// `P = B · A` for row-major `B: [out × r]` and `A: [r × in]`.
fn ba_product(b: &[f64], a: &[f64], out: usize, r: usize, in_f: usize) -> Vec<f64> {
    let mut p = vec![0.0_f64; out * in_f];
    for o in 0..out {
        let b_row = o * r;
        let p_row = o * in_f;
        for k in 0..r {
            let b_ok = b[b_row + k];
            if b_ok == 0.0 {
                continue;
            }
            let a_row = k * in_f;
            for i in 0..in_f {
                p[p_row + i] += b_ok * a[a_row + i];
            }
        }
    }
    p
}

/// Given `dP: [out × in]`, compute `dA = Bᵀ · dP` and `dB = dP · Aᵀ`.
fn grads_from_dp(
    dp: &[f64],
    a: &[f64],
    b: &[f64],
    out: usize,
    r: usize,
    in_f: usize,
) -> (Vec<f64>, Vec<f64>) {
    // dA[k, i] = Σ_o B[o, k] · dP[o, i]
    let mut da = vec![0.0_f64; r * in_f];
    for k in 0..r {
        let row = k * in_f;
        for i in 0..in_f {
            let mut acc = 0.0_f64;
            for o in 0..out {
                acc += b[o * r + k] * dp[o * in_f + i];
            }
            da[row + i] = acc;
        }
    }
    // dB[o, k] = Σ_i dP[o, i] · A[k, i]
    let mut db = vec![0.0_f64; out * r];
    for o in 0..out {
        let row_b = o * r;
        let row_p = o * in_f;
        for k in 0..r {
            let row_a = k * in_f;
            let mut acc = 0.0_f64;
            for i in 0..in_f {
                acc += dp[row_p + i] * a[row_a + i];
            }
            db[row_b + k] = acc;
        }
    }
    (da, db)
}
