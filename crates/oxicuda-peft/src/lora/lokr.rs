//! LoKr — Low-rank Kronecker product adapter.
//!
//! Reference: Edalati, A., Tahaei, M., Kobyzev, I., Nia, V. P., Clark, J. J., & Rezagholizadeh,
//! M. (2022). *Krona: Parameter Efficient Tuning with Kronecker Adapter*.
//! <https://arxiv.org/abs/2212.10650>. The LoRA-flavoured rank-`r` variant used here is the
//! Kohaku-Blueleaf (2023) formulation popularised by the LyCORIS package.
//!
//! LoKr expresses the weight delta as a Kronecker product:
//!
//! ```text
//!   ΔW = s · (W₁ ⊗ W₂),     s = α / rank
//! ```
//!
//! with `W₁ ∈ ℝ^{m₁ × n₁}` stored directly (rank-1 / "full" small factor) and
//! `W₂ ∈ ℝ^{m₂ × n₂}` further factorised as `W₂ = B · A`, `B ∈ ℝ^{m₂ × r}`,
//! `A ∈ ℝ^{r × n₂}`. The Kronecker product yields an `out × in = (m₁ · m₂) × (n₁ · n₂)`
//! dense effective matrix, but the actual parameter cost is only
//! `m₁ · n₁ + r · (m₂ + n₂)`, which is the source of LoKr's compression.
//!
//! ## Efficient block-form forward
//!
//! Reshape `x ∈ ℝ^{n₁ · n₂}` row-major as `X ∈ ℝ^{n₁ × n₂}` with `X[i₁, i₂] = x[i₁ n₂ + i₂]`.
//! By the Kronecker matrix-vector identity,
//!
//! ```text
//!   Y = W₁ · X · W₂ᵀ       (shape m₁ × m₂)
//!   y[o₁ m₂ + o₂] = s · Y[o₁, o₂]
//! ```
//!
//! ## Closed-form gradients
//!
//! With `G[o₁, o₂] = grad_y[o₁ m₂ + o₂]`, `dL/dY = s · G`, and `Y = W₁ · X · W₂ᵀ`:
//!
//! ```text
//!   dW₁ = s · G · W₂ · Xᵀ                 (shape m₁ × n₁)
//!   dW₂ = s · Gᵀ · W₁ · X                 (shape m₂ × n₂)
//! ```
//!
//! Decomposing `W₂ = B · A`:
//!
//! ```text
//!   dB = dW₂ · Aᵀ                         (shape m₂ × r)
//!   dA = Bᵀ · dW₂                         (shape r × n₂)
//! ```
//!
//! At init `B = 0` so the adapter contributes zero (mirroring the LoRA convention).

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

/// Hyper-parameter bundle for a single LoKr adapter.
#[derive(Debug, Clone)]
pub struct LoKrConfig {
    /// Input feature count. Must equal `n1 * n2`.
    pub in_features: usize,
    /// Output feature count. Must equal `m1 * m2`.
    pub out_features: usize,
    /// Row count of the rank-1 factor `W₁`.
    pub m1: usize,
    /// Row count of the low-rank factor `W₂`.
    pub m2: usize,
    /// Column count of the rank-1 factor `W₁`.
    pub n1: usize,
    /// Column count of the low-rank factor `W₂`.
    pub n2: usize,
    /// Inner low rank of the `W₂ = B · A` factorisation.
    pub rank: usize,
    /// Global scaling factor `α`; effective scale is `s = α / rank`.
    pub alpha: f64,
}

impl LoKrConfig {
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
    /// - [`PeftError::DimensionMismatch`] if `m1 * m2 != out_features` or
    ///   `n1 * n2 != in_features`.
    /// - [`PeftError::RankTooLarge`] if `rank > min(m2, n2)`.
    pub fn validate(&self) -> PeftResult<()> {
        if self.in_features == 0
            || self.out_features == 0
            || self.m1 == 0
            || self.m2 == 0
            || self.n1 == 0
            || self.n2 == 0
            || self.rank == 0
        {
            return Err(PeftError::EmptyInput);
        }
        if self.m1 * self.m2 != self.out_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.out_features,
                got: self.m1 * self.m2,
            });
        }
        if self.n1 * self.n2 != self.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.in_features,
                got: self.n1 * self.n2,
            });
        }
        let dim = self.m2.min(self.n2);
        if self.rank > dim {
            return Err(PeftError::RankTooLarge {
                rank: self.rank,
                dim,
            });
        }
        Ok(())
    }
}

/// LoKr adapter storing the rank-1 factor `W₁` directly and the rank-`r` factorisation of `W₂`.
///
/// Row-major layouts:
/// - `w1`: `[m1 × n1]`
/// - `a`:  `[rank × n2]`
/// - `b`:  `[m2 × rank]`
///
/// `W₂` is reconstructed on demand as `B · A`.
#[derive(Debug, Clone)]
pub struct LoKrAdapter {
    /// Rank-1 / "full" small Kronecker factor, row-major `[m1 × n1]`.
    pub w1: Vec<f64>,
    /// Down-projection of `W₂`, row-major `[rank × n2]`.
    pub a: Vec<f64>,
    /// Up-projection of `W₂`, row-major `[m2 × rank]`. Zero-initialised.
    pub b: Vec<f64>,
    /// Captured configuration.
    pub cfg: LoKrConfig,
}

impl LoKrAdapter {
    /// Build a fresh adapter.
    ///
    /// `W₁ ~ N(0, 1/√n₁)`, `A ~ N(0, 1/√n₂)`, `B = 0` so the initial adapter is a no-op
    /// (`W₂ = B · A = 0`).
    ///
    /// # Errors
    ///
    /// Forwards [`LoKrConfig::validate`] errors.
    pub fn new(cfg: LoKrConfig, rng_seed: u64) -> PeftResult<Self> {
        cfg.validate()?;
        let mut rng = LcgRng::new(rng_seed);
        let std_w1 = 1.0_f64 / (cfg.n1 as f64).sqrt();
        let std_a = 1.0_f64 / (cfg.n2 as f64).sqrt();
        let w1 = sample_normal_vec(&mut rng, cfg.m1 * cfg.n1, std_w1);
        let a = sample_normal_vec(&mut rng, cfg.rank * cfg.n2, std_a);
        let b = vec![0.0_f64; cfg.m2 * cfg.rank];
        Ok(Self { w1, a, b, cfg })
    }

    /// Number of trainable parameters: `m₁ n₁ + r · (m₂ + n₂)`.
    #[must_use]
    pub fn n_trainable(&self) -> usize {
        self.cfg.m1 * self.cfg.n1 + self.cfg.rank * (self.cfg.m2 + self.cfg.n2)
    }

    /// Effective scale `s = α / rank`.
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.cfg.scale()
    }

    /// Materialise `W₂ = B · A` as a fresh `[m2 × n2]` row-major matrix.
    #[must_use]
    pub fn w2(&self) -> Vec<f64> {
        ba_product(&self.b, &self.a, self.cfg.m2, self.cfg.rank, self.cfg.n2)
    }

    /// Forward via the efficient block form `Y = W₁ · X · W₂ᵀ`.
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
        let s = self.scale();
        let w2 = self.w2();
        // X is x reshaped as [n1, n2]. Compute T = W₁ · X (shape m1 × n2), then
        // Y = T · W₂ᵀ (shape m1 × m2).
        let t = mat_mul(&self.w1, x, self.cfg.m1, self.cfg.n1, self.cfg.n2);
        // Y[o1, o2] = Σ_i2 T[o1, i2] · W₂[o2, i2]
        let mut y = vec![0.0_f64; self.cfg.m1 * self.cfg.m2];
        for o1 in 0..self.cfg.m1 {
            let trow = o1 * self.cfg.n2;
            for o2 in 0..self.cfg.m2 {
                let wrow = o2 * self.cfg.n2;
                let mut acc = 0.0_f64;
                for i2 in 0..self.cfg.n2 {
                    acc += t[trow + i2] * w2[wrow + i2];
                }
                // Output index in row-major (m1, m2) flattening.
                y[o1 * self.cfg.m2 + o2] = s * acc;
            }
        }
        Ok(y)
    }

    /// Closed-form gradients with respect to `(w1, a, b)`.
    ///
    /// Returns three row-major gradient buffers:
    /// - `dw1`: `[m1 × n1]`
    /// - `da`:  `[rank × n2]`
    /// - `db`:  `[m2 × rank]`
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] when `x.len() != in_features` or
    /// `grad_y.len() != out_features`.
    pub fn backward(
        &self,
        x: &[f64],
        grad_y: &[f64],
    ) -> PeftResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
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
        let w2 = self.w2();
        // dW₁[o1, i1] = s · Σ_{o2, i2} G[o1, o2] · W₂[o2, i2] · X[i1, i2]
        // Compute H[o1, i2] = Σ_{o2} G[o1, o2] · W₂[o2, i2], then dW₁ = s · H · Xᵀ.
        let mut h = vec![0.0_f64; self.cfg.m1 * self.cfg.n2];
        for o1 in 0..self.cfg.m1 {
            let hrow = o1 * self.cfg.n2;
            let grow = o1 * self.cfg.m2;
            for i2 in 0..self.cfg.n2 {
                let mut acc = 0.0_f64;
                for o2 in 0..self.cfg.m2 {
                    acc += grad_y[grow + o2] * w2[o2 * self.cfg.n2 + i2];
                }
                h[hrow + i2] = acc;
            }
        }
        // dW₁[o1, i1] = s · Σ_{i2} H[o1, i2] · X[i1, i2]
        let mut dw1 = vec![0.0_f64; self.cfg.m1 * self.cfg.n1];
        for o1 in 0..self.cfg.m1 {
            let hrow = o1 * self.cfg.n2;
            let drow = o1 * self.cfg.n1;
            for i1 in 0..self.cfg.n1 {
                let xrow = i1 * self.cfg.n2;
                let mut acc = 0.0_f64;
                for i2 in 0..self.cfg.n2 {
                    acc += h[hrow + i2] * x[xrow + i2];
                }
                dw1[drow + i1] = s * acc;
            }
        }
        // dW₂[o2, i2] = s · Σ_{o1, i1} G[o1, o2] · W₁[o1, i1] · X[i1, i2]
        // Compute T = W₁ · X (shape m1 × n2) — same intermediate as forward.
        let t = mat_mul(&self.w1, x, self.cfg.m1, self.cfg.n1, self.cfg.n2);
        // dW₂[o2, i2] = s · Σ_{o1} G[o1, o2] · T[o1, i2]
        let mut dw2 = vec![0.0_f64; self.cfg.m2 * self.cfg.n2];
        for o2 in 0..self.cfg.m2 {
            let drow = o2 * self.cfg.n2;
            for i2 in 0..self.cfg.n2 {
                let mut acc = 0.0_f64;
                for o1 in 0..self.cfg.m1 {
                    acc += grad_y[o1 * self.cfg.m2 + o2] * t[o1 * self.cfg.n2 + i2];
                }
                dw2[drow + i2] = s * acc;
            }
        }
        // dB[o2, k] = Σ_{i2} dW₂[o2, i2] · A[k, i2]
        let mut db = vec![0.0_f64; self.cfg.m2 * self.cfg.rank];
        for o2 in 0..self.cfg.m2 {
            let drow = o2 * self.cfg.n2;
            let brow = o2 * self.cfg.rank;
            for k in 0..self.cfg.rank {
                let arow = k * self.cfg.n2;
                let mut acc = 0.0_f64;
                for i2 in 0..self.cfg.n2 {
                    acc += dw2[drow + i2] * self.a[arow + i2];
                }
                db[brow + k] = acc;
            }
        }
        // dA[k, i2] = Σ_{o2} B[o2, k] · dW₂[o2, i2]
        let mut da = vec![0.0_f64; self.cfg.rank * self.cfg.n2];
        for k in 0..self.cfg.rank {
            let arow = k * self.cfg.n2;
            for i2 in 0..self.cfg.n2 {
                let mut acc = 0.0_f64;
                for o2 in 0..self.cfg.m2 {
                    acc += self.b[o2 * self.cfg.rank + k] * dw2[o2 * self.cfg.n2 + i2];
                }
                da[arow + i2] = acc;
            }
        }
        Ok((dw1, da, db))
    }

    /// SGD update across `w1`, `a`, and `b`.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] when any gradient buffer disagrees with the
    /// adapter's expected shapes.
    pub fn apply_grads(&mut self, dw1: &[f64], da: &[f64], db: &[f64], lr: f64) -> PeftResult<()> {
        let exp_w1 = self.cfg.m1 * self.cfg.n1;
        let exp_a = self.cfg.rank * self.cfg.n2;
        let exp_b = self.cfg.m2 * self.cfg.rank;
        if dw1.len() != exp_w1 {
            return Err(PeftError::DimensionMismatch {
                expected: exp_w1,
                got: dw1.len(),
            });
        }
        if da.len() != exp_a {
            return Err(PeftError::DimensionMismatch {
                expected: exp_a,
                got: da.len(),
            });
        }
        if db.len() != exp_b {
            return Err(PeftError::DimensionMismatch {
                expected: exp_b,
                got: db.len(),
            });
        }
        for (p, g) in self.w1.iter_mut().zip(dw1.iter()) {
            *p -= lr * g;
        }
        for (p, g) in self.a.iter_mut().zip(da.iter()) {
            *p -= lr * g;
        }
        for (p, g) in self.b.iter_mut().zip(db.iter()) {
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

/// `P = B · A` for row-major `B: [rows × inner]` and `A: [inner × cols]`.
fn ba_product(b: &[f64], a: &[f64], rows: usize, inner: usize, cols: usize) -> Vec<f64> {
    let mut p = vec![0.0_f64; rows * cols];
    for r in 0..rows {
        let b_row = r * inner;
        let p_row = r * cols;
        for k in 0..inner {
            let b_rk = b[b_row + k];
            if b_rk == 0.0 {
                continue;
            }
            let a_row = k * cols;
            for c in 0..cols {
                p[p_row + c] += b_rk * a[a_row + c];
            }
        }
    }
    p
}

/// `T = W · X` with `W: [m × n]` row-major and `X: [n × p]` row-major.
fn mat_mul(w: &[f64], x: &[f64], m: usize, n: usize, p: usize) -> Vec<f64> {
    let mut t = vec![0.0_f64; m * p];
    for i in 0..m {
        let w_row = i * n;
        let t_row = i * p;
        for k in 0..n {
            let w_ik = w[w_row + k];
            if w_ik == 0.0 {
                continue;
            }
            let x_row = k * p;
            for j in 0..p {
                t[t_row + j] += w_ik * x[x_row + j];
            }
        }
    }
    t
}
