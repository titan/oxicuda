//! OFT — Orthogonal Fine-Tuning (Qiu et al. 2023).
//!
//! Reference: Qiu, Z., Liu, W., Feng, H., Xue, Y., Feng, Y., Liu, Z., Zhang, D.,
//! Weller, A., & Schölkopf, B. (2023). *Controlling Text-to-Image Diffusion by
//! Orthogonal Finetuning*. NeurIPS 2023. <https://arxiv.org/abs/2306.07280>
//!
//! OFT adapts a frozen weight matrix `W ∈ ℝ^{out × in}` by left-multiplying its
//! output dimension with a *learned orthogonal* matrix `R`:
//!
//! ```text
//!   W' = R · W ,        R ∈ O(out) ,   R Rᵀ = I .
//! ```
//!
//! Because `R` is orthogonal it preserves the pairwise angles (and therefore the
//! "hyperspherical energy") between the rows of `W`, which empirically keeps the
//! pretrained semantics far more stable than an unconstrained low-rank update.
//!
//! ## Cayley parametrisation
//!
//! An orthogonal matrix with determinant `+1` and no `−1` eigenvalue can be written
//! through the Cayley transform of a skew-symmetric matrix `Q = −Qᵀ`:
//!
//! ```text
//!   R = (I + Q)⁻¹ (I − Q) .
//! ```
//!
//! The trainable parameter is an arbitrary square matrix `S`; we skew-symmetrise it
//! with `Q = (S − Sᵀ) / 2` so that `R` is *exactly* orthogonal for any `S`. When
//! `S = 0`, `Q = 0` and `R = I`, so the adapter starts as the identity perturbation
//! of the base layer (mirroring the LoRA zero-init convention).
//!
//! ## Block-diagonal structure
//!
//! A full `out × out` orthogonal matrix has `O(out²)` parameters. OFT instead splits
//! the output dimension into `block_count` equal blocks of size `block_size` and
//! learns an independent orthogonal `block_size × block_size` matrix per block,
//! giving a block-diagonal `R` with only `block_count · block_size²` parameters. The
//! special case `block_count = 1` recovers the dense orthogonal transform.

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

/// Hyper-parameters for an [`OftLinear`] adapter.
#[derive(Debug, Clone)]
pub struct OftConfig {
    /// Number of input features (`in` dimension of `W`).
    pub in_features: usize,
    /// Number of output features (`out` dimension of `W`). Must be divisible by
    /// `block_count`.
    pub out_features: usize,
    /// Number of diagonal blocks. `1` ⇒ a single dense orthogonal transform.
    pub block_count: usize,
    /// If `true`, constrain every block to unit Frobenius-normalised `Q` so that the
    /// rotation angle stays bounded (the paper's "constrained" variant `ε`-ball).
    pub constrained: bool,
    /// Maximum allowed Frobenius norm of each block's `Q` when `constrained` is set.
    pub eps: f64,
}

/// Block-diagonal orthogonal fine-tuning of a single linear layer.
///
/// The base weight `w` (row-major `[out_features × in_features]`) is frozen; only the
/// per-block skew generators in [`OftLinear::s_blocks`] are trainable.
#[derive(Debug, Clone)]
pub struct OftLinear {
    /// Input feature count.
    pub in_features: usize,
    /// Output feature count.
    pub out_features: usize,
    /// Number of diagonal blocks.
    pub block_count: usize,
    /// Size of each square block (`out_features / block_count`).
    pub block_size: usize,
    /// Whether the per-block `Q` norm is clamped to `eps`.
    pub constrained: bool,
    /// Frobenius-norm clamp applied to each block's `Q` when `constrained`.
    pub eps: f64,
    /// Frozen base weight, row-major `[out_features × in_features]`.
    pub w: Vec<f64>,
    /// One trainable `block_size × block_size` generator per block (row-major).
    pub s_blocks: Vec<Vec<f64>>,
}

impl OftLinear {
    /// Construct a new adapter with zero-initialised generators (`R = I`).
    ///
    /// # Errors
    ///
    /// - [`PeftError::EmptyInput`] if any dimension or `block_count` is zero.
    /// - [`PeftError::UnalignedDimension`] if `out_features` is not divisible by
    ///   `block_count`.
    pub fn new(cfg: &OftConfig) -> PeftResult<Self> {
        Self::with_weight(cfg, vec![0.0_f64; cfg.out_features * cfg.in_features])
    }

    /// Construct an adapter around an existing base weight `w`.
    ///
    /// # Errors
    ///
    /// Same validation as [`OftLinear::new`], plus [`PeftError::DimensionMismatch`]
    /// if `w.len() != out_features * in_features`.
    pub fn with_weight(cfg: &OftConfig, w: Vec<f64>) -> PeftResult<Self> {
        if cfg.in_features == 0 || cfg.out_features == 0 || cfg.block_count == 0 {
            return Err(PeftError::EmptyInput);
        }
        if !cfg.out_features.is_multiple_of(cfg.block_count) {
            return Err(PeftError::UnalignedDimension {
                bot: cfg.block_count,
                in_dim: cfg.out_features,
            });
        }
        if w.len() != cfg.out_features * cfg.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: cfg.out_features * cfg.in_features,
                got: w.len(),
            });
        }
        let block_size = cfg.out_features / cfg.block_count;
        let s_blocks = vec![vec![0.0_f64; block_size * block_size]; cfg.block_count];
        Ok(Self {
            in_features: cfg.in_features,
            out_features: cfg.out_features,
            block_count: cfg.block_count,
            block_size,
            constrained: cfg.constrained,
            eps: cfg.eps,
            w,
            s_blocks,
        })
    }

    /// Randomly perturb the generators with small Gaussian noise (scaled by `scale`).
    ///
    /// Useful for tests; production training would update `s_blocks` via gradients.
    pub fn randomise_generators(&mut self, rng: &mut LcgRng, scale: f64) {
        for block in self.s_blocks.iter_mut() {
            for v in block.iter_mut() {
                *v = rng.next_normal() as f64 * scale;
            }
        }
    }

    /// Number of trainable parameters: `block_count · block_size²`.
    #[must_use]
    pub fn n_trainable(&self) -> usize {
        self.block_count * self.block_size * self.block_size
    }

    /// Build the orthogonal block `R_b = (I + Q)⁻¹ (I − Q)` for block index `b`.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::Internal`] if `(I + Q)` is numerically singular.
    pub fn orthogonal_block(&self, b: usize) -> PeftResult<Vec<f64>> {
        if b >= self.block_count {
            return Err(PeftError::LayerOutOfRange {
                idx: b,
                num_layers: self.block_count,
            });
        }
        let n = self.block_size;
        let q = self.skew_block(b);
        // A = I + Q, B = I - Q  (row-major n×n)
        let mut a = vec![0.0_f64; n * n];
        let mut bmat = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let qij = q[i * n + j];
                a[i * n + j] = qij;
                bmat[i * n + j] = -qij;
            }
            a[i * n + i] += 1.0;
            bmat[i * n + i] += 1.0;
        }
        // R = A⁻¹ · B  ⇒ solve A · R = B column-by-column.
        solve_linear_system(&a, &bmat, n)
    }

    /// Skew-symmetrise generator `b` into `Q = (S − Sᵀ)/2`, optionally clamped.
    #[must_use]
    pub fn skew_block(&self, b: usize) -> Vec<f64> {
        let n = self.block_size;
        let s = &self.s_blocks[b];
        let mut q = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                q[i * n + j] = 0.5 * (s[i * n + j] - s[j * n + i]);
            }
        }
        if self.constrained {
            let fro: f64 = q.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if fro > self.eps && fro > 0.0 {
                let factor = self.eps / fro;
                for v in q.iter_mut() {
                    *v *= factor;
                }
            }
        }
        q
    }

    /// Reconstruct the full adapted weight `W' = R · W` (row-major `[out × in]`).
    ///
    /// `R` is block-diagonal, so block `b` only rotates the rows in
    /// `[b·block_size, (b+1)·block_size)`.
    ///
    /// # Errors
    ///
    /// Propagates [`PeftError::Internal`] from [`OftLinear::orthogonal_block`].
    pub fn adapted_weight(&self) -> PeftResult<Vec<f64>> {
        let bs = self.block_size;
        let in_f = self.in_features;
        let mut out = vec![0.0_f64; self.out_features * in_f];
        for b in 0..self.block_count {
            let r = self.orthogonal_block(b)?;
            let row0 = b * bs;
            for i in 0..bs {
                for j in 0..in_f {
                    let mut acc = 0.0_f64;
                    for k in 0..bs {
                        acc += r[i * bs + k] * self.w[(row0 + k) * in_f + j];
                    }
                    out[(row0 + i) * in_f + j] = acc;
                }
            }
        }
        Ok(out)
    }

    /// Forward pass `y = (R · W) · x`.
    ///
    /// `x` must have length `in_features`; the returned vector has length
    /// `out_features`.
    ///
    /// # Errors
    ///
    /// - [`PeftError::DimensionMismatch`] if `x.len() != in_features`.
    /// - Propagates [`PeftError::Internal`] from block inversion.
    pub fn forward(&self, x: &[f64]) -> PeftResult<Vec<f64>> {
        if x.len() != self.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.in_features,
                got: x.len(),
            });
        }
        let bs = self.block_size;
        let in_f = self.in_features;
        // wx[i] = Σ_j W[i,j] · x[j]   (apply the base layer first)
        let mut wx = vec![0.0_f64; self.out_features];
        for (i, wxi) in wx.iter_mut().enumerate() {
            let row_start = i * in_f;
            let mut acc = 0.0_f64;
            for (j, xj) in x.iter().enumerate() {
                acc += self.w[row_start + j] * xj;
            }
            *wxi = acc;
        }
        // y = R · (W x), block by block.
        let mut y = vec![0.0_f64; self.out_features];
        for b in 0..self.block_count {
            let r = self.orthogonal_block(b)?;
            let row0 = b * bs;
            for i in 0..bs {
                let mut acc = 0.0_f64;
                for k in 0..bs {
                    acc += r[i * bs + k] * wx[row0 + k];
                }
                y[row0 + i] = acc;
            }
        }
        Ok(y)
    }
}

/// Solve `A · X = B` for `X` where `A` and `B` are row-major `n×n` matrices, using
/// Gauss-Jordan elimination with partial pivoting on the augmented `[A | B]` system.
///
/// # Errors
///
/// Returns [`PeftError::Internal`] if `A` is numerically singular.
fn solve_linear_system(a: &[f64], b: &[f64], n: usize) -> PeftResult<Vec<f64>> {
    // Augmented matrix M = [A | B] with 2n columns.
    let cols = 2 * n;
    let mut m = vec![0.0_f64; n * cols];
    for i in 0..n {
        for j in 0..n {
            m[i * cols + j] = a[i * n + j];
            m[i * cols + n + j] = b[i * n + j];
        }
    }
    for col in 0..n {
        // Partial pivot: largest |value| in this column at/below the diagonal.
        let mut pivot = col;
        let mut best = m[col * cols + col].abs();
        for r in (col + 1)..n {
            let v = m[r * cols + col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-12 {
            return Err(PeftError::Internal {
                msg: "OFT Cayley transform: (I + Q) is singular".to_string(),
            });
        }
        if pivot != col {
            for c in 0..cols {
                m.swap(col * cols + c, pivot * cols + c);
            }
        }
        // Normalise the pivot row.
        let inv = 1.0 / m[col * cols + col];
        for c in 0..cols {
            m[col * cols + c] *= inv;
        }
        // Eliminate this column from all other rows.
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = m[r * cols + col];
            if factor == 0.0 {
                continue;
            }
            for c in 0..cols {
                m[r * cols + c] -= factor * m[col * cols + c];
            }
        }
    }
    // Right half is A⁻¹ · B.
    let mut x = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            x[i * n + j] = m[i * cols + n + j];
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(in_f: usize, out_f: usize, blocks: usize) -> OftConfig {
        OftConfig {
            in_features: in_f,
            out_features: out_f,
            block_count: blocks,
            constrained: false,
            eps: 1e-3,
        }
    }

    fn orthogonality_error(r: &[f64], n: usize) -> f64 {
        // max |(R Rᵀ)_{ij} − δ_{ij}|
        let mut max_err = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0_f64;
                for k in 0..n {
                    acc += r[i * n + k] * r[j * n + k];
                }
                let target = if i == j { 1.0 } else { 0.0 };
                max_err = max_err.max((acc - target).abs());
            }
        }
        max_err
    }

    #[test]
    fn rejects_zero_dims() {
        assert!(matches!(
            OftLinear::new(&cfg(0, 4, 1)),
            Err(PeftError::EmptyInput)
        ));
        assert!(matches!(
            OftLinear::new(&cfg(4, 0, 1)),
            Err(PeftError::EmptyInput)
        ));
        assert!(matches!(
            OftLinear::new(&cfg(4, 4, 0)),
            Err(PeftError::EmptyInput)
        ));
    }

    #[test]
    fn rejects_unaligned_blocks() {
        // out_features = 6 is not divisible by block_count = 4.
        assert!(matches!(
            OftLinear::new(&cfg(3, 6, 4)),
            Err(PeftError::UnalignedDimension { .. })
        ));
    }

    #[test]
    fn rejects_weight_length_mismatch() {
        let c = cfg(3, 4, 2);
        let bad = vec![0.0_f64; 11]; // should be 12
        assert!(matches!(
            OftLinear::with_weight(&c, bad),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn block_size_and_trainable_count() {
        let oft =
            OftLinear::new(&cfg(5, 8, 4)).expect("OftLinear::new should succeed with valid config");
        assert_eq!(oft.block_size, 2);
        assert_eq!(oft.s_blocks.len(), 4);
        assert_eq!(oft.n_trainable(), 4 * 2 * 2);
    }

    #[test]
    fn identity_rotation_when_zero_init() {
        // S = 0 ⇒ Q = 0 ⇒ R = I, so the adapted weight equals the base weight.
        let c = cfg(4, 4, 1);
        let mut w = vec![0.0_f64; 16];
        for (i, v) in w.iter_mut().enumerate() {
            *v = (i as f64) * 0.1 - 0.7;
        }
        let oft = OftLinear::with_weight(&c, w.clone())
            .expect("OftLinear::with_weight should succeed with valid config and weight");
        let r = oft
            .orthogonal_block(0)
            .expect("orthogonal_block should succeed for valid block index");
        assert!(orthogonality_error(&r, 4) < 1e-12);
        let adapted = oft.adapted_weight().expect("adapted_weight should succeed");
        for (a, b) in adapted.iter().zip(w.iter()) {
            assert!((a - b).abs() < 1e-12, "expected R=I ⇒ W'=W");
        }
    }

    #[test]
    fn cayley_block_is_orthogonal() {
        let c = cfg(4, 4, 1);
        let mut oft = OftLinear::new(&c).expect("OftLinear::new should succeed with valid config");
        let mut rng = LcgRng::new(123);
        oft.randomise_generators(&mut rng, 0.5);
        let r = oft
            .orthogonal_block(0)
            .expect("orthogonal_block should succeed for valid block index");
        let err = orthogonality_error(&r, 4);
        assert!(err < 1e-9, "Cayley output not orthogonal: err={err}");
    }

    #[test]
    fn block_diagonal_blocks_orthogonal() {
        let c = cfg(6, 6, 3); // three 2×2 blocks
        let mut oft = OftLinear::new(&c).expect("OftLinear::new should succeed with valid config");
        let mut rng = LcgRng::new(7);
        oft.randomise_generators(&mut rng, 0.8);
        for b in 0..3 {
            let r = oft
                .orthogonal_block(b)
                .expect("orthogonal_block should succeed for each valid block index");
            assert!(
                orthogonality_error(&r, 2) < 1e-9,
                "block {b} not orthogonal"
            );
        }
    }

    #[test]
    fn forward_preserves_norm_of_wx() {
        // y = R·(Wx); since R is orthogonal, ‖y‖ = ‖Wx‖.
        let c = cfg(4, 4, 1);
        let mut w = vec![0.0_f64; 16];
        let mut rng = LcgRng::new(55);
        for v in w.iter_mut() {
            *v = rng.next_normal() as f64;
        }
        let mut oft = OftLinear::with_weight(&c, w)
            .expect("OftLinear::with_weight should succeed with valid config and weight");
        oft.randomise_generators(&mut rng, 0.6);
        let x: Vec<f64> = (0..4).map(|i| (i as f64) * 0.3 - 0.5).collect();
        // Wx norm
        let mut wx = [0.0_f64; 4];
        for (i, wxi) in wx.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (j, xj) in x.iter().enumerate() {
                acc += oft.w[i * 4 + j] * xj;
            }
            *wxi = acc;
        }
        let wx_norm: f64 = wx.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let y = oft
            .forward(&x)
            .expect("forward pass should succeed with valid input");
        let y_norm: f64 = y.iter().map(|&v| v * v).sum::<f64>().sqrt();
        assert!(
            (wx_norm - y_norm).abs() < 1e-9,
            "orthogonal R must preserve norm: ‖Wx‖={wx_norm}, ‖y‖={y_norm}"
        );
    }

    #[test]
    fn forward_matches_adapted_weight_matvec() {
        let c = cfg(5, 6, 2);
        let mut oft = OftLinear::new(&c).expect("OftLinear::new should succeed with valid config");
        let mut rng = LcgRng::new(99);
        for v in oft.w.iter_mut() {
            *v = rng.next_normal() as f64;
        }
        oft.randomise_generators(&mut rng, 0.4);
        let x: Vec<f64> = (0..5).map(|i| (i as f64) * 0.2 - 0.4).collect();
        let y = oft
            .forward(&x)
            .expect("forward pass should succeed with valid input");
        let wp = oft.adapted_weight().expect("adapted_weight should succeed");
        for i in 0..6 {
            let mut acc = 0.0_f64;
            for j in 0..5 {
                acc += wp[i * 5 + j] * x[j];
            }
            assert!(
                (acc - y[i]).abs() < 1e-9,
                "forward must equal (R·W)·x at row {i}"
            );
        }
    }

    #[test]
    fn forward_dim_mismatch_errors() {
        let oft =
            OftLinear::new(&cfg(4, 4, 1)).expect("OftLinear::new should succeed with valid config");
        let bad = vec![1.0_f64; 3];
        assert!(matches!(
            oft.forward(&bad),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn constrained_clamps_q_norm() {
        let mut c = cfg(4, 4, 1);
        c.constrained = true;
        c.eps = 0.05;
        let mut oft = OftLinear::with_weight(&c, vec![0.0_f64; 16])
            .expect("OftLinear::with_weight should succeed with valid config and weight");
        let mut rng = LcgRng::new(3);
        oft.randomise_generators(&mut rng, 5.0); // huge ⇒ definitely exceeds eps
        let q = oft.skew_block(0);
        let fro: f64 = q.iter().map(|&v| v * v).sum::<f64>().sqrt();
        assert!(
            fro <= 0.05 + 1e-9,
            "constrained Q Frobenius norm must be ≤ eps, got {fro}"
        );
        // Still orthogonal after clamping.
        let r = oft
            .orthogonal_block(0)
            .expect("orthogonal_block should succeed after Q clamping");
        assert!(orthogonality_error(&r, 4) < 1e-9);
    }

    #[test]
    fn block_index_out_of_range_errors() {
        let oft =
            OftLinear::new(&cfg(4, 4, 2)).expect("OftLinear::new should succeed with valid config");
        assert!(matches!(
            oft.orthogonal_block(2),
            Err(PeftError::LayerOutOfRange { .. })
        ));
    }

    #[test]
    fn determinant_one_for_cayley() {
        // The Cayley transform of a skew matrix always has det = +1 (a rotation).
        let c = cfg(3, 3, 1);
        let mut oft = OftLinear::new(&c).expect("OftLinear::new should succeed with valid config");
        let mut rng = LcgRng::new(2024);
        oft.randomise_generators(&mut rng, 0.7);
        let r = oft
            .orthogonal_block(0)
            .expect("orthogonal_block should succeed for valid block index");
        // 3×3 determinant via cofactor expansion.
        let det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6])
            + r[2] * (r[3] * r[7] - r[4] * r[6]);
        assert!((det - 1.0).abs() < 1e-9, "Cayley det must be +1, got {det}");
    }
}
