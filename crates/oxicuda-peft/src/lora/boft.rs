//! BOFT — Butterfly Orthogonal Fine-Tuning (Liu et al. 2024).
//!
//! Reference: Liu, W., Qiu, Z., Feng, Y., Xiu, Y., Xue, Y., Yu, L., Feng, H.,
//! Liu, Z., Heo, J., Peng, S., Wen, Y., Black, M. J., Weller, A., & Schölkopf, B.
//! (2024). *Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization*.
//! ICLR 2024. <https://arxiv.org/abs/2311.06243>
//!
//! BOFT generalises [`crate::lora::oft::OftLinear`] by expressing the dense orthogonal
//! transform `R` as a **product of `m` sparse butterfly-structured orthogonal
//! factors**:
//!
//! ```text
//!   R = B_{m-1} · … · B_1 · B_0 ,        each B_l orthogonal .
//! ```
//!
//! Each factor `B_l` is block-diagonal, but the *block partition shifts* between
//! factors so that, after composing all `m` of them, information mixes across the
//! whole `d × d` space — exactly the connectivity pattern of an FFT butterfly
//! network. This yields an effectively dense orthogonal `R` with only
//! `O(m · d · b)` parameters instead of `O(d²)`.
//!
//! ## Single butterfly factor
//!
//! For a factor at level `l` (`l = 0 … m−1`) we use a block size `b` and a *stride*
//! `s = b · 2ˡ`. Coordinates are grouped so that within one stride-`s` window the
//! rows `{p, p+2ˡ, p+2·2ˡ, …}` form one `b × b` orthogonal block. Each block is
//! built with the same Cayley parametrisation as OFT, `R_blk = (I+Q)⁻¹(I−Q)` with
//! `Q = (S − Sᵀ)/2`, so every factor — and hence their product — is exactly
//! orthogonal for any generator values. With all generators zero, every factor is
//! the identity and `R = I`.
//!
//! Here `d = out_features` must be a power of two and `b = 2`, the canonical
//! radix-2 butterfly used by the reference implementation, so `m = log₂ d` factors
//! reach full connectivity.

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

/// Hyper-parameters for a [`BoftLinear`] adapter.
#[derive(Debug, Clone)]
pub struct BoftConfig {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features. Must be a power of two ≥ 2.
    pub out_features: usize,
    /// Number of butterfly factors `m`. Must satisfy `1 ≤ m ≤ log₂(out_features)`.
    pub n_factors: usize,
}

/// Butterfly-factorised orthogonal fine-tuning of a single linear layer.
///
/// The base weight `w` is frozen; the trainable parameters are the radix-2 block
/// generators of each butterfly factor, stored in [`BoftLinear::factors`].
#[derive(Debug, Clone)]
pub struct BoftLinear {
    /// Input feature count.
    pub in_features: usize,
    /// Output feature count (a power of two).
    pub out_features: usize,
    /// Number of butterfly factors.
    pub n_factors: usize,
    /// Frozen base weight, row-major `[out_features × in_features]`.
    pub w: Vec<f64>,
    /// Per-factor generators. `factors[l]` holds one `2×2` generator (4 values,
    /// row-major) for each of the `out_features / 2` butterfly pairs at level `l`.
    pub factors: Vec<Vec<[f64; 4]>>,
}

impl BoftLinear {
    /// Construct a new adapter with zero-initialised generators (`R = I`).
    ///
    /// # Errors
    ///
    /// - [`PeftError::EmptyInput`] if `in_features == 0` or `out_features == 0`.
    /// - [`PeftError::Internal`] if `out_features` is not a power of two.
    /// - [`PeftError::InvalidTargetRank`] if `n_factors` is `0` or exceeds
    ///   `log₂(out_features)`.
    pub fn new(cfg: &BoftConfig) -> PeftResult<Self> {
        Self::with_weight(cfg, vec![0.0_f64; cfg.out_features * cfg.in_features])
    }

    /// Construct an adapter around an existing base weight `w`.
    ///
    /// # Errors
    ///
    /// Same as [`BoftLinear::new`], plus [`PeftError::DimensionMismatch`] if
    /// `w.len() != out_features * in_features`.
    pub fn with_weight(cfg: &BoftConfig, w: Vec<f64>) -> PeftResult<Self> {
        if cfg.in_features == 0 || cfg.out_features == 0 {
            return Err(PeftError::EmptyInput);
        }
        if !cfg.out_features.is_power_of_two() || cfg.out_features < 2 {
            return Err(PeftError::Internal {
                msg: format!(
                    "BOFT requires out_features to be a power of two ≥ 2, got {}",
                    cfg.out_features
                ),
            });
        }
        let max_factors = cfg.out_features.trailing_zeros() as usize; // log₂(out)
        if cfg.n_factors == 0 || cfg.n_factors > max_factors {
            return Err(PeftError::InvalidTargetRank {
                target_r: cfg.n_factors,
                r: max_factors,
            });
        }
        if w.len() != cfg.out_features * cfg.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: cfg.out_features * cfg.in_features,
                got: w.len(),
            });
        }
        let n_pairs = cfg.out_features / 2;
        let factors = vec![vec![[0.0_f64; 4]; n_pairs]; cfg.n_factors];
        Ok(Self {
            in_features: cfg.in_features,
            out_features: cfg.out_features,
            n_factors: cfg.n_factors,
            w,
            factors,
        })
    }

    /// Randomly perturb every generator with small Gaussian noise.
    pub fn randomise_generators(&mut self, rng: &mut LcgRng, scale: f64) {
        for factor in self.factors.iter_mut() {
            for g in factor.iter_mut() {
                for v in g.iter_mut() {
                    *v = rng.next_normal() as f64 * scale;
                }
            }
        }
    }

    /// Number of trainable parameters: `n_factors · (out_features / 2) · 4`.
    #[must_use]
    pub fn n_trainable(&self) -> usize {
        self.n_factors * (self.out_features / 2) * 4
    }

    /// Index pairs `(p, q)` rotated together at butterfly level `l`.
    ///
    /// For radix-2 the partner of coordinate `p` is `p ^ (1 << l)`; we return each
    /// pair once with `p < q`.
    fn pairs_at_level(&self, level: usize) -> Vec<(usize, usize)> {
        let stride = 1usize << level;
        let mut pairs = Vec::with_capacity(self.out_features / 2);
        for p in 0..self.out_features {
            let q = p ^ stride;
            if p < q {
                pairs.push((p, q));
            }
        }
        pairs
    }

    /// Apply a single butterfly factor `B_l` in place to a length-`out_features`
    /// vector: for each pair `(p, q)` it multiplies `[v_p, v_q]ᵀ` by the orthogonal
    /// `2×2` Cayley rotation of that pair's generator.
    fn apply_factor(&self, level: usize, v: &mut [f64]) {
        let pairs = self.pairs_at_level(level);
        let gens = &self.factors[level];
        for (idx, &(p, q)) in pairs.iter().enumerate() {
            let r = cayley_2x2(&gens[idx]);
            let vp = v[p];
            let vq = v[q];
            v[p] = r[0] * vp + r[1] * vq;
            v[q] = r[2] * vp + r[3] * vq;
        }
    }

    /// Forward pass `y = R · (W · x)` where `R = B_{m-1} · … · B_0`.
    ///
    /// # Errors
    ///
    /// [`PeftError::DimensionMismatch`] if `x.len() != in_features`.
    pub fn forward(&self, x: &[f64]) -> PeftResult<Vec<f64>> {
        if x.len() != self.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.in_features,
                got: x.len(),
            });
        }
        let in_f = self.in_features;
        // y₀ = W · x
        let mut y = vec![0.0_f64; self.out_features];
        for (i, yi) in y.iter_mut().enumerate() {
            let row_start = i * in_f;
            let mut acc = 0.0_f64;
            for (j, xj) in x.iter().enumerate() {
                acc += self.w[row_start + j] * xj;
            }
            *yi = acc;
        }
        // Apply factors left-to-right: B_0 first, then B_1, … (so the matrix product
        // acting on the column vector is B_{m-1} · … · B_0).
        for level in 0..self.n_factors {
            self.apply_factor(level, &mut y);
        }
        Ok(y)
    }

    /// Materialise the full orthogonal transform `R` (row-major `[out × out]`) by
    /// pushing each standard basis vector through the butterfly factors.
    #[must_use]
    pub fn orthogonal_matrix(&self) -> Vec<f64> {
        let d = self.out_features;
        let mut r = vec![0.0_f64; d * d];
        for col in 0..d {
            let mut e = vec![0.0_f64; d];
            e[col] = 1.0;
            for level in 0..self.n_factors {
                self.apply_factor(level, &mut e);
            }
            for (row, &val) in e.iter().enumerate() {
                r[row * d + col] = val;
            }
        }
        r
    }

    /// Reconstruct the adapted weight `W' = R · W` (row-major `[out × in]`).
    #[must_use]
    pub fn adapted_weight(&self) -> Vec<f64> {
        let in_f = self.in_features;
        let d = self.out_features;
        let mut out = vec![0.0_f64; d * in_f];
        for j in 0..in_f {
            // Column j of W ⇒ apply butterfly ⇒ column j of W'.
            let mut col = vec![0.0_f64; d];
            for (i, ci) in col.iter_mut().enumerate() {
                *ci = self.w[i * in_f + j];
            }
            for level in 0..self.n_factors {
                self.apply_factor(level, &mut col);
            }
            for (i, &ci) in col.iter().enumerate() {
                out[i * in_f + j] = ci;
            }
        }
        out
    }
}

/// Cayley transform of a `2×2` generator `S`: `R = (I+Q)⁻¹(I−Q)` with
/// `Q = (S−Sᵀ)/2`. For `2×2` skew matrices `Q = [[0, a], [−a, 0]]` the closed form
/// is a planar rotation `[[c, −s],[s, c]]` with `c = (1−a²)/(1+a²)`,
/// `s = 2a/(1+a²)`; returned row-major as `[r00, r01, r10, r11]`.
#[must_use]
fn cayley_2x2(s: &[f64; 4]) -> [f64; 4] {
    // a is the single off-diagonal of the skew-symmetrised matrix.
    let a = 0.5 * (s[1] - s[2]);
    let denom = 1.0 + a * a;
    let c = (1.0 - a * a) / denom;
    let sn = 2.0 * a / denom;
    [c, -sn, sn, c]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(in_f: usize, out_f: usize, m: usize) -> BoftConfig {
        BoftConfig {
            in_features: in_f,
            out_features: out_f,
            n_factors: m,
        }
    }

    fn orthogonality_error(r: &[f64], n: usize) -> f64 {
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
            BoftLinear::new(&cfg(0, 4, 1)),
            Err(PeftError::EmptyInput)
        ));
        assert!(matches!(
            BoftLinear::new(&cfg(4, 0, 1)),
            Err(PeftError::EmptyInput)
        ));
    }

    #[test]
    fn rejects_non_power_of_two_out() {
        assert!(matches!(
            BoftLinear::new(&cfg(3, 6, 1)),
            Err(PeftError::Internal { .. })
        ));
    }

    #[test]
    fn rejects_too_many_factors() {
        // log₂(8) = 3, so n_factors = 4 must fail.
        assert!(matches!(
            BoftLinear::new(&cfg(3, 8, 4)),
            Err(PeftError::InvalidTargetRank { .. })
        ));
        assert!(matches!(
            BoftLinear::new(&cfg(3, 8, 0)),
            Err(PeftError::InvalidTargetRank { .. })
        ));
    }

    #[test]
    fn rejects_weight_length_mismatch() {
        let c = cfg(3, 4, 2);
        let bad = vec![0.0_f64; 10]; // should be 12
        assert!(matches!(
            BoftLinear::with_weight(&c, bad),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn trainable_count_correct() {
        let boft = BoftLinear::new(&cfg(5, 8, 3))
            .expect("BoftLinear::new should succeed with valid BOFT config");
        // 3 factors × (8/2 = 4 pairs) × 4 generator values.
        assert_eq!(boft.n_trainable(), 3 * 4 * 4);
        assert_eq!(boft.factors.len(), 3);
        assert_eq!(boft.factors[0].len(), 4);
    }

    #[test]
    fn identity_when_zero_init() {
        let c = cfg(4, 4, 2);
        let mut w = vec![0.0_f64; 16];
        for (i, v) in w.iter_mut().enumerate() {
            *v = (i as f64) * 0.05 - 0.3;
        }
        let boft = BoftLinear::with_weight(&c, w.clone())
            .expect("BoftLinear::with_weight should succeed with valid config and weight");
        let r = boft.orthogonal_matrix();
        assert!(orthogonality_error(&r, 4) < 1e-12);
        // R = I ⇒ W' = W.
        let adapted = boft.adapted_weight();
        for (a, b) in adapted.iter().zip(w.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn r_is_orthogonal_after_random() {
        let c = cfg(8, 8, 3);
        let mut boft =
            BoftLinear::new(&c).expect("BoftLinear::new should succeed with valid BOFT config");
        let mut rng = LcgRng::new(17);
        boft.randomise_generators(&mut rng, 0.7);
        let r = boft.orthogonal_matrix();
        let err = orthogonality_error(&r, 8);
        assert!(
            err < 1e-9,
            "product of butterfly factors must be orthogonal: {err}"
        );
    }

    #[test]
    fn forward_preserves_norm() {
        let c = cfg(8, 8, 3);
        let mut boft =
            BoftLinear::new(&c).expect("BoftLinear::new should succeed with valid BOFT config");
        let mut rng = LcgRng::new(91);
        for v in boft.w.iter_mut() {
            *v = rng.next_normal() as f64;
        }
        boft.randomise_generators(&mut rng, 0.5);
        let x: Vec<f64> = (0..8).map(|i| (i as f64) * 0.13 - 0.4).collect();
        // ‖Wx‖
        let mut wx = [0.0_f64; 8];
        for (i, wxi) in wx.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (j, xj) in x.iter().enumerate() {
                acc += boft.w[i * 8 + j] * xj;
            }
            *wxi = acc;
        }
        let wx_norm: f64 = wx.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let y = boft.forward(&x).expect("BOFT forward pass should succeed");
        let y_norm: f64 = y.iter().map(|&v| v * v).sum::<f64>().sqrt();
        assert!(
            (wx_norm - y_norm).abs() < 1e-9,
            "‖Wx‖={wx_norm} vs ‖y‖={y_norm}"
        );
    }

    #[test]
    fn forward_matches_adapted_weight() {
        let c = cfg(5, 8, 3);
        let mut boft =
            BoftLinear::new(&c).expect("BoftLinear::new should succeed with valid BOFT config");
        let mut rng = LcgRng::new(64);
        for v in boft.w.iter_mut() {
            *v = rng.next_normal() as f64;
        }
        boft.randomise_generators(&mut rng, 0.45);
        let x: Vec<f64> = (0..5).map(|i| (i as f64) * 0.2 - 0.5).collect();
        let y = boft
            .forward(&x)
            .expect("BOFT forward pass should succeed with valid input");
        let wp = boft.adapted_weight();
        for i in 0..8 {
            let mut acc = 0.0_f64;
            for j in 0..5 {
                acc += wp[i * 5 + j] * x[j];
            }
            assert!((acc - y[i]).abs() < 1e-9, "row {i} mismatch");
        }
    }

    #[test]
    fn forward_dim_mismatch_errors() {
        let boft = BoftLinear::new(&cfg(4, 4, 1))
            .expect("BoftLinear::new should succeed with valid BOFT config");
        let bad = vec![1.0_f64; 5];
        assert!(matches!(
            boft.forward(&bad),
            Err(PeftError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn full_factors_mix_all_coordinates() {
        // With m = log₂(d) factors and non-zero generators, the orthogonal matrix R
        // should be effectively dense: no row should be a single standard basis
        // vector (which would mean that coordinate never mixed).
        let c = cfg(4, 8, 3); // m = 3 = log₂ 8
        let mut boft =
            BoftLinear::new(&c).expect("BoftLinear::new should succeed with full-factor config");
        let mut rng = LcgRng::new(2024);
        boft.randomise_generators(&mut rng, 0.9);
        let r = boft.orthogonal_matrix();
        for i in 0..8 {
            let nnz = (0..8).filter(|&j| r[i * 8 + j].abs() > 1e-6).count();
            assert!(nnz_ok(nnz), "row {i} only mixes {nnz} coords; expected > 1");
        }
    }

    fn nnz_ok(n: usize) -> bool {
        n > 1
    }

    #[test]
    fn cayley_2x2_is_planar_rotation() {
        // For S = [[0, a],[0, 0]], a = 0.5·(s01 - s10) = 0.5·a_raw; verify c²+s²=1.
        let g = [0.0_f64, 1.3, -0.4, 0.0];
        let r = cayley_2x2(&g);
        let det = r[0] * r[3] - r[1] * r[2];
        assert!(
            (det - 1.0).abs() < 1e-12,
            "rotation det must be 1, got {det}"
        );
        let c2s2 = r[0] * r[0] + r[2] * r[2];
        assert!((c2s2 - 1.0).abs() < 1e-12, "c²+s² must be 1, got {c2s2}");
    }

    #[test]
    fn single_factor_is_block_diagonal_pairs() {
        // With one factor at level 0, coordinate p only mixes with p^1.
        let c = cfg(4, 4, 1);
        let mut boft =
            BoftLinear::new(&c).expect("BoftLinear::new should succeed with single-factor config");
        let mut rng = LcgRng::new(5);
        boft.randomise_generators(&mut rng, 0.6);
        let r = boft.orthogonal_matrix();
        // Row 0 should only have nonzeros in columns {0,1}; row 2 in {2,3}.
        // Indices: row 0 cols 2,3 -> r[2], r[3]; row 2 cols 0,1 -> r[8], r[9].
        assert!(r[2].abs() < 1e-12 && r[3].abs() < 1e-12);
        assert!(r[8].abs() < 1e-12 && r[9].abs() < 1e-12);
    }
}
