//! DoRA (Weight-Decomposed Low-Rank Adaptation) adapter for linear layers.
//!
//! Implements the DoRA decomposition of Liu et al. 2024 ICML, "DoRA:
//! Weight-Decomposed Low-Rank Adaptation". The base weight `W₀ ∈ ℝ^{d_out×d_in}`
//! is conceptually decomposed into a per-output-row magnitude vector
//! `m₀ ∈ ℝ^{d_out}` and a direction matrix `D₀ = W₀ / ‖W₀‖_row` (each row
//! normalised to unit 2-norm). DoRA then adapts the weight as
//!
//! ```text
//! W'_o = m_o · ( W₀_o + (α/r) · (B · A)_o ) / ‖ W₀_o + (α/r) · (B · A)_o ‖_2
//! ```
//!
//! for each output row `o`, where `m ∈ ℝ^{d_out}` is a trainable magnitude
//! vector, `A ∈ ℝ^{r×d_in}` is Gaussian-initialised, and `B ∈ ℝ^{d_out×r}` is
//! zero-initialised (so the initial low-rank direction update is zero and the
//! initial direction equals the base direction `D₀`). The product `B · A` is
//! the LoRA-style low-rank direction update, and `(α/r)` is the LoRA scaling.
//!
//! The defining DoRA invariant is that magnitude and direction are decoupled:
//! every row of the effective weight has 2-norm exactly `m[o]`, irrespective
//! of the low-rank update `B · A`. Changing `A` or `B` rotates the row
//! direction without changing its magnitude; changing `m[o]` rescales the
//! row linearly without changing its direction.
//!
//! # Reference
//! Liu, Wang, Yin, Molchanov, Wang, Cheng & Chen, "DoRA: Weight-Decomposed
//! Low-Rank Adaptation", ICML 2024.

use crate::error::{GenError, GenResult};
use crate::handle::LcgRng;

// ─── DoraConfig ──────────────────────────────────────────────────────────────

/// Configuration for the [`DoraAdapter`].
///
/// Holds the linear-layer dimensions, the LoRA intrinsic rank, and the
/// LoRA scaling factor `α`. The effective scaling applied to `B · A` is
/// `α / r`, matching the standard LoRA convention.
#[derive(Debug, Clone, PartialEq)]
pub struct DoraConfig {
    /// Input feature dimension `d_in`. Must be ≥ 1.
    pub d_in: usize,
    /// Output feature dimension `d_out`. Must be ≥ 1.
    pub d_out: usize,
    /// Intrinsic rank `r` of the low-rank direction update. Must be ≥ 1.
    pub rank: usize,
    /// LoRA scaling parameter `α`. Must be > 0.
    pub alpha: f32,
}

// ─── DoraAdapter ─────────────────────────────────────────────────────────────

/// DoRA adapter for a single linear layer.
///
/// Stores the (conceptually frozen) base weight `W₀ ∈ ℝ^{d_out×d_in}` in
/// row-major order, the LoRA matrices `A ∈ ℝ^{r×d_in}` (Gaussian-initialised)
/// and `B ∈ ℝ^{d_out×r}` (zero-initialised), and the trainable per-row
/// magnitude vector `m ∈ ℝ^{d_out}` (initialised to `‖W₀_row‖_2`).
///
/// # Reference
/// Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", ICML 2024.
#[derive(Debug, Clone)]
pub struct DoraAdapter {
    /// Configuration controlling dimensions, rank, and `α`.
    cfg: DoraConfig,
    /// Base weight `W₀`, shape `d_out × d_in` row-major. Conceptually frozen
    /// (the trainable parameters are `A`, `B`, and `m`); kept inside the
    /// adapter so that [`Self::effective_weight`] is self-contained.
    base_weight: Vec<f32>,
    /// LoRA `A` matrix, shape `rank × d_in` row-major. Gaussian-initialised
    /// with standard deviation `1/√r`.
    matrix_a: Vec<f32>,
    /// LoRA `B` matrix, shape `d_out × rank` row-major. Zero-initialised so
    /// that the initial direction update is zero.
    matrix_b: Vec<f32>,
    /// Per-output-row magnitude vector, length `d_out`. Initialised to the
    /// per-row 2-norm of `W₀` so that the initial effective weight equals
    /// `W₀`.
    magnitude: Vec<f32>,
}

/// Numerical floor used when normalising rows of `W₀ + (α/r)·B·A`.
///
/// If a row has 2-norm below this threshold the normalisation falls back to
/// dividing by the threshold (which yields a near-zero direction vector)
/// instead of producing a NaN/Inf. The threshold is small enough that it does
/// not perturb realistic rows but large enough to keep `f32` arithmetic
/// finite for genuinely zero rows.
const DORA_NORM_EPS: f32 = 1.0e-12;

impl DoraAdapter {
    /// Build a new [`DoraAdapter`] from the given configuration and base
    /// weight.
    ///
    /// `base_weight` is the row-major flattening of `W₀ ∈ ℝ^{d_out×d_in}`.
    /// `A` is filled with `N(0, 1/r)` samples (using `next_normal_pair`), `B`
    /// is zero-initialised, and the per-row magnitude `m[o]` is initialised
    /// to the 2-norm of row `o` of `W₀`, so that the initial effective weight
    /// satisfies `W' ≡ W₀` (up to `f32` rounding).
    ///
    /// # Errors
    /// - [`GenError::EmptyInput`] if any of `d_in`, `d_out`, `rank` is `0`.
    /// - [`GenError::InvalidLoraAlpha`] if `alpha <= 0`.
    /// - [`GenError::DimensionMismatch`] if
    ///   `base_weight.len() != d_out * d_in`.
    pub fn new(cfg: DoraConfig, base_weight: Vec<f32>, rng: &mut LcgRng) -> GenResult<Self> {
        if cfg.d_in == 0 {
            return Err(GenError::EmptyInput("d_in must be >= 1"));
        }
        if cfg.d_out == 0 {
            return Err(GenError::EmptyInput("d_out must be >= 1"));
        }
        if cfg.rank == 0 {
            return Err(GenError::InvalidLoraRank(cfg.rank));
        }
        if cfg.alpha <= 0.0 {
            return Err(GenError::InvalidLoraAlpha(cfg.alpha));
        }
        let expected = cfg.d_out * cfg.d_in;
        if base_weight.len() != expected {
            return Err(GenError::DimensionMismatch {
                expected,
                got: base_weight.len(),
            });
        }

        // A ~ N(0, 1/r): Gaussian init with stddev 1/√r via Box-Muller pairs.
        let a_size = cfg.rank * cfg.d_in;
        let mut matrix_a = vec![0.0_f32; a_size];
        let std = 1.0_f32 / (cfg.rank as f32).sqrt();
        let mut idx = 0;
        while idx + 1 < a_size {
            let (z0, z1) = rng.next_normal_pair();
            matrix_a[idx] = z0 * std;
            matrix_a[idx + 1] = z1 * std;
            idx += 2;
        }
        if idx < a_size {
            let (z0, _) = rng.next_normal_pair();
            matrix_a[idx] = z0 * std;
        }

        // B zero-initialised.
        let matrix_b = vec![0.0_f32; cfg.d_out * cfg.rank];

        // m_o ← ‖W₀_o‖_2.
        let mut magnitude = vec![0.0_f32; cfg.d_out];
        for o in 0..cfg.d_out {
            let row = &base_weight[o * cfg.d_in..(o + 1) * cfg.d_in];
            let mut acc = 0.0_f32;
            for &w in row {
                acc += w * w;
            }
            magnitude[o] = acc.sqrt();
        }

        Ok(Self {
            cfg,
            base_weight,
            matrix_a,
            matrix_b,
            magnitude,
        })
    }

    /// Compute row `o` of `W₀ + (α/r) · B · A`.
    ///
    /// Internal helper used by [`Self::effective_weight`] and
    /// [`Self::forward`] so we avoid materialising the full `d_out × d_in`
    /// direction-update matrix when only a single row is needed.
    fn delta_row(&self, o: usize) -> Vec<f32> {
        let r = self.cfg.rank;
        let d_in = self.cfg.d_in;
        let scale = self.cfg.alpha / r as f32;
        let mut row = self.base_weight[o * d_in..(o + 1) * d_in].to_vec();
        for (j, dst) in row.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for k in 0..r {
                acc += self.matrix_b[o * r + k] * self.matrix_a[k * d_in + j];
            }
            *dst += scale * acc;
        }
        row
    }

    /// Compute the full effective weight `W' ∈ ℝ^{d_out×d_in}` (row-major).
    ///
    /// For each output row `o`, the returned row equals
    /// `m[o] · (W₀_o + (α/r)·(B·A)_o) / ‖W₀_o + (α/r)·(B·A)_o‖_2`,
    /// with an `f32`-safe epsilon floor in the denominator. The resulting
    /// row has 2-norm exactly `m[o]` (up to rounding), which is the defining
    /// DoRA decoupling property.
    ///
    /// # Errors
    /// This method does not currently fail; the return type is `GenResult`
    /// to preserve API parity with other DoRA methods that may grow further
    /// validation.
    pub fn effective_weight(&self) -> GenResult<Vec<f32>> {
        let d_in = self.cfg.d_in;
        let d_out = self.cfg.d_out;
        let mut w = vec![0.0_f32; d_out * d_in];
        for o in 0..d_out {
            let row = self.delta_row(o);
            let mut norm_sq = 0.0_f32;
            for &v in &row {
                norm_sq += v * v;
            }
            let norm = norm_sq.sqrt().max(DORA_NORM_EPS);
            let scale = self.magnitude[o] / norm;
            let dst = &mut w[o * d_in..(o + 1) * d_in];
            for (d, &v) in dst.iter_mut().zip(&row) {
                *d = scale * v;
            }
        }
        Ok(w)
    }

    /// Apply the effective DoRA weight to a single input vector.
    ///
    /// Returns `y = W' · x` where `W'` is the effective weight from
    /// [`Self::effective_weight`] and `x ∈ ℝ^{d_in}`. The output has length
    /// `d_out`.
    ///
    /// To avoid forming the full `d_out × d_in` effective weight, each output
    /// row is normalised in place and applied row-by-row.
    ///
    /// # Errors
    /// - [`GenError::DimensionMismatch`] if `x.len() != d_in`.
    pub fn forward(&self, x: &[f32]) -> GenResult<Vec<f32>> {
        if x.len() != self.cfg.d_in {
            return Err(GenError::DimensionMismatch {
                expected: self.cfg.d_in,
                got: x.len(),
            });
        }
        let d_out = self.cfg.d_out;
        let mut y = vec![0.0_f32; d_out];
        for (o, y_o) in y.iter_mut().enumerate() {
            let row = self.delta_row(o);
            let mut norm_sq = 0.0_f32;
            for &v in &row {
                norm_sq += v * v;
            }
            let norm = norm_sq.sqrt().max(DORA_NORM_EPS);
            let scale = self.magnitude[o] / norm;
            let mut dot = 0.0_f32;
            for (&w, &xi) in row.iter().zip(x) {
                dot += w * xi;
            }
            *y_o = scale * dot;
        }
        Ok(y)
    }

    /// Number of trainable parameters in the adapter.
    ///
    /// DoRA's trainable parameters are `A` (`r·d_in`), `B` (`d_out·r`), and
    /// the magnitude vector `m` (`d_out`). The base weight `W₀` is stored
    /// for convenience but is conceptually frozen, so it is *not* included
    /// in this count.
    pub fn n_params(&self) -> usize {
        self.cfg.rank * self.cfg.d_in + self.cfg.d_out * self.cfg.rank + self.cfg.d_out
    }

    /// Overwrite the magnitude vector.
    ///
    /// # Errors
    /// - [`GenError::DimensionMismatch`] if `m.len() != d_out`.
    pub fn set_magnitude(&mut self, m: &[f32]) -> GenResult<()> {
        if m.len() != self.cfg.d_out {
            return Err(GenError::DimensionMismatch {
                expected: self.cfg.d_out,
                got: m.len(),
            });
        }
        self.magnitude.copy_from_slice(m);
        Ok(())
    }

    /// Return the configuration.
    pub fn config(&self) -> &DoraConfig {
        &self.cfg
    }

    /// Read-only access to the base weight (row-major, `d_out × d_in`).
    pub fn base_weight(&self) -> &[f32] {
        &self.base_weight
    }

    /// Read-only access to the magnitude vector (length `d_out`).
    pub fn magnitude(&self) -> &[f32] {
        &self.magnitude
    }

    /// Read-only access to `A` (shape `rank × d_in`, row-major).
    pub fn matrix_a(&self) -> &[f32] {
        &self.matrix_a
    }

    /// Read-only access to `B` (shape `d_out × rank`, row-major).
    pub fn matrix_b(&self) -> &[f32] {
        &self.matrix_b
    }

    /// Mutable access to `A` (shape `rank × d_in`, row-major).
    pub fn matrix_a_mut(&mut self) -> &mut Vec<f32> {
        &mut self.matrix_a
    }

    /// Mutable access to `B` (shape `d_out × rank`, row-major).
    pub fn matrix_b_mut(&mut self) -> &mut Vec<f32> {
        &mut self.matrix_b
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn make_rng(seed: u64) -> LcgRng {
        LcgRng::new(seed)
    }

    fn make_cfg(d_in: usize, d_out: usize, rank: usize, alpha: f32) -> DoraConfig {
        DoraConfig {
            d_in,
            d_out,
            rank,
            alpha,
        }
    }

    fn nontrivial_base(d_out: usize, d_in: usize) -> Vec<f32> {
        // Deterministic nonzero, non-row-uniform base weight so that the
        // initial per-row norms are all distinct and positive.
        let mut w = vec![0.0_f32; d_out * d_in];
        for o in 0..d_out {
            for j in 0..d_in {
                let s = (o as f32 + 1.0) * 0.1 + (j as f32 + 1.0) * 0.07;
                w[o * d_in + j] = s;
            }
        }
        w
    }

    fn row_norm(slice: &[f32], _d_out: usize, d_in: usize, o: usize) -> f32 {
        let row = &slice[o * d_in..(o + 1) * d_in];
        row.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    #[test]
    fn dora_n_params_trainable_count() {
        // n_params counts only the trainable parameters: A + B + m.
        let cfg = make_cfg(8, 16, 4, 8.0);
        let mut rng = make_rng(42);
        let base = nontrivial_base(cfg.d_out, cfg.d_in);
        let dora = DoraAdapter::new(cfg.clone(), base, &mut rng).unwrap();
        let expected = cfg.rank * cfg.d_in + cfg.d_out * cfg.rank + cfg.d_out;
        assert_eq!(dora.n_params(), expected);
    }

    #[test]
    fn dora_forward_output_length() {
        let cfg = make_cfg(8, 16, 4, 8.0);
        let mut rng = make_rng(42);
        let base = nontrivial_base(cfg.d_out, cfg.d_in);
        let dora = DoraAdapter::new(cfg, base, &mut rng).unwrap();
        let x = vec![0.25_f32; 8];
        let y = dora.forward(&x).unwrap();
        assert_eq!(y.len(), 16);
    }

    #[test]
    fn dora_initial_effective_weight_equals_base() {
        // After construction: B=0 (so ΔW=0) and m_o = ‖W₀_o‖_2, hence
        // W' = m_o · W₀_o / ‖W₀_o‖_2 = W₀_o. The effective weight equals W₀.
        let cfg = make_cfg(6, 5, 3, 4.0);
        let mut rng = make_rng(123);
        let base = nontrivial_base(cfg.d_out, cfg.d_in);
        let dora = DoraAdapter::new(cfg.clone(), base.clone(), &mut rng).unwrap();
        let w = dora.effective_weight().unwrap();
        for (i, (&we, &wb)) in w.iter().zip(&base).enumerate() {
            assert!(
                (we - wb).abs() < EPS,
                "effective != base at idx {i}: {we} vs {wb}"
            );
        }
    }

    #[test]
    fn dora_set_magnitude_updates_state() {
        let cfg = make_cfg(4, 3, 2, 2.0);
        let mut rng = make_rng(7);
        let base = nontrivial_base(cfg.d_out, cfg.d_in);
        let mut dora = DoraAdapter::new(cfg, base, &mut rng).unwrap();
        let new_m = vec![0.5_f32, 1.0, 2.0];
        dora.set_magnitude(&new_m).unwrap();
        for (&a, &b) in dora.magnitude().iter().zip(&new_m) {
            assert!((a - b).abs() < EPS, "{a} != {b}");
        }
    }

    #[test]
    fn dora_effective_row_norm_equals_magnitude_initial() {
        // For the initial adapter (B=0, m=‖W₀_row‖), each effective row
        // has 2-norm equal to m[o] (which equals ‖W₀_row‖_2).
        let cfg = make_cfg(5, 4, 2, 4.0);
        let mut rng = make_rng(11);
        let base = nontrivial_base(cfg.d_out, cfg.d_in);
        let dora = DoraAdapter::new(cfg.clone(), base, &mut rng).unwrap();
        let w = dora.effective_weight().unwrap();
        for o in 0..cfg.d_out {
            let n = row_norm(&w, cfg.d_out, cfg.d_in, o);
            assert!(
                (n - dora.magnitude()[o]).abs() < EPS,
                "row {o}: ‖.‖={n} != m={}",
                dora.magnitude()[o]
            );
        }
    }

    #[test]
    fn dora_effective_row_norm_equals_magnitude_with_nonzero_b() {
        // Defining DoRA invariant: even when B is nonzero, every effective
        // row has 2-norm equal to m[o]. Magnitude and direction are
        // decoupled.
        let cfg = make_cfg(7, 5, 3, 6.0);
        let mut rng = make_rng(31);
        let base = nontrivial_base(cfg.d_out, cfg.d_in);
        let mut dora = DoraAdapter::new(cfg.clone(), base, &mut rng).unwrap();
        for v in dora.matrix_b_mut() {
            *v = 0.03;
        }
        let w = dora.effective_weight().unwrap();
        for o in 0..cfg.d_out {
            let n = row_norm(&w, cfg.d_out, cfg.d_in, o);
            assert!(
                (n - dora.magnitude()[o]).abs() < 1e-4,
                "row {o}: ‖.‖={n} != m={}",
                dora.magnitude()[o]
            );
        }
    }

    #[test]
    fn dora_changing_a_preserves_row_norm() {
        // Direction-only update (perturb A): row norm should stay = m[o].
        let cfg = make_cfg(6, 4, 2, 4.0);
        let mut rng = make_rng(53);
        let base = nontrivial_base(cfg.d_out, cfg.d_in);
        let mut dora = DoraAdapter::new(cfg.clone(), base, &mut rng).unwrap();
        // Force B≠0 first so that A actually contributes to the row.
        for v in dora.matrix_b_mut() {
            *v = 0.05;
        }
        // Now perturb A.
        for (k, v) in dora.matrix_a_mut().iter_mut().enumerate() {
            *v += 0.01 * (k as f32 + 1.0);
        }
        let w = dora.effective_weight().unwrap();
        for o in 0..cfg.d_out {
            let n = row_norm(&w, cfg.d_out, cfg.d_in, o);
            assert!(
                (n - dora.magnitude()[o]).abs() < 1e-4,
                "row {o}: ‖.‖={n} after A perturbation, expected m={}",
                dora.magnitude()[o]
            );
        }
    }

    #[test]
    fn dora_changing_b_preserves_row_norm() {
        // Direction-only update (perturb B): row norm should stay = m[o].
        let cfg = make_cfg(6, 4, 2, 4.0);
        let mut rng = make_rng(91);
        let base = nontrivial_base(cfg.d_out, cfg.d_in);
        let mut dora = DoraAdapter::new(cfg.clone(), base, &mut rng).unwrap();
        for (k, v) in dora.matrix_b_mut().iter_mut().enumerate() {
            *v = 0.02 * ((k % 5) as f32 + 1.0);
        }
        let w = dora.effective_weight().unwrap();
        for o in 0..cfg.d_out {
            let n = row_norm(&w, cfg.d_out, cfg.d_in, o);
            assert!(
                (n - dora.magnitude()[o]).abs() < 1e-4,
                "row {o}: ‖.‖={n} after B perturbation, expected m={}",
                dora.magnitude()[o]
            );
        }
    }

    #[test]
    fn dora_scaling_m_scales_row_linearly() {
        // Doubling m[o] doubles every entry of the effective row (the
        // direction is unchanged).
        let cfg = make_cfg(5, 4, 2, 4.0);
        let mut rng = make_rng(202);
        let base = nontrivial_base(cfg.d_out, cfg.d_in);
        let mut dora = DoraAdapter::new(cfg.clone(), base, &mut rng).unwrap();
        for v in dora.matrix_b_mut() {
            *v = 0.04;
        }
        let w_before = dora.effective_weight().unwrap();
        let original_m = dora.magnitude().to_vec();
        let scaled_m: Vec<f32> = original_m.iter().map(|&v| 2.0 * v).collect();
        dora.set_magnitude(&scaled_m).unwrap();
        let w_after = dora.effective_weight().unwrap();
        for o in 0..cfg.d_out {
            for j in 0..cfg.d_in {
                let idx = o * cfg.d_in + j;
                let a = w_before[idx];
                let b = w_after[idx];
                assert!(
                    (b - 2.0 * a).abs() < 1e-4,
                    "row {o} col {j}: w_after={b} != 2*w_before={}",
                    2.0 * a
                );
            }
        }
    }

    #[test]
    fn dora_deterministic_given_seed() {
        let cfg = make_cfg(5, 4, 2, 4.0);
        let base = nontrivial_base(cfg.d_out, cfg.d_in);
        let mut rng_a = make_rng(777);
        let mut rng_b = make_rng(777);
        let a = DoraAdapter::new(cfg.clone(), base.clone(), &mut rng_a).unwrap();
        let b = DoraAdapter::new(cfg, base, &mut rng_b).unwrap();
        for (x, y) in a.matrix_a().iter().zip(b.matrix_a()) {
            assert!((x - y).abs() < EPS, "A non-deterministic: {x} vs {y}");
        }
        for (x, y) in a.magnitude().iter().zip(b.magnitude()) {
            assert!(
                (x - y).abs() < EPS,
                "magnitude non-deterministic: {x} vs {y}"
            );
        }
    }

    #[test]
    fn dora_err_d_in_zero() {
        let cfg = make_cfg(0, 4, 2, 4.0);
        let mut rng = make_rng(1);
        assert!(matches!(
            DoraAdapter::new(cfg, vec![], &mut rng),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn dora_err_d_out_zero() {
        let cfg = make_cfg(4, 0, 2, 4.0);
        let mut rng = make_rng(1);
        assert!(matches!(
            DoraAdapter::new(cfg, vec![], &mut rng),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn dora_err_rank_zero() {
        let cfg = make_cfg(4, 4, 0, 4.0);
        let mut rng = make_rng(1);
        let base = vec![0.1_f32; 16];
        assert!(matches!(
            DoraAdapter::new(cfg, base, &mut rng),
            Err(GenError::InvalidLoraRank(0))
        ));
    }

    #[test]
    fn dora_err_alpha_non_positive() {
        let cfg = make_cfg(4, 4, 2, 0.0);
        let mut rng = make_rng(1);
        let base = vec![0.1_f32; 16];
        assert!(matches!(
            DoraAdapter::new(cfg, base, &mut rng),
            Err(GenError::InvalidLoraAlpha(_))
        ));
        let cfg = make_cfg(4, 4, 2, -1.0);
        let base = vec![0.1_f32; 16];
        assert!(matches!(
            DoraAdapter::new(cfg, base, &mut rng),
            Err(GenError::InvalidLoraAlpha(_))
        ));
    }

    #[test]
    fn dora_err_base_weight_wrong_length() {
        let cfg = make_cfg(4, 4, 2, 4.0);
        let mut rng = make_rng(1);
        let bad_base = vec![0.0_f32; 5];
        assert!(matches!(
            DoraAdapter::new(cfg, bad_base, &mut rng),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn dora_err_forward_x_wrong_length() {
        let cfg = make_cfg(4, 4, 2, 4.0);
        let mut rng = make_rng(1);
        let base = nontrivial_base(cfg.d_out, cfg.d_in);
        let dora = DoraAdapter::new(cfg, base, &mut rng).unwrap();
        let x = vec![0.0_f32; 3];
        assert!(matches!(
            dora.forward(&x),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn dora_err_set_magnitude_wrong_length() {
        let cfg = make_cfg(4, 4, 2, 4.0);
        let mut rng = make_rng(1);
        let base = nontrivial_base(cfg.d_out, cfg.d_in);
        let mut dora = DoraAdapter::new(cfg, base, &mut rng).unwrap();
        assert!(matches!(
            dora.set_magnitude(&[1.0_f32, 2.0]),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn dora_rank_one_works() {
        let cfg = make_cfg(6, 4, 1, 2.0);
        let mut rng = make_rng(303);
        let base = nontrivial_base(cfg.d_out, cfg.d_in);
        let dora = DoraAdapter::new(cfg, base.clone(), &mut rng).unwrap();
        assert_eq!(dora.matrix_a().len(), 6);
        assert_eq!(dora.matrix_b().len(), 4);
        let w = dora.effective_weight().unwrap();
        // Initial: W' ≡ W₀.
        for (i, (&a, &b)) in w.iter().zip(&base).enumerate() {
            assert!(
                (a - b).abs() < EPS,
                "rank=1 init differs at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn dora_rank_at_least_d_in_works() {
        // rank == d_in is a valid degenerate case (full-rank direction
        // update); rank > d_in is also allowed by the spec.
        let cfg = make_cfg(4, 5, 4, 4.0);
        let mut rng = make_rng(404);
        let base = nontrivial_base(cfg.d_out, cfg.d_in);
        let dora = DoraAdapter::new(cfg.clone(), base.clone(), &mut rng).unwrap();
        assert_eq!(dora.matrix_a().len(), 4 * 4);
        let w = dora.effective_weight().unwrap();
        for (i, (&a, &b)) in w.iter().zip(&base).enumerate() {
            assert!(
                (a - b).abs() < EPS,
                "rank>=d_in init differs at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn dora_zero_row_handled_by_epsilon() {
        // A row of W₀ that is all zero produces a delta_row of all zero
        // (since B=0 initially), so the row norm is 0 and the epsilon guard
        // takes over. The effective row must be finite (in fact, since
        // m_o = 0 too, it should be exactly zero) — never NaN/Inf.
        let cfg = make_cfg(4, 3, 2, 2.0);
        let mut base = vec![0.0_f32; cfg.d_out * cfg.d_in];
        // Row 0 = nonzero, row 1 = all zeros, row 2 = nonzero.
        for j in 0..cfg.d_in {
            base[j] = 0.5;
            base[2 * cfg.d_in + j] = 0.7;
        }
        let mut rng = make_rng(909);
        let dora = DoraAdapter::new(cfg.clone(), base, &mut rng).unwrap();
        let w = dora.effective_weight().unwrap();
        for &v in &w {
            assert!(v.is_finite(), "non-finite entry: {v}");
        }
        // Row 1's effective entries are 0 (because m[1] = 0).
        for j in 0..cfg.d_in {
            assert!(
                w[cfg.d_in + j].abs() < EPS,
                "zero-base row should yield zero effective row, got {}",
                w[cfg.d_in + j]
            );
        }
    }

    #[test]
    fn dora_forward_matches_effective_weight() {
        // Sanity check that `forward` agrees with explicit W' · x.
        let cfg = make_cfg(5, 4, 2, 4.0);
        let mut rng = make_rng(606);
        let base = nontrivial_base(cfg.d_out, cfg.d_in);
        let mut dora = DoraAdapter::new(cfg.clone(), base, &mut rng).unwrap();
        for v in dora.matrix_b_mut() {
            *v = 0.05;
        }
        let x: Vec<f32> = (0..cfg.d_in).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let w = dora.effective_weight().unwrap();
        let mut y_ref = vec![0.0_f32; cfg.d_out];
        for o in 0..cfg.d_out {
            let mut acc = 0.0_f32;
            for j in 0..cfg.d_in {
                acc += w[o * cfg.d_in + j] * x[j];
            }
            y_ref[o] = acc;
        }
        let y = dora.forward(&x).unwrap();
        for (i, (&a, &b)) in y.iter().zip(&y_ref).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "forward[{i}] = {a} disagrees with W'·x = {b}"
            );
        }
    }
}
