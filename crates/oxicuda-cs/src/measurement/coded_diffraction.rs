//! Coded Diffraction Patterns (CDP) and Wirtinger-Flow phase retrieval (Candès, Li & Soltanolkotabi 2015).
//!
//! In ptychography / coherent diffraction imaging one measures squared magnitudes of a
//! *modulated* Fourier transform of the unknown (complex) signal `x ∈ ℂⁿ`:
//!
//! ```text
//! y_{l} = | F (d_l ⊙ x) |² ,   l = 1, …, L
//! ```
//!
//! where `d_l ∈ ℂⁿ` is a random *coded diffraction mask* (a per-pixel modulation) and
//! `F` is the unnormalised DFT. Stacking the `L` patterns gives `m = L·n` real,
//! phaseless measurements — enough (for `L ≳ 4`) to recover `x` up to a global phase.
//!
//! This module provides:
//!
//! - [`CodedDiffraction`] — a set of random complex masks defining the forward operator.
//! - [`CodedDiffraction::forward`] — the phaseless measurement `y = |F(d_l ⊙ x)|²`.
//! - [`CodedDiffraction::wirtinger_flow`] — recovery by Wirtinger Flow: spectral
//!   initialisation followed by gradient descent on the intensity-domain loss
//!   `½ Σ_l ‖ |A_l x|² − y_l ‖²` using the *Wirtinger* gradient.
//!
//! Signals are stored as interleaved real/imaginary `f64` pairs: `x[2k]` is `Re(x_k)`,
//! `x[2k+1]` is `Im(x_k)`. All transforms are implemented from scratch (an O(n²) DFT),
//! so the module depends only on `std`.
//!
//! # References
//!
//! - E. J. Candès, X. Li & M. Soltanolkotabi (2015), "Phase Retrieval from Coded
//!   Diffraction Patterns", Applied and Computational Harmonic Analysis 39(2):277-299.
//! - E. J. Candès, X. Li & M. Soltanolkotabi (2015), "Phase Retrieval via Wirtinger
//!   Flow: Theory and Algorithms", IEEE Trans. Information Theory 61(4):1985-2007.

use crate::error::{CsError, CsResult};
use crate::handle::LcgRng;

// ---------------------------------------------------------------------------
// Lightweight complex helpers on interleaved [re, im] slices
// ---------------------------------------------------------------------------

#[inline]
fn cmul(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
    (ar * br - ai * bi, ar * bi + ai * br)
}

/// Unnormalised DFT of an interleaved complex vector `x` (length `2n`).
///
/// Returns interleaved `X_k = Σ_j x_j exp(−2πi jk/n)`.
fn dft(x: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; 2 * n];
    let two_pi = std::f64::consts::TAU;
    for k in 0..n {
        let mut sr = 0.0_f64;
        let mut si = 0.0_f64;
        for j in 0..n {
            let angle = -two_pi * (j as f64) * (k as f64) / (n as f64);
            let (wr, wi) = (angle.cos(), angle.sin());
            let xr = x[2 * j];
            let xi = x[2 * j + 1];
            let (pr, pi) = cmul(xr, xi, wr, wi);
            sr += pr;
            si += pi;
        }
        out[2 * k] = sr;
        out[2 * k + 1] = si;
    }
    out
}

/// Inverse (unnormalised by `1/n`) DFT of an interleaved complex vector.
fn idft(x: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; 2 * n];
    let two_pi = std::f64::consts::TAU;
    let inv_n = 1.0 / (n as f64);
    for k in 0..n {
        let mut sr = 0.0_f64;
        let mut si = 0.0_f64;
        for j in 0..n {
            let angle = two_pi * (j as f64) * (k as f64) / (n as f64);
            let (wr, wi) = (angle.cos(), angle.sin());
            let xr = x[2 * j];
            let xi = x[2 * j + 1];
            let (pr, pi) = cmul(xr, xi, wr, wi);
            sr += pr;
            si += pi;
        }
        out[2 * k] = sr * inv_n;
        out[2 * k + 1] = si * inv_n;
    }
    out
}

// ---------------------------------------------------------------------------
// Coded diffraction operator
// ---------------------------------------------------------------------------

/// A coded-diffraction forward operator: `L` random complex masks of length `n`.
#[derive(Debug, Clone)]
pub struct CodedDiffraction {
    /// Signal length `n`.
    pub n: usize,
    /// Number of coded patterns `L`.
    pub n_masks: usize,
    /// Masks, interleaved complex, length `n_masks · 2n` (mask `l` occupies
    /// `[l·2n .. (l+1)·2n]`).
    masks: Vec<f64>,
}

/// Distribution from which coded-diffraction mask entries are drawn.
#[derive(Debug, Clone, Copy)]
pub enum MaskKind {
    /// Octanary masks of Candès et al.: `d = b1 · b2` with `b1 ∈ {1, −1, i, −i}` and
    /// `b2 ∈ {√2/2, √3}` (used in the CDP phase-retrieval guarantees).
    Octanary,
    /// Uniform-phase unit-modulus masks `d = exp(iθ)`, `θ ∼ U[0, 2π)`.
    UniformPhase,
    /// Real Rademacher masks `d ∈ {1, −1}` (no imaginary part).
    Rademacher,
}

impl CodedDiffraction {
    /// Construct `n_masks` random masks of length `n`.
    ///
    /// # Errors
    /// [`CsError::InvalidParameter`] for `n == 0` or `n_masks == 0`.
    pub fn new(n: usize, n_masks: usize, kind: MaskKind, rng: &mut LcgRng) -> CsResult<Self> {
        if n == 0 || n_masks == 0 {
            return Err(CsError::InvalidParameter(
                "coded diffraction: n and n_masks must be > 0".into(),
            ));
        }
        let mut masks = vec![0.0_f64; n_masks * 2 * n];
        for l in 0..n_masks {
            for j in 0..n {
                let (re, im) = sample_mask_entry(kind, rng);
                masks[l * 2 * n + 2 * j] = re;
                masks[l * 2 * n + 2 * j + 1] = im;
            }
        }
        Ok(Self { n, n_masks, masks })
    }

    /// Total number of real measurements `m = L · n`.
    #[must_use]
    pub fn n_measurements(&self) -> usize {
        self.n_masks * self.n
    }

    /// Mask `l` as an interleaved complex slice of length `2n`.
    fn mask(&self, l: usize) -> &[f64] {
        &self.masks[l * 2 * self.n..(l + 1) * 2 * self.n]
    }

    /// Apply the `l`-th linear measurement operator `A_l x = F (d_l ⊙ x)` (complex).
    ///
    /// `x` is interleaved complex of length `2n`. Returns interleaved complex `2n`.
    fn apply(&self, l: usize, x: &[f64]) -> Vec<f64> {
        let d = self.mask(l);
        let mut mod_x = vec![0.0_f64; 2 * self.n];
        for j in 0..self.n {
            let (pr, pi) = cmul(d[2 * j], d[2 * j + 1], x[2 * j], x[2 * j + 1]);
            mod_x[2 * j] = pr;
            mod_x[2 * j + 1] = pi;
        }
        dft(&mod_x, self.n)
    }

    /// Apply the adjoint `A_lᴴ z = d̄_l ⊙ F⁻¹(z)` (with the `1/n`-scaled inverse DFT
    /// so that `A_lᴴ A_l` is well-conditioned). `z` interleaved complex length `2n`.
    fn apply_adjoint(&self, l: usize, z: &[f64]) -> Vec<f64> {
        let d = self.mask(l);
        let inv = idft(z, self.n);
        let mut out = vec![0.0_f64; 2 * self.n];
        for j in 0..self.n {
            // d̄ ⊙ inv: conjugate the mask.
            let (pr, pi) = cmul(d[2 * j], -d[2 * j + 1], inv[2 * j], inv[2 * j + 1]);
            out[2 * j] = pr;
            out[2 * j + 1] = pi;
        }
        out
    }

    /// Phaseless forward measurement `y = |A_l x|²` stacked over all masks.
    ///
    /// `x` is interleaved complex (length `2n`). Output is real, length `m = L·n`,
    /// laid out mask-major: `y[l·n + k] = |(A_l x)_k|²`.
    ///
    /// # Errors
    /// [`CsError::DimensionMismatch`] if `x.len() != 2n`.
    pub fn forward(&self, x: &[f64]) -> CsResult<Vec<f64>> {
        if x.len() != 2 * self.n {
            return Err(CsError::DimensionMismatch {
                a: x.len(),
                b: 2 * self.n,
            });
        }
        let mut y = vec![0.0_f64; self.n_measurements()];
        for l in 0..self.n_masks {
            let ax = self.apply(l, x);
            for k in 0..self.n {
                let re = ax[2 * k];
                let im = ax[2 * k + 1];
                y[l * self.n + k] = re * re + im * im;
            }
        }
        Ok(y)
    }

    /// Recover `x` (up to a global phase) from phaseless measurements `y` by Wirtinger Flow.
    ///
    /// * `y`        — measurements from [`forward`](Self::forward), length `m = L·n`.
    /// * `cfg`      — algorithm configuration.
    /// * `rng`      — RNG for the spectral-initialisation power iteration.
    ///
    /// Returns the recovered interleaved-complex signal (length `2n`).
    ///
    /// # Errors
    /// * [`CsError::DimensionMismatch`] if `y.len() != m`.
    /// * [`CsError::NumericalInstability`] if the spectral initialiser degenerates.
    pub fn wirtinger_flow(
        &self,
        y: &[f64],
        cfg: &WirtingerConfig,
        rng: &mut LcgRng,
    ) -> CsResult<Vec<f64>> {
        let m = self.n_measurements();
        if y.len() != m {
            return Err(CsError::DimensionMismatch { a: y.len(), b: m });
        }

        // Normalisation constant λ² = (1/m) Σ y_l ·  (mean intensity scaled by n).
        // Candès WF uses λ² = n · Σ y / Σ ‖a‖²; here ‖a_l‖² aggregates to L·n per pixel.
        let sum_y: f64 = y.iter().sum();
        let lambda_sq = (self.n as f64) * sum_y / (m as f64);
        let lambda = lambda_sq.max(1e-30).sqrt();

        // ── Spectral initialisation: leading eigenvector of  Y = (1/m) Σ y_r a_r a_rᴴ. ──
        // Power iteration using only matrix-vector products via the operators.
        let mut z = random_unit_complex(self.n, rng);
        for _ in 0..cfg.power_iters {
            let yz = self.apply_y(y, &z, m);
            let nrm = cnorm(&yz);
            if nrm < 1e-300 {
                return Err(CsError::NumericalInstability(
                    "WF spectral init: degenerate leading eigenvector".into(),
                ));
            }
            for v in z.iter_mut() {
                *v = 0.0;
            }
            for j in 0..2 * self.n {
                z[j] = yz[j] / nrm;
            }
        }
        // Scale the initial guess to the correct magnitude.
        for v in z.iter_mut() {
            *v *= lambda;
        }

        // ── Wirtinger gradient descent. ──
        // Loss f(x) = (1/2m) Σ_r ( |a_rᴴ x|² − y_r )².
        // Wirtinger gradient: ∇f = (1/m) Σ_r ( |a_rᴴ x|² − y_r ) (a_rᴴ x) a_r.
        let step = cfg.step_size / lambda_sq.max(1e-30);
        for _ in 0..cfg.max_iter {
            let grad = self.wf_gradient(y, &z, m);
            for j in 0..2 * self.n {
                z[j] -= step * grad[j];
            }
        }

        Ok(z)
    }

    /// Compute `Y z = (1/m) Σ_r y_r a_r (a_rᴴ z)` for the spectral initialiser.
    fn apply_y(&self, y: &[f64], z: &[f64], m: usize) -> Vec<f64> {
        let inv_m = 1.0 / (m as f64);
        let mut acc = vec![0.0_f64; 2 * self.n];
        for l in 0..self.n_masks {
            // a_rᴴ z for every pixel r in this mask = (A_l z)*  ... we need per-pixel.
            let az = self.apply(l, z); // (A_l z)_k for all k
            // weight each pixel by y and form contribution back through adjoint.
            let mut w = vec![0.0_f64; 2 * self.n];
            for k in 0..self.n {
                let yr = y[l * self.n + k];
                // (a_rᴴ z) is the k-th entry of A_l z; multiply by y_r.
                w[2 * k] = yr * az[2 * k];
                w[2 * k + 1] = yr * az[2 * k + 1];
            }
            let contrib = self.apply_adjoint(l, &w);
            for j in 0..2 * self.n {
                acc[j] += contrib[j];
            }
        }
        for v in acc.iter_mut() {
            *v *= inv_m;
        }
        acc
    }

    /// Wirtinger gradient of the intensity loss at `z`.
    fn wf_gradient(&self, y: &[f64], z: &[f64], m: usize) -> Vec<f64> {
        let inv_m = 1.0 / (m as f64);
        let mut grad = vec![0.0_f64; 2 * self.n];
        for l in 0..self.n_masks {
            let az = self.apply(l, z);
            let mut resid = vec![0.0_f64; 2 * self.n];
            for k in 0..self.n {
                let re = az[2 * k];
                let im = az[2 * k + 1];
                let mag_sq = re * re + im * im;
                let factor = mag_sq - y[l * self.n + k]; // (|a^H x|² − y_r)
                resid[2 * k] = factor * re;
                resid[2 * k + 1] = factor * im;
            }
            let contrib = self.apply_adjoint(l, &resid);
            for j in 0..2 * self.n {
                grad[j] += contrib[j];
            }
        }
        for v in grad.iter_mut() {
            *v *= inv_m;
        }
        grad
    }
}

/// Configuration for Wirtinger Flow recovery.
#[derive(Debug, Clone)]
pub struct WirtingerConfig {
    /// Number of power-iteration steps for spectral initialisation (default `50`).
    pub power_iters: usize,
    /// Number of gradient-descent iterations (default `400`).
    pub max_iter: usize,
    /// Base step size (scaled internally by `1/λ²`) (default `0.2`).
    pub step_size: f64,
}

impl Default for WirtingerConfig {
    fn default() -> Self {
        Self {
            power_iters: 50,
            max_iter: 400,
            step_size: 0.2,
        }
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Frobenius norm of an interleaved complex vector.
fn cnorm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Random unit-norm interleaved complex vector of `n` entries.
fn random_unit_complex(n: usize, rng: &mut LcgRng) -> Vec<f64> {
    let mut z = vec![0.0_f64; 2 * n];
    for v in z.iter_mut() {
        *v = rng.next_normal();
    }
    let nrm = cnorm(&z).max(1e-300);
    for v in z.iter_mut() {
        *v /= nrm;
    }
    z
}

/// Sample one coded-diffraction mask entry `(re, im)` from `kind`.
fn sample_mask_entry(kind: MaskKind, rng: &mut LcgRng) -> (f64, f64) {
    match kind {
        MaskKind::Octanary => {
            // b1 ∈ {1, −1, i, −i} (uniform), b2 ∈ {√2/2 w.p. 4/5, √3 w.p. 1/5}.
            let phase = rng.next_usize(4);
            let (pr, pi) = match phase {
                0 => (1.0, 0.0),
                1 => (-1.0, 0.0),
                2 => (0.0, 1.0),
                _ => (0.0, -1.0),
            };
            let b2 = if rng.next_f64() < 0.8 {
                std::f64::consts::FRAC_1_SQRT_2
            } else {
                3.0_f64.sqrt()
            };
            (pr * b2, pi * b2)
        }
        MaskKind::UniformPhase => {
            let theta = std::f64::consts::TAU * rng.next_f64();
            (theta.cos(), theta.sin())
        }
        MaskKind::Rademacher => {
            if rng.next_bool() {
                (1.0, 0.0)
            } else {
                (-1.0, 0.0)
            }
        }
    }
}

/// Align two interleaved-complex vectors up to a global phase and return the relative
/// reconstruction error `‖x̂ e^{iφ} − x‖ / ‖x‖` minimised over the global phase `φ`.
///
/// Useful for tests since phase retrieval recovers `x` only up to `e^{iφ}`.
///
/// # Errors
/// [`CsError::DimensionMismatch`] if the two vectors differ in length.
pub fn phase_aligned_error(x_hat: &[f64], x: &[f64]) -> CsResult<f64> {
    if x_hat.len() != x.len() {
        return Err(CsError::DimensionMismatch {
            a: x_hat.len(),
            b: x.len(),
        });
    }
    // Optimal phase: φ = arg(⟨x_hat, x⟩) where ⟨·,·⟩ = Σ conj(x_hat) x.
    let mut ipr = 0.0_f64;
    let mut ipi = 0.0_f64;
    let n = x.len() / 2;
    for k in 0..n {
        let (ar, ai) = (x_hat[2 * k], x_hat[2 * k + 1]);
        let (br, bi) = (x[2 * k], x[2 * k + 1]);
        // conj(a) * b = (ar − i ai)(br + i bi).
        ipr += ar * br + ai * bi;
        ipi += ar * bi - ai * br;
    }
    let mag = (ipr * ipr + ipi * ipi).sqrt();
    let (cr, ci) = if mag > 1e-300 {
        (ipr / mag, ipi / mag)
    } else {
        (1.0, 0.0)
    };
    // x_hat · e^{iφ} with e^{iφ} = (cr, ci).
    let mut err_sq = 0.0_f64;
    let mut x_sq = 0.0_f64;
    for k in 0..n {
        let (ar, ai) = (x_hat[2 * k], x_hat[2 * k + 1]);
        let (rr, ri) = cmul(ar, ai, cr, ci);
        let dr = rr - x[2 * k];
        let di = ri - x[2 * k + 1];
        err_sq += dr * dr + di * di;
        x_sq += x[2 * k] * x[2 * k] + x[2 * k + 1] * x[2 * k + 1];
    }
    Ok((err_sq / x_sq.max(1e-300)).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complex_signal(reals: &[f64], imags: &[f64]) -> Vec<f64> {
        let mut x = Vec::with_capacity(reals.len() * 2);
        for (r, i) in reals.iter().zip(imags.iter()) {
            x.push(*r);
            x.push(*i);
        }
        x
    }

    #[test]
    fn dft_idft_round_trip() {
        let n = 6;
        let x = complex_signal(
            &[1.0, -2.0, 3.0, 0.5, -1.0, 2.0],
            &[0.0, 1.0, -1.0, 0.0, 2.0, -0.5],
        );
        let back = idft(&dft(&x, n), n);
        for k in 0..2 * n {
            assert!(
                (back[k] - x[k]).abs() < 1e-9,
                "k={k}: {} vs {}",
                back[k],
                x[k]
            );
        }
    }

    #[test]
    fn constructor_rejects_zero_dims() {
        let mut rng = LcgRng::new(1);
        assert!(CodedDiffraction::new(0, 4, MaskKind::Octanary, &mut rng).is_err());
        assert!(CodedDiffraction::new(8, 0, MaskKind::Octanary, &mut rng).is_err());
    }

    #[test]
    fn forward_shapes() {
        let mut rng = LcgRng::new(2);
        let cdp = CodedDiffraction::new(8, 5, MaskKind::UniformPhase, &mut rng).expect("ok");
        assert_eq!(cdp.n_measurements(), 40);
        let x = vec![0.1_f64; 16];
        let y = cdp.forward(&x).expect("ok");
        assert_eq!(y.len(), 40);
        assert!(y.iter().all(|&v| v >= 0.0), "intensities non-negative");
    }

    #[test]
    fn forward_dimension_mismatch() {
        let mut rng = LcgRng::new(3);
        let cdp = CodedDiffraction::new(8, 4, MaskKind::Octanary, &mut rng).expect("ok");
        assert!(matches!(
            cdp.forward(&[0.0; 10]),
            Err(CsError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn forward_phase_invariant_intensity() {
        // |A (e^{iφ} x)|² = |A x|²: the global phase must not change the measurements.
        let mut rng = LcgRng::new(4);
        let cdp = CodedDiffraction::new(6, 4, MaskKind::UniformPhase, &mut rng).expect("ok");
        let x = complex_signal(
            &[1.0, 2.0, -1.0, 0.5, 0.0, 3.0],
            &[0.0, -1.0, 1.0, 0.0, 2.0, 0.0],
        );
        let y0 = cdp.forward(&x).expect("ok");
        // Rotate x by φ = π/3.
        let (cr, ci) = (
            std::f64::consts::FRAC_PI_3.cos(),
            std::f64::consts::FRAC_PI_3.sin(),
        );
        let mut x_rot = vec![0.0_f64; x.len()];
        for k in 0..6 {
            let (rr, ri) = cmul(x[2 * k], x[2 * k + 1], cr, ci);
            x_rot[2 * k] = rr;
            x_rot[2 * k + 1] = ri;
        }
        let y1 = cdp.forward(&x_rot).expect("ok");
        for (a, b) in y0.iter().zip(y1.iter()) {
            assert!((a - b).abs() < 1e-8, "{a} vs {b}");
        }
    }

    #[test]
    fn phase_aligned_error_zero_for_rotation() {
        let x = complex_signal(&[1.0, -2.0, 0.5], &[0.5, 1.0, -1.0]);
        let (cr, ci) = (0.6_f64, 0.8_f64); // unit modulus
        let mut x_rot = vec![0.0_f64; x.len()];
        for k in 0..3 {
            let (rr, ri) = cmul(x[2 * k], x[2 * k + 1], cr, ci);
            x_rot[2 * k] = rr;
            x_rot[2 * k + 1] = ri;
        }
        let err = phase_aligned_error(&x_rot, &x).expect("ok");
        assert!(err < 1e-9, "err = {err}");
    }

    #[test]
    fn phase_aligned_error_dim_mismatch() {
        assert!(matches!(
            phase_aligned_error(&[1.0, 0.0], &[1.0, 0.0, 0.0, 0.0]),
            Err(CsError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn wirtinger_flow_recovers_real_signal() {
        // Small real signal, octanary masks, oversampled (L = 10) ⇒ recovery up to
        // global phase. Coded-diffraction phase retrieval needs enough patterns;
        // more masks widen the basin of attraction of the spectral initialiser.
        let n = 5usize;
        let mut rng = LcgRng::new(20);
        let cdp = CodedDiffraction::new(n, 10, MaskKind::Octanary, &mut rng).expect("ok");
        let x = complex_signal(&[1.0, -1.0, 0.5, 2.0, -0.5], &[0.0; 5]);
        let y = cdp.forward(&x).expect("ok");
        let cfg = WirtingerConfig {
            power_iters: 120,
            max_iter: 2500,
            step_size: 0.15,
        };
        let mut rng2 = LcgRng::new(99);
        let x_hat = cdp.wirtinger_flow(&y, &cfg, &mut rng2).expect("ok");
        let err = phase_aligned_error(&x_hat, &x).expect("ok");
        assert!(err < 0.15, "relative error = {err}");
    }

    #[test]
    fn wirtinger_flow_recovers_complex_signal() {
        let n = 4usize;
        let mut rng = LcgRng::new(31);
        let cdp = CodedDiffraction::new(n, 8, MaskKind::Octanary, &mut rng).expect("ok");
        let x = complex_signal(&[1.0, 0.0, -1.0, 0.5], &[0.5, 1.0, 0.0, -1.0]);
        let y = cdp.forward(&x).expect("ok");
        let cfg = WirtingerConfig {
            power_iters: 100,
            max_iter: 2000,
            step_size: 0.15,
        };
        let mut rng2 = LcgRng::new(7);
        let x_hat = cdp.wirtinger_flow(&y, &cfg, &mut rng2).expect("ok");
        let err = phase_aligned_error(&x_hat, &x).expect("ok");
        assert!(err < 0.2, "relative error = {err}");
    }

    #[test]
    fn wirtinger_flow_dimension_mismatch() {
        let mut rng = LcgRng::new(5);
        let cdp = CodedDiffraction::new(8, 4, MaskKind::Octanary, &mut rng).expect("ok");
        let mut rng2 = LcgRng::new(6);
        assert!(matches!(
            cdp.wirtinger_flow(&[0.0; 10], &WirtingerConfig::default(), &mut rng2),
            Err(CsError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn recovered_signal_reproduces_measurements() {
        // The intensity of the WF estimate should closely match y.
        let n = 4usize;
        let mut rng = LcgRng::new(40);
        let cdp = CodedDiffraction::new(n, 7, MaskKind::Octanary, &mut rng).expect("ok");
        let x = complex_signal(&[2.0, -1.0, 0.0, 1.0], &[0.0, 0.5, -0.5, 0.0]);
        let y = cdp.forward(&x).expect("ok");
        let cfg = WirtingerConfig {
            power_iters: 100,
            max_iter: 2000,
            step_size: 0.15,
        };
        let mut rng2 = LcgRng::new(8);
        let x_hat = cdp.wirtinger_flow(&y, &cfg, &mut rng2).expect("ok");
        let y_hat = cdp.forward(&x_hat).expect("ok");
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for (a, b) in y_hat.iter().zip(y.iter()) {
            num += (a - b) * (a - b);
            den += b * b;
        }
        let rel = (num / den.max(1e-30)).sqrt();
        assert!(rel < 0.2, "measurement reproduction error = {rel}");
    }

    #[test]
    fn rademacher_masks_are_real() {
        let mut rng = LcgRng::new(50);
        let cdp = CodedDiffraction::new(6, 3, MaskKind::Rademacher, &mut rng).expect("ok");
        for l in 0..3 {
            let m = cdp.mask(l);
            for j in 0..6 {
                assert_eq!(m[2 * j + 1], 0.0, "imag part must be zero");
                assert!((m[2 * j].abs() - 1.0).abs() < 1e-12, "must be ±1");
            }
        }
    }

    #[test]
    fn zero_signal_zero_measurements() {
        let mut rng = LcgRng::new(60);
        let cdp = CodedDiffraction::new(5, 4, MaskKind::Octanary, &mut rng).expect("ok");
        let y = cdp.forward(&vec![0.0_f64; 10]).expect("ok");
        assert!(y.iter().all(|&v| v == 0.0));
    }
}
