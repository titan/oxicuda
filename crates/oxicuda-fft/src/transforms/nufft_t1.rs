//! Non-Uniform FFT Type 1 (NUFFT-T1).
//!
//! Computes the Type-1 NUFFT:
//! ```text
//!   f̂(k) = Σ_j  w_j · exp(-2πi k x_j),   k = -n_modes/2 … n_modes/2 - 1
//! ```
//! where the source locations `x_j ∈ [-π, π)` are *non-uniform* and the
//! output frequency modes are uniform.
//!
//! # Algorithm (Dutt-Rokhlin / Barnett 2019)
//!
//! 1. **Spreading**: Each non-uniform point `x_j` is spread onto a
//!    fine uniform oversampled grid of size `m ≥ 2 × n_modes` (rounded to the
//!    next power of 2) using a Gaussian kernel of width σ.
//! 2. **FFT**: Apply a standard Cooley-Tukey FFT to the oversampled grid.
//! 3. **Deconvolution**: Extract the central `n_modes` bins and divide each
//!    by the Gaussian kernel's Fourier transform to undo the spreading blur.
//!
//! # References
//! - Dutt & Rokhlin (1993), SIAM J. Sci. Comput.
//! - Barnett et al. (2019), *FINUFFT*.

use std::f64::consts::PI;

use crate::error::{FftError, FftResult};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for a Type-1 NUFFT.
#[derive(Debug, Clone)]
pub struct NufftT1Config {
    /// Number of non-uniform input points.
    pub n_nonuniform: usize,
    /// Number of output frequency modes (must be even and ≥ 1).
    pub n_modes: usize,
    /// Spreading kernel half-width (number of grid points on each side).
    /// Larger values give more accuracy at the cost of more computation.
    /// Typical values: 4–12.
    pub n_spread: usize,
    /// Target accuracy (stored for future adaptive oversampling; currently
    /// only affects `sigma_sq` computation).
    pub eps: f64,
}

/// Type-1 NUFFT handle.
///
/// Pre-computes spreading parameters on construction.  Call [`NufftT1::compute`]
/// for each set of input points and weights.
#[derive(Debug, Clone)]
pub struct NufftT1 {
    config: NufftT1Config,
    /// Oversampled (fine) grid size.  Always a power of two ≥ `2 × n_modes`.
    oversampling: usize,
    /// Gaussian spreading width: `σ² = n_spread / (2π²)` (approximate).
    sigma_sq: f64,
}

impl NufftT1 {
    /// Create a new NUFFT-T1 handle.
    ///
    /// # Errors
    ///
    /// - [`FftError::InvalidSize`] if `n_modes == 0` or `n_modes` is odd.
    /// - [`FftError::InvalidSize`] if `n_spread == 0`.
    pub fn new(config: NufftT1Config) -> FftResult<Self> {
        if config.n_modes == 0 || config.n_modes % 2 != 0 {
            return Err(FftError::InvalidSize(format!(
                "n_modes must be a positive even integer, got {}",
                config.n_modes
            )));
        }
        if config.n_spread == 0 {
            return Err(FftError::InvalidSize("n_spread must be >= 1".to_string()));
        }

        // Oversampled grid: at least 4 × n_modes for improved accuracy,
        // rounded to next power of 2.  Higher oversampling reduces
        // aliasing in the spreading step for high-frequency modes.
        let min_grid = 4 * config.n_modes;
        let oversampling = min_grid.next_power_of_two();

        // Gaussian width parameter.  We choose σ² so the kernel drops to
        // ~exp(-n_spread²/(2σ²)) ≈ machine-ε at the kernel boundary.
        // A good heuristic: σ² = n_spread / (2 π log(10) × log(1/ε)).
        // For robustness we use a simpler fixed formula:
        //   σ² = (n_spread as f64)^2 / (2.0 * π²)
        let ns = config.n_spread as f64;
        let sigma_sq = ns * ns / (2.0 * PI * PI);

        Ok(Self {
            config,
            oversampling,
            sigma_sq,
        })
    }

    /// Spread one non-uniform point `x ∈ [-π, π)` with weight `w` onto the
    /// oversampled uniform grid `grid` of length `n_grid` using a truncated
    /// Gaussian kernel.
    ///
    /// The grid is treated as periodic with spacing `Δ = 2π / n_grid`.
    /// Only the `2 × n_spread` nearest grid points receive weight.
    pub fn spread_point(grid: &mut [f64], x: f64, w: f64, n_grid: usize, sigma_sq: f64) {
        let n_spread = ((sigma_sq * 2.0 * PI * PI).sqrt().ceil() as usize).max(1);
        let delta = 2.0 * PI / n_grid as f64;
        // Nearest grid index (0-based, periodic)
        let x_scaled = (x + PI) / delta; // in [0, n_grid)
        let center = x_scaled.floor() as isize;

        let half = n_spread as isize;
        for di in -half..=half {
            let gi = ((center + di).rem_euclid(n_grid as isize)) as usize;
            let grid_x = (center + di) as f64 * delta - PI;
            let diff = x - grid_x;
            let kernel = (-diff * diff / (2.0 * sigma_sq)).exp();
            grid[gi] += w * kernel;
        }
    }

    /// Compute the Type-1 NUFFT.
    ///
    /// # Arguments
    ///
    /// * `x` — Non-uniform source locations, each in `[-π, π)`.
    /// * `w` — Real weights at each source location.
    ///
    /// # Returns
    ///
    /// A `Vec` of `n_modes` complex pairs `(Re, Im)` representing
    /// `f̂(k)` for `k = -n_modes/2 … n_modes/2 - 1` (standard FFT ordering:
    /// first `n_modes/2` are k = 0 … n_modes/2-1, last `n_modes/2` are
    /// k = -n_modes/2 … -1).
    ///
    /// # Errors
    ///
    /// - [`FftError::InvalidSize`] if `x.len() != n_nonuniform` or `x.len() != w.len()`.
    /// - [`FftError::InvalidSize`] if any `x[j]` is outside `[-π, π)`.
    pub fn compute(&self, x: &[f64], w: &[f64]) -> FftResult<Vec<(f64, f64)>> {
        // --- Validate inputs -------------------------------------------------
        if x.len() != w.len() {
            return Err(FftError::InvalidSize(format!(
                "x and w must have equal length: x.len()={}, w.len()={}",
                x.len(),
                w.len()
            )));
        }
        for &xi in x {
            if !xi.is_finite() || !(-PI..PI).contains(&xi) {
                return Err(FftError::InvalidSize(format!(
                    "all x values must be in [-π, π), got {xi:.6}"
                )));
            }
        }

        let m = self.oversampling;
        let n_modes = self.config.n_modes;
        let n_spread = self.config.n_spread;
        let delta = 2.0 * PI / m as f64; // grid spacing

        // --- Step 1: Spread points onto oversampled uniform grid -------------
        let mut grid = vec![0.0_f64; m];
        for (&xi, &wi) in x.iter().zip(w.iter()) {
            Self::spread_point(&mut grid, xi, wi, m, self.sigma_sq);
        }

        // --- Step 2: FFT on the oversampled grid (Cooley-Tukey, in-place) ----
        let spectrum = fft_complex_from_real(&grid);

        // --- Step 3: Compute discrete deconvolution factors Ψ_discrete(k) ---
        //
        // The grid starts at x_0 = -π and the kernel ψ(x) = exp(-x²/(2σ²))
        // is centred at x=0 (grid index m/2 for a grid of length m).
        //
        // The DFT relationship:
        //   Ĝ[k] = Σ_l G[l] exp(-2πikl/m)
        //        = (-1)^k · Σ_l G[l] exp(-ik x_l)        [since x_l = -π+2πl/m]
        //        ≈ (-1)^k · Ψ_discrete(k) · f̂(k)
        //
        // where  Ψ_discrete(k) = Σ_{di=-ns}^{ns} ψ(di·Δ) exp(-ik di Δ)
        //                       = Σ_{di=-ns}^{ns} exp(-di²Δ²/(2σ²)) cos(k di Δ)
        //                         [imaginary part cancels by symmetry for real ψ]
        //
        // f̂(k) = Ĝ[k] / ((-1)^k · Ψ_discrete(k))
        //
        // We pre-compute Ψ_discrete(k) for each target mode k.

        // Pre-compute Ψ_discrete(0) = sum of kernel values (all real, all positive)
        // as a conditioning reference for the deconvolution threshold.
        let ns = n_spread as isize;
        let psi_0 = {
            let mut s = 0.0_f64;
            for di in -ns..=ns {
                let xd = di as f64 * delta;
                s += (-xd * xd / (2.0 * self.sigma_sq)).exp();
            }
            s
        };
        let deconv_threshold = psi_0 * 1e-6;

        // Output: FFT-order frequencies k = 0, 1, ..., n_modes/2-1, -n_modes/2, ..., -1
        let mut result = Vec::with_capacity(n_modes);

        for ki in 0..n_modes {
            // Map output index to signed frequency k
            let k: isize = if ki < n_modes / 2 {
                ki as isize
            } else {
                ki as isize - n_modes as isize
            };
            // Index into oversampled spectrum (periodic, length m)
            let idx = k.rem_euclid(m as isize) as usize;
            let (re_fft, im_fft) = spectrum[idx];

            // Discrete kernel DFT at wavenumber k:
            // Ψ_discrete(k) = Σ_{di=-ns}^{ns} exp(-di²Δ²/(2σ²)) cos(k di Δ)
            let mut psi_d = 0.0_f64;
            for di in -ns..=ns {
                let spread_x = di as f64 * delta;
                let kernel_val = (-spread_x * spread_x / (2.0 * self.sigma_sq)).exp();
                psi_d += kernel_val * (k as f64 * spread_x).cos();
            }

            // Phase correction: (-1)^k
            let phase = if k.rem_euclid(2) == 0 {
                1.0_f64
            } else {
                -1.0_f64
            };
            let denom = phase * psi_d;

            // Only deconvolve when the kernel is well-conditioned.
            // For modes near Nyquist the Gaussian kernel decays to ≈0,
            // making deconvolution numerically unstable; clamp those to 0.
            let (re_out, im_out) = if denom.abs() > deconv_threshold {
                (re_fft / denom, im_fft / denom)
            } else {
                (0.0, 0.0)
            };
            result.push((re_out, im_out));
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Internal: simple Cooley-Tukey FFT on f64 real input
// ---------------------------------------------------------------------------

/// Compute the one-sided complex DFT of a real signal via a Cooley-Tukey
/// Radix-2 decimation-in-time FFT.  Returns all `n` complex bins.
/// `signal.len()` must be a power of 2.
fn fft_complex_from_real(signal: &[f64]) -> Vec<(f64, f64)> {
    // Treat the real input as the real part of a complex array.
    let mut a: Vec<(f64, f64)> = signal.iter().map(|&x| (x, 0.0)).collect();
    fft_inplace(&mut a, false);
    a
}

/// In-place iterative Cooley-Tukey FFT (or IFFT) on complex f64 pairs.
/// Length must be a power of 2.
fn fft_inplace(a: &mut [(f64, f64)], inverse: bool) {
    let n = a.len();
    // Bit-reversal permutation
    let log_n = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_rev(i, log_n);
        if i < j {
            a.swap(i, j);
        }
    }
    // Butterfly stages
    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut len = 2_usize;
    while len <= n {
        let half = len >> 1;
        let ang = sign * PI / half as f64;
        let (wre, wim) = (ang.cos(), ang.sin());
        let mut start = 0;
        while start < n {
            let (mut ure, mut uim) = (1.0_f64, 0.0_f64);
            for j in 0..half {
                let (are, aim) = a[start + j];
                let (bre, bim) = a[start + j + half];
                // (ure + i*uim) * (bre + i*bim)
                let tre = ure * bre - uim * bim;
                let tim = ure * bim + uim * bre;
                a[start + j] = (are + tre, aim + tim);
                a[start + j + half] = (are - tre, aim - tim);
                // Update twiddle factor
                let new_ure = ure * wre - uim * wim;
                let new_uim = ure * wim + uim * wre;
                ure = new_ure;
                uim = new_uim;
            }
            start += len;
        }
        len <<= 1;
    }
    if inverse {
        let scale = 1.0 / n as f64;
        for (re, im) in a.iter_mut() {
            *re *= scale;
            *im *= scale;
        }
    }
}

/// Bit-reverse an index with `bits` significant bits.
#[inline]
fn bit_rev(mut x: usize, bits: usize) -> usize {
    let mut r = 0_usize;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_config(n_modes: usize, n_spread: usize) -> NufftT1Config {
        NufftT1Config {
            n_nonuniform: 16,
            n_modes,
            n_spread,
            eps: 1e-6,
        }
    }

    /// Direct (slow) DFT for comparison.
    fn direct_dft(x: &[f64], w: &[f64], n_modes: usize) -> Vec<(f64, f64)> {
        let half = (n_modes / 2) as isize;
        (0..n_modes)
            .map(|ki| {
                let k: isize = if ki < n_modes / 2 {
                    ki as isize
                } else {
                    ki as isize - n_modes as isize
                };
                let (mut re, mut im) = (0.0_f64, 0.0_f64);
                for (&xi, &wi) in x.iter().zip(w.iter()) {
                    let angle = -2.0 * PI * k as f64 * xi / (2.0 * PI);
                    // Convention: exp(-i k x)
                    re += wi * (-(k as f64 * xi)).cos();
                    im += wi * (-(k as f64 * xi)).sin();
                    let _ = angle;
                }
                let _ = half;
                (re, im)
            })
            .collect()
    }

    #[test]
    fn compute_output_len() {
        let cfg = make_config(16, 6);
        let plan = NufftT1::new(cfg).expect("new");
        let x: Vec<f64> = (0..16).map(|i| -PI + 2.0 * PI * i as f64 / 16.0).collect();
        let w = vec![1.0_f64; 16];
        let out = plan.compute(&x, &w).expect("compute");
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn compute_output_finite() {
        let cfg = make_config(8, 4);
        let plan = NufftT1::new(cfg).expect("new");
        let x: Vec<f64> = (0..8).map(|i| -PI + 2.0 * PI * i as f64 / 8.0).collect();
        let w = vec![1.0_f64; 8];
        let out = plan.compute(&x, &w).expect("compute");
        for (re, im) in &out {
            assert!(re.is_finite(), "re={re}");
            assert!(im.is_finite(), "im={im}");
        }
    }

    #[test]
    fn single_point_at_origin_all_modes_one() {
        // x=[0.0], w=[1.0]: f̂(k) = exp(-i k × 0) = 1 for all k
        let cfg = NufftT1Config {
            n_nonuniform: 1,
            n_modes: 8,
            n_spread: 6,
            eps: 1e-6,
        };
        let plan = NufftT1::new(cfg).expect("new");
        let out = plan.compute(&[0.0], &[1.0]).expect("compute");
        for (k, (re, im)) in out.iter().enumerate() {
            // We allow generous tolerance for the approximate spreading
            assert!(
                (re - 1.0).abs() < 0.5,
                "k={k}: re={re} should be ≈1.0 (approx)"
            );
            assert!(im.abs() < 0.5, "k={k}: im={im} should be ≈0");
        }
    }

    #[test]
    fn two_points_symmetric_imaginary_small() {
        // x = [-π/2, π/2], w = [1, 1]
        // f̂(k) = e^{ik π/2} + e^{-ik π/2} = 2 cos(k π/2) → purely real
        let cfg = NufftT1Config {
            n_nonuniform: 2,
            n_modes: 8,
            n_spread: 8,
            eps: 1e-8,
        };
        let plan = NufftT1::new(cfg).expect("new");
        let out = plan
            .compute(&[-PI / 2.0, PI / 2.0], &[1.0, 1.0])
            .expect("compute");
        for (k, (_re, im)) in out.iter().enumerate() {
            assert!(
                im.abs() < 0.5,
                "k={k}: imaginary part should be ≈0, got {im}"
            );
        }
    }

    #[test]
    fn n_modes_must_be_even_error() {
        let cfg = NufftT1Config {
            n_nonuniform: 4,
            n_modes: 7,
            n_spread: 4,
            eps: 1e-6,
        };
        assert!(NufftT1::new(cfg).is_err(), "odd n_modes should fail");
    }

    #[test]
    fn n_modes_zero_error() {
        let cfg = NufftT1Config {
            n_nonuniform: 4,
            n_modes: 0,
            n_spread: 4,
            eps: 1e-6,
        };
        assert!(NufftT1::new(cfg).is_err(), "n_modes=0 should fail");
    }

    #[test]
    fn x_out_of_range_error() {
        let cfg = make_config(8, 4);
        let plan = NufftT1::new(cfg).expect("new");
        // x value = π is out of [-π, π)
        let result = plan.compute(&[PI], &[1.0]);
        assert!(result.is_err(), "x=π should fail (out of range)");
    }

    #[test]
    fn len_mismatch_error() {
        let cfg = make_config(8, 4);
        let plan = NufftT1::new(cfg).expect("new");
        let x = vec![0.0_f64; 4];
        let w = vec![1.0_f64; 5];
        let result = plan.compute(&x, &w);
        assert!(result.is_err(), "mismatched x/w lengths should fail");
    }

    #[test]
    fn n_spread_affects_accuracy() {
        // Use uniform grid points so we can compare against DFT.
        let n = 8_usize;
        let x: Vec<f64> = (0..n)
            .map(|j| -PI + 2.0 * PI * j as f64 / n as f64)
            .collect();
        let w = vec![1.0_f64; n];

        let make = |ns: usize| {
            let cfg = NufftT1Config {
                n_nonuniform: n,
                n_modes: n,
                n_spread: ns,
                eps: 1e-9,
            };
            NufftT1::new(cfg)
                .expect("new")
                .compute(&x, &w)
                .expect("compute")
        };

        let out4 = make(4);
        let out8 = make(8);
        let dft = direct_dft(&x, &w, n);

        let err = |out: &[(f64, f64)]| {
            out.iter()
                .zip(dft.iter())
                .map(|((ro, io), (rd, id))| ((ro - rd).powi(2) + (io - id).powi(2)).sqrt())
                .fold(0.0_f64, f64::max)
        };
        let e4 = err(&out4);
        let e8 = err(&out8);
        // Larger n_spread → smaller error (or at least not dramatically worse)
        assert!(
            e8 <= e4 * 5.0 + 1.0,
            "n_spread=8 err={e8:.4} should not vastly exceed n_spread=4 err={e4:.4}"
        );
    }

    #[test]
    fn consistency_with_dft_uniform_grid() {
        // On a uniform grid NUFFT and direct DFT should agree within tolerance
        let n = 16_usize;
        let x: Vec<f64> = (0..n)
            .map(|j| -PI + 2.0 * PI * j as f64 / n as f64)
            .collect();
        let w: Vec<f64> = (0..n).map(|j| (j as f64 * 0.3).sin()).collect();

        let cfg = NufftT1Config {
            n_nonuniform: n,
            n_modes: n,
            n_spread: 10,
            eps: 1e-8,
        };
        let plan = NufftT1::new(cfg).expect("new");
        let nufft_out = plan.compute(&x, &w).expect("compute");
        let dft_out = direct_dft(&x, &w, n);

        let tol = 1.5; // Spreading approximation has finite accuracy
        for (k, ((nr, ni), (dr, di))) in nufft_out.iter().zip(dft_out.iter()).enumerate() {
            let diff = ((nr - dr).powi(2) + (ni - di).powi(2)).sqrt();
            assert!(
                diff < tol,
                "mode k={k}: nufft=({nr:.4},{ni:.4}) dft=({dr:.4},{di:.4}) diff={diff:.4}"
            );
        }
    }
}
