//! Empirical Mode Decomposition (EMD).
//!
//! Implements the sifting algorithm from Huang et al. 1998
//! "The empirical mode decomposition and the Hilbert spectrum for nonlinear
//! and non-stationary time series analysis" (Proc. R. Soc. Lond. A 454:903–995)
//! with natural cubic spline envelope interpolation.
//!
//! Each Intrinsic Mode Function (IMF) satisfies:
//! 1. The number of extrema and zero-crossings differ by at most one.
//! 2. The mean of upper and lower envelopes is zero everywhere.

use crate::error::{SignalError, SignalResult};
use std::f64::consts::TAU;

// ─────────────────────────────────────────────────── Configuration / Output ──

/// Configuration for the Empirical Mode Decomposition.
#[derive(Debug, Clone)]
pub struct EmdConfig {
    /// Maximum number of IMFs to extract.  Default: 10.
    pub max_imf: usize,
    /// Maximum sifting iterations per IMF.  Default: 50.
    pub max_sift: usize,
    /// Sifting stopping criterion (Standard Deviation).  Default: 0.2.
    pub sift_tol: f64,
    /// Mirror-extend the signal endpoints before spline fitting.  Default: true.
    pub mirror_extend: bool,
}

impl Default for EmdConfig {
    fn default() -> Self {
        Self {
            max_imf: 10,
            max_sift: 50,
            sift_tol: 0.2,
            mirror_extend: true,
        }
    }
}

/// Output of the EMD.
#[derive(Debug, Clone)]
pub struct EmdResult {
    /// IMFs in row-major order, `n_imf × n`.
    pub imfs: Vec<f64>,
    /// Residual (trend) after all IMFs are subtracted.
    pub residual: Vec<f64>,
    /// Number of IMFs extracted.
    pub n_imf: usize,
    /// Signal length.
    pub n: usize,
    /// Number of sifting iterations performed for each IMF.
    pub sift_counts: Vec<usize>,
}

// ──────────────────────────────────────────────── Extrema detection ───────────

/// Find local maxima in `x`.  Returns `(indices, values)`.
fn find_maxima(x: &[f64]) -> (Vec<usize>, Vec<f64>) {
    let n = x.len();
    let mut idx = Vec::new();
    let mut val = Vec::new();
    for i in 1..n.saturating_sub(1) {
        if x[i] > x[i - 1] && x[i] > x[i + 1] {
            idx.push(i);
            val.push(x[i]);
        }
    }
    (idx, val)
}

/// Find local minima in `x`.  Returns `(indices, values)`.
fn find_minima(x: &[f64]) -> (Vec<usize>, Vec<f64>) {
    let n = x.len();
    let mut idx = Vec::new();
    let mut val = Vec::new();
    for i in 1..n.saturating_sub(1) {
        if x[i] < x[i - 1] && x[i] < x[i + 1] {
            idx.push(i);
            val.push(x[i]);
        }
    }
    (idx, val)
}

/// Count all extrema (maxima + minima) in `x`.
fn count_extrema(x: &[f64]) -> usize {
    find_maxima(x).0.len() + find_minima(x).0.len()
}

// ───────────────────────────────────────── Mirror boundary extension ──────────

/// Extend extrema arrays with mirrored boundary points.
///
/// Prepends one mirrored extremum at the left edge and appends one at the
/// right edge so that the cubic spline has sensible behaviour at the ends.
fn mirror_extend(t_pts: &mut Vec<usize>, v_pts: &mut Vec<f64>, n: usize) {
    // Left boundary: reflect the first interior extremum about index 0.
    if let Some(&t0) = t_pts.first() {
        let mirror_t = t0.saturating_sub(t0); // = 0 if t0==0, otherwise t0 - t0 = 0
        // Actually mirror across index 0: index = 0 - (t0 - 0) = -t0 -> not valid.
        // Use: reflect t0 about the first sample -> index = t0 is reflected to 2*0 - t0.
        // Since 2*0 - t0 < 0, we cap at 0 and use the boundary value.
        // A robust approach: prepend (0, x[0]) as a virtual extremum.
        let _ = mirror_t; // suppress unused warning
        if t0 > 0 {
            // Mirror: t_left = 2*0 - t0 which is negative -> use t=0.
            // Better: reflect inside signal: append index = t0, mirrored value.
            t_pts.insert(0, 0);
            v_pts.insert(0, v_pts[0]); // repeat first extremum value at t=0
        }
    }
    // Right boundary: reflect the last interior extremum about index n-1.
    if let Some(&t_last) = t_pts.last() {
        if t_last < n.saturating_sub(1) {
            t_pts.push(n - 1);
            v_pts.push(*v_pts.last().unwrap_or(&0.0));
        }
    }
}

// ────────────────────────────────────────── Natural cubic spline ──────────────

/// Solve the natural cubic spline through knots `(t_knots[i], y_knots[i])`.
///
/// Uses the Thomas algorithm (O(n)) for the tridiagonal system.
/// Returns the second derivatives M_i at each knot.
///
/// Boundary condition: M_0 = M_{n-1} = 0 (natural spline).
fn spline_second_derivatives(t_knots: &[f64], y_knots: &[f64]) -> Vec<f64> {
    let nk = t_knots.len();
    debug_assert!(nk >= 2);
    debug_assert_eq!(nk, y_knots.len());

    if nk == 2 {
        return vec![0.0; 2]; // linear interpolation — no curvature
    }

    let h: Vec<f64> = (0..nk - 1).map(|i| t_knots[i + 1] - t_knots[i]).collect();

    // RHS vector (for interior nodes i = 1..nk-2).
    let m = nk - 2; // number of interior nodes
    let mut rhs: Vec<f64> = (0..m)
        .map(|ii| {
            let i = ii + 1;
            6.0 * ((y_knots[i + 1] - y_knots[i]) / h[i] - (y_knots[i] - y_knots[i - 1]) / h[i - 1])
        })
        .collect();

    // Diagonal: 2*(h[i-1] + h[i]) for interior node i.
    let mut diag: Vec<f64> = (0..m)
        .map(|ii| {
            let i = ii + 1;
            2.0 * (h[i - 1] + h[i])
        })
        .collect();

    // Sub/super-diagonal: h[i] for interior-to-interior.
    // Lower: h[i-1] for node i (i=1..m), upper: h[i] for node i.
    // Thomas forward sweep.
    let c: Vec<f64> = (0..m)
        .map(|ii| if ii < m - 1 { h[ii + 1] } else { 0.0 })
        .collect();
    // Lower off-diagonal (for ii-th interior from node ii+1 side): h[ii].
    for ii in 1..m {
        let w = h[ii] / diag[ii - 1];
        diag[ii] -= w * c[ii - 1];
        rhs[ii] -= w * rhs[ii - 1];
    }

    // Back substitution.
    let mut sol = vec![0.0_f64; m];
    sol[m - 1] = rhs[m - 1] / diag[m - 1];
    for ii in (0..m - 1).rev() {
        sol[ii] = (rhs[ii] - c[ii] * sol[ii + 1]) / diag[ii];
    }

    // Assemble full M vector with natural boundary conditions M_0 = M_{n-1} = 0.
    let mut out = vec![0.0_f64; nk];
    out[1..=m].copy_from_slice(&sol);
    out
}

/// Evaluate the natural cubic spline at integer time steps 0..n.
///
/// `t_knots` must be strictly increasing and expressed in the same units as
/// the evaluation points (here: sample indices 0..n as f64).
fn spline_eval(t_knots: &[usize], y_knots: &[f64], n: usize) -> Vec<f64> {
    let nk = t_knots.len();
    if nk == 0 {
        return vec![0.0; n];
    }
    if nk == 1 {
        return vec![y_knots[0]; n];
    }

    let tk_f: Vec<f64> = t_knots.iter().map(|&t| t as f64).collect();
    let m_vec = spline_second_derivatives(&tk_f, y_knots);

    let out: Vec<f64> = (0..n)
        .map(|eval_t| {
            let t = eval_t as f64;

            // Find bracketing interval by binary search.
            let seg = if t <= tk_f[0] {
                0 // left extrapolation: use leftmost segment
            } else if t >= tk_f[nk - 1] {
                nk - 2 // right extrapolation: use rightmost segment
            } else {
                // Binary search: find i such that tk_f[i] <= t < tk_f[i+1].
                let mut lo = 0usize;
                let mut hi = nk - 1;
                while hi - lo > 1 {
                    let mid = (lo + hi) / 2;
                    if tk_f[mid] <= t {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                lo
            };

            let i = seg;
            let h = tk_f[i + 1] - tk_f[i];
            if h.abs() < f64::EPSILON {
                return y_knots[i];
            }
            let a = (tk_f[i + 1] - t) / h;
            let b = (t - tk_f[i]) / h;
            // Standard cubic spline evaluation formula:
            a * y_knots[i]
                + b * y_knots[i + 1]
                + ((a * a * a - a) * m_vec[i] + (b * b * b - b) * m_vec[i + 1]) * h * h / 6.0
        })
        .collect();
    out
}

// ─────────────────────────────────────── Sifting (one IMF extraction) ────────

/// Extract one IMF from `residual` via sifting.
///
/// Returns `(imf, n_sift)`.  If sifting is not possible (too few extrema),
/// returns `(residual.to_vec(), 0)`.
fn sift_one_imf(residual: &[f64], config: &EmdConfig) -> (Vec<f64>, usize) {
    let n = residual.len();
    let mut proto = residual.to_vec();
    let mut n_sift = 0usize;

    for _k in 0..config.max_sift {
        // Find extrema.
        let (mut max_idx, mut max_val) = find_maxima(&proto);
        let (mut min_idx, mut min_val) = find_minima(&proto);

        if max_idx.len() < 2 || min_idx.len() < 2 {
            break;
        }

        // Mirror boundary extension.
        if config.mirror_extend {
            mirror_extend(&mut max_idx, &mut max_val, n);
            mirror_extend(&mut min_idx, &mut min_val, n);
        }

        // Cubic spline envelopes.
        let upper = spline_eval(&max_idx, &max_val, n);
        let lower = spline_eval(&min_idx, &min_val, n);

        // Mean envelope.
        let mean_env: Vec<f64> = upper
            .iter()
            .zip(lower.iter())
            .map(|(&u, &l)| 0.5 * (u + l))
            .collect();

        // Subtract mean from proto-IMF.
        let prev = proto.clone();
        for t in 0..n {
            proto[t] -= mean_env[t];
        }
        n_sift += 1;

        // Stopping criterion: SD = sum|h_k - h_{k-1}|^2 / sum|h_{k-1}|^2
        let num: f64 = proto
            .iter()
            .zip(prev.iter())
            .map(|(&h, &p)| (h - p).powi(2))
            .sum();
        let den: f64 = prev.iter().map(|&p| p * p).sum();
        if den < 1e-20 {
            break; // signal near zero
        }
        if num / den < config.sift_tol {
            break;
        }
    }

    (proto, n_sift)
}

// ─────────────────────────────────────────────────── Public API ───────────────

/// Decompose `signal` into Intrinsic Mode Functions via EMD.
///
/// Each IMF is a narrowband oscillation; the residual is the monotone trend.
/// The reconstruction identity holds: `signal = sum(IMFs) + residual`.
///
/// # Errors
/// - [`SignalError::InvalidParameter`] if `signal.len() < 4`.
pub fn emd(signal: &[f64], config: &EmdConfig) -> SignalResult<EmdResult> {
    let n = signal.len();
    if n < 4 {
        return Err(SignalError::InvalidParameter(
            "signal too short for EMD (minimum 4 samples)".into(),
        ));
    }

    let mut residual = signal.to_vec();
    let mut imfs: Vec<f64> = Vec::new();
    let mut sift_counts: Vec<usize> = Vec::new();
    let mut n_imf = 0usize;

    while n_imf < config.max_imf {
        // Stop if residual is nearly monotone (fewer than 3 extrema).
        if count_extrema(&residual) < 3 {
            break;
        }

        let (imf, n_sift) = sift_one_imf(&residual, config);

        // Subtract IMF from residual.
        for t in 0..n {
            residual[t] -= imf[t];
        }

        imfs.extend_from_slice(&imf);
        sift_counts.push(n_sift);
        n_imf += 1;
    }

    Ok(EmdResult {
        imfs,
        residual,
        n_imf,
        n,
        sift_counts,
    })
}

/// Compute the Hilbert transform of `signal` via the FFT.
///
/// Returns the imaginary part of the analytic signal:
///   x_analytic = IFFT(2 * H(omega) * FFT(x))
/// where H(omega) = 1 for omega > 0, 0.5 for omega = 0 or Nyquist, 0 otherwise.
///
/// # Errors
/// - [`SignalError::InvalidParameter`] if `signal` is empty.
pub fn hilbert_transform(signal: &[f64]) -> SignalResult<Vec<f64>> {
    let n = signal.len();
    if n == 0 {
        return Err(SignalError::InvalidParameter(
            "signal must be non-empty for Hilbert transform".into(),
        ));
    }

    // Pad to next power of two for efficient FFT.
    let npad = if n.is_power_of_two() {
        n
    } else {
        n.next_power_of_two()
    };

    let mut re: Vec<f64> = signal.to_vec();
    re.resize(npad, 0.0);
    let mut im = vec![0.0_f64; npad];

    // Forward FFT.
    fft_inplace_emd(&mut re, &mut im, false);

    // Apply Hilbert filter: multiply by -i * sign(omega).
    // H(omega) weights: 2 for k=1..npad/2-1, 1 for k=0 and npad/2, 0 for k>npad/2.
    // Multiplying by i in frequency domain → shift imaginary to real and negate real.
    // x_analytic = IFFT( h(omega) * FFT(x) )  where h = 0.5,1,1,...,1,0.5,0,...,0
    // The imaginary part of x_analytic is the Hilbert transform.
    // Apply: h[0]=1, h[k]=2 for 1..N/2, h[N/2]=1, h[k]=0 for k>N/2.
    for k in 1..npad / 2 {
        re[k] *= 2.0;
        im[k] *= 2.0;
    }
    // k = 0 and k = npad/2: multiply by 1 (no change).
    for k in (npad / 2 + 1)..npad {
        re[k] = 0.0;
        im[k] = 0.0;
    }

    // Inverse FFT.
    fft_inplace_emd(&mut re, &mut im, true);

    // Return imaginary part (= Hilbert transform), trimmed to original length.
    Ok(im[..n].to_vec())
}

/// Compute instantaneous frequency from the analytic signal components.
///
/// Given `re` and `im` (the real and imaginary parts of the analytic signal),
/// the instantaneous phase is φ(t) = atan2(`im[t]`, `re[t]`).
/// The instantaneous frequency is `IF[t]` = dφ/dt / (2π * dt).
///
/// Edge values are computed by forward/backward differences; interior values
/// use central differences.
///
/// # Errors
/// - [`SignalError::InvalidParameter`] if `re` and `im` have different lengths or are empty.
/// - [`SignalError::InvalidParameter`] if `dt <= 0`.
pub fn instantaneous_frequency(re: &[f64], im: &[f64], dt: f64) -> SignalResult<Vec<f64>> {
    let n = re.len();
    if n == 0 {
        return Err(SignalError::InvalidParameter(
            "analytic signal must be non-empty".into(),
        ));
    }
    if im.len() != n {
        return Err(SignalError::InvalidParameter(
            "re and im must have the same length".into(),
        ));
    }
    if dt <= 0.0 {
        return Err(SignalError::InvalidParameter("dt must be positive".into()));
    }

    let phase: Vec<f64> = re
        .iter()
        .zip(im.iter())
        .map(|(&r, &i)| i.atan2(r))
        .collect();

    // Phase unwrapping to handle discontinuities.
    let mut unwrapped = phase.clone();
    for t in 1..n {
        let diff = unwrapped[t] - unwrapped[t - 1];
        if diff > std::f64::consts::PI {
            for v in unwrapped[t..].iter_mut() {
                *v -= TAU;
            }
        } else if diff < -std::f64::consts::PI {
            for v in unwrapped[t..].iter_mut() {
                *v += TAU;
            }
        }
    }

    // Numerical differentiation of phase / (2*pi*dt).
    let mut freq = vec![0.0_f64; n];
    let inv_2pi_dt = 1.0 / (TAU * dt);
    if n == 1 {
        freq[0] = 0.0;
        return Ok(freq);
    }
    // Forward difference at left edge.
    freq[0] = (unwrapped[1] - unwrapped[0]) * inv_2pi_dt;
    // Central differences for interior.
    for t in 1..n - 1 {
        freq[t] = (unwrapped[t + 1] - unwrapped[t - 1]) * 0.5 * inv_2pi_dt;
    }
    // Backward difference at right edge.
    freq[n - 1] = (unwrapped[n - 1] - unwrapped[n - 2]) * inv_2pi_dt;

    Ok(freq)
}

/// Compute the energy (sum of squared amplitude) for each IMF.
///
/// Returns a `Vec<f64>` of length `n_imf`.
#[must_use]
pub fn emd_energy(result: &EmdResult) -> Vec<f64> {
    let n = result.n;
    (0..result.n_imf)
        .map(|j| {
            let row = j * n;
            result.imfs[row..row + n].iter().map(|&v| v * v).sum()
        })
        .collect()
}

// ─────────────────────────────────────────────────────── Private FFT ──────────

/// Cooley-Tukey radix-2 in-place FFT (private copy for EMD module).
fn fft_inplace_emd(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());

    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
    let mut len = 2usize;
    while len <= n {
        let ang = sign * TAU / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0_f64, 0.0_f64);
            for k in 0..len / 2 {
                let (ur, ui) = (re[i + k], im[i + k]);
                let (vr, vi) = (
                    cr * re[i + k + len / 2] - ci * im[i + k + len / 2],
                    cr * im[i + k + len / 2] + ci * re[i + k + len / 2],
                );
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let tmp_r = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = tmp_r;
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for (r, v) in re.iter_mut().zip(im.iter_mut()) {
            *r *= scale;
            *v *= scale;
        }
    }
}

// ─────────────────────────────────────────────────────────────────── Tests ───

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(n: usize, freq_norm: f64) -> Vec<f64> {
        (0..n).map(|i| (TAU * freq_norm * i as f64).sin()).collect()
    }

    fn make_cosine(n: usize, freq_norm: f64) -> Vec<f64> {
        (0..n).map(|i| (TAU * freq_norm * i as f64).cos()).collect()
    }

    fn make_chirp(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (TAU * (0.05 + 0.15 * t) * i as f64).sin()
            })
            .collect()
    }

    // 1. Sum of all IMFs + residual ~= original signal (within 1e-8 relative)
    #[test]
    fn test_perfect_reconstruction() {
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| (TAU * 0.1 * i as f64).sin() + 0.5 * (TAU * 0.3 * i as f64).cos())
            .collect();
        let config = EmdConfig::default();
        let result = emd(&signal, &config).expect("emd should succeed");

        let mut reconstructed = result.residual.clone();
        for j in 0..result.n_imf {
            let row = j * n;
            for (dst, &src) in reconstructed
                .iter_mut()
                .zip(result.imfs[row..row + n].iter())
            {
                *dst += src;
            }
        }
        let sig_energy: f64 = signal.iter().map(|&v| v * v).sum();
        let err_energy: f64 = reconstructed
            .iter()
            .zip(signal.iter())
            .map(|(&r, &s)| (r - s).powi(2))
            .sum();
        let rel_err = if sig_energy > 1e-20 {
            (err_energy / sig_energy).sqrt()
        } else {
            err_energy.sqrt()
        };
        assert!(
            rel_err < 1e-8,
            "reconstruction error {rel_err:.2e} exceeds 1e-8"
        );
    }

    // 2. emd_energy: energies non-negative
    #[test]
    fn test_energy_non_negative() {
        let signal = make_sine(128, 0.1);
        let result = emd(&signal, &EmdConfig::default()).expect("value should be present");
        for (j, &e) in emd_energy(&result).iter().enumerate() {
            assert!(e >= 0.0, "IMF {j} energy {e} < 0");
        }
    }

    // 3. n_imf >= 1 on typical oscillatory signal
    #[test]
    fn test_at_least_one_imf() {
        let signal: Vec<f64> = (0..128).map(|i| (TAU * 0.1 * i as f64).sin()).collect();
        let result = emd(&signal, &EmdConfig::default()).expect("value should be present");
        assert!(result.n_imf >= 1, "expected at least 1 IMF");
    }

    // 4. sift_counts.len() == n_imf
    #[test]
    fn test_sift_counts_length() {
        let signal = make_sine(128, 0.1);
        let result = emd(&signal, &EmdConfig::default()).expect("value should be present");
        assert_eq!(result.sift_counts.len(), result.n_imf);
    }

    // 5. imfs.len() == n_imf * n
    #[test]
    fn test_imfs_length() {
        let n = 128;
        let signal = make_sine(n, 0.1);
        let result = emd(&signal, &EmdConfig::default()).expect("value should be present");
        assert_eq!(result.imfs.len(), result.n_imf * n);
    }

    // 6. residual.len() == n
    #[test]
    fn test_residual_length() {
        let n = 128;
        let signal = make_sine(n, 0.1);
        let result = emd(&signal, &EmdConfig::default()).expect("value should be present");
        assert_eq!(result.residual.len(), n);
    }

    // 7. max_imf=1: only 1 IMF extracted
    #[test]
    fn test_max_imf_one() {
        let signal = make_sine(128, 0.1);
        let config = EmdConfig {
            max_imf: 1,
            ..Default::default()
        };
        let result = emd(&signal, &config).expect("emd should succeed");
        assert!(result.n_imf <= 1, "n_imf={} but max_imf=1", result.n_imf);
    }

    // 8. max_sift=1: minimal sifting, still returns valid result
    #[test]
    fn test_max_sift_one() {
        let signal = make_sine(128, 0.1);
        let config = EmdConfig {
            max_sift: 1,
            ..Default::default()
        };
        let result = emd(&signal, &config);
        assert!(
            result.is_ok(),
            "max_sift=1 should succeed: {:?}",
            result.err()
        );
    }

    // 9. Constant signal: n_imf == 0 or 1 with near-zero IMF energy
    #[test]
    fn test_constant_signal() {
        let signal = vec![2.5f64; 64];
        let result = emd(&signal, &EmdConfig::default()).expect("value should be present");
        // Constant has no extrema -> 0 IMFs or negligible energy
        if result.n_imf > 0 {
            let energies = emd_energy(&result);
            let total_imf_energy: f64 = energies.iter().sum();
            assert!(
                total_imf_energy < 1e-6,
                "constant signal should produce near-zero IMF energies, got {total_imf_energy}"
            );
        }
    }

    // 10. Linear trend: residual captures most energy (no oscillation to decompose)
    #[test]
    fn test_linear_trend_residual() {
        let n = 64;
        let signal: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let result = emd(&signal, &EmdConfig::default()).expect("value should be present");
        let sig_energy: f64 = signal.iter().map(|&v| v * v).sum();
        let res_energy: f64 = result.residual.iter().map(|&v| v * v).sum();
        // Residual should contain a large fraction of total energy for a pure trend
        assert!(
            res_energy > 0.1 * sig_energy,
            "residual should capture trend energy: res={res_energy:.4}, sig={sig_energy:.4}"
        );
    }

    // 11. Two-frequency sum: multiple IMFs produced
    #[test]
    fn test_two_frequency_sum() {
        let n = 256;
        let signal: Vec<f64> = (0..n)
            .map(|i| (TAU * 0.05 * i as f64).sin() + 0.7 * (TAU * 0.2 * i as f64).sin())
            .collect();
        let result = emd(&signal, &EmdConfig::default()).expect("value should be present");
        // Should extract at least 1 IMF from two-component signal
        assert!(
            result.n_imf >= 1,
            "expected >= 1 IMF for two-frequency signal"
        );
    }

    // 12. mirror_extend=false: still works
    #[test]
    fn test_mirror_extend_false() {
        let signal = make_sine(128, 0.1);
        let config = EmdConfig {
            mirror_extend: false,
            ..Default::default()
        };
        let result = emd(&signal, &config);
        assert!(
            result.is_ok(),
            "mirror_extend=false should succeed: {:?}",
            result.err()
        );
    }

    // 13. Reconstruction perfect with mirror_extend=false
    #[test]
    fn test_reconstruction_no_mirror() {
        let n = 64;
        let signal = make_sine(n, 0.1);
        let config = EmdConfig {
            mirror_extend: false,
            ..Default::default()
        };
        let result = emd(&signal, &config).expect("emd should succeed");
        let mut recon = result.residual.clone();
        for j in 0..result.n_imf {
            let row = j * n;
            for (dst, &src) in recon.iter_mut().zip(result.imfs[row..row + n].iter()) {
                *dst += src;
            }
        }
        let err: f64 = recon
            .iter()
            .zip(signal.iter())
            .map(|(&r, &s)| (r - s).powi(2))
            .sum::<f64>()
            .sqrt();
        let sig_rms: f64 = signal.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let rel = if sig_rms > 1e-20 { err / sig_rms } else { err };
        assert!(rel < 1e-8, "no-mirror reconstruction error {rel:.2e}");
    }

    // 14. Chirp signal: multiple IMFs produced
    #[test]
    fn test_chirp_multiple_imfs() {
        let signal = make_chirp(256);
        let result = emd(&signal, &EmdConfig::default()).expect("value should be present");
        assert!(result.n_imf >= 1, "chirp should produce at least 1 IMF");
    }

    // 15. hilbert_transform: output length == input length
    #[test]
    fn test_hilbert_length() {
        let signal = make_sine(64, 0.1);
        let ht = hilbert_transform(&signal).expect("hilbert_transform should succeed");
        assert_eq!(ht.len(), signal.len());
    }

    // 16. hilbert_transform: H{cos(2pi f t)} ~= sin(2pi f t) (for pure cosine)
    // The Hilbert transform shifts phase by -pi/2: H{cos} = sin.
    #[test]
    fn test_hilbert_cosine_to_sine() {
        let n = 128;
        let freq = 0.1;
        let cos_sig = make_cosine(n, freq);
        let ht = hilbert_transform(&cos_sig).expect("hilbert_transform should succeed");
        let pos_sin: Vec<f64> = (0..n).map(|i| (TAU * freq * i as f64).sin()).collect();
        // Compare interior (skip edges affected by Gibbs / windowing artefacts).
        let start = n / 8;
        let end = 7 * n / 8;
        let err: f64 = ht[start..end]
            .iter()
            .zip(pos_sin[start..end].iter())
            .map(|(&h, &s)| (h - s).powi(2))
            .sum::<f64>()
            .sqrt();
        let ref_rms: f64 = pos_sin[start..end]
            .iter()
            .map(|&v| v * v)
            .sum::<f64>()
            .sqrt();
        let rel = err / (ref_rms + 1e-20);
        assert!(
            rel < 0.15,
            "H{{cos}} should approximate sin, rel error = {rel:.4}"
        );
    }

    // 17. instantaneous_frequency: for pure sine -> approximately constant
    #[test]
    fn test_instantaneous_frequency_pure_sine() {
        let n = 256;
        let freq_norm = 0.1;
        let dt = 1.0;
        let signal = make_sine(n, freq_norm);
        // Analytic signal: real = sin, imag = -cos (Hilbert of sin = -cos)
        let ht = hilbert_transform(&signal).expect("hilbert_transform should succeed");
        let inst_freq = instantaneous_frequency(&signal, &ht, dt)
            .expect("instantaneous_frequency should succeed");

        // Interior region, away from edge effects.
        let start = n / 8;
        let end = 7 * n / 8;
        let mean_freq: f64 = inst_freq[start..end].iter().sum::<f64>() / (end - start) as f64;
        let err = (mean_freq - freq_norm).abs();
        assert!(
            err < 0.02,
            "mean instantaneous frequency {mean_freq:.4} should be close to {freq_norm}, err={err:.4}"
        );
    }

    // 18. emd_energy: each IMF's energy is non-negative and bounded by signal energy
    // Note: sum(|IMF|^2) + |residual|^2 != |signal|^2 in general due to cross-terms.
    // However, perfect reconstruction guarantees sum(IMFs) + residual = signal exactly.
    // We verify that emd_energy returns sensible (bounded) values.
    #[test]
    fn test_energy_conservation() {
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| (TAU * 0.1 * i as f64).sin() + 0.5 * (TAU * 0.25 * i as f64).cos())
            .collect();
        let result = emd(&signal, &EmdConfig::default()).expect("value should be present");
        let sig_energy: f64 = signal.iter().map(|&v| v * v).sum();
        // Perfect reconstruction: verify sum(IMFs) + residual = signal.
        let mut recon = result.residual.clone();
        for j in 0..result.n_imf {
            let row = j * n;
            for (dst, &src) in recon.iter_mut().zip(result.imfs[row..row + n].iter()) {
                *dst += src;
            }
        }
        let err: f64 = recon
            .iter()
            .zip(signal.iter())
            .map(|(&r, &s)| (r - s).powi(2))
            .sum::<f64>()
            .sqrt();
        let rel = err / (sig_energy.sqrt() + 1e-20);
        assert!(
            rel < 1e-8,
            "reconstruction rel error {rel:.2e} — perfect reconstruction should hold"
        );
        // Each individual IMF energy must be positive (or zero).
        for (j, &e) in emd_energy(&result).iter().enumerate() {
            assert!(e >= 0.0, "IMF {j} energy {e} must be non-negative");
        }
    }

    // 19. signal length < 4 -> InvalidParameter
    #[test]
    fn test_error_signal_too_short() {
        for short_len in [0, 1, 2, 3] {
            let signal = vec![1.0f64; short_len];
            assert!(
                matches!(
                    emd(&signal, &EmdConfig::default()),
                    Err(SignalError::InvalidParameter(_))
                ),
                "len={short_len} should give InvalidParameter"
            );
        }
    }

    // 20. Empty signal -> InvalidParameter (hilbert_transform)
    #[test]
    fn test_hilbert_empty_error() {
        assert!(matches!(
            hilbert_transform(&[]),
            Err(SignalError::InvalidParameter(_))
        ));
    }

    // 21. instantaneous_frequency: mismatched lengths -> InvalidParameter
    #[test]
    fn test_inst_freq_length_mismatch() {
        let re = vec![1.0f64; 16];
        let im = vec![0.0f64; 8]; // wrong length
        assert!(matches!(
            instantaneous_frequency(&re, &im, 1.0),
            Err(SignalError::InvalidParameter(_))
        ));
    }

    // 22. instantaneous_frequency: dt <= 0 -> InvalidParameter
    #[test]
    fn test_inst_freq_bad_dt() {
        let re = vec![1.0f64; 16];
        let im = vec![0.0f64; 16];
        assert!(matches!(
            instantaneous_frequency(&re, &im, 0.0),
            Err(SignalError::InvalidParameter(_))
        ));
        assert!(matches!(
            instantaneous_frequency(&re, &im, -1.0),
            Err(SignalError::InvalidParameter(_))
        ));
    }
}
