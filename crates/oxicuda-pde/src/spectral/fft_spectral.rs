//! Periodic pseudo-spectral method using a DFT (direct, O(N^2) — small N).
//!
//! For a periodic function on `[0, 2π)` sampled at `N` equally-spaced points,
//! the second derivative is computed via Fourier modes:
//! `\hat{u}''(k) = -k^2 \hat{u}(k)`.
//!
//! This module uses a direct DFT (no external FFT library).

use crate::error::{PdeError, PdeResult};

/// Compute `u_xx` at `N` periodic nodes using a direct DFT/IDFT.
///
/// Assumes the domain length is `L`, so wave numbers are `k_m = 2π m / L`
/// for `m = -N/2..N/2`.
pub fn periodic_diff2(u: &[f64], length: f64) -> PdeResult<Vec<f64>> {
    let n = u.len();
    if n < 2 {
        return Err(PdeError::InvalidGrid("need n >= 2 samples".into()));
    }
    if length <= 0.0 {
        return Err(PdeError::InvalidParameter {
            name: "length".into(),
            reason: "must be positive".into(),
        });
    }
    let two_pi = std::f64::consts::TAU;
    let nf = n as f64;
    // Forward DFT
    let mut re = vec![0.0_f64; n];
    let mut im = vec![0.0_f64; n];
    for k in 0..n {
        let mut sr = 0.0;
        let mut si = 0.0;
        for (j, &uj) in u.iter().enumerate().take(n) {
            let ang = -two_pi * (k as f64) * (j as f64) / nf;
            sr += uj * ang.cos();
            si += uj * ang.sin();
        }
        re[k] = sr;
        im[k] = si;
    }
    // Multiply by -(2π m / L)^2 (treating the upper half as negative frequencies)
    let half = n / 2;
    for k in 0..n {
        let m = if k <= half {
            k as i64
        } else {
            k as i64 - n as i64
        };
        let kx = two_pi * m as f64 / length;
        let factor = -kx * kx;
        re[k] *= factor;
        im[k] *= factor;
    }
    // Nyquist mode for even N must be zeroed for real-output (signs ambiguous).
    if n % 2 == 0 {
        re[half] = 0.0;
        im[half] = 0.0;
    }
    // Inverse DFT
    let mut out = vec![0.0; n];
    for (j, out_j) in out.iter_mut().enumerate().take(n) {
        let mut s = 0.0;
        for k in 0..n {
            let ang = two_pi * (k as f64) * (j as f64) / nf;
            s += re[k] * ang.cos() - im[k] * ang.sin();
        }
        *out_j = s / nf;
    }
    Ok(out)
}

/// Solve `-u''(x) = f(x)` with periodic BCs using DFT. The mean of `f` must be zero
/// (compatibility condition). Returns `u` with zero mean.
pub fn periodic_poisson_solve(f: &[f64], length: f64) -> PdeResult<Vec<f64>> {
    let n = f.len();
    if n < 2 {
        return Err(PdeError::InvalidGrid("need n >= 2 samples".into()));
    }
    if length <= 0.0 {
        return Err(PdeError::InvalidParameter {
            name: "length".into(),
            reason: "must be positive".into(),
        });
    }
    let two_pi = std::f64::consts::TAU;
    let nf = n as f64;
    // Forward DFT of f
    let mut re = vec![0.0_f64; n];
    let mut im = vec![0.0_f64; n];
    for k in 0..n {
        let mut sr = 0.0;
        let mut si = 0.0;
        for (j, &fj) in f.iter().enumerate().take(n) {
            let ang = -two_pi * (k as f64) * (j as f64) / nf;
            sr += fj * ang.cos();
            si += fj * ang.sin();
        }
        re[k] = sr;
        im[k] = si;
    }
    // Divide by k^2 to invert -u'' = f
    let half = n / 2;
    for k in 0..n {
        if k == 0 {
            re[k] = 0.0;
            im[k] = 0.0;
            continue;
        }
        let m = if k <= half {
            k as i64
        } else {
            k as i64 - n as i64
        };
        let kx = two_pi * m as f64 / length;
        let factor = 1.0 / (kx * kx);
        re[k] *= factor;
        im[k] *= factor;
    }
    if n % 2 == 0 {
        re[half] = 0.0;
        im[half] = 0.0;
    }
    // Inverse DFT
    let mut u = vec![0.0; n];
    for (j, uj) in u.iter_mut().enumerate().take(n) {
        let mut s = 0.0;
        for k in 0..n {
            let ang = two_pi * (k as f64) * (j as f64) / nf;
            s += re[k] * ang.cos() - im[k] * ang.sin();
        }
        *uj = s / nf;
    }
    Ok(u)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_diff2_cosine() {
        // u(x) = cos(2x) on [0, 2π), so u''(x) = -4 cos(2x)
        let two_pi = std::f64::consts::TAU;
        let n = 32;
        let u: Vec<f64> = (0..n)
            .map(|j| {
                let x = two_pi * j as f64 / n as f64;
                (2.0 * x).cos()
            })
            .collect();
        let uxx = periodic_diff2(&u, two_pi).expect("ok");
        for (j, &uxxj) in uxx.iter().enumerate() {
            let x = two_pi * j as f64 / n as f64;
            let expected = -4.0 * (2.0 * x).cos();
            assert!(
                (uxxj - expected).abs() < 1.0e-9,
                "j={j} got={uxxj} expected={expected}"
            );
        }
    }

    #[test]
    fn periodic_poisson_sine() {
        // -u'' = 4 sin(2x), u(x) = sin(2x)
        let two_pi = std::f64::consts::TAU;
        let n = 32;
        let f: Vec<f64> = (0..n)
            .map(|j| {
                let x = two_pi * j as f64 / n as f64;
                4.0 * (2.0 * x).sin()
            })
            .collect();
        let u = periodic_poisson_solve(&f, two_pi).expect("ok");
        for (j, &uj) in u.iter().enumerate() {
            let x = two_pi * j as f64 / n as f64;
            let expected = (2.0 * x).sin();
            assert!(
                (uj - expected).abs() < 1.0e-9,
                "j={j} got={uj} expected={expected}"
            );
        }
    }
}
