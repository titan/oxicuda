//! 2D periodic Poisson solver via FFT.
//!
//! Solves `−Δu = f` on the rectangle `[0, Lx] × [0, Ly]` with periodic boundary
//! conditions on both axes. The discrete grid is `nx × ny` with **power-of-two**
//! dimensions (the radix-2 Cooley-Tukey FFT used internally requires
//! `nx = 2^p`, `ny = 2^q`).
//!
//! # Method
//!
//! Expanding `f` and `u` in a Fourier series on the periodic grid, the equation
//! reads in spectral space
//!
//! ```text
//! ( kx² + ky² ) û(kx, ky)  =  f̂(kx, ky)
//! ```
//!
//! so
//!
//! ```text
//! û(kx, ky) = f̂(kx, ky) / ( kx² + ky² )    for (kx, ky) ≠ (0, 0)
//! û(0, 0)   = 0                              (gauge fix; mean of u is zero)
//! ```
//!
//! For a grid of `n` samples on `[0, L]` with the FFT convention used here the
//! signed wave numbers are
//!
//! ```text
//! kx_m = 2π m / Lx       for m = 0, 1, …, nx/2,  −nx/2+1, …, −1.
//! ```
//!
//! # Implementation
//!
//! A 2D FFT is built from a sequence of 1D radix-2 Cooley-Tukey FFTs (Cormen,
//! Leiserson, Rivest & Stein, *Introduction to Algorithms*, MIT 2009, §30.3).
//! For each row we run a 1D FFT, then for each column. The inverse transform
//! is identical except for sign conventions in the twiddle factors and a final
//! normalisation by `1 / (nx · ny)`.
//!
//! # References
//!
//! * Trefethen, *Spectral Methods in Matlab*, SIAM 2000, chapter 3.
//! * Cooley & Tukey, *An algorithm for the machine calculation of complex
//!   Fourier series*, Math. Comp. 19 (1965), 297-301.

use crate::error::{PdeError, PdeResult};

/// Configuration for the 2D FFT-based periodic Poisson solver.
#[derive(Debug, Clone, Copy)]
pub struct Fourier2dConfig {
    /// Number of grid points along x (power of two, ≥ 4).
    pub nx: usize,
    /// Number of grid points along y (power of two, ≥ 4).
    pub ny: usize,
    /// Domain length along x (must be positive).
    pub lx: f64,
    /// Domain length along y (must be positive).
    pub ly: f64,
}

impl Default for Fourier2dConfig {
    fn default() -> Self {
        Self {
            nx: 32,
            ny: 32,
            lx: 1.0,
            ly: 1.0,
        }
    }
}

#[inline]
fn is_power_of_two(n: usize) -> bool {
    n >= 1 && (n & (n - 1)) == 0
}

fn validate_config(cfg: &Fourier2dConfig) -> PdeResult<()> {
    if cfg.nx == 0 || cfg.ny == 0 {
        return Err(PdeError::InvalidGrid(format!(
            "fourier_2d requires nx>=4, ny>=4 (got nx={}, ny={})",
            cfg.nx, cfg.ny
        )));
    }
    if cfg.nx < 4 || cfg.ny < 4 {
        return Err(PdeError::InvalidGrid(format!(
            "fourier_2d requires nx>=4, ny>=4 (got nx={}, ny={})",
            cfg.nx, cfg.ny
        )));
    }
    if !is_power_of_two(cfg.nx) || !is_power_of_two(cfg.ny) {
        return Err(PdeError::InvalidGrid(format!(
            "fourier_2d requires power-of-two nx, ny (got nx={}, ny={})",
            cfg.nx, cfg.ny
        )));
    }
    if cfg.lx <= 0.0 || !cfg.lx.is_finite() {
        return Err(PdeError::InvalidParameter {
            name: "lx".into(),
            reason: "must be positive and finite".into(),
        });
    }
    if cfg.ly <= 0.0 || !cfg.ly.is_finite() {
        return Err(PdeError::InvalidParameter {
            name: "ly".into(),
            reason: "must be positive and finite".into(),
        });
    }
    Ok(())
}

/// In-place radix-2 iterative Cooley-Tukey FFT.
///
/// `re`, `im` are the real and imaginary parts of a length-`n` complex array,
/// where `n` must be a power of two. `sign = -1.0` performs the forward
/// transform `X_k = Σ x_j · exp(−2π i j k / n)`; `sign = +1.0` performs the
/// unnormalised inverse `Y_k = Σ x_j · exp(+2π i j k / n)`. The caller is
/// responsible for dividing by `n` after the inverse.
///
/// Returns `Err(PdeError::InvalidGrid)` if `n` is not a power of two or if
/// `re` and `im` have mismatched length.
fn fft_radix2(re: &mut [f64], im: &mut [f64], sign: f64) -> PdeResult<()> {
    let n = re.len();
    if im.len() != n {
        return Err(PdeError::DimensionMismatch {
            a: re.len(),
            b: im.len(),
        });
    }
    if !is_power_of_two(n) {
        return Err(PdeError::InvalidGrid(format!(
            "fft_radix2 length must be power of two, got {n}"
        )));
    }
    if n <= 1 {
        return Ok(());
    }
    // Bit-reversal permutation.
    let mut j = 0_usize;
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
    // Butterfly stages.
    let two_pi = std::f64::consts::TAU;
    let mut size = 2_usize;
    while size <= n {
        let half = size / 2;
        let theta = sign * two_pi / size as f64;
        let wpr = theta.cos();
        let wpi = theta.sin();
        let mut start = 0_usize;
        while start < n {
            // Precompute twiddle factors for this block.
            let mut wr = 1.0_f64;
            let mut wi = 0.0_f64;
            for k in 0..half {
                let i0 = start + k;
                let i1 = i0 + half;
                let tr = wr * re[i1] - wi * im[i1];
                let ti = wr * im[i1] + wi * re[i1];
                re[i1] = re[i0] - tr;
                im[i1] = im[i0] - ti;
                re[i0] += tr;
                im[i0] += ti;
                // Advance twiddle: w *= exp(i·theta)
                let new_wr = wr * wpr - wi * wpi;
                let new_wi = wr * wpi + wi * wpr;
                wr = new_wr;
                wi = new_wi;
                // suppress unused warnings if half==1 corner cases
                let _ = k;
            }
            start += size;
        }
        size <<= 1;
    }
    Ok(())
}

/// Run a 2D forward FFT in place. `re` / `im` are row-major `nx × ny`. `sign`
/// is the FFT sign convention as in `fft_radix2`.
fn fft2_in_place(re: &mut [f64], im: &mut [f64], nx: usize, ny: usize, sign: f64) -> PdeResult<()> {
    if re.len() != nx * ny || im.len() != nx * ny {
        return Err(PdeError::ShapeMismatch {
            expected: vec![nx * ny],
            got: vec![re.len()],
        });
    }
    // Row transforms (length-ny FFTs).
    let mut row_re = vec![0.0_f64; ny];
    let mut row_im = vec![0.0_f64; ny];
    for i in 0..nx {
        let base = i * ny;
        row_re.copy_from_slice(&re[base..base + ny]);
        row_im.copy_from_slice(&im[base..base + ny]);
        fft_radix2(&mut row_re, &mut row_im, sign)?;
        re[base..base + ny].copy_from_slice(&row_re);
        im[base..base + ny].copy_from_slice(&row_im);
    }
    // Column transforms (length-nx FFTs).
    let mut col_re = vec![0.0_f64; nx];
    let mut col_im = vec![0.0_f64; nx];
    for j in 0..ny {
        for i in 0..nx {
            col_re[i] = re[i * ny + j];
            col_im[i] = im[i * ny + j];
        }
        fft_radix2(&mut col_re, &mut col_im, sign)?;
        for i in 0..nx {
            re[i * ny + j] = col_re[i];
            im[i * ny + j] = col_im[i];
        }
    }
    Ok(())
}

/// Solve the 2D periodic Poisson equation `−Δu = f` on `[0, Lx] × [0, Ly]`.
///
/// `f` is a row-major `nx × ny` array; the returned vector has the same shape
/// and contains the zero-mean solution `u`.
///
/// # Errors
///
/// * `PdeError::InvalidGrid` if any of `nx`, `ny` is not a power of two or is
///   smaller than 4.
/// * `PdeError::InvalidParameter` if `lx` or `ly` is non-positive.
/// * `PdeError::ShapeMismatch` if `f.len() != cfg.nx * cfg.ny`.
pub fn solve_poisson_2d_fft(f: &[f64], cfg: &Fourier2dConfig) -> PdeResult<Vec<f64>> {
    validate_config(cfg)?;
    let nx = cfg.nx;
    let ny = cfg.ny;
    let n = nx * ny;
    if f.len() != n {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n],
            got: vec![f.len()],
        });
    }
    // Forward FFT of f
    let mut re: Vec<f64> = f.to_vec();
    let mut im: Vec<f64> = vec![0.0_f64; n];
    fft2_in_place(&mut re, &mut im, nx, ny, -1.0)?;
    // Divide by (kx² + ky²); zero the mean (gauge fix).
    let two_pi = std::f64::consts::TAU;
    let half_x = nx / 2;
    let half_y = ny / 2;
    for i in 0..nx {
        let m_x = if i <= half_x {
            i as i64
        } else {
            i as i64 - nx as i64
        };
        let kx = two_pi * m_x as f64 / cfg.lx;
        for j in 0..ny {
            let m_y = if j <= half_y {
                j as i64
            } else {
                j as i64 - ny as i64
            };
            let ky = two_pi * m_y as f64 / cfg.ly;
            let idx = i * ny + j;
            if i == 0 && j == 0 {
                re[idx] = 0.0;
                im[idx] = 0.0;
                continue;
            }
            let denom = kx * kx + ky * ky;
            if denom == 0.0 {
                // Should not occur outside (0,0) for power-of-two grids, but
                // belt-and-braces: zero it.
                re[idx] = 0.0;
                im[idx] = 0.0;
            } else {
                let inv = 1.0 / denom;
                re[idx] *= inv;
                im[idx] *= inv;
            }
        }
    }
    // Zero the Nyquist rows / columns to guarantee a real result (their phase
    // is ambiguous under the forward DFT used).
    if nx % 2 == 0 {
        for j in 0..ny {
            let idx = half_x * ny + j;
            re[idx] = 0.0;
            im[idx] = 0.0;
        }
    }
    if ny % 2 == 0 {
        for i in 0..nx {
            let idx = i * ny + half_y;
            re[idx] = 0.0;
            im[idx] = 0.0;
        }
    }
    // Inverse FFT
    fft2_in_place(&mut re, &mut im, nx, ny, 1.0)?;
    let inv_n = 1.0 / n as f64;
    for v in &mut re {
        *v *= inv_n;
    }
    Ok(re)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PI: f64 = std::f64::consts::PI;

    fn grid_xy(cfg: &Fourier2dConfig) -> (Vec<f64>, Vec<f64>) {
        let dx = cfg.lx / cfg.nx as f64;
        let dy = cfg.ly / cfg.ny as f64;
        let xs: Vec<f64> = (0..cfg.nx).map(|i| i as f64 * dx).collect();
        let ys: Vec<f64> = (0..cfg.ny).map(|j| j as f64 * dy).collect();
        (xs, ys)
    }

    /// Build a single-mode sin·sin rhs and its expected solution.
    fn sine_mode(cfg: &Fourier2dConfig, kx_m: i32, ky_m: i32) -> (Vec<f64>, f64) {
        let (xs, ys) = grid_xy(cfg);
        let mut f = vec![0.0_f64; cfg.nx * cfg.ny];
        for i in 0..cfg.nx {
            for j in 0..cfg.ny {
                f[i * cfg.ny + j] = (2.0 * PI * kx_m as f64 * xs[i] / cfg.lx).sin()
                    * (2.0 * PI * ky_m as f64 * ys[j] / cfg.ly).sin();
            }
        }
        let denom =
            (2.0 * PI * kx_m as f64 / cfg.lx).powi(2) + (2.0 * PI * ky_m as f64 / cfg.ly).powi(2);
        (f, denom)
    }

    #[test]
    fn zero_rhs_zero_solution() {
        let cfg = Fourier2dConfig {
            nx: 8,
            ny: 8,
            lx: 1.0,
            ly: 1.0,
        };
        let f = vec![0.0_f64; cfg.nx * cfg.ny];
        let u = solve_poisson_2d_fft(&f, &cfg).expect("ok");
        for v in &u {
            assert!(v.abs() < 1.0e-14);
        }
    }

    #[test]
    fn single_mode_rhs_recovered() {
        // f(x,y) = sin(2π·kx x / Lx) · sin(2π·ky y / Ly)
        // ⇒ u(x,y) = f / ((2π kx / Lx)^2 + (2π ky / Ly)^2)
        let cfg = Fourier2dConfig {
            nx: 16,
            ny: 16,
            lx: 1.0,
            ly: 1.0,
        };
        let (f, denom) = sine_mode(&cfg, 2, 3);
        let u = solve_poisson_2d_fft(&f, &cfg).expect("ok");
        for k in 0..u.len() {
            let expected = f[k] / denom;
            assert!(
                (u[k] - expected).abs() < 1.0e-10,
                "k={k} got {} expected {expected}",
                u[k]
            );
        }
    }

    #[test]
    fn multiple_modes_superposition() {
        let cfg = Fourier2dConfig {
            nx: 32,
            ny: 32,
            lx: 1.0,
            ly: 1.0,
        };
        let (xs, ys) = grid_xy(&cfg);
        let modes = [(1, 2, 0.5_f64), (3, 1, -0.25_f64), (2, 4, 0.75_f64)];
        let mut f = vec![0.0_f64; cfg.nx * cfg.ny];
        let mut u_expected = vec![0.0_f64; cfg.nx * cfg.ny];
        for &(kx_m, ky_m, amp) in &modes {
            let denom = (2.0 * PI * kx_m as f64 / cfg.lx).powi(2)
                + (2.0 * PI * ky_m as f64 / cfg.ly).powi(2);
            for i in 0..cfg.nx {
                for j in 0..cfg.ny {
                    let v = (2.0 * PI * kx_m as f64 * xs[i] / cfg.lx).sin()
                        * (2.0 * PI * ky_m as f64 * ys[j] / cfg.ly).sin();
                    f[i * cfg.ny + j] += amp * v;
                    u_expected[i * cfg.ny + j] += amp * v / denom;
                }
            }
        }
        let u = solve_poisson_2d_fft(&f, &cfg).expect("ok");
        for i in 0..cfg.nx {
            for j in 0..cfg.ny {
                let e = u_expected[i * cfg.ny + j];
                let g = u[i * cfg.ny + j];
                assert!((g - e).abs() < 1.0e-10, "at ({i},{j}) expected={e} got={g}");
            }
        }
    }

    #[test]
    fn constant_rhs_zero_mean_solution() {
        // Constant rhs: û(0,0) gets zeroed by gauge fix, all other modes are
        // also zero ⇒ output is identically zero.
        let cfg = Fourier2dConfig {
            nx: 8,
            ny: 8,
            lx: 1.0,
            ly: 1.0,
        };
        let f = vec![2.5_f64; cfg.nx * cfg.ny];
        let u = solve_poisson_2d_fft(&f, &cfg).expect("ok");
        for v in &u {
            assert!(v.abs() < 1.0e-12);
        }
        let mean = u.iter().sum::<f64>() / u.len() as f64;
        assert!(mean.abs() < 1.0e-14);
    }

    #[test]
    fn symmetric_rhs_symmetric_u() {
        // f(x,y) = cos(2π x)·cos(2π y) is symmetric under (i,j) → (nx−i, ny−j)
        // — solution must be symmetric too.
        let cfg = Fourier2dConfig {
            nx: 16,
            ny: 16,
            lx: 1.0,
            ly: 1.0,
        };
        let (xs, ys) = grid_xy(&cfg);
        let mut f = vec![0.0_f64; cfg.nx * cfg.ny];
        for i in 0..cfg.nx {
            for j in 0..cfg.ny {
                f[i * cfg.ny + j] =
                    (2.0 * PI * xs[i] / cfg.lx).cos() * (2.0 * PI * ys[j] / cfg.ly).cos();
            }
        }
        let u = solve_poisson_2d_fft(&f, &cfg).expect("ok");
        // Symmetry: u(i,j) == u((nx-i) mod nx, (ny-j) mod ny).
        for i in 0..cfg.nx {
            for j in 0..cfg.ny {
                let i2 = (cfg.nx - i) % cfg.nx;
                let j2 = (cfg.ny - j) % cfg.ny;
                assert!(
                    (u[i * cfg.ny + j] - u[i2 * cfg.ny + j2]).abs() < 1.0e-10,
                    "asymmetry at ({i},{j}) vs ({i2},{j2})"
                );
            }
        }
    }

    #[test]
    fn non_power_of_two_rejected() {
        let cfg = Fourier2dConfig {
            nx: 10,
            ny: 16,
            lx: 1.0,
            ly: 1.0,
        };
        let f = vec![0.0_f64; 10 * 16];
        assert!(matches!(
            solve_poisson_2d_fft(&f, &cfg),
            Err(PdeError::InvalidGrid(_))
        ));
        let cfg = Fourier2dConfig {
            nx: 16,
            ny: 12,
            lx: 1.0,
            ly: 1.0,
        };
        let f = vec![0.0_f64; 16 * 12];
        assert!(matches!(
            solve_poisson_2d_fft(&f, &cfg),
            Err(PdeError::InvalidGrid(_))
        ));
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let cfg = Fourier2dConfig {
            nx: 16,
            ny: 16,
            lx: 1.0,
            ly: 1.0,
        };
        let f = vec![0.0_f64; 8 * 16];
        let res = solve_poisson_2d_fft(&f, &cfg);
        assert!(matches!(res, Err(PdeError::ShapeMismatch { .. })));
    }

    #[test]
    fn larger_problem_convergence() {
        let cfg = Fourier2dConfig {
            nx: 64,
            ny: 64,
            lx: 2.0,
            ly: 2.0,
        };
        let (f, denom) = sine_mode(&cfg, 4, 5);
        let u = solve_poisson_2d_fft(&f, &cfg).expect("ok");
        let max_err = u
            .iter()
            .zip(f.iter())
            .map(|(&g, &fk)| (g - fk / denom).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1.0e-8, "max err {max_err} > 1e-8");
    }

    #[test]
    fn asymmetric_domain_correct() {
        let cfg = Fourier2dConfig {
            nx: 32,
            ny: 16,
            lx: 4.0,
            ly: 1.0,
        };
        let (f, denom) = sine_mode(&cfg, 1, 2);
        let u = solve_poisson_2d_fft(&f, &cfg).expect("ok");
        for k in 0..u.len() {
            let expected = f[k] / denom;
            assert!(
                (u[k] - expected).abs() < 1.0e-10,
                "k={k} got {} expected {expected}",
                u[k]
            );
        }
    }

    #[test]
    fn deterministic_resolve() {
        let cfg = Fourier2dConfig {
            nx: 16,
            ny: 16,
            lx: 1.0,
            ly: 1.0,
        };
        let (xs, ys) = grid_xy(&cfg);
        let mut f = vec![0.0_f64; cfg.nx * cfg.ny];
        for i in 0..cfg.nx {
            for j in 0..cfg.ny {
                f[i * cfg.ny + j] =
                    (2.0 * PI * xs[i] / cfg.lx).sin() * (4.0 * PI * ys[j] / cfg.ly).sin();
            }
        }
        let u1 = solve_poisson_2d_fft(&f, &cfg).expect("ok");
        let u2 = solve_poisson_2d_fft(&f, &cfg).expect("ok");
        assert_eq!(u1.len(), u2.len());
        for k in 0..u1.len() {
            assert_eq!(u1[k].to_bits(), u2[k].to_bits(), "non-deterministic at {k}");
        }
    }

    #[test]
    fn zero_mean_property_preserved() {
        // Take arbitrary smooth rhs; output mean should be zero (gauge fix).
        let cfg = Fourier2dConfig {
            nx: 32,
            ny: 32,
            lx: 1.0,
            ly: 1.0,
        };
        let (xs, ys) = grid_xy(&cfg);
        let mut f = vec![0.0_f64; cfg.nx * cfg.ny];
        for i in 0..cfg.nx {
            for j in 0..cfg.ny {
                let x = xs[i];
                let y = ys[j];
                // mixed-mode rhs with non-zero mean removed:
                let v = (2.0 * PI * x).sin() * (2.0 * PI * y).sin()
                    + 0.5 * (4.0 * PI * x).cos() * (2.0 * PI * y).cos();
                f[i * cfg.ny + j] = v;
            }
        }
        // strip the mean of f so the equation is consistent
        let mean_f: f64 = f.iter().sum::<f64>() / f.len() as f64;
        for v in &mut f {
            *v -= mean_f;
        }
        let u = solve_poisson_2d_fft(&f, &cfg).expect("ok");
        let mean_u: f64 = u.iter().sum::<f64>() / u.len() as f64;
        assert!(mean_u.abs() < 1.0e-12, "mean of u = {mean_u}");
    }

    #[test]
    fn invalid_lengths_rejected() {
        let f = vec![0.0_f64; 8 * 8];
        // lx non-positive ⇒ InvalidParameter
        let cfg = Fourier2dConfig {
            nx: 8,
            ny: 8,
            lx: 0.0,
            ly: 1.0,
        };
        assert!(matches!(
            solve_poisson_2d_fft(&f, &cfg),
            Err(PdeError::InvalidParameter { .. })
        ));
        // ly negative ⇒ InvalidParameter
        let cfg = Fourier2dConfig {
            nx: 8,
            ny: 8,
            lx: 1.0,
            ly: -1.0,
        };
        assert!(matches!(
            solve_poisson_2d_fft(&f, &cfg),
            Err(PdeError::InvalidParameter { .. })
        ));
        // nx = 0 ⇒ InvalidGrid (below min and not power of two)
        let cfg = Fourier2dConfig {
            nx: 0,
            ny: 8,
            lx: 1.0,
            ly: 1.0,
        };
        let small = vec![0.0_f64; 1];
        assert!(matches!(
            solve_poisson_2d_fft(&small, &cfg),
            Err(PdeError::InvalidGrid(_))
        ));
        // nx not a power of two ⇒ InvalidGrid
        let cfg = Fourier2dConfig {
            nx: 6,
            ny: 8,
            lx: 1.0,
            ly: 1.0,
        };
        let f6 = vec![0.0_f64; 6 * 8];
        assert!(matches!(
            solve_poisson_2d_fft(&f6, &cfg),
            Err(PdeError::InvalidGrid(_))
        ));
    }

    #[test]
    fn default_config_round_trips() {
        let cfg = Fourier2dConfig::default();
        assert_eq!(cfg.nx, 32);
        assert_eq!(cfg.ny, 32);
        assert!(cfg.lx > 0.0);
        assert!(cfg.ly > 0.0);
    }
}
