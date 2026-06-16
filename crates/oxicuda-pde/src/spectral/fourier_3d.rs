//! 3D periodic Poisson solver via the FFT.
//!
//! Solves `−Δu = f` on the box `[0, Lx] × [0, Ly] × [0, Lz]` with periodic
//! boundary conditions on all three axes. The discrete grid is `nx × ny × nz`
//! with **power-of-two** dimensions (the radix-2 Cooley–Tukey FFT reused from
//! [`crate::spectral::fourier_2d`] requires `nx = 2^p`, `ny = 2^q`, `nz = 2^r`).
//!
//! # Method
//!
//! Expanding `f` and `u` in a Fourier series on the periodic grid, the equation
//! is diagonal in spectral space,
//!
//! ```text
//! ( kx² + ky² + kz² ) û(kx, ky, kz) = f̂(kx, ky, kz)
//! ```
//!
//! so
//!
//! ```text
//! û(kx, ky, kz) = f̂ / ( kx² + ky² + kz² )    for (kx, ky, kz) ≠ (0, 0, 0)
//! û(0, 0, 0)    = 0                            (gauge fix; mean of u is zero)
//! ```
//!
//! For a grid of `n` samples on `[0, L]` the signed wave numbers are
//!
//! ```text
//! k_m = 2π m / L      for m = 0, 1, …, n/2,  −n/2+1, …, −1.
//! ```
//!
//! # Implementation
//!
//! A 3-D FFT is assembled from a sequence of 1-D radix-2 transforms applied
//! along each axis in turn (separability of the multidimensional DFT). The
//! same in-place kernel `crate::spectral::fourier_2d::fft_radix2` is reused
//! for every line so no FFT code is duplicated. Arrays are stored row-major in
//! C order, `idx(i, j, k) = (i · ny + j) · nz + k`.
//!
//! # References
//!
//! * Trefethen, *Spectral Methods in MATLAB*, SIAM 2000, ch. 3.
//! * Cooley & Tukey, *An algorithm for the machine calculation of complex
//!   Fourier series*, Math. Comp. 19 (1965), 297–301.

use crate::error::{PdeError, PdeResult};
use crate::spectral::fourier_2d::{fft_radix2, is_power_of_two};

/// Configuration for the 3D FFT-based periodic Poisson solver.
#[derive(Debug, Clone, Copy)]
pub struct Fourier3dConfig {
    /// Number of grid points along x (power of two, ≥ 4).
    pub nx: usize,
    /// Number of grid points along y (power of two, ≥ 4).
    pub ny: usize,
    /// Number of grid points along z (power of two, ≥ 4).
    pub nz: usize,
    /// Domain length along x (must be positive and finite).
    pub lx: f64,
    /// Domain length along y (must be positive and finite).
    pub ly: f64,
    /// Domain length along z (must be positive and finite).
    pub lz: f64,
}

impl Default for Fourier3dConfig {
    fn default() -> Self {
        Self {
            nx: 16,
            ny: 16,
            nz: 16,
            lx: 1.0,
            ly: 1.0,
            lz: 1.0,
        }
    }
}

impl Fourier3dConfig {
    /// Total number of grid points `nx · ny · nz`.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Whether the configured grid is empty (always `false` for a valid config).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nx == 0 || self.ny == 0 || self.nz == 0
    }

    /// Row-major C-order flat index `(i · ny + j) · nz + k`.
    #[inline]
    #[must_use]
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.ny + j) * self.nz + k
    }
}

fn validate_config(cfg: &Fourier3dConfig) -> PdeResult<()> {
    for (name, n) in [("nx", cfg.nx), ("ny", cfg.ny), ("nz", cfg.nz)] {
        if n < 4 || !is_power_of_two(n) {
            return Err(PdeError::InvalidGrid(format!(
                "fourier_3d requires power-of-two {name}>=4 (got {name}={n})"
            )));
        }
    }
    for (name, l) in [("lx", cfg.lx), ("ly", cfg.ly), ("lz", cfg.lz)] {
        if l <= 0.0 || !l.is_finite() {
            return Err(PdeError::InvalidParameter {
                name: name.into(),
                reason: "must be positive and finite".into(),
            });
        }
    }
    Ok(())
}

/// Apply a 1-D FFT to every line of constant other-axis indices.
///
/// `bases` lists the flat offset of the first element of each line and `stride`
/// is the spacing between consecutive samples along the transform axis. The
/// `len` samples of each line are gathered into a scratch buffer, transformed
/// in place via [`fft_radix2`], and scattered back.
fn fft_lines(
    re: &mut [f64],
    im: &mut [f64],
    bases: &[usize],
    len: usize,
    stride: usize,
    sign: f64,
) -> PdeResult<()> {
    let mut line_re = vec![0.0_f64; len];
    let mut line_im = vec![0.0_f64; len];
    for &base in bases {
        for t in 0..len {
            let idx = base + t * stride;
            line_re[t] = re[idx];
            line_im[t] = im[idx];
        }
        fft_radix2(&mut line_re, &mut line_im, sign)?;
        for t in 0..len {
            let idx = base + t * stride;
            re[idx] = line_re[t];
            im[idx] = line_im[t];
        }
    }
    Ok(())
}

/// Run a full 3-D FFT in place over a row-major `nx × ny × nz` complex array.
///
/// `sign = -1.0` is the forward transform; `sign = +1.0` is the unnormalised
/// inverse (the caller divides by `nx · ny · nz`).
fn fft3_in_place(
    re: &mut [f64],
    im: &mut [f64],
    nx: usize,
    ny: usize,
    nz: usize,
    sign: f64,
) -> PdeResult<()> {
    let n = nx * ny * nz;
    if re.len() != n || im.len() != n {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n],
            got: vec![re.len()],
        });
    }
    // Axis k (last, contiguous, stride 1): one line per (i, j).
    let mut bases_k = Vec::with_capacity(nx * ny);
    for i in 0..nx {
        for j in 0..ny {
            bases_k.push((i * ny + j) * nz);
        }
    }
    fft_lines(re, im, &bases_k, nz, 1, sign)?;

    // Axis j (middle, stride nz): one line per (i, k).
    let mut bases_j = Vec::with_capacity(nx * nz);
    for i in 0..nx {
        for k in 0..nz {
            bases_j.push(i * ny * nz + k);
        }
    }
    fft_lines(re, im, &bases_j, ny, nz, sign)?;

    // Axis i (first, stride ny·nz): one line per (j, k).
    let mut bases_i = Vec::with_capacity(ny * nz);
    for j in 0..ny {
        for k in 0..nz {
            bases_i.push(j * nz + k);
        }
    }
    fft_lines(re, im, &bases_i, nx, ny * nz, sign)?;
    Ok(())
}

/// Signed wave number `2π m / L` for spectral index `p` on an `n`-point grid.
#[inline]
fn wave_number(p: usize, n: usize, length: f64) -> f64 {
    let half = n / 2;
    let m = if p <= half {
        p as i64
    } else {
        p as i64 - n as i64
    };
    std::f64::consts::TAU * m as f64 / length
}

/// Zero the Nyquist planes (for even `n`) so that the inverse transform is
/// guaranteed real — their phase is ambiguous under the real-input DFT.
fn zero_nyquist_planes(re: &mut [f64], im: &mut [f64], cfg: &Fourier3dConfig) {
    let (nx, ny, nz) = (cfg.nx, cfg.ny, cfg.nz);
    if nx % 2 == 0 {
        let i = nx / 2;
        for j in 0..ny {
            for k in 0..nz {
                let idx = cfg.index(i, j, k);
                re[idx] = 0.0;
                im[idx] = 0.0;
            }
        }
    }
    if ny % 2 == 0 {
        let j = ny / 2;
        for i in 0..nx {
            for k in 0..nz {
                let idx = cfg.index(i, j, k);
                re[idx] = 0.0;
                im[idx] = 0.0;
            }
        }
    }
    if nz % 2 == 0 {
        let k = nz / 2;
        for i in 0..nx {
            for j in 0..ny {
                let idx = cfg.index(i, j, k);
                re[idx] = 0.0;
                im[idx] = 0.0;
            }
        }
    }
}

/// Solve the 3D periodic Poisson equation `−Δu = f` on the box
/// `[0, Lx] × [0, Ly] × [0, Lz]`.
///
/// `f` is a row-major `nx × ny × nz` array (C order); the returned vector has
/// the same layout and contains the zero-mean solution `u`.
///
/// # Errors
///
/// * [`PdeError::InvalidGrid`] if any of `nx`, `ny`, `nz` is not a power of two
///   or is smaller than 4.
/// * [`PdeError::InvalidParameter`] if any length is non-positive or non-finite.
/// * [`PdeError::ShapeMismatch`] if `f.len() != nx · ny · nz`.
pub fn solve_poisson_3d_fft(f: &[f64], cfg: &Fourier3dConfig) -> PdeResult<Vec<f64>> {
    validate_config(cfg)?;
    let n = cfg.len();
    if f.len() != n {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n],
            got: vec![f.len()],
        });
    }
    let (nx, ny, nz) = (cfg.nx, cfg.ny, cfg.nz);
    let mut re = f.to_vec();
    let mut im = vec![0.0_f64; n];
    // Forward transform.
    fft3_in_place(&mut re, &mut im, nx, ny, nz, -1.0)?;
    // Divide each mode by |k|²; gauge-fix the (0,0,0) mode to zero.
    for i in 0..nx {
        let kx = wave_number(i, nx, cfg.lx);
        for j in 0..ny {
            let ky = wave_number(j, ny, cfg.ly);
            for k in 0..nz {
                let kz = wave_number(k, nz, cfg.lz);
                let idx = cfg.index(i, j, k);
                let denom = kx * kx + ky * ky + kz * kz;
                if denom == 0.0 {
                    re[idx] = 0.0;
                    im[idx] = 0.0;
                } else {
                    let inv = 1.0 / denom;
                    re[idx] *= inv;
                    im[idx] *= inv;
                }
            }
        }
    }
    zero_nyquist_planes(&mut re, &mut im, cfg);
    // Inverse transform and normalise.
    fft3_in_place(&mut re, &mut im, nx, ny, nz, 1.0)?;
    let inv_n = 1.0 / n as f64;
    for v in &mut re {
        *v *= inv_n;
    }
    Ok(re)
}

/// Apply the negative Laplacian `−Δu` spectrally to a real periodic field `u`.
///
/// Each Fourier mode is multiplied by `|k|² = kx² + ky² + kz²`; the
/// `(0,0,0)` mode (the mean) and the Nyquist planes are zeroed so the result is
/// real with zero mean. This is the exact inverse of [`solve_poisson_3d_fft`]
/// on the subspace of zero-mean, Nyquist-free fields, which makes it convenient
/// for round-trip verification.
///
/// # Errors
///
/// Same validation as [`solve_poisson_3d_fft`].
pub fn neg_laplacian_3d_spectral(u: &[f64], cfg: &Fourier3dConfig) -> PdeResult<Vec<f64>> {
    validate_config(cfg)?;
    let n = cfg.len();
    if u.len() != n {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n],
            got: vec![u.len()],
        });
    }
    let (nx, ny, nz) = (cfg.nx, cfg.ny, cfg.nz);
    let mut re = u.to_vec();
    let mut im = vec![0.0_f64; n];
    fft3_in_place(&mut re, &mut im, nx, ny, nz, -1.0)?;
    for i in 0..nx {
        let kx = wave_number(i, nx, cfg.lx);
        for j in 0..ny {
            let ky = wave_number(j, ny, cfg.ly);
            for k in 0..nz {
                let kz = wave_number(k, nz, cfg.lz);
                let idx = cfg.index(i, j, k);
                let factor = kx * kx + ky * ky + kz * kz;
                re[idx] *= factor;
                im[idx] *= factor;
            }
        }
    }
    zero_nyquist_planes(&mut re, &mut im, cfg);
    fft3_in_place(&mut re, &mut im, nx, ny, nz, 1.0)?;
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

    fn grid_coords(cfg: &Fourier3dConfig) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let dx = cfg.lx / cfg.nx as f64;
        let dy = cfg.ly / cfg.ny as f64;
        let dz = cfg.lz / cfg.nz as f64;
        (
            (0..cfg.nx).map(|i| i as f64 * dx).collect(),
            (0..cfg.ny).map(|j| j as f64 * dy).collect(),
            (0..cfg.nz).map(|k| k as f64 * dz).collect(),
        )
    }

    /// Build a separable single-mode rhs `sin·sin·sin` with the given integer
    /// wave numbers, returning `(f, λ)` where `λ = |k|²`.
    fn sine_mode(cfg: &Fourier3dConfig, mx: i32, my: i32, mz: i32) -> (Vec<f64>, f64) {
        let (xs, ys, zs) = grid_coords(cfg);
        let mut f = vec![0.0_f64; cfg.len()];
        for i in 0..cfg.nx {
            for j in 0..cfg.ny {
                for k in 0..cfg.nz {
                    f[cfg.index(i, j, k)] = (2.0 * PI * mx as f64 * xs[i] / cfg.lx).sin()
                        * (2.0 * PI * my as f64 * ys[j] / cfg.ly).sin()
                        * (2.0 * PI * mz as f64 * zs[k] / cfg.lz).sin();
                }
            }
        }
        let lambda = (2.0 * PI * mx as f64 / cfg.lx).powi(2)
            + (2.0 * PI * my as f64 / cfg.ly).powi(2)
            + (2.0 * PI * mz as f64 / cfg.lz).powi(2);
        (f, lambda)
    }

    #[test]
    fn single_mode_recovered() {
        // f = sin(k·x) ⇒ u = f / |k|², recovered to ~1e-10.
        let cfg = Fourier3dConfig {
            nx: 8,
            ny: 8,
            nz: 8,
            lx: 1.0,
            ly: 1.0,
            lz: 1.0,
        };
        let (f, lambda) = sine_mode(&cfg, 1, 2, 1);
        let u = solve_poisson_3d_fft(&f, &cfg).expect("solve ok");
        for idx in 0..u.len() {
            let expected = f[idx] / lambda;
            assert!(
                (u[idx] - expected).abs() < 1.0e-10,
                "idx={idx} got={} expected={expected}",
                u[idx]
            );
        }
    }

    #[test]
    fn anisotropic_domain_recovered() {
        let cfg = Fourier3dConfig {
            nx: 16,
            ny: 8,
            nz: 8,
            lx: 4.0,
            ly: 2.0,
            lz: 1.0,
        };
        let (f, lambda) = sine_mode(&cfg, 2, 1, 3);
        let u = solve_poisson_3d_fft(&f, &cfg).expect("solve ok");
        let max_err = u
            .iter()
            .zip(f.iter())
            .map(|(&g, &fk)| (g - fk / lambda).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1.0e-10, "max err {max_err}");
    }

    #[test]
    fn multi_mode_superposition() {
        let cfg = Fourier3dConfig::default();
        let (xs, ys, zs) = grid_coords(&cfg);
        let modes = [
            (1, 1, 2, 0.5_f64),
            (2, 3, 1, -0.75_f64),
            (3, 1, 1, 0.25_f64),
        ];
        let mut f = vec![0.0_f64; cfg.len()];
        let mut u_exact = vec![0.0_f64; cfg.len()];
        for &(mx, my, mz, amp) in &modes {
            let lambda = (2.0 * PI * mx as f64 / cfg.lx).powi(2)
                + (2.0 * PI * my as f64 / cfg.ly).powi(2)
                + (2.0 * PI * mz as f64 / cfg.lz).powi(2);
            for (i, &x) in xs.iter().enumerate() {
                for (j, &y) in ys.iter().enumerate() {
                    for (k, &z) in zs.iter().enumerate() {
                        let v = (2.0 * PI * mx as f64 * x / cfg.lx).sin()
                            * (2.0 * PI * my as f64 * y / cfg.ly).sin()
                            * (2.0 * PI * mz as f64 * z / cfg.lz).sin();
                        let idx = cfg.index(i, j, k);
                        f[idx] += amp * v;
                        u_exact[idx] += amp * v / lambda;
                    }
                }
            }
        }
        let u = solve_poisson_3d_fft(&f, &cfg).expect("solve ok");
        for idx in 0..u.len() {
            assert!(
                (u[idx] - u_exact[idx]).abs() < 1.0e-10,
                "idx={idx} got={} expected={}",
                u[idx],
                u_exact[idx]
            );
        }
    }

    #[test]
    fn parseval_energy_identity() {
        // For a single mode u = f/λ the discrete energy identities hold:
        //   Σ u·f = (1/λ) Σ f·f   and   Σ u·f = λ Σ u·u  ( = Σ|∇u|² , the
        // Parseval/Dirichlet energy ), all strictly positive.
        let cfg = Fourier3dConfig {
            nx: 8,
            ny: 8,
            nz: 8,
            lx: 2.0,
            ly: 1.0,
            lz: 3.0,
        };
        let (f, lambda) = sine_mode(&cfg, 1, 2, 2);
        let u = solve_poisson_3d_fft(&f, &cfg).expect("solve ok");
        let uf: f64 = u.iter().zip(&f).map(|(a, b)| a * b).sum();
        let ff: f64 = f.iter().map(|a| a * a).sum();
        let uu: f64 = u.iter().map(|a| a * a).sum();
        assert!(uf > 0.0, "energy Σ u·f must be positive, got {uf}");
        assert!(
            (uf - ff / lambda).abs() < 1.0e-9 * ff.max(1.0),
            "Σ u·f {uf} != Σ f·f / λ {}",
            ff / lambda
        );
        assert!(
            (uf - lambda * uu).abs() < 1.0e-9 * uf.max(1.0),
            "Σ u·f {uf} != λ Σ u·u {}",
            lambda * uu
        );
    }

    #[test]
    fn round_trip_neg_laplacian() {
        // Solve −Δu = f, then apply −Δ to u: recovers the zero-mean f.
        let cfg = Fourier3dConfig {
            nx: 8,
            ny: 8,
            nz: 16,
            lx: 1.0,
            ly: 1.0,
            lz: 1.0,
        };
        let (xs, ys, zs) = grid_coords(&cfg);
        let mut f = vec![0.0_f64; cfg.len()];
        for i in 0..cfg.nx {
            for j in 0..cfg.ny {
                for k in 0..cfg.nz {
                    let v = (2.0 * PI * xs[i]).sin()
                        * (2.0 * PI * ys[j]).cos()
                        * (2.0 * PI * zs[k]).sin()
                        + 0.5 * (4.0 * PI * xs[i]).cos() * (2.0 * PI * ys[j]).sin();
                    f[cfg.index(i, j, k)] = v;
                }
            }
        }
        // Strip the mean for compatibility.
        let mean = f.iter().sum::<f64>() / f.len() as f64;
        for v in &mut f {
            *v -= mean;
        }
        let u = solve_poisson_3d_fft(&f, &cfg).expect("solve ok");
        let f_back = neg_laplacian_3d_spectral(&u, &cfg).expect("neg lap ok");
        let max_err = f_back
            .iter()
            .zip(&f)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1.0e-9, "round-trip max err {max_err}");
    }

    #[test]
    fn zero_mean_enforced() {
        let cfg = Fourier3dConfig::default();
        let (xs, ys, zs) = grid_coords(&cfg);
        let mut f = vec![0.0_f64; cfg.len()];
        for i in 0..cfg.nx {
            for j in 0..cfg.ny {
                for k in 0..cfg.nz {
                    f[cfg.index(i, j, k)] = (2.0 * PI * xs[i]).sin() * (2.0 * PI * ys[j]).sin()
                        + 3.7 // deliberate non-zero offset; gauge fix must remove it
                        + (2.0 * PI * zs[k]).cos();
                }
            }
        }
        let u = solve_poisson_3d_fft(&f, &cfg).expect("solve ok");
        let mean = u.iter().sum::<f64>() / u.len() as f64;
        assert!(mean.abs() < 1.0e-12, "mean of u = {mean}");
        assert!(u.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn constant_rhs_zero_solution() {
        let cfg = Fourier3dConfig {
            nx: 8,
            ny: 8,
            nz: 8,
            lx: 1.0,
            ly: 1.0,
            lz: 1.0,
        };
        let f = vec![2.5_f64; cfg.len()];
        let u = solve_poisson_3d_fft(&f, &cfg).expect("solve ok");
        for v in &u {
            assert!(
                v.abs() < 1.0e-12,
                "constant rhs should give zero u, got {v}"
            );
        }
    }

    #[test]
    fn deterministic_resolve() {
        let cfg = Fourier3dConfig::default();
        let (f, _) = sine_mode(&cfg, 1, 2, 3);
        let u1 = solve_poisson_3d_fft(&f, &cfg).expect("solve ok");
        let u2 = solve_poisson_3d_fft(&f, &cfg).expect("solve ok");
        assert_eq!(u1.len(), u2.len());
        for idx in 0..u1.len() {
            assert_eq!(
                u1[idx].to_bits(),
                u2[idx].to_bits(),
                "nondeterministic at {idx}"
            );
        }
    }

    #[test]
    fn invalid_grid_rejected() {
        let cfg = Fourier3dConfig {
            nx: 6,
            ny: 8,
            nz: 8,
            lx: 1.0,
            ly: 1.0,
            lz: 1.0,
        };
        let f = vec![0.0_f64; 6 * 8 * 8];
        assert!(matches!(
            solve_poisson_3d_fft(&f, &cfg),
            Err(PdeError::InvalidGrid(_))
        ));
    }

    #[test]
    fn invalid_length_rejected() {
        let cfg = Fourier3dConfig {
            nx: 8,
            ny: 8,
            nz: 8,
            lx: 0.0,
            ly: 1.0,
            lz: 1.0,
        };
        let f = vec![0.0_f64; cfg.len()];
        assert!(matches!(
            solve_poisson_3d_fft(&f, &cfg),
            Err(PdeError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let cfg = Fourier3dConfig::default();
        let f = vec![0.0_f64; cfg.len() - 1];
        assert!(matches!(
            solve_poisson_3d_fft(&f, &cfg),
            Err(PdeError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn config_helpers() {
        let cfg = Fourier3dConfig::default();
        assert_eq!(cfg.len(), 16 * 16 * 16);
        assert!(!cfg.is_empty());
        assert_eq!(cfg.index(2, 1, 3), (2 * cfg.ny + 1) * cfg.nz + 3);
    }
}
