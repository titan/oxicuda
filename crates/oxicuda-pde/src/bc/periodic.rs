//! Periodic boundary conditions for finite-difference stencils.
//!
//! A periodic domain identifies the two ends of each axis: node `−1` wraps to
//! node `n − 1` and node `n` wraps to node `0`. This module provides the index
//! arithmetic and the wrapped second-order central-difference operators used by
//! explicit FDM advection / diffusion schemes on a torus, distinct from the
//! FFT-based pseudo-spectral periodic path in [`crate::spectral`].
//!
//! On a periodic mesh the *last* grid point is conventionally the duplicate of
//! the first, so the independent degrees of freedom are the `n − 1` nodes
//! `0..n−1`; the helpers here operate on that convention (a field of length `n`
//! whose endpoints satisfy `u[n−1] == u[0]`), as well as on a compact field of
//! length `n` treated cyclically.

use crate::error::{PdeError, PdeResult};

/// Wrap a signed neighbour offset into `[0, n)` on a periodic axis of length `n`.
///
/// `n` must be non-zero. Handles offsets of any magnitude via Euclidean
/// remainder, so `wrap_index(-1, n) == n - 1` and `wrap_index(n, n) == 0`.
#[inline]
pub fn wrap_index(idx: isize, n: usize) -> usize {
    debug_assert!(n > 0, "wrap_index requires n > 0");
    let nn = n as isize;
    (((idx % nn) + nn) % nn) as usize
}

/// Periodic second-order central first derivative `u_x` on a uniform mesh.
///
/// `u` is treated as a cyclic field of `n` independent nodes with spacing `h`.
/// Returns a vector of length `n` with `u_x[i] = (u[i+1] − u[i−1]) / (2h)` using
/// wrapped neighbours.
pub fn periodic_first_derivative(u: &[f64], h: f64) -> PdeResult<Vec<f64>> {
    let n = u.len();
    if n < 3 {
        return Err(PdeError::InvalidGrid(format!(
            "periodic first derivative requires n >= 3, got {n}"
        )));
    }
    if h <= 0.0 {
        return Err(PdeError::InvalidParameter {
            name: "h".into(),
            reason: format!("must be > 0, got {h}"),
        });
    }
    let inv_2h = 1.0 / (2.0 * h);
    let mut out = vec![0.0; n];
    for (i, out_i) in out.iter_mut().enumerate() {
        let ip = wrap_index(i as isize + 1, n);
        let im = wrap_index(i as isize - 1, n);
        *out_i = (u[ip] - u[im]) * inv_2h;
    }
    Ok(out)
}

/// Periodic second-order central second derivative `u_xx` (the 1D Laplacian).
///
/// Returns `u_xx[i] = (u[i+1] − 2u[i] + u[i−1]) / h²` with wrapped neighbours.
pub fn periodic_laplacian_1d(u: &[f64], h: f64) -> PdeResult<Vec<f64>> {
    let n = u.len();
    if n < 3 {
        return Err(PdeError::InvalidGrid(format!(
            "periodic laplacian requires n >= 3, got {n}"
        )));
    }
    if h <= 0.0 {
        return Err(PdeError::InvalidParameter {
            name: "h".into(),
            reason: format!("must be > 0, got {h}"),
        });
    }
    let inv_h2 = 1.0 / (h * h);
    let mut out = vec![0.0; n];
    for (i, out_i) in out.iter_mut().enumerate() {
        let ip = wrap_index(i as isize + 1, n);
        let im = wrap_index(i as isize - 1, n);
        *out_i = (u[ip] - 2.0 * u[i] + u[im]) * inv_h2;
    }
    Ok(out)
}

/// Periodic 5-point Laplacian on a 2D cyclic grid stored row-major (`i·ny + j`).
///
/// Both axes are periodic. Returns a vector of length `nx·ny`.
pub fn periodic_laplacian_2d(
    u: &[f64],
    nx: usize,
    ny: usize,
    hx: f64,
    hy: f64,
) -> PdeResult<Vec<f64>> {
    if nx < 3 || ny < 3 {
        return Err(PdeError::InvalidGrid(format!(
            "periodic 2d laplacian requires nx>=3 ny>=3, got nx={nx} ny={ny}"
        )));
    }
    if u.len() != nx * ny {
        return Err(PdeError::ShapeMismatch {
            expected: vec![nx * ny],
            got: vec![u.len()],
        });
    }
    if hx <= 0.0 || hy <= 0.0 {
        return Err(PdeError::InvalidParameter {
            name: "spacing".into(),
            reason: format!("must be > 0, got hx={hx} hy={hy}"),
        });
    }
    let inv_hx2 = 1.0 / (hx * hx);
    let inv_hy2 = 1.0 / (hy * hy);
    let mut out = vec![0.0; nx * ny];
    for i in 0..nx {
        let ip = wrap_index(i as isize + 1, nx);
        let im = wrap_index(i as isize - 1, nx);
        for j in 0..ny {
            let jp = wrap_index(j as isize + 1, ny);
            let jm = wrap_index(j as isize - 1, ny);
            let c = u[i * ny + j];
            let lap_x = (u[ip * ny + j] - 2.0 * c + u[im * ny + j]) * inv_hx2;
            let lap_y = (u[i * ny + jp] - 2.0 * c + u[i * ny + jm]) * inv_hy2;
            out[i * ny + j] = lap_x + lap_y;
        }
    }
    Ok(out)
}

/// Enforce periodicity on a length-`n` field whose last node duplicates the
/// first: sets `u[n−1] = u[0]` (the duplicate-endpoint convention).
///
/// Returns [`PdeError::InvalidGrid`] for `n < 2`.
pub fn enforce_periodic_endpoint(u: &mut [f64]) -> PdeResult<()> {
    let n = u.len();
    if n < 2 {
        return Err(PdeError::InvalidGrid(format!(
            "enforce_periodic_endpoint requires n >= 2, got {n}"
        )));
    }
    u[n - 1] = u[0];
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn wrap_index_basic() {
        assert_eq!(wrap_index(0, 5), 0);
        assert_eq!(wrap_index(4, 5), 4);
        assert_eq!(wrap_index(-1, 5), 4);
        assert_eq!(wrap_index(5, 5), 0);
        assert_eq!(wrap_index(6, 5), 1);
        assert_eq!(wrap_index(-6, 5), 4);
    }

    #[test]
    fn first_derivative_of_constant_is_zero() {
        let u = vec![3.0; 8];
        let d = periodic_first_derivative(&u, 0.1).expect("ok");
        for v in d {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn first_derivative_of_sine_is_cosine() {
        // u = sin(x) on [0, 2π) periodic → u_x = cos(x).
        let n = 64;
        let h = 2.0 * PI / n as f64;
        let u: Vec<f64> = (0..n).map(|i| (i as f64 * h).sin()).collect();
        let d = periodic_first_derivative(&u, h).expect("ok");
        for (i, &di) in d.iter().enumerate() {
            let x = i as f64 * h;
            assert!(
                (di - x.cos()).abs() < 5e-3,
                "i={i} got={di} exp={}",
                x.cos()
            );
        }
    }

    #[test]
    fn laplacian_of_constant_is_zero() {
        let u = vec![-2.5; 10];
        let lap = periodic_laplacian_1d(&u, 0.2).expect("ok");
        for v in lap {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn laplacian_of_sine_is_negative_sine() {
        // u = sin(x), u_xx = −sin(x) on a periodic domain.
        let n = 128;
        let h = 2.0 * PI / n as f64;
        let u: Vec<f64> = (0..n).map(|i| (i as f64 * h).sin()).collect();
        let lap = periodic_laplacian_1d(&u, h).expect("ok");
        for (i, &lap_i) in lap.iter().enumerate() {
            let x = i as f64 * h;
            assert!(
                (lap_i + x.sin()).abs() < 5e-3,
                "i={i} got={lap_i} exp={}",
                -x.sin()
            );
        }
    }

    #[test]
    fn laplacian_wraps_at_boundaries() {
        // Compare endpoint stencil to manual wrapped computation.
        let u = vec![1.0, 2.0, 4.0, 7.0];
        let h = 1.0;
        let lap = periodic_laplacian_1d(&u, h).expect("ok");
        // i=0: neighbours are u[1]=2 and u[3]=7 → 2 - 2*1 + 7 = 7.
        assert!((lap[0] - 7.0).abs() < 1e-12, "lap0={}", lap[0]);
        // i=3: neighbours are u[0]=1 and u[2]=4 → 1 - 2*7 + 4 = -9.
        assert!((lap[3] + 9.0).abs() < 1e-12, "lap3={}", lap[3]);
    }

    #[test]
    fn laplacian_2d_constant_is_zero() {
        let nx = 6;
        let ny = 5;
        let u = vec![1.7; nx * ny];
        let lap = periodic_laplacian_2d(&u, nx, ny, 0.3, 0.4).expect("ok");
        for v in lap {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn laplacian_2d_separable_mode() {
        // u = sin(x)sin(y) on [0,2π)² periodic → Δu = −2 sin(x) sin(y).
        let nx = 48;
        let ny = 48;
        let hx = 2.0 * PI / nx as f64;
        let hy = 2.0 * PI / ny as f64;
        let mut u = vec![0.0; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                u[i * ny + j] = (i as f64 * hx).sin() * (j as f64 * hy).sin();
            }
        }
        let lap = periodic_laplacian_2d(&u, nx, ny, hx, hy).expect("ok");
        for i in 0..nx {
            for j in 0..ny {
                let expected = -2.0 * (i as f64 * hx).sin() * (j as f64 * hy).sin();
                assert!(
                    (lap[i * ny + j] - expected).abs() < 1e-2,
                    "({i},{j}) got={} exp={expected}",
                    lap[i * ny + j]
                );
            }
        }
    }

    #[test]
    fn enforce_endpoint_copies_first_to_last() {
        let mut u = vec![5.0, 1.0, 2.0, 9.0];
        enforce_periodic_endpoint(&mut u).expect("ok");
        assert_eq!(u[3], 5.0);
    }

    #[test]
    fn first_derivative_too_short_errors() {
        let u = vec![1.0, 2.0];
        assert!(periodic_first_derivative(&u, 0.1).is_err());
    }

    #[test]
    fn laplacian_invalid_spacing_errors() {
        let u = vec![1.0, 2.0, 3.0];
        assert!(periodic_laplacian_1d(&u, 0.0).is_err());
    }

    #[test]
    fn laplacian_2d_shape_mismatch_errors() {
        let u = vec![0.0; 10];
        assert!(periodic_laplacian_2d(&u, 4, 4, 0.1, 0.1).is_err());
    }

    #[test]
    fn enforce_endpoint_too_short_errors() {
        let mut u = vec![1.0];
        assert!(enforce_periodic_endpoint(&mut u).is_err());
    }
}
