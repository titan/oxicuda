//! Cahn–Hilliard phase-field equation, the 4th-order conservative gradient flow
//!
//! ```text
//! ∂c/∂t = M ∇²μ,        μ = f'(c) − ε² ∇²c,        f(c) = ¼ (c² − 1)²
//! ```
//!
//! with the double-well bulk potential `f` (so `f'(c) = c³ − c`). The equation is
//! the `H⁻¹` gradient flow of the Ginzburg–Landau free energy
//!
//! ```text
//! E[c] = ∫ ( f(c) + ½ ε² |∇c|² ) dx
//! ```
//!
//! and has two structural invariants: **mass conservation** (`∫c` is constant,
//! because the right-hand side is a divergence) and **energy dissipation**
//! (`E` is non-increasing).
//!
//! # Scheme — stabilised semi-implicit convex splitting (Eyre)
//!
//! On a periodic grid the linear 4th-order term `ε²∇⁴` is treated implicitly and
//! the nonlinear term `∇²f'(c)` explicitly, with a linear stabilisation `S`
//! (Shen–Yang). In Fourier space (`∇² → −κ`, `κ = |k|²`) one step is the *diagonal*
//! update
//!
//! ```text
//! ĉ^{n+1}_k = [ (1 + dt M S κ) ĉ^n_k − dt M κ ĝ^n_k ] / (1 + dt M S κ + dt M ε² κ²),
//! ĝ = DFT(f'(c^n)).
//! ```
//!
//! The `k = 0` mode is untouched (`κ = 0`), so the discrete mass `Σc` is conserved
//! to round-off. With `S ≥ ½ max|f''|` the scheme is unconditionally energy stable.
//!
//! Spatial transforms use a direct `O(N²)` DFT (no external FFT dependency), which is
//! ample for the modest grids these solvers target. Both 1-D ([`CahnHilliard`]) and
//! 2-D ([`CahnHilliard2d`]) variants are provided; the 2-D grid is row-major `i·ny + j`.
//!
//! References: D. Eyre, *Unconditionally gradient stable time marching the
//! Cahn–Hilliard equation* (1998); J. Shen & X. Yang, *Numerical approximations of
//! Allen–Cahn and Cahn–Hilliard equations*, DCDS-A 28 (2010) 1669–1691.

use crate::error::{PdeError, PdeResult};
use std::f64::consts::TAU;

/// Default stabilisation constant `S`. For `f(c) = ¼(c²−1)²`, `f'' = 3c² − 1`, so
/// `½ max|f''| = 2` for `|c| ≲ 1`; `S = 2` keeps the scheme energy stable.
pub const DEFAULT_STABILIZATION: f64 = 2.0;

// ─── Direct DFT primitives (O(N²); sufficient for the small grids used here) ──────

/// Complex 1-D DFT. `sign = −1` is the forward transform, `sign = +1` the inverse
/// (un-normalised — the caller divides by the length).
fn dft_1d(re: &[f64], im: &[f64], sign: f64) -> (Vec<f64>, Vec<f64>) {
    let n = re.len();
    let mut out_re = vec![0.0; n];
    let mut out_im = vec![0.0; n];
    let base = sign * TAU / n as f64;
    for (k, (or, oi)) in out_re.iter_mut().zip(out_im.iter_mut()).enumerate() {
        let bk = base * k as f64;
        let mut sr = 0.0;
        let mut si = 0.0;
        for (j, (&rj, &ij)) in re.iter().zip(im.iter()).enumerate() {
            let (sn, cs) = (bk * j as f64).sin_cos();
            sr += rj * cs - ij * sn;
            si += rj * sn + ij * cs;
        }
        *or = sr;
        *oi = si;
    }
    (out_re, out_im)
}

/// Forward DFT of a real signal.
fn forward_1d(real: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let im = vec![0.0; real.len()];
    dft_1d(real, &im, -1.0)
}

/// Inverse DFT returning the real part (the imaginary residue is round-off).
fn inverse_real_1d(re: &[f64], im: &[f64]) -> Vec<f64> {
    let (out_re, _) = dft_1d(re, im, 1.0);
    let inv_n = 1.0 / re.len() as f64;
    out_re.into_iter().map(|v| v * inv_n).collect()
}

/// Separable complex 2-D DFT on a row-major `nx·ny` grid (`i·ny + j`).
fn dft_2d(re: &[f64], im: &[f64], nx: usize, ny: usize, sign: f64) -> (Vec<f64>, Vec<f64>) {
    let mut rr = re.to_vec();
    let mut ii = im.to_vec();
    // Transform along x (index i, stride ny) for each column j.
    let mut col_re = vec![0.0; nx];
    let mut col_im = vec![0.0; nx];
    for j in 0..ny {
        for i in 0..nx {
            col_re[i] = rr[i * ny + j];
            col_im[i] = ii[i * ny + j];
        }
        let (tr, ti) = dft_1d(&col_re, &col_im, sign);
        for i in 0..nx {
            rr[i * ny + j] = tr[i];
            ii[i * ny + j] = ti[i];
        }
    }
    // Transform along y (contiguous blocks) for each row i.
    for i in 0..nx {
        let off = i * ny;
        let (tr, ti) = dft_1d(&rr[off..off + ny], &ii[off..off + ny], sign);
        rr[off..off + ny].copy_from_slice(&tr);
        ii[off..off + ny].copy_from_slice(&ti);
    }
    (rr, ii)
}

/// Forward 2-D DFT of a real field.
fn forward_2d(real: &[f64], nx: usize, ny: usize) -> (Vec<f64>, Vec<f64>) {
    let im = vec![0.0; real.len()];
    dft_2d(real, &im, nx, ny, -1.0)
}

/// Inverse 2-D DFT returning the real part.
fn inverse_real_2d(re: &[f64], im: &[f64], nx: usize, ny: usize) -> Vec<f64> {
    let (out_re, _) = dft_2d(re, im, nx, ny, 1.0);
    let inv = 1.0 / (nx * ny) as f64;
    out_re.into_iter().map(|v| v * inv).collect()
}

/// Signed wavenumber `k_m = 2π m̃ / L` for DFT index `m` (upper half = negative freqs).
fn wavenumber(m: usize, n: usize, length: f64) -> f64 {
    let half = n / 2;
    let mm = if m <= half {
        m as f64
    } else {
        m as f64 - n as f64
    };
    TAU * mm / length
}

/// `(fac_c, fac_g)` multipliers of the diagonal stabilised semi-implicit update.
#[inline]
fn spectral_factors(kappa: f64, dt: f64, mobility: f64, eps2: f64, stab: f64) -> (f64, f64) {
    let lin = dt * mobility * kappa; // dt · M · κ
    let denom = 1.0 + stab * lin + lin * eps2 * kappa;
    ((1.0 + stab * lin) / denom, lin / denom)
}

/// Double-well bulk potential `f(c) = ¼ (c² − 1)²`.
#[inline]
fn bulk_potential(c: f64) -> f64 {
    let w = c * c - 1.0;
    0.25 * w * w
}

/// Bulk chemical potential `f'(c) = c³ − c`.
#[inline]
fn bulk_derivative(c: f64) -> f64 {
    c * c * c - c
}

fn check_positive(name: &str, value: f64) -> PdeResult<()> {
    if !(value.is_finite() && value > 0.0) {
        return Err(PdeError::InvalidParameter {
            name: name.into(),
            reason: format!("must be finite and > 0, got {value}"),
        });
    }
    Ok(())
}

fn check_dt(dt: f64) -> PdeResult<()> {
    if !(dt.is_finite() && dt > 0.0) {
        return Err(PdeError::InvalidParameter {
            name: "dt".into(),
            reason: format!("time step must be finite and > 0, got {dt}"),
        });
    }
    Ok(())
}

// ─── 1-D solver ──────────────────────────────────────────────────────────────────

/// Spectral stabilised semi-implicit Cahn–Hilliard solver on a 1-D periodic grid.
#[derive(Debug, Clone)]
pub struct CahnHilliard {
    /// Mobility `M > 0`.
    pub mobility: f64,
    /// Interface-width parameter `ε > 0`.
    pub epsilon: f64,
    /// Grid spacing `dx > 0` (domain length `L = n · dx`).
    pub dx: f64,
    /// Number of periodic grid points (`n ≥ 4`).
    pub n: usize,
    /// Linear stabilisation constant `S ≥ 0`.
    pub stabilization: f64,
}

impl CahnHilliard {
    /// Build a 1-D solver with the [`DEFAULT_STABILIZATION`].
    pub fn new(mobility: f64, epsilon: f64, dx: f64, n: usize) -> PdeResult<Self> {
        check_positive("mobility", mobility)?;
        check_positive("epsilon", epsilon)?;
        check_positive("dx", dx)?;
        if n < 4 {
            return Err(PdeError::InvalidGrid(format!(
                "Cahn-Hilliard requires n >= 4, got {n}"
            )));
        }
        Ok(Self {
            mobility,
            epsilon,
            dx,
            n,
            stabilization: DEFAULT_STABILIZATION,
        })
    }

    /// Override the stabilisation constant `S` (must be finite and `≥ 0`).
    pub fn with_stabilization(mut self, stab: f64) -> PdeResult<Self> {
        if !(stab.is_finite() && stab >= 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "stabilization".into(),
                reason: format!("must be finite and >= 0, got {stab}"),
            });
        }
        self.stabilization = stab;
        Ok(self)
    }

    fn check_field(&self, c: &[f64]) -> PdeResult<()> {
        if c.len() != self.n {
            return Err(PdeError::ShapeMismatch {
                expected: vec![self.n],
                got: vec![c.len()],
            });
        }
        Ok(())
    }

    /// Advance `c` by one stabilised semi-implicit step of size `dt`, in place.
    pub fn step(&self, c: &mut [f64], dt: f64) -> PdeResult<()> {
        self.check_field(c)?;
        check_dt(dt)?;
        let n = self.n;
        let length = n as f64 * self.dx;
        let eps2 = self.epsilon * self.epsilon;
        let g: Vec<f64> = c.iter().map(|&v| bulk_derivative(v)).collect();
        let (cr, ci) = forward_1d(c);
        let (gr, gi) = forward_1d(&g);
        let mut or = vec![0.0; n];
        let mut oi = vec![0.0; n];
        for m in 0..n {
            let k = wavenumber(m, n, length);
            let kappa = k * k;
            let (fc, fg) = spectral_factors(kappa, dt, self.mobility, eps2, self.stabilization);
            or[m] = fc * cr[m] - fg * gr[m];
            oi[m] = fc * ci[m] - fg * gi[m];
        }
        let c_new = inverse_real_1d(&or, &oi);
        c.copy_from_slice(&c_new);
        Ok(())
    }

    /// Integrate `n_steps` steps from `c0`, returning the final field.
    pub fn solve(&self, c0: &[f64], dt: f64, n_steps: usize) -> PdeResult<Vec<f64>> {
        self.check_field(c0)?;
        let mut c = c0.to_vec();
        for _ in 0..n_steps {
            self.step(&mut c, dt)?;
        }
        if c.iter().any(|v| !v.is_finite()) {
            return Err(PdeError::NumericalInstability(
                "Cahn-Hilliard solution diverged to non-finite values".into(),
            ));
        }
        Ok(c)
    }

    /// Total discrete mass `∫c ≈ dx Σ c_i` (conserved by the scheme).
    pub fn total_mass(&self, c: &[f64]) -> PdeResult<f64> {
        self.check_field(c)?;
        Ok(c.iter().sum::<f64>() * self.dx)
    }

    /// Ginzburg–Landau free energy `E = ∫(f(c) + ½ε²|∇c|²)`, gradient term via
    /// the spectral (Parseval) representation for consistency with the scheme.
    pub fn free_energy(&self, c: &[f64]) -> PdeResult<f64> {
        self.check_field(c)?;
        let n = self.n;
        let dx = self.dx;
        let length = n as f64 * dx;
        let bulk: f64 = c.iter().map(|&v| bulk_potential(v)).sum::<f64>() * dx;
        let (cr, ci) = forward_1d(c);
        let mut grad = 0.0;
        for m in 0..n {
            let k = wavenumber(m, n, length);
            grad += k * k * (cr[m] * cr[m] + ci[m] * ci[m]);
        }
        grad *= 0.5 * self.epsilon * self.epsilon * dx / n as f64;
        Ok(bulk + grad)
    }
}

// ─── 2-D solver ──────────────────────────────────────────────────────────────────

/// Spectral stabilised semi-implicit Cahn–Hilliard solver on a 2-D periodic grid
/// (row-major `i·ny + j`).
#[derive(Debug, Clone)]
pub struct CahnHilliard2d {
    /// Mobility `M > 0`.
    pub mobility: f64,
    /// Interface-width parameter `ε > 0`.
    pub epsilon: f64,
    /// Grid spacing along x (`dx > 0`).
    pub dx: f64,
    /// Grid spacing along y (`dy > 0`).
    pub dy: f64,
    /// Grid points along x (`nx ≥ 4`).
    pub nx: usize,
    /// Grid points along y (`ny ≥ 4`).
    pub ny: usize,
    /// Linear stabilisation constant `S ≥ 0`.
    pub stabilization: f64,
}

impl CahnHilliard2d {
    /// Build a 2-D solver with the [`DEFAULT_STABILIZATION`].
    pub fn new(
        mobility: f64,
        epsilon: f64,
        dx: f64,
        dy: f64,
        nx: usize,
        ny: usize,
    ) -> PdeResult<Self> {
        check_positive("mobility", mobility)?;
        check_positive("epsilon", epsilon)?;
        check_positive("dx", dx)?;
        check_positive("dy", dy)?;
        if nx < 4 || ny < 4 {
            return Err(PdeError::InvalidGrid(format!(
                "Cahn-Hilliard 2d requires nx,ny >= 4, got nx={nx} ny={ny}"
            )));
        }
        Ok(Self {
            mobility,
            epsilon,
            dx,
            dy,
            nx,
            ny,
            stabilization: DEFAULT_STABILIZATION,
        })
    }

    /// Override the stabilisation constant `S` (must be finite and `≥ 0`).
    pub fn with_stabilization(mut self, stab: f64) -> PdeResult<Self> {
        if !(stab.is_finite() && stab >= 0.0) {
            return Err(PdeError::InvalidParameter {
                name: "stabilization".into(),
                reason: format!("must be finite and >= 0, got {stab}"),
            });
        }
        self.stabilization = stab;
        Ok(self)
    }

    fn n(&self) -> usize {
        self.nx * self.ny
    }

    fn check_field(&self, c: &[f64]) -> PdeResult<()> {
        if c.len() != self.n() {
            return Err(PdeError::ShapeMismatch {
                expected: vec![self.n()],
                got: vec![c.len()],
            });
        }
        Ok(())
    }

    /// Advance `c` by one stabilised semi-implicit step of size `dt`, in place.
    pub fn step(&self, c: &mut [f64], dt: f64) -> PdeResult<()> {
        self.check_field(c)?;
        check_dt(dt)?;
        let (nx, ny) = (self.nx, self.ny);
        let lx = nx as f64 * self.dx;
        let ly = ny as f64 * self.dy;
        let eps2 = self.epsilon * self.epsilon;
        let g: Vec<f64> = c.iter().map(|&v| bulk_derivative(v)).collect();
        let (cr, ci) = forward_2d(c, nx, ny);
        let (gr, gi) = forward_2d(&g, nx, ny);
        let mut or = vec![0.0; self.n()];
        let mut oi = vec![0.0; self.n()];
        for mx in 0..nx {
            let kx = wavenumber(mx, nx, lx);
            for my in 0..ny {
                let ky = wavenumber(my, ny, ly);
                let kappa = kx * kx + ky * ky;
                let (fc, fg) = spectral_factors(kappa, dt, self.mobility, eps2, self.stabilization);
                let idx = mx * ny + my;
                or[idx] = fc * cr[idx] - fg * gr[idx];
                oi[idx] = fc * ci[idx] - fg * gi[idx];
            }
        }
        let c_new = inverse_real_2d(&or, &oi, nx, ny);
        c.copy_from_slice(&c_new);
        Ok(())
    }

    /// Integrate `n_steps` steps from `c0`, returning the final field.
    pub fn solve(&self, c0: &[f64], dt: f64, n_steps: usize) -> PdeResult<Vec<f64>> {
        self.check_field(c0)?;
        let mut c = c0.to_vec();
        for _ in 0..n_steps {
            self.step(&mut c, dt)?;
        }
        if c.iter().any(|v| !v.is_finite()) {
            return Err(PdeError::NumericalInstability(
                "Cahn-Hilliard 2d solution diverged to non-finite values".into(),
            ));
        }
        Ok(c)
    }

    /// Total discrete mass `∫c ≈ dx dy Σ c` (conserved by the scheme).
    pub fn total_mass(&self, c: &[f64]) -> PdeResult<f64> {
        self.check_field(c)?;
        Ok(c.iter().sum::<f64>() * self.dx * self.dy)
    }

    /// Ginzburg–Landau free energy `E = ∫(f(c) + ½ε²|∇c|²)` (spectral gradient term).
    pub fn free_energy(&self, c: &[f64]) -> PdeResult<f64> {
        self.check_field(c)?;
        let (nx, ny) = (self.nx, self.ny);
        let lx = nx as f64 * self.dx;
        let ly = ny as f64 * self.dy;
        let cell = self.dx * self.dy;
        let bulk: f64 = c.iter().map(|&v| bulk_potential(v)).sum::<f64>() * cell;
        let (cr, ci) = forward_2d(c, nx, ny);
        let mut grad = 0.0;
        for mx in 0..nx {
            let kx = wavenumber(mx, nx, lx);
            for my in 0..ny {
                let ky = wavenumber(my, ny, ly);
                let kappa = kx * kx + ky * ky;
                let idx = mx * ny + my;
                grad += kappa * (cr[idx] * cr[idx] + ci[idx] * ci[idx]);
            }
        }
        grad *= 0.5 * self.epsilon * self.epsilon * cell / self.n() as f64;
        Ok(bulk + grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn noisy_field(n: usize, mean: f64, amp: f64, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        (0..n).map(|_| mean + rng.next_range(-amp, amp)).collect()
    }

    // ── 1-D ──────────────────────────────────────────────────────────────────

    #[test]
    fn mass_is_conserved_1d() {
        let n = 32;
        let solver = CahnHilliard::new(1.0, 0.06, 1.0 / n as f64, n).expect("solver");
        let c0 = noisy_field(n, 0.0, 0.4, 11);
        let m0 = solver.total_mass(&c0).expect("mass");
        let c = solver.solve(&c0, 1.0e-4, 40).expect("solve");
        let m1 = solver.total_mass(&c).expect("mass");
        assert!((m1 - m0).abs() < 1.0e-10, "mass drift {}", (m1 - m0).abs());
    }

    #[test]
    fn free_energy_is_non_increasing_1d() {
        let n = 32;
        let dx = 1.0 / n as f64;
        let solver = CahnHilliard::new(1.0, 0.05, dx, n).expect("solver");
        // Smooth mean-zero cosine inside the spinodal region.
        let mut c: Vec<f64> = (0..n)
            .map(|i| 0.3 * (TAU * i as f64 / n as f64).cos())
            .collect();
        let dt = 1.0e-4;
        let mut e_prev = solver.free_energy(&c).expect("energy");
        for _ in 0..60 {
            solver.step(&mut c, dt).expect("step");
            let e = solver.free_energy(&c).expect("energy");
            assert!(e <= e_prev + 1.0e-10, "energy increased: {e_prev} -> {e}");
            e_prev = e;
        }
    }

    #[test]
    fn near_uniform_stays_near_uniform_1d() {
        // c₀ ≈ 0.8 lies on the stable branch (f''(0.8) > 0): perturbations decay.
        let n = 32;
        let solver = CahnHilliard::new(1.0, 0.05, 1.0 / n as f64, n).expect("solver");
        let c0 = noisy_field(n, 0.8, 0.01, 5);
        let c = solver.solve(&c0, 1.0e-4, 60).expect("solve");
        let max_dev = c.iter().fold(0.0_f64, |a, &v| a.max((v - 0.8).abs()));
        assert!(max_dev < 0.05, "max deviation from uniform {max_dev}");
    }

    #[test]
    fn exactly_uniform_is_steady_1d() {
        let n = 16;
        let solver = CahnHilliard::new(1.0, 0.1, 1.0 / n as f64, n).expect("solver");
        let c0 = vec![0.3; n];
        let c = solver.solve(&c0, 1.0e-3, 25).expect("solve");
        for &v in &c {
            assert!((v - 0.3).abs() < 1.0e-12, "uniform drifted to {v}");
        }
    }

    #[test]
    fn phase_separation_amplifies_toward_pm_one_1d() {
        // Spinodal decomposition from small mean-zero noise: amplitude grows
        // toward the wells ±1 while remaining bounded.
        let n = 64;
        let dx = 1.0 / n as f64;
        let solver = CahnHilliard::new(1.0, 2.5 * dx, dx, n).expect("solver");
        let c0 = noisy_field(n, 0.0, 0.05, 2024);
        let amp0 = c0.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        let c = solver.solve(&c0, 2.0e-3, 400).expect("solve");
        let amp1 = c.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        assert!(amp1 > 0.5, "amplitude did not grow: {amp0} -> {amp1}");
        assert!(amp1 < 1.5, "amplitude unbounded: {amp1}");
        assert!(c.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn shape_mismatch_is_rejected_1d() {
        let solver = CahnHilliard::new(1.0, 0.1, 0.1, 16).expect("solver");
        let mut c = vec![0.0; 15];
        assert!(matches!(
            solver.step(&mut c, 1.0e-3),
            Err(PdeError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn invalid_parameters_are_rejected_1d() {
        assert!(CahnHilliard::new(-1.0, 0.1, 0.1, 16).is_err());
        assert!(CahnHilliard::new(1.0, 0.0, 0.1, 16).is_err());
        assert!(CahnHilliard::new(1.0, 0.1, 0.1, 3).is_err());
        let solver = CahnHilliard::new(1.0, 0.1, 0.1, 16).expect("solver");
        assert!(solver.clone().with_stabilization(-1.0).is_err());
        assert!(matches!(
            solver.step(&mut [0.0; 16], 0.0),
            Err(PdeError::InvalidParameter { .. })
        ));
    }

    // ── 2-D ──────────────────────────────────────────────────────────────────

    #[test]
    fn mass_is_conserved_2d() {
        let (nx, ny) = (16, 16);
        let dx = 1.0 / nx as f64;
        let solver = CahnHilliard2d::new(1.0, 2.0 * dx, dx, dx, nx, ny).expect("solver");
        let c0 = noisy_field(nx * ny, 0.0, 0.3, 7);
        let m0 = solver.total_mass(&c0).expect("mass");
        let c = solver.solve(&c0, 1.0e-3, 30).expect("solve");
        let m1 = solver.total_mass(&c).expect("mass");
        assert!(
            (m1 - m0).abs() < 1.0e-10,
            "2d mass drift {}",
            (m1 - m0).abs()
        );
    }

    #[test]
    fn free_energy_is_non_increasing_2d() {
        let (nx, ny) = (16, 16);
        let dx = 1.0 / nx as f64;
        let solver = CahnHilliard2d::new(1.0, 2.0 * dx, dx, dx, nx, ny).expect("solver");
        let mut c: Vec<f64> = (0..nx * ny)
            .map(|idx| {
                let i = idx / ny;
                let j = idx % ny;
                0.25 * (TAU * i as f64 / nx as f64).cos() * (TAU * j as f64 / ny as f64).cos()
            })
            .collect();
        let dt = 5.0e-4;
        let mut e_prev = solver.free_energy(&c).expect("energy");
        for _ in 0..40 {
            solver.step(&mut c, dt).expect("step");
            let e = solver.free_energy(&c).expect("energy");
            assert!(e <= e_prev + 1.0e-10, "2d energy increased {e_prev} -> {e}");
            e_prev = e;
        }
    }

    #[test]
    fn phase_separation_amplifies_2d() {
        let (nx, ny) = (24, 24);
        let dx = 1.0 / nx as f64;
        let solver = CahnHilliard2d::new(1.0, 2.0 * dx, dx, dx, nx, ny).expect("solver");
        let c0 = noisy_field(nx * ny, 0.0, 0.05, 99);
        let c = solver.solve(&c0, 2.0e-3, 160).expect("solve");
        let amp = c.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        assert!(amp > 0.4, "2d amplitude did not grow: {amp}");
        assert!(amp < 1.5, "2d amplitude unbounded: {amp}");
        assert!(c.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn invalid_parameters_are_rejected_2d() {
        assert!(CahnHilliard2d::new(1.0, 0.1, 0.1, 0.1, 3, 8).is_err());
        assert!(CahnHilliard2d::new(1.0, -0.1, 0.1, 0.1, 8, 8).is_err());
    }
}
