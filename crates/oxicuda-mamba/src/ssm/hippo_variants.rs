//! HiPPO-LegT and HiPPO-FOUT alternative polynomial projection operators.
//!
//! HiPPO (High-order Polynomial Projection Operators) provides a principled
//! framework for online function approximation via SSM recurrences.  This module
//! implements two variants beyond the Scaled Legendre (LegS) baseline that is
//! already used in S4/S5:
//!
//! - **HiPPO-LegT** (Legendre, Translated): sliding window of fixed length θ,
//!   projects onto Legendre polynomials on `[t-θ, t]`.
//! - **HiPPO-FOUT** (Fourier, Outside): complex-exponential Fourier basis with
//!   linear damping; state pairs `(cos, sin)` per frequency.
//!
//! Both expose `step` (single-sample Euler update), `encode` (full sequence),
//! and `reconstruct` (recover signal samples from state).
//!
//! Also provides [`hippo_legs_matrix`] (Scaled Legendre companion) and
//! [`compare_hippo_variants`] for cross-variant benchmarking.
//!
//! ## Reference
//!
//! Gu et al. (2020) "HiPPO: Recurrent Memory with Optimal Polynomial
//! Projections", NeurIPS 2020. <https://arxiv.org/abs/2008.07669>

use crate::error::{MambaError, MambaResult};

// ─── Shared matrix struct ────────────────────────────────────────────────────

/// Dense HiPPO SSM matrices `(A, B)`.
///
/// `a` is stored row-major with shape `N × N`; `b` is `N × 1`.
#[derive(Debug, Clone)]
pub struct HippoMatrix {
    /// State transition matrix: `N × N`, row-major.
    pub a: Vec<f32>,
    /// Input matrix: `N`.
    pub b: Vec<f32>,
    /// Polynomial / Fourier order `N`.
    pub n: usize,
}

impl HippoMatrix {
    /// Compute `A @ c` (matrix-vector product), result shape `N`.
    fn matvec(&self, c: &[f32]) -> Vec<f32> {
        let n = self.n;
        (0..n)
            .map(|row| {
                let row_start = row * n;
                self.a[row_start..row_start + n]
                    .iter()
                    .zip(c.iter())
                    .map(|(ai, ci)| ai * ci)
                    .sum()
            })
            .collect()
    }
}

// ─── HiPPO-LegT ──────────────────────────────────────────────────────────────

/// Configuration for HiPPO-LegT (Legendre, Translated sliding window).
#[derive(Debug, Clone)]
pub struct HippoLegTConfig {
    /// Polynomial order `N` (number of basis functions).
    pub order: usize,
    /// Window length θ (time horizon, must be positive).
    pub theta: f32,
}

/// HiPPO-LegT operator: projects a function onto Legendre polynomials on a
/// sliding window `[t-θ, t]`.
///
/// Continuous-time ODE: `dc/dt = (A/θ) c + (B/θ) f(t)`
///
/// Euler discretization: `c_{t+1} = c_t + dt * (A @ c_t + B * f(t)) / θ`
pub struct HippoLegT {
    /// Configuration.
    pub cfg: HippoLegTConfig,
    /// Matrices `(A, B)`.
    pub matrix: HippoMatrix,
}

impl HippoLegT {
    /// Build LegT HiPPO matrices.
    ///
    /// `A[n,k]` (scaled by 1/θ before use in step):
    /// - `-(2n+1)` if `n == k`
    /// - `(2n+1) * (-1)^{n-k}` if `n > k`
    /// - `0` if `n < k`
    ///
    /// `B[n] = 2n+1` (scaled by 1/θ before use in step).
    ///
    /// # Errors
    ///
    /// - [`MambaError::InvalidSsmOrder`] if `order == 0`.
    /// - [`MambaError::NonPositiveDelta`] if `theta <= 0`.
    pub fn new(cfg: HippoLegTConfig) -> MambaResult<Self> {
        if cfg.order == 0 {
            return Err(MambaError::InvalidSsmOrder(0));
        }
        if cfg.theta <= 0.0 {
            return Err(MambaError::NonPositiveDelta(cfg.theta));
        }
        let matrix = build_legt_matrix(cfg.order)?;
        Ok(Self { cfg, matrix })
    }

    /// Euler update: `c_new = c + (dt/θ) * (A @ c + B * f_t)`.
    ///
    /// # Errors
    ///
    /// - [`MambaError::DimensionMismatch`] if `c.len() != N`.
    pub fn step(&self, c: &[f32], f_t: f32, dt: f32) -> MambaResult<Vec<f32>> {
        let n = self.cfg.order;
        if c.len() != n {
            return Err(MambaError::DimensionMismatch {
                expected: n,
                got: c.len(),
            });
        }
        let ac = self.matrix.matvec(c);
        let scale = dt / self.cfg.theta;
        let c_new: Vec<f32> = c
            .iter()
            .zip(ac.iter())
            .zip(self.matrix.b.iter())
            .map(|((ci, aci), bi)| ci + scale * (aci + bi * f_t))
            .collect();
        Ok(c_new)
    }

    /// Encode a sequence of function samples into the LegT state.
    ///
    /// Starts from a zero state and applies Euler steps with uniform `dt`.
    ///
    /// # Errors
    ///
    /// - [`MambaError::EmptyInput`] if `f_values` is empty.
    pub fn encode(&self, f_values: &[f32], dt: f32) -> MambaResult<Vec<f32>> {
        if f_values.is_empty() {
            return Err(MambaError::EmptyInput("f_values"));
        }
        let n = self.cfg.order;
        let mut c = vec![0.0_f32; n];
        for &f_t in f_values {
            c = self.step(&c, f_t, dt)?;
        }
        Ok(c)
    }

    /// Reconstruct the approximated function at `n_points` Legendre evaluation points.
    ///
    /// Uses Legendre polynomial values at `n_points` Chebyshev-like points in `[-1, 1]`:
    /// the reconstruction is `f̂(x) = Σ_k c_k * P_k(x) * sqrt(2k+1)`.
    ///
    /// # Errors
    ///
    /// - [`MambaError::DimensionMismatch`] if `c.len() != N`.
    /// - [`MambaError::EmptyInput`] if `n_points == 0`.
    pub fn reconstruct(&self, c: &[f32], n_points: usize) -> MambaResult<Vec<f32>> {
        let n = self.cfg.order;
        if c.len() != n {
            return Err(MambaError::DimensionMismatch {
                expected: n,
                got: c.len(),
            });
        }
        if n_points == 0 {
            return Err(MambaError::EmptyInput("n_points"));
        }
        // Evaluation points: uniform in [-1, 1]
        let xs: Vec<f32> = (0..n_points)
            .map(|i| -1.0 + 2.0 * (i as f32) / ((n_points - 1).max(1) as f32))
            .collect();

        let mut out = vec![0.0_f32; n_points];
        for (pi, &x) in xs.iter().enumerate() {
            // Evaluate Legendre polynomial P_k(x) via recurrence
            let p_vals = legendre_poly_values(n, x);
            out[pi] = c
                .iter()
                .zip(p_vals.iter())
                .enumerate()
                .map(|(k, (ck, pk))| ck * pk * ((2 * k + 1) as f32).sqrt())
                .sum();
        }
        Ok(out)
    }
}

// ─── HiPPO-FOUT ──────────────────────────────────────────────────────────────

/// Configuration for HiPPO-FOUT (Fourier basis, Outside).
#[derive(Debug, Clone)]
pub struct HippoFouConfig {
    /// Total state dimension `N` (must be even: N/2 cos + N/2 sin frequencies).
    pub order: usize,
    /// Maximum Fourier frequency for normalization (must be > 0).
    pub max_freq: usize,
}

/// HiPPO-FOUT operator: Fourier basis projection with damped complex exponential.
///
/// State `c = [a_1, b_1, a_2, b_2, ..., a_{N/2}, b_{N/2}]`
/// where `a_k` ≈ cos-component and `b_k` ≈ sin-component for frequency `k`.
///
/// Continuous ODE: `dc/dt = A c + B f(t)` with block-diagonal `A`.
pub struct HippoFou {
    /// Configuration.
    pub cfg: HippoFouConfig,
    /// Matrices `(A, B)`.
    pub matrix: HippoMatrix,
}

impl HippoFou {
    /// Build HiPPO-FOUT matrices.
    ///
    /// `order` must be even.
    ///
    /// Block structure for each frequency `k ∈ 1..=N/2`:
    /// ```text
    /// A_k = [[-k_eff,   k_eff*π],
    ///        [-k_eff*π, -k_eff  ]]
    /// ```
    /// where `k_eff = k / max_freq`.
    ///
    /// `B[2k] = 1.0`, `B[2k+1] = 0.0`.
    ///
    /// # Errors
    ///
    /// - [`MambaError::InvalidSsmOrder`] if `order == 0` or `order` is odd.
    /// - [`MambaError::DimensionMismatch`] if `max_freq == 0`.
    pub fn new(cfg: HippoFouConfig) -> MambaResult<Self> {
        if cfg.order == 0 {
            return Err(MambaError::InvalidSsmOrder(0));
        }
        if cfg.order % 2 != 0 {
            return Err(MambaError::InvalidSsmOrder(cfg.order));
        }
        if cfg.max_freq == 0 {
            return Err(MambaError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        let matrix = build_fou_matrix(cfg.order, cfg.max_freq)?;
        Ok(Self { cfg, matrix })
    }

    /// Euler update: `c_new = c + dt * (A @ c + B * f_t)`.
    ///
    /// # Errors
    ///
    /// - [`MambaError::DimensionMismatch`] if `c.len() != N`.
    pub fn step(&self, c: &[f32], f_t: f32, dt: f32) -> MambaResult<Vec<f32>> {
        let n = self.cfg.order;
        if c.len() != n {
            return Err(MambaError::DimensionMismatch {
                expected: n,
                got: c.len(),
            });
        }
        let ac = self.matrix.matvec(c);
        let c_new: Vec<f32> = c
            .iter()
            .zip(ac.iter())
            .zip(self.matrix.b.iter())
            .map(|((ci, aci), bi)| ci + dt * (aci + bi * f_t))
            .collect();
        Ok(c_new)
    }

    /// Encode a sequence into the Fourier state via Euler integration.
    ///
    /// # Errors
    ///
    /// - [`MambaError::EmptyInput`] if `f_values` is empty.
    pub fn encode(&self, f_values: &[f32], dt: f32) -> MambaResult<Vec<f32>> {
        if f_values.is_empty() {
            return Err(MambaError::EmptyInput("f_values"));
        }
        let n = self.cfg.order;
        let mut c = vec![0.0_f32; n];
        for &f_t in f_values {
            c = self.step(&c, f_t, dt)?;
        }
        Ok(c)
    }

    /// Reconstruct approximated signal at `n_points` uniform time samples in `[0, 1]`.
    ///
    /// Reconstruction: `ĝ(t) = Σ_{k=1}^{N/2} [a_k cos(2πkt) + b_k sin(2πkt)]`.
    ///
    /// # Errors
    ///
    /// - [`MambaError::DimensionMismatch`] if `c.len() != N`.
    /// - [`MambaError::EmptyInput`] if `n_points == 0`.
    pub fn reconstruct(&self, c: &[f32], n_points: usize) -> MambaResult<Vec<f32>> {
        let n = self.cfg.order;
        if c.len() != n {
            return Err(MambaError::DimensionMismatch {
                expected: n,
                got: c.len(),
            });
        }
        if n_points == 0 {
            return Err(MambaError::EmptyInput("n_points"));
        }
        let n_freqs = n / 2;
        let mut out = vec![0.0_f32; n_points];
        let two_pi = 2.0 * std::f32::consts::PI;

        for (pi, val) in out.iter_mut().enumerate() {
            let t = pi as f32 / n_points as f32;
            let mut sum = 0.0_f32;
            for k in 1..=n_freqs {
                let a_k = c[2 * (k - 1)];
                let b_k = c[2 * (k - 1) + 1];
                let angle = two_pi * (k as f32) * t;
                sum += a_k * angle.cos() + b_k * angle.sin();
            }
            *val = sum;
        }
        Ok(out)
    }
}

// ─── HiPPO-LegS matrix ───────────────────────────────────────────────────────

/// Compute the HiPPO-LegS (Scaled Legendre) A and B matrices.
///
/// These are the original S4 HiPPO matrices (Gu et al. 2021), included here
/// as a complement to the LegT and FOUT variants for cross-variant comparison.
///
/// - `A[n,k] = -sqrt(2n+1)*sqrt(2k+1)` for `n > k`
/// - `A[n,n] = -(n+1)`
/// - `A[n,k] = 0` for `n < k`
/// - `B[n] = sqrt(2n+1)`
///
/// # Errors
///
/// - [`MambaError::InvalidSsmOrder`] if `order == 0`.
pub fn hippo_legs_matrix(order: usize) -> MambaResult<HippoMatrix> {
    if order == 0 {
        return Err(MambaError::InvalidSsmOrder(0));
    }
    let n = order;
    let mut a = vec![0.0_f32; n * n];
    let mut b = vec![0.0_f32; n];

    for row in 0..n {
        let sqrt_2row_p1 = ((2 * row + 1) as f32).sqrt();
        b[row] = sqrt_2row_p1;

        for col in 0..n {
            a[row * n + col] = if col < row {
                let sqrt_2col_p1 = ((2 * col + 1) as f32).sqrt();
                -(sqrt_2row_p1 * sqrt_2col_p1)
            } else if col == row {
                -((row + 1) as f32)
            } else {
                0.0
            };
        }
    }
    Ok(HippoMatrix { a, b, n })
}

// ─── Cross-variant comparison ─────────────────────────────────────────────────

/// Encode a test signal with all three HiPPO variants and compute mean squared
/// reconstruction errors.
///
/// Returns `(legs_error, legt_error, fou_error)`.
///
/// # Errors
///
/// Propagates errors from matrix construction or encoding.
pub fn compare_hippo_variants(
    f_values: &[f32],
    dt: f32,
    order: usize,
    theta: f32,
) -> MambaResult<(f32, f32, f32)> {
    // LegS
    let legs = hippo_legs_matrix(order)?;
    let c_legs = legs_encode(f_values, dt, &legs)?;
    let n_pts = f_values.len().min(32);
    let recon_legs = legs_reconstruct(&c_legs, n_pts)?;
    let legs_err = mse(f_values, &recon_legs, n_pts);

    // LegT
    let legt_cfg = HippoLegTConfig { order, theta };
    let legt = HippoLegT::new(legt_cfg)?;
    let c_legt = legt.encode(f_values, dt)?;
    let recon_legt = legt.reconstruct(&c_legt, n_pts)?;
    let legt_err = mse(f_values, &recon_legt, n_pts);

    // FOUT (order must be even)
    let fou_order = if order % 2 == 0 { order } else { order + 1 };
    // max_freq = order/2
    let fou_cfg = HippoFouConfig {
        order: fou_order,
        max_freq: (fou_order / 2).max(1),
    };
    let fou = HippoFou::new(fou_cfg)?;
    let c_fou = fou.encode(f_values, dt)?;
    let recon_fou = fou.reconstruct(&c_fou, n_pts)?;
    let fou_err = mse(f_values, &recon_fou, n_pts);

    Ok((legs_err, legt_err, fou_err))
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Build the LegT A matrix (unscaled, scaling by 1/θ is done inside `step`).
fn build_legt_matrix(n: usize) -> MambaResult<HippoMatrix> {
    let mut a = vec![0.0_f32; n * n];
    let mut b = vec![0.0_f32; n];

    for row in 0..n {
        // B[n] = 2n + 1  (scaled by 1/θ in step)
        b[row] = (2 * row + 1) as f32;

        for col in 0..n {
            a[row * n + col] = if col == row {
                // Diagonal: -(2n+1)
                -((2 * row + 1) as f32)
            } else if col < row {
                // Lower-triangular: (2n+1) * (-1)^{n-k}
                let sign = if (row - col) % 2 == 0 {
                    1.0_f32
                } else {
                    -1.0_f32
                };
                (2 * row + 1) as f32 * sign
            } else {
                // Upper-triangular: 0
                0.0
            };
        }
    }
    Ok(HippoMatrix { a, b, n })
}

/// Build the FOUT Fourier matrix.
///
/// Block-diagonal structure: for each frequency index `k ∈ 0..n_freqs`
/// (1-indexed frequency = k+1):
/// ```text
/// A[2k, 2k]   = -k_eff        (cos decay)
/// A[2k+1,2k+1]= -k_eff        (sin decay)
/// A[2k, 2k+1] = k_eff * π     (cos→sin coupling)
/// A[2k+1, 2k] = -k_eff * π    (sin→cos coupling)
/// ```
/// where `k_eff = (k+1) / max_freq`.
fn build_fou_matrix(n: usize, max_freq: usize) -> MambaResult<HippoMatrix> {
    let n_freqs = n / 2;
    let mut a = vec![0.0_f32; n * n];
    let b: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0_f32 } else { 0.0_f32 })
        .collect();

    for k in 0..n_freqs {
        let freq = (k + 1) as f32; // 1-indexed frequency
        let k_eff = freq / (max_freq as f32);
        let row_cos = 2 * k;
        let row_sin = 2 * k + 1;

        // Diagonal: damping
        a[row_cos * n + row_cos] = -k_eff;
        a[row_sin * n + row_sin] = -k_eff;

        // Off-diagonal coupling: cos↔sin rotation
        a[row_cos * n + row_sin] = k_eff * std::f32::consts::PI;
        a[row_sin * n + row_cos] = -k_eff * std::f32::consts::PI;
    }
    Ok(HippoMatrix { a, b, n })
}

/// Encode signal through LegS using raw matrix (forward Euler).
fn legs_encode(f_values: &[f32], dt: f32, mat: &HippoMatrix) -> MambaResult<Vec<f32>> {
    if f_values.is_empty() {
        return Err(MambaError::EmptyInput("f_values"));
    }
    let n = mat.n;
    let mut c = vec![0.0_f32; n];
    for &f_t in f_values {
        let ac = mat.matvec(&c);
        for (i, ci) in c.iter_mut().enumerate() {
            *ci += dt * (ac[i] + mat.b[i] * f_t);
        }
    }
    Ok(c)
}

/// Reconstruct LegS approximation using Legendre polynomial evaluation.
fn legs_reconstruct(c: &[f32], n_points: usize) -> MambaResult<Vec<f32>> {
    if n_points == 0 {
        return Err(MambaError::EmptyInput("n_points"));
    }
    let n = c.len();
    let mut out = vec![0.0_f32; n_points];
    for (pi, val) in out.iter_mut().enumerate() {
        let x = -1.0 + 2.0 * (pi as f32) / ((n_points - 1).max(1) as f32);
        let p_vals = legendre_poly_values(n, x);
        *val = c
            .iter()
            .zip(p_vals.iter())
            .enumerate()
            .map(|(k, (ck, pk))| ck * pk * ((2 * k + 1) as f32).sqrt())
            .sum();
    }
    Ok(out)
}

/// Evaluate the first `n` Legendre polynomials at `x ∈ [-1, 1]` via recurrence.
///
/// P_0(x) = 1, P_1(x) = x,
/// P_{k+1}(x) = ((2k+1)*x*P_k(x) - k*P_{k-1}(x)) / (k+1).
fn legendre_poly_values(n: usize, x: f32) -> Vec<f32> {
    let mut p = vec![0.0_f32; n.max(1)];
    if n == 0 {
        return p;
    }
    p[0] = 1.0;
    if n == 1 {
        return p;
    }
    p[1] = x;
    for k in 1..(n - 1) {
        let two_k_p1 = (2 * k + 1) as f32;
        let k_f = k as f32;
        let k_p1 = (k + 1) as f32;
        p[k + 1] = (two_k_p1 * x * p[k] - k_f * p[k - 1]) / k_p1;
    }
    p
}

/// Mean squared error between `expected` and `predicted` at the first `n` elements.
fn mse(expected: &[f32], predicted: &[f32], n: usize) -> f32 {
    let n = n.min(expected.len()).min(predicted.len());
    if n == 0 {
        return 0.0;
    }
    let sum_sq: f32 = expected[..n]
        .iter()
        .zip(predicted[..n].iter())
        .map(|(e, p)| (e - p).powi(2))
        .sum();
    sum_sq / n as f32
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPS: f32 = 1e-5;

    fn make_legt_cfg(order: usize) -> HippoLegTConfig {
        HippoLegTConfig { order, theta: 1.0 }
    }

    fn make_fou_cfg(order: usize) -> HippoFouConfig {
        HippoFouConfig {
            order,
            max_freq: order / 2,
        }
    }

    // ── LegT shape tests ──────────────────────────────────────────────────────

    #[test]
    fn legt_matrix_shape() {
        let order = 8;
        let legt = HippoLegT::new(make_legt_cfg(order)).expect("LegT n=8");
        assert_eq!(legt.matrix.a.len(), order * order);
        assert_eq!(legt.matrix.b.len(), order);
        assert_eq!(legt.matrix.n, order);
    }

    #[test]
    fn legt_step_shape() {
        let order = 6;
        let legt = HippoLegT::new(make_legt_cfg(order)).expect("LegT n=6");
        let c = vec![0.1_f32; order];
        let c_new = legt.step(&c, 1.0, 0.01).expect("step");
        assert_eq!(c_new.len(), order);
    }

    #[test]
    fn legt_encode_constant_signal() {
        // Encoding a constant signal f(t)=1 should produce non-zero state
        let order = 4;
        let legt = HippoLegT::new(make_legt_cfg(order)).expect("LegT");
        let f_vals = vec![1.0_f32; 100];
        let c = legt.encode(&f_vals, 0.01).expect("encode");
        assert_eq!(c.len(), order);
        // State should be non-zero after many steps
        let c_max = c.iter().cloned().map(f32::abs).fold(0.0_f32, f32::max);
        assert!(c_max > 0.0, "state should be non-zero for constant input");
        assert!(c.iter().all(|v| v.is_finite()), "state must be finite");
    }

    #[test]
    fn legt_encode_output_shape() {
        let order = 10;
        let legt = HippoLegT::new(make_legt_cfg(order)).expect("LegT");
        let f_vals: Vec<f32> = (0..50).map(|i| (i as f32) * 0.1).collect();
        let c = legt.encode(&f_vals, 0.02).expect("encode");
        assert_eq!(c.len(), order);
    }

    #[test]
    fn legt_reconstruct_shape() {
        let order = 6;
        let legt = HippoLegT::new(make_legt_cfg(order)).expect("LegT");
        let c = vec![0.1_f32; order];
        let pts = legt.reconstruct(&c, 20).expect("reconstruct");
        assert_eq!(pts.len(), 20);
    }

    #[test]
    fn legt_encode_then_reconstruct() {
        // Encode a sin signal; verify reconstruction is finite and non-trivial
        let order = 12;
        let theta = 1.0_f32;
        let legt = HippoLegT::new(HippoLegTConfig { order, theta }).expect("LegT");
        let n_samples = 200;
        let f_vals: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * PI * i as f32 / n_samples as f32).sin())
            .collect();
        let c = legt
            .encode(&f_vals, 1.0 / n_samples as f32)
            .expect("encode");
        let recon = legt.reconstruct(&c, 32).expect("reconstruct");
        assert_eq!(recon.len(), 32);
        assert!(
            recon.iter().all(|v| v.is_finite()),
            "reconstruction must be finite"
        );
        // Reconstruction should produce non-zero values for a sin signal
        let max_recon = recon.iter().cloned().map(f32::abs).fold(0.0_f32, f32::max);
        assert!(max_recon > 0.0, "reconstruction should be non-trivial");
    }

    // ── FOUT shape tests ──────────────────────────────────────────────────────

    #[test]
    fn fou_matrix_shape() {
        let order = 8;
        let fou = HippoFou::new(make_fou_cfg(order)).expect("FOUT n=8");
        assert_eq!(fou.matrix.a.len(), order * order);
        assert_eq!(fou.matrix.b.len(), order);
        assert_eq!(fou.matrix.n, order);
    }

    #[test]
    fn fou_step_shape() {
        let order = 6;
        let fou = HippoFou::new(make_fou_cfg(order)).expect("FOUT");
        let c = vec![0.1_f32; order];
        let c_new = fou.step(&c, 1.0, 0.01).expect("step");
        assert_eq!(c_new.len(), order);
    }

    #[test]
    fn fou_encode_shape() {
        let order = 8;
        let fou = HippoFou::new(make_fou_cfg(order)).expect("FOUT");
        let f_vals: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let c = fou.encode(&f_vals, 0.01).expect("encode");
        assert_eq!(c.len(), order);
    }

    #[test]
    fn fou_encode_sinusoidal() {
        // A pure sinusoid at frequency 1 should produce large a_1/b_1 components
        let order = 8;
        let max_freq = order / 2;
        let fou = HippoFou::new(HippoFouConfig { order, max_freq }).expect("FOUT");
        let n_samples = 500;
        let dt = 1.0_f32 / n_samples as f32;
        let f_vals: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * PI * i as f32 * dt).sin())
            .collect();
        let c = fou.encode(&f_vals, dt).expect("encode");
        assert_eq!(c.len(), order);
        assert!(c.iter().all(|v| v.is_finite()), "state must be finite");
        // State should contain non-trivial values
        let c_max = c.iter().cloned().map(f32::abs).fold(0.0_f32, f32::max);
        assert!(c_max > 0.0, "Fourier state should be non-zero for sinusoid");
    }

    #[test]
    fn fou_reconstruct_shape() {
        let order = 6;
        let fou = HippoFou::new(make_fou_cfg(order)).expect("FOUT");
        let c = vec![0.1_f32; order];
        let pts = fou.reconstruct(&c, 24).expect("reconstruct");
        assert_eq!(pts.len(), 24);
    }

    // ── LegS matrix ───────────────────────────────────────────────────────────

    #[test]
    fn legs_matrix_shape() {
        let order = 8;
        let mat = hippo_legs_matrix(order).expect("LegS");
        assert_eq!(mat.a.len(), order * order);
        assert_eq!(mat.b.len(), order);
        assert_eq!(mat.n, order);
    }

    #[test]
    fn legs_matrix_diagonal_negative() {
        // Diagonal entries must be negative for stability
        let order = 8;
        let mat = hippo_legs_matrix(order).expect("LegS");
        for n in 0..order {
            let diag = mat.a[n * order + n];
            assert!(diag < 0.0, "A[{n},{n}]={diag} should be negative");
        }
    }

    // ── Comparison ────────────────────────────────────────────────────────────

    #[test]
    fn compare_hippo_returns_triple() {
        let n_samples = 64;
        let dt = 1.0_f32 / n_samples as f32;
        let f_vals: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * PI * 2.0 * i as f32 * dt).sin())
            .collect();
        let (legs_err, legt_err, fou_err) =
            compare_hippo_variants(&f_vals, dt, 8, 1.0).expect("compare");
        assert!(legs_err.is_finite(), "legs_err must be finite");
        assert!(legt_err.is_finite(), "legt_err must be finite");
        assert!(fou_err.is_finite(), "fou_err must be finite");
        assert!(
            legs_err >= 0.0 && legt_err >= 0.0 && fou_err >= 0.0,
            "errors must be non-negative"
        );
    }

    // ── Error tests ───────────────────────────────────────────────────────────

    #[test]
    fn err_order_zero() {
        let result = HippoLegT::new(HippoLegTConfig {
            order: 0,
            theta: 1.0,
        });
        assert!(result.is_err(), "order=0 should fail");
        let result2 = HippoFou::new(HippoFouConfig {
            order: 0,
            max_freq: 1,
        });
        assert!(result2.is_err(), "order=0 should fail for FOUT");
    }

    #[test]
    fn err_theta_nonpositive() {
        let result = HippoLegT::new(HippoLegTConfig {
            order: 4,
            theta: 0.0,
        });
        assert!(result.is_err(), "theta=0 should fail");
        let result2 = HippoLegT::new(HippoLegTConfig {
            order: 4,
            theta: -1.0,
        });
        assert!(result2.is_err(), "theta<0 should fail");
    }

    #[test]
    fn err_fou_order_odd() {
        // FOUT order must be even
        let result = HippoFou::new(HippoFouConfig {
            order: 5,
            max_freq: 2,
        });
        assert!(result.is_err(), "odd order should fail for FOUT");
        let result2 = HippoFou::new(HippoFouConfig {
            order: 7,
            max_freq: 3,
        });
        assert!(result2.is_err(), "odd order=7 should fail for FOUT");
    }

    #[test]
    fn legt_step_wrong_dim() {
        let order = 4;
        let legt = HippoLegT::new(make_legt_cfg(order)).expect("LegT");
        let bad_c = vec![0.0_f32; order + 1];
        assert!(legt.step(&bad_c, 1.0, 0.01).is_err());
    }

    #[test]
    fn legt_a_diagonal_is_negative() {
        // LegT A diagonal = -(2n+1) which is always negative
        let order = 6;
        let legt = HippoLegT::new(make_legt_cfg(order)).expect("LegT");
        for n in 0..order {
            let diag = legt.matrix.a[n * order + n];
            assert!(diag < 0.0, "LegT A[{n},{n}]={diag} should be negative");
        }
    }

    #[test]
    fn fou_b_vector_structure() {
        // B[2k] = 1.0, B[2k+1] = 0.0
        let order = 8;
        let fou = HippoFou::new(make_fou_cfg(order)).expect("FOUT");
        for k in 0..order / 2 {
            assert!(
                (fou.matrix.b[2 * k] - 1.0).abs() < EPS,
                "B[{}] should be 1.0",
                2 * k
            );
            assert!(
                (fou.matrix.b[2 * k + 1]).abs() < EPS,
                "B[{}] should be 0.0",
                2 * k + 1
            );
        }
    }

    #[test]
    fn legs_matrix_upper_triangular_zero() {
        // LegS is lower-triangular: A[n,k] = 0 for k > n
        let order = 6;
        let mat = hippo_legs_matrix(order).expect("LegS");
        for n in 0..order {
            for k in (n + 1)..order {
                let val = mat.a[n * order + k];
                assert!(
                    val.abs() < EPS,
                    "LegS A[{n},{k}]={val} should be 0 (upper-triangular)"
                );
            }
        }
    }
}
