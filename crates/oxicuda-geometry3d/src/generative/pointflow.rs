//! PointFlow-style continuous-normalizing-flow (CNF) generative core.
//!
//! Reference: Yang, Huang, Hao, Liu, Belongie, Hariharan, *"PointFlow: 3D Point
//! Cloud Generation with Continuous Normalizing Flows"*, ICCV 2019.
//!
//! A continuous normalizing flow (CNF) defines an exactly-invertible map by
//! integrating an ordinary differential equation whose right-hand side is a
//! learnable velocity field `f_theta(x, t)`:
//!
//! ```text
//! dx/dt = f_theta(x, t),            t in [0, 1]
//! ```
//!
//! Integrating from `t = 0` to `t = 1` transports a base sample `x0` to a data
//! sample `x1 = F(x0)`; integrating the *same* field in reverse time recovers
//! `x0 = F^{-1}(x1)`, so `F` is a continuous diffeomorphism (a true bijection).
//!
//! Along the trajectory the log-density evolves by the *instantaneous change of
//! variables* (Chen et al., NeurIPS 2018):
//!
//! ```text
//! d (log p(x(t))) / dt = - tr( d f_theta / d x )
//! ```
//!
//! so that, with `delta_logp = integral_0^1 -tr(df/dx) dt` accumulated along the
//! forward path,
//!
//! ```text
//! log p_data(x1) = log p_base(x0) + delta_logp,      x0 = F^{-1}(x1).
//! ```
//!
//! This is the *real* CNF density — not an approximation of one.
//!
//! # What is verifiable on CPU (and is tested here)
//!
//! Everything in this module is exact, deterministic, and CPU-checkable:
//!
//! * **Invertibility** — `inverse(forward(x0)) == x0` to integration tolerance,
//!   because reverse-time integration retraces the forward trajectory.
//! * **Change of variables / log-det** — the accumulated `delta_logp` matches
//!   `-log|det J_F|` of the overall forward map (cross-checked against a
//!   finite-difference Jacobian), and `delta_logp_forward + delta_logp_inverse
//!   == 0` round-trip.
//! * **Density** — `log_prob` is finite, deterministic under a fixed seed, and
//!   integrates to one over a coarse grid in low dimension (it is a normalized
//!   density by construction).
//!
//! The trace `tr(df/dx)` is computed **exactly** for the small (3-D / small
//! latent) case as the sum of per-dimension diagonal partials via central
//! finite differences — *not* the stochastic Hutchinson estimator — so the
//! result is fully deterministic and verifiable.
//!
//! # What is NOT claimed here (left honestly unverified)
//!
//! The velocity field is **randomly initialized, never trained**. The structure
//! that `sample` produces is therefore a deformation of Gaussian noise, **not a
//! learned shape**. This module makes **no** claim about generated point-cloud
//! realism / Chamfer distance to any dataset — training that to fidelity needs
//! data, an optimizer, and GPU-scale compute, which is outside this CPU
//! unit-verifiable core.
//!
//! # Numerics
//!
//! The flow runs in `f64` throughout. The whole point of the module is rigorous
//! numerical verification (invertibility to `1e-4`, log-det to `1e-2`), for
//! which `f64` integration and finite differencing are required; `f64` is
//! already used by this crate's geometry predicates (Delaunay, marching cubes).

use crate::error::{Geom3dError, Geom3dResult};
use crate::handle::LcgRng;

// ─── RNG helpers (f64, full-range ÷2³²) ───────────────────────────────────────

/// Draw a uniform `f64` in `[0, 1)`.
///
/// `LcgRng::next_u32()` spans the full 32-bit range `[0, 2^32)`, so dividing by
/// `2^32` (never `2^31`) yields a true unit uniform.
#[inline]
fn next_unit_f64(rng: &mut LcgRng) -> f64 {
    rng.next_u32() as f64 / 4_294_967_296.0_f64 // 2^32
}

/// Draw a uniform `f64` in `[-1, 1)`.
#[inline]
fn uniform_pm1_f64(rng: &mut LcgRng) -> f64 {
    next_unit_f64(rng) * 2.0 - 1.0
}

/// Sample two independent `N(0, 1)` values via the Box–Muller transform (`f64`).
#[inline]
fn next_normal_pair_f64(rng: &mut LcgRng) -> (f64, f64) {
    // Clamp away from exactly 0 so `ln` is finite.
    let u1 = next_unit_f64(rng).max(1e-12);
    let u2 = next_unit_f64(rng);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    (r * theta.cos(), r * theta.sin())
}

/// Fill `buf` with `N(0, std^2)` samples via Box–Muller.
fn fill_normal_f64(rng: &mut LcgRng, buf: &mut [f64], std: f64) {
    let mut i = 0;
    while i + 1 < buf.len() {
        let (a, b) = next_normal_pair_f64(rng);
        buf[i] = a * std;
        buf[i + 1] = b * std;
        i += 2;
    }
    if i < buf.len() {
        let (a, _) = next_normal_pair_f64(rng);
        buf[i] = a * std;
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a [`ContinuousNormalizingFlow`].
#[derive(Debug, Clone)]
pub struct CnfConfig {
    /// State dimension `D` (3 for raw point coordinates, or a small latent).
    pub dim: usize,
    /// Hidden width of the velocity-field MLP (ignored when `depth == 0`).
    pub hidden: usize,
    /// Number of hidden (tanh) layers. `depth == 0` yields an exactly *linear*
    /// velocity field `f(x, t) = A·x + c0 + c1·t`, useful for analytic checks.
    pub depth: usize,
    /// Number of fixed-step RK4 integration steps over `t in [0, 1]`.
    pub n_steps: usize,
    /// Weight-initialization scale (kept small so the untrained flow is a mild,
    /// stably-invertible deformation).
    pub init_scale: f64,
    /// Central finite-difference step used for the exact diagonal-trace.
    pub trace_eps: f64,
}

impl Default for CnfConfig {
    fn default() -> Self {
        Self {
            dim: 3,
            hidden: 32,
            depth: 2,
            n_steps: 40,
            init_scale: 0.3,
            trace_eps: 1e-5,
        }
    }
}

// ─── Dense layer ──────────────────────────────────────────────────────────────

/// A single dense layer `out = W·in + b`, row-major `W` of shape `[out × in]`.
#[derive(Debug, Clone)]
struct DenseLayer {
    w: Vec<f64>,
    b: Vec<f64>,
    in_dim: usize,
    out_dim: usize,
}

impl DenseLayer {
    fn new(rng: &mut LcgRng, in_dim: usize, out_dim: usize, scale: f64) -> Self {
        let limit = scale / (in_dim.max(1) as f64).sqrt();
        let mut w = vec![0.0_f64; out_dim * in_dim];
        for v in &mut w {
            *v = uniform_pm1_f64(rng) * limit;
        }
        // Small but nonzero biases ⇒ the velocity field has nonzero divergence
        // generically, so the log-det check is non-vacuous.
        let mut b = vec![0.0_f64; out_dim];
        for v in &mut b {
            *v = uniform_pm1_f64(rng) * limit * 0.5;
        }
        Self {
            w,
            b,
            in_dim,
            out_dim,
        }
    }

    /// Apply the affine map to a single input vector of length `in_dim`.
    fn apply(&self, input: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0_f64; self.out_dim];
        for (o, out_o) in out.iter_mut().enumerate() {
            let row = &self.w[o * self.in_dim..(o + 1) * self.in_dim];
            *out_o = self.b[o] + row.iter().zip(input).map(|(w, x)| w * x).sum::<f64>();
        }
        out
    }
}

// ─── Velocity field f_theta(x, t) ─────────────────────────────────────────────

/// The learnable velocity field `f_theta : R^D × [0, 1] → R^D`.
///
/// Time is concatenated as an extra input coordinate (FFJORD-style). Hidden
/// layers use `tanh` (smooth, bounded derivative — good for stable integration
/// and accurate finite-difference traces); the output layer is linear.
#[derive(Debug, Clone)]
struct VelocityField {
    layers: Vec<DenseLayer>,
    dim: usize,
}

impl VelocityField {
    fn new(cfg: &CnfConfig, rng: &mut LcgRng) -> Self {
        let d = cfg.dim;
        let in0 = d + 1; // [x ; t]
        let mut layers = Vec::new();
        if cfg.depth == 0 {
            // Exactly-linear field: single linear layer (no activation).
            layers.push(DenseLayer::new(rng, in0, d, cfg.init_scale));
        } else {
            layers.push(DenseLayer::new(rng, in0, cfg.hidden, cfg.init_scale));
            for _ in 1..cfg.depth {
                layers.push(DenseLayer::new(rng, cfg.hidden, cfg.hidden, cfg.init_scale));
            }
            layers.push(DenseLayer::new(rng, cfg.hidden, d, cfg.init_scale));
        }
        Self { layers, dim: d }
    }

    /// Evaluate `f_theta(x, t)`.
    fn eval(&self, x: &[f64], t: f64) -> Vec<f64> {
        let mut h = Vec::with_capacity(x.len() + 1);
        h.extend_from_slice(x);
        h.push(t);
        let last = self.layers.len() - 1;
        for (k, layer) in self.layers.iter().enumerate() {
            h = layer.apply(&h);
            if k < last {
                for v in &mut h {
                    *v = v.tanh();
                }
            }
        }
        h
    }

    /// Exact `-tr(df/dx)` via per-dimension central finite differences.
    ///
    /// `tr(df/dx) = Σ_i ∂f_i/∂x_i`; each diagonal partial is estimated by
    /// `(f_i(x + eps·e_i) - f_i(x - eps·e_i)) / (2·eps)`. For a linear field this
    /// is exact for any `eps`; for the tanh field it is accurate to `O(eps^2)`.
    fn neg_trace(&self, x: &[f64], t: f64, eps: f64) -> f64 {
        let mut trace = 0.0_f64;
        let mut probe = x.to_vec();
        for i in 0..self.dim {
            let orig = probe[i];
            probe[i] = orig + eps;
            let fp = self.eval(&probe, t);
            probe[i] = orig - eps;
            let fm = self.eval(&probe, t);
            probe[i] = orig;
            trace += (fp[i] - fm[i]) / (2.0 * eps);
        }
        -trace
    }
}

// ─── Continuous normalizing flow ──────────────────────────────────────────────

/// A continuous normalizing flow over a `D`-dimensional state.
///
/// Integrates `dx/dt = f_theta(x, t)` with a self-contained fixed-step RK4
/// scheme, jointly accumulating the change-of-variables log-density term.
#[derive(Debug, Clone)]
pub struct ContinuousNormalizingFlow {
    field: VelocityField,
    dim: usize,
    n_steps: usize,
    trace_eps: f64,
}

impl ContinuousNormalizingFlow {
    /// Build a flow with the given configuration and RNG-initialized field.
    ///
    /// # Errors
    /// Returns [`Geom3dError::Internal`] if the configuration is degenerate
    /// (`dim == 0`, `n_steps == 0`, non-positive `trace_eps`, or `hidden == 0`
    /// with `depth > 0`).
    pub fn new(cfg: CnfConfig, rng: &mut LcgRng) -> Geom3dResult<Self> {
        if cfg.dim == 0 {
            return Err(Geom3dError::Internal("CNF dim must be >= 1".to_string()));
        }
        if cfg.depth > 0 && cfg.hidden == 0 {
            return Err(Geom3dError::Internal(
                "CNF hidden must be >= 1 when depth > 0".to_string(),
            ));
        }
        if cfg.n_steps == 0 {
            return Err(Geom3dError::Internal(
                "CNF n_steps must be >= 1".to_string(),
            ));
        }
        if !(cfg.trace_eps > 0.0 && cfg.trace_eps.is_finite()) {
            return Err(Geom3dError::Internal(
                "CNF trace_eps must be positive and finite".to_string(),
            ));
        }
        let field = VelocityField::new(&cfg, rng);
        Ok(Self {
            field,
            dim: cfg.dim,
            n_steps: cfg.n_steps,
            trace_eps: cfg.trace_eps,
        })
    }

    /// State dimension `D`.
    #[must_use]
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of RK4 integration steps.
    #[must_use]
    #[inline]
    pub fn n_steps(&self) -> usize {
        self.n_steps
    }

    /// Augmented derivative `(dx/dt, d(logp)/dt) = (f, -tr(df/dx))`.
    fn deriv(&self, x: &[f64], t: f64) -> (Vec<f64>, f64) {
        let f = self.field.eval(x, t);
        let dl = self.field.neg_trace(x, t, self.trace_eps);
        (f, dl)
    }

    /// Integrate the augmented ODE from `t0` to `t1` with fixed-step RK4.
    ///
    /// Returns `(x(t1), Δlogp)` where `Δlogp = integral_{t0}^{t1} -tr(df/dx) dt`.
    /// Works for both forward (`t0 = 0, t1 = 1`) and reverse (`t0 = 1, t1 = 0`,
    /// giving a negative step) integration.
    fn integrate(&self, x0: &[f64], t0: f64, t1: f64) -> (Vec<f64>, f64) {
        let n = self.n_steps;
        let h = (t1 - t0) / n as f64;
        let mut x = x0.to_vec();
        let mut logp = 0.0_f64;
        let mut t = t0;
        for _ in 0..n {
            let (k1x, k1l) = self.deriv(&x, t);
            let x2: Vec<f64> = x.iter().zip(&k1x).map(|(xi, k)| xi + 0.5 * h * k).collect();
            let (k2x, k2l) = self.deriv(&x2, t + 0.5 * h);
            let x3: Vec<f64> = x.iter().zip(&k2x).map(|(xi, k)| xi + 0.5 * h * k).collect();
            let (k3x, k3l) = self.deriv(&x3, t + 0.5 * h);
            let x4: Vec<f64> = x.iter().zip(&k3x).map(|(xi, k)| xi + h * k).collect();
            let (k4x, k4l) = self.deriv(&x4, t + h);
            for (i, xi) in x.iter_mut().enumerate() {
                *xi += h / 6.0 * (k1x[i] + 2.0 * k2x[i] + 2.0 * k3x[i] + k4x[i]);
            }
            logp += h / 6.0 * (k1l + 2.0 * k2l + 2.0 * k3l + k4l);
            t += h;
        }
        (x, logp)
    }

    fn check_dim(&self, x: &[f64]) -> Geom3dResult<()> {
        if x.len() != self.dim {
            return Err(Geom3dError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        Ok(())
    }

    /// Forward map: transport base sample `x0` to data sample `x1`, returning
    /// `(x1, delta_logp)` with `delta_logp = integral_0^1 -tr(df/dx) dt`.
    ///
    /// # Errors
    /// Returns [`Geom3dError::DimensionMismatch`] if `x0.len() != dim`.
    pub fn forward(&self, x0: &[f64]) -> Geom3dResult<(Vec<f64>, f64)> {
        self.check_dim(x0)?;
        Ok(self.integrate(x0, 0.0, 1.0))
    }

    /// Inverse map (reverse-time integration), returning `(x0, delta_logp)`.
    ///
    /// The returned `delta_logp = integral_1^0 -tr(df/dx) dt` satisfies
    /// `forward.delta_logp + inverse.delta_logp ≈ 0` for a forward/inverse pair.
    ///
    /// # Errors
    /// Returns [`Geom3dError::DimensionMismatch`] if `x1.len() != dim`.
    pub fn inverse_with_logp(&self, x1: &[f64]) -> Geom3dResult<(Vec<f64>, f64)> {
        self.check_dim(x1)?;
        Ok(self.integrate(x1, 1.0, 0.0))
    }

    /// Inverse map: recover `x0` from data sample `x1` by reverse-time
    /// integration of the same field (`inverse(forward(x0)) ≈ x0`).
    ///
    /// # Errors
    /// Returns [`Geom3dError::DimensionMismatch`] if `x1.len() != dim`.
    pub fn inverse(&self, x1: &[f64]) -> Geom3dResult<Vec<f64>> {
        Ok(self.inverse_with_logp(x1)?.0)
    }
}

// ─── PointFlow model ──────────────────────────────────────────────────────────

/// A PointFlow-style density: an isotropic Gaussian base distribution composed
/// with a [`ContinuousNormalizingFlow`].
///
/// The same flow is applied independently per point (PointFlow models a point
/// cloud as i.i.d. draws from a shared CNF prior), so `sample(n)` returns `n`
/// flowed points and `log_prob_cloud` sums the per-point log-densities.
#[derive(Debug, Clone)]
pub struct PointFlowModel {
    cnf: ContinuousNormalizingFlow,
    base_std: f64,
    dim: usize,
}

impl PointFlowModel {
    /// Build a model with an isotropic `N(0, base_std^2 · I)` base density.
    ///
    /// # Errors
    /// Returns [`Geom3dError::Internal`] if `base_std` is not positive/finite, or
    /// propagates [`ContinuousNormalizingFlow::new`] configuration errors.
    pub fn new(cfg: CnfConfig, base_std: f64, rng: &mut LcgRng) -> Geom3dResult<Self> {
        if !(base_std > 0.0 && base_std.is_finite()) {
            return Err(Geom3dError::Internal(format!(
                "base_std must be positive and finite, got {base_std}"
            )));
        }
        let dim = cfg.dim;
        let cnf = ContinuousNormalizingFlow::new(cfg, rng)?;
        Ok(Self { cnf, base_std, dim })
    }

    /// State dimension `D`.
    #[must_use]
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Borrow the underlying continuous normalizing flow.
    #[must_use]
    #[inline]
    pub fn cnf(&self) -> &ContinuousNormalizingFlow {
        &self.cnf
    }

    /// Log-density of the isotropic Gaussian base `N(0, base_std^2 · I_D)`.
    fn base_logp(&self, z: &[f64]) -> f64 {
        let d = self.dim as f64;
        let var = self.base_std * self.base_std;
        let sq: f64 = z.iter().map(|v| v * v).sum();
        -0.5 * sq / var - d * self.base_std.ln() - 0.5 * d * std::f64::consts::TAU.ln()
    }

    /// Exact CNF log-density `log p_data(x) = base_logp(F^{-1}(x)) + delta_logp`.
    ///
    /// # Errors
    /// Returns [`Geom3dError::DimensionMismatch`] if `x.len() != dim`.
    pub fn log_prob(&self, x: &[f64]) -> Geom3dResult<f64> {
        let x0 = self.cnf.inverse(x)?;
        let (_, delta_logp) = self.cnf.forward(&x0)?;
        Ok(self.base_logp(&x0) + delta_logp)
    }

    /// Sum of per-point log-densities for a row-major `[n × dim]` point set.
    ///
    /// # Errors
    /// Returns [`Geom3dError::DimensionMismatch`] if `pts.len() != n * dim`.
    pub fn log_prob_cloud(&self, pts: &[f64], n: usize) -> Geom3dResult<f64> {
        if pts.len() != n * self.dim {
            return Err(Geom3dError::DimensionMismatch {
                expected: n * self.dim,
                got: pts.len(),
            });
        }
        let mut total = 0.0_f64;
        for chunk in pts.chunks_exact(self.dim) {
            total += self.log_prob(chunk)?;
        }
        Ok(total)
    }

    /// Draw `n` points by sampling the Gaussian base and flowing each forward.
    ///
    /// **The flow is untrained**: the returned point set is a deformation of
    /// Gaussian noise carrying the *structure* of the model, **not** a learned
    /// shape. No realism / Chamfer claim is made or implied.
    ///
    /// Returns a row-major `[n × dim]` buffer.
    ///
    /// # Errors
    /// Propagates integration dimension errors (none under normal use).
    pub fn sample(&self, n: usize, rng: &mut LcgRng) -> Geom3dResult<Vec<f64>> {
        let dim = self.dim;
        let mut out = vec![0.0_f64; n * dim];
        let mut z = vec![0.0_f64; dim];
        for chunk in out.chunks_mut(dim) {
            fill_normal_f64(rng, &mut z, self.base_std);
            let (x1, _) = self.cnf.forward(&z)?;
            chunk.copy_from_slice(&x1);
        }
        Ok(out)
    }
}

// ─── Tests (mathematically-provable integrity core) ───────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(rng: &mut LcgRng, dim: usize) -> Vec<f64> {
        (0..dim).map(|_| uniform_pm1_f64(rng) * 1.5).collect()
    }

    /// log|det| of a 3×3 row-major matrix.
    fn log_abs_det_3x3(m: &[f64]) -> f64 {
        let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6]);
        det.abs().ln()
    }

    // 1. INVERTIBILITY: inverse(forward(x0)) ≈ x0 (true continuous bijection).
    #[test]
    fn invertibility_roundtrip() {
        let cfg = CnfConfig {
            dim: 3,
            n_steps: 40,
            ..CnfConfig::default()
        };
        let mut init = LcgRng::new(7);
        let cnf = ContinuousNormalizingFlow::new(cfg, &mut init).expect("cnf build");

        let mut rng = LcgRng::new(123);
        let mut max_err = 0.0_f64;
        let mut max_disp = 0.0_f64;
        for _ in 0..64 {
            let x0 = rand_vec(&mut rng, 3);
            let (x1, _) = cnf.forward(&x0).expect("forward");
            let back = cnf.inverse(&x1).expect("inverse");
            for ((a, fwd), b) in x0.iter().zip(&x1).zip(&back) {
                max_err = max_err.max((a - b).abs());
                max_disp = max_disp.max((a - fwd).abs());
            }
        }
        // The flow must genuinely transport points (non-vacuous invertibility):
        // a trivial identity `inverse`/`forward` would make the round-trip pass
        // for the wrong reason.
        assert!(
            max_disp > 1e-3,
            "forward must move points (max displacement {max_disp:e})"
        );
        // RK4 at 40 steps recovers the input far below the 1e-4 requirement.
        assert!(
            max_err < 1e-4,
            "invertibility max error {max_err:e} exceeds 1e-4 (n_steps=40)"
        );
    }

    // 2a. LOG-DET: accumulated delta_logp == -log|det J_F| (finite-diff Jacobian).
    #[test]
    fn logdet_matches_finite_difference() {
        let cfg = CnfConfig {
            dim: 3,
            n_steps: 60,
            init_scale: 0.4,
            ..CnfConfig::default()
        };
        let mut init = LcgRng::new(11);
        let cnf = ContinuousNormalizingFlow::new(cfg, &mut init).expect("cnf build");

        let mut rng = LcgRng::new(2024);
        let mut max_err = 0.0_f64;
        let eps = 1e-5;
        for _ in 0..16 {
            let x0 = rand_vec(&mut rng, 3);
            let (_, delta_logp) = cnf.forward(&x0).expect("forward");

            // Central-difference Jacobian J = dF/dx0 (column j = d x1 / d x0_j).
            let mut jac = vec![0.0_f64; 9];
            for j in 0..3 {
                let mut xp = x0.clone();
                let mut xm = x0.clone();
                xp[j] += eps;
                xm[j] -= eps;
                let (fp, _) = cnf.forward(&xp).expect("forward+");
                let (fm, _) = cnf.forward(&xm).expect("forward-");
                for i in 0..3 {
                    jac[i * 3 + j] = (fp[i] - fm[i]) / (2.0 * eps);
                }
            }
            let logdet_j = log_abs_det_3x3(&jac);
            // delta_logp = ∫ -tr dt = -log|det J_F|  ⇒  delta_logp + logdet_J ≈ 0.
            max_err = max_err.max((delta_logp + logdet_j).abs());
        }
        assert!(
            max_err < 1e-2,
            "delta_logp vs finite-difference log|det J| error {max_err:e} exceeds 1e-2"
        );
    }

    // 2b. ROUND-TRIP log-density conservation: fwd Δlogp + inv Δlogp ≈ 0.
    #[test]
    fn roundtrip_logp_conserved() {
        let cfg = CnfConfig {
            dim: 3,
            n_steps: 50,
            ..CnfConfig::default()
        };
        let mut init = LcgRng::new(5);
        let cnf = ContinuousNormalizingFlow::new(cfg, &mut init).expect("cnf build");

        let mut rng = LcgRng::new(77);
        let mut max_err = 0.0_f64;
        for _ in 0..32 {
            let x0 = rand_vec(&mut rng, 3);
            let (x1, dlp_fwd) = cnf.forward(&x0).expect("forward");
            let (_, dlp_inv) = cnf.inverse_with_logp(&x1).expect("inverse");
            max_err = max_err.max((dlp_fwd + dlp_inv).abs());
        }
        assert!(
            max_err < 1e-6,
            "forward+inverse delta_logp residual {max_err:e} exceeds 1e-6"
        );
    }

    // 2c. ANALYTIC EXACTNESS: for a linear field f(x,t)=A·x+c, the trace is the
    //     constant tr(A) everywhere, so delta_logp == -tr(A) to machine epsilon.
    #[test]
    fn linear_field_analytic_trace() {
        let cfg = CnfConfig {
            dim: 3,
            depth: 0, // exactly-linear velocity field
            n_steps: 8,
            init_scale: 0.5,
            ..CnfConfig::default()
        };
        let mut init = LcgRng::new(99);
        let cnf = ContinuousNormalizingFlow::new(cfg, &mut init).expect("cnf build");

        // The single layer maps [x0,x1,x2,t] → R^3 row-major; the diagonal of the
        // x-block (columns 0..3) is the trace of A = ∂f/∂x.
        let layer = &cnf.field.layers[0];
        let in_dim = layer.in_dim;
        let trace_a: f64 = (0..3).map(|i| layer.w[i * in_dim + i]).sum();

        let mut rng = LcgRng::new(314);
        let mut max_err = 0.0_f64;
        for _ in 0..16 {
            let x0 = rand_vec(&mut rng, 3);
            let (_, delta_logp) = cnf.forward(&x0).expect("forward");
            max_err = max_err.max((delta_logp - (-trace_a)).abs());
        }
        assert!(
            max_err < 1e-9,
            "linear-field delta_logp vs -tr(A) error {max_err:e} exceeds 1e-9"
        );
    }

    // 3a. DENSITY SANITY: log_prob finite (including for far-out inputs).
    #[test]
    fn log_prob_finite_everywhere() {
        let mut init = LcgRng::new(1);
        let model = PointFlowModel::new(CnfConfig::default(), 1.0, &mut init).expect("model");
        let probes = [
            vec![0.0, 0.0, 0.0],
            vec![1.0, -2.0, 3.0],
            vec![10.0, 10.0, 10.0],
            vec![-25.0, 7.0, -13.0],
        ];
        for p in &probes {
            let lp = model.log_prob(p).expect("log_prob");
            assert!(lp.is_finite(), "log_prob({p:?}) = {lp} is not finite");
        }
    }

    // 3b. DENSITY INTEGRATES TO ~1 over a coarse grid in low dimension.
    #[test]
    fn density_integrates_to_one_1d() {
        let cfg = CnfConfig {
            dim: 1,
            hidden: 16,
            depth: 2,
            n_steps: 30,
            init_scale: 0.3,
            trace_eps: 1e-5,
        };
        let mut init = LcgRng::new(42);
        let model = PointFlowModel::new(cfg, 1.0, &mut init).expect("model");

        // Midpoint Riemann sum of p(x) = exp(log_prob(x)) over [-L, L].
        let l = 10.0_f64;
        let dx = 0.02_f64;
        let steps = (2.0 * l / dx) as usize;
        let mut integral = 0.0_f64;
        for k in 0..steps {
            let x = -l + (k as f64 + 0.5) * dx;
            let lp = model.log_prob(&[x]).expect("log_prob");
            integral += lp.exp() * dx;
        }
        assert!(
            (integral - 1.0).abs() < 1e-2,
            "1-D CNF density integral {integral} not within 1e-2 of 1.0"
        );
    }

    // 3c. DETERMINISM: identical seed ⇒ identical log_prob.
    #[test]
    fn log_prob_deterministic_under_seed() {
        let x = [0.3_f64, -0.7, 1.1];
        let mut r1 = LcgRng::new(2025);
        let m1 = PointFlowModel::new(CnfConfig::default(), 1.0, &mut r1).expect("model1");
        let mut r2 = LcgRng::new(2025);
        let m2 = PointFlowModel::new(CnfConfig::default(), 1.0, &mut r2).expect("model2");
        let a = m1.log_prob(&x).expect("lp1");
        let b = m2.log_prob(&x).expect("lp2");
        assert_eq!(a, b, "same seed must give identical log_prob");
    }

    // 3d. SAMPLE: correct shape, finite, deterministic under fixed seed.
    #[test]
    fn sample_shape_finite_deterministic() {
        let mut init = LcgRng::new(8);
        let model = PointFlowModel::new(CnfConfig::default(), 1.0, &mut init).expect("model");

        let mut s1 = LcgRng::new(555);
        let pts1 = model.sample(20, &mut s1).expect("sample1");
        assert_eq!(pts1.len(), 20 * 3, "sample shape must be n*dim");
        assert!(pts1.iter().all(|v| v.is_finite()), "samples must be finite");

        let mut s2 = LcgRng::new(555);
        let pts2 = model.sample(20, &mut s2).expect("sample2");
        assert_eq!(pts1, pts2, "same seed must give identical samples");
    }

    // CNF forward output shape + dimension-mismatch error path.
    #[test]
    fn forward_shape_and_dim_error() {
        let mut init = LcgRng::new(3);
        let cnf = ContinuousNormalizingFlow::new(CnfConfig::default(), &mut init).expect("cnf");
        let (x1, _) = cnf.forward(&[0.1, 0.2, 0.3]).expect("forward");
        assert_eq!(x1.len(), 3, "forward output dim");
        assert_eq!(
            cnf.forward(&[0.1, 0.2]),
            Err(Geom3dError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        );
        assert_eq!(
            cnf.inverse(&[0.1, 0.2, 0.3, 0.4]),
            Err(Geom3dError::DimensionMismatch {
                expected: 3,
                got: 4
            })
        );
    }

    // Two flows from the same seed are bit-identical.
    #[test]
    fn cnf_determinism_same_seed() {
        let mut r1 = LcgRng::new(404);
        let c1 = ContinuousNormalizingFlow::new(CnfConfig::default(), &mut r1).expect("c1");
        let mut r2 = LcgRng::new(404);
        let c2 = ContinuousNormalizingFlow::new(CnfConfig::default(), &mut r2).expect("c2");
        let x0 = [0.5_f64, -0.25, 0.9];
        let (a, da) = c1.forward(&x0).expect("f1");
        let (b, db) = c2.forward(&x0).expect("f2");
        assert_eq!(a, b);
        assert_eq!(da, db);
    }

    // log_prob_cloud sums the per-point log-densities.
    #[test]
    fn log_prob_cloud_sums_points() {
        let mut init = LcgRng::new(6);
        let model = PointFlowModel::new(CnfConfig::default(), 1.0, &mut init).expect("model");
        let pts = vec![0.1_f64, 0.2, 0.3, -0.4, 0.5, -0.6];
        let cloud = model.log_prob_cloud(&pts, 2).expect("cloud");
        let p0 = model.log_prob(&pts[0..3]).expect("p0");
        let p1 = model.log_prob(&pts[3..6]).expect("p1");
        assert!(
            (cloud - (p0 + p1)).abs() < 1e-12,
            "cloud must equal point sum"
        );
        assert_eq!(
            model.log_prob_cloud(&pts, 3),
            Err(Geom3dError::DimensionMismatch {
                expected: 9,
                got: 6
            })
        );
    }

    // Invalid configurations are rejected (no silent fallback).
    #[test]
    fn invalid_config_rejected() {
        let mut rng = LcgRng::new(1);
        assert!(
            ContinuousNormalizingFlow::new(
                CnfConfig {
                    dim: 0,
                    ..CnfConfig::default()
                },
                &mut rng
            )
            .is_err()
        );
        assert!(
            ContinuousNormalizingFlow::new(
                CnfConfig {
                    n_steps: 0,
                    ..CnfConfig::default()
                },
                &mut rng
            )
            .is_err()
        );
        assert!(PointFlowModel::new(CnfConfig::default(), 0.0, &mut rng).is_err());
        assert!(PointFlowModel::new(CnfConfig::default(), -1.0, &mut rng).is_err());
    }
}
