#![allow(clippy::needless_range_loop)]
//! Echo State Network (Jaeger 2001) — rate-coded leaky integrator reservoir.
//!
//! ## State update
//!
//! ```text
//! x_t = (1 − α) · x_{t−1} + α · tanh(W_in · u_t + W_rec · x_{t−1} + b)
//! ```
//!
//! ## Readout
//!
//! ```text
//! y = W_out · concat(x_t, u_t)   [augmented state]
//! ```
//!
//! Readout weights are fitted offline by ridge regression using the
//! Cholesky-Banachiewicz decomposition.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;
use crate::reservoir::lsm::power_iteration_spectral_radius;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration knobs for an Echo State Network.
#[derive(Debug, Clone)]
pub struct EsnConfig {
    /// Number of reservoir units *N*.
    pub n_reservoir: usize,
    /// Input dimensionality *U*.
    pub n_input: usize,
    /// Output (readout) dimensionality *Y*.
    pub n_output: usize,
    /// Target spectral radius ρ(W_rec), e.g. 0.9. Must be > 0.
    pub spectral_radius: f32,
    /// Leak rate α ∈ (0, 1].
    pub leak_rate: f32,
    /// Input weight scale: W_in entries are uniform on [−input_scale, +input_scale].
    pub input_scale: f32,
    /// Fraction of W_rec entries that are non-zero, ∈ (0, 1].
    pub density: f32,
    /// L2 regularisation coefficient for ridge regression.
    pub ridge_lambda: f32,
}

impl Default for EsnConfig {
    fn default() -> Self {
        Self {
            n_reservoir: 100,
            n_input: 1,
            n_output: 1,
            spectral_radius: 0.9,
            leak_rate: 0.3,
            input_scale: 1.0,
            density: 0.1,
            ridge_lambda: 1e-4,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

/// Per-timestep reservoir state vector.
#[derive(Debug, Clone)]
pub struct EsnState {
    /// Reservoir activation `x ∈ ℝ^N`.
    pub x: Vec<f32>,
}

impl EsnState {
    /// Create a zero-initialised state of length `n`.
    #[must_use]
    pub fn zeros(n: usize) -> Self {
        Self {
            x: vec![0.0_f32; n],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ESN struct
// ─────────────────────────────────────────────────────────────────────────────

/// Echo State Network with offline ridge-regression readout.
#[derive(Debug, Clone)]
pub struct Esn {
    /// Validated configuration.
    pub cfg: EsnConfig,
    /// Input weight matrix W_in, row-major `[n_reservoir × n_input]`.
    pub w_in: Vec<f32>,
    /// Recurrent weight matrix W_rec, row-major `[n_reservoir × n_reservoir]`.
    pub w_rec: Vec<f32>,
    /// Bias vector, always zeros in this implementation, `[n_reservoir]`.
    pub bias: Vec<f32>,
    /// Readout weights W_out, row-major `[n_output × (n_reservoir + n_input)]`.
    /// `None` before `fit_readout` is called.
    pub w_out: Option<Vec<f32>>,
}

impl Esn {
    // ─── construction ────────────────────────────────────────────────────────

    /// Build a new ESN, randomly initialising W_in and W_rec from `rng`.
    ///
    /// # Errors
    ///
    /// Returns `SnnError::BadDim` if any dimension is zero, `SnnError::OutOfRange`
    /// if `leak_rate`, `density`, or `spectral_radius` are out of range.
    pub fn new(cfg: EsnConfig, rng: &mut LcgRng) -> SnnResult<Self> {
        // ── validate ─────────────────────────────────────────────────────────
        if cfg.n_reservoir == 0 {
            return Err(SnnError::BadDim { got: 0 });
        }
        if cfg.n_input == 0 {
            return Err(SnnError::BadDim { got: 0 });
        }
        if cfg.n_output == 0 {
            return Err(SnnError::BadDim { got: 0 });
        }
        if cfg.leak_rate <= 0.0 || cfg.leak_rate > 1.0 || cfg.leak_rate.is_nan() {
            return Err(SnnError::OutOfRange {
                name: "leak_rate".to_string(),
                val: cfg.leak_rate,
            });
        }
        if cfg.density <= 0.0 || cfg.density > 1.0 || cfg.density.is_nan() {
            return Err(SnnError::OutOfRange {
                name: "density".to_string(),
                val: cfg.density,
            });
        }
        if cfg.spectral_radius <= 0.0 || cfg.spectral_radius.is_nan() {
            return Err(SnnError::OutOfRange {
                name: "spectral_radius".to_string(),
                val: cfg.spectral_radius,
            });
        }

        let n = cfg.n_reservoir;
        let u = cfg.n_input;

        // ── W_in: uniform on [−input_scale, +input_scale] ───────────────────
        let mut w_in = vec![0.0_f32; n * u];
        for v in &mut w_in {
            *v = cfg.input_scale * (2.0 * rng.next_f32() - 1.0);
        }

        // ── W_rec: sparse normal, rescaled to target spectral radius ─────────
        let mut w_rec = vec![0.0_f32; n * n];
        for idx in 0..(n * n) {
            if rng.next_f32() < cfg.density {
                let (a, _b) = rng.next_normal_pair();
                w_rec[idx] = a;
            }
        }
        let actual_sr = power_iteration_spectral_radius(&w_rec, n, 100);
        if actual_sr > 1e-6 {
            let scale = cfg.spectral_radius / actual_sr;
            for v in &mut w_rec {
                *v *= scale;
            }
        }

        // ── bias: zeros ──────────────────────────────────────────────────────
        let bias = vec![0.0_f32; n];

        Ok(Self {
            cfg,
            w_in,
            w_rec,
            bias,
            w_out: None,
        })
    }

    // ─── dynamics ────────────────────────────────────────────────────────────

    /// Advance the reservoir by one timestep.
    ///
    /// `u` must have length `n_input`.  The returned state has `x.len() == n_reservoir`.
    ///
    /// Equation:
    /// ```text
    /// pre[j] = Σ_i W_in[j·U+i]·u[i] + Σ_k W_rec[j·N+k]·x[k] + bias[j]
    /// x_new[j] = (1−α)·x[j] + α·tanh(pre[j])
    /// ```
    pub fn step(&self, state: &EsnState, u: &[f32]) -> SnnResult<EsnState> {
        let n = self.cfg.n_reservoir;
        let ui = self.cfg.n_input;
        let alpha = self.cfg.leak_rate;

        if u.len() != ui {
            return Err(SnnError::BadShape {
                expected: ui,
                got: u.len(),
            });
        }
        if state.x.len() != n {
            return Err(SnnError::BadShape {
                expected: n,
                got: state.x.len(),
            });
        }

        let mut x_new = vec![0.0_f32; n];
        for j in 0..n {
            // Input contribution.
            let mut pre = self.bias[j];
            for i in 0..ui {
                pre += self.w_in[j * ui + i] * u[i];
            }
            // Recurrent contribution.
            for k in 0..n {
                pre += self.w_rec[j * n + k] * state.x[k];
            }
            x_new[j] = (1.0 - alpha) * state.x[j] + alpha * pre.tanh();
        }
        Ok(EsnState { x: x_new })
    }

    // ─── state collection ────────────────────────────────────────────────────

    /// Run the reservoir for `n_steps` timesteps starting from `init`, discard
    /// the first `washout` steps, and collect the augmented state
    /// `concat(x_t, u_t)` for the remaining steps.
    ///
    /// `inputs` must be row-major `[n_steps × n_input]`.
    ///
    /// Returns row-major `[(n_steps − washout) × (n_reservoir + n_input)]`.
    ///
    /// # Errors
    ///
    /// - `SnnError::BadTimesteps` if `n_steps == 0`.
    /// - `SnnError::BadShape` if `inputs.len() != n_steps * n_input`.
    /// - `SnnError::IncompatibleLength` if `washout >= n_steps`.
    pub fn collect_states(
        &self,
        inputs: &[f32],
        n_steps: usize,
        washout: usize,
        init: &EsnState,
    ) -> SnnResult<Vec<f32>> {
        if n_steps == 0 {
            return Err(SnnError::BadTimesteps { got: 0 });
        }
        let ui = self.cfg.n_input;
        let n = self.cfg.n_reservoir;
        let expected_len = n_steps * ui;
        if inputs.len() != expected_len {
            return Err(SnnError::BadShape {
                expected: expected_len,
                got: inputs.len(),
            });
        }
        if washout >= n_steps {
            return Err(SnnError::IncompatibleLength {
                a: washout,
                b: n_steps,
            });
        }

        let aug_dim = n + ui;
        let collect_steps = n_steps - washout;
        let mut out = Vec::with_capacity(collect_steps * aug_dim);

        let mut state = init.clone();
        for t in 0..n_steps {
            let u = &inputs[t * ui..(t + 1) * ui];
            state = self.step(&state, u)?;
            if t >= washout {
                out.extend_from_slice(&state.x);
                out.extend_from_slice(u);
            }
        }
        Ok(out)
    }

    // ─── readout training ────────────────────────────────────────────────────

    /// Fit the linear readout W_out via ridge regression.
    ///
    /// `states` is row-major `[n_train × (n_reservoir + n_input)]` (augmented).
    /// `targets` is row-major `[n_train × n_output]`.
    ///
    /// Stores the fitted weights as `w_out = Some(W)` where
    /// `W` is `[n_output × (n_reservoir + n_input)]`.
    ///
    /// # Errors
    ///
    /// Propagates errors from `ridge_regression` (e.g. non-positive-definite matrix).
    pub fn fit_readout(
        &mut self,
        states: &[f32],
        targets: &[f32],
        n_train: usize,
    ) -> SnnResult<()> {
        let d = self.cfg.n_reservoir + self.cfg.n_input;
        let k = self.cfg.n_output;
        let w = ridge_regression(states, targets, n_train, d, k, self.cfg.ridge_lambda)?;
        self.w_out = Some(w);
        Ok(())
    }

    // ─── inference ───────────────────────────────────────────────────────────

    /// Compute the readout `y = W_out · aug_state`.
    ///
    /// `aug_state` must have length `n_reservoir + n_input`.
    ///
    /// # Errors
    ///
    /// Returns `SnnError::Internal` if the readout has not been fitted yet.
    pub fn predict(&self, aug_state: &[f32]) -> SnnResult<Vec<f32>> {
        let w_out = self.w_out.as_ref().ok_or_else(|| SnnError::Internal {
            msg: "readout weights not fitted; call fit_readout first".to_string(),
        })?;
        let d = self.cfg.n_reservoir + self.cfg.n_input;
        let k = self.cfg.n_output;
        if aug_state.len() != d {
            return Err(SnnError::BadShape {
                expected: d,
                got: aug_state.len(),
            });
        }
        let mut y = vec![0.0_f32; k];
        for out_k in 0..k {
            let mut acc = 0.0_f32;
            for j in 0..d {
                acc += w_out[out_k * d + j] * aug_state[j];
            }
            y[out_k] = acc;
        }
        Ok(y)
    }

    // ─── utilities ───────────────────────────────────────────────────────────

    /// Estimate the current spectral radius of W_rec via power iteration.
    #[must_use]
    pub fn spectral_radius(&self) -> f32 {
        power_iteration_spectral_radius(&self.w_rec, self.cfg.n_reservoir, 100)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ridge regression
// ─────────────────────────────────────────────────────────────────────────────

/// Offline ridge regression via Cholesky-Banachiewicz decomposition.
///
/// Solves `(X^T X + λI) W^T = X^T Y` for `W`.
///
/// # Arguments
///
/// * `x`      — row-major design matrix `[n × d]`.
/// * `y`      — row-major target matrix `[n × k]`.
/// * `n`      — number of training samples.
/// * `d`      — feature dimension (= n_reservoir + n_input for ESN).
/// * `k`      — number of outputs.
/// * `lambda` — L2 regularisation coefficient.
///
/// # Returns
///
/// Weight matrix `W` row-major `[k × d]`.
///
/// # Errors
///
/// `SnnError::Internal` if the Gram matrix `A` is not positive definite
/// (Cholesky diagonal goes negative).
pub fn ridge_regression(
    x: &[f32],
    y: &[f32],
    n: usize,
    d: usize,
    k: usize,
    lambda: f32,
) -> SnnResult<Vec<f32>> {
    // ── form A = X^T X + λI  (d × d, lower-triangular half needed) ──────────
    let mut a = vec![0.0_f32; d * d];
    for t in 0..n {
        for i in 0..d {
            let xi = x[t * d + i];
            for j in 0..=i {
                a[i * d + j] += xi * x[t * d + j];
            }
        }
    }
    // Mirror to upper triangle and add ridge.
    for i in 0..d {
        for j in (i + 1)..d {
            a[i * d + j] = a[j * d + i];
        }
        a[i * d + i] += lambda;
    }

    // ── form B = X^T Y  (d × k) ─────────────────────────────────────────────
    let mut b = vec![0.0_f32; d * k];
    for t in 0..n {
        for i in 0..d {
            let xi = x[t * d + i];
            for c in 0..k {
                b[i * k + c] += xi * y[t * k + c];
            }
        }
    }

    // ── Cholesky-Banachiewicz: L s.t. A = L L^T ─────────────────────────────
    // L is stored in the lower-triangular part of a d×d buffer.
    let mut l = vec![0.0_f32; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i * d + j];
            for p in 0..j {
                s -= l[i * d + p] * l[j * d + p];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(SnnError::Internal {
                        msg: format!(
                            "Cholesky failed at diagonal ({i},{i}): value {s} (matrix not PD)"
                        ),
                    });
                }
                l[i * d + j] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }

    // ── forward substitution: L Z = B  →  Z (d × k) ─────────────────────────
    let mut z = vec![0.0_f32; d * k];
    for i in 0..d {
        for c in 0..k {
            let mut val = b[i * k + c];
            for p in 0..i {
                val -= l[i * d + p] * z[p * k + c];
            }
            z[i * k + c] = val / l[i * d + i];
        }
    }

    // ── back substitution: L^T W^T = Z  →  W^T (d × k), then transpose ─────
    // W^T[j,c] s.t. Σ_{p≥j} L[p,j]·W^T[p,c] = Z[j,c]
    let mut wt = vec![0.0_f32; d * k]; // W^T stored as [d × k]
    for j in (0..d).rev() {
        for c in 0..k {
            let mut val = z[j * k + c];
            for p in (j + 1)..d {
                val -= l[p * d + j] * wt[p * k + c];
            }
            wt[j * k + c] = val / l[j * d + j];
        }
    }

    // W is [k × d]; W[out_k, feature_j] = W^T[feature_j, out_k].
    let mut w = vec![0.0_f32; k * d];
    for j in 0..d {
        for c in 0..k {
            w[c * d + j] = wt[j * k + c];
        }
    }
    Ok(w)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ── EsnConfig ────────────────────────────────────────────────────────────

    #[test]
    fn default_config_fields() {
        let cfg = EsnConfig::default();
        assert_eq!(cfg.n_reservoir, 100);
        assert_eq!(cfg.n_input, 1);
        assert_eq!(cfg.n_output, 1);
        assert!((cfg.spectral_radius - 0.9).abs() < 1e-6);
        assert!((cfg.leak_rate - 0.3).abs() < 1e-6);
        assert!((cfg.input_scale - 1.0).abs() < 1e-6);
        assert!((cfg.density - 0.1).abs() < 1e-6);
        assert!((cfg.ridge_lambda - 1e-4).abs() < 1e-9);
    }

    // ── EsnState ─────────────────────────────────────────────────────────────

    #[test]
    fn esn_state_zeros_shape() {
        let s = EsnState::zeros(50);
        assert_eq!(s.x.len(), 50);
        assert!(s.x.iter().all(|&v| v == 0.0));
    }

    // ── Esn::new ─────────────────────────────────────────────────────────────

    #[test]
    fn esn_new_matrix_shapes() {
        let cfg = EsnConfig {
            n_reservoir: 20,
            n_input: 3,
            n_output: 2,
            ..EsnConfig::default()
        };
        let n = cfg.n_reservoir;
        let u = cfg.n_input;
        let mut rng = make_rng();
        let esn = Esn::new(cfg, &mut rng).expect("new ok");
        assert_eq!(esn.w_in.len(), n * u);
        assert_eq!(esn.w_rec.len(), n * n);
        assert_eq!(esn.bias.len(), n);
        assert!(esn.bias.iter().all(|&v| v == 0.0));
        assert!(esn.w_out.is_none());
    }

    #[test]
    fn esn_new_zero_reservoir_errors() {
        let cfg = EsnConfig {
            n_reservoir: 0,
            ..EsnConfig::default()
        };
        let mut rng = make_rng();
        assert!(matches!(
            Esn::new(cfg, &mut rng),
            Err(SnnError::BadDim { got: 0 })
        ));
    }

    #[test]
    fn esn_new_zero_input_errors() {
        let cfg = EsnConfig {
            n_input: 0,
            ..EsnConfig::default()
        };
        let mut rng = make_rng();
        assert!(matches!(
            Esn::new(cfg, &mut rng),
            Err(SnnError::BadDim { got: 0 })
        ));
    }

    #[test]
    fn esn_new_zero_output_errors() {
        let cfg = EsnConfig {
            n_output: 0,
            ..EsnConfig::default()
        };
        let mut rng = make_rng();
        assert!(matches!(
            Esn::new(cfg, &mut rng),
            Err(SnnError::BadDim { got: 0 })
        ));
    }

    #[test]
    fn esn_new_leak_rate_zero_errors() {
        let cfg = EsnConfig {
            leak_rate: 0.0,
            ..EsnConfig::default()
        };
        let mut rng = make_rng();
        assert!(matches!(
            Esn::new(cfg, &mut rng),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn esn_new_density_zero_errors() {
        let cfg = EsnConfig {
            density: 0.0,
            ..EsnConfig::default()
        };
        let mut rng = make_rng();
        assert!(matches!(
            Esn::new(cfg, &mut rng),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    // ── step ─────────────────────────────────────────────────────────────────

    #[test]
    fn step_output_length() {
        let cfg = EsnConfig {
            n_reservoir: 15,
            n_input: 4,
            n_output: 1,
            ..EsnConfig::default()
        };
        let n = cfg.n_reservoir;
        let u = cfg.n_input;
        let mut rng = make_rng();
        let esn = Esn::new(cfg, &mut rng).expect("new ok");
        let state = EsnState::zeros(n);
        let input = vec![0.5_f32; u];
        let next = esn.step(&state, &input).expect("step ok");
        assert_eq!(next.x.len(), n);
    }

    #[test]
    fn step_state_sum_finite() {
        let cfg = EsnConfig {
            n_reservoir: 30,
            n_input: 2,
            n_output: 1,
            ..EsnConfig::default()
        };
        let n = cfg.n_reservoir;
        let u = cfg.n_input;
        let mut rng = make_rng();
        let esn = Esn::new(cfg, &mut rng).expect("new ok");
        let mut state = EsnState::zeros(n);
        for _ in 0..10 {
            state = esn.step(&state, &vec![1.0_f32; u]).expect("step ok");
        }
        let sum: f32 = state.x.iter().sum();
        assert!(sum.is_finite(), "state sum should be finite, got {sum}");
    }

    #[test]
    fn step_leak_rate_one_zero_input_squashes() {
        // With leak_rate=1.0, u=0, W_rec=0, bias=0 → x_new = tanh(W_in·0) = 0.
        let n = 10_usize;
        let u = 2_usize;
        let mut rng = make_rng();
        let cfg = EsnConfig {
            n_reservoir: n,
            n_input: u,
            n_output: 1,
            leak_rate: 1.0,
            density: 0.5,
            ..EsnConfig::default()
        };
        let mut esn = Esn::new(cfg, &mut rng).expect("new ok");
        // Force W_rec = 0 so input determines all.
        for v in &mut esn.w_rec {
            *v = 0.0;
        }
        let state = EsnState {
            x: vec![1.0_f32; n],
        };
        let u_vec = vec![0.0_f32; u]; // zero input → tanh(0) = 0
        let next = esn.step(&state, &u_vec).expect("step ok");
        for (i, &v) in next.x.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "x[{i}] = {v}, expected ≈ 0 with zero input and zero recurrent"
            );
        }
    }

    // ── spectral_radius ───────────────────────────────────────────────────────

    #[test]
    fn spectral_radius_close_to_target() {
        let cfg = EsnConfig {
            n_reservoir: 60,
            n_input: 2,
            n_output: 1,
            spectral_radius: 0.9,
            density: 0.15,
            ..EsnConfig::default()
        };
        let mut rng = make_rng();
        let esn = Esn::new(cfg, &mut rng).expect("new ok");
        let rho = esn.spectral_radius();
        assert!(
            (rho - 0.9).abs() < 0.15,
            "spectral radius {rho} not within ±0.15 of target 0.9"
        );
    }

    // ── collect_states ────────────────────────────────────────────────────────

    #[test]
    fn collect_states_output_shape() {
        let n = 20_usize;
        let u = 3_usize;
        let n_steps = 50_usize;
        let washout = 10_usize;
        let cfg = EsnConfig {
            n_reservoir: n,
            n_input: u,
            n_output: 1,
            ..EsnConfig::default()
        };
        let mut rng = make_rng();
        let esn = Esn::new(cfg, &mut rng).expect("new ok");
        let inputs = vec![0.1_f32; n_steps * u];
        let init = EsnState::zeros(n);
        let out = esn
            .collect_states(&inputs, n_steps, washout, &init)
            .expect("collect ok");
        let expected_rows = n_steps - washout;
        let aug_dim = n + u;
        assert_eq!(
            out.len(),
            expected_rows * aug_dim,
            "expected {} got {}",
            expected_rows * aug_dim,
            out.len()
        );
    }

    #[test]
    fn collect_states_zero_washout() {
        let n = 10_usize;
        let u = 2_usize;
        let n_steps = 8_usize;
        let cfg = EsnConfig {
            n_reservoir: n,
            n_input: u,
            n_output: 1,
            ..EsnConfig::default()
        };
        let mut rng = make_rng();
        let esn = Esn::new(cfg, &mut rng).expect("new ok");
        let inputs = vec![0.5_f32; n_steps * u];
        let init = EsnState::zeros(n);
        let out = esn
            .collect_states(&inputs, n_steps, 0, &init)
            .expect("collect ok");
        assert_eq!(out.len(), n_steps * (n + u));
    }

    #[test]
    fn collect_states_washout_n_steps_minus_1_gives_one_row() {
        let n = 10_usize;
        let u = 2_usize;
        let n_steps = 8_usize;
        let washout = n_steps - 1;
        let cfg = EsnConfig {
            n_reservoir: n,
            n_input: u,
            n_output: 1,
            ..EsnConfig::default()
        };
        let mut rng = make_rng();
        let esn = Esn::new(cfg, &mut rng).expect("new ok");
        let inputs = vec![0.3_f32; n_steps * u];
        let init = EsnState::zeros(n);
        let out = esn
            .collect_states(&inputs, n_steps, washout, &init)
            .expect("collect ok");
        assert_eq!(out.len(), n + u, "should have exactly one row");
    }

    #[test]
    fn collect_states_washout_equal_n_steps_errors() {
        let n = 10_usize;
        let u = 2_usize;
        let n_steps = 5_usize;
        let cfg = EsnConfig {
            n_reservoir: n,
            n_input: u,
            n_output: 1,
            ..EsnConfig::default()
        };
        let mut rng = make_rng();
        let esn = Esn::new(cfg, &mut rng).expect("new ok");
        let inputs = vec![0.0_f32; n_steps * u];
        let init = EsnState::zeros(n);
        assert!(matches!(
            esn.collect_states(&inputs, n_steps, n_steps, &init),
            Err(SnnError::IncompatibleLength { .. })
        ));
    }

    // ── fit_readout / predict ────────────────────────────────────────────────

    #[test]
    fn fit_readout_sets_w_out() {
        let n = 10_usize;
        let u = 2_usize;
        let k = 3_usize;
        let n_train = 20_usize;
        let cfg = EsnConfig {
            n_reservoir: n,
            n_input: u,
            n_output: k,
            ..EsnConfig::default()
        };
        let mut rng = make_rng();
        let mut esn = Esn::new(cfg, &mut rng).expect("new ok");
        let d = n + u;
        let states = vec![0.1_f32; n_train * d];
        let targets = vec![0.5_f32; n_train * k];
        esn.fit_readout(&states, &targets, n_train).expect("fit ok");
        assert!(esn.w_out.is_some());
        assert_eq!(
            esn.w_out.as_ref().expect("as_ref should succeed").len(),
            k * d
        );
    }

    #[test]
    fn predict_output_length() {
        let n = 10_usize;
        let u = 2_usize;
        let k = 3_usize;
        let n_train = 20_usize;
        let cfg = EsnConfig {
            n_reservoir: n,
            n_input: u,
            n_output: k,
            ..EsnConfig::default()
        };
        let mut rng = make_rng();
        let mut esn = Esn::new(cfg, &mut rng).expect("new ok");
        let d = n + u;
        let states = vec![0.1_f32; n_train * d];
        let targets = vec![0.5_f32; n_train * k];
        esn.fit_readout(&states, &targets, n_train).expect("fit ok");
        let aug = vec![0.2_f32; d];
        let pred = esn.predict(&aug).expect("predict ok");
        assert_eq!(pred.len(), k);
    }

    #[test]
    fn predict_before_fit_errors() {
        let cfg = EsnConfig {
            n_reservoir: 10,
            n_input: 2,
            n_output: 3,
            ..EsnConfig::default()
        };
        let mut rng = make_rng();
        let esn = Esn::new(cfg, &mut rng).expect("new ok");
        let aug = vec![0.0_f32; 12];
        assert!(matches!(esn.predict(&aug), Err(SnnError::Internal { .. })));
    }

    // ── ridge_regression ─────────────────────────────────────────────────────

    #[test]
    fn ridge_regression_output_shape() {
        // n=10, d=5, k=3 → W shape [k×d] = [3×5]
        let n = 10_usize;
        let d = 5_usize;
        let k = 3_usize;
        let mut rng = make_rng();
        let x: Vec<f32> = (0..n * d).map(|_| rng.next_f32()).collect();
        let y: Vec<f32> = (0..n * k).map(|_| rng.next_f32()).collect();
        let w = ridge_regression(&x, &y, n, d, k, 1e-3).expect("ridge ok");
        assert_eq!(w.len(), k * d);
    }

    #[test]
    fn ridge_regression_large_lambda_gives_small_weights() {
        // With very large λ the solution W ≈ 0.
        let n = 8_usize;
        let d = 4_usize;
        let k = 2_usize;
        // X = identity-ish
        let mut x = vec![0.0_f32; n * d];
        for i in 0..d {
            x[i * d + i] = 1.0;
        }
        // Remaining rows zero.
        let y: Vec<f32> = vec![1.0_f32; n * k];
        let w = ridge_regression(&x, &y, n, d, k, 1e8).expect("ridge large λ ok");
        for (i, &v) in w.iter().enumerate() {
            assert!(
                v.abs() < 1e-3,
                "w[{i}] = {v} should be near zero with huge lambda"
            );
        }
    }

    #[test]
    fn ridge_regression_exact_constant() {
        // n=4, d=1, k=1, X=[[1],[1],[1],[1]], Y=[[2],[2],[2],[2]], λ→0 → W≈[2].
        let n = 4_usize;
        let d = 1_usize;
        let k = 1_usize;
        let x = vec![1.0_f32; n * d];
        let y = vec![2.0_f32; n * k];
        let w = ridge_regression(&x, &y, n, d, k, 1e-10).expect("ridge exact ok");
        assert_eq!(w.len(), 1);
        assert!((w[0] - 2.0).abs() < 1e-4, "expected W≈[2.0], got {}", w[0]);
    }
}
