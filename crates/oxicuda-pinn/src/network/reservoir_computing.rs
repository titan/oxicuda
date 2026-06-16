//! Reservoir computing — Echo State Network (ESN) for chaotic dynamical systems.
//!
//! Jaeger (2001) "The 'echo state' approach to analysing and training recurrent
//! neural networks", GMD Report 148; Jaeger & Haas (2004) "Harnessing
//! Nonlinearity: Predicting Chaotic Systems and Saving Energy in Wireless
//! Communication", Science 304(5667), pp. 78-80; Lukoševičius (2012) "A Practical
//! Guide to Applying Echo State Networks", in *Neural Networks: Tricks of the
//! Trade*, LNCS 7700.
//!
//! An ESN drives a large, fixed, randomly-connected **reservoir** with the input
//! signal and reads the high-dimensional reservoir state out with a *single*
//! trained linear layer. Only the readout is trained — by ridge regression
//! (Tikhonov-regularised least squares), a convex problem with a closed-form
//! solution — which makes ESNs extremely cheap to fit compared to backprop-through
//! -time RNNs, while still modelling rich temporal / chaotic dynamics.
//!
//! ## State update (leaky-integrator reservoir)
//! ```text
//! x(t+1) = (1 − α) · x(t) + α · tanh( W_in · u(t) + W_res · x(t) )
//! y(t)   = W_out · [x(t); 1]
//! ```
//! where `α ∈ (0, 1]` is the leak rate. The recurrent matrix `W_res` is rescaled
//! so its spectral radius equals a target `ρ < 1` (the **echo state property**:
//! the reservoir state asymptotically forgets its initial condition, becoming a
//! fading-memory function of the input history).
//!
//! This implementation is self-contained within the PINN crate (it does not share
//! code with the ESNs in sibling OxiCUDA crates): a deterministic [`LcgRng`]
//! seeds the reservoir, the spectral radius is estimated by normalised power
//! iteration (Gelfand's formula `ρ(A) = limₖ ‖Aᵏv‖^{1/k}`), and the ridge readout
//! is solved by Gaussian elimination with partial pivoting.

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;

/// Default number of power-iteration steps used to estimate the spectral radius.
const SPECTRAL_RADIUS_ITERS: usize = 120;

/// True uniform sample in `[0, 1)`.
///
/// This crate's [`LcgRng::next_u32`] returns the top 31 bits of the LCG state, so
/// its range is `[0, 2³¹)` and the library `next_f32` only spans `[0, 0.5)`.
/// Dividing by `2³¹` recovers a correctly-spread uniform on `[0, 1)`.
#[inline]
fn unit_uniform(rng: &mut LcgRng) -> f32 {
    (rng.next_u32() as f32) / 4_294_967_296.0_f32
}

/// Symmetric uniform sample in `[-1, 1)`.
#[inline]
fn signed_uniform(rng: &mut LcgRng) -> f32 {
    2.0 * unit_uniform(rng) - 1.0
}

/// Euclidean (ℓ²) norm of a slice.
#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Row-major matrix–vector product `A·v` for an `n×n` matrix.
fn matvec(matrix: &[f32], v: &[f32], n: usize) -> Vec<f32> {
    matrix
        .chunks_exact(n)
        .map(|row| row.iter().zip(v.iter()).map(|(&a, &x)| a * x).sum())
        .collect()
}

/// Estimate the spectral radius `ρ(A)` (largest eigenvalue magnitude) of an
/// `n×n` row-major matrix.
///
/// Uses normalised power iteration together with Gelfand's formula
/// `ρ(A) = limₖ ‖Aᵏ v‖^{1/k}`: at each step the per-step growth factor
/// `g = ‖A v‖ / ‖v‖` (with `v` re-normalised) is recorded, and the geometric mean
/// of the growth factors after a short warm-up is returned. For normal matrices
/// the growth factor equals `ρ(A)` every step (including complex-conjugate and
/// negative dominant eigenvalues); for general matrices the geometric mean
/// converges to `ρ(A)`.
///
/// # Errors
/// - [`PinnError::EmptyInput`] if `n == 0`.
/// - [`PinnError::DimensionMismatch`] if `matrix.len() != n * n`.
/// - [`PinnError::NanEncountered`] if the estimate is not finite.
pub fn spectral_radius(matrix: &[f32], n: usize, iters: usize) -> PinnResult<f32> {
    if n == 0 {
        return Err(PinnError::EmptyInput);
    }
    if matrix.len() != n * n {
        return Err(PinnError::DimensionMismatch {
            expected: n * n,
            got: matrix.len(),
        });
    }
    // Deterministic, non-uniform start vector to avoid accidental orthogonality
    // to the dominant eigenvector.
    let mut v: Vec<f32> = (0..n).map(|i| 1.0 + 0.1 * i as f32).collect();
    let norm0 = l2_norm(&v);
    if norm0 == 0.0 {
        return Ok(0.0);
    }
    for x in &mut v {
        *x /= norm0;
    }

    let warmup = (iters / 4).max(1);
    let mut log_growth = 0.0_f32;
    let mut counted = 0_usize;
    for k in 0..iters {
        let av = matvec(matrix, &v, n);
        let g = l2_norm(&av);
        if g <= 0.0 {
            // A maps v to (near) zero: nilpotent-like, dominant magnitude is 0.
            return Ok(0.0);
        }
        if k >= warmup {
            log_growth += g.ln();
            counted += 1;
        }
        for (vi, &avi) in v.iter_mut().zip(av.iter()) {
            *vi = avi / g;
        }
    }
    let rho = if counted > 0 {
        (log_growth / counted as f32).exp()
    } else {
        0.0
    };
    if !rho.is_finite() {
        return Err(PinnError::NanEncountered {
            location: "spectral_radius",
        });
    }
    Ok(rho)
}

/// Solve the `n×n` linear system `A·X = B` with `m` right-hand sides.
///
/// `a` is row-major `n×n` (consumed / overwritten in place); `rhs` is row-major
/// `n×m`. Returns the solution `X` as row-major `n×m`. Uses Gaussian elimination
/// with partial pivoting.
///
/// # Errors
/// - [`PinnError::SolverDivergence`] if `A` is (numerically) singular.
/// - [`PinnError::NanEncountered`] if the solution is not finite.
fn solve_linear_multi(a: &mut [f32], rhs: &mut [f32], n: usize, m: usize) -> PinnResult<Vec<f32>> {
    for col in 0..n {
        // Partial pivot: largest magnitude in the column at or below the diagonal.
        let mut pivot = col;
        let mut best = a[col * n + col].abs();
        for r in (col + 1)..n {
            let val = a[r * n + col].abs();
            if val > best {
                best = val;
                pivot = r;
            }
        }
        if best <= 1e-20 {
            return Err(PinnError::SolverDivergence {
                reason: "singular matrix in ridge solve",
            });
        }
        if pivot != col {
            for c in 0..n {
                a.swap(col * n + c, pivot * n + c);
            }
            for c in 0..m {
                rhs.swap(col * m + c, pivot * m + c);
            }
        }
        let diag = a[col * n + col];
        for r in (col + 1)..n {
            let factor = a[r * n + col] / diag;
            if factor == 0.0 {
                continue;
            }
            for c in col..n {
                a[r * n + c] -= factor * a[col * n + c];
            }
            for c in 0..m {
                rhs[r * m + c] -= factor * rhs[col * m + c];
            }
        }
    }
    // Back-substitution for each right-hand side.
    let mut x = vec![0.0_f32; n * m];
    for col in (0..n).rev() {
        let diag = a[col * n + col];
        for c in 0..m {
            let mut s = rhs[col * m + c];
            for k in (col + 1)..n {
                s -= a[col * n + k] * x[k * m + c];
            }
            x[col * m + c] = s / diag;
        }
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(PinnError::NanEncountered {
            location: "solve_linear_multi",
        });
    }
    Ok(x)
}

// ────────────────────────────── config ───────────────────────────────────────

/// Configuration for an [`EchoStateNetwork`].
#[derive(Debug, Clone)]
pub struct EsnConfig {
    /// Dimensionality of the input signal `u(t)`.
    pub input_dim: usize,
    /// Number of reservoir units (the reservoir state dimension).
    pub reservoir_size: usize,
    /// Dimensionality of the readout `y(t)`.
    pub output_dim: usize,
    /// Target spectral radius `ρ` of the recurrent matrix (typically `< 1` for
    /// the echo state property). Must be `> 0` and finite.
    pub spectral_radius: f32,
    /// Scale of the input weights: entries of `W_in` are `U(−s, s)`.
    pub input_scaling: f32,
    /// Leak rate `α ∈ (0, 1]` of the leaky-integrator update.
    pub leak_rate: f32,
    /// Fraction of non-zero recurrent connections `∈ (0, 1]` (reservoir sparsity).
    pub connectivity: f32,
    /// Tikhonov (ridge) regularisation `λ ≥ 0` for the readout least-squares fit.
    pub ridge_lambda: f32,
}

impl EsnConfig {
    /// Sensible default configuration for the given input / reservoir / output
    /// dimensions: `ρ = 0.9`, `input_scaling = 1.0`, `α = 1.0` (no leak),
    /// `connectivity = 0.1`, `λ = 1e-6`.
    #[must_use]
    pub fn new(input_dim: usize, reservoir_size: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            reservoir_size,
            output_dim,
            spectral_radius: 0.9,
            input_scaling: 1.0,
            leak_rate: 1.0,
            connectivity: 0.1,
            ridge_lambda: 1e-6,
        }
    }
}

// ────────────────────────────── ESN ──────────────────────────────────────────

/// Echo State Network: a fixed random reservoir with a trained linear readout.
#[derive(Debug, Clone)]
pub struct EchoStateNetwork {
    /// Input weights `W_in`: row-major `[reservoir_size × input_dim]`.
    w_in: Vec<f32>,
    /// Recurrent weights `W_res`: row-major `[reservoir_size × reservoir_size]`,
    /// rescaled to the configured spectral radius.
    w_res: Vec<f32>,
    /// Readout weights `W_out`: row-major `[output_dim × (reservoir_size + 1)]`
    /// (the trailing column multiplies the bias term `1`). Zero until fitted.
    w_out: Vec<f32>,
    /// Current reservoir state `x`: `[reservoir_size]`.
    state: Vec<f32>,
    config: EsnConfig,
}

impl EchoStateNetwork {
    /// Construct a new ESN with a randomly-initialised reservoir.
    ///
    /// The recurrent matrix is sampled with the configured connectivity, its
    /// spectral radius estimated, and then rescaled exactly to
    /// `config.spectral_radius`.
    ///
    /// # Errors
    /// - [`PinnError::InvalidLayerWidth`] if `reservoir_size == 0`.
    /// - [`PinnError::EmptyInput`] if `input_dim == 0` or `output_dim == 0`.
    /// - [`PinnError::InvalidWeight`] for an out-of-range `leak_rate`,
    ///   `spectral_radius`, `connectivity`, or `ridge_lambda`.
    pub fn new(config: EsnConfig, rng: &mut LcgRng) -> PinnResult<Self> {
        if config.reservoir_size == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        if config.input_dim == 0 || config.output_dim == 0 {
            return Err(PinnError::EmptyInput);
        }
        if !config.leak_rate.is_finite() || config.leak_rate <= 0.0 || config.leak_rate > 1.0 {
            return Err(PinnError::InvalidWeight {
                weight: config.leak_rate,
            });
        }
        if !config.spectral_radius.is_finite() || config.spectral_radius <= 0.0 {
            return Err(PinnError::InvalidWeight {
                weight: config.spectral_radius,
            });
        }
        if !config.connectivity.is_finite()
            || config.connectivity <= 0.0
            || config.connectivity > 1.0
        {
            return Err(PinnError::InvalidWeight {
                weight: config.connectivity,
            });
        }
        if !config.ridge_lambda.is_finite() || config.ridge_lambda < 0.0 {
            return Err(PinnError::InvalidWeight {
                weight: config.ridge_lambda,
            });
        }

        let r = config.reservoir_size;
        let din = config.input_dim;

        // Input weights: U(−input_scaling, input_scaling).
        let w_in: Vec<f32> = (0..r * din)
            .map(|_| signed_uniform(rng) * config.input_scaling)
            .collect();

        // Recurrent weights: a fraction `connectivity` of entries are non-zero,
        // each U(−1, 1); the rest are zero.
        let mut w_res: Vec<f32> = (0..r * r)
            .map(|_| {
                if unit_uniform(rng) < config.connectivity {
                    signed_uniform(rng)
                } else {
                    0.0
                }
            })
            .collect();

        // Rescale to the target spectral radius.
        let rho_est = spectral_radius(&w_res, r, SPECTRAL_RADIUS_ITERS)?;
        if rho_est > 0.0 {
            let scale = config.spectral_radius / rho_est;
            for v in &mut w_res {
                *v *= scale;
            }
        }

        let w_out = vec![0.0_f32; config.output_dim * (r + 1)];
        let state = vec![0.0_f32; r];

        Ok(Self {
            w_in,
            w_res,
            w_out,
            state,
            config,
        })
    }

    /// Reset the reservoir state to zero.
    pub fn reset_state(&mut self) {
        self.state.fill(0.0);
    }

    /// Advance the reservoir one step with input `u(t)` and return the new state.
    ///
    /// `x(t+1) = (1 − α)·x(t) + α·tanh(W_in·u + W_res·x)`.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `input.len() != input_dim`.
    /// - [`PinnError::NanEncountered`] if a non-finite state is produced.
    pub fn update(&mut self, input: &[f32]) -> PinnResult<&[f32]> {
        let r = self.config.reservoir_size;
        let din = self.config.input_dim;
        if input.len() != din {
            return Err(PinnError::DimensionMismatch {
                expected: din,
                got: input.len(),
            });
        }
        let alpha = self.config.leak_rate;
        let new_state: Vec<f32> = self
            .w_in
            .chunks_exact(din)
            .zip(self.w_res.chunks_exact(r))
            .zip(self.state.iter())
            .map(|((w_in_row, w_res_row), &xi)| {
                let drive_in: f32 = w_in_row
                    .iter()
                    .zip(input.iter())
                    .map(|(&w, &u)| w * u)
                    .sum();
                let drive_res: f32 = w_res_row
                    .iter()
                    .zip(self.state.iter())
                    .map(|(&w, &x)| w * x)
                    .sum();
                (1.0 - alpha) * xi + alpha * (drive_in + drive_res).tanh()
            })
            .collect();
        if new_state.iter().any(|v| !v.is_finite()) {
            return Err(PinnError::NanEncountered {
                location: "esn_update",
            });
        }
        self.state = new_state;
        Ok(&self.state)
    }

    /// Current augmented reservoir state `[x; 1]` of length `reservoir_size + 1`.
    fn augmented_state(&self) -> Vec<f32> {
        let mut aug = self.state.clone();
        aug.push(1.0);
        aug
    }

    /// Compute the readout `y = W_out·[x; 1]` from the current state.
    #[must_use]
    pub fn readout(&self) -> Vec<f32> {
        let aug = self.augmented_state();
        let m = aug.len();
        self.w_out
            .chunks_exact(m)
            .map(|row| row.iter().zip(aug.iter()).map(|(&w, &a)| w * a).sum())
            .collect()
    }

    /// Drive the reservoir through `n_steps` inputs and collect the augmented
    /// states after a `washout` transient.
    ///
    /// The state is reset first. Returns the design matrix as a flat row-major
    /// `[(n_steps − washout) × (reservoir_size + 1)]` vector.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `inputs.len() != n_steps · input_dim`.
    /// - [`PinnError::InvalidGridResolution`] if `washout >= n_steps`.
    pub fn collect_states(
        &mut self,
        inputs: &[f32],
        n_steps: usize,
        washout: usize,
    ) -> PinnResult<Vec<f32>> {
        let din = self.config.input_dim;
        if inputs.len() != n_steps * din {
            return Err(PinnError::DimensionMismatch {
                expected: n_steps * din,
                got: inputs.len(),
            });
        }
        if washout >= n_steps {
            return Err(PinnError::InvalidGridResolution {
                n: n_steps.saturating_sub(washout),
            });
        }
        let m = self.config.reservoir_size + 1;
        let mut design = Vec::with_capacity((n_steps - washout) * m);
        self.reset_state();
        for t in 0..n_steps {
            self.update(&inputs[t * din..(t + 1) * din])?;
            if t >= washout {
                design.extend_from_slice(&self.augmented_state());
            }
        }
        Ok(design)
    }

    /// Fit the linear readout by ridge regression on a driving sequence.
    ///
    /// Solves `(XᵀX + λI)·Θ = XᵀY` where `X` is the post-washout design matrix
    /// and `Y` the matching targets, then stores `W_out = Θᵀ`.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if the input / target lengths are
    ///   inconsistent with `n_steps`.
    /// - [`PinnError::InvalidGridResolution`] if `washout >= n_steps`.
    /// - Propagates solver errors from the ridge system.
    pub fn fit_readout(
        &mut self,
        inputs: &[f32],
        targets: &[f32],
        n_steps: usize,
        washout: usize,
    ) -> PinnResult<()> {
        let dout = self.config.output_dim;
        if targets.len() != n_steps * dout {
            return Err(PinnError::DimensionMismatch {
                expected: n_steps * dout,
                got: targets.len(),
            });
        }
        let design = self.collect_states(inputs, n_steps, washout)?;
        let m = self.config.reservoir_size + 1;
        let t_rows = n_steps - washout;

        // A = XᵀX + λI  (m×m);  B = XᵀY  (m×dout).
        let mut a = vec![0.0_f32; m * m];
        let mut bmat = vec![0.0_f32; m * dout];
        for (row, target) in design
            .chunks_exact(m)
            .zip(targets[washout * dout..].chunks_exact(dout))
            .take(t_rows)
        {
            for (i, &ri) in row.iter().enumerate() {
                for (j, &rj) in row.iter().enumerate() {
                    a[i * m + j] += ri * rj;
                }
                for (o, &yo) in target.iter().enumerate() {
                    bmat[i * dout + o] += ri * yo;
                }
            }
        }
        let lambda = self.config.ridge_lambda;
        for i in 0..m {
            a[i * m + i] += lambda;
        }

        let theta = solve_linear_multi(&mut a, &mut bmat, m, dout)?;
        // W_out[o, k] = Θ[k, o].
        let mut w_out = vec![0.0_f32; dout * m];
        for k in 0..m {
            for o in 0..dout {
                w_out[o * m + k] = theta[k * dout + o];
            }
        }
        self.w_out = w_out;
        Ok(())
    }

    /// Drive the reservoir through `inputs` and read out at each post-washout
    /// step. Returns a flat `[(n_steps − washout) × output_dim]` vector.
    ///
    /// # Errors
    /// As for [`EchoStateNetwork::collect_states`].
    pub fn predict_sequence(
        &mut self,
        inputs: &[f32],
        n_steps: usize,
        washout: usize,
    ) -> PinnResult<Vec<f32>> {
        let din = self.config.input_dim;
        if inputs.len() != n_steps * din {
            return Err(PinnError::DimensionMismatch {
                expected: n_steps * din,
                got: inputs.len(),
            });
        }
        if washout >= n_steps {
            return Err(PinnError::InvalidGridResolution {
                n: n_steps.saturating_sub(washout),
            });
        }
        let dout = self.config.output_dim;
        let mut out = Vec::with_capacity((n_steps - washout) * dout);
        self.reset_state();
        for t in 0..n_steps {
            self.update(&inputs[t * din..(t + 1) * din])?;
            if t >= washout {
                out.extend_from_slice(&self.readout());
            }
        }
        Ok(out)
    }

    /// Autonomously generate `n_generate` steps after warming up on
    /// `warmup_inputs`, feeding each prediction back as the next input.
    ///
    /// Requires `input_dim == output_dim` (the readout must be a valid next
    /// input). Returns a flat `[n_generate × output_dim]` vector.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `input_dim != output_dim` or
    ///   `warmup_inputs.len() != n_warmup · input_dim`.
    /// - [`PinnError::EmptyInput`] if `n_warmup == 0`.
    /// - [`PinnError::NanEncountered`] if a non-finite state arises.
    pub fn generate(
        &mut self,
        warmup_inputs: &[f32],
        n_warmup: usize,
        n_generate: usize,
    ) -> PinnResult<Vec<f32>> {
        let din = self.config.input_dim;
        let dout = self.config.output_dim;
        if din != dout {
            return Err(PinnError::DimensionMismatch {
                expected: din,
                got: dout,
            });
        }
        if warmup_inputs.len() != n_warmup * din {
            return Err(PinnError::DimensionMismatch {
                expected: n_warmup * din,
                got: warmup_inputs.len(),
            });
        }
        if n_warmup == 0 {
            return Err(PinnError::EmptyInput);
        }
        self.reset_state();
        for t in 0..n_warmup {
            self.update(&warmup_inputs[t * din..(t + 1) * din])?;
        }
        let mut last = self.readout();
        let mut out = Vec::with_capacity(n_generate * dout);
        for _ in 0..n_generate {
            self.update(&last)?;
            last = self.readout();
            if last.iter().any(|v| !v.is_finite()) {
                return Err(PinnError::NanEncountered {
                    location: "esn_generate",
                });
            }
            out.extend_from_slice(&last);
        }
        Ok(out)
    }

    /// Estimate the spectral radius of the (rescaled) recurrent matrix.
    ///
    /// # Errors
    /// Propagates errors from [`spectral_radius`].
    pub fn spectral_radius_estimate(&self) -> PinnResult<f32> {
        spectral_radius(
            &self.w_res,
            self.config.reservoir_size,
            SPECTRAL_RADIUS_ITERS,
        )
    }

    /// Current reservoir state.
    #[must_use]
    pub fn state(&self) -> &[f32] {
        &self.state
    }

    /// Trained readout weights `[output_dim × (reservoir_size + 1)]`.
    #[must_use]
    pub fn readout_weights(&self) -> &[f32] {
        &self.w_out
    }

    /// Reservoir size.
    #[must_use]
    pub fn reservoir_size(&self) -> usize {
        self.config.reservoir_size
    }
}

// ─────────────────────────────────── tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_esn(seed: u64) -> EchoStateNetwork {
        let mut rng = LcgRng::new(seed);
        EchoStateNetwork::new(EsnConfig::new(1, 60, 1), &mut rng)
            .expect("ESN construction with valid 1-60-1 config should succeed")
    }

    // ── unit_uniform / signed_uniform spread ──────────────────────────────────

    #[test]
    fn unit_uniform_spans_full_range() {
        // Guards against the LcgRng next_f32 [0, 0.5) hazard.
        let mut rng = LcgRng::new(123);
        let mut max = f32::NEG_INFINITY;
        let mut min = f32::INFINITY;
        for _ in 0..400 {
            let u = unit_uniform(&mut rng);
            assert!((0.0..1.0).contains(&u), "u out of [0,1): {u}");
            max = max.max(u);
            min = min.min(u);
        }
        assert!(max > 0.6, "uniform never exceeded 0.6 (max={max})");
        assert!(min < 0.4, "uniform never fell below 0.4 (min={min})");
    }

    // ── spectral_radius estimator (analytic anchors) ──────────────────────────

    #[test]
    fn spectral_radius_diagonal() {
        // diag(2, -3): ρ = max(|2|, |3|) = 3.
        let a = vec![2.0_f32, 0.0, 0.0, -3.0];
        let rho = spectral_radius(&a, 2, 120).expect("spectral radius estimation should succeed");
        assert!((rho - 3.0).abs() < 1e-2, "ρ(diag(2,-3)) ≈ 3, got {rho}");
    }

    #[test]
    fn spectral_radius_identity() {
        let a = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let rho = spectral_radius(&a, 3, 120).expect("spectral radius estimation should succeed");
        assert!((rho - 1.0).abs() < 1e-3, "ρ(I) = 1, got {rho}");
    }

    #[test]
    fn spectral_radius_complex_eigs() {
        // [[0, -0.7], [0.7, 0]] has eigenvalues ±0.7i, so ρ = 0.7.
        let a = vec![0.0_f32, -0.7, 0.7, 0.0];
        let rho = spectral_radius(&a, 2, 120).expect("spectral radius estimation should succeed");
        assert!(
            (rho - 0.7).abs() < 1e-3,
            "ρ = 0.7 for the rotation, got {rho}"
        );
    }

    #[test]
    fn spectral_radius_dim_mismatch() {
        assert!(matches!(
            spectral_radius(&[1.0, 2.0, 3.0], 2, 50),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    // ── construction / validation ─────────────────────────────────────────────

    #[test]
    fn esn_construct_finite_weights() {
        let esn = make_esn(1);
        assert!(esn.w_in.iter().all(|v| v.is_finite()));
        assert!(esn.w_res.iter().all(|v| v.is_finite()));
        assert_eq!(esn.state.len(), 60);
    }

    #[test]
    fn esn_reservoir_spectral_radius_matches_target() {
        let mut rng = LcgRng::new(7);
        let mut cfg = EsnConfig::new(1, 50, 1);
        cfg.spectral_radius = 0.8;
        cfg.connectivity = 0.2;
        let esn = EchoStateNetwork::new(cfg, &mut rng)
            .expect("ESN construction with valid config should succeed");
        let rho = esn
            .spectral_radius_estimate()
            .expect("spectral radius estimation should succeed");
        assert!(
            (rho - 0.8).abs() < 0.05,
            "rescaled reservoir ρ ≈ 0.8, got {rho}"
        );
    }

    #[test]
    fn esn_zero_reservoir_error() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            EchoStateNetwork::new(EsnConfig::new(1, 0, 1), &mut rng),
            Err(PinnError::InvalidLayerWidth)
        ));
    }

    #[test]
    fn esn_zero_input_dim_error() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            EchoStateNetwork::new(EsnConfig::new(0, 10, 1), &mut rng),
            Err(PinnError::EmptyInput)
        ));
    }

    #[test]
    fn esn_bad_leak_rate_error() {
        let mut rng = LcgRng::new(1);
        let mut cfg = EsnConfig::new(1, 10, 1);
        cfg.leak_rate = 1.5;
        assert!(matches!(
            EchoStateNetwork::new(cfg, &mut rng),
            Err(PinnError::InvalidWeight { .. })
        ));
    }

    #[test]
    fn esn_bad_spectral_radius_error() {
        let mut rng = LcgRng::new(1);
        let mut cfg = EsnConfig::new(1, 10, 1);
        cfg.spectral_radius = 0.0;
        assert!(matches!(
            EchoStateNetwork::new(cfg, &mut rng),
            Err(PinnError::InvalidWeight { .. })
        ));
    }

    #[test]
    fn esn_bad_connectivity_error() {
        let mut rng = LcgRng::new(1);
        let mut cfg = EsnConfig::new(1, 10, 1);
        cfg.connectivity = 0.0;
        assert!(matches!(
            EchoStateNetwork::new(cfg, &mut rng),
            Err(PinnError::InvalidWeight { .. })
        ));
    }

    // ── state update ──────────────────────────────────────────────────────────

    #[test]
    fn esn_update_changes_state() {
        let mut esn = make_esn(2);
        let before = esn.state().to_vec();
        esn.update(&[0.5])
            .expect("ESN state update should succeed for valid input");
        let after = esn.state().to_vec();
        assert!(
            before
                .iter()
                .zip(after.iter())
                .any(|(&a, &b)| (a - b).abs() > 1e-6),
            "state should change after update"
        );
    }

    #[test]
    fn esn_state_bounded_by_one() {
        // Starting from zero, the leaky-tanh update keeps |x_i| ≤ 1.
        let mut esn = make_esn(3);
        for t in 0..50 {
            let u = (0.3 * t as f32).sin();
            esn.update(&[u])
                .expect("ESN state update should succeed for valid input");
            assert!(
                esn.state().iter().all(|&x| x.abs() <= 1.0 + 1e-5),
                "reservoir state must stay in [-1, 1]"
            );
        }
    }

    #[test]
    fn esn_update_dim_mismatch() {
        let mut esn = make_esn(4);
        assert!(matches!(
            esn.update(&[0.1, 0.2]),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    // ── ridge readout fit ─────────────────────────────────────────────────────

    #[test]
    fn esn_fit_constant_target() {
        // The augmented state has a constant '1' column, so a constant target is
        // exactly representable: predictions on the training inputs ≈ const.
        let mut esn = make_esn(5);
        let n = 200;
        let washout = 40;
        let inputs: Vec<f32> = (0..n).map(|t| (0.25 * t as f32).sin()).collect();
        let targets = vec![0.5_f32; n];
        esn.fit_readout(&inputs, &targets, n, washout)
            .expect("fit_readout should succeed for valid training data");
        let preds = esn
            .predict_sequence(&inputs, n, washout)
            .expect("predict_sequence should succeed for valid input");
        let max_err = preds.iter().map(|&p| (p - 0.5).abs()).fold(0.0, f32::max);
        assert!(max_err < 1e-2, "constant fit error too large: {max_err}");
    }

    #[test]
    fn esn_fit_sine_one_step_prediction() {
        // One-step-ahead prediction of a sine wave: a classic ESN benchmark.
        let mut esn = make_esn(6);
        let n = 300;
        let washout = 50;
        let inputs: Vec<f32> = (0..n).map(|t| (0.2 * t as f32).sin()).collect();
        let targets: Vec<f32> = (0..n).map(|t| (0.2 * (t as f32 + 1.0)).sin()).collect();
        esn.fit_readout(&inputs, &targets, n, washout)
            .expect("fit_readout should succeed for valid training data");
        let preds = esn
            .predict_sequence(&inputs, n, washout)
            .expect("predict_sequence should succeed for valid input");
        let aligned = &targets[washout..];
        let mse: f32 = preds
            .iter()
            .zip(aligned.iter())
            .map(|(&p, &y)| (p - y) * (p - y))
            .sum::<f32>()
            / preds.len() as f32;
        assert!(mse < 2e-2, "sine one-step train MSE too large: {mse}");
    }

    #[test]
    fn esn_ridge_shrinks_readout_norm() {
        let n = 200;
        let washout = 40;
        let inputs: Vec<f32> = (0..n).map(|t| (0.2 * t as f32).sin()).collect();
        let targets: Vec<f32> = (0..n).map(|t| (0.2 * (t as f32 + 1.0)).sin()).collect();

        let mut rng_a = LcgRng::new(9);
        let mut cfg_a = EsnConfig::new(1, 50, 1);
        cfg_a.ridge_lambda = 1e-8;
        let mut esn_small = EchoStateNetwork::new(cfg_a, &mut rng_a)
            .expect("ESN construction with valid config should succeed");
        esn_small
            .fit_readout(&inputs, &targets, n, washout)
            .expect("fit_readout should succeed for valid training data");

        let mut rng_b = LcgRng::new(9);
        let mut cfg_b = EsnConfig::new(1, 50, 1);
        cfg_b.ridge_lambda = 10.0;
        let mut esn_big = EchoStateNetwork::new(cfg_b, &mut rng_b)
            .expect("ESN construction with valid config should succeed");
        esn_big
            .fit_readout(&inputs, &targets, n, washout)
            .expect("fit_readout should succeed for valid training data");

        let norm_small = l2_norm(esn_small.readout_weights());
        let norm_big = l2_norm(esn_big.readout_weights());
        assert!(
            norm_big < norm_small,
            "stronger ridge should shrink ‖W_out‖: {norm_big} !< {norm_small}"
        );
    }

    #[test]
    fn esn_fit_dim_mismatch() {
        let mut esn = make_esn(10);
        let inputs = vec![0.1_f32; 100];
        let targets = vec![0.1_f32; 50]; // wrong length for n=100
        assert!(matches!(
            esn.fit_readout(&inputs, &targets, 100, 10),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn esn_collect_states_washout_error() {
        let mut esn = make_esn(11);
        let inputs = vec![0.1_f32; 10];
        assert!(matches!(
            esn.collect_states(&inputs, 10, 10),
            Err(PinnError::InvalidGridResolution { .. })
        ));
    }

    // ── autonomous generation ─────────────────────────────────────────────────

    #[test]
    fn esn_generate_finite_and_bounded() {
        // Autonomous free-running feeds the readout back as input, so a stable
        // rollout requires a well-regularised readout and slightly sub-critical
        // dynamics; the default ridge (1e-6) over-fits and the feedback loop can
        // diverge for some reservoir realisations.
        let mut rng = LcgRng::new(12);
        let mut cfg = EsnConfig::new(1, 60, 1);
        cfg.spectral_radius = 0.8;
        cfg.ridge_lambda = 5e-3;
        let mut esn = EchoStateNetwork::new(cfg, &mut rng)
            .expect("ESN construction with valid config should succeed");
        let n_warm = 250;
        let inputs: Vec<f32> = (0..n_warm).map(|t| (0.2 * t as f32).sin()).collect();
        let targets: Vec<f32> = (0..n_warm)
            .map(|t| (0.2 * (t as f32 + 1.0)).sin())
            .collect();
        esn.fit_readout(&inputs, &targets, n_warm, 50)
            .expect("fit_readout should succeed for valid training data");

        let generated = esn
            .generate(&inputs, n_warm, 60)
            .expect("ESN generation should succeed for valid config");
        assert_eq!(generated.len(), 60);
        assert!(generated.iter().all(|v| v.is_finite()));
        assert!(
            generated.iter().all(|&v| v.abs() < 5.0),
            "autonomous rollout diverged"
        );
        // The rollout should not collapse to a single constant value.
        let span = generated.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - generated.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(span > 1e-3, "generated sequence is essentially constant");
    }

    #[test]
    fn esn_generate_requires_square_io() {
        let mut rng = LcgRng::new(13);
        let mut esn = EchoStateNetwork::new(EsnConfig::new(1, 20, 2), &mut rng)
            .expect("ESN construction with valid config should succeed");
        let inputs = vec![0.1_f32; 10];
        assert!(matches!(
            esn.generate(&inputs, 10, 5),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }
}
