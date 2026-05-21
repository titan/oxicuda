//! S5 sequence layer — Simplified State Space Layer (Smith et al. 2022).
//!
//! # Background
//!
//! S5 ("Simplified State Space Layers for Sequence Modeling", Smith et al. 2022)
//! reduces the complexity of S4 by using a **diagonal real A matrix**: `A = diag(a)`
//! where `a_n < 0` for all `n` (stable poles).  Because A is diagonal, the
//! Zero-Order Hold (ZOH) discretization has a closed-form solution per element:
//!
//! ```text
//! Ā_n = exp(Δ · a_n)
//! B̄_{n,:} = ((Ā_n − 1) / a_n) · B_{n,:}      (L'Hôpital when a_n → 0)
//! ```
//!
//! The layer is fully MIMO (Multi-Input Multi-Output):
//!
//! ```text
//! h_t = Ā ⊙ h_{t-1} + B̄ @ u_t      (⊙ = element-wise, Ā diagonal ↔ vector)
//! y_t = C @ h_t + D_t                 (D_t = D[i] · u_t[i] if U=Y, else D[i] bias)
//! ```
//!
//! ## Initialization
//!
//! * `a_diag[n] = -(n + 1)` for `n = 0..N` — the diagonal of the HiPPO-LegS A
//!   matrix, giving stable poles at `−1, −2, …, −N`.
//! * `B`: Xavier uniform (fan_in = state_dim, fan_out = u_dim).
//! * `C`: Xavier uniform (fan_in = y_dim, fan_out = state_dim).
//! * `D`: zeros (no skip connection by default).

use crate::error::{MambaError, MambaResult};
use crate::handle::LcgRng;

// ─── S5Config ────────────────────────────────────────────────────────────────

/// Configuration for an S5 sequence layer.
#[derive(Debug, Clone)]
pub struct S5Config {
    /// Input dimension `U`.
    pub u_dim: usize,
    /// Output dimension `Y`.
    pub y_dim: usize,
    /// SSM hidden state dimension `N`.
    pub state_dim: usize,
    /// Expected sequence length `L`.
    pub seq_len: usize,
    /// ZOH discretization step `Δ > 0`.
    pub delta: f32,
}

impl S5Config {
    /// Create a new S5 configuration with `delta = 0.01`.
    ///
    /// # Errors
    ///
    /// * [`MambaError::InvalidModelDim`]  — if `u_dim == 0` or `y_dim == 0`.
    /// * [`MambaError::InvalidSsmOrder`]  — if `state_dim == 0`.
    /// * [`MambaError::InvalidSeqLen`]    — if `seq_len == 0`.
    pub fn new(u_dim: usize, y_dim: usize, state_dim: usize, seq_len: usize) -> MambaResult<Self> {
        if u_dim == 0 {
            return Err(MambaError::InvalidModelDim(u_dim));
        }
        if y_dim == 0 {
            return Err(MambaError::InvalidModelDim(y_dim));
        }
        if state_dim == 0 {
            return Err(MambaError::InvalidSsmOrder(state_dim));
        }
        if seq_len == 0 {
            return Err(MambaError::InvalidSeqLen(seq_len));
        }
        Ok(Self {
            u_dim,
            y_dim,
            state_dim,
            seq_len,
            delta: 0.01_f32,
        })
    }

    /// Override the discretization step `Δ`.
    ///
    /// # Errors
    ///
    /// [`MambaError::NonPositiveDelta`] if `delta ≤ 0`.
    pub fn with_delta(mut self, delta: f32) -> MambaResult<Self> {
        if delta <= 0.0 {
            return Err(MambaError::NonPositiveDelta(delta));
        }
        self.delta = delta;
        Ok(self)
    }
}

// ─── S5Weights ───────────────────────────────────────────────────────────────

/// Learnable weights for an S5 layer.
#[derive(Debug, Clone)]
pub struct S5Weights {
    /// Diagonal of the continuous-time A matrix, length `state_dim`.
    ///
    /// All entries should be strictly negative for stability.
    /// Initialised as `a_diag[n] = -(n + 1)` (HiPPO-LegS diagonal).
    pub a_diag: Vec<f32>,
    /// Input matrix B, row-major `[state_dim × u_dim]`.
    pub b: Vec<f32>,
    /// Output matrix C, row-major `[y_dim × state_dim]`.
    pub c: Vec<f32>,
    /// Skip connection D, length `y_dim`.
    ///
    /// Applied as `D[i] * u_t[i]` when `y_dim == u_dim`, otherwise as a bias.
    pub d: Vec<f32>,
}

impl S5Weights {
    /// Initialize S5 weights:
    ///
    /// * `a_diag[n] = -(n + 1)` — HiPPO-LegS diagonal (stable poles).
    /// * `b`: Xavier uniform, `fan_in = state_dim`, `fan_out = u_dim`.
    /// * `c`: Xavier uniform, `fan_in = y_dim`, `fan_out = state_dim`.
    /// * `d`: zeros (no skip by default).
    #[must_use]
    pub fn new(cfg: &S5Config, rng: &mut LcgRng) -> Self {
        let n = cfg.state_dim;
        let u = cfg.u_dim;
        let y = cfg.y_dim;

        // HiPPO-LegS diagonal: -(n+1).
        let a_diag: Vec<f32> = (0..n).map(|i| -((i + 1) as f32)).collect();

        // Xavier uniform fill: scale = sqrt(6 / (fan_in + fan_out)).
        let xavier = |fan_in: usize, fan_out: usize, len: usize, rng: &mut LcgRng| -> Vec<f32> {
            let scale = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
            (0..len)
                .map(|_| rng.next_f32() * 2.0 * scale - scale)
                .collect()
        };

        let b = xavier(n, u, n * u, rng);
        let c = xavier(y, n, y * n, rng);
        let d = vec![0.0_f32; y];

        Self { a_diag, b, c, d }
    }
}

// ─── S5Layer ─────────────────────────────────────────────────────────────────

/// S5 sequence-to-sequence layer.
///
/// Stores the config and the pre-discretized `Ā` and `B̄` matrices for
/// efficient sequential inference.  Call [`S5Layer::update_discretization`]
/// after changing weights during training.
pub struct S5Layer {
    /// Layer configuration.
    pub config: S5Config,
    /// Discretized diagonal `Ā_n = exp(Δ · a_n)`, length `state_dim`.
    a_bar: Vec<f32>,
    /// Discretized input matrix `B̄`, row-major `[state_dim × u_dim]`.
    b_bar: Vec<f32>,
}

/// ZOH discretization for a diagonal A (given as a vector of diagonal entries).
///
/// Returns `(a_bar, b_bar)` where:
/// * `a_bar[n] = exp(Δ · a_n)`.
/// * `b_bar[n, j] = ((exp(Δ · a_n) − 1) / a_n) · b[n, j]`
///   with L'Hôpital limit `Δ` when `|a_n| < 1e-6`.
fn discretize(
    a_diag: &[f32],
    b: &[f32],
    delta: f32,
    n: usize,
    u_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut a_bar = Vec::with_capacity(n);
    let mut b_bar = vec![0.0_f32; n * u_dim];
    for i in 0..n {
        let a_n = a_diag[i];
        let a_bar_n = (delta * a_n).exp();
        a_bar.push(a_bar_n);
        // ZOH scale: (exp(Δ·a_n) − 1) / a_n; L'Hôpital when a_n → 0 gives Δ.
        let scale = if a_n.abs() < 1e-6_f32 {
            delta
        } else {
            (a_bar_n - 1.0) / a_n
        };
        for j in 0..u_dim {
            b_bar[i * u_dim + j] = scale * b[i * u_dim + j];
        }
    }
    (a_bar, b_bar)
}

impl S5Layer {
    /// Build an S5 layer from config and weights (pre-discretizes A and B).
    ///
    /// # Errors
    ///
    /// [`MambaError::DimensionMismatch`] if weight vectors have wrong lengths.
    pub fn new(cfg: S5Config, weights: &S5Weights) -> MambaResult<Self> {
        let n = cfg.state_dim;
        let u = cfg.u_dim;
        let y = cfg.y_dim;

        // Validate weight shapes.
        if weights.a_diag.len() != n {
            return Err(MambaError::DimensionMismatch {
                expected: n,
                got: weights.a_diag.len(),
            });
        }
        if weights.b.len() != n * u {
            return Err(MambaError::DimensionMismatch {
                expected: n * u,
                got: weights.b.len(),
            });
        }
        if weights.c.len() != y * n {
            return Err(MambaError::DimensionMismatch {
                expected: y * n,
                got: weights.c.len(),
            });
        }
        if weights.d.len() != y {
            return Err(MambaError::DimensionMismatch {
                expected: y,
                got: weights.d.len(),
            });
        }

        let (a_bar, b_bar) = discretize(&weights.a_diag, &weights.b, cfg.delta, n, u);
        Ok(Self {
            config: cfg,
            a_bar,
            b_bar,
        })
    }

    /// Recompute `Ā` and `B̄` from updated weights.
    ///
    /// Call this after modifying `weights.a_diag` or `weights.b` (e.g. during
    /// a gradient update step).
    ///
    /// # Errors
    ///
    /// [`MambaError::DimensionMismatch`] if weight vectors have wrong lengths.
    pub fn update_discretization(&mut self, weights: &S5Weights) -> MambaResult<()> {
        let n = self.config.state_dim;
        let u = self.config.u_dim;
        let y = self.config.y_dim;

        if weights.a_diag.len() != n {
            return Err(MambaError::DimensionMismatch {
                expected: n,
                got: weights.a_diag.len(),
            });
        }
        if weights.b.len() != n * u {
            return Err(MambaError::DimensionMismatch {
                expected: n * u,
                got: weights.b.len(),
            });
        }
        if weights.c.len() != y * n {
            return Err(MambaError::DimensionMismatch {
                expected: y * n,
                got: weights.c.len(),
            });
        }
        if weights.d.len() != y {
            return Err(MambaError::DimensionMismatch {
                expected: y,
                got: weights.d.len(),
            });
        }

        let (a_bar, b_bar) = discretize(&weights.a_diag, &weights.b, self.config.delta, n, u);
        self.a_bar = a_bar;
        self.b_bar = b_bar;
        Ok(())
    }

    /// Forward pass: process the full sequence `u: [L × U]` → `y: [L × Y]`.
    ///
    /// State is initialised to zero at `t = 0`.
    ///
    /// # Errors
    ///
    /// [`MambaError::DimensionMismatch`] if `u.len() != seq_len * u_dim`.
    pub fn forward(&self, u: &[f32], weights: &S5Weights) -> MambaResult<Vec<f32>> {
        let l = self.config.seq_len;
        let n = self.config.state_dim;
        let y_dim = self.config.y_dim;
        let u_dim = self.config.u_dim;
        let expected = l * u_dim;

        if u.len() != expected {
            return Err(MambaError::DimensionMismatch {
                expected,
                got: u.len(),
            });
        }

        let mut h = vec![0.0_f32; n];
        let mut output = Vec::with_capacity(l * y_dim);

        for t in 0..l {
            let u_t = &u[t * u_dim..(t + 1) * u_dim];

            // h_t = Ā ⊙ h_{t-1} + B̄ @ u_t
            let mut h_new: Vec<f32> = self
                .a_bar
                .iter()
                .zip(h.iter())
                .map(|(&a, &hv)| a * hv)
                .collect();
            for (i, h_new_i) in h_new.iter_mut().enumerate() {
                let b_row = &self.b_bar[i * u_dim..(i + 1) * u_dim];
                *h_new_i += b_row
                    .iter()
                    .zip(u_t.iter())
                    .map(|(&b, &u)| b * u)
                    .sum::<f32>();
            }
            h = h_new;

            // y_t = C @ h_t + D_term
            let uses_d_skip = u_dim == y_dim;
            let y_t: Vec<f32> = (0..y_dim)
                .map(|i| {
                    let c_row = &weights.c[i * n..(i + 1) * n];
                    let s: f32 = c_row.iter().zip(h.iter()).map(|(&c, &hv)| c * hv).sum();
                    // D skip: per-output scalar applied to u_t[i] when dims match,
                    // otherwise used as a bias term.
                    let d_term = if uses_d_skip {
                        weights.d[i] * u_t[i]
                    } else {
                        weights.d[i]
                    };
                    s + d_term
                })
                .collect();
            output.extend_from_slice(&y_t);
        }

        Ok(output)
    }

    /// Single-step recurrent forward: `(h_{t-1}, u_t) → (y_t, h_t)`.
    ///
    /// # Arguments
    ///
    /// * `h`   — previous hidden state, length `state_dim`.
    /// * `u_t` — current input, length `u_dim`.
    /// * `weights` — model parameters (C and D used for output).
    ///
    /// # Returns
    ///
    /// `(y_t, h_t)` where `y_t` has length `y_dim` and `h_t` has length `state_dim`.
    ///
    /// # Errors
    ///
    /// [`MambaError::DimensionMismatch`] if `h` or `u_t` have wrong lengths.
    pub fn step(
        &self,
        h: &[f32],
        u_t: &[f32],
        weights: &S5Weights,
    ) -> MambaResult<(Vec<f32>, Vec<f32>)> {
        let n = self.config.state_dim;
        let y_dim = self.config.y_dim;
        let u_dim = self.config.u_dim;

        if h.len() != n {
            return Err(MambaError::DimensionMismatch {
                expected: n,
                got: h.len(),
            });
        }
        if u_t.len() != u_dim {
            return Err(MambaError::DimensionMismatch {
                expected: u_dim,
                got: u_t.len(),
            });
        }

        // h_new = Ā ⊙ h + B̄ @ u_t
        let mut h_new: Vec<f32> = self
            .a_bar
            .iter()
            .zip(h.iter())
            .map(|(&a, &hv)| a * hv)
            .collect();
        for (i, h_new_i) in h_new.iter_mut().enumerate() {
            let b_row = &self.b_bar[i * u_dim..(i + 1) * u_dim];
            *h_new_i += b_row
                .iter()
                .zip(u_t.iter())
                .map(|(&b, &u)| b * u)
                .sum::<f32>();
        }

        // y_t = C @ h_new + D_term
        let uses_d_skip = u_dim == y_dim;
        let y_t: Vec<f32> = (0..y_dim)
            .map(|i| {
                let c_row = &weights.c[i * n..(i + 1) * n];
                let s: f32 = c_row.iter().zip(h_new.iter()).map(|(&c, &hv)| c * hv).sum();
                let d_term = if uses_d_skip {
                    weights.d[i] * u_t[i]
                } else {
                    weights.d[i]
                };
                s + d_term
            })
            .collect();

        Ok((y_t, h_new))
    }

    /// Compute the normalised mean squared error between the layer's output and a target.
    ///
    /// ```text
    /// mse = ||y_pred − y_target||² / (L · Y)
    /// ```
    ///
    /// # Errors
    ///
    /// * [`MambaError::DimensionMismatch`] — if `u.len() != L * U` or `y_target.len() != L * Y`.
    pub fn sequence_mse(
        &self,
        u: &[f32],
        y_target: &[f32],
        weights: &S5Weights,
    ) -> MambaResult<f32> {
        let l = self.config.seq_len;
        let y_dim = self.config.y_dim;
        let expected_y = l * y_dim;

        if y_target.len() != expected_y {
            return Err(MambaError::DimensionMismatch {
                expected: expected_y,
                got: y_target.len(),
            });
        }

        let y_pred = self.forward(u, weights)?;
        let total: f32 = y_pred
            .iter()
            .zip(y_target.iter())
            .map(|(&p, &t)| (p - t) * (p - t))
            .sum();
        Ok(total / (expected_y as f32))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    const EPS: f32 = 1e-5;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn randn(rng: &mut LcgRng, n: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    // ── S5Config ──────────────────────────────────────────────────────────────

    #[test]
    fn s5_config_new_valid() {
        let cfg = S5Config::new(4, 4, 8, 10).expect("valid config");
        assert_eq!(cfg.u_dim, 4);
        assert_eq!(cfg.y_dim, 4);
        assert_eq!(cfg.state_dim, 8);
        assert_eq!(cfg.seq_len, 10);
        assert!((cfg.delta - 0.01).abs() < EPS);
    }

    #[test]
    fn s5_config_zero_u_dim_error() {
        let err = S5Config::new(0, 4, 8, 10).expect_err("u_dim=0 must fail");
        assert!(matches!(err, MambaError::InvalidModelDim(0)));
    }

    #[test]
    fn s5_config_zero_state_error() {
        let err = S5Config::new(4, 4, 0, 10).expect_err("state_dim=0 must fail");
        assert!(matches!(err, MambaError::InvalidSsmOrder(0)));
    }

    #[test]
    fn s5_config_zero_seq_len_error() {
        let err = S5Config::new(4, 4, 8, 0).expect_err("seq_len=0 must fail");
        assert!(matches!(err, MambaError::InvalidSeqLen(0)));
    }

    #[test]
    fn s5_config_negative_delta_error() {
        let err = S5Config::new(4, 4, 8, 10)
            .expect("valid")
            .with_delta(-0.1)
            .expect_err("negative delta must fail");
        assert!(matches!(err, MambaError::NonPositiveDelta(_)));
    }

    // ── S5Weights ─────────────────────────────────────────────────────────────

    #[test]
    fn s5_weights_new_a_diag_negative() {
        let cfg = S5Config::new(4, 4, 8, 10).expect("valid");
        let mut rng = make_rng();
        let w = S5Weights::new(&cfg, &mut rng);
        for (i, &v) in w.a_diag.iter().enumerate() {
            assert!(v < 0.0, "a_diag[{i}]={v} should be negative for stability");
        }
    }

    #[test]
    fn s5_weights_a_bar_in_0_1() {
        let cfg = S5Config::new(4, 4, 8, 10).expect("valid");
        let mut rng = make_rng();
        let w = S5Weights::new(&cfg, &mut rng);
        let n = cfg.state_dim;
        let (a_bar, _) = discretize(&w.a_diag, &w.b, cfg.delta, n, cfg.u_dim);
        for (i, &v) in a_bar.iter().enumerate() {
            assert!(
                v > 0.0 && v <= 1.0,
                "a_bar[{i}]={v} should be in (0, 1] for stable a_diag < 0"
            );
        }
    }

    // ── S5Layer ───────────────────────────────────────────────────────────────

    #[test]
    fn s5_forward_output_shape() {
        let cfg = S5Config::new(4, 4, 8, 10).expect("valid");
        let mut rng = make_rng();
        let w = S5Weights::new(&cfg, &mut rng);
        let layer = S5Layer::new(cfg.clone(), &w).expect("layer ok");
        let u = randn(&mut rng, cfg.seq_len * cfg.u_dim);
        let y = layer.forward(&u, &w).expect("forward ok");
        assert_eq!(y.len(), cfg.seq_len * cfg.y_dim, "output shape [L*Y]");
    }

    #[test]
    fn s5_forward_zero_input_finite() {
        let cfg = S5Config::new(4, 4, 8, 10).expect("valid");
        let mut rng = make_rng();
        let w = S5Weights::new(&cfg, &mut rng);
        // Zero d means zero output for zero input (C@0=0, d=0).
        let mut w_zero_d = w.clone();
        w_zero_d.d = vec![0.0_f32; cfg.y_dim];
        // zero B too so that u=0 gives h=0.
        w_zero_d.b = vec![0.0_f32; cfg.state_dim * cfg.u_dim];
        let layer = S5Layer::new(cfg.clone(), &w_zero_d).expect("layer ok");
        let u = vec![0.0_f32; cfg.seq_len * cfg.u_dim];
        let y = layer.forward(&u, &w_zero_d).expect("forward ok");
        for (i, &v) in y.iter().enumerate() {
            assert!(
                v.abs() < EPS,
                "y[{i}]={v} should be zero for zero input with zero B/d"
            );
        }
    }

    #[test]
    fn s5_forward_causal() {
        // Changing u at t=5 must not affect y at t=4 (causality).
        let l = 6;
        let u_dim = 2;
        let y_dim = 2;
        let n = 4;
        let cfg = S5Config::new(u_dim, y_dim, n, l).expect("valid");
        let mut rng = make_rng();
        let w = S5Weights::new(&cfg, &mut rng);
        let layer = S5Layer::new(cfg.clone(), &w).expect("layer ok");

        let mut u_a = randn(&mut rng, l * u_dim);
        let u_b = u_a.clone();
        // Perturb t=5 (last time step) in u_a.
        u_a[(l - 1) * u_dim] += 99.9;
        u_a[(l - 1) * u_dim + 1] += 99.9;

        let y_a = layer.forward(&u_a, &w).expect("forward a");
        let y_b = layer.forward(&u_b, &w).expect("forward b");

        // All outputs t=0..4 must be identical.
        for t in 0..(l - 1) {
            for i in 0..y_dim {
                let pa = y_a[t * y_dim + i];
                let pb = y_b[t * y_dim + i];
                assert!(
                    (pa - pb).abs() < EPS,
                    "y[t={t},i={i}] should be causal: a={pa}, b={pb}"
                );
            }
        }
    }

    #[test]
    fn s5_step_matches_forward_t0() {
        let cfg = S5Config::new(3, 3, 6, 8).expect("valid");
        let mut rng = make_rng();
        let w = S5Weights::new(&cfg, &mut rng);
        let layer = S5Layer::new(cfg.clone(), &w).expect("layer ok");

        let u = randn(&mut rng, cfg.seq_len * cfg.u_dim);
        let y_fwd = layer.forward(&u, &w).expect("forward ok");

        // step at t=0 with h=0
        let h0 = vec![0.0_f32; cfg.state_dim];
        let u_0 = &u[0..cfg.u_dim];
        let (y_step0, _) = layer.step(&h0, u_0, &w).expect("step ok");

        for i in 0..cfg.y_dim {
            let from_fwd = y_fwd[i];
            let from_step = y_step0[i];
            assert!(
                (from_fwd - from_step).abs() < EPS,
                "step/forward mismatch at t=0, i={i}: fwd={from_fwd}, step={from_step}"
            );
        }
    }

    #[test]
    fn s5_step_state_update() {
        let cfg = S5Config::new(2, 2, 4, 6).expect("valid");
        let mut rng = make_rng();
        let w = S5Weights::new(&cfg, &mut rng);
        let layer = S5Layer::new(cfg.clone(), &w).expect("layer ok");

        let h0 = vec![0.0_f32; cfg.state_dim];
        let u_t = randn(&mut rng, cfg.u_dim);
        let (_, h1) = layer.step(&h0, &u_t, &w).expect("step ok");

        // h must change for non-zero input (assuming b_bar != 0).
        let any_change = h0
            .iter()
            .zip(h1.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-10);
        assert!(
            any_change,
            "state must change after step with non-zero input"
        );
    }

    #[test]
    fn s5_step_output_shape() {
        let cfg = S5Config::new(3, 5, 8, 10).expect("valid");
        let mut rng = make_rng();
        let w = S5Weights::new(&cfg, &mut rng);
        let layer = S5Layer::new(cfg.clone(), &w).expect("layer ok");

        let h = vec![0.0_f32; cfg.state_dim];
        let u_t = randn(&mut rng, cfg.u_dim);
        let (y_t, _) = layer.step(&h, &u_t, &w).expect("step ok");
        assert_eq!(y_t.len(), cfg.y_dim, "y_t must have length y_dim");
    }

    #[test]
    fn s5_step_state_shape() {
        let cfg = S5Config::new(3, 5, 8, 10).expect("valid");
        let mut rng = make_rng();
        let w = S5Weights::new(&cfg, &mut rng);
        let layer = S5Layer::new(cfg.clone(), &w).expect("layer ok");

        let h = vec![0.0_f32; cfg.state_dim];
        let u_t = randn(&mut rng, cfg.u_dim);
        let (_, h_new) = layer.step(&h, &u_t, &w).expect("step ok");
        assert_eq!(h_new.len(), cfg.state_dim, "h_t must have length state_dim");
    }

    #[test]
    fn s5_zoh_discretization_a_n_zero() {
        // When a_n = 0 the function must use the L'Hôpital limit (no NaN).
        let a_diag = vec![0.0_f32]; // edge case
        let b = vec![1.0_f32];
        let delta = 0.1_f32;
        let (a_bar, b_bar) = discretize(&a_diag, &b, delta, 1, 1);
        // exp(0) = 1.0
        assert!(
            (a_bar[0] - 1.0_f32).abs() < EPS,
            "a_bar should be 1 for a=0"
        );
        // L'Hôpital: scale = delta = 0.1 → b_bar = 0.1 * 1.0 = 0.1
        assert!(
            (b_bar[0] - delta).abs() < EPS,
            "b_bar should be delta for a=0, got {}",
            b_bar[0]
        );
        assert!(
            a_bar[0].is_finite() && b_bar[0].is_finite(),
            "must be finite"
        );
    }

    #[test]
    fn s5_sequence_mse_zero_residual() {
        // If y_target == y_pred, mse should be exactly 0.
        let cfg = S5Config::new(2, 2, 4, 5).expect("valid");
        let mut rng = make_rng();
        let w = S5Weights::new(&cfg, &mut rng);
        let layer = S5Layer::new(cfg.clone(), &w).expect("layer ok");

        let u = randn(&mut rng, cfg.seq_len * cfg.u_dim);
        let y_pred = layer.forward(&u, &w).expect("forward ok");
        let mse = layer.sequence_mse(&u, &y_pred, &w).expect("mse ok");
        assert!(
            mse.abs() < EPS,
            "mse should be ~0 when y_pred == y_target: {mse}"
        );
    }

    #[test]
    fn s5_sequence_mse_nonneg() {
        let cfg = S5Config::new(2, 2, 4, 5).expect("valid");
        let mut rng = make_rng();
        let w = S5Weights::new(&cfg, &mut rng);
        let layer = S5Layer::new(cfg.clone(), &w).expect("layer ok");

        let u = randn(&mut rng, cfg.seq_len * cfg.u_dim);
        let y_target = randn(&mut rng, cfg.seq_len * cfg.y_dim);
        let mse = layer.sequence_mse(&u, &y_target, &w).expect("mse ok");
        assert!(mse >= 0.0, "mse must be non-negative: {mse}");
    }

    #[test]
    fn s5_forward_wrong_input_shape_error() {
        let cfg = S5Config::new(4, 4, 8, 10).expect("valid");
        let mut rng = make_rng();
        let w = S5Weights::new(&cfg, &mut rng);
        let layer = S5Layer::new(cfg.clone(), &w).expect("layer ok");
        // Wrong length: seq_len * u_dim + 1
        let u = vec![0.0_f32; cfg.seq_len * cfg.u_dim + 1];
        let err = layer
            .forward(&u, &w)
            .expect_err("must fail for wrong shape");
        assert!(matches!(err, MambaError::DimensionMismatch { .. }));
    }
}
