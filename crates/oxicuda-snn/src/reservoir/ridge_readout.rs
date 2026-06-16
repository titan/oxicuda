//! Online ridge-regression (recursive least squares) readout for reservoirs.
//!
//! Reservoir computers (LSM / ESN) keep the recurrent weights fixed and train
//! only a linear *readout* `y = W · x` mapping the reservoir state `x ∈ ℝⁿ` to
//! the output `y ∈ ℝᵐ`. This module trains that readout online with the
//! Recursive Least Squares (RLS) rule that underlies FORCE learning
//! (Sussillo & Abbott 2009): it maintains the running inverse correlation
//! matrix `P ≈ (Σ x xᵀ + αI)⁻¹` and applies a rank-1 update per sample, so the
//! readout tracks the exact ridge solution without ever re-solving a system.
//!
//! For each `(state x, target d)` pair:
//!
//! ```text
//! Px      = P · x
//! gain    = Px / (1 + xᵀ Px)          (Kalman gain, = P_new · x)
//! e       = W · x − d                 (a-priori prediction error)
//! W       ← W − e · gainᵀ            (one Gauss-Newton step per output)
//! P       ← P − gain · Pxᵀ           (Sherman-Morrison rank-1 downdate)
//! ```
//!
//! `P` is initialised to `(1/α) I`, which makes `α` act exactly as the L2 ridge
//! coefficient. A closed-form batch solver [`RidgeReadout::fit_batch`] is also
//! provided for offline comparison; it computes the identical ridge estimate
//! `W = Y Xᵀ (X Xᵀ + αI)⁻¹` via the crate's Cholesky ridge solver.
//!
//! All matrices are row-major: `W` is `[m × n]` (`W[i·n + j]`) and `P` is
//! `[n × n]` (`P[a·n + b]`).

use crate::error::{SnnError, SnnResult};
use crate::reservoir::esn::ridge_regression;

/// Fallback initial scale for `P = scale · I` when `α = 0` (avoids `1/0`).
const P_INIT_FALLBACK: f32 = 1.0e6;

/// Online RLS / FORCE-style ridge readout mapping reservoir states to outputs.
#[derive(Debug, Clone)]
pub struct RidgeReadout {
    /// Readout weights `W`, row-major `[n_out × n_reservoir]`.
    weights: Vec<f32>,
    /// Inverse correlation matrix `P`, row-major `[n_reservoir × n_reservoir]`.
    p: Vec<f32>,
    /// Reservoir (input) dimension `n`.
    n_reservoir: usize,
    /// Output dimension `m`.
    n_out: usize,
    /// Ridge regularisation coefficient `α`.
    alpha: f32,
}

impl RidgeReadout {
    /// Create a readout with zero weights and `P = (1/α) I`.
    ///
    /// Returns [`SnnError::BadDim`] if either dimension is zero and
    /// [`SnnError::OutOfRange`] if `ridge_alpha < 0` (or non-finite).
    pub fn new(n_reservoir: usize, n_out: usize, ridge_alpha: f32) -> SnnResult<Self> {
        if n_reservoir == 0 {
            return Err(SnnError::BadDim { got: n_reservoir });
        }
        if n_out == 0 {
            return Err(SnnError::BadDim { got: n_out });
        }
        if ridge_alpha < 0.0 || !ridge_alpha.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "ridge_alpha".to_string(),
                val: ridge_alpha,
            });
        }
        let p_init = if ridge_alpha > 0.0 {
            1.0 / ridge_alpha
        } else {
            P_INIT_FALLBACK
        };
        let mut p = vec![0.0_f32; n_reservoir * n_reservoir];
        for (i, row) in p.chunks_mut(n_reservoir).enumerate() {
            row[i] = p_init;
        }
        Ok(Self {
            weights: vec![0.0_f32; n_out * n_reservoir],
            p,
            n_reservoir,
            n_out,
            alpha: ridge_alpha,
        })
    }

    /// Reservoir (state) dimension `n`.
    #[must_use]
    pub fn n_reservoir(&self) -> usize {
        self.n_reservoir
    }

    /// Output dimension `m`.
    #[must_use]
    pub fn n_out(&self) -> usize {
        self.n_out
    }

    /// Ridge regularisation coefficient `α`.
    #[must_use]
    pub fn ridge_alpha(&self) -> f32 {
        self.alpha
    }

    /// Immutable view of the readout weights, row-major `[n_out × n_reservoir]`.
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Immutable view of the inverse correlation matrix `P`, row-major
    /// `[n_reservoir × n_reservoir]`.
    #[must_use]
    pub fn p_matrix(&self) -> &[f32] {
        &self.p
    }

    /// Predict the output `y = W · state` for a single reservoir state.
    ///
    /// Returns [`SnnError::BadShape`] if `state.len() != n_reservoir`.
    pub fn predict(&self, state: &[f32]) -> SnnResult<Vec<f32>> {
        if state.len() != self.n_reservoir {
            return Err(SnnError::BadShape {
                expected: self.n_reservoir,
                got: state.len(),
            });
        }
        let n = self.n_reservoir;
        let mut y = vec![0.0_f32; self.n_out];
        for (i, y_i) in y.iter_mut().enumerate() {
            let row = &self.weights[i * n..(i + 1) * n];
            *y_i = row.iter().zip(state.iter()).map(|(&w, &s)| w * s).sum();
        }
        Ok(y)
    }

    /// One RLS update from a single `(state, target)` pair.
    ///
    /// Returns [`SnnError::BadShape`] if `state.len() != n_reservoir` or
    /// `target.len() != n_out`.
    pub fn update(&mut self, state: &[f32], target: &[f32]) -> SnnResult<()> {
        if state.len() != self.n_reservoir {
            return Err(SnnError::BadShape {
                expected: self.n_reservoir,
                got: state.len(),
            });
        }
        if target.len() != self.n_out {
            return Err(SnnError::BadShape {
                expected: self.n_out,
                got: target.len(),
            });
        }
        let n = self.n_reservoir;

        // Px = P · state.
        let mut px = vec![0.0_f32; n];
        for (a, px_a) in px.iter_mut().enumerate() {
            let row = &self.p[a * n..(a + 1) * n];
            *px_a = row.iter().zip(state.iter()).map(|(&p, &s)| p * s).sum();
        }

        // denom = 1 + stateᵀ Px  (≥ 1 since P is positive definite).
        let x_px: f32 = state.iter().zip(px.iter()).map(|(&s, &p)| s * p).sum();
        let denom = 1.0 + x_px;

        // Kalman gain = Px / denom.
        let mut gain = vec![0.0_f32; n];
        for (g, &p) in gain.iter_mut().zip(px.iter()) {
            *g = p / denom;
        }

        // A-priori prediction error e = W·state − target, then W ← W − e · gainᵀ.
        for (i, &t_i) in target.iter().enumerate() {
            let row = &self.weights[i * n..(i + 1) * n];
            let z_i: f32 = row.iter().zip(state.iter()).map(|(&w, &s)| w * s).sum();
            let e_i = z_i - t_i;
            let row_mut = &mut self.weights[i * n..(i + 1) * n];
            for (w_ij, &g_j) in row_mut.iter_mut().zip(gain.iter()) {
                *w_ij -= e_i * g_j;
            }
        }

        // P ← P − gain · Pxᵀ  (symmetric rank-1 downdate).
        for (a, &g_a) in gain.iter().enumerate() {
            let row = &mut self.p[a * n..(a + 1) * n];
            for (p_ab, &px_b) in row.iter_mut().zip(px.iter()) {
                *p_ab -= g_a * px_b;
            }
        }
        Ok(())
    }

    /// Closed-form batch ridge solve `W = Y Xᵀ (X Xᵀ + αI)⁻¹` for offline
    /// comparison, overwriting the current weights.
    ///
    /// `states` is row-major `[n_samples × n_reservoir]` and `targets` is
    /// row-major `[n_samples × n_out]`. Returns [`SnnError::BadTimesteps`] if
    /// `n_samples == 0`, [`SnnError::BadShape`] on a length mismatch, and
    /// propagates [`SnnError::Internal`] if the Gram matrix is not positive
    /// definite.
    pub fn fit_batch(
        &mut self,
        states: &[f32],
        targets: &[f32],
        n_samples: usize,
    ) -> SnnResult<()> {
        if n_samples == 0 {
            return Err(SnnError::BadTimesteps { got: n_samples });
        }
        if states.len() != n_samples * self.n_reservoir {
            return Err(SnnError::BadShape {
                expected: n_samples * self.n_reservoir,
                got: states.len(),
            });
        }
        if targets.len() != n_samples * self.n_out {
            return Err(SnnError::BadShape {
                expected: n_samples * self.n_out,
                got: targets.len(),
            });
        }
        // ridge_regression solves (XᵀX + αI)Wᵀ = XᵀY in sample-major layout,
        // which yields the same W [m × n] as Y Xᵀ (X Xᵀ + αI)⁻¹.
        let w = ridge_regression(
            states,
            targets,
            n_samples,
            self.n_reservoir,
            self.n_out,
            self.alpha,
        )?;
        self.weights = w;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn online_rls_converges_on_linear_map() {
        let n = 4;
        let w_true = [0.5_f32, -1.0, 2.0, 0.3];
        let mut readout = RidgeReadout::new(n, 1, 1e-3).expect("ctor");
        let mut rng = LcgRng::new(123);

        // Stream noise-free samples drawn from a fixed linear map.
        for _ in 0..120 {
            let state: Vec<f32> = (0..n).map(|_| rng.next_normal()).collect();
            let target: f32 = w_true.iter().zip(state.iter()).map(|(&w, &s)| w * s).sum();
            readout.update(&state, &[target]).expect("update");
        }

        // Predictions on held-out states must match the true map.
        let mut max_err = 0.0_f32;
        for _ in 0..20 {
            let state: Vec<f32> = (0..n).map(|_| rng.next_normal()).collect();
            let target: f32 = w_true.iter().zip(state.iter()).map(|(&w, &s)| w * s).sum();
            let pred = readout.predict(&state).expect("predict")[0];
            max_err = max_err.max((pred - target).abs());
        }
        assert!(max_err < 1e-2, "RLS failed to converge, max_err={max_err}");
    }

    #[test]
    fn batch_ridge_reproduces_known_map() {
        let n = 3;
        let m = 2;
        // w_true row-major [m × n].
        let w_true = [1.0_f32, -2.0, 0.5, 0.3, 0.7, -1.0];
        let mut rng = LcgRng::new(7);
        let n_samples = 16;
        let mut states = vec![0.0_f32; n_samples * n];
        let mut targets = vec![0.0_f32; n_samples * m];
        for k in 0..n_samples {
            for j in 0..n {
                states[k * n + j] = rng.next_normal();
            }
            for i in 0..m {
                let mut acc = 0.0_f32;
                for j in 0..n {
                    acc += w_true[i * n + j] * states[k * n + j];
                }
                targets[k * m + i] = acc;
            }
        }

        let mut readout = RidgeReadout::new(n, m, 1e-6).expect("ctor");
        readout
            .fit_batch(&states, &targets, n_samples)
            .expect("fit");

        // Recovered weights match w_true on noise-free data.
        let mut max_err = 0.0_f32;
        for (&got, &want) in readout.weights().iter().zip(w_true.iter()) {
            max_err = max_err.max((got - want).abs());
        }
        assert!(
            max_err < 1e-4,
            "batch ridge weight error too large: {max_err}"
        );
    }

    #[test]
    fn p_stays_finite_and_spd_like() {
        let n = 5;
        let mut readout = RidgeReadout::new(n, 2, 1e-2).expect("ctor");
        let mut rng = LcgRng::new(99);
        for _ in 0..50 {
            let state: Vec<f32> = (0..n).map(|_| rng.next_normal()).collect();
            let target = [rng.next_normal(), rng.next_normal()];
            readout.update(&state, &target).expect("update");
        }
        let p = readout.p_matrix();
        // All entries finite.
        assert!(p.iter().all(|v| v.is_finite()), "P has non-finite entries");
        // Diagonal positive (necessary condition for SPD).
        for i in 0..n {
            assert!(p[i * n + i] > 0.0, "P diagonal {i} not positive");
        }
        // Symmetry (P is a symmetric rank-1 downdate of a symmetric matrix).
        for a in 0..n {
            for b in 0..n {
                let diff = (p[a * n + b] - p[b * n + a]).abs();
                assert!(diff < 1e-4, "P not symmetric at ({a},{b}): {diff}");
            }
        }
    }

    #[test]
    fn dim_mismatch_is_error() {
        let mut readout = RidgeReadout::new(4, 2, 1e-2).expect("ctor");
        assert!(matches!(
            readout.update(&[0.0; 3], &[0.0; 2]),
            Err(SnnError::BadShape { .. })
        ));
        assert!(matches!(
            readout.update(&[0.0; 4], &[0.0; 1]),
            Err(SnnError::BadShape { .. })
        ));
        assert!(matches!(
            readout.predict(&[0.0; 5]),
            Err(SnnError::BadShape { .. })
        ));
        assert!(matches!(
            readout.fit_batch(&[0.0; 4], &[0.0; 2], 2),
            Err(SnnError::BadShape { .. })
        ));
        assert!(matches!(
            readout.fit_batch(&[], &[], 0),
            Err(SnnError::BadTimesteps { .. })
        ));
    }

    #[test]
    fn negative_alpha_is_error() {
        assert!(matches!(
            RidgeReadout::new(4, 1, -1.0),
            Err(SnnError::OutOfRange { .. })
        ));
    }

    #[test]
    fn zero_dimension_is_error() {
        assert!(matches!(
            RidgeReadout::new(0, 1, 1.0),
            Err(SnnError::BadDim { .. })
        ));
        assert!(matches!(
            RidgeReadout::new(2, 0, 1.0),
            Err(SnnError::BadDim { .. })
        ));
    }
}
