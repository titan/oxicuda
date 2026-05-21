//! N-BEATS block: 4-layer MLP → theta → basis expansion → backcast + forecast.
//!
//! Three block variants follow Oreshkin et al. (ICLR 2020):
//! - **Generic**: learned linear projections from theta to backcast/forecast.
//! - **Trend**: polynomial basis (`t^0, t^1, …, t^degree`) applied per-channel.
//! - **Seasonality**: Fourier (cos + sin) basis applied per-channel.

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

// ─── Helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn relu_vec(v: &mut [f32]) {
    for x in v.iter_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

/// Dense linear layer: `out[j] = Σ_i w[j*in_dim + i] * x[i] + b[j]`.
fn linear(x: &[f32], w: &[f32], b: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = b.to_vec();
    for j in 0..out_dim {
        let row = &w[j * in_dim..(j + 1) * in_dim];
        out[j] += row
            .iter()
            .zip(x.iter())
            .map(|(&wi, &xi)| wi * xi)
            .sum::<f32>();
    }
    out
}

/// Xavier-uniform init: `scale = sqrt(6 / (fan_in + fan_out))`.
fn xavier_fill(buf: &mut [f32], fan_in: usize, fan_out: usize, rng: &mut LcgRng) {
    let scale = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
    rng.fill_normal(buf);
    for v in buf.iter_mut() {
        *v *= scale;
    }
}

// ─── Block Type ───────────────────────────────────────────────────────────────

/// The three block types for N-BEATS.
#[derive(Debug, Clone)]
pub enum NBeatsBlockType {
    /// Generic unconstrained basis: theta projected through learned linear heads.
    Generic {
        /// MLP hidden dimension.
        hidden_units: usize,
        /// Theta dimension per direction (backcast / forecast each get `theta_dim` elements).
        theta_dim: usize,
    },
    /// Trend block: polynomial basis of given degree.
    Trend {
        /// MLP hidden dimension.
        hidden_units: usize,
        /// Polynomial degree (basis has `degree + 1` columns).
        degree: usize,
    },
    /// Seasonality block: Fourier basis with `n_harmonics` sinusoidal pairs.
    Seasonality {
        /// MLP hidden dimension.
        hidden_units: usize,
        /// Number of harmonic pairs (cos + sin), giving `2 * n_harmonics` basis columns.
        n_harmonics: usize,
    },
}

impl NBeatsBlockType {
    fn hidden_units(&self) -> usize {
        match self {
            Self::Generic { hidden_units, .. }
            | Self::Trend { hidden_units, .. }
            | Self::Seasonality { hidden_units, .. } => *hidden_units,
        }
    }
}

// ─── NBeatsBlock ─────────────────────────────────────────────────────────────

/// A single N-BEATS block (one element within a stack).
///
/// Implements a 4-layer MLP followed by two expansion heads (backcast and
/// forecast). The expansion uses either learned linear projections (Generic) or
/// precomputed analytical bases (Trend / Seasonality).
#[derive(Debug, Clone)]
pub struct NBeatsBlock {
    // ── 4-layer MLP ─────────────────────────────────────────────────────────
    /// `[hidden × (t*c)]`
    pub mlp_w1: Vec<f32>,
    /// `[hidden]`
    pub mlp_b1: Vec<f32>,
    /// `[hidden × hidden]`
    pub mlp_w2: Vec<f32>,
    /// `[hidden]`
    pub mlp_b2: Vec<f32>,
    /// `[hidden × hidden]`
    pub mlp_w3: Vec<f32>,
    /// `[hidden]`
    pub mlp_b3: Vec<f32>,
    /// `[theta_size × hidden]`
    pub mlp_w4: Vec<f32>,
    /// `[theta_size]`
    pub mlp_b4: Vec<f32>,

    // ── Generic heads (None for Trend / Seasonality) ─────────────────────────
    /// Backcast head weight `[t*c × theta_half]` — Generic only.
    pub backcast_w: Option<Vec<f32>>,
    /// Backcast head bias `[t*c]` — Generic only.
    pub backcast_b: Option<Vec<f32>>,
    /// Forecast head weight `[horizon*c × theta_half]` — Generic only.
    pub forecast_w: Option<Vec<f32>>,
    /// Forecast head bias `[horizon*c]` — Generic only.
    pub forecast_b: Option<Vec<f32>>,

    // ── Precomputed bases (None for Generic) ─────────────────────────────────
    /// Backcast basis `[t × basis_cols]` — Trend / Seasonality.
    pub backcast_basis: Option<Vec<f32>>,
    /// Forecast basis `[horizon × basis_cols]` — Trend / Seasonality.
    pub forecast_basis: Option<Vec<f32>>,

    // ── Config fields ─────────────────────────────────────────────────────────
    /// Block variant.
    pub block_type: NBeatsBlockType,
    /// Input sequence length.
    pub t: usize,
    /// Number of channels.
    pub c: usize,
    /// Forecast horizon.
    pub horizon: usize,
    /// MLP hidden units.
    pub hidden_units: usize,
    /// Total MLP output width (`theta_size = theta_half * 2`).
    pub theta_size: usize,
    /// Number of basis columns per channel (degree+1 or 2*n_harmonics).
    pub basis_cols: usize,
}

impl NBeatsBlock {
    /// Construct an `NBeatsBlock` with Xavier-initialised weights.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidSequenceLength`] when `t == 0`.
    /// - [`TsError::InvalidHorizon`] when `horizon == 0`.
    /// - [`TsError::InvalidNumVariates`] when `c == 0`.
    /// - [`TsError::ShapeMismatch`] for invalid block-specific config.
    pub fn new(
        t: usize,
        c: usize,
        horizon: usize,
        block_type: NBeatsBlockType,
        rng: &mut LcgRng,
    ) -> TsResult<Self> {
        if t == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        if horizon == 0 {
            return Err(TsError::InvalidHorizon(0));
        }
        if c == 0 {
            return Err(TsError::InvalidNumVariates(0));
        }

        let hidden_units = block_type.hidden_units();
        let flat_in = t * c;

        // Compute theta_size and basis_cols depending on block type; also validate.
        let (theta_half, basis_cols) = match &block_type {
            NBeatsBlockType::Generic { theta_dim, .. } => {
                if *theta_dim == 0 {
                    return Err(TsError::ShapeMismatch {
                        msg: "theta_dim must be > 0 for Generic block".to_string(),
                    });
                }
                (*theta_dim, *theta_dim) // basis_cols == theta_dim (no structured basis)
            }
            NBeatsBlockType::Trend { degree, .. } => {
                if *degree > t / 2 {
                    return Err(TsError::ShapeMismatch {
                        msg: format!("degree {degree} too large for t={t} (max t/2={})", t / 2),
                    });
                }
                let poly_cols = degree + 1;
                (poly_cols * c, poly_cols)
            }
            NBeatsBlockType::Seasonality { n_harmonics, .. } => {
                if *n_harmonics == 0 {
                    return Err(TsError::ShapeMismatch {
                        msg: "n_harmonics must be > 0 for Seasonality block".to_string(),
                    });
                }
                let fourier_cols = 2 * n_harmonics;
                (fourier_cols * c, fourier_cols)
            }
        };

        let theta_size = theta_half * 2;

        // ── MLP weights ──────────────────────────────────────────────────────
        let mut mlp_w1 = vec![0.0_f32; hidden_units * flat_in];
        xavier_fill(&mut mlp_w1, flat_in, hidden_units, rng);
        let mlp_b1 = vec![0.0_f32; hidden_units];

        let mut mlp_w2 = vec![0.0_f32; hidden_units * hidden_units];
        xavier_fill(&mut mlp_w2, hidden_units, hidden_units, rng);
        let mlp_b2 = vec![0.0_f32; hidden_units];

        let mut mlp_w3 = vec![0.0_f32; hidden_units * hidden_units];
        xavier_fill(&mut mlp_w3, hidden_units, hidden_units, rng);
        let mlp_b3 = vec![0.0_f32; hidden_units];

        let mut mlp_w4 = vec![0.0_f32; theta_size * hidden_units];
        xavier_fill(&mut mlp_w4, hidden_units, theta_size, rng);
        let mlp_b4 = vec![0.0_f32; theta_size];

        // ── Block-type-specific heads / bases ────────────────────────────────
        let (backcast_w, backcast_b, forecast_w, forecast_b, backcast_basis, forecast_basis) =
            match &block_type {
                NBeatsBlockType::Generic { theta_dim, .. } => {
                    let td = *theta_dim;
                    let mut bw = vec![0.0_f32; flat_in * td];
                    xavier_fill(&mut bw, td, flat_in, rng);
                    let bb = vec![0.0_f32; flat_in];

                    let mut fw = vec![0.0_f32; horizon * c * td];
                    xavier_fill(&mut fw, td, horizon * c, rng);
                    let fb = vec![0.0_f32; horizon * c];

                    (Some(bw), Some(bb), Some(fw), Some(fb), None, None)
                }
                NBeatsBlockType::Trend { degree, .. } => {
                    let poly_cols = degree + 1;

                    // Backcast basis: [t × poly_cols], time normalised to [-1, 1].
                    let mut bb = vec![0.0_f32; t * poly_cols];
                    for ti in 0..t {
                        let tau = if t > 1 {
                            2.0 * (ti as f32) / ((t - 1) as f32) - 1.0
                        } else {
                            0.0
                        };
                        for k in 0..poly_cols {
                            bb[ti * poly_cols + k] = tau.powi(k as i32);
                        }
                    }

                    // Forecast basis: [horizon × poly_cols], continuing time index.
                    let total_len = t + horizon;
                    let mut fb = vec![0.0_f32; horizon * poly_cols];
                    for h in 0..horizon {
                        let idx = t + h;
                        let tau = if total_len > 1 {
                            2.0 * (idx as f32) / ((total_len - 1) as f32) - 1.0
                        } else {
                            0.0
                        };
                        for k in 0..poly_cols {
                            fb[h * poly_cols + k] = tau.powi(k as i32);
                        }
                    }

                    (None, None, None, None, Some(bb), Some(fb))
                }
                NBeatsBlockType::Seasonality { n_harmonics, .. } => {
                    let fourier_cols = 2 * n_harmonics; // cos cols then sin cols

                    // Backcast basis: [t × fourier_cols].
                    let mut bb = vec![0.0_f32; t * fourier_cols];
                    for ti in 0..t {
                        for k in 1..=*n_harmonics {
                            let angle =
                                2.0 * std::f32::consts::PI * (k as f32) * (ti as f32) / (t as f32);
                            bb[ti * fourier_cols + (k - 1)] = angle.cos();
                            bb[ti * fourier_cols + n_harmonics + (k - 1)] = angle.sin();
                        }
                    }

                    // Forecast basis: [horizon × fourier_cols], time idx = T + h.
                    let mut fb = vec![0.0_f32; horizon * fourier_cols];
                    for h in 0..horizon {
                        let ti = t + h;
                        for k in 1..=*n_harmonics {
                            let angle =
                                2.0 * std::f32::consts::PI * (k as f32) * (ti as f32) / (t as f32);
                            fb[h * fourier_cols + (k - 1)] = angle.cos();
                            fb[h * fourier_cols + n_harmonics + (k - 1)] = angle.sin();
                        }
                    }

                    (None, None, None, None, Some(bb), Some(fb))
                }
            };

        Ok(Self {
            mlp_w1,
            mlp_b1,
            mlp_w2,
            mlp_b2,
            mlp_w3,
            mlp_b3,
            mlp_w4,
            mlp_b4,
            backcast_w,
            backcast_b,
            forecast_w,
            forecast_b,
            backcast_basis,
            forecast_basis,
            block_type,
            t,
            c,
            horizon,
            hidden_units,
            theta_size,
            basis_cols,
        })
    }

    /// Forward pass through the N-BEATS block.
    ///
    /// # Returns
    ///
    /// `(backcast, forecast)` where backcast has length `t * c` and forecast
    /// has length `horizon * c`.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != t * c`.
    pub fn forward(&self, x: &[f32]) -> TsResult<(Vec<f32>, Vec<f32>)> {
        let expected = self.t * self.c;
        if x.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        // ── 4-layer MLP ──────────────────────────────────────────────────────
        let mut h = linear(
            x,
            &self.mlp_w1,
            &self.mlp_b1,
            self.t * self.c,
            self.hidden_units,
        );
        relu_vec(&mut h);

        h = linear(
            &h,
            &self.mlp_w2,
            &self.mlp_b2,
            self.hidden_units,
            self.hidden_units,
        );
        relu_vec(&mut h);

        h = linear(
            &h,
            &self.mlp_w3,
            &self.mlp_b3,
            self.hidden_units,
            self.hidden_units,
        );
        relu_vec(&mut h);

        let theta = linear(
            &h,
            &self.mlp_w4,
            &self.mlp_b4,
            self.hidden_units,
            self.theta_size,
        );

        let theta_half = self.theta_size / 2;
        let theta_b = &theta[..theta_half];
        let theta_f = &theta[theta_half..];

        // ── Expand to backcast / forecast ────────────────────────────────────
        let (backcast, forecast) = match &self.block_type {
            NBeatsBlockType::Generic { .. } => {
                let bw = self.backcast_w.as_deref().unwrap_or(&[]);
                let bb = self.backcast_b.as_deref().unwrap_or(&[]);
                let fw = self.forecast_w.as_deref().unwrap_or(&[]);
                let fb = self.forecast_b.as_deref().unwrap_or(&[]);

                let backcast = linear(theta_b, bw, bb, theta_half, self.t * self.c);
                let forecast = linear(theta_f, fw, fb, theta_half, self.horizon * self.c);
                (backcast, forecast)
            }
            NBeatsBlockType::Trend { .. } | NBeatsBlockType::Seasonality { .. } => {
                // theta_b: [basis_cols * c] → reshape to [basis_cols, c]
                // backcast = basis_b @ theta_b_reshaped → [t, c] → flatten
                let basis_b = self.backcast_basis.as_deref().unwrap_or(&[]);
                let basis_f = self.forecast_basis.as_deref().unwrap_or(&[]);

                let backcast = basis_expand(basis_b, theta_b, self.t, self.c, self.basis_cols);
                let forecast =
                    basis_expand(basis_f, theta_f, self.horizon, self.c, self.basis_cols);
                (backcast, forecast)
            }
        };

        Ok((backcast, forecast))
    }
}

/// Expand via basis: `out[time, ci] = Σ_k basis[time, k] * theta[k, ci]`.
///
/// - `basis`: `[time_steps × basis_cols]` row-major.
/// - `theta_flat`: `[basis_cols * c]` — theta reshaped as `[basis_cols, c]`.
/// - Returns `[time_steps * c]` row-major.
fn basis_expand(
    basis: &[f32],
    theta_flat: &[f32],
    time_steps: usize,
    c: usize,
    basis_cols: usize,
) -> Vec<f32> {
    // theta_flat is laid out as [basis_cols * c] with innermost dim = c.
    // theta[k, ci] = theta_flat[k * c + ci]
    let mut out = vec![0.0_f32; time_steps * c];
    for ts in 0..time_steps {
        for ci in 0..c {
            let mut acc = 0.0_f32;
            for k in 0..basis_cols {
                let b_val = basis[ts * basis_cols + k];
                let t_val = theta_flat[k * c + ci];
                acc += b_val * t_val;
            }
            out[ts * c + ci] = acc;
        }
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ── Generic block ──────────────────────────────────────────────────────────

    #[test]
    fn generic_block_new_and_forward_shape() {
        let mut rng = make_rng();
        let t = 16;
        let c = 2;
        let horizon = 4;
        let block = NBeatsBlock::new(
            t,
            c,
            horizon,
            NBeatsBlockType::Generic {
                hidden_units: 32,
                theta_dim: 8,
            },
            &mut rng,
        )
        .expect("ok");
        let x = vec![0.5_f32; t * c];
        let (bc, fc) = block.forward(&x).expect("ok");
        assert_eq!(bc.len(), t * c);
        assert_eq!(fc.len(), horizon * c);
    }

    #[test]
    fn generic_forward_finite() {
        let mut rng = make_rng();
        let t = 20;
        let c = 3;
        let horizon = 6;
        let block = NBeatsBlock::new(
            t,
            c,
            horizon,
            NBeatsBlockType::Generic {
                hidden_units: 32,
                theta_dim: 10,
            },
            &mut rng,
        )
        .expect("ok");
        let mut x = vec![0.0_f32; t * c];
        rng.fill_normal(&mut x);
        let (bc, fc) = block.forward(&x).expect("ok");
        assert!(bc.iter().all(|v| v.is_finite()), "backcast non-finite");
        assert!(fc.iter().all(|v| v.is_finite()), "forecast non-finite");
    }

    #[test]
    fn zero_input_finite_generic() {
        let mut rng = make_rng();
        let block = NBeatsBlock::new(
            12,
            2,
            4,
            NBeatsBlockType::Generic {
                hidden_units: 16,
                theta_dim: 4,
            },
            &mut rng,
        )
        .expect("ok");
        let x = vec![0.0_f32; 12 * 2];
        let (bc, fc) = block.forward(&x).expect("ok");
        assert!(bc.iter().all(|v| v.is_finite()));
        assert!(fc.iter().all(|v| v.is_finite()));
    }

    // ── Trend block ────────────────────────────────────────────────────────────

    #[test]
    fn trend_block_new_and_forward_shape() {
        let mut rng = make_rng();
        let t = 24;
        let c = 2;
        let horizon = 8;
        let block = NBeatsBlock::new(
            t,
            c,
            horizon,
            NBeatsBlockType::Trend {
                hidden_units: 32,
                degree: 2,
            },
            &mut rng,
        )
        .expect("ok");
        let x = vec![0.5_f32; t * c];
        let (bc, fc) = block.forward(&x).expect("ok");
        assert_eq!(bc.len(), t * c);
        assert_eq!(fc.len(), horizon * c);
    }

    #[test]
    fn trend_forward_finite() {
        let mut rng = make_rng();
        let t = 24;
        let c = 2;
        let horizon = 6;
        let block = NBeatsBlock::new(
            t,
            c,
            horizon,
            NBeatsBlockType::Trend {
                hidden_units: 32,
                degree: 3,
            },
            &mut rng,
        )
        .expect("ok");
        let mut x = vec![0.0_f32; t * c];
        rng.fill_normal(&mut x);
        let (bc, fc) = block.forward(&x).expect("ok");
        assert!(bc.iter().all(|v| v.is_finite()), "backcast non-finite");
        assert!(fc.iter().all(|v| v.is_finite()), "forecast non-finite");
    }

    #[test]
    fn trend_basis_normalized_time() {
        // The backcast basis for a trend block must have time points in [-1, 1] for degree=1.
        let mut rng = make_rng();
        let t = 10;
        let block = NBeatsBlock::new(
            t,
            1,
            4,
            NBeatsBlockType::Trend {
                hidden_units: 16,
                degree: 1,
            },
            &mut rng,
        )
        .expect("ok");
        let basis = block.backcast_basis.as_ref().expect("trend has basis");
        // Column index 1 contains the linear time (tau), shape [t × 2].
        let basis_cols = 2; // degree=1 → degree+1=2 cols
        for ti in 0..t {
            let tau = basis[ti * basis_cols + 1]; // index 1 = tau^1
            assert!(
                (-1.0 - 1e-5..=1.0 + 1e-5).contains(&tau),
                "tau[{ti}] = {tau} not in [-1, 1]"
            );
        }
        // First and last should be -1 and 1.
        assert!((basis[1] - (-1.0_f32)).abs() < 1e-5);
        assert!((basis[(t - 1) * basis_cols + 1] - 1.0_f32).abs() < 1e-5);
    }

    // ── Seasonality block ──────────────────────────────────────────────────────

    #[test]
    fn seasonality_block_new_and_forward_shape() {
        let mut rng = make_rng();
        let t = 24;
        let c = 2;
        let horizon = 8;
        let block = NBeatsBlock::new(
            t,
            c,
            horizon,
            NBeatsBlockType::Seasonality {
                hidden_units: 32,
                n_harmonics: 3,
            },
            &mut rng,
        )
        .expect("ok");
        let x = vec![0.5_f32; t * c];
        let (bc, fc) = block.forward(&x).expect("ok");
        assert_eq!(bc.len(), t * c);
        assert_eq!(fc.len(), horizon * c);
    }

    #[test]
    fn seasonality_forward_finite() {
        let mut rng = make_rng();
        let t = 24;
        let c = 2;
        let horizon = 8;
        let block = NBeatsBlock::new(
            t,
            c,
            horizon,
            NBeatsBlockType::Seasonality {
                hidden_units: 32,
                n_harmonics: 4,
            },
            &mut rng,
        )
        .expect("ok");
        let mut x = vec![0.0_f32; t * c];
        rng.fill_normal(&mut x);
        let (bc, fc) = block.forward(&x).expect("ok");
        assert!(bc.iter().all(|v| v.is_finite()), "backcast non-finite");
        assert!(fc.iter().all(|v| v.is_finite()), "forecast non-finite");
    }

    #[test]
    fn seasonality_basis_unit_circle() {
        // cos²(θ) + sin²(θ) ≈ 1 for harmonic k=1 at every time step.
        let mut rng = make_rng();
        let t = 12;
        let n_harmonics = 3;
        let block = NBeatsBlock::new(
            t,
            1,
            4,
            NBeatsBlockType::Seasonality {
                hidden_units: 16,
                n_harmonics,
            },
            &mut rng,
        )
        .expect("ok");
        let basis = block.backcast_basis.as_ref().expect("season has basis");
        let fourier_cols = 2 * n_harmonics;
        for ti in 0..t {
            // k=0 (harmonic 1): cos in col 0, sin in col n_harmonics
            let cos_val = basis[ti * fourier_cols];
            let sin_val = basis[ti * fourier_cols + n_harmonics];
            let mag_sq = cos_val * cos_val + sin_val * sin_val;
            assert!(
                (mag_sq - 1.0).abs() < 1e-5,
                "unit circle violated at ti={ti}: cos²+sin²={mag_sq}"
            );
        }
    }

    // ── Multi-channel ──────────────────────────────────────────────────────────

    #[test]
    fn multi_channel_forward_shape() {
        let mut rng = make_rng();
        let t = 16;
        let c = 3;
        let horizon = 5;
        let block = NBeatsBlock::new(
            t,
            c,
            horizon,
            NBeatsBlockType::Generic {
                hidden_units: 32,
                theta_dim: 6,
            },
            &mut rng,
        )
        .expect("ok");
        let x = vec![1.0_f32; t * c];
        let (bc, fc) = block.forward(&x).expect("ok");
        assert_eq!(bc.len(), t * c);
        assert_eq!(fc.len(), horizon * c);
    }

    // ── Error cases ────────────────────────────────────────────────────────────

    #[test]
    fn nbeats_block_err_invalid_horizon() {
        let mut rng = make_rng();
        assert!(matches!(
            NBeatsBlock::new(
                16,
                2,
                0,
                NBeatsBlockType::Generic {
                    hidden_units: 16,
                    theta_dim: 4
                },
                &mut rng
            )
            .unwrap_err(),
            TsError::InvalidHorizon(0)
        ));
    }

    #[test]
    fn nbeats_block_err_dimension_mismatch() {
        let mut rng = make_rng();
        let block = NBeatsBlock::new(
            16,
            2,
            4,
            NBeatsBlockType::Generic {
                hidden_units: 16,
                theta_dim: 4,
            },
            &mut rng,
        )
        .expect("ok");
        assert!(matches!(
            block.forward(&[0.0_f32; 5]).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn nbeats_block_err_degree_too_large() {
        let mut rng = make_rng();
        assert!(matches!(
            NBeatsBlock::new(
                4,
                1,
                2,
                NBeatsBlockType::Trend {
                    hidden_units: 16,
                    degree: 5
                }, // degree=5 > t/2=2
                &mut rng
            )
            .unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn nbeats_block_err_zero_harmonics() {
        let mut rng = make_rng();
        assert!(matches!(
            NBeatsBlock::new(
                16,
                2,
                4,
                NBeatsBlockType::Seasonality {
                    hidden_units: 16,
                    n_harmonics: 0
                },
                &mut rng
            )
            .unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn nbeats_block_err_zero_theta_dim() {
        let mut rng = make_rng();
        assert!(matches!(
            NBeatsBlock::new(
                16,
                2,
                4,
                NBeatsBlockType::Generic {
                    hidden_units: 16,
                    theta_dim: 0
                },
                &mut rng
            )
            .unwrap_err(),
            TsError::ShapeMismatch { .. }
        ));
    }
}
