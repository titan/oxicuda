//! LST (Ladder Side-Tuning) — Sung-Cho-Bansal 2022 NeurIPS.
//!
//! Reference: Sung Y-L, Cho J, Bansal M (2022) "LST: Ladder Side-Tuning for
//! Parameter and Memory Efficient Transfer Learning", NeurIPS 2022.
//! <https://arxiv.org/abs/2206.06522>
//!
//! Architecture:
//! ```text
//!   side_state_t+1 = side_state_t + side_w · GELU(down_w · trunk_hidden_t + down_b) + side_b
//!   output_t = gate · trunk_final_t + (1 − gate) · (up_w · side_state_t + up_b)
//! ```
//!
//! Key properties:
//! - No back-propagation through the frozen trunk hidden states.
//! - Each layer has an independent side bottleneck: `down → GELU → side_residual`.
//! - The final output blends trunk and side representations via a learned gate `α`.
//! - `down_w` and `side_w` use Kaiming-uniform init; `up_w` and all biases are zero.

use crate::adapter::houlsby::gelu;
use crate::error::{PeftError, PeftResult};
use crate::handle::PeftHandle;

/// Configuration for [`LadderSideTuning::new`].
#[derive(Debug, Clone)]
pub struct LstConfig {
    /// Dimension of the frozen trunk's hidden states.
    pub d_trunk: usize,
    /// Dimension of the side-network bottleneck.
    pub d_side: usize,
    /// Number of transformer layers to attach side blocks to.
    pub num_layers: usize,
    /// Initial value of the gate scalar `α` for each block (e.g. 0.5).
    pub gate_init: f32,
}

/// One per-layer side bottleneck for [`LadderSideTuning`].
///
/// Fields are `pub(crate)` to keep the serialization surface narrow; tests
/// access them through [`LadderSideTuning::total_params`].
#[derive(Debug, Clone)]
pub struct LstBlock {
    /// Down projection weight, shape `d_side × d_trunk` (row-major).
    pub(crate) down_w: Vec<f32>,
    /// Down projection bias, shape `d_side`.
    pub(crate) down_b: Vec<f32>,
    /// Side residual weight, shape `d_side × d_side` (row-major).
    pub(crate) side_w: Vec<f32>,
    /// Side residual bias, shape `d_side`.
    pub(crate) side_b: Vec<f32>,
    /// Up projection weight, shape `d_trunk × d_side` (row-major).
    pub(crate) up_w: Vec<f32>,
    /// Up projection bias, shape `d_trunk`.
    pub(crate) up_b: Vec<f32>,
    /// Gate scalar `α`; output = `α · trunk + (1 − α) · up(side)`.
    pub(crate) gate: f32,
}

/// Ladder Side-Tuning module.
///
/// Maintains one [`LstBlock`] per transformer layer. During forward, the
/// caller passes the frozen trunk's hidden state and a running side state;
/// LST updates the side state without any gradient flowing back through the
/// trunk.
#[derive(Debug)]
pub struct LadderSideTuning {
    /// Per-layer side blocks.
    pub(crate) layers: Vec<LstBlock>,
    /// Configuration captured at construction time.
    pub cfg: LstConfig,
}

impl LadderSideTuning {
    /// Construct a new `LadderSideTuning` with Kaiming-uniform `down_w` /
    /// `side_w` and zero-initialized `up_w` and all biases.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::InvalidTargetRank`] if `num_layers == 0`,
    /// `d_side == 0`, or `d_trunk == 0`.
    pub fn new(cfg: LstConfig, handle: &mut PeftHandle) -> PeftResult<Self> {
        if cfg.num_layers == 0 {
            return Err(PeftError::InvalidTargetRank { target_r: 0, r: 1 });
        }
        if cfg.d_side == 0 {
            return Err(PeftError::InvalidTargetRank { target_r: 0, r: 1 });
        }
        if cfg.d_trunk == 0 {
            return Err(PeftError::InvalidTargetRank { target_r: 0, r: 1 });
        }

        let d_trunk = cfg.d_trunk;
        let d_side = cfg.d_side;

        // Kaiming-uniform limits.
        // down_w: fan_in = d_trunk, fan_out = d_side → limit = sqrt(6/(d_trunk+d_side))
        let down_limit = (6.0_f32 / (d_trunk + d_side) as f32).sqrt();
        // side_w: fan_in = fan_out = d_side → limit = sqrt(6/(2*d_side))
        let side_limit = (6.0_f32 / (2 * d_side) as f32).sqrt();

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            let down_w = kaiming_uniform(d_side * d_trunk, down_limit, handle);
            let down_b = vec![0.0_f32; d_side];

            let side_w = kaiming_uniform(d_side * d_side, side_limit, handle);
            let side_b = vec![0.0_f32; d_side];

            // up_w is zero-initialized (near-identity start for the residual path).
            let up_w = vec![0.0_f32; d_trunk * d_side];
            let up_b = vec![0.0_f32; d_trunk];

            layers.push(LstBlock {
                down_w,
                down_b,
                side_w,
                side_b,
                up_w,
                up_b,
                gate: cfg.gate_init,
            });
        }

        Ok(Self { layers, cfg })
    }

    /// Forward one layer of the LST network.
    ///
    /// Takes the frozen trunk hidden state and the running side state for a
    /// sequence of `seq_len` tokens and returns the updated side state.
    ///
    /// Input shapes (flat, row-major):
    /// - `trunk_hidden`: `seq_len × d_trunk`
    /// - `side_state`:  `seq_len × d_side`
    ///
    /// Output: `seq_len × d_side` (new side state).
    ///
    /// # Errors
    ///
    /// - [`PeftError::LayerOutOfRange`] if `layer >= num_layers`.
    /// - [`PeftError::DimensionMismatch`] if slice lengths disagree with the
    ///   configuration.
    pub fn forward_layer(
        &self,
        layer: usize,
        trunk_hidden: &[f32],
        side_state: &[f32],
        seq_len: usize,
    ) -> PeftResult<Vec<f32>> {
        if layer >= self.cfg.num_layers {
            return Err(PeftError::LayerOutOfRange {
                idx: layer,
                num_layers: self.cfg.num_layers,
            });
        }

        let d_trunk = self.cfg.d_trunk;
        let d_side = self.cfg.d_side;

        if trunk_hidden.len() != seq_len * d_trunk {
            return Err(PeftError::DimensionMismatch {
                expected: seq_len * d_trunk,
                got: trunk_hidden.len(),
            });
        }
        if side_state.len() != seq_len * d_side {
            return Err(PeftError::DimensionMismatch {
                expected: seq_len * d_side,
                got: side_state.len(),
            });
        }

        let blk = &self.layers[layer];
        let mut new_side = vec![0.0_f32; seq_len * d_side];

        for t in 0..seq_len {
            let h = &trunk_hidden[t * d_trunk..(t + 1) * d_trunk];
            let s = &side_state[t * d_side..(t + 1) * d_side];
            let out_slice = &mut new_side[t * d_side..(t + 1) * d_side];

            // Down projection: down_out[j] = Σ_i down_w[j*d_trunk+i] * h[i] + down_b[j]
            // Then apply GELU element-wise.
            let down_out: Vec<f32> = blk
                .down_b
                .iter()
                .enumerate()
                .map(|(j, &bias)| {
                    let row_offset = j * d_trunk;
                    let acc = bias
                        + blk.down_w[row_offset..row_offset + d_trunk]
                            .iter()
                            .zip(h.iter())
                            .map(|(&w, &hi)| w * hi)
                            .sum::<f32>();
                    gelu(acc)
                })
                .collect();

            // Side residual: new_s[j] = s[j] + Σ_i side_w[j*d_side+i] * down_out[i] + side_b[j]
            for (j, slot) in out_slice.iter_mut().enumerate() {
                let row_offset = j * d_side;
                let acc = s[j]
                    + blk.side_b[j]
                    + blk.side_w[row_offset..row_offset + d_side]
                        .iter()
                        .zip(down_out.iter())
                        .map(|(&w, &d)| w * d)
                        .sum::<f32>();
                *slot = acc;
            }
        }

        Ok(new_side)
    }

    /// Compute the final gated output combining the last side state with the
    /// trunk's final hidden states.
    ///
    /// ```text
    /// out[t][j] = gate · trunk_final[t][j] + (1 − gate) · (Σ_i up_w[j*d_side+i] · side_state[t][i] + up_b[j])
    /// ```
    ///
    /// Returns a flat vector of shape `seq_len × d_trunk`.
    ///
    /// # Errors
    ///
    /// - [`PeftError::DimensionMismatch`] if slice lengths disagree with the
    ///   configuration.
    pub fn final_output(
        &self,
        side_state: &[f32],
        trunk_final: &[f32],
        seq_len: usize,
    ) -> PeftResult<Vec<f32>> {
        let d_trunk = self.cfg.d_trunk;
        let d_side = self.cfg.d_side;

        if side_state.len() != seq_len * d_side {
            return Err(PeftError::DimensionMismatch {
                expected: seq_len * d_side,
                got: side_state.len(),
            });
        }
        if trunk_final.len() != seq_len * d_trunk {
            return Err(PeftError::DimensionMismatch {
                expected: seq_len * d_trunk,
                got: trunk_final.len(),
            });
        }

        // Use the last layer's gate (or cfg.gate_init when num_layers == 0,
        // though new() rejects that; this is just a safe fallback).
        let (gate, up_w, up_b) = if let Some(last) = self.layers.last() {
            (last.gate, &last.up_w, &last.up_b)
        } else {
            return Err(PeftError::Internal {
                msg: "LadderSideTuning has no layers".to_string(),
            });
        };
        let one_minus_gate = 1.0_f32 - gate;

        let mut out = vec![0.0_f32; seq_len * d_trunk];

        for t in 0..seq_len {
            let s = &side_state[t * d_side..(t + 1) * d_side];
            let tf = &trunk_final[t * d_trunk..(t + 1) * d_trunk];
            let out_slice = &mut out[t * d_trunk..(t + 1) * d_trunk];

            for j in 0..d_trunk {
                let row_offset = j * d_side;
                let mut up_val = up_b[j];
                for (i, &si) in s.iter().enumerate().take(d_side) {
                    up_val += up_w[row_offset + i] * si;
                }
                out_slice[j] = gate * tf[j] + one_minus_gate * up_val;
            }
        }

        Ok(out)
    }

    /// Count the total number of trainable parameters across all layers.
    ///
    /// Per block: `d_side*d_trunk + d_side + d_side*d_side + d_side + d_trunk*d_side + d_trunk + 1`
    #[must_use]
    pub fn total_params(&self) -> usize {
        let d_trunk = self.cfg.d_trunk;
        let d_side = self.cfg.d_side;
        let per_block = d_side * d_trunk  // down_w
            + d_side                       // down_b
            + d_side * d_side              // side_w
            + d_side                       // side_b
            + d_trunk * d_side             // up_w
            + d_trunk                      // up_b
            + 1; // gate
        per_block * self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// internal helpers
// ---------------------------------------------------------------------------

/// Fill a `Vec<f32>` of length `n` with i.i.d. samples from `U(−limit, +limit)`.
fn kaiming_uniform(n: usize, limit: f32, handle: &mut PeftHandle) -> Vec<f32> {
    let two_limit = 2.0 * limit;
    let mut out = vec![0.0_f32; n];
    for slot in out.iter_mut() {
        let u = handle.rng.next_f32(); // ∈ [0, 1)
        *slot = u * two_limit - limit;
    }
    out
}
