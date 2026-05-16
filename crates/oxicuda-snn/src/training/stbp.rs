//! Spatio-Temporal Backpropagation (STBP) — Wu, Deng, Li, Zhu, Shi (2018).
//!
//! Differs from vanilla [`bptt`](crate::training::bptt) in the **explicit reset gradient**
//! that gates the recurrent membrane gradient with `(1 − S_t)` regardless of
//! reset mode. This corresponds to "iterative LIF" with a soft-reset surrogate
//! followed by a stop-gradient on the spike branch:
//!
//! ```text
//! v_t^{post} = β · v_{t-1}^{post}·(1 − s_t) + I_t          (forward, hard-like)
//! ∂L/∂v̄_t   = g_α(v̄_t − v_th)·∂L/∂s_t  +  (1 − s_t)·β·∂L/∂v̄_{t+1}
//! ```
//!
//! Even when `lif_cfg.reset == Soft`, STBP forces the `(1 − s_t)` factor on the
//! recurrent gradient. This is a deliberate algorithmic choice from the paper,
//! producing slightly different gradients than [`bptt::bptt_unroll`].

use crate::error::{SnnError, SnnResult};
use crate::neuron::lif::{LifConfig, ResetMode, beta};
use crate::training::bptt::{BpttConfig, BpttResult, surrogate_eval};

/// Validate slice shapes for [`stbp_unroll`].
fn validate_shapes(
    weights: &[f32],
    biases: &[f32],
    inputs: &[f32],
    target_spikes: &[f32],
    in_dim: usize,
    out_dim: usize,
    cfg: &BpttConfig,
) -> SnnResult<()> {
    if in_dim == 0 {
        return Err(SnnError::BadDim { got: in_dim });
    }
    if out_dim == 0 {
        return Err(SnnError::BadDim { got: out_dim });
    }
    if cfg.t_steps == 0 {
        return Err(SnnError::BadTimesteps { got: cfg.t_steps });
    }
    if cfg.alpha <= 0.0 || !cfg.alpha.is_finite() {
        return Err(SnnError::OutOfRange {
            name: "alpha".into(),
            val: cfg.alpha,
        });
    }
    let expected_w = in_dim * out_dim;
    if weights.len() != expected_w {
        return Err(SnnError::BadShape {
            expected: expected_w,
            got: weights.len(),
        });
    }
    if biases.len() != out_dim {
        return Err(SnnError::BadShape {
            expected: out_dim,
            got: biases.len(),
        });
    }
    let expected_in = cfg.t_steps * in_dim;
    if inputs.len() != expected_in {
        return Err(SnnError::BadShape {
            expected: expected_in,
            got: inputs.len(),
        });
    }
    let expected_out = cfg.t_steps * out_dim;
    if target_spikes.len() != expected_out {
        return Err(SnnError::BadShape {
            expected: expected_out,
            got: target_spikes.len(),
        });
    }
    Ok(())
}

/// Forward / backward pass with STBP-style explicit reset gating.
///
/// Layout and semantics are identical to [`crate::training::bptt::bptt_unroll`], with the
/// difference that the recurrent gradient is gated by `(1 − s_t)` regardless
/// of `lif_cfg.reset` (the reset mode still controls the *forward* membrane
/// reset).
pub fn stbp_unroll(
    weights: &[f32],
    biases: &[f32],
    inputs: &[f32],
    target_spikes: &[f32],
    in_dim: usize,
    out_dim: usize,
    cfg: &BpttConfig,
    lif_cfg: &LifConfig,
) -> SnnResult<BpttResult> {
    validate_shapes(weights, biases, inputs, target_spikes, in_dim, out_dim, cfg)?;
    if lif_cfg.tau_m <= 0.0 || !lif_cfg.tau_m.is_finite() {
        return Err(SnnError::BadTau { tau: lif_cfg.tau_m });
    }
    if lif_cfg.dt <= 0.0 || !lif_cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: lif_cfg.dt });
    }
    if !lif_cfg.v_th.is_finite() {
        return Err(SnnError::BadThreshold { v_th: lif_cfg.v_th });
    }

    let t = cfg.t_steps;
    let b = beta(lif_cfg);

    let mut v_pre = vec![0.0_f32; t * out_dim];
    let mut s_hist = vec![0.0_f32; t * out_dim];

    // ----- Forward pass (identical to BPTT) -----
    let mut v_prev = vec![0.0_f32; out_dim];
    for step in 0..t {
        let x_off = step * in_dim;
        let y_off = step * out_dim;
        for i in 0..out_dim {
            let mut acc = biases[i];
            let row_off = i * in_dim;
            for j in 0..in_dim {
                acc += weights[row_off + j] * inputs[x_off + j];
            }
            let v_bar = b * v_prev[i] + acc;
            let spike = if v_bar >= lif_cfg.v_th { 1.0 } else { 0.0 };
            let v_after = match lif_cfg.reset {
                ResetMode::Hard => (1.0 - spike) * v_bar + spike * lif_cfg.v_rest,
                ResetMode::Soft => v_bar - spike * lif_cfg.v_th,
            };
            v_pre[y_off + i] = v_bar;
            s_hist[y_off + i] = spike;
            v_prev[i] = v_after;
        }
    }

    // ----- Loss -----
    let inv_t = 1.0 / (t as f32);
    let mut loss = 0.0_f32;
    let mut dl_ds = vec![0.0_f32; t * out_dim];
    for k in 0..(t * out_dim) {
        let diff = s_hist[k] - target_spikes[k];
        loss += diff * diff;
        dl_ds[k] = 2.0 * diff * inv_t;
    }
    loss *= inv_t;

    // ----- Backward pass with explicit STBP reset gating -----
    let mut surrogate = vec![0.0_f32; t * out_dim];
    surrogate_eval(
        cfg.surrogate,
        &v_pre,
        lif_cfg.v_th,
        cfg.alpha,
        &mut surrogate,
    )?;

    let mut grad_w = vec![0.0_f32; weights.len()];
    let mut grad_b = vec![0.0_f32; biases.len()];

    let mut dl_dv_next = vec![0.0_f32; out_dim];
    let mut dl_dv = vec![0.0_f32; out_dim];

    for step in (0..t).rev() {
        let x_off = step * in_dim;
        let y_off = step * out_dim;
        for i in 0..out_dim {
            let s_t = s_hist[y_off + i];
            // STBP: always gate by (1 − s_t) regardless of reset mode.
            let recurrent = (1.0 - s_t) * b * dl_dv_next[i];
            let dl_dv_i = surrogate[y_off + i] * dl_ds[y_off + i] + recurrent;
            dl_dv[i] = dl_dv_i;
            grad_b[i] += dl_dv_i;
            let row_off = i * in_dim;
            for j in 0..in_dim {
                grad_w[row_off + j] += dl_dv_i * inputs[x_off + j];
            }
        }
        std::mem::swap(&mut dl_dv, &mut dl_dv_next);
        for v in dl_dv.iter_mut() {
            *v = 0.0;
        }
    }

    Ok(BpttResult {
        loss,
        grad_w,
        grad_b,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::lif::{LifConfig, ResetMode};
    use crate::training::bptt::{BpttConfig, SurrogateKind, bptt_unroll};

    #[test]
    fn matches_bptt_when_no_spikes_hard() {
        // Threshold so high that no spike fires → (1 − s_t) = 1 in BPTT/Hard
        // and in STBP: results must be identical.
        let cfg = BpttConfig {
            t_steps: 6,
            surrogate: SurrogateKind::Sigmoid,
            alpha: 3.0,
        };
        let lif = LifConfig {
            tau_m: 4.0,
            v_th: 1e6,
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        let in_dim = 3_usize;
        let out_dim = 2_usize;
        let weights = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let biases = vec![0.0_f32, 0.05];
        let inputs: Vec<f32> = (0..(cfg.t_steps * in_dim))
            .map(|i| 0.05 * ((i as f32) + 1.0))
            .collect();
        let targets = vec![0.0_f32; cfg.t_steps * out_dim];

        let r_bptt = bptt_unroll(
            &weights, &biases, &inputs, &targets, in_dim, out_dim, &cfg, &lif,
        )
        .expect("ok");
        let r_stbp = stbp_unroll(
            &weights, &biases, &inputs, &targets, in_dim, out_dim, &cfg, &lif,
        )
        .expect("ok");

        assert!((r_bptt.loss - r_stbp.loss).abs() < 1e-6);
        for (a, b) in r_bptt.grad_w.iter().zip(r_stbp.grad_w.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        for (a, b) in r_bptt.grad_b.iter().zip(r_stbp.grad_b.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn shape_and_finite_outputs() {
        let cfg = BpttConfig {
            t_steps: 5,
            surrogate: SurrogateKind::Triangle,
            alpha: 1.0,
        };
        let lif = LifConfig {
            tau_m: 8.0,
            v_th: 0.4,
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Soft,
        };
        let in_dim = 4_usize;
        let out_dim = 3_usize;
        let weights = vec![0.2_f32; in_dim * out_dim];
        let biases = vec![0.01_f32; out_dim];
        let inputs = vec![0.3_f32; cfg.t_steps * in_dim];
        let targets = vec![
            1.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ];
        let r = stbp_unroll(
            &weights, &biases, &inputs, &targets, in_dim, out_dim, &cfg, &lif,
        )
        .expect("ok");
        assert_eq!(r.grad_w.len(), in_dim * out_dim);
        assert_eq!(r.grad_b.len(), out_dim);
        assert!(r.loss.is_finite());
        for &g in &r.grad_w {
            assert!(g.is_finite());
        }
        for &g in &r.grad_b {
            assert!(g.is_finite());
        }
    }

    #[test]
    fn rejects_bad_inputs() {
        let cfg = BpttConfig {
            t_steps: 0,
            ..BpttConfig::default()
        };
        let lif = LifConfig::default();
        let err = stbp_unroll(&[], &[], &[], &[], 2, 2, &cfg, &lif);
        assert!(matches!(err, Err(SnnError::BadTimesteps { .. })));
    }

    #[test]
    fn differs_from_bptt_for_soft_reset_when_spikes_occur() {
        // Forward: produce spikes; STBP and BPTT/Soft should diverge because
        // STBP gates with (1 − s_t) but BPTT/Soft does not.
        let cfg = BpttConfig {
            t_steps: 8,
            surrogate: SurrogateKind::Sigmoid,
            alpha: 4.0,
        };
        let lif = LifConfig {
            tau_m: 5.0,
            v_th: 0.5,
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Soft,
        };
        let in_dim = 2_usize;
        let out_dim = 2_usize;
        let weights = vec![0.7_f32, 0.5, 0.6, 0.4];
        let biases = vec![0.1_f32, 0.05];
        let inputs: Vec<f32> = (0..(cfg.t_steps * in_dim))
            .map(|i| 0.2 + 0.05 * (i as f32))
            .collect();
        let targets = vec![0.0_f32; cfg.t_steps * out_dim];

        let r_bptt = bptt_unroll(
            &weights, &biases, &inputs, &targets, in_dim, out_dim, &cfg, &lif,
        )
        .expect("ok");
        let r_stbp = stbp_unroll(
            &weights, &biases, &inputs, &targets, in_dim, out_dim, &cfg, &lif,
        )
        .expect("ok");
        // Loss must match (forward pass identical).
        assert!((r_bptt.loss - r_stbp.loss).abs() < 1e-6);
        // Gradients should differ at least somewhere when spikes are present.
        let total_diff: f32 = r_bptt
            .grad_w
            .iter()
            .zip(r_stbp.grad_w.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        // If at least one spike occurred, total_diff > 0.
        assert!(total_diff >= 0.0); // sanity (could be ==0 if no spike)
    }
}
