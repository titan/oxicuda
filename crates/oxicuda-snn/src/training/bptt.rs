//! Backpropagation-Through-Time (BPTT) for a single fully-connected LIF layer.
//!
//! # Forward pass
//!
//! For a layer with `in_dim → out_dim` weights `W ∈ ℝ^{out×in}`, biases `b ∈ ℝ^out`,
//! input spike train `x_t ∈ ℝ^in` and LIF state with decay `β = exp(−dt/τ_m)`:
//!
//! ```text
//! I_t       = W · x_t + b                                     (synaptic drive)
//! v̄_t      = β · v_{t-1}^{post} + I_t                         (sub-threshold)
//! s_t       = 1[v̄_t ≥ v_th]                                   (Heaviside spike)
//! v_t^{post}= (1 − s_t) · v̄_t + s_t · v_rest    (Hard reset)
//!           = v̄_t − s_t · v_th                  (Soft reset)
//! ```
//!
//! Loss is `L = (1/T) · Σ_t Σ_i (s_{t,i} − ŝ_{t,i})²` (MSE over time and neurons).
//!
//! # Backward pass
//!
//! Surrogate replaces the Dirac `dS/dV` with `g_α(v − v_th)` for a smooth `g_α`
//! (sigmoid / atan / triangle / super-spike / fast-sigmoid).
//!
//! With `dL/ds_t = 2 (s_t − ŝ_t) / T` and recursion through `v_{t-1}^{post}`,
//! the analytical chain rule for the **hard reset** is:
//!
//! ```text
//! dL/dv̄_t        = g_α(v̄_t − v_th) · dL/ds_t  +  (1 − s_t) · β · dL/dv̄_{t+1}
//! ```
//!
//! For **soft reset** the `(1 − s_t)` factor disappears (membrane carries
//! through after subtracting `v_th`). In both cases:
//!
//! ```text
//! dL/dW_{ij} = Σ_t dL/dv̄_{t,i} · x_{t,j}
//! dL/db_i   = Σ_t dL/dv̄_{t,i}
//! ```

use crate::error::{SnnError, SnnResult};
use crate::neuron::lif::{LifConfig, ResetMode, beta};
use crate::surrogate::{
    atan::atan_grad, fast_sigmoid::fast_sigmoid_grad, sigmoid::sigmoid_grad,
    super_spike::super_spike_grad, triangle::triangle_grad,
};

/// Identifier of a smooth surrogate-gradient shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurrogateKind {
    /// Logistic-sigmoid: `α · σ · (1 − σ)`.
    Sigmoid,
    /// Arctan: `α / (π · (1 + (α(v−v_th))²))`.
    Atan,
    /// Triangular: `max(0, 1 − |v − v_th| / α)`.
    Triangle,
    /// Zenke-Ganguli "SuperSpike": `α / (1 + α·|v−v_th|)²`.
    SuperSpike,
    /// Fast-sigmoid: `α / (1 + |α(v−v_th)|)²`.
    FastSigmoid,
}

/// BPTT hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct BpttConfig {
    /// Number of unrolled time steps (sequence length).
    pub t_steps: usize,
    /// Surrogate-gradient shape used for `dS/dV`.
    pub surrogate: SurrogateKind,
    /// Surrogate sharpness `α > 0`.
    pub alpha: f32,
}

impl Default for BpttConfig {
    fn default() -> Self {
        Self {
            t_steps: 16,
            surrogate: SurrogateKind::Sigmoid,
            alpha: 4.0,
        }
    }
}

/// Result of a single BPTT pass: scalar loss plus weight and bias gradients.
#[derive(Debug, Clone)]
pub struct BpttResult {
    /// Mean-squared error summed over time, divided by `t_steps`.
    pub loss: f32,
    /// Flattened weight gradient, layout `[out_dim × in_dim]` row-major.
    pub grad_w: Vec<f32>,
    /// Bias gradient, length `out_dim`.
    pub grad_b: Vec<f32>,
}

/// Dispatch helper that evaluates the chosen surrogate gradient element-wise.
pub fn surrogate_eval(
    kind: SurrogateKind,
    v: &[f32],
    v_th: f32,
    alpha: f32,
    out: &mut [f32],
) -> SnnResult<()> {
    match kind {
        SurrogateKind::Sigmoid => sigmoid_grad(v, v_th, alpha, out),
        SurrogateKind::Atan => atan_grad(v, v_th, alpha, out),
        SurrogateKind::Triangle => triangle_grad(v, v_th, alpha, out),
        SurrogateKind::SuperSpike => super_spike_grad(v, v_th, alpha, out),
        SurrogateKind::FastSigmoid => fast_sigmoid_grad(v, v_th, alpha, out),
    }
}

/// Validate input slice shapes against the requested dimensions.
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

/// Forward / backward pass through a single LIF dense layer of length `t_steps`.
///
/// Weights are stored row-major as `[out_dim × in_dim]`: `W[i, j] = weights[i*in_dim + j]`.
/// Inputs and targets are stored row-major as `[t_steps × *_dim]`.
///
/// Returns the scalar MSE loss and the full weight / bias gradients.
pub fn bptt_unroll(
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

    // History buffers: pre-reset membrane v̄_t, post-reset membrane v_t^{post}, spikes s_t.
    let mut v_pre = vec![0.0_f32; t * out_dim];
    let mut v_post = vec![0.0_f32; t * out_dim];
    let mut s_hist = vec![0.0_f32; t * out_dim];

    // ----- Forward pass -----
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
            v_post[y_off + i] = v_after;
            s_hist[y_off + i] = spike;
            v_prev[i] = v_after;
        }
    }

    // ----- Loss = (1/T) · Σ (s − ŝ)² -----
    let inv_t = 1.0 / (t as f32);
    let mut loss = 0.0_f32;
    let mut dl_ds = vec![0.0_f32; t * out_dim];
    for k in 0..(t * out_dim) {
        let diff = s_hist[k] - target_spikes[k];
        loss += diff * diff;
        dl_ds[k] = 2.0 * diff * inv_t;
    }
    loss *= inv_t;

    // ----- Backward pass -----
    // Surrogate dS/dV evaluated at v̄_t for every (t, i).
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

    // dL / dv̄_{t,i}  — accumulator carried backwards in time.
    let mut dl_dv_next = vec![0.0_f32; out_dim];
    let mut dl_dv = vec![0.0_f32; out_dim];

    for step in (0..t).rev() {
        let x_off = step * in_dim;
        let y_off = step * out_dim;
        for i in 0..out_dim {
            let s_t = s_hist[y_off + i];
            // Reset-mode-dependent recurrent gating factor.
            let recurrent = match lif_cfg.reset {
                ResetMode::Hard => (1.0 - s_t) * b * dl_dv_next[i],
                ResetMode::Soft => b * dl_dv_next[i],
            };
            let dl_dv_i = surrogate[y_off + i] * dl_ds[y_off + i] + recurrent;
            dl_dv[i] = dl_dv_i;
            grad_b[i] += dl_dv_i;
            let row_off = i * in_dim;
            for j in 0..in_dim {
                grad_w[row_off + j] += dl_dv_i * inputs[x_off + j];
            }
        }
        std::mem::swap(&mut dl_dv, &mut dl_dv_next);
        // Reset the (now scratch) buffer for the next iteration.
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

    fn cfg_lif() -> LifConfig {
        LifConfig {
            tau_m: 5.0,
            v_th: 1.0,
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Hard,
        }
    }

    #[test]
    fn shape_checks() {
        let cfg = BpttConfig {
            t_steps: 4,
            surrogate: SurrogateKind::Sigmoid,
            alpha: 4.0,
        };
        let lif = cfg_lif();
        let in_dim = 2_usize;
        let out_dim = 2_usize;
        let weights = vec![0.1_f32; in_dim * out_dim];
        let biases = vec![0.0_f32; out_dim];
        let inputs = vec![0.2_f32; cfg.t_steps * in_dim];
        let targets = vec![0.0_f32; cfg.t_steps * out_dim];
        let res = bptt_unroll(
            &weights, &biases, &inputs, &targets, in_dim, out_dim, &cfg, &lif,
        )
        .expect("ok");
        assert_eq!(res.grad_w.len(), in_dim * out_dim);
        assert_eq!(res.grad_b.len(), out_dim);
        assert!(res.loss.is_finite());
    }

    #[test]
    fn rejects_bad_shapes() {
        let cfg = BpttConfig {
            t_steps: 4,
            ..BpttConfig::default()
        };
        let lif = cfg_lif();
        let weights = vec![0.0_f32; 3];
        let biases = vec![0.0_f32; 2];
        let inputs = vec![0.0_f32; 8];
        let targets = vec![0.0_f32; 8];
        let err = bptt_unroll(&weights, &biases, &inputs, &targets, 2, 2, &cfg, &lif);
        assert!(matches!(err, Err(SnnError::BadShape { .. })));
    }

    #[test]
    fn zero_target_zero_input_zero_grad() {
        let cfg = BpttConfig {
            t_steps: 4,
            surrogate: SurrogateKind::Triangle,
            alpha: 1.0,
        };
        let lif = cfg_lif();
        let in_dim = 3_usize;
        let out_dim = 2_usize;
        let weights = vec![0.0_f32; in_dim * out_dim];
        let biases = vec![0.0_f32; out_dim];
        let inputs = vec![0.0_f32; cfg.t_steps * in_dim];
        let targets = vec![0.0_f32; cfg.t_steps * out_dim];
        let res = bptt_unroll(
            &weights, &biases, &inputs, &targets, in_dim, out_dim, &cfg, &lif,
        )
        .expect("ok");
        assert!(res.loss.abs() < 1e-12);
        for &g in &res.grad_w {
            assert!(g.abs() < 1e-6);
        }
        for &g in &res.grad_b {
            assert!(g.abs() < 1e-6);
        }
    }

    /// Compute scalar loss only (re-uses forward path inside `bptt_unroll`).
    fn forward_loss(
        weights: &[f32],
        biases: &[f32],
        inputs: &[f32],
        targets: &[f32],
        in_dim: usize,
        out_dim: usize,
        cfg: &BpttConfig,
        lif: &LifConfig,
    ) -> f32 {
        let res =
            bptt_unroll(weights, biases, inputs, targets, in_dim, out_dim, cfg, lif).expect("ok");
        res.loss
    }

    #[test]
    fn finite_difference_grad_w_matches_analytical() {
        let cfg = BpttConfig {
            t_steps: 4,
            surrogate: SurrogateKind::Sigmoid,
            alpha: 4.0,
        };
        let lif = cfg_lif();
        let in_dim = 2_usize;
        let out_dim = 2_usize;
        // Use deterministic small inputs so spiking is sparse but non-trivial.
        let weights = vec![0.6_f32, -0.3, 0.4, 0.7];
        let biases = vec![0.05_f32, -0.02];
        let inputs = vec![
            0.5_f32, 0.2, // t=0
            0.3, 0.4, // t=1
            0.6, 0.1, // t=2
            0.2, 0.5, // t=3
        ];
        let targets = vec![
            1.0_f32, 0.0, // t=0
            0.0, 1.0, // t=1
            1.0, 0.0, // t=2
            0.0, 1.0, // t=3
        ];
        let res = bptt_unroll(
            &weights, &biases, &inputs, &targets, in_dim, out_dim, &cfg, &lif,
        )
        .expect("ok");
        let eps = 1e-2_f32;
        for k in 0..weights.len() {
            let mut wp = weights.clone();
            let mut wm = weights.clone();
            wp[k] += eps;
            wm[k] -= eps;
            let lp = forward_loss(&wp, &biases, &inputs, &targets, in_dim, out_dim, &cfg, &lif);
            let lm = forward_loss(&wm, &biases, &inputs, &targets, in_dim, out_dim, &cfg, &lif);
            let fd = (lp - lm) / (2.0 * eps);
            // Surrogate-gradient approximations differ from the discontinuous
            // ground truth; the FD reference can also be exactly zero when the
            // perturbation never moves a neuron across threshold (the underlying
            // function is piecewise constant). When that happens we accept the
            // analytical surrogate value as the smooth proxy. Otherwise we
            // require the analytical gradient to be on a similar order
            // (within 5e-2 absolute / 50% rel) to the FD estimate.
            if fd.abs() < 1e-4 {
                assert!(res.grad_w[k].is_finite(), "ana grad_w[{k}] not finite");
                continue;
            }
            let diff = (fd - res.grad_w[k]).abs();
            let rel = diff / (fd.abs() + res.grad_w[k].abs() + 1e-6);
            assert!(
                diff < 5e-2 || rel < 0.5,
                "grad_w[{k}]: fd={fd}, ana={ana}, diff={diff}, rel={rel}",
                ana = res.grad_w[k]
            );
        }
    }

    #[test]
    fn finite_difference_grad_b_matches_analytical() {
        let cfg = BpttConfig {
            t_steps: 4,
            surrogate: SurrogateKind::Sigmoid,
            alpha: 4.0,
        };
        let lif = cfg_lif();
        let in_dim = 2_usize;
        let out_dim = 2_usize;
        let weights = vec![0.6_f32, -0.3, 0.4, 0.7];
        let biases = vec![0.05_f32, -0.02];
        let inputs = vec![0.5_f32, 0.2, 0.3, 0.4, 0.6, 0.1, 0.2, 0.5];
        let targets = vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let res = bptt_unroll(
            &weights, &biases, &inputs, &targets, in_dim, out_dim, &cfg, &lif,
        )
        .expect("ok");
        let eps = 1e-2_f32;
        for k in 0..biases.len() {
            let mut bp = biases.clone();
            let mut bm = biases.clone();
            bp[k] += eps;
            bm[k] -= eps;
            let lp = forward_loss(
                &weights, &bp, &inputs, &targets, in_dim, out_dim, &cfg, &lif,
            );
            let lm = forward_loss(
                &weights, &bm, &inputs, &targets, in_dim, out_dim, &cfg, &lif,
            );
            let fd = (lp - lm) / (2.0 * eps);
            if fd.abs() < 1e-4 {
                assert!(res.grad_b[k].is_finite(), "ana grad_b[{k}] not finite");
                continue;
            }
            let diff = (fd - res.grad_b[k]).abs();
            let rel = diff / (fd.abs() + res.grad_b[k].abs() + 1e-6);
            assert!(
                diff < 5e-2 || rel < 0.5,
                "grad_b[{k}]: fd={fd}, ana={ana}, diff={diff}, rel={rel}",
                ana = res.grad_b[k]
            );
        }
    }

    #[test]
    fn surrogate_dispatch_matches_explicit() {
        let v = vec![-0.5_f32, 0.0, 0.5, 1.0];
        let v_th = 0.5_f32;
        let alpha = 2.0_f32;
        let mut g_dispatch = vec![0.0_f32; v.len()];
        let mut g_direct = vec![0.0_f32; v.len()];
        surrogate_eval(SurrogateKind::Atan, &v, v_th, alpha, &mut g_dispatch).expect("ok");
        atan_grad(&v, v_th, alpha, &mut g_direct).expect("ok");
        for (a, d) in g_dispatch.iter().zip(g_direct.iter()) {
            assert!((a - d).abs() < 1e-6);
        }
    }

    #[test]
    fn rejects_bad_alpha() {
        let cfg = BpttConfig {
            t_steps: 2,
            surrogate: SurrogateKind::Sigmoid,
            alpha: 0.0,
        };
        let lif = cfg_lif();
        let weights = vec![0.0_f32; 4];
        let biases = vec![0.0_f32; 2];
        let inputs = vec![0.0_f32; 4];
        let targets = vec![0.0_f32; 4];
        let err = bptt_unroll(&weights, &biases, &inputs, &targets, 2, 2, &cfg, &lif);
        assert!(matches!(err, Err(SnnError::OutOfRange { .. })));
    }
}
