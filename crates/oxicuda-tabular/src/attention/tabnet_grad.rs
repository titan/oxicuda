//! Analytic backward pass (explicit gradients) for [`TabNetLayer`].
//!
//! TabNet (Arik & Pfister 2021) is a *sequential* attentive model: at each step
//! the prior scale `P` is updated multiplicatively, `P_{i+1} = P_i ⊙ (γ − M_i)`,
//! so the gradient of a late step's mask flows back into **every** earlier
//! step's prior.  This module implements that recurrence exactly together with
//! backprop through:
//!
//! * the attentive transformer `att_logits = W_att · h + b_att`,
//! * the prior-scaling `scaled = P ⊙ att_logits`,
//! * batch-norm (inference statistics `mean = 0`, `var = 1`, with learnable
//!   `γ_bn / β_bn`),
//! * the **sparsemax Jacobian** (support-restricted centring),
//! * feature selection `h_sel = M ⊙ x`,
//! * the shared and step-specific FC → GLU blocks (with the ReLU at the step
//!   output), and
//! * the mean-pooling output head.
//!
//! The result is verified against central finite differences in the unit tests.

use super::tabnet::TabNetLayer;
use crate::error::TabularResult;

// ─── Gradient container ────────────────────────────────────────────────────────

/// Accumulated gradients for every learnable parameter of a [`TabNetLayer`].
#[derive(Debug, Clone)]
pub struct TabNetGradients {
    /// Gradient w.r.t. the shared FC weight `[2(n_d+n_a) * n_features]`.
    pub shared_w: Vec<f32>,
    /// Gradient w.r.t. the shared FC bias.
    pub shared_b: Vec<f32>,
    /// Per-step gradients w.r.t. the step FC weight.
    pub step_w: Vec<Vec<f32>>,
    /// Per-step gradients w.r.t. the step FC bias.
    pub step_b: Vec<Vec<f32>>,
    /// Per-step gradients w.r.t. the attentive-transformer weight.
    pub att_w: Vec<Vec<f32>>,
    /// Per-step gradients w.r.t. the attentive-transformer bias.
    pub att_b: Vec<Vec<f32>>,
    /// Gradient w.r.t. the output-head weight.
    pub final_w: Vec<f32>,
    /// Gradient w.r.t. the output-head bias.
    pub final_b: Vec<f32>,
    /// Gradient w.r.t. the (shared) batch-norm scale.
    pub bn_gamma: Vec<f32>,
    /// Gradient w.r.t. the (shared) batch-norm bias.
    pub bn_beta: Vec<f32>,
}

// ─── Per-step cache ────────────────────────────────────────────────────────────

struct StepCache {
    h_in: Vec<f32>,         // step input h_{i-1}                 [na_nd]
    prior_before: Vec<f32>, // P_i (before update)               [n_features]
    att_logits: Vec<f32>,   // W_att·h + b                         [n_features]
    scaled: Vec<f32>,       // P_i ⊙ att_logits                    [n_features]
    mask: Vec<f32>,         // M_i = sparsemax(bn_out)             [n_features]
    support: Vec<bool>,     // sparsemax active set                [n_features]
    shared_pre: Vec<f32>,   // shared FC pre-GLU                   [2*na_nd]
    shared_glu: Vec<f32>,   // GLU(shared_pre)                     [na_nd]
    step_pre: Vec<f32>,     // step FC pre-GLU                     [2*na_nd]
    step_glu: Vec<f32>,     // GLU(step_pre)                       [na_nd]
    h_out: Vec<f32>,        // ReLU(step_glu) = h_i                [na_nd]
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `y = W·x + b`, W row-major `[out * in]`.
fn matvec(w: &[f32], b: &[f32], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut y = b.to_vec();
    for o in 0..out_dim {
        for (i, &xi) in x.iter().enumerate().take(in_dim) {
            y[o] += w[o * in_dim + i] * xi;
        }
    }
    y
}

/// Sparsemax that also returns the active support mask (output > 0).
fn sparsemax_support(z: &[f32]) -> (Vec<f32>, Vec<bool>) {
    let mut sorted = z.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumsum = 0.0_f32;
    let mut k_star = 0usize;
    for (j, &zj) in sorted.iter().enumerate() {
        cumsum += zj;
        if 1.0 + (j as f32 + 1.0) * zj - cumsum > 0.0 {
            k_star = j;
        }
    }
    let tau = (sorted.iter().take(k_star + 1).sum::<f32>() - 1.0) / (k_star as f32 + 1.0);
    let out: Vec<f32> = z.iter().map(|&zi| (zi - tau).max(0.0)).collect();
    let support: Vec<bool> = out.iter().map(|&o| o > 0.0).collect();
    (out, support)
}

/// Backprop through sparsemax given the upstream `d_out` and the support set.
///
/// Jacobian `J_ij = [i∈S] (δ_ij − [j∈S]/|S|)`, so
/// `d_z = J^T d_out = s ⊙ (d_out − mean_{S}(d_out))` (J is symmetric).
fn sparsemax_backward(d_out: &[f32], support: &[bool]) -> Vec<f32> {
    let k: usize = support.iter().filter(|&&b| b).count();
    if k == 0 {
        return vec![0.0_f32; d_out.len()];
    }
    let mean: f32 = d_out
        .iter()
        .zip(support.iter())
        .filter(|&(_, &s)| s)
        .map(|(&g, _)| g)
        .sum::<f32>()
        / k as f32;
    d_out
        .iter()
        .zip(support.iter())
        .map(|(&g, &s)| if s { g - mean } else { 0.0 })
        .collect()
}

/// GLU backward: given `pre` (`[2h]`) and `d_out` (`[h]`), return `d_pre` (`[2h]`).
/// `out_i = a_i · σ(b_i)` where `a = pre[..h]`, `b = pre[h..]`.
fn glu_backward(pre: &[f32], d_out: &[f32]) -> Vec<f32> {
    let h = d_out.len();
    let mut d_pre = vec![0.0_f32; 2 * h];
    for i in 0..h {
        let a = pre[i];
        let g = sigmoid(pre[i + h]);
        d_pre[i] = d_out[i] * g; // d/d a
        d_pre[i + h] = d_out[i] * a * g * (1.0 - g); // d/d b
    }
    d_pre
}

/// Backward for `y = W·x + b`: accumulate `dW`, `db`, and add into `d_x`.
fn linear_backward(
    d_y: &[f32],
    x: &[f32],
    w: &[f32],
    in_dim: usize,
    out_dim: usize,
    d_w: &mut [f32],
    d_b: &mut [f32],
    d_x: &mut [f32],
) {
    for o in 0..out_dim {
        let go = d_y[o];
        d_b[o] += go;
        if go == 0.0 {
            continue;
        }
        for i in 0..in_dim {
            d_w[o * in_dim + i] += go * x[i];
            d_x[i] += go * w[o * in_dim + i];
        }
    }
}

impl TabNetLayer {
    /// Forward pass caching all intermediates for backprop.
    fn forward_cached(&self, x: &[f32]) -> TabularResult<(Vec<f32>, Vec<StepCache>)> {
        let cfg = self.config_ref();
        let na_nd = cfg.n_a + cfg.n_d;
        let bn = self.bn_ref();
        let inv = 1.0 / (1.0 + bn.eps).sqrt(); // var = 1 at inference

        let mut prior = vec![1.0_f32; cfg.n_features];
        let mut h = vec![0.0_f32; na_nd];
        let mut caches = Vec::with_capacity(cfg.n_steps);

        for step in 0..cfg.n_steps {
            let h_in = h.clone();
            let prior_before = prior.clone();

            let att_logits = matvec(
                self.att_w_ref(step),
                self.att_b_ref(step),
                &h,
                na_nd,
                cfg.n_features,
            );
            let scaled: Vec<f32> = prior
                .iter()
                .zip(att_logits.iter())
                .map(|(&p, &a)| p * a)
                .collect();
            // batch-norm at inference: (scaled - 0)/sqrt(1+eps) * gamma + beta
            let bn_out: Vec<f32> = (0..cfg.n_features)
                .map(|f| scaled[f] * inv * bn.gamma[f] + bn.beta[f])
                .collect();
            let (mask, support) = sparsemax_support(&bn_out);

            for f in 0..cfg.n_features {
                prior[f] *= cfg.gamma - mask[f];
            }

            let h_sel: Vec<f32> = mask.iter().zip(x.iter()).map(|(&m, &xi)| m * xi).collect();
            let shared_pre = matvec(
                self.shared_w_ref(),
                self.shared_b_ref(),
                &h_sel,
                cfg.n_features,
                2 * na_nd,
            );
            let shared_glu = glu_fwd(&shared_pre);
            let step_pre = matvec(
                self.step_w_ref(step),
                self.step_b_ref(step),
                &shared_glu,
                na_nd,
                2 * na_nd,
            );
            let step_glu = glu_fwd(&step_pre);
            let h_out: Vec<f32> = step_glu.iter().map(|&v| v.max(0.0)).collect();
            h = h_out.clone();

            caches.push(StepCache {
                h_in,
                prior_before,
                att_logits,
                scaled,
                mask,
                support,
                shared_pre,
                shared_glu,
                step_pre,
                step_glu,
                h_out,
            });
        }

        // Aggregate mean of n_d portion → head.
        let mut agg = vec![0.0_f32; cfg.n_d];
        for c in &caches {
            for (a, &v) in agg.iter_mut().zip(c.h_out.iter()) {
                *a += v;
            }
        }
        for a in &mut agg {
            *a /= cfg.n_steps as f32;
        }
        let logits = matvec(
            self.final_w_ref(),
            self.final_b_ref(),
            &agg,
            cfg.n_d,
            cfg.n_classes,
        );
        Ok((logits, caches))
    }

    /// Analytic backward pass.
    ///
    /// `grad_logits` is `dL/d logits` (`[n_classes]`).  Returns the parameter
    /// gradients and the gradient w.r.t. the input features (`[n_features]`).
    pub fn backward(
        &self,
        x: &[f32],
        grad_logits: &[f32],
    ) -> TabularResult<(TabNetGradients, Vec<f32>)> {
        let cfg = self.config_ref();
        let na_nd = cfg.n_a + cfg.n_d;
        let nf = cfg.n_features;
        let bn = self.bn_ref();
        let inv = 1.0 / (1.0 + bn.eps).sqrt();

        let (_logits, caches) = self.forward_cached(x)?;
        let mut g = TabNetGradients::zeros(self);
        let mut d_x = vec![0.0_f32; nf];

        // ── Head: logits = final_w · agg + final_b ; agg = mean_i h_i[..n_d] ──
        let mut agg = vec![0.0_f32; cfg.n_d];
        for c in &caches {
            for (a, &v) in agg.iter_mut().zip(c.h_out.iter()) {
                *a += v;
            }
        }
        for a in &mut agg {
            *a /= cfg.n_steps as f32;
        }
        let final_w = self.final_w_ref();
        let mut d_agg = vec![0.0_f32; cfg.n_d];
        for (o, &gl) in grad_logits.iter().enumerate() {
            g.final_b[o] += gl;
            for d in 0..cfg.n_d {
                g.final_w[o * cfg.n_d + d] += gl * agg[d];
                d_agg[d] += gl * final_w[o * cfg.n_d + d];
            }
        }
        // d_h_out_i[..n_d] from aggregation (the n_a tail gets zero from here).
        let inv_steps = 1.0 / cfg.n_steps as f32;

        // Recurrent state: gradient w.r.t. `prior_before` of the step being
        // processed, fed back from all later steps.  Index by feature.
        let mut d_prior_next = vec![0.0_f32; nf]; // d w.r.t prior entering step i+1
        // Gradient w.r.t. h_i (the OUTPUT of step i == input of step i+1's
        // attention).  Seeded per step from aggregation; the next-step coupling
        // is added as we walk backwards.
        let mut d_h_from_next = vec![0.0_f32; na_nd]; // contribution to d_h_out[i] from step i+1's attention

        for step in (0..cfg.n_steps).rev() {
            let c = &caches[step];

            // d_h_out[i] = aggregation term (first n_d) + coupling from step i+1.
            let mut d_h_out = d_h_from_next.clone();
            for d in 0..cfg.n_d {
                d_h_out[d] += d_agg[d] * inv_steps;
            }

            // ── through ReLU ──────────────────────────────────────────────────
            let mut d_step_glu = vec![0.0_f32; na_nd];
            for j in 0..na_nd {
                d_step_glu[j] = if c.step_glu[j] > 0.0 { d_h_out[j] } else { 0.0 };
            }
            // ── through step GLU ──────────────────────────────────────────────
            let d_step_pre = glu_backward(&c.step_pre, &d_step_glu);
            // ── through step FC ───────────────────────────────────────────────
            let mut d_shared_glu = vec![0.0_f32; na_nd];
            linear_backward(
                &d_step_pre,
                &c.shared_glu,
                self.step_w_ref(step),
                na_nd,
                2 * na_nd,
                &mut g.step_w[step],
                &mut g.step_b[step],
                &mut d_shared_glu,
            );
            // ── through shared GLU ────────────────────────────────────────────
            let d_shared_pre = glu_backward(&c.shared_pre, &d_shared_glu);
            // ── through shared FC → d_h_sel ──────────────────────────────────
            let mut d_h_sel = vec![0.0_f32; nf];
            linear_backward(
                &d_shared_pre,
                &{
                    // h_sel = mask ⊙ x  (recompute)
                    c.mask
                        .iter()
                        .zip(x.iter())
                        .map(|(&m, &xi)| m * xi)
                        .collect::<Vec<_>>()
                },
                self.shared_w_ref(),
                nf,
                2 * na_nd,
                &mut g.shared_w,
                &mut g.shared_b,
                &mut d_h_sel,
            );

            // h_sel = mask ⊙ x  → split into mask & x grads
            let mut d_mask = vec![0.0_f32; nf];
            for f in 0..nf {
                d_mask[f] += d_h_sel[f] * x[f];
                d_x[f] += d_h_sel[f] * c.mask[f];
            }

            // ── prior coupling: prior_before_{i+1} = prior_before_i ⊙ (γ − M_i)
            //    d_prior_next is gradient w.r.t. prior_before_{i+1}.
            //    contributes to d_mask and to d_prior_before_i.
            let mut d_prior_before = vec![0.0_f32; nf];
            for f in 0..nf {
                let pb = c.prior_before[f];
                // ∂prior_next/∂M_i = -pb ;  ∂prior_next/∂prior_before_i = (γ - M_i)
                d_mask[f] += d_prior_next[f] * (-pb);
                d_prior_before[f] += d_prior_next[f] * (cfg.gamma - c.mask[f]);
            }

            // ── through sparsemax: d_mask → d_bn_out ─────────────────────────
            let d_bn_out = sparsemax_backward(&d_mask, &c.support);

            // ── through batch-norm: bn_out = scaled*inv*γ + β ────────────────
            let mut d_scaled = vec![0.0_f32; nf];
            for f in 0..nf {
                let go = d_bn_out[f];
                g.bn_gamma[f] += go * c.scaled[f] * inv;
                g.bn_beta[f] += go;
                d_scaled[f] = go * inv * bn.gamma[f];
            }

            // ── through prior-scaling: scaled = prior_before ⊙ att_logits ────
            let mut d_att_logits = vec![0.0_f32; nf];
            for f in 0..nf {
                d_att_logits[f] = d_scaled[f] * c.prior_before[f];
                d_prior_before[f] += d_scaled[f] * c.att_logits[f];
            }

            // ── through attentive transformer: att_logits = W_att·h_in + b ───
            let mut d_h_in = vec![0.0_f32; na_nd];
            linear_backward(
                &d_att_logits,
                &c.h_in,
                self.att_w_ref(step),
                na_nd,
                nf,
                &mut g.att_w[step],
                &mut g.att_b[step],
                &mut d_h_in,
            );

            // Prepare recurrent state for the PREVIOUS step (i-1):
            //  - its prior_before == this step's prior_before (the chain),
            //    so d_prior carried back is d_prior_before.
            //  - its h_out == this step's h_in, so the coupling gradient is d_h_in.
            d_prior_next = d_prior_before;
            d_h_from_next = d_h_in;
        }

        Ok((g, d_x))
    }
}

/// GLU forward (mirror of `tabnet::glu` but infallible for even input).
fn glu_fwd(pre: &[f32]) -> Vec<f32> {
    let h = pre.len() / 2;
    (0..h).map(|i| pre[i] * sigmoid(pre[i + h])).collect()
}

impl TabNetGradients {
    fn zeros(model: &TabNetLayer) -> Self {
        let cfg = model.config_ref();
        let na_nd = cfg.n_a + cfg.n_d;
        let out_shared = 2 * na_nd;
        let mk = |n: usize| vec![0.0_f32; n];
        Self {
            shared_w: mk(out_shared * cfg.n_features),
            shared_b: mk(out_shared),
            step_w: (0..cfg.n_steps).map(|_| mk(out_shared * na_nd)).collect(),
            step_b: (0..cfg.n_steps).map(|_| mk(out_shared)).collect(),
            att_w: (0..cfg.n_steps)
                .map(|_| mk(na_nd * cfg.n_features))
                .collect(),
            att_b: (0..cfg.n_steps).map(|_| mk(cfg.n_features)).collect(),
            final_w: mk(cfg.n_classes * cfg.n_d),
            final_b: mk(cfg.n_classes),
            bn_gamma: mk(cfg.n_features),
            bn_beta: mk(cfg.n_features),
        }
    }
}

// ─── Parameter handle for finite-difference tests ──────────────────────────────

/// Addresses one scalar TabNet parameter (test-only).
#[cfg(test)]
pub(crate) enum TnParam {
    SharedW(usize),
    SharedB(usize),
    StepW(usize, usize),
    StepB(usize, usize),
    AttW(usize, usize),
    AttB(usize, usize),
    FinalW(usize),
    FinalB(usize),
    BnGamma(usize),
    BnBeta(usize),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::tabnet::TabNetConfig;
    use crate::handle::LcgRng;

    fn tiny() -> (TabNetLayer, Vec<f32>) {
        let cfg = TabNetConfig {
            n_features: 5,
            n_d: 3,
            n_a: 3,
            n_steps: 3,
            gamma: 1.4,
            n_classes: 3,
        };
        let mut rng = LcgRng::new(123);
        let layer = TabNetLayer::new(cfg, &mut rng).expect("new");
        let x = vec![0.6_f32, -0.4, 0.9, 0.2, -1.1];
        (layer, x)
    }

    fn loss(logits: &[f32], dir: &[f32]) -> f32 {
        logits.iter().zip(dir.iter()).map(|(&a, &b)| a * b).sum()
    }

    #[test]
    fn forward_cached_matches_forward() {
        let (layer, x) = tiny();
        let (l1, _m) = layer.forward(&x).expect("forward");
        let (l2, _c) = layer.forward_cached(&x).expect("cached");
        for (a, b) in l1.iter().zip(l2.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn grad_check_input() {
        // Full-graph end-to-end check of dL/dx through the sequential model.
        let (layer, x) = tiny();
        let dir = vec![0.4_f32, -0.6, 0.7];
        let (_g, dx) = layer.backward(&x, &dir).expect("bwd");

        let eps = 2e-3_f32;
        for j in 0..x.len() {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += eps;
            xm[j] -= eps;
            let lp = loss(&layer.forward(&xp).expect("f+").0, &dir);
            let lm = loss(&layer.forward(&xm).expect("f-").0, &dir);
            let fd = (lp - lm) / (2.0 * eps);
            let rel = (fd - dx[j]).abs() / fd.abs().max(dx[j].abs()).max(1e-3);
            assert!(rel < 5e-2, "dx[{j}] analytic={} fd={fd} rel={rel}", dx[j]);
        }
    }

    #[test]
    fn grad_check_parameters() {
        let (mut layer, x) = tiny();
        let dir = vec![0.5_f32, 0.3, -0.8];
        let (g, _dx) = layer.backward(&x, &dir).expect("bwd");

        let checks: Vec<(&str, TnParam, f32)> = vec![
            ("shared_w[3]", TnParam::SharedW(3), g.shared_w[3]),
            ("shared_b[2]", TnParam::SharedB(2), g.shared_b[2]),
            ("step_w0[4]", TnParam::StepW(0, 4), g.step_w[0][4]),
            ("step_w2[10]", TnParam::StepW(2, 10), g.step_w[2][10]),
            ("step_b1[1]", TnParam::StepB(1, 1), g.step_b[1][1]),
            ("att_w0[7]", TnParam::AttW(0, 7), g.att_w[0][7]),
            ("att_w2[3]", TnParam::AttW(2, 3), g.att_w[2][3]),
            ("att_b1[2]", TnParam::AttB(1, 2), g.att_b[1][2]),
            ("final_w[5]", TnParam::FinalW(5), g.final_w[5]),
            ("final_b[1]", TnParam::FinalB(1), g.final_b[1]),
            ("bn_gamma[2]", TnParam::BnGamma(2), g.bn_gamma[2]),
            ("bn_beta[3]", TnParam::BnBeta(3), g.bn_beta[3]),
        ];

        let eps = 2e-3_f32;
        for (label, p, analytic) in checks {
            let orig = layer.param_get(&p);
            layer.param_set(&p, orig + eps);
            let lp = loss(&layer.forward(&x).expect("f+").0, &dir);
            layer.param_set(&p, orig - eps);
            let lm = loss(&layer.forward(&x).expect("f-").0, &dir);
            layer.param_set(&p, orig);
            let fd = (lp - lm) / (2.0 * eps);
            let rel = (fd - analytic).abs() / fd.abs().max(analytic.abs()).max(1e-3);
            assert!(
                rel < 5e-2,
                "param {label}: analytic={analytic} fd={fd} rel={rel}"
            );
        }
    }
}
