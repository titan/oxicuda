//! MoE-Mamba / state-space routing: sequence-aware expert selection.
//!
//! Implements the routing primitive behind:
//! Pióro et al. "MoE-Mamba: Efficient Selective State Space Models with Mixture
//! of Experts." 2024.
//!
//! Vanilla MoE routers score each token **independently** — token `t`'s expert
//! depends only on `x_t`. A state-space router instead runs a *selective scan*
//! (the Mamba recurrence) along the sequence so the routing feature for token
//! `t` carries running context from all earlier tokens. Concretely, per state
//! channel `n` the model maintains a hidden state and updates it causally:
//!
//! ```text
//! ā_t = exp(−softplus(Δ_t) · A_n)            (per-step, per-channel decay, 0<ā≤1)
//! b̄_t = (1 − ā_t) · B_n                       (zero-order-hold input gate)
//! h_{t,n} = ā_{t,n} · h_{t−1,n} + b̄_{t,n} · u_t
//! y_t      = Σ_n C_n · h_{t,n}  + D · u_t      (context-mixed scalar per token)
//! ```
//!
//! where `u_t = w_in · x_t` is a learned scalar projection of the token, `Δ_t`
//! is a learned input-dependent step size making the recurrence *selective*, and
//! `A_n < 0` are stable continuous-time poles. The context-mixed sequence is
//! concatenated with the raw token and fed to a standard softmax gate to pick
//! the top-k experts. Because `0 < ā ≤ 1`, the scan is numerically stable and
//! the routing feature is a causal, exponentially-weighted summary of history.
//!
//! The whole router is a pure CPU recurrence; it is the sequence-aware analogue
//! of [`crate::routing::top_k`].

use crate::error::{MoeError, MoeResult};
use crate::handle::LcgRng;
use crate::routing::diff_capacity::softplus;
use crate::routing::top_k::{stable_softmax, topk};

/// Configuration for a [`MambaRouter`].
#[derive(Debug, Clone)]
pub struct MambaRouteConfig {
    /// Token feature dimension (`> 0`).
    pub input_dim: usize,
    /// Number of experts (`> 0`).
    pub n_experts: usize,
    /// Experts selected per token (`1 ≤ top_k ≤ n_experts`).
    pub top_k: usize,
    /// Number of SSM state channels `N` (`> 0`). More channels = richer context.
    pub state_dim: usize,
}

impl MambaRouteConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns [`MoeError`] for any zero dimension or an invalid `top_k`.
    pub fn validate(&self) -> MoeResult<()> {
        if self.input_dim == 0 {
            return Err(MoeError::InvalidInputDim {
                dim: self.input_dim,
            });
        }
        if self.n_experts == 0 {
            return Err(MoeError::InvalidExpertCount {
                n_experts: self.n_experts,
            });
        }
        if self.top_k == 0 || self.top_k > self.n_experts {
            return Err(MoeError::InvalidTopK {
                k: self.top_k,
                n_experts: self.n_experts,
            });
        }
        if self.state_dim == 0 {
            return Err(MoeError::InvalidHiddenDim {
                dim: self.state_dim,
            });
        }
        Ok(())
    }
}

/// Result of a sequence-aware routing pass.
#[derive(Debug, Clone)]
pub struct MambaRouteResult {
    /// Selected expert indices, shape `[seq_len · top_k]`.
    pub indices: Vec<usize>,
    /// Renormalised top-k gate scores (each token sums to `1`),
    /// shape `[seq_len · top_k]`.
    pub scores: Vec<f32>,
    /// The context-mixed scalar per token produced by the selective scan,
    /// shape `[seq_len]`. Exposed for inspection / auxiliary losses.
    pub ssm_features: Vec<f32>,
    /// Raw gate logits, shape `[seq_len · n_experts]`.
    pub logits: Vec<f32>,
}

/// A selective-state-space (Mamba-style) router producing sequence-aware
/// top-k expert assignments.
#[derive(Debug, Clone)]
pub struct MambaRouter {
    /// Input projection `w_in`, shape `[input_dim]` → scalar token signal `u_t`.
    w_in: Vec<f32>,
    /// Step-size projection `w_delta`, shape `[input_dim]` → `Δ_t` per token.
    w_delta: Vec<f32>,
    /// Stable continuous poles `A_n < 0`, shape `[state_dim]`.
    a_log: Vec<f32>,
    /// Input matrix `B_n`, shape `[state_dim]`.
    b: Vec<f32>,
    /// Output matrix `C_n`, shape `[state_dim]`.
    c: Vec<f32>,
    /// Skip term `D`.
    d: f32,
    /// Gate matrix over `[token ∥ ssm_feature]`, row-major
    /// `[n_experts · (input_dim + 1)]`.
    gate: Vec<f32>,
    /// Configuration.
    pub config: MambaRouteConfig,
}

impl MambaRouter {
    /// Build a router with randomly initialised SSM parameters and gate.
    ///
    /// The poles are parameterised as `A_n = −softplus(a_log_n)` so they are
    /// always strictly negative (a stable HiPPO-style initialisation), giving a
    /// well-behaved decaying recurrence regardless of the random draw.
    ///
    /// # Errors
    /// Propagates [`MambaRouteConfig::validate`].
    pub fn new(cfg: MambaRouteConfig, rng: &mut LcgRng) -> MoeResult<Self> {
        cfg.validate()?;
        let d = cfg.input_dim;
        let n = cfg.state_dim;

        let mut w_in = vec![0.0_f32; d];
        let mut w_delta = vec![0.0_f32; d];
        rng.fill_normal_scaled(&mut w_in, 0.1);
        rng.fill_normal_scaled(&mut w_delta, 0.05);

        // Initialise A_n ≈ -(n+1): the canonical S4D-real spread of stable poles.
        // Store `a_log` so that softplus(a_log) ≈ (n+1) ⇒ A_n = -softplus(a_log).
        let a_log: Vec<f32> = (0..n)
            .map(|i| {
                let target = (i + 1) as f32; // desired |A_n|
                // invert softplus: a_log = ln(e^target - 1)
                (target.exp() - 1.0).max(1e-6).ln()
            })
            .collect();

        let mut b = vec![0.0_f32; n];
        let mut c = vec![0.0_f32; n];
        rng.fill_normal_scaled(&mut b, 0.2);
        rng.fill_normal_scaled(&mut c, 0.2);

        let gate_cols = d + 1;
        let mut gate = vec![0.0_f32; cfg.n_experts * gate_cols];
        rng.fill_normal_scaled(&mut gate, 0.02);

        Ok(Self {
            w_in,
            w_delta,
            a_log,
            b,
            c,
            d: 1.0,
            gate,
            config: cfg,
        })
    }

    /// Run the selective scan over `seq_len` tokens (`x` shape `[seq_len·d]`),
    /// returning the context-mixed scalar feature per token.
    ///
    /// The recurrence is strictly causal: `ssm[t]` depends only on tokens
    /// `0..=t`.
    fn selective_scan(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let d = self.config.input_dim;
        let n = self.config.state_dim;
        // Continuous poles A_n = -softplus(a_log_n) < 0.
        let poles: Vec<f32> = self.a_log.iter().map(|&al| -softplus(al)).collect();

        let mut state = vec![0.0_f32; n]; // h_{t,n}
        let mut out = vec![0.0_f32; seq_len];
        for t in 0..seq_len {
            let row = &x[t * d..(t + 1) * d];
            let u: f32 = row
                .iter()
                .zip(self.w_in.iter())
                .map(|(&xi, &wi)| xi * wi)
                .sum();
            let delta_pre: f32 = row
                .iter()
                .zip(self.w_delta.iter())
                .map(|(&xi, &wi)| xi * wi)
                .sum();
            let delta = softplus(delta_pre); // positive step size

            let mut y = self.d * u;
            for ch in 0..n {
                // Zero-order-hold discretisation:
                //   ā = exp(Δ·A),  b̄ = (ā − 1)/A · B  (≈ (1−ā)·B for stable A<0)
                let a_bar = (delta * poles[ch]).exp(); // 0 < a_bar ≤ 1
                let b_bar = if poles[ch].abs() > 1e-6 {
                    (a_bar - 1.0) / poles[ch] * self.b[ch]
                } else {
                    delta * self.b[ch]
                };
                state[ch] = a_bar * state[ch] + b_bar * u;
                y += self.c[ch] * state[ch];
            }
            out[t] = y;
        }
        out
    }

    /// Route a sequence of `seq_len` tokens. `x` has shape `[seq_len · input_dim]`.
    ///
    /// # Errors
    /// Returns [`MoeError`] on empty input, a shape mismatch, or non-finite
    /// scores.
    pub fn route(&self, x: &[f32], seq_len: usize) -> MoeResult<MambaRouteResult> {
        let cfg = &self.config;
        if seq_len == 0 {
            return Err(MoeError::EmptyInput);
        }
        let d = cfg.input_dim;
        if x.len() != seq_len * d {
            return Err(MoeError::DimensionMismatch {
                expected: seq_len * d,
                got: x.len(),
            });
        }

        let ssm_features = self.selective_scan(x, seq_len);
        if ssm_features.iter().any(|v| !v.is_finite()) {
            return Err(MoeError::NanEncountered {
                context: "mamba selective scan".to_string(),
            });
        }

        let gate_cols = d + 1;
        let mut logits = vec![0.0_f32; seq_len * cfg.n_experts];
        for t in 0..seq_len {
            let row = &x[t * d..(t + 1) * d];
            let feat = ssm_features[t];
            for e in 0..cfg.n_experts {
                let w = &self.gate[e * gate_cols..(e + 1) * gate_cols];
                // logit = w[..d]·x + w[d]·ssm_feature
                let mut acc: f32 = w
                    .iter()
                    .take(d)
                    .zip(row.iter())
                    .map(|(&wi, &xi)| wi * xi)
                    .sum();
                acc += w[d] * feat;
                logits[t * cfg.n_experts + e] = acc;
            }
        }

        let mut indices = vec![0_usize; seq_len * cfg.top_k];
        let mut scores = vec![0.0_f32; seq_len * cfg.top_k];
        for t in 0..seq_len {
            let probs = stable_softmax(&logits[t * cfg.n_experts..(t + 1) * cfg.n_experts]);
            let (top_vals, top_idx) = topk(&probs, cfg.top_k)?;
            let denom: f32 = top_vals.iter().sum::<f32>().max(1e-12);
            for slot in 0..cfg.top_k {
                scores[t * cfg.top_k + slot] = top_vals[slot] / denom;
                indices[t * cfg.top_k + slot] = top_idx[slot];
            }
        }

        if scores.iter().any(|v| !v.is_finite()) {
            return Err(MoeError::NanEncountered {
                context: "mamba router scores".to_string(),
            });
        }

        Ok(MambaRouteResult {
            indices,
            scores,
            ssm_features,
            logits,
        })
    }

    /// Total router parameter count.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.w_in.len()
            + self.w_delta.len()
            + self.a_log.len()
            + self.b.len()
            + self.c.len()
            + 1
            + self.gate.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MambaRouteConfig {
        MambaRouteConfig {
            input_dim: 8,
            n_experts: 4,
            top_k: 2,
            state_dim: 6,
        }
    }

    #[test]
    fn scores_sum_to_one_and_indices_valid() {
        let mut rng = LcgRng::new(1);
        let router = MambaRouter::new(cfg(), &mut rng).expect("new should succeed");
        let seq_len = 10;
        let mut x = vec![0.0_f32; seq_len * 8];
        rng.fill_normal_scaled(&mut x, 1.0);
        let res = router.route(&x, seq_len).expect("route should succeed");
        for t in 0..seq_len {
            let s: f32 = res.scores[t * 2..t * 2 + 2].iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "token {t} scores sum {s}");
            for slot in 0..2 {
                assert!(res.indices[t * 2 + slot] < 4);
            }
        }
    }

    #[test]
    fn scan_is_causal() {
        // Changing a *later* token must not alter the ssm feature of an earlier
        // one (strict causality of the recurrence).
        let mut rng = LcgRng::new(2);
        let router = MambaRouter::new(cfg(), &mut rng).expect("new should succeed");
        let seq_len = 6;
        let mut x = vec![0.0_f32; seq_len * 8];
        rng.fill_normal_scaled(&mut x, 1.0);
        let base = router.selective_scan(&x, seq_len);

        // Perturb the LAST token only.
        let mut x2 = x.clone();
        for v in x2[(seq_len - 1) * 8..seq_len * 8].iter_mut() {
            *v += 5.0;
        }
        let perturbed = router.selective_scan(&x2, seq_len);

        for t in 0..seq_len - 1 {
            assert!(
                (base[t] - perturbed[t]).abs() < 1e-6,
                "token {t} feature changed by a future token (non-causal)"
            );
        }
        // The last token's own feature should change.
        assert!((base[seq_len - 1] - perturbed[seq_len - 1]).abs() > 1e-6);
    }

    #[test]
    fn scan_mixes_context() {
        // Two sequences that share token t but differ in earlier history should
        // generally produce different ssm features at t — context dependence.
        let mut rng = LcgRng::new(3);
        let router = MambaRouter::new(cfg(), &mut rng).expect("new should succeed");
        let seq_len = 5;
        let mut x = vec![0.0_f32; seq_len * 8];
        rng.fill_normal_scaled(&mut x, 1.0);

        // Perturb the FIRST token; later tokens unchanged.
        let mut x2 = x.clone();
        for v in x2[0..8].iter_mut() {
            *v += 3.0;
        }
        let a = router.selective_scan(&x, seq_len);
        let b = router.selective_scan(&x2, seq_len);
        // A later token's feature must reflect the changed history.
        assert!(
            (a[seq_len - 1] - b[seq_len - 1]).abs() > 1e-6,
            "later token did not absorb earlier-token context"
        );
    }

    #[test]
    fn scan_is_stable_for_large_inputs() {
        // Stable poles (A<0, 0<ā≤1) ⇒ the scan must stay finite even for large,
        // long sequences.
        let mut rng = LcgRng::new(4);
        let router = MambaRouter::new(cfg(), &mut rng).expect("new should succeed");
        let seq_len = 200;
        let x = vec![10.0_f32; seq_len * 8];
        let feats = router.selective_scan(&x, seq_len);
        assert!(feats.iter().all(|v| v.is_finite()), "scan diverged");
    }

    #[test]
    fn empty_and_mismatch_errors() {
        let mut rng = LcgRng::new(5);
        let router = MambaRouter::new(cfg(), &mut rng).expect("new should succeed");
        assert!(matches!(router.route(&[], 0), Err(MoeError::EmptyInput)));
        let x = vec![0.0_f32; 9];
        assert!(matches!(
            router.route(&x, 2),
            Err(MoeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn param_count_positive() {
        let mut rng = LcgRng::new(6);
        let router = MambaRouter::new(cfg(), &mut rng).expect("new should succeed");
        assert!(router.param_count() > 0);
    }
}
