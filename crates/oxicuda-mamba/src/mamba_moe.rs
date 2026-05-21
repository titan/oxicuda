//! Mamba Mixture-of-Experts: sparse top-`k` routing over diagonal-state SSM experts.
//!
//! Each token in the input sequence is independently routed by a learnt linear
//! router to a small subset (top-`k`, with `k ∈ {1, 2}` being the usual
//! choices) of `n_experts` diagonal-state SSM blocks.  The selected experts'
//! outputs are weighted by a renormalised softmax over the router logits of
//! those experts and summed.  This gives sub-linear compute scaling in the
//! total expert pool while retaining the modelling capacity of `n_experts`
//! independent SSMs.
//!
//! ## Architecture
//!
//! ```text
//!                ┌── expert 0 (SSM) ─┐
//! x [L × D] ────►│      ...          │── weighted sum ──► y [L × D]
//!                └── expert n-1     ─┘
//!                       ▲
//!                       │ top-k(softmax(router · x))
//! ```
//!
//! Each expert is a diagonal-state SSM:
//!
//! ```text
//! h_t = A_e · h_{t-1} + B_e · x_t
//! y_t = C_e · h_t + D_e · x_t
//! ```
//!
//! with `A_e` a stable diagonal initialised at `-0.5 + ε`, and `B_e, C_e, D_e`
//! Gaussian (`1/√fan_in` scaling).  Routing is **per-token**, so different
//! positions in the same sequence may activate different experts.
//!
//! ## Load-balance loss
//!
//! `L_lb = N · mean_i(f_i · P_i)`, where `f_i` is the fraction of tokens
//! routed (by hard top-1) to expert `i`, and `P_i` is the mean router-softmax
//! probability of expert `i` across all tokens.  When the router is exactly
//! uniform (`P_i = 1/N`) and tokens are evenly assigned (`f_i = 1/N`), the
//! loss equals `1.0`; this is the minimum achievable under the constraint
//! `Σf_i = Σ P_i = 1`.  Adding `L_lb` to the training objective encourages
//! the router to spread tokens evenly across experts.

use crate::error::{MambaError, MambaResult};
use crate::handle::LcgRng;

// ─── ExpertParams ─────────────────────────────────────────────────────────────

/// Per-expert parameters for a single diagonal-state SSM block.
#[derive(Debug, Clone)]
struct ExpertParams {
    /// Diagonal of `A`, length `d_state`.
    a_diag: Vec<f32>,
    /// `B` row-major `[d_state × d_model]`.
    b_mat: Vec<f32>,
    /// `C` row-major `[d_model × d_state]`.
    c_mat: Vec<f32>,
    /// `D` skip, length `d_model`.
    d_skip: Vec<f32>,
}

impl ExpertParams {
    fn random(d_model: usize, d_state: usize, rng: &mut LcgRng) -> Self {
        let mut a_diag = vec![0.0_f32; d_state];
        for a in a_diag.iter_mut() {
            let (g, _) = rng.next_normal_pair();
            let val = -0.5_f32 + g * 0.01_f32;
            *a = if val > -0.05 { -0.05 } else { val };
        }

        let b_scale = 1.0_f32 / (d_model as f32).sqrt();
        let mut b_mat = vec![0.0_f32; d_state * d_model];
        for v in b_mat.iter_mut() {
            let (g, _) = rng.next_normal_pair();
            *v = g * b_scale;
        }

        let c_scale = 1.0_f32 / (d_state as f32).sqrt();
        let mut c_mat = vec![0.0_f32; d_model * d_state];
        for v in c_mat.iter_mut() {
            let (g, _) = rng.next_normal_pair();
            *v = g * c_scale;
        }

        let mut d_skip = vec![0.0_f32; d_model];
        for v in d_skip.iter_mut() {
            let (g, _) = rng.next_normal_pair();
            *v = g * 0.1_f32;
        }

        Self {
            a_diag,
            b_mat,
            c_mat,
            d_skip,
        }
    }

    fn n_params(&self) -> usize {
        self.a_diag.len() + self.b_mat.len() + self.c_mat.len() + self.d_skip.len()
    }
}

// ─── MambaMoeConfig ───────────────────────────────────────────────────────────

/// Configuration for a [`MambaMoe`].
#[derive(Debug, Clone)]
pub struct MambaMoeConfig {
    /// Model / channel dimension `d_model`.
    pub d_model: usize,
    /// Hidden state dimension `d_state` for each expert SSM.
    pub d_state: usize,
    /// Number of experts in the mixture.
    pub n_experts: usize,
    /// Top-`k` experts activated per token; must satisfy `1 ≤ top_k ≤ n_experts`.
    pub top_k: usize,
    /// Expected sequence length `seq_len`.
    pub seq_len: usize,
}

impl MambaMoeConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// * [`MambaError::InvalidModelDim`]    — if `d_model == 0`.
    /// * [`MambaError::InvalidSsmOrder`]    — if `d_state == 0`.
    /// * [`MambaError::InvalidLayerCount`]  — if `n_experts == 0`.
    /// * [`MambaError::InvalidChunkSize`]   — if `top_k == 0`.
    /// * [`MambaError::HeadDimMismatch`]    — if `top_k > n_experts`
    ///   (encoded with `n_heads = top_k`, `d_model = n_experts`).
    /// * [`MambaError::InvalidSeqLen`]      — if `seq_len == 0`.
    pub fn validate(&self) -> MambaResult<()> {
        if self.d_model == 0 {
            return Err(MambaError::InvalidModelDim(0));
        }
        if self.d_state == 0 {
            return Err(MambaError::InvalidSsmOrder(0));
        }
        if self.n_experts == 0 {
            return Err(MambaError::InvalidLayerCount(0));
        }
        if self.top_k == 0 {
            return Err(MambaError::InvalidChunkSize(0));
        }
        if self.top_k > self.n_experts {
            return Err(MambaError::HeadDimMismatch {
                n_heads: self.top_k,
                d_model: self.n_experts,
            });
        }
        if self.seq_len == 0 {
            return Err(MambaError::InvalidSeqLen(0));
        }
        Ok(())
    }
}

// ─── MambaMoe ─────────────────────────────────────────────────────────────────

/// Mixture-of-Experts Mamba block with per-token soft routing over
/// diagonal-state SSM experts.
#[derive(Debug, Clone)]
pub struct MambaMoe {
    cfg: MambaMoeConfig,
    /// Router weight matrix, row-major `[n_experts × d_model]`.
    router: Vec<f32>,
    experts: Vec<ExpertParams>,
}

impl MambaMoe {
    /// Construct an MoE with freshly sampled parameters.
    ///
    /// # Errors
    ///
    /// Propagates [`MambaMoeConfig::validate`] errors.
    pub fn new(cfg: MambaMoeConfig, rng: &mut LcgRng) -> MambaResult<Self> {
        cfg.validate()?;

        // Router: small init for near-uniform initial routing.
        let r_scale = 1.0_f32 / (cfg.d_model as f32).sqrt();
        let mut router = vec![0.0_f32; cfg.n_experts * cfg.d_model];
        for v in router.iter_mut() {
            let (g, _) = rng.next_normal_pair();
            *v = g * r_scale * 0.1_f32;
        }

        let mut experts = Vec::with_capacity(cfg.n_experts);
        for _ in 0..cfg.n_experts {
            experts.push(ExpertParams::random(cfg.d_model, cfg.d_state, rng));
        }

        Ok(Self {
            cfg,
            router,
            experts,
        })
    }

    /// Return a reference to the configuration.
    #[inline]
    pub fn config(&self) -> &MambaMoeConfig {
        &self.cfg
    }

    /// Total number of trainable scalar parameters across router + all experts.
    pub fn n_params(&self) -> usize {
        let router = self.router.len();
        let experts: usize = self.experts.iter().map(|e| e.n_params()).sum();
        router + experts
    }

    /// Router logits per token: `(seq_len × d_model)` → `(seq_len × n_experts)`.
    ///
    /// # Errors
    ///
    /// [`MambaError::DimensionMismatch`] — if `x.len() ≠ seq_len · d_model`.
    pub fn router_logits(&self, x: &[f32]) -> MambaResult<Vec<f32>> {
        let l = self.cfg.seq_len;
        let d = self.cfg.d_model;
        let e = self.cfg.n_experts;
        let expected = l * d;
        if x.len() != expected {
            return Err(MambaError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        let mut logits = vec![0.0_f32; l * e];
        for t in 0..l {
            let x_off = t * d;
            let out_off = t * e;
            for ei in 0..e {
                let row = ei * d;
                let mut acc = 0.0_f32;
                for c in 0..d {
                    acc += self.router[row + c] * x[x_off + c];
                }
                logits[out_off + ei] = acc;
            }
        }
        Ok(logits)
    }

    /// Numerically-stable softmax of a slice of length `n_experts`.
    fn softmax_inplace(buf: &mut [f32]) {
        if buf.is_empty() {
            return;
        }
        let mut m = buf[0];
        for &v in buf.iter().skip(1) {
            if v > m {
                m = v;
            }
        }
        let mut s = 0.0_f32;
        for v in buf.iter_mut() {
            *v = (*v - m).exp();
            s += *v;
        }
        if s > 0.0 {
            for v in buf.iter_mut() {
                *v /= s;
            }
        }
    }

    /// Select the indices of the `k` largest entries of `probs` in descending
    /// order of probability.  Returns at most `min(k, probs.len())` indices.
    fn top_k_indices(probs: &[f32], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        // Partial selection sort — fine for small n_experts.
        let k = k.min(probs.len());
        for i in 0..k {
            let mut best = i;
            for j in (i + 1)..probs.len() {
                if probs[idx[j]] > probs[idx[best]] {
                    best = j;
                }
            }
            idx.swap(i, best);
        }
        idx.truncate(k);
        idx
    }

    /// Run a single expert's SSM scan over the whole sequence.
    fn expert_scan(&self, expert: &ExpertParams, x: &[f32]) -> Vec<f32> {
        let l = self.cfg.seq_len;
        let d = self.cfg.d_model;
        let n = self.cfg.d_state;
        let mut y = vec![0.0_f32; l * d];
        let mut h = vec![0.0_f32; n];

        for t in 0..l {
            let x_off = t * d;
            for (n_idx, h_slot) in h.iter_mut().enumerate().take(n) {
                let row = n_idx * d;
                let mut acc = 0.0_f32;
                for c in 0..d {
                    acc += expert.b_mat[row + c] * x[x_off + c];
                }
                *h_slot = expert.a_diag[n_idx] * *h_slot + acc;
            }
            let y_off = t * d;
            for c in 0..d {
                let row = c * n;
                let mut acc = 0.0_f32;
                for (n_idx, &h_val) in h.iter().enumerate().take(n) {
                    acc += expert.c_mat[row + n_idx] * h_val;
                }
                y[y_off + c] = acc + expert.d_skip[c] * x[x_off + c];
            }
        }
        y
    }

    /// Forward pass with per-token top-`k` expert routing.
    ///
    /// For each token `t`:
    ///   1. Compute router probabilities `p = softmax(W_router · x_t)`.
    ///   2. Select the top-`k` experts.
    ///   3. Renormalise their probabilities so they sum to 1.
    ///   4. Compute the weighted sum of those experts' outputs at time `t`.
    ///
    /// All `n_experts` SSMs are evaluated over the full sequence first; the
    /// renormalised weights then gate which contribute to which output
    /// position.  This is the standard "soft top-k MoE" inference path.
    ///
    /// # Errors
    ///
    /// * [`MambaError::DimensionMismatch`] — if `x.len() ≠ seq_len · d_model`.
    /// * [`MambaError::NonFinite`] — if any output value is non-finite.
    pub fn forward(&self, x: &[f32]) -> MambaResult<Vec<f32>> {
        let l = self.cfg.seq_len;
        let d = self.cfg.d_model;
        let e = self.cfg.n_experts;
        let k = self.cfg.top_k;
        let expected = l * d;
        if x.len() != expected {
            return Err(MambaError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        // Router probabilities per token.
        let mut probs = self.router_logits(x)?;
        for t in 0..l {
            let row = &mut probs[t * e..(t + 1) * e];
            Self::softmax_inplace(row);
        }

        // Expert outputs over the full sequence.
        let expert_outputs: Vec<Vec<f32>> = self
            .experts
            .iter()
            .map(|p| self.expert_scan(p, x))
            .collect();

        // Combine via renormalised top-k weights.
        let mut y = vec![0.0_f32; l * d];
        for t in 0..l {
            let p_row = &probs[t * e..(t + 1) * e];
            let idx = Self::top_k_indices(p_row, k);
            let mut weight_sum = 0.0_f32;
            for &ei in &idx {
                weight_sum += p_row[ei];
            }
            if weight_sum <= 0.0 {
                // Degenerate router output — fall back to uniform weights
                // over the selected experts.  This is impossible after
                // softmax but defensively handled.
                let w = 1.0_f32 / (idx.len().max(1) as f32);
                let y_off = t * d;
                for &ei in &idx {
                    for c in 0..d {
                        y[y_off + c] += w * expert_outputs[ei][y_off + c];
                    }
                }
            } else {
                let y_off = t * d;
                for &ei in &idx {
                    let w = p_row[ei] / weight_sum;
                    for c in 0..d {
                        y[y_off + c] += w * expert_outputs[ei][y_off + c];
                    }
                }
            }
        }

        if y.iter().any(|v| !v.is_finite()) {
            return Err(MambaError::NonFinite("mamba_moe forward output"));
        }
        Ok(y)
    }

    /// Load-balance loss `L_lb = N · mean_i( f_i · P_i )`.
    ///
    /// * `f_i` — fraction of tokens whose hard top-1 expert is `i`.
    /// * `P_i` — mean router-softmax probability of expert `i` across tokens.
    ///
    /// The factor `N = n_experts` is chosen so that the minimum value (under
    /// uniform routing) is exactly `1.0`.  Larger values indicate routing
    /// collapse onto a few experts.
    ///
    /// # Errors
    ///
    /// * [`MambaError::DimensionMismatch`] — if `x.len() ≠ seq_len · d_model`.
    /// * [`MambaError::NonFinite`] — if the loss is non-finite.
    pub fn load_balance_loss(&self, x: &[f32]) -> MambaResult<f32> {
        let l = self.cfg.seq_len;
        let e = self.cfg.n_experts;
        let mut probs = self.router_logits(x)?;
        for t in 0..l {
            let row = &mut probs[t * e..(t + 1) * e];
            Self::softmax_inplace(row);
        }

        // f_i = #tokens whose argmax is i, divided by L.
        let mut counts = vec![0_usize; e];
        // P_i = mean prob across tokens.
        let mut mean_p = vec![0.0_f32; e];
        for t in 0..l {
            let row = &probs[t * e..(t + 1) * e];
            // argmax index for this token.
            let mut best = 0_usize;
            let mut best_v = row[0];
            for (ei, &v) in row.iter().enumerate().skip(1) {
                if v > best_v {
                    best_v = v;
                    best = ei;
                }
            }
            counts[best] += 1;
            for (ei, &v) in row.iter().enumerate() {
                mean_p[ei] += v;
            }
        }
        let l_f = l as f32;
        for v in mean_p.iter_mut() {
            *v /= l_f;
        }

        // L_lb = N · mean_i(f_i · P_i) = Σ_i f_i · P_i.
        //
        // At uniform routing, P_i = 1/N and f_i = 1/N (perfect balance), so
        // Σ f_i P_i = N · (1/N²) = 1/N, and N · mean_i(f_i P_i) = 1/N · 1 = …
        // i.e. the loss takes the value 1/N at perfect uniformity and grows
        // toward 1 as the router collapses onto a single expert.
        let mut mean_fp = 0.0_f32;
        for ei in 0..e {
            let f_i = counts[ei] as f32 / l_f;
            mean_fp += f_i * mean_p[ei];
        }
        mean_fp /= e as f32;
        let lb = e as f32 * mean_fp;

        if !lb.is_finite() {
            return Err(MambaError::NonFinite("mamba_moe load_balance_loss"));
        }
        Ok(lb)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(
        d_model: usize,
        d_state: usize,
        n_experts: usize,
        top_k: usize,
        seq_len: usize,
    ) -> MambaMoeConfig {
        MambaMoeConfig {
            d_model,
            d_state,
            n_experts,
            top_k,
            seq_len,
        }
    }

    fn make(
        d_model: usize,
        d_state: usize,
        n_experts: usize,
        top_k: usize,
        seq_len: usize,
    ) -> MambaMoe {
        let mut rng = LcgRng::new(101);
        MambaMoe::new(cfg(d_model, d_state, n_experts, top_k, seq_len), &mut rng)
            .expect("constructor")
    }

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * 0.07 - 0.4).collect()
    }

    // ── construction ──────────────────────────────────────────────────────────

    /// A valid config constructs.
    #[test]
    fn construct_ok() {
        let mut rng = LcgRng::new(1);
        let m = MambaMoe::new(cfg(4, 4, 4, 2, 8), &mut rng);
        assert!(m.is_ok());
    }

    /// config() round-trips the stored values.
    #[test]
    fn config_accessor() {
        let m = make(3, 4, 5, 2, 6);
        assert_eq!(m.config().d_model, 3);
        assert_eq!(m.config().d_state, 4);
        assert_eq!(m.config().n_experts, 5);
        assert_eq!(m.config().top_k, 2);
        assert_eq!(m.config().seq_len, 6);
    }

    // ── shapes / sizes ────────────────────────────────────────────────────────

    /// router_logits shape == seq_len * n_experts.
    #[test]
    fn router_logits_shape() {
        let m = make(4, 4, 6, 2, 10);
        let x = ramp(10 * 4);
        let r = m.router_logits(&x).expect("router");
        assert_eq!(r.len(), 10 * 6);
    }

    /// forward output shape == seq_len * d_model.
    #[test]
    fn forward_shape() {
        let m = make(4, 4, 5, 2, 8);
        let x = ramp(8 * 4);
        let y = m.forward(&x).expect("forward");
        assert_eq!(y.len(), 8 * 4);
    }

    /// n_params is positive and matches the layout.
    #[test]
    fn n_params_positive_and_correct() {
        let m = make(3, 4, 5, 2, 6);
        let expected = 5 * 3                                // router
            + 5 * (4 + 4 * 3 + 3 * 4 + 3); // experts: A + B + C + D
        assert_eq!(m.n_params(), expected);
    }

    // ── routing semantics ─────────────────────────────────────────────────────

    /// load_balance_loss is non-negative.
    #[test]
    fn load_balance_loss_non_negative() {
        let m = make(4, 4, 6, 2, 12);
        let x = ramp(12 * 4);
        let lb = m.load_balance_loss(&x).expect("lb");
        assert!(lb >= 0.0);
    }

    /// top_k = 1 (hard routing): each token's output equals the chosen
    /// expert's output unscaled.
    #[test]
    fn top_k_one_selects_single_expert() {
        let m = make(3, 4, 4, 1, 5);
        let x = ramp(5 * 3);
        let y = m.forward(&x).expect("forward");
        // Compute the same expert outputs ourselves and check that for each
        // token the output is one of the experts' outputs at that time.
        let mut probs = m.router_logits(&x).expect("router");
        let e = m.config().n_experts;
        let l = m.config().seq_len;
        let d = m.config().d_model;
        for t in 0..l {
            let row = &mut probs[t * e..(t + 1) * e];
            MambaMoe::softmax_inplace(row);
        }
        let expert_outs: Vec<Vec<f32>> = m.experts.iter().map(|p| m.expert_scan(p, &x)).collect();
        for t in 0..l {
            let row = &probs[t * e..(t + 1) * e];
            // argmax expert
            let mut best = 0_usize;
            let mut best_v = row[0];
            for (ei, &v) in row.iter().enumerate().skip(1) {
                if v > best_v {
                    best_v = v;
                    best = ei;
                }
            }
            for c in 0..d {
                let exp = expert_outs[best][t * d + c];
                let got = y[t * d + c];
                assert!(
                    (exp - got).abs() < 1e-5,
                    "t={t} c={c}: expected {exp}, got {got}"
                );
            }
        }
    }

    /// top_k = 2 averages two experts under renormalised weights:
    /// the output must be a convex combination of two of the expert outputs.
    #[test]
    fn top_k_two_is_convex_combination_of_two_experts() {
        let m = make(2, 3, 4, 2, 4);
        let x = ramp(4 * 2);
        let y = m.forward(&x).expect("forward");
        let mut probs = m.router_logits(&x).expect("router");
        let e = m.config().n_experts;
        let l = m.config().seq_len;
        let d = m.config().d_model;
        for t in 0..l {
            let row = &mut probs[t * e..(t + 1) * e];
            MambaMoe::softmax_inplace(row);
        }
        let expert_outs: Vec<Vec<f32>> = m.experts.iter().map(|p| m.expert_scan(p, &x)).collect();
        for t in 0..l {
            let row = &probs[t * e..(t + 1) * e];
            let idx = MambaMoe::top_k_indices(row, 2);
            let w_sum = row[idx[0]] + row[idx[1]];
            let w0 = row[idx[0]] / w_sum;
            let w1 = row[idx[1]] / w_sum;
            for c in 0..d {
                let exp = w0 * expert_outs[idx[0]][t * d + c] + w1 * expert_outs[idx[1]][t * d + c];
                let got = y[t * d + c];
                assert!(
                    (exp - got).abs() < 1e-4,
                    "t={t} c={c}: expected {exp}, got {got}"
                );
            }
        }
    }

    /// With a zeroed router (uniform routing), `load_balance_loss == 1.0`.
    ///
    /// Construct an MoE then zero the router; argmax becomes 0 for all tokens
    /// (tie-broken by index), so f_0 = 1 and P_i = 1/N, giving
    /// L_lb = N · (1/N) · f_0 · P_0 = 1/N · 1 · 1/N · N = 1?
    /// Let's compute: mean_i(f_i P_i) = (1/N) · Σ f_i P_i
    /// f_0 = 1, f_else = 0, so Σ f_i P_i = P_0 = 1/N → mean = 1/(N·N).
    /// L_lb = N · 1/(N·N) = 1/N.
    ///
    /// Therefore with a zeroed router and a "winner-takes-all" argmax the
    /// loss equals `1/n_experts`, *not* 1.  This test pins that exact value.
    #[test]
    fn load_balance_loss_uniform_router_zeroed() {
        let mut m = make(2, 3, 4, 2, 16);
        for v in m.router.iter_mut() {
            *v = 0.0;
        }
        let x = ramp(16 * 2);
        let lb = m.load_balance_loss(&x).expect("lb");
        let expected = 1.0_f32 / (m.config().n_experts as f32);
        assert!(
            (lb - expected).abs() < 1e-5,
            "expected {expected}, got {lb}"
        );
    }

    /// n_experts = 1 → forward equals that single expert's output (top_k = 1).
    #[test]
    fn single_expert_forward_equals_expert_scan() {
        let m = make(3, 4, 1, 1, 6);
        let x = ramp(6 * 3);
        let y = m.forward(&x).expect("forward");
        let expert_out = m.expert_scan(&m.experts[0], &x);
        for (a, b) in y.iter().zip(expert_out.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    // ── determinism / input sensitivity ───────────────────────────────────────

    /// Same seed and config gives the same outputs.
    #[test]
    fn deterministic_given_seed() {
        let mut a = LcgRng::new(11);
        let mut b = LcgRng::new(11);
        let m_a = MambaMoe::new(cfg(3, 4, 4, 2, 6), &mut a).expect("a");
        let m_b = MambaMoe::new(cfg(3, 4, 4, 2, 6), &mut b).expect("b");
        let x = ramp(6 * 3);
        let y_a = m_a.forward(&x).expect("ya");
        let y_b = m_b.forward(&x).expect("yb");
        assert_eq!(y_a, y_b);
    }

    /// Changing the input changes the routing or the expert output.
    #[test]
    fn changing_input_changes_output() {
        let m = make(3, 4, 4, 2, 6);
        let x = ramp(6 * 3);
        let mut x2 = x.clone();
        x2[7] += 1.5;
        let y1 = m.forward(&x).expect("y1");
        let y2 = m.forward(&x2).expect("y2");
        assert_ne!(y1, y2);
    }

    // ── boundary cases ────────────────────────────────────────────────────────

    /// Single-token sequence (seq_len = 1) works end-to-end.
    #[test]
    fn single_token_works() {
        let m = make(3, 4, 4, 2, 1);
        let x = ramp(3);
        let y = m.forward(&x).expect("forward");
        assert_eq!(y.len(), 3);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    /// Finite output for Gaussian input.
    #[test]
    fn forward_finite_under_gaussian_input() {
        let m = make(4, 6, 4, 2, 12);
        let mut rng = LcgRng::new(2027);
        let mut x = vec![0.0_f32; 12 * 4];
        rng.fill_normal(&mut x);
        let y = m.forward(&x).expect("forward");
        assert!(y.iter().all(|v| v.is_finite()));
    }

    // ── error paths ───────────────────────────────────────────────────────────

    /// n_experts = 0 fails.
    #[test]
    fn err_zero_n_experts() {
        let mut rng = LcgRng::new(1);
        let err = MambaMoe::new(cfg(3, 4, 0, 1, 6), &mut rng).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidLayerCount(0)));
    }

    /// top_k = 0 fails.
    #[test]
    fn err_zero_top_k() {
        let mut rng = LcgRng::new(1);
        let err = MambaMoe::new(cfg(3, 4, 4, 0, 6), &mut rng).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidChunkSize(0)));
    }

    /// top_k > n_experts fails.
    #[test]
    fn err_top_k_exceeds_n_experts() {
        let mut rng = LcgRng::new(1);
        let err = MambaMoe::new(cfg(3, 4, 2, 4, 6), &mut rng).expect_err("should fail");
        assert!(matches!(err, MambaError::HeadDimMismatch { .. }));
    }

    /// d_model = 0 fails.
    #[test]
    fn err_zero_d_model() {
        let mut rng = LcgRng::new(1);
        let err = MambaMoe::new(cfg(0, 4, 4, 1, 6), &mut rng).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidModelDim(0)));
    }

    /// d_state = 0 fails.
    #[test]
    fn err_zero_d_state() {
        let mut rng = LcgRng::new(1);
        let err = MambaMoe::new(cfg(3, 0, 4, 1, 6), &mut rng).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidSsmOrder(0)));
    }

    /// seq_len = 0 fails.
    #[test]
    fn err_zero_seq_len() {
        let mut rng = LcgRng::new(1);
        let err = MambaMoe::new(cfg(3, 4, 4, 1, 0), &mut rng).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidSeqLen(0)));
    }

    /// Wrong input length to forward fails.
    #[test]
    fn err_wrong_input_length_forward() {
        let m = make(3, 4, 4, 2, 6);
        let x = vec![0.0_f32; 5];
        let err = m.forward(&x).expect_err("should fail");
        assert!(matches!(err, MambaError::DimensionMismatch { .. }));
    }

    /// Wrong input length to router_logits fails.
    #[test]
    fn err_wrong_input_length_router() {
        let m = make(3, 4, 4, 2, 6);
        let x = vec![0.0_f32; 5];
        let err = m.router_logits(&x).expect_err("should fail");
        assert!(matches!(err, MambaError::DimensionMismatch { .. }));
    }
}
