//! MoLoRA — Mixture of Low-Rank Adapters routed per token.
//!
//! References:
//! - Zadouri, T. et al. (2023). *Pushing Mixture of Experts to the Limit:
//!   Extremely Parameter Efficient MoE for Instruction Tuning*.
//!   <https://arxiv.org/abs/2309.05444>
//! - Wu, X. et al. (2024). *Mixture of LoRA Experts*. <https://arxiv.org/abs/2404.13628>
//!
//! Each token vector `x_t` is routed through `n_experts` low-rank pairs `(A_k, B_k)` by a
//! gating matrix `W_g ∈ ℝ^{K × in}`:
//!
//! ```text
//!   logits_t = W_g · x_t
//!   gate_t   = softmax( logits_t / τ )            (or top-k mask then renormalise)
//!   Δy_t     = s · Σ_k  gate_t[k] · B_k · A_k · x_t,   s = α / rank
//! ```
//!
//! The base linear projection `W · x_t` is added externally by the caller — this module
//! returns only the adapter delta `Δy_t`, mirroring the convention of the other LoRA
//! variants in this crate.

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

/// `(dA_k, dB_k, dW_g)` row-major gradients produced by [`MoLoraAdapter::backward`].
pub type MoLoraGrads = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>);

/// Hyper-parameter bundle for a single MoLoRA adapter.
#[derive(Clone, Debug)]
pub struct MoLoraConfig {
    /// Input feature count.
    pub in_features: usize,
    /// Output feature count.
    pub out_features: usize,
    /// Low-rank dimension shared by every expert.
    pub rank: usize,
    /// Global LoRA scaling factor `α`; effective scale is `s = α / rank`.
    pub alpha: f64,
    /// Number of expert pairs `(A_k, B_k)`.
    pub n_experts: usize,
    /// Number of experts kept per token after top-k masking.
    pub top_k: usize,
    /// Softmax temperature `τ`. Lower values sharpen the gating distribution.
    pub temperature: f64,
}

impl MoLoraConfig {
    /// Effective scale `s = α / rank`.
    #[must_use]
    pub fn scale(&self) -> f64 {
        if self.rank == 0 {
            0.0
        } else {
            self.alpha / self.rank as f64
        }
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// - [`PeftError::EmptyInput`] if any dimension is zero.
    /// - [`PeftError::RankTooLarge`] if `rank > min(in_features, out_features)`.
    /// - [`PeftError::Internal`] for `top_k > n_experts` or `temperature ≤ 0`.
    pub fn validate(&self) -> PeftResult<()> {
        if self.in_features == 0
            || self.out_features == 0
            || self.rank == 0
            || self.n_experts == 0
            || self.top_k == 0
        {
            return Err(PeftError::EmptyInput);
        }
        let dim = self.in_features.min(self.out_features);
        if self.rank > dim {
            return Err(PeftError::RankTooLarge {
                rank: self.rank,
                dim,
            });
        }
        if self.top_k > self.n_experts {
            return Err(PeftError::Internal {
                msg: format!(
                    "top_k {} must not exceed n_experts {}",
                    self.top_k, self.n_experts
                ),
            });
        }
        if self.temperature <= 0.0 {
            return Err(PeftError::Internal {
                msg: format!("temperature must be > 0, got {}", self.temperature),
            });
        }
        Ok(())
    }
}

/// Per-token routing diagnostics returned by [`MoLoraAdapter::forward_with_route`].
#[derive(Clone, Debug)]
pub struct MoLoraRouteInfo {
    /// Full-length gate vector after softmax + top-k masking (length `n_experts`).
    pub gates: Vec<f64>,
    /// Indices of the experts kept after top-k selection (length `top_k`).
    pub selected: Vec<usize>,
    /// Variance of the gate sums across a batch — diagnostic for load imbalance.
    pub load_balance_var: f64,
}

/// MoLoRA adapter with `K` low-rank experts and a single linear gate.
pub struct MoLoraAdapter {
    pub(crate) cfg: MoLoraConfig,
    pub(crate) a_experts: Vec<Vec<f64>>,
    pub(crate) b_experts: Vec<Vec<f64>>,
    pub(crate) w_gate: Vec<f64>,
}

impl MoLoraAdapter {
    /// Build a fresh adapter.
    ///
    /// Each `A_k` is sampled from `N(0, 1/√in_features)`; each `B_k` is zero-initialised;
    /// `W_g` uses a Xavier-Glorot uniform draw on `±√(6 / (in + K))`.
    ///
    /// # Errors
    ///
    /// Forwards [`MoLoraConfig::validate`] errors.
    pub fn new(cfg: MoLoraConfig, seed: u64) -> PeftResult<Self> {
        cfg.validate()?;
        let mut rng = LcgRng::new(seed);
        let std_dev = 1.0_f64 / (cfg.in_features as f64).sqrt();
        let mut a_experts = Vec::with_capacity(cfg.n_experts);
        let mut b_experts = Vec::with_capacity(cfg.n_experts);
        let a_len = cfg.rank * cfg.in_features;
        for _ in 0..cfg.n_experts {
            let mut a_k = vec![0.0_f64; a_len];
            let mut i = 0;
            while i + 1 < a_len {
                let (u, v) = rng.next_normal_pair();
                a_k[i] = (u as f64) * std_dev;
                a_k[i + 1] = (v as f64) * std_dev;
                i += 2;
            }
            if i < a_len {
                a_k[i] = (rng.next_normal() as f64) * std_dev;
            }
            a_experts.push(a_k);
            b_experts.push(vec![0.0_f64; cfg.out_features * cfg.rank]);
        }
        let g_len = cfg.n_experts * cfg.in_features;
        let bound = (6.0_f64 / (cfg.in_features as f64 + cfg.n_experts as f64)).sqrt();
        let mut w_gate = vec![0.0_f64; g_len];
        for w in w_gate.iter_mut() {
            let u = rng.next_f32() as f64;
            *w = (u * 2.0 - 1.0) * bound;
        }
        Ok(Self {
            cfg,
            a_experts,
            b_experts,
            w_gate,
        })
    }

    /// Borrow the per-expert down-projections, each row-major `[rank × in_features]`.
    #[must_use]
    pub fn a_experts(&self) -> &[Vec<f64>] {
        &self.a_experts
    }

    /// Borrow the per-expert up-projections, each row-major `[out_features × rank]`.
    #[must_use]
    pub fn b_experts(&self) -> &[Vec<f64>] {
        &self.b_experts
    }

    /// Borrow the gating matrix, row-major `[n_experts × in_features]`.
    #[must_use]
    pub fn w_gate(&self) -> &[f64] {
        &self.w_gate
    }

    /// Effective scale `s = α / rank`.
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.cfg.scale()
    }

    /// Compute `Δy = s · Σ_k gate[k] · B_k · A_k · x`.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] when `x.len() != in_features`.
    pub fn forward(&self, x: &[f64]) -> PeftResult<Vec<f64>> {
        let (y, _) = self.forward_with_route(x)?;
        Ok(y)
    }

    /// Forward pass that also returns the [`MoLoraRouteInfo`] for the token.
    ///
    /// `load_balance_var` is set to `0.0` (single-token variance is zero by definition);
    /// use [`Self::forward_batch`] when batch-level load balance is needed.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] when `x.len() != in_features`.
    pub fn forward_with_route(&self, x: &[f64]) -> PeftResult<(Vec<f64>, MoLoraRouteInfo)> {
        if x.len() != self.cfg.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.cfg.in_features,
                got: x.len(),
            });
        }
        let (gates, selected) = self.route(x);
        let y = self.combine(x, &gates, &selected);
        Ok((
            y,
            MoLoraRouteInfo {
                gates,
                selected,
                load_balance_var: 0.0,
            },
        ))
    }

    /// Batched forward, returning per-token outputs and route infos. The
    /// `load_balance_var` field of each [`MoLoraRouteInfo`] is set to the variance of the
    /// summed gate masses across the batch, so a perfectly balanced router yields zero.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] when any input row has wrong length.
    pub fn forward_batch(
        &self,
        xs: &[Vec<f64>],
    ) -> PeftResult<(Vec<Vec<f64>>, Vec<MoLoraRouteInfo>)> {
        if xs.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut ys = Vec::with_capacity(xs.len());
        let mut infos = Vec::with_capacity(xs.len());
        let mut totals = vec![0.0_f64; self.cfg.n_experts];
        for x in xs {
            let (y, info) = self.forward_with_route(x)?;
            for (acc, g) in totals.iter_mut().zip(info.gates.iter()) {
                *acc += g;
            }
            ys.push(y);
            infos.push(info);
        }
        let mean = totals.iter().sum::<f64>() / totals.len() as f64;
        let variance = totals.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / totals.len() as f64;
        for info in infos.iter_mut() {
            info.load_balance_var = variance;
        }
        Ok((ys, infos))
    }

    /// Closed-form gradients for the per-expert pairs and the gating matrix.
    ///
    /// Returned tuple is `(grad_a_per_expert, grad_b_per_expert, grad_w_gate)`. For experts
    /// not selected by top-k the gradients are zero-length-correct (all zeros).
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] when `x.len() != in_features` or
    /// `grad_y.len() != out_features`.
    pub fn backward(&self, x: &[f64], grad_y: &[f64]) -> PeftResult<MoLoraGrads> {
        if x.len() != self.cfg.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.cfg.in_features,
                got: x.len(),
            });
        }
        if grad_y.len() != self.cfg.out_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.cfg.out_features,
                got: grad_y.len(),
            });
        }
        let r = self.cfg.rank;
        let in_f = self.cfg.in_features;
        let out = self.cfg.out_features;
        let k_exp = self.cfg.n_experts;
        let s = self.scale();
        let tau = self.cfg.temperature;
        let (gates, selected) = self.route(x);
        let mut grad_a_all = vec![vec![0.0_f64; r * in_f]; k_exp];
        let mut grad_b_all = vec![vec![0.0_f64; out * r]; k_exp];
        // d Δy_t / d gate[k] for selected experts: u_k = B_k · A_k · x.  We use this to
        // compute both ∂L/∂(B_k·A_k·x) (for parameter grads) and ∂L/∂gate[k] (for W_g).
        let mut d_gate = vec![0.0_f64; k_exp];
        for &k in &selected {
            let t_k = mat_vec(&self.a_experts[k], x, r, in_f);
            let z_k = mat_vec(&self.b_experts[k], &t_k, out, r);
            let g_k = gates[k];
            // grad_b_k = s · gate[k] · grad_y · t_kᵀ
            for (i, g_i) in grad_y.iter().enumerate() {
                let row = i * r;
                let scaled = s * g_k * g_i;
                for (kk, t_kk) in t_k.iter().enumerate() {
                    grad_b_all[k][row + kk] = scaled * t_kk;
                }
            }
            // grad_a_k = s · gate[k] · (B_kᵀ · grad_y) · xᵀ
            let mut u = vec![0.0_f64; r];
            for (kk, u_kk) in u.iter_mut().enumerate() {
                let mut acc = 0.0_f64;
                for (i, g_i) in grad_y.iter().enumerate() {
                    acc += self.b_experts[k][i * r + kk] * g_i;
                }
                *u_kk = acc;
            }
            for (kk, u_kk) in u.iter().enumerate() {
                let row = kk * in_f;
                let scaled = s * g_k * u_kk;
                for (j, x_j) in x.iter().enumerate() {
                    grad_a_all[k][row + j] = scaled * x_j;
                }
            }
            // d L / d gate[k] = s · grad_yᵀ · z_k   (z_k = B_k · A_k · x)
            d_gate[k] = s * grad_y
                .iter()
                .zip(z_k.iter())
                .map(|(a, b)| a * b)
                .sum::<f64>();
        }
        // Backprop d L / d gate into d L / d logits through the top-k masked softmax.
        // For a softmax `g = softmax(l/τ)`, ∂g_i/∂l_j = (g_i (δ_ij − g_j)) / τ.
        // After top-k masking only the selected indices have non-zero entries; the
        // resulting Jacobian restricted to those indices is the standard softmax
        // Jacobian (scaled by 1/τ), so the same identity applies on the kept set.
        let mut d_logits = vec![0.0_f64; k_exp];
        for &i in &selected {
            let mut acc = 0.0_f64;
            for &j in &selected {
                let delta = if i == j { 1.0 } else { 0.0 };
                acc += d_gate[j] * gates[i] * (delta - gates[j]);
            }
            d_logits[i] = acc / tau;
        }
        // grad_w_gate row k = d L / d logits[k] · xᵀ
        let mut grad_w_gate = vec![0.0_f64; k_exp * in_f];
        for (k, &d_lk) in d_logits.iter().enumerate() {
            let row = k * in_f;
            for (j, x_j) in x.iter().enumerate() {
                grad_w_gate[row + j] = d_lk * x_j;
            }
        }
        Ok((grad_a_all, grad_b_all, grad_w_gate))
    }

    fn compute_logits(&self, x: &[f64]) -> Vec<f64> {
        let k_exp = self.cfg.n_experts;
        let in_f = self.cfg.in_features;
        let mut logits = vec![0.0_f64; k_exp];
        for (k, l_k) in logits.iter_mut().enumerate() {
            let row = k * in_f;
            let mut acc = 0.0_f64;
            for (j, x_j) in x.iter().enumerate() {
                acc += self.w_gate[row + j] * x_j;
            }
            *l_k = acc;
        }
        logits
    }

    fn route(&self, x: &[f64]) -> (Vec<f64>, Vec<usize>) {
        let logits = self.compute_logits(x);
        let scaled: Vec<f64> = logits.iter().map(|l| l / self.cfg.temperature).collect();
        // Pick top_k indices by scaled-logit value (which is the same ordering as gates).
        let mut idx: Vec<usize> = (0..self.cfg.n_experts).collect();
        idx.sort_by(|&a, &b| {
            scaled[b]
                .partial_cmp(&scaled[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let selected: Vec<usize> = idx.into_iter().take(self.cfg.top_k).collect();
        // Numerically stable softmax over the selected subset only.
        let max_l = selected
            .iter()
            .map(|&k| scaled[k])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut gates = vec![0.0_f64; self.cfg.n_experts];
        let mut sum_exp = 0.0_f64;
        for &k in &selected {
            let e = (scaled[k] - max_l).exp();
            gates[k] = e;
            sum_exp += e;
        }
        if sum_exp > 0.0 {
            for &k in &selected {
                gates[k] /= sum_exp;
            }
        }
        (gates, selected)
    }

    fn combine(&self, x: &[f64], gates: &[f64], selected: &[usize]) -> Vec<f64> {
        let r = self.cfg.rank;
        let in_f = self.cfg.in_features;
        let out = self.cfg.out_features;
        let s = self.scale();
        let mut y = vec![0.0_f64; out];
        for &k in selected {
            let g_k = gates[k];
            if g_k == 0.0 {
                continue;
            }
            let t_k = mat_vec(&self.a_experts[k], x, r, in_f);
            let z_k = mat_vec(&self.b_experts[k], &t_k, out, r);
            for (y_i, z_i) in y.iter_mut().zip(z_k.iter()) {
                *y_i += s * g_k * z_i;
            }
        }
        y
    }
}

fn mat_vec(m: &[f64], v: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; rows];
    for (i, o_i) in out.iter_mut().enumerate() {
        let row = i * cols;
        let mut acc = 0.0_f64;
        for (j, v_j) in v.iter().enumerate() {
            acc += m[row + j] * v_j;
        }
        *o_i = acc;
    }
    out
}
