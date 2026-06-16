//! MoSA — Mixture of Sparse Adapters for parameter-efficient fine-tuning.
//!
//! Reference: Zeng, Q. et al. (2024). *MoSA: Mixture of Sparse Adapters for Visual
//! Efficient Tuning*. <https://arxiv.org/abs/2312.02923>
//!
//! A single LoRA adapter is split into `n_experts` *sparse* low-rank experts. Each expert
//! `e` owns its own factor pair `(A_e ∈ ℝ^{rank×in}, B_e ∈ ℝ^{out×rank})`, but only a fixed
//! fraction (`density`) of every expert's parameters is kept non-zero — the remainder is
//! forced to zero by a static binary mask `mask_e`. The experts are combined through an
//! MoE-style softmax gate `g(x)` with optional top-`k` routing:
//!
//! ```text
//!   logits = W_g · x                         (W_g ∈ ℝ^{n_experts × in})
//!   g(x)   = softmax( logits / τ )  over the top-k experts (others zeroed)
//!   y      = W₀ · x  +  s · Σ_e  g_e(x) · (B_e ⊙ mask_e^B) · (A_e ⊙ mask_e^A) · x
//! ```
//!
//! with effective scale `s = α / rank`. The static masks make every expert genuinely sparse
//! (that is the parameter-efficiency mechanism of MoSA), while the router specialises experts
//! on different inputs. With `n_experts = 1`, `density = 1.0` and a single selected expert the
//! gate degenerates to `1.0` and the forward pass collapses to a plain LoRA layer.

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;
use crate::lora::lora::mat_vec_mul;

/// Hyper-parameter bundle for a [`MosaAdapter`].
#[derive(Debug, Clone)]
pub struct MosaConfig {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Low-rank dimension shared by every expert.
    pub rank: usize,
    /// LoRA scaling factor `α`; the effective scale is `s = α / rank`.
    pub alpha: f32,
    /// Number of sparse experts.
    pub n_experts: usize,
    /// Number of experts kept per input after top-`k` routing.
    pub top_k: usize,
    /// Fraction of each expert's parameters kept non-zero, in `(0, 1]`
    /// (`1.0` = fully dense, smaller = sparser). The mask sparsity is `1 - density`.
    pub density: f32,
    /// Standard deviation used to initialise each expert's `A` factor.
    pub init_scale: f32,
    /// Softmax temperature `τ` for the gate; lower values sharpen routing.
    pub temperature: f32,
}

impl MosaConfig {
    /// Effective scale `s = α / rank` (returns `0.0` for an empty rank).
    #[must_use]
    pub fn scale(&self) -> f32 {
        if self.rank == 0 {
            0.0
        } else {
            self.alpha / self.rank as f32
        }
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// - [`PeftError::EmptyInput`] when any count is zero.
    /// - [`PeftError::RankTooLarge`] when `rank > min(in_features, out_features)`.
    /// - [`PeftError::InvalidDensity`] when `density` is outside `(0, 1]`.
    /// - [`PeftError::Internal`] when `top_k > n_experts` or `temperature ≤ 0`.
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
        if !(self.density > 0.0 && self.density <= 1.0) {
            return Err(PeftError::InvalidDensity {
                density: self.density,
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

/// Mixture-of-Sparse-Adapters layer with `n_experts` masked low-rank experts and a linear gate.
#[derive(Debug, Clone)]
pub struct MosaAdapter {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Low-rank dimension shared by every expert.
    pub rank: usize,
    /// Number of sparse experts.
    pub n_experts: usize,
    /// Experts kept per input after top-`k` routing.
    pub top_k: usize,
    /// Fraction of each expert's parameters kept non-zero.
    pub density: f32,
    /// Softmax temperature `τ`.
    pub temperature: f32,
    /// Effective scale `s = α / rank`.
    pub scale: f32,
    /// Frozen base weight, row-major `[out_features × in_features]`.
    pub w: Vec<f32>,
    /// Per-expert down-projections `A_e`, each row-major `[rank × in_features]`.
    pub a: Vec<Vec<f32>>,
    /// Per-expert up-projections `B_e`, each row-major `[out_features × rank]`.
    pub b: Vec<Vec<f32>>,
    /// Per-expert binary masks for `A_e`, same shape as `a`.
    pub mask_a: Vec<Vec<f32>>,
    /// Per-expert binary masks for `B_e`, same shape as `b`.
    pub mask_b: Vec<Vec<f32>>,
    /// Gating matrix `W_g`, row-major `[n_experts × in_features]`.
    pub w_gate: Vec<f32>,
}

impl MosaAdapter {
    /// Build a fresh MoSA adapter.
    ///
    /// `W₀` is zero-initialised; each `A_e ~ N(0, init_scale²)`; each `B_e` is zero
    /// (so the initial adapter delta is zero, matching the LoRA convention); the gate
    /// `W_g ~ N(0, 1/in_features)`. Each expert's sparsity mask keeps exactly
    /// [`Self::expected_nnz`] non-zero parameters drawn uniformly without replacement.
    ///
    /// # Errors
    ///
    /// Forwards [`MosaConfig::validate`].
    pub fn new(cfg: MosaConfig, rng: &mut LcgRng) -> PeftResult<Self> {
        cfg.validate()?;
        let in_f = cfg.in_features;
        let out_f = cfg.out_features;
        let r = cfg.rank;
        let scale = cfg.scale();
        let w = vec![0.0_f32; out_f * in_f];

        let mut a = Vec::with_capacity(cfg.n_experts);
        let mut b = Vec::with_capacity(cfg.n_experts);
        let mut mask_a = Vec::with_capacity(cfg.n_experts);
        let mut mask_b = Vec::with_capacity(cfg.n_experts);
        for _ in 0..cfg.n_experts {
            let mut a_e = vec![0.0_f32; r * in_f];
            rng.fill_normal(&mut a_e);
            for v in a_e.iter_mut() {
                *v *= cfg.init_scale;
            }
            a.push(a_e);
            b.push(vec![0.0_f32; out_f * r]);
            let (m_a, m_b) = build_sparse_masks(in_f, r, out_f, cfg.density, rng);
            mask_a.push(m_a);
            mask_b.push(m_b);
        }

        let g_std = 1.0_f32 / (in_f as f32).sqrt();
        let mut w_gate = vec![0.0_f32; cfg.n_experts * in_f];
        rng.fill_normal(&mut w_gate);
        for v in w_gate.iter_mut() {
            *v *= g_std;
        }

        Ok(Self {
            in_features: in_f,
            out_features: out_f,
            rank: r,
            n_experts: cfg.n_experts,
            top_k: cfg.top_k,
            density: cfg.density,
            temperature: cfg.temperature,
            scale,
            w,
            a,
            b,
            mask_a,
            mask_b,
            w_gate,
        })
    }

    /// Number of non-zero parameters every expert keeps after masking,
    /// `round(density · (rank·in + out·rank))`, clamped to the parameter count.
    #[must_use]
    pub fn expected_nnz(&self) -> usize {
        let total = self.rank * self.in_features + self.out_features * self.rank;
        ((self.density * total as f32).round() as usize).min(total)
    }

    /// Actual non-zero count of expert `e`'s combined mask (`A` and `B`).
    ///
    /// # Errors
    ///
    /// [`PeftError::LayerOutOfRange`] when `e >= n_experts`.
    pub fn expert_nnz(&self, e: usize) -> PeftResult<usize> {
        self.check_expert(e)?;
        let na = self.mask_a[e].iter().filter(|&&m| m != 0.0).count();
        let nb = self.mask_b[e].iter().filter(|&&m| m != 0.0).count();
        Ok(na + nb)
    }

    /// Forward pass `y = W₀·x + s·Σ_e g_e(x)·(B_e⊙m^B_e)(A_e⊙m^A_e)·x`.
    ///
    /// # Errors
    ///
    /// [`PeftError::DimensionMismatch`] when `x.len() != in_features`.
    pub fn forward(&self, x: &[f32]) -> PeftResult<Vec<f32>> {
        let (y, _gates, _selected) = self.forward_with_route(x)?;
        Ok(y)
    }

    /// Forward pass that also returns the full-length gate vector and the selected experts.
    ///
    /// The returned `gates` vector has length `n_experts`, is non-negative and sums to `1`;
    /// entries of experts not selected by top-`k` are exactly `0`.
    ///
    /// # Errors
    ///
    /// [`PeftError::DimensionMismatch`] when `x.len() != in_features`.
    pub fn forward_with_route(&self, x: &[f32]) -> PeftResult<(Vec<f32>, Vec<f32>, Vec<usize>)> {
        if x.len() != self.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.in_features,
                got: x.len(),
            });
        }
        let (gates, selected) = self.route(x);
        let mut y = mat_vec_mul(&self.w, x, self.out_features, self.in_features);
        for &e in &selected {
            let g_e = gates[e];
            if g_e == 0.0 {
                continue;
            }
            let z = self.expert_apply(e, x);
            for (y_i, z_i) in y.iter_mut().zip(z.iter()) {
                *y_i += g_e * z_i;
            }
        }
        Ok((y, gates, selected))
    }

    /// Softmax-over-top-`k` gate vector for `x` (length `n_experts`, sums to one).
    ///
    /// # Errors
    ///
    /// [`PeftError::DimensionMismatch`] when `x.len() != in_features`.
    pub fn gate(&self, x: &[f32]) -> PeftResult<Vec<f32>> {
        if x.len() != self.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.in_features,
                got: x.len(),
            });
        }
        Ok(self.route(x).0)
    }

    /// Per-expert sparse delta `s·(B_e⊙m^B_e)(A_e⊙m^A_e)`, row-major `[out_features × in_features]`.
    ///
    /// # Errors
    ///
    /// [`PeftError::LayerOutOfRange`] when `e >= n_experts`.
    pub fn expert_delta(&self, e: usize) -> PeftResult<Vec<f32>> {
        self.check_expert(e)?;
        let in_f = self.in_features;
        let r = self.rank;
        let out_f = self.out_features;
        let a_e = &self.a[e];
        let ma = &self.mask_a[e];
        let b_e = &self.b[e];
        let mb = &self.mask_b[e];
        let mut delta = vec![0.0_f32; out_f * in_f];
        for i in 0..out_f {
            for k in 0..r {
                let b_ik = b_e[i * r + k] * mb[i * r + k];
                if b_ik == 0.0 {
                    continue;
                }
                let s_b = self.scale * b_ik;
                for j in 0..in_f {
                    delta[i * in_f + j] += s_b * a_e[k * in_f + j] * ma[k * in_f + j];
                }
            }
        }
        Ok(delta)
    }

    /// Input-dependent effective delta `Σ_e g_e(x)·s·(B_e⊙m^B_e)(A_e⊙m^A_e)`,
    /// row-major `[out_features × in_features]`. This is the matrix that, applied to `x`,
    /// reproduces the adapter contribution of [`Self::forward`] (i.e. `y - W₀·x`).
    ///
    /// # Errors
    ///
    /// [`PeftError::DimensionMismatch`] when `x.len() != in_features`.
    pub fn effective_delta(&self, x: &[f32]) -> PeftResult<Vec<f32>> {
        if x.len() != self.in_features {
            return Err(PeftError::DimensionMismatch {
                expected: self.in_features,
                got: x.len(),
            });
        }
        let (gates, selected) = self.route(x);
        let mut acc = vec![0.0_f32; self.out_features * self.in_features];
        for &e in &selected {
            let g_e = gates[e];
            if g_e == 0.0 {
                continue;
            }
            let de = self.expert_delta(e)?;
            for (a_i, d_i) in acc.iter_mut().zip(de.iter()) {
                *a_i += g_e * d_i;
            }
        }
        Ok(acc)
    }

    fn check_expert(&self, e: usize) -> PeftResult<()> {
        if e >= self.n_experts {
            return Err(PeftError::LayerOutOfRange {
                idx: e,
                num_layers: self.n_experts,
            });
        }
        Ok(())
    }

    /// Apply expert `e` to `x`, returning `s·(B_e⊙m^B_e)(A_e⊙m^A_e)·x` (length `out_features`).
    fn expert_apply(&self, e: usize, x: &[f32]) -> Vec<f32> {
        let in_f = self.in_features;
        let r = self.rank;
        let out_f = self.out_features;
        let a_e = &self.a[e];
        let ma = &self.mask_a[e];
        let b_e = &self.b[e];
        let mb = &self.mask_b[e];
        // t = (A_e ⊙ m^A_e) · x   (length rank)
        let mut t = vec![0.0_f32; r];
        for (k, t_k) in t.iter_mut().enumerate() {
            let row = k * in_f;
            let mut sum = 0.0_f32;
            for (j, x_j) in x.iter().enumerate() {
                sum += a_e[row + j] * ma[row + j] * x_j;
            }
            *t_k = sum;
        }
        // z = s · (B_e ⊙ m^B_e) · t   (length out_features)
        let mut z = vec![0.0_f32; out_f];
        for (i, z_i) in z.iter_mut().enumerate() {
            let row = i * r;
            let mut sum = 0.0_f32;
            for (k, t_k) in t.iter().enumerate() {
                sum += b_e[row + k] * mb[row + k] * t_k;
            }
            *z_i = self.scale * sum;
        }
        z
    }

    fn gate_logits(&self, x: &[f32]) -> Vec<f32> {
        let in_f = self.in_features;
        let mut logits = vec![0.0_f32; self.n_experts];
        for (e, l_e) in logits.iter_mut().enumerate() {
            let row = e * in_f;
            let mut sum = 0.0_f32;
            for (j, x_j) in x.iter().enumerate() {
                sum += self.w_gate[row + j] * x_j;
            }
            *l_e = sum;
        }
        logits
    }

    fn route(&self, x: &[f32]) -> (Vec<f32>, Vec<usize>) {
        let logits = self.gate_logits(x);
        let scaled: Vec<f32> = logits.iter().map(|l| l / self.temperature).collect();
        let mut idx: Vec<usize> = (0..self.n_experts).collect();
        idx.sort_by(|&a, &b| {
            scaled[b]
                .partial_cmp(&scaled[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let selected: Vec<usize> = idx.into_iter().take(self.top_k).collect();
        let max_l = selected
            .iter()
            .map(|&k| scaled[k])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut gates = vec![0.0_f32; self.n_experts];
        let mut sum_exp = 0.0_f32;
        for &k in &selected {
            let ex = (scaled[k] - max_l).exp();
            gates[k] = ex;
            sum_exp += ex;
        }
        if sum_exp > 0.0 {
            for &k in &selected {
                gates[k] /= sum_exp;
            }
        }
        (gates, selected)
    }
}

/// Build the per-expert sparsity masks `(mask_a, mask_b)` keeping exactly
/// `round(density · total)` non-zero positions, chosen uniformly without replacement
/// across the concatenated `[A | B]` parameter space via partial Fisher–Yates.
fn build_sparse_masks(
    in_f: usize,
    r: usize,
    out_f: usize,
    density: f32,
    rng: &mut LcgRng,
) -> (Vec<f32>, Vec<f32>) {
    let a_len = r * in_f;
    let b_len = out_f * r;
    let total = a_len + b_len;
    let nnz = ((density * total as f32).round() as usize).min(total);
    let mut idx: Vec<usize> = (0..total).collect();
    for i in 0..nnz {
        let j = i + rng.next_usize(total - i);
        idx.swap(i, j);
    }
    let mut mask_a = vec![0.0_f32; a_len];
    let mut mask_b = vec![0.0_f32; b_len];
    for &p in idx.iter().take(nnz) {
        if p < a_len {
            mask_a[p] = 1.0;
        } else {
            mask_b[p - a_len] = 1.0;
        }
    }
    (mask_a, mask_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lora::lora::LoraLinear;

    fn cfg(n_experts: usize, top_k: usize, density: f32) -> MosaConfig {
        MosaConfig {
            in_features: 6,
            out_features: 5,
            rank: 3,
            alpha: 6.0,
            n_experts,
            top_k,
            density,
            init_scale: 0.05,
            temperature: 1.0,
        }
    }

    #[test]
    fn mask_nnz_exactly_enforced() {
        let mut rng = LcgRng::new(1);
        let m = MosaAdapter::new(cfg(4, 2, 0.5), &mut rng)
            .expect("MoSA adapter creation should succeed with valid config");
        let expected = m.expected_nnz();
        // total params per expert = 3*6 + 5*3 = 18 + 15 = 33, density 0.5 -> round = 16 or 17.
        assert!(expected > 0 && expected < 33);
        for e in 0..m.n_experts {
            assert_eq!(
                m.expert_nnz(e).expect("expert_nnz should succeed"),
                expected,
                "expert {e} mask nnz mismatch"
            );
        }
    }

    #[test]
    fn full_density_keeps_all_params() {
        let mut rng = LcgRng::new(2);
        let m = MosaAdapter::new(cfg(3, 3, 1.0), &mut rng).expect("value should be present");
        let total = m.rank * m.in_features + m.out_features * m.rank;
        assert_eq!(m.expected_nnz(), total);
        for e in 0..m.n_experts {
            assert_eq!(m.expert_nnz(e).expect("expert_nnz should succeed"), total);
            assert!(m.mask_a[e].iter().all(|&v| v == 1.0));
            assert!(m.mask_b[e].iter().all(|&v| v == 1.0));
        }
    }

    #[test]
    fn gates_nonneg_and_sum_to_one() {
        let mut rng = LcgRng::new(3);
        let m = MosaAdapter::new(cfg(5, 3, 0.4), &mut rng).expect("value should be present");
        for s in 0..4 {
            let x: Vec<f32> = (0..m.in_features)
                .map(|i| (i as f32 - 2.0) * (s as f32 + 1.0) * 0.3)
                .collect();
            let (_y, gates, _sel) = m
                .forward_with_route(&x)
                .expect("forward_with_route should succeed");
            assert_eq!(gates.len(), m.n_experts);
            let mut sum = 0.0_f32;
            for &g in &gates {
                assert!(g >= 0.0, "gate must be non-negative, got {g}");
                sum += g;
            }
            assert!((sum - 1.0).abs() < 1e-5, "gates must sum to 1, got {sum}");
        }
    }

    #[test]
    fn top_k_zeros_non_selected_experts() {
        let mut rng = LcgRng::new(4);
        let m = MosaAdapter::new(cfg(4, 2, 0.5), &mut rng).expect("value should be present");
        let x: Vec<f32> = (0..m.in_features).map(|i| (i as f32 + 1.0) * 0.4).collect();
        let (_y, gates, selected) = m
            .forward_with_route(&x)
            .expect("forward_with_route should succeed");
        assert_eq!(selected.len(), 2, "top_k experts must be selected");
        let nonzero = gates.iter().filter(|&&g| g != 0.0).count();
        assert_eq!(nonzero, 2, "exactly top_k gates may be non-zero");
        for (e, &g) in gates.iter().enumerate() {
            if !selected.contains(&e) {
                assert_eq!(g, 0.0, "non-selected expert {e} must have zero gate");
            }
        }
    }

    #[test]
    fn output_shape_correct() {
        let mut rng = LcgRng::new(5);
        let m = MosaAdapter::new(cfg(3, 2, 0.6), &mut rng).expect("value should be present");
        let x = vec![0.2_f32; m.in_features];
        let y = m.forward(&x).expect("forward should succeed");
        assert_eq!(y.len(), m.out_features);
    }

    #[test]
    fn single_expert_full_density_matches_plain_lora() {
        let mut rng = LcgRng::new(6);
        let mut m = MosaAdapter::new(cfg(1, 1, 1.0), &mut rng).expect("value should be present");
        // Make the adapter non-trivial: set W₀ and the single expert's B to known values.
        for (i, w) in m.w.iter_mut().enumerate() {
            *w = (i as f32 % 7.0) * 0.03 - 0.1;
        }
        for (i, bv) in m.b[0].iter_mut().enumerate() {
            *bv = (i as f32 % 5.0) * 0.07 - 0.15;
        }
        // Build an equivalent plain LoRA from the same parameters.
        let lora = LoraLinear {
            in_features: m.in_features,
            out_features: m.out_features,
            rank: m.rank,
            scale: m.scale,
            w: m.w.clone(),
            a: m.a[0].clone(),
            b: m.b[0].clone(),
        };
        let x: Vec<f32> = (0..m.in_features)
            .map(|i| (i as f32 - 3.0) * 0.25)
            .collect();
        let y_mosa = m.forward(&x).expect("forward should succeed");
        let y_lora = lora.forward(&x);
        assert_eq!(y_mosa.len(), y_lora.len());
        for (a, b) in y_mosa.iter().zip(y_lora.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "MoSA(1 expert, dense, uniform gate) must equal plain LoRA: {a} vs {b}"
            );
        }
    }

    #[test]
    fn effective_delta_reproduces_adapter_contribution() {
        let mut rng = LcgRng::new(7);
        let mut m = MosaAdapter::new(cfg(4, 2, 0.7), &mut rng).expect("value should be present");
        for bv in m.b.iter_mut() {
            rng.fill_normal(bv);
            for v in bv.iter_mut() {
                *v *= 0.1;
            }
        }
        let x: Vec<f32> = (0..m.in_features).map(|i| (i as f32 + 1.0) * 0.2).collect();
        let y = m.forward(&x).expect("forward should succeed");
        let base = mat_vec_mul(&m.w, &x, m.out_features, m.in_features);
        let delta = m
            .effective_delta(&x)
            .expect("effective_delta should succeed");
        let dx = mat_vec_mul(&delta, &x, m.out_features, m.in_features);
        for i in 0..m.out_features {
            let recon = base[i] + dx[i];
            assert!(
                (recon - y[i]).abs() < 1e-4,
                "effective_delta mismatch at {i}: {recon} vs {}",
                y[i]
            );
        }
    }

    #[test]
    fn outputs_are_finite() {
        let mut rng = LcgRng::new(8);
        let mut m = MosaAdapter::new(cfg(5, 3, 0.5), &mut rng).expect("value should be present");
        for bv in m.b.iter_mut() {
            rng.fill_normal(bv);
        }
        let x: Vec<f32> = (0..m.in_features).map(|i| (i as f32 - 2.5) * 1.7).collect();
        let y = m.forward(&x).expect("forward should succeed");
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn deterministic_for_same_seed() {
        let mut r1 = LcgRng::new(9);
        let mut r2 = LcgRng::new(9);
        let m1 = MosaAdapter::new(cfg(4, 2, 0.5), &mut r1).expect("value should be present");
        let m2 = MosaAdapter::new(cfg(4, 2, 0.5), &mut r2).expect("value should be present");
        let x: Vec<f32> = (0..m1.in_features)
            .map(|i| (i as f32 + 0.5) * 0.3)
            .collect();
        let y1 = m1.forward(&x).expect("forward should succeed");
        let y2 = m2.forward(&x).expect("forward should succeed");
        assert_eq!(y1, y2);
        for e in 0..m1.n_experts {
            assert_eq!(m1.mask_a[e], m2.mask_a[e]);
            assert_eq!(m1.mask_b[e], m2.mask_b[e]);
        }
    }

    #[test]
    fn invalid_config_rejected() {
        let mut rng = LcgRng::new(10);
        let mut bad = cfg(2, 5, 0.5); // top_k > n_experts
        assert!(MosaAdapter::new(bad.clone(), &mut rng).is_err());
        bad = cfg(2, 1, 1.5); // density > 1
        assert!(MosaAdapter::new(bad, &mut rng).is_err());
    }
}
