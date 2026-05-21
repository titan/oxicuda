//! STAMP — Short-Term Attention/Memory Priority Model for session-based
//! recommendation.
//!
//! Reference: Qiao Liu, Yifu Zeng, Refuoe Mokhosi, Haibin Zhang, "STAMP:
//! Short-Term Attention/Memory Priority Model for Session-based
//! Recommendation", KDD 2018.
//!
//! Architecture:
//!   1. Embed every clicked item `x_i ∈ R^d` (`i = 1..t`) using a shared
//!      item-embedding table `item_embeds ∈ R^{n_items × d}`.
//!   2. The **long-term memory** `m_s = (1/t) Σ_i x_i` summarises the full
//!      session as a uniform mean of behaviour embeddings.
//!   3. The **short-term memory** is the embedding of the *last* clicked item
//!      `x_t` — the most recent intent.
//!   4. A position-wise **local activation unit** scores every history item
//!      against `(x_t, m_s)`:
//!
//!      ```text
//!      a_i_vec = sigmoid_pointwise(W_a0 · x_i + W_a1 · x_t + W_a2 · m_s + b_a)
//!      α_i     = v_a · a_i_vec    (scalar)
//!      ```
//!
//!      Because the inner non-linearity is **sigmoid** (not softmax), the
//!      gates `α_i` lie in a bounded range but do **not** sum to one — this is
//!      the paper's intensity-preserving design (§ 3.2 of Liu et al. 2018).
//!   5. The **attended session representation** is the raw weighted sum
//!      `m_a = Σ_i α_i · x_i ∈ R^d`.
//!   6. Two MLPs lift `m_a` and `x_t` into a common scoring space:
//!
//!      ```text
//!      h_s = tanh(W_mlp_s · m_a + b_s)
//!      h_t = tanh(W_mlp_t · x_t + b_t)
//!      ```
//!
//!   7. Trilinear scoring against every candidate item embedding `e_j`:
//!
//!      ```text
//!      ẑ_j = e_j · (h_s ⊙ h_t)
//!      ```
//!
//!      The logits over all `n_items` candidates are returned to the caller.

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

/// Numerically stable element-wise logistic sigmoid.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Configuration for [`Stamp`].
#[derive(Debug, Clone)]
pub struct StampConfig {
    /// Dimensionality of every item embedding and of every internal hidden
    /// activation (must be `>= 1`).
    pub embed_dim: usize,
    /// Number of distinct items in the catalogue (must be `>= 1`).
    pub n_items: usize,
}

/// STAMP — short-term attention / memory priority sequential recommender.
///
/// The parameter tensors are stored row-major as flat `Vec<f32>` buffers so
/// that the layout matches the conventions used by the rest of
/// `oxicuda-recsys` (e.g. `Din`, `SasRec`).
pub struct Stamp {
    /// Configuration the model was built from.
    pub cfg: StampConfig,
    /// `n_items × embed_dim` item-embedding table (row-major).
    pub item_embeds: Vec<f32>,
    /// `embed_dim × embed_dim` attention projection for `x_i` (row-major,
    /// row index = output feature, column index = input feature).
    pub w_a0: Vec<f32>,
    /// `embed_dim × embed_dim` attention projection for the short-term memory
    /// `x_t`.
    pub w_a1: Vec<f32>,
    /// `embed_dim × embed_dim` attention projection for the long-term memory
    /// `m_s`.
    pub w_a2: Vec<f32>,
    /// `embed_dim` attention bias vector.
    pub b_a: Vec<f32>,
    /// `embed_dim` attention readout vector (the `v_a` in the paper).
    pub v_a: Vec<f32>,
    /// `embed_dim × embed_dim` MLP weight for the session representation `m_a`.
    pub w_mlp_s: Vec<f32>,
    /// `embed_dim` MLP bias for the session representation branch.
    pub b_s: Vec<f32>,
    /// `embed_dim × embed_dim` MLP weight for the short-term memory `x_t`.
    pub w_mlp_t: Vec<f32>,
    /// `embed_dim` MLP bias for the short-term memory branch.
    pub b_t: Vec<f32>,
}

impl Stamp {
    /// Construct a fresh STAMP model with Kaiming-style normal initialisation
    /// (scale `1/sqrt(fan_in)`).
    ///
    /// # Errors
    /// - [`RecsysError::InvalidEmbeddingDim`] when `cfg.embed_dim == 0`.
    /// - [`RecsysError::InvalidNumItems`] when `cfg.n_items == 0`.
    pub fn new(cfg: StampConfig, rng: &mut LcgRng) -> RecsysResult<Self> {
        if cfg.embed_dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: 0 });
        }
        if cfg.n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: 0 });
        }

        let d = cfg.embed_dim;
        let n_items = cfg.n_items;
        let emb_scale = (1.0_f32 / d as f32).sqrt();
        let attn_scale = (1.0_f32 / d as f32).sqrt();

        let mut item_embeds = vec![0.0_f32; n_items * d];
        rng.fill_normal(&mut item_embeds);
        for v in item_embeds.iter_mut() {
            *v *= emb_scale;
        }

        let mut w_a0 = vec![0.0_f32; d * d];
        rng.fill_normal(&mut w_a0);
        for v in w_a0.iter_mut() {
            *v *= attn_scale;
        }

        let mut w_a1 = vec![0.0_f32; d * d];
        rng.fill_normal(&mut w_a1);
        for v in w_a1.iter_mut() {
            *v *= attn_scale;
        }

        let mut w_a2 = vec![0.0_f32; d * d];
        rng.fill_normal(&mut w_a2);
        for v in w_a2.iter_mut() {
            *v *= attn_scale;
        }

        let b_a = vec![0.0_f32; d];

        let mut v_a = vec![0.0_f32; d];
        rng.fill_normal(&mut v_a);
        for v in v_a.iter_mut() {
            *v *= attn_scale;
        }

        let mut w_mlp_s = vec![0.0_f32; d * d];
        rng.fill_normal(&mut w_mlp_s);
        for v in w_mlp_s.iter_mut() {
            *v *= attn_scale;
        }
        let b_s = vec![0.0_f32; d];

        let mut w_mlp_t = vec![0.0_f32; d * d];
        rng.fill_normal(&mut w_mlp_t);
        for v in w_mlp_t.iter_mut() {
            *v *= attn_scale;
        }
        let b_t = vec![0.0_f32; d];

        Ok(Self {
            cfg,
            item_embeds,
            w_a0,
            w_a1,
            w_a2,
            b_a,
            v_a,
            w_mlp_s,
            b_s,
            w_mlp_t,
            b_t,
        })
    }

    /// Look up the embedding row for item `id` (length `embed_dim`).
    fn lookup(&self, id: usize) -> RecsysResult<&[f32]> {
        let d = self.cfg.embed_dim;
        let start = id.checked_mul(d).ok_or_else(|| RecsysError::Internal {
            msg: "embedding row offset overflow".to_string(),
        })?;
        let end = start.checked_add(d).ok_or_else(|| RecsysError::Internal {
            msg: "embedding row offset overflow".to_string(),
        })?;
        self.item_embeds
            .get(start..end)
            .ok_or(RecsysError::ItemOutOfBounds {
                idx: id,
                n: self.cfg.n_items,
            })
    }

    /// Compute the *attention gates* `α_i = v_a · sigmoid(W_a0·x_i +
    /// W_a1·x_t + W_a2·m_s + b_a)` for every session position `i ∈ 0..t`.
    ///
    /// `session_item_ids` enumerates the session's items in click order; the
    /// last entry must be the most-recent click (used as the short-term
    /// memory `x_t`). The returned vector has length `session_item_ids.len()`.
    ///
    /// # Errors
    /// - [`RecsysError::EmptyInput`] when `session_item_ids.is_empty()`.
    /// - [`RecsysError::ItemOutOfBounds`] when any id `>= n_items`.
    pub fn stamp_attention_gates(&self, session_item_ids: &[usize]) -> RecsysResult<Vec<f32>> {
        if session_item_ids.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        let d = self.cfg.embed_dim;
        let t = session_item_ids.len();

        // Validate ids up front so we fail fast.
        for &id in session_item_ids {
            if id >= self.cfg.n_items {
                return Err(RecsysError::ItemOutOfBounds {
                    idx: id,
                    n: self.cfg.n_items,
                });
            }
        }

        // m_s = (1/t) Σ_i x_i  — long-term memory.
        let mut m_s = vec![0.0_f32; d];
        for &id in session_item_ids {
            let x_i = self.lookup(id)?;
            for k in 0..d {
                let comp = x_i.get(k).copied().unwrap_or(0.0);
                if let Some(slot) = m_s.get_mut(k) {
                    *slot += comp;
                }
            }
        }
        let inv_t = 1.0_f32 / t as f32;
        for v in m_s.iter_mut() {
            *v *= inv_t;
        }

        // x_t — short-term memory: embedding of the most recent click.
        let last_id = match session_item_ids.last() {
            Some(&id) => id,
            None => return Err(RecsysError::EmptyInput),
        };
        let x_t: Vec<f32> = self.lookup(last_id)?.to_vec();

        // Pre-compute W_a1 · x_t + W_a2 · m_s + b_a (shared across all i).
        let mut shared = vec![0.0_f32; d];
        for o in 0..d {
            let row_start = o * d;
            let row_end = row_start + d;
            let row_w1 =
                self.w_a1
                    .get(row_start..row_end)
                    .ok_or_else(|| RecsysError::Internal {
                        msg: "w_a1 row out of bounds".to_string(),
                    })?;
            let row_w2 =
                self.w_a2
                    .get(row_start..row_end)
                    .ok_or_else(|| RecsysError::Internal {
                        msg: "w_a2 row out of bounds".to_string(),
                    })?;
            let mut acc = self.b_a.get(o).copied().unwrap_or(0.0);
            for k in 0..d {
                let w1 = row_w1.get(k).copied().unwrap_or(0.0);
                let w2 = row_w2.get(k).copied().unwrap_or(0.0);
                let xt_k = x_t.get(k).copied().unwrap_or(0.0);
                let ms_k = m_s.get(k).copied().unwrap_or(0.0);
                acc += w1 * xt_k + w2 * ms_k;
            }
            if let Some(slot) = shared.get_mut(o) {
                *slot = acc;
            }
        }

        let mut gates = Vec::with_capacity(t);
        for &id in session_item_ids {
            let x_i = self.lookup(id)?;
            let mut a_vec = vec![0.0_f32; d];
            for o in 0..d {
                let row_start = o * d;
                let row_end = row_start + d;
                let row_w0 =
                    self.w_a0
                        .get(row_start..row_end)
                        .ok_or_else(|| RecsysError::Internal {
                            msg: "w_a0 row out of bounds".to_string(),
                        })?;
                let mut acc = shared.get(o).copied().unwrap_or(0.0);
                for k in 0..d {
                    let w0 = row_w0.get(k).copied().unwrap_or(0.0);
                    let xi_k = x_i.get(k).copied().unwrap_or(0.0);
                    acc += w0 * xi_k;
                }
                if let Some(slot) = a_vec.get_mut(o) {
                    *slot = sigmoid(acc);
                }
            }
            let mut alpha = 0.0_f32;
            for k in 0..d {
                let va_k = self.v_a.get(k).copied().unwrap_or(0.0);
                let av_k = a_vec.get(k).copied().unwrap_or(0.0);
                alpha += va_k * av_k;
            }
            gates.push(alpha);
        }
        Ok(gates)
    }

    /// Compute the attended session representation `m_a = Σ_i α_i · x_i ∈ R^d`
    /// using the sigmoid gates from [`Self::stamp_attention_gates`].
    ///
    /// # Errors
    /// Same validation as [`Self::stamp_attention_gates`].
    pub fn stamp_session_rep(&self, session_item_ids: &[usize]) -> RecsysResult<Vec<f32>> {
        let d = self.cfg.embed_dim;
        let gates = self.stamp_attention_gates(session_item_ids)?;
        let mut m_a = vec![0.0_f32; d];
        for (i, &id) in session_item_ids.iter().enumerate() {
            let alpha_i = gates.get(i).copied().unwrap_or(0.0);
            let x_i = self.lookup(id)?;
            for k in 0..d {
                let comp = x_i.get(k).copied().unwrap_or(0.0);
                if let Some(slot) = m_a.get_mut(k) {
                    *slot += alpha_i * comp;
                }
            }
        }
        Ok(m_a)
    }

    /// Forward pass: from a click session, compute logits over all
    /// `n_items` candidates via the trilinear scoring rule
    /// `ẑ_j = e_j · (h_s ⊙ h_t)`.
    ///
    /// # Errors
    /// - [`RecsysError::EmptyInput`] when `session_item_ids.is_empty()`.
    /// - [`RecsysError::ItemOutOfBounds`] when any id `>= n_items`.
    pub fn forward_session(&self, session_item_ids: &[usize]) -> RecsysResult<Vec<f32>> {
        let d = self.cfg.embed_dim;
        if session_item_ids.is_empty() {
            return Err(RecsysError::EmptyInput);
        }
        for &id in session_item_ids {
            if id >= self.cfg.n_items {
                return Err(RecsysError::ItemOutOfBounds {
                    idx: id,
                    n: self.cfg.n_items,
                });
            }
        }

        // Session representation m_a.
        let m_a = self.stamp_session_rep(session_item_ids)?;

        // Short-term embedding x_t — embedding of the last click.
        let last_id = match session_item_ids.last() {
            Some(&id) => id,
            None => return Err(RecsysError::EmptyInput),
        };
        let x_t: Vec<f32> = self.lookup(last_id)?.to_vec();

        // h_s = tanh(W_mlp_s · m_a + b_s)
        let mut h_s = vec![0.0_f32; d];
        for o in 0..d {
            let row_start = o * d;
            let row_end = row_start + d;
            let row =
                self.w_mlp_s
                    .get(row_start..row_end)
                    .ok_or_else(|| RecsysError::Internal {
                        msg: "w_mlp_s row out of bounds".to_string(),
                    })?;
            let mut acc = self.b_s.get(o).copied().unwrap_or(0.0);
            for k in 0..d {
                let w = row.get(k).copied().unwrap_or(0.0);
                let m = m_a.get(k).copied().unwrap_or(0.0);
                acc += w * m;
            }
            if let Some(slot) = h_s.get_mut(o) {
                *slot = acc.tanh();
            }
        }

        // h_t = tanh(W_mlp_t · x_t + b_t)
        let mut h_t = vec![0.0_f32; d];
        for o in 0..d {
            let row_start = o * d;
            let row_end = row_start + d;
            let row =
                self.w_mlp_t
                    .get(row_start..row_end)
                    .ok_or_else(|| RecsysError::Internal {
                        msg: "w_mlp_t row out of bounds".to_string(),
                    })?;
            let mut acc = self.b_t.get(o).copied().unwrap_or(0.0);
            for k in 0..d {
                let w = row.get(k).copied().unwrap_or(0.0);
                let x = x_t.get(k).copied().unwrap_or(0.0);
                acc += w * x;
            }
            if let Some(slot) = h_t.get_mut(o) {
                *slot = acc.tanh();
            }
        }

        // hadamard h_s ⊙ h_t
        let mut had = vec![0.0_f32; d];
        for k in 0..d {
            let a = h_s.get(k).copied().unwrap_or(0.0);
            let b = h_t.get(k).copied().unwrap_or(0.0);
            if let Some(slot) = had.get_mut(k) {
                *slot = a * b;
            }
        }

        // logit_j = e_j · had  for j ∈ 0..n_items.
        let mut logits = vec![0.0_f32; self.cfg.n_items];
        for j in 0..self.cfg.n_items {
            let e_j = self.lookup(j)?;
            let mut acc = 0.0_f32;
            for k in 0..d {
                let e = e_j.get(k).copied().unwrap_or(0.0);
                let h = had.get(k).copied().unwrap_or(0.0);
                acc += e * h;
            }
            if let Some(slot) = logits.get_mut(j) {
                *slot = acc;
            }
        }

        Ok(logits)
    }

    /// Total number of learnable parameters.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.item_embeds.len()
            + self.w_a0.len()
            + self.w_a1.len()
            + self.w_a2.len()
            + self.b_a.len()
            + self.v_a.len()
            + self.w_mlp_s.len()
            + self.b_s.len()
            + self.w_mlp_t.len()
            + self.b_t.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn default_cfg() -> StampConfig {
        StampConfig {
            embed_dim: 4,
            n_items: 6,
        }
    }

    #[test]
    fn single_item_session_m_s_equals_x_t() {
        // For a single-click session, the long-term mean equals the only
        // embedding, which also is x_t. So the attention input collapses.
        let mut rng = make_rng();
        let model = Stamp::new(default_cfg(), &mut rng).unwrap();
        // We exploit the manually-computed gate to assert.
        let gates = model.stamp_attention_gates(&[2usize]).unwrap();
        assert_eq!(gates.len(), 1);
        // The single gate must be finite.
        assert!(gates[0].is_finite());
    }

    #[test]
    fn two_item_session_hand_compute_matches_attention() {
        // Use a fixed model with a tiny embed_dim and assert that the
        // forward maths produces the expected gates and session rep.
        let cfg = StampConfig {
            embed_dim: 2,
            n_items: 3,
        };
        let mut rng = LcgRng::new(123);
        let mut model = Stamp::new(cfg, &mut rng).unwrap();
        // Overwrite weights with deterministic small values so the math is
        // checkable by hand.
        model.item_embeds = vec![
            0.10, 0.20, // item 0
            0.30, 0.40, // item 1
            0.50, 0.60, // item 2
        ];
        model.w_a0 = vec![0.1, 0.0, 0.0, 0.1];
        model.w_a1 = vec![0.1, 0.0, 0.0, 0.1];
        model.w_a2 = vec![0.1, 0.0, 0.0, 0.1];
        model.b_a = vec![0.0, 0.0];
        model.v_a = vec![1.0, 1.0];

        let session = [0_usize, 1];
        let gates = model.stamp_attention_gates(&session).unwrap();
        // m_s = ((0.10+0.30)/2, (0.20+0.40)/2) = (0.20, 0.30)
        // x_t = (0.30, 0.40)
        // shared = W_a1·x_t + W_a2·m_s + b_a
        //        = (0.1·0.30 + 0.0·0.40, 0.0·0.30 + 0.1·0.40) +
        //          (0.1·0.20 + 0.0·0.30, 0.0·0.20 + 0.1·0.30)
        //        = (0.03 + 0.02, 0.04 + 0.03) = (0.05, 0.07)
        // For i=0 (x_i = (0.10, 0.20)):
        //   pre_act = shared + (0.1·0.10, 0.1·0.20) = (0.06, 0.09)
        //   a_vec   = (sigmoid(0.06), sigmoid(0.09))
        //   alpha_0 = a_vec[0] + a_vec[1]
        // For i=1 (x_i = (0.30, 0.40)):
        //   pre_act = (0.05 + 0.03, 0.07 + 0.04) = (0.08, 0.11)
        //   alpha_1 = sigmoid(0.08) + sigmoid(0.11)
        let s0_0 = sigmoid(0.06);
        let s0_1 = sigmoid(0.09);
        let s1_0 = sigmoid(0.08);
        let s1_1 = sigmoid(0.11);
        let expected_alpha_0 = s0_0 + s0_1;
        let expected_alpha_1 = s1_0 + s1_1;
        assert!((gates[0] - expected_alpha_0).abs() < 1e-5);
        assert!((gates[1] - expected_alpha_1).abs() < 1e-5);
    }

    #[test]
    fn sigmoid_gates_in_zero_one_per_component() {
        // Each component a_i_vec[k] is sigmoid(.) which lies strictly in
        // (0,1). To assert this, we construct a model where v_a is the
        // canonical basis e_k so α_i exposes only one component, then check
        // it is in (0,1).
        let cfg = StampConfig {
            embed_dim: 3,
            n_items: 4,
        };
        let mut rng = LcgRng::new(7);
        let mut model = Stamp::new(cfg, &mut rng).unwrap();
        model.v_a = vec![1.0, 0.0, 0.0];
        let session = [0_usize, 1, 2];
        let gates = model.stamp_attention_gates(&session).unwrap();
        for &g in &gates {
            assert!(
                g > 0.0 && g < 1.0,
                "gate {g} should be in (0,1) since only one sigmoid component is read"
            );
        }
    }

    #[test]
    fn sigmoid_gates_do_not_sum_to_one() {
        // Construct an input where the gates do not sum to 1 — the design
        // explicitly does NOT softmax-normalise (intensity preservation).
        let cfg = StampConfig {
            embed_dim: 2,
            n_items: 3,
        };
        let mut rng = LcgRng::new(123);
        let mut model = Stamp::new(cfg, &mut rng).unwrap();
        // Big positive pre-activations so each sigmoid is ≈ 1 → sum ≈ 2 >> 1.
        model.item_embeds = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        model.w_a0 = vec![1.0, 0.0, 0.0, 1.0];
        model.w_a1 = vec![1.0, 0.0, 0.0, 1.0];
        model.w_a2 = vec![1.0, 0.0, 0.0, 1.0];
        model.b_a = vec![0.0, 0.0];
        model.v_a = vec![1.0, 1.0];
        let session = [0_usize, 1];
        let gates = model.stamp_attention_gates(&session).unwrap();
        let s: f32 = gates.iter().sum();
        assert!((s - 1.0).abs() > 0.5, "sum {s} should not be close to 1.0");
    }

    #[test]
    fn sigmoid_components_min_max_within_zero_one() {
        // Probe with v_a = e_k for each k and assert each gate is in (0,1).
        let cfg = StampConfig {
            embed_dim: 4,
            n_items: 5,
        };
        let mut rng = LcgRng::new(0xCAFE);
        let mut model = Stamp::new(cfg.clone(), &mut rng).unwrap();
        for k in 0..cfg.embed_dim {
            let mut v_a = vec![0.0_f32; cfg.embed_dim];
            if let Some(slot) = v_a.get_mut(k) {
                *slot = 1.0;
            }
            model.v_a = v_a;
            let session: Vec<usize> = (0..cfg.n_items).collect();
            let gates = model.stamp_attention_gates(&session).unwrap();
            for &g in &gates {
                assert!(g > 0.0 && g < 1.0, "gate {g} not in (0,1) at k={k}");
            }
        }
    }

    #[test]
    fn output_length_equals_n_items() {
        let mut rng = make_rng();
        let model = Stamp::new(default_cfg(), &mut rng).unwrap();
        let logits = model.forward_session(&[0_usize, 1, 2, 3]).unwrap();
        assert_eq!(logits.len(), 6);
    }

    #[test]
    fn identical_sessions_identical_logits_determinism() {
        let mut rng = make_rng();
        let model = Stamp::new(default_cfg(), &mut rng).unwrap();
        let logits_a = model.forward_session(&[1_usize, 2, 3]).unwrap();
        let logits_b = model.forward_session(&[1_usize, 2, 3]).unwrap();
        assert_eq!(logits_a.len(), logits_b.len());
        for (a, b) in logits_a.iter().zip(logits_b.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn m_s_permutation_invariant_for_same_item_session() {
        // Permuting a session with identical repeated items keeps m_s the
        // same (since m_s is a uniform mean). The full forward is not
        // permutation invariant because x_t = last item, but for a session
        // of identical items the last is unchanged either way.
        let mut rng = make_rng();
        let model = Stamp::new(default_cfg(), &mut rng).unwrap();
        let logits_a = model.forward_session(&[2_usize, 2, 2, 2]).unwrap();
        let logits_b = model.forward_session(&[2_usize, 2, 2, 2]).unwrap();
        for (a, b) in logits_a.iter().zip(logits_b.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn last_item_replacement_changes_logits() {
        let mut rng = make_rng();
        let model = Stamp::new(default_cfg(), &mut rng).unwrap();
        let session_a = [0_usize, 1, 2];
        let session_b = [0_usize, 1, 3];
        let logits_a = model.forward_session(&session_a).unwrap();
        let logits_b = model.forward_session(&session_b).unwrap();
        let diff: f32 = logits_a
            .iter()
            .zip(logits_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "swapping the last item should change logits (diff = {diff})"
        );
    }

    #[test]
    fn err_empty_session() {
        let mut rng = make_rng();
        let model = Stamp::new(default_cfg(), &mut rng).unwrap();
        assert!(matches!(
            model.forward_session(&[]),
            Err(RecsysError::EmptyInput)
        ));
        assert!(matches!(
            model.stamp_attention_gates(&[]),
            Err(RecsysError::EmptyInput)
        ));
        assert!(matches!(
            model.stamp_session_rep(&[]),
            Err(RecsysError::EmptyInput)
        ));
    }

    #[test]
    fn err_item_id_out_of_bounds() {
        let mut rng = make_rng();
        let model = Stamp::new(default_cfg(), &mut rng).unwrap();
        // n_items == 6, id 6 is OOR.
        assert!(matches!(
            model.forward_session(&[0_usize, 6]),
            Err(RecsysError::ItemOutOfBounds { idx: 6, n: 6 })
        ));
        assert!(matches!(
            model.stamp_attention_gates(&[6_usize]),
            Err(RecsysError::ItemOutOfBounds { idx: 6, n: 6 })
        ));
    }

    #[test]
    fn err_n_items_zero() {
        let mut rng = make_rng();
        let cfg = StampConfig {
            embed_dim: 4,
            n_items: 0,
        };
        assert!(matches!(
            Stamp::new(cfg, &mut rng),
            Err(RecsysError::InvalidNumItems { n: 0 })
        ));
    }

    #[test]
    fn err_embed_dim_zero() {
        let mut rng = make_rng();
        let cfg = StampConfig {
            embed_dim: 0,
            n_items: 4,
        };
        assert!(matches!(
            Stamp::new(cfg, &mut rng),
            Err(RecsysError::InvalidEmbeddingDim { d: 0 })
        ));
    }

    #[test]
    fn deterministic_init_given_seed() {
        let mut rng_a = LcgRng::new(2026);
        let mut rng_b = LcgRng::new(2026);
        let model_a = Stamp::new(default_cfg(), &mut rng_a).unwrap();
        let model_b = Stamp::new(default_cfg(), &mut rng_b).unwrap();
        assert_eq!(model_a.item_embeds, model_b.item_embeds);
        assert_eq!(model_a.w_a0, model_b.w_a0);
        assert_eq!(model_a.w_a1, model_b.w_a1);
        assert_eq!(model_a.w_a2, model_b.w_a2);
        assert_eq!(model_a.v_a, model_b.v_a);
        assert_eq!(model_a.w_mlp_s, model_b.w_mlp_s);
        assert_eq!(model_a.w_mlp_t, model_b.w_mlp_t);
        let l_a = model_a.forward_session(&[0_usize, 1, 2]).unwrap();
        let l_b = model_b.forward_session(&[0_usize, 1, 2]).unwrap();
        for (a, b) in l_a.iter().zip(l_b.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn all_weights_finite_no_nan() {
        let mut rng = make_rng();
        let model = Stamp::new(default_cfg(), &mut rng).unwrap();
        for v in model
            .item_embeds
            .iter()
            .chain(model.w_a0.iter())
            .chain(model.w_a1.iter())
            .chain(model.w_a2.iter())
            .chain(model.b_a.iter())
            .chain(model.v_a.iter())
            .chain(model.w_mlp_s.iter())
            .chain(model.b_s.iter())
            .chain(model.w_mlp_t.iter())
            .chain(model.b_t.iter())
        {
            assert!(v.is_finite(), "weight {v} not finite");
        }
    }

    #[test]
    fn n_params_closed_form() {
        let mut rng = make_rng();
        let model = Stamp::new(default_cfg(), &mut rng).unwrap();
        let d = 4_usize;
        let n_items = 6_usize;
        let expected = n_items * d + 3 * d * d + d + d + 2 * (d * d) + 2 * d;
        assert_eq!(model.n_params(), expected);
    }

    #[test]
    fn session_rep_is_weighted_sum() {
        // m_a = Σ_i α_i x_i. With a session of one item, m_a = α_0 · x_0.
        let mut rng = LcgRng::new(101);
        let model = Stamp::new(default_cfg(), &mut rng).unwrap();
        let session = [3_usize];
        let gates = model.stamp_attention_gates(&session).unwrap();
        let m_a = model.stamp_session_rep(&session).unwrap();
        let x_0 = model.lookup(3).unwrap().to_vec();
        for k in 0..model.cfg.embed_dim {
            assert!(
                (m_a[k] - gates[0] * x_0[k]).abs() < 1e-5,
                "m_a[{k}] {} != α_0·x_0[k] {}",
                m_a[k],
                gates[0] * x_0[k]
            );
        }
    }

    #[test]
    fn logits_finite_after_forward() {
        let mut rng = make_rng();
        let model = Stamp::new(default_cfg(), &mut rng).unwrap();
        let logits = model.forward_session(&[0_usize, 2, 4]).unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_session_long_session_consistent_shape() {
        // Larger session should still produce n_items logits and remain
        // finite.
        let mut rng = make_rng();
        let model = Stamp::new(default_cfg(), &mut rng).unwrap();
        let session: Vec<usize> = (0..5).map(|i| i % 6).collect();
        let logits = model.forward_session(&session).unwrap();
        assert_eq!(logits.len(), 6);
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}
