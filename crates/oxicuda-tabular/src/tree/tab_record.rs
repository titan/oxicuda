//! TabRecord — a TabR-style retrieval forward for tabular data.
//!
//! This is the *forward inference* path of a retrieval-augmented tabular model in
//! the spirit of TabR (Gorishniy et al. 2023, "TabR: Tabular Deep Learning Meets
//! Nearest Neighbors").  Given a query feature vector and an in-memory **context**
//! of `(features, value)` candidates, the layer:
//!
//! 1. **Encodes** the query and every candidate with a shared linear+ReLU encoder
//!    `e = ReLU(W_e · x + b_e) ∈ ℝ^{embed_dim}`.
//! 2. **Keys** them with a linear map `k = W_k · e ∈ ℝ^{key_dim}`.
//! 3. **Scores** each candidate by the natural nearest-neighbour metric — the
//!    (scaled) negative squared Euclidean distance between the query key `q` and
//!    the candidate key `k_i`:  `S_i = −sim_scale · ‖q − k_i‖²`.
//! 4. **Attends** with an entmax-α simplex projection (reused from
//!    [`crate::tree::node_oblivious`]):  `α = entmax_α(S)`.  `α = 2` ⇒ sparsemax ⇒
//!    a genuinely *sparse* neighbour selection (hard top-k), `α = 1.5` ⇒ a soft but
//!    still sparse retrieval.  The weights are non-negative and sum to 1.
//! 5. **Aggregates** the candidate **values** by the attention weights:
//!    `out = Σ_i α_i · v_i`,  where `v_i = W_v · e_i + y_i` combines an
//!    encoder-derived term with the stored context value `y_i ∈ ℝ^{value_dim}`.
//!
//! Because `α` lies on the probability simplex, the output is a **convex
//! combination** of the candidate value vectors `{v_i}` and therefore lies inside
//! their convex hull — a property the unit tests check directly.  This module is
//! forward-only (fixed-seed weights, no training), so every test is analytic.
//!
//! ## References
//! - Gorishniy, Y., Rubachev, I., Kartashev, N., Shlenskii, D., Kotelnikov, A. &
//!   Babenko, A. (2023). "TabR: Tabular Deep Learning Meets Nearest Neighbors."
//! - Martins, A. F. T. & Astudillo, R. F. (2016). "From Softmax to Sparsemax."

use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;
use crate::tree::node_oblivious::{entmax_alpha_f64, fill_normal_f64};

// ─── Context ─────────────────────────────────────────────────────────────────────

/// An in-memory retrieval context: `n_context` candidates, each a
/// `(features ∈ ℝ^{num_features}, value ∈ ℝ^{value_dim})` pair.
#[derive(Debug, Clone)]
pub struct TabRecordContext {
    /// Candidate features, flattened `[n_context * num_features]`.
    features: Vec<f64>,
    /// Candidate values / label encodings, flattened `[n_context * value_dim]`.
    values: Vec<f64>,
    n_context: usize,
    num_features: usize,
    value_dim: usize,
}

impl TabRecordContext {
    /// Build a context from flat `features` (`[n_context * num_features]`) and flat
    /// `values` (`[n_context * value_dim]`).
    ///
    /// # Errors
    /// - [`TabularError::EmptyInput`] if `n_context == 0`.
    /// - [`TabularError::InvalidFeatureCount`] / [`TabularError::InvalidParameter`]
    ///   if `num_features == 0` or `value_dim == 0`.
    /// - [`TabularError::DimensionMismatch`] if the buffers do not match the implied
    ///   shapes.
    pub fn new(
        features: Vec<f64>,
        values: Vec<f64>,
        n_context: usize,
        num_features: usize,
        value_dim: usize,
    ) -> TabularResult<Self> {
        if n_context == 0 {
            return Err(TabularError::EmptyInput);
        }
        if num_features == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if value_dim == 0 {
            return Err(TabularError::InvalidParameter {
                name: "value_dim".into(),
                msg: "must be ≥ 1".into(),
            });
        }
        let want_feat = n_context * num_features;
        if features.len() != want_feat {
            return Err(TabularError::DimensionMismatch {
                expected: want_feat,
                got: features.len(),
            });
        }
        let want_val = n_context * value_dim;
        if values.len() != want_val {
            return Err(TabularError::DimensionMismatch {
                expected: want_val,
                got: values.len(),
            });
        }
        Ok(Self {
            features,
            values,
            n_context,
            num_features,
            value_dim,
        })
    }

    /// Number of candidates in the context.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n_context
    }

    /// Whether the context is empty (always `false`; constructed contexts have ≥ 1
    /// candidate, but the predicate satisfies clippy's `len`/`is_empty` pairing).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n_context == 0
    }

    /// The `i`-th candidate's feature row.
    fn features_row(&self, i: usize) -> &[f64] {
        &self.features[i * self.num_features..(i + 1) * self.num_features]
    }

    /// The `i`-th candidate's stored value row.
    fn value_row(&self, i: usize) -> &[f64] {
        &self.values[i * self.value_dim..(i + 1) * self.value_dim]
    }
}

// ─── Configuration ───────────────────────────────────────────────────────────────

/// Configuration for a [`TabRecordLayer`].
#[derive(Debug, Clone)]
pub struct TabRecordConfig {
    /// Number of input features (≥ 1).
    pub num_features: usize,
    /// Encoder embedding dimension (≥ 1).
    pub embed_dim: usize,
    /// Key/query dimension used for the similarity (≥ 1).
    pub key_dim: usize,
    /// Value / output dimension (≥ 1).
    pub value_dim: usize,
    /// Entmax temperature `α ∈ (1, 2]` for the attention simplex.  `2.0` ⇒
    /// sparsemax (hard sparse retrieval).
    pub entmax_alpha: f64,
    /// Positive sharpness on the negative-distance similarity.
    pub sim_scale: f64,
    /// RNG seed for parameter initialisation.
    pub seed: u64,
}

impl TabRecordConfig {
    /// A sensible default: `embed_dim = 16`, `key_dim = 8`, entmax-1.5.
    #[must_use]
    pub fn new(num_features: usize, value_dim: usize) -> Self {
        Self {
            num_features,
            embed_dim: 16,
            key_dim: 8,
            value_dim,
            entmax_alpha: 1.5,
            sim_scale: 1.0,
            seed: 0,
        }
    }

    fn validate(&self) -> TabularResult<()> {
        if self.num_features == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if self.embed_dim == 0 {
            return Err(TabularError::InvalidEmbedDim { dim: 0 });
        }
        if self.key_dim == 0 {
            return Err(TabularError::InvalidAttentionDim { dim: 0 });
        }
        if self.value_dim == 0 {
            return Err(TabularError::InvalidParameter {
                name: "value_dim".into(),
                msg: "must be ≥ 1".into(),
            });
        }
        if !(self.entmax_alpha > 1.0 && self.entmax_alpha <= 2.0) {
            return Err(TabularError::InvalidParameter {
                name: "entmax_alpha".into(),
                msg: format!("must lie in (1, 2], got {}", self.entmax_alpha),
            });
        }
        if !self.sim_scale.is_finite() || self.sim_scale <= 0.0 {
            return Err(TabularError::InvalidParameter {
                name: "sim_scale".into(),
                msg: format!("must be a positive finite value, got {}", self.sim_scale),
            });
        }
        Ok(())
    }
}

// ─── Retrieval layer ─────────────────────────────────────────────────────────────

/// A TabR-style retrieval layer with fixed (seeded) weights.
#[derive(Debug, Clone)]
pub struct TabRecordLayer {
    /// Encoder weights `[embed_dim * num_features]`.
    w_enc: Vec<f64>,
    /// Encoder bias `[embed_dim]`.
    b_enc: Vec<f64>,
    /// Key map `[key_dim * embed_dim]`.
    w_key: Vec<f64>,
    /// Value map `[value_dim * embed_dim]`.
    w_val: Vec<f64>,
    num_features: usize,
    embed_dim: usize,
    key_dim: usize,
    value_dim: usize,
    entmax_alpha: f64,
    sim_scale: f64,
}

impl TabRecordLayer {
    /// Build a randomly-initialised layer from `config`, seeding from `config.seed`.
    ///
    /// Weights are drawn `N(0, 1/√fan_in)` (fan-in scaling), biases zeroed — reusing
    /// the sibling module's `fill_normal_f64` so the RNG stream is shared.
    ///
    /// # Errors
    /// [`TabularError::InvalidFeatureCount`] / [`TabularError::InvalidEmbedDim`] /
    /// [`TabularError::InvalidAttentionDim`] / [`TabularError::InvalidParameter`] for
    /// an invalid configuration.
    pub fn new(config: TabRecordConfig) -> TabularResult<Self> {
        config.validate()?;
        let mut rng = LcgRng::new(config.seed);
        Self::new_with_rng(config, &mut rng)
    }

    /// Build using a caller-supplied RNG so the stream can be threaded.
    ///
    /// # Errors
    /// As [`TabRecordLayer::new`].
    pub fn new_with_rng(config: TabRecordConfig, rng: &mut LcgRng) -> TabularResult<Self> {
        config.validate()?;
        let mut w_enc = vec![0.0_f64; config.embed_dim * config.num_features];
        fill_normal_f64(rng, &mut w_enc, 1.0 / (config.num_features as f64).sqrt());
        let b_enc = vec![0.0_f64; config.embed_dim];
        let mut w_key = vec![0.0_f64; config.key_dim * config.embed_dim];
        fill_normal_f64(rng, &mut w_key, 1.0 / (config.embed_dim as f64).sqrt());
        let mut w_val = vec![0.0_f64; config.value_dim * config.embed_dim];
        fill_normal_f64(rng, &mut w_val, 1.0 / (config.embed_dim as f64).sqrt());

        Ok(Self {
            w_enc,
            b_enc,
            w_key,
            w_val,
            num_features: config.num_features,
            embed_dim: config.embed_dim,
            key_dim: config.key_dim,
            value_dim: config.value_dim,
            entmax_alpha: config.entmax_alpha,
            sim_scale: config.sim_scale,
        })
    }

    /// Number of input features expected for a query / context row.
    #[must_use]
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Output / value dimension.
    #[must_use]
    pub fn value_dim(&self) -> usize {
        self.value_dim
    }

    /// Encode a feature vector: `ReLU(W_e · x + b_e) ∈ ℝ^{embed_dim}`.
    fn encode(&self, x: &[f64]) -> Vec<f64> {
        let mut e = self.b_enc.clone();
        for (j, ej) in e.iter_mut().enumerate() {
            let base = j * self.num_features;
            let mut acc = *ej;
            for (f, &xf) in x.iter().enumerate() {
                acc += self.w_enc[base + f] * xf;
            }
            *ej = acc.max(0.0); // ReLU
        }
        e
    }

    /// Linear key map `W_k · e ∈ ℝ^{key_dim}`.
    fn key(&self, e: &[f64]) -> Vec<f64> {
        let mut k = vec![0.0_f64; self.key_dim];
        for (j, kj) in k.iter_mut().enumerate() {
            let base = j * self.embed_dim;
            let mut acc = 0.0_f64;
            for (d, &ed) in e.iter().enumerate() {
                acc += self.w_key[base + d] * ed;
            }
            *kj = acc;
        }
        k
    }

    /// Value of a candidate: `W_v · e + y ∈ ℝ^{value_dim}`.
    fn value(&self, e: &[f64], stored: &[f64]) -> Vec<f64> {
        let mut v = vec![0.0_f64; self.value_dim];
        for (j, vj) in v.iter_mut().enumerate() {
            let base = j * self.embed_dim;
            let mut acc = stored[j];
            for (d, &ed) in e.iter().enumerate() {
                acc += self.w_val[base + d] * ed;
            }
            *vj = acc;
        }
        v
    }

    /// The per-candidate value vectors `v_i = W_v e_i + y_i` for the whole context.
    ///
    /// # Errors
    /// [`TabularError::DimensionMismatch`] if the context's feature/value dims do not
    /// match the layer.
    pub fn context_values(&self, context: &TabRecordContext) -> TabularResult<Vec<Vec<f64>>> {
        self.check_context(context)?;
        let mut out = Vec::with_capacity(context.len());
        for i in 0..context.len() {
            let e = self.encode(context.features_row(i));
            out.push(self.value(&e, context.value_row(i)));
        }
        Ok(out)
    }

    /// The attention weights `α = entmax_α(S)` over the context for `query`.
    /// Non-negative and summing to ≈ 1.
    ///
    /// # Errors
    /// [`TabularError::DimensionMismatch`] for a mis-sized query or context;
    /// propagates entmax solver errors.
    pub fn attention(&self, query: &[f64], context: &TabRecordContext) -> TabularResult<Vec<f64>> {
        if query.len() != self.num_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.num_features,
                got: query.len(),
            });
        }
        self.check_context(context)?;
        let q = self.key(&self.encode(query));
        let mut scores = Vec::with_capacity(context.len());
        for i in 0..context.len() {
            let ki = self.key(&self.encode(context.features_row(i)));
            // Negative squared Euclidean distance (the nearest-neighbour metric).
            let mut dist2 = 0.0_f64;
            for (a, b) in q.iter().zip(ki.iter()) {
                let d = a - b;
                dist2 += d * d;
            }
            scores.push(-self.sim_scale * dist2);
        }
        entmax_alpha_f64(&scores, self.entmax_alpha)
    }

    /// Retrieval forward: `out = Σ_i α_i · v_i ∈ ℝ^{value_dim}` — a convex
    /// combination of the candidate values.
    ///
    /// # Errors
    /// As [`Self::attention`].
    pub fn forward(&self, query: &[f64], context: &TabRecordContext) -> TabularResult<Vec<f64>> {
        let alpha = self.attention(query, context)?;
        let values = self.context_values(context)?;
        let mut out = vec![0.0_f64; self.value_dim];
        for (&a, v) in alpha.iter().zip(values.iter()) {
            for (o, &vd) in out.iter_mut().zip(v.iter()) {
                *o += a * vd;
            }
        }
        Ok(out)
    }

    /// Batched retrieval over many queries sharing one context.  `queries` is flat
    /// `[batch_size * num_features]`; the result is flat `[batch_size * value_dim]`.
    ///
    /// # Errors
    /// [`TabularError::EmptyInput`] if `batch_size == 0`;
    /// [`TabularError::DimensionMismatch`] for shape errors; propagates solver errors.
    pub fn forward_batch(
        &self,
        queries: &[f64],
        batch_size: usize,
        context: &TabRecordContext,
    ) -> TabularResult<Vec<f64>> {
        if batch_size == 0 {
            return Err(TabularError::EmptyInput);
        }
        let in_d = self.num_features;
        if queries.len() != batch_size * in_d {
            return Err(TabularError::DimensionMismatch {
                expected: batch_size * in_d,
                got: queries.len(),
            });
        }
        // Precompute the candidate values once (query-independent).
        let values = self.context_values(context)?;
        let mut out = Vec::with_capacity(batch_size * self.value_dim);
        for b in 0..batch_size {
            let row = &queries[b * in_d..(b + 1) * in_d];
            let alpha = self.attention(row, context)?;
            let mut pred = vec![0.0_f64; self.value_dim];
            for (&a, v) in alpha.iter().zip(values.iter()) {
                for (o, &vd) in pred.iter_mut().zip(v.iter()) {
                    *o += a * vd;
                }
            }
            out.extend_from_slice(&pred);
        }
        Ok(out)
    }

    fn check_context(&self, context: &TabRecordContext) -> TabularResult<()> {
        if context.num_features != self.num_features {
            return Err(TabularError::DimensionMismatch {
                expected: self.num_features,
                got: context.num_features,
            });
        }
        if context.value_dim != self.value_dim {
            return Err(TabularError::DimensionMismatch {
                expected: self.value_dim,
                got: context.value_dim,
            });
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::tree::node_oblivious::next_normal_f64;

    fn sum(v: &[f64]) -> f64 {
        v.iter().sum()
    }

    fn random_vec(seed: u64, n: usize) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        (0..n).map(|_| next_normal_f64(&mut rng, 1.0)).collect()
    }

    fn demo_context(
        num_features: usize,
        value_dim: usize,
        n: usize,
        seed: u64,
    ) -> TabRecordContext {
        let feats = random_vec(seed, n * num_features);
        let vals = random_vec(seed.wrapping_add(7), n * value_dim);
        TabRecordContext::new(feats, vals, n, num_features, value_dim).expect("context")
    }

    fn demo_layer(
        num_features: usize,
        value_dim: usize,
        alpha: f64,
        sim_scale: f64,
    ) -> TabRecordLayer {
        let cfg = TabRecordConfig {
            num_features,
            embed_dim: 12,
            key_dim: 6,
            value_dim,
            entmax_alpha: alpha,
            sim_scale,
            seed: 17,
        };
        TabRecordLayer::new(cfg).expect("layer")
    }

    // Attention weights form a valid simplex (non-negative, sum ≈ 1).
    #[test]
    fn attention_is_a_simplex() {
        let layer = demo_layer(5, 3, 1.5, 1.0);
        let ctx = demo_context(5, 3, 9, 100);
        for s in 0..5 {
            let q = random_vec(2000 + s, 5);
            let a = layer.attention(&q, &ctx).expect("attention");
            assert_eq!(a.len(), ctx.len());
            assert!((sum(&a) - 1.0).abs() < 1e-6, "attention sum={}", sum(&a));
            assert!(a.iter().all(|&w| w >= -1e-12), "attention non-negative");
        }
    }

    // forward == Σ α_i v_i, and the output is a convex combination of the candidate
    // values (each coordinate lies within the candidate min/max — the convex hull).
    #[test]
    fn output_is_convex_combination_of_values() {
        let layer = demo_layer(4, 3, 1.5, 1.0);
        let ctx = demo_context(4, 3, 8, 55);
        let q = random_vec(321, 4);
        let alpha = layer.attention(&q, &ctx).expect("attention");
        let values = layer.context_values(&ctx).expect("values");
        let out = layer.forward(&q, &ctx).expect("forward");
        assert_eq!(out.len(), layer.value_dim());

        // (a) Consistency: forward equals the explicit weighted sum.
        let mut manual = vec![0.0_f64; layer.value_dim()];
        for (&a, v) in alpha.iter().zip(values.iter()) {
            for (m, &vd) in manual.iter_mut().zip(v.iter()) {
                *m += a * vd;
            }
        }
        for (o, m) in out.iter().zip(manual.iter()) {
            assert!((o - m).abs() < 1e-12, "forward must equal Σ α_i v_i");
        }

        // (b) Convex-hull bound: out[d] ∈ [min_i v_i[d], max_i v_i[d]].
        for d in 0..layer.value_dim() {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for v in &values {
                lo = lo.min(v[d]);
                hi = hi.max(v[d]);
            }
            assert!(
                out[d] >= lo - 1e-9 && out[d] <= hi + 1e-9,
                "out[{d}]={} outside hull [{lo}, {hi}]",
                out[d]
            );
            assert!(out[d].is_finite(), "output must be finite");
        }
    }

    // Output dimension equals value_dim and is finite.
    #[test]
    fn forward_output_dimension() {
        let layer = demo_layer(6, 4, 1.5, 1.0);
        let ctx = demo_context(6, 4, 5, 11);
        let q = random_vec(7, 6);
        let out = layer.forward(&q, &ctx).expect("forward");
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // Determinism: same seed + same input ⇒ identical output.
    #[test]
    fn determinism_with_same_seed() {
        let a = demo_layer(5, 3, 1.5, 1.0);
        let b = demo_layer(5, 3, 1.5, 1.0);
        let ctx = demo_context(5, 3, 7, 88);
        let q = random_vec(1234, 5);
        let oa = a.forward(&q, &ctx).expect("forward a");
        let ob = b.forward(&q, &ctx).expect("forward b");
        assert_eq!(oa, ob, "same seed must give identical outputs");
    }

    // Nearest-neighbour: a query equal to candidate 0's features puts the largest
    // attention on candidate 0; with sparsemax + a sharp scale it becomes one-hot,
    // so the output equals that candidate's value vector.
    #[test]
    fn sparsemax_retrieves_exact_match() {
        let num_features = 4;
        let value_dim = 3;
        let n = 6;
        // sparsemax (α = 2) + large sim_scale ⇒ hard nearest-neighbour pick.
        let layer = demo_layer(num_features, value_dim, 2.0, 50.0);
        let ctx = demo_context(num_features, value_dim, n, 4242);
        // Query exactly equals candidate 0's features.
        let q = ctx.features_row(0).to_vec();

        let alpha = layer.attention(&q, &ctx).expect("attention");
        // Candidate 0 must carry the maximum weight.
        let a0 = alpha[0];
        assert!(
            alpha.iter().all(|&w| w <= a0 + 1e-12),
            "exact match must have the largest attention weight"
        );
        // With sparsemax + sharp scale the weight collapses onto candidate 0.
        assert!(
            (a0 - 1.0).abs() < 1e-9,
            "expected one-hot retrieval, got {a0}"
        );

        let out = layer.forward(&q, &ctx).expect("forward");
        let values = layer.context_values(&ctx).expect("values");
        for (o, &v0) in out.iter().zip(values[0].iter()) {
            assert!(
                (o - v0).abs() < 1e-9,
                "output must equal the retrieved value"
            );
        }
    }

    // Batched forward matches per-query forward.
    #[test]
    fn batched_matches_per_query() {
        let layer = demo_layer(5, 3, 1.5, 1.0);
        let ctx = demo_context(5, 3, 8, 909);
        let batch = 4;
        let q = random_vec(31337, batch * 5);
        let batched = layer.forward_batch(&q, batch, &ctx).expect("batch");
        assert_eq!(batched.len(), batch * 3);
        for b in 0..batch {
            let row = &q[b * 5..(b + 1) * 5];
            let single = layer.forward(row, &ctx).expect("single");
            for d in 0..3 {
                assert!(
                    (batched[b * 3 + d] - single[d]).abs() < 1e-12,
                    "batched must match per-query"
                );
            }
        }
    }

    // Configuration and shape validation.
    #[test]
    fn validation_errors() {
        let mut c = TabRecordConfig::new(5, 3);
        c.num_features = 0;
        assert!(matches!(
            TabRecordLayer::new(c),
            Err(TabularError::InvalidFeatureCount { .. })
        ));

        let mut c = TabRecordConfig::new(5, 3);
        c.embed_dim = 0;
        assert!(matches!(
            TabRecordLayer::new(c),
            Err(TabularError::InvalidEmbedDim { .. })
        ));

        let mut c = TabRecordConfig::new(5, 3);
        c.key_dim = 0;
        assert!(matches!(
            TabRecordLayer::new(c),
            Err(TabularError::InvalidAttentionDim { .. })
        ));

        let mut c = TabRecordConfig::new(5, 3);
        c.entmax_alpha = 3.0;
        assert!(matches!(
            TabRecordLayer::new(c),
            Err(TabularError::InvalidParameter { .. })
        ));

        let mut c = TabRecordConfig::new(5, 3);
        c.sim_scale = -1.0;
        assert!(matches!(
            TabRecordLayer::new(c),
            Err(TabularError::InvalidParameter { .. })
        ));

        // Context shape errors.
        assert!(matches!(
            TabRecordContext::new(vec![0.0; 3], vec![0.0; 4], 0, 3, 2),
            Err(TabularError::EmptyInput)
        ));
        assert!(matches!(
            TabRecordContext::new(vec![0.0; 5], vec![0.0; 4], 2, 3, 2),
            Err(TabularError::DimensionMismatch { .. })
        ));

        // Query/context vs layer mismatch.
        let layer = demo_layer(5, 3, 1.5, 1.0);
        let ctx = demo_context(5, 3, 4, 1);
        assert!(matches!(
            layer.attention(&[0.0; 2], &ctx),
            Err(TabularError::DimensionMismatch { .. })
        ));
        let wrong_ctx = demo_context(4, 3, 4, 1);
        let q = random_vec(2, 5);
        assert!(matches!(
            layer.attention(&q, &wrong_ctx),
            Err(TabularError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            layer.forward_batch(&q, 0, &ctx),
            Err(TabularError::EmptyInput)
        ));
    }
}
