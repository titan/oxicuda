//! Mixed-rank LoRA: per-layer-independent rank selection.
//!
//! Standard LoRA ([`crate::lora::LoraModel`]) fixes a single rank `r` for
//! every adapted layer. In practice the intrinsic dimensionality of the
//! update varies across a network — attention `query`/`value` projections
//! often deserve a larger rank than feed-forward layers — so allocating the
//! same `r` everywhere either wastes parameters on low-rank layers or
//! under-fits high-rank ones.
//!
//! [`MixedRankLoraModel`] stores, for each named layer, a [`LoraLinear`]
//! adapter whose rank (and scaling `α/r`) is chosen **per layer**. A
//! [`RankBudget`] helper distributes a global parameter budget across layers
//! either uniformly or proportionally to layer width, so a caller can ask for
//! "the best per-layer ranks fitting `P` LoRA parameters" instead of hand
//! tuning each one.

use std::collections::HashMap;

use crate::error::{GenError, GenResult};
use crate::handle::LcgRng;
use crate::lora::adapter::{LoraConfig, LoraLinear};

// ─── Per-layer specification ────────────────────────────────────────────────

/// Shape + rank specification for one adapted layer.
#[derive(Debug, Clone)]
pub struct LayerSpec {
    /// Layer name (e.g. `"blocks.0.attn.q_proj"`).
    pub name: String,
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// LoRA rank for **this** layer.
    pub rank: usize,
    /// LoRA scaling numerator `α` for this layer (scaling = `α/rank`).
    pub alpha: f32,
}

impl LayerSpec {
    /// Create a layer spec.
    ///
    /// # Errors
    /// * [`GenError::EmptyInput`] on a zero feature dimension.
    /// * [`GenError::InvalidLoraRank`] if `rank == 0`.
    /// * [`GenError::InvalidLoraAlpha`] if `alpha <= 0`.
    pub fn new(
        name: impl Into<String>,
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
    ) -> GenResult<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(GenError::EmptyInput("feature dims must be > 0"));
        }
        if rank == 0 {
            return Err(GenError::InvalidLoraRank(rank));
        }
        if alpha <= 0.0 {
            return Err(GenError::InvalidLoraAlpha(alpha));
        }
        Ok(Self {
            name: name.into(),
            in_features,
            out_features,
            rank,
            alpha,
        })
    }

    /// Trainable LoRA parameter count for this layer: `r·(in + out)`.
    pub fn param_count(&self) -> usize {
        self.rank * (self.in_features + self.out_features)
    }
}

// ─── MixedRankLoraModel ─────────────────────────────────────────────────────

/// A collection of named LoRA adapters, each with its own rank.
#[derive(Debug, Clone, Default)]
pub struct MixedRankLoraModel {
    adapters: HashMap<String, LoraLinear>,
    /// Insertion order, for deterministic iteration / reporting.
    order: Vec<String>,
}

impl MixedRankLoraModel {
    /// Create an empty mixed-rank model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a model from a set of per-layer specs, initialising each adapter
    /// with the shared `rng`.
    ///
    /// Adapters are created in slice order so the RNG draw — and therefore the
    /// resulting weights — is deterministic for a fixed seed.
    ///
    /// # Errors
    /// Propagates [`LoraLinear::new`] errors; returns
    /// [`GenError::Internal`] on a duplicate layer name.
    pub fn from_specs(specs: &[LayerSpec], rng: &mut LcgRng) -> GenResult<Self> {
        let mut model = Self::new();
        for spec in specs {
            if model.adapters.contains_key(&spec.name) {
                return Err(GenError::Internal(format!(
                    "duplicate layer name '{}'",
                    spec.name
                )));
            }
            let cfg = LoraConfig::new(spec.rank, spec.alpha)?;
            let adapter = LoraLinear::new(spec.in_features, spec.out_features, &cfg, rng)?;
            model.insert(spec.name.clone(), adapter);
        }
        Ok(model)
    }

    /// Insert (or replace) a named adapter.
    pub fn insert(&mut self, name: impl Into<String>, adapter: LoraLinear) {
        let name = name.into();
        if !self.adapters.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.adapters.insert(name, adapter);
    }

    /// Look up an adapter by layer name.
    pub fn get(&self, name: &str) -> Option<&LoraLinear> {
        self.adapters.get(name)
    }

    /// Apply the LoRA correction for a single named layer.
    ///
    /// # Errors
    /// * [`GenError::Internal`] if `name` is unknown.
    /// * Propagates [`LoraLinear::forward`] shape errors.
    pub fn forward_layer(
        &self,
        name: &str,
        x: &[f32],
        base_output: &[f32],
        batch: usize,
    ) -> GenResult<Vec<f32>> {
        let adapter = self
            .adapters
            .get(name)
            .ok_or_else(|| GenError::Internal(format!("unknown layer '{name}'")))?;
        adapter.forward(x, base_output, batch)
    }

    /// Number of adapted layers.
    pub fn len(&self) -> usize {
        self.adapters.len()
    }

    /// Whether the model has no adapters.
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    /// Layer names in insertion order.
    pub fn layer_names(&self) -> &[String] {
        &self.order
    }

    /// The rank used for a given layer, if present.
    pub fn rank_of(&self, name: &str) -> Option<usize> {
        self.adapters.get(name).map(LoraLinear::rank)
    }

    /// Set of distinct ranks present across all layers (sorted ascending).
    pub fn distinct_ranks(&self) -> Vec<usize> {
        let mut ranks: Vec<usize> = self.adapters.values().map(LoraLinear::rank).collect();
        ranks.sort_unstable();
        ranks.dedup();
        ranks
    }

    /// Total trainable LoRA parameters across all layers: `Σ r·(in + out)`.
    pub fn total_params(&self) -> usize {
        self.adapters
            .values()
            .map(|a| a.rank() * (a.in_features() + a.out_features()))
            .sum()
    }
}

// ─── RankBudget ─────────────────────────────────────────────────────────────

/// How a parameter budget is split across layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetStrategy {
    /// Every layer receives the same rank.
    Uniform,
    /// Each layer's rank is proportional to its width `in + out`, so wider
    /// layers (which can absorb more rank) get more.
    WidthProportional,
}

/// Allocates per-layer ranks under a global parameter budget.
#[derive(Debug, Clone)]
pub struct RankBudget {
    max_params: usize,
    strategy: BudgetStrategy,
    min_rank: usize,
    max_rank: usize,
}

impl RankBudget {
    /// Create a budget allocator.
    ///
    /// * `max_params` — total LoRA parameter ceiling `Σ r·(in+out)`.
    /// * `strategy`   — distribution rule.
    /// * `min_rank`   — floor for every layer (`>= 1`).
    /// * `max_rank`   — ceiling for every layer (`>= min_rank`).
    ///
    /// # Errors
    /// [`GenError::InvalidLoraRank`] if `min_rank == 0` or `max_rank <
    /// min_rank`.
    pub fn new(
        max_params: usize,
        strategy: BudgetStrategy,
        min_rank: usize,
        max_rank: usize,
    ) -> GenResult<Self> {
        if min_rank == 0 {
            return Err(GenError::InvalidLoraRank(min_rank));
        }
        if max_rank < min_rank {
            return Err(GenError::InvalidLoraRank(max_rank));
        }
        Ok(Self {
            max_params,
            strategy,
            min_rank,
            max_rank,
        })
    }

    /// Compute per-layer ranks for `(name, in, out)` triples, returning a
    /// vector of [`LayerSpec`] with `alpha = rank` (i.e. scaling `1.0`) by
    /// default; callers may overwrite `alpha` afterwards.
    ///
    /// The allocation greedily assigns each layer its `min_rank`, then raises
    /// ranks (respecting `max_rank` and the chosen strategy's weights) one
    /// unit at a time, charging `(in+out)` parameters per increment, until the
    /// budget is exhausted.
    ///
    /// # Errors
    /// * [`GenError::EmptyInput`] if `layers` is empty or any dimension is `0`.
    pub fn allocate(&self, layers: &[(String, usize, usize)]) -> GenResult<Vec<LayerSpec>> {
        if layers.is_empty() {
            return Err(GenError::EmptyInput("no layers to allocate"));
        }
        let mut ranks = vec![self.min_rank; layers.len()];
        let widths: Vec<usize> = layers
            .iter()
            .map(|(_, i, o)| i.checked_add(*o).unwrap_or(usize::MAX))
            .collect();
        for &w in &widths {
            if w == 0 {
                return Err(GenError::EmptyInput("layer width must be > 0"));
            }
        }

        // Parameters already committed by the min-rank floor.
        let mut used: usize = ranks
            .iter()
            .zip(&widths)
            .map(|(&r, &w)| r.saturating_mul(w))
            .sum();

        // Priority weight per layer for the "which layer to bump next" choice.
        let priority = |idx: usize| -> f64 {
            match self.strategy {
                BudgetStrategy::Uniform => 1.0,
                BudgetStrategy::WidthProportional => widths[idx] as f64,
            }
        };

        // Greedy bump loop: repeatedly grow the highest-priority not-maxed
        // layer that still fits in the remaining budget.
        loop {
            // Pick the candidate with the largest (priority / current_rank)
            // ratio — uniform strategy then balances ranks, proportional
            // strategy favours wide layers.
            let mut best: Option<usize> = None;
            let mut best_score = f64::NEG_INFINITY;
            for idx in 0..layers.len() {
                if ranks[idx] >= self.max_rank {
                    continue;
                }
                if used.saturating_add(widths[idx]) > self.max_params {
                    continue;
                }
                let score = priority(idx) / ranks[idx] as f64;
                if score > best_score {
                    best_score = score;
                    best = Some(idx);
                }
            }
            match best {
                Some(idx) => {
                    ranks[idx] += 1;
                    used += widths[idx];
                }
                None => break,
            }
        }

        let mut specs = Vec::with_capacity(layers.len());
        for (idx, (name, in_f, out_f)) in layers.iter().enumerate() {
            specs.push(LayerSpec::new(
                name.clone(),
                *in_f,
                *out_f,
                ranks[idx],
                ranks[idx] as f32,
            )?);
        }
        Ok(specs)
    }

    /// Configured parameter ceiling.
    pub fn max_params(&self) -> usize {
        self.max_params
    }

    /// Configured allocation strategy.
    pub fn strategy(&self) -> BudgetStrategy {
        self.strategy
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> LcgRng {
        LcgRng::new(42)
    }

    #[test]
    fn layer_spec_validation() {
        assert!(LayerSpec::new("a", 0, 4, 2, 1.0).is_err());
        assert!(LayerSpec::new("a", 4, 4, 0, 1.0).is_err());
        assert!(LayerSpec::new("a", 4, 4, 2, 0.0).is_err());
        let s = LayerSpec::new("a", 8, 16, 4, 4.0).expect("valid");
        assert_eq!(s.param_count(), 4 * (8 + 16));
    }

    #[test]
    fn build_model_with_distinct_ranks() {
        let specs = vec![
            LayerSpec::new("attn.q", 64, 64, 16, 16.0).expect("spec"),
            LayerSpec::new("attn.k", 64, 64, 8, 8.0).expect("spec"),
            LayerSpec::new("ffn.up", 64, 256, 4, 4.0).expect("spec"),
        ];
        let model = MixedRankLoraModel::from_specs(&specs, &mut rng()).expect("model");
        assert_eq!(model.len(), 3);
        assert_eq!(model.rank_of("attn.q"), Some(16));
        assert_eq!(model.rank_of("attn.k"), Some(8));
        assert_eq!(model.rank_of("ffn.up"), Some(4));
        assert_eq!(model.distinct_ranks(), vec![4, 8, 16]);
        assert_eq!(model.layer_names(), &["attn.q", "attn.k", "ffn.up"]);
    }

    #[test]
    fn duplicate_name_rejected() {
        let specs = vec![
            LayerSpec::new("dup", 8, 8, 2, 2.0).expect("spec"),
            LayerSpec::new("dup", 8, 8, 4, 4.0).expect("spec"),
        ];
        assert!(MixedRankLoraModel::from_specs(&specs, &mut rng()).is_err());
    }

    #[test]
    fn total_params_sum() {
        let specs = vec![
            LayerSpec::new("a", 8, 8, 2, 2.0).expect("spec"),
            LayerSpec::new("b", 16, 32, 4, 4.0).expect("spec"),
        ];
        let model = MixedRankLoraModel::from_specs(&specs, &mut rng()).expect("model");
        // 2*(8+8) + 4*(16+32) = 32 + 192 = 224
        assert_eq!(model.total_params(), 224);
    }

    #[test]
    fn forward_layer_uses_per_layer_rank() {
        // Two layers of different rank; B is zero-initialised so each layer's
        // forward returns its base output unchanged, but the call must route
        // to the correctly-shaped adapter without dimension errors.
        let specs = vec![
            LayerSpec::new("small", 8, 16, 2, 2.0).expect("spec"),
            LayerSpec::new("large", 32, 64, 16, 16.0).expect("spec"),
        ];
        let model = MixedRankLoraModel::from_specs(&specs, &mut rng()).expect("model");

        let x_s = vec![0.3_f32; 8];
        let base_s = vec![0.5_f32; 16];
        let out_s = model
            .forward_layer("small", &x_s, &base_s, 1)
            .expect("forward small");
        for (&o, &b) in out_s.iter().zip(&base_s) {
            assert!((o - b).abs() < 1e-5, "B=0 keeps base: {o} vs {b}");
        }

        let x_l = vec![0.1_f32; 32];
        let base_l = vec![0.0_f32; 64];
        let out_l = model
            .forward_layer("large", &x_l, &base_l, 1)
            .expect("forward large");
        assert_eq!(out_l.len(), 64);
    }

    #[test]
    fn forward_unknown_layer_errors() {
        let model = MixedRankLoraModel::new();
        assert!(model.is_empty());
        assert!(model.forward_layer("nope", &[0.0], &[0.0], 1).is_err());
    }

    #[test]
    fn budget_uniform_respects_ceiling_and_bounds() {
        let layers = vec![
            ("a".to_string(), 64usize, 64usize),
            ("b".to_string(), 64, 64),
            ("c".to_string(), 64, 64),
        ];
        // width = 128 each. Budget 1536 ⇒ 1536/128 = 12 rank-units total ⇒ 4 each.
        let budget = RankBudget::new(1536, BudgetStrategy::Uniform, 1, 32).expect("budget");
        let specs = budget.allocate(&layers).expect("allocate");
        assert_eq!(specs.len(), 3);
        let total: usize = specs.iter().map(LayerSpec::param_count).sum();
        assert!(total <= 1536, "over budget: {total}");
        for s in &specs {
            assert!(s.rank >= 1 && s.rank <= 32);
        }
        // Uniform ⇒ ranks balanced (all equal here).
        assert_eq!(specs[0].rank, 4);
        assert_eq!(specs[1].rank, 4);
        assert_eq!(specs[2].rank, 4);
    }

    #[test]
    fn budget_width_proportional_favours_wide_layers() {
        let layers = vec![
            ("narrow".to_string(), 16usize, 16usize), // width 32
            ("wide".to_string(), 256, 256),           // width 512
        ];
        let budget =
            RankBudget::new(20_000, BudgetStrategy::WidthProportional, 1, 64).expect("budget");
        let specs = budget.allocate(&layers).expect("allocate");
        let narrow = specs.iter().find(|s| s.name == "narrow").expect("narrow");
        let wide = specs.iter().find(|s| s.name == "wide").expect("wide");
        // The wider layer should receive a rank at least as large (and here
        // strictly larger because the proportional priority favours it until
        // it caps).
        assert!(
            wide.rank >= narrow.rank,
            "wide rank {} should be >= narrow rank {}",
            wide.rank,
            narrow.rank
        );
        let total: usize = specs.iter().map(LayerSpec::param_count).sum();
        assert!(total <= 20_000, "over budget: {total}");
    }

    #[test]
    fn budget_validation() {
        assert!(RankBudget::new(100, BudgetStrategy::Uniform, 0, 4).is_err());
        assert!(RankBudget::new(100, BudgetStrategy::Uniform, 8, 4).is_err());
        let b = RankBudget::new(100, BudgetStrategy::Uniform, 2, 8).expect("ok");
        assert!(b.allocate(&[]).is_err());
        assert_eq!(b.max_params(), 100);
        assert_eq!(b.strategy(), BudgetStrategy::Uniform);
    }

    #[test]
    fn budget_min_rank_floor_when_no_extra_budget() {
        // Budget exactly equal to the min-rank floor leaves every layer at
        // min_rank.
        let layers = vec![("a".to_string(), 10usize, 10usize)]; // width 20
        let budget = RankBudget::new(20, BudgetStrategy::Uniform, 1, 8).expect("budget");
        let specs = budget.allocate(&layers).expect("allocate");
        assert_eq!(specs[0].rank, 1);
    }

    #[test]
    fn allocated_specs_build_a_model() {
        let layers = vec![
            ("attn".to_string(), 128usize, 128usize),
            ("ffn".to_string(), 128, 512),
        ];
        let budget =
            RankBudget::new(8_000, BudgetStrategy::WidthProportional, 2, 32).expect("budget");
        let specs = budget.allocate(&layers).expect("allocate");
        let model = MixedRankLoraModel::from_specs(&specs, &mut rng()).expect("model");
        assert_eq!(model.len(), 2);
        assert!(model.total_params() <= 8_000);
    }
}
