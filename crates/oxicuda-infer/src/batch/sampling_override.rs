//! # Per-Sequence Sampling Parameter Overrides
//!
//! The base engine carries one [`SamplingParams`] per request, but a real
//! serving API also wants *server-default* sampling that each request may
//! partially override (OpenAI-style: a deployment sets sensible defaults; a
//! request overrides only `temperature` and `max_tokens`). This module supplies
//! that two-layer resolution.
//!
//! [`SamplingOverride`] is an all-optional mirror of [`SamplingParams`]: a
//! `None` field means "inherit the default". [`SamplingOverride::apply`] merges
//! an override onto a base [`SamplingParams`], producing the effective
//! per-sequence parameters. A [`SamplingOverrideTable`] maps `SequenceId →
//! SamplingOverride`, so the scheduler can resolve each sequence's effective
//! sampling without forcing every caller to construct a full `SamplingParams`.

use std::collections::HashMap;

use crate::batch::sequence::{SamplingParams, SequenceId};

// ─── SamplingOverride ────────────────────────────────────────────────────────

/// An all-optional partial override of [`SamplingParams`].
///
/// Each `Some(_)` field replaces the corresponding base field; each `None`
/// inherits the base value.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingOverride {
    /// Override softmax temperature.
    pub temperature: Option<f32>,
    /// Override the top-K filter. `Some(None)` disables top-K; `Some(Some(k))`
    /// sets it; `None` inherits.
    pub top_k: Option<Option<usize>>,
    /// Override the top-P filter (same `Some(None)` = disable semantics).
    pub top_p: Option<Option<f32>>,
    /// Override the maximum new-token budget.
    pub max_new_tokens: Option<usize>,
    /// Override the EOS token id (`Some(None)` removes the EOS stop).
    pub eos_token_id: Option<Option<u32>>,
    /// Override the repetition penalty.
    pub repetition_penalty: Option<f32>,
}

impl SamplingOverride {
    /// An empty override (inherits everything).
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// Are all fields unset (a pure inherit)?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.temperature.is_none()
            && self.top_k.is_none()
            && self.top_p.is_none()
            && self.max_new_tokens.is_none()
            && self.eos_token_id.is_none()
            && self.repetition_penalty.is_none()
    }

    /// Builder: set the temperature override.
    #[must_use]
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Builder: set the max-new-tokens override.
    #[must_use]
    pub fn with_max_new_tokens(mut self, n: usize) -> Self {
        self.max_new_tokens = Some(n);
        self
    }

    /// Builder: set the top-P override (`None` disables top-P).
    #[must_use]
    pub fn with_top_p(mut self, p: Option<f32>) -> Self {
        self.top_p = Some(p);
        self
    }

    /// Builder: set the top-K override (`None` disables top-K).
    #[must_use]
    pub fn with_top_k(mut self, k: Option<usize>) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Resolve effective sampling parameters by layering `self` onto `base`.
    #[must_use]
    pub fn apply(&self, base: &SamplingParams) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature.unwrap_or(base.temperature),
            top_k: self.top_k.unwrap_or(base.top_k),
            top_p: self.top_p.unwrap_or(base.top_p),
            max_new_tokens: self.max_new_tokens.unwrap_or(base.max_new_tokens),
            eos_token_id: self.eos_token_id.unwrap_or(base.eos_token_id),
            repetition_penalty: self.repetition_penalty.unwrap_or(base.repetition_penalty),
        }
    }
}

// ─── SamplingOverrideTable ───────────────────────────────────────────────────

/// Per-sequence override registry layered on a shared base [`SamplingParams`].
#[derive(Debug, Clone)]
pub struct SamplingOverrideTable {
    base: SamplingParams,
    overrides: HashMap<SequenceId, SamplingOverride>,
}

impl SamplingOverrideTable {
    /// Create a table with the given server-default base parameters.
    #[must_use]
    pub fn new(base: SamplingParams) -> Self {
        Self {
            base,
            overrides: HashMap::new(),
        }
    }

    /// Register an override for a sequence (replaces any prior override).
    pub fn set(&mut self, seq_id: SequenceId, ov: SamplingOverride) {
        self.overrides.insert(seq_id, ov);
    }

    /// Remove a sequence's override (e.g. when it finishes).
    pub fn remove(&mut self, seq_id: SequenceId) {
        self.overrides.remove(&seq_id);
    }

    /// Number of registered overrides.
    #[must_use]
    pub fn len(&self) -> usize {
        self.overrides.len()
    }

    /// Are there no registered overrides?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.overrides.is_empty()
    }

    /// Resolve the effective [`SamplingParams`] for `seq_id`.
    ///
    /// If the sequence has no override, the base parameters are returned
    /// unchanged.
    #[must_use]
    pub fn resolve(&self, seq_id: SequenceId) -> SamplingParams {
        match self.overrides.get(&seq_id) {
            Some(ov) => ov.apply(&self.base),
            None => self.base.clone(),
        }
    }

    /// Immutable access to the base parameters.
    #[must_use]
    pub fn base(&self) -> &SamplingParams {
        &self.base
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> SamplingParams {
        SamplingParams {
            temperature: 1.0,
            top_k: Some(50),
            top_p: Some(0.9),
            max_new_tokens: 256,
            eos_token_id: Some(2),
            repetition_penalty: 1.1,
        }
    }

    #[test]
    fn empty_override_inherits_all() {
        let ov = SamplingOverride::none();
        assert!(ov.is_empty());
        let eff = ov.apply(&base());
        assert_eq!(eff.temperature, 1.0);
        assert_eq!(eff.top_k, Some(50));
        assert_eq!(eff.max_new_tokens, 256);
        assert_eq!(eff.eos_token_id, Some(2));
    }

    #[test]
    fn partial_override_replaces_only_set_fields() {
        let ov = SamplingOverride::none()
            .with_temperature(0.2)
            .with_max_new_tokens(16);
        assert!(!ov.is_empty());
        let eff = ov.apply(&base());
        assert_eq!(eff.temperature, 0.2, "overridden");
        assert_eq!(eff.max_new_tokens, 16, "overridden");
        assert_eq!(eff.top_k, Some(50), "inherited");
        assert_eq!(eff.repetition_penalty, 1.1, "inherited");
    }

    #[test]
    fn override_can_disable_top_k_and_top_p() {
        let ov = SamplingOverride::none().with_top_k(None).with_top_p(None);
        let eff = ov.apply(&base());
        assert_eq!(eff.top_k, None, "top-k disabled via Some(None)");
        assert_eq!(eff.top_p, None, "top-p disabled via Some(None)");
    }

    #[test]
    fn override_can_remove_eos() {
        let mut ov = SamplingOverride::none();
        ov.eos_token_id = Some(None);
        let eff = ov.apply(&base());
        assert_eq!(eff.eos_token_id, None, "EOS stop removed");
    }

    #[test]
    fn table_resolves_per_sequence() {
        let mut table = SamplingOverrideTable::new(base());
        assert!(table.is_empty());
        table.set(7, SamplingOverride::none().with_temperature(0.0));
        assert_eq!(table.len(), 1);

        // Sequence 7 uses the override; an unknown sequence uses the base.
        let s7 = table.resolve(7);
        assert_eq!(s7.temperature, 0.0);
        assert_eq!(s7.top_k, Some(50), "non-overridden field inherits base");

        let s_default = table.resolve(99);
        assert_eq!(s_default.temperature, 1.0);
    }

    #[test]
    fn table_remove_reverts_to_base() {
        let mut table = SamplingOverrideTable::new(base());
        table.set(3, SamplingOverride::none().with_max_new_tokens(4));
        assert_eq!(table.resolve(3).max_new_tokens, 4);
        table.remove(3);
        assert_eq!(table.resolve(3).max_new_tokens, 256, "back to base");
        assert!(table.is_empty());
    }

    #[test]
    fn base_accessor() {
        let table = SamplingOverrideTable::new(base());
        assert_eq!(table.base().temperature, 1.0);
    }

    #[test]
    fn override_setting_eos_token_to_value() {
        let mut ov = SamplingOverride::none();
        ov.eos_token_id = Some(Some(99));
        let eff = ov.apply(&base());
        assert_eq!(eff.eos_token_id, Some(99));
    }
}
