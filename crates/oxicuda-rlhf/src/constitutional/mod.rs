//! Constitutional AI (CAI): revision-based self-critique pipeline primitives.
//!
//! Reference: Bai et al. 2022, "Constitutional AI: Harmlessness from AI
//! Feedback", arXiv:2212.08073.
//!
//! Constitutional AI removes humans from the harmlessness loop. Its supervised
//! stage is an iterative *critique → revise* loop driven by a written
//! constitution (a list of principles):
//!
//! ```text
//! response_0  = initial model answer
//! for each round:
//!     critique   = model("Identify how response violates principle P")
//!     response_k = model("Rewrite the response to address that critique")
//! collect (prompt, final response_k) as SL fine-tuning data
//! ```
//!
//! The revised responses become supervised-learning targets, and (original,
//! revised) pairs seed the later preference / RLAIF stage where "revised is
//! preferred".
//!
//! Because this crate executes no language model, the *generation* steps are
//! injected by the caller as closures (`critique_fn`, `revise_fn`) — the module
//! owns the deterministic **pipeline control flow**: principle rotation across
//! rounds, fixed-round iteration, collection of the revision trace, assembly of
//! supervised-learning records, and construction of preference pairs that mark
//! the revised text as chosen. All of that is real, testable logic; only the
//! token-level generation is delegated. A trivial built-in
//! [`heuristic_revision`] lets the pipeline be exercised end-to-end in tests
//! without any model.

use crate::error::{RlhfError, RlhfResult};

// ── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the constitutional self-critique loop.
#[derive(Debug, Clone)]
pub struct ConstitutionalConfig {
    /// The constitution: a non-empty list of principles. Successive critique
    /// rounds cycle through these in order.
    pub principles: Vec<String>,
    /// Number of critique→revise rounds to run (≥ 1).
    pub rounds: usize,
}

impl ConstitutionalConfig {
    /// Validate: at least one principle and at least one round.
    ///
    /// # Errors
    ///
    /// Returns [`RlhfError::EmptyInput`] if `principles` is empty or
    /// `rounds == 0`.
    pub fn validate(&self) -> RlhfResult<()> {
        if self.principles.is_empty() {
            return Err(RlhfError::EmptyInput);
        }
        if self.rounds == 0 {
            return Err(RlhfError::EmptyInput);
        }
        Ok(())
    }

    /// The principle used in round `round` (0-based), cycling through the list.
    ///
    /// # Errors
    ///
    /// Returns [`RlhfError::EmptyInput`] if there are no principles.
    pub fn principle_for_round(&self, round: usize) -> RlhfResult<&str> {
        if self.principles.is_empty() {
            return Err(RlhfError::EmptyInput);
        }
        let idx = round % self.principles.len();
        Ok(self.principles[idx].as_str())
    }
}

// ── Revision trace ──────────────────────────────────────────────────────────

/// One critique→revise round's record.
#[derive(Debug, Clone, PartialEq)]
pub struct RevisionRound {
    /// The principle applied in this round.
    pub principle: String,
    /// The critique text the model produced for the current response.
    pub critique: String,
    /// The revised response produced from that critique.
    pub revised: String,
}

/// The full output of a constitutional revision run for one prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstitutionalTrace {
    /// The originating prompt.
    pub prompt: String,
    /// The initial (pre-revision) response.
    pub initial_response: String,
    /// Per-round critique/revision records, in order.
    pub rounds: Vec<RevisionRound>,
}

impl ConstitutionalTrace {
    /// The final revised response (the last round's revision, or the initial
    /// response if no rounds ran — which cannot happen for a validated config).
    #[must_use]
    pub fn final_response(&self) -> &str {
        self.rounds
            .last()
            .map_or(self.initial_response.as_str(), |r| r.revised.as_str())
    }
}

// ── Supervised-learning record / preference pair ────────────────────────────

/// A supervised-learning record harvested from a constitutional trace: the
/// prompt paired with its final revised (target) response.
#[derive(Debug, Clone, PartialEq)]
pub struct CaiSlRecord {
    /// Prompt text.
    pub prompt: String,
    /// Final revised response used as the SL target.
    pub target_response: String,
}

/// A preference pair for the RLAIF stage: the revised response is chosen, the
/// original is rejected.
#[derive(Debug, Clone, PartialEq)]
pub struct CaiPreferencePair {
    /// Prompt text.
    pub prompt: String,
    /// The preferred (revised) response.
    pub chosen: String,
    /// The dispreferred (original) response.
    pub rejected: String,
}

// ── Pipeline driver ─────────────────────────────────────────────────────────

/// Run the constitutional critique→revise loop for one prompt.
///
/// `critique_fn(principle, current_response)` returns a critique string;
/// `revise_fn(principle, critique, current_response)` returns the revised
/// response that the next round operates on. The loop runs `cfg.rounds` rounds,
/// cycling principles via [`ConstitutionalConfig::principle_for_round`], and
/// records every round.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`] for an empty prompt / initial response or
/// an invalid config, and propagates any error returned by the supplied
/// closures.
pub fn run_constitutional_revision<C, R>(
    prompt: &str,
    initial_response: &str,
    cfg: &ConstitutionalConfig,
    mut critique_fn: C,
    mut revise_fn: R,
) -> RlhfResult<ConstitutionalTrace>
where
    C: FnMut(&str, &str) -> RlhfResult<String>,
    R: FnMut(&str, &str, &str) -> RlhfResult<String>,
{
    cfg.validate()?;
    if prompt.is_empty() || initial_response.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let mut current = initial_response.to_string();
    let mut rounds = Vec::with_capacity(cfg.rounds);
    for round in 0..cfg.rounds {
        let principle = cfg.principle_for_round(round)?.to_string();
        let critique = critique_fn(&principle, &current)?;
        let revised = revise_fn(&principle, &critique, &current)?;
        rounds.push(RevisionRound {
            principle,
            critique,
            revised: revised.clone(),
        });
        current = revised;
    }
    Ok(ConstitutionalTrace {
        prompt: prompt.to_string(),
        initial_response: initial_response.to_string(),
        rounds,
    })
}

/// Extract the supervised-learning record (prompt → final revised response) from
/// a trace.
#[must_use]
pub fn collect_sl_record(trace: &ConstitutionalTrace) -> CaiSlRecord {
    CaiSlRecord {
        prompt: trace.prompt.clone(),
        target_response: trace.final_response().to_string(),
    }
}

/// Build the (revised-chosen, original-rejected) preference pair from a trace.
///
/// # Errors
///
/// Returns [`RlhfError::NoValidPair`] if the final revision is byte-identical to
/// the initial response (no actual revision happened, so the pair carries no
/// preference signal).
pub fn collect_preference_pair(trace: &ConstitutionalTrace) -> RlhfResult<CaiPreferencePair> {
    let chosen = trace.final_response().to_string();
    if chosen == trace.initial_response {
        return Err(RlhfError::NoValidPair {
            msg: "revised response identical to original — no preference signal".to_string(),
        });
    }
    Ok(CaiPreferencePair {
        prompt: trace.prompt.clone(),
        chosen,
        rejected: trace.initial_response.clone(),
    })
}

// ── Built-in heuristic revision (model-free, for testing/bootstrapping) ──────

/// A trivial model-free revision step: appends a principle-tagged disclaimer.
///
/// This is **not** a language model — it deterministically transforms the text
/// so the pipeline can be exercised end-to-end without one. It returns
/// `(critique, revised)` where the critique names the principle and the revised
/// text appends a bracketed note. Real deployments pass their own closures to
/// [`run_constitutional_revision`].
#[must_use]
pub fn heuristic_revision(principle: &str, response: &str) -> (String, String) {
    let critique = format!("response may violate principle: {principle}");
    let revised = format!("{response} [revised per: {principle}]");
    (critique, revised)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n_principles: usize, rounds: usize) -> ConstitutionalConfig {
        let principles = (0..n_principles).map(|i| format!("P{i}")).collect();
        ConstitutionalConfig { principles, rounds }
    }

    fn run_with_heuristic(
        prompt: &str,
        init: &str,
        cfg: &ConstitutionalConfig,
    ) -> ConstitutionalTrace {
        run_constitutional_revision(
            prompt,
            init,
            cfg,
            |principle, resp| Ok(heuristic_revision(principle, resp).0),
            |principle, _crit, resp| Ok(heuristic_revision(principle, resp).1),
        )
        .expect("revision run")
    }

    // 1. Principle rotation cycles through the constitution.
    #[test]
    fn principle_rotation() {
        let c = cfg(2, 5);
        assert_eq!(c.principle_for_round(0).expect("p"), "P0");
        assert_eq!(c.principle_for_round(1).expect("p"), "P1");
        assert_eq!(c.principle_for_round(2).expect("p"), "P0");
        assert_eq!(c.principle_for_round(3).expect("p"), "P1");
        assert_eq!(c.principle_for_round(4).expect("p"), "P0");
    }

    // 2. Run produces exactly `rounds` revision records.
    #[test]
    fn run_records_all_rounds() {
        let c = cfg(2, 3);
        let trace = run_with_heuristic("How do I X?", "Initial answer.", &c);
        assert_eq!(trace.rounds.len(), 3);
        assert_eq!(trace.prompt, "How do I X?");
        assert_eq!(trace.initial_response, "Initial answer.");
    }

    // 3. Each round's revision feeds the next round's input.
    #[test]
    fn revision_chains_through_rounds() {
        let c = cfg(1, 2);
        let trace = run_with_heuristic("Q", "Base", &c);
        // Round 0 revises "Base"; round 1 must operate on round 0's output.
        let r0 = &trace.rounds[0].revised;
        // Round 1's revised must contain round 0's revised as a prefix.
        assert!(
            trace.rounds[1].revised.starts_with(r0.as_str()),
            "round 1 should build on round 0: '{}' not prefixed by '{}'",
            trace.rounds[1].revised,
            r0
        );
    }

    // 4. final_response is the last round's revision.
    #[test]
    fn final_response_is_last_revision() {
        let c = cfg(1, 2);
        let trace = run_with_heuristic("Q", "Base", &c);
        assert_eq!(trace.final_response(), trace.rounds[1].revised.as_str());
        assert_ne!(
            trace.final_response(),
            "Base",
            "revision should change text"
        );
    }

    // 5. Critique text names the round's principle.
    #[test]
    fn critique_references_principle() {
        let c = cfg(2, 2);
        let trace = run_with_heuristic("Q", "Base", &c);
        assert!(trace.rounds[0].critique.contains("P0"));
        assert!(trace.rounds[1].critique.contains("P1"));
    }

    // 6. SL record carries prompt and final revision.
    #[test]
    fn sl_record_extraction() {
        let c = cfg(1, 2);
        let trace = run_with_heuristic("Prompt", "Base", &c);
        let rec = collect_sl_record(&trace);
        assert_eq!(rec.prompt, "Prompt");
        assert_eq!(rec.target_response, trace.final_response());
    }

    // 7. Preference pair marks revised as chosen, original as rejected.
    #[test]
    fn preference_pair_orientation() {
        let c = cfg(1, 1);
        let trace = run_with_heuristic("Prompt", "Base", &c);
        let pair = collect_preference_pair(&trace).expect("pair");
        assert_eq!(pair.rejected, "Base");
        assert_eq!(pair.chosen, trace.final_response());
        assert_ne!(pair.chosen, pair.rejected);
    }

    // 8. Identical revision → NoValidPair.
    #[test]
    fn identical_revision_no_pair() {
        let c = cfg(1, 1);
        // Identity revise_fn: no change.
        let trace = run_constitutional_revision(
            "Q",
            "Same",
            &c,
            |_, _| Ok("crit".to_string()),
            |_, _, resp| Ok(resp.to_string()),
        )
        .expect("run");
        assert!(matches!(
            collect_preference_pair(&trace),
            Err(RlhfError::NoValidPair { .. })
        ));
    }

    // 9. Custom closures are honoured and threaded correctly.
    #[test]
    fn custom_closures_used() {
        let c = cfg(1, 2);
        let trace = run_constitutional_revision(
            "Q",
            "x",
            &c,
            |_p, resp| Ok(format!("critique({resp})")),
            |_p, _c, resp| Ok(format!("{resp}+")),
        )
        .expect("run");
        // "x" -> "x+" -> "x++"
        assert_eq!(trace.rounds[0].revised, "x+");
        assert_eq!(trace.rounds[1].revised, "x++");
        assert_eq!(trace.final_response(), "x++");
    }

    // 10. Closure error propagates.
    #[test]
    fn closure_error_propagates() {
        let c = cfg(1, 1);
        let res = run_constitutional_revision(
            "Q",
            "x",
            &c,
            |_, _| {
                Err(RlhfError::Internal {
                    msg: "boom".to_string(),
                })
            },
            |_, _c, resp| Ok(resp.to_string()),
        );
        assert!(matches!(res, Err(RlhfError::Internal { .. })));
    }

    // 11. Empty principles / zero rounds rejected.
    #[test]
    fn invalid_config_errors() {
        let no_principles = ConstitutionalConfig {
            principles: vec![],
            rounds: 1,
        };
        assert!(matches!(
            no_principles.validate(),
            Err(RlhfError::EmptyInput)
        ));
        let zero_rounds = ConstitutionalConfig {
            principles: vec!["P".to_string()],
            rounds: 0,
        };
        assert!(matches!(zero_rounds.validate(), Err(RlhfError::EmptyInput)));
    }

    // 12. Empty prompt / response rejected.
    #[test]
    fn empty_inputs_rejected() {
        let c = cfg(1, 1);
        let r1 =
            run_constitutional_revision("", "x", &c, |_, _| Ok("c".into()), |_, _, r| Ok(r.into()));
        assert!(matches!(r1, Err(RlhfError::EmptyInput)));
        let r2 =
            run_constitutional_revision("p", "", &c, |_, _| Ok("c".into()), |_, _, r| Ok(r.into()));
        assert!(matches!(r2, Err(RlhfError::EmptyInput)));
    }

    // 13. Single-round run with one principle works.
    #[test]
    fn single_round_single_principle() {
        let c = cfg(1, 1);
        let trace = run_with_heuristic("Q", "Base", &c);
        assert_eq!(trace.rounds.len(), 1);
        assert_eq!(trace.rounds[0].principle, "P0");
    }

    // 14. More principles than rounds: only the first `rounds` are used.
    #[test]
    fn more_principles_than_rounds() {
        let c = cfg(5, 2);
        let trace = run_with_heuristic("Q", "Base", &c);
        assert_eq!(trace.rounds[0].principle, "P0");
        assert_eq!(trace.rounds[1].principle, "P1");
    }
}
