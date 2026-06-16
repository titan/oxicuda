//! Successive Halving (SHA) and Hyperband multi-fidelity NAS.
//!
//! References:
//! - Jamieson & Talwalkar, "Non-stochastic Best Arm Identification and
//!   Hyperparameter Optimization", AISTATS 2016 (Successive Halving).
//! - Li, Jamieson, DeSalvo, Rostamizadeh & Talwalkar, "Hyperband: A Novel
//!   Bandit-Based Approach to Hyperparameter Optimization", JMLR 2017.
//!
//! # Successive Halving
//!
//! Given `N` candidate architectures and a reduction factor `η > 1`, SHA spends
//! a *small* budget on *every* candidate, keeps only the top `⌈n/η⌉`, multiplies
//! the per-candidate budget by `η`, and repeats. Poor candidates are eliminated
//! cheaply; budget concentrates on the survivors. After `≈ log_η(N)` rounds a
//! single candidate remains (or the maximum budget is reached). The total work
//! per round, `n_k · r_k`, is roughly constant, so SHA evaluates exponentially
//! more configurations than a fixed-budget random search for the same compute.
//!
//! # Hyperband
//!
//! SHA must trade off "many candidates, little budget each" against "few
//! candidates, lots of budget each". Hyperband hedges by running a geometric
//! sequence of SHA *brackets*, each with a different `(n, r)` starting point,
//! and returns the best architecture found across all brackets.
//!
//! The caller supplies an `evaluate(candidate, budget)` closure returning a
//! score where **higher is better**; SHA assumes the score is (weakly) monotone
//! in budget so that ranking at a small budget is informative about ranking at a
//! larger one.

use std::cmp::Ordering;

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;

// ─── ShaConfig ─────────────────────────────────────────────────────────────────

/// Configuration for [`SuccessiveHalving`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShaConfig {
    /// Reduction factor `η`. Each round keeps the top `⌈n/η⌉` candidates and
    /// multiplies the budget by `η`. Must be finite and `> 1`.
    pub eta: f32,
    /// Budget allocated to every candidate in the first round. Must be finite
    /// and `> 0`.
    pub min_budget: f32,
    /// Maximum per-candidate budget; halving stops once a round reaches it.
    /// Must be finite and `>= min_budget`.
    pub max_budget: f32,
}

impl ShaConfig {
    /// Construct a Successive-Halving configuration.
    #[must_use]
    pub fn new(eta: f32, min_budget: f32, max_budget: f32) -> Self {
        Self {
            eta,
            min_budget,
            max_budget,
        }
    }

    /// Validate the schedule parameters.
    ///
    /// # Errors
    /// [`NasError::Internal`] with a descriptive message if `eta`, `min_budget`
    /// or `max_budget` are non-finite or out of range.
    pub fn validate(&self) -> NasResult<()> {
        if !self.eta.is_finite() || self.eta <= 1.0 {
            return Err(NasError::Internal(format!(
                "successive halving requires a finite reduction factor eta > 1, got {}",
                self.eta
            )));
        }
        if !self.min_budget.is_finite() || self.min_budget <= 0.0 {
            return Err(NasError::Internal(format!(
                "min_budget must be finite and > 0, got {}",
                self.min_budget
            )));
        }
        if !self.max_budget.is_finite() || self.max_budget < self.min_budget {
            return Err(NasError::Internal(format!(
                "max_budget {} must be finite and >= min_budget {}",
                self.max_budget, self.min_budget
            )));
        }
        Ok(())
    }
}

// ─── ShaResult ─────────────────────────────────────────────────────────────────

/// Per-round bookkeeping for a Successive-Halving run.
#[derive(Debug, Clone, PartialEq)]
pub struct RoundInfo {
    /// Per-candidate budget used in this round.
    pub budget: f32,
    /// Original candidate indices evaluated in this round (the survivors that
    /// entered the round).
    pub survivors: Vec<usize>,
    /// Number of candidates evaluated this round (`== survivors.len()`).
    pub n_evaluated: usize,
}

/// Result of a Successive-Halving run.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaResult {
    /// Index (into the input `candidates` slice) of the winning architecture —
    /// the highest-scoring survivor in the final (largest-budget) round.
    pub best_index: usize,
    /// Score of the winner in the final round.
    pub best_score: f32,
    /// Round-by-round schedule (budgets, survivor sets, counts).
    pub rounds: Vec<RoundInfo>,
}

impl ShaResult {
    /// Number of rounds executed.
    #[must_use]
    pub fn n_rounds(&self) -> usize {
        self.rounds.len()
    }

    /// Total resource consumed: `Σ_k n_evaluated_k · budget_k`.
    #[must_use]
    pub fn total_resource(&self) -> f32 {
        self.rounds
            .iter()
            .map(|r| r.n_evaluated as f32 * r.budget)
            .sum()
    }
}

// ─── Selection helper ──────────────────────────────────────────────────────────

/// Order `(index, score)` pairs descending by score, ties broken by lower index,
/// with non-finite scores (`NaN`) sinking to the end. Matches the convention of
/// [`crate::proxy::zero_cost::rank_architectures`].
fn sort_desc_by_score(scored: &mut [(usize, f32)]) {
    scored.sort_by(|&(ia, sa), &(ib, sb)| match sb.partial_cmp(&sa) {
        Some(Ordering::Equal) => ia.cmp(&ib),
        Some(o) => o,
        None => match (sa.is_nan(), sb.is_nan()) {
            (false, true) => Ordering::Less,
            (true, false) => Ordering::Greater,
            _ => ia.cmp(&ib),
        },
    });
}

/// Number of survivors kept from `n` candidates under reduction factor `eta`:
/// `⌈n / eta⌉`, clamped to `[1, n-1]` so the population strictly shrinks every
/// round (guaranteeing termination even for non-integer `eta`).
fn keep_count(n: usize, eta: f32) -> usize {
    debug_assert!(n >= 2);
    let raw = (n as f32 / eta).ceil() as usize;
    raw.clamp(1, n - 1)
}

// ─── SuccessiveHalving ──────────────────────────────────────────────────────────

/// Successive-Halving search over a fixed candidate pool.
#[derive(Debug, Clone, Copy)]
pub struct SuccessiveHalving {
    config: ShaConfig,
}

impl SuccessiveHalving {
    /// Create a Successive-Halving searcher from its configuration.
    #[must_use]
    pub fn new(config: ShaConfig) -> Self {
        Self { config }
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &ShaConfig {
        &self.config
    }

    /// Run Successive Halving over `candidates`.
    ///
    /// `evaluate(candidate, budget)` returns a score (**higher is better**);
    /// it is assumed weakly monotone in `budget`. Returns the winning candidate
    /// index, its final-round score, and the full round schedule.
    ///
    /// # Errors
    /// - [`NasError::EmptySearchSpace`] if `candidates` is empty.
    /// - [`NasError::Internal`] if the [`ShaConfig`] schedule is invalid.
    pub fn run<C, F>(&self, candidates: &[C], evaluate: F) -> NasResult<ShaResult>
    where
        F: Fn(&C, f32) -> f32,
    {
        self.config.validate()?;
        if candidates.is_empty() {
            return Err(NasError::EmptySearchSpace);
        }

        let eta = self.config.eta;
        let max_budget = self.config.max_budget;
        let mut budget = self.config.min_budget;
        let mut survivors: Vec<usize> = (0..candidates.len()).collect();
        let mut rounds: Vec<RoundInfo> = Vec::new();

        // Iterate halving rounds; break out with the final round's ranking
        // (the `(index, score)` pairs sorted best-first).
        let ranked: Vec<(usize, f32)> = loop {
            // Evaluate every current survivor at the current budget.
            let mut scored: Vec<(usize, f32)> = survivors
                .iter()
                .map(|&i| (i, evaluate(&candidates[i], budget)))
                .collect();
            rounds.push(RoundInfo {
                budget,
                survivors: survivors.clone(),
                n_evaluated: survivors.len(),
            });
            sort_desc_by_score(&mut scored);

            // Stop when a single candidate remains or the budget cap is reached.
            if survivors.len() == 1 || budget >= max_budget {
                break scored;
            }

            let keep = keep_count(survivors.len(), eta);
            survivors = scored.iter().take(keep).map(|&(i, _)| i).collect();
            budget = (budget * eta).min(max_budget);
        };

        let (best_index, best_score) = match ranked.first() {
            Some(&pair) => pair,
            None => return Err(NasError::EmptySearchSpace),
        };
        Ok(ShaResult {
            best_index,
            best_score,
            rounds,
        })
    }
}

// ─── Hyperband ──────────────────────────────────────────────────────────────────

/// Configuration for [`Hyperband`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperbandConfig {
    /// Reduction factor `η` shared by every bracket. Must be finite and `> 1`.
    pub eta: f32,
    /// Smallest per-candidate budget (`R · η^{-s_max}`). Must be finite and `> 0`.
    pub min_budget: f32,
    /// Largest per-candidate budget `R`. Must be finite and `>= min_budget`.
    pub max_budget: f32,
}

impl HyperbandConfig {
    /// Construct a Hyperband configuration.
    #[must_use]
    pub fn new(eta: f32, min_budget: f32, max_budget: f32) -> Self {
        Self {
            eta,
            min_budget,
            max_budget,
        }
    }

    /// Validate the parameters (same rules as [`ShaConfig::validate`]).
    ///
    /// # Errors
    /// [`NasError::Internal`] for non-finite / out-of-range parameters.
    pub fn validate(&self) -> NasResult<()> {
        ShaConfig::new(self.eta, self.min_budget, self.max_budget).validate()
    }

    /// Maximum bracket index `s_max = ⌊log_η(max_budget / min_budget)⌋`.
    #[must_use]
    pub fn s_max(&self) -> usize {
        let ratio = (self.max_budget / self.min_budget) as f64;
        let eta = self.eta as f64;
        ratio.log(eta).floor().max(0.0) as usize
    }
}

/// Summary of one Hyperband bracket.
#[derive(Debug, Clone, PartialEq)]
pub struct BracketResult {
    /// Bracket index `s` (larger `s` ⇒ more candidates, smaller initial budget).
    pub s: usize,
    /// Number of candidates this bracket started with.
    pub n_configs: usize,
    /// Initial per-candidate budget for this bracket (`R · η^{-s}`).
    pub initial_budget: f32,
    /// Best score found within this bracket.
    pub best_score: f32,
}

/// Result of a Hyperband run.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperbandResult<C> {
    /// Best architecture found across all brackets.
    pub best: C,
    /// Score of [`HyperbandResult::best`].
    pub best_score: f32,
    /// Per-bracket summaries, ordered from the largest `s` down to `s = 0`.
    pub brackets: Vec<BracketResult>,
}

/// Hyperband: a sequence of Successive-Halving brackets over freshly sampled
/// candidate pools.
#[derive(Debug, Clone, Copy)]
pub struct Hyperband {
    config: HyperbandConfig,
}

impl Hyperband {
    /// Create a Hyperband searcher from its configuration.
    #[must_use]
    pub fn new(config: HyperbandConfig) -> Self {
        Self { config }
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &HyperbandConfig {
        &self.config
    }

    /// Run Hyperband.
    ///
    /// `factory(rng)` samples a fresh candidate (called `n_s` times per bracket);
    /// `evaluate(candidate, budget)` scores a candidate at a budget (**higher is
    /// better**). Returns the globally best candidate, its score, and per-bracket
    /// summaries.
    ///
    /// # Errors
    /// - [`NasError::Internal`] if the [`HyperbandConfig`] is invalid.
    /// - [`NasError::NoFeasibleArchitecture`] if no bracket produced a candidate
    ///   (only possible for a degenerate schedule).
    pub fn run<C, G, F>(
        &self,
        mut factory: G,
        evaluate: F,
        rng: &mut LcgRng,
    ) -> NasResult<HyperbandResult<C>>
    where
        C: Clone,
        G: FnMut(&mut LcgRng) -> C,
        F: Fn(&C, f32) -> f32,
    {
        self.config.validate()?;
        let eta = self.config.eta;
        let r_max = self.config.max_budget;
        let s_max = self.config.s_max();
        // Budget B = (s_max + 1) · R distributes evenly across brackets.
        let b_over_r = (s_max + 1) as f32;

        let mut brackets: Vec<BracketResult> = Vec::with_capacity(s_max + 1);
        let mut best: Option<C> = None;
        let mut best_score = f32::NEG_INFINITY;

        // Brackets run from the most-exploratory (s = s_max) down to s = 0.
        for s in (0..=s_max).rev() {
            let eta_s = (eta as f64).powi(s as i32);
            let n_configs = (b_over_r as f64 * eta_s / (s as f64 + 1.0)).ceil() as usize;
            let n_configs = n_configs.max(1);
            let initial_budget = (r_max as f64 / eta_s) as f32;

            // Sample this bracket's candidate pool.
            let pool: Vec<C> = (0..n_configs).map(|_| factory(rng)).collect();

            // Run SHA within the bracket, starting at `initial_budget`.
            let sha = SuccessiveHalving::new(ShaConfig::new(eta, initial_budget, r_max));
            let res = sha.run(&pool, &evaluate)?;
            let bracket_best = match pool.get(res.best_index) {
                Some(c) => c.clone(),
                None => return Err(NasError::Internal("bracket best index out of range".into())),
            };

            brackets.push(BracketResult {
                s,
                n_configs,
                initial_budget,
                best_score: res.best_score,
            });

            if best.is_none() || res.best_score > best_score {
                best_score = res.best_score;
                best = Some(bracket_best);
            }
        }

        let best = best.ok_or(NasError::NoFeasibleArchitecture)?;
        Ok(HyperbandResult {
            best,
            best_score,
            brackets,
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// `evaluate` whose ranking is fixed by candidate "quality" (its value),
    /// with a vanishing budget bonus so it is strictly monotone in budget yet
    /// never reorders candidates.
    fn quality_eval(c: &usize, budget: f32) -> f32 {
        *c as f32 + 1e-4 * budget
    }

    #[test]
    fn best_candidate_survives_to_final_round() {
        // Candidate N-1 has the highest quality at every budget ⇒ always kept.
        let candidates: Vec<usize> = (0..8).collect();
        let sha = SuccessiveHalving::new(ShaConfig::new(2.0, 1.0, 8.0));
        let res = sha.run(&candidates, quality_eval).expect("sha");
        assert_eq!(res.best_index, 7);
    }

    #[test]
    fn survivor_counts_and_budget_follow_schedule() {
        // eta = 2, N = 8: 8 → 4 → 2 → 1 over budgets 1, 2, 4, 8.
        let candidates: Vec<usize> = (0..8).collect();
        let sha = SuccessiveHalving::new(ShaConfig::new(2.0, 1.0, 8.0));
        let res = sha.run(&candidates, quality_eval).expect("sha");

        let counts: Vec<usize> = res.rounds.iter().map(|r| r.n_evaluated).collect();
        assert_eq!(counts, vec![8, 4, 2, 1]);
        let budgets: Vec<f32> = res.rounds.iter().map(|r| r.budget).collect();
        assert_eq!(budgets, vec![1.0, 2.0, 4.0, 8.0]);

        // Each round keeps ⌈n / eta⌉ of the previous round.
        for w in counts.windows(2) {
            let expected = ((w[0] as f32) / 2.0).ceil() as usize;
            assert_eq!(w[1], expected.clamp(1, w[0].saturating_sub(1).max(1)));
        }

        // Total resource = Σ n_k · r_k = 8·1 + 4·2 + 2·4 + 1·8 = 32.
        assert_eq!(res.total_resource(), 32.0);
        assert_eq!(res.n_rounds(), 4);
    }

    #[test]
    fn eta_three_schedule() {
        // eta = 3, N = 9: 9 → 3 → 1 over budgets 1, 3, 9.
        let candidates: Vec<usize> = (0..9).collect();
        let sha = SuccessiveHalving::new(ShaConfig::new(3.0, 1.0, 9.0));
        let res = sha.run(&candidates, quality_eval).expect("sha");
        let counts: Vec<usize> = res.rounds.iter().map(|r| r.n_evaluated).collect();
        assert_eq!(counts, vec![9, 3, 1]);
        assert_eq!(res.best_index, 8);
    }

    #[test]
    fn single_candidate_trivially_wins() {
        let candidates = vec![42usize];
        let sha = SuccessiveHalving::new(ShaConfig::new(2.0, 1.0, 8.0));
        let res = sha.run(&candidates, quality_eval).expect("sha");
        assert_eq!(res.best_index, 0);
        assert_eq!(res.n_rounds(), 1);
        assert_eq!(res.rounds[0].n_evaluated, 1);
    }

    #[test]
    fn non_integer_eta_terminates() {
        // eta = 1.5 would keep ⌈2/1.5⌉ = 2 of 2 without the strict-shrink guard;
        // the clamp forces progress so the run terminates.
        let candidates: Vec<usize> = (0..5).collect();
        let sha = SuccessiveHalving::new(ShaConfig::new(1.5, 1.0, 16.0));
        let res = sha.run(&candidates, quality_eval).expect("sha");
        assert_eq!(res.best_index, 4);
        // Strictly decreasing survivor counts down to 1.
        let counts: Vec<usize> = res.rounds.iter().map(|r| r.n_evaluated).collect();
        for w in counts.windows(2) {
            assert!(w[1] < w[0], "counts must strictly decrease: {counts:?}");
        }
        assert_eq!(*counts.last().expect("last should succeed"), 1);
    }

    #[test]
    fn eta_le_one_errors() {
        let candidates: Vec<usize> = (0..4).collect();
        let sha = SuccessiveHalving::new(ShaConfig::new(1.0, 1.0, 8.0));
        assert!(matches!(
            sha.run(&candidates, quality_eval),
            Err(NasError::Internal(_))
        ));
    }

    #[test]
    fn empty_candidates_errors() {
        let candidates: Vec<usize> = Vec::new();
        let sha = SuccessiveHalving::new(ShaConfig::new(2.0, 1.0, 8.0));
        assert_eq!(
            sha.run(&candidates, quality_eval),
            Err(NasError::EmptySearchSpace)
        );
    }

    #[test]
    fn bad_budget_errors() {
        let candidates: Vec<usize> = (0..4).collect();
        let sha = SuccessiveHalving::new(ShaConfig::new(2.0, 4.0, 2.0)); // max < min
        assert!(matches!(
            sha.run(&candidates, quality_eval),
            Err(NasError::Internal(_))
        ));
    }

    #[test]
    fn deterministic_given_fixed_evaluations() {
        let candidates: Vec<usize> = (0..16).collect();
        let sha = SuccessiveHalving::new(ShaConfig::new(2.0, 1.0, 16.0));
        let a = sha.run(&candidates, quality_eval).expect("a");
        let b = sha.run(&candidates, quality_eval).expect("b");
        assert_eq!(a, b);
    }

    #[test]
    fn tie_break_prefers_lower_index() {
        // All-equal scores: lower index must survive each round.
        let candidates: Vec<usize> = (0..4).collect();
        let sha = SuccessiveHalving::new(ShaConfig::new(2.0, 1.0, 4.0));
        let res = sha.run(&candidates, |_, _| 1.0).expect("sha");
        assert_eq!(res.best_index, 0);
    }

    // ── Hyperband ──────────────────────────────────────────────────────────────

    #[test]
    fn hyperband_bracket_schedule() {
        // R = 27, eta = 3, min = 1 ⇒ s_max = 3, four brackets.
        let hb = Hyperband::new(HyperbandConfig::new(3.0, 1.0, 27.0));
        assert_eq!(hb.config().s_max(), 3);

        let counter = Cell::new(0usize);
        let factory = |_: &mut LcgRng| {
            let id = counter.get();
            counter.set(id + 1);
            id
        };
        let mut rng = LcgRng::new(1);
        let res = hb
            .run(factory, |c: &usize, b: f32| *c as f32 + 1e-4 * b, &mut rng)
            .expect("hyperband");

        // Brackets emitted from s = 3 down to s = 0.
        let s_vals: Vec<usize> = res.brackets.iter().map(|b| b.s).collect();
        assert_eq!(s_vals, vec![3, 2, 1, 0]);
        let n_configs: Vec<usize> = res.brackets.iter().map(|b| b.n_configs).collect();
        assert_eq!(n_configs, vec![27, 12, 6, 4]);
        let budgets: Vec<f32> = res.brackets.iter().map(|b| b.initial_budget).collect();
        assert_eq!(budgets, vec![1.0, 3.0, 9.0, 27.0]);
    }

    #[test]
    fn hyperband_best_is_global_max() {
        // Each generated candidate gets a unique increasing id; SHA always keeps
        // the highest-scoring (highest-id) survivor, so the global winner is the
        // last id generated = total_configs - 1.
        let hb = Hyperband::new(HyperbandConfig::new(3.0, 1.0, 27.0));
        let counter = Cell::new(0usize);
        let factory = |_: &mut LcgRng| {
            let id = counter.get();
            counter.set(id + 1);
            id
        };
        let mut rng = LcgRng::new(7);
        let res = hb
            .run(factory, |c: &usize, _b: f32| *c as f32, &mut rng)
            .expect("hyperband");
        let total: usize = res.brackets.iter().map(|b| b.n_configs).sum();
        assert_eq!(total, 49); // 27 + 12 + 6 + 4
        assert_eq!(res.best, total - 1);
        assert_eq!(res.best_score, (total - 1) as f32);
    }

    #[test]
    fn hyperband_deterministic_given_seed() {
        let hb = Hyperband::new(HyperbandConfig::new(2.0, 1.0, 8.0));
        let make = || {
            let counter = Cell::new(0usize);
            move |_: &mut LcgRng| {
                let id = counter.get();
                counter.set(id + 1);
                id
            }
        };
        let mut rng_a = LcgRng::new(99);
        let mut rng_b = LcgRng::new(99);
        let ra = hb
            .run(make(), |c: &usize, _b: f32| *c as f32, &mut rng_a)
            .expect("a");
        let rb = hb
            .run(make(), |c: &usize, _b: f32| *c as f32, &mut rng_b)
            .expect("b");
        assert_eq!(ra.best, rb.best);
        assert_eq!(ra.best_score, rb.best_score);
        assert_eq!(ra.brackets, rb.brackets);
    }
}
