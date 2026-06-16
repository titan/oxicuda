//! Constraint handling methods for evolutionary algorithms.
//!
//! Implements canonical techniques from:
//!   * Deb (2000) — "An efficient constraint handling method for genetic algorithms"
//!   * Bean & Hadj-Alouane (1994) — adaptive penalty factors
//!   * Takahama & Sakai (2006) — ε-constrained method (εDE)
//!
//! # Methods
//! - **Deb's feasibility rules**: lexicographic tournament comparison that
//!   always prefers feasible individuals and, among infeasible ones, the one
//!   with the smallest constraint violation.
//! - **Static penalty**: `f_pen = f(x) + Σ coeff_i * v_i²`
//! - **Adaptive penalty** (Bean-Hadj-Alouane): penalty coefficient adapts
//!   each generation based on the feasible/infeasible ratio.
//! - **Epsilon-constraint** (Takahama-Sakai): relaxed feasibility threshold ε
//!   anneals toward 0 over `tc` generations.
//! - **Constrained tournament selection**: standard k-tournament using Deb's
//!   feasibility rules as the comparison predicate.

use crate::{EvolError, EvolResult, handle::LcgRng};

/// Type alias for a boxed constraint function `g: Rⁿ → R`.
pub type ConstraintFn = Box<dyn Fn(&[f64]) -> f64>;

// ─── Constraint representation ────────────────────────────────────────────────

/// The type of inequality or equality constraint.
///
/// Given a constraint function `g: Rⁿ → R` and a bound `b`, the three kinds
/// encode:
/// - `Leq`: g(x) ≤ b  (most common)
/// - `Geq`: g(x) ≥ b
/// - `Eq`:  g(x) = b
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    /// g(x) ≤ bound
    Leq,
    /// g(x) ≥ bound
    Geq,
    /// g(x) = bound
    Eq,
}

/// A single constraint: `g(x) ⋈ bound` where ⋈ is determined by `kind`.
#[derive(Debug, Clone, Copy)]
pub struct Constraint {
    /// Sense of the constraint.
    pub kind: ConstraintKind,
    /// Right-hand side bound.
    pub bound: f64,
}

impl Constraint {
    /// Construct a new constraint.
    #[must_use]
    pub fn new(kind: ConstraintKind, bound: f64) -> Self {
        Self { kind, bound }
    }

    /// Compute the violation amount for a raw function value `g`.
    ///
    /// Returns 0.0 if the constraint is satisfied, otherwise the (positive)
    /// amount of violation:
    /// - Leq: `max(0, g - bound)`
    /// - Geq: `max(0, bound - g)`
    /// - Eq:  `|g - bound|`
    #[inline]
    #[must_use]
    pub fn violation(&self, g: f64) -> f64 {
        match self.kind {
            ConstraintKind::Leq => (g - self.bound).max(0.0),
            ConstraintKind::Geq => (self.bound - g).max(0.0),
            ConstraintKind::Eq => (g - self.bound).abs(),
        }
    }

    /// True if constraint is satisfied (violation < `tol`).
    #[inline]
    #[must_use]
    pub fn is_satisfied(&self, g: f64, tol: f64) -> bool {
        self.violation(g) < tol
    }
}

/// The raw violation record for a single constraint.
///
/// Stores the actual function value `g` together with the `Constraint`
/// specification; the violation amount can be recomputed via
/// `constraint.violation(g)`.
#[derive(Debug, Clone, Copy)]
pub struct ConstraintViolation {
    /// Actual value of the constraint function g(x).
    pub g: f64,
    /// The constraint that was evaluated.
    pub constraint: Constraint,
}

impl ConstraintViolation {
    /// Compute the (non-negative) violation amount.
    #[inline]
    #[must_use]
    pub fn amount(&self) -> f64 {
        self.constraint.violation(self.g)
    }
}

// ─── Evaluation helpers ───────────────────────────────────────────────────────

/// Evaluate all constraints at point `x` and return a violation vector.
///
/// The `i`-th entry is `constraints[i].violation(constraint_fns[i](x))`.
/// A value of 0.0 means constraint `i` is satisfied.
///
/// # Errors
/// Returns `EvolError::DimensionMismatch` if `constraint_fns.len() != constraints.len()`.
pub fn evaluate_constraints(
    x: &[f64],
    constraint_fns: &[ConstraintFn],
    constraints: &[Constraint],
) -> EvolResult<Vec<f64>> {
    if constraint_fns.len() != constraints.len() {
        return Err(EvolError::DimensionMismatch {
            expected: constraints.len(),
            got: constraint_fns.len(),
        });
    }
    let violations: Vec<f64> = constraint_fns
        .iter()
        .zip(constraints.iter())
        .map(|(f, c)| c.violation(f(x)))
        .collect();
    Ok(violations)
}

/// Sum all violation amounts.
///
/// Individual violations are non-negative, so the sum is the total
/// infeasibility measure (0 iff all constraints are satisfied).
#[inline]
#[must_use]
pub fn total_violation(violations: &[f64]) -> f64 {
    violations.iter().sum()
}

// ─── Deb's feasibility rules ──────────────────────────────────────────────────

/// Return `true` if individual A is strictly better than individual B according
/// to Deb's (2000) feasibility-based tournament rules:
///
/// 1. Both feasible (`viol == 0`): compare objective values — smaller is better.
/// 2. A feasible, B infeasible: A wins unconditionally.
/// 3. Both infeasible: smaller total violation wins.
///
/// "Feasible" means `total_violation == 0`.
#[inline]
#[must_use]
pub fn deb_feasible_better(viol_a: f64, obj_a: f64, viol_b: f64, obj_b: f64) -> bool {
    let a_feas = viol_a <= 0.0;
    let b_feas = viol_b <= 0.0;
    match (a_feas, b_feas) {
        (true, true) => obj_a < obj_b,
        (true, false) => true,
        (false, true) => false,
        (false, false) => viol_a < viol_b,
    }
}

// ─── Static penalty ───────────────────────────────────────────────────────────

/// Apply a static quadratic penalty to the objective:
///
/// `f_pen = obj + Σ_i penalty_coefficients[i] * violations[i]²`
///
/// # Errors
/// Returns `EvolError::DimensionMismatch` if the slice lengths differ.
pub fn static_penalty(
    obj: f64,
    violations: &[f64],
    penalty_coefficients: &[f64],
) -> EvolResult<f64> {
    if violations.len() != penalty_coefficients.len() {
        return Err(EvolError::DimensionMismatch {
            expected: penalty_coefficients.len(),
            got: violations.len(),
        });
    }
    let penalty: f64 = violations
        .iter()
        .zip(penalty_coefficients.iter())
        .map(|(&v, &c)| c * v * v)
        .sum();
    Ok(obj + penalty)
}

// ─── Adaptive penalty (Bean & Hadj-Alouane 1994) ─────────────────────────────

/// State for the adaptive penalty method.
///
/// The penalty coefficient `k` adapts each generation based on whether too
/// many or too few feasible individuals were found, targeting a 50 % feasible
/// rate as the balance criterion.
#[derive(Debug, Clone)]
pub struct AdaptivePenaltyState {
    /// Current penalty coefficient (multiplies `total_violation²`).
    pub penalty: f64,
    /// Best (lowest) objective among feasible individuals in this generation.
    pub best_feasible: Option<f64>,
    /// Best (lowest) objective + penalty among infeasible individuals.
    pub best_infeasible: Option<f64>,
    /// Current generation counter.
    pub generation: usize,
}

impl AdaptivePenaltyState {
    /// Construct an initial adaptive penalty state.
    #[must_use]
    pub fn new(initial_penalty: f64) -> Self {
        Self {
            penalty: initial_penalty.max(1e-6),
            best_feasible: None,
            best_infeasible: None,
            generation: 0,
        }
    }
}

/// Update the adaptive penalty coefficient after observing the current
/// generation's feasible/infeasible counts.
///
/// Rule (Bean & Hadj-Alouane):
/// - If `feasibles > infeasibles` (too many feasible): *decrease* penalty by
///   a factor to push exploration back into the infeasible region.
/// - If `feasibles < infeasibles` (too many infeasible): *increase* penalty to
///   steer the population toward feasibility.
/// - If equal: leave unchanged.
///
/// Factor: `β = 1 + 0.5 * |ratio - 0.5|` where `ratio = feasibles / total`.
pub fn update_penalty(state: &mut AdaptivePenaltyState, feasibles: usize, infeasibles: usize) {
    let total = feasibles + infeasibles;
    if total == 0 {
        state.generation += 1;
        return;
    }
    let ratio = feasibles as f64 / total as f64;
    // Magnitude of deviation from the 50 % target.
    let dev = (ratio - 0.5).abs();
    let factor = 1.0 + 0.5 * dev;

    if feasibles > infeasibles {
        // Too easy to be feasible: reduce penalty.
        state.penalty /= factor;
    } else if infeasibles > feasibles {
        // Too hard to be feasible: increase penalty.
        state.penalty *= factor;
    }
    // Clamp to a reasonable range to prevent numerical blow-up.
    state.penalty = state.penalty.clamp(1e-9, 1e12);
    state.generation += 1;
}

/// Compute the adaptive penalised objective for a single individual.
///
/// `f_pen = obj + penalty * (total_violation)²`
#[inline]
#[must_use]
pub fn adaptive_penalty_obj(state: &AdaptivePenaltyState, obj: f64, total_viol: f64) -> f64 {
    obj + state.penalty * total_viol * total_viol
}

// ─── Epsilon-constraint method (Takahama & Sakai 2006) ────────────────────────

/// State for the ε-constrained method.
///
/// The relaxed tolerance ε starts at `epsilon0` and anneals to 0 over `tc`
/// generations following a power schedule: `ε(t) = ε0 * (1 - t/tc)^cp`.
#[derive(Debug, Clone)]
pub struct EpsilonState {
    /// Current ε threshold.
    pub epsilon: f64,
    /// Initial ε (maximum relaxation at generation 0).
    pub epsilon0: f64,
    /// Generation at which ε reaches 0 (control parameter).
    pub tc: usize,
    /// Power exponent controlling the annealing curve (typically 5–20).
    pub cp: f64,
}

impl EpsilonState {
    /// Construct a new ε-state.
    ///
    /// # Panics (debug)
    /// Asserts that `epsilon0 >= 0`, `tc >= 1`, and `cp > 0`.
    #[must_use]
    pub fn new(epsilon0: f64, tc: usize, cp: f64) -> Self {
        debug_assert!(epsilon0 >= 0.0, "epsilon0 must be >= 0");
        debug_assert!(tc >= 1, "tc must be >= 1");
        debug_assert!(cp > 0.0, "cp must be > 0");
        Self {
            epsilon: epsilon0,
            epsilon0,
            tc,
            cp,
        }
    }
}

/// Return true if a solution with total violation `violation` is considered
/// "feasible" under the current ε relaxation, i.e. `violation <= epsilon`.
#[inline]
#[must_use]
pub fn epsilon_feasible(violation: f64, eps_state: &EpsilonState) -> bool {
    violation <= eps_state.epsilon
}

/// Update ε to generation `gen` using the annealing schedule:
/// ```text
/// ε(gen) = ε₀ · (1 - min(gen, tc) / tc)^cp
/// ```
/// Once `gen >= tc`, ε is set to 0 (strictly infeasible tolerance).
pub fn update_epsilon(state: &mut EpsilonState, generation: usize) {
    if generation >= state.tc {
        state.epsilon = 0.0;
    } else {
        let frac = 1.0 - generation as f64 / state.tc as f64;
        state.epsilon = state.epsilon0 * frac.powf(state.cp);
    }
}

// ─── Constrained tournament selection ────────────────────────────────────────

/// k-tournament selection using Deb's feasibility rules.
///
/// Samples `tournament_size` individuals at random (with replacement) from
/// `population` and returns the index of the "best" one as determined by
/// `deb_feasible_better`.
///
/// # Errors
/// - `EvolError::EmptyPopulation` if `population` is empty.
/// - `EvolError::InvalidParameter` if `tournament_size == 0`.
/// - `EvolError::DimensionMismatch` if `objectives` or `violations` lengths
///   differ from `population` length.
pub fn constrained_tournament_select(
    population: &[Vec<f64>],
    objectives: &[f64],
    violations: &[f64],
    tournament_size: usize,
    rng: &mut LcgRng,
) -> EvolResult<usize> {
    let n = population.len();
    if n == 0 {
        return Err(EvolError::EmptyPopulation);
    }
    if tournament_size == 0 {
        return Err(EvolError::InvalidParameter(
            "tournament_size must be >= 1".to_owned(),
        ));
    }
    if objectives.len() != n {
        return Err(EvolError::DimensionMismatch {
            expected: n,
            got: objectives.len(),
        });
    }
    if violations.len() != n {
        return Err(EvolError::DimensionMismatch {
            expected: n,
            got: violations.len(),
        });
    }

    let first = rng.next_usize(n);
    let mut best_idx = first;
    let mut best_viol = violations[first];
    let mut best_obj = objectives[first];

    for _ in 1..tournament_size {
        let candidate = rng.next_usize(n);
        let cand_viol = violations[candidate];
        let cand_obj = objectives[candidate];
        if deb_feasible_better(cand_viol, cand_obj, best_viol, best_obj) {
            best_idx = candidate;
            best_viol = cand_viol;
            best_obj = cand_obj;
        }
    }
    Ok(best_idx)
}

// ─── Result type ──────────────────────────────────────────────────────────────

/// The result of a constrained optimisation run: solution vector, objective
/// value, per-constraint violation amounts, and feasibility flag.
#[derive(Debug, Clone)]
pub struct ConstraintResult {
    /// Decision variable vector.
    pub x: Vec<f64>,
    /// Objective value (without penalty).
    pub objective: f64,
    /// Per-constraint violation amounts (0 if satisfied).
    pub violations: Vec<f64>,
    /// True if all violations are (approximately) 0.
    pub is_feasible: bool,
}

impl ConstraintResult {
    /// Construct a result and compute `is_feasible` automatically.
    #[must_use]
    pub fn new(x: Vec<f64>, objective: f64, violations: Vec<f64>) -> Self {
        let is_feasible = violations.iter().all(|&v| v <= 0.0);
        Self {
            x,
            objective,
            violations,
            is_feasible,
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1: Deb's rule — feasible beats infeasible ──────────────────────────

    #[test]
    fn deb_feasible_beats_infeasible() {
        // A feasible (viol=0), B infeasible (viol=5)
        assert!(
            deb_feasible_better(0.0, 100.0, 5.0, 1.0),
            "feasible must beat infeasible regardless of objective"
        );
        // Reversed: B feasible, A infeasible
        assert!(
            !deb_feasible_better(5.0, 1.0, 0.0, 100.0),
            "infeasible must not beat feasible"
        );
    }

    // ── 2: Deb's rule — both infeasible → smaller violation wins ──────────

    #[test]
    fn deb_both_infeasible_min_violation_wins() {
        assert!(
            deb_feasible_better(2.0, 10.0, 5.0, 10.0),
            "smaller violation (2 < 5) must win"
        );
        assert!(
            !deb_feasible_better(5.0, 10.0, 2.0, 10.0),
            "larger violation (5 > 2) must lose"
        );
    }

    // ── 3: Deb's rule — both feasible → smaller objective wins ────────────

    #[test]
    fn deb_both_feasible_min_obj_wins() {
        assert!(
            deb_feasible_better(0.0, 3.0, 0.0, 7.0),
            "lower objective (3 < 7) must win when both feasible"
        );
        assert!(
            !deb_feasible_better(0.0, 7.0, 0.0, 3.0),
            "higher objective must not win"
        );
    }

    // ── 4: Constraint violation amounts are correct ────────────────────────

    #[test]
    fn constraint_violation_amounts() {
        let leq = Constraint::new(ConstraintKind::Leq, 5.0);
        assert_eq!(leq.violation(3.0), 0.0); // 3 ≤ 5, satisfied
        assert!((leq.violation(8.0) - 3.0).abs() < 1e-12); // 8 - 5 = 3

        let geq = Constraint::new(ConstraintKind::Geq, 5.0);
        assert_eq!(geq.violation(7.0), 0.0); // 7 ≥ 5, satisfied
        assert!((geq.violation(2.0) - 3.0).abs() < 1e-12); // 5 - 2 = 3

        let eq = Constraint::new(ConstraintKind::Eq, 5.0);
        assert_eq!(eq.violation(5.0), 0.0); // exact
        assert!((eq.violation(8.0) - 3.0).abs() < 1e-12); // |8-5|=3
        assert!((eq.violation(2.0) - 3.0).abs() < 1e-12); // |2-5|=3
    }

    // ── 5: static penalty adds positive amount for infeasible ─────────────

    #[test]
    fn static_penalty_infeasible_positive() {
        let obj = 10.0_f64;
        let violations = [3.0_f64, 2.0];
        let coeffs = [1.0_f64, 1.0];
        let penalised =
            static_penalty(obj, &violations, &coeffs).expect("static_penalty should succeed");
        // penalty = 1*9 + 1*4 = 13; total = 23
        assert!((penalised - 23.0).abs() < 1e-12);
        assert!(penalised > obj, "penalised must exceed raw objective");
    }

    // ── 6: static penalty = obj for feasible (zero violations) ────────────

    #[test]
    fn static_penalty_feasible_equals_obj() {
        let obj = 42.0_f64;
        let violations = [0.0_f64, 0.0];
        let coeffs = [100.0_f64, 100.0];
        let penalised =
            static_penalty(obj, &violations, &coeffs).expect("static_penalty should succeed");
        assert!((penalised - obj).abs() < 1e-12);
    }

    // ── 7: static_penalty dimension mismatch returns Err ─────────────────

    #[test]
    fn static_penalty_dim_mismatch_err() {
        let result = static_penalty(0.0, &[1.0, 2.0], &[1.0]);
        assert!(result.is_err());
    }

    // ── 8: adaptive penalty increases when too many infeasibles ───────────

    #[test]
    fn adaptive_penalty_increases_for_infeasibles() {
        let mut state = AdaptivePenaltyState::new(1.0);
        let init_penalty = state.penalty;
        // Many infeasibles (80 infeasible, 20 feasible)
        update_penalty(&mut state, 20, 80);
        assert!(
            state.penalty > init_penalty,
            "penalty must increase when infeasibles dominate: before={init_penalty}, after={}",
            state.penalty
        );
    }

    // ── 9: adaptive penalty decreases when too many feasibles ─────────────

    #[test]
    fn adaptive_penalty_decreases_for_feasibles() {
        let mut state = AdaptivePenaltyState::new(10.0);
        let init_penalty = state.penalty;
        // Many feasibles (80 feasible, 20 infeasible)
        update_penalty(&mut state, 80, 20);
        assert!(
            state.penalty < init_penalty,
            "penalty must decrease when feasibles dominate: before={init_penalty}, after={}",
            state.penalty
        );
    }

    // ── 10: adaptive penalty unchanged when equal counts ──────────────────

    #[test]
    fn adaptive_penalty_unchanged_when_equal() {
        let mut state = AdaptivePenaltyState::new(5.0);
        let init_penalty = state.penalty;
        update_penalty(&mut state, 50, 50);
        assert!(
            (state.penalty - init_penalty).abs() < 1e-12,
            "penalty must not change when counts are equal"
        );
    }

    // ── 11: epsilon constraint anneals to 0 ───────────────────────────────

    #[test]
    fn epsilon_anneals_to_zero() {
        let tc = 100_usize;
        let mut eps = EpsilonState::new(1.0, tc, 5.0);
        assert!(
            (eps.epsilon - 1.0).abs() < 1e-12,
            "initial epsilon must be epsilon0"
        );

        // Partway through.
        update_epsilon(&mut eps, 50);
        assert!(
            eps.epsilon > 0.0 && eps.epsilon < 1.0,
            "epsilon must be in (0, 1) at gen=50/100: {}",
            eps.epsilon
        );

        // At tc, epsilon is exactly 0.
        update_epsilon(&mut eps, tc);
        assert_eq!(eps.epsilon, 0.0, "epsilon must be 0 at generation tc");
    }

    // ── 12: epsilon_feasible uses current epsilon threshold ───────────────

    #[test]
    fn epsilon_feasible_threshold() {
        let eps = EpsilonState::new(2.0, 100, 5.0);
        assert!(epsilon_feasible(1.5, &eps), "1.5 ≤ 2.0 must be feasible");
        assert!(
            epsilon_feasible(2.0, &eps),
            "2.0 ≤ 2.0 must be feasible (boundary)"
        );
        assert!(!epsilon_feasible(2.5, &eps), "2.5 > 2.0 must be infeasible");
    }

    // ── 13: constrained tournament selects feasible over infeasible ────────

    #[test]
    fn tournament_selects_feasible() {
        let mut rng = LcgRng::new(42);
        // Population with ONE feasible individual that has the worst raw objective.
        // Deb's rules must still select it when it appears in the tournament.
        // We use tournament_size == population_size so all 3 are always sampled.
        let population = vec![
            vec![1.0_f64], // index 0 — infeasible, large violation
            vec![2.0_f64], // index 1 — feasible (violation 0)
            vec![3.0_f64], // index 2 — infeasible, smaller violation
        ];
        let objectives = [100.0_f64, 200.0, 300.0]; // feasible has worst raw obj
        let violations = [5.0_f64, 0.0, 3.0]; // only index 1 is feasible

        // Direct call to deb_feasible_better verifies the rule is correct.
        assert!(
            deb_feasible_better(violations[1], objectives[1], violations[0], objectives[0]),
            "feasible (idx 1) must beat infeasible (idx 0)"
        );
        assert!(
            deb_feasible_better(violations[1], objectives[1], violations[2], objectives[2]),
            "feasible (idx 1) must beat infeasible (idx 2)"
        );

        // Run many tournaments — feasible must win a majority when n=3.
        // (with replacement sampling, feasible appears ~33% as first pick,
        //  but always wins when included due to Deb's rules.)
        let mut wins = 0_usize;
        for _ in 0..200 {
            let winner =
                constrained_tournament_select(&population, &objectives, &violations, 3, &mut rng)
                    .expect("value should be present");
            if winner == 1 {
                wins += 1;
            }
        }
        // With 3 candidates and replacement, Prob(index 1 not sampled at all) =
        // (2/3)^3 ≈ 0.296. So feasible wins ≈ 70% of the time.
        assert!(
            wins > 100,
            "feasible individual must win majority of tournaments, got {wins}/200"
        );
    }

    // ── 14: tournament returns Err on empty population ─────────────────────

    #[test]
    fn tournament_empty_population_err() {
        let mut rng = LcgRng::new(0);
        let result = constrained_tournament_select(&[], &[], &[], 3, &mut rng);
        assert!(result.is_err());
    }

    // ── 15: evaluate_constraints length mismatch returns Err ──────────────

    #[test]
    fn evaluate_constraints_len_mismatch_err() {
        type ConstraintFn = Box<dyn Fn(&[f64]) -> f64>;
        let fns: Vec<ConstraintFn> = vec![Box::new(|x: &[f64]| x[0])];
        let constraints = [
            Constraint::new(ConstraintKind::Leq, 1.0),
            Constraint::new(ConstraintKind::Leq, 2.0),
        ]; // 2 constraints but only 1 function
        let result = evaluate_constraints(&[1.0], &fns, &constraints);
        assert!(result.is_err());
    }

    // ── 16: total_violation sums correctly ────────────────────────────────

    #[test]
    fn total_violation_sum() {
        let violations = [0.0, 2.5, 0.0, 1.5];
        assert!((total_violation(&violations) - 4.0).abs() < 1e-12);
    }

    // ── 17: ConstraintResult sets is_feasible correctly ───────────────────

    #[test]
    fn constraint_result_is_feasible() {
        let r1 = ConstraintResult::new(vec![1.0], 5.0, vec![0.0, 0.0]);
        assert!(r1.is_feasible);
        let r2 = ConstraintResult::new(vec![1.0], 5.0, vec![0.0, 0.1]);
        assert!(!r2.is_feasible);
    }

    // ── 18: adaptive penalty generation counter increments ────────────────

    #[test]
    fn adaptive_penalty_gen_increments() {
        let mut state = AdaptivePenaltyState::new(1.0);
        assert_eq!(state.generation, 0);
        update_penalty(&mut state, 50, 50);
        assert_eq!(state.generation, 1);
        update_penalty(&mut state, 50, 50);
        assert_eq!(state.generation, 2);
    }
}
