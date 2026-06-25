//! Symbolic regression via tree-based genetic programming.
//!
//! Koza (1992) "Genetic Programming". Evolves a population of mathematical
//! expression trees to fit a scalar function `y = f(x)` of a single input
//! variable. The search proceeds by:
//!
//! 1. **Initialisation** — a ramped half-and-half population of random trees
//!    (a mix of `grow` and `full` trees over a range of depths).
//! 2. **Evaluation** — each tree's *fitness* is the mean-squared error between
//!    its predictions and the target on a set of sample points (lower is fitter),
//!    with a small parsimony penalty on node count to discourage bloat.
//! 3. **Selection** — *tournament selection*: pick `k` random individuals and
//!    keep the fittest.
//! 4. **Variation** — *subtree crossover* (swap a random subtree between two
//!    parents) and *point mutation* (replace a random subtree with a fresh random
//!    tree), with configurable probabilities.
//! 5. **Elitism** — the single best individual is copied unchanged into the next
//!    generation so the best-so-far never regresses.
//!
//! All stochastic choices are drawn from the crate's deterministic [`LcgRng`], so
//! a fixed seed reproduces the entire run bit-for-bit. The operator set is
//! `+ − × ÷` (protected division) plus the unary primitives `sin`, `cos`, `exp`;
//! terminals are the input variable `x` and *ephemeral random constants* sampled
//! once at node-creation time.
//!
//! ## Example
//! ```
//! use oxicuda_pinn::handle::LcgRng;
//! use oxicuda_pinn::symbolic::regression::{SymbolicRegressor, SymbolicConfig};
//!
//! // Target: f(x) = x·x + 1 on [-2, 2].
//! let xs: Vec<f32> = (0..21).map(|i| -2.0 + i as f32 * 0.2).collect();
//! let ys: Vec<f32> = xs.iter().map(|&x| x * x + 1.0).collect();
//!
//! let cfg = SymbolicConfig::new();
//! let mut rng = LcgRng::new(7);
//! let mut reg = SymbolicRegressor::new(cfg);
//! let best = reg.fit(&xs, &ys, &mut rng).expect("fit should succeed");
//! assert!(best.fitness.is_finite());
//! ```

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;

/// A node in a symbolic-regression expression tree.
///
/// Leaves are [`Expr::Var`] (the single input `x`) and [`Expr::Const`]; internal
/// nodes are binary (`+ − × ÷`) or unary (`sin`, `cos`, `exp`).
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// The input variable `x`.
    Var,
    /// A numeric (ephemeral) constant.
    Const(f32),
    /// `a + b`.
    Add(Box<Expr>, Box<Expr>),
    /// `a − b`.
    Sub(Box<Expr>, Box<Expr>),
    /// `a × b`.
    Mul(Box<Expr>, Box<Expr>),
    /// `a ÷ b` (protected: division by a near-zero denominator returns `1.0`).
    Div(Box<Expr>, Box<Expr>),
    /// `sin(a)`.
    Sin(Box<Expr>),
    /// `cos(a)`.
    Cos(Box<Expr>),
    /// `exp(a)` (argument clamped to avoid overflow).
    Exp(Box<Expr>),
}

impl Expr {
    /// Evaluate the expression at a given value of the input variable `x`.
    ///
    /// Uses *protected* division (denominator magnitude below `1e-6` yields
    /// `1.0`) and a clamped exponent (`exp` argument capped to `±30`) so that
    /// evaluation never produces `inf`/`NaN` from the primitive operations
    /// themselves. The result is still checked by callers via fitness finiteness.
    #[must_use]
    pub fn eval(&self, x: f32) -> f32 {
        match self {
            Expr::Var => x,
            Expr::Const(c) => *c,
            Expr::Add(a, b) => a.eval(x) + b.eval(x),
            Expr::Sub(a, b) => a.eval(x) - b.eval(x),
            Expr::Mul(a, b) => a.eval(x) * b.eval(x),
            Expr::Div(a, b) => {
                let denom = b.eval(x);
                if denom.abs() < 1e-6 {
                    1.0
                } else {
                    a.eval(x) / denom
                }
            }
            Expr::Sin(a) => a.eval(x).sin(),
            Expr::Cos(a) => a.eval(x).cos(),
            Expr::Exp(a) => a.eval(x).clamp(-30.0, 30.0).exp(),
        }
    }

    /// Total number of nodes in the tree (its *size*), used for parsimony.
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            Expr::Var | Expr::Const(_) => 1,
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
                1 + a.size() + b.size()
            }
            Expr::Sin(a) | Expr::Cos(a) | Expr::Exp(a) => 1 + a.size(),
        }
    }

    /// Maximum depth of the tree (a single leaf has depth `1`).
    #[must_use]
    pub fn depth(&self) -> usize {
        match self {
            Expr::Var | Expr::Const(_) => 1,
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
                1 + a.depth().max(b.depth())
            }
            Expr::Sin(a) | Expr::Cos(a) | Expr::Exp(a) => 1 + a.depth(),
        }
    }

    /// Render the expression as an infix string (mostly for debugging / display).
    #[must_use]
    pub fn to_infix(&self) -> String {
        match self {
            Expr::Var => "x".to_string(),
            Expr::Const(c) => format!("{c:.4}"),
            Expr::Add(a, b) => format!("({} + {})", a.to_infix(), b.to_infix()),
            Expr::Sub(a, b) => format!("({} - {})", a.to_infix(), b.to_infix()),
            Expr::Mul(a, b) => format!("({} * {})", a.to_infix(), b.to_infix()),
            Expr::Div(a, b) => format!("({} / {})", a.to_infix(), b.to_infix()),
            Expr::Sin(a) => format!("sin({})", a.to_infix()),
            Expr::Cos(a) => format!("cos({})", a.to_infix()),
            Expr::Exp(a) => format!("exp({})", a.to_infix()),
        }
    }

    /// Immutable reference to the `idx`-th node in a fixed pre-order traversal.
    ///
    /// Pre-order visits the node, then its children left-to-right. Returns `None`
    /// when `idx` is past the end of the traversal.
    fn node_at(&self, idx: usize) -> Option<&Expr> {
        fn walk<'a>(node: &'a Expr, target: usize, counter: &mut usize) -> Option<&'a Expr> {
            if *counter == target {
                return Some(node);
            }
            *counter += 1;
            match node {
                Expr::Var | Expr::Const(_) => None,
                Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
                    walk(a, target, counter).or_else(|| walk(b, target, counter))
                }
                Expr::Sin(a) | Expr::Cos(a) | Expr::Exp(a) => walk(a, target, counter),
            }
        }
        let mut counter = 0;
        walk(self, idx, &mut counter)
    }

    /// Replace the subtree rooted at pre-order index `idx` with `replacement`.
    ///
    /// Returns `true` if a replacement was made. The traversal order matches
    /// [`Expr::node_at`].
    fn replace_at(&mut self, idx: usize, replacement: Expr) -> bool {
        fn walk(
            node: &mut Expr,
            target: usize,
            counter: &mut usize,
            repl: &mut Option<Expr>,
        ) -> bool {
            if *counter == target {
                if let Some(r) = repl.take() {
                    *node = r;
                }
                return true;
            }
            *counter += 1;
            match node {
                Expr::Var | Expr::Const(_) => false,
                Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
                    walk(a, target, counter, repl) || walk(b, target, counter, repl)
                }
                Expr::Sin(a) | Expr::Cos(a) | Expr::Exp(a) => walk(a, target, counter, repl),
            }
        }
        let mut repl = Some(replacement);
        let mut counter = 0;
        walk(self, idx, &mut counter, &mut repl)
    }
}

/// The binary operator primitives available to the search.
const N_BINARY: usize = 4;
/// The unary operator primitives available to the search.
const N_UNARY: usize = 3;

/// Configuration for [`SymbolicRegressor`].
#[derive(Debug, Clone)]
pub struct SymbolicConfig {
    /// Number of individuals in the population.
    pub population: usize,
    /// Number of generations to evolve.
    pub generations: usize,
    /// Tournament size for selection (`>= 1`).
    pub tournament_size: usize,
    /// Maximum tree depth used when generating / mutating individuals.
    pub max_depth: usize,
    /// Probability of subtree crossover when producing a child (`[0, 1]`).
    pub crossover_prob: f32,
    /// Probability of point mutation applied to a child (`[0, 1]`).
    pub mutation_prob: f32,
    /// Half-range of the uniform distribution for ephemeral random constants:
    /// constants are sampled in `[-const_range, const_range]`.
    pub const_range: f32,
    /// Parsimony coefficient: fitness is penalised by `parsimony · node_count`.
    pub parsimony: f32,
}

impl SymbolicConfig {
    /// Reasonable defaults for recovering simple low-order expressions.
    #[must_use]
    pub fn new() -> Self {
        Self {
            population: 300,
            generations: 40,
            tournament_size: 5,
            max_depth: 4,
            crossover_prob: 0.8,
            mutation_prob: 0.2,
            const_range: 3.0,
            parsimony: 1e-3,
        }
    }

    /// Validate the configuration.
    ///
    /// # Errors
    /// - [`PinnError::EmptyInput`] if `population == 0` or `generations == 0`.
    /// - [`PinnError::InvalidLayerWidth`] if `tournament_size == 0` or
    ///   `max_depth == 0`.
    /// - [`PinnError::InvalidWeight`] if either probability is outside `[0, 1]`.
    pub fn validate(&self) -> PinnResult<()> {
        if self.population == 0 || self.generations == 0 {
            return Err(PinnError::EmptyInput);
        }
        if self.tournament_size == 0 || self.max_depth == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        for w in [self.crossover_prob, self.mutation_prob] {
            if !(0.0..=1.0).contains(&w) {
                return Err(PinnError::InvalidWeight { weight: w });
            }
        }
        Ok(())
    }
}

impl Default for SymbolicConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// A scored individual: an expression tree together with its fitness.
#[derive(Debug, Clone)]
pub struct Individual {
    /// The expression tree.
    pub expr: Expr,
    /// Penalised mean-squared error (lower is better); `+inf` if non-finite.
    pub fitness: f32,
    /// Raw mean-squared error before the parsimony penalty.
    pub mse: f32,
}

/// Tree-based genetic-programming symbolic regressor.
pub struct SymbolicRegressor {
    config: SymbolicConfig,
}

impl SymbolicRegressor {
    /// Construct a regressor with the given configuration.
    #[must_use]
    pub fn new(config: SymbolicConfig) -> Self {
        Self { config }
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &SymbolicConfig {
        &self.config
    }

    /// Sample an ephemeral random constant in `[-const_range, const_range]`.
    fn random_const(&self, rng: &mut LcgRng) -> Expr {
        let c = (rng.next_f32() * 2.0 - 1.0) * self.config.const_range;
        Expr::Const(c)
    }

    /// Sample a random terminal (`x` with probability ~⅔, otherwise a constant).
    fn random_terminal(&self, rng: &mut LcgRng) -> Expr {
        if rng.next_usize(3) == 0 {
            self.random_const(rng)
        } else {
            Expr::Var
        }
    }

    /// Build a random function (internal) node whose children are produced by
    /// `child` (a closure that recurses with the appropriate depth budget).
    fn random_function<F: FnMut(&mut LcgRng) -> Expr>(
        &self,
        rng: &mut LcgRng,
        mut child: F,
    ) -> Expr {
        // Choose among binary and unary primitives uniformly.
        let total = N_BINARY + N_UNARY;
        let pick = rng.next_usize(total);
        if pick < N_BINARY {
            let a = Box::new(child(rng));
            let b = Box::new(child(rng));
            match pick {
                0 => Expr::Add(a, b),
                1 => Expr::Sub(a, b),
                2 => Expr::Mul(a, b),
                _ => Expr::Div(a, b),
            }
        } else {
            let a = Box::new(child(rng));
            match pick - N_BINARY {
                0 => Expr::Sin(a),
                1 => Expr::Cos(a),
                _ => Expr::Exp(a),
            }
        }
    }

    /// Generate a random tree using the *grow* method up to `max_depth`.
    ///
    /// At each node below the maximum depth, a terminal or a function is chosen
    /// at random; at the maximum depth only terminals are produced.
    fn gen_grow(&self, rng: &mut LcgRng, max_depth: usize) -> Expr {
        if max_depth <= 1 {
            return self.random_terminal(rng);
        }
        // ~30% chance to stop early at a terminal (the "grow" irregularity).
        if rng.next_usize(10) < 3 {
            return self.random_terminal(rng);
        }
        self.random_function(rng, |r| self.gen_grow(r, max_depth - 1))
    }

    /// Generate a random tree using the *full* method: every path reaches
    /// `max_depth` (internal nodes everywhere until the leaves).
    fn gen_full(&self, rng: &mut LcgRng, max_depth: usize) -> Expr {
        if max_depth <= 1 {
            return self.random_terminal(rng);
        }
        self.random_function(rng, |r| self.gen_full(r, max_depth - 1))
    }

    /// Ramped half-and-half: alternate `grow`/`full` over depths `2..=max_depth`.
    fn gen_individual(&self, rng: &mut LcgRng, index: usize) -> Expr {
        let span = (self.config.max_depth.max(2)) - 1; // depths 2..=max_depth
        let depth = 2 + (index % span);
        if index % 2 == 0 {
            self.gen_full(rng, depth)
        } else {
            self.gen_grow(rng, depth)
        }
    }

    /// Mean-squared error of `expr` over the sample points.
    ///
    /// Returns `+inf` if any prediction is non-finite (so such trees are never
    /// selected). The inputs are assumed validated by the caller.
    fn mean_squared_error(expr: &Expr, xs: &[f32], ys: &[f32]) -> f32 {
        let mut acc = 0.0_f32;
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            let pred = expr.eval(x);
            if !pred.is_finite() {
                return f32::INFINITY;
            }
            let e = pred - y;
            acc += e * e;
        }
        acc / xs.len() as f32
    }

    /// Score an expression: penalised fitness and raw MSE.
    fn score(&self, expr: &Expr, xs: &[f32], ys: &[f32]) -> Individual {
        let mse = Self::mean_squared_error(expr, xs, ys);
        let fitness = if mse.is_finite() {
            mse + self.config.parsimony * expr.size() as f32
        } else {
            f32::INFINITY
        };
        Individual {
            expr: expr.clone(),
            fitness,
            mse,
        }
    }

    /// Tournament selection: return the index of the fittest of `k` random
    /// individuals drawn (with replacement) from `pop`.
    fn tournament(&self, pop: &[Individual], rng: &mut LcgRng) -> usize {
        let k = self.config.tournament_size.min(pop.len()).max(1);
        let mut best = rng.next_usize(pop.len());
        let mut best_fit = pop[best].fitness;
        for _ in 1..k {
            let cand = rng.next_usize(pop.len());
            if pop[cand].fitness < best_fit {
                best = cand;
                best_fit = pop[cand].fitness;
            }
        }
        best
    }

    /// Subtree crossover: return a child that is `parent_a` with one random
    /// subtree replaced by a random subtree taken from `parent_b`.
    fn crossover(&self, parent_a: &Expr, parent_b: &Expr, rng: &mut LcgRng) -> Expr {
        let mut child = parent_a.clone();
        let a_size = child.size();
        let b_size = parent_b.size();
        let a_idx = rng.next_usize(a_size);
        let b_idx = rng.next_usize(b_size);
        let donor = parent_b
            .node_at(b_idx)
            .cloned()
            .unwrap_or_else(|| parent_b.clone());
        child.replace_at(a_idx, donor);
        child
    }

    /// Point mutation: replace a random subtree of `expr` with a freshly grown
    /// random tree of bounded depth.
    fn mutate(&self, expr: &Expr, rng: &mut LcgRng) -> Expr {
        let mut child = expr.clone();
        let size = child.size();
        let idx = rng.next_usize(size);
        // Bound the replacement depth so mutation does not explode tree size.
        let new_depth = 1 + rng.next_usize(self.config.max_depth.max(1));
        let replacement = self.gen_grow(rng, new_depth);
        child.replace_at(idx, replacement);
        child
    }

    /// Build and evolve a population to minimise the MSE of `expr(x)` against the
    /// targets `ys` at the sample points `xs`. Returns the fittest individual
    /// found across all generations (including the initial population).
    ///
    /// # Errors
    /// - [`PinnError::EmptyInput`] if `xs` is empty.
    /// - [`PinnError::DimensionMismatch`] if `xs.len() != ys.len()`.
    /// - Configuration errors from [`SymbolicConfig::validate`].
    /// - [`PinnError::SolverDivergence`] if no finite-fitness individual is ever
    ///   produced (should not happen with the protected primitives).
    pub fn fit(&mut self, xs: &[f32], ys: &[f32], rng: &mut LcgRng) -> PinnResult<Individual> {
        self.config.validate()?;
        if xs.is_empty() {
            return Err(PinnError::EmptyInput);
        }
        if xs.len() != ys.len() {
            return Err(PinnError::DimensionMismatch {
                expected: xs.len(),
                got: ys.len(),
            });
        }

        // Initial population (ramped half-and-half), scored.
        let mut pop: Vec<Individual> = (0..self.config.population)
            .map(|i| {
                let e = self.gen_individual(rng, i);
                self.score(&e, xs, ys)
            })
            .collect();

        let mut best = best_of(&pop).clone();

        for _gen in 0..self.config.generations {
            let mut next: Vec<Individual> = Vec::with_capacity(pop.len());

            // Elitism: carry the current best forward unchanged.
            next.push(best.clone());

            while next.len() < self.config.population {
                // Parent A by tournament.
                let pa = self.tournament(&pop, rng);
                // Crossover with parent B (probabilistic), else clone A.
                let mut child = if rng.next_f32() < self.config.crossover_prob {
                    let pb = self.tournament(&pop, rng);
                    self.crossover(&pop[pa].expr, &pop[pb].expr, rng)
                } else {
                    pop[pa].expr.clone()
                };
                // Point mutation (probabilistic).
                if rng.next_f32() < self.config.mutation_prob {
                    child = self.mutate(&child, rng);
                }
                // Reject pathologically deep trees (keep search well-conditioned).
                if child.depth() > self.config.max_depth * 3 {
                    child = self.gen_grow(rng, self.config.max_depth);
                }
                let scored = self.score(&child, xs, ys);
                next.push(scored);
            }

            pop = next;
            let gen_best = best_of(&pop);
            if gen_best.fitness < best.fitness {
                best = gen_best.clone();
            }
        }

        if !best.fitness.is_finite() {
            return Err(PinnError::SolverDivergence {
                reason: "symbolic regression produced no finite-fitness individual",
            });
        }
        Ok(best)
    }
}

/// Return the fittest individual in a non-empty population.
fn best_of(pop: &[Individual]) -> &Individual {
    pop.iter()
        .min_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Greater)
        })
        .unwrap_or(&pop[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expr_eval_basic_ops() {
        let e = Expr::Add(Box::new(Expr::Var), Box::new(Expr::Const(2.0)));
        assert!((e.eval(3.0) - 5.0).abs() < 1e-6);
        let m = Expr::Mul(Box::new(Expr::Var), Box::new(Expr::Var));
        assert!((m.eval(4.0) - 16.0).abs() < 1e-6);
    }

    #[test]
    fn expr_protected_division() {
        // Division by ~0 returns 1.0 (no inf/NaN).
        let e = Expr::Div(Box::new(Expr::Const(5.0)), Box::new(Expr::Const(0.0)));
        assert_eq!(e.eval(0.0), 1.0);
        let normal = Expr::Div(Box::new(Expr::Const(6.0)), Box::new(Expr::Const(2.0)));
        assert!((normal.eval(0.0) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn expr_exp_clamped_finite() {
        let e = Expr::Exp(Box::new(Expr::Const(1000.0)));
        assert!(e.eval(0.0).is_finite());
    }

    #[test]
    fn expr_size_and_depth() {
        // sin(x + 1): nodes = Sin, Add, Var, Const = 4; depth = 3.
        let e = Expr::Sin(Box::new(Expr::Add(
            Box::new(Expr::Var),
            Box::new(Expr::Const(1.0)),
        )));
        assert_eq!(e.size(), 4);
        assert_eq!(e.depth(), 3);
    }

    #[test]
    fn expr_node_at_preorder() {
        // Pre-order of Add(Var, Const): [Add, Var, Const].
        let e = Expr::Add(Box::new(Expr::Var), Box::new(Expr::Const(7.0)));
        assert!(matches!(e.node_at(0), Some(Expr::Add(..))));
        assert!(matches!(e.node_at(1), Some(Expr::Var)));
        assert!(matches!(e.node_at(2), Some(Expr::Const(_))));
        assert!(e.node_at(3).is_none());
    }

    #[test]
    fn expr_replace_at_swaps_subtree() {
        let mut e = Expr::Add(Box::new(Expr::Var), Box::new(Expr::Const(7.0)));
        // Replace index 1 (the Var) with Const(2).
        assert!(e.replace_at(1, Expr::Const(2.0)));
        // Now (2 + 7) = 9 for any x.
        assert!((e.eval(123.0) - 9.0).abs() < 1e-6);
    }

    #[test]
    fn config_validation_rejects_bad() {
        let mut cfg = SymbolicConfig::new();
        cfg.population = 0;
        assert!(matches!(cfg.validate(), Err(PinnError::EmptyInput)));

        let mut cfg = SymbolicConfig::new();
        cfg.crossover_prob = 1.5;
        assert!(matches!(
            cfg.validate(),
            Err(PinnError::InvalidWeight { .. })
        ));

        let mut cfg = SymbolicConfig::new();
        cfg.tournament_size = 0;
        assert!(matches!(cfg.validate(), Err(PinnError::InvalidLayerWidth)));
    }

    #[test]
    fn fit_empty_input_errors() {
        let cfg = SymbolicConfig::new();
        let mut rng = LcgRng::new(1);
        let mut reg = SymbolicRegressor::new(cfg);
        assert!(matches!(
            reg.fit(&[], &[], &mut rng),
            Err(PinnError::EmptyInput)
        ));
    }

    #[test]
    fn fit_length_mismatch_errors() {
        let cfg = SymbolicConfig::new();
        let mut rng = LcgRng::new(1);
        let mut reg = SymbolicRegressor::new(cfg);
        assert!(matches!(
            reg.fit(&[0.0, 1.0], &[0.0], &mut rng),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn fit_is_deterministic_given_seed() {
        let xs: Vec<f32> = (0..11).map(|i| -1.0 + i as f32 * 0.2).collect();
        let ys: Vec<f32> = xs.iter().map(|&x| 2.0 * x).collect();
        let mut cfg = SymbolicConfig::new();
        cfg.population = 60;
        cfg.generations = 8;

        let run = |seed: u64| -> Individual {
            let mut rng = LcgRng::new(seed);
            let mut reg = SymbolicRegressor::new(cfg.clone());
            reg.fit(&xs, &ys, &mut rng)
                .expect("symbolic regression fit should succeed")
        };
        let a = run(123);
        let b = run(123);
        assert_eq!(a.expr, b.expr, "same seed must give the same best tree");
        assert!((a.fitness - b.fitness).abs() < 1e-9);
    }

    #[test]
    fn fit_recovers_linear_target() {
        // Target: f(x) = 2x on [-1, 1].
        let xs: Vec<f32> = (0..21).map(|i| -1.0 + i as f32 * 0.1).collect();
        let ys: Vec<f32> = xs.iter().map(|&x| 2.0 * x).collect();
        let signal_var = {
            let mean = ys.iter().sum::<f32>() / ys.len() as f32;
            ys.iter().map(|&y| (y - mean) * (y - mean)).sum::<f32>() / ys.len() as f32
        };
        let mut cfg = SymbolicConfig::new();
        cfg.population = 400;
        cfg.generations = 60;
        let mut rng = LcgRng::new(2025);
        let mut reg = SymbolicRegressor::new(cfg);
        let best = reg
            .fit(&xs, &ys, &mut rng)
            .expect("symbolic regression fit should succeed");
        assert!(
            best.mse < 0.05 * signal_var,
            "recovered MSE {} should be well below signal variance {}: best = {}",
            best.mse,
            signal_var,
            best.expr.to_infix()
        );
    }

    #[test]
    fn fit_recovers_quadratic_target() {
        // Target: f(x) = x·x + 1 on [-2, 2].
        let xs: Vec<f32> = (0..21).map(|i| -2.0 + i as f32 * 0.2).collect();
        let ys: Vec<f32> = xs.iter().map(|&x| x * x + 1.0).collect();
        let signal_var = {
            let mean = ys.iter().sum::<f32>() / ys.len() as f32;
            ys.iter().map(|&y| (y - mean) * (y - mean)).sum::<f32>() / ys.len() as f32
        };
        let mut cfg = SymbolicConfig::new();
        cfg.population = 500;
        cfg.generations = 80;
        let mut rng = LcgRng::new(99);
        let mut reg = SymbolicRegressor::new(cfg);
        let best = reg
            .fit(&xs, &ys, &mut rng)
            .expect("symbolic regression fit should succeed");
        assert!(
            best.mse < 0.05 * signal_var,
            "recovered MSE {} should be well below signal variance {}: best = {}",
            best.mse,
            signal_var,
            best.expr.to_infix()
        );
    }

    #[test]
    fn fit_best_fitness_is_finite() {
        let xs: Vec<f32> = (0..11).map(|i| i as f32 * 0.1).collect();
        let ys: Vec<f32> = xs.iter().map(|&x| x.sin()).collect();
        let mut cfg = SymbolicConfig::new();
        cfg.population = 80;
        cfg.generations = 10;
        let mut rng = LcgRng::new(7);
        let mut reg = SymbolicRegressor::new(cfg);
        let best = reg
            .fit(&xs, &ys, &mut rng)
            .expect("symbolic regression fit should succeed");
        assert!(best.fitness.is_finite());
        assert!(best.mse.is_finite());
    }
}
