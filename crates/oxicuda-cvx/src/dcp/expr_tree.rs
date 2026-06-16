//! Disciplined Convex Programming expression tree.
//!
//! Implements a small expression algebra for convex programs together with the
//! curvature ruleset of Grant & Boyd (2008), "Graph Implementations for
//! Nonsmooth Convex Programs" (the theoretical basis of CVX / CVXPY). Each node
//! carries enough structure to infer its [`Curvature`] compositionally, to
//! evaluate it at a point, to compute a gradient, and to decide whether an
//! optimisation problem is DCP-compliant via [`is_dcp`].
//!
//! # Curvature ruleset
//!
//! The inference follows the standard DCP composition theorem. For a scalar
//! atom `f` applied to arguments, the result is convex when `f` is convex and,
//! for each argument, either
//! - `f` is nondecreasing in that argument and the argument is convex, or
//! - `f` is nonincreasing in that argument and the argument is concave, or
//! - the argument is affine.
//!
//! Concavity is the mirror image. Affine ± affine is affine; a nonnegative
//! scaling preserves curvature while a negative scaling flips convex ↔ concave;
//! negation flips curvature; `max` of convex terms is convex; `min` of concave
//! terms is concave.

use crate::error::{CvxError, CvxResult};

/// Curvature classification of an expression in the DCP ruleset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Curvature {
    /// A constant (no dependence on the variables).
    Constant,
    /// An affine function (both convex and concave).
    Affine,
    /// A convex function.
    Convex,
    /// A concave function.
    Concave,
    /// Curvature could not be certified under the DCP rules.
    Unknown,
}

impl Curvature {
    /// Whether this curvature is admissible where convexity is required
    /// (constant and affine count as convex).
    #[must_use]
    pub fn is_convex(self) -> bool {
        matches!(
            self,
            Curvature::Constant | Curvature::Affine | Curvature::Convex
        )
    }

    /// Whether this curvature is admissible where concavity is required
    /// (constant and affine count as concave).
    #[must_use]
    pub fn is_concave(self) -> bool {
        matches!(
            self,
            Curvature::Constant | Curvature::Affine | Curvature::Concave
        )
    }

    /// Whether this curvature is affine (constant or affine).
    #[must_use]
    pub fn is_affine(self) -> bool {
        matches!(self, Curvature::Constant | Curvature::Affine)
    }

    /// Curvature obtained by negating an expression of this curvature
    /// (convex ↔ concave; affine and constant are self-dual).
    #[must_use]
    pub fn negate(self) -> Curvature {
        match self {
            Curvature::Constant => Curvature::Constant,
            Curvature::Affine => Curvature::Affine,
            Curvature::Convex => Curvature::Concave,
            Curvature::Concave => Curvature::Convex,
            Curvature::Unknown => Curvature::Unknown,
        }
    }

    /// Curvature of the sum of two expressions with these curvatures.
    #[must_use]
    pub fn combine_sum(self, other: Curvature) -> Curvature {
        match (self, other) {
            (Curvature::Constant, c) | (c, Curvature::Constant) => c,
            (Curvature::Affine, Curvature::Affine) => Curvature::Affine,
            (a, b) if a.is_convex() && b.is_convex() => Curvature::Convex,
            (a, b) if a.is_concave() && b.is_concave() => Curvature::Concave,
            _ => Curvature::Unknown,
        }
    }
}

/// Monotonicity of an atom in one of its arguments, used by the composition rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Monotonicity {
    /// Nondecreasing in the argument.
    Increasing,
    /// Nonincreasing in the argument.
    Decreasing,
    /// Neither (the argument must be affine for the composition to certify).
    Nonmonotonic,
}

/// An expression node in a DCP tree over a flat variable vector.
///
/// Variables are indexed by position into the point passed to [`Expr::eval`].
#[derive(Debug, Clone)]
pub enum Expr {
    /// The `i`-th decision variable.
    Var(usize),
    /// A constant scalar.
    Const(f64),
    /// Sum of two expressions.
    Add(Box<Expr>, Box<Expr>),
    /// Difference of two expressions (`lhs − rhs`).
    Sub(Box<Expr>, Box<Expr>),
    /// Negation `−e`.
    Neg(Box<Expr>),
    /// Scalar scaling `α · e` (sign of `α` controls curvature flipping).
    Scale(f64, Box<Expr>),
    /// `e²` — convex, nondecreasing only for nonnegative arguments (so the DCP
    /// rule requires an affine argument).
    Square(Box<Expr>),
    /// `|e|` — convex; nonmonotonic, so requires an affine argument.
    Abs(Box<Expr>),
    /// `max(e₁, …, eₖ)` — convex, nondecreasing in every argument.
    MaxComp(Vec<Expr>),
    /// `‖(e₁, …, eₖ)‖₂` — convex; nonmonotonic, so requires affine arguments.
    Norm2(Vec<Expr>),
    /// `numᵀ num / den` quadratic-over-linear with `num` a vector and `den`
    /// scalar; convex on `den > 0`, nonincreasing in `den`, nonmonotonic in
    /// `num`, so arguments must be affine.
    QuadOverLin(Vec<Expr>, Box<Expr>),
    /// `max(e, 0)` — convex and nondecreasing.
    Pos(Box<Expr>),
    /// `√e` — concave and nondecreasing on `e ≥ 0`.
    Sqrt(Box<Expr>),
    /// `log(e)` — concave and nondecreasing on `e > 0`.
    Log(Box<Expr>),
    /// `min(e₁, …, eₖ)` — concave, nondecreasing in every argument.
    MinComp(Vec<Expr>),
    /// `(e₁ · e₂ · … · eₖ)^{1/k}` geometric mean — concave and nondecreasing on
    /// nonnegative arguments.
    GeoMean(Vec<Expr>),
}

/// The relational kind of a constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    /// `lhs ≤ rhs`.
    LessEq,
    /// `lhs ≥ rhs`.
    GreaterEq,
    /// `lhs = rhs`.
    Equal,
}

/// A single constraint `lhs (kind) rhs`.
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Left-hand-side expression.
    pub lhs: Expr,
    /// Right-hand-side expression.
    pub rhs: Expr,
    /// The relational operator.
    pub kind: ConstraintKind,
}

impl Constraint {
    /// Build a `lhs ≤ rhs` constraint.
    #[must_use]
    pub fn less_eq(lhs: Expr, rhs: Expr) -> Self {
        Self {
            lhs,
            rhs,
            kind: ConstraintKind::LessEq,
        }
    }

    /// Build a `lhs ≥ rhs` constraint.
    #[must_use]
    pub fn greater_eq(lhs: Expr, rhs: Expr) -> Self {
        Self {
            lhs,
            rhs,
            kind: ConstraintKind::GreaterEq,
        }
    }

    /// Build a `lhs = rhs` constraint.
    #[must_use]
    pub fn equal(lhs: Expr, rhs: Expr) -> Self {
        Self {
            lhs,
            rhs,
            kind: ConstraintKind::Equal,
        }
    }
}

impl Expr {
    /// Infer the [`Curvature`] of this expression under the DCP ruleset.
    #[must_use]
    pub fn curvature(&self) -> Curvature {
        match self {
            Expr::Var(_) => Curvature::Affine,
            Expr::Const(_) => Curvature::Constant,
            Expr::Add(a, b) => a.curvature().combine_sum(b.curvature()),
            Expr::Sub(a, b) => a.curvature().combine_sum(b.curvature().negate()),
            Expr::Neg(e) => e.curvature().negate(),
            Expr::Scale(alpha, e) => {
                let c = e.curvature();
                if *alpha >= 0.0 { c } else { c.negate() }
            }
            // Convex atoms.
            Expr::Square(e) => convex_atom_curvature(
                std::slice::from_ref(e.as_ref()),
                &[Monotonicity::Nonmonotonic],
            ),
            Expr::Abs(e) => convex_atom_curvature(
                std::slice::from_ref(e.as_ref()),
                &[Monotonicity::Nonmonotonic],
            ),
            Expr::Pos(e) => convex_atom_curvature(
                std::slice::from_ref(e.as_ref()),
                &[Monotonicity::Increasing],
            ),
            Expr::MaxComp(args) => {
                if args.is_empty() {
                    return Curvature::Unknown;
                }
                let monos = vec![Monotonicity::Increasing; args.len()];
                convex_atom_curvature(args, &monos)
            }
            Expr::Norm2(args) => {
                if args.is_empty() {
                    return Curvature::Constant;
                }
                let monos = vec![Monotonicity::Nonmonotonic; args.len()];
                convex_atom_curvature(args, &monos)
            }
            Expr::QuadOverLin(num, den) => {
                // Convex; nonincreasing in den, nonmonotonic in num.
                let mut args: Vec<&Expr> = num.iter().collect();
                args.push(den.as_ref());
                let mut monos = vec![Monotonicity::Nonmonotonic; num.len()];
                monos.push(Monotonicity::Decreasing);
                convex_atom_curvature_refs(&args, &monos)
            }
            // Concave atoms.
            Expr::Sqrt(e) => concave_atom_curvature(
                std::slice::from_ref(e.as_ref()),
                &[Monotonicity::Increasing],
            ),
            Expr::Log(e) => concave_atom_curvature(
                std::slice::from_ref(e.as_ref()),
                &[Monotonicity::Increasing],
            ),
            Expr::MinComp(args) => {
                if args.is_empty() {
                    return Curvature::Unknown;
                }
                let monos = vec![Monotonicity::Increasing; args.len()];
                concave_atom_curvature(args, &monos)
            }
            Expr::GeoMean(args) => {
                if args.is_empty() {
                    return Curvature::Unknown;
                }
                let monos = vec![Monotonicity::Increasing; args.len()];
                concave_atom_curvature(args, &monos)
            }
        }
    }

    /// Evaluate the expression at the variable assignment `point`.
    ///
    /// # Errors
    ///
    /// Returns [`CvxError::IndexOutOfBounds`] for an out-of-range variable index
    /// and [`CvxError::EmptyInput`] for an empty `max`/`min`/`geomean`.
    pub fn eval(&self, point: &[f64]) -> CvxResult<f64> {
        match self {
            Expr::Var(i) => point.get(*i).copied().ok_or(CvxError::IndexOutOfBounds {
                index: *i,
                len: point.len(),
            }),
            Expr::Const(c) => Ok(*c),
            Expr::Add(a, b) => Ok(a.eval(point)? + b.eval(point)?),
            Expr::Sub(a, b) => Ok(a.eval(point)? - b.eval(point)?),
            Expr::Neg(e) => Ok(-e.eval(point)?),
            Expr::Scale(alpha, e) => Ok(alpha * e.eval(point)?),
            Expr::Square(e) => {
                let v = e.eval(point)?;
                Ok(v * v)
            }
            Expr::Abs(e) => Ok(e.eval(point)?.abs()),
            Expr::Pos(e) => Ok(e.eval(point)?.max(0.0)),
            Expr::MaxComp(args) => {
                if args.is_empty() {
                    return Err(CvxError::EmptyInput);
                }
                let mut best = f64::NEG_INFINITY;
                for a in args {
                    best = best.max(a.eval(point)?);
                }
                Ok(best)
            }
            Expr::MinComp(args) => {
                if args.is_empty() {
                    return Err(CvxError::EmptyInput);
                }
                let mut best = f64::INFINITY;
                for a in args {
                    best = best.min(a.eval(point)?);
                }
                Ok(best)
            }
            Expr::Norm2(args) => {
                let mut s = 0.0_f64;
                for a in args {
                    let v = a.eval(point)?;
                    s += v * v;
                }
                Ok(s.sqrt())
            }
            Expr::QuadOverLin(num, den) => {
                let d = den.eval(point)?;
                if d <= 0.0 {
                    return Err(CvxError::InvalidParameter(format!(
                        "quad_over_lin denominator must be > 0, got {d}"
                    )));
                }
                let mut s = 0.0_f64;
                for a in num {
                    let v = a.eval(point)?;
                    s += v * v;
                }
                Ok(s / d)
            }
            Expr::Sqrt(e) => {
                let v = e.eval(point)?;
                if v < 0.0 {
                    return Err(CvxError::InvalidParameter(format!(
                        "sqrt of negative value {v}"
                    )));
                }
                Ok(v.sqrt())
            }
            Expr::Log(e) => {
                let v = e.eval(point)?;
                if v <= 0.0 {
                    return Err(CvxError::InvalidParameter(format!(
                        "log of non-positive value {v}"
                    )));
                }
                Ok(v.ln())
            }
            Expr::GeoMean(args) => {
                if args.is_empty() {
                    return Err(CvxError::EmptyInput);
                }
                let k = args.len() as f64;
                let mut log_sum = 0.0_f64;
                for a in args {
                    let v = a.eval(point)?;
                    if v < 0.0 {
                        return Err(CvxError::InvalidParameter(format!(
                            "geo_mean of negative value {v}"
                        )));
                    }
                    if v == 0.0 {
                        return Ok(0.0);
                    }
                    log_sum += v.ln();
                }
                Ok((log_sum / k).exp())
            }
        }
    }

    /// Gradient of the expression with respect to all variables at `point`.
    ///
    /// Analytic where a closed form is available; otherwise a central
    /// finite-difference approximation is used. Returns a vector of length
    /// `point.len()`.
    ///
    /// # Errors
    ///
    /// Propagates evaluation errors from [`Expr::eval`].
    pub fn grad(&self, point: &[f64]) -> CvxResult<Vec<f64>> {
        // Central finite differences provide a uniformly-valid fallback and a
        // cross-check; the analytic forms below match it to O(h²).
        match self {
            Expr::Var(i) => {
                let mut g = vec![0.0_f64; point.len()];
                if *i >= point.len() {
                    return Err(CvxError::IndexOutOfBounds {
                        index: *i,
                        len: point.len(),
                    });
                }
                g[*i] = 1.0;
                Ok(g)
            }
            Expr::Const(_) => Ok(vec![0.0_f64; point.len()]),
            Expr::Add(a, b) => {
                let ga = a.grad(point)?;
                let gb = b.grad(point)?;
                Ok(ga.iter().zip(gb.iter()).map(|(x, y)| x + y).collect())
            }
            Expr::Sub(a, b) => {
                let ga = a.grad(point)?;
                let gb = b.grad(point)?;
                Ok(ga.iter().zip(gb.iter()).map(|(x, y)| x - y).collect())
            }
            Expr::Neg(e) => Ok(e.grad(point)?.iter().map(|v| -v).collect()),
            Expr::Scale(alpha, e) => Ok(e.grad(point)?.iter().map(|v| alpha * v).collect()),
            Expr::Square(e) => {
                // d/dx (e²) = 2 e · ∇e.
                let val = e.eval(point)?;
                let ge = e.grad(point)?;
                Ok(ge.iter().map(|v| 2.0 * val * v).collect())
            }
            Expr::Norm2(args) => {
                // d/dx ‖a‖ = (1/‖a‖) Σ aᵢ ∇aᵢ ; zero subgradient at the origin.
                let mut vals = Vec::with_capacity(args.len());
                let mut norm_sq = 0.0_f64;
                for a in args {
                    let v = a.eval(point)?;
                    norm_sq += v * v;
                    vals.push(v);
                }
                let norm = norm_sq.sqrt();
                let mut g = vec![0.0_f64; point.len()];
                if norm <= 0.0 {
                    return Ok(g);
                }
                for (a, &v) in args.iter().zip(vals.iter()) {
                    let ga = a.grad(point)?;
                    for (gj, gaj) in g.iter_mut().zip(ga.iter()) {
                        *gj += (v / norm) * gaj;
                    }
                }
                Ok(g)
            }
            _ => self.grad_finite_difference(point),
        }
    }

    /// Central finite-difference gradient with a relative step size.
    fn grad_finite_difference(&self, point: &[f64]) -> CvxResult<Vec<f64>> {
        let mut g = vec![0.0_f64; point.len()];
        let mut work = point.to_vec();
        for i in 0..point.len() {
            let base = point[i];
            let h = 1.0e-6 * base.abs().max(1.0);
            work[i] = base + h;
            let fp = self.eval(&work)?;
            work[i] = base - h;
            let fm = self.eval(&work)?;
            work[i] = base;
            g[i] = (fp - fm) / (2.0 * h);
        }
        Ok(g)
    }

    /// Whether every argument of this node is affine (used by DCP atom checks).
    fn args_affine(&self) -> bool {
        match self {
            Expr::Square(e) | Expr::Abs(e) => e.curvature().is_affine(),
            Expr::Norm2(args) => args.iter().all(|a| a.curvature().is_affine()),
            Expr::QuadOverLin(num, den) => {
                num.iter().all(|a| a.curvature().is_affine()) && den.curvature().is_affine()
            }
            _ => true,
        }
    }
}

/// Curvature of a convex atom under the DCP composition rule, given the
/// monotonicity of the atom in each argument.
fn convex_atom_curvature(args: &[Expr], monos: &[Monotonicity]) -> Curvature {
    let refs: Vec<&Expr> = args.iter().collect();
    convex_atom_curvature_refs(&refs, monos)
}

/// Reference-slice variant of [`convex_atom_curvature`].
fn convex_atom_curvature_refs(args: &[&Expr], monos: &[Monotonicity]) -> Curvature {
    if args.len() != monos.len() {
        return Curvature::Unknown;
    }
    for (arg, mono) in args.iter().zip(monos.iter()) {
        let c = arg.curvature();
        let ok = match mono {
            Monotonicity::Increasing => c.is_convex(),
            Monotonicity::Decreasing => c.is_concave(),
            Monotonicity::Nonmonotonic => c.is_affine(),
        };
        if !ok {
            return Curvature::Unknown;
        }
    }
    Curvature::Convex
}

/// Curvature of a concave atom under the (mirror) DCP composition rule.
fn concave_atom_curvature(args: &[Expr], monos: &[Monotonicity]) -> Curvature {
    if args.len() != monos.len() {
        return Curvature::Unknown;
    }
    for (arg, mono) in args.iter().zip(monos.iter()) {
        let c = arg.curvature();
        let ok = match mono {
            // Concave nondecreasing of concave → concave.
            Monotonicity::Increasing => c.is_concave(),
            // Concave nonincreasing of convex → concave.
            Monotonicity::Decreasing => c.is_convex(),
            Monotonicity::Nonmonotonic => c.is_affine(),
        };
        if !ok {
            return Curvature::Unknown;
        }
    }
    Curvature::Concave
}

/// Recursively verify that every atom in `expr` has DCP-admissible arguments.
///
/// This guards against trees whose top-level curvature happens to resolve but
/// whose internal atoms are applied to non-affine arguments where the DCP rule
/// demands affinity (for example `Square` of a non-affine convex expression).
fn atoms_well_formed(expr: &Expr) -> bool {
    if !expr.args_affine() {
        return false;
    }
    match expr {
        Expr::Var(_) | Expr::Const(_) => true,
        Expr::Neg(e)
        | Expr::Scale(_, e)
        | Expr::Square(e)
        | Expr::Abs(e)
        | Expr::Pos(e)
        | Expr::Sqrt(e)
        | Expr::Log(e) => atoms_well_formed(e),
        Expr::Add(a, b) | Expr::Sub(a, b) => atoms_well_formed(a) && atoms_well_formed(b),
        Expr::MaxComp(args) | Expr::MinComp(args) | Expr::Norm2(args) | Expr::GeoMean(args) => {
            args.iter().all(atoms_well_formed)
        }
        Expr::QuadOverLin(num, den) => num.iter().all(atoms_well_formed) && atoms_well_formed(den),
    }
}

/// Check whether the problem `minimize obj subject to constraints` is DCP.
///
/// The objective must be convex (for a minimisation). Each constraint must be
/// sign-convex: `convex ≤ concave`, `concave ≥ convex`, or `affine = affine`.
/// Every atom must additionally be applied to DCP-admissible arguments.
#[must_use]
pub fn is_dcp(obj: &Expr, constraints: &[Constraint]) -> bool {
    // Objective: convex (minimisation convention) and structurally well formed.
    if !obj.curvature().is_convex() || !atoms_well_formed(obj) {
        return false;
    }
    for con in constraints {
        if !atoms_well_formed(&con.lhs) || !atoms_well_formed(&con.rhs) {
            return false;
        }
        let lc = con.lhs.curvature();
        let rc = con.rhs.curvature();
        let ok = match con.kind {
            // convex ≤ concave.
            ConstraintKind::LessEq => lc.is_convex() && rc.is_concave(),
            // concave ≥ convex.
            ConstraintKind::GreaterEq => lc.is_concave() && rc.is_convex(),
            // affine = affine.
            ConstraintKind::Equal => lc.is_affine() && rc.is_affine(),
        };
        if !ok {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(i: usize) -> Expr {
        Expr::Var(i)
    }
    fn cst(v: f64) -> Expr {
        Expr::Const(v)
    }
    fn boxed(e: Expr) -> Box<Expr> {
        Box::new(e)
    }

    #[test]
    fn curvature_sum_of_squares_is_convex() {
        // Square(x0) + Square(x1).
        let e = Expr::Add(
            boxed(Expr::Square(boxed(var(0)))),
            boxed(Expr::Square(boxed(var(1)))),
        );
        assert_eq!(e.curvature(), Curvature::Convex);
    }

    #[test]
    fn curvature_neg_square_is_concave() {
        let e = Expr::Neg(boxed(Expr::Square(boxed(var(0)))));
        assert_eq!(e.curvature(), Curvature::Concave);
    }

    #[test]
    fn curvature_affine_combo_is_affine() {
        // 3·x0 − 2·x1 + 5.
        let e = Expr::Add(
            boxed(Expr::Sub(
                boxed(Expr::Scale(3.0, boxed(var(0)))),
                boxed(Expr::Scale(2.0, boxed(var(1)))),
            )),
            boxed(cst(5.0)),
        );
        assert_eq!(e.curvature(), Curvature::Affine);
    }

    #[test]
    fn negative_scale_flips_curvature() {
        // −2 · Square(x0) is concave; +2 · Square(x0) is convex.
        let conv = Expr::Scale(2.0, boxed(Expr::Square(boxed(var(0)))));
        let conc = Expr::Scale(-2.0, boxed(Expr::Square(boxed(var(0)))));
        assert_eq!(conv.curvature(), Curvature::Convex);
        assert_eq!(conc.curvature(), Curvature::Concave);
    }

    #[test]
    fn eval_constants_and_square() {
        let e = Expr::Add(boxed(cst(1.0)), boxed(cst(2.0)));
        assert!((e.eval(&[]).expect("eval") - 3.0).abs() < 1.0e-12);
        let sq = Expr::Square(boxed(var(0)));
        assert!((sq.eval(&[4.0]).expect("eval") - 16.0).abs() < 1.0e-12);
    }

    #[test]
    fn eval_norm2_and_max_min() {
        let n = Expr::Norm2(vec![var(0), var(1)]);
        assert!((n.eval(&[3.0, 4.0]).expect("eval") - 5.0).abs() < 1.0e-12);
        let mx = Expr::MaxComp(vec![var(0), var(1), cst(2.0)]);
        assert!((mx.eval(&[1.0, -1.0]).expect("eval") - 2.0).abs() < 1.0e-12);
        let mn = Expr::MinComp(vec![var(0), var(1), cst(2.0)]);
        assert!((mn.eval(&[1.0, -1.0]).expect("eval") + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn eval_log_sqrt_geomean() {
        let lg = Expr::Log(boxed(var(0)));
        assert!((lg.eval(&[std::f64::consts::E]).expect("eval") - 1.0).abs() < 1.0e-12);
        let sq = Expr::Sqrt(boxed(var(0)));
        assert!((sq.eval(&[9.0]).expect("eval") - 3.0).abs() < 1.0e-12);
        let gm = Expr::GeoMean(vec![var(0), var(1)]);
        // geo_mean(4, 9) = 6.
        assert!((gm.eval(&[4.0, 9.0]).expect("eval") - 6.0).abs() < 1.0e-12);
    }

    #[test]
    fn grad_square_matches_finite_difference() {
        // f = Square(x0) → ∇ = (2 x0, 0).
        let e = Expr::Square(boxed(var(0)));
        let point = [3.0, 1.0];
        let g = e.grad(&point).expect("grad");
        let fd = e.grad_finite_difference(&point).expect("fd");
        assert!((g[0] - 6.0).abs() < 1.0e-9, "g0 = {}", g[0]);
        assert!(
            (g[0] - fd[0]).abs() < 1.0e-4,
            "analytic {} vs fd {}",
            g[0],
            fd[0]
        );
        assert!(g[1].abs() < 1.0e-12);
    }

    #[test]
    fn grad_norm2_matches_finite_difference() {
        let e = Expr::Norm2(vec![var(0), var(1)]);
        let point = [3.0, 4.0];
        let g = e.grad(&point).expect("grad");
        let fd = e.grad_finite_difference(&point).expect("fd");
        // ∇‖x‖ = x/‖x‖ = (0.6, 0.8).
        assert!((g[0] - 0.6).abs() < 1.0e-9, "g0 = {}", g[0]);
        assert!((g[1] - 0.8).abs() < 1.0e-9, "g1 = {}", g[1]);
        for (gi, fdi) in g.iter().zip(fd.iter()) {
            assert!((gi - fdi).abs() < 1.0e-4, "analytic {gi} vs fd {fdi}");
        }
    }

    #[test]
    fn grad_log_via_finite_difference() {
        // f = Log(x0) → ∇ = (1/x0, 0).
        let e = Expr::Log(boxed(var(0)));
        let point = [2.0, 7.0];
        let g = e.grad(&point).expect("grad");
        assert!((g[0] - 0.5).abs() < 1.0e-4, "g0 = {}", g[0]);
        assert!(g[1].abs() < 1.0e-8);
    }

    #[test]
    fn is_dcp_accepts_convex_quadratic() {
        // minimize Square(x0) + Square(x1) s.t. x0 + x1 = 1, ‖(x0,x1)‖ ≤ 2.
        let obj = Expr::Add(
            boxed(Expr::Square(boxed(var(0)))),
            boxed(Expr::Square(boxed(var(1)))),
        );
        let eq = Constraint::equal(Expr::Add(boxed(var(0)), boxed(var(1))), cst(1.0));
        let soc = Constraint::less_eq(Expr::Norm2(vec![var(0), var(1)]), cst(2.0));
        assert!(is_dcp(&obj, &[eq, soc]));
    }

    #[test]
    fn is_dcp_rejects_square_of_nonaffine() {
        // Square applied to a convex (non-affine) argument violates DCP.
        // Inner = Square(x0) (convex). Outer = Square(inner) → not DCP.
        let inner = Expr::Square(boxed(var(0)));
        let bad = Expr::Square(boxed(inner));
        // Curvature is Unknown and atoms are not well-formed.
        assert_eq!(bad.curvature(), Curvature::Unknown);
        assert!(!is_dcp(&bad, &[]));
    }

    #[test]
    fn is_dcp_rejects_convex_equality() {
        // Convex == affine equality is not DCP (equality needs affine == affine).
        let obj = cst(0.0);
        let bad_eq = Constraint::equal(Expr::Square(boxed(var(0))), cst(1.0));
        assert!(!is_dcp(&obj, &[bad_eq]));
    }

    #[test]
    fn is_dcp_rejects_convex_geq_convex() {
        // convex ≥ convex is not DCP (≥ needs concave ≥ convex).
        let obj = cst(0.0);
        let bad = Constraint::greater_eq(Expr::Square(boxed(var(0))), Expr::Square(boxed(var(1))));
        assert!(!is_dcp(&obj, &[bad]));
    }

    #[test]
    fn is_dcp_accepts_concave_geq_constraint() {
        // concave ≥ convex: Sqrt(x0) ≥ Square(x1) is DCP-valid structurally.
        let obj = cst(0.0);
        let good = Constraint::greater_eq(Expr::Sqrt(boxed(var(0))), Expr::Square(boxed(var(1))));
        assert!(is_dcp(&obj, &[good]));
    }

    #[test]
    fn max_of_convex_is_convex() {
        let e = Expr::MaxComp(vec![Expr::Square(boxed(var(0))), Expr::Abs(boxed(var(1)))]);
        assert_eq!(e.curvature(), Curvature::Convex);
    }

    #[test]
    fn min_of_concave_is_concave() {
        let e = Expr::MinComp(vec![Expr::Sqrt(boxed(var(0))), Expr::Log(boxed(var(1)))]);
        assert_eq!(e.curvature(), Curvature::Concave);
    }

    #[test]
    fn max_of_concave_is_unknown() {
        // max of a concave term is NOT certifiably convex.
        let e = Expr::MaxComp(vec![Expr::Sqrt(boxed(var(0))), Expr::Log(boxed(var(1)))]);
        assert_eq!(e.curvature(), Curvature::Unknown);
    }

    #[test]
    fn quad_over_lin_is_convex() {
        // ‖(x0,x1)‖² / x2 is convex for affine arguments.
        let e = Expr::QuadOverLin(vec![var(0), var(1)], boxed(var(2)));
        assert_eq!(e.curvature(), Curvature::Convex);
        // Evaluate: (3² + 4²) / 5 = 25/5 = 5.
        assert!((e.eval(&[3.0, 4.0, 5.0]).expect("eval") - 5.0).abs() < 1.0e-12);
    }

    #[test]
    fn log_of_concave_is_concave() {
        // log(sqrt(x0)) — log (concave, increasing) of concave-nonneg = concave.
        let e = Expr::Log(boxed(Expr::Sqrt(boxed(var(0)))));
        assert_eq!(e.curvature(), Curvature::Concave);
    }

    #[test]
    fn pos_of_affine_is_convex() {
        let e = Expr::Pos(boxed(Expr::Sub(boxed(var(0)), boxed(cst(1.0)))));
        assert_eq!(e.curvature(), Curvature::Convex);
        assert!((e.eval(&[0.5]).expect("eval")).abs() < 1.0e-12);
        assert!((e.eval(&[3.0]).expect("eval") - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn eval_out_of_bounds_errors() {
        let e = var(5);
        assert!(matches!(
            e.eval(&[1.0]),
            Err(CvxError::IndexOutOfBounds { .. })
        ));
    }
}
