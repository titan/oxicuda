//! Parametric maximum flow (Gallo-Grigoriadis-Tarjan 1989 / Hochbaum 2008).
//!
//! Solves a *family* of `s`-`t` maximum-flow problems in which some arc
//! capacities vary monotonically with a scalar parameter `lambda`. Each arc
//! capacity is modelled as an affine function `cap(lambda) = base + slope * lambda`
//! (clamped at `0`). The Gallo-Grigoriadis-Tarjan (GGT) *monotone* structure is
//! enforced:
//!
//! * arcs leaving the source (`from == source`) have **non-decreasing** capacity
//!   (`slope >= 0`),
//! * arcs entering the sink (`to == sink`) have **non-increasing** capacity
//!   (`slope <= 0`),
//! * every other arc has a **constant** capacity (`slope == 0`).
//!
//! Under this structure the minimal source side `S(lambda)` of the minimum cut is
//! a *non-decreasing* (nested) function of `lambda`: for `l1 <= l2` we have
//! `S(l1) ⊆ S(l2)`. This module exposes
//!
//! * [`ParametricMaxFlow::solve_at`] — a single `(value, source-set)` solve at one
//!   `lambda`, reusing the crate's Edmonds-Karp / residual-reachability min-cut,
//! * [`ParametricMaxFlow::solve_grid`] — the whole family over a `lambda` grid,
//! * [`ParametricMaxFlow::find_breakpoints`] — the GGT divide-and-conquer
//!   "slicing" that locates every `lambda` at which the minimal min-cut changes,
//!   exploiting the nested-cut structure (one max-flow solve per slice).
//!
//! The per-`lambda` solves are exact single-parameter max-flows, so they agree
//! with an independent solve at every `lambda`.

use crate::error::{GraphalgError, GraphalgResult};
use crate::max_flow::edmonds_karp::FlowNetwork;
use crate::max_flow::min_cut::min_cut_from_max_flow;

/// Numerical tolerance for capacity / slack comparisons.
const EPS: f64 = 1.0e-9;

/// An arc whose capacity is affine in the parameter: `cap(lambda) = base + slope * lambda`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParametricArc {
    /// Tail endpoint.
    pub from: usize,
    /// Head endpoint.
    pub to: usize,
    /// Constant part of the capacity.
    pub base: f64,
    /// Per-unit-`lambda` part of the capacity.
    pub slope: f64,
}

/// Result of one single-`lambda` parametric solve.
#[derive(Debug, Clone, PartialEq)]
pub struct ParametricSolution {
    /// The parameter value.
    pub lambda: f64,
    /// Maximum flow value (= minimum cut capacity) at this `lambda`.
    pub flow_value: f64,
    /// Minimal source side of the minimum cut (residual-reachable from the
    /// source), sorted ascending. Nested/monotone in `lambda`.
    pub source_set: Vec<usize>,
}

/// A `lambda` value at which the optimal minimal min-cut source side changes.
#[derive(Debug, Clone, PartialEq)]
pub struct ParametricBreakpoint {
    /// Parameter value of the transition.
    pub lambda: f64,
    /// Minimal source side that is optimal immediately *above* this breakpoint
    /// (until the next breakpoint), sorted ascending.
    pub source_set: Vec<usize>,
}

/// A parametric `s`-`t` flow network with affine, GGT-monotone arc capacities.
#[derive(Debug, Clone)]
pub struct ParametricMaxFlow {
    n: usize,
    source: usize,
    sink: usize,
    arcs: Vec<ParametricArc>,
}

impl ParametricMaxFlow {
    /// Create an empty parametric network on `n` nodes with the given source/sink.
    pub fn new(n: usize, source: usize, sink: usize) -> GraphalgResult<Self> {
        if n == 0 {
            return Err(GraphalgError::EmptyInput);
        }
        if source >= n || sink >= n {
            return Err(GraphalgError::SourceOutOfRange {
                node: source.max(sink),
                n,
            });
        }
        if source == sink {
            return Err(GraphalgError::InvalidParameter(
                "source and sink must differ".to_string(),
            ));
        }
        Ok(Self {
            n,
            source,
            sink,
            arcs: Vec::new(),
        })
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.n
    }

    /// Source node.
    pub fn source(&self) -> usize {
        self.source
    }

    /// Sink node.
    pub fn sink(&self) -> usize {
        self.sink
    }

    /// Arcs of the network.
    pub fn arcs(&self) -> &[ParametricArc] {
        &self.arcs
    }

    /// Add a parametric arc `from -> to` with capacity `base + slope * lambda`.
    ///
    /// Enforces the GGT monotone structure (see module docs) and rejects
    /// out-of-range endpoints and non-finite coefficients.
    pub fn add_arc(&mut self, from: usize, to: usize, base: f64, slope: f64) -> GraphalgResult<()> {
        if from >= self.n || to >= self.n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: from.max(to),
                len: self.n,
            });
        }
        if from == to {
            return Err(GraphalgError::InvalidParameter(format!(
                "self-loop arc at node {from} is not allowed"
            )));
        }
        if !base.is_finite() || !slope.is_finite() {
            return Err(GraphalgError::InvalidEdgeWeight(format!(
                "arc ({from},{to}) has non-finite coefficients base={base} slope={slope}"
            )));
        }
        let is_source_arc = from == self.source;
        let is_sink_arc = to == self.sink;
        if is_source_arc && slope < -EPS {
            return Err(GraphalgError::InvalidConfiguration(format!(
                "source arc ({from},{to}) must be non-decreasing in lambda (slope {slope} < 0)"
            )));
        }
        if is_sink_arc && slope > EPS {
            return Err(GraphalgError::InvalidConfiguration(format!(
                "sink arc ({from},{to}) must be non-increasing in lambda (slope {slope} > 0)"
            )));
        }
        if !is_source_arc && !is_sink_arc && slope.abs() > EPS {
            return Err(GraphalgError::InvalidConfiguration(format!(
                "interior arc ({from},{to}) must have constant capacity (slope {slope} != 0)"
            )));
        }
        self.arcs.push(ParametricArc {
            from,
            to,
            base,
            slope,
        });
        Ok(())
    }

    /// Build a parametric network from parallel coefficient slices.
    ///
    /// All slices must share the same length, otherwise a [`GraphalgError::DimensionMismatch`]
    /// is returned. Each tuple position `i` describes arc `from[i] -> to[i]` with
    /// capacity `base[i] + slope[i] * lambda`.
    pub fn from_linear_capacities(
        n: usize,
        source: usize,
        sink: usize,
        from: &[usize],
        to: &[usize],
        base: &[f64],
        slope: &[f64],
    ) -> GraphalgResult<Self> {
        let m = from.len();
        if to.len() != m {
            return Err(GraphalgError::DimensionMismatch { a: m, b: to.len() });
        }
        if base.len() != m {
            return Err(GraphalgError::DimensionMismatch {
                a: m,
                b: base.len(),
            });
        }
        if slope.len() != m {
            return Err(GraphalgError::DimensionMismatch {
                a: m,
                b: slope.len(),
            });
        }
        let mut net = Self::new(n, source, sink)?;
        for i in 0..m {
            net.add_arc(from[i], to[i], base[i], slope[i])?;
        }
        Ok(net)
    }

    /// Capacity of an arc at a given `lambda`, clamped to be non-negative.
    fn arc_capacity(arc: &ParametricArc, lambda: f64) -> f64 {
        (arc.base + arc.slope * lambda).max(0.0)
    }

    /// Materialise a concrete [`FlowNetwork`] at the given `lambda`.
    fn build_network(&self, lambda: f64) -> GraphalgResult<FlowNetwork> {
        if !lambda.is_finite() {
            return Err(GraphalgError::InvalidParameter(format!(
                "lambda must be finite, got {lambda}"
            )));
        }
        let mut net = FlowNetwork::new(self.n);
        for arc in &self.arcs {
            let cap = Self::arc_capacity(arc, lambda);
            if cap > 0.0 {
                net.add_edge(arc.from, arc.to, cap)?;
            }
        }
        Ok(net)
    }

    /// Solve the maximum flow at a single `lambda`, returning the flow value and
    /// the minimal source side of the minimum cut (sorted ascending).
    pub fn solve_at(&self, lambda: f64) -> GraphalgResult<ParametricSolution> {
        let net = self.build_network(lambda)?;
        let cut = min_cut_from_max_flow(&net, self.source, self.sink)?;
        let mut source_set = cut.source_side;
        source_set.sort_unstable();
        Ok(ParametricSolution {
            lambda,
            flow_value: cut.value,
            source_set,
        })
    }

    /// Solve the whole family over the provided `lambda` grid.
    ///
    /// Each entry is an exact independent single-parameter max-flow solve. When
    /// the grid is non-decreasing the returned `source_set`s are nested.
    pub fn solve_grid(&self, lambdas: &[f64]) -> GraphalgResult<Vec<ParametricSolution>> {
        let mut out = Vec::with_capacity(lambdas.len());
        for &lambda in lambdas {
            out.push(self.solve_at(lambda)?);
        }
        Ok(out)
    }

    /// Affine cut-capacity line `(intercept, slope)` of a given source side
    /// `S`: the sum over arcs crossing from `S` to its complement of
    /// `(base, slope)`. Uses the raw (un-clamped) coefficients, valid wherever
    /// the capacities stay non-negative.
    fn cut_line(&self, in_source: &[bool]) -> (f64, f64) {
        let mut intercept = 0.0;
        let mut slope = 0.0;
        for arc in &self.arcs {
            if in_source[arc.from] && !in_source[arc.to] {
                intercept += arc.base;
                slope += arc.slope;
            }
        }
        (intercept, slope)
    }

    /// Build a membership mask from a sorted source-set.
    fn membership(&self, source_set: &[usize]) -> Vec<bool> {
        let mut mask = vec![false; self.n];
        for &v in source_set {
            mask[v] = true;
        }
        mask
    }

    /// Locate every `lambda` in `[lambda_min, lambda_max]` at which the minimal
    /// minimum-cut source side changes, via Gallo-Grigoriadis-Tarjan slicing.
    ///
    /// Returns the breakpoints sorted by ascending `lambda`. Each breakpoint
    /// carries the source side optimal immediately above it. Assumes the affine
    /// capacities stay non-negative across the interval (the standard GGT
    /// regime); otherwise the breakpoint geometry uses the un-clamped lines.
    pub fn find_breakpoints(
        &self,
        lambda_min: f64,
        lambda_max: f64,
    ) -> GraphalgResult<Vec<ParametricBreakpoint>> {
        if !lambda_min.is_finite() || !lambda_max.is_finite() {
            return Err(GraphalgError::InvalidParameter(
                "lambda bounds must be finite".to_string(),
            ));
        }
        if lambda_min > lambda_max {
            return Err(GraphalgError::InvalidParameter(format!(
                "lambda_min ({lambda_min}) must not exceed lambda_max ({lambda_max})"
            )));
        }
        let lo = self.slice_end(lambda_min)?;
        let hi = self.slice_end(lambda_max)?;
        let mut out = Vec::new();
        let budget = 8 * self.n + 16;
        self.slice(&lo, &hi, budget, &mut out)?;
        out.sort_by(|a, b| {
            a.lambda
                .partial_cmp(&b.lambda)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(out)
    }

    /// Solve at `lambda` and package the source side together with its cut line.
    fn slice_end(&self, lambda: f64) -> GraphalgResult<SliceEnd> {
        let sol = self.solve_at(lambda)?;
        let mask = self.membership(&sol.source_set);
        let line = self.cut_line(&mask);
        Ok(SliceEnd {
            lambda,
            flow_value: sol.flow_value,
            source_set: sol.source_set,
            line,
        })
    }

    fn slice(
        &self,
        lo: &SliceEnd,
        hi: &SliceEnd,
        budget: usize,
        out: &mut Vec<ParametricBreakpoint>,
    ) -> GraphalgResult<()> {
        if budget == 0 {
            return Err(GraphalgError::NotConverged {
                iter: 8 * self.n + 16,
            });
        }
        if lo.source_set == hi.source_set {
            return Ok(());
        }
        let (a_lo, b_lo) = lo.line;
        let (a_hi, b_hi) = hi.line;
        let denom = b_lo - b_hi;
        if denom.abs() <= EPS {
            // Parallel cut lines with differing sets: record the transition at the
            // upper endpoint of the slice; the upper set dominates above it.
            out.push(ParametricBreakpoint {
                lambda: hi.lambda,
                source_set: hi.source_set.clone(),
            });
            return Ok(());
        }
        let lam_star = (a_hi - a_lo) / denom;
        if lam_star <= lo.lambda + EPS || lam_star >= hi.lambda - EPS {
            // Intersection falls outside the open interval: the two cuts already
            // describe the cut structure here; record the transition at lam_star
            // clamped into the interval.
            let clamped = lam_star.clamp(lo.lambda, hi.lambda);
            out.push(ParametricBreakpoint {
                lambda: clamped,
                source_set: hi.source_set.clone(),
            });
            return Ok(());
        }
        let mid = self.slice_end(lam_star)?;
        let y_int = a_lo + b_lo * lam_star;
        if mid.flow_value >= y_int - EPS {
            // No cut beats the intersection of the two bounding lines: lam_star is
            // a genuine breakpoint between lo and hi source sides.
            out.push(ParametricBreakpoint {
                lambda: lam_star,
                source_set: hi.source_set.clone(),
            });
            Ok(())
        } else {
            self.slice(lo, &mid, budget - 1, out)?;
            self.slice(&mid, hi, budget - 1, out)?;
            Ok(())
        }
    }
}

/// One endpoint of a GGT slice: a solved `lambda` with its source side and cut line.
#[derive(Debug, Clone)]
struct SliceEnd {
    lambda: f64,
    flow_value: f64,
    source_set: Vec<usize>,
    line: (f64, f64),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-7
    }

    /// Independent single max-flow at a given lambda (oracle), via Dinic.
    fn independent_flow(net: &ParametricMaxFlow, lambda: f64) -> f64 {
        let fnet = net.build_network(lambda).expect("build");
        crate::max_flow::dinic::dinic_max_flow(&fnet, net.source(), net.sink()).expect("dinic")
    }

    /// Build the canonical nested example: s=0, a=1, b=2, t=3.
    /// s->a, s->b capacities = lambda (source arcs, slope 1); a->t, b->t = 1.
    fn nested_net() -> ParametricMaxFlow {
        let mut net = ParametricMaxFlow::new(4, 0, 3).expect("new");
        net.add_arc(0, 1, 0.0, 1.0).expect("s->a");
        net.add_arc(0, 2, 0.0, 1.0).expect("s->b");
        net.add_arc(1, 3, 1.0, 0.0).expect("a->t");
        net.add_arc(2, 3, 1.0, 0.0).expect("b->t");
        net
    }

    #[test]
    fn grid_matches_independent_solves() {
        let net = nested_net();
        for k in 0..=20 {
            let lambda = k as f64 * 0.2;
            let sol = net.solve_at(lambda).expect("solve");
            let indep = independent_flow(&net, lambda);
            assert!(
                approx(sol.flow_value, indep),
                "lambda={lambda} parametric={} independent={indep}",
                sol.flow_value
            );
        }
    }

    #[test]
    fn source_sets_are_nested() {
        let net = nested_net();
        let lambdas: Vec<f64> = (0..=20).map(|k| k as f64 * 0.25).collect();
        let sols = net.solve_grid(&lambdas).expect("grid");
        for w in sols.windows(2) {
            let prev: std::collections::BTreeSet<usize> = w[0].source_set.iter().copied().collect();
            let next: std::collections::BTreeSet<usize> = w[1].source_set.iter().copied().collect();
            assert!(
                prev.is_subset(&next),
                "not nested at lambda {} -> {}: {:?} vs {:?}",
                w[0].lambda,
                w[1].lambda,
                w[0].source_set,
                w[1].source_set
            );
        }
        // sanity: low lambda -> just the source; high lambda -> source side grows.
        assert_eq!(sols.first().expect("first").source_set, vec![0]);
        assert!(sols.last().expect("last").source_set.len() >= 3);
    }

    #[test]
    fn constant_capacities_give_constant_flow() {
        let mut net = ParametricMaxFlow::new(4, 0, 3).expect("new");
        net.add_arc(0, 1, 3.0, 0.0).expect("e");
        net.add_arc(0, 2, 2.0, 0.0).expect("e");
        net.add_arc(1, 3, 3.0, 0.0).expect("e");
        net.add_arc(2, 3, 2.0, 0.0).expect("e");
        let lambdas: Vec<f64> = (-5..=5).map(|k| k as f64).collect();
        let sols = net.solve_grid(&lambdas).expect("grid");
        for s in &sols {
            assert!(
                approx(s.flow_value, 5.0),
                "flow {} != 5 at {}",
                s.flow_value,
                s.lambda
            );
        }
        // identical source sets across all lambda
        let first = &sols[0].source_set;
        for s in &sols {
            assert_eq!(&s.source_set, first);
        }
    }

    #[test]
    fn analytic_two_node_min_formula() {
        // s=0 -> a=1 with cap = lambda; a=1 -> t=2 with cap = C.
        // max flow = min(max(0, lambda), C).
        let other = 2.5;
        let mut net = ParametricMaxFlow::new(3, 0, 2).expect("new");
        net.add_arc(0, 1, 0.0, 1.0).expect("s->a");
        net.add_arc(1, 2, other, 0.0).expect("a->t");
        for k in 0..=12 {
            let lambda = k as f64 * 0.5;
            let sol = net.solve_at(lambda).expect("solve");
            let expected = lambda.max(0.0).min(other);
            assert!(
                approx(sol.flow_value, expected),
                "lambda={lambda}: got {} expected {expected}",
                sol.flow_value
            );
        }
    }

    #[test]
    fn breakpoint_detected_for_nested_example() {
        let net = nested_net();
        // flow = min(2 lambda, 2): breakpoint at lambda = 1 where the cut changes
        // from the source side {0} to {0,1,2}.
        let bps = net.find_breakpoints(0.0, 3.0).expect("breakpoints");
        assert!(!bps.is_empty(), "expected at least one breakpoint");
        let near_one = bps.iter().any(|b| (b.lambda - 1.0).abs() < 1e-4);
        assert!(near_one, "expected a breakpoint near lambda=1, got {bps:?}");
    }

    #[test]
    fn rejects_non_monotone_arcs() {
        let mut net = ParametricMaxFlow::new(4, 0, 3).expect("new");
        // source arc with negative slope -> error
        assert!(net.add_arc(0, 1, 1.0, -1.0).is_err());
        // sink arc with positive slope -> error
        assert!(net.add_arc(1, 3, 1.0, 1.0).is_err());
        // interior arc with non-zero slope -> error
        assert!(net.add_arc(1, 2, 1.0, 0.5).is_err());
    }

    #[test]
    fn rejects_bad_construction() {
        assert!(ParametricMaxFlow::new(0, 0, 0).is_err());
        assert!(ParametricMaxFlow::new(3, 0, 0).is_err());
        assert!(ParametricMaxFlow::new(3, 5, 1).is_err());
        // mismatched capacity dimensions
        let from = [0usize, 1];
        let to = [1usize, 2];
        let base = [1.0];
        let slope = [0.0, 0.0];
        assert!(
            ParametricMaxFlow::from_linear_capacities(3, 0, 2, &from, &to, &base, &slope).is_err()
        );
    }

    #[test]
    fn from_linear_capacities_builds() {
        let from = [0usize, 0, 1, 2];
        let to = [1usize, 2, 3, 3];
        let base = [0.0, 0.0, 1.0, 1.0];
        let slope = [1.0, 1.0, 0.0, 0.0];
        let net = ParametricMaxFlow::from_linear_capacities(4, 0, 3, &from, &to, &base, &slope)
            .expect("build");
        assert_eq!(net.arcs().len(), 4);
        let sol = net.solve_at(0.5).expect("solve");
        assert!(approx(sol.flow_value, 1.0));
    }
}
