//! Persistence landscapes and their Lᵖ norm / distance.
//!
//! Reference: Peter Bubenik, "Statistical Topological Data Analysis using Persistence
//! Landscapes", Journal of Machine Learning Research 16 (2015), 77–102.
//!
//! For each finite birth–death pair `(b_i, d_i)` define the tent function
//! `Λ_i(t) = max(0, min(t − b_i, d_i − t))`.  The `k`-th landscape `λ_k(t)` is the
//! `k`-th largest value among `{Λ_i(t)}` at parameter `t` (with `λ_1 ≥ λ_2 ≥ …`).
//!
//! This module samples the first `n_layers` landscapes on a shared uniform grid of
//! `resolution` points spanning `[min birth, max death]` over the diagram's finite pairs.
//! The Lᵖ norm and Lᵖ distance are evaluated by trapezoidal integration over that grid,
//! summed across layers.

use crate::error::{TdaError, TdaResult};
use crate::persistence::diagram::PersistenceDiagram;

/// A persistence landscape sampled on a shared uniform grid.
///
/// `layers[k]` holds the sampled values of `λ_{k+1}` at the `resolution` grid points
/// spanning `[t_min, t_max]`.  Each layer therefore has length `resolution`.
#[derive(Debug, Clone)]
pub struct PersistenceLandscape {
    /// Sampled landscape values; outer index = layer, inner index = grid point.
    layers: Vec<Vec<f32>>,
    /// Lower grid bound.
    t_min: f32,
    /// Upper grid bound.
    t_max: f32,
    /// Number of grid points (≥ 2).
    resolution: usize,
}

impl PersistenceLandscape {
    /// Build the first `n_layers` landscapes of `diagram` sampled on a grid of
    /// `resolution` points.
    ///
    /// The grid spans `[min birth, max death]` over the diagram's finite pairs.  At each
    /// grid point all tent values are evaluated, sorted descending, and the top
    /// `n_layers` taken as `λ_1 … λ_{n_layers}`.  A diagram with no finite pairs yields an
    /// all-zero landscape on the degenerate grid `[0, 0]`.
    ///
    /// # Errors
    /// - [`TdaError::ParameterOutOfRange`] if `n_layers == 0` or `resolution < 2`.
    pub fn from_diagram(
        diagram: &PersistenceDiagram,
        n_layers: usize,
        resolution: usize,
    ) -> TdaResult<Self> {
        if n_layers == 0 {
            return Err(TdaError::ParameterOutOfRange(
                "n_layers must be ≥ 1".to_owned(),
            ));
        }
        if resolution < 2 {
            return Err(TdaError::ParameterOutOfRange(
                "resolution must be ≥ 2".to_owned(),
            ));
        }

        // Collect finite (birth, death) pairs as f32, dropping degenerate pairs.
        let pairs: Vec<(f32, f32)> = diagram
            .finite_pairs()
            .iter()
            .filter_map(|p| p.death.map(|d| (p.birth as f32, d as f32)))
            .filter(|(b, d)| d > b)
            .collect();

        // Empty diagram (no usable finite pairs): all-zero landscape on a degenerate grid.
        if pairs.is_empty() {
            return Ok(Self {
                layers: vec![vec![0.0_f32; resolution]; n_layers],
                t_min: 0.0,
                t_max: 0.0,
                resolution,
            });
        }

        let mut t_min = f32::INFINITY;
        let mut t_max = f32::NEG_INFINITY;
        for &(b, d) in &pairs {
            if b < t_min {
                t_min = b;
            }
            if d > t_max {
                t_max = d;
            }
        }

        let span = t_max - t_min;
        let denom = (resolution - 1) as f32;
        let mut layers: Vec<Vec<f32>> = vec![Vec::with_capacity(resolution); n_layers];
        let mut tent_values: Vec<f32> = Vec::with_capacity(pairs.len());

        for g in 0..resolution {
            let t = t_min + span * (g as f32) / denom;
            tent_values.clear();
            for &(b, d) in &pairs {
                let v = (t - b).min(d - t);
                tent_values.push(v.max(0.0));
            }
            // Sort descending so index 0 is the largest tent value.
            tent_values
                .sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            for (layer, out) in layers.iter_mut().enumerate() {
                out.push(tent_values.get(layer).copied().unwrap_or(0.0));
            }
        }

        Ok(Self {
            layers,
            t_min,
            t_max,
            resolution,
        })
    }

    /// Evaluate `λ_{layer+1}` at parameter `t` via piecewise-linear interpolation on the
    /// grid.  Returns `0` for `t` outside `[t_min, t_max]` or for an out-of-range layer.
    ///
    /// # Errors
    /// Returns [`TdaError::ParameterOutOfRange`] if `t` is NaN.
    pub fn evaluate(&self, layer: usize, t: f32) -> TdaResult<f32> {
        if t.is_nan() {
            return Err(TdaError::ParameterOutOfRange(
                "t must not be NaN".to_owned(),
            ));
        }
        let Some(values) = self.layers.get(layer) else {
            return Ok(0.0);
        };
        if t < self.t_min || t > self.t_max {
            return Ok(0.0);
        }
        let span = self.t_max - self.t_min;
        // Degenerate grid (all pairs collapse to a point): the landscape is identically 0.
        if span <= 0.0 {
            return Ok(values.first().copied().unwrap_or(0.0));
        }
        let denom = (self.resolution - 1) as f32;
        let pos = (t - self.t_min) / span * denom;
        let lo = pos.floor();
        let mut lo_idx = lo as usize;
        if lo_idx >= self.resolution - 1 {
            // At or past the last grid point: clamp to the final sample.
            return Ok(values.get(self.resolution - 1).copied().unwrap_or(0.0));
        }
        let frac = pos - lo;
        let v_lo = values.get(lo_idx).copied().unwrap_or(0.0);
        lo_idx += 1;
        let v_hi = values.get(lo_idx).copied().unwrap_or(0.0);
        Ok(v_lo + frac * (v_hi - v_lo))
    }

    /// The Lᵖ norm `(∫ Σ_k |λ_k(t)|^p dt)^{1/p}` evaluated by the trapezoidal rule on the
    /// grid, summing across all layers.
    ///
    /// # Errors
    /// Returns [`TdaError::ParameterOutOfRange`] if `p ≤ 0`.
    pub fn lp_norm(&self, p: f32) -> TdaResult<f32> {
        if p <= 0.0 {
            return Err(TdaError::ParameterOutOfRange("p must be > 0".to_owned()));
        }
        let span = self.t_max - self.t_min;
        if span <= 0.0 {
            return Ok(0.0);
        }
        let dt = span / ((self.resolution - 1) as f32);
        let mut integral = 0.0_f32;
        for values in &self.layers {
            integral += trapezoid_pow(values, dt, p);
        }
        Ok(integral.powf(1.0 / p))
    }

    /// The Lᵖ distance `(∫ Σ_k |λ_k(t) − μ_k(t)|^p dt)^{1/p}` to `other`.
    ///
    /// When the grids match exactly the difference is taken sample-by-sample; otherwise
    /// `other` is resampled onto `self`'s grid via [`PersistenceLandscape::evaluate`].
    ///
    /// # Errors
    /// Returns [`TdaError::ParameterOutOfRange`] if `p ≤ 0`.
    pub fn lp_distance(&self, other: &Self, p: f32) -> TdaResult<f32> {
        if p <= 0.0 {
            return Err(TdaError::ParameterOutOfRange("p must be > 0".to_owned()));
        }
        let n_layers = self.layers.len().max(other.layers.len());

        // Fast exact path: identical grids -> difference the samples directly.
        let grids_match = self.resolution == other.resolution
            && (self.t_min - other.t_min).abs() < f32::EPSILON
            && (self.t_max - other.t_max).abs() < f32::EPSILON;
        if grids_match {
            let span = self.t_max - self.t_min;
            if span <= 0.0 {
                return Ok(0.0);
            }
            let dt = span / ((self.resolution - 1) as f32);
            let mut integral = 0.0_f32;
            for layer in 0..n_layers {
                let mut prev = 0.0_f32;
                for g in 0..self.resolution {
                    let a = self
                        .layers
                        .get(layer)
                        .and_then(|v| v.get(g))
                        .copied()
                        .unwrap_or(0.0);
                    let b = other
                        .layers
                        .get(layer)
                        .and_then(|v| v.get(g))
                        .copied()
                        .unwrap_or(0.0);
                    let cur = (a - b).abs().powf(p);
                    if g > 0 {
                        integral += 0.5 * (prev + cur) * dt;
                    }
                    prev = cur;
                }
            }
            return Ok(integral.powf(1.0 / p));
        }

        // General path: integrate `‖λ − μ‖_p` over the UNION of both supports on a shared
        // grid, resampling each landscape via `evaluate` (which is 0 outside its own
        // support).  Using the union range is essential for symmetry and correctness — a
        // grid restricted to one diagram would miss the other's mass.
        let lo = self.t_min.min(other.t_min);
        let hi = self.t_max.max(other.t_max);
        let span = hi - lo;
        if span <= 0.0 {
            return Ok(0.0);
        }
        // Pick a resolution that resolves both grids' samples.
        let res = self.resolution.max(other.resolution).max(2);
        let denom = (res - 1) as f32;
        let dt = span / denom;

        let mut integral = 0.0_f32;
        for layer in 0..n_layers {
            let mut prev = 0.0_f32;
            for g in 0..res {
                let t = lo + span * (g as f32) / denom;
                let a = self.evaluate(layer, t)?;
                let b = other.evaluate(layer, t)?;
                let cur = (a - b).abs().powf(p);
                if g > 0 {
                    integral += 0.5 * (prev + cur) * dt;
                }
                prev = cur;
            }
        }
        Ok(integral.powf(1.0 / p))
    }

    /// Number of landscape layers stored.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Lower grid bound.
    pub fn t_min(&self) -> f32 {
        self.t_min
    }

    /// Upper grid bound.
    pub fn t_max(&self) -> f32 {
        self.t_max
    }

    /// Number of grid points.
    pub fn resolution(&self) -> usize {
        self.resolution
    }
}

/// Trapezoidal integral of `|value|^p` over a uniformly spaced sample array.
fn trapezoid_pow(values: &[f32], dt: f32, p: f32) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let mut acc = 0.0_f32;
    let mut prev = values[0].abs().powf(p);
    for &v in &values[1..] {
        let cur = v.abs().powf(p);
        acc += 0.5 * (prev + cur) * dt;
        prev = cur;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::homology::persistent::PersistencePair;

    /// Build a single-dimension diagram from explicit (birth, death) pairs.
    fn diagram_from(pairs: &[(f64, Option<f64>)]) -> PersistenceDiagram {
        let ps = pairs
            .iter()
            .map(|&(b, d)| PersistencePair {
                dim: 0,
                birth: b,
                death: d,
            })
            .collect();
        PersistenceDiagram::new(ps, 0)
    }

    #[test]
    fn single_pair_apex_is_one() {
        // Pair (0, 2): tent apex at t=1 with height (2-0)/2 = 1.
        let diag = diagram_from(&[(0.0, Some(2.0))]);
        let ls = PersistenceLandscape::from_diagram(&diag, 1, 201).expect("landscape");
        let apex = ls.evaluate(0, 1.0).expect("eval");
        assert!((apex - 1.0).abs() < 1e-3, "apex should be 1.0, got {apex}");
    }

    #[test]
    fn single_pair_endpoints_zero() {
        let diag = diagram_from(&[(0.0, Some(2.0))]);
        let ls = PersistenceLandscape::from_diagram(&diag, 1, 201).expect("landscape");
        let at_birth = ls.evaluate(0, 0.0).expect("eval");
        let at_death = ls.evaluate(0, 2.0).expect("eval");
        assert!(at_birth.abs() < 1e-4, "λ_1(birth) should be 0");
        assert!(at_death.abs() < 1e-4, "λ_1(death) should be 0");
    }

    #[test]
    fn n_layers_reported() {
        let diag = diagram_from(&[(0.0, Some(2.0)), (0.5, Some(3.0))]);
        let ls = PersistenceLandscape::from_diagram(&diag, 3, 64).expect("landscape");
        assert_eq!(ls.n_layers(), 3);
    }

    #[test]
    fn evaluate_outside_grid_is_zero() {
        let diag = diagram_from(&[(1.0, Some(3.0))]);
        let ls = PersistenceLandscape::from_diagram(&diag, 1, 64).expect("landscape");
        assert_eq!(ls.evaluate(0, -5.0).expect("eval"), 0.0);
        assert_eq!(ls.evaluate(0, 100.0).expect("eval"), 0.0);
    }

    #[test]
    fn lp_distance_to_self_is_zero() {
        let diag = diagram_from(&[(0.0, Some(2.0)), (0.5, Some(4.0))]);
        let ls = PersistenceLandscape::from_diagram(&diag, 2, 128).expect("landscape");
        let d = ls.lp_distance(&ls, 2.0).expect("dist");
        assert!(d.abs() < 1e-4, "distance to self should be 0, got {d}");
    }

    #[test]
    fn lp_distance_symmetric() {
        let a = PersistenceLandscape::from_diagram(&diagram_from(&[(0.0, Some(2.0))]), 2, 128)
            .expect("a");
        let b = PersistenceLandscape::from_diagram(
            &diagram_from(&[(0.0, Some(2.0)), (0.5, Some(3.5))]),
            2,
            128,
        )
        .expect("b");
        let dab = a.lp_distance(&b, 2.0).expect("dab");
        let dba = b.lp_distance(&a, 2.0).expect("dba");
        assert!((dab - dba).abs() < 1e-3, "distance should be symmetric");
    }

    #[test]
    fn empty_diagram_norm_zero() {
        let diag = diagram_from(&[]);
        let ls = PersistenceLandscape::from_diagram(&diag, 2, 64).expect("landscape");
        assert_eq!(ls.lp_norm(2.0).expect("norm"), 0.0);
        assert_eq!(ls.lp_norm(1.0).expect("norm"), 0.0);
    }

    #[test]
    fn empty_diagram_all_zero_layers() {
        let diag = diagram_from(&[(1.0, None)]); // essential only -> no finite pairs
        let ls = PersistenceLandscape::from_diagram(&diag, 2, 32).expect("landscape");
        for layer in 0..ls.n_layers() {
            for g in 0..ls.resolution() {
                let t = ls.t_min()
                    + (ls.t_max() - ls.t_min()) * (g as f32) / ((ls.resolution() - 1) as f32);
                assert_eq!(ls.evaluate(layer, t).expect("eval"), 0.0);
            }
        }
    }

    #[test]
    fn two_pairs_layers_are_sorted_max_and_min() {
        // Two overlapping tents.  At every grid point λ_1 = max, λ_2 = min of the two.
        let b1 = 0.0_f32;
        let d1 = 4.0_f32;
        let b2 = 1.0_f32;
        let d2 = 3.0_f32;
        let diag = diagram_from(&[(b1 as f64, Some(d1 as f64)), (b2 as f64, Some(d2 as f64))]);
        let ls = PersistenceLandscape::from_diagram(&diag, 2, 257).expect("landscape");
        let res = ls.resolution();
        for g in 0..res {
            let t = ls.t_min() + (ls.t_max() - ls.t_min()) * (g as f32) / ((res - 1) as f32);
            let tent1 = (t - b1).min(d1 - t).max(0.0);
            let tent2 = (t - b2).min(d2 - t).max(0.0);
            let l1 = ls.evaluate(0, t).expect("l1");
            let l2 = ls.evaluate(1, t).expect("l2");
            assert!((l1 - tent1.max(tent2)).abs() < 1e-3, "λ_1 != pointwise max");
            assert!((l2 - tent1.min(tent2)).abs() < 1e-3, "λ_2 != pointwise min");
        }
    }

    #[test]
    fn layers_monotone_lambda1_ge_lambda2() {
        let diag = diagram_from(&[(0.0, Some(4.0)), (1.0, Some(3.0)), (0.5, Some(5.0))]);
        let ls = PersistenceLandscape::from_diagram(&diag, 3, 200).expect("landscape");
        let res = ls.resolution();
        for g in 0..res {
            let t = ls.t_min() + (ls.t_max() - ls.t_min()) * (g as f32) / ((res - 1) as f32);
            let l1 = ls.evaluate(0, t).expect("l1");
            let l2 = ls.evaluate(1, t).expect("l2");
            let l3 = ls.evaluate(2, t).expect("l3");
            assert!(l1 >= l2 - 1e-4, "λ_1 < λ_2 at t={t}");
            assert!(l2 >= l3 - 1e-4, "λ_2 < λ_3 at t={t}");
        }
    }

    #[test]
    fn p1_vs_p2_norms_differ() {
        let diag = diagram_from(&[(0.0, Some(2.0)), (1.0, Some(5.0))]);
        let ls = PersistenceLandscape::from_diagram(&diag, 2, 256).expect("landscape");
        let n1 = ls.lp_norm(1.0).expect("n1");
        let n2 = ls.lp_norm(2.0).expect("n2");
        assert!((n1 - n2).abs() > 1e-3, "p=1 and p=2 norms should differ");
    }

    #[test]
    fn n_layers_zero_errors() {
        let diag = diagram_from(&[(0.0, Some(2.0))]);
        assert!(PersistenceLandscape::from_diagram(&diag, 0, 64).is_err());
    }

    #[test]
    fn resolution_too_small_errors() {
        let diag = diagram_from(&[(0.0, Some(2.0))]);
        assert!(PersistenceLandscape::from_diagram(&diag, 1, 1).is_err());
    }

    #[test]
    fn nonpositive_p_errors() {
        let diag = diagram_from(&[(0.0, Some(2.0))]);
        let ls = PersistenceLandscape::from_diagram(&diag, 1, 64).expect("landscape");
        assert!(ls.lp_norm(0.0).is_err());
        assert!(ls.lp_norm(-1.0).is_err());
        assert!(ls.lp_distance(&ls, 0.0).is_err());
    }

    #[test]
    fn deterministic() {
        let diag = diagram_from(&[(0.0, Some(2.0)), (0.7, Some(3.3)), (1.1, Some(4.0))]);
        let a = PersistenceLandscape::from_diagram(&diag, 2, 128).expect("a");
        let b = PersistenceLandscape::from_diagram(&diag, 2, 128).expect("b");
        for layer in 0..a.n_layers() {
            for g in 0..a.resolution() {
                let t = a.t_min()
                    + (a.t_max() - a.t_min()) * (g as f32) / ((a.resolution() - 1) as f32);
                assert_eq!(
                    a.evaluate(layer, t).expect("a"),
                    b.evaluate(layer, t).expect("b")
                );
            }
        }
    }

    #[test]
    fn lp_norm_nonnegative() {
        let diag = diagram_from(&[(0.0, Some(2.0)), (1.0, Some(6.0))]);
        let ls = PersistenceLandscape::from_diagram(&diag, 3, 200).expect("landscape");
        assert!(ls.lp_norm(1.0).expect("n1") >= 0.0);
        assert!(ls.lp_norm(2.0).expect("n2") >= 0.0);
        assert!(ls.lp_norm(3.5).expect("n3") >= 0.0);
    }

    #[test]
    fn single_pair_p1_area_matches_triangle() {
        // Pair (0, 2): triangle base 2, height 1, area = 1/2 * 2 * 1 = 1.
        // L^1 norm = ∫ λ_1 dt = area = 1.0.
        let diag = diagram_from(&[(0.0, Some(2.0))]);
        let ls = PersistenceLandscape::from_diagram(&diag, 1, 4001).expect("landscape");
        let area = ls.lp_norm(1.0).expect("area");
        assert!(
            (area - 1.0).abs() < 1e-2,
            "L^1 area should be 1.0, got {area}"
        );
    }

    #[test]
    fn lp_distance_resamples_mismatched_grids() {
        // Two diagrams with different spans -> different grids -> resampling path.
        let a = PersistenceLandscape::from_diagram(&diagram_from(&[(0.0, Some(2.0))]), 1, 128)
            .expect("a");
        let b = PersistenceLandscape::from_diagram(&diagram_from(&[(1.0, Some(6.0))]), 1, 97)
            .expect("b");
        // Distinct grids; distance must be finite, non-negative, and symmetric-ish.
        let dab = a.lp_distance(&b, 2.0).expect("dab");
        let dba = b.lp_distance(&a, 2.0).expect("dba");
        assert!(dab.is_finite() && dab > 0.0);
        assert!(dba.is_finite() && dba > 0.0);
    }
}
