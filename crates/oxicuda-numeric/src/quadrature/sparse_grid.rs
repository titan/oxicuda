//! Smolyak sparse-grid quadrature on `[-1, 1]^d` (or a mapped box).
//!
//! High-dimensional integration by the Smolyak combination of nested one-
//! dimensional Clenshaw-Curtis rules (Smolyak 1963; Gerstner & Griebel 1998).
//! The full tensor product of a level-`L` 1D rule needs `m^d` points, which is
//! intractable for moderate `d`. The Smolyak construction keeps only the
//! "important" mixed-order combinations:
//!
//! ```text
//! A(L, d) = Σ_{max(d, L-d+1) ≤ |i| ≤ L} (-1)^{L-|i|} · C(d-1, L-|i|) · (U_{i_1} ⊗ … ⊗ U_{i_d})
//! ```
//!
//! where each `i_k ≥ 1`, `|i| = Σ_k i_k`, and `U_i` is the one-dimensional
//! Clenshaw-Curtis rule with `m(i)` points using the *nested* growth
//!
//! ```text
//! m(1) = 1,   m(i) = 2^{i-1} + 1   (i ≥ 2).
//! ```
//!
//! Nestedness (`points(i) ⊂ points(i+1)`) means the same abscissa appears in
//! many tensor terms; the construction here accumulates the (signed) weights
//! into a single deduplicated point→weight table, so each distinct abscissa is
//! evaluated exactly once.
//!
//! The grid is built once and reused: [`SparseGrid::level`] builds the table of
//! `(point, weight)` pairs on `[-1, 1]^d`, and [`SparseGrid::integrate`] maps it
//! to an arbitrary axis-aligned box and contracts against a function.

use crate::error::{NumericError, NumericResult};
use crate::quadrature::clenshaw_curtis::clenshaw_curtis_nodes;
use std::collections::HashMap;

/// Number of points of the nested Clenshaw-Curtis rule at 1D level `i ≥ 1`.
///
/// `m(1) = 1`, `m(i) = 2^{i-1} + 1` for `i ≥ 2`.
#[must_use]
pub fn cc_level_point_count(i: usize) -> usize {
    match i {
        0 => 0,
        1 => 1,
        _ => (1usize << (i - 1)) + 1,
    }
}

/// One-dimensional nested Clenshaw-Curtis rule at level `i` on `[-1, 1]`.
///
/// Level 1 is the single midpoint `{0}` with weight `2` (the whole interval
/// length). Higher levels delegate to the order-`(m(i) - 1)` Clenshaw-Curtis
/// rule, which is the standard nested abscissa set `cos(k·π / (m-1))`.
fn cc_rule_1d(i: usize) -> NumericResult<(Vec<f64>, Vec<f64>)> {
    match i {
        0 => Err(NumericError::InvalidParameter(
            "1D Clenshaw-Curtis level must be ≥ 1".into(),
        )),
        1 => Ok((vec![0.0_f64], vec![2.0_f64])),
        _ => {
            // m(i) = 2^{i-1} + 1 points  ⇒  Clenshaw-Curtis order n = m - 1.
            let m = cc_level_point_count(i);
            clenshaw_curtis_nodes(m - 1)
        }
    }
}

/// Integer binomial coefficient `C(n, k)` (returns 0 for `k > n`).
fn binomial(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut num: u64 = 1;
    let mut den: u64 = 1;
    for j in 0..k {
        num = num.saturating_mul((n - j) as u64);
        den = den.saturating_mul((j + 1) as u64);
    }
    num / den
}

/// Enumerate all multi-indices `i ∈ ℕ^d` with each `i_k ≥ 1` and `Σ i_k = total`.
fn compositions(d: usize, total: usize, out: &mut Vec<Vec<usize>>) {
    if total < d {
        return;
    }
    let mut idx = vec![1usize; d];
    // Distribute the remaining (total - d) units across the d slots.
    fn rec(pos: usize, remaining: usize, d: usize, idx: &mut [usize], out: &mut Vec<Vec<usize>>) {
        if pos == d - 1 {
            idx[pos] = 1 + remaining;
            out.push(idx.to_vec());
            return;
        }
        for give in 0..=remaining {
            idx[pos] = 1 + give;
            rec(pos + 1, remaining - give, d, idx, out);
        }
    }
    rec(0, total - d, d, &mut idx, out);
}

/// A pre-built Smolyak sparse grid on `[-1, 1]^d`: deduplicated points and the
/// associated Smolyak quadrature weights.
#[derive(Debug, Clone)]
pub struct SparseGrid {
    /// Dimension of the grid.
    dim: usize,
    /// Smolyak level `L ≥ d`.
    level: usize,
    /// Distinct abscissae on `[-1, 1]^d`, one row of length `dim` per point.
    points: Vec<Vec<f64>>,
    /// Smolyak weight associated with each point (on `[-1, 1]^d`).
    weights: Vec<f64>,
}

/// Key used to deduplicate points by quantising each coordinate.
///
/// Nested Clenshaw-Curtis abscissae coincide exactly in exact arithmetic but
/// can differ by rounding across the different `cos(kπ/m)` evaluations that
/// produce them. Quantising to `1e-12` collapses coincident points robustly.
fn quantise(coords: &[f64]) -> Vec<i64> {
    coords
        .iter()
        .map(|&c| (c / 1.0e-12).round() as i64)
        .collect()
}

impl SparseGrid {
    /// Build the Smolyak sparse grid of dimension `dim` at level `level`.
    ///
    /// # Errors
    /// Returns [`NumericError::InvalidParameter`] if `dim == 0` or
    /// `level < dim` (the Smolyak sum is empty below `L = d`).
    pub fn level(dim: usize, level: usize) -> NumericResult<Self> {
        if dim == 0 {
            return Err(NumericError::InvalidParameter(
                "sparse-grid dimension must be ≥ 1".into(),
            ));
        }
        if level < dim {
            return Err(NumericError::InvalidParameter(
                "sparse-grid level must satisfy L ≥ d".into(),
            ));
        }

        // Accumulate signed weights into a point table keyed by quantised coords.
        let mut table: HashMap<Vec<i64>, (Vec<f64>, f64)> = HashMap::new();

        let q_lo = if level >= dim {
            level.saturating_sub(dim) + 1
        } else {
            1
        };
        let q_lo = q_lo.max(dim);

        for q in q_lo..=level {
            // Smolyak combination coefficient (-1)^{L-|i|} C(d-1, L-|i|).
            let diff = level - q;
            let coeff_mag = binomial(dim - 1, diff);
            if coeff_mag == 0 {
                continue;
            }
            let sign = if diff % 2 == 0 { 1.0 } else { -1.0 };
            let combo = sign * coeff_mag as f64;

            // All multi-indices i with |i| = q, i_k ≥ 1.
            let mut multi: Vec<Vec<usize>> = Vec::new();
            compositions(dim, q, &mut multi);

            for idx in &multi {
                // Build the 1D rules for this multi-index.
                let mut rules_1d: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(dim);
                for &ik in idx {
                    rules_1d.push(cc_rule_1d(ik)?);
                }
                // Tensor product of the 1D rules → contributions to the table.
                let counts: Vec<usize> = rules_1d.iter().map(|(p, _)| p.len()).collect();
                let total_pts: usize = counts.iter().product();
                let mut coord = vec![0.0_f64; dim];
                let mut sel = vec![0usize; dim];
                for _ in 0..total_pts {
                    let mut w = combo;
                    for k in 0..dim {
                        coord[k] = rules_1d[k].0[sel[k]];
                        w *= rules_1d[k].1[sel[k]];
                    }
                    let key = quantise(&coord);
                    let entry = table.entry(key).or_insert_with(|| (coord.clone(), 0.0));
                    entry.1 += w;
                    // Mixed-radix increment over selection indices.
                    let mut k = 0;
                    while k < dim {
                        sel[k] += 1;
                        if sel[k] < counts[k] {
                            break;
                        }
                        sel[k] = 0;
                        k += 1;
                    }
                }
            }
        }

        let mut points = Vec::with_capacity(table.len());
        let mut weights = Vec::with_capacity(table.len());
        for (_, (coord, w)) in table {
            points.push(coord);
            weights.push(w);
        }

        Ok(Self {
            dim,
            level,
            points,
            weights,
        })
    }

    /// Dimension of the grid.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Smolyak level of the grid.
    #[must_use]
    pub fn smolyak_level(&self) -> usize {
        self.level
    }

    /// Number of distinct points in the grid (each evaluated once).
    #[must_use]
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Distinct grid points on `[-1, 1]^d`.
    #[must_use]
    pub fn points(&self) -> &[Vec<f64>] {
        &self.points
    }

    /// Smolyak weights on `[-1, 1]^d`, aligned with [`points`](Self::points).
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Integrate `f` over `[-1, 1]^d` using the sparse grid.
    ///
    /// # Errors
    /// Propagates any error returned by `f`.
    pub fn integrate_unit<F>(&self, f: F) -> NumericResult<f64>
    where
        F: Fn(&[f64]) -> NumericResult<f64>,
    {
        let mut acc = 0.0_f64;
        for (p, &w) in self.points.iter().zip(self.weights.iter()) {
            acc += w * f(p)?;
        }
        Ok(acc)
    }

    /// Integrate `f` over the axis-aligned box `∏_k [lo_k, hi_k]`.
    ///
    /// Each coordinate is affinely mapped from `[-1, 1]`; the weights pick up the
    /// product of half-widths (the Jacobian of the map).
    ///
    /// # Errors
    /// Returns [`NumericError::DimensionMismatch`] if `lo`/`hi` lengths disagree
    /// with the grid dimension, and propagates any error returned by `f`.
    pub fn integrate<F>(&self, f: F, lo: &[f64], hi: &[f64]) -> NumericResult<f64>
    where
        F: Fn(&[f64]) -> NumericResult<f64>,
    {
        if lo.len() != self.dim || hi.len() != self.dim {
            return Err(NumericError::DimensionMismatch {
                a: lo.len(),
                b: hi.len(),
            });
        }
        let mut jac = 1.0_f64;
        for k in 0..self.dim {
            jac *= 0.5 * (hi[k] - lo[k]);
        }
        let mut x = vec![0.0_f64; self.dim];
        let mut acc = 0.0_f64;
        for (p, &w) in self.points.iter().zip(self.weights.iter()) {
            for k in 0..self.dim {
                let mid = 0.5 * (hi[k] + lo[k]);
                let half = 0.5 * (hi[k] - lo[k]);
                x[k] = mid + half * p[k];
            }
            acc += w * f(&x)?;
        }
        Ok(jac * acc)
    }
}

/// Convenience: build the level-`L` grid and integrate `f` over `[-1, 1]^d`.
///
/// # Errors
/// Propagates grid-construction errors and any error from `f`.
pub fn smolyak_integrate_unit<F>(dim: usize, level: usize, f: F) -> NumericResult<f64>
where
    F: Fn(&[f64]) -> NumericResult<f64>,
{
    SparseGrid::level(dim, level)?.integrate_unit(f)
}

/// Convenience: build the level-`L` grid and integrate `f` over a box.
///
/// # Errors
/// Propagates grid-construction errors and any error from `f`.
pub fn smolyak_integrate<F>(level: usize, lo: &[f64], hi: &[f64], f: F) -> NumericResult<f64>
where
    F: Fn(&[f64]) -> NumericResult<f64>,
{
    if lo.len() != hi.len() {
        return Err(NumericError::DimensionMismatch {
            a: lo.len(),
            b: hi.len(),
        });
    }
    let grid = SparseGrid::level(lo.len(), level)?;
    grid.integrate(f, lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference full-tensor Clenshaw-Curtis count per axis at level `L`.
    fn tensor_axis_count(level: usize) -> usize {
        cc_level_point_count(level)
    }

    #[test]
    fn weights_sum_to_volume_2d() {
        // Σ weights on [-1,1]^d must equal the volume 2^d (integral of f ≡ 1).
        for level in 2..=6 {
            let grid = SparseGrid::level(2, level).expect("grid");
            let s: f64 = grid.weights().iter().sum();
            assert!((s - 4.0).abs() < 1.0e-9, "level {level}: sum={s}");
        }
    }

    #[test]
    fn weights_sum_to_volume_3d_4d() {
        let g3 = SparseGrid::level(3, 5).expect("grid");
        let s3: f64 = g3.weights().iter().sum();
        assert!((s3 - 8.0).abs() < 1.0e-9, "3d sum={s3}");

        let g4 = SparseGrid::level(4, 6).expect("grid");
        let s4: f64 = g4.weights().iter().sum();
        assert!((s4 - 16.0).abs() < 1.0e-9, "4d sum={s4}");
    }

    #[test]
    fn integrates_constant() {
        let grid = SparseGrid::level(3, 5).expect("grid");
        let v = grid.integrate_unit(|_x| Ok(1.0)).expect("integrate");
        assert!((v - 8.0).abs() < 1.0e-9, "v={v}");
    }

    #[test]
    fn integrates_low_degree_polynomials_exactly() {
        // A level-L Smolyak grid (nested CC) is exact for low total degree.
        // Test several monomials x^a y^b z^c with small a+b+c.
        // Classic Barthelmann-Novak-Ritter (2000) result: the level-L nested
        // Clenshaw-Curtis Smolyak rule in d dimensions integrates every
        // polynomial of TOTAL degree ≤ 2(L - d) + 1 exactly. With d = 3 and
        // L = 6 the guaranteed total degree is 7; exercise monomials up to that.
        let dim = 3usize;
        let level = 6usize;
        let guaranteed = 2 * (level - dim) + 1; // = 7
        let grid = SparseGrid::level(dim, level).expect("grid");
        // ∫_{-1}^{1} x^p dx = 0 (odd p), 2/(p+1) (even p).
        let mono = |p: usize| -> f64 {
            if p % 2 == 1 {
                0.0
            } else {
                2.0 / (p as f64 + 1.0)
            }
        };
        // Every monomial x^a y^b z^c with a + b + c ≤ guaranteed must be exact.
        for a in 0..=guaranteed {
            for b in 0..=(guaranteed - a) {
                for c in 0..=(guaranteed - a - b) {
                    let exact = mono(a) * mono(b) * mono(c);
                    let v = grid
                        .integrate_unit(move |x| {
                            Ok(x[0].powi(a as i32) * x[1].powi(b as i32) * x[2].powi(c as i32))
                        })
                        .expect("integrate");
                    assert!(
                        (v - exact).abs() < 1.0e-9,
                        "x^{a} y^{b} z^{c} (total {}): got {v}, want {exact}",
                        a + b + c
                    );
                }
            }
        }
    }

    #[test]
    fn far_fewer_points_than_full_tensor_3d_4d() {
        // The whole point of Smolyak: point count ≪ (1D count)^d.
        let level = 5;
        for d in [3usize, 4usize] {
            let grid = SparseGrid::level(d, level).expect("grid");
            let axis = tensor_axis_count(level);
            let full = (axis as u128).pow(d as u32);
            let sparse = grid.num_points() as u128;
            assert!(
                sparse < full,
                "d={d}: sparse {sparse} not < full tensor {full} (axis={axis})"
            );
            // And it must be *dramatically* fewer, not marginally.
            assert!(
                sparse * 4 < full,
                "d={d}: sparse {sparse} not ≪ full {full}"
            );
        }
    }

    #[test]
    fn gaussian_error_decreases_with_level() {
        // ∫_{-1}^{1}^2 exp(-(x²+y²)) dx dy = (∫_{-1}^{1} e^{-t²} dt)².
        // ∫_{-1}^{1} e^{-t²} dt = sqrt(π) erf(1).
        let erf1 = 0.842_700_792_949_714_9_f64; // erf(1)
        let line = std::f64::consts::PI.sqrt() * erf1;
        let exact = line * line;
        let f = |x: &[f64]| -> NumericResult<f64> { Ok((-(x[0] * x[0] + x[1] * x[1])).exp()) };

        let mut prev_err = f64::INFINITY;
        let mut last_err = f64::INFINITY;
        for level in 2..=8 {
            let grid = SparseGrid::level(2, level).expect("grid");
            let v = grid.integrate_unit(f).expect("integrate");
            let err = (v - exact).abs();
            // Non-increasing trend (allow a tiny tolerance for plateau).
            assert!(
                err <= prev_err + 1.0e-12,
                "level {level}: err {err} > prev {prev_err}"
            );
            prev_err = err;
            last_err = err;
        }
        assert!(last_err < 1.0e-8, "final error too large: {last_err}");
    }

    #[test]
    fn d1_reduces_to_clenshaw_curtis() {
        // In 1D, the Smolyak grid at level L is exactly the CC rule with m(L) pts.
        for level in 1..=6 {
            let grid = SparseGrid::level(1, level).expect("grid");
            // Reference: CC rule of the same point count.
            let (ref_nodes, ref_weights) = cc_rule_1d(level).expect("cc");
            assert_eq!(
                grid.num_points(),
                ref_nodes.len(),
                "level {level}: point count differs"
            );
            // Integrate a polynomial that the 1D CC rule integrates exactly and
            // confirm the sparse grid matches the direct CC rule bit-for-bit
            // (up to rounding).
            let f = |x: f64| x.powi(4) - 2.0 * x.powi(2) + 1.0;
            let direct: f64 = ref_nodes
                .iter()
                .zip(ref_weights.iter())
                .map(|(&xn, &wn)| wn * f(xn))
                .sum();
            let viaspar = grid.integrate_unit(|x| Ok(f(x[0]))).expect("integrate");
            assert!(
                (viaspar - direct).abs() < 1.0e-10,
                "level {level}: sparse {viaspar} vs CC {direct}"
            );
        }
    }

    #[test]
    fn nested_points_reused_no_duplicates() {
        // The deduplicated table must contain strictly fewer points than the
        // naive sum of all tensor-term sizes (proof that nested points coincide
        // and are counted once).
        let dim = 2;
        let level = 5;
        let grid = SparseGrid::level(dim, level).expect("grid");

        // Naive count: sum over all contributing multi-indices of ∏ m(i_k).
        let mut naive = 0usize;
        let q_lo = (level - dim + 1).max(dim);
        for q in q_lo..=level {
            if binomial(dim - 1, level - q) == 0 {
                continue;
            }
            let mut multi: Vec<Vec<usize>> = Vec::new();
            compositions(dim, q, &mut multi);
            for idx in &multi {
                let prod: usize = idx.iter().map(|&ik| cc_level_point_count(ik)).product();
                naive += prod;
            }
        }
        assert!(
            grid.num_points() < naive,
            "dedup {} not < naive {}",
            grid.num_points(),
            naive
        );

        // Also verify there are genuinely no duplicate coordinates remaining.
        let mut seen = std::collections::HashSet::new();
        for p in grid.points() {
            let key = quantise(p);
            assert!(seen.insert(key), "duplicate point survived dedup");
        }
    }

    #[test]
    fn mapped_box_integration() {
        // ∫_0^2 ∫_0^3 (x + y) dx dy = ∫_0^2 (3·x? ) ... compute directly:
        // ∫_0^2 ∫_0^3 (x+y) dy dx = ∫_0^2 [3x + 9/2] dx = [3/2 x² + 9/2 x]_0^2
        //  = 6 + 9 = 15.
        let grid = SparseGrid::level(2, 4).expect("grid");
        let v = grid
            .integrate(|p| Ok(p[0] + p[1]), &[0.0, 0.0], &[2.0, 3.0])
            .expect("integrate");
        assert!((v - 15.0).abs() < 1.0e-9, "v={v}");
    }

    #[test]
    fn convenience_helpers_agree() {
        let f = |x: &[f64]| -> NumericResult<f64> { Ok(x[0] * x[0] + x[1] * x[1]) };
        let viahelper = smolyak_integrate_unit(2, 5, f).expect("helper");
        let grid = SparseGrid::level(2, 5).expect("grid");
        let direct = grid.integrate_unit(f).expect("direct");
        assert!((viahelper - direct).abs() < 1.0e-12);

        let box_helper =
            smolyak_integrate(4, &[0.0, 0.0], &[1.0, 1.0], |p| Ok(p[0] + p[1])).expect("box");
        // ∫_0^1 ∫_0^1 (x+y) = 1.
        assert!((box_helper - 1.0).abs() < 1.0e-9, "box={box_helper}");
    }

    #[test]
    fn invalid_parameters() {
        assert!(matches!(
            SparseGrid::level(0, 3),
            Err(NumericError::InvalidParameter(_))
        ));
        assert!(matches!(
            SparseGrid::level(3, 2),
            Err(NumericError::InvalidParameter(_))
        ));
        let grid = SparseGrid::level(2, 3).expect("grid");
        assert!(matches!(
            grid.integrate(|_p| Ok(0.0), &[0.0], &[1.0, 2.0]),
            Err(NumericError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn level_count_formula() {
        assert_eq!(cc_level_point_count(1), 1);
        assert_eq!(cc_level_point_count(2), 3);
        assert_eq!(cc_level_point_count(3), 5);
        assert_eq!(cc_level_point_count(4), 9);
        assert_eq!(cc_level_point_count(5), 17);
    }
}
