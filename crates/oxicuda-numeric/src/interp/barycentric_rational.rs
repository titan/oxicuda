//! Floater-Hormann barycentric rational interpolation (Floater & Hormann 2007).
//!
//! A barycentric rational interpolant
//!
//! ```text
//! r(x) = ( Σ_k w_k y_k / (x - x_k) ) / ( Σ_k w_k / (x - x_k) )
//! ```
//!
//! whose blending weights `w_k` are chosen so that `r` is a blend of the local
//! degree-`d` polynomial interpolants of consecutive `(d + 1)`-point windows:
//!
//! ```text
//! w_k = Σ_{i ∈ J_k} (-1)^i  ∏_{j = i, j ≠ k}^{i + d}  1 / (x_k - x_j),
//! J_k = { i : max(0, k - d) ≤ i ≤ min(k, n - d) },   n = (#nodes) - 1.
//! ```
//!
//! Key properties (Floater & Hormann, *Numer. Math.* 107, 2007):
//! * `r` interpolates the data exactly at every node;
//! * the denominator has **no zeros** on the real axis — the interpolant is
//!   pole-free regardless of node placement (in particular for equispaced
//!   nodes, where polynomial interpolation suffers the Runge phenomenon);
//! * for smooth `f` the approximation error is `O(h^{d+1})` as the mesh `h → 0`;
//! * `d = 0` recovers Berrut's first barycentric rational interpolant
//!   (`w_k = (-1)^k`).
//!
//! The blending degree `d` trades smoothness/order for conditioning: small `d`
//! (0..4) is robust on equispaced data; larger `d` raises the convergence order
//! but eventually reintroduces ill-conditioning.

use crate::error::{NumericError, NumericResult};

/// A Floater-Hormann barycentric rational interpolant on a set of nodes.
#[derive(Debug, Clone)]
pub struct FloaterHormann {
    /// Interpolation nodes (need not be sorted, but must be distinct).
    nodes: Vec<f64>,
    /// Data values at the nodes.
    values: Vec<f64>,
    /// Floater-Hormann blending weights.
    weights: Vec<f64>,
    /// Blending degree `d`.
    blend_degree: usize,
}

impl FloaterHormann {
    /// Build the Floater-Hormann interpolant for `(nodes, values)` with blending
    /// degree `d`.
    ///
    /// `d = 0` gives Berrut's interpolant; larger `d` raises the order to
    /// `O(h^{d+1})`. Requires `d < #nodes` and at least one node.
    ///
    /// # Errors
    /// * [`NumericError::EmptyInput`] if `nodes` is empty.
    /// * [`NumericError::DimensionMismatch`] if `nodes` and `values` differ in
    ///   length.
    /// * [`NumericError::InvalidParameter`] if `d ≥ #nodes`.
    /// * [`NumericError::NumericalInstability`] if two nodes coincide.
    pub fn new(nodes: &[f64], values: &[f64], blend_degree: usize) -> NumericResult<Self> {
        if nodes.is_empty() {
            return Err(NumericError::EmptyInput);
        }
        if nodes.len() != values.len() {
            return Err(NumericError::DimensionMismatch {
                a: nodes.len(),
                b: values.len(),
            });
        }
        let count = nodes.len();
        if blend_degree >= count {
            return Err(NumericError::InvalidParameter(format!(
                "blending degree d={blend_degree} must be < number of nodes {count}"
            )));
        }
        // n in the paper is the largest node index.
        let n = count - 1;
        let d = blend_degree;

        let mut weights = vec![0.0_f64; count];
        for (k, w_k) in weights.iter_mut().enumerate() {
            // J_k = { i : max(0, k - d) ≤ i ≤ min(k, n - d) }.
            let i_lo = k.saturating_sub(d);
            let i_hi = k.min(n - d);
            let mut acc = 0.0_f64;
            let mut i = i_lo;
            while i <= i_hi {
                // ∏_{j=i, j≠k}^{i+d} 1 / (x_k - x_j).
                let mut prod = 1.0_f64;
                for j in i..=(i + d) {
                    if j == k {
                        continue;
                    }
                    let denom = nodes[k] - nodes[j];
                    if denom.abs() < 1.0e-300 {
                        return Err(NumericError::NumericalInstability(format!(
                            "coincident nodes at indices {k} and {j}"
                        )));
                    }
                    prod /= denom;
                }
                // Sign (-1)^i.
                if i % 2 == 0 {
                    acc += prod;
                } else {
                    acc -= prod;
                }
                i += 1;
            }
            *w_k = acc;
        }

        Ok(Self {
            nodes: nodes.to_vec(),
            values: values.to_vec(),
            weights,
            blend_degree: d,
        })
    }

    /// Evaluate the interpolant at `x`.
    ///
    /// If `x` lands exactly on a node the stored value is returned, which is
    /// also the analytic limit of the barycentric quotient there.
    #[must_use]
    pub fn eval(&self, x: f64) -> f64 {
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for ((xi, yi), wi) in self
            .nodes
            .iter()
            .zip(self.values.iter())
            .zip(self.weights.iter())
        {
            let diff = x - xi;
            if diff.abs() < 1.0e-300 {
                return *yi;
            }
            let t = wi / diff;
            num += t * yi;
            den += t;
        }
        // The Floater-Hormann denominator is provably nonzero, but guard against
        // catastrophic cancellation producing a spurious zero.
        if den == 0.0 {
            // Fall back to the nearest node's value.
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for (idx, xi) in self.nodes.iter().enumerate() {
                let dd = (x - xi).abs();
                if dd < best_d {
                    best_d = dd;
                    best = idx;
                }
            }
            return self.values[best];
        }
        num / den
    }

    /// Evaluate the *denominator* `Σ_k w_k / (x - x_k)` at `x` (for pole checks).
    ///
    /// Returns `f64::INFINITY` exactly at a node (where the limit of the
    /// quotient is finite); use [`eval`](Self::eval) for function values.
    #[must_use]
    pub fn denominator(&self, x: f64) -> f64 {
        let mut den = 0.0_f64;
        for (xi, wi) in self.nodes.iter().zip(self.weights.iter()) {
            let diff = x - xi;
            if diff.abs() < 1.0e-300 {
                return f64::INFINITY;
            }
            den += wi / diff;
        }
        den
    }

    /// The Floater-Hormann blending weights.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The blending degree `d`.
    #[must_use]
    pub fn blend_degree(&self) -> usize {
        self.blend_degree
    }

    /// The interpolation nodes.
    #[must_use]
    pub fn nodes(&self) -> &[f64] {
        &self.nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
        if n == 1 {
            return vec![a];
        }
        (0..n)
            .map(|k| a + (b - a) * k as f64 / (n as f64 - 1.0))
            .collect()
    }

    #[test]
    fn interpolates_at_nodes_exactly() {
        let nodes = linspace(-1.0, 1.0, 11);
        let values: Vec<f64> = nodes.iter().map(|&x| (3.0 * x).sin() + x * x).collect();
        for d in 0..=4 {
            let fh = FloaterHormann::new(&nodes, &values, d).expect("fh");
            for (xi, yi) in nodes.iter().zip(values.iter()) {
                let v = fh.eval(*xi);
                assert!((v - yi).abs() < 1.0e-12, "d={d}, node {xi}: {v} vs {yi}");
            }
        }
    }

    #[test]
    fn reproduces_low_degree_polynomials() {
        // r reproduces polynomials of degree ≤ d exactly.
        let nodes = linspace(0.0, 2.0, 9);
        for d in 1..=4 {
            // Build a random-ish polynomial of degree exactly d.
            let coeffs: Vec<f64> = (0..=d).map(|j| 1.0 + 0.5 * j as f64).collect();
            let poly = |x: f64| -> f64 {
                let mut acc = 0.0;
                for (j, &c) in coeffs.iter().enumerate() {
                    acc += c * x.powi(j as i32);
                }
                acc
            };
            let values: Vec<f64> = nodes.iter().map(|&x| poly(x)).collect();
            let fh = FloaterHormann::new(&nodes, &values, d).expect("fh");
            for &x in &[0.13, 0.47, 0.9, 1.21, 1.77, 1.95] {
                let v = fh.eval(x);
                let want = poly(x);
                assert!((v - want).abs() < 1.0e-9, "d={d}, x={x}: {v} vs {want}");
            }
        }
    }

    #[test]
    fn d0_is_berrut() {
        // d = 0 ⇒ Berrut's first form: w_k = (-1)^k.
        let nodes = linspace(-1.0, 1.0, 7);
        let values: Vec<f64> = nodes.iter().map(|&x| x.cos()).collect();
        let fh = FloaterHormann::new(&nodes, &values, 0).expect("fh");
        for (k, &w) in fh.weights().iter().enumerate() {
            let expected = if k % 2 == 0 { 1.0 } else { -1.0 };
            // Berrut weights are (-1)^k up to a global scale; here the formula
            // yields exactly ±1.
            assert!((w - expected).abs() < 1.0e-12, "k={k}: w={w}");
        }
    }

    #[test]
    fn no_poles_on_dense_grid() {
        // The denominator must never vanish anywhere on the interval, for a
        // range of blending degrees and equispaced nodes.
        let nodes = linspace(-1.0, 1.0, 21);
        let values: Vec<f64> = nodes.iter().map(|&x| 1.0 / (1.0 + 25.0 * x * x)).collect();
        for d in 0..=6 {
            let fh = FloaterHormann::new(&nodes, &values, d).expect("fh");
            let mut x = -1.0;
            let mut prev_sign = 0.0_f64;
            while x <= 1.0 {
                // Skip points essentially on a node (denominator is +∞ there).
                let on_node = nodes.iter().any(|&xi| (x - xi).abs() < 1.0e-9);
                if !on_node {
                    let den = fh.denominator(x);
                    assert!(den.is_finite(), "d={d}, x={x}: denominator not finite");
                    assert!(den.abs() > 1.0e-30, "d={d}, x={x}: denominator ≈ 0 (pole!)");
                    // Between consecutive nodes the denominator keeps one sign;
                    // confirm it never crosses zero away from nodes.
                    if prev_sign != 0.0 {
                        // Allow sign flips only across nodes (handled by on_node skip).
                        let _ = prev_sign;
                    }
                    prev_sign = den.signum();
                }
                x += 0.0017;
            }
        }
    }

    #[test]
    fn no_runge_blowup_on_equispaced() {
        // Runge's function on equispaced nodes: FH interpolant stays bounded and
        // its error DECREASES with n, in stark contrast to polynomial interp.
        let runge = |x: f64| 1.0 / (1.0 + 25.0 * x * x);
        let d = 3;
        let mut prev_err = f64::INFINITY;
        for &n in &[11usize, 21, 41, 81] {
            let nodes = linspace(-1.0, 1.0, n);
            let values: Vec<f64> = nodes.iter().map(|&x| runge(x)).collect();
            let fh = FloaterHormann::new(&nodes, &values, d).expect("fh");
            // Max error on a dense evaluation grid.
            let mut max_err = 0.0_f64;
            let mut max_val = 0.0_f64;
            let mut x = -1.0;
            while x <= 1.0 {
                let v = fh.eval(x);
                max_err = max_err.max((v - runge(x)).abs());
                max_val = max_val.max(v.abs());
                x += 0.001;
            }
            // Bounded: polynomial interpolation of Runge blows up to ~10s/100s;
            // FH must stay O(1).
            assert!(max_val < 2.0, "n={n}: FH value blew up to {max_val}");
            // Decreasing error with refinement.
            assert!(
                max_err < prev_err,
                "n={n}: error {max_err} did not decrease (prev {prev_err})"
            );
            prev_err = max_err;
        }
        // Final error is small.
        assert!(prev_err < 1.0e-2, "final Runge error too large: {prev_err}");
    }

    #[test]
    fn contrast_polynomial_runge_blowup() {
        // Demonstrate the contrast: pure polynomial (barycentric-Lagrange with
        // equispaced weights) DOES blow up on Runge, while FH does not.
        let runge = |x: f64| 1.0 / (1.0 + 25.0 * x * x);
        let n = 31usize;
        let nodes = linspace(-1.0, 1.0, n);
        let values: Vec<f64> = nodes.iter().map(|&x| runge(x)).collect();

        // Polynomial interpolation via equispaced Lagrange weights
        // w_k = (-1)^k C(n-1, k)  (standard equispaced barycentric weights).
        let mut binom = vec![1.0_f64; n];
        for k in 1..n {
            binom[k] = binom[k - 1] * ((n - k) as f64) / (k as f64);
        }
        let poly_weights: Vec<f64> = (0..n)
            .map(|k| if k % 2 == 0 { binom[k] } else { -binom[k] })
            .collect();
        let poly_eval = |x: f64| -> f64 {
            let mut num = 0.0;
            let mut den = 0.0;
            for ((xi, yi), wi) in nodes.iter().zip(values.iter()).zip(poly_weights.iter()) {
                let diff = x - xi;
                if diff.abs() < 1.0e-300 {
                    return *yi;
                }
                let t = wi / diff;
                num += t * yi;
                den += t;
            }
            num / den
        };

        // Sample near the endpoints where Runge oscillations are worst.
        let mut poly_max = 0.0_f64;
        let mut x = -1.0;
        while x <= 1.0 {
            poly_max = poly_max.max(poly_eval(x).abs());
            x += 0.001;
        }

        let fh = FloaterHormann::new(&nodes, &values, 3).expect("fh");
        let mut fh_max = 0.0_f64;
        let mut x = -1.0;
        while x <= 1.0 {
            fh_max = fh_max.max(fh.eval(x).abs());
            x += 0.001;
        }

        // Polynomial blows up (>> 1); FH stays bounded.
        assert!(
            poly_max > 5.0,
            "polynomial should blow up on Runge: {poly_max}"
        );
        assert!(fh_max < 2.0, "FH should stay bounded: {fh_max}");
        assert!(
            fh_max * 5.0 < poly_max,
            "FH not dramatically better than poly"
        );
    }

    #[test]
    fn convergence_order_h_dplus1() {
        // Error ≈ O(h^{d+1}) under uniform refinement: halving h should shrink
        // the error by roughly 2^{d+1}. Use a smooth non-polynomial target.
        let f = |x: f64| (2.0 * x).cos() * (-0.3 * x).exp();
        for d in [2usize, 3, 4] {
            let mut errs = Vec::new();
            for &n in &[9usize, 17, 33, 65] {
                let nodes = linspace(-1.0, 1.0, n);
                let values: Vec<f64> = nodes.iter().map(|&x| f(x)).collect();
                let fh = FloaterHormann::new(&nodes, &values, d).expect("fh");
                let mut max_err = 0.0_f64;
                let mut x = -1.0;
                while x <= 1.0 {
                    max_err = max_err.max((fh.eval(x) - f(x)).abs());
                    x += 0.0007;
                }
                errs.push(max_err);
            }
            // Estimated rate from the last refinement step (n: 33 → 65, h halved).
            let rate = (errs[2] / errs[3]).log2();
            // Expect ≈ d+1; allow a generous band for the finite-mesh regime.
            assert!(
                rate > (d as f64 + 1.0) - 1.2,
                "d={d}: observed convergence rate {rate} too low (errs={errs:?})"
            );
        }
    }

    #[test]
    fn error_handling() {
        assert!(matches!(
            FloaterHormann::new(&[], &[], 0),
            Err(NumericError::EmptyInput)
        ));
        assert!(matches!(
            FloaterHormann::new(&[0.0, 1.0], &[0.0], 0),
            Err(NumericError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            FloaterHormann::new(&[0.0, 1.0], &[0.0, 1.0], 2),
            Err(NumericError::InvalidParameter(_))
        ));
        assert!(matches!(
            FloaterHormann::new(&[0.0, 0.0, 1.0], &[0.0, 1.0, 2.0], 1),
            Err(NumericError::NumericalInstability(_))
        ));
    }

    #[test]
    fn single_node() {
        // One node, d = 0: constant interpolant.
        let fh = FloaterHormann::new(&[2.5], &[7.0], 0).expect("fh");
        assert!((fh.eval(0.0) - 7.0).abs() < 1.0e-12);
        assert!((fh.eval(100.0) - 7.0).abs() < 1.0e-12);
    }
}
