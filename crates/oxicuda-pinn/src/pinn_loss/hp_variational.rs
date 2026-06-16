//! hp-Variational Physics-Informed Neural Networks (hp-VPINN).
//!
//! Kharazmi, Zhang & Karniadakis (2021) "hp-VPINNs: Variational Physics-Informed
//! Neural Networks With Domain Decomposition", Computer Methods in Applied
//! Mechanics and Engineering (CMAME), vol. 374, 113547.
//!
//! Instead of minimising the strong-form PDE residual at collocation points, the
//! variational PINN minimises a **weak (Petrov-Galerkin) residual**: for a PDE
//! `N[u] = f` on a 1-D element `Ω_e = [a, b]`, and a set of test functions
//! `{v_k}`, the weak residual is
//!
//! ```text
//! R_k^e = ∫_{Ω_e} ( N[u](x) − f(x) ) · v_k(x) dx .
//! ```
//!
//! The hp-VPINN method (i) partitions the domain into `n_elem` non-overlapping
//! elements (`h`-refinement) and (ii) on each element uses the first `p` test
//! functions of a polynomial space (`p`-refinement). Here the test space is the
//! set of **shifted Legendre polynomials** `P_0, …, P_{p-1}` mapped onto each
//! element, which are orthogonal on the reference interval `[-1, 1]` — exactly the
//! basis used in the original paper. The element integrals are evaluated by
//! Gauss-Legendre quadrature so that polynomial integrands up to the chosen order
//! are integrated essentially exactly.
//!
//! The total variational loss is the mean-square weak residual over all
//! (element, test-function) pairs:
//!
//! ```text
//! L_var = (1 / (n_elem · p)) Σ_e Σ_k ( R_k^e )² .
//! ```
//!
//! This module provides the variational machinery (Legendre test functions,
//! Gauss-Legendre quadrature, element decomposition, weak-residual assembly) given
//! a user-supplied closure that evaluates the **strong residual** `N[u](x) − f(x)`
//! pointwise. The closure can wrap any network / analytic field; the variational
//! formulation only needs its values at the quadrature nodes.

use crate::error::{PinnError, PinnResult};

/// Configuration for an hp-variational PINN weak-residual assembly.
#[derive(Debug, Clone)]
pub struct HpVariationalConfig {
    /// Lower bound `a` of the global 1-D domain `[a, b]`.
    pub domain_lo: f32,
    /// Upper bound `b` of the global 1-D domain `[a, b]`.
    pub domain_hi: f32,
    /// Number of non-overlapping elements (`h`-refinement). Must be `>= 1`.
    pub n_elem: usize,
    /// Number of polynomial test functions per element (`p`-refinement),
    /// i.e. shifted Legendre polynomials `P_0 .. P_{p-1}`. Must be `>= 1`.
    pub n_test: usize,
    /// Number of Gauss-Legendre quadrature nodes per element. Must be `>= 1`.
    /// Choose `>= ceil((p + integrand_order)/2)` for exact integration.
    pub n_quad: usize,
}

impl Default for HpVariationalConfig {
    fn default() -> Self {
        Self {
            domain_lo: 0.0,
            domain_hi: 1.0,
            n_elem: 4,
            n_test: 3,
            n_quad: 5,
        }
    }
}

/// hp-Variational PINN weak-form residual assembler over a 1-D domain.
#[derive(Debug, Clone)]
pub struct HpVariationalPinn {
    config: HpVariationalConfig,
    /// Gauss-Legendre nodes on the reference interval `[-1, 1]`, length `n_quad`.
    quad_nodes: Vec<f32>,
    /// Gauss-Legendre weights on the reference interval `[-1, 1]`, length `n_quad`.
    quad_weights: Vec<f32>,
}

impl HpVariationalPinn {
    /// Construct an hp-VPINN assembler.
    ///
    /// # Errors
    /// - [`PinnError::InvalidTimeInterval`] (re-used for spatial bounds) if
    ///   `domain_hi <= domain_lo`.
    /// - [`PinnError::InvalidGridResolution`] if `n_elem == 0`.
    /// - [`PinnError::InvalidLayerWidth`] if `n_test == 0` or `n_quad == 0`.
    pub fn new(config: HpVariationalConfig) -> PinnResult<Self> {
        if !config.domain_lo.is_finite()
            || !config.domain_hi.is_finite()
            || config.domain_hi <= config.domain_lo
        {
            return Err(PinnError::InvalidTimeInterval {
                t0: config.domain_lo,
                t1: config.domain_hi,
            });
        }
        if config.n_elem == 0 {
            return Err(PinnError::InvalidGridResolution { n: config.n_elem });
        }
        if config.n_test == 0 || config.n_quad == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        let (quad_nodes, quad_weights) = gauss_legendre(config.n_quad);
        Ok(Self {
            config,
            quad_nodes,
            quad_weights,
        })
    }

    /// Number of elements (`h`-refinement).
    #[must_use]
    pub fn n_elem(&self) -> usize {
        self.config.n_elem
    }

    /// Number of test functions per element (`p`-refinement).
    #[must_use]
    pub fn n_test(&self) -> usize {
        self.config.n_test
    }

    /// The width of each (uniform) element `h = (b − a) / n_elem`.
    #[must_use]
    pub fn element_width(&self) -> f32 {
        (self.config.domain_hi - self.config.domain_lo) / self.config.n_elem as f32
    }

    /// Physical bounds `[lo, hi]` of element `e` (`0 <= e < n_elem`).
    #[must_use]
    pub fn element_bounds(&self, e: usize) -> (f32, f32) {
        let h = self.element_width();
        let lo = self.config.domain_lo + e as f32 * h;
        (lo, lo + h)
    }

    /// Physical quadrature nodes for element `e` (length `n_quad`).
    ///
    /// Maps reference nodes `ξ ∈ [-1, 1]` onto `[lo, hi]` via
    /// `x = lo + (ξ + 1)/2 · h`.
    #[must_use]
    pub fn element_quad_points(&self, e: usize) -> Vec<f32> {
        let (lo, hi) = self.element_bounds(e);
        let half = 0.5 * (hi - lo);
        let mid = 0.5 * (hi + lo);
        self.quad_nodes.iter().map(|&xi| mid + half * xi).collect()
    }

    /// All physical quadrature nodes across every element, element-major
    /// (`n_elem * n_quad`). The user evaluates the strong residual at exactly
    /// these points and passes the values back to [`Self::weak_residuals`].
    #[must_use]
    pub fn all_quad_points(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.config.n_elem * self.config.n_quad);
        for e in 0..self.config.n_elem {
            out.extend(self.element_quad_points(e));
        }
        out
    }

    /// Assemble the weak (variational) residuals `R_k^e` for every element and
    /// every test function.
    ///
    /// `strong_residual` holds the pointwise strong residual `N[u](x) − f(x)`
    /// evaluated at [`Self::all_quad_points`] (element-major, length
    /// `n_elem * n_quad`).
    ///
    /// Returns an `n_elem * n_test` vector laid out element-major: index
    /// `e * n_test + k` is `R_k^e = ∫_{Ω_e} r(x) · P_k(ξ(x)) dx`, evaluated by the
    /// element's Gauss-Legendre rule with Jacobian `h/2`.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `strong_residual.len() != n_elem * n_quad`.
    /// - [`PinnError::NanEncountered`] if any input or result is non-finite.
    pub fn weak_residuals(&self, strong_residual: &[f32]) -> PinnResult<Vec<f32>> {
        let n_quad = self.config.n_quad;
        let n_elem = self.config.n_elem;
        let n_test = self.config.n_test;
        let expected = n_elem * n_quad;
        if strong_residual.len() != expected {
            return Err(PinnError::DimensionMismatch {
                expected,
                got: strong_residual.len(),
            });
        }
        if strong_residual.iter().any(|v| !v.is_finite()) {
            return Err(PinnError::NanEncountered {
                location: "hp_variational::weak_residuals(input)",
            });
        }

        // Precompute Legendre values P_k(ξ_q) for every test function and node.
        // legendre_vals[k * n_quad + q] = P_k(quad_nodes[q]).
        let mut legendre_vals = vec![0.0_f32; n_test * n_quad];
        for (q, &xi) in self.quad_nodes.iter().enumerate() {
            let pk = legendre_basis(xi, n_test);
            for (k, &val) in pk.iter().enumerate() {
                legendre_vals[k * n_quad + q] = val;
            }
        }

        let jac = 0.5 * self.element_width(); // dx = (h/2) dξ
        let mut out = vec![0.0_f32; n_elem * n_test];
        for e in 0..n_elem {
            let r_off = e * n_quad;
            for k in 0..n_test {
                let mut acc = 0.0_f32;
                for q in 0..n_quad {
                    let r = strong_residual[r_off + q];
                    let pk = legendre_vals[k * n_quad + q];
                    acc += self.quad_weights[q] * r * pk;
                }
                out[e * n_test + k] = acc * jac;
            }
        }

        if out.iter().any(|v| !v.is_finite()) {
            return Err(PinnError::NanEncountered {
                location: "hp_variational::weak_residuals(output)",
            });
        }
        Ok(out)
    }

    /// Mean-square variational loss `L_var = (1/(n_elem·p)) Σ (R_k^e)²`.
    ///
    /// # Errors
    /// Propagates errors from [`Self::weak_residuals`].
    pub fn variational_loss(&self, strong_residual: &[f32]) -> PinnResult<f32> {
        let weak = self.weak_residuals(strong_residual)?;
        if weak.is_empty() {
            return Err(PinnError::EmptyCollocationSet);
        }
        let mse = weak.iter().map(|&r| r * r).sum::<f32>() / weak.len() as f32;
        if !mse.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "hp_variational::variational_loss",
            });
        }
        Ok(mse)
    }
}

/// Evaluate the shifted Legendre basis `[P_0(x), …, P_{n-1}(x)]` on `[-1, 1]`
/// via Bonnet's recurrence
/// `(k+1) P_{k+1}(x) = (2k+1) x P_k(x) − k P_{k-1}(x)`.
#[must_use]
pub fn legendre_basis(x: f32, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    out.push(1.0_f32); // P_0
    if n == 1 {
        return out;
    }
    out.push(x); // P_1
    for k in 1..n - 1 {
        let kf = k as f32;
        let next = ((2.0 * kf + 1.0) * x * out[k] - kf * out[k - 1]) / (kf + 1.0);
        out.push(next);
    }
    out
}

/// Gauss-Legendre nodes and weights on `[-1, 1]` for `n` points.
///
/// Roots of `P_n` are found by Newton's method seeded with the
/// Chebyshev approximation `cos(π(i + 0.75)/(n + 0.5))`; weights use
/// `w_i = 2 / ((1 − x_i²) [P_n'(x_i)]²)`.
#[must_use]
pub fn gauss_legendre(n: usize) -> (Vec<f32>, Vec<f32>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n == 1 {
        return (vec![0.0_f32], vec![2.0_f32]);
    }
    // Work in f64 for stable root finding, then narrow to f32.
    let nf = n as f64;
    let mut nodes = vec![0.0_f64; n];
    let mut weights = vec![0.0_f64; n];
    for i in 0..n {
        // Initial guess (Chebyshev / asymptotic).
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        // Newton iterations on P_n(x) = 0.
        for _ in 0..100 {
            // Evaluate P_n and P_{n-1} by recurrence.
            let mut p_prev = 1.0_f64; // P_0
            let mut p_curr = x; // P_1
            for k in 1..n {
                let kf = k as f64;
                let p_next = ((2.0 * kf + 1.0) * x * p_curr - kf * p_prev) / (kf + 1.0);
                p_prev = p_curr;
                p_curr = p_next;
            }
            // p_curr = P_n(x); derivative P_n'(x).
            let dp = nf * (x * p_curr - p_prev) / (x * x - 1.0);
            let dx = p_curr / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        // Recompute derivative at the converged root for the weight.
        let mut p_prev = 1.0_f64;
        let mut p_curr = x;
        for k in 1..n {
            let kf = k as f64;
            let p_next = ((2.0 * kf + 1.0) * x * p_curr - kf * p_prev) / (kf + 1.0);
            p_prev = p_curr;
            p_curr = p_next;
        }
        let dp = nf * (x * p_curr - p_prev) / (x * x - 1.0);
        nodes[i] = x;
        weights[i] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
    (
        nodes.into_iter().map(|v| v as f32).collect(),
        weights.into_iter().map(|v| v as f32).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn legendre_basis_known_values() {
        // P_0=1, P_1=x, P_2=(3x²−1)/2, P_3=(5x³−3x)/2 at x=0.5.
        let p = legendre_basis(0.5, 4);
        assert!(approx(p[0], 1.0, 1e-6));
        assert!(approx(p[1], 0.5, 1e-6));
        assert!(approx(p[2], (3.0 * 0.25 - 1.0) / 2.0, 1e-6));
        assert!(approx(p[3], (5.0 * 0.125 - 3.0 * 0.5) / 2.0, 1e-6));
    }

    #[test]
    fn legendre_endpoints_are_one() {
        // P_k(1) = 1 for all k.
        let p = legendre_basis(1.0, 6);
        for (k, &v) in p.iter().enumerate() {
            assert!(approx(v, 1.0, 1e-5), "P_{k}(1) = {v}");
        }
    }

    #[test]
    fn gauss_legendre_weights_sum_to_two() {
        for n in 1..=8 {
            let (_, w) = gauss_legendre(n);
            let s: f32 = w.iter().sum();
            assert!(approx(s, 2.0, 1e-4), "n={n} weight sum = {s}");
        }
    }

    #[test]
    fn gauss_legendre_integrates_polynomials_exactly() {
        // ∫_{-1}^{1} x² dx = 2/3, exact for n>=2 nodes.
        let (x, w) = gauss_legendre(3);
        let integral: f32 = x.iter().zip(w.iter()).map(|(&xi, &wi)| wi * xi * xi).sum();
        assert!(approx(integral, 2.0 / 3.0, 1e-5), "got {integral}");
        // ∫_{-1}^{1} x⁴ dx = 2/5.
        let i4: f32 = x
            .iter()
            .zip(w.iter())
            .map(|(&xi, &wi)| wi * xi.powi(4))
            .sum();
        assert!(approx(i4, 2.0 / 5.0, 1e-4), "got {i4}");
    }

    #[test]
    fn element_decomposition_partitions_domain() {
        let cfg = HpVariationalConfig {
            domain_lo: 0.0,
            domain_hi: 2.0,
            n_elem: 4,
            n_test: 2,
            n_quad: 3,
        };
        let vp = HpVariationalPinn::new(cfg)
            .expect("HpVariationalPinn construction with valid params should succeed");
        assert!(approx(vp.element_width(), 0.5, 1e-6));
        let (lo0, hi0) = vp.element_bounds(0);
        let (lo3, hi3) = vp.element_bounds(3);
        assert!(approx(lo0, 0.0, 1e-6) && approx(hi0, 0.5, 1e-6));
        assert!(approx(lo3, 1.5, 1e-6) && approx(hi3, 2.0, 1e-6));
    }

    #[test]
    fn all_quad_points_count_and_in_domain() {
        let cfg = HpVariationalConfig {
            domain_lo: -1.0,
            domain_hi: 1.0,
            n_elem: 3,
            n_test: 2,
            n_quad: 4,
        };
        let vp = HpVariationalPinn::new(cfg)
            .expect("HpVariationalPinn construction with valid params should succeed");
        let pts = vp.all_quad_points();
        assert_eq!(pts.len(), 3 * 4);
        for &x in &pts {
            assert!((-1.0..=1.0).contains(&x), "quad point {x} out of domain");
        }
    }

    #[test]
    fn weak_residual_zero_for_zero_strong_residual() {
        let cfg = HpVariationalConfig::default();
        let vp = HpVariationalPinn::new(cfg)
            .expect("HpVariationalPinn construction with valid default config should succeed");
        let pts = vp.all_quad_points();
        let r = vec![0.0_f32; pts.len()];
        let weak = vp
            .weak_residuals(&r)
            .expect("weak_residuals computation with zero input should succeed");
        assert!(
            weak.iter().all(|&v| v.abs() < 1e-7),
            "exact solution → zero weak residual"
        );
        assert!(
            vp.variational_loss(&r)
                .expect("hp-variational loss computation with zero residual should succeed")
                < 1e-12
        );
    }

    #[test]
    fn weak_residual_constant_against_p0_is_integral() {
        // For a constant strong residual r≡c, the P_0 (=1) weak residual on each
        // element equals c · (element width); higher modes integrate to ~0.
        let cfg = HpVariationalConfig {
            domain_lo: 0.0,
            domain_hi: 1.0,
            n_elem: 2,
            n_test: 3,
            n_quad: 5,
        };
        let vp = HpVariationalPinn::new(cfg)
            .expect("HpVariationalPinn construction with valid params should succeed");
        let pts = vp.all_quad_points();
        let c = 2.5_f32;
        let r = vec![c; pts.len()];
        let weak = vp
            .weak_residuals(&r)
            .expect("weak_residuals computation with constant residual should succeed");
        let h = vp.element_width();
        for e in 0..vp.n_elem() {
            // P_0 mode = c·h
            assert!(
                approx(weak[e * vp.n_test()], c * h, 1e-4),
                "P0 weak residual element {e}"
            );
            // P_1, P_2 modes ~ 0 (orthogonality of Legendre to constant)
            assert!(weak[e * vp.n_test() + 1].abs() < 1e-4);
            assert!(weak[e * vp.n_test() + 2].abs() < 1e-4);
        }
    }

    #[test]
    fn weak_residual_nonzero_for_nonzero_strong() {
        let cfg = HpVariationalConfig::default();
        let vp = HpVariationalPinn::new(cfg)
            .expect("HpVariationalPinn construction with valid default config should succeed");
        let pts = vp.all_quad_points();
        // r(x) = x − 0.5 (a non-constant residual) → some modes nonzero.
        let r: Vec<f32> = pts.iter().map(|&x| x - 0.5).collect();
        let loss = vp
            .variational_loss(&r)
            .expect("hp-variational loss computation with non-zero residual should succeed");
        assert!(loss > 0.0, "non-zero strong residual → positive loss");
    }

    #[test]
    fn weak_residual_deterministic() {
        let cfg = HpVariationalConfig::default();
        let vp = HpVariationalPinn::new(cfg)
            .expect("HpVariationalPinn construction with valid default config should succeed");
        let pts = vp.all_quad_points();
        let r: Vec<f32> = pts.iter().map(|&x| (3.0 * x).sin()).collect();
        let a = vp
            .weak_residuals(&r)
            .expect("weak_residuals computation should succeed for valid sinusoidal residual");
        let b = vp.weak_residuals(&r).expect(
            "weak_residuals second computation should succeed for valid sinusoidal residual",
        );
        assert_eq!(a, b);
    }

    #[test]
    fn dimension_mismatch_errors() {
        let cfg = HpVariationalConfig::default();
        let vp = HpVariationalPinn::new(cfg)
            .expect("HpVariationalPinn construction with valid default config should succeed");
        let bad = vec![0.0_f32; 3];
        assert!(matches!(
            vp.weak_residuals(&bad),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn invalid_config_errors() {
        // Degenerate domain.
        assert!(
            HpVariationalPinn::new(HpVariationalConfig {
                domain_lo: 1.0,
                domain_hi: 1.0,
                n_elem: 2,
                n_test: 2,
                n_quad: 2,
            })
            .is_err()
        );
        // Zero elements.
        assert!(
            HpVariationalPinn::new(HpVariationalConfig {
                n_elem: 0,
                ..Default::default()
            })
            .is_err()
        );
        // Zero test functions.
        assert!(
            HpVariationalPinn::new(HpVariationalConfig {
                n_test: 0,
                ..Default::default()
            })
            .is_err()
        );
    }

    #[test]
    fn weak_residual_nan_input_errors() {
        let cfg = HpVariationalConfig::default();
        let vp = HpVariationalPinn::new(cfg)
            .expect("HpVariationalPinn construction with valid default config should succeed");
        let mut r = vec![0.0_f32; vp.n_elem() * vp.config.n_quad];
        r[0] = f32::NAN;
        assert!(matches!(
            vp.weak_residuals(&r),
            Err(PinnError::NanEncountered { .. })
        ));
    }
}
