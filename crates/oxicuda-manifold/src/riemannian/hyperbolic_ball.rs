//! Curvature-parametrised Poincaré ball model of hyperbolic space (Ganea 2018, Nickel-Kiela 2017).
//!
//! While [`crate::riemannian::hyperbolic_poincare`] fixes the curvature at `-1` (the unit ball),
//! this module exposes a [`PoincareBall`] equipped with a configurable negative curvature `-c`
//! (`c > 0`). The ball is `B^d_c = {x ∈ ℝ^d : c‖x‖² < 1}` with Riemannian metric
//! `g_x^c = (λ_x^c)² I_d`, conformal factor `λ_x^c = 2 / (1 - c‖x‖²)`.
//!
//! # Gyrovector operations
//!
//! ## Möbius addition (curvature `c`)
//! ```text
//! x ⊕_c y = ((1 + 2c⟨x,y⟩ + c‖y‖²) x + (1 - c‖x‖²) y)
//!            / (1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²)
//! ```
//!
//! ## Distance
//! ```text
//! d_c(x, y) = (2 / √c) · arctanh(√c · ‖(-x) ⊕_c y‖)
//! ```
//!
//! ## Exponential / logarithmic maps at `x`
//! ```text
//! exp_x^c(v) = x ⊕_c ( tanh(√c · λ_x^c · ‖v‖ / 2) · v / (√c · ‖v‖) )
//! log_x^c(y) = (2 / (√c · λ_x^c)) · arctanh(√c · ‖u‖) · u / ‖u‖,   u = (-x) ⊕_c y
//! ```
//!
//! ## Parallel transport (via gyration)
//! ```text
//! P_{x→y}^c(v) = (λ_x^c / λ_y^c) · gyr[y, -x] v
//! gyr[u, v] w  = ⊖_c (u ⊕_c v) ⊕_c ( u ⊕_c (v ⊕_c w) )
//! ```
//!
//! ## Riemannian gradient from a Euclidean gradient
//! Because `g_x^c = (λ_x^c)² I`, the inverse metric scales the Euclidean gradient by
//! `(λ_x^c)^{-2} = (1 - c‖x‖²)² / 4`.
//!
//! # References
//! - Ganea, Bécigneul, Hofmann (NeurIPS 2018), *Hyperbolic Neural Networks*.
//! - Nickel, Kiela (NeurIPS 2017), *Poincaré Embeddings for Learning Hierarchical Representations*.
//! - Ungar (2008), *A Gyrovector Space Approach to Hyperbolic Geometry*.

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;

/// A Poincaré ball with configurable negative curvature `-c` (`c > 0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoincareBall {
    /// Curvature magnitude `c > 0`; the manifold has constant sectional curvature `-c`.
    pub curvature: f64,
    /// Safety margin keeping points strictly inside the ball (`c‖x‖² <= 1 - eps`).
    pub epsilon: f64,
}

impl Default for PoincareBall {
    fn default() -> Self {
        Self {
            curvature: 1.0,
            epsilon: 1e-5,
        }
    }
}

impl PoincareBall {
    /// Construct a Poincaré ball with curvature `-curvature`.
    ///
    /// # Errors
    /// Returns [`ManifoldError::InvalidParameter`] when `curvature <= 0` or `epsilon` is not in `(0, 1)`.
    pub fn new(curvature: f64) -> ManifoldResult<Self> {
        Self::with_epsilon(curvature, 1e-5)
    }

    /// Construct a Poincaré ball with curvature `-curvature` and explicit boundary margin.
    ///
    /// # Errors
    /// Returns [`ManifoldError::InvalidParameter`] when `curvature <= 0` or `epsilon` is not in `(0, 1)`.
    pub fn with_epsilon(curvature: f64, epsilon: f64) -> ManifoldResult<Self> {
        if !curvature.is_finite() || curvature <= 0.0 {
            return Err(ManifoldError::InvalidParameter {
                name: "curvature".into(),
                reason: "must be a finite, strictly positive value".into(),
            });
        }
        if !(epsilon > 0.0 && epsilon < 1.0) {
            return Err(ManifoldError::InvalidParameter {
                name: "epsilon".into(),
                reason: "must be in (0, 1)".into(),
            });
        }
        Ok(Self { curvature, epsilon })
    }

    /// `√c`.
    #[inline]
    fn sqrt_c(&self) -> f64 {
        self.curvature.sqrt()
    }

    /// Conformal factor `λ_x^c = 2 / (1 - c‖x‖²)`.
    #[inline]
    pub fn conformal_factor(&self, x: &[f64]) -> f64 {
        let n2: f64 = dot(x, x);
        let denom = (1.0 - self.curvature * n2).max(1e-30);
        2.0 / denom
    }

    /// Project `x` so that `c‖x‖² <= 1 - epsilon` (returns a point strictly inside the ball).
    #[must_use]
    pub fn project(&self, x: &[f64]) -> Vec<f64> {
        let n2: f64 = dot(x, x);
        // Maximum allowed squared Euclidean norm.
        let max_n2 = (1.0 - self.epsilon) / self.curvature;
        if n2 <= max_n2 || n2 == 0.0 {
            return x.to_vec();
        }
        let scale = (max_n2 / n2).sqrt();
        x.iter().map(|v| v * scale).collect()
    }

    /// Return `true` iff `x` lies strictly inside the ball (`c‖x‖² < 1`).
    #[must_use]
    pub fn contains(&self, x: &[f64]) -> bool {
        self.curvature * dot(x, x) < 1.0
    }

    /// Möbius addition `x ⊕_c y`.
    ///
    /// # Errors
    /// Returns [`ManifoldError::DimensionMismatch`] when the operands differ in length.
    pub fn mobius_add(&self, x: &[f64], y: &[f64]) -> ManifoldResult<Vec<f64>> {
        check_dim(x, y)?;
        let c = self.curvature;
        let xn2 = dot(x, x);
        let yn2 = dot(y, y);
        let xy = dot(x, y);
        let num_x = 1.0 + 2.0 * c * xy + c * yn2;
        let num_y = 1.0 - c * xn2;
        let denom = (1.0 + 2.0 * c * xy + c * c * xn2 * yn2).abs().max(1e-30);
        Ok(x.iter()
            .zip(y)
            .map(|(a, b)| (num_x * a + num_y * b) / denom)
            .collect())
    }

    /// Möbius subtraction `x ⊖_c y = x ⊕_c (-y)`.
    ///
    /// # Errors
    /// Returns [`ManifoldError::DimensionMismatch`] when the operands differ in length.
    pub fn mobius_sub(&self, x: &[f64], y: &[f64]) -> ManifoldResult<Vec<f64>> {
        let neg_y: Vec<f64> = y.iter().map(|v| -v).collect();
        self.mobius_add(x, &neg_y)
    }

    /// Möbius scalar multiplication `r ⊗_c x`.
    ///
    /// `r ⊗_c x = (1/√c) tanh(r · artanh(√c ‖x‖)) · x / ‖x‖`.
    #[must_use]
    pub fn mobius_scalar_mul(&self, r: f64, x: &[f64]) -> Vec<f64> {
        let norm = dot(x, x).sqrt();
        if norm < 1e-15 {
            return vec![0.0; x.len()];
        }
        let sc = self.sqrt_c();
        let arg = (sc * norm).clamp(-1.0 + 1e-15, 1.0 - 1e-15);
        let scale = (r * artanh(arg)).tanh() / (sc * norm);
        x.iter().map(|v| v * scale).collect()
    }

    /// Hyperbolic distance `d_c(x, y)`.
    ///
    /// # Errors
    /// Returns [`ManifoldError::DimensionMismatch`] when the operands differ in length.
    pub fn distance(&self, x: &[f64], y: &[f64]) -> ManifoldResult<f64> {
        check_dim(x, y)?;
        let neg_x: Vec<f64> = x.iter().map(|v| -v).collect();
        let diff = self.mobius_add(&neg_x, y)?;
        let sc = self.sqrt_c();
        let arg = (sc * dot(&diff, &diff).sqrt()).clamp(0.0, 1.0 - 1e-15);
        Ok((2.0 / sc) * artanh(arg))
    }

    /// Exponential map `exp_x^c(v)` taking a tangent vector `v ∈ T_x` onto the ball.
    ///
    /// # Errors
    /// Returns [`ManifoldError::DimensionMismatch`] when `x` and `v` differ in length.
    pub fn exp_map(&self, x: &[f64], v: &[f64]) -> ManifoldResult<Vec<f64>> {
        check_dim(x, v)?;
        let v_norm = dot(v, v).sqrt();
        if v_norm < 1e-15 {
            return Ok(x.to_vec());
        }
        let sc = self.sqrt_c();
        let lambda = self.conformal_factor(x);
        // tanh argument, clamped to avoid overflow in tanh's series for large magnitudes.
        let arg = (sc * lambda * v_norm / 2.0).clamp(-88.0, 88.0);
        let scale = arg.tanh() / (sc * v_norm);
        let dir: Vec<f64> = v.iter().map(|vi| vi * scale).collect();
        let raw = self.mobius_add(x, &dir)?;
        Ok(self.project(&raw))
    }

    /// Logarithmic map `log_x^c(y)` returning the tangent vector at `x` pointing to `y`.
    ///
    /// # Errors
    /// Returns [`ManifoldError::DimensionMismatch`] when `x` and `y` differ in length.
    pub fn log_map(&self, x: &[f64], y: &[f64]) -> ManifoldResult<Vec<f64>> {
        check_dim(x, y)?;
        let neg_x: Vec<f64> = x.iter().map(|v| -v).collect();
        let u = self.mobius_add(&neg_x, y)?;
        let u_norm = dot(&u, &u).sqrt();
        if u_norm < 1e-15 {
            return Ok(vec![0.0; x.len()]);
        }
        let sc = self.sqrt_c();
        let lambda = self.conformal_factor(x);
        let arg = (sc * u_norm).clamp(-1.0 + 1e-15, 1.0 - 1e-15);
        let scale = (2.0 / (sc * lambda)) * artanh(arg) / u_norm;
        Ok(u.iter().map(|ui| ui * scale).collect())
    }

    /// Gyration operator `gyr[u, v] w` (the automorphism correcting non-associativity).
    fn gyration(&self, u: &[f64], v: &[f64], w: &[f64]) -> ManifoldResult<Vec<f64>> {
        let uv = self.mobius_add(u, v)?;
        let vw = self.mobius_add(v, w)?;
        let u_vw = self.mobius_add(u, &vw)?;
        // ⊖(u ⊕ v) ⊕ (u ⊕ (v ⊕ w))
        let neg_uv: Vec<f64> = uv.iter().map(|x| -x).collect();
        self.mobius_add(&neg_uv, &u_vw)
    }

    /// Parallel transport of tangent vector `v` from `T_x` to `T_y`.
    ///
    /// `P_{x→y}^c(v) = (λ_x^c / λ_y^c) · gyr[y, -x] v`.
    ///
    /// # Errors
    /// Returns [`ManifoldError::DimensionMismatch`] when any operands differ in length.
    pub fn parallel_transport(&self, x: &[f64], y: &[f64], v: &[f64]) -> ManifoldResult<Vec<f64>> {
        check_dim(x, y)?;
        check_dim(x, v)?;
        let neg_x: Vec<f64> = x.iter().map(|t| -t).collect();
        let g = self.gyration(y, &neg_x, v)?;
        let lambda_x = self.conformal_factor(x);
        let lambda_y = self.conformal_factor(y);
        let ratio = lambda_x / lambda_y;
        Ok(g.iter().map(|gi| gi * ratio).collect())
    }

    /// Convert a Euclidean gradient at `x` into the Riemannian gradient.
    ///
    /// `grad_R f(x) = (1 - c‖x‖²)² / 4 · grad_E f(x)`.
    #[must_use]
    pub fn egrad_to_rgrad(&self, x: &[f64], egrad: &[f64]) -> Vec<f64> {
        let n2 = dot(x, x);
        let factor = {
            let s = (1.0 - self.curvature * n2).max(0.0);
            s * s / 4.0
        };
        egrad.iter().map(|g| g * factor).collect()
    }

    /// One Riemannian gradient-descent step: `x_new = exp_x^c(-lr · grad_R f(x))`.
    ///
    /// `rgrad` must already be the Riemannian gradient (see [`Self::egrad_to_rgrad`]).
    ///
    /// # Errors
    /// Returns [`ManifoldError::DimensionMismatch`] when `x` and `rgrad` differ in length.
    pub fn sgd_step(&self, x: &[f64], rgrad: &[f64], lr: f64) -> ManifoldResult<Vec<f64>> {
        check_dim(x, rgrad)?;
        let step: Vec<f64> = rgrad.iter().map(|g| -lr * g).collect();
        self.exp_map(x, &step)
    }

    /// Sample a point uniformly-ish inside the ball (small radius), useful for initialisation.
    #[must_use]
    pub fn random_point(&self, dim: usize, rng: &mut LcgRng) -> Vec<f64> {
        let raw: Vec<f64> = (0..dim).map(|_| rng.next_range(-1e-3, 1e-3)).collect();
        self.project(&raw)
    }
}

/// Fréchet (Karcher) mean of points on a [`PoincareBall`] via Riemannian gradient descent.
///
/// Minimises `F(m) = (1/N) Σ_i d_c(m, x_i)²` by iterating
/// `m ← exp_m( (1/N) Σ_i log_m(x_i) )` until the tangent update is small.
///
/// `points` is row-major `(n, dim)`.
///
/// # Errors
/// Returns [`ManifoldError::EmptyInput`] for no points and [`ManifoldError::ShapeMismatch`]
/// when `points.len() != n * dim`.
pub fn poincare_frechet_mean(
    ball: &PoincareBall,
    points: &[f64],
    n: usize,
    dim: usize,
    max_iter: usize,
    tol: f64,
) -> ManifoldResult<Vec<f64>> {
    if n == 0 || dim == 0 {
        return Err(ManifoldError::EmptyInput);
    }
    if points.len() != n * dim {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, dim],
            got: vec![points.len()],
        });
    }
    // Initialise at the (projected) Euclidean centroid.
    let mut m = vec![0.0; dim];
    for i in 0..n {
        for k in 0..dim {
            m[k] += points[i * dim + k];
        }
    }
    for v in &mut m {
        *v /= n as f64;
    }
    m = ball.project(&m);

    for _ in 0..max_iter {
        let mut tangent = vec![0.0; dim];
        for i in 0..n {
            let xi = &points[i * dim..(i + 1) * dim];
            let lg = ball.log_map(&m, xi)?;
            for k in 0..dim {
                tangent[k] += lg[k];
            }
        }
        for v in &mut tangent {
            *v /= n as f64;
        }
        let step_norm = dot(&tangent, &tangent).sqrt();
        m = ball.exp_map(&m, &tangent)?;
        if step_norm < tol {
            break;
        }
    }
    Ok(m)
}

// ─────────────────────────────────────────────────────────────────────────────
// Free helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn check_dim(a: &[f64], b: &[f64]) -> ManifoldResult<()> {
    if a.len() != b.len() {
        Err(ManifoldError::DimensionMismatch {
            a: a.len(),
            b: b.len(),
        })
    } else {
        Ok(())
    }
}

/// Numerically stable `artanh(x) = 0.5 ln((1+x)/(1-x))`.
#[inline]
fn artanh(x: f64) -> f64 {
    let cx = x.clamp(-1.0 + 1e-15, 1.0 - 1e-15);
    0.5 * ((1.0 + cx) / (1.0 - cx)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit() -> PoincareBall {
        PoincareBall::new(1.0).expect("valid curvature")
    }

    #[test]
    fn invalid_curvature_rejected() {
        assert!(PoincareBall::new(0.0).is_err());
        assert!(PoincareBall::new(-1.0).is_err());
        assert!(PoincareBall::with_epsilon(1.0, 0.0).is_err());
        assert!(PoincareBall::with_epsilon(1.0, 1.0).is_err());
    }

    #[test]
    fn distance_to_self_is_zero() {
        let b = unit();
        let x = vec![0.1, -0.2, 0.05];
        let d = b.distance(&x, &x).expect("ok");
        assert!(d.abs() < 1e-9, "self distance {d}");
    }

    #[test]
    fn distance_symmetric() {
        let b = PoincareBall::new(2.5).expect("ok");
        let x = vec![0.1, 0.2];
        let y = vec![-0.3, 0.05];
        let d1 = b.distance(&x, &y).expect("ok");
        let d2 = b.distance(&y, &x).expect("ok");
        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn distance_matches_unit_ball_closed_form() {
        // For c = 1 the curvature-agnostic closed form is
        // d = arcosh(1 + 2‖x-y‖² / ((1-‖x‖²)(1-‖y‖²))).
        let b = unit();
        let x = vec![0.2, -0.1];
        let y = vec![-0.15, 0.3];
        let xn2: f64 = dot(&x, &x);
        let yn2: f64 = dot(&y, &y);
        let dn2: f64 = x.iter().zip(&y).map(|(a, c)| (a - c) * (a - c)).sum();
        let arg = 1.0 + 2.0 * dn2 / ((1.0 - xn2) * (1.0 - yn2));
        let expected = (arg + (arg * arg - 1.0).sqrt()).ln();
        let got = b.distance(&x, &y).expect("ok");
        assert!(
            (got - expected).abs() < 1e-9,
            "got {got} expected {expected}"
        );
    }

    #[test]
    fn mobius_add_zero_identity() {
        let b = PoincareBall::new(0.7).expect("ok");
        let x = vec![0.3, -0.2, 0.1];
        let z = vec![0.0; 3];
        let s = b.mobius_add(&x, &z).expect("ok");
        for (a, c) in x.iter().zip(&s) {
            assert!((a - c).abs() < 1e-12);
        }
        // Left identity too.
        let s2 = b.mobius_add(&z, &x).expect("ok");
        for (a, c) in x.iter().zip(&s2) {
            assert!((a - c).abs() < 1e-12);
        }
    }

    #[test]
    fn mobius_left_cancellation() {
        // (-x) ⊕ (x ⊕ y) = y.
        let b = PoincareBall::new(1.3).expect("ok");
        let x = vec![0.2, -0.1];
        let y = vec![-0.05, 0.15];
        let xy = b.mobius_add(&x, &y).expect("ok");
        let neg_x: Vec<f64> = x.iter().map(|v| -v).collect();
        let recovered = b.mobius_add(&neg_x, &xy).expect("ok");
        for (a, c) in y.iter().zip(&recovered) {
            assert!((a - c).abs() < 1e-9, "cancellation: {a} vs {c}");
        }
    }

    #[test]
    fn exp_log_roundtrip() {
        // log_x(exp_x(v)) = v for any tangent v.
        for c in [0.5_f64, 1.0, 3.0] {
            let b = PoincareBall::new(c).expect("ok");
            let x = vec![0.1, -0.2, 0.05];
            let v = vec![0.15, 0.1, -0.05];
            let y = b.exp_map(&x, &v).expect("ok");
            assert!(b.contains(&y));
            let v_rec = b.log_map(&x, &y).expect("ok");
            for (a, c2) in v.iter().zip(&v_rec) {
                assert!((a - c2).abs() < 1e-7, "c={c}: {a} vs {c2}");
            }
        }
    }

    #[test]
    fn exp_log_distance_consistency() {
        // d_c(x, exp_x(v)) = ‖v‖_x = λ_x ‖v‖ (Riemannian norm of the tangent vector).
        let b = PoincareBall::new(1.0).expect("ok");
        let x = vec![0.05, -0.1];
        let v = vec![0.2, 0.1];
        let y = b.exp_map(&x, &v).expect("ok");
        let d = b.distance(&x, &y).expect("ok");
        let lambda = b.conformal_factor(&x);
        let riem_norm = lambda * dot(&v, &v).sqrt();
        assert!((d - riem_norm).abs() < 1e-7, "d={d} riem_norm={riem_norm}");
    }

    #[test]
    fn project_keeps_points_inside() {
        let b = PoincareBall::with_epsilon(4.0, 1e-3).expect("ok");
        let x = vec![10.0, -5.0];
        let p = b.project(&x);
        assert!(b.contains(&p));
        assert!(b.curvature * dot(&p, &p) <= 1.0 - 1e-3 + 1e-9);
    }

    #[test]
    fn parallel_transport_preserves_riemannian_norm() {
        // Parallel transport is an isometry: ‖P_{x→y}v‖_y = ‖v‖_x.
        let b = PoincareBall::new(1.0).expect("ok");
        let x = vec![0.1, -0.05, 0.2];
        let y = vec![-0.15, 0.1, 0.05];
        let v = vec![0.2, 0.1, -0.1];
        let pv = b.parallel_transport(&x, &y, &v).expect("ok");
        let lam_x = b.conformal_factor(&x);
        let lam_y = b.conformal_factor(&y);
        let norm_x = lam_x * dot(&v, &v).sqrt();
        let norm_y = lam_y * dot(&pv, &pv).sqrt();
        assert!(
            (norm_x - norm_y).abs() < 1e-7,
            "‖v‖_x={norm_x} ‖Pv‖_y={norm_y}"
        );
    }

    #[test]
    fn mobius_scalar_mul_one_and_two() {
        // 1 ⊗ x = x and 2 ⊗ x = x ⊕ x.
        let b = PoincareBall::new(1.0).expect("ok");
        let x = vec![0.2, -0.1];
        let one_x = b.mobius_scalar_mul(1.0, &x);
        for (a, c) in x.iter().zip(&one_x) {
            assert!((a - c).abs() < 1e-9);
        }
        let two_x = b.mobius_scalar_mul(2.0, &x);
        let x_plus_x = b.mobius_add(&x, &x).expect("ok");
        for (a, c) in two_x.iter().zip(&x_plus_x) {
            assert!((a - c).abs() < 1e-9, "2⊗x: {a} vs {c}");
        }
    }

    #[test]
    fn sgd_descends_distance_to_target() {
        // Minimise f(x) = 0.5 d_c(x, target)²; its Riemannian gradient is -log_x(target).
        let b = PoincareBall::new(1.0).expect("ok");
        let target = vec![0.4, -0.3];
        let mut x = vec![-0.2, 0.1];
        let mut prev = b.distance(&x, &target).expect("ok");
        for _ in 0..200 {
            // grad_R (0.5 d²) = -log_x(target)
            let lg = b.log_map(&x, &target).expect("ok");
            let rgrad: Vec<f64> = lg.iter().map(|v| -v).collect();
            x = b.sgd_step(&x, &rgrad, 0.3).expect("ok");
            let d = b.distance(&x, &target).expect("ok");
            assert!(d <= prev + 1e-9, "distance increased: {prev} -> {d}");
            prev = d;
        }
        assert!(prev < 1e-3, "did not converge to target, residual {prev}");
    }

    #[test]
    fn frechet_mean_of_symmetric_points_is_origin() {
        // The Fréchet mean of {x, -x} is the origin by symmetry.
        let b = PoincareBall::new(1.0).expect("ok");
        let points = vec![0.3, 0.1, -0.3, -0.1];
        let m = poincare_frechet_mean(&b, &points, 2, 2, 100, 1e-10).expect("ok");
        for v in &m {
            assert!(v.abs() < 1e-6, "mean component {v} not ~0");
        }
    }

    #[test]
    fn frechet_mean_reduces_objective() {
        let b = PoincareBall::new(1.5).expect("ok");
        let mut rng = LcgRng::new(2026);
        let n = 12;
        let dim = 3;
        let mut points = vec![0.0; n * dim];
        for i in 0..n {
            let raw: Vec<f64> = (0..dim).map(|_| rng.next_range(-0.4, 0.4)).collect();
            let p = b.project(&raw);
            points[i * dim..(i + 1) * dim].copy_from_slice(&p);
        }
        let objective = |m: &[f64]| -> f64 {
            (0..n)
                .map(|i| {
                    let d = b.distance(m, &points[i * dim..(i + 1) * dim]).expect("ok");
                    d * d
                })
                .sum::<f64>()
                / n as f64
        };
        // Euclidean centroid (projected) as a baseline.
        let mut centroid = vec![0.0; dim];
        for i in 0..n {
            for k in 0..dim {
                centroid[k] += points[i * dim + k];
            }
        }
        for v in &mut centroid {
            *v /= n as f64;
        }
        centroid = b.project(&centroid);
        let baseline = objective(&centroid);
        let mean = poincare_frechet_mean(&b, &points, n, dim, 100, 1e-12).expect("ok");
        let refined = objective(&mean);
        assert!(
            refined <= baseline + 1e-9,
            "objective grew: {baseline} -> {refined}"
        );
    }

    #[test]
    fn random_point_inside_ball() {
        let b = PoincareBall::new(3.0).expect("ok");
        let mut rng = LcgRng::new(7);
        for _ in 0..50 {
            let p = b.random_point(4, &mut rng);
            assert!(b.contains(&p));
        }
    }

    #[test]
    fn dimension_mismatch_errors() {
        let b = unit();
        assert!(b.distance(&[0.1, 0.2], &[0.1]).is_err());
        assert!(b.mobius_add(&[0.1, 0.2], &[0.1]).is_err());
        assert!(b.exp_map(&[0.1, 0.2], &[0.1]).is_err());
        assert!(b.log_map(&[0.1, 0.2], &[0.1]).is_err());
    }
}
