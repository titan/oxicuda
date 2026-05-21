//! Hard boundary condition enforcement via output transform.
//!
//! References:
//!
//! - Lagaris, I. E., Likas, A., & Fotiadis, D. I. (1998). *Artificial neural
//!   networks for solving ordinary and partial differential equations*.
//!   IEEE Transactions on Neural Networks, 9(5), 987–1000.
//! - Berg, J. & Nyström, K. (2018). *A unified deep artificial neural network
//!   approach to partial differential equations in complex geometries*.
//!   Neurocomputing, 317, 28–41.
//!
//! Instead of penalising boundary error with an additional MSE term (the
//! "soft" Dirichlet loss `Σ (u_θ − g)²`), this module enforces
//! `u = g` on `∂Ω` **exactly** by reparameterising the network output:
//!
//! ```text
//! û(x) = g(x) + B(x) · N_θ(x)
//! ```
//!
//! with `B(x) = 0` on `∂Ω` and `B(x) > 0` inside `Ω`. Any choice of network
//! weights `θ` satisfies the Dirichlet condition by construction, so the loss
//! reduces to the interior PDE residual only — no boundary penalty term, no
//! competing-objective trade-off, and no boundary-loss weighting tuning.
//!
//! Two domain types are provided:
//!
//! - `Interval { lo, hi }`: 1-D interval `[lo, hi]`. Window
//!   `B(x) = (x − lo) · (hi − x) / r²` with `r = (hi − lo)/2`, so
//!   `B(lo) = B(hi) = 0` exactly and `B((lo + hi)/2) = 1`.
//! - `Rectangle { x_lo, x_hi, y_lo, y_hi }`: 2-D axis-aligned box. The window
//!   is the **separable product** `B(x, y) = B_x(x) · B_y(y)` of the two
//!   per-axis 1-D windows.
//!
//! The chain rule for the gradient of `û` is
//!
//! ```text
//! ∂û/∂x_i = ∂g/∂x_i + (∂B/∂x_i) · N_θ(x) + B(x) · ∂N_θ/∂x_i,
//! ```
//!
//! which is provided by [`HardBc::apply_grad`] as a helper that takes both
//! `N_θ(x)` and its spatial gradient as arguments.

use crate::error::{PinnError, PinnResult};

// ─── Domain ──────────────────────────────────────────────────────────────────

/// Supported domain geometries for hard Dirichlet boundary enforcement.
#[derive(Debug, Clone)]
pub enum BoundaryDomain {
    /// 1-D interval `[lo, hi]` with window
    /// `B(x) = (x − lo)·(hi − x) / ((hi − lo)/2)²`.
    Interval {
        /// Left endpoint of the interval.
        lo: f32,
        /// Right endpoint of the interval (must satisfy `hi > lo`).
        hi: f32,
    },
    /// 2-D axis-aligned rectangle. The window is the separable product of two
    /// 1-D `Interval` windows along `x` and `y`.
    Rectangle {
        /// Lower `x` bound.
        x_lo: f32,
        /// Upper `x` bound (must satisfy `x_hi > x_lo`).
        x_hi: f32,
        /// Lower `y` bound.
        y_lo: f32,
        /// Upper `y` bound (must satisfy `y_hi > y_lo`).
        y_hi: f32,
    },
}

/// Configuration for [`HardBc`].
#[derive(Debug, Clone)]
pub struct HardBcConfig {
    /// Domain geometry (1-D interval or 2-D rectangle).
    pub domain: BoundaryDomain,
    /// Spatial dimension of the domain (must be `1` or `2`).
    pub dim: usize,
}

// ─── 1-D window helper ───────────────────────────────────────────────────────

/// Compute the 1-D interval window `B(x) = (x − lo) · (hi − x) / r²`
/// with `r = (hi − lo)/2`.
///
/// Properties:
///
/// - `B(lo) = B(hi) = 0` (exactly, by construction).
/// - `B((lo + hi)/2) = 1` (the centre normalisation).
/// - `B(x) > 0` for `x ∈ (lo, hi)`, `B(x) < 0` for `x ∉ [lo, hi]`.
#[inline]
fn interval_b(x: f32, lo: f32, hi: f32) -> f32 {
    let r = (hi - lo) * 0.5;
    let denom = r * r;
    ((x - lo) * (hi - x)) / denom
}

/// Derivative of [`interval_b`] in `x`:
/// `B'(x) = (lo + hi − 2x) / r²` with `r = (hi − lo)/2`.
#[inline]
fn interval_b_grad(x: f32, lo: f32, hi: f32) -> f32 {
    let r = (hi - lo) * 0.5;
    let denom = r * r;
    (lo + hi - 2.0 * x) / denom
}

// ─── HardBc ──────────────────────────────────────────────────────────────────

/// Hard Dirichlet enforcement via output transform `û = g + B · N`.
///
/// The boundary-value function `g` is supplied as a closure of type
/// `F: Fn(&[f32]) -> f32`. It is the user's responsibility to ensure `g` is
/// the desired prescribed Dirichlet value on `∂Ω`; the transform does not
/// alter `g` at the boundary, only multiplies the network output by the
/// vanishing window `B`.
pub struct HardBc<F>
where
    F: Fn(&[f32]) -> f32,
{
    cfg: HardBcConfig,
    g_fn: F,
}

impl<F> HardBc<F>
where
    F: Fn(&[f32]) -> f32,
{
    /// Construct a new hard-BC transform.
    ///
    /// # Errors
    ///
    /// - [`PinnError::DimensionMismatch`] if `cfg.dim ∉ {1, 2}` or if `cfg.dim`
    ///   does not match the variant of `cfg.domain`.
    /// - [`PinnError::InvalidTimeInterval`] if any 1-D window has `lo ≥ hi`.
    pub fn new(cfg: HardBcConfig, g_fn: F) -> PinnResult<Self> {
        validate_cfg(&cfg)?;
        Ok(Self { cfg, g_fn })
    }

    /// Return the configured spatial dimension.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.cfg.dim
    }

    /// Borrow the configuration.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &HardBcConfig {
        &self.cfg
    }

    /// Compute the window factor `B(x)`.
    ///
    /// `B(x) = 0` exactly on `∂Ω` and `B(x) > 0` inside `Ω`. For 2-D rectangles
    /// the value is the separable product of two 1-D windows.
    ///
    /// # Errors
    ///
    /// - [`PinnError::DimensionMismatch`] if `x.len() != dim`.
    pub fn b_factor(&self, x: &[f32]) -> PinnResult<f32> {
        if x.len() != self.cfg.dim {
            return Err(PinnError::DimensionMismatch {
                expected: self.cfg.dim,
                got: x.len(),
            });
        }
        let v = match &self.cfg.domain {
            BoundaryDomain::Interval { lo, hi } => interval_b(x[0], *lo, *hi),
            BoundaryDomain::Rectangle {
                x_lo,
                x_hi,
                y_lo,
                y_hi,
            } => interval_b(x[0], *x_lo, *x_hi) * interval_b(x[1], *y_lo, *y_hi),
        };
        Ok(v)
    }

    /// Compute `∇B(x)` (one entry per spatial dimension).
    ///
    /// For the 2-D separable product `B(x, y) = B_x(x) · B_y(y)`, the chain
    /// rule yields `(∂B/∂x, ∂B/∂y) = (B'_x · B_y, B_x · B'_y)`.
    ///
    /// # Errors
    ///
    /// - [`PinnError::DimensionMismatch`] if `x.len() != dim`.
    pub fn b_grad(&self, x: &[f32]) -> PinnResult<Vec<f32>> {
        if x.len() != self.cfg.dim {
            return Err(PinnError::DimensionMismatch {
                expected: self.cfg.dim,
                got: x.len(),
            });
        }
        match &self.cfg.domain {
            BoundaryDomain::Interval { lo, hi } => Ok(vec![interval_b_grad(x[0], *lo, *hi)]),
            BoundaryDomain::Rectangle {
                x_lo,
                x_hi,
                y_lo,
                y_hi,
            } => {
                let bx = interval_b(x[0], *x_lo, *x_hi);
                let by = interval_b(x[1], *y_lo, *y_hi);
                let dbx = interval_b_grad(x[0], *x_lo, *x_hi);
                let dby = interval_b_grad(x[1], *y_lo, *y_hi);
                Ok(vec![dbx * by, bx * dby])
            }
        }
    }

    /// Apply the output transform: `û(x) = g(x) + B(x) · N_θ(x)`.
    ///
    /// `net_value` is the scalar value `N_θ(x)` produced by the underlying
    /// network at `x`. The returned value satisfies `û(x) = g(x)` exactly
    /// whenever `x ∈ ∂Ω` (where `B(x) = 0`), regardless of `net_value`.
    ///
    /// # Errors
    ///
    /// - [`PinnError::DimensionMismatch`] if `x.len() != dim`.
    pub fn apply(&self, x: &[f32], net_value: f32) -> PinnResult<f32> {
        let b = self.b_factor(x)?;
        let g = (self.g_fn)(x);
        Ok(g + b * net_value)
    }

    /// Apply the chain rule for the spatial gradient:
    /// `∂û/∂x_i = ∂g/∂x_i + (∂B/∂x_i) · N_θ(x) + B(x) · ∂N_θ/∂x_i`.
    ///
    /// `net_grad_x` and `g_grad_x` are the spatial gradients of the network
    /// output and of the prescribed boundary-value function `g`, respectively,
    /// each of length `dim`.
    ///
    /// # Errors
    ///
    /// - [`PinnError::DimensionMismatch`] if `x.len() != dim`,
    ///   `net_grad_x.len() != dim`, or `g_grad_x.len() != dim`.
    pub fn apply_grad(
        &self,
        x: &[f32],
        net_value: f32,
        net_grad_x: &[f32],
        g_grad_x: &[f32],
    ) -> PinnResult<Vec<f32>> {
        if x.len() != self.cfg.dim {
            return Err(PinnError::DimensionMismatch {
                expected: self.cfg.dim,
                got: x.len(),
            });
        }
        if net_grad_x.len() != self.cfg.dim {
            return Err(PinnError::DimensionMismatch {
                expected: self.cfg.dim,
                got: net_grad_x.len(),
            });
        }
        if g_grad_x.len() != self.cfg.dim {
            return Err(PinnError::DimensionMismatch {
                expected: self.cfg.dim,
                got: g_grad_x.len(),
            });
        }
        let b = self.b_factor(x)?;
        let db = self.b_grad(x)?;
        let mut out = Vec::with_capacity(self.cfg.dim);
        for i in 0..self.cfg.dim {
            out.push(g_grad_x[i] + db[i] * net_value + b * net_grad_x[i]);
        }
        Ok(out)
    }
}

// ─── Validation ──────────────────────────────────────────────────────────────

fn validate_cfg(cfg: &HardBcConfig) -> PinnResult<()> {
    if cfg.dim == 0 || cfg.dim > 2 {
        return Err(PinnError::DimensionMismatch {
            expected: 1,
            got: cfg.dim,
        });
    }
    match &cfg.domain {
        BoundaryDomain::Interval { lo, hi } => {
            if cfg.dim != 1 {
                return Err(PinnError::DimensionMismatch {
                    expected: 1,
                    got: cfg.dim,
                });
            }
            if !valid_interval(*lo, *hi) {
                return Err(PinnError::InvalidTimeInterval { t0: *lo, t1: *hi });
            }
        }
        BoundaryDomain::Rectangle {
            x_lo,
            x_hi,
            y_lo,
            y_hi,
        } => {
            if cfg.dim != 2 {
                return Err(PinnError::DimensionMismatch {
                    expected: 2,
                    got: cfg.dim,
                });
            }
            if !valid_interval(*x_lo, *x_hi) {
                return Err(PinnError::InvalidTimeInterval {
                    t0: *x_lo,
                    t1: *x_hi,
                });
            }
            if !valid_interval(*y_lo, *y_hi) {
                return Err(PinnError::InvalidTimeInterval {
                    t0: *y_lo,
                    t1: *y_hi,
                });
            }
        }
    }
    Ok(())
}

/// True iff `lo` and `hi` are both finite and `lo < hi` under `partial_cmp`
/// (i.e. neither value is NaN and they are strictly ordered).
fn valid_interval(lo: f32, hi: f32) -> bool {
    if !(lo.is_finite() && hi.is_finite()) {
        return false;
    }
    matches!(lo.partial_cmp(&hi), Some(core::cmp::Ordering::Less))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn iv_cfg(lo: f32, hi: f32) -> HardBcConfig {
        HardBcConfig {
            domain: BoundaryDomain::Interval { lo, hi },
            dim: 1,
        }
    }

    fn rect_cfg(x_lo: f32, x_hi: f32, y_lo: f32, y_hi: f32) -> HardBcConfig {
        HardBcConfig {
            domain: BoundaryDomain::Rectangle {
                x_lo,
                x_hi,
                y_lo,
                y_hi,
            },
            dim: 2,
        }
    }

    // ── b_factor: boundary == 0 (EXACT) ─────────────────────────────────────

    #[test]
    fn b_factor_interval_zero_at_boundaries_exact() {
        let h = HardBc::new(iv_cfg(0.0, 1.0), |_x| 0.0).unwrap();
        // EXACT zero at lo and hi: (x - lo) is identically 0 at x=lo,
        // (hi - x) is identically 0 at x=hi.
        assert_eq!(h.b_factor(&[0.0]).unwrap(), 0.0);
        assert_eq!(h.b_factor(&[1.0]).unwrap(), 0.0);
    }

    #[test]
    fn b_factor_interval_one_at_centre() {
        let h = HardBc::new(iv_cfg(0.0, 1.0), |_x| 0.0).unwrap();
        let v = h.b_factor(&[0.5]).unwrap();
        assert!((v - 1.0).abs() < 1e-6, "B(centre) = {v}, expected 1.0");
    }

    #[test]
    fn b_factor_interval_strictly_positive_inside() {
        let h = HardBc::new(iv_cfg(-2.0, 3.0), |_x| 0.0).unwrap();
        for k in 1..10 {
            let x = -2.0 + (k as f32) * 0.5;
            let v = h.b_factor(&[x]).unwrap();
            assert!(v > 0.0, "B should be > 0 strictly inside at x={x}: got {v}");
        }
    }

    #[test]
    fn b_factor_rectangle_zero_at_all_four_corners_exact() {
        let h = HardBc::new(rect_cfg(0.0, 1.0, 0.0, 2.0), |_x| 0.0).unwrap();
        let corners = [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 2.0]];
        for c in corners {
            assert_eq!(
                h.b_factor(&c).unwrap(),
                0.0,
                "B at corner {c:?} must be EXACTLY 0"
            );
        }
    }

    #[test]
    fn b_factor_rectangle_zero_at_edge_midpoints() {
        let h = HardBc::new(rect_cfg(0.0, 1.0, 0.0, 2.0), |_x| 0.0).unwrap();
        // Midpoints of the four edges: one coordinate sits at lo/hi → product is 0.
        let edges = [[0.5, 0.0], [0.5, 2.0], [0.0, 1.0], [1.0, 1.0]];
        for e in edges {
            assert_eq!(
                h.b_factor(&e).unwrap(),
                0.0,
                "B at edge mid {e:?} must be 0"
            );
        }
    }

    #[test]
    fn b_factor_rectangle_one_at_centre() {
        let h = HardBc::new(rect_cfg(0.0, 1.0, 0.0, 2.0), |_x| 0.0).unwrap();
        let v = h.b_factor(&[0.5, 1.0]).unwrap();
        // B(centre) = B_x(0.5) * B_y(1.0) = 1 * 1.
        assert!((v - 1.0).abs() < 1e-6, "B(rect centre) = {v}, expected 1.0");
    }

    // ── b_grad ──────────────────────────────────────────────────────────────

    #[test]
    fn b_grad_interval_zero_at_centre_by_symmetry() {
        let h = HardBc::new(iv_cfg(-1.0, 1.0), |_x| 0.0).unwrap();
        let g = h.b_grad(&[0.0]).unwrap();
        assert_eq!(g.len(), 1);
        assert!(
            g[0].abs() < 1e-6,
            "B'(centre) = {} by symmetry, expected 0",
            g[0]
        );
    }

    #[test]
    fn b_grad_interval_matches_finite_diff() {
        let h = HardBc::new(iv_cfg(-2.0, 3.0), |_x| 0.0).unwrap();
        let eps = 1e-3_f32;
        for k in 1..8 {
            let x = -1.5 + (k as f32) * 0.5;
            let analytical = h.b_grad(&[x]).unwrap()[0];
            let bp = h.b_factor(&[x + eps]).unwrap();
            let bm = h.b_factor(&[x - eps]).unwrap();
            let numerical = (bp - bm) / (2.0 * eps);
            assert!(
                (analytical - numerical).abs() < 1e-2,
                "B' mismatch at x={x}: analytical={analytical}, numerical={numerical}"
            );
        }
    }

    #[test]
    fn b_grad_rectangle_length_is_two() {
        let h = HardBc::new(rect_cfg(0.0, 1.0, 0.0, 2.0), |_x| 0.0).unwrap();
        let g = h.b_grad(&[0.3, 0.7]).unwrap();
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn b_grad_rectangle_matches_finite_diff() {
        let h = HardBc::new(rect_cfg(0.0, 2.0, -1.0, 1.0), |_x| 0.0).unwrap();
        let eps = 1e-3_f32;
        let pts = [[0.7, 0.0], [1.2, 0.4], [0.3, -0.6]];
        for p in pts {
            let analytical = h.b_grad(&p).unwrap();
            let bx_plus = h.b_factor(&[p[0] + eps, p[1]]).unwrap();
            let bx_minus = h.b_factor(&[p[0] - eps, p[1]]).unwrap();
            let by_plus = h.b_factor(&[p[0], p[1] + eps]).unwrap();
            let by_minus = h.b_factor(&[p[0], p[1] - eps]).unwrap();
            let dx_num = (bx_plus - bx_minus) / (2.0 * eps);
            let dy_num = (by_plus - by_minus) / (2.0 * eps);
            assert!(
                (analytical[0] - dx_num).abs() < 1e-2,
                "∂B/∂x at {p:?} mismatch {} vs {}",
                analytical[0],
                dx_num
            );
            assert!(
                (analytical[1] - dy_num).abs() < 1e-2,
                "∂B/∂y at {p:?} mismatch {} vs {}",
                analytical[1],
                dy_num
            );
        }
    }

    // ── apply: hard-BC defining property ────────────────────────────────────

    #[test]
    fn apply_at_boundary_equals_g_exactly_interval() {
        // The defining property: û(boundary) = g(boundary) EXACTLY, no matter
        // what the network produces, no matter the network gradient.
        let g_fn = |x: &[f32]| 3.7 * x[0] + 1.1;
        let h = HardBc::new(iv_cfg(-1.0, 2.0), g_fn).unwrap();
        for net in [-100.0_f32, -1.0, 0.0, 1.0, 1e6] {
            let lo_val = h.apply(&[-1.0], net).unwrap();
            let hi_val = h.apply(&[2.0], net).unwrap();
            assert_eq!(
                lo_val,
                g_fn(&[-1.0]),
                "Hard-BC violated at lo for net={net}"
            );
            assert_eq!(hi_val, g_fn(&[2.0]), "Hard-BC violated at hi for net={net}");
        }
    }

    #[test]
    fn apply_at_boundary_equals_g_exactly_rectangle() {
        let g_fn = |x: &[f32]| 2.0 * x[0] - x[1] + 0.5;
        let h = HardBc::new(rect_cfg(0.0, 1.0, 0.0, 2.0), g_fn).unwrap();
        let boundary_pts = [[0.0, 0.5], [1.0, 1.5], [0.4, 0.0], [0.7, 2.0]];
        for p in boundary_pts {
            for net in [-50.0_f32, 0.0, 7.3] {
                let v = h.apply(&p, net).unwrap();
                assert_eq!(v, g_fn(&p), "Hard-BC violated at {p:?} for net={net}");
            }
        }
    }

    #[test]
    fn apply_interior_with_zero_net_equals_g() {
        let g_fn = |x: &[f32]| (x[0] * x[0]).cos();
        let h = HardBc::new(iv_cfg(0.0, 1.0), g_fn).unwrap();
        for k in 1..10 {
            let x = (k as f32) * 0.1;
            let v = h.apply(&[x], 0.0).unwrap();
            assert!(
                (v - g_fn(&[x])).abs() < 1e-6,
                "û(interior, net=0) should equal g; got {v} vs {}",
                g_fn(&[x])
            );
        }
    }

    #[test]
    fn apply_is_linear_in_net_value() {
        let g_fn = |_x: &[f32]| 0.0;
        let h = HardBc::new(iv_cfg(0.0, 1.0), g_fn).unwrap();
        // û(x, α·N) = g + B·α·N = α · (g + B·N) only when g=0; the
        // stronger linearity statement is û(x, N_1+N_2) − g = (û(x, N_1) − g)
        // + (û(x, N_2) − g), i.e. (û − g) is linear in N.
        let x = [0.3_f32];
        let n1 = 2.5_f32;
        let n2 = -1.7_f32;
        let v1 = h.apply(&x, n1).unwrap();
        let v2 = h.apply(&x, n2).unwrap();
        let v_sum = h.apply(&x, n1 + n2).unwrap();
        assert!(
            (v_sum - (v1 + v2)).abs() < 1e-5,
            "Linearity violated: {v_sum} vs {} + {} = {}",
            v1,
            v2,
            v1 + v2
        );
    }

    // ── apply_grad ──────────────────────────────────────────────────────────

    #[test]
    fn apply_grad_length_equals_dim_interval() {
        let h = HardBc::new(iv_cfg(0.0, 1.0), |_x| 0.0).unwrap();
        let g = h.apply_grad(&[0.3], 1.2, &[0.5], &[0.0]).unwrap();
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn apply_grad_length_equals_dim_rectangle() {
        let h = HardBc::new(rect_cfg(0.0, 1.0, 0.0, 1.0), |_x| 0.0).unwrap();
        let g = h
            .apply_grad(&[0.3, 0.7], 1.2, &[0.4, 0.6], &[0.0, 0.0])
            .unwrap();
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn apply_grad_matches_finite_diff_known_net() {
        // Define a simple known "net": N(x) = 2x + 0.5, with ∂N/∂x = 2.
        // g(x) = sin(x), ∂g/∂x = cos(x).
        let g_fn = |x: &[f32]| x[0].sin();
        let dg_fn = |x: f32| x.cos();
        let net_fn = |x: f32| 2.0 * x + 0.5;
        let dnet_dx = 2.0_f32;
        let h = HardBc::new(iv_cfg(0.0, 1.0), g_fn).unwrap();
        let eps = 1e-3_f32;
        for k in 1..8 {
            let x = 0.05 + (k as f32) * 0.1;
            let analytic = h
                .apply_grad(&[x], net_fn(x), &[dnet_dx], &[dg_fn(x)])
                .unwrap()[0];
            let up = h.apply(&[x + eps], net_fn(x + eps)).unwrap();
            let dn = h.apply(&[x - eps], net_fn(x - eps)).unwrap();
            let numerical = (up - dn) / (2.0 * eps);
            assert!(
                (analytic - numerical).abs() < 5e-3,
                "apply_grad mismatch at x={x}: analytic={analytic}, numerical={numerical}"
            );
        }
    }

    #[test]
    fn apply_grad_rectangle_matches_finite_diff_known_net() {
        let g_fn = |x: &[f32]| x[0] + 2.0 * x[1];
        let h = HardBc::new(rect_cfg(0.0, 1.0, 0.0, 1.0), g_fn).unwrap();
        // N(x, y) = 3x − y + 0.4, ∂N/∂x = 3, ∂N/∂y = −1.
        let net_at = |x: f32, y: f32| 3.0 * x - y + 0.4;
        let eps = 1e-3_f32;
        let pts = [[0.3_f32, 0.4], [0.7, 0.6]];
        for p in pts {
            let analytic = h
                .apply_grad(&p, net_at(p[0], p[1]), &[3.0, -1.0], &[1.0, 2.0])
                .unwrap();
            let upx = h
                .apply(&[p[0] + eps, p[1]], net_at(p[0] + eps, p[1]))
                .unwrap();
            let dnx = h
                .apply(&[p[0] - eps, p[1]], net_at(p[0] - eps, p[1]))
                .unwrap();
            let upy = h
                .apply(&[p[0], p[1] + eps], net_at(p[0], p[1] + eps))
                .unwrap();
            let dny = h
                .apply(&[p[0], p[1] - eps], net_at(p[0], p[1] - eps))
                .unwrap();
            let dx_num = (upx - dnx) / (2.0 * eps);
            let dy_num = (upy - dny) / (2.0 * eps);
            assert!(
                (analytic[0] - dx_num).abs() < 5e-3,
                "∂û/∂x at {p:?}: analytic {} vs numerical {}",
                analytic[0],
                dx_num
            );
            assert!(
                (analytic[1] - dy_num).abs() < 5e-3,
                "∂û/∂y at {p:?}: analytic {} vs numerical {}",
                analytic[1],
                dy_num
            );
        }
    }

    // ── Errors ──────────────────────────────────────────────────────────────

    #[test]
    fn err_dim_zero() {
        let cfg = HardBcConfig {
            domain: BoundaryDomain::Interval { lo: 0.0, hi: 1.0 },
            dim: 0,
        };
        let r = HardBc::new(cfg, |_x| 0.0);
        assert!(matches!(r, Err(PinnError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_dim_three() {
        let cfg = HardBcConfig {
            domain: BoundaryDomain::Interval { lo: 0.0, hi: 1.0 },
            dim: 3,
        };
        let r = HardBc::new(cfg, |_x| 0.0);
        assert!(matches!(r, Err(PinnError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_dim_mismatches_domain_interval() {
        // Interval domain but dim=2 → mismatch.
        let cfg = HardBcConfig {
            domain: BoundaryDomain::Interval { lo: 0.0, hi: 1.0 },
            dim: 2,
        };
        let r = HardBc::new(cfg, |_x| 0.0);
        assert!(matches!(r, Err(PinnError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_dim_mismatches_domain_rectangle() {
        // Rectangle domain but dim=1 → mismatch.
        let cfg = HardBcConfig {
            domain: BoundaryDomain::Rectangle {
                x_lo: 0.0,
                x_hi: 1.0,
                y_lo: 0.0,
                y_hi: 1.0,
            },
            dim: 1,
        };
        let r = HardBc::new(cfg, |_x| 0.0);
        assert!(matches!(r, Err(PinnError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_lo_ge_hi_interval() {
        let cfg = HardBcConfig {
            domain: BoundaryDomain::Interval { lo: 1.0, hi: 1.0 },
            dim: 1,
        };
        let r = HardBc::new(cfg, |_x| 0.0);
        assert!(matches!(r, Err(PinnError::InvalidTimeInterval { .. })));
    }

    #[test]
    fn err_x_lo_ge_x_hi_rectangle() {
        let cfg = HardBcConfig {
            domain: BoundaryDomain::Rectangle {
                x_lo: 1.0,
                x_hi: 0.0,
                y_lo: 0.0,
                y_hi: 1.0,
            },
            dim: 2,
        };
        let r = HardBc::new(cfg, |_x| 0.0);
        assert!(matches!(r, Err(PinnError::InvalidTimeInterval { .. })));
    }

    #[test]
    fn err_y_lo_ge_y_hi_rectangle() {
        let cfg = HardBcConfig {
            domain: BoundaryDomain::Rectangle {
                x_lo: 0.0,
                x_hi: 1.0,
                y_lo: 2.0,
                y_hi: 2.0,
            },
            dim: 2,
        };
        let r = HardBc::new(cfg, |_x| 0.0);
        assert!(matches!(r, Err(PinnError::InvalidTimeInterval { .. })));
    }

    #[test]
    fn err_b_factor_wrong_x_length() {
        let h = HardBc::new(iv_cfg(0.0, 1.0), |_x| 0.0).unwrap();
        let r = h.b_factor(&[0.5, 0.5]);
        assert!(matches!(r, Err(PinnError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_apply_grad_wrong_x_length() {
        let h = HardBc::new(rect_cfg(0.0, 1.0, 0.0, 1.0), |_x| 0.0).unwrap();
        let r = h.apply_grad(&[0.3], 1.0, &[0.0, 0.0], &[0.0, 0.0]);
        assert!(matches!(r, Err(PinnError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_apply_grad_wrong_net_grad_length() {
        let h = HardBc::new(rect_cfg(0.0, 1.0, 0.0, 1.0), |_x| 0.0).unwrap();
        let r = h.apply_grad(&[0.3, 0.4], 1.0, &[0.0], &[0.0, 0.0]);
        assert!(matches!(r, Err(PinnError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_apply_grad_wrong_g_grad_length() {
        let h = HardBc::new(rect_cfg(0.0, 1.0, 0.0, 1.0), |_x| 0.0).unwrap();
        let r = h.apply_grad(&[0.3, 0.4], 1.0, &[0.0, 0.0], &[0.0, 0.0, 0.0]);
        assert!(matches!(r, Err(PinnError::DimensionMismatch { .. })));
    }

    // ── Determinism ────────────────────────────────────────────────────────

    #[test]
    fn determinism_apply() {
        // Two independently-constructed HardBc instances with the same g_fn
        // should produce bit-identical apply results.
        let g_fn = |x: &[f32]| 0.7 * x[0] - 0.2;
        let h_a = HardBc::new(iv_cfg(0.0, 1.0), g_fn).unwrap();
        let h_b = HardBc::new(iv_cfg(0.0, 1.0), g_fn).unwrap();
        for k in 0..21 {
            let x = (k as f32) * 0.05;
            let net = (k as f32 - 10.0) * 0.3;
            let a = h_a.apply(&[x], net).unwrap();
            let b = h_b.apply(&[x], net).unwrap();
            assert_eq!(a, b);
        }
    }
}
