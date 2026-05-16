//! 1D nodal Discontinuous Galerkin with Legendre-Gauss-Lobatto nodes.

use crate::error::{PdeError, PdeResult};

/// Legendre-Gauss-Lobatto (LGL) nodes on `[-1, 1]` for polynomial order `p` (returns `p+1` nodes).
///
/// Special cases:
/// - p=1: {-1, 1}
/// - p=2: {-1, 0, 1}
/// - p=3: {-1, -1/sqrt(5), 1/sqrt(5), 1}
/// - p=4: {-1, -sqrt(3/7), 0, sqrt(3/7), 1}
///
/// For higher p, computed via Newton iteration on `P'_p(x) = 0`.
pub fn lgl_nodes(p: usize) -> PdeResult<Vec<f64>> {
    if p == 0 {
        return Err(PdeError::UnsupportedDegree(p));
    }
    if p == 1 {
        return Ok(vec![-1.0, 1.0]);
    }
    if p == 2 {
        return Ok(vec![-1.0, 0.0, 1.0]);
    }
    if p == 3 {
        let a = 1.0 / 5.0_f64.sqrt();
        return Ok(vec![-1.0, -a, a, 1.0]);
    }
    if p == 4 {
        let a = (3.0_f64 / 7.0).sqrt();
        return Ok(vec![-1.0, -a, 0.0, a, 1.0]);
    }
    if p > 8 {
        return Err(PdeError::UnsupportedDegree(p));
    }
    // Generic case: chebyshev-like initial guess + Newton iteration on P'_p(x) = 0
    let mut x = vec![0.0; p + 1];
    x[0] = -1.0;
    x[p] = 1.0;
    let pi = std::f64::consts::PI;
    for (i, xi) in x.iter_mut().enumerate().take(p).skip(1) {
        // Approximate (Chebyshev-Lobatto) interior LGL guess for roots of P'_p.
        *xi = -(pi * i as f64 / p as f64).cos();
    }
    // Newton iteration: f(x) = P'_p(x) = 0; f'(x) = P''_p(x).
    // Use the differential identity: (1-x^2) P''_p(x) = 2 x P'_p(x) - p (p+1) P_p(x).
    let max_iter = 200;
    for _ in 0..max_iter {
        let mut x_new = x.clone();
        let mut moved = false;
        for i in 1..p {
            let (p_val, dp_val) = legendre_p_and_pp(p, x[i]);
            // d^2 P_p / dx^2  =  (2 x dP - p(p+1) P) / (1 - x^2)
            let one_minus_x2 = 1.0 - x[i] * x[i];
            let denom = one_minus_x2.max(1.0e-300);
            let ddp_val = (2.0 * x[i] * dp_val - (p as f64) * (p as f64 + 1.0) * p_val) / denom;
            if ddp_val.abs() < 1.0e-300 {
                continue;
            }
            let dx = dp_val / ddp_val;
            x_new[i] -= dx;
            if dx.abs() > 1e-15 {
                moved = true;
            }
        }
        x = x_new;
        if !moved {
            break;
        }
    }
    // Symmetrize: enforce mirror symmetry exactly so that x[i] = -x[p-i].
    let half = p / 2;
    for i in 0..=half {
        let avg = 0.5 * (x[i] - x[p - i]);
        x[i] = avg;
        x[p - i] = -avg;
    }
    Ok(x)
}

/// Legendre polynomial `P_p(x)` and its derivative `P'_p(x)` using a 3-term recurrence.
fn legendre_p_and_pp(p: usize, x: f64) -> (f64, f64) {
    if p == 0 {
        return (1.0, 0.0);
    }
    if p == 1 {
        return (x, 1.0);
    }
    let mut p_prev = 1.0;
    let mut p_curr = x;
    let mut dp_prev = 0.0;
    let mut dp_curr = 1.0;
    for k in 2..=p {
        let kf = k as f64;
        let p_next = ((2.0 * kf - 1.0) * x * p_curr - (kf - 1.0) * p_prev) / kf;
        let dp_next = ((2.0 * kf - 1.0) * (p_curr + x * dp_curr) - (kf - 1.0) * dp_prev) / kf;
        p_prev = p_curr;
        p_curr = p_next;
        dp_prev = dp_curr;
        dp_curr = dp_next;
    }
    (p_curr, dp_curr)
}

/// LGL quadrature weights on `[-1,1]` for polynomial order `p`.
///
/// Formula: `w_i = 2 / (p*(p+1)*[P_p(x_i)]^2)` (interior); for endpoints,
/// the same formula gives `2/(p*(p+1))`.
pub fn lgl_weights(p: usize) -> PdeResult<Vec<f64>> {
    if p == 0 {
        return Err(PdeError::UnsupportedDegree(p));
    }
    let nodes = lgl_nodes(p)?;
    let mut w = vec![0.0; p + 1];
    let pf = p as f64;
    for i in 0..=p {
        let (lp, _) = legendre_p_and_pp(p, nodes[i]);
        w[i] = 2.0 / (pf * (pf + 1.0) * lp * lp);
    }
    Ok(w)
}

/// 1D nodal DG space on a uniform mesh of `n_elem` elements, polynomial degree `p`.
#[derive(Debug, Clone)]
pub struct Dg1dSpace {
    pub n_elem: usize,
    pub p: usize,
    pub x_left: f64,
    pub x_right: f64,
    pub lgl: Vec<f64>,
    pub weights: Vec<f64>,
    pub element_size: f64,
}

impl Dg1dSpace {
    pub fn new(n_elem: usize, p: usize, x_left: f64, x_right: f64) -> PdeResult<Self> {
        if n_elem == 0 {
            return Err(PdeError::EmptyMesh("dg1d: n_elem=0".into()));
        }
        if x_right <= x_left {
            return Err(PdeError::InvalidGrid("x_right must be > x_left".into()));
        }
        let lgl = lgl_nodes(p)?;
        let weights = lgl_weights(p)?;
        let element_size = (x_right - x_left) / n_elem as f64;
        Ok(Self {
            n_elem,
            p,
            x_left,
            x_right,
            lgl,
            weights,
            element_size,
        })
    }

    /// Returns the physical x-coordinates of the DOFs on element `e`.
    pub fn element_nodes(&self, e: usize) -> PdeResult<Vec<f64>> {
        if e >= self.n_elem {
            return Err(PdeError::IndexOutOfBounds {
                index: e,
                len: self.n_elem,
            });
        }
        let xl = self.x_left + e as f64 * self.element_size;
        Ok(self
            .lgl
            .iter()
            .map(|&xi| xl + 0.5 * self.element_size * (xi + 1.0))
            .collect())
    }

    /// Total number of DOFs (= `n_elem * (p+1)`).
    pub fn n_dofs(&self) -> usize {
        self.n_elem * (self.p + 1)
    }

    /// Diagonal mass matrix entries on each element (nodal basis is diagonal in LGL).
    pub fn mass_diag(&self) -> Vec<f64> {
        // M_ii = (h/2) * w_i for each element
        self.weights
            .iter()
            .map(|&w| 0.5 * self.element_size * w)
            .collect()
    }
}

/// Lax-Friedrichs flux for `u_t + a u_x = 0` on a 1D mesh interface.
///
/// `u_l` is the left state, `u_r` is the right state. Returns the flux.
/// `F_lf = 0.5*(F(u_l) + F(u_r)) - 0.5*max(|a|)*(u_r - u_l)`
pub fn lax_friedrichs_flux(u_l: f64, u_r: f64, a: f64) -> f64 {
    0.5 * a * (u_l + u_r) - 0.5 * a.abs() * (u_r - u_l)
}

/// Upwind flux for `u_t + a u_x = 0`. Returns the flux.
pub fn upwind_flux(u_l: f64, u_r: f64, a: f64) -> f64 {
    if a >= 0.0 { a * u_l } else { a * u_r }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lgl_nodes_known() {
        let p2 = lgl_nodes(2).expect("ok");
        assert_eq!(p2, vec![-1.0, 0.0, 1.0]);
        let p3 = lgl_nodes(3).expect("ok");
        assert!((p3[0] + 1.0).abs() < 1e-12);
        assert!((p3[3] - 1.0).abs() < 1e-12);
        assert!((p3[1] + 1.0 / 5.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn lgl_weights_sum_2() {
        for p in 1..=4 {
            let w = lgl_weights(p).expect("ok");
            let s: f64 = w.iter().sum();
            assert!((s - 2.0).abs() < 1e-10, "p={p} sum={s}");
        }
    }

    #[test]
    fn dg1d_space_dofs() {
        let s = Dg1dSpace::new(4, 3, 0.0, 1.0).expect("ok");
        assert_eq!(s.n_dofs(), 16);
        assert_eq!(s.lgl.len(), 4);
    }

    #[test]
    fn dg1d_element_nodes() {
        let s = Dg1dSpace::new(2, 2, 0.0, 1.0).expect("ok");
        let n0 = s.element_nodes(0).expect("ok");
        // LGL p=2 -> {-1, 0, 1}; mapped to [0, 0.5] -> {0, 0.25, 0.5}
        assert!((n0[0] - 0.0).abs() < 1e-12);
        assert!((n0[1] - 0.25).abs() < 1e-12);
        assert!((n0[2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn upwind_flux_positive_a() {
        assert!((upwind_flux(1.0, 2.0, 0.5) - 0.5).abs() < 1e-12);
        assert!((upwind_flux(1.0, 2.0, -0.5) - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn lax_friedrichs_dissipation() {
        // For a discontinuity (u_l=1, u_r=0) with a=0, LF flux = 0.5*0*(1) - 0.5*0*(−1) = 0
        let f0 = lax_friedrichs_flux(1.0, 0.0, 0.0);
        assert!((f0 - 0.0).abs() < 1e-12);
        // For a=1: f = 0.5*1*1 - 0.5*1*(-1) = 0.5 + 0.5 = 1.0
        let f1 = lax_friedrichs_flux(1.0, 0.0, 1.0);
        assert!((f1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn lgl_p5_nodes_symmetric() {
        let n = lgl_nodes(5).expect("ok");
        // Should have 6 nodes, symmetric about zero.
        assert_eq!(n.len(), 6);
        for i in 0..6 {
            assert!((n[i] + n[5 - i]).abs() < 1.0e-10);
        }
    }
}
