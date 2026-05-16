//! Robin boundary condition: `alpha * u + beta * du/dn = g(x)`.

use crate::error::{PdeError, PdeResult};

/// Robin BC: `alpha * u + beta * du/dn = gamma`.
#[derive(Debug, Clone, Copy)]
pub struct RobinBc {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl RobinBc {
    pub const fn new(alpha: f64, beta: f64, gamma: f64) -> Self {
        Self { alpha, beta, gamma }
    }

    /// Validate that not both alpha and beta are zero.
    pub fn check(&self) -> PdeResult<()> {
        if self.alpha.abs() < 1.0e-15 && self.beta.abs() < 1.0e-15 {
            return Err(PdeError::InvalidParameter {
                name: "robin".into(),
                reason: "both alpha and beta are zero".into(),
            });
        }
        Ok(())
    }

    /// Express as 1D-right-boundary modification: returns `(diag_add, rhs_add)`
    /// such that the boundary equation `u[N] - h/beta * (alpha u[N] - gamma) = ...`
    /// is enforced for `beta != 0`.
    pub fn one_d_right_corrections(&self, h: f64) -> PdeResult<(f64, f64)> {
        self.check()?;
        if self.beta.abs() < 1.0e-15 {
            // Pure Dirichlet: u[N] = gamma/alpha
            return Ok((1.0, self.gamma / self.alpha));
        }
        // Approx: (u[N] - u[N-1])/h = (gamma - alpha u[N])/beta
        // => u[N](1 + h*alpha/beta) = u[N-1] + h*gamma/beta
        let diag_add = 1.0 + h * self.alpha / self.beta;
        let rhs_add = h * self.gamma / self.beta;
        Ok((diag_add, rhs_add))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robin_check_zero_zero_errors() {
        let bc = RobinBc::new(0.0, 0.0, 1.0);
        assert!(bc.check().is_err());
    }

    #[test]
    fn robin_pure_dirichlet_branch() {
        let bc = RobinBc::new(2.0, 0.0, 4.0);
        let (d, r) = bc.one_d_right_corrections(0.1).expect("ok");
        assert!((d - 1.0).abs() < 1.0e-12);
        assert!((r - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn robin_mixed_corrections() {
        let bc = RobinBc::new(1.0, 1.0, 0.5);
        let (d, r) = bc.one_d_right_corrections(0.1).expect("ok");
        assert!((d - 1.1).abs() < 1.0e-12);
        assert!((r - 0.05).abs() < 1.0e-12);
    }
}
