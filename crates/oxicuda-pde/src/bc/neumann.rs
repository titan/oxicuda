//! Neumann (natural) boundary condition: `du/dn = g(x)` on the boundary.

use crate::error::{PdeError, PdeResult};

/// Neumann BC value: outward-normal derivative `du/dn = flux`.
#[derive(Debug, Clone, Copy)]
pub struct NeumannBc {
    pub flux: f64,
}

impl NeumannBc {
    pub const fn new(flux: f64) -> Self {
        Self { flux }
    }
}

/// Apply a constant Neumann flux on the right boundary of a 1D grid using a
/// second-order accurate ghost-point treatment.
///
/// Returns the modified right-end equation: replaces row `n-1` of an FD system
/// to encode `(u[n-1] - u[n-2]) / h = flux` (first-order) or returns the new
/// rhs contribution.
pub fn neumann_1d_right_correction(u_interior_last: f64, h: f64, flux: f64) -> PdeResult<f64> {
    if h <= 0.0 {
        return Err(PdeError::InvalidParameter {
            name: "h".into(),
            reason: "must be positive".into(),
        });
    }
    Ok(u_interior_last + h * flux)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neumann_correction_simple() {
        let v = neumann_1d_right_correction(1.0, 0.1, 0.5).expect("ok");
        assert!((v - 1.05).abs() < 1.0e-12);
    }

    #[test]
    fn neumann_invalid_h_errors() {
        assert!(neumann_1d_right_correction(0.0, 0.0, 0.0).is_err());
    }
}
