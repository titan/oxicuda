//! Suzuki-Trotter decomposition coefficients.
//!
//! For a Hamiltonian `H = A + B` (e.g. odd-bond + even-bond Heisenberg), we expose the
//! time-step factors of three orders:
//!
//! * 1st order: `(A, dt) (B, dt)` — error `O(dt²)`.
//! * 2nd order (Strang): `(A, dt/2) (B, dt) (A, dt/2)` — error `O(dt³)`.
//! * 4th order (Forest-Ruth / Suzuki): nested decomposition with five inner steps.

/// Order of the splitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrotterOrder {
    First,
    Second,
    Fourth,
}

/// Return a vector of (which_block, factor) pairs.
///
/// `which_block` is 0 for A (odd bonds) and 1 for B (even bonds).
/// `factor` multiplies `dt` for that sub-step.
#[must_use]
pub fn trotter_factors(order: TrotterOrder) -> Vec<(u8, f64)> {
    match order {
        TrotterOrder::First => vec![(0, 1.0), (1, 1.0)],
        TrotterOrder::Second => vec![(0, 0.5), (1, 1.0), (0, 0.5)],
        TrotterOrder::Fourth => {
            // Suzuki S4: w1 = 1/(2 - 2^{1/3})
            let w1 = 1.0 / (2.0 - 2f64.powf(1.0 / 3.0));
            let w0 = 1.0 - 2.0 * w1;
            // Each substep is a Strang step of weight w1 or w0:
            //   Strang(w) = (A, w/2)(B, w)(A, w/2)
            // Combined: (A, w1/2)(B, w1)(A, w1/2 + w0/2)(B, w0)(A, w0/2 + w1/2)(B, w1)(A, w1/2)
            let a1 = w1 / 2.0;
            let a2 = (w1 + w0) / 2.0;
            let a3 = (w0 + w1) / 2.0;
            let a4 = w1 / 2.0;
            vec![
                (0, a1),
                (1, w1),
                (0, a2),
                (1, w0),
                (0, a3),
                (1, w1),
                (0, a4),
            ]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum_factor(steps: &[(u8, f64)], block: u8) -> f64 {
        steps.iter().filter(|x| x.0 == block).map(|x| x.1).sum()
    }

    #[test]
    fn first_order_unit() {
        let s = trotter_factors(TrotterOrder::First);
        assert!((sum_factor(&s, 0) - 1.0).abs() < 1e-15);
        assert!((sum_factor(&s, 1) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn second_order_unit() {
        let s = trotter_factors(TrotterOrder::Second);
        assert!((sum_factor(&s, 0) - 1.0).abs() < 1e-15);
        assert!((sum_factor(&s, 1) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn fourth_order_unit() {
        let s = trotter_factors(TrotterOrder::Fourth);
        assert!((sum_factor(&s, 0) - 1.0).abs() < 1e-12);
        assert!((sum_factor(&s, 1) - 1.0).abs() < 1e-12);
    }
}
