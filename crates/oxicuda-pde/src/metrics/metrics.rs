//! Norm and convergence-rate metrics.

use crate::error::{PdeError, PdeResult};

/// Discrete L^2 norm on a uniform 1D grid: `||u||_2 = sqrt(h * sum_i u_i^2)`.
pub fn l2_norm_1d(u: &[f64], h: f64) -> PdeResult<f64> {
    if u.is_empty() {
        return Err(PdeError::EmptyMesh("l2_norm_1d: empty vector".into()));
    }
    if h <= 0.0 {
        return Err(PdeError::InvalidParameter {
            name: "h".into(),
            reason: "must be positive".into(),
        });
    }
    let s: f64 = u.iter().map(|x| x * x).sum();
    Ok((h * s).sqrt())
}

/// Discrete H^1 seminorm on a uniform 1D grid:
/// `|u|_{H^1} = sqrt(h * sum_i ((u_{i+1} - u_i)/h)^2)`.
pub fn h1_seminorm_1d(u: &[f64], h: f64) -> PdeResult<f64> {
    if u.len() < 2 {
        return Err(PdeError::EmptyMesh("h1_seminorm needs n>=2".into()));
    }
    if h <= 0.0 {
        return Err(PdeError::InvalidParameter {
            name: "h".into(),
            reason: "must be positive".into(),
        });
    }
    let mut s = 0.0;
    for i in 0..u.len() - 1 {
        let d = (u[i + 1] - u[i]) / h;
        s += d * d;
    }
    Ok((h * s).sqrt())
}

/// Max-norm (l-infinity) of a vector.
pub fn max_norm(u: &[f64]) -> f64 {
    u.iter().fold(0.0_f64, |a, &x| a.max(x.abs()))
}

/// Estimate the convergence order from a pair of (h, error) measurements:
/// `order = log(e1/e2) / log(h1/h2)`.
pub fn convergence_order(h1: f64, e1: f64, h2: f64, e2: f64) -> PdeResult<f64> {
    if h1 <= 0.0 || h2 <= 0.0 || e1 <= 0.0 || e2 <= 0.0 {
        return Err(PdeError::InvalidParameter {
            name: "convergence_order inputs".into(),
            reason: "all must be positive".into(),
        });
    }
    if (h1 / h2 - 1.0).abs() < 1.0e-15 {
        return Err(PdeError::InvalidParameter {
            name: "h1/h2".into(),
            reason: "h1==h2".into(),
        });
    }
    Ok((e1 / e2).ln() / (h1 / h2).ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_norm_unit() {
        let u = vec![1.0; 11];
        let h = 0.1;
        let nrm = l2_norm_1d(&u, h).expect("ok");
        // sqrt(0.1 * 11) = sqrt(1.1)
        assert!((nrm - 1.1_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn h1_seminorm_linear() {
        // u = x linear from 0 to 1, du/dx = 1 -> seminorm = sqrt(h * n * 1^2) but with n-1 intervals
        let n = 11;
        let h = 0.1;
        let u: Vec<f64> = (0..n).map(|i| i as f64 * h).collect();
        let s = h1_seminorm_1d(&u, h).expect("ok");
        // 10 intervals of (1)^2 * h = 1.0 -> sqrt(1.0)
        assert!((s - 1.0).abs() < 1e-12);
    }

    #[test]
    fn max_norm_basic() {
        assert!((max_norm(&[1.0, -3.0, 2.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn convergence_order_o2() {
        // Errors quartered when h is halved => order 2
        let order = convergence_order(0.1, 1.0e-2, 0.05, 2.5e-3).expect("ok");
        assert!((order - 2.0).abs() < 1e-9);
    }

    #[test]
    fn empty_vector_errors() {
        let res = l2_norm_1d(&[], 0.1);
        assert!(res.is_err());
    }
}
