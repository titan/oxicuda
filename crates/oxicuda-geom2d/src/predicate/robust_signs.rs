//! Sign predicates with explicit numerical tolerance.

/// Strict sign: -1 if `x < 0`, +1 if `x > 0`, else 0.
#[must_use]
pub fn sign_strict(x: f64) -> i8 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

/// Tolerant sign: 0 within `[-eps, eps]`.
#[must_use]
pub fn sign_with_eps(x: f64, eps: f64) -> i8 {
    if x > eps {
        1
    } else if x < -eps {
        -1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_signs() {
        assert_eq!(sign_strict(1.0), 1);
        assert_eq!(sign_strict(-1.0), -1);
        assert_eq!(sign_strict(0.0), 0);
    }

    #[test]
    fn eps_signs() {
        assert_eq!(sign_with_eps(1e-9, 1e-7), 0);
        assert_eq!(sign_with_eps(-1e-9, 1e-7), 0);
        assert_eq!(sign_with_eps(0.001, 1e-7), 1);
        assert_eq!(sign_with_eps(-0.001, 1e-7), -1);
    }
}
