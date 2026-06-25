//! Independent reference numerics used to *verify* the production estimators.
//!
//! Nothing in this module is used by the algorithms themselves; it exists purely
//! so that verification tests have a second, deliberately-different implementation
//! to compare against (e.g. a Jacobi-eigendecomposition matrix exponential to
//! cross-check the Padé(1,1) scaling-and-squaring path, and an exact erf-based
//! standard-normal CDF to calibrate the Fisher-Z critical values).

/// Standard-normal CDF Φ(z) via the complementary error function.
///
/// `erf` uses Abramowitz & Stegun 7.1.26 (a maximum absolute error of
/// 1.5 × 10⁻⁷), which is far tighter than the f32 round-off we compare against.
#[must_use]
pub fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Error function via Abramowitz & Stegun 7.1.26.
#[must_use]
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    // A&S 7.1.26 coefficients.
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    let y = 1.0 - poly * (-x * x).exp();
    sign * y
}

/// Two-sided standard-normal quantile z such that `P(|Z| > z) = alpha`.
///
/// Found by bisection on the monotone tail probability `2(1 − Φ(z))`. Returns the
/// `z` where the two-sided tail mass equals `alpha`.
#[must_use]
pub fn two_sided_z_quantile(alpha: f64) -> f64 {
    let target = alpha; // two-sided tail mass
    let mut lo = 0.0_f64;
    let mut hi = 12.0_f64;
    // tail(z) = 2 * (1 - Phi(z)) is strictly decreasing in z.
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let tail = 2.0 * (1.0 - normal_cdf(mid));
        if tail > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Matrix exponential of a symmetric `n × n` matrix via cyclic-Jacobi
/// eigendecomposition: `A = Q Λ Qᵀ ⇒ exp(A) = Q exp(Λ) Qᵀ`.
///
/// This is an entirely different algorithm from the production Padé(1,1)
/// scaling-and-squaring path, so agreement between the two is a genuine check.
/// Input is taken in `f32` (matching the production data type) but the whole
/// computation runs in `f64` to keep the reference error well below the
/// comparison tolerance.
#[must_use]
pub fn expm_symmetric_eig(a: &[f32], n: usize) -> Vec<f64> {
    // Working copy in f64.
    let mut m = vec![0.0_f64; n * n];
    for (dst, &src) in m.iter_mut().zip(a.iter()) {
        *dst = src as f64;
    }
    // Symmetrise defensively (A may be only numerically symmetric).
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (m[i * n + j] + m[j * n + i]);
            m[i * n + j] = avg;
            m[j * n + i] = avg;
        }
    }
    let mut q = vec![0.0_f64; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }
    // Cyclic Jacobi sweeps.
    for _sweep in 0..100 {
        let mut off = 0.0_f64;
        for p in 0..n {
            for qi in (p + 1)..n {
                off += m[p * n + qi] * m[p * n + qi];
            }
        }
        if off < 1e-24 {
            break;
        }
        for p in 0..n {
            for qi in (p + 1)..n {
                let apq = m[p * n + qi];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[qi * n + qi];
                let phi = 0.5 * (aqq - app) / apq;
                let t = phi.signum() / (phi.abs() + (phi * phi + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Rotate rows/cols p and qi of M.
                for k in 0..n {
                    let mkp = m[k * n + p];
                    let mkq = m[k * n + qi];
                    m[k * n + p] = c * mkp - s * mkq;
                    m[k * n + qi] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[p * n + k];
                    let mqk = m[qi * n + k];
                    m[p * n + k] = c * mpk - s * mqk;
                    m[qi * n + k] = s * mpk + c * mqk;
                }
                // Accumulate eigenvectors.
                for k in 0..n {
                    let qkp = q[k * n + p];
                    let qkq = q[k * n + qi];
                    q[k * n + p] = c * qkp - s * qkq;
                    q[k * n + qi] = s * qkp + c * qkq;
                }
            }
        }
    }
    // exp(A) = Q diag(exp(lambda)) Q^T.
    let exp_lambda: Vec<f64> = (0..n).map(|i| m[i * n + i].exp()).collect();
    let mut result = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for k in 0..n {
                acc += q[i * n + k] * exp_lambda[k] * q[j * n + k];
            }
            result[i * n + j] = acc;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_known_values() {
        assert!((erf(0.0)).abs() < 1e-7);
        // erf(1) = 0.8427007929...
        assert!((erf(1.0) - 0.842_700_792_9).abs() < 1e-6);
        assert!((erf(-1.0) + 0.842_700_792_9).abs() < 1e-6);
        // erf(2) = 0.995322265...
        assert!((erf(2.0) - 0.995_322_265).abs() < 1e-6);
    }

    #[test]
    fn normal_cdf_known_values() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-7);
        // Phi(1.96) ~ 0.975.
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-3);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 1e-3);
    }

    #[test]
    fn two_sided_quantiles() {
        // Classic textbook two-sided critical values.
        assert!((two_sided_z_quantile(0.05) - 1.959_963_98).abs() < 1e-3);
        assert!((two_sided_z_quantile(0.01) - 2.575_829_3).abs() < 1e-3);
        assert!((two_sided_z_quantile(0.10) - 1.644_853_6).abs() < 1e-3);
    }

    #[test]
    fn eig_expm_identity_and_diagonal() {
        // exp(0) = I.
        let e = expm_symmetric_eig(&[0.0_f32; 9], 3);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((e[i * 3 + j] - want).abs() < 1e-9);
            }
        }
        // exp(diag) = diag(exp).
        let mut a = vec![0.0_f32; 9];
        a[0] = 0.5;
        a[4] = -1.2;
        a[8] = 2.0;
        let e = expm_symmetric_eig(&a, 3);
        assert!((e[0] - 0.5_f64.exp()).abs() < 1e-7);
        assert!((e[4] - (-1.2_f64).exp()).abs() < 1e-7);
        assert!((e[8] - 2.0_f64.exp()).abs() < 1e-6);
    }

    #[test]
    fn eig_expm_symmetric_2x2() {
        // A = [[0,1],[1,0]] has eigenvalues +-1; exp(A) = [[cosh1, sinh1],[sinh1, cosh1]].
        let a = vec![0.0_f32, 1.0, 1.0, 0.0];
        let e = expm_symmetric_eig(&a, 2);
        let ch = 1.0_f64.cosh();
        let sh = 1.0_f64.sinh();
        assert!((e[0] - ch).abs() < 1e-7);
        assert!((e[1] - sh).abs() < 1e-7);
        assert!((e[2] - sh).abs() < 1e-7);
        assert!((e[3] - ch).abs() < 1e-7);
    }
}
