//! Anderson-Rubin weak-instrument-robust confidence sets — Anderson TW &
//! Rubin H (1949), "Estimation of the parameters of a single equation in a
//! complete system of stochastic equations", Annals of Mathematical
//! Statistics 20:46–63. The modern F-form review is Andrews-Marmer-Yu 2019
//! ("On Optimal Inference in the Linear IV Regression Model with Many
//! Instruments").
//!
//! Given the structural equation `y = β·d + u` and instruments `Z`, the
//! Anderson-Rubin test rejects `H_0: β = β_0` when the projection of the
//! residual `e(β_0) = y − β_0·d` onto the instrument space is large relative
//! to its orthogonal complement. Under H_0 the F-form statistic
//!
//! ```text
//!   AR(β) = ((n − q) · ‖P_Z e‖²) / (q · ‖M_Z e‖²)  ~  F(q, n − q)
//! ```
//!
//! does **not** require strong identification, so AR-based confidence sets
//! retain the correct coverage even when instruments are weak. We build the
//! confidence set by grid-evaluating `AR(β)` over `[grid_min, grid_max]` and
//! collecting contiguous β intervals where `AR(β) ≤ F⁻¹(1 − α; q, n − q)`.

use crate::error::{CausalError, CausalResult};

/// Configuration for [`AndersonRubin`].
#[derive(Clone, Debug)]
pub struct AndersonRubinConfig {
    /// Two-sided test level.
    pub alpha: f64,
    /// Lower endpoint of the β grid scanned by `confidence_set`.
    pub grid_min: f64,
    /// Upper endpoint of the β grid scanned by `confidence_set`.
    pub grid_max: f64,
    /// Number of grid points (≥ 2).
    pub grid_size: usize,
}

impl Default for AndersonRubinConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            grid_min: -10.0,
            grid_max: 10.0,
            grid_size: 401,
        }
    }
}

/// Result returned by an Anderson-Rubin test or confidence-set computation.
#[derive(Clone, Debug)]
pub struct AndersonRubinResult {
    /// AR F-statistic at the null β.
    pub ar_statistic: f64,
    /// P-value `P(F(q, n − q) > AR_stat)`.
    pub p_value: f64,
    /// Contiguous (lo, hi) β intervals where AR(β) ≤ critical_value.
    /// Empty when the test rejects everywhere on the grid.
    pub conf_set: Vec<(f64, f64)>,
    /// Critical value `F⁻¹(1 − α; q, n − q)`.
    pub critical_value: f64,
}

/// Stateless namespace for the Anderson-Rubin operations.
pub struct AndersonRubin;

impl AndersonRubin {
    /// Test the single null β = β₀ at level α. The returned
    /// [`AndersonRubinResult`] also carries the AR confidence interval over
    /// the grid configured in `cfg`.
    pub fn test(
        y: &[f64],
        d: &[f64],
        z: &[Vec<f64>],
        beta_null: f64,
        cfg: &AndersonRubinConfig,
    ) -> CausalResult<AndersonRubinResult> {
        validate_inputs(y, d, z, cfg)?;
        let n = y.len();
        let q = z[0].len();
        let z_flat = pack_z(z);
        let ztz_inv = invert_ztz(&z_flat, n, q)?;
        let ar_stat = ar_statistic(y, d, &z_flat, &ztz_inv, n, q, beta_null);
        let p_value = f_sf(ar_stat, q as f64, (n - q) as f64);
        let crit = f_inverse_cdf(1.0 - cfg.alpha, q as f64, (n - q) as f64);
        let conf_set = build_confidence_set(y, d, &z_flat, &ztz_inv, n, q, cfg, crit);
        Ok(AndersonRubinResult {
            ar_statistic: ar_stat,
            p_value,
            conf_set,
            critical_value: crit,
        })
    }

    /// Compute only the confidence set; `ar_statistic` and `p_value` are
    /// reported at the midpoint of the grid as a reference.
    pub fn confidence_set(
        y: &[f64],
        d: &[f64],
        z: &[Vec<f64>],
        cfg: &AndersonRubinConfig,
    ) -> CausalResult<AndersonRubinResult> {
        let midpoint = 0.5 * (cfg.grid_min + cfg.grid_max);
        Self::test(y, d, z, midpoint, cfg)
    }
}

fn validate_inputs(
    y: &[f64],
    d: &[f64],
    z: &[Vec<f64>],
    cfg: &AndersonRubinConfig,
) -> CausalResult<()> {
    if !(cfg.alpha > 0.0 && cfg.alpha < 1.0) {
        return Err(CausalError::Internal {
            msg: format!("alpha must be in (0, 1), got {}", cfg.alpha),
        });
    }
    if cfg.grid_size < 2 {
        return Err(CausalError::Internal {
            msg: format!("grid_size must be ≥ 2, got {}", cfg.grid_size),
        });
    }
    if !cfg.grid_min.is_finite() || !cfg.grid_max.is_finite() {
        return Err(CausalError::Internal {
            msg: "grid_min / grid_max must be finite".to_string(),
        });
    }
    if cfg.grid_max <= cfg.grid_min {
        return Err(CausalError::Internal {
            msg: format!(
                "grid_max ({}) must exceed grid_min ({})",
                cfg.grid_max, cfg.grid_min
            ),
        });
    }
    if y.is_empty() || d.is_empty() || z.is_empty() {
        return Err(CausalError::EmptyInput);
    }
    let n = y.len();
    if d.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n,
            got: d.len(),
        });
    }
    if z.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n,
            got: z.len(),
        });
    }
    let q = z[0].len();
    if q == 0 {
        return Err(CausalError::EmptyInput);
    }
    if n <= q + 1 {
        return Err(CausalError::EmptyInput);
    }
    for (i, row) in z.iter().enumerate() {
        if row.len() != q {
            return Err(CausalError::DimensionMismatch {
                expected: q,
                got: row.len(),
            });
        }
        for &v in row.iter() {
            if !v.is_finite() {
                return Err(CausalError::Internal {
                    msg: format!("z[{i}] contains non-finite value"),
                });
            }
        }
    }
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(CausalError::Internal {
                msg: format!("y[{i}] is not finite"),
            });
        }
    }
    for (i, &v) in d.iter().enumerate() {
        if !v.is_finite() {
            return Err(CausalError::Internal {
                msg: format!("d[{i}] is not finite"),
            });
        }
    }
    Ok(())
}

/// Pack the row-major matrix `z[i][j]` into a flat n×q row-major buffer.
fn pack_z(z: &[Vec<f64>]) -> Vec<f64> {
    let n = z.len();
    let q = z[0].len();
    let mut out = vec![0.0_f64; n * q];
    for (i, row) in z.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[i * q + j] = v;
        }
    }
    out
}

/// Compute (Z'Z + λI)⁻¹ via inline Gauss-Jordan with a tiny ridge for
/// numerical stability when Z is near-collinear.
fn invert_ztz(z_flat: &[f64], n: usize, q: usize) -> CausalResult<Vec<f64>> {
    let mut ztz = vec![0.0_f64; q * q];
    for row in 0..n {
        for i in 0..q {
            for j in 0..q {
                ztz[i * q + j] += z_flat[row * q + i] * z_flat[row * q + j];
            }
        }
    }
    for i in 0..q {
        ztz[i * q + i] += 1e-10;
    }
    gauss_jordan_inv(&ztz, q).ok_or(CausalError::MatrixSingular)
}

fn gauss_jordan_inv(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut m = vec![0.0_f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            m[i * 2 * n + j] = a[i * n + j];
        }
        m[i * 2 * n + n + i] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if m[row * 2 * n + col].abs() > m[pivot * 2 * n + col].abs() {
                pivot = row;
            }
        }
        if m[pivot * 2 * n + col].abs() < 1e-14 {
            return None;
        }
        if pivot != col {
            for k in 0..(2 * n) {
                m.swap(col * 2 * n + k, pivot * 2 * n + k);
            }
        }
        let div = m[col * 2 * n + col];
        for k in 0..(2 * n) {
            m[col * 2 * n + k] /= div;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = m[row * 2 * n + col];
            if factor.abs() < 1e-18 {
                continue;
            }
            for k in 0..(2 * n) {
                let v = m[col * 2 * n + k] * factor;
                m[row * 2 * n + k] -= v;
            }
        }
    }
    let mut inv = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = m[i * 2 * n + n + j];
        }
    }
    Some(inv)
}

/// Compute the AR F-form statistic at one β value, reusing the cached
/// (Z'Z)⁻¹ inverse.
fn ar_statistic(
    y: &[f64],
    d: &[f64],
    z_flat: &[f64],
    ztz_inv: &[f64],
    n: usize,
    q: usize,
    beta: f64,
) -> f64 {
    // e = y - β d
    let mut e = vec![0.0_f64; n];
    let mut sse_total = 0.0_f64;
    for i in 0..n {
        let ei = y[i] - beta * d[i];
        e[i] = ei;
        sse_total += ei * ei;
    }
    // Z'e (length q)
    let mut zte = vec![0.0_f64; q];
    for row in 0..n {
        for j in 0..q {
            zte[j] += z_flat[row * q + j] * e[row];
        }
    }
    // (Z'Z)⁻¹ Z'e
    let mut tmp = vec![0.0_f64; q];
    for i in 0..q {
        let mut acc = 0.0_f64;
        for j in 0..q {
            acc += ztz_inv[i * q + j] * zte[j];
        }
        tmp[i] = acc;
    }
    // ||P_Z e||² = (Z'e)' (Z'Z)⁻¹ (Z'e)
    let mut p_norm_sq = 0.0_f64;
    for j in 0..q {
        p_norm_sq += zte[j] * tmp[j];
    }
    // Numerical guard: projection can drift slightly negative for near-zero
    // residual via accumulated rounding; clamp to 0.
    if p_norm_sq < 0.0 {
        p_norm_sq = 0.0;
    }
    let m_norm_sq = (sse_total - p_norm_sq).max(1e-300);
    let num = (n as f64 - q as f64) * p_norm_sq;
    let den = q as f64 * m_norm_sq;
    if den < 1e-300 {
        return f64::INFINITY;
    }
    num / den
}

/// Scan `cfg.grid_size` β values uniformly in `[cfg.grid_min, cfg.grid_max]`
/// and stitch contiguous intervals where AR(β) ≤ `crit`.
fn build_confidence_set(
    y: &[f64],
    d: &[f64],
    z_flat: &[f64],
    ztz_inv: &[f64],
    n: usize,
    q: usize,
    cfg: &AndersonRubinConfig,
    crit: f64,
) -> Vec<(f64, f64)> {
    let m = cfg.grid_size;
    let step = (cfg.grid_max - cfg.grid_min) / (m - 1) as f64;
    let mut intervals: Vec<(f64, f64)> = Vec::new();
    let mut current_lo: Option<f64> = None;
    let mut last_inside = f64::NAN;
    for k in 0..m {
        let beta = cfg.grid_min + step * k as f64;
        let stat = ar_statistic(y, d, z_flat, ztz_inv, n, q, beta);
        let inside = stat <= crit;
        if inside {
            if current_lo.is_none() {
                current_lo = Some(beta);
            }
            last_inside = beta;
        } else if let Some(lo) = current_lo.take() {
            intervals.push((lo, last_inside));
        }
    }
    if let Some(lo) = current_lo {
        intervals.push((lo, last_inside));
    }
    intervals
}

/// Regularized incomplete beta function `I_x(a, b)` (Numerical Recipes
/// §6.4, continued-fraction expansion).
fn incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_cf(x, a, b) / a
    } else {
        1.0 - bt * beta_cf(1.0 - x, b, a) / b
    }
}

/// Continued-fraction kernel for the incomplete beta (Lentz's method).
fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    let max_iters = 200_usize;
    let eps = 3e-12_f64;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0_f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-300 {
        d = 1e-300;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=max_iters {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-300 {
            d = 1e-300;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-300 {
            c = 1e-300;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-300 {
            d = 1e-300;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-300 {
            c = 1e-300;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < eps {
            break;
        }
    }
    h
}

/// Lanczos approximation to log Γ(x), good to ~1e-14 for x > 0.
fn ln_gamma(x: f64) -> f64 {
    let coef = [
        76.180_091_729_471_46_f64,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.001_208_650_973_866_179,
        -0.000_005_395_239_384_953,
    ];
    let mut y = x;
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut ser = 1.000_000_000_190_015_f64;
    for c in &coef {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.506_628_274_631_000_5_f64 * ser / x).ln()
}

/// Survival function `P(F(d1, d2) > f)` via `I_{d2/(d2+d1·f)}(d2/2, d1/2)`.
fn f_sf(f: f64, d1: f64, d2: f64) -> f64 {
    if f <= 0.0 {
        return 1.0;
    }
    if !f.is_finite() {
        return 0.0;
    }
    let x = d2 / (d2 + d1 * f);
    incomplete_beta(x, d2 / 2.0, d1 / 2.0)
}

/// CDF `P(F(d1, d2) ≤ f) = 1 − sf`.
fn f_cdf(f: f64, d1: f64, d2: f64) -> f64 {
    1.0 - f_sf(f, d1, d2)
}

#[cfg(test)]
pub(super) fn f_cdf_pub(f: f64, d1: f64, d2: f64) -> f64 {
    f_cdf(f, d1, d2)
}

#[cfg(test)]
pub(super) fn f_inverse_cdf_pub(p: f64, d1: f64, d2: f64) -> f64 {
    f_inverse_cdf(p, d1, d2)
}

/// Bisection inverse: smallest `f` with `f_cdf(f) ≥ p`. We use a generous
/// bracket and 80 iterations for ~1e-9 precision on common df.
fn f_inverse_cdf(p: f64, d1: f64, d2: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    while f_cdf(hi, d1, d2) < p {
        hi *= 2.0;
        if hi > 1e18 {
            return hi;
        }
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        let val = f_cdf(mid, d1, d2);
        if val < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}
