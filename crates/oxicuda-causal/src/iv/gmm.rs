//! Generalized Method of Moments (GMM) — Hansen (1982).
//!
//! Hansen LP. *Large sample properties of generalized method of moments
//! estimators.* Econometrica 50(4): 1029–1054 (1982). Textbook treatment:
//! Hayashi (2000), *Econometrics*, §3.5–3.7.
//!
//! GMM estimates `θ ∈ ℝ^p` by minimizing `Q(θ; W) = ḡ(θ)ᵀ W ḡ(θ)` where
//! `g(z_i, θ) ∈ ℝ^q` are moment conditions with `E[g(z_i, θ₀)] = 0`.
//! For the linear IV moment `g_i = z_i · (y_i − x_iᵀ θ)` the criterion has
//! the closed form `θ̂(W) = (XᵀZ W ZᵀX)⁻¹ XᵀZ W Zᵀy`.
//!
//! Two-step efficient GMM:
//! 1. Stage 1: `W₁ = (ZᵀZ + λI)⁻¹` → 2SLS in the just-identified case.
//! 2. Moments: `g_i = z_i · (y_i − x_iᵀ θ̂₁)`.
//! 3. Optimal heteroskedasticity-robust weight
//!    `Ŵ = ((1/n)·Σᵢ g_i g_iᵀ + λI)⁻¹`.
//! 4. Stage 2: re-solve; continuously-updating GMM iterates until
//!    `|Δθ|_∞ < tol` or `max_iters` reached.
//! 5. `Var(θ̂) = (1/n)·(XᵀZ Ŵ ZᵀX + λI)⁻¹`.
//! 6. Hansen J: `J = n · ḡ(θ̂)ᵀ Ŵ ḡ(θ̂) ~ χ²(q − p)`; just-identified
//!    models return `J = 0`, p-value = 1.

use crate::error::{CausalError, CausalResult};

/// Configuration for [`Gmm::estimate`].
#[derive(Clone, Debug)]
pub struct GmmConfig {
    /// Ridge added to every inverted matrix. Must be strictly positive.
    pub ridge_lambda: f64,
    /// If `true`, perform efficient two-step GMM (optionally iterating);
    /// if `false`, return the inefficient stage-1 estimate.
    pub two_step: bool,
    /// Convergence tolerance on `|Δθ|_∞` for iterative refinement.
    pub tol: f64,
    /// Cap on GMM iterations (≥ 1).
    pub max_iters: usize,
}

impl Default for GmmConfig {
    fn default() -> Self {
        Self {
            ridge_lambda: 1e-6,
            two_step: true,
            tol: 1e-8,
            max_iters: 50,
        }
    }
}

/// Result of [`Gmm::estimate`].
#[derive(Clone, Debug)]
pub struct GmmResult {
    /// Estimated parameter vector (length `p`).
    pub theta: Vec<f64>,
    /// Standard errors `sqrt(diag(Var(θ̂)))` (length `p`).
    pub se: Vec<f64>,
    /// Hansen overidentification statistic `J = n · ḡᵀ Ŵ ḡ`.
    pub j_stat: f64,
    /// `1 − F_{χ²(q−p)}(J)`. Equals `1.0` when just-identified.
    pub j_pvalue: f64,
    /// Sample size.
    pub n: usize,
    /// Number of moments `q`.
    pub n_moments: usize,
    /// Number of weight-matrix updates performed.
    pub n_iters: usize,
}

/// Stateless namespace for the GMM estimator.
pub struct Gmm;

impl Gmm {
    /// Estimate `θ` by (optionally two-step) GMM with optimal
    /// heteroskedasticity-robust weighting.
    pub fn estimate(
        y: &[f64],
        x: &[Vec<f64>],
        z: &[Vec<f64>],
        cfg: &GmmConfig,
    ) -> CausalResult<GmmResult> {
        validate(y, x, z, cfg)?;
        let work = build_workspace(y, x, z, cfg)?;
        let theta_initial = solve_for(&work, &work.weight_stage1, cfg.ridge_lambda)?;
        let (theta_final, weight_final, n_iters) = if cfg.two_step {
            run_two_step(&work, &theta_initial, cfg)?
        } else {
            (theta_initial.clone(), work.weight_stage1.clone(), 0_usize)
        };
        let se = compute_se(
            &work.xtz,
            &weight_final,
            work.n,
            work.p,
            work.q,
            cfg.ridge_lambda,
        )?;
        let (j_stat, j_pvalue) = hansen_j(&work, &theta_final, &weight_final);
        Ok(GmmResult {
            theta: theta_final,
            se,
            j_stat,
            j_pvalue,
            n: work.n,
            n_moments: work.q,
            n_iters,
        })
    }
}

// Shared workspace ---------------------------------------------------------

struct GmmWorkspace<'a> {
    y: &'a [f64],
    x_flat: Vec<f64>,
    z_flat: Vec<f64>,
    xtz: Vec<f64>,
    zty: Vec<f64>,
    weight_stage1: Vec<f64>,
    n: usize,
    p: usize,
    q: usize,
}

fn build_workspace<'a>(
    y: &'a [f64],
    x: &[Vec<f64>],
    z: &[Vec<f64>],
    cfg: &GmmConfig,
) -> CausalResult<GmmWorkspace<'a>> {
    let n = y.len();
    let p = x[0].len();
    let q = z[0].len();
    let x_flat = pack_rows(x, n, p);
    let z_flat = pack_rows(z, n, q);
    let xtz = xt_mul_y(&x_flat, &z_flat, n, p, q);
    let ztz = xt_mul_y(&z_flat, &z_flat, n, q, q);
    let zty = xt_mul_vec(&z_flat, y, n, q);
    let mut ztz_ridge = ztz;
    for k in 0..q {
        ztz_ridge[k * q + k] += cfg.ridge_lambda;
    }
    let weight_stage1 = invert_with_ridge(&ztz_ridge, q, cfg.ridge_lambda)?;
    Ok(GmmWorkspace {
        y,
        x_flat,
        z_flat,
        xtz,
        zty,
        weight_stage1,
        n,
        p,
        q,
    })
}

// Validation ---------------------------------------------------------------

fn validate(y: &[f64], x: &[Vec<f64>], z: &[Vec<f64>], cfg: &GmmConfig) -> CausalResult<()> {
    if !(cfg.ridge_lambda.is_finite() && cfg.ridge_lambda > 0.0) {
        return Err(CausalError::IncompatibleData);
    }
    if !(cfg.tol.is_finite() && cfg.tol > 0.0) {
        return Err(CausalError::IncompatibleData);
    }
    if cfg.max_iters == 0 {
        return Err(CausalError::IncompatibleData);
    }
    if y.is_empty() || x.is_empty() || z.is_empty() {
        return Err(CausalError::EmptyInput);
    }
    let n = y.len();
    if x.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n,
            got: x.len(),
        });
    }
    if z.len() != n {
        return Err(CausalError::DimensionMismatch {
            expected: n,
            got: z.len(),
        });
    }
    let p = x[0].len();
    let q = z[0].len();
    if p == 0 || q == 0 {
        return Err(CausalError::EmptyInput);
    }
    if q < p {
        return Err(CausalError::IncompatibleData);
    }
    if n <= p {
        return Err(CausalError::EmptyInput);
    }
    check_rows(x, p)?;
    check_rows(z, q)?;
    for &v in y.iter() {
        if !v.is_finite() {
            return Err(CausalError::IncompatibleData);
        }
    }
    Ok(())
}

fn check_rows(rows: &[Vec<f64>], width: usize) -> CausalResult<()> {
    for row in rows.iter() {
        if row.len() != width {
            return Err(CausalError::DimensionMismatch {
                expected: width,
                got: row.len(),
            });
        }
        for &v in row.iter() {
            if !v.is_finite() {
                return Err(CausalError::IncompatibleData);
            }
        }
    }
    Ok(())
}

// Matrix helpers -----------------------------------------------------------

fn pack_rows(rows: &[Vec<f64>], n: usize, p: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n * p];
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[i * p + j] = v;
        }
    }
    out
}

/// Aᵀ·B: A row-major `n × p`, B row-major `n × q`, output `p × q`.
fn xt_mul_y(a: &[f64], b: &[f64], n: usize, p: usize, q: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; p * q];
    for row in 0..n {
        for i in 0..p {
            let av = a[row * p + i];
            for j in 0..q {
                out[i * q + j] += av * b[row * q + j];
            }
        }
    }
    out
}

fn xt_mul_vec(a: &[f64], v: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; p];
    for row in 0..n {
        for i in 0..p {
            out[i] += a[row * p + i] * v[row];
        }
    }
    out
}

fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, p: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; m * p];
    for i in 0..m {
        for l in 0..k {
            let av = a[i * k + l];
            for j in 0..p {
                out[i * p + j] += av * b[l * p + j];
            }
        }
    }
    out
}

fn mat_vec(a: &[f64], v: &[f64], m: usize, p: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; m];
    for i in 0..m {
        let mut s = 0.0_f64;
        for j in 0..p {
            s += a[i * p + j] * v[j];
        }
        out[i] = s;
    }
    out
}

fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut at = vec![0.0_f64; cols * rows];
    for i in 0..rows {
        for j in 0..cols {
            at[j * rows + i] = a[i * cols + j];
        }
    }
    at
}

/// Invert with ridge fallback (escalates by ×10 up to three times).
fn invert_with_ridge(a: &[f64], n: usize, ridge: f64) -> CausalResult<Vec<f64>> {
    if let Some(inv) = gauss_jordan_invert(a, n) {
        return Ok(inv);
    }
    let mut current = a.to_vec();
    let mut bump = ridge.max(1e-10);
    for _ in 0..3 {
        for k in 0..n {
            current[k * n + k] += bump;
        }
        if let Some(inv) = gauss_jordan_invert(&current, n) {
            return Ok(inv);
        }
        bump *= 10.0;
    }
    Err(CausalError::MatrixSingular)
}

fn gauss_jordan_invert(a: &[f64], n: usize) -> Option<Vec<f64>> {
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

// GMM closed-form & iterative refinement -----------------------------------

fn solve_closed_form(
    xtz: &[f64],
    weight: &[f64],
    zty: &[f64],
    p: usize,
    q: usize,
    ridge: f64,
) -> CausalResult<Vec<f64>> {
    let xtz_w = matmul(xtz, weight, p, q, q);
    let ztx = transpose(xtz, p, q);
    let mut a = matmul(&xtz_w, &ztx, p, q, p);
    for k in 0..p {
        a[k * p + k] += ridge;
    }
    let rhs = mat_vec(&xtz_w, zty, p, q);
    let a_inv = invert_with_ridge(&a, p, ridge)?;
    Ok(mat_vec(&a_inv, &rhs, p, p))
}

fn compute_optimal_weight(
    work: &GmmWorkspace<'_>,
    theta: &[f64],
    ridge: f64,
) -> CausalResult<Vec<f64>> {
    let mut omega = vec![0.0_f64; work.q * work.q];
    let n_f = work.n as f64;
    for row in 0..work.n {
        let mut pred = 0.0_f64;
        for (k, &tk) in theta.iter().enumerate().take(work.p) {
            pred += work.x_flat[row * work.p + k] * tk;
        }
        let e = work.y[row] - pred;
        for i in 0..work.q {
            let zi = work.z_flat[row * work.q + i];
            for j in 0..work.q {
                omega[i * work.q + j] += zi * work.z_flat[row * work.q + j] * e * e;
            }
        }
    }
    for v in omega.iter_mut() {
        *v /= n_f;
    }
    for k in 0..work.q {
        omega[k * work.q + k] += ridge;
    }
    invert_with_ridge(&omega, work.q, ridge)
}

fn solve_for(work: &GmmWorkspace<'_>, weight: &[f64], ridge: f64) -> CausalResult<Vec<f64>> {
    solve_closed_form(&work.xtz, weight, &work.zty, work.p, work.q, ridge)
}

fn run_two_step(
    work: &GmmWorkspace<'_>,
    theta_initial: &[f64],
    cfg: &GmmConfig,
) -> CausalResult<(Vec<f64>, Vec<f64>, usize)> {
    let mut theta_prev = theta_initial.to_vec();
    let mut weight = compute_optimal_weight(work, &theta_prev, cfg.ridge_lambda)?;
    let mut theta_curr = solve_for(work, &weight, cfg.ridge_lambda)?;
    let mut iters_done = 1_usize;
    while iters_done < cfg.max_iters {
        let mut max_change = 0.0_f64;
        for k in 0..work.p {
            let diff = (theta_curr[k] - theta_prev[k]).abs();
            if diff > max_change {
                max_change = diff;
            }
        }
        if max_change < cfg.tol {
            break;
        }
        theta_prev = theta_curr.clone();
        weight = compute_optimal_weight(work, &theta_prev, cfg.ridge_lambda)?;
        theta_curr = solve_for(work, &weight, cfg.ridge_lambda)?;
        iters_done += 1;
    }
    Ok((theta_curr, weight, iters_done))
}

// Variance & Hansen J ------------------------------------------------------

fn compute_se(
    xtz: &[f64],
    weight: &[f64],
    n: usize,
    p: usize,
    q: usize,
    ridge: f64,
) -> CausalResult<Vec<f64>> {
    // Sample-moment forms Σ_XZ = X'Z / n and Σ_ZX = Z'X / n. The
    // asymptotic variance is (1/n) · (Σ_XZ · Ŵ · Σ_ZX)⁻¹.
    let n_f = n as f64;
    let mut sxz = xtz.to_vec();
    for v in sxz.iter_mut() {
        *v /= n_f;
    }
    let sxz_w = matmul(&sxz, weight, p, q, q);
    let szx = transpose(&sxz, p, q);
    let mut bread = matmul(&sxz_w, &szx, p, q, p);
    for k in 0..p {
        bread[k * p + k] += ridge;
    }
    let bread_inv = invert_with_ridge(&bread, p, ridge)?;
    let mut se = vec![0.0_f64; p];
    for k in 0..p {
        let v = (bread_inv[k * p + k] / n_f).max(0.0);
        se[k] = v.sqrt();
    }
    Ok(se)
}

fn hansen_j(work: &GmmWorkspace<'_>, theta: &[f64], weight: &[f64]) -> (f64, f64) {
    if work.q == work.p {
        return (0.0, 1.0);
    }
    let mut g_bar = vec![0.0_f64; work.q];
    let n_f = work.n as f64;
    for row in 0..work.n {
        let mut pred = 0.0_f64;
        for (k, &tk) in theta.iter().enumerate().take(work.p) {
            pred += work.x_flat[row * work.p + k] * tk;
        }
        let e = work.y[row] - pred;
        for (j, gv) in g_bar.iter_mut().enumerate().take(work.q) {
            *gv += work.z_flat[row * work.q + j] * e;
        }
    }
    for v in g_bar.iter_mut() {
        *v /= n_f;
    }
    let wg = mat_vec(weight, &g_bar, work.q, work.q);
    let mut j_val = 0.0_f64;
    for j in 0..work.q {
        j_val += g_bar[j] * wg[j];
    }
    j_val *= n_f;
    if !j_val.is_finite() || j_val < 0.0 {
        j_val = 0.0;
    }
    let df = (work.q - work.p) as f64;
    let p_val = chi2_sf(j_val, df);
    (j_val, p_val)
}

// chi² survival function via incomplete gamma (Numerical Recipes §6.2:
// series `gser` for x < a+1, continued fraction `gcf` otherwise; Lanczos
// ln Γ).
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

fn gser(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..200_usize {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 3e-12_f64 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

fn gcf(a: f64, x: f64) -> f64 {
    let fpmin = 1e-300_f64;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=200_usize {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 3e-12_f64 {
            break;
        }
    }
    h * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Survival function `P(χ²(df) > x)`.
pub(crate) fn chi2_sf(x: f64, df: f64) -> f64 {
    if df <= 0.0 || x <= 0.0 {
        return 1.0;
    }
    if !x.is_finite() {
        return 0.0;
    }
    let a = df / 2.0;
    let scaled = x / 2.0;
    if scaled < a + 1.0 {
        (1.0 - gser(a, scaled)).clamp(0.0, 1.0)
    } else {
        gcf(a, scaled).clamp(0.0, 1.0)
    }
}
