//! Structural Time Series (STS) decomposition via Kalman filtering and smoothing.
//!
//! Harvey (1989/1990) Basic Structural Model (BSM). The observed series is
//! modelled as the sum of an unobserved **local-linear trend**, a stochastic
//! **seasonal** component, and an **irregular** (observation noise) term:
//!
//! ```text
//! y_t   = μ_t + γ_t + ε_t,                 ε_t ~ N(0, σ²_ε)   (observation)
//! μ_t   = μ_{t-1} + β_{t-1} + η_t,         η_t ~ N(0, σ²_η)   (level)
//! β_t   = β_{t-1} + ζ_t,                   ζ_t ~ N(0, σ²_ζ)   (slope)
//! γ_t   = −Σ_{j=1}^{s-1} γ_{t-j} + ω_t,    ω_t ~ N(0, σ²_ω)   (seasonal)
//! ```
//!
//! The state vector is `α_t = [μ_t, β_t, γ_t, γ_{t-1}, …, γ_{t-(s-2)}]ᵀ`, of
//! dimension `m = 2 + (s-1)`. The decomposition is obtained by running a Kalman
//! filter forward and the Durbin–Koopman disturbance smoother backward. The
//! observation/state variances may be supplied directly or estimated by a few
//! EM iterations.
//!
//! All public input/output is `f32`; the recursions run in `f64` internally for
//! numerical stability and cast back at the boundary.
//!
//! References:
//! - Harvey, A. C. (1990). *Forecasting, Structural Time Series Models and the
//!   Kalman Filter.* Cambridge University Press.
//! - Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space
//!   Methods.* 2nd ed., Oxford University Press, §4.4–4.5.

use crate::error::{TsError, TsResult};

/// Lower floor applied to every estimated variance to avoid collapse to zero.
const VAR_FLOOR: f64 = 1e-10;

// ── Config ───────────────────────────────────────────────────────────────────

/// Hyperparameters for [`StsDecomposer`].
///
/// The four variances control the signal-to-noise trade-off. Smaller
/// `obs_var` (relative to the state variances) makes the trend/seasonal track
/// the data more tightly; a smaller `level_var`/`slope_var` makes the trend
/// smoother; a larger `seasonal_var` lets the seasonal pattern adapt.
///
/// When `em_iters > 0` these values are used only as the EM starting point and
/// are then re-estimated from the data.
#[derive(Debug, Clone, PartialEq)]
pub struct StsConfig {
    /// Observation-noise variance σ²_ε (the irregular component).
    pub obs_var: f32,
    /// Level-disturbance variance σ²_η.
    pub level_var: f32,
    /// Slope-disturbance variance σ²_ζ.
    pub slope_var: f32,
    /// Seasonal-disturbance variance σ²_ω.
    pub seasonal_var: f32,
    /// Number of EM iterations used to estimate the variances (0 = none).
    pub em_iters: usize,
}

impl Default for StsConfig {
    fn default() -> Self {
        Self {
            obs_var: 1.0,
            level_var: 0.1,
            slope_var: 0.01,
            seasonal_var: 0.1,
            em_iters: 0,
        }
    }
}

impl StsConfig {
    /// Construct the default configuration (alias for [`Default::default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set the number of EM iterations.
    #[must_use]
    pub fn with_em_iters(mut self, iters: usize) -> Self {
        self.em_iters = iters;
        self
    }

    /// Builder: set all four variances at once.
    #[must_use]
    pub fn with_variances(mut self, obs: f32, level: f32, slope: f32, seasonal: f32) -> Self {
        self.obs_var = obs;
        self.level_var = level;
        self.slope_var = slope;
        self.seasonal_var = seasonal;
        self
    }
}

// ── Decomposer ───────────────────────────────────────────────────────────────

/// Structural time-series decomposer (local-linear-trend + seasonal + irregular).
#[derive(Debug, Clone)]
pub struct StsDecomposer {
    season_length: usize,
    /// State dimension `m = 2 + sea_dim`.
    m: usize,
    /// Number of seasonal states `s-1` (0 when `season_length <= 1`).
    sea_dim: usize,
    /// Transition matrix `T` (`m × m`, row-major).
    transition: Vec<f64>,
    // Variances actually used by the last fit (post-EM if EM was run).
    obs_var: f64,
    level_var: f64,
    slope_var: f64,
    seasonal_var: f64,
    em_iters: usize,
    // Fit outputs.
    trend: Vec<f32>,
    seasonal: Vec<f32>,
    irregular: Vec<f32>,
    final_state: Vec<f64>,
    log_likelihood: f64,
    fitted: bool,
}

/// Output bundle produced by the filter + smoother sweep.
struct SmoothOut {
    trend: Vec<f32>,
    seasonal: Vec<f32>,
    irregular: Vec<f32>,
    final_state: Vec<f64>,
    log_likelihood: f64,
    sum_eps2: f64,
    sum_eta_level: f64,
    sum_eta_slope: f64,
    sum_eta_seasonal: f64,
}

impl StsDecomposer {
    /// Create a new decomposer for the given seasonal period.
    ///
    /// `season_length == 0` is rejected. `season_length == 1` yields a pure
    /// local-linear-trend model with no seasonal component; `season_length >= 2`
    /// adds the standard `s-1` seasonal states.
    ///
    /// # Errors
    ///
    /// Returns [`TsError::InvalidSequenceLength`] when `season_length == 0`.
    pub fn new(season_length: usize, config: StsConfig) -> TsResult<Self> {
        if season_length == 0 {
            return Err(TsError::InvalidSequenceLength(0));
        }
        let sea_dim = if season_length >= 2 {
            season_length - 1
        } else {
            0
        };
        let m = 2 + sea_dim;
        let transition = build_transition(m, sea_dim);
        Ok(Self {
            season_length,
            m,
            sea_dim,
            transition,
            obs_var: f64::from(config.obs_var).max(VAR_FLOOR),
            level_var: f64::from(config.level_var).max(VAR_FLOOR),
            slope_var: f64::from(config.slope_var).max(VAR_FLOOR),
            seasonal_var: f64::from(config.seasonal_var).max(VAR_FLOOR),
            em_iters: config.em_iters,
            trend: Vec::new(),
            seasonal: Vec::new(),
            irregular: Vec::new(),
            final_state: Vec::new(),
            log_likelihood: f64::NEG_INFINITY,
            fitted: false,
        })
    }

    /// Seasonal period this decomposer was created with.
    #[must_use]
    pub fn season_length(&self) -> usize {
        self.season_length
    }

    /// State dimension `m`.
    #[must_use]
    pub fn state_dim(&self) -> usize {
        self.m
    }

    /// Run the Kalman filter + smoother (with optional EM) on `y`.
    ///
    /// After a successful fit, [`trend`](Self::trend), [`seasonal`](Self::seasonal),
    /// [`irregular`](Self::irregular) and [`forecast`](Self::forecast) become
    /// available.
    ///
    /// # Errors
    ///
    /// Returns [`TsError::EmptyInput`] when `y` is empty and
    /// [`TsError::NonFinite`] when `y` contains a non-finite value.
    pub fn fit(&mut self, y: &[f32]) -> TsResult<()> {
        if y.is_empty() {
            return Err(TsError::EmptyInput {
                msg: "STS fit requires a non-empty series".to_string(),
            });
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(TsError::NonFinite);
        }
        let yf: Vec<f64> = y.iter().map(|&v| f64::from(v)).collect();
        let n = yf.len() as f64;

        let mut obs = self.obs_var;
        let mut lev = self.level_var;
        let mut slo = self.slope_var;
        let mut sea = self.seasonal_var;

        // EM: re-estimate the variances from the smoothed disturbances.
        for _ in 0..self.em_iters {
            let stats = self.filter_smooth(&yf, obs, lev, slo, sea);
            obs = (stats.sum_eps2 / n).max(VAR_FLOOR);
            lev = (stats.sum_eta_level / n).max(VAR_FLOOR);
            slo = (stats.sum_eta_slope / n).max(VAR_FLOOR);
            if self.sea_dim > 0 {
                sea = (stats.sum_eta_seasonal / n).max(VAR_FLOOR);
            }
        }

        // Final sweep producing the reported components.
        let out = self.filter_smooth(&yf, obs, lev, slo, sea);
        self.obs_var = obs;
        self.level_var = lev;
        self.slope_var = slo;
        self.seasonal_var = sea;
        self.trend = out.trend;
        self.seasonal = out.seasonal;
        self.irregular = out.irregular;
        self.final_state = out.final_state;
        self.log_likelihood = out.log_likelihood;
        self.fitted = true;
        Ok(())
    }

    /// Smoothed trend (level) component, length `T`. Empty before [`fit`](Self::fit).
    #[must_use]
    pub fn trend(&self) -> &[f32] {
        &self.trend
    }

    /// Smoothed seasonal component, length `T`. Empty before [`fit`](Self::fit).
    #[must_use]
    pub fn seasonal(&self) -> &[f32] {
        &self.seasonal
    }

    /// Smoothed irregular (residual) component, length `T`.
    #[must_use]
    pub fn irregular(&self) -> &[f32] {
        &self.irregular
    }

    /// Gaussian log-likelihood of the fitted model (`f32`).
    #[must_use]
    pub fn log_likelihood(&self) -> f32 {
        self.log_likelihood as f32
    }

    /// Variances used by the last fit, as `(obs, level, slope, seasonal)`.
    #[must_use]
    pub fn variances(&self) -> (f32, f32, f32, f32) {
        (
            self.obs_var as f32,
            self.level_var as f32,
            self.slope_var as f32,
            self.seasonal_var as f32,
        )
    }

    /// Forecast `h` steps ahead by projecting the smoothed final state forward
    /// through the transition equation.
    ///
    /// # Errors
    ///
    /// Returns [`TsError::EmptyInput`] when called before [`fit`](Self::fit).
    pub fn forecast(&self, h: usize) -> TsResult<Vec<f32>> {
        if !self.fitted {
            return Err(TsError::EmptyInput {
                msg: "STS forecast requires fit() first".to_string(),
            });
        }
        if h == 0 {
            return Ok(Vec::new());
        }
        let m = self.m;
        let mut state = self.final_state.clone();
        let mut out = Vec::with_capacity(h);
        for _ in 0..h {
            state = mat_vec(&self.transition, &state, m);
            let z = state[0] + if self.sea_dim > 0 { state[2] } else { 0.0 };
            out.push(z as f32);
        }
        Ok(out)
    }

    // ── Core recursion ────────────────────────────────────────────────────────

    /// One forward Kalman-filter pass plus one backward disturbance-smoother
    /// pass with the supplied variances. Produces the smoothed components, the
    /// log-likelihood, and the EM sufficient statistics in a single sweep.
    fn filter_smooth(&self, y: &[f64], obs: f64, lev: f64, slo: f64, sea: f64) -> SmoothOut {
        let n = y.len();
        let m = self.m;
        let sea_dim = self.sea_dim;
        let sea_idx = 2usize;
        let t = &self.transition;
        let h = obs.max(VAR_FLOOR);

        // Diffuse-ish initialisation: level at the first observation, large prior
        // covariance scaled by the data spread.
        let kappa = 1e6 * (1.0 + sample_variance(y));
        let mut a = vec![0.0_f64; m];
        a[0] = y[0];
        let mut p = vec![0.0_f64; m * m];
        for i in 0..m {
            p[i * m + i] = kappa;
        }

        // Forward storage.
        let mut a_store = vec![0.0_f64; n * m];
        let mut p_store = vec![0.0_f64; n * m * m];
        let mut v_store = vec![0.0_f64; n];
        let mut f_store = vec![0.0_f64; n];
        let mut k_store = vec![0.0_f64; n * m];
        let mut log_lik = 0.0_f64;
        let two_pi_ln = std::f64::consts::TAU.ln();

        for ti in 0..n {
            a_store[ti * m..ti * m + m].copy_from_slice(&a);
            p_store[ti * m * m..(ti + 1) * m * m].copy_from_slice(&p);

            // Innovation v = y - Z a.
            let za = a[0] + if sea_dim > 0 { a[sea_idx] } else { 0.0 };
            let v = y[ti] - za;

            // P Zᵀ = column 0 (+ column sea_idx).
            let pz: Vec<f64> = (0..m)
                .map(|i| p[i * m] + if sea_dim > 0 { p[i * m + sea_idx] } else { 0.0 })
                .collect();
            let f = (pz[0] + if sea_dim > 0 { pz[sea_idx] } else { 0.0 } + h).max(1e-12);

            // Gain K = T (P Zᵀ) / F.
            let tpz = mat_vec(t, &pz, m);
            let k: Vec<f64> = tpz.iter().map(|&val| val / f).collect();

            v_store[ti] = v;
            f_store[ti] = f;
            k_store[ti * m..ti * m + m].copy_from_slice(&k);

            log_lik += -0.5 * (two_pi_ln + f.ln() + v * v / f);

            // a_{t+1} = T a + K v.
            let ta = mat_vec(t, &a, m);
            let a_next: Vec<f64> = ta
                .iter()
                .zip(k.iter())
                .map(|(&tv, &kv)| tv + kv * v)
                .collect();

            // L = T − K Z (Z selects columns 0 and sea_idx).
            let mut l = t.clone();
            for (i, &kv) in k.iter().enumerate() {
                l[i * m] -= kv;
                if sea_dim > 0 {
                    l[i * m + sea_idx] -= kv;
                }
            }

            // P_{t+1} = T P Lᵀ + R Q Rᵀ.
            let tp = mat_mul(t, &p, m);
            let mut p_next = mat_mul_bt(&tp, &l, m);
            p_next[0] += lev;
            p_next[m + 1] += slo;
            if sea_dim > 0 {
                p_next[sea_idx * m + sea_idx] += sea;
            }

            a = a_next;
            p = p_next;
        }

        // Backward disturbance smoother.
        let mut r = vec![0.0_f64; m];
        let mut nmat = vec![0.0_f64; m * m];
        let mut trend = vec![0.0_f32; n];
        let mut seasonal = vec![0.0_f32; n];
        let mut irregular = vec![0.0_f32; n];
        let mut final_state = vec![0.0_f64; m];
        let (mut s_eps2, mut s_lev, mut s_slo, mut s_sea) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

        for ti in (0..n).rev() {
            let a_t = &a_store[ti * m..ti * m + m];
            let p_t = &p_store[ti * m * m..(ti + 1) * m * m];
            let v = v_store[ti];
            let f = f_store[ti];
            let k = &k_store[ti * m..ti * m + m];

            // EM statistics use the *incoming* r_t, N_t (Durbin–Koopman 4.69).
            let kr = dot(k, &r);
            let eps_hat = h * (v / f - kr);
            let knk = quad(k, &nmat, m);
            let var_eps = (h - h * h * (1.0 / f + knk)).max(0.0);
            s_eps2 += eps_hat * eps_hat + var_eps;

            let eta_lev = lev * r[0];
            s_lev += eta_lev * eta_lev + (lev - lev * lev * nmat[0]).max(0.0);
            let eta_slo = slo * r[1];
            s_slo += eta_slo * eta_slo + (slo - slo * slo * nmat[m + 1]).max(0.0);
            if sea_dim > 0 {
                let eta_sea = sea * r[sea_idx];
                let nnn = nmat[sea_idx * m + sea_idx];
                s_sea += eta_sea * eta_sea + (sea - sea * sea * nnn).max(0.0);
            }

            // L = T − K Z.
            let mut l = t.clone();
            for (i, &kv) in k.iter().enumerate() {
                l[i * m] -= kv;
                if sea_dim > 0 {
                    l[i * m + sea_idx] -= kv;
                }
            }

            // r_{t-1} = Zᵀ v/F + Lᵀ r_t.
            let mut r_new = mat_t_vec(&l, &r, m);
            let vf = v / f;
            r_new[0] += vf;
            if sea_dim > 0 {
                r_new[sea_idx] += vf;
            }

            // N_{t-1} = Zᵀ Z / F + Lᵀ N_t L.
            let nl = mat_mul(&nmat, &l, m);
            let mut n_new = mat_mul_at(&l, &nl, m);
            let inv_f = 1.0 / f;
            n_new[0] += inv_f;
            if sea_dim > 0 {
                n_new[sea_idx] += inv_f;
                n_new[sea_idx * m] += inv_f;
                n_new[sea_idx * m + sea_idx] += inv_f;
            }

            // Smoothed state α̂_t = a_t + P_t r_{t-1}.
            let pr = mat_vec(p_t, &r_new, m);
            let level = a_t[0] + pr[0];
            let seas = if sea_dim > 0 {
                a_t[sea_idx] + pr[sea_idx]
            } else {
                0.0
            };
            trend[ti] = level as f32;
            seasonal[ti] = seas as f32;
            irregular[ti] = y[ti] as f32 - level as f32 - seas as f32;

            if ti == n - 1 {
                let alpha: Vec<f64> = a_t
                    .iter()
                    .zip(pr.iter())
                    .map(|(&av, &pv)| av + pv)
                    .collect();
                final_state.copy_from_slice(&alpha);
            }

            r = r_new;
            nmat = n_new;
        }

        SmoothOut {
            trend,
            seasonal,
            irregular,
            final_state,
            log_likelihood: log_lik,
            sum_eps2: s_eps2,
            sum_eta_level: s_lev,
            sum_eta_slope: s_slo,
            sum_eta_seasonal: s_sea,
        }
    }
}

// ── Free helpers ───────────────────────────────────────────────────────────────

/// Build the `m × m` structural transition matrix `T` (row-major).
fn build_transition(m: usize, sea_dim: usize) -> Vec<f64> {
    let mut t = vec![0.0_f64; m * m];
    t[0] = 1.0; // level ← level
    t[1] = 1.0; // level ← slope
    t[m + 1] = 1.0; // slope ← slope
    if sea_dim > 0 {
        let sea = 2usize;
        for j in sea..m {
            t[sea * m + j] = -1.0; // γ_t = −Σ γ_{t-j}
        }
        for r in (sea + 1)..m {
            t[r * m + (r - 1)] = 1.0; // seasonal shift register
        }
    }
    t
}

/// Population variance of `v` (0 for fewer than two elements).
fn sample_variance(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / v.len() as f64
}

/// Dot product of two equal-length slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Quadratic form `kᵀ N k` for an `m × m` matrix `n` (row-major).
fn quad(k: &[f64], n: &[f64], m: usize) -> f64 {
    (0..m)
        .map(|i| k[i] * (0..m).map(|j| n[i * m + j] * k[j]).sum::<f64>())
        .sum()
}

/// Matrix–vector product `a · x` for an `m × m` matrix `a`.
fn mat_vec(a: &[f64], x: &[f64], m: usize) -> Vec<f64> {
    (0..m)
        .map(|i| {
            a[i * m..i * m + m]
                .iter()
                .zip(x.iter())
                .map(|(&av, &xv)| av * xv)
                .sum()
        })
        .collect()
}

/// Transposed matrix–vector product `aᵀ · x` for an `m × m` matrix `a`.
fn mat_t_vec(a: &[f64], x: &[f64], m: usize) -> Vec<f64> {
    (0..m)
        .map(|j| (0..m).map(|i| a[i * m + j] * x[i]).sum())
        .collect()
}

/// Matrix product `a · b` for `m × m` matrices (row-major).
fn mat_mul(a: &[f64], b: &[f64], m: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * m];
    for i in 0..m {
        for l in 0..m {
            let ail = a[i * m + l];
            for j in 0..m {
                c[i * m + j] += ail * b[l * m + j];
            }
        }
    }
    c
}

/// Matrix product `aᵀ · b` for `m × m` matrices (row-major).
fn mat_mul_at(a: &[f64], b: &[f64], m: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * m];
    for l in 0..m {
        for i in 0..m {
            let ali = a[l * m + i];
            for j in 0..m {
                c[i * m + j] += ali * b[l * m + j];
            }
        }
    }
    c
}

/// Matrix product `a · bᵀ` for `m × m` matrices (row-major).
fn mat_mul_bt(a: &[f64], b: &[f64], m: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * m];
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0.0_f64;
            for l in 0..m {
                acc += a[i * m + l] * b[j * m + l];
            }
            c[i * m + j] = acc;
        }
    }
    c
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn line(n: usize, a: f32, b: f32) -> Vec<f32> {
        (0..n).map(|t| a + b * t as f32).collect()
    }

    #[test]
    fn sts_recovers_pure_linear_trend() {
        let n = 80;
        let (a, b) = (2.0_f32, 0.3_f32);
        let y = line(n, a, b);
        let cfg = StsConfig::default().with_variances(1e-3, 1e-2, 1e-3, 1e-6);
        let mut sts = StsDecomposer::new(4, cfg).expect("new");
        sts.fit(&y).expect("fit");
        let range = b * (n as f32 - 1.0);
        let tol = 0.05 * range;
        let (trend, seasonal, irregular) = (sts.trend(), sts.seasonal(), sts.irregular());
        for (t, &yt) in y.iter().enumerate().take(n - 8).skip(8) {
            assert!(
                (trend[t] - yt).abs() < tol,
                "trend[{t}]={} y={yt} tol={tol}",
                trend[t]
            );
            assert!(seasonal[t].abs() < tol, "seasonal[{t}]={}", seasonal[t]);
            assert!(irregular[t].abs() < tol, "irregular[{t}]={}", irregular[t]);
        }
    }

    #[test]
    fn sts_recovers_known_seasonal() {
        let n = 80;
        let s = 4usize;
        let pat = [1.0_f32, -0.5, 0.5, -1.0]; // zero-mean over a period
        let y: Vec<f32> = (0..n).map(|t| 0.2 * t as f32 + pat[t % s]).collect();
        let cfg = StsConfig::default().with_variances(1e-3, 1e-3, 1e-4, 0.1);
        let mut sts = StsDecomposer::new(s, cfg).expect("new");
        sts.fit(&y).expect("fit");
        // Seasonal sums to ≈0 over each period and matches the injected shape.
        for t in 8..n - 8 - s {
            let win: f32 = (0..s).map(|j| sts.seasonal()[t + j]).sum();
            assert!(win.abs() < 0.2, "seasonal period sum at {t} = {win}");
            assert!(
                (sts.seasonal()[t] - pat[t % s]).abs() < 0.25,
                "seasonal[{t}]={} expected={}",
                sts.seasonal()[t],
                pat[t % s]
            );
        }
    }

    #[test]
    fn sts_components_reconstruct_series() {
        let n = 60;
        let s = 4usize;
        let pat = [0.8_f32, -0.4, 0.3, -0.7];
        let y: Vec<f32> = (0..n).map(|t| 0.15 * t as f32 + pat[t % s]).collect();
        let cfg = StsConfig::default().with_variances(1e-2, 1e-2, 1e-3, 0.1);
        let mut sts = StsDecomposer::new(s, cfg).expect("new");
        sts.fit(&y).expect("fit");
        let (trend, seasonal, irregular) = (sts.trend(), sts.seasonal(), sts.irregular());
        for (t, &yt) in y.iter().enumerate() {
            let recon = trend[t] + seasonal[t] + irregular[t];
            assert!((recon - yt).abs() < 1e-3, "t={t} recon={recon} y={yt}");
        }
    }

    #[test]
    fn sts_forecast_continues_line() {
        let n = 80;
        let (a, b) = (1.0_f32, 0.4_f32);
        let y = line(n, a, b);
        let cfg = StsConfig::default().with_variances(1e-3, 1e-2, 1e-3, 1e-6);
        let mut sts = StsDecomposer::new(4, cfg).expect("new");
        sts.fit(&y).expect("fit");
        let h = 10;
        let fc = sts.forecast(h).expect("forecast");
        assert_eq!(fc.len(), h);
        let range = b * (n as f32 - 1.0);
        let tol = 0.06 * range;
        for (i, &val) in fc.iter().enumerate() {
            let expected = a + b * (n + i) as f32;
            assert!(
                (val - expected).abs() < tol,
                "fc[{i}]={val} expected={expected} tol={tol}"
            );
        }
    }

    #[test]
    fn sts_em_increases_log_likelihood() {
        let n = 80;
        let s = 4usize;
        let pat = [1.0_f32, -0.6, 0.4, -0.8];
        let mut rng = LcgRng::new(7);
        let mut noise = vec![0.0_f32; n];
        rng.fill_normal(&mut noise);
        let y: Vec<f32> = (0..n)
            .map(|t| 0.1 * t as f32 + pat[t % s] + 0.3 * noise[t])
            .collect();

        let base_cfg = StsConfig::default().with_variances(0.5, 0.1, 0.01, 0.1);
        let mut no_em = StsDecomposer::new(s, base_cfg.clone()).expect("new");
        no_em.fit(&y).expect("fit");
        let ll0 = no_em.log_likelihood();

        let mut with_em = StsDecomposer::new(s, base_cfg.with_em_iters(15)).expect("new");
        with_em.fit(&y).expect("fit");
        let ll1 = with_em.log_likelihood();

        assert!(
            ll1.is_finite() && ll0.is_finite(),
            "log-likelihoods must be finite"
        );
        assert!(
            ll1 >= ll0 - 1e-2,
            "EM decreased log-likelihood: {ll0} -> {ll1}"
        );
        let (o, l, sl, se) = with_em.variances();
        assert!(
            o > 0.0 && l > 0.0 && sl > 0.0 && se > 0.0,
            "variances must stay positive"
        );
    }

    #[test]
    fn sts_err_zero_season() {
        assert!(matches!(
            StsDecomposer::new(0, StsConfig::default()).unwrap_err(),
            TsError::InvalidSequenceLength(0)
        ));
    }

    #[test]
    fn sts_err_empty_series() {
        let mut sts = StsDecomposer::new(4, StsConfig::default()).expect("new");
        assert!(matches!(
            sts.fit(&[]).unwrap_err(),
            TsError::EmptyInput { .. }
        ));
    }

    #[test]
    fn sts_forecast_before_fit_errors() {
        let sts = StsDecomposer::new(4, StsConfig::default()).expect("new");
        assert!(sts.forecast(3).is_err());
    }

    #[test]
    fn sts_forecast_h_zero_is_empty() {
        let y = line(40, 0.0, 1.0);
        let mut sts = StsDecomposer::new(4, StsConfig::default()).expect("new");
        sts.fit(&y).expect("fit");
        assert!(sts.forecast(0).expect("ok").is_empty());
    }

    #[test]
    fn sts_non_seasonal_local_linear_trend() {
        // season_length == 1 → pure local-linear trend, no seasonal states.
        let y = line(50, 5.0, -0.2);
        let cfg = StsConfig::default().with_variances(1e-3, 1e-2, 1e-3, 1e-6);
        let mut sts = StsDecomposer::new(1, cfg).expect("new");
        assert_eq!(sts.state_dim(), 2);
        sts.fit(&y).expect("fit");
        let (trend, seasonal) = (sts.trend(), sts.seasonal());
        for (t, &yt) in y.iter().enumerate().take(44).skip(6) {
            assert!((trend[t] - yt).abs() < 0.6, "trend[{t}]={}", trend[t]);
            assert_eq!(seasonal[t], 0.0);
        }
    }
}
