//! Kalman filter and Rauch–Tung–Striebel (RTS) smoother for linear-Gaussian
//! state-space models.
//!
//! The model is
//!
//! ```text
//! xₖ = F xₖ₋₁ + B uₖ + wₖ,   wₖ ~ N(0, Q)   (state / process model)
//! zₖ = H xₖ        + vₖ,      vₖ ~ N(0, R)   (measurement model)
//! ```
//!
//! The **filter** computes, for every step k, the mean and covariance of the
//! state given measurements up to k. The **smoother** runs a backward pass to
//! condition on the *entire* measurement sequence, which can only reduce the
//! posterior variance.
//!
//! The covariance update uses the **Joseph stabilised form**
//! `P = (I − KH) P⁻ (I − KH)ᵀ + K R Kᵀ`, which preserves symmetry and positive
//! semi-definiteness even with finite-precision arithmetic.
//!
//! # References
//! - Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction
//!   Problems." *J. Basic Eng.* 82(1):35-45.
//! - Rauch, Tung & Striebel (1965). "Maximum Likelihood Estimates of Linear
//!   Dynamic Systems." *AIAA Journal* 3(8):1445-1450.
//! - Särkkä, S. (2013). *Bayesian Filtering and Smoothing*, CUP.

use crate::error::{StatsError, StatsResult};

// ─────────────────────────────────────────────────────────────────────────────
// Dense row-major linear-algebra helpers (no external dependencies)
// ─────────────────────────────────────────────────────────────────────────────

/// `C = A · B` where `A` is `m × k` and `B` is `k × n` (all row-major).
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            if aip == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += aip * b[p * n + j];
            }
        }
    }
    c
}

/// `Aᵀ` for an `m × n` row-major matrix, returning an `n × m` matrix.
fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut t = vec![0.0_f64; n * m];
    for i in 0..m {
        for j in 0..n {
            t[j * m + i] = a[i * n + j];
        }
    }
    t
}

/// Element-wise `A + B` for equally-sized matrices.
fn mat_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// Element-wise `A − B` for equally-sized matrices.
fn mat_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// Symmetrise a square `d × d` matrix as `(A + Aᵀ)/2` to suppress round-off skew.
fn symmetrise(a: &[f64], d: usize) -> Vec<f64> {
    let mut s = vec![0.0_f64; d * d];
    for i in 0..d {
        for j in 0..d {
            s[i * d + j] = 0.5 * (a[i * d + j] + a[j * d + i]);
        }
    }
    s
}

/// `d × d` identity matrix.
fn identity(d: usize) -> Vec<f64> {
    let mut m = vec![0.0_f64; d * d];
    for i in 0..d {
        m[i * d + i] = 1.0;
    }
    m
}

/// Invert a square `n × n` row-major matrix via Gauss–Jordan elimination with
/// partial pivoting. Returns an error if the matrix is singular.
fn inverse(a: &[f64], n: usize) -> StatsResult<Vec<f64>> {
    let mut aug = vec![0.0_f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }
    for col in 0..n {
        // Partial pivot: largest magnitude in this column at or below the diagonal.
        let mut pivot = col;
        let mut best = aug[col * 2 * n + col].abs();
        for r in (col + 1)..n {
            let v = aug[r * 2 * n + col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-300 {
            return Err(StatsError::SingularMatrix(
                "kalman matrix inverse".to_string(),
            ));
        }
        if pivot != col {
            for j in 0..2 * n {
                aug.swap(col * 2 * n + j, pivot * 2 * n + j);
            }
        }
        // Normalise the pivot row.
        let diag = aug[col * 2 * n + col];
        for j in 0..2 * n {
            aug[col * 2 * n + j] /= diag;
        }
        // Eliminate the column from every other row.
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = aug[r * 2 * n + col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..2 * n {
                aug[r * 2 * n + j] -= factor * aug[col * 2 * n + j];
            }
        }
    }
    let mut inv = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    Ok(inv)
}

// ─────────────────────────────────────────────────────────────────────────────
// Model and per-step records
// ─────────────────────────────────────────────────────────────────────────────

/// Linear-Gaussian state-space model definition (row-major matrices).
///
/// State dimension `n_state`, measurement dimension `n_obs`, optional control
/// dimension `n_ctrl` (set the `b` matrix to `None` if there is no control input).
#[derive(Debug, Clone)]
pub struct LinearGaussianModel {
    /// State-transition matrix `F` (`n_state × n_state`).
    pub f: Vec<f64>,
    /// Observation matrix `H` (`n_obs × n_state`).
    pub h: Vec<f64>,
    /// Process-noise covariance `Q` (`n_state × n_state`).
    pub q: Vec<f64>,
    /// Measurement-noise covariance `R` (`n_obs × n_obs`).
    pub r: Vec<f64>,
    /// Optional control matrix `B` (`n_state × n_ctrl`).
    pub b: Option<Vec<f64>>,
    /// State dimension.
    pub n_state: usize,
    /// Measurement dimension.
    pub n_obs: usize,
    /// Control dimension (0 when there is no control input).
    pub n_ctrl: usize,
}

impl LinearGaussianModel {
    /// Construct a model without a control input.
    pub fn new(
        f: Vec<f64>,
        h: Vec<f64>,
        q: Vec<f64>,
        r: Vec<f64>,
        n_state: usize,
        n_obs: usize,
    ) -> StatsResult<Self> {
        if n_state == 0 || n_obs == 0 {
            return Err(StatsError::InvalidParameter {
                name: "dimensions".to_string(),
                reason: "n_state and n_obs must be ≥ 1".to_string(),
            });
        }
        if f.len() != n_state * n_state {
            return Err(StatsError::ShapeMismatch {
                expected: vec![n_state, n_state],
                got: vec![f.len()],
            });
        }
        if h.len() != n_obs * n_state {
            return Err(StatsError::ShapeMismatch {
                expected: vec![n_obs, n_state],
                got: vec![h.len()],
            });
        }
        if q.len() != n_state * n_state {
            return Err(StatsError::ShapeMismatch {
                expected: vec![n_state, n_state],
                got: vec![q.len()],
            });
        }
        if r.len() != n_obs * n_obs {
            return Err(StatsError::ShapeMismatch {
                expected: vec![n_obs, n_obs],
                got: vec![r.len()],
            });
        }
        Ok(Self {
            f,
            h,
            q,
            r,
            b: None,
            n_state,
            n_obs,
            n_ctrl: 0,
        })
    }

    /// Attach a control matrix `B` (`n_state × n_ctrl`).
    pub fn with_control(mut self, b: Vec<f64>, n_ctrl: usize) -> StatsResult<Self> {
        if b.len() != self.n_state * n_ctrl {
            return Err(StatsError::ShapeMismatch {
                expected: vec![self.n_state, n_ctrl],
                got: vec![b.len()],
            });
        }
        self.b = Some(b);
        self.n_ctrl = n_ctrl;
        Ok(self)
    }
}

/// Per-step records produced by the forward Kalman pass.
#[derive(Debug, Clone)]
pub struct KalmanFilterResult {
    /// Filtered state means `x_{k|k}`, row-major `n_steps × n_state`.
    pub filtered_mean: Vec<f64>,
    /// Filtered covariances `P_{k|k}`, row-major `n_steps × n_state × n_state`.
    pub filtered_cov: Vec<f64>,
    /// Predicted state means `x_{k|k−1}`, row-major `n_steps × n_state`.
    pub predicted_mean: Vec<f64>,
    /// Predicted covariances `P_{k|k−1}`, row-major `n_steps × n_state × n_state`.
    pub predicted_cov: Vec<f64>,
    /// Innovations `z_k − H x_{k|k−1}`, row-major `n_steps × n_obs`.
    pub innovations: Vec<f64>,
    /// Total log-likelihood of the observations under the model.
    pub log_likelihood: f64,
    /// Number of time steps.
    pub n_steps: usize,
    /// State dimension.
    pub n_state: usize,
    /// Measurement dimension.
    pub n_obs: usize,
}

impl KalmanFilterResult {
    /// Borrow the filtered mean at step `k`.
    #[must_use]
    pub fn filtered_mean_at(&self, k: usize) -> &[f64] {
        &self.filtered_mean[k * self.n_state..(k + 1) * self.n_state]
    }

    /// Borrow the filtered covariance at step `k` (`n_state × n_state`).
    #[must_use]
    pub fn filtered_cov_at(&self, k: usize) -> &[f64] {
        let s = self.n_state * self.n_state;
        &self.filtered_cov[k * s..(k + 1) * s]
    }

    /// Borrow the innovation at step `k`.
    #[must_use]
    pub fn innovation_at(&self, k: usize) -> &[f64] {
        &self.innovations[k * self.n_obs..(k + 1) * self.n_obs]
    }
}

/// Result of the RTS backward smoother.
#[derive(Debug, Clone)]
pub struct KalmanSmootherResult {
    /// Smoothed state means `x_{k|N}`, row-major `n_steps × n_state`.
    pub smoothed_mean: Vec<f64>,
    /// Smoothed covariances `P_{k|N}`, row-major `n_steps × n_state × n_state`.
    pub smoothed_cov: Vec<f64>,
    /// Number of time steps.
    pub n_steps: usize,
    /// State dimension.
    pub n_state: usize,
}

impl KalmanSmootherResult {
    /// Borrow the smoothed mean at step `k`.
    #[must_use]
    pub fn smoothed_mean_at(&self, k: usize) -> &[f64] {
        &self.smoothed_mean[k * self.n_state..(k + 1) * self.n_state]
    }

    /// Borrow the smoothed covariance at step `k` (`n_state × n_state`).
    #[must_use]
    pub fn smoothed_cov_at(&self, k: usize) -> &[f64] {
        let s = self.n_state * self.n_state;
        &self.smoothed_cov[k * s..(k + 1) * s]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kalman filter (forward pass)
// ─────────────────────────────────────────────────────────────────────────────

/// Run the forward Kalman filter over `n_steps` measurements.
///
/// # Parameters
/// - `model` — the linear-Gaussian model.
/// - `x0` / `p0` — the prior state mean (`n_state`) and covariance
///   (`n_state × n_state`).
/// - `measurements` — row-major `n_steps × n_obs`.
/// - `controls` — optional row-major `n_steps × n_ctrl` control inputs; required
///   exactly when the model carries a control matrix.
pub fn kalman_filter(
    model: &LinearGaussianModel,
    x0: &[f64],
    p0: &[f64],
    measurements: &[f64],
    controls: Option<&[f64]>,
) -> StatsResult<KalmanFilterResult> {
    let ns = model.n_state;
    let no = model.n_obs;
    if x0.len() != ns {
        return Err(StatsError::DimensionMismatch { a: x0.len(), b: ns });
    }
    if p0.len() != ns * ns {
        return Err(StatsError::ShapeMismatch {
            expected: vec![ns, ns],
            got: vec![p0.len()],
        });
    }
    if measurements.is_empty() {
        return Err(StatsError::EmptyInput);
    }
    if measurements.len() % no != 0 {
        return Err(StatsError::ShapeMismatch {
            expected: vec![0, no],
            got: vec![measurements.len()],
        });
    }
    let n_steps = measurements.len() / no;

    if let Some(b) = &model.b {
        let nc = model.n_ctrl;
        let ctrl = controls.ok_or_else(|| StatsError::InvalidParameter {
            name: "controls".to_string(),
            reason: "model has a control matrix but no controls were supplied".to_string(),
        })?;
        if ctrl.len() != n_steps * nc {
            return Err(StatsError::ShapeMismatch {
                expected: vec![n_steps, nc],
                got: vec![ctrl.len()],
            });
        }
        let _ = b;
    }

    let ft = transpose(&model.f, ns, ns);
    let ht = transpose(&model.h, no, ns);
    let id = identity(ns);
    let two_pi = std::f64::consts::TAU;

    let mut filtered_mean = vec![0.0_f64; n_steps * ns];
    let mut filtered_cov = vec![0.0_f64; n_steps * ns * ns];
    let mut predicted_mean = vec![0.0_f64; n_steps * ns];
    let mut predicted_cov = vec![0.0_f64; n_steps * ns * ns];
    let mut innovations = vec![0.0_f64; n_steps * no];
    let mut log_likelihood = 0.0_f64;

    let mut x = x0.to_vec();
    let mut p = p0.to_vec();

    for k in 0..n_steps {
        // ── Predict ──────────────────────────────────────────────────────────
        // x⁻ = F x  (+ B u)
        let mut x_pred = matmul(&model.f, &x, ns, ns, 1);
        if let (Some(b), Some(ctrl)) = (&model.b, controls) {
            let nc = model.n_ctrl;
            let uk = &ctrl[k * nc..(k + 1) * nc];
            let bu = matmul(b, uk, ns, nc, 1);
            for i in 0..ns {
                x_pred[i] += bu[i];
            }
        }
        // P⁻ = F P Fᵀ + Q
        let fp = matmul(&model.f, &p, ns, ns, ns);
        let fpft = matmul(&fp, &ft, ns, ns, ns);
        let p_pred = symmetrise(&mat_add(&fpft, &model.q), ns);

        predicted_mean[k * ns..(k + 1) * ns].copy_from_slice(&x_pred);
        predicted_cov[k * ns * ns..(k + 1) * ns * ns].copy_from_slice(&p_pred);

        // ── Update ───────────────────────────────────────────────────────────
        let zk = &measurements[k * no..(k + 1) * no];
        // Innovation y = z − H x⁻
        let hx = matmul(&model.h, &x_pred, no, ns, 1);
        let mut y = vec![0.0_f64; no];
        for i in 0..no {
            y[i] = zk[i] - hx[i];
        }
        // Innovation covariance S = H P⁻ Hᵀ + R
        let hp = matmul(&model.h, &p_pred, no, ns, ns);
        let hpht = matmul(&hp, &ht, no, ns, no);
        let s = symmetrise(&mat_add(&hpht, &model.r), no);
        let s_inv = inverse(&s, no)?;

        // Kalman gain K = P⁻ Hᵀ S⁻¹
        let pht = matmul(&p_pred, &ht, ns, ns, no);
        let k_gain = matmul(&pht, &s_inv, ns, no, no);

        // State update x = x⁻ + K y
        let ky = matmul(&k_gain, &y, ns, no, 1);
        for i in 0..ns {
            x[i] = x_pred[i] + ky[i];
        }

        // Joseph-form covariance: P = (I − KH) P⁻ (I − KH)ᵀ + K R Kᵀ
        let kh = matmul(&k_gain, &model.h, ns, no, ns);
        let i_kh = mat_sub(&id, &kh);
        let i_kh_t = transpose(&i_kh, ns, ns);
        let term1 = matmul(&matmul(&i_kh, &p_pred, ns, ns, ns), &i_kh_t, ns, ns, ns);
        let kr = matmul(&k_gain, &model.r, ns, no, no);
        let krk = matmul(&kr, &transpose(&k_gain, ns, no), ns, no, ns);
        p = symmetrise(&mat_add(&term1, &krk), ns);

        filtered_mean[k * ns..(k + 1) * ns].copy_from_slice(&x);
        filtered_cov[k * ns * ns..(k + 1) * ns * ns].copy_from_slice(&p);
        innovations[k * no..(k + 1) * no].copy_from_slice(&y);

        // ── Log-likelihood contribution: N(y; 0, S) ──────────────────────────
        // det(S) via the product of pivots from an LU factorisation with
        // partial pivoting.
        let det = det_via_lu(&s, no).unwrap_or(1.0);
        if det > 0.0 {
            let sy = matmul(&s_inv, &y, no, no, 1);
            let quad: f64 = (0..no).map(|i| y[i] * sy[i]).sum();
            log_likelihood += -0.5 * (no as f64 * two_pi.ln() + det.ln() + quad);
        }
    }

    Ok(KalmanFilterResult {
        filtered_mean,
        filtered_cov,
        predicted_mean,
        predicted_cov,
        innovations,
        log_likelihood,
        n_steps,
        n_state: ns,
        n_obs: no,
    })
}

/// Determinant of a square `n × n` matrix via LU with partial pivoting.
fn det_via_lu(a: &[f64], n: usize) -> Option<f64> {
    let mut m = a.to_vec();
    let mut det = 1.0_f64;
    for col in 0..n {
        let mut pivot = col;
        let mut best = m[col * n + col].abs();
        for r in (col + 1)..n {
            let v = m[r * n + col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-300 {
            return Some(0.0);
        }
        if pivot != col {
            for j in 0..n {
                m.swap(col * n + j, pivot * n + j);
            }
            det = -det;
        }
        let diag = m[col * n + col];
        det *= diag;
        for r in (col + 1)..n {
            let factor = m[r * n + col] / diag;
            for j in col..n {
                m[r * n + j] -= factor * m[col * n + j];
            }
        }
    }
    Some(det)
}

// ─────────────────────────────────────────────────────────────────────────────
// RTS smoother (backward pass)
// ─────────────────────────────────────────────────────────────────────────────

/// Run the Rauch–Tung–Striebel backward smoother on a completed filter pass.
///
/// For each step (from `N−1` down to 0):
///
/// ```text
/// C   = P_f Fᵀ (P⁻_{k+1})⁻¹
/// x_s = x_f + C (x_s,{k+1} − x⁻_{k+1})
/// P_s = P_f + C (P_s,{k+1} − P⁻_{k+1}) Cᵀ
/// ```
pub fn rts_smoother(
    model: &LinearGaussianModel,
    filter: &KalmanFilterResult,
) -> StatsResult<KalmanSmootherResult> {
    let ns = model.n_state;
    let n = filter.n_steps;
    let s2 = ns * ns;
    let ft = transpose(&model.f, ns, ns);

    let mut smoothed_mean = filter.filtered_mean.clone();
    let mut smoothed_cov = filter.filtered_cov.clone();

    // The last step's smoothed estimate equals the filtered estimate.
    for k in (0..n.saturating_sub(1)).rev() {
        let xf = &filter.filtered_mean[k * ns..(k + 1) * ns];
        let pf = &filter.filtered_cov[k * s2..(k + 1) * s2];
        let x_pred_next = &filter.predicted_mean[(k + 1) * ns..(k + 2) * ns];
        let p_pred_next = &filter.predicted_cov[(k + 1) * s2..(k + 2) * s2];
        let p_pred_inv = inverse(p_pred_next, ns)?;

        // Smoother gain C = P_f Fᵀ (P⁻_{k+1})⁻¹
        let pf_ft = matmul(pf, &ft, ns, ns, ns);
        let c = matmul(&pf_ft, &p_pred_inv, ns, ns, ns);

        // x_s = x_f + C (x_s,{k+1} − x⁻_{k+1})
        let xs_next = smoothed_mean[(k + 1) * ns..(k + 2) * ns].to_vec();
        let mut dx = vec![0.0_f64; ns];
        for i in 0..ns {
            dx[i] = xs_next[i] - x_pred_next[i];
        }
        let cdx = matmul(&c, &dx, ns, ns, 1);
        for i in 0..ns {
            smoothed_mean[k * ns + i] = xf[i] + cdx[i];
        }

        // P_s = P_f + C (P_s,{k+1} − P⁻_{k+1}) Cᵀ
        let ps_next = smoothed_cov[(k + 1) * s2..(k + 2) * s2].to_vec();
        let dp = mat_sub(&ps_next, p_pred_next);
        let ct = transpose(&c, ns, ns);
        let cdp = matmul(&c, &dp, ns, ns, ns);
        let cdpct = matmul(&cdp, &ct, ns, ns, ns);
        let ps = symmetrise(&mat_add(pf, &cdpct), ns);
        smoothed_cov[k * s2..(k + 1) * s2].copy_from_slice(&ps);
    }

    Ok(KalmanSmootherResult {
        smoothed_mean,
        smoothed_cov,
        n_steps: n,
        n_state: ns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // 2×2 symmetric-matrix eigenvalues (for the PSD check).
    fn eigenvalues_2x2(m: &[f64]) -> (f64, f64) {
        let a = m[0];
        let b = m[1];
        let d = m[3];
        let trace = a + d;
        let det = a * d - b * b;
        let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
        ((trace + disc) / 2.0, (trace - disc) / 2.0)
    }

    #[test]
    fn constant_position_converges_and_cov_decreases() {
        // 1-D constant position with measurement noise. Filtered variance must
        // decrease monotonically to a steady state and the estimate approach truth.
        let model = LinearGaussianModel::new(
            vec![1.0],  // F
            vec![1.0],  // H
            vec![1e-5], // Q (almost-constant state)
            vec![0.5],  // R
            1,
            1,
        )
        .expect("model ok");

        let truth = 3.0;
        let mut rng = LcgRng::new(11);
        let n = 60;
        let measurements: Vec<f64> = (0..n)
            .map(|_| truth + 0.5_f64.sqrt() * rng.next_normal())
            .collect();

        let out = kalman_filter(&model, &[0.0], &[10.0], &measurements, None).expect("filter ok");
        // Variance monotone non-increasing.
        for k in 1..n {
            let prev = out.filtered_cov_at(k - 1)[0];
            let cur = out.filtered_cov_at(k)[0];
            assert!(
                cur <= prev + 1e-9,
                "P increased at step {k}: {prev} -> {cur}"
            );
        }
        // Final estimate close to truth.
        let est = out.filtered_mean_at(n - 1)[0];
        assert!((est - truth).abs() < 0.3, "estimate {est} vs truth {truth}");
        // Final variance much smaller than the prior.
        assert!(out.filtered_cov_at(n - 1)[0] < 0.1);
    }

    #[test]
    fn constant_velocity_tracks_ramp() {
        // State = [position, velocity]; constant-velocity dynamics with dt = 1.
        let dt = 1.0;
        let model = LinearGaussianModel::new(
            vec![1.0, dt, 0.0, 1.0], // F
            vec![1.0, 0.0],          // H (observe position)
            vec![1e-4, 0.0, 0.0, 1e-4],
            vec![0.25], // R
            2,
            1,
        )
        .expect("model ok");

        let velocity = 0.5;
        let mut rng = LcgRng::new(7);
        let n = 80;
        let measurements: Vec<f64> = (0..n)
            .map(|k| velocity * k as f64 + 0.5 * rng.next_normal())
            .collect();

        let out = kalman_filter(
            &model,
            &[0.0, 0.0],
            &[5.0, 0.0, 0.0, 5.0],
            &measurements,
            None,
        )
        .expect("filter ok");

        // Late position error should be small.
        let k = n - 1;
        let true_pos = velocity * k as f64;
        let est_pos = out.filtered_mean_at(k)[0];
        assert!(
            (est_pos - true_pos).abs() < 1.0,
            "pos {est_pos} vs {true_pos}"
        );
        // Velocity estimate should approach the true velocity.
        let est_vel = out.filtered_mean_at(k)[1];
        assert!(
            (est_vel - velocity).abs() < 0.25,
            "vel {est_vel} vs {velocity}"
        );
    }

    #[test]
    fn covariance_stays_symmetric_and_psd() {
        let model = LinearGaussianModel::new(
            vec![1.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.01, 0.0, 0.0, 0.01],
            vec![0.3],
            2,
            1,
        )
        .expect("model ok");
        let mut rng = LcgRng::new(99);
        let n = 50;
        let measurements: Vec<f64> = (0..n).map(|k| 0.2 * k as f64 + rng.next_normal()).collect();
        let out = kalman_filter(
            &model,
            &[0.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0],
            &measurements,
            None,
        )
        .expect("filter ok");
        for k in 0..n {
            let p = out.filtered_cov_at(k);
            // Symmetry.
            assert!((p[1] - p[2]).abs() < 1e-9, "asymmetric P at {k}");
            // PSD: both eigenvalues ≥ −1e-9.
            let (l1, l2) = eigenvalues_2x2(p);
            assert!(l1 >= -1e-9 && l2 >= -1e-9, "non-PSD at {k}: {l1}, {l2}");
        }
    }

    #[test]
    fn steady_state_gain_matches_riccati() {
        // Scalar model: F = H = 1. The steady-state predicted variance p̄ solves
        // the algebraic Riccati equation
        //   p̄ = (p̄ - p̄²/(p̄+R)) + Q   ⇔   p̄² ... ; with steady gain k = p̄/(p̄+R).
        let q = 0.1;
        let r = 1.0;
        let model = LinearGaussianModel::new(vec![1.0], vec![1.0], vec![q], vec![r], 1, 1)
            .expect("model ok");
        let mut rng = LcgRng::new(3);
        let n = 400;
        let mut x = 0.0;
        let measurements: Vec<f64> = (0..n)
            .map(|_| {
                x += q.sqrt() * rng.next_normal();
                x + r.sqrt() * rng.next_normal()
            })
            .collect();
        let out = kalman_filter(&model, &[0.0], &[1.0], &measurements, None).expect("filter ok");
        // Empirical steady-state predicted variance.
        let p_pred_ss = out.predicted_cov[(n - 1)..n][0];

        // Analytic Riccati fixed point: p_pred = F²·p_filt + Q, p_filt = p_pred·R/(p_pred+R)
        // ⇒ p_pred = p_pred·R/(p_pred+R) + Q ⇒ p_pred² - Q·p_pred - Q·R = 0.
        let disc = (q * q + 4.0 * q * r).sqrt();
        let p_pred_star = 0.5 * (q + disc);
        let k_star = p_pred_star / (p_pred_star + r);

        assert!(
            (p_pred_ss - p_pred_star).abs() < 0.02,
            "p̄ {p_pred_ss} vs {p_pred_star}"
        );
        // Steady gain from the empirical predicted variance.
        let k_emp = p_pred_ss / (p_pred_ss + r);
        assert!((k_emp - k_star).abs() < 0.02, "k {k_emp} vs {k_star}");
    }

    #[test]
    fn smoother_variance_not_greater_than_filter() {
        let model = LinearGaussianModel::new(
            vec![1.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.02, 0.0, 0.0, 0.02],
            vec![0.4],
            2,
            1,
        )
        .expect("model ok");
        let mut rng = LcgRng::new(31);
        let n = 40;
        let measurements: Vec<f64> = (0..n).map(|k| 0.3 * k as f64 + rng.next_normal()).collect();
        let filt = kalman_filter(
            &model,
            &[0.0, 0.0],
            &[2.0, 0.0, 0.0, 2.0],
            &measurements,
            None,
        )
        .expect("filter ok");
        let smooth = rts_smoother(&model, &filt).expect("smoother ok");
        for k in 0..n {
            let pf = filt.filtered_cov_at(k);
            let ps = smooth.smoothed_cov_at(k);
            // Compare the position variance (entry [0,0]); smoother ≤ filter.
            assert!(
                ps[0] <= pf[0] + 1e-7,
                "step {k}: smoother {} > filter {}",
                ps[0],
                pf[0]
            );
            // Symmetry of smoothed covariance.
            assert!((ps[1] - ps[2]).abs() < 1e-8);
        }
    }

    #[test]
    fn innovations_zero_mean_under_true_model() {
        let q = 0.05;
        let r = 0.5;
        let model = LinearGaussianModel::new(vec![1.0], vec![1.0], vec![q], vec![r], 1, 1)
            .expect("model ok");
        let mut rng = LcgRng::new(202);
        let n = 500;
        let mut x = 0.0;
        let measurements: Vec<f64> = (0..n)
            .map(|_| {
                x += q.sqrt() * rng.next_normal();
                x + r.sqrt() * rng.next_normal()
            })
            .collect();
        let out = kalman_filter(&model, &[0.0], &[1.0], &measurements, None).expect("filter ok");
        // Drop the first few transient innovations.
        let mean_innov: f64 = out.innovations[5..].iter().sum::<f64>() / (n - 5) as f64;
        assert!(mean_innov.abs() < 0.15, "mean innovation {mean_innov}");
    }

    #[test]
    fn control_input_shifts_prediction() {
        // x_k = x_{k-1} + u_k; constant control of 1.0 should ramp the state.
        let model = LinearGaussianModel::new(vec![1.0], vec![1.0], vec![1e-6], vec![0.01], 1, 1)
            .expect("model ok")
            .with_control(vec![1.0], 1)
            .expect("control ok");
        let n = 10;
        let controls = vec![1.0; n];
        // Truth: x_k = k (starting from x_0 prediction = 0 + 1 = 1 at k=0).
        let measurements: Vec<f64> = (0..n).map(|k| (k + 1) as f64).collect();
        let out = kalman_filter(&model, &[0.0], &[1.0], &measurements, Some(&controls))
            .expect("filter ok");
        let est = out.filtered_mean_at(n - 1)[0];
        assert!((est - n as f64).abs() < 0.5, "controlled estimate {est}");
    }

    #[test]
    fn rejects_singular_innovation_covariance() {
        // R = 0 and P0 = 0 ⇒ S = 0 ⇒ singular.
        let model = LinearGaussianModel::new(vec![1.0], vec![1.0], vec![0.0], vec![0.0], 1, 1)
            .expect("model ok");
        let res = kalman_filter(&model, &[0.0], &[0.0], &[1.0, 2.0], None);
        assert!(res.is_err());
    }
}
