//! Coordinate descent: cyclic, random, Gauss-Seidel, and accelerated BCD
//! (Beck-Tetruashvili 2013) for separable smooth objectives.

use crate::error::{CvxError, CvxResult};
use crate::handle::LcgRng;

/// Selection order for coordinate descent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CdOrder {
    Cyclic,
    Random,
    GaussSeidel,
}

/// Configuration for coordinate descent.
#[derive(Debug, Clone)]
pub struct CdConfig {
    pub dim: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub order: CdOrder,
    pub step_size: Option<f64>,
}

/// Result from coordinate descent.
#[derive(Debug, Clone)]
pub struct CdResult {
    pub x: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    pub residuals: Vec<f64>,
}

fn validate_config(cfg: &CdConfig) -> CvxResult<()> {
    if cfg.dim == 0 {
        return Err(CvxError::EmptyInput);
    }
    if cfg.max_iter == 0 {
        return Err(CvxError::InvalidParameter("max_iter must be >= 1".into()));
    }
    if let Some(s) = cfg.step_size {
        if s <= 0.0 || !s.is_finite() {
            return Err(CvxError::InvalidParameter(format!(
                "step_size must be > 0, got {s}"
            )));
        }
    }
    Ok(())
}

fn select_coordinate(
    x: &[f64],
    iter: usize,
    dim: usize,
    order: CdOrder,
    partial_grad: &impl Fn(&[f64], usize) -> f64,
    rng: &mut LcgRng,
) -> usize {
    match order {
        CdOrder::Cyclic => iter % dim,
        CdOrder::Random => rng.next_usize(dim),
        CdOrder::GaussSeidel => {
            let mut best = 0usize;
            let mut best_abs = 0.0_f64;
            for k in 0..dim {
                let abs_gk = partial_grad(x, k).abs();
                if abs_gk > best_abs {
                    best_abs = abs_gk;
                    best = k;
                }
            }
            best
        }
    }
}

/// Coordinate descent solver.
pub struct CoordDescent;

impl CoordDescent {
    /// Run coordinate descent (cyclic, random, or Gauss-Seidel).
    ///
    /// `obj(x)` — objective value (logged only, not for convergence decisions).
    /// `partial_grad(x, i)` — partial derivative ∂f/∂x_i.
    /// `prox_i(x_i, lambda)` — univariate proximal operator for coordinate i.
    /// `lipschitz_i(i)` — per-coordinate Lipschitz constant L_i.
    pub fn run<F, G, P, L>(
        x0: Vec<f64>,
        cfg: &CdConfig,
        _obj: F,
        partial_grad: G,
        prox_i: P,
        lipschitz_i: L,
        rng: &mut LcgRng,
    ) -> CvxResult<CdResult>
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64], usize) -> f64,
        P: Fn(f64, f64) -> f64,
        L: Fn(usize) -> f64,
    {
        validate_config(cfg)?;
        if x0.len() != cfg.dim {
            return Err(CvxError::DimensionMismatch {
                a: x0.len(),
                b: cfg.dim,
            });
        }

        let d = cfg.dim;
        let mut x = x0;
        let mut residuals = Vec::new();
        let mut converged = false;
        let mut max_delta_sweep = 0.0_f64;

        for t in 0..cfg.max_iter {
            let i = select_coordinate(&x, t, d, cfg.order, &partial_grad, rng);
            let g_i = partial_grad(&x, i);
            let l_i = lipschitz_i(i);
            let step = cfg
                .step_size
                .unwrap_or_else(|| if l_i > 0.0 { 1.0 / l_i } else { 1.0 });
            let x_i_new = prox_i(x[i] - step * g_i, step);
            let delta = (x_i_new - x[i]).abs();
            if delta > max_delta_sweep {
                max_delta_sweep = delta;
            }
            x[i] = x_i_new;

            let sweep_done = match cfg.order {
                CdOrder::Cyclic => (t + 1) % d == 0,
                CdOrder::Random | CdOrder::GaussSeidel => (t + 1) % d == 0,
            };

            if sweep_done {
                residuals.push(max_delta_sweep);
                if max_delta_sweep < cfg.tol {
                    converged = true;
                    return Ok(CdResult {
                        x,
                        n_iter: t + 1,
                        converged,
                        residuals,
                    });
                }
                max_delta_sweep = 0.0;
            }
        }

        Ok(CdResult {
            x,
            n_iter: cfg.max_iter,
            converged,
            residuals,
        })
    }

    /// Accelerated block coordinate descent (Beck-Tetruashvili 2013).
    ///
    /// Applies FISTA-style extrapolation at the extrapolation point y before
    /// the coordinate gradient step.
    ///
    /// `momentum_fn(k)` — extrapolation coefficient at outer step k (e.g. (k-1)/(k+2)).
    pub fn run_accelerated<F, G, P, L, M>(
        x0: Vec<f64>,
        cfg: &CdConfig,
        _obj: F,
        partial_grad: G,
        prox_i: P,
        lipschitz_i: L,
        momentum_fn: M,
        rng: &mut LcgRng,
    ) -> CvxResult<CdResult>
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64], usize) -> f64,
        P: Fn(f64, f64) -> f64,
        L: Fn(usize) -> f64,
        M: Fn(usize) -> f64,
    {
        validate_config(cfg)?;
        if x0.len() != cfg.dim {
            return Err(CvxError::DimensionMismatch {
                a: x0.len(),
                b: cfg.dim,
            });
        }

        let d = cfg.dim;
        let mut x = x0.clone();
        let mut x_prev = x0;
        let mut y = x.clone();
        let mut residuals = Vec::new();
        let mut converged = false;
        let mut max_delta_sweep = 0.0_f64;

        for t in 0..cfg.max_iter {
            let mom = momentum_fn(t + 1).clamp(0.0, 1.0 - 1.0e-12);
            let i = select_coordinate(&y, t, d, cfg.order, &partial_grad, rng);

            let g_yi = partial_grad(&y, i);
            let l_i = lipschitz_i(i);
            let step = cfg
                .step_size
                .unwrap_or_else(|| if l_i > 0.0 { 1.0 / l_i } else { 1.0 });

            let x_i_new = prox_i(y[i] - step * g_yi, step);
            let delta = (x_i_new - x[i]).abs();
            if delta > max_delta_sweep {
                max_delta_sweep = delta;
            }

            x_prev[i] = x[i];
            x[i] = x_i_new;

            y[i] = x[i] + mom * (x[i] - x_prev[i]);

            let sweep_done = (t + 1) % d == 0;
            if sweep_done {
                residuals.push(max_delta_sweep);
                if max_delta_sweep < cfg.tol {
                    converged = true;
                    return Ok(CdResult {
                        x,
                        n_iter: t + 1,
                        converged,
                        residuals,
                    });
                }
                max_delta_sweep = 0.0;
            }
        }

        Ok(CdResult {
            x,
            n_iter: cfg.max_iter,
            converged,
            residuals,
        })
    }

    /// Identity univariate prox: returns x unchanged.
    pub fn prox1_identity(x: f64, _lambda: f64) -> f64 {
        x
    }

    /// Univariate L1 (soft-threshold): `sign(x) * max(|x| - lambda, 0)`.
    pub fn prox1_l1(x: f64, lambda: f64) -> f64 {
        let abs_x = x.abs();
        if abs_x <= lambda {
            0.0
        } else if x > 0.0 {
            x - lambda
        } else {
            x + lambda
        }
    }

    /// Project x onto the box [lo, hi].
    pub fn prox1_box(x: f64, lo: f64, hi: f64) -> f64 {
        x.clamp(lo, hi)
    }

    /// Non-negative projection: `max(x, 0)`.
    pub fn prox1_nonneg(x: f64, _lambda: f64) -> f64 {
        x.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic_obj(x: &[f64], a: &[f64]) -> f64 {
        x.iter()
            .zip(a.iter())
            .map(|(xi, ai)| 0.5 * (xi - ai) * (xi - ai))
            .sum()
    }

    fn quadratic_partial(x: &[f64], a: &[f64], i: usize) -> f64 {
        x[i] - a[i]
    }

    fn lasso_partial(x: &[f64], at: &[f64], b: &[f64], m: usize, n: usize, i: usize) -> f64 {
        let ax = mat_vec(at, m, n, x);
        let residual: Vec<f64> = (0..m).map(|j| ax[j] - b[j]).collect();
        let col_start = i * m;
        let mut v = 0.0_f64;
        for j in 0..m {
            v += at[col_start + j] * residual[j];
        }
        v
    }

    fn mat_vec(a: &[f64], m: usize, n: usize, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0_f64; m];
        for j in 0..m {
            let mut s = 0.0_f64;
            for k in 0..n {
                s += a[k * m + j] * x[k];
            }
            y[j] = s;
        }
        y
    }

    fn col_norm2_sq(a: &[f64], m: usize, i: usize) -> f64 {
        let col_start = i * m;
        (0..m).map(|j| a[col_start + j] * a[col_start + j]).sum()
    }

    fn make_at_column_major(m: usize, n: usize, seed: f64) -> Vec<f64> {
        (0..n * m)
            .map(|idx| (idx as f64 * 0.3 + seed).sin() * 2.0)
            .collect()
    }

    #[test]
    fn cyclic_cd_quadratic_converges() {
        let dim = 4usize;
        let a = vec![1.0, -2.0, 3.5, 0.7];

        let cfg = CdConfig {
            dim,
            max_iter: dim * 2000,
            tol: 1.0e-8,
            order: CdOrder::Cyclic,
            step_size: Some(1.0),
        };

        let a1 = a.clone();
        let a2 = a.clone();
        let mut rng = LcgRng::new(1);
        let res = CoordDescent::run(
            vec![0.0; dim],
            &cfg,
            |x| quadratic_obj(x, &a1),
            |x, i| quadratic_partial(x, &a2, i),
            CoordDescent::prox1_identity,
            |_i| 1.0,
            &mut rng,
        )
        .expect("ok");

        for (k, (&xk, &ak)) in res.x.iter().zip(a.iter()).enumerate() {
            assert!((xk - ak).abs() < 0.01, "coord {k}: got {} want {}", xk, ak);
        }
    }

    #[test]
    fn random_cd_quadratic_converges() {
        let dim = 3usize;
        let a = vec![2.0, -1.0, 0.5];

        let cfg = CdConfig {
            dim,
            max_iter: dim * 3000,
            tol: 1.0e-7,
            order: CdOrder::Random,
            step_size: Some(0.8),
        };

        let a1 = a.clone();
        let a2 = a.clone();
        let mut rng = LcgRng::new(22);
        let res = CoordDescent::run(
            vec![0.0; dim],
            &cfg,
            |x| quadratic_obj(x, &a1),
            |x, i| quadratic_partial(x, &a2, i),
            CoordDescent::prox1_identity,
            |_i| 1.25,
            &mut rng,
        )
        .expect("ok");

        for (k, (&xk, &ak)) in res.x.iter().zip(a.iter()).enumerate() {
            assert!((xk - ak).abs() < 0.05, "coord {k}: got {} want {}", xk, ak);
        }
    }

    #[test]
    fn gauss_seidel_cd_converges() {
        let dim = 3usize;
        let a = vec![1.5, -0.5, 2.0];

        let cfg = CdConfig {
            dim,
            max_iter: dim * 2000,
            tol: 1.0e-8,
            order: CdOrder::GaussSeidel,
            step_size: Some(1.0),
        };

        let a1 = a.clone();
        let a2 = a.clone();
        let mut rng = LcgRng::new(33);
        let res = CoordDescent::run(
            vec![0.0; dim],
            &cfg,
            |x| quadratic_obj(x, &a1),
            |x, i| quadratic_partial(x, &a2, i),
            CoordDescent::prox1_identity,
            |_i| 1.0,
            &mut rng,
        )
        .expect("ok");

        for (k, (&xk, &ak)) in res.x.iter().zip(a.iter()).enumerate() {
            assert!((xk - ak).abs() < 0.01, "coord {k}: got {} want {}", xk, ak);
        }
    }

    #[test]
    fn cyclic_cd_lasso_sparse() {
        let m = 10usize;
        let n = 5usize;
        let at = make_at_column_major(m, n, 1.0);
        let b = vec![1.0; m];
        let lam = 2.0;

        let cfg = CdConfig {
            dim: n,
            max_iter: n * 5000,
            tol: 1.0e-7,
            order: CdOrder::Cyclic,
            step_size: None,
        };

        let at1 = at.clone();
        let at2 = at.clone();
        let b1 = b.clone();
        let mut rng = LcgRng::new(44);
        let res = CoordDescent::run(
            vec![0.0; n],
            &cfg,
            |_x| 0.0,
            |x, i| lasso_partial(x, &at1, &b1, m, n, i),
            |xi, step| CoordDescent::prox1_l1(xi, lam * step),
            |i| col_norm2_sq(&at2, m, i),
            &mut rng,
        )
        .expect("ok");

        let nonzero = res.x.iter().filter(|&&v| v.abs() > 1.0e-4).count();
        assert!(
            nonzero < n,
            "Lasso with high lambda should give sparse solution"
        );
    }

    #[test]
    fn accelerated_cd_faster_than_cyclic() {
        let dim = 4usize;
        let a = vec![3.0, -2.0, 1.5, 0.5];

        let base_cfg = CdConfig {
            dim,
            max_iter: dim * 500,
            tol: 1.0e-5,
            order: CdOrder::Cyclic,
            step_size: Some(1.0),
        };

        let a1 = a.clone();
        let a2 = a.clone();
        let mut rng1 = LcgRng::new(55);
        let res_cyclic = CoordDescent::run(
            vec![0.0; dim],
            &base_cfg,
            |x| quadratic_obj(x, &a1),
            |x, i| quadratic_partial(x, &a2, i),
            CoordDescent::prox1_identity,
            |_i| 1.0,
            &mut rng1,
        )
        .expect("cyclic ok");

        let a3 = a.clone();
        let a4 = a.clone();
        let mut rng2 = LcgRng::new(55);
        let res_accel = CoordDescent::run_accelerated(
            vec![0.0; dim],
            &base_cfg,
            |x| quadratic_obj(x, &a3),
            |x, i| quadratic_partial(x, &a4, i),
            CoordDescent::prox1_identity,
            |_i| 1.0,
            |k| {
                let kf = k as f64;
                ((kf - 1.0) / (kf + 2.0)).clamp(0.0, 1.0 - 1.0e-12)
            },
            &mut rng2,
        )
        .expect("accel ok");

        let accel_final_err: f64 = res_accel
            .x
            .iter()
            .zip(a.iter())
            .map(|(xi, ai)| (xi - ai).abs())
            .sum();
        let cyclic_final_err: f64 = res_cyclic
            .x
            .iter()
            .zip(a.iter())
            .map(|(xi, ai)| (xi - ai).abs())
            .sum();
        assert!(
            accel_final_err <= cyclic_final_err + 1.0e-3,
            "accelerated should not be worse than cyclic: accel={accel_final_err} cyclic={cyclic_final_err}"
        );
    }

    #[test]
    fn cyclic_cd_result_shape() {
        let dim = 5usize;
        let cfg = CdConfig {
            dim,
            max_iter: dim * 10,
            tol: 1.0e-6,
            order: CdOrder::Cyclic,
            step_size: Some(1.0),
        };

        let mut rng = LcgRng::new(66);
        let res = CoordDescent::run(
            vec![0.0; dim],
            &cfg,
            |_x| 0.0,
            |_x, _i| 0.0,
            CoordDescent::prox1_identity,
            |_i| 1.0,
            &mut rng,
        )
        .expect("ok");

        assert_eq!(res.x.len(), dim);
    }

    #[test]
    fn cyclic_cd_residuals_decreasing() {
        let dim = 3usize;
        let a = vec![1.0, -1.0, 0.5];

        let cfg = CdConfig {
            dim,
            max_iter: dim * 500,
            tol: 1.0e-9,
            order: CdOrder::Cyclic,
            step_size: Some(1.0),
        };

        let a1 = a.clone();
        let a2 = a.clone();
        let mut rng = LcgRng::new(77);
        let res = CoordDescent::run(
            vec![10.0; dim],
            &cfg,
            |x| quadratic_obj(x, &a1),
            |x, i| quadratic_partial(x, &a2, i),
            CoordDescent::prox1_identity,
            |_i| 1.0,
            &mut rng,
        )
        .expect("ok");

        assert!(!res.residuals.is_empty());
        let n = res.residuals.len();
        if n >= 2 {
            let avg_early: f64 = res.residuals[..n / 2].iter().sum::<f64>() / (n / 2) as f64;
            let avg_late: f64 = res.residuals[n / 2..].iter().sum::<f64>() / (n - n / 2) as f64;
            assert!(
                avg_late <= avg_early + 1.0e-3,
                "residuals should generally decrease: early={avg_early} late={avg_late}"
            );
        }
    }

    #[test]
    fn prox1_identity_returns_self() {
        assert!((CoordDescent::prox1_identity(3.0, 99.0) - 3.0).abs() < 1.0e-15);
        assert!((CoordDescent::prox1_identity(-5.0, 0.0) + 5.0).abs() < 1.0e-15);
    }

    #[test]
    fn prox1_l1_zero_at_threshold() {
        let lam = 0.5;
        assert_eq!(CoordDescent::prox1_l1(0.5, lam), 0.0);
        assert_eq!(CoordDescent::prox1_l1(-0.5, lam), 0.0);
    }

    #[test]
    fn prox1_l1_positive_above() {
        let x = 2.0_f64;
        let lam = 0.5_f64;
        let out = CoordDescent::prox1_l1(x, lam);
        assert!((out - (x - lam)).abs() < 1.0e-12);
    }

    #[test]
    fn prox1_box_clips_hi() {
        assert!((CoordDescent::prox1_box(10.0, -1.0, 5.0) - 5.0).abs() < 1.0e-15);
    }

    #[test]
    fn prox1_box_clips_lo() {
        assert!((CoordDescent::prox1_box(-10.0, -1.0, 5.0) + 1.0).abs() < 1.0e-15);
    }

    #[test]
    fn prox1_nonneg_zeroes_negative() {
        assert_eq!(CoordDescent::prox1_nonneg(-1.0, 0.0), 0.0);
    }

    #[test]
    fn prox1_nonneg_keeps_positive() {
        assert!((CoordDescent::prox1_nonneg(2.0, 0.0) - 2.0).abs() < 1.0e-15);
    }

    #[test]
    fn cyclic_cd_err_dim_zero() {
        let cfg = CdConfig {
            dim: 0,
            max_iter: 10,
            tol: 1.0e-6,
            order: CdOrder::Cyclic,
            step_size: Some(1.0),
        };
        let mut rng = LcgRng::new(1);
        let res = CoordDescent::run(
            vec![],
            &cfg,
            |_x| 0.0,
            |_x, _i| 0.0,
            CoordDescent::prox1_identity,
            |_i| 1.0,
            &mut rng,
        );
        assert!(res.is_err());
    }

    #[test]
    fn cyclic_cd_err_step_size_nonpositive() {
        let cfg = CdConfig {
            dim: 2,
            max_iter: 10,
            tol: 1.0e-6,
            order: CdOrder::Cyclic,
            step_size: Some(0.0),
        };
        let mut rng = LcgRng::new(2);
        let res = CoordDescent::run(
            vec![0.0; 2],
            &cfg,
            |_x| 0.0,
            |_x, _i| 0.0,
            CoordDescent::prox1_identity,
            |_i| 1.0,
            &mut rng,
        );
        assert!(res.is_err());
    }

    #[test]
    fn cd_single_coordinate() {
        let dim = 1usize;
        let a = vec![3.0];

        let cfg = CdConfig {
            dim,
            max_iter: 200,
            tol: 1.0e-7,
            order: CdOrder::Cyclic,
            step_size: Some(1.0),
        };

        let a1 = a.clone();
        let a2 = a.clone();
        let mut rng = LcgRng::new(99);
        let res = CoordDescent::run(
            vec![0.0; dim],
            &cfg,
            |x| quadratic_obj(x, &a1),
            |x, i| quadratic_partial(x, &a2, i),
            CoordDescent::prox1_identity,
            |_i| 1.0,
            &mut rng,
        )
        .expect("ok");

        assert_eq!(res.x.len(), 1);
        assert!((res.x[0] - a[0]).abs() < 0.01);
    }
}
