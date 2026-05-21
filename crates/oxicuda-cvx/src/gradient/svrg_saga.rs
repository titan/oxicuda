//! SVRG (Johnson-Zhang 2013) and SAGA (Defazio-Bach-Lacoste-Julien 2014)
//! variance-reduced stochastic gradient methods with proximal step.

use crate::error::{CvxError, CvxResult};
use crate::handle::LcgRng;
use crate::linalg::matvec::norm2;

/// Configuration for SVRG (and SVRG+ with momentum).
#[derive(Debug, Clone)]
pub struct SvrgConfig {
    pub n_samples: usize,
    pub dim: usize,
    pub step_size: f64,
    pub epoch_length: usize,
    pub max_epochs: usize,
    pub tol: f64,
    pub momentum: f64,
}

/// Configuration for SAGA.
#[derive(Debug, Clone)]
pub struct SagaConfig {
    pub n_samples: usize,
    pub dim: usize,
    pub step_size: f64,
    pub max_iter: usize,
    pub tol: f64,
}

/// Result returned by both SVRG and SAGA.
#[derive(Debug, Clone)]
pub struct VrsgResult {
    pub x: Vec<f64>,
    pub history: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
}

fn validate_svrg(cfg: &SvrgConfig) -> CvxResult<()> {
    if cfg.dim == 0 {
        return Err(CvxError::EmptyInput);
    }
    if cfg.n_samples == 0 {
        return Err(CvxError::InvalidParameter("n_samples must be >= 1".into()));
    }
    if cfg.step_size <= 0.0 || !cfg.step_size.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "step_size must be > 0, got {}",
            cfg.step_size
        )));
    }
    if cfg.epoch_length == 0 {
        return Err(CvxError::InvalidParameter(
            "epoch_length must be >= 1".into(),
        ));
    }
    if cfg.max_epochs == 0 {
        return Err(CvxError::InvalidParameter("max_epochs must be >= 1".into()));
    }
    Ok(())
}

fn validate_saga(cfg: &SagaConfig) -> CvxResult<()> {
    if cfg.dim == 0 {
        return Err(CvxError::EmptyInput);
    }
    if cfg.n_samples == 0 {
        return Err(CvxError::InvalidParameter("n_samples must be >= 1".into()));
    }
    if cfg.step_size <= 0.0 || !cfg.step_size.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "step_size must be > 0, got {}",
            cfg.step_size
        )));
    }
    if cfg.max_iter == 0 {
        return Err(CvxError::InvalidParameter("max_iter must be >= 1".into()));
    }
    Ok(())
}

/// SVRG: Johnson-Zhang 2013 NIPS algorithm.
pub struct Svrg;

impl Svrg {
    /// Run SVRG.
    ///
    /// `grad_i(x, i)` — stochastic gradient for sample i.
    /// `full_grad(x)` — full batch gradient (used once per epoch for snapshot).
    /// `prox(x, lambda)` — proximal operator for regulariser g.
    pub fn run<F, G, P>(
        x0: Vec<f64>,
        cfg: &SvrgConfig,
        grad_i: F,
        full_grad: G,
        prox: P,
        rng: &mut LcgRng,
    ) -> CvxResult<VrsgResult>
    where
        F: Fn(&[f64], usize) -> Vec<f64>,
        G: Fn(&[f64]) -> Vec<f64>,
        P: Fn(&[f64], f64) -> Vec<f64>,
    {
        validate_svrg(cfg)?;
        if x0.len() != cfg.dim {
            return Err(CvxError::DimensionMismatch {
                a: x0.len(),
                b: cfg.dim,
            });
        }

        let eta = cfg.step_size;
        let d = cfg.dim;
        let mut x = x0;
        let mut history = Vec::with_capacity(cfg.max_epochs);
        let mut converged = false;
        let mut total_inner = 0usize;

        for _epoch in 0..cfg.max_epochs {
            let x_snap = x.clone();
            let mu = full_grad(&x_snap);

            let mut x_inner = x.clone();

            for _t in 0..cfg.epoch_length {
                let idx = rng.next_usize(cfg.n_samples);
                let g_cur = grad_i(&x_inner, idx);
                let g_snap = grad_i(&x_snap, idx);

                let mut v = vec![0.0_f64; d];
                for k in 0..d {
                    v[k] = g_cur[k] - g_snap[k] + mu[k];
                }

                let x_candidate: Vec<f64> = (0..d).map(|k| x_inner[k] - eta * v[k]).collect();
                x_inner = prox(&x_candidate, eta);
                total_inner += 1;
            }

            let diff_norm = {
                let mut sq = 0.0_f64;
                for k in 0..d {
                    let diff = x_inner[k] - x_snap[k];
                    sq += diff * diff;
                }
                sq.sqrt()
            };

            let full_g = full_grad(&x_inner);
            let grad_norm = norm2(&full_g);
            history.push(grad_norm);

            if cfg.momentum > 0.0 {
                for k in 0..d {
                    x[k] = x_inner[k] + cfg.momentum * (x_inner[k] - x_snap[k]);
                }
            } else {
                x = x_inner;
            }

            if diff_norm < cfg.tol {
                converged = true;
                break;
            }
        }

        Ok(VrsgResult {
            x,
            history,
            n_iter: total_inner,
            converged,
        })
    }
}

/// SAGA: Defazio-Bach-Lacoste-Julien 2014 NIPS algorithm.
pub struct Saga;

impl Saga {
    /// Run SAGA.
    ///
    /// `grad_i(x, i)` — stochastic gradient for sample i.
    /// `prox(x, lambda)` — proximal operator for regulariser g.
    pub fn run<F, P>(
        x0: Vec<f64>,
        cfg: &SagaConfig,
        grad_i: F,
        prox: P,
        rng: &mut LcgRng,
    ) -> CvxResult<VrsgResult>
    where
        F: Fn(&[f64], usize) -> Vec<f64>,
        P: Fn(&[f64], f64) -> Vec<f64>,
    {
        validate_saga(cfg)?;
        if x0.len() != cfg.dim {
            return Err(CvxError::DimensionMismatch {
                a: x0.len(),
                b: cfg.dim,
            });
        }

        let eta = cfg.step_size;
        let n = cfg.n_samples;
        let d = cfg.dim;

        let mut table = Self::init_table(&x0, n, d, &grad_i);

        let mut mean_grad = vec![0.0_f64; d];
        for i in 0..n {
            let row = i * d;
            for k in 0..d {
                mean_grad[k] += table[row + k];
            }
        }
        let n_inv = 1.0 / n as f64;
        for v in mean_grad.iter_mut() {
            *v *= n_inv;
        }

        let mut x = x0;
        let mut history = Vec::new();
        let mut converged = false;

        for t in 0..cfg.max_iter {
            let j = rng.next_usize(n);
            let g_j = grad_i(&x, j);

            let row_j = j * d;
            let mut v = vec![0.0_f64; d];
            for k in 0..d {
                let old_alpha_jk = table[row_j + k];
                v[k] = g_j[k] - old_alpha_jk + mean_grad[k];
                mean_grad[k] += (g_j[k] - old_alpha_jk) * n_inv;
                table[row_j + k] = g_j[k];
            }

            let x_candidate: Vec<f64> = (0..d).map(|k| x[k] - eta * v[k]).collect();
            let x_new = prox(&x_candidate, eta);

            let diff_norm = {
                let mut sq = 0.0_f64;
                for k in 0..d {
                    let diff = x_new[k] - x[k];
                    sq += diff * diff;
                }
                sq.sqrt()
            };

            x = x_new;

            if (t + 1) % n == 0 {
                let grad_norm = norm2(&mean_grad);
                history.push(grad_norm);
            }

            if diff_norm < cfg.tol {
                converged = true;
                break;
            }
        }

        Ok(VrsgResult {
            x,
            history,
            n_iter: cfg.max_iter,
            converged,
        })
    }

    /// Initialize gradient table: table[i*dim..i*dim+dim] = grad_i(x0, i).
    pub fn init_table<F>(x0: &[f64], n: usize, dim: usize, grad_i: F) -> Vec<f64>
    where
        F: Fn(&[f64], usize) -> Vec<f64>,
    {
        let mut table = vec![0.0_f64; n * dim];
        for i in 0..n {
            let g = grad_i(x0, i);
            let row = i * dim;
            let copy_len = dim.min(g.len());
            table[row..row + copy_len].copy_from_slice(&g[..copy_len]);
        }
        table
    }
}

/// No-op proximal operator (identity): returns x unchanged.
pub fn prox_identity(x: &[f64], _lambda: f64) -> Vec<f64> {
    x.to_vec()
}

/// L1 proximal operator: soft-thresholding `sign(x_i) * max(|x_i| - lambda, 0)`.
pub fn prox_l1(x: &[f64], lambda: f64) -> Vec<f64> {
    x.iter()
        .map(|&xi| {
            let abs_xi = xi.abs();
            if abs_xi <= lambda {
                0.0
            } else if xi > 0.0 {
                xi - lambda
            } else {
                xi + lambda
            }
        })
        .collect()
}

/// L2-squared proximal operator (Tikhonov): `x_i / (1 + 2*lambda)`.
pub fn prox_l2_sq(x: &[f64], lambda: f64) -> Vec<f64> {
    let denom = 1.0 + 2.0 * lambda;
    x.iter().map(|&xi| xi / denom).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic_grad_i(x: &[f64], a_flat: &[f64], dim: usize, i: usize) -> Vec<f64> {
        let row = i * dim;
        (0..dim).map(|k| x[k] - a_flat[row + k]).collect()
    }

    fn quadratic_full_grad(x: &[f64], a_flat: &[f64], n: usize, dim: usize) -> Vec<f64> {
        let mut grad = vec![0.0_f64; dim];
        for i in 0..n {
            let row = i * dim;
            for (k, gk) in grad.iter_mut().enumerate().take(dim) {
                *gk += x[k] - a_flat[row + k];
            }
        }
        let n_inv = 1.0 / n as f64;
        for gk in grad.iter_mut() {
            *gk *= n_inv;
        }
        grad
    }

    fn make_a_flat(n: usize, dim: usize, seed: f64) -> Vec<f64> {
        (0..n * dim)
            .map(|idx| (idx as f64 * 0.1 + seed).sin())
            .collect()
    }

    fn mean_a(a_flat: &[f64], n: usize, dim: usize) -> Vec<f64> {
        let mut m = vec![0.0_f64; dim];
        for i in 0..n {
            let row = i * dim;
            for k in 0..dim {
                m[k] += a_flat[row + k];
            }
        }
        let n_inv = 1.0 / n as f64;
        for mk in m.iter_mut().take(dim) {
            *mk *= n_inv;
        }
        m
    }

    #[test]
    fn svrg_converges_on_quadratic() {
        let n = 20usize;
        let dim = 4usize;
        let a_flat = make_a_flat(n, dim, 1.0);
        let target = mean_a(&a_flat, n, dim);

        let cfg = SvrgConfig {
            n_samples: n,
            dim,
            step_size: 0.05,
            epoch_length: 2 * n,
            max_epochs: 200,
            tol: 1.0e-6,
            momentum: 0.0,
        };

        let af1 = a_flat.clone();
        let af2 = a_flat.clone();
        let mut rng = LcgRng::new(42);
        let result = Svrg::run(
            vec![0.0; dim],
            &cfg,
            |x, i| quadratic_grad_i(x, &af1, dim, i),
            |x| quadratic_full_grad(x, &af2, n, dim),
            prox_identity,
            &mut rng,
        )
        .expect("svrg ok");

        for (k, (&xk, &tk)) in result.x.iter().zip(target.iter()).enumerate() {
            assert!(
                (xk - tk).abs() < 0.1,
                "coord {} diff {}",
                k,
                (xk - tk).abs()
            );
        }
    }

    #[test]
    fn svrg_history_length() {
        let n = 5usize;
        let dim = 2usize;
        let a_flat = make_a_flat(n, dim, 0.5);
        let max_epochs = 7usize;

        let cfg = SvrgConfig {
            n_samples: n,
            dim,
            step_size: 0.05,
            epoch_length: n,
            max_epochs,
            tol: 1.0e-12,
            momentum: 0.0,
        };

        let af1 = a_flat.clone();
        let af2 = a_flat.clone();
        let mut rng = LcgRng::new(7);
        let result = Svrg::run(
            vec![0.0; dim],
            &cfg,
            |x, i| quadratic_grad_i(x, &af1, dim, i),
            |x| quadratic_full_grad(x, &af2, n, dim),
            prox_identity,
            &mut rng,
        )
        .expect("ok");

        assert!(result.history.len() <= max_epochs);
        assert!(!result.history.is_empty());
    }

    #[test]
    fn svrg_with_l1_prox_sparse() {
        let n = 10usize;
        let dim = 4usize;
        let a_flat = vec![0.0; n * dim];

        let cfg = SvrgConfig {
            n_samples: n,
            dim,
            step_size: 0.05,
            epoch_length: 2 * n,
            max_epochs: 100,
            tol: 1.0e-6,
            momentum: 0.0,
        };

        let mut rng = LcgRng::new(13);
        let x0: Vec<f64> = (0..dim).map(|k| 0.3 * (k as f64 + 1.0)).collect();
        let af1 = a_flat.clone();
        let af2 = a_flat.clone();
        let result = Svrg::run(
            x0,
            &cfg,
            |x, i| quadratic_grad_i(x, &af1, dim, i),
            |x| quadratic_full_grad(x, &af2, n, dim),
            |x, lam| prox_l1(x, lam * 5.0),
            &mut rng,
        )
        .expect("ok");

        let nonzero = result.x.iter().filter(|&&v| v.abs() > 1.0e-6).count();
        assert!(nonzero < dim, "L1 prox should produce sparsity");
    }

    #[test]
    fn svrg_with_l2_prox_shrinks() {
        let n = 10usize;
        let dim = 3usize;
        let a_flat = vec![0.0; n * dim];

        let cfg = SvrgConfig {
            n_samples: n,
            dim,
            step_size: 0.05,
            epoch_length: 2 * n,
            max_epochs: 80,
            tol: 1.0e-8,
            momentum: 0.0,
        };

        let mut rng = LcgRng::new(17);
        let x0 = vec![5.0, -3.0, 2.0];
        let af1 = a_flat.clone();
        let af2 = a_flat.clone();
        let result = Svrg::run(
            x0,
            &cfg,
            |x, i| quadratic_grad_i(x, &af1, dim, i),
            |x| quadratic_full_grad(x, &af2, n, dim),
            prox_l2_sq,
            &mut rng,
        )
        .expect("ok");

        let nrm: f64 = result.x.iter().map(|&v| v * v).sum::<f64>().sqrt();
        assert!(nrm < 5.0, "L2sq prox should shrink solution, got nrm={nrm}");
    }

    #[test]
    fn saga_converges_on_quadratic() {
        let n = 20usize;
        let dim = 3usize;
        let a_flat = make_a_flat(n, dim, 2.0);
        let target = mean_a(&a_flat, n, dim);

        let cfg = SagaConfig {
            n_samples: n,
            dim,
            step_size: 0.02,
            max_iter: 5000,
            tol: 1.0e-6,
        };

        let mut rng = LcgRng::new(99);
        let af = a_flat.clone();
        let result = Saga::run(
            vec![0.0; dim],
            &cfg,
            |x, i| quadratic_grad_i(x, &af, dim, i),
            prox_identity,
            &mut rng,
        )
        .expect("saga ok");

        for (k, (&xk, &tk)) in result.x.iter().zip(target.iter()).enumerate() {
            assert!(
                (xk - tk).abs() < 0.15,
                "coord {} diff {}",
                k,
                (xk - tk).abs()
            );
        }
    }

    #[test]
    fn saga_init_table_shape() {
        let n = 8usize;
        let dim = 5usize;
        let x0 = vec![1.0; dim];
        let table = Saga::init_table(&x0, n, dim, |x, _i| x.to_vec());
        assert_eq!(table.len(), n * dim);
    }

    #[test]
    fn saga_init_table_values() {
        let n = 3usize;
        let dim = 2usize;
        let a_flat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x0 = vec![0.0; dim];
        let af = a_flat.clone();
        let table = Saga::init_table(&x0, n, dim, |x, i| {
            let row = i * dim;
            (0..dim).map(|k| x[k] - af[row + k]).collect()
        });
        for i in 0..n {
            let row = i * dim;
            for k in 0..dim {
                let expected = 0.0 - a_flat[i * dim + k];
                assert!(
                    (table[row + k] - expected).abs() < 1.0e-12,
                    "table[{i},{k}] expected {expected} got {}",
                    table[row + k]
                );
            }
        }
    }

    #[test]
    fn saga_history_recorded() {
        let n = 5usize;
        let dim = 2usize;
        let a_flat = make_a_flat(n, dim, 3.0);

        let cfg = SagaConfig {
            n_samples: n,
            dim,
            step_size: 0.02,
            max_iter: n * 10,
            tol: 1.0e-12,
        };

        let mut rng = LcgRng::new(55);
        let af = a_flat.clone();
        let result = Saga::run(
            vec![0.0; dim],
            &cfg,
            |x, i| quadratic_grad_i(x, &af, dim, i),
            prox_identity,
            &mut rng,
        )
        .expect("ok");

        assert!(!result.history.is_empty(), "history should have entries");
    }

    #[test]
    fn prox_identity_returns_x() {
        let x = vec![1.0, -2.0, 3.5];
        let out = prox_identity(&x, 99.0);
        for (a, b) in x.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1.0e-15);
        }
    }

    #[test]
    fn prox_l1_zeros_below_threshold() {
        let x = vec![0.4, -0.3, 0.1];
        let out = prox_l1(&x, 0.5);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn prox_l1_shrinks_above() {
        let x = vec![2.0, -3.0, 1.5];
        let lam = 0.5;
        let out = prox_l1(&x, lam);
        assert!((out[0] - (2.0 - lam)).abs() < 1.0e-12);
        assert!((out[1] - (-3.0 + lam)).abs() < 1.0e-12);
        assert!((out[2] - (1.5 - lam)).abs() < 1.0e-12);
    }

    #[test]
    fn prox_l2_sq_shrinks_uniformly() {
        let x = vec![4.0, -6.0, 2.0];
        let out = prox_l2_sq(&x, 0.5);
        for (xi, oi) in x.iter().zip(out.iter()) {
            assert!((oi - xi / 2.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn svrg_err_empty_x0() {
        let cfg = SvrgConfig {
            n_samples: 5,
            dim: 0,
            step_size: 0.1,
            epoch_length: 10,
            max_epochs: 5,
            tol: 1.0e-6,
            momentum: 0.0,
        };
        let mut rng = LcgRng::new(1);
        let res = Svrg::run(
            vec![],
            &cfg,
            |x, _i| x.to_vec(),
            |x| x.to_vec(),
            prox_identity,
            &mut rng,
        );
        assert!(res.is_err());
    }

    #[test]
    fn svrg_err_zero_step() {
        let cfg = SvrgConfig {
            n_samples: 5,
            dim: 2,
            step_size: 0.0,
            epoch_length: 5,
            max_epochs: 3,
            tol: 1.0e-6,
            momentum: 0.0,
        };
        let mut rng = LcgRng::new(2);
        let res = Svrg::run(
            vec![0.0; 2],
            &cfg,
            |x, _i| x.to_vec(),
            |x| x.to_vec(),
            prox_identity,
            &mut rng,
        );
        assert!(res.is_err());
    }

    #[test]
    fn saga_err_n_samples_zero() {
        let cfg = SagaConfig {
            n_samples: 0,
            dim: 2,
            step_size: 0.1,
            max_iter: 10,
            tol: 1.0e-6,
        };
        let mut rng = LcgRng::new(3);
        let res = Saga::run(
            vec![0.0; 2],
            &cfg,
            |x, _i| x.to_vec(),
            prox_identity,
            &mut rng,
        );
        assert!(res.is_err());
    }

    #[test]
    fn svrg_momentum_path() {
        let n = 10usize;
        let dim = 2usize;
        let a_flat = make_a_flat(n, dim, 4.0);

        let cfg = SvrgConfig {
            n_samples: n,
            dim,
            step_size: 0.04,
            epoch_length: 2 * n,
            max_epochs: 100,
            tol: 1.0e-5,
            momentum: 0.5,
        };

        let mut rng = LcgRng::new(77);
        let af1 = a_flat.clone();
        let af2 = a_flat.clone();
        let result = Svrg::run(
            vec![3.0; dim],
            &cfg,
            |x, i| quadratic_grad_i(x, &af1, dim, i),
            |x| quadratic_full_grad(x, &af2, n, dim),
            prox_identity,
            &mut rng,
        );
        assert!(result.is_ok(), "momentum SVRG should run without error");
    }

    #[test]
    fn saga_with_l1_prox_converges() {
        let n = 15usize;
        let dim = 4usize;
        let a_flat = vec![0.0; n * dim];

        let cfg = SagaConfig {
            n_samples: n,
            dim,
            step_size: 0.01,
            max_iter: 8000,
            tol: 1.0e-5,
        };

        let mut rng = LcgRng::new(88);
        let x0 = vec![2.0, -1.5, 0.8, -0.3];
        let af = a_flat.clone();
        let result = Saga::run(
            x0,
            &cfg,
            |x, i| quadratic_grad_i(x, &af, dim, i),
            |x, lam| prox_l1(x, lam * 3.0),
            &mut rng,
        )
        .expect("ok");

        let nrm: f64 = result.x.iter().map(|&v| v * v).sum::<f64>().sqrt();
        assert!(nrm < 3.0, "SAGA+L1 should converge toward 0, nrm={nrm}");
    }
}
