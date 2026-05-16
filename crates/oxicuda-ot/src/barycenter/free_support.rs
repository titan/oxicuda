#![allow(clippy::needless_range_loop)]
//! Free-support Wasserstein barycenter.
//!
//! Solve
//!
//! ```text
//! min_{Y, b}  Σ_k λ_k W_2²(μ_k, ν(Y, b))
//! ```
//!
//! where `μ_k = (X_k, a_k)` are the input measures and `ν(Y, b)` is the
//! barycenter measure with `n_bary` free support points. The Cuturi-Doucet
//! scheme alternates between
//!
//! 1. fixing `Y` and computing entropic OT plans `T_k` between `(Y, b)` and
//!    each `(X_k, a_k)`, and
//! 2. updating each support point `Y_i` by the λ-weighted barycentric
//!    projection
//!    `Y_i ← Σ_k λ_k Σ_j T_{k,ij} X_{k,j} / (Σ_j T_{k,ij} + δ)`,
//!
//! with a small δ to guard against zero rows. We initialise `Y` as the
//! λ-weighted mean of the inputs and `b` as the uniform `1 / n_bary` weight.

use crate::error::{OtError, OtResult};
use crate::handle::LcgRng;
use crate::sinkhorn::sinkhorn::{SinkhornConfig, sinkhorn};

/// Configuration for the free-support barycenter solver.
#[derive(Debug, Clone)]
pub struct BaryConfig {
    /// Inner Sinkhorn entropic regularisation `ε > 0`.
    pub eps: f32,
    /// Number of outer alternating iterations.
    pub n_outer: usize,
    /// Maximum Sinkhorn iterations per outer step.
    pub n_inner: usize,
    /// Convergence tolerance on inner Sinkhorn marginal residual.
    pub tol: f32,
}

impl Default for BaryConfig {
    fn default() -> Self {
        Self {
            eps: 0.05,
            n_outer: 20,
            n_inner: 100,
            tol: 1e-4,
        }
    }
}

/// Numerical guard against division by tiny `Σ_j T_{k,ij}` row sums.
const ROW_SUM_FLOOR: f32 = 1e-12;

/// Validate inputs.
fn validate(
    measures_x: &[Vec<f32>],
    measures_a: &[Vec<f32>],
    dim: usize,
    n_bary: usize,
    lambdas: &[f32],
    cfg: &BaryConfig,
) -> OtResult<()> {
    if measures_x.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if measures_x.len() != measures_a.len() {
        return Err(OtError::IncompatibleLength {
            a: measures_x.len(),
            b: measures_a.len(),
        });
    }
    if measures_x.len() != lambdas.len() {
        return Err(OtError::IncompatibleLength {
            a: measures_x.len(),
            b: lambdas.len(),
        });
    }
    if dim == 0 {
        return Err(OtError::BadDim { got: dim });
    }
    if n_bary == 0 {
        return Err(OtError::BadCount { got: n_bary });
    }
    if cfg.eps <= 0.0 {
        return Err(OtError::BadEpsilon { eps: cfg.eps });
    }
    let mut lam_sum = 0.0_f32;
    for &lam in lambdas {
        if lam < 0.0 || !lam.is_finite() {
            return Err(OtError::NegativeWeight);
        }
        lam_sum += lam;
    }
    if (lam_sum - 1.0).abs() > 1e-3 {
        return Err(OtError::NotProbability);
    }
    for (xs, ws) in measures_x.iter().zip(measures_a.iter()) {
        if xs.is_empty() || ws.is_empty() {
            return Err(OtError::EmptyInput);
        }
        if !xs.len().is_multiple_of(dim) {
            return Err(OtError::IncompatibleLength {
                a: xs.len(),
                b: dim,
            });
        }
        if xs.len() / dim != ws.len() {
            return Err(OtError::IncompatibleLength {
                a: xs.len() / dim,
                b: ws.len(),
            });
        }
        let mut wsum = 0.0_f32;
        for &w in ws.iter() {
            if w < 0.0 || !w.is_finite() {
                return Err(OtError::NegativeWeight);
            }
            wsum += w;
        }
        if !wsum.is_finite() {
            return Err(OtError::NegativeWeight);
        }
    }
    Ok(())
}

/// Initialise `Y` as the λ-weighted average of the input means.
///
/// For each input measure we compute its weighted centroid `μ_k = Σ_j a_{k,j}
/// X_{k,j} / Σ_j a_{k,j}`, then average those centroids with the λ weights to
/// obtain a single anchor. Each of the `n_bary` support points is then the
/// anchor plus a small radial spread driven by the RNG so that downstream
/// sweeps do not see degenerate inputs.
fn init_support(
    measures_x: &[Vec<f32>],
    measures_a: &[Vec<f32>],
    dim: usize,
    n_bary: usize,
    lambdas: &[f32],
    rng: &mut LcgRng,
) -> Vec<f32> {
    let mut anchor = vec![0.0_f32; dim];
    for (k, xs) in measures_x.iter().enumerate() {
        let ws = &measures_a[k];
        let mut sum_w = 0.0_f32;
        let mut centroid = vec![0.0_f32; dim];
        for (j, &w) in ws.iter().enumerate() {
            sum_w += w;
            for (d, slot) in centroid.iter_mut().enumerate() {
                *slot += w * xs[j * dim + d];
            }
        }
        let inv = if sum_w > ROW_SUM_FLOOR {
            1.0 / sum_w
        } else {
            0.0
        };
        for (d, slot) in anchor.iter_mut().enumerate() {
            *slot += lambdas[k] * centroid[d] * inv;
        }
    }
    let mut y = vec![0.0_f32; n_bary * dim];
    for i in 0..n_bary {
        let off = i * dim;
        for (d, slot) in anchor.iter().enumerate() {
            // Small radial perturbation so the support points are distinct.
            let jitter = 1e-3 * (rng.next_f32() - 0.5);
            y[off + d] = *slot + jitter;
        }
    }
    y
}

/// Build the cost matrix `C_k_ij = ½ ‖Y_i − X_{k,j}‖²` as an `(n_bary × n_k)`
/// row-major slab.
fn build_cost(y: &[f32], xs: &[f32], n_bary: usize, n_k: usize, dim: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; n_bary * n_k];
    for i in 0..n_bary {
        let yi = &y[i * dim..(i + 1) * dim];
        let row_off = i * n_k;
        for j in 0..n_k {
            let xj = &xs[j * dim..(j + 1) * dim];
            let mut sq = 0.0_f32;
            for d in 0..dim {
                let diff = yi[d] - xj[d];
                sq += diff * diff;
            }
            c[row_off + j] = 0.5 * sq;
        }
    }
    c
}

/// Compute the free-support barycenter and return `(support_y, weights_b)`.
pub fn free_support_barycenter(
    measures_x: &[Vec<f32>],
    measures_a: &[Vec<f32>],
    dim: usize,
    n_bary: usize,
    lambdas: &[f32],
    cfg: &BaryConfig,
    rng: &mut LcgRng,
) -> OtResult<(Vec<f32>, Vec<f32>)> {
    validate(measures_x, measures_a, dim, n_bary, lambdas, cfg)?;
    let mut y = init_support(measures_x, measures_a, dim, n_bary, lambdas, rng);
    let b = vec![1.0_f32 / n_bary as f32; n_bary];

    let inner_cfg = SinkhornConfig {
        eps: cfg.eps,
        max_iter: cfg.n_inner,
        tol: cfg.tol,
    };

    let mut new_y = vec![0.0_f32; n_bary * dim];
    let mut row_sum = vec![0.0_f32; n_bary];

    for _ in 0..cfg.n_outer {
        for slot in new_y.iter_mut() {
            *slot = 0.0;
        }
        let lam_total: f32 = lambdas.iter().copied().sum();
        let inv_lam_total = if lam_total > ROW_SUM_FLOOR {
            1.0 / lam_total
        } else {
            1.0
        };
        for (k, xs) in measures_x.iter().enumerate() {
            let ws = &measures_a[k];
            let n_k = ws.len();
            let cost = build_cost(&y, xs, n_bary, n_k, dim);
            // Renormalise input weights so the inner Sinkhorn marginal
            // matches `b` (uniform 1/n_bary).
            let mut renorm_a = vec![0.0_f32; n_k];
            let total: f32 = ws.iter().copied().sum();
            if total <= ROW_SUM_FLOOR {
                continue;
            }
            for (j, slot) in renorm_a.iter_mut().enumerate() {
                *slot = ws[j] / total;
            }
            // The Sinkhorn problem is symmetric: `b` is the row-marginal of
            // the barycenter (n_bary), `renorm_a` is the column-marginal of
            // the input measure.
            let res = sinkhorn(&cost, &b, &renorm_a, n_bary, n_k, &inner_cfg)?;
            // Barycentric projection of T_k onto the input support.
            for (i, rs) in row_sum.iter_mut().enumerate() {
                *rs = 0.0;
                for j in 0..n_k {
                    *rs += res.plan[i * n_k + j];
                }
            }
            for i in 0..n_bary {
                let inv = if row_sum[i] > ROW_SUM_FLOOR {
                    1.0 / row_sum[i]
                } else {
                    0.0
                };
                let off = i * dim;
                for j in 0..n_k {
                    let t_ij = res.plan[i * n_k + j];
                    let xj_off = j * dim;
                    for d in 0..dim {
                        new_y[off + d] += lambdas[k] * t_ij * xs[xj_off + d] * inv;
                    }
                }
            }
        }
        // Apply the λ-normalisation factor (1 / Σ λ_k) — equals 1 when the
        // weights are a probability simplex, but we keep it for safety.
        for slot in new_y.iter_mut() {
            *slot *= inv_lam_total;
        }
        std::mem::swap(&mut y, &mut new_y);
    }

    Ok((y, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn lambda_must_be_simplex() {
        let cfg = BaryConfig::default();
        let mut rng = LcgRng::new(0);
        // λ does not sum to 1.
        let xs = vec![vec![0.0_f32, 0.0], vec![1.0_f32, 1.0]];
        let ws = vec![vec![1.0_f32], vec![1.0_f32]];
        let lambdas = vec![0.7_f32, 0.7];
        let res = free_support_barycenter(&xs, &ws, 2, 2, &lambdas, &cfg, &mut rng);
        assert!(matches!(res, Err(OtError::NotProbability)));
    }

    #[test]
    fn empty_inputs_rejected() {
        let cfg = BaryConfig::default();
        let mut rng = LcgRng::new(0);
        let xs: Vec<Vec<f32>> = vec![];
        let ws: Vec<Vec<f32>> = vec![];
        let lambdas: Vec<f32> = vec![];
        let res = free_support_barycenter(&xs, &ws, 2, 2, &lambdas, &cfg, &mut rng);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn dim_zero_rejected() {
        let cfg = BaryConfig::default();
        let mut rng = LcgRng::new(0);
        let xs = vec![vec![0.0_f32]];
        let ws = vec![vec![1.0_f32]];
        let lambdas = vec![1.0_f32];
        let res = free_support_barycenter(&xs, &ws, 0, 1, &lambdas, &cfg, &mut rng);
        assert!(matches!(res, Err(OtError::BadDim { .. })));
    }

    #[test]
    fn n_bary_zero_rejected() {
        let cfg = BaryConfig::default();
        let mut rng = LcgRng::new(0);
        let xs = vec![vec![0.0_f32, 0.0]];
        let ws = vec![vec![1.0_f32]];
        let lambdas = vec![1.0_f32];
        let res = free_support_barycenter(&xs, &ws, 2, 0, &lambdas, &cfg, &mut rng);
        assert!(matches!(res, Err(OtError::BadCount { .. })));
    }

    #[test]
    fn singleton_self_barycenter_recovers_centroid() {
        // A single measure barycentered with λ = 1: the resulting support
        // should land within the convex hull / vicinity of the input mean.
        let cfg = BaryConfig {
            eps: 0.05,
            n_outer: 30,
            n_inner: 200,
            tol: 1e-5,
        };
        let mut rng = LcgRng::new(7);
        let xs = vec![vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]];
        let ws = vec![vec![0.25_f32; 4]];
        let lambdas = vec![1.0_f32];
        let (y, b) =
            free_support_barycenter(&xs, &ws, 2, 1, &lambdas, &cfg, &mut rng).expect("converges");
        // Single barycenter point ⇒ should sit close to the (0.5, 0.5) mean.
        assert_eq!(y.len(), 2);
        assert!(approx(y[0], 0.5, 0.1), "y[0]={}", y[0]);
        assert!(approx(y[1], 0.5, 0.1), "y[1]={}", y[1]);
        assert!(approx(b[0], 1.0, 1e-6));
    }

    #[test]
    fn weights_are_uniform() {
        let cfg = BaryConfig::default();
        let mut rng = LcgRng::new(11);
        let xs = vec![vec![0.0_f32, 0.0, 1.0, 0.0], vec![5.0_f32, 5.0, 6.0, 5.0]];
        let ws = vec![vec![0.5_f32, 0.5], vec![0.5_f32, 0.5]];
        let lambdas = vec![0.5_f32, 0.5];
        let (_y, b) =
            free_support_barycenter(&xs, &ws, 2, 3, &lambdas, &cfg, &mut rng).expect("converges");
        for &bj in &b {
            assert!(approx(bj, 1.0 / 3.0, 1e-6));
        }
    }
}
