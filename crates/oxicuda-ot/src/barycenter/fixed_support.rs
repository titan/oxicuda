#![allow(clippy::needless_range_loop)]
//! Fixed-support Wasserstein barycenter (Cuturi-Doucet 2014).
//!
//! Given a fixed barycenter support of size `n_bary`, pre-computed cost
//! matrices `C_k ∈ R^{n_bary × n_k}` from the barycenter onto each input
//! support, and input measures `(a_k, λ_k)`, the barycentric weight vector
//! `b ∈ Δ_{n_bary}` is updated by the entropic geometric-mean formula
//!
//! ```text
//! b_new = ∏_k ( K_kᵀ · ( a_k / (K_k · b) ) )^{λ_k}
//! ```
//!
//! where `K_k = exp(− C_k / ε)`. We renormalise after the geometric mean so
//! `Σ_i b_i = 1`. The iteration is a Bregman projection scheme guaranteed to
//! converge to the entropic barycenter.

use crate::error::{OtError, OtResult};

/// Configuration for the fixed-support barycenter.
#[derive(Debug, Clone)]
pub struct FixedBaryConfig {
    /// Entropic regularisation strength `ε > 0`.
    pub eps: f32,
    /// Maximum number of outer iterations.
    pub max_iter: usize,
    /// Convergence tolerance on `‖b_{t+1} − b_t‖_∞`.
    pub tol: f32,
}

impl Default for FixedBaryConfig {
    fn default() -> Self {
        Self {
            eps: 0.05,
            max_iter: 200,
            tol: 1e-5,
        }
    }
}

/// Numerical floor for divisions and zeroth powers.
const FLOOR: f32 = 1e-30;

/// Validate inputs and return the inferred `n_bary` from the cost shapes.
fn validate(
    costs: &[Vec<f32>],
    measures_a: &[Vec<f32>],
    lambdas: &[f32],
    n_bary: usize,
    cfg: &FixedBaryConfig,
) -> OtResult<()> {
    if costs.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if costs.len() != measures_a.len() {
        return Err(OtError::IncompatibleLength {
            a: costs.len(),
            b: measures_a.len(),
        });
    }
    if costs.len() != lambdas.len() {
        return Err(OtError::IncompatibleLength {
            a: costs.len(),
            b: lambdas.len(),
        });
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
    for (k, ck) in costs.iter().enumerate() {
        let n_k = measures_a[k].len();
        if ck.len() != n_bary * n_k {
            return Err(OtError::MarginalMismatch {
                m: n_bary,
                n: n_k,
                a_len: ck.len(),
                b_len: n_bary * n_k,
            });
        }
        if measures_a[k].is_empty() {
            return Err(OtError::EmptyInput);
        }
        for &v in measures_a[k].iter() {
            if v < 0.0 || !v.is_finite() {
                return Err(OtError::NegativeWeight);
            }
        }
        for &v in ck.iter() {
            if !v.is_finite() {
                return Err(OtError::Internal {
                    msg: "non-finite cost".into(),
                });
            }
        }
    }
    Ok(())
}

/// Compute the fixed-support entropic barycenter weights.
///
/// `costs[k]` is shape `[n_bary × n_k]` row-major; `measures_a[k]` has length
/// `n_k`. Returns the barycenter weights `b` (length `n_bary`).
pub fn fixed_support_barycenter(
    costs: &[Vec<f32>],
    measures_a: &[Vec<f32>],
    lambdas: &[f32],
    n_bary: usize,
    cfg: &FixedBaryConfig,
) -> OtResult<Vec<f32>> {
    validate(costs, measures_a, lambdas, n_bary, cfg)?;
    let n_meas = costs.len();
    let inv_eps = 1.0_f32 / cfg.eps;

    // Precompute the kernels K_k = exp(−C_k / ε).
    let kernels: Vec<Vec<f32>> = costs
        .iter()
        .map(|ck| {
            ck.iter()
                .map(|c| (-c * inv_eps).exp())
                .collect::<Vec<f32>>()
        })
        .collect();
    // Renormalise input weights to a probability simplex.
    let renorm_a: Vec<Vec<f32>> = measures_a
        .iter()
        .map(|ws| {
            let total: f32 = ws.iter().copied().sum();
            if total <= FLOOR {
                vec![0.0_f32; ws.len()]
            } else {
                ws.iter().map(|&w| w / total).collect::<Vec<f32>>()
            }
        })
        .collect();

    let mut b = vec![1.0_f32 / n_bary as f32; n_bary];
    let mut prev = b.clone();

    let mut tmp_kb = vec![0.0_f32; n_bary]; // per-measure K_k · b (length n_bary).
    let mut log_sum = vec![0.0_f32; n_bary];

    for _ in 0..cfg.max_iter {
        for slot in log_sum.iter_mut() {
            *slot = 0.0;
        }
        for k in 0..n_meas {
            let n_k = measures_a[k].len();
            let kk = &kernels[k];
            // K_k · b: row-major (n_bary × n_k) times (length n_bary)? In the
            // formulation `K_k · b` is the marginal `Σ_i K_k_ij b_i` of length
            // n_k. Compute that into `marg_b` (length n_k).
            let mut marg_b = vec![0.0_f32; n_k];
            for i in 0..n_bary {
                let row_off = i * n_k;
                let bi = b[i];
                if bi == 0.0 {
                    continue;
                }
                for j in 0..n_k {
                    marg_b[j] += kk[row_off + j] * bi;
                }
            }
            // ratio_j = a_kj / (K_k · b)_j with a small floor.
            let a_k = &renorm_a[k];
            let mut ratio = vec![0.0_f32; n_k];
            for j in 0..n_k {
                let denom = if marg_b[j] > FLOOR { marg_b[j] } else { FLOOR };
                ratio[j] = a_k[j] / denom;
            }
            // out_i = Σ_j K_k_ij · ratio_j (length n_bary).
            for slot in tmp_kb.iter_mut() {
                *slot = 0.0;
            }
            for i in 0..n_bary {
                let row_off = i * n_k;
                let mut acc = 0.0_f32;
                for j in 0..n_k {
                    acc += kk[row_off + j] * ratio[j];
                }
                tmp_kb[i] = acc;
            }
            // Accumulate the log of the geometric-mean factor:
            //   log b_new += λ_k · log(out_i)
            let lam = lambdas[k];
            for (i, &x) in tmp_kb.iter().enumerate() {
                let safe = if x > FLOOR { x } else { FLOOR };
                log_sum[i] += lam * safe.ln();
            }
        }

        // Multiply with the current b, normalise.
        let mut new_b = vec![0.0_f32; n_bary];
        let mut total = 0.0_f32;
        for (i, &lse) in log_sum.iter().enumerate() {
            let v = b[i] * lse.exp();
            new_b[i] = v;
            total += v;
        }
        let inv_total = if total > FLOOR { 1.0 / total } else { 1.0 };
        for slot in new_b.iter_mut() {
            *slot *= inv_total;
        }

        // Convergence check on max abs change.
        let mut max_change = 0.0_f32;
        for (a, b_old) in new_b.iter().zip(prev.iter()) {
            let d = (a - b_old).abs();
            if d > max_change {
                max_change = d;
            }
        }
        prev.copy_from_slice(&new_b);
        b = new_b;
        if max_change < cfg.tol {
            break;
        }
    }

    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn empty_inputs_rejected() {
        let cfg = FixedBaryConfig::default();
        let res = fixed_support_barycenter(&[], &[], &[], 2, &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn lambda_validates_simplex() {
        let cfg = FixedBaryConfig::default();
        let costs = vec![vec![0.0_f32; 4], vec![0.0_f32; 4]];
        let aks = vec![vec![0.5_f32, 0.5], vec![0.5_f32, 0.5]];
        let lambdas = vec![0.7_f32, 0.7]; // sum = 1.4
        let res = fixed_support_barycenter(&costs, &aks, &lambdas, 2, &cfg);
        assert!(matches!(res, Err(OtError::NotProbability)));
    }

    #[test]
    fn cost_shape_validates() {
        let cfg = FixedBaryConfig::default();
        let costs = vec![vec![0.0_f32; 3]];
        let aks = vec![vec![0.5_f32, 0.5]];
        let lambdas = vec![1.0_f32];
        let res = fixed_support_barycenter(&costs, &aks, &lambdas, 2, &cfg);
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn weights_sum_to_one() {
        let cfg = FixedBaryConfig {
            eps: 0.1,
            max_iter: 200,
            tol: 1e-6,
        };
        // Two measures, three barycenter points, two source points each.
        let costs = vec![
            vec![0.0_f32, 1.0, 1.0, 0.0, 2.0, 1.0],
            vec![0.5_f32, 0.5, 0.5, 0.5, 0.5, 0.5],
        ];
        let aks = vec![vec![0.5_f32, 0.5], vec![0.4_f32, 0.6]];
        let lambdas = vec![0.5_f32, 0.5];
        let b = fixed_support_barycenter(&costs, &aks, &lambdas, 3, &cfg).expect("converges");
        let s: f32 = b.iter().copied().sum();
        assert!(approx(s, 1.0, 1e-3), "sum={s}");
        for &bi in &b {
            assert!(bi >= -1e-6 && bi.is_finite());
        }
    }

    #[test]
    fn equal_measures_yields_uniform_barycenter() {
        // Two identical inputs: uniform measure over the same support.
        // Identity-style cost: zero diagonal, uniform off-diagonal.
        let n_bary = 3;
        let n_k = 3;
        let mut cost = vec![1.0_f32; n_bary * n_k];
        for i in 0..n_bary {
            cost[i * n_k + i] = 0.0;
        }
        let cfg = FixedBaryConfig {
            eps: 0.1,
            max_iter: 500,
            tol: 1e-7,
        };
        let costs = vec![cost.clone(), cost];
        let aks = vec![vec![1.0_f32 / 3.0; 3], vec![1.0_f32 / 3.0; 3]];
        let lambdas = vec![0.5_f32, 0.5];
        let b = fixed_support_barycenter(&costs, &aks, &lambdas, n_bary, &cfg).expect("converges");
        for &bi in &b {
            assert!(approx(bi, 1.0 / 3.0, 5e-2), "b_i = {bi}");
        }
    }
}
