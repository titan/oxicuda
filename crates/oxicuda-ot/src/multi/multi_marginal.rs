#![allow(clippy::needless_range_loop)]
//! Multi-marginal optimal transport — tensor scaling in log-domain.
//!
//! ## Problem
//!
//! Given a non-negative cost tensor `C ∈ ℝ^{d_1 × d_2 × … × d_k}` and a list of
//! prescribed marginals `a^{(1)}, …, a^{(k)}`, solve
//!
//! ```text
//! min_T   ⟨C, T⟩ + ε · KL(T ‖ ⊗_i a^{(i)})
//!        s.t.   T 1_{¬ i} = a^{(i)} ,   i = 1, …, k
//! ```
//!
//! ## Algorithm
//!
//! We carry the log-plan in factorised form
//!
//! ```text
//! log T(idx) = (Σ_i f^{(i)}_{idx_i} − C(idx)) / ε
//! ```
//!
//! and update each potential `f^{(i)}` so that the corresponding marginal is
//! exact:
//!
//! ```text
//! f^{(i)}_{x} ← ε · log a^{(i)}_x − ε · LSE_{idx : idx_i = x} (
//!                 (Σ_{j ≠ i} f^{(j)}_{idx_j} − C(idx)) / ε
//!              )
//! ```
//!
//! For `k = 2` this reduces *exactly* to the standard log-domain Sinkhorn
//! algorithm.

use crate::error::{OtError, OtResult};

/// Configuration for the multi-marginal solver.
#[derive(Debug, Clone)]
pub struct MmConfig {
    /// Entropic regularisation strength ε > 0.
    pub eps: f32,
    /// Maximum number of full sweeps over all axes.
    pub max_iter: usize,
    /// Marginal-residual convergence tolerance.
    pub tol: f32,
}

impl Default for MmConfig {
    fn default() -> Self {
        Self {
            eps: 0.1,
            max_iter: 200,
            tol: 1e-4,
        }
    }
}

/// `log(x)` clamped from below by `log(f32::MIN_POSITIVE)`.
#[inline]
fn safe_ln(x: f32) -> f32 {
    let floor = f32::MIN_POSITIVE;
    if x <= floor { floor.ln() } else { x.ln() }
}

/// Compute the row-major strides of a tensor with the given shape.
fn strides_from_dims(dims: &[usize]) -> Vec<usize> {
    let k = dims.len();
    let mut strides = vec![1_usize; k];
    for i in (0..k.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1].saturating_mul(dims[i + 1]);
    }
    strides
}

/// Decode a flat index into a multi-index according to `strides`.
fn unravel(mut flat: usize, strides: &[usize], dims: &[usize], idx: &mut [usize]) {
    for (axis, (stride, dim)) in strides.iter().zip(dims.iter()).enumerate() {
        let s = *stride;
        if s == 0 || *dim == 0 {
            idx[axis] = 0;
            continue;
        }
        idx[axis] = flat / s;
        flat %= s;
    }
}

/// Validate inputs and return the total tensor length when all checks pass.
fn validate(
    cost: &[f32],
    marginals: &[Vec<f32>],
    dims: &[usize],
    cfg: &MmConfig,
) -> OtResult<usize> {
    if dims.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if cfg.eps <= 0.0 {
        return Err(OtError::BadEpsilon { eps: cfg.eps });
    }
    if marginals.len() != dims.len() {
        return Err(OtError::IncompatibleLength {
            a: marginals.len(),
            b: dims.len(),
        });
    }
    let mut total: usize = 1;
    for (i, &d) in dims.iter().enumerate() {
        if d == 0 {
            return Err(OtError::EmptyInput);
        }
        total = total.checked_mul(d).ok_or(OtError::Internal {
            msg: "tensor size overflow".to_string(),
        })?;
        if marginals[i].len() != d {
            return Err(OtError::MarginalMismatch {
                m: d,
                n: marginals[i].len(),
                a_len: d,
                b_len: marginals[i].len(),
            });
        }
        for &v in &marginals[i] {
            if v < 0.0 || !v.is_finite() {
                return Err(OtError::NegativeWeight);
            }
        }
    }
    if cost.len() != total {
        return Err(OtError::MarginalMismatch {
            m: total,
            n: cost.len(),
            a_len: total,
            b_len: cost.len(),
        });
    }
    Ok(total)
}

/// Compute the marginal-`axis` of the implicit log-plan into `out`.
///
/// `out[x] = LSE_{idx : idx_axis = x} ( (Σ_j f^{(j)}_{idx_j} − C(idx)) / ε )`.
/// We use the running max-trick variant by first scanning to find the per-x
/// maximum, then accumulating the exponential sum.
fn marginal_lse_along_axis(
    cost: &[f32],
    potentials: &[Vec<f32>],
    dims: &[usize],
    strides: &[usize],
    eps: f32,
    axis: usize,
    out: &mut [f32],
) {
    let k = dims.len();
    let total = cost.len();
    let mut idx = vec![0_usize; k];

    // Pass 1: max per `idx_axis` over the OTHER potentials (excluding axis).
    let mut max_per_x = vec![f32::NEG_INFINITY; dims[axis]];
    for flat in 0..total {
        unravel(flat, strides, dims, &mut idx);
        let mut s = 0.0_f32;
        for (j, p) in potentials.iter().enumerate() {
            if j == axis {
                continue;
            }
            s += p[idx[j]];
        }
        let val = (s - cost[flat]) / eps;
        let x = idx[axis];
        if val > max_per_x[x] {
            max_per_x[x] = val;
        }
    }

    // Pass 2: accumulate exp(value − max) per `idx_axis`.
    let mut sum_per_x = vec![0.0_f32; dims[axis]];
    for flat in 0..total {
        unravel(flat, strides, dims, &mut idx);
        let mut s = 0.0_f32;
        for (j, p) in potentials.iter().enumerate() {
            if j == axis {
                continue;
            }
            s += p[idx[j]];
        }
        let val = (s - cost[flat]) / eps;
        let x = idx[axis];
        let m = max_per_x[x];
        if m.is_finite() {
            sum_per_x[x] += (val - m).exp();
        }
    }

    for x in 0..dims[axis] {
        let m = max_per_x[x];
        if !m.is_finite() {
            out[x] = m;
        } else {
            out[x] = m + sum_per_x[x].ln();
        }
    }
}

/// Compute `axis`-marginal of the *current* materialised plan (not log-domain).
fn marginal_along_axis(plan: &[f32], dims: &[usize], strides: &[usize], axis: usize) -> Vec<f32> {
    let k = dims.len();
    let mut out = vec![0.0_f32; dims[axis]];
    let mut idx = vec![0_usize; k];
    for (flat, &p) in plan.iter().enumerate() {
        unravel(flat, strides, dims, &mut idx);
        out[idx[axis]] += p;
    }
    out
}

/// Run multi-marginal optimal transport via tensor scaling.
///
/// `cost` is the cost tensor flattened in row-major order with shape
/// `[dims[0] × dims[1] × … × dims[k-1]]`. `marginals[i]` has length
/// `dims[i]`. Returns the joint plan `T` flattened in the same row-major
/// layout. Precondition: all marginals must have the same total mass.
pub fn multi_marginal_ot(
    cost: &[f32],
    marginals: &[Vec<f32>],
    dims: &[usize],
    cfg: &MmConfig,
) -> OtResult<Vec<f32>> {
    let total = validate(cost, marginals, dims, cfg)?;
    let k = dims.len();
    let strides = strides_from_dims(dims);
    let eps = cfg.eps;

    // Initialise potentials f^{(i)} = ε · log a^{(i)}.
    let mut potentials: Vec<Vec<f32>> = Vec::with_capacity(k);
    for marginal in marginals.iter().take(k) {
        let mut f = vec![0.0_f32; marginal.len()];
        for (x, fx) in f.iter_mut().enumerate() {
            *fx = eps * safe_ln(marginal[x]);
        }
        potentials.push(f);
    }

    let mut buf = vec![0.0_f32; *dims.iter().max().unwrap_or(&1)];

    let mut converged = false;
    for outer in 0..cfg.max_iter {
        for axis in 0..k {
            marginal_lse_along_axis(
                cost,
                &potentials,
                dims,
                &strides,
                eps,
                axis,
                &mut buf[..dims[axis]],
            );
            for x in 0..dims[axis] {
                let target = eps * safe_ln(marginals[axis][x]);
                potentials[axis][x] = target - eps * buf[x];
            }
        }

        // Materialise plan to compute residuals.
        let plan = materialise_plan(cost, &potentials, dims, &strides, eps);
        let mut max_residual = 0.0_f32;
        for axis in 0..k {
            let m = marginal_along_axis(&plan, dims, &strides, axis);
            for (x, &mx) in m.iter().enumerate() {
                let r = (mx - marginals[axis][x]).abs();
                if r > max_residual {
                    max_residual = r;
                }
            }
        }
        if max_residual < cfg.tol {
            converged = true;
            break;
        }
        if outer + 1 == cfg.max_iter && max_residual >= cfg.tol {
            return Err(OtError::NotConverged {
                iter: cfg.max_iter,
                tol: cfg.tol,
            });
        }
    }

    let plan = materialise_plan(cost, &potentials, dims, &strides, eps);
    if !converged {
        // Final safety check (loop above either breaks or returns NotConverged).
        debug_assert!(plan.len() == total);
    }
    Ok(plan)
}

/// Compose `T(idx) = exp((Σ_j f^{(j)}_{idx_j} − C(idx)) / ε)`.
fn materialise_plan(
    cost: &[f32],
    potentials: &[Vec<f32>],
    dims: &[usize],
    strides: &[usize],
    eps: f32,
) -> Vec<f32> {
    let k = dims.len();
    let total = cost.len();
    let mut plan = vec![0.0_f32; total];
    let mut idx = vec![0_usize; k];
    for (flat, slot) in plan.iter_mut().enumerate() {
        unravel(flat, strides, dims, &mut idx);
        let mut s = 0.0_f32;
        for (j, p) in potentials.iter().enumerate() {
            s += p[idx[j]];
        }
        *slot = ((s - cost[flat]) / eps).exp();
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sinkhorn::sinkhorn::{SinkhornConfig, sinkhorn};

    #[test]
    fn k2_matches_sinkhorn() {
        let m = 4;
        let n = 4;
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let d = i as f32 - j as f32;
                c[i * n + j] = d * d;
            }
        }
        let a = vec![0.25_f32; m];
        let b = vec![0.25_f32; n];
        let cfg_mm = MmConfig {
            eps: 0.4,
            max_iter: 8000,
            tol: 1e-3,
        };
        let cfg_sk = SinkhornConfig {
            eps: 0.4,
            max_iter: 8000,
            tol: 1e-3,
        };
        let plan_mm =
            multi_marginal_ot(&c, &[a.clone(), b.clone()], &[m, n], &cfg_mm).expect("mm converges");
        let plan_sk = sinkhorn(&c, &a, &b, m, n, &cfg_sk).expect("sk converges");
        for k in 0..m * n {
            assert!(
                (plan_mm[k] - plan_sk.plan[k]).abs() < 5e-3,
                "entry {k}: mm {} vs sk {}",
                plan_mm[k],
                plan_sk.plan[k]
            );
        }
    }

    #[test]
    fn k3_marginals_match() {
        let dims = [3_usize, 3, 3];
        let total = dims.iter().product::<usize>();
        let mut c = vec![0.0_f32; total];
        // Cost penalises disagreement: c(i,j,k) = (i−j)² + (j−k)².
        let strides = strides_from_dims(&dims);
        for flat in 0..total {
            let mut idx = [0_usize; 3];
            unravel(flat, &strides, &dims, &mut idx);
            let i = idx[0] as f32;
            let j = idx[1] as f32;
            let k = idx[2] as f32;
            c[flat] = (i - j) * (i - j) + (j - k) * (j - k);
        }
        let a = vec![1.0_f32 / 3.0; 3];
        let b = vec![1.0_f32 / 3.0; 3];
        let cm = vec![1.0_f32 / 3.0; 3];
        let cfg = MmConfig {
            eps: 0.5,
            max_iter: 8000,
            tol: 1e-3,
        };
        let plan = multi_marginal_ot(&c, &[a.clone(), b.clone(), cm.clone()], &dims, &cfg)
            .expect("converges");
        // Marginal projections on each axis.
        let strides = strides_from_dims(&dims);
        let m0 = marginal_along_axis(&plan, &dims, &strides, 0);
        let m1 = marginal_along_axis(&plan, &dims, &strides, 1);
        let m2 = marginal_along_axis(&plan, &dims, &strides, 2);
        for x in 0..3 {
            assert!((m0[x] - a[x]).abs() < 5e-3);
            assert!((m1[x] - b[x]).abs() < 5e-3);
            assert!((m2[x] - cm[x]).abs() < 5e-3);
        }
        // Total mass should equal target mass (1.0).
        let total_mass: f32 = plan.iter().sum();
        assert!((total_mass - 1.0).abs() < 5e-3);
    }

    #[test]
    fn k3_uniform_under_constant_cost() {
        let dims = [2_usize, 2, 2];
        let total = dims.iter().product::<usize>();
        let c = vec![1.0_f32; total];
        let a = vec![0.5_f32; 2];
        let cfg = MmConfig {
            eps: 0.5,
            max_iter: 4000,
            tol: 1e-3,
        };
        let plan =
            multi_marginal_ot(&c, &[a.clone(), a.clone(), a.clone()], &dims, &cfg).expect("ok");
        let expected = 1.0_f32 / total as f32;
        for &p in &plan {
            assert!((p - expected).abs() < 1e-3);
        }
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let cfg = MmConfig::default();
        let res = multi_marginal_ot(
            &[0.0_f32; 4],
            &[vec![0.5_f32; 2], vec![0.5_f32; 2]],
            &[2, 2, 2],
            &cfg,
        );
        assert!(matches!(res, Err(OtError::IncompatibleLength { .. })));
    }

    #[test]
    fn rejects_marginal_length_mismatch() {
        let cfg = MmConfig::default();
        let res = multi_marginal_ot(
            &[0.0_f32; 9],
            &[vec![0.5_f32; 2], vec![0.5_f32; 3]],
            &[3, 3],
            &cfg,
        );
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn rejects_cost_length_mismatch() {
        let cfg = MmConfig::default();
        let res = multi_marginal_ot(
            &[0.0_f32; 5],
            &[vec![0.5_f32; 2], vec![0.5_f32; 2]],
            &[2, 2],
            &cfg,
        );
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn rejects_bad_eps() {
        let cfg = MmConfig {
            eps: 0.0,
            ..Default::default()
        };
        let res = multi_marginal_ot(
            &[0.0_f32; 4],
            &[vec![0.5_f32; 2], vec![0.5_f32; 2]],
            &[2, 2],
            &cfg,
        );
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn rejects_negative_marginal() {
        let cfg = MmConfig::default();
        let res = multi_marginal_ot(
            &[0.0_f32; 4],
            &[vec![-0.1_f32, 0.6], vec![0.5_f32; 2]],
            &[2, 2],
            &cfg,
        );
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }

    #[test]
    fn rejects_empty_input() {
        let cfg = MmConfig::default();
        let res = multi_marginal_ot(&[], &[], &[], &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }
}
