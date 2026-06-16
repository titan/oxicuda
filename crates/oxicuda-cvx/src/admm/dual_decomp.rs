//! Dual Decomposition (Boyd §7.2).
//!
//! Solves `min Σᵢ fᵢ(xᵢ)  s.t.  Σᵢ Aᵢ xᵢ = b` via dual ascent.
//!
//! Each block i supplies a closure `x_updates[i](λ) → xᵢ*` that performs the
//! minimisation `argmin_xᵢ [fᵢ(xᵢ) + λᵀ Aᵢ xᵢ]` in closed form (or by any
//! inner solver).  The master routine updates the shared dual variable λ by
//! gradient ascent on the dual function.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::{mat_vec, norm2};

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for dual decomposition.
#[derive(Debug, Clone)]
pub struct DualDecompConfig {
    /// Dual ascent step size (must be > 0).
    pub step_size: f64,
    /// Maximum number of outer iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the constraint-residual norm.
    pub tol: f64,
}

impl Default for DualDecompConfig {
    fn default() -> Self {
        Self {
            step_size: 0.1,
            max_iter: 1000,
            tol: 1e-6,
        }
    }
}

/// Output of a successful dual decomposition run.
#[derive(Debug, Clone)]
pub struct DualDecompResult {
    /// Optimal primal blocks `xᵢ*`.
    pub x_blocks: Vec<Vec<f64>>,
    /// Final dual variable λ (Lagrange multiplier for `Σᵢ Aᵢ xᵢ = b`).
    pub lambda: Vec<f64>,
    /// Number of outer iterations performed.
    pub iter: usize,
    /// Constraint residual `||Σᵢ Aᵢ xᵢ - b||₂` at termination.
    pub residual: f64,
    /// Whether the algorithm converged within `max_iter`.
    pub converged: bool,
}

// ──────────────────────────────────────────────────────────────────────────────
// Main solver
// ──────────────────────────────────────────────────────────────────────────────

/// Dual decomposition solver (Boyd §7.2).
///
/// # Arguments
///
/// * `a_blocks`   – Row-major flat matrices `Aᵢ`, one slice per block.
/// * `a_dims`     – `(m, nᵢ)` shape of each `Aᵢ`; all must share the same row count `m`.
/// * `b`          – Shared right-hand side, length `m`.
/// * `x_updates`  – Closures `λ → xᵢ* = argmin [fᵢ(xᵢ) + λᵀ Aᵢ xᵢ]`.
/// * `cfg`        – Algorithm hyper-parameters.
///
/// # Errors
///
/// Returns `CvxError::EmptyInput` when no blocks are supplied.
/// Returns `CvxError::ShapeMismatch` for length mismatches across the inputs.
/// Returns `CvxError::InvalidParameter` for non-positive `step_size` or `tol`.
/// Propagates any error returned by an `x_updates` closure.
#[allow(clippy::type_complexity)]
pub fn dual_decomp(
    a_blocks: &[&[f64]],
    a_dims: &[(usize, usize)], // (m, n_i) for each block i
    b: &[f64],                 // shared RHS, length m
    x_updates: &[&dyn Fn(&[f64]) -> CvxResult<Vec<f64>>],
    cfg: &DualDecompConfig,
) -> CvxResult<DualDecompResult> {
    // ── Input validation ──────────────────────────────────────────────────────
    let num_blocks = a_blocks.len();

    if num_blocks == 0 {
        return Err(CvxError::EmptyInput);
    }

    if x_updates.len() != num_blocks {
        return Err(CvxError::ShapeMismatch {
            expected: vec![num_blocks],
            got: vec![x_updates.len()],
        });
    }

    if a_dims.len() != num_blocks {
        return Err(CvxError::ShapeMismatch {
            expected: vec![num_blocks],
            got: vec![a_dims.len()],
        });
    }

    if cfg.step_size <= 0.0 {
        return Err(CvxError::InvalidParameter(
            "step_size must be positive".into(),
        ));
    }

    if cfg.tol <= 0.0 {
        return Err(CvxError::InvalidParameter("tol must be positive".into()));
    }

    // Determine shared constraint dimension m from a_dims[0].0 and verify
    // consistency with b and all other blocks.
    let m = a_dims[0].0;

    if b.len() != m {
        return Err(CvxError::ShapeMismatch {
            expected: vec![m],
            got: vec![b.len()],
        });
    }

    for (i, &(mi, ni)) in a_dims.iter().enumerate() {
        if mi != m {
            return Err(CvxError::ShapeMismatch {
                expected: vec![m],
                got: vec![mi],
            });
        }
        let expected_len = mi * ni;
        if a_blocks[i].len() != expected_len {
            return Err(CvxError::ShapeMismatch {
                expected: vec![expected_len],
                got: vec![a_blocks[i].len()],
            });
        }
    }

    // ── Initialisation ────────────────────────────────────────────────────────
    let mut lambda = vec![0.0_f64; m];

    // Warm-start all blocks with λ = 0.
    let mut x_blocks: Vec<Vec<f64>> = (0..num_blocks)
        .map(|i| x_updates[i](&lambda))
        .collect::<CvxResult<_>>()?;

    // ── Main dual-ascent loop ─────────────────────────────────────────────────
    let mut residual = 0.0_f64;
    let mut converged = false;
    let mut iter = 0usize;

    for _it in 0..cfg.max_iter {
        iter += 1;

        // Block updates: xᵢ ← argmin [fᵢ(xᵢ) + λᵀ Aᵢ xᵢ]
        for i in 0..num_blocks {
            x_blocks[i] = x_updates[i](&lambda)?;
        }

        // Constraint residual: r = Σᵢ Aᵢ xᵢ − b  (length m)
        let mut r = vec![0.0_f64; m];
        for i in 0..num_blocks {
            let ax_i = mat_vec(a_blocks[i], a_dims[i].0, a_dims[i].1, &x_blocks[i])?;
            for k in 0..m {
                r[k] += ax_i[k];
            }
        }
        for k in 0..m {
            r[k] -= b[k];
        }

        // Dual ascent: λ ← λ + α · r
        for k in 0..m {
            lambda[k] += cfg.step_size * r[k];
        }

        // Convergence check
        residual = norm2(&r);
        if residual < cfg.tol {
            converged = true;
            break;
        }
    }

    Ok(DualDecompResult {
        x_blocks,
        lambda,
        iter,
        residual,
        converged,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CvxResult;

    /// A single block sub-problem closure `λ → xᵢ*`.
    type SubproblemFn<'a> = &'a dyn Fn(&[f64]) -> CvxResult<Vec<f64>>;
    /// A slice of block sub-problem closures, one per block.
    type SubproblemFns<'a> = &'a [SubproblemFn<'a>];

    // ── helper ────────────────────────────────────────────────────────────────

    /// Build a 1×1 identity block as a slice.
    fn scalar_block() -> [f64; 1] {
        [1.0_f64]
    }

    // ── test 1 ────────────────────────────────────────────────────────────────

    /// Two equal-weight blocks: f₁(x)=½x², f₂(x)=½x², A₁=A₂=[1], b=[1].
    /// Optimal: x₁*=x₂*=0.5.
    #[test]
    fn test_two_block_equal_weights() {
        let a = scalar_block();
        // min ½x² + λx  →  x* = −λ  (derivative: x + λ = 0)
        let f1 = |lam: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![-lam[0]]) };
        let f2 = |lam: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![-lam[0]]) };
        let upds: SubproblemFns = &[&f1 as SubproblemFn, &f2 as SubproblemFn];

        let cfg = DualDecompConfig {
            step_size: 0.1,
            max_iter: 2000,
            tol: 1e-6,
        };
        let result = dual_decomp(&[&a[..], &a[..]], &[(1, 1), (1, 1)], &[1.0], upds, &cfg)
            .expect("dual_decomp should succeed");

        let x1 = result.x_blocks[0][0];
        let x2 = result.x_blocks[1][0];
        assert!(
            (x1 + x2 - 1.0).abs() < 1e-4,
            "constraint: x1+x2={} != 1.0",
            x1 + x2
        );
        assert!((x1 - 0.5).abs() < 1e-3, "x1={} != 0.5", x1);
        assert!((x2 - 0.5).abs() < 1e-3, "x2={} != 0.5", x2);
    }

    // ── test 2 ────────────────────────────────────────────────────────────────

    /// Two unequal-weight blocks: f₁(x)=x², f₂(x)=4x².
    /// KKT: x₁=4/5=0.8, x₂=1/5=0.2.
    #[test]
    fn test_two_block_unequal_weights() {
        let a = scalar_block();
        // min x² + λx  →  x* = −λ/2
        let f1 = |lam: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![-lam[0] / 2.0]) };
        // min 4x² + λx  →  x* = −λ/8
        let f2 = |lam: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![-lam[0] / 8.0]) };
        let upds: SubproblemFns = &[&f1 as SubproblemFn, &f2 as SubproblemFn];

        let cfg = DualDecompConfig {
            step_size: 0.05,
            max_iter: 5000,
            tol: 1e-6,
        };
        let result = dual_decomp(&[&a[..], &a[..]], &[(1, 1), (1, 1)], &[1.0], upds, &cfg)
            .expect("dual_decomp should succeed");

        let x1 = result.x_blocks[0][0];
        let x2 = result.x_blocks[1][0];
        assert!((x1 - 0.8).abs() < 1e-3, "x1={} != 0.8", x1);
        assert!((x2 - 0.2).abs() < 1e-3, "x2={} != 0.2", x2);
    }

    // ── test 3 ────────────────────────────────────────────────────────────────

    /// Constraint satisfaction: |x1+x2-1| < 1e-4.
    #[test]
    fn test_constraint_satisfied_at_convergence() {
        let a = scalar_block();
        let f1 = |lam: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![-lam[0]]) };
        let f2 = |lam: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![-lam[0]]) };
        let upds: SubproblemFns = &[&f1 as SubproblemFn, &f2 as SubproblemFn];

        let cfg = DualDecompConfig {
            step_size: 0.1,
            max_iter: 2000,
            tol: 1e-6,
        };
        let result = dual_decomp(&[&a[..], &a[..]], &[(1, 1), (1, 1)], &[1.0], upds, &cfg)
            .expect("dual_decomp should succeed");

        let sum = result.x_blocks[0][0] + result.x_blocks[1][0];
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "constraint violation |x1+x2-1|={}",
            (sum - 1.0).abs()
        );
    }

    // ── test 4 ────────────────────────────────────────────────────────────────

    /// Three equal-weight blocks: f_i(x)=1/2*x², x₁+x₂+x₃=3. Optimal: xᵢ*=1.
    #[test]
    fn test_three_blocks() {
        let a = scalar_block();
        let f = |lam: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![-lam[0]]) };
        let upds: SubproblemFns = &[&f as SubproblemFn, &f as SubproblemFn, &f as SubproblemFn];

        let cfg = DualDecompConfig {
            step_size: 0.1,
            max_iter: 3000,
            tol: 1e-6,
        };
        let result = dual_decomp(
            &[&a[..], &a[..], &a[..]],
            &[(1, 1), (1, 1), (1, 1)],
            &[3.0],
            upds,
            &cfg,
        )
        .expect("dual_decomp should succeed");

        for (i, xi) in result.x_blocks.iter().enumerate() {
            assert!((xi[0] - 1.0).abs() < 1e-3, "block {i}: x={} != 1.0", xi[0]);
        }
    }

    // ── test 5 ────────────────────────────────────────────────────────────────

    /// The final lambda is non-trivial (|lambda| > 0.1) for the equal-weights
    /// two-block problem; at KKT: lambda = -x1 = -0.5.
    #[test]
    fn test_lambda_final_is_lagrange_multiplier() {
        let a = scalar_block();
        let f1 = |lam: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![-lam[0]]) };
        let f2 = |lam: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![-lam[0]]) };
        let upds: SubproblemFns = &[&f1 as SubproblemFn, &f2 as SubproblemFn];

        let cfg = DualDecompConfig {
            step_size: 0.1,
            max_iter: 2000,
            tol: 1e-6,
        };
        let result = dual_decomp(&[&a[..], &a[..]], &[(1, 1), (1, 1)], &[1.0], upds, &cfg)
            .expect("dual_decomp should succeed");

        assert!(
            result.lambda[0].abs() > 0.1,
            "lambda={} is trivial",
            result.lambda[0]
        );
    }

    // ── test 6 ────────────────────────────────────────────────────────────────

    /// Empty block list must return `CvxError::EmptyInput`.
    #[test]
    fn test_empty_blocks_error() {
        let result = dual_decomp(&[], &[], &[1.0], &[], &DualDecompConfig::default());
        assert!(
            matches!(result, Err(CvxError::EmptyInput)),
            "expected EmptyInput, got {:?}",
            result
        );
    }

    // ── test 7 ────────────────────────────────────────────────────────────────

    /// Mismatched number of x_updates vs a_blocks must return `ShapeMismatch`.
    #[test]
    fn test_mismatch_blocks_error() {
        let a = scalar_block();
        let upd1 = |lam: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![-lam[0]]) };
        let result = dual_decomp(
            &[&a[..], &a[..]], // 2 blocks
            &[(1, 1), (1, 1)],
            &[1.0],
            &[&upd1 as SubproblemFn], // only 1 update
            &DualDecompConfig::default(),
        );
        assert!(
            matches!(result, Err(CvxError::ShapeMismatch { .. })),
            "expected ShapeMismatch, got {:?}",
            result
        );
    }

    // ── test 8 ────────────────────────────────────────────────────────────────

    /// Non-positive step_size must return `CvxError::InvalidParameter`.
    #[test]
    fn test_negative_step_size_error() {
        let cfg = DualDecompConfig {
            step_size: 0.0,
            max_iter: 100,
            tol: 1e-6,
        };
        let a = scalar_block();
        let upd = |lam: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![-lam[0]]) };
        let result = dual_decomp(&[&a[..]], &[(1, 1)], &[1.0], &[&upd as SubproblemFn], &cfg);
        assert!(
            matches!(result, Err(CvxError::InvalidParameter(_))),
            "expected InvalidParameter, got {:?}",
            result
        );
    }

    // ── test 9 ────────────────────────────────────────────────────────────────

    /// Trivial 1-block, b=[0], f(x)=x* -> x=0.  Must converge immediately.
    #[test]
    fn test_convergence_marker() {
        let a = scalar_block();
        // min x² + lambda*x  ->  x* = -lambda/2
        let upd = |lam: &[f64]| -> CvxResult<Vec<f64>> { Ok(vec![-lam[0] / 2.0]) };

        let cfg = DualDecompConfig {
            step_size: 0.1,
            max_iter: 1000,
            tol: 1e-6,
        };
        let result = dual_decomp(&[&a[..]], &[(1, 1)], &[0.0], &[&upd as SubproblemFn], &cfg)
            .expect("dual_decomp should succeed");

        assert!(result.converged, "should have converged");
    }
}
