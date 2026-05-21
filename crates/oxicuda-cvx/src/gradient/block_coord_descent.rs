//! Block Coordinate Descent (BCD) for structured problems.
//!
//! References:
//! - Tseng, P. (2001), "Convergence of a Block Coordinate Descent Method
//!   for Nondifferentiable Minimization", JOTA, 109(3), 475-494.
//! - Beck, A. & Tetruashvili, L. (2013), "On the Convergence of Block Coordinate
//!   Descent Type Methods", SIAM J. Optim., 23(4), 2037-2060.
//!
//! Unlike single-coordinate descent (`coord_descent.rs`), BCD updates groups
//! ("blocks") of coordinates simultaneously. The user supplies a partition of
//! `0..d` into `M` blocks and a `block_step` closure that returns the new
//! sub-vector for a given block given the current full `x`. For quadratic
//! objectives `min ½ xᵀA x − bᵀx`, a specialised entry point performs each
//! per-block update by an EXACT solve of the corresponding SPD sub-system
//! `A_BB · x_B = b_B − A_BO · x_O` using `linalg::solve::solve_dense`.

use crate::error::{CvxError, CvxResult};
use crate::handle::LcgRng;
use crate::linalg::solve::solve_dense;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Block sweep strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BcdSweep {
    /// Iterate blocks 0..M in fixed order (one sweep = one pass over all blocks).
    Cyclic,
    /// Randomly pick one block per sub-step (one sweep = `n_blocks` sub-steps).
    Random,
}

/// Inner solver applied within each block update for the specialised
/// quadratic API.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InnerSolver {
    /// Exact dense solve of the block sub-system (SPD via LU; reuses
    /// `linalg::solve::solve_dense`).
    ExactQuadratic,
    /// Gradient descent on the block restriction with a fixed step.
    GradientDescent { inner_iter: usize, inner_lr: f64 },
}

/// Configuration for block coordinate descent.
#[derive(Debug, Clone)]
pub struct BcdConfig {
    /// Number of blocks `M` (must equal `blocks.len()`).
    pub n_blocks: usize,
    /// Maximum number of full sweeps.
    pub max_iter: usize,
    /// Convergence tolerance: stop when `||x_new - x_old||_2 < tol` after a sweep.
    pub tol: f64,
    /// Sweep strategy.
    pub sweep: BcdSweep,
    /// Inner per-block solver (used by `block_coord_descent_quadratic`).
    pub inner: InnerSolver,
}

/// State returned by BCD solvers.
#[derive(Debug, Clone)]
pub struct BcdState {
    /// Final iterate.
    pub x: Vec<f64>,
    /// Number of FULL sweeps completed.
    pub iter: usize,
    /// Objective value recorded after each full sweep.
    pub objective_history: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_cfg(cfg: &BcdConfig) -> CvxResult<()> {
    if cfg.n_blocks == 0 {
        return Err(CvxError::InvalidParameter("n_blocks must be >= 1".into()));
    }
    if cfg.max_iter == 0 {
        return Err(CvxError::InvalidParameter("max_iter must be >= 1".into()));
    }
    if !(cfg.tol > 0.0 && cfg.tol.is_finite()) {
        return Err(CvxError::InvalidParameter(format!(
            "tol must be > 0 and finite, got {}",
            cfg.tol
        )));
    }
    if let InnerSolver::GradientDescent {
        inner_iter,
        inner_lr,
    } = cfg.inner
    {
        if inner_iter == 0 {
            return Err(CvxError::InvalidParameter("inner_iter must be >= 1".into()));
        }
        if !(inner_lr > 0.0 && inner_lr.is_finite()) {
            return Err(CvxError::InvalidParameter(format!(
                "inner_lr must be > 0 and finite, got {inner_lr}"
            )));
        }
    }
    Ok(())
}

/// Validate that `blocks` is a clean partition of `0..d`, i.e. every index
/// appears EXACTLY ONCE across all blocks and `n_blocks == blocks.len()`.
fn validate_partition(blocks: &[Vec<usize>], d: usize, n_blocks: usize) -> CvxResult<()> {
    if blocks.is_empty() {
        return Err(CvxError::EmptyInput);
    }
    if blocks.len() != n_blocks {
        return Err(CvxError::InvalidParameter(format!(
            "n_blocks={n_blocks} but blocks.len()={}",
            blocks.len()
        )));
    }
    if d == 0 {
        return Err(CvxError::EmptyInput);
    }
    let mut seen = vec![false; d];
    let mut total = 0usize;
    for (b_idx, block) in blocks.iter().enumerate() {
        if block.is_empty() {
            return Err(CvxError::InvalidParameter(format!(
                "block {b_idx} is empty"
            )));
        }
        for &idx in block {
            if idx >= d {
                return Err(CvxError::IndexOutOfBounds { index: idx, len: d });
            }
            let seen_idx = seen
                .get_mut(idx)
                .ok_or(CvxError::IndexOutOfBounds { index: idx, len: d })?;
            if *seen_idx {
                return Err(CvxError::InvalidParameter(format!(
                    "index {idx} appears in more than one block"
                )));
            }
            *seen_idx = true;
            total += 1;
        }
    }
    if total != d {
        return Err(CvxError::InvalidParameter(format!(
            "blocks cover {total} indices but d={d}; missing index(es)"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#[inline]
fn diff_l2_norm(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0_f64;
    for (ai, bi) in a.iter().zip(b.iter()) {
        let d = ai - bi;
        s += d * d;
    }
    s.sqrt()
}

/// Compute `½ xᵀ A x − bᵀx` (row-major SPD `A`).
fn quadratic_objective(a: &[f64], b: &[f64], x: &[f64]) -> CvxResult<f64> {
    let d = x.len();
    if a.len() != d * d {
        return Err(CvxError::ShapeMismatch {
            expected: vec![d, d],
            got: vec![a.len()],
        });
    }
    if b.len() != d {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: d });
    }
    let mut quad = 0.0_f64;
    let mut lin = 0.0_f64;
    for i in 0..d {
        let row_start = i * d;
        let xi = *x
            .get(i)
            .ok_or(CvxError::IndexOutOfBounds { index: i, len: d })?;
        let mut row_sum = 0.0_f64;
        for j in 0..d {
            let a_ij = *a.get(row_start + j).ok_or(CvxError::IndexOutOfBounds {
                index: row_start + j,
                len: a.len(),
            })?;
            let xj = *x
                .get(j)
                .ok_or(CvxError::IndexOutOfBounds { index: j, len: d })?;
            row_sum += a_ij * xj;
        }
        quad += xi * row_sum;
        let bi = *b
            .get(i)
            .ok_or(CvxError::IndexOutOfBounds { index: i, len: d })?;
        lin += bi * xi;
    }
    Ok(0.5 * quad - lin)
}

/// Build `A_BB` (block × block) and `r_B = b_B − A_BO · x_O` for the per-block
/// exact quadratic update.
fn build_block_system(
    a: &[f64],
    b: &[f64],
    x: &[f64],
    d: usize,
    block: &[usize],
) -> CvxResult<(Vec<f64>, Vec<f64>)> {
    let p = block.len();
    let mut a_bb = vec![0.0_f64; p * p];
    let mut r_b = vec![0.0_f64; p];
    for (local_i, &gi) in block.iter().enumerate() {
        if gi >= d {
            return Err(CvxError::IndexOutOfBounds { index: gi, len: d });
        }
        let row_global = gi * d;
        // A_BB row: columns from block indices.
        for (local_j, &gj) in block.iter().enumerate() {
            if gj >= d {
                return Err(CvxError::IndexOutOfBounds { index: gj, len: d });
            }
            let a_val = *a.get(row_global + gj).ok_or(CvxError::IndexOutOfBounds {
                index: row_global + gj,
                len: a.len(),
            })?;
            let store_idx = local_i * p + local_j;
            let cell = a_bb.get_mut(store_idx).ok_or(CvxError::IndexOutOfBounds {
                index: store_idx,
                len: p * p,
            })?;
            *cell = a_val;
        }
        // r_B = b_B − A_BO · x_O (only sum over non-block indices).
        let b_val = *b
            .get(gi)
            .ok_or(CvxError::IndexOutOfBounds { index: gi, len: d })?;
        let mut rhs = b_val;
        for k in 0..d {
            if block.contains(&k) {
                continue;
            }
            let a_ik = *a.get(row_global + k).ok_or(CvxError::IndexOutOfBounds {
                index: row_global + k,
                len: a.len(),
            })?;
            let xk = *x
                .get(k)
                .ok_or(CvxError::IndexOutOfBounds { index: k, len: d })?;
            rhs -= a_ik * xk;
        }
        let rcell = r_b.get_mut(local_i).ok_or(CvxError::IndexOutOfBounds {
            index: local_i,
            len: p,
        })?;
        *rcell = rhs;
    }
    Ok((a_bb, r_b))
}

/// Per-block ExactQuadratic update: solve `A_BB · x_B = b_B − A_BO · x_O`.
fn exact_quadratic_block_step(
    a: &[f64],
    b: &[f64],
    x: &[f64],
    d: usize,
    block: &[usize],
) -> CvxResult<Vec<f64>> {
    let (a_bb, r_b) = build_block_system(a, b, x, d, block)?;
    let p = block.len();
    solve_dense(&a_bb, p, &r_b)
}

/// Per-block GradientDescent update: starting from `x_B` (current values),
/// take `inner_iter` steps of `x_B ← x_B − η · (A_BB x_B − r_B)` (which is the
/// gradient of `½ x_Bᵀ A_BB x_B − r_Bᵀ x_B`).
fn gd_quadratic_block_step(
    a: &[f64],
    b: &[f64],
    x: &[f64],
    d: usize,
    block: &[usize],
    inner_iter: usize,
    inner_lr: f64,
) -> CvxResult<Vec<f64>> {
    let (a_bb, r_b) = build_block_system(a, b, x, d, block)?;
    let p = block.len();
    let mut x_b = vec![0.0_f64; p];
    for (local_i, &gi) in block.iter().enumerate() {
        let xi = *x
            .get(gi)
            .ok_or(CvxError::IndexOutOfBounds { index: gi, len: d })?;
        let cell = x_b.get_mut(local_i).ok_or(CvxError::IndexOutOfBounds {
            index: local_i,
            len: p,
        })?;
        *cell = xi;
    }
    let mut grad = vec![0.0_f64; p];
    for _ in 0..inner_iter {
        // grad = A_BB · x_B − r_B
        for i in 0..p {
            let row_start = i * p;
            let mut s = 0.0_f64;
            for j in 0..p {
                let a_ij = *a_bb.get(row_start + j).ok_or(CvxError::IndexOutOfBounds {
                    index: row_start + j,
                    len: p * p,
                })?;
                let xj = *x_b
                    .get(j)
                    .ok_or(CvxError::IndexOutOfBounds { index: j, len: p })?;
                s += a_ij * xj;
            }
            let ri = *r_b
                .get(i)
                .ok_or(CvxError::IndexOutOfBounds { index: i, len: p })?;
            let gcell = grad
                .get_mut(i)
                .ok_or(CvxError::IndexOutOfBounds { index: i, len: p })?;
            *gcell = s - ri;
        }
        for i in 0..p {
            let gi_val = *grad
                .get(i)
                .ok_or(CvxError::IndexOutOfBounds { index: i, len: p })?;
            let cell = x_b
                .get_mut(i)
                .ok_or(CvxError::IndexOutOfBounds { index: i, len: p })?;
            *cell -= inner_lr * gi_val;
        }
    }
    Ok(x_b)
}

/// Apply a freshly-computed `x_block` (length `block.len()`) into the full
/// `x` vector at the indices given by `block`.
fn scatter_block(x: &mut [f64], block: &[usize], x_block: &[f64]) -> CvxResult<()> {
    if x_block.len() != block.len() {
        return Err(CvxError::DimensionMismatch {
            a: x_block.len(),
            b: block.len(),
        });
    }
    let d = x.len();
    for (local_i, &gi) in block.iter().enumerate() {
        if gi >= d {
            return Err(CvxError::IndexOutOfBounds { index: gi, len: d });
        }
        let val = *x_block.get(local_i).ok_or(CvxError::IndexOutOfBounds {
            index: local_i,
            len: x_block.len(),
        })?;
        let cell = x
            .get_mut(gi)
            .ok_or(CvxError::IndexOutOfBounds { index: gi, len: d })?;
        *cell = val;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Generic BCD: user-supplied block step
// ---------------------------------------------------------------------------

/// Generic block coordinate descent.
///
/// - `initial_x`: starting iterate of length `d`.
/// - `blocks`: partition of `0..d` into `cfg.n_blocks` blocks; each block is a
///   `Vec<usize>` of global indices.
/// - `objective(x) -> f64`: full-x objective recorded after each full sweep.
/// - `block_step(x, block_idx, block_indices) -> Vec<f64>`: returns the new
///   values for `x[block_indices]` given the current full `x`.
/// - `cfg`: solver configuration.
/// - `rng`: source of randomness (used only by `BcdSweep::Random`).
pub fn block_coord_descent<F, G>(
    initial_x: &[f64],
    blocks: &[Vec<usize>],
    objective: F,
    block_step: G,
    cfg: &BcdConfig,
    rng: &mut LcgRng,
) -> CvxResult<BcdState>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64], usize, &[usize]) -> CvxResult<Vec<f64>>,
{
    validate_cfg(cfg)?;
    let d = initial_x.len();
    validate_partition(blocks, d, cfg.n_blocks)?;

    let mut x = initial_x.to_vec();
    let mut history = Vec::with_capacity(cfg.max_iter);

    for sweep in 0..cfg.max_iter {
        let x_old = x.clone();
        // One "full sweep" consists of `n_blocks` sub-steps regardless of
        // strategy: cyclic walks blocks 0..M; random picks each sub-step
        // independently via `rng.next_usize(n_blocks)`.
        for sub in 0..cfg.n_blocks {
            let b_idx = match cfg.sweep {
                BcdSweep::Cyclic => sub,
                BcdSweep::Random => rng.next_usize(cfg.n_blocks),
            };
            let block = blocks.get(b_idx).ok_or(CvxError::IndexOutOfBounds {
                index: b_idx,
                len: blocks.len(),
            })?;
            let new_x_block = block_step(&x, b_idx, block)?;
            scatter_block(&mut x, block, &new_x_block)?;
        }
        let obj_val = objective(&x);
        history.push(obj_val);
        let delta = diff_l2_norm(&x, &x_old);
        if delta < cfg.tol {
            return Ok(BcdState {
                x,
                iter: sweep + 1,
                objective_history: history,
            });
        }
    }
    Ok(BcdState {
        x,
        iter: cfg.max_iter,
        objective_history: history,
    })
}

// ---------------------------------------------------------------------------
// Specialised quadratic BCD: min ½ xᵀA x − bᵀx
// ---------------------------------------------------------------------------

/// Block coordinate descent for the convex quadratic `min ½ xᵀA x − bᵀx`
/// with SPD `A` (row-major, length `d²`).
///
/// Per-block update reuses `linalg::solve::solve_dense` when
/// `cfg.inner == InnerSolver::ExactQuadratic`; alternatively a few iterations
/// of fixed-step gradient descent are taken when
/// `cfg.inner == InnerSolver::GradientDescent { .. }`. The objective
/// `½ xᵀA x − bᵀx` is recorded into `objective_history` after each full sweep.
pub fn block_coord_descent_quadratic(
    a: &[f64],
    b: &[f64],
    initial_x: &[f64],
    blocks: &[Vec<usize>],
    cfg: &BcdConfig,
    rng: &mut LcgRng,
) -> CvxResult<BcdState> {
    validate_cfg(cfg)?;
    let d = initial_x.len();
    if d == 0 {
        return Err(CvxError::EmptyInput);
    }
    if a.len() != d * d {
        return Err(CvxError::ShapeMismatch {
            expected: vec![d, d],
            got: vec![a.len()],
        });
    }
    if b.len() != d {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: d });
    }
    validate_partition(blocks, d, cfg.n_blocks)?;

    let inner = cfg.inner;
    let a_buf = a.to_vec();
    let b_buf = b.to_vec();

    let objective =
        |x: &[f64]| -> f64 { quadratic_objective(&a_buf, &b_buf, x).unwrap_or(f64::INFINITY) };
    let a_step = a.to_vec();
    let b_step = b.to_vec();
    let step_fn = move |x: &[f64], _b_idx: usize, block: &[usize]| -> CvxResult<Vec<f64>> {
        match inner {
            InnerSolver::ExactQuadratic => {
                exact_quadratic_block_step(&a_step, &b_step, x, d, block)
            }
            InnerSolver::GradientDescent {
                inner_iter,
                inner_lr,
            } => gd_quadratic_block_step(&a_step, &b_step, x, d, block, inner_iter, inner_lr),
        }
    };
    block_coord_descent(initial_x, blocks, objective, step_fn, cfg, rng)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::linalg::solve::solve_dense as direct_solve;

    fn make_spd(d: usize, seed: f64) -> Vec<f64> {
        // A = M Mᵀ + d·I for SPD with a known spectrum bound.
        let mut m = vec![0.0_f64; d * d];
        for i in 0..d {
            for j in 0..d {
                m[i * d + j] = ((i as f64 + 1.0) * (j as f64 + 0.5) + seed).sin();
            }
        }
        let mut a = vec![0.0_f64; d * d];
        for i in 0..d {
            for j in 0..d {
                let mut s = 0.0_f64;
                for k in 0..d {
                    s += m[i * d + k] * m[j * d + k];
                }
                a[i * d + j] = s;
                if i == j {
                    a[i * d + j] += (d as f64) + 1.0;
                }
            }
        }
        a
    }

    fn rhs_vec(d: usize, seed: f64) -> Vec<f64> {
        (0..d)
            .map(|i| ((i as f64 + 1.0) * 0.7 + seed).cos())
            .collect()
    }

    fn default_cfg(n_blocks: usize) -> BcdConfig {
        BcdConfig {
            n_blocks,
            max_iter: 500,
            tol: 1.0e-10,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::ExactQuadratic,
        }
    }

    #[test]
    fn bcd_quadratic_converges_to_solve_dense() {
        let d = 6usize;
        let a = make_spd(d, 0.3);
        let b = rhs_vec(d, 0.1);
        let x_ref = direct_solve(&a, d, &b).expect("ref solve");
        let blocks = vec![vec![0usize, 1], vec![2usize, 3], vec![4usize, 5]];
        let cfg = default_cfg(3);
        let mut rng = LcgRng::new(11);
        let state = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng)
            .expect("bcd ok");
        for i in 0..d {
            assert!(
                (state.x[i] - x_ref[i]).abs() < 1.0e-7,
                "coord {i}: bcd={} ref={}",
                state.x[i],
                x_ref[i]
            );
        }
        assert!(state.iter <= cfg.max_iter);
    }

    #[test]
    fn single_block_solves_in_one_sweep() {
        let d = 5usize;
        let a = make_spd(d, 0.7);
        let b = rhs_vec(d, 0.4);
        let blocks = vec![(0..d).collect::<Vec<_>>()];
        let cfg = BcdConfig {
            n_blocks: 1,
            max_iter: 50,
            tol: 1.0e-12,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::ExactQuadratic,
        };
        let mut rng = LcgRng::new(2);
        let state = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng)
            .expect("ok");
        // Single block with exact solve closes in 1 sweep (delta on sweep 2 is 0).
        assert_eq!(
            state.iter, 2,
            "single-block exact should close after 1 update"
        );
        let x_ref = direct_solve(&a, d, &b).expect("ref");
        for i in 0..d {
            assert!((state.x[i] - x_ref[i]).abs() < 1.0e-10);
        }
    }

    #[test]
    fn cyclic_and_random_both_converge() {
        let d = 4usize;
        let a = make_spd(d, 0.2);
        let b = rhs_vec(d, 0.9);
        let x_ref = direct_solve(&a, d, &b).expect("ref");
        let blocks = vec![vec![0usize, 1], vec![2usize, 3]];

        let mut cfg_c = default_cfg(2);
        cfg_c.sweep = BcdSweep::Cyclic;
        cfg_c.max_iter = 400;
        let mut rng_c = LcgRng::new(123);
        let state_c =
            block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg_c, &mut rng_c)
                .expect("cyclic ok");
        for i in 0..d {
            assert!((state_c.x[i] - x_ref[i]).abs() < 1.0e-6);
        }

        let mut cfg_r = default_cfg(2);
        cfg_r.sweep = BcdSweep::Random;
        cfg_r.max_iter = 800;
        let mut rng_r = LcgRng::new(456);
        let state_r =
            block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg_r, &mut rng_r)
                .expect("random ok");
        for i in 0..d {
            assert!((state_r.x[i] - x_ref[i]).abs() < 1.0e-4);
        }
    }

    #[test]
    fn exact_inner_solver_converges() {
        let d = 4usize;
        let a = make_spd(d, 1.1);
        let b = rhs_vec(d, 0.3);
        let blocks = vec![vec![0, 1], vec![2, 3]];
        let cfg = default_cfg(2);
        let mut rng = LcgRng::new(7);
        let state = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng)
            .expect("exact ok");
        let x_ref = direct_solve(&a, d, &b).expect("ref");
        for i in 0..d {
            assert!((state.x[i] - x_ref[i]).abs() < 1.0e-7);
        }
    }

    #[test]
    fn gradient_descent_inner_solver_converges() {
        let d = 4usize;
        let a = make_spd(d, 0.5);
        let b = rhs_vec(d, 0.2);
        // Estimate largest eigenvalue cheaply via row-sum upper bound.
        let mut lmax: f64 = 0.0;
        for i in 0..d {
            let mut s = 0.0_f64;
            for j in 0..d {
                s += a[i * d + j].abs();
            }
            if s > lmax {
                lmax = s;
            }
        }
        let lr = 0.9 / lmax;
        let cfg = BcdConfig {
            n_blocks: 2,
            max_iter: 2000,
            tol: 1.0e-9,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::GradientDescent {
                inner_iter: 200,
                inner_lr: lr,
            },
        };
        let blocks = vec![vec![0, 1], vec![2, 3]];
        let mut rng = LcgRng::new(31);
        let state = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng)
            .expect("gd inner ok");
        let x_ref = direct_solve(&a, d, &b).expect("ref");
        for i in 0..d {
            assert!(
                (state.x[i] - x_ref[i]).abs() < 1.0e-3,
                "coord {i}: bcd={} ref={}",
                state.x[i],
                x_ref[i]
            );
        }
    }

    #[test]
    fn objective_history_non_increasing_exact() {
        let d = 6usize;
        let a = make_spd(d, 0.42);
        let b = rhs_vec(d, 0.17);
        let blocks = vec![vec![0, 1], vec![2, 3], vec![4, 5]];
        let cfg = BcdConfig {
            n_blocks: 3,
            max_iter: 80,
            tol: 1.0e-12,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::ExactQuadratic,
        };
        let mut rng = LcgRng::new(99);
        let state = block_coord_descent_quadratic(&a, &b, &vec![3.0; d], &blocks, &cfg, &mut rng)
            .expect("ok");
        assert!(!state.objective_history.is_empty());
        for w in state.objective_history.windows(2) {
            // Floor by 1e-10 to absorb FP roundoff.
            assert!(
                w[1] <= w[0] + 1.0e-10,
                "objective increased: prev={} next={}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn random_sweep_deterministic_given_seed() {
        let d = 4usize;
        let a = make_spd(d, 0.6);
        let b = rhs_vec(d, 0.8);
        let blocks = vec![vec![0, 1], vec![2, 3]];
        let cfg = BcdConfig {
            n_blocks: 2,
            max_iter: 50,
            tol: 1.0e-14,
            sweep: BcdSweep::Random,
            inner: InnerSolver::ExactQuadratic,
        };
        let mut rng1 = LcgRng::new(2024);
        let s1 = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng1)
            .expect("ok");
        let mut rng2 = LcgRng::new(2024);
        let s2 = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng2)
            .expect("ok");
        assert_eq!(s1.iter, s2.iter);
        for i in 0..d {
            assert!((s1.x[i] - s2.x[i]).abs() < 1.0e-15);
        }
    }

    #[test]
    fn tol_respected_stops_iteration() {
        let d = 4usize;
        let a = make_spd(d, 0.1);
        let b = rhs_vec(d, 0.3);
        let blocks = vec![vec![0, 1], vec![2, 3]];
        let cfg = BcdConfig {
            n_blocks: 2,
            max_iter: 1000,
            tol: 1.0e-6,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::ExactQuadratic,
        };
        let mut rng = LcgRng::new(8);
        let state = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng)
            .expect("ok");
        assert!(state.iter < cfg.max_iter, "should stop early via tol");
    }

    #[test]
    fn max_iter_upper_bound() {
        let d = 3usize;
        let a = make_spd(d, 0.5);
        let b = rhs_vec(d, 0.5);
        let blocks = vec![vec![0], vec![1], vec![2]];
        let cfg = BcdConfig {
            n_blocks: 3,
            max_iter: 4,
            tol: 1.0e-12,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::ExactQuadratic,
        };
        let mut rng = LcgRng::new(0);
        let state = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng)
            .expect("ok");
        assert!(state.iter <= cfg.max_iter);
    }

    #[test]
    fn err_n_blocks_zero() {
        let d = 2usize;
        let a = make_spd(d, 0.1);
        let b = rhs_vec(d, 0.1);
        let blocks: Vec<Vec<usize>> = vec![vec![0, 1]];
        let cfg = BcdConfig {
            n_blocks: 0,
            max_iter: 10,
            tol: 1.0e-6,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::ExactQuadratic,
        };
        let mut rng = LcgRng::new(1);
        let res = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng);
        assert!(res.is_err());
    }

    #[test]
    fn err_blocks_empty() {
        let d = 2usize;
        let a = make_spd(d, 0.1);
        let b = rhs_vec(d, 0.1);
        let blocks: Vec<Vec<usize>> = vec![];
        let cfg = BcdConfig {
            n_blocks: 0,
            max_iter: 10,
            tol: 1.0e-6,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::ExactQuadratic,
        };
        let mut rng = LcgRng::new(1);
        let res = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng);
        assert!(res.is_err());
    }

    #[test]
    fn err_duplicate_index() {
        let d = 3usize;
        let a = make_spd(d, 0.1);
        let b = rhs_vec(d, 0.1);
        let blocks = vec![vec![0, 1], vec![1, 2]];
        let cfg = BcdConfig {
            n_blocks: 2,
            max_iter: 10,
            tol: 1.0e-6,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::ExactQuadratic,
        };
        let mut rng = LcgRng::new(1);
        let res = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng);
        assert!(res.is_err());
    }

    #[test]
    fn err_missing_index() {
        let d = 4usize;
        let a = make_spd(d, 0.1);
        let b = rhs_vec(d, 0.1);
        let blocks = vec![vec![0, 1], vec![2]];
        let cfg = BcdConfig {
            n_blocks: 2,
            max_iter: 10,
            tol: 1.0e-6,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::ExactQuadratic,
        };
        let mut rng = LcgRng::new(1);
        let res = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng);
        assert!(res.is_err());
    }

    #[test]
    fn err_tol_nonpositive() {
        let d = 2usize;
        let a = make_spd(d, 0.1);
        let b = rhs_vec(d, 0.1);
        let blocks = vec![vec![0], vec![1]];
        let cfg = BcdConfig {
            n_blocks: 2,
            max_iter: 10,
            tol: 0.0,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::ExactQuadratic,
        };
        let mut rng = LcgRng::new(1);
        let res = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng);
        assert!(res.is_err());
    }

    #[test]
    fn err_max_iter_zero() {
        let d = 2usize;
        let a = make_spd(d, 0.1);
        let b = rhs_vec(d, 0.1);
        let blocks = vec![vec![0], vec![1]];
        let cfg = BcdConfig {
            n_blocks: 2,
            max_iter: 0,
            tol: 1.0e-6,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::ExactQuadratic,
        };
        let mut rng = LcgRng::new(1);
        let res = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng);
        assert!(res.is_err());
    }

    #[test]
    fn err_initial_x_wrong_dim() {
        let d = 3usize;
        let a = make_spd(d, 0.1);
        let b = rhs_vec(d, 0.1);
        let blocks = vec![vec![0], vec![1, 2]];
        let cfg = BcdConfig {
            n_blocks: 2,
            max_iter: 10,
            tol: 1.0e-6,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::ExactQuadratic,
        };
        let mut rng = LcgRng::new(1);
        // initial_x has dim 2 ≠ d = 3
        let res = block_coord_descent_quadratic(&a, &b, &vec![0.0; 2], &blocks, &cfg, &mut rng);
        assert!(res.is_err());
    }

    #[test]
    fn err_a_wrong_size() {
        let d = 3usize;
        let a = vec![1.0_f64; d * d + 1];
        let b = rhs_vec(d, 0.1);
        let blocks = vec![vec![0], vec![1, 2]];
        let cfg = BcdConfig {
            n_blocks: 2,
            max_iter: 10,
            tol: 1.0e-6,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::ExactQuadratic,
        };
        let mut rng = LcgRng::new(1);
        let res = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng);
        assert!(res.is_err());
    }

    #[test]
    fn err_b_wrong_size() {
        let d = 3usize;
        let a = make_spd(d, 0.1);
        let b = vec![1.0_f64; d - 1];
        let blocks = vec![vec![0], vec![1, 2]];
        let cfg = BcdConfig {
            n_blocks: 2,
            max_iter: 10,
            tol: 1.0e-6,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::ExactQuadratic,
        };
        let mut rng = LcgRng::new(1);
        let res = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng);
        assert!(res.is_err());
    }

    #[test]
    fn err_gd_inner_iter_zero() {
        let d = 2usize;
        let a = make_spd(d, 0.1);
        let b = rhs_vec(d, 0.1);
        let blocks = vec![vec![0], vec![1]];
        let cfg = BcdConfig {
            n_blocks: 2,
            max_iter: 10,
            tol: 1.0e-6,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::GradientDescent {
                inner_iter: 0,
                inner_lr: 0.1,
            },
        };
        let mut rng = LcgRng::new(1);
        let res = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng);
        assert!(res.is_err());
    }

    #[test]
    fn err_gd_inner_lr_nonpositive() {
        let d = 2usize;
        let a = make_spd(d, 0.1);
        let b = rhs_vec(d, 0.1);
        let blocks = vec![vec![0], vec![1]];
        let cfg = BcdConfig {
            n_blocks: 2,
            max_iter: 10,
            tol: 1.0e-6,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::GradientDescent {
                inner_iter: 5,
                inner_lr: -0.1,
            },
        };
        let mut rng = LcgRng::new(1);
        let res = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng);
        assert!(res.is_err());
    }

    #[test]
    fn generic_api_runs_with_user_step() {
        // A custom non-quadratic separable example:
        //   f(x) = Σ (x_i − target_i)^4
        // optimum is x = target. Block step is a few inner GD steps.
        let d = 4usize;
        let target = vec![1.0, -2.0, 0.5, 3.0];
        let blocks = vec![vec![0, 1], vec![2, 3]];
        let cfg = BcdConfig {
            n_blocks: 2,
            max_iter: 1000,
            tol: 1.0e-8,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::ExactQuadratic, // unused by generic API
        };
        let target_obj = target.clone();
        let target_step = target.clone();
        let mut rng = LcgRng::new(5);
        let state = block_coord_descent(
            &vec![0.0; d],
            &blocks,
            |x| {
                x.iter()
                    .zip(target_obj.iter())
                    .map(|(xi, ti)| (xi - ti).powi(4))
                    .sum()
            },
            |x, _b_idx, block| {
                // 100 GD steps with step 0.05 on the block coords.
                let mut x_b: Vec<f64> = block.iter().map(|&i| x[i]).collect();
                for _ in 0..100 {
                    for (local_i, &gi) in block.iter().enumerate() {
                        let g = 4.0 * (x_b[local_i] - target_step[gi]).powi(3);
                        x_b[local_i] -= 0.05 * g;
                    }
                }
                Ok(x_b)
            },
            &cfg,
            &mut rng,
        )
        .expect("generic ok");
        for i in 0..d {
            assert!((state.x[i] - target[i]).abs() < 1.0e-2);
        }
    }

    #[test]
    fn objective_history_length_matches_iter() {
        let d = 4usize;
        let a = make_spd(d, 0.3);
        let b = rhs_vec(d, 0.4);
        let blocks = vec![vec![0, 1], vec![2, 3]];
        let cfg = BcdConfig {
            n_blocks: 2,
            max_iter: 10,
            tol: 1.0e-14,
            sweep: BcdSweep::Cyclic,
            inner: InnerSolver::ExactQuadratic,
        };
        let mut rng = LcgRng::new(1);
        let state = block_coord_descent_quadratic(&a, &b, &vec![0.0; d], &blocks, &cfg, &mut rng)
            .expect("ok");
        assert_eq!(state.objective_history.len(), state.iter);
    }
}
