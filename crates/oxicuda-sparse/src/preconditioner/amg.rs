//! Algebraic multigrid via smoothed aggregation (Vaněk-Mandel-Brezina).
//!
//! This module builds a multigrid hierarchy for a symmetric, (near) positive-
//! definite sparse matrix `A` purely from its algebraic entries -- no geometric
//! grid information is required. The setup phase coarsens `A` by aggregating
//! strongly-connected nodes, builds a tentative piecewise-constant interpolation
//! that exactly represents the near-null-space (the constant vector), smooths it
//! with one Jacobi step to improve interpolation quality, and forms the coarse
//! operator through the Galerkin triple product `A_c = Pᵀ A P`. The solve phase
//! applies V-cycles: weighted-Jacobi pre/post smoothing on each level and a
//! direct dense solve on the coarsest level.
//!
//! ## Setup (smoothed aggregation)
//!
//! 1. **Strength of connection.** Nodes `i` and `j` are strongly connected when
//!    `|A_ij| ≥ θ · sqrt(|A_ii| · |A_jj|)`, with `θ = strength_threshold`.
//! 2. **Aggregation (greedy two-pass).** Pass 1 forms a new aggregate
//!    `{i} ∪ strong_neighbours(i)` for every node `i` whose strong neighbours
//!    are all still free. Pass 2 attaches each leftover node to an adjacent
//!    existing aggregate, or makes it a singleton.
//! 3. **Tentative prolongator `P0`.** Column `g` is the constant near-null-space
//!    vector restricted to aggregate `g`, normalised to unit Euclidean norm, so
//!    `range(P0)` contains the constant and `P0` has orthonormal columns.
//! 4. **Smoothed prolongator.** `P = (I − ω D⁻¹ A) P0` with
//!    `ω = smooth_omega / ρ̂(D⁻¹A)`, where `ρ̂` is estimated by power iteration.
//! 5. **Galerkin coarse operator.** `A_c = Pᵀ A P`, computed with the host
//!    sparse-sparse product.
//!
//! The recursion stops when the operator order drops to `max_coarse` or the
//! level count reaches `max_levels`.

use crate::error::{SparseError, SparseResult};
use crate::host_csr::{HostCsr, dense_solve};

/// A single level of the multigrid hierarchy.
pub struct AmgLevel {
    /// The (fine) operator on this level.
    pub a: HostCsr,
    /// The prolongation (interpolation) operator to this level from the next
    /// coarser level: `fine = P · coarse`.
    pub p: HostCsr,
    /// The restriction operator `R = Pᵀ`.
    pub r: HostCsr,
    /// The diagonal of `a`, cached for weighted-Jacobi smoothing.
    pub diag: Vec<f64>,
}

/// A complete smoothed-aggregation multigrid hierarchy.
pub struct AmgHierarchy {
    /// Levels ordered finest-first. Each level carries its operator and the
    /// transfer operators connecting it to the next coarser level.
    pub levels: Vec<AmgLevel>,
    /// The coarsest operator, solved directly during the V-cycle.
    pub coarse: HostCsr,
    /// Number of weighted-Jacobi pre-smoothing sweeps applied per level.
    pub pre_sweeps: usize,
    /// Number of weighted-Jacobi post-smoothing sweeps applied per level.
    pub post_sweeps: usize,
}

/// Tunable parameters for the smoothed-aggregation setup and the V-cycle.
pub struct AmgOptions {
    /// Strength-of-connection threshold `θ` (e.g. `0.08`).
    pub strength_threshold: f64,
    /// Maximum number of levels (including the finest) before forcing a coarse
    /// solve.
    pub max_levels: usize,
    /// Stop coarsening once the operator order is `≤ max_coarse`.
    pub max_coarse: usize,
    /// Prolongator-smoothing weight numerator; the effective Jacobi weight is
    /// `smooth_omega / ρ̂(D⁻¹A)` (e.g. `smooth_omega = 4/3`).
    pub smooth_omega: f64,
    /// Number of pre-smoothing sweeps per level.
    pub pre_sweeps: usize,
    /// Number of post-smoothing sweeps per level.
    pub post_sweeps: usize,
}

impl Default for AmgOptions {
    fn default() -> Self {
        Self {
            strength_threshold: 0.08,
            max_levels: 25,
            max_coarse: 16,
            smooth_omega: 4.0 / 3.0,
            pre_sweeps: 2,
            post_sweeps: 2,
        }
    }
}

/// Builds a smoothed-aggregation multigrid hierarchy for `a`.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `a` is not square,
/// [`SparseError::InvalidArgument`] if `a` is empty or an option is out of
/// range, [`SparseError::SingularMatrix`] if a zero diagonal is encountered, or
/// propagates errors from the Galerkin product.
pub fn amg_setup(a: &HostCsr, opts: &AmgOptions) -> SparseResult<AmgHierarchy> {
    if a.nrows != a.ncols {
        return Err(SparseError::DimensionMismatch(format!(
            "AMG requires a square matrix, got {}x{}",
            a.nrows, a.ncols
        )));
    }
    if a.nrows == 0 {
        return Err(SparseError::InvalidArgument(
            "cannot build a hierarchy for an empty matrix".to_string(),
        ));
    }
    if opts.max_levels == 0 {
        return Err(SparseError::InvalidArgument(
            "max_levels must be >= 1".to_string(),
        ));
    }

    let mut levels: Vec<AmgLevel> = Vec::new();
    let mut current = a.clone();

    loop {
        let n = current.nrows;
        // Stop if we are small enough or have reached the level cap. The number
        // of stored levels is the number of coarsening steps; the finest level
        // is level 0.
        if n <= opts.max_coarse || levels.len() + 1 >= opts.max_levels {
            break;
        }

        // 1. Strength of connection.
        let strength = strength_of_connection(&current, opts.strength_threshold);

        // 2. Aggregation.
        let (aggregates, num_agg) = aggregate(&current, &strength);
        // Guard against failure to coarsen (e.g. every node a singleton): if no
        // reduction occurred, stop to avoid an infinite hierarchy.
        if num_agg == 0 || num_agg >= n {
            break;
        }

        // 3. Tentative prolongator P0 (column-normalised constant).
        let p0 = tentative_prolongator(&aggregates, num_agg, n);

        // 4. Smoothed prolongator P = (I - omega D^{-1} A) P0.
        let diag = current.diagonal();
        for &d in &diag {
            if d == 0.0 {
                return Err(SparseError::SingularMatrix);
            }
        }
        let rho = spectral_radius_estimate(&current, &diag);
        let omega = opts.smooth_omega / rho.max(1e-30);
        let p = smooth_prolongator(&current, &diag, &p0, omega)?;

        // 5. Galerkin coarse operator A_c = Pᵀ A P.
        let r = p.transpose();
        let ap = current.matmul(&p)?;
        let a_coarse = r.matmul(&ap)?;

        let level = AmgLevel {
            a: current.clone(),
            p,
            r,
            diag,
        };
        levels.push(level);

        current = a_coarse;
    }

    Ok(AmgHierarchy {
        levels,
        coarse: current,
        pre_sweeps: opts.pre_sweeps.max(1),
        post_sweeps: opts.post_sweeps.max(1),
    })
}

/// Applies one multigrid V-cycle, updating `x` in place toward the solution of
/// `A x = b` on the finest level.
///
/// The vectors `b` and `x` must have length equal to the order of the finest
/// operator. Out-of-range lengths are clamped by the smoother bounds and leave
/// the unaffected entries untouched.
pub fn amg_v_cycle(h: &AmgHierarchy, b: &[f64], x: &mut [f64]) {
    v_cycle_recursive(h, 0, b, x);
}

/// Recursive V-cycle on level `level`. `b` is the right-hand side for the
/// operator at `level`, `x` the current iterate (updated in place).
fn v_cycle_recursive(h: &AmgHierarchy, level: usize, b: &[f64], x: &mut [f64]) {
    if level >= h.levels.len() {
        // Coarsest level: direct dense solve.
        let n = h.coarse.nrows;
        if n == 0 {
            return;
        }
        let dense = h.coarse.to_dense();
        match dense_solve(&dense, b, n) {
            Ok(sol) => {
                x[..n].copy_from_slice(&sol[..n]);
            }
            Err(_) => {
                // Fall back to a few Jacobi sweeps if the coarse solve is
                // singular (e.g. a pure-Neumann constant null space).
                let diag = h.coarse.diagonal();
                jacobi_sweeps(&h.coarse, &diag, b, x, 30, 2.0 / 3.0);
            }
        }
        return;
    }

    let lvl = &h.levels[level];

    // Effective smoothing weight: weighted Jacobi with omega = 2/3.
    let omega = 2.0 / 3.0;

    // Pre-smoothing.
    jacobi_sweeps(&lvl.a, &lvl.diag, b, x, h.pre_sweeps, omega);

    // Residual r = b - A x.
    let ax = lvl.a.matvec(x);
    let mut residual = vec![0.0f64; b.len()];
    for i in 0..b.len() {
        residual[i] = b[i] - ax[i];
    }

    // Restrict to the coarse level: r_c = R r.
    let r_coarse = lvl.r.matvec(&residual);
    let coarse_n = r_coarse.len();
    let mut e_coarse = vec![0.0f64; coarse_n];

    // Recurse to solve A_c e_c = r_c.
    v_cycle_recursive(h, level + 1, &r_coarse, &mut e_coarse);

    // Prolong the correction and apply: x += P e_c.
    let correction = lvl.p.matvec(&e_coarse);
    for i in 0..x.len().min(correction.len()) {
        x[i] += correction[i];
    }

    // Post-smoothing.
    jacobi_sweeps(&lvl.a, &lvl.diag, b, x, h.post_sweeps, omega);
}

/// Iteratively solves `A x = b` to a relative residual tolerance using V-cycles.
///
/// Returns `(x, iters, final_rel_residual)`.
///
/// # Errors
///
/// Propagates setup errors from [`amg_setup`]. If the right-hand side is the
/// zero vector the solution is the zero vector and zero iterations are reported.
pub fn amg_solve(
    a: &HostCsr,
    b: &[f64],
    opts: &AmgOptions,
    max_iter: usize,
    tol: f64,
) -> SparseResult<(Vec<f64>, usize, f64)> {
    if b.len() != a.nrows {
        return Err(SparseError::DimensionMismatch(format!(
            "rhs length {} must equal matrix order {}",
            b.len(),
            a.nrows
        )));
    }
    let hierarchy = amg_setup(a, opts)?;

    let b_norm = norm(b);
    if b_norm == 0.0 {
        return Ok((vec![0.0; a.nrows], 0, 0.0));
    }

    let mut x = vec![0.0f64; a.nrows];
    let mut final_rel = 1.0;
    let mut iters = 0;
    for it in 0..max_iter {
        iters = it + 1;
        amg_v_cycle(&hierarchy, b, &mut x);
        let ax = a.matvec(&x);
        let mut res = vec![0.0f64; b.len()];
        for i in 0..b.len() {
            res[i] = b[i] - ax[i];
        }
        final_rel = norm(&res) / b_norm;
        if final_rel < tol {
            break;
        }
    }

    Ok((x, iters, final_rel))
}

// ---------------------------------------------------------------------------
// Setup building blocks.
// ---------------------------------------------------------------------------

/// Per-node list of strongly-connected neighbours (excluding the node itself).
type Strength = Vec<Vec<usize>>;

/// Computes the strength-of-connection graph: `i ~ j` iff
/// `|A_ij| ≥ θ · sqrt(|A_ii| · |A_jj|)`.
fn strength_of_connection(a: &HostCsr, theta: f64) -> Strength {
    let n = a.nrows;
    let diag = a.diagonal();
    let mut strength: Strength = vec![Vec::new(); n];
    for i in 0..n {
        let start = a.row_ptr[i];
        let end = a.row_ptr[i + 1];
        let dii = diag[i].abs();
        for kk in start..end {
            let j = a.col_indices[kk];
            if j == i {
                continue;
            }
            let aij = a.values[kk].abs();
            let djj = diag[j].abs();
            let bound = theta * (dii * djj).sqrt();
            if aij >= bound && aij > 0.0 {
                strength[i].push(j);
            }
        }
    }
    strength
}

/// Greedy two-pass aggregation. Returns `(aggregate_of_node, num_aggregates)`.
fn aggregate(a: &HostCsr, strength: &Strength) -> (Vec<usize>, usize) {
    let n = a.nrows;
    const UNASSIGNED: usize = usize::MAX;
    let mut agg = vec![UNASSIGNED; n];
    let mut num_agg = 0usize;

    // Pass 1: form root aggregates from nodes whose strong neighbourhood is
    // entirely free.
    for i in 0..n {
        if agg[i] != UNASSIGNED {
            continue;
        }
        let all_free = strength[i].iter().all(|&j| agg[j] == UNASSIGNED);
        if !all_free {
            continue;
        }
        // Create a new aggregate containing i and its strong neighbours.
        let g = num_agg;
        num_agg += 1;
        agg[i] = g;
        for &j in &strength[i] {
            agg[j] = g;
        }
    }

    // Pass 2: sweep remaining nodes, attaching each to the aggregate of a
    // strong neighbour (preferring the most strongly-connected one). Nodes with
    // no aggregated strong neighbour become singletons.
    for i in 0..n {
        if agg[i] != UNASSIGNED {
            continue;
        }
        // Find the best already-aggregated strong neighbour.
        let mut best: Option<usize> = None;
        let mut best_mag = 0.0f64;
        let start = a.row_ptr[i];
        let end = a.row_ptr[i + 1];
        for &j in &strength[i] {
            if agg[j] != UNASSIGNED {
                // Magnitude of A_ij for tie-breaking.
                let mut mag = 0.0;
                for kk in start..end {
                    if a.col_indices[kk] == j {
                        mag = a.values[kk].abs();
                        break;
                    }
                }
                if best.is_none() || mag > best_mag {
                    best = Some(agg[j]);
                    best_mag = mag;
                }
            }
        }
        match best {
            Some(g) => agg[i] = g,
            None => {
                let g = num_agg;
                num_agg += 1;
                agg[i] = g;
            }
        }
    }

    (agg, num_agg)
}

/// Builds the tentative prolongator `P0` (`n × num_agg`) whose column `g` is the
/// constant near-null-space restricted to aggregate `g`, normalised to unit
/// Euclidean norm. With this scaling `P0` has orthonormal columns and its range
/// contains the constant vector.
fn tentative_prolongator(agg: &[usize], num_agg: usize, n: usize) -> HostCsr {
    // Aggregate sizes for normalisation (constant column has norm sqrt(size)).
    let mut sizes = vec![0usize; num_agg];
    for &g in agg {
        sizes[g] += 1;
    }
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_indices = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        let g = agg[i];
        let norm = (sizes[g] as f64).sqrt();
        col_indices.push(g);
        values.push(1.0 / norm);
        row_ptr[i + 1] = col_indices.len();
    }
    HostCsr {
        nrows: n,
        ncols: num_agg,
        row_ptr,
        col_indices,
        values,
    }
}

/// Forms the smoothed prolongator `P = (I − ω D⁻¹ A) P0`.
///
/// Computed as `P = P0 − ω D⁻¹ (A P0)` row by row, where `D⁻¹` scales row `i`
/// by `1 / A_ii`.
fn smooth_prolongator(
    a: &HostCsr,
    diag: &[f64],
    p0: &HostCsr,
    omega: f64,
) -> SparseResult<HostCsr> {
    // A * P0 (sparse-sparse).
    let ap0 = a.matmul(p0)?;
    let n = a.nrows;
    let ncols = p0.ncols;

    // Accumulate each row of P = P0 - omega * Dinv * (A P0) in a dense buffer.
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_indices: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();

    let mut accum = vec![0.0f64; ncols];
    let mut touched: Vec<usize> = Vec::new();
    let mut is_touched = vec![false; ncols];

    for i in 0..n {
        let dinv = 1.0 / diag[i];

        // P0 contribution (row i of P0 has a single entry).
        let p0_s = p0.row_ptr[i];
        let p0_e = p0.row_ptr[i + 1];
        for kk in p0_s..p0_e {
            let c = p0.col_indices[kk];
            if !is_touched[c] {
                is_touched[c] = true;
                touched.push(c);
            }
            accum[c] += p0.values[kk];
        }

        // -omega * Dinv * (A P0) contribution.
        let ap_s = ap0.row_ptr[i];
        let ap_e = ap0.row_ptr[i + 1];
        for kk in ap_s..ap_e {
            let c = ap0.col_indices[kk];
            if !is_touched[c] {
                is_touched[c] = true;
                touched.push(c);
            }
            accum[c] -= omega * dinv * ap0.values[kk];
        }

        touched.sort_unstable();
        for &c in &touched {
            let v = accum[c];
            if v != 0.0 {
                col_indices.push(c);
                values.push(v);
            }
            accum[c] = 0.0;
            is_touched[c] = false;
        }
        touched.clear();
        row_ptr[i + 1] = col_indices.len();
    }

    Ok(HostCsr {
        nrows: n,
        ncols,
        row_ptr,
        col_indices,
        values,
    })
}

/// Estimates the spectral radius `ρ(D⁻¹A)` by a few power iterations starting
/// from a deterministic vector. `D⁻¹A` is applied implicitly.
fn spectral_radius_estimate(a: &HostCsr, diag: &[f64]) -> f64 {
    let n = a.nrows;
    if n == 0 {
        return 1.0;
    }
    // Deterministic, non-constant starting vector.
    let mut v: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64 * 0.3).collect();
    normalize(&mut v);

    let mut lambda = 1.0;
    for _ in 0..15 {
        // w = D^{-1} A v.
        let av = a.matvec(&v);
        let mut w = vec![0.0f64; n];
        for i in 0..n {
            w[i] = av[i] / diag[i];
        }
        let nrm = norm(&w);
        if nrm < 1e-300 {
            break;
        }
        // Rayleigh-style estimate.
        lambda = dot(&v, &w);
        for i in 0..n {
            v[i] = w[i] / nrm;
        }
    }
    // For an SPD M-matrix, D^{-1}A has positive spectrum; guard the estimate.
    lambda.abs().max(1.0)
}

// ---------------------------------------------------------------------------
// Smoother and vector helpers.
// ---------------------------------------------------------------------------

/// Applies `sweeps` weighted-Jacobi sweeps `x ← x + ω D⁻¹ (b − A x)` in place.
fn jacobi_sweeps(a: &HostCsr, diag: &[f64], b: &[f64], x: &mut [f64], sweeps: usize, omega: f64) {
    let n = a.nrows;
    for _ in 0..sweeps {
        let ax = a.matvec(x);
        for i in 0..n {
            if diag[i] != 0.0 {
                x[i] += omega * (b[i] - ax[i]) / diag[i];
            }
        }
    }
}

/// Euclidean inner product.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Euclidean norm.
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Scales a vector to unit Euclidean norm (no-op for the zero vector).
fn normalize(v: &mut [f64]) {
    let nrm = norm(v);
    if nrm > 1e-300 {
        let inv = 1.0 / nrm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host_csr::{laplacian_1d, laplacian_2d};

    #[test]
    fn poisson_1d_converges() {
        // 1D Poisson: recover a known solution in few V-cycles.
        let n = 64;
        let a = laplacian_1d(n);
        let x_star: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.1).sin() + 1.0).collect();
        let b = a.matvec(&x_star);
        let opts = AmgOptions::default();
        let (x, iters, rel) = amg_solve(&a, &b, &opts, 60, 1e-8).expect("solve");
        assert!(rel < 1e-8, "did not converge: rel = {rel}");
        assert!(iters < 20, "took too many V-cycles: {iters}");
        for i in 0..n {
            assert!(
                (x[i] - x_star[i]).abs() < 1e-6,
                "solution mismatch at {i}: {} vs {}",
                x[i],
                x_star[i]
            );
        }
    }

    #[test]
    fn poisson_2d_geometric_residual_drop() {
        // 2D Poisson on a 16x16 grid: residual must drop geometrically.
        let g = 16;
        let a = laplacian_2d(g, g);
        let n = g * g;
        let x_star: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64).collect();
        let b = a.matvec(&x_star);
        let opts = AmgOptions::default();
        let hierarchy = amg_setup(&a, &opts).expect("setup");

        let b_norm = norm(&b);
        let mut x = vec![0.0f64; n];
        let mut residuals = Vec::new();
        for _ in 0..6 {
            amg_v_cycle(&hierarchy, &b, &mut x);
            let ax = a.matvec(&x);
            let mut res = vec![0.0; n];
            for i in 0..n {
                res[i] = b[i] - ax[i];
            }
            residuals.push(norm(&res) / b_norm);
        }
        // Geometric drop: each cycle reduces the residual by at least 0.7x.
        for w in residuals.windows(2) {
            if w[0] > 1e-12 {
                assert!(
                    w[1] < 0.7 * w[0],
                    "residual did not drop geometrically: {} -> {}",
                    w[0],
                    w[1]
                );
            }
        }
    }

    #[test]
    fn mesh_independence() {
        // Iteration counts to reach tol stay bounded as n grows -- the
        // hallmark of multigrid.
        let opts = AmgOptions::default();
        let tol = 1e-8;

        let a64 = laplacian_1d(64);
        let xs64: Vec<f64> = (0..64).map(|i| 1.0 + (i % 3) as f64).collect();
        let b64 = a64.matvec(&xs64);
        let (_x, iters64, _r) = amg_solve(&a64, &b64, &opts, 50, tol).expect("solve64");

        let g = 16;
        let a256 = laplacian_2d(g, g);
        let xs256: Vec<f64> = (0..256).map(|i| 1.0 + (i % 4) as f64).collect();
        let b256 = a256.matvec(&xs256);
        let (_x2, iters256, _r2) = amg_solve(&a256, &b256, &opts, 50, tol).expect("solve256");

        assert!(iters64 < 25, "n=64 took {iters64} cycles");
        assert!(iters256 < 25, "n=256 took {iters256} cycles");
    }

    #[test]
    fn galerkin_coarse_is_symmetric() {
        let a = laplacian_2d(8, 8);
        let opts = AmgOptions::default();
        let h = amg_setup(&a, &opts).expect("setup");
        // Check every coarse operator (including the final one) is symmetric.
        for lvl in &h.levels {
            let ac = if lvl.a.nrows == a.nrows {
                // skip the finest; check the coarse via Galerkin on this level
                let ap = lvl.a.matmul(&lvl.p).expect("ap");
                lvl.r.matmul(&ap).expect("rap")
            } else {
                lvl.a.clone()
            };
            for i in 0..ac.nrows {
                for j in 0..ac.ncols {
                    let aij = ac.get(i, j).unwrap_or(0.0);
                    let aji = ac.get(j, i).unwrap_or(0.0);
                    assert!(
                        (aij - aji).abs() < 1e-9,
                        "coarse operator not symmetric at ({i},{j}): {aij} vs {aji}"
                    );
                }
            }
        }
        // Final coarse operator symmetric too.
        let ac = &h.coarse;
        for i in 0..ac.nrows {
            for j in 0..ac.ncols {
                assert!((ac.get(i, j).unwrap_or(0.0) - ac.get(j, i).unwrap_or(0.0)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn tentative_prolongator_preserves_constant() {
        // P0 · 1_coarse == 1_fine exactly, where 1_coarse[g] = sqrt(size(g))
        // because the columns are unit-normalised constants.
        let a = laplacian_2d(6, 6);
        let opts = AmgOptions::default();
        let strength = strength_of_connection(&a, opts.strength_threshold);
        let (agg, num_agg) = aggregate(&a, &strength);
        let p0 = tentative_prolongator(&agg, num_agg, a.nrows);
        // Build the coarse constant representation.
        let mut sizes = vec![0usize; num_agg];
        for &g in &agg {
            sizes[g] += 1;
        }
        let one_coarse: Vec<f64> = sizes.iter().map(|&s| (s as f64).sqrt()).collect();
        let one_fine = p0.matvec(&one_coarse);
        for &v in &one_fine {
            assert!(
                (v - 1.0).abs() < 1e-12,
                "P0 does not preserve constant: {v}"
            );
        }
    }

    #[test]
    fn smoothed_prolongator_approximates_constant() {
        // The smoothed P interpolates the constant up to the boundary smoothing
        // term ω D⁻¹ A 1, which is zero in the interior of the Laplacian.
        let a = laplacian_1d(32);
        let opts = AmgOptions::default();
        let strength = strength_of_connection(&a, opts.strength_threshold);
        let (agg, num_agg) = aggregate(&a, &strength);
        let p0 = tentative_prolongator(&agg, num_agg, a.nrows);
        let diag = a.diagonal();
        let rho = spectral_radius_estimate(&a, &diag);
        let omega = opts.smooth_omega / rho;
        let p = smooth_prolongator(&a, &diag, &p0, omega).expect("smooth");

        let mut sizes = vec![0usize; num_agg];
        for &g in &agg {
            sizes[g] += 1;
        }
        let one_coarse: Vec<f64> = sizes.iter().map(|&s| (s as f64).sqrt()).collect();
        let one_fine = p.matvec(&one_coarse);
        // Interior nodes (away from the two boundaries) reproduce the constant
        // exactly because A·1 = 0 there.
        let mut interior_ok = 0;
        for (i, &v) in one_fine.iter().enumerate() {
            if i > 2 && i < a.nrows - 3 {
                assert!(
                    (v - 1.0).abs() < 1e-9,
                    "smoothed P does not preserve constant in interior at {i}: {v}"
                );
                interior_ok += 1;
            }
        }
        assert!(interior_ok > 0);
    }

    #[test]
    fn rejects_non_square() {
        let a = HostCsr::new(2, 3, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).expect("valid");
        let opts = AmgOptions::default();
        assert!(matches!(
            amg_setup(&a, &opts),
            Err(SparseError::DimensionMismatch(_))
        ));
    }

    #[test]
    fn strength_threshold_behaviour() {
        // With a high threshold no connections are strong; with a low threshold
        // the off-diagonals are strong.
        let a = laplacian_1d(10);
        let high = strength_of_connection(&a, 0.9);
        // |A_ij| = 1, sqrt(|A_ii A_jj|) = 2, so threshold 0.9 -> bound 1.8 > 1.
        assert!(high.iter().all(|nbrs| nbrs.is_empty()));
        let low = strength_of_connection(&a, 0.4);
        // bound 0.8 < 1 -> all off-diagonals strong.
        assert!(low.iter().any(|nbrs| !nbrs.is_empty()));
    }

    #[test]
    fn aggregation_covers_all_nodes() {
        let a = laplacian_2d(7, 7);
        let opts = AmgOptions::default();
        let strength = strength_of_connection(&a, opts.strength_threshold);
        let (agg, num_agg) = aggregate(&a, &strength);
        assert_eq!(agg.len(), a.nrows);
        assert!(num_agg > 0 && num_agg < a.nrows);
        for &g in &agg {
            assert!(g < num_agg);
        }
    }

    #[test]
    fn empty_rhs_zero_solution() {
        let a = laplacian_1d(16);
        let b = vec![0.0; 16];
        let opts = AmgOptions::default();
        let (x, iters, rel) = amg_solve(&a, &b, &opts, 10, 1e-8).expect("solve");
        assert_eq!(iters, 0);
        assert_eq!(rel, 0.0);
        assert!(x.iter().all(|&v| v == 0.0));
    }

    /// Cross-feature integration test: on a single 1-D Poisson matrix exercise
    /// all three algorithms implemented in this crate and check they agree with
    /// each other and with analytic theory.
    ///
    /// * LOBPCG recovers the smallest eigenvalue `λ₁ = 2 − 2cos(π/(n+1))`.
    /// * The complete IC(k) factor solves `A x = b` exactly.
    /// * AMG solves the same system to tolerance.
    /// * The IC(k) and AMG solutions match.
    #[test]
    fn cross_feature_poisson_consistency() {
        use crate::eig::lobpcg::lobpcg;
        use crate::preconditioner::ick::ic_k;
        use std::f64::consts::PI;

        let n = 48;
        let a = laplacian_1d(n);

        // (1) LOBPCG: smallest eigenvalue matches analytic value.
        let eig = lobpcg(&a, 1, 300, 1e-7, None).expect("lobpcg");
        let lambda1 = 2.0 - 2.0 * (PI / ((n + 1) as f64)).cos();
        assert!(
            (eig.eigenvalues[0] - lambda1).abs() < 1e-6,
            "LOBPCG smallest eig {} vs analytic {}",
            eig.eigenvalues[0],
            lambda1
        );

        // Known solution and right-hand side.
        let x_star: Vec<f64> = (0..n).map(|i| 1.0 + ((i * 7) % 5) as f64).collect();
        let b = a.matvec(&x_star);

        // (2) Complete IC(k) factor solves the system exactly.
        let fac = ic_k(&a, n + 1).expect("complete cholesky");
        let x_ic = fac.apply(&b);
        for i in 0..n {
            assert!(
                (x_ic[i] - x_star[i]).abs() < 1e-7,
                "IC complete solve mismatch at {i}: {} vs {}",
                x_ic[i],
                x_star[i]
            );
        }

        // (3) AMG solves the same system to tolerance.
        let opts = AmgOptions::default();
        let (x_amg, iters, rel) = amg_solve(&a, &b, &opts, 50, 1e-9).expect("amg solve");
        assert!(rel < 1e-9, "AMG did not converge: rel = {rel}");
        assert!(iters < 25, "AMG took too many cycles: {iters}");

        // (4) The two independent solvers agree.
        for i in 0..n {
            assert!(
                (x_amg[i] - x_ic[i]).abs() < 1e-6,
                "AMG and IC solutions disagree at {i}: {} vs {}",
                x_amg[i],
                x_ic[i]
            );
        }
    }
}
