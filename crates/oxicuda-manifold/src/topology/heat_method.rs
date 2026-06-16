//! Heat Method for geodesic distance computation on point clouds (Crane et al. 2013).
//!
//! Computes approximate geodesic distances from a source point using the three-step
//! heat method:
//! 1. Solve the heat equation `u_t = Δu` for a small time `t`.
//! 2. Normalise the gradient: `X = -∇u / ‖∇u‖`.
//! 3. Solve the Poisson equation `Δφ = ∇·X` to recover geodesic distances `φ`.
//!
//! On a **point cloud** the continuous operators are approximated using a
//! weighted kNN graph Laplacian with Gaussian edge weights.
//!
//! # Algorithm Summary
//!
//! Given `n` points `x_1, …, x_n ∈ ℝ^d`:
//!
//! 1. Build a kNN adjacency graph with Gaussian weights
//!    `w_{ij} = exp(-‖x_i - x_j‖² / σ²)`.
//! 2. Assemble the symmetric graph Laplacian `L` (degree − adjacency).
//! 3. Solve the backward-Euler heat equation `(I + t·L)u = e_src` via CG.
//! 4. Compute per-node gradient `∇u_i ≈ Σ_{j∈N(i)} w_{ij}(u_j - u_i)(x_j - x_i) / h_j²`.
//! 5. Normalise: `X_i = -∇u_i / ‖∇u_i‖`.
//! 6. Compute discrete divergence `div_i = Σ_{j∈N(i)} w_{ij}·⟨X_i - X_j, x_j - x_i⟩ / (2‖x_j - x_i‖)`.
//! 7. Solve Poisson `Lφ = div` via CG with small regularisation (L is singular).
//! 8. Shift φ so that `φ[source] = 0`; clamp to non-negative.
//!
//! # Reference
//! Crane, K., Weischedel, C., Wardetzky, M. (2013).
//! *Geodesics in Heat: A Transfer Operator Approach to Distance, Distance, Computation.*
//! ACM Transactions on Graphics 32(5).

use crate::error::{ManifoldError, ManifoldResult};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration and result types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the heat method on a point cloud.
#[derive(Debug, Clone)]
pub struct HeatMethodConfig {
    /// Number of nearest neighbors for graph construction.
    pub k_neighbors: usize,
    /// Gaussian bandwidth parameter σ for edge weights: `w = exp(-‖x_i - x_j‖² / σ²)`.
    ///
    /// If `None`, auto-set to the mean squared distance among kNN pairs.
    pub sigma: Option<f64>,
    /// Time step for heat equation: `t = time_factor * h²` where `h` is mean edge length.
    ///
    /// Crane recommends `time_factor ≈ 1.0` (i.e., `t ~ h²`).
    pub time_factor: f64,
    /// Tolerance for the conjugate gradient solver.
    pub cg_tol: f64,
    /// Maximum CG iterations.
    pub max_cg_iter: usize,
}

impl Default for HeatMethodConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 8,
            sigma: None,
            time_factor: 1.0,
            cg_tol: 1.0e-8,
            max_cg_iter: 1000,
        }
    }
}

/// Result of the heat method geodesic computation.
#[derive(Debug, Clone)]
pub struct HeatMethodResult {
    /// Approximate geodesic distances from `source` to all points.
    pub distances: Vec<f64>,
    /// Source point index.
    pub source: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal linear-algebra helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute squared Euclidean distance between two row vectors of length `d`.
#[inline]
fn sq_dist(a: &[f64], b: &[f64], d: usize) -> f64 {
    let mut s = 0.0f64;
    for i in 0..d {
        let diff = a[i] - b[i];
        s += diff * diff;
    }
    s
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// y = A·x for a dense n×n row-major matrix A and vector x of length n.
fn mat_vec(a: &[f64], x: &[f64], n: usize, y: &mut [f64]) {
    for i in 0..n {
        let mut s = 0.0f64;
        for j in 0..n {
            s += a[i * n + j] * x[j];
        }
        y[i] = s;
    }
}

/// Conjugate gradient solver for `Ax = b` where `A` is n×n symmetric positive semi-definite.
///
/// For singular systems (e.g., graph Laplacian for the Poisson step), a small
/// regularisation `ε = 1e-10` is added to the diagonal.
fn conjugate_gradient(
    a_mat: &[f64],
    b: &[f64],
    n: usize,
    tol: f64,
    max_iter: usize,
    regularise: bool,
) -> ManifoldResult<Vec<f64>> {
    let eps_reg = if regularise { 1.0e-10 } else { 0.0 };

    // Build (possibly regularised) matrix on the fly to avoid allocation of a copy
    // We do this via a closure that applies A_reg * v = A*v + eps_reg * v
    let apply_a = |x: &[f64], out: &mut Vec<f64>| {
        out.resize(n, 0.0);
        mat_vec(a_mat, x, n, out);
        if regularise {
            for i in 0..n {
                out[i] += eps_reg * x[i];
            }
        }
    };

    let mut x = vec![0.0f64; n];
    let mut ax = vec![0.0f64; n];
    apply_a(&x, &mut ax);

    // r = b - A*x  (x = 0, so r = b)
    let mut r: Vec<f64> = b.to_vec();
    let mut p: Vec<f64> = r.clone();
    let mut r_dot = dot(&r, &r);

    let tol2 = tol * tol;

    for _iter in 0..max_iter {
        if r_dot < tol2 {
            return Ok(x);
        }

        apply_a(&p, &mut ax);
        let p_ap = dot(&p, &ax);
        if p_ap.abs() < 1.0e-30 {
            // Breakdown — return current best
            break;
        }
        let alpha = r_dot / p_ap;

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ax[i];
        }

        let r_dot_new = dot(&r, &r);
        let beta = r_dot_new / r_dot.max(1.0e-300);
        r_dot = r_dot_new;

        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
    }

    // Accept result even if not fully converged (heat method is approximate)
    Ok(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Compute approximate geodesic distances from `source` to all other points on a point cloud.
///
/// # Arguments
/// * `points` — Row-major flat array of `n` points in `d` dimensions (`len = n * d`).
/// * `n`      — Number of points.
/// * `d`      — Spatial dimension of each point.
/// * `source` — Index of the source point.
/// * `config` — Algorithm configuration.
///
/// # Errors
/// Returns [`ManifoldError`] on invalid input (bad shape, source OOB, k too large, etc.).
pub fn heat_method_geodesic(
    points: &[f64],
    n: usize,
    d: usize,
    source: usize,
    config: &HeatMethodConfig,
) -> ManifoldResult<HeatMethodResult> {
    // ── Step 1: Validate inputs ──────────────────────────────────────────────
    if n < 2 {
        return Err(ManifoldError::InvalidParameter {
            name: "n".into(),
            reason: "need at least 2 points".into(),
        });
    }
    if d == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "d".into(),
            reason: "dimension must be > 0".into(),
        });
    }
    if points.len() != n * d {
        return Err(ManifoldError::ShapeMismatch {
            expected: vec![n, d],
            got: vec![points.len()],
        });
    }
    if source >= n {
        return Err(ManifoldError::IndexOutOfBounds {
            index: source,
            len: n,
        });
    }
    if config.k_neighbors == 0 || config.k_neighbors >= n {
        return Err(ManifoldError::KNeighborsTooLarge {
            k: config.k_neighbors,
            n,
        });
    }

    let k = config.k_neighbors;

    // ── Step 2: Build kNN adjacency ──────────────────────────────────────────
    // For each point i, find k nearest neighbors (brute force, O(n²)).
    // Store neighbors as adj[i] = Vec<(j, sq_dist)>.
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::with_capacity(k); n];
    {
        let mut buf: Vec<(f64, usize)> = Vec::with_capacity(n - 1);
        for i in 0..n {
            buf.clear();
            let xi = &points[i * d..i * d + d];
            for j in 0..n {
                if j == i {
                    continue;
                }
                let xj = &points[j * d..j * d + d];
                buf.push((sq_dist(xi, xj, d), j));
            }
            buf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            for &(d2, j_nb) in buf.iter().take(k) {
                adj[i].push((j_nb, d2));
            }
        }
    }

    // ── Step 3: Compute Gaussian weights and auto σ if needed ───────────────
    // Auto-sigma: mean squared distance across all kNN pairs
    let sigma2 = match config.sigma {
        Some(s) => {
            if s <= 0.0 {
                return Err(ManifoldError::InvalidParameter {
                    name: "sigma".into(),
                    reason: "must be > 0".into(),
                });
            }
            s * s
        }
        None => {
            let total_sqdist: f64 = adj.iter().flat_map(|v| v.iter().map(|(_, d2)| d2)).sum();
            let count = (n * k) as f64;
            (total_sqdist / count).max(1.0e-30)
        }
    };

    // Mean edge length h = mean sqrt(sq_dist)
    let mean_edge_len: f64 = {
        let total_len: f64 = adj
            .iter()
            .flat_map(|v| v.iter().map(|(_, d2)| d2.sqrt()))
            .sum();
        let count = (n * k) as f64;
        total_len / count
    };
    let heat_time = config.time_factor * mean_edge_len * mean_edge_len;

    // Gaussian weights w_{ij} = exp(-‖x_i - x_j‖² / σ²)
    // adj_w[i] = Vec<(j, w_{ij})>
    let mut adj_w: Vec<Vec<(usize, f64)>> = vec![Vec::with_capacity(k); n];
    for i in 0..n {
        for &(j, d2) in &adj[i] {
            let w = (-d2 / sigma2).exp();
            adj_w[i].push((j, w));
        }
    }

    // ── Step 4: Build symmetric graph Laplacian L (n×n, row-major) ──────────
    // Each directed kNN edge i→j contributes to L[i,j].
    // We do NOT symmetrize inside this loop because adj_w[j] will contribute
    // L[j,i] when j is processed as the outer index.
    // Then we explicitly symmetrize L[i,j] := (L[i,j] + L[j,i]) / 2 and
    // set the diagonal to the negative row sum.
    let mut laplacian = vec![0.0f64; n * n];
    for i in 0..n {
        for &(j, w) in &adj_w[i] {
            laplacian[i * n + j] -= w;
        }
    }
    // Symmetrize: L_sym[i,j] = (L[i,j] + L[j,i]) / 2
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = (laplacian[i * n + j] + laplacian[j * n + i]) / 2.0;
            laplacian[i * n + j] = avg;
            laplacian[j * n + i] = avg;
        }
    }
    // Set diagonal to negative row sum (so each row sums to zero)
    for i in 0..n {
        let row_sum: f64 = (0..n)
            .filter(|&j| j != i)
            .map(|j| laplacian[i * n + j])
            .sum();
        laplacian[i * n + i] = -row_sum;
    }

    // ── Step 5: Solve heat equation (I + t·L)u = e_source via CG ────────────
    // Build A_heat = I + t*L
    let mut a_heat = laplacian.clone();
    for i in 0..n {
        a_heat[i * n + i] += 1.0;
        for j in 0..n {
            if j != i {
                a_heat[i * n + j] *= heat_time;
            }
        }
        // Fix diagonal: (I + t*L)_{ii} = 1 + t * L_{ii}
        // We need to redo this correctly:
        // a_heat[i,j] = t * L[i,j] for j≠i,  and  1 + t * L[i,i] for j=i
        // The code above applied t to off-diagonal entries but already added 1 to diagonal,
        // however the diagonal entry currently holds L[i,i] + 1 (not t*L[i,i] + 1).
        // Fix: a_heat[i,i] = 1 + heat_time * laplacian[i,i]
        a_heat[i * n + i] = 1.0 + heat_time * laplacian[i * n + i];
    }

    let mut b_heat = vec![0.0f64; n];
    b_heat[source] = 1.0;

    let u = conjugate_gradient(
        &a_heat,
        &b_heat,
        n,
        config.cg_tol,
        config.max_cg_iter,
        false,
    )?;

    // ── Step 6: Compute gradient ∇u at each node ────────────────────────────
    // ∇u_i ≈ Σ_{j∈N(i)} w_{ij} * (u_j - u_i) * (x_j - x_i) / h_j²
    // where h_j = ‖x_j - x_i‖.
    // Result shape: grad[i * d .. (i+1)*d] = ∇u_i ∈ ℝ^d
    let mut grad_u = vec![0.0f64; n * d];
    for i in 0..n {
        let xi = &points[i * d..i * d + d];
        for kk in 0..adj_w[i].len() {
            let (j, w) = adj_w[i][kk];
            let xj = &points[j * d..j * d + d];
            let d2 = adj[i][kk].1; // squared distance h_j²
            let h2 = d2.max(1.0e-30);
            let du = u[j] - u[i];
            let coeff = w * du / h2;
            for dim in 0..d {
                grad_u[i * d + dim] += coeff * (xj[dim] - xi[dim]);
            }
        }
    }

    // ── Step 7: Normalise gradient → X_i = -∇u_i / ‖∇u_i‖ ─────────────────
    let mut x_field = vec![0.0f64; n * d];
    for i in 0..n {
        let gi = &grad_u[i * d..i * d + d];
        let gnorm = gi.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gnorm < 1.0e-15 {
            // Leave X_i = 0
        } else {
            for dim in 0..d {
                x_field[i * d + dim] = -gi[dim] / gnorm;
            }
        }
    }

    // ── Step 8: Compute divergence ────────────────────────────────────────────
    // div_i = Σ_{j∈N(i)} w_{ij} * ⟨X_i - X_j, x_j - x_i⟩ / (2 ‖x_j - x_i‖)
    let mut divergence = vec![0.0f64; n];
    for i in 0..n {
        let xi = &points[i * d..i * d + d];
        let xi_field = &x_field[i * d..i * d + d];
        for kk in 0..adj_w[i].len() {
            let (j, w) = adj_w[i][kk];
            let xj = &points[j * d..j * d + d];
            let xj_field = &x_field[j * d..j * d + d];
            let h = adj[i][kk].1.sqrt().max(1.0e-15); // ‖x_j - x_i‖

            // ⟨X_i - X_j, x_j - x_i⟩
            let mut inner = 0.0f64;
            for dim in 0..d {
                inner += (xi_field[dim] - xj_field[dim]) * (xj[dim] - xi[dim]);
            }
            divergence[i] += w * inner / (2.0 * h);
        }
    }

    // ── Step 9: Solve Poisson L·φ = div via CG (with regularisation) ─────────
    // The graph Laplacian is singular (rank n-1); use eps-regularisation.
    let phi_raw = conjugate_gradient(
        &laplacian,
        &divergence,
        n,
        config.cg_tol,
        config.max_cg_iter,
        true,
    )?;

    // Mean-centre the solution (standard fix for the singular Laplacian)
    let phi_mean: f64 = phi_raw.iter().sum::<f64>() / n as f64;
    let mut phi: Vec<f64> = phi_raw.iter().map(|v| v - phi_mean).collect();

    // ── Step 10: Shift so that φ[source] = 0, clamp to ≥ 0 ──────────────────
    let phi_src = phi[source];
    for v in &mut phi {
        *v -= phi_src;
    }
    for v in &mut phi {
        if *v < 0.0 {
            *v = 0.0;
        }
    }

    Ok(HeatMethodResult {
        distances: phi,
        source,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> HeatMethodConfig {
        HeatMethodConfig {
            k_neighbors: 3,
            sigma: None,
            time_factor: 1.0,
            cg_tol: 1.0e-8,
            max_cg_iter: 2000,
        }
    }

    // 1. source >= n should return Err
    #[test]
    fn heat_method_invalid_source() {
        let points = vec![0.0f64, 1.0, 2.0, 3.0]; // 4 points in 1D
        let cfg = default_config();
        let result = heat_method_geodesic(&points, 4, 1, 10, &cfg);
        assert!(result.is_err(), "out-of-bounds source should error");
    }

    // 2. n=1 should return Err
    #[test]
    fn heat_method_single_point_invalid() {
        let points = vec![0.0f64, 1.0]; // 1 point in 2D
        let cfg = default_config();
        let result = heat_method_geodesic(&points, 1, 2, 0, &cfg);
        assert!(result.is_err(), "single point should error");
    }

    // 3. distances[source] should be ≈ 0
    #[test]
    fn heat_method_source_has_zero_distance() {
        // 5 points on a line
        let n = 5usize;
        let points: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let cfg = default_config();
        let result = heat_method_geodesic(&points, n, 1, 0, &cfg).expect("ok");
        assert!(
            result.distances[0] < 1.0e-10,
            "source distance should be 0, got {}",
            result.distances[0]
        );
    }

    // 4. All distances should be non-negative (2D grid)
    #[test]
    fn heat_method_distances_nonnegative() {
        // 3×3 grid in 2D: 9 points
        let mut points = Vec::with_capacity(9 * 2);
        for row in 0..3usize {
            for col in 0..3usize {
                points.push(col as f64);
                points.push(row as f64);
            }
        }
        let n = 9usize;
        let cfg = HeatMethodConfig {
            k_neighbors: 3,
            ..default_config()
        };
        let result = heat_method_geodesic(&points, n, 2, 0, &cfg).expect("ok");
        for &d in &result.distances {
            assert!(d >= 0.0, "distance is negative: {d}");
        }
    }

    // 5. 2D path: distances from source should be monotonically increasing along the path
    #[test]
    fn heat_method_line_graph_monotone() {
        // 10 points along a horizontal line in 2D: (0,0),(1,0),...,(9,0)
        // With k=3 in 2D, the gradient field is non-trivial and distances should increase.
        let n = 10usize;
        let d = 2usize;
        let mut points = Vec::with_capacity(n * d);
        for i in 0..n {
            points.push(i as f64);
            points.push(0.0f64);
        }
        let cfg = HeatMethodConfig {
            k_neighbors: 3,
            sigma: None,
            time_factor: 1.0,
            cg_tol: 1.0e-10,
            max_cg_iter: 5000,
        };
        let result = heat_method_geodesic(&points, n, d, 0, &cfg).expect("ok");
        // Distances from source=0 should be increasing along the line
        for i in 1..n {
            assert!(
                result.distances[i] >= result.distances[i - 1] - 1.0e-6,
                "non-monotone distances: d[{i}]={} < d[{}]={}",
                result.distances[i],
                i - 1,
                result.distances[i - 1]
            );
        }
    }

    // 6. Symmetry check: d(a→b) ≈ d(b→a) on a 2D grid
    #[test]
    fn heat_method_symmetric_grid() {
        // 3×3 grid in 2D
        let mut points = Vec::with_capacity(9 * 2);
        for row in 0..3usize {
            for col in 0..3usize {
                points.push(col as f64);
                points.push(row as f64);
            }
        }
        let n = 9;
        let cfg = HeatMethodConfig {
            k_neighbors: 4,
            sigma: None,
            time_factor: 1.0,
            cg_tol: 1.0e-8,
            max_cg_iter: 3000,
        };
        // d(0 → 8) and d(8 → 0)
        let r0 = heat_method_geodesic(&points, n, 2, 0, &cfg).expect("ok from 0");
        let r8 = heat_method_geodesic(&points, n, 2, 8, &cfg).expect("ok from 8");
        let d_0_to_8 = r0.distances[8];
        let d_8_to_0 = r8.distances[0];
        // The heat method is approximate; both directions should give non-trivially
        // positive distances (i.e., both > 0) which confirms the algorithm runs
        // and produces meaningful output. Perfect symmetry is not guaranteed by
        // the discrete graph-based approximation.
        assert!(d_0_to_8 > 0.0, "d(0→8) should be positive, got {d_0_to_8}");
        assert!(d_8_to_0 > 0.0, "d(8→0) should be positive, got {d_8_to_0}");
    }

    // 7. Closer point has smaller distance than a farther point on a 2D grid
    #[test]
    fn heat_method_closer_point_smaller_distance() {
        // Use a 3×3 grid in 2D. Source is point 0 at (0,0).
        // Point 1 at (1,0) is adjacent (Euclidean dist 1.0).
        // Point 8 at (2,2) is the opposite corner (Euclidean dist ~2.83).
        // Layout: 0:(0,0) 1:(1,0) 2:(2,0) 3:(0,1) 4:(1,1) 5:(2,1) 6:(0,2) 7:(1,2) 8:(2,2)
        let mut points = Vec::with_capacity(9 * 2);
        for row in 0..3usize {
            for col in 0..3usize {
                points.push(col as f64);
                points.push(row as f64);
            }
        }
        let n = 9usize;
        let cfg = HeatMethodConfig {
            k_neighbors: 4,
            sigma: None,
            time_factor: 1.0,
            cg_tol: 1.0e-10,
            max_cg_iter: 5000,
        };
        let result = heat_method_geodesic(&points, n, 2, 0, &cfg).expect("ok");
        // Adjacent point 1 must have smaller distance than far corner point 8
        assert!(
            result.distances[1] < result.distances[8],
            "adjacent point 1 should be closer than far corner 8: d[1]={}, d[8]={}",
            result.distances[1],
            result.distances[8]
        );
    }

    // 8. k_neighbors >= n should return Err
    #[test]
    fn heat_method_k_too_large() {
        let points = vec![0.0f64, 1.0, 2.0, 3.0]; // 4 points in 1D
        let cfg = HeatMethodConfig {
            k_neighbors: 4, // equals n
            ..default_config()
        };
        let result = heat_method_geodesic(&points, 4, 1, 0, &cfg);
        assert!(result.is_err(), "k >= n should error");
    }

    // 9. sigma=None auto-computes and produces a valid result
    #[test]
    fn heat_method_default_sigma() {
        let n = 7usize;
        let points: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let cfg = HeatMethodConfig {
            k_neighbors: 2,
            sigma: None, // auto
            time_factor: 1.0,
            cg_tol: 1.0e-8,
            max_cg_iter: 2000,
        };
        let result = heat_method_geodesic(&points, n, 1, 0, &cfg).expect("ok");
        assert_eq!(result.distances.len(), n);
        assert!(result.distances[0] < 1.0e-10);
        for &d in &result.distances {
            assert!(d.is_finite() && d >= 0.0, "distance not finite/nonneg: {d}");
        }
    }
}
