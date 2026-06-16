//! Tangential complex (Boissonnat & Ghosh, "Manifold reconstruction using
//! tangential Delaunay complexes", *Discrete & Computational Geometry* 2014).
//!
//! Given a point sample `P ⊂ ℝ^D` lying near an (unknown) `d`-dimensional
//! manifold `ℳ`, the **tangential complex** `TC` reconstructs `ℳ` from purely
//! local information:
//!
//! 1. **Local tangent estimation.** At every sample point `p` the tangent space
//!    `T_p ≈ T_p ℳ` is estimated by *local PCA*: collect the `k` nearest
//!    neighbours of `p`, centre them, form the `D × D` covariance matrix, and
//!    take its top-`d` eigenvectors (the directions of greatest local variance)
//!    as an orthonormal basis of `T_p`.  For data sampled from a genuinely flat
//!    patch this recovers the tangent *exactly* (the remaining `D − d`
//!    eigenvalues are `0`).
//!
//! 2. **Local (star) Delaunay triangulation.** The neighbours of `p` are
//!    orthogonally projected into the `d`-dimensional tangent space `T_p`
//!    (coordinates expressed in the estimated basis).  In that flat chart we
//!    build the **star** of `p`: the set of `d`-simplices incident to `p` whose
//!    circumscribing ball (in `T_p`) is *empty* of all other projected
//!    neighbours — i.e. local Delaunay simplices.  Boissonnat & Ghosh call this
//!    the *star* `St(p)` of `p` in its tangent triangulation.
//!
//! 3. **Star gluing.** The stars of all sample points are collected into a
//!    single [`SimplicialComplex`] (closed under faces).  For a sufficiently
//!    dense, well-behaved sample the union of stars is a faithful triangulation
//!    of `ℳ`, and in particular has intrinsic dimension `d`.
//!
//! The construction is *intrinsically `d`-dimensional* even though the points
//! live in `ℝ^D`, which is the central advantage of the tangential complex over
//! an ambient Delaunay/Čech construction: its size and the dimension of its
//! simplices scale with the manifold dimension `d`, not the ambient dimension
//! `D`.
//!
//! All linear algebra (covariance, symmetric eigendecomposition via the cyclic
//! Jacobi rotation method, circumball computation) is implemented here in pure
//! `f64` arithmetic, reusing the crate's [`pairwise_euclidean`] /
//! [`knn_graph`] primitives for neighbour search.

use crate::complex::complex::SimplicialComplex;
use crate::complex::simplex::Simplex;
use crate::distance::pairwise::{knn_graph, pairwise_euclidean};
use crate::error::{TdaError, TdaResult};

/// Number of cyclic-Jacobi sweeps used by the symmetric eigensolver.
///
/// Each sweep annihilates every off-diagonal entry once; the Frobenius norm of
/// the off-diagonal part decays quadratically, so a handful of sweeps drive a
/// small symmetric matrix (here `D × D` with `D` the ambient dimension) to
/// machine-precision diagonal form.  30 sweeps is far more than required for the
/// modest `D` of typical manifold samples yet remains negligibly cheap.
const JACOBI_SWEEPS: usize = 30;

/// Convergence threshold for an individual Jacobi rotation: off-diagonal entries
/// whose magnitude is below this (relative to the matrix scale already removed)
/// are treated as numerically zero and skipped.
const JACOBI_EPS: f64 = 1.0e-300;

/// Relative tolerance used when testing whether a projected neighbour lies on
/// the circumsphere of a candidate Delaunay simplex (the empty-ball predicate).
const CIRCUMBALL_TOL: f64 = 1.0e-9;

/// An estimated tangent space at a sample point.
///
/// `basis` stores `d` orthonormal `D`-vectors spanning the estimated tangent
/// `T_p`, row-major as `basis[i * ambient_dim + j]` = `j`-th ambient coordinate
/// of the `i`-th basis vector.  `eigenvalues` are the corresponding (sorted
/// descending) local-PCA variances; the trailing ambient eigenvalues (the normal
/// directions) are returned separately in `normal_eigenvalues`.
#[derive(Debug, Clone)]
pub struct TangentSpace {
    /// Index of the sample point this tangent was estimated at.
    pub point: usize,
    /// Ambient dimension `D`.
    pub ambient_dim: usize,
    /// Manifold (tangent) dimension `d`.
    pub manifold_dim: usize,
    /// `d × D` orthonormal tangent basis, row-major.
    pub basis: Vec<f64>,
    /// The top-`d` PCA eigenvalues (variances along the tangent directions),
    /// sorted descending.
    pub eigenvalues: Vec<f64>,
    /// The trailing `D − d` PCA eigenvalues (variances along the normal
    /// directions), sorted descending.  These are ~`0` for flat data.
    pub normal_eigenvalues: Vec<f64>,
}

impl TangentSpace {
    /// Orthogonally project an ambient `D`-vector `v` (already centred at the
    /// sample point) into local tangent coordinates `ℝ^d`.
    ///
    /// The `i`-th output coordinate is `⟨v, e_i⟩` with `e_i` the `i`-th tangent
    /// basis vector.
    #[must_use]
    pub fn project(&self, v: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0_f64; self.manifold_dim];
        for (i, coord) in out.iter_mut().enumerate() {
            let base = i * self.ambient_dim;
            let row = &self.basis[base..base + self.ambient_dim];
            *coord = row.iter().zip(v.iter()).map(|(&e, &x)| e * x).sum();
        }
        out
    }

    /// The estimated unit normal directions: an orthonormal basis of the
    /// orthogonal complement of the tangent space, returned as `(D − d)`
    /// ambient `D`-vectors row-major.
    ///
    /// For a `d`-manifold in `ℝ^D` there are `D − d` independent normals; for the
    /// common hypersurface case `D = d + 1` there is a single normal.
    #[must_use]
    pub fn normals(&self) -> Vec<f64> {
        // The full eigenbasis is orthonormal; the normals are exactly the
        // trailing D-d eigenvectors, which we recompute alongside the tangent
        // basis in `estimate_tangent_space`.  They are stored contiguously after
        // the tangent rows in `full_basis`, but to keep `TangentSpace` compact we
        // recompute the complement here from the tangent rows via Gram–Schmidt
        // against the ambient standard basis.
        gram_schmidt_complement(&self.basis, self.manifold_dim, self.ambient_dim)
    }
}

/// The tangential complex itself: the glued stars plus the per-point tangent
/// spaces used to build them.
#[derive(Debug, Clone)]
pub struct TangentialComplex {
    /// The reconstructed simplicial complex (closed under faces), over the
    /// global sample-point indices.
    pub complex: SimplicialComplex,
    /// Estimated tangent space at every sample point, indexed by point.
    pub tangents: Vec<TangentSpace>,
    /// Manifold dimension `d`.
    pub manifold_dim: usize,
    /// Ambient dimension `D`.
    pub ambient_dim: usize,
}

impl TangentialComplex {
    /// Intrinsic dimension of the reconstruction (the manifold dimension `d`).
    ///
    /// Equivalently `self.complex.max_dim()` for a well-sampled manifold; we
    /// return the configured `d`, which the star construction guarantees as the
    /// top simplex dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.manifold_dim
    }

    /// The star of a sample point `p`: every maximal (`d`-)simplex of the
    /// complex that contains `p` as a vertex.
    #[must_use]
    pub fn star(&self, point: usize) -> Vec<&Simplex> {
        self.complex
            .simplices_of_dim(self.manifold_dim)
            .into_iter()
            .filter(|s| s.vertices.contains(&point))
            .collect()
    }
}

/// Estimate the tangent space at a single point by **local PCA**.
///
/// `points` is the row-major ambient point cloud (`points[i*D + j]`), `point`
/// is the index of the centre, and `neighbours` are the indices of its `k`
/// nearest neighbours (excluding itself).  The covariance of the centred
/// neighbourhood is diagonalised by cyclic Jacobi rotations; the eigenvectors of
/// the `d` largest eigenvalues form the tangent basis.
///
/// # Errors
/// - [`TdaError::DimensionMismatch`] if `ambient_dim == 0`.
/// - [`TdaError::DimensionTooLarge`] if `manifold_dim > ambient_dim`.
/// - [`TdaError::LandmarkSelectionFailed`] if fewer than `manifold_dim`
///   neighbours are available (the covariance cannot span a `d`-flat).
pub fn estimate_tangent_space(
    points: &[f64],
    ambient_dim: usize,
    point: usize,
    neighbours: &[usize],
    manifold_dim: usize,
) -> TdaResult<TangentSpace> {
    if ambient_dim == 0 {
        return Err(TdaError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if manifold_dim > ambient_dim {
        return Err(TdaError::DimensionTooLarge(manifold_dim));
    }
    if neighbours.len() < manifold_dim {
        return Err(TdaError::LandmarkSelectionFailed(format!(
            "point {point} has {} neighbours but manifold dimension {manifold_dim} \
             requires at least that many to span the tangent",
            neighbours.len()
        )));
    }
    let n_pts = points.len() / ambient_dim;
    if point >= n_pts {
        return Err(TdaError::DimensionMismatch {
            expected: n_pts,
            got: point,
        });
    }

    // Centre the neighbourhood on the sample point itself (the natural origin of
    // the local chart; Boissonnat–Ghosh centre on `p`, not on the neighbour
    // mean, so the chart passes through `p`).
    let centre = &points[point * ambient_dim..point * ambient_dim + ambient_dim];

    // Accumulate the D×D scatter matrix Σ (p_j - p)(p_j - p)ᵀ.
    let mut cov = vec![0.0_f64; ambient_dim * ambient_dim];
    for &nb in neighbours {
        if nb >= n_pts {
            return Err(TdaError::DimensionMismatch {
                expected: n_pts,
                got: nb,
            });
        }
        let row = &points[nb * ambient_dim..nb * ambient_dim + ambient_dim];
        for a in 0..ambient_dim {
            let da = row[a] - centre[a];
            for b in a..ambient_dim {
                let db = row[b] - centre[b];
                cov[a * ambient_dim + b] += da * db;
            }
        }
    }
    // Symmetrise (we only filled the upper triangle).
    for a in 0..ambient_dim {
        for b in (a + 1)..ambient_dim {
            cov[b * ambient_dim + a] = cov[a * ambient_dim + b];
        }
    }

    let (eigenvalues, eigenvectors) = jacobi_symmetric_eig(&cov, ambient_dim);

    // `jacobi_symmetric_eig` returns eigenpairs sorted by eigenvalue descending,
    // with `eigenvectors[i*D + j]` the j-th component of the i-th eigenvector.
    let mut basis = Vec::with_capacity(manifold_dim * ambient_dim);
    let mut tangent_eigs = Vec::with_capacity(manifold_dim);
    for i in 0..manifold_dim {
        tangent_eigs.push(eigenvalues[i]);
        basis.extend_from_slice(&eigenvectors[i * ambient_dim..i * ambient_dim + ambient_dim]);
    }
    let normal_eigs = eigenvalues[manifold_dim..].to_vec();

    Ok(TangentSpace {
        point,
        ambient_dim,
        manifold_dim,
        basis,
        eigenvalues: tangent_eigs,
        normal_eigenvalues: normal_eigs,
    })
}

/// Build the complete tangential complex for a point cloud.
///
/// `points` is the row-major ambient cloud (`points[i*ambient_dim + j]`), `d`
/// the target manifold dimension and `k` the local PCA / star neighbourhood
/// size.  Returns the glued complex together with all estimated tangents.
///
/// # Algorithm
/// For each sample point `p`:
/// 1. estimate `T_p` by local PCA over its `k` nearest neighbours
///    ([`estimate_tangent_space`]);
/// 2. project those neighbours into `T_p` and build the **local Delaunay star**
///    of `p` (the `d`-simplices incident to `p` with an empty circumball in
///    `T_p`);
/// 3. add each star simplex (with its face closure) to the global complex.
///
/// # Errors
/// - [`TdaError::EmptyPointCloud`] if `points` is empty.
/// - [`TdaError::DimensionMismatch`] if `points.len()` is not a multiple of
///   `ambient_dim`, or `ambient_dim == 0`.
/// - [`TdaError::DimensionTooLarge`] if `d > ambient_dim`.
/// - [`TdaError::ParameterOutOfRange`] if `k == 0`.
/// - [`TdaError::LandmarkSelectionFailed`] if any point has too few neighbours
///   to span a `d`-flat (fewer than `d` neighbours), or if the whole cloud has
///   fewer than `d + 1` points (no `d`-simplex can exist).
pub fn tangential_complex(
    points: &[f64],
    ambient_dim: usize,
    d: usize,
    k: usize,
) -> TdaResult<TangentialComplex> {
    if ambient_dim == 0 {
        return Err(TdaError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if points.is_empty() {
        return Err(TdaError::EmptyPointCloud);
    }
    if !points.len().is_multiple_of(ambient_dim) {
        return Err(TdaError::DimensionMismatch {
            expected: (points.len() / ambient_dim) * ambient_dim,
            got: points.len(),
        });
    }
    if d > ambient_dim {
        return Err(TdaError::DimensionTooLarge(d));
    }
    if k == 0 {
        return Err(TdaError::ParameterOutOfRange(
            "neighbourhood size k must be positive".to_owned(),
        ));
    }
    let n_pts = points.len() / ambient_dim;
    if n_pts < d + 1 {
        return Err(TdaError::LandmarkSelectionFailed(format!(
            "{n_pts} points cannot form a {d}-simplex (need at least {})",
            d + 1
        )));
    }

    // Reuse the crate's neighbour search.
    let dist = pairwise_euclidean(points, ambient_dim)?;
    let neighbours = knn_graph(&dist, n_pts, k)?;

    // Estimate every tangent space first (so the complex dimension is well
    // defined even for d == 0 degenerate inputs).
    let mut tangents = Vec::with_capacity(n_pts);
    for (p, nbrs) in neighbours.iter().enumerate() {
        tangents.push(estimate_tangent_space(points, ambient_dim, p, nbrs, d)?);
    }

    let mut complex = SimplicialComplex::new();

    if d == 0 {
        // A 0-manifold is a set of isolated points: the complex is the vertices.
        for p in 0..n_pts {
            complex.add_simplex_with_closure(Simplex::new(vec![p])?)?;
        }
        return Ok(TangentialComplex {
            complex,
            tangents,
            manifold_dim: d,
            ambient_dim,
        });
    }

    for (p, (tangent, nbrs)) in tangents.iter().zip(neighbours.iter()).enumerate() {
        let star = local_delaunay_star(points, ambient_dim, p, nbrs, tangent, d)?;
        for simplex_vertices in star {
            complex.add_simplex_with_closure(Simplex::new(simplex_vertices)?)?;
        }
    }

    Ok(TangentialComplex {
        complex,
        tangents,
        manifold_dim: d,
        ambient_dim,
    })
}

/// Build the local Delaunay **star** of point `p` in its estimated tangent space.
///
/// Returns the vertex lists (global indices, including `p`) of the `d`-simplices
/// incident to `p` whose circumscribing ball — computed in the `d`-dimensional
/// tangent chart `T_p` — is *empty* of every other projected neighbour.  This is
/// the classical empty-ball (Delaunay) predicate restricted to the star of `p`.
///
/// The candidate vertex set is `{p} ∪ neighbours`; we project all of them into
/// `T_p`, enumerate every `d`-subset of the neighbours together with `p`, and
/// keep the simplex iff its circumball is empty.  For the modest neighbourhood
/// sizes used in manifold reconstruction this exhaustive local enumeration is
/// both exact and cheap.
fn local_delaunay_star(
    points: &[f64],
    ambient_dim: usize,
    p: usize,
    neighbours: &[usize],
    tangent: &TangentSpace,
    d: usize,
) -> TdaResult<Vec<Vec<usize>>> {
    // Project p (origin of the chart -> 0) and all neighbours into T_p.
    let centre = &points[p * ambient_dim..p * ambient_dim + ambient_dim];
    let mut local_ids: Vec<usize> = Vec::with_capacity(neighbours.len() + 1);
    let mut local_coords: Vec<Vec<f64>> = Vec::with_capacity(neighbours.len() + 1);
    local_ids.push(p);
    local_coords.push(vec![0.0_f64; d]); // p projects to the origin.
    for &nb in neighbours {
        let row = &points[nb * ambient_dim..nb * ambient_dim + ambient_dim];
        let mut centred = vec![0.0_f64; ambient_dim];
        for j in 0..ambient_dim {
            centred[j] = row[j] - centre[j];
        }
        local_ids.push(nb);
        local_coords.push(tangent.project(&centred));
    }

    let m = local_ids.len();
    let mut stars: Vec<Vec<usize>> = Vec::new();

    // Enumerate every d-subset of the *neighbour* indices (1..m), each combined
    // with p (index 0), to form a candidate d-simplex of the star.
    let mut combo: Vec<usize> = (1..=d).collect();
    if d > m - 1 {
        // Not enough neighbours to form any d-simplex incident to p.
        return Ok(stars);
    }
    loop {
        // Vertices of the candidate simplex: p (local index 0) plus the combo.
        let mut local_vertices = Vec::with_capacity(d + 1);
        local_vertices.push(0usize);
        local_vertices.extend_from_slice(&combo);

        if let Some((centre_pt, radius_sq)) =
            circumball(&local_coords, &local_vertices, d, CIRCUMBALL_TOL)
        {
            // Empty-ball test: no *other* projected point lies strictly inside.
            let mut empty = true;
            for (idx, coord) in local_coords.iter().enumerate() {
                if local_vertices.contains(&idx) {
                    continue;
                }
                let mut dist_sq = 0.0_f64;
                for j in 0..d {
                    let diff = coord[j] - centre_pt[j];
                    dist_sq += diff * diff;
                }
                // Strictly inside (with relative slack) ⇒ not Delaunay.
                if dist_sq < radius_sq * (1.0 - CIRCUMBALL_TOL) {
                    empty = false;
                    break;
                }
            }
            if empty {
                let global: Vec<usize> = local_vertices.iter().map(|&li| local_ids[li]).collect();
                stars.push(global);
            }
        }

        if !next_combination(&mut combo, m - 1, d) {
            break;
        }
    }

    Ok(stars)
}

/// Advance `combo` to the next `d`-combination of `{1, …, n}` in colexicographic
/// order, returning `false` when the last combination has been passed.
///
/// `combo` holds `d` strictly increasing values in `1..=n`.
fn next_combination(combo: &mut [usize], n: usize, d: usize) -> bool {
    if d == 0 {
        return false;
    }
    let mut i = d - 1;
    loop {
        // Maximum value the i-th element may take.
        let max_val = n - (d - 1 - i);
        if combo[i] < max_val {
            combo[i] += 1;
            for j in (i + 1)..d {
                combo[j] = combo[j - 1] + 1;
            }
            return true;
        }
        if i == 0 {
            return false;
        }
        i -= 1;
    }
}

/// Circumscribing ball of a `d`-simplex in `ℝ^d` given by `d + 1` points.
///
/// `coords` holds the projected coordinates and `vertices` indexes the `d + 1`
/// simplex corners.  Solves the linear system that places the centre equidistant
/// from every vertex.  Returns `None` (degenerate / nearly coplanar simplex) when
/// the system is singular.
fn circumball(
    coords: &[Vec<f64>],
    vertices: &[usize],
    d: usize,
    tol: f64,
) -> Option<(Vec<f64>, f64)> {
    // d+1 points p_0..p_d in R^d.  The circumcentre c satisfies
    // |c - p_i|^2 = |c - p_0|^2 for i = 1..d, i.e.
    //   2 (p_i - p_0) · c = |p_i|^2 - |p_0|^2.
    if vertices.len() != d + 1 {
        return None;
    }
    let p0 = &coords[vertices[0]];
    let mut a = vec![0.0_f64; d * d];
    let mut rhs = vec![0.0_f64; d];
    let norm0: f64 = p0.iter().map(|v| v * v).sum();
    for i in 1..=d {
        let pi = &coords[vertices[i]];
        let normi: f64 = pi.iter().map(|v| v * v).sum();
        for j in 0..d {
            a[(i - 1) * d + j] = 2.0 * (pi[j] - p0[j]);
        }
        rhs[i - 1] = normi - norm0;
    }
    let centre = solve_linear(&mut a, &mut rhs, d, tol)?;
    let mut radius_sq = 0.0_f64;
    for j in 0..d {
        let diff = centre[j] - p0[j];
        radius_sq += diff * diff;
    }
    Some((centre, radius_sq))
}

/// Solve the `n × n` linear system `A x = b` by Gaussian elimination with partial
/// pivoting.  Returns `None` when the matrix is singular to within `tol`.
///
/// `a` (row-major) and `b` are consumed (overwritten) by the elimination.
fn solve_linear(a: &mut [f64], b: &mut [f64], n: usize, tol: f64) -> Option<Vec<f64>> {
    for col in 0..n {
        // Partial pivot: largest magnitude in this column at or below the diagonal.
        let mut pivot = col;
        let mut best = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best <= tol {
            return None;
        }
        if pivot != col {
            for j in 0..n {
                a.swap(col * n + j, pivot * n + j);
            }
            b.swap(col, pivot);
        }
        let diag = a[col * n + col];
        for r in (col + 1)..n {
            let factor = a[r * n + col] / diag;
            if factor == 0.0 {
                continue;
            }
            for j in col..n {
                a[r * n + j] -= factor * a[col * n + j];
            }
            b[r] -= factor * b[col];
        }
    }
    // Back-substitution.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut acc = b[i];
        for j in (i + 1)..n {
            acc -= a[i * n + j] * x[j];
        }
        x[i] = acc / a[i * n + i];
    }
    Some(x)
}

/// Symmetric eigendecomposition of an `n × n` symmetric matrix by the cyclic
/// Jacobi rotation method.
///
/// Returns `(eigenvalues, eigenvectors)` sorted by eigenvalue **descending**,
/// with `eigenvectors[i * n + j]` the `j`-th ambient component of the `i`-th
/// (unit) eigenvector.  The Jacobi method is backward-stable and yields a fully
/// orthonormal eigenbasis even for repeated eigenvalues, which is exactly what
/// local PCA needs (the normal-space eigenvalues are typically degenerate at 0).
fn jacobi_symmetric_eig(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    // Working copy of the matrix (will be diagonalised in place).
    let mut a = matrix.to_vec();
    // Accumulated rotation matrix V (columns are eigenvectors); start at I.
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    if n == 1 {
        return (vec![a[0]], vec![1.0]);
    }

    for _sweep in 0..JACOBI_SWEEPS {
        // Off-diagonal Frobenius norm; stop early once negligible.
        let mut off = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off <= JACOBI_EPS {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() <= JACOBI_EPS {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                // Jacobi rotation angle: cot(2θ) = (aqq - app) / (2 a_pq).
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                // Apply the rotation to rows/columns p and q of A.
                for i in 0..n {
                    let aip = a[i * n + p];
                    let aiq = a[i * n + q];
                    a[i * n + p] = c * aip - s * aiq;
                    a[i * n + q] = s * aip + c * aiq;
                }
                for i in 0..n {
                    let api = a[p * n + i];
                    let aqi = a[q * n + i];
                    a[p * n + i] = c * api - s * aqi;
                    a[q * n + i] = s * api + c * aqi;
                }
                // Accumulate the rotation into V.
                for i in 0..n {
                    let vip = v[i * n + p];
                    let viq = v[i * n + q];
                    v[i * n + p] = c * vip - s * viq;
                    v[i * n + q] = s * vip + c * viq;
                }
            }
        }
    }

    // Extract eigenvalues from the diagonal and pair with eigenvectors (columns
    // of V), then sort descending.
    let mut pairs: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|i| {
            let lambda = a[i * n + i];
            let vec_i: Vec<f64> = (0..n).map(|r| v[r * n + i]).collect();
            (lambda, vec_i)
        })
        .collect();
    pairs.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = Vec::with_capacity(n * n);
    for (lambda, vec_i) in pairs {
        eigenvalues.push(lambda);
        // Normalise (Jacobi keeps it unit, but renormalise for safety) and fix
        // sign so the first non-trivial component is non-negative (deterministic
        // orientation).
        let norm: f64 = vec_i.iter().map(|x| x * x).sum::<f64>().sqrt();
        let inv = if norm > 0.0 { 1.0 / norm } else { 1.0 };
        let mut signed = vec_i;
        let sign = signed
            .iter()
            .copied()
            .find(|x| x.abs() > 1.0e-12)
            .map(|x| if x < 0.0 { -1.0 } else { 1.0 })
            .unwrap_or(1.0);
        for x in &mut signed {
            *x = *x * inv * sign;
        }
        eigenvectors.extend_from_slice(&signed);
    }
    (eigenvalues, eigenvectors)
}

/// Compute an orthonormal basis of the orthogonal complement of the row space of
/// `basis` (a `rows × n` orthonormal set) within `ℝ^n`, via modified
/// Gram–Schmidt against the ambient standard basis.
///
/// Returns the `n − rows` complement vectors row-major.  Used to expose the
/// estimated normal directions of a tangent space.
fn gram_schmidt_complement(basis: &[f64], rows: usize, n: usize) -> Vec<f64> {
    let mut spanned: Vec<Vec<f64>> = (0..rows)
        .map(|i| basis[i * n..i * n + n].to_vec())
        .collect();
    let mut complement: Vec<f64> = Vec::new();
    let mut found = 0usize;
    for e in 0..n {
        if found == n - rows {
            break;
        }
        // Candidate = e-th standard basis vector.
        let mut cand = vec![0.0_f64; n];
        cand[e] = 1.0;
        // Remove components along every already-spanned vector.
        for sp in &spanned {
            let dot: f64 = cand.iter().zip(sp.iter()).map(|(a, b)| a * b).sum();
            for j in 0..n {
                cand[j] -= dot * sp[j];
            }
        }
        let norm: f64 = cand.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1.0e-9 {
            for x in &mut cand {
                *x /= norm;
            }
            complement.extend_from_slice(&cand);
            spanned.push(cand);
            found += 1;
        }
    }
    complement
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Build the row-major distance + kNN and estimate the tangent at one point.
    fn tangent_at(points: &[f64], dim: usize, point: usize, d: usize, k: usize) -> TangentSpace {
        let dist = pairwise_euclidean(points, dim).expect("dist");
        let nbrs = knn_graph(&dist, points.len() / dim, k).expect("knn");
        estimate_tangent_space(points, dim, point, &nbrs[point], d).expect("tangent")
    }

    // (a) Local PCA recovers the tangent plane of a 2-flat in R^3: the estimated
    //     normal is ⟂ the plane (z-axis) to ~1e-6 and the principal angle ≈ 0.
    #[test]
    fn plane_in_r3_tangent_is_exact() {
        // Sample a grid on the z = 0 plane, embedded in R^3.
        let mut points: Vec<f64> = Vec::new();
        for ix in 0..6 {
            for iy in 0..6 {
                points.push(ix as f64 * 0.5);
                points.push(iy as f64 * 0.5);
                points.push(0.0); // z = 0 ⇒ flat plane.
            }
        }
        let tangent = tangent_at(&points, 3, 20, 2, 8);
        // The two trailing (normal) eigenvalues must be ~0 (data is perfectly flat).
        assert_eq!(tangent.normal_eigenvalues.len(), 1);
        assert!(
            tangent.normal_eigenvalues[0].abs() < 1.0e-9,
            "normal variance should vanish for a flat plane: {}",
            tangent.normal_eigenvalues[0]
        );
        // The estimated normal must be parallel to the z-axis (0,0,±1).
        let normals = tangent.normals();
        assert_eq!(normals.len(), 3);
        assert!(
            normals[0].abs() < 1.0e-6 && normals[1].abs() < 1.0e-6,
            "normal not perpendicular to plane: {normals:?}"
        );
        assert!((normals[2].abs() - 1.0).abs() < 1.0e-6, "normal not unit-z");
        // Principal angle between estimated tangent and true (xy) plane ≈ 0:
        // each estimated basis vector must have ~zero z-component.
        for i in 0..2 {
            assert!(
                tangent.basis[i * 3 + 2].abs() < 1.0e-6,
                "tangent basis vector {i} leaks into z"
            );
        }
    }

    // (b) On a unit circle in R^2 the estimated tangent is ⟂ the radius.
    #[test]
    fn circle_tangent_perpendicular_to_radius() {
        let n = 64;
        let mut points: Vec<f64> = Vec::new();
        for i in 0..n {
            let theta = 2.0 * PI * i as f64 / n as f64;
            points.push(theta.cos());
            points.push(theta.sin());
        }
        // Tangent dimension d = 1 (a curve), neighbourhood of 6 points.
        let mut max_dot = 0.0_f64;
        for i in 0..n {
            let tangent = tangent_at(&points, 2, i, 1, 6);
            // Radius direction at point i is (cos θ, sin θ) = the point itself.
            let theta = 2.0 * PI * i as f64 / n as f64;
            let radius = [theta.cos(), theta.sin()];
            let tx = tangent.basis[0];
            let ty = tangent.basis[1];
            let dot = (tx * radius[0] + ty * radius[1]).abs();
            max_dot = max_dot.max(dot);
        }
        // A densely-sampled circle: tangent ⟂ radius to high accuracy.
        assert!(
            max_dot < 1.0e-2,
            "tangent not perpendicular to radius: max |t·r| = {max_dot}"
        );
    }

    // (c) The complex dimension equals the manifold dimension d.
    #[test]
    fn complex_dimension_equals_manifold_dimension() {
        // 2-flat patch in R^3, d = 2.
        let mut points: Vec<f64> = Vec::new();
        for ix in 0..5 {
            for iy in 0..5 {
                points.push(ix as f64);
                points.push(iy as f64);
                points.push(0.0);
            }
        }
        let tc = tangential_complex(&points, 3, 2, 8).expect("complex");
        assert_eq!(tc.dimension(), 2);
        // The glued complex must actually contain 2-simplices.
        assert!(
            tc.complex.max_dim() == 2,
            "expected top dimension 2, got {}",
            tc.complex.max_dim()
        );
        assert!(
            !tc.complex.simplices_of_dim(2).is_empty(),
            "no 2-simplices were built"
        );
    }

    // (d) Local stars are consistent: every d-simplex in a point's star is a
    //     valid Delaunay simplex in its tangent space (empty circumball).
    #[test]
    fn local_stars_are_valid_delaunay() {
        let mut points: Vec<f64> = Vec::new();
        for ix in 0..6 {
            for iy in 0..6 {
                points.push(ix as f64);
                points.push(iy as f64);
                points.push(0.0);
            }
        }
        let dim = 3usize;
        let d = 2usize;
        let k = 10usize;
        let dist = pairwise_euclidean(&points, dim).expect("dist");
        let nbrs = knn_graph(&dist, points.len() / dim, k).expect("knn");

        // Check the star of an interior point (index 14 ≈ centre of the grid).
        let p = 14usize;
        let tangent = estimate_tangent_space(&points, dim, p, &nbrs[p], d).expect("tangent");
        let star = local_delaunay_star(&points, dim, p, &nbrs[p], &tangent, d).expect("star");
        assert!(
            !star.is_empty(),
            "interior point should have a non-empty star"
        );

        // Re-verify the empty-ball predicate independently for each star simplex.
        let centre = &points[p * dim..p * dim + dim];
        let mut local_coords: Vec<(usize, Vec<f64>)> = Vec::new();
        local_coords.push((p, vec![0.0; d]));
        for &nb in &nbrs[p] {
            let row = &points[nb * dim..nb * dim + dim];
            let centred: Vec<f64> = (0..dim).map(|j| row[j] - centre[j]).collect();
            local_coords.push((nb, tangent.project(&centred)));
        }
        for simplex in &star {
            // Gather the local coords of this simplex's vertices.
            let coords: Vec<Vec<f64>> = simplex
                .iter()
                .map(|gid| {
                    local_coords
                        .iter()
                        .find(|(id, _)| id == gid)
                        .map(|(_, c)| c.clone())
                        .expect("vertex present")
                })
                .collect();
            let verts: Vec<usize> = (0..coords.len()).collect();
            let (cc, r2) = circumball(&coords, &verts, d, CIRCUMBALL_TOL).expect("circumball");
            // No other projected neighbour strictly inside the circumball.
            for (gid, c) in &local_coords {
                if simplex.contains(gid) {
                    continue;
                }
                let dsq: f64 = (0..d).map(|j| (c[j] - cc[j]).powi(2)).sum();
                assert!(
                    dsq >= r2 * (1.0 - 1.0e-6),
                    "star simplex {simplex:?} is not empty: point {gid} inside"
                );
            }
        }
    }

    // (e) Tangent PCA is EXACT for perfectly planar data: the estimated tangent
    //     spans exactly the data plane and the normal eigenvalues are exactly 0.
    #[test]
    fn tangent_pca_exact_for_planar_data() {
        // A tilted plane spanned by u = (1,0,1)/√2 and w = (0,1,0) in R^3.
        let u = [1.0 / 2.0_f64.sqrt(), 0.0, 1.0 / 2.0_f64.sqrt()];
        let w = [0.0, 1.0, 0.0];
        let mut points: Vec<f64> = Vec::new();
        for ia in 0..7 {
            for ib in 0..7 {
                let a = ia as f64 - 3.0;
                let b = ib as f64 - 3.0;
                for c in 0..3 {
                    points.push(a * u[c] + b * w[c]);
                }
            }
        }
        let tangent = tangent_at(&points, 3, 24, 2, 12);
        // Normal eigenvalue is exactly zero (perfectly planar).
        assert!(
            tangent.normal_eigenvalues[0].abs() < 1.0e-9,
            "non-zero normal variance for planar data: {}",
            tangent.normal_eigenvalues[0]
        );
        // The plane normal is u × w = (-1/√2, 0, 1/√2); the estimated normal must
        // be parallel to it ⇒ its tangent components vanish against u and w only
        // through the normal direction.  Check the estimated normal ⟂ both u, w.
        let normals = tangent.normals();
        let dot_u: f64 = (0..3).map(|c| normals[c] * u[c]).sum();
        let dot_w: f64 = (0..3).map(|c| normals[c] * w[c]).sum();
        assert!(dot_u.abs() < 1.0e-9, "normal not ⟂ u: {dot_u}");
        assert!(dot_w.abs() < 1.0e-9, "normal not ⟂ w: {dot_w}");
    }

    // (f) Degenerate / insufficient-neighbour cases error gracefully.
    #[test]
    fn degenerate_cases_error() {
        // Empty cloud.
        assert!(matches!(
            tangential_complex(&[], 3, 2, 5),
            Err(TdaError::EmptyPointCloud)
        ));
        // Zero ambient dimension.
        assert!(matches!(
            tangential_complex(&[1.0, 2.0], 0, 1, 3),
            Err(TdaError::DimensionMismatch { .. })
        ));
        // Manifold dimension exceeds ambient dimension.
        assert!(matches!(
            tangential_complex(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0], 2, 3, 2),
            Err(TdaError::DimensionTooLarge(3))
        ));
        // k == 0.
        assert!(matches!(
            tangential_complex(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0], 2, 1, 0),
            Err(TdaError::ParameterOutOfRange(_))
        ));
        // Too few points to form a d-simplex (2 points, d = 2 needs ≥ 3).
        assert!(matches!(
            tangential_complex(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3, 2, 1),
            Err(TdaError::LandmarkSelectionFailed(_))
        ));
        // Length not a multiple of ambient dim.
        assert!(matches!(
            tangential_complex(&[0.0, 0.0, 1.0], 2, 1, 1),
            Err(TdaError::DimensionMismatch { .. })
        ));
    }

    // Extra: the symmetric eigensolver itself is correct (sanity for the core
    // primitive the whole module rests on).
    #[test]
    fn jacobi_eig_diagonalises() {
        // A known 2x2 symmetric matrix [[2,1],[1,2]] has eigenvalues 3 and 1
        // with eigenvectors (1,1)/√2 and (1,-1)/√2.
        let m = vec![2.0, 1.0, 1.0, 2.0];
        let (eigs, vecs) = jacobi_symmetric_eig(&m, 2);
        assert!((eigs[0] - 3.0).abs() < 1.0e-10, "lambda0 = {}", eigs[0]);
        assert!((eigs[1] - 1.0).abs() < 1.0e-10, "lambda1 = {}", eigs[1]);
        // First eigenvector ∝ (1,1).
        assert!((vecs[0].abs() - vecs[1].abs()).abs() < 1.0e-10);
        // Eigenvectors orthonormal.
        let dot = vecs[0] * vecs[2] + vecs[1] * vecs[3];
        assert!(dot.abs() < 1.0e-10, "eigenvectors not orthogonal: {dot}");
    }

    #[test]
    fn project_round_trips_in_basis() {
        // Projecting a basis vector onto itself yields a coordinate of 1.
        let mut points: Vec<f64> = Vec::new();
        for ix in 0..5 {
            for iy in 0..5 {
                points.push(ix as f64);
                points.push(iy as f64);
                points.push(0.0);
            }
        }
        let tangent = tangent_at(&points, 3, 12, 2, 8);
        let e0 = tangent.basis[0..3].to_vec();
        let coords = tangent.project(&e0);
        assert!((coords[0] - 1.0).abs() < 1.0e-9, "self-projection ≠ 1");
        assert!(coords[1].abs() < 1.0e-9, "off-axis leak");
    }
}
