//! Mesh smoothing via Laplacian and Taubin (λ|μ) filters.
//!
//! Surface smoothing removes high-frequency noise from a triangle mesh by moving
//! each vertex toward the average of its one-ring neighbours. Two operators are
//! provided:
//!
//! * **Laplacian smoothing** — `v ← v + λ · L(v)` where `L(v)` is the umbrella
//!   (uniform-weight) Laplacian `mean(neighbours) − v`. Repeated iterations act
//!   as a low-pass filter but cause progressive *shrinkage* of the surface.
//! * **Taubin smoothing** (Taubin 1995) — alternates a positive Laplacian step
//!   `λ > 0` with a negative step `μ < 0` (with `μ < −λ`). The two passes form a
//!   band-pass transfer function that attenuates noise while preserving overall
//!   volume, eliminating the shrinkage of pure Laplacian smoothing.
//!
//! Meshes follow the crate convention: vertices are a flat `&[f64]` of length
//! `3 · n_vertices` (row-major `[x, y, z]`) and faces are `&[[usize; 3]]`.

use crate::error::{Geom3dError, Geom3dResult};

/// Build the one-ring (vertex→neighbour) adjacency lists from a triangle mesh.
///
/// Each undirected edge `(a, b)` of every face contributes `b` to `a`'s set and
/// `a` to `b`'s set. Duplicate neighbours (shared edges) are de-duplicated.
fn build_adjacency(n_vertices: usize, faces: &[[usize; 3]]) -> Geom3dResult<Vec<Vec<usize>>> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n_vertices];
    for f in faces {
        for k in 0..3 {
            let a = f[k];
            let b = f[(k + 1) % 3];
            if a >= n_vertices || b >= n_vertices {
                return Err(Geom3dError::InvalidTopology {
                    reason: "face references out-of-range vertex",
                });
            }
            if a != b {
                if !adj[a].contains(&b) {
                    adj[a].push(b);
                }
                if !adj[b].contains(&a) {
                    adj[b].push(a);
                }
            }
        }
    }
    Ok(adj)
}

/// Apply one uniform-Laplacian displacement step to `vertices` in place.
///
/// For each vertex `i` with neighbours `N(i)`, the new position is
/// `v_i + factor · (mean_{j∈N(i)} v_j − v_i)`. Isolated vertices (no neighbours)
/// are left unchanged.
fn laplacian_step(vertices: &mut [f64], adj: &[Vec<usize>], factor: f64) {
    let n = adj.len();
    let mut deltas = vec![0.0f64; vertices.len()];
    for i in 0..n {
        let neighbours = &adj[i];
        if neighbours.is_empty() {
            continue;
        }
        let inv = 1.0 / neighbours.len() as f64;
        let mut centroid = [0.0f64; 3];
        for &j in neighbours {
            centroid[0] += vertices[j * 3];
            centroid[1] += vertices[j * 3 + 1];
            centroid[2] += vertices[j * 3 + 2];
        }
        for c in &mut centroid {
            *c *= inv;
        }
        deltas[i * 3] = factor * (centroid[0] - vertices[i * 3]);
        deltas[i * 3 + 1] = factor * (centroid[1] - vertices[i * 3 + 1]);
        deltas[i * 3 + 2] = factor * (centroid[2] - vertices[i * 3 + 2]);
    }
    for (v, d) in vertices.iter_mut().zip(deltas.iter()) {
        *v += d;
    }
}

/// Smooth a mesh with `iterations` uniform-Laplacian passes of strength `lambda`.
///
/// Returns the new vertex buffer (faces are unchanged and need not be returned).
/// `lambda` is typically in `(0, 1]`; larger values smooth faster but may
/// over-shoot for `lambda > 1`.
///
/// # Errors
/// * [`Geom3dError::EmptyPointCloud`] if `vertices` is empty.
/// * [`Geom3dError::InvalidPointDim`] if `vertices.len()` is not a multiple of 3.
/// * [`Geom3dError::Internal`] if `lambda` is not finite.
/// * [`Geom3dError::InvalidTopology`] if a face indexes a missing vertex.
pub fn laplacian_smooth(
    vertices: &[f64],
    faces: &[[usize; 3]],
    lambda: f64,
    iterations: usize,
) -> Geom3dResult<Vec<f64>> {
    if vertices.is_empty() {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if vertices.len() % 3 != 0 {
        return Err(Geom3dError::InvalidPointDim {
            dim: vertices.len() % 3,
        });
    }
    if !lambda.is_finite() {
        return Err(Geom3dError::Internal("lambda must be finite".into()));
    }
    let n_vertices = vertices.len() / 3;
    let adj = build_adjacency(n_vertices, faces)?;
    let mut out = vertices.to_vec();
    for _ in 0..iterations {
        laplacian_step(&mut out, &adj, lambda);
    }
    Ok(out)
}

/// Smooth a mesh with Taubin's λ|μ band-pass filter.
///
/// Each of the `iterations` passes performs a shrinking Laplacian step with
/// `lambda > 0` followed by an inflating step with `mu < 0`. To suppress
/// shrinkage the pass-band edge condition `mu < −lambda` should hold; a common
/// choice is `lambda = 0.5`, `mu = −0.53`.
///
/// # Errors
/// * [`Geom3dError::EmptyPointCloud`] if `vertices` is empty.
/// * [`Geom3dError::InvalidPointDim`] if `vertices.len()` is not a multiple of 3.
/// * [`Geom3dError::Internal`] if `lambda`/`mu` are not finite or `lambda <= 0`
///   or `mu >= 0`.
/// * [`Geom3dError::InvalidTopology`] if a face indexes a missing vertex.
pub fn taubin_smooth(
    vertices: &[f64],
    faces: &[[usize; 3]],
    lambda: f64,
    mu: f64,
    iterations: usize,
) -> Geom3dResult<Vec<f64>> {
    if vertices.is_empty() {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if vertices.len() % 3 != 0 {
        return Err(Geom3dError::InvalidPointDim {
            dim: vertices.len() % 3,
        });
    }
    if !lambda.is_finite() || !mu.is_finite() {
        return Err(Geom3dError::Internal("lambda/mu must be finite".into()));
    }
    if lambda <= 0.0 || mu >= 0.0 {
        return Err(Geom3dError::Internal(
            "Taubin requires lambda > 0 and mu < 0".into(),
        ));
    }
    let n_vertices = vertices.len() / 3;
    let adj = build_adjacency(n_vertices, faces)?;
    let mut out = vertices.to_vec();
    for _ in 0..iterations {
        laplacian_step(&mut out, &adj, lambda); // shrink
        laplacian_step(&mut out, &adj, mu); // un-shrink
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::curvature::icosphere;

    /// A unit cube as 8 vertices + 12 triangle faces.
    fn unit_cube() -> (Vec<f64>, Vec<[usize; 3]>) {
        let v = vec![
            0.0, 0.0, 0.0, // 0
            1.0, 0.0, 0.0, // 1
            1.0, 1.0, 0.0, // 2
            0.0, 1.0, 0.0, // 3
            0.0, 0.0, 1.0, // 4
            1.0, 0.0, 1.0, // 5
            1.0, 1.0, 1.0, // 6
            0.0, 1.0, 1.0, // 7
        ];
        let f = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 4, 7],
            [0, 7, 3],
        ];
        (v, f)
    }

    fn centroid(v: &[f64]) -> [f64; 3] {
        let n = v.len() / 3;
        let mut c = [0.0; 3];
        for i in 0..n {
            c[0] += v[i * 3];
            c[1] += v[i * 3 + 1];
            c[2] += v[i * 3 + 2];
        }
        [c[0] / n as f64, c[1] / n as f64, c[2] / n as f64]
    }

    fn bbox_diag(v: &[f64]) -> f64 {
        let n = v.len() / 3;
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for i in 0..n {
            for k in 0..3 {
                lo[k] = lo[k].min(v[i * 3 + k]);
                hi[k] = hi[k].max(v[i * 3 + k]);
            }
        }
        ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt()
    }

    #[test]
    fn adjacency_cube_each_vertex_has_neighbours() {
        let (_, faces) = unit_cube();
        let adj = build_adjacency(8, &faces).expect("build_adjacency should succeed");
        for (i, nbrs) in adj.iter().enumerate() {
            assert!(!nbrs.is_empty(), "vertex {i} has no neighbours");
            assert!(!nbrs.contains(&i), "vertex {i} is its own neighbour");
        }
    }

    #[test]
    fn adjacency_rejects_bad_face() {
        let faces = [[0usize, 1, 99]];
        assert!(build_adjacency(3, &faces).is_err());
    }

    #[test]
    fn laplacian_empty_errors() {
        assert!(laplacian_smooth(&[], &[], 0.5, 1).is_err());
    }

    #[test]
    fn laplacian_bad_dim_errors() {
        assert!(laplacian_smooth(&[0.0, 1.0], &[], 0.5, 1).is_err());
    }

    #[test]
    fn laplacian_nonfinite_lambda_errors() {
        let (v, f) = unit_cube();
        assert!(laplacian_smooth(&v, &f, f64::NAN, 1).is_err());
    }

    #[test]
    fn laplacian_preserves_vertex_count() {
        let (v, f) = unit_cube();
        let out = laplacian_smooth(&v, &f, 0.5, 3).expect("laplacian_smooth should succeed");
        assert_eq!(out.len(), v.len());
    }

    #[test]
    fn laplacian_preserves_centroid_on_regular_mesh() {
        // On a *regular* mesh (every vertex the same valence with a symmetric
        // one-ring) the uniform Laplacian is centroid-preserving. A single
        // triangle is the minimal regular case: each vertex's neighbours are the
        // other two, so Σ_i L(v_i) = 0 exactly.
        let v = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0];
        let f = vec![[0usize, 1, 2]];
        let c0 = centroid(&v);
        let out = laplacian_smooth(&v, &f, 0.5, 5).expect("laplacian_smooth should succeed");
        let c1 = centroid(&out);
        for k in 0..3 {
            assert!((c0[k] - c1[k]).abs() < 1e-9, "centroid drift axis {k}");
        }
    }

    #[test]
    fn laplacian_centroid_drift_is_bounded_on_irregular_mesh() {
        // On an irregular triangulation the centroid is not exactly preserved,
        // but the drift stays small (bounded by the smoothing strength × extent).
        let (v, f) = unit_cube();
        let c0 = centroid(&v);
        let out = laplacian_smooth(&v, &f, 0.5, 5).expect("laplacian_smooth should succeed");
        let c1 = centroid(&out);
        for k in 0..3 {
            assert!(
                (c0[k] - c1[k]).abs() < 0.2,
                "centroid drift axis {k} too large"
            );
        }
    }

    #[test]
    fn laplacian_shrinks_sphere() {
        // Pure Laplacian smoothing shrinks a closed surface.
        let (v, f) = icosphere(2);
        let d0 = bbox_diag(&v);
        let out = laplacian_smooth(&v, &f, 0.5, 10).expect("laplacian_smooth should succeed");
        let d1 = bbox_diag(&out);
        assert!(d1 < d0, "expected shrinkage: {d0} → {d1}");
    }

    #[test]
    fn laplacian_zero_iterations_is_identity() {
        let (v, f) = unit_cube();
        let out = laplacian_smooth(&v, &f, 0.5, 0).expect("laplacian_smooth should succeed");
        assert_eq!(out, v);
    }

    #[test]
    fn taubin_rejects_bad_signs() {
        let (v, f) = unit_cube();
        assert!(taubin_smooth(&v, &f, -0.5, -0.53, 1).is_err());
        assert!(taubin_smooth(&v, &f, 0.5, 0.53, 1).is_err());
    }

    #[test]
    fn taubin_preserves_vertex_count() {
        let (v, f) = icosphere(2);
        let out = taubin_smooth(&v, &f, 0.5, -0.53, 5).expect("taubin_smooth should succeed");
        assert_eq!(out.len(), v.len());
    }

    #[test]
    fn taubin_resists_shrinkage_better_than_laplacian() {
        // Taubin's μ-pass counteracts the λ-pass shrinkage, so after equal
        // passes its bounding box is larger (closer to original) than pure
        // Laplacian's.
        let (v, f) = icosphere(3);
        let d0 = bbox_diag(&v);
        let lap = laplacian_smooth(&v, &f, 0.5, 10).expect("laplacian_smooth should succeed");
        let tau = taubin_smooth(&v, &f, 0.5, -0.53, 5).expect("taubin_smooth should succeed");
        let d_lap = bbox_diag(&lap);
        let d_tau = bbox_diag(&tau);
        assert!(
            d_tau > d_lap,
            "Taubin should shrink less: laplacian={d_lap}, taubin={d_tau}, orig={d0}"
        );
    }

    #[test]
    fn taubin_smooths_noisy_sphere() {
        // Add radial noise to a sphere, then check Taubin reduces the variance of
        // vertex radii (noise removal) while keeping the mesh near unit radius.
        let (mut v, f) = icosphere(3);
        let n = v.len() / 3;
        // Deterministic pseudo-noise via index hashing.
        for i in 0..n {
            let h = ((i.wrapping_mul(2654435761)) & 0xffff) as f64 / 65535.0 - 0.5;
            let r = (v[i * 3].powi(2) + v[i * 3 + 1].powi(2) + v[i * 3 + 2].powi(2)).sqrt();
            let scale = (r + 0.15 * h) / r;
            for k in 0..3 {
                v[i * 3 + k] *= scale;
            }
        }
        let radius_var = |buf: &[f64]| {
            let m = buf.len() / 3;
            let radii: Vec<f64> = (0..m)
                .map(|i| {
                    (buf[i * 3].powi(2) + buf[i * 3 + 1].powi(2) + buf[i * 3 + 2].powi(2)).sqrt()
                })
                .collect();
            let mean = radii.iter().sum::<f64>() / m as f64;
            radii.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / m as f64
        };
        let var_before = radius_var(&v);
        let out = taubin_smooth(&v, &f, 0.5, -0.53, 8).expect("taubin_smooth should succeed");
        let var_after = radius_var(&out);
        assert!(
            var_after < var_before,
            "Taubin should reduce radius variance: {var_before} → {var_after}"
        );
    }

    #[test]
    fn taubin_empty_errors() {
        assert!(taubin_smooth(&[], &[], 0.5, -0.53, 1).is_err());
    }
}
