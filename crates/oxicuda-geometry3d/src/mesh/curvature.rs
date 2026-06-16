//! Discrete differential-geometry curvature on triangle meshes.
//!
//! Implements the operators of Meyer, Desbrun, Schröder & Barr, "Discrete
//! Differential-Geometry Operators for Triangulated 2-Manifolds" (2003):
//!
//! - **Gaussian curvature** via the angle defect
//!   `K_i = (2π − Σ_t θ_{i,t}) / A_mixed_i`.
//! - **Mean curvature** via the cotangent Laplace-Beltrami operator
//!   `K(x_i) = 1/(2 A_mixed_i) Σ_{j∈N(i)} (cot α_{ij} + cot β_{ij})(x_i − x_j)`,
//!   with `H_i = ½·|K(x_i)|`.
//! - **Principal curvatures** `κ₁,₂ = H ± √(max(H²−K, 0))`.
//!
//! `A_mixed` is the per-vertex mixed area: Voronoi area for non-obtuse
//! triangles, and a barycentric split for obtuse ones (½ or ¼ of the triangle
//! area depending on whether the obtuse angle is at the vertex), exactly as in
//! Meyer et al. §3.3.
//!
//! A unit-sphere [`icosphere`] is provided as a test helper (and is `pub` so
//! integration tests can build analytic-oracle meshes).

use crate::error::{Geom3dError, Geom3dResult};

/// Per-vertex discrete curvature quantities (all length `n`).
#[derive(Debug, Clone)]
pub struct VertexCurvature {
    /// Mean curvature `H_i = ½·|Laplace-Beltrami(x_i)|`.
    pub mean: Vec<f64>,
    /// Gaussian curvature `K_i` (angle defect over mixed area).
    pub gaussian: Vec<f64>,
    /// Maximal principal curvature `κ₁ = H + √(max(H²−K,0))`.
    pub k1: Vec<f64>,
    /// Minimal principal curvature `κ₂ = H − √(max(H²−K,0))`.
    pub k2: Vec<f64>,
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Cotangent of the angle between vectors `u` and `v` (sharing an apex):
/// `cot θ = (u·v) / |u×v|`, guarded against a zero/degenerate cross product.
#[inline]
fn cotangent(u: [f64; 3], v: [f64; 3]) -> f64 {
    let c = norm(cross(u, v));
    if c < 1e-14 { 0.0 } else { dot(u, v) / c }
}

/// Interior angle (radians) of the triangle at the apex `apex`, between the two
/// edges to `q` and `r`. Numerically stable via `atan2`.
#[inline]
fn corner_angle(apex: [f64; 3], q: [f64; 3], r: [f64; 3]) -> f64 {
    let u = sub(q, apex);
    let v = sub(r, apex);
    let cr = norm(cross(u, v));
    let dt = dot(u, v);
    cr.atan2(dt)
}

/// Fetch vertex `i` from a flat `[n*3]` buffer.
#[inline]
fn fetch(vertices: &[f64], i: usize) -> [f64; 3] {
    [vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]]
}

/// Compute discrete mean / Gaussian / principal curvature per vertex.
///
/// `vertices` is flat `[n*3]` (row-major `x,y,z`); `triangles` lists vertex
/// index triples. Boundary vertices (whose incident triangles do not close a
/// full umbrella) still receive an angle-defect value, but the cotangent
/// Laplacian only accumulates the contributions of the triangles actually
/// present, so open-mesh boundaries are handled without panicking.
///
/// # Errors
/// - [`Geom3dError::EmptyPointCloud`] if `n == 0`.
/// - [`Geom3dError::DimensionMismatch`] if `vertices.len() != n*3`.
/// - [`Geom3dError::InvalidTopology`] if any triangle indexes a vertex `>= n`.
/// - [`Geom3dError::NanEncountered`] if any coordinate is non-finite.
pub fn discrete_curvature(
    vertices: &[f64],
    n: usize,
    triangles: &[[usize; 3]],
) -> Geom3dResult<VertexCurvature> {
    if n == 0 {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if vertices.len() != n * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: n * 3,
            got: vertices.len(),
        });
    }
    for &v in vertices {
        if !v.is_finite() {
            return Err(Geom3dError::NanEncountered {
                location: "curvature::discrete_curvature vertices",
            });
        }
    }
    for tri in triangles {
        if tri[0] >= n || tri[1] >= n || tri[2] >= n {
            return Err(Geom3dError::InvalidTopology {
                reason: "triangle references vertex index out of range",
            });
        }
    }

    let mut angle_sum = vec![0.0_f64; n]; // Σ interior angles at vertex
    let mut area_mixed = vec![0.0_f64; n]; // A_mixed per vertex
    let mut laplacian = vec![[0.0_f64; 3]; n]; // cotangent Laplace-Beltrami

    for tri in triangles {
        let (i0, i1, i2) = (tri[0], tri[1], tri[2]);
        let p0 = fetch(vertices, i0);
        let p1 = fetch(vertices, i1);
        let p2 = fetch(vertices, i2);

        // Interior angles at each corner.
        let a0 = corner_angle(p0, p1, p2);
        let a1 = corner_angle(p1, p2, p0);
        let a2 = corner_angle(p2, p0, p1);
        angle_sum[i0] += a0;
        angle_sum[i1] += a1;
        angle_sum[i2] += a2;

        // Triangle area.
        let area = 0.5 * norm(cross(sub(p1, p0), sub(p2, p0)));
        if area < 1e-20 {
            continue; // Degenerate sliver: skip its area / Laplacian terms.
        }

        // Cotangents of the three corner angles.
        let cot0 = cotangent(sub(p1, p0), sub(p2, p0));
        let cot1 = cotangent(sub(p2, p1), sub(p0, p1));
        let cot2 = cotangent(sub(p0, p2), sub(p1, p2));

        // ── Mixed area (Meyer et al. §3.3). ─────────────────────────────────
        let obtuse_at = |angle: f64| angle > std::f64::consts::FRAC_PI_2;
        if obtuse_at(a0) || obtuse_at(a1) || obtuse_at(a2) {
            // Obtuse triangle: barycentric split (½ at the obtuse vertex, ¼ at
            // the others).
            area_mixed[i0] += if obtuse_at(a0) {
                area / 2.0
            } else {
                area / 4.0
            };
            area_mixed[i1] += if obtuse_at(a1) {
                area / 2.0
            } else {
                area / 4.0
            };
            area_mixed[i2] += if obtuse_at(a2) {
                area / 2.0
            } else {
                area / 4.0
            };
        } else {
            // Non-obtuse: Voronoi area. The portion belonging to a vertex is
            // ⅛ (cot of the angle opposite each of its two edges)·|edge|².
            let l01 = dot(sub(p1, p0), sub(p1, p0));
            let l12 = dot(sub(p2, p1), sub(p2, p1));
            let l20 = dot(sub(p0, p2), sub(p0, p2));
            // Vertex 0 sees edges (0-1) opp angle a2 and (0-2) opp angle a1.
            area_mixed[i0] += (cot2 * l01 + cot1 * l20) / 8.0;
            area_mixed[i1] += (cot0 * l12 + cot2 * l01) / 8.0;
            area_mixed[i2] += (cot1 * l20 + cot0 * l12) / 8.0;
        }

        // ── Cotangent Laplace-Beltrami accumulation. ───────────────────────
        // For edge (i,j) the weight is (cot of the two angles opposite it). We
        // add per-triangle the single opposite-angle contribution; the second
        // opposite angle arrives from the adjacent triangle sharing the edge.
        // Edge (0-1) is opposite corner 2 → weight cot2.
        accumulate_edge(&mut laplacian, i0, i1, p0, p1, cot2);
        // Edge (1-2) opposite corner 0 → weight cot0.
        accumulate_edge(&mut laplacian, i1, i2, p1, p2, cot0);
        // Edge (2-0) opposite corner 1 → weight cot1.
        accumulate_edge(&mut laplacian, i2, i0, p2, p0, cot1);
    }

    let two_pi = 2.0 * std::f64::consts::PI;
    let mut mean = vec![0.0_f64; n];
    let mut gaussian = vec![0.0_f64; n];
    let mut k1 = vec![0.0_f64; n];
    let mut k2 = vec![0.0_f64; n];

    for i in 0..n {
        let area = area_mixed[i];
        if area < 1e-20 {
            // Isolated / unreferenced vertex: leave all curvatures at zero.
            continue;
        }
        // Gaussian (angle defect).
        let k = (two_pi - angle_sum[i]) / area;
        // Mean: H = ½ |K(x_i)|, K(x_i) = laplacian / (2 A_mixed).
        let lap = laplacian[i];
        let kvec = [
            lap[0] / (2.0 * area),
            lap[1] / (2.0 * area),
            lap[2] / (2.0 * area),
        ];
        let h = 0.5 * norm(kvec);

        let disc = (h * h - k).max(0.0).sqrt();
        gaussian[i] = k;
        mean[i] = h;
        k1[i] = h + disc;
        k2[i] = h - disc;
    }

    Ok(VertexCurvature {
        mean,
        gaussian,
        k1,
        k2,
    })
}

/// Accumulate `w * (x_i − x_j)` into both endpoints' Laplacian (antisymmetric).
#[inline]
fn accumulate_edge(lap: &mut [[f64; 3]], i: usize, j: usize, pi: [f64; 3], pj: [f64; 3], w: f64) {
    let d = sub(pi, pj); // x_i − x_j
    lap[i][0] += w * d[0];
    lap[i][1] += w * d[1];
    lap[i][2] += w * d[2];
    lap[j][0] -= w * d[0];
    lap[j][1] -= w * d[1];
    lap[j][2] -= w * d[2];
}

/// Build a unit icosphere by recursively subdividing an icosahedron.
///
/// Starts from the 12 canonical icosahedron vertices (built from the golden
/// ratio) and 20 faces, and performs `subdivisions` rounds of 1→4 triangle
/// splitting, projecting every new midpoint back onto the unit sphere. Returns
/// `(vertices_flat[v*3], triangles)`.
pub fn icosphere(subdivisions: usize) -> (Vec<f64>, Vec<[usize; 3]>) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    // 12 icosahedron vertices (un-normalized), then normalized to the sphere.
    let raw = [
        [-1.0, phi, 0.0],
        [1.0, phi, 0.0],
        [-1.0, -phi, 0.0],
        [1.0, -phi, 0.0],
        [0.0, -1.0, phi],
        [0.0, 1.0, phi],
        [0.0, -1.0, -phi],
        [0.0, 1.0, -phi],
        [phi, 0.0, -1.0],
        [phi, 0.0, 1.0],
        [-phi, 0.0, -1.0],
        [-phi, 0.0, 1.0],
    ];
    let mut verts: Vec<[f64; 3]> = raw
        .iter()
        .map(|v| {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / l, v[1] / l, v[2] / l]
        })
        .collect();

    // 20 icosahedron faces.
    let mut faces: Vec<[usize; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    use std::collections::HashMap;
    for _ in 0..subdivisions {
        let mut midpoint: HashMap<(usize, usize), usize> = HashMap::new();
        let mut new_faces: Vec<[usize; 3]> = Vec::with_capacity(faces.len() * 4);

        let mut get_mid = |a: usize, b: usize, verts: &mut Vec<[f64; 3]>| -> usize {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&m) = midpoint.get(&key) {
                return m;
            }
            let va = verts[a];
            let vb = verts[b];
            let mut mid = [
                0.5 * (va[0] + vb[0]),
                0.5 * (va[1] + vb[1]),
                0.5 * (va[2] + vb[2]),
            ];
            let l = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]).sqrt();
            mid = [mid[0] / l, mid[1] / l, mid[2] / l]; // project to unit sphere
            let idx = verts.len();
            verts.push(mid);
            midpoint.insert(key, idx);
            idx
        };

        for f in &faces {
            let a = get_mid(f[0], f[1], &mut verts);
            let b = get_mid(f[1], f[2], &mut verts);
            let c = get_mid(f[2], f[0], &mut verts);
            new_faces.push([f[0], a, c]);
            new_faces.push([f[1], b, a]);
            new_faces.push([f[2], c, b]);
            new_faces.push([a, b, c]);
        }
        faces = new_faces;
    }

    let mut flat = Vec::with_capacity(verts.len() * 3);
    for v in &verts {
        flat.push(v[0]);
        flat.push(v[1]);
        flat.push(v[2]);
    }
    (flat, faces)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icosphere_counts() {
        // Subdivision multiplies faces by 4 each round; 20·4^k faces,
        // 10·4^k+2 vertices (Euler).
        let (v0, f0) = icosphere(0);
        assert_eq!(v0.len() / 3, 12);
        assert_eq!(f0.len(), 20);
        let (v2, f2) = icosphere(2);
        assert_eq!(f2.len(), 20 * 16);
        assert_eq!(v2.len() / 3, 10 * 16 + 2);
        // All vertices lie on the unit sphere.
        for c in v2.chunks_exact(3) {
            let r = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            assert!(
                (r - 1.0).abs() < 1e-9,
                "icosphere vertex off the sphere: {r}"
            );
        }
    }

    #[test]
    fn unit_sphere_curvatures() {
        // Analytic oracle: unit sphere has K = 1/r² = 1 and H = 1/r = 1.
        let (verts, tris) = icosphere(3);
        let n = verts.len() / 3;
        let c = discrete_curvature(&verts, n, &tris).expect("discrete_curvature should succeed");

        let mean_k: f64 = c.gaussian.iter().sum::<f64>() / n as f64;
        let mean_h: f64 = c.mean.iter().sum::<f64>() / n as f64;
        assert!(
            (mean_k - 1.0).abs() < 0.12,
            "mean Gaussian curvature ≈ 1, got {mean_k}"
        );
        assert!(
            (mean_h - 1.0).abs() < 0.12,
            "mean mean-curvature ≈ 1, got {mean_h}"
        );

        // Principal curvatures both ≈ 1 (umbilic sphere).
        let mean_k1: f64 = c.k1.iter().sum::<f64>() / n as f64;
        let mean_k2: f64 = c.k2.iter().sum::<f64>() / n as f64;
        assert!((mean_k1 - 1.0).abs() < 0.2, "κ₁ ≈ 1, got {mean_k1}");
        assert!((mean_k2 - 1.0).abs() < 0.2, "κ₂ ≈ 1, got {mean_k2}");
    }

    #[test]
    fn gauss_bonnet() {
        // Σ_i K_i · A_mixed_i ≈ 2π·χ = 4π for a sphere (χ = 2).
        let (verts, tris) = icosphere(3);
        let n = verts.len() / 3;

        // Recompute A_mixed by re-deriving from the public curvature is not
        // possible directly, so reconstruct the integral via angle defects:
        // Σ K_i A_i = Σ (2π − Σθ). We instead validate by integrating K·A using
        // the same area as the curvature routine — recompute area defects.
        let c = discrete_curvature(&verts, n, &tris).expect("discrete_curvature should succeed");

        // Reconstruct A_mixed from the angle-defect identity:
        // K_i = (2π − Σθ_i)/A_i ⇒ A_i = (2π − Σθ_i)/K_i. Summing K_i·A_i then
        // equals Σ (2π − Σθ_i), which is the total angle defect = 4π. Compute
        // the total angle defect directly to validate Gauss-Bonnet.
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut angle_sum = vec![0.0_f64; n];
        for tri in &tris {
            let p0 = fetch(&verts, tri[0]);
            let p1 = fetch(&verts, tri[1]);
            let p2 = fetch(&verts, tri[2]);
            angle_sum[tri[0]] += corner_angle(p0, p1, p2);
            angle_sum[tri[1]] += corner_angle(p1, p2, p0);
            angle_sum[tri[2]] += corner_angle(p2, p0, p1);
        }
        let total_defect: f64 = (0..n).map(|i| two_pi - angle_sum[i]).sum();
        let four_pi = 4.0 * std::f64::consts::PI;
        assert!(
            (total_defect - four_pi).abs() < 1e-6,
            "Gauss-Bonnet: total angle defect should be 4π, got {total_defect}"
        );
        // And the per-vertex Gaussian curvature is positive everywhere.
        assert!(c.gaussian.iter().all(|&k| k > 0.0));
    }

    #[test]
    fn flat_grid_zero_curvature() {
        // A flat triangulated square in z=0: interior vertices have K≈0, H≈0.
        let res = 5usize;
        let mut verts = Vec::new();
        for j in 0..=res {
            for i in 0..=res {
                verts.push(i as f64 / res as f64);
                verts.push(j as f64 / res as f64);
                verts.push(0.0);
            }
        }
        let w = res + 1;
        let mut tris = Vec::new();
        for j in 0..res {
            for i in 0..res {
                let a = j * w + i;
                let b = j * w + i + 1;
                let cc = (j + 1) * w + i;
                let d = (j + 1) * w + i + 1;
                tris.push([a, b, d]);
                tris.push([a, d, cc]);
            }
        }
        let n = verts.len() / 3;
        let c = discrete_curvature(&verts, n, &tris).expect("discrete_curvature should succeed");
        // Check interior vertices (not on the boundary).
        for j in 1..res {
            for i in 1..res {
                let idx = j * w + i;
                assert!(
                    c.gaussian[idx].abs() < 1e-6,
                    "flat K≈0, got {}",
                    c.gaussian[idx]
                );
                assert!(c.mean[idx].abs() < 1e-6, "flat H≈0, got {}", c.mean[idx]);
            }
        }
    }

    #[test]
    fn errors_on_bad_input() {
        assert!(matches!(
            discrete_curvature(&[], 0, &[]),
            Err(Geom3dError::EmptyPointCloud)
        ));
        assert!(matches!(
            discrete_curvature(&[0.0; 6], 4, &[]),
            Err(Geom3dError::DimensionMismatch { .. })
        ));
        let verts = vec![0.0; 9];
        assert!(matches!(
            discrete_curvature(&verts, 3, &[[0, 1, 5]]),
            Err(Geom3dError::InvalidTopology { .. })
        ));
    }
}
