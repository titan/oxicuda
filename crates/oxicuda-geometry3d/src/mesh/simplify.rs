//! Mesh simplification via Quadric Error Metrics (Garland & Heckbert 1997).
//!
//! Surface simplification reduces a triangle mesh's face count while preserving
//! its shape. The Garland–Heckbert algorithm assigns every vertex a `4×4`
//! *quadric* `Q` that measures the summed squared distance to the planes of its
//! incident faces. The error of placing a vertex at homogeneous position
//! `v = (x, y, z, 1)` is the quadratic form `vᵀ Q v`.
//!
//! Each candidate **edge collapse** `(i, j) → v̄` merges two vertices; its cost
//! is `v̄ᵀ (Q_i + Q_j) v̄`, minimised over the placement `v̄`. Collapses are
//! applied greedily in increasing cost until the target triangle count is
//! reached (or no further valid collapse exists). Degenerate and topology-
//! breaking collapses (those that would flip a face normal or collapse a
//! triangle to a line) are skipped.
//!
//! This is an *exact, deterministic* CPU implementation: the priority order is a
//! recomputed minimum each step (`O(E)` per collapse), which is simple and
//! robust for the moderate mesh sizes used in tests and pre-processing.
//!
//! Meshes follow the crate convention: vertices are a flat `Vec<f64>` of length
//! `3 · n_vertices` (`[x, y, z]` row-major) and faces are `Vec<[usize; 3]>`.

use std::collections::HashSet;

use crate::error::{Geom3dError, Geom3dResult};

/// A symmetric `4×4` quadric stored as its 10 upper-triangular entries.
///
/// Layout (row-major upper triangle):
/// `[a00, a01, a02, a03, a11, a12, a13, a22, a23, a33]`.
#[derive(Debug, Clone, Copy, Default)]
struct Quadric {
    m: [f64; 10],
}

impl Quadric {
    /// Quadric of a plane `(a, b, c, d)` with `a²+b²+c² = 1`: the outer product
    /// `p pᵀ`, whose form `vᵀ Q v = (p · v)²` is the squared point-plane distance.
    fn from_plane(p: [f64; 4]) -> Self {
        let [a, b, c, d] = p;
        Self {
            m: [
                a * a,
                a * b,
                a * c,
                a * d,
                b * b,
                b * c,
                b * d,
                c * c,
                c * d,
                d * d,
            ],
        }
    }

    fn add(&self, other: &Quadric) -> Quadric {
        let mut m = [0.0; 10];
        for (slot, (a, b)) in m.iter_mut().zip(self.m.iter().zip(other.m.iter())) {
            *slot = a + b;
        }
        Quadric { m }
    }

    /// Evaluate `vᵀ Q v` for `v = (x, y, z, 1)`.
    fn error_at(&self, x: f64, y: f64, z: f64) -> f64 {
        let m = &self.m;
        // Symmetric expansion of vᵀ Q v with the off-diagonals doubled.
        m[0] * x * x
            + 2.0 * m[1] * x * y
            + 2.0 * m[2] * x * z
            + 2.0 * m[3] * x
            + m[4] * y * y
            + 2.0 * m[5] * y * z
            + 2.0 * m[6] * y
            + m[7] * z * z
            + 2.0 * m[8] * z
            + m[9]
    }

    /// Solve for the optimal contraction point `v̄` minimising `vᵀ Q v`.
    ///
    /// This requires solving the `3×3` linear system formed by the gradient
    /// `∂(vᵀQv)/∂(x,y,z) = 0`. Returns `None` when the system is singular (the
    /// caller then falls back to the edge midpoint / endpoints).
    fn optimal_point(&self) -> Option<[f64; 3]> {
        let m = &self.m;
        // Upper-left 3×3 block A and right-hand side b = −(a03, a13, a23).
        let a = [
            m[0], m[1], m[2], //
            m[1], m[4], m[5], //
            m[2], m[5], m[7], //
        ];
        let b = [-m[3], -m[6], -m[8]];
        solve_3x3(&a, &b)
    }
}

/// Solve `A x = b` for a `3×3` `A` (row-major) via Cramer's rule.
fn solve_3x3(a: &[f64; 9], b: &[f64; 3]) -> Option<[f64; 3]> {
    let det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    let dx = b[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (b[1] * a[8] - a[5] * b[2])
        + a[2] * (b[1] * a[7] - a[4] * b[2]);
    let dy = a[0] * (b[1] * a[8] - a[5] * b[2]) - b[0] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * b[2] - b[1] * a[6]);
    let dz = a[0] * (a[4] * b[2] - b[1] * a[7]) - a[1] * (a[3] * b[2] - b[1] * a[6])
        + b[0] * (a[3] * a[7] - a[4] * a[6]);
    Some([dx * inv_det, dy * inv_det, dz * inv_det])
}

/// Result of [`simplify_mesh`].
#[derive(Debug, Clone)]
pub struct SimplifyResult {
    /// Compacted vertex buffer (flat `[x, y, z]` row-major).
    pub vertices: Vec<f64>,
    /// Triangle faces indexing into [`Self::vertices`].
    pub faces: Vec<[usize; 3]>,
    /// Number of edge collapses performed.
    pub collapses: usize,
}

#[inline]
fn vget(v: &[f64], i: usize) -> [f64; 3] {
    [v[i * 3], v[i * 3 + 1], v[i * 3 + 2]]
}

/// Plane `(a, b, c, d)` of a triangle, normal normalised; `None` if degenerate.
fn face_plane(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> Option<[f64; 4]> {
    let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len < 1e-14 {
        return None;
    }
    let nn = [n[0] / len, n[1] / len, n[2] / len];
    let d = -(nn[0] * p0[0] + nn[1] * p0[1] + nn[2] * p0[2]);
    Some([nn[0], nn[1], nn[2], d])
}

/// Unnormalised face normal (for orientation / flip detection).
fn face_normal(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> [f64; 3] {
    let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ]
}

/// Simplify a triangle mesh to at most `target_faces` triangles via QEM edge
/// collapse.
///
/// Returns a compacted mesh (unreferenced vertices removed). The simplification
/// stops early if no further collapse can be applied without breaking the
/// surface. When `target_faces` is `>=` the input face count, the input is
/// returned unchanged (after compaction).
///
/// # Errors
/// * [`Geom3dError::EmptyPointCloud`] if `vertices` is empty.
/// * [`Geom3dError::InvalidPointDim`] if `vertices.len()` is not a multiple of 3.
/// * [`Geom3dError::InvalidTopology`] if a face indexes a missing vertex.
pub fn simplify_mesh(
    vertices: &[f64],
    faces: &[[usize; 3]],
    target_faces: usize,
) -> Geom3dResult<SimplifyResult> {
    if vertices.is_empty() {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if vertices.len() % 3 != 0 {
        return Err(Geom3dError::InvalidPointDim {
            dim: vertices.len() % 3,
        });
    }
    let n_vertices = vertices.len() / 3;
    for f in faces {
        if f[0] >= n_vertices || f[1] >= n_vertices || f[2] >= n_vertices {
            return Err(Geom3dError::InvalidTopology {
                reason: "face references out-of-range vertex",
            });
        }
    }

    let mut pos = vertices.to_vec();
    // Mutable face list; collapsed/degenerate faces are marked invalid (None).
    let mut tris: Vec<Option<[usize; 3]>> = faces.iter().map(|&f| Some(f)).collect();

    // Per-vertex quadric = sum of incident face plane quadrics.
    let mut quadrics = vec![Quadric::default(); n_vertices];
    for f in faces {
        if let Some(plane) = face_plane(vget(&pos, f[0]), vget(&pos, f[1]), vget(&pos, f[2])) {
            let q = Quadric::from_plane(plane);
            for &vi in f {
                quadrics[vi] = quadrics[vi].add(&q);
            }
        }
    }

    let mut collapses = 0usize;
    let mut active_faces = tris.iter().filter(|t| t.is_some()).count();

    // Greedy loop: repeatedly find and apply the cheapest valid edge collapse.
    while active_faces > target_faces {
        // Gather the current unique undirected edges.
        let mut edges: HashSet<(usize, usize)> = HashSet::new();
        for t in tris.iter().flatten() {
            for k in 0..3 {
                let a = t[k];
                let b = t[(k + 1) % 3];
                edges.insert(if a < b { (a, b) } else { (b, a) });
            }
        }
        if edges.is_empty() {
            break;
        }

        // Find the minimum-cost collapsible edge.
        let mut best: Option<(f64, usize, usize, [f64; 3])> = None;
        for &(i, j) in &edges {
            let qsum = quadrics[i].add(&quadrics[j]);
            let target = match qsum.optimal_point() {
                Some(p) => p,
                None => {
                    // Fall back to the better of midpoint / endpoints.
                    let pi = vget(&pos, i);
                    let pj = vget(&pos, j);
                    let mid = [
                        (pi[0] + pj[0]) * 0.5,
                        (pi[1] + pj[1]) * 0.5,
                        (pi[2] + pj[2]) * 0.5,
                    ];
                    let candidates = [pi, pj, mid];
                    let mut bp = mid;
                    let mut be = f64::INFINITY;
                    for c in candidates {
                        let e = qsum.error_at(c[0], c[1], c[2]);
                        if e < be {
                            be = e;
                            bp = c;
                        }
                    }
                    bp
                }
            };
            if would_flip(&tris, &pos, i, j, target) {
                continue;
            }
            let cost = qsum.error_at(target[0], target[1], target[2]).max(0.0);
            match &best {
                Some((bc, _, _, _)) if *bc <= cost => {}
                _ => best = Some((cost, i, j, target)),
            }
        }

        let (_, i, j, target) = match best {
            Some(b) => b,
            None => break, // no collapsible edge left
        };

        // Apply the collapse: move `i` to `target`, retarget `j → i`.
        pos[i * 3] = target[0];
        pos[i * 3 + 1] = target[1];
        pos[i * 3 + 2] = target[2];
        quadrics[i] = quadrics[i].add(&quadrics[j]);
        for t in tris.iter_mut() {
            if let Some(tri) = t {
                for vk in tri.iter_mut() {
                    if *vk == j {
                        *vk = i;
                    }
                }
                // Drop faces that became degenerate (a repeated vertex).
                if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
                    *t = None;
                }
            }
        }
        collapses += 1;
        active_faces = tris.iter().filter(|t| t.is_some()).count();
    }

    // Compact: keep only referenced vertices and remap indices.
    let mut remap = vec![usize::MAX; n_vertices];
    let mut out_v: Vec<f64> = Vec::new();
    let mut out_f: Vec<[usize; 3]> = Vec::new();
    for t in tris.iter().flatten() {
        let mut nf = [0usize; 3];
        for (slot, &vi) in nf.iter_mut().zip(t.iter()) {
            if remap[vi] == usize::MAX {
                remap[vi] = out_v.len() / 3;
                out_v.push(pos[vi * 3]);
                out_v.push(pos[vi * 3 + 1]);
                out_v.push(pos[vi * 3 + 2]);
            }
            *slot = remap[vi];
        }
        out_f.push(nf);
    }

    Ok(SimplifyResult {
        vertices: out_v,
        faces: out_f,
        collapses,
    })
}

/// Return `true` if moving the edge `(i, j)` to `target` would flip the normal of
/// any face incident to `i` or `j` (a common cause of self-intersection).
fn would_flip(
    tris: &[Option<[usize; 3]>],
    pos: &[f64],
    i: usize,
    j: usize,
    target: [f64; 3],
) -> bool {
    for tri in tris.iter().flatten() {
        if !(tri.contains(&i) || tri.contains(&j)) {
            continue;
        }
        // The face that contains the edge {i, j} disappears; skip it.
        if tri.contains(&i) && tri.contains(&j) {
            continue;
        }
        // Original normal.
        let before = face_normal(vget(pos, tri[0]), vget(pos, tri[1]), vget(pos, tri[2]));
        // Substitute i and j with the target position.
        let sub = |v: usize| -> [f64; 3] {
            if v == i || v == j {
                target
            } else {
                vget(pos, v)
            }
        };
        let after = face_normal(sub(tri[0]), sub(tri[1]), sub(tri[2]));
        let dot = before[0] * after[0] + before[1] * after[1] + before[2] * after[2];
        let len_after = (after[0] * after[0] + after[1] * after[1] + after[2] * after[2]).sqrt();
        if len_after < 1e-14 || dot < 0.0 {
            return true; // collapsed-to-line or flipped
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::curvature::icosphere;

    fn unit_cube() -> (Vec<f64>, Vec<[usize; 3]>) {
        let v = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
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

    #[test]
    fn quadric_plane_error_is_squared_distance() {
        // Plane z = 0 ⇒ (0,0,1,0). Error at (x,y,h) should be h².
        let q = Quadric::from_plane([0.0, 0.0, 1.0, 0.0]);
        assert!((q.error_at(3.0, -2.0, 2.0) - 4.0).abs() < 1e-12);
        assert!(q.error_at(5.0, 7.0, 0.0).abs() < 1e-12);
    }

    #[test]
    fn quadric_add_is_componentwise() {
        let a = Quadric::from_plane([1.0, 0.0, 0.0, 0.0]);
        let b = Quadric::from_plane([0.0, 1.0, 0.0, 0.0]);
        let c = a.add(&b);
        // Error at (2,3,0) = 2² + 3² = 13.
        assert!((c.error_at(2.0, 3.0, 0.0) - 13.0).abs() < 1e-12);
    }

    #[test]
    fn solve_3x3_identity() {
        let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = [3.0, -2.0, 5.0];
        let x = solve_3x3(&a, &b).expect("solve_3x3 should succeed");
        assert!((x[0] - 3.0).abs() < 1e-12);
        assert!((x[1] + 2.0).abs() < 1e-12);
        assert!((x[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn solve_3x3_singular_is_none() {
        let a = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 1.0, 1.0]; // rows 1,2 dependent
        assert!(solve_3x3(&a, &[1.0, 2.0, 3.0]).is_none());
    }

    #[test]
    fn face_plane_unit_normal() {
        let p = face_plane([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
            .expect("face_plane should succeed");
        let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-12);
        assert!(p[2].abs() > 0.999);
    }

    #[test]
    fn simplify_empty_errors() {
        assert!(simplify_mesh(&[], &[], 1).is_err());
    }

    #[test]
    fn simplify_bad_dim_errors() {
        assert!(simplify_mesh(&[0.0, 1.0], &[], 1).is_err());
    }

    #[test]
    fn simplify_bad_face_errors() {
        let (v, _) = unit_cube();
        let bad = vec![[0usize, 1, 999]];
        assert!(simplify_mesh(&v, &bad, 1).is_err());
    }

    #[test]
    fn simplify_reduces_face_count() {
        let (v, f) = icosphere(2); // 320 faces
        let orig = f.len();
        let res = simplify_mesh(&v, &f, orig / 2).expect("simplify_mesh should succeed");
        assert!(
            res.faces.len() <= orig / 2 + 2,
            "faces {} not reduced toward target {}",
            res.faces.len(),
            orig / 2
        );
        assert!(res.collapses > 0);
    }

    #[test]
    fn simplify_target_above_input_keeps_all() {
        let (v, f) = unit_cube();
        let res = simplify_mesh(&v, &f, 1000).expect("simplify_mesh should succeed");
        assert_eq!(res.faces.len(), f.len());
        assert_eq!(res.collapses, 0);
    }

    #[test]
    fn simplify_output_indices_valid() {
        let (v, f) = icosphere(2);
        let res = simplify_mesh(&v, &f, 100).expect("simplify_mesh should succeed");
        let nv = res.vertices.len() / 3;
        for face in &res.faces {
            for &idx in face {
                assert!(idx < nv, "index {idx} out of range {nv}");
            }
        }
    }

    #[test]
    fn simplify_no_degenerate_output_faces() {
        let (v, f) = icosphere(2);
        let res = simplify_mesh(&v, &f, 120).expect("simplify_mesh should succeed");
        for face in &res.faces {
            assert!(
                face[0] != face[1] && face[1] != face[2] && face[0] != face[2],
                "degenerate face {face:?}"
            );
        }
    }

    #[test]
    fn simplify_preserves_sphere_shape_roughly() {
        // After simplifying a unit icosphere, remaining vertices should still lie
        // near the unit sphere (QEM keeps them on the surface).
        let (v, f) = icosphere(2); // 320 faces
        let res = simplify_mesh(&v, &f, 120).expect("simplify_mesh should succeed");
        let nv = res.vertices.len() / 3;
        let mut max_dev = 0.0f64;
        for i in 0..nv {
            let p = vget(&res.vertices, i);
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            max_dev = max_dev.max((r - 1.0).abs());
        }
        assert!(max_dev < 0.25, "max radial deviation {max_dev} too large");
    }

    #[test]
    fn simplify_compacts_unused_vertices() {
        let (v, f) = icosphere(2);
        let res = simplify_mesh(&v, &f, 80).expect("simplify_mesh should succeed");
        // Every retained vertex must be referenced by at least one face.
        let nv = res.vertices.len() / 3;
        let mut used = vec![false; nv];
        for face in &res.faces {
            for &idx in face {
                used[idx] = true;
            }
        }
        assert!(used.iter().all(|&u| u), "compaction left orphan vertices");
    }

    #[test]
    fn simplify_keeps_some_geometry() {
        // Even an aggressive target leaves a non-empty valid mesh.
        let (v, f) = icosphere(2);
        let res = simplify_mesh(&v, &f, 4).expect("simplify_mesh should succeed");
        assert!(!res.faces.is_empty());
        assert!(!res.vertices.is_empty());
    }
}
