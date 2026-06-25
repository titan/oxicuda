//! Triangle-mesh topology validation.
//!
//! Computes the combinatorial invariants of a triangle mesh and flags the
//! conditions that break manifoldness:
//!
//! * **degenerate faces** — a triangle that repeats a vertex (zero area in the
//!   combinatorial sense);
//! * **out-of-range indices** — a face referencing a vertex `>= n`;
//! * **non-manifold edges** — an undirected edge incident to more than two
//!   faces;
//! * **boundary edges** — an undirected edge incident to exactly one face (the
//!   mesh has a border there);
//! * **orientation consistency** — for a consistently oriented closed surface
//!   every interior edge is traversed once in each direction; a directed edge
//!   seen twice in the *same* direction indicates a flipped neighbour.
//!
//! It also reports the Euler characteristic `χ = V − E + F` (using *used*
//! vertices and *undirected* edges), from which the genus of a closed
//! orientable manifold follows as `g = (2 − χ) / 2`.
//!
//! [`validate_mesh`] returns `Ok(report)` for a clean closed orientable
//! manifold and an [`crate::error::Geom3dError::InvalidTopology`] otherwise;
//! [`analyze_topology`] always returns the full [`TopologyReport`] without
//! failing, for callers that want to inspect borders.

use crate::error::{Geom3dError, Geom3dResult};
use std::collections::HashMap;

/// Full topology analysis of a triangle mesh.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyReport {
    /// Number of vertices actually referenced by at least one face.
    pub used_vertices: usize,
    /// Number of distinct undirected edges.
    pub edges: usize,
    /// Number of (non-degenerate) faces.
    pub faces: usize,
    /// Euler characteristic `V − E + F` over used vertices / undirected edges.
    pub euler_characteristic: i64,
    /// Count of faces that repeat a vertex.
    pub degenerate_faces: usize,
    /// Count of faces with an out-of-range vertex index.
    pub out_of_range_faces: usize,
    /// Undirected edges incident to exactly one face.
    pub boundary_edges: usize,
    /// Undirected edges incident to three or more faces.
    pub non_manifold_edges: usize,
    /// `true` if every interior edge is traversed once per direction.
    pub orientable: bool,
    /// `true` if the mesh is a clean closed orientable 2-manifold.
    pub is_closed_manifold: bool,
}

/// Canonical undirected edge key (sorted endpoints).
fn undirected(a: usize, b: usize) -> (usize, usize) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Analyze the topology of a triangle mesh.
///
/// `vertices` is the flat `[n × 3]` coordinate buffer (used only for the vertex
/// count `n`); `triangles` is the list of vertex-index triples.
///
/// This function never errors — inspect [`TopologyReport::is_closed_manifold`]
/// (and the individual fields) to decide validity.
#[must_use]
pub fn analyze_topology(n_vertices: usize, triangles: &[[usize; 3]]) -> TopologyReport {
    // Directed edge multiplicity (for orientation) and undirected face count.
    let mut directed: HashMap<(usize, usize), usize> = HashMap::new();
    let mut undirected_faces: HashMap<(usize, usize), usize> = HashMap::new();
    let mut used = vec![false; n_vertices];

    let mut degenerate = 0usize;
    let mut out_of_range = 0usize;
    let mut good_faces = 0usize;

    for tri in triangles {
        let [a, b, c] = *tri;
        if a >= n_vertices || b >= n_vertices || c >= n_vertices {
            out_of_range += 1;
            continue;
        }
        if a == b || b == c || a == c {
            degenerate += 1;
            continue;
        }
        used[a] = true;
        used[b] = true;
        used[c] = true;
        good_faces += 1;

        for &(u, v) in &[(a, b), (b, c), (c, a)] {
            *directed.entry((u, v)).or_insert(0) += 1;
            *undirected_faces.entry(undirected(u, v)).or_insert(0) += 1;
        }
    }

    let used_vertices = used.iter().filter(|&&u| u).count();
    let edges = undirected_faces.len();

    let mut boundary = 0usize;
    let mut non_manifold = 0usize;
    for &count in undirected_faces.values() {
        if count == 1 {
            boundary += 1;
        } else if count > 2 {
            non_manifold += 1;
        }
    }

    // Orientability: a directed edge seen more than once means two faces share
    // it in the *same* direction → inconsistent winding.
    let orientable = directed.values().all(|&c| c <= 1);

    let euler = used_vertices as i64 - edges as i64 + good_faces as i64;
    let is_closed_manifold = out_of_range == 0
        && degenerate == 0
        && boundary == 0
        && non_manifold == 0
        && orientable
        && good_faces > 0;

    TopologyReport {
        used_vertices,
        edges,
        faces: good_faces,
        euler_characteristic: euler,
        degenerate_faces: degenerate,
        out_of_range_faces: out_of_range,
        boundary_edges: boundary,
        non_manifold_edges: non_manifold,
        orientable,
        is_closed_manifold,
    }
}

/// Validate that a triangle mesh is a clean closed orientable 2-manifold.
///
/// Returns the [`TopologyReport`] on success.
///
/// # Errors
///
/// Returns [`Geom3dError::InvalidTopology`] with a reason describing the first
/// violated condition (empty/degenerate faces, out-of-range indices,
/// non-manifold edges, open boundary, or inconsistent orientation).
pub fn validate_mesh(n_vertices: usize, triangles: &[[usize; 3]]) -> Geom3dResult<TopologyReport> {
    let report = analyze_topology(n_vertices, triangles);
    if report.out_of_range_faces > 0 {
        return Err(Geom3dError::InvalidTopology {
            reason: "face references an out-of-range vertex index",
        });
    }
    if report.faces == 0 {
        return Err(Geom3dError::InvalidTopology {
            reason: "mesh has no valid (non-degenerate) faces",
        });
    }
    if report.degenerate_faces > 0 {
        return Err(Geom3dError::InvalidTopology {
            reason: "mesh contains degenerate faces (repeated vertex)",
        });
    }
    if report.non_manifold_edges > 0 {
        return Err(Geom3dError::InvalidTopology {
            reason: "mesh has non-manifold edges (edge shared by >2 faces)",
        });
    }
    if report.boundary_edges > 0 {
        return Err(Geom3dError::InvalidTopology {
            reason: "mesh is not closed (has boundary edges)",
        });
    }
    if !report.orientable {
        return Err(Geom3dError::InvalidTopology {
            reason: "mesh is not consistently oriented",
        });
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::curvature::icosphere;

    /// A consistently wound unit tetrahedron (closed, orientable, χ = 2).
    fn tetrahedron() -> (usize, Vec<[usize; 3]>) {
        // Outward-facing winding for the 4 triangles of a tetra.
        let tris = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        (4, tris)
    }

    #[test]
    fn tetrahedron_is_closed_manifold() {
        let (n, tris) = tetrahedron();
        let report = validate_mesh(n, &tris).expect("tetra must validate");
        assert_eq!(report.faces, 4);
        assert_eq!(report.edges, 6);
        assert_eq!(report.used_vertices, 4);
        assert_eq!(report.euler_characteristic, 2);
        assert_eq!(report.boundary_edges, 0);
        assert_eq!(report.non_manifold_edges, 0);
        assert!(report.orientable);
        assert!(report.is_closed_manifold);
    }

    #[test]
    fn icosphere_euler_is_two() {
        // Any subdivided icosphere is a closed orientable sphere: χ = 2.
        let (verts, tris) = icosphere(2);
        let n = verts.len() / 3;
        let report = validate_mesh(n, &tris).expect("icosphere must validate");
        assert_eq!(report.euler_characteristic, 2);
        assert!(report.is_closed_manifold);
        // Genus 0: (2 − χ)/2 == 0.
        assert_eq!((2 - report.euler_characteristic) / 2, 0);
    }

    #[test]
    fn single_triangle_has_boundary() {
        let tris = vec![[0, 1, 2]];
        let report = analyze_topology(3, &tris);
        assert_eq!(report.faces, 1);
        assert_eq!(report.edges, 3);
        assert_eq!(report.boundary_edges, 3, "all three edges are boundary");
        assert!(!report.is_closed_manifold);
        assert!(validate_mesh(3, &tris).is_err());
    }

    #[test]
    fn open_tetra_detects_boundary() {
        // Drop one face → an open surface with a triangular hole.
        let tris = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2]];
        let report = analyze_topology(4, &tris);
        assert_eq!(report.faces, 3);
        assert_eq!(report.boundary_edges, 3);
        assert!(!report.is_closed_manifold);
    }

    #[test]
    fn non_manifold_edge_detected() {
        // Three triangles sharing edge (0,1): non-manifold.
        let tris = vec![[0, 1, 2], [0, 1, 3], [0, 1, 4]];
        let report = analyze_topology(5, &tris);
        assert!(
            report.non_manifold_edges >= 1,
            "edge (0,1) shared by 3 faces"
        );
        assert!(validate_mesh(5, &tris).is_err());
    }

    #[test]
    fn inconsistent_orientation_detected() {
        // Two triangles sharing edge (0,1) wound the SAME way → both emit the
        // directed edge (0,1), breaking orientation.
        let tris = vec![[0, 1, 2], [0, 1, 3]];
        let report = analyze_topology(4, &tris);
        assert!(!report.orientable, "shared edge in same direction");
    }

    #[test]
    fn degenerate_face_detected() {
        let tris = vec![[0, 1, 1], [0, 1, 2]];
        let report = analyze_topology(3, &tris);
        assert_eq!(report.degenerate_faces, 1);
        assert_eq!(report.faces, 1, "only the non-degenerate face counts");
        assert!(validate_mesh(3, &tris).is_err());
    }

    #[test]
    fn out_of_range_index_detected() {
        let tris = vec![[0, 1, 9]]; // vertex 9 does not exist for n=3
        let report = analyze_topology(3, &tris);
        assert_eq!(report.out_of_range_faces, 1);
        assert!(validate_mesh(3, &tris).is_err());
    }

    #[test]
    fn empty_mesh_errors() {
        assert!(validate_mesh(0, &[]).is_err());
        let report = analyze_topology(0, &[]);
        assert_eq!(report.faces, 0);
        assert!(!report.is_closed_manifold);
    }
}
