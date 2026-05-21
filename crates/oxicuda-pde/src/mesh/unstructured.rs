//! Generic unstructured mesh data structure.
//!
//! Supports the four common element families used by finite-element and
//! finite-volume codes:
//!
//! | Variant         | Vertices | Spatial dim |
//! |-----------------|---------:|------------:|
//! | `Triangle`      |        3 |           2 |
//! | `Quadrilateral` |        4 |           2 |
//! | `Tetrahedron`   |        4 |           3 |
//! | `Hexahedron`    |        8 |           3 |
//!
//! Nodes are always stored as `[x, y, z]` triples — 2D meshes simply use
//! `z = 0`.  Elements store their *vertex indices* (zero-based) into the global
//! node array together with an explicit [`ElementKind`].  The container
//! supports heterogeneous (mixed-element) meshes.
//!
//! The module additionally exposes the two combinatorial helpers most often
//! needed during PDE assembly:
//!
//! * [`UnstructuredMesh::build_vertex_to_elements`] — inverse connectivity
//!   (which elements contain a given vertex).
//! * [`UnstructuredMesh::boundary_edges_2d`] /
//!   [`UnstructuredMesh::boundary_faces_3d`] — boundary detection by counting
//!   how many elements each edge / face appears in (faces appearing exactly
//!   once are on the boundary).
//!
//! The implementation is deterministic, pure-`std`, and does not allocate
//! beyond what is strictly necessary.

use std::collections::HashMap;

use crate::error::{PdeError, PdeResult};

// ── Element kinds ─────────────────────────────────────────────────────────────

/// Supported element families for unstructured meshes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementKind {
    /// 3-node linear triangle (2D).
    Triangle,
    /// 4-node bilinear quadrilateral (2D).
    Quadrilateral,
    /// 4-node linear tetrahedron (3D).
    Tetrahedron,
    /// 8-node trilinear hexahedron (3D).
    Hexahedron,
}

impl ElementKind {
    /// Number of vertices that define an element of this kind.
    #[inline]
    pub fn n_vertices(self) -> usize {
        match self {
            ElementKind::Triangle => 3,
            ElementKind::Quadrilateral => 4,
            ElementKind::Tetrahedron => 4,
            ElementKind::Hexahedron => 8,
        }
    }

    /// Topological dimension of the element (2 for surface, 3 for volume).
    #[inline]
    pub fn dim(self) -> usize {
        match self {
            ElementKind::Triangle | ElementKind::Quadrilateral => 2,
            ElementKind::Tetrahedron | ElementKind::Hexahedron => 3,
        }
    }

    /// Number of boundary edges of a *2D* element.  Returns `0` for 3D kinds.
    #[inline]
    fn n_edges_2d(self) -> usize {
        match self {
            ElementKind::Triangle => 3,
            ElementKind::Quadrilateral => 4,
            _ => 0,
        }
    }
}

// ── The mesh container ────────────────────────────────────────────────────────

/// Generic unstructured mesh.
///
/// Nodes are 3D coordinates; elements reference nodes by index.  The mesh is
/// validated incrementally — every call to [`UnstructuredMesh::add_element`]
/// checks vertex-count and index bounds.
#[derive(Debug, Clone)]
pub struct UnstructuredMesh {
    /// Node coordinates `[x, y, z]`.  For 2D meshes `z` is conventionally `0`.
    pub nodes: Vec<[f64; 3]>,
    /// One vertex-index list per element.  The length always equals
    /// `element_kinds[e].n_vertices()`.
    pub elements: Vec<Vec<usize>>,
    /// Element-type tag for each entry in `elements`.
    pub element_kinds: Vec<ElementKind>,
}

impl UnstructuredMesh {
    /// Construct a mesh from a node list.  Elements are added later via
    /// [`UnstructuredMesh::add_element`].
    ///
    /// # Errors
    /// Returns `Err(EmptyMesh)` if `nodes` is empty.
    pub fn new(nodes: Vec<[f64; 3]>) -> PdeResult<Self> {
        if nodes.is_empty() {
            return Err(PdeError::EmptyMesh(
                "UnstructuredMesh::new requires at least one node".to_string(),
            ));
        }
        Ok(Self {
            nodes,
            elements: Vec::new(),
            element_kinds: Vec::new(),
        })
    }

    /// Append a new element of type `kind` with the given `vertices`.
    ///
    /// # Errors
    /// - `Err(InvalidParameter)` if `vertices.len() != kind.n_vertices()`.
    /// - `Err(IndexOutOfBounds)` if any vertex index is `>= n_nodes()`.
    pub fn add_element(&mut self, kind: ElementKind, vertices: Vec<usize>) -> PdeResult<()> {
        let expected = kind.n_vertices();
        if vertices.len() != expected {
            return Err(PdeError::InvalidParameter {
                name: "vertices".to_string(),
                reason: format!(
                    "element kind {:?} expects {} vertices, got {}",
                    kind,
                    expected,
                    vertices.len()
                ),
            });
        }
        let n_nodes = self.nodes.len();
        for &v in &vertices {
            if v >= n_nodes {
                return Err(PdeError::IndexOutOfBounds {
                    index: v,
                    len: n_nodes,
                });
            }
        }
        self.elements.push(vertices);
        self.element_kinds.push(kind);
        Ok(())
    }

    /// Total node count.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Total element count.
    #[inline]
    pub fn n_elements(&self) -> usize {
        self.elements.len()
    }

    /// Build the inverse connectivity table:
    /// `vertex_to_elements[v]` is the (sorted, deterministic) list of element
    /// indices that contain vertex `v`.
    pub fn build_vertex_to_elements(&self) -> Vec<Vec<usize>> {
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); self.n_nodes()];
        for (e_idx, elem) in self.elements.iter().enumerate() {
            for &v in elem {
                if v < adj.len() {
                    // Avoid duplicate insertion if the same vertex appears
                    // twice (degenerate input). last() catches the typical
                    // back-to-back case cheaply.
                    if adj[v].last().copied() != Some(e_idx) {
                        adj[v].push(e_idx);
                    }
                }
            }
        }
        adj
    }

    /// Return the boundary edges of every 2D element in the mesh.
    ///
    /// An edge is on the boundary if and only if it appears in exactly one
    /// element.  Edges are returned as `(min, max)` index pairs in the order
    /// they were first encountered; the result is therefore deterministic for
    /// a given mesh.
    ///
    /// 3D elements (if any) cause the function to return
    /// `Err(InvalidParameter)`.  Mixed *2D* meshes (Triangle + Quadrilateral)
    /// are handled correctly.
    ///
    /// # Errors
    /// Returns `Err(InvalidParameter)` if the mesh contains any 3D element.
    pub fn boundary_edges_2d(&self) -> PdeResult<Vec<(usize, usize)>> {
        for &k in &self.element_kinds {
            if k.dim() != 2 {
                return Err(PdeError::InvalidParameter {
                    name: "element_kinds".to_string(),
                    reason: format!("boundary_edges_2d requires 2D elements only; found {k:?}"),
                });
            }
        }

        // (min, max) → (count, insertion_index).  Using a HashMap keeps the
        // count step at amortized O(1) per edge.
        let mut edge_counts: HashMap<(usize, usize), usize> = HashMap::new();
        let mut order: Vec<(usize, usize)> = Vec::new();

        for (elem, kind) in self.elements.iter().zip(self.element_kinds.iter()) {
            let n_edges = kind.n_edges_2d();
            for i in 0..n_edges {
                let a = elem[i];
                let b = elem[(i + 1) % n_edges];
                if a == b {
                    return Err(PdeError::InvalidParameter {
                        name: "element".to_string(),
                        reason: "degenerate edge: repeated vertex".to_string(),
                    });
                }
                let key = if a < b { (a, b) } else { (b, a) };
                let entry = edge_counts.entry(key).or_insert_with(|| {
                    order.push(key);
                    0_usize
                });
                *entry += 1;
            }
        }

        let mut boundary: Vec<(usize, usize)> = Vec::new();
        for key in &order {
            if let Some(&count) = edge_counts.get(key) {
                if count == 1 {
                    boundary.push(*key);
                }
            }
        }
        Ok(boundary)
    }

    /// Return the boundary faces of every 3D element in the mesh.
    ///
    /// Each face is returned as a vertex-index list (length 3 for tet faces,
    /// length 4 for hex faces).  A face is on the boundary if and only if it
    /// is incident to exactly one element.  The match is independent of
    /// vertex order and orientation: a face is identified by its set of
    /// vertices (lexicographically sorted), but the returned listing uses
    /// the original CCW/CW order from the element where it was first seen.
    ///
    /// # Errors
    /// Returns `Err(InvalidParameter)` if the mesh contains any 2D element.
    pub fn boundary_faces_3d(&self) -> PdeResult<Vec<Vec<usize>>> {
        for &k in &self.element_kinds {
            if k.dim() != 3 {
                return Err(PdeError::InvalidParameter {
                    name: "element_kinds".to_string(),
                    reason: format!("boundary_faces_3d requires 3D elements only; found {k:?}"),
                });
            }
        }

        // Key = sorted vertex tuple, value = (count, first oriented face)
        let mut face_counts: HashMap<Vec<usize>, (usize, Vec<usize>)> = HashMap::new();
        let mut order: Vec<Vec<usize>> = Vec::new();

        for (elem, kind) in self.elements.iter().zip(self.element_kinds.iter()) {
            let faces = local_faces_3d(*kind, elem)?;
            for face in faces {
                let mut key = face.clone();
                key.sort_unstable();
                // Skip degenerate faces with a repeated vertex.
                if has_duplicate(&key) {
                    return Err(PdeError::InvalidParameter {
                        name: "element".to_string(),
                        reason: "degenerate face: repeated vertex".to_string(),
                    });
                }
                if let Some(entry) = face_counts.get_mut(&key) {
                    entry.0 += 1;
                } else {
                    face_counts.insert(key.clone(), (1, face));
                    order.push(key);
                }
            }
        }

        let mut boundary: Vec<Vec<usize>> = Vec::new();
        for key in &order {
            if let Some(&(count, ref face)) = face_counts.get(key) {
                if count == 1 {
                    boundary.push(face.clone());
                }
            }
        }
        Ok(boundary)
    }
}

// ── Local face tables ─────────────────────────────────────────────────────────

/// Return the local face vertex lists for a 3D element `kind`, expressed in
/// global vertex indices.  Face vertex order follows the standard outward-
/// pointing convention but the algorithm is orientation-independent.
fn local_faces_3d(kind: ElementKind, elem: &[usize]) -> PdeResult<Vec<Vec<usize>>> {
    match kind {
        ElementKind::Tetrahedron => {
            if elem.len() != 4 {
                return Err(PdeError::InvalidParameter {
                    name: "element".to_string(),
                    reason: format!("tetrahedron expects 4 vertices, got {}", elem.len()),
                });
            }
            // Faces opposite to vertex 0,1,2,3 respectively.
            Ok(vec![
                vec![elem[1], elem[2], elem[3]],
                vec![elem[0], elem[3], elem[2]],
                vec![elem[0], elem[1], elem[3]],
                vec![elem[0], elem[2], elem[1]],
            ])
        }
        ElementKind::Hexahedron => {
            if elem.len() != 8 {
                return Err(PdeError::InvalidParameter {
                    name: "element".to_string(),
                    reason: format!("hexahedron expects 8 vertices, got {}", elem.len()),
                });
            }
            // CGNS/VTK hex node ordering (bottom 0–3 CCW, top 4–7 CCW above).
            //   z+
            //   4───5
            //  /│  /│
            // 7─┼─6 │
            // │ 0─┼─1
            // │/  │/
            // 3───2 → x+
            Ok(vec![
                vec![elem[0], elem[3], elem[2], elem[1]], // bottom (-z)
                vec![elem[4], elem[5], elem[6], elem[7]], // top (+z)
                vec![elem[0], elem[1], elem[5], elem[4]], // front (-y)
                vec![elem[1], elem[2], elem[6], elem[5]], // right (+x)
                vec![elem[2], elem[3], elem[7], elem[6]], // back (+y)
                vec![elem[3], elem[0], elem[4], elem[7]], // left (-x)
            ])
        }
        _ => Err(PdeError::InvalidParameter {
            name: "kind".to_string(),
            reason: format!("local_faces_3d called for non-3D kind {kind:?}"),
        }),
    }
}

/// Return `true` if a sorted vertex slice contains any duplicate value.
fn has_duplicate(sorted_vertices: &[usize]) -> bool {
    for i in 1..sorted_vertices.len() {
        if sorted_vertices[i] == sorted_vertices[i - 1] {
            return true;
        }
    }
    false
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ElementKind metadata ──────────────────────────────────────────────

    #[test]
    fn element_kind_n_vertices() {
        assert_eq!(ElementKind::Triangle.n_vertices(), 3);
        assert_eq!(ElementKind::Quadrilateral.n_vertices(), 4);
        assert_eq!(ElementKind::Tetrahedron.n_vertices(), 4);
        assert_eq!(ElementKind::Hexahedron.n_vertices(), 8);
    }

    #[test]
    fn element_kind_dim() {
        assert_eq!(ElementKind::Triangle.dim(), 2);
        assert_eq!(ElementKind::Quadrilateral.dim(), 2);
        assert_eq!(ElementKind::Tetrahedron.dim(), 3);
        assert_eq!(ElementKind::Hexahedron.dim(), 3);
    }

    // ── Construction ──────────────────────────────────────────────────────

    #[test]
    fn new_empty_nodes_errors() {
        let res = UnstructuredMesh::new(Vec::new());
        assert!(res.is_err(), "empty node list should yield Err");
        match res {
            Err(PdeError::EmptyMesh(_)) => {}
            other => panic!("expected EmptyMesh, got {other:?}"),
        }
    }

    #[test]
    fn new_with_nodes_succeeds() {
        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let m = UnstructuredMesh::new(nodes).expect("non-empty nodes ok");
        assert_eq!(m.n_nodes(), 3);
        assert_eq!(m.n_elements(), 0);
    }

    // ── add_element validation ────────────────────────────────────────────

    #[test]
    fn add_element_out_of_range_errors() {
        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut m = UnstructuredMesh::new(nodes).expect("ok");
        let res = m.add_element(ElementKind::Triangle, vec![0, 1, 5]);
        assert!(res.is_err(), "out-of-range vertex should error");
        match res {
            Err(PdeError::IndexOutOfBounds { index, len }) => {
                assert_eq!(index, 5);
                assert_eq!(len, 3);
            }
            other => panic!("expected IndexOutOfBounds, got {other:?}"),
        }
    }

    #[test]
    fn add_element_wrong_count_errors() {
        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut m = UnstructuredMesh::new(nodes).expect("ok");
        // Triangle expects 3 vertices, give 2.
        let res = m.add_element(ElementKind::Triangle, vec![0, 1]);
        assert!(res.is_err(), "wrong vertex count should error");
        match res {
            Err(PdeError::InvalidParameter { name, .. }) => assert_eq!(name, "vertices"),
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    #[test]
    fn add_element_correct_count_succeeds() {
        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut m = UnstructuredMesh::new(nodes).expect("ok");
        m.add_element(ElementKind::Triangle, vec![0, 1, 2])
            .expect("ok");
        assert_eq!(m.n_elements(), 1);
        assert_eq!(m.element_kinds[0], ElementKind::Triangle);
    }

    // ── vertex_to_elements adjacency ──────────────────────────────────────

    #[test]
    fn vertex_to_elements_two_triangles_share_edge() {
        // Square split into two triangles sharing edge (1,2).
        //   3───2
        //   │ ╱ │
        //   0───1
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let mut m = UnstructuredMesh::new(nodes).expect("ok");
        m.add_element(ElementKind::Triangle, vec![0, 1, 2])
            .expect("ok");
        m.add_element(ElementKind::Triangle, vec![0, 2, 3])
            .expect("ok");
        let adj = m.build_vertex_to_elements();
        assert_eq!(adj.len(), 4);
        // Vertex 0 belongs to both elements.
        assert_eq!(adj[0], vec![0, 1]);
        // Vertex 2 (shared) belongs to both elements.
        assert_eq!(adj[2], vec![0, 1]);
        // Vertex 1 only in element 0.
        assert_eq!(adj[1], vec![0]);
        // Vertex 3 only in element 1.
        assert_eq!(adj[3], vec![1]);
    }

    #[test]
    fn n_nodes_n_elements_accurate() {
        let nodes: Vec<[f64; 3]> = (0..5).map(|i| [i as f64, 0.0, 0.0]).collect();
        let mut m = UnstructuredMesh::new(nodes).expect("ok");
        assert_eq!(m.n_nodes(), 5);
        assert_eq!(m.n_elements(), 0);
        m.add_element(ElementKind::Triangle, vec![0, 1, 2])
            .expect("ok");
        m.add_element(ElementKind::Triangle, vec![2, 3, 4])
            .expect("ok");
        assert_eq!(m.n_nodes(), 5);
        assert_eq!(m.n_elements(), 2);
    }

    // ── 2D boundary edges ─────────────────────────────────────────────────

    #[test]
    fn boundary_edges_single_triangle() {
        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut m = UnstructuredMesh::new(nodes).expect("ok");
        m.add_element(ElementKind::Triangle, vec![0, 1, 2])
            .expect("ok");
        let edges = m.boundary_edges_2d().expect("ok");
        assert_eq!(edges.len(), 3, "single triangle has 3 boundary edges");
        // The edge set must be exactly {(0,1), (1,2), (0,2)}.
        let mut sorted = edges.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn boundary_edges_two_triangles_share_edge() {
        // Two triangles sharing edge (0,2); interior edge should be filtered.
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let mut m = UnstructuredMesh::new(nodes).expect("ok");
        m.add_element(ElementKind::Triangle, vec![0, 1, 2])
            .expect("ok");
        m.add_element(ElementKind::Triangle, vec![0, 2, 3])
            .expect("ok");
        let edges = m.boundary_edges_2d().expect("ok");
        assert_eq!(
            edges.len(),
            4,
            "two triangles share one interior edge → 4 boundary"
        );
        let mut sorted = edges.clone();
        sorted.sort_unstable();
        // Boundary edges: (0,1), (1,2), (2,3), (0,3).
        assert_eq!(sorted, vec![(0, 1), (0, 3), (1, 2), (2, 3)]);
    }

    #[test]
    fn boundary_edges_mixed_2d_mesh() {
        // One triangle + one quad sharing an edge.
        //  3───2───4
        //  │   │ ╲ │
        //  0───1   5
        let nodes = vec![
            [0.0, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0], // 1
            [1.0, 1.0, 0.0], // 2
            [0.0, 1.0, 0.0], // 3
            [2.0, 1.0, 0.0], // 4
            [2.0, 0.0, 0.0], // 5
        ];
        let mut m = UnstructuredMesh::new(nodes).expect("ok");
        // Quad (0,1,2,3) – left square
        m.add_element(ElementKind::Quadrilateral, vec![0, 1, 2, 3])
            .expect("ok");
        // Triangle (1,5,4) – right triangle
        m.add_element(ElementKind::Triangle, vec![1, 5, 4])
            .expect("ok");
        // Triangle (1,4,2) – glues triangle's hypotenuse to the quad's right edge (1,2).
        m.add_element(ElementKind::Triangle, vec![1, 4, 2])
            .expect("ok");
        let edges = m.boundary_edges_2d().expect("ok");
        // Boundary: (0,1), (0,3), (2,3), (1,5), (4,5), (2,4)
        let mut sorted = edges.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![(0, 1), (0, 3), (1, 5), (2, 3), (2, 4), (4, 5)]);
    }

    #[test]
    fn boundary_edges_errors_on_3d_element() {
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let mut m = UnstructuredMesh::new(nodes).expect("ok");
        m.add_element(ElementKind::Tetrahedron, vec![0, 1, 2, 3])
            .expect("ok");
        let res = m.boundary_edges_2d();
        assert!(res.is_err(), "3D element should make boundary_edges_2d err");
    }

    #[test]
    fn boundary_edges_empty_mesh_is_empty() {
        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let m = UnstructuredMesh::new(nodes).expect("ok");
        let edges = m.boundary_edges_2d().expect("ok");
        assert!(edges.is_empty(), "no elements → no edges");
    }

    // ── 3D boundary faces ─────────────────────────────────────────────────

    #[test]
    fn boundary_faces_single_tet() {
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let mut m = UnstructuredMesh::new(nodes).expect("ok");
        m.add_element(ElementKind::Tetrahedron, vec![0, 1, 2, 3])
            .expect("ok");
        let faces = m.boundary_faces_3d().expect("ok");
        assert_eq!(faces.len(), 4, "single tet has 4 boundary faces");
        // Sanity: every face has 3 vertices.
        for f in &faces {
            assert_eq!(f.len(), 3);
        }
    }

    #[test]
    fn boundary_faces_two_tets_share_face() {
        // Two tets sharing the face (1,2,3).
        //   2
        //   │╲╲
        //   │ ╲╲
        //   │  3
        //   │ ╱╲
        //   │╱  ╲
        //   0    4
        //    ╲   ╱
        //     1
        let nodes = vec![
            [0.0, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0], // 1
            [0.0, 1.0, 0.0], // 2
            [0.0, 0.0, 1.0], // 3
            [1.0, 1.0, 1.0], // 4
        ];
        let mut m = UnstructuredMesh::new(nodes).expect("ok");
        m.add_element(ElementKind::Tetrahedron, vec![0, 1, 2, 3])
            .expect("ok");
        m.add_element(ElementKind::Tetrahedron, vec![4, 1, 2, 3])
            .expect("ok");
        let faces = m.boundary_faces_3d().expect("ok");
        // 4 + 4 = 8 total, 2 interior (shared face counted twice) → 6 boundary.
        assert_eq!(faces.len(), 6, "two tets share 1 face → 6 boundary");
    }

    #[test]
    fn boundary_faces_single_hex() {
        // Unit cube hexahedron with the canonical CGNS/VTK node ordering.
        let nodes = vec![
            [0.0, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0], // 1
            [1.0, 1.0, 0.0], // 2
            [0.0, 1.0, 0.0], // 3
            [0.0, 0.0, 1.0], // 4
            [1.0, 0.0, 1.0], // 5
            [1.0, 1.0, 1.0], // 6
            [0.0, 1.0, 1.0], // 7
        ];
        let mut m = UnstructuredMesh::new(nodes).expect("ok");
        m.add_element(ElementKind::Hexahedron, vec![0, 1, 2, 3, 4, 5, 6, 7])
            .expect("ok");
        let faces = m.boundary_faces_3d().expect("ok");
        assert_eq!(faces.len(), 6, "single hex has 6 boundary faces");
        for f in &faces {
            assert_eq!(f.len(), 4);
        }
    }

    #[test]
    fn boundary_faces_errors_on_2d_element() {
        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut m = UnstructuredMesh::new(nodes).expect("ok");
        m.add_element(ElementKind::Triangle, vec![0, 1, 2])
            .expect("ok");
        let res = m.boundary_faces_3d();
        assert!(res.is_err(), "2D element must make boundary_faces_3d err");
    }

    #[test]
    fn boundary_faces_empty_mesh_is_empty() {
        let nodes = vec![[0.0, 0.0, 0.0]];
        let m = UnstructuredMesh::new(nodes).expect("ok");
        let faces = m.boundary_faces_3d().expect("ok");
        assert!(faces.is_empty(), "no elements → no faces");
    }

    // ── Determinism ───────────────────────────────────────────────────────

    #[test]
    fn boundary_edges_deterministic_across_runs() {
        // Build the same mesh twice and compare outputs.
        let build = || {
            let nodes = vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ];
            let mut m = UnstructuredMesh::new(nodes).expect("ok");
            m.add_element(ElementKind::Triangle, vec![0, 1, 2])
                .expect("ok");
            m.add_element(ElementKind::Triangle, vec![0, 2, 3])
                .expect("ok");
            m
        };
        let m1 = build();
        let m2 = build();
        let e1 = m1.boundary_edges_2d().expect("ok");
        let e2 = m2.boundary_edges_2d().expect("ok");
        assert_eq!(e1, e2, "boundary edges must be deterministic");
    }

    #[test]
    fn boundary_faces_deterministic_across_runs() {
        let build = || {
            let nodes = vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ];
            let mut m = UnstructuredMesh::new(nodes).expect("ok");
            m.add_element(ElementKind::Tetrahedron, vec![0, 1, 2, 3])
                .expect("ok");
            m
        };
        let f1 = build().boundary_faces_3d().expect("ok");
        let f2 = build().boundary_faces_3d().expect("ok");
        assert_eq!(f1, f2, "boundary faces must be deterministic");
    }

    // ── Edge / face counts vs Euler ───────────────────────────────────────

    #[test]
    fn quad_alone_has_four_boundary_edges() {
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let mut m = UnstructuredMesh::new(nodes).expect("ok");
        m.add_element(ElementKind::Quadrilateral, vec![0, 1, 2, 3])
            .expect("ok");
        let edges = m.boundary_edges_2d().expect("ok");
        assert_eq!(edges.len(), 4);
    }
}
