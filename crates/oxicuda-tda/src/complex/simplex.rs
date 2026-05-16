//! Simplex type: a sorted list of vertex indices representing a k-simplex.

use crate::error::{TdaError, TdaResult};

/// A k-simplex: a sorted `Vec` of vertex indices (length k+1).
///
/// Invariants enforced by `new()`:
/// - Non-empty (at least one vertex).
/// - Vertices are sorted in ascending order.
/// - No duplicate vertices.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Simplex {
    pub vertices: Vec<usize>,
}

impl Simplex {
    /// Create a new simplex from the given vertex list.
    ///
    /// Sorts and deduplicates the vertices.  Returns an error if the input is
    /// empty or contains duplicates after sorting.
    pub fn new(mut vertices: Vec<usize>) -> TdaResult<Self> {
        if vertices.is_empty() {
            return Err(TdaError::InvalidSimplex("empty vertex list".to_owned()));
        }
        vertices.sort_unstable();
        // Check for duplicates
        for i in 1..vertices.len() {
            if vertices[i] == vertices[i - 1] {
                return Err(TdaError::InvalidSimplex(format!(
                    "duplicate vertex {}",
                    vertices[i]
                )));
            }
        }
        Ok(Self { vertices })
    }

    /// The dimension of the simplex: k for a (k+1)-vertex simplex.
    pub fn dim(&self) -> usize {
        self.vertices.len() - 1
    }

    /// All (dim-1)-dimensional faces of this simplex.
    ///
    /// A face is obtained by removing exactly one vertex.
    pub fn faces(&self) -> Vec<Simplex> {
        if self.vertices.len() <= 1 {
            return vec![];
        }
        (0..self.vertices.len())
            .map(|i| {
                let verts: Vec<usize> = self
                    .vertices
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, &v)| v)
                    .collect();
                // SAFETY: verts is non-empty (we only call this when len > 1)
                // and sorted (sub-sequence of sorted sequence).
                Simplex { vertices: verts }
            })
            .collect()
    }

    /// The boundary operator ∂σ = Σ_i (-1)^i · (σ without vertex i).
    ///
    /// Returns a list of `(coefficient, face)` pairs with coefficient ±1.
    pub fn boundary(&self) -> Vec<(i8, Simplex)> {
        if self.vertices.len() <= 1 {
            return vec![];
        }
        (0..self.vertices.len())
            .map(|i| {
                let sign: i8 = if i % 2 == 0 { 1 } else { -1 };
                let verts: Vec<usize> = self
                    .vertices
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, &v)| v)
                    .collect();
                let face = Simplex { vertices: verts };
                (sign, face)
            })
            .collect()
    }

    /// Check whether `other` is a face of `self` (i.e., `other.vertices ⊂ self.vertices`).
    pub fn contains_face(&self, other: &Simplex) -> bool {
        if other.vertices.len() >= self.vertices.len() {
            return false;
        }
        // Both vertex lists are sorted; use a two-pointer sub-set check.
        let mut si = 0usize;
        let mut oi = 0usize;
        while oi < other.vertices.len() && si < self.vertices.len() {
            if self.vertices[si] == other.vertices[oi] {
                si += 1;
                oi += 1;
            } else if self.vertices[si] < other.vertices[oi] {
                si += 1;
            } else {
                return false;
            }
        }
        oi == other.vertices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_sorts_and_validates() {
        let s = Simplex::new(vec![3, 1, 2]).expect("valid simplex");
        assert_eq!(s.vertices, vec![1, 2, 3]);
        assert_eq!(s.dim(), 2);
    }

    #[test]
    fn new_rejects_duplicates() {
        assert!(Simplex::new(vec![1, 1, 2]).is_err());
    }

    #[test]
    fn new_rejects_empty() {
        assert!(Simplex::new(vec![]).is_err());
    }

    #[test]
    fn faces_of_edge() {
        let e = Simplex::new(vec![0, 1]).expect("edge");
        let fs = e.faces();
        assert_eq!(fs.len(), 2);
    }

    #[test]
    fn faces_of_vertex_empty() {
        let v = Simplex::new(vec![0]).expect("vertex");
        assert!(v.faces().is_empty());
    }

    #[test]
    fn boundary_squared_zero() {
        // ∂²σ = 0: boundary of all faces should cancel in pairs.
        let t = Simplex::new(vec![0, 1, 2, 3]).expect("3-simplex");
        let bd1 = t.boundary(); // 4 faces of dim 2
        // Collect all (coeff, face_of_face) terms.
        let mut terms: Vec<(i8, Simplex)> = Vec::new();
        for (c1, face) in &bd1 {
            for (c2, ff) in face.boundary() {
                terms.push((c1 * c2, ff));
            }
        }
        // Group by face, sum coefficients; all should be 0.
        use std::collections::HashMap;
        let mut sums: HashMap<Simplex, i32> = HashMap::new();
        for (c, f) in terms {
            *sums.entry(f).or_insert(0) += c as i32;
        }
        for v in sums.values() {
            assert_eq!(*v, 0, "∂² ≠ 0");
        }
    }

    #[test]
    fn contains_face_checks() {
        let t = Simplex::new(vec![0, 1, 2]).expect("triangle");
        let e = Simplex::new(vec![0, 1]).expect("edge");
        let v = Simplex::new(vec![0]).expect("vertex");
        assert!(t.contains_face(&e));
        assert!(t.contains_face(&v));
        assert!(!e.contains_face(&t));
        assert!(!t.contains_face(&t)); // not a proper face
    }
}
