//! `SimplicialComplex`: a collection of simplices closed under the face relation.

use crate::complex::simplex::Simplex;
use crate::error::{TdaError, TdaResult};

/// A simplicial complex: a finite collection of simplices satisfying the closure property
/// (every face of every simplex is also in the complex).
///
/// Simplices are stored in a `Vec` sorted first by dimension, then lexicographically.
#[derive(Debug, Clone, Default)]
pub struct SimplicialComplex {
    simplices: Vec<Simplex>,
    max_dim: usize,
}

impl SimplicialComplex {
    /// Create an empty simplicial complex.
    pub fn new() -> Self {
        Self {
            simplices: Vec::new(),
            max_dim: 0,
        }
    }

    /// Add a simplex to the complex together with **all** of its faces (closure property).
    ///
    /// If any face or the simplex itself already exists it is silently skipped.
    pub fn add_simplex_with_closure(&mut self, s: Simplex) -> TdaResult<()> {
        // Recursively collect all faces down to vertices.
        let mut to_add: Vec<Simplex> = Vec::new();
        self.collect_closure(&s, &mut to_add);
        to_add.push(s);
        for simplex in to_add {
            if !self.simplices.contains(&simplex) {
                let dim = simplex.dim();
                if dim > self.max_dim {
                    self.max_dim = dim;
                }
                self.simplices.push(simplex);
            }
        }
        self.simplices.sort();
        Ok(())
    }

    fn collect_closure(&self, s: &Simplex, out: &mut Vec<Simplex>) {
        for face in s.faces() {
            if !out.contains(&face) && !self.simplices.contains(&face) {
                self.collect_closure(&face, out);
                out.push(face);
            }
        }
    }

    /// Add a simplex that is assumed to be already closed (all faces must exist).
    ///
    /// Returns `ClosureViolation` if any face is missing from the complex.
    pub fn add_simplex(&mut self, s: Simplex) -> TdaResult<()> {
        // Verify all faces are present.
        for face in s.faces() {
            if !self.simplices.contains(&face) {
                return Err(TdaError::ClosureViolation(format!(
                    "face {:?} of {:?} not in complex",
                    face.vertices, s.vertices
                )));
            }
        }
        if !self.simplices.contains(&s) {
            let dim = s.dim();
            if dim > self.max_dim {
                self.max_dim = dim;
            }
            self.simplices.push(s);
            self.simplices.sort();
        }
        Ok(())
    }

    /// All simplices of a given dimension.
    pub fn simplices_of_dim(&self, dim: usize) -> Vec<&Simplex> {
        self.simplices.iter().filter(|s| s.dim() == dim).collect()
    }

    /// Total number of simplices.
    pub fn n_simplices(&self) -> usize {
        self.simplices.len()
    }

    /// Maximum dimension of any simplex in the complex.
    pub fn max_dim(&self) -> usize {
        self.max_dim
    }

    /// Check whether the complex contains the given simplex.
    pub fn contains(&self, s: &Simplex) -> bool {
        self.simplices.contains(s)
    }

    /// Verify the closure property: every face of every simplex is in the complex.
    pub fn verify_closure(&self) -> TdaResult<()> {
        for s in &self.simplices {
            for face in s.faces() {
                if !self.simplices.contains(&face) {
                    return Err(TdaError::ClosureViolation(format!(
                        "face {:?} of {:?} missing",
                        face.vertices, s.vertices
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_with_closure_adds_faces() {
        let mut cx = SimplicialComplex::new();
        let tri = Simplex::new(vec![0, 1, 2]).expect("ok");
        cx.add_simplex_with_closure(tri).expect("ok");
        assert!(cx.contains(&Simplex::new(vec![0]).expect("ok")));
        assert!(cx.contains(&Simplex::new(vec![0, 1]).expect("ok")));
        assert!(cx.contains(&Simplex::new(vec![0, 1, 2]).expect("ok")));
        cx.verify_closure().expect("closed");
    }

    #[test]
    fn add_simplex_rejects_missing_face() {
        let mut cx = SimplicialComplex::new();
        // Add only the vertex 0
        cx.add_simplex_with_closure(Simplex::new(vec![0]).expect("ok"))
            .expect("ok");
        // Try to add edge [0,1] without vertex 1
        let result = cx.add_simplex(Simplex::new(vec![0, 1]).expect("ok"));
        assert!(result.is_err());
    }
}
