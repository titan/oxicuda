//! Sparse boundary matrix over Z₂ for persistent homology computation.

use crate::complex::filtration::Filtration;
use crate::error::{TdaError, TdaResult};

/// Sparse boundary matrix over Z₂.
///
/// Columns are indexed by simplex in filtration order.
/// Rows are also indexed by simplex in filtration order.
/// `columns[j]` = sorted list of row indices r such that M[r, j] = 1 (mod 2),
/// i.e. simplex r is a face of simplex j.
#[derive(Debug, Clone)]
pub struct BoundaryMatrix {
    pub n_rows: usize,
    pub n_cols: usize,
    /// Each column: sorted list of nonzero row indices (over Z₂).
    pub columns: Vec<Vec<usize>>,
}

impl BoundaryMatrix {
    /// Build the boundary matrix from a `Filtration`.
    ///
    /// Both rows and columns correspond to simplices in filtration order.
    /// Column j has nonzero rows exactly at the indices of the (dim-1)-faces of simplex j
    /// (looked up in the filtration index).
    pub fn from_filtration(filtration: &Filtration) -> TdaResult<Self> {
        let n = filtration.simplices.len();
        if n == 0 {
            return Err(TdaError::EmptyComplex);
        }

        // Build reverse map: simplex vertices → filtration index.
        // Use a sorted Vec for a deterministic, no-HashMap implementation.
        // (For small complexes this is fine; for large ones a HashMap would be faster.)
        let mut index_map: Vec<(Vec<usize>, usize)> = filtration
            .simplices
            .iter()
            .enumerate()
            .map(|(i, fs)| (fs.simplex.vertices.clone(), i))
            .collect();
        index_map.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let lookup = |verts: &[usize]| -> Option<usize> {
            index_map
                .binary_search_by(|(v, _)| v.as_slice().cmp(verts))
                .ok()
                .map(|pos| index_map[pos].1)
        };

        let mut columns: Vec<Vec<usize>> = Vec::with_capacity(n);
        for fs in &filtration.simplices {
            let faces = fs.simplex.faces();
            let mut col: Vec<usize> = Vec::with_capacity(faces.len());
            for face in &faces {
                match lookup(&face.vertices) {
                    Some(row_idx) => col.push(row_idx),
                    None => {
                        return Err(TdaError::ClosureViolation(format!(
                            "face {:?} of {:?} not in filtration",
                            face.vertices, fs.simplex.vertices
                        )));
                    }
                }
            }
            col.sort_unstable();
            columns.push(col);
        }

        Ok(Self {
            n_rows: n,
            n_cols: n,
            columns,
        })
    }

    /// Return the index of the lowest (maximum) nonzero row in column `col`.
    pub fn low(&self, col: usize) -> Option<usize> {
        self.columns[col].last().copied()
    }

    /// Add column `source` to column `target` over Z₂ (symmetric difference of row sets).
    pub fn add_cols(&mut self, target: usize, source: usize) {
        // Perform XOR (symmetric difference) of two sorted lists.
        let src = self.columns[source].clone();
        let tgt = std::mem::take(&mut self.columns[target]);
        let mut result: Vec<usize> = Vec::with_capacity(tgt.len() + src.len());
        let mut ti = 0usize;
        let mut si = 0usize;
        while ti < tgt.len() && si < src.len() {
            match tgt[ti].cmp(&src[si]) {
                std::cmp::Ordering::Less => {
                    result.push(tgt[ti]);
                    ti += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.push(src[si]);
                    si += 1;
                }
                std::cmp::Ordering::Equal => {
                    // Both have this row: XOR cancels it (coefficient becomes 0 mod 2)
                    ti += 1;
                    si += 1;
                }
            }
        }
        while ti < tgt.len() {
            result.push(tgt[ti]);
            ti += 1;
        }
        while si < src.len() {
            result.push(src[si]);
            si += 1;
        }
        self.columns[target] = result;
    }

    /// Return `true` if column `col` is the zero column.
    pub fn is_zero(&self, col: usize) -> bool {
        self.columns[col].is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::filtration::{FilteredSimplex, Filtration};
    use crate::complex::simplex::Simplex;

    fn make_triangle_filtration() -> Filtration {
        // 3 vertices + 3 edges + 1 triangle, all at different values
        let simplices = vec![
            FilteredSimplex {
                simplex: Simplex { vertices: vec![0] },
                value: 0.0,
            },
            FilteredSimplex {
                simplex: Simplex { vertices: vec![1] },
                value: 0.0,
            },
            FilteredSimplex {
                simplex: Simplex { vertices: vec![2] },
                value: 0.0,
            },
            FilteredSimplex {
                simplex: Simplex {
                    vertices: vec![0, 1],
                },
                value: 1.0,
            },
            FilteredSimplex {
                simplex: Simplex {
                    vertices: vec![0, 2],
                },
                value: 1.0,
            },
            FilteredSimplex {
                simplex: Simplex {
                    vertices: vec![1, 2],
                },
                value: 1.0,
            },
            FilteredSimplex {
                simplex: Simplex {
                    vertices: vec![0, 1, 2],
                },
                value: 2.0,
            },
        ];
        Filtration::new(simplices).expect("ok")
    }

    #[test]
    fn boundary_matrix_from_triangle() {
        let filt = make_triangle_filtration();
        let m = BoundaryMatrix::from_filtration(&filt).expect("ok");
        assert_eq!(m.n_cols, 7);
        // Vertex columns have no boundary
        assert!(m.is_zero(0));
        // Edge [0,1] has boundary rows for vertex 0 and vertex 1
        let edge_col = m.columns[3].clone(); // [0,1] is 4th simplex (index 3)
        assert_eq!(edge_col.len(), 2);
    }

    #[test]
    fn add_cols_xor() {
        let mut m = BoundaryMatrix {
            n_rows: 3,
            n_cols: 2,
            columns: vec![vec![0, 2], vec![0, 1]],
        };
        m.add_cols(0, 1);
        // XOR: {0,2} ⊕ {0,1} = {1,2}
        assert_eq!(m.columns[0], vec![1, 2]);
    }
}
