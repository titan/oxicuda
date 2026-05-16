//! Standard persistence algorithm: column reduction of boundary matrix over Z₂.
//!
//! Implements the Edelsbrunner-Letscher-Zomorodian (2002) algorithm.

use crate::homology::boundary::BoundaryMatrix;
use std::collections::HashMap;

/// Reduce the boundary matrix using the standard persistence algorithm (ELZ 2002).
///
/// Operates in-place on `matrix` (column-adds over Z₂).
///
/// Returns a map `pivot_col[j] = Some(j')` where `j'` is the column whose pivot row is `j`,
/// meaning simplex `j` (a cycle) is killed by simplex `j'`.  `pivot_col[j] = None` means no
/// column currently has `j` as its lowest nonzero row.
///
/// After reduction, each nonzero column has a **unique** pivot row (invariant of the
/// reduced matrix).
pub fn reduce_boundary_matrix(matrix: &mut BoundaryMatrix) -> Vec<Option<usize>> {
    let n = matrix.n_cols;
    // low_to_col: lowest nonzero row r → the leftmost column j with low(j) = r.
    let mut low_to_col: HashMap<usize, usize> = HashMap::new();

    for j in 0..n {
        loop {
            if matrix.is_zero(j) {
                break;
            }
            // low(j) is Some because column is non-zero
            let low_j = matrix.low(j).expect("column known non-zero");
            match low_to_col.get(&low_j).copied() {
                Some(j_prime) => {
                    // Reduce: add column j_prime to column j
                    matrix.add_cols(j, j_prime);
                    // Continue: low(j) may have changed
                }
                None => {
                    // Column j is now reduced; record its pivot
                    low_to_col.insert(low_j, j);
                    break;
                }
            }
        }
    }

    // Build result: pivot_col[row] = column that has this row as its pivot
    let mut result = vec![None; n];
    for (&row, &col) in &low_to_col {
        if row < n {
            result[row] = Some(col);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduces_simple_matrix() {
        // Build a simple 3-column boundary matrix manually
        // Column 0: empty (vertex)
        // Column 1: empty (vertex)
        // Column 2: rows {0, 1} (edge between vertices 0 and 1)
        let mut m = BoundaryMatrix {
            n_rows: 3,
            n_cols: 3,
            columns: vec![vec![], vec![], vec![0, 1]],
        };
        let pivots = reduce_boundary_matrix(&mut m);
        // Column 2 has pivot at row 1
        assert_eq!(pivots[1], Some(2));
        assert_eq!(pivots[0], None);
    }
}
