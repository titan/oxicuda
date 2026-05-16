//! 2-D pairwise CRF (image labelling).

use crate::error::{SeqError, SeqResult};

/// Pairwise 2-D CRF on a 4-connected grid.
///
/// * `unary[row * n_cols * n_labels + col * n_labels + l]` — unary cost φ_i(l)
/// * `pairwise[l_i * n_labels + l_j]` — pairwise compatibility ψ(l_i, l_j)
#[derive(Debug, Clone)]
pub struct GridCrf {
    pub n_rows: usize,
    pub n_cols: usize,
    pub n_labels: usize,
    pub unary: Vec<f64>,
    pub pairwise: Vec<f64>,
}

impl GridCrf {
    /// Create a CRF, validating shapes.
    pub fn new(
        n_rows: usize,
        n_cols: usize,
        n_labels: usize,
        unary: Vec<f64>,
        pairwise: Vec<f64>,
    ) -> SeqResult<Self> {
        if n_rows == 0 || n_cols == 0 || n_labels == 0 {
            return Err(SeqError::InvalidConfiguration(
                "grid dims must be > 0".to_string(),
            ));
        }
        if unary.len() != n_rows * n_cols * n_labels {
            return Err(SeqError::ShapeMismatch {
                expected: n_rows * n_cols * n_labels,
                got: unary.len(),
            });
        }
        if pairwise.len() != n_labels * n_labels {
            return Err(SeqError::ShapeMismatch {
                expected: n_labels * n_labels,
                got: pairwise.len(),
            });
        }
        Ok(Self {
            n_rows,
            n_cols,
            n_labels,
            unary,
            pairwise,
        })
    }

    /// Argmax of the unary cost at site (r, c) (used as ICM seed).
    pub fn unary_argmin(&self, r: usize, c: usize) -> SeqResult<usize> {
        if r >= self.n_rows || c >= self.n_cols {
            return Err(SeqError::IndexOutOfBounds {
                index: r * self.n_cols + c,
                len: self.n_rows * self.n_cols,
            });
        }
        let base = (r * self.n_cols + c) * self.n_labels;
        let mut best_l = 0usize;
        let mut best_v = f64::INFINITY;
        for l in 0..self.n_labels {
            let v = self.unary[base + l];
            if v < best_v {
                best_v = v;
                best_l = l;
            }
        }
        Ok(best_l)
    }

    /// Total energy of a labelling.
    pub fn energy(&self, labels: &[usize]) -> SeqResult<f64> {
        if labels.len() != self.n_rows * self.n_cols {
            return Err(SeqError::ShapeMismatch {
                expected: self.n_rows * self.n_cols,
                got: labels.len(),
            });
        }
        let mut e = 0.0;
        for r in 0..self.n_rows {
            for c in 0..self.n_cols {
                let l = labels[r * self.n_cols + c];
                if l >= self.n_labels {
                    return Err(SeqError::IndexOutOfBounds {
                        index: l,
                        len: self.n_labels,
                    });
                }
                e += self.unary[(r * self.n_cols + c) * self.n_labels + l];
                if r + 1 < self.n_rows {
                    let lr = labels[(r + 1) * self.n_cols + c];
                    e += self.pairwise[l * self.n_labels + lr];
                }
                if c + 1 < self.n_cols {
                    let lc = labels[r * self.n_cols + (c + 1)];
                    e += self.pairwise[l * self.n_labels + lc];
                }
            }
        }
        Ok(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_construct_and_energy() {
        let n_rows = 2;
        let n_cols = 2;
        let n_labels = 2;
        let unary = vec![0.0f64; n_rows * n_cols * n_labels];
        let pairwise = vec![0.0f64; n_labels * n_labels];
        let g = GridCrf::new(n_rows, n_cols, n_labels, unary, pairwise).expect("ok");
        let labs = vec![0, 1, 1, 0];
        assert_eq!(g.energy(&labs).expect("ok"), 0.0);
    }
}
