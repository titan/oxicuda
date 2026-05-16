//! Pairwise MRF on a generic graph plus an Ising-model specialisation.

use crate::error::{SeqError, SeqResult};

/// Pairwise Markov Random Field on an undirected graph.
///
/// * `unary[i*n_labels + l]` — local cost on node `i` for label `l`
/// * `pairwise[(edge_idx)*n_labels*n_labels + l_i*n_labels + l_j]` — pairwise
///   compatibility on `edge[edge_idx] = (u, v)`.
#[derive(Debug, Clone)]
pub struct Mrf {
    pub n_nodes: usize,
    pub n_labels: usize,
    pub edges: Vec<(usize, usize)>,
    pub unary: Vec<f64>,
    pub pairwise: Vec<f64>,
}

impl Mrf {
    /// Build a pairwise MRF, validating shapes.
    pub fn new(
        n_nodes: usize,
        n_labels: usize,
        edges: Vec<(usize, usize)>,
        unary: Vec<f64>,
        pairwise: Vec<f64>,
    ) -> SeqResult<Self> {
        if n_nodes == 0 || n_labels == 0 {
            return Err(SeqError::InvalidConfiguration(
                "n_nodes and n_labels must be > 0".to_string(),
            ));
        }
        if unary.len() != n_nodes * n_labels {
            return Err(SeqError::ShapeMismatch {
                expected: n_nodes * n_labels,
                got: unary.len(),
            });
        }
        if pairwise.len() != edges.len() * n_labels * n_labels {
            return Err(SeqError::ShapeMismatch {
                expected: edges.len() * n_labels * n_labels,
                got: pairwise.len(),
            });
        }
        for &(u, v) in &edges {
            if u >= n_nodes || v >= n_nodes || u == v {
                return Err(SeqError::GraphInvariantViolated(format!(
                    "edge ({u}, {v}) invalid for n_nodes={n_nodes}"
                )));
            }
        }
        Ok(Self {
            n_nodes,
            n_labels,
            edges,
            unary,
            pairwise,
        })
    }

    /// Energy of an assignment.
    pub fn energy(&self, labels: &[usize]) -> SeqResult<f64> {
        if labels.len() != self.n_nodes {
            return Err(SeqError::ShapeMismatch {
                expected: self.n_nodes,
                got: labels.len(),
            });
        }
        let mut e = 0.0;
        for (i, &l) in labels.iter().enumerate() {
            if l >= self.n_labels {
                return Err(SeqError::IndexOutOfBounds {
                    index: l,
                    len: self.n_labels,
                });
            }
            e += self.unary[i * self.n_labels + l];
        }
        let l2 = self.n_labels * self.n_labels;
        for (e_idx, &(u, v)) in self.edges.iter().enumerate() {
            let lu = labels[u];
            let lv = labels[v];
            e += self.pairwise[e_idx * l2 + lu * self.n_labels + lv];
        }
        Ok(e)
    }
}

/// Ising model on a regular 2-D grid (4-connected).
///
/// Energy: `E(s) = − Σ_i h s_i − Σ_(i,j) J s_i s_j`, with `s_i ∈ {−1, +1}`.
#[derive(Debug, Clone)]
pub struct IsingModel {
    pub n_rows: usize,
    pub n_cols: usize,
    pub field: f64,
    pub coupling: f64,
    pub beta: f64,
}

impl IsingModel {
    /// Construct an Ising model.
    pub fn new(
        n_rows: usize,
        n_cols: usize,
        field: f64,
        coupling: f64,
        beta: f64,
    ) -> SeqResult<Self> {
        if n_rows == 0 || n_cols == 0 {
            return Err(SeqError::InvalidConfiguration(
                "grid dims must be > 0".to_string(),
            ));
        }
        if beta <= 0.0 || !beta.is_finite() {
            return Err(SeqError::InvalidParameter {
                name: "beta".to_string(),
                value: beta,
            });
        }
        Ok(Self {
            n_rows,
            n_cols,
            field,
            coupling,
            beta,
        })
    }

    /// Total energy of an Ising spin configuration (`±1` values).
    pub fn energy(&self, spins: &[i32]) -> SeqResult<f64> {
        if spins.len() != self.n_rows * self.n_cols {
            return Err(SeqError::ShapeMismatch {
                expected: self.n_rows * self.n_cols,
                got: spins.len(),
            });
        }
        let mut e = 0.0;
        for r in 0..self.n_rows {
            for c in 0..self.n_cols {
                let s = spins[r * self.n_cols + c] as f64;
                e -= self.field * s;
                if r + 1 < self.n_rows {
                    let s2 = spins[(r + 1) * self.n_cols + c] as f64;
                    e -= self.coupling * s * s2;
                }
                if c + 1 < self.n_cols {
                    let s2 = spins[r * self.n_cols + (c + 1)] as f64;
                    e -= self.coupling * s * s2;
                }
            }
        }
        Ok(e)
    }

    /// Mean magnetisation `(1/N) Σ s_i`.
    pub fn magnetisation(&self, spins: &[i32]) -> SeqResult<f64> {
        if spins.len() != self.n_rows * self.n_cols {
            return Err(SeqError::ShapeMismatch {
                expected: self.n_rows * self.n_cols,
                got: spins.len(),
            });
        }
        let s: i64 = spins.iter().map(|&x| x as i64).sum();
        Ok(s as f64 / spins.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mrf_construct_and_energy() {
        let m = Mrf::new(
            3,
            2,
            vec![(0, 1), (1, 2)],
            vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0],
        )
        .expect("ok");
        let e = m.energy(&[0, 0, 0]).expect("ok");
        assert!(e.is_finite());
    }

    #[test]
    fn ising_all_up_magnetisation_one() {
        let m = IsingModel::new(3, 3, 0.0, 1.0, 1.0).expect("ok");
        let spins = vec![1i32; 9];
        assert!((m.magnetisation(&spins).expect("ok") - 1.0).abs() < 1e-12);
    }
}
