//! Architecture encoding and genetic operators for evolutionary NAS.
//!
//! An architecture is encoded as `Vec<usize>`: one op-index per edge.
//! Uniform crossover swaps genes with probability ~0.5; mutation changes one gene.

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;

// ─── ArchEncoding ────────────────────────────────────────────────────────────

/// Architecture encoding: a vector of op indices, one per edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchEncoding {
    /// Op index per edge.
    pub genes: Vec<usize>,
    /// Total number of candidate ops (upper bound for each gene).
    pub n_ops: usize,
}

impl ArchEncoding {
    /// Create an encoding with validation.
    pub fn new(genes: Vec<usize>, n_ops: usize) -> NasResult<Self> {
        if n_ops == 0 {
            return Err(NasError::InvalidNumOps);
        }
        for &g in &genes {
            if g >= n_ops {
                return Err(NasError::InvalidArchEncoding);
            }
        }
        Ok(Self { genes, n_ops })
    }

    /// Sample a random encoding with `n_edges` genes.
    #[must_use]
    pub fn random(n_edges: usize, n_ops: usize, rng: &mut LcgRng) -> Self {
        let genes = (0..n_edges).map(|_| rng.next_usize(n_ops)).collect();
        Self { genes, n_ops }
    }

    /// Number of edges (genes) in this encoding.
    #[must_use]
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    /// Returns true if the encoding is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.genes.is_empty()
    }

    /// Uniform crossover: each gene is taken from `self` or `other` with equal probability.
    ///
    /// Returns the child encoding.
    pub fn crossover(&self, other: &Self, rng: &mut LcgRng) -> NasResult<Self> {
        if self.len() != other.len() {
            return Err(NasError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }
        if self.n_ops != other.n_ops {
            return Err(NasError::InvalidArchEncoding);
        }
        let genes = self
            .genes
            .iter()
            .zip(other.genes.iter())
            .map(|(&a, &b)| if rng.next_f32() < 0.5 { a } else { b })
            .collect();
        Ok(Self {
            genes,
            n_ops: self.n_ops,
        })
    }

    /// Mutate exactly one randomly chosen gene to a random different op index.
    pub fn mutate_one(&mut self, rng: &mut LcgRng) -> NasResult<()> {
        let n = self.len();
        if n == 0 {
            return Err(NasError::EmptySearchSpace);
        }
        if self.n_ops <= 1 {
            // nothing to change
            return Ok(());
        }
        let idx = rng.next_usize(n);
        let old = self.genes[idx];
        // Sample a different value
        let mut new_val = rng.next_usize(self.n_ops - 1);
        if new_val >= old {
            new_val += 1;
        }
        self.genes[idx] = new_val;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_encoding_valid_genes() {
        let mut rng = LcgRng::new(42);
        let enc = ArchEncoding::random(14, 8, &mut rng);
        assert_eq!(enc.len(), 14);
        assert!(enc.genes.iter().all(|&g| g < 8));
    }

    #[test]
    fn crossover_child_genes_from_parents() {
        let mut rng = LcgRng::new(1);
        let a = ArchEncoding::random(10, 4, &mut rng);
        let b = ArchEncoding::random(10, 4, &mut rng);
        let c = a
            .crossover(&b, &mut rng)
            .expect("test invariant: crossover");
        assert_eq!(c.len(), 10);
        for i in 0..10 {
            assert!(c.genes[i] == a.genes[i] || c.genes[i] == b.genes[i]);
        }
    }

    #[test]
    fn mutate_changes_exactly_one_gene() {
        let mut rng = LcgRng::new(7);
        let original = ArchEncoding::random(14, 8, &mut rng);
        let mut mutated = original.clone();
        mutated
            .mutate_one(&mut rng)
            .expect("test invariant: mutate");
        let diff: usize = original
            .genes
            .iter()
            .zip(mutated.genes.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(diff, 1, "exactly 1 gene must change");
    }
}
