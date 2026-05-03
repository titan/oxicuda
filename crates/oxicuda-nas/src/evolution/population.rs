//! Population management for evolutionary NAS.

use crate::error::{NasError, NasResult};
use crate::evolution::encoding::ArchEncoding;
use crate::evolution::nsga2::Individual;
use crate::handle::LcgRng;

// ─── Population ───────────────────────────────────────────────────────────────

/// A population of `Individual`s for evolutionary NAS.
#[derive(Debug, Clone)]
pub struct Population {
    /// All individuals in the current population.
    pub individuals: Vec<Individual>,
    /// Number of candidate ops (for generating new random individuals).
    pub n_ops: usize,
    /// Number of edges per individual.
    pub n_edges: usize,
}

impl Population {
    /// Construct a random initial population.
    pub fn random(
        pop_size: usize,
        n_edges: usize,
        n_ops: usize,
        n_objectives: usize,
        rng: &mut LcgRng,
    ) -> NasResult<Self> {
        if pop_size < 2 {
            return Err(NasError::PopulationTooSmall {
                min: 2,
                got: pop_size,
            });
        }
        let individuals = (0..pop_size)
            .map(|_| {
                let enc = ArchEncoding::random(n_edges, n_ops, rng);
                Individual {
                    encoding: enc.genes,
                    objectives: vec![0.0_f32; n_objectives],
                    rank: 0,
                    crowding_distance: 0.0,
                }
            })
            .collect();
        Ok(Self {
            individuals,
            n_ops,
            n_edges,
        })
    }

    /// Population size.
    #[must_use]
    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    /// Returns true if population is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }

    /// Elitism: return the top `k` individuals by (rank, -crowding_distance).
    #[must_use]
    pub fn elites(&self, k: usize) -> Vec<&Individual> {
        let mut sorted: Vec<&Individual> = self.individuals.iter().collect();
        sorted.sort_by(|a, b| {
            a.rank.cmp(&b.rank).then(
                b.crowding_distance
                    .partial_cmp(&a.crowding_distance)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
        sorted.into_iter().take(k).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_population_correct_size() {
        let mut rng = LcgRng::new(42);
        let pop =
            Population::random(20, 14, 8, 2, &mut rng).expect("test invariant: random population");
        assert_eq!(pop.len(), 20);
    }

    #[test]
    fn population_too_small_errors() {
        let mut rng = LcgRng::new(1);
        let result = Population::random(1, 14, 8, 2, &mut rng);
        assert!(result.is_err());
    }
}
