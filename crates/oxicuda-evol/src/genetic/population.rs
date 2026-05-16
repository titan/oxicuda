//! Population management: random initialisation, batch evaluation, sorting.

use super::individual::Individual;
use crate::{EvolError, EvolResult, handle::LcgRng};

/// A collection of `Individual`s with a shared dimensionality.
pub struct Population {
    /// All candidate solutions in this population.
    pub individuals: Vec<Individual>,
    /// Number of decision variables per individual.
    pub n_dims: usize,
}

impl Population {
    /// Create a population of `pop_size` individuals with genomes uniformly sampled in
    /// `[bounds.0, bounds.1]^n_dims`.
    pub fn new_random(
        pop_size: usize,
        n_dims: usize,
        bounds: (f64, f64),
        rng: &mut LcgRng,
    ) -> EvolResult<Self> {
        if pop_size == 0 {
            return Err(EvolError::EmptyPopulation);
        }
        if n_dims == 0 {
            return Err(EvolError::EmptyGenome);
        }
        if bounds.0 >= bounds.1 {
            return Err(EvolError::InvalidParameter(format!(
                "bounds ({}, {}) are invalid: lower must be strictly less than upper",
                bounds.0, bounds.1
            )));
        }
        let range = bounds.1 - bounds.0;
        let individuals = (0..pop_size)
            .map(|_| {
                let genome = (0..n_dims)
                    .map(|_| bounds.0 + rng.next_f64() * range)
                    .collect();
                Individual::new(genome)
            })
            .collect();
        Ok(Self {
            individuals,
            n_dims,
        })
    }

    /// Evaluate all individuals in-place using the provided objective function.
    pub fn evaluate_all<F: Fn(&[f64]) -> f64>(&mut self, f: &F) {
        for ind in &mut self.individuals {
            ind.evaluate(f);
        }
    }

    /// Sort individuals by fitness in ascending order (best first for minimisation).
    pub fn sort_by_fitness(&mut self) {
        self.individuals.sort_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Return a reference to the best (lowest-fitness) individual.
    pub fn best(&self) -> EvolResult<&Individual> {
        self.individuals
            .iter()
            .min_by(|a, b| {
                a.fitness
                    .partial_cmp(&b.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or(EvolError::EmptyPopulation)
    }

    /// Return the current population size.
    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    /// Return `true` if the population has no individuals.
    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }
}
