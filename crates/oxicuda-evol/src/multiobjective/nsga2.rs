//! NSGA-II: Non-dominated Sorting Genetic Algorithm II.
//!
//! Reference: K. Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II",
//! IEEE Trans. Evol. Comput. 6(2):182-197, 2002.

#![allow(clippy::needless_range_loop)]

use crate::genetic::crossover::sbx_crossover;
use crate::genetic::mutation::polynomial_mutate;
use crate::{EvolError, EvolResult, handle::LcgRng};

/// An individual candidate solution in a multi-objective setting.
#[derive(Debug, Clone)]
pub struct MultiObjectiveIndividual {
    /// Decision variable vector.
    pub genome: Vec<f64>,
    /// Objective function values (length = n_objectives).
    pub objectives: Vec<f64>,
    /// Non-domination rank (0 = Pareto front, higher = worse).
    pub rank: usize,
    /// Crowding distance (∞ for boundary individuals; used for diversity preservation).
    pub crowding_dist: f64,
}

impl MultiObjectiveIndividual {
    /// Returns `true` if `self` dominates `other` (all objectives ≤, at least one <).
    pub fn dominates(&self, other: &Self) -> bool {
        let mut at_least_one_strictly_less = false;
        for (a, b) in self.objectives.iter().zip(other.objectives.iter()) {
            if a > b {
                return false;
            }
            if a < b {
                at_least_one_strictly_less = true;
            }
        }
        at_least_one_strictly_less
    }
}

/// Fast non-dominated sort of a population.
///
/// Returns a list of fronts: `fronts[0]` = Pareto front indices, `fronts[1]` = second
/// front, etc.  Time complexity: O(M·N²) where M = number of objectives, N = pop size.
pub fn fast_nondominated_sort(population: &[MultiObjectiveIndividual]) -> Vec<Vec<usize>> {
    let n = population.len();
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut domination_count: Vec<usize> = vec![0; n];
    let mut fronts: Vec<Vec<usize>> = vec![Vec::new()];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if population[i].dominates(&population[j]) {
                dominated_by[i].push(j);
            } else if population[j].dominates(&population[i]) {
                domination_count[i] += 1;
            }
        }
        if domination_count[i] == 0 {
            fronts[0].push(i);
        }
    }

    let mut front_idx = 0;
    while !fronts[front_idx].is_empty() {
        let mut next_front = Vec::new();
        for &i in &fronts[front_idx] {
            for &j in &dominated_by[i] {
                domination_count[j] = domination_count[j].saturating_sub(1);
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        front_idx += 1;
        if next_front.is_empty() {
            break;
        }
        fronts.push(next_front);
    }

    fronts
}

/// Compute and assign crowding distances for all individuals in one front.
///
/// Boundary individuals (min/max on any objective) receive `f64::INFINITY`.
pub fn crowding_distance(population: &mut [MultiObjectiveIndividual], front: &[usize]) {
    let m = if let Some(ind) = population.first() {
        ind.objectives.len()
    } else {
        return;
    };
    let f_len = front.len();
    if f_len == 0 {
        return;
    }
    // Reset distances for this front
    for &i in front {
        population[i].crowding_dist = 0.0;
    }

    for obj in 0..m {
        // Sort front by this objective
        let mut sorted_front = front.to_vec();
        sorted_front.sort_by(|&a, &b| {
            population[a].objectives[obj]
                .partial_cmp(&population[b].objectives[obj])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let f_min = population[sorted_front[0]].objectives[obj];
        let f_max = population[sorted_front[f_len - 1]].objectives[obj];
        let range = (f_max - f_min).max(1e-300);

        // Boundary points get infinite crowding distance
        population[sorted_front[0]].crowding_dist = f64::INFINITY;
        population[sorted_front[f_len - 1]].crowding_dist = f64::INFINITY;

        for k in 1..(f_len - 1) {
            let prev = population[sorted_front[k - 1]].objectives[obj];
            let next = population[sorted_front[k + 1]].objectives[obj];
            let contribution = (next - prev) / range;
            if population[sorted_front[k]].crowding_dist.is_finite() {
                population[sorted_front[k]].crowding_dist += contribution;
            }
        }
    }
}

/// NSGA-II binary tournament selection.
///
/// Compares two random individuals: prefers lower rank; breaks ties by higher crowding distance.
pub fn nsga2_tournament(pop: &[MultiObjectiveIndividual], rng: &mut LcgRng) -> usize {
    let n = pop.len();
    let a = rng.next_usize(n);
    let b = rng.next_usize(n);
    if pop[a].rank < pop[b].rank {
        a
    } else if pop[b].rank < pop[a].rank {
        b
    } else if pop[a].crowding_dist > pop[b].crowding_dist {
        a
    } else {
        b
    }
}

/// Hyper-parameters for an NSGA-II run.
#[derive(Debug, Clone)]
pub struct Nsga2Config {
    /// Number of decision variables.
    pub n_dims: usize,
    /// Number of objective functions.
    pub n_objectives: usize,
    /// Population size (must be even).
    pub pop_size: usize,
    /// Number of generations.
    pub max_generations: usize,
    /// SBX distribution index.
    pub crossover_eta: f64,
    /// Polynomial mutation distribution index.
    pub mutation_eta: f64,
    /// Per-gene mutation probability (typically 1/n_dims).
    pub mutation_prob: f64,
    /// Shared decision variable bounds.
    pub bounds: (f64, f64),
}

/// Run NSGA-II and return the final population (with ranks and crowding distances assigned).
pub fn nsga2_run<F>(
    objective_fn: F,
    cfg: &Nsga2Config,
    rng: &mut LcgRng,
) -> EvolResult<Vec<MultiObjectiveIndividual>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if cfg.pop_size < 2 {
        return Err(EvolError::PopulationTooSmall {
            size: cfg.pop_size,
            op: "NSGA-II",
        });
    }
    let (lb, ub) = cfg.bounds;
    let range = ub - lb;

    // Initialise population
    let mut population: Vec<MultiObjectiveIndividual> = (0..cfg.pop_size)
        .map(|_| {
            let genome: Vec<f64> = (0..cfg.n_dims)
                .map(|_| lb + rng.next_f64() * range)
                .collect();
            let objectives = objective_fn(&genome);
            MultiObjectiveIndividual {
                genome,
                objectives,
                rank: 0,
                crowding_dist: 0.0,
            }
        })
        .collect();

    // Sort and assign ranks + crowding
    assign_ranks_and_crowding(&mut population);

    for _gen in 0..cfg.max_generations {
        // Generate offspring
        let mut offspring = Vec::with_capacity(cfg.pop_size);
        while offspring.len() < cfg.pop_size {
            let p1 = nsga2_tournament(&population, rng);
            let p2 = nsga2_tournament(&population, rng);
            let (mut c1_genome, mut c2_genome) = sbx_crossover(
                &population[p1].genome,
                &population[p2].genome,
                cfg.crossover_eta,
                cfg.bounds,
                rng,
            )?;
            polynomial_mutate(
                &mut c1_genome,
                cfg.mutation_eta,
                cfg.mutation_prob,
                cfg.bounds,
                rng,
            );
            polynomial_mutate(
                &mut c2_genome,
                cfg.mutation_eta,
                cfg.mutation_prob,
                cfg.bounds,
                rng,
            );
            let obj1 = objective_fn(&c1_genome);
            let obj2 = objective_fn(&c2_genome);
            if offspring.len() < cfg.pop_size {
                offspring.push(MultiObjectiveIndividual {
                    genome: c1_genome,
                    objectives: obj1,
                    rank: 0,
                    crowding_dist: 0.0,
                });
            }
            if offspring.len() < cfg.pop_size {
                offspring.push(MultiObjectiveIndividual {
                    genome: c2_genome,
                    objectives: obj2,
                    rank: 0,
                    crowding_dist: 0.0,
                });
            }
        }

        // Combine parent + offspring
        let mut combined = population;
        combined.extend(offspring);
        assign_ranks_and_crowding(&mut combined);

        // Environmental selection: fill next generation front by front
        let fronts = fast_nondominated_sort(&combined);
        let mut next_population = Vec::with_capacity(cfg.pop_size);
        'outer: for front in &fronts {
            if next_population.len() + front.len() <= cfg.pop_size {
                for &i in front {
                    next_population.push(combined[i].clone());
                }
            } else {
                // Partial front: sort by crowding distance descending
                let mut partial: Vec<usize> = front.clone();
                partial.sort_by(|&a, &b| {
                    combined[b]
                        .crowding_dist
                        .partial_cmp(&combined[a].crowding_dist)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let remaining = cfg.pop_size - next_population.len();
                for &i in partial.iter().take(remaining) {
                    next_population.push(combined[i].clone());
                }
                break 'outer;
            }
        }

        population = next_population;
        assign_ranks_and_crowding(&mut population);
    }

    Ok(population)
}

/// Helper: run fast non-dominated sort and assign ranks + crowding distances.
fn assign_ranks_and_crowding(population: &mut [MultiObjectiveIndividual]) {
    let fronts = fast_nondominated_sort(population);
    for (rank, front) in fronts.iter().enumerate() {
        for &i in front {
            population[i].rank = rank;
        }
        crowding_distance(population, front);
    }
}
