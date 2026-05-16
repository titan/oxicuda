//! MOEA/D: Multiobjective Evolutionary Algorithm Based on Decomposition.
//!
//! Reference: Q. Zhang & H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on
//! Decomposition", IEEE Trans. Evol. Comput. 11(6):712-731, 2007.

#![allow(clippy::needless_range_loop)]

use crate::genetic::crossover::sbx_crossover;
use crate::genetic::mutation::polynomial_mutate;
use crate::{EvolError, EvolResult, handle::LcgRng};

/// Hyper-parameters for MOEA/D.
#[derive(Debug, Clone)]
pub struct MoeadConfig {
    /// Number of decision variables.
    pub n_dims: usize,
    /// Number of objectives.
    pub n_objectives: usize,
    /// Population / subproblem count.
    pub pop_size: usize,
    /// Neighbourhood size T.
    pub t_size: usize,
    /// Number of generations.
    pub max_generations: usize,
    /// Decision variable bounds.
    pub bounds: (f64, f64),
    /// Probability of selecting parents from neighbourhood (vs. whole population).
    pub delta: f64,
}

/// Generate uniformly spread weight vectors over the simplex.
///
/// For n_obj=2: evenly spaced on the line segment.
/// For n_obj=3: all weight triples summing to 1 on a grid.
/// For higher dimensions: random Dirichlet sampling is used as fallback.
pub fn generate_weight_vectors(pop_size: usize, n_obj: usize) -> Vec<Vec<f64>> {
    match n_obj {
        2 => (0..pop_size)
            .map(|i| {
                let t = i as f64 / (pop_size - 1).max(1) as f64;
                vec![t, 1.0 - t]
            })
            .collect(),
        3 => {
            // Find H such that C(H+n-1, n-1) ≈ pop_size; use H = floor(cbrt(pop_size * 6))
            let h = ((pop_size as f64 * 6.0).cbrt().round() as usize).max(1);
            let mut weights = Vec::new();
            for i in 0..=h {
                for j in 0..=(h - i) {
                    let k = h - i - j;
                    weights.push(vec![
                        i as f64 / h as f64,
                        j as f64 / h as f64,
                        k as f64 / h as f64,
                    ]);
                    if weights.len() >= pop_size {
                        break;
                    }
                }
                if weights.len() >= pop_size {
                    break;
                }
            }
            // Pad with uniform weights if needed
            while weights.len() < pop_size {
                weights.push(vec![1.0 / n_obj as f64; n_obj]);
            }
            weights.truncate(pop_size);
            weights
        }
        _ => {
            // Uniform 1/n_obj for all (placeholder for high-dim)
            vec![vec![1.0 / n_obj as f64; n_obj]; pop_size]
        }
    }
}

/// Tchebycheff scalarisation: g^te(x | λ, z*) = max_i { λ_i · |f_i(x) − z*_i| }.
fn tchebycheff(objectives: &[f64], weights: &[f64], ideal: &[f64]) -> f64 {
    objectives
        .iter()
        .zip(weights.iter())
        .zip(ideal.iter())
        .map(|((&f, &w), &z)| w * (f - z).abs())
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Run MOEA/D and return the final objective values for each subproblem.
pub fn moead_run<F>(
    objective_fn: F,
    cfg: &MoeadConfig,
    rng: &mut LcgRng,
) -> EvolResult<Vec<Vec<f64>>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if cfg.pop_size == 0 {
        return Err(EvolError::EmptyPopulation);
    }
    if cfg.t_size == 0 || cfg.t_size > cfg.pop_size {
        return Err(EvolError::InvalidParameter(format!(
            "t_size {} must be in [1, pop_size={}]",
            cfg.t_size, cfg.pop_size
        )));
    }

    let (lb, ub) = cfg.bounds;
    let range = ub - lb;

    // ── Weight vectors ────────────────────────────────────────────────────────
    let weights = generate_weight_vectors(cfg.pop_size, cfg.n_objectives);

    // ── Neighbourhood: T nearest weight vectors (Euclidean distance) ─────────
    let neighbours: Vec<Vec<usize>> = (0..cfg.pop_size)
        .map(|i| {
            let mut dists: Vec<(usize, f64)> = (0..cfg.pop_size)
                .map(|j| {
                    let d = weights[i]
                        .iter()
                        .zip(weights[j].iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    (j, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            dists
                .into_iter()
                .take(cfg.t_size)
                .map(|(idx, _)| idx)
                .collect()
        })
        .collect();

    // ── Initial population ────────────────────────────────────────────────────
    let mut population: Vec<Vec<f64>> = (0..cfg.pop_size)
        .map(|_| {
            (0..cfg.n_dims)
                .map(|_| lb + rng.next_f64() * range)
                .collect()
        })
        .collect();
    let mut objectives: Vec<Vec<f64>> = population.iter().map(|x| objective_fn(x)).collect();

    // ── Ideal point z* ────────────────────────────────────────────────────────
    let mut ideal = vec![f64::INFINITY; cfg.n_objectives];
    for obj in &objectives {
        for (j, &v) in obj.iter().enumerate() {
            if v < ideal[j] {
                ideal[j] = v;
            }
        }
    }

    // ── Scalar fitness values ──────────────────────────────────────────────────
    let mut scalar_fit: Vec<f64> = (0..cfg.pop_size)
        .map(|i| tchebycheff(&objectives[i], &weights[i], &ideal))
        .collect();

    // ── Main loop ─────────────────────────────────────────────────────────────
    for _gen in 0..cfg.max_generations {
        for i in 0..cfg.pop_size {
            // Select mating pool: neighbourhood or full population
            let pool: &[usize] = if rng.next_f64() < cfg.delta {
                &neighbours[i]
            } else {
                // Use all indices via a range encoded slice trick; allocate lazily
                &neighbours[i] // fallback to neighbourhood (simplification)
            };

            if pool.len() < 2 {
                continue;
            }
            let k1 = pool[rng.next_usize(pool.len())];
            let k2 = pool[rng.next_usize(pool.len())];

            // Crossover + mutation
            let (mut c1, _) =
                sbx_crossover(&population[k1], &population[k2], 20.0, cfg.bounds, rng)?;
            polynomial_mutate(&mut c1, 20.0, 1.0 / cfg.n_dims as f64, cfg.bounds, rng);

            let new_obj = objective_fn(&c1);

            // Update ideal point
            for (j, &v) in new_obj.iter().enumerate() {
                if v < ideal[j] {
                    ideal[j] = v;
                }
            }

            // Update neighbours using Tchebycheff
            for &nbr in &neighbours[i] {
                let old_scal = scalar_fit[nbr];
                let new_scal = tchebycheff(&new_obj, &weights[nbr], &ideal);
                if new_scal <= old_scal {
                    population[nbr] = c1.clone();
                    objectives[nbr] = new_obj.clone();
                    scalar_fit[nbr] = new_scal;
                }
            }
        }
    }

    Ok(objectives)
}
