//! Selection operators: tournament, roulette, rank.

use super::individual::Individual;
use crate::{EvolError, EvolResult, handle::LcgRng};

/// k-tournament selection: sample k individuals at random; return the index of the one
/// with the lowest fitness (best for minimisation).
///
/// # Errors
/// Returns `EvolError::EmptyPopulation` if `pop` is empty or `EvolError::InvalidParameter`
/// if `k == 0`.
pub fn tournament_select(pop: &[Individual], k: usize, rng: &mut LcgRng) -> EvolResult<usize> {
    if pop.is_empty() {
        return Err(EvolError::EmptyPopulation);
    }
    if k == 0 {
        return Err(EvolError::InvalidParameter(
            "tournament size k must be >= 1".to_owned(),
        ));
    }
    let n = pop.len();
    let first = rng.next_usize(n);
    let mut best_idx = first;
    let mut best_fit = pop[first].fitness;
    for _ in 1..k {
        let candidate = rng.next_usize(n);
        if pop[candidate].fitness < best_fit {
            best_fit = pop[candidate].fitness;
            best_idx = candidate;
        }
    }
    Ok(best_idx)
}

/// Fitness-proportional (roulette wheel) selection for minimisation.
///
/// Each individual is assigned a weight proportional to `max_f - f(x) + eps` so that
/// the lowest-fitness (best) individual has the highest selection probability.
///
/// # Errors
/// Returns `EvolError::EmptyPopulation` if `pop` is empty.
pub fn roulette_select(pop: &[Individual], rng: &mut LcgRng) -> EvolResult<usize> {
    if pop.is_empty() {
        return Err(EvolError::EmptyPopulation);
    }
    // Find finite maximum fitness; fall back to 0 if all are infinite.
    let max_f = pop
        .iter()
        .filter(|ind| ind.fitness.is_finite())
        .map(|ind| ind.fitness)
        .fold(f64::NEG_INFINITY, f64::max);
    let max_f = if max_f.is_finite() { max_f } else { 0.0 };

    // Weight = max_f - f + 1 (always positive)
    let weights: Vec<f64> = pop
        .iter()
        .map(|ind| {
            if ind.fitness.is_finite() {
                max_f - ind.fitness + 1.0
            } else {
                // Infinite-fitness individuals get near-zero weight
                1e-300
            }
        })
        .collect();

    let total: f64 = weights.iter().sum();
    let threshold = rng.next_f64() * total;
    let mut cumsum = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumsum += w;
        if cumsum >= threshold {
            return Ok(i);
        }
    }
    // Fallback: return last
    Ok(pop.len() - 1)
}

/// Linear rank selection: assign ranks 1..=n (rank 1 = worst) and select proportionally
/// to rank. This decouples selection pressure from raw fitness magnitudes.
///
/// The population must already be sorted by fitness (best first).
///
/// # Errors
/// Returns `EvolError::EmptyPopulation` if `pop` is empty.
pub fn rank_select(pop: &[Individual], rng: &mut LcgRng) -> EvolResult<usize> {
    if pop.is_empty() {
        return Err(EvolError::EmptyPopulation);
    }
    let n = pop.len();
    // Rank n (highest) for index 0 (best fitness), rank 1 for index n-1 (worst).
    // Total = n*(n+1)/2
    let total = (n * (n + 1)) / 2;
    let threshold = rng.next_usize(total) + 1; // [1, total]
    let mut cumsum = 0usize;
    for i in 0..n {
        let rank = n - i; // index 0 gets rank n
        cumsum += rank;
        if cumsum >= threshold {
            return Ok(i);
        }
    }
    Ok(n - 1)
}
