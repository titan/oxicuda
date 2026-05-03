//! NSGA-II: fast non-dominated sorting, crowding distance, and tournament selection.
//!
//! Reference: Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II",
//! IEEE TEVC 6(2), 2002.

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;

// ─── Individual ───────────────────────────────────────────────────────────────

/// A single candidate architecture with multi-objective evaluation results.
#[derive(Debug, Clone)]
pub struct Individual {
    /// Op-index per edge (architecture encoding).
    pub encoding: Vec<usize>,
    /// Objective values (e.g., `[accuracy_proxy, latency_ms]`), to be minimised.
    pub objectives: Vec<f32>,
    /// Non-domination rank assigned by NSGA-II (0 = Pareto front).
    pub rank: usize,
    /// Crowding distance within its front.
    pub crowding_distance: f32,
}

impl Individual {
    /// Returns true if `self` Pareto-dominates `other`.
    ///
    /// `self` dominates `other` iff:
    /// - all objectives of `self` ≤ all objectives of `other`, AND
    /// - at least one objective of `self` < the corresponding objective of `other`.
    #[must_use]
    pub fn dominates(&self, other: &Self) -> bool {
        if self.objectives.len() != other.objectives.len() {
            return false;
        }
        let all_leq = self
            .objectives
            .iter()
            .zip(other.objectives.iter())
            .all(|(a, b)| a <= b);
        let any_lt = self
            .objectives
            .iter()
            .zip(other.objectives.iter())
            .any(|(a, b)| a < b);
        all_leq && any_lt
    }
}

// ─── fast_non_dominated_sort ──────────────────────────────────────────────────

/// NSGA-II fast non-dominated sorting.
///
/// Assigns `rank` to each individual: rank 0 = Pareto-optimal front,
/// rank 1 = Pareto-optimal after removing rank-0, etc.
///
/// Time complexity: O(M × N²) where M = number of objectives, N = population size.
pub fn fast_non_dominated_sort(individuals: &mut [Individual]) {
    let n = individuals.len();
    // domination_count[i] = number of solutions that dominate i
    let mut dom_count = vec![0usize; n];
    // dominated_by[i] = set of solutions dominated by i
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    // fronts[0] = indices of front 0, etc.
    let mut fronts: Vec<Vec<usize>> = vec![Vec::new()];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            // Check dominance using objectives directly (avoid borrow issues)
            let len = individuals[i]
                .objectives
                .len()
                .min(individuals[j].objectives.len());
            let all_leq =
                (0..len).all(|k| individuals[i].objectives[k] <= individuals[j].objectives[k]);
            let any_lt =
                (0..len).any(|k| individuals[i].objectives[k] < individuals[j].objectives[k]);
            if all_leq && any_lt {
                // i dominates j
                dominated_by[i].push(j);
            }
            let all_leq_ji =
                (0..len).all(|k| individuals[j].objectives[k] <= individuals[i].objectives[k]);
            let any_lt_ji =
                (0..len).any(|k| individuals[j].objectives[k] < individuals[i].objectives[k]);
            if all_leq_ji && any_lt_ji {
                // j dominates i
                dom_count[i] += 1;
            }
        }
        // Avoid double counting: the inner loop counted each j twice
    }
    // Re-derive dom_count correctly (above double-counts; recompute)
    let mut dom_count2 = vec![0usize; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let len = individuals[j]
                .objectives
                .len()
                .min(individuals[i].objectives.len());
            let all_leq =
                (0..len).all(|k| individuals[j].objectives[k] <= individuals[i].objectives[k]);
            let any_lt =
                (0..len).any(|k| individuals[j].objectives[k] < individuals[i].objectives[k]);
            if all_leq && any_lt {
                dom_count2[i] += 1;
            }
        }
    }

    for i in 0..n {
        if dom_count2[i] == 0 {
            individuals[i].rank = 0;
            fronts[0].push(i);
        }
    }

    let mut current_front = 0usize;
    while !fronts[current_front].is_empty() {
        let mut next_front = Vec::new();
        for &i in &fronts[current_front] {
            for &j in &dominated_by[i] {
                dom_count2[j] = dom_count2[j].saturating_sub(1);
                if dom_count2[j] == 0 {
                    individuals[j].rank = current_front + 1;
                    next_front.push(j);
                }
            }
        }
        current_front += 1;
        fronts.push(next_front);
    }
}

// ─── crowding_distance ────────────────────────────────────────────────────────

/// Compute crowding distance for a set of individuals in the same Pareto front.
///
/// Boundary individuals receive `f32::INFINITY`.
pub fn crowding_distance(front_indices: &[usize], individuals: &mut [Individual]) {
    let l = front_indices.len();
    if l == 0 {
        return;
    }
    for &i in front_indices {
        individuals[i].crowding_distance = 0.0;
    }
    if l == 1 {
        individuals[front_indices[0]].crowding_distance = f32::INFINITY;
        return;
    }

    let n_obj = individuals[front_indices[0]].objectives.len();
    for m in 0..n_obj {
        // Sort front by objective m
        let mut sorted = front_indices.to_vec();
        sorted.sort_by(|&a, &b| {
            individuals[a].objectives[m]
                .partial_cmp(&individuals[b].objectives[m])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Boundary individuals get infinity
        individuals[sorted[0]].crowding_distance = f32::INFINITY;
        individuals[sorted[l - 1]].crowding_distance = f32::INFINITY;

        let obj_min = individuals[sorted[0]].objectives[m];
        let obj_max = individuals[sorted[l - 1]].objectives[m];
        let range = obj_max - obj_min;
        if range < 1e-10 {
            continue;
        }

        for k in 1..l - 1 {
            let prev = individuals[sorted[k - 1]].objectives[m];
            let next = individuals[sorted[k + 1]].objectives[m];
            if individuals[sorted[k]].crowding_distance < f32::INFINITY {
                individuals[sorted[k]].crowding_distance += (next - prev) / range;
            }
        }
    }
}

// ─── tournament_select ────────────────────────────────────────────────────────

/// Binary tournament selection: compare by (rank ASC, crowding_distance DESC).
///
/// Draws `tournament_size` candidates at random and returns the best.
pub fn tournament_select<'a>(
    individuals: &'a [Individual],
    tournament_size: usize,
    rng: &mut LcgRng,
) -> NasResult<&'a Individual> {
    let n = individuals.len();
    if n == 0 {
        return Err(NasError::PopulationTooSmall { min: 1, got: 0 });
    }
    if tournament_size == 0 || tournament_size > n {
        return Err(NasError::InvalidTournamentSize);
    }

    let mut best_idx = rng.next_usize(n);
    for _ in 1..tournament_size {
        let challenger = rng.next_usize(n);
        let bi = &individuals[best_idx];
        let ci = &individuals[challenger];
        let better = bi.rank > ci.rank
            || (bi.rank == ci.rank && bi.crowding_distance < ci.crowding_distance);
        if better {
            best_idx = challenger;
        }
    }
    Ok(&individuals[best_idx])
}

// ─── nsga2_select ─────────────────────────────────────────────────────────────

/// Full NSGA-II selection: keep the top `n_select` individuals from a combined pool.
///
/// 1. Run fast non-dominated sort on `combined`.
/// 2. Fill `selected` by adding full fronts until the next front would overflow.
/// 3. Fill remaining slots from the last front by crowding distance (descending).
pub fn nsga2_select(mut combined: Vec<Individual>, n_select: usize) -> NasResult<Vec<Individual>> {
    if combined.len() < n_select {
        return Err(NasError::PopulationTooSmall {
            min: n_select,
            got: combined.len(),
        });
    }

    fast_non_dominated_sort(&mut combined);

    // Group by rank
    let max_rank = combined.iter().map(|i| i.rank).max().unwrap_or(0);
    let mut fronts: Vec<Vec<usize>> = vec![Vec::new(); max_rank + 1];
    for (idx, ind) in combined.iter().enumerate() {
        fronts[ind.rank].push(idx);
    }

    // Compute crowding distances per front
    for front in &fronts {
        crowding_distance(front, &mut combined);
    }

    let mut selected_indices: Vec<usize> = Vec::with_capacity(n_select);
    for front in &fronts {
        if selected_indices.len() + front.len() <= n_select {
            selected_indices.extend_from_slice(front);
        } else {
            // Sort this front by crowding distance DESC and take what we need
            let remaining = n_select - selected_indices.len();
            let mut partial = front.clone();
            partial.sort_by(|&a, &b| {
                combined[b]
                    .crowding_distance
                    .partial_cmp(&combined[a].crowding_distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            selected_indices.extend(partial.into_iter().take(remaining));
            break;
        }
        if selected_indices.len() >= n_select {
            break;
        }
    }

    // Extract selected individuals
    let mut selected: Vec<Individual> = selected_indices
        .into_iter()
        .filter_map(|i| {
            if i < combined.len() {
                Some(combined[i].clone())
            } else {
                None
            }
        })
        .collect();

    // Ensure exact count
    selected.truncate(n_select);
    while selected.len() < n_select {
        if let Some(ind) = combined.first() {
            selected.push(ind.clone());
        } else {
            break;
        }
    }

    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_individual(objectives: Vec<f32>) -> Individual {
        Individual {
            encoding: vec![0],
            objectives,
            rank: 0,
            crowding_distance: 0.0,
        }
    }

    #[test]
    fn pareto_dominance_correct() {
        let a = make_individual(vec![1.0, 1.0]);
        let b = make_individual(vec![2.0, 2.0]);
        assert!(a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    #[test]
    fn non_dominated_sort_ranks_pareto_front() {
        let mut individuals = vec![
            make_individual(vec![1.0, 4.0]),
            make_individual(vec![2.0, 3.0]),
            make_individual(vec![3.0, 2.0]),
            make_individual(vec![2.5, 2.5]), // dominated by nothing
        ];
        fast_non_dominated_sort(&mut individuals);
        // All are non-dominated → all rank 0
        assert!(individuals.iter().all(|i| i.rank == 0));
    }

    #[test]
    fn dominated_individual_gets_higher_rank() {
        let mut individuals = vec![
            make_individual(vec![1.0, 1.0]), // dominates the next
            make_individual(vec![2.0, 2.0]), // dominated by first
        ];
        fast_non_dominated_sort(&mut individuals);
        assert_eq!(individuals[0].rank, 0);
        assert_eq!(individuals[1].rank, 1);
    }

    #[test]
    fn crowding_distance_boundary_is_inf() {
        let mut individuals = vec![
            make_individual(vec![1.0, 4.0]),
            make_individual(vec![2.0, 2.0]),
            make_individual(vec![4.0, 1.0]),
        ];
        let indices = vec![0, 1, 2];
        crowding_distance(&indices, &mut individuals);
        assert_eq!(individuals[0].crowding_distance, f32::INFINITY);
        assert_eq!(individuals[2].crowding_distance, f32::INFINITY);
    }

    #[test]
    fn nsga2_select_returns_correct_count() {
        let individuals: Vec<Individual> = (0..20)
            .map(|i| make_individual(vec![i as f32, (20 - i) as f32]))
            .collect();
        let selected = nsga2_select(individuals, 10).expect("test invariant: nsga2 select");
        assert_eq!(selected.len(), 10);
    }
}
