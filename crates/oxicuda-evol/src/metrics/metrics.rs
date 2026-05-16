//! Multi-objective quality metrics: hypervolume (2D), IGD, GD, spacing, Pareto extraction.

use crate::{EvolError, EvolResult};

/// Compute the 2D hypervolume indicator for a front of (f1, f2) pairs.
///
/// Returns the area dominated by the front but not by the reference point.
/// Solutions are sorted by f1 ascending; the dominated area is swept.
///
/// # Errors
/// Returns `EvolError::EmptyPopulation` if the front is empty.
pub fn hypervolume_2d(front: &[(f64, f64)], reference: (f64, f64)) -> EvolResult<f64> {
    if front.is_empty() {
        return Err(EvolError::EmptyPopulation);
    }

    // Filter solutions that are actually dominated by reference
    let mut valid: Vec<(f64, f64)> = front
        .iter()
        .filter(|&&(f1, f2)| f1 < reference.0 && f2 < reference.1)
        .copied()
        .collect();

    if valid.is_empty() {
        return Ok(0.0);
    }

    // Sort by f1 ascending
    valid.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut area = 0.0;
    let mut prev_f2 = reference.1;

    for &(f1, f2) in &valid {
        // Width in f1 direction: reference.f1 - f1 (note: we're going left to right, so we
        // accumulate differently — each point contributes a rectangle to the right)
        if prev_f2 > f2 {
            area += (reference.0 - f1) * (prev_f2 - f2);
            prev_f2 = f2;
        }
    }

    Ok(area)
}

/// Euclidean distance between two objective vectors.
fn euclid_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Inverted Generational Distance (IGD).
///
/// For each reference front point, find the nearest approximation point.
/// IGD = mean of those nearest distances.
///
/// # Errors
/// Returns an error if either set is empty or has inconsistent dimension.
pub fn igd(approximation: &[Vec<f64>], reference_front: &[Vec<f64>]) -> EvolResult<f64> {
    if approximation.is_empty() {
        return Err(EvolError::EmptyPopulation);
    }
    if reference_front.is_empty() {
        return Err(EvolError::EmptyPopulation);
    }

    let total: f64 = reference_front
        .iter()
        .map(|r| {
            approximation
                .iter()
                .map(|a| euclid_dist(a, r))
                .fold(f64::INFINITY, f64::min)
        })
        .sum();
    Ok(total / reference_front.len() as f64)
}

/// Generational Distance (GD).
///
/// For each approximation point, find the nearest reference front point.
/// GD = mean of those nearest distances.
pub fn generational_distance(
    approximation: &[Vec<f64>],
    reference_front: &[Vec<f64>],
) -> EvolResult<f64> {
    if approximation.is_empty() {
        return Err(EvolError::EmptyPopulation);
    }
    if reference_front.is_empty() {
        return Err(EvolError::EmptyPopulation);
    }

    let total: f64 = approximation
        .iter()
        .map(|a| {
            reference_front
                .iter()
                .map(|r| euclid_dist(a, r))
                .fold(f64::INFINITY, f64::min)
        })
        .sum();
    Ok(total / approximation.len() as f64)
}

/// Spacing metric: standard deviation of nearest-neighbour distances on the front.
///
/// A smaller value indicates a more uniformly spaced front.
pub fn spacing(front: &[Vec<f64>]) -> EvolResult<f64> {
    if front.is_empty() {
        return Err(EvolError::EmptyPopulation);
    }
    if front.len() == 1 {
        return Ok(0.0);
    }
    let nn_dists: Vec<f64> = front
        .iter()
        .map(|a| {
            front
                .iter()
                .filter(|b| !std::ptr::eq(a.as_ptr(), b.as_ptr()))
                .map(|b| euclid_dist(a, b))
                .fold(f64::INFINITY, f64::min)
        })
        .collect();

    let mean = nn_dists.iter().sum::<f64>() / nn_dists.len() as f64;
    let variance = nn_dists.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / nn_dists.len() as f64;
    Ok(variance.sqrt())
}

/// Average nearest-neighbour distance: a measure of raw front diversity.
pub fn average_nn_distance(front: &[Vec<f64>]) -> EvolResult<f64> {
    if front.is_empty() {
        return Err(EvolError::EmptyPopulation);
    }
    if front.len() == 1 {
        return Ok(0.0);
    }
    let total: f64 = front
        .iter()
        .map(|a| {
            front
                .iter()
                .filter(|b| !std::ptr::eq(a.as_ptr(), b.as_ptr()))
                .map(|b| euclid_dist(a, b))
                .fold(f64::INFINITY, f64::min)
        })
        .sum();
    Ok(total / front.len() as f64)
}

/// Extract Pareto-optimal (non-dominated) solution indices from a set of objective vectors.
///
/// A solution is Pareto-optimal if no other solution in the set dominates it.
/// Domination: all objectives ≤ AND at least one strictly <.
pub fn extract_pareto_front(objectives: &[Vec<f64>]) -> Vec<usize> {
    let n = objectives.len();
    let mut is_dominated = vec![false; n];

    for i in 0..n {
        if is_dominated[i] {
            continue;
        }
        for j in 0..n {
            if i == j || is_dominated[j] {
                continue;
            }
            // Check if j dominates i
            let j_dom_i = objectives[j]
                .iter()
                .zip(objectives[i].iter())
                .all(|(a, b)| a <= b)
                && objectives[j]
                    .iter()
                    .zip(objectives[i].iter())
                    .any(|(a, b)| a < b);
            if j_dom_i {
                is_dominated[i] = true;
                break;
            }
        }
    }

    (0..n).filter(|&i| !is_dominated[i]).collect()
}
