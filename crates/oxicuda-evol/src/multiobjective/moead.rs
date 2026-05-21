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

/// Maximum recursion depth for the simplex-lattice enumeration.
///
/// One recursion level is spent per objective, so this also bounds `n_obj`.
/// A 32-objective problem is already far beyond any realistic MOEA/D use.
const MAX_SIMPLEX_RECURSION: usize = 32;

/// Number of objectives at or above which the two-layer (boundary + inside)
/// Deb–Jain design replaces the single-layer Das–Dennis lattice. Past this
/// point a single layer either overshoots `pop_size` at `H = 1` or needs an
/// impractically large `H` to provide interior coverage.
const TWO_LAYER_THRESHOLD: usize = 6;

/// Compute the binomial coefficient `C(n, k)` with saturating arithmetic.
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    if k == 0 {
        return 1;
    }
    let mut result = 1usize;
    for i in 0..k {
        result = result.saturating_mul(n - i);
        result /= i + 1;
    }
    result
}

/// Number of Das–Dennis lattice points for `n_obj` objectives and `h` divisions:
/// `C(h + n_obj - 1, n_obj - 1)`.
fn das_dennis_count(n_obj: usize, h: usize) -> usize {
    if n_obj == 0 {
        return 0;
    }
    binomial(h + n_obj - 1, n_obj - 1)
}

/// Pick the number of divisions `h >= 1` whose lattice size `C(h+n_obj-1,n_obj-1)`
/// is the smallest value `>= target` (so the caller can truncate down to
/// `target`). If even `h = 1` already overshoots, `h = 1` is returned.
fn choose_h(n_obj: usize, target: usize) -> usize {
    if n_obj < 2 || target <= 1 {
        return 1;
    }
    let mut h = 1usize;
    loop {
        if das_dennis_count(n_obj, h) >= target {
            return h;
        }
        h += 1;
        // Safety cap: H is never realistically this large for sane pop sizes.
        if h > 1000 {
            return h;
        }
    }
}

/// Recursively enumerate every non-negative integer composition of `remaining`
/// into the coordinates `[dim, n_obj)`, appending each completed integer tuple
/// to `out`. Divide by `h` afterwards to land on the unit simplex.
///
/// `depth` carries the recursion level and is checked against
/// [`MAX_SIMPLEX_RECURSION`] so a pathological `n_obj` cannot blow the stack.
fn enumerate_compositions(
    n_obj: usize,
    dim: usize,
    depth: usize,
    remaining: usize,
    current: &mut [usize],
    out: &mut Vec<Vec<usize>>,
) {
    if depth > MAX_SIMPLEX_RECURSION {
        return;
    }
    if dim + 1 == n_obj {
        // Last coordinate is fully determined: it absorbs the remainder.
        current[dim] = remaining;
        out.push(current.to_vec());
        return;
    }
    for v in 0..=remaining {
        current[dim] = v;
        enumerate_compositions(n_obj, dim + 1, depth + 1, remaining - v, current, out);
    }
}

/// Build a single Das–Dennis simplex lattice: every weight vector has `n_obj`
/// coordinates that are non-negative multiples of `1/h` summing to exactly 1.
fn das_dennis_layer(n_obj: usize, h: usize) -> Vec<Vec<f64>> {
    if n_obj == 0 || h == 0 {
        return Vec::new();
    }
    let mut raw: Vec<Vec<usize>> = Vec::new();
    let mut current = vec![0usize; n_obj];
    enumerate_compositions(n_obj, 0, 0, h, &mut current, &mut raw);
    let inv_h = 1.0 / h as f64;
    raw.into_iter()
        .map(|tuple| normalise_simplex(tuple.iter().map(|&v| v as f64 * inv_h).collect()))
        .collect()
}

/// Renormalise a weight vector so its coordinates sum to exactly 1.0,
/// cancelling any floating-point drift accumulated during construction.
fn normalise_simplex(mut weights: Vec<f64>) -> Vec<f64> {
    let sum: f64 = weights.iter().sum();
    if sum > 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    } else if !weights.is_empty() {
        let uniform = 1.0 / weights.len() as f64;
        weights.fill(uniform);
    }
    weights
}

/// Two-layer (boundary + inside) Deb–Jain design for many-objective problems.
///
/// The *boundary* layer is a coarse Das–Dennis lattice on the simplex surface.
/// The *inside* layer is a second Das–Dennis lattice contracted toward the
/// centroid `(1/n_obj, …)` by `shrink ∈ (0,1)`, so its points populate the
/// simplex interior instead of duplicating the boundary. This keeps the total
/// count tractable when a single fine lattice would explode combinatorially.
///
/// Reference: K. Deb & H. Jain, "An Evolutionary Many-Objective Optimization
/// Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I",
/// IEEE Trans. Evol. Comput. 18(4):577-601, 2014.
fn two_layer_weights(n_obj: usize, pop_size: usize) -> Vec<Vec<f64>> {
    // Split the budget between the two layers; each layer targets roughly half
    // of `pop_size` so neither dominates after the final truncation.
    let outer_target = (pop_size / 2).max(1);
    let inner_target = pop_size.saturating_sub(outer_target).max(1);

    let h_outer = choose_h(n_obj, outer_target);
    let h_inner = choose_h(n_obj, inner_target);

    let boundary = das_dennis_layer(n_obj, h_outer);

    // Inside layer: a second lattice contracted toward the centroid so its
    // points fall strictly inside the simplex instead of on the boundary.
    let shrink = 0.5_f64;
    let centroid = 1.0 / n_obj as f64;
    let inside: Vec<Vec<f64>> = das_dennis_layer(n_obj, h_inner)
        .into_iter()
        .map(|w| {
            let contracted: Vec<f64> = w
                .iter()
                .map(|&v| centroid + shrink * (v - centroid))
                .collect();
            normalise_simplex(contracted)
        })
        .collect();

    // Interleave the two layers round-robin: a later truncation to `pop_size`
    // then keeps a balanced mix of boundary and interior reference points
    // rather than draining one layer entirely.
    let mut weights = Vec::with_capacity(boundary.len() + inside.len());
    let mut bi = boundary.into_iter();
    let mut ii = inside.into_iter();
    loop {
        let mut progressed = false;
        if let Some(w) = bi.next() {
            weights.push(w);
            progressed = true;
        }
        if let Some(w) = ii.next() {
            weights.push(w);
            progressed = true;
        }
        if !progressed {
            break;
        }
    }
    weights
}

/// Generate uniformly spread weight vectors over the unit simplex.
///
/// For `n_obj == 2` the vectors are evenly spaced on the line segment; for any
/// `n_obj >= 3` a **Das–Dennis simplex-lattice design** is used: every weight
/// vector's coordinates are non-negative multiples of `1/H` summing to 1, where
/// `H` is chosen so the lattice size `C(H+n_obj-1, n_obj-1)` is at least
/// `pop_size`. For many-objective problems (`n_obj >= 6`) a **two-layer
/// boundary + inside Deb–Jain design** keeps the point count tractable.
///
/// The result is always exactly `pop_size` vectors (the lattice is padded with
/// the centroid weight or truncated as needed) and every vector sums to 1.0.
pub fn generate_weight_vectors(pop_size: usize, n_obj: usize) -> Vec<Vec<f64>> {
    if pop_size == 0 || n_obj == 0 {
        return Vec::new();
    }
    if n_obj == 1 {
        return vec![vec![1.0]; pop_size];
    }

    let mut weights = if n_obj == 2 {
        (0..pop_size)
            .map(|i| {
                let t = i as f64 / (pop_size - 1).max(1) as f64;
                normalise_simplex(vec![t, 1.0 - t])
            })
            .collect::<Vec<_>>()
    } else if n_obj >= TWO_LAYER_THRESHOLD || das_dennis_count(n_obj, 1) >= pop_size {
        // Many-objective, or a single layer already overshoots at H = 1:
        // use the two-layer boundary + inside design.
        two_layer_weights(n_obj, pop_size)
    } else {
        // Single-layer Das–Dennis lattice sized to cover `pop_size`.
        let h = choose_h(n_obj, pop_size);
        das_dennis_layer(n_obj, h)
    };

    // Match the n_obj==2/3 behaviour: pad with the centroid, then truncate.
    while weights.len() < pop_size {
        weights.push(vec![1.0 / n_obj as f64; n_obj]);
    }
    weights.truncate(pop_size);
    weights
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every weight vector must have `n_obj` coordinates summing to 1.0.
    fn assert_valid_simplex(weights: &[Vec<f64>], n_obj: usize) {
        for (idx, w) in weights.iter().enumerate() {
            assert_eq!(w.len(), n_obj, "vector {idx} has wrong dimensionality");
            let sum: f64 = w.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "vector {idx} sums to {sum}, expected 1.0"
            );
            for &c in w {
                assert!(
                    (-1e-12..=1.0 + 1e-12).contains(&c),
                    "vector {idx} has out-of-range coordinate {c}"
                );
            }
        }
    }

    /// Count weight vectors that are pairwise distinct (rounded to 1e-9).
    fn distinct_count(weights: &[Vec<f64>]) -> usize {
        let mut keys: Vec<Vec<i64>> = weights
            .iter()
            .map(|w| w.iter().map(|&v| (v * 1e9).round() as i64).collect())
            .collect();
        keys.sort();
        keys.dedup();
        keys.len()
    }

    #[test]
    fn binomial_basic_values() {
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(10, 3), 120);
        assert_eq!(binomial(3, 7), 0);
    }

    #[test]
    fn das_dennis_count_matches_layer_size() {
        // The closed-form count must equal the enumerated lattice size.
        for n_obj in 2..=6 {
            for h in 1..=5 {
                let layer = das_dennis_layer(n_obj, h);
                assert_eq!(
                    layer.len(),
                    das_dennis_count(n_obj, h),
                    "count mismatch n_obj={n_obj} h={h}"
                );
            }
        }
    }

    #[test]
    fn das_dennis_layer_lies_on_simplex() {
        // Each enumerated lattice point sums to 1 and is non-negative.
        let layer = das_dennis_layer(4, 4);
        assert_valid_simplex(&layer, 4);
        assert_eq!(layer.len(), das_dennis_count(4, 4));
    }

    #[test]
    fn two_objective_weights_are_evenly_spaced() {
        let w = generate_weight_vectors(11, 2);
        assert_eq!(w.len(), 11);
        assert_valid_simplex(&w, 2);
        assert!((w[0][0] - 0.0).abs() < 1e-12);
        assert!((w[10][0] - 1.0).abs() < 1e-12);
        assert!((w[5][0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn three_objective_weights_are_a_real_lattice() {
        let w = generate_weight_vectors(91, 3);
        assert_eq!(w.len(), 91);
        assert_valid_simplex(&w, 3);
        // A genuine lattice must contain many distinct vectors, not clones.
        assert!(distinct_count(&w) >= 91, "3-objective lattice collapsed");
    }

    #[test]
    fn high_dim_weights_are_distinct_and_normalised() {
        // Regression: the old catch-all returned identical 1/n_obj vectors for
        // every subproblem, collapsing MOEA/D decomposition. The Das–Dennis /
        // two-layer design must yield `pop_size` *distinct* simplex vectors.
        for &n_obj in &[4usize, 5, 8] {
            for &pop_size in &[50usize, 91, 120] {
                let w = generate_weight_vectors(pop_size, n_obj);
                assert_eq!(
                    w.len(),
                    pop_size,
                    "n_obj={n_obj} pop_size={pop_size}: wrong count"
                );
                assert_valid_simplex(&w, n_obj);
                assert_eq!(
                    distinct_count(&w),
                    pop_size,
                    "n_obj={n_obj} pop_size={pop_size}: weight vectors not all distinct"
                );
                // Non-degenerate: not every vector equals the uniform centroid.
                let centroid = 1.0 / n_obj as f64;
                let all_uniform = w
                    .iter()
                    .all(|v| v.iter().all(|&c| (c - centroid).abs() < 1e-9));
                assert!(
                    !all_uniform,
                    "n_obj={n_obj} pop_size={pop_size}: lattice degenerated to centroid"
                );
            }
        }
    }

    #[test]
    fn two_layer_design_used_for_many_objectives() {
        // For n_obj >= 6 the two-layer design supplies boundary + interior
        // points; interior points must have all coordinates strictly inside
        // (0,1) — a property a pure boundary lattice cannot guarantee.
        let w = generate_weight_vectors(120, 8);
        assert_valid_simplex(&w, 8);
        let has_interior = w
            .iter()
            .any(|v| v.iter().all(|&c| c > 1e-6 && c < 1.0 - 1e-6));
        assert!(has_interior, "two-layer design produced no interior points");
    }

    #[test]
    fn recursion_depth_guard_caps_objectives() {
        // n_obj beyond MAX_SIMPLEX_RECURSION must not recurse unboundedly;
        // the guard simply yields no enumerated points for that layer.
        let raw = das_dennis_layer(MAX_SIMPLEX_RECURSION + 4, 2);
        assert!(
            raw.is_empty(),
            "depth guard failed to stop deep enumeration"
        );
    }

    #[test]
    fn weight_vectors_written_to_temp_file_roundtrip() {
        // Exercise file I/O via std::env::temp_dir() per workspace test policy.
        use std::io::{Read, Write};
        let w = generate_weight_vectors(60, 5);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxicuda_evol_moead_weights_{}.txt",
            std::process::id()
        ));
        {
            let mut f = std::fs::File::create(&path).expect("create temp file");
            for v in &w {
                let line: Vec<String> = v.iter().map(|c| format!("{c:.12}")).collect();
                writeln!(f, "{}", line.join(" ")).expect("write weights");
            }
        }
        let mut contents = String::new();
        std::fs::File::open(&path)
            .expect("open temp file")
            .read_to_string(&mut contents)
            .expect("read weights");
        let _ = std::fs::remove_file(&path);

        let parsed: Vec<Vec<f64>> = contents
            .lines()
            .map(|l| {
                l.split_whitespace()
                    .map(|t| t.parse::<f64>().unwrap_or(0.0))
                    .collect()
            })
            .collect();
        assert_eq!(parsed.len(), 60);
        assert_valid_simplex(&parsed, 5);
    }
}
