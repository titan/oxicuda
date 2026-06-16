//! NSGA-III: Non-dominated Sorting Genetic Algorithm III with structured reference points.
//!
//! Reference: K. Deb & H. Jain, "An Evolutionary Many-Objective Optimization Algorithm Using
//! Reference-Point-Based Non-dominated Sorting Approach, Part I", IEEE Trans. Evol. Comput.
//! 18(4):577-601, 2014.

#![allow(clippy::needless_range_loop)]

use crate::{EvolError, EvolResult, handle::LcgRng};

/// Hyper-parameters for an NSGA-III run.
#[derive(Debug, Clone)]
pub struct Nsga3Config {
    /// Number of objectives.
    pub n_obj: usize,
    /// Population size (should be approximately equal to the number of reference points).
    pub n_pop: usize,
    /// Number of generations.
    pub n_gen: usize,
    /// SBX distribution index for crossover.
    pub crossover_eta: f64,
    /// Polynomial mutation distribution index.
    pub mutation_eta: f64,
    /// Random seed.
    pub seed: u64,
}

/// Mutable NSGA-III state returned at the end of a run.
#[derive(Debug, Clone)]
pub struct Nsga3State {
    /// Decision variable vectors for all individuals in the final population.
    pub population: Vec<Vec<f64>>,
    /// Objective vectors corresponding to each individual.
    pub objectives: Vec<Vec<f64>>,
    /// Reference points on the unit simplex.
    pub ref_points: Vec<Vec<f64>>,
    /// Current generation counter.
    pub generation: usize,
}

/// Generate Das-Dennis structured reference points on the unit simplex.
///
/// Creates all lattice points with exactly `h` divisions per axis such that
/// the coordinates sum to 1.0. Uses recursive enumeration.
pub fn generate_reference_points(n_obj: usize, h: usize) -> Vec<Vec<f64>> {
    let mut points = Vec::new();
    let mut current = vec![0usize; n_obj];
    enumerate_lattice(n_obj, h, 0, h, &mut current, &mut points);
    points
        .into_iter()
        .map(|p| p.iter().map(|&v| v as f64 / h as f64).collect())
        .collect()
}

/// Recursive helper to enumerate all integer lattice points summing to `remaining`
/// for indices `[dim, n_obj)`.
fn enumerate_lattice(
    n_obj: usize,
    _h: usize,
    dim: usize,
    remaining: usize,
    current: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    if dim == n_obj - 1 {
        current[dim] = remaining;
        out.push(current.clone());
        return;
    }
    for v in 0..=remaining {
        current[dim] = v;
        enumerate_lattice(n_obj, _h, dim + 1, remaining - v, current, out);
    }
}

/// Compute the smallest H (number of divisions) such that the number of Das-Dennis
/// reference points C(n_obj + H - 1, H) is at least `n_pop`.
fn compute_h_for_pop(n_obj: usize, n_pop: usize) -> usize {
    let mut h = 1usize;
    loop {
        let count = n_ref_points(n_obj, h);
        if count >= n_pop {
            return h;
        }
        h += 1;
        // Safety cap
        if h > 200 {
            return h;
        }
    }
}

/// Compute C(n_obj + h - 1, h) = number of Das-Dennis reference points.
fn n_ref_points(n_obj: usize, h: usize) -> usize {
    // C(n_obj + h - 1, h) using the multiplicative formula
    let n = n_obj + h - 1;
    let k = h.min(n_obj - 1);
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

/// Fast non-dominated sort returning fronts as lists of indices.
///
/// front[0] = Pareto front indices, front[1] = second front, etc.
fn fast_non_dominated_sort(objectives: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let n = objectives.len();
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut domination_count: Vec<usize> = vec![0; n];
    let mut fronts: Vec<Vec<usize>> = vec![Vec::new()];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let ij_dom = dominates(&objectives[i], &objectives[j]);
            let ji_dom = dominates(&objectives[j], &objectives[i]);
            if ij_dom {
                dominated_by[i].push(j);
            } else if ji_dom {
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

/// Returns true if `a` dominates `b` (a ≤ b in all objectives, a < b in at least one).
fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut at_least_one_less = false;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            return false;
        }
        if ai < bi {
            at_least_one_less = true;
        }
    }
    at_least_one_less
}

/// Normalize objectives using the ideal point and nadir point approximation.
///
/// Translates each objective by subtracting the ideal point, then scales using
/// the nadir (worst) values per objective so the Pareto front lives in [0,1]^m.
/// Returns the normalized objectives and the ideal point used.
fn normalize_objectives(
    objectives: &[Vec<f64>],
    n_obj: usize,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    if objectives.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    // Ideal point: minimum per objective
    let mut ideal = vec![f64::INFINITY; n_obj];
    for obj in objectives {
        for (m, &v) in obj.iter().enumerate() {
            if v < ideal[m] {
                ideal[m] = v;
            }
        }
    }

    // Translate
    let translated: Vec<Vec<f64>> = objectives
        .iter()
        .map(|obj| obj.iter().enumerate().map(|(m, &v)| v - ideal[m]).collect())
        .collect();

    // Nadir approximation using hyperplane intercepts through extreme points per objective
    // For each objective m, find the individual with minimum normalized value; build
    // the hyperplane from those extreme points (Deb & Jain 2014 §III-B).
    // Simplified: use the max of translated per objective as nadir.
    let mut nadir = vec![0.0f64; n_obj];
    for obj in &translated {
        for (m, &v) in obj.iter().enumerate() {
            if v > nadir[m] {
                nadir[m] = v;
            }
        }
    }
    // Clamp to avoid division by zero
    for v in &mut nadir {
        if *v < 1e-12 {
            *v = 1.0;
        }
    }

    let normalized: Vec<Vec<f64>> = translated
        .into_iter()
        .map(|obj| {
            obj.into_iter()
                .enumerate()
                .map(|(m, v)| v / nadir[m])
                .collect()
        })
        .collect();

    (normalized, ideal, nadir)
}

/// Compute the perpendicular (Euclidean) distance from point `p` to reference line `r`.
///
/// The reference line goes from the origin to `r`. The distance is:
///   d = ||p - (p·r/r·r) * r||
fn point_to_line_distance(p: &[f64], r: &[f64]) -> f64 {
    let dot_pr: f64 = p.iter().zip(r.iter()).map(|(&pi, &ri)| pi * ri).sum();
    let dot_rr: f64 = r.iter().map(|&ri| ri * ri).sum::<f64>().max(1e-300);
    let scale = dot_pr / dot_rr;
    // projection: proj_i = scale * r_i
    // distance^2 = sum (p_i - scale * r_i)^2
    p.iter()
        .zip(r.iter())
        .map(|(&pi, &ri)| (pi - scale * ri).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// For each individual in `indices`, find the nearest reference point and compute distance.
/// Returns Vec of (ref_point_index, distance) for each individual.
fn associate_to_ref_points(
    normalized_objs: &[Vec<f64>],
    ref_points: &[Vec<f64>],
    indices: &[usize],
) -> Vec<(usize, f64)> {
    indices
        .iter()
        .map(|&idx| {
            let obj = &normalized_objs[idx];
            let (best_rp, best_dist) = ref_points
                .iter()
                .enumerate()
                .map(|(rp_idx, rp)| (rp_idx, point_to_line_distance(obj, rp)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, f64::INFINITY));
            (best_rp, best_dist)
        })
        .collect()
}

/// Simulated Binary Crossover (SBX) for NSGA-III (per-gene, per-individual bounds).
fn sbx_crossover(
    p1: &[f64],
    p2: &[f64],
    eta: f64,
    bounds: &[(f64, f64)],
    rng: &mut LcgRng,
) -> (Vec<f64>, Vec<f64>) {
    let n = p1.len();
    let mut c1 = Vec::with_capacity(n);
    let mut c2 = Vec::with_capacity(n);

    for i in 0..n {
        let (lb, ub) = bounds[i];
        if rng.next_f64() <= 0.5 {
            let x1 = p1[i].clamp(lb, ub);
            let x2 = p2[i].clamp(lb, ub);
            let (lo, hi) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
            let diff = (hi - lo).max(1e-14);
            let u = rng.next_f64();
            let beta_q = if u <= 0.5 {
                let beta = 1.0 + 2.0 * (lo - lb) / diff;
                let alpha = 2.0 - beta.powf(-(eta + 1.0));
                let u2 = u * 2.0;
                if u2 <= 1.0 / alpha {
                    (u2 * alpha).powf(1.0 / (eta + 1.0))
                } else {
                    (1.0 / (2.0 - u2 * alpha)).powf(1.0 / (eta + 1.0))
                }
            } else {
                let beta = 1.0 + 2.0 * (ub - hi) / diff;
                let alpha = 2.0 - beta.powf(-(eta + 1.0));
                let u2 = 2.0 * (1.0 - u);
                if u2 <= 1.0 / alpha {
                    (u2 * alpha).powf(1.0 / (eta + 1.0))
                } else {
                    (1.0 / (2.0 - u2 * alpha)).powf(1.0 / (eta + 1.0))
                }
            };
            let o1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1));
            let o2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1));
            c1.push(o1.clamp(lb, ub));
            c2.push(o2.clamp(lb, ub));
        } else {
            c1.push(p1[i]);
            c2.push(p2[i]);
        }
    }
    (c1, c2)
}

/// Polynomial mutation for NSGA-III (per-gene bounds).
fn polynomial_mutation(
    genome: &mut [f64],
    eta: f64,
    prob: f64,
    bounds: &[(f64, f64)],
    rng: &mut LcgRng,
) {
    for (i, gene) in genome.iter_mut().enumerate() {
        if rng.next_f64() < prob {
            let (lb, ub) = bounds[i];
            let range = (ub - lb).max(1e-14);
            let delta1 = (*gene - lb) / range;
            let delta2 = (ub - *gene) / range;
            let u = rng.next_f64();
            let delta_q = if u <= 0.5 {
                let val = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta1).powf(eta + 1.0);
                val.powf(1.0 / (eta + 1.0)) - 1.0
            } else {
                let val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta2).powf(eta + 1.0);
                1.0 - val.powf(1.0 / (eta + 1.0))
            };
            *gene = (*gene + delta_q * range).clamp(lb, ub);
        }
    }
}

/// NSGA-III niche-preservation selection.
///
/// Given combined population of 2N, selects exactly `n_select` individuals using
/// non-dominated sorting + reference-point niche-preservation for the critical front.
pub fn nsga3_selection(
    population: &[Vec<f64>],
    objectives: &[Vec<f64>],
    ref_points: &[Vec<f64>],
    n_select: usize,
    n_obj: usize,
) -> Vec<usize> {
    let n = population.len();
    if n <= n_select {
        return (0..n).collect();
    }

    // Non-dominated sorting
    let fronts = fast_non_dominated_sort(objectives);

    // Fill next population front by front until critical front
    let mut selected: Vec<usize> = Vec::with_capacity(n_select);
    let mut critical_front: Vec<usize> = Vec::new();

    'fill: for front in &fronts {
        if selected.len() + front.len() <= n_select {
            selected.extend_from_slice(front);
        } else {
            // This is the critical front — we need niche-preservation here
            critical_front = front.clone();
            break 'fill;
        }
    }

    let need = n_select - selected.len();
    if need == 0 || critical_front.is_empty() {
        selected.truncate(n_select);
        return selected;
    }

    // Normalize objectives for the combined set of selected + critical_front
    let all_indices: Vec<usize> = selected
        .iter()
        .chain(critical_front.iter())
        .copied()
        .collect();
    let all_objs: Vec<Vec<f64>> = all_indices.iter().map(|&i| objectives[i].clone()).collect();
    let (norm_objs_all, _, _) = normalize_objectives(&all_objs, n_obj);

    // Build a map from global index -> normalized objective
    let mut norm_map: std::collections::HashMap<usize, Vec<f64>> = all_indices
        .iter()
        .zip(norm_objs_all.iter())
        .map(|(&idx, norm)| (idx, norm.clone()))
        .collect();

    // Build a flat normalized-objectives slice for selected individuals
    let sel_norm_objs: Vec<Vec<f64>> = selected
        .iter()
        .map(|&idx| norm_map.get(&idx).cloned().unwrap_or_default())
        .collect();
    let sel_indices_local: Vec<usize> = (0..selected.len()).collect();

    // Count niche counts for already-selected individuals using the helper
    let mut niche_count: Vec<usize> = vec![0; ref_points.len()];
    for (rp, _dist) in associate_to_ref_points(&sel_norm_objs, ref_points, &sel_indices_local) {
        niche_count[rp] += 1;
    }

    // For each critical front member, compute their nearest ref point and distance
    let critical_norm_objs: Vec<Vec<f64>> = critical_front
        .iter()
        .map(|&idx| norm_map.remove(&idx).unwrap_or_default())
        .collect();
    let crit_indices_local: Vec<usize> = (0..critical_front.len()).collect();
    let critical_assoc: Vec<(usize, f64)> =
        associate_to_ref_points(&critical_norm_objs, ref_points, &crit_indices_local);

    // Niche-preservation: iteratively select from critical front
    // Track which critical front members are still available
    let mut available: Vec<bool> = vec![true; critical_front.len()];
    let mut added = 0;

    while added < need {
        // Find the reference point with minimum niche count among those that have
        // at least one available critical front member associated with them
        let min_nc = (0..ref_points.len())
            .filter(|&rp| {
                critical_assoc
                    .iter()
                    .enumerate()
                    .any(|(ci, (cp_rp, _))| available[ci] && *cp_rp == rp)
            })
            .map(|rp| niche_count[rp])
            .min();

        let min_nc = match min_nc {
            Some(v) => v,
            None => break,
        };

        // Among ref points with that min niche count, collect candidates
        let candidate_rps: Vec<usize> = (0..ref_points.len())
            .filter(|&rp| {
                niche_count[rp] == min_nc
                    && critical_assoc
                        .iter()
                        .enumerate()
                        .any(|(ci, (cp_rp, _))| available[ci] && *cp_rp == rp)
            })
            .collect();

        if candidate_rps.is_empty() {
            break;
        }

        // Pick the first candidate ref point (deterministic tie-break)
        let chosen_rp = candidate_rps[0];

        // Among individuals associated to chosen_rp in critical front, pick smallest distance
        // (or random if niche count is 0 — here we always pick smallest distance)
        let best_ci = critical_assoc
            .iter()
            .enumerate()
            .filter(|(ci, (cp_rp, _))| available[*ci] && *cp_rp == chosen_rp)
            .min_by(|(_, (_, d1)), (_, (_, d2))| {
                d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(ci, _)| ci);

        match best_ci {
            Some(ci) => {
                selected.push(critical_front[ci]);
                available[ci] = false;
                niche_count[chosen_rp] += 1;
                added += 1;
            }
            None => break,
        }
    }

    selected
}

/// Run NSGA-III optimization.
///
/// Returns `(decision_vars, objectives)` for each individual in the final Pareto front.
pub fn nsga3_run<F>(
    fitness_fn: F,
    n_var: usize,
    bounds: &[(f64, f64)],
    cfg: &Nsga3Config,
) -> EvolResult<Vec<(Vec<f64>, Vec<f64>)>>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if cfg.n_obj < 2 {
        return Err(EvolError::InvalidParameter("n_obj must be >= 2".to_owned()));
    }
    if cfg.n_pop < 2 {
        return Err(EvolError::PopulationTooSmall {
            size: cfg.n_pop,
            op: "NSGA-III",
        });
    }
    if n_var == 0 {
        return Err(EvolError::InvalidParameter("n_var must be >= 1".to_owned()));
    }
    if bounds.len() != n_var {
        return Err(EvolError::DimensionMismatch {
            expected: n_var,
            got: bounds.len(),
        });
    }

    let mut rng = LcgRng::new(cfg.seed);

    // Generate reference points
    let h = compute_h_for_pop(cfg.n_obj, cfg.n_pop);
    let ref_points = generate_reference_points(cfg.n_obj, h);

    // Initialize population
    let mut population: Vec<Vec<f64>> = (0..cfg.n_pop)
        .map(|_| {
            (0..n_var)
                .map(|d| {
                    let (lb, ub) = bounds[d];
                    lb + rng.next_f64() * (ub - lb)
                })
                .collect()
        })
        .collect();

    let mut objectives: Vec<Vec<f64>> = population.iter().map(|x| fitness_fn(x)).collect();

    let n_pop = cfg.n_pop;
    let n_obj = cfg.n_obj;
    let p_mut = 1.0 / n_var as f64;

    for _gen in 0..cfg.n_gen {
        // Generate offspring via tournament + SBX + polynomial mutation
        let mut offspring_pop: Vec<Vec<f64>> = Vec::with_capacity(n_pop);
        let mut offspring_obj: Vec<Vec<f64>> = Vec::with_capacity(n_pop);

        while offspring_pop.len() < n_pop {
            // Tournament selection (rank-based among current population)
            let fronts = fast_non_dominated_sort(&objectives);
            let mut rank = vec![0usize; n_pop];
            for (r, front) in fronts.iter().enumerate() {
                for &i in front {
                    rank[i] = r;
                }
            }

            // Binary tournament by rank
            let p1 = {
                let a = rng.next_usize(n_pop);
                let b = rng.next_usize(n_pop);
                if rank[a] <= rank[b] { a } else { b }
            };
            let p2 = {
                let a = rng.next_usize(n_pop);
                let b = rng.next_usize(n_pop);
                if rank[a] <= rank[b] { a } else { b }
            };

            let (mut c1, mut c2) = sbx_crossover(
                &population[p1],
                &population[p2],
                cfg.crossover_eta,
                bounds,
                &mut rng,
            );
            polynomial_mutation(&mut c1, cfg.mutation_eta, p_mut, bounds, &mut rng);
            polynomial_mutation(&mut c2, cfg.mutation_eta, p_mut, bounds, &mut rng);

            let o1 = fitness_fn(&c1);
            let o2 = fitness_fn(&c2);

            if offspring_pop.len() < n_pop {
                offspring_pop.push(c1);
                offspring_obj.push(o1);
            }
            if offspring_pop.len() < n_pop {
                offspring_pop.push(c2);
                offspring_obj.push(o2);
            }
        }

        // Combine parent + offspring
        let mut combined_pop = population.clone();
        combined_pop.extend(offspring_pop);
        let mut combined_obj = objectives.clone();
        combined_obj.extend(offspring_obj);

        // Select n_pop survivors using NSGA-III reference-point mechanism
        let selected = nsga3_selection(&combined_pop, &combined_obj, &ref_points, n_pop, n_obj);

        population = selected.iter().map(|&i| combined_pop[i].clone()).collect();
        objectives = selected.iter().map(|&i| combined_obj[i].clone()).collect();
    }

    // Return final Pareto front (rank-0 individuals)
    let fronts = fast_non_dominated_sort(&objectives);
    let pareto_front = if fronts.is_empty() {
        vec![]
    } else {
        fronts[0].clone()
    };

    Ok(pareto_front
        .into_iter()
        .map(|i| (population[i].clone(), objectives[i].clone()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dtlz1_2obj(x: &[f64]) -> Vec<f64> {
        // Simplified 2-objective DTLZ1 (k=1): f1 = 0.5*x[0]*(1+g), f2 = 0.5*(1-x[0])*(1+g)
        // g = 100*(sum of (x[i]-0.5)^2 for i>=1)
        let g: f64 = x[1..].iter().map(|&xi| (xi - 0.5).powi(2) * 100.0).sum();
        let f1 = 0.5 * x[0] * (1.0 + g);
        let f2 = 0.5 * (1.0 - x[0]) * (1.0 + g);
        vec![f1, f2]
    }

    fn zdt1(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let f1 = x[0];
        let g: f64 = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (n - 1) as f64;
        let f2 = g * (1.0 - (f1 / g).sqrt());
        vec![f1, f2]
    }

    fn zdt2(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let f1 = x[0];
        let g: f64 = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (n - 1) as f64;
        let f2 = g * (1.0 - (f1 / g).powi(2));
        vec![f1, f2]
    }

    // ── Reference point generation tests ─────────────────────────────────────

    #[test]
    fn test_ref_points_2obj_h3() {
        let pts = generate_reference_points(2, 3);
        // C(2+3-1, 3) = C(4,3) = 4
        assert_eq!(pts.len(), 4);
        for pt in &pts {
            let sum: f64 = pt.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "sum = {sum}");
        }
    }

    #[test]
    fn test_ref_points_3obj_h4() {
        let pts = generate_reference_points(3, 4);
        // C(3+4-1, 4) = C(6,4) = 15
        assert_eq!(pts.len(), 15);
        for pt in &pts {
            let sum: f64 = pt.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "sum = {sum}");
            assert!(pt.iter().all(|&v| (0.0..=1.0).contains(&v)));
        }
    }

    #[test]
    fn test_ref_points_2obj_h1() {
        let pts = generate_reference_points(2, 1);
        assert_eq!(pts.len(), 2);
    }

    #[test]
    fn test_n_ref_points_formula() {
        assert_eq!(n_ref_points(2, 3), 4);
        assert_eq!(n_ref_points(3, 4), 15);
        assert_eq!(n_ref_points(2, 10), 11);
    }

    #[test]
    fn test_compute_h_for_pop() {
        // For n_obj=2, n_pop=10 → need H=9 (gives 10 points)
        let h = compute_h_for_pop(2, 10);
        assert!(n_ref_points(2, h) >= 10);
        // For n_obj=3, n_pop=15 → H=4 gives exactly 15
        let h3 = compute_h_for_pop(3, 15);
        assert!(n_ref_points(3, h3) >= 15);
    }

    // ── Non-dominated sort tests ──────────────────────────────────────────────

    #[test]
    fn test_fast_non_dominated_sort_basic() {
        let objs = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.5, 0.5],
            vec![1.0, 1.0],
        ];
        let fronts = fast_non_dominated_sort(&objs);
        // Pareto front: indices 0,1,2 (none dominate each other)
        // index 3 is dominated by all three
        assert!(fronts[0].contains(&0));
        assert!(fronts[0].contains(&1));
        assert!(fronts[0].contains(&2));
        assert!(fronts.len() >= 2);
        assert!(fronts[1].contains(&3));
    }

    #[test]
    fn test_dominates_fn() {
        assert!(dominates(&[0.0, 0.0], &[1.0, 1.0]));
        assert!(!dominates(&[1.0, 0.0], &[0.0, 1.0]));
        assert!(!dominates(&[1.0, 1.0], &[1.0, 1.0]));
    }

    // ── Normalization test ────────────────────────────────────────────────────

    #[test]
    fn test_normalize_objectives() {
        let objs = vec![
            vec![1.0, 4.0],
            vec![2.0, 3.0],
            vec![3.0, 2.0],
            vec![4.0, 1.0],
        ];
        let (norm, ideal, nadir) = normalize_objectives(&objs, 2);
        assert_eq!(norm.len(), 4);
        // ideal should be [1.0, 1.0]
        assert!((ideal[0] - 1.0).abs() < 1e-10);
        assert!((ideal[1] - 1.0).abs() < 1e-10);
        // nadir should be [3.0, 3.0]
        assert!((nadir[0] - 3.0).abs() < 1e-10);
        // All normalized values in [0,1]
        for n_obj in &norm {
            for &v in n_obj {
                assert!((-1e-10..=1.0 + 1e-10).contains(&v), "v={v}");
            }
        }
    }

    // ── Distance computation test ─────────────────────────────────────────────

    #[test]
    fn test_point_to_line_distance() {
        // Point on the line: distance should be 0
        let r = vec![1.0, 1.0];
        let p = vec![2.0, 2.0];
        let d = point_to_line_distance(&p, &r);
        assert!(d < 1e-10, "d={d}");

        // Point perpendicular: (1,0) to line [1,1]
        let p2 = vec![1.0, 0.0];
        let d2 = point_to_line_distance(&p2, &r);
        assert!(d2 > 0.0);
    }

    // ── Full run tests ────────────────────────────────────────────────────────

    #[test]
    fn test_nsga3_run_2obj_zdt1() {
        let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); 5];
        let cfg = Nsga3Config {
            n_obj: 2,
            n_pop: 20,
            n_gen: 20,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
            seed: 42,
        };
        let result = nsga3_run(zdt1, 5, &bounds, &cfg).expect("nsga3_run should succeed");
        // Should return some Pareto front members
        assert!(!result.is_empty());
        // Each result has correct dimensions
        for (dec, obj) in &result {
            assert_eq!(dec.len(), 5);
            assert_eq!(obj.len(), 2);
            assert!(obj[0] >= 0.0 && obj[1] >= 0.0);
        }
    }

    #[test]
    fn test_nsga3_run_2obj_zdt2() {
        let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); 4];
        let cfg = Nsga3Config {
            n_obj: 2,
            n_pop: 16,
            n_gen: 15,
            crossover_eta: 15.0,
            mutation_eta: 15.0,
            seed: 123,
        };
        let result = nsga3_run(zdt2, 4, &bounds, &cfg).expect("nsga3_run should succeed");
        assert!(!result.is_empty());
        for (dec, obj) in &result {
            assert_eq!(dec.len(), 4);
            assert_eq!(obj.len(), 2);
        }
    }

    #[test]
    fn test_nsga3_run_dtlz1_2obj() {
        let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); 3];
        let cfg = Nsga3Config {
            n_obj: 2,
            n_pop: 12,
            n_gen: 10,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
            seed: 7,
        };
        let result = nsga3_run(dtlz1_2obj, 3, &bounds, &cfg).expect("nsga3_run should succeed");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_nsga3_error_n_obj_lt2() {
        let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); 2];
        let cfg = Nsga3Config {
            n_obj: 1,
            n_pop: 10,
            n_gen: 5,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
            seed: 0,
        };
        assert!(nsga3_run(|_| vec![0.0], 2, &bounds, &cfg).is_err());
    }

    #[test]
    fn test_nsga3_error_bounds_mismatch() {
        let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); 3];
        let cfg = Nsga3Config {
            n_obj: 2,
            n_pop: 10,
            n_gen: 5,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
            seed: 0,
        };
        // n_var=5 but bounds.len()=3
        assert!(nsga3_run(|_| vec![0.0, 1.0], 5, &bounds, &cfg).is_err());
    }

    #[test]
    fn test_nsga3_pareto_front_non_dominated() {
        let bounds: Vec<(f64, f64)> = vec![(0.0, 1.0); 3];
        let cfg = Nsga3Config {
            n_obj: 2,
            n_pop: 12,
            n_gen: 5,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
            seed: 99,
        };
        let result = nsga3_run(zdt1, 3, &bounds, &cfg).expect("nsga3_run should succeed");
        // Verify none of the returned individuals dominates another
        for i in 0..result.len() {
            for j in 0..result.len() {
                if i != j {
                    assert!(
                        !dominates(&result[i].1, &result[j].1),
                        "individual {i} dominates {j}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_associate_to_ref_points_basic() {
        let ref_pts = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let norm_objs = vec![vec![0.9, 0.1], vec![0.1, 0.9], vec![0.8, 0.2]];
        let assoc = associate_to_ref_points(&norm_objs, &ref_pts, &[0, 1, 2]);
        // index 0 and 2 should be nearest to ref_pt 0 ([1,0])
        assert_eq!(assoc[0].0, 0);
        assert_eq!(assoc[1].0, 1);
        assert_eq!(assoc[2].0, 0);
    }
}
