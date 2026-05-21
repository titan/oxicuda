//! SMS-EMOA: S-Metric Selection Evolutionary Multiobjective Optimisation Algorithm.
//!
//! Hypervolume-indicator-based steady-state MOEA maintaining a fixed population of N
//! individuals.  At each step one offspring is generated, added to the population (N+1),
//! and the individual contributing the least hypervolume to the last non-dominated front
//! is removed, returning the population to N.
//!
//! # References
//! - N. Beume, B. Naujoks & M. Emmerich, "SMS-EMOA: Multiobjective selection based on
//!   dominated hypervolume", EJOR 181(3):1653-1669, 2007.
//! - M. Emmerich, N. Beume & B. Naujoks, "An EMO algorithm using the hypervolume measure
//!   as selection criterion", EMO 2005, LNCS 3410, pp. 62-76.

#![allow(clippy::needless_range_loop)]

use crate::genetic::crossover::sbx_crossover;
use crate::genetic::mutation::polynomial_mutate;
use crate::{EvolError, EvolResult, handle::LcgRng};

// ── Hypervolume helpers ───────────────────────────────────────────────────────

/// Sort indices by the k-th objective ascending.
fn sort_by_obj(objectives: &[Vec<f64>], indices: &[usize], k: usize) -> Vec<usize> {
    let mut sorted = indices.to_vec();
    sorted.sort_by(|&a, &b| {
        objectives[a][k]
            .partial_cmp(&objectives[b][k])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted
}

/// Compute the 2D hypervolume for a set of points given by `indices` into `objectives`.
/// Uses the sweep-line approach: sort by obj[0] ascending, accumulate slice areas.
fn hypervolume_2d_subset(objectives: &[Vec<f64>], indices: &[usize], ref_point: &[f64]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let sorted = sort_by_obj(objectives, indices, 0);

    let mut area = 0.0;
    let mut prev_f2 = ref_point[1];

    for &i in &sorted {
        let f1 = objectives[i][0];
        let f2 = objectives[i][1];
        if f1 >= ref_point[0] || f2 >= ref_point[1] {
            continue;
        }
        if prev_f2 > f2 {
            area += (ref_point[0] - f1) * (prev_f2 - f2);
            prev_f2 = f2;
        }
    }
    area
}

/// Exclusive 2D hypervolume contribution of point at `idx` to a front.
///
/// HV_contribution(i) = HV(front) - HV(front \ {i}).
pub fn hv_contribution_2d(front_objectives: &[Vec<f64>], ref_point: &[f64], idx: usize) -> f64 {
    let n = front_objectives.len();
    let all_indices: Vec<usize> = (0..n).collect();
    let without_idx: Vec<usize> = (0..n).filter(|&j| j != idx).collect();

    let hv_full = hypervolume_2d_subset(front_objectives, &all_indices, ref_point);
    let hv_without = hypervolume_2d_subset(front_objectives, &without_idx, ref_point);
    (hv_full - hv_without).max(0.0)
}

/// n-dimensional hypervolume contribution via Monte Carlo approximation (10 000 samples).
///
/// Estimates HV(front) - HV(front \ {idx}) by sampling the bounding box
/// [min_obj_j, ref_j) for each dimension.
pub fn hv_contribution_nd(
    front_objectives: &[Vec<f64>],
    ref_point: &[f64],
    idx: usize,
    rng: &mut LcgRng,
) -> f64 {
    let n_obj = ref_point.len();
    if front_objectives.is_empty() {
        return 0.0;
    }

    // Bounding box lower bounds = minimum objective values across front
    let lower: Vec<f64> = (0..n_obj)
        .map(|k| {
            front_objectives
                .iter()
                .map(|obj| obj[k])
                .fold(f64::INFINITY, f64::min)
        })
        .collect();

    // Box volume
    let box_vol: f64 = (0..n_obj)
        .map(|k| (ref_point[k] - lower[k]).max(0.0))
        .product();
    if box_vol <= 0.0 {
        return 0.0;
    }

    const N_SAMPLES: usize = 10_000;
    let mut dominated_full = 0usize;
    let mut dominated_without = 0usize;

    for _ in 0..N_SAMPLES {
        // Sample a point uniformly in the box
        let point: Vec<f64> = (0..n_obj)
            .map(|k| lower[k] + rng.next_f64() * (ref_point[k] - lower[k]))
            .collect();

        // Check if dominated by the full front
        let dom_full = front_objectives.iter().any(|obj| {
            obj.iter().zip(point.iter()).all(|(o, p)| o <= p)
                && obj.iter().zip(point.iter()).any(|(o, p)| o < p)
        });
        if dom_full {
            dominated_full += 1;
        }

        // Check if dominated by front without idx
        let dom_without = front_objectives.iter().enumerate().any(|(j, obj)| {
            if j == idx {
                return false;
            }
            obj.iter().zip(point.iter()).all(|(o, p)| o <= p)
                && obj.iter().zip(point.iter()).any(|(o, p)| o < p)
        });
        if dom_without {
            dominated_without += 1;
        }
    }

    let hv_full = box_vol * dominated_full as f64 / N_SAMPLES as f64;
    let hv_without = box_vol * dominated_without as f64 / N_SAMPLES as f64;
    (hv_full - hv_without).max(0.0)
}

// ── Non-dominated sort helpers ────────────────────────────────────────────────

/// Returns `true` if objective vector `a` weakly dominates `b` (all ≤, at least one <).
fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut strictly_less = false;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            return false;
        }
        if ai < bi {
            strictly_less = true;
        }
    }
    strictly_less
}

/// Fast non-dominated sort on a slice of objective vectors (indices 0..n).
/// Returns fronts as lists of indices.
fn fast_nondominated_sort_objs(objectives: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let n = objectives.len();
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut domination_count: Vec<usize> = vec![0; n];
    let mut fronts: Vec<Vec<usize>> = vec![Vec::new()];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if dominates(&objectives[i], &objectives[j]) {
                dominated_by[i].push(j);
            } else if dominates(&objectives[j], &objectives[i]) {
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

// ── Public types ──────────────────────────────────────────────────────────────

/// Configuration for SMS-EMOA.
#[derive(Debug, Clone)]
pub struct SmsEmoaConfig {
    /// Fixed population size N.
    pub n_pop: usize,
    /// Number of generations to run.
    pub n_gen: usize,
    /// Number of objective functions.
    pub n_obj: usize,
    /// SBX crossover distribution index (η_c).
    pub crossover_eta: f64,
    /// Polynomial mutation distribution index (η_m).
    pub mutation_eta: f64,
    /// Hypervolume reference point (must strictly dominate all expected solutions).
    pub ref_point: Vec<f64>,
    /// Random seed.
    pub seed: u64,
}

impl SmsEmoaConfig {
    /// Construct a default SMS-EMOA config for `n_obj` objectives.
    pub fn new(n_pop: usize, n_gen: usize, n_obj: usize, ref_point: Vec<f64>) -> EvolResult<Self> {
        if n_pop < 2 {
            return Err(EvolError::PopulationTooSmall {
                size: n_pop,
                op: "SMS-EMOA",
            });
        }
        if n_obj == 0 {
            return Err(EvolError::InvalidParameter("n_obj must be >= 1".to_owned()));
        }
        if ref_point.len() != n_obj {
            return Err(EvolError::DimensionMismatch {
                expected: n_obj,
                got: ref_point.len(),
            });
        }
        Ok(Self {
            n_pop,
            n_gen,
            n_obj,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
            ref_point,
            seed: 0,
        })
    }
}

/// State returned by an SMS-EMOA run.
#[derive(Debug, Clone)]
pub struct SmsEmoaState {
    /// Decision variable vectors for each individual (length = n_pop).
    pub population: Vec<Vec<f64>>,
    /// Objective function values for each individual (length = n_pop).
    pub objectives: Vec<Vec<f64>>,
    /// Number of generations completed.
    pub generation: usize,
}

// ── Per-generation SMS-EMOA step ──────────────────────────────────────────────

/// Execute one SMS-EMOA generation step:
/// 1. Generate 1 offspring via SBX + polynomial mutation.
/// 2. Add to population (size N+1).
/// 3. Find the last non-dominated front.
/// 4. Remove the individual from that front with the smallest HV contribution.
fn sms_emoa_step<F>(
    population: &mut Vec<Vec<f64>>,
    objectives: &mut Vec<Vec<f64>>,
    fitness_fn: &F,
    bounds: &[(f64, f64)],
    cfg: &SmsEmoaConfig,
    rng: &mut LcgRng,
) -> EvolResult<()>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n_pop = population.len();
    let n_var = if n_pop > 0 { population[0].len() } else { 0 };
    let n_obj = cfg.n_obj;

    if n_pop < 2 {
        return Err(EvolError::PopulationTooSmall {
            size: n_pop,
            op: "SMS-EMOA step",
        });
    }

    // ── Step 1: select 2 parents uniformly at random ──────────────────────────
    let p1_idx = rng.next_usize(n_pop);
    let mut p2_idx = rng.next_usize(n_pop - 1);
    if p2_idx >= p1_idx {
        p2_idx += 1;
    }

    // ── Step 2: SBX crossover + polynomial mutation ───────────────────────────
    // SBX uses a single scalar bound; use the first bound pair for compatibility
    // and then per-gene clamp.
    let global_bounds = if bounds.is_empty() {
        (f64::NEG_INFINITY, f64::INFINITY)
    } else {
        bounds[0]
    };

    let (mut child_genome, _) = sbx_crossover(
        &population[p1_idx],
        &population[p2_idx],
        cfg.crossover_eta,
        global_bounds,
        rng,
    )?;

    let p_mut = 1.0 / n_var.max(1) as f64;
    polynomial_mutate(
        &mut child_genome,
        cfg.mutation_eta,
        p_mut,
        global_bounds,
        rng,
    );

    // Clamp each gene to its individual bounds
    for (i, g) in child_genome.iter_mut().enumerate() {
        if i < bounds.len() {
            *g = g.clamp(bounds[i].0, bounds[i].1);
        }
    }

    // ── Step 3: evaluate offspring and add to population (N+1) ───────────────
    let child_obj = fitness_fn(&child_genome);
    if child_obj.len() != n_obj {
        return Err(EvolError::ObjectiveCountMismatch);
    }
    population.push(child_genome);
    objectives.push(child_obj);

    // ── Step 4: find last non-dominated front ─────────────────────────────────
    let fronts = fast_nondominated_sort_objs(objectives);
    let last_front = fronts.last().cloned().unwrap_or_default();

    if last_front.is_empty() {
        // Fallback: remove worst by first objective
        let worst = objectives
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(n_pop);
        population.remove(worst);
        objectives.remove(worst);
        return Ok(());
    }

    // ── Step 5: find the individual in the last front with least HV contribution
    let last_front_objs: Vec<Vec<f64>> =
        last_front.iter().map(|&i| objectives[i].clone()).collect();

    let remove_local_idx = if n_obj == 2 {
        // Efficient exact 2D contribution
        let contributions: Vec<f64> = (0..last_front.len())
            .map(|j| hv_contribution_2d(&last_front_objs, &cfg.ref_point, j))
            .collect();
        contributions
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(j, _)| j)
            .unwrap_or(0)
    } else {
        // N-dimensional: Monte Carlo approximation
        let contributions: Vec<f64> = (0..last_front.len())
            .map(|j| hv_contribution_nd(&last_front_objs, &cfg.ref_point, j, rng))
            .collect();
        contributions
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(j, _)| j)
            .unwrap_or(0)
    };

    let remove_global_idx = last_front[remove_local_idx];

    // ── Step 6: remove that individual ───────────────────────────────────────
    population.remove(remove_global_idx);
    objectives.remove(remove_global_idx);

    Ok(())
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Run SMS-EMOA optimisation.
///
/// # Parameters
/// - `fitness_fn`: maps a decision variable vector to a vector of `n_obj` objective values
///   (all minimised).
/// - `n_var`: number of decision variables.
/// - `bounds`: per-variable `(lower, upper)` bounds (length must equal `n_var`).
/// - `cfg`: algorithm configuration.
///
/// # Errors
/// Returns an error if bounds or config are inconsistent.
pub fn sms_emoa_run<F>(
    fitness_fn: F,
    n_var: usize,
    bounds: &[(f64, f64)],
    cfg: &SmsEmoaConfig,
) -> EvolResult<SmsEmoaState>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if n_var == 0 {
        return Err(EvolError::InvalidParameter("n_var must be >= 1".to_owned()));
    }
    if bounds.len() != n_var {
        return Err(EvolError::DimensionMismatch {
            expected: n_var,
            got: bounds.len(),
        });
    }
    if cfg.n_pop < 2 {
        return Err(EvolError::PopulationTooSmall {
            size: cfg.n_pop,
            op: "SMS-EMOA",
        });
    }

    let mut rng = LcgRng::new(cfg.seed);

    // ── Initialise population uniformly within bounds ─────────────────────────
    let mut population: Vec<Vec<f64>> = (0..cfg.n_pop)
        .map(|_| {
            bounds
                .iter()
                .map(|&(lo, hi)| lo + rng.next_f64() * (hi - lo))
                .collect()
        })
        .collect();

    let mut objectives: Vec<Vec<f64>> = population.iter().map(|x| fitness_fn(x)).collect();

    // Validate objective lengths
    for obj in &objectives {
        if obj.len() != cfg.n_obj {
            return Err(EvolError::ObjectiveCountMismatch);
        }
    }

    // ── Main loop ─────────────────────────────────────────────────────────────
    for _ in 0..cfg.n_gen {
        sms_emoa_step(
            &mut population,
            &mut objectives,
            &fitness_fn,
            bounds,
            cfg,
            &mut rng,
        )?;
    }

    Ok(SmsEmoaState {
        population,
        objectives,
        generation: cfg.n_gen,
    })
}

/// Extract the Pareto front from a completed SMS-EMOA state.
///
/// Returns a vector of `(decision_variables, objectives)` pairs for each
/// non-dominated individual in the population.
pub fn sms_emoa_pareto_front(state: &SmsEmoaState) -> Vec<(Vec<f64>, Vec<f64>)> {
    let n = state.population.len();
    if n == 0 {
        return Vec::new();
    }

    let fronts = fast_nondominated_sort_objs(&state.objectives);
    let pareto_indices = if fronts.is_empty() {
        Vec::new()
    } else {
        fronts[0].clone()
    };

    pareto_indices
        .into_iter()
        .map(|i| (state.population[i].clone(), state.objectives[i].clone()))
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 2-objective ZDT1-like test function on [0,1]^n_var.
    fn zdt1(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let f1 = x[0];
        let g: f64 = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (n - 1) as f64;
        let h = 1.0 - (f1 / g).sqrt();
        vec![f1, g * h]
    }

    /// Simple 2-objective sphere variant: (x^2, (x-1)^2).
    fn biobjective_sphere(x: &[f64]) -> Vec<f64> {
        let f1: f64 = x.iter().map(|&xi| xi * xi).sum();
        let f2: f64 = x.iter().map(|&xi| (xi - 1.0) * (xi - 1.0)).sum();
        vec![f1, f2]
    }

    /// 3-objective test: (x1^2, x2^2, x3^2) on [0,1]^3.
    fn triobj(x: &[f64]) -> Vec<f64> {
        vec![x[0] * x[0], x[1] * x[1], x[2] * x[2]]
    }

    // ── Config construction ───────────────────────────────────────────────────

    #[test]
    fn test_config_new_valid() {
        let cfg = SmsEmoaConfig::new(10, 50, 2, vec![2.0, 2.0]).unwrap();
        assert_eq!(cfg.n_pop, 10);
        assert_eq!(cfg.n_obj, 2);
        assert_eq!(cfg.ref_point.len(), 2);
    }

    #[test]
    fn test_config_new_pop_too_small() {
        assert!(SmsEmoaConfig::new(1, 10, 2, vec![2.0, 2.0]).is_err());
    }

    #[test]
    fn test_config_new_zero_obj() {
        assert!(SmsEmoaConfig::new(10, 10, 0, vec![]).is_err());
    }

    #[test]
    fn test_config_ref_point_mismatch() {
        assert!(SmsEmoaConfig::new(10, 10, 2, vec![1.0]).is_err());
    }

    // ── Population size invariant ─────────────────────────────────────────────

    #[test]
    fn test_final_population_size_equals_n_pop() {
        let cfg = SmsEmoaConfig {
            n_pop: 20,
            n_gen: 50,
            n_obj: 2,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
            ref_point: vec![2.0, 2.0],
            seed: 1,
        };
        let bounds = vec![(0.0f64, 1.0f64); 5];
        let state = sms_emoa_run(zdt1, 5, &bounds, &cfg).unwrap();
        assert_eq!(state.population.len(), 20);
        assert_eq!(state.objectives.len(), 20);
    }

    #[test]
    fn test_generation_counter_correct() {
        let cfg = SmsEmoaConfig {
            n_pop: 10,
            n_gen: 30,
            n_obj: 2,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
            ref_point: vec![2.0, 2.0],
            seed: 2,
        };
        let bounds = vec![(0.0, 1.0); 3];
        let state = sms_emoa_run(biobjective_sphere, 3, &bounds, &cfg).unwrap();
        assert_eq!(state.generation, 30);
    }

    // ── Objective validity ────────────────────────────────────────────────────

    #[test]
    fn test_all_objectives_finite() {
        let cfg = SmsEmoaConfig {
            n_pop: 15,
            n_gen: 40,
            n_obj: 2,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
            ref_point: vec![2.0, 2.0],
            seed: 3,
        };
        let bounds = vec![(0.0, 1.0); 5];
        let state = sms_emoa_run(zdt1, 5, &bounds, &cfg).unwrap();
        for obj in &state.objectives {
            for &v in obj {
                assert!(v.is_finite(), "objective value is not finite: {v}");
            }
        }
    }

    #[test]
    fn test_n_gen_zero_returns_initial_population() {
        let cfg = SmsEmoaConfig {
            n_pop: 10,
            n_gen: 0,
            n_obj: 2,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
            ref_point: vec![2.0, 2.0],
            seed: 4,
        };
        let bounds = vec![(0.0, 1.0); 4];
        let state = sms_emoa_run(biobjective_sphere, 4, &bounds, &cfg).unwrap();
        assert_eq!(state.population.len(), 10);
        assert_eq!(state.generation, 0);
    }

    // ── Pareto front ──────────────────────────────────────────────────────────

    #[test]
    fn test_pareto_front_non_dominated() {
        let cfg = SmsEmoaConfig {
            n_pop: 20,
            n_gen: 100,
            n_obj: 2,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
            ref_point: vec![2.0, 2.0],
            seed: 5,
        };
        let bounds = vec![(0.0, 1.0); 5];
        let state = sms_emoa_run(zdt1, 5, &bounds, &cfg).unwrap();
        let front = sms_emoa_pareto_front(&state);

        // Verify non-domination within the returned front
        let objs: Vec<Vec<f64>> = front.iter().map(|(_, o)| o.clone()).collect();
        for i in 0..objs.len() {
            for j in 0..objs.len() {
                if i != j {
                    assert!(
                        !dominates(&objs[j], &objs[i]),
                        "individual {j} dominates {i} in pareto front"
                    );
                }
            }
        }
    }

    #[test]
    fn test_pareto_front_not_empty_after_run() {
        let cfg = SmsEmoaConfig {
            n_pop: 15,
            n_gen: 50,
            n_obj: 2,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
            ref_point: vec![2.0, 2.0],
            seed: 6,
        };
        let bounds = vec![(0.0, 1.0); 4];
        let state = sms_emoa_run(biobjective_sphere, 4, &bounds, &cfg).unwrap();
        let front = sms_emoa_pareto_front(&state);
        assert!(!front.is_empty());
    }

    // ── HV contribution ───────────────────────────────────────────────────────

    #[test]
    fn test_hv_contribution_2d_non_negative() {
        let front_objs = vec![vec![0.1, 0.9], vec![0.5, 0.5], vec![0.9, 0.1]];
        let ref_point = vec![1.1, 1.1];
        for i in 0..front_objs.len() {
            let c = hv_contribution_2d(&front_objs, &ref_point, i);
            assert!(c >= 0.0, "contribution[{i}]={c} is negative");
        }
    }

    #[test]
    fn test_hv_contribution_nd_non_negative() {
        let mut rng = LcgRng::new(99);
        let front_objs = vec![
            vec![0.1, 0.9, 0.5],
            vec![0.5, 0.5, 0.5],
            vec![0.9, 0.1, 0.3],
        ];
        let ref_point = vec![1.1, 1.1, 1.1];
        for i in 0..front_objs.len() {
            let c = hv_contribution_nd(&front_objs, &ref_point, i, &mut rng);
            assert!(c >= 0.0, "3D contribution[{i}]={c} is negative");
        }
    }

    #[test]
    fn test_hv_contribution_2d_single_point_equals_full_hv() {
        // With one point, the contribution equals the full 2D hypervolume
        let front_objs = vec![vec![0.3, 0.4]];
        let ref_point = vec![1.0, 1.0];
        let contrib = hv_contribution_2d(&front_objs, &ref_point, 0);
        let expected = (1.0 - 0.3) * (1.0 - 0.4);
        assert!(
            (contrib - expected).abs() < 1e-9,
            "contrib={contrib} expected={expected}"
        );
    }

    // ── 2D convergence ────────────────────────────────────────────────────────

    #[test]
    fn test_2d_biobjective_converges_toward_pareto() {
        // After enough generations, some individuals should be near Pareto front
        // f1 + f2 = x^2 + (x-1)^2; minimum of sum near x = 0.5 → both objectives ~0.25
        let cfg = SmsEmoaConfig {
            n_pop: 20,
            n_gen: 200,
            n_obj: 2,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
            ref_point: vec![2.0, 2.0],
            seed: 42,
        };
        let bounds = vec![(0.0, 1.0); 1];
        let state = sms_emoa_run(biobjective_sphere, 1, &bounds, &cfg).unwrap();

        // All objectives should be within [0, 2] on this problem
        for obj in &state.objectives {
            assert!(obj[0] >= 0.0 && obj[0] <= 2.0);
            assert!(obj[1] >= 0.0 && obj[1] <= 2.0);
        }
    }

    // ── 3-objective case ──────────────────────────────────────────────────────

    #[test]
    fn test_3obj_population_size_maintained() {
        let cfg = SmsEmoaConfig {
            n_pop: 20,
            n_gen: 50,
            n_obj: 3,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
            ref_point: vec![1.5, 1.5, 1.5],
            seed: 77,
        };
        let bounds = vec![(0.0, 1.0); 3];
        let state = sms_emoa_run(triobj, 3, &bounds, &cfg).unwrap();
        assert_eq!(state.population.len(), cfg.n_pop);
    }

    #[test]
    fn test_3obj_all_objectives_finite() {
        let cfg = SmsEmoaConfig {
            n_pop: 15,
            n_gen: 30,
            n_obj: 3,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
            ref_point: vec![1.5, 1.5, 1.5],
            seed: 88,
        };
        let bounds = vec![(0.0, 1.0); 3];
        let state = sms_emoa_run(triobj, 3, &bounds, &cfg).unwrap();
        for obj in &state.objectives {
            for &v in obj {
                assert!(v.is_finite());
            }
        }
    }

    // ── Error paths ───────────────────────────────────────────────────────────

    #[test]
    fn test_bounds_mismatch_error() {
        let cfg = SmsEmoaConfig::new(10, 10, 2, vec![2.0, 2.0]).unwrap();
        // n_var = 5 but bounds has 3 entries
        assert!(sms_emoa_run(zdt1, 5, &[(0.0, 1.0); 3], &cfg).is_err());
    }

    #[test]
    fn test_zero_n_var_error() {
        let cfg = SmsEmoaConfig::new(10, 10, 2, vec![2.0, 2.0]).unwrap();
        let bounds: Vec<(f64, f64)> = Vec::new();
        assert!(sms_emoa_run(|_: &[f64]| vec![0.0, 0.0], 0, &bounds, &cfg).is_err());
    }
}
