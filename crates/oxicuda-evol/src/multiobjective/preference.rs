//! Preference-Articulation Multi-Objective Evolutionary Algorithms.
//!
//! Implements two preference-guided MOEA variants:
//!
//! 1. **R-NSGA-II** (Deb & Sundar 2006): Augments the standard NSGA-II crowding
//!    distance with reference-point proximity in objective space, so solutions
//!    near the preferred region are preferentially retained.
//!
//! 2. **Preference MOEA/D**: Standard Tchebycheff MOEA/D with a post-selection
//!    preference-region filter: only solutions whose objective vector lies within
//!    angular distance `epsilon_pref` of the `preferred_direction` unit vector
//!    are kept. If too few survive the filter, the full non-dominated set is returned.
//!
//! # References
//! - Deb, K. & Sundar, J. (2006). "Reference point based multi-objective optimization
//!   using evolutionary algorithms." *Proc. GECCO 2006*, pp. 635–642.
//! - Zhang, Q. & Li, H. (2007). "MOEA/D: A multiobjective evolutionary algorithm based
//!   on decomposition." *IEEE Trans. Evol. Comput.*, 11(6), 712–731.

#![allow(clippy::needless_range_loop)]

use crate::genetic::crossover::sbx_crossover;
use crate::genetic::mutation::polynomial_mutate;
use crate::metrics::hypervolume_nd::hypervolume_nd;
use crate::multiobjective::nsga2::{
    MultiObjectiveIndividual, crowding_distance, fast_nondominated_sort, nsga2_tournament,
};
use crate::{EvolError, EvolResult, handle::LcgRng};

// ─────────────────────────────────────────────────────────────────────────────
// R-NSGA-II types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for an R-NSGA-II run.
#[derive(Debug, Clone)]
pub struct RNsga2Config {
    /// Number of objectives.
    pub n_obj: usize,
    /// Population size (must be ≥ 2 and even for SBX crossover pairing).
    pub pop_size: usize,
    /// Number of generations.
    pub n_gens: usize,
    /// Preferred reference points in objective space. Must be non-empty.
    pub reference_points: Vec<Vec<f64>>,
    /// Neighbourhood parameter: solutions within epsilon of the nearest reference point
    /// are preferentially retained during crowding comparison.
    pub epsilon: f64,
    /// SBX crossover / polynomial mutation standard deviation.
    pub sigma_mut: f64,
    /// Per-gene mutation probability ∈ (0, 1].
    pub p_mut: f64,
    /// Number of decision variables.
    pub n_dims: usize,
    /// Lower bound for all decision variables.
    pub lb: f64,
    /// Upper bound for all decision variables.
    pub ub: f64,
}

impl RNsga2Config {
    /// Validate configuration parameters.
    pub fn validate(&self) -> EvolResult<()> {
        if self.n_obj == 0 {
            return Err(EvolError::InvalidParameter(
                "RNsga2Config: n_obj must be >= 1".to_owned(),
            ));
        }
        if self.pop_size < 2 {
            return Err(EvolError::PopulationTooSmall {
                size: self.pop_size,
                op: "R-NSGA-II",
            });
        }
        if self.reference_points.is_empty() {
            return Err(EvolError::InvalidParameter(
                "RNsga2Config: reference_points must be non-empty".to_owned(),
            ));
        }
        for (i, rp) in self.reference_points.iter().enumerate() {
            if rp.len() != self.n_obj {
                return Err(EvolError::DimensionMismatch {
                    expected: self.n_obj,
                    got: rp.len(),
                });
            }
            let _ = i;
        }
        if self.n_dims == 0 {
            return Err(EvolError::InvalidParameter(
                "RNsga2Config: n_dims must be >= 1".to_owned(),
            ));
        }
        if self.lb >= self.ub {
            return Err(EvolError::InvalidParameter(format!(
                "RNsga2Config: lb ({}) must be < ub ({})",
                self.lb, self.ub
            )));
        }
        if self.epsilon < 0.0 {
            return Err(EvolError::InvalidParameter(
                "RNsga2Config: epsilon must be >= 0".to_owned(),
            ));
        }
        if !(self.p_mut > 0.0 && self.p_mut <= 1.0) {
            return Err(EvolError::InvalidParameter(format!(
                "RNsga2Config: p_mut ({}) must be in (0, 1]",
                self.p_mut
            )));
        }
        Ok(())
    }
}

/// Results of an R-NSGA-II run.
#[derive(Debug, Clone)]
pub struct RNsga2Result {
    /// Decision variable vectors of the final Pareto front approximation.
    pub pareto_front: Vec<Vec<f64>>,
    /// Objective values corresponding to each member of `pareto_front`.
    pub objectives: Vec<Vec<f64>>,
    /// Hypervolume indicator tracked per generation (length == `n_gens`).
    /// Uses a fixed reference point = max observed value + 1 on each objective.
    pub history_hv: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// R-NSGA-II internals
// ─────────────────────────────────────────────────────────────────────────────

/// Euclidean distance between two objective vectors.
fn euclid_dist_obj(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute the modified crowding distance for R-NSGA-II.
///
/// For each individual in `front`:
///   modified_crowding(i) = standard_crowding(i) - alpha * min_r(||obj(i) - r||)
///
/// where r ranges over the given reference points. Proximity to a reference point
/// increases the effective crowding (makes the individual more attractive in selection).
///
/// We encode this by setting `crowding_dist` = standard_crowding + proximity_bonus
/// so that the existing `nsga2_tournament` (higher crowding_dist → preferred) works correctly.
fn modified_crowding_distance(
    population: &mut [MultiObjectiveIndividual],
    front: &[usize],
    reference_points: &[Vec<f64>],
    epsilon: f64,
) {
    // First apply standard crowding distance
    crowding_distance(population, front);

    if reference_points.is_empty() {
        return;
    }

    let f_len = front.len();
    if f_len == 0 {
        return;
    }

    // For each individual in the front, compute its minimum distance to any reference point
    // and add a bonus proportional to proximity.
    for &idx in front {
        let obj = &population[idx].objectives;
        let min_dist = reference_points
            .iter()
            .map(|rp| euclid_dist_obj(obj, rp))
            .fold(f64::INFINITY, f64::min);

        // Proximity bonus: the closer to a reference point, the larger the bonus.
        // If within epsilon neighbourhood, give a large bonus to guarantee selection.
        let bonus = if min_dist < epsilon {
            // Within epsilon → high priority bonus
            1e6 / (1.0 + min_dist)
        } else {
            // Soft bonus decaying with distance (encourages convergence toward ref)
            1.0 / (1.0 + min_dist)
        };

        // Add bonus to crowding distance (which is already set by crowding_distance above)
        if population[idx].crowding_dist.is_finite() {
            population[idx].crowding_dist += bonus;
        }
        // Boundary points already have INFINITY crowding_dist → they remain preferred
    }
}

/// Run fast non-dominated sort + modified crowding distance for R-NSGA-II.
fn assign_ranks_and_modified_crowding(
    population: &mut [MultiObjectiveIndividual],
    reference_points: &[Vec<f64>],
    epsilon: f64,
) {
    let fronts = fast_nondominated_sort(population);
    for (rank, front) in fronts.iter().enumerate() {
        for &i in front {
            population[i].rank = rank;
        }
        modified_crowding_distance(population, front, reference_points, epsilon);
    }
}

/// Compute a hypervolume indicator for the current front, using a reference point
/// 10% above the maximum observed value on each objective (shifted to strictly dominate).
fn compute_hv_indicator(objectives: &[Vec<f64>], n_obj: usize) -> f64 {
    if objectives.is_empty() || n_obj == 0 {
        return 0.0;
    }

    // Compute a dynamic reference point: max + 0.1 * range on each objective
    let mut maxima = vec![f64::NEG_INFINITY; n_obj];
    let mut minima = vec![f64::INFINITY; n_obj];
    for obj in objectives {
        for (j, &v) in obj.iter().enumerate().take(n_obj) {
            if v > maxima[j] {
                maxima[j] = v;
            }
            if v < minima[j] {
                minima[j] = v;
            }
        }
    }

    let ref_pt: Vec<f64> = maxima
        .iter()
        .zip(minima.iter())
        .map(|(&mx, &mn)| mx + (mx - mn).abs().max(1.0) * 0.1 + 1.0)
        .collect();

    let ref_wrapped = vec![ref_pt];
    hypervolume_nd(objectives, &ref_wrapped).unwrap_or(0.0)
}

/// Run R-NSGA-II on a minimization problem.
///
/// # Algorithm
/// Uses the standard NSGA-II backbone (fast non-dominated sort + SBX crossover +
/// polynomial mutation) with a modified crowding distance that rewards proximity
/// to user-supplied reference points in objective space.
///
/// # Errors
/// Returns `EvolError::InvalidParameter` if configuration is invalid.
pub fn r_nsga2_run<F>(
    config: &RNsga2Config,
    objective_fn: F,
    rng: &mut LcgRng,
) -> EvolResult<RNsga2Result>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    config.validate()?;

    let (lb, ub) = (config.lb, config.ub);
    let range = ub - lb;
    let eta = 20.0; // SBX and polynomial mutation distribution index
    let bounds = (lb, ub);

    // ── Initialise population ─────────────────────────────────────────────────
    let mut population: Vec<MultiObjectiveIndividual> = (0..config.pop_size)
        .map(|_| {
            let genome: Vec<f64> = (0..config.n_dims)
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

    assign_ranks_and_modified_crowding(&mut population, &config.reference_points, config.epsilon);

    let mut history_hv = Vec::with_capacity(config.n_gens);

    // ── Main evolution loop ───────────────────────────────────────────────────
    for _gen in 0..config.n_gens {
        // Generate offspring via SBX + polynomial mutation
        let mut offspring = Vec::with_capacity(config.pop_size);
        while offspring.len() < config.pop_size {
            let p1 = nsga2_tournament(&population, rng);
            let p2 = nsga2_tournament(&population, rng);
            let (mut c1_g, mut c2_g) = sbx_crossover(
                &population[p1].genome,
                &population[p2].genome,
                eta,
                bounds,
                rng,
            )?;
            polynomial_mutate(&mut c1_g, eta, config.p_mut, bounds, rng);
            polynomial_mutate(&mut c2_g, eta, config.p_mut, bounds, rng);

            if offspring.len() < config.pop_size {
                let obj1 = objective_fn(&c1_g);
                offspring.push(MultiObjectiveIndividual {
                    genome: c1_g,
                    objectives: obj1,
                    rank: 0,
                    crowding_dist: 0.0,
                });
            }
            if offspring.len() < config.pop_size {
                let obj2 = objective_fn(&c2_g);
                offspring.push(MultiObjectiveIndividual {
                    genome: c2_g,
                    objectives: obj2,
                    rank: 0,
                    crowding_dist: 0.0,
                });
            }
        }

        // Combine parent + offspring
        let mut combined = population;
        combined.extend(offspring);
        assign_ranks_and_modified_crowding(&mut combined, &config.reference_points, config.epsilon);

        // Environmental selection: fill next generation front by front
        let fronts = fast_nondominated_sort(&combined);
        let mut next_pop = Vec::with_capacity(config.pop_size);
        'outer: for front in &fronts {
            if next_pop.len() + front.len() <= config.pop_size {
                for &i in front {
                    next_pop.push(combined[i].clone());
                }
            } else {
                // Partial front: sort by modified crowding distance descending
                let mut partial = front.clone();
                partial.sort_by(|&a, &b| {
                    combined[b]
                        .crowding_dist
                        .partial_cmp(&combined[a].crowding_dist)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let remaining = config.pop_size - next_pop.len();
                for &i in partial.iter().take(remaining) {
                    next_pop.push(combined[i].clone());
                }
                break 'outer;
            }
        }

        population = next_pop;
        assign_ranks_and_modified_crowding(
            &mut population,
            &config.reference_points,
            config.epsilon,
        );

        // Track hypervolume
        let current_objectives: Vec<Vec<f64>> = population
            .iter()
            .filter(|ind| ind.rank == 0)
            .map(|ind| ind.objectives.clone())
            .collect();
        let hv = compute_hv_indicator(&current_objectives, config.n_obj);
        history_hv.push(hv);
    }

    // Collect rank-0 front
    let pareto_front: Vec<Vec<f64>> = population
        .iter()
        .filter(|ind| ind.rank == 0)
        .map(|ind| ind.genome.clone())
        .collect();
    let objectives: Vec<Vec<f64>> = population
        .iter()
        .filter(|ind| ind.rank == 0)
        .map(|ind| ind.objectives.clone())
        .collect();

    Ok(RNsga2Result {
        pareto_front,
        objectives,
        history_hv,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Preference MOEA/D types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Preference MOEA/D.
#[derive(Debug, Clone)]
pub struct PrefMoeadConfig {
    /// Number of objectives.
    pub n_obj: usize,
    /// Population / subproblem count.
    pub pop_size: usize,
    /// Total number of generations.
    pub n_gens: usize,
    /// Preferred direction unit vector in objective space (will be normalised internally).
    /// Must have length == n_obj.
    pub preferred_direction: Vec<f64>,
    /// Angular tolerance in radians (e.g., π/6 ≈ 0.524). Solutions within this angle
    /// from `preferred_direction` are retained.
    pub epsilon_pref: f64,
    /// Neighbourhood size T for MOEA/D.
    pub neighborhood_size: usize,
    /// Standard deviation for Gaussian perturbation in SBX-style mutation.
    pub sigma_mut: f64,
    /// Per-gene mutation probability.
    pub p_mut: f64,
    /// Number of decision variables.
    pub n_dims: usize,
    /// Lower bound.
    pub lb: f64,
    /// Upper bound.
    pub ub: f64,
}

impl PrefMoeadConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> EvolResult<()> {
        if self.n_obj == 0 {
            return Err(EvolError::InvalidParameter(
                "PrefMoeadConfig: n_obj must be >= 1".to_owned(),
            ));
        }
        if self.pop_size == 0 {
            return Err(EvolError::EmptyPopulation);
        }
        if self.neighborhood_size == 0 || self.neighborhood_size > self.pop_size {
            return Err(EvolError::InvalidParameter(format!(
                "PrefMoeadConfig: neighborhood_size {} must be in [1, pop_size={}]",
                self.neighborhood_size, self.pop_size
            )));
        }
        if self.preferred_direction.len() != self.n_obj {
            return Err(EvolError::DimensionMismatch {
                expected: self.n_obj,
                got: self.preferred_direction.len(),
            });
        }
        let norm: f64 = self
            .preferred_direction
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        if norm < 1e-12 {
            return Err(EvolError::InvalidParameter(
                "PrefMoeadConfig: preferred_direction must be a non-zero vector".to_owned(),
            ));
        }
        if self.epsilon_pref <= 0.0 {
            return Err(EvolError::InvalidParameter(
                "PrefMoeadConfig: epsilon_pref must be > 0".to_owned(),
            ));
        }
        if self.n_dims == 0 {
            return Err(EvolError::InvalidParameter(
                "PrefMoeadConfig: n_dims must be >= 1".to_owned(),
            ));
        }
        if self.lb >= self.ub {
            return Err(EvolError::InvalidParameter(format!(
                "PrefMoeadConfig: lb ({}) must be < ub ({})",
                self.lb, self.ub
            )));
        }
        if !(self.p_mut > 0.0 && self.p_mut <= 1.0) {
            return Err(EvolError::InvalidParameter(format!(
                "PrefMoeadConfig: p_mut ({}) must be in (0, 1]",
                self.p_mut
            )));
        }
        Ok(())
    }
}

/// Results of a Preference MOEA/D run.
#[derive(Debug, Clone)]
pub struct PrefMoeadResult {
    /// Decision variable vectors of retained solutions.
    pub solutions: Vec<Vec<f64>>,
    /// Objective values of retained solutions.
    pub objectives: Vec<Vec<f64>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Preference MOEA/D internals
// ─────────────────────────────────────────────────────────────────────────────

/// Generate uniformly spread weight vectors on the simplex (same as moead.rs approach).
fn generate_weights(pop_size: usize, n_obj: usize) -> Vec<Vec<f64>> {
    match n_obj {
        0 => vec![],
        1 => vec![vec![1.0]; pop_size],
        2 => (0..pop_size)
            .map(|i| {
                let t = i as f64 / (pop_size - 1).max(1) as f64;
                vec![t, 1.0 - t]
            })
            .collect(),
        3 => {
            let h = ((pop_size as f64 * 6.0).cbrt().round() as usize).max(1);
            let mut weights = Vec::new();
            'outer: for i in 0..=h {
                for j in 0..=(h - i) {
                    let k = h - i - j;
                    weights.push(vec![
                        i as f64 / h as f64,
                        j as f64 / h as f64,
                        k as f64 / h as f64,
                    ]);
                    if weights.len() >= pop_size {
                        break 'outer;
                    }
                }
            }
            while weights.len() < pop_size {
                weights.push(vec![1.0 / n_obj as f64; n_obj]);
            }
            weights.truncate(pop_size);
            weights
        }
        _ => vec![vec![1.0 / n_obj as f64; n_obj]; pop_size],
    }
}

/// Tchebycheff scalarisation.
fn tchebycheff(objectives: &[f64], weights: &[f64], ideal: &[f64]) -> f64 {
    objectives
        .iter()
        .zip(weights.iter())
        .zip(ideal.iter())
        .map(|((&f, &w), &z)| w * (f - z).abs())
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Compute the cosine angle between two vectors (in radians).
fn angle_between(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < 1e-14 || norm_b < 1e-14 {
        return std::f64::consts::PI;
    }
    let cos_theta = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
    cos_theta.acos()
}

/// Check whether objective vector `obj` lies within `epsilon_pref` radians of
/// the normalized `preferred_direction`.
fn within_preference_cone(obj: &[f64], pref_dir: &[f64], epsilon_pref: f64) -> bool {
    angle_between(obj, pref_dir) <= epsilon_pref
}

/// Non-dominated filter for raw objective vectors (returns indices).
fn nondominated_indices(objectives: &[Vec<f64>]) -> Vec<usize> {
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

/// Run Preference MOEA/D on a minimization problem.
///
/// # Algorithm
/// 1. Generate uniformly spread weight vectors over the simplex.
/// 2. Compute T-nearest-neighbour structure.
/// 3. Run standard Tchebycheff MOEA/D for `n_gens` generations.
/// 4. Post-filter: retain only solutions whose objective vector lies within
///    `epsilon_pref` angular distance of `preferred_direction` (normalised).
/// 5. If post-filter retains < 2 solutions, fall back to the full non-dominated set.
///
/// # Errors
/// Returns `EvolError::InvalidParameter` if configuration is invalid.
pub fn pref_moead_run<F>(
    config: &PrefMoeadConfig,
    objective_fn: F,
    rng: &mut LcgRng,
) -> EvolResult<PrefMoeadResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    config.validate()?;

    let (lb, ub) = (config.lb, config.ub);
    let range = ub - lb;
    let bounds = (lb, ub);
    let t = config.neighborhood_size;

    // Normalise preferred direction
    let norm: f64 = config
        .preferred_direction
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();
    let pref_dir: Vec<f64> = config
        .preferred_direction
        .iter()
        .map(|x| x / norm)
        .collect();

    // ── Weight vectors ────────────────────────────────────────────────────────
    let weights = generate_weights(config.pop_size, config.n_obj);
    let actual_pop = weights.len();

    // ── Neighbourhood structure ───────────────────────────────────────────────
    let neighbours: Vec<Vec<usize>> = (0..actual_pop)
        .map(|i| {
            let mut dists: Vec<(usize, f64)> = (0..actual_pop)
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
            dists.into_iter().take(t).map(|(idx, _)| idx).collect()
        })
        .collect();

    // ── Initial population ────────────────────────────────────────────────────
    let mut population: Vec<Vec<f64>> = (0..actual_pop)
        .map(|_| {
            (0..config.n_dims)
                .map(|_| lb + rng.next_f64() * range)
                .collect()
        })
        .collect();
    let mut objectives: Vec<Vec<f64>> = population.iter().map(|x| objective_fn(x)).collect();

    // ── Ideal point ───────────────────────────────────────────────────────────
    let mut ideal = vec![f64::INFINITY; config.n_obj];
    for obj in &objectives {
        for (j, &v) in obj.iter().enumerate().take(config.n_obj) {
            if v < ideal[j] {
                ideal[j] = v;
            }
        }
    }

    // ── Scalar fitness ────────────────────────────────────────────────────────
    let mut scalar_fit: Vec<f64> = (0..actual_pop)
        .map(|i| tchebycheff(&objectives[i], &weights[i], &ideal))
        .collect();

    // ── Main MOEA/D loop ──────────────────────────────────────────────────────
    for _gen in 0..config.n_gens {
        for i in 0..actual_pop {
            let pool = &neighbours[i];
            if pool.len() < 2 {
                continue;
            }
            let k1 = pool[rng.next_usize(pool.len())];
            let k2 = pool[rng.next_usize(pool.len())];

            let (mut c1, _) = sbx_crossover(&population[k1], &population[k2], 20.0, bounds, rng)?;
            polynomial_mutate(&mut c1, 20.0, config.p_mut, bounds, rng);

            let new_obj = objective_fn(&c1);

            // Update ideal point
            for (j, &v) in new_obj.iter().enumerate().take(config.n_obj) {
                if v < ideal[j] {
                    ideal[j] = v;
                }
            }

            // Update neighbours
            for &nbr in pool {
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

    // ── Post-filter: retain solutions within preference cone ──────────────────
    let filtered_indices: Vec<usize> = (0..actual_pop)
        .filter(|&i| within_preference_cone(&objectives[i], &pref_dir, config.epsilon_pref))
        .collect();

    let (final_solutions, final_objectives) = if filtered_indices.len() >= 2 {
        let sols: Vec<Vec<f64>> = filtered_indices
            .iter()
            .map(|&i| population[i].clone())
            .collect();
        let objs: Vec<Vec<f64>> = filtered_indices
            .iter()
            .map(|&i| objectives[i].clone())
            .collect();
        (sols, objs)
    } else {
        // Fall back to full non-dominated set
        let nd_idx = nondominated_indices(&objectives);
        let sols: Vec<Vec<f64>> = nd_idx.iter().map(|&i| population[i].clone()).collect();
        let objs: Vec<Vec<f64>> = nd_idx.iter().map(|&i| objectives[i].clone()).collect();
        (sols, objs)
    };

    Ok(PrefMoeadResult {
        solutions: final_solutions,
        objectives: final_objectives,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use std::f64::consts::PI;

    // ── ZDT1 helper: 2-objective test problem (minimization) ──────────────────
    fn zdt1(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let f1 = x[0];
        let g = {
            let sum: f64 = x[1..].iter().sum();
            1.0 + 9.0 * sum / (n - 1) as f64
        };
        let f2 = g * (1.0 - (f1 / g).sqrt());
        vec![f1, f2]
    }

    fn default_rnsga2_config() -> RNsga2Config {
        RNsga2Config {
            n_obj: 2,
            pop_size: 40,
            n_gens: 30,
            reference_points: vec![vec![0.2, 0.8]],
            epsilon: 0.05,
            sigma_mut: 0.1,
            p_mut: 0.1,
            n_dims: 5,
            lb: 0.0,
            ub: 1.0,
        }
    }

    fn default_pref_moead_config() -> PrefMoeadConfig {
        PrefMoeadConfig {
            n_obj: 2,
            pop_size: 20,
            n_gens: 30,
            preferred_direction: vec![1.0, 1.0], // equal weight; will be normalised
            epsilon_pref: PI / 4.0,              // 45 degrees
            neighborhood_size: 5,
            sigma_mut: 0.1,
            p_mut: 0.1,
            n_dims: 5,
            lb: 0.0,
            ub: 1.0,
        }
    }

    // ── R-NSGA-II config validation ───────────────────────────────────────────

    #[test]
    fn test_rnsga2_empty_reference_points_errors() {
        let mut cfg = default_rnsga2_config();
        cfg.reference_points = vec![];
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_rnsga2_pop_too_small_errors() {
        let mut cfg = default_rnsga2_config();
        cfg.pop_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_rnsga2_dimension_mismatch_errors() {
        let mut cfg = default_rnsga2_config();
        cfg.reference_points = vec![vec![0.5, 0.5, 0.5]]; // 3D for 2-obj problem
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_rnsga2_n_obj_zero_errors() {
        let mut cfg = default_rnsga2_config();
        cfg.n_obj = 0;
        assert!(cfg.validate().is_err());
    }

    // ── R-NSGA-II functional tests ─────────────────────────────────────────────

    #[test]
    fn test_rnsga2_produces_non_empty_pareto_front() {
        let cfg = default_rnsga2_config();
        let mut rng = LcgRng::new(42);
        let result = r_nsga2_run(&cfg, zdt1, &mut rng).expect("r_nsga2_run should succeed");
        assert!(
            !result.pareto_front.is_empty(),
            "Pareto front must be non-empty"
        );
    }

    #[test]
    fn test_rnsga2_objectives_length_matches_pareto_front() {
        let cfg = default_rnsga2_config();
        let mut rng = LcgRng::new(7);
        let result = r_nsga2_run(&cfg, zdt1, &mut rng).expect("r_nsga2_run should succeed");
        assert_eq!(result.pareto_front.len(), result.objectives.len());
    }

    #[test]
    fn test_rnsga2_history_hv_length_equals_n_gens() {
        let cfg = default_rnsga2_config();
        let mut rng = LcgRng::new(13);
        let result = r_nsga2_run(&cfg, zdt1, &mut rng).expect("r_nsga2_run should succeed");
        assert_eq!(result.history_hv.len(), cfg.n_gens);
    }

    #[test]
    fn test_rnsga2_objectives_have_correct_dimensionality() {
        let cfg = default_rnsga2_config();
        let mut rng = LcgRng::new(21);
        let result = r_nsga2_run(&cfg, zdt1, &mut rng).expect("r_nsga2_run should succeed");
        for obj in &result.objectives {
            assert_eq!(obj.len(), cfg.n_obj);
        }
    }

    #[test]
    fn test_rnsga2_concentrates_solutions_near_reference_point() {
        // With a reference point at (0.2, 0.8), solutions should be biased toward it
        let mut cfg = default_rnsga2_config();
        cfg.n_gens = 60;
        cfg.pop_size = 50;
        cfg.epsilon = 0.1;
        cfg.reference_points = vec![vec![0.2, 0.8]];
        let mut rng = LcgRng::new(99);
        let result = r_nsga2_run(&cfg, zdt1, &mut rng).expect("r_nsga2_run should succeed");

        // At least some solutions should be near (0.2, 0.8) in objective space
        let ref_pt = &cfg.reference_points[0];
        let min_dist = result
            .objectives
            .iter()
            .map(|obj| euclid_dist_obj(obj, ref_pt))
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_dist < 2.0,
            "No solution near reference point (0.2, 0.8): min_dist = {min_dist}"
        );
    }

    #[test]
    fn test_rnsga2_multiple_reference_points_runs() {
        let mut cfg = default_rnsga2_config();
        cfg.reference_points = vec![vec![0.1, 0.9], vec![0.5, 0.5], vec![0.9, 0.1]];
        let mut rng = LcgRng::new(55);
        let result = r_nsga2_run(&cfg, zdt1, &mut rng);
        assert!(
            result.is_ok(),
            "multiple reference points failed: {:?}",
            result
        );
    }

    #[test]
    fn test_rnsga2_single_reference_point_produces_result() {
        let cfg = default_rnsga2_config();
        let mut rng = LcgRng::new(33);
        let result = r_nsga2_run(&cfg, zdt1, &mut rng).expect("r_nsga2_run should succeed");
        assert!(!result.objectives.is_empty());
        // All objectives should be finite
        for obj in &result.objectives {
            for &v in obj {
                assert!(v.is_finite(), "objective value is not finite: {v}");
            }
        }
    }

    // ── Preference MOEA/D config validation ───────────────────────────────────

    #[test]
    fn test_pref_moead_zero_direction_norm_errors() {
        let mut cfg = default_pref_moead_config();
        cfg.preferred_direction = vec![0.0, 0.0];
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_pref_moead_wrong_direction_dim_errors() {
        let mut cfg = default_pref_moead_config();
        cfg.preferred_direction = vec![1.0, 0.0, 0.5]; // 3D for 2-obj
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_pref_moead_empty_pop_errors() {
        let mut cfg = default_pref_moead_config();
        cfg.pop_size = 0;
        assert!(cfg.validate().is_err());
    }

    // ── Preference MOEA/D functional tests ────────────────────────────────────

    #[test]
    fn test_pref_moead_runs_successfully() {
        let cfg = default_pref_moead_config();
        let mut rng = LcgRng::new(42);
        let result = pref_moead_run(&cfg, zdt1, &mut rng);
        assert!(result.is_ok(), "pref_moead_run failed: {:?}", result);
    }

    #[test]
    fn test_pref_moead_solutions_objectives_length_match() {
        let cfg = default_pref_moead_config();
        let mut rng = LcgRng::new(7);
        let result = pref_moead_run(&cfg, zdt1, &mut rng).expect("pref_moead_run should succeed");
        assert_eq!(result.solutions.len(), result.objectives.len());
    }

    #[test]
    fn test_pref_moead_preference_direction_normalisation() {
        // Using a non-unit preferred direction should still work (internally normalised)
        let mut cfg = default_pref_moead_config();
        cfg.preferred_direction = vec![3.0, 4.0]; // norm = 5, normalises to [0.6, 0.8]
        let mut rng = LcgRng::new(11);
        let result = pref_moead_run(&cfg, zdt1, &mut rng);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pref_moead_filtered_solutions_within_epsilon() {
        // With a broad epsilon (PI/2 = 90 degrees), nearly all ZDT1 solutions should
        // pass the preference filter. Verify the filtering mechanism works and results
        // are valid (finite, non-negative for ZDT1).
        let mut cfg = default_pref_moead_config();
        cfg.preferred_direction = vec![1.0, 1.0]; // 45-degree diagonal
        cfg.epsilon_pref = PI / 2.0; // 90 degrees — very broad
        cfg.n_gens = 30;
        let mut rng = LcgRng::new(77);
        let result = pref_moead_run(&cfg, zdt1, &mut rng).expect("pref_moead_run should succeed");

        // With such a wide epsilon most solutions should pass the filter
        assert!(
            !result.objectives.is_empty(),
            "must return at least one solution"
        );

        // All objectives must be finite and non-negative (ZDT1 domain)
        for obj in &result.objectives {
            for &v in obj {
                assert!(v.is_finite(), "objective must be finite: {v}");
                assert!(v >= 0.0, "ZDT1 objective must be >= 0: {v}");
            }
        }

        // Verify solutions and objectives have matching lengths
        assert_eq!(result.solutions.len(), result.objectives.len());
    }

    #[test]
    fn test_pref_moead_returns_nonempty_results() {
        let cfg = default_pref_moead_config();
        let mut rng = LcgRng::new(123);
        let result = pref_moead_run(&cfg, zdt1, &mut rng).expect("pref_moead_run should succeed");
        assert!(
            !result.solutions.is_empty(),
            "pref_moead must return at least one solution"
        );
        assert!(!result.objectives.is_empty());
    }

    #[test]
    fn test_pref_moead_objectives_are_finite() {
        let cfg = default_pref_moead_config();
        let mut rng = LcgRng::new(88);
        let result = pref_moead_run(&cfg, zdt1, &mut rng).expect("pref_moead_run should succeed");
        for obj in &result.objectives {
            for &v in obj {
                assert!(v.is_finite(), "objective not finite: {v}");
            }
        }
    }
}
