//! RVEA: Reference-Vector-Guided Evolutionary Algorithm for many-objective optimisation.
//!
//! Reference: R. Cheng, Y. Jin, M. Olhofer, B. Sendhoff, "A Reference Vector Guided
//! Evolutionary Algorithm for Many-Objective Optimization", IEEE Trans. Evol. Comput.
//! 20(5):773-791, 2016. <https://doi.org/10.1109/TEVC.2016.2519378>
//!
//! ## Overview
//! RVEA partitions the objective space with a set of unit **reference vectors** spread over
//! the positive orthant. Each generation:
//!
//! 1. Offspring are produced with SBX crossover + polynomial mutation and merged with the
//!    parents.
//! 2. Objectives are *translated* by the ideal point so they live in the positive orthant.
//! 3. Each candidate is **associated** with the reference vector of smallest *acute angle*.
//! 4. Within each reference-vector partition, the survivor is the candidate minimising the
//!    **Angle-Penalized Distance (APD)**
//!
//!    ```text
//!    APD(x) = (1 + M · (t/t_max)^α · θ_{x} / γ_{v}) · ‖f'(x)‖
//!    ```
//!
//!    where `M` is the number of objectives, `θ_x` the acute angle between the (translated)
//!    objective vector and its reference vector, and `γ_v` the smallest angle between
//!    reference vector `v` and its neighbours. The `(t/t_max)^α` schedule shifts the balance
//!    from diversity (early, small penalty) to convergence (late, large penalty).
//! 5. Periodically (every `fr·t_max` generations) the reference vectors are **adapted** to the
//!    observed objective ranges, which is what lets RVEA handle disparately-scaled objectives.

#![allow(clippy::needless_range_loop)]

use crate::genetic::crossover::sbx_crossover;
use crate::genetic::mutation::polynomial_mutate;
use crate::multiobjective::nsga3::generate_reference_points;
use crate::{EvolError, EvolResult, handle::LcgRng};

/// Hyper-parameters for an RVEA run.
#[derive(Debug, Clone)]
pub struct RveaConfig {
    /// Number of decision variables.
    pub n_dims: usize,
    /// Number of objectives M.
    pub n_obj: usize,
    /// Number of generations `t_max`.
    pub n_gen: usize,
    /// Number of Das-Dennis divisions H for the initial reference vectors (controls how many
    /// reference vectors / the population size).
    pub divisions: usize,
    /// SBX distribution index.
    pub crossover_eta: f64,
    /// Polynomial-mutation distribution index.
    pub mutation_eta: f64,
    /// Per-gene mutation probability (defaults to `1/n_dims` if set to a negative value).
    pub mutation_prob: f64,
    /// Decision-space bounds shared by all variables.
    pub bounds: (f64, f64),
    /// APD penalty schedule exponent α (paper default 2.0).
    pub alpha: f64,
    /// Reference-vector adaptation frequency `fr ∈ (0, 1]`: vectors are re-scaled every
    /// `⌊fr · n_gen⌋` generations (paper default 0.1).
    pub adapt_freq: f64,
}

impl RveaConfig {
    /// Build a default configuration for an `M`-objective, `n`-dimensional problem.
    pub fn new(n_dims: usize, n_obj: usize, divisions: usize) -> EvolResult<Self> {
        if n_dims == 0 {
            return Err(EvolError::InvalidParameter(
                "n_dims must be >= 1".to_owned(),
            ));
        }
        if n_obj < 2 {
            return Err(EvolError::InvalidParameter("n_obj must be >= 2".to_owned()));
        }
        if divisions == 0 {
            return Err(EvolError::InvalidParameter(
                "divisions must be >= 1".to_owned(),
            ));
        }
        Ok(Self {
            n_dims,
            n_obj,
            n_gen: 200,
            divisions,
            crossover_eta: 20.0,
            mutation_eta: 20.0,
            mutation_prob: -1.0,
            bounds: (0.0, 1.0),
            alpha: 2.0,
            adapt_freq: 0.1,
        })
    }

    fn validate(&self) -> EvolResult<()> {
        if self.n_dims == 0 {
            return Err(EvolError::InvalidParameter(
                "n_dims must be >= 1".to_owned(),
            ));
        }
        if self.n_obj < 2 {
            return Err(EvolError::InvalidParameter("n_obj must be >= 2".to_owned()));
        }
        if self.divisions == 0 {
            return Err(EvolError::InvalidParameter(
                "divisions must be >= 1".to_owned(),
            ));
        }
        if self.bounds.0 >= self.bounds.1 {
            return Err(EvolError::InvalidParameter(
                "bounds: lower must be < upper".to_owned(),
            ));
        }
        if !(0.0..=1.0).contains(&self.adapt_freq) || self.adapt_freq <= 0.0 {
            return Err(EvolError::InvalidParameter(
                "adapt_freq must be in (0, 1]".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Final state of an RVEA run.
#[derive(Debug, Clone)]
pub struct Rvea {
    /// Decision vectors of the final population.
    pub population: Vec<Vec<f64>>,
    /// Objective vectors of the final population.
    pub objectives: Vec<Vec<f64>>,
    /// Unit reference vectors after the last adaptation.
    pub reference_vectors: Vec<Vec<f64>>,
    /// Ideal point (per-objective minimum) at the end of the run.
    pub ideal: Vec<f64>,
}

impl Rvea {
    /// Decision/objective pairs of the final non-dominated front (rank 0).
    pub fn pareto_front(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        let fronts = fast_nondominated_sort(&self.objectives);
        if fronts.is_empty() {
            return Vec::new();
        }
        fronts[0]
            .iter()
            .map(|&i| (self.population[i].clone(), self.objectives[i].clone()))
            .collect()
    }
}

/// Normalise a vector to unit Euclidean length (returns a zero vector unchanged).
fn normalize_unit(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm < 1e-300 {
        return v.to_vec();
    }
    v.iter().map(|&x| x / norm).collect()
}

/// Acute angle (radians) between two vectors via the clamped cosine.
fn angle_between(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if na < 1e-300 || nb < 1e-300 {
        return std::f64::consts::FRAC_PI_2;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0).acos()
}

/// For each reference vector, the smallest angle to any *other* reference vector (`γ_v`).
fn smallest_neighbour_angles(refs: &[Vec<f64>]) -> Vec<f64> {
    let k = refs.len();
    let mut gamma = vec![std::f64::consts::FRAC_PI_2; k];
    for i in 0..k {
        let mut min_ang = f64::INFINITY;
        for j in 0..k {
            if i == j {
                continue;
            }
            let ang = angle_between(&refs[i], &refs[j]);
            if ang < min_ang {
                min_ang = ang;
            }
        }
        if min_ang.is_finite() {
            gamma[i] = min_ang.max(1e-6);
        }
    }
    gamma
}

/// `dominates(a, b)` — true when `a` Pareto-dominates `b` (minimisation).
fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut strictly = false;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            return false;
        }
        if ai < bi {
            strictly = true;
        }
    }
    strictly
}

/// Fast non-dominated sort returning fronts of indices (front 0 = Pareto front).
fn fast_nondominated_sort(objectives: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let n = objectives.len();
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut dom_count = vec![0usize; n];
    let mut fronts: Vec<Vec<usize>> = vec![Vec::new()];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if dominates(&objectives[i], &objectives[j]) {
                dominated_by[i].push(j);
            } else if dominates(&objectives[j], &objectives[i]) {
                dom_count[i] += 1;
            }
        }
        if dom_count[i] == 0 {
            fronts[0].push(i);
        }
    }
    let mut fi = 0;
    while !fronts[fi].is_empty() {
        let mut next = Vec::new();
        for &i in &fronts[fi] {
            for &j in &dominated_by[i] {
                dom_count[j] = dom_count[j].saturating_sub(1);
                if dom_count[j] == 0 {
                    next.push(j);
                }
            }
        }
        fi += 1;
        if next.is_empty() {
            break;
        }
        fronts.push(next);
    }
    fronts
}

/// Binary-tournament parent selection over the population indices (random, since RVEA's
/// elitism happens entirely in the APD environmental-selection step).
fn random_parent(pop_len: usize, rng: &mut LcgRng) -> usize {
    rng.next_usize(pop_len)
}

/// Run RVEA, returning the final population, objectives, and reference vectors.
///
/// `objective` maps a decision vector (length `n_dims`) to an objective vector (length
/// `n_obj`). Decision variables are confined to `cfg.bounds`. Objectives are *minimised*.
///
/// # Errors
/// Returns `EvolError` on invalid configuration or if `objective` returns a vector whose
/// length differs from `cfg.n_obj`.
pub fn rvea_run<F>(objective: F, cfg: &RveaConfig, rng: &mut LcgRng) -> EvolResult<Rvea>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    cfg.validate()?;
    let m = cfg.n_obj;
    let n = cfg.n_dims;
    let (lb, ub) = cfg.bounds;
    let range = ub - lb;

    // ── Initial unit reference vectors V₀ (Das-Dennis simplex, normalised) ────
    let ref_points = generate_reference_points(m, cfg.divisions);
    let ref0: Vec<Vec<f64>> = ref_points.iter().map(|p| normalize_unit(p)).collect();
    let mut refs = ref0.clone();
    let pop_size = ref0.len().max(m + 1);

    let mut population: Vec<Vec<f64>> = (0..pop_size)
        .map(|_| (0..n).map(|_| lb + rng.next_f64() * range).collect())
        .collect();
    let mut objectives: Vec<Vec<f64>> = Vec::with_capacity(pop_size);
    for x in &population {
        let f = objective(x);
        if f.len() != m {
            return Err(EvolError::ObjectiveCountMismatch);
        }
        objectives.push(f);
    }

    let mut_prob = if cfg.mutation_prob < 0.0 {
        1.0 / n as f64
    } else {
        cfg.mutation_prob
    };
    let adapt_interval = ((cfg.adapt_freq * cfg.n_gen as f64).floor() as usize).max(1);

    let mut ideal = column_min(&objectives, m);

    for t in 0..cfg.n_gen {
        // ── Offspring generation (SBX + polynomial mutation) ──────────────────
        let mut offspring: Vec<Vec<f64>> = Vec::with_capacity(pop_size);
        while offspring.len() < pop_size {
            let p1 = random_parent(pop_size, rng);
            let mut p2 = random_parent(pop_size, rng);
            if p2 == p1 {
                p2 = (p2 + 1) % pop_size;
            }
            let (mut c1, mut c2) = sbx_crossover(
                &population[p1],
                &population[p2],
                cfg.crossover_eta,
                cfg.bounds,
                rng,
            )?;
            polynomial_mutate(&mut c1, cfg.mutation_eta, mut_prob, cfg.bounds, rng);
            polynomial_mutate(&mut c2, cfg.mutation_eta, mut_prob, cfg.bounds, rng);
            offspring.push(c1);
            if offspring.len() < pop_size {
                offspring.push(c2);
            }
        }

        // Evaluate offspring and merge.
        let mut combined_pop = population.clone();
        let mut combined_obj = objectives.clone();
        for x in &offspring {
            let f = objective(x);
            if f.len() != m {
                return Err(EvolError::ObjectiveCountMismatch);
            }
            combined_obj.push(f);
            combined_pop.push(x.clone());
        }

        // ── Update ideal point and translate objectives ───────────────────────
        ideal = column_min(&combined_obj, m);
        let translated: Vec<Vec<f64>> = combined_obj
            .iter()
            .map(|f| (0..m).map(|k| (f[k] - ideal[k]).max(0.0)).collect())
            .collect();

        // ── APD environmental selection ───────────────────────────────────────
        let gamma = smallest_neighbour_angles(&refs);
        let theta_scale = ((t as f64 + 1.0) / cfg.n_gen.max(1) as f64).powf(cfg.alpha) * m as f64;

        // Associate each candidate to its nearest reference vector and bucket it.
        let n_refs = refs.len();
        let mut buckets: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_refs];
        for (idx, tf) in translated.iter().enumerate() {
            // Skip candidates that collapsed onto the ideal point (zero vector): give them
            // the largest possible penalty so they only survive if a partition is otherwise
            // empty.
            let unit = normalize_unit(tf);
            let mut best_ref = 0usize;
            let mut best_ang = f64::INFINITY;
            for (r, v) in refs.iter().enumerate() {
                let ang = angle_between(&unit, v);
                if ang < best_ang {
                    best_ang = ang;
                    best_ref = r;
                }
            }
            buckets[best_ref].push((idx, best_ang));
        }

        let mut survivors: Vec<usize> = Vec::with_capacity(n_refs);
        for r in 0..n_refs {
            if buckets[r].is_empty() {
                continue;
            }
            // Choose the candidate minimising APD in this partition.
            let mut best_idx = buckets[r][0].0;
            let mut best_apd = f64::INFINITY;
            for &(idx, ang) in &buckets[r] {
                let dist: f64 = translated[idx].iter().map(|&x| x * x).sum::<f64>().sqrt();
                let apd = (1.0 + theta_scale * ang / gamma[r]) * dist;
                if apd < best_apd {
                    best_apd = apd;
                    best_idx = idx;
                }
            }
            survivors.push(best_idx);
        }

        // If association left us with fewer than pop_size survivors (empty partitions),
        // top up with the best remaining candidates by non-dominated rank to keep the
        // population size stable.
        if survivors.len() < pop_size {
            let chosen: std::collections::HashSet<usize> = survivors.iter().copied().collect();
            let fronts = fast_nondominated_sort(&combined_obj);
            'fill: for front in &fronts {
                for &i in front {
                    if survivors.len() >= pop_size {
                        break 'fill;
                    }
                    if !chosen.contains(&i) {
                        survivors.push(i);
                    }
                }
            }
        }
        survivors.truncate(pop_size);

        population = survivors.iter().map(|&i| combined_pop[i].clone()).collect();
        objectives = survivors.iter().map(|&i| combined_obj[i].clone()).collect();

        // ── Periodic reference-vector adaptation ──────────────────────────────
        if (t + 1) % adapt_interval == 0 {
            let z_min = column_min(&objectives, m);
            let z_max = column_max(&objectives, m);
            refs = adapt_reference_vectors(&ref0, &z_min, &z_max);
        }
    }

    Ok(Rvea {
        population,
        objectives,
        reference_vectors: refs,
        ideal,
    })
}

/// Per-objective minimum across a set of objective vectors.
fn column_min(objs: &[Vec<f64>], m: usize) -> Vec<f64> {
    let mut out = vec![f64::INFINITY; m];
    for f in objs {
        for k in 0..m {
            if f[k] < out[k] {
                out[k] = f[k];
            }
        }
    }
    for v in out.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    out
}

/// Per-objective maximum across a set of objective vectors.
fn column_max(objs: &[Vec<f64>], m: usize) -> Vec<f64> {
    let mut out = vec![f64::NEG_INFINITY; m];
    for f in objs {
        for k in 0..m {
            if f[k] > out[k] {
                out[k] = f[k];
            }
        }
    }
    for v in out.iter_mut() {
        if !v.is_finite() {
            *v = 1.0;
        }
    }
    out
}

/// Re-scale the initial reference vectors to the observed objective ranges (Cheng 2016 eq. 11):
/// `v'_i = (V₀_i ⊙ (z_max − z_min)) / ‖V₀_i ⊙ (z_max − z_min)‖`.
fn adapt_reference_vectors(ref0: &[Vec<f64>], z_min: &[f64], z_max: &[f64]) -> Vec<Vec<f64>> {
    let m = z_min.len();
    let spread: Vec<f64> = (0..m).map(|k| (z_max[k] - z_min[k]).max(1e-12)).collect();
    ref0.iter()
        .map(|v| {
            let scaled: Vec<f64> = (0..m).map(|k| v[k] * spread[k]).collect();
            normalize_unit(&scaled)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Two-objective convex front: minimise (x², (x−2)²) over x ∈ [0, 2] (n=1 toy).
    // Pareto front in objective space is the curve {(t², (t−2)²) : t ∈ [0,2]}.
    fn biobjective(x: &[f64]) -> Vec<f64> {
        let t = x[0];
        vec![t * t, (t - 2.0) * (t - 2.0)]
    }

    // ZDT1-style: f1 = x0, f2 = g(1 - sqrt(f1/g)) with convex front at g=1.
    fn zdt1(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        let f1 = x[0];
        let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (n - 1).max(1) as f64;
        let f2 = g * (1.0 - (f1 / g).max(0.0).sqrt());
        vec![f1, f2]
    }

    // DTLZ2 (3-objective): concave spherical front of radius 1.
    fn dtlz2_3obj(x: &[f64]) -> Vec<f64> {
        let k = x.len() - 2;
        let g: f64 = x[2..].iter().map(|&xi| (xi - 0.5).powi(2)).sum::<f64>();
        let _ = k;
        let half_pi = std::f64::consts::FRAC_PI_2;
        let f1 = (1.0 + g) * (x[0] * half_pi).cos() * (x[1] * half_pi).cos();
        let f2 = (1.0 + g) * (x[0] * half_pi).cos() * (x[1] * half_pi).sin();
        let f3 = (1.0 + g) * (x[0] * half_pi).sin();
        vec![f1, f2, f3]
    }

    #[test]
    fn config_rejects_bad_params() {
        assert!(RveaConfig::new(0, 2, 4).is_err());
        assert!(RveaConfig::new(3, 1, 4).is_err());
        assert!(RveaConfig::new(3, 2, 0).is_err());
    }

    #[test]
    fn normalize_unit_is_unit_length() {
        let u = normalize_unit(&[3.0, 4.0]);
        let norm: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn angle_between_orthogonal_is_half_pi() {
        let a = angle_between(&[1.0, 0.0], &[0.0, 1.0]);
        assert!((a - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        let b = angle_between(&[1.0, 1.0], &[1.0, 1.0]);
        assert!(b.abs() < 1e-7);
    }

    #[test]
    fn dominates_basic() {
        assert!(dominates(&[0.0, 0.0], &[1.0, 1.0]));
        assert!(!dominates(&[1.0, 0.0], &[0.0, 1.0]));
        assert!(!dominates(&[1.0, 1.0], &[1.0, 1.0]));
    }

    #[test]
    fn adapt_reference_vectors_rescales_and_normalises() {
        let ref0 = vec![normalize_unit(&[1.0, 0.0]), normalize_unit(&[0.0, 1.0])];
        let adapted = adapt_reference_vectors(&ref0, &[0.0, 0.0], &[10.0, 1.0]);
        for v in &adapted {
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn biobjective_front_is_recovered() {
        let mut cfg = RveaConfig::new(1, 2, 20).expect("ok");
        cfg.bounds = (0.0, 2.0);
        cfg.n_gen = 100;
        let mut rng = LcgRng::new(42);
        let res = rvea_run(biobjective, &cfg, &mut rng).expect("ok");
        let front = res.pareto_front();
        assert!(!front.is_empty(), "front must be non-empty");
        // Every front point must satisfy the analytic relation √f1 + √f2 ≈ 2
        // (since f1 = t², f2 = (2−t)² ⇒ √f1 + √f2 = t + (2−t) = 2 for t ∈ [0,2]).
        let mut good = 0;
        for (_x, f) in &front {
            let rel = f[0].sqrt() + f[1].sqrt();
            if (rel - 2.0).abs() < 0.1 {
                good += 1;
            }
        }
        assert!(
            good >= (front.len() * 8) / 10,
            "most front points must lie on √f1+√f2=2; {good}/{} did",
            front.len()
        );
    }

    #[test]
    fn zdt1_front_converges_near_true_front() {
        let mut cfg = RveaConfig::new(6, 2, 30).expect("ok");
        cfg.bounds = (0.0, 1.0);
        cfg.n_gen = 200;
        let mut rng = LcgRng::new(7);
        let res = rvea_run(zdt1, &cfg, &mut rng).expect("ok");
        let front = res.pareto_front();
        assert!(!front.is_empty());
        // True ZDT1 front: f2 = 1 - sqrt(f1). Measure mean vertical gap above it.
        let mut sum_gap = 0.0;
        let mut count = 0;
        for (_x, f) in &front {
            if (0.0..=1.0).contains(&f[0]) {
                let f2_true = 1.0 - f[0].sqrt();
                sum_gap += (f[1] - f2_true).max(0.0);
                count += 1;
            }
        }
        assert!(count > 0);
        let mean_gap = sum_gap / count as f64;
        assert!(
            mean_gap < 0.15,
            "RVEA should approach the ZDT1 front (mean gap {mean_gap} < 0.15)"
        );
    }

    #[test]
    fn dtlz2_three_objective_runs_and_spreads() {
        let mut cfg = RveaConfig::new(7, 3, 8).expect("ok");
        cfg.bounds = (0.0, 1.0);
        cfg.n_gen = 150;
        let mut rng = LcgRng::new(11);
        let res = rvea_run(dtlz2_3obj, &cfg, &mut rng).expect("ok");
        let front = res.pareto_front();
        assert!(front.len() >= 3, "3-obj front should retain several points");
        // DTLZ2 front: f1²+f2²+f3² ≈ 1. Check the radius is close to 1 for most points.
        let mut good = 0;
        for (_x, f) in &front {
            let r2: f64 = f.iter().map(|v| v * v).sum();
            if (r2.sqrt() - 1.0).abs() < 0.2 {
                good += 1;
            }
        }
        assert!(
            good >= front.len() / 2,
            "at least half the front should lie near the unit sphere ({good}/{})",
            front.len()
        );
    }

    #[test]
    fn objective_count_mismatch_errors() {
        let bad = |_x: &[f64]| vec![0.0]; // returns 1 obj but cfg says 2
        let cfg = RveaConfig::new(2, 2, 4).expect("ok");
        let mut rng = LcgRng::new(1);
        assert!(rvea_run(bad, &cfg, &mut rng).is_err());
    }
}
