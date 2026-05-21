//! Multi-Objective Particle Swarm Optimization (MOPSO).
//!
//! Reference: C. A. Coello Coello, G. T. Pulido & M. S. Lechuga,
//! "Handling Multiple Objectives with Particle Swarm Optimization",
//! IEEE Transactions on Evolutionary Computation 8(3):256-279, 2004.
//!
//! # Algorithm overview
//!
//! Particles are evaluated on `n_obj` objectives simultaneously.  An external archive
//! stores all non-dominated solutions found so far, bounded to `max_archive` entries.
//! When the archive exceeds its capacity, the most crowded cell in a hypergrid covering
//! the current Pareto front is pruned.  Leaders for the velocity update are selected from
//! the archive proportionally to the *inverse* of the grid cell population (prefer less
//! crowded regions of objective space).

#![allow(clippy::needless_range_loop)]

use crate::{EvolError, EvolResult, handle::LcgRng};

/// MOPSO hyper-parameters.
#[derive(Debug, Clone)]
pub struct MopsoConfig {
    /// Number of particles.
    pub n_particles: usize,
    /// Number of generations.
    pub n_gen: usize,
    /// Number of objectives.
    pub n_obj: usize,
    /// Number of grid divisions per objective axis for the hypergrid.
    pub grid_divisions: usize,
    /// Maximum archive size (non-dominated repository).
    pub max_archive: usize,
    /// Inertia weight.
    pub w: f64,
    /// Cognitive acceleration (particle toward its personal best).
    pub c1: f64,
    /// Social acceleration (particle toward archive leader).
    pub c2: f64,
    /// Random seed.
    pub seed: u64,
}

impl MopsoConfig {
    /// Construct `MopsoConfig` with sensible defaults.
    pub fn new(n_particles: usize, n_gen: usize, n_obj: usize) -> EvolResult<Self> {
        if n_particles == 0 {
            return Err(EvolError::SwarmEmpty);
        }
        if n_gen == 0 {
            return Err(EvolError::InvalidParameter("n_gen must be >= 1".to_owned()));
        }
        if n_obj < 2 {
            return Err(EvolError::InvalidParameter(
                "n_obj must be >= 2 for multi-objective".to_owned(),
            ));
        }
        Ok(Self {
            n_particles,
            n_gen,
            n_obj,
            grid_divisions: 10,
            max_archive: 100,
            w: 0.4,
            c1: 2.0,
            c2: 2.0,
            seed: 0,
        })
    }
}

/// Mutable MOPSO state.
pub struct MopsoState {
    /// Current particle positions (n_particles × n_var).
    pub positions: Vec<Vec<f64>>,
    /// Current particle velocities (n_particles × n_var).
    pub velocities: Vec<Vec<f64>>,
    /// Personal best positions (n_particles × n_var).
    pub pbest: Vec<Vec<f64>>,
    /// Objective values at personal best (n_particles × n_obj).
    pub pbest_obj: Vec<Vec<f64>>,
    /// Archive of non-dominated decision vectors (archive_size × n_var).
    pub archive: Vec<Vec<f64>>,
    /// Objective values of archive members (archive_size × n_obj).
    pub archive_obj: Vec<Vec<f64>>,
    /// Number of completed generations.
    pub generation: usize,
}

// ── Dominance helpers ─────────────────────────────────────────────────────────

/// Returns `true` if `a` dominates `b` (all objectives ≤, at least one <).
#[inline]
fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut any_strictly_less = false;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            return false;
        }
        if ai < bi {
            any_strictly_less = true;
        }
    }
    any_strictly_less
}

// ── Hypergrid helpers ─────────────────────────────────────────────────────────

/// Compute the axis-aligned bounding box of a set of objective vectors.
///
/// Returns `(min_per_obj, max_per_obj)`.
fn objective_bounds(obj_vecs: &[Vec<f64>], n_obj: usize) -> (Vec<f64>, Vec<f64>) {
    let mut lo = vec![f64::INFINITY; n_obj];
    let mut hi = vec![f64::NEG_INFINITY; n_obj];
    for v in obj_vecs {
        for k in 0..n_obj {
            if v[k] < lo[k] {
                lo[k] = v[k];
            }
            if v[k] > hi[k] {
                hi[k] = v[k];
            }
        }
    }
    (lo, hi)
}

/// Map an objective vector to a flat grid cell index.
///
/// Each objective axis is discretised into `grid_divisions` equal-width bins.
fn grid_cell_index(obj: &[f64], lo: &[f64], hi: &[f64], grid_divisions: usize) -> usize {
    let mut idx = 0_usize;
    let n_obj = obj.len();
    for k in 0..n_obj {
        let range = (hi[k] - lo[k]).max(1e-300);
        let bin = ((obj[k] - lo[k]) / range * grid_divisions as f64)
            .floor()
            .clamp(0.0, (grid_divisions - 1) as f64) as usize;
        idx = idx * grid_divisions + bin;
    }
    idx
}

/// Compute grid cell populations for the current archive.
///
/// Returns `(cell_counts, cell_indices_per_member)`.
fn compute_grid_density(
    archive_obj: &[Vec<f64>],
    n_obj: usize,
    grid_divisions: usize,
) -> (Vec<usize>, Vec<usize>) {
    if archive_obj.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let (lo, hi) = objective_bounds(archive_obj, n_obj);
    let n_cells = grid_divisions.pow(n_obj as u32);
    let mut counts = vec![0_usize; n_cells];
    let mut member_cells: Vec<usize> = Vec::with_capacity(archive_obj.len());

    for obj in archive_obj {
        let cell = grid_cell_index(obj, &lo, &hi, grid_divisions);
        counts[cell] += 1;
        member_cells.push(cell);
    }

    (counts, member_cells)
}

/// Select a leader from the archive via inverse-density roulette on the hypergrid.
///
/// Members in less-crowded grid cells have higher selection probability:
/// weight_i = 1 / (count_of_cell(i) + 1).
///
/// Returns the archive index of the chosen leader.  Falls back to index 0 if the
/// archive is empty (caller must guard against this).
fn select_leader(
    archive_obj: &[Vec<f64>],
    n_obj: usize,
    grid_divisions: usize,
    rng: &mut LcgRng,
) -> usize {
    let n = archive_obj.len();
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 0;
    }

    let (counts, member_cells) = compute_grid_density(archive_obj, n_obj, grid_divisions);

    // Weight = 1 / (cell_count + 1) for each archive member.
    let weights: Vec<f64> = member_cells
        .iter()
        .map(|&c| 1.0 / (counts[c] + 1) as f64)
        .collect();
    let total: f64 = weights.iter().sum::<f64>().max(1e-300);

    let r = rng.next_f64() * total;
    let mut cum = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cum += w;
        if cum >= r {
            return i;
        }
    }
    n - 1
}

/// Update the archive with a new solution `(new_x, new_obj)`.
///
/// 1. If `new_obj` is dominated by any archive member, discard it.
/// 2. Otherwise, remove any archive members dominated by `new_obj` and add the new one.
/// 3. If the archive exceeds `max_archive`, prune the most crowded grid cell.
pub fn update_archive(
    archive: &mut Vec<Vec<f64>>,
    archive_obj: &mut Vec<Vec<f64>>,
    new_x: Vec<f64>,
    new_obj: Vec<f64>,
    max_archive: usize,
    n_obj: usize,
    grid_divisions: usize,
) {
    // Check if new solution is dominated by any current archive member.
    for existing_obj in archive_obj.iter() {
        if dominates(existing_obj, &new_obj) {
            return; // dominated; discard
        }
    }

    // Remove archive members that are dominated by the new solution.
    let mut to_remove: Vec<usize> = archive_obj
        .iter()
        .enumerate()
        .filter_map(|(i, obj)| {
            if dominates(&new_obj, obj) {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    // Remove in reverse order to preserve indices.
    to_remove.sort_unstable_by(|a, b| b.cmp(a));
    for i in to_remove {
        archive.remove(i);
        archive_obj.remove(i);
    }

    // Add new solution.
    archive.push(new_x);
    archive_obj.push(new_obj);

    // Prune if archive exceeds capacity.
    while archive.len() > max_archive {
        prune_most_crowded(archive, archive_obj, n_obj, grid_divisions);
    }
}

/// Remove the member from the most crowded grid cell.
fn prune_most_crowded(
    archive: &mut Vec<Vec<f64>>,
    archive_obj: &mut Vec<Vec<f64>>,
    n_obj: usize,
    grid_divisions: usize,
) {
    if archive.is_empty() {
        return;
    }
    let (counts, member_cells) = compute_grid_density(archive_obj, n_obj, grid_divisions);

    // Find the archive member with the highest cell count; break ties by highest index.
    let victim = member_cells
        .iter()
        .enumerate()
        .max_by(|&(ia, &ca), &(ib, &cb)| counts[ca].cmp(&counts[cb]).then(ia.cmp(&ib)))
        .map(|(i, _)| i)
        .unwrap_or(0);

    archive.remove(victim);
    archive_obj.remove(victim);
}

/// Run MOPSO.
///
/// `fitness_fn` maps a decision vector to a `Vec<f64>` of `n_obj` objective values
/// (all minimised).  `bounds` must have length `n_var`.
///
/// Returns the final `MopsoState`; use `mopso_pareto_front` to extract the archive.
pub fn mopso_run<F>(
    fitness_fn: F,
    n_var: usize,
    bounds: &[(f64, f64)],
    cfg: &MopsoConfig,
) -> EvolResult<MopsoState>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if bounds.len() != n_var {
        return Err(EvolError::DimensionMismatch {
            expected: n_var,
            got: bounds.len(),
        });
    }
    if bounds.is_empty() {
        return Err(EvolError::InvalidParameter("bounds is empty".to_owned()));
    }
    if cfg.n_particles == 0 {
        return Err(EvolError::SwarmEmpty);
    }
    if cfg.n_obj < 2 {
        return Err(EvolError::InvalidParameter("n_obj must be >= 2".to_owned()));
    }

    let mut rng = LcgRng::new(cfg.seed);

    // ── Initialise particles ──────────────────────────────────────────────────
    let positions: Vec<Vec<f64>> = (0..cfg.n_particles)
        .map(|_| {
            (0..n_var)
                .map(|d| {
                    let (lb, ub) = bounds[d];
                    lb + rng.next_f64() * (ub - lb)
                })
                .collect()
        })
        .collect();

    // Initial velocities: small random fractions of the range.
    let velocities: Vec<Vec<f64>> = (0..cfg.n_particles)
        .map(|_| {
            (0..n_var)
                .map(|d| {
                    let (lb, ub) = bounds[d];
                    (rng.next_f64() * 2.0 - 1.0) * 0.1 * (ub - lb)
                })
                .collect()
        })
        .collect();

    let pbest_obj: Vec<Vec<f64>> = positions.iter().map(|x| fitness_fn(x)).collect();
    let pbest = positions.clone();

    // Validate objective dimension.
    for (i, obj) in pbest_obj.iter().enumerate() {
        if obj.len() != cfg.n_obj {
            return Err(EvolError::DimensionMismatch {
                expected: cfg.n_obj,
                got: obj.len(),
            });
        }
        let _ = i;
    }

    let mut state = MopsoState {
        positions,
        velocities,
        pbest,
        pbest_obj,
        archive: Vec::new(),
        archive_obj: Vec::new(),
        generation: 0,
    };

    // Populate initial archive.
    for i in 0..cfg.n_particles {
        let x = state.pbest[i].clone();
        let obj = state.pbest_obj[i].clone();
        update_archive(
            &mut state.archive,
            &mut state.archive_obj,
            x,
            obj,
            cfg.max_archive,
            cfg.n_obj,
            cfg.grid_divisions,
        );
    }

    // ── Main loop ─────────────────────────────────────────────────────────────
    for _ in 0..cfg.n_gen {
        mopso_step(&mut state, &fitness_fn, bounds, cfg, &mut rng);
    }

    Ok(state)
}

/// Execute one MOPSO generation.
fn mopso_step<F: Fn(&[f64]) -> Vec<f64>>(
    state: &mut MopsoState,
    fitness_fn: &F,
    bounds: &[(f64, f64)],
    cfg: &MopsoConfig,
    rng: &mut LcgRng,
) {
    let n_particles = state.positions.len();
    let n_var = bounds.len();

    for i in 0..n_particles {
        // Select leader from archive (or use pbest if archive is empty).
        let leader: Vec<f64> = if state.archive.is_empty() {
            state.pbest[i].clone()
        } else {
            let leader_idx = select_leader(&state.archive_obj, cfg.n_obj, cfg.grid_divisions, rng);
            state.archive[leader_idx].clone()
        };

        // Velocity and position update.
        let r1 = rng.next_f64();
        let r2 = rng.next_f64();

        for d in 0..n_var {
            let (lb, ub) = bounds[d];
            let v_max = 0.2 * (ub - lb);

            let v_new = cfg.w * state.velocities[i][d]
                + cfg.c1 * r1 * (state.pbest[i][d] - state.positions[i][d])
                + cfg.c2 * r2 * (leader[d] - state.positions[i][d]);

            state.velocities[i][d] = v_new.clamp(-v_max, v_max);
            state.positions[i][d] = (state.positions[i][d] + state.velocities[i][d]).clamp(lb, ub);
        }

        // Evaluate objectives at new position.
        let new_obj = fitness_fn(&state.positions[i]);

        // Update personal best if new position Pareto-dominates previous pbest.
        if dominates(&new_obj, &state.pbest_obj[i]) {
            state.pbest[i] = state.positions[i].clone();
            state.pbest_obj[i] = new_obj.clone();
        } else if !dominates(&state.pbest_obj[i], &new_obj) {
            // Non-dominated each other: randomly update pbest with 50% probability.
            if rng.next_bool() {
                state.pbest[i] = state.positions[i].clone();
                state.pbest_obj[i] = new_obj.clone();
            }
        }

        // Update archive with new position.
        update_archive(
            &mut state.archive,
            &mut state.archive_obj,
            state.positions[i].clone(),
            new_obj,
            cfg.max_archive,
            cfg.n_obj,
            cfg.grid_divisions,
        );
    }

    state.generation += 1;
}

/// Extract the Pareto front from the final MOPSO state.
///
/// Returns a `Vec` of `(decision_vector, objective_vector)` pairs for each archive member.
pub fn mopso_pareto_front(state: &MopsoState) -> Vec<(Vec<f64>, Vec<f64>)> {
    state
        .archive
        .iter()
        .cloned()
        .zip(state.archive_obj.iter().cloned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Multi-objective test functions ────────────────────────────────────────

    /// ZDT1: n_var decision variables, 2 objectives.
    /// Pareto front: f1 ∈ [0,1], f2 = 1 - √f1.
    fn zdt1(x: &[f64]) -> Vec<f64> {
        let f1 = x[0];
        let n = x.len() as f64;
        let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (n - 1.0);
        let f2 = g * (1.0 - (f1 / g).sqrt());
        vec![f1, f2]
    }

    /// SCH: 1D decision variable, 2 objectives. Simple bi-objective.
    fn sch(x: &[f64]) -> Vec<f64> {
        let x0 = x[0];
        vec![x0 * x0, (x0 - 2.0) * (x0 - 2.0)]
    }

    /// Bi-sphere: 2 objectives both = sum of squares.  All points dominate/dominated trivially.
    fn bi_sphere(x: &[f64]) -> Vec<f64> {
        let f1: f64 = x.iter().map(|&xi| xi * xi).sum();
        let f2: f64 = x.iter().map(|&xi| (xi - 1.0) * (xi - 1.0)).sum();
        vec![f1, f2]
    }

    // ── Config tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_config_new_valid() {
        let cfg = MopsoConfig::new(30, 100, 2).unwrap();
        assert_eq!(cfg.n_particles, 30);
        assert_eq!(cfg.n_gen, 100);
        assert_eq!(cfg.n_obj, 2);
        assert_eq!(cfg.grid_divisions, 10);
        assert_eq!(cfg.max_archive, 100);
    }

    #[test]
    fn test_config_new_zero_particles() {
        assert!(MopsoConfig::new(0, 100, 2).is_err());
    }

    #[test]
    fn test_config_new_zero_gen() {
        assert!(MopsoConfig::new(10, 0, 2).is_err());
    }

    #[test]
    fn test_config_new_single_objective() {
        assert!(MopsoConfig::new(10, 100, 1).is_err());
    }

    // ── Dominance tests ───────────────────────────────────────────────────────

    #[test]
    fn test_dominates_true() {
        assert!(dominates(&[1.0, 1.0], &[2.0, 2.0]));
        assert!(dominates(&[1.0, 2.0], &[2.0, 2.0]));
    }

    #[test]
    fn test_dominates_false_equal() {
        // Equal vectors: neither dominates.
        assert!(!dominates(&[1.0, 1.0], &[1.0, 1.0]));
    }

    #[test]
    fn test_dominates_false_incomparable() {
        assert!(!dominates(&[1.0, 3.0], &[2.0, 2.0]));
        assert!(!dominates(&[2.0, 2.0], &[1.0, 3.0]));
    }

    // ── Grid helpers tests ────────────────────────────────────────────────────

    #[test]
    fn test_grid_cell_index_boundaries() {
        let lo = vec![0.0, 0.0];
        let hi = vec![1.0, 1.0];
        // (0,0) → cell 0; (1,1) → cell 99 with 10 divisions.
        let cell_origin = grid_cell_index(&[0.0, 0.0], &lo, &hi, 10);
        let cell_corner = grid_cell_index(&[1.0, 1.0], &lo, &hi, 10);
        assert_eq!(cell_origin, 0);
        assert_eq!(cell_corner, 99); // (9*10 + 9)
    }

    #[test]
    fn test_update_archive_adds_nondominated() {
        let mut archive: Vec<Vec<f64>> = Vec::new();
        let mut archive_obj: Vec<Vec<f64>> = Vec::new();
        update_archive(
            &mut archive,
            &mut archive_obj,
            vec![0.5],
            vec![1.0, 2.0],
            100,
            2,
            10,
        );
        assert_eq!(archive.len(), 1);
    }

    #[test]
    fn test_update_archive_rejects_dominated() {
        let mut archive: Vec<Vec<f64>> = vec![vec![0.0]];
        let mut archive_obj: Vec<Vec<f64>> = vec![vec![1.0, 1.0]];
        // New solution dominated by existing.
        update_archive(
            &mut archive,
            &mut archive_obj,
            vec![0.5],
            vec![2.0, 2.0],
            100,
            2,
            10,
        );
        assert_eq!(archive.len(), 1, "dominated solution should not be added");
    }

    #[test]
    fn test_update_archive_removes_dominated_existing() {
        let mut archive: Vec<Vec<f64>> = vec![vec![0.0]];
        let mut archive_obj: Vec<Vec<f64>> = vec![vec![2.0, 2.0]];
        // New solution dominates existing.
        update_archive(
            &mut archive,
            &mut archive_obj,
            vec![0.5],
            vec![1.0, 1.0],
            100,
            2,
            10,
        );
        assert_eq!(archive.len(), 1);
        assert_eq!(archive_obj[0], vec![1.0, 1.0]);
    }

    #[test]
    fn test_update_archive_prunes_at_max() {
        let mut archive: Vec<Vec<f64>> = Vec::new();
        let mut archive_obj: Vec<Vec<f64>> = Vec::new();
        // Add 5 non-dominated solutions (trade-off front: (i, 5-i)).
        for i in 0..5_usize {
            update_archive(
                &mut archive,
                &mut archive_obj,
                vec![i as f64],
                vec![i as f64, (10 - i) as f64],
                4, // max = 4
                2,
                5,
            );
        }
        assert!(
            archive.len() <= 4,
            "archive size {} > max_archive 4",
            archive.len()
        );
    }

    // ── Full run tests ────────────────────────────────────────────────────────

    #[test]
    fn test_mopso_run_sch_archive_nonempty() {
        let bounds = vec![(-4.0_f64, 4.0_f64)];
        let cfg = MopsoConfig {
            n_particles: 20,
            n_gen: 50,
            n_obj: 2,
            grid_divisions: 10,
            max_archive: 50,
            w: 0.4,
            c1: 2.0,
            c2: 2.0,
            seed: 42,
        };
        let state = mopso_run(sch, 1, &bounds, &cfg).unwrap();
        assert!(!state.archive.is_empty(), "archive should be non-empty");
        assert_eq!(state.generation, 50);
    }

    #[test]
    fn test_mopso_run_archive_within_max() {
        let bounds = vec![(0.0_f64, 1.0_f64); 3];
        let cfg = MopsoConfig {
            n_particles: 15,
            n_gen: 30,
            n_obj: 2,
            grid_divisions: 5,
            max_archive: 20,
            w: 0.4,
            c1: 2.0,
            c2: 2.0,
            seed: 7,
        };
        let state = mopso_run(zdt1, 3, &bounds, &cfg).unwrap();
        assert!(
            state.archive.len() <= 20,
            "archive size {} exceeds max_archive 20",
            state.archive.len()
        );
    }

    #[test]
    fn test_mopso_run_archive_nondominated() {
        let bounds = vec![(-2.0_f64, 2.0_f64); 2];
        let cfg = MopsoConfig {
            n_particles: 20,
            n_gen: 40,
            n_obj: 2,
            grid_divisions: 8,
            max_archive: 50,
            w: 0.4,
            c1: 2.0,
            c2: 2.0,
            seed: 11,
        };
        let state = mopso_run(bi_sphere, 2, &bounds, &cfg).unwrap();
        // Verify pairwise non-domination in archive.
        let n = state.archive_obj.len();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert!(
                        !dominates(&state.archive_obj[i], &state.archive_obj[j]),
                        "archive member {i} dominates {j}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_mopso_pareto_front_matches_archive() {
        let bounds = vec![(-1.0_f64, 1.0_f64)];
        let cfg = MopsoConfig {
            n_particles: 10,
            n_gen: 20,
            n_obj: 2,
            grid_divisions: 5,
            max_archive: 30,
            w: 0.4,
            c1: 2.0,
            c2: 2.0,
            seed: 5,
        };
        let state = mopso_run(sch, 1, &bounds, &cfg).unwrap();
        let front = mopso_pareto_front(&state);
        assert_eq!(front.len(), state.archive.len());
        for (i, (x, obj)) in front.iter().enumerate() {
            assert_eq!(x, &state.archive[i]);
            assert_eq!(obj, &state.archive_obj[i]);
        }
    }

    #[test]
    fn test_mopso_run_zdt1_front_quality() {
        // ZDT1: front is f1 ∈ [0,1], f2 = 1 - √f1.
        // After enough iterations, archive members should have low f1+f2.
        let bounds = vec![(0.0_f64, 1.0_f64); 5];
        let cfg = MopsoConfig {
            n_particles: 30,
            n_gen: 100,
            n_obj: 2,
            grid_divisions: 10,
            max_archive: 50,
            w: 0.4,
            c1: 2.0,
            c2: 2.0,
            seed: 33,
        };
        let state = mopso_run(zdt1, 5, &bounds, &cfg).unwrap();
        // At least some archive member should have f1 < 0.9.
        let has_low_f1 = state.archive_obj.iter().any(|obj| obj[0] < 0.9);
        assert!(has_low_f1, "expected low-f1 solutions in archive");
    }

    #[test]
    fn test_mopso_run_deterministic() {
        let bounds = vec![(-2.0_f64, 2.0_f64); 2];
        let cfg = MopsoConfig {
            n_particles: 10,
            n_gen: 20,
            n_obj: 2,
            grid_divisions: 5,
            max_archive: 20,
            w: 0.4,
            c1: 2.0,
            c2: 2.0,
            seed: 999,
        };
        let s1 = mopso_run(bi_sphere, 2, &bounds, &cfg).unwrap();
        let s2 = mopso_run(bi_sphere, 2, &bounds, &cfg).unwrap();
        assert_eq!(
            s1.archive.len(),
            s2.archive.len(),
            "determinism failed: archive sizes differ"
        );
    }

    #[test]
    fn test_mopso_run_error_bounds_mismatch() {
        let bounds = vec![(-1.0_f64, 1.0_f64); 3]; // 3 bounds but n_var = 2
        let cfg = MopsoConfig::new(10, 10, 2).unwrap();
        assert!(mopso_run(bi_sphere, 2, &bounds, &cfg).is_err());
    }

    #[test]
    fn test_mopso_run_generation_count() {
        let bounds = vec![(-1.0_f64, 1.0_f64)];
        let cfg = MopsoConfig {
            n_particles: 5,
            n_gen: 25,
            n_obj: 2,
            grid_divisions: 5,
            max_archive: 20,
            w: 0.4,
            c1: 2.0,
            c2: 2.0,
            seed: 1,
        };
        let state = mopso_run(sch, 1, &bounds, &cfg).unwrap();
        assert_eq!(state.generation, 25);
    }

    #[test]
    fn test_select_leader_returns_valid_index() {
        let archive_obj = vec![vec![0.1, 0.9], vec![0.5, 0.5], vec![0.9, 0.1]];
        let mut rng = LcgRng::new(42);
        for _ in 0..100 {
            let idx = select_leader(&archive_obj, 2, 5, &mut rng);
            assert!(idx < 3, "leader index {idx} out of range");
        }
    }
}
