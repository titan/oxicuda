//! Sinkhorn k-means on a collection of discrete measures (Cuturi & Doucet 2014).
//!
//! Standard Wasserstein k-means uses the exact W_2 distance for assignment and
//! the free-support barycenter for the centroid update. The entropic variant
//! (Sinkhorn k-means) replaces both by their regularised counterparts:
//!
//! - **Assignment**: `c(i) = argmin_k W_{2,ε}(μ_i, centroid_k)` computed via
//!   log-domain Sinkhorn.
//! - **Centroid update**: free-support Sinkhorn barycenter of the assigned cluster
//!   (re-using `barycenter::free_support`).
//!
//! Because the entropic W_2 is differentiable and smoother than the exact W_2,
//! this variant converges faster and is more stable in practice.

use crate::barycenter::free_support::{BaryConfig, free_support_barycenter};
use crate::error::{OtError, OtResult};
use crate::handle::LcgRng;

/// Configuration for Sinkhorn k-means.
#[derive(Debug, Clone)]
pub struct SinkhornKmeansConfig {
    /// Number of clusters `k ≥ 1`.
    pub k: usize,
    /// Maximum number of outer Lloyd iterations.
    pub n_iter: usize,
    /// Entropic regularisation for Sinkhorn assignment distance.
    pub eps: f64,
    /// Inner Sinkhorn iterations for barycenter computation.
    pub bary_iter: usize,
    /// Barycenter inner Sinkhorn convergence tolerance.
    pub bary_tol: f64,
    /// Separate ε for the assignment step (can differ from `eps`).
    pub assignment_eps: f64,
    /// RNG seed for centroid initialisation.
    pub seed: u64,
}

impl Default for SinkhornKmeansConfig {
    fn default() -> Self {
        Self {
            k: 3,
            n_iter: 20,
            eps: 0.1,
            bary_iter: 50,
            bary_tol: 1e-4,
            assignment_eps: 0.1,
            seed: 42,
        }
    }
}

/// Result of Sinkhorn k-means.
#[derive(Debug, Clone)]
pub struct SinkhornKmeansResult {
    /// Cluster assignment index per input measure.
    pub assignments: Vec<usize>,
    /// `(weights, support_points)` per centroid.
    pub centroids: Vec<(Vec<f64>, Vec<Vec<f64>>)>,
    /// Total cost (sum of entropic W_2 distances to assigned centroids).
    pub total_cost: f64,
    /// Total cost per outer iteration.
    pub history: Vec<f64>,
}

// ─────────────────────────────── helpers ────────────────────────────────────

/// Safe log clamped at `log(MIN_POSITIVE)`.
#[inline]
fn safe_ln_f64(x: f64) -> f64 {
    let floor = f64::MIN_POSITIVE;
    if x <= floor { floor.ln() } else { x.ln() }
}

/// Numerically stable log-sum-exp over a slice.
#[inline]
fn log_sum_exp_f64(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_v = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_v.is_finite() {
        return max_v;
    }
    let sum: f64 = vals.iter().map(|&v| (v - max_v).exp()).sum();
    max_v + sum.ln()
}

/// Compute the squared Euclidean cost matrix between two point sets.
///
/// Points are `Vec<Vec<f64>>` (each inner vec is a d-dimensional point).
/// Returns a `n_a × n_b` row-major cost matrix.
fn sq_cost_matrix(pts_a: &[Vec<f64>], pts_b: &[Vec<f64>]) -> Vec<f64> {
    let n_a = pts_a.len();
    let n_b = pts_b.len();
    let mut c = vec![0.0_f64; n_a * n_b];
    for (i, a) in pts_a.iter().enumerate() {
        for (j, b) in pts_b.iter().enumerate() {
            let mut sq = 0.0_f64;
            for (da, db) in a.iter().zip(b.iter()) {
                let diff = da - db;
                sq += diff * diff;
            }
            c[i * n_b + j] = sq;
        }
    }
    c
}

/// Normalise a weight vector to sum to 1; returns as-is if total ≤ 1e-15.
fn renorm(w: &[f64]) -> Vec<f64> {
    let total: f64 = w.iter().sum();
    if total <= 1e-15 {
        return w.to_vec();
    }
    let inv = 1.0 / total;
    w.iter().map(|&v| v * inv).collect()
}

// ─────────────────────────────── Sinkhorn W2 ────────────────────────────────

/// Compute the entropic W_2 distance between two discrete measures via
/// log-domain Sinkhorn, returning the primal transport cost.
///
/// `support_a` and `support_b` are `Vec<Vec<f64>>` point sets (dimension d).
/// `weights_a` and `weights_b` are probability vectors.
pub fn sinkhorn_w2_distance(
    weights_a: &[f64],
    support_a: &[Vec<f64>],
    weights_b: &[f64],
    support_b: &[Vec<f64>],
    eps: f64,
    max_iter: usize,
) -> OtResult<f64> {
    let n_a = weights_a.len();
    let n_b = weights_b.len();
    if n_a == 0 || n_b == 0 {
        return Err(OtError::EmptyInput);
    }
    if n_a != support_a.len() || n_b != support_b.len() {
        return Err(OtError::IncompatibleLength { a: n_a, b: n_b });
    }
    if eps <= 0.0 {
        return Err(OtError::BadEpsilon { eps: eps as f32 });
    }

    let a = renorm(weights_a);
    let b = renorm(weights_b);
    let cost = sq_cost_matrix(support_a, support_b);

    let mut u = a.iter().map(|&v| eps * safe_ln_f64(v)).collect::<Vec<_>>();
    let mut v = b.iter().map(|&v| eps * safe_ln_f64(v)).collect::<Vec<_>>();

    for _ in 0..max_iter {
        // Row update: u_i ← ε*log(a_i) - ε*LSE_j((v_j - c_{ij})/ε)
        for i in 0..n_a {
            let buf: Vec<f64> = (0..n_b).map(|j| (v[j] - cost[i * n_b + j]) / eps).collect();
            let lse = log_sum_exp_f64(&buf);
            u[i] = eps * safe_ln_f64(a[i]) - eps * lse;
        }
        // Column update: v_j ← ε*log(b_j) - ε*LSE_i((u_i - c_{ij})/ε)
        for j in 0..n_b {
            let buf: Vec<f64> = (0..n_a).map(|i| (u[i] - cost[i * n_b + j]) / eps).collect();
            let lse = log_sum_exp_f64(&buf);
            v[j] = eps * safe_ln_f64(b[j]) - eps * lse;
        }
    }

    // Transport cost: Σ_{ij} T_{ij} * c_{ij}
    let mut transport_cost = 0.0_f64;
    for i in 0..n_a {
        for j in 0..n_b {
            let t_ij = ((u[i] + v[j] - cost[i * n_b + j]) / eps).exp();
            transport_cost += t_ij * cost[i * n_b + j];
        }
    }
    Ok(transport_cost)
}

// ─────────────────────────────── initialisation ─────────────────────────────

/// Forgy initialisation: pick k distinct measures as initial centroids.
fn init_centroids_forgy(
    measures: &[(&Vec<f64>, &Vec<Vec<f64>>)],
    k: usize,
    rng: &mut LcgRng,
) -> Vec<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = measures.len();
    let mut chosen = Vec::with_capacity(k);
    let mut used = vec![false; n];
    while chosen.len() < k {
        let idx = rng.next_usize(n);
        if !used[idx] {
            used[idx] = true;
            chosen.push(idx);
        }
    }
    chosen
        .iter()
        .map(|&idx| {
            let (w, s) = measures[idx];
            (renorm(w), s.clone())
        })
        .collect()
}

// ─────────────────────────────── validation ─────────────────────────────────

fn validate_sinkhorn_kmeans(
    measures: &[(&Vec<f64>, &Vec<Vec<f64>>)],
    config: &SinkhornKmeansConfig,
) -> OtResult<()> {
    if measures.is_empty() {
        return Err(OtError::EmptyInput);
    }
    if config.k == 0 {
        return Err(OtError::BadCount { got: 0 });
    }
    if config.k > measures.len() {
        return Err(OtError::BadCount { got: config.k });
    }
    if config.eps <= 0.0 {
        return Err(OtError::BadEpsilon {
            eps: config.eps as f32,
        });
    }
    if config.assignment_eps <= 0.0 {
        return Err(OtError::BadEpsilon {
            eps: config.assignment_eps as f32,
        });
    }

    // Determine reference dimension from first non-empty measure.
    let d = measures
        .iter()
        .find_map(|(_, s)| s.first().map(|pt| pt.len()))
        .ok_or(OtError::EmptyInput)?;

    if d == 0 {
        return Err(OtError::BadDim { got: 0 });
    }

    for (weights, support) in measures.iter() {
        if weights.is_empty() || support.is_empty() {
            return Err(OtError::EmptyInput);
        }
        if weights.len() != support.len() {
            return Err(OtError::IncompatibleLength {
                a: weights.len(),
                b: support.len(),
            });
        }
        for &w in weights.iter() {
            if w < 0.0 || !w.is_finite() {
                return Err(OtError::NegativeWeight);
            }
        }
        for pt in support.iter() {
            if pt.len() != d {
                return Err(OtError::IncompatibleLength { a: d, b: pt.len() });
            }
        }
    }
    Ok(())
}

// ─────────────────────────────── core ───────────────────────────────────────

/// Run Sinkhorn k-means on a collection of discrete probability measures.
///
/// Each measure is represented as `(weights, support_points)` where
/// `weights` is a probability vector and `support_points[j]` is a
/// d-dimensional point for the j-th atom.
pub fn sinkhorn_kmeans(
    measures: &[(&Vec<f64>, &Vec<Vec<f64>>)],
    config: &SinkhornKmeansConfig,
) -> OtResult<SinkhornKmeansResult> {
    validate_sinkhorn_kmeans(measures, config)?;

    let n_meas = measures.len();
    let k = config.k;

    let mut rng = LcgRng::new(config.seed);

    // Forgy initialisation.
    let mut centroids: Vec<(Vec<f64>, Vec<Vec<f64>>)> = init_centroids_forgy(measures, k, &mut rng);

    let mut assignments = vec![0_usize; n_meas];
    let mut history = Vec::with_capacity(config.n_iter);

    let bary_cfg_eps = config.eps as f32;
    let bary_cfg = BaryConfig {
        eps: bary_cfg_eps,
        n_outer: 10,
        n_inner: config.bary_iter,
        tol: config.bary_tol as f32,
    };

    let assignment_iters = (config.bary_iter / 2).max(20);

    for _outer in 0..config.n_iter {
        // ── Assignment step ──────────────────────────────────────────────────
        let mut iter_cost = 0.0_f64;
        for i in 0..n_meas {
            let (w_i, s_i) = measures[i];
            let mut best_k = 0_usize;
            let mut best_dist = f64::INFINITY;

            for (ki, (w_c, s_c)) in centroids.iter().enumerate() {
                let dist = sinkhorn_w2_distance(
                    w_i,
                    s_i,
                    w_c,
                    s_c,
                    config.assignment_eps,
                    assignment_iters,
                )?;
                if dist < best_dist {
                    best_dist = dist;
                    best_k = ki;
                }
            }
            assignments[i] = best_k;
            iter_cost += best_dist;
        }
        history.push(iter_cost);

        // ── Centroid update step ─────────────────────────────────────────────
        for (ki, centroid_slot) in centroids.iter_mut().enumerate() {
            let members: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter_map(|(i, &c)| if c == ki { Some(i) } else { None })
                .collect();

            if members.is_empty() {
                // Empty cluster: keep current centroid.
                continue;
            }

            let m = members.len();
            let lambdas = vec![1.0_f32 / m as f32; m];

            // Build flat support for free_support_barycenter (expects Vec<f32>).
            // free_support_barycenter uses flat row-major representation.
            let d = measures[members[0]].1[0].len();
            let measures_x_flat: Vec<Vec<f32>> = members
                .iter()
                .map(|&idx| {
                    let (_, support) = measures[idx];
                    support
                        .iter()
                        .flat_map(|pt| pt.iter().map(|&v| v as f32))
                        .collect()
                })
                .collect();
            let measures_a_flat: Vec<Vec<f32>> = members
                .iter()
                .map(|&idx| {
                    let (w, _) = measures[idx];
                    let norm: f32 = w.iter().map(|&v| v as f32).sum::<f32>().max(1e-15);
                    w.iter().map(|&v| v as f32 / norm).collect()
                })
                .collect();

            // Determine n_bary: match the size of the largest member.
            let n_bary = members
                .iter()
                .map(|&idx| measures[idx].0.len())
                .max()
                .unwrap_or(1);

            match free_support_barycenter(
                &measures_x_flat,
                &measures_a_flat,
                d,
                n_bary,
                &lambdas,
                &bary_cfg,
                &mut rng,
            ) {
                Ok((new_y_flat, new_b)) => {
                    // Convert flat f32 support back to Vec<Vec<f64>>.
                    let new_support: Vec<Vec<f64>> = (0..n_bary)
                        .map(|zi| (0..d).map(|dim| new_y_flat[zi * d + dim] as f64).collect())
                        .collect();
                    let new_weights: Vec<f64> = new_b.iter().map(|&v| v as f64).collect();
                    *centroid_slot = (new_weights, new_support);
                }
                Err(_) => {
                    // Keep previous centroid on inner failure.
                }
            }
        }
    }

    // Final total cost.
    let mut total_cost = 0.0_f64;
    for i in 0..n_meas {
        let ki = assignments[i];
        let (w_i, s_i) = measures[i];
        let (w_c, s_c) = &centroids[ki];
        if let Ok(d) =
            sinkhorn_w2_distance(w_i, s_i, w_c, s_c, config.assignment_eps, assignment_iters)
        {
            total_cost += d;
        }
    }

    Ok(SinkhornKmeansResult {
        assignments,
        centroids,
        total_cost,
        history,
    })
}

// ─────────────────────────────── tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_w(n: usize) -> Vec<f64> {
        vec![1.0 / n as f64; n]
    }

    fn dirac(x: Vec<f64>) -> (Vec<f64>, Vec<Vec<f64>>) {
        (vec![1.0], vec![x])
    }

    /// Two Dirac masses at x and y, far apart.
    fn dirac_at(x: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
        dirac(vec![x])
    }

    #[test]
    fn k1_assigns_everything_to_single_cluster() {
        let m1 = dirac_at(0.0);
        let m2 = dirac_at(1.0);
        let m3 = dirac_at(2.0);
        let measures: Vec<(&Vec<f64>, &Vec<Vec<f64>>)> =
            vec![(&m1.0, &m1.1), (&m2.0, &m2.1), (&m3.0, &m3.1)];
        let cfg = SinkhornKmeansConfig {
            k: 1,
            n_iter: 3,
            eps: 0.1,
            bary_iter: 10,
            bary_tol: 1e-4,
            assignment_eps: 0.1,
            seed: 0,
        };
        let res = sinkhorn_kmeans(&measures, &cfg).expect("ok");
        assert_eq!(res.assignments.len(), 3);
        for &a in &res.assignments {
            assert_eq!(a, 0, "single cluster: all assigned to 0");
        }
        assert_eq!(res.centroids.len(), 1);
    }

    #[test]
    fn k2_separates_two_clusters() {
        // 3 measures near 0, 3 near 100 — should split into 2 clusters.
        let near_zero: Vec<(Vec<f64>, Vec<Vec<f64>>)> =
            (0..3).map(|i| dirac_at(i as f64 * 0.1)).collect();
        let near_hundred: Vec<(Vec<f64>, Vec<Vec<f64>>)> =
            (0..3).map(|i| dirac_at(100.0 + i as f64 * 0.1)).collect();
        let all: Vec<(Vec<f64>, Vec<Vec<f64>>)> = near_zero
            .iter()
            .chain(near_hundred.iter())
            .cloned()
            .collect();
        let measures: Vec<(&Vec<f64>, &Vec<Vec<f64>>)> = all.iter().map(|(w, s)| (w, s)).collect();
        let cfg = SinkhornKmeansConfig {
            k: 2,
            n_iter: 5,
            eps: 0.5,
            bary_iter: 10,
            bary_tol: 1e-3,
            assignment_eps: 0.5,
            seed: 13,
        };
        let res = sinkhorn_kmeans(&measures, &cfg).expect("ok");
        // The first 3 and last 3 should be in different clusters.
        let cluster_of_0 = res.assignments[0];
        let cluster_of_3 = res.assignments[3];
        assert_ne!(
            cluster_of_0, cluster_of_3,
            "should be in different clusters"
        );
        // Within each group, assignments should be same.
        for i in 0..3 {
            assert_eq!(res.assignments[i], cluster_of_0);
        }
        for i in 3..6 {
            assert_eq!(res.assignments[i], cluster_of_3);
        }
    }

    #[test]
    fn assignment_vector_length() {
        let m1 = dirac_at(0.0);
        let m2 = dirac_at(1.0);
        let m3 = dirac_at(2.0);
        let m4 = dirac_at(3.0);
        let measures: Vec<(&Vec<f64>, &Vec<Vec<f64>>)> = vec![
            (&m1.0, &m1.1),
            (&m2.0, &m2.1),
            (&m3.0, &m3.1),
            (&m4.0, &m4.1),
        ];
        let cfg = SinkhornKmeansConfig {
            k: 2,
            n_iter: 2,
            ..Default::default()
        };
        let res = sinkhorn_kmeans(&measures, &cfg).expect("ok");
        assert_eq!(res.assignments.len(), 4);
    }

    #[test]
    fn history_length_equals_n_iter() {
        let m1 = dirac_at(0.0);
        let m2 = dirac_at(5.0);
        let measures: Vec<(&Vec<f64>, &Vec<Vec<f64>>)> = vec![(&m1.0, &m1.1), (&m2.0, &m2.1)];
        let cfg = SinkhornKmeansConfig {
            k: 2,
            n_iter: 7,
            eps: 0.1,
            bary_iter: 5,
            bary_tol: 1e-3,
            assignment_eps: 0.1,
            seed: 1,
        };
        let res = sinkhorn_kmeans(&measures, &cfg).expect("ok");
        assert_eq!(res.history.len(), 7, "history length should equal n_iter");
    }

    #[test]
    fn centroid_weights_sum_to_one() {
        let measures: Vec<(Vec<f64>, Vec<Vec<f64>>)> = (0..4).map(|i| dirac_at(i as f64)).collect();
        let refs: Vec<(&Vec<f64>, &Vec<Vec<f64>>)> = measures.iter().map(|(w, s)| (w, s)).collect();
        let cfg = SinkhornKmeansConfig {
            k: 2,
            n_iter: 3,
            eps: 0.1,
            bary_iter: 10,
            bary_tol: 1e-3,
            assignment_eps: 0.1,
            seed: 99,
        };
        let res = sinkhorn_kmeans(&refs, &cfg).expect("ok");
        for (ki, (w, _)) in res.centroids.iter().enumerate() {
            let total: f64 = w.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-5,
                "centroid {ki} weights sum {total} != 1.0"
            );
        }
    }

    #[test]
    fn total_cost_non_negative_finite() {
        let m1 = dirac_at(0.0);
        let m2 = dirac_at(1.0);
        let measures: Vec<(&Vec<f64>, &Vec<Vec<f64>>)> = vec![(&m1.0, &m1.1), (&m2.0, &m2.1)];
        let cfg = SinkhornKmeansConfig {
            k: 2,
            n_iter: 2,
            eps: 0.1,
            bary_iter: 5,
            bary_tol: 1e-3,
            assignment_eps: 0.1,
            seed: 0,
        };
        let res = sinkhorn_kmeans(&measures, &cfg).expect("ok");
        assert!(
            res.total_cost.is_finite() && res.total_cost >= 0.0,
            "total_cost = {}",
            res.total_cost
        );
    }

    #[test]
    fn single_point_measures() {
        let points: Vec<(Vec<f64>, Vec<Vec<f64>>)> = vec![
            (vec![1.0], vec![vec![0.0, 0.0]]),
            (vec![1.0], vec![vec![10.0, 0.0]]),
            (vec![1.0], vec![vec![0.0, 10.0]]),
        ];
        let measures: Vec<(&Vec<f64>, &Vec<Vec<f64>>)> =
            points.iter().map(|(w, s)| (w, s)).collect();
        let cfg = SinkhornKmeansConfig {
            k: 2,
            n_iter: 3,
            eps: 0.5,
            bary_iter: 5,
            bary_tol: 1e-3,
            assignment_eps: 0.5,
            seed: 7,
        };
        let res = sinkhorn_kmeans(&measures, &cfg).expect("ok");
        assert_eq!(res.assignments.len(), 3);
        for &a in &res.assignments {
            assert!(a < 2);
        }
    }

    #[test]
    fn seed_reproducibility() {
        let points: Vec<(Vec<f64>, Vec<Vec<f64>>)> = (0..5).map(|i| dirac_at(i as f64)).collect();
        let refs: Vec<(&Vec<f64>, &Vec<Vec<f64>>)> = points.iter().map(|(w, s)| (w, s)).collect();
        let cfg = SinkhornKmeansConfig {
            k: 2,
            n_iter: 3,
            eps: 0.2,
            bary_iter: 5,
            bary_tol: 1e-3,
            assignment_eps: 0.2,
            seed: 42,
        };
        let res1 = sinkhorn_kmeans(&refs, &cfg).expect("ok");
        let res2 = sinkhorn_kmeans(&refs, &cfg).expect("ok");
        assert_eq!(res1.assignments, res2.assignments);
    }

    #[test]
    fn eps_effect_on_cost() {
        // Larger eps → smoother distances → generally different (often larger) costs.
        let points: Vec<(Vec<f64>, Vec<Vec<f64>>)> =
            (0..4).map(|i| dirac_at(i as f64 * 2.0)).collect();
        let refs: Vec<(&Vec<f64>, &Vec<Vec<f64>>)> = points.iter().map(|(w, s)| (w, s)).collect();
        let cfg_small = SinkhornKmeansConfig {
            k: 2,
            n_iter: 3,
            eps: 0.01,
            bary_iter: 10,
            bary_tol: 1e-3,
            assignment_eps: 0.01,
            seed: 0,
        };
        let cfg_large = SinkhornKmeansConfig {
            k: 2,
            n_iter: 3,
            eps: 5.0,
            bary_iter: 10,
            bary_tol: 1e-3,
            assignment_eps: 5.0,
            seed: 0,
        };
        let res_small = sinkhorn_kmeans(&refs, &cfg_small).expect("ok");
        let res_large = sinkhorn_kmeans(&refs, &cfg_large).expect("ok");
        // Both should give finite non-negative costs.
        assert!(res_small.total_cost.is_finite() && res_small.total_cost >= 0.0);
        assert!(res_large.total_cost.is_finite() && res_large.total_cost >= 0.0);
    }

    #[test]
    fn sinkhorn_w2_distance_zero_for_same_measure() {
        let w = vec![0.5, 0.5];
        let s = vec![vec![0.0], vec![1.0]];
        let d = sinkhorn_w2_distance(&w, &s, &w, &s, 0.1, 100).expect("ok");
        assert!(d.abs() < 1e-4, "same measure distance = {d}");
    }

    #[test]
    fn sinkhorn_w2_distance_positive_for_different_measures() {
        let w = vec![1.0];
        let s_a = vec![vec![0.0]];
        let s_b = vec![vec![10.0]];
        let d = sinkhorn_w2_distance(&w, &s_a, &w, &s_b, 0.1, 100).expect("ok");
        assert!(d > 0.0, "distance between distant measures should be > 0");
    }

    #[test]
    fn rejects_zero_clusters() {
        let m = dirac_at(0.0);
        let measures = vec![(&m.0, &m.1)];
        let cfg = SinkhornKmeansConfig {
            k: 0,
            ..Default::default()
        };
        let res = sinkhorn_kmeans(&measures, &cfg);
        assert!(matches!(res, Err(OtError::BadCount { .. })));
    }

    #[test]
    fn rejects_k_larger_than_n_measures() {
        let m = dirac_at(0.0);
        let measures = vec![(&m.0, &m.1)];
        let cfg = SinkhornKmeansConfig {
            k: 3,
            ..Default::default()
        };
        let res = sinkhorn_kmeans(&measures, &cfg);
        assert!(matches!(res, Err(OtError::BadCount { .. })));
    }

    #[test]
    fn measures_with_same_support_stay_in_one_cluster() {
        // All measures identical → all should be in the same cluster (k=1).
        let m = (uniform_w(3), vec![vec![0.0], vec![1.0], vec![2.0]]);
        let measures: Vec<(&Vec<f64>, &Vec<Vec<f64>>)> =
            vec![(&m.0, &m.1), (&m.0, &m.1), (&m.0, &m.1)];
        let cfg = SinkhornKmeansConfig {
            k: 1,
            n_iter: 3,
            eps: 0.1,
            bary_iter: 5,
            bary_tol: 1e-3,
            assignment_eps: 0.1,
            seed: 0,
        };
        let res = sinkhorn_kmeans(&measures, &cfg).expect("ok");
        for &a in &res.assignments {
            assert_eq!(a, 0);
        }
    }
}
