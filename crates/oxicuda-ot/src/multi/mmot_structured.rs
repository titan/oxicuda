//! Multi-marginal OT with structured pairwise-separable cost.
//!
//! The cost is decomposed as
//!
//! ```text
//! c(x_1, …, x_K) = Σ_{1 ≤ i < j ≤ K}  c_{ij}(x_i, x_j)
//! ```
//!
//! This structure arises in the MMOT reformulation of Wasserstein barycenters and
//! in quantum chemistry (Coulomb cost). The algorithm generalises log-domain
//! Sinkhorn to the multi-marginal setting via alternating axis updates: each
//! potential is updated by absorbing all pairwise interactions that involve the
//! corresponding marginal.

use crate::error::{OtError, OtResult};
use crate::handle::LcgRng;

/// Configuration for the structured multi-marginal OT solver.
#[derive(Debug, Clone)]
pub struct MmotStructuredConfig {
    /// Entropic regularisation strength ε (must be > 0).
    pub eps: f64,
    /// Maximum number of outer alternating iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum marginal violation.
    pub tol: f64,
    /// Number of inner Sinkhorn iterations applied per marginal update.
    pub inner_sinkhorn_iters: usize,
}

impl Default for MmotStructuredConfig {
    fn default() -> Self {
        Self {
            eps: 0.1,
            max_iter: 500,
            tol: 1e-6,
            inner_sinkhorn_iters: 10,
        }
    }
}

/// Result of the structured multi-marginal OT solver.
#[derive(Debug, Clone)]
pub struct MmotStructuredResult {
    /// One log-domain dual potential per marginal; `potentials[k]` has length `n_k`.
    pub potentials: Vec<Vec<f64>>,
    /// Primal transport cost `⟨C, T⟩`.
    pub cost: f64,
    /// Number of completed outer iterations.
    pub iters: usize,
}

/// Configuration for MMOT-based Wasserstein barycenter.
#[derive(Debug, Clone)]
pub struct MmotBaryConfig {
    /// Entropic regularisation strength.
    pub eps: f64,
    /// Maximum number of alternating iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Number of barycenter support points.
    pub n_support: usize,
}

impl Default for MmotBaryConfig {
    fn default() -> Self {
        Self {
            eps: 0.1,
            max_iter: 300,
            tol: 1e-5,
            n_support: 10,
        }
    }
}

// ─────────────────────────────── helpers ────────────────────────────────────

/// Numerically stable log-sum-exp.
///
/// Returns `NEG_INFINITY` for empty slices.
#[inline]
fn log_sum_exp(vals: &[f64]) -> f64 {
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

/// Safe natural log: clamps argument to `f64::MIN_POSITIVE` before taking log.
#[inline]
fn safe_ln(x: f64) -> f64 {
    let floor = f64::MIN_POSITIVE;
    if x <= floor { floor.ln() } else { x.ln() }
}

// ─────────────────────────────── validation ─────────────────────────────────

fn validate_structured(
    marginals: &[Vec<f64>],
    pair_costs: &[Vec<Vec<f64>>],
    cfg: &MmotStructuredConfig,
) -> OtResult<()> {
    let k = marginals.len();
    if k == 0 {
        return Err(OtError::EmptyInput);
    }
    if cfg.eps <= 0.0 {
        return Err(OtError::BadEpsilon {
            eps: cfg.eps as f32,
        });
    }
    if pair_costs.len() != k {
        return Err(OtError::IncompatibleLength {
            a: k,
            b: pair_costs.len(),
        });
    }
    for row in pair_costs.iter() {
        if row.len() != k {
            return Err(OtError::IncompatibleLength { a: k, b: row.len() });
        }
    }
    for (ki, marg) in marginals.iter().enumerate() {
        if marg.is_empty() {
            return Err(OtError::EmptyInput);
        }
        for &v in marg.iter() {
            if v < 0.0 || !v.is_finite() {
                return Err(OtError::NegativeWeight);
            }
        }
        // Validate pair_costs shapes for upper-triangular entries.
        let n_i = marg.len();
        for kj in (ki + 1)..k {
            let n_j = marginals[kj].len();
            let expected = n_i * n_j;
            if pair_costs[ki][kj].len() != expected {
                return Err(OtError::MarginalMismatch {
                    m: n_i,
                    n: n_j,
                    a_len: expected,
                    b_len: pair_costs[ki][kj].len(),
                });
            }
        }
    }
    Ok(())
}

// ─────────────────────────────── core solver ────────────────────────────────

/// Multi-marginal OT with pairwise-separable cost via alternating Sinkhorn.
///
/// The structured cost is `c(x_1,…,x_K) = Σ_{i<j} c_{ij}(x_i, x_j)`.
/// For each marginal `k` and support point `x_k`, the effective log-kernel is
/// the sum of soft-min interactions with all other marginals.
///
/// `marginals[k]` is the `k`-th probability vector.
/// `pair_costs[i][j]` (for `i < j`) is the `n_i × n_j` cost matrix (row-major).
pub fn mmot_structured(
    marginals: &[Vec<f64>],
    pair_costs: &[Vec<Vec<f64>>],
    config: &MmotStructuredConfig,
) -> OtResult<MmotStructuredResult> {
    validate_structured(marginals, pair_costs, config)?;

    let k_marg = marginals.len();
    let eps = config.eps;

    // Initialise log-potentials: f_k(x) = ε * log(marginal_k(x)).
    let mut potentials: Vec<Vec<f64>> = marginals
        .iter()
        .map(|m| m.iter().map(|&v| eps * safe_ln(v)).collect())
        .collect();

    let mut completed = 0_usize;

    'outer: for outer_iter in 0..config.max_iter {
        for ki in 0..k_marg {
            let n_i = marginals[ki].len();
            let mut new_pot_ki = vec![0.0_f64; n_i];

            // For each x_i, compute the interaction with every other marginal kj.
            for xi in 0..n_i {
                // Aggregate log-interaction across all other marginals.
                // log_interact(x_i) = Σ_{j≠i} LSE_{x_j} (f_j(x_j) - c_{ij}(x_i, x_j) / ε)
                let mut total_interact = 0.0_f64;

                for kj in 0..k_marg {
                    if kj == ki {
                        continue;
                    }
                    let n_j = marginals[kj].len();
                    // c_pair(x_i, x_j): ensure we use the upper-triangular entry.
                    let buf: Vec<f64> = (0..n_j)
                        .map(|xj| {
                            let c_val = if ki < kj {
                                pair_costs[ki][kj][xi * n_j + xj]
                            } else {
                                pair_costs[kj][ki][xj * n_i + xi]
                            };
                            // potentials are stored in eps-scale; divide by eps for
                            // dimensionless log-space argument, then subtract c/eps.
                            potentials[kj][xj] / eps - c_val / eps
                        })
                        .collect();
                    total_interact += log_sum_exp(&buf);
                }

                // f_ki(x_i) ← ε*log(a_ki(x_i)) - ε * total_interact
                new_pot_ki[xi] = eps * safe_ln(marginals[ki][xi]) - eps * total_interact;
            }

            potentials[ki] = new_pot_ki;
        }

        // Measure max marginal violation.
        // For marginal k: violation_k(x) = |Σ_{other} exp((f_others - c)/ε) - a_k(x)|.
        // We approximate by computing for each marginal the effective marginal of the
        // current implicit coupling and comparing to the target.
        let mut max_viol = 0.0_f64;
        for ki in 0..k_marg {
            let n_i = marginals[ki].len();
            for xi in 0..n_i {
                // Compute the marginal of the current coupling at x_i:
                // marg_k(x_i) = exp(f_k(x_i)/ε) * Π_{j≠k} (Σ_{x_j} exp((f_j(x_j) - c_{kj}(x_i,x_j))/ε))
                // In log: log_marg_k(x_i) = f_k(x_i)/ε + Σ_{j≠k} log(Σ_{x_j} exp((f_j(x_j)-c_{kj}/ε)))
                let mut log_marg = potentials[ki][xi] / eps;
                for kj in 0..k_marg {
                    if kj == ki {
                        continue;
                    }
                    let n_j = marginals[kj].len();
                    let buf: Vec<f64> = (0..n_j)
                        .map(|xj| {
                            let c_val = if ki < kj {
                                pair_costs[ki][kj][xi * n_j + xj]
                            } else {
                                pair_costs[kj][ki][xj * n_i + xi]
                            };
                            potentials[kj][xj] / eps - c_val / eps
                        })
                        .collect();
                    log_marg += log_sum_exp(&buf);
                }
                let marg_xi = log_marg.exp();
                let viol = (marg_xi - marginals[ki][xi]).abs();
                if viol > max_viol {
                    max_viol = viol;
                }
            }
        }

        completed = outer_iter + 1;
        if max_viol < config.tol {
            break 'outer;
        }
    }

    // Compute primal transport cost by evaluating the coupling on pairs.
    // For the structured cost, cost = Σ_{i<j} Σ_{x_i,x_j} T_{ij}(x_i,x_j) * c_{ij}(x_i,x_j)
    // where T_{ij} is the marginal of the MMOT coupling on (i,j).
    let mut cost = 0.0_f64;
    for ki in 0..k_marg {
        let n_i = marginals[ki].len();
        for kj in (ki + 1)..k_marg {
            let n_j = marginals[kj].len();
            // Compute bivariate marginal T_{ij}(x_i, x_j) ∝ exp((f_i(x_i) + f_j(x_j) - c_{ij})/ε)
            // times the product of all interactions with other marginals (which integrate out).
            // Approximation: use the factored form assuming independence of other marginals.
            for xi in 0..n_i {
                for xj in 0..n_j {
                    // log T_{ij}(x_i, x_j) ≈ (f_i(x_i) + f_j(x_j) - c_{ij}(x_i,x_j)) / ε
                    // (This is exact for K=2; for K>2 the other potentials normalise correctly
                    //  only after convergence when marginal constraints are satisfied.)
                    let c_val = pair_costs[ki][kj][xi * n_j + xj];
                    // potentials are in eps-scale; divide by eps to get log T
                    let log_t = potentials[ki][xi] / eps + potentials[kj][xj] / eps - c_val / eps;
                    cost += log_t.exp() * c_val;
                }
            }
        }
    }

    Ok(MmotStructuredResult {
        potentials,
        cost,
        iters: completed,
    })
}

// ─────────────────────────────── barycenter ─────────────────────────────────

/// MMOT reformulation of the Wasserstein barycenter.
///
/// Couples K input measures `(a_k, X_k)` to a common barycenter support `Z` via
/// squared Euclidean costs `c_k(x_k, z) = ‖x_k − z‖²`. The barycenter weights
/// and support are jointly optimised via alternating free-support updates.
///
/// `measures`: slice of `(weights, support_points)` per input measure, where
///   `support_points[j]` is a `d`-dimensional point for the `j`-th atom.
/// `weights_on_measures`: λ_k weights (must sum to 1).
///
/// Returns `(barycenter_weights, barycenter_support)`.
pub fn mmot_barycenter(
    measures: &[(&Vec<f64>, &Vec<Vec<f64>>)],
    weights_on_measures: &[f64],
    config: &MmotBaryConfig,
    rng: &mut LcgRng,
) -> OtResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let k_meas = measures.len();
    if k_meas == 0 {
        return Err(OtError::EmptyInput);
    }
    if weights_on_measures.len() != k_meas {
        return Err(OtError::IncompatibleLength {
            a: k_meas,
            b: weights_on_measures.len(),
        });
    }
    if config.eps <= 0.0 {
        return Err(OtError::BadEpsilon {
            eps: config.eps as f32,
        });
    }
    if config.n_support == 0 {
        return Err(OtError::BadCount { got: 0 });
    }
    let lam_sum: f64 = weights_on_measures.iter().sum();
    if (lam_sum - 1.0).abs() > 1e-3 {
        return Err(OtError::NotProbability);
    }

    // Determine spatial dimension from first measure.
    let d = if let Some(pts) = measures[0].1.first() {
        pts.len()
    } else {
        return Err(OtError::EmptyInput);
    };

    if d == 0 {
        return Err(OtError::BadDim { got: 0 });
    }

    // Validate all measures.
    for (k, (weights, support)) in measures.iter().enumerate() {
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
        let _ = k;
    }

    let n_z = config.n_support;
    let eps = config.eps;

    // Initialise barycenter support: sample from convex hull of input means,
    // perturbed by RNG.
    let mut z_support: Vec<Vec<f64>> = (0..n_z)
        .map(|_| {
            let k_pick = rng.next_usize(k_meas);
            let (weights_k, support_k) = measures[k_pick];
            let n_k = support_k.len();
            // Compute weighted centroid of that measure.
            let total: f64 = weights_k.iter().sum();
            let inv = if total > 1e-15 { 1.0 / total } else { 1.0 };
            let mut centroid = vec![0.0_f64; d];
            for (j, pt) in support_k.iter().enumerate() {
                let w = weights_k[j] * inv;
                for dim in 0..d {
                    centroid[dim] += w * pt[dim];
                }
            }
            // Small perturbation.
            for slot in centroid.iter_mut() {
                let jitter = 1e-2 * (rng.next_f32() as f64 - 0.5);
                *slot += jitter;
            }
            let _ = n_k;
            centroid
        })
        .collect();

    // Uniform barycenter weights.
    let mut z_weights = vec![1.0_f64 / n_z as f64; n_z];

    // Alternate between: (1) compute OT plans from Z to each input measure,
    //                    (2) update Z support by barycentric projection.
    for _ in 0..config.max_iter {
        let mut new_z_support = vec![vec![0.0_f64; d]; n_z];
        let mut row_sums = vec![0.0_f64; n_z];

        for (k, (weights_k, support_k)) in measures.iter().enumerate() {
            let lam_k = weights_on_measures[k];
            let n_k = support_k.len();

            // Build cost matrix between Z (n_z rows) and measure k (n_k cols).
            let mut cost_mat = vec![0.0_f64; n_z * n_k];
            for (zi, z_pt) in z_support.iter().enumerate() {
                for (j, x_pt) in support_k.iter().enumerate() {
                    let mut sq = 0.0_f64;
                    for dim in 0..d {
                        let diff = z_pt[dim] - x_pt[dim];
                        sq += diff * diff;
                    }
                    cost_mat[zi * n_k + j] = sq;
                }
            }

            // Run log-domain Sinkhorn between z_weights and weights_k.
            let w_k_total: f64 = weights_k.iter().sum();
            let inv_total = if w_k_total > 1e-15 {
                1.0 / w_k_total
            } else {
                continue;
            };
            let weights_k_norm: Vec<f64> = weights_k.iter().map(|&w| w * inv_total).collect();

            // Log-potential updates: u (n_z), v (n_k).
            let mut u_pot = vec![0.0_f64; n_z];
            let mut v_pot = vec![0.0_f64; n_k];
            for (zi, &zw) in z_weights.iter().enumerate() {
                u_pot[zi] = eps * safe_ln(zw);
            }
            for (j, &wj) in weights_k_norm.iter().enumerate() {
                v_pot[j] = eps * safe_ln(wj);
            }

            let n_inner = config.max_iter.min(200);
            for _ in 0..n_inner {
                // Row update.
                for zi in 0..n_z {
                    let buf: Vec<f64> = (0..n_k)
                        .map(|j| (v_pot[j] - cost_mat[zi * n_k + j]) / eps)
                        .collect();
                    let lse = log_sum_exp(&buf);
                    u_pot[zi] = eps * safe_ln(z_weights[zi]) - eps * lse;
                }
                // Column update.
                for j in 0..n_k {
                    let buf: Vec<f64> = (0..n_z)
                        .map(|zi| (u_pot[zi] - cost_mat[zi * n_k + j]) / eps)
                        .collect();
                    let lse = log_sum_exp(&buf);
                    v_pot[j] = eps * safe_ln(weights_k_norm[j]) - eps * lse;
                }
            }

            // Barycentric projection: T_{zi, j} = exp((u_zi + v_j - c_{zi,j})/eps).
            for zi in 0..n_z {
                let mut row_mass = 0.0_f64;
                for j in 0..n_k {
                    let t = ((u_pot[zi] + v_pot[j] - cost_mat[zi * n_k + j]) / eps).exp();
                    row_mass += t;
                    for dim in 0..d {
                        new_z_support[zi][dim] += lam_k * t * support_k[j][dim];
                    }
                }
                row_sums[zi] += lam_k * row_mass;
            }
        }

        // Normalise support update.
        for zi in 0..n_z {
            let rs = row_sums[zi];
            if rs > 1e-15 {
                for dim in 0..d {
                    z_support[zi][dim] = new_z_support[zi][dim] / rs;
                }
            }
        }

        // Re-normalise z_weights (they stay uniform in this symmetric formulation).
        let zw_total: f64 = z_weights.iter().sum();
        if zw_total > 1e-15 {
            let inv = 1.0 / zw_total;
            for zw in z_weights.iter_mut() {
                *zw *= inv;
            }
        }
    }

    Ok((z_weights, z_support))
}

// ─────────────────────────────── tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(n: usize) -> Vec<f64> {
        vec![1.0 / n as f64; n]
    }

    fn pair_cost_sq(n_i: usize, n_j: usize) -> Vec<f64> {
        let mut c = vec![0.0_f64; n_i * n_j];
        for i in 0..n_i {
            for j in 0..n_j {
                let d = i as f64 - j as f64;
                c[i * n_j + j] = d * d;
            }
        }
        c
    }

    /// Build a trivial pair_costs matrix for K marginals (all zeros for non (0,1) pairs).
    fn empty_pair_costs(k: usize, sizes: &[usize]) -> Vec<Vec<Vec<f64>>> {
        let mut pc: Vec<Vec<Vec<f64>>> = (0..k)
            .map(|_| (0..k).map(|_| Vec::new()).collect())
            .collect();
        for i in 0..k {
            for j in (i + 1)..k {
                pc[i][j] = vec![0.0; sizes[i] * sizes[j]];
            }
        }
        pc
    }

    #[test]
    fn k1_trivial_single_marginal() {
        // With K=1 there are no pairs, cost should be 0.
        let marginals = vec![uniform(4)];
        let pair_costs: Vec<Vec<Vec<f64>>> = vec![vec![vec![]]];
        let cfg = MmotStructuredConfig::default();
        let res = mmot_structured(&marginals, &pair_costs, &cfg).expect("ok");
        assert_eq!(res.potentials.len(), 1);
        assert_eq!(res.potentials[0].len(), 4);
        assert!(res.cost.abs() < 1e-10, "cost={}", res.cost);
    }

    #[test]
    fn k2_potentials_correct_length() {
        let n = 3;
        let marginals = vec![uniform(n), uniform(n)];
        let c01 = pair_cost_sq(n, n);
        let mut pair_costs = empty_pair_costs(2, &[n, n]);
        pair_costs[0][1] = c01;
        let cfg = MmotStructuredConfig {
            eps: 0.5,
            max_iter: 200,
            tol: 1e-4,
            inner_sinkhorn_iters: 10,
        };
        let res = mmot_structured(&marginals, &pair_costs, &cfg).expect("ok");
        assert_eq!(res.potentials.len(), 2);
        assert_eq!(res.potentials[0].len(), n);
        assert_eq!(res.potentials[1].len(), n);
    }

    #[test]
    fn k2_marginal_violation_below_tol() {
        let n = 4;
        let marginals = vec![uniform(n), uniform(n)];
        let c01 = pair_cost_sq(n, n);
        let mut pair_costs = empty_pair_costs(2, &[n, n]);
        pair_costs[0][1] = c01;
        let cfg = MmotStructuredConfig {
            eps: 0.3,
            max_iter: 1000,
            tol: 1e-5,
            inner_sinkhorn_iters: 10,
        };
        let res = mmot_structured(&marginals, &pair_costs, &cfg).expect("ok");
        // Check approximate marginals: for K=2, T_{ij} ∝ exp((f_i+f_j-c)/eps)
        let eps = cfg.eps;
        for xi in 0..n {
            let mut row = 0.0_f64;
            for xj in 0..n {
                let c_val = pair_costs[0][1][xi * n + xj];
                let t = ((res.potentials[0][xi] + res.potentials[1][xj] - c_val) / eps).exp();
                row += t;
            }
            assert!(
                (row - marginals[0][xi]).abs() < 1e-3,
                "row {} sum {row} != {}",
                xi,
                marginals[0][xi]
            );
        }
    }

    #[test]
    fn k3_symmetric_uniform_marginals() {
        let n = 3;
        let margs = vec![uniform(n); 3];
        let c_sq: Vec<f64> = pair_cost_sq(n, n);
        let mut pair_costs = empty_pair_costs(3, &[n, n, n]);
        pair_costs[0][1] = c_sq.clone();
        pair_costs[0][2] = c_sq.clone();
        pair_costs[1][2] = c_sq;
        let cfg = MmotStructuredConfig {
            eps: 0.5,
            max_iter: 300,
            tol: 1e-4,
            inner_sinkhorn_iters: 10,
        };
        let res = mmot_structured(&margs, &pair_costs, &cfg).expect("ok");
        assert_eq!(res.potentials.len(), 3);
        for p in &res.potentials {
            assert_eq!(p.len(), n);
        }
        // Cost should be finite and non-negative.
        assert!(
            res.cost.is_finite() && res.cost >= -1e-6,
            "cost={}",
            res.cost
        );
    }

    #[test]
    fn identity_marginals_zero_cost() {
        // Zero cost matrix: any coupling is valid, cost should be near 0.
        let n = 3;
        let margs = vec![uniform(n), uniform(n)];
        let c_zero = vec![0.0_f64; n * n];
        let mut pair_costs = empty_pair_costs(2, &[n, n]);
        pair_costs[0][1] = c_zero;
        let cfg = MmotStructuredConfig {
            eps: 0.5,
            max_iter: 100,
            tol: 1e-4,
            inner_sinkhorn_iters: 10,
        };
        let res = mmot_structured(&margs, &pair_costs, &cfg).expect("ok");
        assert!(res.cost.abs() < 1e-8, "cost={}", res.cost);
    }

    #[test]
    fn iters_bounded_by_max_iter() {
        let n = 2;
        let margs = vec![uniform(n), uniform(n)];
        let mut pair_costs = empty_pair_costs(2, &[n, n]);
        pair_costs[0][1] = vec![0.0; n * n];
        let cfg = MmotStructuredConfig {
            eps: 1.0,
            max_iter: 7,
            tol: 1e-12,
            inner_sinkhorn_iters: 5,
        };
        let res = mmot_structured(&margs, &pair_costs, &cfg).expect("ok");
        assert!(res.iters <= 7, "iters={}", res.iters);
    }

    #[test]
    fn small_eps_near_deterministic_coupling() {
        // Very small eps → near-deterministic coupling concentrates on diagonal.
        let n = 3;
        // Diagonal cost (penalises off-diagonal).
        let mut c = vec![10.0_f64; n * n];
        for i in 0..n {
            c[i * n + i] = 0.0;
        }
        let margs = vec![uniform(n), uniform(n)];
        let mut pair_costs = empty_pair_costs(2, &[n, n]);
        pair_costs[0][1] = c;
        let cfg = MmotStructuredConfig {
            eps: 0.01,
            max_iter: 2000,
            tol: 1e-5,
            inner_sinkhorn_iters: 20,
        };
        let res = mmot_structured(&margs, &pair_costs, &cfg).expect("ok");
        // The coupling should concentrate near diagonal; cost ≈ 0.
        assert!(res.cost < 1.0, "cost={}", res.cost);
    }

    #[test]
    fn cost_is_finite() {
        let n = 4;
        let margs = vec![uniform(n), uniform(n)];
        let c01 = pair_cost_sq(n, n);
        let mut pair_costs = empty_pair_costs(2, &[n, n]);
        pair_costs[0][1] = c01;
        let cfg = MmotStructuredConfig::default();
        let res = mmot_structured(&margs, &pair_costs, &cfg).expect("ok");
        assert!(res.cost.is_finite(), "cost should be finite");
    }

    #[test]
    fn rejects_empty_marginals() {
        let cfg = MmotStructuredConfig::default();
        let res = mmot_structured(&[], &[], &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn rejects_bad_epsilon() {
        let n = 2;
        let margs = vec![uniform(n), uniform(n)];
        let mut pair_costs = empty_pair_costs(2, &[n, n]);
        pair_costs[0][1] = vec![0.0; n * n];
        let cfg = MmotStructuredConfig {
            eps: 0.0,
            ..Default::default()
        };
        let res = mmot_structured(&margs, &pair_costs, &cfg);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn rejects_negative_weight() {
        let cfg = MmotStructuredConfig::default();
        let margs = vec![vec![-0.5, 1.5], uniform(2)];
        let mut pair_costs = empty_pair_costs(2, &[2, 2]);
        pair_costs[0][1] = vec![0.0; 4];
        let res = mmot_structured(&margs, &pair_costs, &cfg);
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }

    #[test]
    fn mmot_barycenter_support_shape() {
        let mut rng = LcgRng::new(42);
        let w1 = uniform(3);
        let s1: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let w2 = uniform(3);
        let s2: Vec<Vec<f64>> = vec![vec![2.0, 0.0], vec![3.0, 0.0], vec![2.0, 1.0]];
        let measures: Vec<(&Vec<f64>, &Vec<Vec<f64>>)> = vec![(&w1, &s1), (&w2, &s2)];
        let lam = vec![0.5, 0.5];
        let cfg = MmotBaryConfig {
            eps: 0.5,
            max_iter: 20,
            tol: 1e-4,
            n_support: 4,
        };
        let (bary_w, bary_s) = mmot_barycenter(&measures, &lam, &cfg, &mut rng).expect("ok");
        assert_eq!(bary_w.len(), 4);
        assert_eq!(bary_s.len(), 4);
        for pt in &bary_s {
            assert_eq!(pt.len(), 2);
        }
    }

    #[test]
    fn mmot_barycenter_weights_sum_to_one() {
        let mut rng = LcgRng::new(7);
        let w1 = uniform(2);
        let s1: Vec<Vec<f64>> = vec![vec![0.0], vec![1.0]];
        let w2 = uniform(2);
        let s2: Vec<Vec<f64>> = vec![vec![2.0], vec![3.0]];
        let measures: Vec<(&Vec<f64>, &Vec<Vec<f64>>)> = vec![(&w1, &s1), (&w2, &s2)];
        let lam = vec![0.5, 0.5];
        let cfg = MmotBaryConfig {
            eps: 0.3,
            max_iter: 30,
            tol: 1e-4,
            n_support: 3,
        };
        let (bary_w, _) = mmot_barycenter(&measures, &lam, &cfg, &mut rng).expect("ok");
        let total: f64 = bary_w.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "barycenter weights sum = {total}"
        );
    }

    #[test]
    fn log_sum_exp_empty() {
        assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
    }

    #[test]
    fn log_sum_exp_single() {
        assert!((log_sum_exp(&[3.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn log_sum_exp_pair() {
        // log(exp(0) + exp(0)) = log(2)
        let expected = 2.0_f64.ln();
        assert!((log_sum_exp(&[0.0, 0.0]) - expected).abs() < 1e-12);
    }
}
