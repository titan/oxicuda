//! Byzantine-robust gradient aggregation algorithms for federated learning.
//!
//! Implements the following algorithms for aggregation under adversarial clients:
//!
//! - **Krum / Multi-Krum** — Blanchard et al., "Machine Learning with Adversaries:
//!   Byzantine Tolerant Gradient Descent", NeurIPS 2017.
//! - **Trimmed-Mean / Coordinate-wise Median** — Yin et al., "Byzantine-Robust
//!   Distributed Learning: Towards Optimal Statistical Rates", ICML 2018.
//! - **Bulyan** — El Mhamdi et al., "The Hidden Vulnerability of Distributed
//!   Learning in Byzantium", NeurIPS 2018.

use crate::error::{FedError, FedResult};

/// Configuration shared by all robust aggregation methods.
#[derive(Debug, Clone)]
pub struct RobustAggConfig {
    /// Total number of clients `n`.
    pub n_clients: usize,
    /// Upper bound on adversarial (Byzantine) clients `f`.
    /// Must satisfy `2f + 3 ≤ n` for Krum-family methods.
    pub n_byzantine: usize,
    /// For Multi-Krum: number of gradients to select (`m`).
    /// Must satisfy `m ≥ 1` and `m ≤ n − 2f − 2`.
    pub multi_krum_m: usize,
    /// For Trimmed-Mean: fraction to remove from each end, in `[0, 0.5)`.
    pub trim_fraction: f64,
}

/// Result returned by all robust aggregation methods.
#[derive(Debug, Clone)]
pub struct RobustAggResult {
    /// Aggregated gradient, same length as each client gradient.
    pub aggregated: Vec<f32>,
    /// Indices of selected clients.
    /// Krum/Multi-Krum: the selected subset; others: all clients (0..n).
    pub selected_clients: Vec<usize>,
}

/// Stateless handle providing all Byzantine-robust aggregation methods.
pub struct RobustAggregator;

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Validate common preconditions shared by all methods.
///
/// Checks:
/// - `grads` is non-empty
/// - `cfg.n_clients == grads.len()`
/// - all gradients have the same length > 0
/// - `cfg.n_byzantine < cfg.n_clients`
fn validate_common(grads: &[Vec<f32>], cfg: &RobustAggConfig) -> FedResult<usize> {
    if grads.is_empty() {
        return Err(FedError::EmptyClientList);
    }
    if grads.len() != cfg.n_clients {
        return Err(FedError::DimensionMismatch {
            expected: cfg.n_clients,
            got: grads.len(),
        });
    }
    let dim = grads[0].len();
    if dim == 0 {
        return Err(FedError::EmptyClientList);
    }
    for g in grads.iter().skip(1) {
        if g.len() != dim {
            return Err(FedError::DimensionMismatch {
                expected: dim,
                got: g.len(),
            });
        }
    }
    if cfg.n_byzantine >= cfg.n_clients {
        return Err(FedError::InsufficientClients {
            min: cfg.n_byzantine + 1,
            got: cfg.n_clients,
        });
    }
    Ok(dim)
}

/// Squared Euclidean distance between two gradients (computed in f64 for precision).
fn squared_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let diff = ai as f64 - bi as f64;
            diff * diff
        })
        .sum()
}

/// Compute the Krum score for gradient `i` given a set of candidate indices.
///
/// score(i) = Σ_{j ∈ NN(i, k_neighbors)} ||g_i − g_j||²
/// where NN(i, k) are the `k_neighbors` closest neighbours of `i` among `candidates`
/// (excluding `i` itself).
fn krum_score(i: usize, candidates: &[usize], grads: &[Vec<f32>], k_neighbors: usize) -> f64 {
    // Collect squared distances from i to all other candidates.
    let mut dists: Vec<f64> = candidates
        .iter()
        .filter(|&&j| j != i)
        .map(|&j| squared_distance(&grads[i], &grads[j]))
        .collect();

    // Take the k_neighbors smallest distances.
    dists.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    dists.iter().take(k_neighbors).sum()
}

/// Validate the Krum/Multi-Krum requirement: `n ≥ 2f + 3`.
fn validate_krum_feasibility(n_clients: usize, n_byzantine: usize) -> FedResult<usize> {
    let min_needed = 2 * n_byzantine + 3;
    if n_clients < min_needed {
        return Err(FedError::InsufficientClients {
            min: min_needed,
            got: n_clients,
        });
    }
    // k_neighbors = n - f - 2  (must be ≥ 1 given n ≥ 2f+3 ⟹ n-f-2 ≥ f+1 ≥ 1)
    let k_neighbors = n_clients - n_byzantine - 2;
    Ok(k_neighbors)
}

// ─── Public implementation ────────────────────────────────────────────────────

impl RobustAggregator {
    /// **Krum** robust aggregator (Blanchard et al. 2017).
    ///
    /// Selects the single gradient with the minimum Krum score and returns it
    /// unchanged as the aggregated gradient.
    ///
    /// # Errors
    /// - [`FedError::EmptyClientList`] — grads is empty or dimension is 0.
    /// - [`FedError::DimensionMismatch`] — `n_clients ≠ grads.len()` or unequal gradient lengths.
    /// - [`FedError::InsufficientClients`] — `n_clients < 2·n_byzantine + 3`.
    pub fn krum(grads: &[Vec<f32>], cfg: &RobustAggConfig) -> FedResult<RobustAggResult> {
        validate_common(grads, cfg)?;
        let k_neighbors = validate_krum_feasibility(cfg.n_clients, cfg.n_byzantine)?;

        let candidates: Vec<usize> = (0..cfg.n_clients).collect();

        let best_idx = candidates
            .iter()
            .copied()
            .min_by(|&i, &j| {
                let si = krum_score(i, &candidates, grads, k_neighbors);
                let sj = krum_score(j, &candidates, grads, k_neighbors);
                si.partial_cmp(&sj).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        Ok(RobustAggResult {
            aggregated: grads[best_idx].clone(),
            selected_clients: vec![best_idx],
        })
    }

    /// **Multi-Krum** robust aggregator (Blanchard et al. 2017).
    ///
    /// Iteratively selects `m` gradients with lowest Krum scores (greedy, without
    /// replacement), then returns their coordinate-wise mean.
    ///
    /// # Errors
    /// - Same as [`Self::krum`].
    /// - [`FedError::InsufficientClients`] if `multi_krum_m > n − 2f − 2`.
    /// - [`FedError::InvalidWeight`] if `multi_krum_m == 0`.
    pub fn multi_krum(grads: &[Vec<f32>], cfg: &RobustAggConfig) -> FedResult<RobustAggResult> {
        let dim = validate_common(grads, cfg)?;
        let k_neighbors = validate_krum_feasibility(cfg.n_clients, cfg.n_byzantine)?;

        if cfg.multi_krum_m == 0 {
            return Err(FedError::InvalidWeight { weight: 0.0 });
        }
        let m_max = cfg.n_clients - 2 * cfg.n_byzantine - 2;
        if cfg.multi_krum_m > m_max {
            return Err(FedError::InsufficientClients {
                min: cfg.multi_krum_m + 2 * cfg.n_byzantine + 2,
                got: cfg.n_clients,
            });
        }

        let mut candidates: Vec<usize> = (0..cfg.n_clients).collect();
        let mut selected: Vec<usize> = Vec::with_capacity(cfg.multi_krum_m);

        for _ in 0..cfg.multi_krum_m {
            let best = candidates
                .iter()
                .copied()
                .min_by(|&i, &j| {
                    let si = krum_score(i, &candidates, grads, k_neighbors);
                    let sj = krum_score(j, &candidates, grads, k_neighbors);
                    si.partial_cmp(&sj).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(candidates[0]);
            selected.push(best);
            candidates.retain(|&idx| idx != best);
        }

        // Coordinate-wise mean of selected gradients.
        let mut aggregated = vec![0.0_f64; dim];
        for &idx in &selected {
            for (acc, &val) in aggregated.iter_mut().zip(grads[idx].iter()) {
                *acc += val as f64;
            }
        }
        let inv_m = 1.0 / selected.len() as f64;
        let aggregated: Vec<f32> = aggregated.iter().map(|&v| (v * inv_m) as f32).collect();

        Ok(RobustAggResult {
            aggregated,
            selected_clients: selected,
        })
    }

    /// **Trimmed-Mean** robust aggregator (Yin et al. 2018).
    ///
    /// For each coordinate, sorts values across clients, drops the `floor(n·α)` lowest
    /// and `floor(n·α)` highest values, and returns the mean of the remainder.
    ///
    /// # Errors
    /// - Common errors from `validate_common`.
    /// - [`FedError::InsufficientClients`] if `2·n_trim ≥ n_clients`.
    pub fn trimmed_mean(grads: &[Vec<f32>], cfg: &RobustAggConfig) -> FedResult<RobustAggResult> {
        let dim = validate_common(grads, cfg)?;
        let n = cfg.n_clients;

        let n_trim = (n as f64 * cfg.trim_fraction).floor() as usize;
        if n_trim * 2 >= n {
            return Err(FedError::InsufficientClients {
                min: n_trim * 2 + 1,
                got: n,
            });
        }

        let keep = n - 2 * n_trim;
        let inv_keep = 1.0 / keep as f64;

        let mut aggregated = vec![0.0_f32; dim];
        for d in 0..dim {
            let mut col: Vec<f32> = grads.iter().map(|g| g[d]).collect();
            col.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let sum: f64 = col[n_trim..n - n_trim].iter().map(|&v| v as f64).sum();
            aggregated[d] = (sum * inv_keep) as f32;
        }

        let selected_clients: Vec<usize> = (0..n).collect();
        Ok(RobustAggResult {
            aggregated,
            selected_clients,
        })
    }

    /// **Coordinate-wise Median** robust aggregator (Yin et al. 2018).
    ///
    /// For each coordinate, returns the median of client values.
    /// For even `n`, uses the midpoint of the two central values.
    ///
    /// # Errors
    /// - Common errors from `validate_common`.
    pub fn median(grads: &[Vec<f32>], cfg: &RobustAggConfig) -> FedResult<RobustAggResult> {
        let dim = validate_common(grads, cfg)?;
        let n = cfg.n_clients;

        let mut aggregated = vec![0.0_f32; dim];
        for d in 0..dim {
            let mut col: Vec<f32> = grads.iter().map(|g| g[d]).collect();
            col.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med = if n % 2 == 1 {
                col[n / 2]
            } else {
                (col[n / 2 - 1] + col[n / 2]) / 2.0
            };
            aggregated[d] = med;
        }

        let selected_clients: Vec<usize> = (0..n).collect();
        Ok(RobustAggResult {
            aggregated,
            selected_clients,
        })
    }

    /// **Bulyan** robust aggregator (El Mhamdi et al. 2018).
    ///
    /// Two-phase algorithm:
    /// 1. Iterative Krum selection of `m = n − 2f` gradients (greedy, without replacement).
    ///    Uses `k_neighbors = n − f − 2` per step, scored against the full original
    ///    candidate pool at each iteration — following Algorithm 1 of El Mhamdi et al.
    /// 2. Trimmed-Mean on the selected subset (using `cfg.trim_fraction`).
    ///
    /// Note: Bulyan's selection step is intentionally looser than Multi-Krum's public
    /// API (which requires `m ≤ n − 2f − 2`). The selection here uses the paper's
    /// formulation directly to avoid that constraint.
    ///
    /// # Errors
    /// - Common errors from `validate_common`.
    /// - [`FedError::InsufficientClients`] if `n < 2f + 3` (Krum precondition).
    /// - [`FedError::InsufficientClients`] if `2·n_byzantine ≥ n_clients`.
    pub fn bulyan(grads: &[Vec<f32>], cfg: &RobustAggConfig) -> FedResult<RobustAggResult> {
        validate_common(grads, cfg)?;

        let n = cfg.n_clients;
        let f = cfg.n_byzantine;

        // Krum feasibility: n >= 2f + 3.
        let k_neighbors = validate_krum_feasibility(n, f)?;

        // Need at least 1 element after selection: m = n - 2f >= 1.
        if 2 * f >= n {
            return Err(FedError::InsufficientClients {
                min: 2 * f + 1,
                got: n,
            });
        }
        let m = n - 2 * f;

        // Phase 1: Iterative Krum-based selection of m gradients.
        // At each step, scores are computed against only the remaining candidates,
        // using k_neighbors derived from the original n (per the paper).
        let mut candidates: Vec<usize> = (0..n).collect();
        let mut selected: Vec<usize> = Vec::with_capacity(m);

        for _ in 0..m {
            let best = candidates
                .iter()
                .copied()
                .min_by(|&i, &j| {
                    let si = krum_score(i, &candidates, grads, k_neighbors);
                    let sj = krum_score(j, &candidates, grads, k_neighbors);
                    si.partial_cmp(&sj).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(candidates[0]);
            selected.push(best);
            candidates.retain(|&idx| idx != best);
        }

        // Build the selected gradient subset.
        let selected_grads: Vec<Vec<f32>> = selected.iter().map(|&i| grads[i].clone()).collect();

        // Phase 2: Trimmed-Mean on the selected subset.
        let subset_cfg = RobustAggConfig {
            n_clients: selected_grads.len(),
            n_byzantine: 0,
            multi_krum_m: 1,
            trim_fraction: cfg.trim_fraction,
        };
        let tm_result = Self::trimmed_mean(&selected_grads, &subset_cfg)?;

        Ok(RobustAggResult {
            aggregated: tm_result.aggregated,
            selected_clients: selected,
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n: usize, f: usize) -> RobustAggConfig {
        RobustAggConfig {
            n_clients: n,
            n_byzantine: f,
            multi_krum_m: 1,
            trim_fraction: 0.1,
        }
    }

    fn cfg_with_m(n: usize, f: usize, m: usize, trim: f64) -> RobustAggConfig {
        RobustAggConfig {
            n_clients: n,
            n_byzantine: f,
            multi_krum_m: m,
            trim_fraction: trim,
        }
    }

    // ── Test 1: krum_identical_grads ─────────────────────────────────────────
    #[test]
    fn krum_identical_grads() {
        let g = vec![1.0_f32, 2.0, 3.0];
        let grads = vec![g.clone(), g.clone(), g.clone(), g.clone(), g.clone()];
        let result =
            RobustAggregator::krum(&grads, &cfg(5, 1)).expect("test invariant: valid krum");
        for (&a, &b) in result.aggregated.iter().zip(g.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "identical grads: result must equal grad"
            );
        }
    }

    // ── Test 2: multi_krum_m_eq_1_same_as_krum ───────────────────────────────
    #[test]
    fn multi_krum_m_eq_1_same_as_krum() {
        let grads = vec![
            vec![1.0_f32, 0.0],
            vec![1.1_f32, 0.0],
            vec![1.05_f32, 0.0],
            vec![100.0_f32, 0.0], // outlier
            vec![0.95_f32, 0.0],
        ];
        let c = cfg_with_m(5, 1, 1, 0.1);
        let krum_res = RobustAggregator::krum(&grads, &c).expect("test invariant: valid krum");
        let mk_res =
            RobustAggregator::multi_krum(&grads, &c).expect("test invariant: valid multi_krum");
        // m=1 multi_krum should select the same index as krum.
        assert_eq!(
            krum_res.selected_clients[0], mk_res.selected_clients[0],
            "multi_krum m=1 and krum must select same client"
        );
    }

    // ── Test 3: trimmed_mean_fraction_zero_is_mean ───────────────────────────
    #[test]
    fn trimmed_mean_fraction_zero_is_mean() {
        let grads = vec![vec![1.0_f32, 4.0], vec![3.0_f32, 2.0], vec![5.0_f32, 6.0]];
        let c = cfg_with_m(3, 0, 1, 0.0);
        let result =
            RobustAggregator::trimmed_mean(&grads, &c).expect("test invariant: valid trimmed_mean");
        assert!((result.aggregated[0] - 3.0).abs() < 1e-5);
        assert!((result.aggregated[1] - 4.0).abs() < 1e-5);
    }

    // ── Test 4: median_three_elements ────────────────────────────────────────
    #[test]
    fn median_three_elements() {
        let grads = vec![vec![1.0_f32], vec![3.0_f32], vec![2.0_f32]];
        let c = cfg(3, 0);
        let result = RobustAggregator::median(&grads, &c).expect("test invariant: valid median");
        assert!(
            (result.aggregated[0] - 2.0).abs() < 1e-5,
            "median of [1,2,3]=2"
        );
    }

    // ── Test 5: median_even_elements ─────────────────────────────────────────
    #[test]
    fn median_even_elements() {
        let grads = vec![vec![1.0_f32], vec![2.0_f32], vec![3.0_f32], vec![4.0_f32]];
        let c = cfg(4, 0);
        let result =
            RobustAggregator::median(&grads, &c).expect("test invariant: valid median even");
        // median of [1,2,3,4] = (2+3)/2 = 2.5
        assert!(
            (result.aggregated[0] - 2.5).abs() < 1e-5,
            "even median must be 2.5"
        );
    }

    // ── Test 6: krum_outlier_filtered ────────────────────────────────────────
    #[test]
    fn krum_outlier_filtered() {
        // One extreme outlier; krum should NOT select it.
        let grads = vec![
            vec![1.0_f32, 1.0],
            vec![1.1_f32, 0.9],
            vec![0.9_f32, 1.1],
            vec![1.05_f32, 0.95],
            vec![1000.0_f32, 1000.0], // outlier: index 4
        ];
        let result =
            RobustAggregator::krum(&grads, &cfg(5, 1)).expect("test invariant: valid krum outlier");
        assert_ne!(
            result.selected_clients[0], 4,
            "krum must not select the outlier"
        );
    }

    // ── Test 7: multi_krum_returns_m_indices ─────────────────────────────────
    #[test]
    fn multi_krum_returns_m_indices() {
        // n=7, f=1 → k_neighbors=4, m_max=n-2f-2=3; use m=2 (valid).
        let grads = vec![
            vec![0.0_f32],
            vec![0.1_f32],
            vec![0.2_f32],
            vec![0.3_f32],
            vec![0.15_f32],
            vec![0.05_f32],
            vec![100.0_f32], // outlier
        ];
        let c = cfg_with_m(7, 1, 2, 0.1);
        let result =
            RobustAggregator::multi_krum(&grads, &c).expect("test invariant: valid multi_krum m=2");
        assert_eq!(
            result.selected_clients.len(),
            2,
            "must return exactly m=2 indices"
        );
    }

    // ── Test 8: multi_krum_aggregated_len ────────────────────────────────────
    #[test]
    fn multi_krum_aggregated_len() {
        let grad_len = 8_usize;
        let grads: Vec<Vec<f32>> = (0..7).map(|i| vec![i as f32; grad_len]).collect();
        let c = cfg_with_m(7, 1, 2, 0.0);
        let result =
            RobustAggregator::multi_krum(&grads, &c).expect("test invariant: valid multi_krum len");
        assert_eq!(result.aggregated.len(), grad_len);
    }

    // ── Test 9: trimmed_mean_len ─────────────────────────────────────────────
    #[test]
    fn trimmed_mean_len() {
        let grad_len = 5_usize;
        let grads: Vec<Vec<f32>> = (0..6).map(|i| vec![i as f32; grad_len]).collect();
        let c = cfg_with_m(6, 0, 1, 0.1);
        let result = RobustAggregator::trimmed_mean(&grads, &c)
            .expect("test invariant: valid trimmed_mean len");
        assert_eq!(result.aggregated.len(), grad_len);
    }

    // ── Test 10: median_len ──────────────────────────────────────────────────
    #[test]
    fn median_len() {
        let grad_len = 4_usize;
        let grads: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32; grad_len]).collect();
        let c = cfg(5, 1);
        let result =
            RobustAggregator::median(&grads, &c).expect("test invariant: valid median len");
        assert_eq!(result.aggregated.len(), grad_len);
    }

    // ── Test 11: bulyan_no_error ─────────────────────────────────────────────
    #[test]
    fn bulyan_no_error() {
        // n=10, f=2, m=n-2f=6 → needs n≥2f+3=7 ✓
        let grads: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, (i as f32) * 2.0]).collect();
        let c = cfg_with_m(10, 2, 6, 0.1);
        RobustAggregator::bulyan(&grads, &c)
            .expect("test invariant: bulyan must not error for valid config");
    }

    // ── Test 12: err_empty_grads ─────────────────────────────────────────────
    #[test]
    fn err_empty_grads() {
        let grads: Vec<Vec<f32>> = vec![];
        let c = cfg(0, 0);
        assert!(matches!(
            RobustAggregator::krum(&grads, &c),
            Err(FedError::EmptyClientList)
        ));
    }

    // ── Test 13: err_n_clients_mismatch ──────────────────────────────────────
    #[test]
    fn err_n_clients_mismatch() {
        // cfg says n_clients=5 but only 3 grads provided.
        let grads = vec![vec![1.0_f32], vec![2.0_f32], vec![3.0_f32]];
        let c = cfg(5, 1);
        assert!(matches!(
            RobustAggregator::krum(&grads, &c),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    // ── Test 14: err_grads_different_len ─────────────────────────────────────
    #[test]
    fn err_grads_different_len() {
        let grads = vec![
            vec![1.0_f32, 2.0],
            vec![1.0_f32, 2.0, 3.0], // different length
            vec![1.0_f32, 2.0],
            vec![1.0_f32, 2.0],
            vec![1.0_f32, 2.0],
        ];
        let c = cfg(5, 1);
        assert!(matches!(
            RobustAggregator::krum(&grads, &c),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    // ── Test 15: err_krum_insufficient_clients ───────────────────────────────
    #[test]
    fn err_krum_insufficient_clients() {
        // n=4, f=2 → need 2*2+3=7, but got 4.
        let grads: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32]).collect();
        let c = cfg(4, 2);
        assert!(matches!(
            RobustAggregator::krum(&grads, &c),
            Err(FedError::InsufficientClients { .. })
        ));
    }

    // ── Test 16: err_trim_fraction_half ──────────────────────────────────────
    #[test]
    fn err_trim_fraction_half() {
        // trim_fraction=0.5 → n_trim=1 from 2 total → 2*1>=2 → error.
        let grads = vec![vec![1.0_f32], vec![2.0_f32]];
        let c = cfg_with_m(2, 0, 1, 0.5);
        assert!(matches!(
            RobustAggregator::trimmed_mean(&grads, &c),
            Err(FedError::InsufficientClients { .. })
        ));
    }

    // ── Test 17: krum_score_correct ───────────────────────────────────────────
    #[test]
    fn krum_score_correct() {
        // Cluster A at ~0, cluster B far away. Krum should select from cluster A.
        let grads = vec![
            vec![0.0_f32],   // cluster A
            vec![0.1_f32],   // cluster A
            vec![0.05_f32],  // cluster A
            vec![0.08_f32],  // cluster A
            vec![500.0_f32], // outlier
        ];
        // n=5, f=1 → need 2+3=5 ✓, k_neighbors=2
        let result =
            RobustAggregator::krum(&grads, &cfg(5, 1)).expect("test invariant: krum_score_correct");
        // The selected gradient must be from cluster A (indices 0-3, not 4).
        assert!(
            result.selected_clients[0] < 4,
            "krum should select from cluster A, got {}",
            result.selected_clients[0]
        );
    }
}
