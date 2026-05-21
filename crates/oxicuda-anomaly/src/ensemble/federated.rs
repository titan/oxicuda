//! Federated Anomaly Detection — privacy-preserving distributed anomaly scoring.
//!
//! Each client trains a local linear anomaly detector on its private data.
//! A central aggregator combines client models via FedAvg-style parameter averaging
//! or score-level averaging, without accessing raw client data.
//!
//! # Client Model
//!
//! Each client maintains a weight vector `w ∈ ℝ^d` that defines a learned
//! projection direction. The anomaly score for a sample `x` is the **linear
//! reconstruction error** — the squared residual after projecting `x` onto `w`:
//!
//! ```text
//! score(x) = ‖x − (wᵀx / ‖w‖²) w‖²
//! ```
//!
//! This is equivalent to the squared distance from `x` to its projection onto
//! the rank-1 subspace spanned by `w`, and equals zero when `x` is a scalar
//! multiple of `w`.
//!
//! # Training (local SGD)
//!
//! The unsupervised objective is to minimise the **mean reconstruction error**
//! over local training data:
//!
//! ```text
//! L(w) = (1/n) Σᵢ score(xᵢ)
//! ```
//!
//! SGD update for `w` (analytical gradient derived from the projection residual).
//!
//! # Aggregation Strategies
//!
//! * `ScoreAverage`: compute scores from every client model independently, then average.
//! * `ParameterAverage` / `WeightedByDataSize`: federated averaging — global weight
//!   is the (data-size-weighted) mean of client weights; clients are updated with
//!   the global weight before the next round.
//!
//! # Reference
//! McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).
//! Communication-efficient learning of deep networks from decentralized data.
//! *AISTATS 2017*.

use crate::error::{AnomalyError, AnomalyResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// How client models are combined by the aggregator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationMethod {
    /// Average anomaly scores across all client models at inference time.
    ScoreAverage,
    /// FedAvg: aggregate parameters as (uniformly) weighted mean; redistribute to clients.
    ParameterAverage,
    /// FedAvg with data-size weighting: `w_global = Σ (n_k / N) * w_k`.
    WeightedByDataSize,
}

/// Configuration for federated anomaly detection.
#[derive(Debug, Clone)]
pub struct FederatedConfig {
    /// Number of participating clients.
    pub n_clients: usize,
    /// Number of federated rounds (global aggregation steps).
    pub n_rounds: usize,
    /// Number of local SGD epochs per round per client.
    pub local_epochs: usize,
    /// How to combine client models.
    pub aggregation: AggregationMethod,
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            n_clients: 3,
            n_rounds: 5,
            local_epochs: 10,
            aggregation: AggregationMethod::WeightedByDataSize,
        }
    }
}

// ─── Client model ─────────────────────────────────────────────────────────────

/// Local anomaly model held by one federated client.
///
/// Defines a rank-1 linear reconstruction scorer.
#[derive(Debug, Clone)]
pub struct ClientModel {
    /// Learned projection direction `w ∈ ℝ^d`.
    pub weights: Vec<f64>,
    /// Bias vector (unused in the score formula, reserved for future extensions).
    pub bias: Vec<f64>,
    /// Number of local training samples.
    pub data_size: usize,
    /// Zero-indexed client identifier.
    pub client_id: usize,
}

// ─── Fitted model ─────────────────────────────────────────────────────────────

/// Fitted federated anomaly detection model.
#[derive(Debug, Clone)]
pub struct FederatedAnomalyFit {
    /// Aggregated global weight vector `w_global ∈ ℝ^d`.
    pub global_weights: Vec<f64>,
    /// Number of input features.
    pub n_features: usize,
    /// Individual client models (post-training, post-aggregation).
    pub client_models: Vec<ClientModel>,
    /// Number of federated rounds completed.
    pub rounds_completed: usize,
}

// ─── Score helpers ────────────────────────────────────────────────────────────

/// Compute the linear reconstruction error for one sample `x` given weight `w`.
///
/// `score = ‖x − (wᵀx / (‖w‖² + ε)) w‖²`
fn linear_recon_score(x: &[f64], w: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), w.len());
    let d = x.len();

    let w_sq: f64 = w.iter().map(|v| v * v).sum::<f64>();
    if w_sq < 1e-20 {
        // w is zero — score equals ‖x‖²
        return x.iter().map(|v| v * v).sum::<f64>();
    }

    let dot: f64 = x.iter().zip(w.iter()).map(|(xi, wi)| xi * wi).sum::<f64>();
    let proj_coeff = dot / (w_sq + 1e-12);

    let mut score = 0.0_f64;
    for i in 0..d {
        let residual = x[i] - proj_coeff * w[i];
        score += residual * residual;
    }
    score
}

/// Gradient of `linear_recon_score` w.r.t. `w`, evaluated at `(x, w)`.
///
/// Derived analytically from the projection residual formula.
fn linear_recon_score_grad(x: &[f64], w: &[f64]) -> Vec<f64> {
    let d = x.len();
    let w_sq: f64 = w.iter().map(|v| v * v).sum::<f64>() + 1e-12;
    let dot: f64 = x.iter().zip(w.iter()).map(|(xi, wi)| xi * wi).sum::<f64>();

    // r_i = x_i - (dot / w_sq) * w_i
    let coeff = dot / w_sq;
    let residuals: Vec<f64> = (0..d).map(|i| x[i] - coeff * w[i]).collect();

    // dL/dw_j = d/dw_j Σ_i r_i²
    //         = 2 Σ_i r_i * d(r_i)/dw_j
    //
    // d(r_i)/dw_j = d/dw_j [x_i - (dot/w_sq) * w_i]
    //             = - d/dw_j [(dot/w_sq) * w_i]
    //
    // Let c = dot / w_sq, then:
    // d(c * w_i)/dw_j = δ_{ij} * c + w_i * d(c)/dw_j
    // d(c)/dw_j = d(dot/w_sq)/dw_j
    //           = (x_j * w_sq - dot * 2 w_j) / w_sq²
    //           = (x_j - 2c w_j) / w_sq
    //
    // So d(r_i)/dw_j = -(δ_{ij} * c + w_i * (x_j - 2c w_j) / w_sq)
    //
    // dL/dw_j = 2 Σ_i r_i * [-(δ_{ij} * c + w_i * (x_j - 2c w_j) / w_sq)]
    //         = -2 [c * r_j + (Σ_i r_i w_i) * (x_j - 2c w_j) / w_sq]

    let r_dot_w: f64 = residuals
        .iter()
        .zip(w.iter())
        .map(|(r, wi)| r * wi)
        .sum::<f64>();

    let mut grad = vec![0.0_f64; d];
    for j in 0..d {
        grad[j] = -2.0 * (coeff * residuals[j] + r_dot_w * (x[j] - 2.0 * coeff * w[j]) / w_sq);
    }
    grad
}

// ─── Local training ───────────────────────────────────────────────────────────

/// Run `epochs` SGD passes on client `c`'s local data to minimise mean reconstruction error.
fn local_train(client: &mut ClientModel, data: &[f64], n: usize, epochs: usize, lr: f64) {
    let d = client.weights.len();
    if n == 0 || d == 0 {
        return;
    }

    for _ in 0..epochs {
        // Mini-batch SGD (stochastic, one sample at a time)
        for i in 0..n {
            let xi = &data[i * d..(i + 1) * d];
            let grad = linear_recon_score_grad(xi, &client.weights);
            // Normalise by n (mean loss) then apply lr
            for (w_j, g_j) in client.weights.iter_mut().zip(grad.iter()) {
                *w_j -= lr / n as f64 * g_j;
            }
        }
        // L2 normalise weights to avoid collapse / explosion
        let norm: f64 = client.weights.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for w_j in client.weights.iter_mut() {
                *w_j /= norm;
            }
        }
    }
}

// ─── Aggregation ─────────────────────────────────────────────────────────────

/// Compute the (possibly weighted) average of client weight vectors.
fn aggregate_weights(clients: &[ClientModel], method: AggregationMethod) -> Vec<f64> {
    if clients.is_empty() {
        return Vec::new();
    }
    let d = clients[0].weights.len();
    let total_data: usize = clients.iter().map(|c| c.data_size).sum();

    let mut global = vec![0.0_f64; d];

    match method {
        AggregationMethod::ScoreAverage => {
            // In score-average mode, all client models are kept separately;
            // we still store the uniform average for `global_weights`.
            let weight = 1.0 / clients.len() as f64;
            for client in clients {
                for (g, &w) in global.iter_mut().zip(client.weights.iter()) {
                    *g += weight * w;
                }
            }
        }
        AggregationMethod::ParameterAverage => {
            let weight = 1.0 / clients.len() as f64;
            for client in clients {
                for (g, &w) in global.iter_mut().zip(client.weights.iter()) {
                    *g += weight * w;
                }
            }
        }
        AggregationMethod::WeightedByDataSize => {
            if total_data == 0 {
                // Fall back to uniform
                let weight = 1.0 / clients.len() as f64;
                for client in clients {
                    for (g, &w) in global.iter_mut().zip(client.weights.iter()) {
                        *g += weight * w;
                    }
                }
            } else {
                for client in clients {
                    let frac = client.data_size as f64 / total_data as f64;
                    for (g, &w) in global.iter_mut().zip(client.weights.iter()) {
                        *g += frac * w;
                    }
                }
            }
        }
    }

    // L2 normalise the aggregate
    let norm: f64 = global.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 1e-10 {
        for g in global.iter_mut() {
            *g /= norm;
        }
    }
    global
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Fit a federated anomaly detection model.
///
/// `data_per_client[k]` is a row-major `[n_per_client[k] × d]` slice for client `k`.
pub fn federated_fit(
    data_per_client: &[&[f64]],
    n_per_client: &[usize],
    d: usize,
    cfg: &FederatedConfig,
    seed: u64,
) -> AnomalyResult<FederatedAnomalyFit> {
    // Validation
    if cfg.n_clients == 0 {
        return Err(AnomalyError::InvalidFeatureCount { n: 0 });
    }
    if d == 0 {
        return Err(AnomalyError::InvalidFeatureCount { n: 0 });
    }
    if data_per_client.len() != cfg.n_clients || n_per_client.len() != cfg.n_clients {
        return Err(AnomalyError::DimensionMismatch {
            expected: cfg.n_clients,
            got: data_per_client.len(),
        });
    }
    for (k, (&n_k, &data_k)) in n_per_client.iter().zip(data_per_client.iter()).enumerate() {
        if data_k.len() != n_k * d {
            return Err(AnomalyError::DimensionMismatch {
                expected: n_k * d,
                got: data_k.len(),
            });
        }
        // Warn on empty clients rather than failing (just skip training)
        let _ = k;
    }

    let lr = 1e-2; // local SGD learning rate (fixed for simplicity)
    let mut rng = LcgRng::new(seed);

    // ── Initialise client models ──────────────────────────────────────────────
    let mut clients: Vec<ClientModel> = (0..cfg.n_clients)
        .map(|k| {
            // Random unit initialisation
            let mut w: Vec<f64> = (0..d).map(|_| rng.next_normal() as f64).collect();
            let norm: f64 = w.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
            for wi in w.iter_mut() {
                *wi /= norm;
            }
            ClientModel {
                weights: w,
                bias: vec![0.0_f64; d],
                data_size: n_per_client[k],
                client_id: k,
            }
        })
        .collect();

    // ── Federated training rounds ─────────────────────────────────────────────
    for _round in 0..cfg.n_rounds {
        // Each client performs local training
        for k in 0..cfg.n_clients {
            let n_k = n_per_client[k];
            let data_k = data_per_client[k];
            if n_k == 0 {
                continue;
            }
            local_train(&mut clients[k], data_k, n_k, cfg.local_epochs, lr);
        }

        // Aggregator step (for parameter-averaging methods)
        if cfg.aggregation != AggregationMethod::ScoreAverage {
            let global_w = aggregate_weights(&clients, cfg.aggregation);
            // Redistribute global weights to all clients
            for client in clients.iter_mut() {
                client.weights.clone_from(&global_w);
            }
        }
    }

    // ── Compute final global weights ──────────────────────────────────────────
    let global_weights = aggregate_weights(&clients, cfg.aggregation);

    Ok(FederatedAnomalyFit {
        global_weights,
        n_features: d,
        client_models: clients,
        rounds_completed: cfg.n_rounds,
    })
}

/// Compute anomaly scores for `n` test samples.
///
/// Scoring strategy depends on the aggregation method stored in `fit`:
/// * For `ScoreAverage`: average scores from all client models.
/// * For `ParameterAverage` / `WeightedByDataSize`: use the global weight vector.
///
/// `x` is row-major `[n × d]`. Returns `[n]` scores.
pub fn federated_score(fit: &FederatedAnomalyFit, x: &[f64], n: usize) -> AnomalyResult<Vec<f64>> {
    if n == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    let d = fit.n_features;
    if x.len() != n * d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * d,
            got: x.len(),
        });
    }

    let mut scores = vec![0.0_f64; n];
    let n_clients = fit.client_models.len();

    for i in 0..n {
        let xi = &x[i * d..(i + 1) * d];
        if n_clients > 0 {
            // Average across client models
            let mut sum = 0.0_f64;
            for client in &fit.client_models {
                sum += linear_recon_score(xi, &client.weights);
            }
            scores[i] = sum / n_clients as f64;
        } else {
            scores[i] = linear_recon_score(xi, &fit.global_weights);
        }
    }

    Ok(scores)
}

/// Predict binary anomaly labels (score > threshold → anomaly).
pub fn federated_predict(
    fit: &FederatedAnomalyFit,
    x: &[f64],
    n: usize,
    threshold: f64,
) -> AnomalyResult<Vec<bool>> {
    let scores = federated_score(fit, x, n)?;
    Ok(scores.iter().map(|&s| s > threshold).collect())
}

/// Compute anomaly scores using only one specific client's model.
pub fn federated_client_score(
    fit: &FederatedAnomalyFit,
    client_id: usize,
    x: &[f64],
    n: usize,
) -> AnomalyResult<Vec<f64>> {
    if client_id >= fit.client_models.len() {
        return Err(AnomalyError::DimensionMismatch {
            expected: fit.client_models.len(),
            got: client_id,
        });
    }
    if n == 0 {
        return Err(AnomalyError::EmptyInput);
    }
    let d = fit.n_features;
    if x.len() != n * d {
        return Err(AnomalyError::DimensionMismatch {
            expected: n * d,
            got: x.len(),
        });
    }

    let client = &fit.client_models[client_id];
    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let xi = &x[i * d..(i + 1) * d];
        scores.push(linear_recon_score(xi, &client.weights));
    }
    Ok(scores)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(n: usize, d: usize, seed: u64) -> Vec<f64> {
        let mut rng = LcgRng::new(seed);
        (0..n * d).map(|_| rng.next_normal() as f64 * 0.5).collect()
    }

    fn default_cfg() -> FederatedConfig {
        FederatedConfig {
            n_clients: 3,
            n_rounds: 3,
            local_epochs: 5,
            aggregation: AggregationMethod::WeightedByDataSize,
        }
    }

    // ── Test 1: fit returns Ok ────────────────────────────────────────────────

    #[test]
    fn federated_fit_ok() {
        let d = 4_usize;
        let cfg = default_cfg();
        let data0 = make_data(10, d, 1);
        let data1 = make_data(10, d, 2);
        let data2 = make_data(10, d, 3);
        let result = federated_fit(&[&data0, &data1, &data2], &[10, 10, 10], d, &cfg, 42);
        assert!(result.is_ok(), "{:?}", result.err());
    }

    // ── Test 2: score returns correct length ──────────────────────────────────

    #[test]
    fn federated_score_length() {
        let d = 4_usize;
        let cfg = default_cfg();
        let data0 = make_data(10, d, 10);
        let data1 = make_data(10, d, 11);
        let data2 = make_data(10, d, 12);
        let fit = federated_fit(&[&data0, &data1, &data2], &[10, 10, 10], d, &cfg, 42).unwrap();
        let test = make_data(7, d, 99);
        let scores = federated_score(&fit, &test, 7).unwrap();
        assert_eq!(scores.len(), 7);
    }

    // ── Test 3: scores are finite and non-negative ────────────────────────────

    #[test]
    fn federated_scores_finite_nonneg() {
        let d = 6_usize;
        let cfg = default_cfg();
        let data0 = make_data(15, d, 20);
        let data1 = make_data(15, d, 21);
        let data2 = make_data(15, d, 22);
        let fit = federated_fit(&[&data0, &data1, &data2], &[15, 15, 15], d, &cfg, 7).unwrap();
        let test = make_data(10, d, 55);
        let scores = federated_score(&fit, &test, 10).unwrap();
        for (i, &s) in scores.iter().enumerate() {
            assert!(s.is_finite(), "score[{i}] = {s} not finite");
            assert!(s >= 0.0, "score[{i}] = {s} negative");
        }
    }

    // ── Test 4: predict returns correct length ────────────────────────────────

    #[test]
    fn federated_predict_len() {
        let d = 4_usize;
        let cfg = default_cfg();
        let data0 = make_data(10, d, 30);
        let data1 = make_data(10, d, 31);
        let data2 = make_data(10, d, 32);
        let fit = federated_fit(&[&data0, &data1, &data2], &[10, 10, 10], d, &cfg, 42).unwrap();
        let test = make_data(5, d, 77);
        let preds = federated_predict(&fit, &test, 5, 0.5).unwrap();
        assert_eq!(preds.len(), 5);
    }

    // ── Test 5: client score uses correct client model ────────────────────────

    #[test]
    fn federated_client_score_ok() {
        let d = 4_usize;
        let cfg = default_cfg();
        let data0 = make_data(10, d, 40);
        let data1 = make_data(10, d, 41);
        let data2 = make_data(10, d, 42);
        let fit = federated_fit(&[&data0, &data1, &data2], &[10, 10, 10], d, &cfg, 42).unwrap();
        let test = make_data(4, d, 88);
        let s0 = federated_client_score(&fit, 0, &test, 4).unwrap();
        let s2 = federated_client_score(&fit, 2, &test, 4).unwrap();
        assert_eq!(s0.len(), 4);
        assert_eq!(s2.len(), 4);
        // Client scores may differ (different local data)
        assert!(s0.iter().all(|v| v.is_finite()));
        assert!(s2.iter().all(|v| v.is_finite()));
    }

    // ── Test 6: invalid client_id returns error ───────────────────────────────

    #[test]
    fn federated_client_score_invalid_id() {
        let d = 4_usize;
        let cfg = default_cfg();
        let data0 = make_data(10, d, 50);
        let data1 = make_data(10, d, 51);
        let data2 = make_data(10, d, 52);
        let fit = federated_fit(&[&data0, &data1, &data2], &[10, 10, 10], d, &cfg, 42).unwrap();
        let test = make_data(3, d, 66);
        let result = federated_client_score(&fit, 99, &test, 3);
        assert!(result.is_err());
    }

    // ── Test 7: ScoreAverage aggregation ─────────────────────────────────────

    #[test]
    fn federated_score_average_method() {
        let d = 4_usize;
        let cfg = FederatedConfig {
            n_clients: 2,
            n_rounds: 2,
            local_epochs: 3,
            aggregation: AggregationMethod::ScoreAverage,
        };
        let data0 = make_data(10, d, 60);
        let data1 = make_data(10, d, 61);
        let fit = federated_fit(&[&data0, &data1], &[10, 10], d, &cfg, 42).unwrap();
        let test = make_data(5, d, 70);
        let scores = federated_score(&fit, &test, 5).unwrap();
        assert_eq!(scores.len(), 5);
        assert!(scores.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    // ── Test 8: ParameterAverage aggregation ─────────────────────────────────

    #[test]
    fn federated_parameter_average_method() {
        let d = 4_usize;
        let cfg = FederatedConfig {
            n_clients: 3,
            n_rounds: 3,
            local_epochs: 5,
            aggregation: AggregationMethod::ParameterAverage,
        };
        let data0 = make_data(8, d, 70);
        let data1 = make_data(8, d, 71);
        let data2 = make_data(8, d, 72);
        let fit = federated_fit(&[&data0, &data1, &data2], &[8, 8, 8], d, &cfg, 1).unwrap();
        assert!(fit.global_weights.iter().all(|v| v.is_finite()));
        assert_eq!(fit.rounds_completed, 3);
    }

    // ── Test 9: rounds_completed stored correctly ─────────────────────────────

    #[test]
    fn federated_rounds_stored() {
        let d = 4_usize;
        let mut cfg = default_cfg();
        cfg.n_rounds = 7;
        let data0 = make_data(5, d, 80);
        let data1 = make_data(5, d, 81);
        let data2 = make_data(5, d, 82);
        let fit = federated_fit(&[&data0, &data1, &data2], &[5, 5, 5], d, &cfg, 42).unwrap();
        assert_eq!(fit.rounds_completed, 7);
    }

    // ── Test 10: deterministic with same seed ─────────────────────────────────

    #[test]
    fn federated_deterministic() {
        let d = 4_usize;
        let cfg = default_cfg();
        let data0 = make_data(10, d, 90);
        let data1 = make_data(10, d, 91);
        let data2 = make_data(10, d, 92);

        let fit1 = federated_fit(&[&data0, &data1, &data2], &[10, 10, 10], d, &cfg, 777).unwrap();
        let fit2 = federated_fit(&[&data0, &data1, &data2], &[10, 10, 10], d, &cfg, 777).unwrap();

        for (a, b) in fit1.global_weights.iter().zip(fit2.global_weights.iter()) {
            assert_eq!(a, b);
        }
    }

    // ── Test 11: high threshold → zero anomaly predictions ───────────────────

    #[test]
    fn federated_predict_high_threshold_zero() {
        let d = 4_usize;
        let cfg = default_cfg();
        let data0 = make_data(10, d, 100);
        let data1 = make_data(10, d, 101);
        let data2 = make_data(10, d, 102);
        let fit = federated_fit(&[&data0, &data1, &data2], &[10, 10, 10], d, &cfg, 42).unwrap();
        let test = make_data(10, d, 200);
        let preds = federated_predict(&fit, &test, 10, 1e12).unwrap();
        assert!(preds.iter().all(|&b| !b));
    }

    // ── Test 12: n_features stored correctly ─────────────────────────────────

    #[test]
    fn federated_n_features_stored() {
        let d = 7_usize;
        let cfg = FederatedConfig {
            n_clients: 2,
            n_rounds: 1,
            local_epochs: 1,
            aggregation: AggregationMethod::WeightedByDataSize,
        };
        let data0 = make_data(5, d, 110);
        let data1 = make_data(5, d, 111);
        let fit = federated_fit(&[&data0, &data1], &[5, 5], d, &cfg, 42).unwrap();
        assert_eq!(fit.n_features, d);
        assert_eq!(fit.global_weights.len(), d);
    }

    // ── Test 13: linear_recon_score zero for exact projection direction ────────

    #[test]
    fn recon_score_zero_for_aligned_vector() {
        // If x is exactly parallel to w, residual should be ≈ 0
        let w = vec![1.0_f64, 0.0, 0.0, 0.0];
        let x = vec![5.0_f64, 0.0, 0.0, 0.0]; // parallel to w
        let score = linear_recon_score(&x, &w);
        assert!(
            score.abs() < 1e-10,
            "score should be ~0 for aligned vector, got {score}"
        );
    }

    // ── Test 14: linear_recon_score positive for orthogonal vector ─────────────

    #[test]
    fn recon_score_positive_for_orthogonal() {
        let w = vec![1.0_f64, 0.0, 0.0, 0.0];
        let x = vec![0.0_f64, 3.0, 0.0, 0.0]; // orthogonal to w → full residual
        let score = linear_recon_score(&x, &w);
        // residual = x - (0/1)*w = x, so score = 9
        assert!((score - 9.0).abs() < 1e-10, "expected score=9, got {score}");
    }

    // ── Test 15: different client data sizes weight correctly ─────────────────

    #[test]
    fn federated_weighted_aggregation_runs() {
        let d = 4_usize;
        let cfg = FederatedConfig {
            n_clients: 3,
            n_rounds: 2,
            local_epochs: 3,
            aggregation: AggregationMethod::WeightedByDataSize,
        };
        // Unequal data sizes
        let data0 = make_data(20, d, 120);
        let data1 = make_data(5, d, 121);
        let data2 = make_data(10, d, 122);
        let fit = federated_fit(&[&data0, &data1, &data2], &[20, 5, 10], d, &cfg, 42).unwrap();
        assert_eq!(fit.client_models[0].data_size, 20);
        assert_eq!(fit.client_models[1].data_size, 5);
        assert!(fit.global_weights.iter().all(|v| v.is_finite()));
    }

    // ── Test 16: empty input error on score ───────────────────────────────────

    #[test]
    fn federated_score_empty_error() {
        let d = 4_usize;
        let cfg = default_cfg();
        let data0 = make_data(5, d, 130);
        let data1 = make_data(5, d, 131);
        let data2 = make_data(5, d, 132);
        let fit = federated_fit(&[&data0, &data1, &data2], &[5, 5, 5], d, &cfg, 42).unwrap();
        let result = federated_score(&fit, &[], 0);
        assert!(result.is_err());
    }
}
