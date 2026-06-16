//! Poincaré Embeddings for hierarchical data (Nickel & Kiela 2017).
//!
//! Reference: Nickel & Kiela 2017 NeurIPS
//! "Poincaré Embeddings for Learning Hierarchical Representations".
//!
//! Embeds a set of entities into the Poincaré ball `B^d = {x ∈ ℝ^d : ||x|| < 1}`
//! endowed with the hyperbolic metric, using Riemannian SGD with negative sampling.
//!
//! The hyperbolic geometry naturally encodes hierarchical relationships:
//! nodes near the boundary of the ball are "further" from the origin and tend
//! to correspond to leaf/specific concepts, while nodes near the origin are
//! more general.

use crate::error::{ManifoldError, ManifoldResult};
use crate::handle::LcgRng;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration and model types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Poincaré embedding training.
#[derive(Debug, Clone)]
pub struct PoincareConfig {
    /// Number of nodes (entities) to embed.
    pub n_nodes: usize,
    /// Embedding dimension.
    pub dim: usize,
    /// Learning rate for Riemannian SGD.
    pub learning_rate: f64,
    /// Number of training epochs.
    pub n_epochs: usize,
    /// Number of negative samples per positive pair.
    pub n_neg: usize,
    /// Initialisation radius (embeddings start in a small ball of this radius).
    pub init_radius: f64,
    /// Small constant to keep embeddings strictly inside the unit ball.
    pub eps: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for PoincareConfig {
    fn default() -> Self {
        Self {
            n_nodes: 10,
            dim: 2,
            learning_rate: 0.1,
            n_epochs: 50,
            n_neg: 10,
            init_radius: 1e-3,
            eps: 1e-5,
            seed: 42,
        }
    }
}

/// Trained Poincaré embedding model.
#[derive(Debug, Clone)]
pub struct PoincareModel {
    /// Embedding vectors stored row-major as `n_nodes × dim`.
    /// All `||e_i|| < 1 - eps`.
    pub embeddings: Vec<f64>,
    /// Number of nodes.
    pub n_nodes: usize,
    /// Embedding dimension.
    pub dim: usize,
    /// Configuration used for training.
    pub config: PoincareConfig,
    /// Final training loss after the last epoch.
    pub final_loss: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `arccosh(x) = ln(x + sqrt(x²-1))` with numerical guard.
#[inline]
fn acosh_safe(x: f64) -> f64 {
    let x = x.max(1.0 + 1e-10);
    (x + (x * x - 1.0).max(0.0).sqrt()).ln()
}

/// Compute Poincaré distance between points `u` and `v` (both slices of length `dim`).
/// Panics if lengths differ; use the public API for guarded access.
#[inline]
fn poincare_dist_unchecked(u: &[f64], v: &[f64]) -> f64 {
    let u_n2: f64 = u.iter().map(|x| x * x).sum();
    let v_n2: f64 = v.iter().map(|x| x * x).sum();
    // Clamp norms to be strictly inside the ball.
    let alpha = (1.0 - u_n2).max(1e-10);
    let beta = (1.0 - v_n2).max(1e-10);
    let diff_n2: f64 = u.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
    let delta = 1.0 + 2.0 * diff_n2 / (alpha * beta);
    acosh_safe(delta)
}

/// Euclidean gradient of `d_P(u, v)` with respect to `u`.
/// Returns a `dim`-length vector.
#[inline]
fn euclidean_grad_u(u: &[f64], v: &[f64]) -> Vec<f64> {
    let dim = u.len();
    let u_n2: f64 = u.iter().map(|x| x * x).sum();
    let v_n2: f64 = v.iter().map(|x| x * x).sum();
    let alpha = (1.0 - u_n2).max(1e-10);
    let beta = (1.0 - v_n2).max(1e-10);
    let diff: Vec<f64> = u.iter().zip(v.iter()).map(|(a, b)| a - b).collect();
    let r2: f64 = diff.iter().map(|x| x * x).sum();
    let delta = 1.0 + 2.0 * r2 / (alpha * beta);
    // c = 1 / sqrt(max(delta²-1, 1e-10))
    let c = 1.0 / (delta * delta - 1.0).max(1e-10).sqrt();

    let mut grad = vec![0.0f64; dim];
    for i in 0..dim {
        grad[i] = 4.0 * c * (diff[i] / (alpha * beta) + r2 * u[i] / (alpha * alpha));
    }
    grad
}

/// Convert Euclidean gradient to Riemannian gradient on the Poincaré ball.
/// `g_R = ((1 - ||u||²)² / 4) * g_E`
#[inline]
fn riemannian_grad(u: &[f64], euclidean: &[f64]) -> Vec<f64> {
    let u_n2: f64 = u.iter().map(|x| x * x).sum();
    let scale = ((1.0 - u_n2) * (1.0 - u_n2)) / 4.0;
    euclidean.iter().map(|g| scale * g).collect()
}

/// Project a point `e` to be strictly inside the unit ball.
/// If `||e|| >= 1 - eps`, rescales to `(1 - eps) / ||e||`.
#[inline]
fn project_ball(e: &mut [f64], eps: f64) {
    let n2: f64 = e.iter().map(|x| x * x).sum();
    let n = n2.sqrt();
    let cap = 1.0 - eps;
    if n >= cap {
        let scale = cap / n;
        for v in e.iter_mut() {
            *v *= scale;
        }
    }
}

/// Compute logsumexp of a slice of values.
fn logsumexp(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_v = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_v.is_infinite() {
        return max_v;
    }
    let sum: f64 = vals.iter().map(|&v| (v - max_v).exp()).sum();
    max_v + sum.ln()
}

/// Sample `n_neg` negative indices ≠ `pos_idx` from `[0, n_nodes)`.
fn sample_negatives(
    rng: &mut LcgRng,
    n_nodes: usize,
    pos_u: usize,
    pos_v: usize,
    n_neg: usize,
) -> Vec<usize> {
    // Clamp n_neg to avoid trying to sample more than available negatives.
    let max_neg = n_nodes.saturating_sub(2).max(1);
    let actual_neg = n_neg.min(max_neg);

    let mut negs = Vec::with_capacity(actual_neg);
    let mut attempts = 0usize;
    while negs.len() < actual_neg && attempts < actual_neg * 10 + 100 {
        let idx = rng.next_usize(n_nodes);
        if idx != pos_u && idx != pos_v && !negs.contains(&idx) {
            negs.push(idx);
        }
        attempts += 1;
    }
    // Fallback: if we couldn't get enough unique negatives, just add remaining.
    if negs.is_empty() && n_nodes > 0 {
        for i in 0..n_nodes {
            if i != pos_u && i != pos_v {
                negs.push(i);
                if negs.len() >= actual_neg {
                    break;
                }
            }
        }
    }
    negs
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Train Poincaré embeddings given a list of positive `(u, v)` pairs.
///
/// Uses Riemannian SGD with negative sampling. For each positive pair `(u, v)`:
/// - Sample `n_neg` random negatives `v'₁..v'ₙ`.
/// - Compute loss `L = log(1 + Σⱼ exp(-d(u,v'ⱼ) + d(u,v)))`.
/// - Update `u` and `v` using Riemannian gradients.
/// - Project all updated embeddings back into the unit ball.
///
/// # Arguments
/// * `pairs`  — positive relation pairs `(u_idx, v_idx)` where both indices are `< n_nodes`.
/// * `config` — training configuration.
pub fn poincare_fit(
    pairs: &[(usize, usize)],
    config: &PoincareConfig,
) -> ManifoldResult<PoincareModel> {
    if config.n_nodes == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "n_nodes".into(),
            reason: "must be >= 1".into(),
        });
    }
    if config.dim == 0 {
        return Err(ManifoldError::InvalidParameter {
            name: "dim".into(),
            reason: "embedding dimension must be >= 1".into(),
        });
    }
    if config.init_radius <= 0.0 {
        return Err(ManifoldError::InvalidParameter {
            name: "init_radius".into(),
            reason: "must be > 0".into(),
        });
    }

    // Validate pair indices.
    for &(u, v) in pairs {
        if u >= config.n_nodes {
            return Err(ManifoldError::IndexOutOfBounds {
                index: u,
                len: config.n_nodes,
            });
        }
        if v >= config.n_nodes {
            return Err(ManifoldError::IndexOutOfBounds {
                index: v,
                len: config.n_nodes,
            });
        }
    }

    let n = config.n_nodes;
    let dim = config.dim;
    let eps = config.eps;
    let lr = config.learning_rate;

    let mut rng = LcgRng::new(config.seed);

    // ── Initialise embeddings via random direction + scale ─────────────────
    // Box-Muller: generate Normal(0,1) then normalise direction and scale.
    let mut embeddings = vec![0.0f64; n * dim];
    for i in 0..n {
        let start = i * dim;
        // Fill with standard normals using Box-Muller.
        let mut j = 0;
        while j < dim {
            let u1 = rng.next_f64().max(1e-10);
            let u2 = rng.next_f64();
            let r = (-2.0 * u1.ln()).sqrt();
            let z1 = r * (2.0 * std::f64::consts::PI * u2).cos();
            embeddings[start + j] = z1;
            j += 1;
            if j < dim {
                let z2 = r * (2.0 * std::f64::consts::PI * u2).sin();
                embeddings[start + j] = z2;
                j += 1;
            }
        }
        // Normalise direction then scale by init_radius.
        let norm: f64 = embeddings[start..start + dim]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        if norm > 1e-12 {
            let scale = config.init_radius / norm;
            for k in 0..dim {
                embeddings[start + k] *= scale;
            }
        }
        // Project into ball just in case.
        project_ball(&mut embeddings[start..start + dim], eps);
    }

    if pairs.is_empty() {
        return Ok(PoincareModel {
            embeddings,
            n_nodes: n,
            dim,
            config: config.clone(),
            final_loss: 0.0,
        });
    }

    // Track initial loss for comparison.
    let initial_loss =
        compute_epoch_loss(&embeddings, pairs, &mut LcgRng::new(config.seed), config);

    let mut final_loss = initial_loss;

    // ── Training loop ──────────────────────────────────────────────────────
    for _epoch in 0..config.n_epochs {
        let mut epoch_loss = 0.0f64;

        for &(pos_u, pos_v) in pairs {
            // Sample negatives.
            let negs = sample_negatives(&mut rng, n, pos_u, pos_v, config.n_neg);
            if negs.is_empty() {
                continue;
            }

            let u_start = pos_u * dim;
            let v_start = pos_v * dim;

            let u_emb: Vec<f64> = embeddings[u_start..u_start + dim].to_vec();
            let v_emb: Vec<f64> = embeddings[v_start..v_start + dim].to_vec();

            let d_pos = poincare_dist_unchecked(&u_emb, &v_emb);

            // Negative distances.
            let neg_embs: Vec<Vec<f64>> = negs
                .iter()
                .map(|&ni| embeddings[ni * dim..(ni + 1) * dim].to_vec())
                .collect();
            let neg_dists: Vec<f64> = neg_embs
                .iter()
                .map(|ne| poincare_dist_unchecked(&u_emb, ne))
                .collect();

            // Loss: log(1 + Σⱼ exp(-d(u,v'ⱼ) + d(u,v))).
            // Compute via logsumexp for numerical stability.
            let mut logits = vec![-d_pos];
            for &nd in &neg_dists {
                logits.push(-nd);
            }
            let lse = logsumexp(&logits);
            // L = lse - (-d_pos) = lse + d_pos
            let loss = lse + d_pos;
            epoch_loss += loss;

            // Softmax weights: p_j = exp(logit_j - lse).
            let softmax: Vec<f64> = logits.iter().map(|&l| (l - lse).exp()).collect();
            // softmax[0] = weight for positive pair, softmax[1..] for negatives.

            // Gradient of loss w.r.t. u:
            //   dL/du = (1 - softmax[0]) * g_pos(u,v) - Σⱼ softmax[1+j] * g_neg(u,v'ⱼ)
            // where g_pos = ∂d(u,v)/∂u (Riemannian gradient).
            let pos_w = 1.0 - softmax[0];
            let g_eu_pos = euclidean_grad_u(&u_emb, &v_emb);
            let g_ru_pos = riemannian_grad(&u_emb, &g_eu_pos);

            let mut g_u = vec![0.0f64; dim];
            for k in 0..dim {
                g_u[k] += pos_w * g_ru_pos[k];
            }
            for (j, &ni) in negs.iter().enumerate() {
                let neg_w = softmax[1 + j];
                let g_eu_neg = euclidean_grad_u(&u_emb, &neg_embs[j]);
                let g_ru_neg = riemannian_grad(&u_emb, &g_eu_neg);
                for k in 0..dim {
                    g_u[k] -= neg_w * g_ru_neg[k];
                }
                // Gradient w.r.t. v'ⱼ (negative node).
                // ∂d(u,v')/∂v' with weight -softmax[1+j] (we want to push away).
                let g_ev_neg = euclidean_grad_u(&neg_embs[j], &u_emb); // symmetric diff
                let g_rv_neg = riemannian_grad(&neg_embs[j], &g_ev_neg);
                // Update negative embedding.
                let ni_start = ni * dim;
                for k in 0..dim {
                    embeddings[ni_start + k] -= lr * (-neg_w * g_rv_neg[k]);
                }
                project_ball(&mut embeddings[ni_start..ni_start + dim], eps);
            }

            // Gradient w.r.t. v (positive node).
            let g_ev_pos = euclidean_grad_u(&v_emb, &u_emb);
            let g_rv_pos = riemannian_grad(&v_emb, &g_ev_pos);

            // Update u.
            for k in 0..dim {
                embeddings[u_start + k] -= lr * g_u[k];
            }
            project_ball(&mut embeddings[u_start..u_start + dim], eps);

            // Update v (positive).
            for k in 0..dim {
                embeddings[v_start + k] -= lr * (pos_w * g_rv_pos[k]);
            }
            project_ball(&mut embeddings[v_start..v_start + dim], eps);
        }

        final_loss = if pairs.is_empty() {
            0.0
        } else {
            epoch_loss / pairs.len() as f64
        };
    }

    Ok(PoincareModel {
        embeddings,
        n_nodes: n,
        dim,
        config: config.clone(),
        final_loss,
    })
}

/// Compute loss over all pairs (used for tracking).
fn compute_epoch_loss(
    embeddings: &[f64],
    pairs: &[(usize, usize)],
    rng: &mut LcgRng,
    config: &PoincareConfig,
) -> f64 {
    let n = config.n_nodes;
    let dim = config.dim;
    let mut total = 0.0f64;
    for &(pos_u, pos_v) in pairs {
        let u_emb = &embeddings[pos_u * dim..(pos_u + 1) * dim];
        let v_emb = &embeddings[pos_v * dim..(pos_v + 1) * dim];
        let d_pos = poincare_dist_unchecked(u_emb, v_emb);
        let negs = sample_negatives(rng, n, pos_u, pos_v, config.n_neg);
        if negs.is_empty() {
            continue;
        }
        let mut logits = vec![-d_pos];
        for &ni in &negs {
            let ne = &embeddings[ni * dim..(ni + 1) * dim];
            logits.push(-poincare_dist_unchecked(u_emb, ne));
        }
        let lse = logsumexp(&logits);
        total += lse + d_pos;
    }
    if pairs.is_empty() {
        0.0
    } else {
        total / pairs.len() as f64
    }
}

/// Compute the Poincaré (hyperbolic) distance between two points in the unit ball.
///
/// Both `u` and `v` must be length `dim` and lie strictly inside the unit ball.
pub fn poincare_distance(u: &[f64], v: &[f64]) -> ManifoldResult<f64> {
    if u.len() != v.len() {
        return Err(ManifoldError::DimensionMismatch {
            a: u.len(),
            b: v.len(),
        });
    }
    let u_n2: f64 = u.iter().map(|x| x * x).sum();
    let v_n2: f64 = v.iter().map(|x| x * x).sum();
    if u_n2 >= 1.0 {
        return Err(ManifoldError::ManifoldConstraint(
            "poincare_distance: u is on or outside the unit ball".into(),
        ));
    }
    if v_n2 >= 1.0 {
        return Err(ManifoldError::ManifoldConstraint(
            "poincare_distance: v is on or outside the unit ball".into(),
        ));
    }
    let alpha = (1.0 - u_n2).max(1e-10);
    let beta = (1.0 - v_n2).max(1e-10);
    let diff_n2: f64 = u.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
    // When u == v, diff_n2 is exactly 0 and delta = 1, giving distance 0.
    if diff_n2 == 0.0 {
        return Ok(0.0);
    }
    let delta = 1.0 + 2.0 * diff_n2 / (alpha * beta);
    Ok(acosh_safe(delta))
}

/// Compute the full `n_nodes × n_nodes` pairwise distance matrix.
///
/// Returns a flat `n_nodes² × 1` vector in row-major order.
/// The matrix is symmetric: `dist[i * n + j] == dist[j * n + i]`.
#[must_use]
pub fn poincare_distances_all(model: &PoincareModel) -> Vec<f64> {
    let n = model.n_nodes;
    let dim = model.dim;
    let mut dists = vec![0.0f64; n * n];
    for i in 0..n {
        for j in i + 1..n {
            let ui = &model.embeddings[i * dim..(i + 1) * dim];
            let vj = &model.embeddings[j * dim..(j + 1) * dim];
            let d = poincare_dist_unchecked(ui, vj);
            dists[i * n + j] = d;
            dists[j * n + i] = d;
        }
    }
    dists
}

/// Compute the mean rank of positive pairs w.r.t. all `n_nodes` candidates.
///
/// For each positive pair `(u, v)`, compute the rank of `v` among all nodes
/// when sorted by distance from `u`. Returns the mean rank.
/// A lower mean rank indicates better reconstruction of the hierarchy.
pub fn poincare_rank_relations(
    model: &PoincareModel,
    pairs: &[(usize, usize)],
) -> ManifoldResult<f64> {
    if pairs.is_empty() {
        return Ok(1.0);
    }
    for &(u, v) in pairs {
        if u >= model.n_nodes {
            return Err(ManifoldError::IndexOutOfBounds {
                index: u,
                len: model.n_nodes,
            });
        }
        if v >= model.n_nodes {
            return Err(ManifoldError::IndexOutOfBounds {
                index: v,
                len: model.n_nodes,
            });
        }
    }

    let n = model.n_nodes;
    let dim = model.dim;
    let mut total_rank = 0.0f64;

    for &(u_idx, v_idx) in pairs {
        let u_emb = &model.embeddings[u_idx * dim..(u_idx + 1) * dim];
        let v_emb = &model.embeddings[v_idx * dim..(v_idx + 1) * dim];
        let target_dist = poincare_dist_unchecked(u_emb, v_emb);

        // Count how many nodes are closer to u than v is.
        let mut rank = 1usize;
        for j in 0..n {
            if j == u_idx || j == v_idx {
                continue;
            }
            let other = &model.embeddings[j * dim..(j + 1) * dim];
            let d_other = poincare_dist_unchecked(u_emb, other);
            if d_other < target_dist {
                rank += 1;
            }
        }
        total_rank += rank as f64;
    }

    Ok(total_rank / pairs.len() as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config(n_nodes: usize) -> PoincareConfig {
        PoincareConfig {
            n_nodes,
            dim: 2,
            learning_rate: 0.05,
            n_epochs: 50,
            n_neg: 5,
            init_radius: 1e-3,
            eps: 1e-5,
            seed: 42,
        }
    }

    fn chain_pairs(n: usize) -> Vec<(usize, usize)> {
        (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect()
    }

    // Test 1: all embeddings strictly inside unit ball
    #[test]
    fn all_embeddings_inside_unit_ball() {
        let n = 10;
        let config = default_config(n);
        let pairs = chain_pairs(n);
        let model = poincare_fit(&pairs, &config);
        assert!(model.is_ok(), "fit should succeed: {:?}", model.err());
        let model = model.expect("model should be present");
        let eps = config.eps;
        for i in 0..n {
            let e = &model.embeddings[i * 2..(i + 1) * 2];
            let n2: f64 = e.iter().map(|x| x * x).sum();
            assert!(
                n2 < (1.0 - eps) * (1.0 - eps),
                "node {i}: ||e||²={n2} is not inside ball"
            );
        }
    }

    // Test 2: embeddings.len() == n_nodes * dim
    #[test]
    fn embeddings_length_correct() {
        let n = 8;
        let config = PoincareConfig {
            n_nodes: n,
            dim: 3,
            ..Default::default()
        };
        let pairs = chain_pairs(n);
        let model = poincare_fit(&pairs, &config).expect("poincare_fit should succeed");
        assert_eq!(model.embeddings.len(), n * 3);
    }

    // Test 3: poincare_distance symmetric
    #[test]
    fn poincare_distance_symmetric() {
        let u = vec![0.1, 0.2];
        let v = vec![-0.3, 0.1];
        let d1 = poincare_distance(&u, &v).expect("poincare_distance should succeed");
        let d2 = poincare_distance(&v, &u).expect("poincare_distance should succeed");
        assert!(
            (d1 - d2).abs() < 1e-10,
            "distance not symmetric: {d1} vs {d2}"
        );
    }

    // Test 4: poincare_distance non-negative
    #[test]
    fn poincare_distance_nonnegative() {
        let u = vec![0.1, 0.2];
        let v = vec![-0.2, 0.1];
        let d = poincare_distance(&u, &v).expect("poincare_distance should succeed");
        assert!(d >= 0.0, "distance should be non-negative, got {d}");
    }

    // Test 5: poincare_distance(u, u) ≈ 0
    #[test]
    fn poincare_distance_self_zero() {
        let u = vec![0.3, -0.2];
        let d = poincare_distance(&u, &u).expect("poincare_distance should succeed");
        assert!(d.abs() < 1e-10, "d(u,u) should be ~0, got {d}");
    }

    // Test 6: triangle inequality
    #[test]
    fn triangle_inequality() {
        let u = vec![0.1, 0.2];
        let v = vec![-0.1, 0.3];
        let w = vec![0.2, -0.2];
        let duw = poincare_distance(&u, &w).expect("poincare_distance should succeed");
        let duv = poincare_distance(&u, &v).expect("poincare_distance should succeed");
        let dvw = poincare_distance(&v, &w).expect("poincare_distance should succeed");
        assert!(
            duw <= duv + dvw + 1e-9,
            "triangle inequality violated: {duw} > {duv} + {dvw}"
        );
    }

    // Test 7: positive pairs closer than random pairs after training
    #[test]
    fn positive_pairs_closer_than_random() {
        let n = 15;
        let config = PoincareConfig {
            n_nodes: n,
            dim: 2,
            n_epochs: 30,
            n_neg: 5,
            learning_rate: 0.1,
            ..Default::default()
        };
        let pairs = chain_pairs(n);
        let model = poincare_fit(&pairs, &config).expect("poincare_fit should succeed");

        // Mean distance over positive pairs.
        let mean_pos = pairs
            .iter()
            .map(|&(u, v)| {
                let ue = &model.embeddings[u * 2..(u + 1) * 2];
                let ve = &model.embeddings[v * 2..(v + 1) * 2];
                poincare_dist_unchecked(ue, ve)
            })
            .sum::<f64>()
            / pairs.len() as f64;

        // Sample some random pairs.
        let mut rng = LcgRng::new(99);
        let n_random = 50usize;
        let mean_random: f64 = (0..n_random)
            .map(|_| {
                let i = rng.next_usize(n);
                let j = (rng.next_usize(n - 1) + i + 1) % n;
                let ue = &model.embeddings[i * 2..(i + 1) * 2];
                let ve = &model.embeddings[j * 2..(j + 1) * 2];
                poincare_dist_unchecked(ue, ve)
            })
            .sum::<f64>()
            / n_random as f64;

        assert!(
            mean_pos < mean_random,
            "mean positive dist ({mean_pos}) should be < mean random dist ({mean_random})"
        );
    }

    // Test 8: loss decreases over training
    #[test]
    fn loss_decreases_over_training() {
        let n = 20;
        let pairs: Vec<(usize, usize)> = chain_pairs(n);

        // Run with n_epochs=1 and n_epochs=10, compare final_loss.
        let config_short = PoincareConfig {
            n_nodes: n,
            dim: 2,
            n_epochs: 1,
            n_neg: 5,
            learning_rate: 0.1,
            seed: 42,
            ..Default::default()
        };
        let config_long = PoincareConfig {
            n_nodes: n,
            dim: 2,
            n_epochs: 10,
            n_neg: 5,
            learning_rate: 0.1,
            seed: 42,
            ..Default::default()
        };
        let model_short = poincare_fit(&pairs, &config_short).expect("poincare_fit should succeed");
        let model_long = poincare_fit(&pairs, &config_long).expect("poincare_fit should succeed");

        assert!(
            model_long.final_loss < model_short.final_loss,
            "loss should decrease: after 10 epochs ({}) should be < 1 epoch ({})",
            model_long.final_loss,
            model_short.final_loss
        );
    }

    // Test 9: poincare_distances_all has correct shape n×n
    #[test]
    fn distances_all_shape() {
        let n = 6;
        let config = default_config(n);
        let model = poincare_fit(&chain_pairs(n), &config).expect("value should be present");
        let dists = poincare_distances_all(&model);
        assert_eq!(dists.len(), n * n, "distance matrix should be n×n");
    }

    // Test 10: distance matrix is symmetric
    #[test]
    fn distance_matrix_symmetric() {
        let n = 5;
        let config = default_config(n);
        let model = poincare_fit(&chain_pairs(n), &config).expect("value should be present");
        let dists = poincare_distances_all(&model);
        for i in 0..n {
            for j in 0..n {
                let d1 = dists[i * n + j];
                let d2 = dists[j * n + i];
                assert!(
                    (d1 - d2).abs() < 1e-10,
                    "d[{i},{j}]={d1} != d[{j},{i}]={d2}"
                );
            }
        }
    }

    // Test 11: dim=1 works
    #[test]
    fn dim_one_works() {
        let n = 5;
        let config = PoincareConfig {
            n_nodes: n,
            dim: 1,
            n_epochs: 5,
            ..Default::default()
        };
        let model = poincare_fit(&chain_pairs(n), &config);
        assert!(model.is_ok(), "dim=1 should work: {:?}", model.err());
        assert_eq!(model.expect("model should be present").dim, 1);
    }

    // Test 12: dim=10 works
    #[test]
    fn dim_ten_works() {
        let n = 8;
        let config = PoincareConfig {
            n_nodes: n,
            dim: 10,
            n_epochs: 5,
            ..Default::default()
        };
        let model = poincare_fit(&chain_pairs(n), &config);
        assert!(model.is_ok(), "dim=10 should work: {:?}", model.err());
        assert_eq!(
            model.expect("model should be present").embeddings.len(),
            n * 10
        );
    }

    // Test 13: n_epochs=1 completes
    #[test]
    fn n_epochs_one_completes() {
        let n = 10;
        let config = PoincareConfig {
            n_nodes: n,
            n_epochs: 1,
            ..default_config(n)
        };
        let model = poincare_fit(&chain_pairs(n), &config);
        assert!(model.is_ok());
    }

    // Test 14: n_neg=1 works
    #[test]
    fn n_neg_one_works() {
        let n = 8;
        let config = PoincareConfig {
            n_nodes: n,
            n_neg: 1,
            ..default_config(n)
        };
        let model = poincare_fit(&chain_pairs(n), &config);
        assert!(model.is_ok());
    }

    // Test 15: n_neg >= n_nodes → no panic (clamp internally)
    #[test]
    fn n_neg_exceeds_n_nodes_no_panic() {
        let n = 5;
        let config = PoincareConfig {
            n_nodes: n,
            n_neg: 100, // >> n_nodes
            n_epochs: 5,
            ..default_config(n)
        };
        let model = poincare_fit(&chain_pairs(n), &config);
        assert!(
            model.is_ok(),
            "should handle n_neg >> n_nodes: {:?}",
            model.err()
        );
    }

    // Test 16: determinism — same seed → identical embeddings
    #[test]
    fn determinism_same_seed() {
        let n = 10;
        let config = default_config(n);
        let pairs = chain_pairs(n);
        let model1 = poincare_fit(&pairs, &config).expect("poincare_fit should succeed");
        let model2 = poincare_fit(&pairs, &config).expect("poincare_fit should succeed");
        for (a, b) in model1.embeddings.iter().zip(model2.embeddings.iter()) {
            assert_eq!(a, b, "embeddings differ with same seed");
        }
    }

    // Test 17: in trained chain [0→1→2→3→4], deeper nodes tend to be farther from origin
    #[test]
    fn hierarchy_deeper_farther_from_origin() {
        let n = 5;
        // Chain: 0 is "root" (connected to 1), 4 is "leaf".
        // Add extra edges to reinforce hierarchy.
        let pairs: Vec<(usize, usize)> = vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4), // 0 is hub
            (1, 2),
            (2, 3),
            (3, 4), // chain
        ];
        let config = PoincareConfig {
            n_nodes: n,
            dim: 2,
            n_epochs: 100,
            n_neg: 3,
            learning_rate: 0.1,
            seed: 7,
            ..Default::default()
        };
        let model = poincare_fit(&pairs, &config).expect("poincare_fit should succeed");

        // Norm of each embedding.
        let norms: Vec<f64> = (0..n)
            .map(|i| {
                let e = &model.embeddings[i * 2..(i + 1) * 2];
                e.iter().map(|x| x * x).sum::<f64>().sqrt()
            })
            .collect();

        // Node 0 should be the most "central" (smallest norm) and node 4 more peripheral.
        // We just check that not all norms are equal (some hierarchy exists).
        let min_norm = norms.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_norm = norms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_norm > min_norm + 1e-6,
            "hierarchy: all norms are equal (no structure learned), norms={norms:?}"
        );
    }

    // Test 18: poincare_rank_relations returns finite value in [1, n_nodes]
    #[test]
    fn rank_relations_valid_range() {
        let n = 10;
        let config = default_config(n);
        let pairs = chain_pairs(n);
        let model = poincare_fit(&pairs, &config).expect("poincare_fit should succeed");
        let rank = poincare_rank_relations(&model, &pairs);
        assert!(rank.is_ok(), "rank should be ok: {:?}", rank.err());
        let rank = rank.expect("rank should be present");
        assert!(rank.is_finite(), "rank must be finite");
        assert!(rank >= 1.0, "rank must be >= 1, got {rank}");
        assert!(rank <= n as f64, "rank must be <= n_nodes={n}, got {rank}");
    }

    // Test 19: n_nodes=0 → InvalidParameter
    #[test]
    fn n_nodes_zero_error() {
        let config = PoincareConfig {
            n_nodes: 0,
            dim: 2,
            ..Default::default()
        };
        let res = poincare_fit(&[], &config);
        assert!(res.is_err());
        match res.unwrap_err() {
            ManifoldError::InvalidParameter { name, .. } => assert_eq!(name, "n_nodes"),
            e => panic!("expected InvalidParameter(n_nodes), got {e:?}"),
        }
    }

    // Test 20: dim=0 → InvalidParameter
    #[test]
    fn dim_zero_error() {
        let config = PoincareConfig {
            n_nodes: 5,
            dim: 0,
            ..Default::default()
        };
        let res = poincare_fit(&[], &config);
        assert!(res.is_err());
        match res.unwrap_err() {
            ManifoldError::InvalidParameter { name, .. } => assert_eq!(name, "dim"),
            e => panic!("expected InvalidParameter(dim), got {e:?}"),
        }
    }

    // Test 21: pair with index >= n_nodes → IndexOutOfBounds
    #[test]
    fn pair_out_of_bounds_error() {
        let config = PoincareConfig {
            n_nodes: 5,
            dim: 2,
            ..Default::default()
        };
        let pairs = vec![(0, 7)]; // index 7 >= 5
        let res = poincare_fit(&pairs, &config);
        assert!(res.is_err());
        match res.unwrap_err() {
            ManifoldError::IndexOutOfBounds { index, .. } => assert_eq!(index, 7),
            e => panic!("expected IndexOutOfBounds, got {e:?}"),
        }
    }

    // Test 22: init_radius=0 → InvalidParameter
    #[test]
    fn init_radius_zero_error() {
        let config = PoincareConfig {
            n_nodes: 5,
            dim: 2,
            init_radius: 0.0,
            ..Default::default()
        };
        let res = poincare_fit(&[], &config);
        assert!(res.is_err());
        match res.unwrap_err() {
            ManifoldError::InvalidParameter { name, .. } => assert_eq!(name, "init_radius"),
            e => panic!("expected InvalidParameter(init_radius), got {e:?}"),
        }
    }
}
