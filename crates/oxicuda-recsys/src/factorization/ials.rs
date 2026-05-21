use std::collections::HashSet;

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for iALS (implicit ALS with Conjugate Gradient solver).
///
/// Reference: Hu, Koren & Volinsky (2008 ICDM) "Collaborative Filtering for
/// Implicit Feedback Datasets" + CG solver variant by Pilászy et al. (2010).
#[derive(Debug, Clone)]
pub struct IalsConfig {
    pub n_users: usize,
    pub n_items: usize,
    /// Embedding dimensionality.
    pub dim: usize,
    /// L2 regularisation strength (λ). Must be > 0.
    pub lambda: f32,
    /// Confidence scaling for observed interactions: c_ui = 1 + alpha. (default 40.0)
    pub alpha: f32,
    /// Number of full ALS epochs. (default 10)
    pub n_epochs: usize,
    /// Maximum CG iterations per entity. (default 3)
    pub n_cg_steps: usize,
    /// CG residual norm tolerance for early stopping. (default 1e-4)
    pub cg_tol: f32,
}

impl Default for IalsConfig {
    fn default() -> Self {
        Self {
            n_users: 0,
            n_items: 0,
            dim: 0,
            lambda: 0.01,
            alpha: 40.0,
            n_epochs: 10,
            n_cg_steps: 3,
            cg_tol: 1e-4,
        }
    }
}

// ── Model ─────────────────────────────────────────────────────────────────────

/// iALS with a sparse Conjugate Gradient solver.
///
/// Learns user / item embeddings from implicit feedback (binary observations)
/// with confidence weighting c_ui = 1 + α · p_ui, where p_ui ∈ {0, 1}.
///
/// Each entity embedding is updated by solving the PSD linear system:
///   (Y^T C Y + λ I) x = Y^T C p
/// iteratively via CG without forming the d × d dense matrix explicitly.
pub struct Ials {
    pub cfg: IalsConfig,
    /// Row-major [n_users × dim] user embeddings.
    pub user_emb: Vec<f32>,
    /// Row-major [n_items × dim] item embeddings.
    pub item_emb: Vec<f32>,
}

impl Ials {
    // ── Constructor ───────────────────────────────────────────────────────

    /// Create a new iALS model with N(0, 1/√dim) initialised embeddings.
    pub fn new(cfg: IalsConfig, rng: &mut LcgRng) -> RecsysResult<Self> {
        if cfg.n_users == 0 {
            return Err(RecsysError::InvalidNumUsers { n: cfg.n_users });
        }
        if cfg.n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: cfg.n_items });
        }
        if cfg.dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: cfg.dim });
        }
        if cfg.lambda <= 0.0 {
            return Err(RecsysError::InvalidLambda { val: cfg.lambda });
        }
        if cfg.alpha <= 0.0 {
            return Err(RecsysError::InvalidConfig {
                msg: format!("alpha must be > 0, got {}", cfg.alpha),
            });
        }

        let scale = (1.0_f32 / cfg.dim as f32).sqrt();
        let total_user = cfg.n_users * cfg.dim;
        let total_item = cfg.n_items * cfg.dim;
        let mut user_emb = vec![0.0_f32; total_user];
        let mut item_emb = vec![0.0_f32; total_item];
        for v in &mut user_emb {
            *v = rng.next_normal() * scale;
        }
        for v in &mut item_emb {
            *v = rng.next_normal() * scale;
        }

        Ok(Self {
            cfg,
            user_emb,
            item_emb,
        })
    }

    // ── Training ──────────────────────────────────────────────────────────

    /// Fit the model on binary implicit feedback.
    ///
    /// `interactions`: list of `(user_id, item_id)` observation pairs.
    /// Duplicates are deduplicated per user–item pair.
    pub fn fit(&mut self, interactions: &[(usize, usize)], _rng: &mut LcgRng) -> RecsysResult<()> {
        if interactions.is_empty() {
            return Err(RecsysError::EmptyInteraction);
        }

        let n_users = self.cfg.n_users;
        let n_items = self.cfg.n_items;
        let dim = self.cfg.dim;
        let lambda = self.cfg.lambda;
        let alpha = self.cfg.alpha;
        let n_epochs = self.cfg.n_epochs;
        let n_cg_steps = self.cfg.n_cg_steps;
        let cg_tol = self.cfg.cg_tol;

        // Validate all ids.
        for &(u, i) in interactions {
            if u >= n_users {
                return Err(RecsysError::UnknownUser { id: u });
            }
            if i >= n_items {
                return Err(RecsysError::UnknownItem { id: i });
            }
        }

        // Build per-user and per-item adjacency lists (deduplicated).
        let mut user_items: Vec<HashSet<usize>> = vec![HashSet::new(); n_users];
        let mut item_users: Vec<HashSet<usize>> = vec![HashSet::new(); n_items];
        for &(u, i) in interactions {
            user_items[u].insert(i);
            item_users[i].insert(u);
        }

        // Convert to sorted Vecs for deterministic iteration order.
        let user_items: Vec<Vec<usize>> = user_items
            .into_iter()
            .map(|s| {
                let mut v: Vec<usize> = s.into_iter().collect();
                v.sort_unstable();
                v
            })
            .collect();
        let item_users: Vec<Vec<usize>> = item_users
            .into_iter()
            .map(|s| {
                let mut v: Vec<usize> = s.into_iter().collect();
                v.sort_unstable();
                v
            })
            .collect();

        for _epoch in 0..n_epochs {
            // ── User update ───────────────────────────────────────────────
            // For each user u, solve: (Y^T C^u Y + λI) x_u = Y^T C^u p^u
            // where Y = item_emb, C^u_ii = 1+alpha if i∈items_u else 1,
            // p^u_i = 1 if i∈items_u else 0.
            //
            // Sparse CG: only the observed items contribute non-trivially to
            // the (C-I) term: A·v = Y^T Y·v + λv + Σ_{i∈items_u} alpha*(y_i·v)*y_i
            // Equivalently: A·v = Σ_all y_i*(y_i·v) + λv + Σ_{i∈obs} alpha*(y_i·v)*y_i
            //
            // We use the compact form: A·v = λv + Σ_{i∈obs} (1+alpha)*(y_i·v)*y_i
            //                                   + Σ_{i∉obs} 1*(y_i·v)*y_i
            // But forming all-items sum is O(n_items*dim). Instead we use the
            // gram-matrix trick:  Y^T Y·v (precomputed gram = dim×dim) + sparse (C-I) term.

            // Pre-compute gram matrix G = Y^T Y (dim × dim).
            let gram = compute_gram(&self.item_emb, n_items, dim);

            for (u, obs) in user_items.iter().enumerate() {
                // Gather observed item confidences and preferences.
                // confidence for observed items: 1 + alpha; preference: 1.
                let n_obs = obs.len();
                let mut obs_items: Vec<usize> = obs.to_vec();
                obs_items.sort_unstable();

                // All observed items have confidence 1+alpha and preference 1.
                let confidences = vec![1.0_f32 + alpha; n_obs];

                // Build compact Y_obs matrix (n_obs × dim).
                let mut y_obs = vec![0.0_f32; n_obs * dim];
                for (k, &item_id) in obs_items.iter().enumerate() {
                    let src = &self.item_emb[item_id * dim..(item_id + 1) * dim];
                    y_obs[k * dim..(k + 1) * dim].copy_from_slice(src);
                }

                // Initial embedding for this user.
                let x_init: Vec<f32> = self.user_emb[u * dim..(u + 1) * dim].to_vec();

                // Compute rhs = Y_obs^T * (conf .* pref).
                // Since pref = 1 for all obs rows, rhs[d] = Σ_{k∈obs} conf_k * y_obs[k,d].
                let mut rhs = vec![0.0_f32; dim];
                for (k, &c) in confidences.iter().enumerate() {
                    let y = &y_obs[k * dim..(k + 1) * dim];
                    for (d_idx, &yi) in y.iter().enumerate() {
                        rhs[d_idx] += c * yi;
                    }
                }

                // Use factored CG with gram + sparse correction.
                let new_x = Self::cg_solve_gram(
                    &gram,
                    &y_obs,
                    n_obs,
                    dim,
                    &confidences,
                    &rhs,
                    &x_init,
                    lambda,
                    n_cg_steps,
                    cg_tol,
                )?;

                // Write back.
                self.user_emb[u * dim..(u + 1) * dim].copy_from_slice(&new_x);
            }

            // ── Item update ───────────────────────────────────────────────
            // Symmetric: for each item i, use user_emb as the "Y" matrix.
            let gram_u = compute_gram(&self.user_emb, n_users, dim);

            for (i, obs) in item_users.iter().enumerate() {
                let n_obs = obs.len();
                let mut obs_users: Vec<usize> = obs.to_vec();
                obs_users.sort_unstable();

                let confidences = vec![1.0_f32 + alpha; n_obs];

                let mut y_obs = vec![0.0_f32; n_obs * dim];
                for (k, &user_id) in obs_users.iter().enumerate() {
                    let src = &self.user_emb[user_id * dim..(user_id + 1) * dim];
                    y_obs[k * dim..(k + 1) * dim].copy_from_slice(src);
                }

                let x_init: Vec<f32> = self.item_emb[i * dim..(i + 1) * dim].to_vec();

                // Since pref=1 for all obs rows: rhs[d] = Σ_{k∈obs} conf_k * y_obs[k,d].
                let mut rhs = vec![0.0_f32; dim];
                for (k, &c) in confidences.iter().enumerate() {
                    let y = &y_obs[k * dim..(k + 1) * dim];
                    for (d_idx, &yi) in y.iter().enumerate() {
                        rhs[d_idx] += c * yi;
                    }
                }

                let new_x = Self::cg_solve_gram(
                    &gram_u,
                    &y_obs,
                    n_obs,
                    dim,
                    &confidences,
                    &rhs,
                    &x_init,
                    lambda,
                    n_cg_steps,
                    cg_tol,
                )?;

                self.item_emb[i * dim..(i + 1) * dim].copy_from_slice(&new_x);
            }
        }

        Ok(())
    }

    // ── Inference ─────────────────────────────────────────────────────────

    /// Predict score (dot product of embeddings) for a (user, item) pair.
    pub fn predict(&self, user: usize, item: usize) -> RecsysResult<f32> {
        if user >= self.cfg.n_users {
            return Err(RecsysError::UnknownUser { id: user });
        }
        if item >= self.cfg.n_items {
            return Err(RecsysError::UnknownItem { id: item });
        }
        let dim = self.cfg.dim;
        let dot: f32 = self.user_emb[user * dim..(user + 1) * dim]
            .iter()
            .zip(self.item_emb[item * dim..(item + 1) * dim].iter())
            .map(|(&u, &i)| u * i)
            .sum();
        Ok(dot)
    }

    /// Rank all items for a user by descending predicted score.
    ///
    /// Returns item indices sorted from highest score to lowest.
    pub fn rank_items(&self, user: usize) -> RecsysResult<Vec<usize>> {
        if user >= self.cfg.n_users {
            return Err(RecsysError::UnknownUser { id: user });
        }
        let n_items = self.cfg.n_items;
        let dim = self.cfg.dim;
        let u_emb = &self.user_emb[user * dim..(user + 1) * dim];

        let mut scores: Vec<(usize, f32)> = (0..n_items)
            .map(|item| {
                let dot: f32 = u_emb
                    .iter()
                    .zip(self.item_emb[item * dim..(item + 1) * dim].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                (item, dot)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scores.into_iter().map(|(idx, _)| idx).collect())
    }

    // ── CG Solver (public, gram-based internal variant) ───────────────────

    /// Conjugate Gradient solver for the normal equation:
    ///   (Y^T C Y + λI) x = Y^T C p
    ///
    /// This public interface takes the full embedding matrix and confidence /
    /// preference vectors (length `n_other`) and forms the matvec A·v on the
    /// fly without constructing the d×d matrix.
    ///
    /// - `y_matrix`:    \[n_other × dim\] row-major "other" embedding matrix
    /// - `n_other`:     number of entities on the "other" side
    /// - `dim`:         embedding dimension
    /// - `confidences`: \[n_other\] confidence weights c_ui
    /// - `preferences`: \[n_other\] preference labels p_ui ∈ {0,1}
    /// - `x_init`:      initial solution \[dim\]
    /// - `lambda`:      regularization coefficient
    /// - `n_steps`:     maximum CG iterations
    /// - `tol`:         residual ‖r‖ convergence tolerance
    #[allow(clippy::too_many_arguments)]
    pub fn cg_solve(
        y_matrix: &[f32],
        n_other: usize,
        dim: usize,
        confidences: &[f32],
        preferences: &[f32],
        x_init: &[f32],
        lambda: f32,
        n_steps: usize,
        tol: f32,
    ) -> RecsysResult<Vec<f32>> {
        if y_matrix.len() != n_other * dim {
            return Err(RecsysError::DimensionMismatch {
                expected: n_other * dim,
                got: y_matrix.len(),
            });
        }
        if confidences.len() != n_other {
            return Err(RecsysError::DimensionMismatch {
                expected: n_other,
                got: confidences.len(),
            });
        }
        if preferences.len() != n_other {
            return Err(RecsysError::DimensionMismatch {
                expected: n_other,
                got: preferences.len(),
            });
        }
        if x_init.len() != dim {
            return Err(RecsysError::DimensionMismatch {
                expected: dim,
                got: x_init.len(),
            });
        }

        // rhs = Y^T (C p)
        let mut rhs = vec![0.0_f32; dim];
        for i in 0..n_other {
            let cp = confidences[i] * preferences[i];
            if cp == 0.0 {
                continue;
            }
            let y = &y_matrix[i * dim..(i + 1) * dim];
            for (d_idx, &yi) in y.iter().enumerate() {
                rhs[d_idx] += cp * yi;
            }
        }

        // A·v = Σ_i c_i * y_i * (y_i · v) + λ v  (implicit Y^T C Y + λI)
        let matvec = |v: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0_f32; dim];
            // Regulariser term.
            for (k, &vk) in v.iter().enumerate() {
                out[k] += lambda * vk;
            }
            // Y^T C Y · v term.
            for i in 0..n_other {
                let ci = confidences[i];
                let y = &y_matrix[i * dim..(i + 1) * dim];
                // dot = y_i · v
                let dot: f32 = y.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
                let scale = ci * dot;
                for (k, &yk) in y.iter().enumerate() {
                    out[k] += scale * yk;
                }
            }
            out
        };

        cg_iterate(&rhs, x_init, n_steps, tol, matvec)
    }

    // ── CG Solver (gram-matrix optimised, used internally) ────────────────

    /// CG solve using a pre-computed gram G = Y^T Y (dim×dim) plus a sparse
    /// correction for the (C-I) rows (only observed entities with c > 1).
    ///
    /// A·v = G·v + λv + Σ_{k∈obs} (c_k - 1)*(y_k·v)*y_k
    ///      = (Y^T Y + λI)·v + Σ_{k∈obs} alpha*(y_k·v)*y_k
    ///
    /// Here `y_obs` is the compact n_obs×dim matrix of *observed* rows,
    /// `confidences` are all `1+alpha` (so `c_k - 1 = alpha`), and
    /// `rhs` = Y^T C p (pre-computed).
    #[allow(clippy::too_many_arguments)]
    fn cg_solve_gram(
        gram: &[f32],
        y_obs: &[f32],
        n_obs: usize,
        dim: usize,
        confidences: &[f32],
        rhs: &[f32],
        x_init: &[f32],
        lambda: f32,
        n_steps: usize,
        tol: f32,
    ) -> RecsysResult<Vec<f32>> {
        // A·v = G·v + λv + Σ_{k∈obs} (c_k-1)*(y_k·v)*y_k
        let matvec = |v: &[f32]| -> Vec<f32> {
            // Start with gram-matvec + regulariser.
            let mut out = mat_vec_mul(gram, v, dim);
            for (k, &vk) in v.iter().enumerate() {
                out[k] += lambda * vk;
            }
            // Sparse correction for observed rows.
            for k in 0..n_obs {
                let c_minus_1 = confidences[k] - 1.0;
                if c_minus_1.abs() < 1e-15 {
                    continue;
                }
                let y = &y_obs[k * dim..(k + 1) * dim];
                let dot: f32 = y.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
                let scale = c_minus_1 * dot;
                for (d_idx, &yk) in y.iter().enumerate() {
                    out[d_idx] += scale * yk;
                }
            }
            out
        };

        cg_iterate(rhs, x_init, n_steps, tol, matvec)
    }
}

// ── Free helpers ──────────────────────────────────────────────────────────────

/// Compute gram matrix G = Y^T Y  (dim × dim, row-major).
fn compute_gram(y: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let mut g = vec![0.0_f32; dim * dim];
    for row in 0..n {
        let y_r = &y[row * dim..(row + 1) * dim];
        for ki in 0..dim {
            let yi = y_r[ki];
            for kj in ki..dim {
                let val = yi * y_r[kj];
                g[ki * dim + kj] += val;
                if kj != ki {
                    g[kj * dim + ki] += val;
                }
            }
        }
    }
    g
}

/// Symmetric dim×dim matrix-vector product: out = A·v.
#[inline]
fn mat_vec_mul(a: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; dim];
    for i in 0..dim {
        let row = &a[i * dim..(i + 1) * dim];
        out[i] = row.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
    }
    out
}

/// Core CG loop: solve A·x = b given an `matvec` closure implementing A·v.
fn cg_iterate<F>(
    rhs: &[f32],
    x_init: &[f32],
    n_steps: usize,
    tol: f32,
    matvec: F,
) -> RecsysResult<Vec<f32>>
where
    F: Fn(&[f32]) -> Vec<f32>,
{
    let dim = rhs.len();
    let mut x = x_init.to_vec();

    // r = b - A·x
    let ax = matvec(&x);
    let mut r: Vec<f32> = rhs
        .iter()
        .zip(ax.iter())
        .map(|(&bi, &axi)| bi - axi)
        .collect();
    let mut p = r.clone();
    let mut rs_old: f32 = r.iter().map(|&ri| ri * ri).sum();

    for _ in 0..n_steps {
        if rs_old.sqrt() < tol {
            break;
        }
        let ap = matvec(&p);
        let p_ap: f32 = p.iter().zip(ap.iter()).map(|(&pi, &api)| pi * api).sum();
        let alpha = rs_old / (p_ap + 1e-15);

        for k in 0..dim {
            x[k] += alpha * p[k];
            r[k] -= alpha * ap[k];
        }

        let rs_new: f32 = r.iter().map(|&ri| ri * ri).sum();
        let beta = rs_new / (rs_old + 1e-15);

        for k in 0..dim {
            p[k] = r[k] + beta * p[k];
        }
        rs_old = rs_new;
    }

    Ok(x)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg(n_users: usize, n_items: usize, dim: usize) -> IalsConfig {
        IalsConfig {
            n_users,
            n_items,
            dim,
            lambda: 0.01,
            alpha: 40.0,
            n_epochs: 2,
            n_cg_steps: 3,
            cg_tol: 1e-4,
        }
    }

    // ── Constructor tests ─────────────────────────────────────────────────

    #[test]
    fn new_valid_cfg_succeeds() {
        let mut rng = LcgRng::new(42);
        let cfg = default_cfg(5, 10, 8);
        let model = Ials::new(cfg, &mut rng).unwrap();
        assert_eq!(model.user_emb.len(), 5 * 8);
        assert_eq!(model.item_emb.len(), 10 * 8);
    }

    #[test]
    fn new_invalid_dim_zero_returns_err() {
        let mut rng = LcgRng::new(1);
        let cfg = default_cfg(5, 10, 0);
        assert!(matches!(
            Ials::new(cfg, &mut rng),
            Err(RecsysError::InvalidEmbeddingDim { .. })
        ));
    }

    #[test]
    fn new_invalid_n_users_zero_returns_err() {
        let mut rng = LcgRng::new(1);
        let cfg = default_cfg(0, 10, 8);
        assert!(matches!(
            Ials::new(cfg, &mut rng),
            Err(RecsysError::InvalidNumUsers { .. })
        ));
    }

    #[test]
    fn new_invalid_n_items_zero_returns_err() {
        let mut rng = LcgRng::new(1);
        let cfg = default_cfg(5, 0, 8);
        assert!(matches!(
            Ials::new(cfg, &mut rng),
            Err(RecsysError::InvalidNumItems { .. })
        ));
    }

    // ── Fit tests ─────────────────────────────────────────────────────────

    #[test]
    fn fit_empty_interactions_returns_err() {
        let mut rng = LcgRng::new(2);
        let cfg = default_cfg(5, 10, 4);
        let mut model = Ials::new(cfg, &mut rng).unwrap();
        let result = model.fit(&[], &mut rng);
        assert!(matches!(result, Err(RecsysError::EmptyInteraction)));
    }

    #[test]
    fn fit_single_interaction_succeeds() {
        let mut rng = LcgRng::new(3);
        let cfg = default_cfg(2, 3, 4);
        let mut model = Ials::new(cfg, &mut rng).unwrap();
        model.fit(&[(0, 0)], &mut rng).unwrap();
        let s = model.predict(0, 0).unwrap();
        assert!(s.is_finite());
    }

    #[test]
    fn fit_multiple_epochs_does_not_crash() {
        let mut rng = LcgRng::new(99);
        let mut cfg = default_cfg(4, 6, 8);
        cfg.n_epochs = 5;
        let mut model = Ials::new(cfg, &mut rng).unwrap();
        let interactions = vec![(0, 0), (0, 1), (1, 2), (2, 3), (3, 5)];
        model.fit(&interactions, &mut rng).unwrap();
    }

    #[test]
    fn fit_out_of_bounds_item_returns_err() {
        let mut rng = LcgRng::new(7);
        let cfg = default_cfg(5, 10, 4);
        let mut model = Ials::new(cfg, &mut rng).unwrap();
        // item id = 10 is out of bounds for n_items = 10
        let result = model.fit(&[(0, 10)], &mut rng);
        assert!(matches!(result, Err(RecsysError::UnknownItem { .. })));
    }

    #[test]
    fn fit_n_cg_steps_one_works() {
        let mut rng = LcgRng::new(11);
        let mut cfg = default_cfg(3, 5, 4);
        cfg.n_cg_steps = 1;
        let mut model = Ials::new(cfg, &mut rng).unwrap();
        model.fit(&[(0, 0), (1, 1), (2, 2)], &mut rng).unwrap();
    }

    // ── Predict tests ─────────────────────────────────────────────────────

    #[test]
    fn predict_returns_finite_value_after_fit() {
        let mut rng = LcgRng::new(5);
        let cfg = default_cfg(5, 8, 6);
        let mut model = Ials::new(cfg, &mut rng).unwrap();
        model
            .fit(&[(0, 0), (0, 2), (1, 1), (2, 3)], &mut rng)
            .unwrap();
        let s = model.predict(0, 0).unwrap();
        assert!(s.is_finite(), "predict returned non-finite: {s}");
    }

    #[test]
    fn predict_unknown_user_returns_err() {
        let mut rng = LcgRng::new(6);
        let cfg = default_cfg(5, 8, 4);
        let model = Ials::new(cfg, &mut rng).unwrap();
        assert!(matches!(
            model.predict(5, 0),
            Err(RecsysError::UnknownUser { .. })
        ));
    }

    #[test]
    fn predict_unknown_item_returns_err() {
        let mut rng = LcgRng::new(8);
        let cfg = default_cfg(5, 8, 4);
        let model = Ials::new(cfg, &mut rng).unwrap();
        assert!(matches!(
            model.predict(0, 8),
            Err(RecsysError::UnknownItem { .. })
        ));
    }

    #[test]
    fn user_embedding_changes_after_fit() {
        let mut rng = LcgRng::new(77);
        let cfg = default_cfg(3, 5, 8);
        let mut model = Ials::new(cfg, &mut rng).unwrap();
        let before: Vec<f32> = model.user_emb.clone();
        model.fit(&[(0, 0), (1, 2), (2, 4)], &mut rng).unwrap();
        let changed = before
            .iter()
            .zip(model.user_emb.iter())
            .any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(changed, "user embeddings should change after fit");
    }

    // ── Rank items tests ──────────────────────────────────────────────────

    #[test]
    fn rank_items_returns_all_items() {
        let mut rng = LcgRng::new(13);
        let cfg = default_cfg(3, 7, 4);
        let mut model = Ials::new(cfg, &mut rng).unwrap();
        model.fit(&[(0, 0), (1, 3), (2, 6)], &mut rng).unwrap();
        let ranked = model.rank_items(0).unwrap();
        assert_eq!(
            ranked.len(),
            7,
            "rank_items should return all n_items entries"
        );
    }

    #[test]
    fn rank_items_sorted_descending() {
        let mut rng = LcgRng::new(14);
        let cfg = default_cfg(3, 8, 6);
        let mut model = Ials::new(cfg, &mut rng).unwrap();
        model
            .fit(&[(0, 0), (0, 1), (1, 2), (2, 3)], &mut rng)
            .unwrap();
        let ranked = model.rank_items(0).unwrap();
        // Verify scores are non-increasing.
        let scores: Vec<f32> = ranked
            .iter()
            .map(|&i| model.predict(0, i).unwrap())
            .collect();
        for w in scores.windows(2) {
            assert!(
                w[0] >= w[1] - 1e-6,
                "rank_items not sorted descending: {} < {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn rank_items_top_is_most_observed_item() {
        // User 0 interacts heavily with item 3 across many synthetic users.
        let mut rng = LcgRng::new(2024);
        let mut cfg = default_cfg(20, 10, 8);
        cfg.n_epochs = 10;
        cfg.n_cg_steps = 5;
        let mut model = Ials::new(cfg, &mut rng).unwrap();

        // Build interactions: item 3 is observed by all 20 users.
        let mut interactions: Vec<(usize, usize)> = (0..20).map(|u| (u, 3)).collect();
        // Add some noise interactions.
        for u in 0..5 {
            interactions.push((u, 0));
        }
        model.fit(&interactions, &mut rng).unwrap();

        let ranked = model.rank_items(0).unwrap();
        // Item 3 should rank first for user 0 (it was the only item they interacted with).
        assert_eq!(
            ranked[0],
            3,
            "most interacted item should rank highest, got {:?}",
            &ranked[..3]
        );
    }

    #[test]
    fn rank_items_unknown_user_returns_err() {
        let mut rng = LcgRng::new(20);
        let cfg = default_cfg(3, 5, 4);
        let model = Ials::new(cfg, &mut rng).unwrap();
        assert!(matches!(
            model.rank_items(3),
            Err(RecsysError::UnknownUser { .. })
        ));
    }

    // ── CG solver tests ───────────────────────────────────────────────────

    #[test]
    fn cg_solve_identity_system_exact() {
        // Y = I_3 (3×3 identity), conf = [1,1,1], pref = [1,0,0]
        // Normal equation: (I + λI) x = e_1 → x = e_1/(1+λ)
        let dim = 3;
        let lambda = 1.0_f32;
        let y: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let conf = vec![1.0_f32; 3];
        let pref = vec![1.0_f32, 0.0, 0.0];
        let x_init = vec![0.0_f32; 3];
        let x = Ials::cg_solve(&y, 3, dim, &conf, &pref, &x_init, lambda, 20, 1e-6).unwrap();
        let expected = 1.0 / (1.0 + lambda);
        assert!(
            (x[0] - expected).abs() < 1e-4,
            "x[0] = {}, expected {expected}",
            x[0]
        );
        assert!(x[1].abs() < 1e-4, "x[1] = {} should be ~0", x[1]);
        assert!(x[2].abs() < 1e-4, "x[2] = {} should be ~0", x[2]);
    }

    #[test]
    fn cg_solve_single_basis_vector() {
        // Y = [[1,0]], conf = [1+40], pref = [1]
        // Normal eq: ((1+40)*1*1 + λ) x[0] = (1+40)*1  → x[0] = 41/(41+λ)
        let alpha = 40.0_f32;
        let lambda = 0.1_f32;
        let y = vec![1.0_f32, 0.0_f32];
        let conf = vec![1.0 + alpha];
        let pref = vec![1.0_f32];
        let x_init = vec![0.0_f32; 2];
        let x = Ials::cg_solve(&y, 1, 2, &conf, &pref, &x_init, lambda, 20, 1e-6).unwrap();
        let expected = (1.0 + alpha) / ((1.0 + alpha) + lambda);
        assert!(
            (x[0] - expected).abs() < 1e-4,
            "x[0] = {} expected {expected}",
            x[0]
        );
        assert!(x[1].abs() < 1e-4, "x[1] = {} should be ~0", x[1]);
    }

    #[test]
    fn cg_solve_output_length_matches_dim() {
        let dim = 5;
        let n_other = 4;
        let y = vec![0.5_f32; n_other * dim];
        let conf = vec![1.0_f32; n_other];
        let pref = vec![1.0_f32; n_other];
        let x_init = vec![0.0_f32; dim];
        let result = Ials::cg_solve(&y, n_other, dim, &conf, &pref, &x_init, 0.01, 5, 1e-4);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), dim);
    }

    #[test]
    fn cg_solve_dimension_mismatch_returns_err() {
        // y_matrix wrong length.
        let dim = 3;
        let n_other = 2;
        let y = vec![0.0_f32; 5]; // should be 6
        let conf = vec![1.0_f32; n_other];
        let pref = vec![1.0_f32; n_other];
        let x_init = vec![0.0_f32; dim];
        assert!(matches!(
            Ials::cg_solve(&y, n_other, dim, &conf, &pref, &x_init, 0.01, 5, 1e-4),
            Err(RecsysError::DimensionMismatch { .. })
        ));
    }
}
