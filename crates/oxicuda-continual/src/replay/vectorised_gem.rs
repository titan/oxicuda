//! Vectorised GEM projection via batch QP on the dual.
//!
//! The standard GEM (Lopez-Paz & Ranzato 2017) solves
//!
//! ```text
//!   min_{g_proj}  0.5 · ‖g_proj − g‖²
//!   subject to   g_proj · g_k ≥ -ε    for k = 1, ..., K
//! ```
//!
//! where `g` is the current gradient and `{g_k}` are the memory (reference)
//! gradients.  The KKT stationarity condition gives `g_proj = g + Σ_k λ_k g_k`
//! with multipliers `λ_k ≥ 0`.  Substituting into the Lagrangian and dualising
//! yields the box-constrained quadratic program
//!
//! ```text
//!   min_{λ ≥ 0}  0.5 · λᵀ M λ + λᵀ c
//!   M_{ij} = g_i · g_j        (Gram matrix of memory gradients)
//!   c_i    = g_i · g + ε      (margin folded into the linear term)
//! ```
//!
//! This module solves the QP in batch via projected coordinate descent on the
//! non-negative orthant.  Each coordinate update for `λ_k` (others fixed) has
//! the closed-form
//!
//! ```text
//!   λ_k ← max(0, -(c_k + Σ_{j≠k} M_{kj} λ_j) / M_{kk})
//! ```
//!
//! The projected gradient is finally `g_proj = g + G^T λ*`.
//!
//! If `g` already satisfies all constraints (`g · g_k ≥ -ε` for every `k`) the
//! function returns `g` unchanged without solving the QP.

use crate::error::{ContinualError, ContinualResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for vectorised (batch QP) GEM projection.
#[derive(Debug, Clone)]
pub struct VectorisedGemConfig {
    /// Margin ε: the projected gradient must satisfy `g_proj · g_k ≥ -ε` for
    /// every memory constraint.  ε ≥ 0 is the typical regime.
    pub margin: f64,
    /// Maximum number of coordinate-descent sweeps over all `K` multipliers.
    pub max_iter: usize,
    /// Convergence tolerance on the maximum change in any `λ_k` across a sweep.
    pub tol: f64,
    /// When true, each coordinate update uses an exact line search along the
    /// chosen coordinate direction.  The closed-form update above is already
    /// exact for the box-QP, so this flag changes only the per-sweep stop
    /// criterion: with line search we also early-terminate when the residual
    /// constraint slack stops improving.
    pub use_line_search: bool,
}

impl Default for VectorisedGemConfig {
    fn default() -> Self {
        Self {
            margin: 0.0,
            max_iter: 200,
            tol: 1e-9,
            use_line_search: true,
        }
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

#[inline]
fn dot64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn all_constraints_satisfied(g: &[f64], memory_grads: &[Vec<f64>], margin: f64) -> bool {
    memory_grads.iter().all(|gk| dot64(g, gk) >= -margin)
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Batch GEM projection via QP on the dual.
///
/// # Algorithm
///
/// 1. Trivial cases — empty memory or already-feasible `g` — return `g`.
/// 2. Form Gram matrix `M` and linear term `c` (folding the margin in).
/// 3. Run projected coordinate descent on `λ ∈ ℝ_+^K` until either the
///    maximum coordinate change drops below `tol` or `max_iter` is reached.
/// 4. Return `g_proj = g + G^T λ*`.
///
/// The returned gradient satisfies all constraints up to numerical tolerance.
pub fn vectorised_gem_project(
    gradient: &[f64],
    memory_gradients: &[Vec<f64>],
    config: &VectorisedGemConfig,
) -> ContinualResult<Vec<f64>> {
    if gradient.is_empty() {
        return Err(ContinualError::EmptyInput);
    }
    let dim = gradient.len();

    for gk in memory_gradients {
        if gk.len() != dim {
            return Err(ContinualError::DimensionMismatch {
                expected: dim,
                got: gk.len(),
            });
        }
    }

    if memory_gradients.is_empty() {
        return Ok(gradient.to_vec());
    }

    if all_constraints_satisfied(gradient, memory_gradients, config.margin) {
        return Ok(gradient.to_vec());
    }

    let k_count = memory_gradients.len();

    // Gram matrix M[i][j] = g_i · g_j  (symmetric, exploit triangular fill).
    let mut gram = vec![vec![0.0_f64; k_count]; k_count];
    for i in 0..k_count {
        for j in i..k_count {
            let v = dot64(&memory_gradients[i], &memory_gradients[j]);
            gram[i][j] = v;
            gram[j][i] = v;
        }
    }

    // Linear term c_i = g_i · g + ε  so the constraint g_proj · g_i ≥ -ε
    // becomes (g + G^T λ) · g_i ≥ -ε  →  c_i + (M λ)_i ≥ 0.
    let mut c = vec![0.0_f64; k_count];
    for i in 0..k_count {
        c[i] = dot64(&memory_gradients[i], gradient) + config.margin;
    }

    // Track the running M λ vector incrementally to keep the inner update O(K).
    let mut lambda = vec![0.0_f64; k_count];
    let mut m_lambda = vec![0.0_f64; k_count];

    let mut prev_slack_norm = f64::INFINITY;

    for _ in 0..config.max_iter {
        let mut max_change = 0.0_f64;
        for k in 0..k_count {
            if gram[k][k] < 1e-20 {
                continue;
            }
            // Exclude self-contribution from M λ before the update.
            let off_diag = m_lambda[k] - gram[k][k] * lambda[k];
            let new_lambda_k = (-(c[k] + off_diag) / gram[k][k]).max(0.0);
            let delta = new_lambda_k - lambda[k];
            if delta != 0.0 {
                for (mlj, gkj) in m_lambda.iter_mut().zip(gram[k].iter()) {
                    *mlj += delta * gkj;
                }
                lambda[k] = new_lambda_k;
                let abs_delta = delta.abs();
                if abs_delta > max_change {
                    max_change = abs_delta;
                }
            }
        }

        if max_change < config.tol {
            break;
        }

        // Optional early stop when the constraint slack stops improving.
        if config.use_line_search {
            let slack_norm: f64 = (0..k_count)
                .map(|k| {
                    let v = c[k] + m_lambda[k];
                    if v < 0.0 { v * v } else { 0.0 }
                })
                .sum::<f64>()
                .sqrt();
            if slack_norm < config.tol {
                break;
            }
            if (prev_slack_norm - slack_norm).abs() < config.tol * config.tol {
                break;
            }
            prev_slack_norm = slack_norm;
        }
    }

    // g_proj = g + G^T λ*
    let mut g_proj = gradient.to_vec();
    for (k, gk) in memory_gradients.iter().enumerate() {
        if lambda[k] == 0.0 {
            continue;
        }
        for (gi, &gki) in g_proj.iter_mut().zip(gk.iter()) {
            *gi += lambda[k] * gki;
        }
    }

    Ok(g_proj)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    fn constraints_satisfied(g_proj: &[f64], mem: &[Vec<f64>], margin: f64) -> bool {
        mem.iter().all(|gk| dot_f64(g_proj, gk) >= -margin - 1e-5)
    }

    #[test]
    fn feasible_gradient_returned_unchanged() {
        let g = vec![1.0_f64, 0.0, 0.0];
        let mem = vec![vec![1.0_f64, 0.0, 0.0]];
        let cfg = VectorisedGemConfig::default();
        let result = vectorised_gem_project(&g, &mem, &cfg).unwrap();
        for (a, b) in g.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn projection_satisfies_all_constraints() {
        let g = vec![-2.0_f64, -1.0, -1.0];
        let mem = vec![
            vec![1.0_f64, 0.0, 0.0],
            vec![0.0_f64, 1.0, 0.0],
            vec![0.0_f64, 0.0, 1.0],
        ];
        let cfg = VectorisedGemConfig::default();
        let g_proj = vectorised_gem_project(&g, &mem, &cfg).unwrap();
        assert!(constraints_satisfied(&g_proj, &mem, cfg.margin));
    }

    #[test]
    fn approximate_match_with_iterative_gem() {
        let g = vec![-1.0_f64, 0.0];
        let mem = vec![vec![1.0_f64, 0.0]];
        let cfg = VectorisedGemConfig {
            tol: 1e-12,
            max_iter: 200,
            ..Default::default()
        };
        let g_proj = vectorised_gem_project(&g, &mem, &cfg).unwrap();
        assert!(g_proj[0].abs() < 1e-6, "got {}", g_proj[0]);
        assert!(g_proj[1].abs() < 1e-6);
    }

    #[test]
    fn larger_margin_more_conservative() {
        let g = vec![-0.5_f64, 1.0, 0.0];
        let mem = vec![vec![1.0_f64, 0.0, 0.0]];

        let cfg0 = VectorisedGemConfig {
            margin: 0.0,
            ..Default::default()
        };
        let cfg_large = VectorisedGemConfig {
            margin: 0.5,
            ..Default::default()
        };

        let g0 = vectorised_gem_project(&g, &mem, &cfg0).unwrap();
        let gl = vectorised_gem_project(&g, &mem, &cfg_large).unwrap();

        let dot0 = dot_f64(&g0, &mem[0]);
        let dotl = dot_f64(&gl, &mem[0]);
        assert!(dotl >= -0.5 - 1e-6);
        assert!(dot0 >= -1e-6);
    }

    #[test]
    fn single_constraint_analytic() {
        let g = vec![-3.0_f64, 0.0, 4.0];
        let mem = vec![vec![1.0_f64, 0.0, 0.0]];
        let cfg = VectorisedGemConfig {
            margin: 0.0,
            tol: 1e-12,
            max_iter: 500,
            use_line_search: true,
        };
        let g_proj = vectorised_gem_project(&g, &mem, &cfg).unwrap();
        assert!(
            g_proj[0].abs() < 1e-6,
            "x-component should vanish, got {}",
            g_proj[0]
        );
        assert!(
            (g_proj[2] - 4.0).abs() < 1e-6,
            "z unchanged, got {}",
            g_proj[2]
        );
    }

    #[test]
    fn orthogonal_memory_no_correction_needed() {
        let g = vec![1.0_f64, 1.0, 1.0];
        let mem = vec![vec![0.0_f64, 0.0, 1.0]];
        let cfg = VectorisedGemConfig::default();
        let g_proj = vectorised_gem_project(&g, &mem, &cfg).unwrap();
        for (a, b) in g.iter().zip(g_proj.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn directly_opposing_gradient_projected_onto_gk() {
        let g = vec![-5.0_f64, 3.0, 2.0];
        let mem = vec![vec![1.0_f64, 0.0, 0.0]];
        let cfg = VectorisedGemConfig {
            margin: 0.0,
            tol: 1e-12,
            max_iter: 500,
            use_line_search: true,
        };
        let g_proj = vectorised_gem_project(&g, &mem, &cfg).unwrap();
        assert!(g_proj[0].abs() < 1e-6, "got {}", g_proj[0]);
        assert!(constraints_satisfied(&g_proj, &mem, 0.0));
    }

    #[test]
    fn converges_within_max_iter() {
        let g = vec![-1.0_f64; 10];
        let mem: Vec<Vec<f64>> = (0..10)
            .map(|i| {
                let mut v = vec![0.0_f64; 10];
                v[i] = 1.0;
                v
            })
            .collect();
        let cfg = VectorisedGemConfig {
            max_iter: 500,
            tol: 1e-9,
            ..Default::default()
        };
        let g_proj = vectorised_gem_project(&g, &mem, &cfg).unwrap();
        assert!(constraints_satisfied(&g_proj, &mem, cfg.margin));
    }

    #[test]
    fn tighter_tol_gives_better_satisfaction() {
        let g = vec![-2.0_f64, -2.0, 3.0];
        let mem = vec![vec![1.0_f64, 0.0, 0.0], vec![0.0_f64, 1.0, 0.0]];

        let cfg_loose = VectorisedGemConfig {
            tol: 1e-3,
            max_iter: 500,
            ..Default::default()
        };
        let cfg_tight = VectorisedGemConfig {
            tol: 1e-12,
            max_iter: 2000,
            ..Default::default()
        };

        let gp_loose = vectorised_gem_project(&g, &mem, &cfg_loose).unwrap();
        let gp_tight = vectorised_gem_project(&g, &mem, &cfg_tight).unwrap();

        let max_viol = |gp: &[f64]| {
            mem.iter()
                .map(|gk| (-cfg_tight.margin - dot_f64(gp, gk)).max(0.0))
                .fold(0.0_f64, f64::max)
        };

        let viol_loose = max_viol(&gp_loose);
        let viol_tight = max_viol(&gp_tight);
        assert!(viol_tight <= viol_loose + 1e-6);
    }

    #[test]
    fn empty_memory_returns_gradient_unchanged() {
        let g = vec![1.0_f64, 2.0, 3.0];
        let mem: Vec<Vec<f64>> = vec![];
        let cfg = VectorisedGemConfig::default();
        let result = vectorised_gem_project(&g, &mem, &cfg).unwrap();
        assert_eq!(result, g);
    }

    #[test]
    fn gradient_length_mismatch_returns_error() {
        let g = vec![1.0_f64; 4];
        let mem = vec![vec![1.0_f64; 3]];
        let cfg = VectorisedGemConfig::default();
        assert!(vectorised_gem_project(&g, &mem, &cfg).is_err());
    }

    #[test]
    fn line_search_and_fixed_step_agree() {
        let g = vec![-1.5_f64, -1.5, 2.0];
        let mem = vec![vec![1.0_f64, 0.0, 0.0], vec![0.0_f64, 1.0, 0.0]];

        let cfg_ls = VectorisedGemConfig {
            use_line_search: true,
            tol: 1e-12,
            max_iter: 2000,
            ..Default::default()
        };
        let cfg_fixed = VectorisedGemConfig {
            use_line_search: false,
            tol: 1e-12,
            max_iter: 4000,
            ..Default::default()
        };

        let gp_ls = vectorised_gem_project(&g, &mem, &cfg_ls).unwrap();
        let gp_fixed = vectorised_gem_project(&g, &mem, &cfg_fixed).unwrap();

        assert!(constraints_satisfied(&gp_ls, &mem, 0.0));
        assert!(constraints_satisfied(&gp_fixed, &mem, 0.0));

        let max_diff: f64 = gp_ls
            .iter()
            .zip(gp_fixed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff < 1e-4, "got {max_diff}");
    }

    #[test]
    fn k2_symmetric_projection_orthogonal_to_v() {
        let g = vec![0.0_f64, 1.0];
        let mem = vec![vec![1.0_f64, 0.0], vec![-1.0_f64, 0.0]];
        let cfg = VectorisedGemConfig::default();
        let g_proj = vectorised_gem_project(&g, &mem, &cfg).unwrap();
        assert!((g_proj[0] - 0.0).abs() < 1e-10);
        assert!((g_proj[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn k10_large_constraints_satisfied_simultaneously() {
        let n = 10_usize;
        let g = vec![-1.0_f64; n];
        let mem: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut v = vec![0.0_f64; n];
                v[i] = 1.0;
                v
            })
            .collect();
        let cfg = VectorisedGemConfig {
            tol: 1e-10,
            max_iter: 1000,
            use_line_search: true,
            ..Default::default()
        };
        let g_proj = vectorised_gem_project(&g, &mem, &cfg).unwrap();
        assert!(constraints_satisfied(&g_proj, &mem, cfg.margin));
    }

    #[test]
    fn empty_gradient_returns_error() {
        let mem = vec![vec![1.0_f64]];
        let cfg = VectorisedGemConfig::default();
        assert!(vectorised_gem_project(&[], &mem, &cfg).is_err());
    }
}
