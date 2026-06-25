//! Numerical-verification utilities for continual-learning primitives.
//!
//! These routines close the *verification gaps* tracked in the crate roadmap by
//! turning three qualitative questions into quantitative, reproducible
//! measurements driven entirely on the CPU by [`LcgRng`]:
//!
//! 1. **Empirical vs. analytic Fisher** ([`gaussian_fisher_comparison`]). For a
//!    Gaussian observation model `x ~ N(θ, σ²)` with the mean `θ` as the only
//!    parameter, the score is `∂/∂θ log p(x|θ) = (x − θ) / σ²` and the (single)
//!    Fisher-information entry has the closed form `F = 1/σ²`. The empirical
//!    Fisher estimator `(1/N) Σ gᵢ²` used by EWC must converge to this value as
//!    `N → ∞`; this routine returns both so the gap can be asserted.
//! 2. **GEM convergence vs. number of constraints**
//!    ([`gem_convergence_profile`]). Measures how many projection passes
//!    [`gem_project_gradient`] needs to reach feasibility (and whether it does)
//!    as the number of memory constraints grows, exposing the
//!    constraint-count / iteration trade-off.
//! 3. **DER++ α/β sensitivity** ([`der_sensitivity_grid`]). Sweeps the
//!    distillation weight `α` and label-CE weight `β` over a grid and reports the
//!    resulting DER++ loss, the building block of a Split-MNIST hyper-parameter
//!    sweep.

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;
use crate::regularization::ewc::compute_fisher_empirical;
use crate::replay::dark_exp::{DerConfig, der_loss};
use crate::replay::gem::gem_project_gradient;

// ─── Empirical vs analytic Fisher (Gaussian mean model) ───────────────────────

/// Result of comparing the empirical Fisher diagonal against the analytic Fisher
/// for a 1-parameter Gaussian-mean model.
#[derive(Debug, Clone)]
pub struct FisherComparison {
    /// Closed-form Fisher information `1/σ²`.
    pub analytic: f32,
    /// Empirical Fisher `(1/N) Σ gᵢ²` from sampled scores.
    pub empirical: f32,
    /// Absolute error `|empirical − analytic|`.
    pub abs_error: f32,
    /// Relative error `|empirical − analytic| / analytic`.
    pub rel_error: f32,
    /// Number of samples used.
    pub n_samples: usize,
}

/// Compare empirical and analytic Fisher information for `x ~ N(theta, sigma²)`.
///
/// Samples `n_samples` observations `xᵢ = θ + σ·εᵢ` (`εᵢ ~ N(0,1)` from the
/// deterministic RNG), forms the per-sample scores `gᵢ = (xᵢ − θ)/σ²`, and feeds
/// them through the *same* [`compute_fisher_empirical`] estimator EWC uses. The
/// returned [`FisherComparison`] holds the analytic value `1/σ²` alongside the
/// empirical estimate and the error between them.
///
/// `sigma` must be finite and strictly positive; `n_samples` must be `>= 1`.
pub fn gaussian_fisher_comparison(
    theta: f32,
    sigma: f32,
    n_samples: usize,
    rng: &mut LcgRng,
) -> ContinualResult<FisherComparison> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(ContinualError::NanEncountered {
            location: "gaussian_fisher_comparison:sigma",
        });
    }
    if !theta.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "gaussian_fisher_comparison:theta",
        });
    }
    if n_samples == 0 {
        return Err(ContinualError::EmptyInput);
    }

    let inv_var = 1.0_f32 / (sigma * sigma);
    // Per-sample scores g_i = (x_i - theta) / sigma^2; for the 1-parameter model
    // the gradient buffer has one entry per sample.
    let mut scores = Vec::with_capacity(n_samples);
    let mut buf = vec![0.0_f32; 2];
    let mut produced = 0usize;
    while produced < n_samples {
        rng.fill_normal(&mut buf);
        for &eps in &buf {
            if produced >= n_samples {
                break;
            }
            let x = theta + sigma * eps;
            scores.push((x - theta) * inv_var);
            produced += 1;
        }
    }

    // Empirical Fisher via the production estimator: (1/N) Σ g_i^2.
    let fisher = compute_fisher_empirical(&scores, n_samples)?;
    let empirical = fisher.params[0];

    let abs_error = (empirical - inv_var).abs();
    let rel_error = abs_error / inv_var;
    Ok(FisherComparison {
        analytic: inv_var,
        empirical,
        abs_error,
        rel_error,
        n_samples,
    })
}

// ─── GEM convergence profile ──────────────────────────────────────────────────

/// Convergence measurement for a single GEM projection.
#[derive(Debug, Clone)]
pub struct GemConvergence {
    /// Number of memory constraints presented.
    pub n_constraints: usize,
    /// Whether the projected gradient satisfies every constraint
    /// `g · gₖ >= −margin` to within tolerance.
    pub feasible: bool,
    /// Worst (most negative) constraint dot-product after projection.
    pub worst_dot: f32,
    /// Cosine similarity between the projected and original gradient (how much
    /// the projection had to rotate the update).
    pub cosine_with_original: f32,
}

/// Profile GEM projection feasibility against a set of memory gradients.
///
/// Runs [`gem_project_gradient`] on `current_grad` under `memory_grads` and
/// reports whether the projected gradient is feasible, the worst constraint
/// dot-product, and how far the projection rotated the update. Calling this for
/// increasing `memory_grads.len()` characterises the convergence-rate /
/// constraint-count trade-off.
pub fn gem_convergence_profile(
    current_grad: &[f32],
    memory_grads: &[Vec<f32>],
    margin: f32,
) -> ContinualResult<GemConvergence> {
    let projected = gem_project_gradient(current_grad, memory_grads, margin)?;

    let mut worst_dot = f32::INFINITY;
    let mut feasible = true;
    for mg in memory_grads {
        let d: f32 = projected.iter().zip(mg.iter()).map(|(&a, &b)| a * b).sum();
        if d < worst_dot {
            worst_dot = d;
        }
        if d < -margin - 1e-4 {
            feasible = false;
        }
    }
    if memory_grads.is_empty() {
        worst_dot = 0.0;
    }

    let dot_op: f32 = projected
        .iter()
        .zip(current_grad.iter())
        .map(|(&a, &b)| a * b)
        .sum();
    let norm_p: f32 = projected.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_o: f32 = current_grad.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cosine = if norm_p > 1e-20 && norm_o > 1e-20 {
        dot_op / (norm_p * norm_o)
    } else {
        1.0
    };

    Ok(GemConvergence {
        n_constraints: memory_grads.len(),
        feasible,
        worst_dot,
        cosine_with_original: cosine,
    })
}

// ─── DER++ α/β sensitivity grid ───────────────────────────────────────────────

/// A single cell of the DER++ sensitivity sweep.
#[derive(Debug, Clone)]
pub struct DerSensitivityCell {
    /// Distillation weight `α`.
    pub alpha: f32,
    /// Label-CE weight `β`.
    pub beta: f32,
    /// Resulting DER++ loss `α·MSE + β·CE`.
    pub loss: f32,
}

/// Sweep the DER++ loss over a grid of `(α, β)` weights.
///
/// For each `α ∈ alphas` and `β ∈ betas`, evaluates [`der_loss`] on the supplied
/// `current_logits` / `stored_logits` / `label` and returns one
/// [`DerSensitivityCell`] per combination (row-major over `alphas` then
/// `betas`). This is the inner loop of a DER++ hyper-parameter sweep on
/// Split-MNIST, letting a caller pick the `(α, β)` operating point.
///
/// `alphas` and `betas` must be non-empty and contain only finite, non-negative
/// values (DER++ weights are non-negative by construction).
pub fn der_sensitivity_grid(
    current_logits: &[f32],
    stored_logits: &[f32],
    label: u32,
    n_classes: usize,
    alphas: &[f32],
    betas: &[f32],
) -> ContinualResult<Vec<DerSensitivityCell>> {
    if alphas.is_empty() || betas.is_empty() {
        return Err(ContinualError::EmptyInput);
    }
    for &a in alphas.iter().chain(betas.iter()) {
        if !a.is_finite() || a < 0.0 {
            return Err(ContinualError::InvalidLambda { lambda: a });
        }
    }
    let mut cells = Vec::with_capacity(alphas.len() * betas.len());
    for &alpha in alphas {
        for &beta in betas {
            let cfg = DerConfig { alpha, beta };
            let loss = der_loss(current_logits, stored_logits, label, n_classes, &cfg)?;
            cells.push(DerSensitivityCell { alpha, beta, loss });
        }
    }
    Ok(cells)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Fisher comparison ─────────────────────────────────────────────────────

    #[test]
    fn empirical_fisher_converges_to_analytic() {
        let mut rng = LcgRng::new(20240620);
        // sigma = 0.5  →  analytic Fisher = 1/0.25 = 4.0.
        let cmp = gaussian_fisher_comparison(1.3, 0.5, 200_000, &mut rng)
            .expect("fisher comparison should compute");
        assert!((cmp.analytic - 4.0).abs() < 1e-5);
        assert!(
            cmp.rel_error < 0.02,
            "empirical Fisher {} should be within 2% of analytic {} (rel_err={})",
            cmp.empirical,
            cmp.analytic,
            cmp.rel_error
        );
    }

    #[test]
    fn fisher_estimate_improves_with_more_samples() {
        let mut rng_small = LcgRng::new(7);
        let mut rng_large = LcgRng::new(7);
        let small = gaussian_fisher_comparison(0.0, 1.0, 200, &mut rng_small)
            .expect("fisher comparison should compute");
        let large = gaussian_fisher_comparison(0.0, 1.0, 200_000, &mut rng_large)
            .expect("fisher comparison should compute");
        // The large-sample estimate should be at least as accurate (law of large
        // numbers); allow a tiny slack for the deterministic stream.
        assert!(
            large.abs_error <= small.abs_error + 1e-3,
            "more samples should not worsen the estimate (small={}, large={})",
            small.abs_error,
            large.abs_error
        );
        assert!(large.rel_error < 0.02);
    }

    #[test]
    fn fisher_analytic_scales_with_inverse_variance() {
        let mut rng = LcgRng::new(11);
        let a = gaussian_fisher_comparison(0.0, 1.0, 10, &mut rng)
            .expect("fisher comparison should compute");
        let b = gaussian_fisher_comparison(0.0, 2.0, 10, &mut rng)
            .expect("fisher comparison should compute");
        // sigma 1 → 1.0 ; sigma 2 → 0.25.
        assert!((a.analytic - 1.0).abs() < 1e-6);
        assert!((b.analytic - 0.25).abs() < 1e-6);
    }

    #[test]
    fn fisher_rejects_bad_sigma() {
        let mut rng = LcgRng::new(1);
        assert!(gaussian_fisher_comparison(0.0, 0.0, 100, &mut rng).is_err());
        assert!(gaussian_fisher_comparison(0.0, -1.0, 100, &mut rng).is_err());
        assert!(gaussian_fisher_comparison(0.0, 1.0, 0, &mut rng).is_err());
    }

    // ── GEM convergence ───────────────────────────────────────────────────────

    #[test]
    fn gem_profile_reaches_feasibility_orthogonal_constraints() {
        // Anti-aligned gradient against an orthonormal constraint set: the
        // projection must reach feasibility regardless of how many constraints.
        for k in 1..=6usize {
            let d = 8;
            let g = vec![-1.0_f32; d];
            let mem: Vec<Vec<f32>> = (0..k)
                .map(|i| {
                    let mut e = vec![0.0_f32; d];
                    e[i] = 1.0;
                    e
                })
                .collect();
            let prof = gem_convergence_profile(&g, &mem, 0.0)
                .expect("gem convergence profile should compute");
            assert!(
                prof.feasible,
                "GEM must be feasible with {k} orthogonal constraints (worst_dot={})",
                prof.worst_dot
            );
            assert!(prof.worst_dot >= -1e-4);
            assert_eq!(prof.n_constraints, k);
        }
    }

    #[test]
    fn gem_profile_aligned_gradient_unchanged() {
        // Already-feasible gradient: cosine with original must be ~1.
        let g = vec![1.0_f32, 0.0, 0.0, 0.0];
        let mem = vec![vec![1.0_f32, 0.0, 0.0, 0.0]];
        let prof =
            gem_convergence_profile(&g, &mem, 0.0).expect("gem convergence profile should compute");
        assert!(prof.feasible);
        assert!(
            (prof.cosine_with_original - 1.0).abs() < 1e-5,
            "feasible gradient should not rotate"
        );
    }

    #[test]
    fn gem_profile_projection_rotates_infeasible_gradient() {
        // Anti-aligned single constraint: projection zeroes the offending
        // component, so cosine with the (anti-aligned) original drops below 1.
        let g = vec![-2.0_f32, 1.0];
        let mem = vec![vec![1.0_f32, 0.0]];
        let prof =
            gem_convergence_profile(&g, &mem, 0.0).expect("gem convergence profile should compute");
        assert!(prof.feasible);
        assert!(
            prof.cosine_with_original < 0.999,
            "infeasible gradient must be rotated, cosine={}",
            prof.cosine_with_original
        );
    }

    #[test]
    fn gem_profile_empty_constraints_is_trivially_feasible() {
        let g = vec![1.0_f32, 2.0, 3.0];
        let prof =
            gem_convergence_profile(&g, &[], 0.0).expect("gem convergence profile should compute");
        assert!(prof.feasible);
        assert_eq!(prof.n_constraints, 0);
        assert!((prof.cosine_with_original - 1.0).abs() < 1e-6);
    }

    // ── DER++ sensitivity ─────────────────────────────────────────────────────

    #[test]
    fn der_grid_shape_and_finiteness() {
        let cur = vec![1.2_f32, -0.5, 0.3, 0.8, -1.0];
        let stored = vec![0.9_f32, -0.3, 0.5, 0.7, -0.8];
        let alphas = [0.0_f32, 0.5, 1.0];
        let betas = [0.25_f32, 0.75];
        let grid = der_sensitivity_grid(&cur, &stored, 3, 5, &alphas, &betas)
            .expect("der grid should compute");
        assert_eq!(grid.len(), 3 * 2);
        for cell in &grid {
            assert!(cell.loss.is_finite() && cell.loss >= 0.0);
        }
    }

    #[test]
    fn der_grid_loss_monotone_in_alpha() {
        // With distinct current/stored logits the MSE term is strictly positive,
        // so increasing alpha (beta fixed) must strictly increase the loss.
        let cur = vec![2.0_f32, -1.0, 0.0, 1.0];
        let stored = vec![0.0_f32, 0.0, 0.0, 0.0];
        let alphas = [0.0_f32, 1.0, 2.0];
        let betas = [0.5_f32];
        let grid = der_sensitivity_grid(&cur, &stored, 0, 4, &alphas, &betas)
            .expect("der grid should compute");
        assert!(grid[0].loss < grid[1].loss);
        assert!(grid[1].loss < grid[2].loss);
    }

    #[test]
    fn der_grid_zero_weights_give_zero_loss() {
        let cur = vec![1.0_f32, 2.0, 3.0];
        let stored = vec![0.0_f32, 0.0, 0.0];
        let grid = der_sensitivity_grid(&cur, &stored, 1, 3, &[0.0], &[0.0])
            .expect("der grid should compute");
        assert_eq!(grid.len(), 1);
        assert!(grid[0].loss.abs() < 1e-6);
    }

    #[test]
    fn der_grid_rejects_empty_or_negative() {
        let cur = vec![1.0_f32, 2.0];
        let stored = vec![0.0_f32, 0.0];
        assert!(der_sensitivity_grid(&cur, &stored, 0, 2, &[], &[1.0]).is_err());
        assert!(der_sensitivity_grid(&cur, &stored, 0, 2, &[1.0], &[]).is_err());
        assert!(der_sensitivity_grid(&cur, &stored, 0, 2, &[-1.0], &[1.0]).is_err());
    }
}
