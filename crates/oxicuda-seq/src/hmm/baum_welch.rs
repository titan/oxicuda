//! Baum-Welch (EM) parameter learning for discrete and Gaussian HMMs.

use super::forward_backward::{forward_backward, forward_backward_gaussian};
use super::hmm::{HmmDiscrete, HmmGaussian};
use crate::error::{SeqError, SeqResult};

/// Floor applied to re-estimated emission variances to keep the Gaussian
/// densities finite and the EM update numerically stable when a state collapses
/// onto (near-)identical observations.
const VAR_FLOOR: f64 = 1e-6;

/// Result of Baum-Welch training: re-estimated HMM and the per-iteration log-likelihood trace.
#[derive(Debug, Clone)]
pub struct BaumWelchResult {
    pub model: HmmDiscrete,
    pub log_likelihoods: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

/// Run Baum-Welch (single-sequence variant) until convergence.
///
/// * `init` — starting HMM
/// * `obs`  — observation sequence
/// * `max_iter` — maximum EM iterations
/// * `tol`  — convergence threshold on Δlog-likelihood
pub fn baum_welch_discrete(
    init: &HmmDiscrete,
    obs: &[usize],
    max_iter: usize,
    tol: f64,
) -> SeqResult<BaumWelchResult> {
    if obs.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let mut model = init.clone();
    let n = model.n_states;
    let k = model.n_obs;
    let t_max = obs.len();

    let mut history: Vec<f64> = Vec::with_capacity(max_iter + 1);
    let mut prev_ll = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iter_used = 0;

    for it in 0..max_iter {
        iter_used = it + 1;
        let fb = forward_backward(&model, obs)?;
        history.push(fb.log_likelihood);

        // Check convergence
        if (fb.log_likelihood - prev_ll).abs() < tol && it > 0 {
            converged = true;
            break;
        }
        prev_ll = fb.log_likelihood;

        // M-step
        // π_i = γ₀(i)
        for i in 0..n {
            model.pi[i] = fb.gamma[i];
        }

        // A_ij = Σ_t ξ_t(i,j) / Σ_t γ_t(i) (over t = 0..T-1, since ξ is T-1)
        for i in 0..n {
            let denom: f64 = (0..t_max - 1).map(|t| fb.gamma[t * n + i]).sum();
            for j in 0..n {
                let num: f64 = (0..t_max - 1).map(|t| fb.xi[t * n * n + i * n + j]).sum();
                model.a[i * n + j] = if denom > 1e-300 {
                    num / denom
                } else {
                    1.0 / n as f64
                };
            }
            // Re-normalise A row (defensive)
            let row_sum: f64 = model.a[i * n..i * n + n].iter().sum();
            if row_sum > 1e-300 {
                for v in model.a[i * n..i * n + n].iter_mut() {
                    *v /= row_sum;
                }
            } else {
                for v in model.a[i * n..i * n + n].iter_mut() {
                    *v = 1.0 / n as f64;
                }
            }
        }

        // B_j(k) = Σ_{t: o_t=k} γ_t(j) / Σ_t γ_t(j) (over all t)
        for j in 0..n {
            let denom: f64 = (0..t_max).map(|t| fb.gamma[t * n + j]).sum();
            for sym in 0..k {
                let num: f64 = (0..t_max)
                    .filter(|&t| obs[t] == sym)
                    .map(|t| fb.gamma[t * n + j])
                    .sum();
                model.b[j * k + sym] = if denom > 1e-300 {
                    num / denom
                } else {
                    1.0 / k as f64
                };
            }
            let row_sum: f64 = model.b[j * k..j * k + k].iter().sum();
            if row_sum > 1e-300 {
                for v in model.b[j * k..j * k + k].iter_mut() {
                    *v /= row_sum;
                }
            } else {
                for v in model.b[j * k..j * k + k].iter_mut() {
                    *v = 1.0 / k as f64;
                }
            }
        }

        // Re-normalise π
        let s: f64 = model.pi.iter().sum();
        if s > 1e-300 {
            for v in model.pi.iter_mut() {
                *v /= s;
            }
        } else {
            for v in model.pi.iter_mut() {
                *v = 1.0 / n as f64;
            }
        }
    }

    // Final likelihood for the trace
    let fb_final = forward_backward(&model, obs)?;
    history.push(fb_final.log_likelihood);

    Ok(BaumWelchResult {
        model,
        log_likelihoods: history,
        iterations: iter_used,
        converged,
    })
}

/// Result of Gaussian Baum-Welch training: the re-estimated Gaussian HMM and the
/// per-iteration log-likelihood trace.
#[derive(Debug, Clone)]
pub struct BaumWelchGaussianResult {
    pub model: HmmGaussian,
    pub log_likelihoods: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

/// Run Baum-Welch (EM) for a Gaussian-emission HMM with diagonal covariance.
///
/// Each iteration runs the [`forward_backward_gaussian`] E-step to obtain the
/// posterior state probabilities γ and edge posteriors ξ, then performs the
/// closed-form Gaussian M-step:
///
/// * initial    `π_i  = γ_0(i)`
/// * transition `A_ij = Σ_t ξ_t(i,j) / Σ_{t<T-1} γ_t(i)`
/// * mean       `μ_k(d) = Σ_t γ_t(k) x_t(d)            / Σ_t γ_t(k)`
/// * variance   `σ²_k(d) = Σ_t γ_t(k) (x_t(d) − μ_k(d))² / Σ_t γ_t(k)`
///
/// Variances are floored at `VAR_FLOOR` so densities remain finite even when a
/// state collapses onto near-identical observations. Supports both scalar
/// (`dim == 1`) and multivariate-diagonal (`dim > 1`) emissions.
///
/// * `init`     — starting Gaussian HMM
/// * `x`        — observation sequence, `T × dim` row-major (length must be a
///   multiple of `init.dim`)
/// * `max_iter` — maximum EM iterations
/// * `tol`      — convergence threshold on Δlog-likelihood
///
/// Mirrors [`baum_welch_discrete`]: returns the fitted model together with the
/// full per-iteration log-likelihood trace, the number of iterations performed,
/// and whether convergence was reached.
pub fn baum_welch_gaussian(
    init: &HmmGaussian,
    x: &[f64],
    max_iter: usize,
    tol: f64,
) -> SeqResult<BaumWelchGaussianResult> {
    if x.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    if x.len() % init.dim != 0 {
        return Err(SeqError::DimensionMismatch {
            a: x.len(),
            b: init.dim,
        });
    }
    let mut model = init.clone();
    let n = model.n_states;
    let dim = model.dim;
    let t_max = x.len() / dim;

    let mut history: Vec<f64> = Vec::with_capacity(max_iter + 1);
    let mut prev_ll = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iter_used = 0;

    for it in 0..max_iter {
        iter_used = it + 1;
        let fb = forward_backward_gaussian(&model, x)?;
        history.push(fb.log_likelihood);

        // Check convergence (after the first iteration).
        if (fb.log_likelihood - prev_ll).abs() < tol && it > 0 {
            converged = true;
            break;
        }
        prev_ll = fb.log_likelihood;

        // M-step.
        // π_i = γ_0(i)
        for i in 0..n {
            model.pi[i] = fb.gamma[i];
        }

        // A_ij = Σ_t ξ_t(i,j) / Σ_t γ_t(i)  (over t = 0..T-1, since ξ is T-1)
        for i in 0..n {
            let denom: f64 = (0..t_max - 1).map(|t| fb.gamma[t * n + i]).sum();
            for j in 0..n {
                let num: f64 = (0..t_max - 1).map(|t| fb.xi[t * n * n + i * n + j]).sum();
                model.a[i * n + j] = if denom > 1e-300 {
                    num / denom
                } else {
                    1.0 / n as f64
                };
            }
            // Re-normalise A row (defensive).
            let row_sum: f64 = model.a[i * n..i * n + n].iter().sum();
            if row_sum > 1e-300 {
                for v in model.a[i * n..i * n + n].iter_mut() {
                    *v /= row_sum;
                }
            } else {
                for v in model.a[i * n..i * n + n].iter_mut() {
                    *v = 1.0 / n as f64;
                }
            }
        }

        // Gaussian emissions: γ-weighted sample mean / variance per state & dim.
        for k in 0..n {
            let denom: f64 = (0..t_max).map(|t| fb.gamma[t * n + k]).sum();
            if denom > 1e-300 {
                for d in 0..dim {
                    let mut mean = 0.0;
                    for t in 0..t_max {
                        mean += fb.gamma[t * n + k] * x[t * dim + d];
                    }
                    mean /= denom;
                    let mut var = 0.0;
                    for t in 0..t_max {
                        let diff = x[t * dim + d] - mean;
                        var += fb.gamma[t * n + k] * diff * diff;
                    }
                    var /= denom;
                    model.means[k * dim + d] = mean;
                    model.vars[k * dim + d] = var.max(VAR_FLOOR);
                }
            } else {
                // State carries (effectively) no posterior mass: keep its means,
                // only enforce the variance floor.
                for d in 0..dim {
                    model.vars[k * dim + d] = model.vars[k * dim + d].max(VAR_FLOOR);
                }
            }
        }

        // Re-normalise π.
        let s: f64 = model.pi.iter().sum();
        if s > 1e-300 {
            for v in model.pi.iter_mut() {
                *v /= s;
            }
        } else {
            for v in model.pi.iter_mut() {
                *v = 1.0 / n as f64;
            }
        }
    }

    // Final likelihood for the trace.
    let fb_final = forward_backward_gaussian(&model, x)?;
    history.push(fb_final.log_likelihood);

    Ok(BaumWelchGaussianResult {
        model,
        log_likelihoods: history,
        iterations: iter_used,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn baum_welch_monotone_nondecreasing() {
        let init = HmmDiscrete::new(
            2,
            2,
            vec![0.5, 0.5],
            vec![0.6, 0.4, 0.4, 0.6],
            vec![0.7, 0.3, 0.3, 0.7],
        )
        .expect("ok");
        let obs = vec![0, 0, 1, 1, 0, 1, 0, 0, 1, 0];
        let r = baum_welch_discrete(&init, &obs, 20, 1e-6).expect("ok");
        // Likelihood should be non-decreasing across iterations.
        for w in r.log_likelihoods.windows(2) {
            assert!(
                w[1] + 1e-6 >= w[0],
                "log-lik decreased: {} -> {}",
                w[0],
                w[1]
            );
        }
    }

    // ---- Gaussian Baum-Welch (EM) -----------------------------------------

    /// Synthesise an observation sequence (`T × dim` row-major) from a known
    /// Gaussian HMM using the crate's seeded `LcgRng`. `means` / `sigmas` are
    /// laid out `[state * dim + d]`; `sigmas` are standard deviations.
    fn synth_gaussian_hmm(
        pi: &[f64],
        a: &[f64],
        means: &[f64],
        sigmas: &[f64],
        dim: usize,
        t_max: usize,
        seed: u64,
    ) -> Vec<f64> {
        let n = pi.len();
        let mut rng = LcgRng::new(seed);
        let mut x = Vec::with_capacity(t_max * dim);
        let mut state = rng.sample_categorical(pi);
        for _ in 0..t_max {
            for d in 0..dim {
                let mu = means[state * dim + d];
                let sigma = sigmas[state * dim + d];
                x.push(mu + sigma * rng.next_normal());
            }
            state = rng.sample_categorical(&a[state * n..state * n + n]);
        }
        x
    }

    /// Bit-for-bit equality of two `f64` slices (determinism check, clippy-safe).
    fn bits_eq(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn baum_welch_gaussian_recovers_parameters() {
        // Well-separated 3-state scalar (dim = 1) Gaussian HMM: 6σ apart.
        let pi = [1.0, 0.0, 0.0];
        let a = [0.7, 0.2, 0.1, 0.1, 0.7, 0.2, 0.2, 0.1, 0.7];
        let true_means = [-6.0, 0.0, 6.0];
        let true_sigmas = [1.0, 1.0, 1.0];
        let x = synth_gaussian_hmm(&pi, &a, &true_means, &true_sigmas, 1, 3000, 0xBEEF);

        // Reasonable init: distinct means, each unambiguously nearest one cluster.
        let init = HmmGaussian::new(
            3,
            1,
            vec![1.0 / 3.0; 3],
            vec![1.0 / 3.0; 9],
            vec![-5.0, 0.5, 5.0],
            vec![1.5, 1.5, 1.5],
        )
        .expect("ok");
        let r = baum_welch_gaussian(&init, &x, 100, 1e-7).expect("ok");

        // Outputs must be finite.
        for &m in &r.model.means {
            assert!(m.is_finite());
        }
        for &v in &r.model.vars {
            assert!(v.is_finite() && v >= VAR_FLOOR);
        }

        // Match each true state to the recovered state with the nearest mean
        // (label switching is expected), then compare μ and σ.
        let mut max_mean_err = 0.0_f64;
        let mut max_sigma_err = 0.0_f64;
        for s in 0..3 {
            let mut best = 0;
            let mut best_d = f64::INFINITY;
            for k in 0..3 {
                let d = (r.model.means[k] - true_means[s]).abs();
                if d < best_d {
                    best_d = d;
                    best = k;
                }
            }
            max_mean_err = max_mean_err.max((r.model.means[best] - true_means[s]).abs());
            let sigma = r.model.vars[best].sqrt();
            max_sigma_err = max_sigma_err.max((sigma - true_sigmas[s]).abs());
        }
        assert!(max_mean_err < 0.5, "mean error too large: {max_mean_err}");
        assert!(
            max_sigma_err < 0.3,
            "sigma error too large: {max_sigma_err}"
        );
    }

    #[test]
    fn baum_welch_gaussian_monotone_log_likelihood() {
        let pi = [0.6, 0.4];
        let a = [0.8, 0.2, 0.3, 0.7];
        let true_means = [-2.0, 3.0];
        let true_sigmas = [1.0, 1.5];
        let x = synth_gaussian_hmm(&pi, &a, &true_means, &true_sigmas, 1, 400, 0xC0FFEE);

        let init = HmmGaussian::new(
            2,
            1,
            vec![0.5, 0.5],
            vec![0.5, 0.5, 0.5, 0.5],
            vec![-1.0, 1.0],
            vec![1.0, 1.0],
        )
        .expect("ok");
        let r = baum_welch_gaussian(&init, &x, 50, 1e-9).expect("ok");

        assert!(r.log_likelihoods.len() >= 2);
        // EM guarantee: log-likelihood is non-decreasing (up to float roundoff).
        for w in r.log_likelihoods.windows(2) {
            assert!(
                w[1] + 1e-6 >= w[0],
                "log-lik decreased: {} -> {}",
                w[0],
                w[1]
            );
        }
        for &ll in &r.log_likelihoods {
            assert!(ll.is_finite(), "non-finite log-likelihood: {ll}");
        }
    }

    #[test]
    fn baum_welch_gaussian_deterministic_shapes_finite() {
        // Multivariate-diagonal (dim = 2) Gaussian HMM.
        let pi = [1.0, 0.0];
        let a = [0.85, 0.15, 0.2, 0.8];
        // means laid out [s0_d0, s0_d1, s1_d0, s1_d1]
        let true_means = [0.0, 0.0, 5.0, -5.0];
        let true_sigmas = [1.0, 1.0, 1.0, 1.0];
        let x = synth_gaussian_hmm(&pi, &a, &true_means, &true_sigmas, 2, 500, 20_260_621);

        let init = HmmGaussian::new(
            2,
            2,
            vec![0.5, 0.5],
            vec![0.6, 0.4, 0.4, 0.6],
            vec![1.0, 1.0, 4.0, -4.0],
            vec![2.0, 2.0, 2.0, 2.0],
        )
        .expect("ok");
        let r1 = baum_welch_gaussian(&init, &x, 60, 1e-8).expect("ok");
        let r2 = baum_welch_gaussian(&init, &x, 60, 1e-8).expect("ok");

        // Determinism: identical inputs → bit-for-bit identical outputs.
        assert!(bits_eq(&r1.model.means, &r2.model.means));
        assert!(bits_eq(&r1.model.vars, &r2.model.vars));
        assert!(bits_eq(&r1.model.a, &r2.model.a));
        assert!(bits_eq(&r1.model.pi, &r2.model.pi));
        assert!(bits_eq(&r1.log_likelihoods, &r2.log_likelihoods));
        assert_eq!(r1.iterations, r2.iterations);

        // Shapes.
        assert_eq!(r1.model.n_states, 2);
        assert_eq!(r1.model.dim, 2);
        assert_eq!(r1.model.pi.len(), 2);
        assert_eq!(r1.model.a.len(), 4);
        assert_eq!(r1.model.means.len(), 4);
        assert_eq!(r1.model.vars.len(), 4);

        // Finite outputs; variance floor respected (no zero/negative variance).
        for &m in &r1.model.means {
            assert!(m.is_finite());
        }
        for &v in &r1.model.vars {
            assert!(v.is_finite() && v >= VAR_FLOOR, "variance {v} below floor");
        }
        for &p in &r1.model.pi {
            assert!(p.is_finite() && (0.0..=1.0).contains(&p));
        }
        for &aij in &r1.model.a {
            assert!(aij.is_finite() && (0.0..=1.0).contains(&aij));
        }
        for &ll in &r1.log_likelihoods {
            assert!(ll.is_finite());
        }
    }
}
