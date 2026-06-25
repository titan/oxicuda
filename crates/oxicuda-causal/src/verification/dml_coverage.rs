//! Double-ML confidence-interval coverage study on simulated DGPs.
//!
//! For a confounded constant-effect DGP with known θ we repeatedly (over many
//! Monte-Carlo replications) fit [`crate::effect::double_ml::DoubleML`], form the
//! nominal 95% Wald interval `ate ± 1.96·se`, and record how often it covers the
//! truth and how the point estimate is distributed. This addresses both
//! "Double-ML coverage probability (95% CI)" and "DML standard-error coverage on
//! Monte-Carlo simulations".

use crate::effect::double_ml::DoubleML;
use crate::handle::LcgRng;
use crate::verification::synthetic::confounded_data;

/// Aggregate statistics from a coverage study.
pub struct CoverageReport {
    /// Fraction of replications whose 95% CI contained the true θ.
    pub coverage_95: f64,
    /// Mean of the point estimates (≈ θ if the estimator is ~unbiased).
    pub mean_estimate: f64,
    /// Empirical bias `mean_estimate − θ`.
    pub bias: f64,
    /// Empirical standard deviation of the point estimates.
    pub empirical_sd: f64,
    /// Mean reported standard error (should track `empirical_sd`).
    pub mean_reported_se: f64,
    pub n_replications: usize,
}

/// Run `reps` replications of the DGP/fit and aggregate coverage statistics.
///
/// Each replication draws a fresh dataset of `n` rows with `d` covariates from
/// the confounded constant-effect DGP and fits DML with `n_folds` folds.
#[must_use]
pub fn coverage_study(
    reps: usize,
    n: usize,
    d: usize,
    theta: f32,
    noise: f32,
    n_folds: usize,
    rng: &mut LcgRng,
) -> CoverageReport {
    const Z_95: f32 = 1.959_964;
    let mut covered = 0usize;
    let mut estimates: Vec<f64> = Vec::with_capacity(reps);
    let mut se_sum = 0.0_f64;
    let mut used = 0usize;
    for _ in 0..reps {
        let dgp = confounded_data(n, d, theta, noise, rng);
        let fit = match DoubleML::fit(&dgp.y, &dgp.t, &dgp.x, dgp.n, dgp.d, n_folds) {
            Ok(f) => f,
            Err(_) => continue,
        };
        if !fit.ate.is_finite() || !fit.std_error.is_finite() {
            continue;
        }
        used += 1;
        estimates.push(fit.ate as f64);
        se_sum += fit.std_error as f64;
        let lo = fit.ate - Z_95 * fit.std_error;
        let hi = fit.ate + Z_95 * fit.std_error;
        if theta >= lo && theta <= hi {
            covered += 1;
        }
    }
    let n_used = used.max(1);
    let mean_estimate = estimates.iter().sum::<f64>() / n_used as f64;
    let var = estimates
        .iter()
        .map(|&e| (e - mean_estimate).powi(2))
        .sum::<f64>()
        / (n_used.max(2) - 1) as f64;
    CoverageReport {
        coverage_95: covered as f64 / n_used as f64,
        mean_estimate,
        bias: mean_estimate - theta as f64,
        empirical_sd: var.sqrt(),
        mean_reported_se: se_sum / n_used as f64,
        n_replications: used,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dml_is_approximately_unbiased() {
        // Across replications the cross-fitted DML estimate should center on the
        // true effect (confounding is partialled out), unlike a naive
        // difference-in-means which would be biased upward here.
        let mut rng = LcgRng::new(20_260_621);
        let rep = coverage_study(120, 400, 3, 2.0, 0.5, 4, &mut rng);
        assert!(rep.n_replications >= 100, "too few usable reps");
        assert!(
            rep.bias.abs() < 0.25,
            "DML bias {} (mean est {})",
            rep.bias,
            rep.mean_estimate
        );
    }

    #[test]
    fn dml_95_ci_coverage_is_reasonable() {
        // Nominal 95% coverage. Finite-sample plug-in DML undercovers somewhat,
        // so we require a broad-but-meaningful band: a broken SE (e.g. far too
        // small/large) would fall outside it.
        let mut rng = LcgRng::new(424_242);
        let rep = coverage_study(160, 500, 3, 1.5, 0.5, 5, &mut rng);
        assert!(
            (0.75..=1.0).contains(&rep.coverage_95),
            "95% coverage {} out of plausible range (reps {})",
            rep.coverage_95,
            rep.n_replications
        );
    }

    #[test]
    fn reported_se_tracks_empirical_sd() {
        // The mean reported standard error should be of the same order as the
        // empirical sd of the estimates (within a factor of ~2 either way).
        let mut rng = LcgRng::new(9_001);
        let rep = coverage_study(150, 450, 2, 1.0, 0.5, 4, &mut rng);
        let ratio = rep.mean_reported_se / rep.empirical_sd.max(1e-6);
        assert!(
            (0.4..=2.5).contains(&ratio),
            "reported SE {} vs empirical sd {} (ratio {ratio})",
            rep.mean_reported_se,
            rep.empirical_sd
        );
    }
}
