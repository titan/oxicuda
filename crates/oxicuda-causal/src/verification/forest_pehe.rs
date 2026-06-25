//! Causal-forest PEHE on simulated heterogeneous-effect DGPs.
//!
//! We sample a DGP with a known per-unit CATE `τ(X)` that varies with the first
//! covariate, fit [`crate::forest::causal_forest::CausalForest`], and measure the
//! Precision in Estimation of Heterogeneous Effects (PEHE) of its predictions via
//! [`crate::metrics::causal_metrics::pehe`]. We further confirm the forest beats a
//! constant-ATE baseline (which is the right comparison for *heterogeneous*
//! effects) and that its predictions correlate positively with the truth.

use crate::forest::causal_forest::CausalForest;
use crate::handle::LcgRng;
use crate::metrics::causal_metrics::pehe;
use crate::verification::synthetic::HeteroEffectData;

/// Result of a forest PEHE evaluation.
pub struct ForestPeheReport {
    /// PEHE = sqrt(mean((τ̂ − τ)²)) of the forest predictions.
    pub forest_pehe: f32,
    /// PEHE of the constant-ATE baseline (predict the average effect everywhere).
    pub baseline_pehe: f32,
    /// Pearson correlation between predicted and true CATE.
    pub cate_correlation: f32,
    /// Mean predicted effect (should be close to the true ATE).
    pub mean_prediction: f32,
    pub ate_true: f32,
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let ma = a.iter().sum::<f32>() / n as f32;
    let mb = b.iter().sum::<f32>() / n as f32;
    let mut num = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..n {
        num += (a[i] - ma) * (b[i] - mb);
        va += (a[i] - ma).powi(2);
        vb += (b[i] - mb).powi(2);
    }
    let den = (va.sqrt() * vb.sqrt()) + 1e-12;
    (num / den).clamp(-1.0, 1.0)
}

/// Fit a causal forest on `dgp` and compute its PEHE report.
pub fn evaluate(
    dgp: &HeteroEffectData,
    n_trees: usize,
    min_samples: usize,
    seed: u64,
) -> crate::error::CausalResult<ForestPeheReport> {
    let mut rng = LcgRng::new(seed);
    let mut forest = CausalForest::new(n_trees, dgp.d, min_samples, &mut rng);
    forest.fit(&dgp.x, &dgp.t, &dgp.y, dgp.n)?;
    let preds = forest.predict(&dgp.x, dgp.n)?;
    let forest_pehe = pehe(&preds, &dgp.cate_true);
    // Constant-ATE baseline.
    let baseline: Vec<f32> = vec![dgp.ate_true; dgp.n];
    let baseline_pehe = pehe(&baseline, &dgp.cate_true);
    let cate_correlation = pearson(&preds, &dgp.cate_true);
    let mean_prediction = preds.iter().sum::<f32>() / dgp.n as f32;
    Ok(ForestPeheReport {
        forest_pehe,
        baseline_pehe,
        cate_correlation,
        mean_prediction,
        ate_true: dgp.ate_true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verification::synthetic::hetero_effect_data;

    #[test]
    fn forest_beats_constant_baseline_on_heterogeneous_effects() {
        // Strong heterogeneity (slope 1.5): a forest that learns the structure
        // should achieve lower PEHE than predicting a single average effect.
        let mut rng = LcgRng::new(2_024);
        let dgp = hetero_effect_data(1200, 4, 2.0, 1.5, 0.3, &mut rng);
        let rep = evaluate(&dgp, 60, 8, 7).expect("forest evaluate");
        assert!(
            rep.forest_pehe < rep.baseline_pehe,
            "forest PEHE {} not below baseline {}",
            rep.forest_pehe,
            rep.baseline_pehe
        );
        // Predicted CATE should be positively correlated with the truth.
        assert!(
            rep.cate_correlation > 0.2,
            "weak CATE correlation: {}",
            rep.cate_correlation
        );
    }

    #[test]
    fn forest_mean_prediction_tracks_ate() {
        // Averaged over units the forest's CATE predictions should land near the
        // true ATE (the forest is honest and treatment is randomized).
        let mut rng = LcgRng::new(55);
        let dgp = hetero_effect_data(1500, 3, 1.0, 1.0, 0.3, &mut rng);
        let rep = evaluate(&dgp, 80, 8, 13).expect("forest evaluate");
        assert!(
            (rep.mean_prediction - rep.ate_true).abs() < 0.6,
            "mean prediction {} far from ATE {}",
            rep.mean_prediction,
            rep.ate_true
        );
    }

    #[test]
    fn pehe_is_small_for_low_noise_strong_signal() {
        // With low noise and a strong, learnable effect the absolute PEHE should
        // be a modest fraction of the effect's spread.
        let mut rng = LcgRng::new(909);
        let dgp = hetero_effect_data(1500, 3, 2.0, 1.0, 0.2, &mut rng);
        let rep = evaluate(&dgp, 80, 8, 21).expect("forest evaluate");
        // Spread of the true CATE is |slope| * sd(X0) ~ 1.0; require PEHE below it.
        assert!(
            rep.forest_pehe < 1.0,
            "PEHE {} too large for strong signal",
            rep.forest_pehe
        );
    }
}
