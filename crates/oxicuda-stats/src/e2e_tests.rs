//! End-to-end integration tests for `oxicuda-stats`.

use crate::chi_squared::chi2_independence::chi2_independence;
use crate::chi_squared::fisher_exact::fisher_exact_2x2;
use crate::chi_squared::mcnemar::mcnemar;
use crate::ci::normal_ci::normal_ci;
use crate::ci::proportion_ci::wilson_ci;
use crate::ci::t_ci::t_ci;
use crate::correlation::kendall_tau::kendall_tau;
use crate::correlation::pearson::pearson_r;
use crate::correlation::spearman::spearman_rho;
use crate::descriptive::quantile::quantile;
use crate::descriptive::summary::{mean, sample_std};
use crate::distributions::normal::Normal;
use crate::distributions::student_t::StudentT;
use crate::goodness_of_fit::jarque_bera::jarque_bera;
use crate::goodness_of_fit::ks::ks_one_sample;
use crate::handle::LcgRng;
use crate::multiple::bh_fdr::bh_fdr;
use crate::multiple::bonferroni::bonferroni;
use crate::nonparametric::kruskal_wallis::kruskal_wallis;
use crate::nonparametric::mann_whitney::mann_whitney_u;
use crate::parametric::anova::one_way_anova;
use crate::parametric::t_test::{one_sample_t, two_sample_t};
use crate::ptx_kernels::{
    bootstrap_resample_ptx, chi2_cell_ptx, histogram_bin_ptx, lr_normal_eq_ptx, mean_var_ptx,
    permute_labels_ptx, rank_assign_ptx,
};
use crate::regression::linear::ols;
use crate::resampling::bootstrap::bootstrap;
use crate::resampling::jackknife::jackknife;
use crate::special::erf::erf;
use crate::special::gammaln::lgamma;

// 1. erf(0)=0, erf(1)~0.8427, erf(2)~0.9953
#[test]
fn special_erf_known_values() {
    assert!(erf(0.0).abs() < 1e-10);
    assert!((erf(1.0) - 0.842_700_793).abs() < 1e-6);
    assert!((erf(2.0) - 0.995_322_265).abs() < 1e-6);
}

// 2. lgamma(5)=ln(24), lgamma(0.5)=ln(sqrt(pi))
#[test]
fn special_lgamma_known_values() {
    assert!((lgamma(5.0) - 24f64.ln()).abs() < 1e-10);
    assert!((lgamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-10);
}

// 3. Student-t cdf at t=0, df=10 is 0.5
#[test]
fn student_t_cdf_at_zero_half() {
    let dist = StudentT::new(10.0).expect("ok");
    let v = dist.cdf(0.0).expect("ok");
    assert!((v - 0.5).abs() < 1e-12);
}

// 4. Normal cdf monotone increasing
#[test]
fn normal_cdf_monotone() {
    let n = Normal::standard();
    for &(a, b) in &[(-2.0_f64, -1.0_f64), (-1.0, 0.0), (0.0, 1.0), (1.0, 2.0)] {
        assert!(n.cdf(a) < n.cdf(b));
    }
}

// 5. Mean + sample_std on a known dataset
#[test]
fn descriptive_mean_std() {
    let x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    assert!((mean(&x).expect("ok") - 5.0).abs() < 1e-12);
    // sample std (n-1): sqrt(32 / 7) ~ 2.138
    assert!((sample_std(&x).expect("ok") - (32f64 / 7f64).sqrt()).abs() < 1e-10);
}

// 6. One-sample t-test rejects when mu0 is far from mean
#[test]
fn t_test_one_sample_far_mu_rejects() {
    let x: Vec<f64> = (1..=20).map(|v| v as f64).collect();
    let r = one_sample_t(&x, 0.0).expect("ok");
    assert!(r.p_value_two_sided < 1.0e-6);
}

// 7. Two-sample t-test for clearly different groups
#[test]
fn t_test_two_sample_clear_difference() {
    let x1: Vec<f64> = (1..=20).map(|v| v as f64).collect();
    let x2: Vec<f64> = (10..=29).map(|v| v as f64).collect();
    let r = two_sample_t(&x1, &x2).expect("ok");
    assert!(r.p_value_two_sided < 1.0e-3);
}

// 8. ANOVA on 3 groups matches scipy
#[test]
fn anova_three_groups_f_eq_12() {
    let g1: &[f64] = &[1.0, 2.0, 3.0];
    let g2: &[f64] = &[3.0, 4.0, 5.0];
    let g3: &[f64] = &[5.0, 6.0, 7.0];
    let r = one_way_anova(&[g1, g2, g3]).expect("ok");
    assert!((r.f_statistic - 12.0).abs() < 1e-9);
}

// 9. Mann-Whitney detects shift
#[test]
fn mann_whitney_detects_shift() {
    let x1: Vec<f64> = (1..=15).map(|v| v as f64).collect();
    let x2: Vec<f64> = (5..=19).map(|v| v as f64).collect();
    let r = mann_whitney_u(&x1, &x2).expect("ok");
    assert!(r.p_value_two_sided < 0.1);
}

// 10. Kruskal-Wallis detects distinct groups
#[test]
fn kruskal_wallis_detects_distinct() {
    let g1: &[f64] = &[1.0, 2.0, 3.0];
    let g2: &[f64] = &[10.0, 11.0, 12.0];
    let g3: &[f64] = &[20.0, 21.0, 22.0];
    let r = kruskal_wallis(&[g1, g2, g3]).expect("ok");
    assert!(r.p_value < 0.05);
}

// 11. KS one-sample on standard normal gives small D, moderate p
#[test]
fn ks_one_sample_normal_matches() {
    let mut rng = LcgRng::new(5);
    let x: Vec<f64> = (0..200).map(|_| rng.next_normal()).collect();
    let n = Normal::standard();
    let r = ks_one_sample(&x, |t| n.cdf(t)).expect("ok");
    // Sample is normal -> D should be modest
    assert!(r.d_statistic < 0.15);
}

// 12. Chi-square independence on strong association
#[test]
fn chi2_independence_strong() {
    let obs = [90.0, 10.0, 10.0, 90.0];
    let r = chi2_independence(&obs, 2, 2).expect("ok");
    assert!(r.p_value < 1e-10);
}

// 13. Fisher exact on small clear table
#[test]
fn fisher_exact_clear_table() {
    let r = fisher_exact_2x2(8, 2, 1, 9).expect("ok");
    assert!(r.p_value_two_sided < 0.05);
}

// 14. McNemar test on paired data
#[test]
fn mcnemar_paired_test() {
    let r = mcnemar(30, 10, true).expect("ok");
    assert!(r.p_value < 0.01);
}

// 15. Bonferroni and BH FDR are monotone
#[test]
fn multiple_comparison_monotone() {
    let p = [0.001, 0.01, 0.03, 0.05, 0.5];
    let b = bonferroni(&p).expect("ok");
    let h = bh_fdr(&p).expect("ok");
    // Bonferroni always >= BH for each p
    for (bv, hv) in b.iter().zip(&h) {
        assert!(bv >= hv);
    }
}

// 16. Bootstrap mean CI contains true value
#[test]
fn bootstrap_mean_contains_truth() {
    let mut rng = LcgRng::new(7);
    let data: Vec<f64> = (1..=40).map(|v| v as f64).collect();
    let r = bootstrap(&data, 400, 0.95, mean, &mut rng).expect("ok");
    // True mean of 1..40 = 20.5
    assert!(r.ci_lower <= 20.5 && r.ci_upper >= 20.5);
}

// 17. Pearson + Spearman + Kendall agree in sign on monotone data
#[test]
fn correlations_agree_on_monotone() {
    let x: Vec<f64> = (1..=20).map(|v| v as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| v.powi(2)).collect();
    let p = pearson_r(&x, &y).expect("ok");
    let s = spearman_rho(&x, &y).expect("ok");
    let k = kendall_tau(&x, &y).expect("ok");
    assert!(p.r > 0.95);
    assert!((s.r - 1.0).abs() < 1e-10);
    assert!((k.tau - 1.0).abs() < 1e-10);
}

// 18. Quantile + CI sanity + OLS fits perfectly
#[test]
fn quantile_ci_and_ols() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let med = quantile(&x, 0.5).expect("ok");
    assert!((med - 3.0).abs() < 1e-12);
    let nci = normal_ci(&x, 1.0, 0.95).expect("ok");
    assert!(nci.lower < 3.0 && nci.upper > 3.0);
    let tci = t_ci(&x, 0.95).expect("ok");
    assert!(tci.lower < 3.0 && tci.upper > 3.0);
    let wci = wilson_ci(40, 100, 0.95).expect("ok");
    assert!(wci.lower < 0.4 && wci.upper > 0.4);
    // OLS on perfect line
    let mut design = Vec::new();
    let xs = [1.0f64, 2.0, 3.0, 4.0, 5.0];
    let ys: Vec<f64> = xs.iter().map(|x| 1.0 + 2.0 * x).collect();
    for &x in &xs {
        design.push(1.0);
        design.push(x);
    }
    let m = ols(&design, &ys, 5, 2).expect("ok");
    assert!(m.residual_sum_squares < 1e-18);
    // PTX kernels non-empty for all 6 SM versions x 7 kernels
    type KFn = fn(u32) -> String;
    let kernels: &[(&str, KFn)] = &[
        ("mean_var", mean_var_ptx),
        ("rank_assign", rank_assign_ptx),
        ("histogram_bin", histogram_bin_ptx),
        ("bootstrap_resample", bootstrap_resample_ptx),
        ("permute_labels", permute_labels_ptx),
        ("chi2_cell", chi2_cell_ptx),
        ("lr_normal_eq", lr_normal_eq_ptx),
    ];
    for sm in [75u32, 80, 86, 89, 90, 100] {
        for (name, f) in kernels {
            let s = f(sm);
            assert!(!s.is_empty(), "{name} sm={sm}");
            assert!(s.contains(".visible .entry"));
        }
    }
    // Jackknife runs
    let jk = jackknife(&xs, mean).expect("ok");
    assert!(jk.std_error.is_finite());
    // Jarque-Bera runs
    let mut rng = LcgRng::new(13);
    let z: Vec<f64> = (0..100).map(|_| rng.next_normal()).collect();
    let _ = jarque_bera(&z).expect("ok");
}
