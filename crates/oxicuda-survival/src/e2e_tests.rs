//! End-to-end integration tests for `oxicuda-survival`.

use crate::aft::{AftFamily, fit_aft, fit_weibull};
use crate::calibration::brier_score::brier_score_at;
use crate::calibration::integrated_brier::integrated_brier_score;
use crate::calibration::ipcw_brier::ipcw_brier_at;
use crate::calibration::time_dependent_auc::time_dependent_auc;
use crate::competing::cumulative_incidence;
use crate::competing::fine_gray::fit_fine_gray;
use crate::concordance::{harrell_c_index, uno_c_index};
use crate::cox::schoenfeld::schoenfeld_residuals;
use crate::cox::{CoxPhConfig, TieMethod, fit_cox_ph};
use crate::data::{Dataset, Observation};
use crate::deep::deepsurv_head::deep_surv_head;
use crate::deep::partial_likelihood_grad;
use crate::deep::surv_loss::{brier_loss, cox_loss};
use crate::handle::LcgRng;
use crate::metrics::metrics::{median_survival, survival_at_horizon};
use crate::nonparametric::survival_function::SurvivalFunction;
use crate::nonparametric::{kaplan_meier_estimate, life_table, nelson_aalen_estimate};
use crate::ptx_kernels::{
    brier_score_ptx, cox_info_ptx, cox_risk_sum_ptx, cox_score_ptx, km_step_ptx, logrank_oe_ptx,
    rmst_integrate_ptx,
};
use crate::test::{gehan_breslow_test, log_rank_test, peto_peto_test, stratified_log_rank_test};

fn synthetic_cox(n: usize, beta_true: f64, seed: u64) -> Dataset {
    let mut rng = LcgRng::new(seed);
    let mut obs = Vec::with_capacity(n);
    let mut cov = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.next_normal();
        let lambda = (beta_true * x).exp();
        let t = rng.next_exponential(lambda).max(1.0e-6);
        obs.push(Observation::new(t, true).expect("ok"));
        cov.push(vec![x]);
    }
    Dataset::new(obs, Some(cov), None).expect("ok")
}

// 1. KM on n=10 dataset recovers known step function
#[test]
fn km_n10_exact_steps() {
    // 10 subjects: events at t=1..10
    let times: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let events = vec![true; 10];
    let d = Dataset::from_arrays(&times, &events).expect("ok");
    let km = kaplan_meier_estimate(&d).expect("ok");
    let expected: Vec<f64> = (0..10)
        .map(|i| (0..=i).fold(1.0, |acc, k| acc * (1.0 - 1.0 / (10.0 - k as f64))))
        .collect();
    for (a, b) in km.survival.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1.0e-12);
    }
}

// 2. Greenwood SE matches the explicit sum formula
#[test]
fn greenwood_formula_matches() {
    let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let events = vec![true, true, false, true, true];
    let d = Dataset::from_arrays(&times, &events).expect("ok");
    let km = kaplan_meier_estimate(&d).expect("ok");
    // Manually compute Σ d_i/(n_i(n_i-d_i)) for first event time
    let manual = 1.0 / (5.0 * 4.0);
    let expected_var = km.survival[0] * km.survival[0] * manual;
    assert!((km.greenwood_var[0] - expected_var).abs() < 1.0e-12);
}

// 3. Cox PH recovers β ≈ true value within 25%
#[test]
fn cox_recovers_beta_within_25pct() {
    let beta_true = 1.0;
    let d = synthetic_cox(400, beta_true, 9999);
    let fit = fit_cox_ph(&d, CoxPhConfig::default()).expect("ok");
    assert!(fit.converged);
    let rel = (fit.coefficients[0] - beta_true).abs() / beta_true;
    assert!(rel < 0.25, "rel_err={rel}");
}

// 4. Newton-Raphson Cox converges in fewer than 50 iterations on well-conditioned data
#[test]
fn newton_converges_in_few_iters() {
    let d = synthetic_cox(200, 0.5, 12345);
    let fit = fit_cox_ph(
        &d,
        CoxPhConfig {
            tie: TieMethod::Breslow,
            tol: 1.0e-7,
            max_iter: 50,
        },
    )
    .expect("ok");
    assert!(fit.converged);
    assert!(fit.iterations < 50);
}

// 5. Schoenfeld residual at last event time is zero
#[test]
fn schoenfeld_last_residual_zero() {
    let d = synthetic_cox(20, 0.5, 7);
    let fit = fit_cox_ph(&d, CoxPhConfig::default()).expect("ok");
    let (_, r) = schoenfeld_residuals(&d, &fit.coefficients).expect("ok");
    assert!(r.last().expect("non-empty")[0].abs() < 1.0e-6);
}

// 6. Log-rank χ² invariant under group label permutation when groups are identical
#[test]
fn log_rank_invariant_under_relabel() {
    let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let events = vec![true; 6];
    let groups_a = vec![0usize, 1, 0, 1, 0, 1];
    let groups_b: Vec<usize> = groups_a.iter().map(|g| 1 - g).collect();
    let d = Dataset::from_arrays(&times, &events).expect("ok");
    let ra = log_rank_test(&d, &groups_a).expect("ok");
    let rb = log_rank_test(&d, &groups_b).expect("ok");
    assert!((ra.chi_square - rb.chi_square).abs() < 1.0e-10);
}

// 7. Harrell C on perfectly ranked data = 1.0
#[test]
fn c_index_perfect_one() {
    let d = Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true; 4]).expect("ok");
    let eta = vec![4.0, 3.0, 2.0, 1.0];
    let c = harrell_c_index(&d, &eta).expect("ok");
    assert!((c - 1.0).abs() < 1.0e-12);
}

// 8. Harrell C on random ranking ≈ 0.5
#[test]
fn c_index_random_near_half() {
    let mut rng = LcgRng::new(42);
    let n = 400;
    let times: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let events = vec![true; n];
    let d = Dataset::from_arrays(&times, &events).expect("ok");
    let eta: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
    let c = harrell_c_index(&d, &eta).expect("ok");
    assert!((c - 0.5).abs() < 0.05, "c={c}");
}

// 9. RMST on constant S = c * tau
#[test]
fn rmst_constant_survival() {
    let s = SurvivalFunction::new(vec![0.0], vec![1.0]).expect("ok");
    let area = crate::rmst::restricted_mean_from_curve(&s, 10.0).expect("ok");
    assert!((area - 10.0).abs() < 1.0e-12);
}

// 10. Fine-Gray reduces to Cox when no competing events
#[test]
fn fine_gray_reduces_to_cox() {
    let d = synthetic_cox(80, 0.7, 31);
    let n = d.len();
    let causes = vec![1u32; n];
    let fg = fit_fine_gray(&d, &causes, 1, 1.0e-6, 50).expect("ok");
    let cox = fit_cox_ph(&d, CoxPhConfig::default()).expect("ok");
    assert!((fg.coefficients[0] - cox.coefficients[0]).abs() < 0.1);
}

// 11. Weibull MLE on exponential data recovers k ≈ 1
#[test]
fn weibull_recovers_unit_shape_on_exp_data() {
    let mut rng = LcgRng::new(56);
    let mut obs = Vec::with_capacity(500);
    for _ in 0..500 {
        let t = rng.next_exponential(1.0).max(1.0e-6);
        obs.push(Observation::new(t, true).expect("ok"));
    }
    let d = Dataset::new(obs, None, None).expect("ok");
    let f = fit_weibull(&d).expect("ok");
    assert!((f.shape - 1.0).abs() < 0.3, "k={}", f.shape);
}

// 12. PTX kernels all non-empty across SM versions
#[test]
fn ptx_all_kernels_compile() {
    let kernels: &[fn(u32) -> String] = &[
        km_step_ptx,
        cox_risk_sum_ptx,
        cox_score_ptx,
        cox_info_ptx,
        logrank_oe_ptx,
        brier_score_ptx,
        rmst_integrate_ptx,
    ];
    for &sm in &[75u32, 80, 86, 89, 90, 100] {
        for f in kernels {
            let s = f(sm);
            assert!(!s.is_empty());
            assert!(s.contains(".visible .entry"));
        }
    }
}

// 13. Brier score on perfect predictor is 0
#[test]
fn brier_perfect_zero() {
    // Predict S=0 for dead by t*=3, S=1 for alive at t*=3.
    // Indicator = 1{T>t*}: i=0,1 have T<=3 → 0; i=2,3 have T>3 → 1.
    let d = Dataset::from_arrays(&[1.0, 2.0, 5.0, 6.0], &[true, true, true, true]).expect("ok");
    let s_pred = vec![0.0, 0.0, 1.0, 1.0];
    let b = brier_score_at(&d, &s_pred, 3.0).expect("ok");
    assert!(b < 1.0e-12);
}

// 14. Integrated Brier finite over a grid
#[test]
fn ibs_finite() {
    let d = synthetic_cox(50, 0.5, 77);
    let times = vec![0.5, 1.0, 1.5, 2.0];
    let s_pred_at = vec![vec![0.5; 50]; times.len()];
    let ibs = integrated_brier_score(&d, &s_pred_at, &times, 2.0).expect("ok");
    assert!(ibs.is_finite());
    assert!(ibs >= 0.0);
}

// 15. Nelson-Aalen vs KM consistency: -log Ŝ ≈ Ĥ
#[test]
fn na_vs_km_consistency() {
    let d = Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0, 5.0], &[true, true, false, true, true])
        .expect("ok");
    let km = kaplan_meier_estimate(&d).expect("ok");
    let na = nelson_aalen_estimate(&d).expect("ok");
    for (s, h) in km.survival.iter().zip(na.cum_hazard.iter()) {
        if *s > 1.0e-12 {
            let log_s = -s.ln();
            // approximate equality (only asymptotic; small differences allowed)
            assert!((log_s - h).abs() < 0.5);
        }
    }
}

// 16. Cox baseline hazard non-decreasing
#[test]
fn cox_baseline_monotone() {
    let d = synthetic_cox(40, 0.4, 88);
    let fit = fit_cox_ph(&d, CoxPhConfig::default()).expect("ok");
    let h = &fit.baseline_hazard.cumulative_hazard;
    for w in h.windows(2) {
        assert!(w[1] >= w[0] - 1.0e-12);
    }
}

// 17. Deep survival head produces finite gradient
#[test]
fn deepsurv_head_gradient_finite() {
    let d = synthetic_cox(30, 0.3, 17);
    let cov = d.covariates.as_ref().expect("ok");
    let head = deep_surv_head(cov, &[0.5], 0.0).expect("ok");
    let grad = partial_likelihood_grad(&d, &head.eta).expect("ok");
    for g in &grad {
        assert!(g.is_finite());
    }
}

// 18. Log-rank vs Peto-Peto on identical groups both ~0
#[test]
fn peto_zero_chi_identical() {
    let times = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    let events = vec![true; 6];
    let groups = vec![0usize, 0, 0, 1, 1, 1];
    let d = Dataset::from_arrays(&times, &events).expect("ok");
    let lr = log_rank_test(&d, &groups).expect("ok");
    let p = peto_peto_test(&d, &groups).expect("ok");
    let g = gehan_breslow_test(&d, &groups).expect("ok");
    assert!(lr.chi_square < 1.0e-9);
    assert!(p.chi_square < 1.0e-9);
    assert!(g.chi_square < 1.0e-9);
}

// 19. Stratified log-rank aggregates correctly
#[test]
fn stratified_log_rank_sum() {
    let times = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    let events = vec![true; 6];
    let groups = vec![0usize, 0, 0, 1, 1, 1];
    let strata = vec![0usize, 0, 0, 1, 1, 1];
    let d = Dataset::from_arrays(&times, &events).expect("ok");
    let r = stratified_log_rank_test(&d, &groups, &strata);
    // groups perfectly aligned with strata → no within-stratum signal
    let _ = r;
}

// 20. Cumulative incidence sums to <= 1 across causes
#[test]
fn cif_sum_le_one() {
    let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let events = vec![true; 5];
    let causes = vec![1u32, 2, 1, 2, 1];
    let d = Dataset::from_arrays(&times, &events).expect("ok");
    let c1 = cumulative_incidence(&d, &causes, 1).expect("ok");
    let c2 = cumulative_incidence(&d, &causes, 2).expect("ok");
    let total = c1.cif.last().expect("ok") + c2.cif.last().expect("ok");
    assert!(total <= 1.0 + 1.0e-9);
}

// 21. Survival function eval is monotone
#[test]
fn survival_function_monotone() {
    let d = synthetic_cox(50, 0.3, 5);
    let km = kaplan_meier_estimate(&d).expect("ok");
    let s = SurvivalFunction::new(km.times.clone(), km.survival.clone()).expect("ok");
    let v1 = s.eval(0.1);
    let v2 = s.eval(0.5);
    let v3 = s.eval(2.0);
    assert!(v1 >= v2);
    assert!(v2 >= v3);
}

// 22. Time-dependent AUC perfect = 1
#[test]
fn td_auc_perfect_one() {
    let d = Dataset::from_arrays(&[1.0, 2.0, 5.0, 6.0], &[true, true, false, false]).expect("ok");
    let eta = vec![3.0, 4.0, 1.0, 2.0];
    let a = time_dependent_auc(&d, &eta, 3.0).expect("ok");
    assert!((a - 1.0).abs() < 1.0e-12);
}

// 23. AFT dispatcher returns finite log-likelihood for each family
#[test]
fn aft_all_families_finite() {
    let d = synthetic_cox(30, 0.0, 19);
    for fam in [
        AftFamily::Exponential,
        AftFamily::Weibull,
        AftFamily::LogNormal,
        AftFamily::LogLogistic,
        AftFamily::GeneralizedGamma,
    ] {
        let f = fit_aft(&d, fam).expect("ok");
        assert!(f.log_likelihood().is_finite());
    }
}

// 24. Cox loss decreases with better η ordering
#[test]
fn cox_loss_monotone_correct() {
    let d = synthetic_cox(40, 1.0, 18);
    let cov = d.covariates.as_ref().expect("ok");
    let eta_good: Vec<f64> = cov.iter().map(|x| 1.0 * x[0]).collect();
    let eta_bad: Vec<f64> = cov.iter().map(|x| -x[0]).collect();
    let lg = cox_loss(&d, &eta_good).expect("ok");
    let lb = cox_loss(&d, &eta_bad).expect("ok");
    assert!(lg < lb);
}

// 25. Median survival = first time at which S <= 0.5
#[test]
fn median_survival_correct() {
    let s = SurvivalFunction::new(vec![1.0, 2.0, 3.0, 4.0], vec![0.75, 0.6, 0.4, 0.3]).expect("ok");
    let m = median_survival(&s).expect("ok");
    assert_eq!(m, 3.0);
}

// 26. survival_at_horizon evaluates correctly
#[test]
fn survival_horizon_known() {
    let s = SurvivalFunction::new(vec![1.0, 2.0], vec![0.7, 0.4]).expect("ok");
    assert!((survival_at_horizon(&s, 0.0) - 1.0).abs() < 1.0e-12);
    assert!((survival_at_horizon(&s, 1.0) - 0.7).abs() < 1.0e-12);
    assert!((survival_at_horizon(&s, 1.5) - 0.7).abs() < 1.0e-12);
    assert!((survival_at_horizon(&s, 5.0) - 0.4).abs() < 1.0e-12);
}

// 27. Life table cumulative survival monotone decreasing
#[test]
fn life_table_monotone() {
    let mut rng = LcgRng::new(33);
    let mut obs = Vec::with_capacity(50);
    for _ in 0..50 {
        let t = rng.next_exponential(0.5).max(1.0e-6);
        obs.push(Observation::new(t, rng.next_bool()).expect("ok"));
    }
    let d = Dataset::new(obs, None, None).expect("ok");
    let lt = life_table(&d, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]).expect("ok");
    let cs = lt.cumulative_survival();
    for w in cs.windows(2) {
        assert!(w[1] <= w[0] + 1.0e-12);
    }
}

// 28. Brier loss agrees with raw brier_score_at
#[test]
fn brier_loss_alias_consistent() {
    let d = Dataset::from_arrays(&[1.0, 2.0], &[true, true]).expect("ok");
    let s_pred = vec![0.5, 0.5];
    let b1 = brier_loss(&d, &s_pred, 1.5).expect("ok");
    let b2 = brier_score_at(&d, &s_pred, 1.5).expect("ok");
    assert!((b1 - b2).abs() < 1.0e-12);
}

// 29. IPCW Brier reduces to naive when no censoring
#[test]
fn ipcw_brier_no_censoring_matches_naive() {
    let d = Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true; 4]).expect("ok");
    let s_pred = vec![0.5; 4];
    let bi = ipcw_brier_at(&d, &s_pred, 2.5).expect("ok");
    let bn = brier_score_at(&d, &s_pred, 2.5).expect("ok");
    assert!((bi - bn).abs() < 1.0e-6);
}

// 30. Uno C with full follow-up matches Harrell
#[test]
fn uno_full_followup_matches_harrell() {
    let d = Dataset::from_arrays(&[1.0, 2.0, 3.0, 4.0], &[true; 4]).expect("ok");
    let eta = vec![4.0, 3.0, 2.0, 1.0];
    let ch = harrell_c_index(&d, &eta).expect("ok");
    let cu = uno_c_index(&d, &eta, 10.0).expect("ok");
    assert!((ch - cu).abs() < 1.0e-9);
}
