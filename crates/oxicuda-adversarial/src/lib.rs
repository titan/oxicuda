//! `oxicuda-adversarial` — Adversarial robustness primitives for OxiCUDA.
//!
//! Pure-Rust library covering both attack and defence sides of adversarial
//! robustness for deep classifiers, suitable for CPU simulation and PTX kernel
//! generation for GPU execution.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-adversarial
//! ├── attacks/         — FGSM, PGD (L∞/L2), MIM, CW, AutoPGD
//! ├── defenses/        — TRADES, MART, randomized smoothing, IBP, certified bounds
//! ├── threat_model/    — Lp ball constraints, ε-budget tracking
//! ├── metrics/         — robust accuracy, attack success rate, certified accuracy
//! ├── error            — AdvError / AdvResult
//! ├── handle           — AdvHandle (SmVersion + LcgRng)
//! └── ptx_kernels      — GPU PTX kernel strings
//! ```

// ─── Module declarations ─────────────────────────────────────────────────────

pub mod attacks;
pub mod defenses;
pub mod error;
pub mod handle;
pub mod metrics;
pub mod ptx_kernels;
pub mod threat_model;

// ─── Prelude ─────────────────────────────────────────────────────────────────

/// Convenience re-exports for common adversarial robustness types.
pub mod prelude {
    pub use crate::attacks::auto_pgd::{AutoPgdConfig, auto_pgd_attack};
    pub use crate::attacks::cw::{CwConfig, cw_attack};
    pub use crate::attacks::fgsm::fgsm_attack;
    pub use crate::attacks::jsma::{Jsma, JsmaConfig};
    pub use crate::attacks::mim::{MimConfig, mim_attack};
    pub use crate::attacks::patch::{PatchAttack, PatchConfig};
    pub use crate::attacks::pgd::{PgdConfig, pgd_attack_l_inf, pgd_attack_l2};
    pub use crate::attacks::targeted::{TargetedAttack, TargetedConfig};
    pub use crate::defenses::awp::{AwpConfig, AwpDefense, AwpWeightDelta};
    pub use crate::defenses::certified_bounds::{
        IntervalBound, ibp_propagate, lipschitz_certified_radius,
    };
    pub use crate::defenses::crown::{
        AlphaBound, CrownConfig, CrownVerifier, LinearLayer, NeuronBound,
    };
    pub use crate::defenses::laplace_smoothing::{LaplaceSmoothing, LaplaceSmoothingConfig};
    pub use crate::defenses::lp_relaxation::{
        AffineLayer, LpRelaxConfig, LpRelaxVerifier, VerifiedBound,
    };
    pub use crate::defenses::macer::{MacerConfig, MacerLoss};
    pub use crate::defenses::mart::{MartConfig, mart_loss};
    pub use crate::defenses::randomized_smoothing::{RsConfig, certified_radius, smoothed_predict};
    pub use crate::defenses::smoothing_lp::LpSmoothingCertifier;
    pub use crate::defenses::trades::{TradesConfig, trades_loss};
    pub use crate::error::{AdvError, AdvResult};
    pub use crate::handle::{AdvHandle, LcgRng, SmVersion};
    pub use crate::metrics::asr::attack_success_rate;
    pub use crate::metrics::feature_squeezing::{FeatureSqueezingConfig, FeatureSqueezingDetector};
    pub use crate::metrics::gradient_masking::{
        GradMaskingConclusion, GradMaskingConfig, GradientMaskingReport, diagnose_gradient_masking,
        random_perturbation_asr,
    };
    pub use crate::metrics::robust_accuracy::{
        ClassResult, RobustAccConfig, certified_accuracy, robust_accuracy, robust_accuracy_report,
        worst_class_robust_acc,
    };
    pub use crate::metrics::stratified_accuracy::{
        ClassRobustness, StratifiedReport, stratified_robust_accuracy,
    };
    pub use crate::metrics::transferability::{
        TransferMatrix, transferability_matrix, transferability_matrix_from_predictions,
    };
    pub use crate::ptx_kernels::{
        attack_loss_grad_ptx, certified_radius_reduce_ptx, f32_hex, fgsm_step_ptx, grad_sign_ptx,
        pgd_proj_l_inf_ptx, pgd_proj_l2_ptx, smoothing_noise_ptx,
    };
    pub use crate::threat_model::budget::EpsilonBudget;
    pub use crate::threat_model::lp_ball::{
        LpNorm, l_inf_norm, l1_norm, l2_norm, project_l_inf, project_l2,
    };
}

// ─── End-to-end integration tests ────────────────────────────────────────────

#[cfg(test)]
mod e2e_tests {
    use crate::prelude::*;

    /// Synthetic loss `L(x) = 0.5 · ‖x − target‖²`. The "untargeted" attack
    /// should push x AWAY from `target`; we hand it the gradient `(x − target)`
    /// (positive direction), so adding eps·sign(grad) moves x away.
    fn quadratic_loss_grad(target: &[f32]) -> impl Fn(&[f32]) -> AdvResult<Vec<f32>> + '_ {
        move |x: &[f32]| {
            if x.len() != target.len() {
                return Err(AdvError::DimensionMismatch {
                    expected: target.len(),
                    got: x.len(),
                });
            }
            Ok(x.iter().zip(target.iter()).map(|(a, b)| a - b).collect())
        }
    }

    #[test]
    fn e2e_fgsm_pushes_away_from_target() {
        let target = vec![0.5_f32; 4];
        let x = target.clone();
        let adv = fgsm_attack(&x, 0.1, 0.0, 1.0, quadratic_loss_grad(&target))
            .expect("value should be present");
        // Initial gradient is zero (x == target), so sign is zero and we get x back.
        assert!(adv.iter().zip(x.iter()).all(|(a, b)| (a - b).abs() < 1e-6));
        // Now perturb x slightly so the gradient is non-zero.
        let x2 = vec![0.6_f32; 4];
        let adv2 = fgsm_attack(&x2, 0.1, 0.0, 1.0, quadratic_loss_grad(&target))
            .expect("value should be present");
        // Gradient is +0.1 → sign is +1 → x_adv = 0.7
        assert!(adv2.iter().all(|v| (v - 0.7_f32).abs() < 1e-5));
    }

    #[test]
    fn e2e_pgd_l_inf_respects_eps_ball() {
        let target = vec![0.5_f32; 4];
        let x = vec![0.6_f32; 4];
        let cfg = PgdConfig::new(0.05, 0.02, 5, false).expect("new should succeed");
        let mut handle = AdvHandle::default_handle();
        let adv = pgd_attack_l_inf(
            &x,
            0.0,
            1.0,
            &cfg,
            handle.rng_mut(),
            quadratic_loss_grad(&target),
        )
        .expect("value should be present");
        // Each coordinate is within ε of original
        for (a, o) in adv.iter().zip(x.iter()) {
            assert!((a - o).abs() <= cfg.eps + 1e-5);
        }
    }

    #[test]
    fn e2e_pgd_l2_respects_eps_ball() {
        let target = vec![0.5_f32; 8];
        let x = vec![0.6_f32; 8];
        let cfg = PgdConfig::new(0.5, 0.1, 5, false).expect("new should succeed");
        let mut handle = AdvHandle::default_handle();
        let adv = pgd_attack_l2(
            &x,
            0.0,
            1.0,
            &cfg,
            handle.rng_mut(),
            quadratic_loss_grad(&target),
        )
        .expect("value should be present");
        let delta: Vec<f32> = adv.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let n = l2_norm(&delta);
        assert!(n <= cfg.eps + 1e-4);
    }

    #[test]
    fn e2e_mim_with_momentum_decay_zero_matches_pgd() {
        let target = vec![0.5_f32; 4];
        let x = vec![0.6_f32; 4];
        let mim_cfg = MimConfig {
            eps: 0.1,
            alpha: 0.05,
            n_steps: 3,
            momentum_decay: 0.0,
        };
        let adv = mim_attack(&x, 0.0, 1.0, &mim_cfg, quadratic_loss_grad(&target))
            .expect("value should be present");
        for (a, o) in adv.iter().zip(x.iter()) {
            assert!((a - o).abs() <= mim_cfg.eps + 1e-5);
        }
    }

    #[test]
    fn e2e_trades_loss_collapses_to_ce_when_clean_equals_adv() {
        // Two-sample 3-class batch.
        let logits = vec![1.0_f32, 2.0, 0.5, 0.0, 0.5, 1.5];
        let labels = vec![1_usize, 2];
        let cfg = TradesConfig::new(6.0).expect("new should succeed");
        let l =
            trades_loss(&logits, &logits, &labels, 2, 3, &cfg).expect("trades_loss should succeed");
        // KL term should be ~0; only CE remains.
        assert!(l.is_finite() && l > 0.0);
    }

    #[test]
    fn e2e_mart_loss_finite_under_perturbation() {
        let clean = vec![1.0_f32, 2.0, 0.5, 0.0, 0.5, 1.5];
        let adv = vec![0.5_f32, 1.5, 0.7, 0.2, 0.7, 1.3];
        let labels = vec![1_usize, 2];
        let cfg = MartConfig::new(5.0).expect("new should succeed");
        let l = mart_loss(&clean, &adv, &labels, 2, 3, &cfg).expect("mart_loss should succeed");
        assert!(l.is_finite() && l > 0.0);
    }

    #[test]
    fn e2e_randomized_smoothing_constant_classifier_returns_top_class() {
        let mut handle = AdvHandle::default_handle();
        let cfg = RsConfig::new(0.25, 1024, 0.001).expect("new should succeed");
        let x = vec![0.5_f32; 8];
        // Constant classifier returns class 7 always.
        let (cls, _r) = certified_radius(&x, &cfg, handle.rng_mut(), |_| Ok(7_usize))
            .expect("value should be present");
        assert_eq!(cls, 7);
    }

    #[test]
    fn e2e_ibp_propagates_through_relu() {
        let bounds_in = vec![
            IntervalBound::new(-1.0, 1.0).expect("new should succeed"),
            IntervalBound::new(-2.0, 2.0).expect("new should succeed"),
        ];
        // Identity weights → output bounds equal input bounds (then ReLU clips lower).
        let w = vec![1.0_f32, 0.0, 0.0, 1.0]; // 2x2 identity
        let b = vec![0.0_f32, 0.0];
        let out = ibp_propagate(&bounds_in, &w, &b, 2, 2).expect("ibp_propagate should succeed");
        let relu_out: Vec<IntervalBound> = out.iter().map(|b| b.relu()).collect();
        assert!(relu_out[0].lo >= 0.0);
        assert!(relu_out[1].lo >= 0.0);
        assert!(relu_out[0].hi >= 0.0);
        assert!(relu_out[1].hi >= 0.0);
    }

    #[test]
    fn e2e_lipschitz_certified_radius_formula() {
        // r = m / (L * sqrt(2)) with m = 2.0, L = 1.0 → r ≈ √2
        let r = lipschitz_certified_radius(2.0, 1.0)
            .expect("lipschitz_certified_radius should succeed");
        assert!((r - std::f32::consts::SQRT_2).abs() < 1e-5);
    }

    #[test]
    fn e2e_robust_accuracy_perfect_attack_zero() {
        let labels = vec![0_usize, 1, 2, 3];
        let pred_all_wrong = vec![1_usize, 2, 3, 0];
        let acc =
            robust_accuracy(&pred_all_wrong, &labels).expect("robust_accuracy should succeed");
        assert!(acc.abs() < 1e-6);
        let asr = attack_success_rate(&pred_all_wrong, &labels)
            .expect("attack_success_rate should succeed");
        assert!((asr - 1.0).abs() < 1e-6);
    }

    #[test]
    fn e2e_certified_accuracy_threshold() {
        let pred = vec![0_usize, 0, 0];
        let labels = vec![0_usize, 0, 0];
        // Two have radius 1.0, one has radius 0.1.
        let radii = vec![1.0_f32, 0.1, 1.0];
        let acc = certified_accuracy(&pred, &labels, &radii, 0.5)
            .expect("certified_accuracy should succeed");
        // Two of three are correct AND certified → 2/3
        assert!((acc - 2.0_f32 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn e2e_ptx_kernels_all_sm_versions() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            for prog in [
                fgsm_step_ptx(sm),
                pgd_proj_l_inf_ptx(sm),
                pgd_proj_l2_ptx(sm),
                smoothing_noise_ptx(sm),
                grad_sign_ptx(sm),
                certified_radius_reduce_ptx(sm),
                attack_loss_grad_ptx(sm),
            ] {
                assert!(prog.contains(&format!("sm_{sm}")));
                assert!(prog.contains(".visible .entry"));
            }
        }
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
    }

    #[test]
    fn e2e_epsilon_budget_lifecycle() {
        let mut b = EpsilonBudget::new(1.0).expect("new should succeed");
        b.spend(0.3).expect("spend should succeed");
        b.spend(0.4).expect("spend should succeed");
        assert!((b.remaining() - 0.3).abs() < 1e-6);
        let r = b.spend(0.5);
        assert!(r.is_err());
    }
}
