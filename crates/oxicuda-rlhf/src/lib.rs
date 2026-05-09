//! `oxicuda-rlhf` — RLHF and alignment algorithm primitives for OxiCUDA.
//!
//! Pure-Rust implementation of preference-based alignment algorithms including DPO, IPO,
//! KTO, ORPO, SimPO, SFT, PPO-RLHF, and reward modelling, suitable for CPU simulation
//! and PTX kernel generation for GPU execution.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-rlhf
//! ├── dpo/         — DPO, IPO, KTO direct preference optimisation losses
//! ├── orpo/        — ORPO and SimPO odds-ratio preference optimisation
//! ├── ppo_rlhf/    — PPO-based RLHF (rollout, KL control, policy step)
//! ├── preference/  — Bradley-Terry reward loss, preference pair batching
//! ├── reward/      — Reward model and reward normalizer
//! ├── sft/         — Supervised fine-tuning loss (masked token cross-entropy)
//! ├── metrics/     — Alignment metrics (win rate, KL, perplexity, reward gap)
//! ├── handle       — LcgRng, RlhfHandle, SmVersion
//! ├── error        — RlhfError / RlhfResult
//! ├── ptx_kernels  — GPU PTX kernel strings (7 kernels × 6 SM versions)
//! └── prelude      — Convenience re-exports of common types
//! ```

pub mod dpo;
pub mod error;
pub mod handle;
pub mod metrics;
pub mod orpo;
pub mod ppo_rlhf;
pub mod preference;
pub mod ptx_kernels;
pub mod reward;
pub mod sft;

pub mod prelude {
    pub use crate::dpo::dpo::{DpoConfig, dpo_log_ratio, dpo_loss, dpo_loss_per_pair};
    pub use crate::dpo::ipo::{IpoConfig, ipo_loss};
    pub use crate::dpo::kto::{KtoConfig, kto_loss};
    pub use crate::error::{RlhfError, RlhfResult};
    pub use crate::handle::{LcgRng, RlhfHandle, SmVersion};
    pub use crate::metrics::alignment::{
        AlignmentMetrics, compute_alignment_metrics, kl_from_ref, perplexity, reward_gap, win_rate,
    };
    pub use crate::orpo::orpo::{OrpoConfig, log_odds, orpo_loss};
    pub use crate::orpo::simpo::{SimpoConfig, simpo_loss};
    pub use crate::ppo_rlhf::kl_control::{KlController, kl_divergence_from_logps};
    pub use crate::ppo_rlhf::ppo_step::{RlhfPpoConfig, rlhf_ppo_loss};
    pub use crate::ppo_rlhf::rollout::RlhfRollout;
    pub use crate::preference::bradley_terry::{RewardHead, bt_reward_loss};
    pub use crate::preference::pair::{PairBatch, PreferencePair};
    pub use crate::ptx_kernels::{
        bt_reward_loss_ptx, dpo_loss_ptx, f32_hex, ipo_loss_ptx, kto_loss_ptx, orpo_odds_ptx,
        rlhf_kl_ptx, sft_mask_ptx,
    };
    pub use crate::reward::model::RewardModel;
    pub use crate::reward::normalize::RewardNormalizer;
    pub use crate::sft::loss::{masked_token_ce, sft_loss};
}

#[cfg(test)]
mod e2e_tests {
    use super::prelude::*;

    #[test]
    fn e2e_bt_loss_zero_equal_rewards() {
        let chosen = [1.0_f32, 2.0, 0.5];
        let rejected = [1.0_f32, 2.0, 0.5];
        let loss = bt_reward_loss(&chosen, &rejected).unwrap();
        let expected = -(0.5_f32.ln());
        assert!(
            (loss - expected).abs() < 1e-4,
            "loss={loss}, expected={expected}"
        );
    }

    #[test]
    fn e2e_bt_loss_decreases_with_gap() {
        let chosen_small = [1.1_f32];
        let rejected_small = [1.0_f32];
        let chosen_large = [3.0_f32];
        let rejected_large = [0.0_f32];
        let loss_small = bt_reward_loss(&chosen_small, &rejected_small).unwrap();
        let loss_large = bt_reward_loss(&chosen_large, &rejected_large).unwrap();
        assert!(
            loss_large < loss_small,
            "BT loss should decrease with larger reward gap: small={loss_small}, large={loss_large}"
        );
    }

    #[test]
    fn e2e_dpo_loss_finite() {
        let batch = PairBatch::new(
            vec![-1.0_f32, -2.0, -1.5],
            vec![-2.0_f32, -3.0, -2.5],
            vec![-1.1_f32, -2.1, -1.6],
            vec![-2.1_f32, -3.1, -2.6],
        )
        .unwrap();
        let cfg = DpoConfig { beta: 0.1 };
        let loss = dpo_loss(&batch, &cfg).unwrap();
        assert!(loss.is_finite(), "DPO loss must be finite, got {loss}");
    }

    #[test]
    fn e2e_dpo_lower_for_aligned_pairs() {
        let aligned_batch = PairBatch::new(
            vec![-0.5_f32],
            vec![-3.0_f32],
            vec![-1.0_f32],
            vec![-1.0_f32],
        )
        .unwrap();
        let unaligned_batch = PairBatch::new(
            vec![-3.0_f32],
            vec![-0.5_f32],
            vec![-1.0_f32],
            vec![-1.0_f32],
        )
        .unwrap();
        let cfg = DpoConfig { beta: 0.5 };
        let loss_aligned = dpo_loss(&aligned_batch, &cfg).unwrap();
        let loss_unaligned = dpo_loss(&unaligned_batch, &cfg).unwrap();
        assert!(
            loss_aligned < loss_unaligned,
            "Aligned DPO loss {loss_aligned} should be lower than unaligned {loss_unaligned}"
        );
    }

    #[test]
    fn e2e_ipo_loss_finite() {
        let batch = PairBatch::new(
            vec![-1.0_f32, -2.0],
            vec![-2.5_f32, -3.0],
            vec![-1.2_f32, -2.2],
            vec![-2.7_f32, -3.2],
        )
        .unwrap();
        let cfg = IpoConfig { beta: 0.1 };
        let loss = ipo_loss(&batch, &cfg).unwrap();
        assert!(loss.is_finite(), "IPO loss must be finite, got {loss}");
        assert!(loss >= 0.0, "IPO loss must be non-negative, got {loss}");
    }

    #[test]
    fn e2e_kto_loss_nonneg() {
        let desirable = [0.5_f32, 1.0, 0.3];
        let undesirable = [-0.5_f32, -1.0, -0.3];
        let cfg = KtoConfig {
            beta: 0.1,
            lambda_d: 1.0,
            lambda_u: 1.0,
        };
        let loss = kto_loss(&desirable, &undesirable, &cfg).unwrap();
        assert!(loss.is_finite(), "KTO loss must be finite");
        assert!(loss >= 0.0, "KTO loss must be non-negative, got {loss}");
    }

    #[test]
    fn e2e_orpo_structure() {
        let chosen_logps = [-1.0_f32, -1.5];
        let rejected_logps = [-2.0_f32, -3.0];
        let sft_loss_val = 2.0_f32;
        let cfg = OrpoConfig { lambda: 0.5 };
        let loss = orpo_loss(&chosen_logps, &rejected_logps, sft_loss_val, &cfg).unwrap();
        assert!(loss.is_finite(), "ORPO loss must be finite, got {loss}");
        assert!(
            loss >= sft_loss_val,
            "ORPO loss {loss} should be >= sft_loss {sft_loss_val} when lambda>0 and penalty>0"
        );
    }

    #[test]
    fn e2e_simpo_length_normalized() {
        let chosen_sum = [-10.0_f32];
        let rejected_sum = [-5.0_f32];
        let chosen_len = [10_usize];
        let rejected_len = [2_usize];
        let cfg = SimpoConfig {
            beta: 2.0,
            gamma: 0.5,
        };
        let loss =
            simpo_loss(&chosen_sum, &rejected_sum, &chosen_len, &rejected_len, &cfg).unwrap();
        assert!(loss.is_finite(), "SimPO loss must be finite, got {loss}");
    }

    #[test]
    fn e2e_sft_loss_correct_prediction() {
        let n_vocab = 4_usize;
        let n_tokens = 1_usize;
        let label = 2_u32;
        let mut logits = vec![0.0_f32; n_tokens * n_vocab];
        logits[label as usize] = 100.0;
        let labels = [label];
        let mask = [1_u8];
        let loss = sft_loss(&logits, &labels, &mask, n_vocab).unwrap();
        assert!(
            loss < 0.01,
            "SFT loss should be near 0 for strongly correct prediction, got {loss}"
        );
    }

    #[test]
    fn e2e_kl_zero_at_ref() {
        let log_probs = [-1.0_f32, -2.0, -0.5, -1.5];
        let kl = kl_from_ref(&log_probs, &log_probs).unwrap();
        assert!(
            kl.abs() < 1e-5,
            "KL from ref should be 0 when lp == ref_lp, got {kl}"
        );
    }

    #[test]
    fn e2e_reward_normalizer_unit_variance() {
        let mut norm = RewardNormalizer::new();
        let values: Vec<f32> = (0..100).map(|i| i as f32).collect();
        for &v in &values {
            norm.update(v);
        }
        let normalized: Vec<f32> = values.iter().map(|&v| norm.normalize(v).unwrap()).collect();
        let mean: f32 = normalized.iter().sum::<f32>() / normalized.len() as f32;
        let variance: f32 = normalized
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f32>()
            / normalized.len() as f32;
        assert!(
            mean.abs() < 0.01,
            "Normalized mean should be near 0, got {mean}"
        );
        assert!(
            (variance - 1.0).abs() < 0.05,
            "Normalized variance should be near 1, got {variance}"
        );
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn e2e_ptx_kernels_all_sm_versions() {
        let sm_versions = [75_u32, 80, 86, 90, 100, 120];
        let kernel_fns: &[(&str, fn(u32) -> String)] = &[
            ("bt_reward_loss_kernel", bt_reward_loss_ptx),
            ("dpo_loss_kernel", dpo_loss_ptx),
            ("ipo_loss_kernel", ipo_loss_ptx),
            ("kto_loss_kernel", kto_loss_ptx),
            ("orpo_odds_kernel", orpo_odds_ptx),
            ("rlhf_kl_kernel", rlhf_kl_ptx),
            ("sft_mask_kernel", sft_mask_ptx),
        ];
        for sm in sm_versions {
            for (kernel_name, gen_fn) in kernel_fns {
                let ptx = gen_fn(sm);
                assert!(
                    ptx.contains(&format!("sm_{sm}")),
                    "PTX for {kernel_name} sm={sm} missing sm target"
                );
                assert!(
                    ptx.contains(".version"),
                    "PTX for {kernel_name} sm={sm} missing .version"
                );
                assert!(
                    ptx.contains(".visible .entry"),
                    "PTX for {kernel_name} sm={sm} missing .visible .entry"
                );
                assert!(
                    ptx.contains(kernel_name),
                    "PTX for {kernel_name} sm={sm} missing kernel name"
                );
            }
        }
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
    }
}
