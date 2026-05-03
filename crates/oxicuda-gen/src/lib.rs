//! # oxicuda-gen
//!
//! Generative AI primitives for OxiCUDA: diffusion schedulers, classifier-free
//! guidance, VAE codec, LoRA adapters, and score network building blocks —
//! pure Rust, zero CUDA SDK dependency.

pub mod error;
pub mod guidance;
pub mod handle;
pub mod lora;
pub mod ptx_kernels;
pub mod scheduler;
pub mod score;
pub mod vae;

/// Re-exports of the most commonly used types.
pub mod prelude {
    pub use crate::error::{GenError, GenResult};
    pub use crate::guidance::{
        AdaptiveCfgPolicy, AdaptiveCfgScheduler, CfgConfig, CfgGuidance, PerpNegGuidance,
    };
    pub use crate::handle::{GenHandle, LcgRng, SmVersion};
    pub use crate::lora::{
        LoraConfig, LoraLinear, LoraModel, merge_lora, unmerge_lora, verify_merge_roundtrip,
    };
    pub use crate::ptx_kernels::{
        cfg_combine_ptx, ddpm_step_ptx, f32_hex, flow_velocity_ptx, lora_apply_ptx,
        timestep_embed_ptx, vae_kl_loss_ptx,
    };
    pub use crate::scheduler::{
        BetaSchedule, BetaScheduleType, DdimScheduler, DdpmScheduler, DpmOrder, DpmSolverScheduler,
        FlowMatchingPath, FlowMatchingScheduler,
    };
    pub use crate::score::{
        CrossAttentionBlock, FourierEmbedding, SelfAttentionBlock, SinusoidalEmbedding,
        UNetResBlock,
    };
    pub use crate::vae::{
        Decoder, DecoderConfig, DecoderWeights, Encoder, EncoderConfig, EncoderWeights,
        GaussianLatent, VqCodebook,
    };
}

// ─── Integration tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::guidance::cfg::{CfgConfig, CfgGuidance};
    use crate::handle::LcgRng;
    use crate::lora::adapter::{LoraConfig, LoraLinear};
    use crate::ptx_kernels::{
        cfg_combine_ptx, ddpm_step_ptx, flow_velocity_ptx, lora_apply_ptx, timestep_embed_ptx,
        vae_kl_loss_ptx,
    };
    use crate::scheduler::beta_schedule::BetaSchedule;
    use crate::scheduler::ddim::DdimScheduler;
    use crate::scheduler::ddpm::DdpmScheduler;
    use crate::scheduler::dpm_solver::{DpmOrder, DpmSolverScheduler};
    use crate::scheduler::flow_matching::FlowMatchingScheduler;
    use crate::score::timestep::SinusoidalEmbedding;
    use crate::vae::kl::GaussianLatent;
    use crate::vae::quantize::VqCodebook;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn randn(rng: &mut LcgRng, n: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    #[test]
    fn e2e_ddpm_forward_reverse_consistency() {
        let sched = DdpmScheduler::new(1000).unwrap();
        let mut rng = make_rng();
        let x0 = randn(&mut rng, 64);
        let noise = randn(&mut rng, 64);
        let t = 100;
        let x_t = sched.add_noise(&x0, &noise, t).unwrap();
        assert_eq!(x_t.len(), x0.len());
        let diff: f32 = x0
            .iter()
            .zip(&x_t)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(diff > 1e-4, "x_t should differ from x_0: diff={diff}");
        let step_noise = randn(&mut rng, 64);
        let x_prev = sched.step(&noise, &x_t, t, &step_noise).unwrap();
        assert_eq!(x_prev.len(), 64);
        assert!(x_prev.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn e2e_ddim_deterministic_at_eta_zero() {
        let sched = DdimScheduler::new(1000, 10, 0.0).unwrap();
        let mut rng = make_rng();
        let eps = randn(&mut rng, 32);
        let x_t = randn(&mut rng, 32);
        let noise1 = randn(&mut rng, 32);
        let noise2 = randn(&mut rng, 32);
        let out1 = sched.step(&eps, &x_t, 0, &noise1).unwrap();
        let out2 = sched.step(&eps, &x_t, 0, &noise2).unwrap();
        let max_diff: f32 = out1
            .iter()
            .zip(&out2)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_diff < 1e-5,
            "DDIM eta=0 should be deterministic: diff={max_diff}"
        );
    }

    #[test]
    fn e2e_dpm_solver_step_shape() {
        let sched = DpmSolverScheduler::new(1000, 20, DpmOrder::Second).unwrap();
        let mut rng = make_rng();
        let model_out = randn(&mut rng, 48);
        let x_t = randn(&mut rng, 48);
        let out = sched.step(&model_out, None, &x_t, 0).unwrap();
        assert_eq!(out.len(), 48);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn e2e_flow_matching_boundary_conditions() {
        let sched = FlowMatchingScheduler::new(50);
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let x1 = vec![4.0_f32, 5.0, 6.0];
        let at_zero = sched.interpolate(&x0, &x1, 0.0).unwrap();
        for (&a, &b) in at_zero.iter().zip(&x0) {
            assert!((a - b).abs() < 1e-5, "t=0 should give x0: {a} vs {b}");
        }
        let at_one = sched.interpolate(&x0, &x1, 1.0).unwrap();
        for (&a, &b) in at_one.iter().zip(&x1) {
            assert!((a - b).abs() < 1e-5, "t=1 should give x1: {a} vs {b}");
        }
    }

    #[test]
    fn e2e_cfg_scale_one_is_uncond() {
        let config = CfgConfig::new(1.0).unwrap();
        let guide = CfgGuidance::new(config);
        let cond = vec![2.0_f32, 3.0, 4.0];
        let uncond = vec![1.0_f32, 1.0, 1.0];
        let out = guide.apply(&cond, &uncond).unwrap();
        // scale=1: out = uncond + 1*(cond - uncond) = cond
        for (&o, &c) in out.iter().zip(&cond) {
            assert!((o - c).abs() < 1e-5, "scale=1 should give cond: {o} vs {c}");
        }
    }

    #[test]
    fn e2e_lora_zero_b_is_identity() {
        let config = LoraConfig::new(4, 4.0).unwrap();
        let mut rng = make_rng();
        let lora = LoraLinear::new(8, 16, &config, &mut rng).unwrap();
        let x = randn(&mut rng, 8);
        let base = randn(&mut rng, 16);
        let out = lora.forward(&x, &base, 1).unwrap();
        for (&o, &b) in out.iter().zip(&base) {
            assert!(
                (o - b).abs() < 1e-5,
                "B=0 LoRA should be identity: {o} vs {b}"
            );
        }
    }

    #[test]
    fn e2e_vae_kl_zero_for_standard_normal() {
        let latent = GaussianLatent::standard_normal(128).unwrap();
        let kl = latent.kl_loss().unwrap();
        assert!(kl.abs() < 1e-4, "KL for standard normal should be ~0: {kl}");
    }

    #[test]
    fn e2e_vq_codebook_nearest_lookup() {
        let n_codes = 4;
        let embed_dim = 4;
        let mut embeddings = vec![0.0_f32; n_codes * embed_dim];
        for i in 0..n_codes {
            embeddings[i * embed_dim + i] = 1.0;
        }
        let cb = VqCodebook::from_embeddings(embeddings, n_codes, embed_dim).unwrap();
        let z = vec![0.0_f32, 0.1, 0.9, 0.0];
        let (_, idx) = cb.quantize(&z).unwrap();
        assert_eq!(idx[0], 2, "should map to code 2, got {}", idx[0]);
    }

    #[test]
    fn e2e_sinusoidal_embed_dimensions() {
        let emb = SinusoidalEmbedding::new(128).unwrap();
        let out = emb.embed_timestep(500.0);
        assert_eq!(out.len(), 128);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "embedding should be finite"
        );
        for i in 0..64 {
            let s = out[2 * i];
            let c = out[2 * i + 1];
            assert!((s * s + c * c - 1.0).abs() < 1e-4, "sin²+cos²≠1 at i={i}");
        }
    }

    #[test]
    fn e2e_ptx_kernels_all_sm_versions() {
        let sm_versions: &[u32] = &[75, 80, 86, 90, 100, 120];
        for &sm in sm_versions {
            let target = format!("sm_{sm}");
            for ptx in [
                ddpm_step_ptx(sm),
                cfg_combine_ptx(sm),
                lora_apply_ptx(sm),
                flow_velocity_ptx(sm),
                vae_kl_loss_ptx(sm),
                timestep_embed_ptx(sm),
            ] {
                assert!(ptx.contains(&target), "PTX missing .target {target}");
                assert!(
                    ptx.contains(".address_size 64"),
                    "PTX missing .address_size 64"
                );
            }
        }
    }

    #[test]
    fn e2e_beta_schedule_chain() {
        let sched = BetaSchedule::linear(1000, 0.0001, 0.02).unwrap();
        assert_eq!(sched.num_steps(), 1000);
        assert!(sched.sqrt_alphas_bar()[0] > 0.99);
        assert!(sched.sqrt_alphas_bar()[999] < 0.2);
    }
}
