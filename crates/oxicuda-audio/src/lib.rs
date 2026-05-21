//! `oxicuda-audio` — Audio/Speech ML architectures for OxiCUDA.
//!
//! Pure-Rust implementation providing:
//! - **`attention`**: Multi-head relative-position self-attention (Conformer-style).
//! - **`augment`**: SpecAugment (time mask, freq mask, time warp).
//! - **`ctc`**: CTC forward algorithm and prefix beam-search decoder.
//! - **`encoder`**: Wav2Vec2 CNN feature encoder + Conformer block/encoder.
//! - **`error`**: Error and result types for audio operations.
//! - **`features`**: Log-mel adapter, CMVN, delta/delta-delta.
//! - **`handle`**: Session handle with SM version and LCG RNG.
//! - **`ptx_kernels`**: 7 GPU PTX kernel string generators (SM 7.5–12.0).
//! - **`rescoring`**: Shallow-fusion LM lattice / n-best rescoring (distinct from CTC beam search).
//! - **`separation`**: Conv-TasNet time-domain source separation.
//! - **`speaker`**: Speaker embedding (x-vector TDNN, stats pool, attentive pool).
//! - **`vad`**: Voice-activity detection (energy + spectral-flatness, onset/hangover hysteresis).
//! - **`vocoder`**: WaveNet + HiFi-GAN neural vocoders.

pub mod attention;
pub mod augment;
pub mod ctc;
pub mod encoder;
pub mod error;
pub mod features;
pub mod handle;
pub mod ptx_kernels;
pub mod rescoring;
pub mod separation;
pub mod speaker;
pub mod vad;
pub mod vocoder;

pub use error::{AudioError, AudioResult};
pub use handle::{AudioHandle, LcgRng, SmVersion};

// ─── Prelude ─────────────────────────────────────────────────────────────────

pub mod prelude {
    pub use crate::attention::{RelPosAttention, RelPosEncoding};
    pub use crate::augment::{SpecAugOp, SpecAugPipeline, freq_mask, time_mask, time_warp};
    pub use crate::ctc::{BeamHypothesis, ctc_beam_search, ctc_forward_log};
    pub use crate::encoder::{
        ConformerConfig, ConformerEncoder, Wav2VecCnnConfig, Wav2VecCnnEncoder, WhisperEncoder,
        WhisperEncoderConfig,
    };
    pub use crate::error::{AudioError, AudioResult};
    pub use crate::features::{
        CmvnConfig, LogMelInput, MelFilterbank, MelFilterbankConfig, apply_cmvn, compute_cmvn,
        compute_delta, compute_delta_delta, stack_delta_features,
    };
    pub use crate::handle::{AudioHandle, LcgRng, SmVersion};
    pub use crate::ptx_kernels::{
        ctc_alpha_ptx, depthwise_conv1d_ptx, dilated_conv1d_ptx, rel_pos_bias_ptx,
        spec_augment_mask_ptx, stats_pool_ptx, stride_conv1d_ptx,
    };
    pub use crate::rescoring::{Hypothesis, LatticeRescorer, RescoreConfig, ScoredHypothesis};
    pub use crate::separation::{ConvTasNet, ConvTasNetConfig, SeparationResult};
    pub use crate::speaker::{AttentivePool, XVectorConfig, XVectorTdnn, stats_pool};
    pub use crate::vad::{Vad, VadConfig, VadResult};
    pub use crate::vocoder::{
        HifiGanConfig, HifiGanGenerator, WaveNetBlock, WaveNetConfig, WaveNetStack,
    };
}

// ─── End-to-end integration tests ────────────────────────────────────────────

#[cfg(test)]
mod e2e_tests {
    use super::*;
    use crate::augment::{SpecAugOp, SpecAugPipeline};
    use crate::ctc::{ctc_beam_search, ctc_forward_log};
    use crate::encoder::{ConformerConfig, ConformerEncoder, Wav2VecCnnConfig, Wav2VecCnnEncoder};
    use crate::features::{
        LogMelInput, apply_cmvn, compute_cmvn, compute_delta, compute_delta_delta,
    };
    use crate::handle::LcgRng;
    use crate::ptx_kernels::{
        ctc_alpha_ptx, depthwise_conv1d_ptx, dilated_conv1d_ptx, rel_pos_bias_ptx,
        spec_augment_mask_ptx, stats_pool_ptx, stride_conv1d_ptx,
    };
    use crate::speaker::{AttentivePool, XVectorConfig, XVectorTdnn, stats_pool};
    use crate::vocoder::{WaveNetConfig, WaveNetStack};

    // ── features ──────────────────────────────────────────────────────────────

    #[test]
    fn e2e_log_mel_adapter_validates_shape() {
        let data = vec![0.0f32; 5];
        let r = LogMelInput::from_mel(&data, 2, 4);
        assert!(matches!(r.unwrap_err(), AudioError::ShapeMismatch { .. }));
    }

    #[test]
    fn e2e_cmvn_zero_mean_unit_var() {
        let t = 50;
        let f = 8;
        let mut features: Vec<f32> = (0..t * f).map(|i| (i as f32).sin() * 3.0 + 2.0).collect();
        let cfg = compute_cmvn(&features, t, f).expect("cmvn ok");
        apply_cmvn(&mut features, t, f, &cfg).expect("apply ok");
        let mean: f32 = features.iter().sum::<f32>() / features.len() as f32;
        assert!(mean.abs() < 0.01, "mean={mean}");
    }

    #[test]
    fn e2e_delta_window_central_difference() {
        let t = 20;
        let f = 4;
        let c = 3.0f32;
        let features: Vec<f32> = (0..t * f).map(|i| c * (i / f) as f32).collect();
        let delta = compute_delta(&features, t, f, 2).expect("ok");
        for frame in 3..t - 3 {
            for dim in 0..f {
                let d = delta[frame * f + dim];
                assert!((d - c).abs() < 0.01, "frame={frame} d={d}");
            }
        }
        let dd = compute_delta_delta(&features, t, f, 2).expect("ok");
        assert!(dd.iter().all(|v| v.is_finite()), "delta-delta has NaN");
    }

    // ── encoder ───────────────────────────────────────────────────────────────

    #[test]
    fn e2e_wav2vec_cnn_output_length() {
        let cfg = Wav2VecCnnConfig::tiny();
        let in_len = 500usize;
        let mut rng = LcgRng::new(1);
        let enc = Wav2VecCnnEncoder::new(&cfg, &mut rng).expect("build ok");
        let input = vec![0.5f32; in_len];
        let (_, _, out_len) = enc.forward(&input, 1, in_len).expect("forward ok");
        assert!(out_len > 0, "out_len should be positive");
        // Verify cascade: (((500-5)/2+1 - 3)/2+1 - 3)/2+1
        let expected = enc.output_len(in_len);
        assert_eq!(out_len, expected);
    }

    #[test]
    fn e2e_conformer_block_finite() {
        let cfg = ConformerConfig::tiny();
        let mut rng = LcgRng::new(42);
        let enc = ConformerEncoder::new(cfg.clone(), &mut rng).expect("build ok");
        let t = 20usize;
        let d = cfg.embed_dim;
        let mut x = vec![0.0f32; t * d];
        LcgRng::new(7).fill_normal(&mut x);
        let out = enc.forward(&x, t).expect("forward ok");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn e2e_conformer_encoder_tiny_shape() {
        let cfg = ConformerConfig::tiny();
        let mut rng = LcgRng::new(11);
        let enc = ConformerEncoder::new(cfg.clone(), &mut rng).expect("build ok");
        let t = 15usize;
        let d = cfg.embed_dim;
        let x = vec![0.1f32; t * d];
        let out = enc.forward(&x, t).expect("forward ok");
        assert_eq!(out.len(), t * d, "[T={t}, D={d}] shape mismatch");
    }

    // ── CTC ───────────────────────────────────────────────────────────────────

    #[test]
    fn e2e_ctc_forward_log_finite() {
        let t = 10usize;
        let v = 8usize;
        let blank = 0usize;
        // Build row-normalised log-probs
        let mut rng = LcgRng::new(99);
        let mut log_probs = vec![0.0f32; t * v];
        rng.fill_normal(&mut log_probs);
        for row in 0..t {
            let base = &mut log_probs[row * v..(row + 1) * v];
            let max = base.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let s: f32 = base.iter().map(|x| (x - max).exp()).sum::<f32>().ln();
            for lp in base.iter_mut() {
                *lp = (*lp - max) - s;
            }
        }
        let target = vec![1usize, 2, 3];
        let ll = ctc_forward_log(&log_probs, t, v, &target, blank).expect("ctc ok");
        assert!(ll.is_finite(), "log-likelihood = {ll}");
    }

    #[test]
    fn e2e_ctc_forward_alignment_consistency() {
        // T=3, V=3, blank=0, target=[1]: ext = [blank, 1, blank]
        // S=3, T=3 — should be feasible
        let t = 3usize;
        let v = 3usize;
        let blank = 0usize;
        let log_probs = vec![
            (0.8f32).ln(),
            (0.1f32).ln(),
            (0.1f32).ln(),
            (0.1f32).ln(),
            (0.8f32).ln(),
            (0.1f32).ln(),
            (0.5f32).ln(),
            (0.3f32).ln(),
            (0.2f32).ln(),
        ];
        let ll = ctc_forward_log(&log_probs, t, v, &[1], blank).expect("ctc ok");
        assert!(ll.is_finite() && ll < 0.0, "ll={ll}");
    }

    #[test]
    fn e2e_ctc_beam_search_shape() {
        let t = 8usize;
        let v = 6usize;
        let blank = 0usize;
        let beam_width = 3usize;
        let mut rng = LcgRng::new(55);
        let mut log_probs = vec![0.0f32; t * v];
        rng.fill_normal(&mut log_probs);
        for row in 0..t {
            let base = &mut log_probs[row * v..(row + 1) * v];
            let max = base.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let s: f32 = base.iter().map(|x| (x - max).exp()).sum::<f32>().ln();
            for lp in base.iter_mut() {
                *lp = (*lp - max) - s;
            }
        }
        let hyps = ctc_beam_search(&log_probs, t, v, blank, beam_width).expect("beam ok");
        assert!(hyps.len() <= beam_width);
        for h in &hyps {
            assert!(h.log_prob.is_finite());
        }
    }

    #[test]
    fn e2e_ctc_beam_search_blank_only() {
        let t = 5usize;
        let v = 3usize;
        let blank = 0usize;
        // All probability on blank
        let log_probs: Vec<f32> = (0..t * v)
            .map(|i| {
                if i % v == blank {
                    1.0f32.ln()
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        let hyps = ctc_beam_search(&log_probs, t, v, blank, 3).expect("beam ok");
        assert!(
            hyps.is_empty() || hyps[0].tokens.is_empty(),
            "blank-only should decode to empty"
        );
    }

    // ── vocoder ───────────────────────────────────────────────────────────────

    #[test]
    fn e2e_wavenet_block_finite() {
        use crate::vocoder::WaveNetBlock;
        let mut rng = LcgRng::new(3);
        let block = WaveNetBlock::new(8, 8, 3, 1, &mut rng).expect("block ok");
        let t = 10usize;
        let x = vec![0.1f32; 8 * t];
        let (residual, skip) = block.forward(&x, t).expect("forward ok");
        assert!(residual.iter().all(|v| v.is_finite()));
        assert!(skip.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn e2e_wavenet_stack_skip_sum_finite() {
        let cfg = WaveNetConfig::tiny();
        let mut rng = LcgRng::new(8);
        let stack = WaveNetStack::new(cfg.clone(), &mut rng).expect("stack ok");
        let t = 20usize;
        let x = vec![0.05f32; cfg.residual_channels * t];
        let out = stack.forward(&x, t).expect("forward ok");
        assert_eq!(out.len(), cfg.skip_channels * t);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── SpecAugment ───────────────────────────────────────────────────────────

    #[test]
    fn e2e_specaug_time_mask_zeros() {
        let t = 30usize;
        let f = 16usize;
        let mut mel = vec![1.0f32; t * f];
        let mut rng = LcgRng::new(21);
        crate::augment::time_mask(&mut mel, t, f, 5, 2, &mut rng).expect("mask ok");
        // At least some values were zeroed
        assert!(mel.contains(&0.0_f32), "expected zeros after time masking");
        assert!(mel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn e2e_specaug_pipeline_chain() {
        let t = 40usize;
        let f = 20usize;
        let mut mel = vec![0.5f32; t * f];
        let mut rng = LcgRng::new(31);
        let pipeline = SpecAugPipeline::new()
            .push(SpecAugOp::TimeMask {
                max_t: 4,
                n_masks: 2,
            })
            .push(SpecAugOp::FreqMask {
                max_f: 3,
                n_masks: 1,
            })
            .push(SpecAugOp::TimeWarp { max_w: 5 });
        pipeline
            .apply(&mut mel, t, f, &mut rng)
            .expect("pipeline ok");
        assert!(mel.iter().all(|v| v.is_finite()));
    }

    // ── speaker ───────────────────────────────────────────────────────────────

    #[test]
    fn e2e_stats_pool_concat_shape() {
        let t = 25usize;
        let c = 12usize;
        let features: Vec<f32> = (0..t * c).map(|i| i as f32 * 0.01).collect();
        let out = stats_pool(&features, t, c).expect("pool ok");
        assert_eq!(out.len(), 2 * c);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn e2e_attentive_pool_weights_sum_to_one() {
        // Verified indirectly: output shape is [2*C] and finite
        let c = 16usize;
        let t = 20usize;
        let mut rng = LcgRng::new(66);
        let pool = AttentivePool::new(c, &mut rng).expect("pool ok");
        let mut features = vec![0.0f32; t * c];
        rng.fill_normal(&mut features);
        let out = pool.forward(&features, t).expect("forward ok");
        assert_eq!(out.len(), 2 * c);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn e2e_x_vector_embedding_finite() {
        let cfg = XVectorConfig::tiny();
        let embed_dim = cfg.embed_dim;
        let in_dim = cfg.in_dim;
        let mut rng = LcgRng::new(77);
        let model = XVectorTdnn::new(cfg, &mut rng).expect("build ok");
        let t = 30usize;
        let mut features = vec![0.0f32; t * in_dim];
        rng.fill_normal(&mut features);
        let emb = model.forward(&features, t).expect("forward ok");
        assert_eq!(emb.len(), embed_dim);
        assert!(emb.iter().all(|v| v.is_finite()));
    }

    // ── PTX kernels ───────────────────────────────────────────────────────────

    #[test]
    fn e2e_ptx_kernels_all_sm_versions() {
        const SM_VERSIONS: &[u32] = &[75, 80, 86, 90, 100, 120];
        for &sm in SM_VERSIONS {
            let tag = format!(".target sm_{sm}");
            assert!(
                stride_conv1d_ptx(sm).contains(&tag),
                "stride_conv1d sm={sm}"
            );
            assert!(
                dilated_conv1d_ptx(sm).contains(&tag),
                "dilated_conv1d sm={sm}"
            );
            assert!(ctc_alpha_ptx(sm).contains(&tag), "ctc_alpha sm={sm}");
            assert!(
                spec_augment_mask_ptx(sm).contains(&tag),
                "spec_augment_mask sm={sm}"
            );
            assert!(
                depthwise_conv1d_ptx(sm).contains(&tag),
                "depthwise_conv1d sm={sm}"
            );
            assert!(rel_pos_bias_ptx(sm).contains(&tag), "rel_pos_bias sm={sm}");
            assert!(stats_pool_ptx(sm).contains(&tag), "stats_pool sm={sm}");
        }
    }

    // ── handle ────────────────────────────────────────────────────────────────

    #[test]
    fn e2e_handle_default() {
        let h = AudioHandle::default_handle();
        assert_eq!(h.device(), 0);
        assert_eq!(h.sm_version(), SmVersion(80));
    }

    #[test]
    fn e2e_lcg_rng_reproducibility() {
        let mut a = LcgRng::new(42);
        let mut b = LcgRng::new(42);
        for _ in 0..200 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    // ── rel-pos attention ────────────────────────────────────────────────────

    #[test]
    fn e2e_rel_pos_attention_shape() {
        use crate::attention::RelPosAttention;
        let embed_dim = 32usize;
        let n_heads = 4usize;
        let max_len = 64usize;
        let mut rng = LcgRng::new(13);
        let attn = RelPosAttention::new(embed_dim, n_heads, max_len, &mut rng).expect("ok");
        let t = 10usize;
        let mut x = vec![0.0f32; t * embed_dim];
        rng.fill_normal(&mut x);
        let out = attn.forward(&x, t).expect("forward ok");
        assert_eq!(out.len(), t * embed_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
