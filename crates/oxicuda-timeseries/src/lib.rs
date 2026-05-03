//! `oxicuda-timeseries` — Time-series forecasting primitives for OxiCUDA.
//!
//! Pure-Rust implementations of modern time-series forecasting architectures,
//! designed for CPU simulation and PTX kernel generation for GPU execution.
//!
//! All tensors use **time-major `[T, C]`** layout (channels last) throughout.
//!
//! # Modules
//!
//! ```text
//! oxicuda-timeseries
//! ├── decomp/       — Series decomposition (MovingAvg, SeriesDecomp)
//! ├── norm/         — Normalisation (RevIN, InstanceNorm1d)
//! ├── patch/        — Patch extraction (PatchEmbed1d)
//! ├── patchtst/     — PatchTST encoder (Nie et al. 2023)
//! ├── head/         — Forecasting heads (LinearHead, MlpHead)
//! ├── tcn/          — Temporal Convolutional Network (Bai et al. 2018)
//! ├── nhits/        — NHiTS hierarchical forecaster (Challu et al. 2022)
//! ├── itransformer/ — iTransformer (Liu et al. 2024)
//! ├── timesnet/     — TimesNet 2-D variation model (Wu et al. 2023)
//! ├── error         — TsError / TsResult
//! ├── handle        — TsHandle / LcgRng / SmVersion
//! └── ptx_kernels   — GPU PTX kernel strings
//! ```

pub mod decomp;
pub mod error;
pub mod handle;
pub mod head;
pub mod itransformer;
pub mod nhits;
pub mod norm;
pub mod patch;
pub mod patchtst;
pub mod ptx_kernels;
pub mod tcn;
pub mod timesnet;

/// Convenience re-exports for common time-series types.
pub mod prelude {
    pub use crate::decomp::{DecompResult, MovingAvg, SeriesDecomp};
    pub use crate::error::{TsError, TsResult};
    pub use crate::handle::{LcgRng, SmVersion, TsHandle};
    pub use crate::head::{LinearHead, MlpHead};
    pub use crate::itransformer::{ITransformer, ITransformerConfig, InvertedBlock};
    pub use crate::nhits::{MultiRateSampler, NHits, NHitsBlock, NHitsConfig};
    pub use crate::norm::{InstanceNorm1d, RevIn};
    pub use crate::patch::PatchEmbed1d;
    pub use crate::patchtst::{PatchTst, PatchTstConfig};
    pub use crate::tcn::{TcnBlock, TcnConfig, TcnEncoder};
    pub use crate::timesnet::{TimesBlock, TimesNet, TimesNetConfig};
}

// ─── E2E Integration Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod e2e_tests {
    use super::*;
    use handle::LcgRng;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ── Handle & RNG ─────────────────────────────────────────────────────────

    #[test]
    fn e2e_handle_default() {
        use handle::{SmVersion, TsHandle};
        let h = TsHandle::default_handle();
        assert_eq!(h.device(), 0);
        assert_eq!(h.sm_version(), SmVersion(80));
    }

    #[test]
    fn e2e_lcg_rng_reproducibility() {
        let mut a = LcgRng::new(123);
        let mut b = LcgRng::new(123);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    // ── PTX kernels ──────────────────────────────────────────────────────────

    #[test]
    fn e2e_ptx_kernels_all_sm_versions() {
        #[allow(clippy::type_complexity)]
        let kernels: &[(&str, fn(u32) -> String)] = &[
            ("moving_average", ptx_kernels::moving_average_ptx),
            ("patch_embed_1d", ptx_kernels::patch_embed_1d_ptx),
            (
                "causal_temporal_conv",
                ptx_kernels::causal_temporal_conv_ptx,
            ),
            ("auto_correlation", ptx_kernels::auto_correlation_ptx),
            ("revin_normalize", ptx_kernels::revin_normalize_ptx),
            ("multirate_pool", ptx_kernels::multirate_pool_ptx),
            ("period_detect", ptx_kernels::period_detect_ptx),
        ];
        for &sm in &[75u32, 80, 86, 90, 100, 120] {
            for &(name, kernel_fn) in kernels {
                let ptx = kernel_fn(sm);
                assert!(
                    ptx.contains(&format!("sm_{sm}")),
                    "kernel={name} sm={sm}: missing .target"
                );
            }
        }
    }

    // ── Decomposition ────────────────────────────────────────────────────────

    #[test]
    fn e2e_series_decomp_trend_plus_seasonal() {
        let decomp = decomp::SeriesDecomp::new(25).expect("ok");
        let t = 100;
        let c = 8;
        let features: Vec<f32> = (0..t * c)
            .map(|i| ((i as f32) * 0.05).sin() + (i as f32) * 0.002)
            .collect();
        let res = decomp.forward(&features, t, c).expect("ok");
        // trend + seasonal must reconstruct the original
        for (i, (&orig, (&tr, &se))) in features
            .iter()
            .zip(res.trend.iter().zip(res.seasonal.iter()))
            .enumerate()
        {
            assert!((orig - (tr + se)).abs() < 1e-5, "idx={i}");
        }
    }

    // ── RevIN ────────────────────────────────────────────────────────────────

    #[test]
    fn e2e_revin_zero_mean_unit_std() {
        let rv = norm::RevIn::new(8).expect("ok");
        let mut rng = make_rng();
        let mut x = vec![0.0_f32; 64 * 8];
        rng.fill_normal(&mut x);
        let (out, _, _) = rv.forward(&x, 64).expect("ok");
        for ci in 0..8 {
            let s: f32 = (0..64).map(|ti| out[ti * 8 + ci]).sum::<f32>() / 64.0;
            assert!(s.abs() < 1e-5, "variate {ci} mean after RevIN = {s}");
        }
    }

    #[test]
    fn e2e_revin_inverse_roundtrip() {
        let rv = norm::RevIn::new(4).expect("ok");
        let x: Vec<f32> = (0..20 * 4).map(|i| i as f32 * 0.1).collect();
        let (normed, mean, std) = rv.forward(&x, 20).expect("ok");
        let recovered = rv.inverse(&normed, 20, &mean, &std).expect("ok");
        for (a, b) in x.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    // ── PatchEmbed ───────────────────────────────────────────────────────────

    #[test]
    fn e2e_patch_embed_shape() {
        let mut rng = make_rng();
        let pe = patch::PatchEmbed1d::new(16, 8, 64, &mut rng).expect("ok");
        let t = 96;
        let c = 4;
        let x = vec![0.5_f32; t * c];
        let out = pe.forward_mv(&x, t, c).expect("ok");
        let np = pe.num_patches(t); // (96-16)/8+1 = 11
        assert_eq!(out.len(), c * np * 64);
    }

    // ── TCN ──────────────────────────────────────────────────────────────────

    #[test]
    fn e2e_tcn_tiny_shape() {
        let mut rng = make_rng();
        let cfg = tcn::TcnConfig::tiny();
        let enc = tcn::TcnEncoder::new(cfg.clone(), &mut rng).expect("ok");
        let t = 100;
        let x = vec![0.1_f32; t * cfg.in_channels];
        let out = enc.forward(&x, t).expect("ok");
        assert_eq!(out.len(), t * cfg.out_channels);
    }

    #[test]
    fn e2e_tcn_output_finite() {
        let mut rng = make_rng();
        let cfg = tcn::TcnConfig::tiny();
        let enc = tcn::TcnEncoder::new(cfg.clone(), &mut rng).expect("ok");
        let t = 64;
        let mut x = vec![0.0_f32; t * cfg.in_channels];
        rng.fill_normal(&mut x);
        let out = enc.forward(&x, t).expect("ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "TCN produced non-finite output"
        );
    }

    // ── NHiTS ────────────────────────────────────────────────────────────────

    #[test]
    fn e2e_nhits_output_shape() {
        let mut rng = make_rng();
        let t = 96;
        let c = 4;
        let horizon = 24;
        let cfg = nhits::NHitsConfig::tiny(t, c, horizon);
        let model = nhits::NHits::new(cfg, &mut rng).expect("ok");
        let x = vec![0.1_f32; t * c];
        let out = model.forward(&x).expect("ok");
        assert_eq!(out.len(), horizon * c);
    }

    #[test]
    fn e2e_nhits_output_finite() {
        let mut rng = make_rng();
        let t = 96;
        let c = 3;
        let horizon = 12;
        let cfg = nhits::NHitsConfig::tiny(t, c, horizon);
        let model = nhits::NHits::new(cfg, &mut rng).expect("ok");
        let mut x = vec![0.0_f32; t * c];
        rng.fill_normal(&mut x);
        let out = model.forward(&x).expect("ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "NHiTS produced non-finite output"
        );
    }

    // ── PatchTST ─────────────────────────────────────────────────────────────

    #[test]
    fn e2e_patchtst_output_shape() {
        let mut rng = make_rng();
        let t = 96;
        let c = 4;
        let horizon = 24;
        let cfg = patchtst::PatchTstConfig::tiny(c, t, horizon);
        let model = patchtst::PatchTst::new(cfg, &mut rng).expect("ok");
        let x = vec![0.1_f32; t * c];
        let out = model.forward(&x).expect("ok");
        assert_eq!(out.len(), horizon * c);
    }

    #[test]
    fn e2e_patchtst_output_finite() {
        let mut rng = make_rng();
        let t = 96;
        let c = 3;
        let horizon = 12;
        let cfg = patchtst::PatchTstConfig::tiny(c, t, horizon);
        let model = patchtst::PatchTst::new(cfg, &mut rng).expect("ok");
        let mut x = vec![0.0_f32; t * c];
        rng.fill_normal(&mut x);
        let out = model.forward(&x).expect("ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "PatchTST produced non-finite output"
        );
    }

    // ── TimesNet ─────────────────────────────────────────────────────────────

    #[test]
    fn e2e_timesnet_output_shape() {
        let mut rng = make_rng();
        let t = 64;
        let c = 4;
        let horizon = 16;
        let cfg = timesnet::TimesNetConfig::tiny(c, t, horizon);
        let model = timesnet::TimesNet::new(cfg, &mut rng).expect("ok");
        let x = vec![0.1_f32; t * c];
        let out = model.forward(&x).expect("ok");
        assert_eq!(out.len(), horizon * c);
    }

    #[test]
    fn e2e_timesnet_output_finite() {
        let mut rng = make_rng();
        let t = 64;
        let c = 3;
        let horizon = 8;
        let cfg = timesnet::TimesNetConfig::tiny(c, t, horizon);
        let model = timesnet::TimesNet::new(cfg, &mut rng).expect("ok");
        let mut x = vec![0.0_f32; t * c];
        rng.fill_normal(&mut x);
        let out = model.forward(&x).expect("ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "TimesNet produced non-finite output"
        );
    }

    // ── iTransformer ─────────────────────────────────────────────────────────

    #[test]
    fn e2e_itransformer_output_shape() {
        let mut rng = make_rng();
        let t = 96;
        let c = 4;
        let horizon = 24;
        let cfg = itransformer::ITransformerConfig::tiny(c, t, horizon);
        let model = itransformer::ITransformer::new(cfg, &mut rng).expect("ok");
        let x = vec![0.1_f32; t * c];
        let out = model.forward(&x).expect("ok");
        assert_eq!(out.len(), horizon * c);
    }

    #[test]
    fn e2e_itransformer_output_finite() {
        let mut rng = make_rng();
        let t = 96;
        let c = 3;
        let horizon = 12;
        let cfg = itransformer::ITransformerConfig::tiny(c, t, horizon);
        let model = itransformer::ITransformer::new(cfg, &mut rng).expect("ok");
        let mut x = vec![0.0_f32; t * c];
        rng.fill_normal(&mut x);
        let out = model.forward(&x).expect("ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "iTransformer produced non-finite output"
        );
    }

    // ── Forecasting heads ────────────────────────────────────────────────────

    #[test]
    fn e2e_linear_head_ts_shape() {
        let mut rng = make_rng();
        let t = 50;
        let c = 6;
        let horizon = 10;
        let head = head::LinearHead::new(t, horizon, &mut rng).expect("ok");
        let x: Vec<f32> = (0..t * c).map(|i| i as f32 * 0.1).collect();
        let out = head.forward_ts(&x, t, c).expect("ok");
        assert_eq!(out.len(), horizon * c);
    }

    #[test]
    fn e2e_mlp_head_ts_shape() {
        let mut rng = make_rng();
        let t = 50;
        let c = 6;
        let horizon = 10;
        let head = head::MlpHead::new(t, 32, horizon, &mut rng).expect("ok");
        let x: Vec<f32> = (0..t * c).map(|i| i as f32 * 0.1).collect();
        let out = head.forward_ts(&x, t, c).expect("ok");
        assert_eq!(out.len(), horizon * c);
    }

    // ── Instance norm ────────────────────────────────────────────────────────

    #[test]
    fn e2e_instance_norm_zero_mean() {
        let ln = norm::InstanceNorm1d::new(4).expect("ok");
        let mut x: Vec<f32> = (0..20 * 4).map(|i| i as f32 * 0.3 - 3.0).collect();
        ln.forward(&mut x, 20).expect("ok");
        for ci in 0..4 {
            let s: f32 = (0..20).map(|ti| x[ti * 4 + ci]).sum::<f32>() / 20.0;
            assert!(s.abs() < 1e-4, "channel {ci} mean={s}");
        }
    }
}
