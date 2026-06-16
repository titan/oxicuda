//! `oxicuda-distill` — Knowledge-distillation primitives for OxiCUDA.
//!
//! Pure-Rust implementation of teacher-student training techniques covering logit-level,
//! feature-level, relation-based, attention-based, online, data-free, and progressive
//! distillation, together with evaluation metrics and GPU PTX kernel templates.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-distill
//! ├── logit/          — Hinton KD, DIST, DKD
//! ├── feature/        — FitNets, AT, PKT
//! ├── relation/       — RKD, CRD, CC (Gram)
//! ├── attention/      — Attention, Value (MiniLM), MHA distillation
//! ├── online/         — DML, BYOT, EMA self-distillation
//! ├── born_again/     — BAN, TAS, Progressive distillation
//! ├── data_free/      — DAFL, ZSKD
//! ├── metrics/        — Agreement, Divergence, Compression
//! ├── handle          — SmVersion, LcgRng, DistillHandle
//! ├── error           — DistillError / DistillResult
//! └── ptx_kernels     — 7 GPU PTX kernel strings × 6 SM versions
//! ```

pub mod attention;
pub mod born_again;
pub mod data_free;
pub mod error;
pub mod feature;
pub mod handle;
pub mod logit;
pub mod losses;
pub mod metrics;
pub mod online;
pub mod ptx_kernels;
pub mod regularization;
pub mod relation;

#[cfg(test)]
mod e2e_tests {
    use super::*;

    // ── Test 1 ──────────────────────────────────────────────────────────────
    /// When temperature = 1 and alpha = 0 the KD loss must equal the plain CE.
    #[test]
    fn hinton_kd_t1_equals_ce() {
        use logit::hinton_kd::{HintonKdConfig, cross_entropy, kd_loss};
        let cfg = HintonKdConfig {
            temperature: 1.0,
            alpha: 0.0,
        };
        let s = vec![1.0_f32, 2.0, 3.0, 0.5];
        let t = vec![2.0_f32, 1.0, 3.0, 0.5];
        let label = 2_usize;
        let kd = kd_loss(&s, &t, label, &cfg).expect("kd_loss should succeed");
        let ce = cross_entropy(&s, label);
        assert!(
            (kd - ce).abs() < 1e-4,
            "kd={kd} ce={ce}: with alpha=0 kd loss must equal CE"
        );
    }

    // ── Test 2 ──────────────────────────────────────────────────────────────
    /// Identical student and teacher logits → KL contribution must be 0.
    #[test]
    fn hinton_kd_symmetric_logits_zero_kl() {
        use logit::hinton_kd::{kl_divergence, softmax_with_temp};
        let logits = vec![1.0_f32, 2.0, 3.0];
        let p = softmax_with_temp(&logits, 4.0);
        let kl = kl_divergence(&p, &p);
        assert!(kl < 1e-5, "KL(p||p) must be ~0, got {kl}");
    }

    // ── Test 3 ──────────────────────────────────────────────────────────────
    /// Pearson correlation must lie in [-1, 1].
    #[test]
    fn dist_pearson_corr_range() {
        use logit::dist_distill::pearson_corr;
        let x: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..10).map(|i| (10 - i) as f32).collect();
        let r = pearson_corr(&x, &y);
        assert!((-1.0 - 1e-5..=1.0 + 1e-5).contains(&r), "pearson={r}");
    }

    // ── Test 4 ──────────────────────────────────────────────────────────────
    /// `at_map` output length must equal height × width.
    #[test]
    fn at_map_shape() {
        use feature::at::at_map;
        let ch = 4_usize;
        let h = 5_usize;
        let w = 6_usize;
        let feat: Vec<f32> = (0..ch * h * w).map(|i| i as f32).collect();
        let map = at_map(&feat, ch, h, w, 2.0);
        assert_eq!(map.len(), h * w);
    }

    // ── Test 5 ──────────────────────────────────────────────────────────────
    /// L2-normalised AT map should have unit norm.
    #[test]
    fn at_map_normalized_unit_norm() {
        use feature::at::{at_map, l2_normalize};
        let ch = 3_usize;
        let h = 4_usize;
        let w = 4_usize;
        let feat: Vec<f32> = (0..ch * h * w).map(|i| i as f32 + 1.0).collect();
        let map = at_map(&feat, ch, h, w, 2.0);
        let norm_map = l2_normalize(&map);
        let norm: f32 = norm_map.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
    }

    // ── Test 6 ──────────────────────────────────────────────────────────────
    /// RKD distance loss must be non-negative.
    #[test]
    fn rkd_dist_loss_nonneg() {
        use relation::rkd::distance_loss;
        let s: Vec<Vec<f32>> = (0..5)
            .map(|i| vec![i as f32, (i + 1) as f32, (i + 2) as f32])
            .collect();
        let t: Vec<Vec<f32>> = (0..5)
            .map(|i| vec![i as f32 * 0.9, (i + 1) as f32 * 1.1, (i + 2) as f32])
            .collect();
        let loss = distance_loss(&s, &t).expect("distance_loss should succeed");
        assert!(loss >= 0.0 && loss.is_finite(), "loss={loss}");
    }

    // ── Test 7 ──────────────────────────────────────────────────────────────
    /// After an EMA bank update the stored feature must differ from the initial.
    #[test]
    fn crd_bank_update() {
        use handle::LcgRng;
        use relation::crd::CrdMemoryBank;

        let mut rng = LcgRng::new(88);
        let mut bank = CrdMemoryBank::new(10, 8, 0.9, &mut rng);
        let original = bank.feats[0].clone();
        let new_feat: Vec<f32> = (0..8).map(|i| i as f32).collect();
        bank.update(0, &new_feat).expect("update should succeed");
        assert_ne!(bank.feats[0], original, "bank must change after EMA update");
    }

    // ── Test 8 ──────────────────────────────────────────────────────────────
    /// Gram matrix must be symmetric: G[i,j] == G[j,i] for all i, j.
    #[test]
    fn cc_gram_matrix_symmetric() {
        use relation::cc::gram_matrix;
        let feats: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32, (i + 1) as f32]).collect();
        let g = gram_matrix(&feats);
        let d = 2_usize;
        for i in 0..d {
            for j in 0..d {
                let gij = g[i * d + j];
                let gji = g[j * d + i];
                assert!(
                    (gij - gji).abs() < 1e-5,
                    "G[{i},{j}]={gij} != G[{j},{i}]={gji}"
                );
            }
        }
    }

    // ── Test 9 ──────────────────────────────────────────────────────────────
    /// `dml_all_losses` must return exactly one loss per peer.
    #[test]
    fn dml_all_losses_count() {
        use online::dml::dml_all_losses;
        let n_peers = 4_usize;
        let logits: Vec<Vec<f32>> = (0..n_peers)
            .map(|i| vec![i as f32, 2.0, (4 - i) as f32])
            .collect();
        let labels: Vec<usize> = (0..n_peers).map(|i| i % 3).collect();
        let losses = dml_all_losses(&logits, &labels).expect("dml_all_losses should succeed");
        assert_eq!(losses.len(), n_peers);
        for l in &losses {
            assert!(l.is_finite(), "loss not finite: {l}");
        }
    }

    // ── Test 10 ─────────────────────────────────────────────────────────────
    /// After one EMA update, teacher params must be the expected weighted mix.
    #[test]
    fn ema_teacher_momentum_update() {
        use online::sd_ema::EmaTeacher;
        let init = vec![0.0_f32; 4];
        let mut ema = EmaTeacher::new(&init, 0.9);
        let student = vec![1.0_f32; 4];
        ema.update(&student);
        // Expected: 0.9 * 0 + 0.1 * 1 = 0.1
        for &p in &ema.params {
            assert!((p - 0.1).abs() < 1e-5, "param={p} expected 0.1");
        }
    }

    // ── Test 11 ─────────────────────────────────────────────────────────────
    /// `top_k_agreement` on identical predictions must return 1.0.
    #[test]
    fn agreement_perfect() {
        use metrics::agreement::top_k_agreement;
        let logits: Vec<Vec<f32>> = vec![
            vec![3.0_f32, 1.0, 2.0],
            vec![1.0_f32, 3.0, 2.0],
            vec![2.0_f32, 1.0, 3.0],
        ];
        let agree = top_k_agreement(&logits, &logits, 1).expect("top_k_agreement should succeed");
        assert!(
            (agree - 1.0).abs() < 1e-5,
            "perfect agreement must be 1.0, got {agree}"
        );
    }

    // ── Test 12 ─────────────────────────────────────────────────────────────
    /// All 7 PTX kernels must return non-empty strings for all 6 SM targets.
    #[test]
    fn ptx_kernels_non_empty_all_sm() {
        use ptx_kernels::*;
        for sm in [75u32, 80, 86, 89, 90, 100] {
            assert!(!kd_loss_ptx(sm).is_empty(), "kd_loss sm={sm}");
            assert!(!mse_distill_ptx(sm).is_empty(), "mse_distill sm={sm}");
            assert!(!attn_distill_ptx(sm).is_empty(), "attn_distill sm={sm}");
            assert!(!at_pool_ptx(sm).is_empty(), "at_pool sm={sm}");
            assert!(!dml_loss_ptx(sm).is_empty(), "dml_loss sm={sm}");
            assert!(!crd_score_ptx(sm).is_empty(), "crd_score sm={sm}");
            assert!(!gram_matrix_ptx(sm).is_empty(), "gram_matrix sm={sm}");
        }
    }
}
