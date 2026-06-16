//! `oxicuda-recsys` — Recommender system primitives for OxiCUDA.
//!
//! Pure-Rust implementation of collaborative filtering, neural recommendation models,
//! and graph-based recommenders, suitable for CPU simulation and PTX kernel generation
//! for GPU execution.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-recsys
//! ├── factorization/  — ALS, BPR, NMF matrix factorization
//! ├── ncf/            — Neural Collaborative Filtering
//! ├── two_tower/      — Two-Tower retrieval model
//! ├── deepfm/         — DeepFM and Wide & Deep models
//! ├── sequential/     — SASRec self-attention sequential model
//! ├── graph_recsys/   — LightGCN graph-based recommendation
//! ├── multitask/      — MMoE / PLE multi-task learning
//! ├── sampling/       — Uniform negative sampling
//! ├── metrics/        — NDCG@k, Precision@k ranking metrics
//! ├── handle          — LcgRng (deterministic PRNG)
//! ├── error           — RecSysError / RecSysResult
//! └── ptx_kernels     — GPU PTX kernel strings (7 kernels × 6 SM versions)
//! ```

pub mod error;
pub mod handle;
pub mod ptx_kernels;

pub mod deepfm;
pub mod dlrm;
pub mod factorization;
pub mod fibinet;
pub mod graph_recsys;
pub mod metrics;
pub mod multitask;
pub mod ncf;
pub mod ranking;
pub mod sampling;
pub mod sequential;
pub mod two_tower;

pub use crate::deepfm::dcn::{CrossKind, Dcn, DcnConfig};
pub use crate::dlrm::{Dlrm, DlrmConfig};
pub use crate::factorization::ffm::{Ffm, FfmConfig, FfmEntry};
pub use crate::factorization::fism::{Fism, FismConfig};
pub use crate::factorization::ials::{Ials, IalsConfig};
pub use crate::fibinet::{BilinearType, Fibinet, FibinetConfig};
pub use crate::graph_recsys::graphrec::GraphRec;
pub use crate::ranking::fairness_ranking::{FairnessRanker, FairnessRankerConfig};
pub use crate::sequential::cl4srec::{Cl4sRec, Cl4sRecConfig};

#[cfg(test)]
mod e2e_tests {
    use std::collections::{BTreeSet, HashSet};

    use crate::{
        deepfm::{deepfm::DeepFm, wide_deep::WideDeep},
        factorization::{als::Als, bpr::Bpr, nmf::Nmf},
        graph_recsys::lightgcn::LightGcn,
        handle::LcgRng,
        metrics::recsys_metrics::{ndcg_at_k, precision_at_k},
        ncf::ncf::Ncf,
        ptx_kernels,
        sampling::uniform_neg::UniformNegSampler,
        sequential::sasrec::SasRec,
        two_tower::two_tower::TwoTower,
    };

    #[test]
    fn als_score_finite() {
        let mut rng = LcgRng::new(42);
        let mut model = Als::new(5, 8, 4, 0.01, &mut rng).expect("new should succeed");
        let interactions = vec![(0, 0, 1.0), (0, 2, 2.0), (1, 1, 1.0), (2, 3, 3.0)];
        model.fit(&interactions, 2).expect("fit should succeed");
        let s = model.score(0, 0).expect("score should succeed");
        assert!(s.is_finite(), "ALS score must be finite, got {s}");
    }

    #[test]
    fn bpr_step_loss_finite() {
        let mut rng = LcgRng::new(7);
        let mut model = Bpr::new(5, 10, 4, 0.01, 0.001, &mut rng).expect("new should succeed");
        let triplets = vec![(0, 1, 3), (1, 2, 4), (2, 0, 5)];
        let loss = model.train_step(&triplets);
        assert!(loss.is_finite(), "BPR loss must be finite, got {loss}");
    }

    #[test]
    fn nmf_fit_no_error() {
        let mut rng = LcgRng::new(13);
        let data = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 2, 3.0), (2, 0, 1.5)];
        let model = Nmf::fit(&data, 3, 4, 3, 5, &mut rng).expect("fit should succeed");
        let s = model.score(0, 0).expect("score should succeed");
        assert!(s.is_finite(), "NMF score must be finite, got {s}");
    }

    #[test]
    fn ncf_forward_in_0_1() {
        let mut rng = LcgRng::new(99);
        let model = Ncf::new(10, 20, 8, vec![16, 8], &mut rng).expect("new should succeed");
        let pred = model.forward(0, 0).expect("forward should succeed");
        assert!(
            (0.0..=1.0).contains(&pred),
            "NCF output {pred} not in [0,1]"
        );
    }

    #[test]
    fn two_tower_score_finite() {
        let mut rng = LcgRng::new(11);
        let model = TwoTower::new(8, 16, 8, 2, &mut rng).expect("new should succeed");
        let user_x: Vec<f32> = (0..8).map(|_| rng.next_f32()).collect();
        let item_x: Vec<f32> = (0..8).map(|_| rng.next_f32()).collect();
        let score = model.score(&user_x, &item_x).expect("score should succeed");
        assert!(
            score.is_finite(),
            "TwoTower score must be finite, got {score}"
        );
    }

    #[test]
    fn deepfm_forward_in_0_1() {
        let mut rng = LcgRng::new(55);
        let field_dims = vec![10, 20, 5];
        let model = DeepFm::new(field_dims, 8, &[32, 16], &mut rng).expect("new should succeed");
        let field_ids = vec![0usize, 3, 2];
        let pred = model.forward(&field_ids).expect("forward should succeed");
        assert!(
            (0.0..=1.0).contains(&pred),
            "DeepFM output {pred} not in [0,1]"
        );
    }

    #[test]
    fn wide_deep_forward_in_0_1() {
        let mut rng = LcgRng::new(33);
        let model = WideDeep::new(16, &[32, 16], &mut rng).expect("new should succeed");
        let x: Vec<f32> = (0..16).map(|_| rng.next_f32()).collect();
        let pred = model.forward(&x).expect("forward should succeed");
        assert!(
            (0.0..=1.0).contains(&pred),
            "WideDeep output {pred} not in [0,1]"
        );
    }

    #[test]
    fn sasrec_forward_finite() {
        let mut rng = LcgRng::new(77);
        let model = SasRec::new(50, 16, 2, 2, 20, &mut rng).expect("new should succeed");
        let item_ids = vec![0usize, 5, 12, 3];
        let logits = model.forward(&item_ids).expect("forward should succeed");
        assert_eq!(logits.len(), 50);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "SASRec logits must all be finite"
        );
    }

    #[test]
    fn lightgcn_score_finite() {
        let mut rng = LcgRng::new(21);
        let mut model = LightGcn::new(5, 8, 8, 2, &mut rng).expect("new should succeed");
        let edges = vec![(0, 0), (0, 1), (1, 2), (2, 3), (3, 4)];
        model.propagate(&edges).expect("propagate should succeed");
        let s = model.score(0, 1);
        assert!(s.is_finite(), "LightGCN score must be finite, got {s}");
    }

    #[test]
    fn ndcg_perfect_ranking() {
        let recommended: Vec<usize> = (0..10).collect();
        let relevant: HashSet<usize> = (0..5).collect();
        let ndcg = ndcg_at_k(&recommended, &relevant, 10);
        let prec = precision_at_k(&recommended, &relevant, 5);
        assert!(
            (ndcg - 1.0).abs() < 1e-5,
            "NDCG for perfect ranking must be 1.0, got {ndcg}"
        );
        assert!(
            (prec - 1.0).abs() < 1e-5,
            "Precision@5 for perfect ranking must be 1.0, got {prec}"
        );
    }

    #[test]
    fn uniform_neg_not_in_positives() {
        let mut rng = LcgRng::new(3);
        let sampler = UniformNegSampler::new(100).expect("new should succeed");
        let mut positives = BTreeSet::new();
        positives.insert(0usize);
        positives.insert(1);
        positives.insert(2);
        for _ in 0..50 {
            let neg = sampler
                .sample(0, &positives, &mut rng)
                .expect("sample should succeed");
            assert!(
                !positives.contains(&neg),
                "Sampled negative {neg} must not be in positives"
            );
        }
    }

    #[test]
    fn ptx_kernels_non_empty_all_sm() {
        let sm_versions = [75u32, 80, 86, 89, 90, 100];
        type KernelFn = Box<dyn Fn(u32) -> String>;
        let generators: Vec<(&str, KernelFn)> = vec![
            ("als_step", Box::new(ptx_kernels::als_step_ptx)),
            ("bpr_grad", Box::new(ptx_kernels::bpr_grad_ptx)),
            (
                "embedding_lookup",
                Box::new(ptx_kernels::embedding_lookup_ptx),
            ),
            ("dot_score", Box::new(ptx_kernels::dot_score_ptx)),
            ("softmax_topk", Box::new(ptx_kernels::softmax_topk_ptx)),
            (
                "negsample_uniform",
                Box::new(ptx_kernels::negsample_uniform_ptx),
            ),
            (
                "lightgcn_propagate",
                Box::new(ptx_kernels::lightgcn_propagate_ptx),
            ),
        ];
        for sm in sm_versions {
            for (name, kernel_fn) in &generators {
                let ptx = kernel_fn(sm);
                assert!(
                    !ptx.is_empty(),
                    "PTX kernel {name} for SM {sm} must be non-empty"
                );
                assert!(
                    ptx.contains(&format!("sm_{sm}")),
                    "PTX kernel {name} for SM {sm} must contain .target sm_{sm}"
                );
            }
        }
    }
}
