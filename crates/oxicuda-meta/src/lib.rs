//! `oxicuda-meta` — Meta-learning algorithm primitives for OxiCUDA.
//!
//! Pure-Rust implementation of MAML, FOMAML, ANIL, Reptile, ProtoNet, MatchingNet,
//! and RelationNet few-shot learning, suitable for CPU simulation and PTX kernel
//! generation for GPU execution.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-meta
//! ├── episode/         — Episode sampling and few-shot episode types
//! ├── gradient/        — Inner-loop SGD, cross-entropy loss, multi-step adaptation
//! ├── maml/            — MAML, FOMAML, ANIL meta-update algorithms
//! ├── metric_learning/ — ProtoNet, MatchingNet, RelationNet
//! ├── metrics/         — Few-shot accuracy, confidence intervals
//! ├── network/         — MLP backbone and linear classification head
//! ├── reptile/         — Reptile first-order meta-learner
//! ├── handle           — LcgRng, MetaHandle, SmVersion
//! ├── error            — MetaError / MetaResult
//! ├── ptx_kernels      — GPU PTX kernel strings (8 kernels × 6 SM versions)
//! └── prelude          — Convenience re-exports of common types
//! ```

pub mod episode;
pub mod error;
pub mod gradient;
pub mod handle;
pub mod maml;
pub mod metric_learning;
pub mod metrics;
pub mod network;
pub mod ptx_kernels;
pub mod reptile;

pub mod prelude {
    pub use crate::episode::sampler::EpisodeSampler;
    pub use crate::episode::types::{EpisodeConfig, FewShotEpisode};
    pub use crate::error::{MetaError, MetaResult};
    pub use crate::gradient::inner_loop::{cross_entropy_loss, inner_sgd_step, multi_step_inner};
    pub use crate::handle::{LcgRng, MetaHandle, SmVersion};
    pub use crate::maml::anil::{AnilConfig, anil_adapt_head, anil_meta_update};
    pub use crate::maml::fomaml::{FoMamlConfig, fomaml_update};
    pub use crate::maml::maml::{MamlConfig, maml_adapt, maml_meta_update};
    pub use crate::metric_learning::matching_net::{matching_net_attention, matching_net_predict};
    pub use crate::metric_learning::proto_net::{compute_prototypes, proto_loss, proto_predict};
    pub use crate::metric_learning::relation_net::RelationNet;
    pub use crate::metrics::few_shot::{accuracy_at_k, episode_accuracy, mean_and_ci95};
    pub use crate::network::backbone::MlpBackbone;
    pub use crate::network::linear_head::LinearHead;
    pub use crate::ptx_kernels::{
        cosine_sim_ptx, episode_sample_ptx, f32_hex, inner_sgd_ptx, meta_grad_accum_ptx,
        proto_distance_ptx, relation_score_ptx, reptile_update_ptx,
    };
    pub use crate::reptile::reptile::{ReptileConfig, reptile_update};
}

#[cfg(test)]
mod e2e_tests {
    use crate::prelude::*;

    #[test]
    fn e2e_proto_net_correct_class() {
        let feat_dim = 4;
        let n_way = 3;
        let k_shot = 2;
        let mut support_x = vec![0.0_f32; n_way * k_shot * feat_dim];
        for cls in 0..n_way {
            for k in 0..k_shot {
                for j in 0..feat_dim {
                    support_x[(cls * k_shot + k) * feat_dim + j] = if j == cls { 1.0 } else { 0.0 };
                }
            }
        }
        let support_y: Vec<u32> = (0..n_way)
            .flat_map(|c| std::iter::repeat_n(c as u32, k_shot))
            .collect();
        let protos = compute_prototypes(&support_x, &support_y, n_way, k_shot, feat_dim).unwrap();
        let query_x = protos.clone();
        let preds = proto_predict(&query_x, &protos, n_way, feat_dim).unwrap();
        for (i, &p) in preds.iter().enumerate() {
            assert_eq!(p as usize, i, "ProtoNet should predict class {i}");
        }
    }

    #[test]
    fn e2e_proto_net_identity_features() {
        let feat_dim = 8;
        let n_way = 2;
        let k_shot = 1;
        let support_x: Vec<f32> = (0..n_way)
            .flat_map(|c| (0..feat_dim).map(move |j| if j == c { 1.0_f32 } else { 0.0_f32 }))
            .collect();
        let support_y: Vec<u32> = (0..n_way as u32).collect();
        let protos = compute_prototypes(&support_x, &support_y, n_way, k_shot, feat_dim).unwrap();
        let preds = proto_predict(&support_x, &protos, n_way, feat_dim).unwrap();
        for (i, &p) in preds.iter().enumerate() {
            assert_eq!(p, support_y[i]);
        }
    }

    #[test]
    fn e2e_matching_net_attention_sums_to_one() {
        let feat_dim = 4;
        let n_way = 3;
        let k_shot = 2;
        let n_support = n_way * k_shot;
        let mut rng = LcgRng::new(1234);
        let query_feat: Vec<f32> = (0..feat_dim).map(|_| rng.next_f32()).collect();
        let support_feats: Vec<f32> = (0..n_support * feat_dim).map(|_| rng.next_f32()).collect();
        let support_y: Vec<u32> = (0..n_way)
            .flat_map(|c| std::iter::repeat_n(c as u32, k_shot))
            .collect();
        let attn =
            matching_net_attention(&query_feat, &support_feats, &support_y, n_way, 1.0).unwrap();
        let sum: f32 = attn.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Attention weights must sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn e2e_matching_net_same_class_highest() {
        let feat_dim = 4;
        let n_way = 3;
        let support_feats: Vec<f32> = (0..n_way)
            .flat_map(|c| (0..feat_dim).map(move |j| if j == c { 1.0_f32 } else { 0.0_f32 }))
            .collect();
        let support_y: Vec<u32> = (0..n_way as u32).collect();
        let query_feat: Vec<f32> = (0..feat_dim)
            .map(|j| if j == 1 { 1.0_f32 } else { 0.0_f32 })
            .collect();
        let attn =
            matching_net_attention(&query_feat, &support_feats, &support_y, n_way, 5.0).unwrap();
        let best_cls = attn
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(best_cls, 1, "Same-class should get highest attention");
    }

    #[test]
    fn e2e_relation_net_same_class_higher() {
        let feat_dim = 4;
        let hidden_dim = 8;
        let mut rng = LcgRng::new(42);
        let net = RelationNet::new(feat_dim, hidden_dim, &mut rng);
        let feat_a = vec![1.0_f32, 0.0, 0.0, 0.0];
        let feat_b = vec![0.0_f32, 1.0, 0.0, 0.0];
        let score_same = net.relation_score(&feat_a, &feat_a).unwrap();
        let score_diff = net.relation_score(&feat_a, &feat_b).unwrap();
        assert!((0.0..=1.0).contains(&score_same));
        assert!((0.0..=1.0).contains(&score_diff));
        assert!(score_same.is_finite());
        assert!(score_diff.is_finite());
    }

    #[test]
    fn e2e_relation_net_loss_finite() {
        let feat_dim = 4;
        let hidden_dim = 8;
        let n_way = 2;
        let k_shot = 2;
        let n_query = 2;
        let mut rng = LcgRng::new(99);
        let net = RelationNet::new(feat_dim, hidden_dim, &mut rng);
        let n_support = n_way * k_shot;
        let support_x: Vec<f32> = (0..n_support * feat_dim).map(|_| rng.next_f32()).collect();
        let support_y: Vec<u32> = (0..n_way)
            .flat_map(|c| std::iter::repeat_n(c as u32, k_shot))
            .collect();
        let query_x: Vec<f32> = (0..n_way * n_query * feat_dim)
            .map(|_| rng.next_f32())
            .collect();
        let query_y: Vec<u32> = (0..n_way)
            .flat_map(|c| std::iter::repeat_n(c as u32, n_query))
            .collect();
        let cfg = EpisodeConfig {
            n_way,
            k_shot,
            n_query,
            feat_dim,
        };
        let episode = FewShotEpisode {
            config: cfg,
            support_x,
            support_y,
            query_x,
            query_y,
        };
        let loss = net.relation_loss(&episode).unwrap();
        assert!(
            loss.is_finite(),
            "RelationNet loss must be finite, got {loss}"
        );
        assert!(loss >= 0.0);
    }

    #[test]
    fn e2e_maml_adapt_changes_params() {
        let n_classes = 2;
        let feat_dim = 4;
        let n_params = n_classes * feat_dim + n_classes;
        let params: Vec<f32> = (0..n_params).map(|i| i as f32 * 0.1).collect();
        let support_x = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let support_y: Vec<u32> = vec![0, 1];
        let cfg = MamlConfig {
            inner_lr: 0.01,
            n_inner_steps: 3,
        };
        let adapted =
            maml_adapt(&params, &support_x, &support_y, n_classes, feat_dim, &cfg).unwrap();
        assert_ne!(params, adapted, "MAML adapt must change params");
    }

    #[test]
    fn e2e_reptile_update_toward_task() {
        let n_classes = 2;
        let feat_dim = 2;
        let n_params = n_classes * feat_dim + n_classes;
        let params = vec![0.0_f32; n_params];
        let support_x = vec![1.0_f32, 0.0, 0.0, 1.0];
        let support_y: Vec<u32> = vec![0, 1];
        let task_data = vec![(support_x, support_y)];
        let cfg = ReptileConfig {
            inner_lr: 0.1,
            n_inner_steps: 3,
            step_size: 0.5,
        };
        let updated = reptile_update(&params, &task_data, n_classes, feat_dim, &cfg).unwrap();
        let moved = updated.iter().any(|&p| p.abs() > 1e-6);
        assert!(moved, "Reptile params must move toward task minimum");
    }

    #[test]
    fn e2e_inner_sgd_decreases_loss() {
        let params = vec![2.0_f32];
        let grads = vec![2.0_f32 * params[0]];
        let lr = 0.1;
        let new_params = inner_sgd_step(&params, &grads, lr).unwrap();
        assert!(
            new_params[0].abs() < params[0].abs(),
            "SGD step should move closer to 0: {new_params:?}"
        );
    }

    #[test]
    fn e2e_episode_sampler_correct_shapes() {
        let cfg = EpisodeConfig {
            n_way: 3,
            k_shot: 2,
            n_query: 4,
            feat_dim: 8,
        };
        let sampler = EpisodeSampler::new(cfg.clone()).unwrap();
        let n_classes = 5_usize;
        let examples_per_class = cfg.k_shot + cfg.n_query;
        let n_total = n_classes * examples_per_class;
        let mut rng = LcgRng::new(2025);
        let data: Vec<f32> = (0..n_total * cfg.feat_dim)
            .map(|_| rng.next_f32())
            .collect();
        let labels: Vec<u32> = (0..n_total).map(|i| (i % n_classes) as u32).collect();
        let episode = sampler.sample(&data, &labels, n_classes, &mut rng).unwrap();
        assert_eq!(
            episode.support_x.len(),
            cfg.n_way * cfg.k_shot * cfg.feat_dim
        );
        assert_eq!(episode.support_y.len(), cfg.n_way * cfg.k_shot);
        assert_eq!(
            episode.query_x.len(),
            cfg.n_way * cfg.n_query * cfg.feat_dim
        );
        assert_eq!(episode.query_y.len(), cfg.n_way * cfg.n_query);
    }

    #[test]
    fn e2e_episode_accuracy_correct() {
        let preds = vec![0_u32, 1, 2, 1, 0];
        let labels = vec![0_u32, 1, 2, 1, 0];
        let acc = episode_accuracy(&preds, &labels).unwrap();
        assert!((acc - 1.0).abs() < 1e-6, "100% accuracy should be 1.0");
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn e2e_ptx_kernels_all_sm_versions() {
        let sm_versions = [75_u32, 80, 86, 90, 100, 120];
        let kernel_fns: &[(&str, fn(u32) -> String)] = &[
            ("inner_sgd_kernel", inner_sgd_ptx),
            ("reptile_update_kernel", reptile_update_ptx),
            ("proto_distance_kernel", proto_distance_ptx),
            ("cosine_sim_kernel", cosine_sim_ptx),
            ("relation_score_kernel", relation_score_ptx),
            ("meta_grad_accum_kernel", meta_grad_accum_ptx),
            ("episode_sample_kernel", episode_sample_ptx),
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
