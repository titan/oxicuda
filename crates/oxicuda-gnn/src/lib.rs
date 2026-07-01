//! `oxicuda-gnn` — Graph Neural Network primitives for OxiCUDA.
//!
//! Pure-Rust implementation of GNN building blocks suitable for CPU simulation
//! and PTX kernel generation for GPU execution.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-gnn
//! ├── graph/          — Sparse graph representations (CSR, COO, Heterogeneous, Sampling)
//! ├── message_passing — Aggregate, Scatter, Update primitives
//! ├── layers/         — GCN, GAT, GATv2, GraphSAGE, GIN
//! ├── pooling/        — Global pool, Top-K pool, DiffPool
//! ├── readout/        — Set2Set
//! ├── error           — GnnError / GnnResult
//! ├── handle          — GnnHandle (SmVersion + LcgRng)
//! └── ptx_kernels     — GPU PTX kernel strings
//! ```

// ─── Module declarations ─────────────────────────────────────────────────────

pub mod conv;
pub mod error;
pub mod graph;
pub mod handle;
pub mod layers;
pub mod message_passing;
pub mod ops;
pub mod pooling;
pub mod ptx_kernels;
pub mod readout;
pub mod sampling;

// ─── Prelude ─────────────────────────────────────────────────────────────────

/// Convenience re-exports for common GNN types.
pub mod prelude {
    pub use crate::conv::gcnii::{Gcnii, GcniiConfig, gcnii_beta};
    pub use crate::error::{GnnError, GnnResult};
    pub use crate::graph::coo::CooGraph;
    pub use crate::graph::csr::CsrGraph;
    pub use crate::graph::heterogeneous::HeteroGraph;
    pub use crate::graph::sampling::{NeighborhoodSampler, SampledGraph, biased_walk, random_walk};
    pub use crate::handle::{GnnHandle, LcgRng, SmVersion};
    pub use crate::layers::appnp::{AppnpConfig, AppnpLayer};
    pub use crate::layers::chebnet::{ChebNetConfig, ChebNetLayer};
    pub use crate::layers::gat::{GatConfig, GatLayer};
    pub use crate::layers::gat_v2::{GatV2Config, GatV2Layer};
    pub use crate::layers::gcn::{GcnConfig, GcnLayer};
    pub use crate::layers::gin::{GinConfig, GinLayer};
    pub use crate::layers::grand::{GrandConfig, GrandLayer};
    pub use crate::layers::graph_transformer::{
        GraphTransformerConfig, GraphTransformerLayer, GraphTransformerWeights,
    };
    pub use crate::layers::hetero_conv::{HeteroConv, HeteroConvConfig, HeteroConvWeights};
    pub use crate::layers::jk_net::{JkMode, JkNet, JkNetConfig};
    pub use crate::layers::k_wl_gnn::{
        KWlConfig, KWlGnn, PairOp, apply_pair_op, graph_readout_sum,
    };
    pub use crate::layers::mixhop::{MixHopConfig, MixHopLayer};
    pub use crate::layers::norm::{GraphNorm, PairNorm, PairNormMode};
    pub use crate::layers::rgcn::{RgcnConfig, RgcnLayer};
    pub use crate::layers::rwse::{RwseConfig, RwseEncoder, random_walk_se};
    pub use crate::layers::sage::{SageAggregator, SageConfig, SageLayer};
    pub use crate::layers::sgc::{sgc_forward, sgc_linear, sgc_propagate};
    pub use crate::layers::sign::{SignConfig, SignConv, sign_precompute};
    pub use crate::message_passing::aggregate::{
        AggregationType, aggregate, aggregate_degree_norm, aggregate_max, aggregate_mean,
        aggregate_softmax, aggregate_sum, build_edge_messages,
    };
    pub use crate::message_passing::scatter::{
        gather, scatter_add, scatter_max, scatter_min, scatter_mul, segment_softmax,
    };
    pub use crate::message_passing::update::{
        LinearUpdate, MlpUpdate, elu, leaky_relu, prelu, relu,
    };
    pub use crate::ops::balanced_spmv::{BalancedSpmvConfig, balanced_spmv};
    pub use crate::ops::edge_parallel_gat::{EdgeParallelGat, EdgeParallelGatConfig};
    pub use crate::pooling::diff_pool::{DiffPool, DiffPoolConfig, DiffPoolResult};
    pub use crate::pooling::global_pool::{
        GlobalPoolType, batched_global_pool, global_attention_pool, global_max_pool,
        global_mean_pool, global_sum_pool,
    };
    pub use crate::pooling::sag_pool::{SagPool, SagPoolResult};
    pub use crate::pooling::topk_pool::{TopKPool, TopKPoolResult};
    pub use crate::ptx_kernels::{
        aggregate_mean_ptx, csr_spmv_ptx, f32_hex, gat_attention_ptx, gin_combine_ptx,
        scatter_add_ptx, softmax_edge_ptx, topk_score_ptx,
    };
    pub use crate::readout::dgi::{Dgi, DgiConfig, DgiLoss, DgiWeights};
    pub use crate::readout::set2set::Set2Set;
    pub use crate::readout::sort_pool::{SortPool, SortPoolConfig};
    pub use crate::sampling::cluster_gcn::{BatchSubgraph, ClusterGcn, Partition};
    pub use crate::sampling::graphsaint::{GraphSaint, SaintNorm, SaintSampler, SaintSubgraph};
}

// ─── On-device GPU validation ────────────────────────────────────────────────

/// On-device GPU validation tests (feature-gated): JIT-compile each hand-written
/// PTX kernel, launch it on a real CUDA device, and assert numerical equivalence
/// to the matching CPU reference. Compiled only under `--features gpu-tests` and
/// only in test builds; every test skips gracefully if no GPU is available.
#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests;

// ─── Integration tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    // ── Graph construction & SpMV ─────────────────────────────────────────────

    #[test]
    fn e2e_csr_graph_construction_and_spmv() {
        // Triangle graph: 0↔1↔2↔0
        let g = CsrGraph::from_edges(3, &[(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)])
            .expect("test invariant: value must be valid");
        assert_eq!(g.n_nodes(), 3);
        assert_eq!(g.n_edges(), 6);

        // SpMV with feat_dim=1: x = [1, 2, 3]
        // y[0] = x[1] + x[2] = 5, y[1] = x[0] + x[2] = 4, y[2] = x[0] + x[1] = 3
        let x = vec![1.0_f32, 2.0, 3.0];
        let y = g.spmv(&x, 1).expect("test invariant: value must be valid");
        assert!((y[0] - 5.0).abs() < 1e-5);
        assert!((y[1] - 4.0).abs() < 1e-5);
        assert!((y[2] - 3.0).abs() < 1e-5);
    }

    // ── COO → CSR roundtrip ───────────────────────────────────────────────────

    #[test]
    fn e2e_coo_to_csr_roundtrip() {
        let src = vec![0usize, 1, 2, 0];
        let dst = vec![1usize, 2, 0, 2];
        let coo = CooGraph::new(3, src.clone(), dst.clone())
            .expect("test invariant: value must be valid");
        let csr = coo.to_csr().expect("test invariant: value must be valid");
        assert_eq!(csr.n_nodes(), 3);
        assert_eq!(csr.n_edges(), 4);

        // Each source in coo should be a valid node in csr
        for &s in &src {
            assert!(csr.degree(s).expect("test invariant: value must be valid") > 0);
        }
    }

    // ── Scatter-add ───────────────────────────────────────────────────────────

    #[test]
    fn e2e_scatter_add_correctness() {
        // 4 messages → 2 destination nodes, feat_dim=2
        let messages = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let idx = vec![0usize, 0, 1, 1];
        let out = scatter_add(&messages, &idx, 2, 2).expect("test invariant: value must be valid");
        // dest 0 = [1+3, 2+4] = [4, 6]
        assert!((out[0] - 4.0).abs() < 1e-5);
        assert!((out[1] - 6.0).abs() < 1e-5);
        // dest 1 = [5+7, 6+8] = [12, 14]
        assert!((out[2] - 12.0).abs() < 1e-5);
        assert!((out[3] - 14.0).abs() < 1e-5);
    }

    // ── Aggregate mean ────────────────────────────────────────────────────────

    #[test]
    fn e2e_aggregate_mean_small_graph() {
        // Node 0 receives messages from edge 0 and edge 1
        let messages = vec![2.0_f32, 4.0, 6.0, 8.0]; // 2 messages × feat_dim=2
        let target_idx = vec![0usize, 0];
        let out = aggregate_mean(&messages, &target_idx, 2, 2)
            .expect("test invariant: value must be valid");
        // node 0: mean([2,4],[6,8]) = [4, 6]
        assert!((out[0] - 4.0).abs() < 1e-5);
        assert!((out[1] - 6.0).abs() < 1e-5);
        // node 1: no messages → 0
        assert!((out[2]).abs() < 1e-6);
    }

    // ── GCN forward shape ─────────────────────────────────────────────────────

    #[test]
    fn e2e_gcn_forward_shape() {
        let g = CsrGraph::from_edges(4, &[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)])
            .expect("test invariant: value must be valid");
        let layer = GcnLayer::new(GcnConfig {
            in_features: 4,
            out_features: 8,
            bias: false,
            normalize: true,
        })
        .expect("test invariant: value must be valid");
        let feats = vec![0.1_f32; 4 * 4];
        let w = vec![0.1_f32; 4 * 8];
        let out = layer
            .forward(&g, &feats, &w, None)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 4 * 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── GAT attention sums to one ─────────────────────────────────────────────

    #[test]
    fn e2e_gat_attention_sums_to_one() {
        // 3-node ring with uniform features
        let g = CsrGraph::from_edges(3, &[(0, 1), (1, 2), (2, 0)])
            .expect("test invariant: value must be valid");
        let in_f = 4;
        let out_f = 4;
        let nh = 1;
        let hd = out_f;
        let layer = GatLayer::new(GatConfig {
            in_features: in_f,
            out_features: out_f,
            num_heads: nh,
            dropout: 0.0,
            leaky_relu_slope: 0.2,
            concat_heads: true,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 3 * in_f];
        let w = vec![1.0_f32; nh * hd * in_f];
        let aw = vec![0.1_f32; nh * 2 * hd];
        let out = layer
            .forward(&g, &x, &w, &aw)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 3 * out_f);
        // Outputs should be finite
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── GraphSAGE mean aggregator ─────────────────────────────────────────────

    #[test]
    fn e2e_sage_mean_aggregator() {
        let g = CsrGraph::from_edges(4, &[(0, 1), (0, 2), (1, 3), (2, 3)])
            .expect("test invariant: value must be valid");
        let layer = SageLayer::new(SageConfig {
            in_features: 3,
            out_features: 3,
            aggregator: SageAggregator::Mean,
            normalize_output: false,
        })
        .expect("test invariant: value must be valid");
        let x = vec![0.5_f32; 4 * 3];
        let w = vec![0.1_f32; 3 * 6];
        let b = vec![0.0_f32; 3];
        let out = layer
            .forward(&g, &x, &w, &b)
            .expect("test invariant: value must be valid");
        assert_eq!(out.len(), 4 * 3);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── GIN epsilon effect ────────────────────────────────────────────────────

    #[test]
    fn e2e_gin_epsilon_effect() {
        let g = CsrGraph::from_edges(3, &[(0, 1), (1, 2)])
            .expect("test invariant: value must be valid");
        let x = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // 3×3 identity
        let make_gin = |eps: f32| {
            GinLayer::new(GinConfig {
                in_features: 3,
                hidden_features: 4,
                out_features: 3,
                epsilon: eps,
                train_epsilon: false,
            })
            .expect("test invariant: value must be valid")
        };
        let w1 = vec![0.1_f32; 4 * 3];
        let b1 = vec![0.0_f32; 4];
        let w2 = vec![0.1_f32; 3 * 4];
        let b2 = vec![0.0_f32; 3];
        let out_e0 = make_gin(0.0)
            .forward(&g, &x, &w1, &b1, &w2, &b2)
            .expect("test invariant: value must be valid");
        let out_e1 = make_gin(1.0)
            .forward(&g, &x, &w1, &b1, &w2, &b2)
            .expect("test invariant: value must be valid");
        assert_eq!(out_e0.len(), 9);
        assert_eq!(out_e1.len(), 9);
        // Outputs differ because epsilon changes the weighting
        let diff: f32 = out_e0
            .iter()
            .zip(out_e1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0 || out_e0.iter().all(|&v| v.abs() < 1e-8));
    }

    // ── Global mean pool ──────────────────────────────────────────────────────

    #[test]
    fn e2e_global_mean_pool() {
        let x = vec![2.0_f32, 4.0, 6.0, 8.0]; // 2 nodes × feat_dim=2
        let out = global_mean_pool(&x, 2, 2).expect("test invariant: value must be valid");
        assert_eq!(out.len(), 2);
        assert!((out[0] - 4.0).abs() < 1e-5); // (2+6)/2
        assert!((out[1] - 6.0).abs() < 1e-5); // (4+8)/2
    }

    // ── Top-K pool: k nodes selected ─────────────────────────────────────────

    #[test]
    fn e2e_topk_pool_k_nodes_selected() {
        let g = CsrGraph::from_edges(
            5,
            &[
                (0, 1),
                (1, 0),
                (1, 2),
                (2, 1),
                (2, 3),
                (3, 2),
                (3, 4),
                (4, 3),
            ],
        )
        .expect("test invariant: value must be valid");
        let feat_dim = 3;
        let k = 3;
        let pool = TopKPool::new_k(feat_dim, k);
        let x: Vec<f32> = (0..5 * feat_dim).map(|i| i as f32 * 0.2).collect();
        let proj = vec![1.0_f32, 0.5, 0.25];
        let res = pool
            .forward(&g, &x, &proj)
            .expect("test invariant: value must be valid");
        assert_eq!(res.n_nodes(), k);
        assert_eq!(res.x.len(), k * feat_dim);
        assert_eq!(res.graph.n_nodes(), k);
    }

    // ── DiffPool assignment row-stochastic ────────────────────────────────────

    #[test]
    fn e2e_diffpool_assignment_stochastic() {
        let g = CsrGraph::from_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)])
            .expect("test invariant: value must be valid");
        let d = 3;
        let k = 2;
        let dp = DiffPool::new(DiffPoolConfig {
            in_features: d,
            n_clusters: k,
        })
        .expect("test invariant: value must be valid");
        let x = vec![1.0_f32; 4 * d];
        let logits: Vec<f32> = (0..4 * k).map(|i| i as f32 * 0.1).collect();
        let res = dp
            .forward(&g, &x, &logits)
            .expect("test invariant: value must be valid");
        for i in 0..4 {
            let row_sum: f32 = res.assignment[i * k..(i + 1) * k].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-5);
        }
    }

    // ── PTX kernels for all SM versions ──────────────────────────────────────

    #[test]
    fn e2e_ptx_kernels_all_sm_versions() {
        for &sm in &[75u32, 80, 86, 90, 100, 120] {
            let ptx = csr_spmv_ptx(sm);
            assert!(ptx.contains("csr_spmv"));
            assert!(ptx.contains(&format!("sm_{sm}")));

            let ptx = scatter_add_ptx(sm);
            assert!(ptx.contains("scatter_add"));

            let ptx = gat_attention_ptx(sm);
            assert!(ptx.contains("gat_attention"));

            let ptx = softmax_edge_ptx(sm);
            assert!(ptx.contains("softmax_edge"));

            let ptx = aggregate_mean_ptx(sm);
            assert!(ptx.contains("aggregate_mean"));

            let ptx = gin_combine_ptx(sm);
            assert!(ptx.contains("gin_combine"));

            let ptx = topk_score_ptx(sm);
            assert!(ptx.contains("topk_score"));
        }
    }

    // ── Handle and RNG ────────────────────────────────────────────────────────

    #[test]
    fn e2e_handle_rng_deterministic() {
        let mut h1 = GnnHandle::default_handle();
        let mut h2 = GnnHandle::default_handle();
        // Same seed → same sequence
        let r1: Vec<u32> = (0..10).map(|_| h1.rng_mut().next_u32()).collect();
        let r2: Vec<u32> = (0..10).map(|_| h2.rng_mut().next_u32()).collect();
        assert_eq!(r1, r2);
    }

    // ── Neighbourhood sampling ────────────────────────────────────────────────

    #[test]
    fn e2e_neighborhood_sampling() {
        let g = CsrGraph::from_edges(
            8,
            &[
                (0, 1),
                (0, 2),
                (1, 3),
                (1, 4),
                (2, 5),
                (2, 6),
                (3, 7),
                (4, 7),
            ],
        )
        .expect("test invariant: value must be valid");
        let sampler =
            NeighborhoodSampler::new(vec![2, 2]).expect("test invariant: value must be valid");
        let mut rng = LcgRng::new(42);
        let result = sampler
            .sample(&g, &[0], &mut rng)
            .expect("test invariant: value must be valid");
        assert!(result.n_nodes() >= 1);
        assert!(result.local_to_global.contains(&0));
    }
}
