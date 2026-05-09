//! `oxicuda-continual` — Continual and lifelong learning primitives for OxiCUDA.
//!
//! Pure-Rust implementation of canonical continual learning algorithms,
//! suitable for CPU simulation and PTX kernel generation for GPU execution.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-continual
//! ├── regularization/   — EWC, SI (Synaptic Intelligence), MAS
//! ├── architecture/     — PackNet, Piggyback, Progressive Neural Networks
//! ├── replay/           — Experience Replay, GEM, A-GEM, DER++
//! ├── metrics/          — Forgetting, BWT, FWT, Intransigence
//! ├── stream/           — Task-incremental and class-incremental data streams
//! ├── error             — ContinualError / ContinualResult
//! ├── handle            — ContinualHandle (SmVersion + LcgRng)
//! └── ptx_kernels       — GPU PTX kernel strings (7 kernels × 6 SM versions)
//! ```

// ─── Module declarations ─────────────────────────────────────────────────────

pub mod architecture;
pub mod error;
pub mod handle;
pub mod metrics;
pub mod ptx_kernels;
pub mod regularization;
pub mod replay;
pub mod stream;

// ─── Prelude ─────────────────────────────────────────────────────────────────

/// Convenience re-exports for common continual learning types.
pub mod prelude {
    // Error handling
    pub use crate::error::{ContinualError, ContinualResult};

    // Handle
    pub use crate::handle::{ContinualHandle, LcgRng, SmVersion};

    // Regularization
    pub use crate::regularization::ewc::{
        EwcConfig, EwcRegularizer, FisherDiag, add_task as ewc_add_task, compute_fisher_empirical,
        ewc_loss,
    };
    pub use crate::regularization::mas::{
        MasConfig, MasImportance, mas_importance_update, mas_penalty,
    };
    pub use crate::regularization::si::{SiConfig, SiState, si_importance_update, si_penalty};

    // Architecture
    pub use crate::architecture::packnet::{
        PackNetMask, apply_mask, freeze_task_weights, prune_weights_l1,
    };
    pub use crate::architecture::piggyback::{
        PiggybackConfig, PiggybackMask, binarize_mask, piggyback_forward,
    };
    pub use crate::architecture::progressive::{
        LateralConnection, ProgNnColumn, ProgNnNetwork, add_column, prog_forward,
    };

    // Replay
    pub use crate::replay::a_gem::{AGemConfig, a_gem_project, average_gradients};
    pub use crate::replay::dark_exp::{DerBuffer, DerConfig, der_add, der_loss};
    pub use crate::replay::er::{ErBuffer, er_add, er_buffer_new, er_sample_batch};
    pub use crate::replay::gem::{GemConfig, GemMemory, gem_project_gradient};

    // Metrics
    pub use crate::metrics::forgetting::{
        AccuracyMatrix, TaskAccuracy, average_forgetting, backward_transfer, plasticity,
    };
    pub use crate::metrics::intransigence::{
        forward_transfer, intransigence, per_task_intransigence,
    };

    // Stream
    pub use crate::stream::class_stream::{
        ClassIncStream, advance_class_inc, class_inc_batch, class_inc_new, init_class_inc,
        n_classes_seen,
    };
    pub use crate::stream::task_stream::{
        Task, TaskStream, current_task, next_task, task_batch, task_stream_new,
    };

    // PTX kernels
    pub use crate::ptx_kernels::{
        ewc_penalty_ptx, f32_hex, fisher_diag_ptx, gradient_project_ptx, logit_distill_ptx,
        mask_apply_ptx, replay_sample_ptx, si_omega_update_ptx,
    };
}

// ─── End-to-end integration tests ────────────────────────────────────────────

#[cfg(test)]
mod e2e_tests {
    use crate::prelude::*;

    // ── Test 1: EWC loss = 0 at anchor ───────────────────────────────────────

    #[test]
    fn e2e_ewc_loss_zero_at_anchor() {
        let params = vec![0.5_f32, -1.0, 2.0, 0.3, -0.7];
        let fisher = FisherDiag {
            params: vec![1.0_f32, 2.0, 0.5, 3.0, 1.5],
        };
        let mut reg = EwcRegularizer::new();
        ewc_add_task(&mut reg, params.clone(), fisher);
        let cfg = EwcConfig {
            lambda: 2.0,
            n_tasks: 5,
        };
        let loss = ewc_loss(&params, &reg, &cfg).unwrap();
        assert!(
            loss.abs() < 1e-5,
            "E2E: EWC loss must be 0 at anchor, got {loss}"
        );
    }

    // ── Test 2: SI penalty grows with displacement ────────────────────────────

    #[test]
    fn e2e_si_penalty_grows_with_displacement() {
        let anchor = vec![0.0_f32; 8];
        let omega = vec![1.0_f32; 8];
        let cfg = SiConfig {
            lambda: 1.0,
            xi: 0.01,
        };
        let small_disp = vec![0.1_f32; 8];
        let large_disp = vec![1.0_f32; 8];
        let pen_small = si_penalty(&small_disp, &anchor, &omega, &cfg).unwrap();
        let pen_large = si_penalty(&large_disp, &anchor, &omega, &cfg).unwrap();
        assert!(
            pen_large > pen_small,
            "E2E: SI penalty must grow with displacement ({pen_small} < {pen_large})"
        );
    }

    // ── Test 3: MAS importance tracks gradient ────────────────────────────────

    #[test]
    fn e2e_mas_importance_tracks_gradient() {
        let mut omega = vec![0.0_f32; 4];
        let gradient = vec![3.0_f32, -2.0, 1.5, -0.5];
        mas_importance_update(&mut omega, &gradient, 0.0).unwrap();
        // With momentum=0: omega = |gradient|
        assert!((omega[0] - 3.0).abs() < 1e-5);
        assert!((omega[1] - 2.0).abs() < 1e-5);
        assert!((omega[2] - 1.5).abs() < 1e-5);
        assert!((omega[3] - 0.5).abs() < 1e-5);
    }

    // ── Test 4: PackNet sparsity respected ────────────────────────────────────

    #[test]
    fn e2e_packnet_sparsity_respected() {
        let weights: Vec<f32> = (1..=20).map(|i| i as f32).collect();
        let sparsity = 0.5;
        let mask = prune_weights_l1(&weights, sparsity, 0).unwrap();
        let n_active = mask.n_active();
        let expected_keep = (20.0 * (1.0 - sparsity)).floor() as usize;
        assert_eq!(
            n_active, expected_keep,
            "E2E: PackNet should keep {expected_keep} weights, got {n_active}"
        );
    }

    // ── Test 5: Piggyback binarization ────────────────────────────────────────

    #[test]
    fn e2e_piggyback_binarization() {
        let real_mask = vec![-0.5_f32, 0.5, -0.1, 0.1, 0.0, 1.0];
        let threshold = 0.0;
        let bin = binarize_mask(&real_mask, threshold).unwrap();
        assert_eq!(bin, vec![0, 1, 0, 1, 0, 1]);
    }

    // ── Test 6: ProgNN multi-column shape ─────────────────────────────────────

    #[test]
    fn e2e_progressive_multi_column_shape() {
        let mut rng = LcgRng::new(2024);
        let mut net = ProgNnNetwork::new();
        add_column(&mut net, 8, 2, &mut rng).unwrap();
        add_column(&mut net, 8, 2, &mut rng).unwrap();
        let input = vec![0.3_f32; 8];
        let out0 = prog_forward(&net, &input, 0).unwrap();
        let out1 = prog_forward(&net, &input, 1).unwrap();
        assert_eq!(out0.len(), 8, "E2E: Column 0 output shape should be 8");
        assert_eq!(out1.len(), 8, "E2E: Column 1 output shape should be 8");
        assert!(out0.iter().all(|v| v.is_finite()));
        assert!(out1.iter().all(|v| v.is_finite()));
    }

    // ── Test 7: ER reservoir bounded ─────────────────────────────────────────

    #[test]
    fn e2e_er_reservoir_bounded() {
        let mut rng = LcgRng::new(314);
        let mut buf = er_buffer_new(16).unwrap();
        for i in 0..200_usize {
            er_add(&mut buf, vec![i as f32; 4], (i % 5) as u32, &mut rng);
        }
        assert_eq!(
            buf.data.len(),
            16,
            "E2E: ER buffer must not exceed capacity of 16"
        );
        assert_eq!(buf.n_seen, 200);
    }

    // ── Test 8: GEM project satisfies constraint ──────────────────────────────

    #[test]
    fn e2e_gem_project_satisfies_constraint() {
        let g = vec![-3.0_f32, 1.0, -2.0, 0.5];
        let mem_grads = vec![
            vec![1.0_f32, 0.0, 0.0, 0.0],
            vec![0.0_f32, 1.0, 0.0, 0.0],
            vec![0.0_f32, 0.0, 1.0, 0.0],
        ];
        let margin = 0.0;
        let g_proj = gem_project_gradient(&g, &mem_grads, margin).unwrap();
        for mg in &mem_grads {
            let d: f32 = g_proj.iter().zip(mg.iter()).map(|(&a, &b)| a * b).sum();
            assert!(
                d >= -margin - 1e-4,
                "E2E: GEM projected gradient violates constraint: dot={d} < -margin={margin}"
            );
        }
    }

    // ── Test 9: A-GEM aligned unchanged ──────────────────────────────────────

    #[test]
    fn e2e_a_gem_aligned_unchanged() {
        let g = vec![1.0_f32, 1.0, 1.0, 1.0];
        let g_ref = vec![0.5_f32, 0.5, 0.5, 0.5]; // same direction
        let margin = 0.0;
        let g_proj = a_gem_project(&g, &g_ref, margin).unwrap();
        // g · g_ref = 2.0 > 0 = -margin → no projection
        for (a, b) in g.iter().zip(g_proj.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "E2E: Aligned A-GEM gradient should be unchanged"
            );
        }
    }

    // ── Test 10: DER++ loss is finite ─────────────────────────────────────────

    #[test]
    fn e2e_der_loss_finite() {
        let current_logits = vec![1.2_f32, -0.5, 0.3, 0.8, -1.0];
        let stored_logits = vec![0.9_f32, -0.3, 0.5, 0.7, -0.8];
        let label = 3_u32;
        let n_classes = 5;
        let cfg = DerConfig {
            alpha: 0.2,
            beta: 0.8,
        };
        let loss = der_loss(&current_logits, &stored_logits, label, n_classes, &cfg).unwrap();
        assert!(
            loss.is_finite(),
            "E2E: DER++ loss must be finite, got {loss}"
        );
        assert!(loss >= 0.0, "E2E: DER++ loss must be non-negative");
    }

    // ── Test 11: Forgetting = 0 for perfect retention ────────────────────────

    #[test]
    fn e2e_forgetting_zero_perfect_retention() {
        let n_tasks = 5;
        let mut mat = AccuracyMatrix::new(n_tasks);
        // Perfect retention: all tasks maintain 0.95 accuracy at all times
        for t in 0..n_tasks {
            for k in 0..=t {
                mat.data[t][k] = 0.95;
            }
        }
        let af = average_forgetting(&mat).unwrap();
        assert!(
            af.abs() < 1e-5,
            "E2E: Average forgetting must be 0 for perfect retention, got {af}"
        );
        let bwt = backward_transfer(&mat).unwrap();
        assert!(
            bwt.abs() < 1e-5,
            "E2E: BWT must be 0 for perfect retention, got {bwt}"
        );
    }

    // ── Test 12: All 7 PTX generators × 6 SM versions produce valid PTX ──────

    #[test]
    #[allow(clippy::type_complexity)]
    fn e2e_ptx_kernels_all_sm_versions() {
        let sm_versions = [75_u32, 80, 86, 90, 100, 120];
        let kernel_fns: &[(&str, fn(u32) -> String)] = &[
            ("ewc_penalty_kernel", ewc_penalty_ptx),
            ("fisher_diag_kernel", fisher_diag_ptx),
            ("gradient_project_kernel", gradient_project_ptx),
            ("mask_apply_kernel", mask_apply_ptx),
            ("si_omega_update_kernel", si_omega_update_ptx),
            ("logit_distill_kernel", logit_distill_ptx),
            ("replay_sample_kernel", replay_sample_ptx),
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
        // Smoke-test f32_hex
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
    }
}
