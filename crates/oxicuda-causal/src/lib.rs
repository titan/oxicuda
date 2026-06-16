//! `oxicuda-causal` — Causal inference primitives for OxiCUDA.
//!
//! Pure-Rust implementation of causal discovery, effect estimation, and do-calculus,
//! suitable for CPU simulation and PTX kernel generation for GPU execution.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-causal
//! ├── dag/             — Directed Acyclic Graph, d-separation, topological sort
//! ├── discovery/       — NOTEARS (SEM), PC algorithm (constraint-based)
//! ├── do_calculus/     — Backdoor criterion, identification
//! ├── effect/          — IPW, Double ML (DML), DragonNet
//! ├── forest/          — Causal forest (heterogeneous treatment effects)
//! ├── iv/              — Instrumental variable estimation
//! ├── counterfactual/  — Counterfactual prediction
//! ├── metrics/         — Causal evaluation metrics
//! ├── handle           — LcgRng (deterministic PRNG)
//! ├── error            — CausalError / CausalResult
//! └── ptx_kernels      — GPU PTX kernel strings (7 kernels × 6 SM versions)
//! ```

pub mod counterfactual;
pub mod dag;
pub mod discovery;
pub mod do_calculus;
pub mod effect;
pub mod error;
pub mod forest;
pub mod handle;
pub mod iv;
pub mod metrics;
pub mod ptx_kernels;
pub mod sensitivity;

#[cfg(test)]
mod e2e_tests {
    use super::*;
    use dag::d_separation::d_separated;
    use dag::dag::Dag;
    use discovery::notears::NotearsSem;
    use discovery::pc::PcAlgorithm;
    use do_calculus::identification::backdoor_admissible;
    use effect::double_ml::DoubleML;
    use effect::dragonnet::DragonNet;
    use effect::ipw::ipw_ate;
    use effect::propensity::PropensityModel;
    use forest::causal_forest::CausalForest;
    use handle::LcgRng;

    #[test]
    fn dag_add_remove_edge() {
        let mut dag = Dag::new(5);
        dag.add_edge(0, 1).expect("add edge 0->1");
        dag.add_edge(1, 2).expect("add edge 1->2");
        dag.add_edge(2, 3).expect("add edge 2->3");
        assert!(dag.has_edge(0, 1));
        assert!(!dag.has_edge(1, 0));
        dag.remove_edge(1, 2);
        assert!(!dag.has_edge(1, 2));
        let order = dag.topo_sort().expect("topo_sort on acyclic dag");
        assert_eq!(order.len(), 5);
    }

    #[test]
    fn dag_cycle_detected() {
        let mut dag = Dag::new(4);
        dag.add_edge(0, 1).expect("add edge 0->1 in cycle test");
        dag.add_edge(1, 2).expect("add edge 1->2 in cycle test");
        dag.add_edge(2, 3).expect("add edge 2->3 in cycle test");
        let result = dag.add_edge(3, 0);
        assert!(result.is_err());
    }

    #[test]
    fn d_separation_chain() {
        // X -> Z -> Y: X d-sep Y given {Z}
        let mut dag = Dag::new(3);
        dag.add_edge(0, 2).expect("add edge 0->2");
        dag.add_edge(2, 1).expect("add edge 2->1");
        assert!(d_separated(&dag, 0, 1, &[2]));
        assert!(!d_separated(&dag, 0, 1, &[]));
    }

    #[test]
    fn notears_fit_acyclic() {
        let mut sem = NotearsSem::new(3);
        let n = 50_usize;
        let d = 3_usize;
        let mut rng = LcgRng::new(42);
        let mut x = vec![0.0_f32; n * d];
        for i in 0..n {
            x[i * d] = rng.next_normal();
            x[i * d + 1] = 0.5 * x[i * d] + rng.next_normal() * 0.3;
            x[i * d + 2] = 0.5 * x[i * d + 1] + rng.next_normal() * 0.3;
        }
        let _ = sem.fit(&x, n, 0.1, 50);
        let h: f32 = sem.w.iter().map(|&v| v * v).sum();
        assert!(h.is_finite());
    }

    #[test]
    fn pc_runs_small() {
        let n = 20;
        let d = 3;
        let mut rng = LcgRng::new(99);
        let mut data = vec![0.0_f32; n * d];
        for i in 0..n {
            data[i * d] = rng.next_normal();
            data[i * d + 1] = 0.7 * data[i * d] + rng.next_normal() * 0.5;
            data[i * d + 2] = rng.next_normal();
        }
        let result = PcAlgorithm::run(&data, n, d, 0.05);
        assert!(result.is_ok());
        let pc = result.expect("PC algorithm should succeed on valid input");
        assert!(pc.skeleton.len() <= d * (d - 1) / 2);
    }

    #[test]
    fn propensity_in_0_1() {
        let n = 30;
        let d = 4;
        let mut rng = LcgRng::new(77);
        let mut x = vec![0.0_f32; n * d];
        for v in x.iter_mut() {
            *v = rng.next_normal();
        }
        let t: Vec<f32> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.0 }).collect();
        let mut model = PropensityModel::new(d, &mut rng);
        model
            .fit(&x, &t, n, 0.01, 100)
            .expect("propensity model fit should succeed");
        let preds = model
            .predict(&x, n)
            .expect("propensity model predict should succeed");
        assert_eq!(preds.len(), n);
        for &p in &preds {
            assert!((0.0..=1.0).contains(&p), "propensity={p} out of [0,1]");
        }
    }

    #[test]
    fn ipw_ate_finite() {
        let n = 40;
        let y: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let t: Vec<f32> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.0 }).collect();
        let pi: Vec<f32> = vec![0.6_f32; n / 2]
            .into_iter()
            .chain(vec![0.4_f32; n / 2])
            .collect();
        let ate = ipw_ate(&y, &t, &pi).expect("IPW ATE estimation should succeed");
        assert!(ate.is_finite());
    }

    #[test]
    fn double_ml_ate_finite() {
        let n = 60;
        let d = 3;
        let mut rng = LcgRng::new(11);
        let mut x = vec![0.0_f32; n * d];
        for v in x.iter_mut() {
            *v = rng.next_normal();
        }
        let t: Vec<f32> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.0 }).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| x[i * d] * 0.5 + t[i] * 2.0 + rng.next_normal() * 0.1)
            .collect();
        let result = DoubleML::fit(&y, &t, &x, n, d, 3).expect("DoubleML fit should succeed");
        assert!(result.ate.is_finite());
        assert!(result.std_error >= 0.0);
    }

    #[test]
    fn dragonnet_forward_finite() {
        let mut rng = LcgRng::new(33);
        let net = DragonNet::new(5, 16, 2, &mut rng);
        let x: Vec<f32> = (0..5).map(|i| i as f32 * 0.1 + 0.05).collect();
        let (mu0, mu1, pi) = net
            .forward(&x)
            .expect("DragonNet forward pass should succeed");
        assert!(mu0.is_finite());
        assert!(mu1.is_finite());
        assert!(pi.is_finite());
        assert!(pi > 0.0 && pi < 1.0);
    }

    #[test]
    fn causal_forest_fit_predict() {
        let mut rng = LcgRng::new(200);
        let n = 50;
        let d = 4;
        let mut x = vec![0.0_f32; n * d];
        for v in x.iter_mut() {
            *v = rng.next_normal();
        }
        let t: Vec<f32> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.0 }).collect();
        let y: Vec<f32> = (0..n).map(|i| x[i * d] + t[i] * 1.5).collect();
        let mut forest = CausalForest::new(5, d, 3, &mut rng);
        forest
            .fit(&x, &t, &y, n)
            .expect("causal forest fit should succeed");
        let preds = forest
            .predict(&x, n)
            .expect("causal forest predict should succeed");
        assert_eq!(preds.len(), n);
        for &p in &preds {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn backdoor_admissible_chain() {
        let mut dag = Dag::new(3);
        dag.add_edge(0, 1).expect("add_edge should succeed");
        dag.add_edge(0, 2).expect("add_edge should succeed");
        assert!(backdoor_admissible(&dag, 0, 1, &[]));
    }

    #[test]
    fn ptx_kernels_non_empty_all_sm() {
        use ptx_kernels::*;
        for sm in [75u32, 80, 86, 89, 90, 100] {
            assert!(!partial_corr_ptx(sm).is_empty(), "partial_corr sm={sm}");
            assert!(!notears_loss_ptx(sm).is_empty(), "notears_loss sm={sm}");
            assert!(!expm_pade_ptx(sm).is_empty(), "expm_pade sm={sm}");
            assert!(
                !propensity_logit_ptx(sm).is_empty(),
                "propensity_logit sm={sm}"
            );
            assert!(!ipw_estimator_ptx(sm).is_empty(), "ipw_estimator sm={sm}");
            assert!(!dml_residual_ptx(sm).is_empty(), "dml_residual sm={sm}");
            assert!(
                !causal_split_score_ptx(sm).is_empty(),
                "causal_split_score sm={sm}"
            );
        }
    }
}
