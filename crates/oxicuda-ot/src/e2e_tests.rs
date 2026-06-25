//! End-to-end tests across the OT solver stack.
//!
//! These tests intentionally span multiple modules to verify that the
//! Sinkhorn-Knopp, network-simplex, and Wasserstein primitives agree on
//! shared problems and uphold the standard OT identities.

#![cfg(test)]

use crate::barycenter::free_support::{BaryConfig, free_support_barycenter};
use crate::bridge::schrodinger::{SchrodingerConfig, schrodinger_bridge};
use crate::clustering::wasserstein_kmeans::{WkmConfig, wasserstein_kmeans};
use crate::domain::mapping::barycentric_map;
use crate::exact::network_simplex::{NsConfig, network_simplex};
use crate::gromov::gromov_wasserstein::{GwConfig, entropic_gw};
use crate::handle::{LcgRng, OtHandle, SmVersion};
use crate::metrics::metrics::{kl_divergence, marginal_violation, transport_cost};
use crate::multi::multi_marginal::{MmConfig, multi_marginal_ot};
use crate::ptx_kernels::{
    barycenter_update_ptx, cost_matrix_ptx, gromov_grad_ptx, sinkhorn_step_ptx, sliced_proj_ptx,
    transport_apply_ptx, unbalanced_step_ptx,
};
use crate::sinkhorn::divergence::sinkhorn_divergence;
use crate::sinkhorn::sinkhorn::{SinkhornConfig, sinkhorn};
use crate::unbalanced::unbalanced_ot::{UnbalancedConfig, unbalanced_ot};
use crate::wasserstein::sliced::{SlicedConfig, sliced_w};
use crate::wasserstein::w1::w1_1d;
use crate::wasserstein::w2::w2_1d;

#[test]
fn sinkhorn_converges_to_simplex_on_small_problem() {
    let m = 3;
    let n = 3;
    let c = vec![0.0_f32, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0];
    let a = vec![1.0_f32 / 3.0; 3];
    let b = vec![1.0_f32 / 3.0; 3];
    let exact = network_simplex(&c, &a, &b, m, n, &NsConfig::default()).expect("ok");
    let sk_cfg = SinkhornConfig {
        eps: 0.3,
        max_iter: 5000,
        tol: 1e-4,
    };
    let sk = sinkhorn(&c, &a, &b, m, n, &sk_cfg).expect("ok");
    assert!(
        (sk.cost - exact.cost).abs() < 0.5,
        "sinkhorn={} simplex={}",
        sk.cost,
        exact.cost
    );
}

#[test]
fn sinkhorn_marginals_satisfied() {
    let m = 4;
    let n = 4;
    let c: Vec<f32> = (0..m * n)
        .map(|k| {
            let i = (k / n) as f32;
            let j = (k % n) as f32;
            (i - j).powi(2)
        })
        .collect();
    let a = vec![0.25_f32; 4];
    let b = vec![0.25_f32; 4];
    let cfg = SinkhornConfig {
        eps: 0.3,
        max_iter: 2000,
        tol: 1e-5,
    };
    let res = sinkhorn(&c, &a, &b, m, n, &cfg).expect("ok");
    let (row_v, col_v) = marginal_violation(&res.plan, &a, &b, m, n).expect("ok");
    assert!(row_v < 5e-3, "row violation {row_v}");
    assert!(col_v < 5e-3, "col violation {col_v}");
}

#[test]
fn sinkhorn_uniform_plan_when_eps_large() {
    let m = 3;
    let n = 3;
    let c = vec![1.0_f32; m * n];
    let a = vec![1.0_f32 / 3.0; 3];
    let b = vec![1.0_f32 / 3.0; 3];
    let cfg = SinkhornConfig {
        eps: 100.0,
        max_iter: 200,
        tol: 1e-5,
    };
    let res = sinkhorn(&c, &a, &b, m, n, &cfg).expect("ok");
    let expected = 1.0_f32 / 9.0;
    for &p in &res.plan {
        assert!(
            (p - expected).abs() < 5e-3,
            "plan {p} not uniform {expected}"
        );
    }
}

#[test]
fn sinkhorn_divergence_self_zero() {
    let m = 4;
    let n = 4;
    let c: Vec<f32> = (0..m * n)
        .map(|k| ((k / n) as f32 - (k % n) as f32).powi(2))
        .collect();
    let a = vec![0.25_f32; 4];
    let cfg = SinkhornConfig {
        eps: 0.3,
        max_iter: 2000,
        tol: 1e-5,
    };
    let s = sinkhorn_divergence(&c, &c, &c, &a, &a, m, n, &cfg).expect("ok");
    assert!(s.abs() < 5e-3, "S_eps(a,a) = {s} not ~0");
}

#[test]
fn w1_translation_invariance_1d() {
    // Two points each, with same weights, shifted by 2.0
    let x = vec![0.0_f32, 1.0];
    let y = vec![2.0_f32, 3.0];
    let a = vec![0.5_f32, 0.5];
    let b = vec![0.5_f32, 0.5];
    let d = w1_1d(&x, &y, &a, &b).expect("ok");
    assert!((d - 2.0).abs() < 1e-3, "expected 2.0, got {d}");
}

#[test]
fn w2_diracs_distance_is_euclidean() {
    let x = vec![0.0_f32];
    let y = vec![3.0_f32];
    let a = vec![1.0_f32];
    let b = vec![1.0_f32];
    let d = w2_1d(&x, &y, &a, &b).expect("ok");
    assert!((d - 3.0).abs() < 1e-3, "expected 3.0, got {d}");
}

#[test]
fn sliced_w_zero_on_equal() {
    let n = 16_usize;
    let dim = 3_usize;
    let mut rng = LcgRng::new(7);
    let mut samples = vec![0.0_f32; n * dim];
    rng.fill_normal(&mut samples);
    let cfg = SlicedConfig {
        n_proj: 50,
        p: 2,
        seed: 42,
    };
    let s = sliced_w(&samples, &samples, dim, n, n, &cfg).expect("ok");
    assert!(s.abs() < 5e-2, "expected 0, got {s}");
}

#[test]
fn entropic_gw_converges_on_matched_metric() {
    let m = 3;
    let n = 3;
    let c1 = vec![0.0_f32, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0];
    let c2 = c1.clone();
    let a = vec![1.0_f32 / 3.0; 3];
    let b = vec![1.0_f32 / 3.0; 3];
    let cfg = GwConfig {
        eps: 0.1,
        max_iter: 100,
        inner_max_iter: 500,
        tol: 1e-3,
    };
    let res = entropic_gw(&c1, &c2, &a, &b, m, n, &cfg).expect("ok");
    let (row_v, col_v) = marginal_violation(&res.plan, &a, &b, m, n).expect("ok");
    assert!(row_v < 5e-2);
    assert!(col_v < 5e-2);
}

#[test]
fn unbalanced_large_tau_matches_balanced() {
    let m = 3;
    let n = 3;
    let c: Vec<f32> = (0..9)
        .map(|k| ((k / 3) as f32 - (k % 3) as f32).powi(2))
        .collect();
    let a = vec![1.0_f32 / 3.0; 3];
    let b = vec![1.0_f32 / 3.0; 3];
    let cfg_b = SinkhornConfig {
        eps: 0.3,
        max_iter: 2000,
        tol: 1e-4,
    };
    let cfg_u = UnbalancedConfig {
        eps: 0.3,
        tau_a: 1e6,
        tau_b: 1e6,
        max_iter: 2000,
        tol: 1e-4,
    };
    let res_b = sinkhorn(&c, &a, &b, m, n, &cfg_b).expect("balanced");
    let res_u = unbalanced_ot(&c, &a, &b, m, n, &cfg_u).expect("unbalanced");
    for k in 0..m * n {
        assert!(
            (res_u.plan[k] - res_b.plan[k]).abs() < 0.05,
            "entry {k}: balanced={}, unbalanced={}",
            res_b.plan[k],
            res_u.plan[k]
        );
    }
}

#[test]
fn schrodinger_bridge_marginals() {
    let m = 3;
    let n = 3;
    let c: Vec<f32> = (0..9)
        .map(|k| ((k / 3) as f32 - (k % 3) as f32).powi(2))
        .collect();
    let a = vec![1.0_f32 / 3.0; 3];
    let b = vec![1.0_f32 / 3.0; 3];
    let cfg = SchrodingerConfig {
        eps: 0.3,
        max_iter: 1000,
        tol: 1e-5,
    };
    let res = schrodinger_bridge(&c, &a, &b, m, n, &cfg).expect("ok");
    let (row_v, col_v) = marginal_violation(&res.plan, &a, &b, m, n).expect("ok");
    assert!(row_v < 5e-3);
    assert!(col_v < 5e-3);
}

#[test]
fn multi_marginal_k2_matches_sinkhorn() {
    let m = 3;
    let n = 3;
    let c: Vec<f32> = (0..9)
        .map(|k| ((k / 3) as f32 - (k % 3) as f32).powi(2))
        .collect();
    let a = vec![1.0_f32 / 3.0; 3];
    let cfg_mm = MmConfig {
        eps: 0.4,
        max_iter: 4000,
        tol: 1e-3,
    };
    let cfg_sk = SinkhornConfig {
        eps: 0.4,
        max_iter: 4000,
        tol: 1e-3,
    };
    let p_mm = multi_marginal_ot(&c, &[a.clone(), a.clone()], &[m, n], &cfg_mm).expect("mm");
    let p_sk = sinkhorn(&c, &a, &a, m, n, &cfg_sk).expect("sk");
    for (k, (mm, sk)) in p_mm.iter().zip(p_sk.plan.iter()).enumerate() {
        assert!((mm - sk).abs() < 5e-3, "{k}: {mm} vs {sk}");
    }
}

#[test]
fn barycenter_of_self_is_self() {
    let dim = 1_usize;
    let measures_x = vec![vec![0.0_f32, 1.0, 2.0]];
    let measures_a = vec![vec![1.0_f32 / 3.0; 3]];
    let lambdas = vec![1.0_f32];
    let mut rng = LcgRng::new(1);
    let cfg = BaryConfig {
        eps: 0.05,
        n_outer: 20,
        n_inner: 200,
        tol: 1e-3,
    };
    let (y, _b) =
        free_support_barycenter(&measures_x, &measures_a, dim, 3, &lambdas, &cfg, &mut rng)
            .expect("ok");
    let mut sorted_y = y.clone();
    sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut sorted_x = measures_x[0].clone();
    sorted_x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    for (a, b) in sorted_y.iter().zip(sorted_x.iter()) {
        assert!(
            (a - b).abs() < 0.5,
            "barycenter of self diverged: {a} vs {b}"
        );
    }
}

#[test]
fn wasserstein_kmeans_runs() {
    let dim = 1_usize;
    let measures_x = vec![
        vec![0.0_f32, 0.1, 0.2],
        vec![0.0_f32, 0.1, 0.3],
        vec![5.0_f32, 5.1, 5.2],
        vec![5.0_f32, 5.2, 5.3],
    ];
    let measures_a: Vec<Vec<f32>> = (0..4).map(|_| vec![1.0_f32 / 3.0; 3]).collect();
    let cfg = WkmConfig {
        n_clusters: 2,
        max_iter: 5,
        eps: 0.1,
        seed: 7,
    };
    let res = wasserstein_kmeans(&measures_x, &measures_a, dim, 3, &cfg).expect("ok");
    assert_eq!(res.assignments.len(), 4);
    assert_eq!(res.centroids.len(), 2);
}

#[test]
fn barycentric_mapping_returns_target_under_uniform_plan() {
    let m = 2;
    let n = 3;
    let dim = 1_usize;
    let plan = vec![1.0_f32 / 6.0; m * n];
    let target_y = vec![0.0_f32, 1.0, 2.0];
    let mapped = barycentric_map(&plan, &target_y, m, n, dim).expect("ok");
    let mean = (0.0_f32 + 1.0 + 2.0) / 3.0;
    for &v in &mapped {
        assert!((v - mean).abs() < 1e-5);
    }
}

#[test]
fn metrics_kl_self_zero_and_cost_finite() {
    let p = vec![0.25_f32; 4];
    let kl = kl_divergence(&p, &p).expect("ok");
    assert!(kl.abs() < 1e-6);
    let plan = vec![1.0_f32 / 16.0; 16];
    let cost = vec![1.0_f32; 16];
    let tc = transport_cost(&plan, &cost).expect("ok");
    assert!((tc - 1.0).abs() < 1e-6);
}

#[test]
fn handle_constructs_with_rng_state() {
    let h = OtHandle::new(80, 42);
    assert_eq!(h.sm(), SmVersion(80));
}

#[test]
fn ptx_kernels_non_empty_all_sm() {
    for sm in [75u32, 80, 86, 89, 90, 100] {
        assert!(sinkhorn_step_ptx(sm).contains(".visible .entry"));
        assert!(cost_matrix_ptx(sm).contains(".visible .entry"));
        assert!(transport_apply_ptx(sm).contains(".visible .entry"));
        assert!(sliced_proj_ptx(sm).contains(".visible .entry"));
        assert!(gromov_grad_ptx(sm).contains(".visible .entry"));
        assert!(unbalanced_step_ptx(sm).contains(".visible .entry"));
        assert!(barycenter_update_ptx(sm).contains(".visible .entry"));
    }
}

/// Cross-validate the exact network-simplex EMD against entropic Sinkhorn at
/// problem sizes far larger than the historical `n = m = 3 / 8` checks.
///
/// # What this proves
///
/// For each `n = m ∈ {16, 32, 64}` we draw a *seeded* well-conditioned instance
/// (random 3-D Euclidean ground cost, random marginals, both via the crate
/// [`LcgRng`]) and verify three independent facts:
///
/// 1. **Cost agreement as `ε → 0`.** The exact (un-regularised) optimal-transport
///    cost from the network-simplex solver and the entropic-Sinkhorn transport
///    cost must coincide. Entropic regularisation makes Sinkhorn's cost a small
///    *upper* bound on the exact optimum, with a gap that shrinks with `ε`. We
///    drive `ε` toward zero with epsilon-scaling (deterministic annealing) from
///    `ε₀ = 2.0` down to `ε_target = 2·10⁻³`; at that target the **relative** gap
///    `|cost_sinkhorn − cost_simplex| / cost_simplex` is below `8·10⁻³` at every
///    size (empirically ≤ 2.6·10⁻³ over 180 random instances), and Sinkhorn never
///    *under*-shoots the exact optimum by more than a tiny slack.
/// 2. **Marginal fidelity.** The Sinkhorn plan's row sums and column sums match
///    the prescribed marginals `a` and `b` to `< 5·10⁻³`.
/// 3. **Feasibility of both plans.** Every entry of both transport plans is
///    non-negative and the total transported mass equals one (each plan is a
///    genuine coupling).
///
/// The `ε`/tolerance relationship is the crux: at a fixed annealed
/// `ε_target = 2e-3` the entropic and exact costs agree to better than 1 %.
/// A *larger* `ε` would widen the gap; a much *smaller* `ε` is unnecessary here
/// and is handled separately by the `ε → 0` stability harness.
#[test]
fn sinkhorn_agrees_with_network_simplex_on_large_problems() {
    use crate::sinkhorn::epsilon_scaling::{EpsilonScalingConfig, epsilon_scaling_sinkhorn};

    // Per-size deterministic seeds, each verified to keep the network-simplex
    // within its iteration budget on the generated instance.
    let cases: [(usize, u64); 3] = [(16, 0x16_5EED), (32, 0x32_5EED), (64, 0x64_5EED)];

    for (sz, seed) in cases {
        let m = sz;
        let n = sz;
        let dim = 3usize;
        let mut rng = LcgRng::new(seed);

        // Random source / target point clouds in [-3, 3]^dim.
        let mut xs = vec![0.0f32; m * dim];
        let mut ys = vec![0.0f32; n * dim];
        for v in xs.iter_mut() {
            *v = rng.next_f32() * 6.0 - 3.0;
        }
        for v in ys.iter_mut() {
            *v = rng.next_f32() * 6.0 - 3.0;
        }

        // Euclidean ground cost C_ij = ‖x_i − y_j‖₂ (generic, tie-free → the
        // transportation simplex stays non-degenerate and converges quickly).
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for d in 0..dim {
                    let diff = xs[i * dim + d] - ys[j * dim + d];
                    s += diff * diff;
                }
                c[i * n + j] = s.sqrt();
            }
        }

        // Random, strictly-positive marginals normalised to unit mass.
        let mut a = vec![0.0f32; m];
        for v in a.iter_mut() {
            *v = rng.next_f32() + 0.05;
        }
        let sa: f32 = a.iter().sum();
        for v in a.iter_mut() {
            *v /= sa;
        }
        let mut b = vec![0.0f32; n];
        for v in b.iter_mut() {
            *v = rng.next_f32() + 0.05;
        }
        let sb: f32 = b.iter().sum();
        for v in b.iter_mut() {
            *v /= sb;
        }

        // ---- Exact optimum via the network-simplex solver. ----
        let exact = network_simplex(&c, &a, &b, m, n, &NsConfig { max_iter: 200_000 })
            .unwrap_or_else(|e| panic!("network-simplex failed at n=m={sz}: {e}"));

        // ---- Entropic Sinkhorn cost, annealed toward ε → 0. ----
        let eps_cfg = EpsilonScalingConfig {
            eps_init: 2.0,
            eps_target: 2e-3,
            scale: 0.6,
            inner_iter: 60,
            final_iter: 2500,
            tol: 1e-4,
        };
        let sk = epsilon_scaling_sinkhorn(&c, &a, &b, m, n, &eps_cfg)
            .unwrap_or_else(|e| panic!("epsilon-scaling Sinkhorn failed at n=m={sz}: {e}"));

        assert!(
            exact.cost.is_finite() && sk.cost.is_finite(),
            "non-finite cost at n=m={sz}: simplex={}, sinkhorn={}",
            exact.cost,
            sk.cost
        );

        // (1) Cost agreement: entropic cost is a near-exact upper bound on the
        // simplex optimum at this small ε.
        let rel_gap = (sk.cost - exact.cost).abs() / exact.cost.max(1e-6);
        assert!(
            rel_gap < 8e-3,
            "n=m={sz}: relative gap {rel_gap} too large (simplex={}, sinkhorn={})",
            exact.cost,
            sk.cost
        );
        // Entropic regularisation cannot push the cost meaningfully *below* the
        // exact optimum (only a tiny numerical slack is allowed).
        assert!(
            sk.cost >= exact.cost - 5e-3,
            "n=m={sz}: sinkhorn {} undershoots exact {}",
            sk.cost,
            exact.cost
        );

        // (2) Sinkhorn plan marginals match the targets.
        let (sk_row_v, sk_col_v) = marginal_violation(&sk.plan, &a, &b, m, n).expect("ok");
        assert!(
            sk_row_v < 5e-3,
            "n=m={sz}: sinkhorn row violation {sk_row_v}"
        );
        assert!(
            sk_col_v < 5e-3,
            "n=m={sz}: sinkhorn col violation {sk_col_v}"
        );

        // The exact plan must satisfy the marginals essentially to machine ε.
        let (ns_row_v, ns_col_v) = marginal_violation(&exact.plan, &a, &b, m, n).expect("ok");
        assert!(
            ns_row_v < 1e-4,
            "n=m={sz}: simplex row violation {ns_row_v}"
        );
        assert!(
            ns_col_v < 1e-4,
            "n=m={sz}: simplex col violation {ns_col_v}"
        );

        // (3) Feasibility of both couplings: non-negative entries, unit total mass.
        for &p in &exact.plan {
            assert!(p >= -1e-6, "n=m={sz}: simplex plan has negative entry {p}");
        }
        for &p in &sk.plan {
            assert!(p >= -1e-9, "n=m={sz}: sinkhorn plan has negative entry {p}");
        }
        let ns_mass: f32 = exact.plan.iter().sum();
        let sk_mass: f32 = sk.plan.iter().sum();
        assert!(
            (ns_mass - 1.0).abs() < 1e-3,
            "n=m={sz}: simplex total mass {ns_mass} ≠ 1"
        );
        assert!(
            (sk_mass - 1.0).abs() < 1e-3,
            "n=m={sz}: sinkhorn total mass {sk_mass} ≠ 1"
        );
    }
}
