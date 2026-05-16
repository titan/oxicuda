//! End-to-end integration tests for `oxicuda-tn`.

use crate::contraction::einsum::{LabelledTensor, einsum_binary};
use crate::contraction::path::{execute_path, greedy_path};
use crate::cp::als::cp_als;
use crate::dmrg::dmrg::{DmrgConfig, dmrg_two_site, mpo_expectation};
use crate::dmrg::lanczos::lanczos_smallest;
use crate::handle::LcgRng;
use crate::metrics::metrics::{entanglement_entropy, fidelity};
use crate::mpo::contraction::apply_mpo_to_mps;
use crate::mpo::mpo::Mpo;
use crate::mps::canonical::{left_canonicalize, right_canonicalize};
use crate::mps::mps::Mps;
use crate::mps::tensor::MpsTensor;
use crate::ptx_kernels::{
    dmrg_local_apply_ptx, hosvd_unfold_ptx, mpo_apply_ptx, svd_jacobi_step_ptx,
    tensor_contract_ptx, trotter_step_ptx, tt_round_ptx,
};
use crate::svd::svd_jacobi;
use crate::tt::tt_svd::tt_svd;
use crate::tucker::hosvd::{hosvd, tucker_reconstruct};

fn fro_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// 1. MPS norm of product state = 1
#[test]
fn mps_product_state_norm_one() {
    let local = vec![vec![1.0, 0.0]; 4];
    let mps = Mps::from_product_state(&local).expect("ok");
    let n = mps.norm_squared().expect("ok");
    assert!((n - 1.0).abs() < 1e-12);
}

// 2. MPS left-canonical sites are column-orthonormal
#[test]
fn mps_left_canonical_orthonormal() {
    let mut rng = LcgRng::new(7);
    let mut mps = Mps::random_mps(4, 2, 3, &mut rng).expect("ok");
    left_canonicalize(&mut mps).expect("ok");
    let n = mps.n_sites();
    for s in 0..n - 1 {
        let t = &mps.site_tensors[s];
        let m = t.d_l * t.d_p;
        let dr = t.d_r;
        for i in 0..dr {
            for j in 0..dr {
                let mut acc = 0.0;
                for r in 0..m {
                    acc += t.data[r * dr + i] * t.data[r * dr + j];
                }
                let target = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (acc - target).abs() < 1e-7,
                    "site {s} (i={i}, j={j}) = {acc}, expected {target}"
                );
            }
        }
    }
}

// 3. MPS right-canonical sites are row-orthonormal
#[test]
fn mps_right_canonical_orthonormal() {
    let mut rng = LcgRng::new(13);
    let mut mps = Mps::random_mps(4, 2, 3, &mut rng).expect("ok");
    right_canonicalize(&mut mps).expect("ok");
    for s in 1..mps.n_sites() {
        let t = &mps.site_tensors[s];
        let dl = t.d_l;
        let cols = t.d_p * t.d_r;
        for i in 0..dl {
            for j in 0..dl {
                let mut acc = 0.0;
                for c in 0..cols {
                    acc += t.data[i * cols + c] * t.data[j * cols + c];
                }
                let target = if i == j { 1.0 } else { 0.0 };
                assert!((acc - target).abs() < 1e-7);
            }
        }
    }
}

// 4. MPO·MPS application preserves length (under identity MPO)
#[test]
fn mpo_mps_identity_preserves_norm() {
    let local = vec![vec![0.6, 0.8]; 3];
    let mps = Mps::from_product_state(&local).expect("ok");
    let mpo = Mpo::identity(3, 2).expect("ok");
    let out = apply_mpo_to_mps(&mpo, &mps, 4, 1e-12).expect("ok");
    let n_in = mps.norm_squared().expect("ok");
    let n_out = out.norm_squared().expect("ok");
    assert!((n_in - n_out).abs() < 1e-8);
}

// 5. DMRG on 3-site identity MPO recovers <I> = 1
#[test]
fn dmrg_identity_mpo_energy_one() {
    let mut rng = LcgRng::new(11);
    let mpo = Mpo::identity(3, 2).expect("ok");
    let init = Mps::random_mps(3, 2, 3, &mut rng).expect("ok");
    let cfg = DmrgConfig {
        max_sweeps: 1,
        chi_max: 4,
        ..DmrgConfig::default()
    };
    let r = dmrg_two_site(&mpo, init, cfg, &mut rng).expect("ok");
    assert!((r.energy - 1.0).abs() < 1.0e-5, "energy = {}", r.energy);
}

// 6. TEBD identity gate over T=1 preserves norm
#[test]
fn tebd_identity_gate_conserves_norm() {
    use crate::tebd::tebd::{TebdConfig, apply_two_site_gate};
    let local = vec![vec![0.6, 0.8]; 4];
    let mut mps = Mps::from_product_state(&local).expect("ok");
    let d = 2;
    let mut id_gate = vec![0.0; d * d * d * d];
    for p1 in 0..d {
        for p2 in 0..d {
            id_gate[((p1 * d + p2) * d + p1) * d + p2] = 1.0;
        }
    }
    let n_before = mps.norm_squared().expect("ok");
    for _ in 0..10 {
        for s in 0..3 {
            apply_two_site_gate(&mut mps, s, &id_gate, TebdConfig::default()).expect("ok");
        }
    }
    let n_after = mps.norm_squared().expect("ok");
    assert!((n_before - n_after).abs() < 1.0e-6);
}

// 7. TT-SVD round-trip reconstructs full tensor
#[test]
fn tt_svd_roundtrip() {
    let mut rng = LcgRng::new(11);
    let dims = vec![3, 4, 2];
    let total: usize = dims.iter().product();
    let data: Vec<f64> = (0..total).map(|_| rng.next_normal()).collect();
    let tt = tt_svd(&data, &dims, 24, 1.0e-14).expect("ok");
    let rec = tt.reconstruct().expect("ok");
    assert!(fro_diff(&data, &rec) < 1e-7);
}

// 8. HOSVD reconstructs full 3-tensor with full ranks
#[test]
fn hosvd_full_rank_reconstruction() {
    let mut rng = LcgRng::new(19);
    let d0 = 3;
    let d1 = 4;
    let d2 = 2;
    let data: Vec<f64> = (0..d0 * d1 * d2).map(|_| rng.next_normal()).collect();
    let res = hosvd(&data, d0, d1, d2, d0, d1, d2).expect("ok");
    let rec = tucker_reconstruct(&res);
    assert!(fro_diff(&data, &rec) < 1.0e-8);
}

// 9. CP/ALS converges on a rank-1 input
#[test]
fn cp_als_rank1_converges() {
    let a = [1.0, 2.0, 3.0];
    let b = [0.5, 4.0];
    let c = [1.0, 0.5, 2.0];
    let d0 = 3;
    let d1 = 2;
    let d2 = 3;
    let mut data = vec![0.0; d0 * d1 * d2];
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                data[(i * d1 + j) * d2 + k] = a[i] * b[j] * c[k];
            }
        }
    }
    let mut rng = LcgRng::new(11);
    let res = cp_als(&data, d0, d1, d2, 1, 200, 1e-12, &mut rng).expect("ok");
    assert!(res.residual < 1e-5);
}

// 10. Einsum binary matmul matches direct computation
#[test]
fn einsum_matmul_matches_loop() {
    let a = LabelledTensor::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        vec!['i', 'j'],
    )
    .expect("ok");
    let b = LabelledTensor::new(
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![3, 2],
        vec!['j', 'k'],
    )
    .expect("ok");
    let c = einsum_binary(&a, &b).expect("ok");
    let manual = vec![
        1.0 * 7.0 + 2.0 * 9.0 + 3.0 * 11.0,
        1.0 * 8.0 + 2.0 * 10.0 + 3.0 * 12.0,
        4.0 * 7.0 + 5.0 * 9.0 + 6.0 * 11.0,
        4.0 * 8.0 + 5.0 * 10.0 + 6.0 * 12.0,
    ];
    for (got, expect) in c.data.iter().zip(&manual) {
        assert!((got - expect).abs() < 1e-12);
    }
}

// 11. Entanglement entropy of singlet (|01> - |10>)/sqrt(2) = ln 2
#[test]
fn entanglement_entropy_singlet_ln2() {
    // Singlet state on 2 qubits.
    // Site 0: tensor of shape (1, 2, 2): (1/sqrt(2)) * [[[0, 1]], [[-1, 0]]]
    // Site 1: tensor of shape (2, 2, 1): [[[1], [0]], [[0], [1]]]
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let t0 = MpsTensor::new(1, 2, 2, vec![0.0, inv_sqrt2, -inv_sqrt2, 0.0]).expect("ok");
    let t1 = MpsTensor::new(2, 2, 1, vec![1.0, 0.0, 0.0, 1.0]).expect("ok");
    let mps = Mps::from_tensors(vec![t0, t1]).expect("ok");
    let h = entanglement_entropy(&mps, 0).expect("ok");
    assert!((h - 2.0_f64.ln()).abs() < 1e-7, "entropy = {h}");
}

// 12. Lanczos recovers smallest eigenvalue of a 5×5 symmetric matrix
#[test]
fn lanczos_recovers_smallest_eigenvalue() {
    let n = 5;
    let h = {
        // Build a known symmetric matrix with spectrum [1, 2, 3, 4, 5]
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            m[i * n + i] = (i + 1) as f64;
        }
        m
    };
    let apply = |v: &[f64]| {
        let mut out = vec![0.0; n];
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..n {
                acc += h[i * n + j] * v[j];
            }
            out[i] = acc;
        }
        out
    };
    let v0 = vec![1.0; n];
    let r = lanczos_smallest(apply, n, &v0, n, 1e-13).expect("ok");
    assert!((r.eigenvalue - 1.0).abs() < 1e-8);
}

// 13. SVD round-trip: U·diag(s)·V^T = A
#[test]
fn svd_roundtrip_random() {
    let mut rng = LcgRng::new(5);
    let m = 5;
    let n = 4;
    let mat: Vec<f64> = (0..m * n).map(|_| rng.next_normal()).collect();
    let r = svd_jacobi(&mat, m, n).expect("ok");
    let mut rec = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for c in 0..r.k {
                acc += r.u[i * r.k + c] * r.s[c] * r.vt[c * n + j];
            }
            rec[i * n + j] = acc;
        }
    }
    assert!(fro_diff(&mat, &rec) < 1e-9);
}

// 14. Greedy contraction path returns the same answer as manual chain
#[test]
fn greedy_path_executes() {
    let a = LabelledTensor::new(vec![1.0; 6], vec![2, 3], vec!['a', 'b']).expect("ok");
    let b = LabelledTensor::new(vec![1.0; 12], vec![3, 4], vec!['b', 'c']).expect("ok");
    let c = LabelledTensor::new(vec![1.0; 8], vec![4, 2], vec!['c', 'd']).expect("ok");
    let path = greedy_path(&[a.clone(), b.clone(), c.clone()]).expect("ok");
    let result = execute_path(vec![a, b, c], &path).expect("ok");
    assert_eq!(result.dims, vec![2, 2]);
    // Each entry should equal 3*4 = 12 (sum of ones across the contracted axes b,c)
    for v in &result.data {
        assert!((v - 12.0).abs() < 1e-9);
    }
}

// 15. PTX kernel strings non-empty across 6 SM versions × 7 kernels
#[test]
fn ptx_kernels_all_sm_versions() {
    type KFn = fn(u32) -> String;
    let kernels: &[(&str, KFn)] = &[
        ("tensor_contract", tensor_contract_ptx),
        ("svd_jacobi_step", svd_jacobi_step_ptx),
        ("dmrg_local_apply", dmrg_local_apply_ptx),
        ("mpo_apply", mpo_apply_ptx),
        ("trotter_step", trotter_step_ptx),
        ("hosvd_unfold", hosvd_unfold_ptx),
        ("tt_round", tt_round_ptx),
    ];
    let sms = [75u32, 80, 86, 89, 90, 100];
    for sm in sms {
        for (name, f) in kernels {
            let s = f(sm);
            assert!(!s.is_empty(), "kernel {name} sm={sm} empty");
            assert!(s.contains(".visible .entry"));
        }
    }
}

// 16. Fidelity of an MPS with itself is 1
#[test]
fn fidelity_self_one() {
    let mut rng = LcgRng::new(3);
    let mps = Mps::random_mps(3, 2, 2, &mut rng).expect("ok");
    let f = fidelity(&mps, &mps).expect("ok");
    assert!((f - 1.0).abs() < 1e-9);
}

// 17. MPO expectation of identity MPO equals norm-ratio (which is 1)
#[test]
fn mpo_expectation_identity() {
    let local = vec![vec![0.6, 0.8]; 3];
    let mps = Mps::from_product_state(&local).expect("ok");
    let mpo = Mpo::identity(3, 2).expect("ok");
    let e = mpo_expectation(&mpo, &mps).expect("ok");
    assert!((e - 1.0).abs() < 1e-9);
}

// 18. Pipeline: random tensor → HOSVD with truncated ranks → reconstruction error finite
#[test]
fn hosvd_truncated_finite_error() {
    let mut rng = LcgRng::new(101);
    let d0 = 4;
    let d1 = 4;
    let d2 = 4;
    let data: Vec<f64> = (0..d0 * d1 * d2).map(|_| rng.next_normal()).collect();
    let res = hosvd(&data, d0, d1, d2, 2, 2, 2).expect("ok");
    let rec = tucker_reconstruct(&res);
    let err = fro_diff(&data, &rec);
    assert!(err.is_finite());
}
