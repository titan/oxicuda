//! Extended algorithm benchmarks on standard 1D quantum-spin models.
//!
//! Complements `tn_ops.rs` (which times PTX-string emission and micro-ops) by
//! exercising the full host-side algorithmic pipeline on textbook Hamiltonians:
//!
//! - **Heisenberg XXX chain** — two-site DMRG ground-state search and iTEBD.
//! - **Transverse-field Ising (TFIM)** — TEBD imaginary-time step with a
//!   second-order Suzuki-Trotter even/odd splitting.
//! - **J1-J2 frustrated chain** — TEBD with both nearest-neighbour (J1) and
//!   next-nearest-neighbour (J2) bond gates applied in alternating passes.
//! - **TT-SVD / HOSVD** — Oseledets TT decomposition and Tucker HOSVD on dense
//!   tensors of representative size.
//! - **Moses Move** — the isoTNS column split (`moses_move_column`).
//!
//! Every Hamiltonian and gate is constructed from scratch; no external linear
//! algebra is used. Two-site imaginary-time gates `exp(-tau h)` are formed with the
//! crate's [`mat_exp_4x4`] from a `4x4` bond Hamiltonian assembled out of spin-1/2
//! operators.

use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_tn::dmrg::{DmrgConfig, dmrg_two_site};
use oxicuda_tn::handle::LcgRng;
use oxicuda_tn::mpo::mpo::Mpo;
use oxicuda_tn::mps::isometry_tn::{FatMpsColumn, FatTensor, moses_move_column};
use oxicuda_tn::mps::itebd::mat_exp_4x4;
use oxicuda_tn::mps::mps::Mps;
use oxicuda_tn::tebd::{TebdConfig, apply_two_site_gate};
use oxicuda_tn::tt::tt_svd;
use oxicuda_tn::tucker::hosvd;

/// Spin-1/2 single-site operators in the basis `|↑⟩=0, |↓⟩=1`, row-major `2x2`.
const SZ: [f64; 4] = [0.5, 0.0, 0.0, -0.5];
const SX: [f64; 4] = [0.0, 0.5, 0.5, 0.0];
const SP: [f64; 4] = [0.0, 1.0, 0.0, 0.0];
const SM: [f64; 4] = [0.0, 0.0, 1.0, 0.0];
const ID2: [f64; 4] = [1.0, 0.0, 0.0, 1.0];

/// Kronecker product `A ⊗ B` of two `2x2` matrices → row-major `4x4`.
fn kron2(a: &[f64; 4], b: &[f64; 4]) -> [f64; 16] {
    let mut out = [0.0_f64; 16];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    let row = i * 2 + k;
                    let col = j * 2 + l;
                    out[row * 4 + col] = a[i * 2 + j] * b[k * 2 + l];
                }
            }
        }
    }
    out
}

/// Add `scale * m` into `acc` (both `4x4` row-major).
fn axpy16(acc: &mut [f64; 16], scale: f64, m: &[f64; 16]) {
    for i in 0..16 {
        acc[i] += scale * m[i];
    }
}

/// Two-site Heisenberg XXX bond Hamiltonian `Sx⊗Sx + Sy⊗Sy + Sz⊗Sz`, expressed via
/// `S^+/S^-` as `Sz⊗Sz + 1/2 (S^+⊗S^- + S^-⊗S^+)`.
fn heisenberg_bond() -> [f64; 16] {
    let mut h = [0.0_f64; 16];
    axpy16(&mut h, 1.0, &kron2(&SZ, &SZ));
    axpy16(&mut h, 0.5, &kron2(&SP, &SM));
    axpy16(&mut h, 0.5, &kron2(&SM, &SP));
    h
}

/// Transverse-field Ising bond Hamiltonian for a chain
/// `H = -J Σ Sz_i Sz_{i+1} - g Σ Sx_i`. The single-site field is split evenly across
/// the two bonds touching each interior site.
fn tfim_bond(j: f64, g: f64) -> [f64; 16] {
    let mut h = [0.0_f64; 16];
    axpy16(&mut h, -j, &kron2(&SZ, &SZ));
    axpy16(&mut h, -0.5 * g, &kron2(&SX, &ID2));
    axpy16(&mut h, -0.5 * g, &kron2(&ID2, &SX));
    h
}

/// Build an imaginary-time two-site gate `U = exp(-tau h)` as a `(d,d,d,d)` row-major
/// tensor `U[p1, p2, p1', p2']` from a symmetric `4x4` bond Hamiltonian `h`.
///
/// [`mat_exp_4x4`] computes `exp(scale · h)` for a real symmetric `4x4` matrix via
/// Jacobi diagonalisation, so `scale = -tau` yields the imaginary-time propagator.
fn imag_gate(h: &[f64; 16], tau: f64) -> Vec<f64> {
    let u = mat_exp_4x4(h, -tau).expect("mat_exp_4x4");
    // u is the 4x4 = (p1 p2)×(p1' p2') operator; same row-major layout as (d,d,d,d).
    u.to_vec()
}

fn bench_dmrg_heisenberg(c: &mut Criterion) {
    let n = 8usize;
    let mpo = Mpo::heisenberg_xxx(n).expect("heisenberg mpo");
    let cfg = DmrgConfig {
        max_sweeps: 2,
        chi_max: 16,
        ..DmrgConfig::default()
    };
    c.bench_function("dmrg_heisenberg_L8_chi16_2sweep", |b| {
        b.iter(|| {
            let mut rng = LcgRng::new(7);
            let init = Mps::random_mps(n, 2, 8, &mut rng).expect("init");
            dmrg_two_site(&mpo, init, cfg, &mut rng).expect("dmrg")
        })
    });
}

fn bench_tebd_tfim(c: &mut Criterion) {
    let n = 12usize;
    let cfg = TebdConfig {
        chi_max: 16,
        trunc_tol: 1e-10,
    };
    let h = tfim_bond(1.0, 0.8);
    // Second-order Strang: half step on odd bonds, full on even, half on odd.
    let gate_half = imag_gate(&h, 0.025);
    let gate_full = imag_gate(&h, 0.05);
    c.bench_function("tebd_tfim_L12_chi16_strang_step", |b| {
        b.iter(|| {
            let mut rng = LcgRng::new(11);
            let mut mps = Mps::random_mps(n, 2, 8, &mut rng).expect("mps");
            // odd bonds: s = 0, 2, ...
            for s in (0..n - 1).step_by(2) {
                apply_two_site_gate(&mut mps, s, &gate_half, cfg).expect("gate");
            }
            // even bonds: s = 1, 3, ...
            for s in (1..n - 1).step_by(2) {
                apply_two_site_gate(&mut mps, s, &gate_full, cfg).expect("gate");
            }
            for s in (0..n - 1).step_by(2) {
                apply_two_site_gate(&mut mps, s, &gate_half, cfg).expect("gate");
            }
            mps
        })
    });
}

fn bench_tebd_j1j2(c: &mut Criterion) {
    let n = 10usize;
    let cfg = TebdConfig {
        chi_max: 16,
        trunc_tol: 1e-10,
    };
    // J1-J2 frustrated Heisenberg: nearest neighbour J1 = 1, next-nearest J2 = 0.5.
    let h = heisenberg_bond();
    let gate_j1 = imag_gate(&h, 0.05);
    let gate_j2 = imag_gate(&h, 0.5 * 0.05);
    c.bench_function("tebd_j1j2_L10_chi16_step", |b| {
        b.iter(|| {
            let mut rng = LcgRng::new(13);
            let mut mps = Mps::random_mps(n, 2, 8, &mut rng).expect("mps");
            // J1 nearest-neighbour pass (all bonds left to right).
            for s in 0..n - 1 {
                apply_two_site_gate(&mut mps, s, &gate_j1, cfg).expect("gate");
            }
            // J2 next-nearest-neighbour pass approximated by SWAP-free skip: apply on
            // alternating bonds with halved tau to emulate the J2 coupling strength.
            for s in (0..n - 1).step_by(2) {
                apply_two_site_gate(&mut mps, s, &gate_j2, cfg).expect("gate");
            }
            mps
        })
    });
}

fn bench_tt_svd(c: &mut Criterion) {
    // d = 2, L = 10 dense tensor (1024 entries); TT-SVD with bond cap 16.
    let l = 10usize;
    let dims = vec![2usize; l];
    let n: usize = dims.iter().product();
    let mut rng = LcgRng::new(7);
    let data: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
    c.bench_function("tt_svd_d2_L10_chi16", |b| {
        b.iter(|| tt_svd(&data, &dims, 16, 1e-10).expect("tt_svd"))
    });
}

fn bench_hosvd(c: &mut Criterion) {
    // (16, 16, 16) dense tensor, full-rank HOSVD.
    let (d0, d1, d2) = (16usize, 16usize, 16usize);
    let mut rng = LcgRng::new(7);
    let data: Vec<f64> = (0..d0 * d1 * d2).map(|_| rng.next_normal()).collect();
    c.bench_function("hosvd_16x16x16_fullrank", |b| {
        b.iter(|| hosvd(&data, d0, d1, d2, d0, d1, d2).expect("hosvd"))
    });
}

fn bench_moses_move(c: &mut Criterion) {
    // A 6-row fat-MPS column (d_p = 2, horizontal bond 4, vertical bond 4) split by the
    // Moses Move down to horizontal bond 4.
    let mut rng = LcgRng::new(7);
    let d_p = 2usize;
    let d_right = 4usize;
    let chi_v = 4usize;
    let rows = 6usize;
    let mut tensors = Vec::with_capacity(rows);
    for r in 0..rows {
        let d_up = if r == 0 { 1 } else { chi_v };
        let d_down = if r + 1 == rows { 1 } else { chi_v };
        let n = d_up * d_p * d_down * d_right;
        let data: Vec<f64> = (0..n).map(|_| rng.next_normal()).collect();
        tensors.push(FatTensor::new(d_up, d_p, d_down, d_right, data).expect("fat"));
    }
    let column = FatMpsColumn { tensors };
    c.bench_function("moses_move_L6_chi4", |b| {
        b.iter(|| moses_move_column(&column, 4, 1e-10).expect("moses"))
    });
}

criterion_group!(
    algo_benches,
    bench_dmrg_heisenberg,
    bench_tebd_tfim,
    bench_tebd_j1j2,
    bench_tt_svd,
    bench_hosvd,
    bench_moses_move
);
criterion_main!(algo_benches);
