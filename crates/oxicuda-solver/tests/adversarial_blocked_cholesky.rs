//! Independent adversarial verification of the blocked (n > 64) Cholesky path.
//!
//! This test is intentionally NOT derived from the in-crate tests. It uses its
//! own RNG (splitmix64), its own host reference factorization, and its own
//! reconstruction / solve residual checks. For each tested size it builds a
//! fixed SPD matrix `A = M·Mᵀ + n·I` (column-major), factors it through the
//! public `cholesky` entry point, and verifies BOTH:
//!   * `A ≈ L·Lᵀ`  (factorization correctness), and
//!   * `A·x̂ ≈ b`   (solve correctness) for a known `x`, single- and 2-RHS.

use std::sync::Arc;

use oxicuda_blas::types::FillMode;
use oxicuda_driver::Context;
use oxicuda_memory::DeviceBuffer;
use oxicuda_solver::SolverHandle;
use oxicuda_solver::dense::{cholesky, cholesky_solve};

/// splitmix64: a stream distinct from the in-crate LCG, so the random `M` here
/// is genuinely independent of the implementation's own test fixtures.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    fn next_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11; // 53 mantissa bits
        (bits as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// Build a FIXED, well-conditioned SPD matrix `A = M·Mᵀ + n·I`, column-major.
/// `a[col * n + row]` is `A[row, col]`.
fn build_spd(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = SplitMix64::new(seed);
    let mut m = vec![0.0_f64; n * n];
    for v in m.iter_mut() {
        *v = rng.next_unit();
    }
    let mut a = vec![0.0_f64; n * n];
    for j in 0..n {
        for i in 0..n {
            // (M·Mᵀ)[i,j] = sum_k M[i,k]·M[j,k]; M[i,k] = m[k*n + i].
            let mut s = 0.0_f64;
            for k in 0..n {
                s += m[k * n + i] * m[k * n + j];
            }
            if i == j {
                s += n as f64;
            }
            a[j * n + i] = s;
        }
    }
    a
}

/// `y = A · x` for column-major `A` (`a[col*n + row] == A[row, col]`).
fn matvec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    for j in 0..n {
        let xj = x[j];
        for i in 0..n {
            y[i] += a[j * n + i] * xj;
        }
    }
    y
}

/// Max abs reconstruction error over the full symmetric matrix using the
/// lower-triangular factor `l` (column-major; strict upper ignored).
fn max_recon_error(a: &[f64], l: &[f64], n: usize) -> f64 {
    let mut worst = 0.0_f64;
    for j in 0..n {
        for i in 0..n {
            // (L·Lᵀ)[i,j] = sum_{k <= min(i,j)} L[i,k]·L[j,k].
            let kmax = i.min(j);
            let recon: f64 = (0..=kmax).map(|k| l[k * n + i] * l[k * n + j]).sum();
            let err = (recon - a[j * n + i]).abs();
            if err > worst {
                worst = err;
            }
        }
    }
    worst
}

fn handle() -> Option<(Arc<Context>, SolverHandle)> {
    if oxicuda_driver::init().is_err() {
        eprintln!("skipping: CUDA driver unavailable");
        return None;
    }
    let has = oxicuda_driver::device::Device::count()
        .map(|c| c > 0)
        .unwrap_or(false);
    if !has {
        eprintln!("skipping: no NVIDIA device");
        return None;
    }
    let dev = oxicuda_driver::device::Device::get(0).expect("device 0");
    let ctx = Arc::new(Context::new(&dev).expect("context"));
    let h = SolverHandle::new(&ctx).expect("solver handle");
    Some((ctx, h))
}

/// Factor + reconstruct + solve (single & 2-RHS), returning the worst absolute
/// error observed across all three checks for this `n`.
fn verify_size(h: &mut SolverHandle, n: usize, recon_tol: f64, solve_tol: f64) -> f64 {
    let a = build_spd(
        n,
        0xDEAD_BEEF_CAFE_F00D ^ (n as u64).wrapping_mul(0x100000001B3),
    );

    // ---- Factor through the PUBLIC entry point (lower). ----
    let mut d_a = DeviceBuffer::from_host(&a).expect("upload A");
    cholesky::<f64>(h, FillMode::Lower, &mut d_a, n as u32, n as u32).expect("device cholesky");
    let mut l = vec![0.0_f64; n * n];
    d_a.copy_to_host(&mut l).expect("download factor");

    // ---- Check A ≈ L·Lᵀ. ----
    let recon_err = max_recon_error(&a, &l, n);
    assert!(
        recon_err < recon_tol,
        "n={n}: reconstruction max-abs error {recon_err:e} exceeds tol {recon_tol:e}"
    );

    // ---- Solve check: known x, b = A·x, then solve A·x̂ = b. ----
    // Single RHS.
    let x_known: Vec<f64> = (0..n).map(|i| 1.0 + 0.37 * (i as f64).sin()).collect();
    let b1 = matvec(&a, &x_known, n);
    let mut d_b = DeviceBuffer::from_host(&b1).expect("upload b1");
    cholesky_solve::<f64>(h, FillMode::Lower, &d_a, &mut d_b, n as u32, 1).expect("solve 1rhs");
    let mut x_hat = vec![0.0_f64; n];
    d_b.copy_to_host(&mut x_hat).expect("download x1");
    // Residual A·x̂ ≈ b.
    let ax = matvec(&a, &x_hat, n);
    let mut solve_err = 0.0_f64;
    for i in 0..n {
        let e = (ax[i] - b1[i]).abs();
        if e > solve_err {
            solve_err = e;
        }
    }
    // Also direct solution accuracy x̂ ≈ x.
    let mut sol_acc = 0.0_f64;
    for i in 0..n {
        let e = (x_hat[i] - x_known[i]).abs();
        if e > sol_acc {
            sol_acc = e;
        }
    }
    assert!(
        solve_err < solve_tol,
        "n={n}: single-RHS residual |A·x̂-b| {solve_err:e} exceeds tol {solve_tol:e}"
    );
    assert!(
        sol_acc < solve_tol,
        "n={n}: single-RHS solution |x̂-x| {sol_acc:e} exceeds tol {solve_tol:e}"
    );

    // 2-RHS (column-major n x 2): x columns are x_known and a reversed variant.
    let mut x2 = vec![0.0_f64; n * 2];
    for i in 0..n {
        x2[i] = x_known[i];
        x2[n + i] = 2.0 - 0.21 * (i as f64).cos();
    }
    let mut b2 = vec![0.0_f64; n * 2];
    for c in 0..2 {
        let bc = matvec(&a, &x2[c * n..(c + 1) * n], n);
        b2[c * n..(c + 1) * n].copy_from_slice(&bc);
    }
    let mut d_b2 = DeviceBuffer::from_host(&b2).expect("upload b2");
    cholesky_solve::<f64>(h, FillMode::Lower, &d_a, &mut d_b2, n as u32, 2).expect("solve 2rhs");
    let mut xh2 = vec![0.0_f64; n * 2];
    d_b2.copy_to_host(&mut xh2).expect("download x2");
    for c in 0..2 {
        let ax = matvec(&a, &xh2[c * n..(c + 1) * n], n);
        for i in 0..n {
            let e = (ax[i] - b2[c * n + i]).abs();
            assert!(
                e < solve_tol,
                "n={n} rhs{c}: 2-RHS residual {e:e} exceeds tol {solve_tol:e}"
            );
            if e > solve_err {
                solve_err = e;
            }
        }
    }

    let worst = recon_err.max(solve_err).max(sol_acc);
    println!(
        "n={n:>4}: recon={recon_err:.3e} solve_resid={solve_err:.3e} sol_acc={sol_acc:.3e} worst={worst:.3e}"
    );
    worst
}

#[test]
fn adversarial_blocked_cholesky_factor_and_solve() {
    let Some((_ctx, mut h)) = handle() else {
        return;
    };
    // Tolerances: reconstruction is a backward-error quantity ~ n·eps·||A||,
    // with ||A|| ~ 2n here, so allow n·1e-11 (far below 1e-8 even at n=256).
    // The forward solve loses a few more digits through two triangular solves;
    // 1e-8 is the headline tolerance, loosened to 5e-8 at the largest size.
    let mut overall_worst = 0.0_f64;
    for &n in &[65usize, 96, 128, 192, 256] {
        let recon_tol = (n as f64) * 1e-11;
        let solve_tol = if n >= 192 { 5e-8 } else { 1e-8 };
        let w = verify_size(&mut h, n, recon_tol, solve_tol);
        if w > overall_worst {
            overall_worst = w;
        }
    }
    println!("overall worst abs error across all sizes = {overall_worst:.3e}");
}

#[test]
fn adversarial_boundary_64_vs_65() {
    let Some((_ctx, mut h)) = handle() else {
        return;
    };
    // n=64 exercises the single-block (unblocked) path; n=65 is the first size
    // that triggers the blocked path (one trailing block of size 1). Both must
    // reconstruct and solve to the same tight tolerance.
    let w64 = verify_size(&mut h, 64, 64.0 * 1e-11, 1e-8);
    let w65 = verify_size(&mut h, 65, 65.0 * 1e-11, 1e-8);
    println!("boundary worst: n=64 -> {w64:.3e}, n=65 -> {w65:.3e}");
}
