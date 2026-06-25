//! Worked example: HOSVD compression of a Tucker-structured 3-way tensor.
//!
//! Synthesises a tensor `T` that is exactly Tucker-rank `(r0, r1, r2)` by contracting
//! a small random core with three random orthonormal factor matrices, then recovers it
//! with the Higher-Order SVD. Two HOSVD calls illustrate the central trade-off:
//!
//! 1. a *full-rank* HOSVD reconstructs `T` to machine precision (round-trip check);
//! 2. a *truncated* HOSVD at the true multilinear rank compresses `T` losslessly,
//!    while truncating below the true rank introduces a controlled, finite error.
//!
//! Run with:
//! ```text
//! cargo run -p oxicuda-tn --example hosvd_compression
//! ```

use oxicuda_tn::handle::LcgRng;
use oxicuda_tn::tucker::{hosvd, tucker_reconstruct};

/// Build a column-orthonormal `(rows × cols)` matrix (rows ≥ cols) by Gram-Schmidt on
/// random columns; row-major.
fn random_orthonormal(rows: usize, cols: usize, rng: &mut LcgRng) -> Vec<f64> {
    let mut q = vec![0.0_f64; rows * cols];
    for c in 0..cols {
        // Start from a random column.
        let mut v: Vec<f64> = (0..rows).map(|_| rng.next_normal()).collect();
        // Orthogonalise against earlier columns.
        for prev in 0..c {
            let mut dot = 0.0_f64;
            for r in 0..rows {
                dot += q[r * cols + prev] * v[r];
            }
            for r in 0..rows {
                v[r] -= dot * q[r * cols + prev];
            }
        }
        let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
        for r in 0..rows {
            q[r * cols + c] = v[r] / nrm;
        }
    }
    q
}

/// Tucker contraction `T[i,j,k] = Σ_{a,b,c} S[a,b,c] U0[i,a] U1[j,b] U2[k,c]`.
fn tucker_build(
    core: &[f64],
    r: (usize, usize, usize),
    u0: &[f64],
    u1: &[f64],
    u2: &[f64],
    d: (usize, usize, usize),
) -> Vec<f64> {
    let (r0, r1, r2) = r;
    let (d0, d1, d2) = d;
    let mut t = vec![0.0_f64; d0 * d1 * d2];
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                let mut acc = 0.0_f64;
                for a in 0..r0 {
                    for b in 0..r1 {
                        for c in 0..r2 {
                            acc += core[(a * r1 + b) * r2 + c]
                                * u0[i * r0 + a]
                                * u1[j * r1 + b]
                                * u2[k * r2 + c];
                        }
                    }
                }
                t[(i * d1 + j) * d2 + k] = acc;
            }
        }
    }
    t
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (d0, d1, d2) = (20usize, 18usize, 16usize);
    let (r0, r1, r2) = (4usize, 3usize, 5usize); // true multilinear rank

    let mut rng = LcgRng::new(20240621);
    let core: Vec<f64> = (0..r0 * r1 * r2).map(|_| rng.next_normal()).collect();
    let u0 = random_orthonormal(d0, r0, &mut rng);
    let u1 = random_orthonormal(d1, r1, &mut rng);
    let u2 = random_orthonormal(d2, r2, &mut rng);

    let t = tucker_build(&core, (r0, r1, r2), &u0, &u1, &u2, (d0, d1, d2));
    let t_norm = t.iter().map(|x| x * x).sum::<f64>().sqrt();

    println!("Tucker-structured tensor T of shape ({d0}, {d1}, {d2})");
    println!("  true multilinear rank: ({r0}, {r1}, {r2})");
    println!("  ||T||_F = {t_norm:.6}");

    // (1) Full-rank HOSVD: lossless round trip.
    let full = hosvd(&t, d0, d1, d2, d0, d1, d2)?;
    let rec_full = tucker_reconstruct(&full);
    println!(
        "  full-rank HOSVD round-trip max|ΔT| = {:.3e}",
        max_abs_diff(&t, &rec_full)
    );

    // (2) Truncated HOSVD at the true rank: still lossless (T lives in that subspace).
    let exact = hosvd(&t, d0, d1, d2, r0, r1, r2)?;
    let rec_exact = tucker_reconstruct(&exact);
    let core_entries = r0 * r1 * r2;
    let dense_entries = d0 * d1 * d2;
    println!(
        "  rank-({r0},{r1},{r2}) HOSVD max|ΔT| = {:.3e}   (compressed core: {core_entries} vs dense {dense_entries})",
        max_abs_diff(&t, &rec_exact)
    );

    // (3) Over-truncation below the true rank: controlled, finite error.
    let lossy = hosvd(&t, d0, d1, d2, r0 - 1, r1, r2)?;
    let rec_lossy = tucker_reconstruct(&lossy);
    let rel = max_abs_diff(&t, &rec_lossy) / t_norm.max(1e-300);
    println!(
        "  over-truncated (rank {}) relative error = {rel:.3e}",
        r0 - 1
    );

    Ok(())
}
