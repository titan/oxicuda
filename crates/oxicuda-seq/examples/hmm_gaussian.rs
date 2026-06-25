//! Worked example: recover Gaussian-HMM emission parameters from synthetic data.
//!
//! Synthesises observations from a known 3-state scalar Gaussian HMM using the
//! crate's seeded `LcgRng`, fits the model with `baum_welch_gaussian` (EM), then
//! prints recovered vs. true means / std-devs (matched by nearest mean, since
//! label switching is expected) together with the per-iteration log-likelihood.
//!
//! Run with: `cargo run -p oxicuda-seq --example hmm_gaussian`

// Numerical kernels here index parallel arrays by state, mirroring the crate's
// lib-level stance (see `lib.rs`); examples do not inherit that crate attribute.
#![allow(clippy::needless_range_loop)]

use oxicuda_seq::LcgRng;
use oxicuda_seq::hmm::{HmmGaussian, baum_welch_gaussian};

fn main() {
    // --- Ground-truth 3-state Gaussian HMM (scalar emissions, 6σ apart) ---
    let n = 3usize;
    let pi = [1.0, 0.0, 0.0];
    let a = [0.7, 0.2, 0.1, 0.1, 0.7, 0.2, 0.2, 0.1, 0.7];
    let true_means = [-6.0, 0.0, 6.0];
    let true_sigmas = [1.0, 1.0, 1.0];

    // --- Synthesise T observations with the seeded LCG RNG ---
    let t_max = 2000usize;
    let mut rng = LcgRng::new(0x5EED_1234);
    let mut x = Vec::with_capacity(t_max);
    let mut state = rng.sample_categorical(&pi);
    for _ in 0..t_max {
        x.push(true_means[state] + true_sigmas[state] * rng.next_normal());
        state = rng.sample_categorical(&a[state * n..state * n + n]);
    }

    // --- Fit from a reasonable (distinct-mean) initialisation ---
    let init = HmmGaussian::new(
        n,
        1,
        vec![1.0 / 3.0; 3],
        vec![1.0 / 3.0; 9],
        vec![-5.0, 0.5, 5.0],
        vec![1.5, 1.5, 1.5],
    )
    .expect("valid init HMM");
    let result = baum_welch_gaussian(&init, &x, 100, 1e-7).expect("EM fit");

    // --- Report recovered vs. true parameters (match by nearest mean) ---
    println!("Gaussian-HMM Baum-Welch (EM): {t_max} samples, {n} states");
    println!(
        "converged = {} after {} iterations\n",
        result.converged, result.iterations
    );
    println!("state | true mu | rec. mu | true sigma | rec. sigma");
    println!("------+---------+---------+------------+-----------");
    for s in 0..n {
        // Nearest recovered state to this true mean (handles label switching).
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for k in 0..n {
            let d = (result.model.means[k] - true_means[s]).abs();
            if d < best_d {
                best_d = d;
                best = k;
            }
        }
        let rec_mu = result.model.means[best];
        let rec_sigma = result.model.vars[best].sqrt();
        println!(
            "  {s}   | {:+7.3} | {:+7.3} | {:10.3} | {:10.3}",
            true_means[s], rec_mu, true_sigmas[s], rec_sigma
        );
    }

    // --- Log-likelihood trace (non-decreasing under EM) ---
    println!("\nper-iteration log-likelihood (monotone non-decreasing):");
    for (it, ll) in result.log_likelihoods.iter().enumerate() {
        println!("  iter {it:2}: {ll:.4}");
    }
}
