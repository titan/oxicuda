//! Worked example: TEBD quench dynamics of the transverse-field Ising model (TFIM).
//!
//! Prepares the fully-polarised product state `|↑↑…↑⟩` and evolves it in imaginary
//! time under
//!
//! ```text
//!     H = -J Σ_i Sz_i Sz_{i+1} - g Σ_i Sx_i
//! ```
//!
//! using a second-order Suzuki-Trotter (Strang) TEBD step. The transverse
//! magnetisation `⟨Sx⟩` and the longitudinal `⟨Sz⟩` are tracked as the state relaxes
//! toward the ground state of the quenched Hamiltonian — the canonical demonstration
//! of TEBD time evolution. (Imaginary time is used because the crate's real-valued
//! gate exponential targets Hermitian generators; the same machinery drives real-time
//! quenches once complex gates are supplied.)
//!
//! Run with:
//! ```text
//! cargo run -p oxicuda-tn --example tebd_tfim_quench
//! ```

use oxicuda_tn::mps::itebd::mat_exp_4x4;
use oxicuda_tn::mps::mps::Mps;
use oxicuda_tn::tebd::{TebdConfig, apply_two_site_gate};

/// Spin-1/2 operators, basis `|↑⟩=0, |↓⟩=1`, row-major `2x2`.
const SZ: [f64; 4] = [0.5, 0.0, 0.0, -0.5];
const SX: [f64; 4] = [0.0, 0.5, 0.5, 0.0];
const ID2: [f64; 4] = [1.0, 0.0, 0.0, 1.0];

fn kron2(a: &[f64; 4], b: &[f64; 4]) -> [f64; 16] {
    let mut out = [0.0_f64; 16];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    out[(i * 2 + k) * 4 + (j * 2 + l)] = a[i * 2 + j] * b[k * 2 + l];
                }
            }
        }
    }
    out
}

/// TFIM bond Hamiltonian `-J Sz⊗Sz - g/2 (Sx⊗I + I⊗Sx)` (field split across bonds).
fn tfim_bond(j: f64, g: f64) -> [f64; 16] {
    let zz = kron2(&SZ, &SZ);
    let xi = kron2(&SX, &ID2);
    let ix = kron2(&ID2, &SX);
    let mut h = [0.0_f64; 16];
    for idx in 0..16 {
        h[idx] = -j * zz[idx] - 0.5 * g * (xi[idx] + ix[idx]);
    }
    h
}

/// Imaginary-time gate `exp(-tau h)` as a `(d,d,d,d)` tensor.
fn imag_gate(h: &[f64; 16], tau: f64) -> Vec<f64> {
    mat_exp_4x4(h, -tau).expect("mat_exp_4x4").to_vec()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 12usize;
    let j = 1.0;
    let g = 1.0; // critical point of the infinite TFIM
    let cfg = TebdConfig {
        chi_max: 24,
        trunc_tol: 1e-12,
    };

    // Initial product state |↑↑…↑⟩.
    let local: Vec<Vec<f64>> = (0..n).map(|_| vec![1.0, 0.0]).collect();
    let mut mps = Mps::from_product_state(&local)?;

    let h = tfim_bond(j, g);
    let tau = 0.05;
    let gate_full = imag_gate(&h, tau);
    let gate_half = imag_gate(&h, 0.5 * tau);

    // Single-site observables: `expectation_local` inserts an operator at *every*
    // site simultaneously, so to measure ⟨O_s⟩ at one site we place `O` at site `s`
    // and the identity elsewhere, then average over sites for the mean magnetisation.
    let single_site = |m: &Mps, op: &[f64; 4]| -> Result<f64, Box<dyn std::error::Error>> {
        let nrm = m.norm_squared()?;
        let mut total = 0.0_f64;
        for s in 0..n {
            let ops: Vec<Vec<f64>> = (0..n)
                .map(|site| if site == s { op.to_vec() } else { ID2.to_vec() })
                .collect();
            total += m.expectation_local(&ops)? / nrm;
        }
        Ok(total / n as f64)
    };

    let observe = |m: &Mps| -> Result<(f64, f64), Box<dyn std::error::Error>> {
        let sx = single_site(m, &SX)?;
        let sz = single_site(m, &SZ)?;
        Ok((sx, sz))
    };

    let (sx0, sz0) = observe(&mps)?;
    println!("TFIM quench, L = {n}, J = {j}, g = {g} (imaginary-time TEBD, 2nd order)");
    println!("  step    tau*t      <Sx>        <Sz>");
    println!("  {:>4}  {:>8.3}  {:>9.5}  {:>9.5}", 0, 0.0, sx0, sz0);

    let n_steps = 60usize;
    for step in 1..=n_steps {
        // exp(-tau/2 H_odd) exp(-tau H_even) exp(-tau/2 H_odd).
        for s in (0..n - 1).step_by(2) {
            apply_two_site_gate(&mut mps, s, &gate_half, cfg)?;
        }
        for s in (1..n - 1).step_by(2) {
            apply_two_site_gate(&mut mps, s, &gate_full, cfg)?;
        }
        for s in (0..n - 1).step_by(2) {
            apply_two_site_gate(&mut mps, s, &gate_half, cfg)?;
        }
        let nrm = mps.norm()?;
        mps.rescale(1.0 / nrm)?;

        if step % 10 == 0 || step == n_steps {
            let (sx, sz) = observe(&mps)?;
            println!(
                "  {:>4}  {:>8.3}  {:>9.5}  {:>9.5}",
                step,
                step as f64 * tau,
                sx,
                sz
            );
        }
    }

    println!("  (state relaxed toward the g = {g} ground state)");
    Ok(())
}
