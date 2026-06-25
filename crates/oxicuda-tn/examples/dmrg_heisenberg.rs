//! Worked example: two-site DMRG ground state of the spin-1/2 Heisenberg chain.
//!
//! Builds the open-boundary Heisenberg XXX MPO, runs two-site DMRG from a random
//! initial MPS, and prints the converged ground-state energy together with the energy
//! per bond. For small `n` the value can be checked against exact diagonalisation; the
//! thermodynamic-limit energy per site is the Bethe-ansatz constant `1/4 - ln 2`.
//!
//! Run with:
//! ```text
//! cargo run -p oxicuda-tn --example dmrg_heisenberg
//! ```

use oxicuda_tn::dmrg::{DmrgConfig, dmrg_two_site};
use oxicuda_tn::handle::LcgRng;
use oxicuda_tn::mpo::mpo::Mpo;
use oxicuda_tn::mps::mps::Mps;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 12usize; // number of spin-1/2 sites
    let mpo = Mpo::heisenberg_xxx(n)?;

    let mut rng = LcgRng::new(20240621);
    let init = Mps::random_mps(n, 2, 8, &mut rng)?;

    let cfg = DmrgConfig {
        max_sweeps: 12,
        chi_max: 32,
        trunc_tol: 1e-12,
        energy_tol: 1e-10,
        lanczos_iter: 60,
        lanczos_tol: 1e-12,
    };

    let result = dmrg_two_site(&mpo, init, cfg, &mut rng)?;

    println!("Heisenberg XXX chain, L = {n} (open boundary)");
    println!("  sweeps performed : {}", result.sweeps_done);
    println!("  ground energy E0 : {:.10}", result.energy);
    println!(
        "  energy per bond  : {:.10}",
        result.energy / (n as f64 - 1.0)
    );
    println!(
        "  Bethe per-site   : {:.10}  (1/4 - ln 2, thermodynamic limit)",
        0.25 - 2.0_f64.ln()
    );

    if let Some(&last) = result.energy_history.last() {
        println!("  final-sweep energy: {last:.10}");
    }
    Ok(())
}
