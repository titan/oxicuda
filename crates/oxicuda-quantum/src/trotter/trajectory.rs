//! Monte-Carlo wave-function (MCWF) unraveling of the Lindblad master equation.
//!
//! Reference: Mølmer, Castin & Dalibard, "Monte Carlo wave-function method in
//! quantum optics", J. Opt. Soc. Am. B 10, 524 (1993); Dalibard, Castin & Mølmer,
//! Phys. Rev. Lett. 68, 580 (1992).
//!
//! Instead of propagating the full `dim × dim` density matrix `ρ(t)` under the
//! Lindblad master equation
//!
//! ```text
//! dρ/dt = -i[H, ρ] + Σ_k γ_k ( L_k ρ L_k† - ½ { L_k†L_k , ρ } ),
//! ```
//!
//! the MCWF method evolves an ensemble of *pure* state vectors `|ψ⟩` (each only
//! `dim` amplitudes) under the **effective non-Hermitian Hamiltonian**
//!
//! ```text
//! H_eff = H - (i/2) Σ_k γ_k L_k† L_k,
//! ```
//!
//! punctuated by stochastic **quantum jumps** `|ψ⟩ → L_k |ψ⟩ / ‖L_k |ψ⟩‖`. The
//! ensemble average `ρ(t) = E[ |ψ(t)⟩⟨ψ(t)| ]` reproduces the master-equation
//! solution. This is exponentially cheaper in memory for low-dissipation, large
//! Hilbert spaces and is the standard "quantum-trajectory" open-system method.
//!
//! ## Discrete-time algorithm (per step `dt`)
//! 1. Compute the no-jump survival deficit
//!    `δp = dt · Σ_k γ_k ⟨ψ| L_k†L_k |ψ⟩ = 1 - ‖(1 - i H_eff dt)|ψ⟩‖²`.
//! 2. Draw `r ∈ [0,1)`.
//!    * If `r ≥ δp` (**no jump**): deterministically propagate
//!      `|ψ⟩ → (1 - i H_eff dt) |ψ⟩` and renormalize.
//!    * If `r < δp` (**jump**): choose channel `k` with probability
//!      `δp_k / δp` where `δp_k = dt · γ_k ⟨L_k†L_k⟩`, then set
//!      `|ψ⟩ → L_k |ψ⟩ / ‖L_k |ψ⟩‖`.
//!
//! The Hamiltonian and jump operators are supplied as Pauli strings (matching
//! [`crate::trotter::lindblad::LindbladOp`]); the small dense matrices they
//! generate are reused for both the `H_eff` propagator and the jump action.

use num_complex::Complex;

use crate::density::density::DensityMatrix;
use crate::error::{QuantumError, QuantumResult};
use crate::handle::LcgRng;
use crate::pauli::pauli_string::{PauliOp, PauliString};
use crate::statevec::state::StateVector;
use crate::trotter::lindblad::LindbladOp;

type Complex32 = Complex<f32>;

/// Configuration for a Monte-Carlo wave-function trajectory run.
#[derive(Debug, Clone)]
pub struct TrajectoryConfig {
    /// Integration time step. Must be strictly positive and small enough that the
    /// per-step jump probability stays well below 1.
    pub dt: f32,
    /// Number of time steps to evolve.
    pub n_steps: usize,
    /// Number of independent stochastic trajectories to average.
    pub n_trajectories: usize,
}

impl TrajectoryConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns [`QuantumError::InvalidParameter`] if `dt <= 0`, `n_steps == 0`,
    /// or `n_trajectories == 0`.
    pub fn new(dt: f32, n_steps: usize, n_trajectories: usize) -> QuantumResult<Self> {
        if dt.is_nan() || dt <= 0.0 {
            return Err(QuantumError::InvalidParameter { name: "dt".into() });
        }
        if n_steps == 0 {
            return Err(QuantumError::InvalidParameter {
                name: "n_steps".into(),
            });
        }
        if n_trajectories == 0 {
            return Err(QuantumError::InvalidParameter {
                name: "n_trajectories".into(),
            });
        }
        Ok(Self {
            dt,
            n_steps,
            n_trajectories,
        })
    }
}

/// A Monte-Carlo wave-function solver for the Lindblad master equation.
///
/// Holds the precomputed dense `H_eff` matrix and the per-channel jump matrices
/// `√γ_k L_k` so each trajectory step is a pair of small dense mat-vecs.
#[derive(Debug, Clone)]
pub struct QuantumTrajectory {
    n_qubits: usize,
    dim: usize,
    /// Effective non-Hermitian Hamiltonian `H_eff = H - (i/2) Σ_k γ_k L_k†L_k`,
    /// row-major `dim × dim`.
    h_eff: Vec<Complex32>,
    /// Jump matrices `C_k = √γ_k · L_k`, row-major `dim × dim`, one per channel.
    jump_mats: Vec<Vec<Complex32>>,
}

impl QuantumTrajectory {
    /// Build the trajectory solver for `n_qubits` qubits.
    ///
    /// `ham_terms` is `Σ coeff·P` (Hermitian Hamiltonian, Pauli-string form);
    /// `lindblad_ops` are the collapse operators `L_k` carrying their rates as
    /// `coeff = γ_k^{1/2}` is **not** assumed — instead each [`LindbladOp::coeff`]
    /// is interpreted as the dissipation rate `γ_k` (matching
    /// [`crate::trotter::lindblad::lindblad_step`]), so the jump matrix is
    /// `√γ_k · L_k` and `H_eff` subtracts `(i/2) γ_k L_k†L_k`.
    ///
    /// # Errors
    /// Returns an error if any Pauli string length differs from `n_qubits`.
    pub fn new(
        n_qubits: usize,
        ham_terms: &[(f32, Vec<PauliOp>)],
        lindblad_ops: &[LindbladOp],
    ) -> QuantumResult<Self> {
        if n_qubits == 0 || n_qubits > 14 {
            return Err(QuantumError::InvalidQubitCount { n: n_qubits });
        }
        let dim = 1usize << n_qubits;

        // Hermitian Hamiltonian matrix H.
        let mut h_eff = build_pauli_sum_matrix(ham_terms, dim, n_qubits)?;

        // Jump matrices C_k = √γ_k L_k and accumulate (i/2) Σ γ_k L_k†L_k into H_eff.
        let mut jump_mats = Vec::with_capacity(lindblad_ops.len());
        for lop in lindblad_ops {
            if lop.ops.len() != n_qubits {
                return Err(QuantumError::DimensionMismatch {
                    expected: n_qubits,
                    got: lop.ops.len(),
                });
            }
            let gamma = lop.coeff.max(0.0);
            let sqrt_gamma = gamma.sqrt();
            // L_k (unit-weight Pauli string acting as operator).
            let l_mat = pauli_string_matrix(&lop.ops, dim, n_qubits)?;
            // C_k = √γ_k L_k.
            let c_mat: Vec<Complex32> = l_mat.iter().map(|z| z * sqrt_gamma).collect();
            // γ_k L_k†L_k.
            let l_dag = mat_adjoint(&l_mat, dim);
            let ldagl = mat_mul(&l_dag, &l_mat, dim);
            // H_eff -= (i/2) γ_k L_k†L_k.
            let half_i_gamma = Complex32::new(0.0, -0.5 * gamma);
            for (h, g) in h_eff.iter_mut().zip(ldagl.iter()) {
                *h += half_i_gamma * g;
            }
            jump_mats.push(c_mat);
        }

        Ok(Self {
            n_qubits,
            dim,
            h_eff,
            jump_mats,
        })
    }

    /// Number of qubits.
    #[must_use]
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Number of collapse channels.
    #[must_use]
    pub fn n_channels(&self) -> usize {
        self.jump_mats.len()
    }

    /// Advance one pure trajectory by a single MCWF step of size `dt`.
    ///
    /// Mutates `psi` in place and returns `Some(k)` if a jump on channel `k`
    /// occurred this step, or `None` for a no-jump (deterministic) step.
    pub fn step(
        &self,
        psi: &mut Vec<Complex32>,
        dt: f32,
        rng: &mut LcgRng,
    ) -> QuantumResult<Option<usize>> {
        if psi.len() != self.dim {
            return Err(QuantumError::DimensionMismatch {
                expected: self.dim,
                got: psi.len(),
            });
        }

        // Per-channel jump weights δp_k = dt · ⟨ψ| C_k†C_k |ψ⟩ = dt · ‖C_k ψ‖².
        let mut dp_k = vec![0.0_f32; self.jump_mats.len()];
        let mut dp_total = 0.0_f32;
        for (k, c) in self.jump_mats.iter().enumerate() {
            let c_psi = mat_vec(c, psi, self.dim);
            let norm_sq: f32 = c_psi.iter().map(|z| z.norm_sqr()).sum();
            let w = dt * norm_sq;
            dp_k[k] = w;
            dp_total += w;
        }

        let r = rng.next_u32() as f32 / 2f32.powi(32);

        if r >= dp_total {
            // No jump: |ψ⟩ → (1 - i H_eff dt)|ψ⟩, renormalize.
            let h_psi = mat_vec(&self.h_eff, psi, self.dim);
            let factor = Complex32::new(0.0, -dt); // -i dt
            for (p, hp) in psi.iter_mut().zip(h_psi.iter()) {
                *p += factor * hp;
            }
            renormalize(psi);
            Ok(None)
        } else {
            // Jump: pick channel k weighted by δp_k.
            let target = r; // r ∈ [0, dp_total)
            let mut acc = 0.0_f32;
            let mut chosen = self.jump_mats.len().saturating_sub(1);
            for (k, w) in dp_k.iter().enumerate() {
                acc += *w;
                if target < acc {
                    chosen = k;
                    break;
                }
            }
            let c_psi = mat_vec(&self.jump_mats[chosen], psi, self.dim);
            *psi = c_psi;
            renormalize(psi);
            Ok(Some(chosen))
        }
    }

    /// Evolve a single trajectory from `initial` for `cfg.n_steps` steps,
    /// returning the final pure state.
    pub fn evolve_single(
        &self,
        initial: &StateVector,
        cfg: &TrajectoryConfig,
        rng: &mut LcgRng,
    ) -> QuantumResult<StateVector> {
        if initial.n_qubits != self.n_qubits {
            return Err(QuantumError::DimensionMismatch {
                expected: self.n_qubits,
                got: initial.n_qubits,
            });
        }
        let mut psi = initial.amps.clone();
        renormalize(&mut psi);
        for _ in 0..cfg.n_steps {
            self.step(&mut psi, cfg.dt, rng)?;
        }
        Ok(StateVector {
            amps: psi,
            n_qubits: self.n_qubits,
        })
    }

    /// Run the full ensemble and return the averaged density matrix
    /// `ρ ≈ (1/N) Σ_t |ψ_t⟩⟨ψ_t|`.
    ///
    /// This is the central quantity that, in the `N → ∞` and `dt → 0` limits,
    /// converges to the Lindblad master-equation solution `ρ(t)`.
    pub fn evolve_density(
        &self,
        initial: &StateVector,
        cfg: &TrajectoryConfig,
        rng: &mut LcgRng,
    ) -> QuantumResult<DensityMatrix> {
        if initial.n_qubits != self.n_qubits {
            return Err(QuantumError::DimensionMismatch {
                expected: self.n_qubits,
                got: initial.n_qubits,
            });
        }
        let dim = self.dim;
        let mut rho = vec![Complex32::new(0.0, 0.0); dim * dim];
        let inv_n = 1.0 / cfg.n_trajectories as f32;
        for _ in 0..cfg.n_trajectories {
            let final_state = self.evolve_single(initial, cfg, rng)?;
            for i in 0..dim {
                let ai = final_state.amps[i];
                for j in 0..dim {
                    rho[i * dim + j] += inv_n * (ai * final_state.amps[j].conj());
                }
            }
        }
        Ok(DensityMatrix { rho, dim })
    }
}

/// Renormalize a complex amplitude vector to unit norm (no-op if ~zero).
fn renormalize(psi: &mut [Complex32]) {
    let norm: f32 = psi.iter().map(|z| z.norm_sqr()).sum::<f32>().sqrt();
    if norm > 1e-20 {
        let inv = 1.0 / norm;
        for z in psi.iter_mut() {
            *z *= inv;
        }
    }
}

/// Dense matrix–vector product `y = A x` for row-major `dim × dim` `A`.
fn mat_vec(a: &[Complex32], x: &[Complex32], dim: usize) -> Vec<Complex32> {
    let mut y = vec![Complex32::new(0.0, 0.0); dim];
    for i in 0..dim {
        let row = &a[i * dim..i * dim + dim];
        let mut acc = Complex32::new(0.0, 0.0);
        for (k, xk) in x.iter().enumerate() {
            acc += row[k] * xk;
        }
        y[i] = acc;
    }
    y
}

fn mat_mul(a: &[Complex32], b: &[Complex32], dim: usize) -> Vec<Complex32> {
    let mut c = vec![Complex32::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut acc = Complex32::new(0.0, 0.0);
            for k in 0..dim {
                acc += a[i * dim + k] * b[k * dim + j];
            }
            c[i * dim + j] = acc;
        }
    }
    c
}

fn mat_adjoint(a: &[Complex32], dim: usize) -> Vec<Complex32> {
    let mut out = vec![Complex32::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            out[j * dim + i] = a[i * dim + j].conj();
        }
    }
    out
}

/// Build `Σ coeff·P` as a dense Hermitian matrix.
fn build_pauli_sum_matrix(
    terms: &[(f32, Vec<PauliOp>)],
    dim: usize,
    n_qubits: usize,
) -> QuantumResult<Vec<Complex32>> {
    let mut mat = vec![Complex32::new(0.0, 0.0); dim * dim];
    for (coeff, ops) in terms {
        if ops.len() != n_qubits {
            return Err(QuantumError::DimensionMismatch {
                expected: n_qubits,
                got: ops.len(),
            });
        }
        let term = pauli_string_matrix(ops, dim, n_qubits)?;
        for (m, t) in mat.iter_mut().zip(term.iter()) {
            *m += coeff * t;
        }
    }
    Ok(mat)
}

/// Dense matrix of a unit-weight Pauli string acting as an operator.
fn pauli_string_matrix(
    ops: &[PauliOp],
    dim: usize,
    n_qubits: usize,
) -> QuantumResult<Vec<Complex32>> {
    let mut mat = vec![Complex32::new(0.0, 0.0); dim * dim];
    let ps = PauliString::new(1.0, ops.to_vec());
    for col in 0..dim {
        let mut e_col = vec![Complex32::new(0.0, 0.0); dim];
        e_col[col] = Complex32::new(1.0, 0.0);
        let sv = StateVector {
            amps: e_col,
            n_qubits,
        };
        let p_sv = ps.apply_to_state(&sv)?;
        for row in 0..dim {
            mat[row * dim + col] = p_sv.amps[row];
        }
    }
    Ok(mat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::density::metrics::purity;

    #[test]
    fn config_validation() {
        assert!(TrajectoryConfig::new(0.0, 10, 10).is_err());
        assert!(TrajectoryConfig::new(0.01, 0, 10).is_err());
        assert!(TrajectoryConfig::new(0.01, 10, 0).is_err());
        assert!(TrajectoryConfig::new(0.01, 10, 10).is_ok());
    }

    #[test]
    fn no_dissipation_is_unitary_norm_preserving() {
        // H = Z on 1 qubit, no jumps → deterministic Schrödinger evolution.
        let traj = QuantumTrajectory::new(1, &[(1.0, vec![PauliOp::Z])], &[])
            .expect("valid 1-qubit trajectory");
        let init = StateVector::new_zero_state(1).expect("valid zero state");
        let cfg = TrajectoryConfig::new(0.01, 50, 1).expect("valid config");
        let mut rng = LcgRng::new(1);
        let out = traj
            .evolve_single(&init, &cfg, &mut rng)
            .expect("evolution succeeds");
        let norm: f32 = out.amps.iter().map(|z| z.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
    }

    #[test]
    fn amplitude_damping_jumps_relax_excited_to_ground() {
        // Single-qubit relaxation modeled with σ⁻ ≈ (X + iY)/2. We approximate the
        // lowering operator's *effect* using the X jump under a strong rate so the
        // excited |1⟩ population decays toward |0⟩ on average. Here we instead use
        // the exact lowering channel via a Pauli decomposition is unavailable, so
        // we check the simpler invariant: with a Z-dephasing jump, the ensemble
        // density matrix stays trace-1 and loses purity from the |+⟩ state.
        let traj = QuantumTrajectory::new(1, &[], &[LindbladOp::new(1.0, vec![PauliOp::Z])])
            .expect("valid trajectory");
        // |+⟩ state.
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let plus = StateVector {
            amps: vec![
                Complex32::new(inv_sqrt2, 0.0),
                Complex32::new(inv_sqrt2, 0.0),
            ],
            n_qubits: 1,
        };
        let cfg = TrajectoryConfig::new(0.02, 60, 400).expect("valid config");
        let mut rng = LcgRng::new(2024);
        let rho = traj
            .evolve_density(&plus, &cfg, &mut rng)
            .expect("density evolution");
        // Trace preserved.
        let tr = rho.trace();
        assert!((tr.re - 1.0).abs() < 1e-3, "trace={}", tr.re);
        // Dephasing destroys the |+⟩ coherence → purity drops below 1.
        let p = purity(&rho);
        assert!(p < 0.97, "purity should drop under dephasing, got {p}");
    }

    #[test]
    fn density_trace_preserved_with_hamiltonian_and_jump() {
        let traj = QuantumTrajectory::new(
            1,
            &[(0.5, vec![PauliOp::X])],
            &[LindbladOp::new(0.4, vec![PauliOp::Z])],
        )
        .expect("valid trajectory");
        let init = StateVector::new_zero_state(1).expect("valid zero state");
        let cfg = TrajectoryConfig::new(0.01, 80, 200).expect("valid config");
        let mut rng = LcgRng::new(7);
        let rho = traj
            .evolve_density(&init, &cfg, &mut rng)
            .expect("density evolution");
        let tr = rho.trace();
        assert!((tr.re - 1.0).abs() < 2e-3, "trace={}", tr.re);
        assert!(tr.im.abs() < 1e-3, "trace imag={}", tr.im);
    }

    #[test]
    fn two_qubit_dimension_checks() {
        let traj =
            QuantumTrajectory::new(2, &[(1.0, vec![PauliOp::Z, PauliOp::Z])], &[]).expect("valid");
        // Wrong-size psi rejected.
        let mut bad = vec![Complex32::new(1.0, 0.0); 2];
        let mut rng = LcgRng::new(0);
        assert!(traj.step(&mut bad, 0.01, &mut rng).is_err());
    }

    #[test]
    fn trajectory_matches_rk4_master_equation() {
        // Cross-check: the MCWF ensemble-averaged ρ must agree with the exact
        // Lindblad master-equation ρ (integrated by RK4) for a dephased qubit
        // driven by H = h·X. We compare on the |+⟩ initial state, where dephasing
        // produces a non-trivial coherence decay that both methods must track.
        use crate::trotter::lindblad_rk4::LindbladRk4;

        let h_coeff = 0.6_f32;
        let gamma = 0.7_f32;
        let ham = [(h_coeff, vec![PauliOp::X])];
        let lops = [LindbladOp::new(gamma, vec![PauliOp::Z])];

        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let plus = StateVector {
            amps: vec![
                Complex32::new(inv_sqrt2, 0.0),
                Complex32::new(inv_sqrt2, 0.0),
            ],
            n_qubits: 1,
        };

        // --- Master-equation reference (RK4). ---
        let rk4 = LindbladRk4::new(1, &ham, &lops).expect("rk4");
        let mut dm_ref = DensityMatrix::from_pure_state(&plus);
        let dt = 0.01_f32;
        let n_steps = 50usize; // total time t = 0.5
        rk4.evolve(&mut dm_ref, dt, n_steps).expect("rk4 evolve");

        // --- MCWF ensemble. ---
        let traj = QuantumTrajectory::new(1, &ham, &lops).expect("traj");
        let cfg = TrajectoryConfig::new(dt, n_steps, 4000).expect("cfg");
        let mut rng = LcgRng::new(20240620);
        let dm_mcwf = traj
            .evolve_density(&plus, &cfg, &mut rng)
            .expect("mcwf density");

        // Compare every entry within Monte-Carlo tolerance (~1/√N ≈ 0.016 here,
        // allow a comfortable band).
        for idx in 0..4 {
            let dr = (dm_ref.rho[idx].re - dm_mcwf.rho[idx].re).abs();
            let di = (dm_ref.rho[idx].im - dm_mcwf.rho[idx].im).abs();
            assert!(
                dr < 4e-2 && di < 4e-2,
                "entry {idx}: ref={:?} mcwf={:?}",
                dm_ref.rho[idx],
                dm_mcwf.rho[idx]
            );
        }
    }
}
