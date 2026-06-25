//! Fourth-order Runge-Kutta integrator for the Lindblad master equation.
//!
//! Reference: Breuer & Petruccione, *The Theory of Open Quantum Systems*,
//! Oxford University Press (2002), §3.2 (the Lindblad / GKSL generator) and the
//! standard classical RK4 scheme.
//!
//! This is the higher-accuracy complement to the first-order forward-Euler
//! [`crate::trotter::lindblad::lindblad_step`]. It integrates
//!
//! ```text
//! dρ/dt = L[ρ] = -i[H, ρ] + Σ_k γ_k ( L_k ρ L_k† - ½ { L_k†L_k , ρ } )
//! ```
//!
//! by evaluating the Lindblad super-operator `L[·]` (a linear map on the space
//! of `dim × dim` matrices) at four stages and combining them with the classical
//! RK4 weights `(k₁ + 2k₂ + 2k₃ + k₄)/6`. The local truncation error is
//! `O(dt⁵)` per step versus `O(dt²)` for forward Euler, so for a given accuracy
//! RK4 permits a far larger `dt` and is the production-grade choice for smooth
//! (time-independent) Lindbladians.

use num_complex::Complex;

use crate::density::density::DensityMatrix;
use crate::error::{QuantumError, QuantumResult};
use crate::pauli::pauli_string::{PauliOp, PauliString};
use crate::statevec::state::StateVector;
use crate::trotter::lindblad::LindbladOp;

type Complex32 = Complex<f32>;

/// A reusable RK4 Lindblad integrator that precomputes the dense operator
/// matrices `H`, `C_k = √γ_k L_k`, and `C_k†C_k` once.
#[derive(Debug, Clone)]
pub struct LindbladRk4 {
    dim: usize,
    /// Hermitian Hamiltonian, row-major `dim × dim`.
    h_mat: Vec<Complex32>,
    /// Collapse matrices `C_k = √γ_k L_k`.
    c_mats: Vec<Vec<Complex32>>,
    /// Precomputed `C_k† C_k`.
    cdc_mats: Vec<Vec<Complex32>>,
}

impl LindbladRk4 {
    /// Build the integrator for an `n_qubits` system.
    ///
    /// Each [`LindbladOp::coeff`] is interpreted as the dissipation **rate**
    /// `γ_k`, consistent with [`crate::trotter::lindblad::lindblad_step`]; the
    /// stored collapse matrix is therefore `√γ_k · L_k`.
    ///
    /// # Errors
    /// Returns an error for an invalid qubit count or a Pauli string whose length
    /// does not equal `n_qubits`.
    pub fn new(
        n_qubits: usize,
        ham_terms: &[(f32, Vec<PauliOp>)],
        lindblad_ops: &[LindbladOp],
    ) -> QuantumResult<Self> {
        if n_qubits == 0 || n_qubits > 14 {
            return Err(QuantumError::InvalidQubitCount { n: n_qubits });
        }
        let dim = 1usize << n_qubits;

        let h_mat = build_pauli_sum_matrix(ham_terms, dim, n_qubits)?;

        let mut c_mats = Vec::with_capacity(lindblad_ops.len());
        let mut cdc_mats = Vec::with_capacity(lindblad_ops.len());
        for lop in lindblad_ops {
            if lop.ops.len() != n_qubits {
                return Err(QuantumError::DimensionMismatch {
                    expected: n_qubits,
                    got: lop.ops.len(),
                });
            }
            let sqrt_gamma = lop.coeff.max(0.0).sqrt();
            let l_mat = pauli_string_matrix(&lop.ops, dim, n_qubits)?;
            let c_mat: Vec<Complex32> = l_mat.iter().map(|z| z * sqrt_gamma).collect();
            let c_dag = mat_adjoint(&c_mat, dim);
            let cdc = mat_mul(&c_dag, &c_mat, dim);
            c_mats.push(c_mat);
            cdc_mats.push(cdc);
        }

        Ok(Self {
            dim,
            h_mat,
            c_mats,
            cdc_mats,
        })
    }

    /// Dimension `2^n` of the Hilbert space.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Evaluate the Lindblad generator `L[ρ]` (the time derivative `dρ/dt`).
    fn derivative(&self, rho: &[Complex32]) -> Vec<Complex32> {
        let dim = self.dim;
        // -i[H, ρ] = -i (Hρ - ρH).
        let hr = mat_mul(&self.h_mat, rho, dim);
        let rh = mat_mul(rho, &self.h_mat, dim);
        let minus_i = Complex32::new(0.0, -1.0);
        let mut d = vec![Complex32::new(0.0, 0.0); dim * dim];
        for idx in 0..dim * dim {
            d[idx] = minus_i * (hr[idx] - rh[idx]);
        }
        // Dissipator Σ_k ( C_k ρ C_k† - ½ {C_k†C_k, ρ} ).
        for (c, cdc) in self.c_mats.iter().zip(self.cdc_mats.iter()) {
            let c_dag = mat_adjoint(c, dim);
            let c_rho = mat_mul(c, rho, dim);
            let c_rho_cd = mat_mul(&c_rho, &c_dag, dim);
            let cdc_rho = mat_mul(cdc, rho, dim);
            let rho_cdc = mat_mul(rho, cdc, dim);
            for idx in 0..dim * dim {
                d[idx] += c_rho_cd[idx] - 0.5 * cdc_rho[idx] - 0.5 * rho_cdc[idx];
            }
        }
        d
    }

    /// Advance the density matrix by one RK4 step of size `dt` (in place).
    ///
    /// # Errors
    /// Returns [`QuantumError::DimensionMismatch`] if `dm.dim` differs from the
    /// integrator's dimension.
    pub fn step(&self, dm: &mut DensityMatrix, dt: f32) -> QuantumResult<()> {
        if dm.dim != self.dim {
            return Err(QuantumError::DimensionMismatch {
                expected: self.dim,
                got: dm.dim,
            });
        }
        let n = self.dim * self.dim;
        let rho = &dm.rho;

        // k1 = L[ρ].
        let k1 = self.derivative(rho);
        // k2 = L[ρ + dt/2 · k1].
        let mid1: Vec<Complex32> = (0..n).map(|i| rho[i] + 0.5 * dt * k1[i]).collect();
        let k2 = self.derivative(&mid1);
        // k3 = L[ρ + dt/2 · k2].
        let mid2: Vec<Complex32> = (0..n).map(|i| rho[i] + 0.5 * dt * k2[i]).collect();
        let k3 = self.derivative(&mid2);
        // k4 = L[ρ + dt · k3].
        let end: Vec<Complex32> = (0..n).map(|i| rho[i] + dt * k3[i]).collect();
        let k4 = self.derivative(&end);

        let sixth = dt / 6.0;
        let mut out = vec![Complex32::new(0.0, 0.0); n];
        for i in 0..n {
            out[i] = rho[i] + sixth * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        dm.rho = out;
        Ok(())
    }

    /// Integrate `n_steps` RK4 steps of size `dt`, mutating `dm` in place.
    pub fn evolve(&self, dm: &mut DensityMatrix, dt: f32, n_steps: usize) -> QuantumResult<()> {
        for _ in 0..n_steps {
            self.step(dm, dt)?;
        }
        Ok(())
    }
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
    fn closed_system_preserves_trace_and_purity() {
        // H = Z, no dissipation → unitary evolution, ρ stays pure & trace-1.
        let integ = LindbladRk4::new(1, &[(1.0, vec![PauliOp::Z])], &[]).expect("valid");
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let plus = StateVector {
            amps: vec![
                Complex32::new(inv_sqrt2, 0.0),
                Complex32::new(inv_sqrt2, 0.0),
            ],
            n_qubits: 1,
        };
        let mut dm = DensityMatrix::from_pure_state(&plus);
        integ.evolve(&mut dm, 0.01, 100).expect("evolve ok");
        let tr = dm.trace();
        assert!((tr.re - 1.0).abs() < 1e-4, "trace={}", tr.re);
        let p = purity(&dm);
        assert!((p - 1.0).abs() < 1e-3, "purity={p}");
    }

    #[test]
    fn dephasing_decays_off_diagonal_analytically() {
        // Pure dephasing L = √γ Z on |+⟩: ρ01(t) = ½ e^{-2γ t}.
        // (Z-dephasing damps coherences at rate 2γ.)
        let gamma = 0.5_f32;
        let integ =
            LindbladRk4::new(1, &[], &[LindbladOp::new(gamma, vec![PauliOp::Z])]).expect("valid");
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let plus = StateVector {
            amps: vec![
                Complex32::new(inv_sqrt2, 0.0),
                Complex32::new(inv_sqrt2, 0.0),
            ],
            n_qubits: 1,
        };
        let mut dm = DensityMatrix::from_pure_state(&plus);
        let dt = 0.005_f32;
        let n = 200usize;
        integ.evolve(&mut dm, dt, n).expect("evolve ok");
        let t = dt * n as f32;
        let expected = 0.5 * (-2.0 * gamma * t).exp();
        let off = dm.rho[1].re; // ρ01 (row 0, col 1) in a 2×2 row-major matrix
        assert!(
            (off - expected).abs() < 5e-3,
            "ρ01={off}, expected={expected}"
        );
        // Diagonal populations stay at ½ each.
        assert!((dm.rho[0].re - 0.5).abs() < 1e-3, "ρ00={}", dm.rho[0].re);
        assert!((dm.rho[3].re - 0.5).abs() < 1e-3, "ρ11={}", dm.rho[3].re);
        // Trace preserved.
        assert!((dm.trace().re - 1.0).abs() < 1e-4);
    }

    #[test]
    fn rk4_more_accurate_than_euler_for_dephasing() {
        // Same analytic target; RK4 with a coarse dt should beat Euler with the
        // same dt at matching the closed-form ρ01(t).
        use crate::trotter::lindblad::lindblad_step;
        let gamma = 0.5_f32;
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let plus = StateVector {
            amps: vec![
                Complex32::new(inv_sqrt2, 0.0),
                Complex32::new(inv_sqrt2, 0.0),
            ],
            n_qubits: 1,
        };
        let dt = 0.05_f32;
        let n = 20usize;
        let t = dt * n as f32;
        let expected = 0.5 * (-2.0 * gamma * t).exp();

        let integ =
            LindbladRk4::new(1, &[], &[LindbladOp::new(gamma, vec![PauliOp::Z])]).expect("valid");
        let mut dm_rk4 = DensityMatrix::from_pure_state(&plus);
        integ.evolve(&mut dm_rk4, dt, n).expect("evolve ok");
        let err_rk4 = (dm_rk4.rho[1].re - expected).abs();

        let mut dm_euler = DensityMatrix::from_pure_state(&plus);
        for _ in 0..n {
            lindblad_step(
                &mut dm_euler,
                &[],
                &[LindbladOp::new(gamma, vec![PauliOp::Z])],
                dt,
            )
            .expect("euler step ok");
        }
        let err_euler = (dm_euler.rho[1].re - expected).abs();

        assert!(
            err_rk4 <= err_euler + 1e-6,
            "RK4 err={err_rk4} should be ≤ Euler err={err_euler}"
        );
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let integ = LindbladRk4::new(1, &[], &[]).expect("valid");
        let sv = StateVector::new_zero_state(2).expect("valid 2q");
        let mut dm = DensityMatrix::from_pure_state(&sv);
        assert!(integ.step(&mut dm, 0.01).is_err());
    }
}
