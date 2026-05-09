use num_complex::Complex;

use crate::density::density::DensityMatrix;
use crate::error::QuantumResult;
use crate::pauli::pauli_string::{PauliOp, PauliString};

type Complex32 = Complex<f32>;

/// A Lindblad jump operator with its decay rate.
#[derive(Debug, Clone)]
pub struct LindbladOp {
    pub coeff: f32,
    pub ops: Vec<PauliOp>,
}

impl LindbladOp {
    #[must_use]
    pub fn new(coeff: f32, ops: Vec<PauliOp>) -> Self {
        Self { coeff, ops }
    }
}

/// One Euler step of the Lindblad master equation:
///
/// dρ/dt = -i\[H,ρ\] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
///
/// Uses forward Euler with step size `dt`; the caller is responsible for
/// choosing a sufficiently small `dt` for stability.
pub fn lindblad_step(
    dm: &mut DensityMatrix,
    ham_terms: &[(f32, Vec<PauliOp>)],
    lindblad_ops: &[LindbladOp],
    dt: f32,
) -> QuantumResult<()> {
    let dim = dm.dim;
    let n_qubits = dim.trailing_zeros() as usize;

    // Build full Hamiltonian matrix by summing Pauli terms
    let h_mat = build_pauli_sum_matrix(ham_terms, dim, n_qubits)?;

    // Commutator: -i[H, ρ] = -i(Hρ - ρH)
    let mut d_rho = commutator_deriv(&h_mat, &dm.rho, dim);

    // Lindblad dissipator
    for lop in lindblad_ops {
        let l_mat = pauli_string_matrix(&lop.ops, dim, n_qubits, Complex32::new(lop.coeff, 0.0))?;
        let l_dag = mat_adjoint(&l_mat, dim);
        let ldagl = mat_mul(&l_dag, &l_mat, dim);

        // L ρ L†
        let lro = mat_mul(&l_mat, &dm.rho, dim);
        let lro_ld = mat_mul(&lro, &l_dag, dim);

        // ½{L†L, ρ} = ½(L†L·ρ + ρ·L†L)
        let ldlr = mat_mul(&ldagl, &dm.rho, dim);
        let rldl = mat_mul(&dm.rho, &ldagl, dim);

        for idx in 0..(dim * dim) {
            d_rho[idx] += lro_ld[idx] - 0.5 * ldlr[idx] - 0.5 * rldl[idx];
        }
    }

    // Euler step: ρ += dt * dρ
    for (r, dr) in dm.rho.iter_mut().zip(d_rho.iter()) {
        *r += dt * dr;
    }

    Ok(())
}

fn build_pauli_sum_matrix(
    terms: &[(f32, Vec<PauliOp>)],
    dim: usize,
    n_qubits: usize,
) -> QuantumResult<Vec<Complex32>> {
    let mut mat = vec![Complex32::new(0.0, 0.0); dim * dim];
    for (coeff, ops) in terms {
        let term = pauli_string_matrix(ops, dim, n_qubits, Complex32::new(*coeff, 0.0))?;
        for (m, t) in mat.iter_mut().zip(term.iter()) {
            *m += t;
        }
    }
    Ok(mat)
}

fn pauli_string_matrix(
    ops: &[PauliOp],
    dim: usize,
    n_qubits: usize,
    scale: Complex32,
) -> QuantumResult<Vec<Complex32>> {
    // Build matrix via action on basis vectors (column-by-column)
    let mut mat = vec![Complex32::new(0.0, 0.0); dim * dim];
    let ps = PauliString::new(scale.re, ops.to_vec());
    use crate::statevec::state::StateVector;

    for col in 0..dim {
        let mut e_col = vec![Complex32::new(0.0, 0.0); dim];
        e_col[col] = Complex32::new(1.0, 0.0);
        let sv = StateVector {
            amps: e_col,
            n_qubits,
        };
        let p_sv = ps.apply_to_state(&sv)?;
        // If scale has imaginary part, we need to handle it:
        // Pauli string only supports real weight in apply_to_state, so multiply by scale/|scale|
        let extra = if scale.re.abs() > 1e-10 {
            Complex32::new(1.0, 0.0)
        } else {
            Complex32::new(0.0, 1.0) * scale.im.signum()
        };
        for row in 0..dim {
            mat[row * dim + col] = p_sv.amps[row] * extra;
        }
    }
    Ok(mat)
}

fn commutator_deriv(h: &[Complex32], rho: &[Complex32], dim: usize) -> Vec<Complex32> {
    // -i(H*ρ - ρ*H)
    let hr = mat_mul(h, rho, dim);
    let rh = mat_mul(rho, h, dim);
    let mi = Complex32::new(0.0, -1.0);
    (0..(dim * dim)).map(|k| mi * (hr[k] - rh[k])).collect()
}

fn mat_mul(a: &[Complex32], b: &[Complex32], dim: usize) -> Vec<Complex32> {
    let mut c = vec![Complex32::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            for k in 0..dim {
                c[i * dim + j] += a[i * dim + k] * b[k * dim + j];
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::density::density::DensityMatrix;
    use crate::statevec::state::StateVector;

    #[test]
    fn lindblad_step_approximately_traces_to_one() {
        let sv = StateVector::new_zero_state(1).unwrap();
        let mut dm = DensityMatrix::from_pure_state(&sv);
        lindblad_step(&mut dm, &[], &[], 0.01).unwrap();
        let tr = dm.trace();
        assert!((tr.re - 1.0).abs() < 1e-3, "trace={}", tr.re);
    }
}
