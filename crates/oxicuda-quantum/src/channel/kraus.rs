use num_complex::Complex;

use crate::density::density::DensityMatrix;
use crate::error::{QuantumError, QuantumResult};

type Complex32 = Complex<f32>;

/// A quantum channel represented by Kraus operators {K_i}.
///
/// Must satisfy the completeness relation Σ_i K_i†K_i = I.
#[derive(Debug, Clone)]
pub struct KrausChannel {
    /// List of Kraus operators, each of size `dim × dim`, stored row-major.
    pub ops: Vec<Vec<Complex32>>,
    pub dim: usize,
}

impl KrausChannel {
    /// Construct and validate completeness.
    pub fn new(ops: Vec<Vec<Complex32>>, dim: usize) -> QuantumResult<Self> {
        if ops.is_empty() {
            return Err(QuantumError::EmptyInput);
        }
        for (k, op) in ops.iter().enumerate() {
            if op.len() != dim * dim {
                return Err(QuantumError::DimensionMismatch {
                    expected: dim * dim,
                    got: op.len(),
                });
            }
            let _ = k;
        }

        // Check Σ K†K = I
        let mut sum = vec![Complex32::new(0.0, 0.0); dim * dim];
        for op in &ops {
            let kdagk = mat_adjoint_mul(op, dim);
            for (s, k) in sum.iter_mut().zip(kdagk.iter()) {
                *s += k;
            }
        }

        let mut residual = 0.0_f32;
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = (sum[i * dim + j].re - expected).abs() + sum[i * dim + j].im.abs();
                residual = residual.max(diff);
            }
        }

        if residual > 1e-3 {
            return Err(QuantumError::KrausNotComplete { residual });
        }

        Ok(Self { ops, dim })
    }

    /// Apply the channel: ρ' = Σ_i K_i ρ K_i†.
    pub fn apply(&self, dm: &DensityMatrix) -> QuantumResult<DensityMatrix> {
        if dm.dim != self.dim {
            return Err(QuantumError::DimensionMismatch {
                expected: self.dim,
                got: dm.dim,
            });
        }
        let dim = self.dim;
        let mut rho_out = vec![Complex32::new(0.0, 0.0); dim * dim];

        for op in &self.ops {
            // K * ρ
            let k_rho = mat_mul(op, &dm.rho, dim);
            // (K * ρ) * K†
            let k_rho_kd = mat_mul_adjoint(&k_rho, op, dim);
            for (r, k) in rho_out.iter_mut().zip(k_rho_kd.iter()) {
                *r += k;
            }
        }

        Ok(DensityMatrix { rho: rho_out, dim })
    }
}

pub fn mat_mul(a: &[Complex32], b: &[Complex32], dim: usize) -> Vec<Complex32> {
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

/// Compute A * B†.
fn mat_mul_adjoint(a: &[Complex32], b: &[Complex32], dim: usize) -> Vec<Complex32> {
    let mut c = vec![Complex32::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            for k in 0..dim {
                c[i * dim + j] += a[i * dim + k] * b[j * dim + k].conj();
            }
        }
    }
    c
}

/// Compute A† * A.
fn mat_adjoint_mul(a: &[Complex32], dim: usize) -> Vec<Complex32> {
    let mut c = vec![Complex32::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            for k in 0..dim {
                // (A†)[i,k] = conj(A[k,i])
                c[i * dim + j] += a[k * dim + i].conj() * a[k * dim + j];
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_channel_preserves_state() {
        let id_op = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 0.0),
        ];
        let ch = KrausChannel::new(vec![id_op], 2)
            .expect("the 2×2 identity operator satisfies Σ K†K = I exactly, so channel construction cannot fail");
        use crate::statevec::state::StateVector;
        let sv = StateVector::new_zero_state(1).expect(
            "n_qubits=1 is always a valid qubit count, so zero-state construction cannot fail",
        );
        let dm = DensityMatrix::from_pure_state(&sv);
        let out = ch
            .apply(&dm)
            .expect("the density matrix dim=2 matches the channel dim=2, so apply cannot fail");
        assert!((out.rho[0].re - 1.0).abs() < 1e-6);
    }
}
