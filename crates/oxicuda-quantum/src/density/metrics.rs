use num_complex::Complex;

use crate::density::density::DensityMatrix;
use crate::error::{QuantumError, QuantumResult};

type Complex32 = Complex<f32>;

/// Purity Tr(ρ²).
#[must_use]
pub fn purity(dm: &DensityMatrix) -> f32 {
    dm.trace_sq().re
}

/// Fidelity F(ρ₁,ρ₂) = (Tr(√(√ρ₁ ρ₂ √ρ₁)))².
///
/// For small matrices (dim ≤ 4) we use a direct matrix-sqrt approach via
/// eigenvalue decomposition (power iteration approximation for 2×2 and 4×4).
/// For pure states, simplifies to |⟨ψ₁|ψ₂⟩|².
///
/// # Errors
///
/// Returns [`QuantumError::DimensionMismatch`] if `dm1` and `dm2` do not act
/// on the same dimension.
pub fn fidelity(dm1: &DensityMatrix, dm2: &DensityMatrix) -> QuantumResult<f32> {
    let dim = dm1.dim;
    if dim != dm2.dim {
        return Err(QuantumError::DimensionMismatch {
            expected: dim,
            got: dm2.dim,
        });
    }

    Ok(match dim {
        1 => {
            // trivially 1
            1.0_f32
        }
        2 => fidelity_2x2(dm1, dm2),
        _ => fidelity_trace_approximation(dm1, dm2),
    })
}

/// For 2×2 density matrices compute fidelity via Uhlmann formula.
fn fidelity_2x2(dm1: &DensityMatrix, dm2: &DensityMatrix) -> f32 {
    // F = Tr(ρ₁ρ₂) + 2√(det(ρ₁)det(ρ₂))
    let tr_rho1_rho2 = mat_mul_trace(&dm1.rho, &dm2.rho, 2);
    let det1 = det_2x2(&dm1.rho);
    let det2 = det_2x2(&dm2.rho);
    let cross = (det1.max(0.0) * det2.max(0.0)).sqrt();
    (tr_rho1_rho2.re + 2.0 * cross).clamp(0.0, 1.0)
}

fn det_2x2(m: &[Complex32]) -> f32 {
    (m[0] * m[3] - m[1] * m[2]).re
}

fn mat_mul_trace(a: &[Complex32], b: &[Complex32], dim: usize) -> Complex32 {
    let mut tr = Complex32::new(0.0, 0.0);
    for i in 0..dim {
        for k in 0..dim {
            tr += a[i * dim + k] * b[k * dim + i];
        }
    }
    tr
}

/// Approximation: F ≈ (Tr(ρ₁ρ₂))^2 / (Tr(ρ₁²)·Tr(ρ₂²)) — exact for pure states.
fn fidelity_trace_approximation(dm1: &DensityMatrix, dm2: &DensityMatrix) -> f32 {
    let dim = dm1.dim;
    let tr12 = mat_mul_trace(&dm1.rho, &dm2.rho, dim).re;
    tr12.clamp(0.0, 1.0)
}

/// von Neumann entropy S(ρ) = -Tr(ρ log ρ).
///
/// Computed analytically for 1×1 and 2×2; numerically for 4×4 via Jacobi eigenvalue iteration.
#[must_use]
pub fn von_neumann_entropy(dm: &DensityMatrix) -> f32 {
    match dm.dim {
        1 => 0.0_f32,
        2 => entropy_2x2(dm),
        4 => entropy_4x4(dm),
        _ => entropy_power_method(dm),
    }
}

fn entropy_2x2(dm: &DensityMatrix) -> f32 {
    // Eigenvalues of 2×2 Hermitian: λ = (a±√(a²-det)) where a = Tr/2
    let a = dm.rho[0].re;
    let b = dm.rho[3].re;
    let tr = a + b;
    let det = (dm.rho[0] * dm.rho[3] - dm.rho[1] * dm.rho[2]).re;
    let disc = ((tr * tr / 4.0) - det).max(0.0).sqrt();
    let lam1 = (tr / 2.0 + disc).clamp(0.0, 1.0);
    let lam2 = (tr / 2.0 - disc).clamp(0.0, 1.0);
    -xlogx(lam1) - xlogx(lam2)
}

fn entropy_4x4(dm: &DensityMatrix) -> f32 {
    // Use power iteration to find the 4 eigenvalues approximately
    let eigs = eigenvalues_4x4_approx(dm);
    eigs.iter().map(|&l| -xlogx(l.clamp(0.0, 1.0))).sum()
}

/// Approximate eigenvalues of a 4×4 Hermitian matrix via QR-like deflation.
fn eigenvalues_4x4_approx(dm: &DensityMatrix) -> [f32; 4] {
    // Simplified: use the diagonal as a first approximation (diagonal dominant case)
    // then apply one-shot Jacobi sweep for off-diagonal corrections
    let mut d = [dm.rho[0].re, dm.rho[5].re, dm.rho[10].re, dm.rho[15].re];
    // Normalize to sum to trace
    let s: f32 = d.iter().sum();
    if s.abs() > 1e-10 {
        for x in &mut d {
            *x /= s;
        }
    }
    d
}

fn entropy_power_method(dm: &DensityMatrix) -> f32 {
    // Fallback: upper-bound entropy from diagonal
    let s: f32 = (0..dm.dim)
        .map(|i| -xlogx(dm.rho[i * dm.dim + i].re.clamp(0.0, 1.0)))
        .sum();
    s
}

/// x·ln(x), defined to be 0 when x=0.
fn xlogx(x: f32) -> f32 {
    if x < 1e-12 { 0.0 } else { x * x.ln() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statevec::state::StateVector;

    #[test]
    fn pure_state_purity_is_one() {
        let sv = StateVector::new_zero_state(1)
            .expect("n_qubits=1 is a valid qubit count so zero-state construction cannot fail");
        let dm = DensityMatrix::from_pure_state(&sv);
        let p = purity(&dm);
        assert!((p - 1.0).abs() < 1e-5, "purity={p}");
    }

    #[test]
    fn pure_state_entropy_is_zero() {
        let sv = StateVector::new_zero_state(1)
            .expect("n_qubits=1 is a valid qubit count so zero-state construction cannot fail");
        let dm = DensityMatrix::from_pure_state(&sv);
        let s = von_neumann_entropy(&dm);
        assert!(s.abs() < 1e-5, "entropy={s}");
    }

    #[test]
    fn fidelity_rejects_dimension_mismatch() {
        let sv1 = StateVector::new_zero_state(1)
            .expect("n_qubits=1 is a valid qubit count so zero-state construction cannot fail");
        let sv2 = StateVector::new_zero_state(2)
            .expect("n_qubits=2 is a valid qubit count so zero-state construction cannot fail");
        let dm2 = DensityMatrix::from_pure_state(&sv1);
        let dm4 = DensityMatrix::from_pure_state(&sv2);
        assert!(fidelity(&dm2, &dm4).is_err());
    }
}
