use num_complex::Complex;

use crate::channel::kraus::KrausChannel;
use crate::error::QuantumResult;

type Complex32 = Complex<f32>;

#[inline]
fn c(re: f32, im: f32) -> Complex32 {
    Complex32::new(re, im)
}

/// Depolarizing channel: ρ' = (1-p)ρ + (p/4)(Iρ + XρX + YρY + ZρZ) for a single qubit.
///
/// This is equivalent to: K0 = √(1-3p/4) I, K1 = √(p/4) X, K2 = √(p/4) Y, K3 = √(p/4) Z.
/// `p` ∈ [0, 1].
pub fn depolarizing_channel(p: f32, dim: usize) -> QuantumResult<KrausChannel> {
    let p_each = p / 4.0;
    let k0_coeff = (1.0 - 3.0 * p_each).sqrt().max(0.0);
    let kn_coeff = p_each.sqrt().max(0.0);

    // For single-qubit (dim=2)
    let _ = dim; // explicit dim for future multi-qubit extension
    let k0 = vec![c(k0_coeff, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(k0_coeff, 0.0)];
    let k1 = vec![c(0.0, 0.0), c(kn_coeff, 0.0), c(kn_coeff, 0.0), c(0.0, 0.0)];
    let k2 = vec![
        c(0.0, 0.0),
        c(0.0, -kn_coeff),
        c(0.0, kn_coeff),
        c(0.0, 0.0),
    ];
    let k3 = vec![
        c(kn_coeff, 0.0),
        c(0.0, 0.0),
        c(0.0, 0.0),
        c(-kn_coeff, 0.0),
    ];

    KrausChannel::new(vec![k0, k1, k2, k3], 2)
}

/// Amplitude damping channel: models T1 relaxation (|1⟩ → |0⟩ decay).
///
/// K0 = [[1, 0], [0, √(1-γ)]], K1 = [[0, √γ], [0, 0]].
pub fn amplitude_damping_channel(gamma: f32) -> QuantumResult<KrausChannel> {
    let sqrt_gamma = gamma.sqrt().max(0.0);
    let sqrt_1mg = (1.0 - gamma).sqrt().max(0.0);

    let k0 = vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(sqrt_1mg, 0.0)];
    let k1 = vec![c(0.0, 0.0), c(sqrt_gamma, 0.0), c(0.0, 0.0), c(0.0, 0.0)];

    KrausChannel::new(vec![k0, k1], 2)
}

/// Phase damping channel: models T2 dephasing without energy relaxation.
///
/// K0 = [[1, 0], [0, √(1-γ)]], K1 = [[0, 0], [0, √γ]].
pub fn phase_damping_channel(gamma: f32) -> QuantumResult<KrausChannel> {
    let sqrt_gamma = gamma.sqrt().max(0.0);
    let sqrt_1mg = (1.0 - gamma).sqrt().max(0.0);

    let k0 = vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(sqrt_1mg, 0.0)];
    let k1 = vec![c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(sqrt_gamma, 0.0)];

    KrausChannel::new(vec![k0, k1], 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::density::density::DensityMatrix;
    use crate::statevec::state::StateVector;

    #[test]
    fn depolarizing_zero_p_is_identity() {
        let ch = depolarizing_channel(0.0, 2).expect("valid depolarizing channel with p=0");
        let sv = StateVector::new_zero_state(1).expect("valid 1-qubit zero state");
        let dm = DensityMatrix::from_pure_state(&sv);
        let out = ch.apply(&dm).expect("channel application must succeed");
        assert!((out.rho[0].re - 1.0).abs() < 1e-5);
    }

    #[test]
    fn amplitude_damping_gamma_zero_is_identity() {
        let ch =
            amplitude_damping_channel(0.0).expect("valid amplitude damping channel with gamma=0");
        let sv = StateVector::new_zero_state(1).expect("valid 1-qubit zero state");
        let dm = DensityMatrix::from_pure_state(&sv);
        let out = ch.apply(&dm).expect("channel application must succeed");
        assert!((out.rho[0].re - 1.0).abs() < 1e-5);
    }
}
