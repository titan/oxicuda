//! General single-qubit **Pauli channel** and the **Pauli twirl**.
//!
//! A Pauli channel applies `I`, `X`, `Y`, `Z` with probabilities
//! `(p_I, p_X, p_Y, p_Z)`:
//!
//! ```text
//! ρ ↦ p_I ρ + p_X XρX + p_Y YρY + p_Z ZρZ,   p_I = 1 − p_X − p_Y − p_Z.
//! ```
//!
//! These complement the relaxation/dephasing models in [`crate::channel::noise`]
//! (amplitude/phase damping, depolarizing): the depolarizing channel is the
//! special case `p_X = p_Y = p_Z = p/4`, and the bit-flip, phase-flip and
//! bit-phase-flip channels are the single-axis cases.
//!
//! [`pauli_twirl`] maps an *arbitrary* single-qubit channel to the Pauli
//! channel that shares its diagonal in the Pauli-transfer (χ-matrix) basis:
//!
//! ```text
//! E_twirl(ρ) = (1/4) Σ_{P∈{I,X,Y,Z}} P† E(P ρ P†) P.
//! ```
//!
//! Twirling decoheres a channel into stochastic Pauli noise — the standard
//! tool for turning coherent error models into the Pauli noise assumed by most
//! quantum-error-correction analyses. The twirled probabilities are
//! `p_Q = Σ_i |Tr(Q† K_i)/2|²` over the channel's Kraus operators `{K_i}`.

use num_complex::Complex;

use crate::channel::kraus::KrausChannel;
use crate::error::{QuantumError, QuantumResult};

type Complex32 = Complex<f32>;

#[inline]
fn c(re: f32, im: f32) -> Complex32 {
    Complex32::new(re, im)
}

/// Tolerance for accepting probabilities that are negative / sum above one only
/// by floating-point round-off.
const PROB_EPS: f32 = 1e-4;

/// Build a single-qubit Pauli channel with the given `X`, `Y`, `Z`
/// probabilities. The identity probability is `1 − p_x − p_y − p_z`.
///
/// # Errors
/// [`QuantumError::InvalidParameter`] if any probability is negative or if
/// `p_x + p_y + p_z > 1` (each beyond a small round-off tolerance).
pub fn pauli_channel(p_x: f32, p_y: f32, p_z: f32) -> QuantumResult<KrausChannel> {
    if p_x < -PROB_EPS || p_y < -PROB_EPS || p_z < -PROB_EPS {
        return Err(QuantumError::InvalidParameter {
            name: "Pauli probabilities must be non-negative".into(),
        });
    }
    let sum = p_x + p_y + p_z;
    if sum > 1.0 + PROB_EPS {
        return Err(QuantumError::InvalidParameter {
            name: "Pauli probabilities must sum to at most 1".into(),
        });
    }
    let p_i = (1.0 - sum).max(0.0);

    let si = p_i.max(0.0).sqrt();
    let sx = p_x.max(0.0).sqrt();
    let sy = p_y.max(0.0).sqrt();
    let sz = p_z.max(0.0).sqrt();

    // Row-major 2×2 Kraus operators √p · {I, X, Y, Z}.
    let k_i = vec![c(si, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(si, 0.0)];
    let k_x = vec![c(0.0, 0.0), c(sx, 0.0), c(sx, 0.0), c(0.0, 0.0)];
    let k_y = vec![c(0.0, 0.0), c(0.0, -sy), c(0.0, sy), c(0.0, 0.0)];
    let k_z = vec![c(sz, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-sz, 0.0)];

    KrausChannel::new(vec![k_i, k_x, k_y, k_z], 2)
}

/// Bit-flip channel: applies `X` with probability `p`.
///
/// # Errors
/// As [`pauli_channel`].
pub fn bit_flip_channel(p: f32) -> QuantumResult<KrausChannel> {
    pauli_channel(p, 0.0, 0.0)
}

/// Phase-flip channel: applies `Z` with probability `p`.
///
/// # Errors
/// As [`pauli_channel`].
pub fn phase_flip_channel(p: f32) -> QuantumResult<KrausChannel> {
    pauli_channel(0.0, 0.0, p)
}

/// Bit-phase-flip channel: applies `Y` with probability `p`.
///
/// # Errors
/// As [`pauli_channel`].
pub fn bit_phase_flip_channel(p: f32) -> QuantumResult<KrausChannel> {
    pauli_channel(0.0, p, 0.0)
}

/// Pauli-twirl an arbitrary single-qubit channel, returning the Pauli channel
/// with the same diagonal Pauli-transfer-matrix entries.
///
/// The twirled error probabilities are `p_Q = Σ_i |Tr(Q† K_i)/2|²` for
/// `Q ∈ {I, X, Y, Z}`; these sum to one because `Σ_i K_i† K_i = I`.
///
/// # Errors
/// * [`QuantumError::DimensionMismatch`] if `channel` is not single-qubit.
/// * Propagates [`pauli_channel`] errors (which cannot trigger for a valid
///   CPTP input, but are surfaced rather than hidden).
pub fn pauli_twirl(channel: &KrausChannel) -> QuantumResult<KrausChannel> {
    if channel.dim != 2 {
        return Err(QuantumError::DimensionMismatch {
            expected: 2,
            got: channel.dim,
        });
    }

    let (mut p_x, mut p_y, mut p_z) = (0.0f32, 0.0f32, 0.0f32);
    let i_unit = c(0.0, 1.0);
    for op in &channel.ops {
        // op is row-major [k00, k01, k10, k11].
        let k01 = op[1];
        let k10 = op[2];
        let k00 = op[0];
        let k11 = op[3];
        // Pauli-basis coefficients c_Q = Tr(Q† K) / 2.
        let c_x = (k01 + k10) * 0.5;
        let c_y = i_unit * (k01 - k10) * 0.5;
        let c_z = (k00 - k11) * 0.5;
        p_x += c_x.norm_sqr();
        p_y += c_y.norm_sqr();
        p_z += c_z.norm_sqr();
    }

    pauli_channel(p_x, p_y, p_z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::noise::{amplitude_damping_channel, depolarizing_channel};
    use crate::density::density::DensityMatrix;
    use crate::gates::hadamard::gate_h;
    use crate::statevec::apply_1q::apply_1q_inplace;
    use crate::statevec::state::StateVector;

    fn plus_density() -> DensityMatrix {
        let mut sv =
            StateVector::new_zero_state(1).expect("single-qubit zero state allocation cannot fail");
        apply_1q_inplace(&mut sv, 0, &gate_h())
            .expect("Hadamard on qubit 0 of a 1-qubit state vector cannot fail");
        DensityMatrix::from_pure_state(&sv)
    }

    fn max_diff(a: &DensityMatrix, b: &DensityMatrix) -> f32 {
        a.rho
            .iter()
            .zip(b.rho.iter())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f32::max)
    }

    #[test]
    fn pauli_channel_completeness_holds() {
        // KrausChannel::new validates Σ K†K = I; construction succeeding proves it.
        assert!(pauli_channel(0.1, 0.2, 0.3).is_ok());
        assert!(pauli_channel(0.0, 0.0, 0.0).is_ok()); // identity channel
        assert!(pauli_channel(1.0, 0.0, 0.0).is_ok()); // deterministic X
    }

    #[test]
    fn pauli_channel_rejects_invalid_probabilities() {
        assert!(pauli_channel(0.5, 0.5, 0.5).is_err()); // sum 1.5 > 1
        assert!(pauli_channel(-0.1, 0.0, 0.0).is_err()); // negative
    }

    #[test]
    fn bit_flip_channel_flips_zero_state() {
        // X with prob p maps |0⟩⟨0| ↦ (1−p)|0⟩⟨0| + p|1⟩⟨1|.
        let p = 0.3f32;
        let ch = bit_flip_channel(p)
            .expect("bit-flip channel with probability 0.3 is within valid bounds");
        let sv =
            StateVector::new_zero_state(1).expect("single-qubit zero state allocation cannot fail");
        let dm = DensityMatrix::from_pure_state(&sv);
        let out = ch
            .apply(&dm)
            .expect("applying a Pauli channel to a valid density matrix cannot fail");
        assert!(
            (out.rho[0].re - (1.0 - p)).abs() < 1e-5,
            "rho00={}",
            out.rho[0].re
        );
        assert!((out.rho[3].re - p).abs() < 1e-5, "rho11={}", out.rho[3].re);
        assert!(out.rho[1].norm() < 1e-6 && out.rho[2].norm() < 1e-6);
        assert!((out.trace().re - 1.0).abs() < 1e-5);
    }

    #[test]
    fn phase_flip_channel_kills_coherence_of_plus() {
        // Z with prob p maps |+⟩⟨+| coherence 1/2 ↦ (1−2p)/2.
        let p = 0.25f32;
        let ch = phase_flip_channel(p)
            .expect("phase-flip channel with probability 0.25 is within valid bounds");
        let out = ch
            .apply(&plus_density())
            .expect("applying a Pauli channel to a valid density matrix cannot fail");
        let expected_off = (1.0 - 2.0 * p) / 2.0;
        assert!(
            (out.rho[1].re - expected_off).abs() < 1e-5,
            "off={}",
            out.rho[1].re
        );
    }

    #[test]
    fn pauli_channel_reproduces_depolarizing() {
        // Depolarizing(p) ≡ Pauli(p/4, p/4, p/4).
        let p = 0.4f32;
        let pauli = pauli_channel(p / 4.0, p / 4.0, p / 4.0)
            .expect("Pauli channel with equal probabilities summing to 0.3 is valid");
        let depol = depolarizing_channel(p, 2)
            .expect("depolarizing channel with probability 0.4 on a 2-dim space is valid");
        let dm = plus_density();
        let a = pauli
            .apply(&dm)
            .expect("applying a Pauli channel to a valid plus-state density matrix cannot fail");
        let b = depol.apply(&dm).expect(
            "applying a depolarizing channel to a valid plus-state density matrix cannot fail",
        );
        assert!(max_diff(&a, &b) < 1e-5, "max_diff={}", max_diff(&a, &b));
    }

    #[test]
    fn pauli_twirl_is_identity_on_a_pauli_channel() {
        // Twirling a channel that is already Pauli returns the same channel.
        let original = bit_flip_channel(0.2)
            .expect("bit-flip channel with probability 0.2 is within valid bounds");
        let twirled = pauli_twirl(&original)
            .expect("twirling a valid single-qubit Pauli channel must succeed");
        let dm = plus_density();
        let a = original
            .apply(&dm)
            .expect("applying the original Pauli channel to a valid density matrix cannot fail");
        let b = twirled
            .apply(&dm)
            .expect("applying the twirled Pauli channel to a valid density matrix cannot fail");
        assert!(max_diff(&a, &b) < 1e-5, "max_diff={}", max_diff(&a, &b));
    }

    #[test]
    fn pauli_twirl_of_amplitude_damping_matches_analytic_pauli_channel() {
        // Twirling amplitude damping γ yields Pauli(γ/4, γ/4, ((1−√(1−γ))/2)²).
        let gamma = 0.3f32;
        let ad_channel = amplitude_damping_channel(gamma)
            .expect("amplitude-damping channel with γ=0.3 must construct successfully");
        let twirled = pauli_twirl(&ad_channel)
            .expect("Pauli-twirling a valid single-qubit amplitude-damping channel must succeed");
        let s = (1.0 - gamma).sqrt();
        let p_xy = gamma / 4.0;
        let p_z = ((1.0 - s) / 2.0).powi(2);
        let expected = pauli_channel(p_xy, p_xy, p_z)
            .expect("analytically derived Pauli probabilities from amplitude damping are valid");
        // Compare on |+⟩ (sensitive to X, Y and Z error components).
        let dm = plus_density();
        let a = twirled.apply(&dm).expect(
            "twirled amplitude-damping channel applied to a valid density matrix cannot fail",
        );
        let b = expected
            .apply(&dm)
            .expect("analytic Pauli channel applied to a valid density matrix cannot fail");
        assert!(max_diff(&a, &b) < 1e-5, "max_diff={}", max_diff(&a, &b));
        // Twirled channel is trace preserving.
        assert!((a.trace().re - 1.0).abs() < 1e-5);
    }

    #[test]
    fn pauli_twirl_rejects_multiqubit_channel() {
        // A 2-qubit (dim-4) identity channel is not single-qubit.
        let id4: Vec<Complex32> = (0..16)
            .map(|i| if i % 5 == 0 { c(1.0, 0.0) } else { c(0.0, 0.0) })
            .collect();
        let ch = KrausChannel::new(vec![id4], 4)
            .expect("4×4 identity Kraus channel (one operator) is a valid CPTP map");
        assert!(pauli_twirl(&ch).is_err());
    }
}
