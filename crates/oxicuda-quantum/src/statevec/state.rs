use num_complex::Complex;

use crate::error::{QuantumError, QuantumResult};
use crate::handle::LcgRng;

type Complex32 = Complex<f32>;

/// Pure quantum state represented as a complex amplitude vector.
#[derive(Debug, Clone)]
pub struct StateVector {
    pub amps: Vec<Complex32>,
    pub n_qubits: usize,
}

impl StateVector {
    /// Construct |0⟩^n = [1, 0, 0, …].
    pub fn new_zero_state(n_qubits: usize) -> QuantumResult<Self> {
        if n_qubits == 0 {
            return Err(QuantumError::InvalidQubitCount { n: 0 });
        }
        if n_qubits > 30 {
            return Err(QuantumError::InvalidQubitCount { n: n_qubits });
        }
        let dim = 1usize << n_qubits;
        let mut amps = vec![Complex32::new(0.0, 0.0); dim];
        amps[0] = Complex32::new(1.0, 0.0);
        Ok(Self { amps, n_qubits })
    }

    /// Construct from a given amplitude vector; validates length = 2^n and unit norm.
    pub fn new_from_amps(amps: Vec<Complex32>, n_qubits: usize) -> QuantumResult<Self> {
        if n_qubits == 0 || n_qubits > 30 {
            return Err(QuantumError::InvalidQubitCount { n: n_qubits });
        }
        let expected = 1usize << n_qubits;
        if amps.len() != expected {
            return Err(QuantumError::DimensionMismatch {
                expected,
                got: amps.len(),
            });
        }
        let sv = Self { amps, n_qubits };
        let norm = sv.norm_sq().sqrt();
        if (norm - 1.0).abs() > 1e-4 {
            return Err(QuantumError::NonNormalizedState { norm });
        }
        Ok(sv)
    }

    /// Sum of squared magnitudes: ‖ψ‖².
    #[must_use]
    pub fn norm_sq(&self) -> f32 {
        self.amps.iter().map(|a| a.norm_sqr()).sum()
    }

    /// Normalize in place; no-op if already unit.
    pub fn normalize_inplace(&mut self) {
        let norm = self.norm_sq().sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            for a in &mut self.amps {
                *a *= inv;
            }
        }
    }

    /// Inner product ⟨self|other⟩ = Σ conj(self\[i\]) * other\[i\].
    pub fn inner_product(&self, other: &Self) -> QuantumResult<Complex32> {
        if self.n_qubits != other.n_qubits {
            return Err(QuantumError::DimensionMismatch {
                expected: self.amps.len(),
                got: other.amps.len(),
            });
        }
        Ok(self
            .amps
            .iter()
            .zip(other.amps.iter())
            .map(|(a, b)| a.conj() * b)
            .fold(Complex32::new(0.0, 0.0), |acc, x| acc + x))
    }

    /// Probability of measuring `qubit` in state `outcome`.
    pub fn measure_prob(&self, qubit: usize, outcome: bool) -> QuantumResult<f32> {
        if qubit >= self.n_qubits {
            return Err(QuantumError::QubitIndexOutOfRange {
                index: qubit,
                n_qubits: self.n_qubits,
            });
        }
        let mask = 1usize << qubit;
        let prob = self
            .amps
            .iter()
            .enumerate()
            .filter(|(i, _)| ((i & mask) != 0) == outcome)
            .map(|(_, a)| a.norm_sqr())
            .sum();
        Ok(prob)
    }

    /// Sample a measurement of `qubit`, returning (outcome, post-measurement StateVector).
    pub fn sample_measure(&self, qubit: usize, rng: &mut LcgRng) -> QuantumResult<(bool, Self)> {
        let p1 = self.measure_prob(qubit, true)?;
        let r = rng.next_f32();
        let outcome = r < p1;

        let mask = 1usize << qubit;
        let mut new_amps = self.amps.clone();
        let mut norm_sq = 0.0_f32;

        for (i, a) in new_amps.iter_mut().enumerate() {
            let bit_set = (i & mask) != 0;
            if bit_set != outcome {
                *a = Complex32::new(0.0, 0.0);
            } else {
                norm_sq += a.norm_sqr();
            }
        }

        let norm = norm_sq.sqrt();
        if norm < 1e-12 {
            return Err(QuantumError::MeasurementFailed);
        }
        let inv = 1.0 / norm;
        for a in &mut new_amps {
            *a *= inv;
        }

        Ok((
            outcome,
            Self {
                amps: new_amps,
                n_qubits: self.n_qubits,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_state_norm_is_one() {
        let sv = StateVector::new_zero_state(3).unwrap();
        let n = sv.norm_sq();
        assert!((n - 1.0).abs() < 1e-6, "norm={n}");
    }

    #[test]
    fn invalid_qubit_count() {
        assert!(StateVector::new_zero_state(0).is_err());
    }

    #[test]
    fn from_amps_rejects_unnormalized() {
        let amps = vec![Complex32::new(1.0, 0.0), Complex32::new(1.0, 0.0)];
        assert!(StateVector::new_from_amps(amps, 1).is_err());
    }
}
