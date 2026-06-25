//! Sparse state-vector representation for low-occupancy circuits.
//!
//! Many circuits of interest keep the state supported on only a *small* subset of
//! the `2^n` computational-basis kets at any time: a permutation circuit (X /
//! CNOT / Toffoli) keeps it a single basis state; an oracle / arithmetic circuit
//! touches only a polynomial number of branches; the early layers of a sparse
//! ansatz stay near `|0…0⟩`. Storing the full dense `2^n` amplitude array is then
//! wasteful.
//!
//! [`SparseStateVector`] stores only the **nonzero** amplitudes in a hash map
//! keyed by the basis index. Single-qubit and CNOT/Toffoli gate application
//! operate directly on the populated entries, allocating new keys only when an
//! amplitude actually becomes nonzero, so the time and memory cost scales with
//! the *occupancy* (number of nonzero amplitudes) rather than with `2^n`.
//! Near-zero amplitudes are pruned after each operation to keep the
//! representation tight.

use std::collections::HashMap;

use num_complex::Complex;

use crate::error::{QuantumError, QuantumResult};
use crate::statevec::state::StateVector;

type Complex32 = Complex<f32>;

/// Amplitudes with magnitude below this are dropped from the sparse map.
const PRUNE_EPS: f32 = 1e-12;

/// A sparse pure state: only nonzero basis amplitudes are stored.
#[derive(Debug, Clone)]
pub struct SparseStateVector {
    /// Map from computational-basis index → amplitude (only nonzero kept).
    amps: HashMap<usize, Complex32>,
    /// Number of qubits.
    pub n_qubits: usize,
}

impl SparseStateVector {
    /// Construct the sparse `|0…0⟩` state (a single populated key).
    ///
    /// # Errors
    /// Returns [`QuantumError::InvalidQubitCount`] for `n_qubits == 0` or
    /// `n_qubits > 30`.
    pub fn new_zero_state(n_qubits: usize) -> QuantumResult<Self> {
        if n_qubits == 0 || n_qubits > 30 {
            return Err(QuantumError::InvalidQubitCount { n: n_qubits });
        }
        let mut amps = HashMap::new();
        amps.insert(0usize, Complex32::new(1.0, 0.0));
        Ok(Self { amps, n_qubits })
    }

    /// Build a sparse representation from a dense state vector (drops zeros).
    #[must_use]
    pub fn from_dense(sv: &StateVector) -> Self {
        let mut amps = HashMap::new();
        for (i, a) in sv.amps.iter().enumerate() {
            if a.norm_sqr() > PRUNE_EPS * PRUNE_EPS {
                amps.insert(i, *a);
            }
        }
        Self {
            amps,
            n_qubits: sv.n_qubits,
        }
    }

    /// Expand back to a dense [`StateVector`].
    ///
    /// # Errors
    /// Propagates the dense constructor's qubit-count validation.
    pub fn to_dense(&self) -> QuantumResult<StateVector> {
        let dim = 1usize << self.n_qubits;
        let mut dense = vec![Complex32::new(0.0, 0.0); dim];
        for (&i, &a) in &self.amps {
            if i < dim {
                dense[i] = a;
            }
        }
        Ok(StateVector {
            amps: dense,
            n_qubits: self.n_qubits,
        })
    }

    /// Number of stored (nonzero) amplitudes.
    #[must_use]
    pub fn occupancy(&self) -> usize {
        self.amps.len()
    }

    /// Squared norm Σ|a|².
    #[must_use]
    pub fn norm_sq(&self) -> f32 {
        self.amps.values().map(|a| a.norm_sqr()).sum()
    }

    /// Amplitude at a basis index (zero if not stored).
    #[must_use]
    pub fn amp(&self, idx: usize) -> Complex32 {
        self.amps
            .get(&idx)
            .copied()
            .unwrap_or_else(|| Complex32::new(0.0, 0.0))
    }

    /// Remove entries that have decayed below the prune threshold.
    fn prune(&mut self) {
        self.amps
            .retain(|_, a| a.norm_sqr() > PRUNE_EPS * PRUNE_EPS);
    }

    /// Apply a single-qubit gate to `qubit`.
    ///
    /// Iterates the current support, pairs each key with its `qubit`-flipped
    /// partner, and writes the mixed amplitudes into a fresh map (so concurrent
    /// reads of the old amplitudes are consistent).
    ///
    /// # Errors
    /// Returns [`QuantumError::QubitIndexOutOfRange`] if `qubit >= n_qubits`.
    pub fn apply_1q(&mut self, qubit: usize, gate: &[[Complex32; 2]; 2]) -> QuantumResult<()> {
        if qubit >= self.n_qubits {
            return Err(QuantumError::QubitIndexOutOfRange {
                index: qubit,
                n_qubits: self.n_qubits,
            });
        }
        let mask = 1usize << qubit;

        // Collect the set of "low" keys (qubit bit = 0) that need processing; for
        // every populated key, its partner is key ^ mask.
        let mut low_keys: Vec<usize> = Vec::new();
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for &k in self.amps.keys() {
            let low = k & !mask;
            if seen.insert(low) {
                low_keys.push(low);
            }
        }

        let mut new_amps = self.amps.clone();
        for low in low_keys {
            let high = low | mask;
            let x0 = self.amp(low);
            let x1 = self.amp(high);
            let y0 = gate[0][0] * x0 + gate[0][1] * x1;
            let y1 = gate[1][0] * x0 + gate[1][1] * x1;
            write_or_remove(&mut new_amps, low, y0);
            write_or_remove(&mut new_amps, high, y1);
        }
        self.amps = new_amps;
        self.prune();
        Ok(())
    }

    /// Apply a CNOT with the given control and target.
    ///
    /// A CNOT only permutes basis indices (flips `tgt` when `ctrl=1`), so this is
    /// a pure key remap that exactly preserves occupancy.
    ///
    /// # Errors
    /// Returns [`QuantumError::QubitIndexOutOfRange`] for an out-of-range index or
    /// [`QuantumError::InvalidParameter`] if `ctrl == tgt`.
    pub fn apply_cnot(&mut self, ctrl: usize, tgt: usize) -> QuantumResult<()> {
        if ctrl >= self.n_qubits {
            return Err(QuantumError::QubitIndexOutOfRange {
                index: ctrl,
                n_qubits: self.n_qubits,
            });
        }
        if tgt >= self.n_qubits {
            return Err(QuantumError::QubitIndexOutOfRange {
                index: tgt,
                n_qubits: self.n_qubits,
            });
        }
        if ctrl == tgt {
            return Err(QuantumError::InvalidParameter {
                name: "ctrl and tgt must differ".into(),
            });
        }
        let cmask = 1usize << ctrl;
        let tmask = 1usize << tgt;
        let mut new_amps = HashMap::with_capacity(self.amps.len());
        for (&k, &a) in &self.amps {
            let nk = if k & cmask != 0 { k ^ tmask } else { k };
            new_amps.insert(nk, a);
        }
        self.amps = new_amps;
        Ok(())
    }

    /// Apply a Toffoli (CCX): flips `tgt` when both controls are 1. Pure key remap.
    ///
    /// # Errors
    /// Returns an error for out-of-range or non-distinct qubit indices.
    pub fn apply_ccx(&mut self, c0: usize, c1: usize, tgt: usize) -> QuantumResult<()> {
        for &q in &[c0, c1, tgt] {
            if q >= self.n_qubits {
                return Err(QuantumError::QubitIndexOutOfRange {
                    index: q,
                    n_qubits: self.n_qubits,
                });
            }
        }
        if c0 == c1 || c0 == tgt || c1 == tgt {
            return Err(QuantumError::InvalidParameter {
                name: "controls and target must be distinct".into(),
            });
        }
        let m0 = 1usize << c0;
        let m1 = 1usize << c1;
        let tmask = 1usize << tgt;
        let mut new_amps = HashMap::with_capacity(self.amps.len());
        for (&k, &a) in &self.amps {
            let nk = if (k & m0 != 0) && (k & m1 != 0) {
                k ^ tmask
            } else {
                k
            };
            new_amps.insert(nk, a);
        }
        self.amps = new_amps;
        Ok(())
    }

    /// Normalize to unit norm (no-op if ~zero).
    pub fn normalize(&mut self) {
        let norm = self.norm_sq().sqrt();
        if norm > 1e-20 {
            let inv = 1.0 / norm;
            for a in self.amps.values_mut() {
                *a *= inv;
            }
        }
    }
}

/// Insert `value` at `key`, or remove the key if `value` is ~zero.
fn write_or_remove(map: &mut HashMap<usize, Complex32>, key: usize, value: Complex32) {
    if value.norm_sqr() > PRUNE_EPS * PRUNE_EPS {
        map.insert(key, value);
    } else {
        map.remove(&key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::controlled::apply_cnot as dense_cnot;
    use crate::gates::hadamard::gate_h;
    use crate::gates::pauli::gate_x;
    use crate::statevec::apply_1q::apply_1q_inplace;

    #[test]
    fn zero_state_occupancy_is_one() {
        let s = SparseStateVector::new_zero_state(5).expect("valid");
        assert_eq!(s.occupancy(), 1);
        assert!((s.norm_sq() - 1.0).abs() < 1e-6);
        assert!((s.amp(0).re - 1.0).abs() < 1e-6);
    }

    #[test]
    fn permutation_circuit_stays_sparse() {
        // X on a few qubits + CNOT ladder keeps occupancy = 1 (a single basis ket).
        let mut s = SparseStateVector::new_zero_state(8).expect("valid");
        s.apply_1q(0, &gate_x()).expect("x0");
        s.apply_1q(3, &gate_x()).expect("x3");
        s.apply_cnot(0, 1).expect("cnot");
        s.apply_ccx(0, 3, 5).expect("ccx");
        assert_eq!(s.occupancy(), 1, "permutation must keep occupancy 1");
        assert!((s.norm_sq() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hadamard_doubles_occupancy() {
        let mut s = SparseStateVector::new_zero_state(3).expect("valid");
        s.apply_1q(0, &gate_h()).expect("h");
        assert_eq!(s.occupancy(), 2);
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        assert!((s.amp(0).re - inv_sqrt2).abs() < 1e-5);
        assert!((s.amp(1).re - inv_sqrt2).abs() < 1e-5);
    }

    #[test]
    fn bell_state_matches_dense() {
        // Build Bell state sparsely and densely; compare amplitudes.
        let mut s = SparseStateVector::new_zero_state(2).expect("valid");
        s.apply_1q(0, &gate_h()).expect("h");
        s.apply_cnot(0, 1).expect("cnot");

        let mut dv = StateVector::new_zero_state(2).expect("valid");
        apply_1q_inplace(&mut dv, 0, &gate_h()).expect("h");
        dense_cnot(&mut dv, 0, 1).expect("cnot");

        let recon = s.to_dense().expect("dense");
        for i in 0..4 {
            assert!(
                (recon.amps[i] - dv.amps[i]).norm() < 1e-5,
                "idx {i}: sparse {:?} dense {:?}",
                recon.amps[i],
                dv.amps[i]
            );
        }
    }

    #[test]
    fn roundtrip_dense_sparse_dense() {
        let mut dv = StateVector::new_zero_state(3).expect("valid");
        apply_1q_inplace(&mut dv, 0, &gate_h()).expect("h0");
        apply_1q_inplace(&mut dv, 2, &gate_h()).expect("h2");
        let sparse = SparseStateVector::from_dense(&dv);
        let back = sparse.to_dense().expect("dense");
        for (a, b) in back.amps.iter().zip(dv.amps.iter()) {
            assert!((a - b).norm() < 1e-6);
        }
    }

    #[test]
    fn out_of_range_rejected() {
        let mut s = SparseStateVector::new_zero_state(2).expect("valid");
        assert!(s.apply_1q(5, &gate_x()).is_err());
        assert!(s.apply_cnot(0, 0).is_err());
    }
}
