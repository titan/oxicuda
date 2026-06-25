//! Tensor-network contraction back-end for shallow, wide quantum circuits.
//!
//! For circuits that are *shallow* (low depth) but *wide* (many qubits), the
//! full `2^n` state vector is intractable, yet individual output amplitudes
//! `⟨x| U |0…0⟩` can be evaluated by contracting the circuit's tensor network.
//! Each single-qubit gate is a rank-2 tensor, each two-qubit gate a rank-4
//! tensor, and the contraction follows the wire connectivity. This file provides
//!
//! * [`Tensor`] — a dense complex tensor with named [`Index`] legs and a general
//!   pairwise [`Tensor::contract`] (a `tensordot` over all shared legs), plus an
//!   outer product and a scalar extractor;
//! * [`TensorNetwork`] — an ordered collection of tensors contracted greedily;
//! * [`amplitude`] — builds the network for a gate list and returns
//!   `⟨bitstring| U |0…0⟩`.
//!
//! The contraction is *exact*: for any circuit it returns the same amplitude as
//! the dense state-vector simulator, but the per-amplitude cost scales with the
//! treewidth of the circuit graph rather than `2^n`, which is the advantage for
//! shallow wide circuits where only a few amplitudes (or marginals) are needed.

use num_complex::Complex;

use crate::error::{QuantumError, QuantumResult};

type Complex32 = Complex<f32>;

/// A unique label for a tensor leg (a qubit "wire" at a given time slice).
///
/// Two tensors are contracted over every [`Index`] they share.
pub type Index = u32;

/// A dense complex tensor with named legs.
///
/// `legs[d]` is the [`Index`] of dimension `d`; every leg has extent 2 (a qubit
/// wire). Data is stored row-major in leg order: the flat offset of a multi-index
/// `(i_0, …, i_{r-1})` is `Σ_d i_d · 2^{r-1-d}`.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Names of the legs, in storage order.
    pub legs: Vec<Index>,
    /// Dense amplitudes, length `2^legs.len()`.
    pub data: Vec<Complex32>,
}

impl Tensor {
    /// Build a tensor from explicit legs and data, validating the length.
    ///
    /// # Errors
    /// Returns [`QuantumError::DimensionMismatch`] if `data.len() != 2^legs.len()`.
    pub fn new(legs: Vec<Index>, data: Vec<Complex32>) -> QuantumResult<Self> {
        let expected = 1usize << legs.len();
        if data.len() != expected {
            return Err(QuantumError::DimensionMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(Self { legs, data })
    }

    /// Rank-0 scalar tensor.
    #[must_use]
    pub fn scalar(value: Complex32) -> Self {
        Self {
            legs: Vec::new(),
            data: vec![value],
        }
    }

    /// The single-qubit `|0⟩` state vector as a rank-1 tensor on leg `wire`.
    #[must_use]
    pub fn ket_zero(wire: Index) -> Self {
        Self {
            legs: vec![wire],
            data: vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)],
        }
    }

    /// The single-qubit basis bra `⟨b|` as a rank-1 tensor on leg `wire`.
    #[must_use]
    pub fn bra_basis(wire: Index, bit: bool) -> Self {
        let data = if bit {
            vec![Complex32::new(0.0, 0.0), Complex32::new(1.0, 0.0)]
        } else {
            vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 0.0)]
        };
        Self {
            legs: vec![wire],
            data,
        }
    }

    /// A single-qubit gate tensor `out·in` with legs `[out_wire, in_wire]`.
    ///
    /// `m[r][c]` is the matrix entry `⟨r|G|c⟩`; the `out_wire` leg carries the
    /// row index, the `in_wire` leg the column index.
    #[must_use]
    pub fn gate_1q(out_wire: Index, in_wire: Index, m: &[[Complex32; 2]; 2]) -> Self {
        // Storage order [out, in] → offset = out*2 + in.
        let data = vec![m[0][0], m[0][1], m[1][0], m[1][1]];
        Self {
            legs: vec![out_wire, in_wire],
            data,
        }
    }

    /// A two-qubit gate tensor with legs `[out0, out1, in0, in1]`.
    ///
    /// `m` is the `4 × 4` matrix in row-major form using the convention that the
    /// composite index is `2*q0 + q1` (qubit 0 most significant), i.e. the same
    /// ordering used by the dense two-qubit application path.
    #[must_use]
    pub fn gate_2q(
        out0: Index,
        out1: Index,
        in0: Index,
        in1: Index,
        m: &[[Complex32; 4]; 4],
    ) -> Self {
        // Legs [out0,out1,in0,in1]; flat offset = ((out0*2+out1)*2+in0)*2+in1.
        // For two qubits the composite index already equals out0*2+out1 (and
        // in0*2+in1), so the leg layout coincides with the matrix layout and the
        // matrix can be copied row-major directly.
        let mut data = vec![Complex32::new(0.0, 0.0); 16];
        for (o, row) in m.iter().enumerate() {
            for (i, &entry) in row.iter().enumerate() {
                data[o * 4 + i] = entry;
            }
        }
        Self {
            legs: vec![out0, out1, in0, in1],
            data,
        }
    }

    /// Rank of the tensor (number of legs).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.legs.len()
    }

    /// Extract the scalar value of a rank-0 tensor.
    ///
    /// # Errors
    /// Returns [`QuantumError::Internal`] if the tensor is not rank-0.
    pub fn as_scalar(&self) -> QuantumResult<Complex32> {
        if self.legs.is_empty() && self.data.len() == 1 {
            Ok(self.data[0])
        } else {
            Err(QuantumError::Internal {
                msg: format!("tensor is rank {}, not a scalar", self.rank()),
            })
        }
    }

    /// Contract `self` with `other` over **all shared legs** (a `tensordot`).
    ///
    /// The resulting tensor's legs are `self`'s remaining legs followed by
    /// `other`'s remaining legs, both in their original order. Shared legs are
    /// summed over. With no shared legs this reduces to an outer product.
    ///
    /// # Errors
    /// Currently infallible for well-formed inputs; returns a `Result` for
    /// interface uniformity and future fallibility.
    pub fn contract(&self, other: &Self) -> QuantumResult<Self> {
        // Partition legs into shared / free for each operand.
        let mut shared: Vec<Index> = Vec::new();
        for &l in &self.legs {
            if other.legs.contains(&l) {
                shared.push(l);
            }
        }
        let a_free: Vec<Index> = self
            .legs
            .iter()
            .copied()
            .filter(|l| !shared.contains(l))
            .collect();
        let b_free: Vec<Index> = other
            .legs
            .iter()
            .copied()
            .filter(|l| !shared.contains(l))
            .collect();

        let s = shared.len();
        let af = a_free.len();
        let bf = b_free.len();

        // Precompute, for each operand, the bit position (in its flat layout) of
        // every shared and free leg, so we can assemble flat offsets quickly.
        let a_pos = leg_positions(&self.legs);
        let b_pos = leg_positions(&other.legs);

        let mut out_data = vec![Complex32::new(0.0, 0.0); 1usize << (af + bf)];

        // Iterate over all combinations of (free_a, free_b, shared) bit patterns.
        let n_af = 1usize << af;
        let n_bf = 1usize << bf;
        let n_s = 1usize << s;

        for fa in 0..n_af {
            for fb in 0..n_bf {
                let mut acc = Complex32::new(0.0, 0.0);
                for sv in 0..n_s {
                    // Build A's flat index.
                    let mut a_idx = 0usize;
                    for (d, leg) in a_free.iter().enumerate() {
                        let bit = (fa >> (af - 1 - d)) & 1;
                        a_idx |= bit << (self.legs.len() - 1 - a_pos[leg]);
                    }
                    for (d, leg) in shared.iter().enumerate() {
                        let bit = (sv >> (s - 1 - d)) & 1;
                        a_idx |= bit << (self.legs.len() - 1 - a_pos[leg]);
                    }
                    // Build B's flat index.
                    let mut b_idx = 0usize;
                    for (d, leg) in b_free.iter().enumerate() {
                        let bit = (fb >> (bf - 1 - d)) & 1;
                        b_idx |= bit << (other.legs.len() - 1 - b_pos[leg]);
                    }
                    for (d, leg) in shared.iter().enumerate() {
                        let bit = (sv >> (s - 1 - d)) & 1;
                        b_idx |= bit << (other.legs.len() - 1 - b_pos[leg]);
                    }
                    acc += self.data[a_idx] * other.data[b_idx];
                }
                let out_idx = (fa << bf) | fb;
                out_data[out_idx] = acc;
            }
        }

        let mut out_legs = a_free;
        out_legs.extend(b_free);
        Ok(Self {
            legs: out_legs,
            data: out_data,
        })
    }
}

/// Map each leg name to its position in a leg list.
fn leg_positions(legs: &[Index]) -> std::collections::HashMap<Index, usize> {
    legs.iter().enumerate().map(|(i, &l)| (l, i)).collect()
}

/// An unordered bag of tensors to be contracted into a scalar.
#[derive(Debug, Clone, Default)]
pub struct TensorNetwork {
    pub tensors: Vec<Tensor>,
}

impl TensorNetwork {
    /// Empty network.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
        }
    }

    /// Add a tensor to the network.
    pub fn push(&mut self, t: Tensor) {
        self.tensors.push(t);
    }

    /// Greedily contract the whole network and return the resulting tensor.
    ///
    /// The contraction order is a simple greedy heuristic: repeatedly contract
    /// the pair of tensors sharing the most legs (falling back to the first pair
    /// when none share a leg, producing an outer product). For shallow circuits
    /// this keeps intermediate ranks small.
    ///
    /// # Errors
    /// Returns [`QuantumError::EmptyInput`] if the network is empty.
    pub fn contract_all(&self) -> QuantumResult<Tensor> {
        if self.tensors.is_empty() {
            return Err(QuantumError::EmptyInput);
        }
        let mut work: Vec<Tensor> = self.tensors.clone();
        while work.len() > 1 {
            // Find the pair with the most shared legs.
            let mut best = (0usize, 1usize);
            let mut best_shared = -1i64;
            for i in 0..work.len() {
                for j in (i + 1)..work.len() {
                    let shared = work[i]
                        .legs
                        .iter()
                        .filter(|l| work[j].legs.contains(l))
                        .count() as i64;
                    if shared > best_shared {
                        best_shared = shared;
                        best = (i, j);
                    }
                }
            }
            let (i, j) = best;
            // Remove j first (higher index) then i to keep indices valid.
            let tj = work.remove(j);
            let ti = work.remove(i);
            let merged = ti.contract(&tj)?;
            work.push(merged);
        }
        Ok(work.into_iter().next().unwrap_or_else(|| {
            // Unreachable: loop guarantees exactly one element, but avoid panic.
            Tensor::scalar(Complex32::new(0.0, 0.0))
        }))
    }

    /// Contract to a scalar (the full amplitude).
    ///
    /// # Errors
    /// Returns an error if the final tensor is not rank-0.
    pub fn contract_to_scalar(&self) -> QuantumResult<Complex32> {
        self.contract_all()?.as_scalar()
    }
}

/// A circuit gate for the tensor-network amplitude evaluator.
#[derive(Debug, Clone)]
pub enum TnGate {
    /// Single-qubit gate `m` on `qubit`.
    OneQ {
        qubit: usize,
        m: [[Complex32; 2]; 2],
    },
    /// Two-qubit gate `m` (4×4, composite index `2*q0+q1`) on `(q0, q1)`.
    TwoQ {
        q0: usize,
        q1: usize,
        m: [[Complex32; 4]; 4],
    },
}

/// Compute the amplitude `⟨bitstring| U |0…0⟩` for an `n_qubits` circuit by
/// tensor-network contraction.
///
/// `bitstring[q]` is the desired measured value of qubit `q` (little-endian,
/// matching the state-vector index convention). Wires are time-sliced: each gate
/// allocates fresh output [`Index`] legs so the network is a DAG with no leg
/// reuse, then `ket_zero` caps the inputs and `bra_basis` caps the outputs.
///
/// # Errors
/// Returns an error for an invalid qubit count, a `bitstring` of the wrong
/// length, or an out-of-range qubit in any gate.
pub fn amplitude(
    n_qubits: usize,
    gates: &[TnGate],
    bitstring: &[bool],
) -> QuantumResult<Complex32> {
    if n_qubits == 0 || n_qubits > 30 {
        return Err(QuantumError::InvalidQubitCount { n: n_qubits });
    }
    if bitstring.len() != n_qubits {
        return Err(QuantumError::DimensionMismatch {
            expected: n_qubits,
            got: bitstring.len(),
        });
    }

    let mut net = TensorNetwork::new();
    // Fresh-leg allocator.
    let mut next_index: Index = 0;
    let mut fresh = || {
        let i = next_index;
        next_index += 1;
        i
    };

    // current_wire[q] = the leg currently dangling on qubit q's timeline.
    let mut current_wire: Vec<Index> = (0..n_qubits).map(|_| fresh()).collect();
    // Cap each qubit's input with |0⟩.
    for &w in &current_wire {
        net.push(Tensor::ket_zero(w));
    }

    for g in gates {
        match g {
            TnGate::OneQ { qubit, m } => {
                if *qubit >= n_qubits {
                    return Err(QuantumError::QubitIndexOutOfRange {
                        index: *qubit,
                        n_qubits,
                    });
                }
                let in_w = current_wire[*qubit];
                let out_w = fresh();
                net.push(Tensor::gate_1q(out_w, in_w, m));
                current_wire[*qubit] = out_w;
            }
            TnGate::TwoQ { q0, q1, m } => {
                if *q0 >= n_qubits || *q1 >= n_qubits || q0 == q1 {
                    return Err(QuantumError::InvalidParameter {
                        name: "two-qubit gate indices".into(),
                    });
                }
                let in0 = current_wire[*q0];
                let in1 = current_wire[*q1];
                let out0 = fresh();
                let out1 = fresh();
                net.push(Tensor::gate_2q(out0, out1, in0, in1, m));
                current_wire[*q0] = out0;
                current_wire[*q1] = out1;
            }
        }
    }

    // Cap each qubit's output with ⟨bitstring[q]|.
    for (q, &w) in current_wire.iter().enumerate() {
        net.push(Tensor::bra_basis(w, bitstring[q]));
    }

    net.contract_to_scalar()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::controlled::apply_cnot;
    use crate::gates::hadamard::gate_h;
    use crate::gates::pauli::gate_x;
    use crate::handle::LcgRng;
    use crate::statevec::apply_1q::apply_1q_inplace;
    use crate::statevec::state::StateVector;

    fn h_mat() -> [[Complex32; 2]; 2] {
        gate_h()
    }

    fn cnot_mat() -> [[Complex32; 4]; 4] {
        // CNOT with q0 as control, q1 as target, composite index 2*q0+q1.
        let z = Complex32::new(0.0, 0.0);
        let o = Complex32::new(1.0, 0.0);
        [[o, z, z, z], [z, o, z, z], [z, z, z, o], [z, z, o, z]]
    }

    #[test]
    fn scalar_contract_inner_product() {
        // ⟨0|0⟩ = 1.
        let ket = Tensor::ket_zero(0);
        let bra = Tensor::bra_basis(0, false);
        let s = bra
            .contract(&ket)
            .expect("contract")
            .as_scalar()
            .expect("scalar");
        assert!((s.re - 1.0).abs() < 1e-6 && s.im.abs() < 1e-6);
        // ⟨1|0⟩ = 0.
        let bra1 = Tensor::bra_basis(0, true);
        let s2 = bra1
            .contract(&Tensor::ket_zero(0))
            .expect("c")
            .as_scalar()
            .expect("s");
        assert!(s2.norm() < 1e-6);
    }

    #[test]
    fn single_hadamard_amplitudes() {
        // H|0⟩ = (|0⟩ + |1⟩)/√2.
        let gates = vec![TnGate::OneQ {
            qubit: 0,
            m: h_mat(),
        }];
        let a0 = amplitude(1, &gates, &[false]).expect("amp0");
        let a1 = amplitude(1, &gates, &[true]).expect("amp1");
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        assert!((a0.re - inv_sqrt2).abs() < 1e-5, "a0={a0:?}");
        assert!((a1.re - inv_sqrt2).abs() < 1e-5, "a1={a1:?}");
    }

    #[test]
    fn bell_state_amplitudes_match_statevector() {
        // H on q0, CNOT(0→1): Bell state. Compare all 4 amplitudes to statevec.
        let gates = vec![
            TnGate::OneQ {
                qubit: 0,
                m: h_mat(),
            },
            TnGate::TwoQ {
                q0: 0,
                q1: 1,
                m: cnot_mat(),
            },
        ];

        // Reference state vector.
        let mut sv = StateVector::new_zero_state(2).expect("2q");
        apply_1q_inplace(&mut sv, 0, &gate_h()).expect("H");
        apply_cnot(&mut sv, 0, 1).expect("cnot");

        for idx in 0..4usize {
            let b0 = (idx & 1) != 0; // qubit 0 is bit 0 (little-endian)
            let b1 = (idx & 2) != 0;
            let amp = amplitude(2, &gates, &[b0, b1]).expect("amp");
            let reference = sv.amps[idx];
            assert!(
                (amp.re - reference.re).abs() < 1e-5 && (amp.im - reference.im).abs() < 1e-5,
                "idx={idx}: tn={amp:?} sv={reference:?}"
            );
        }
    }

    #[test]
    fn deep_random_circuit_matches_statevector() {
        // 3-qubit moderately deep circuit; check a few amplitudes exactly.
        let mut rng = LcgRng::new(123);
        let mut gates: Vec<TnGate> = Vec::new();
        let mut sv = StateVector::new_zero_state(3).expect("3q");

        for _ in 0..6 {
            // Random single-qubit RY-like gates on each qubit.
            for q in 0..3usize {
                let theta = rng.next_f32() * std::f32::consts::TAU;
                let c = (theta * 0.5).cos();
                let s = (theta * 0.5).sin();
                let m = [
                    [Complex32::new(c, 0.0), Complex32::new(-s, 0.0)],
                    [Complex32::new(s, 0.0), Complex32::new(c, 0.0)],
                ];
                gates.push(TnGate::OneQ { qubit: q, m });
                apply_1q_inplace(&mut sv, q, &m).expect("apply ry");
            }
            // A CNOT ladder.
            gates.push(TnGate::TwoQ {
                q0: 0,
                q1: 1,
                m: cnot_mat(),
            });
            apply_cnot(&mut sv, 0, 1).expect("cnot01");
            gates.push(TnGate::TwoQ {
                q0: 1,
                q1: 2,
                m: cnot_mat(),
            });
            apply_cnot(&mut sv, 1, 2).expect("cnot12");
        }

        for idx in 0..8usize {
            let b0 = (idx & 1) != 0;
            let b1 = (idx & 2) != 0;
            let b2 = (idx & 4) != 0;
            let amp = amplitude(3, &gates, &[b0, b1, b2]).expect("amp");
            let reference = sv.amps[idx];
            assert!(
                (amp.re - reference.re).abs() < 1e-3 && (amp.im - reference.im).abs() < 1e-3,
                "idx={idx}: tn={amp:?} sv={reference:?}"
            );
        }
    }

    #[test]
    fn amplitude_probabilities_sum_to_one() {
        // Random shallow wide-ish circuit: total probability over all bitstrings
        // must be 1 (unitarity), verified via tensor-network amplitudes.
        let mut rng = LcgRng::new(77);
        let n = 4usize;
        let mut gates: Vec<TnGate> = Vec::new();
        for q in 0..n {
            let theta = rng.next_f32() * std::f32::consts::TAU;
            let c = (theta * 0.5).cos();
            let s = (theta * 0.5).sin();
            let m = [
                [Complex32::new(c, 0.0), Complex32::new(-s, 0.0)],
                [Complex32::new(s, 0.0), Complex32::new(c, 0.0)],
            ];
            gates.push(TnGate::OneQ { qubit: q, m });
        }
        gates.push(TnGate::TwoQ {
            q0: 0,
            q1: 1,
            m: cnot_mat(),
        });
        gates.push(TnGate::TwoQ {
            q0: 2,
            q1: 3,
            m: cnot_mat(),
        });

        let mut total = 0.0_f32;
        for idx in 0..(1usize << n) {
            let bits: Vec<bool> = (0..n).map(|q| (idx >> q) & 1 != 0).collect();
            let amp = amplitude(n, &gates, &bits).expect("amp");
            total += amp.norm_sqr();
        }
        assert!((total - 1.0).abs() < 1e-4, "total prob={total}");
    }

    #[test]
    fn x_gate_flips_amplitude() {
        let gates = vec![TnGate::OneQ {
            qubit: 0,
            m: gate_x(),
        }];
        let a1 = amplitude(1, &gates, &[true]).expect("a1");
        let a0 = amplitude(1, &gates, &[false]).expect("a0");
        assert!((a1.re - 1.0).abs() < 1e-6);
        assert!(a0.norm() < 1e-6);
    }

    #[test]
    fn empty_network_errors() {
        let net = TensorNetwork::new();
        assert!(net.contract_all().is_err());
    }
}
