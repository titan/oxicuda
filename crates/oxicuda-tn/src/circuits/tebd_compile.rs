//! Compile a [`Circuit`] into TEBD-compatible two-site gate sequences.
//!
//! The output is a `Vec<(bond_index, [f64; 16])>` where each entry describes a
//! two-site gate to apply at the bond between sites `bond_index` and
//! `bond_index + 1`.  Single-qubit gates are embedded into 2-site gates by
//! tensoring with the identity on the neighbouring site.

use crate::circuits::mod_impl::{Circuit, CircuitGate};
use crate::{TnError, TnResult};

/// Embed a single-qubit 2×2 gate on the **left** site of a bond as a 4×4 gate.
///
/// `(U ⊗ I₂)` in the basis `{|00⟩,|01⟩,|10⟩,|11⟩}`:
///
/// ```text
/// [ u[0] 0  u[1] 0  ]
/// [ 0  u[0]  0  u[1] ]   — wait, let's think carefully:
/// ```
///
/// With row-major ordering `(p1, p2)`:
/// `M[p1, p2, p1', p2'] = U[p1, p1'] * δ[p2, p2']`
/// which flattened `(p1*2+p2, p1'*2+p2')` gives a 4×4 matrix.
fn embed_left(u: &[f64; 4]) -> [f64; 16] {
    let mut m = [0.0f64; 16];
    // d = 2
    for p1 in 0..2usize {
        for p2 in 0..2usize {
            for p1p in 0..2usize {
                // p2' must equal p2 (identity on right)
                let row = p1 * 2 + p2;
                let col = p1p * 2 + p2;
                m[row * 4 + col] = u[p1 * 2 + p1p];
            }
        }
    }
    m
}

/// Embed a single-qubit 2×2 gate on the **right** site of a bond as a 4×4 gate.
///
/// `(I₂ ⊗ U)` in the basis `{|00⟩,|01⟩,|10⟩,|11⟩}`:
/// `M[p1, p2, p1', p2'] = δ[p1, p1'] * U[p2, p2']`
fn embed_right(u: &[f64; 4]) -> [f64; 16] {
    let mut m = [0.0f64; 16];
    for p1 in 0..2usize {
        for p2 in 0..2usize {
            for p2p in 0..2usize {
                // p1' must equal p1 (identity on left)
                let row = p1 * 2 + p2;
                let col = p1 * 2 + p2p;
                m[row * 4 + col] = u[p2 * 2 + p2p];
            }
        }
    }
    m
}

/// Compile a [`Circuit`] into an ordered sequence of TEBD two-site gates.
///
/// Each entry in the output is `(bond_index, gate_matrix)` where `gate_matrix`
/// is a 16-element row-major `[4×4]` gate to apply at the bond between sites
/// `bond_index` and `bond_index + 1`.
///
/// **Single-qubit gates** are embedded via `U ⊗ I` on the bond to the right of
/// the qubit, or `I ⊗ U` on the bond to the left — whichever keeps the qubit
/// inside the lattice.  For a single-qubit gate on site `q`:
/// - If `q + 1 < n_qubits`: embed as `(U ⊗ I)` at bond `q`.
/// - Else embed as `(I ⊗ U)` at bond `q - 1`.
///
/// **Two-qubit gates** are returned directly.  Only adjacent gates
/// (`|q1 - q2| == 1`) are supported; non-adjacent gates return an error.
pub fn compile_circuit_to_tebd_gates(circuit: &Circuit) -> TnResult<Vec<(usize, [f64; 16])>> {
    let n = circuit.n_qubits;
    if n < 2 {
        return Err(TnError::InvalidConfiguration(
            "TEBD compilation requires at least 2 qubits".into(),
        ));
    }

    let mut result: Vec<(usize, [f64; 16])> = Vec::with_capacity(circuit.gates.len());

    for gate in &circuit.gates {
        match gate {
            CircuitGate::Single { qubit, matrix } => {
                let q = *qubit;
                if q >= n {
                    return Err(TnError::IndexOutOfBounds { index: q, len: n });
                }
                let (bond, embedded) = if q + 1 < n {
                    (q, embed_left(matrix))
                } else {
                    // q == n - 1: use bond q-1, embed on right
                    (q - 1, embed_right(matrix))
                };
                result.push((bond, embedded));
            }
            CircuitGate::Two {
                qubit1,
                qubit2,
                matrix,
            } => {
                let (q1, q2) = (*qubit1, *qubit2);
                if q1 >= n {
                    return Err(TnError::IndexOutOfBounds { index: q1, len: n });
                }
                if q2 >= n {
                    return Err(TnError::IndexOutOfBounds { index: q2, len: n });
                }
                // Only adjacent qubits are supported.
                if q1.abs_diff(q2) != 1 {
                    return Err(TnError::InvalidConfiguration(format!(
                        "TEBD only supports adjacent two-qubit gates, got qubits {q1} and {q2}"
                    )));
                }
                let bond = q1.min(q2);
                result.push((bond, *matrix));
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits::gates;
    use crate::circuits::mod_impl::Circuit;

    #[test]
    fn bell_circuit_produces_two_tebd_gates() {
        // H on qubit 0, CNOT on (0,1): should compile to 2 TEBD gate entries.
        let mut circ = Circuit::new(2);
        circ.h(0).expect("h should succeed");
        circ.cnot(0, 1).expect("cnot should succeed");
        let compiled = compile_circuit_to_tebd_gates(&circ)
            .expect("compile_circuit_to_tebd_gates should succeed");
        assert_eq!(
            compiled.len(),
            2,
            "Bell circuit should produce 2 TEBD gates"
        );
        // First gate at bond 0 (H on qubit 0, embedded left)
        assert_eq!(compiled[0].0, 0);
        // Second gate at bond 0 (CNOT on qubits 0-1)
        assert_eq!(compiled[1].0, 0);
    }

    #[test]
    fn single_qubit_on_last_site_uses_right_embed() {
        let mut circ = Circuit::new(3);
        circ.x(2).expect("x should succeed"); // qubit 2, last site → bond 1, embed right
        let compiled = compile_circuit_to_tebd_gates(&circ)
            .expect("compile_circuit_to_tebd_gates should succeed");
        assert_eq!(compiled.len(), 1);
        assert_eq!(compiled[0].0, 1, "Last qubit gate should use bond n-2");
    }

    #[test]
    fn non_adjacent_two_qubit_gate_is_error() {
        let mut circ = Circuit::new(4);
        // Force a non-adjacent gate bypassing the API check (inject directly)
        circ.gates.push(CircuitGate::Two {
            qubit1: 0,
            qubit2: 2,
            matrix: gates::cnot(),
        });
        let r = compile_circuit_to_tebd_gates(&circ);
        assert!(r.is_err(), "Non-adjacent gate should be an error");
    }

    #[test]
    fn empty_circuit_compiles_to_empty() {
        let circ = Circuit::new(4);
        let compiled = compile_circuit_to_tebd_gates(&circ)
            .expect("compile_circuit_to_tebd_gates should succeed");
        assert!(compiled.is_empty());
    }

    #[test]
    fn single_qubit_only_circuit_embeds_correctly() {
        let mut circ = Circuit::new(3);
        circ.h(0).expect("h should succeed");
        circ.z(1).expect("z should succeed");
        let compiled = compile_circuit_to_tebd_gates(&circ)
            .expect("compile_circuit_to_tebd_gates should succeed");
        assert_eq!(compiled.len(), 2);
        // H on qubit 0 → bond 0 (embed left)
        assert_eq!(compiled[0].0, 0);
        // Z on qubit 1 → bond 1 (embed left, since 1+1<3)
        assert_eq!(compiled[1].0, 1);
    }

    #[test]
    fn single_qubit_too_few_qubits_errors() {
        let circ = Circuit::new(1);
        let r = compile_circuit_to_tebd_gates(&circ);
        assert!(r.is_err(), "1-qubit circuit should fail TEBD compilation");
    }

    #[test]
    fn embed_left_identity_on_right() {
        // U = identity → U⊗I = identity_4
        let id2: [f64; 4] = [1.0, 0.0, 0.0, 1.0];
        let m = embed_left(&id2);
        // Check diagonal
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (m[i * 4 + j] - expected).abs() < 1e-14,
                    "embed_left(I)[{i},{j}] = {}",
                    m[i * 4 + j]
                );
            }
        }
    }

    #[test]
    fn embed_right_identity_on_left() {
        let id2: [f64; 4] = [1.0, 0.0, 0.0, 1.0];
        let m = embed_right(&id2);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (m[i * 4 + j] - expected).abs() < 1e-14,
                    "embed_right(I)[{i},{j}] = {}",
                    m[i * 4 + j]
                );
            }
        }
    }
}
