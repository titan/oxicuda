//! Core [`Circuit`] implementation: gate accumulation and MPS execution.

use crate::circuits::gates;
use crate::mps::tensor::MpsTensor;
use crate::mps::truncation::svd_truncate;
use crate::svd::svd_jacobi;
use crate::{TnError, TnResult};

// ─── Data structures ──────────────────────────────────────────────────────────

/// A single gate stored inside a [`Circuit`].
#[derive(Debug, Clone)]
pub enum CircuitGate {
    /// Single-qubit gate: a 2×2 real matrix stored row-major.
    Single { qubit: usize, matrix: [f64; 4] },
    /// Two-qubit gate: a 4×4 real matrix stored row-major in the basis
    /// `{|00⟩, |01⟩, |10⟩, |11⟩}`.
    Two {
        qubit1: usize,
        qubit2: usize,
        matrix: [f64; 16],
    },
}

/// Configuration for MPS simulation of a circuit.
#[derive(Debug, Clone, Copy)]
pub struct CircuitConfig {
    /// Maximum bond dimension after each TEBD gate.
    pub chi_max: usize,
    /// SVD truncation tolerance (relative to the largest singular value).
    pub svd_tol: f64,
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self {
            chi_max: 64,
            svd_tol: 1.0e-10,
        }
    }
}

/// A quantum circuit: ordered list of single- and two-qubit gates.
///
/// Gates are applied to an MPS via [`apply_to_mps`][Circuit::apply_to_mps],
/// which iterates through the gate list and updates the site tensors in place.
#[derive(Debug, Clone)]
pub struct Circuit {
    /// Number of qubits (= number of MPS sites).
    pub n_qubits: usize,
    /// Ordered list of gates.
    pub gates: Vec<CircuitGate>,
}

impl Circuit {
    /// Create an empty circuit with `n_qubits` qubits.
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            gates: Vec::new(),
        }
    }

    // ── Gate builders ─────────────────────────────────────────────────────────

    /// Hadamard gate on `qubit`.
    pub fn h(&mut self, qubit: usize) -> TnResult<()> {
        self.add_single(qubit, gates::hadamard())
    }

    /// Pauli-X (NOT) gate on `qubit`.
    pub fn x(&mut self, qubit: usize) -> TnResult<()> {
        self.add_single(qubit, gates::pauli_x())
    }

    /// Pauli-Y gate (real approximation) on `qubit`.
    pub fn y(&mut self, qubit: usize) -> TnResult<()> {
        self.add_single(qubit, gates::pauli_y())
    }

    /// Pauli-Z gate on `qubit`.
    pub fn z(&mut self, qubit: usize) -> TnResult<()> {
        self.add_single(qubit, gates::pauli_z())
    }

    /// Rx(θ) rotation on `qubit`.
    pub fn rx(&mut self, qubit: usize, theta: f64) -> TnResult<()> {
        self.add_single(qubit, gates::rx(theta))
    }

    /// Ry(θ) rotation on `qubit`.
    pub fn ry(&mut self, qubit: usize, theta: f64) -> TnResult<()> {
        self.add_single(qubit, gates::ry(theta))
    }

    /// Rz(θ) — real approximation — on `qubit`.
    pub fn rz(&mut self, qubit: usize, theta: f64) -> TnResult<()> {
        self.add_single(qubit, gates::rz_real(theta))
    }

    /// CNOT (CX) gate with `control` and `target` qubits.  Only adjacent
    /// qubits (`|control - target| == 1`) are supported.
    pub fn cnot(&mut self, control: usize, target: usize) -> TnResult<()> {
        self.add_two(control, target, gates::cnot())
    }

    /// CZ gate on `qubit1` and `qubit2`.
    pub fn cz(&mut self, qubit1: usize, qubit2: usize) -> TnResult<()> {
        self.add_two(qubit1, qubit2, gates::cz())
    }

    /// SWAP gate on `qubit1` and `qubit2`.
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) -> TnResult<()> {
        self.add_two(qubit1, qubit2, gates::swap())
    }

    /// Controlled-U gate: applies `u` (a 2×2 real matrix) to `target` when
    /// `control` is `|1⟩`.
    pub fn cu(&mut self, control: usize, target: usize, u: &[f64; 4]) -> TnResult<()> {
        self.add_two(control, target, gates::controlled_u(u))
    }

    // ── Gate introspection ────────────────────────────────────────────────────

    /// Total number of gates in the circuit.
    pub fn depth(&self) -> usize {
        self.gates.len()
    }

    /// Number of two-qubit gates in the circuit.
    pub fn n_two_qubit_gates(&self) -> usize {
        self.gates
            .iter()
            .filter(|g| matches!(g, CircuitGate::Two { .. }))
            .count()
    }

    // ── MPS execution ─────────────────────────────────────────────────────────

    /// Apply the circuit to an MPS described by raw data and shapes.
    ///
    /// # Parameters
    ///
    /// - `mps_data`: one `Vec<f64>` per site, row-major data of the site tensor.
    /// - `mps_shapes`: one `[d_l, d_p, d_r]` per site.
    /// - `config`: SVD truncation parameters.
    ///
    /// # Returns
    ///
    /// `(new_data, new_shapes)` after applying all gates.
    ///
    /// # Errors
    ///
    /// - [`TnError::IndexOutOfBounds`] if any qubit index ≥ `n_qubits` or ≥ the
    ///   number of sites in `mps_data`.
    /// - [`TnError::InvalidConfiguration`] if a two-qubit gate acts on non-adjacent
    ///   qubits.
    /// - [`TnError::ShapeMismatch`] / [`TnError::DimensionMismatch`] on internal
    ///   tensor bookkeeping failures.
    #[allow(clippy::type_complexity)]
    pub fn apply_to_mps(
        &self,
        mps_data: &[Vec<f64>],
        mps_shapes: &[[usize; 3]],
        config: &CircuitConfig,
    ) -> TnResult<(Vec<Vec<f64>>, Vec<[usize; 3]>)> {
        let n_sites = mps_data.len();
        if mps_shapes.len() != n_sites {
            return Err(TnError::ShapeMismatch {
                expected: vec![n_sites],
                got: vec![mps_shapes.len()],
            });
        }
        if n_sites != self.n_qubits {
            return Err(TnError::InvalidConfiguration(format!(
                "Circuit has {} qubits but MPS has {} sites",
                self.n_qubits, n_sites
            )));
        }

        // Build mutable MpsTensor list from input.
        let mut tensors: Vec<MpsTensor> = mps_data
            .iter()
            .zip(mps_shapes.iter())
            .map(|(data, &[d_l, d_p, d_r])| MpsTensor::new(d_l, d_p, d_r, data.clone()))
            .collect::<TnResult<_>>()?;

        // Apply gates one by one.
        for gate in &self.gates {
            match gate {
                CircuitGate::Single { qubit, matrix } => {
                    let q = *qubit;
                    if q >= n_sites {
                        return Err(TnError::IndexOutOfBounds {
                            index: q,
                            len: n_sites,
                        });
                    }
                    apply_single_qubit_gate(&mut tensors[q], matrix)?;
                }
                CircuitGate::Two {
                    qubit1,
                    qubit2,
                    matrix,
                } => {
                    let (q1, q2) = (*qubit1, *qubit2);
                    if q1 >= n_sites {
                        return Err(TnError::IndexOutOfBounds {
                            index: q1,
                            len: n_sites,
                        });
                    }
                    if q2 >= n_sites {
                        return Err(TnError::IndexOutOfBounds {
                            index: q2,
                            len: n_sites,
                        });
                    }
                    if q1.abs_diff(q2) != 1 {
                        return Err(TnError::InvalidConfiguration(format!(
                            "Only adjacent two-qubit gates supported; got qubits {q1} and {q2}"
                        )));
                    }
                    let left_site = q1.min(q2);
                    apply_two_qubit_gate_tebd(&mut tensors, left_site, matrix, config)?;
                }
            }
        }

        // Reconstruct output vecs.
        let new_shapes: Vec<[usize; 3]> = tensors.iter().map(|t| [t.d_l, t.d_p, t.d_r]).collect();
        let new_data: Vec<Vec<f64>> = tensors.into_iter().map(|t| t.data).collect();

        Ok((new_data, new_shapes))
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn add_single(&mut self, qubit: usize, matrix: [f64; 4]) -> TnResult<()> {
        if qubit >= self.n_qubits {
            return Err(TnError::IndexOutOfBounds {
                index: qubit,
                len: self.n_qubits,
            });
        }
        self.gates.push(CircuitGate::Single { qubit, matrix });
        Ok(())
    }

    fn add_two(&mut self, qubit1: usize, qubit2: usize, matrix: [f64; 16]) -> TnResult<()> {
        if qubit1 >= self.n_qubits {
            return Err(TnError::IndexOutOfBounds {
                index: qubit1,
                len: self.n_qubits,
            });
        }
        if qubit2 >= self.n_qubits {
            return Err(TnError::IndexOutOfBounds {
                index: qubit2,
                len: self.n_qubits,
            });
        }
        if qubit1.abs_diff(qubit2) != 1 {
            return Err(TnError::InvalidConfiguration(format!(
                "Only adjacent two-qubit gates are supported; got qubits {qubit1} and {qubit2}"
            )));
        }
        self.gates.push(CircuitGate::Two {
            qubit1,
            qubit2,
            matrix,
        });
        Ok(())
    }
}

// ─── Gate application primitives ─────────────────────────────────────────────

/// Apply a single-qubit 2×2 gate `U` to a site tensor `M[α, σ, β]`.
///
/// `M'[α, σ', β] = Σ_σ U[σ', σ] * M[α, σ, β]`
fn apply_single_qubit_gate(tensor: &mut MpsTensor, u: &[f64; 4]) -> TnResult<()> {
    let dl = tensor.d_l;
    let d = tensor.d_p;
    let dr = tensor.d_r;
    if d != 2 {
        return Err(TnError::InvalidConfiguration(format!(
            "Single-qubit gate requires physical dim = 2, got {d}"
        )));
    }
    let old = tensor.data.clone();
    for a in 0..dl {
        for sp in 0..d {
            for b in 0..dr {
                let mut acc = 0.0;
                for s in 0..d {
                    // U is stored row-major: U[sp, s] = u[sp*d + s]
                    acc += u[sp * d + s] * old[(a * d + s) * dr + b];
                }
                tensor.data[(a * d + sp) * dr + b] = acc;
            }
        }
    }
    Ok(())
}

/// Apply a 4×4 two-site gate via TEBD SVD at the bond between sites `s` and `s+1`.
///
/// Gate convention: `gate[p1, p2, p1', p2']` stored row-major as a 4×4 matrix
/// `gate[(p1*d + p2) * d^2 + (p1'*d + p2')]` — i.e. the gate acts on the
/// `(p1, p2)` space and maps it to `(p1', p2')`.
///
/// This matches the TEBD convention used in `tebd/tebd.rs`.
fn apply_two_qubit_gate_tebd(
    tensors: &mut [MpsTensor],
    s: usize,
    gate: &[f64; 16],
    config: &CircuitConfig,
) -> TnResult<()> {
    if s + 1 >= tensors.len() {
        return Err(TnError::IndexOutOfBounds {
            index: s,
            len: tensors.len(),
        });
    }

    let lt = tensors[s].clone();
    let rt = tensors[s + 1].clone();
    let (dl, dp1, dm) = (lt.d_l, lt.d_p, lt.d_r);
    let (dm_r, dp2, dr) = (rt.d_l, rt.d_p, rt.d_r);

    if dm != dm_r {
        return Err(TnError::DimensionMismatch { a: dm, b: dm_r });
    }
    if dp1 != 2 || dp2 != 2 {
        return Err(TnError::InvalidConfiguration(format!(
            "Two-qubit gate requires physical dim = 2, got ({dp1}, {dp2})"
        )));
    }
    let d = 2usize;

    // theta[a, p1, p2, b] = Σ_c lt[a, p1, c] * rt[c, p2, b]
    let mut theta = vec![0.0f64; dl * d * d * dr];
    for a in 0..dl {
        for p1 in 0..d {
            for p2 in 0..d {
                for b in 0..dr {
                    let mut acc = 0.0;
                    for c in 0..dm {
                        acc += lt.data[(a * d + p1) * dm + c] * rt.data[(c * d + p2) * dr + b];
                    }
                    theta[((a * d + p1) * d + p2) * dr + b] = acc;
                }
            }
        }
    }

    // Apply gate: new_theta[a, p1, p2, b] = Σ_{p1', p2'} gate[p1, p2, p1', p2'] * theta[a, p1', p2', b]
    let mut new_theta = vec![0.0f64; dl * d * d * dr];
    for a in 0..dl {
        for p1 in 0..d {
            for p2 in 0..d {
                for b in 0..dr {
                    let mut acc = 0.0;
                    for p1p in 0..d {
                        for p2p in 0..d {
                            let gv = gate[(p1 * d + p2) * d * d + p1p * d + p2p];
                            let tv = theta[((a * d + p1p) * d + p2p) * dr + b];
                            acc += gv * tv;
                        }
                    }
                    new_theta[((a * d + p1) * d + p2) * dr + b] = acc;
                }
            }
        }
    }

    // Reshape and SVD: view as (dl*d) × (d*dr)
    let m_rows = dl * d;
    let m_cols = d * dr;
    let svd = svd_jacobi(&new_theta, m_rows, m_cols)?;
    let (svd, _) = svd_truncate(svd, config.chi_max, config.svd_tol)?;
    let k = svd.k;

    // Left tensor: (dl, d, k) from U[:, :k]
    let mut left_data = vec![0.0f64; dl * d * k];
    for i in 0..m_rows {
        for j in 0..k {
            left_data[i * k + j] = svd.u[i * k + j];
        }
    }

    // Right tensor: (k, d, dr) from diag(s) * Vt[:k, :]
    let mut right_data = vec![0.0f64; k * d * dr];
    for i in 0..k {
        let sv = svd.s[i];
        for j in 0..m_cols {
            right_data[i * m_cols + j] = sv * svd.vt[i * m_cols + j];
        }
    }

    tensors[s] = MpsTensor::new(dl, d, k, left_data)?;
    tensors[s + 1] = MpsTensor::new(k, d, dr, right_data)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_1_SQRT_2, PI};

    fn product_state_zero(n: usize) -> (Vec<Vec<f64>>, Vec<[usize; 3]>) {
        let data: Vec<Vec<f64>> = (0..n).map(|_| vec![1.0, 0.0]).collect();
        let shapes: Vec<[usize; 3]> = (0..n).map(|_| [1, 2, 1]).collect();
        (data, shapes)
    }

    fn norm_sq(data: &[Vec<f64>], shapes: &[[usize; 3]]) -> f64 {
        // Accumulate environment left-to-right.
        let mut env = vec![1.0f64]; // 1×1
        let mut env_rows = 1usize;
        for (d, &[dl, dp, dr]) in data.iter().zip(shapes.iter()) {
            let mut new_env = vec![0.0f64; dr * dr];
            for b in 0..dr {
                for bp in 0..dr {
                    let mut acc = 0.0;
                    for a in 0..dl {
                        for ap in 0..dl {
                            let eaa = env[a * env_rows + ap];
                            for p in 0..dp {
                                let m1 = d[(a * dp + p) * dr + b];
                                let m2 = d[(ap * dp + p) * dr + bp];
                                acc += eaa * m1 * m2;
                            }
                        }
                    }
                    new_env[b * dr + bp] = acc;
                }
            }
            env = new_env;
            env_rows = dr;
        }
        env[0]
    }

    // ── Circuit construction ──────────────────────────────────────────────────

    #[test]
    fn new_circuit_has_zero_depth() {
        let circ = Circuit::new(4);
        assert_eq!(circ.depth(), 0);
        assert_eq!(circ.n_two_qubit_gates(), 0);
    }

    #[test]
    fn add_h_increases_depth() {
        let mut circ = Circuit::new(4);
        circ.h(0).unwrap();
        assert_eq!(circ.depth(), 1);
        assert_eq!(circ.n_two_qubit_gates(), 0);
    }

    #[test]
    fn add_gate_to_invalid_qubit_errors() {
        let mut circ = Circuit::new(3);
        assert!(circ.h(3).is_err());
        assert!(circ.x(10).is_err());
        assert!(circ.cnot(0, 3).is_err());
    }

    #[test]
    fn non_adjacent_cnot_errors() {
        let mut circ = Circuit::new(4);
        assert!(circ.cnot(0, 2).is_err(), "Non-adjacent CNOT should fail");
    }

    #[test]
    fn n_two_qubit_gates_counts_correctly() {
        let mut circ = Circuit::new(4);
        circ.h(0).unwrap();
        circ.cnot(0, 1).unwrap();
        circ.z(2).unwrap();
        circ.cnot(2, 3).unwrap();
        assert_eq!(circ.depth(), 4);
        assert_eq!(circ.n_two_qubit_gates(), 2);
    }

    // ── MPS application ───────────────────────────────────────────────────────

    #[test]
    fn empty_circuit_leaves_mps_unchanged() {
        let circ = Circuit::new(3);
        let (data, shapes) = product_state_zero(3);
        let (new_data, new_shapes) = circ
            .apply_to_mps(&data, &shapes, &CircuitConfig::default())
            .unwrap();
        assert_eq!(new_shapes, shapes);
        for (d1, d2) in new_data.iter().zip(data.iter()) {
            for (&a, &b) in d1.iter().zip(d2.iter()) {
                assert!((a - b).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn hadamard_on_product_state_creates_superposition() {
        // H|0⟩ = (|0⟩ + |1⟩)/√2, so amplitude[0] = amplitude[1] = 1/√2.
        let mut circ = Circuit::new(2);
        circ.h(0).unwrap();
        let (data, shapes) = product_state_zero(2);
        let (new_data, _new_shapes) = circ
            .apply_to_mps(&data, &shapes, &CircuitConfig::default())
            .unwrap();
        // Site 0 should be [1/√2, 1/√2] (bond dim stays 1, shape [1,2,1])
        let s0 = &new_data[0];
        assert!((s0[0] - FRAC_1_SQRT_2).abs() < 1e-12, "s0[0] = {}", s0[0]);
        assert!((s0[1] - FRAC_1_SQRT_2).abs() < 1e-12, "s0[1] = {}", s0[1]);
    }

    #[test]
    fn bell_state_has_bond_dim_2() {
        // H on qubit 0, CNOT on (0,1) → Bell state → bond dim should be 2.
        let mut circ = Circuit::new(2);
        circ.h(0).unwrap();
        circ.cnot(0, 1).unwrap();
        let (data, shapes) = product_state_zero(2);
        let (_, new_shapes) = circ
            .apply_to_mps(&data, &shapes, &CircuitConfig::default())
            .unwrap();
        // Bond dim at bond 0 = new_shapes[0][2]
        let bond_dim = new_shapes[0][2];
        assert!(
            bond_dim >= 2,
            "Bell state bond dim should be ≥ 2, got {}",
            bond_dim
        );
    }

    #[test]
    fn bell_state_norm_preserved() {
        let mut circ = Circuit::new(2);
        circ.h(0).unwrap();
        circ.cnot(0, 1).unwrap();
        let (data, shapes) = product_state_zero(2);
        let (new_data, new_shapes) = circ
            .apply_to_mps(&data, &shapes, &CircuitConfig::default())
            .unwrap();
        let n2 = norm_sq(&new_data, &new_shapes);
        assert!((n2 - 1.0).abs() < 1e-10, "Bell state norm² = {n2}");
    }

    #[test]
    fn rx_zero_leaves_mps_unchanged() {
        let mut circ = Circuit::new(2);
        circ.rx(0, 0.0).unwrap();
        let (data, shapes) = product_state_zero(2);
        let (new_data, _) = circ
            .apply_to_mps(&data, &shapes, &CircuitConfig::default())
            .unwrap();
        for (&a, &b) in new_data[0].iter().zip(data[0].iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn rx_pi_flips_zero_to_one() {
        // Rx(π)|0⟩ in the real approx gives [0, 1] up to sign (sin(π/2) = 1).
        let mut circ = Circuit::new(2);
        circ.rx(0, PI).unwrap();
        let (data, shapes) = product_state_zero(2);
        let (new_data, _) = circ
            .apply_to_mps(&data, &shapes, &CircuitConfig::default())
            .unwrap();
        // Rx(π) = [[0, -1],[1, 0]], so |0⟩ → [[0,-1],[1,0]] * [1, 0]^T = [0, 1]^T
        let s0 = &new_data[0];
        assert!(
            s0[0].abs() < 1e-12,
            "Rx(π)|0⟩ amplitude 0 should be 0, got {}",
            s0[0]
        );
        assert!(
            (s0[1] - 1.0).abs() < 1e-12,
            "Rx(π)|0⟩ amplitude 1 should be 1, got {}",
            s0[1]
        );
    }

    #[test]
    fn circuit_mismatched_n_qubits_errors() {
        let circ = Circuit::new(3);
        let (data, shapes) = product_state_zero(2); // wrong number of sites
        assert!(
            circ.apply_to_mps(&data, &shapes, &CircuitConfig::default())
                .is_err()
        );
    }
}
