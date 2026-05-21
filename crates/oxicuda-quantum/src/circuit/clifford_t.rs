use crate::circuit::circuit::{GateOp, QuantumCircuit};
use crate::error::{QuantumError, QuantumResult};

const PI: f32 = std::f32::consts::PI;
const FRAC_PI_4: f32 = std::f32::consts::FRAC_PI_4;
const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Single-qubit SU(2) unitary as a 2×2 complex matrix stored as [a, b, c, d]
/// where M = [[a, b], [c, d]] and each complex number is (re, im).
#[derive(Debug, Clone)]
pub struct Su2 {
    pub data: [(f32, f32); 4],
}

#[inline]
fn cadd(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    (a.0 + b.0, a.1 + b.1)
}

#[inline]
fn cmul(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

#[inline]
fn cconj(a: (f32, f32)) -> (f32, f32) {
    (a.0, -a.1)
}

#[inline]
fn cnorm_sq(a: (f32, f32)) -> f32 {
    a.0 * a.0 + a.1 * a.1
}

impl Su2 {
    #[must_use]
    pub fn identity() -> Self {
        Self {
            data: [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
        }
    }

    #[must_use]
    pub fn from_gate(gate: CliffordTGate) -> Self {
        gate.to_su2()
    }

    #[must_use]
    pub fn matmul(&self, other: &Su2) -> Su2 {
        let [a, b, c, d] = self.data;
        let [e, f, g, h] = other.data;
        Su2 {
            data: [
                cadd(cmul(a, e), cmul(b, g)),
                cadd(cmul(a, f), cmul(b, h)),
                cadd(cmul(c, e), cmul(d, g)),
                cadd(cmul(c, f), cmul(d, h)),
            ],
        }
    }

    #[must_use]
    pub fn dagger(&self) -> Su2 {
        let [a, b, c, d] = self.data;
        Su2 {
            data: [cconj(a), cconj(c), cconj(b), cconj(d)],
        }
    }

    #[must_use]
    pub fn trace(&self) -> (f32, f32) {
        cadd(self.data[0], self.data[3])
    }

    /// ||U - V||_F / 2 (Frobenius distance divided by 2).
    #[must_use]
    pub fn distance(&self, other: &Su2) -> f32 {
        let mut sum = 0.0_f32;
        for i in 0..4 {
            let d = (
                self.data[i].0 - other.data[i].0,
                self.data[i].1 - other.data[i].1,
            );
            sum += cnorm_sq(d);
        }
        (sum * 0.25).sqrt()
    }

    /// Rz(θ) = diag(e^{-iθ/2}, e^{iθ/2}).
    #[must_use]
    pub fn from_rz(theta: f32) -> Su2 {
        let half = theta * 0.5;
        Su2 {
            data: [
                (half.cos(), -half.sin()),
                (0.0, 0.0),
                (0.0, 0.0),
                (half.cos(), half.sin()),
            ],
        }
    }

    /// Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]].
    #[must_use]
    pub fn from_ry(theta: f32) -> Su2 {
        let half = theta * 0.5;
        Su2 {
            data: [
                (half.cos(), 0.0),
                (-half.sin(), 0.0),
                (half.sin(), 0.0),
                (half.cos(), 0.0),
            ],
        }
    }

    /// Rx(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]].
    #[must_use]
    pub fn from_rx(theta: f32) -> Su2 {
        let half = theta * 0.5;
        Su2 {
            data: [
                (half.cos(), 0.0),
                (0.0, -half.sin()),
                (0.0, -half.sin()),
                (half.cos(), 0.0),
            ],
        }
    }

    #[must_use]
    pub fn is_close(&self, other: &Su2, tol: f32) -> bool {
        self.distance(other) < tol
    }
}

/// Clifford+T gate set elements (discrete single-qubit gates).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliffordTGate {
    I,
    H,
    S,
    Sdg,
    T,
    Tdg,
    X,
    Y,
    Z,
}

impl CliffordTGate {
    #[must_use]
    pub fn to_su2(&self) -> Su2 {
        match self {
            CliffordTGate::I => Su2::identity(),
            CliffordTGate::H => Su2 {
                data: [
                    (INV_SQRT2, 0.0),
                    (INV_SQRT2, 0.0),
                    (INV_SQRT2, 0.0),
                    (-INV_SQRT2, 0.0),
                ],
            },
            CliffordTGate::S => Su2 {
                data: [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 1.0)],
            },
            CliffordTGate::Sdg => Su2 {
                data: [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, -1.0)],
            },
            CliffordTGate::T => Su2 {
                data: [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (INV_SQRT2, INV_SQRT2)],
            },
            CliffordTGate::Tdg => Su2 {
                data: [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (INV_SQRT2, -INV_SQRT2)],
            },
            CliffordTGate::X => Su2 {
                data: [(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)],
            },
            CliffordTGate::Y => Su2 {
                data: [(0.0, 0.0), (0.0, -1.0), (0.0, 1.0), (0.0, 0.0)],
            },
            CliffordTGate::Z => Su2 {
                data: [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-1.0, 0.0)],
            },
        }
    }

    /// Map this gate to the corresponding GateOp for circuit construction.
    #[must_use]
    pub fn to_gate_op(&self, _qubit: usize) -> GateOp {
        match self {
            CliffordTGate::I => GateOp::Z,
            CliffordTGate::H => GateOp::H,
            CliffordTGate::S => GateOp::S,
            CliffordTGate::Sdg => GateOp::Rz(-PI * 0.5),
            CliffordTGate::T => GateOp::T,
            CliffordTGate::Tdg => GateOp::Rz(-FRAC_PI_4),
            CliffordTGate::X => GateOp::X,
            CliffordTGate::Y => GateOp::Y,
            CliffordTGate::Z => GateOp::Z,
        }
    }
}

/// Clifford+T circuit decomposer.
///
/// Synthesizes discrete Clifford+T gate sequences that approximate arbitrary
/// single-qubit rotations to within a given tolerance.
pub struct CliffordTDecomposer;

const ALPHABET: [CliffordTGate; 5] = [
    CliffordTGate::H,
    CliffordTGate::T,
    CliffordTGate::Tdg,
    CliffordTGate::S,
    CliffordTGate::Sdg,
];

impl CliffordTDecomposer {
    /// Distance metric d(U, V) = 1 - |tr(U† V)| / 2 ∈ [0, 1].
    #[must_use]
    pub fn unitary_distance(u: &Su2, v: &Su2) -> f32 {
        let ud = u.dagger();
        let prod = ud.matmul(v);
        let tr = prod.trace();
        let abs_tr = (tr.0 * tr.0 + tr.1 * tr.1).sqrt();
        (1.0 - abs_tr * 0.5).max(0.0)
    }

    /// Fold a sequence of gates left-to-right into a single Su2 product.
    #[must_use]
    pub fn sequence_to_su2(seq: &[CliffordTGate]) -> Su2 {
        seq.iter()
            .fold(Su2::identity(), |acc, g| acc.matmul(&g.to_su2()))
    }

    fn check_exact_rz(theta_norm: f32, tol: f32) -> Option<Vec<CliffordTGate>> {
        let candidates: &[(f32, &[CliffordTGate])] = &[
            (0.0, &[CliffordTGate::I]),
            (FRAC_PI_4, &[CliffordTGate::T]),
            (PI * 0.5, &[CliffordTGate::S]),
            (PI, &[CliffordTGate::Z]),
            (-FRAC_PI_4, &[CliffordTGate::Tdg]),
            (-PI * 0.5, &[CliffordTGate::Sdg]),
            (FRAC_PI_4 * 3.0, &[CliffordTGate::T, CliffordTGate::S]),
            (-FRAC_PI_4 * 3.0, &[CliffordTGate::Tdg, CliffordTGate::Sdg]),
        ];
        for (angle, seq) in candidates {
            let diff = (theta_norm - angle).abs();
            let wrapped = if diff > PI {
                (2.0 * PI - diff).abs()
            } else {
                diff
            };
            if wrapped < tol {
                return Some(seq.to_vec());
            }
        }
        None
    }

    fn normalize_angle(theta: f32) -> f32 {
        let mut t = theta % (2.0 * PI);
        if t > PI {
            t -= 2.0 * PI;
        } else if t <= -PI {
            t += 2.0 * PI;
        }
        t
    }

    /// Decompose Rz(theta) into a Clifford+T sequence within tolerance.
    pub fn decompose_rz(
        theta: f32,
        max_depth: usize,
        tol: f32,
    ) -> QuantumResult<Vec<CliffordTGate>> {
        if max_depth == 0 {
            return Err(QuantumError::InvalidParameter {
                name: "max_depth must be > 0".to_string(),
            });
        }

        let t_norm = Self::normalize_angle(theta);
        if let Some(exact) = Self::check_exact_rz(t_norm, tol) {
            return Ok(exact);
        }

        let target = Su2::from_rz(theta);
        let mut best_seq: Vec<CliffordTGate> = vec![CliffordTGate::T];
        let mut best_dist = Self::unitary_distance(&target, &CliffordTGate::T.to_su2());

        for depth in 1..=max_depth {
            let found = Self::search_sequences(&target, depth, tol);
            if !found.is_empty() {
                return Ok(found.into_iter().next().unwrap_or_default());
            }
            let candidates = Self::search_sequences_best(&target, depth);
            if let Some((dist, seq)) = candidates
                && dist < best_dist
            {
                best_dist = dist;
                best_seq = seq;
            }
        }

        Ok(best_seq)
    }

    /// Decompose Ry(theta) = H Rz(theta) H.
    pub fn decompose_ry(
        theta: f32,
        max_depth: usize,
        tol: f32,
    ) -> QuantumResult<Vec<CliffordTGate>> {
        if max_depth == 0 {
            return Err(QuantumError::InvalidParameter {
                name: "max_depth must be > 0".to_string(),
            });
        }

        let target = Su2::from_ry(theta);
        for depth in 1..=max_depth {
            let found = Self::search_sequences(&target, depth, tol);
            if !found.is_empty() {
                return Ok(found.into_iter().next().unwrap_or_default());
            }
        }

        let h = CliffordTGate::H;
        let inner = Self::decompose_rz(theta, max_depth.saturating_sub(2).max(1), tol)?;
        let mut seq = vec![h];
        seq.extend(inner);
        seq.push(h);
        Ok(seq)
    }

    /// Decompose Rx(theta) = H Rz(theta) H.
    pub fn decompose_rx(
        theta: f32,
        max_depth: usize,
        tol: f32,
    ) -> QuantumResult<Vec<CliffordTGate>> {
        if max_depth == 0 {
            return Err(QuantumError::InvalidParameter {
                name: "max_depth must be > 0".to_string(),
            });
        }

        let target = Su2::from_rx(theta);
        for depth in 1..=max_depth {
            let found = Self::search_sequences(&target, depth, tol);
            if !found.is_empty() {
                return Ok(found.into_iter().next().unwrap_or_default());
            }
        }

        let h = CliffordTGate::H;
        let inner = Self::decompose_rz(theta, max_depth.saturating_sub(2).max(1), tol)?;
        let mut seq = vec![h];
        seq.extend(inner);
        seq.push(h);
        Ok(seq)
    }

    /// Transpile a circuit, replacing all parametric gates with Clifford+T sequences.
    pub fn transpile(
        circuit: &QuantumCircuit,
        max_depth: usize,
        tol: f32,
    ) -> QuantumResult<QuantumCircuit> {
        let mut out = QuantumCircuit::new(circuit.n_qubits);

        for (qubit, op) in &circuit.ops {
            let q = *qubit;
            match op {
                GateOp::Rx(theta) => {
                    let seq = Self::decompose_rx(*theta, max_depth, tol)?;
                    for gate in seq {
                        if gate == CliffordTGate::I {
                            continue;
                        }
                        out.ops.push((q, gate.to_gate_op(q)));
                    }
                }
                GateOp::Ry(theta) => {
                    let seq = Self::decompose_ry(*theta, max_depth, tol)?;
                    for gate in seq {
                        if gate == CliffordTGate::I {
                            continue;
                        }
                        out.ops.push((q, gate.to_gate_op(q)));
                    }
                }
                GateOp::Rz(theta) => {
                    let seq = Self::decompose_rz(*theta, max_depth, tol)?;
                    for gate in seq {
                        if gate == CliffordTGate::I {
                            continue;
                        }
                        out.ops.push((q, gate.to_gate_op(q)));
                    }
                }
                GateOp::H => out.ops.push((q, GateOp::H)),
                GateOp::X => out.ops.push((q, GateOp::X)),
                GateOp::Y => out.ops.push((q, GateOp::Y)),
                GateOp::Z => out.ops.push((q, GateOp::Z)),
                GateOp::S => out.ops.push((q, GateOp::S)),
                GateOp::T => out.ops.push((q, GateOp::T)),
                GateOp::Cnot { ctrl, tgt } => out.ops.push((
                    q,
                    GateOp::Cnot {
                        ctrl: *ctrl,
                        tgt: *tgt,
                    },
                )),
                GateOp::Cz { ctrl, tgt } => out.ops.push((
                    q,
                    GateOp::Cz {
                        ctrl: *ctrl,
                        tgt: *tgt,
                    },
                )),
                GateOp::Swap { q0, q1 } => out.ops.push((q, GateOp::Swap { q0: *q0, q1: *q1 })),
                GateOp::Measure { qubit: mq } => out.ops.push((q, GateOp::Measure { qubit: *mq })),
            }
        }

        Ok(out)
    }

    /// Generate all sequences of exactly `depth` gates within `tol` of `target`.
    pub fn search_sequences(target: &Su2, depth: usize, tol: f32) -> Vec<Vec<CliffordTGate>> {
        let mut results = Vec::new();
        let mut current = Vec::with_capacity(depth);
        Self::dfs_search(
            target,
            depth,
            tol,
            Su2::identity(),
            &mut current,
            &mut results,
        );
        results
    }

    fn dfs_search(
        target: &Su2,
        remaining: usize,
        tol: f32,
        current_su2: Su2,
        current_seq: &mut Vec<CliffordTGate>,
        results: &mut Vec<Vec<CliffordTGate>>,
    ) {
        if remaining == 0 {
            let dist = Self::unitary_distance(&current_su2, target);
            if dist < tol {
                results.push(current_seq.clone());
            }
            return;
        }

        for &gate in &ALPHABET {
            let next_su2 = current_su2.matmul(&gate.to_su2());
            current_seq.push(gate);
            Self::dfs_search(target, remaining - 1, tol, next_su2, current_seq, results);
            current_seq.pop();
        }
    }

    fn search_sequences_best(target: &Su2, depth: usize) -> Option<(f32, Vec<CliffordTGate>)> {
        let mut best: Option<(f32, Vec<CliffordTGate>)> = None;
        let mut current = Vec::with_capacity(depth);
        Self::dfs_best(target, depth, Su2::identity(), &mut current, &mut best);
        best
    }

    fn dfs_best(
        target: &Su2,
        remaining: usize,
        current_su2: Su2,
        current_seq: &mut Vec<CliffordTGate>,
        best: &mut Option<(f32, Vec<CliffordTGate>)>,
    ) {
        if remaining == 0 {
            let dist = Self::unitary_distance(&current_su2, target);
            let update = match best {
                None => true,
                Some((d, _)) => dist < *d,
            };
            if update {
                *best = Some((dist, current_seq.clone()));
            }
            return;
        }

        for &gate in &ALPHABET {
            let next_su2 = current_su2.matmul(&gate.to_su2());
            current_seq.push(gate);
            Self::dfs_best(target, remaining - 1, next_su2, current_seq, best);
            current_seq.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    fn su2_approx_eq(a: &Su2, b: &Su2) -> bool {
        a.distance(b) < TOL
    }

    #[test]
    fn su2_identity_is_identity() {
        let id = Su2::identity();
        let h = CliffordTGate::H.to_su2();
        assert!(su2_approx_eq(&id.matmul(&h), &h));
        assert!(su2_approx_eq(&h.matmul(&id), &h));
    }

    #[test]
    fn su2_matmul_associative() {
        let a = CliffordTGate::H.to_su2();
        let b = CliffordTGate::T.to_su2();
        let c = CliffordTGate::S.to_su2();
        let lhs = a.matmul(&b).matmul(&c);
        let rhs = a.matmul(&b.matmul(&c));
        assert!(su2_approx_eq(&lhs, &rhs));
    }

    #[test]
    fn su2_dagger_inverts() {
        let h = CliffordTGate::H.to_su2();
        let hd = h.dagger();
        let prod = h.matmul(&hd);
        assert!(su2_approx_eq(&prod, &Su2::identity()));
    }

    #[test]
    fn cliffordt_gate_to_su2_h_is_hadamard() {
        let h = CliffordTGate::H.to_su2();
        let expected_re = INV_SQRT2;
        assert!((h.data[0].0 - expected_re).abs() < TOL);
        assert!((h.data[0].1).abs() < TOL);
        assert!((h.data[1].0 - expected_re).abs() < TOL);
        assert!((h.data[2].0 - expected_re).abs() < TOL);
        assert!((h.data[3].0 + expected_re).abs() < TOL);
    }

    #[test]
    fn cliffordt_gate_to_su2_t_is_correct() {
        let t = CliffordTGate::T.to_su2();
        assert!((t.data[0].0 - 1.0).abs() < TOL);
        assert!((t.data[0].1).abs() < TOL);
        assert!((t.data[3].0 - INV_SQRT2).abs() < TOL);
        assert!((t.data[3].1 - INV_SQRT2).abs() < TOL);
    }

    #[test]
    fn sequence_to_su2_empty_is_identity() {
        let result = CliffordTDecomposer::sequence_to_su2(&[]);
        assert!(su2_approx_eq(&result, &Su2::identity()));
    }

    #[test]
    fn sequence_to_su2_hh_is_identity() {
        let seq = [CliffordTGate::H, CliffordTGate::H];
        let result = CliffordTDecomposer::sequence_to_su2(&seq);
        assert!(su2_approx_eq(&result, &Su2::identity()));
    }

    #[test]
    fn sequence_to_su2_ss_is_z() {
        let seq = [CliffordTGate::S, CliffordTGate::S];
        let result = CliffordTDecomposer::sequence_to_su2(&seq);
        let z = CliffordTGate::Z.to_su2();
        assert!(su2_approx_eq(&result, &z));
    }

    #[test]
    fn sequence_to_su2_tt_is_s() {
        let seq = [CliffordTGate::T, CliffordTGate::T];
        let result = CliffordTDecomposer::sequence_to_su2(&seq);
        let s = CliffordTGate::S.to_su2();
        assert!(su2_approx_eq(&result, &s));
    }

    #[test]
    fn decompose_rz_exact_pi4() {
        let seq = CliffordTDecomposer::decompose_rz(FRAC_PI_4, 3, 1e-3).unwrap();
        let result = CliffordTDecomposer::sequence_to_su2(&seq);
        let target = Su2::from_rz(FRAC_PI_4);
        let dist = CliffordTDecomposer::unitary_distance(&result, &target);
        assert!(dist < 1e-3, "dist={dist}");
    }

    #[test]
    fn decompose_rz_zero_is_identity() {
        let seq = CliffordTDecomposer::decompose_rz(0.0, 4, 1e-3).unwrap();
        let result = CliffordTDecomposer::sequence_to_su2(&seq);
        let dist = CliffordTDecomposer::unitary_distance(&result, &Su2::identity());
        assert!(dist < 1e-3, "dist={dist}");
    }

    #[test]
    fn decompose_rz_result_close_to_target() {
        let theta = 0.7f32;
        let seq = CliffordTDecomposer::decompose_rz(theta, 6, 0.3).unwrap();
        let result = CliffordTDecomposer::sequence_to_su2(&seq);
        let target = Su2::from_rz(theta);
        let dist = CliffordTDecomposer::unitary_distance(&result, &target);
        assert!(dist < 0.35, "dist={dist}");
    }

    #[test]
    fn decompose_ry_via_rz() {
        let theta = 1.1f32;
        let seq = CliffordTDecomposer::decompose_ry(theta, 6, 0.4).unwrap();
        let result = CliffordTDecomposer::sequence_to_su2(&seq);
        let target = Su2::from_ry(theta);
        let dist = CliffordTDecomposer::unitary_distance(&result, &target);
        assert!(dist < 0.45, "dist={dist}");
    }

    #[test]
    fn transpile_circuit_no_parametric_gates() {
        let mut circ = QuantumCircuit::new(2);
        circ.add_gate(GateOp::H);
        circ.add_gate(GateOp::T);
        circ.add_gate(GateOp::Cnot { ctrl: 0, tgt: 1 });
        let transpiled = CliffordTDecomposer::transpile(&circ, 4, 1e-3).unwrap();
        let has_parametric = transpiled
            .ops
            .iter()
            .any(|(_, op)| matches!(op, GateOp::Rx(_) | GateOp::Ry(_)));
        assert!(
            !has_parametric,
            "should have no Rx/Ry after transpilation of non-parametric circuit"
        );
    }

    #[test]
    fn transpile_circuit_with_rz() {
        let mut circ = QuantumCircuit::new(1);
        circ.add_gate(GateOp::Rz(1.0));
        let transpiled = CliffordTDecomposer::transpile(&circ, 5, 0.4).unwrap();
        let has_rz = transpiled
            .ops
            .iter()
            .any(|(_, op)| matches!(op, GateOp::Rx(_) | GateOp::Ry(_)));
        assert!(!has_rz, "should have no Rx/Ry in transpiled output");
    }

    #[test]
    fn unitary_distance_self_is_zero() {
        let t = CliffordTGate::T.to_su2();
        let dist = CliffordTDecomposer::unitary_distance(&t, &t);
        assert!(dist < TOL, "dist={dist}");
    }

    #[test]
    fn unitary_distance_h_and_identity() {
        let h = CliffordTGate::H.to_su2();
        let id = Su2::identity();
        let dist = CliffordTDecomposer::unitary_distance(&h, &id);
        assert!(dist > 0.01, "dist={dist}");
    }

    #[test]
    fn from_rx_from_rz_consistent() {
        let theta = 0.8f32;
        let rx = Su2::from_rx(theta);
        let h = CliffordTGate::H.to_su2();
        let rz = Su2::from_rz(theta);
        let h_rz_h = h.matmul(&rz).matmul(&h);
        let dist = rx.distance(&h_rz_h);
        assert!(dist < 1e-4, "dist={dist}");
    }

    #[test]
    fn sdg_is_inverse_of_s() {
        let s = CliffordTGate::S.to_su2();
        let sdg = CliffordTGate::Sdg.to_su2();
        let prod = s.matmul(&sdg);
        assert!(su2_approx_eq(&prod, &Su2::identity()));
    }

    #[test]
    fn tdg_is_inverse_of_t() {
        let t = CliffordTGate::T.to_su2();
        let tdg = CliffordTGate::Tdg.to_su2();
        let prod = t.matmul(&tdg);
        assert!(su2_approx_eq(&prod, &Su2::identity()));
    }
}
