//! Stabilizer-formalism (CHP) tableau simulator.
//!
//! Implements the Aaronson–Gottesman 2004 "Improved Simulation of Stabilizer
//! Circuits" tableau algorithm. Clifford circuits (built from `H`, `S`, `CNOT`
//! and Pauli gates) acting on `n` qubits are simulated in time polynomial in
//! `n` by tracking a binary tableau of stabilizer/destabilizer generators
//! rather than the exponentially large state vector.
//!
//! Tableau layout (Aaronson–Gottesman): there are `2n + 1` rows of `n` columns.
//! Rows `0..n` are the *destabilizer* generators, rows `n..2n` are the
//! *stabilizer* generators and row `2n` is a scratch row used by deterministic
//! measurement. Each row `i` stores `x[i][0..n]`, `z[i][0..n]` (the symplectic
//! representation of a Pauli operator) and a phase bit `r[i]` (0 → `+`,
//! 1 → `−`). This module is fully self-contained and does NOT use the
//! state-vector simulator.

use crate::error::{QuantumError, QuantumResult};
use crate::handle::LcgRng;

/// CHP stabilizer tableau for poly-time Clifford-circuit simulation.
///
/// The `x` and `z` fields are flat row-major buffers of length `(2n + 1) * n`;
/// element `(row, col)` lives at `row * n + col`. The phase buffer `r` has
/// length `2n + 1`.
#[derive(Debug, Clone)]
pub struct StabilizerTableau {
    n_qubits: usize,
    /// X bits, row-major `(2n+1) × n`.
    x: Vec<bool>,
    /// Z bits, row-major `(2n+1) × n`.
    z: Vec<bool>,
    /// Phase bits, length `2n + 1`.
    r: Vec<bool>,
}

impl StabilizerTableau {
    /// Build the tableau for the computational-basis state |0…0⟩.
    ///
    /// Destabilizers are initialized to `X_i` (row `i`, `i < n`) and stabilizers
    /// to `Z_i` (row `n + i`); all phases are `+`.
    pub fn new(n_qubits: usize) -> QuantumResult<Self> {
        if n_qubits == 0 {
            return Err(QuantumError::InvalidQubitCount { n: 0 });
        }
        let n_rows = 2 * n_qubits + 1;
        let mut x = vec![false; n_rows * n_qubits];
        let mut z = vec![false; n_rows * n_qubits];
        let r = vec![false; n_rows];

        for i in 0..n_qubits {
            // Destabilizer i = X_i.
            x[i * n_qubits + i] = true;
            // Stabilizer i (row n + i) = Z_i.
            z[(n_qubits + i) * n_qubits + i] = true;
        }

        Ok(Self { n_qubits, x, z, r })
    }

    /// Number of qubits the tableau tracks.
    #[must_use]
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Number of rows in the tableau (`2n + 1`).
    #[inline]
    fn n_rows(&self) -> usize {
        2 * self.n_qubits + 1
    }

    /// Flat index for tableau element `(row, col)`.
    #[inline]
    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.n_qubits + col
    }

    /// Validate that `a` is a legal qubit index.
    #[inline]
    fn check_qubit(&self, a: usize) -> QuantumResult<()> {
        if a >= self.n_qubits {
            return Err(QuantumError::QubitIndexOutOfRange {
                index: a,
                n_qubits: self.n_qubits,
            });
        }
        Ok(())
    }

    /// Apply a Hadamard on qubit `a`.
    ///
    /// For each row `i`: `r[i] ^= x[i][a] & z[i][a]` then swap `x[i][a]`,
    /// `z[i][a]`.
    pub fn h(&mut self, a: usize) -> QuantumResult<()> {
        self.check_qubit(a)?;
        let n_rows = self.n_rows();
        for i in 0..n_rows {
            let xi = self.idx(i, a);
            let xa = self.x[xi];
            let za = self.z[xi];
            self.r[i] ^= xa & za;
            self.x[xi] = za;
            self.z[xi] = xa;
        }
        Ok(())
    }

    /// Apply a phase gate `S` on qubit `a`.
    ///
    /// For each row `i`: `r[i] ^= x[i][a] & z[i][a]` then `z[i][a] ^= x[i][a]`.
    pub fn s(&mut self, a: usize) -> QuantumResult<()> {
        self.check_qubit(a)?;
        let n_rows = self.n_rows();
        for i in 0..n_rows {
            let xi = self.idx(i, a);
            let xa = self.x[xi];
            self.r[i] ^= xa & self.z[xi];
            self.z[xi] ^= xa;
        }
        Ok(())
    }

    /// Apply a controlled-NOT with the given `control` and `target` qubits.
    ///
    /// For each row `i`:
    /// `r[i] ^= x[i][c] & z[i][t] & (x[i][t] ^ z[i][c] ^ true)`,
    /// `x[i][t] ^= x[i][c]`, `z[i][c] ^= z[i][t]`.
    pub fn cnot(&mut self, control: usize, target: usize) -> QuantumResult<()> {
        self.check_qubit(control)?;
        self.check_qubit(target)?;
        if control == target {
            return Err(QuantumError::InvalidParameter {
                name: "control and target must differ".into(),
            });
        }
        let n_rows = self.n_rows();
        for i in 0..n_rows {
            let ci = self.idx(i, control);
            let ti = self.idx(i, target);
            let xc = self.x[ci];
            let zc = self.z[ci];
            let xt = self.x[ti];
            let zt = self.z[ti];
            self.r[i] ^= xc & zt & (xt ^ zc ^ true);
            self.x[ti] = xt ^ xc;
            self.z[ci] = zc ^ zt;
        }
        Ok(())
    }

    /// Apply a Pauli-X on qubit `a`, realized as `X = H S S H`.
    pub fn x(&mut self, a: usize) -> QuantumResult<()> {
        self.check_qubit(a)?;
        self.h(a)?;
        self.s(a)?;
        self.s(a)?;
        self.h(a)
    }

    /// Apply a Pauli-Z on qubit `a`, realized as `Z = S S`.
    pub fn z(&mut self, a: usize) -> QuantumResult<()> {
        self.check_qubit(a)?;
        self.s(a)?;
        self.s(a)
    }

    /// Apply a Pauli-Y on qubit `a`.
    ///
    /// `Y = i·X·Z`; applying `Z` then `X` realizes `Y` up to a global phase,
    /// which is unobservable in the stabilizer formalism.
    pub fn y(&mut self, a: usize) -> QuantumResult<()> {
        self.check_qubit(a)?;
        self.z(a)?;
        self.x(a)
    }

    /// Phase contribution `g` (power of `i`, in `{-1, 0, 1}`) accrued when
    /// left-multiplying the Pauli `(x1, z1)` by `(x2, z2)` — the function from
    /// the Aaronson–Gottesman paper.
    #[inline]
    fn g(x1: bool, z1: bool, x2: bool, z2: bool) -> i32 {
        let x2i = i32::from(x2);
        let z2i = i32::from(z2);
        match (x1, z1) {
            (false, false) => 0,
            (true, true) => z2i - x2i,
            (true, false) => z2i * (2 * x2i - 1),
            (false, true) => x2i * (1 - 2 * z2i),
        }
    }

    /// `rowsum(h, i)`: left-multiply row `h` by row `i`, accumulating the phase
    /// exactly as in the paper. Sets row `h` to the product `R_i · R_h`.
    fn rowsum(&mut self, h: usize, i: usize) {
        let mut acc: i32 = 2 * i32::from(self.r[h]) + 2 * i32::from(self.r[i]);
        for j in 0..self.n_qubits {
            let hj = self.idx(h, j);
            let ij = self.idx(i, j);
            acc += Self::g(self.x[ij], self.z[ij], self.x[hj], self.z[hj]);
        }
        // acc mod 4 ∈ {0, 2}: 0 → phase +, 2 → phase −.
        let m = acc.rem_euclid(4);
        self.r[h] = m == 2;
        for j in 0..self.n_qubits {
            let hj = self.idx(h, j);
            let ij = self.idx(i, j);
            self.x[hj] ^= self.x[ij];
            self.z[hj] ^= self.z[ij];
        }
    }

    /// Whether a Z-basis measurement of qubit `a` is deterministic.
    ///
    /// The outcome is random iff some stabilizer generator anticommutes with
    /// `Z_a`, i.e. some row `p` in `n..2n` has `x[p][a] == true`.
    pub fn is_deterministic(&self, a: usize) -> QuantumResult<bool> {
        self.check_qubit(a)?;
        for p in self.n_qubits..(2 * self.n_qubits) {
            if self.x[self.idx(p, a)] {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Copy row `src` into row `dst` (x, z and phase bits).
    fn copy_row(&mut self, dst: usize, src: usize) {
        for j in 0..self.n_qubits {
            let d = self.idx(dst, j);
            let s = self.idx(src, j);
            self.x[d] = self.x[s];
            self.z[d] = self.z[s];
        }
        self.r[dst] = self.r[src];
    }

    /// Zero every x/z/phase bit of row `row`.
    fn zero_row(&mut self, row: usize) {
        for j in 0..self.n_qubits {
            let idx = self.idx(row, j);
            self.x[idx] = false;
            self.z[idx] = false;
        }
        self.r[row] = false;
    }

    /// Perform a Z-basis measurement of qubit `a`, collapsing the tableau and
    /// returning the outcome (`true` = |1⟩).
    pub fn measure(&mut self, a: usize, rng: &mut LcgRng) -> QuantumResult<bool> {
        self.check_qubit(a)?;
        let n = self.n_qubits;

        // Find a stabilizer row that anticommutes with Z_a (x[p][a] == true).
        let mut pivot: Option<usize> = None;
        for p in n..(2 * n) {
            if self.x[self.idx(p, a)] {
                pivot = Some(p);
                break;
            }
        }

        match pivot {
            Some(p) => {
                // Random outcome.
                for i in 0..(2 * n) {
                    if i != p && self.x[self.idx(i, a)] {
                        self.rowsum(i, p);
                    }
                }
                // Destabilizer p-n becomes the old stabilizer p.
                self.copy_row(p - n, p);
                // Row p becomes Z_a with a uniformly random phase.
                self.zero_row(p);
                let bit = (rng.next_u32() & 1) == 1;
                self.r[p] = bit;
                let pa = self.idx(p, a);
                self.z[pa] = true;
                Ok(bit)
            }
            None => {
                // Deterministic outcome: assemble the sign via the scratch row.
                let scratch = 2 * n;
                self.zero_row(scratch);
                for i in 0..n {
                    if self.x[self.idx(i, a)] {
                        self.rowsum(scratch, i + n);
                    }
                }
                Ok(self.r[scratch])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_zero_state_measures_all_zero_deterministically() {
        let mut tab = StabilizerTableau::new(4).expect("valid 4-qubit tableau");
        let mut rng = LcgRng::new(1);
        for q in 0..4 {
            assert!(
                tab.is_deterministic(q).expect("qubit index in range"),
                "qubit {q} not deterministic"
            );
            assert!(
                !tab.measure(q, &mut rng).expect("measure qubit in range"),
                "qubit {q} measured 1"
            );
        }
    }

    #[test]
    fn hadamard_makes_qubit_non_deterministic() {
        let mut tab = StabilizerTableau::new(2).expect("valid 2-qubit tableau");
        tab.h(0).expect("hadamard on qubit 0");
        assert!(!tab.is_deterministic(0).expect("qubit 0 determinism check"));
        // Untouched qubit stays deterministic.
        assert!(tab.is_deterministic(1).expect("qubit 1 determinism check"));
    }

    #[test]
    fn hadamard_measure_reproducible_fixed_seed() {
        let outcome = |seed: u64| {
            let mut tab = StabilizerTableau::new(1).expect("valid 1-qubit tableau");
            tab.h(0).expect("hadamard on qubit 0");
            let mut rng = LcgRng::new(seed);
            tab.measure(0, &mut rng).expect("measure qubit 0")
        };
        assert_eq!(outcome(123), outcome(123));
        assert_eq!(outcome(999), outcome(999));
    }

    #[test]
    fn bell_state_measurements_correlated() {
        for seed in [1u64, 2, 3, 7, 42, 100, 2024] {
            let mut tab = StabilizerTableau::new(2).expect("valid 2-qubit tableau");
            tab.h(0).expect("hadamard on qubit 0");
            tab.cnot(0, 1).expect("CNOT with control 0 and target 1");
            let mut rng = LcgRng::new(seed);
            let m0 = tab.measure(0, &mut rng).expect("measure qubit 0");
            let m1 = tab.measure(1, &mut rng).expect("measure qubit 1");
            assert_eq!(m0, m1, "Bell pair not correlated for seed {seed}");
        }
    }

    #[test]
    fn ghz_state_all_measurements_equal() {
        for seed in [0u64, 5, 11, 77, 256, 9999] {
            let mut tab = StabilizerTableau::new(3).expect("valid 3-qubit tableau");
            tab.h(0).expect("hadamard on qubit 0");
            tab.cnot(0, 1).expect("CNOT with control 0 and target 1");
            tab.cnot(1, 2).expect("CNOT with control 1 and target 2");
            let mut rng = LcgRng::new(seed);
            let m0 = tab.measure(0, &mut rng).expect("measure qubit 0");
            let m1 = tab.measure(1, &mut rng).expect("measure qubit 1");
            let m2 = tab.measure(2, &mut rng).expect("measure qubit 2");
            assert_eq!(m0, m1, "GHZ q0!=q1 seed {seed}");
            assert_eq!(m1, m2, "GHZ q1!=q2 seed {seed}");
        }
    }

    #[test]
    fn x_gate_flips_zero_to_one() {
        let mut tab = StabilizerTableau::new(1).expect("valid 1-qubit tableau");
        tab.x(0).expect("X gate on qubit 0");
        let mut rng = LcgRng::new(1);
        assert!(tab.is_deterministic(0).expect("qubit 0 determinism check"));
        assert!(tab.measure(0, &mut rng).expect("measure qubit 0"));
    }

    #[test]
    fn double_hadamard_is_identity() {
        let mut tab = StabilizerTableau::new(1).expect("valid 1-qubit tableau");
        tab.h(0).expect("first hadamard on qubit 0");
        tab.h(0).expect("second hadamard on qubit 0");
        let mut rng = LcgRng::new(5);
        assert!(tab.is_deterministic(0).expect("qubit 0 determinism check"));
        assert!(!tab.measure(0, &mut rng).expect("measure qubit 0"));
    }

    #[test]
    fn z_from_s_squared_leaves_zero_deterministic() {
        let mut tab = StabilizerTableau::new(1).expect("valid 1-qubit tableau");
        tab.z(0).expect("Z gate on qubit 0");
        let mut rng = LcgRng::new(8);
        assert!(tab.is_deterministic(0).expect("qubit 0 determinism check"));
        assert!(!tab.measure(0, &mut rng).expect("measure qubit 0"));
    }

    #[test]
    fn measuring_same_qubit_twice_gives_same_result() {
        for seed in [1u64, 13, 88, 314] {
            let mut tab = StabilizerTableau::new(1).expect("valid 1-qubit tableau");
            tab.h(0).expect("hadamard on qubit 0");
            let mut rng = LcgRng::new(seed);
            let first = tab
                .measure(0, &mut rng)
                .expect("first measurement of qubit 0");
            // After collapse the qubit is deterministic and yields the same bit.
            assert!(
                tab.is_deterministic(0)
                    .expect("qubit 0 determinism check after collapse")
            );
            let second = tab
                .measure(0, &mut rng)
                .expect("second measurement of qubit 0");
            assert_eq!(first, second, "collapse not respected for seed {seed}");
        }
    }

    #[test]
    fn n_qubits_getter() {
        let tab = StabilizerTableau::new(5).expect("valid 5-qubit tableau");
        assert_eq!(tab.n_qubits(), 5);
    }

    #[test]
    fn out_of_range_qubit_errors() {
        let mut tab = StabilizerTableau::new(2).expect("valid 2-qubit tableau");
        let mut rng = LcgRng::new(1);
        assert!(tab.h(2).is_err());
        assert!(tab.s(5).is_err());
        assert!(tab.cnot(0, 9).is_err());
        assert!(tab.cnot(7, 1).is_err());
        assert!(tab.measure(3, &mut rng).is_err());
        assert!(tab.is_deterministic(4).is_err());
        assert!(tab.x(2).is_err());
        assert!(tab.y(2).is_err());
        assert!(tab.z(2).is_err());
    }

    #[test]
    fn zero_qubit_count_errors() {
        assert!(StabilizerTableau::new(0).is_err());
    }

    #[test]
    fn cnot_control_equals_target_errors() {
        let mut tab = StabilizerTableau::new(3).expect("valid 3-qubit tableau");
        assert!(tab.cnot(1, 1).is_err());
    }

    #[test]
    fn deterministic_given_seed_identical_sequences() {
        let run = || {
            let mut tab = StabilizerTableau::new(3).expect("valid 3-qubit tableau");
            let mut rng = LcgRng::new(2024);
            tab.h(0).expect("hadamard on qubit 0");
            tab.cnot(0, 1).expect("CNOT with control 0 and target 1");
            tab.cnot(1, 2).expect("CNOT with control 1 and target 2");
            tab.h(2).expect("hadamard on qubit 2");
            let a = tab.measure(0, &mut rng).expect("measure qubit 0");
            let b = tab.measure(1, &mut rng).expect("measure qubit 1");
            let c = tab.measure(2, &mut rng).expect("measure qubit 2");
            (a, b, c)
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn x_via_h_s_s_h_flips_zero() {
        let mut tab = StabilizerTableau::new(1).expect("valid 1-qubit tableau");
        // X = H S S H (exactly the decomposition used internally).
        tab.h(0).expect("hadamard on qubit 0");
        tab.s(0).expect("first S gate on qubit 0");
        tab.s(0).expect("second S gate on qubit 0");
        tab.h(0).expect("second hadamard on qubit 0");
        let mut rng = LcgRng::new(3);
        assert!(tab.measure(0, &mut rng).expect("measure qubit 0"));
    }

    #[test]
    fn y_gate_flips_zero_to_one() {
        // Y|0⟩ = i|1⟩, so a Z-basis measurement is deterministically 1.
        let mut tab = StabilizerTableau::new(1).expect("valid 1-qubit tableau");
        tab.y(0).expect("Y gate on qubit 0");
        let mut rng = LcgRng::new(4);
        assert!(tab.is_deterministic(0).expect("qubit 0 determinism check"));
        assert!(tab.measure(0, &mut rng).expect("measure qubit 0"));
    }

    #[test]
    fn double_x_returns_to_zero() {
        let mut tab = StabilizerTableau::new(1).expect("valid 1-qubit tableau");
        tab.x(0).expect("first X gate on qubit 0");
        tab.x(0).expect("second X gate on qubit 0");
        let mut rng = LcgRng::new(6);
        assert!(!tab.measure(0, &mut rng).expect("measure qubit 0"));
    }

    #[test]
    fn random_outcome_distribution_balanced() {
        // A single H on |0⟩ should give both outcomes across many seeds.
        let mut zeros = 0_u32;
        let mut ones = 0_u32;
        for seed in 0..200u64 {
            let mut tab = StabilizerTableau::new(1).expect("valid 1-qubit tableau");
            tab.h(0).expect("hadamard on qubit 0");
            let mut rng = LcgRng::new(seed.wrapping_mul(2_654_435_761));
            if tab.measure(0, &mut rng).expect("measure qubit 0") {
                ones += 1;
            } else {
                zeros += 1;
            }
        }
        assert!(
            zeros > 20 && ones > 20,
            "imbalanced: zeros={zeros} ones={ones}"
        );
    }

    #[test]
    fn two_qubit_independent_hadamards_uncorrelated_marginals() {
        // H on each of two qubits, then measure: both qubits individually random.
        let mut q0_ones = 0_u32;
        let mut q1_ones = 0_u32;
        for seed in 0..150u64 {
            let mut tab = StabilizerTableau::new(2).expect("valid 2-qubit tableau");
            tab.h(0).expect("hadamard on qubit 0");
            tab.h(1).expect("hadamard on qubit 1");
            let mut rng = LcgRng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
            if tab.measure(0, &mut rng).expect("measure qubit 0") {
                q0_ones += 1;
            }
            if tab.measure(1, &mut rng).expect("measure qubit 1") {
                q1_ones += 1;
            }
        }
        assert!(q0_ones > 15 && q0_ones < 135, "q0 not random: {q0_ones}");
        assert!(q1_ones > 15 && q1_ones < 135, "q1 not random: {q1_ones}");
    }
}
