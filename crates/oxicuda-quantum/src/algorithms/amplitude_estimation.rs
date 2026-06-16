//! Quantum Amplitude Estimation (QAE).
//!
//! Reference: Brassard, Høyer, Mosca, Tapp, *"Quantum Amplitude Amplification
//! and Estimation"*, Contemporary Mathematics 305 (2002), 53–74
//! (arXiv:quant-ph/0005055).
//!
//! # Problem
//!
//! Let `A` be a state-preparation unitary on `m` qubits whose last qubit (the
//! *flag*) marks "good" vs "bad" outcomes:
//!
//! ```text
//! A|0…0⟩ = √(1−a) |ψ_0⟩|0⟩  +  √a |ψ_1⟩|1⟩,
//! ```
//!
//! where `a ∈ [0, 1]` is the probability of measuring the flag in `|1⟩`. QAE
//! estimates `a` to a precision set by the number of *counting* qubits, with a
//! quadratically better query complexity than classical Monte-Carlo sampling.
//!
//! # The amplitude operator `Q`
//!
//! Amplitude amplification iterates the Grover-like operator
//!
//! ```text
//! Q = − A · S_0 · A† · S_χ,
//! ```
//!
//! where
//!
//! * `S_χ` is the reflection that **flips the sign of the good subspace** — here a
//!   `Z` on the flag qubit: `S_χ = I − 2|·⟩⟨·|_{flag=1}` acting as
//!   `|x⟩|1⟩ ↦ −|x⟩|1⟩`.
//! * `S_0` is the reflection about `|0…0⟩`: `S_0 = I − 2|0…0⟩⟨0…0|`.
//!
//! Restricted to the two-dimensional span of `|ψ_good⟩` and `|ψ_bad⟩`, `Q` is a
//! rotation by `2θ` where `a = sin²θ`. Its eigenvalues are `e^{±2iθ}`, so phase
//! estimation of `Q` reads `θ` (equivalently `ϕ = θ/π`) and recovers
//! `a = sin²(πϕ)`.
//!
//! # Algorithm (this implementation)
//!
//! 1. Build the `2^m × 2^m` dense matrix of `A`, of `Q`, and of `A†` from `A`
//!    applied to the computational basis (the simulator gives us `A`'s columns
//!    exactly).
//! 2. Lay out `n` counting qubits as the low bits `[0, n)` and the `m`-qubit
//!    state-prep register as `[n, n+m)`. Hadamard the counting register, apply
//!    `A` to the prep register, then the controlled-`Q^{2^k}` ladder, then the
//!    inverse QFT on the counting register — i.e. **standard QPE on `Q`**,
//!    reusing this crate's QFT.
//! 3. Read the most probable counting integer `y`, set `ϕ = y / 2^n`, and return
//!    `a = sin²(πϕ)`.
//!
//! Because the eigenvalues come in the conjugate pair `e^{±2iθ}`, the readout
//! peaks at both `ϕ` and `1−ϕ`, which map to the **same** `a = sin²(πϕ)`; the
//! routine folds them together.
//!
//! # Numerics
//!
//! The dense `A`/`Q` matrices and all reflections are assembled in `f32`
//! [`Complex`] consistent with the [`StateVector`]; the controlled-power ladder
//! squares `Q` densely. The estimator is exact up to the `f32` state vector and
//! the counting-grid resolution.

use crate::error::{QuantumError, QuantumResult};
use crate::fourier::qft::qft_inverse_inplace;
use crate::gates::hadamard::gate_h;
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;
use num_complex::Complex;

type Complex32 = Complex<f32>;

/// A state-preparation unitary `A` on `m` qubits, supplied as the closure that
/// applies `A` to a [`StateVector`]. The last qubit (`m − 1`) is the *flag*.
///
/// The dense matrix of `A` is extracted once (by applying `A` to each
/// computational basis state) and reused to build `A†` and the amplitude
/// operator `Q`.
pub struct StatePreparation {
    /// Number of qubits the preparation acts on (system + flag).
    m: usize,
    /// Row-major `dim×dim` matrix of `A` (`dim = 2^m`): column `j` is `A|j⟩`.
    matrix: Vec<Complex32>,
}

impl StatePreparation {
    /// Build from a closure that applies `A` in place to an `m`-qubit state.
    ///
    /// The closure is invoked `2^m` times (once per basis column) to extract the
    /// dense matrix; it must implement a genuine unitary or downstream unitarity
    /// checks will fail.
    ///
    /// # Errors
    /// * [`QuantumError::InvalidQubitCount`] if `m == 0` or `m > 12` (the dense
    ///   `2^m × 2^m` matrix would be impractical beyond that).
    /// * Propagates any error from the supplied `apply` closure.
    pub fn from_apply<F>(m: usize, mut apply: F) -> QuantumResult<Self>
    where
        F: FnMut(&mut StateVector) -> QuantumResult<()>,
    {
        if m == 0 || m > 12 {
            return Err(QuantumError::InvalidQubitCount { n: m });
        }
        let dim = 1usize << m;
        let mut matrix = vec![Complex32::new(0.0, 0.0); dim * dim];
        for j in 0..dim {
            // Basis state |j⟩.
            let mut amps = vec![Complex32::new(0.0, 0.0); dim];
            amps[j] = Complex32::new(1.0, 0.0);
            let mut sv = StateVector { amps, n_qubits: m };
            apply(&mut sv)?;
            // Column j of A is A|j⟩.
            for (r, a) in sv.amps.iter().enumerate() {
                matrix[r * dim + j] = *a;
            }
        }
        Ok(Self { m, matrix })
    }

    /// Number of qubits `m` (system + flag).
    #[must_use]
    pub fn m(&self) -> usize {
        self.m
    }

    /// The "good" probability `a = ⟨0…0| A† Π_1 A |0…0⟩`, computed exactly from
    /// the dense matrix: it is the total `|amplitude|²` of `A|0…0⟩` over basis
    /// states whose flag bit (qubit `m − 1`) is set.
    #[must_use]
    pub fn good_probability(&self) -> f64 {
        let dim = 1usize << self.m;
        let flag_mask = 1usize << (self.m - 1);
        // A|0…0⟩ is column 0 of the matrix: entry [r][0] = matrix[r * dim].
        let mut p = 0.0_f64;
        for r in 0..dim {
            if r & flag_mask != 0 {
                p += self.matrix[r * dim].norm_sqr() as f64;
            }
        }
        p
    }
}

/// Outcome of [`amplitude_estimation`].
#[derive(Debug, Clone)]
pub struct AmplitudeEstimationResult {
    /// Estimated good probability `a = sin²(πϕ) ∈ [0, 1]`.
    pub a: f64,
    /// Estimated angle `θ = πϕ ∈ [0, π/2]` (folded into the principal branch).
    pub theta: f64,
    /// The folded counting fraction `ϕ ∈ [0, 1/2]` driving the estimate.
    pub phi: f64,
    /// Winning counting-register integer `y` (before folding).
    pub integer: usize,
    /// Probability mass on the winning integer.
    pub probability: f64,
    /// Number of counting qubits used.
    pub n_counting: usize,
}

/// Row-major dense `dim×dim` matrix product `c = a · b`.
fn matmul(a: &[Complex32], b: &[Complex32], dim: usize) -> Vec<Complex32> {
    let mut c = vec![Complex32::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for k in 0..dim {
            let aik = a[i * dim + k];
            if aik.re == 0.0 && aik.im == 0.0 {
                continue;
            }
            for j in 0..dim {
                c[i * dim + j] += aik * b[k * dim + j];
            }
        }
    }
    c
}

/// Conjugate-transpose (adjoint) of a row-major `dim×dim` matrix.
fn dagger(a: &[Complex32], dim: usize) -> Vec<Complex32> {
    let mut out = vec![Complex32::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            out[i * dim + j] = a[j * dim + i].conj();
        }
    }
    out
}

/// Build `S_χ`: a diagonal reflection that flips the sign of every state whose
/// flag qubit (`m − 1`) is set. Returned as a dense `dim×dim` matrix.
fn reflection_chi(m: usize) -> Vec<Complex32> {
    let dim = 1usize << m;
    let flag_mask = 1usize << (m - 1);
    let mut s = vec![Complex32::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        let sign = if i & flag_mask != 0 { -1.0 } else { 1.0 };
        s[i * dim + i] = Complex32::new(sign, 0.0);
    }
    s
}

/// Build `S_0 = I − 2|0…0⟩⟨0…0|`: a diagonal reflection that flips the sign of
/// the all-zeros state only. Returned as a dense `dim×dim` matrix.
fn reflection_zero(m: usize) -> Vec<Complex32> {
    let dim = 1usize << m;
    let mut s = vec![Complex32::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        let sign = if i == 0 { -1.0 } else { 1.0 };
        s[i * dim + i] = Complex32::new(sign, 0.0);
    }
    s
}

/// Assemble the amplitude operator `Q = − A · S_0 · A† · S_χ` (dense, row-major).
fn amplitude_operator(prep: &StatePreparation) -> Vec<Complex32> {
    let dim = 1usize << prep.m;
    let a = &prep.matrix;
    let a_dag = dagger(a, dim);
    let s_chi = reflection_chi(prep.m);
    let s_zero = reflection_zero(prep.m);

    // Q = −(A · (S_0 · (A† · S_χ))).
    let t1 = matmul(&a_dag, &s_chi, dim); // A† S_χ
    let t2 = matmul(&s_zero, &t1, dim); // S_0 A† S_χ
    let t3 = matmul(a, &t2, dim); // A S_0 A† S_χ
    let mut q = t3;
    for z in &mut q {
        *z = -*z;
    }
    q
}

/// Apply a dense `dim×dim` matrix `u` to the contiguous low-bit *prep* block
/// `[base_qubit, base_qubit + m)` of `sv`, conditioned on `ctrl = 1`.
///
/// The prep qubits are contiguous starting at `base_qubit`, so within each
/// outer configuration the `dim = 2^m` prep amplitudes are gathered, multiplied
/// by `u`, and scattered back.
fn apply_controlled_dense(
    sv: &mut StateVector,
    ctrl: usize,
    base_qubit: usize,
    m: usize,
    u: &[Complex32],
) -> QuantumResult<()> {
    if ctrl >= sv.n_qubits {
        return Err(QuantumError::QubitIndexOutOfRange {
            index: ctrl,
            n_qubits: sv.n_qubits,
        });
    }
    let dim = 1usize << m;
    if u.len() != dim * dim {
        return Err(QuantumError::DimensionMismatch {
            expected: dim * dim,
            got: u.len(),
        });
    }
    let ctrl_mask = 1usize << ctrl;
    let prep_mask = (dim - 1) << base_qubit;
    let total = sv.amps.len();

    let mut base = 0usize;
    while base < total {
        // Block representative: control set, prep bits all zero.
        if (base & ctrl_mask) != 0 && (base & prep_mask) == 0 {
            let mut input = vec![Complex32::new(0.0, 0.0); dim];
            for (s, slot) in input.iter_mut().enumerate() {
                *slot = sv.amps[base | (s << base_qubit)];
            }
            for r in 0..dim {
                let mut acc = Complex32::new(0.0, 0.0);
                for (col, &xv) in input.iter().enumerate() {
                    acc += u[r * dim + col] * xv;
                }
                sv.amps[base | (r << base_qubit)] = acc;
            }
        }
        base += 1;
    }
    Ok(())
}

/// Apply a dense `dim×dim` matrix `u` to the contiguous low-bit prep block
/// `[base_qubit, base_qubit + m)` of `sv`, unconditionally.
fn apply_dense(
    sv: &mut StateVector,
    base_qubit: usize,
    m: usize,
    u: &[Complex32],
) -> QuantumResult<()> {
    let dim = 1usize << m;
    if u.len() != dim * dim {
        return Err(QuantumError::DimensionMismatch {
            expected: dim * dim,
            got: u.len(),
        });
    }
    let prep_mask = (dim - 1) << base_qubit;
    let total = sv.amps.len();
    let mut base = 0usize;
    while base < total {
        if base & prep_mask == 0 {
            let mut input = vec![Complex32::new(0.0, 0.0); dim];
            for (s, slot) in input.iter_mut().enumerate() {
                *slot = sv.amps[base | (s << base_qubit)];
            }
            for r in 0..dim {
                let mut acc = Complex32::new(0.0, 0.0);
                for (col, &xv) in input.iter().enumerate() {
                    acc += u[r * dim + col] * xv;
                }
                sv.amps[base | (r << base_qubit)] = acc;
            }
        }
        base += 1;
    }
    Ok(())
}

/// Run Quantum Amplitude Estimation with `n_counting` counting qubits.
///
/// Returns the estimated good probability `a` together with the intermediate
/// angle/phase quantities. The estimate's resolution is set by `n_counting`:
/// `a` lands on the grid `sin²(π y / 2^n)`.
///
/// # Errors
/// * [`QuantumError::InvalidParameter`] if `n_counting == 0`.
/// * [`QuantumError::InvalidQubitCount`] if the total register
///   `n_counting + m` exceeds the simulator's 30-qubit limit.
pub fn amplitude_estimation(
    prep: &StatePreparation,
    n_counting: usize,
) -> QuantumResult<AmplitudeEstimationResult> {
    if n_counting == 0 {
        return Err(QuantumError::InvalidParameter {
            name: "n_counting must be ≥ 1".into(),
        });
    }
    let m = prep.m;
    let n_qubits = n_counting + m;
    if n_qubits > 30 {
        return Err(QuantumError::InvalidQubitCount { n: n_qubits });
    }

    // Layout (little-endian): counting = [0, n_counting); prep = [n_counting,
    // n_counting + m).
    let count_qubits: Vec<usize> = (0..n_counting).collect();
    let base = n_counting;

    let mut sv = StateVector::new_zero_state(n_qubits)?;

    // Hadamard the counting register.
    for &q in &count_qubits {
        apply_1q_inplace(&mut sv, q, &gate_h())?;
    }

    // Prepare A|0…0⟩ on the prep register.
    apply_dense(&mut sv, base, m, &prep.matrix)?;

    // Controlled-Q^{2^k} ladder via repeated dense squaring.
    let dim = 1usize << m;
    let mut q_pow = amplitude_operator(prep);
    for &cq in &count_qubits {
        apply_controlled_dense(&mut sv, cq, base, m, &q_pow)?;
        q_pow = matmul(&q_pow, &q_pow, dim);
    }

    // Inverse QFT on the counting register (little-endian).
    qft_inverse_inplace(&mut sv, &count_qubits)?;

    // Marginal readout over the counting register.
    let n_outcomes = 1usize << n_counting;
    let mut probs = vec![0.0_f64; n_outcomes];
    for (i, a) in sv.amps.iter().enumerate() {
        let mut c = 0usize;
        for (k, &q) in count_qubits.iter().enumerate() {
            c |= ((i >> q) & 1) << k;
        }
        probs[c] += a.norm_sqr() as f64;
    }

    let mut integer = 0usize;
    let mut best = f64::NEG_INFINITY;
    for (c, &p) in probs.iter().enumerate() {
        if p > best {
            best = p;
            integer = c;
        }
    }

    // ϕ = y / 2^n; fold the conjugate eigenvalue branch (ϕ and 1−ϕ give the same
    // a) into [0, 1/2].
    let n_levels = n_outcomes as f64;
    let raw_phi = integer as f64 / n_levels;
    let phi = if raw_phi > 0.5 {
        1.0 - raw_phi
    } else {
        raw_phi
    };
    let theta = std::f64::consts::PI * phi;
    let a = theta.sin().powi(2);

    Ok(AmplitudeEstimationResult {
        a,
        theta,
        phi,
        integer,
        probability: best,
        n_counting,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::parametric::gate_ry;
    use std::f64::consts::PI;

    /// A single-qubit state-prep `A = R_y(2θ)` so that
    /// `A|0⟩ = cos θ |0⟩ + sin θ |1⟩`, hence the flag (the only qubit) is `|1⟩`
    /// with probability `a = sin²θ`.
    fn ry_prep(theta: f64) -> StatePreparation {
        StatePreparation::from_apply(1, move |sv| {
            apply_1q_inplace(sv, 0, &gate_ry((2.0 * theta) as f32))
        })
        .expect("Ry state preparation with valid single-qubit operator")
    }

    /// Conjugate-transpose check helper: ‖U U† − I‖_∞.
    fn unitarity_residual(u: &[Complex32], dim: usize) -> f32 {
        let ud = dagger(u, dim);
        let prod = matmul(u, &ud, dim);
        let mut worst = 0.0_f32;
        for i in 0..dim {
            for j in 0..dim {
                let expect = if i == j {
                    Complex32::new(1.0, 0.0)
                } else {
                    Complex32::new(0.0, 0.0)
                };
                worst = worst.max((prod[i * dim + j] - expect).norm());
            }
        }
        worst
    }

    // (a) a ∈ {1/4, 1/2, 3/4} land near QPE grid points → estimate ≈ a.
    #[test]
    fn estimates_quarter_half_three_quarter() {
        // a = sin²θ. θ = π·ϕ with ϕ = y/2^n. For a = 1/2, θ = π/4 ⇒ ϕ = 1/4,
        // a grid point at n ≥ 2. For a = 1/4, θ = π/6 ⇒ ϕ = 1/6 (≈ near 2/12);
        // pick n large enough that the nearest grid point is close. For a = 3/4,
        // θ = π/3 ⇒ ϕ = 1/3.
        let n = 6usize;
        for &a_true in &[0.25_f64, 0.5, 0.75] {
            let theta = a_true.sqrt().asin();
            let prep = ry_prep(theta);
            let res = amplitude_estimation(&prep, n)
                .expect("amplitude estimation succeeds for valid preparation");
            assert!(
                (res.a - a_true).abs() < 0.05,
                "a_true={a_true}: estimate {} (ϕ={}, y={})",
                res.a,
                res.phi,
                res.integer
            );
        }
    }

    // a = 1/2 is an EXACT grid point (ϕ = 1/4) and must be recovered tightly.
    #[test]
    fn exact_grid_point_half() {
        let prep = ry_prep((0.5_f64).sqrt().asin()); // θ = π/4
        let res = amplitude_estimation(&prep, 4)
            .expect("amplitude estimation with 4 counting qubits succeeds");
        assert!((res.a - 0.5).abs() < 1e-3, "a={}", res.a);
        assert!((res.phi - 0.25).abs() < 1e-6, "phi={}", res.phi);
    }

    // (b) a = 0 → 0 and a = 1 → 1.
    #[test]
    fn extremes_zero_and_one() {
        // a = 0: θ = 0, A = I (R_y(0)). Good probability 0.
        let prep0 = ry_prep(0.0);
        assert!(prep0.good_probability().abs() < 1e-7);
        let res0 = amplitude_estimation(&prep0, 5)
            .expect("amplitude estimation for zero-amplitude preparation");
        assert!(res0.a.abs() < 1e-6, "a(0)={}", res0.a);

        // a = 1: θ = π/2, A|0⟩ = |1⟩. Good probability 1; ϕ = 1/2 grid point.
        let prep1 = ry_prep(PI / 2.0);
        assert!((prep1.good_probability() - 1.0).abs() < 1e-6);
        let res1 = amplitude_estimation(&prep1, 5)
            .expect("amplitude estimation for unit-amplitude preparation");
        assert!((res1.a - 1.0).abs() < 1e-6, "a(1)={}", res1.a);
    }

    // (c) Q is unitary: Q Q† = I to 1e-5.
    #[test]
    fn amplitude_operator_is_unitary() {
        for &theta in &[0.1_f64, PI / 6.0, PI / 4.0, PI / 3.0, 1.2, PI / 2.0] {
            let prep = ry_prep(theta);
            let q = amplitude_operator(&prep);
            let dim = 1usize << prep.m;
            let res = unitarity_residual(&q, dim);
            assert!(res < 1e-5, "θ={theta}: ‖QQ†−I‖={res}");
        }
        // Also a 2-qubit prep (system + flag) to exercise the dense path.
        let prep2 = StatePreparation::from_apply(2, |sv| {
            apply_1q_inplace(sv, 0, &gate_ry(0.7))?;
            // Make qubit 1 (the flag) depend on qubit 0 via a CNOT-like effect.
            apply_1q_inplace(sv, 1, &gate_ry(1.1))
        })
        .expect("2-qubit state preparation with two Ry rotations succeeds");
        let q2 = amplitude_operator(&prep2);
        let r2 = unitarity_residual(&q2, 4);
        assert!(r2 < 1e-5, "2-qubit ‖QQ†−I‖={r2}");
    }

    // (d) The eigenphase relation θ = arcsin(√a) holds: Q's eigenvalues are
    //     e^{±2iθ}, so the QPE-recovered θ matches arcsin(√a).
    #[test]
    fn eigenphase_relation_holds() {
        let n = 8usize;
        for &a_true in &[0.25_f64, 0.5, 0.75] {
            let theta_true = a_true.sqrt().asin();
            let prep = ry_prep(theta_true);
            let res = amplitude_estimation(&prep, n)
                .expect("amplitude estimation succeeds for eigenphase test");
            // Recovered θ = π·ϕ should match arcsin(√a_true) closely.
            assert!(
                (res.theta - theta_true).abs() < 0.05,
                "a={a_true}: θ_est={} vs arcsin√a={theta_true}",
                res.theta
            );
        }
    }

    // (e) More counting qubits → strictly finer resolution (error shrinks) for a
    //     non-grid value of a.
    #[test]
    fn more_qubits_finer_resolution() {
        // a with θ = 1.0 rad (not a dyadic multiple of π): ϕ = 1/π ≈ 0.3183,
        // never exactly on the grid, so the error is resolution-limited and must
        // shrink monotonically (loosely) as n grows.
        let theta = 1.0_f64;
        let a_true = theta.sin().powi(2);
        let mut prev_err = f64::INFINITY;
        for n in [3usize, 5, 7, 9] {
            let prep = ry_prep(theta);
            let res = amplitude_estimation(&prep, n)
                .expect("amplitude estimation with increasing counting qubits succeeds");
            let err = (res.a - a_true).abs();
            assert!(
                err <= prev_err + 1e-9,
                "n={n}: error {err} did not shrink (prev {prev_err})"
            );
            prev_err = err;
        }
        // And the finest resolution is genuinely accurate.
        let res_fine = amplitude_estimation(&ry_prep(theta), 10)
            .expect("fine-resolution amplitude estimation with 10 counting qubits");
        assert!((res_fine.a - a_true).abs() < 0.01, "fine a={}", res_fine.a);
    }

    // (f) S_0 and S_χ reflections are correct on small states (unit tests).
    #[test]
    fn reflections_are_correct() {
        // S_χ on 1 qubit flips |1⟩ only: diag(1, -1) = Z.
        let s_chi = reflection_chi(1);
        assert!((s_chi[0] - Complex32::new(1.0, 0.0)).norm() < 1e-7); // [0][0]
        assert!(s_chi[1].norm() < 1e-7); // [0][1]
        assert!(s_chi[2].norm() < 1e-7); // [1][0]
        assert!((s_chi[3] - Complex32::new(-1.0, 0.0)).norm() < 1e-7); // [1][1]

        // S_0 on 2 qubits flips |00⟩ only.
        let s0 = reflection_zero(2);
        assert!((s0[0] - Complex32::new(-1.0, 0.0)).norm() < 1e-7); // [0][0] = -1
        for i in 1..4 {
            assert!(
                (s0[i * 4 + i] - Complex32::new(1.0, 0.0)).norm() < 1e-7,
                "S0 diag[{i}]"
            );
        }

        // S_χ on 2 qubits (flag = qubit 1) flips states with bit 1 set: indices
        // 2 (|10⟩) and 3 (|11⟩).
        let s_chi2 = reflection_chi(2);
        let diag: Vec<f32> = (0..4).map(|i| s_chi2[i * 4 + i].re).collect();
        assert_eq!(diag, vec![1.0, 1.0, -1.0, -1.0], "S_χ diag={diag:?}");
    }

    // good_probability matches the |R_y| amplitude formula a = sin²θ exactly.
    #[test]
    fn good_probability_matches_formula() {
        for &theta in &[0.0_f64, 0.3, PI / 6.0, PI / 4.0, 1.0, PI / 2.0] {
            let prep = ry_prep(theta);
            let want = theta.sin().powi(2);
            assert!(
                (prep.good_probability() - want).abs() < 1e-6,
                "θ={theta}: good_prob={} vs sin²θ={want}",
                prep.good_probability()
            );
        }
    }

    // Validation: n_counting = 0 and oversized registers error out.
    #[test]
    fn rejects_invalid_inputs() {
        let prep = ry_prep(0.5);
        assert!(amplitude_estimation(&prep, 0).is_err());
        // m = 0 prep is rejected at construction.
        assert!(StatePreparation::from_apply(0, |_| Ok(())).is_err());
    }
}
