//! Variational Quantum Linear Solver (VQLS).
//!
//! Reference: Bravo-Prieto, LaRose, Cerezo, Subasi, Cincio, Coles, *"Variational
//! Quantum Linear Solver"*, Quantum 7, 1188 (2023) (arXiv:1909.05820, 2019).
//!
//! # Problem
//!
//! Solve `A|x⟩ = |b⟩` for `|x⟩`, where the matrix is a *linear combination of
//! unitaries* (LCU)
//!
//! ```text
//! A = Σ_l c_l P_l,
//! ```
//!
//! with real coefficients `c_l` and Pauli-string unitaries `P_l`. VQLS prepares a
//! trial solution `|x(θ)⟩ = V(θ)|0⟩` from a parameterised *ansatz* `V(θ)` and
//! minimises a cost that vanishes precisely when `A|x(θ)⟩ ∝ |b⟩`.
//!
//! # Cost function
//!
//! This is a state-vector simulator, so rather than estimating the cost from
//! Hadamard tests we evaluate the **global cost** exactly:
//!
//! ```text
//! C(θ) = 1 − |⟨b| A |x(θ)⟩|² / ⟨x(θ)| A†A |x(θ)⟩.
//! ```
//!
//! `C(θ) ≥ 0` always, and `C(θ) = 0` **iff** the normalised `A|x(θ)⟩` equals
//! `|b⟩` up to a global phase, i.e. `|x(θ)⟩ ∝ A⁻¹|b⟩`. Minimising `C` therefore
//! drives the ansatz state to the (normalised) solution.
//!
//! # Ansatz
//!
//! `V(θ)` is a hardware-efficient layered circuit: each layer applies an `R_y(θ)`
//! to every qubit followed by a ladder of nearest-neighbour `CNOT`s. With enough
//! layers this reaches any real-amplitude state, which suffices for the
//! real-coefficient test systems here.
//!
//! # Optimiser
//!
//! The parameters are optimised by **finite-difference gradient descent** with a
//! simple backtracking step (halving the learning rate when a step fails to
//! decrease the cost). All gradients are central differences of the exactly
//! evaluated cost.
//!
//! # Numerics
//!
//! Pauli strings are expanded to dense `f32` matrices and combined into the dense
//! `A`; the ansatz drives an `f32` [`StateVector`] through the shared gate
//! machinery. Cost, gradients, and the LCU assembly are exact up to `f32`.

use crate::error::{QuantumError, QuantumResult};
use crate::gates::controlled::apply_cnot;
use crate::gates::parametric::gate_ry;
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;
use num_complex::Complex;

type Complex32 = Complex<f32>;

/// A single-qubit Pauli factor used to build Pauli-string unitaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pauli {
    /// Identity.
    I,
    /// Pauli-X.
    X,
    /// Pauli-Y.
    Y,
    /// Pauli-Z.
    Z,
}

impl Pauli {
    /// The `2×2` matrix of this Pauli, row-major.
    #[must_use]
    fn matrix(self) -> [[Complex32; 2]; 2] {
        let zero = Complex32::new(0.0, 0.0);
        let one = Complex32::new(1.0, 0.0);
        let neg = Complex32::new(-1.0, 0.0);
        let i = Complex32::new(0.0, 1.0);
        let ni = Complex32::new(0.0, -1.0);
        match self {
            Pauli::I => [[one, zero], [zero, one]],
            Pauli::X => [[zero, one], [one, zero]],
            Pauli::Y => [[zero, ni], [i, zero]],
            Pauli::Z => [[one, zero], [zero, neg]],
        }
    }
}

/// One term `c_l · P_l` of the LCU `A = Σ_l c_l P_l`, where `P_l` is a Pauli
/// string over `n` qubits.
#[derive(Debug, Clone)]
pub struct PauliTerm {
    /// Real coefficient `c_l`.
    pub coeff: f32,
    /// Per-qubit Pauli factors; `paulis[q]` acts on qubit `q` (little-endian).
    pub paulis: Vec<Pauli>,
}

/// The LCU operator `A = Σ_l c_l P_l` and its dense `f32` matrix.
#[derive(Debug, Clone)]
pub struct LcuOperator {
    /// Number of qubits `n` the operator acts on.
    n_qubits: usize,
    /// Row-major dense `dim×dim` matrix of `A` (`dim = 2^n`).
    matrix: Vec<Complex32>,
}

impl LcuOperator {
    /// Build the dense matrix of `A = Σ_l c_l P_l` from a list of Pauli-string
    /// terms over `n_qubits` qubits.
    ///
    /// # Errors
    /// * [`QuantumError::EmptyInput`] if `terms` is empty.
    /// * [`QuantumError::InvalidQubitCount`] if `n_qubits == 0` or `n_qubits > 8`
    ///   (the dense matrix would be impractical beyond that).
    /// * [`QuantumError::DimensionMismatch`] if any term's Pauli-string length is
    ///   not `n_qubits`.
    pub fn new(terms: &[PauliTerm], n_qubits: usize) -> QuantumResult<Self> {
        if terms.is_empty() {
            return Err(QuantumError::EmptyInput);
        }
        if n_qubits == 0 || n_qubits > 8 {
            return Err(QuantumError::InvalidQubitCount { n: n_qubits });
        }
        let dim = 1usize << n_qubits;
        let mut matrix = vec![Complex32::new(0.0, 0.0); dim * dim];
        for term in terms {
            if term.paulis.len() != n_qubits {
                return Err(QuantumError::DimensionMismatch {
                    expected: n_qubits,
                    got: term.paulis.len(),
                });
            }
            let pmat = pauli_string_matrix(&term.paulis, n_qubits);
            let c = Complex32::new(term.coeff, 0.0);
            for (dst, src) in matrix.iter_mut().zip(pmat.iter()) {
                *dst += c * *src;
            }
        }
        Ok(Self { n_qubits, matrix })
    }

    /// Number of qubits `n`.
    #[must_use]
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// The dense row-major matrix entries of `A`.
    #[must_use]
    pub fn matrix(&self) -> &[Complex32] {
        &self.matrix
    }

    /// Apply `A` to a vector `v` (length `dim`), returning `A·v`.
    #[must_use]
    fn apply_to(&self, v: &[Complex32]) -> Vec<Complex32> {
        let dim = 1usize << self.n_qubits;
        let mut out = vec![Complex32::new(0.0, 0.0); dim];
        for (r, slot) in out.iter_mut().enumerate() {
            let mut acc = Complex32::new(0.0, 0.0);
            for (col, &xv) in v.iter().enumerate() {
                acc += self.matrix[r * dim + col] * xv;
            }
            *slot = acc;
        }
        out
    }

    /// Classical normalised solution `A⁻¹|b⟩` (the reference target), or `None`
    /// if `A` is singular within tolerance. `b` is normalised internally.
    #[must_use]
    pub fn classical_solution(&self, b: &[Complex32]) -> Option<Vec<Complex32>> {
        let dim = 1usize << self.n_qubits;
        if b.len() != dim {
            return None;
        }
        // Solve A x = b by dense Gauss–Jordan elimination in f64 for stability.
        let mut aug = vec![Complex::<f64>::new(0.0, 0.0); dim * (dim + 1)];
        for r in 0..dim {
            for c in 0..dim {
                let z = self.matrix[r * dim + c];
                aug[r * (dim + 1) + c] = Complex::<f64>::new(z.re as f64, z.im as f64);
            }
            let bz = b[r];
            aug[r * (dim + 1) + dim] = Complex::<f64>::new(bz.re as f64, bz.im as f64);
        }
        // Forward elimination with partial pivoting on |pivot|.
        for col in 0..dim {
            // Find pivot row.
            let mut pivot_row = col;
            let mut best = aug[col * (dim + 1) + col].norm();
            for r in (col + 1)..dim {
                let mag = aug[r * (dim + 1) + col].norm();
                if mag > best {
                    best = mag;
                    pivot_row = r;
                }
            }
            if best < 1e-12 {
                return None; // singular
            }
            if pivot_row != col {
                for c in 0..=dim {
                    aug.swap(col * (dim + 1) + c, pivot_row * (dim + 1) + c);
                }
            }
            let pivot = aug[col * (dim + 1) + col];
            for c in col..=dim {
                aug[col * (dim + 1) + c] /= pivot;
            }
            for r in 0..dim {
                if r == col {
                    continue;
                }
                let factor = aug[r * (dim + 1) + col];
                if factor.norm() < 1e-300 {
                    continue;
                }
                for c in col..=dim {
                    let sub = factor * aug[col * (dim + 1) + c];
                    aug[r * (dim + 1) + c] -= sub;
                }
            }
        }
        // Extract and normalise the solution.
        let mut x: Vec<Complex32> = (0..dim)
            .map(|r| {
                let z = aug[r * (dim + 1) + dim];
                Complex32::new(z.re as f32, z.im as f32)
            })
            .collect();
        let norm: f32 = x.iter().map(|z| z.norm_sqr()).sum::<f32>().sqrt();
        if norm < 1e-9 {
            return None;
        }
        for z in &mut x {
            *z /= norm;
        }
        Some(x)
    }
}

/// Build the dense `2^n × 2^n` matrix of a Pauli string (tensor product, with
/// qubit `0` the least-significant bit), row-major.
fn pauli_string_matrix(paulis: &[Pauli], n_qubits: usize) -> Vec<Complex32> {
    let dim = 1usize << n_qubits;
    let single: Vec<[[Complex32; 2]; 2]> = paulis.iter().map(|p| p.matrix()).collect();
    let mut out = vec![Complex32::new(0.0, 0.0); dim * dim];
    for row in 0..dim {
        for col in 0..dim {
            // Matrix element = Π_q P_q[row_bit_q][col_bit_q].
            let mut elem = Complex32::new(1.0, 0.0);
            for (q, pmat) in single.iter().enumerate() {
                let rb = (row >> q) & 1;
                let cb = (col >> q) & 1;
                elem *= pmat[rb][cb];
                if elem.re == 0.0 && elem.im == 0.0 {
                    break;
                }
            }
            out[row * dim + col] = elem;
        }
    }
    out
}

/// A hardware-efficient ansatz: `n_layers` layers, each an `R_y` on every qubit
/// followed by a nearest-neighbour `CNOT` ladder.
#[derive(Debug, Clone)]
pub struct HardwareEfficientAnsatz {
    /// Number of qubits.
    n_qubits: usize,
    /// Number of layers.
    n_layers: usize,
}

impl HardwareEfficientAnsatz {
    /// Create an ansatz over `n_qubits` qubits with `n_layers` layers.
    ///
    /// # Errors
    /// [`QuantumError::IncompatibleAnsatz`] if `n_qubits == 0` or `n_layers == 0`.
    pub fn new(n_qubits: usize, n_layers: usize) -> QuantumResult<Self> {
        if n_qubits == 0 || n_layers == 0 {
            return Err(QuantumError::IncompatibleAnsatz);
        }
        Ok(Self { n_qubits, n_layers })
    }

    /// Number of variational parameters: one `R_y` angle per qubit per layer.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.n_qubits * self.n_layers
    }

    /// Prepare `|x(θ)⟩ = V(θ)|0…0⟩`.
    ///
    /// # Errors
    /// [`QuantumError::DimensionMismatch`] if `params.len() != n_params()`;
    /// propagates gate-application errors.
    pub fn prepare(&self, params: &[f32]) -> QuantumResult<StateVector> {
        if params.len() != self.n_params() {
            return Err(QuantumError::DimensionMismatch {
                expected: self.n_params(),
                got: params.len(),
            });
        }
        let mut sv = StateVector::new_zero_state(self.n_qubits)?;
        let mut idx = 0usize;
        for _layer in 0..self.n_layers {
            for q in 0..self.n_qubits {
                apply_1q_inplace(&mut sv, q, &gate_ry(params[idx]))?;
                idx += 1;
            }
            // Nearest-neighbour CNOT ladder (skipped for a single qubit).
            for q in 0..self.n_qubits.saturating_sub(1) {
                apply_cnot(&mut sv, q, q + 1)?;
            }
        }
        Ok(sv)
    }
}

/// The VQLS optimiser: holds the operator `A`, the normalised target `|b⟩`, and
/// the ansatz, and minimises the global cost over the ansatz parameters.
#[derive(Debug, Clone)]
pub struct VqlsSolver {
    operator: LcuOperator,
    /// Normalised right-hand side `|b⟩` (length `dim`).
    b: Vec<Complex32>,
    ansatz: HardwareEfficientAnsatz,
}

/// Result of a VQLS optimisation run.
#[derive(Debug, Clone)]
pub struct VqlsResult {
    /// Optimised parameters.
    pub params: Vec<f32>,
    /// Final cost `C(θ*)`.
    pub final_cost: f64,
    /// Cost history, one entry per iteration (including the initial cost).
    pub cost_history: Vec<f64>,
    /// The optimised normalised ansatz state `|x(θ*)⟩`.
    pub solution: Vec<Complex32>,
}

impl VqlsSolver {
    /// Build a solver for `A|x⟩ = |b⟩`. `b` is normalised internally and must be
    /// non-zero with `b.len() == 2^{n_qubits}`.
    ///
    /// # Errors
    /// * [`QuantumError::DimensionMismatch`] if `b`'s length or the ansatz qubit
    ///   count disagree with the operator's qubit count.
    /// * [`QuantumError::InvalidParameter`] if `b` is the zero vector.
    pub fn new(
        operator: LcuOperator,
        b: &[Complex32],
        ansatz: HardwareEfficientAnsatz,
    ) -> QuantumResult<Self> {
        let dim = 1usize << operator.n_qubits();
        if b.len() != dim {
            return Err(QuantumError::DimensionMismatch {
                expected: dim,
                got: b.len(),
            });
        }
        if ansatz.n_qubits != operator.n_qubits() {
            return Err(QuantumError::DimensionMismatch {
                expected: operator.n_qubits(),
                got: ansatz.n_qubits,
            });
        }
        let norm: f32 = b.iter().map(|z| z.norm_sqr()).sum::<f32>().sqrt();
        if norm < 1e-12 {
            return Err(QuantumError::InvalidParameter {
                name: "right-hand side b must be non-zero".into(),
            });
        }
        let inv = 1.0 / norm;
        let b_norm: Vec<Complex32> = b.iter().map(|z| *z * inv).collect();
        Ok(Self {
            operator,
            b: b_norm,
            ansatz,
        })
    }

    /// Evaluate the global cost
    /// `C(θ) = 1 − |⟨b|A|x⟩|² / ⟨x|A†A|x⟩` for a given parameter vector.
    ///
    /// # Errors
    /// Propagates ansatz preparation errors.
    pub fn cost(&self, params: &[f32]) -> QuantumResult<f64> {
        let x = self.ansatz.prepare(params)?;
        Ok(self.cost_of_state(&x.amps))
    }

    /// Global cost of an explicit (not necessarily normalised) state amplitude
    /// vector `|x⟩`.
    fn cost_of_state(&self, x: &[Complex32]) -> f64 {
        let ax = self.operator.apply_to(x);
        // Denominator ⟨x|A†A|x⟩ = ‖A x‖².
        let denom: f64 = ax.iter().map(|z| z.norm_sqr() as f64).sum();
        if denom < 1e-18 {
            // A|x⟩ = 0 ⇒ cannot align with |b⟩; maximal cost.
            return 1.0;
        }
        // Numerator |⟨b|A x⟩|².
        let mut inner = Complex::<f64>::new(0.0, 0.0);
        for (bz, az) in self.b.iter().zip(ax.iter()) {
            let b64 = Complex::<f64>::new(bz.re as f64, bz.im as f64);
            let a64 = Complex::<f64>::new(az.re as f64, az.im as f64);
            inner += b64.conj() * a64;
        }
        let overlap = inner.norm_sqr();
        let cost = 1.0 - overlap / denom;
        // Numerical hygiene: clamp tiny negative round-off to 0.
        cost.max(0.0)
    }

    /// The normalised ansatz state `|x(θ)⟩` for inspection.
    ///
    /// # Errors
    /// Propagates ansatz preparation errors.
    pub fn solution_state(&self, params: &[f32]) -> QuantumResult<Vec<Complex32>> {
        let mut x = self.ansatz.prepare(params)?;
        x.normalize_inplace();
        Ok(x.amps)
    }

    /// Minimise the cost by finite-difference gradient descent.
    ///
    /// Starts from `init_params`, runs up to `max_iters` iterations with central
    /// difference step `eps` and an initial learning rate `lr` that is halved
    /// (backtracking) whenever a step fails to decrease the cost. Stops early when
    /// the cost drops below `tol` or the learning rate underflows.
    ///
    /// # Errors
    /// * [`QuantumError::DimensionMismatch`] if `init_params.len() != n_params()`.
    /// * Propagates cost-evaluation errors.
    pub fn optimize(
        &self,
        init_params: &[f32],
        max_iters: usize,
        lr: f32,
        eps: f32,
        tol: f64,
    ) -> QuantumResult<VqlsResult> {
        let n = self.ansatz.n_params();
        if init_params.len() != n {
            return Err(QuantumError::DimensionMismatch {
                expected: n,
                got: init_params.len(),
            });
        }
        let mut params = init_params.to_vec();
        let mut current = self.cost(&params)?;
        let mut history = vec![current];
        let mut step = lr;

        for _ in 0..max_iters {
            if current < tol {
                break;
            }
            // Central-difference gradient.
            let mut grad = vec![0.0_f32; n];
            for (i, g) in grad.iter_mut().enumerate() {
                let saved = params[i];
                params[i] = saved + eps;
                let c_plus = self.cost(&params)?;
                params[i] = saved - eps;
                let c_minus = self.cost(&params)?;
                params[i] = saved;
                *g = ((c_plus - c_minus) / (2.0 * eps as f64)) as f32;
            }
            // Backtracking line search along −grad.
            let mut accepted = false;
            let mut local_step = step;
            for _try in 0..20 {
                let trial: Vec<f32> = params
                    .iter()
                    .zip(grad.iter())
                    .map(|(&p, &g)| p - local_step * g)
                    .collect();
                let c_trial = self.cost(&trial)?;
                if c_trial < current {
                    params = trial;
                    current = c_trial;
                    accepted = true;
                    // Mildly grow the step for the next iteration.
                    step = local_step * 1.2;
                    break;
                }
                local_step *= 0.5;
            }
            history.push(current);
            if !accepted {
                // Converged to a (local) stationary point at this resolution.
                break;
            }
        }

        let solution = self.solution_state(&params)?;
        Ok(VqlsResult {
            params,
            final_cost: current,
            cost_history: history,
            solution,
        })
    }
}

/// Fidelity `|⟨u|v⟩|²` between two normalised `f32` amplitude vectors.
#[must_use]
pub fn fidelity(u: &[Complex32], v: &[Complex32]) -> f64 {
    let mut inner = Complex::<f64>::new(0.0, 0.0);
    for (a, b) in u.iter().zip(v.iter()) {
        let a64 = Complex::<f64>::new(a.re as f64, a.im as f64);
        let b64 = Complex::<f64>::new(b.re as f64, b.im as f64);
        inner += a64.conj() * b64;
    }
    inner.norm_sqr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn c(re: f32) -> Complex32 {
        Complex32::new(re, 0.0)
    }

    /// Random initial parameters in [−π, π].
    fn random_params(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..n)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * std::f32::consts::PI)
            .collect()
    }

    // (f) The LCU Σ c_l P_l reconstructs the intended A (unit test).
    #[test]
    fn lcu_reconstructs_matrix() {
        // A = 0.6·I + 0.8·X on 1 qubit = [[0.6, 0.8], [0.8, 0.6]].
        let terms = vec![
            PauliTerm {
                coeff: 0.6,
                paulis: vec![Pauli::I],
            },
            PauliTerm {
                coeff: 0.8,
                paulis: vec![Pauli::X],
            },
        ];
        let op = LcuOperator::new(&terms, 1).expect("valid 1-qubit LCU operator from I+X terms");
        let m = op.matrix();
        assert!((m[0] - c(0.6)).norm() < 1e-6, "[0][0]={:?}", m[0]);
        assert!((m[1] - c(0.8)).norm() < 1e-6, "[0][1]={:?}", m[1]);
        assert!((m[2] - c(0.8)).norm() < 1e-6, "[1][0]={:?}", m[2]);
        assert!((m[3] - c(0.6)).norm() < 1e-6, "[1][1]={:?}", m[3]);

        // 2-qubit: A = 1.0·(Z⊗I) + 0.5·(X⊗X). Check a couple of entries.
        // Z⊗I is diag(1,1,-1,-1) in little-endian (Z on qubit 1). X⊗X is the
        // anti-diagonal all-ones. paulis[0] acts on qubit 0.
        let terms2 = vec![
            PauliTerm {
                coeff: 1.0,
                paulis: vec![Pauli::I, Pauli::Z], // I on q0, Z on q1
            },
            PauliTerm {
                coeff: 0.5,
                paulis: vec![Pauli::X, Pauli::X],
            },
        ];
        let op2 = LcuOperator::new(&terms2, 2)
            .expect("valid 2-qubit LCU operator from Z\u{2297}I and X\u{2297}X terms");
        let m2 = op2.matrix();
        let at = |row: usize, col: usize| m2[row * 4 + col];
        // Diagonal from Z on q1: index bit-1 = 0 → +1 (indices 0,1); bit-1 = 1 →
        // −1 (indices 2,3).
        assert!((at(0, 0) - c(1.0)).norm() < 1e-6);
        assert!((at(1, 1) - c(1.0)).norm() < 1e-6);
        assert!((at(2, 2) - c(-1.0)).norm() < 1e-6);
        assert!((at(3, 3) - c(-1.0)).norm() < 1e-6);
        // X⊗X couples |00⟩↔|11⟩ and |01⟩↔|10⟩ with 0.5.
        assert!((at(0, 3) - c(0.5)).norm() < 1e-6, "00-11");
        assert!((at(1, 2) - c(0.5)).norm() < 1e-6, "01-10");
    }

    // (e) Cost is non-negative for arbitrary parameters.
    #[test]
    fn cost_is_non_negative() {
        let terms = vec![
            PauliTerm {
                coeff: 0.7,
                paulis: vec![Pauli::I],
            },
            PauliTerm {
                coeff: 0.4,
                paulis: vec![Pauli::X],
            },
        ];
        let op = LcuOperator::new(&terms, 1).expect("valid 1-qubit LCU operator from I+X terms");
        let b = vec![c(1.0), c(0.0)];
        let ansatz = HardwareEfficientAnsatz::new(1, 2).expect("valid 1-qubit 2-layer ansatz");
        let solver = VqlsSolver::new(op, &b, ansatz).expect("valid VQLS solver");
        for seed in 0..20 {
            let p = random_params(solver.ansatz.n_params(), seed);
            let cost = solver.cost(&p).expect("cost evaluation should succeed");
            assert!(cost >= -1e-9, "seed {seed}: cost {cost} negative");
            assert!(cost <= 1.0 + 1e-9, "seed {seed}: cost {cost} > 1");
        }
    }

    // (c) cost = 0 ⟺ |x⟩ ∝ A⁻¹|b⟩: the exact classical solution gives ~0 cost.
    #[test]
    fn exact_solution_gives_zero_cost() {
        // A = 0.6·I + 0.8·X; b = |0⟩.
        let terms = vec![
            PauliTerm {
                coeff: 0.6,
                paulis: vec![Pauli::I],
            },
            PauliTerm {
                coeff: 0.8,
                paulis: vec![Pauli::X],
            },
        ];
        let op = LcuOperator::new(&terms, 1).expect("valid 1-qubit LCU operator from I+X terms");
        let b = vec![c(1.0), c(0.0)];
        let ansatz = HardwareEfficientAnsatz::new(1, 1).expect("valid 1-qubit 1-layer ansatz");
        let target = op
            .classical_solution(&b)
            .expect("classical solution should exist for invertible A");
        let solver = VqlsSolver::new(op, &b, ansatz).expect("valid VQLS solver");
        // Feed the exact (normalised) solution amplitudes directly into the cost.
        let cost = solver.cost_of_state(&target);
        assert!(cost < 1e-6, "exact-solution cost {cost} should be ~0");
    }

    // (a) After optimisation the normalised |x⟩ has fidelity > 0.95 with A⁻¹|b⟩.
    #[test]
    fn optimized_state_has_high_fidelity() {
        // A = 0.8·I + 0.6·Z (diagonal, invertible): A = diag(1.4, 0.2). b = (1,1)/√2.
        let terms = vec![
            PauliTerm {
                coeff: 0.8,
                paulis: vec![Pauli::I],
            },
            PauliTerm {
                coeff: 0.6,
                paulis: vec![Pauli::Z],
            },
        ];
        let op = LcuOperator::new(&terms, 1).expect("valid 1-qubit LCU operator from I+Z terms");
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let b = vec![c(inv_sqrt2), c(inv_sqrt2)];
        let target = op
            .classical_solution(&b)
            .expect("classical solution should exist for invertible A");
        let ansatz = HardwareEfficientAnsatz::new(1, 2).expect("valid 1-qubit 2-layer ansatz");
        let solver = VqlsSolver::new(op, &b, ansatz).expect("valid VQLS solver");

        // Try a few seeds; gradient descent should reach high fidelity.
        let mut best_fid = 0.0_f64;
        for seed in 0..6 {
            let init = random_params(solver.ansatz.n_params(), 100 + seed);
            let res = solver
                .optimize(&init, 400, 0.5, 1e-3, 1e-8)
                .expect("optimization should succeed");
            let fid = fidelity(&res.solution, &target);
            best_fid = best_fid.max(fid);
            if best_fid > 0.95 {
                break;
            }
        }
        assert!(best_fid > 0.95, "best fidelity {best_fid} ≤ 0.95");
    }

    // (b) The cost decreases over iterations and reaches near 0.
    #[test]
    fn cost_decreases_and_reaches_zero() {
        let terms = vec![
            PauliTerm {
                coeff: 0.8,
                paulis: vec![Pauli::I],
            },
            PauliTerm {
                coeff: 0.6,
                paulis: vec![Pauli::Z],
            },
        ];
        let op = LcuOperator::new(&terms, 1).expect("valid 1-qubit LCU operator from I+Z terms");
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let b = vec![c(inv_sqrt2), c(inv_sqrt2)];
        let ansatz = HardwareEfficientAnsatz::new(1, 2).expect("valid 1-qubit 2-layer ansatz");
        let solver = VqlsSolver::new(op, &b, ansatz).expect("valid VQLS solver");

        // Find a run that converges low and verify monotone-ish non-increase.
        let mut converged = None;
        for seed in 0..8 {
            let init = random_params(solver.ansatz.n_params(), 7 + seed);
            let res = solver
                .optimize(&init, 400, 0.5, 1e-3, 1e-8)
                .expect("optimization should succeed");
            if res.final_cost < 1e-3 {
                converged = Some(res);
                break;
            }
        }
        let res = converged.expect("at least one seed should converge near 0");
        // Cost history is non-increasing (backtracking only accepts improvements).
        for w in res.cost_history.windows(2) {
            assert!(w[1] <= w[0] + 1e-9, "cost increased: {} → {}", w[0], w[1]);
        }
        assert!(
            res.final_cost < 1e-3,
            "final cost {} not near 0",
            res.final_cost
        );
        // First entry should be substantially larger than the last (real progress).
        assert!(
            res.cost_history[0] > res.final_cost,
            "no progress: {} → {}",
            res.cost_history[0],
            res.final_cost
        );
    }

    // (d) A = I → optimal |x⟩ = |b⟩.
    #[test]
    fn identity_solution_is_b() {
        // A = I on 1 qubit.
        let terms = vec![PauliTerm {
            coeff: 1.0,
            paulis: vec![Pauli::I],
        }];
        let op = LcuOperator::new(&terms, 1).expect("valid 1-qubit identity LCU operator");
        // b = R_y target: choose b = (cos(0.3), sin(0.3)) (real, reachable by 1
        // R_y layer). With A = I, A⁻¹|b⟩ = |b⟩.
        let b = vec![c(0.3_f32.cos()), c(0.3_f32.sin())];
        let target = op
            .classical_solution(&b)
            .expect("classical solution should exist for identity A");
        let ansatz = HardwareEfficientAnsatz::new(1, 1).expect("valid 1-qubit 1-layer ansatz");
        let solver = VqlsSolver::new(op, &b, ansatz).expect("valid VQLS solver");

        let mut best_fid = 0.0_f64;
        for seed in 0..6 {
            let init = random_params(solver.ansatz.n_params(), 200 + seed);
            let res = solver
                .optimize(&init, 400, 0.5, 1e-3, 1e-9)
                .expect("optimization should succeed");
            best_fid = best_fid.max(fidelity(&res.solution, &target));
            if best_fid > 0.98 {
                break;
            }
        }
        // Target equals normalised |b⟩.
        let bnorm: f32 = b.iter().map(|z| z.norm_sqr()).sum::<f32>().sqrt();
        let b_normed: Vec<Complex32> = b.iter().map(|z| *z / bnorm).collect();
        assert!(
            fidelity(&target, &b_normed) > 0.999,
            "target should equal |b⟩ for A=I"
        );
        assert!(best_fid > 0.98, "fidelity to |b⟩={best_fid}");
    }

    // 2-qubit end-to-end smoke: assemble A, optimise, and reach a low cost.
    #[test]
    fn two_qubit_solver_converges() {
        // A = 1.0·I⊗I + 0.3·Z⊗I + 0.2·I⊗Z (diagonal, strictly diagonally
        // dominant ⇒ invertible). b = uniform.
        let terms = vec![
            PauliTerm {
                coeff: 1.0,
                paulis: vec![Pauli::I, Pauli::I],
            },
            PauliTerm {
                coeff: 0.3,
                paulis: vec![Pauli::Z, Pauli::I],
            },
            PauliTerm {
                coeff: 0.2,
                paulis: vec![Pauli::I, Pauli::Z],
            },
        ];
        let op = LcuOperator::new(&terms, 2).expect("valid 2-qubit diagonal LCU operator");
        let half = 0.5_f32;
        let b = vec![c(half), c(half), c(half), c(half)];
        let target = op
            .classical_solution(&b)
            .expect("classical solution should exist for diagonally dominant A");
        let ansatz = HardwareEfficientAnsatz::new(2, 3).expect("valid 2-qubit 3-layer ansatz");
        let solver = VqlsSolver::new(op, &b, ansatz).expect("valid VQLS solver");

        let mut best_fid = 0.0_f64;
        let mut best_cost = 1.0_f64;
        for seed in 0..10 {
            let init = random_params(solver.ansatz.n_params(), 300 + seed);
            let res = solver
                .optimize(&init, 500, 0.4, 1e-3, 1e-8)
                .expect("optimization should succeed");
            best_cost = best_cost.min(res.final_cost);
            best_fid = best_fid.max(fidelity(&res.solution, &target));
            if best_fid > 0.95 {
                break;
            }
        }
        assert!(best_cost < 1e-2, "2-qubit best cost {best_cost}");
        assert!(best_fid > 0.95, "2-qubit best fidelity {best_fid}");
    }

    // Validation: dimension and emptiness checks.
    #[test]
    fn validation_errors() {
        // Empty term list.
        assert!(LcuOperator::new(&[], 1).is_err());
        // Wrong Pauli-string length.
        let bad = vec![PauliTerm {
            coeff: 1.0,
            paulis: vec![Pauli::X, Pauli::X],
        }];
        assert!(LcuOperator::new(&bad, 1).is_err());
        // Ansatz with zero layers.
        assert!(HardwareEfficientAnsatz::new(2, 0).is_err());
        // Mismatched b length.
        let op = LcuOperator::new(
            &[PauliTerm {
                coeff: 1.0,
                paulis: vec![Pauli::I],
            }],
            1,
        )
        .expect("valid single-qubit identity LCU operator");
        let ansatz = HardwareEfficientAnsatz::new(1, 1).expect("valid 1-qubit 1-layer ansatz");
        assert!(VqlsSolver::new(op, &[c(1.0), c(0.0), c(0.0)], ansatz).is_err());
        // Zero b.
        let op2 = LcuOperator::new(
            &[PauliTerm {
                coeff: 1.0,
                paulis: vec![Pauli::I],
            }],
            1,
        )
        .expect("valid single-qubit identity LCU operator for zero-b test");
        let ansatz2 = HardwareEfficientAnsatz::new(1, 1).expect("valid 1-qubit 1-layer ansatz");
        assert!(VqlsSolver::new(op2, &[c(0.0), c(0.0)], ansatz2).is_err());
    }
}
