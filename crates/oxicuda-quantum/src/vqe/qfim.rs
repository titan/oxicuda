//! Natural-gradient VQE via the quantum Fisher information matrix.
//!
//! Implements the quantum natural gradient of Stokes, Izaac, Killoran & Carleo
//! 2020 ("Quantum Natural Gradient"). The Euclidean parameter gradient is
//! preconditioned by the inverse of the Fubini–Study metric tensor (the
//! quantum geometric tensor's real part), which adapts the update to the
//! curvature of the variational state manifold and typically accelerates VQE
//! convergence over vanilla gradient descent.
//!
//! ## Conventions
//! The quantum geometric tensor (QGT) is
//! `G_ij = ⟨∂_iψ|∂_jψ⟩ − ⟨∂_iψ|ψ⟩⟨ψ|∂_jψ⟩`.
//! The metric used to precondition the gradient is the real part,
//! `F = Re(G)`. (Some references multiply by 4; we deliberately adopt the
//! `Re(G)` convention so that the well-conditioned limit `F ≈ 0` recovers
//! `δ ≈ grad / reg`.) Derivatives `|∂_iψ⟩` are estimated with a central finite
//! difference of the statevector with respect to each parameter.

use num_complex::Complex;

use crate::error::{QuantumError, QuantumResult};
use crate::handle::LcgRng;
use crate::statevec::state::StateVector;
use crate::vqe::ansatz::HardwareEfficientAnsatz;

type Complex32 = Complex<f32>;

/// Hyperparameters for one quantum-natural-gradient update.
#[derive(Debug, Clone)]
pub struct QngConfig {
    /// Learning rate (step size) applied to the preconditioned gradient.
    pub lr: f32,
    /// Finite-difference step `ε` used to estimate state derivatives.
    pub finite_diff_eps: f32,
    /// Tikhonov regularization added to the metric diagonal (`F + reg·I`),
    /// guarding against ill-conditioning / singularity.
    pub regularization: f32,
}

impl Default for QngConfig {
    fn default() -> Self {
        Self {
            lr: 0.1,
            finite_diff_eps: 1e-2,
            regularization: 1e-3,
        }
    }
}

/// Quantum natural-gradient engine wrapping a hardware-efficient ansatz.
#[derive(Debug, Clone)]
pub struct QuantumNaturalGradient {
    ansatz: HardwareEfficientAnsatz,
    n_qubits: usize,
}

impl QuantumNaturalGradient {
    /// Construct the engine from an ansatz.
    #[must_use]
    pub fn new(ansatz: HardwareEfficientAnsatz) -> Self {
        let n_qubits = ansatz.n_qubits;
        Self { ansatz, n_qubits }
    }

    /// Number of variational parameters of the wrapped ansatz.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.ansatz.n_params()
    }

    /// Number of qubits.
    #[must_use]
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Build the ansatz circuit at `params` and simulate it on |0…0⟩,
    /// returning the resulting statevector amplitudes.
    pub fn statevector(&self, params: &[f32]) -> QuantumResult<Vec<Complex32>> {
        if params.len() != self.ansatz.n_params() {
            return Err(QuantumError::DimensionMismatch {
                expected: self.ansatz.n_params(),
                got: params.len(),
            });
        }
        let circ = self.ansatz.build_circuit(params)?;
        let init = StateVector::new_zero_state(self.n_qubits)?;
        // The ansatz contains no measurements, so the RNG is unused; a fixed
        // seed keeps the evaluation deterministic.
        let mut rng = LcgRng::new(0);
        let sv = circ.exec_on_state(&init, &mut rng)?;
        Ok(sv.amps)
    }

    /// Central finite-difference derivative `|∂_iψ⟩` of the statevector with
    /// respect to parameter `i`: `(ψ(θ + ε e_i) − ψ(θ − ε e_i)) / (2ε)`.
    fn derivative(&self, params: &[f32], i: usize, eps: f32) -> QuantumResult<Vec<Complex32>> {
        let mut p_plus = params.to_vec();
        let mut p_minus = params.to_vec();
        p_plus[i] += eps;
        p_minus[i] -= eps;
        let psi_plus = self.statevector(&p_plus)?;
        let psi_minus = self.statevector(&p_minus)?;
        let inv = 1.0 / (2.0 * eps);
        let deriv = psi_plus
            .iter()
            .zip(psi_minus.iter())
            .map(|(a, b)| (a - b) * inv)
            .collect();
        Ok(deriv)
    }

    /// Inner product `⟨a|b⟩ = Σ conj(a_k) · b_k` over equal-length amplitude
    /// vectors.
    #[inline]
    fn braket(a: &[Complex32], b: &[Complex32]) -> Complex32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x.conj() * y)
            .fold(Complex32::new(0.0, 0.0), |acc, v| acc + v)
    }

    /// Fubini–Study metric `F = Re(G)`, returned as a row-major
    /// `n_params × n_params` matrix.
    ///
    /// `F_ij = Re(⟨∂_iψ|∂_jψ⟩) − Re(⟨∂_iψ|ψ⟩·⟨ψ|∂_jψ⟩)`, with
    /// `⟨ψ|∂_jψ⟩ = conj(⟨∂_jψ|ψ⟩)`.
    pub fn qfim(&self, params: &[f32], eps: f32) -> QuantumResult<Vec<f32>> {
        if params.len() != self.ansatz.n_params() {
            return Err(QuantumError::DimensionMismatch {
                expected: self.ansatz.n_params(),
                got: params.len(),
            });
        }
        if eps <= 0.0 {
            return Err(QuantumError::InvalidParameter {
                name: "eps must be positive".into(),
            });
        }
        let n = self.ansatz.n_params();
        let psi = self.statevector(params)?;

        // Precompute all derivative vectors and the ⟨∂_iψ|ψ⟩ overlaps.
        let mut derivs: Vec<Vec<Complex32>> = Vec::with_capacity(n);
        let mut d_psi: Vec<Complex32> = Vec::with_capacity(n);
        for i in 0..n {
            let di = self.derivative(params, i, eps)?;
            d_psi.push(Self::braket(&di, &psi));
            derivs.push(di);
        }

        let mut f = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                // ⟨∂_iψ|∂_jψ⟩
                let term1 = Self::braket(&derivs[i], &derivs[j]);
                // ⟨∂_iψ|ψ⟩ · ⟨ψ|∂_jψ⟩ = d_psi[i] · conj(d_psi[j])
                let term2 = d_psi[i] * d_psi[j].conj();
                f[i * n + j] = (term1 - term2).re;
            }
        }
        Ok(f)
    }

    /// One natural-gradient update: solve `(F + reg·I) δ = grad` and return
    /// `params − lr·δ`.
    pub fn natural_gradient_step(
        &self,
        params: &[f32],
        grad: &[f32],
        cfg: &QngConfig,
    ) -> QuantumResult<Vec<f32>> {
        let n = self.ansatz.n_params();
        if params.len() != n {
            return Err(QuantumError::DimensionMismatch {
                expected: n,
                got: params.len(),
            });
        }
        if grad.len() != n {
            return Err(QuantumError::DimensionMismatch {
                expected: n,
                got: grad.len(),
            });
        }
        if cfg.finite_diff_eps <= 0.0 {
            return Err(QuantumError::InvalidParameter {
                name: "finite_diff_eps must be positive".into(),
            });
        }
        if cfg.regularization < 0.0 {
            return Err(QuantumError::InvalidParameter {
                name: "regularization must be non-negative".into(),
            });
        }

        let mut a = self.qfim(params, cfg.finite_diff_eps)?;
        // Tikhonov regularization on the diagonal: A = F + reg·I.
        for i in 0..n {
            a[i * n + i] += cfg.regularization;
        }

        let delta = solve_linear_system(&a, grad, n)?;

        let next = params
            .iter()
            .zip(delta.iter())
            .map(|(p, d)| p - cfg.lr * d)
            .collect();
        Ok(next)
    }
}

/// Solve `A x = b` for a dense `n × n` row-major matrix `A` via Gaussian
/// elimination with partial pivoting. Returns an error if `A` is singular.
fn solve_linear_system(a: &[f32], b: &[f32], n: usize) -> QuantumResult<Vec<f32>> {
    if n == 0 {
        return Err(QuantumError::EmptyInput);
    }
    // Work on local copies so the caller's data is untouched.
    let mut m = a.to_vec();
    let mut rhs = b.to_vec();

    for col in 0..n {
        // Partial pivot: largest magnitude in this column at or below the diagonal.
        let mut pivot_row = col;
        let mut pivot_mag = m[col * n + col].abs();
        for row in (col + 1)..n {
            let mag = m[row * n + col].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = row;
            }
        }
        if pivot_mag < 1e-12 {
            return Err(QuantumError::Internal {
                msg: "singular metric matrix in natural-gradient solve".into(),
            });
        }
        if pivot_row != col {
            for k in 0..n {
                m.swap(col * n + k, pivot_row * n + k);
            }
            rhs.swap(col, pivot_row);
        }

        // Eliminate the column below the pivot.
        let pivot = m[col * n + col];
        for row in (col + 1)..n {
            let factor = m[row * n + col] / pivot;
            if factor != 0.0 {
                for k in col..n {
                    let sub = factor * m[col * n + k];
                    m[row * n + k] -= sub;
                }
                rhs[row] -= factor * rhs[col];
            }
        }
    }

    // Back-substitution.
    let mut x = vec![0.0_f32; n];
    for row in (0..n).rev() {
        let mut acc = rhs[row];
        for k in (row + 1)..n {
            acc -= m[row * n + k] * x[k];
        }
        x[row] = acc / m[row * n + row];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine(n_qubits: usize, depth: usize) -> QuantumNaturalGradient {
        QuantumNaturalGradient::new(HardwareEfficientAnsatz::new(n_qubits, depth))
    }

    fn random_params(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..n)
            .map(|_| (rng.next_u32() as f32 / (u32::MAX as f32 + 1.0)) * 2.0 - 1.0)
            .collect()
    }

    #[test]
    fn statevector_length_and_norm() {
        let qng = engine(3, 2);
        let params = random_params(qng.n_params(), 1);
        let sv = qng
            .statevector(&params)
            .expect("params has exactly n_params() elements for the 3-qubit 2-depth ansatz");
        assert_eq!(sv.len(), 1 << 3);
        let norm: f32 = sv.iter().map(num_complex::Complex::norm_sqr).sum();
        assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
    }

    #[test]
    fn qfim_length_is_n_params_squared() {
        let qng = engine(2, 2);
        let n = qng.n_params();
        let params = random_params(n, 2);
        let f = qng
            .qfim(&params, 1e-2)
            .expect("params has n_params() elements and eps=1e-2 is positive");
        assert_eq!(f.len(), n * n);
    }

    #[test]
    fn qfim_symmetric() {
        let qng = engine(3, 1);
        let n = qng.n_params();
        let params = random_params(n, 3);
        let f = qng.qfim(&params, 1e-2).expect("params has n_params() elements for the 3-qubit 1-depth ansatz and eps=1e-2 is positive");
        for i in 0..n {
            for j in 0..n {
                let diff = (f[i * n + j] - f[j * n + i]).abs();
                assert!(diff < 1e-3, "F not symmetric at ({i},{j}): {diff}");
            }
        }
    }

    #[test]
    fn qfim_positive_semidefinite() {
        let qng = engine(3, 2);
        let n = qng.n_params();
        let params = random_params(n, 4);
        let f = qng.qfim(&params, 1e-2).expect("params has n_params() elements for the 3-qubit 2-depth ansatz and eps=1e-2 is positive");
        for vseed in 0..8u64 {
            let v = random_params(n, 100 + vseed);
            let mut quad = 0.0_f32;
            for i in 0..n {
                for j in 0..n {
                    quad += v[i] * f[i * n + j] * v[j];
                }
            }
            assert!(quad >= -1e-3, "vᵀFv negative: {quad} (vseed={vseed})");
        }
    }

    #[test]
    fn natural_gradient_step_output_length() {
        let qng = engine(2, 1);
        let n = qng.n_params();
        let params = random_params(n, 5);
        let grad = random_params(n, 6);
        let cfg = QngConfig::default();
        let next = qng.natural_gradient_step(&params, &grad, &cfg).expect("params and grad both have n_params() elements and default cfg has positive eps and non-negative reg");
        assert_eq!(next.len(), n);
    }

    #[test]
    fn natural_gradient_step_well_conditioned_runs() {
        let qng = engine(3, 2);
        let n = qng.n_params();
        let params = random_params(n, 7);
        let grad = random_params(n, 8);
        let cfg = QngConfig {
            lr: 0.05,
            finite_diff_eps: 1e-2,
            regularization: 1e-2,
        };
        let next = qng.natural_gradient_step(&params, &grad, &cfg).expect("params and grad both have n_params() elements and the well-conditioned cfg has positive eps and non-negative reg");
        for (i, v) in next.iter().enumerate() {
            assert!(v.is_finite(), "param {i} not finite: {v}");
        }
    }

    #[test]
    fn large_reg_recovers_scaled_gradient() {
        // When reg dominates the (O(1)) metric entries, (F + reg·I) ≈ reg·I, so
        // δ ≈ grad/reg and hence next ≈ params − lr·grad/reg.
        let qng = engine(2, 1);
        let n = qng.n_params();
        let params = vec![0.3_f32; n];
        let grad: Vec<f32> = (0..n).map(|k| 0.1 * (k as f32 + 1.0)).collect();
        let reg = 1.0e4_f32;
        let lr = 0.1_f32;
        let cfg = QngConfig {
            lr,
            finite_diff_eps: 1e-2,
            regularization: reg,
        };
        let next = qng.natural_gradient_step(&params, &grad, &cfg).expect("params and grad have n_params() elements and large regularization ensures the metric is non-singular");
        for k in 0..n {
            let expected = params[k] - lr * grad[k] / reg;
            assert!(
                (next[k] - expected).abs() < 1e-4,
                "k={k}: got {} expected {expected}",
                next[k]
            );
        }
    }

    #[test]
    fn deterministic_same_params() {
        let qng = engine(3, 1);
        let n = qng.n_params();
        let params = random_params(n, 9);
        let f1 = qng
            .qfim(&params, 1e-2)
            .expect("params has n_params() elements and eps=1e-2 is positive");
        let f2 = qng
            .qfim(&params, 1e-2)
            .expect("same params and positive eps, deterministic evaluation must succeed");
        assert_eq!(f1, f2);
        let sv1 = qng
            .statevector(&params)
            .expect("params has exactly n_params() elements");
        let sv2 = qng
            .statevector(&params)
            .expect("same params with correct length, statevector evaluation must succeed");
        assert_eq!(sv1, sv2);
    }

    #[test]
    fn statevector_wrong_param_len_errors() {
        let qng = engine(2, 1);
        let bad = vec![0.0_f32; qng.n_params() + 1];
        assert!(qng.statevector(&bad).is_err());
    }

    #[test]
    fn qfim_wrong_param_len_errors() {
        let qng = engine(2, 1);
        let bad = vec![0.0_f32; qng.n_params() + 2];
        assert!(qng.qfim(&bad, 1e-2).is_err());
    }

    #[test]
    fn qfim_non_positive_eps_errors() {
        let qng = engine(2, 1);
        let params = random_params(qng.n_params(), 10);
        assert!(qng.qfim(&params, 0.0).is_err());
        assert!(qng.qfim(&params, -1e-2).is_err());
    }

    #[test]
    fn step_wrong_param_len_errors() {
        let qng = engine(2, 1);
        let n = qng.n_params();
        let bad_params = vec![0.0_f32; n + 1];
        let grad = vec![0.0_f32; n];
        assert!(
            qng.natural_gradient_step(&bad_params, &grad, &QngConfig::default())
                .is_err()
        );
    }

    #[test]
    fn step_wrong_grad_len_errors() {
        let qng = engine(2, 1);
        let n = qng.n_params();
        let params = vec![0.0_f32; n];
        let bad_grad = vec![0.0_f32; n + 3];
        assert!(
            qng.natural_gradient_step(&params, &bad_grad, &QngConfig::default())
                .is_err()
        );
    }

    #[test]
    fn step_negative_reg_errors() {
        let qng = engine(2, 1);
        let n = qng.n_params();
        let params = vec![0.0_f32; n];
        let grad = vec![0.0_f32; n];
        let cfg = QngConfig {
            lr: 0.1,
            finite_diff_eps: 1e-2,
            regularization: -1.0,
        };
        assert!(qng.natural_gradient_step(&params, &grad, &cfg).is_err());
    }

    #[test]
    fn step_non_positive_eps_errors() {
        let qng = engine(2, 1);
        let n = qng.n_params();
        let params = vec![0.0_f32; n];
        let grad = vec![0.0_f32; n];
        let cfg = QngConfig {
            lr: 0.1,
            finite_diff_eps: 0.0,
            regularization: 1e-3,
        };
        assert!(qng.natural_gradient_step(&params, &grad, &cfg).is_err());
    }

    #[test]
    fn one_qubit_ansatz_statevector_sane() {
        let qng = engine(1, 1);
        let n = qng.n_params();
        // RY(0) on every layer leaves |0⟩ unchanged.
        let params = vec![0.0_f32; n];
        let sv = qng
            .statevector(&params)
            .expect("all-zeros params for a 1-qubit 1-depth ansatz has the right length");
        assert_eq!(sv.len(), 2);
        assert!((sv[0].re - 1.0).abs() < 1e-5, "amp0={:?}", sv[0]);
        assert!(sv[1].norm() < 1e-5, "amp1={:?}", sv[1]);
    }

    #[test]
    fn qfim_tiny_ansatz_finite() {
        let qng = engine(1, 0);
        let n = qng.n_params();
        let params = random_params(n, 11);
        let f = qng.qfim(&params, 1e-2).expect("params has n_params() elements for the 1-qubit 0-depth ansatz and eps=1e-2 is positive");
        for (k, v) in f.iter().enumerate() {
            assert!(v.is_finite(), "F[{k}] not finite: {v}");
        }
    }

    #[test]
    fn natural_gradient_aligns_with_grad_on_diagonal_metric() {
        // With a positive-definite + reg metric, the solve of A δ = grad yields
        // a δ whose dot product with grad is positive (descent on grad sign).
        let qng = engine(2, 1);
        let n = qng.n_params();
        let params = random_params(n, 12);
        let grad = random_params(n, 13);
        let cfg = QngConfig {
            lr: 0.1,
            finite_diff_eps: 1e-2,
            regularization: 0.5,
        };
        let next = qng.natural_gradient_step(&params, &grad, &cfg).expect("params and grad have n_params() elements and reg=0.5 ensures a well-conditioned metric");
        // step direction = next - params = -lr·δ; ⟨grad, δ⟩ should be ≥ 0 for a PSD A.
        let mut dot = 0.0_f32;
        for k in 0..n {
            let delta_k = (params[k] - next[k]) / cfg.lr; // = δ_k
            dot += grad[k] * delta_k;
        }
        assert!(dot >= -1e-3, "⟨grad, δ⟩ negative for PSD metric: {dot}");
    }

    #[test]
    fn changing_params_changes_qfim() {
        let qng = engine(2, 2);
        let n = qng.n_params();
        let p1 = random_params(n, 14);
        let mut p2 = p1.clone();
        p2[0] += 0.7;
        p2[1] -= 0.5;
        let f1 = qng.qfim(&p1, 1e-2).expect(
            "p1 has n_params() elements for the 2-qubit 2-depth ansatz and eps=1e-2 is positive",
        );
        let f2 = qng
            .qfim(&p2, 1e-2)
            .expect("p2 has n_params() elements (modified from p1) and eps=1e-2 is positive");
        let max_diff = f1
            .iter()
            .zip(f2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_diff > 1e-4, "QFIM did not change: max_diff={max_diff}");
    }

    #[test]
    fn solve_linear_system_identity() {
        // Sanity check of the internal solver on the identity matrix.
        let n = 3;
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![2.0, -3.0, 5.0];
        let x = solve_linear_system(&a, &b, n).expect(
            "3x3 identity matrix is non-singular so the linear system is uniquely solvable",
        );
        for k in 0..n {
            assert!((x[k] - b[k]).abs() < 1e-6, "x[{k}]={}", x[k]);
        }
    }

    #[test]
    fn solve_linear_system_singular_errors() {
        // A row of zeros makes the system singular.
        let n = 2;
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 1.0];
        assert!(solve_linear_system(&a, &b, n).is_err());
    }

    #[test]
    fn n_params_and_n_qubits_getters() {
        let qng = engine(4, 3);
        assert_eq!(qng.n_qubits(), 4);
        assert_eq!(qng.n_params(), (3 + 1) * 4);
    }
}
