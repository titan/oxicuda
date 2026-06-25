//! Quantum generative model (qGAN-style distribution loader).
//!
//! Reference: Zoufal, Lucchi & Woerner, "Quantum Generative Adversarial Networks
//! for learning and loading random distributions", npj Quantum Inf. 5, 103
//! (2019); Lloyd & Weedbrook, PRL 121, 040502 (2018).
//!
//! A parametrized quantum circuit `G(θ)` prepares a state `|ψ_θ⟩`; measuring it
//! in the computational basis yields a probability distribution
//! `p_θ(x) = |⟨x|ψ_θ⟩|²` over the `2^n` integers `x`. The **generator** is
//! trained so that `p_θ` approximates a target distribution `q(x)` — the central
//! task in quantum GANs for loading classical distributions into a quantum
//! register (e.g. for amplitude-estimation finance pipelines).
//!
//! Rather than an explicit adversarial discriminator network (which would itself
//! need a second optimization loop), we train against the **maximum mean
//! discrepancy** (MMD) with a Gaussian mixture kernel — the loss used in
//! quantum-circuit Born machines (Liu & Wang, PRA 98, 062324 (2018)). The MMD is
//! a proper integral probability metric: `MMD² = 0 ⟺ p_θ = q`, it is smooth and
//! differentiable, and its gradient admits the parameter-shift rule, giving a
//! self-contained, sampling-free generative trainer. The generator ansatz is the
//! hardware-efficient `Ry`/`Rz` + CNOT layered circuit.

use crate::error::{QuantumError, QuantumResult};
use crate::gates::controlled::apply_cnot;
use crate::gates::parametric::{gate_ry, gate_rz};
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;

/// A trainable quantum generator over `2^n` outcomes.
#[derive(Debug, Clone)]
pub struct QuantumGenerator {
    n_qubits: usize,
    depth: usize,
    /// Bandwidths of the Gaussian mixture MMD kernel (over integer outcomes).
    sigmas: Vec<f32>,
}

impl QuantumGenerator {
    /// Construct a generator with `depth` hardware-efficient layers.
    ///
    /// # Errors
    /// Returns [`QuantumError::InvalidQubitCount`] for a qubit count outside
    /// `[1, 12]`.
    pub fn new(n_qubits: usize, depth: usize) -> QuantumResult<Self> {
        if n_qubits == 0 || n_qubits > 12 {
            return Err(QuantumError::InvalidQubitCount { n: n_qubits });
        }
        Ok(Self {
            n_qubits,
            depth,
            sigmas: vec![0.5, 1.0, 2.0, 4.0],
        })
    }

    /// Number of qubits.
    #[must_use]
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Dimension `2^n` of the outcome space.
    #[must_use]
    pub fn dim(&self) -> usize {
        1usize << self.n_qubits
    }

    /// Number of trainable parameters: `(depth + 1)` layers × `2` rotations × `n`.
    #[must_use]
    pub fn n_params(&self) -> usize {
        (self.depth + 1) * 2 * self.n_qubits
    }

    /// Build the generator state `|ψ_θ⟩`.
    fn state(&self, params: &[f32]) -> QuantumResult<StateVector> {
        if params.len() != self.n_params() {
            return Err(QuantumError::DimensionMismatch {
                expected: self.n_params(),
                got: params.len(),
            });
        }
        let n = self.n_qubits;
        let mut sv = StateVector::new_zero_state(n)?;
        let mut idx = 0usize;
        for layer in 0..=self.depth {
            for q in 0..n {
                apply_1q_inplace(&mut sv, q, &gate_ry(params[idx]))?;
                idx += 1;
                apply_1q_inplace(&mut sv, q, &gate_rz(params[idx]))?;
                idx += 1;
            }
            if layer < self.depth {
                for q in 0..n.saturating_sub(1) {
                    apply_cnot(&mut sv, q, q + 1)?;
                }
            }
        }
        Ok(sv)
    }

    /// Exact output distribution `p_θ(x) = |⟨x|ψ_θ⟩|²`.
    ///
    /// # Errors
    /// Propagates state-construction errors.
    pub fn distribution(&self, params: &[f32]) -> QuantumResult<Vec<f32>> {
        let sv = self.state(params)?;
        Ok(sv.amps.iter().map(|a| a.norm_sqr()).collect())
    }

    /// Gaussian mixture kernel `k(i, j)` over integer outcomes `i, j`.
    fn mmd_kernel(&self, i: usize, j: usize) -> f32 {
        let d = i as f32 - j as f32;
        let d2 = d * d;
        let mut acc = 0.0_f32;
        for &s in &self.sigmas {
            acc += (-d2 / (2.0 * s * s)).exp();
        }
        acc / self.sigmas.len() as f32
    }

    /// Squared MMD between distribution `p` and target `q`:
    /// `Σ_{ij} (p_i - q_i) k(i,j) (p_j - q_j)`.
    fn mmd_sq(&self, p: &[f32], q: &[f32]) -> f32 {
        let dim = self.dim();
        let mut acc = 0.0_f32;
        for i in 0..dim {
            let di = p[i] - q[i];
            if di.abs() < 1e-12 {
                continue;
            }
            for j in 0..dim {
                let dj = p[j] - q[j];
                acc += di * self.mmd_kernel(i, j) * dj;
            }
        }
        acc.max(0.0)
    }

    /// MMD² loss between the generator's distribution at `params` and `target`.
    ///
    /// # Errors
    /// Returns [`QuantumError::DimensionMismatch`] if `target.len() != 2^n`;
    /// propagates state errors.
    pub fn loss(&self, params: &[f32], target: &[f32]) -> QuantumResult<f32> {
        if target.len() != self.dim() {
            return Err(QuantumError::DimensionMismatch {
                expected: self.dim(),
                got: target.len(),
            });
        }
        let p = self.distribution(params)?;
        Ok(self.mmd_sq(&p, target))
    }

    /// Parameter-shift gradient of the MMD² loss.
    ///
    /// # Errors
    /// Propagates loss-evaluation errors.
    pub fn gradient(&self, params: &[f32], target: &[f32]) -> QuantumResult<Vec<f32>> {
        let shift = std::f32::consts::FRAC_PI_2;
        let mut grad = vec![0.0_f32; params.len()];
        for p in 0..params.len() {
            let mut pp = params.to_vec();
            let mut pm = params.to_vec();
            pp[p] += shift;
            pm[p] -= shift;
            let lp = self.loss(&pp, target)?;
            let lm = self.loss(&pm, target)?;
            grad[p] = 0.5 * (lp - lm);
        }
        Ok(grad)
    }

    /// Train the generator to match `target` by gradient descent on MMD².
    ///
    /// Returns the loss at every iteration (length `iters + 1`).
    ///
    /// # Errors
    /// Returns [`QuantumError::DimensionMismatch`] for a wrong-size target or
    /// parameter vector; propagates state errors.
    pub fn train(
        &self,
        params: &mut [f32],
        target: &[f32],
        iters: usize,
        lr: f32,
    ) -> QuantumResult<Vec<f32>> {
        if params.len() != self.n_params() {
            return Err(QuantumError::DimensionMismatch {
                expected: self.n_params(),
                got: params.len(),
            });
        }
        if target.len() != self.dim() {
            return Err(QuantumError::DimensionMismatch {
                expected: self.dim(),
                got: target.len(),
            });
        }
        let mut history = Vec::with_capacity(iters + 1);
        history.push(self.loss(params, target)?);
        for _ in 0..iters {
            let grad = self.gradient(params, target)?;
            for (p, g) in params.iter_mut().zip(grad.iter()) {
                *p -= lr * g;
            }
            history.push(self.loss(params, target)?);
        }
        Ok(history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn rejects_bad_qubit_count() {
        assert!(QuantumGenerator::new(0, 1).is_err());
        assert!(QuantumGenerator::new(13, 1).is_err());
        assert!(QuantumGenerator::new(3, 2).is_ok());
    }

    #[test]
    fn distribution_is_normalized() {
        let qgen = QuantumGenerator::new(3, 2).expect("valid");
        let mut rng = LcgRng::new(7);
        let params: Vec<f32> = (0..qgen.n_params()).map(|_| rng.next_normal()).collect();
        let p = qgen.distribution(&params).expect("dist");
        let total: f32 = p.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "total={total}");
        assert!(p.iter().all(|&x| x >= -1e-7));
    }

    #[test]
    fn mmd_zero_for_identical_distributions() {
        let qgen = QuantumGenerator::new(2, 1).expect("valid");
        let params = vec![0.0_f32; qgen.n_params()];
        let p = qgen.distribution(&params).expect("dist");
        // Loss against itself must be ~0.
        let l = qgen.loss(&params, &p).expect("loss");
        assert!(l.abs() < 1e-5, "self-MMD={l}");
    }

    #[test]
    fn loss_nonnegative_and_gradient_finite() {
        let qgen = QuantumGenerator::new(3, 2).expect("valid");
        let mut rng = LcgRng::new(11);
        let params: Vec<f32> = (0..qgen.n_params())
            .map(|_| rng.next_normal() * 0.5)
            .collect();
        // Target: peaked at outcome 0.
        let mut target = vec![0.0_f32; qgen.dim()];
        target[0] = 1.0;
        let l = qgen.loss(&params, &target).expect("loss");
        assert!(l >= 0.0, "loss={l}");
        let g = qgen.gradient(&params, &target).expect("grad");
        assert_eq!(g.len(), qgen.n_params());
        assert!(g.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn training_matches_a_delta_target() {
        // Target: all mass on |0…0⟩. The generator (which starts near |0…0⟩ for
        // small params) should reduce its MMD loss toward 0.
        let qgen = QuantumGenerator::new(2, 2).expect("valid");
        let mut rng = LcgRng::new(2024);
        let mut params: Vec<f32> = (0..qgen.n_params())
            .map(|_| rng.next_normal() * 0.3)
            .collect();
        let mut target = vec![0.0_f32; qgen.dim()];
        target[0] = 1.0;
        let history = qgen.train(&mut params, &target, 40, 0.3).expect("train");
        let first = history[0];
        let last = *history.last().expect("history");
        assert!(last < first, "loss did not decrease: {first} → {last}");
        assert!(
            last < first * 0.8,
            "insufficient improvement: {first} → {last}"
        );
    }

    #[test]
    fn training_learns_bimodal_target() {
        // Target distribution with mass on outcomes 0 and 3 (a 2-qubit bimodal).
        let qgen = QuantumGenerator::new(2, 3).expect("valid");
        let mut rng = LcgRng::new(99);
        let mut params: Vec<f32> = (0..qgen.n_params())
            .map(|_| rng.next_normal() * 0.4)
            .collect();
        let target = vec![0.5_f32, 0.0, 0.0, 0.5];
        let history = qgen.train(&mut params, &target, 60, 0.25).expect("train");
        let first = history[0];
        let last = *history.last().expect("history");
        assert!(last < first, "loss did not decrease: {first} → {last}");
    }
}
