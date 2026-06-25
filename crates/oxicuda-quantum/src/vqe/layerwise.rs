//! Layer-wise warm-start initialization to mitigate barren plateaus.
//!
//! Reference: Grant, Wossnig, Ostaszewski & Benedetti, "An initialization
//! strategy for addressing barren plateaus in parametrized quantum circuits",
//! Quantum 3, 214 (2019); see also Skolik et al. 2021 on layer-wise learning.
//!
//! Deep hardware-efficient ansätze suffer from **barren plateaus**: the variance
//! of the cost gradient vanishes exponentially in the number of qubits, so a
//! randomly initialized deep circuit is essentially un-trainable. A practical
//! remedy is to **grow the circuit one layer at a time**:
//!
//! 1. Start from a shallow (depth-0) ansatz and optimize it to (near) convergence.
//! 2. Append one fresh layer whose rotation parameters are **zero-initialized**.
//!    Because `Ry(0) = I`, the appended layer is the identity at initialization,
//!    so the enlarged circuit reproduces the previously optimized state exactly —
//!    the optimizer therefore starts each stage already in a low-cost,
//!    high-gradient region rather than on a random plateau.
//! 3. Optimize the enlarged parameter set, then repeat until the target depth is
//!    reached.
//!
//! The optimizer below realizes this schedule on top of the existing
//! [`HardwareEfficientAnsatz`] and the parameter-shift energy/gradient already
//! provided by [`crate::vqe::vqe::VqeOptimizer`].

use crate::error::{QuantumError, QuantumResult};
use crate::handle::LcgRng;
use crate::pauli::hamiltonian::Hamiltonian;
use crate::vqe::ansatz::HardwareEfficientAnsatz;
use crate::vqe::vqe::VqeOptimizer;

/// Configuration for layer-wise warm-start VQE training.
#[derive(Debug, Clone)]
pub struct LayerwiseConfig {
    /// Number of qubits in the ansatz.
    pub n_qubits: usize,
    /// Final ansatz depth to grow to (number of entangling layers).
    pub target_depth: usize,
    /// Parameter-shift gradient-descent iterations spent at **each** depth stage.
    pub iters_per_stage: usize,
    /// Learning rate for gradient descent.
    pub lr: f32,
    /// Small std-dev for randomizing the very first (depth-0) layer; the grown
    /// layers are always zero-initialized to preserve the identity-block warm
    /// start, so only the seed layer needs symmetry breaking.
    pub seed_noise: f32,
}

impl LayerwiseConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns [`QuantumError::InvalidParameter`] / [`QuantumError::InvalidQubitCount`]
    /// for non-positive qubit counts, zero iterations, or a non-finite learning rate.
    pub fn new(
        n_qubits: usize,
        target_depth: usize,
        iters_per_stage: usize,
        lr: f32,
        seed_noise: f32,
    ) -> QuantumResult<Self> {
        if n_qubits == 0 {
            return Err(QuantumError::InvalidQubitCount { n: n_qubits });
        }
        if iters_per_stage == 0 {
            return Err(QuantumError::InvalidParameter {
                name: "iters_per_stage".into(),
            });
        }
        if !lr.is_finite() || lr <= 0.0 {
            return Err(QuantumError::InvalidParameter { name: "lr".into() });
        }
        Ok(Self {
            n_qubits,
            target_depth,
            iters_per_stage,
            lr,
            seed_noise,
        })
    }
}

/// Result of a layer-wise warm-start training run.
#[derive(Debug, Clone)]
pub struct LayerwiseResult {
    /// Final optimized parameters for the full target-depth ansatz.
    pub params: Vec<f32>,
    /// Final energy ⟨ψ(params)|H|ψ(params)⟩.
    pub energy: f32,
    /// Energy after each depth stage, `energies[d]` = energy at depth `d`.
    pub stage_energies: Vec<f32>,
}

/// Layer-wise warm-start VQE trainer.
#[derive(Debug, Clone)]
pub struct LayerwiseVqe {
    cfg: LayerwiseConfig,
    ham: Hamiltonian,
}

impl LayerwiseVqe {
    /// Construct the trainer from a validated config and a Hamiltonian.
    #[must_use]
    pub fn new(cfg: LayerwiseConfig, ham: Hamiltonian) -> Self {
        Self { cfg, ham }
    }

    /// Run the full layer-wise schedule from depth 0 to `target_depth`.
    ///
    /// At each stage `d`, the previous stage's optimized parameters are carried
    /// over verbatim and the freshly added layer's `n_qubits` rotation angles are
    /// appended as zeros (identity block), guaranteeing the stage starts at the
    /// previous optimum's energy. Gradient descent (parameter-shift) then refines
    /// the enlarged parameter vector.
    ///
    /// # Errors
    /// Propagates any error from energy/gradient evaluation.
    pub fn train(&self, rng: &mut LcgRng) -> QuantumResult<LayerwiseResult> {
        let mut stage_energies = Vec::with_capacity(self.cfg.target_depth + 1);

        // --- Stage 0: depth-0 ansatz with a small random seed. ---
        let ansatz0 = HardwareEfficientAnsatz::new(self.cfg.n_qubits, 0);
        let mut params: Vec<f32> = (0..ansatz0.n_params())
            .map(|_| rng.next_normal() * self.cfg.seed_noise)
            .collect();
        let mut opt = VqeOptimizer {
            ansatz: ansatz0,
            ham: self.ham.clone(),
            params: params.clone(),
        };
        let (e0, p0) = opt.optimize(self.cfg.iters_per_stage, self.cfg.lr)?;
        params = p0;
        stage_energies.push(e0);

        // --- Grow one entangling layer at a time. ---
        for depth in 1..=self.cfg.target_depth {
            let ansatz = HardwareEfficientAnsatz::new(self.cfg.n_qubits, depth);
            // Carry over previous params and append a zero (identity) RY layer.
            let mut grown = params.clone();
            grown.extend(std::iter::repeat_n(0.0_f32, self.cfg.n_qubits));
            debug_assert_eq!(grown.len(), ansatz.n_params());

            let mut stage_opt = VqeOptimizer {
                ansatz,
                ham: self.ham.clone(),
                params: grown,
            };
            let (e_stage, p_stage) = stage_opt.optimize(self.cfg.iters_per_stage, self.cfg.lr)?;
            params = p_stage;
            stage_energies.push(e_stage);
        }

        let final_energy = *stage_energies.last().unwrap_or(&f32::NAN);
        Ok(LayerwiseResult {
            params,
            energy: final_energy,
            stage_energies,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pauli::pauli_string::PauliOp;

    #[test]
    fn config_validation() {
        assert!(LayerwiseConfig::new(0, 2, 5, 0.1, 0.05).is_err());
        assert!(LayerwiseConfig::new(2, 2, 0, 0.1, 0.05).is_err());
        assert!(LayerwiseConfig::new(2, 2, 5, 0.0, 0.05).is_err());
        assert!(LayerwiseConfig::new(2, 2, 5, 0.1, 0.05).is_ok());
    }

    #[test]
    fn identity_block_warm_start_does_not_increase_energy_at_grow_boundary() {
        // The energy after growing should never be worse than before growing
        // (because the new layer starts as identity, then is optimized).
        let cfg = LayerwiseConfig::new(2, 3, 8, 0.15, 0.05).expect("valid cfg");
        let mut ham = Hamiltonian::new();
        ham.add_term(1.0, vec![PauliOp::Z, PauliOp::Z]);
        ham.add_term(-0.5, vec![PauliOp::X, PauliOp::I]);
        let trainer = LayerwiseVqe::new(cfg, ham);
        let mut rng = LcgRng::new(2024);
        let res = trainer.train(&mut rng).expect("training succeeds");

        assert_eq!(res.stage_energies.len(), 4); // depths 0,1,2,3
        // Monotone non-increasing (within numerical slack): deeper warm-started
        // ansatz can always match the shallower optimum.
        for w in res.stage_energies.windows(2) {
            assert!(
                w[1] <= w[0] + 5e-2,
                "stage energy increased: {} → {}",
                w[0],
                w[1]
            );
        }
        assert!(res.energy.is_finite());
    }

    #[test]
    fn final_params_have_target_depth_length() {
        let cfg = LayerwiseConfig::new(3, 2, 3, 0.1, 0.05).expect("valid cfg");
        let mut ham = Hamiltonian::new();
        ham.add_term(1.0, vec![PauliOp::Z, PauliOp::I, PauliOp::Z]);
        let trainer = LayerwiseVqe::new(cfg, ham);
        let mut rng = LcgRng::new(11);
        let res = trainer.train(&mut rng).expect("train");
        let expected = HardwareEfficientAnsatz::new(3, 2).n_params();
        assert_eq!(res.params.len(), expected);
    }

    #[test]
    fn ground_state_of_single_z_is_recovered() {
        // H = Z on 1 qubit, ground energy -1. Layer-wise VQE should approach it.
        let cfg = LayerwiseConfig::new(1, 2, 40, 0.3, 0.1).expect("valid cfg");
        let mut ham = Hamiltonian::new();
        ham.add_term(1.0, vec![PauliOp::Z]);
        let trainer = LayerwiseVqe::new(cfg, ham);
        let mut rng = LcgRng::new(5);
        let res = trainer.train(&mut rng).expect("train");
        assert!(
            res.energy < -0.9,
            "energy={} should approach -1",
            res.energy
        );
    }
}
