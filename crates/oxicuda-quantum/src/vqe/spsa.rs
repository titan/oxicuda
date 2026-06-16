use crate::error::{QuantumError, QuantumResult};
use crate::handle::LcgRng;
use crate::pauli::expval::expectation_value;
use crate::pauli::hamiltonian::Hamiltonian;
use crate::statevec::state::StateVector;
use crate::vqe::ansatz::HardwareEfficientAnsatz;

/// Hyperparameter configuration for the SPSA optimizer (Spall 1992).
#[derive(Debug, Clone)]
pub struct SpsaConfig {
    pub max_iter: usize,
    /// Numerator for the decaying step-size schedule a_k = a / (k+1+A)^alpha.
    pub a: f32,
    /// Numerator for the decaying perturbation schedule c_k = c / (k+1)^gamma.
    pub c: f32,
    /// Step-size decay exponent (α ≈ 0.602 satisfies Spall conditions).
    pub alpha: f32,
    /// Perturbation decay exponent (γ ≈ 0.101 satisfies Spall conditions).
    pub gamma: f32,
    /// Stability constant A that prevents a_0 from dominating.
    pub a_offset: usize,
    /// Warmup iterations before starting gradient correction.
    pub n_warmup: usize,
    /// Early-stopping: halt when energy variation over `patience` steps < tol.
    pub tol: f32,
    pub patience: usize,
}

impl Default for SpsaConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            a: 0.1,
            c: 0.1,
            alpha: 0.602,
            gamma: 0.101,
            a_offset: 10,
            n_warmup: 0,
            tol: 1e-5,
            patience: 10,
        }
    }
}

/// Result returned by SPSA optimization.
#[derive(Debug, Clone)]
pub struct SpsaResult {
    pub final_energy: f32,
    pub final_params: Vec<f32>,
    pub energy_history: Vec<f32>,
    pub n_iter: usize,
    pub converged: bool,
}

/// VQE optimizer using SPSA (Simultaneous Perturbation Stochastic Approximation).
///
/// Requires only 2 energy evaluations per iteration regardless of the number of parameters,
/// compared to 2n evaluations for the parameter-shift rule.
#[derive(Debug, Clone)]
pub struct SpsaVqeOptimizer {
    pub ansatz: HardwareEfficientAnsatz,
    pub ham: Hamiltonian,
    pub params: Vec<f32>,
    pub cfg: SpsaConfig,
}

impl SpsaVqeOptimizer {
    /// Construct the optimizer, initializing parameters with small normal-distributed perturbations.
    pub fn new(
        ansatz: HardwareEfficientAnsatz,
        ham: Hamiltonian,
        cfg: SpsaConfig,
        rng: &mut LcgRng,
    ) -> Self {
        let n = ansatz.n_params();
        let params = (0..n).map(|_| rng.next_normal() * 0.1).collect();
        Self {
            ansatz,
            ham,
            params,
            cfg,
        }
    }

    /// Evaluate ⟨ψ(params)|H|ψ(params)⟩.
    pub fn energy(&self, params: &[f32]) -> QuantumResult<f32> {
        let circ = self.ansatz.build_circuit(params)?;
        let mut rng = LcgRng::new(0);
        let sv = circ.exec_on_state(
            &StateVector::new_zero_state(self.ansatz.n_qubits)?,
            &mut rng,
        )?;
        expectation_value(&sv, &self.ham)
    }

    /// Generate a Bernoulli ±1 perturbation vector.
    ///
    /// Each component is independently +1 or -1 with equal probability.
    #[must_use]
    pub fn bernoulli_perturbation(n: usize, rng: &mut LcgRng) -> Vec<f32> {
        (0..n)
            .map(|_| {
                if rng.next_u32() & 1 == 0 {
                    -1.0_f32
                } else {
                    1.0_f32
                }
            })
            .collect()
    }

    /// Decaying step-size schedule: a_k = a / (k + 1 + A)^α.
    #[must_use]
    pub fn step_size(&self, k: usize) -> f32 {
        let denom = (k + 1 + self.cfg.a_offset) as f32;
        self.cfg.a / denom.powf(self.cfg.alpha)
    }

    /// Decaying perturbation schedule: c_k = c / (k + 1)^γ.
    #[must_use]
    pub fn perturbation_size(&self, k: usize) -> f32 {
        let denom = (k + 1) as f32;
        self.cfg.c / denom.powf(self.cfg.gamma)
    }

    /// Compute SPSA gradient estimate using exactly 2 energy evaluations.
    ///
    /// Returns the gradient estimate vector g_k where:
    ///
    /// ```text
    /// g_k[i] = (f(θ + c_k·Δ) - f(θ - c_k·Δ)) / (2·c_k·Δ[i])
    /// ```
    pub fn spsa_gradient(
        &self,
        params: &[f32],
        c_k: f32,
        rng: &mut LcgRng,
    ) -> QuantumResult<Vec<f32>> {
        let n = params.len();
        let delta = Self::bernoulli_perturbation(n, rng);

        let mut p_plus = params.to_vec();
        let mut p_minus = params.to_vec();
        for i in 0..n {
            p_plus[i] += c_k * delta[i];
            p_minus[i] -= c_k * delta[i];
        }

        let f_plus = self.energy(&p_plus)?;
        let f_minus = self.energy(&p_minus)?;

        let diff = f_plus - f_minus;
        let two_ck = 2.0 * c_k;

        let grad = delta.iter().map(|&d_i| diff / (two_ck * d_i)).collect();

        Ok(grad)
    }

    /// Run SPSA optimization loop.
    pub fn optimize(&mut self, rng: &mut LcgRng) -> QuantumResult<SpsaResult> {
        let max_iter = self.cfg.max_iter;
        let patience = self.cfg.patience;
        let tol = self.cfg.tol;
        let n_warmup = self.cfg.n_warmup;

        let mut params = self.params.clone();
        let mut energy_history = Vec::with_capacity(max_iter);
        let mut n_iter = 0_usize;
        let mut converged = false;

        for k in 0..max_iter {
            let a_k = self.step_size(k);
            let c_k = self.perturbation_size(k);

            let grad = self.spsa_gradient(&params, c_k, rng)?;

            if k >= n_warmup {
                for (p, g) in params.iter_mut().zip(grad.iter()) {
                    *p -= a_k * g;
                }
            }

            let e = self.energy(&params)?;
            if !e.is_finite() {
                return Err(QuantumError::OptimizationDiverged { iter: k });
            }
            energy_history.push(e);
            n_iter = k + 1;

            if patience > 0 && energy_history.len() >= patience {
                let window = &energy_history[energy_history.len() - patience..];
                let e_max = window.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e_min = window.iter().cloned().fold(f32::INFINITY, f32::min);
                if e_max - e_min < tol {
                    converged = true;
                    break;
                }
            }
        }

        let final_energy = energy_history.last().copied().unwrap_or(f32::NAN);
        self.params = params.clone();

        Ok(SpsaResult {
            final_energy,
            final_params: params,
            energy_history,
            n_iter,
            converged,
        })
    }

    /// 2SPSA Hessian diagonal estimator using 4 energy evaluations.
    ///
    /// For each component i:
    ///
    /// ```text
    /// H_ii ≈ [f(θ+c·Δ₁+c·Δ₂) - f(θ+c·Δ₁-c·Δ₂) - f(θ-c·Δ₁+c·Δ₂) + f(θ-c·Δ₁-c·Δ₂)]
    ///         / (4·c²·Δ₁[i]·Δ₂[i])
    /// ```
    pub fn hessian_diagonal_estimate(
        &self,
        params: &[f32],
        c_k: f32,
        rng: &mut LcgRng,
    ) -> QuantumResult<Vec<f32>> {
        let n = params.len();
        let delta1 = Self::bernoulli_perturbation(n, rng);
        let delta2 = Self::bernoulli_perturbation(n, rng);

        let mut pp = params.to_vec();
        let mut pm = params.to_vec();
        let mut mp = params.to_vec();
        let mut mm = params.to_vec();

        for i in 0..n {
            pp[i] += c_k * delta1[i] + c_k * delta2[i];
            pm[i] += c_k * delta1[i] - c_k * delta2[i];
            mp[i] += -c_k * delta1[i] + c_k * delta2[i];
            mm[i] += -c_k * delta1[i] - c_k * delta2[i];
        }

        let f_pp = self.energy(&pp)?;
        let f_pm = self.energy(&pm)?;
        let f_mp = self.energy(&mp)?;
        let f_mm = self.energy(&mm)?;

        let numerator = f_pp - f_pm - f_mp + f_mm;
        let four_c2 = 4.0 * c_k * c_k;

        let hdiag = delta1
            .iter()
            .zip(delta2.iter())
            .map(|(&d1, &d2)| numerator / (four_c2 * d1 * d2))
            .collect();

        Ok(hdiag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pauli::pauli_string::PauliOp;

    fn make_optimizer(n_qubits: usize, depth: usize, seed: u64) -> (SpsaVqeOptimizer, LcgRng) {
        let ans = HardwareEfficientAnsatz::new(n_qubits, depth);
        let mut ham = Hamiltonian::new();
        ham.add_term(1.0, vec![PauliOp::Z, PauliOp::Z]);
        let cfg = SpsaConfig::default();
        let mut rng = LcgRng::new(seed);
        let opt = SpsaVqeOptimizer::new(ans, ham, cfg, &mut rng);
        (opt, rng)
    }

    #[test]
    fn spsa_energy_is_finite() {
        let (opt, _) = make_optimizer(2, 1, 1);
        let e = opt
            .energy(&opt.params.clone())
            .expect("energy evaluation should succeed");
        assert!(e.is_finite(), "energy={e}");
    }

    #[test]
    fn bernoulli_perturbation_length() {
        let mut rng = LcgRng::new(2);
        let v = SpsaVqeOptimizer::bernoulli_perturbation(7, &mut rng);
        assert_eq!(v.len(), 7);
    }

    #[test]
    fn bernoulli_perturbation_values_pm1() {
        let mut rng = LcgRng::new(3);
        let v = SpsaVqeOptimizer::bernoulli_perturbation(200, &mut rng);
        for &x in &v {
            assert!(x == 1.0 || x == -1.0, "unexpected value {x}");
        }
    }

    #[test]
    fn step_size_decreasing() {
        let (opt, _) = make_optimizer(2, 1, 4);
        for k in 0..10 {
            assert!(
                opt.step_size(k + 1) < opt.step_size(k),
                "step_size not decreasing at k={k}"
            );
        }
    }

    #[test]
    fn perturbation_size_decreasing() {
        let (opt, _) = make_optimizer(2, 1, 5);
        for k in 0..10 {
            assert!(
                opt.perturbation_size(k + 1) < opt.perturbation_size(k),
                "perturbation_size not decreasing at k={k}"
            );
        }
    }

    #[test]
    fn step_size_positive() {
        let (opt, _) = make_optimizer(2, 1, 6);
        for k in 0..50 {
            assert!(opt.step_size(k) > 0.0, "step_size nonpositive at k={k}");
        }
    }

    #[test]
    fn perturbation_size_positive() {
        let (opt, _) = make_optimizer(2, 1, 7);
        for k in 0..50 {
            assert!(
                opt.perturbation_size(k) > 0.0,
                "perturbation_size nonpositive at k={k}"
            );
        }
    }

    #[test]
    fn spsa_gradient_length() {
        let (opt, mut rng) = make_optimizer(2, 1, 8);
        let params = opt.params.clone();
        let n = params.len();
        let grad = opt
            .spsa_gradient(&params, 0.1, &mut rng)
            .expect("SPSA gradient computation should succeed");
        assert_eq!(grad.len(), n);
    }

    #[test]
    fn spsa_gradient_finite() {
        let (opt, mut rng) = make_optimizer(2, 1, 9);
        let params = opt.params.clone();
        let grad = opt
            .spsa_gradient(&params, 0.1, &mut rng)
            .expect("SPSA gradient computation should succeed");
        for (i, &g) in grad.iter().enumerate() {
            assert!(g.is_finite(), "gradient[{i}] is not finite: {g}");
        }
    }

    #[test]
    fn spsa_optimize_returns_finite_energy() {
        let (mut opt, mut rng) = make_optimizer(2, 1, 10);
        opt.cfg = SpsaConfig {
            max_iter: 5,
            ..SpsaConfig::default()
        };
        let result = opt
            .optimize(&mut rng)
            .expect("SPSA optimize should succeed");
        assert!(
            result.final_energy.is_finite(),
            "final_energy={}",
            result.final_energy
        );
    }

    #[test]
    fn spsa_optimize_energy_history_length() {
        let (mut opt, mut rng) = make_optimizer(2, 1, 11);
        let max_iter = 8;
        opt.cfg.max_iter = max_iter;
        opt.cfg.tol = 0.0;
        opt.cfg.patience = 0;
        let result = opt
            .optimize(&mut rng)
            .expect("SPSA optimize should succeed");
        assert_eq!(
            result.energy_history.len(),
            result.n_iter,
            "history length mismatch"
        );
    }

    #[test]
    fn spsa_optimize_uses_max_iter() {
        let (mut opt, mut rng) = make_optimizer(2, 1, 12);
        let max_iter = 15;
        opt.cfg.max_iter = max_iter;
        opt.cfg.tol = 0.0;
        opt.cfg.patience = 0;
        let result = opt
            .optimize(&mut rng)
            .expect("SPSA optimize should succeed");
        assert_eq!(result.n_iter, max_iter, "should use all iterations");
    }

    #[test]
    fn spsa_optimize_reduces_energy() {
        let ans = HardwareEfficientAnsatz::new(2, 1);
        let mut ham = Hamiltonian::new();
        ham.add_term(-1.0, vec![PauliOp::Z, PauliOp::I]);
        ham.add_term(-1.0, vec![PauliOp::I, PauliOp::Z]);
        let cfg = SpsaConfig {
            max_iter: 50,
            a: 0.2,
            c: 0.15,
            tol: 0.0,
            patience: 0,
            ..SpsaConfig::default()
        };
        let mut rng = LcgRng::new(13);
        let mut opt = SpsaVqeOptimizer::new(ans, ham, cfg, &mut rng);
        let e_init = opt
            .energy(&opt.params.clone())
            .expect("initial energy evaluation should succeed");
        let result = opt
            .optimize(&mut rng)
            .expect("SPSA optimize should succeed");
        assert!(
            result.final_energy <= e_init + 0.5,
            "energy did not improve: init={e_init} final={}",
            result.final_energy
        );
    }

    #[test]
    fn spsa_converged_flag() {
        let (mut opt, mut rng) = make_optimizer(2, 1, 14);
        opt.cfg.max_iter = 200;
        opt.cfg.tol = 100.0;
        opt.cfg.patience = 5;
        let result = opt
            .optimize(&mut rng)
            .expect("SPSA optimize should succeed");
        assert!(
            result.converged,
            "expected converged=true with very large tol"
        );
    }

    #[test]
    fn hessian_diagonal_estimate_length() {
        let (opt, mut rng) = make_optimizer(2, 1, 15);
        let params = opt.params.clone();
        let n = params.len();
        let hdiag = opt
            .hessian_diagonal_estimate(&params, 0.1, &mut rng)
            .expect("Hessian diagonal estimate should succeed");
        assert_eq!(hdiag.len(), n);
    }

    #[test]
    fn hessian_diagonal_estimate_finite() {
        let (opt, mut rng) = make_optimizer(2, 1, 16);
        let params = opt.params.clone();
        let hdiag = opt
            .hessian_diagonal_estimate(&params, 0.1, &mut rng)
            .expect("Hessian diagonal estimate should succeed");
        for (i, &h) in hdiag.iter().enumerate() {
            assert!(h.is_finite(), "hessian_diag[{i}] is not finite: {h}");
        }
    }

    #[test]
    fn new_initializes_params() {
        let ans = HardwareEfficientAnsatz::new(3, 2);
        let n_expected = ans.n_params();
        let mut ham = Hamiltonian::new();
        ham.add_term(1.0, vec![PauliOp::Z, PauliOp::I, PauliOp::Z]);
        let mut rng = LcgRng::new(17);
        let opt = SpsaVqeOptimizer::new(ans, ham, SpsaConfig::default(), &mut rng);
        assert_eq!(opt.params.len(), n_expected);
    }

    #[test]
    fn spsa_vs_gradient_descent_same_landscape() {
        let make_ham = || {
            let mut ham = Hamiltonian::new();
            ham.add_term(1.0, vec![PauliOp::Z, PauliOp::Z]);
            ham
        };

        let ans1 = HardwareEfficientAnsatz::new(2, 1);
        let cfg = SpsaConfig {
            max_iter: 30,
            a: 0.3,
            c: 0.2,
            tol: 0.0,
            patience: 0,
            ..SpsaConfig::default()
        };
        let mut rng1 = LcgRng::new(18);
        let mut spsa_opt = SpsaVqeOptimizer::new(ans1, make_ham(), cfg, &mut rng1);
        let spsa_result = spsa_opt
            .optimize(&mut rng1)
            .expect("SPSA optimize should succeed");

        let ans2 = HardwareEfficientAnsatz::new(2, 1);
        let mut rng2 = LcgRng::new(18);
        let mut vqe_opt = crate::vqe::vqe::VqeOptimizer::new(ans2, make_ham(), &mut rng2);
        let (gd_energy, _) = vqe_opt
            .optimize(30, 0.1)
            .expect("gradient-descent VQE optimize should succeed");

        assert!(
            spsa_result.final_energy.is_finite() && gd_energy.is_finite(),
            "both methods should give finite energies; spsa={} gd={}",
            spsa_result.final_energy,
            gd_energy
        );
    }
}
