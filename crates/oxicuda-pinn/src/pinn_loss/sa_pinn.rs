//! Self-Adaptive PINN (SA-PINN) with per-point trainable weights.
//!
//! McClenny & Braga-Neto (2021) "Self-Adaptive Physics-Informed Neural Networks using a
//! Soft Attention Mechanism", arXiv:2009.04544.
//!
//! SA-PINN assigns a trainable raw weight parameter λ_i to each collocation point and
//! uses a **maximin** formulation: the outer maximisation over λ finds the hardest
//! collocation points, while the inner minimisation over the network θ satisfies physics.
//!
//! ## Update rule
//! - `λ_i ← λ_i + lr · r_i²`   (gradient ascent — harder points get higher weight)
//! - Positive weights: `softplus(λ_i) = log(1 + exp(λ_i))`
//! - Normalised weights (optional): `w_i = softplus(λ_i) / Σ_j softplus(λ_j)`
//! - Weighted PDE loss: `L = Σ_i w_i · r_i²`

use crate::error::{PinnError, PinnResult};

// ────────────────────────────── helpers ──────────────────────────────────────

/// Numerically stable softplus: `log(1 + exp(x))`.
///
/// - For `x > 20`: `softplus(x) ≈ x` (avoids overflow in `exp`)
/// - For `x < -20`: `softplus(x) ≈ exp(x)` (avoids cancellation in `1 + exp(x)`)
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0_f32 + x.exp()).ln()
    }
}

/// Sigmoid (derivative of softplus): `1 / (1 + exp(-x))`.
///
/// - For `x > 20`: `sigmoid(x) ≈ 1`
/// - For `x < -20`: `sigmoid(x) ≈ 0`
fn sigmoid(x: f32) -> f32 {
    if x > 20.0 {
        1.0
    } else if x < -20.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

// ────────────────────────────── config ───────────────────────────────────────

/// Configuration for SA-PINN self-adaptive weighting.
#[derive(Debug, Clone)]
pub struct SaPinnConfig {
    /// Number of collocation points.
    pub n_points: usize,
    /// Learning rate for the `λ` gradient-ascent step.
    /// Typical range: 1e-4 to 1e-2.
    pub lambda_lr: f32,
    /// Initial raw λ value for all points.
    /// `softplus(init_lambda)` is the initial effective weight per point.
    pub init_lambda: f32,
    /// If `true`, normalise the effective weights so that `Σ_i w_i = 1`.
    pub normalize_weights: bool,
}

impl SaPinnConfig {
    /// Default SA-PINN config for `n_points` collocation points.
    pub fn new(n_points: usize) -> Self {
        Self {
            n_points,
            lambda_lr: 1e-3,
            init_lambda: 0.0,
            normalize_weights: true,
        }
    }
}

// ────────────────────────────── SA-PINN ──────────────────────────────────────

/// Self-Adaptive PINN with per-point trainable weights.
///
/// Uses gradient ascent on raw λ values to focus the residual loss on hard collocation
/// points, implementing the "maximise over λ" step of the maximin formulation.
///
/// # Example
/// ```
/// use oxicuda_pinn::pinn_loss::sa_pinn::{SaPinn, SaPinnConfig};
///
/// let config = SaPinnConfig::new(4);
/// let mut sa = SaPinn::new(config).unwrap();
/// let residuals = vec![0.1_f32, 0.5, 0.3, 0.8];
/// let loss = sa.weighted_loss(&residuals).unwrap();
/// sa.update_lambdas(&residuals).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SaPinn {
    /// Raw (un-activated) per-point weight parameters.
    /// Effective positive weights are `softplus(lambdas[i])`.
    pub lambdas: Vec<f32>,
    pub config: SaPinnConfig,
}

impl SaPinn {
    /// Create a new SA-PINN with all `λ` values initialised to `config.init_lambda`.
    ///
    /// # Errors
    /// - `EmptyCollocationSet` if `n_points == 0`.
    /// - `InvalidWeight { weight: lambda_lr }` if `lambda_lr ≤ 0` or not finite.
    pub fn new(config: SaPinnConfig) -> PinnResult<Self> {
        if config.n_points == 0 {
            return Err(PinnError::EmptyCollocationSet);
        }
        if !config.lambda_lr.is_finite() || config.lambda_lr <= 0.0 {
            return Err(PinnError::InvalidWeight {
                weight: config.lambda_lr,
            });
        }
        let lambdas = vec![config.init_lambda; config.n_points];
        Ok(Self { lambdas, config })
    }

    /// Compute effective per-point weights from raw λ values.
    ///
    /// Returns `softplus(λ_i)` for each point. If `normalize_weights` is `true`,
    /// divides each by the total so that `Σ w_i = 1`.
    ///
    /// If the total of all softplus values is ≤ 0 (pathological), falls back to
    /// uniform weights `1 / n_points`.
    pub fn weights(&self) -> Vec<f32> {
        let sp: Vec<f32> = self.lambdas.iter().map(|&l| softplus(l)).collect();
        if !self.config.normalize_weights {
            return sp;
        }
        let total: f32 = sp.iter().sum();
        if total <= 0.0 {
            return vec![1.0 / self.config.n_points as f32; self.config.n_points];
        }
        sp.iter().map(|&s| s / total).collect()
    }

    /// Compute the weighted PDE residual loss.
    ///
    /// If `normalize_weights`: `L = Σ_i w_i · r_i²` where `Σ w_i = 1`.
    /// Otherwise: `L = Σ_i softplus(λ_i) · r_i²`.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `residuals.len() != n_points`.
    /// - `NanEncountered` if the loss is not finite.
    pub fn weighted_loss(&self, residuals: &[f32]) -> PinnResult<f32> {
        if residuals.len() != self.config.n_points {
            return Err(PinnError::DimensionMismatch {
                expected: self.config.n_points,
                got: residuals.len(),
            });
        }
        let w = self.weights();
        let loss: f32 = w
            .iter()
            .zip(residuals.iter())
            .map(|(&wi, &r)| wi * r * r)
            .sum();
        if !loss.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "sa_pinn_weighted_loss",
            });
        }
        Ok(loss)
    }

    /// Update raw λ values by gradient ascent: `λ_i ← λ_i + lr · r_i²`.
    ///
    /// This implements the "maximise over λ" step of the maximin formulation.
    /// Points with larger residuals accumulate higher weights for subsequent iterations.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `residuals.len() != n_points`.
    pub fn update_lambdas(&mut self, residuals: &[f32]) -> PinnResult<()> {
        if residuals.len() != self.config.n_points {
            return Err(PinnError::DimensionMismatch {
                expected: self.config.n_points,
                got: residuals.len(),
            });
        }
        for (lambda, &r) in self.lambdas.iter_mut().zip(residuals.iter()) {
            *lambda += self.config.lambda_lr * r * r;
        }
        Ok(())
    }

    /// Reset all λ values to `config.init_lambda`.
    pub fn reset(&mut self) {
        for l in &mut self.lambdas {
            *l = self.config.init_lambda;
        }
    }

    /// Compute the effective number of active collocation points (perplexity of weight
    /// distribution).
    ///
    /// ```text
    /// N_eff = exp(H)    where H = -Σ_i w_i · log(w_i)
    /// ```
    /// - `N_eff = 1`: only one point has all the weight.
    /// - `N_eff = n_points`: all weights are equal (uniform focus).
    ///
    /// # Errors
    /// - `NanEncountered` if the entropy is not finite.
    pub fn effective_n(&self) -> PinnResult<f32> {
        let w = self.weights();
        // Ensure weights are probability-normalised for entropy computation.
        let total: f32 = w.iter().sum();
        let norm_w: Vec<f32> = if (total - 1.0).abs() > 1e-6 {
            w.iter().map(|&x| x / total).collect()
        } else {
            w.clone()
        };
        let entropy: f32 = norm_w
            .iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| -x * x.ln())
            .sum();
        let n_eff = entropy.exp();
        if !n_eff.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "sa_pinn_effective_n",
            });
        }
        Ok(n_eff)
    }

    /// Return the index of the collocation point with the highest effective weight.
    pub fn argmax_weight(&self) -> usize {
        let w = self.weights();
        let mut best_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &wi) in w.iter().enumerate() {
            if wi > best_val {
                best_val = wi;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Compute the gradient of the weighted loss with respect to each raw λ_i.
    ///
    /// For normalised weights the gradient takes the quotient-rule form:
    ///
    /// ```text
    /// ∂L/∂λ_i = sp'(λ_i) · (r_i² / S - W)
    /// ```
    ///
    /// where:
    /// - `S = Σ_j sp(λ_j)` (total softplus)
    /// - `W = Σ_j sp(λ_j)·r_j² / S²` (weighted mean of squared residuals)
    /// - `sp'(x) = sigmoid(x)`
    ///
    /// For un-normalised weights: `∂L/∂λ_i = sigmoid(λ_i) · r_i²`.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `residuals.len() != n_points`.
    /// - `NanEncountered` if the total softplus is zero or any gradient element is
    ///   not finite.
    pub fn lambda_gradient(&self, residuals: &[f32]) -> PinnResult<Vec<f32>> {
        if residuals.len() != self.config.n_points {
            return Err(PinnError::DimensionMismatch {
                expected: self.config.n_points,
                got: residuals.len(),
            });
        }

        if !self.config.normalize_weights {
            // Un-normalised: ∂L/∂λ_i = sigmoid(λ_i) · r_i²
            let mut grad = Vec::with_capacity(self.config.n_points);
            for (&lambda_i, &r_i) in self.lambdas.iter().zip(residuals.iter()) {
                let g = sigmoid(lambda_i) * r_i * r_i;
                if !g.is_finite() {
                    return Err(PinnError::NanEncountered {
                        location: "lambda_gradient_elem",
                    });
                }
                grad.push(g);
            }
            return Ok(grad);
        }

        // Normalised weights: quotient rule
        let sp: Vec<f32> = self.lambdas.iter().map(|&l| softplus(l)).collect();
        let sp_d: Vec<f32> = self.lambdas.iter().map(|&l| sigmoid(l)).collect();
        let sp_sum: f32 = sp.iter().sum();
        if sp_sum <= 0.0 {
            return Err(PinnError::NanEncountered {
                location: "lambda_gradient",
            });
        }
        // Weighted mean: Σ_j (sp_j / sp_sum) · r_j²
        let weighted_r2: f32 = sp
            .iter()
            .zip(residuals.iter())
            .map(|(&si, &ri)| si * ri * ri)
            .sum::<f32>()
            / (sp_sum * sp_sum);

        let mut grad = Vec::with_capacity(self.config.n_points);
        for (&sp_di, &r_i) in sp_d.iter().zip(residuals.iter()) {
            let r2_i = r_i * r_i;
            // ∂L/∂λ_i = sp'_i · (r²_i / sp_sum − weighted_r2)
            let g = sp_di * (r2_i / sp_sum - weighted_r2);
            if !g.is_finite() {
                return Err(PinnError::NanEncountered {
                    location: "lambda_gradient_elem",
                });
            }
            grad.push(g);
        }
        Ok(grad)
    }
}

// ─────────────────────────────────── tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sa(n: usize) -> SaPinn {
        SaPinn::new(SaPinnConfig::new(n)).unwrap()
    }

    // ── softplus / sigmoid helpers ────────────────────────────────────────────

    #[test]
    fn sa_pinn_softplus_stability() {
        // softplus(-100) ≈ exp(-100) ≈ 3.7e-44, must not be NaN or 0
        let val = softplus(-100.0_f32);
        assert!(val.is_finite(), "softplus(-100) must be finite, got {val}");
        assert!(val > 0.0, "softplus(-100) must be > 0, got {val}");
        // softplus(100) ≈ 100 (large branch)
        let large = softplus(100.0_f32);
        assert!(
            (large - 100.0_f32).abs() < 1.0,
            "softplus(100) ≈ 100, got {large}"
        );
    }

    #[test]
    fn sa_pinn_sigmoid_bounds() {
        assert!((sigmoid(0.0_f32) - 0.5_f32).abs() < 1e-6);
        assert!((sigmoid(100.0_f32) - 1.0_f32).abs() < 1e-6);
        assert!(sigmoid(-100.0_f32).abs() < 1e-6);
    }

    // ── new / construction ────────────────────────────────────────────────────

    #[test]
    fn sa_pinn_new_init_lambdas() {
        let cfg = SaPinnConfig {
            n_points: 5,
            lambda_lr: 1e-3,
            init_lambda: 2.0,
            normalize_weights: true,
        };
        let sa = SaPinn::new(cfg).unwrap();
        assert!(
            sa.lambdas.iter().all(|&l| (l - 2.0_f32).abs() < 1e-7),
            "All lambdas should be init_lambda=2.0"
        );
    }

    #[test]
    fn sa_pinn_empty_n_points() {
        let cfg = SaPinnConfig::new(0);
        assert!(matches!(
            SaPinn::new(cfg),
            Err(PinnError::EmptyCollocationSet)
        ));
    }

    #[test]
    fn sa_pinn_invalid_lr_zero() {
        let cfg = SaPinnConfig {
            n_points: 4,
            lambda_lr: 0.0,
            init_lambda: 0.0,
            normalize_weights: true,
        };
        assert!(matches!(
            SaPinn::new(cfg),
            Err(PinnError::InvalidWeight { .. })
        ));
    }

    #[test]
    fn sa_pinn_invalid_lr_negative() {
        let cfg = SaPinnConfig {
            n_points: 4,
            lambda_lr: -1e-3,
            init_lambda: 0.0,
            normalize_weights: true,
        };
        assert!(matches!(
            SaPinn::new(cfg),
            Err(PinnError::InvalidWeight { .. })
        ));
    }

    // ── weights ───────────────────────────────────────────────────────────────

    #[test]
    fn sa_pinn_weights_sum_to_one_normalized() {
        let sa = make_sa(8);
        let w = sa.weights();
        let total: f32 = w.iter().sum();
        assert!(
            (total - 1.0_f32).abs() < 1e-6,
            "Normalised weights must sum to 1, got {total}"
        );
    }

    #[test]
    fn sa_pinn_weights_all_positive() {
        let sa = make_sa(10);
        let w = sa.weights();
        assert!(
            w.iter().all(|&wi| wi > 0.0),
            "All weights must be strictly positive"
        );
    }

    #[test]
    fn sa_pinn_weights_uniform_when_equal_lambda() {
        let sa = make_sa(6);
        let w = sa.weights();
        let expected = 1.0_f32 / 6.0;
        for (i, &wi) in w.iter().enumerate() {
            assert!(
                (wi - expected).abs() < 1e-6,
                "Equal lambdas → equal normalised weights: w[{i}]={wi} expected {expected}"
            );
        }
    }

    #[test]
    fn sa_pinn_un_normalized_weights_not_sum_one() {
        let cfg = SaPinnConfig {
            n_points: 4,
            lambda_lr: 1e-3,
            init_lambda: 0.0,
            normalize_weights: false,
        };
        let sa = SaPinn::new(cfg).unwrap();
        let w = sa.weights();
        // softplus(0) = ln(2) ≈ 0.693; sum of 4 such weights = 4*ln(2) ≈ 2.77 ≠ 1
        let total: f32 = w.iter().sum();
        let ln2 = 2.0_f32.ln();
        assert!(
            (total - 4.0 * ln2).abs() < 1e-5,
            "Unnormalised weights: sum should be 4·ln(2)≈{:.4}, got {total:.4}",
            4.0 * ln2
        );
    }

    #[test]
    fn sa_pinn_high_lambda_weight_large() {
        // A very high lambda should dominate the weights
        let cfg = SaPinnConfig {
            n_points: 3,
            lambda_lr: 1e-3,
            init_lambda: 0.0,
            normalize_weights: true,
        };
        let mut sa = SaPinn::new(cfg).unwrap();
        sa.lambdas[1] = 1000.0; // hugely increased
        let w = sa.weights();
        // softplus(1000) ≈ 1000 >> softplus(0) ≈ 0.693
        assert!(
            w[1] > 0.99,
            "High lambda should dominate normalised weight, w[1]={:.6}",
            w[1]
        );
    }

    // ── weighted_loss ─────────────────────────────────────────────────────────

    #[test]
    fn sa_pinn_weighted_loss_zero_residuals() {
        let sa = make_sa(5);
        let r = vec![0.0_f32; 5];
        let loss = sa.weighted_loss(&r).unwrap();
        assert!(loss.abs() < 1e-8, "Zero residuals → loss = 0, got {loss}");
    }

    #[test]
    fn sa_pinn_weighted_loss_positive() {
        let sa = make_sa(4);
        let r = vec![0.5_f32, 1.0, 0.3, 0.7];
        let loss = sa.weighted_loss(&r).unwrap();
        assert!(loss > 0.0, "Non-zero residuals → loss > 0, got {loss}");
    }

    #[test]
    fn sa_pinn_dimension_mismatch() {
        let sa = make_sa(4);
        let r = vec![0.5_f32; 3]; // wrong length
        assert!(matches!(
            sa.weighted_loss(&r),
            Err(PinnError::DimensionMismatch {
                expected: 4,
                got: 3
            })
        ));
    }

    // ── update_lambdas ────────────────────────────────────────────────────────

    #[test]
    fn sa_pinn_update_increases_lambda() {
        let mut sa = make_sa(3);
        let init = sa.lambdas[1];
        let r = vec![0.0_f32, 2.0, 0.0]; // only point 1 has residual
        sa.update_lambdas(&r).unwrap();
        assert!(
            sa.lambdas[1] > init,
            "High residual should increase lambda: {init} → {}",
            sa.lambdas[1]
        );
    }

    #[test]
    fn sa_pinn_update_low_residual_small_change() {
        let mut sa = make_sa(3);
        let init = sa.lambdas[0];
        let r = vec![0.0_f32, 2.0, 2.0]; // point 0 has zero residual
        sa.update_lambdas(&r).unwrap();
        assert!(
            (sa.lambdas[0] - init).abs() < 1e-10,
            "Zero residual → lambda unchanged: {init} → {}",
            sa.lambdas[0]
        );
    }

    #[test]
    fn sa_pinn_update_dimension_mismatch() {
        let mut sa = make_sa(4);
        let r = vec![0.5_f32; 5]; // wrong length
        assert!(matches!(
            sa.update_lambdas(&r),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    // ── reset ─────────────────────────────────────────────────────────────────

    #[test]
    fn sa_pinn_reset_restores_init() {
        let cfg = SaPinnConfig {
            n_points: 4,
            lambda_lr: 1e-3,
            init_lambda: 1.5,
            normalize_weights: true,
        };
        let mut sa = SaPinn::new(cfg).unwrap();
        let r = vec![3.0_f32; 4];
        sa.update_lambdas(&r).unwrap();
        // lambdas are now > 1.5
        sa.reset();
        assert!(
            sa.lambdas.iter().all(|&l| (l - 1.5_f32).abs() < 1e-7),
            "After reset, all lambdas should equal init_lambda=1.5"
        );
    }

    // ── effective_n ───────────────────────────────────────────────────────────

    #[test]
    fn sa_pinn_effective_n_uniform() {
        let n = 8_usize;
        let sa = make_sa(n);
        // Uniform weights → entropy = ln(n) → N_eff = n
        let n_eff = sa.effective_n().unwrap();
        assert!(
            (n_eff - n as f32).abs() < 1e-4,
            "Uniform weights: N_eff should be {n}, got {n_eff}"
        );
    }

    #[test]
    fn sa_pinn_effective_n_concentrated() {
        let n = 6_usize;
        let cfg = SaPinnConfig {
            n_points: n,
            lambda_lr: 1e-3,
            init_lambda: 0.0,
            normalize_weights: true,
        };
        let mut sa = SaPinn::new(cfg).unwrap();
        // Drive one lambda very high so that point dominates
        sa.lambdas[2] = 1000.0;
        let n_eff = sa.effective_n().unwrap();
        assert!(
            n_eff < 2.0,
            "Concentrated weight: N_eff should approach 1, got {n_eff}"
        );
    }

    // ── argmax_weight ─────────────────────────────────────────────────────────

    #[test]
    fn sa_pinn_argmax_weight_correct() {
        let cfg = SaPinnConfig {
            n_points: 5,
            lambda_lr: 1e-3,
            init_lambda: 0.0,
            normalize_weights: true,
        };
        let mut sa = SaPinn::new(cfg).unwrap();
        sa.lambdas[3] = 50.0; // highest lambda at index 3
        assert_eq!(sa.argmax_weight(), 3, "argmax_weight should return index 3");
    }

    // ── lambda_gradient ───────────────────────────────────────────────────────

    #[test]
    fn sa_pinn_lambda_gradient_length() {
        let n = 5;
        let sa = make_sa(n);
        let r = vec![0.3_f32, 0.7, 0.1, 0.9, 0.5];
        let grad = sa.lambda_gradient(&r).unwrap();
        assert_eq!(grad.len(), n, "Gradient length must equal n_points");
    }

    #[test]
    fn sa_pinn_lambda_gradient_sign_high_residual() {
        // With normalised weights, the gradient at the high-residual point should be
        // positive (gradient ascent raises that point's lambda).
        let n = 3;
        let sa = make_sa(n);
        // Point 2 has a much larger residual
        let r = vec![0.01_f32, 0.01, 10.0];
        let grad = sa.lambda_gradient(&r).unwrap();
        assert!(
            grad[2] > 0.0,
            "High-residual point gradient should be positive, got {}",
            grad[2]
        );
    }

    #[test]
    fn sa_pinn_lambda_gradient_all_finite() {
        let sa = make_sa(4);
        let r = vec![0.5_f32, 1.2, 0.8, 0.3];
        let grad = sa.lambda_gradient(&r).unwrap();
        assert!(
            grad.iter().all(|&g| g.is_finite()),
            "All gradient elements must be finite: {grad:?}"
        );
    }

    #[test]
    fn sa_pinn_lambda_gradient_dimension_mismatch() {
        let sa = make_sa(4);
        let r = vec![0.5_f32; 3]; // wrong length
        assert!(matches!(
            sa.lambda_gradient(&r),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn sa_pinn_lambda_gradient_unnormalized() {
        // Unnormalised: ∂L/∂λ_i = sigmoid(λ_i) · r_i²  — always non-negative
        let cfg = SaPinnConfig {
            n_points: 4,
            lambda_lr: 1e-3,
            init_lambda: 0.0,
            normalize_weights: false,
        };
        let sa = SaPinn::new(cfg).unwrap();
        let r = vec![0.5_f32, 1.0, 0.3, 0.7];
        let grad = sa.lambda_gradient(&r).unwrap();
        for (i, &g) in grad.iter().enumerate() {
            assert!(
                g >= 0.0,
                "Unnormalised gradient must be non-negative: grad[{i}]={g}"
            );
        }
    }
}
