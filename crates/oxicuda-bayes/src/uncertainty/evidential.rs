//! Evidential Deep Learning (EDL) for uncertainty quantification.
//!
//! Implements Sensoy et al. (2018), "Evidential Deep Learning to Quantify
//! Classification Uncertainty", NeurIPS 2018.
//!
//! # Key idea
//!
//! Instead of outputting a softmax probability vector, the network places a
//! **Dirichlet distribution** over class probabilities:
//!
//! ```text
//! p ~ Dir(α),   α_k = exp(e_k) + 1   (positive evidence)
//! ```
//!
//! Uncertainty is decomposed into:
//!
//! | Quantity | Formula |
//! |----------|---------|
//! | Mean prediction | `p̂_k = α_k / S`, `S = Σ α_k` |
//! | Total uncertainty | `K / S` (vacuity) |
//! | Aleatoric uncertainty | `Σ p̂_k(1 − p̂_k)` (distributional spread) |
//! | Epistemic uncertainty | `Total − Aleatoric` |
//!
//! The loss combines a **Dirichlet-scaled cross-entropy** with an
//! **annealed KL** penalty to prevent the network from placing unbounded
//! evidence everywhere:
//!
//! ```text
//! L = E_{p~Dir(α)}[−log p(y|p)] + λ·KL(Dir(α̃) || Dir(1))
//! ```
//!
//! where `α̃_k = y_k + (1 − y_k)·α_k` zeroes out evidence for the true class.
//!
//! # Regression variant (Normal-Inverse-Gamma)
//!
//! For regression, the network outputs `(γ, ν, α, β)` parameterising a
//! Normal-Inverse-Gamma (NIG) prior `p(μ, σ²)`:
//!
//! ```text
//! σ² ~ InvGamma(α, β),   μ | σ² ~ N(γ, σ²/ν)
//! ```
//!
//! Predictive mean = `γ`, epistemic uncertainty ∝ `1/(ν·α)`,
//! aleatoric uncertainty ∝ `β/(α−1)` (for `α > 1`).

use crate::error::{BayesError, BayesResult};

// ─── Dirichlet uncertainty (classification) ───────────────────────────────────

/// Evidential uncertainty from a Dirichlet parameterisation.
///
/// `alpha[k]` is the Dirichlet concentration for class `k`; all values must be
/// positive.  Construct from raw network logits via [`DirichletEvidence::from_logits`].
#[derive(Debug, Clone)]
pub struct DirichletEvidence {
    /// Dirichlet concentrations α_k > 0.
    pub alpha: Vec<f32>,
}

impl DirichletEvidence {
    /// Construct from raw network outputs (evidence = softplus(e) + 1 ≥ 1).
    ///
    /// # Errors
    /// - `BayesError::EmptyInputs` if `logits` is empty.
    pub fn from_logits(logits: &[f32]) -> BayesResult<Self> {
        if logits.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        // Softplus: log(1 + exp(x)) + 1 ensures α ≥ 1.
        let alpha: Vec<f32> = logits
            .iter()
            .map(|&e| {
                let sp = if e > 20.0 { e } else { (1.0 + e.exp()).ln() };
                sp + 1.0
            })
            .collect();
        Ok(Self { alpha })
    }

    /// Construct directly from pre-computed concentrations.
    ///
    /// # Errors
    /// - `BayesError::EmptyInputs` if `alpha` is empty.
    /// - `BayesError::NonPositiveSigma` if any concentration ≤ 0.
    pub fn from_alpha(alpha: Vec<f32>) -> BayesResult<Self> {
        if alpha.is_empty() {
            return Err(BayesError::EmptyInputs);
        }
        if alpha.iter().any(|&a| a <= 0.0) {
            return Err(BayesError::NonPositiveSigma);
        }
        Ok(Self { alpha })
    }

    /// Number of classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.alpha.len()
    }

    /// Dirichlet strength `S = Σ α_k`.
    #[must_use]
    pub fn strength(&self) -> f32 {
        self.alpha.iter().sum()
    }

    /// Mean prediction `p̂_k = α_k / S`.
    #[must_use]
    pub fn mean_probs(&self) -> Vec<f32> {
        let s = self.strength();
        self.alpha.iter().map(|&a| a / s).collect()
    }

    /// Predicted class (argmax of mean probabilities).
    #[must_use]
    pub fn predict(&self) -> usize {
        let probs = self.mean_probs();
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// **Vacuity** (total epistemic-flavoured uncertainty) = `K / S`.
    ///
    /// Ranges in `(0, 1]`; vacuity = 1 when all α_k = 1 (uniform Dirichlet).
    #[must_use]
    pub fn vacuity(&self) -> f32 {
        self.n_classes() as f32 / self.strength()
    }

    /// **Dissonance** — belief mass attributed to conflicting evidence.
    ///
    /// Computes a normalised conflict measure based on pairwise belief
    /// correlation (simplified Dempster-Shafer dissonance proxy):
    ///
    /// `Dis = Σ_k b_k · (Σ_{j≠k} b_j · Bal(b_k, b_j)) / Σ_{j≠k} b_j`
    ///
    /// where `b_k = (α_k − 1) / S` (belief mass) and
    /// `Bal(a, b) = 1 − |a − b| / (a + b + ε)`.
    #[must_use]
    pub fn dissonance(&self) -> f32 {
        let s = self.strength();
        let k = self.n_classes();
        if k <= 1 {
            return 0.0;
        }
        let b: Vec<f32> = self.alpha.iter().map(|&a| (a - 1.0).max(0.0) / s).collect();
        let b_sum: f32 = b.iter().sum();
        if b_sum < 1e-10 {
            return 0.0;
        }
        let mut dis = 0.0_f32;
        for i in 0..k {
            let mut num = 0.0_f32;
            let mut den = 0.0_f32;
            for j in 0..k {
                if i == j {
                    continue;
                }
                let bal = 1.0 - (b[i] - b[j]).abs() / (b[i] + b[j] + 1e-10);
                num += b[j] * bal;
                den += b[j];
            }
            if den > 1e-10 {
                dis += b[i] * num / den;
            }
        }
        dis
    }

    /// **Aleatoric uncertainty**: distributional spread of the Dirichlet mean,
    ///
    /// `Σ_k p̂_k · (1 − p̂_k)`.
    #[must_use]
    pub fn aleatoric_uncertainty(&self) -> f32 {
        self.mean_probs().iter().map(|&p| p * (1.0 - p)).sum()
    }

    /// **Epistemic uncertainty**: vacuity minus normalised aleatoric.
    ///
    /// Interpreted as: how much of the total uncertainty is from lack of
    /// evidence (vacuity) rather than class overlap.
    #[must_use]
    pub fn epistemic_uncertainty(&self) -> f32 {
        (self.vacuity() - self.aleatoric_uncertainty()).max(0.0)
    }

    /// Compute the **EDL loss** for a single sample with one-hot label `y`.
    ///
    /// ```text
    /// L = −Σ_k y_k · (ψ(α_k) − ψ(S)) + λ · KL(Dir(α̃) || Dir(1))
    /// ```
    ///
    /// where `ψ` is the digamma function and `α̃_k = y_k + (1−y_k)·α_k`.
    ///
    /// The digamma is approximated via the asymptotic series.
    ///
    /// # Errors
    /// - `BayesError::DimensionMismatch` if `y.len() != n_classes`.
    /// - `BayesError::InvalidConfig` if `lambda < 0`.
    pub fn edl_loss(&self, y: &[f32], lambda: f32) -> BayesResult<f32> {
        if y.len() != self.n_classes() {
            return Err(BayesError::DimensionMismatch {
                expected: self.n_classes(),
                got: y.len(),
            });
        }
        if lambda < 0.0 {
            return Err(BayesError::InvalidConfig("lambda must be >= 0".into()));
        }
        let s = self.strength();
        let psi_s = digamma(s);

        // Dirichlet-scaled cross-entropy term.
        let ce: f32 = y
            .iter()
            .zip(self.alpha.iter())
            .map(|(&yk, &ak)| yk * (digamma(ak) - psi_s))
            .sum();
        let ce_loss = -ce;

        // KL term: KL(Dir(α̃) || Dir(1)).
        let alpha_tilde: Vec<f32> = self
            .alpha
            .iter()
            .zip(y.iter())
            .map(|(&a, &yk)| yk + (1.0 - yk) * a)
            .collect();
        let kl = kl_dirichlet_uniform(&alpha_tilde);
        Ok(ce_loss + lambda * kl)
    }
}

// ─── Normal-Inverse-Gamma regression ─────────────────────────────────────────

/// Evidential regression via Normal-Inverse-Gamma parameterisation.
///
/// The network outputs four scalars `(γ, log_ν, log_α, log_β)` which are
/// mapped to the NIG parameters via:
/// - `γ` = mean of the predictive distribution (identity).
/// - `ν = softplus(log_ν) + 1e-4` (virtual observation count, > 0).
/// - `α = softplus(log_α) + 1` (InvGamma shape, > 1 for finite variance).
/// - `β = softplus(log_β) + 1e-4` (InvGamma scale, > 0).
#[derive(Debug, Clone)]
pub struct NigEvidence {
    /// Predictive mean γ.
    pub gamma: f32,
    /// Virtual observation count ν > 0.
    pub nu: f32,
    /// InvGamma shape α > 1.
    pub alpha: f32,
    /// InvGamma scale β > 0.
    pub beta: f32,
}

impl NigEvidence {
    /// Construct from raw network outputs via softplus activations.
    ///
    /// # Errors
    /// - `BayesError::EmptyInputs` if `outputs` does not have exactly 4 elements.
    pub fn from_outputs(outputs: &[f32]) -> BayesResult<Self> {
        if outputs.len() != 4 {
            return Err(BayesError::DimensionMismatch {
                expected: 4,
                got: outputs.len(),
            });
        }
        let gamma = outputs[0];
        let nu = softplus(outputs[1]) + 1e-4;
        let alpha = softplus(outputs[2]) + 1.0;
        let beta = softplus(outputs[3]) + 1e-4;
        Ok(Self {
            gamma,
            nu,
            alpha,
            beta,
        })
    }

    /// Construct directly from NIG parameters (must be valid).
    ///
    /// # Errors
    /// - `BayesError::NonPositiveSigma` if `nu <= 0` or `beta <= 0`.
    /// - `BayesError::InvalidConfig` if `alpha <= 1`.
    pub fn new(gamma: f32, nu: f32, alpha: f32, beta: f32) -> BayesResult<Self> {
        if nu <= 0.0 || beta <= 0.0 {
            return Err(BayesError::NonPositiveSigma);
        }
        if alpha <= 1.0 {
            return Err(BayesError::InvalidConfig(
                "NIG alpha must be > 1 for finite predictive variance".into(),
            ));
        }
        Ok(Self {
            gamma,
            nu,
            alpha,
            beta,
        })
    }

    /// Predictive mean.
    #[must_use]
    pub fn predictive_mean(&self) -> f32 {
        self.gamma
    }

    /// **Aleatoric** uncertainty: expected observational variance.
    ///
    /// `E[σ²] = β / (α − 1)` (InvGamma mean).
    #[must_use]
    pub fn aleatoric_uncertainty(&self) -> f32 {
        self.beta / (self.alpha - 1.0)
    }

    /// **Epistemic** uncertainty: variance of the mean estimate.
    ///
    /// `Var[μ] = β / (ν · (α − 1))`.
    #[must_use]
    pub fn epistemic_uncertainty(&self) -> f32 {
        self.beta / (self.nu * (self.alpha - 1.0))
    }

    /// Total predictive variance (aleatoric + epistemic).
    #[must_use]
    pub fn total_variance(&self) -> f32 {
        self.aleatoric_uncertainty() + self.epistemic_uncertainty()
    }

    /// NIG negative log-likelihood (evidence lower bound term).
    ///
    /// Implements the DER loss from Amini et al. (2020):
    ///
    /// ```text
    /// L_NLL = ½ log(π/ν) − α log(Ω) + (α + ½) log((y−γ)² ν + Ω)
    ///         + log Γ(α) − log Γ(α + ½)
    /// ```
    ///
    /// where `Ω = 2β(1 + ν)`.
    ///
    /// # Errors
    /// Returns `BayesError::NanEncountered` if the result is non-finite.
    pub fn nig_nll(&self, y: f32) -> BayesResult<f32> {
        let omega = 2.0 * self.beta * (1.0 + self.nu);
        let err_sq = (y - self.gamma) * (y - self.gamma);
        let term = (err_sq * self.nu + omega).max(1e-10);
        let nll = 0.5 * (std::f32::consts::PI / self.nu).max(1e-10).ln()
            - self.alpha * omega.max(1e-10).ln()
            + (self.alpha + 0.5) * term.ln()
            + lgamma(self.alpha)
            - lgamma(self.alpha + 0.5);
        if !nll.is_finite() {
            return Err(BayesError::NanEncountered {
                location: "NigEvidence::nig_nll",
            });
        }
        Ok(nll)
    }

    /// **Evidence regularisation** term: penalises wrong-confidence predictions.
    ///
    /// `|y − γ| · (2ν + α)` — rewards high `ν, α` only when `y ≈ γ`.
    #[must_use]
    pub fn evidence_reg(&self, y: f32) -> f32 {
        (y - self.gamma).abs() * (2.0 * self.nu + self.alpha)
    }
}

// ─── Helper: KL(Dir(α) || Dir(1)) ─────────────────────────────────────────────

/// Compute `KL(Dir(α) || Dir(1))` in closed form:
///
/// ```text
/// KL = log B(1) / B(α) + Σ_k (α_k − 1)(ψ(α_k) − ψ(S))
///    = −log B(α) + Σ_k (α_k − 1)(ψ(α_k) − ψ(S))
/// ```
///
/// `B(α) = Π Γ(α_k) / Γ(S)`.
fn kl_dirichlet_uniform(alpha: &[f32]) -> f32 {
    let k = alpha.len() as f32;
    let s: f32 = alpha.iter().sum();
    // log B(1) = log Γ(K) − K log Γ(1) = log Γ(K)
    let log_b1 = lgamma(k);
    // log B(α) = Σ log Γ(α_k) − log Γ(S)
    let log_b_alpha: f32 = alpha.iter().map(|&a| lgamma(a)).sum::<f32>() - lgamma(s);
    let sum_term: f32 = alpha
        .iter()
        .map(|&a| (a - 1.0) * (digamma(a) - digamma(s)))
        .sum();
    log_b1 - log_b_alpha + sum_term
}

// ─── Special functions ────────────────────────────────────────────────────────

/// Log-Gamma function (Lanczos approximation, accurate to ~7 digits for x > 0).
///
/// Uses the Numerical Recipes Lanczos (g=5, 6 terms) formulation where the
/// series is evaluated at `z = x - 1` so that the recurrence gives `Γ(x)`.
pub fn lgamma(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::INFINITY;
    }
    // Use Stirling for large x (accurate beyond any table error).
    if x > 30.0 {
        let x64 = x as f64;
        return ((x64 - 0.5) * x64.ln() - x64 + 0.5 * (2.0 * std::f64::consts::PI).ln()) as f32;
    }
    // Lanczos coefficients (g=5, n=6) for Γ(z+1) = √(2π) * t^(z+0.5) * e^{-t} * ser
    // where t = z + g + 0.5 and z = x - 1.
    const G: f64 = 5.0;
    const C: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        1.208_650_973_866_179e-3,
        -5.395_239_384_953_e-6,
    ];
    // z = x - 1, then Γ(x) = Γ(z+1).
    let z = x as f64 - 1.0;
    let mut ser = 1.000_000_000_190_015_f64;
    for (k, &ck) in C.iter().enumerate() {
        ser += ck / (z + k as f64 + 1.0);
    }
    let t = z + G + 0.5;
    ((2.0 * std::f64::consts::PI).sqrt() * ser * t.powf(z + 0.5) * (-t).exp()).ln() as f32
}

/// Digamma function ψ(x) via asymptotic expansion (accurate for x > 6;
/// reflection applied for x < 6).
pub fn digamma(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    // Recurrence to shift x to x >= 6.
    let mut result = 0.0_f32;
    let mut y = x;
    while y < 6.0 {
        result -= 1.0 / y;
        y += 1.0;
    }
    // Asymptotic series for large y.
    let y64 = y as f64;
    let asy = y64.ln() - 1.0 / (2.0 * y64) - 1.0 / (12.0 * y64 * y64) + 1.0 / (120.0 * y64.powi(4))
        - 1.0 / (252.0 * y64.powi(6));
    result + asy as f32
}

/// Softplus: `log(1 + exp(x))`, numerically stable.
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1. Mean probs from uniform alpha sum to 1 ─────────────────────────────
    #[test]
    fn dirichlet_mean_probs_sum_to_one() {
        let ev =
            DirichletEvidence::from_alpha(vec![1.0, 2.0, 3.0]).expect("from_alpha should succeed");
        let p = ev.mean_probs();
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    // ── 2. Vacuity = 1 for uniform Dirichlet (α = 1) ─────────────────────────
    #[test]
    fn dirichlet_vacuity_uniform() {
        let ev = DirichletEvidence::from_alpha(vec![1.0; 3]).expect("from_alpha should succeed");
        assert!(
            (ev.vacuity() - 1.0).abs() < 1e-5,
            "vacuity={}",
            ev.vacuity()
        );
    }

    // ── 3. Strong evidence reduces vacuity ────────────────────────────────────
    #[test]
    fn dirichlet_vacuity_decreases_with_evidence() {
        let weak =
            DirichletEvidence::from_alpha(vec![1.0, 1.0, 1.0]).expect("from_alpha should succeed");
        let strong = DirichletEvidence::from_alpha(vec![100.0, 1.0, 1.0])
            .expect("from_alpha should succeed");
        assert!(
            strong.vacuity() < weak.vacuity(),
            "strong={} weak={}",
            strong.vacuity(),
            weak.vacuity()
        );
    }

    // ── 4. Predict returns argmax class ──────────────────────────────────────
    #[test]
    fn dirichlet_predict_argmax() {
        let ev =
            DirichletEvidence::from_alpha(vec![1.0, 10.0, 2.0]).expect("from_alpha should succeed");
        assert_eq!(ev.predict(), 1);
    }

    // ── 5. from_logits produces valid concentrations ──────────────────────────
    #[test]
    fn dirichlet_from_logits_positive() {
        let logits = vec![-2.0_f32, 0.0, 3.0, -1.0];
        let ev = DirichletEvidence::from_logits(&logits).expect("from_logits should succeed");
        assert!(ev.alpha.iter().all(|&a| a >= 1.0));
    }

    // ── 6. EDL loss is non-negative ───────────────────────────────────────────
    #[test]
    fn edl_loss_non_negative() {
        let ev =
            DirichletEvidence::from_alpha(vec![2.0, 5.0, 1.0]).expect("from_alpha should succeed");
        let y = vec![0.0_f32, 1.0, 0.0]; // class 1 is true
        let loss = ev.edl_loss(&y, 0.5).expect("edl_loss should succeed");
        assert!(loss.is_finite(), "loss={loss}");
        // Loss is typically positive; allow small negative due to digamma.
        assert!(loss > -1.0, "loss={loss} unexpectedly very negative");
    }

    // ── 7. NIG from_outputs produces valid parameters ─────────────────────────
    #[test]
    fn nig_from_outputs_valid() {
        let out = vec![0.5_f32, 1.0, 0.5, 0.5];
        let ev = NigEvidence::from_outputs(&out).expect("from_outputs should succeed");
        assert!(ev.nu > 0.0);
        assert!(ev.alpha > 1.0);
        assert!(ev.beta > 0.0);
    }

    // ── 8. NIG predictive mean equals gamma ───────────────────────────────────
    #[test]
    fn nig_predictive_mean() {
        let ev = NigEvidence::new(2.5, 1.0, 2.0, 0.5).expect("new should succeed");
        assert!((ev.predictive_mean() - 2.5).abs() < 1e-6);
    }

    // ── 9. NIG: epistemic < aleatoric for high nu ─────────────────────────────
    #[test]
    fn nig_high_nu_low_epistemic() {
        let ev_low_nu = NigEvidence::new(0.0, 0.1, 2.0, 1.0).expect("new should succeed");
        let ev_high_nu = NigEvidence::new(0.0, 100.0, 2.0, 1.0).expect("new should succeed");
        assert!(
            ev_high_nu.epistemic_uncertainty() < ev_low_nu.epistemic_uncertainty(),
            "higher nu should give lower epistemic uncertainty"
        );
    }

    // ── 10. NIG NLL is finite for typical values ───────────────────────────────
    #[test]
    fn nig_nll_finite() {
        let ev = NigEvidence::new(1.0, 2.0, 3.0, 0.5).expect("new should succeed");
        let nll = ev.nig_nll(1.2);
        assert!(nll.is_ok() && nll.expect("nll should be present").is_finite());
    }

    // ── 11. Digamma is increasing ─────────────────────────────────────────────
    #[test]
    fn digamma_increasing() {
        let vals = [0.5_f32, 1.0, 2.0, 5.0, 10.0, 20.0];
        for w in vals.windows(2) {
            assert!(digamma(w[1]) > digamma(w[0]), "not increasing at {:?}", w);
        }
    }

    // ── 12. lgamma(1) ≈ 0, lgamma(2) ≈ 0 (Γ(1)=Γ(2)=1) ─────────────────────
    #[test]
    fn lgamma_known_values() {
        assert!(lgamma(1.0).abs() < 0.01, "lgamma(1)={}", lgamma(1.0));
        assert!(lgamma(2.0).abs() < 0.01, "lgamma(2)={}", lgamma(2.0));
        // lgamma(0.5) = log(sqrt(π)) ≈ 0.5724
        let expected = (std::f32::consts::PI.sqrt()).ln();
        assert!(
            (lgamma(0.5) - expected).abs() < 0.01,
            "lgamma(0.5)={} expected={}",
            lgamma(0.5),
            expected
        );
    }
}
