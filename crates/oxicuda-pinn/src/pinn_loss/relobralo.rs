//! ReLoBRaLo — Relative Loss Balancing with Random Lookback.
//!
//! Bischof & Kraus (2021) "Multi-Objective Loss Balancing for Physics-Informed
//! Deep Learning" (arXiv:2110.09813).
//!
//! A PINN minimises a weighted sum of competing objectives (PDE residual,
//! boundary, initial, data, …):
//!
//! ```text
//! L(t) = Σ_i λ_i(t) · L_i(t) .
//! ```
//!
//! Picking the scalarisation weights `λ_i` by hand is brittle. ReLoBRaLo sets them
//! automatically from the **relative training progress** of each term: a term that
//! has improved little relative to a reference step is up-weighted, a term that has
//! already improved a lot is down-weighted. Concretely, given a reference step `t'`
//! and a temperature `T`, the balanced weights are a temperature-scaled softmax of
//! the per-term loss ratios, normalised so they sum to the number of terms `n`:
//!
//! ```text
//! λ̂_i(t; t') = n · softmax_i( L_i(t) / (T · L_i(t')) )
//!            = n · exp( L_i(t)/(T·L_i(t')) ) / Σ_j exp( L_j(t)/(T·L_j(t')) ) .
//! ```
//!
//! **Random lookback.** Always referencing the *initial* losses forgets recent
//! dynamics; always referencing the *previous* step is noisy. ReLoBRaLo interpolates
//! between the two with a Bernoulli random variable `ρ ∈ {0, 1}` drawn each step
//! (with `E[ρ]` close to 1, i.e. mostly long memory):
//!
//! ```text
//! λ^bal_i(t) = ρ · λ̂_i(t; 0) + (1 − ρ) · λ̂_i(t; t−1) .
//! ```
//!
//! **Exponential moving average.** Finally the live weights are smoothed with rate
//! `α ∈ [0, 1]` to damp oscillations:
//!
//! ```text
//! λ_i(t) = α · λ_i(t−1) + (1 − α) · λ^bal_i(t) .
//! ```
//!
//! Special cases: `α = 0` disables the EMA (pure per-step balancing); forcing
//! `ρ ≡ 1` recovers plain *relative loss balancing* against the initial losses; and
//! equal relative progress across all terms yields the uniform softmax, hence
//! `λ_i ≡ 1` (the unweighted multi-objective sum).

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;

/// Configuration for ReLoBRaLo weight balancing.
#[derive(Debug, Clone)]
pub struct ReloBraLoConfig {
    /// Number of loss terms `n` (`>= 1`).
    pub n_terms: usize,
    /// Softmax temperature `T` (`> 0`). Larger `T` ⇒ softer, more uniform weights.
    pub temperature: f32,
    /// EMA rate `α ∈ [0, 1]`. `0` disables smoothing; closer to `1` ⇒ slower change.
    pub alpha: f32,
    /// Probability that the Bernoulli lookback `ρ = 1` (reference the initial step).
    /// Typical values are close to `1` (e.g. `0.999`). Must lie in `[0, 1]`.
    pub rho_prob: f32,
}

impl ReloBraLoConfig {
    /// Convenience constructor with the paper's defaults (`T = 1`, `α = 0.999`,
    /// `E[ρ] = 0.999`).
    pub fn new(n_terms: usize) -> Self {
        Self {
            n_terms,
            temperature: 1.0,
            alpha: 0.999,
            rho_prob: 0.999,
        }
    }
}

/// ReLoBRaLo adaptive multi-objective loss balancer.
///
/// Holds the initial losses `L_i(0)`, the previous-step losses `L_i(t−1)`, and the
/// live EMA weights `λ_i(t)`. Call [`Self::step`] (or [`Self::step_with_rho`] for a
/// deterministic lookback) once per optimisation iteration with the current
/// unweighted term losses.
#[derive(Debug, Clone)]
pub struct ReloBraLo {
    config: ReloBraLoConfig,
    /// Live balancing weights `λ_i(t)`; start at `1`.
    weights: Vec<f32>,
    /// Initial losses `L_i(0)`; captured on the first `step`.
    initial: Vec<f32>,
    /// Previous-step losses `L_i(t−1)`.
    previous: Vec<f32>,
    /// Whether the first step has been taken (initial/previous are populated).
    started: bool,
}

impl ReloBraLo {
    /// Construct a balancer with all weights initialised to `1`.
    ///
    /// # Errors
    /// - [`PinnError::InvalidLayerWidth`] if `n_terms == 0`.
    /// - [`PinnError::InvalidPdeCoefficient`] if `temperature` is not finite or `<= 0`.
    /// - [`PinnError::InvalidWeight`] if `alpha` or `rho_prob` lie outside `[0, 1]`.
    pub fn new(config: ReloBraLoConfig) -> PinnResult<Self> {
        if config.n_terms == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        if !config.temperature.is_finite() || config.temperature <= 0.0 {
            return Err(PinnError::InvalidPdeCoefficient {
                name: "temperature",
                value: config.temperature,
            });
        }
        if !config.alpha.is_finite() || !(0.0..=1.0).contains(&config.alpha) {
            return Err(PinnError::InvalidWeight {
                weight: config.alpha,
            });
        }
        if !config.rho_prob.is_finite() || !(0.0..=1.0).contains(&config.rho_prob) {
            return Err(PinnError::InvalidWeight {
                weight: config.rho_prob,
            });
        }
        let n = config.n_terms;
        Ok(Self {
            config,
            weights: vec![1.0_f32; n],
            initial: vec![1.0_f32; n],
            previous: vec![1.0_f32; n],
            started: false,
        })
    }

    /// Current live balancing weights `λ_i(t)`.
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Number of loss terms `n`.
    #[must_use]
    pub fn n_terms(&self) -> usize {
        self.config.n_terms
    }

    /// Temperature-scaled, sum-to-`n` softmax of the loss ratios `L_i(t)/(T·L_i(ref))`.
    ///
    /// Uses the usual max-shift for numerical stability. A reference loss `<= 0`
    /// (or non-finite) is treated as a tiny positive epsilon so the ratio stays finite.
    fn balanced_against(&self, current: &[f32], reference: &[f32]) -> Vec<f32> {
        let n = self.config.n_terms;
        let t = self.config.temperature;
        let eps = 1e-12_f32;
        // logits_i = L_i(t) / (T · L_i(ref))
        let logits: Vec<f32> = (0..n)
            .map(|i| {
                let denom = (t * reference[i]).max(eps);
                current[i] / denom
            })
            .collect();
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let scale = n as f32 / sum.max(eps);
        exps.iter().map(|&e| e * scale).collect()
    }

    /// Advance one step with an explicit lookback value `rho ∈ [0, 1]`.
    ///
    /// `rho = 1` references the initial losses; `rho = 0` references the previous
    /// step; intermediate values blend the two. On the very first call the initial
    /// and previous losses are captured and the weights remain `1` (no history yet).
    ///
    /// Returns the updated weights `λ_i(t)`.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `losses.len() != n_terms`.
    /// - [`PinnError::InvalidWeight`] if `rho` is not finite or outside `[0, 1]`.
    /// - [`PinnError::NanEncountered`] if any input loss or resulting weight is not finite.
    pub fn step_with_rho(&mut self, losses: &[f32], rho: f32) -> PinnResult<Vec<f32>> {
        if losses.len() != self.config.n_terms {
            return Err(PinnError::DimensionMismatch {
                expected: self.config.n_terms,
                got: losses.len(),
            });
        }
        if !rho.is_finite() || !(0.0..=1.0).contains(&rho) {
            return Err(PinnError::InvalidWeight { weight: rho });
        }
        if losses.iter().any(|v| !v.is_finite()) {
            return Err(PinnError::NanEncountered {
                location: "relobralo::step(input)",
            });
        }

        if !self.started {
            // First step: seed history, keep λ = 1 (no relative progress yet).
            self.initial.copy_from_slice(losses);
            self.previous.copy_from_slice(losses);
            self.started = true;
            return Ok(self.weights.clone());
        }

        let against_init = self.balanced_against(losses, &self.initial);
        let against_prev = self.balanced_against(losses, &self.previous);

        let alpha = self.config.alpha;
        for i in 0..self.config.n_terms {
            let bal = rho * against_init[i] + (1.0 - rho) * against_prev[i];
            let updated = alpha * self.weights[i] + (1.0 - alpha) * bal;
            if !updated.is_finite() {
                return Err(PinnError::NanEncountered {
                    location: "relobralo::step(weight)",
                });
            }
            self.weights[i] = updated;
        }
        self.previous.copy_from_slice(losses);
        Ok(self.weights.clone())
    }

    /// Advance one step, drawing the Bernoulli lookback `ρ` from the supplied RNG
    /// with `P(ρ = 1) = rho_prob`.
    ///
    /// # Errors
    /// See [`Self::step_with_rho`].
    pub fn step(&mut self, losses: &[f32], rng: &mut LcgRng) -> PinnResult<Vec<f32>> {
        let rho = if rng.next_f32() < self.config.rho_prob {
            1.0
        } else {
            0.0
        };
        self.step_with_rho(losses, rho)
    }

    /// Total balanced loss `L = Σ_i λ_i(t) · L_i(t)` using the current weights.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if `losses.len() != n_terms`.
    /// - [`PinnError::NanEncountered`] if the result is not finite.
    pub fn total_loss(&self, losses: &[f32]) -> PinnResult<f32> {
        if losses.len() != self.config.n_terms {
            return Err(PinnError::DimensionMismatch {
                expected: self.config.n_terms,
                got: losses.len(),
            });
        }
        let total: f32 = self
            .weights
            .iter()
            .zip(losses.iter())
            .map(|(&w, &l)| w * l)
            .sum();
        if !total.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "relobralo::total_loss",
            });
        }
        Ok(total)
    }

    /// Reset weights to `1` and forget all loss history.
    pub fn reset(&mut self) {
        for w in &mut self.weights {
            *w = 1.0;
        }
        for v in &mut self.initial {
            *v = 1.0;
        }
        for v in &mut self.previous {
            *v = 1.0;
        }
        self.started = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn make(n: usize) -> ReloBraLo {
        ReloBraLo::new(ReloBraLoConfig::new(n))
            .expect("ReloBraLo construction with valid params should succeed")
    }

    // ── construction / validation ──────────────────────────────────────────────────
    #[test]
    fn construct_initial_weights_are_one() {
        let r = make(3);
        assert_eq!(r.weights(), &[1.0, 1.0, 1.0]);
        assert_eq!(r.n_terms(), 3);
    }

    #[test]
    fn construction_validation() {
        assert!(matches!(
            ReloBraLo::new(ReloBraLoConfig::new(0)),
            Err(PinnError::InvalidLayerWidth)
        ));
        assert!(matches!(
            ReloBraLo::new(ReloBraLoConfig {
                n_terms: 2,
                temperature: 0.0,
                alpha: 0.5,
                rho_prob: 0.9,
            }),
            Err(PinnError::InvalidPdeCoefficient { .. })
        ));
        assert!(matches!(
            ReloBraLo::new(ReloBraLoConfig {
                n_terms: 2,
                temperature: 1.0,
                alpha: 1.5,
                rho_prob: 0.9,
            }),
            Err(PinnError::InvalidWeight { .. })
        ));
        assert!(matches!(
            ReloBraLo::new(ReloBraLoConfig {
                n_terms: 2,
                temperature: 1.0,
                alpha: 0.5,
                rho_prob: -0.1,
            }),
            Err(PinnError::InvalidWeight { .. })
        ));
    }

    // ── LOAD-BEARING: the scaling identity Σ_i λ̂_i = n (softmax sums to n) ───────────
    #[test]
    fn balanced_weights_sum_to_n() {
        // α = 0 so the EMA passes the raw balanced weights straight through.
        let cfg = ReloBraLoConfig {
            n_terms: 4,
            temperature: 1.0,
            alpha: 0.0,
            rho_prob: 1.0,
        };
        let mut r =
            ReloBraLo::new(cfg).expect("ReloBraLo construction with valid params should succeed");
        // Seed history.
        r.step_with_rho(&[1.0, 2.0, 3.0, 4.0], 1.0)
            .expect("seeding step_with_rho should succeed for valid input");
        // A second, different step → non-trivial weights.
        let w = r
            .step_with_rho(&[0.5, 4.0, 1.0, 8.0], 1.0)
            .expect("step_with_rho should succeed for valid input");
        let sum: f32 = w.iter().sum();
        assert!(
            approx(sum, 4.0, 1e-3),
            "ReLoBRaLo weights must sum to n_terms = 4, got {sum}"
        );
        assert!(w.iter().all(|&x| x > 0.0), "weights must be positive");
    }

    // ── LOAD-BEARING: equal relative progress ⇒ uniform weights (all 1) ──────────────
    #[test]
    fn equal_relative_progress_gives_uniform_weights() {
        // Every term shrinks by the same factor ⇒ identical ratios ⇒ uniform softmax
        // ⇒ each λ̂ = n/n = 1.
        let cfg = ReloBraLoConfig {
            n_terms: 3,
            temperature: 1.0,
            alpha: 0.0,
            rho_prob: 1.0,
        };
        let mut r =
            ReloBraLo::new(cfg).expect("ReloBraLo construction with valid params should succeed");
        r.step_with_rho(&[2.0, 4.0, 8.0], 1.0)
            .expect("seeding step_with_rho should succeed for valid input"); // initial
        // Halve every term: ratios L(t)/L(0) all equal 0.5.
        let w = r
            .step_with_rho(&[1.0, 2.0, 4.0], 1.0)
            .expect("step_with_rho should succeed for valid input");
        for (i, &wi) in w.iter().enumerate() {
            assert!(
                approx(wi, 1.0, 1e-4),
                "equal relative progress ⇒ λ_{i} = 1, got {wi}"
            );
        }
    }

    // ── LOAD-BEARING: a lagging term is up-weighted relative to a fast one ───────────
    #[test]
    fn lagging_term_is_up_weighted() {
        // Term 0 barely improves (ratio ≈ 1), term 1 improves a lot (ratio ≪ 1).
        // Larger ratio ⇒ larger softmax logit ⇒ larger weight ⇒ λ_0 > λ_1.
        let cfg = ReloBraLoConfig {
            n_terms: 2,
            temperature: 1.0,
            alpha: 0.0,
            rho_prob: 1.0,
        };
        let mut r =
            ReloBraLo::new(cfg).expect("ReloBraLo construction with valid params should succeed");
        r.step_with_rho(&[1.0, 1.0], 1.0)
            .expect("seeding step_with_rho should succeed for valid input"); // initial
        let w = r
            .step_with_rho(&[0.95, 0.05], 1.0)
            .expect("step_with_rho should succeed for valid input");
        assert!(
            w[0] > w[1],
            "lagging term 0 should be up-weighted: λ_0={} λ_1={}",
            w[0],
            w[1]
        );
        assert!(approx(w[0] + w[1], 2.0, 1e-4));
    }

    // ── LOAD-BEARING: temperature softens the weight spread ──────────────────────────
    #[test]
    fn higher_temperature_softens_weights() {
        let losses_init = [1.0_f32, 1.0];
        let losses_now = [1.0_f32, 0.1]; // term 1 improved much more
        let spread = |temp: f32| -> f32 {
            let cfg = ReloBraLoConfig {
                n_terms: 2,
                temperature: temp,
                alpha: 0.0,
                rho_prob: 1.0,
            };
            let mut r = ReloBraLo::new(cfg)
                .expect("ReloBraLo construction with valid params should succeed");
            r.step_with_rho(&losses_init, 1.0)
                .expect("seeding step_with_rho should succeed for valid input");
            let w = r
                .step_with_rho(&losses_now, 1.0)
                .expect("step_with_rho should succeed for valid input");
            (w[0] - w[1]).abs()
        };
        let spread_cold = spread(0.5);
        let spread_hot = spread(5.0);
        assert!(
            spread_hot < spread_cold,
            "higher temperature should reduce weight spread: hot={spread_hot} cold={spread_cold}"
        );
    }

    // ── LOAD-BEARING: random-lookback ρ interpolates init vs previous balancing ──────
    #[test]
    fn rho_interpolates_between_init_and_previous() {
        // Build a state where referencing the initial step and the previous step give
        // different balanced weights, then check ρ=1, ρ=0 and ρ=0.5 line up.
        let base_cfg = |alpha: f32| ReloBraLoConfig {
            n_terms: 2,
            temperature: 1.0,
            alpha,
            rho_prob: 1.0,
        };
        // Helper: drive identical history, then take a final step at a given ρ.
        let final_weights = |rho: f32| -> Vec<f32> {
            let mut r = ReloBraLo::new(base_cfg(0.0))
                .expect("ReloBraLo construction with valid params should succeed");
            r.step_with_rho(&[1.0, 1.0], 1.0)
                .expect("seeding step_with_rho should succeed for valid input"); // initial = (1,1)
            r.step_with_rho(&[0.5, 0.9], 1.0)
                .expect("step_with_rho should succeed for valid input"); // previous becomes (0.5, 0.9)
            r.step_with_rho(&[0.4, 0.4], rho)
                .expect("final step_with_rho should succeed for valid input") // final step at chosen ρ
        };
        let w_init = final_weights(1.0); // referenced (1,1)
        let w_prev = final_weights(0.0); // referenced (0.5,0.9)
        let w_mid = final_weights(0.5); // blend
        for i in 0..2 {
            let blended = 0.5 * w_init[i] + 0.5 * w_prev[i];
            assert!(
                approx(w_mid[i], blended, 1e-4),
                "ρ=0.5 weight[{i}]={} must equal mean of ρ=1 ({}) and ρ=0 ({})",
                w_mid[i],
                w_init[i],
                w_prev[i]
            );
        }
        // The two references genuinely differ for this setup.
        assert!(
            (w_init[0] - w_prev[0]).abs() > 1e-4,
            "init and previous references should differ"
        );
    }

    // ── EMA: α blends previous weights with the new balanced weights ─────────────────
    #[test]
    fn ema_blends_old_and_new_weights() {
        let cfg = ReloBraLoConfig {
            n_terms: 2,
            temperature: 1.0,
            alpha: 0.5,
            rho_prob: 1.0,
        };
        let mut r =
            ReloBraLo::new(cfg).expect("ReloBraLo construction with valid params should succeed");
        r.step_with_rho(&[1.0, 1.0], 1.0)
            .expect("seeding step_with_rho should succeed for valid input"); // weights stay (1,1)
        let w = r
            .step_with_rho(&[0.9, 0.1], 1.0)
            .expect("step_with_rho should succeed for valid input");
        // λ = 0.5·1 + 0.5·λ̂ ; with sum-to-2 balanced weights, the EMA result also
        // averages to 1 (0.5·2 + 0.5·2)/2 = 2 total.
        let sum: f32 = w.iter().sum();
        assert!(
            approx(sum, 2.0, 1e-3),
            "EMA of weights summing to 2 stays 2"
        );
        // Each lies strictly between the old weight (1) and the raw balanced weight.
        assert!(
            w[0] > 1.0 && w[1] < 1.0,
            "EMA moved toward balanced weights"
        );
    }

    // ── first step keeps weights = 1 (no history) ────────────────────────────────────
    #[test]
    fn first_step_keeps_unit_weights() {
        let mut r = make(3);
        let w = r
            .step_with_rho(&[5.0, 0.1, 2.0], 1.0)
            .expect("step_with_rho should succeed for valid input");
        assert_eq!(w, vec![1.0, 1.0, 1.0], "first step has no relative history");
    }

    // ── total loss uses current weights ──────────────────────────────────────────────
    #[test]
    fn total_loss_is_weighted_sum() {
        let mut r = make(2);
        r.step_with_rho(&[1.0, 1.0], 1.0)
            .expect("seeding step_with_rho should succeed for valid input");
        // After the seeding step weights are (1,1) ⇒ total = sum of losses.
        let total = r
            .total_loss(&[0.3, 0.7])
            .expect("total_loss computation should succeed for valid input");
        assert!(
            approx(total, 1.0, 1e-6),
            "unit weights ⇒ plain sum, got {total}"
        );
    }

    #[test]
    fn total_loss_reflects_updated_weights() {
        let cfg = ReloBraLoConfig {
            n_terms: 2,
            temperature: 1.0,
            alpha: 0.0,
            rho_prob: 1.0,
        };
        let mut r =
            ReloBraLo::new(cfg).expect("ReloBraLo construction with valid params should succeed");
        r.step_with_rho(&[1.0, 1.0], 1.0)
            .expect("seeding step_with_rho should succeed for valid input");
        let cur = [0.9_f32, 0.1];
        let w = r
            .step_with_rho(&cur, 1.0)
            .expect("step_with_rho should succeed for valid input");
        let expected: f32 = w.iter().zip(cur.iter()).map(|(&a, &b)| a * b).sum();
        let total = r
            .total_loss(&cur)
            .expect("total_loss computation should succeed for valid input");
        assert!(approx(total, expected, 1e-5));
    }

    // ── stochastic step is deterministic given the RNG seed and stays valid ──────────
    #[test]
    fn stochastic_step_deterministic_and_valid() {
        let run = || -> Vec<f32> {
            let mut rng = LcgRng::new(42);
            let cfg = ReloBraLoConfig {
                n_terms: 3,
                temperature: 1.0,
                alpha: 0.0,
                rho_prob: 0.7, // genuinely stochastic lookback
            };
            let mut r = ReloBraLo::new(cfg)
                .expect("ReloBraLo construction with valid params should succeed");
            r.step(&[1.0, 2.0, 3.0], &mut rng)
                .expect("seeding stochastic step should succeed for valid input");
            let mut last = vec![];
            for k in 1..6 {
                let f = 0.9_f32.powi(k);
                last = r
                    .step(&[1.0 * f, 2.0 * f * f, 3.0 * f], &mut rng)
                    .expect("stochastic step should succeed for valid input");
            }
            last
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "same seed ⇒ identical weight trajectory");
        let sum: f32 = a.iter().sum();
        assert!(approx(sum, 3.0, 1e-2), "weights sum to n=3, got {sum}");
        assert!(a.iter().all(|&w| w.is_finite() && w > 0.0));
    }

    // ── ρ ≡ 1 reduces to relative loss balancing against the INITIAL losses ──────────
    #[test]
    fn rho_one_references_initial_losses() {
        let cfg = ReloBraLoConfig {
            n_terms: 2,
            temperature: 1.0,
            alpha: 0.0,
            rho_prob: 1.0,
        };
        // Path A: take an intermediate step, then a final step with ρ=1.
        let mut ra = ReloBraLo::new(cfg.clone())
            .expect("ReloBraLo construction with valid params should succeed");
        ra.step_with_rho(&[1.0, 1.0], 1.0)
            .expect("seeding step_with_rho should succeed for valid input"); // initial=(1,1)
        ra.step_with_rho(&[0.7, 0.2], 1.0)
            .expect("step_with_rho should succeed for valid input"); // previous becomes (0.7,0.2)
        let wa = ra
            .step_with_rho(&[0.5, 0.5], 1.0)
            .expect("final step_with_rho should succeed for valid input");

        // Path B: balance (0.5,0.5) directly against the SAME initial (1,1).
        let mut rb =
            ReloBraLo::new(cfg).expect("ReloBraLo construction with valid params should succeed");
        rb.step_with_rho(&[1.0, 1.0], 1.0)
            .expect("seeding step_with_rho should succeed for valid input");
        let wb = rb
            .step_with_rho(&[0.5, 0.5], 1.0)
            .expect("step_with_rho against initial losses should succeed for valid input");

        for i in 0..2 {
            assert!(
                approx(wa[i], wb[i], 1e-4),
                "ρ=1 must reference initial losses only: {} vs {}",
                wa[i],
                wb[i]
            );
        }
    }

    // ── reset restores the pristine state ────────────────────────────────────────────
    #[test]
    fn reset_restores_unit_state() {
        let cfg = ReloBraLoConfig {
            n_terms: 2,
            temperature: 1.0,
            alpha: 0.0,
            rho_prob: 1.0,
        };
        let mut r =
            ReloBraLo::new(cfg).expect("ReloBraLo construction with valid params should succeed");
        r.step_with_rho(&[1.0, 1.0], 1.0)
            .expect("seeding step_with_rho should succeed for valid input");
        r.step_with_rho(&[0.1, 0.9], 1.0)
            .expect("step_with_rho should succeed for valid input");
        r.reset();
        assert_eq!(r.weights(), &[1.0, 1.0]);
        // After reset the next step behaves like a fresh first step.
        let w = r
            .step_with_rho(&[3.0, 0.2], 1.0)
            .expect("step_with_rho after reset should succeed for valid input");
        assert_eq!(w, vec![1.0, 1.0]);
    }

    // ── dimension / finiteness guards ────────────────────────────────────────────────
    #[test]
    fn step_dimension_mismatch_errors() {
        let mut r = make(3);
        assert!(matches!(
            r.step_with_rho(&[1.0, 2.0], 1.0),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn step_rejects_bad_rho_and_nan_loss() {
        let mut r = make(2);
        assert!(matches!(
            r.step_with_rho(&[1.0, 1.0], 1.5),
            Err(PinnError::InvalidWeight { .. })
        ));
        let mut r2 = make(2);
        assert!(matches!(
            r2.step_with_rho(&[1.0, f32::NAN], 1.0),
            Err(PinnError::NanEncountered { .. })
        ));
    }

    #[test]
    fn total_loss_dimension_mismatch_errors() {
        let r = make(2);
        assert!(matches!(
            r.total_loss(&[1.0]),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn weights_stay_finite_over_long_run() {
        let mut rng = LcgRng::new(9);
        let mut r = make(3);
        for k in 0..200 {
            let f = 0.99_f32.powi(k);
            let w = r
                .step(&[1.0 * f, 0.5 * f, 2.0 * f], &mut rng)
                .expect("stochastic step should succeed for valid input");
            assert!(w.iter().all(|&v| v.is_finite() && v >= 0.0));
        }
    }
}
