//! AutoAttack ensemble (Croce & Hein 2020 ICML).
//!
//! A parameter-free, reliable adversarial attack combining APGD-CE + APGD-DLR
//! (untargeted) + Square Attack in sequence. Returns the best adversarial
//! example found (first one that changes the class).
//!
//! APGD-DLR uses the Difference-of-Logits-Ratio loss:
//!   `L_DLR(f(x), y) = -(f_y - max_{k≠y} f_k) / (f_{π_1} - f_{π_3})`
//! where π_1, π_2, π_3 are indices sorted by decreasing logit value. This
//! loss is scale-invariant and does not saturate.
//!
//! Reference: Croce & Hein (2020), *"Reliable Evaluation of Adversarial
//! Robustness with an Ensemble of Diverse Parameter-free Attacks"*, ICML.

use crate::attacks::auto_pgd::{AutoPgdConfig, auto_pgd_attack};
use crate::attacks::square::{SquareAttackConfig, square_attack};
use crate::error::{AdvError, AdvResult};
use crate::handle::LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the AutoAttack ensemble.
#[derive(Debug, Clone, Copy)]
pub struct AutoAttackConfig {
    /// L∞ perturbation budget ε > 0.
    pub eps: f32,
    /// APGD-CE: number of steps.
    pub apgd_n_steps: usize,
    /// APGD-CE: checkpoint ratio.
    pub apgd_checkpoint_ratio: f32,
    /// Square Attack: number of queries.
    pub square_n_queries: usize,
    /// Box lower bound.
    pub lo: f32,
    /// Box upper bound.
    pub hi: f32,
}

impl Default for AutoAttackConfig {
    fn default() -> Self {
        Self {
            eps: 8.0 / 255.0,
            apgd_n_steps: 100,
            apgd_checkpoint_ratio: 0.22,
            square_n_queries: 1000,
            lo: 0.0,
            hi: 1.0,
        }
    }
}

// ─── DLR loss ─────────────────────────────────────────────────────────────────

/// Compute DLR loss for a given set of logits and true class index.
///
/// Returns `-(f_y - max_{k≠y} f_k) / (f_{π_1} - f_{π_3})` if n_classes ≥ 3.
/// For n_classes == 2: returns `-(f_y - f_{k≠y})` (simplified, denom=1).
/// Returns `None` if denominator ≈ 0 (degenerate case) or input is too small.
pub fn dlr_loss(logits: &[f32], true_class: usize) -> Option<f32> {
    let n = logits.len();
    if n < 2 || true_class >= n {
        return None;
    }

    // Two-class special case.
    if n == 2 {
        let k = 1 - true_class;
        return Some(-(logits[true_class] - logits[k]));
    }

    // Find π_1 = argmax(logits).
    let pi_1 = argmax_usize(logits);

    // Find k_star = argmax_{k ≠ true_class} logits[k].
    let mut k_star = usize::MAX;
    let mut k_star_val = f32::NEG_INFINITY;
    for (k, &lv) in logits.iter().enumerate() {
        if k != true_class && lv > k_star_val {
            k_star_val = lv;
            k_star = k;
        }
    }
    if k_star == usize::MAX {
        return None;
    }

    // Find π_3: index of the 3rd largest logit (0-indexed position 2).
    // Build a sorted list of (logit_value, index), descending.
    let mut indexed: Vec<(f32, usize)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    indexed.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let pi_3_idx = if indexed.len() >= 3 {
        indexed[2].1
    } else {
        indexed[indexed.len() - 1].1
    };

    let denom = logits[pi_1] - logits[pi_3_idx];
    if denom.abs() < 1e-8 {
        return None;
    }

    Some(-(logits[true_class] - logits[k_star]) / denom)
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Index of the maximum element. Assumes `v` is non-empty.
#[inline]
fn argmax_usize(v: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = v[0];
    for (i, &val) in v.iter().enumerate().skip(1) {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    best_idx
}

/// Finite-difference gradient of the DLR loss w.r.t. `x`.
///
/// For each coordinate j:
///   grad[j] = (dlr(logit_fn(x + eps_fd * e_j), tc) - dlr(logit_fn(x - eps_fd * e_j), tc)) / (2 * eps_fd)
fn dlr_grad_fd<FL>(x: &[f32], logit_fn: &FL, true_class: usize) -> AdvResult<Vec<f32>>
where
    FL: Fn(&[f32]) -> AdvResult<Vec<f32>>,
{
    let dim = x.len();
    let eps_fd = 1e-4_f32;
    let mut grad = vec![0.0_f32; dim];

    let mut x_p = x.to_vec();
    let mut x_m = x.to_vec();

    for j in 0..dim {
        x_p[j] = x[j] + eps_fd;
        x_m[j] = x[j] - eps_fd;

        let lp = logit_fn(&x_p)?;
        let lm = logit_fn(&x_m)?;

        let dlr_p = dlr_loss(&lp, true_class).unwrap_or(0.0);
        let dlr_m = dlr_loss(&lm, true_class).unwrap_or(0.0);

        grad[j] = (dlr_p - dlr_m) / (2.0 * eps_fd);

        // Restore.
        x_p[j] = x[j];
        x_m[j] = x[j];
    }

    Ok(grad)
}

// ─── Main attack ──────────────────────────────────────────────────────────────

/// Run the AutoAttack ensemble on input `x`.
///
/// `loss_grad_ce`: closure for cross-entropy loss and its gradient.
///   Takes `x` → `AdvResult<(loss: f32, grad: Vec<f32>)>`
///
/// `logit_fn`: closure for raw logits (used for DLR and Square).
///   Takes `x` → `AdvResult<Vec<f32>>`
///
/// `score_fn`: closure for Square Attack (correct-class score, lower = more adversarial).
///   Typically: `|x| Ok(-logit_fn(x)?[true_class])`.
///
/// Returns the adversarial example found by whichever component succeeded first,
/// or the input `x` perturbed by the last Square Attack result if none changed class.
///
/// # Errors
/// - [`AdvError::EmptyInput`] if `x` is empty.
/// - [`AdvError::InvalidEpsilon`] if eps ≤ 0 or non-finite.
/// - Propagates errors from component attacks.
pub fn autoattack<FG, FL, FS>(
    x: &[f32],
    true_class: usize,
    n_classes: usize,
    loss_grad_ce: FG,
    logit_fn: FL,
    score_fn: FS,
    cfg: &AutoAttackConfig,
    rng: &mut LcgRng,
) -> AdvResult<Vec<f32>>
where
    FG: Fn(&[f32]) -> AdvResult<(f32, Vec<f32>)>,
    FL: Fn(&[f32]) -> AdvResult<Vec<f32>>,
    FS: Fn(&[f32]) -> AdvResult<f32>,
{
    // ── Validation ────────────────────────────────────────────────────────────
    if x.is_empty() {
        return Err(AdvError::EmptyInput);
    }
    if !(cfg.eps.is_finite() && cfg.eps > 0.0) {
        return Err(AdvError::InvalidEpsilon { eps: cfg.eps });
    }
    if n_classes < 2 {
        return Err(AdvError::Internal("n_classes must be >= 2".to_owned()));
    }
    if true_class >= n_classes {
        return Err(AdvError::Internal(format!(
            "true_class={true_class} >= n_classes={n_classes}"
        )));
    }

    // Build APGD config shared by both APGD phases.
    let apgd_cfg = AutoPgdConfig {
        eps: cfg.eps,
        n_steps: cfg.apgd_n_steps,
        checkpoint_ratio: cfg.apgd_checkpoint_ratio,
    };

    // ── Step 1: APGD-CE (untargeted) ──────────────────────────────────────────
    let ce_grad_closure = |x_cur: &[f32]| -> AdvResult<(f32, Vec<f32>)> { loss_grad_ce(x_cur) };
    let adv_ce = auto_pgd_attack(x, cfg.lo, cfg.hi, &apgd_cfg, rng, ce_grad_closure)?;
    {
        let logits_ce = logit_fn(&adv_ce)?;
        if argmax_usize(&logits_ce) != true_class {
            return Ok(adv_ce);
        }
    }
    // APGD-CE did not flip class; continue to step 2.

    // ── Step 2: APGD-DLR (untargeted) ─────────────────────────────────────────
    let dlr_grad_closure = |x_cur: &[f32]| -> AdvResult<(f32, Vec<f32>)> {
        let grad = dlr_grad_fd(x_cur, &logit_fn, true_class)?;
        // Compute DLR loss value for the current point.
        let logits_cur = logit_fn(x_cur)?;
        let loss_val = dlr_loss(&logits_cur, true_class).unwrap_or(0.0);
        Ok((loss_val, grad))
    };
    let adv_dlr = auto_pgd_attack(x, cfg.lo, cfg.hi, &apgd_cfg, rng, dlr_grad_closure)?;
    {
        let logits_dlr = logit_fn(&adv_dlr)?;
        if argmax_usize(&logits_dlr) != true_class {
            return Ok(adv_dlr);
        }
    }
    // APGD-DLR did not flip class; continue to step 3.

    // ── Step 3: Square Attack ─────────────────────────────────────────────────
    // Square Attack is the final phase; return its result regardless of class change.
    let sq_cfg = SquareAttackConfig {
        eps: cfg.eps,
        n_queries: cfg.square_n_queries,
        init_window_frac: 0.5,
        untargeted: true,
    };
    square_attack(x, score_fn, &sq_cfg, cfg.lo, cfg.hi, rng)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── Helper: linear logit oracle ───────────────────────────────────────────
    // weights: [n_classes, dim] row-major; logits[k] = weights[k,:] · x + biases[k].
    fn linear_logit_fn(
        weights: Vec<f32>,
        biases: Vec<f32>,
        n_classes: usize,
        dim: usize,
    ) -> impl Fn(&[f32]) -> AdvResult<Vec<f32>> {
        move |x: &[f32]| {
            let mut logits = vec![0.0_f32; n_classes];
            for k in 0..n_classes {
                let mut dot = biases[k];
                for i in 0..dim {
                    dot += weights[k * dim + i] * x[i];
                }
                logits[k] = dot;
            }
            Ok(logits)
        }
    }

    // ── Helper: CE loss_grad from linear logit oracle ─────────────────────────
    // Cross-entropy gradient for class `true_class`: -(e_{true_class} - softmax) * weights.
    // Simplified: approximate as -grad of true-class logit (gradient ascent = away from true class).
    fn linear_loss_grad_ce(
        weights: Vec<f32>,
        biases: Vec<f32>,
        n_classes: usize,
        dim: usize,
        true_class: usize,
    ) -> impl Fn(&[f32]) -> AdvResult<(f32, Vec<f32>)> {
        move |x: &[f32]| {
            let mut logits = vec![0.0_f32; n_classes];
            for k in 0..n_classes {
                let mut dot = biases[k];
                for i in 0..dim {
                    dot += weights[k * dim + i] * x[i];
                }
                logits[k] = dot;
            }
            // Loss = -logits[true_class] (gradient ascent on true class logit
            // increases loss and drives the attack).
            let loss = -logits[true_class];
            // Gradient of -f_{true_class} w.r.t. x = -weights[true_class, :].
            let grad: Vec<f32> = (0..dim).map(|i| -weights[true_class * dim + i]).collect();
            Ok((loss, grad))
        }
    }

    // ── Test 1 ────────────────────────────────────────────────────────────────

    #[test]
    fn dlr_loss_two_classes() {
        // 2-class: for correctly classified (true_class=0, logits[0]>logits[1])
        // loss = -(f_0 - f_1) which is negative (margin positive).
        let logits = vec![2.0_f32, 0.5_f32];
        let result = dlr_loss(&logits, 0);
        assert!(result.is_some(), "expected Some for 2-class");
        let val = result.expect("result should be present");
        assert!(
            val < 0.0,
            "correctly classified 2-class DLR should be negative, got {val}"
        );
        // Expected: -(2.0 - 0.5) = -1.5.
        assert!((val - (-1.5_f32)).abs() < 1e-5, "expected -1.5, got {val}");
    }

    // ── Test 2 ────────────────────────────────────────────────────────────────

    #[test]
    fn dlr_loss_three_classes_correct() {
        // 3-class: correctly classified (true_class=0, logits[0] is largest).
        // logits = [3, 1, 0], true_class=0.
        // π_1=0 (max=3), k_star=1 (max k≠0 is idx 1, val 1).
        // π_3: sorted desc [3,1,0] → π_3=idx 2.
        // denom = logits[0] - logits[2] = 3-0 = 3.
        // loss = -(3 - 1) / 3 = -2/3 ≈ -0.6667 (negative = correctly classified).
        let logits = vec![3.0_f32, 1.0_f32, 0.0_f32];
        let val = dlr_loss(&logits, 0).expect("dlr_loss should succeed");
        assert!(
            val < 0.0,
            "correctly classified 3-class DLR should be negative, got {val}"
        );
        let expected = -2.0_f32 / 3.0;
        assert!(
            (val - expected).abs() < 1e-5,
            "expected {expected}, got {val}"
        );
    }

    // ── Test 3 ────────────────────────────────────────────────────────────────

    #[test]
    fn dlr_loss_three_classes_wrong_class() {
        // 3-class: misclassified (true_class=0 but logits[1] is largest).
        // logits = [0, 3, 1], true_class=0.
        // π_1=1 (max=3), k_star=2 (max k≠0 is idx 2, val 1; wait: idx 1=3, idx 2=1).
        // k_star = argmax_{k≠0} → k=1 (val=3) is larger than k=2 (val=1) → k_star=1.
        // π_3: sorted desc [3,1,0] at idx [1,2,0] → π_3=idx 0.
        // denom = logits[1] - logits[0] = 3 - 0 = 3.
        // loss = -(logits[0] - logits[1]) / 3 = -(0-3)/3 = 1.0 (positive = misclassified).
        let logits = vec![0.0_f32, 3.0_f32, 1.0_f32];
        let val = dlr_loss(&logits, 0).expect("dlr_loss should succeed");
        assert!(
            val > 0.0,
            "misclassified 3-class DLR should be positive, got {val}"
        );
    }

    // ── Test 4 ────────────────────────────────────────────────────────────────

    #[test]
    fn dlr_loss_returns_none_degenerate() {
        // Top logits are equal → denominator ≈ 0 → None.
        // logits = [5, 5, 4]: π_1=0 or 1 (tie — partial_cmp handles), π_3=2.
        // But π_1 and π_2 tie: denom = logits[pi_1] - logits[pi_3].
        // If π_1=0 (first max), sorted desc: [5,5,4] at idx [0,1,2] or [1,0,2].
        // denom = 5 - 4 = 1. That's not zero. Let's use [5, 5, 5]:
        // π_3 = idx of 3rd = some index, logits[pi_1] = 5, logits[pi_3] = 5.
        // denom = 0 → None.
        let logits = vec![5.0_f32, 5.0_f32, 5.0_f32];
        let result = dlr_loss(&logits, 0);
        assert!(
            result.is_none(),
            "degenerate (all equal) should return None"
        );
    }

    // ── Test 5 ────────────────────────────────────────────────────────────────

    #[test]
    fn autoattack_output_shape() {
        let dim = 4_usize;
        let n_classes = 2_usize;
        let true_class = 0_usize;
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0_f32];
        let biases = vec![2.0, 0.0_f32];

        let logit_fn = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let loss_grad =
            linear_loss_grad_ce(weights.clone(), biases.clone(), n_classes, dim, true_class);
        let logit_fn2 = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let score_fn = move |x: &[f32]| -> AdvResult<f32> {
            let logits = logit_fn2(x)?;
            Ok(-logits[true_class])
        };

        let x = vec![0.5_f32; dim];
        let cfg = AutoAttackConfig {
            eps: 0.1,
            apgd_n_steps: 10,
            square_n_queries: 20,
            ..Default::default()
        };
        let mut rng = LcgRng::new(42);
        let result = autoattack(
            &x, true_class, n_classes, loss_grad, logit_fn, score_fn, &cfg, &mut rng,
        )
        .expect("value should be present");
        assert_eq!(result.len(), dim);
    }

    // ── Test 6 ────────────────────────────────────────────────────────────────

    #[test]
    fn autoattack_linf_bound() {
        let dim = 4_usize;
        let n_classes = 2_usize;
        let true_class = 0_usize;
        let eps = 0.1_f32;
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0_f32];
        let biases = vec![2.0, 0.0_f32];

        let logit_fn = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let loss_grad =
            linear_loss_grad_ce(weights.clone(), biases.clone(), n_classes, dim, true_class);
        let logit_fn2 = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let score_fn = move |x: &[f32]| -> AdvResult<f32> {
            let logits = logit_fn2(x)?;
            Ok(-logits[true_class])
        };

        let x = vec![0.5_f32; dim];
        let cfg = AutoAttackConfig {
            eps,
            apgd_n_steps: 10,
            square_n_queries: 20,
            lo: 0.0,
            hi: 1.0,
            ..Default::default()
        };
        let mut rng = LcgRng::new(7);
        let result = autoattack(
            &x, true_class, n_classes, loss_grad, logit_fn, score_fn, &cfg, &mut rng,
        )
        .expect("value should be present");
        for (r, &xi) in result.iter().zip(x.iter()) {
            assert!(
                (r - xi).abs() <= eps + 1e-5,
                "L∞ bound violated: |{r} - {xi}| > {eps}"
            );
        }
    }

    // ── Test 7 ────────────────────────────────────────────────────────────────

    #[test]
    fn autoattack_box_constraint() {
        let dim = 4_usize;
        let n_classes = 2_usize;
        let true_class = 0_usize;
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0_f32];
        let biases = vec![2.0, 0.0_f32];

        let logit_fn = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let loss_grad =
            linear_loss_grad_ce(weights.clone(), biases.clone(), n_classes, dim, true_class);
        let logit_fn2 = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let score_fn = move |x: &[f32]| -> AdvResult<f32> {
            let logits = logit_fn2(x)?;
            Ok(-logits[true_class])
        };

        let x = vec![0.5_f32; dim];
        let lo = 0.0_f32;
        let hi = 1.0_f32;
        let cfg = AutoAttackConfig {
            eps: 0.1,
            apgd_n_steps: 10,
            square_n_queries: 20,
            lo,
            hi,
            ..Default::default()
        };
        let mut rng = LcgRng::new(3);
        let result = autoattack(
            &x, true_class, n_classes, loss_grad, logit_fn, score_fn, &cfg, &mut rng,
        )
        .expect("value should be present");
        for &v in &result {
            assert!(
                (lo - 1e-5..=hi + 1e-5).contains(&v),
                "box constraint violated: {v} not in [{lo}, {hi}]"
            );
        }
    }

    // ── Test 8 ────────────────────────────────────────────────────────────────

    #[test]
    fn autoattack_finds_adversarial_linear() {
        // Simple linear classifier: true_class=0 with bias=0.1 (barely wins).
        // APGD or Square should flip class with eps=0.5.
        let dim = 2_usize;
        let n_classes = 2_usize;
        let true_class = 0_usize;
        // class 0: w=[1,0]+bias=0.1; class 1: w=[0,1]+bias=0.
        // x=[0.2, 0.0]: logits=[0.3, 0.0] → class 0.
        let weights = vec![1.0, 0.0, 0.0, 1.0_f32];
        let biases = vec![0.1, 0.0_f32];

        let logit_fn = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let loss_grad =
            linear_loss_grad_ce(weights.clone(), biases.clone(), n_classes, dim, true_class);
        let logit_fn2 = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let logit_fn3 = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let score_fn = move |x: &[f32]| -> AdvResult<f32> {
            let logits = logit_fn2(x)?;
            Ok(-logits[true_class])
        };

        let x = vec![0.2_f32, 0.0_f32];
        let cfg = AutoAttackConfig {
            eps: 0.5,
            apgd_n_steps: 50,
            square_n_queries: 200,
            lo: -1.0,
            hi: 1.0,
            ..Default::default()
        };
        let mut rng = LcgRng::new(42);
        let result = autoattack(
            &x, true_class, n_classes, loss_grad, logit_fn, score_fn, &cfg, &mut rng,
        )
        .expect("value should be present");
        let final_logits = logit_fn3(&result).expect("logit_fn3 should succeed");
        let final_class = argmax_usize(&final_logits);
        assert_ne!(
            final_class, true_class,
            "AutoAttack should have flipped class; logits={final_logits:?}"
        );
    }

    // ── Test 9 ────────────────────────────────────────────────────────────────

    #[test]
    fn autoattack_single_component_runs() {
        // Basic smoke test: runs on small dim=4 without error.
        let dim = 4_usize;
        let n_classes = 2_usize;
        let true_class = 0_usize;
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0_f32];
        let biases = vec![1.0, 0.0_f32];

        let logit_fn = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let loss_grad =
            linear_loss_grad_ce(weights.clone(), biases.clone(), n_classes, dim, true_class);
        let logit_fn2 = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let score_fn = move |x: &[f32]| -> AdvResult<f32> {
            let logits = logit_fn2(x)?;
            Ok(-logits[true_class])
        };

        let x = vec![0.5_f32; dim];
        let cfg = AutoAttackConfig {
            eps: 0.1,
            apgd_n_steps: 5,
            square_n_queries: 10,
            ..Default::default()
        };
        let mut rng = LcgRng::new(0);
        let result = autoattack(
            &x, true_class, n_classes, loss_grad, logit_fn, score_fn, &cfg, &mut rng,
        );
        assert!(result.is_ok(), "autoattack should complete without error");
        assert_eq!(result.expect("result should be present").len(), dim);
    }

    // ── Test 10 ───────────────────────────────────────────────────────────────

    #[test]
    fn autoattack_err_empty() {
        let x: Vec<f32> = vec![];
        let cfg = AutoAttackConfig::default();
        let mut rng = LcgRng::new(0);
        let loss_grad = |_: &[f32]| Ok((0.0_f32, vec![]));
        let logit_fn = |_: &[f32]| Ok(vec![1.0_f32, 0.0]);
        let score_fn = |_: &[f32]| Ok(0.0_f32);
        let result = autoattack(&x, 0, 2, loss_grad, logit_fn, score_fn, &cfg, &mut rng);
        assert!(matches!(result.unwrap_err(), AdvError::EmptyInput));
    }

    // ── Test 11 ───────────────────────────────────────────────────────────────

    #[test]
    fn autoattack_err_eps_zero() {
        let x = vec![0.5_f32; 4];
        let cfg = AutoAttackConfig {
            eps: 0.0,
            ..Default::default()
        };
        let mut rng = LcgRng::new(0);
        let loss_grad = |_: &[f32]| Ok((0.0_f32, vec![1.0_f32; 4]));
        let logit_fn = |_: &[f32]| Ok(vec![1.0_f32, 0.0]);
        let score_fn = |_: &[f32]| Ok(0.0_f32);
        let result = autoattack(&x, 0, 2, loss_grad, logit_fn, score_fn, &cfg, &mut rng);
        assert!(matches!(
            result.unwrap_err(),
            AdvError::InvalidEpsilon { .. }
        ));
    }

    // ── Test 12 ───────────────────────────────────────────────────────────────

    #[test]
    fn autoattack_deterministic_rng() {
        let dim = 4_usize;
        let n_classes = 2_usize;
        let true_class = 0_usize;
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0_f32];
        let biases = vec![2.0, 0.0_f32];

        let make_closures = || {
            let logit_fn = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
            let loss_grad =
                linear_loss_grad_ce(weights.clone(), biases.clone(), n_classes, dim, true_class);
            let logit_fn2 = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
            let score_fn = move |x: &[f32]| -> AdvResult<f32> {
                let logits = logit_fn2(x)?;
                Ok(-logits[true_class])
            };
            (logit_fn, loss_grad, score_fn)
        };

        let x = vec![0.5_f32; dim];
        let cfg = AutoAttackConfig {
            eps: 0.1,
            apgd_n_steps: 10,
            square_n_queries: 20,
            ..Default::default()
        };

        let (logit_fn1, loss_grad1, score_fn1) = make_closures();
        let (logit_fn2, loss_grad2, score_fn2) = make_closures();

        let mut rng1 = LcgRng::new(2024);
        let mut rng2 = LcgRng::new(2024);

        let res1 = autoattack(
            &x, true_class, n_classes, loss_grad1, logit_fn1, score_fn1, &cfg, &mut rng1,
        )
        .expect("value should be present");
        let res2 = autoattack(
            &x, true_class, n_classes, loss_grad2, logit_fn2, score_fn2, &cfg, &mut rng2,
        )
        .expect("value should be present");

        for (a, b) in res1.iter().zip(res2.iter()) {
            assert!((a - b).abs() < 1e-6, "same seed should produce same result");
        }
    }

    // ── Test 13 ───────────────────────────────────────────────────────────────

    #[test]
    fn autoattack_two_classes() {
        // Verify n_classes=2 works end-to-end.
        let dim = 4_usize;
        let n_classes = 2_usize;
        let true_class = 0_usize;
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0_f32];
        let biases = vec![1.5, 0.0_f32];

        let logit_fn = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let loss_grad =
            linear_loss_grad_ce(weights.clone(), biases.clone(), n_classes, dim, true_class);
        let logit_fn2 = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let score_fn = move |x: &[f32]| -> AdvResult<f32> {
            let logits = logit_fn2(x)?;
            Ok(-logits[true_class])
        };

        let x = vec![0.5_f32; dim];
        let cfg = AutoAttackConfig {
            eps: 0.1,
            apgd_n_steps: 10,
            square_n_queries: 20,
            ..Default::default()
        };
        let mut rng = LcgRng::new(5);
        let result = autoattack(
            &x, true_class, n_classes, loss_grad, logit_fn, score_fn, &cfg, &mut rng,
        );
        assert!(result.is_ok());
    }

    // ── Test 14 ───────────────────────────────────────────────────────────────

    #[test]
    fn autoattack_true_class_bounds() {
        // true_class=0 < n_classes=2 — should succeed.
        let dim = 4_usize;
        let n_classes = 2_usize;
        let true_class = 0_usize;
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0_f32];
        let biases = vec![1.0, 0.0_f32];

        let logit_fn = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let loss_grad =
            linear_loss_grad_ce(weights.clone(), biases.clone(), n_classes, dim, true_class);
        let logit_fn2 = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let score_fn = move |x: &[f32]| -> AdvResult<f32> {
            let logits = logit_fn2(x)?;
            Ok(-logits[true_class])
        };

        let x = vec![0.5_f32; dim];
        let cfg = AutoAttackConfig {
            eps: 0.1,
            apgd_n_steps: 5,
            square_n_queries: 10,
            ..Default::default()
        };
        let mut rng = LcgRng::new(1);
        let result = autoattack(
            &x, true_class, n_classes, loss_grad, logit_fn, score_fn, &cfg, &mut rng,
        );
        assert!(result.is_ok(), "true_class=0 < n_classes=2 should succeed");
    }

    // ── Test 15 ───────────────────────────────────────────────────────────────

    #[test]
    fn dlr_loss_all_zeros() {
        // All-zero logits: denominator = 0 → None or handled gracefully.
        let logits = vec![0.0_f32, 0.0_f32, 0.0_f32];
        let result = dlr_loss(&logits, 0);
        // All zeros means all top logits equal → denom=0 → None.
        assert!(
            result.is_none(),
            "all-zeros logits should give None (denom=0)"
        );
    }

    // ── Test 16 ───────────────────────────────────────────────────────────────

    #[test]
    fn autoattack_result_finite() {
        let dim = 4_usize;
        let n_classes = 2_usize;
        let true_class = 0_usize;
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0_f32];
        let biases = vec![1.0, 0.0_f32];

        let logit_fn = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let loss_grad =
            linear_loss_grad_ce(weights.clone(), biases.clone(), n_classes, dim, true_class);
        let logit_fn2 = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let score_fn = move |x: &[f32]| -> AdvResult<f32> {
            let logits = logit_fn2(x)?;
            Ok(-logits[true_class])
        };

        let x = vec![0.5_f32; dim];
        let cfg = AutoAttackConfig {
            eps: 0.1,
            apgd_n_steps: 10,
            square_n_queries: 20,
            ..Default::default()
        };
        let mut rng = LcgRng::new(99);
        let result = autoattack(
            &x, true_class, n_classes, loss_grad, logit_fn, score_fn, &cfg, &mut rng,
        )
        .expect("value should be present");
        for &v in &result {
            assert!(v.is_finite(), "result contains non-finite value: {v}");
        }
    }

    // ── Test 17 ───────────────────────────────────────────────────────────────

    #[test]
    fn autoattack_square_fallback() {
        // Even if APGD phases produce no class change, square fills in a result.
        // Use a robust classifier (large bias) that APGD won't crack but Square
        // still produces a finite result.
        let dim = 4_usize;
        let n_classes = 2_usize;
        let true_class = 0_usize;
        // Very large bias for class 0 → hard to fool with small eps.
        let weights = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0_f32];
        let biases = vec![100.0, 0.0_f32];

        let logit_fn = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let loss_grad =
            linear_loss_grad_ce(weights.clone(), biases.clone(), n_classes, dim, true_class);
        let logit_fn2 = linear_logit_fn(weights.clone(), biases.clone(), n_classes, dim);
        let score_fn = move |x: &[f32]| -> AdvResult<f32> {
            let logits = logit_fn2(x)?;
            Ok(-logits[true_class])
        };

        let x = vec![0.5_f32; dim];
        let cfg = AutoAttackConfig {
            eps: 0.05,
            apgd_n_steps: 5,
            square_n_queries: 50,
            lo: 0.0,
            hi: 1.0,
            ..Default::default()
        };
        let mut rng = LcgRng::new(77);
        let result = autoattack(
            &x, true_class, n_classes, loss_grad, logit_fn, score_fn, &cfg, &mut rng,
        );
        // Should always return a result (Square fallback).
        assert!(
            result.is_ok(),
            "should return a result even when APGD fails to fool"
        );
        assert_eq!(result.expect("result should be present").len(), dim);
    }

    // ── Test 18 ───────────────────────────────────────────────────────────────

    #[test]
    fn dlr_loss_large_logits() {
        // Large logit values (1000.0) should produce finite result.
        let logits = vec![1000.0_f32, 500.0_f32, 100.0_f32];
        let result = dlr_loss(&logits, 0);
        assert!(result.is_some(), "large logits should give Some");
        let val = result.expect("result should be present");
        assert!(
            val.is_finite(),
            "DLR with large logits should be finite, got {val}"
        );
        // Correctly classified: should be negative.
        assert!(
            val < 0.0,
            "correctly classified large logits should give negative DLR, got {val}"
        );
    }
}
