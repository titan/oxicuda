//! SmoothAdv — adversarial training of *smoothed* classifiers.
//!
//! Reference: Salman, Li, Razenshteyn, Zhang, Zhang, Bubeck & Yang (2019),
//! *"Provably Robust Deep Learning via Adversarially Trained Smoothed
//! Classifiers"*, NeurIPS.
//!
//! Cohen et al.'s randomized smoothing (see
//! [`crate::defenses::randomized_smoothing`]) turns a base classifier `f` into
//! a certified-robust smoothed classifier
//!
//! ```text
//! g(x) = argmax_c  Pr_{η ∼ N(0, σ² I)} [ f(x + η) = c ].
//! ```
//!
//! SmoothAdv improves the *certified radius* of `g` by training `f` on
//! adversarial examples that attack the **smoothed** soft prediction rather
//! than `f` itself. Concretely, for each clean input `x` we look for a
//! perturbation `δ` within an ε-ball that maximises the *smoothed cross-entropy*
//!
//! ```text
//! L_smooth(x + δ, y) = − log  E_{η ∼ N(0, σ² I)} [ softmax( f(x + δ + η) )_y ],
//! ```
//!
//! i.e. the loss of the Monte-Carlo–averaged soft prediction (Salman et al.,
//! Eq. 4 — "SmoothAdv_PGD"). The expectation over the smoothing noise is
//! estimated with `m_noise` i.i.d. samples and the inner maximisation is a
//! standard PGD ascent on `δ`, projecting back onto the L∞ (or L2) ε-ball at
//! every step.
//!
//! This module is a *CPU reference* for the SmoothAdv inner loop. It does not
//! perform any weight update itself (training is owned by the caller); instead
//! it produces, for a batch of inputs, the SmoothAdv adversarial examples that
//! the caller then feeds — together with the smoothing noise — into its own
//! forward/backward pass. The two public entry points are:
//!
//! * [`SmoothAdvConfig`] — validated hyper-parameters (`sigma`, `eps`,
//!   `alpha`, `n_steps`, `m_noise`, `norm`).
//! * [`smooth_adv_attack`] — produces a single SmoothAdv adversarial example
//!   from a clean input and a *soft-prediction* closure
//!   `soft_predict: Fn(&[f32]) -> AdvResult<Vec<f32>>` returning class
//!   probabilities (a softmax vector) for an input.
//! * [`smooth_adv_batch`] — the same over a flattened batch.
//!
//! The closure is given the **already-noised** input `x + δ + η`; the module
//! draws the `N(0, σ² I)` noise internally and accumulates the gradient of the
//! smoothed-NLL with respect to `δ` via a finite-difference-free analytic chain
//! rule that only needs the soft prediction and the per-sample input gradient
//! supplied by the caller's `soft_grad` closure (the gradient of the scalar
//! smoothed-NLL with respect to the noised input). Where an analytic gradient
//! is unavailable, [`smooth_adv_attack_spsa`] offers a gradient-free
//! Simultaneous-Perturbation Stochastic-Approximation estimator that only needs
//! the soft prediction.

use crate::error::{AdvError, AdvResult};
use crate::handle::LcgRng;
use crate::threat_model::lp_ball::{l2_norm, project_l2};

// ─── Threat-model norm selector ──────────────────────────────────────────────

/// Lp ball used for the SmoothAdv inner maximisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmoothAdvNorm {
    /// L∞ ball: project each coordinate to `[x − ε, x + ε]`.
    LInf,
    /// L2 ball: project the perturbation so `‖δ‖₂ ≤ ε`.
    L2,
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Hyper-parameters for SmoothAdv adversarial training of smoothed classifiers.
///
/// Construct with [`SmoothAdvConfig::new`], which validates every field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmoothAdvConfig {
    /// Standard deviation of the smoothing noise `N(0, σ² I)`. Finite, `> 0`.
    pub sigma: f32,
    /// Perturbation budget for the inner PGD attack. Finite, `> 0`.
    pub eps: f32,
    /// PGD step size. Finite, `> 0`.
    pub alpha: f32,
    /// Number of inner PGD steps. `>= 1`.
    pub n_steps: usize,
    /// Number of Monte-Carlo noise samples used to estimate the smoothed
    /// soft prediction at each step. `>= 1`. Salman et al. use `m ∈ {1, 2, 4, 8}`.
    pub m_noise: usize,
    /// Which Lp ball the inner attack is projected onto.
    pub norm: SmoothAdvNorm,
}

impl SmoothAdvConfig {
    /// Validating constructor.
    ///
    /// # Errors
    /// * [`AdvError::InvalidNoiseSigma`] — non-finite or non-positive `sigma`.
    /// * [`AdvError::InvalidEpsilon`]    — non-finite or non-positive `eps`.
    /// * [`AdvError::InvalidAlpha`]      — non-finite or non-positive `alpha`.
    /// * [`AdvError::InvalidNumSteps`]   — `n_steps == 0`.
    /// * [`AdvError::InsufficientCertSamples`] — `m_noise == 0`.
    pub fn new(
        sigma: f32,
        eps: f32,
        alpha: f32,
        n_steps: usize,
        m_noise: usize,
        norm: SmoothAdvNorm,
    ) -> AdvResult<Self> {
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(AdvError::InvalidNoiseSigma { sigma });
        }
        if !(eps.is_finite() && eps > 0.0) {
            return Err(AdvError::InvalidEpsilon { eps });
        }
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(AdvError::InvalidAlpha { alpha });
        }
        if n_steps == 0 {
            return Err(AdvError::InvalidNumSteps);
        }
        if m_noise == 0 {
            return Err(AdvError::InsufficientCertSamples { min: 1, got: 0 });
        }
        Ok(Self {
            sigma,
            eps,
            alpha,
            n_steps,
            m_noise,
            norm,
        })
    }
}

impl Default for SmoothAdvConfig {
    fn default() -> Self {
        Self {
            sigma: 0.25,
            eps: 0.5,
            alpha: 0.1,
            n_steps: 4,
            m_noise: 2,
            norm: SmoothAdvNorm::L2,
        }
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Validate the clean input and the box `[lo, hi]`.
fn validate_input(x: &[f32], lo: f32, hi: f32) -> AdvResult<()> {
    if x.is_empty() {
        return Err(AdvError::EmptyInput);
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(AdvError::NanEncountered {
            location: "smooth_adv:x",
        });
    }
    if !(lo.is_finite() && hi.is_finite()) || lo >= hi {
        return Err(AdvError::InvalidLossWeight { weight: hi - lo });
    }
    Ok(())
}

/// Validate a gradient/probability vector returned by a caller closure.
fn check_vec(g: &[f32], expected_len: usize, where_: &'static str) -> AdvResult<()> {
    if g.len() != expected_len {
        return Err(AdvError::DimensionMismatch {
            expected: expected_len,
            got: g.len(),
        });
    }
    if g.iter().any(|v| !v.is_finite()) {
        return Err(AdvError::NanEncountered { location: where_ });
    }
    Ok(())
}

/// Element-wise sign with exact `0.0` on zeros.
#[inline]
fn sign(v: f32) -> f32 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Project `adv` onto the L∞ ε-ball around `x_orig`, then clamp to `[lo, hi]`.
fn project_l_inf_box(adv: &[f32], x_orig: &[f32], eps: f32, lo: f32, hi: f32) -> Vec<f32> {
    adv.iter()
        .zip(x_orig.iter())
        .map(|(&a, &o)| a.clamp(o - eps, o + eps).clamp(lo, hi))
        .collect()
}

/// Average soft prediction over `m_noise` Gaussian draws, returning the smoothed
/// probability vector `p̄ = (1/m) Σ softmax(f(x + δ + η_k))`.
///
/// The caller's `soft_predict` is expected to already return a softmax (a
/// non-negative vector that sums to ≈ 1); we re-normalise defensively so that
/// caller logits or unnormalised scores still yield a valid distribution.
fn smoothed_soft<F>(
    point: &[f32],
    sigma: f32,
    m_noise: usize,
    n_classes_hint: Option<usize>,
    rng: &mut LcgRng,
    noise: &mut [f32],
    noised: &mut [f32],
    soft_predict: &F,
) -> AdvResult<Vec<f32>>
where
    F: Fn(&[f32]) -> AdvResult<Vec<f32>>,
{
    let d = point.len();
    let mut acc: Vec<f32> = match n_classes_hint {
        Some(k) => vec![0.0; k],
        None => Vec::new(),
    };
    for _ in 0..m_noise {
        rng.fill_normal(noise);
        for i in 0..d {
            noised[i] = point[i] + sigma * noise[i];
        }
        let p = soft_predict(noised)?;
        if p.is_empty() {
            return Err(AdvError::EmptyInput);
        }
        if p.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(AdvError::NanEncountered {
                location: "smooth_adv:soft_predict",
            });
        }
        let s: f32 = p.iter().sum();
        let s = if s > 0.0 { s } else { 1.0 };
        if acc.is_empty() {
            acc = vec![0.0; p.len()];
        } else if acc.len() != p.len() {
            return Err(AdvError::DimensionMismatch {
                expected: acc.len(),
                got: p.len(),
            });
        }
        for (a, v) in acc.iter_mut().zip(p.iter()) {
            *a += *v / s;
        }
    }
    let inv = 1.0 / m_noise as f32;
    for a in acc.iter_mut() {
        *a *= inv;
    }
    Ok(acc)
}

// ─── Analytic SmoothAdv (gradient supplied by the caller) ─────────────────────

/// Run the SmoothAdv inner maximisation with a caller-supplied analytic
/// gradient of the smoothed negative-log-likelihood w.r.t. the noised input.
///
/// For each PGD step the module:
/// 1. draws `m_noise` noise vectors `η_k ∼ N(0, σ² I)`,
/// 2. forms the noised iterate `x + δ + η_k` and queries `soft_grad` for the
///    gradient `∂/∂(input) [ − log p̄_y ]` at that point,
/// 3. averages the gradients over the `m_noise` draws (the Monte-Carlo estimate
///    of `∇_δ L_smooth`),
/// 4. takes a signed (L∞) or normalised (L2) ascent step of size `α`,
/// 5. projects back onto the ε-ball and clamps to `[lo, hi]`.
///
/// `soft_grad(point) -> Vec<f32>` must return the length-`d` input gradient of
/// the scalar smoothed-NLL for the *target class* `y` evaluated at `point`. The
/// closure owns `y`. Returns the resulting adversarial example `x + δ*`.
///
/// # Errors
/// Validation errors of [`SmoothAdvConfig::new`] plus [`AdvError::EmptyInput`],
/// [`AdvError::DimensionMismatch`], [`AdvError::NanEncountered`].
pub fn smooth_adv_attack<G>(
    x: &[f32],
    lo: f32,
    hi: f32,
    cfg: &SmoothAdvConfig,
    rng: &mut LcgRng,
    soft_grad: G,
) -> AdvResult<Vec<f32>>
where
    G: Fn(&[f32]) -> AdvResult<Vec<f32>>,
{
    validate_input(x, lo, hi)?;
    let d = x.len();
    let mut adv = x.to_vec();
    let mut noise = vec![0.0_f32; d];
    let mut noised = vec![0.0_f32; d];
    let mut grad_acc = vec![0.0_f32; d];

    for _ in 0..cfg.n_steps {
        grad_acc.iter_mut().for_each(|g| *g = 0.0);
        for _ in 0..cfg.m_noise {
            rng.fill_normal(&mut noise);
            for i in 0..d {
                noised[i] = adv[i] + cfg.sigma * noise[i];
            }
            let g = soft_grad(&noised)?;
            check_vec(&g, d, "smooth_adv:soft_grad")?;
            for (acc, gi) in grad_acc.iter_mut().zip(g.iter()) {
                *acc += *gi;
            }
        }
        let inv = 1.0 / cfg.m_noise as f32;
        for g in grad_acc.iter_mut() {
            *g *= inv;
        }

        // Gradient *ascent* on the NLL (maximise the loss of the true class).
        match cfg.norm {
            SmoothAdvNorm::LInf => {
                let stepped: Vec<f32> = adv
                    .iter()
                    .zip(grad_acc.iter())
                    .map(|(&a, &g)| a + cfg.alpha * sign(g))
                    .collect();
                adv = project_l_inf_box(&stepped, x, cfg.eps, lo, hi);
            }
            SmoothAdvNorm::L2 => {
                let nrm = l2_norm(&grad_acc).max(1e-12);
                let stepped: Vec<f32> = adv
                    .iter()
                    .zip(grad_acc.iter())
                    .map(|(&a, &g)| a + cfg.alpha * g / nrm)
                    .collect();
                adv = project_l2(&stepped, x, cfg.eps, lo, hi)?;
            }
        }
    }
    Ok(adv)
}

// ─── Gradient-free SmoothAdv (SPSA estimator on the smoothed NLL) ─────────────

/// Run the SmoothAdv inner maximisation **without** an analytic gradient, using
/// a Simultaneous-Perturbation Stochastic-Approximation (SPSA) estimate of
/// `∇_δ L_smooth` from the soft-prediction closure alone.
///
/// At each PGD step we draw a Rademacher direction `Δ ∈ {−1, +1}^d` and a
/// finite-difference probe size `c` (defaulting to `σ / 16`), evaluate the
/// smoothed NLL at `δ ± cΔ` via `smoothed_soft`, and form the SPSA gradient
/// estimate
///
/// ```text
/// ĝ_i = ( L_smooth(δ + cΔ) − L_smooth(δ − cΔ) ) / (2 c Δ_i).
/// ```
///
/// This needs `2 · m_noise` model evaluations per step but no gradient, which
/// is convenient when the caller's model is only available as a forward pass.
/// `target_class` selects the NLL term `− log p̄_{target_class}`.
///
/// # Errors
/// Same as [`smooth_adv_attack`].
pub fn smooth_adv_attack_spsa<F>(
    x: &[f32],
    lo: f32,
    hi: f32,
    target_class: usize,
    cfg: &SmoothAdvConfig,
    rng: &mut LcgRng,
    soft_predict: F,
) -> AdvResult<Vec<f32>>
where
    F: Fn(&[f32]) -> AdvResult<Vec<f32>>,
{
    validate_input(x, lo, hi)?;
    let d = x.len();
    let mut adv = x.to_vec();
    let mut noise = vec![0.0_f32; d];
    let mut noised = vec![0.0_f32; d];
    let probe = (cfg.sigma / 16.0).max(1e-4);

    // Smoothed NLL of `target_class` at a perturbed iterate.
    let nll = |point: &[f32],
               rng: &mut LcgRng,
               noise: &mut [f32],
               noised: &mut [f32]|
     -> AdvResult<f32> {
        let p = smoothed_soft(
            point,
            cfg.sigma,
            cfg.m_noise,
            None,
            rng,
            noise,
            noised,
            &soft_predict,
        )?;
        if target_class >= p.len() {
            return Err(AdvError::DimensionMismatch {
                expected: p.len(),
                got: target_class + 1,
            });
        }
        // Numerically stable −log p̄_y with a floor on the probability.
        Ok(-(p[target_class].max(1e-12)).ln())
    };

    for _ in 0..cfg.n_steps {
        // Rademacher direction.
        let mut dir = vec![0.0_f32; d];
        for di in dir.iter_mut() {
            *di = if rng.next_f32() < 0.5 { -1.0 } else { 1.0 };
        }
        let plus: Vec<f32> = adv
            .iter()
            .zip(dir.iter())
            .map(|(&a, &s)| a + probe * s)
            .collect();
        let minus: Vec<f32> = adv
            .iter()
            .zip(dir.iter())
            .map(|(&a, &s)| a - probe * s)
            .collect();
        let l_plus = nll(&plus, rng, &mut noise, &mut noised)?;
        let l_minus = nll(&minus, rng, &mut noise, &mut noised)?;
        let scale = (l_plus - l_minus) / (2.0 * probe);
        // ĝ_i = scale / Δ_i ; for Rademacher Δ_i ∈ {±1}, 1/Δ_i = Δ_i.
        let grad: Vec<f32> = dir.iter().map(|&s| scale * s).collect();

        match cfg.norm {
            SmoothAdvNorm::LInf => {
                let stepped: Vec<f32> = adv
                    .iter()
                    .zip(grad.iter())
                    .map(|(&a, &g)| a + cfg.alpha * sign(g))
                    .collect();
                adv = project_l_inf_box(&stepped, x, cfg.eps, lo, hi);
            }
            SmoothAdvNorm::L2 => {
                let nrm = l2_norm(&grad).max(1e-12);
                let stepped: Vec<f32> = adv
                    .iter()
                    .zip(grad.iter())
                    .map(|(&a, &g)| a + cfg.alpha * g / nrm)
                    .collect();
                adv = project_l2(&stepped, x, cfg.eps, lo, hi)?;
            }
        }
    }
    Ok(adv)
}

// ─── Batch driver ─────────────────────────────────────────────────────────────

/// Apply [`smooth_adv_attack`] to every sample in a flattened `batch` of shape
/// `[n_samples * dim]`, returning the flattened adversarial batch.
///
/// `soft_grad_for(sample_idx)` returns a per-sample analytic-gradient closure
/// (so callers can capture each sample's true label).
///
/// # Errors
/// * [`AdvError::EmptyInput`] — empty batch / zero `dim`.
/// * [`AdvError::DimensionMismatch`] — `batch.len()` not divisible by `dim`.
/// * Plus every error of [`smooth_adv_attack`].
pub fn smooth_adv_batch<G, M>(
    batch: &[f32],
    dim: usize,
    lo: f32,
    hi: f32,
    cfg: &SmoothAdvConfig,
    rng: &mut LcgRng,
    soft_grad_for: M,
) -> AdvResult<Vec<f32>>
where
    G: Fn(&[f32]) -> AdvResult<Vec<f32>>,
    M: Fn(usize) -> G,
{
    if batch.is_empty() || dim == 0 {
        return Err(AdvError::EmptyInput);
    }
    if batch.len() % dim != 0 {
        return Err(AdvError::DimensionMismatch {
            expected: dim,
            got: batch.len(),
        });
    }
    let n = batch.len() / dim;
    let mut out = Vec::with_capacity(batch.len());
    for s in 0..n {
        let x = &batch[s * dim..(s + 1) * dim];
        let adv = smooth_adv_attack(x, lo, hi, cfg, rng, soft_grad_for(s))?;
        out.extend_from_slice(&adv);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-class soft prediction for a 1-D linear model `score = w·x`:
    /// `p = softmax([0, score])`, so class 1's probability grows with `w·x`.
    /// The smoothed-NLL gradient of the *true* class `y=1` w.r.t. the input is
    /// `−(1 − p_1) · w` (ascending pushes `w·x` down, away from class 1).
    fn linear_soft_predict(w: Vec<f32>) -> impl Fn(&[f32]) -> AdvResult<Vec<f32>> {
        move |x: &[f32]| {
            let score: f32 = w.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            let m = score.max(0.0);
            let e0 = (0.0 - m).exp();
            let e1 = (score - m).exp();
            let z = e0 + e1;
            Ok(vec![e0 / z, e1 / z])
        }
    }

    /// Analytic gradient of `−log p_1` w.r.t. the input for the above model.
    fn linear_soft_grad_class1(w: Vec<f32>) -> impl Fn(&[f32]) -> AdvResult<Vec<f32>> {
        move |x: &[f32]| {
            let score: f32 = w.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            let p1 = 1.0 / (1.0 + (-score).exp());
            // d(−log p_1)/d x = −(1 − p_1) · w
            Ok(w.iter().map(|&wi| -(1.0 - p1) * wi).collect())
        }
    }

    #[test]
    fn config_validates() {
        assert!(SmoothAdvConfig::new(0.25, 0.5, 0.1, 4, 2, SmoothAdvNorm::L2).is_ok());
        assert!(SmoothAdvConfig::new(0.0, 0.5, 0.1, 4, 2, SmoothAdvNorm::L2).is_err());
        assert!(SmoothAdvConfig::new(0.25, -0.5, 0.1, 4, 2, SmoothAdvNorm::L2).is_err());
        assert!(SmoothAdvConfig::new(0.25, 0.5, 0.0, 4, 2, SmoothAdvNorm::L2).is_err());
        assert!(SmoothAdvConfig::new(0.25, 0.5, 0.1, 0, 2, SmoothAdvNorm::L2).is_err());
        assert!(SmoothAdvConfig::new(0.25, 0.5, 0.1, 4, 0, SmoothAdvNorm::L2).is_err());
        assert!(SmoothAdvConfig::new(f32::NAN, 0.5, 0.1, 4, 2, SmoothAdvNorm::LInf).is_err());
        let d = SmoothAdvConfig::default();
        assert_eq!(d.norm, SmoothAdvNorm::L2);
        assert!((d.sigma - 0.25).abs() < 1e-6);
    }

    #[test]
    fn empty_and_nan_input_rejected() {
        let cfg = SmoothAdvConfig::default();
        let mut rng = LcgRng::new(0);
        let empty: Vec<f32> = vec![];
        assert_eq!(
            smooth_adv_attack(&empty, 0.0, 1.0, &cfg, &mut rng, |_| Ok(vec![])).unwrap_err(),
            AdvError::EmptyInput
        );
        let nan = vec![f32::NAN, 0.0];
        assert!(matches!(
            smooth_adv_attack(&nan, 0.0, 1.0, &cfg, &mut rng, |_| Ok(vec![0.0, 0.0])).unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }

    #[test]
    fn l_inf_respects_eps_ball_and_box() {
        // Strong positive weights ⇒ the attack drives the input down.
        let w = vec![2.0_f32, 2.0, 2.0, 2.0];
        let x = vec![0.7_f32; 4];
        let cfg = SmoothAdvConfig::new(0.1, 0.1, 0.03, 20, 4, SmoothAdvNorm::LInf)
            .expect("cfg should build");
        let mut rng = LcgRng::new(7);
        let adv = smooth_adv_attack(&x, 0.0, 1.0, &cfg, &mut rng, linear_soft_grad_class1(w))
            .expect("attack should succeed");
        for (a, o) in adv.iter().zip(x.iter()) {
            assert!((a - o).abs() <= cfg.eps + 1e-5, "coord outside ε-ball");
            assert!((0.0..=1.0).contains(a), "coord outside box");
        }
    }

    #[test]
    fn l2_respects_eps_ball() {
        let w = vec![1.5_f32; 6];
        let x = vec![0.5_f32; 6];
        let cfg = SmoothAdvConfig::new(0.2, 0.5, 0.1, 15, 4, SmoothAdvNorm::L2)
            .expect("cfg should build");
        let mut rng = LcgRng::new(11);
        let adv = smooth_adv_attack(&x, -10.0, 10.0, &cfg, &mut rng, linear_soft_grad_class1(w))
            .expect("attack should succeed");
        let delta: Vec<f32> = adv.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        assert!(l2_norm(&delta) <= cfg.eps + 1e-4, "δ escaped the L2 ball");
    }

    #[test]
    fn attack_reduces_true_class_probability() {
        // SmoothAdv on class 1 must lower the smoothed probability of class 1.
        // w is deliberately small so the sigmoid is NOT saturated: at w=3 the clean
        // prob is 0.9993 and the max achievable drop within eps=0.2 is only ~0.007
        // (the attack reaches the eps-boundary optimally, but the logit barely moves).
        // With w=1 the eps move x:0.6->0.4 lowers the logit 2.4->1.6 ⇒ a real ~0.085 drop.
        let w = vec![1.0_f32, 1.0, 1.0, 1.0];
        let x = vec![0.6_f32; 4];
        let predict = linear_soft_predict(w.clone());
        let p_clean = predict(&x).expect("predict")[1];
        let cfg = SmoothAdvConfig::new(0.05, 0.2, 0.04, 25, 4, SmoothAdvNorm::LInf)
            .expect("cfg should build");
        let mut rng = LcgRng::new(3);
        let adv = smooth_adv_attack(&x, -10.0, 10.0, &cfg, &mut rng, linear_soft_grad_class1(w))
            .expect("attack should succeed");
        let p_adv = predict(&adv).expect("predict")[1];
        assert!(
            p_adv < p_clean - 0.05,
            "expected class-1 prob to drop: clean={p_clean}, adv={p_adv}"
        );
    }

    #[test]
    fn spsa_attack_reduces_true_class_probability() {
        // Gradient-free variant should also push class-1 probability down.
        let w = vec![3.0_f32, 3.0, 3.0, 3.0];
        let x = vec![0.6_f32; 4];
        let p_clean = linear_soft_predict(w.clone())(&x).expect("predict")[1];
        let cfg = SmoothAdvConfig::new(0.05, 0.25, 0.05, 40, 4, SmoothAdvNorm::LInf)
            .expect("cfg should build");
        let mut rng = LcgRng::new(9);
        let adv = smooth_adv_attack_spsa(
            &x,
            -10.0,
            10.0,
            1,
            &cfg,
            &mut rng,
            linear_soft_predict(w.clone()),
        )
        .expect("spsa attack should succeed");
        let p_adv = linear_soft_predict(w)(&adv).expect("predict")[1];
        assert!(
            p_adv < p_clean,
            "expected class-1 prob to drop under SPSA: clean={p_clean}, adv={p_adv}"
        );
    }

    #[test]
    fn smoothed_soft_normalises_to_distribution() {
        // The averaged soft prediction must be a valid distribution (sums to 1).
        let predict = linear_soft_predict(vec![1.0_f32, -1.0]);
        let mut rng = LcgRng::new(5);
        let mut noise = vec![0.0_f32; 2];
        let mut noised = vec![0.0_f32; 2];
        let p = smoothed_soft(
            &[0.3_f32, 0.4],
            0.2,
            64,
            None,
            &mut rng,
            &mut noise,
            &mut noised,
            &predict,
        )
        .expect("smoothed soft");
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-4, "distribution sum = {s}");
        assert!(p.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn deterministic_with_same_seed() {
        let w = vec![2.0_f32; 5];
        let x = vec![0.4_f32; 5];
        let cfg = SmoothAdvConfig::new(0.1, 0.3, 0.05, 10, 2, SmoothAdvNorm::L2)
            .expect("cfg should build");
        let mut r1 = LcgRng::new(42);
        let mut r2 = LcgRng::new(42);
        let a1 = smooth_adv_attack(
            &x,
            -5.0,
            5.0,
            &cfg,
            &mut r1,
            linear_soft_grad_class1(w.clone()),
        )
        .expect("a1");
        let a2 = smooth_adv_attack(&x, -5.0, 5.0, &cfg, &mut r2, linear_soft_grad_class1(w))
            .expect("a2");
        for (a, b) in a1.iter().zip(a2.iter()) {
            assert!((a - b).abs() < 1e-6, "non-deterministic with fixed seed");
        }
    }

    #[test]
    fn grad_dim_mismatch_caught() {
        let x = vec![0.5_f32; 4];
        let cfg = SmoothAdvConfig::new(0.1, 0.3, 0.05, 3, 1, SmoothAdvNorm::LInf)
            .expect("cfg should build");
        let mut rng = LcgRng::new(0);
        let bad = |_x: &[f32]| Ok(vec![1.0_f32; 3]); // wrong length
        assert!(matches!(
            smooth_adv_attack(&x, -1.0, 1.0, &cfg, &mut rng, bad).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn batch_driver_shapes_and_eps() {
        // 3 samples × dim 4 flattened.
        let dim = 4;
        let batch: Vec<f32> = (0..12).map(|i| 0.3 + (i % 4) as f32 * 0.05).collect();
        let cfg = SmoothAdvConfig::new(0.1, 0.1, 0.03, 8, 2, SmoothAdvNorm::LInf)
            .expect("cfg should build");
        let mut rng = LcgRng::new(21);
        let w = vec![2.0_f32; 4];
        let out = smooth_adv_batch(&batch, dim, 0.0, 1.0, &cfg, &mut rng, |_s| {
            linear_soft_grad_class1(w.clone())
        })
        .expect("batch attack");
        assert_eq!(out.len(), batch.len());
        for s in 0..3 {
            for i in 0..dim {
                let o = batch[s * dim + i];
                let a = out[s * dim + i];
                assert!((a - o).abs() <= cfg.eps + 1e-5);
                assert!((0.0..=1.0).contains(&a));
            }
        }
    }

    #[test]
    fn batch_rejects_bad_shape() {
        let cfg = SmoothAdvConfig::default();
        let mut rng = LcgRng::new(0);
        let batch = vec![0.5_f32; 10]; // not divisible by dim=4
        let w = vec![1.0_f32; 4];
        assert!(matches!(
            smooth_adv_batch(&batch, 4, 0.0, 1.0, &cfg, &mut rng, |_s| {
                linear_soft_grad_class1(w.clone())
            })
            .unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
        assert_eq!(
            smooth_adv_batch(&[], 4, 0.0, 1.0, &cfg, &mut rng, |_s| {
                linear_soft_grad_class1(vec![1.0; 4])
            })
            .unwrap_err(),
            AdvError::EmptyInput
        );
    }

    #[test]
    fn zero_weight_model_is_a_fixed_point() {
        // With w = 0 the model is constant (p = [0.5, 0.5]); gradient is 0, so
        // the L∞ attack must leave the input unchanged (sign(0) = 0).
        let w = vec![0.0_f32; 4];
        let x = vec![0.5_f32; 4];
        let cfg = SmoothAdvConfig::new(0.1, 0.2, 0.05, 10, 4, SmoothAdvNorm::LInf)
            .expect("cfg should build");
        let mut rng = LcgRng::new(1);
        let adv = smooth_adv_attack(&x, 0.0, 1.0, &cfg, &mut rng, linear_soft_grad_class1(w))
            .expect("attack");
        for (a, o) in adv.iter().zip(x.iter()) {
            assert!(
                (a - o).abs() < 1e-6,
                "zero-gradient should not move iterate"
            );
        }
    }
}
