//! Loss-landscape probing utilities.
//!
//! A robust model should be hard to attack *from every direction*, not merely
//! from the single gradient direction that a one-shot attack explores. When a
//! defence "looks" robust under a fixed PGD run but its empirical robustness
//! evaporates under stronger or differently-initialised attacks, the symptom is
//! a **rugged / masked loss landscape**: the loss surface around a clean input
//! has many shallow local maxima, so where an attack *lands* depends heavily on
//! its random starting point (Athalye, Carlini & Wagner 2018; Madry et al.
//! 2018 §B).
//!
//! This module quantifies that phenomenon directly. Given a clean input and a
//! loss-gradient closure, it runs **multi-restart PGD** from many random L∞ (or
//! L2) starting points and records, for every restart:
//!
//! * the final loss value reached (the depth of the local maximum found), and
//! * the L2 distance the iterate travelled from the *clean* input.
//!
//! From these it derives:
//!
//! * a [`Histogram`] of final-iterate distances (the "distance histogram"
//!   referenced by the PGD robustness literature) and of final losses;
//! * the spread (max − min, standard deviation) of the final losses across
//!   restarts — large spread ⇒ the landscape is multi-modal and a single
//!   restart is unreliable;
//! * the **best-restart loss** (the worst-case loss an attacker would actually
//!   report) and how much it improves over the *mean* restart, a cheap proxy
//!   for "how many restarts you need".
//!
//! It also exposes a 1-D **loss-profile probe** ([`loss_profile`]) that walks
//! along a chosen direction (typically the clean-input gradient sign) and
//! samples the loss at evenly-spaced multiples of a step, which makes
//! gradient-masking-induced non-monotonicities (loss that *decreases* as you
//! step along the ascent direction) immediately visible.

use crate::error::{AdvError, AdvResult};
use crate::handle::LcgRng;
use crate::threat_model::lp_ball::{l2_norm, project_l2};

// ─── norm selector ────────────────────────────────────────────────────────────

/// Lp ball the multi-restart probe initialises and projects within.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeNorm {
    /// L∞ ball.
    LInf,
    /// L2 ball.
    L2,
}

// ─── Histogram ────────────────────────────────────────────────────────────────

/// A fixed-width histogram over `[lo, hi]` with `n_bins` equal-width buckets.
///
/// Values `< lo` fall into bin 0 and values `>= hi` into the last bin
/// (clamping rather than dropping, so the total count always equals the number
/// of inserted samples).
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    /// Lower edge of the first bin.
    pub lo: f32,
    /// Upper edge of the last bin.
    pub hi: f32,
    /// Per-bin counts, length `n_bins`.
    pub counts: Vec<usize>,
}

impl Histogram {
    /// Build an empty histogram over `[lo, hi]` with `n_bins` buckets.
    ///
    /// # Errors
    /// * [`AdvError::Internal`] — `n_bins == 0` or `hi <= lo` or non-finite edge.
    pub fn new(lo: f32, hi: f32, n_bins: usize) -> AdvResult<Self> {
        if n_bins == 0 {
            return Err(AdvError::Internal("histogram n_bins must be > 0".into()));
        }
        if !(lo.is_finite() && hi.is_finite()) || hi <= lo {
            return Err(AdvError::Internal(
                "histogram requires lo < hi, both finite".into(),
            ));
        }
        Ok(Self {
            lo,
            hi,
            counts: vec![0; n_bins],
        })
    }

    /// Insert one value, clamping out-of-range values into the edge bins.
    pub fn insert(&mut self, v: f32) {
        let n = self.counts.len();
        let t = ((v - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0);
        let mut idx = (t * n as f32) as usize;
        if idx >= n {
            idx = n - 1;
        }
        self.counts[idx] += 1;
    }

    /// Total number of inserted samples.
    #[must_use]
    pub fn total(&self) -> usize {
        self.counts.iter().sum()
    }

    /// Index of the most populated bin (lowest index on ties).
    #[must_use]
    pub fn mode_bin(&self) -> usize {
        let mut best = 0;
        let mut best_c = 0;
        for (i, &c) in self.counts.iter().enumerate() {
            if c > best_c {
                best_c = c;
                best = i;
            }
        }
        best
    }
}

// ─── multi-restart probe config + report ─────────────────────────────────────

/// Configuration for [`multi_restart_probe`].
#[derive(Debug, Clone, Copy)]
pub struct LandscapeProbeConfig {
    /// Perturbation budget ε. Finite, `> 0`.
    pub eps: f32,
    /// PGD step size α. Finite, `> 0`.
    pub alpha: f32,
    /// Number of PGD steps per restart. `>= 1`.
    pub n_steps: usize,
    /// Number of random restarts. `>= 1`.
    pub n_restarts: usize,
    /// Number of histogram bins. `>= 1`.
    pub n_bins: usize,
    /// Which Lp ball to initialise / project within.
    pub norm: ProbeNorm,
}

impl LandscapeProbeConfig {
    /// Validating constructor.
    ///
    /// # Errors
    /// * [`AdvError::InvalidEpsilon`]  — non-finite/non-positive `eps`.
    /// * [`AdvError::InvalidAlpha`]    — non-finite/non-positive `alpha`.
    /// * [`AdvError::InvalidNumSteps`] — `n_steps == 0` or `n_restarts == 0`.
    /// * [`AdvError::Internal`]        — `n_bins == 0`.
    pub fn new(
        eps: f32,
        alpha: f32,
        n_steps: usize,
        n_restarts: usize,
        n_bins: usize,
        norm: ProbeNorm,
    ) -> AdvResult<Self> {
        if !(eps.is_finite() && eps > 0.0) {
            return Err(AdvError::InvalidEpsilon { eps });
        }
        if !(alpha.is_finite() && alpha > 0.0) {
            return Err(AdvError::InvalidAlpha { alpha });
        }
        if n_steps == 0 || n_restarts == 0 {
            return Err(AdvError::InvalidNumSteps);
        }
        if n_bins == 0 {
            return Err(AdvError::Internal("n_bins must be > 0".into()));
        }
        Ok(Self {
            eps,
            alpha,
            n_steps,
            n_restarts,
            n_bins,
            norm,
        })
    }
}

/// Result of a multi-restart loss-landscape probe.
#[derive(Debug, Clone, PartialEq)]
pub struct LandscapeReport {
    /// Final loss reached by each restart (length `n_restarts`).
    pub final_losses: Vec<f32>,
    /// L2 distance from the clean input to each restart's final iterate.
    pub final_distances: Vec<f32>,
    /// Histogram of `final_losses` over `[min, max]`.
    pub loss_hist: Histogram,
    /// Histogram of `final_distances` over `[0, eps_effective]`.
    pub distance_hist: Histogram,
    /// Largest final loss over all restarts (the worst case an attacker reports).
    pub best_loss: f32,
    /// Mean final loss over restarts.
    pub mean_loss: f32,
    /// Standard deviation of final losses across restarts.
    pub loss_std: f32,
    /// `best_loss − mean_loss`; large values ⇒ many restarts pay off ⇒ rugged
    /// landscape / potential gradient masking.
    pub restart_gain: f32,
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn check_grad(g: &[f32], expected_len: usize, where_: &'static str) -> AdvResult<()> {
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

fn project_l_inf_box(adv: &[f32], x: &[f32], eps: f32, lo: f32, hi: f32) -> Vec<f32> {
    adv.iter()
        .zip(x.iter())
        .map(|(&a, &o)| a.clamp(o - eps, o + eps).clamp(lo, hi))
        .collect()
}

/// Random L∞ init in `[x − eps, x + eps]` clamped to the box.
fn rand_init_l_inf(x: &[f32], eps: f32, lo: f32, hi: f32, rng: &mut LcgRng) -> Vec<f32> {
    x.iter()
        .map(|&xi| (xi + (2.0 * rng.next_f32() - 1.0) * eps).clamp(lo, hi))
        .collect()
}

/// Random L2 init uniform in the ε-ball, clamped to the box.
fn rand_init_l2(x: &[f32], eps: f32, lo: f32, hi: f32, rng: &mut LcgRng) -> Vec<f32> {
    let n = x.len();
    let mut delta = vec![0.0_f32; n];
    rng.fill_normal(&mut delta);
    let nrm = l2_norm(&delta).max(1e-12);
    let u = rng.next_f32().max(1e-12);
    let r = eps * u.powf(1.0 / n as f32);
    let scale = r / nrm;
    x.iter()
        .zip(delta.iter())
        .map(|(&xi, &di)| (xi + scale * di).clamp(lo, hi))
        .collect()
}

// ─── multi-restart probe ──────────────────────────────────────────────────────

/// Probe the loss landscape around `x` with `cfg.n_restarts` random-restart PGD
/// runs, accumulating final-loss and final-distance statistics + histograms.
///
/// `loss_grad(point) -> Vec<f32>` returns the input gradient of the (scalar)
/// attack loss the caller wants maximised. `loss_value(point) -> f32` returns
/// the scalar loss itself (so the probe can record the *depth* each restart
/// reaches without re-implementing the model). Both must agree on dimension `d`.
///
/// # Errors
/// * Validation errors of [`LandscapeProbeConfig::new`].
/// * [`AdvError::EmptyInput`] — empty `x`.
/// * [`AdvError::DimensionMismatch`] / [`AdvError::NanEncountered`] from bad
///   closure outputs.
pub fn multi_restart_probe<G, V>(
    x: &[f32],
    lo: f32,
    hi: f32,
    cfg: &LandscapeProbeConfig,
    rng: &mut LcgRng,
    loss_grad: G,
    loss_value: V,
) -> AdvResult<LandscapeReport>
where
    G: Fn(&[f32]) -> AdvResult<Vec<f32>>,
    V: Fn(&[f32]) -> AdvResult<f32>,
{
    if x.is_empty() {
        return Err(AdvError::EmptyInput);
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(AdvError::NanEncountered {
            location: "multi_restart_probe:x",
        });
    }
    if !(lo.is_finite() && hi.is_finite()) || lo >= hi {
        return Err(AdvError::InvalidLossWeight { weight: hi - lo });
    }
    let d = x.len();

    let mut final_losses = Vec::with_capacity(cfg.n_restarts);
    let mut final_distances = Vec::with_capacity(cfg.n_restarts);

    for _ in 0..cfg.n_restarts {
        let mut adv = match cfg.norm {
            ProbeNorm::LInf => rand_init_l_inf(x, cfg.eps, lo, hi, rng),
            ProbeNorm::L2 => rand_init_l2(x, cfg.eps, lo, hi, rng),
        };
        // Project the start onto the feasible set.
        adv = match cfg.norm {
            ProbeNorm::LInf => project_l_inf_box(&adv, x, cfg.eps, lo, hi),
            ProbeNorm::L2 => project_l2(&adv, x, cfg.eps, lo, hi)?,
        };

        for _ in 0..cfg.n_steps {
            let g = loss_grad(&adv)?;
            check_grad(&g, d, "multi_restart_probe:loss_grad")?;
            adv = match cfg.norm {
                ProbeNorm::LInf => {
                    let stepped: Vec<f32> = adv
                        .iter()
                        .zip(g.iter())
                        .map(|(&a, &gi)| a + cfg.alpha * sign(gi))
                        .collect();
                    project_l_inf_box(&stepped, x, cfg.eps, lo, hi)
                }
                ProbeNorm::L2 => {
                    let nrm = l2_norm(&g).max(1e-12);
                    let stepped: Vec<f32> = adv
                        .iter()
                        .zip(g.iter())
                        .map(|(&a, &gi)| a + cfg.alpha * gi / nrm)
                        .collect();
                    project_l2(&stepped, x, cfg.eps, lo, hi)?
                }
            };
        }

        let loss = loss_value(&adv)?;
        if !loss.is_finite() {
            return Err(AdvError::NanEncountered {
                location: "multi_restart_probe:loss_value",
            });
        }
        let delta: Vec<f32> = adv.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        final_distances.push(l2_norm(&delta));
        final_losses.push(loss);
    }

    // Aggregate statistics.
    let n_f = cfg.n_restarts as f32;
    let mean_loss = final_losses.iter().sum::<f32>() / n_f;
    let var = final_losses
        .iter()
        .map(|&l| (l - mean_loss) * (l - mean_loss))
        .sum::<f32>()
        / n_f;
    let loss_std = var.max(0.0).sqrt();
    let best_loss = final_losses
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_loss = final_losses.iter().copied().fold(f32::INFINITY, f32::min);
    let restart_gain = best_loss - mean_loss;

    // Histograms. Guard against a degenerate (all-equal) range.
    let (l_lo, l_hi) = if (best_loss - min_loss).abs() < 1e-12 {
        (min_loss - 0.5, best_loss + 0.5)
    } else {
        (min_loss, best_loss)
    };
    let mut loss_hist = Histogram::new(l_lo, l_hi, cfg.n_bins)?;
    for &l in &final_losses {
        loss_hist.insert(l);
    }

    // L2 perturbation magnitude is bounded by eps for the L2 ball; for the L∞
    // ball the L2 distance can reach eps·√d, so widen the range accordingly.
    let dist_hi = match cfg.norm {
        ProbeNorm::LInf => cfg.eps * (d as f32).sqrt(),
        ProbeNorm::L2 => cfg.eps,
    }
    .max(1e-6);
    let mut distance_hist = Histogram::new(0.0, dist_hi, cfg.n_bins)?;
    for &dst in &final_distances {
        distance_hist.insert(dst);
    }

    Ok(LandscapeReport {
        final_losses,
        final_distances,
        loss_hist,
        distance_hist,
        best_loss,
        mean_loss,
        loss_std,
        restart_gain,
    })
}

// ─── 1-D loss profile ─────────────────────────────────────────────────────────

/// Sample the loss along a fixed direction starting at `x`.
///
/// Returns `n_points` pairs `(t_k, loss(x + t_k · dir_unit))` where
/// `t_k = k · step` for `k = 0 .. n_points` and `dir` is L2-normalised before
/// use. The iterate is clamped to `[lo, hi]` at every probe so the profile
/// stays inside the valid input box. A non-monotone profile along the gradient
/// (sign) direction is a hallmark of obfuscated gradients.
///
/// # Errors
/// * [`AdvError::EmptyInput`]      — empty `x`/`dir` or `n_points == 0`.
/// * [`AdvError::DimensionMismatch`] — `dir.len() != x.len()`.
/// * [`AdvError::InvalidAlpha`]    — non-finite/non-positive `step`.
/// * [`AdvError::NanEncountered`]  — non-finite input or loss.
pub fn loss_profile<V>(
    x: &[f32],
    dir: &[f32],
    step: f32,
    n_points: usize,
    lo: f32,
    hi: f32,
    loss_value: V,
) -> AdvResult<Vec<(f32, f32)>>
where
    V: Fn(&[f32]) -> AdvResult<f32>,
{
    if x.is_empty() || dir.is_empty() || n_points == 0 {
        return Err(AdvError::EmptyInput);
    }
    if dir.len() != x.len() {
        return Err(AdvError::DimensionMismatch {
            expected: x.len(),
            got: dir.len(),
        });
    }
    if !(step.is_finite() && step > 0.0) {
        return Err(AdvError::InvalidAlpha { alpha: step });
    }
    if x.iter().chain(dir.iter()).any(|v| !v.is_finite()) {
        return Err(AdvError::NanEncountered {
            location: "loss_profile:input",
        });
    }
    let nrm = l2_norm(dir).max(1e-12);
    let unit: Vec<f32> = dir.iter().map(|&d| d / nrm).collect();

    let mut out = Vec::with_capacity(n_points + 1);
    let mut point = vec![0.0_f32; x.len()];
    for k in 0..=n_points {
        let t = k as f32 * step;
        for i in 0..x.len() {
            point[i] = (x[i] + t * unit[i]).clamp(lo, hi);
        }
        let l = loss_value(&point)?;
        if !l.is_finite() {
            return Err(AdvError::NanEncountered {
                location: "loss_profile:loss_value",
            });
        }
        out.push((t, l));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gradient of the concave quadratic loss `L(x) = c − 0.5·‖x − peak‖²`:
    /// `∇L = (peak − x)`, which always points toward the single global maximum,
    /// so all restarts converge to the same loss ⇒ small spread.
    fn unimodal_grad(peak: Vec<f32>) -> impl Fn(&[f32]) -> AdvResult<Vec<f32>> {
        move |x: &[f32]| Ok(x.iter().zip(peak.iter()).map(|(a, p)| p - a).collect())
    }

    /// Value of the concave quadratic loss `L(x) = c − 0.5·‖x − peak‖²`.
    fn unimodal_val(peak: Vec<f32>, c: f32) -> impl Fn(&[f32]) -> AdvResult<f32> {
        move |x: &[f32]| {
            let sq: f32 = x
                .iter()
                .zip(peak.iter())
                .map(|(a, p)| (a - p) * (a - p))
                .sum();
            Ok(c - 0.5 * sq)
        }
    }

    /// Paired gradient/value closures for the concave quadratic loss
    /// `L(x) = c − 0.5·‖x − peak‖²` (single global maximum at `peak`). The
    /// gradient `(peak − x)` is offset-independent, so `c` only shifts the loss
    /// level and never the ascent trajectory or restart spread.
    #[allow(clippy::type_complexity)]
    fn unimodal(
        peak: Vec<f32>,
        c: f32,
    ) -> (
        impl Fn(&[f32]) -> AdvResult<Vec<f32>>,
        impl Fn(&[f32]) -> AdvResult<f32>,
    ) {
        (unimodal_grad(peak.clone()), unimodal_val(peak, c))
    }

    #[test]
    fn histogram_basic_and_clamping() {
        let mut h = Histogram::new(0.0, 1.0, 4).expect("hist");
        h.insert(-0.5); // clamps to bin 0
        h.insert(0.1); // bin 0
        h.insert(0.6); // bin 2
        h.insert(2.0); // clamps to last bin (3)
        assert_eq!(h.total(), 4);
        assert_eq!(h.counts[0], 2);
        assert_eq!(h.counts[2], 1);
        assert_eq!(h.counts[3], 1);
        assert_eq!(h.mode_bin(), 0);
    }

    #[test]
    fn histogram_rejects_bad_args() {
        assert!(Histogram::new(0.0, 1.0, 0).is_err());
        assert!(Histogram::new(1.0, 0.0, 4).is_err());
        assert!(Histogram::new(f32::NAN, 1.0, 4).is_err());
    }

    #[test]
    fn config_validates() {
        assert!(LandscapeProbeConfig::new(0.1, 0.02, 10, 5, 8, ProbeNorm::LInf).is_ok());
        assert!(LandscapeProbeConfig::new(-0.1, 0.02, 10, 5, 8, ProbeNorm::LInf).is_err());
        assert!(LandscapeProbeConfig::new(0.1, 0.0, 10, 5, 8, ProbeNorm::LInf).is_err());
        assert!(LandscapeProbeConfig::new(0.1, 0.02, 0, 5, 8, ProbeNorm::LInf).is_err());
        assert!(LandscapeProbeConfig::new(0.1, 0.02, 10, 0, 8, ProbeNorm::LInf).is_err());
        assert!(LandscapeProbeConfig::new(0.1, 0.02, 10, 5, 0, ProbeNorm::LInf).is_err());
    }

    #[test]
    fn unimodal_landscape_has_small_restart_gain() {
        // A smooth concave loss ⇒ every restart finds (nearly) the same maximum
        // ⇒ best ≈ mean ⇒ tiny restart_gain and tiny loss_std.
        let peak = vec![0.4_f32; 6];
        let x = vec![0.5_f32; 6];
        let (grad, val) = unimodal(peak, 10.0);
        let cfg = LandscapeProbeConfig::new(0.15, 0.03, 40, 12, 8, ProbeNorm::LInf)
            .expect("cfg should build");
        let mut rng = LcgRng::new(7);
        let rep = multi_restart_probe(&x, 0.0, 1.0, &cfg, &mut rng, grad, val)
            .expect("probe should succeed");
        assert_eq!(rep.final_losses.len(), 12);
        assert_eq!(rep.distance_hist.total(), 12);
        assert_eq!(rep.loss_hist.total(), 12);
        assert!(rep.best_loss >= rep.mean_loss - 1e-5);
        assert!(rep.loss_std < 0.05, "loss_std={} too large", rep.loss_std);
        assert!(rep.restart_gain < 0.05, "restart_gain={}", rep.restart_gain);
        // best loss must dominate every individual restart.
        for &l in &rep.final_losses {
            assert!(rep.best_loss >= l - 1e-6);
        }
    }

    #[test]
    fn multimodal_landscape_has_large_spread() {
        // A cosine "egg-carton" loss has many local maxima; different random
        // starts land in different basins ⇒ noticeable loss_std and restart_gain.
        let x = vec![0.5_f32; 4];
        // Higher frequency ⇒ more local maxima within the eps-ball ⇒ random restarts
        // land in more distinct basins ⇒ clearly multimodal final-loss spread.
        let freq = 60.0_f32;
        let grad = move |p: &[f32]| {
            // d/dx Σ cos(freq·x_i) = −freq·sin(freq·x_i)
            Ok(p.iter().map(|&v| -freq * (freq * v).sin()).collect())
        };
        let val = move |p: &[f32]| Ok(p.iter().map(|&v| (freq * v).cos()).sum());
        let cfg = LandscapeProbeConfig::new(0.2, 0.01, 30, 24, 10, ProbeNorm::LInf)
            .expect("cfg should build");
        let mut rng = LcgRng::new(123);
        let rep = multi_restart_probe(&x, 0.0, 1.0, &cfg, &mut rng, grad, val)
            .expect("probe should succeed");
        assert!(
            rep.loss_std > 0.05,
            "expected multimodal spread, std={}",
            rep.loss_std
        );
        assert!(rep.restart_gain > 0.0);
    }

    #[test]
    fn distance_histogram_within_bounds_l2() {
        let peak = vec![5.0_f32; 5]; // far away ⇒ attack pushes to the ε-ball edge
        let x = vec![0.5_f32; 5];
        let (grad, val) = unimodal(peak, 0.0);
        let cfg = LandscapeProbeConfig::new(0.3, 0.1, 30, 10, 6, ProbeNorm::L2)
            .expect("cfg should build");
        let mut rng = LcgRng::new(5);
        let rep = multi_restart_probe(&x, -10.0, 10.0, &cfg, &mut rng, grad, val)
            .expect("probe should succeed");
        // No final distance may exceed eps for the L2 ball.
        for &dst in &rep.final_distances {
            assert!(dst <= cfg.eps + 1e-4, "distance {dst} > eps");
        }
        assert_eq!(rep.distance_hist.total(), 10);
    }

    #[test]
    fn probe_rejects_empty_and_nan() {
        let cfg = LandscapeProbeConfig::new(0.1, 0.02, 5, 3, 4, ProbeNorm::LInf)
            .expect("cfg should build");
        let mut rng = LcgRng::new(0);
        let g = |_x: &[f32]| Ok(vec![0.0_f32; 4]);
        let v = |_x: &[f32]| Ok(0.0_f32);
        assert_eq!(
            multi_restart_probe(&[], 0.0, 1.0, &cfg, &mut rng, g, v).unwrap_err(),
            AdvError::EmptyInput
        );
        let g2 = |_x: &[f32]| Ok(vec![0.0_f32; 2]);
        let v2 = |_x: &[f32]| Ok(0.0_f32);
        assert!(matches!(
            multi_restart_probe(&[f32::NAN, 0.0], 0.0, 1.0, &cfg, &mut rng, g2, v2).unwrap_err(),
            AdvError::NanEncountered { .. }
        ));
    }

    #[test]
    fn loss_profile_monotone_along_ascent() {
        // Linear loss L(x) = Σ x_i has gradient sign = +1 everywhere; walking
        // along +1 increases the loss monotonically.
        let x = vec![0.0_f32; 4];
        let dir = vec![1.0_f32; 4];
        let val = |p: &[f32]| Ok(p.iter().sum::<f32>());
        let prof = loss_profile(&x, &dir, 0.1, 5, -10.0, 10.0, val).expect("profile");
        assert_eq!(prof.len(), 6); // k = 0..=5
        for w in prof.windows(2) {
            assert!(w[1].1 >= w[0].1 - 1e-6, "loss not monotone increasing");
        }
        // t values are evenly spaced multiples of step.
        for (k, (t, _)) in prof.iter().enumerate() {
            assert!((t - k as f32 * 0.1).abs() < 1e-6);
        }
    }

    #[test]
    fn loss_profile_clamps_to_box() {
        // With a tight box the profile saturates: once clamped, the loss stops
        // changing.
        let x = vec![0.9_f32; 3];
        let dir = vec![1.0_f32; 3];
        let val = |p: &[f32]| Ok(p.iter().sum::<f32>());
        let prof = loss_profile(&x, &dir, 1.0, 4, 0.0, 1.0, val).expect("profile");
        // last few points are all at the box edge (sum == 3.0).
        let last = prof.last().expect("non-empty").1;
        assert!((last - 3.0).abs() < 1e-5);
    }

    #[test]
    fn loss_profile_validates() {
        let x = vec![0.0_f32; 3];
        let val = |_p: &[f32]| Ok(0.0_f32);
        assert_eq!(
            loss_profile(&x, &[1.0, 1.0], 0.1, 3, 0.0, 1.0, val).unwrap_err(),
            AdvError::DimensionMismatch {
                expected: 3,
                got: 2
            }
        );
        let val2 = |_p: &[f32]| Ok(0.0_f32);
        assert!(matches!(
            loss_profile(&x, &[1.0, 1.0, 1.0], 0.0, 3, 0.0, 1.0, val2).unwrap_err(),
            AdvError::InvalidAlpha { .. }
        ));
    }

    #[test]
    fn deterministic_with_same_seed() {
        let peak = vec![0.4_f32; 4];
        let x = vec![0.5_f32; 4];
        let cfg = LandscapeProbeConfig::new(0.1, 0.02, 10, 6, 5, ProbeNorm::L2)
            .expect("cfg should build");
        let (g1, v1) = unimodal(peak.clone(), 1.0);
        let (g2, v2) = unimodal(peak, 1.0);
        let mut r1 = LcgRng::new(42);
        let mut r2 = LcgRng::new(42);
        let a = multi_restart_probe(&x, 0.0, 1.0, &cfg, &mut r1, g1, v1).expect("a");
        let b = multi_restart_probe(&x, 0.0, 1.0, &cfg, &mut r2, g2, v2).expect("b");
        assert_eq!(a.final_losses, b.final_losses);
        assert_eq!(a.final_distances, b.final_distances);
    }
}
