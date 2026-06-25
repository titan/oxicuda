//! Decoding strategies for autoregressive caption generation.
//!
//! [`crate::caption::prefix_lm::PrefixLm`] decodes greedily. This module adds the
//! sampling and search strategies that production captioners use, all as pure,
//! deterministic CPU primitives operating on a per-step logit vector:
//!
//! - [`temperature_softmax`] — temperature-scaled softmax over logits.
//! - [`top_k_filter`] — keep only the `k` highest-logit tokens (Fan 2018).
//! - [`nucleus_filter`] — keep the smallest token set whose cumulative
//!   probability ≥ `top_p` (Holtzman 2020, "nucleus" / top-p sampling).
//! - [`sample_categorical`] — inverse-CDF categorical sampling from a probability
//!   vector using the crate's deterministic [`LcgRng`].
//! - [`sample_token`] — the full per-step pipeline: temperature → top-k →
//!   nucleus → renormalise → sample.
//! - [`beam_search`] — width-`B` length-normalised beam search driven by a
//!   user-supplied `next_logits` closure, so it composes with *any* decoder
//!   (including `PrefixLm`).

use crate::error::{MmResult, MultiModalError};
use crate::handle::LcgRng;

/// Sampling hyper-parameters for [`sample_token`].
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Softmax temperature ( > 0 ). Higher → flatter, lower → sharper.
    pub temperature: f32,
    /// Top-k cutoff. `0` disables top-k filtering.
    pub top_k: usize,
    /// Nucleus (top-p) cumulative-probability threshold in `(0, 1]`. Values
    /// `>= 1.0` disable nucleus filtering.
    pub top_p: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

impl SamplingConfig {
    /// Validate the hyper-parameters.
    ///
    /// # Errors
    /// - [`MultiModalError::InvalidTemperature`] when `temperature` is not a
    ///   finite positive number.
    /// - [`MultiModalError::Internal`] when `top_p` is not in `(0, 1]` (or
    ///   non-finite).
    pub fn validate(&self) -> MmResult<()> {
        if self.temperature <= 0.0 || !self.temperature.is_finite() {
            return Err(MultiModalError::InvalidTemperature {
                temp: self.temperature,
            });
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 {
            return Err(MultiModalError::Internal(
                "top_p must be finite and > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Temperature-scaled, numerically-stable softmax over `logits`.
///
/// # Errors
/// - [`MultiModalError::EmptyInput`] when `logits` is empty.
/// - [`MultiModalError::InvalidTemperature`] when `temperature <= 0` / non-finite.
pub fn temperature_softmax(logits: &[f32], temperature: f32) -> MmResult<Vec<f32>> {
    if logits.is_empty() {
        return Err(MultiModalError::EmptyInput);
    }
    if temperature <= 0.0 || !temperature.is_finite() {
        return Err(MultiModalError::InvalidTemperature { temp: temperature });
    }
    let inv_t = 1.0 / temperature;
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = vec![0.0_f32; logits.len()];
    let mut sum = 0.0_f32;
    for (p, &l) in probs.iter_mut().zip(logits.iter()) {
        let e = ((l - max_l) * inv_t).exp();
        *p = e;
        sum += e;
    }
    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }
    Ok(probs)
}

/// Return a copy of `logits` with all but the `k` highest entries set to
/// `-inf` (so a subsequent softmax assigns them zero probability). `k == 0` or
/// `k >= len` is a no-op.
///
/// # Errors
/// [`MultiModalError::EmptyInput`] when `logits` is empty.
pub fn top_k_filter(logits: &[f32], k: usize) -> MmResult<Vec<f32>> {
    if logits.is_empty() {
        return Err(MultiModalError::EmptyInput);
    }
    if k == 0 || k >= logits.len() {
        return Ok(logits.to_vec());
    }
    // Find the k-th largest logit as a threshold.
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];
    // Keep entries >= threshold, but cap the number kept at k to handle ties
    // (drop excess tied entries deterministically by index order).
    let mut kept = 0usize;
    let mut out = vec![f32::NEG_INFINITY; logits.len()];
    for (i, &l) in logits.iter().enumerate() {
        if l > threshold {
            out[i] = l;
            kept += 1;
        }
    }
    // Fill remaining slots from the tied-at-threshold entries by index order.
    if kept < k {
        for (i, &l) in logits.iter().enumerate() {
            if kept >= k {
                break;
            }
            if l == threshold && out[i] == f32::NEG_INFINITY {
                out[i] = l;
                kept += 1;
            }
        }
    }
    Ok(out)
}

/// Nucleus (top-p) filter: keep the smallest set of highest-probability tokens
/// whose cumulative probability reaches `top_p`, zeroing the rest, then
/// renormalise. `top_p >= 1.0` is a no-op (returns `probs` unchanged).
///
/// # Errors
/// - [`MultiModalError::EmptyInput`] when `probs` is empty.
/// - [`MultiModalError::Internal`] when `top_p <= 0` or non-finite.
pub fn nucleus_filter(probs: &[f32], top_p: f32) -> MmResult<Vec<f32>> {
    if probs.is_empty() {
        return Err(MultiModalError::EmptyInput);
    }
    if !top_p.is_finite() || top_p <= 0.0 {
        return Err(MultiModalError::Internal(
            "top_p must be finite and > 0".to_string(),
        ));
    }
    if top_p >= 1.0 {
        return Ok(probs.to_vec());
    }
    // Order indices by descending probability.
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut kept = vec![false; probs.len()];
    let mut cum = 0.0_f32;
    for &idx in &order {
        kept[idx] = true;
        cum += probs[idx];
        if cum >= top_p {
            break; // always include the token that crosses the threshold
        }
    }
    let mut out = vec![0.0_f32; probs.len()];
    let mut sum = 0.0_f32;
    for i in 0..probs.len() {
        if kept[i] {
            out[i] = probs[i];
            sum += probs[i];
        }
    }
    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    for v in out.iter_mut() {
        *v *= inv_sum;
    }
    Ok(out)
}

/// Inverse-CDF categorical sample from a (not necessarily normalised) weight
/// vector using the deterministic [`LcgRng`]. Returns the sampled index.
///
/// # Errors
/// - [`MultiModalError::EmptyInput`] when `weights` is empty.
/// - [`MultiModalError::Internal`] when the weights sum to a non-positive /
///   non-finite total (no valid distribution to sample).
pub fn sample_categorical(weights: &[f32], rng: &mut LcgRng) -> MmResult<usize> {
    if weights.is_empty() {
        return Err(MultiModalError::EmptyInput);
    }
    let total: f32 = weights.iter().map(|&w| w.max(0.0)).sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(MultiModalError::Internal(
            "categorical weights must sum to a positive finite value".to_string(),
        ));
    }
    let u = rng.next_f32() * total;
    let mut acc = 0.0_f32;
    for (i, &w) in weights.iter().enumerate() {
        acc += w.max(0.0);
        if u < acc {
            return Ok(i);
        }
    }
    // Floating-point guard: return the last positive-weight index.
    for i in (0..weights.len()).rev() {
        if weights[i] > 0.0 {
            return Ok(i);
        }
    }
    Ok(weights.len() - 1)
}

/// Full per-step token sampler: temperature softmax → top-k → nucleus →
/// renormalise → categorical sample.
///
/// # Errors
/// Propagates the errors of the constituent stages.
pub fn sample_token(logits: &[f32], cfg: &SamplingConfig, rng: &mut LcgRng) -> MmResult<usize> {
    cfg.validate()?;
    let filtered = top_k_filter(logits, cfg.top_k)?;
    let probs = temperature_softmax(&filtered, cfg.temperature)?;
    let nucleus = nucleus_filter(&probs, cfg.top_p)?;
    sample_categorical(&nucleus, rng)
}

/// A scored partial hypothesis used by [`beam_search`].
#[derive(Debug, Clone)]
pub struct Beam {
    /// Token ids generated so far (excluding the seed).
    pub tokens: Vec<u32>,
    /// Sum of log-probabilities along the path.
    pub log_prob: f32,
    /// Whether this beam has emitted the EOS token.
    pub finished: bool,
}

/// Length-normalised beam search.
///
/// `next_logits(&prefix)` must return the next-token logit vector (length =
/// vocabulary) given the tokens generated so far. The search keeps `beam_width`
/// hypotheses, expands each by its top-`beam_width` continuations, and ranks the
/// pool by `log_prob / len^length_penalty`. A beam that emits `eos_token` is
/// frozen. Returns the highest-scoring hypothesis after at most `max_len` steps.
///
/// # Errors
/// - [`MultiModalError::InvalidBatchSize`] when `beam_width == 0`.
/// - [`MultiModalError::EmptyInput`] when `next_logits` returns an empty vector.
/// - [`MultiModalError::Internal`] when no hypothesis is produced.
pub fn beam_search<F>(
    mut next_logits: F,
    beam_width: usize,
    max_len: usize,
    eos_token: u32,
    length_penalty: f32,
) -> MmResult<Beam>
where
    F: FnMut(&[u32]) -> MmResult<Vec<f32>>,
{
    if beam_width == 0 {
        return Err(MultiModalError::InvalidBatchSize);
    }
    let mut beams = vec![Beam {
        tokens: Vec::new(),
        log_prob: 0.0,
        finished: false,
    }];

    for _ in 0..max_len {
        if beams.iter().all(|b| b.finished) {
            break;
        }
        let mut candidates: Vec<Beam> = Vec::new();
        for beam in &beams {
            if beam.finished {
                candidates.push(beam.clone());
                continue;
            }
            let logits = next_logits(&beam.tokens)?;
            if logits.is_empty() {
                return Err(MultiModalError::EmptyInput);
            }
            let log_probs = log_softmax(&logits);
            // Top `beam_width` continuations by log-prob.
            let mut idx: Vec<usize> = (0..log_probs.len()).collect();
            idx.sort_by(|&a, &b| {
                log_probs[b]
                    .partial_cmp(&log_probs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            for &t in idx.iter().take(beam_width) {
                let mut tokens = beam.tokens.clone();
                tokens.push(t as u32);
                let finished = t as u32 == eos_token;
                candidates.push(Beam {
                    tokens,
                    log_prob: beam.log_prob + log_probs[t],
                    finished,
                });
            }
        }
        // Rank by length-normalised score and keep the top `beam_width`.
        candidates.sort_by(|a, b| {
            score(b, length_penalty)
                .partial_cmp(&score(a, length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(beam_width);
        beams = candidates;
    }

    beams
        .into_iter()
        .max_by(|a, b| {
            score(a, length_penalty)
                .partial_cmp(&score(b, length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| MultiModalError::Internal("beam search produced no hypothesis".to_string()))
}

/// Length-normalised beam score `log_prob / len^penalty` (len ≥ 1).
fn score(beam: &Beam, penalty: f32) -> f32 {
    let len = beam.tokens.len().max(1) as f32;
    beam.log_prob / len.powf(penalty)
}

/// Numerically-stable log-softmax.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for &l in logits {
        sum += (l - max_l).exp();
    }
    let log_sum = max_l + sum.ln();
    logits.iter().map(|&l| l - log_sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let logits = [1.0_f32, 2.0, 3.0, -1.0];
        let p = temperature_softmax(&logits, 1.0).expect("softmax");
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        // Monotonic: larger logit → larger prob.
        assert!(p[2] > p[1] && p[1] > p[0] && p[0] > p[3]);
    }

    #[test]
    fn high_temperature_flattens() {
        let logits = [0.0_f32, 5.0];
        let cold = temperature_softmax(&logits, 0.1).expect("cold");
        let hot = temperature_softmax(&logits, 10.0).expect("hot");
        // Cold pushes mass to the max; hot pulls towards uniform (0.5).
        assert!(cold[1] > 0.99);
        assert!((hot[1] - 0.5).abs() < 0.2);
    }

    #[test]
    fn top_k_keeps_exactly_k() {
        let logits = [3.0_f32, 1.0, 2.0, 0.5, 4.0];
        let f = top_k_filter(&logits, 2).expect("topk");
        let kept = f.iter().filter(|&&v| v.is_finite()).count();
        assert_eq!(kept, 2);
        // The two finite survivors must be the largest two (4.0 at idx4, 3.0 at idx0).
        assert!(f[4].is_finite() && f[0].is_finite());
        assert!(!f[1].is_finite() && !f[2].is_finite() && !f[3].is_finite());
    }

    #[test]
    fn top_k_handles_ties() {
        let logits = [2.0_f32, 2.0, 2.0, 1.0];
        let f = top_k_filter(&logits, 2).expect("topk");
        let kept = f.iter().filter(|&&v| v.is_finite()).count();
        assert_eq!(kept, 2, "ties must not over-keep");
    }

    #[test]
    fn nucleus_keeps_threshold_set() {
        // probs: token0 dominates with 0.6. top_p=0.5 → keep just token0.
        let probs = [0.6_f32, 0.3, 0.1];
        let f = nucleus_filter(&probs, 0.5).expect("nucleus");
        assert!(f[0] > 0.99, "nucleus should collapse to the dominant token");
        assert!(f[1].abs() < 1e-6 && f[2].abs() < 1e-6);
    }

    #[test]
    fn nucleus_renormalises() {
        let probs = [0.5_f32, 0.3, 0.2];
        let f = nucleus_filter(&probs, 0.7).expect("nucleus");
        let s: f32 = f.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "renormalised sum {s}");
    }

    #[test]
    fn nucleus_full_p_is_identity() {
        let probs = [0.5_f32, 0.3, 0.2];
        let f = nucleus_filter(&probs, 1.0).expect("nucleus");
        assert_eq!(f, probs.to_vec());
    }

    #[test]
    fn categorical_picks_only_nonzero() {
        // Only index 2 has weight → always sampled.
        let w = [0.0_f32, 0.0, 1.0, 0.0];
        let mut rng = LcgRng::new(1);
        for _ in 0..50 {
            assert_eq!(sample_categorical(&w, &mut rng).expect("cat"), 2);
        }
    }

    #[test]
    fn categorical_distribution_is_reasonable() {
        // Heavily weighted index 0 should dominate the empirical counts.
        let w = [0.9_f32, 0.05, 0.05];
        let mut rng = LcgRng::new(2);
        let mut count0 = 0;
        let n = 2000;
        for _ in 0..n {
            if sample_categorical(&w, &mut rng).expect("cat") == 0 {
                count0 += 1;
            }
        }
        let frac = count0 as f32 / n as f32;
        assert!(frac > 0.8, "index 0 fraction {frac} should be ~0.9");
    }

    #[test]
    fn categorical_deterministic_with_seed() {
        let w = [0.3_f32, 0.3, 0.4];
        let mut a = LcgRng::new(7);
        let mut b = LcgRng::new(7);
        for _ in 0..100 {
            assert_eq!(
                sample_categorical(&w, &mut a).expect("a"),
                sample_categorical(&w, &mut b).expect("b")
            );
        }
    }

    #[test]
    fn sample_token_greedy_via_low_temp_topk1() {
        // top_k=1 forces the argmax; verify it always returns the max-logit index.
        let logits = [0.1_f32, 5.0, 0.2, 0.3];
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
        };
        let mut rng = LcgRng::new(3);
        for _ in 0..20 {
            assert_eq!(sample_token(&logits, &cfg, &mut rng).expect("sample"), 1);
        }
    }

    #[test]
    fn sample_token_invalid_temperature_errors() {
        let logits = [1.0_f32, 2.0];
        let cfg = SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        };
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            sample_token(&logits, &cfg, &mut rng),
            Err(MultiModalError::InvalidTemperature { .. })
        ));
    }

    #[test]
    fn beam_search_recovers_best_path() {
        // A deterministic 2-step toy LM over a vocab of 3 tokens (0,1,2=EOS).
        // Step where prefix is empty strongly prefers token 1; after token 1 the
        // model emits EOS. Beam search must return [1, 2].
        let next = |prefix: &[u32]| -> MmResult<Vec<f32>> {
            if prefix.is_empty() {
                Ok(vec![0.0, 5.0, 0.0]) // prefer token 1
            } else {
                Ok(vec![0.0, 0.0, 5.0]) // prefer EOS (token 2)
            }
        };
        let best = beam_search(next, 2, 5, 2, 1.0).expect("beam");
        assert_eq!(best.tokens, vec![1, 2]);
        assert!(best.finished);
    }

    #[test]
    fn beam_search_width_zero_errors() {
        let next = |_: &[u32]| -> MmResult<Vec<f32>> { Ok(vec![1.0, 2.0]) };
        assert!(matches!(
            beam_search(next, 0, 3, 1, 1.0),
            Err(MultiModalError::InvalidBatchSize)
        ));
    }

    #[test]
    fn beam_search_respects_max_len() {
        // A model that never emits EOS — beam must stop at max_len.
        let next = |_: &[u32]| -> MmResult<Vec<f32>> { Ok(vec![5.0, 0.0, 0.0]) };
        let best = beam_search(next, 2, 4, 99, 1.0).expect("beam");
        assert_eq!(best.tokens.len(), 4);
        assert!(!best.finished);
    }

    #[test]
    fn beam_width_one_equals_greedy() {
        // With beam_width=1 the search is greedy argmax decoding.
        let next = |prefix: &[u32]| -> MmResult<Vec<f32>> {
            // token == position index, terminate after 3 tokens via EOS=3.
            let pos = prefix.len();
            let mut v = vec![0.0_f32; 4];
            if pos < 3 {
                v[pos] = 10.0;
            } else {
                v[3] = 10.0;
            }
            Ok(v)
        };
        let best = beam_search(next, 1, 6, 3, 1.0).expect("beam");
        assert_eq!(best.tokens, vec![0, 1, 2, 3]);
    }
}
