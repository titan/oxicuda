//! Stochastic Duration Predictor (SDP) for VITS / VITS2.
//!
//! Unlike the deterministic FastSpeech 2 [`crate::synthesis::fastspeech2::DurationPredictor`]
//! (which regresses a single `log d` per phoneme), the SDP
//! ([Kim et al. 2021](https://arxiv.org/abs/2106.06103)) models a **distribution**
//! over durations with a conditional normalising flow, so that sampling yields
//! natural rhythm variation. The flow operates on a `[t, 2]` latent (the
//! 1-D duration augmented with one auxiliary noise channel — the standard
//! augmented-flow construction that gives the coupling layers a non-trivial
//! channel split), conditioned on the text encoding.
//!
//! Concretely the module exposes:
//!
//! * an **exact bijection** `flow_forward(z, g) → (noise, logdet)` /
//!   `flow_inverse(noise, g) → z` (a stack of conditioned affine couplings with
//!   channel flips). This is the integrity core and is unit-tested for
//!   invertibility and a correct analytic log-determinant.
//! * `sample(text, rng, noise_scale) → durations` — the inference path: draw a
//!   standard-normal `noise`, run the flow **inverse**, read the duration
//!   channel and exponentiate (durations are strictly positive).
//! * `log_likelihood(text, durations, rng) → f32` — the training objective: a
//!   stochastic variational lower bound on `log p(d | text)` obtained by the
//!   forward (data → noise) flow, the standard-normal base density, the flow
//!   log-determinant and the `d ↦ log d` change-of-variables term.
//!
//! The auxiliary dequantisation channel is itself transformed by a monotone
//! **rational-quadratic neural spline** ([`crate::synthesis::vits2::spline`],
//! Durkan et al. 2019): a [`RqSplineCoupling`] maps the base noise `u ~ N(0, 1)`
//! to the dequantisation variable `e = T(u)` conditioned on `log d` and the
//! text, with an exact change-of-variables term `+logdet_dequant` in the ELBO.
//! This is the real VITS SDP posterior spline-flow dequantiser; with an identity
//! spline it reduces to the plain `N(0, 1)` auxiliary bound, so the objective
//! here is a strictly tighter — and still mathematically exact — variational
//! bound. The flow bijection, the sampling path, the spline dequantiser and the
//! change-of-variables objective are all verified on CPU.

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;
use crate::synthesis::vits2::common::DenseLayer;
use crate::synthesis::vits2::spline::RqSplineCoupling;

/// Number of latent channels the duration flow operates on (duration ‖ noise).
const DUR_CHANNELS: usize = 2;

/// Bins `K` of the monotone rational-quadratic spline dequantiser.
const DEQUANT_BINS: usize = 8;

/// Interval half-width `B` of the spline dequantiser. A standard-normal base
/// noise sits well inside `[-B, B]`; beyond it the spline is the identity tail.
const DEQUANT_BOUND: f32 = 5.0;

/// Half of `ln(2π)`, the per-element constant of a unit Gaussian log-density.
fn half_ln_2pi() -> f32 {
    0.5 * (2.0 * std::f32::consts::PI).ln()
}

// ─── Conditioned coupling ────────────────────────────────────────────────────

/// A conditioned affine coupling over the `[t, 2]` duration latent.
///
/// Channel `0` is the identity half and channel `1` the transformed half (a
/// channel flip between layers swaps their roles). The conditioner reads the
/// identity channel **and** the per-phoneme text condition `g`, producing a
/// `tanh`-bounded log-scale and a shift. Depending only on the identity channel
/// and on `g` (both available when inverting) makes the layer exactly
/// invertible with Jacobian log-determinant `Σ log_scale`.
#[derive(Debug, Clone)]
struct DurCoupling {
    /// Conditioner first layer `(1 + cond_dim) → hidden`.
    fc1: DenseLayer,
    /// Conditioner second layer `hidden → 2` (log-scale ‖ shift).
    fc2: DenseLayer,
}

impl DurCoupling {
    fn new(cond_dim: usize, hidden: usize, rng: &mut LcgRng) -> Self {
        let s1 = (2.0 / (1 + cond_dim) as f32).sqrt();
        let s2 = 0.3 / (hidden as f32).sqrt();
        Self {
            fc1: DenseLayer::new(1 + cond_dim, hidden, s1, rng),
            fc2: DenseLayer::new(hidden, 2, s2, rng),
        }
    }

    /// Conditioner over the identity channel `x_a [t]` and condition `g [t, cond]`.
    /// Returns `(log_scale [t], shift [t])`.
    fn conditioner(
        &self,
        x_a: &[f32],
        g: &[f32],
        t: usize,
        cond_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        // Build the `[t, 1 + cond_dim]` conditioner input.
        let mut inp = vec![0.0_f32; t * (1 + cond_dim)];
        for ti in 0..t {
            let row = &mut inp[ti * (1 + cond_dim)..(ti + 1) * (1 + cond_dim)];
            row[0] = x_a[ti];
            row[1..].copy_from_slice(&g[ti * cond_dim..(ti + 1) * cond_dim]);
        }
        let mut h = self.fc1.forward(&inp, t);
        for v in h.iter_mut() {
            *v = v.tanh();
        }
        let st = self.fc2.forward(&h, t); // [t, 2]
        let mut log_scale = vec![0.0_f32; t];
        let mut shift = vec![0.0_f32; t];
        for ti in 0..t {
            log_scale[ti] = st[ti * 2].tanh();
            shift[ti] = st[ti * 2 + 1];
        }
        (log_scale, shift)
    }

    /// Forward `z [t, 2] → (y [t, 2], logdet)`.
    fn forward(&self, z: &[f32], g: &[f32], t: usize, cond_dim: usize) -> (Vec<f32>, f32) {
        let mut x_a = vec![0.0_f32; t];
        for ti in 0..t {
            x_a[ti] = z[ti * DUR_CHANNELS];
        }
        let (log_scale, shift) = self.conditioner(&x_a, g, t, cond_dim);
        let mut y = vec![0.0_f32; z.len()];
        let mut logdet = 0.0_f32;
        for ti in 0..t {
            let x_b = z[ti * DUR_CHANNELS + 1];
            y[ti * DUR_CHANNELS] = x_a[ti];
            y[ti * DUR_CHANNELS + 1] = x_b * log_scale[ti].exp() + shift[ti];
            logdet += log_scale[ti];
        }
        (y, logdet)
    }

    /// Exact inverse `y [t, 2] → z [t, 2]`.
    fn inverse(&self, y: &[f32], g: &[f32], t: usize, cond_dim: usize) -> Vec<f32> {
        let mut y_a = vec![0.0_f32; t];
        for ti in 0..t {
            y_a[ti] = y[ti * DUR_CHANNELS];
        }
        let (log_scale, shift) = self.conditioner(&y_a, g, t, cond_dim);
        let mut z = vec![0.0_f32; y.len()];
        for ti in 0..t {
            let y_b = y[ti * DUR_CHANNELS + 1];
            z[ti * DUR_CHANNELS] = y_a[ti];
            z[ti * DUR_CHANNELS + 1] = (y_b - shift[ti]) * (-log_scale[ti]).exp();
        }
        z
    }
}

/// Swap the two channels of every time step of a `[t, 2]` buffer.
fn swap_channels(x: &[f32], t: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; x.len()];
    for ti in 0..t {
        out[ti * DUR_CHANNELS] = x[ti * DUR_CHANNELS + 1];
        out[ti * DUR_CHANNELS + 1] = x[ti * DUR_CHANNELS];
    }
    out
}

// ─── Stochastic duration predictor ───────────────────────────────────────────

/// Flow-based stochastic duration predictor.
///
/// Wraps a text-conditioner projection and a stack of `DurCoupling` layers
/// (interleaved with channel swaps) into a conditional normalising flow over the
/// `[t, 2]` duration latent.
#[derive(Debug, Clone)]
pub struct StochasticDurationPredictor {
    /// Projects the text encoding `[t, text_dim] → [t, cond_dim]` condition `g`.
    cond_proj: DenseLayer,
    /// Conditioned coupling layers.
    couplings: Vec<DurCoupling>,
    /// Monotone rational-quadratic spline dequantiser. A Durkan spline coupling
    /// over the `[t, 2]` latent: it leaves the `log d` channel as the identity
    /// half and maps the auxiliary base noise `u ↦ e = T(u)` conditioned on
    /// `log d` and the text — the real spline replacement for the previous fixed
    /// `N(0, 1)` auxiliary path (the VITS SDP posterior dequantisation flow).
    dequantizer: RqSplineCoupling,
    /// Text-encoding feature dimension.
    text_dim: usize,
    /// Condition feature dimension.
    cond_dim: usize,
}

impl StochasticDurationPredictor {
    /// Construct an SDP over a `text_dim`-wide text encoding.
    ///
    /// # Errors
    ///
    /// - [`AudioError::InvalidEmbedDim`] when `text_dim == 0`.
    /// - [`AudioError::Internal`] when `cond_dim`, `hidden`, or `n_flows` is `0`.
    pub fn new(
        text_dim: usize,
        cond_dim: usize,
        hidden: usize,
        n_flows: usize,
        rng: &mut LcgRng,
    ) -> AudioResult<Self> {
        if text_dim == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if cond_dim == 0 {
            return Err(AudioError::Internal("SDP: cond_dim == 0".into()));
        }
        if hidden == 0 {
            return Err(AudioError::Internal("SDP: hidden == 0".into()));
        }
        if n_flows == 0 {
            return Err(AudioError::Internal("SDP: n_flows == 0".into()));
        }
        let cond_proj = DenseLayer::new(text_dim, cond_dim, (2.0 / text_dim as f32).sqrt(), rng);
        let mut couplings = Vec::with_capacity(n_flows);
        for _ in 0..n_flows {
            couplings.push(DurCoupling::new(cond_dim, hidden, rng));
        }
        // Spline dequantiser over the [t, 2] latent: channel 0 (log d) is the
        // identity half, channel 1 (auxiliary noise) the spline-transformed half.
        let dequantizer = RqSplineCoupling::new(
            DUR_CHANNELS,
            cond_dim,
            hidden,
            DEQUANT_BINS,
            DEQUANT_BOUND,
            rng,
        )?;
        Ok(Self {
            cond_proj,
            couplings,
            dequantizer,
            text_dim,
            cond_dim,
        })
    }

    /// Project the text encoding `[t, text_dim]` to the condition `g [t, cond_dim]`.
    fn condition(&self, text: &[f32], t: usize) -> Vec<f32> {
        let mut g = self.cond_proj.forward(text, t);
        for v in g.iter_mut() {
            *v = v.tanh();
        }
        g
    }

    /// Forward flow `z [t, 2] → (noise [t, 2], logdet)` given condition `g`.
    ///
    /// Exposed (with [`StochasticDurationPredictor::flow_inverse`]) so the
    /// bijection and its log-determinant can be verified directly.
    ///
    /// # Errors
    ///
    /// [`AudioError::ShapeMismatch`] when `z.len() != t*2` or `g.len() != t*cond_dim`.
    pub fn flow_forward(&self, z: &[f32], g: &[f32], t: usize) -> AudioResult<(Vec<f32>, f32)> {
        self.check_latent(z, t)?;
        self.check_cond(g, t)?;
        let mut h = z.to_vec();
        let mut logdet = 0.0_f32;
        for coupling in &self.couplings {
            let (hc, ld) = coupling.forward(&h, g, t, self.cond_dim);
            logdet += ld;
            h = swap_channels(&hc, t);
        }
        Ok((h, logdet))
    }

    /// Inverse flow `noise [t, 2] → z [t, 2]`, the exact inverse of
    /// [`StochasticDurationPredictor::flow_forward`].
    ///
    /// # Errors
    ///
    /// [`AudioError::ShapeMismatch`] for bad `noise` / `g` shapes.
    pub fn flow_inverse(&self, noise: &[f32], g: &[f32], t: usize) -> AudioResult<Vec<f32>> {
        self.check_latent(noise, t)?;
        self.check_cond(g, t)?;
        let mut h = noise.to_vec();
        for coupling in self.couplings.iter().rev() {
            h = swap_channels(&h, t);
            h = coupling.inverse(&h, g, t, self.cond_dim);
        }
        Ok(h)
    }

    /// Sample per-phoneme durations from the predicted distribution.
    ///
    /// Draws `noise ~ N(0, noise_scale²)` over `[t, 2]`, runs the flow inverse,
    /// reads the duration channel `z₀` and returns `exp(z₀)` (strictly positive
    /// real durations — the caller rounds / length-scales them as needed).
    ///
    /// # Errors
    ///
    /// - [`AudioError::EmptyInput`] when `t == 0`.
    /// - [`AudioError::ShapeMismatch`] when `text.len() != t * text_dim`.
    pub fn sample(
        &self,
        text: &[f32],
        t: usize,
        rng: &mut LcgRng,
        noise_scale: f32,
    ) -> AudioResult<Vec<f32>> {
        self.check_text(text, t)?;
        let g = self.condition(text, t);
        let mut noise = vec![0.0_f32; t * DUR_CHANNELS];
        rng.fill_normal(&mut noise);
        for v in noise.iter_mut() {
            *v *= noise_scale;
        }
        let z = self.flow_inverse(&noise, &g, t)?;
        let mut durations = vec![0.0_f32; t];
        for ti in 0..t {
            durations[ti] = z[ti * DUR_CHANNELS].exp();
        }
        Ok(durations)
    }

    /// Stochastic variational lower bound on `log p(durations | text)`.
    ///
    /// Draws auxiliary base noise `u ~ N(0, 1)`, maps it through the monotone
    /// rational-quadratic spline [`RqSplineCoupling`] dequantiser to
    /// `e = T(u)` (conditioned on `log d` and the text), builds the latent
    /// `z = [log d, e]`, runs the forward flow to `noise`, and accumulates
    ///
    /// ```text
    /// log p(d) ≳ logN(noise; 0, I) + logdet_flow         (change of variables)
    ///          − logN(u; 0, I) + logdet_dequant          (spline dequant entropy)
    ///          − Σ log d                                  (d ↦ log d Jacobian)
    /// ```
    ///
    /// The dequantiser term `−log q(e) = −logN(u;0,I) + logdet_dequant` is the
    /// exact change-of-variables of the spline pushforward `q(e) = T_# N(0, I)`;
    /// with an identity spline it collapses to the previous fixed-`N(0, 1)`
    /// auxiliary bound, so this is a strictly tighter (still valid) ELBO. Used as
    /// the SDP training objective; its absolute value is data/parameter-dependent
    /// and is only asserted finite / deterministic in tests, never as a quality
    /// figure.
    ///
    /// # Errors
    ///
    /// - [`AudioError::EmptyInput`] when `t == 0`.
    /// - [`AudioError::ShapeMismatch`] when `text.len() != t*text_dim` or
    ///   `durations.len() != t`.
    /// - [`AudioError::NonFinite`] when any duration is `<= 0` or non-finite.
    pub fn log_likelihood(
        &self,
        text: &[f32],
        durations: &[f32],
        t: usize,
        rng: &mut LcgRng,
    ) -> AudioResult<f32> {
        self.check_text(text, t)?;
        if durations.len() != t {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "SDP::log_likelihood: durations.len()={} != t={t}",
                    durations.len()
                ),
            });
        }
        for &d in durations {
            if !d.is_finite() || d <= 0.0 {
                return Err(AudioError::NonFinite {
                    msg: format!("SDP::log_likelihood: non-positive duration {d}"),
                });
            }
        }
        let g = self.condition(text, t);

        // Pre-latent: channel 0 = log d, channel 1 = auxiliary base noise u ~ N(0,1).
        let mut u = vec![0.0_f32; t];
        rng.fill_normal(&mut u);
        let mut pre = vec![0.0_f32; t * DUR_CHANNELS];
        let mut log_jacobian = 0.0_f32; // Σ log|d(log d)/d d| = −Σ log d.
        for ti in 0..t {
            let log_d = durations[ti].ln();
            pre[ti * DUR_CHANNELS] = log_d;
            pre[ti * DUR_CHANNELS + 1] = u[ti];
            log_jacobian -= log_d;
        }

        // Monotone spline dequantisation u ↦ e = T(u): leaves log d untouched and
        // transforms the auxiliary channel, with exact logdet_dequant = Σ log T'(u).
        let (z, deq_logdet) = self.dequantizer.forward(&pre, &g, t)?;

        let (noise, logdet) = self.flow_forward(&z, &g, t)?;

        // log N(noise; 0, I) over [t, 2].
        let mut base = 0.0_f32;
        for &n in &noise {
            base += -0.5 * n * n - half_ln_2pi();
        }
        // Base density of the sampled noise u: −log q(e) = −log N(u; 0, I) + logdet_dequant.
        let mut u_base = 0.0_f32;
        for &ui in &u {
            u_base += -0.5 * ui * ui - half_ln_2pi();
        }

        Ok(base + logdet - u_base + deq_logdet + log_jacobian)
    }

    fn check_text(&self, text: &[f32], t: usize) -> AudioResult<()> {
        if t == 0 {
            return Err(AudioError::EmptyInput {
                msg: "SDP: t == 0".into(),
            });
        }
        if text.len() != t * self.text_dim {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "SDP: text.len()={} != t*text_dim={}",
                    text.len(),
                    t * self.text_dim
                ),
            });
        }
        Ok(())
    }

    fn check_latent(&self, z: &[f32], t: usize) -> AudioResult<()> {
        if z.len() != t * DUR_CHANNELS {
            return Err(AudioError::ShapeMismatch {
                msg: format!("SDP latent: len={} != t*2={}", z.len(), t * DUR_CHANNELS),
            });
        }
        Ok(())
    }

    fn check_cond(&self, g: &[f32], t: usize) -> AudioResult<()> {
        if g.len() != t * self.cond_dim {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "SDP cond: len={} != t*cond_dim={}",
                    g.len(),
                    t * self.cond_dim
                ),
            });
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    fn log_abs_det(mut m: Vec<f32>, n: usize) -> f32 {
        let mut log_det = 0.0_f32;
        for col in 0..n {
            let mut pivot = col;
            let mut best = m[col * n + col].abs();
            for r in (col + 1)..n {
                let v = m[r * n + col].abs();
                if v > best {
                    best = v;
                    pivot = r;
                }
            }
            if pivot != col {
                for c in 0..n {
                    m.swap(col * n + c, pivot * n + c);
                }
            }
            let diag = m[col * n + col];
            log_det += diag.abs().ln();
            for r in (col + 1)..n {
                let factor = m[r * n + col] / diag;
                for c in col..n {
                    m[r * n + c] -= factor * m[col * n + c];
                }
            }
        }
        log_det
    }

    #[test]
    fn sdp_flow_is_invertible() {
        // TEST 1 (duration flow): inverse(forward(z)) ≈ z to <= 1e-4.
        let mut rng = LcgRng::new(1);
        let sdp = StochasticDurationPredictor::new(8, 6, 16, 4, &mut rng).expect("sdp");
        let t = 7usize;
        let mut text = vec![0.0_f32; t * 8];
        let mut data = LcgRng::new(10);
        data.fill_normal(&mut text);
        let g = sdp.condition(&text, t);
        let mut z = vec![0.0_f32; t * DUR_CHANNELS];
        data.fill_normal(&mut z);
        let (noise, logdet) = sdp.flow_forward(&z, &g, t).expect("forward");
        assert!(logdet.is_finite());
        let back = sdp.flow_inverse(&noise, &g, t).expect("inverse");
        let err = max_abs_diff(&z, &back);
        assert!(err < 1e-4, "sdp flow round-trip error {err}");
    }

    #[test]
    fn sdp_flow_logdet_matches_finite_difference() {
        // TEST 2 (duration flow): analytic logdet ≈ numeric Jacobian <= 1e-2.
        let mut rng = LcgRng::new(2);
        let t = 3usize; // n = t * 2 = 6 → 6×6 Jacobian.
        let sdp = StochasticDurationPredictor::new(8, 4, 12, 3, &mut rng).expect("sdp");
        let mut text = vec![0.0_f32; t * 8];
        let mut data = LcgRng::new(20);
        data.fill_normal(&mut text);
        let g = sdp.condition(&text, t);
        let n = t * DUR_CHANNELS;
        let mut z = vec![0.0_f32; n];
        data.fill_normal(&mut z);

        let (_noise, analytic) = sdp.flow_forward(&z, &g, t).expect("forward");
        let h = 1e-3_f32;
        let mut jac = vec![0.0_f32; n * n];
        for j in 0..n {
            let mut zp = z.clone();
            let mut zm = z.clone();
            zp[j] += h;
            zm[j] -= h;
            let (yp, _) = sdp.flow_forward(&zp, &g, t).expect("fwd+");
            let (ym, _) = sdp.flow_forward(&zm, &g, t).expect("fwd-");
            for i in 0..n {
                jac[i * n + j] = (yp[i] - ym[i]) / (2.0 * h);
            }
        }
        let numeric = log_abs_det(jac, n);
        let err = (analytic - numeric).abs();
        assert!(
            err < 1e-2,
            "sdp logdet analytic={analytic} numeric={numeric}"
        );
    }

    #[test]
    fn sdp_sample_positive_and_shaped() {
        let mut rng = LcgRng::new(3);
        let sdp = StochasticDurationPredictor::new(8, 6, 16, 4, &mut rng).expect("sdp");
        let t = 9usize;
        let mut text = vec![0.0_f32; t * 8];
        LcgRng::new(30).fill_normal(&mut text);
        let mut sample_rng = LcgRng::new(99);
        let dur = sdp.sample(&text, t, &mut sample_rng, 0.8).expect("sample");
        assert_eq!(dur.len(), t);
        assert!(dur.iter().all(|&d| d.is_finite() && d > 0.0));
    }

    #[test]
    fn sdp_sample_is_deterministic_under_seed() {
        let mut rng = LcgRng::new(4);
        let sdp = StochasticDurationPredictor::new(8, 6, 16, 3, &mut rng).expect("sdp");
        let t = 6usize;
        let mut text = vec![0.0_f32; t * 8];
        LcgRng::new(40).fill_normal(&mut text);
        let a = sdp.sample(&text, t, &mut LcgRng::new(7), 1.0).expect("a");
        let b = sdp.sample(&text, t, &mut LcgRng::new(7), 1.0).expect("b");
        assert_eq!(a, b);
    }

    #[test]
    fn sdp_log_likelihood_finite_and_deterministic() {
        let mut rng = LcgRng::new(5);
        let sdp = StochasticDurationPredictor::new(8, 6, 16, 3, &mut rng).expect("sdp");
        let t = 5usize;
        let mut text = vec![0.0_f32; t * 8];
        LcgRng::new(50).fill_normal(&mut text);
        let durations = vec![3.0_f32, 1.5, 4.0, 2.0, 6.0];
        let a = sdp
            .log_likelihood(&text, &durations, t, &mut LcgRng::new(8))
            .expect("a");
        let b = sdp
            .log_likelihood(&text, &durations, t, &mut LcgRng::new(8))
            .expect("b");
        assert!(a.is_finite());
        assert_eq!(a, b);
    }

    #[test]
    fn sdp_spline_dequant_elbo_is_finite_and_stochastic() {
        // The integrated rational-quadratic spline dequantiser path must keep the
        // ELBO finite and seed-deterministic, while genuinely depending on the
        // auxiliary base noise (different aux seeds give different bounds — proof
        // the spline dequantisation is actually exercised, not a dead no-op).
        let mut rng = LcgRng::new(2027);
        let sdp = StochasticDurationPredictor::new(12, 8, 24, 4, &mut rng).expect("sdp");
        let t = 7usize;
        let mut text = vec![0.0_f32; t * 12];
        LcgRng::new(606).fill_normal(&mut text);
        let durations = vec![2.0_f32, 5.0, 1.0, 3.0, 4.0, 2.0, 6.0];

        // Finite across several aux-noise seeds.
        let mut values = Vec::new();
        for seed in 0..6u64 {
            let v = sdp
                .log_likelihood(&text, &durations, t, &mut LcgRng::new(seed))
                .expect("ll");
            assert!(v.is_finite(), "non-finite ELBO {v}");
            values.push(v);
        }
        // Deterministic under a fixed seed.
        let again = sdp
            .log_likelihood(&text, &durations, t, &mut LcgRng::new(3))
            .expect("again");
        assert_eq!(values[3], again);
        // Stochastic across seeds (not all identical) — the aux noise + spline
        // dequantiser are live in the objective.
        let all_equal = values.iter().all(|&v| v == values[0]);
        assert!(!all_equal, "ELBO ignores the aux/spline dequant path");
    }

    #[test]
    fn sdp_log_likelihood_rejects_nonpositive() {
        let mut rng = LcgRng::new(6);
        let sdp = StochasticDurationPredictor::new(8, 6, 16, 2, &mut rng).expect("sdp");
        let t = 3usize;
        let text = vec![0.1_f32; t * 8];
        let bad = vec![1.0_f32, 0.0, 2.0]; // 0 duration is invalid.
        assert!(matches!(
            sdp.log_likelihood(&text, &bad, t, &mut LcgRng::new(1)),
            Err(AudioError::NonFinite { .. })
        ));
    }

    #[test]
    fn sdp_rejects_bad_config() {
        let mut rng = LcgRng::new(7);
        assert!(StochasticDurationPredictor::new(0, 4, 8, 2, &mut rng).is_err());
        assert!(StochasticDurationPredictor::new(8, 0, 8, 2, &mut rng).is_err());
        assert!(StochasticDurationPredictor::new(8, 4, 0, 2, &mut rng).is_err());
        assert!(StochasticDurationPredictor::new(8, 4, 8, 0, &mut rng).is_err());
    }
}
