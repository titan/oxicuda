//! Invertible normalising-flow stack for VITS2 (`Vits2Flow`).
//!
//! The prior of a VITS conditional VAE is enhanced by a normalising flow `f`
//! that maps the posterior latent `z` to a point `z_p = f(z)` evaluated under a
//! Gaussian prior. To remain a valid density transform the flow must be an
//! **exact bijection** with an analytically known log-determinant of its
//! Jacobian.
//!
//! Each flow step here is the Glow factorisation
//! ([Kingma & Dhariwal 2018](https://arxiv.org/abs/1807.03039)):
//!
//! ```text
//! x → ActNorm → AffineCoupling → channel-flip → y
//! ```
//!
//! * **ActNorm** — a per-channel affine transform `y = x·exp(s) + b`. Its
//!   log-determinant is `t · Σ_c s_c` (one term per channel, replicated over
//!   the `t` time steps). Invertible by construction.
//! * **AffineCoupling** — a RealNVP / Glow affine coupling
//!   ([Dinh et al. 2017](https://arxiv.org/abs/1605.08803)). The channel axis is
//!   split into an *identity* half `x_a` and a *transformed* half `x_b`. A small
//!   conditioner network reads **only** `x_a` and emits a per-element log-scale
//!   `s = tanh(·)` and shift `t`; then `y_b = x_b·exp(s) + t`. Because the
//!   conditioner depends only on the untouched half, the map inverts exactly and
//!   its Jacobian is triangular with log-determinant `Σ s`.
//! * **channel-flip** — a fixed reversal permutation so that successive coupling
//!   layers transform every channel. A permutation has determinant `±1`, hence
//!   log-determinant `0`.
//!
//! Stacking `K` such steps gives `forward(x) → (z, logdet)` and `inverse(z) → x`
//! with `logdet = Σ_steps (actnorm_logdet + coupling_logdet)`. Both directions
//! are verified by the unit tests: `inverse(forward(x)) ≈ x` to `1e-4`, and the
//! analytic `logdet` matches the log-determinant of the numerically estimated
//! Jacobian (finite differences) to `1e-2`.

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;
use crate::synthesis::vits2::common::{DenseLayer, make_normal_vec};

// ─── ActNorm ───────────────────────────────────────────────────────────────────

/// Per-channel activation normalisation `y[:, c] = x[:, c]·exp(log_scale[c]) +
/// bias[c]`.
///
/// The classic Glow ActNorm initialises `log_scale` / `bias` from the first data
/// batch so the post-transform activations are unit-variance / zero-mean. Here
/// the parameters are seeded deterministically (data-independent) — functionally
/// still a per-channel affine bijection with an exact log-determinant; only the
/// *initial* statistics differ, which is irrelevant to invertibility and to the
/// analytic Jacobian. Its log-determinant is `t · Σ_c log_scale[c]`.
#[derive(Debug, Clone)]
pub struct ActNorm {
    /// Per-channel log-scale `[channels]`.
    pub log_scale: Vec<f32>,
    /// Per-channel bias `[channels]`.
    pub bias: Vec<f32>,
    /// Number of channels.
    pub channels: usize,
}

impl ActNorm {
    /// Construct an ActNorm with small deterministic `log_scale` / `bias`.
    fn new(channels: usize, rng: &mut LcgRng) -> Self {
        Self {
            log_scale: make_normal_vec(channels, 0.1, rng),
            bias: make_normal_vec(channels, 0.1, rng),
            channels,
        }
    }

    /// Forward transform of `x` of `[t, channels]`. Returns `(y, logdet)`.
    fn forward(&self, x: &[f32], t: usize) -> (Vec<f32>, f32) {
        let c = self.channels;
        let mut y = vec![0.0_f32; x.len()];
        for ti in 0..t {
            for ch in 0..c {
                let idx = ti * c + ch;
                y[idx] = x[idx] * self.log_scale[ch].exp() + self.bias[ch];
            }
        }
        let logdet = t as f32 * self.log_scale.iter().sum::<f32>();
        (y, logdet)
    }

    /// Exact inverse of [`ActNorm::forward`] (drops the log-determinant).
    fn inverse(&self, y: &[f32], t: usize) -> Vec<f32> {
        let c = self.channels;
        let mut x = vec![0.0_f32; y.len()];
        for ti in 0..t {
            for ch in 0..c {
                let idx = ti * c + ch;
                x[idx] = (y[idx] - self.bias[ch]) * (-self.log_scale[ch]).exp();
            }
        }
        x
    }
}

// ─── Affine coupling ───────────────────────────────────────────────────────────

/// A RealNVP / Glow affine coupling layer over a `[t, channels]` sequence.
///
/// The channels are split `[0, c_a)` (identity half) and `[c_a, channels)`
/// (transformed half). A two-layer `tanh`-MLP conditioner reads the identity
/// half and produces a per-element log-scale and shift for the transformed
/// half. The log-scale is squashed through `tanh`, bounding the per-step volume
/// change and keeping the inverse numerically robust.
#[derive(Debug, Clone)]
pub struct AffineCoupling {
    /// Conditioner first layer `c_a → hidden`.
    fc1: DenseLayer,
    /// Conditioner second layer `hidden → 2·c_b` (log-scale ‖ shift).
    fc2: DenseLayer,
    /// Identity-half channel count.
    c_a: usize,
    /// Transformed-half channel count.
    c_b: usize,
    /// Total channel count (`c_a + c_b`).
    channels: usize,
}

impl AffineCoupling {
    /// Construct a coupling layer over `channels` (`>= 2`) channels.
    fn new(channels: usize, hidden: usize, rng: &mut LcgRng) -> Self {
        let c_a = channels / 2;
        let c_b = channels - c_a;
        let s1 = (2.0 / c_a.max(1) as f32).sqrt();
        // A modest second-layer scale keeps the initial log-scales non-trivial
        // (so the flow is genuinely volume-changing) while `tanh` bounds them.
        let s2 = 0.3 / (hidden as f32).sqrt();
        Self {
            fc1: DenseLayer::new(c_a, hidden, s1, rng),
            fc2: DenseLayer::new(hidden, 2 * c_b, s2, rng),
            c_a,
            c_b,
            channels,
        }
    }

    /// Conditioner: `x_a [t, c_a] → (log_scale [t, c_b], shift [t, c_b])`.
    fn conditioner(&self, x_a: &[f32], t: usize) -> (Vec<f32>, Vec<f32>) {
        let mut h = self.fc1.forward(x_a, t);
        for v in h.iter_mut() {
            *v = v.tanh();
        }
        let st = self.fc2.forward(&h, t); // [t, 2*c_b]
        let mut log_scale = vec![0.0_f32; t * self.c_b];
        let mut shift = vec![0.0_f32; t * self.c_b];
        for ti in 0..t {
            let src = &st[ti * 2 * self.c_b..(ti + 1) * 2 * self.c_b];
            let ls = &mut log_scale[ti * self.c_b..(ti + 1) * self.c_b];
            let sh = &mut shift[ti * self.c_b..(ti + 1) * self.c_b];
            for j in 0..self.c_b {
                ls[j] = src[j].tanh(); // bounded log-scale in (-1, 1)
                sh[j] = src[self.c_b + j];
            }
        }
        (log_scale, shift)
    }

    /// Split a `[t, channels]` buffer into its identity / transformed halves.
    fn split(&self, x: &[f32], t: usize) -> (Vec<f32>, Vec<f32>) {
        let mut x_a = vec![0.0_f32; t * self.c_a];
        let mut x_b = vec![0.0_f32; t * self.c_b];
        for ti in 0..t {
            let row = &x[ti * self.channels..(ti + 1) * self.channels];
            x_a[ti * self.c_a..(ti + 1) * self.c_a].copy_from_slice(&row[..self.c_a]);
            x_b[ti * self.c_b..(ti + 1) * self.c_b].copy_from_slice(&row[self.c_a..]);
        }
        (x_a, x_b)
    }

    /// Re-join identity / transformed halves into `[t, channels]`.
    fn join(&self, x_a: &[f32], x_b: &[f32], t: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; t * self.channels];
        for ti in 0..t {
            let row = &mut out[ti * self.channels..(ti + 1) * self.channels];
            row[..self.c_a].copy_from_slice(&x_a[ti * self.c_a..(ti + 1) * self.c_a]);
            row[self.c_a..].copy_from_slice(&x_b[ti * self.c_b..(ti + 1) * self.c_b]);
        }
        out
    }

    /// Forward transform of `x` of `[t, channels]`. Returns `(y, logdet)`.
    fn forward(&self, x: &[f32], t: usize) -> (Vec<f32>, f32) {
        let (x_a, x_b) = self.split(x, t);
        let (log_scale, shift) = self.conditioner(&x_a, t);
        let mut y_b = vec![0.0_f32; x_b.len()];
        let mut logdet = 0.0_f32;
        for (i, ((&xb, &ls), &sh)) in x_b
            .iter()
            .zip(log_scale.iter())
            .zip(shift.iter())
            .enumerate()
        {
            y_b[i] = xb * ls.exp() + sh;
            logdet += ls;
        }
        (self.join(&x_a, &y_b, t), logdet)
    }

    /// Exact inverse of [`AffineCoupling::forward`].
    fn inverse(&self, y: &[f32], t: usize) -> Vec<f32> {
        let (y_a, y_b) = self.split(y, t);
        // The conditioner reads only the identity half, which the forward pass
        // left untouched (`y_a == x_a`), so it reproduces the exact same scale.
        let (log_scale, shift) = self.conditioner(&y_a, t);
        let mut x_b = vec![0.0_f32; y_b.len()];
        for (i, ((&yb, &ls), &sh)) in y_b
            .iter()
            .zip(log_scale.iter())
            .zip(shift.iter())
            .enumerate()
        {
            x_b[i] = (yb - sh) * (-ls).exp();
        }
        self.join(&y_a, &x_b, t)
    }
}

// ─── Channel flip ───────────────────────────────────────────────────────────────

/// Reverse the channel order of every time step of a `[t, channels]` buffer.
///
/// A fixed reversal is its own inverse and has Jacobian determinant `±1`, so it
/// contributes `0` to the flow log-determinant while ensuring that consecutive
/// coupling layers transform complementary channel sets.
fn channel_flip(x: &[f32], t: usize, channels: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; x.len()];
    for ti in 0..t {
        let src = &x[ti * channels..(ti + 1) * channels];
        let dst = &mut out[ti * channels..(ti + 1) * channels];
        for (ch, slot) in dst.iter_mut().enumerate() {
            *slot = src[channels - 1 - ch];
        }
    }
    out
}

// ─── Flow stack ─────────────────────────────────────────────────────────────────

/// A stack of `K` Glow steps `[ActNorm → AffineCoupling → channel-flip]`.
///
/// This is the VITS prior flow: an **exact bijection** with an analytic
/// log-determinant. `forward` maps a posterior latent to the prior space and
/// returns the accumulated `logdet`; `inverse` reconstructs the latent.
#[derive(Debug, Clone)]
pub struct Vits2Flow {
    /// Per-step activation-norm layers.
    acts: Vec<ActNorm>,
    /// Per-step affine coupling layers.
    couplings: Vec<AffineCoupling>,
    /// Channel (latent) dimension.
    channels: usize,
}

impl Vits2Flow {
    /// Build a flow with `n_layers` Glow steps over `channels` channels.
    ///
    /// # Errors
    ///
    /// - [`AudioError::InvalidEmbedDim`] when `channels < 2` (affine coupling
    ///   needs at least one channel in each half).
    /// - [`AudioError::Internal`] when `n_layers == 0` or `hidden == 0`.
    pub fn new(
        channels: usize,
        hidden: usize,
        n_layers: usize,
        rng: &mut LcgRng,
    ) -> AudioResult<Self> {
        if channels < 2 {
            return Err(AudioError::InvalidEmbedDim(channels));
        }
        if n_layers == 0 {
            return Err(AudioError::Internal("Vits2Flow: n_layers == 0".into()));
        }
        if hidden == 0 {
            return Err(AudioError::Internal("Vits2Flow: hidden == 0".into()));
        }
        let mut acts = Vec::with_capacity(n_layers);
        let mut couplings = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            acts.push(ActNorm::new(channels, rng));
            couplings.push(AffineCoupling::new(channels, hidden, rng));
        }
        Ok(Self {
            acts,
            couplings,
            channels,
        })
    }

    /// Channel (latent) dimension this flow operates on.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Validate that `x.len() == t * channels` and `t > 0`.
    fn check(&self, x: &[f32], t: usize) -> AudioResult<()> {
        if t == 0 {
            return Err(AudioError::EmptyInput {
                msg: "Vits2Flow: t == 0".into(),
            });
        }
        if x.len() != t * self.channels {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "Vits2Flow: x.len()={} != t*channels={}",
                    x.len(),
                    t * self.channels
                ),
            });
        }
        Ok(())
    }

    /// Forward pass `x → (z, logdet)` over `x` of shape `[t, channels]`.
    ///
    /// `logdet` is the exact `log|det ∂z/∂x|`, the analytic sum of all ActNorm
    /// and coupling log-scale terms (the flips contribute `0`).
    ///
    /// # Errors
    ///
    /// [`AudioError::EmptyInput`] / [`AudioError::ShapeMismatch`] for bad shape.
    pub fn forward(&self, x: &[f32], t: usize) -> AudioResult<(Vec<f32>, f32)> {
        self.check(x, t)?;
        let mut h = x.to_vec();
        let mut logdet = 0.0_f32;
        for (act, coupling) in self.acts.iter().zip(self.couplings.iter()) {
            let (ha, ld_a) = act.forward(&h, t);
            logdet += ld_a;
            let (hc, ld_c) = coupling.forward(&ha, t);
            logdet += ld_c;
            h = channel_flip(&hc, t, self.channels);
        }
        Ok((h, logdet))
    }

    /// Inverse pass `z → x`, the exact inverse of [`Vits2Flow::forward`].
    ///
    /// # Errors
    ///
    /// [`AudioError::EmptyInput`] / [`AudioError::ShapeMismatch`] for bad shape.
    pub fn inverse(&self, z: &[f32], t: usize) -> AudioResult<Vec<f32>> {
        self.check(z, t)?;
        let mut h = z.to_vec();
        for (act, coupling) in self.acts.iter().zip(self.couplings.iter()).rev() {
            h = channel_flip(&h, t, self.channels);
            h = coupling.inverse(&h, t);
            h = act.inverse(&h, t);
        }
        Ok(h)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Maximum absolute difference between two equal-length buffers.
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// `log|det J|` of a square `[n, n]` matrix via Gaussian elimination with
    /// partial pivoting (test-only reference for the finite-difference check).
    fn log_abs_det(mut m: Vec<f32>, n: usize) -> f32 {
        let mut log_det = 0.0_f32;
        for col in 0..n {
            // Partial pivot.
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
    fn actnorm_is_invertible() {
        let mut rng = LcgRng::new(1);
        let act = ActNorm::new(4, &mut rng);
        let t = 6usize;
        let mut x = vec![0.0_f32; t * 4];
        rng.fill_normal(&mut x);
        let (y, _ld) = act.forward(&x, t);
        let back = act.inverse(&y, t);
        assert!(max_abs_diff(&x, &back) < 1e-5);
    }

    #[test]
    fn coupling_is_invertible() {
        let mut rng = LcgRng::new(2);
        let coupling = AffineCoupling::new(6, 16, &mut rng);
        let t = 5usize;
        let mut x = vec![0.0_f32; t * 6];
        rng.fill_normal(&mut x);
        let (y, _ld) = coupling.forward(&x, t);
        let back = coupling.inverse(&y, t);
        assert!(max_abs_diff(&x, &back) < 1e-5);
    }

    #[test]
    fn channel_flip_is_self_inverse() {
        let t = 3usize;
        let c = 5usize;
        let mut rng = LcgRng::new(3);
        let mut x = vec![0.0_f32; t * c];
        rng.fill_normal(&mut x);
        let once = channel_flip(&x, t, c);
        let twice = channel_flip(&once, t, c);
        assert_eq!(x, twice);
    }

    #[test]
    fn flow_inverse_recovers_input() {
        // TEST 1 (Vits2Flow): inverse(forward(x)) ≈ x to <= 1e-4.
        let mut rng = LcgRng::new(7);
        let flow = Vits2Flow::new(8, 24, 4, &mut rng).expect("flow");
        let t = 11usize;
        let mut x = vec![0.0_f32; t * 8];
        let mut data_rng = LcgRng::new(70);
        data_rng.fill_normal(&mut x);
        let (z, logdet) = flow.forward(&x, t).expect("forward");
        assert!(logdet.is_finite());
        assert!(z.iter().all(|v| v.is_finite()));
        let back = flow.inverse(&z, t).expect("inverse");
        let err = max_abs_diff(&x, &back);
        assert!(err < 1e-4, "flow round-trip error {err}");
    }

    #[test]
    fn flow_logdet_matches_finite_difference() {
        // TEST 2 (Vits2Flow): analytic logdet ≈ log|det J| (finite diff) <= 1e-2.
        let mut rng = LcgRng::new(9);
        let channels = 4usize;
        let t = 1usize; // n = t * channels = 4 → 4×4 Jacobian.
        let flow = Vits2Flow::new(channels, 16, 3, &mut rng).expect("flow");
        let n = t * channels;
        let mut x = vec![0.0_f32; n];
        let mut data_rng = LcgRng::new(90);
        data_rng.fill_normal(&mut x);

        let (_z0, analytic) = flow.forward(&x, t).expect("forward");

        // Central-difference Jacobian: J[i][j] = d out_i / d in_j.
        let h = 1e-3_f32;
        let mut jac = vec![0.0_f32; n * n];
        for j in 0..n {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += h;
            xm[j] -= h;
            let (yp, _) = flow.forward(&xp, t).expect("fwd+");
            let (ym, _) = flow.forward(&xm, t).expect("fwd-");
            for i in 0..n {
                jac[i * n + j] = (yp[i] - ym[i]) / (2.0 * h);
            }
        }
        let numeric = log_abs_det(jac, n);
        let err = (analytic - numeric).abs();
        assert!(
            err < 1e-2,
            "logdet mismatch: analytic={analytic} numeric={numeric} err={err}"
        );
    }

    #[test]
    fn flow_logdet_matches_finite_difference_dim6() {
        // Same check at the upper end of the requested small range (dim 6).
        let mut rng = LcgRng::new(21);
        let channels = 6usize;
        let t = 1usize;
        let flow = Vits2Flow::new(channels, 16, 2, &mut rng).expect("flow");
        let n = t * channels;
        let mut x = vec![0.0_f32; n];
        let mut data_rng = LcgRng::new(210);
        data_rng.fill_normal(&mut x);
        let (_z0, analytic) = flow.forward(&x, t).expect("forward");
        let h = 1e-3_f32;
        let mut jac = vec![0.0_f32; n * n];
        for j in 0..n {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += h;
            xm[j] -= h;
            let (yp, _) = flow.forward(&xp, t).expect("fwd+");
            let (ym, _) = flow.forward(&xm, t).expect("fwd-");
            for i in 0..n {
                jac[i * n + j] = (yp[i] - ym[i]) / (2.0 * h);
            }
        }
        let numeric = log_abs_det(jac, n);
        assert!((analytic - numeric).abs() < 1e-2);
    }

    #[test]
    fn flow_rejects_bad_shape() {
        let mut rng = LcgRng::new(11);
        let flow = Vits2Flow::new(4, 8, 2, &mut rng).expect("flow");
        assert!(flow.forward(&[0.0; 6], 2).is_err()); // 6 != 2*4
        assert!(flow.forward(&[], 0).is_err());
    }

    #[test]
    fn flow_rejects_bad_config() {
        let mut rng = LcgRng::new(12);
        assert!(Vits2Flow::new(1, 8, 2, &mut rng).is_err()); // channels < 2
        assert!(Vits2Flow::new(4, 8, 0, &mut rng).is_err()); // n_layers 0
        assert!(Vits2Flow::new(4, 0, 2, &mut rng).is_err()); // hidden 0
    }

    #[test]
    fn flow_is_deterministic() {
        let mut rng_a = LcgRng::new(123);
        let flow_a = Vits2Flow::new(8, 16, 3, &mut rng_a).expect("a");
        let mut rng_b = LcgRng::new(123);
        let flow_b = Vits2Flow::new(8, 16, 3, &mut rng_b).expect("b");
        let t = 7usize;
        let mut x = vec![0.0_f32; t * 8];
        LcgRng::new(4).fill_normal(&mut x);
        let (za, lda) = flow_a.forward(&x, t).expect("a");
        let (zb, ldb) = flow_b.forward(&x, t).expect("b");
        assert_eq!(za, zb);
        assert_eq!(lda, ldb);
    }
}
