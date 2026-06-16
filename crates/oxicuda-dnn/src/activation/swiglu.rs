//! SwiGLU gated feed-forward network — Shazeer 2020, CPU reference.
//!
//! "GLU Variants Improve Transformer" (Shazeer, 2020) replaces the standard
//! `FFN(x) = max(0, xW₁)W₂` with a *gated* linear unit. SwiGLU uses the Swish
//! (SiLU) gate:
//!
//! ```text
//! swish(z) = z · sigmoid(z)
//! gate     = swish(x · Wᵀ)        (∈ ℝ^{d_ffn})
//! value    = x · Vᵀ               (∈ ℝ^{d_ffn})
//! hidden   = gate ⊙ value         (element-wise product)
//! out      = hidden · W₂ᵀ         (∈ ℝ^{d_model})
//! ```
//!
//! The element-wise product lets the network *gate* the value projection by a
//! learned, input-dependent mask, which empirically improves quality at fixed
//! parameter budget. SwiGLU is the FFN used by PaLM, LLaMA, and Mistral. To
//! keep the parameter count comparable to a vanilla FFN, real models shrink
//! `d_ffn` (typically to `≈ 2/3 · 4 · d_model`); this reference imposes no such
//! constraint — `d_ffn` is taken verbatim from the configuration.
//!
//! Weight layouts (all flat row-major, output-major):
//! - `w`  : `[d_ffn × d_model]`   (gate projection)
//! - `v`  : `[d_ffn × d_model]`   (value projection)
//! - `w2` : `[d_model × d_ffn]`   (output projection)

use crate::error::{DnnError, DnnResult};
use crate::position::DnnRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for [`SwiGlu`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwiGluConfig {
    /// Model (input / output) dimension.
    pub d_model: usize,
    /// Hidden (gated) dimension.
    pub d_ffn: usize,
}

impl SwiGluConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// - [`DnnError::InvalidArgument`] if `d_model == 0` or `d_ffn == 0`.
    pub fn validate(&self) -> DnnResult<()> {
        if self.d_model == 0 {
            return Err(DnnError::InvalidArgument(
                "SwiGLU d_model must be > 0".into(),
            ));
        }
        if self.d_ffn == 0 {
            return Err(DnnError::InvalidArgument("SwiGLU d_ffn must be > 0".into()));
        }
        Ok(())
    }
}

/// Numerically-stable logistic sigmoid.
#[inline]
fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Swish / SiLU activation: `z · sigmoid(z)`.
#[inline]
fn swish(z: f32) -> f32 {
    z * sigmoid(z)
}

// ─── SwiGlu ──────────────────────────────────────────────────────────────────

/// SwiGLU gated feed-forward layer.
pub struct SwiGlu {
    /// Gate projection, flat `[d_ffn × d_model]` row-major.
    w: Vec<f32>,
    /// Value projection, flat `[d_ffn × d_model]` row-major.
    v: Vec<f32>,
    /// Output projection, flat `[d_model × d_ffn]` row-major.
    w2: Vec<f32>,
    /// Validated configuration.
    config: SwiGluConfig,
}

impl SwiGlu {
    /// Construct a new SwiGLU layer with random parameters.
    ///
    /// Each projection is initialised `N(0, 1/√fan_in)`, where `fan_in` is the
    /// input dimension of that projection (`d_model` for `w`/`v`, `d_ffn` for
    /// `w2`).
    ///
    /// # Errors
    /// Propagates [`SwiGluConfig::validate`] failures.
    pub fn new(config: SwiGluConfig, rng: &mut DnnRng) -> DnnResult<Self> {
        config.validate()?;
        let d_model = config.d_model;
        let d_ffn = config.d_ffn;

        let in_scale = 1.0 / (d_model as f32).sqrt();
        let out_scale = 1.0 / (d_ffn as f32).sqrt();

        let mut w = vec![0.0_f32; d_ffn * d_model];
        rng.fill_normal(&mut w);
        for x in &mut w {
            *x *= in_scale;
        }

        let mut v = vec![0.0_f32; d_ffn * d_model];
        rng.fill_normal(&mut v);
        for x in &mut v {
            *x *= in_scale;
        }

        let mut w2 = vec![0.0_f32; d_model * d_ffn];
        rng.fill_normal(&mut w2);
        for x in &mut w2 {
            *x *= out_scale;
        }

        Ok(Self { w, v, w2, config })
    }

    /// Model (input / output) dimension.
    #[must_use]
    #[inline]
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }

    /// Hidden (gated) dimension.
    #[must_use]
    #[inline]
    pub fn d_ffn(&self) -> usize {
        self.config.d_ffn
    }

    /// Forward pass over `n_tokens` rows of `x` (`[n_tokens × d_model]`).
    ///
    /// Returns `[n_tokens × d_model]`.
    ///
    /// # Errors
    /// - [`DnnError::InvalidArgument`] if `n_tokens == 0`.
    /// - [`DnnError::InvalidDimension`] if `x.len() != n_tokens · d_model`.
    /// - [`DnnError::InvalidArgument`] if a non-finite value is produced.
    pub fn forward(&self, x: &[f32], n_tokens: usize) -> DnnResult<Vec<f32>> {
        if n_tokens == 0 {
            return Err(DnnError::InvalidArgument(
                "SwiGLU forward: n_tokens must be > 0".into(),
            ));
        }
        let d_model = self.config.d_model;
        let d_ffn = self.config.d_ffn;
        let expected = n_tokens * d_model;
        if x.len() != expected {
            return Err(DnnError::InvalidDimension(format!(
                "SwiGLU forward: expected {expected} elements, got {}",
                x.len()
            )));
        }

        let mut out = vec![0.0_f32; n_tokens * d_model];
        // Per-token hidden buffer (gate ⊙ value) reused across tokens.
        let mut hidden = vec![0.0_f32; d_ffn];

        for t in 0..n_tokens {
            let x_row = &x[t * d_model..(t + 1) * d_model];

            // hidden[f] = swish(<w_f, x>) · <v_f, x>
            for (f, slot) in hidden.iter_mut().enumerate() {
                let w_row = &self.w[f * d_model..(f + 1) * d_model];
                let v_row = &self.v[f * d_model..(f + 1) * d_model];
                let mut gate_pre = 0.0_f32;
                let mut value = 0.0_f32;
                for k in 0..d_model {
                    let xk = x_row[k];
                    gate_pre += w_row[k] * xk;
                    value += v_row[k] * xk;
                }
                *slot = swish(gate_pre) * value;
            }

            // out_row = hidden · W₂ᵀ
            let out_row = &mut out[t * d_model..(t + 1) * d_model];
            for (o, slot) in out_row.iter_mut().enumerate() {
                let w2_row = &self.w2[o * d_ffn..(o + 1) * d_ffn];
                let mut acc = 0.0_f32;
                for f in 0..d_ffn {
                    acc += w2_row[f] * hidden[f];
                }
                *slot = acc;
            }
        }

        if out.iter().any(|v| !v.is_finite()) {
            return Err(DnnError::InvalidArgument(
                "SwiGLU produced non-finite output".into(),
            ));
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(d_model: usize, d_ffn: usize) -> SwiGluConfig {
        SwiGluConfig { d_model, d_ffn }
    }

    #[test]
    fn forward_shape() {
        let mut rng = DnnRng::new(1);
        let layer = SwiGlu::new(cfg(8, 16), &mut rng).expect("ok");
        let n_tokens = 4;
        let x = vec![0.1_f32; n_tokens * 8];
        let out = layer.forward(&x, n_tokens).expect("ok");
        assert_eq!(out.len(), n_tokens * 8);
    }

    #[test]
    fn forward_finite() {
        let mut rng = DnnRng::new(2);
        let layer = SwiGlu::new(cfg(8, 16), &mut rng).expect("ok");
        let n_tokens = 5;
        let mut x = vec![0.0_f32; n_tokens * 8];
        rng.fill_normal(&mut x);
        let out = layer.forward(&x, n_tokens).expect("ok");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn swish_property() {
        // swish(0) = 0; swish(z) → z for large z; swish(z) → 0 for very neg z.
        assert!(swish(0.0).abs() < 1e-9);
        assert!((swish(20.0) - 20.0).abs() < 1e-3);
        assert!(swish(-20.0).abs() < 1e-3);
        // sigmoid(0) = 0.5 exactly.
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn different_inputs_different_outputs() {
        let mut rng = DnnRng::new(3);
        let layer = SwiGlu::new(cfg(8, 16), &mut rng).expect("ok");
        let x_a = vec![0.2_f32; 8];
        let mut x_b = vec![0.2_f32; 8];
        x_b[0] = 1.0;
        let out_a = layer.forward(&x_a, 1).expect("ok");
        let out_b = layer.forward(&x_b, 1).expect("ok");
        let diff: f32 = out_a
            .iter()
            .zip(out_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6, "different inputs must give different outputs");
    }

    #[test]
    fn gate_modulates() {
        // Build a layer where w2 = identity-ish (d_model == d_ffn) and v = I,
        // w = 0. Then gate_pre = 0 ⇒ swish(0) = 0 ⇒ hidden = 0 ⇒ output = 0,
        // demonstrating the gate fully suppresses the value path.
        let mut rng = DnnRng::new(4);
        let mut layer = SwiGlu::new(cfg(4, 4), &mut rng).expect("ok");
        for x in &mut layer.w {
            *x = 0.0; // gate pre-activation = 0 everywhere
        }
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = layer.forward(&x, 1).expect("ok");
        for o in &out {
            assert!(o.abs() < 1e-6, "zero gate must zero the output, got {o}");
        }
    }

    #[test]
    fn d_model_0_error() {
        let mut rng = DnnRng::new(5);
        let r = SwiGlu::new(cfg(0, 16), &mut rng);
        assert!(matches!(r, Err(DnnError::InvalidArgument(_))));
    }

    #[test]
    fn d_ffn_0_error() {
        let mut rng = DnnRng::new(6);
        let r = SwiGlu::new(cfg(8, 0), &mut rng);
        assert!(matches!(r, Err(DnnError::InvalidArgument(_))));
    }

    #[test]
    fn n_tokens_1() {
        let mut rng = DnnRng::new(7);
        let layer = SwiGlu::new(cfg(8, 16), &mut rng).expect("ok");
        let x = vec![0.3_f32; 8];
        let out = layer.forward(&x, 1).expect("ok");
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn output_not_input() {
        // Output dimension matches input, but with random non-degenerate
        // weights the values should differ from the input.
        let mut rng = DnnRng::new(8);
        let layer = SwiGlu::new(cfg(8, 16), &mut rng).expect("ok");
        let x: Vec<f32> = (0..8).map(|i| 0.5 + i as f32 * 0.1).collect();
        let out = layer.forward(&x, 1).expect("ok");
        let same = x.iter().zip(out.iter()).all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(!same, "output should transform the input");
    }

    #[test]
    fn forward_n_tokens_0_error() {
        let mut rng = DnnRng::new(9);
        let layer = SwiGlu::new(cfg(8, 16), &mut rng).expect("ok");
        let x = vec![0.1_f32; 8];
        let r = layer.forward(&x, 0);
        assert!(matches!(r, Err(DnnError::InvalidArgument(_))));
    }

    #[test]
    fn forward_wrong_len_error() {
        let mut rng = DnnRng::new(10);
        let layer = SwiGlu::new(cfg(8, 16), &mut rng).expect("ok");
        let x = vec![0.1_f32; 10]; // not n_tokens · d_model
        let r = layer.forward(&x, 2);
        assert!(matches!(r, Err(DnnError::InvalidDimension(_))));
    }

    #[test]
    fn accessors() {
        let mut rng = DnnRng::new(11);
        let layer = SwiGlu::new(cfg(8, 16), &mut rng).expect("ok");
        assert_eq!(layer.d_model(), 8);
        assert_eq!(layer.d_ffn(), 16);
    }
}
