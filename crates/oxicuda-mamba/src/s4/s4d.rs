//! S4D — Diagonal State Space Model (Gu et al. 2022).
//!
//! # Background
//!
//! S4D ("On the Parameterization and Initialization of Diagonal State Space
//! Models", Gu, Goel, Gupta & Ré 2022) replaces the Diagonal-Plus-Low-Rank
//! (DPLR) parameterization of the original S4 with a **purely diagonal**
//! complex state matrix `A = diag(λ)`, `λ_n ∈ ℂ`.  The authors show that the
//! low-rank correction in S4's NPLR form is empirically negligible, so a plain
//! diagonal SSM recovers almost all of S4's performance while being far simpler
//! to implement and discretize (every mode is independent).
//!
//! A single (complex) diagonal mode obeys the continuous-time recurrence
//!
//! ```text
//! ḣ_n(t) = λ_n h_n(t) + b_n u(t)
//! y(t)   = Σ_n  c_n h_n(t)        (real part taken at the end)
//! ```
//!
//! and — being a scalar linear ODE — has the closed-form ZOH discretization
//!
//! ```text
//! Ā_n = exp(Δ λ_n)
//! B̄_n = (Ā_n − 1) / λ_n · b_n              (→ Δ·b_n  as λ_n → 0)
//! ```
//!
//! The layer output kernel of length `L` is the SSM impulse response
//!
//! ```text
//! K[ℓ] = Re Σ_n c_n Ā_n^ℓ B̄_n ,   ℓ = 0 … L−1
//! ```
//!
//! which we convolve causally with the input.  Because each `λ_n` is complex
//! we carry the real / imaginary parts explicitly as `f32` pairs (no external
//! complex-number dependency).
//!
//! # Initialization
//!
//! Two diagonal initializations from the paper are supported:
//!
//! * [`S4DInit::Lin`] — **S4D-Lin**: `λ_n = −½ + i π n`.  The imaginary parts
//!   are evenly spaced, giving a Fourier-like basis; this is the recommended
//!   default and closely matches the HiPPO-LegS spectrum.
//! * [`S4DInit::Inv`] — **S4D-Inv**: `λ_n = −½ + i · (N/π) · (N/(2n+1) − 1)`,
//!   the "inverse" law that approximates the true HiPPO-LegS eigenvalue
//!   distribution (denser low frequencies).
//!
//! The real part is fixed to `−½` for every mode (stable, contractive poles),
//! matching the normalized HiPPO-LegS diagonal `Re λ_n = −½`.

use crate::error::{MambaError, MambaResult};
use crate::handle::LcgRng;

// ─── Complex helper ──────────────────────────────────────────────────────────

/// Minimal complex multiply `(a_re, a_im) · (b_re, b_im)`.
#[inline]
fn cmul(a_re: f32, a_im: f32, b_re: f32, b_im: f32) -> (f32, f32) {
    (a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re)
}

/// Complex exponential `exp(re + i·im) = e^re · (cos im, sin im)`.
#[inline]
fn cexp(re: f32, im: f32) -> (f32, f32) {
    let mag = re.exp();
    (mag * im.cos(), mag * im.sin())
}

/// Complex reciprocal `1 / (re + i·im)`.
#[inline]
fn crecip(re: f32, im: f32) -> (f32, f32) {
    let denom = re * re + im * im;
    (re / denom, -im / denom)
}

// ─── Initialization scheme ───────────────────────────────────────────────────

/// Diagonal initialization law for the complex poles `λ_n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum S4DInit {
    /// S4D-Lin: `λ_n = −½ + i π n` (evenly spaced imaginary parts).
    Lin,
    /// S4D-Inv: HiPPO-approximating inverse law (denser low frequencies).
    Inv,
}

impl S4DInit {
    /// Return the `N` complex poles `λ_n = (re, im)` for this law.
    ///
    /// `state_dim` is the number of diagonal modes `N`.
    fn poles(self, state_dim: usize) -> Vec<(f32, f32)> {
        let n = state_dim;
        let re = -0.5_f32;
        match self {
            S4DInit::Lin => (0..n)
                .map(|k| (re, std::f32::consts::PI * k as f32))
                .collect(),
            S4DInit::Inv => {
                let nf = n as f32;
                (0..n)
                    .map(|k| {
                        // λ_n = −½ + i · (N/π) · (N/(2n+1) − 1)
                        let im = (nf / std::f32::consts::PI) * (nf / (2.0 * k as f32 + 1.0) - 1.0);
                        (re, im)
                    })
                    .collect()
            }
        }
    }
}

// ─── S4DConfig ───────────────────────────────────────────────────────────────

/// Configuration for an [`S4D`] diagonal SSM layer.
#[derive(Debug, Clone)]
pub struct S4DConfig {
    /// Number of channels `D` (independent SISO SSMs).
    pub d_model: usize,
    /// Diagonal state order `N` (complex modes per channel).
    pub state_dim: usize,
    /// Sequence length `L` the layer is specialised for.
    pub seq_len: usize,
    /// ZOH discretization step `Δ > 0`.
    pub delta: f32,
    /// Pole initialization law.
    pub init: S4DInit,
}

impl S4DConfig {
    /// Create a new S4D configuration (`Δ = 0.01`, S4D-Lin init).
    ///
    /// # Errors
    ///
    /// * [`MambaError::InvalidModelDim`] — if `d_model == 0`.
    /// * [`MambaError::InvalidSsmOrder`] — if `state_dim == 0`.
    /// * [`MambaError::InvalidSeqLen`]   — if `seq_len == 0`.
    pub fn new(d_model: usize, state_dim: usize, seq_len: usize) -> MambaResult<Self> {
        if d_model == 0 {
            return Err(MambaError::InvalidModelDim(0));
        }
        if state_dim == 0 {
            return Err(MambaError::InvalidSsmOrder(0));
        }
        if seq_len == 0 {
            return Err(MambaError::InvalidSeqLen(0));
        }
        Ok(Self {
            d_model,
            state_dim,
            seq_len,
            delta: 0.01_f32,
            init: S4DInit::Lin,
        })
    }

    /// Override the discretization step `Δ`.
    ///
    /// # Errors
    ///
    /// [`MambaError::NonPositiveDelta`] if `delta ≤ 0`.
    pub fn with_delta(mut self, delta: f32) -> MambaResult<Self> {
        if delta <= 0.0 {
            return Err(MambaError::NonPositiveDelta(delta));
        }
        self.delta = delta;
        Ok(self)
    }

    /// Override the pole initialization law.
    #[must_use]
    pub fn with_init(mut self, init: S4DInit) -> Self {
        self.init = init;
        self
    }
}

// ─── S4D ─────────────────────────────────────────────────────────────────────

/// A diagonal (S4D) sequence-to-sequence layer.
///
/// Internally each channel owns `N` complex poles `λ` plus complex `B` and `C`
/// vectors.  `B` is initialised to all-ones (`b_n = 1`, the S4D convention,
/// since the `B`/`C` product is what matters), and `C` to `N(0, 1)` (real and
/// imaginary parts independent).  A per-channel real skip `D` defaults to zero.
#[derive(Debug, Clone)]
pub struct S4D {
    config: S4DConfig,
    /// Per-channel complex poles, `[D][N]` flattened as `(re, im)` pairs.
    lambda: Vec<(f32, f32)>,
    /// Per-channel complex `B`, `[D][N]`.
    b: Vec<(f32, f32)>,
    /// Per-channel complex `C`, `[D][N]`.
    c: Vec<(f32, f32)>,
    /// Per-channel real skip connection `D`, length `[D]`.
    d_skip: Vec<f32>,
}

impl S4D {
    /// Construct an S4D layer with paper-faithful initialization.
    ///
    /// # Errors
    ///
    /// Propagates configuration validation errors (`d_model`/`state_dim`/`seq_len`).
    pub fn new(config: S4DConfig, rng: &mut LcgRng) -> MambaResult<Self> {
        let d = config.d_model;
        let n = config.state_dim;
        let base = config.init.poles(n);

        let mut lambda = Vec::with_capacity(d * n);
        let mut b = Vec::with_capacity(d * n);
        let mut c = Vec::with_capacity(d * n);
        for _ in 0..d {
            for &(re, im) in &base {
                lambda.push((re, im));
                b.push((1.0_f32, 0.0_f32));
                let (cr, ci) = rng.next_normal_pair();
                c.push((cr, ci));
            }
        }
        let d_skip = vec![0.0_f32; d];

        Ok(Self {
            config,
            lambda,
            b,
            c,
            d_skip,
        })
    }

    /// Return a reference to the configuration.
    #[inline]
    pub fn config(&self) -> &S4DConfig {
        &self.config
    }

    /// Read-only view of the complex poles `λ` (length `D·N`).
    #[inline]
    pub fn lambda(&self) -> &[(f32, f32)] {
        &self.lambda
    }

    /// Mutable view of the per-channel real skip `D` (length `D`).
    #[inline]
    pub fn d_skip_mut(&mut self) -> &mut [f32] {
        &mut self.d_skip
    }

    /// Compute the discrete SSM impulse-response kernel for one channel.
    ///
    /// Returns the real length-`L` kernel `K[ℓ] = Re Σ_n c_n Ā_n^ℓ B̄_n`.
    ///
    /// `ch` selects the channel (`0 ≤ ch < D`); out-of-range returns
    /// [`MambaError::DimensionMismatch`].
    pub fn channel_kernel(&self, ch: usize) -> MambaResult<Vec<f32>> {
        let n = self.config.state_dim;
        let l = self.config.seq_len;
        let d = self.config.d_model;
        if ch >= d {
            return Err(MambaError::DimensionMismatch {
                expected: d,
                got: ch,
            });
        }
        let delta = self.config.delta;
        let off = ch * n;

        // Precompute per-mode Ā_n and the (C·B̄)_n product.
        let mut a_bar = Vec::with_capacity(n);
        let mut cb = Vec::with_capacity(n);
        for j in 0..n {
            let (lr, li) = self.lambda[off + j];
            // Ā = exp(Δ λ)
            let (abr, abi) = cexp(delta * lr, delta * li);
            // B̄ = (Ā − 1)/λ · b ; L'Hôpital limit Δ·b for |λ|→0.
            let (br, bi) = self.b[off + j];
            let lam_mag = lr * lr + li * li;
            let (b_bar_re, b_bar_im) = if lam_mag < 1e-12 {
                cmul(delta, 0.0, br, bi)
            } else {
                let (inv_re, inv_im) = crecip(lr, li);
                // (Ā − 1)
                let (num_re, num_im) = (abr - 1.0, abi);
                let (q_re, q_im) = cmul(num_re, num_im, inv_re, inv_im);
                cmul(q_re, q_im, br, bi)
            };
            let (cr, ci) = self.c[off + j];
            // (C · B̄)_n
            cb.push(cmul(cr, ci, b_bar_re, b_bar_im));
            a_bar.push((abr, abi));
        }

        // K[ℓ] = Re Σ_n cb_n · Ā_n^ℓ, accumulated by repeatedly multiplying
        // a running power Ā_n^ℓ.
        let mut kernel = vec![0.0_f32; l];
        let mut pow: Vec<(f32, f32)> = vec![(1.0, 0.0); n]; // Ā_n^0 = 1
        for k in kernel.iter_mut() {
            let mut acc_re = 0.0_f32;
            for j in 0..n {
                let (pr, pi) = pow[j];
                let (cbr, cbi) = cb[j];
                // Re(cb · pow)
                acc_re += cbr * pr - cbi * pi;
                // advance power: pow ← pow · Ā
                let (abr, abi) = a_bar[j];
                pow[j] = cmul(pr, pi, abr, abi);
            }
            *k = acc_re;
        }
        Ok(kernel)
    }

    /// Forward pass: `u [L × D]` → `y [L × D]` (row-major `u[t·D + ch]`).
    ///
    /// For each channel the SSM kernel is convolved causally with the channel
    /// input and the real skip `D·u` is added.
    ///
    /// # Errors
    ///
    /// [`MambaError::DimensionMismatch`] if `u.len() ≠ L·D`.
    pub fn forward(&self, u: &[f32]) -> MambaResult<Vec<f32>> {
        let l = self.config.seq_len;
        let d = self.config.d_model;
        let expected = l * d;
        if u.len() != expected {
            return Err(MambaError::DimensionMismatch {
                expected,
                got: u.len(),
            });
        }

        let mut y = vec![0.0_f32; expected];
        for ch in 0..d {
            let kernel = self.channel_kernel(ch)?;
            let d_skip = self.d_skip[ch];
            // Causal convolution y[t] = Σ_{τ=0}^{t} K[τ] u[t−τ] + D·u[t].
            for t in 0..l {
                let mut acc = 0.0_f32;
                for tau in 0..=t {
                    acc += kernel[tau] * u[(t - tau) * d + ch];
                }
                y[t * d + ch] = acc + d_skip * u[t * d + ch];
            }
        }
        Ok(y)
    }

    /// Recurrent (state-space) forward for one channel, returning `[L]`.
    ///
    /// Runs the complex state recurrence `h ← Ā ⊙ h + B̄ u` directly and reads
    /// `y = Re(C·h) + D·u`.  This must agree with the convolution path
    /// produced by [`S4D::channel_kernel`] (verified by tests).
    ///
    /// # Errors
    ///
    /// * [`MambaError::DimensionMismatch`] — if `ch ≥ D` or `u_ch.len() ≠ L`.
    pub fn channel_recurrent(&self, ch: usize, u_ch: &[f32]) -> MambaResult<Vec<f32>> {
        let n = self.config.state_dim;
        let l = self.config.seq_len;
        let d = self.config.d_model;
        if ch >= d {
            return Err(MambaError::DimensionMismatch {
                expected: d,
                got: ch,
            });
        }
        if u_ch.len() != l {
            return Err(MambaError::DimensionMismatch {
                expected: l,
                got: u_ch.len(),
            });
        }
        let delta = self.config.delta;
        let off = ch * n;

        // Discrete per-mode parameters.
        let mut a_bar = Vec::with_capacity(n);
        let mut b_bar = Vec::with_capacity(n);
        for j in 0..n {
            let (lr, li) = self.lambda[off + j];
            let (abr, abi) = cexp(delta * lr, delta * li);
            let (br, bi) = self.b[off + j];
            let lam_mag = lr * lr + li * li;
            let bb = if lam_mag < 1e-12 {
                cmul(delta, 0.0, br, bi)
            } else {
                let (inv_re, inv_im) = crecip(lr, li);
                let (num_re, num_im) = (abr - 1.0, abi);
                let (q_re, q_im) = cmul(num_re, num_im, inv_re, inv_im);
                cmul(q_re, q_im, br, bi)
            };
            a_bar.push((abr, abi));
            b_bar.push(bb);
        }

        let mut h = vec![(0.0_f32, 0.0_f32); n];
        let mut out = vec![0.0_f32; l];
        for (t, &ut) in u_ch.iter().enumerate() {
            let mut y_re = 0.0_f32;
            for j in 0..n {
                let (hr, hi) = h[j];
                let (abr, abi) = a_bar[j];
                let (bbr, bbi) = b_bar[j];
                // h ← Ā·h + B̄·u
                let (mr, mi) = cmul(hr, hi, abr, abi);
                let nh = (mr + bbr * ut, mi + bbi * ut);
                h[j] = nh;
                // y += Re(C·h)
                let (cr, ci) = self.c[off + j];
                y_re += cr * nh.0 - ci * nh.1;
            }
            out[t] = y_re + self.d_skip[ch] * ut;
        }
        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rng() -> LcgRng {
        LcgRng::new(2024)
    }

    #[test]
    fn config_rejects_zero_dims() {
        assert!(matches!(
            S4DConfig::new(0, 4, 8),
            Err(MambaError::InvalidModelDim(0))
        ));
        assert!(matches!(
            S4DConfig::new(4, 0, 8),
            Err(MambaError::InvalidSsmOrder(0))
        ));
        assert!(matches!(
            S4DConfig::new(4, 4, 0),
            Err(MambaError::InvalidSeqLen(0))
        ));
    }

    #[test]
    fn config_with_delta_rejects_nonpositive() {
        let cfg = S4DConfig::new(2, 4, 8).expect("cfg");
        assert!(matches!(
            cfg.with_delta(0.0),
            Err(MambaError::NonPositiveDelta(_))
        ));
    }

    #[test]
    fn lin_poles_have_fixed_real_part() {
        let poles = S4DInit::Lin.poles(6);
        assert_eq!(poles.len(), 6);
        for (re, _) in &poles {
            assert!((re - (-0.5)).abs() < 1e-6, "Re λ must be −½");
        }
        // imaginary parts are 0, π, 2π, …
        assert!((poles[0].1 - 0.0).abs() < 1e-6);
        assert!((poles[1].1 - std::f32::consts::PI).abs() < 1e-5);
        assert!((poles[2].1 - 2.0 * std::f32::consts::PI).abs() < 1e-5);
    }

    #[test]
    fn inv_poles_real_part_and_finite() {
        let poles = S4DInit::Inv.poles(8);
        assert_eq!(poles.len(), 8);
        for (re, im) in &poles {
            assert!((re - (-0.5)).abs() < 1e-6);
            assert!(im.is_finite());
        }
    }

    #[test]
    fn poles_are_stable() {
        // All poles must have strictly negative real part (contractive).
        for init in [S4DInit::Lin, S4DInit::Inv] {
            for (re, _) in init.poles(10) {
                assert!(re < 0.0, "pole real part {re} must be < 0 for {init:?}");
            }
        }
    }

    #[test]
    fn new_layer_shapes() {
        let cfg = S4DConfig::new(3, 5, 7).expect("cfg");
        let layer = S4D::new(cfg, &mut rng()).expect("layer");
        assert_eq!(layer.lambda().len(), 3 * 5);
    }

    #[test]
    fn discrete_poles_inside_unit_disk() {
        // Ā = exp(Δ λ): for stable λ (Re < 0) the discrete pole magnitude < 1.
        let cfg = S4DConfig::new(1, 6, 4).expect("cfg");
        let layer = S4D::new(cfg, &mut rng()).expect("layer");
        let delta = layer.config().delta;
        for &(lr, li) in layer.lambda() {
            let (ar, ai) = cexp(delta * lr, delta * li);
            let mag = (ar * ar + ai * ai).sqrt();
            assert!(mag < 1.0, "|Ā| = {mag} must be < 1");
        }
    }

    #[test]
    fn kernel_has_correct_length() {
        let cfg = S4DConfig::new(2, 4, 9).expect("cfg");
        let layer = S4D::new(cfg, &mut rng()).expect("layer");
        let k = layer.channel_kernel(0).expect("kernel");
        assert_eq!(k.len(), 9);
        assert!(k.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn kernel_out_of_range_channel_errors() {
        let cfg = S4DConfig::new(2, 4, 5).expect("cfg");
        let layer = S4D::new(cfg, &mut rng()).expect("layer");
        assert!(layer.channel_kernel(2).is_err());
        assert!(layer.channel_recurrent(2, &[0.0; 5]).is_err());
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = S4DConfig::new(3, 4, 6).expect("cfg");
        let layer = S4D::new(cfg, &mut rng()).expect("layer");
        let mut r = rng();
        let mut u = vec![0.0_f32; 6 * 3];
        r.fill_normal(&mut u);
        let y = layer.forward(&u).expect("forward");
        assert_eq!(y.len(), 6 * 3);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_rejects_bad_length() {
        let cfg = S4DConfig::new(2, 4, 6).expect("cfg");
        let layer = S4D::new(cfg, &mut rng()).expect("layer");
        assert!(matches!(
            layer.forward(&[0.0; 5]),
            Err(MambaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn convolution_matches_recurrence() {
        // The convolution path (kernel) and the state recurrence must agree.
        let cfg = S4DConfig::new(2, 5, 12).expect("cfg");
        let layer = S4D::new(cfg, &mut rng()).expect("layer");
        let mut r = rng();
        let l = 12;
        let d = 2;
        let mut u = vec![0.0_f32; l * d];
        r.fill_normal(&mut u);
        let y_conv = layer.forward(&u).expect("conv");
        for ch in 0..d {
            let u_ch: Vec<f32> = (0..l).map(|t| u[t * d + ch]).collect();
            let y_rec = layer.channel_recurrent(ch, &u_ch).expect("rec");
            for t in 0..l {
                let a = y_conv[t * d + ch];
                let b = y_rec[t];
                assert!((a - b).abs() < 1e-3, "ch {ch} t {t}: conv {a} vs rec {b}");
            }
        }
    }

    #[test]
    fn d_skip_adds_passthrough() {
        // With a unit impulse and a known D, the t=0 output includes D·u[0].
        let cfg = S4DConfig::new(1, 3, 4).expect("cfg");
        let mut layer = S4D::new(cfg, &mut rng()).expect("layer");
        layer.d_skip_mut()[0] = 2.0;
        let mut u = vec![0.0_f32; 4];
        u[0] = 1.0;
        let y = layer.forward(&u).expect("fwd");
        let k0 = layer.channel_kernel(0).expect("k")[0];
        // y[0] = K[0]·u[0] + D·u[0] = K[0] + 2.
        assert!((y[0] - (k0 + 2.0)).abs() < 1e-4, "y0={}", y[0]);
    }

    #[test]
    fn zero_input_zero_output() {
        let cfg = S4DConfig::new(2, 4, 5).expect("cfg");
        let layer = S4D::new(cfg, &mut rng()).expect("layer");
        let y = layer.forward(&[0.0_f32; 10]).expect("fwd");
        assert!(y.iter().all(|&v| v.abs() < 1e-9));
    }

    #[test]
    fn cexp_matches_definition() {
        let (re, im) = cexp(0.0, std::f32::consts::FRAC_PI_2);
        assert!(re.abs() < 1e-6, "cos(π/2) ≈ 0, got {re}");
        assert!((im - 1.0).abs() < 1e-6, "sin(π/2) ≈ 1, got {im}");
    }

    #[test]
    fn inv_init_layer_forward_finite() {
        let cfg = S4DConfig::new(2, 6, 8)
            .expect("cfg")
            .with_init(S4DInit::Inv);
        let layer = S4D::new(cfg, &mut rng()).expect("layer");
        let mut r = rng();
        let mut u = vec![0.0_f32; 8 * 2];
        r.fill_normal(&mut u);
        let y = layer.forward(&u).expect("fwd");
        assert!(y.iter().all(|v| v.is_finite()));
    }
}
