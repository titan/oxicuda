//! Hyena hierarchy — implicit long-convolution operator with multiplicative gating.
//!
//! Reference: Poli et al. (2023), "Hyena Hierarchy: Towards Larger Convolutional
//! Language Models" (<https://arxiv.org/abs/2302.10866>).
//!
//! The Hyena operator is a sub-quadratic drop-in for attention.  It interleaves
//! *implicit* long convolutions with element-wise (multiplicative) gating in a
//! recurrence of depth `order`.  Two properties make it efficient:
//!
//! 1. **Implicit filters** — the long-conv filter is *not* stored as `seq_len`
//!    explicit taps per channel.  Instead a small MLP maps positional features
//!    `[sin(t), cos(t), t/seq_len, 1]` at each position `t` to the `d_model`
//!    filter values.  This decouples the filter length from the parameter
//!    count, so arbitrarily long filters cost a fixed number of parameters.
//!
//! 2. **FFT convolution** — applying a length-`L` filter to a length-`L` signal
//!    is done in `O(L log L)` with [`crate::s4::s4_fft::s4_fft_conv`] instead of
//!    the `O(L²)` direct sum.
//!
//! ## The recurrence
//!
//! Given input `x ∈ ℝ^{L×D}` the operator first forms `order + 1` projections
//! (linear maps `D → D` applied per time-step):
//!
//! ```text
//! branch[k][t, :] = W_k · x[t, :]      for k = 0 .. order
//! ```
//!
//! It then runs the gated long-conv recurrence, initialised by the last branch:
//!
//! ```text
//! z = branch[order]
//! for k in 0 .. order:
//!     z[:, c] = branch[k][:, c] ⊙ causal_conv( filter_k[:, c], z[:, c] )   (per channel c)
//! output = z
//! ```
//!
//! where `filter_k` is the `k`-th implicit filter head and `⊙` is element-wise
//! multiplication along time.  The convolution is causal (channel-independent),
//! so the operator respects autoregressive ordering.

use crate::error::{MambaError, MambaResult};
use crate::handle::LcgRng;
use crate::s4::s4_fft::s4_fft_conv;

// ─── HyenaConfig ───────────────────────────────────────────────────────────────

/// Configuration for a [`HyenaOperator`].
#[derive(Debug, Clone)]
pub struct HyenaConfig {
    /// Sequence length `L` the operator is specialised for.
    pub seq_len: usize,
    /// Model / channel dimension `D`.
    pub d_model: usize,
    /// Recurrence depth (number of gated long-conv steps).
    pub order: usize,
    /// Hidden width of the implicit-filter MLP.
    pub filter_mlp_hidden: usize,
}

impl HyenaConfig {
    /// Number of positional features fed to the implicit-filter MLP:
    /// `[sin(t), cos(t), t/seq_len, 1]`.
    pub const POS_FEATURES: usize = 4;

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// * [`MambaError::InvalidSeqLen`]   — if `seq_len == 0`.
    /// * [`MambaError::InvalidModelDim`] — if `d_model == 0`.
    /// * [`MambaError::InvalidSsmOrder`] — if `order == 0`.
    /// * [`MambaError::InvalidChunkSize`] — if `filter_mlp_hidden == 0`.
    pub fn validate(&self) -> MambaResult<()> {
        if self.seq_len == 0 {
            return Err(MambaError::InvalidSeqLen(0));
        }
        if self.d_model == 0 {
            return Err(MambaError::InvalidModelDim(0));
        }
        if self.order == 0 {
            return Err(MambaError::InvalidSsmOrder(0));
        }
        if self.filter_mlp_hidden == 0 {
            return Err(MambaError::InvalidChunkSize(0));
        }
        Ok(())
    }
}

// ─── HyenaOperator ─────────────────────────────────────────────────────────────

/// Hyena operator: `(order + 1)` input projections + an implicit-filter MLP +
/// a multiplicative gating recurrence over `order` long convolutions.
///
/// Field layout (all row-major, flat `Vec<f32>`):
///
/// * `in_proj` — `(order + 1)` matrices, each `D × D`; matrix `k` occupies
///   `in_proj[k * D * D .. (k + 1) * D * D]`, row-major (`row · D + col`).
/// * `filter_w1` / `filter_b1` — first MLP layer, `hidden × POS_FEATURES`
///   and `hidden`.
/// * `filter_w2` / `filter_b2` — second MLP layer producing `order` distinct
///   filter heads of width `D`: output dimension is `order * D`, so
///   `filter_w2` is `(order * D) × hidden` and `filter_b2` is `order * D`.
#[derive(Debug, Clone)]
pub struct HyenaOperator {
    cfg: HyenaConfig,
    in_proj: Vec<f32>,
    filter_w1: Vec<f32>,
    filter_b1: Vec<f32>,
    filter_w2: Vec<f32>,
    filter_b2: Vec<f32>,
}

impl HyenaOperator {
    /// Construct a Hyena operator with `N(0, 1)`-scaled random parameters.
    ///
    /// # Errors
    ///
    /// Propagates [`HyenaConfig::validate`] errors.
    pub fn new(cfg: HyenaConfig, rng: &mut LcgRng) -> MambaResult<Self> {
        cfg.validate()?;

        let d = cfg.d_model;
        let hidden = cfg.filter_mlp_hidden;
        let order = cfg.order;
        let pos = HyenaConfig::POS_FEATURES;

        // (order + 1) input projection matrices D×D, scaled by 1/sqrt(D)
        // for unit-variance pre-activations.
        let in_scale = 1.0_f32 / (d as f32).sqrt();
        let mut in_proj = vec![0.0_f32; (order + 1) * d * d];
        for w in in_proj.iter_mut() {
            *w = rng.next_normal_pair().0 * in_scale;
        }

        // Implicit-filter MLP: pos → hidden → (order * D).
        let w1_scale = 1.0_f32 / (pos as f32).sqrt();
        let mut filter_w1 = vec![0.0_f32; hidden * pos];
        for w in filter_w1.iter_mut() {
            *w = rng.next_normal_pair().0 * w1_scale;
        }
        let mut filter_b1 = vec![0.0_f32; hidden];
        for b in filter_b1.iter_mut() {
            *b = rng.next_normal_pair().0 * 0.01;
        }

        let w2_scale = 1.0_f32 / (hidden as f32).sqrt();
        let mut filter_w2 = vec![0.0_f32; order * d * hidden];
        for w in filter_w2.iter_mut() {
            *w = rng.next_normal_pair().0 * w2_scale;
        }
        let mut filter_b2 = vec![0.0_f32; order * d];
        for b in filter_b2.iter_mut() {
            *b = rng.next_normal_pair().0 * 0.01;
        }

        Ok(Self {
            cfg,
            in_proj,
            filter_w1,
            filter_b1,
            filter_w2,
            filter_b2,
        })
    }

    /// Return a reference to the operator configuration.
    #[inline]
    pub fn config(&self) -> &HyenaConfig {
        &self.cfg
    }

    /// Positional feature vector at time `t`: `[sin(t), cos(t), t/seq_len, 1]`.
    #[inline]
    fn pos_features(&self, t: usize) -> [f32; HyenaConfig::POS_FEATURES] {
        let tf = t as f32;
        [tf.sin(), tf.cos(), tf / (self.cfg.seq_len as f32), 1.0_f32]
    }

    /// Evaluate the implicit-filter MLP and return all `order` heads.
    ///
    /// The returned vector has length `order * seq_len * d_model`; head `k`
    /// occupies the slice `[k * L * D .. (k + 1) * L * D]` in row-major
    /// `(t · D + c)` layout.  Each value is produced by the 2-layer MLP
    /// `pos → tanh → (order * D)` applied at every position `t`.
    ///
    /// # Errors
    ///
    /// [`MambaError::NonFinite`] if any filter value is not finite.
    fn implicit_filter_heads(&self) -> MambaResult<Vec<f32>> {
        let l = self.cfg.seq_len;
        let d = self.cfg.d_model;
        let hidden = self.cfg.filter_mlp_hidden;
        let order = self.cfg.order;
        let pos = HyenaConfig::POS_FEATURES;
        let out_dim = order * d;

        let mut heads = vec![0.0_f32; order * l * d];
        let mut h = vec![0.0_f32; hidden];

        for t in 0..l {
            let feats = self.pos_features(t);

            // Layer 1: hidden = tanh(W1 · feats + b1).
            for (j, hj) in h.iter_mut().enumerate() {
                let mut acc = self.filter_b1[j];
                let row = j * pos;
                for (p, &f) in feats.iter().enumerate() {
                    acc += self.filter_w1[row + p] * f;
                }
                *hj = acc.tanh();
            }

            // Layer 2: out = W2 · hidden + b2  (linear; out_dim = order * D).
            for o in 0..out_dim {
                let mut acc = self.filter_b2[o];
                let row = o * hidden;
                for (k, &hk) in h.iter().enumerate() {
                    acc += self.filter_w2[row + k] * hk;
                }
                if !acc.is_finite() {
                    return Err(MambaError::NonFinite("hyena implicit filter"));
                }
                // Map (head, channel) = divmod(o, D) to head-major layout.
                let head = o / d;
                let ch = o % d;
                heads[head * l * d + t * d + ch] = acc;
            }
        }

        Ok(heads)
    }

    /// Compute the implicit long-conv filter as a single `seq_len × d_model`
    /// tensor (row-major `(t · D + c)`).
    ///
    /// This is the sum of the per-order filter heads — a convenient summary of
    /// the implicit filter; the forward pass uses the individual heads.
    ///
    /// # Errors
    ///
    /// Propagates errors from the implicit-filter MLP evaluation.
    pub fn implicit_filter(&self) -> MambaResult<Vec<f32>> {
        let l = self.cfg.seq_len;
        let d = self.cfg.d_model;
        let order = self.cfg.order;
        let heads = self.implicit_filter_heads()?;

        let mut filter = vec![0.0_f32; l * d];
        for head in 0..order {
            let base = head * l * d;
            for (i, f) in filter.iter_mut().enumerate() {
                *f += heads[base + i];
            }
        }
        Ok(filter)
    }

    /// Project the input into `(order + 1)` branches.
    ///
    /// `x` is `L × D` row-major; each returned branch is `L × D` and computed
    /// as `branch[k][t, :] = W_k · x[t, :]`.
    fn project_branches(&self, x: &[f32]) -> Vec<Vec<f32>> {
        let l = self.cfg.seq_len;
        let d = self.cfg.d_model;
        let order = self.cfg.order;
        let mut branches = Vec::with_capacity(order + 1);

        for k in 0..=order {
            let w_base = k * d * d;
            let mut branch = vec![0.0_f32; l * d];
            for t in 0..l {
                let x_row = t * d;
                let out_row = t * d;
                for o in 0..d {
                    let w_row = w_base + o * d;
                    let mut acc = 0.0_f32;
                    for i in 0..d {
                        acc += self.in_proj[w_row + i] * x[x_row + i];
                    }
                    branch[out_row + o] = acc;
                }
            }
            branches.push(branch);
        }
        branches
    }

    /// Forward pass: `x [L × D]` → `y [L × D]`.
    ///
    /// Implements the gated long-conv recurrence documented at the module
    /// level: `z = branch[order]`; then for each `k` in `0..order`,
    /// `z[:, c] = branch[k][:, c] ⊙ causal_conv(filter_k[:, c], z[:, c])`
    /// per channel `c`, using the FFT convolution from
    /// [`crate::s4::s4_fft::s4_fft_conv`].
    ///
    /// # Errors
    ///
    /// * [`MambaError::DimensionMismatch`] — if `x.len() != seq_len * d_model`.
    /// * Propagates filter-MLP and FFT-conv errors.
    /// * [`MambaError::NonFinite`] — if the output contains a non-finite value.
    pub fn forward(&self, x: &[f32]) -> MambaResult<Vec<f32>> {
        let l = self.cfg.seq_len;
        let d = self.cfg.d_model;
        let order = self.cfg.order;
        let expected = l * d;
        if x.len() != expected {
            return Err(MambaError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        // Project input into (order + 1) branches and the implicit filters.
        let branches = self.project_branches(x);
        let filters = self.implicit_filter_heads()?;

        // Initialise the running signal with the last branch v = branch[order].
        let mut z = branches[order].clone();

        // Scratch buffers for one channel of the signal and one filter head.
        let mut z_ch = vec![0.0_f32; l];
        let mut filt_ch = vec![0.0_f32; l];

        // Gated long-conv recurrence over the first `order` gating branches.
        for (k, branch) in branches.iter().enumerate().take(order) {
            let filter_base = k * l * d;
            for c in 0..d {
                // Gather channel c of z and of filter head k.
                for t in 0..l {
                    z_ch[t] = z[t * d + c];
                    filt_ch[t] = filters[filter_base + t * d + c];
                }
                // Causal long convolution (O(L log L)).
                let conv = s4_fft_conv(&filt_ch, &z_ch)?;
                // Multiplicative gating by branch k, written back into z.
                for t in 0..l {
                    z[t * d + c] = branch[t * d + c] * conv[t];
                }
            }
        }

        if z.iter().any(|v| !v.is_finite()) {
            return Err(MambaError::NonFinite("hyena forward output"));
        }
        Ok(z)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(seq_len: usize, d_model: usize, order: usize, hidden: usize) -> HyenaConfig {
        HyenaConfig {
            seq_len,
            d_model,
            order,
            filter_mlp_hidden: hidden,
        }
    }

    fn make_op(seq_len: usize, d_model: usize, order: usize, hidden: usize) -> HyenaOperator {
        let mut rng = LcgRng::new(123);
        HyenaOperator::new(cfg(seq_len, d_model, order, hidden), &mut rng).expect("operator")
    }

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * 0.05 - 0.3).collect()
    }

    // ── construction / config ──────────────────────────────────────────────────

    /// A valid config constructs successfully.
    #[test]
    fn construct_ok() {
        let mut rng = LcgRng::new(1);
        let op = HyenaOperator::new(cfg(8, 4, 2, 16), &mut rng);
        assert!(op.is_ok());
    }

    /// config() round-trips the stored values.
    #[test]
    fn config_accessor() {
        let op = make_op(6, 3, 2, 8);
        assert_eq!(op.config().seq_len, 6);
        assert_eq!(op.config().d_model, 3);
        assert_eq!(op.config().order, 2);
        assert_eq!(op.config().filter_mlp_hidden, 8);
    }

    // ── implicit filter ─────────────────────────────────────────────────────────

    /// implicit_filter has length seq_len * d_model and is finite.
    #[test]
    fn implicit_filter_shape_finite() {
        let op = make_op(12, 5, 3, 16);
        let f = op.implicit_filter().expect("filter");
        assert_eq!(f.len(), 12 * 5);
        assert!(f.iter().all(|v| v.is_finite()), "filter must be finite");
    }

    /// implicit_filter_heads has length order * seq_len * d_model.
    #[test]
    fn implicit_filter_heads_shape() {
        let op = make_op(10, 4, 3, 8);
        let heads = op.implicit_filter_heads().expect("heads");
        assert_eq!(heads.len(), 3 * 10 * 4);
        assert!(heads.iter().all(|v| v.is_finite()));
    }

    /// Positional features are bounded: sin,cos in [-1,1], t/L in [0,1), bias=1.
    #[test]
    fn positional_features_bounded() {
        let op = make_op(16, 2, 1, 4);
        for t in 0..16 {
            let f = op.pos_features(t);
            assert!((-1.0..=1.0).contains(&f[0]), "sin out of range: {}", f[0]);
            assert!((-1.0..=1.0).contains(&f[1]), "cos out of range: {}", f[1]);
            assert!((0.0..1.0).contains(&f[2]), "t/L out of range: {}", f[2]);
            assert!((f[3] - 1.0).abs() < 1e-6, "bias must be 1");
        }
    }

    /// implicit_filter is deterministic for a fixed seed.
    #[test]
    fn implicit_filter_deterministic() {
        let mut rng_a = LcgRng::new(55);
        let mut rng_b = LcgRng::new(55);
        let a = HyenaOperator::new(cfg(8, 3, 2, 8), &mut rng_a).expect("a");
        let b = HyenaOperator::new(cfg(8, 3, 2, 8), &mut rng_b).expect("b");
        assert_eq!(
            a.implicit_filter().expect("fa"),
            b.implicit_filter().expect("fb")
        );
    }

    // ── forward shape / order coverage ──────────────────────────────────────────

    /// forward output length == seq_len * d_model.
    #[test]
    fn forward_output_length() {
        let op = make_op(8, 4, 2, 16);
        let x = ramp(8 * 4);
        let y = op.forward(&x).expect("forward");
        assert_eq!(y.len(), 8 * 4);
    }

    /// order = 1 works (minimum recurrence depth).
    #[test]
    fn forward_order_one() {
        let op = make_op(8, 4, 1, 8);
        let x = ramp(8 * 4);
        let y = op.forward(&x).expect("forward");
        assert_eq!(y.len(), 8 * 4);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    /// Single channel d_model = 1 works.
    #[test]
    fn forward_single_channel() {
        let op = make_op(10, 1, 2, 8);
        let x = ramp(10);
        let y = op.forward(&x).expect("forward");
        assert_eq!(y.len(), 10);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    /// Higher order (3) still produces a correctly shaped finite output.
    #[test]
    fn forward_order_three() {
        let op = make_op(12, 3, 3, 16);
        let x = ramp(12 * 3);
        let y = op.forward(&x).expect("forward");
        assert_eq!(y.len(), 12 * 3);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    // ── numerical behaviour ─────────────────────────────────────────────────────

    /// All outputs are finite for a non-trivial input.
    #[test]
    fn forward_finite() {
        let op = make_op(16, 6, 2, 32);
        let mut rng = LcgRng::new(404);
        let mut x = vec![0.0_f32; 16 * 6];
        rng.fill_normal(&mut x);
        let y = op.forward(&x).expect("forward");
        assert!(y.iter().all(|v| v.is_finite()), "output must be finite");
    }

    /// forward is deterministic for a fixed seed and input.
    #[test]
    fn forward_deterministic() {
        let op = make_op(8, 4, 2, 16);
        let x = ramp(8 * 4);
        let a = op.forward(&x).expect("a");
        let b = op.forward(&x).expect("b");
        assert_eq!(a, b, "forward must be deterministic");
    }

    /// A zero input yields a zero output (linear projections of 0 are 0,
    /// and the multiplicative gating of zeros stays zero).
    #[test]
    fn forward_zero_input_zero_output() {
        let op = make_op(8, 4, 2, 16);
        let x = vec![0.0_f32; 8 * 4];
        let y = op.forward(&x).expect("forward");
        assert!(
            y.iter().all(|&v| v.abs() < 1e-6),
            "zero input must give zero output, got {y:?}"
        );
    }

    /// Changing the input changes the output (input is wired through).
    #[test]
    fn forward_input_sensitivity() {
        let op = make_op(8, 4, 2, 16);
        let x1 = ramp(8 * 4);
        let mut x2 = x1.clone();
        x2[3] += 1.0;
        let y1 = op.forward(&x1).expect("y1");
        let y2 = op.forward(&x2).expect("y2");
        assert_ne!(y1, y2, "output must depend on the input");
    }

    /// Two distinct inputs produce distinct outputs.
    #[test]
    fn forward_distinct_inputs_distinct_outputs() {
        let op = make_op(8, 3, 2, 8);
        let x1 = ramp(8 * 3);
        let x2: Vec<f32> = ramp(8 * 3).iter().map(|v| v * -2.0 + 0.5).collect();
        let y1 = op.forward(&x1).expect("y1");
        let y2 = op.forward(&x2).expect("y2");
        assert_ne!(y1, y2, "distinct inputs must give distinct outputs");
    }

    /// Mutating a gating projection (branch 0) changes the output, proving the
    /// multiplicative gating is wired into the recurrence.
    #[test]
    fn forward_gating_projection_wired() {
        let mut op = make_op(8, 4, 2, 16);
        let x = ramp(8 * 4);
        let y_before = op.forward(&x).expect("before");
        // Perturb the first input-projection matrix (gating branch k=0).
        op.in_proj[0] += 5.0;
        let y_after = op.forward(&x).expect("after");
        assert_ne!(
            y_before, y_after,
            "perturbing a gating projection must change the output"
        );
    }

    /// Mutating the last-branch projection (the recurrence seed) changes output.
    #[test]
    fn forward_value_projection_wired() {
        let mut op = make_op(8, 4, 2, 16);
        let x = ramp(8 * 4);
        let y_before = op.forward(&x).expect("before");
        // Last branch is matrix index `order`.
        let order = op.config().order;
        let d = op.config().d_model;
        op.in_proj[order * d * d] += 5.0;
        let y_after = op.forward(&x).expect("after");
        assert_ne!(
            y_before, y_after,
            "perturbing the value branch must change output"
        );
    }

    /// Mutating a filter-MLP weight changes the output (filter is applied).
    #[test]
    fn forward_filter_mlp_wired() {
        let mut op = make_op(8, 4, 2, 16);
        let x = ramp(8 * 4);
        let y_before = op.forward(&x).expect("before");
        op.filter_w2[0] += 5.0;
        let y_after = op.forward(&x).expect("after");
        assert_ne!(
            y_before, y_after,
            "perturbing a filter weight must change the output"
        );
    }

    /// The FFT causal conv used inside forward preserves length L per channel.
    #[test]
    fn reuses_fft_conv_length_preserved() {
        // Directly exercise the dependency the forward pass relies on.
        let filt = vec![0.5_f32, -0.2, 0.1, 0.4];
        let sig = vec![1.0_f32, 2.0, 3.0, 4.0];
        let conv = s4_fft_conv(&filt, &sig).expect("causal conv");
        assert_eq!(conv.len(), sig.len(), "causal conv must preserve length L");
    }

    // ── error paths ─────────────────────────────────────────────────────────────

    /// seq_len = 0 fails validation.
    #[test]
    fn err_zero_seq_len() {
        let mut rng = LcgRng::new(1);
        let err = HyenaOperator::new(cfg(0, 4, 2, 8), &mut rng).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidSeqLen(0)));
    }

    /// d_model = 0 fails validation.
    #[test]
    fn err_zero_d_model() {
        let mut rng = LcgRng::new(1);
        let err = HyenaOperator::new(cfg(8, 0, 2, 8), &mut rng).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidModelDim(0)));
    }

    /// order = 0 fails validation.
    #[test]
    fn err_zero_order() {
        let mut rng = LcgRng::new(1);
        let err = HyenaOperator::new(cfg(8, 4, 0, 8), &mut rng).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidSsmOrder(0)));
    }

    /// filter_mlp_hidden = 0 fails validation.
    #[test]
    fn err_zero_hidden() {
        let mut rng = LcgRng::new(1);
        let err = HyenaOperator::new(cfg(8, 4, 2, 0), &mut rng).expect_err("should fail");
        assert!(matches!(err, MambaError::InvalidChunkSize(0)));
    }

    /// Wrong input length returns DimensionMismatch.
    #[test]
    fn err_wrong_input_length() {
        let op = make_op(8, 4, 2, 16);
        let x = vec![0.0_f32; 10]; // expected 8*4 = 32
        let err = op.forward(&x).expect_err("should fail");
        assert!(matches!(err, MambaError::DimensionMismatch { .. }));
    }
}
