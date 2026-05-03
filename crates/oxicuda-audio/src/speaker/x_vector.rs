//! TDNN-based x-vector speaker embedding network.
//!
//! Implements the architecture from Snyder et al. (2018) "X-Vectors: Robust
//! DNN Embeddings for Speaker Recognition".  A stack of dilated 1-D temporal
//! convolution (TDNN) layers extracts frame-level features, which are then
//! aggregated by statistics pooling and projected to a compact embedding.
//!
//! Default configuration:
//! ```text
//! Layer 1: TDNN, kernel 5, dilation 1, in→512, ReLU
//! Layer 2: TDNN, kernel 5, dilation 2, 512→512, ReLU
//! Layer 3: TDNN, kernel 7, dilation 3, 512→512, ReLU
//! Layer 4: TDNN, kernel 1, dilation 1, 512→512, ReLU
//! Layer 5: TDNN, kernel 1, dilation 1, 512→1500, ReLU
//! StatsPool → [3000]
//! Affine    → [512]  (embedding layer)
//! ```

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;
use crate::speaker::stats_pool::stats_pool;

// ─── Private helpers ─────────────────────────────────────────────────────────

/// ReLU activation (computed per-scalar to avoid closures).
#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Xavier (Glorot) uniform initialisation: uniform in `[-limit, +limit]`
/// where `limit = sqrt(6 / (fan_in + fan_out))`.
fn xavier_init(rng: &mut LcgRng, fan_in: usize, fan_out: usize, buf: &mut [f32]) {
    let limit = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
    for v in buf.iter_mut() {
        // rng.next_f32() ∈ [0, 1) → map to [-limit, +limit)
        *v = (rng.next_f32() * 2.0 - 1.0) * limit;
    }
}

/// Dilated 1-D causal-symmetric convolution with boundary replication padding.
///
/// `x`       — `[t, in_channels]` row-major.
/// `layer`   — TDNN layer weights.
/// `t`       — sequence length.
///
/// Returns `[t, out_channels]`.  The output length equals the input length
/// because boundary frames are replicated to fill the receptive field.
fn tdnn_forward(x: &[f32], layer: &TdnnLayer, t: usize) -> Vec<f32> {
    let ic = layer.in_channels;
    let oc = layer.out_channels;
    let ks = layer.kernel_size;
    let d = layer.dilation;

    let half = (ks / 2) as isize;

    let mut out = vec![0.0_f32; t * oc];

    for t_out in 0..t {
        for o in 0..oc {
            let mut acc = layer.bias[o];
            for k in 0..ks {
                // Compute the source frame with dilation and symmetric offset.
                let offset = (k as isize - half) * d as isize;
                let t_src_raw = t_out as isize + offset;
                // Replicate-boundary clamping.
                let t_src = t_src_raw.clamp(0, (t as isize) - 1) as usize;

                let w_off = o * ic * ks + k; // weight[o, :, k] starts here
                let x_off = t_src * ic;
                for i in 0..ic {
                    acc += layer.weight[w_off + i * ks] * x[x_off + i];
                }
            }
            out[t_out * oc + o] = relu(acc);
        }
    }
    out
}

// ─── XVectorConfig ───────────────────────────────────────────────────────────

/// Construction configuration for [`XVectorTdnn`].
#[derive(Debug, Clone)]
pub struct XVectorConfig {
    /// Number of input features per frame (e.g. 23 for MFCCs).
    pub in_dim: usize,
    /// Output channel count for each TDNN layer.
    pub tdnn_channels: Vec<usize>,
    /// Dilation factor for each TDNN layer.
    pub tdnn_dilations: Vec<usize>,
    /// Kernel size for each TDNN layer.
    pub tdnn_kernel_sizes: Vec<usize>,
    /// Stats-pool output dimension (`= 2 * tdnn_channels.last()`).
    pub stats_pool_dim: usize,
    /// Speaker embedding dimension.
    pub embed_dim: usize,
}

impl XVectorConfig {
    /// Standard x-vector configuration (Snyder et al. 2018):
    /// 23-dim MFCC input, 5 TDNN layers, 512-dim embedding.
    #[must_use]
    pub fn default_config() -> Self {
        let tdnn_channels = vec![512, 512, 512, 512, 1500];
        let stats_pool_dim = 2 * tdnn_channels.last().copied().unwrap_or(0);
        Self {
            in_dim: 23,
            tdnn_channels,
            tdnn_dilations: vec![1, 2, 3, 1, 1],
            tdnn_kernel_sizes: vec![5, 5, 7, 1, 1],
            stats_pool_dim,
            embed_dim: 512,
        }
    }

    /// Tiny configuration for fast unit tests:
    /// 16-dim input, 3 TDNN layers, 64-dim embedding.
    #[must_use]
    pub fn tiny() -> Self {
        let tdnn_channels = vec![64, 64, 64];
        let stats_pool_dim = 2 * tdnn_channels.last().copied().unwrap_or(0);
        Self {
            in_dim: 16,
            tdnn_channels,
            tdnn_dilations: vec![1, 2, 1],
            tdnn_kernel_sizes: vec![3, 3, 1],
            stats_pool_dim,
            embed_dim: 64,
        }
    }

    /// Number of TDNN layers.
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.tdnn_channels.len()
    }

    /// Validate that all per-layer vectors have the same length.
    pub fn validate(&self) -> AudioResult<()> {
        let n = self.n_layers();
        if self.tdnn_dilations.len() != n || self.tdnn_kernel_sizes.len() != n {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "XVectorConfig: tdnn_channels len={n}, dilations len={}, kernel_sizes len={}",
                    self.tdnn_dilations.len(),
                    self.tdnn_kernel_sizes.len()
                ),
            });
        }
        if self.embed_dim == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if self.in_dim == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        Ok(())
    }
}

// ─── TdnnLayer ───────────────────────────────────────────────────────────────

/// A single dilated TDNN layer with weight and bias tensors.
pub struct TdnnLayer {
    /// Weight tensor `[out_channels, in_channels, kernel_size]`.
    pub weight: Vec<f32>,
    /// Bias vector `[out_channels]`.
    pub bias: Vec<f32>,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Convolution kernel width.
    pub kernel_size: usize,
    /// Dilation factor.
    pub dilation: usize,
}

// ─── XVectorTdnn ─────────────────────────────────────────────────────────────

/// TDNN-based x-vector speaker embedding network.
pub struct XVectorTdnn {
    /// Ordered list of TDNN layers.
    pub layers: Vec<TdnnLayer>,
    /// Embedding (segment 6) weight `[embed_dim, stats_pool_dim]`.
    pub embed_weight: Vec<f32>,
    /// Embedding layer bias `[embed_dim]`.
    pub embed_bias: Vec<f32>,
    /// Construction configuration (retained for inspection).
    pub config: XVectorConfig,
}

impl XVectorTdnn {
    /// Construct a new `XVectorTdnn` with Xavier-initialised weights.
    ///
    /// # Errors
    ///
    /// Returns an [`AudioError`] if `config.validate()` fails.
    pub fn new(config: XVectorConfig, rng: &mut LcgRng) -> AudioResult<Self> {
        config.validate()?;

        let n = config.n_layers();
        let mut layers = Vec::with_capacity(n);

        let mut prev_channels = config.in_dim;
        for layer_idx in 0..n {
            let oc = config.tdnn_channels[layer_idx];
            let ks = config.tdnn_kernel_sizes[layer_idx];
            let dilation = config.tdnn_dilations[layer_idx];

            let fan_in = prev_channels * ks;
            let fan_out = oc * ks;
            let weight_len = oc * prev_channels * ks;
            let mut weight = vec![0.0_f32; weight_len];
            xavier_init(rng, fan_in, fan_out, &mut weight);

            let bias = vec![0.0_f32; oc];

            layers.push(TdnnLayer {
                weight,
                bias,
                in_channels: prev_channels,
                out_channels: oc,
                kernel_size: ks,
                dilation,
            });

            prev_channels = oc;
        }

        // Embedding (affine) layer: [embed_dim, stats_pool_dim]
        let sp_dim = config.stats_pool_dim;
        let ed = config.embed_dim;
        let mut embed_weight = vec![0.0_f32; ed * sp_dim];
        xavier_init(rng, sp_dim, ed, &mut embed_weight);
        let embed_bias = vec![0.0_f32; ed];

        Ok(Self {
            layers,
            embed_weight,
            embed_bias,
            config,
        })
    }

    /// Forward pass: `features [T, in_dim]` → speaker embedding `[embed_dim]`.
    ///
    /// Steps:
    /// 1. Apply each TDNN layer sequentially (dilated conv + ReLU).
    /// 2. Stats pool the final frame-level features to `[stats_pool_dim]`.
    /// 3. Affine projection to `[embed_dim]`.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::DimensionMismatch`] when
    /// `features.len() != t * config.in_dim`.
    pub fn forward(&self, features: &[f32], t: usize) -> AudioResult<Vec<f32>> {
        let in_dim = self.config.in_dim;
        let expected = t * in_dim;
        if features.len() != expected {
            return Err(AudioError::DimensionMismatch {
                expected,
                got: features.len(),
            });
        }
        if t == 0 {
            return Err(AudioError::InvalidSequenceLength(0));
        }

        // ── 1. TDNN layers ───────────────────────────────────────────────────
        let mut current = features.to_vec();
        for layer in &self.layers {
            current = tdnn_forward(&current, layer, t);
        }

        // ── 2. Statistics pooling → [stats_pool_dim] ─────────────────────────
        let last_channels = self.config.tdnn_channels.last().copied().unwrap_or(0);
        let pooled = stats_pool(&current, t, last_channels)?;

        // ── 3. Affine embedding projection ───────────────────────────────────
        let ed = self.config.embed_dim;
        let sp_dim = self.config.stats_pool_dim;
        let mut out = vec![0.0_f32; ed];
        for (o, out_val) in out.iter_mut().enumerate() {
            let mut acc = self.embed_bias[o];
            let w_row = &self.embed_weight[o * sp_dim..(o + 1) * sp_dim];
            for (wv, pv) in w_row.iter().zip(pooled.iter()) {
                acc += wv * pv;
            }
            *out_val = acc;
        }

        Ok(out)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speaker::attentive_pool::AttentivePool;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    // ── XVectorConfig ─────────────────────────────────────────────────────

    #[test]
    fn xvector_default_config_layers() {
        let cfg = XVectorConfig::default_config();
        assert_eq!(cfg.n_layers(), 5);
    }

    #[test]
    fn xvector_default_embed_dim() {
        let cfg = XVectorConfig::default_config();
        assert_eq!(cfg.embed_dim, 512);
    }

    #[test]
    fn xvector_config_tiny_valid() {
        let cfg = XVectorConfig::tiny();
        assert!(cfg.validate().is_ok());
    }

    // ── XVectorTdnn construction ───────────────────────────────────────────

    #[test]
    fn xvector_tiny_new_ok() {
        let mut rng = make_rng();
        let net = XVectorTdnn::new(XVectorConfig::tiny(), &mut rng);
        assert!(net.is_ok());
    }

    #[test]
    fn xvector_tiny_embed_dim_correct() {
        let mut rng = make_rng();
        let net = XVectorTdnn::new(XVectorConfig::tiny(), &mut rng).expect("ok");
        assert_eq!(net.config.embed_dim, 64);
    }

    // ── Forward pass ──────────────────────────────────────────────────────

    #[test]
    fn xvector_tiny_forward_shape() {
        let mut rng = make_rng();
        let net = XVectorTdnn::new(XVectorConfig::tiny(), &mut rng).expect("ok");
        let t = 20_usize;
        let in_dim = net.config.in_dim;
        let features = vec![1.0_f32; t * in_dim];
        let out = net.forward(&features, t).expect("forward ok");
        assert_eq!(out.len(), net.config.embed_dim);
    }

    #[test]
    fn xvector_tiny_forward_finite() {
        let mut rng = make_rng();
        let net = XVectorTdnn::new(XVectorConfig::tiny(), &mut rng).expect("ok");
        let t = 15_usize;
        let in_dim = net.config.in_dim;
        let mut features = vec![0.0_f32; t * in_dim];
        rng.fill_normal(&mut features);
        let out = net.forward(&features, t).expect("forward ok");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite embedding");
    }

    #[test]
    fn tdnn_forward_output_len_preserved() {
        let ks = 3_usize;
        let ic = 4_usize;
        let oc = 8_usize;
        let t = 10_usize;
        let layer = TdnnLayer {
            weight: vec![0.1_f32; oc * ic * ks],
            bias: vec![0.0_f32; oc],
            in_channels: ic,
            out_channels: oc,
            kernel_size: ks,
            dilation: 1,
        };
        let x = vec![1.0_f32; t * ic];
        let out = tdnn_forward(&x, &layer, t);
        assert_eq!(out.len(), t * oc, "output T must equal input T");
    }

    #[test]
    fn tdnn_forward_finite() {
        let ks = 5_usize;
        let ic = 4_usize;
        let oc = 4_usize;
        let t = 8_usize;
        let mut rng = make_rng();
        let mut weight = vec![0.0_f32; oc * ic * ks];
        xavier_init(&mut rng, ic * ks, oc * ks, &mut weight);
        let layer = TdnnLayer {
            weight,
            bias: vec![0.0_f32; oc],
            in_channels: ic,
            out_channels: oc,
            kernel_size: ks,
            dilation: 2,
        };
        let mut x = vec![0.0_f32; t * ic];
        rng.fill_normal(&mut x);
        let out = tdnn_forward(&x, &layer, t);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "tdnn output has non-finite values"
        );
    }

    #[test]
    fn attentive_pool_in_xvector_pipeline() {
        // Replace stats-pool with AttentivePool to verify integration.
        let c = 64_usize;
        let t = 12_usize;
        let mut rng = make_rng();
        let pool = AttentivePool::new(c, &mut rng).expect("pool ok");
        let features = vec![1.0_f32; t * c];
        let out = pool.forward(&features, t).expect("pool forward ok");
        assert_eq!(out.len(), 2 * c);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn xvector_different_t_values() {
        let mut rng = make_rng();
        let net = XVectorTdnn::new(XVectorConfig::tiny(), &mut rng).expect("ok");
        let in_dim = net.config.in_dim;
        let embed_dim = net.config.embed_dim;

        for &t_val in &[10_usize, 50] {
            let features = vec![0.5_f32; t_val * in_dim];
            let out = net.forward(&features, t_val).expect("forward ok");
            assert_eq!(out.len(), embed_dim, "embed_dim mismatch for T={t_val}");
            assert!(
                out.iter().all(|v| v.is_finite()),
                "non-finite for T={t_val}"
            );
        }
    }

    #[test]
    fn xvector_default_forward_smoke() {
        // Full default-config forward pass on a small number of frames.
        let mut rng = make_rng();
        let net = XVectorTdnn::new(XVectorConfig::default_config(), &mut rng).expect("ok");
        let t = 30_usize;
        let in_dim = net.config.in_dim;
        let mut features = vec![0.0_f32; t * in_dim];
        rng.fill_normal(&mut features);
        let out = net.forward(&features, t).expect("forward ok");
        assert_eq!(out.len(), net.config.embed_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn xvector_tiny_dim_mismatch_error() {
        let mut rng = make_rng();
        let net = XVectorTdnn::new(XVectorConfig::tiny(), &mut rng).expect("ok");
        let features = vec![1.0_f32; 5]; // wrong size
        let err = net.forward(&features, 3).unwrap_err();
        assert!(matches!(err, AudioError::DimensionMismatch { .. }));
    }
}
