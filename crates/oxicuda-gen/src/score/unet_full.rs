//! Full U-Net forward assembly for denoising score networks.
//!
//! Wires the per-block primitives from [`crate::score::unet_block`]
//! ([`UNetResBlock`], [`SelfAttentionBlock`]) into a complete encoder /
//! bottleneck / decoder ("down / mid / up") U-Net with skip connections and a
//! timestep embedding threaded through every residual block.
//!
//! # Spatial layout
//!
//! Activations are stored channel-last and row-major as a flat buffer of shape
//! `[height × width × channels]`, i.e. the value of channel `c` at pixel
//! `(y, x)` lives at index `((y * width) + x) * channels + c`. The token count
//! `height × width` is the spatial resolution.
//!
//! # Resolution flow
//!
//! - **Down**: per level, run `num_res_blocks` residual blocks (channel
//!   `in → out`), record the result as a skip, then **2×2 average-pool**
//!   downsample (each spatial dim halves). The final level is *not* downsampled.
//! - **Mid**: one residual block followed by a multi-head self-attention block
//!   over the flattened spatial tokens at the bottleneck.
//! - **Up**: per level (reverse order), **2×2 nearest-neighbour** upsample, add
//!   the matching down-path skip, then run residual blocks (channel narrowing).
//! - **Out**: a final residual block projects back to the input channel count.
//!
//! Because the up path exactly inverts the spatial halving of the down path and
//! the output projection restores the input channel count, a forward pass over
//! a `[H × W × C]` input returns a `[H × W × C]` output (verified in tests),
//! provided `H` and `W` are divisible by `2^(levels − 1)`.

use crate::error::{GenError, GenResult};
use crate::score::unet_block::{SelfAttentionBlock, UNetResBlock};

// ─── Local helpers (kept private; deterministic, allocation-light) ────────────

/// 2×2 average-pool downsample of a channel-last `[h × w × c]` buffer.
///
/// Returns a `[(h/2) × (w/2) × c]` buffer. Requires `h` and `w` even.
fn avg_pool2x2(x: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    let oh = h / 2;
    let ow = w / 2;
    let mut out = vec![0.0_f32; oh * ow * c];
    for oy in 0..oh {
        for ox in 0..ow {
            for ch in 0..c {
                let mut acc = 0.0_f32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let iy = oy * 2 + dy;
                        let ix = ox * 2 + dx;
                        acc += x[(iy * w + ix) * c + ch];
                    }
                }
                out[(oy * ow + ox) * c + ch] = acc * 0.25;
            }
        }
    }
    out
}

/// 2×2 nearest-neighbour upsample of a channel-last `[h × w × c]` buffer.
///
/// Returns a `[(2h) × (2w) × c]` buffer.
fn upsample2x2(x: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
    let oh = h * 2;
    let ow = w * 2;
    let mut out = vec![0.0_f32; oh * ow * c];
    for oy in 0..oh {
        for ox in 0..ow {
            let iy = oy / 2;
            let ix = ox / 2;
            for ch in 0..c {
                out[(oy * ow + ox) * c + ch] = x[(iy * w + ix) * c + ch];
            }
        }
    }
    out
}

/// Broadcast a per-token timestep embedding row.
///
/// `UNetResBlock::forward` expects `time_emb` of shape `[batch × time_emb_dim]`
/// where `batch` is the number of tokens. We hold a single `[time_emb_dim]`
/// vector and replicate it across all tokens.
fn broadcast_time_emb(time_emb: &[f32], tokens: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; tokens * time_emb.len()];
    for t in 0..tokens {
        out[t * time_emb.len()..(t + 1) * time_emb.len()].copy_from_slice(time_emb);
    }
    out
}

// ─── Per-block weight bundle ──────────────────────────────────────────────────

/// Weights for a single [`UNetResBlock`] within the assembly.
#[derive(Debug, Clone)]
pub struct ResBlockWeights {
    /// First linear weight, `[out_channels × in_channels]`.
    pub w1: Vec<f32>,
    /// Second linear weight, `[out_channels × out_channels]`.
    pub w2: Vec<f32>,
    /// Time-embedding projection, `[2*out_channels × time_emb_dim]`.
    pub wt: Vec<f32>,
}

impl ResBlockWeights {
    /// Zero-initialise weights for an `in→out` residual block.
    pub fn zeros(in_channels: usize, out_channels: usize, time_emb_dim: usize) -> Self {
        Self {
            w1: vec![0.0_f32; out_channels * in_channels],
            w2: vec![0.0_f32; out_channels * out_channels],
            wt: vec![0.0_f32; 2 * out_channels * time_emb_dim],
        }
    }
}

/// Weights for the bottleneck [`SelfAttentionBlock`].
#[derive(Debug, Clone)]
pub struct AttnWeights {
    /// QKV projection, `[3*embed_dim × embed_dim]`.
    pub qkv: Vec<f32>,
    /// Output projection, `[embed_dim × embed_dim]`.
    pub out: Vec<f32>,
}

impl AttnWeights {
    /// Zero-initialise attention weights for the given embedding dim.
    pub fn zeros(embed_dim: usize) -> Self {
        Self {
            qkv: vec![0.0_f32; 3 * embed_dim * embed_dim],
            out: vec![0.0_f32; embed_dim * embed_dim],
        }
    }
}

// ─── UNetConfig ───────────────────────────────────────────────────────────────

/// Configuration for the full U-Net assembly.
#[derive(Debug, Clone)]
pub struct UNetConfig {
    /// Number of input (and output) channels.
    pub in_channels: usize,
    /// Base channel width (level-0 width).
    pub base_channels: usize,
    /// Per-level channel multipliers (length = number of resolution levels).
    pub channel_mult: Vec<usize>,
    /// Residual blocks per level (both down and up paths).
    pub num_res_blocks: usize,
    /// Timestep-embedding dimensionality.
    pub time_emb_dim: usize,
    /// Number of attention heads at the bottleneck.
    pub num_heads: usize,
}

impl UNetConfig {
    /// Create a new U-Net config.
    ///
    /// # Errors
    /// - `EmptyInput` if any scalar dimension is zero or `channel_mult` is empty.
    /// - `DimensionMismatch` if the bottleneck width is not divisible by `num_heads`.
    pub fn new(
        in_channels: usize,
        base_channels: usize,
        channel_mult: Vec<usize>,
        num_res_blocks: usize,
        time_emb_dim: usize,
        num_heads: usize,
    ) -> GenResult<Self> {
        if in_channels == 0 || base_channels == 0 || time_emb_dim == 0 || num_heads == 0 {
            return Err(GenError::EmptyInput("U-Net dimensions must be > 0"));
        }
        if num_res_blocks == 0 {
            return Err(GenError::EmptyInput("num_res_blocks must be > 0"));
        }
        if channel_mult.is_empty() {
            return Err(GenError::EmptyInput("channel_mult must not be empty"));
        }
        let bottleneck = base_channels * channel_mult[channel_mult.len() - 1];
        if bottleneck % num_heads != 0 {
            return Err(GenError::DimensionMismatch {
                expected: bottleneck - bottleneck % num_heads,
                got: bottleneck,
            });
        }
        Ok(Self {
            in_channels,
            base_channels,
            channel_mult,
            num_res_blocks,
            time_emb_dim,
            num_heads,
        })
    }

    /// Channel width at the given level.
    pub fn channels_at_level(&self, level: usize) -> usize {
        self.base_channels * self.channel_mult.get(level).copied().unwrap_or(1)
    }

    /// Number of resolution levels.
    pub fn num_levels(&self) -> usize {
        self.channel_mult.len()
    }

    /// Channel width at the bottleneck (deepest level).
    pub fn bottleneck_channels(&self) -> usize {
        self.channels_at_level(self.num_levels() - 1)
    }

    /// Spatial down-sampling factor (each side divided by this on the way down).
    ///
    /// Equals `2^(num_levels − 1)` because the deepest level is not pooled.
    pub fn spatial_factor(&self) -> usize {
        1usize << (self.num_levels().saturating_sub(1))
    }
}

// ─── UNetWeights ──────────────────────────────────────────────────────────────

/// Full weight container for [`UNet`].
#[derive(Debug, Clone)]
pub struct UNetWeights {
    /// Down-path residual block weights, level-major then block-major.
    pub down: Vec<Vec<ResBlockWeights>>,
    /// Mid-path single residual block.
    pub mid_res: ResBlockWeights,
    /// Mid-path self-attention.
    pub mid_attn: AttnWeights,
    /// Up-path residual block weights, level-major (reverse order) then block-major.
    pub up: Vec<Vec<ResBlockWeights>>,
    /// Final output residual block (bottleneck-free, back to `in_channels`).
    pub out_res: ResBlockWeights,
}

impl UNetWeights {
    /// Zero-initialise all weights for the given config.
    ///
    /// The block shapes mirror exactly the structure walked by
    /// [`UNet::forward`], so a zero-weight forward pass is a well-defined
    /// (identity-skip-dominated) transform.
    pub fn zeros(config: &UNetConfig) -> Self {
        let n_levels = config.num_levels();
        let te = config.time_emb_dim;
        // ── Down path ──
        let mut down = Vec::with_capacity(n_levels);
        for level in 0..n_levels {
            let in_ch = if level == 0 {
                config.in_channels
            } else {
                config.channels_at_level(level - 1)
            };
            let out_ch = config.channels_at_level(level);
            let mut blocks = Vec::with_capacity(config.num_res_blocks);
            for res in 0..config.num_res_blocks {
                let bin = if res == 0 { in_ch } else { out_ch };
                blocks.push(ResBlockWeights::zeros(bin, out_ch, te));
            }
            down.push(blocks);
        }
        // ── Mid ──
        let mid_ch = config.bottleneck_channels();
        let mid_res = ResBlockWeights::zeros(mid_ch, mid_ch, te);
        let mid_attn = AttnWeights::zeros(mid_ch);
        // ── Up path (reverse level order) ──
        let mut up = Vec::with_capacity(n_levels);
        for level in (0..n_levels).rev() {
            let out_ch = config.channels_at_level(level);
            // After upsample+skip-add the activation has `out_ch_above` channels,
            // where `out_ch_above` is the width of the level we are entering from
            // below (i.e. the next-deeper level's width) for the first block.
            let in_ch_first = if level == n_levels - 1 {
                config.bottleneck_channels()
            } else {
                config.channels_at_level(level + 1)
            };
            let mut blocks = Vec::with_capacity(config.num_res_blocks);
            for res in 0..config.num_res_blocks {
                let bin = if res == 0 { in_ch_first } else { out_ch };
                blocks.push(ResBlockWeights::zeros(bin, out_ch, te));
            }
            up.push(blocks);
        }
        // ── Output projection ──
        let last_ch = config.channels_at_level(0);
        let out_res = ResBlockWeights::zeros(last_ch, config.in_channels, te);
        Self {
            down,
            mid_res,
            mid_attn,
            up,
            out_res,
        }
    }
}

// ─── UNet ─────────────────────────────────────────────────────────────────────

/// A complete U-Net denoising network assembled from per-block primitives.
#[derive(Debug, Clone)]
pub struct UNet {
    config: UNetConfig,
}

impl UNet {
    /// Build a U-Net from the given config.
    pub fn new(config: UNetConfig) -> GenResult<Self> {
        Ok(Self { config })
    }

    /// Return the config.
    pub fn config(&self) -> &UNetConfig {
        &self.config
    }

    /// Run a full forward pass.
    ///
    /// # Arguments
    /// - `x`: Input activation, channel-last `[height × width × in_channels]`.
    /// - `height`, `width`: Spatial resolution. Each must be divisible by
    ///   [`UNetConfig::spatial_factor`].
    /// - `time_emb`: A single `[time_emb_dim]` timestep-embedding vector,
    ///   broadcast across all spatial tokens at every residual block.
    /// - `weights`: Full weight bundle from [`UNetWeights::zeros`] or trained.
    ///
    /// # Returns
    /// Output activation of shape `[height × width × in_channels]` — the input
    /// spatial resolution and channel count are preserved.
    ///
    /// # Errors
    /// - `EmptyInput` if `x` or `time_emb` is empty.
    /// - `DimensionMismatch` if shapes or divisibility constraints are violated.
    /// - Propagates `WeightShapeMismatch` from the underlying primitives.
    pub fn forward(
        &self,
        x: &[f32],
        height: usize,
        width: usize,
        time_emb: &[f32],
        weights: &UNetWeights,
    ) -> GenResult<Vec<f32>> {
        if x.is_empty() {
            return Err(GenError::EmptyInput("x is empty"));
        }
        if time_emb.is_empty() {
            return Err(GenError::EmptyInput("time_emb is empty"));
        }
        if time_emb.len() != self.config.time_emb_dim {
            return Err(GenError::DimensionMismatch {
                expected: self.config.time_emb_dim,
                got: time_emb.len(),
            });
        }
        let c_in = self.config.in_channels;
        let expected = height
            .checked_mul(width)
            .and_then(|hw| hw.checked_mul(c_in))
            .ok_or_else(|| GenError::Internal("h*w*c overflow".to_string()))?;
        if x.len() != expected {
            return Err(GenError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        let factor = self.config.spatial_factor();
        if height % factor != 0 || width % factor != 0 {
            return Err(GenError::DimensionMismatch {
                expected: factor,
                got: height.min(width),
            });
        }

        let n_levels = self.config.num_levels();
        let mut h = x.to_vec();
        let mut cur_h = height;
        let mut cur_w = width;
        let mut cur_c = c_in;
        // Skips recorded as (buffer, h, w, c) after each level's residual blocks,
        // *before* the downsample (so they line up with the up path).
        let mut skips: Vec<(Vec<f32>, usize, usize, usize)> = Vec::with_capacity(n_levels);

        // ── DOWN ──
        for level in 0..n_levels {
            let out_ch = self.config.channels_at_level(level);
            for res in 0..self.config.num_res_blocks {
                let in_ch = if res == 0 { cur_c } else { out_ch };
                h = self.run_res_block(
                    &h,
                    cur_h,
                    cur_w,
                    in_ch,
                    out_ch,
                    time_emb,
                    &weights.down[level][res],
                )?;
                cur_c = out_ch;
            }
            // Record skip at this resolution.
            skips.push((h.clone(), cur_h, cur_w, cur_c));
            // Downsample everywhere except the deepest level.
            if level + 1 < n_levels {
                h = avg_pool2x2(&h, cur_h, cur_w, cur_c);
                cur_h /= 2;
                cur_w /= 2;
            }
        }

        // ── MID ──
        let mid_ch = self.config.bottleneck_channels();
        h = self.run_res_block(&h, cur_h, cur_w, mid_ch, mid_ch, time_emb, &weights.mid_res)?;
        // Self-attention over the flattened spatial tokens.
        let attn = SelfAttentionBlock::new(mid_ch, self.config.num_heads)?;
        let seq_len = cur_h * cur_w;
        h = attn.forward(&h, &weights.mid_attn.qkv, &weights.mid_attn.out, seq_len)?;

        // ── UP ──
        // `weights.up` is in reverse-level order (deepest first), matching the
        // order we pop skips. Levels are visited n_levels-1 .. 0.
        for (up_idx, level) in (0..n_levels).rev().enumerate() {
            // Upsample to the resolution of the matching skip, except for the
            // deepest level which is already at skip resolution.
            let (skip_buf, skip_h, skip_w, skip_c) = skips
                .pop()
                .ok_or_else(|| GenError::Internal("skip underflow".to_string()))?;
            if level + 1 < n_levels {
                h = upsample2x2(&h, cur_h, cur_w, cur_c);
                cur_h *= 2;
                cur_w *= 2;
            }
            debug_assert_eq!(cur_h, skip_h);
            debug_assert_eq!(cur_w, skip_w);
            // Add the skip. The skip has `skip_c` channels and the current
            // activation has `cur_c`; we add over the overlapping channels.
            h = add_skip(&h, cur_c, &skip_buf, skip_c, cur_h * cur_w);
            // Residual blocks narrowing back to this level's width.
            let out_ch = self.config.channels_at_level(level);
            for res in 0..self.config.num_res_blocks {
                let in_ch = if res == 0 { cur_c } else { out_ch };
                h = self.run_res_block(
                    &h,
                    cur_h,
                    cur_w,
                    in_ch,
                    out_ch,
                    time_emb,
                    &weights.up[up_idx][res],
                )?;
                cur_c = out_ch;
            }
        }

        // ── OUT ──
        h = self.run_res_block(&h, cur_h, cur_w, cur_c, c_in, time_emb, &weights.out_res)?;
        cur_c = c_in;

        debug_assert_eq!(cur_h, height);
        debug_assert_eq!(cur_w, width);
        debug_assert_eq!(cur_c, c_in);
        Ok(h)
    }

    /// Run one residual block over a channel-last spatial buffer.
    ///
    /// The block treats each spatial token as a batch element; the timestep
    /// embedding is broadcast to every token.
    #[allow(clippy::too_many_arguments)]
    fn run_res_block(
        &self,
        x: &[f32],
        h: usize,
        w: usize,
        in_ch: usize,
        out_ch: usize,
        time_emb: &[f32],
        weights: &ResBlockWeights,
    ) -> GenResult<Vec<f32>> {
        let tokens = h * w;
        let block = UNetResBlock::new(in_ch, out_ch, self.config.time_emb_dim);
        let te = broadcast_time_emb(time_emb, tokens);
        block.forward(x, &te, &weights.w1, &weights.w2, &weights.wt)
    }
}

/// Add a skip activation onto the current activation over overlapping channels.
///
/// `cur` is `[tokens × cur_c]`, `skip` is `[tokens × skip_c]`. The result keeps
/// `cur_c` channels; channel `c < min(cur_c, skip_c)` gets `cur + skip`, the
/// remaining channels pass through unchanged.
fn add_skip(cur: &[f32], cur_c: usize, skip: &[f32], skip_c: usize, tokens: usize) -> Vec<f32> {
    let mut out = cur.to_vec();
    let min_c = cur_c.min(skip_c);
    for t in 0..tokens {
        for c in 0..min_c {
            out[t * cur_c + c] += skip[t * skip_c + c];
        }
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> UNetConfig {
        // 2 levels → spatial factor 2; bottleneck width = 8*2 = 16, heads=4.
        UNetConfig::new(3, 8, vec![1, 2], 1, 16, 4).expect("valid U-Net config")
    }

    fn deterministic_fill(buf: &mut [f32], seed: f32) {
        let mut t = seed;
        for v in buf.iter_mut() {
            t += 1.0;
            *v = (t * 0.017).sin() * 0.05;
        }
    }

    #[test]
    fn config_spatial_factor() {
        let cfg = make_config();
        assert_eq!(cfg.num_levels(), 2);
        assert_eq!(cfg.spatial_factor(), 2);
        assert_eq!(cfg.bottleneck_channels(), 16);
    }

    #[test]
    fn config_rejects_non_divisible_heads() {
        // bottleneck = 5*3 = 15, heads = 4 → 15 % 4 != 0.
        assert!(UNetConfig::new(3, 5, vec![1, 3], 1, 16, 4).is_err());
    }

    #[test]
    fn forward_preserves_resolution_and_channels_zero_weights() {
        let cfg = make_config();
        let weights = UNetWeights::zeros(&cfg);
        let unet = UNet::new(cfg.clone()).expect("unet");
        let (h, w) = (4, 4); // divisible by spatial factor 2
        let x = vec![0.3_f32; h * w * cfg.in_channels];
        let time_emb = vec![0.1_f32; cfg.time_emb_dim];
        let out = unet
            .forward(&x, h, w, &time_emb, &weights)
            .expect("forward");
        assert_eq!(out.len(), h * w * cfg.in_channels, "shape preserved");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_preserves_resolution_and_channels_random_weights() {
        let cfg = make_config();
        let mut weights = UNetWeights::zeros(&cfg);
        let mut s = 1.0_f32;
        for lvl in &mut weights.down {
            for blk in lvl {
                deterministic_fill(&mut blk.w1, s);
                s += 7.0;
                deterministic_fill(&mut blk.w2, s);
                s += 7.0;
                deterministic_fill(&mut blk.wt, s);
                s += 7.0;
            }
        }
        deterministic_fill(&mut weights.mid_res.w1, s);
        s += 7.0;
        deterministic_fill(&mut weights.mid_res.w2, s);
        s += 7.0;
        deterministic_fill(&mut weights.mid_res.wt, s);
        s += 7.0;
        deterministic_fill(&mut weights.mid_attn.qkv, s);
        s += 7.0;
        deterministic_fill(&mut weights.mid_attn.out, s);
        s += 7.0;
        for lvl in &mut weights.up {
            for blk in lvl {
                deterministic_fill(&mut blk.w1, s);
                s += 7.0;
                deterministic_fill(&mut blk.w2, s);
                s += 7.0;
                deterministic_fill(&mut blk.wt, s);
                s += 7.0;
            }
        }
        deterministic_fill(&mut weights.out_res.w1, s);
        s += 7.0;
        deterministic_fill(&mut weights.out_res.w2, s);
        s += 7.0;
        deterministic_fill(&mut weights.out_res.wt, s);

        let unet = UNet::new(cfg.clone()).expect("unet");
        let (h, w) = (8, 4);
        let x: Vec<f32> = (0..h * w * cfg.in_channels)
            .map(|i| (i as f32 * 0.013).cos())
            .collect();
        let time_emb: Vec<f32> = (0..cfg.time_emb_dim)
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let out = unet
            .forward(&x, h, w, &time_emb, &weights)
            .expect("forward");
        assert_eq!(out.len(), h * w * cfg.in_channels);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "U-Net output must be finite under non-trivial weights"
        );
    }

    #[test]
    fn forward_threads_timestep_embedding() {
        // Different timestep embeddings must produce different outputs once the
        // time-projection weights are non-zero — confirming the embedding is
        // actually threaded through the residual blocks.
        let cfg = make_config();
        let mut weights = UNetWeights::zeros(&cfg);
        // For the timestep to influence the output it must survive the block's
        // *second* linear (otherwise w2 == 0 zeroes the time-modulated path).
        // Give the down-level-0 block non-zero w1, w2 and wt so the broadcast
        // shift propagates all the way to the residual output.
        for v in weights.down[0][0].w1.iter_mut() {
            *v = 0.1;
        }
        for v in weights.down[0][0].w2.iter_mut() {
            *v = 0.1;
        }
        for v in weights.down[0][0].wt.iter_mut() {
            *v = 0.2;
        }
        let unet = UNet::new(cfg.clone()).expect("unet");
        let (h, w) = (2, 2);
        let x = vec![0.5_f32; h * w * cfg.in_channels];
        let t_a = vec![1.0_f32; cfg.time_emb_dim];
        let t_b = vec![-1.0_f32; cfg.time_emb_dim];
        let out_a = unet.forward(&x, h, w, &t_a, &weights).expect("forward a");
        let out_b = unet.forward(&x, h, w, &t_b, &weights).expect("forward b");
        let max_diff = out_a
            .iter()
            .zip(&out_b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "timestep embedding should influence the output: diff={max_diff}"
        );
    }

    #[test]
    fn forward_rejects_indivisible_resolution() {
        let cfg = make_config(); // factor 2
        let weights = UNetWeights::zeros(&cfg);
        let unet = UNet::new(cfg.clone()).expect("unet");
        let (h, w) = (3, 4); // 3 not divisible by 2
        let x = vec![0.1_f32; h * w * cfg.in_channels];
        let time_emb = vec![0.0_f32; cfg.time_emb_dim];
        assert!(matches!(
            unet.forward(&x, h, w, &time_emb, &weights),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn forward_rejects_bad_input_len() {
        let cfg = make_config();
        let weights = UNetWeights::zeros(&cfg);
        let unet = UNet::new(cfg.clone()).expect("unet");
        let time_emb = vec![0.0_f32; cfg.time_emb_dim];
        let x = vec![0.1_f32; 10]; // not h*w*c for any valid h,w
        assert!(matches!(
            unet.forward(&x, 4, 4, &time_emb, &weights),
            Err(GenError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn forward_empty_inputs_rejected() {
        let cfg = make_config();
        let weights = UNetWeights::zeros(&cfg);
        let unet = UNet::new(cfg.clone()).expect("unet");
        let time_emb = vec![0.0_f32; cfg.time_emb_dim];
        assert!(matches!(
            unet.forward(&[], 4, 4, &time_emb, &weights),
            Err(GenError::EmptyInput(_))
        ));
        let x = vec![0.1_f32; 4 * 4 * cfg.in_channels];
        assert!(matches!(
            unet.forward(&x, 4, 4, &[], &weights),
            Err(GenError::EmptyInput(_))
        ));
    }

    #[test]
    fn avg_pool_then_upsample_shapes() {
        let h = 4;
        let w = 4;
        let c = 2;
        let x: Vec<f32> = (0..h * w * c).map(|i| i as f32).collect();
        let down = avg_pool2x2(&x, h, w, c);
        assert_eq!(down.len(), (h / 2) * (w / 2) * c);
        let up = upsample2x2(&down, h / 2, w / 2, c);
        assert_eq!(up.len(), h * w * c);
    }

    #[test]
    fn single_level_unet_no_downsample() {
        // One level → spatial factor 1, so any resolution is valid and no pooling.
        let cfg = UNetConfig::new(2, 4, vec![1], 1, 8, 2).expect("single-level cfg");
        assert_eq!(cfg.spatial_factor(), 1);
        let weights = UNetWeights::zeros(&cfg);
        let unet = UNet::new(cfg.clone()).expect("unet");
        let (h, w) = (3, 5);
        let x = vec![0.2_f32; h * w * cfg.in_channels];
        let time_emb = vec![0.1_f32; cfg.time_emb_dim];
        let out = unet
            .forward(&x, h, w, &time_emb, &weights)
            .expect("forward");
        assert_eq!(out.len(), h * w * cfg.in_channels);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
