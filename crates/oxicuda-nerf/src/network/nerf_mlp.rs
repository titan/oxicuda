//! Full NeRF MLP with 8 fully-connected layers.
//!
//! Architecture:
//! - Layers 0–3: FC(xyz_enc_dim → 256) → ReLU
//! - Layer 4: FC(256 + xyz_enc_dim → 256) → ReLU (skip connection)
//! - Layers 5–6: FC(256 → 256) → ReLU
//! - Density head: FC(256 → 1) → ReLU (σ, non-negative)
//! - Feature: FC(256 → 256) → ReLU (bottleneck)
//! - Color: FC(256 + dir_enc_dim → 128) → ReLU → FC(128 → 3) → Sigmoid (RGB)

use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the full NeRF MLP.
#[derive(Debug, Clone)]
pub struct NerfMlpConfig {
    /// Output dimension of positional encoding for xyz.
    pub xyz_enc_dim: usize,
    /// Output dimension of positional encoding for view direction.
    pub dir_enc_dim: usize,
    /// Hidden layer width (typically 256).
    pub hidden_dim: usize,
}

// ─── Xavier initialization helper ────────────────────────────────────────────

fn xavier_fill(buf: &mut [f32], fan_in: usize, rng: &mut LcgRng) {
    let scale = (2.0_f32 / fan_in as f32).sqrt();
    let mut i = 0;
    while i + 1 < buf.len() {
        let (a, b) = rng.next_normal_pair();
        buf[i] = a * scale;
        buf[i + 1] = b * scale;
        i += 2;
    }
    if i < buf.len() {
        let (a, _) = rng.next_normal_pair();
        buf[i] = a * scale;
    }
}

fn make_layer(in_dim: usize, out_dim: usize, rng: &mut LcgRng) -> (Vec<f32>, Vec<f32>) {
    let mut w = vec![0.0_f32; out_dim * in_dim];
    let bias = vec![0.0_f32; out_dim];
    xavier_fill(&mut w, in_dim, rng);
    (w, bias)
}

// ─── NerfMlp ─────────────────────────────────────────────────────────────────

/// Full NeRF MLP.
#[derive(Debug, Clone)]
pub struct NerfMlp {
    // Main backbone layers (weight, bias) — indices 0..=6
    layers: Vec<(Vec<f32>, Vec<f32>)>,
    // Density head: 256 → 1
    density_w: Vec<f32>,
    density_b: Vec<f32>,
    // Feature layer: 256 → 256
    feat_w: Vec<f32>,
    feat_b: Vec<f32>,
    // Color layers: [256 + dir_enc] → 128 → 3
    color_w1: Vec<f32>,
    color_b1: Vec<f32>,
    color_w2: Vec<f32>,
    color_b2: Vec<f32>,
    config: NerfMlpConfig,
}

impl NerfMlp {
    /// Create a new NeRF MLP with Xavier-initialized weights.
    ///
    /// # Errors
    ///
    /// Returns `InvalidFeatureDim` if any dimension is zero.
    pub fn new(cfg: NerfMlpConfig, rng: &mut LcgRng) -> NerfResult<Self> {
        if cfg.xyz_enc_dim == 0 || cfg.dir_enc_dim == 0 || cfg.hidden_dim == 0 {
            return Err(NerfError::InvalidFeatureDim { dim: 0 });
        }
        let h = cfg.hidden_dim;
        let x = cfg.xyz_enc_dim;
        let d = cfg.dir_enc_dim;

        // Layers 0–3: xyz_enc → hidden (4 layers)
        let mut backbone = Vec::with_capacity(7);
        backbone.push(make_layer(x, h, rng)); // layer 0
        for _ in 1..4 {
            backbone.push(make_layer(h, h, rng));
        }
        // Layer 4: skip — (hidden + xyz_enc) → hidden
        backbone.push(make_layer(h + x, h, rng));
        // Layers 5–6: hidden → hidden
        for _ in 5..7 {
            backbone.push(make_layer(h, h, rng));
        }

        let (density_w, density_b) = make_layer(h, 1, rng);
        let (feat_w, feat_b) = make_layer(h, h, rng);
        let color_in = h + d;
        let (color_w1, color_b1) = make_layer(color_in, 128, rng);
        let (color_w2, color_b2) = make_layer(128, 3, rng);

        Ok(Self {
            layers: backbone,
            density_w,
            density_b,
            feat_w,
            feat_b,
            color_w1,
            color_b1,
            color_w2,
            color_b2,
            config: cfg,
        })
    }

    /// Forward pass for a single point.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if input sizes are wrong.
    pub fn forward(&self, xyz_enc: &[f32], dir_enc: &[f32]) -> NerfResult<(f32, [f32; 3])> {
        if xyz_enc.len() != self.config.xyz_enc_dim {
            return Err(NerfError::DimensionMismatch {
                expected: self.config.xyz_enc_dim,
                got: xyz_enc.len(),
            });
        }
        if dir_enc.len() != self.config.dir_enc_dim {
            return Err(NerfError::DimensionMismatch {
                expected: self.config.dir_enc_dim,
                got: dir_enc.len(),
            });
        }

        let h = self.config.hidden_dim;

        // Backbone forward through layers 0–3
        let mut act = fc_relu(xyz_enc, &self.layers[0].0, &self.layers[0].1, h);
        for i in 1..4 {
            act = fc_relu(&act.clone(), &self.layers[i].0, &self.layers[i].1, h);
        }
        // Layer 4: skip connection
        let mut skip_input = Vec::with_capacity(h + self.config.xyz_enc_dim);
        skip_input.extend_from_slice(&act);
        skip_input.extend_from_slice(xyz_enc);
        act = fc_relu(&skip_input, &self.layers[4].0, &self.layers[4].1, h);
        // Layers 5–6
        for i in 5..7 {
            act = fc_relu(&act.clone(), &self.layers[i].0, &self.layers[i].1, h);
        }

        // Density head: FC(h → 1) → ReLU
        let density_raw = fc_linear(&act, &self.density_w, &self.density_b, 1);
        let sigma = density_raw[0].max(0.0);

        // Feature layer
        let feat = fc_relu(&act, &self.feat_w, &self.feat_b, h);

        // Color head
        let mut color_in = Vec::with_capacity(h + self.config.dir_enc_dim);
        color_in.extend_from_slice(&feat);
        color_in.extend_from_slice(dir_enc);
        let hidden128 = fc_relu(&color_in, &self.color_w1, &self.color_b1, 128);
        let rgb_raw = fc_linear(&hidden128, &self.color_w2, &self.color_b2, 3);

        let rgb = [
            sigmoid(rgb_raw[0]),
            sigmoid(rgb_raw[1]),
            sigmoid(rgb_raw[2]),
        ];

        Ok((sigma, rgb))
    }

    /// Batch forward for N points.
    ///
    /// - `xyz_enc`: `[N * xyz_enc_dim]`
    /// - `dir_enc`: `[N * dir_enc_dim]`
    ///
    /// Returns `(sigma: [N], rgb: [N*3])`.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if sizes don't match.
    pub fn forward_batch(
        &self,
        xyz_enc: &[f32],
        dir_enc: &[f32],
        n: usize,
    ) -> NerfResult<(Vec<f32>, Vec<f32>)> {
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if xyz_enc.len() != n * self.config.xyz_enc_dim {
            return Err(NerfError::DimensionMismatch {
                expected: n * self.config.xyz_enc_dim,
                got: xyz_enc.len(),
            });
        }
        if dir_enc.len() != n * self.config.dir_enc_dim {
            return Err(NerfError::DimensionMismatch {
                expected: n * self.config.dir_enc_dim,
                got: dir_enc.len(),
            });
        }
        let xd = self.config.xyz_enc_dim;
        let dd = self.config.dir_enc_dim;
        let mut sigma_out = Vec::with_capacity(n);
        let mut rgb_out = Vec::with_capacity(n * 3);
        for i in 0..n {
            let (s, c) = self.forward(
                &xyz_enc[i * xd..(i + 1) * xd],
                &dir_enc[i * dd..(i + 1) * dd],
            )?;
            sigma_out.push(s);
            rgb_out.extend_from_slice(&c);
        }
        Ok((sigma_out, rgb_out))
    }
}

// ─── Activation utilities ────────────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Fully-connected layer: output[i] = sum_j(w[i*in + j] * x[j]) + b[i], then ReLU.
fn fc_relu(x: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    let mut out = vec![0.0_f32; out_dim];
    for (o, (wo, &bi)) in out.iter_mut().zip(w.chunks(in_dim).zip(b.iter())) {
        *o = (wo
            .iter()
            .zip(x.iter())
            .map(|(&wi, &xi)| wi * xi)
            .sum::<f32>()
            + bi)
            .max(0.0);
    }
    out
}

/// Fully-connected layer: output[i] = sum_j(w[i*in + j] * x[j]) + b[i], no activation.
fn fc_linear(x: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    let mut out = vec![0.0_f32; out_dim];
    for (o, (wo, &bi)) in out.iter_mut().zip(w.chunks(in_dim).zip(b.iter())) {
        *o = wo
            .iter()
            .zip(x.iter())
            .map(|(&wi, &xi)| wi * xi)
            .sum::<f32>()
            + bi;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_mlp() -> NerfMlp {
        let cfg = NerfMlpConfig {
            xyz_enc_dim: 24, // 4 freq, 3D, no raw
            dir_enc_dim: 16, // 2 freq, 4D... or keep it small
            hidden_dim: 16,  // Small for test speed
        };
        let mut rng = LcgRng::new(123);
        NerfMlp::new(cfg, &mut rng).expect("new should succeed")
    }

    #[test]
    fn forward_output_shape() {
        let mlp = make_test_mlp();
        let xyz = vec![0.0_f32; 24];
        let dir = vec![0.0_f32; 16];
        let (sigma, rgb) = mlp.forward(&xyz, &dir).expect("forward should succeed");
        assert!(sigma >= 0.0);
        assert!(rgb.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn batch_forward() {
        let mlp = make_test_mlp();
        let xyz = vec![0.1_f32; 3 * 24];
        let dir = vec![0.2_f32; 3 * 16];
        let (sigma, rgb) = mlp
            .forward_batch(&xyz, &dir, 3)
            .expect("forward_batch should succeed");
        assert_eq!(sigma.len(), 3);
        assert_eq!(rgb.len(), 9);
    }
}
