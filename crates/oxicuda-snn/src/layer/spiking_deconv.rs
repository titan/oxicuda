//! Spiking 2-D transposed convolution (deconvolution) layer.
//!
//! Transposed convolution is the spatial *up-sampling* dual of convolution: it
//! is the operation whose forward pass equals the gradient of a strided
//! convolution. Implemented directly it is a **scatter-add** — every input cell
//! distributes its (weighted) value across a `kh × kw` neighbourhood of the
//! output, with neighbourhoods spaced `stride` apart. Stacking these scatter
//! contributions enlarges the spatial grid, which is exactly what spike-based
//! generative / decoder SNNs need to turn a coarse latent spike map into a
//! higher-resolution one. After the transposed convolution accumulates the
//! synaptic drive, a per-output-pixel LIF non-linearity (reused from
//! [`crate::neuron::lif`]) emits the binary output spikes.
//!
//! For input spatial size `(ih, iw)`, kernel `(kh, kw)`, `stride` and symmetric
//! `pad`, the output spatial size is
//!
//! ```text
//! oh = (ih − 1) · stride + kh − 2 · pad
//! ow = (iw − 1) · stride + kw − 2 · pad
//! ```
//!
//! and an input cell `(iy, ix)` scatters to output cell
//! `(iy · stride + u − pad, ix · stride + v − pad)` for every kernel offset
//! `(u, v)`. The weight tensor uses the PyTorch transposed-conv layout
//! `[in_c, out_c, kh, kw]`.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;
use crate::neuron::lif::{LifConfig, LifState, lif_step};

/// Configuration for a [`SpikingDeconv2d`] layer.
#[derive(Debug, Clone)]
pub struct SpikingDeconv2dConfig {
    /// Number of input channels; must be `> 0`.
    pub in_c: usize,
    /// Number of output channels; must be `> 0`.
    pub out_c: usize,
    /// Kernel height; must be `> 0`.
    pub kh: usize,
    /// Kernel width; must be `> 0`.
    pub kw: usize,
    /// Up-sampling stride; must be `> 0`.
    pub stride: usize,
    /// Symmetric zero-padding removed from each output border.
    pub pad: usize,
    /// Input spatial height; must be `> 0`.
    pub ih: usize,
    /// Input spatial width; must be `> 0`.
    pub iw: usize,
    /// LIF dynamics applied to the accumulated drive.
    pub lif: LifConfig,
}

impl SpikingDeconv2dConfig {
    /// Convenience constructor leaving `pad = 0` and a default [`LifConfig`].
    #[must_use]
    pub fn new(in_c: usize, out_c: usize, kh: usize, kw: usize, stride: usize) -> Self {
        Self {
            in_c,
            out_c,
            kh,
            kw,
            stride,
            pad: 0,
            ih: 0,
            iw: 0,
            lif: LifConfig::default(),
        }
    }
}

/// Output spatial extent of a transposed convolution along one axis.
///
/// Returns `(in_size − 1) · stride + kernel − 2 · pad`, or
/// [`SnnError::BadDim`] when the result would be non-positive.
fn transposed_out_dim(
    in_size: usize,
    kernel: usize,
    stride: usize,
    pad: usize,
) -> SnnResult<usize> {
    let core = (in_size as i64 - 1) * stride as i64 + kernel as i64;
    let out = core - 2 * pad as i64;
    if out <= 0 {
        return Err(SnnError::BadDim { got: 0 });
    }
    Ok(out as usize)
}

/// Spiking transposed-convolution layer: scatter-add up-sampling followed by LIF.
#[derive(Debug, Clone)]
pub struct SpikingDeconv2d {
    /// Filter weights, row-major `[in_c, out_c, kh, kw]`.
    pub w: Vec<f32>,
    /// Bias per output channel, length `out_c`.
    pub b: Vec<f32>,
    /// Number of input channels.
    pub in_c: usize,
    /// Number of output channels.
    pub out_c: usize,
    /// Kernel height.
    pub kh: usize,
    /// Kernel width.
    pub kw: usize,
    /// Up-sampling stride.
    pub stride: usize,
    /// Symmetric output padding.
    pub pad: usize,
    /// Input spatial height.
    pub ih: usize,
    /// Input spatial width.
    pub iw: usize,
    /// Output spatial height `oh = (ih−1)·stride + kh − 2·pad`.
    pub oh: usize,
    /// Output spatial width `ow = (iw−1)·stride + kw − 2·pad`.
    pub ow: usize,
    /// LIF parameters used by [`SpikingDeconv2d::forward_step`].
    pub lif_cfg: LifConfig,
    /// Persistent LIF membrane state over the flattened `[out_c, oh, ow]` grid.
    pub state: LifState,
}

impl SpikingDeconv2d {
    /// Allocate a layer with Kaiming-normal weights (fan-in `in_c·kh·kw`).
    ///
    /// Returns [`SnnError::BadDim`] when any channel / kernel / spatial size is
    /// zero or the derived output size is non-positive, and
    /// [`SnnError::OutOfRange`] when `stride == 0`.
    pub fn new(cfg: &SpikingDeconv2dConfig, rng: &mut LcgRng) -> SnnResult<Self> {
        if cfg.in_c == 0 || cfg.out_c == 0 || cfg.kh == 0 || cfg.kw == 0 {
            return Err(SnnError::BadDim { got: 0 });
        }
        if cfg.ih == 0 || cfg.iw == 0 {
            return Err(SnnError::BadDim { got: 0 });
        }
        if cfg.stride == 0 {
            return Err(SnnError::OutOfRange {
                name: "stride".to_string(),
                val: 0.0,
            });
        }
        let oh = transposed_out_dim(cfg.ih, cfg.kh, cfg.stride, cfg.pad)?;
        let ow = transposed_out_dim(cfg.iw, cfg.kw, cfg.stride, cfg.pad)?;

        let fan_in = cfg.in_c * cfg.kh * cfg.kw;
        let scale = (2.0_f32 / fan_in as f32).sqrt();
        let mut w = vec![0.0_f32; cfg.in_c * cfg.out_c * cfg.kh * cfg.kw];
        rng.fill_normal(&mut w);
        for v in &mut w {
            *v *= scale;
        }
        let b = vec![0.0_f32; cfg.out_c];
        let state = LifState::new(cfg.out_c * oh * ow);

        Ok(Self {
            w,
            b,
            in_c: cfg.in_c,
            out_c: cfg.out_c,
            kh: cfg.kh,
            kw: cfg.kw,
            stride: cfg.stride,
            pad: cfg.pad,
            ih: cfg.ih,
            iw: cfg.iw,
            oh,
            ow,
            lif_cfg: cfg.lif,
            state,
        })
    }

    /// Reset the persistent LIF membrane to zero.
    pub fn reset_state(&mut self) {
        for v in &mut self.state.v {
            *v = 0.0;
        }
    }

    /// Number of output spike values `out_c · oh · ow`.
    #[must_use]
    pub fn out_len(&self) -> usize {
        self.out_c * self.oh * self.ow
    }

    /// One forward timestep. `x` is `[in_c, ih, iw]` flattened; `out` receives
    /// the `[out_c, oh, ow]` binary spike map.
    ///
    /// Returns [`SnnError::BadShape`] when either slice length is wrong.
    pub fn forward_step(&mut self, x: &[f32], out: &mut [f32]) -> SnnResult<()> {
        let in_len = self.in_c * self.ih * self.iw;
        if x.len() != in_len {
            return Err(SnnError::BadShape {
                expected: in_len,
                got: x.len(),
            });
        }
        let out_len = self.out_len();
        if out.len() != out_len {
            return Err(SnnError::BadShape {
                expected: out_len,
                got: out.len(),
            });
        }

        let plane = self.oh * self.ow;
        let mut current = vec![0.0_f32; out_len];

        // Scatter-add: every input cell distributes weighted charge across a
        // kh×kw output neighbourhood spaced `stride` apart.
        for ic in 0..self.in_c {
            for iy in 0..self.ih {
                for ix in 0..self.iw {
                    let x_val = x[(ic * self.ih + iy) * self.iw + ix];
                    let oy0 = iy as i64 * self.stride as i64 - self.pad as i64;
                    let ox0 = ix as i64 * self.stride as i64 - self.pad as i64;
                    for oc in 0..self.out_c {
                        let w_base = ((ic * self.out_c + oc) * self.kh) * self.kw;
                        let cur_plane = oc * plane;
                        for u in 0..self.kh {
                            let oy = oy0 + u as i64;
                            if oy < 0 || oy >= self.oh as i64 {
                                continue;
                            }
                            let row = cur_plane + oy as usize * self.ow;
                            let w_row = w_base + u * self.kw;
                            for v in 0..self.kw {
                                let ox = ox0 + v as i64;
                                if ox < 0 || ox >= self.ow as i64 {
                                    continue;
                                }
                                current[row + ox as usize] += self.w[w_row + v] * x_val;
                            }
                        }
                    }
                }
            }
        }

        // Per-output-channel bias.
        for (oc_chunk, &bias) in current.chunks_mut(plane).zip(self.b.iter()) {
            for c in oc_chunk.iter_mut() {
                *c += bias;
            }
        }

        lif_step(&mut self.state, &current, &self.lif_cfg, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_shape_follows_transposed_formula() {
        let mut rng = LcgRng::new(1);
        let mut cfg = SpikingDeconv2dConfig::new(2, 3, 3, 3, 2);
        cfg.ih = 4;
        cfg.iw = 4;
        let layer = SpikingDeconv2d::new(&cfg, &mut rng).expect("ctor");
        // oh = (4-1)*2 + 3 - 0 = 9 (upsamples ~2x from 4).
        assert_eq!(layer.oh, 9);
        assert_eq!(layer.ow, 9);
        assert_eq!(layer.w.len(), 2 * 3 * 3 * 3);
        assert_eq!(layer.out_len(), 3 * 9 * 9);
    }

    #[test]
    fn padding_shrinks_output() {
        let mut rng = LcgRng::new(2);
        let mut cfg = SpikingDeconv2dConfig::new(1, 1, 3, 3, 2);
        cfg.ih = 3;
        cfg.iw = 3;
        cfg.pad = 1;
        let layer = SpikingDeconv2d::new(&cfg, &mut rng).expect("ctor");
        // oh = (3-1)*2 + 3 - 2*1 = 5.
        assert_eq!(layer.oh, 5);
        assert_eq!(layer.ow, 5);
    }

    #[test]
    fn single_spike_scatters_to_kernel_region() {
        let mut rng = LcgRng::new(3);
        let mut cfg = SpikingDeconv2dConfig::new(1, 1, 3, 3, 2);
        cfg.ih = 3;
        cfg.iw = 3;
        cfg.lif.v_th = 0.5;
        let mut layer = SpikingDeconv2d::new(&cfg, &mut rng).expect("ctor");
        for w in &mut layer.w {
            *w = 1.0;
        }
        // Single input spike at (0,0); with all-ones weights it scatters a
        // kh×kw = 3×3 ones block at the top-left, every cell crossing v_th.
        let mut x = vec![0.0_f32; 3 * 3];
        x[0] = 1.0;
        let mut out = vec![0.0_f32; layer.out_len()];
        layer.forward_step(&x, &mut out).expect("forward");
        let spikes: usize = out.iter().map(|&s| s as usize).sum();
        assert_eq!(spikes, 9, "one input spike should light a 3x3 region");
        // Those spikes form the top-left 3x3 block of the oh×ow grid.
        for oy in 0..3 {
            for ox in 0..3 {
                assert_eq!(out[oy * layer.ow + ox], 1.0);
            }
        }
    }

    #[test]
    fn changing_a_weight_changes_output() {
        let mut rng = LcgRng::new(4);
        let mut cfg = SpikingDeconv2dConfig::new(1, 1, 3, 3, 2);
        cfg.ih = 2;
        cfg.iw = 2;
        cfg.lif.v_th = 0.5;
        // Zero-weight layer never spikes for any input.
        let mut zero = SpikingDeconv2d::new(&cfg, &mut rng).expect("ctor");
        for w in &mut zero.w {
            *w = 0.0;
        }
        let mut x = vec![0.0_f32; 2 * 2];
        x[0] = 1.0;
        let mut out_zero = vec![0.0_f32; zero.out_len()];
        zero.forward_step(&x, &mut out_zero).expect("forward zero");
        assert!(out_zero.iter().all(|&s| s == 0.0));

        // Flip one weight: the cell it scatters to now crosses threshold.
        let mut bumped = zero.clone();
        bumped.reset_state();
        bumped.w[0] = 5.0; // (ic=0,oc=0,u=0,v=0) → output (0,0)
        let mut out_bumped = vec![0.0_f32; bumped.out_len()];
        bumped
            .forward_step(&x, &mut out_bumped)
            .expect("forward bumped");
        assert_ne!(out_zero, out_bumped, "weight change must alter the output");
        assert_eq!(out_bumped[0], 1.0);
    }

    #[test]
    fn spikes_are_binary_and_finite() {
        let mut rng = LcgRng::new(5);
        let mut cfg = SpikingDeconv2dConfig::new(2, 2, 3, 3, 2);
        cfg.ih = 3;
        cfg.iw = 3;
        cfg.lif.v_th = 0.3;
        let mut layer = SpikingDeconv2d::new(&cfg, &mut rng).expect("ctor");
        let x: Vec<f32> = (0..2 * 3 * 3).map(|i| (i % 2) as f32).collect();
        let mut out = vec![0.0_f32; layer.out_len()];
        for _ in 0..6 {
            layer.forward_step(&x, &mut out).expect("forward");
        }
        for &s in &out {
            assert!(s == 0.0 || s == 1.0, "non-binary spike {s}");
        }
        for &v in &layer.state.v {
            assert!(v.is_finite(), "non-finite membrane {v}");
        }
    }

    #[test]
    fn determinism_under_fixed_seed() {
        let mut cfg = SpikingDeconv2dConfig::new(2, 2, 3, 3, 2);
        cfg.ih = 3;
        cfg.iw = 3;
        cfg.lif.v_th = 0.4;
        let mut ra = LcgRng::new(77);
        let mut rb = LcgRng::new(77);
        let mut a = SpikingDeconv2d::new(&cfg, &mut ra).expect("a");
        let mut b = SpikingDeconv2d::new(&cfg, &mut rb).expect("b");
        let x: Vec<f32> = (0..2 * 3 * 3)
            .map(|i| ((i * 7) % 3 == 0) as i32 as f32)
            .collect();
        let mut oa = vec![0.0_f32; a.out_len()];
        let mut ob = vec![0.0_f32; b.out_len()];
        for _ in 0..5 {
            a.forward_step(&x, &mut oa).expect("a step");
            b.forward_step(&x, &mut ob).expect("b step");
            assert_eq!(oa, ob);
        }
        assert_eq!(a.state.v, b.state.v);
    }

    #[test]
    fn rejects_bad_config_and_shapes() {
        let mut rng = LcgRng::new(6);
        let mut zero_ch = SpikingDeconv2dConfig::new(0, 1, 3, 3, 2);
        zero_ch.ih = 3;
        zero_ch.iw = 3;
        assert!(matches!(
            SpikingDeconv2d::new(&zero_ch, &mut rng),
            Err(SnnError::BadDim { .. })
        ));

        let mut zero_stride = SpikingDeconv2dConfig::new(1, 1, 3, 3, 0);
        zero_stride.ih = 3;
        zero_stride.iw = 3;
        assert!(matches!(
            SpikingDeconv2d::new(&zero_stride, &mut rng),
            Err(SnnError::OutOfRange { .. })
        ));

        // pad so large the output collapses.
        let mut huge_pad = SpikingDeconv2dConfig::new(1, 1, 3, 3, 1);
        huge_pad.ih = 2;
        huge_pad.iw = 2;
        huge_pad.pad = 5;
        assert!(matches!(
            SpikingDeconv2d::new(&huge_pad, &mut rng),
            Err(SnnError::BadDim { .. })
        ));

        let mut ok = SpikingDeconv2dConfig::new(1, 1, 3, 3, 2);
        ok.ih = 3;
        ok.iw = 3;
        let mut layer = SpikingDeconv2d::new(&ok, &mut rng).expect("ctor");
        let mut out = vec![0.0_f32; layer.out_len()];
        assert!(matches!(
            layer.forward_step(&[0.0; 4], &mut out),
            Err(SnnError::BadShape { .. })
        ));
        let x = vec![0.0_f32; 3 * 3];
        assert!(matches!(
            layer.forward_step(&x, &mut [0.0; 3]),
            Err(SnnError::BadShape { .. })
        ));
    }
}
