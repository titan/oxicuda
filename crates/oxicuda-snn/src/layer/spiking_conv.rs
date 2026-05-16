//! 2-D spiking convolution layer with sliding-window correlation followed by per-output-pixel LIF.

use crate::error::{SnnError, SnnResult};
use crate::handle::LcgRng;
use crate::neuron::lif::{LifConfig, LifState, lif_step};

/// Spiking 2D convolutional layer — naive direct sliding-window convolution.
#[derive(Debug, Clone)]
pub struct SpikingConv2d {
    /// Filter weights flattened as `[out_c, in_c, kh, kw]`.
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
    /// Output spatial height.
    pub oh: usize,
    /// Output spatial width.
    pub ow: usize,
    /// Persistent LIF membrane state across timesteps over flattened `[out_c, oh, ow]`.
    pub state: LifState,
}

impl SpikingConv2d {
    /// Construct a new layer.
    ///
    /// Uses Kaiming-normal initialisation with fan-in `in_c·kh·kw`. No padding and
    /// stride 1 — output `(oh, ow) = (ih − kh + 1, iw − kw + 1)`.
    #[must_use]
    pub fn new(
        in_c: usize,
        out_c: usize,
        kh: usize,
        kw: usize,
        ih: usize,
        iw: usize,
        rng: &mut LcgRng,
    ) -> Self {
        let oh = ih.saturating_sub(kh - 1);
        let ow = iw.saturating_sub(kw - 1);
        let oh = if oh == 0 { 1 } else { oh };
        let ow = if ow == 0 { 1 } else { ow };
        let fan_in = in_c.max(1) * kh.max(1) * kw.max(1);
        let scale = (2.0_f32 / fan_in as f32).sqrt();
        let mut w = vec![0.0_f32; out_c * in_c * kh * kw];
        rng.fill_normal(&mut w);
        for v in &mut w {
            *v *= scale;
        }
        let b = vec![0.0_f32; out_c];
        let state = LifState::new(out_c * oh * ow);
        Self {
            w,
            b,
            in_c,
            out_c,
            kh,
            kw,
            oh,
            ow,
            state,
        }
    }

    /// Single forward timestep. `x` shape: `[in_c, ih, iw]` flattened. `out` shape: `[out_c, oh, ow]`.
    pub fn forward_step(
        &mut self,
        x: &[f32],
        ih: usize,
        iw: usize,
        lif_cfg: &LifConfig,
        out: &mut [f32],
    ) -> SnnResult<()> {
        if x.len() != self.in_c * ih * iw {
            return Err(SnnError::BadShape {
                expected: self.in_c * ih * iw,
                got: x.len(),
            });
        }
        if out.len() != self.out_c * self.oh * self.ow {
            return Err(SnnError::BadShape {
                expected: self.out_c * self.oh * self.ow,
                got: out.len(),
            });
        }
        let mut current = vec![0.0_f32; self.out_c * self.oh * self.ow];
        for oc in 0..self.out_c {
            let bias = self.b[oc];
            for r in 0..self.oh {
                for c in 0..self.ow {
                    let mut acc = bias;
                    for ic in 0..self.in_c {
                        for u in 0..self.kh {
                            for v in 0..self.kw {
                                let in_idx = ic * ih * iw + (r + u) * iw + (c + v);
                                let w_idx = ((oc * self.in_c + ic) * self.kh + u) * self.kw + v;
                                acc += x[in_idx] * self.w[w_idx];
                            }
                        }
                    }
                    let out_idx = oc * self.oh * self.ow + r * self.ow + c;
                    current[out_idx] = acc;
                }
            }
        }
        lif_step(&mut self.state, &current, lif_cfg, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_correct() {
        let mut rng = LcgRng::new(1);
        let layer = SpikingConv2d::new(2, 3, 3, 3, 8, 8, &mut rng);
        assert_eq!(layer.oh, 6);
        assert_eq!(layer.ow, 6);
        assert_eq!(layer.w.len(), 3 * 2 * 3 * 3);
    }

    #[test]
    fn zero_input_zero_output() {
        let mut rng = LcgRng::new(2);
        let mut layer = SpikingConv2d::new(1, 2, 3, 3, 5, 5, &mut rng);
        for w in &mut layer.w {
            *w = 0.0;
        }
        let x = vec![0.0_f32; 5 * 5];
        let mut out = vec![0.0_f32; 2 * 3 * 3];
        layer
            .forward_step(&x, 5, 5, &LifConfig::default(), &mut out)
            .expect("forward ok");
        for &s in &out {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn bad_shape_errors() {
        let mut rng = LcgRng::new(3);
        let mut layer = SpikingConv2d::new(1, 1, 3, 3, 5, 5, &mut rng);
        let x = vec![0.0_f32; 5];
        let mut out = vec![0.0_f32; 3 * 3];
        assert!(
            layer
                .forward_step(&x, 5, 5, &LifConfig::default(), &mut out)
                .is_err()
        );
    }
}
