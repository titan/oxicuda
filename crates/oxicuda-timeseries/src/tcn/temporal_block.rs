//! Single TCN residual block with weight-normalised dilated causal convolutions.
//!
//! Architecture per block:
//!   x → dilated_causal_conv_wn → ReLU → dilated_causal_conv_wn → ReLU → + residual → ReLU
//!
//! The residual path applies a 1×1 conv when C_in ≠ C_out.
//! At inference dropout is identity (no stochasticity needed).

use crate::error::{TsError, TsResult};
use crate::handle::LcgRng;

#[inline]
fn relu(v: f32) -> f32 {
    v.max(0.0)
}

/// Apply weight-normalised dilated causal convolution to a `[T, C_in]` input.
///
/// Weight normalisation: `w_eff[o, i, k] = g[o] * w[o, i, k] / ||w[o, :, :]||_2`
///
/// Causal padding: left-pad `(kernel_size - 1) * dilation` zeros so that
/// output timestep `t` only sees inputs at `t, t-d, t-2d, …, t-(K-1)*d`.
/// Any tap index below 0 is treated as zero (implicit zero-padding).
///
/// Returns `[T, C_out]`.
fn dil_causal_conv_wn(
    x: &[f32],
    t: usize,
    c_in: usize,
    w_raw: &[f32],
    w_g: &[f32],
    bias: &[f32],
    c_out: usize,
    k: usize,
    d: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; t * c_out];

    for o in 0..c_out {
        // Compute L2 norm of the raw weight vector for output channel `o`:
        // shape slice is [c_in * k] elements at w_raw[o * c_in * k ..]
        let w_start = o * c_in * k;
        let w_slice = &w_raw[w_start..w_start + c_in * k];
        let norm_sq: f32 = w_slice.iter().map(|&v| v * v).sum();
        let norm = norm_sq.sqrt().max(1e-12);
        let scale = w_g[o] / norm;

        for ti in 0..t {
            let mut acc = bias[o];
            for ki in 0..k {
                // tap at `ti - ki * d`; negative index → zero (causal pad)
                let offset = ki * d;
                if ti < offset {
                    // implicit zero — contributes nothing
                    continue;
                }
                let src_t = ti - offset;
                for ci in 0..c_in {
                    // w_raw layout: [c_out, c_in, k] row-major
                    let w_idx = w_start + ci * k + ki;
                    acc += x[src_t * c_in + ci] * w_raw[w_idx] * scale;
                }
            }
            out[ti * c_out + o] = acc;
        }
    }
    out
}

/// One TCN residual block.
///
/// Two dilated causal conv layers (weight-normalised) with ReLU activations,
/// plus a residual shortcut (1×1 conv when `c_in != c_out`).
#[derive(Debug, Clone)]
pub struct TcnBlock {
    /// Raw weights for first conv `[C_out, C_in, K]`.
    pub weight1_raw: Vec<f32>,
    /// Weight-norm scale for first conv `[C_out]`.
    pub weight1_g: Vec<f32>,
    /// Bias for first conv `[C_out]`.
    pub bias1: Vec<f32>,
    /// Raw weights for second conv `[C_out, C_out, K]`.
    pub weight2_raw: Vec<f32>,
    /// Weight-norm scale for second conv `[C_out]`.
    pub weight2_g: Vec<f32>,
    /// Bias for second conv `[C_out]`.
    pub bias2: Vec<f32>,
    /// Optional 1×1 residual projection `[C_out, C_in]`, present when `c_in != c_out`.
    pub residual_weight: Option<Vec<f32>>,
    pub c_in: usize,
    pub c_out: usize,
    pub kernel_size: usize,
    pub dilation: usize,
}

impl TcnBlock {
    /// Construct a TCN residual block with Kaiming He weight initialisation.
    ///
    /// # Errors
    ///
    /// - [`TsError::InvalidKernelSize`] when `kernel_size == 0`.
    /// - [`TsError::InvalidDilation`] when `dilation == 0`.
    pub fn new(
        c_in: usize,
        c_out: usize,
        kernel_size: usize,
        dilation: usize,
        rng: &mut LcgRng,
    ) -> TsResult<Self> {
        if kernel_size == 0 {
            return Err(TsError::InvalidKernelSize(0));
        }
        if dilation == 0 {
            return Err(TsError::InvalidDilation(0));
        }

        // Kaiming He std = sqrt(2 / (c_in * kernel_size))
        let std1 = (2.0_f32 / (c_in * kernel_size) as f32).sqrt();
        let mut weight1_raw = vec![0.0_f32; c_out * c_in * kernel_size];
        rng.fill_normal(&mut weight1_raw);
        for v in &mut weight1_raw {
            *v *= std1;
        }

        // Second conv: c_out → c_out
        let std2 = (2.0_f32 / (c_out * kernel_size) as f32).sqrt();
        let mut weight2_raw = vec![0.0_f32; c_out * c_out * kernel_size];
        rng.fill_normal(&mut weight2_raw);
        for v in &mut weight2_raw {
            *v *= std2;
        }

        let weight1_g = vec![1.0_f32; c_out];
        let weight2_g = vec![1.0_f32; c_out];
        let bias1 = vec![0.0_f32; c_out];
        let bias2 = vec![0.0_f32; c_out];

        // 1×1 residual projection when channel counts differ
        let residual_weight = if c_in != c_out {
            let scale = (6.0_f32 / (c_in + c_out) as f32).sqrt();
            let mut w = vec![0.0_f32; c_out * c_in];
            rng.fill_normal(&mut w);
            for v in &mut w {
                *v *= scale;
            }
            Some(w)
        } else {
            None
        };

        Ok(Self {
            weight1_raw,
            weight1_g,
            bias1,
            weight2_raw,
            weight2_g,
            bias2,
            residual_weight,
            c_in,
            c_out,
            kernel_size,
            dilation,
        })
    }

    /// Run the block forward pass on a `[T, C_in]` input.
    ///
    /// Returns `[T, C_out]`.
    ///
    /// # Errors
    ///
    /// - [`TsError::DimensionMismatch`] when `x.len() != t * self.c_in`.
    pub fn forward(&self, x: &[f32], t: usize) -> TsResult<Vec<f32>> {
        let expected = t * self.c_in;
        if x.len() != expected {
            return Err(TsError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        // First dilated causal conv + ReLU (dropout = identity at inference)
        let h = dil_causal_conv_wn(
            x,
            t,
            self.c_in,
            &self.weight1_raw,
            &self.weight1_g,
            &self.bias1,
            self.c_out,
            self.kernel_size,
            self.dilation,
        );
        let h: Vec<f32> = h.iter().map(|&v| relu(v)).collect();

        // Second dilated causal conv + ReLU
        let h2 = dil_causal_conv_wn(
            &h,
            t,
            self.c_out,
            &self.weight2_raw,
            &self.weight2_g,
            &self.bias2,
            self.c_out,
            self.kernel_size,
            self.dilation,
        );
        let h2: Vec<f32> = h2.iter().map(|&v| relu(v)).collect();

        // Residual shortcut: project x to c_out if needed, then add
        let res: Vec<f32> = match &self.residual_weight {
            Some(rw) => {
                // 1×1 conv: [T, C_in] × [C_out, C_in]^T → [T, C_out]
                let mut proj = vec![0.0_f32; t * self.c_out];
                for ti in 0..t {
                    for o in 0..self.c_out {
                        let mut acc = 0.0_f32;
                        for ci in 0..self.c_in {
                            acc += x[ti * self.c_in + ci] * rw[o * self.c_in + ci];
                        }
                        proj[ti * self.c_out + o] = acc;
                    }
                }
                proj
            }
            None => x.to_vec(),
        };

        // Output = ReLU(conv_output + residual)
        let out: Vec<f32> = h2
            .iter()
            .zip(res.iter())
            .map(|(&h, &r)| relu(h + r))
            .collect();

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    #[test]
    fn tcn_block_output_shape_same_channels() {
        let mut rng = make_rng();
        let block = TcnBlock::new(8, 8, 3, 1, &mut rng).expect("ok");
        let t = 20;
        let x = vec![0.5_f32; t * 8];
        let out = block.forward(&x, t).expect("ok");
        assert_eq!(out.len(), t * 8);
    }

    #[test]
    fn tcn_block_output_shape_channel_expand() {
        let mut rng = make_rng();
        let block = TcnBlock::new(4, 16, 3, 2, &mut rng).expect("ok");
        let t = 24;
        let mut x = vec![0.0_f32; t * 4];
        rng.fill_normal(&mut x);
        let out = block.forward(&x, t).expect("ok");
        assert_eq!(out.len(), t * 16);
    }

    #[test]
    fn tcn_block_output_finite() {
        let mut rng = make_rng();
        let block = TcnBlock::new(8, 8, 3, 4, &mut rng).expect("ok");
        let t = 32;
        let mut x = vec![0.0_f32; t * 8];
        rng.fill_normal(&mut x);
        let out = block.forward(&x, t).expect("ok");
        assert!(out.iter().all(|v| v.is_finite()), "non-finite output");
    }

    #[test]
    fn tcn_block_output_nonneg_due_to_relu() {
        // Final ReLU means all outputs must be >= 0
        let mut rng = make_rng();
        let block = TcnBlock::new(4, 8, 3, 1, &mut rng).expect("ok");
        let t = 16;
        let mut x = vec![0.0_f32; t * 4];
        rng.fill_normal(&mut x);
        let out = block.forward(&x, t).expect("ok");
        assert!(out.iter().all(|&v| v >= 0.0), "output has negative values");
    }

    #[test]
    fn tcn_block_zero_kernel_error() {
        let mut rng = make_rng();
        assert!(matches!(
            TcnBlock::new(4, 8, 0, 1, &mut rng).unwrap_err(),
            TsError::InvalidKernelSize(0)
        ));
    }

    #[test]
    fn tcn_block_zero_dilation_error() {
        let mut rng = make_rng();
        assert!(matches!(
            TcnBlock::new(4, 8, 3, 0, &mut rng).unwrap_err(),
            TsError::InvalidDilation(0)
        ));
    }

    #[test]
    fn tcn_block_dim_mismatch_error() {
        let mut rng = make_rng();
        let block = TcnBlock::new(4, 8, 3, 1, &mut rng).expect("ok");
        let x = vec![0.0_f32; 5]; // wrong size
        assert!(matches!(
            block.forward(&x, 2).unwrap_err(),
            TsError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn tcn_block_residual_weight_present_when_channels_differ() {
        let mut rng = make_rng();
        let block = TcnBlock::new(4, 8, 3, 1, &mut rng).expect("ok");
        assert!(block.residual_weight.is_some());
        assert_eq!(
            block
                .residual_weight
                .as_ref()
                .expect("residual weight present")
                .len(),
            8 * 4
        );
    }

    #[test]
    fn tcn_block_no_residual_weight_when_channels_equal() {
        let mut rng = make_rng();
        let block = TcnBlock::new(8, 8, 3, 1, &mut rng).expect("ok");
        assert!(block.residual_weight.is_none());
    }
}
