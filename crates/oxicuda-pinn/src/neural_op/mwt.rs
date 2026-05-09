//! Multiwavelet Transform Operator (1D Haar wavelet).

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;

/// Configuration for MWT operator.
pub struct MwtConfig {
    /// Number of decomposition levels.
    pub levels: usize,
    /// Feature width per level.
    pub width: usize,
    /// Input dimensionality.
    pub d_in: usize,
    /// Output dimensionality.
    pub d_out: usize,
}

/// Multiwavelet Transform Operator.
pub struct Mwt {
    config: MwtConfig,
    /// Per-level kernel weights [width × width] applied in wavelet domain.
    level_w: Vec<Vec<f32>>,
    level_b: Vec<Vec<f32>>,
    /// Lift and project layers.
    lift_w: Vec<f32>,
    lift_b: Vec<f32>,
    project_w: Vec<f32>,
    project_b: Vec<f32>,
}

impl Mwt {
    /// Construct a new MWT operator.
    pub fn new(config: MwtConfig, rng: &mut LcgRng) -> Self {
        let w = config.width;
        let d_in = config.d_in;
        let d_out = config.d_out;

        let scale_lift = (2.0 / d_in as f32).sqrt();
        let lift_w: Vec<f32> = (0..w * d_in)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale_lift)
            .collect();
        let lift_b = vec![0.0_f32; w];

        let scale_proj = (2.0 / w as f32).sqrt();
        let project_w: Vec<f32> = (0..d_out * w)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale_proj)
            .collect();
        let project_b = vec![0.0_f32; d_out];

        let mut level_w = Vec::with_capacity(config.levels);
        let mut level_b = Vec::with_capacity(config.levels);
        let scale_k = (2.0 / w as f32).sqrt();
        for _ in 0..config.levels {
            let kw: Vec<f32> = (0..w * w)
                .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale_k)
                .collect();
            level_w.push(kw);
            level_b.push(vec![0.0_f32; w]);
        }

        Self {
            config,
            level_w,
            level_b,
            lift_w,
            lift_b,
            project_w,
            project_b,
        }
    }

    /// Haar wavelet decomposition.
    ///
    /// Returns `(approx, detail)` each of length `n/2`.
    /// Requires `n` to be even.
    pub fn haar_decompose(x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = x.len();
        let half = n / 2;
        let sqrt2_inv = 1.0_f32 / 2.0_f32.sqrt();
        let approx: Vec<f32> = (0..half)
            .map(|i| (x[2 * i] + x[2 * i + 1]) * sqrt2_inv)
            .collect();
        let detail: Vec<f32> = (0..half)
            .map(|i| (x[2 * i] - x[2 * i + 1]) * sqrt2_inv)
            .collect();
        (approx, detail)
    }

    /// Haar wavelet reconstruction.
    ///
    /// Returns length `2 * approx.len()`.
    pub fn haar_reconstruct(approx: &[f32], detail: &[f32]) -> Vec<f32> {
        let half = approx.len();
        let sqrt2_inv = 1.0_f32 / 2.0_f32.sqrt();
        let mut out = vec![0.0_f32; 2 * half];
        for i in 0..half {
            out[2 * i] = (approx[i] + detail[i]) * sqrt2_inv;
            out[2 * i + 1] = (approx[i] - detail[i]) * sqrt2_inv;
        }
        out
    }

    /// Forward pass: `[n × d_in]` → `[n × d_out]`.
    ///
    /// Applies multi-level Haar decomposition, per-level linear kernel,
    /// then reconstructs.
    pub fn forward(&self, input: &[f32], n: usize) -> PinnResult<Vec<f32>> {
        let w = self.config.width;
        let d_in = self.config.d_in;
        let d_out = self.config.d_out;
        let levels = self.config.levels;

        if input.len() != n * d_in {
            return Err(PinnError::DimensionMismatch {
                expected: n * d_in,
                got: input.len(),
            });
        }
        if n < 2 {
            return Err(PinnError::InvalidGridResolution { n });
        }

        // Check that n is divisible by 2^levels
        let min_n = 1usize << levels;
        if n < min_n || n % min_n != 0 {
            return Err(PinnError::InvalidGridResolution { n });
        }

        // Lift: [n × d_in] → [n × w]
        let mut x_w = vec![0.0_f32; n * w];
        for i in 0..n {
            for c_out in 0..w {
                let dot: f32 = (0..d_in)
                    .map(|c_in| self.lift_w[c_out * d_in + c_in] * input[i * d_in + c_in])
                    .sum();
                x_w[i * w + c_out] = dot + self.lift_b[c_out];
            }
        }

        // Multi-level wavelet decomposition + kernel + reconstruction
        for (lev, (kw, kb)) in self.level_w.iter().zip(self.level_b.iter()).enumerate() {
            let cur_n = n >> lev; // size at this level
            if cur_n < 2 {
                break;
            }
            // Decompose each channel
            let mut approx_ch = vec![0.0_f32; (cur_n / 2) * w];
            let mut detail_ch = vec![0.0_f32; (cur_n / 2) * w];
            for c in 0..w {
                let ch: Vec<f32> = (0..cur_n).map(|i| x_w[i * w + c]).collect();
                let (a, d) = Self::haar_decompose(&ch);
                for i in 0..cur_n / 2 {
                    approx_ch[i * w + c] = a[i];
                    detail_ch[i * w + c] = d[i];
                }
            }

            // Apply linear kernel to approximation coefficients
            let half_n = cur_n / 2;
            let mut approx_out = vec![0.0_f32; half_n * w];
            for i in 0..half_n {
                for c_out in 0..w {
                    let dot: f32 = (0..w)
                        .map(|c_in| kw[c_out * w + c_in] * approx_ch[i * w + c_in])
                        .sum();
                    approx_out[i * w + c_out] = (dot + kb[c_out]).tanh();
                }
            }

            // Reconstruct from transformed approx + original detail
            for c in 0..w {
                let a_ch: Vec<f32> = (0..half_n).map(|i| approx_out[i * w + c]).collect();
                let d_ch: Vec<f32> = (0..half_n).map(|i| detail_ch[i * w + c]).collect();
                let recon = Self::haar_reconstruct(&a_ch, &d_ch);
                for i in 0..cur_n {
                    x_w[i * w + c] = recon[i];
                }
            }
        }

        // Project: [n × w] → [n × d_out]
        let mut output = vec![0.0_f32; n * d_out];
        for i in 0..n {
            for c_out in 0..d_out {
                let dot: f32 = (0..w)
                    .map(|c_in| self.project_w[c_out * w + c_in] * x_w[i * w + c_in])
                    .sum();
                output[i * d_out + c_out] = dot + self.project_b[c_out];
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haar_roundtrip() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (a, d) = Mwt::haar_decompose(&x);
        let x_back = Mwt::haar_reconstruct(&a, &d);
        for (xi, xbi) in x.iter().zip(x_back.iter()) {
            assert!(
                (xi - xbi).abs() < 1e-5,
                "Haar roundtrip failed: {xi} vs {xbi}"
            );
        }
    }

    #[test]
    fn haar_decompose_length() {
        let x = vec![0.0_f32; 16];
        let (a, d) = Mwt::haar_decompose(&x);
        assert_eq!(a.len(), 8);
        assert_eq!(d.len(), 8);
    }

    #[test]
    fn haar_reconstruct_length() {
        let a = vec![1.0_f32; 4];
        let d = vec![0.5_f32; 4];
        let r = Mwt::haar_reconstruct(&a, &d);
        assert_eq!(r.len(), 8);
    }

    #[test]
    fn mwt_construct_no_panic() {
        let mut rng = LcgRng::new(1);
        let cfg = MwtConfig {
            levels: 2,
            width: 8,
            d_in: 1,
            d_out: 1,
        };
        let _mwt = Mwt::new(cfg, &mut rng);
    }

    #[test]
    fn mwt_forward_shape() {
        let mut rng = LcgRng::new(2);
        let cfg = MwtConfig {
            levels: 2,
            width: 8,
            d_in: 1,
            d_out: 1,
        };
        let mwt = Mwt::new(cfg, &mut rng);
        let n = 16;
        let input = vec![0.5_f32; n];
        let output = mwt.forward(&input, n).unwrap();
        assert_eq!(output.len(), n);
    }

    #[test]
    fn mwt_forward_finite() {
        let mut rng = LcgRng::new(3);
        let cfg = MwtConfig {
            levels: 2,
            width: 8,
            d_in: 1,
            d_out: 1,
        };
        let mwt = Mwt::new(cfg, &mut rng);
        let n = 16;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let output = mwt.forward(&input, n).unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }
}
