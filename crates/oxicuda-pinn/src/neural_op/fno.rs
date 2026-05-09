//! Fourier Neural Operator (1D and 2D).

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;

// ─── DFT helpers ─────────────────────────────────────────────────────────────

/// O(N²) DFT of a real signal. Returns `(real, imag)` of length N.
pub fn dft_1d(x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = x.len();
    let mut real = vec![0.0_f32; n];
    let mut imag = vec![0.0_f32; n];
    for k in 0..n {
        for (j, &xj) in x.iter().enumerate() {
            let angle = -2.0 * std::f32::consts::PI * k as f32 * j as f32 / n as f32;
            real[k] += xj * angle.cos();
            imag[k] += xj * angle.sin();
        }
    }
    (real, imag)
}

/// O(N²) IDFT. Returns real output of length N.
pub fn idft_1d(real: &[f32], imag: &[f32]) -> Vec<f32> {
    let n = real.len();
    (0..n)
        .map(|j| {
            let s: f32 = (0..n)
                .map(|k| {
                    let angle = 2.0 * std::f32::consts::PI * k as f32 * j as f32 / n as f32;
                    real[k] * angle.cos() - imag[k] * angle.sin()
                })
                .sum();
            s / n as f32
        })
        .collect()
}

// ─── FNO 1D ──────────────────────────────────────────────────────────────────

/// Configuration for 1D Fourier Neural Operator.
pub struct Fno1dConfig {
    pub d_in: usize,
    pub d_out: usize,
    pub width: usize,
    pub k_max: usize,
    pub n_blocks: usize,
}

struct SpectralBlock1d {
    /// Complex weights [width × width × k_max] stored as real/imag separately.
    w_real: Vec<f32>,
    w_imag: Vec<f32>,
    /// Pointwise linear weights [width × width].
    local_w: Vec<f32>,
    local_b: Vec<f32>,
}

impl SpectralBlock1d {
    fn new(width: usize, k_max: usize, rng: &mut LcgRng) -> Self {
        let scale = (2.0 / (width * k_max) as f32).sqrt();
        let n_spec = width * width * k_max;
        let n_loc = width * width;

        let w_real: Vec<f32> = (0..n_spec)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();
        let w_imag: Vec<f32> = (0..n_spec)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();
        let local_w: Vec<f32> = (0..n_loc)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * (2.0 / width as f32).sqrt())
            .collect();
        let local_b = vec![0.0_f32; width];

        Self {
            w_real,
            w_imag,
            local_w,
            local_b,
        }
    }

    /// Spectral conv: x_real/imag [width × n_modes] → out_real/imag [width × n_modes].
    fn spectral_conv(
        &self,
        x_real: &[f32],
        x_imag: &[f32],
        n_out: usize,
        k_max: usize,
        width: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut out_r = vec![0.0_f32; n_out * k_max];
        let mut out_i = vec![0.0_f32; n_out * k_max];

        for j in 0..n_out {
            for k in 0..k_max {
                let mut r = 0.0_f32;
                let mut im = 0.0_f32;
                for i in 0..width {
                    let w_r = self.w_real[i * n_out * k_max + j * k_max + k];
                    let w_i = self.w_imag[i * n_out * k_max + j * k_max + k];
                    let xr = x_real[i * k_max + k];
                    let xi = x_imag[i * k_max + k];
                    r += xr * w_r - xi * w_i;
                    im += xr * w_i + xi * w_r;
                }
                out_r[j * k_max + k] = r;
                out_i[j * k_max + k] = im;
            }
        }
        (out_r, out_i)
    }
}

/// 1D Fourier Neural Operator.
pub struct Fno1d {
    config: Fno1dConfig,
    lift_w: Vec<f32>,
    lift_b: Vec<f32>,
    project_w: Vec<f32>,
    project_b: Vec<f32>,
    blocks: Vec<SpectralBlock1d>,
}

fn gelu(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

impl Fno1d {
    /// Construct a new FNO-1D.
    pub fn new(config: Fno1dConfig, rng: &mut LcgRng) -> Self {
        let d_in = config.d_in;
        let d_out = config.d_out;
        let w = config.width;
        let k = config.k_max;

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

        let blocks = (0..config.n_blocks)
            .map(|_| SpectralBlock1d::new(w, k, rng))
            .collect();

        Self {
            config,
            lift_w,
            lift_b,
            project_w,
            project_b,
            blocks,
        }
    }

    /// Forward pass: input `[n × d_in]` → output `[n × d_out]`.
    pub fn forward(&self, input: &[f32], n: usize) -> PinnResult<Vec<f32>> {
        let d_in = self.config.d_in;
        let d_out = self.config.d_out;
        let w = self.config.width;
        let k_max = self.config.k_max;

        if input.len() != n * d_in {
            return Err(PinnError::DimensionMismatch {
                expected: n * d_in,
                got: input.len(),
            });
        }
        if n < 2 {
            return Err(PinnError::InvalidGridResolution { n });
        }
        if k_max > n / 2 {
            return Err(PinnError::TooManyFourierModes {
                k_max,
                n_half: n / 2,
            });
        }

        // Step 1: Lift [n × d_in] → [n × width]
        let mut x_w = vec![0.0_f32; n * w];
        for i in 0..n {
            for c_out in 0..w {
                let dot: f32 = (0..d_in)
                    .map(|c_in| self.lift_w[c_out * d_in + c_in] * input[i * d_in + c_in])
                    .sum();
                x_w[i * w + c_out] = dot + self.lift_b[c_out];
            }
        }

        // Step 2: Spectral blocks
        for block in &self.blocks {
            // DFT per channel: x_w is [n × w], produce [w × n] Fourier
            // Then take first k_max modes
            let mut xr_modes = vec![0.0_f32; w * k_max];
            let mut xi_modes = vec![0.0_f32; w * k_max];

            for c in 0..w {
                let channel: Vec<f32> = (0..n).map(|i| x_w[i * w + c]).collect();
                let (r, im) = dft_1d(&channel);
                for k in 0..k_max {
                    xr_modes[c * k_max + k] = r[k];
                    xi_modes[c * k_max + k] = im[k];
                }
            }

            // Spectral multiply
            let (out_r, out_i) = block.spectral_conv(&xr_modes, &xi_modes, w, k_max, w);

            // IDFT back to spatial (zero-pad high modes)
            let mut x_spectral = vec![0.0_f32; n * w];
            for c in 0..w {
                let mut r_full = vec![0.0_f32; n];
                let mut i_full = vec![0.0_f32; n];
                for k in 0..k_max {
                    r_full[k] = out_r[c * k_max + k];
                    i_full[k] = out_i[c * k_max + k];
                }
                let spatial = idft_1d(&r_full, &i_full);
                for i in 0..n {
                    x_spectral[i * w + c] = spatial[i];
                }
            }

            // Local pointwise linear
            let mut x_local = vec![0.0_f32; n * w];
            for i in 0..n {
                for c_out in 0..w {
                    let dot: f32 = (0..w)
                        .map(|c_in| block.local_w[c_out * w + c_in] * x_w[i * w + c_in])
                        .sum();
                    x_local[i * w + c_out] = dot + block.local_b[c_out];
                }
            }

            // Add skip + GeLU
            for idx in 0..n * w {
                x_w[idx] = gelu(x_spectral[idx] + x_local[idx]);
            }
        }

        // Step 3: Project [n × width] → [n × d_out]
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

// ─── FNO 2D ──────────────────────────────────────────────────────────────────

/// Configuration for 2D Fourier Neural Operator.
pub struct Fno2dConfig {
    pub d_in: usize,
    pub d_out: usize,
    pub width: usize,
    pub k_max: usize,
    pub n_blocks: usize,
}

/// 2D spectral block.
struct SpectralBlock2d {
    w_real: Vec<f32>, // [width × width × k_max × k_max]
    w_imag: Vec<f32>,
    local_w: Vec<f32>,
    local_b: Vec<f32>,
}

impl SpectralBlock2d {
    fn new(width: usize, k_max: usize, rng: &mut LcgRng) -> Self {
        let scale = (2.0 / (width * k_max * k_max) as f32).sqrt();
        let n_spec = width * width * k_max * k_max;
        let n_loc = width * width;

        let w_real: Vec<f32> = (0..n_spec)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();
        let w_imag: Vec<f32> = (0..n_spec)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
            .collect();
        let local_w: Vec<f32> = (0..n_loc)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * (2.0 / width as f32).sqrt())
            .collect();
        let local_b = vec![0.0_f32; width];

        Self {
            w_real,
            w_imag,
            local_w,
            local_b,
        }
    }
}

/// 2D DFT via separable 1D DFTs (row-wise then column-wise).
fn dft_2d(x: &[f32], nx: usize, ny: usize) -> (Vec<f32>, Vec<f32>) {
    // Row-wise DFT
    let mut r1 = vec![0.0_f32; nx * ny];
    let mut i1 = vec![0.0_f32; nx * ny];
    for row in 0..nx {
        let row_data: Vec<f32> = (0..ny).map(|col| x[row * ny + col]).collect();
        let (rr, ri) = dft_1d(&row_data);
        for col in 0..ny {
            r1[row * ny + col] = rr[col];
            i1[row * ny + col] = ri[col];
        }
    }
    // Column-wise DFT on row-DFT result
    let mut r2 = vec![0.0_f32; nx * ny];
    let mut i2 = vec![0.0_f32; nx * ny];
    for col in 0..ny {
        let col_r: Vec<f32> = (0..nx).map(|row| r1[row * ny + col]).collect();
        let col_i: Vec<f32> = (0..nx).map(|row| i1[row * ny + col]).collect();
        // DFT of complex sequence: handle real/imag together
        for k in 0..nx {
            let mut sr = 0.0_f32;
            let mut si = 0.0_f32;
            for n in 0..nx {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / nx as f32;
                let (ca, sa) = (angle.cos(), angle.sin());
                sr += col_r[n] * ca - col_i[n] * sa;
                si += col_r[n] * sa + col_i[n] * ca;
            }
            r2[k * ny + col] = sr;
            i2[k * ny + col] = si;
        }
    }
    (r2, i2)
}

/// 2D IDFT via separable column-wise then row-wise IDFT.
fn idft_2d(real: &[f32], imag: &[f32], nx: usize, ny: usize) -> Vec<f32> {
    // Column-wise IDFT first (equivalent to conjugate-DFT / n)
    let mut r1 = vec![0.0_f32; nx * ny];
    let mut i1 = vec![0.0_f32; nx * ny];
    for col in 0..ny {
        for n in 0..nx {
            let mut sr = 0.0_f32;
            let mut si = 0.0_f32;
            for k in 0..nx {
                let angle = 2.0 * std::f32::consts::PI * k as f32 * n as f32 / nx as f32;
                let (ca, sa) = (angle.cos(), angle.sin());
                sr += real[k * ny + col] * ca - imag[k * ny + col] * sa;
                si += real[k * ny + col] * sa + imag[k * ny + col] * ca;
            }
            r1[n * ny + col] = sr / nx as f32;
            i1[n * ny + col] = si / nx as f32;
        }
    }
    // Row-wise IDFT
    let mut out = vec![0.0_f32; nx * ny];
    for row in 0..nx {
        let row_r: Vec<f32> = (0..ny).map(|col| r1[row * ny + col]).collect();
        let row_i: Vec<f32> = (0..ny).map(|col| i1[row * ny + col]).collect();
        let spatial = idft_1d(&row_r, &row_i);
        for col in 0..ny {
            out[row * ny + col] = spatial[col];
        }
    }
    out
}

/// 2D Fourier Neural Operator.
pub struct Fno2d {
    config: Fno2dConfig,
    lift_w: Vec<f32>,
    lift_b: Vec<f32>,
    project_w: Vec<f32>,
    project_b: Vec<f32>,
    blocks: Vec<SpectralBlock2d>,
}

impl Fno2d {
    /// Construct a new FNO-2D.
    pub fn new(config: Fno2dConfig, rng: &mut LcgRng) -> Self {
        let d_in = config.d_in;
        let d_out = config.d_out;
        let w = config.width;

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

        let k = config.k_max;
        let blocks = (0..config.n_blocks)
            .map(|_| SpectralBlock2d::new(w, k, rng))
            .collect();

        Self {
            config,
            lift_w,
            lift_b,
            project_w,
            project_b,
            blocks,
        }
    }

    /// Forward pass: input `[nx × ny × d_in]` → output `[nx × ny × d_out]`.
    pub fn forward(&self, input: &[f32], nx: usize, ny: usize) -> PinnResult<Vec<f32>> {
        let d_in = self.config.d_in;
        let d_out = self.config.d_out;
        let w = self.config.width;
        let k_max = self.config.k_max;

        if input.len() != nx * ny * d_in {
            return Err(PinnError::DimensionMismatch {
                expected: nx * ny * d_in,
                got: input.len(),
            });
        }
        if nx < 2 || ny < 2 {
            return Err(PinnError::InvalidGridResolution { n: nx.min(ny) });
        }
        if k_max > nx / 2 || k_max > ny / 2 {
            return Err(PinnError::TooManyFourierModes {
                k_max,
                n_half: nx.min(ny) / 2,
            });
        }

        // Lift
        let mut x_w = vec![0.0_f32; nx * ny * w];
        for i in 0..nx * ny {
            for c_out in 0..w {
                let dot: f32 = (0..d_in)
                    .map(|c_in| self.lift_w[c_out * d_in + c_in] * input[i * d_in + c_in])
                    .sum();
                x_w[i * w + c_out] = dot + self.lift_b[c_out];
            }
        }

        // Spectral blocks
        for block in &self.blocks {
            let mut x_spectral = vec![0.0_f32; nx * ny * w];

            for c in 0..w {
                // Extract channel: [nx × ny]
                let channel: Vec<f32> = (0..nx * ny).map(|i| x_w[i * w + c]).collect();
                let (fr, fi) = dft_2d(&channel, nx, ny);

                // Keep only k_max × k_max corner modes, zero others
                // For each output channel j
                let mut out_fr = fr.clone();
                let mut out_fi = fi.clone();
                // Apply spectral weights in top-left k_max×k_max corner
                for kx in 0..k_max {
                    for ky in 0..k_max {
                        let in_idx = kx * ny + ky;
                        let wr = block.w_real
                            [c * w * k_max * k_max + c * k_max * k_max + kx * k_max + ky];
                        let wi = block.w_imag
                            [c * w * k_max * k_max + c * k_max * k_max + kx * k_max + ky];
                        let xr = fr[in_idx];
                        let xi = fi[in_idx];
                        out_fr[in_idx] = xr * wr - xi * wi;
                        out_fi[in_idx] = xr * wi + xi * wr;
                    }
                }
                // Zero high modes
                for kx in 0..nx {
                    for ky in 0..ny {
                        if kx >= k_max || ky >= k_max {
                            out_fr[kx * ny + ky] = 0.0;
                            out_fi[kx * ny + ky] = 0.0;
                        }
                    }
                }

                let spatial = idft_2d(&out_fr, &out_fi, nx, ny);
                for i in 0..nx * ny {
                    x_spectral[i * w + c] = spatial[i];
                }
            }

            // Local pointwise
            let mut x_local = vec![0.0_f32; nx * ny * w];
            for i in 0..nx * ny {
                for c_out in 0..w {
                    let dot: f32 = (0..w)
                        .map(|c_in| block.local_w[c_out * w + c_in] * x_w[i * w + c_in])
                        .sum();
                    x_local[i * w + c_out] = dot + block.local_b[c_out];
                }
            }

            for idx in 0..nx * ny * w {
                x_w[idx] = gelu(x_spectral[idx] + x_local[idx]);
            }
        }

        // Project
        let mut output = vec![0.0_f32; nx * ny * d_out];
        for i in 0..nx * ny {
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
    fn dft_idft_roundtrip() {
        let x: Vec<f32> = (0..8).map(|i| (i as f32).sin()).collect();
        let (r, im) = dft_1d(&x);
        let x_back = idft_1d(&r, &im);
        for (a, b) in x.iter().zip(x_back.iter()) {
            assert!((a - b).abs() < 1e-3, "DFT roundtrip failed: {a} vs {b}");
        }
    }

    #[test]
    fn dft_dc_component() {
        // DC component = sum of input
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let (r, _) = dft_1d(&x);
        assert!((r[0] - 10.0).abs() < 1e-4, "DC = {}", r[0]);
    }

    #[test]
    fn fno1d_new_no_panic() {
        let mut rng = LcgRng::new(1);
        let cfg = Fno1dConfig {
            d_in: 1,
            d_out: 1,
            width: 8,
            k_max: 4,
            n_blocks: 2,
        };
        let _fno = Fno1d::new(cfg, &mut rng);
    }

    #[test]
    fn fno1d_forward_shape() {
        let mut rng = LcgRng::new(2);
        let cfg = Fno1dConfig {
            d_in: 1,
            d_out: 1,
            width: 8,
            k_max: 4,
            n_blocks: 2,
        };
        let fno = Fno1d::new(cfg, &mut rng);
        let n = 16;
        let input = vec![0.5_f32; n];
        let output = fno.forward(&input, n).unwrap();
        assert_eq!(output.len(), n);
    }

    #[test]
    fn fno1d_forward_all_finite() {
        let mut rng = LcgRng::new(3);
        let cfg = Fno1dConfig {
            d_in: 1,
            d_out: 1,
            width: 8,
            k_max: 4,
            n_blocks: 1,
        };
        let fno = Fno1d::new(cfg, &mut rng);
        let n = 16;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let output = fno.forward(&input, n).unwrap();
        assert!(
            output.iter().all(|v| v.is_finite()),
            "FNO1d output not finite"
        );
    }

    #[test]
    fn fno1d_too_many_modes_error() {
        let mut rng = LcgRng::new(4);
        let cfg = Fno1dConfig {
            d_in: 1,
            d_out: 1,
            width: 4,
            k_max: 10,
            n_blocks: 1,
        };
        let fno = Fno1d::new(cfg, &mut rng);
        let result = fno.forward(&[0.0; 8], 8);
        assert!(matches!(result, Err(PinnError::TooManyFourierModes { .. })));
    }

    #[test]
    fn fno1d_dim_mismatch_error() {
        let mut rng = LcgRng::new(5);
        let cfg = Fno1dConfig {
            d_in: 2,
            d_out: 1,
            width: 4,
            k_max: 2,
            n_blocks: 1,
        };
        let fno = Fno1d::new(cfg, &mut rng);
        let result = fno.forward(&[0.0; 10], 8); // expects 16 (8 * 2)
        assert!(result.is_err());
    }

    #[test]
    fn fno2d_new_no_panic() {
        let mut rng = LcgRng::new(6);
        let cfg = Fno2dConfig {
            d_in: 1,
            d_out: 1,
            width: 4,
            k_max: 2,
            n_blocks: 1,
        };
        let _fno = Fno2d::new(cfg, &mut rng);
    }

    #[test]
    fn fno2d_forward_shape() {
        let mut rng = LcgRng::new(7);
        let cfg = Fno2dConfig {
            d_in: 1,
            d_out: 1,
            width: 4,
            k_max: 2,
            n_blocks: 1,
        };
        let fno = Fno2d::new(cfg, &mut rng);
        let (nx, ny) = (8, 8);
        let input = vec![0.1_f32; nx * ny];
        let output = fno.forward(&input, nx, ny).unwrap();
        assert_eq!(output.len(), nx * ny);
    }

    #[test]
    fn fno2d_forward_all_finite() {
        let mut rng = LcgRng::new(8);
        let cfg = Fno2dConfig {
            d_in: 1,
            d_out: 1,
            width: 4,
            k_max: 2,
            n_blocks: 1,
        };
        let fno = Fno2d::new(cfg, &mut rng);
        let input = vec![0.3_f32; 8 * 8];
        let output = fno.forward(&input, 8, 8).unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dft_pure_cosine() {
        // cos(2πk/N) should produce peaks at k and N-k
        let n = 8;
        let freq = 1;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq as f32 * i as f32 / n as f32).cos())
            .collect();
        let (r, _) = dft_1d(&x);
        // DC component should be ~0, peak at freq should be ~N/2
        assert!(r[0].abs() < 0.01);
        assert!(r[freq] > 3.0);
    }

    #[test]
    fn fno1d_multi_channel_forward() {
        let mut rng = LcgRng::new(9);
        let cfg = Fno1dConfig {
            d_in: 2,
            d_out: 3,
            width: 8,
            k_max: 4,
            n_blocks: 1,
        };
        let fno = Fno1d::new(cfg, &mut rng);
        let n = 16;
        let input = vec![0.1_f32; n * 2];
        let output = fno.forward(&input, n).unwrap();
        assert_eq!(output.len(), n * 3);
    }
}
