//! Fourier Neural Operator — 3D (volumetric).
//!
//! Implements the volumetric Fourier Neural Operator from Li et al. 2021
//! ("Fourier Neural Operator for Parametric Partial Differential Equations"),
//! generalized to three spatial axes. A feature field of shape
//! `(in_channels, grid_x, grid_y, grid_z)` is transformed by a 3D DFT,
//! the top `(modes_x, modes_y, modes_z)` Fourier modes are kept (all higher
//! modes are zeroed out), a learnable complex-valued linear map is applied
//! per kept mode across channels, the resulting truncated spectrum is
//! transformed back to real space by the inverse 3D DFT, and finally a
//! 1×1×1 (per-voxel) linear residual `W·x + b` over channels is added.
//!
//! 3D DFTs are computed by three successive O(N²) 1D DFTs along the
//! `x`, `y`, `z` axes — sufficient for the test-scale grids (≤ 8) targeted
//! by the unit tests in this module.
//!
//! Reference: Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B.,
//! Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021).
//! *Fourier Neural Operator for Parametric Partial Differential Equations.*
//! ICLR 2021.

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;

// ───────────────────────────── Configuration ─────────────────────────────

/// Configuration for a 3D Fourier Neural Operator.
#[derive(Debug, Clone)]
pub struct Fno3dConfig {
    /// Number of input feature channels.
    pub in_channels: usize,
    /// Number of output feature channels.
    pub out_channels: usize,
    /// Number of Fourier modes kept along `x`. Must satisfy `modes_x ≤ grid_x/2 + 1`.
    pub modes_x: usize,
    /// Number of Fourier modes kept along `y`. Must satisfy `modes_y ≤ grid_y/2 + 1`.
    pub modes_y: usize,
    /// Number of Fourier modes kept along `z`. Must satisfy `modes_z ≤ grid_z/2 + 1`.
    pub modes_z: usize,
    /// Number of spatial samples along `x`. Must be ≥ 1.
    pub grid_x: usize,
    /// Number of spatial samples along `y`. Must be ≥ 1.
    pub grid_y: usize,
    /// Number of spatial samples along `z`. Must be ≥ 1.
    pub grid_z: usize,
}

// ─────────────────────────── Operator structure ──────────────────────────

/// 3D Fourier Neural Operator.
///
/// Internal layout:
///
/// - `spectral_real`, `spectral_imag`: per-mode `in_channels × out_channels`
///   complex weights, flattened as
///   `[mode_x][mode_y][mode_z][channel_in][channel_out]`, total length
///   `modes_x · modes_y · modes_z · in_channels · out_channels`.
/// - `residual_w`: `out_channels × in_channels` 1×1×1 linear residual matrix,
///   flattened row-major as `[out_channel][in_channel]`.
/// - `residual_b`: `out_channels` length residual bias vector.
pub struct Fno3d {
    cfg: Fno3dConfig,
    spectral_real: Vec<f32>,
    spectral_imag: Vec<f32>,
    residual_w: Vec<f32>,
    residual_b: Vec<f32>,
}

impl Fno3d {
    /// Construct a new `Fno3d` with Gaussian-initialised spectral weights and
    /// He-uniform residual weights.
    ///
    /// # Errors
    /// - `InvalidLayerWidth` if `in_channels == 0` or `out_channels == 0`.
    /// - `InvalidGridResolution` if any of `grid_x, grid_y, grid_z` is zero.
    /// - `TooManyFourierModes` if any `modes_*` exceeds `grid_*/2 + 1`.
    pub fn new(cfg: Fno3dConfig, rng: &mut LcgRng) -> PinnResult<Self> {
        Self::validate_cfg(&cfg)?;

        let n_modes = cfg.modes_x * cfg.modes_y * cfg.modes_z;
        let n_spec = n_modes * cfg.in_channels * cfg.out_channels;
        let n_res = cfg.out_channels * cfg.in_channels;

        // Spectral scale: 1 / (in_channels · out_channels) keeps initial
        // forward magnitudes well bounded across modes.
        let denom = (cfg.in_channels * cfg.out_channels).max(1) as f32;
        let spec_scale = 1.0_f32 / denom;
        let mut spectral_real = Vec::with_capacity(n_spec);
        let mut spectral_imag = Vec::with_capacity(n_spec);
        for _ in 0..n_spec {
            let (a, b) = rng.next_normal_pair();
            spectral_real.push(a * spec_scale);
            spectral_imag.push(b * spec_scale);
        }

        // He-style residual: scale = sqrt(2 / in_channels).
        let res_scale = (2.0_f32 / (cfg.in_channels as f32).max(1.0)).sqrt();
        let mut residual_w = Vec::with_capacity(n_res);
        for _ in 0..n_res {
            let u = (rng.next_u32() as f32) / (u32::MAX as f32 + 1.0);
            residual_w.push((u * 2.0 - 1.0) * res_scale);
        }
        let residual_b = vec![0.0_f32; cfg.out_channels];

        Ok(Self {
            cfg,
            spectral_real,
            spectral_imag,
            residual_w,
            residual_b,
        })
    }

    fn validate_cfg(cfg: &Fno3dConfig) -> PinnResult<()> {
        if cfg.in_channels == 0 || cfg.out_channels == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        if cfg.grid_x == 0 || cfg.grid_y == 0 || cfg.grid_z == 0 {
            return Err(PinnError::InvalidGridResolution {
                n: cfg.grid_x.min(cfg.grid_y).min(cfg.grid_z),
            });
        }
        let n_half_x = cfg.grid_x / 2 + 1;
        let n_half_y = cfg.grid_y / 2 + 1;
        let n_half_z = cfg.grid_z / 2 + 1;
        if cfg.modes_x > n_half_x {
            return Err(PinnError::TooManyFourierModes {
                k_max: cfg.modes_x,
                n_half: n_half_x,
            });
        }
        if cfg.modes_y > n_half_y {
            return Err(PinnError::TooManyFourierModes {
                k_max: cfg.modes_y,
                n_half: n_half_y,
            });
        }
        if cfg.modes_z > n_half_z {
            return Err(PinnError::TooManyFourierModes {
                k_max: cfg.modes_z,
                n_half: n_half_z,
            });
        }
        Ok(())
    }

    /// Total number of trainable parameters: complex spectral weights are
    /// counted as two real numbers per (mode × in_channel × out_channel),
    /// plus the real residual `W` (`out × in`) and the bias (`out`).
    #[must_use]
    pub fn n_params(&self) -> usize {
        let n_modes = self.cfg.modes_x * self.cfg.modes_y * self.cfg.modes_z;
        let spec_real_terms = n_modes * self.cfg.in_channels * self.cfg.out_channels;
        // Real + imag: factor 2.
        let spec_total = 2 * spec_real_terms;
        let res_w = self.cfg.out_channels * self.cfg.in_channels;
        let res_b = self.cfg.out_channels;
        spec_total + res_w + res_b
    }

    /// Configuration accessor.
    #[must_use]
    pub fn config(&self) -> &Fno3dConfig {
        &self.cfg
    }

    // ─────────────────────── DFT / iDFT (complex, 3D) ────────────────────

    /// Forward 3D DFT of a complex volumetric field.
    ///
    /// Both inputs and outputs have length `grid_x · grid_y · grid_z` and use
    /// row-major layout `[ix][iy][iz]`.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `real.len()` or `imag.len()` differs from the
    ///   expected `grid_x · grid_y · grid_z`.
    pub fn dft_3d(&self, real: &[f32], imag: &[f32]) -> PinnResult<(Vec<f32>, Vec<f32>)> {
        dft_3d_volumetric(
            real,
            imag,
            self.cfg.grid_x,
            self.cfg.grid_y,
            self.cfg.grid_z,
        )
    }

    /// Inverse 3D DFT of a complex volumetric field, normalised by
    /// `1 / (grid_x · grid_y · grid_z)`.
    ///
    /// # Errors
    /// - `DimensionMismatch` if input lengths do not match the configured grid.
    pub fn idft_3d(&self, real: &[f32], imag: &[f32]) -> PinnResult<(Vec<f32>, Vec<f32>)> {
        idft_3d_volumetric(
            real,
            imag,
            self.cfg.grid_x,
            self.cfg.grid_y,
            self.cfg.grid_z,
        )
    }

    // ─────────────────────────────── Forward ─────────────────────────────

    /// Forward pass over a real volumetric feature field.
    ///
    /// `x` has length `in_channels · grid_x · grid_y · grid_z` and row-major
    /// layout `[channel][ix][iy][iz]`. The returned vector has length
    /// `out_channels · grid_x · grid_y · grid_z` with the same layout.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `x.len()` is not the expected size.
    /// - `NanEncountered` if any computed value is non-finite.
    pub fn forward(&self, x: &[f32]) -> PinnResult<Vec<f32>> {
        let nx = self.cfg.grid_x;
        let ny = self.cfg.grid_y;
        let nz = self.cfg.grid_z;
        let cin = self.cfg.in_channels;
        let cout = self.cfg.out_channels;
        let voxels = nx * ny * nz;
        let expected = cin * voxels;
        if x.len() != expected {
            return Err(PinnError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }

        // 1. Forward 3D DFT of each input channel (real input → imag = 0).
        let mut in_real: Vec<Vec<f32>> = Vec::with_capacity(cin);
        let mut in_imag: Vec<Vec<f32>> = Vec::with_capacity(cin);
        for c in 0..cin {
            let slice = &x[c * voxels..(c + 1) * voxels];
            let imag = vec![0.0_f32; voxels];
            let (fr, fi) = dft_3d_volumetric(slice, &imag, nx, ny, nz)?;
            in_real.push(fr);
            in_imag.push(fi);
        }

        // 2. Build truncated output spectrum: only the (modes_x, modes_y,
        //    modes_z) low-frequency corner is filled by the per-mode complex
        //    linear map across channels; all higher modes remain zero.
        let mut out_real: Vec<Vec<f32>> = (0..cout).map(|_| vec![0.0_f32; voxels]).collect();
        let mut out_imag: Vec<Vec<f32>> = (0..cout).map(|_| vec![0.0_f32; voxels]).collect();

        let mx = self.cfg.modes_x;
        let my = self.cfg.modes_y;
        let mz = self.cfg.modes_z;

        for kx in 0..mx {
            for ky in 0..my {
                for kz in 0..mz {
                    let spec_idx_base = ((kx * my + ky) * mz + kz) * cin * cout;
                    let vox_idx = (kx * ny + ky) * nz + kz;

                    for co in 0..cout {
                        let mut acc_r = 0.0_f32;
                        let mut acc_i = 0.0_f32;
                        for ci in 0..cin {
                            let wi = spec_idx_base + ci * cout + co;
                            let wr = self.spectral_real[wi];
                            let wim = self.spectral_imag[wi];
                            let xr = in_real[ci][vox_idx];
                            let xi = in_imag[ci][vox_idx];
                            // (xr + j xi) (wr + j wim) = (xr wr - xi wim) + j(xr wim + xi wr)
                            acc_r += xr * wr - xi * wim;
                            acc_i += xr * wim + xi * wr;
                        }
                        out_real[co][vox_idx] = acc_r;
                        out_imag[co][vox_idx] = acc_i;
                    }
                }
            }
        }

        // 3. Inverse 3D DFT per output channel, keep real part only.
        let mut spectral_out = vec![0.0_f32; cout * voxels];
        for co in 0..cout {
            let (ir, _) = idft_3d_volumetric(&out_real[co], &out_imag[co], nx, ny, nz)?;
            for v in 0..voxels {
                spectral_out[co * voxels + v] = ir[v];
            }
        }

        // 4. 1×1×1 linear residual W·x + b applied per voxel over channels,
        //    then sum spectral path + residual.
        let mut output = vec![0.0_f32; cout * voxels];
        for v in 0..voxels {
            for co in 0..cout {
                let mut dot = self.residual_b[co];
                for ci in 0..cin {
                    dot += self.residual_w[co * cin + ci] * x[ci * voxels + v];
                }
                output[co * voxels + v] = spectral_out[co * voxels + v] + dot;
            }
        }

        for v in &output {
            if !v.is_finite() {
                return Err(PinnError::NanEncountered {
                    location: "fno_3d::forward",
                });
            }
        }
        Ok(output)
    }
}

// ───────────────────────── Test-only mutators ────────────────────────────
//
// The struct-level mutators below are only compiled in tests; they let the
// in-module unit tests exercise the spectral path / residual path
// independently (e.g. identity test, mode-truncation test, linearity test).

#[cfg(test)]
impl Fno3d {
    /// Zero every entry of the spectral weights.
    pub(crate) fn zero_spectral(&mut self) {
        for v in &mut self.spectral_real {
            *v = 0.0;
        }
        for v in &mut self.spectral_imag {
            *v = 0.0;
        }
    }

    /// Zero the residual `W` and bias `b`.
    pub(crate) fn zero_residual(&mut self) {
        for v in &mut self.residual_w {
            *v = 0.0;
        }
        for v in &mut self.residual_b {
            *v = 0.0;
        }
    }

    /// Set the residual to the identity matrix `W = I`, `b = 0`.
    /// Requires `in_channels == out_channels`.
    pub(crate) fn residual_identity(&mut self) {
        let cin = self.cfg.in_channels;
        let cout = self.cfg.out_channels;
        for v in &mut self.residual_w {
            *v = 0.0;
        }
        for v in &mut self.residual_b {
            *v = 0.0;
        }
        let n = cin.min(cout);
        for i in 0..n {
            self.residual_w[i * cin + i] = 1.0;
        }
    }

    /// Set per-mode spectral weights to the identity map (real part = δ_{ci,co},
    /// imag = 0). Requires `in_channels == out_channels`.
    pub(crate) fn spectral_identity(&mut self) {
        let cin = self.cfg.in_channels;
        let cout = self.cfg.out_channels;
        for v in &mut self.spectral_real {
            *v = 0.0;
        }
        for v in &mut self.spectral_imag {
            *v = 0.0;
        }
        let mx = self.cfg.modes_x;
        let my = self.cfg.modes_y;
        let mz = self.cfg.modes_z;
        let n = cin.min(cout);
        for kx in 0..mx {
            for ky in 0..my {
                for kz in 0..mz {
                    let base = ((kx * my + ky) * mz + kz) * cin * cout;
                    for i in 0..n {
                        self.spectral_real[base + i * cout + i] = 1.0;
                    }
                }
            }
        }
    }
}

// ─────────────────────── DFT / iDFT helpers (3D) ─────────────────────────

/// 1D complex DFT (O(N²)). Caller passes complex input `(real, imag)` of
/// length `n`; output is complex of length `n`.
fn dft_1d_complex(real: &[f32], imag: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut out_r = vec![0.0_f32; n];
    let mut out_i = vec![0.0_f32; n];
    for k in 0..n {
        let mut sr = 0.0_f32;
        let mut si = 0.0_f32;
        for j in 0..n {
            let angle = -2.0 * std::f32::consts::PI * (k as f32) * (j as f32) / (n as f32);
            let (ca, sa) = (angle.cos(), angle.sin());
            // (xr + j xi)(ca + j sa) = (xr ca - xi sa) + j(xr sa + xi ca)
            sr += real[j] * ca - imag[j] * sa;
            si += real[j] * sa + imag[j] * ca;
        }
        out_r[k] = sr;
        out_i[k] = si;
    }
    (out_r, out_i)
}

/// 1D complex inverse DFT (O(N²)), normalised by `1/n`.
fn idft_1d_complex(real: &[f32], imag: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut out_r = vec![0.0_f32; n];
    let mut out_i = vec![0.0_f32; n];
    let inv_n = 1.0_f32 / (n as f32);
    for j in 0..n {
        let mut sr = 0.0_f32;
        let mut si = 0.0_f32;
        for k in 0..n {
            let angle = 2.0 * std::f32::consts::PI * (k as f32) * (j as f32) / (n as f32);
            let (ca, sa) = (angle.cos(), angle.sin());
            sr += real[k] * ca - imag[k] * sa;
            si += real[k] * sa + imag[k] * ca;
        }
        out_r[j] = sr * inv_n;
        out_i[j] = si * inv_n;
    }
    (out_r, out_i)
}

/// Forward 3D DFT computed as three successive 1D DFTs (axis-x, then
/// axis-y, then axis-z). Inputs and outputs use row-major `[ix][iy][iz]`
/// layout with length `nx · ny · nz`.
fn dft_3d_volumetric(
    real: &[f32],
    imag: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
) -> PinnResult<(Vec<f32>, Vec<f32>)> {
    let total = nx * ny * nz;
    if real.len() != total {
        return Err(PinnError::DimensionMismatch {
            expected: total,
            got: real.len(),
        });
    }
    if imag.len() != total {
        return Err(PinnError::DimensionMismatch {
            expected: total,
            got: imag.len(),
        });
    }
    if total == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut r1 = real.to_vec();
    let mut i1 = imag.to_vec();

    // Axis x: fix (iy, iz), transform along ix.
    if nx > 1 {
        let mut buf_r = vec![0.0_f32; nx];
        let mut buf_i = vec![0.0_f32; nx];
        for iy in 0..ny {
            for iz in 0..nz {
                for ix in 0..nx {
                    let idx = (ix * ny + iy) * nz + iz;
                    buf_r[ix] = r1[idx];
                    buf_i[ix] = i1[idx];
                }
                let (tr, ti) = dft_1d_complex(&buf_r, &buf_i, nx);
                for ix in 0..nx {
                    let idx = (ix * ny + iy) * nz + iz;
                    r1[idx] = tr[ix];
                    i1[idx] = ti[ix];
                }
            }
        }
    }

    // Axis y: fix (ix, iz), transform along iy.
    if ny > 1 {
        let mut buf_r = vec![0.0_f32; ny];
        let mut buf_i = vec![0.0_f32; ny];
        for ix in 0..nx {
            for iz in 0..nz {
                for iy in 0..ny {
                    let idx = (ix * ny + iy) * nz + iz;
                    buf_r[iy] = r1[idx];
                    buf_i[iy] = i1[idx];
                }
                let (tr, ti) = dft_1d_complex(&buf_r, &buf_i, ny);
                for iy in 0..ny {
                    let idx = (ix * ny + iy) * nz + iz;
                    r1[idx] = tr[iy];
                    i1[idx] = ti[iy];
                }
            }
        }
    }

    // Axis z: fix (ix, iy), transform along iz.
    if nz > 1 {
        let mut buf_r = vec![0.0_f32; nz];
        let mut buf_i = vec![0.0_f32; nz];
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let idx = (ix * ny + iy) * nz + iz;
                    buf_r[iz] = r1[idx];
                    buf_i[iz] = i1[idx];
                }
                let (tr, ti) = dft_1d_complex(&buf_r, &buf_i, nz);
                for iz in 0..nz {
                    let idx = (ix * ny + iy) * nz + iz;
                    r1[idx] = tr[iz];
                    i1[idx] = ti[iz];
                }
            }
        }
    }

    Ok((r1, i1))
}

/// Inverse 3D DFT computed as three successive 1D inverse DFTs along each
/// axis, normalised by `1 / (nx · ny · nz)` overall (each axis pass
/// normalises by `1/length`).
fn idft_3d_volumetric(
    real: &[f32],
    imag: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
) -> PinnResult<(Vec<f32>, Vec<f32>)> {
    let total = nx * ny * nz;
    if real.len() != total {
        return Err(PinnError::DimensionMismatch {
            expected: total,
            got: real.len(),
        });
    }
    if imag.len() != total {
        return Err(PinnError::DimensionMismatch {
            expected: total,
            got: imag.len(),
        });
    }
    if total == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut r1 = real.to_vec();
    let mut i1 = imag.to_vec();

    if nx > 1 {
        let mut buf_r = vec![0.0_f32; nx];
        let mut buf_i = vec![0.0_f32; nx];
        for iy in 0..ny {
            for iz in 0..nz {
                for ix in 0..nx {
                    let idx = (ix * ny + iy) * nz + iz;
                    buf_r[ix] = r1[idx];
                    buf_i[ix] = i1[idx];
                }
                let (tr, ti) = idft_1d_complex(&buf_r, &buf_i, nx);
                for ix in 0..nx {
                    let idx = (ix * ny + iy) * nz + iz;
                    r1[idx] = tr[ix];
                    i1[idx] = ti[ix];
                }
            }
        }
    }

    if ny > 1 {
        let mut buf_r = vec![0.0_f32; ny];
        let mut buf_i = vec![0.0_f32; ny];
        for ix in 0..nx {
            for iz in 0..nz {
                for iy in 0..ny {
                    let idx = (ix * ny + iy) * nz + iz;
                    buf_r[iy] = r1[idx];
                    buf_i[iy] = i1[idx];
                }
                let (tr, ti) = idft_1d_complex(&buf_r, &buf_i, ny);
                for iy in 0..ny {
                    let idx = (ix * ny + iy) * nz + iz;
                    r1[idx] = tr[iy];
                    i1[idx] = ti[iy];
                }
            }
        }
    }

    if nz > 1 {
        let mut buf_r = vec![0.0_f32; nz];
        let mut buf_i = vec![0.0_f32; nz];
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let idx = (ix * ny + iy) * nz + iz;
                    buf_r[iz] = r1[idx];
                    buf_i[iz] = i1[idx];
                }
                let (tr, ti) = idft_1d_complex(&buf_r, &buf_i, nz);
                for iz in 0..nz {
                    let idx = (ix * ny + iy) * nz + iz;
                    r1[idx] = tr[iz];
                    i1[idx] = ti[iz];
                }
            }
        }
    }

    Ok((r1, i1))
}

// ─────────────────────────────────── tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_default(rng_seed: u64) -> Fno3d {
        let mut rng = LcgRng::new(rng_seed);
        let cfg = Fno3dConfig {
            in_channels: 2,
            out_channels: 2,
            modes_x: 2,
            modes_y: 2,
            modes_z: 2,
            grid_x: 4,
            grid_y: 4,
            grid_z: 4,
        };
        Fno3d::new(cfg, &mut rng).expect("Fno3d construction with valid params should succeed")
    }

    #[test]
    fn fno3d_forward_output_length() {
        let m = make_default(1);
        let cin = m.cfg.in_channels;
        let cout = m.cfg.out_channels;
        let voxels = m.cfg.grid_x * m.cfg.grid_y * m.cfg.grid_z;
        let input = vec![0.1_f32; cin * voxels];
        let output = m
            .forward(&input)
            .expect("Fno3d forward pass should succeed");
        assert_eq!(output.len(), cout * voxels);
    }

    #[test]
    fn fno3d_n_params_formula() {
        let mut rng = LcgRng::new(42);
        let cfg = Fno3dConfig {
            in_channels: 3,
            out_channels: 4,
            modes_x: 2,
            modes_y: 3,
            modes_z: 2,
            grid_x: 4,
            grid_y: 6,
            grid_z: 4,
        };
        let m =
            Fno3d::new(cfg, &mut rng).expect("Fno3d construction with valid params should succeed");
        // 2*3*2 = 12 modes; 12 * 3 * 4 = 144 complex weights → 288 reals
        // Residual: 4*3 = 12 weights + 4 bias = 16
        let expected = 288 + 16;
        assert_eq!(m.n_params(), expected);
    }

    #[test]
    fn fno3d_dft_idft_roundtrip() {
        let m = make_default(7);
        let nx = m.cfg.grid_x;
        let ny = m.cfg.grid_y;
        let nz = m.cfg.grid_z;
        let total = nx * ny * nz;
        let mut real = Vec::with_capacity(total);
        let mut imag = Vec::with_capacity(total);
        for i in 0..total {
            let f = i as f32;
            real.push((f * 0.13).sin());
            imag.push((f * 0.07).cos() * 0.5);
        }
        let (fr, fi) = m
            .dft_3d(&real, &imag)
            .expect("3D DFT should succeed for valid input");
        let (rr, ri) = m
            .idft_3d(&fr, &fi)
            .expect("3D IDFT should succeed after forward DFT");
        for i in 0..total {
            assert!(
                (rr[i] - real[i]).abs() < 1e-3,
                "real round-trip i={i}: {} vs {}",
                rr[i],
                real[i]
            );
            assert!(
                (ri[i] - imag[i]).abs() < 1e-3,
                "imag round-trip i={i}: {} vs {}",
                ri[i],
                imag[i]
            );
        }
    }

    #[test]
    fn fno3d_residual_identity_zero_spectral_is_identity() {
        let mut rng = LcgRng::new(11);
        let cfg = Fno3dConfig {
            in_channels: 2,
            out_channels: 2,
            modes_x: 1,
            modes_y: 1,
            modes_z: 1,
            grid_x: 3,
            grid_y: 3,
            grid_z: 3,
        };
        let mut m =
            Fno3d::new(cfg, &mut rng).expect("Fno3d construction with valid params should succeed");
        m.zero_spectral();
        m.residual_identity();

        let cin = m.cfg.in_channels;
        let voxels = m.cfg.grid_x * m.cfg.grid_y * m.cfg.grid_z;
        let mut input = Vec::with_capacity(cin * voxels);
        for i in 0..(cin * voxels) {
            input.push(((i as f32) * 0.3).sin());
        }
        let output = m
            .forward(&input)
            .expect("Fno3d forward pass should succeed");
        for i in 0..(cin * voxels) {
            assert!(
                (output[i] - input[i]).abs() < 1e-5,
                "identity at {i}: {} vs {}",
                output[i],
                input[i]
            );
        }
    }

    #[test]
    fn fno3d_changing_input_changes_output() {
        let m = make_default(2);
        let cin = m.cfg.in_channels;
        let voxels = m.cfg.grid_x * m.cfg.grid_y * m.cfg.grid_z;
        let a = vec![0.1_f32; cin * voxels];
        let mut b = a.clone();
        b[3] += 1.0;
        let oa = m.forward(&a).expect("Fno3d forward pass should succeed");
        let ob = m.forward(&b).expect("Fno3d forward pass should succeed");
        let mut diff = 0.0_f32;
        for (x, y) in oa.iter().zip(ob.iter()) {
            diff += (x - y).abs();
        }
        assert!(
            diff > 1e-6,
            "output should change when input changes; diff={diff}"
        );
    }

    #[test]
    fn fno3d_deterministic_given_seed() {
        let m1 = make_default(13);
        let m2 = make_default(13);
        let cin = m1.cfg.in_channels;
        let voxels = m1.cfg.grid_x * m1.cfg.grid_y * m1.cfg.grid_z;
        let input: Vec<f32> = (0..(cin * voxels)).map(|i| (i as f32) * 0.01).collect();
        let o1 = m1
            .forward(&input)
            .expect("Fno3d forward pass should be deterministic");
        let o2 = m2
            .forward(&input)
            .expect("Fno3d forward pass should be deterministic");
        for i in 0..o1.len() {
            assert!(
                (o1[i] - o2[i]).abs() < 1e-8,
                "Determinism failed at {i}: {} vs {}",
                o1[i],
                o2[i]
            );
        }
    }

    #[test]
    fn fno3d_mode_truncation_filters_high_frequency() {
        // Modes (1,1,1) → only the DC (constant) mode survives. A pure
        // cos(2π · ix / nx) input has zero mean along the x axis, so its
        // 3D DFT has no DC component and the spectral path must produce
        // (numerically) zero. The residual is also zeroed for this test
        // so the entire forward should be ≈ 0.
        let mut rng = LcgRng::new(17);
        let cfg = Fno3dConfig {
            in_channels: 1,
            out_channels: 1,
            modes_x: 1,
            modes_y: 1,
            modes_z: 1,
            grid_x: 4,
            grid_y: 4,
            grid_z: 4,
        };
        let mut m =
            Fno3d::new(cfg, &mut rng).expect("Fno3d construction with valid params should succeed");
        m.zero_residual();

        let nx = m.cfg.grid_x;
        let ny = m.cfg.grid_y;
        let nz = m.cfg.grid_z;
        let voxels = nx * ny * nz;
        let mut input = vec![0.0_f32; voxels];
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let idx = (ix * ny + iy) * nz + iz;
                    input[idx] = (2.0 * std::f32::consts::PI * (ix as f32) / (nx as f32)).cos();
                }
            }
        }
        let output = m
            .forward(&input)
            .expect("Fno3d forward pass should succeed");
        for &v in &output {
            assert!(
                v.abs() < 1e-4,
                "high-freq input should be filtered: got {v}"
            );
        }
    }

    #[test]
    fn fno3d_mode_truncation_residual_still_passes_high_frequency() {
        // Same construction as the previous test but with the residual set
        // to identity. The residual path should pass the high-frequency
        // input through unchanged (or at least produce a non-zero output).
        let mut rng = LcgRng::new(18);
        let cfg = Fno3dConfig {
            in_channels: 1,
            out_channels: 1,
            modes_x: 1,
            modes_y: 1,
            modes_z: 1,
            grid_x: 4,
            grid_y: 4,
            grid_z: 4,
        };
        let mut m =
            Fno3d::new(cfg, &mut rng).expect("Fno3d construction with valid params should succeed");
        m.zero_spectral();
        m.residual_identity();

        let nx = m.cfg.grid_x;
        let ny = m.cfg.grid_y;
        let nz = m.cfg.grid_z;
        let voxels = nx * ny * nz;
        let mut input = vec![0.0_f32; voxels];
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let idx = (ix * ny + iy) * nz + iz;
                    input[idx] = (2.0 * std::f32::consts::PI * (ix as f32) / (nx as f32)).cos();
                }
            }
        }
        let output = m
            .forward(&input)
            .expect("Fno3d forward pass should succeed");
        for i in 0..voxels {
            assert!(
                (output[i] - input[i]).abs() < 1e-5,
                "residual should pass-through at {i}: {} vs {}",
                output[i],
                input[i]
            );
        }
    }

    #[test]
    fn fno3d_modes_one_extracts_volumetric_average() {
        // With spectral-identity (per-mode identity), modes (1,1,1) and
        // zero residual, the spectral path keeps only the DC component,
        // whose inverse DFT spreads the mean of the input uniformly across
        // every voxel. Therefore output ≈ mean(input) per channel everywhere.
        let mut rng = LcgRng::new(19);
        let cfg = Fno3dConfig {
            in_channels: 1,
            out_channels: 1,
            modes_x: 1,
            modes_y: 1,
            modes_z: 1,
            grid_x: 4,
            grid_y: 4,
            grid_z: 4,
        };
        let mut m =
            Fno3d::new(cfg, &mut rng).expect("Fno3d construction with valid params should succeed");
        m.spectral_identity();
        m.zero_residual();

        let voxels = m.cfg.grid_x * m.cfg.grid_y * m.cfg.grid_z;
        let input: Vec<f32> = (0..voxels).map(|i| 0.1_f32 + (i as f32) * 0.05).collect();
        let mean = input.iter().sum::<f32>() / (voxels as f32);
        let output = m
            .forward(&input)
            .expect("Fno3d forward pass should succeed");
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - mean).abs() < 1e-4,
                "DC-only spectral path → uniform mean; voxel {i}: {} vs {}",
                v,
                mean
            );
        }
    }

    #[test]
    fn fno3d_err_modes_too_large() {
        let mut rng = LcgRng::new(20);
        let cfg = Fno3dConfig {
            in_channels: 1,
            out_channels: 1,
            modes_x: 10,
            modes_y: 2,
            modes_z: 2,
            grid_x: 4,
            grid_y: 4,
            grid_z: 4,
        };
        let r = Fno3d::new(cfg, &mut rng);
        assert!(matches!(r, Err(PinnError::TooManyFourierModes { .. })));
    }

    #[test]
    fn fno3d_err_grid_zero() {
        let mut rng = LcgRng::new(21);
        let cfg = Fno3dConfig {
            in_channels: 1,
            out_channels: 1,
            modes_x: 1,
            modes_y: 1,
            modes_z: 1,
            grid_x: 4,
            grid_y: 0,
            grid_z: 4,
        };
        let r = Fno3d::new(cfg, &mut rng);
        assert!(matches!(r, Err(PinnError::InvalidGridResolution { .. })));
    }

    #[test]
    fn fno3d_err_input_wrong_length() {
        let m = make_default(22);
        let voxels = m.cfg.grid_x * m.cfg.grid_y * m.cfg.grid_z;
        let bad = vec![0.0_f32; m.cfg.in_channels * voxels + 3];
        let r = m.forward(&bad);
        assert!(matches!(r, Err(PinnError::DimensionMismatch { .. })));
    }

    #[test]
    fn fno3d_err_channels_zero() {
        let mut rng = LcgRng::new(23);
        let cfg = Fno3dConfig {
            in_channels: 0,
            out_channels: 1,
            modes_x: 1,
            modes_y: 1,
            modes_z: 1,
            grid_x: 4,
            grid_y: 4,
            grid_z: 4,
        };
        let r = Fno3d::new(cfg, &mut rng);
        assert!(matches!(r, Err(PinnError::InvalidLayerWidth)));

        let mut rng2 = LcgRng::new(24);
        let cfg2 = Fno3dConfig {
            in_channels: 1,
            out_channels: 0,
            modes_x: 1,
            modes_y: 1,
            modes_z: 1,
            grid_x: 4,
            grid_y: 4,
            grid_z: 4,
        };
        let r2 = Fno3d::new(cfg2, &mut rng2);
        assert!(matches!(r2, Err(PinnError::InvalidLayerWidth)));
    }

    #[test]
    fn fno3d_tiny_grid_2x2x2() {
        let mut rng = LcgRng::new(25);
        let cfg = Fno3dConfig {
            in_channels: 1,
            out_channels: 1,
            modes_x: 2,
            modes_y: 2,
            modes_z: 2,
            grid_x: 2,
            grid_y: 2,
            grid_z: 2,
        };
        let m =
            Fno3d::new(cfg, &mut rng).expect("Fno3d construction with valid params should succeed");
        let input: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();
        let output = m
            .forward(&input)
            .expect("Fno3d forward pass should succeed");
        assert_eq!(output.len(), 8);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn fno3d_output_finite() {
        let m = make_default(26);
        let cin = m.cfg.in_channels;
        let voxels = m.cfg.grid_x * m.cfg.grid_y * m.cfg.grid_z;
        let input: Vec<f32> = (0..(cin * voxels))
            .map(|i| ((i as f32) * 0.2).sin())
            .collect();
        let output = m
            .forward(&input)
            .expect("Fno3d forward pass should succeed");
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn fno3d_spectral_path_is_linear() {
        // With the residual zeroed, the entire forward reduces to the
        // spectral path which is linear in `x`. Verify additivity AND
        // homogeneity (scaling).
        let mut rng = LcgRng::new(27);
        let cfg = Fno3dConfig {
            in_channels: 1,
            out_channels: 1,
            modes_x: 2,
            modes_y: 2,
            modes_z: 2,
            grid_x: 4,
            grid_y: 4,
            grid_z: 4,
        };
        let mut m =
            Fno3d::new(cfg, &mut rng).expect("Fno3d construction with valid params should succeed");
        m.zero_residual();

        let voxels = m.cfg.grid_x * m.cfg.grid_y * m.cfg.grid_z;
        let a: Vec<f32> = (0..voxels).map(|i| ((i as f32) * 0.11).sin()).collect();
        let b: Vec<f32> = (0..voxels).map(|i| ((i as f32) * 0.07).cos()).collect();
        let sum: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        let scaled: Vec<f32> = a.iter().map(|x| x * 2.5).collect();

        let oa = m.forward(&a).expect("Fno3d forward pass should succeed");
        let ob = m.forward(&b).expect("Fno3d forward pass should succeed");
        let osum = m
            .forward(&sum)
            .expect("Fno3d forward on sum of inputs should succeed");
        let oscaled = m
            .forward(&scaled)
            .expect("Fno3d forward on scaled input should succeed");

        for i in 0..voxels {
            let lhs = osum[i];
            let rhs = oa[i] + ob[i];
            assert!(
                (lhs - rhs).abs() < 1e-3,
                "Additivity violated at {i}: {} vs {}",
                lhs,
                rhs
            );
            let lhs2 = oscaled[i];
            let rhs2 = oa[i] * 2.5;
            assert!(
                (lhs2 - rhs2).abs() < 1e-3,
                "Homogeneity violated at {i}: {} vs {}",
                lhs2,
                rhs2
            );
        }
    }

    #[test]
    fn fno3d_dft_dc_is_sum() {
        // For a constant volumetric input v(x,y,z) = c, the DC bin of the
        // DFT equals the volume sum c · nx · ny · nz; all other bins are 0.
        let m = make_default(28);
        let nx = m.cfg.grid_x;
        let ny = m.cfg.grid_y;
        let nz = m.cfg.grid_z;
        let total = nx * ny * nz;
        let c = 1.5_f32;
        let real = vec![c; total];
        let imag = vec![0.0_f32; total];
        let (fr, fi) = m
            .dft_3d(&real, &imag)
            .expect("3D DFT should succeed for valid input");
        let dc = c * total as f32;
        assert!((fr[0] - dc).abs() < 1e-3, "DC bin = sum, got {}", fr[0]);
        assert!(fi[0].abs() < 1e-3);
        for i in 1..total {
            assert!(
                fr[i].abs() < 1e-3,
                "non-DC real {i} should be ~0, got {}",
                fr[i]
            );
            assert!(
                fi[i].abs() < 1e-3,
                "non-DC imag {i} should be ~0, got {}",
                fi[i]
            );
        }
    }

    #[test]
    fn fno3d_multi_channel_forward_shape() {
        let mut rng = LcgRng::new(29);
        let cfg = Fno3dConfig {
            in_channels: 3,
            out_channels: 5,
            modes_x: 2,
            modes_y: 2,
            modes_z: 2,
            grid_x: 4,
            grid_y: 4,
            grid_z: 4,
        };
        let m =
            Fno3d::new(cfg, &mut rng).expect("Fno3d construction with valid params should succeed");
        let voxels = 64;
        let input = vec![0.2_f32; 3 * voxels];
        let output = m
            .forward(&input)
            .expect("Fno3d forward pass should succeed");
        assert_eq!(output.len(), 5 * voxels);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn fno3d_dft_3d_length_matches_grid() {
        let m = make_default(30);
        let total = m.cfg.grid_x * m.cfg.grid_y * m.cfg.grid_z;
        let real = vec![0.5_f32; total];
        let imag = vec![0.0_f32; total];
        let (fr, fi) = m
            .dft_3d(&real, &imag)
            .expect("3D DFT should succeed for valid input");
        assert_eq!(fr.len(), total);
        assert_eq!(fi.len(), total);
    }

    #[test]
    fn fno3d_dft_wrong_length_err() {
        let m = make_default(31);
        let bad = vec![0.0_f32; 3];
        let r = m.dft_3d(&bad, &bad);
        assert!(matches!(r, Err(PinnError::DimensionMismatch { .. })));
    }

    #[test]
    fn fno3d_idft_wrong_length_err() {
        let m = make_default(32);
        let bad = vec![0.0_f32; 3];
        let r = m.idft_3d(&bad, &bad);
        assert!(matches!(r, Err(PinnError::DimensionMismatch { .. })));
    }

    #[test]
    fn fno3d_config_accessor() {
        let m = make_default(33);
        let c = m.config();
        assert_eq!(c.in_channels, 2);
        assert_eq!(c.out_channels, 2);
        assert_eq!(c.grid_x, 4);
    }
}
