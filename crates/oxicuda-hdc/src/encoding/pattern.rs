//! Spatial/temporal pattern encoding for 2D inputs (e.g., images).
//!
//! Each pixel position (r,c) gets a position HV via `bind(row_hv[r], col_hv[c])`.
//! Pattern HV = Bundle over active pixels (pixel value > threshold).

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::ops::binding::binary_bind;
use crate::ops::bundling::bundle_binary;
use crate::vector::binary::random_binary;

/// Spatial pattern encoder for 2D inputs.
pub struct PatternEncoder {
    /// Number of rows.
    rows: usize,
    /// Number of columns.
    cols: usize,
    /// Hypervector dimension.
    dim: usize,
    /// HV per row index.
    row_hvs: Vec<Vec<i8>>,
    /// HV per column index.
    col_hvs: Vec<Vec<i8>>,
}

impl PatternEncoder {
    /// Create a new pattern encoder for inputs of shape (rows, cols).
    pub fn new(rows: usize, cols: usize, dim: usize, rng: &mut LcgRng) -> HdcResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(HdcError::EmptyInput);
        }
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        let mut row_hvs = Vec::with_capacity(rows);
        for _ in 0..rows {
            row_hvs.push(random_binary(dim, rng)?);
        }
        let mut col_hvs = Vec::with_capacity(cols);
        for _ in 0..cols {
            col_hvs.push(random_binary(dim, rng)?);
        }
        Ok(Self {
            rows,
            cols,
            dim,
            row_hvs,
            col_hvs,
        })
    }

    /// Return the position HV for pixel (r, c) = bind(row_hv[r], col_hv[c]).
    fn position_hv(&self, r: usize, c: usize) -> HdcResult<Vec<i8>> {
        binary_bind(&self.row_hvs[r], &self.col_hvs[c])
    }

    /// Encode a flat pixel array (rows*cols f32 in \[0,1\]) with given threshold.
    /// Active pixels (value > threshold) contribute their position HVs to the bundle.
    pub fn encode(&self, pixels: &[f32], threshold: f32, rng: &mut LcgRng) -> HdcResult<Vec<i8>> {
        let expected_len = self.rows * self.cols;
        if pixels.len() != expected_len {
            return Err(HdcError::DimensionMismatch {
                expected: expected_len,
                got: pixels.len(),
            });
        }
        let mut active_hvs: Vec<Vec<i8>> = Vec::new();
        for r in 0..self.rows {
            for c in 0..self.cols {
                let pixel_val = pixels[r * self.cols + c];
                if pixel_val > threshold {
                    active_hvs.push(self.position_hv(r, c)?);
                }
            }
        }
        if active_hvs.is_empty() {
            // No active pixels: return all-+1 HV as null pattern
            return Ok(vec![1i8; self.dim]);
        }
        bundle_binary(&active_hvs, rng)
    }

    /// Multi-level encoding: encode at multiple thresholds and bundle results.
    pub fn encode_multilevel(
        &self,
        pixels: &[f32],
        thresholds: &[f32],
        rng: &mut LcgRng,
    ) -> HdcResult<Vec<i8>> {
        if thresholds.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        let mut level_hvs: Vec<Vec<i8>> = Vec::with_capacity(thresholds.len());
        for &thresh in thresholds {
            level_hvs.push(self.encode(pixels, thresh, rng)?);
        }
        bundle_binary(&level_hvs, rng)
    }

    /// Return the dimension of this encoder.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return (rows, cols).
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::hamming::hamming_frac;
    use crate::handle::LcgRng;

    #[test]
    fn pattern_encoder_all_active_vs_none() {
        let mut rng = LcgRng::new(110);
        let enc = PatternEncoder::new(4, 4, 256, &mut rng).expect("new");
        let all_on = vec![1.0f32; 16];
        let all_off = vec![0.0f32; 16];
        let hv_on = enc.encode(&all_on, 0.5, &mut rng).expect("on");
        let hv_off = enc.encode(&all_off, 0.5, &mut rng).expect("off");
        let dist = hamming_frac(&hv_on, &hv_off).expect("hamming");
        // All-on and all-off should be distinguishable
        assert!(dist > 0.3, "dist={dist:.3}");
    }
}
