//! Two-dimensional (image / grid) encoding for hyperdimensional computing.
//!
//! A 2-D grid of discrete cell values is encoded into a single binary hypervector that records
//! *what* value sits at *which* position, so that two grids differing by a spatial shift or a
//! transpose map to distinct hypervectors. This is the standard VSA construction for images and
//! spatial sensor maps (Kleyko 2018; Hassan 2021): every cell is bound to a position
//! hypervector and all bound cells are superposed.
//!
//! - **Positions.** A random base row hypervector and a random base column hypervector are
//!   generated; the position hypervector of row `r` is the circular shift `ρ^{r}` of the base
//!   row hypervector and likewise `ρ^{c}` for column `c`. Distinct rows (columns) are therefore
//!   (nearly) orthogonal, and the joint position of cell `(r, c)` is `ρ^{r}(row) ⊗ ρ^{c}(col)`.
//!   Because rows and columns use *different* base hypervectors, transposing the grid changes
//!   every off-diagonal cell's position binding.
//!
//! - **Cells.** Each distinct cell value owns a random hypervector. Cell `(r, c)` with value
//!   `v` contributes `value(v) ⊗ ρ^{r}(row) ⊗ ρ^{c}(col)`, and the grid encoding is the binary
//!   majority bundle of all cell contributions.
//!
//! All hypervectors are `Vec<i8>` in `{−1, +1}`, matching the crate-wide binary representation.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::ops::binding::binary_bind;
use crate::ops::bundling::bundle_binary;
use crate::ops::permutation::cyclic_shift;
use crate::vector::binary::random_binary;

/// Encoder mapping a fixed-shape 2-D grid of discrete values into a binary hypervector.
pub struct SpatialHdEncoder {
    /// Hypervector dimension.
    dim: usize,
    /// Number of grid rows.
    n_rows: usize,
    /// Number of grid columns.
    n_cols: usize,
    /// Number of distinct cell values.
    n_values: usize,
    /// Position hypervector per row (`ρ^{r}` of the base row hypervector).
    row_hvs: Vec<Vec<i8>>,
    /// Position hypervector per column (`ρ^{c}` of the base column hypervector).
    col_hvs: Vec<Vec<i8>>,
    /// Random hypervector per distinct cell value.
    value_hvs: Vec<Vec<i8>>,
}

impl SpatialHdEncoder {
    /// Create an encoder for an `n_rows × n_cols` grid drawn from `n_values` distinct values.
    ///
    /// # Errors
    ///
    /// - [`HdcError::ZeroDimension`] if `dim == 0`.
    /// - [`HdcError::EmptyInput`] if `n_rows`, `n_cols`, or `n_values` is `0`.
    pub fn new(
        n_rows: usize,
        n_cols: usize,
        n_values: usize,
        dim: usize,
        rng: &mut LcgRng,
    ) -> HdcResult<Self> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        if n_rows == 0 || n_cols == 0 || n_values == 0 {
            return Err(HdcError::EmptyInput);
        }

        let base_row = random_binary(dim, rng)?;
        let base_col = random_binary(dim, rng)?;

        let mut row_hvs = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            if r == 0 {
                row_hvs.push(base_row.clone());
            } else {
                row_hvs.push(cyclic_shift(&base_row, r)?);
            }
        }
        let mut col_hvs = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            if c == 0 {
                col_hvs.push(base_col.clone());
            } else {
                col_hvs.push(cyclic_shift(&base_col, c)?);
            }
        }
        let mut value_hvs = Vec::with_capacity(n_values);
        for _ in 0..n_values {
            value_hvs.push(random_binary(dim, rng)?);
        }

        Ok(Self {
            dim,
            n_rows,
            n_cols,
            n_values,
            row_hvs,
            col_hvs,
            value_hvs,
        })
    }

    /// Hypervector dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of grid rows.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of grid columns.
    #[must_use]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Number of distinct cell values.
    #[must_use]
    pub fn n_values(&self) -> usize {
        self.n_values
    }

    /// Hypervector for a distinct cell value.
    ///
    /// # Errors
    ///
    /// - [`HdcError::FeatureIndexOutOfRange`] if `value >= n_values`.
    pub fn value_hv(&self, value: usize) -> HdcResult<&[i8]> {
        if value >= self.n_values {
            return Err(HdcError::FeatureIndexOutOfRange {
                feat: value,
                max: self.n_values,
            });
        }
        Ok(&self.value_hvs[value])
    }

    /// Joint position hypervector `ρ^{r}(row) ⊗ ρ^{c}(col)` for cell `(r, c)`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::FeatureIndexOutOfRange`] if `r >= n_rows` or `c >= n_cols`.
    pub fn position_hv(&self, r: usize, c: usize) -> HdcResult<Vec<i8>> {
        if r >= self.n_rows {
            return Err(HdcError::FeatureIndexOutOfRange {
                feat: r,
                max: self.n_rows,
            });
        }
        if c >= self.n_cols {
            return Err(HdcError::FeatureIndexOutOfRange {
                feat: c,
                max: self.n_cols,
            });
        }
        binary_bind(&self.row_hvs[r], &self.col_hvs[c])
    }

    /// Encode a row-major grid of value indices (`image.len() == n_rows * n_cols`).
    ///
    /// Cell `(r, c)` is read from `image[r * n_cols + c]` and contributes
    /// `value(image[r][c]) ⊗ ρ^{r}(row) ⊗ ρ^{c}(col)`; all contributions are bundled.
    ///
    /// `rng` is only consulted to break ties in the majority bundle.
    ///
    /// # Errors
    ///
    /// - [`HdcError::DimensionMismatch`] if `image.len() != n_rows * n_cols`.
    /// - [`HdcError::FeatureIndexOutOfRange`] if any cell value is `>= n_values`.
    pub fn encode(&self, image: &[usize], rng: &mut LcgRng) -> HdcResult<Vec<i8>> {
        let expected = self.n_rows * self.n_cols;
        if image.len() != expected {
            return Err(HdcError::DimensionMismatch {
                expected,
                got: image.len(),
            });
        }
        let mut cell_hvs: Vec<Vec<i8>> = Vec::with_capacity(expected);
        for r in 0..self.n_rows {
            for c in 0..self.n_cols {
                let value = image[r * self.n_cols + c];
                if value >= self.n_values {
                    return Err(HdcError::FeatureIndexOutOfRange {
                        feat: value,
                        max: self.n_values,
                    });
                }
                let pos = binary_bind(&self.row_hvs[r], &self.col_hvs[c])?;
                let cell = binary_bind(&self.value_hvs[value], &pos)?;
                cell_hvs.push(cell);
            }
        }
        bundle_binary(&cell_hvs, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::cosine::cosine_binary;

    #[test]
    fn shift_and_transpose_change_encoding() {
        // (a) A row-shifted grid and a transposed grid both differ from the original.
        let dim = 1024;
        let mut rng = LcgRng::new(101);
        let enc = SpatialHdEncoder::new(3, 3, 4, dim, &mut rng).expect("new");

        // Non-symmetric 3×3 grid (row-major).
        let img: Vec<usize> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0];

        // Cyclically shift rows down by one.
        let mut shifted = vec![0usize; img.len()];
        for r in 0..3 {
            for c in 0..3 {
                shifted[((r + 1) % 3) * 3 + c] = img[r * 3 + c];
            }
        }
        // Transpose.
        let mut transposed = vec![0usize; img.len()];
        for r in 0..3 {
            for c in 0..3 {
                transposed[c * 3 + r] = img[r * 3 + c];
            }
        }

        let mut r0 = LcgRng::new(1);
        let mut r1 = LcgRng::new(1);
        let mut r2 = LcgRng::new(1);
        let h = enc.encode(&img, &mut r0).expect("h");
        let hs = enc.encode(&shifted, &mut r1).expect("hs");
        let ht = enc.encode(&transposed, &mut r2).expect("ht");

        let sim_shift = cosine_binary(&h, &hs).expect("cos shift");
        let sim_trans = cosine_binary(&h, &ht).expect("cos trans");
        assert!(
            sim_shift < 0.6,
            "row shift not captured: sim={sim_shift:.3}"
        );
        assert!(
            sim_trans < 0.8,
            "transpose not captured: sim={sim_trans:.3}"
        );
    }

    #[test]
    fn identical_grid_identical_encoding() {
        // (b) Same grid → same hypervector (same-seed bundle tie-breaks).
        let dim = 512;
        let mut rng = LcgRng::new(202);
        let enc = SpatialHdEncoder::new(4, 4, 3, dim, &mut rng).expect("new");
        let img: Vec<usize> = (0..16).map(|i| i % 3).collect();
        let mut r1 = LcgRng::new(5);
        let mut r2 = LcgRng::new(5);
        let h1 = enc.encode(&img, &mut r1).expect("h1");
        let h2 = enc.encode(&img, &mut r2).expect("h2");
        assert_eq!(h1, h2);
    }

    #[test]
    fn single_pixel_roundtrips() {
        // (c) A 1×1 grid encodes to value ⊗ position(0,0); unbinding recovers the value HV.
        let dim = 512;
        let mut rng = LcgRng::new(303);
        let enc = SpatialHdEncoder::new(1, 1, 4, dim, &mut rng).expect("new");

        let v = 2usize;
        let mut r = LcgRng::new(9);
        let h = enc.encode(&[v], &mut r).expect("encode");
        assert_eq!(h.len(), dim);

        // Encoding of one cell is exactly value ⊗ position; unbind position → value HV.
        let pos = enc.position_hv(0, 0).expect("pos");
        let recovered = binary_bind(&h, &pos).expect("unbind");
        let mut best = 0usize;
        let mut best_sim = f32::NEG_INFINITY;
        for cand in 0..enc.n_values() {
            let sim = cosine_binary(&recovered, enc.value_hv(cand).expect("val")).expect("cos");
            if sim > best_sim {
                best_sim = sim;
                best = cand;
            }
        }
        assert_eq!(
            best, v,
            "single pixel value not recovered (best_sim={best_sim:.3})"
        );

        // Different value → different encoding.
        let h_other = enc
            .encode(&[0usize], &mut LcgRng::new(9))
            .expect("encode 0");
        let sim = cosine_binary(&h, &h_other).expect("cos");
        assert!(sim < 0.5, "distinct values should differ: sim={sim:.3}");
    }

    #[test]
    fn out_of_range_dims_and_values_error() {
        // (d) Wrong grid length and out-of-range values are rejected.
        let dim = 256;
        let mut rng = LcgRng::new(404);
        let enc = SpatialHdEncoder::new(2, 3, 4, dim, &mut rng).expect("new");

        // image length must equal 2*3 = 6.
        let mut r = LcgRng::new(1);
        assert!(matches!(
            enc.encode(&[0, 1, 2], &mut r),
            Err(HdcError::DimensionMismatch {
                expected: 6,
                got: 3
            })
        ));
        // value 9 ≥ n_values (4).
        assert!(matches!(
            enc.encode(&[0, 1, 2, 3, 9, 0], &mut r),
            Err(HdcError::FeatureIndexOutOfRange { feat: 9, max: 4 })
        ));
        // accessor bounds.
        assert!(matches!(
            enc.value_hv(4),
            Err(HdcError::FeatureIndexOutOfRange { feat: 4, max: 4 })
        ));
        assert!(matches!(
            enc.position_hv(2, 0),
            Err(HdcError::FeatureIndexOutOfRange { feat: 2, max: 2 })
        ));
        assert!(matches!(
            enc.position_hv(0, 3),
            Err(HdcError::FeatureIndexOutOfRange { feat: 3, max: 3 })
        ));
    }

    #[test]
    fn constructor_rejects_bad_args() {
        let mut rng = LcgRng::new(505);
        assert!(matches!(
            SpatialHdEncoder::new(2, 2, 2, 0, &mut rng),
            Err(HdcError::ZeroDimension)
        ));
        assert!(matches!(
            SpatialHdEncoder::new(0, 2, 2, 64, &mut rng),
            Err(HdcError::EmptyInput)
        ));
        assert!(matches!(
            SpatialHdEncoder::new(2, 0, 2, 64, &mut rng),
            Err(HdcError::EmptyInput)
        ));
        assert!(matches!(
            SpatialHdEncoder::new(2, 2, 0, 64, &mut rng),
            Err(HdcError::EmptyInput)
        ));
    }

    #[test]
    fn dimension_consistency() {
        // (e) Output and component hypervectors all have length `dim`.
        let dim = 768;
        let mut rng = LcgRng::new(606);
        let enc = SpatialHdEncoder::new(3, 2, 5, dim, &mut rng).expect("new");
        assert_eq!(enc.dim(), dim);
        assert_eq!(enc.n_rows(), 3);
        assert_eq!(enc.n_cols(), 2);
        assert_eq!(enc.n_values(), 5);
        assert_eq!(enc.value_hv(0).expect("val").len(), dim);
        assert_eq!(enc.position_hv(2, 1).expect("pos").len(), dim);
        let img: Vec<usize> = vec![0, 1, 2, 3, 4, 0];
        let h = enc.encode(&img, &mut LcgRng::new(1)).expect("encode");
        assert_eq!(h.len(), dim);
    }
}
