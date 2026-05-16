//! Restriction (fine -> coarse) and prolongation (coarse -> fine) operators.

use crate::error::{PdeError, PdeResult};

/// 1D full-weighting restriction (1/4, 1/2, 1/4).
/// `fine.len()` must be `2*coarse.len()-1`. Coarse boundary nodes copy the fine boundary.
pub fn restrict_1d(fine: &[f64]) -> PdeResult<Vec<f64>> {
    let nf = fine.len();
    if nf < 3 || nf % 2 == 0 {
        return Err(PdeError::InvalidGrid(format!(
            "restrict_1d: nf={nf} must be odd >= 3"
        )));
    }
    let nc = nf.div_ceil(2);
    let mut coarse = vec![0.0; nc];
    coarse[0] = fine[0];
    coarse[nc - 1] = fine[nf - 1];
    for (j, c_j) in coarse.iter_mut().enumerate().take(nc - 1).skip(1) {
        let i = 2 * j;
        *c_j = 0.25 * fine[i - 1] + 0.5 * fine[i] + 0.25 * fine[i + 1];
    }
    Ok(coarse)
}

/// 1D linear prolongation (coarse -> fine).
/// `fine.len()` is `2*coarse.len()-1`.
pub fn prolong_1d(coarse: &[f64]) -> PdeResult<Vec<f64>> {
    let nc = coarse.len();
    if nc < 2 {
        return Err(PdeError::InvalidGrid("prolong_1d: nc must be >= 2".into()));
    }
    let nf = 2 * nc - 1;
    let mut fine = vec![0.0; nf];
    for j in 0..nc {
        fine[2 * j] = coarse[j];
    }
    for j in 0..nc - 1 {
        fine[2 * j + 1] = 0.5 * (coarse[j] + coarse[j + 1]);
    }
    Ok(fine)
}

/// 2D full-weighting restriction.
pub fn restrict_2d(fine: &[f64], nx_f: usize, ny_f: usize) -> PdeResult<(Vec<f64>, usize, usize)> {
    if nx_f < 3 || ny_f < 3 || nx_f % 2 == 0 || ny_f % 2 == 0 {
        return Err(PdeError::InvalidGrid(format!(
            "restrict_2d requires odd nx,ny >= 3, got nx={nx_f} ny={ny_f}"
        )));
    }
    if fine.len() != nx_f * ny_f {
        return Err(PdeError::DimensionMismatch {
            a: fine.len(),
            b: nx_f * ny_f,
        });
    }
    let nx_c = nx_f.div_ceil(2);
    let ny_c = ny_f.div_ceil(2);
    let mut coarse = vec![0.0; nx_c * ny_c];
    let idx = |i: usize, j: usize| i * ny_f + j;
    for ic in 0..nx_c {
        for jc in 0..ny_c {
            let i = 2 * ic;
            let j = 2 * jc;
            if ic == 0 || jc == 0 || ic + 1 == nx_c || jc + 1 == ny_c {
                // boundary: copy
                coarse[ic * ny_c + jc] = fine[idx(i, j)];
            } else {
                let center = fine[idx(i, j)];
                let edges = fine[idx(i - 1, j)]
                    + fine[idx(i + 1, j)]
                    + fine[idx(i, j - 1)]
                    + fine[idx(i, j + 1)];
                let corners = fine[idx(i - 1, j - 1)]
                    + fine[idx(i - 1, j + 1)]
                    + fine[idx(i + 1, j - 1)]
                    + fine[idx(i + 1, j + 1)];
                coarse[ic * ny_c + jc] = (4.0 * center + 2.0 * edges + corners) / 16.0;
            }
        }
    }
    Ok((coarse, nx_c, ny_c))
}

/// 2D bilinear prolongation.
pub fn prolong_2d(coarse: &[f64], nx_c: usize, ny_c: usize) -> PdeResult<(Vec<f64>, usize, usize)> {
    if nx_c < 2 || ny_c < 2 {
        return Err(PdeError::InvalidGrid("prolong_2d needs nx,ny >= 2".into()));
    }
    if coarse.len() != nx_c * ny_c {
        return Err(PdeError::DimensionMismatch {
            a: coarse.len(),
            b: nx_c * ny_c,
        });
    }
    let nx_f = 2 * nx_c - 1;
    let ny_f = 2 * ny_c - 1;
    let mut fine = vec![0.0; nx_f * ny_f];
    let idx_f = |i: usize, j: usize| i * ny_f + j;
    let idx_c = |i: usize, j: usize| i * ny_c + j;
    // copy injection at even indices
    for ic in 0..nx_c {
        for jc in 0..ny_c {
            fine[idx_f(2 * ic, 2 * jc)] = coarse[idx_c(ic, jc)];
        }
    }
    // horizontal edges (even row, odd col)
    for ic in 0..nx_c {
        for jc in 0..ny_c - 1 {
            fine[idx_f(2 * ic, 2 * jc + 1)] =
                0.5 * (coarse[idx_c(ic, jc)] + coarse[idx_c(ic, jc + 1)]);
        }
    }
    // vertical edges (odd row, even col)
    for ic in 0..nx_c - 1 {
        for jc in 0..ny_c {
            fine[idx_f(2 * ic + 1, 2 * jc)] =
                0.5 * (coarse[idx_c(ic, jc)] + coarse[idx_c(ic + 1, jc)]);
        }
    }
    // centers (odd row, odd col)
    for ic in 0..nx_c - 1 {
        for jc in 0..ny_c - 1 {
            fine[idx_f(2 * ic + 1, 2 * jc + 1)] = 0.25
                * (coarse[idx_c(ic, jc)]
                    + coarse[idx_c(ic + 1, jc)]
                    + coarse[idx_c(ic, jc + 1)]
                    + coarse[idx_c(ic + 1, jc + 1)]);
        }
    }
    Ok((fine, nx_f, ny_f))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restrict_1d_basic() {
        let fine = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let coarse = restrict_1d(&fine).expect("ok");
        // coarse[0] = fine[0] = 0
        // coarse[1] = 0.25*fine[1] + 0.5*fine[2] + 0.25*fine[3] = 0.25 + 1.0 + 0.75 = 2.0
        // coarse[2] = fine[4] = 4
        assert_eq!(coarse.len(), 3);
        assert!((coarse[0] - 0.0).abs() < 1e-12);
        assert!((coarse[1] - 2.0).abs() < 1e-12);
        assert!((coarse[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn prolong_1d_basic() {
        let coarse = vec![0.0, 2.0, 4.0];
        let fine = prolong_1d(&coarse).expect("ok");
        assert_eq!(fine.len(), 5);
        assert!((fine[0] - 0.0).abs() < 1e-12);
        assert!((fine[1] - 1.0).abs() < 1e-12);
        assert!((fine[2] - 2.0).abs() < 1e-12);
        assert!((fine[3] - 3.0).abs() < 1e-12);
        assert!((fine[4] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn restrict_prolong_1d_constant_invariant() {
        let fine = vec![3.7; 9];
        let coarse = restrict_1d(&fine).expect("ok");
        for v in &coarse {
            assert!((v - 3.7).abs() < 1e-12);
        }
        let fine_back = prolong_1d(&coarse).expect("ok");
        for v in &fine_back {
            assert!((v - 3.7).abs() < 1e-12);
        }
    }

    #[test]
    fn restrict_2d_constant_invariant() {
        let nx = 5;
        let ny = 5;
        let fine = vec![2.5; nx * ny];
        let (coarse, nx_c, ny_c) = restrict_2d(&fine, nx, ny).expect("ok");
        assert_eq!(nx_c, 3);
        assert_eq!(ny_c, 3);
        for v in &coarse {
            assert!((v - 2.5).abs() < 1e-12);
        }
    }

    #[test]
    fn prolong_2d_constant_invariant() {
        let nx_c = 3;
        let ny_c = 3;
        let coarse = vec![1.5; nx_c * ny_c];
        let (fine, nx_f, ny_f) = prolong_2d(&coarse, nx_c, ny_c).expect("ok");
        assert_eq!(nx_f, 5);
        assert_eq!(ny_f, 5);
        for v in &fine {
            assert!((v - 1.5).abs() < 1e-12);
        }
    }
}
