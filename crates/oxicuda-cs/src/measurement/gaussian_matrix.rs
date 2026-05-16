//! Gaussian sensing matrix `Φᵢⱼ ~ N(0, 1/m)`.

use crate::error::{CsError, CsResult};
use crate::handle::LcgRng;

/// Build a random `m × n` Gaussian sensing matrix with entries `N(0, 1/m)`.
///
/// Returns row-major flat `Vec<f64>` of length `m * n`.
pub fn gaussian_matrix(m: usize, n: usize, rng: &mut LcgRng) -> CsResult<Vec<f64>> {
    if m == 0 || n == 0 {
        return Err(CsError::EmptyInput);
    }
    let scale = 1.0 / (m as f64).sqrt();
    let mut a = vec![0.0_f64; m * n];
    for v in a.iter_mut() {
        *v = scale * rng.next_normal();
    }
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_matrix_shape() {
        let mut rng = LcgRng::new(42);
        let a = gaussian_matrix(20, 50, &mut rng).expect("ok");
        assert_eq!(a.len(), 20 * 50);
    }

    #[test]
    fn gaussian_matrix_column_variance() {
        let mut rng = LcgRng::new(7);
        let m = 200;
        let n = 8;
        let a = gaussian_matrix(m, n, &mut rng).expect("ok");
        // Column norms should be ≈ 1 since variance is 1/m and there are m entries.
        for j in 0..n {
            let mut s = 0.0_f64;
            for i in 0..m {
                s += a[i * n + j] * a[i * n + j];
            }
            assert!((s - 1.0).abs() < 0.3, "col {j} norm² = {s}");
        }
    }
}
