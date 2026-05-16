//! Partial Fourier sensing matrix: randomly chosen rows of the discrete cosine transform.
//!
//! For a real-valued sensing matrix in pure Rust we use the real DCT-II basis, where row k
//! has entries `(1/√(m)) * cos(π (j + 0.5) k / n)`. We pick `m` random row indices.

use crate::error::{CsError, CsResult};
use crate::handle::LcgRng;

/// Build a `m × n` row-subsampled DCT matrix.
///
/// Rows are uniformly random samples without replacement from `0..n` and each entry follows
/// the DCT-II basis. The scaling `1/√m` makes the columns approximately unit-norm.
pub fn partial_fourier(m: usize, n: usize, rng: &mut LcgRng) -> CsResult<Vec<f64>> {
    if m == 0 || n == 0 {
        return Err(CsError::EmptyInput);
    }
    if m > n {
        return Err(CsError::InvalidParameter(format!(
            "partial Fourier requires m ≤ n; got m={m}, n={n}"
        )));
    }
    // Sample m distinct row indices via Fisher-Yates shuffle prefix.
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..m {
        let j = i + rng.next_usize(n - i);
        indices.swap(i, j);
    }
    let mut a = vec![0.0_f64; m * n];
    let scale = (2.0 / (m as f64)).sqrt();
    for (rr, &k) in indices.iter().enumerate().take(m) {
        for j in 0..n {
            let phase = std::f64::consts::PI * (j as f64 + 0.5) * (k as f64) / (n as f64);
            a[rr * n + j] = scale * phase.cos();
        }
    }
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_fourier_shape() {
        let mut rng = LcgRng::new(101);
        let a = partial_fourier(8, 32, &mut rng).expect("ok");
        assert_eq!(a.len(), 8 * 32);
    }

    #[test]
    fn partial_fourier_rejects_oversize() {
        let mut rng = LcgRng::new(1);
        assert!(partial_fourier(10, 5, &mut rng).is_err());
    }
}
