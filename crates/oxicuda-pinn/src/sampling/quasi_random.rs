//! Quasi-random sequences for low-discrepancy sampling.

use crate::error::{PinnError, PinnResult};

/// First 10 prime numbers (bases for Halton sequence).
const PRIMES: &[u32] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29];

/// Radical inverse in base `b` for index `i`.
///
/// `halton(i, b) = Σ d_k · b^{-(k+1)}` where `d_k` are the digits of `i` in base `b`.
pub fn halton(i: usize, base: u32) -> f32 {
    let b = base as f64;
    let mut f = 1.0_f64;
    let mut r = 0.0_f64;
    let mut n = i;
    while n > 0 {
        f /= b;
        r += f * (n % base as usize) as f64;
        n /= base as usize;
    }
    r as f32
}

/// Multi-dimensional Halton sequence.
///
/// Uses the first `d` primes as bases. Returns a flat `[n × d]` array.
pub fn halton_sequence(n: usize, d: usize) -> PinnResult<Vec<f32>> {
    if d > PRIMES.len() {
        return Err(PinnError::DimensionMismatch {
            expected: PRIMES.len(),
            got: d,
        });
    }
    let mut out = vec![0.0_f32; n * d];
    for i in 0..n {
        for j in 0..d {
            out[i * d + j] = halton(i + 1, PRIMES[j]); // start at i+1 to avoid 0
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halton_base2_known_values() {
        assert!((halton(0, 2) - 0.0).abs() < 1e-7, "halton(0,2) should be 0");
        assert!(
            (halton(1, 2) - 0.5).abs() < 1e-7,
            "halton(1,2) should be 0.5"
        );
        assert!(
            (halton(2, 2) - 0.25).abs() < 1e-7,
            "halton(2,2) should be 0.25"
        );
        assert!(
            (halton(3, 2) - 0.75).abs() < 1e-7,
            "halton(3,2) should be 0.75"
        );
    }

    #[test]
    fn halton_base3_known_values() {
        assert!(
            (halton(1, 3) - 1.0 / 3.0).abs() < 1e-6,
            "halton(1,3) should be 1/3"
        );
        assert!(
            (halton(2, 3) - 2.0 / 3.0).abs() < 1e-6,
            "halton(2,3) should be 2/3"
        );
    }

    #[test]
    fn halton_in_unit_interval() {
        for i in 0..100 {
            let v = halton(i, 2);
            assert!(
                (0.0..=1.0).contains(&v),
                "halton({i}, 2) = {v} not in [0,1]"
            );
        }
    }

    #[test]
    fn halton_sequence_shape() {
        let s = halton_sequence(20, 3).expect(
            "halton_sequence with 3 dimensions is within the 10-prime table and should succeed",
        );
        assert_eq!(s.len(), 60);
    }

    #[test]
    fn halton_sequence_in_unit_cube() {
        let s = halton_sequence(50, 5).expect(
            "halton_sequence with 5 dimensions is within the 10-prime table and should succeed",
        );
        for &v in &s {
            assert!((0.0..=1.0).contains(&v), "Halton point {v} not in [0,1]");
        }
    }

    #[test]
    fn halton_sequence_too_many_dims_error() {
        let result = halton_sequence(10, 11);
        assert!(result.is_err());
    }

    #[test]
    fn halton_low_discrepancy_2d() {
        // 100 points in 2D: verify they cover the space reasonably
        let s = halton_sequence(100, 2).expect(
            "halton_sequence with 2 dimensions should succeed for low-discrepancy coverage test",
        );
        // Check that no 0.1×0.1 cell has more than 4 points (expected ~1)
        let n_cells = 10;
        let mut counts = vec![0_usize; n_cells * n_cells];
        for i in 0..100 {
            let x = s[i * 2];
            let y = s[i * 2 + 1];
            let ix = (x * n_cells as f32).min(n_cells as f32 - 1.0) as usize;
            let iy = (y * n_cells as f32).min(n_cells as f32 - 1.0) as usize;
            counts[ix * n_cells + iy] += 1;
        }
        let max_count = *counts
            .iter()
            .max()
            .expect("cell counts vec has 100 elements so max() always returns Some");
        assert!(
            max_count <= 4,
            "Max cell count too high: {max_count} (low-discrepancy should spread evenly)"
        );
    }
}
