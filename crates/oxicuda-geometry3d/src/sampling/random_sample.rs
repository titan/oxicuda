//! Uniform random sampling without replacement.

use crate::error::{Geom3dError, Geom3dResult};
use crate::handle::LcgRng;

/// Uniform random sampling without replacement via partial Fisher-Yates.
///
/// Builds `indices: Vec<usize> = (0..n).collect()`. For i in 0..m:
/// swap `indices[i]` with `indices[i + rng.next_usize(n-i)]`.
/// Returns `indices[..m]`.
pub fn random_sample(n: usize, m: usize, rng: &mut LcgRng) -> Geom3dResult<Vec<usize>> {
    if n == 0 {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if m > n {
        return Err(Geom3dError::InvalidSampleCount {
            requested: m,
            available: n,
        });
    }
    if m == 0 {
        return Ok(Vec::new());
    }

    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..m {
        let j = i + rng.next_usize(n - i);
        indices.swap(i, j);
    }
    Ok(indices[..m].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_sample_empty_error() {
        let mut rng = LcgRng::new(0);
        assert_eq!(
            random_sample(0, 1, &mut rng),
            Err(Geom3dError::EmptyPointCloud)
        );
    }

    #[test]
    fn random_sample_m_exceeds_n() {
        let mut rng = LcgRng::new(0);
        assert_eq!(
            random_sample(5, 6, &mut rng),
            Err(Geom3dError::InvalidSampleCount {
                requested: 6,
                available: 5
            })
        );
    }

    #[test]
    fn random_sample_m_zero_returns_empty() {
        let mut rng = LcgRng::new(0);
        let r = random_sample(5, 0, &mut rng).expect("random_sample should succeed");
        assert!(r.is_empty());
    }

    #[test]
    fn random_sample_correct_count() {
        let mut rng = LcgRng::new(42);
        let r = random_sample(100, 20, &mut rng).expect("random_sample should succeed");
        assert_eq!(r.len(), 20);
    }

    #[test]
    fn random_sample_all_distinct() {
        let mut rng = LcgRng::new(42);
        let r = random_sample(100, 50, &mut rng).expect("random_sample should succeed");
        let mut sorted = r.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 50);
    }

    #[test]
    fn random_sample_indices_in_range() {
        let mut rng = LcgRng::new(7);
        let r = random_sample(50, 25, &mut rng).expect("random_sample should succeed");
        assert!(r.iter().all(|&i| i < 50));
    }

    #[test]
    fn random_sample_all_n_when_m_equals_n() {
        let mut rng = LcgRng::new(1);
        let r = random_sample(10, 10, &mut rng).expect("random_sample should succeed");
        assert_eq!(r.len(), 10);
        let mut sorted = r.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }
}
