//! Experience Replay (ER) with reservoir sampling.
//!
//! Maintains a fixed-size replay buffer of past experiences sampled uniformly
//! at random using reservoir sampling, ensuring unbiased coverage of the
//! full data stream seen so far.

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

/// Fixed-capacity experience replay buffer.
#[derive(Debug, Clone)]
pub struct ErBuffer {
    /// Stored feature vectors.
    pub data: Vec<Vec<f32>>,
    /// Corresponding class labels.
    pub labels: Vec<u32>,
    /// Maximum number of samples the buffer can hold.
    pub capacity: usize,
    /// Total number of samples seen (including those not in buffer).
    pub n_seen: usize,
}

/// Create a new empty experience replay buffer.
pub fn er_buffer_new(capacity: usize) -> ContinualResult<ErBuffer> {
    if capacity == 0 {
        return Err(ContinualError::BufferCapacityTooSmall);
    }
    Ok(ErBuffer {
        data: Vec::with_capacity(capacity),
        labels: Vec::with_capacity(capacity),
        capacity,
        n_seen: 0,
    })
}

/// Add a sample to the buffer using reservoir sampling.
///
/// - If `n_seen < capacity`: always insert (buffer not yet full).
/// - Otherwise: replace a random existing slot with probability `capacity / (n_seen + 1)`.
pub fn er_add(buf: &mut ErBuffer, sample: Vec<f32>, label: u32, rng: &mut LcgRng) {
    let n = buf.n_seen;
    if n < buf.capacity {
        // Buffer not yet full: just append
        buf.data.push(sample);
        buf.labels.push(label);
    } else {
        // Reservoir sampling: replace slot r with probability cap/(n+1)
        let r = rng.next_usize(n + 1);
        if r < buf.capacity {
            buf.data[r] = sample;
            buf.labels[r] = label;
        }
    }
    buf.n_seen += 1;
}

/// Sample a random mini-batch of `n` items from the buffer without replacement.
///
/// Returns `(data_batch, labels_batch)`. Returns `Err` if the buffer is empty
/// or `n` exceeds the current buffer size.
pub fn er_sample_batch(
    buf: &ErBuffer,
    n: usize,
    rng: &mut LcgRng,
) -> ContinualResult<(Vec<Vec<f32>>, Vec<u32>)> {
    let buf_size = buf.data.len();
    if buf_size == 0 {
        return Err(ContinualError::BufferEmpty);
    }
    if n > buf_size {
        return Err(ContinualError::BatchExceedsBuffer {
            requested: n,
            available: buf_size,
        });
    }
    // Fisher-Yates partial shuffle to sample without replacement
    let mut indices: Vec<usize> = (0..buf_size).collect();
    for i in 0..n {
        let j = i + rng.next_usize(buf_size - i);
        indices.swap(i, j);
    }
    let data_batch = indices[..n].iter().map(|&i| buf.data[i].clone()).collect();
    let labels_batch = indices[..n].iter().map(|&i| buf.labels[i]).collect();
    Ok((data_batch, labels_batch))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn er_buffer_bounded_by_capacity() {
        let mut rng = LcgRng::new(42);
        let mut buf = er_buffer_new(10).expect("ER buffer should initialize with valid capacity");
        for i in 0..50_usize {
            er_add(&mut buf, vec![i as f32], i as u32, &mut rng);
        }
        assert_eq!(buf.data.len(), 10, "Buffer must not exceed capacity");
        assert_eq!(buf.n_seen, 50);
    }

    #[test]
    fn er_buffer_fills_before_capacity() {
        let mut rng = LcgRng::new(7);
        let mut buf = er_buffer_new(20).expect("should succeed with valid test inputs");
        for i in 0..15_usize {
            er_add(&mut buf, vec![i as f32], i as u32, &mut rng);
        }
        assert_eq!(buf.data.len(), 15);
    }

    #[test]
    fn er_sample_batch_size_respected() {
        let mut rng = LcgRng::new(13);
        let mut buf = er_buffer_new(50).expect("should succeed with valid test inputs");
        for i in 0..30_usize {
            er_add(&mut buf, vec![i as f32], i as u32, &mut rng);
        }
        let (batch, labels) =
            er_sample_batch(&buf, 8, &mut rng).expect("should succeed with valid test inputs");
        assert_eq!(batch.len(), 8);
        assert_eq!(labels.len(), 8);
    }

    #[test]
    fn er_sample_no_duplicates_small_buffer() {
        let mut rng = LcgRng::new(99);
        let mut buf = er_buffer_new(5).expect("should succeed with valid test inputs");
        for i in 0..5_usize {
            er_add(&mut buf, vec![i as f32], i as u32, &mut rng);
        }
        let (batch, _) =
            er_sample_batch(&buf, 5, &mut rng).expect("should succeed with valid test inputs");
        let mut seen = batch.iter().map(|v| v[0] as usize).collect::<Vec<_>>();
        seen.sort_unstable();
        assert_eq!(
            seen,
            vec![0, 1, 2, 3, 4],
            "All 5 samples should be included"
        );
    }

    #[test]
    fn er_sample_empty_returns_err() {
        let mut rng = LcgRng::new(1);
        let buf = er_buffer_new(10).expect("should succeed with valid test inputs");
        assert!(er_sample_batch(&buf, 1, &mut rng).is_err());
    }

    #[test]
    fn er_sample_exceeds_buffer_returns_err() {
        let mut rng = LcgRng::new(2);
        let mut buf = er_buffer_new(10).expect("should succeed with valid test inputs");
        er_add(&mut buf, vec![1.0], 0, &mut rng);
        assert!(er_sample_batch(&buf, 5, &mut rng).is_err());
    }

    #[test]
    fn er_capacity_zero_returns_err() {
        assert!(er_buffer_new(0).is_err());
    }

    #[test]
    fn er_reservoir_distributes_approximately() {
        // After filling with 100 items into capacity-10 buffer,
        // each original slot should be replaced with probability ~10/100.
        // We just verify the buffer contains valid indices.
        let mut rng = LcgRng::new(55);
        let mut buf = er_buffer_new(10).expect("should succeed with valid test inputs");
        for i in 0..100_usize {
            er_add(&mut buf, vec![i as f32], i as u32, &mut rng);
        }
        // All stored values should be valid sample indices
        for val in &buf.data {
            let idx = val[0] as usize;
            assert!(idx < 100, "Stored index must be valid");
        }
    }
}
