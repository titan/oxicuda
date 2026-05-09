//! MoCo — He et al. 2020 — Momentum Contrast with a memory queue.
//!
//! MoCo decouples the size of the negative-key set from the batch size by
//! maintaining a fixed-size FIFO queue of D-dimensional embeddings (computed
//! by a momentum-updated key encoder). The contrastive loss becomes
//!
//! ```text
//!   L = -log( exp(q·k_+/τ) / [ exp(q·k_+/τ) + Σ_{k_n ∈ Q} exp(q·k_n/τ) ] )
//! ```
//!
//! The queue is updated with the current batch of `k_+` embeddings after each
//! step, evicting the oldest entries.

use crate::error::{SslError, SslResult};

/// FIFO circular queue of D-dimensional embedding vectors.
#[derive(Debug, Clone)]
pub struct MocoQueue {
    /// Capacity (max number of stored vectors).
    pub capacity: usize,
    /// Embedding dimensionality.
    pub dim: usize,
    /// Backing storage: `[capacity × dim]` flat row-major buffer.
    pub data: Vec<f32>,
    /// Index where the next enqueue writes (mod capacity).
    pub head: usize,
    /// Number of valid entries currently in the queue (`<= capacity`).
    pub len: usize,
}

impl MocoQueue {
    /// Create an empty queue with given capacity and embedding dim.
    ///
    /// # Errors
    /// - [`SslError::QueueCapacityTooSmall`] if `capacity == 0`.
    /// - [`SslError::InvalidFeatureDim`] if `dim == 0`.
    pub fn new(capacity: usize, dim: usize) -> SslResult<Self> {
        if capacity == 0 {
            return Err(SslError::QueueCapacityTooSmall);
        }
        if dim == 0 {
            return Err(SslError::InvalidFeatureDim);
        }
        Ok(Self {
            capacity,
            dim,
            data: vec![0.0_f32; capacity * dim],
            head: 0,
            len: 0,
        })
    }

    /// Enqueue a batch of `[batch × dim]` row-major key embeddings.
    /// Old entries are evicted FIFO if the queue overflows.
    ///
    /// # Errors
    /// - [`SslError::DimensionMismatch`] when `batch.len() % self.dim != 0`.
    pub fn enqueue(&mut self, batch: &[f32]) -> SslResult<()> {
        if batch.is_empty() {
            return Ok(());
        }
        if batch.len() % self.dim != 0 {
            return Err(SslError::DimensionMismatch {
                expected: self.dim,
                got: batch.len(),
            });
        }
        let n = batch.len() / self.dim;
        for i in 0..n {
            let src = &batch[i * self.dim..(i + 1) * self.dim];
            let dst = &mut self.data[self.head * self.dim..(self.head + 1) * self.dim];
            dst.copy_from_slice(src);
            self.head = (self.head + 1) % self.capacity;
            if self.len < self.capacity {
                self.len += 1;
            }
        }
        Ok(())
    }

    /// Number of currently stored entries (≤ capacity).
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Snapshot of the currently stored entries in the order they appear in the
    /// backing buffer (i.e., **not** strictly chronological — both directions
    /// are negatives, so order does not matter).
    #[must_use]
    pub fn entries(&self) -> &[f32] {
        &self.data[..self.len * self.dim]
    }
}

/// MoCo contrastive loss for a batch of `B` query embeddings against a single
/// positive key per query plus a fixed memory bank.
///
/// `q` is `[B × D]`, `k_pos` is `[B × D]` (the *current* batch of positives),
/// `queue` is the [`MocoQueue`] of historical negatives.
///
/// All inputs should be L2-normalised on the host (we re-normalise defensively).
///
/// Returns the average per-query loss `(1/B) Σ_i L_i`.
///
/// # Errors
/// - [`SslError::DimensionMismatch`] if shapes disagree.
/// - [`SslError::EmptyInput`] if `q` or `k_pos` is empty.
/// - [`SslError::QueueEmpty`] if the queue has no negatives.
/// - [`SslError::InvalidTemperature`] if `temperature <= 0` or non-finite.
pub fn moco_loss(
    q: &[f32],
    k_pos: &[f32],
    batch: usize,
    dim: usize,
    queue: &MocoQueue,
    temperature: f32,
) -> SslResult<f32> {
    if q.is_empty() || batch == 0 || dim == 0 {
        return Err(SslError::EmptyInput);
    }
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err(SslError::InvalidTemperature { temp: temperature });
    }
    if q.len() != batch * dim {
        return Err(SslError::DimensionMismatch {
            expected: batch * dim,
            got: q.len(),
        });
    }
    if k_pos.len() != batch * dim {
        return Err(SslError::DimensionMismatch {
            expected: batch * dim,
            got: k_pos.len(),
        });
    }
    if queue.dim != dim {
        return Err(SslError::DimensionMismatch {
            expected: dim,
            got: queue.dim,
        });
    }
    if queue.is_empty() {
        return Err(SslError::QueueEmpty);
    }

    let q_n = l2_normalize_clone(q, batch, dim);
    let k_n = l2_normalize_clone(k_pos, batch, dim);
    // Negative bank is already accumulated by the caller; defensive
    // re-normalisation costs O(K·D) but improves stability.
    let neg = l2_normalize_clone(queue.entries(), queue.len(), dim);

    let inv_t = 1.0_f32 / temperature;
    let mut total_loss = 0.0_f64;
    for i in 0..batch {
        let q_row = &q_n[i * dim..(i + 1) * dim];
        let k_row = &k_n[i * dim..(i + 1) * dim];
        // Positive logit
        let mut pos = 0.0_f32;
        for (a, b) in q_row.iter().zip(k_row.iter()) {
            pos += a * b;
        }
        pos *= inv_t;
        // Negative logits
        let mut max_v = pos;
        let mut neg_logits: Vec<f32> = Vec::with_capacity(queue.len());
        for k in 0..queue.len() {
            let neg_row = &neg[k * dim..(k + 1) * dim];
            let mut s = 0.0_f32;
            for (a, b) in q_row.iter().zip(neg_row.iter()) {
                s += a * b;
            }
            let l = s * inv_t;
            if l > max_v {
                max_v = l;
            }
            neg_logits.push(l);
        }
        // log-sum-exp denom
        let mut sum_exp = ((pos - max_v) as f64).exp();
        for &l in &neg_logits {
            sum_exp += ((l - max_v) as f64).exp();
        }
        let log_z = (max_v as f64) + sum_exp.ln();
        total_loss += -((pos as f64) - log_z);
    }
    Ok((total_loss / batch as f64) as f32)
}

fn l2_normalize_clone(z: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut out = z.to_vec();
    for row in out.chunks_mut(d) {
        let s: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        let inv = if s > 1e-12 { 1.0 / s } else { 1.0 };
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
    let _ = n;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_new_zero_capacity_errors() {
        assert!(MocoQueue::new(0, 4).is_err());
    }

    #[test]
    fn queue_new_zero_dim_errors() {
        assert!(MocoQueue::new(4, 0).is_err());
    }

    #[test]
    fn queue_enqueue_grows_until_capacity() {
        let mut q = MocoQueue::new(4, 2).unwrap();
        let batch = vec![1.0_f32, 0.0, 0.0, 1.0];
        q.enqueue(&batch).unwrap();
        assert_eq!(q.len(), 2);
        q.enqueue(&batch).unwrap();
        assert_eq!(q.len(), 4);
        // Overflow: existing entries are evicted FIFO.
        q.enqueue(&[0.5_f32, 0.5]).unwrap();
        assert_eq!(q.len(), 4);
    }

    #[test]
    fn queue_enqueue_empty_batch_ok() {
        let mut q = MocoQueue::new(4, 2).unwrap();
        q.enqueue(&[]).unwrap();
        assert!(q.is_empty());
    }

    #[test]
    fn queue_enqueue_rejects_misaligned() {
        let mut q = MocoQueue::new(4, 3).unwrap();
        let r = q.enqueue(&[1.0_f32, 2.0]);
        assert!(r.is_err());
    }

    #[test]
    fn moco_loss_perfect_positives_low() {
        let mut q = MocoQueue::new(8, 4).unwrap();
        // Random negatives: small inner products with the positives.
        let mut rng = 42u64;
        let mut neg = vec![0.0_f32; 8 * 4];
        for v in neg.iter_mut() {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *v = ((rng >> 33) as f32 / (u32::MAX as f32 + 1.0)) - 0.5;
        }
        q.enqueue(&neg).unwrap();
        // Identical query and key positive → max similarity, so loss is small.
        let pos = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let loss = moco_loss(&pos, &pos, 2, 4, &q, 0.1).unwrap();
        assert!(loss.is_finite());
        assert!(loss < 1.0);
    }

    #[test]
    fn moco_loss_empty_queue_errors() {
        let q = MocoQueue::new(4, 2).unwrap();
        let pos = vec![1.0_f32, 0.0];
        let r = moco_loss(&pos, &pos, 1, 2, &q, 0.1);
        assert!(r.is_err());
    }

    #[test]
    fn moco_loss_dim_mismatch_errors() {
        let mut q = MocoQueue::new(2, 4).unwrap();
        q.enqueue(&[1.0_f32; 8]).unwrap();
        let r = moco_loss(&[1.0_f32; 4], &[1.0_f32; 4], 1, 2, &q, 0.1);
        assert!(r.is_err());
    }

    #[test]
    fn moco_loss_temperature_must_be_positive() {
        let mut q = MocoQueue::new(2, 2).unwrap();
        q.enqueue(&[1.0_f32; 4]).unwrap();
        let r = moco_loss(&[1.0_f32, 0.0], &[1.0_f32, 0.0], 1, 2, &q, 0.0);
        assert!(r.is_err());
    }

    #[test]
    fn queue_entries_view_correct_length() {
        let mut q = MocoQueue::new(4, 2).unwrap();
        q.enqueue(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let entries = q.entries();
        assert_eq!(entries.len(), 4);
    }
}
