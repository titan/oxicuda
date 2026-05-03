//! Secure aggregation protocol coordinator.
//!
//! Manages the server-side state for collecting masked client updates
//! and finalising the aggregated result (which equals the plaintext
//! sum when all clients participate and their pairwise masks cancel).

use crate::error::{FedError, FedResult};

/// Server-side state for the secure aggregation protocol.
#[derive(Debug, Clone)]
pub struct SecureAggregator {
    /// Running sum of masked client updates (u32 arithmetic mod 2^32).
    accumulated: Vec<u32>,
    /// Number of client updates added.
    n_updates: usize,
    /// Expected number of elements per update.
    n_params: usize,
}

impl SecureAggregator {
    /// Create a new SecureAggregator for `n_params` parameters.
    #[must_use]
    pub fn new(n_params: usize) -> Self {
        Self {
            accumulated: vec![0u32; n_params],
            n_updates: 0,
            n_params,
        }
    }

    /// Add a masked client update to the running sum.
    ///
    /// Performs element-wise modular addition (mod 2^32) of the masked update.
    ///
    /// # Errors
    /// Returns `DimensionMismatch` if `masked_update` has wrong length.
    pub fn add_masked_update(&mut self, masked_update: &[u32]) -> FedResult<()> {
        if masked_update.len() != self.n_params {
            return Err(FedError::DimensionMismatch {
                expected: self.n_params,
                got: masked_update.len(),
            });
        }
        for (acc, &val) in self.accumulated.iter_mut().zip(masked_update.iter()) {
            *acc = acc.wrapping_add(val);
        }
        self.n_updates += 1;
        Ok(())
    }

    /// Finalise the aggregation, returning the accumulated sum as raw u32 bits.
    ///
    /// When all n parties have participated and their pairwise masks cancel,
    /// `sum_bits[i] = f32::to_bits(Σ update_i[i])` (bitwise, mod 2^32).
    /// For the floating-point sum to be recoverable, the calling code must
    /// know how to interpret these bits (typically the sum is in f32 bit space).
    ///
    /// # Errors
    /// Returns `InsufficientClients` if no updates were added.
    pub fn finalize_raw(&self) -> FedResult<Vec<u32>> {
        if self.n_updates == 0 {
            return Err(FedError::InsufficientClients { min: 1, got: 0 });
        }
        Ok(self.accumulated.clone())
    }

    /// Finalise and interpret the sum as f32 values.
    ///
    /// This works correctly only when all pairwise masks have cancelled,
    /// i.e. every party whose mask was added has also subtracted their mask.
    ///
    /// # Errors
    /// Returns `InsufficientClients` if no updates were added.
    pub fn finalize(&self) -> FedResult<Vec<f32>> {
        let raw = self.finalize_raw()?;
        Ok(raw.iter().map(|&b| f32::from_bits(b)).collect())
    }

    /// Reset the aggregator for the next round.
    pub fn reset(&mut self) {
        self.accumulated.fill(0);
        self.n_updates = 0;
    }

    /// Return the number of updates received so far.
    #[must_use]
    pub fn n_updates(&self) -> usize {
        self.n_updates
    }

    /// Return the expected number of parameters.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.n_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secure_aggregator_new() {
        let agg = SecureAggregator::new(5);
        assert_eq!(agg.n_params(), 5);
        assert_eq!(agg.n_updates(), 0);
    }

    #[test]
    fn secure_aggregator_finalize_empty_error() {
        let agg = SecureAggregator::new(4);
        assert!(matches!(
            agg.finalize_raw(),
            Err(FedError::InsufficientClients { .. })
        ));
    }

    #[test]
    fn secure_aggregator_add_and_finalize() {
        let mut agg = SecureAggregator::new(3);
        let u1: Vec<u32> = [1.0f32, 2.0, 3.0].iter().map(|v| v.to_bits()).collect();
        agg.add_masked_update(&u1)
            .expect("test invariant: valid add");
        assert_eq!(agg.n_updates(), 1);
        let raw = agg.finalize_raw().expect("test invariant: valid finalize");
        assert_eq!(raw.len(), 3);
    }

    #[test]
    fn secure_aggregator_dimension_mismatch() {
        let mut agg = SecureAggregator::new(3);
        let update = vec![0u32; 5]; // wrong size
        assert!(matches!(
            agg.add_masked_update(&update),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn secure_aggregator_reset() {
        let mut agg = SecureAggregator::new(2);
        let update = vec![100u32, 200];
        agg.add_masked_update(&update)
            .expect("test invariant: valid add");
        agg.reset();
        assert_eq!(agg.n_updates(), 0);
        assert!(matches!(
            agg.finalize_raw(),
            Err(FedError::InsufficientClients { .. })
        ));
    }
}
