//! Associative memory (Hopfield-style content-addressable).
//!
//! Encoding: M = Σ_i bind(key_i, value_i)
//! Retrieval: value_i ≈ unbind(query_key_i, M), then query item memory.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::ops::binding::binary_bind;
use crate::vector::binary::threshold_binary;

/// Associative memory storing (key_hv, value_hv) pairs as HDC records.
pub struct AssocMemory {
    dim: usize,
    /// Thresholded binary memory HV (updated after finalize()).
    memory_hv: Vec<i8>,
    /// Accumulator for superposition of bound pairs.
    memory_acc: Vec<i32>,
    n_stored: usize,
}

impl AssocMemory {
    /// Create a new associative memory of the given dimension.
    pub fn new(dim: usize) -> HdcResult<Self> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        Ok(Self {
            dim,
            memory_hv: vec![1i8; dim],
            memory_acc: vec![0i32; dim],
            n_stored: 0,
        })
    }

    /// Store a (key, value) pair by binding them and adding to the superposition accumulator.
    pub fn store(&mut self, key: &[i8], value: &[i8]) -> HdcResult<()> {
        if key.len() != self.dim {
            return Err(HdcError::AssocDimensionMismatch);
        }
        if value.len() != self.dim {
            return Err(HdcError::AssocDimensionMismatch);
        }
        let bound = binary_bind(key, value)?;
        for (a, &v) in self.memory_acc.iter_mut().zip(bound.iter()) {
            *a += v as i32;
        }
        self.n_stored += 1;
        Ok(())
    }

    /// Threshold the accumulator into a binary memory HV (must call after all stores).
    pub fn finalize(&mut self, rng: &mut LcgRng) -> HdcResult<()> {
        self.memory_hv = threshold_binary(&self.memory_acc, rng)?;
        Ok(())
    }

    /// Retrieve: unbind the memory with the query key to get an approximate value HV.
    /// Returns the raw unbound HV (caller should query item memory to decode).
    pub fn retrieve(&self, key: &[i8]) -> HdcResult<Vec<i8>> {
        if key.len() != self.dim {
            return Err(HdcError::AssocDimensionMismatch);
        }
        // unbind = binary_bind(key, memory_hv) since bind is self-inverse in ±1 domain
        binary_bind(key, &self.memory_hv)
    }

    /// Theoretical capacity estimate: ≈ D / (2 * ln(2)).
    pub fn capacity_estimate(&self) -> usize {
        let cap = self.dim as f64 / (2.0 * std::f64::consts::LN_2);
        cap as usize
    }

    /// Number of (key, value) pairs stored.
    pub fn n_stored(&self) -> usize {
        self.n_stored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::vector::binary::{binary_dot, random_binary};

    #[test]
    fn assoc_memory_single_pair_retrieval() {
        let mut rng = LcgRng::new(60);
        let dim = 512;
        let key = random_binary(dim, &mut rng).expect("key");
        let val = random_binary(dim, &mut rng).expect("val");

        let mut mem = AssocMemory::new(dim).expect("new");
        mem.store(&key, &val).expect("store");
        mem.finalize(&mut rng).expect("finalize");

        let retrieved = mem.retrieve(&key).expect("retrieve");
        // With a single pair and no noise, retrieve should match val exactly.
        let dot = binary_dot(&retrieved, &val).expect("dot");
        let frac = dot.abs() as f64 / dim as f64;
        assert!(frac > 0.9, "retrieval correlation too low: {frac:.3}");
    }
}
