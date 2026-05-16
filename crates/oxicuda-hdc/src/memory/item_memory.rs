//! Item memory: associative store mapping symbol IDs to binary hypervectors.
//! Lookup returns the nearest neighbor by Hamming distance (binary dot product proxy).

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::vector::binary;

/// Item memory: associates integer keys (symbol IDs) with binary hypervectors.
pub struct ItemMemory {
    dim: usize,
    items: Vec<(usize, Vec<i8>)>, // (symbol_id, HV)
}

impl ItemMemory {
    /// Create a new item memory for hypervectors of the given dimension.
    pub fn new(dim: usize) -> HdcResult<Self> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        Ok(Self {
            dim,
            items: Vec::new(),
        })
    }

    /// Add a symbol with its hypervector.
    pub fn add(&mut self, id: usize, hv: Vec<i8>) -> HdcResult<()> {
        if hv.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: hv.len(),
            });
        }
        binary::validate_binary(&hv)?;
        self.items.push((id, hv));
        Ok(())
    }

    /// Get the HV for a symbol ID (returns reference).
    pub fn get(&self, id: usize) -> HdcResult<&[i8]> {
        for (sid, hv) in &self.items {
            if *sid == id {
                return Ok(hv.as_slice());
            }
        }
        Err(HdcError::ItemNotFound(id))
    }

    /// Check if a symbol ID is present.
    pub fn contains(&self, id: usize) -> bool {
        self.items.iter().any(|(sid, _)| *sid == id)
    }

    /// Number of items stored.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// True if no items are stored.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Nearest-neighbor query by Hamming distance (binary dot product proxy).
    /// Returns the symbol_id whose HV has the maximum dot product with the query.
    pub fn query(&self, hv: &[i8]) -> HdcResult<usize> {
        if self.items.is_empty() {
            return Err(HdcError::EmptyItemMemory);
        }
        if hv.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: hv.len(),
            });
        }
        let mut best_id = self.items[0].0;
        let mut best_dot = i64::MIN;
        for (sid, stored_hv) in &self.items {
            let dot = binary::binary_dot(hv, stored_hv)?;
            if dot > best_dot {
                best_dot = dot;
                best_id = *sid;
            }
        }
        Ok(best_id)
    }

    /// Generate and store a random binary HV for a new symbol.
    pub fn add_random(&mut self, id: usize, rng: &mut LcgRng) -> HdcResult<()> {
        let hv = binary::random_binary(self.dim, rng)?;
        self.items.push((id, hv));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn item_memory_query_exact_match() {
        let mut rng = LcgRng::new(50);
        let mut mem = ItemMemory::new(256).expect("new");
        for id in 0..5 {
            mem.add_random(id, &mut rng).expect("add_random");
        }
        let hv = mem.get(2).expect("get").to_vec();
        let found = mem.query(&hv).expect("query");
        assert_eq!(found, 2);
    }
}
