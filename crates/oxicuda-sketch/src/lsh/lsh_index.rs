//! Generic LSH index: bucket-and-probe API for approximate nearest neighbour queries.
//!
//! Stores items keyed by their banded signatures; on query, looks up all candidate items
//! sharing at least one band key.

use crate::error::SketchResult;
use std::collections::BTreeMap;

/// LSH index over a generic `Item` type. Each item is associated with a slice of band keys.
#[derive(Debug, Clone)]
pub struct LshIndex {
    pub b: usize,
    /// Map from (band_index, band_key) -> list of item ids.
    pub buckets: BTreeMap<(u32, u64), Vec<u64>>,
}

impl LshIndex {
    /// Construct an empty index with `b` bands.
    #[must_use]
    pub fn new(b: usize) -> Self {
        Self {
            b,
            buckets: BTreeMap::new(),
        }
    }

    /// Insert an item id with its band keys (must have length `b`).
    pub fn insert(&mut self, item_id: u64, band_keys: &[u64]) -> SketchResult<()> {
        if band_keys.len() != self.b {
            return Err(crate::error::SketchError::DimensionMismatch {
                a: band_keys.len(),
                b: self.b,
            });
        }
        for (i, &k) in band_keys.iter().enumerate() {
            self.buckets.entry((i as u32, k)).or_default().push(item_id);
        }
        Ok(())
    }

    /// Query: return unique candidate item ids sharing at least one band key with the query.
    pub fn query(&self, query_band_keys: &[u64]) -> SketchResult<Vec<u64>> {
        if query_band_keys.len() != self.b {
            return Err(crate::error::SketchError::DimensionMismatch {
                a: query_band_keys.len(),
                b: self.b,
            });
        }
        use std::collections::BTreeSet;
        let mut seen: BTreeSet<u64> = BTreeSet::new();
        for (i, &k) in query_band_keys.iter().enumerate() {
            if let Some(items) = self.buckets.get(&(i as u32, k)) {
                for &id in items {
                    seen.insert(id);
                }
            }
        }
        Ok(seen.into_iter().collect())
    }

    /// Number of stored unique (band, key) buckets.
    #[must_use]
    pub fn n_buckets(&self) -> usize {
        self.buckets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsh_index_constructs() {
        let idx = LshIndex::new(4);
        assert_eq!(idx.n_buckets(), 0);
    }

    #[test]
    fn lsh_index_insert_query_roundtrip() {
        let mut idx = LshIndex::new(3);
        idx.insert(1, &[10, 20, 30]).expect("ok");
        idx.insert(2, &[10, 99, 30]).expect("ok");
        let cands = idx.query(&[10, 5, 5]).expect("ok");
        assert!(cands.contains(&1));
        assert!(cands.contains(&2));
    }

    #[test]
    fn lsh_index_no_overlap_no_candidates() {
        let mut idx = LshIndex::new(2);
        idx.insert(1, &[10, 20]).expect("ok");
        let cands = idx.query(&[99, 99]).expect("ok");
        assert!(cands.is_empty());
    }
}
