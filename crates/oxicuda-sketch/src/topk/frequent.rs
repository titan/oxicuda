//! Frequent items algorithm: report items appearing more than `n / k_slots` times.
//!
//! Uses Misra-Gries under the hood; the public interface returns the candidate set
//! which is guaranteed to contain every item whose true frequency exceeds the threshold.
//! Note that the Misra-Gries estimate may undercount by up to `n / k_slots` (where
//! `k_slots` is the number of slots, equal to the constructor `k`). To preserve the
//! "items with TRUE freq > n/k IS in output" guarantee, we report ALL items in the
//! candidate list with positive estimated count.

use crate::error::SketchResult;
use crate::topk::misra_gries::MisraGries;

/// Frequent-items sketch (wrapper around Misra-Gries).
#[derive(Debug, Clone)]
pub struct FrequentItems {
    pub inner: MisraGries,
}

impl FrequentItems {
    /// New sketch with parameter `k`. Uses `k + 1` slots internally so that an item with
    /// true frequency `> n / k` is guaranteed to remain in the candidate list.
    pub fn new(k: usize) -> SketchResult<Self> {
        Ok(Self {
            inner: MisraGries::new(k + 1)?,
        })
    }

    /// Insert an item.
    pub fn add(&mut self, x: u64) {
        self.inner.add(x);
    }

    /// Candidate items in the Misra-Gries table — superset of true frequent items.
    /// Any element with true frequency > `n / k` (where k was the constructor parameter)
    /// IS guaranteed to be in this list.
    #[must_use]
    pub fn frequent(&self) -> Vec<(u64, u64)> {
        self.inner
            .candidates()
            .iter()
            .filter(|(_, c)| *c > 0)
            .copied()
            .collect()
    }

    /// Total inserts.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.inner.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequent_constructs() {
        let f = FrequentItems::new(5).expect("ok");
        assert_eq!(f.count(), 0);
    }

    #[test]
    fn frequent_includes_heavy() {
        let mut f = FrequentItems::new(4).expect("ok");
        for _ in 0..300 {
            f.add(7);
        }
        for i in 0..700u64 {
            f.add(i + 100);
        }
        let freq = f.frequent();
        let keys: Vec<u64> = freq.iter().map(|(k, _)| *k).collect();
        assert!(keys.contains(&7), "missing frequent item 7");
    }
}
