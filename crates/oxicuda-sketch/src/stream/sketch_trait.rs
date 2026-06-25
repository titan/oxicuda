//! Unified streaming-sketch interface: [`StreamingSketch`].
//!
//! A small trait that captures the common contract of the linear / mergeable
//! sketches in this crate — *update* with stream items, *merge* compatible
//! sketches, and *query* a summary statistic — so generic streaming pipelines
//! can be written once and instantiated over any concrete sketch.
//!
//! ```text
//!   trait StreamingSketch<Item> {
//!       type Query;                         // what `query` returns
//!       fn update(&mut self, item: Item);   // absorb one stream element
//!       fn merge(&mut self, other) -> ...;  // combine two summaries
//!       fn query(&self) -> Self::Query;     // current estimate
//!   }
//! ```
//!
//! Concrete impls are provided for:
//!
//! * [`HyperLogLog`] — `update(u64)`, `query` ⇒ estimated distinct count (`f64`).
//! * [`CountMinSketch`] — `update((key, count))`, `query` ⇒ `()` (use
//!   [`CountMinSketch::query`] for a specific key; the trait's `query` reports
//!   the empty summary because Count-Min has no single scalar).
//! * [`BloomFilter`] — `update(u64)`, `query` ⇒ estimated load (`f64`).
//! * [`crate::cardinality::theta_sketch::ThetaSketch`] — `update(u64)`,
//!   `query` ⇒ estimated cardinality (`f64`).
//!
//! Sketches that also implement [`crate::serde::SketchSerialize`] gain
//! `serialize`/`deserialize` for free via the blanket
//! [`SerializableStreamingSketch`] super-interface, fulfilling the
//! `update / merge / query / serialize` streaming-interface contract.

use crate::cardinality::hll::HyperLogLog;
use crate::cardinality::theta_sketch::ThetaSketch;
use crate::error::SketchResult;
use crate::frequency::count_min::CountMinSketch;
use crate::membership::bloom::BloomFilter;

/// Common interface for a mergeable streaming sketch over items of type `Item`.
pub trait StreamingSketch<Item> {
    /// The type returned by [`StreamingSketch::query`].
    type Query;

    /// Absorb one stream element.
    fn update(&mut self, item: Item);

    /// Merge another compatible sketch into `self`.
    ///
    /// Returns an error if the two sketches are structurally incompatible (e.g.
    /// different dimensions or hash seeds).
    fn merge(&mut self, other: &Self) -> SketchResult<()>;

    /// Compute the current summary / estimate.
    fn query(&self) -> Self::Query;
}

/// Convenience super-interface for streaming sketches that can also serialise
/// themselves, giving the full `update / merge / query / serialize` contract.
pub trait SerializableStreamingSketch<Item>:
    StreamingSketch<Item> + crate::serde::SketchSerialize
{
    /// Serialise to OxiCUDA's binary sketch format.
    fn serialize(&self) -> Vec<u8> {
        self.to_bytes()
    }

    /// Deserialise from OxiCUDA's binary sketch format.
    fn deserialize(bytes: &[u8]) -> SketchResult<Self> {
        Self::from_bytes(bytes)
    }
}

impl<Item, T> SerializableStreamingSketch<Item> for T where
    T: StreamingSketch<Item> + crate::serde::SketchSerialize
{
}

impl StreamingSketch<u64> for HyperLogLog {
    type Query = f64;

    fn update(&mut self, item: u64) {
        self.add_u64(item);
    }

    fn merge(&mut self, other: &Self) -> SketchResult<()> {
        HyperLogLog::merge(self, other)
    }

    fn query(&self) -> f64 {
        self.estimate()
    }
}

impl StreamingSketch<(u64, u64)> for CountMinSketch {
    /// Count-Min has no single scalar summary; the per-key estimate is obtained
    /// from [`CountMinSketch::query`].
    type Query = ();

    fn update(&mut self, item: (u64, u64)) {
        let (key, count) = item;
        CountMinSketch::update(self, key, count);
    }

    fn merge(&mut self, other: &Self) -> SketchResult<()> {
        CountMinSketch::merge(self, other)
    }

    fn query(&self) {}
}

impl StreamingSketch<u64> for BloomFilter {
    type Query = f64;

    fn update(&mut self, item: u64) {
        self.insert(item);
    }

    fn merge(&mut self, other: &Self) -> SketchResult<()> {
        BloomFilter::merge(self, other)
    }

    fn query(&self) -> f64 {
        self.estimate_load()
    }
}

impl StreamingSketch<u64> for ThetaSketch {
    type Query = f64;

    fn update(&mut self, item: u64) {
        self.add_bytes(&item.to_le_bytes());
    }

    fn merge(&mut self, other: &Self) -> SketchResult<()> {
        let unioned = ThetaSketch::union(self, other)?;
        *self = unioned;
        Ok(())
    }

    fn query(&self) -> f64 {
        self.estimate_cardinality()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::serde::SketchSerialize;

    #[test]
    fn streaming_hll_update_query_merge() {
        let mut a: HyperLogLog = HyperLogLog::new(14, 0).expect("ok");
        let mut b: HyperLogLog = HyperLogLog::new(14, 0).expect("ok");
        for i in 0..3000u64 {
            StreamingSketch::update(&mut a, i);
        }
        for i in 3000..6000u64 {
            StreamingSketch::update(&mut b, i);
        }
        StreamingSketch::merge(&mut a, &b).expect("merge ok");
        let est = StreamingSketch::query(&a);
        assert!((est - 6000.0).abs() / 6000.0 < 0.05, "merged HLL est {est}");
    }

    #[test]
    fn streaming_count_min_update_merge() {
        let mut rng = LcgRng::new(7);
        let mut cm = CountMinSketch::new(4, 512, &mut rng).expect("ok");
        for _ in 0..100 {
            StreamingSketch::update(&mut cm, (42u64, 1u64));
        }
        // The trait query yields unit; the concrete per-key query works.
        #[allow(clippy::let_unit_value)]
        let _ = StreamingSketch::query(&cm);
        assert!(cm.query(42) >= 100);
    }

    #[test]
    fn streaming_bloom_update_query() {
        let mut bf = BloomFilter::new(8192, 5, 0).expect("ok");
        for i in 0..500u64 {
            StreamingSketch::update(&mut bf, i);
        }
        let load = StreamingSketch::query(&bf);
        assert!((load - 500.0).abs() / 500.0 < 0.2, "load est {load}");
    }

    #[test]
    fn streaming_theta_update_query() {
        let mut t = ThetaSketch::new(4096, 0).expect("ok");
        for i in 0..5000u64 {
            StreamingSketch::update(&mut t, i);
        }
        let est = StreamingSketch::query(&t);
        assert!((est - 5000.0).abs() / 5000.0 < 0.10, "theta est {est}");
    }

    #[test]
    fn serializable_streaming_roundtrip() {
        let mut hll: HyperLogLog = HyperLogLog::new(12, 5).expect("ok");
        for i in 0..2000u64 {
            StreamingSketch::update(&mut hll, i);
        }
        let bytes = SerializableStreamingSketch::<u64>::serialize(&hll);
        let back = <HyperLogLog as SketchSerialize>::from_bytes(&bytes).expect("deserialize ok");
        assert_eq!(back.registers, hll.registers);
    }
}
