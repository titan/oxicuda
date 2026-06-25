//! Combined frequency + cardinality sketch — one structure that simultaneously
//! answers *how many distinct keys* (cardinality, `F_0`) **and** *how often each
//! key occurs* (frequency, `f_i`), plus a tug-of-war estimate of the **distinct
//! support size via AMS over distinct elements**.
//!
//! This fuses three classic sketches behind one streaming interface so a single
//! pass over a `(key, count)` stream yields:
//!
//! * **`F_0` (distinct cardinality)** from an embedded HyperLogLog over the keys
//!   ([`crate::cardinality::hll::HyperLogLog`]).
//! * **per-key frequency `f_i`** from an embedded Count-Min Sketch
//!   ([`crate::frequency::count_min::CountMinSketch`]).
//! * **`F_0` via AMS over distinct elements** — a tug-of-war (`±1`) sketch fed by
//!   each key *exactly once on its first appearance*, so the squared sketch
//!   estimates `Σ_e 1² = F_0` (the second moment of the **indicator** vector of
//!   the distinct support). First-appearance is detected from the Count-Min:
//!   a key is "new" iff its current Count-Min estimate is zero before the update.
//!
//! ## Why two `F_0` estimators?
//!
//! HyperLogLog uses harmonic-mean register statistics; the AMS-over-distinct
//! path uses Rademacher tug-of-war and median-of-means. They have *independent*
//! error mechanisms, so cross-checking them is a cheap consistency signal — if
//! they disagree wildly the stream likely violated an assumption (e.g. counts
//! were not folded and a key recurs with the same sign by chance). Both are
//! returned so the caller can combine or compare them.
//!
//! The first-appearance detection is exact for **insert-only** streams with
//! positive counts (no deletions): once Count-Min over-estimates ≥ 1 the key is
//! permanently treated as seen. With deletions the distinct path degrades
//! gracefully (it may re-count a key that returned to zero), which matches the
//! semantics of an insertion-stream cardinality sketch.

use crate::cardinality::hll::HyperLogLog;
use crate::error::{SketchError, SketchResult};
use crate::frequency::count_min::CountMinSketch;
use crate::handle::LcgRng;
use crate::moment::ams_f2::AmsF2Sketch;

/// Configuration for [`FreqCardSketch`].
#[derive(Debug, Clone)]
pub struct FreqCardConfig {
    /// HyperLogLog precision `p` (`m = 2^p` registers, `4 ≤ p ≤ 16`).
    pub hll_precision: u32,
    /// Count-Min depth `d` (number of rows).
    pub cm_depth: usize,
    /// Count-Min width `w` (columns per row).
    pub cm_width: usize,
    /// AMS distinct sketch median rows `d`.
    pub ams_rows: usize,
    /// AMS distinct sketch mean columns `t`.
    pub ams_cols: usize,
}

impl FreqCardConfig {
    /// A reasonable default: HLL `p = 12`, Count-Min `8 × 2048`, AMS `15 × 2048`.
    #[must_use]
    pub fn balanced() -> Self {
        Self {
            hll_precision: 12,
            cm_depth: 8,
            cm_width: 2048,
            ams_rows: 15,
            ams_cols: 2048,
        }
    }

    fn validate(&self) -> SketchResult<()> {
        if !(4..=16).contains(&self.hll_precision) {
            return Err(SketchError::InvalidPrecision(self.hll_precision));
        }
        if self.cm_depth == 0 || self.cm_width == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(cm_depth, cm_width)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        if self.ams_rows == 0 || self.ams_cols == 0 {
            return Err(SketchError::InvalidParameter {
                name: "(ams_rows, ams_cols)".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(())
    }
}

/// Combined frequency + cardinality sketch.
#[derive(Debug, Clone)]
pub struct FreqCardSketch {
    hll: HyperLogLog,
    cm: CountMinSketch,
    ams_distinct: AmsF2Sketch,
    /// Total of all positive counts streamed (`F_1`, the length of the stream by weight).
    total_count: u64,
    /// Exact distinct count of *new-key first-appearance* events (an upper-checkable
    /// reference; equals true `F_0` for insert-only streams).
    distinct_events: u64,
}

impl FreqCardSketch {
    /// Construct from a configuration. The `seed` drives every embedded sketch's
    /// hash family so two sketches built with the same `(config, seed)` are
    /// mergeable.
    pub fn new(config: &FreqCardConfig, seed: u64) -> SketchResult<Self> {
        config.validate()?;
        let mut cm_rng = LcgRng::new(seed ^ 0x5151_5151_5151_5151);
        let hll = HyperLogLog::new(config.hll_precision, seed)?;
        let cm = CountMinSketch::new(config.cm_depth, config.cm_width, &mut cm_rng)?;
        let ams_distinct = AmsF2Sketch::new(
            config.ams_rows,
            config.ams_cols,
            seed ^ 0xA5A5_A5A5_A5A5_A5A5,
        )?;
        Ok(Self {
            hll,
            cm,
            ams_distinct,
            total_count: 0,
            distinct_events: 0,
        })
    }

    /// Stream a `(key, count)` update with `count ≥ 1`.
    ///
    /// A `count` of zero is a no-op. The HyperLogLog and Count-Min always see the
    /// update; the AMS-distinct path fires only on the key's first appearance.
    pub fn update(&mut self, key: u64, count: u64) {
        if count == 0 {
            return;
        }
        // Detect first appearance via Count-Min BEFORE inserting.
        let seen_before = self.cm.query(key) > 0;
        self.hll.add_u64(key);
        self.cm.update(key, count);
        self.total_count = self.total_count.saturating_add(count);
        if !seen_before {
            // Feed the distinct support indicator: each distinct key contributes +1.
            self.ams_distinct.update(key, 1.0);
            self.distinct_events += 1;
        }
    }

    /// Insert a single occurrence of `key` (`count = 1`).
    pub fn add(&mut self, key: u64) {
        self.update(key, 1);
    }

    /// Estimate the per-key frequency `f_key` (Count-Min over-estimate).
    #[must_use]
    pub fn frequency(&self, key: u64) -> u64 {
        self.cm.query(key)
    }

    /// Estimate distinct cardinality `F_0` from the embedded HyperLogLog.
    #[must_use]
    pub fn cardinality_hll(&self) -> f64 {
        self.hll.estimate()
    }

    /// Estimate distinct cardinality `F_0` from the AMS-over-distinct tug-of-war
    /// sketch: `F_0 ≈ Σ_e 1² = ‖indicator‖₂²`.
    #[must_use]
    pub fn cardinality_ams(&self) -> f64 {
        self.ams_distinct.estimate_f2()
    }

    /// Total weight `F_1 = Σ counts` streamed so far.
    #[must_use]
    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    /// Number of first-appearance events observed (exact `F_0` for insert-only
    /// streams; useful as ground truth in tests / small streams).
    #[must_use]
    pub fn distinct_events(&self) -> u64 {
        self.distinct_events
    }

    /// Merge another combined sketch built with the same configuration and seed.
    ///
    /// HyperLogLog (register max) and Count-Min (cell sums) merge exactly, so
    /// `cardinality_hll` keeps estimating the true union `|A ∪ B|`.
    ///
    /// The AMS-distinct sketch is **linearly** merged. Because the two sketches
    /// share the same sign hashes, after merging each distinct element `e`
    /// carries `m_e` copies of its `±1` sign, where `m_e ∈ {1, 2}` is the number
    /// of merged sketches in which `e` appeared. Hence `cardinality_ams` then
    /// estimates `Σ_e m_e²` — the squared `L2` norm of the per-element
    /// *appearance-count* vector. For **disjoint** key sets every `m_e = 1`, so
    /// it equals `|A| + |B| = |A ∪ B|`; for fully-overlapping sets every
    /// `m_e = 2`, so it equals `4·|A ∩ B|`. The exact `distinct_events` counter
    /// follows the same additive (`|A| + |B|`) semantics as the raw event count.
    pub fn merge(&mut self, other: &FreqCardSketch) -> SketchResult<()> {
        self.hll.merge(&other.hll)?;
        self.cm.merge(&other.cm)?;
        self.ams_distinct.merge(&other.ams_distinct)?;
        self.total_count = self.total_count.saturating_add(other.total_count);
        self.distinct_events = self.distinct_events.saturating_add(other.distinct_events);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fc_invalid_config() {
        let mut cfg = FreqCardConfig::balanced();
        cfg.hll_precision = 2;
        assert!(FreqCardSketch::new(&cfg, 0).is_err());
        let mut cfg = FreqCardConfig::balanced();
        cfg.cm_width = 0;
        assert!(FreqCardSketch::new(&cfg, 0).is_err());
        let mut cfg = FreqCardConfig::balanced();
        cfg.ams_cols = 0;
        assert!(FreqCardSketch::new(&cfg, 0).is_err());
    }

    #[test]
    fn fc_constructs() {
        let s = FreqCardSketch::new(&FreqCardConfig::balanced(), 1).expect("ok");
        assert_eq!(s.total_count(), 0);
        assert_eq!(s.distinct_events(), 0);
    }

    #[test]
    fn fc_frequency_overestimates_truth() {
        let mut s = FreqCardSketch::new(&FreqCardConfig::balanced(), 7).expect("ok");
        for _ in 0..250 {
            s.add(42);
        }
        for i in 0..1000u64 {
            s.add(i + 1000);
        }
        let f = s.frequency(42);
        // Count-Min never under-estimates; with width 2048 the over-estimate is small.
        assert!(f >= 250, "frequency {f} under-estimated true 250");
        assert!(f < 250 + 50, "frequency {f} grossly over-estimated");
    }

    #[test]
    fn fc_total_count_exact() {
        let mut s = FreqCardSketch::new(&FreqCardConfig::balanced(), 3).expect("ok");
        let mut truth = 0u64;
        for i in 0..200u64 {
            let c = (i % 4) + 1;
            s.update(i, c);
            truth += c;
        }
        assert_eq!(s.total_count(), truth);
    }

    #[test]
    fn fc_cardinality_hll_accurate() {
        // Small AMS (this test only checks the HLL path) to keep the
        // first-appearance updates cheap.
        let cfg = FreqCardConfig {
            hll_precision: 14,
            ams_rows: 3,
            ams_cols: 64,
            ..FreqCardConfig::balanced()
        };
        let mut s = FreqCardSketch::new(&cfg, 0).expect("ok");
        let n = 8_000u64;
        for i in 0..n {
            // Each key inserted twice — distinct cardinality is still n.
            s.add(i);
            s.add(i);
        }
        let est = s.cardinality_hll();
        let rel = (est - n as f64).abs() / n as f64;
        assert!(rel < 0.05, "HLL cardinality {est} off from {n} (rel {rel})");
    }

    #[test]
    fn fc_cardinality_ams_counts_distinct_not_multiplicity() {
        // 500 distinct keys, each repeated many times. AMS-over-distinct must
        // estimate ~500 (the indicator second moment), NOT the total weight.
        let cfg = FreqCardConfig {
            ams_rows: 15,
            ams_cols: 1024,
            ..FreqCardConfig::balanced()
        };
        let mut s = FreqCardSketch::new(&cfg, 123_456).expect("ok");
        let k = 500u64;
        for r in 0..20 {
            for i in 0..k {
                s.update(i, 1 + r);
            }
        }
        assert_eq!(
            s.distinct_events(),
            k,
            "first-appearance count must be exact"
        );
        let est = s.cardinality_ams();
        let rel = (est - k as f64).abs() / k as f64;
        assert!(
            rel < 0.12,
            "AMS distinct estimate {est} off from {k} (rel {rel})"
        );
    }

    #[test]
    fn fc_distinct_events_exact_insert_only() {
        let mut s = FreqCardSketch::new(&FreqCardConfig::balanced(), 9).expect("ok");
        for i in 0..300u64 {
            s.add(i);
            s.add(i); // repeats must not bump distinct_events
        }
        assert_eq!(s.distinct_events(), 300);
    }

    #[test]
    fn fc_zero_count_noop() {
        let mut s = FreqCardSketch::new(&FreqCardConfig::balanced(), 1).expect("ok");
        s.update(5, 0);
        assert_eq!(s.total_count(), 0);
        assert_eq!(s.distinct_events(), 0);
        assert_eq!(s.frequency(5), 0);
    }

    #[test]
    fn fc_merge_disjoint_hll_and_ams_give_union() {
        // Disjoint key sets ⇒ every appearance-count m_e = 1, so BOTH the HLL
        // path and the AMS-over-distinct path estimate the true union |A|+|B|.
        let cfg = FreqCardConfig {
            hll_precision: 14,
            ams_rows: 15,
            ams_cols: 1024,
            ..FreqCardConfig::balanced()
        };
        let mut a = FreqCardSketch::new(&cfg, 42).expect("ok");
        let mut b = FreqCardSketch::new(&cfg, 42).expect("ok");
        for i in 0..1500u64 {
            a.add(i);
        }
        for i in 1500..3000u64 {
            b.add(i); // disjoint ⇒ union is 3000
        }
        a.merge(&b).expect("merge ok");
        let hll = a.cardinality_hll();
        assert!(
            (hll - 3000.0).abs() / 3000.0 < 0.05,
            "merged HLL union {hll} should be ≈ 3000"
        );
        let ams = a.cardinality_ams();
        assert!(
            (ams - 3000.0).abs() / 3000.0 < 0.12,
            "disjoint AMS union {ams} should be ≈ 3000"
        );
        assert_eq!(a.distinct_events(), 3000);
    }

    #[test]
    fn fc_merge_overlap_hll_union_ams_appearance_l2() {
        // Identical key sets ⇒ every m_e = 2, so the AMS path estimates
        // Σ m_e² = 4·K while the HLL path still estimates the union K.
        let cfg = FreqCardConfig {
            hll_precision: 14,
            ams_rows: 15,
            ams_cols: 1024,
            ..FreqCardConfig::balanced()
        };
        let mut a = FreqCardSketch::new(&cfg, 42).expect("ok");
        let mut b = FreqCardSketch::new(&cfg, 42).expect("ok");
        let k = 1500u64;
        for i in 0..k {
            a.add(i);
            b.add(i); // identical key set
        }
        a.merge(&b).expect("merge ok");
        let hll = a.cardinality_hll();
        assert!(
            (hll - k as f64).abs() / (k as f64) < 0.05,
            "merged HLL union {hll} should be ≈ {k}"
        );
        // Σ m_e² = 4·K because each of the K shared keys now carries sign·2.
        let ams = a.cardinality_ams();
        let expected = 4.0 * k as f64;
        assert!(
            (ams - expected).abs() / expected < 0.12,
            "overlap AMS appearance-L2 {ams} should be ≈ {expected}"
        );
        assert_eq!(a.distinct_events(), 2 * k);
    }

    #[test]
    fn fc_merge_rejects_mismatched_config() {
        let a = FreqCardSketch::new(&FreqCardConfig::balanced(), 1).expect("ok");
        let cfg2 = FreqCardConfig {
            hll_precision: 13,
            ..FreqCardConfig::balanced()
        };
        let mut b = FreqCardSketch::new(&cfg2, 1).expect("ok");
        assert!(b.merge(&a).is_err());
    }
}
