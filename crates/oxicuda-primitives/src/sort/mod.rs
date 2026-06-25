//! Device-wide sort algorithms.
//!
//! * [`radix_sort`] — 4-bit LSD radix sort; fastest for integer keys.
//! * [`radix_sort_8bit`] — 8-bit LSD radix sort (CUB default digit width).
//! * [`radix_sort_pairs`] — key+value, descending, and floating-point radix sort.
//! * [`onesweep`] — single-kernel-per-pass decoupled-lookback radix sort.
//! * [`merge_sort`] — stable bitonic-block + binary-search merge sort.
//! * [`merge_pairs`] — key+value co-rank merge (sort pass + standalone merge).
//! * [`segmented_sort`] — per-segment block bitonic sort given segment offsets.

pub mod merge_pairs;
pub mod merge_sort;
pub mod onesweep;
pub mod radix_sort;
pub mod radix_sort_8bit;
pub mod radix_sort_pairs;
pub mod segmented_sort;

pub use merge_pairs::{MergePairsConfig, MergePairsTemplate};
pub use merge_sort::{MergeSortConfig, MergeSortTemplate};
pub use onesweep::{OnesweepConfig, OnesweepTemplate};
pub use radix_sort::{RadixSortConfig, RadixSortTemplate};
pub use radix_sort_8bit::{RadixSort8Config, RadixSort8Template};
pub use radix_sort_pairs::{
    FloatTwiddleConfig, FloatTwiddleTemplate, RadixPairsConfig, RadixPairsTemplate, SortOrder,
};
pub use segmented_sort::{SegmentedSortConfig, SegmentedSortTemplate};
