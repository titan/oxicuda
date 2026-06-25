//! Device-wide parallel primitives operating across all thread blocks.
//!
//! This module provides CUB-equivalent algorithms that aggregate or transform
//! entire device arrays:
//!
//! * [`reduce`] — compute a single aggregate value from all elements
//! * [`scan`]   — compute a prefix sum / prefix op across all elements
//! * [`select`] — stream compaction: keep only elements satisfying a predicate
//! * [`histogram`] — count elements in each equal-width bin
//! * [`run_length_encode`] — compact identical consecutive runs
//! * [`segmented`] — per-segment reduce and scan given segment offsets
//! * [`partition`] — two-output flag partition and consecutive-unique compaction
//! * [`decoupled_scan`] — single-kernel decoupled-lookback scan

pub mod decoupled_scan;
pub mod histogram;
pub mod partition;
pub mod reduce;
pub mod run_length_encode;
pub mod scan;
pub mod segmented;
pub mod select;

pub use decoupled_scan::{DecoupledScanConfig, DecoupledScanTemplate, ScanKind};
pub use histogram::{DeviceHistogramConfig, DeviceHistogramMode, DeviceHistogramTemplate};
pub use partition::{
    DevicePartitionConfig, DevicePartitionTemplate, DeviceSelectUniqueConfig,
    DeviceSelectUniqueTemplate,
};
pub use reduce::{DEFAULT_BLOCK_SIZE, DeviceReduceConfig, DeviceReduceTemplate};
pub use run_length_encode::{DeviceRunLengthEncodeConfig, DeviceRunLengthEncodeTemplate};
pub use scan::{DeviceScanConfig, DeviceScanTemplate};
pub use segmented::{
    SegScanKind, SegmentedReduceConfig, SegmentedReduceTemplate, SegmentedScanConfig,
    SegmentedScanTemplate,
};
pub use select::{DeviceSelectConfig, DeviceSelectTemplate, SelectPredicate};
