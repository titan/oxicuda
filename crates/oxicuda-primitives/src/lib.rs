//! OxiCUDA Primitives — CUB-equivalent high-performance parallel GPU primitives.
//!
//! This crate provides PTX code generators for GPU parallel algorithms without
//! any dependency on the CUDA SDK.  All kernels are generated as PTX source
//! strings at runtime and JIT-compiled via `cuModuleLoadData`.
//!
//! # Sub-modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`warp`] | Warp-level reduce and scan using `shfl.sync.*` |
//! | [`block`] | Block-level reduce and scan via shared memory |
//! | [`device`] | Device-wide reduce, scan, select, histogram, RLE, segmented, partition, decoupled-lookback |
//! | [`sort`] | Device-wide radix sort (4/8-bit, pairs, onesweep) and merge sort (keys + pairs) |
//! | [`ptx_helpers`] | Shared PTX code-generation utilities |
//! | [`handle`] | Execution-context holder with SM version info |
//! | [`host_reference`] | CPU reference implementations for cross-checking GPU output |
//! | [`error`] | Error and result types |
//!
//! # Quick start
//!
//! ```
//! use oxicuda_primitives::device::reduce::{DeviceReduceConfig, DeviceReduceTemplate};
//! use oxicuda_primitives::ptx_helpers::ReduceOp;
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = DeviceReduceConfig::new(ReduceOp::Sum, PtxType::F32);
//! let (pass1_ptx, pass2_ptx) = DeviceReduceTemplate::new(cfg)
//!     .generate(SmVersion::Sm80)
//!     .expect("PTX generation failed");
//!
//! // JIT-compile and launch pass1_ptx and pass2_ptx via the CUDA driver API.
//! assert!(pass1_ptx.contains("device_reduce_pass1_sum_f32"));
//! ```
//!
//! # CUB → oxicuda-primitives mapping
//!
//! Find the right template by its CUB name:
//!
//! | # | CUB primitive                       | oxicuda-primitives template                                   |
//! |---|-------------------------------------|---------------------------------------------------------------|
//! | 1 | `DeviceReduce::Reduce`              | [`device::reduce::DeviceReduceTemplate`]                       |
//! | 2 | `DeviceScan::InclusiveScan`         | [`device::scan::DeviceScanTemplate`]                          |
//! | 3 | `DeviceScan` (decoupled lookback)   | [`device::decoupled_scan::DecoupledScanTemplate`]            |
//! | 4 | `DeviceSelect::If`                   | [`device::select::DeviceSelectTemplate`]                      |
//! | 5 | `DeviceSelect::Unique`              | [`device::partition::DeviceSelectUniqueTemplate`]            |
//! | 6 | `DevicePartition::If`               | [`device::partition::DevicePartitionTemplate`]              |
//! | 7 | `DeviceHistogram::HistogramEven`    | [`device::histogram::DeviceHistogramTemplate`]              |
//! | 8 | `DeviceRunLengthEncode::Encode`     | [`device::run_length_encode::DeviceRunLengthEncodeTemplate`] |
//! | 9 | `DeviceSegmentedReduce::Reduce`     | [`device::segmented::SegmentedReduceTemplate`]              |
//! |10 | `DeviceSegmentedScan` (per-segment) | [`device::segmented::SegmentedScanTemplate`]                |
//! |11 | `DeviceRadixSort::SortKeys` (4-bit) | [`sort::radix_sort::RadixSortTemplate`]                       |
//! |12 | `DeviceRadixSort::SortKeys` (8-bit) | [`sort::radix_sort_8bit::RadixSort8Template`]               |
//! |13 | `DeviceRadixSort::SortPairs`        | [`sort::radix_sort_pairs::RadixPairsTemplate`]              |
//! |14 | `DeviceRadixSort` (onesweep)        | [`sort::onesweep::OnesweepTemplate`]                         |
//! |15 | `DeviceRadixSort` (float keys)      | [`sort::radix_sort_pairs::FloatTwiddleTemplate`]            |
//! |16 | `DeviceMergeSort::SortKeys`         | [`sort::merge_sort::MergeSortTemplate`]                      |
//! |17 | `DeviceMergeSort::SortPairs`        | [`sort::merge_pairs::MergePairsTemplate`]                   |
//! |18 | `DeviceMerge::MergeKeys`            | [`sort::merge_pairs::MergePairsTemplate`]                   |
//!
//! Every device template also exposes a `workspace_bytes(input_len)` query so
//! callers can pre-allocate scratch exactly, and the [`host_reference`] module
//! mirrors each algorithm on the CPU for development-time cross-checking.

pub mod block;
pub mod device;
pub mod error;
pub mod handle;
pub mod host_reference;
pub mod ptx_helpers;
pub mod sort;
pub mod warp;

pub use error::{PrimitivesError, PrimitivesResult};
pub use handle::PrimitivesHandle;
pub use ptx_helpers::{PrimitiveType, ReduceOp};
