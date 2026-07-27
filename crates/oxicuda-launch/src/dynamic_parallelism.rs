//! Dynamic parallelism support for device-side kernel launches.
//!
//! CUDA dynamic parallelism allows kernels running on the GPU to launch
//! child kernels without returning to the host. This module provides
//! configuration, planning, and PTX code generation for nested kernel
//! launches.
//!
//! # Architecture requirements
//!
//! Dynamic parallelism requires compute capability 3.5+ (sm_35). All
//! [`SmVersion`] variants in this crate are sm_75+, so they all support
//! dynamic parallelism.
//!
//! # CUDA nesting limits
//!
//! - Maximum nesting depth: 24
//! - Default pending launch limit: 2048
//! - Each pending launch consumes device memory for bookkeeping
//!
//! # Example
//!
//! ```rust
//! use oxicuda_launch::dynamic_parallelism::{
//!     DynamicParallelismConfig, ChildKernelSpec, GridSpec,
//!     validate_dynamic_config, plan_dynamic_launch,
//!     generate_child_launch_ptx, generate_device_sync_ptx,
//!     estimate_launch_overhead, max_nesting_for_sm,
//! };
//! use oxicuda_launch::Dim3;
//! use oxicuda_ptx::arch::SmVersion;
//! use oxicuda_ptx::PtxType;
//!
//! let config = DynamicParallelismConfig {
//!     max_nesting_depth: 4,
//!     max_pending_launches: 2048,
//!     sync_depth: 2,
//!     child_grid: Dim3::x(128),
//!     child_block: Dim3::x(256),
//!     child_shared_mem: 0,
//!     sm_version: SmVersion::Sm80,
//! };
//!
//! validate_dynamic_config(&config).ok();
//! let plan = plan_dynamic_launch(&config).ok();
//!
//! let child = ChildKernelSpec {
//!     name: "child_kernel".to_string(),
//!     param_types: vec![PtxType::U64, PtxType::U32],
//!     grid_dim: GridSpec::Fixed(Dim3::x(128)),
//!     block_dim: Dim3::x(256),
//!     shared_mem_bytes: 0,
//! };
//!
//! let ptx = generate_child_launch_ptx("parent_kernel", &child, SmVersion::Sm80);
//! let sync_ptx = generate_device_sync_ptx(SmVersion::Sm80);
//! let overhead = estimate_launch_overhead(4, 2048);
//! let max_depth = max_nesting_for_sm(SmVersion::Sm80);
//! ```

use std::fmt;

use oxicuda_ptx::PtxType;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::error::PtxGenError;

use crate::error::LaunchError;
use crate::grid::Dim3;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum nesting depth allowed by CUDA hardware.
const CUDA_MAX_NESTING_DEPTH: u32 = 24;

/// Default maximum number of pending (un-synchronized) child launches.
const DEFAULT_MAX_PENDING_LAUNCHES: u32 = 2048;

/// Base memory overhead per pending launch in bytes.
/// This accounts for the device-side launch descriptor, parameter storage,
/// and internal bookkeeping structures.
const BASE_LAUNCH_OVERHEAD_BYTES: u64 = 2048;

/// Additional overhead per nesting level in bytes.
/// Deeper nesting requires additional stack frames and synchronization state.
const PER_DEPTH_OVERHEAD_BYTES: u64 = 4096;

// ---------------------------------------------------------------------------
// DynamicParallelismConfig
// ---------------------------------------------------------------------------

/// Configuration for dynamic parallelism (device-side kernel launches).
///
/// Controls nesting depth, pending launch limits, synchronization behavior,
/// and child kernel launch dimensions.
///
/// # CUDA constraints
///
/// - `max_nesting_depth` must be in `1..=24`.
/// - `max_pending_launches` must be at least 1.
/// - `sync_depth` must be less than or equal to `max_nesting_depth`.
/// - All grid and block dimensions must be non-zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicParallelismConfig {
    /// Maximum nesting depth for child kernel launches (CUDA limit: 24).
    pub max_nesting_depth: u32,
    /// Maximum number of pending (un-synchronized) child launches (default 2048).
    pub max_pending_launches: u32,
    /// Depth at which to insert synchronization barriers.
    ///
    /// Child kernels launched at depths >= `sync_depth` will synchronize
    /// before returning to the parent, preventing unbounded pending launches.
    pub sync_depth: u32,
    /// Grid dimensions for child kernel launches.
    pub child_grid: Dim3,
    /// Block dimensions for child kernel launches.
    pub child_block: Dim3,
    /// Dynamic shared memory allocation for child kernels (bytes).
    pub child_shared_mem: u32,
    /// Target GPU architecture.
    pub sm_version: SmVersion,
}

impl DynamicParallelismConfig {
    /// Creates a new configuration with default values.
    ///
    /// Defaults:
    /// - `max_nesting_depth`: 4
    /// - `max_pending_launches`: 2048
    /// - `sync_depth`: 2
    /// - `child_grid`: 128 blocks
    /// - `child_block`: 256 threads
    /// - `child_shared_mem`: 0
    #[must_use]
    pub fn new(sm_version: SmVersion) -> Self {
        Self {
            max_nesting_depth: 4,
            max_pending_launches: DEFAULT_MAX_PENDING_LAUNCHES,
            sync_depth: 2,
            child_grid: Dim3::x(128),
            child_block: Dim3::x(256),
            child_shared_mem: 0,
            sm_version,
        }
    }
}

impl fmt::Display for DynamicParallelismConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DynParallelism(depth={}, pending={}, sync@{}, grid={}, block={}, smem={}, {})",
            self.max_nesting_depth,
            self.max_pending_launches,
            self.sync_depth,
            self.child_grid,
            self.child_block,
            self.child_shared_mem,
            self.sm_version,
        )
    }
}

// ---------------------------------------------------------------------------
// DynamicLaunchPlan
// ---------------------------------------------------------------------------

/// A validated plan for a dynamic (device-side) kernel launch.
///
/// Contains the configuration, kernel names, and estimated resource usage.
/// Created by [`plan_dynamic_launch`].
#[derive(Debug, Clone)]
pub struct DynamicLaunchPlan {
    /// The validated configuration.
    pub config: DynamicParallelismConfig,
    /// Name of the parent kernel that launches child kernels.
    pub parent_kernel_name: String,
    /// Name of the child kernel to be launched from device code.
    pub child_kernel_name: String,
    /// Estimated total number of child kernel launches.
    pub estimated_child_launches: u64,
    /// Estimated memory overhead per launch in bytes.
    pub memory_overhead_bytes: u64,
}

impl fmt::Display for DynamicLaunchPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DynamicLaunchPlan {{ parent: '{}', child: '{}', \
             est_launches: {}, overhead: {} bytes, config: {} }}",
            self.parent_kernel_name,
            self.child_kernel_name,
            self.estimated_child_launches,
            self.memory_overhead_bytes,
            self.config,
        )
    }
}

// ---------------------------------------------------------------------------
// ChildKernelSpec
// ---------------------------------------------------------------------------

/// Specification for a child kernel to be launched from device code.
///
/// Describes the kernel signature, grid/block dimensions, and shared
/// memory requirements needed to generate the device-side launch PTX.
#[derive(Debug, Clone)]
pub struct ChildKernelSpec {
    /// Name of the child kernel function.
    pub name: String,
    /// PTX types of the kernel parameters, in order.
    pub param_types: Vec<PtxType>,
    /// How the grid dimensions are determined.
    pub grid_dim: GridSpec,
    /// Block dimensions (threads per block).
    pub block_dim: Dim3,
    /// Dynamic shared memory in bytes.
    pub shared_mem_bytes: u32,
}

// ---------------------------------------------------------------------------
// GridSpec
// ---------------------------------------------------------------------------

/// Specifies how child kernel grid dimensions are determined.
///
/// Device-side kernel launches can use fixed grid sizes, data-dependent
/// sizes derived from kernel parameters, or per-thread launches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GridSpec {
    /// A constant grid size known at code generation time.
    Fixed(Dim3),
    /// Grid size derived from a kernel parameter at runtime.
    ///
    /// The `param_index` identifies which parameter of the parent kernel
    /// contains the element count. The generated PTX computes the grid
    /// size as `ceil(param / block_size)`.
    DataDependent {
        /// Index of the parent kernel parameter holding the element count.
        param_index: u32,
    },
    /// Launch one child kernel per thread in the parent kernel.
    ///
    /// Each thread in the parent launches exactly one child grid.
    /// The child grid size is typically 1 block.
    ThreadDependent,
}

impl fmt::Display for GridSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fixed(dim) => write!(f, "Fixed({dim})"),
            Self::DataDependent { param_index } => {
                write!(f, "DataDependent(param[{param_index}])")
            }
            Self::ThreadDependent => write!(f, "ThreadDependent"),
        }
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validates a dynamic parallelism configuration.
///
/// Checks all CUDA hardware constraints:
/// - Nesting depth must be in `1..=24`.
/// - Pending launches must be at least 1.
/// - Sync depth must not exceed nesting depth.
/// - All child grid and block dimensions must be non-zero.
/// - Total threads per child block must not exceed the architecture limit.
/// - Child shared memory must not exceed the architecture limit.
///
/// # Errors
///
/// Returns [`LaunchError`] describing the first constraint violation found.
pub fn validate_dynamic_config(config: &DynamicParallelismConfig) -> Result<(), LaunchError> {
    // Nesting depth
    if config.max_nesting_depth == 0 || config.max_nesting_depth > CUDA_MAX_NESTING_DEPTH {
        return Err(LaunchError::InvalidDimension {
            dim: "max_nesting_depth",
            value: config.max_nesting_depth,
        });
    }

    // Pending launches
    if config.max_pending_launches == 0 {
        return Err(LaunchError::InvalidDimension {
            dim: "max_pending_launches",
            value: 0,
        });
    }

    // Sync depth
    if config.sync_depth > config.max_nesting_depth {
        return Err(LaunchError::InvalidDimension {
            dim: "sync_depth",
            value: config.sync_depth,
        });
    }

    // Child grid dimensions
    if config.child_grid.x == 0 {
        return Err(LaunchError::InvalidDimension {
            dim: "child_grid.x",
            value: 0,
        });
    }
    if config.child_grid.y == 0 {
        return Err(LaunchError::InvalidDimension {
            dim: "child_grid.y",
            value: 0,
        });
    }
    if config.child_grid.z == 0 {
        return Err(LaunchError::InvalidDimension {
            dim: "child_grid.z",
            value: 0,
        });
    }

    // Child block dimensions
    if config.child_block.x == 0 {
        return Err(LaunchError::InvalidDimension {
            dim: "child_block.x",
            value: 0,
        });
    }
    if config.child_block.y == 0 {
        return Err(LaunchError::InvalidDimension {
            dim: "child_block.y",
            value: 0,
        });
    }
    if config.child_block.z == 0 {
        return Err(LaunchError::InvalidDimension {
            dim: "child_block.z",
            value: 0,
        });
    }

    // Block size limit
    let max_threads = config.sm_version.max_threads_per_block();
    let block_total = config.child_block.total();
    if block_total > max_threads {
        return Err(LaunchError::BlockSizeExceedsLimit {
            requested: block_total,
            max: max_threads,
        });
    }

    // Shared memory limit
    let max_smem = config.sm_version.max_shared_mem_per_block();
    if config.child_shared_mem > max_smem {
        return Err(LaunchError::SharedMemoryExceedsLimit {
            requested: config.child_shared_mem,
            max: max_smem,
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

/// Creates a validated launch plan from a dynamic parallelism configuration.
///
/// Validates the configuration, then estimates the number of child launches
/// and per-launch memory overhead. The parent and child kernel names are
/// generated from the configuration.
///
/// # Errors
///
/// Returns [`LaunchError`] if the configuration is invalid.
pub fn plan_dynamic_launch(
    config: &DynamicParallelismConfig,
) -> Result<DynamicLaunchPlan, LaunchError> {
    validate_dynamic_config(config)?;

    let parent_grid_total = config.child_grid.total_u64();
    let estimated_child_launches =
        parent_grid_total.saturating_mul(config.child_block.total() as u64);
    let memory_overhead_bytes =
        estimate_launch_overhead(config.max_nesting_depth, config.max_pending_launches);

    Ok(DynamicLaunchPlan {
        config: config.clone(),
        parent_kernel_name: String::from("parent_kernel"),
        child_kernel_name: String::from("child_kernel"),
        estimated_child_launches,
        memory_overhead_bytes,
    })
}

// ---------------------------------------------------------------------------
// Overhead estimation
// ---------------------------------------------------------------------------

/// Estimates the device memory overhead for dynamic parallelism in bytes.
///
/// The overhead comes from:
/// - Per-launch descriptors (`BASE_LAUNCH_OVERHEAD_BYTES` per pending launch)
/// - Per-depth stack and synchronization state (`PER_DEPTH_OVERHEAD_BYTES` per level)
///
/// # Arguments
///
/// - `depth` — maximum nesting depth
/// - `pending` — maximum number of pending (un-synchronized) launches
///
/// # Returns
///
/// Estimated total overhead in bytes.
pub fn estimate_launch_overhead(depth: u32, pending: u32) -> u64 {
    let per_launch = BASE_LAUNCH_OVERHEAD_BYTES.saturating_mul(pending as u64);
    let per_depth = PER_DEPTH_OVERHEAD_BYTES.saturating_mul(depth as u64);
    per_launch.saturating_add(per_depth)
}

/// Returns the maximum supported nesting depth for a given SM version.
///
/// All architectures from sm_35 onward support dynamic parallelism with
/// a hardware maximum of 24 nesting levels. The available SM versions
/// in this crate (sm_75+) all support the full nesting depth.
///
/// For practical purposes, deep nesting (>8) is rarely beneficial due
/// to launch overhead and memory consumption.
pub fn max_nesting_for_sm(sm: SmVersion) -> u32 {
    // All supported SM versions (75+) support dynamic parallelism.
    // Newer architectures have the same 24-level limit but with
    // improved launch latency.
    match sm {
        SmVersion::Sm75 => CUDA_MAX_NESTING_DEPTH,
        SmVersion::Sm80 | SmVersion::Sm86 => CUDA_MAX_NESTING_DEPTH,
        SmVersion::Sm89 => CUDA_MAX_NESTING_DEPTH,
        SmVersion::Sm90 | SmVersion::Sm90a => CUDA_MAX_NESTING_DEPTH,
        SmVersion::Sm100 => CUDA_MAX_NESTING_DEPTH,
        SmVersion::Sm120 => CUDA_MAX_NESTING_DEPTH,
    }
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Generates PTX code for a device-side child kernel launch.
///
/// Produces a `.func` that computes the child grid dimensions according to
/// the [`GridSpec`], obtains a parameter buffer from the CUDA device runtime
/// via `cudaGetParameterBufferV2`, marshals the kernel arguments into that
/// buffer, and enqueues the child grid with `cudaLaunchDeviceV2`. The real
/// `cudaError_t` returned by `cudaLaunchDeviceV2` is propagated to the
/// caller.
///
/// The generated module declares `cudaGetParameterBufferV2` and
/// `cudaLaunchDeviceV2` as `.extern .func` prototypes using the device
/// runtime ABI, so it must be linked against `cudadevrt` at JIT/load time.
///
/// # Arguments
///
/// - `parent_name` — name of the parent kernel (used for symbol naming)
/// - `child` — specification of the child kernel to launch
/// - `sm` — target architecture for PTX ISA version selection
///
/// # Errors
///
/// Returns [`PtxGenError`] if the child specification is invalid or
/// the target architecture does not support dynamic parallelism
/// (all sm_75+ architectures do).
pub fn generate_child_launch_ptx(
    parent_name: &str,
    child: &ChildKernelSpec,
    sm: SmVersion,
) -> Result<String, PtxGenError> {
    // Validate child spec
    if child.name.is_empty() {
        return Err(PtxGenError::GenerationFailed(
            "child kernel name must not be empty".to_string(),
        ));
    }
    if child.block_dim.x == 0 || child.block_dim.y == 0 || child.block_dim.z == 0 {
        return Err(PtxGenError::GenerationFailed(
            "child block dimensions must be non-zero".to_string(),
        ));
    }

    let (isa_major, isa_minor) = sm.ptx_isa_version();
    let target = sm.as_ptx_str();

    let mut ptx = String::with_capacity(2048);

    // PTX header
    ptx.push_str(&format!(
        "// Dynamic parallelism: {parent_name} -> {child_name}\n",
        child_name = child.name,
    ));
    ptx.push_str(&format!(
        ".version {isa_major}.{isa_minor}\n\
         .target {target}\n\
         .address_size 64\n\n"
    ));

    // CUDA device runtime prototypes (linked against cudadevrt).
    append_device_runtime_externs(&mut ptx);

    // Extern declaration for the child kernel
    ptx.push_str(&format!(
        "// Child kernel declaration\n\
         .extern .entry {child_name}(\n",
        child_name = child.name,
    ));
    for (i, ptype) in child.param_types.iter().enumerate() {
        let comma = if i + 1 < child.param_types.len() {
            ","
        } else {
            ""
        };
        ptx.push_str(&format!(
            "    .param {ty} _param_{i}{comma}\n",
            ty = ptype.as_ptx_str(),
        ));
    }
    ptx.push_str(")\n\n");

    // Launch helper function
    let func_name = format!(
        "__{parent_name}_launch_{child_name}",
        child_name = child.name
    );
    ptx.push_str("// Device-side launch helper\n");
    ptx.push_str(&format!(".func (.param .s32 _retval) {func_name}(\n"));

    // Parameters for the launch helper (same as child kernel params)
    for (i, ptype) in child.param_types.iter().enumerate() {
        let comma = if i + 1 < child.param_types.len() {
            ","
        } else {
            ""
        };
        ptx.push_str(&format!(
            "    .param {ty} arg_{i}{comma}\n",
            ty = ptype.as_ptx_str(),
        ));
    }
    ptx.push_str(")\n{\n");

    // Register declarations
    ptx.push_str("    // Register declarations\n");
    ptx.push_str("    .reg .s32 %retval;\n");
    ptx.push_str("    .reg .u32 %grid_x, %grid_y, %grid_z;\n");
    ptx.push_str("    .reg .u32 %block_x, %block_y, %block_z;\n");
    ptx.push_str("    .reg .u32 %shared_mem;\n");
    ptx.push_str("    .reg .u64 %stream;\n");
    // Registers and address-taken symbol needed for the device-runtime calls.
    ptx.push_str("    .reg .u64 %func_addr, %param_buf;\n");
    ptx.push_str("    .reg .pred %p_buf_null;\n");

    // Additional registers for data-dependent grid
    if let GridSpec::DataDependent { .. } = &child.grid_dim {
        ptx.push_str("    .reg .u32 %n_elements, %block_size;\n");
    }
    if matches!(&child.grid_dim, GridSpec::ThreadDependent) {
        ptx.push_str("    .reg .u32 %tid_x, %ntid_x, %ctaid_x;\n");
    }

    ptx.push('\n');

    // Set grid dimensions based on GridSpec
    match &child.grid_dim {
        GridSpec::Fixed(dim) => {
            ptx.push_str(&format!(
                "    // Fixed grid dimensions\n\
                 mov.u32 %grid_x, {gx};\n\
                 mov.u32 %grid_y, {gy};\n\
                 mov.u32 %grid_z, {gz};\n",
                gx = dim.x,
                gy = dim.y,
                gz = dim.z,
            ));
        }
        GridSpec::DataDependent { param_index } => {
            ptx.push_str(&format!(
                "    // Data-dependent grid: ceil(param[{param_index}] / block.x)\n\
                 ld.param.u32 %n_elements, [arg_{param_index}];\n\
                 mov.u32 %block_size, {bx};\n\
                 add.u32 %grid_x, %n_elements, %block_size;\n\
                 sub.u32 %grid_x, %grid_x, 1;\n\
                 div.u32 %grid_x, %grid_x, %block_size;\n\
                 mov.u32 %grid_y, 1;\n\
                 mov.u32 %grid_z, 1;\n",
                bx = child.block_dim.x,
            ));
        }
        GridSpec::ThreadDependent => {
            ptx.push_str(
                "    // Thread-dependent: one child launch per parent thread\n\
                 mov.u32 %tid_x, %tid.x;\n\
                 mov.u32 %ntid_x, %ntid.x;\n\
                 mov.u32 %ctaid_x, %ctaid.x;\n\
                 // Each thread launches a 1-block child grid\n\
                 mov.u32 %grid_x, 1;\n\
                 mov.u32 %grid_y, 1;\n\
                 mov.u32 %grid_z, 1;\n",
            );
        }
    }

    // Set block dimensions
    ptx.push_str(&format!(
        "\n    // Block dimensions\n\
         mov.u32 %block_x, {bx};\n\
         mov.u32 %block_y, {by};\n\
         mov.u32 %block_z, {bz};\n",
        bx = child.block_dim.x,
        by = child.block_dim.y,
        bz = child.block_dim.z,
    ));

    // Shared memory and stream
    ptx.push_str(&format!(
        "\n    // Shared memory and stream (NULL = default stream)\n\
         mov.u32 %shared_mem, {smem};\n\
         mov.u64 %stream, 0;\n",
        smem = child.shared_mem_bytes,
    ));

    // Take the address of the child kernel entry. cudaGetParameterBufferV2
    // and cudaLaunchDeviceV2 identify the kernel by its entry-point address.
    ptx.push_str(&format!(
        "\n    // Resolve child kernel entry address\n\
         mov.u64 %func_addr, {child_name};\n",
        child_name = child.name,
    ));

    // Obtain a device-runtime parameter buffer sized for the child grid.
    // cudaGetParameterBufferV2 reserves storage in the device launch pool;
    // the kernel arguments are then written into the returned buffer.
    ptx.push_str(
        "\n    // Allocate parameter buffer via the CUDA device runtime\n\
         // void *cudaGetParameterBufferV2(void *func,\n\
         //                                dim3 gridDim, dim3 blockDim,\n\
         //                                unsigned int sharedMem)\n\
         {\n\
         .param .b64 retval_pb;\n\
         .param .b64 param0_pb;\n\
         .param .align 4 .b8 param1_pb[12];\n\
         .param .align 4 .b8 param2_pb[12];\n\
         .param .b32 param3_pb;\n\
         st.param.b64 [param0_pb], %func_addr;\n\
         st.param.b32 [param1_pb+0], %grid_x;\n\
         st.param.b32 [param1_pb+4], %grid_y;\n\
         st.param.b32 [param1_pb+8], %grid_z;\n\
         st.param.b32 [param2_pb+0], %block_x;\n\
         st.param.b32 [param2_pb+4], %block_y;\n\
         st.param.b32 [param2_pb+8], %block_z;\n\
         st.param.b32 [param3_pb], %shared_mem;\n\
         call.uni (retval_pb), cudaGetParameterBufferV2,\n\
         \x20    (param0_pb, param1_pb, param2_pb, param3_pb);\n\
         ld.param.b64 %param_buf, [retval_pb];\n\
         }\n",
    );

    // A NULL parameter buffer means the device launch pool is exhausted;
    // report cudaErrorLaunchPendingCountExceeded (== 209) to the caller.
    ptx.push_str(
        "\n    // Bail out if the device runtime could not provide a buffer\n\
         setp.eq.u64 %p_buf_null, %param_buf, 0;\n\
         @%p_buf_null mov.s32 %retval, 209;\n\
         @%p_buf_null bra $L_launch_done;\n",
    );

    // Marshal the child kernel arguments into the parameter buffer.
    // cudaGetParameterBufferV2 guarantees the buffer is aligned for the
    // largest parameter; each argument is written at its natural offset.
    // Loads and stores go through the type's bit-width form so the move is
    // independent of the argument's value type.
    ptx.push_str("\n    // Marshal child kernel arguments into the buffer\n");
    let mut buf_offset: u32 = 0;
    for (i, ptype) in child.param_types.iter().enumerate() {
        let size = ptx_param_size_bytes(*ptype);
        // Natural alignment: round the running offset up to the type size.
        buf_offset = align_up(buf_offset, size);
        let move_ty = ptx_param_move_type(*ptype);
        let scratch = format!("%arg_scratch_{i}");
        ptx.push_str(&format!(
            "    .reg {move_ty} {scratch};\n\
             ld.param{move_ty} {scratch}, [arg_{i}];\n\
             st.global{move_ty} [%param_buf+{buf_offset}], {scratch};\n",
        ));
        buf_offset = buf_offset.saturating_add(size);
    }

    // Enqueue the child grid on the requested stream. cudaLaunchDeviceV2
    // consumes the formatted parameter buffer produced above and returns a
    // cudaError_t describing whether the grid was successfully enqueued.
    ptx.push_str(&format!(
        "\n    // Launch child kernel {child_name} via the CUDA device runtime\n\
         // cudaError_t cudaLaunchDeviceV2(void *parameterBuffer,\n\
         //                                cudaStream_t stream)\n\
         {{\n\
         .param .b32 retval_ld;\n\
         .param .b64 param0_ld;\n\
         .param .b64 param1_ld;\n\
         st.param.b64 [param0_ld], %param_buf;\n\
         st.param.b64 [param1_ld], %stream;\n\
         call.uni (retval_ld), cudaLaunchDeviceV2, (param0_ld, param1_ld);\n\
         ld.param.b32 %retval, [retval_ld];\n\
         }}\n",
        child_name = child.name,
    ));

    // Store return value and close function
    ptx.push_str(
        "\n$L_launch_done:\n\
         \x20   st.param.s32 [_retval], %retval;\n\
         ret;\n\
         }\n",
    );

    Ok(ptx)
}

/// Returns the byte size of a [`PtxType`] when laid out in a launch
/// parameter buffer.
///
/// The CUDA device runtime parameter buffer stores each kernel argument at
/// its natural size and alignment. This delegates to [`PtxType::size_bytes`]
/// so it stays correct as new types are added to the IR.
fn ptx_param_size_bytes(ty: PtxType) -> u32 {
    // `size_bytes()` returns at most 16, well within `u32` range.
    u32::try_from(ty.size_bytes()).unwrap_or(u32::MAX)
}

/// Returns the bit-width PTX type suffix (`.b8`/`.b16`/`.b32`/`.b64`/`.b128`)
/// used for the `ld.param` / `st.global` pair that copies a kernel argument
/// into the parameter buffer.
///
/// Arguments are moved through their bit-typed form so the copy matches the
/// destination layout regardless of the register's value type (integer,
/// float, packed or sub-word).
fn ptx_param_move_type(ty: PtxType) -> &'static str {
    match ptx_param_size_bytes(ty) {
        0 | 1 => ".b8",
        2 => ".b16",
        3..=4 => ".b32",
        5..=8 => ".b64",
        _ => ".b128",
    }
}

/// Rounds `value` up to the next multiple of `align`.
///
/// `align` is a power-of-two type size produced by [`ptx_param_size_bytes`];
/// an `align` of zero or one leaves `value` unchanged.
fn align_up(value: u32, align: u32) -> u32 {
    if align <= 1 {
        return value;
    }
    value.div_ceil(align).saturating_mul(align)
}

/// Emits the `.extern .func` prototypes for the CUDA device runtime
/// (`cudadevrt`) entry points used by device-side kernel launches.
///
/// The generated module links against `cudadevrt` at JIT/load time. The
/// prototypes use the device-runtime ABI exactly:
///
/// - `cudaGetParameterBufferV2` — reserves a parameter buffer; takes the
///   kernel entry address, the grid and block `dim3` values (each a
///   12-byte, 4-aligned aggregate) and the dynamic shared-memory size,
///   and returns a `void*` buffer pointer.
/// - `cudaLaunchDeviceV2` — enqueues the child grid; takes the formatted
///   parameter buffer and a `cudaStream_t`, and returns a `cudaError_t`.
fn append_device_runtime_externs(ptx: &mut String) {
    ptx.push_str(
        "// CUDA device runtime (cudadevrt) entry points - resolved at link time\n\
         .extern .func (.param .b64 func_retval0) cudaGetParameterBufferV2\n\
         (\n\
         \x20   .param .b64 cudaGetParameterBufferV2_param_0,\n\
         \x20   .param .align 4 .b8 cudaGetParameterBufferV2_param_1[12],\n\
         \x20   .param .align 4 .b8 cudaGetParameterBufferV2_param_2[12],\n\
         \x20   .param .b32 cudaGetParameterBufferV2_param_3\n\
         )\n\
         ;\n\
         .extern .func (.param .b32 func_retval0) cudaLaunchDeviceV2\n\
         (\n\
         \x20   .param .b64 cudaLaunchDeviceV2_param_0,\n\
         \x20   .param .b64 cudaLaunchDeviceV2_param_1\n\
         )\n\
         ;\n\n",
    );
}

/// Generates PTX code for device-side synchronization.
///
/// Produces a `.func` that declares `cudaDeviceSynchronize` as an
/// `.extern .func` prototype using the CUDA device runtime ABI and issues a
/// `call.uni` to it. The real `cudaError_t` it returns is propagated to the
/// caller. The call synchronizes all pending child kernel launches within
/// the current thread's scope; the generated module must be linked against
/// `cudadevrt` at JIT/load time.
///
/// # Arguments
///
/// - `sm` — target architecture for PTX ISA version selection
///
/// # Errors
///
/// Returns [`PtxGenError`] if PTX generation fails.
pub fn generate_device_sync_ptx(sm: SmVersion) -> Result<String, PtxGenError> {
    let (isa_major, isa_minor) = sm.ptx_isa_version();
    let target = sm.as_ptx_str();

    let ptx = format!(
        "// Device-side synchronization\n\
         .version {isa_major}.{isa_minor}\n\
         .target {target}\n\
         .address_size 64\n\
         \n\
         // CUDA device runtime (cudadevrt) entry point — resolved at link time\n\
         // cudaError_t cudaDeviceSynchronize(void)\n\
         .extern .func (.param .b32 func_retval0) cudaDeviceSynchronize\n\
         ;\n\
         \n\
         // cudaDeviceSynchronize() from device code.\n\
         // Blocks the calling thread until every child grid it launched has\n\
         // completed, then returns the device runtime cudaError_t status.\n\
         .func (.param .s32 _retval) __device_synchronize()\n\
         {{\n\
         .reg .s32 %retval;\n\
         \n\
         // Invoke the device runtime synchronization entry point.\n\
         {{\n\
         .param .b32 retval_sync;\n\
         call.uni (retval_sync), cudaDeviceSynchronize, ();\n\
         ld.param.b32 %retval, [retval_sync];\n\
         }}\n\
         \n\
         st.param.s32 [_retval], %retval;\n\
         ret;\n\
         }}\n"
    );

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DynamicParallelismConfig {
        DynamicParallelismConfig::new(SmVersion::Sm80)
    }

    // -- Validation tests --

    #[test]
    fn validate_default_config_ok() {
        let config = default_config();
        assert!(validate_dynamic_config(&config).is_ok());
    }

    #[test]
    fn validate_zero_nesting_depth_fails() {
        let mut config = default_config();
        config.max_nesting_depth = 0;
        let err = validate_dynamic_config(&config);
        assert!(err.is_err());
        let err = err.err();
        assert!(matches!(
            err,
            Some(LaunchError::InvalidDimension {
                dim: "max_nesting_depth",
                ..
            })
        ));
    }

    #[test]
    fn validate_excessive_nesting_depth_fails() {
        let mut config = default_config();
        config.max_nesting_depth = 25;
        let err = validate_dynamic_config(&config);
        assert!(err.is_err());
    }

    #[test]
    fn validate_max_nesting_depth_boundary() {
        let mut config = default_config();
        config.max_nesting_depth = CUDA_MAX_NESTING_DEPTH;
        config.sync_depth = CUDA_MAX_NESTING_DEPTH;
        assert!(validate_dynamic_config(&config).is_ok());
    }

    #[test]
    fn validate_zero_pending_launches_fails() {
        let mut config = default_config();
        config.max_pending_launches = 0;
        assert!(validate_dynamic_config(&config).is_err());
    }

    #[test]
    fn validate_sync_depth_exceeds_nesting_fails() {
        let mut config = default_config();
        config.max_nesting_depth = 4;
        config.sync_depth = 5;
        assert!(validate_dynamic_config(&config).is_err());
    }

    #[test]
    fn validate_zero_child_block_fails() {
        let mut config = default_config();
        config.child_block = Dim3::new(0, 256, 1);
        assert!(validate_dynamic_config(&config).is_err());
    }

    #[test]
    fn validate_zero_child_grid_fails() {
        let mut config = default_config();
        config.child_grid = Dim3::new(128, 0, 1);
        assert!(validate_dynamic_config(&config).is_err());
    }

    #[test]
    fn validate_block_size_exceeds_limit() {
        let mut config = default_config();
        // 32 * 32 * 2 = 2048, exceeds 1024 max
        config.child_block = Dim3::new(32, 32, 2);
        let err = validate_dynamic_config(&config);
        assert!(matches!(
            err,
            Err(LaunchError::BlockSizeExceedsLimit { .. })
        ));
    }

    #[test]
    fn validate_shared_mem_exceeds_limit() {
        let mut config = default_config();
        config.child_shared_mem = 500_000; // exceeds any SM limit
        let err = validate_dynamic_config(&config);
        assert!(matches!(
            err,
            Err(LaunchError::SharedMemoryExceedsLimit { .. })
        ));
    }

    // -- Plan generation tests --

    #[test]
    fn plan_dynamic_launch_ok() {
        let config = default_config();
        let plan = plan_dynamic_launch(&config);
        assert!(plan.is_ok());
        let plan = plan.ok();
        assert!(plan.is_some());
        if let Some(plan) = plan {
            assert!(plan.estimated_child_launches > 0);
            assert!(plan.memory_overhead_bytes > 0);
            assert_eq!(plan.parent_kernel_name, "parent_kernel");
            assert_eq!(plan.child_kernel_name, "child_kernel");
        }
    }

    #[test]
    fn plan_dynamic_launch_invalid_config_fails() {
        let mut config = default_config();
        config.max_nesting_depth = 0;
        let plan = plan_dynamic_launch(&config);
        assert!(plan.is_err());
    }

    #[test]
    fn plan_display() {
        let config = default_config();
        let plan = plan_dynamic_launch(&config);
        if let Ok(plan) = plan {
            let display = format!("{plan}");
            assert!(display.contains("parent_kernel"));
            assert!(display.contains("child_kernel"));
            assert!(display.contains("bytes"));
        }
    }

    // -- Overhead estimation tests --

    #[test]
    fn estimate_overhead_basic() {
        let overhead = estimate_launch_overhead(1, 1);
        assert_eq!(
            overhead,
            BASE_LAUNCH_OVERHEAD_BYTES + PER_DEPTH_OVERHEAD_BYTES
        );
    }

    #[test]
    fn estimate_overhead_default() {
        let overhead = estimate_launch_overhead(4, 2048);
        let expected = BASE_LAUNCH_OVERHEAD_BYTES * 2048 + PER_DEPTH_OVERHEAD_BYTES * 4;
        assert_eq!(overhead, expected);
    }

    #[test]
    fn estimate_overhead_zero() {
        let overhead = estimate_launch_overhead(0, 0);
        assert_eq!(overhead, 0);
    }

    // -- SM nesting tests --

    #[test]
    fn max_nesting_all_sm_versions() {
        assert_eq!(max_nesting_for_sm(SmVersion::Sm75), 24);
        assert_eq!(max_nesting_for_sm(SmVersion::Sm80), 24);
        assert_eq!(max_nesting_for_sm(SmVersion::Sm86), 24);
        assert_eq!(max_nesting_for_sm(SmVersion::Sm89), 24);
        assert_eq!(max_nesting_for_sm(SmVersion::Sm90), 24);
        assert_eq!(max_nesting_for_sm(SmVersion::Sm90a), 24);
        assert_eq!(max_nesting_for_sm(SmVersion::Sm100), 24);
        assert_eq!(max_nesting_for_sm(SmVersion::Sm120), 24);
    }

    // -- PTX generation tests --

    #[test]
    fn generate_child_launch_ptx_basic() {
        let child = ChildKernelSpec {
            name: "child_add".to_string(),
            param_types: vec![PtxType::U64, PtxType::U64, PtxType::U32],
            grid_dim: GridSpec::Fixed(Dim3::x(64)),
            block_dim: Dim3::x(256),
            shared_mem_bytes: 0,
        };
        let result = generate_child_launch_ptx("parent_add", &child, SmVersion::Sm80);
        assert!(result.is_ok());
        let ptx = result.ok();
        assert!(ptx.is_some());
        if let Some(ptx) = ptx {
            assert!(ptx.contains("child_add"));
            assert!(ptx.contains("parent_add"));
            assert!(ptx.contains(".version 7.0"));
            assert!(ptx.contains("sm_80"));
            assert!(ptx.contains("mov.u32 %grid_x, 64"));
            assert!(ptx.contains(".u64"));
            assert!(ptx.contains(".u32"));
        }
    }

    #[test]
    fn generate_child_launch_ptx_emits_device_runtime_externs() {
        // The generated module must declare the cudadevrt entry points so it
        // links against the CUDA device runtime at JIT/load time.
        let child = ChildKernelSpec {
            name: "child_dr".to_string(),
            param_types: vec![PtxType::U64, PtxType::U32],
            grid_dim: GridSpec::Fixed(Dim3::x(32)),
            block_dim: Dim3::x(128),
            shared_mem_bytes: 0,
        };
        let result = generate_child_launch_ptx("parent_dr", &child, SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            // External prototypes with the device-runtime ABI.
            assert!(
                ptx.contains(".extern .func (.param .b64 func_retval0) cudaGetParameterBufferV2")
            );
            assert!(ptx.contains(".extern .func (.param .b32 func_retval0) cudaLaunchDeviceV2"));
            // dim3 grid/block aggregates are 12-byte, 4-aligned params.
            assert!(ptx.contains(".param .align 4 .b8 cudaGetParameterBufferV2_param_1[12]"));
            assert!(ptx.contains(".param .align 4 .b8 cudaGetParameterBufferV2_param_2[12]"));
            // Real call.uni invocations into the device runtime.
            assert!(ptx.contains("call.uni (retval_pb), cudaGetParameterBufferV2"));
            assert!(
                ptx.contains("call.uni (retval_ld), cudaLaunchDeviceV2, (param0_ld, param1_ld);")
            );
        }
    }

    #[test]
    fn generate_child_launch_ptx_sets_up_parameter_buffer() {
        // The parameter buffer must be obtained, NULL-checked, and have the
        // child arguments marshalled into it before the launch.
        let child = ChildKernelSpec {
            name: "child_buf".to_string(),
            param_types: vec![PtxType::U64, PtxType::U64, PtxType::F32],
            grid_dim: GridSpec::Fixed(Dim3::x(16)),
            block_dim: Dim3::x(64),
            shared_mem_bytes: 256,
        };
        let result = generate_child_launch_ptx("parent_buf", &child, SmVersion::Sm90);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            // Parameter-buffer plumbing.
            assert!(ptx.contains(".reg .u64 %func_addr, %param_buf;"));
            assert!(ptx.contains("mov.u64 %func_addr, child_buf;"));
            assert!(ptx.contains("ld.param.b64 %param_buf, [retval_pb];"));
            // grid / block / smem are stored into the buffer-allocation params.
            assert!(ptx.contains("st.param.b32 [param1_pb+0], %grid_x;"));
            assert!(ptx.contains("st.param.b32 [param2_pb+8], %block_z;"));
            assert!(ptx.contains("st.param.b32 [param3_pb], %shared_mem;"));
            // NULL parameter buffer bails out with the pending-count error.
            assert!(ptx.contains("setp.eq.u64 %p_buf_null, %param_buf, 0;"));
            assert!(ptx.contains("@%p_buf_null mov.s32 %retval, 209;"));
            // Arguments marshalled at natural offsets: u64,u64,f32 -> 0,8,16.
            assert!(ptx.contains("st.global.b64 [%param_buf+0],"));
            assert!(ptx.contains("st.global.b64 [%param_buf+8],"));
            assert!(ptx.contains("st.global.b32 [%param_buf+16],"));
            // The launch return code is captured into %retval.
            assert!(ptx.contains("ld.param.b32 %retval, [retval_ld];"));
        }
    }

    #[test]
    fn generate_child_launch_ptx_no_stub_comments() {
        // The stale placeholder comments must be gone once implemented.
        let child = ChildKernelSpec {
            name: "child_clean".to_string(),
            param_types: vec![PtxType::U64],
            grid_dim: GridSpec::Fixed(Dim3::x(8)),
            block_dim: Dim3::x(32),
            shared_mem_bytes: 0,
        };
        let result = generate_child_launch_ptx("parent_clean", &child, SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(!ptx.contains("We model this with"));
            assert!(!ptx.contains("actual device-side launch uses cudaLaunchDeviceV2"));
            assert!(!ptx.contains("mov.s32 %retval, 0; // cudaSuccess"));
        }
    }

    #[test]
    fn generate_child_launch_ptx_aligns_mixed_arguments() {
        // A u32 followed by a u64 must pad the u64 to an 8-byte offset.
        let child = ChildKernelSpec {
            name: "child_align".to_string(),
            param_types: vec![PtxType::U32, PtxType::U64],
            grid_dim: GridSpec::Fixed(Dim3::x(4)),
            block_dim: Dim3::x(32),
            shared_mem_bytes: 0,
        };
        let result = generate_child_launch_ptx("parent_align", &child, SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("st.global.b32 [%param_buf+0],"));
            // u64 padded from offset 4 up to its natural 8-byte boundary.
            assert!(ptx.contains("st.global.b64 [%param_buf+8],"));
        }
    }

    #[test]
    fn generate_child_launch_ptx_data_dependent() {
        let child = ChildKernelSpec {
            name: "child_scale".to_string(),
            param_types: vec![PtxType::U64, PtxType::U32],
            grid_dim: GridSpec::DataDependent { param_index: 1 },
            block_dim: Dim3::x(128),
            shared_mem_bytes: 1024,
        };
        let result = generate_child_launch_ptx("parent_scale", &child, SmVersion::Sm90);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("Data-dependent"));
            assert!(ptx.contains("arg_1"));
            assert!(ptx.contains("div.u32"));
        }
    }

    #[test]
    fn generate_child_launch_ptx_thread_dependent() {
        let child = ChildKernelSpec {
            name: "child_per_thread".to_string(),
            param_types: vec![PtxType::U64],
            grid_dim: GridSpec::ThreadDependent,
            block_dim: Dim3::x(32),
            shared_mem_bytes: 0,
        };
        let result = generate_child_launch_ptx("parent", &child, SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("Thread-dependent"));
            assert!(ptx.contains("%tid.x"));
        }
    }

    #[test]
    fn generate_child_launch_ptx_empty_name_fails() {
        let child = ChildKernelSpec {
            name: String::new(),
            param_types: vec![],
            grid_dim: GridSpec::Fixed(Dim3::x(1)),
            block_dim: Dim3::x(1),
            shared_mem_bytes: 0,
        };
        let result = generate_child_launch_ptx("parent", &child, SmVersion::Sm80);
        assert!(result.is_err());
    }

    #[test]
    fn generate_child_launch_ptx_zero_block_fails() {
        let child = ChildKernelSpec {
            name: "child".to_string(),
            param_types: vec![],
            grid_dim: GridSpec::Fixed(Dim3::x(1)),
            block_dim: Dim3::new(0, 1, 1),
            shared_mem_bytes: 0,
        };
        let result = generate_child_launch_ptx("parent", &child, SmVersion::Sm80);
        assert!(result.is_err());
    }

    #[test]
    fn generate_device_sync_ptx_basic() {
        let result = generate_device_sync_ptx(SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains("__device_synchronize"));
            assert!(ptx.contains(".version 7.0"));
            assert!(ptx.contains("sm_80"));
            assert!(ptx.contains("cudaDeviceSynchronize"));
        }
    }

    #[test]
    fn generate_device_sync_ptx_hopper() {
        let result = generate_device_sync_ptx(SmVersion::Sm90);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains(".version 8.0"));
            assert!(ptx.contains("sm_90"));
        }
    }

    #[test]
    fn generate_device_sync_ptx_emits_real_extern_and_call() {
        // The sync helper must declare the cudadevrt entry point and call it.
        let result = generate_device_sync_ptx(SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(ptx.contains(".extern .func (.param .b32 func_retval0) cudaDeviceSynchronize"));
            assert!(ptx.contains("call.uni (retval_sync), cudaDeviceSynchronize, ();"));
            // The real return code is stored into %retval.
            assert!(ptx.contains("ld.param.b32 %retval, [retval_sync];"));
        }
    }

    #[test]
    fn generate_device_sync_ptx_no_stub_comments() {
        // The stale placeholder comments must be gone once implemented.
        let result = generate_device_sync_ptx(SmVersion::Sm80);
        assert!(result.is_ok());
        if let Ok(ptx) = result {
            assert!(!ptx.contains("(placeholder)"));
            assert!(!ptx.contains("For code generation, we emit the call pattern"));
            assert!(!ptx.contains("mov.s32 %retval, 0;"));
        }
    }

    #[test]
    fn ptx_param_layout_helpers() {
        // Buffer size matches PtxType::size_bytes for representative types.
        assert_eq!(ptx_param_size_bytes(PtxType::U8), 1);
        assert_eq!(ptx_param_size_bytes(PtxType::F16), 2);
        assert_eq!(ptx_param_size_bytes(PtxType::U32), 4);
        assert_eq!(ptx_param_size_bytes(PtxType::U64), 8);
        assert_eq!(ptx_param_size_bytes(PtxType::B128), 16);
        // Bit-width move type selection.
        assert_eq!(ptx_param_move_type(PtxType::S8), ".b8");
        assert_eq!(ptx_param_move_type(PtxType::BF16), ".b16");
        assert_eq!(ptx_param_move_type(PtxType::F32), ".b32");
        assert_eq!(ptx_param_move_type(PtxType::F64), ".b64");
        assert_eq!(ptx_param_move_type(PtxType::B128), ".b128");
        // align_up rounds to the next multiple, identity below 2.
        assert_eq!(align_up(0, 8), 0);
        assert_eq!(align_up(4, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 4), 12);
        assert_eq!(align_up(7, 1), 7);
    }

    // -- Display tests --

    #[test]
    fn config_display() {
        let config = default_config();
        let display = format!("{config}");
        assert!(display.contains("depth=4"));
        assert!(display.contains("pending=2048"));
        assert!(display.contains("sync@2"));
        assert!(display.contains("sm_80"));
    }

    #[test]
    fn grid_spec_display() {
        assert_eq!(format!("{}", GridSpec::Fixed(Dim3::x(64))), "Fixed(64)");
        assert_eq!(
            format!("{}", GridSpec::DataDependent { param_index: 2 }),
            "DataDependent(param[2])"
        );
        assert_eq!(format!("{}", GridSpec::ThreadDependent), "ThreadDependent");
    }
}
