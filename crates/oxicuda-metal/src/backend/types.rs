//! `MetalBackend` struct, intrinsic helpers, and Metal-API dispatch helpers.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use oxicuda_backend::{BackendError, BackendResult, BackendTranspose, BinaryOp, ReduceOp, UnaryOp};

use crate::{device::MetalDevice, memory::MetalMemoryManager, pipeline::MetalComputePipeline};

#[cfg(target_os = "macos")]
use super::functions::next_power_of_2;

/// Apple Metal GPU compute backend.
///
/// On macOS this selects the system-default Metal device and allocates
/// shared-memory buffers that are directly accessible from both CPU and GPU.
///
/// On non-macOS platforms every operation returns
/// [`BackendError::DeviceError`] (wrapping [`crate::error::MetalError::UnsupportedPlatform`]).
///
/// # Lifecycle
///
/// 1. `MetalBackend::new()` — create an uninitialised backend.
/// 2. `init()` — acquire the Metal device and set up the memory manager.
/// 3. Use `alloc`, `copy_htod`, compute ops, `copy_dtoh`, `free`.
/// 4. `synchronize()` — wait for all pending GPU work to finish.
#[derive(Debug)]
pub struct MetalBackend {
    pub(super) device: Option<Arc<MetalDevice>>,
    pub(super) memory: Option<Arc<MetalMemoryManager>>,
    pub(super) initialized: bool,
    /// Cache of compiled custom-MSL pipelines keyed by
    /// `(function_name, msl-source-hash)`, so repeated
    /// [`launch_custom_kernel`](MetalBackend::launch_custom_kernel) calls reuse
    /// the compiled pipeline and its command queue instead of recompiling.
    pub(super) pipeline_cache: Mutex<HashMap<(String, u64), Arc<MetalComputePipeline>>>,
}
impl MetalBackend {
    /// Create a new, uninitialised Metal backend.
    pub fn new() -> Self {
        Self {
            device: None,
            memory: None,
            initialized: false,
            pipeline_cache: Mutex::new(HashMap::new()),
        }
    }
    /// Return an error if the backend has not been initialised yet.
    pub(super) fn check_init(&self) -> BackendResult<()> {
        if self.initialized {
            Ok(())
        } else {
            Err(BackendError::NotInitialized)
        }
    }
    /// Convenience accessor: get the memory manager or return `NotInitialized`.
    pub(super) fn memory(&self) -> BackendResult<&Arc<MetalMemoryManager>> {
        self.memory.as_ref().ok_or(BackendError::NotInitialized)
    }

    /// Compile (or reuse a cached) compute pipeline from `msl_source` /
    /// `function_name` and dispatch it over `total_threads` GPU threads (1-D).
    ///
    /// Device-buffer `handles` bind to `buffer(0)`, `buffer(1)`, … in order; each
    /// `scalar_bytes` blob binds with `set_bytes` to the indices immediately
    /// following the buffers (`buffer(handles.len())`, …). This lets callers pass,
    /// for example, an element count as a `u32` and clamp constants as `f32`, each
    /// encoded as raw little-endian bytes.
    ///
    /// Compiled pipelines are cached by `(function_name, msl-source-hash)`, so
    /// repeating a call with the same kernel reuses the pipeline and command
    /// queue rather than recompiling.
    ///
    /// The dispatch is synchronous: it waits for GPU completion before returning.
    /// Because the grid is rounded up to whole threadgroups, the kernel **must**
    /// bounds-check its `thread_position_in_grid` against the element count.
    ///
    /// # Errors
    /// * [`BackendError::NotInitialized`] if [`init`](oxicuda_backend::ComputeBackend::init)
    ///   has not been called — always the case on non-macOS, where Metal is
    ///   unavailable.
    /// * [`BackendError::DeviceError`] if MSL compilation or pipeline creation fails.
    /// * [`BackendError::InvalidArgument`] for an unknown buffer handle or an
    ///   empty `scalar_bytes` entry.
    pub fn launch_custom_kernel(
        &self,
        msl_source: &str,
        function_name: &str,
        handles: &[u64],
        scalar_bytes: &[&[u8]],
        total_threads: usize,
    ) -> BackendResult<()> {
        self.check_init()?;
        if total_threads == 0 {
            return Ok(());
        }
        let pipeline = self.custom_pipeline(msl_source, function_name)?;
        let memory = self.memory()?;
        pipeline
            .dispatch(memory, handles, scalar_bytes, total_threads)
            .map_err(BackendError::from)
    }

    /// Fetch a cached compiled pipeline for `(function_name, msl_source)`,
    /// compiling and caching it on first use.
    fn custom_pipeline(
        &self,
        msl_source: &str,
        function_name: &str,
    ) -> BackendResult<Arc<MetalComputePipeline>> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        msl_source.hash(&mut hasher);
        let key = (function_name.to_string(), hasher.finish());
        let mut cache = self
            .pipeline_cache
            .lock()
            .map_err(|_| BackendError::DeviceError("pipeline cache mutex poisoned".into()))?;
        if let Some(existing) = cache.get(&key) {
            return Ok(Arc::clone(existing));
        }
        let device = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
        let pipeline = Arc::new(
            MetalComputePipeline::new(device, msl_source, function_name)
                .map_err(BackendError::from)?,
        );
        cache.insert(key, Arc::clone(&pipeline));
        Ok(pipeline)
    }
}
#[cfg(target_os = "macos")]
impl MetalBackend {
    pub(super) fn dispatch_unary(
        &self,
        op: UnaryOp,
        input_ptr: u64,
        output_ptr: u64,
        n: usize,
    ) -> BackendResult<()> {
        let op_str = match op {
            UnaryOp::Relu => "relu",
            UnaryOp::Sigmoid => "sigmoid",
            UnaryOp::Tanh => "tanh",
            UnaryOp::Exp => "exp",
            UnaryOp::Log => "log",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Abs => "abs",
            UnaryOp::Neg => "neg",
        };
        let device = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
        let memory = self.memory()?;
        let msl = crate::msl::elementwise_msl(op_str);
        let pipeline = crate::pipeline::MetalComputePipeline::new(device, &msl, "elementwise_f32")
            .map_err(BackendError::from)?;
        let buffers = memory.lock_buffers().map_err(BackendError::from)?;
        let input_info = buffers.get(&input_ptr).ok_or_else(|| {
            BackendError::InvalidArgument(format!("unknown input handle {input_ptr}"))
        })?;
        let output_info = buffers.get(&output_ptr).ok_or_else(|| {
            BackendError::InvalidArgument(format!("unknown output handle {output_ptr}"))
        })?;
        let command_buffer = pipeline.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&input_info.buffer), 0);
        encoder.set_buffer(1, Some(&output_info.buffer), 0);
        let count = n as u32;
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const std::ffi::c_void,
        );
        let tg_size = 256u64.min(n as u64);
        let groups = (n as u64).div_ceil(tg_size);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(groups, 1, 1),
            metal::MTLSize::new(tg_size, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }
    pub(super) fn dispatch_binary(
        &self,
        op: BinaryOp,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
        n: usize,
    ) -> BackendResult<()> {
        let op_str = match op {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Max => "max",
            BinaryOp::Min => "min",
        };
        let device = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
        let memory = self.memory()?;
        let msl = crate::msl::binary_msl(op_str);
        let pipeline = crate::pipeline::MetalComputePipeline::new(device, &msl, "binary_f32")
            .map_err(BackendError::from)?;
        let buffers = memory.lock_buffers().map_err(BackendError::from)?;
        let a_info = buffers
            .get(&a_ptr)
            .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {a_ptr}")))?;
        let b_info = buffers
            .get(&b_ptr)
            .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {b_ptr}")))?;
        let out_info = buffers.get(&output_ptr).ok_or_else(|| {
            BackendError::InvalidArgument(format!("unknown output handle {output_ptr}"))
        })?;
        let command_buffer = pipeline.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&a_info.buffer), 0);
        encoder.set_buffer(1, Some(&b_info.buffer), 0);
        encoder.set_buffer(2, Some(&out_info.buffer), 0);
        let count = n as u32;
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const std::ffi::c_void,
        );
        let tg_size = 256u64.min(n as u64);
        let groups = (n as u64).div_ceil(tg_size);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(groups, 1, 1),
            metal::MTLSize::new(tg_size, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }
    pub(super) fn dispatch_reduce(
        &self,
        op: ReduceOp,
        input_ptr: u64,
        output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()> {
        let op_str = match op {
            ReduceOp::Sum => "sum",
            ReduceOp::Max => "max",
            ReduceOp::Min => "min",
            ReduceOp::Mean => "mean",
        };
        let device = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
        let memory = self.memory()?;
        let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
        let reduce_size = shape[axis];
        let inner_size: usize = shape[axis + 1..].iter().product::<usize>().max(1);
        let msl = crate::msl::reduction_msl(op_str);
        if msl.is_empty() {
            return Err(BackendError::Unsupported(format!(
                "Metal reduction op '{op_str}' not supported"
            )));
        }
        let fn_name = crate::msl::reduction_function_name(op_str);
        let pipeline = crate::pipeline::MetalComputePipeline::new(device, &msl, fn_name)
            .map_err(BackendError::from)?;
        let buffers = memory.lock_buffers().map_err(BackendError::from)?;
        let input_info = buffers.get(&input_ptr).ok_or_else(|| {
            BackendError::InvalidArgument(format!("unknown input handle {input_ptr}"))
        })?;
        let out_info = buffers.get(&output_ptr).ok_or_else(|| {
            BackendError::InvalidArgument(format!("unknown output handle {output_ptr}"))
        })?;
        let command_buffer = pipeline.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&input_info.buffer), 0);
        encoder.set_buffer(1, Some(&out_info.buffer), 0);
        let outer_u32 = outer_size as u32;
        let reduce_u32 = reduce_size as u32;
        let inner_u32 = inner_size as u32;
        encoder.set_bytes(2, 4, &outer_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(3, 4, &reduce_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &inner_u32 as *const u32 as *const std::ffi::c_void);
        let tg_size = next_power_of_2(reduce_size).min(256) as u64;
        encoder.set_threadgroup_memory_length(0, tg_size * std::mem::size_of::<f32>() as u64);
        let total_groups = (outer_size * inner_size) as u64;
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(total_groups, 1, 1),
            metal::MTLSize::new(tg_size, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_gemm(
        &self,
        _trans_a: BackendTranspose,
        _trans_b: BackendTranspose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a_ptr: u64,
        _lda: usize,
        b_ptr: u64,
        _ldb: usize,
        beta: f64,
        c_ptr: u64,
        _ldc: usize,
    ) -> BackendResult<()> {
        let device = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
        let memory = self.memory()?;
        let msl = crate::msl::gemm_msl();
        let pipeline = crate::pipeline::MetalComputePipeline::new(device, msl, "gemm_f32")
            .map_err(BackendError::from)?;
        let buffers = memory.lock_buffers().map_err(BackendError::from)?;
        let a_info = buffers
            .get(&a_ptr)
            .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {a_ptr}")))?;
        let b_info = buffers
            .get(&b_ptr)
            .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {b_ptr}")))?;
        let c_info = buffers
            .get(&c_ptr)
            .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {c_ptr}")))?;
        #[repr(C)]
        struct GemmParams {
            m: u32,
            n: u32,
            k: u32,
            alpha: f32,
            beta: f32,
        }
        let params = GemmParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            alpha: alpha as f32,
            beta: beta as f32,
        };
        let command_buffer = pipeline.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&a_info.buffer), 0);
        encoder.set_buffer(1, Some(&b_info.buffer), 0);
        encoder.set_buffer(2, Some(&c_info.buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<GemmParams>() as u64,
            &params as *const GemmParams as *const std::ffi::c_void,
        );
        let tg_w = 16u64;
        let tg_h = 16u64;
        let groups_x = (n as u64).div_ceil(tg_w);
        let groups_y = (m as u64).div_ceil(tg_h);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(groups_x, groups_y, 1),
            metal::MTLSize::new(tg_w, tg_h, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_batched_gemm(
        &self,
        _trans_a: BackendTranspose,
        _trans_b: BackendTranspose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a_ptr: u64,
        _lda: usize,
        stride_a: usize,
        b_ptr: u64,
        _ldb: usize,
        stride_b: usize,
        beta: f64,
        c_ptr: u64,
        _ldc: usize,
        stride_c: usize,
        batch_count: usize,
    ) -> BackendResult<()> {
        let device = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
        let memory = self.memory()?;
        let msl = crate::msl::batched_gemm_msl();
        let pipeline = crate::pipeline::MetalComputePipeline::new(device, msl, "batched_gemm_f32")
            .map_err(BackendError::from)?;
        let buffers = memory.lock_buffers().map_err(BackendError::from)?;
        let a_info = buffers
            .get(&a_ptr)
            .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {a_ptr}")))?;
        let b_info = buffers
            .get(&b_ptr)
            .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {b_ptr}")))?;
        let c_info = buffers
            .get(&c_ptr)
            .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {c_ptr}")))?;
        #[repr(C)]
        struct BatchedGemmParams {
            m: u32,
            n: u32,
            k: u32,
            alpha: f32,
            beta: f32,
            batch_count: u32,
            stride_a: u32,
            stride_b: u32,
            stride_c: u32,
        }
        let params = BatchedGemmParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            alpha: alpha as f32,
            beta: beta as f32,
            batch_count: batch_count as u32,
            stride_a: stride_a as u32,
            stride_b: stride_b as u32,
            stride_c: stride_c as u32,
        };
        let command_buffer = pipeline.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&a_info.buffer), 0);
        encoder.set_buffer(1, Some(&b_info.buffer), 0);
        encoder.set_buffer(2, Some(&c_info.buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<BatchedGemmParams>() as u64,
            &params as *const BatchedGemmParams as *const std::ffi::c_void,
        );
        let tg_w = 16u64;
        let tg_h = 16u64;
        let groups_x = (n as u64).div_ceil(tg_w);
        let groups_y = (m as u64).div_ceil(tg_h);
        let groups_z = batch_count as u64;
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(groups_x, groups_y, groups_z),
            metal::MTLSize::new(tg_w, tg_h, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }
    /// Half-precision GEMM: `C = alpha * A * B + beta * C` using FP16 storage.
    ///
    /// This is an inherent method (not on the `ComputeBackend` trait) since
    /// the trait operates on f32/f64 data. Element size is 2 bytes (half).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16(
        &self,
        _trans_a: BackendTranspose,
        _trans_b: BackendTranspose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a_ptr: u64,
        _lda: usize,
        b_ptr: u64,
        _ldb: usize,
        beta: f32,
        c_ptr: u64,
        _ldc: usize,
    ) -> BackendResult<()> {
        self.check_init()?;
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        let device = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
        let memory = self.memory()?;
        let msl = crate::msl::gemm_msl_f16();
        let pipeline = crate::pipeline::MetalComputePipeline::new(device, msl, "gemm_f16")
            .map_err(BackendError::from)?;
        let buffers = memory.lock_buffers().map_err(BackendError::from)?;
        let a_info = buffers
            .get(&a_ptr)
            .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {a_ptr}")))?;
        let b_info = buffers
            .get(&b_ptr)
            .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {b_ptr}")))?;
        let c_info = buffers
            .get(&c_ptr)
            .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {c_ptr}")))?;
        #[repr(C)]
        struct GemmParamsF16 {
            m: u32,
            n: u32,
            k: u32,
            alpha: f32,
            beta: f32,
        }
        let params = GemmParamsF16 {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            alpha,
            beta,
        };
        let command_buffer = pipeline.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.pipeline_state);
        encoder.set_buffer(0, Some(&a_info.buffer), 0);
        encoder.set_buffer(1, Some(&b_info.buffer), 0);
        encoder.set_buffer(2, Some(&c_info.buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<GemmParamsF16>() as u64,
            &params as *const GemmParamsF16 as *const std::ffi::c_void,
        );
        let tg_w = 16u64;
        let tg_h = 16u64;
        let groups_x = (n as u64).div_ceil(tg_w);
        let groups_y = (m as u64).div_ceil(tg_h);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(groups_x, groups_y, 1),
            metal::MTLSize::new(tg_w, tg_h, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }
}
#[cfg(not(target_os = "macos"))]
impl MetalBackend {
    pub(super) fn dispatch_unary(
        &self,
        _op: UnaryOp,
        _input_ptr: u64,
        _output_ptr: u64,
        _n: usize,
    ) -> BackendResult<()> {
        Err(BackendError::DeviceError("Metal requires macOS".into()))
    }
    pub(super) fn dispatch_binary(
        &self,
        _op: BinaryOp,
        _a_ptr: u64,
        _b_ptr: u64,
        _output_ptr: u64,
        _n: usize,
    ) -> BackendResult<()> {
        Err(BackendError::DeviceError("Metal requires macOS".into()))
    }
    pub(super) fn dispatch_reduce(
        &self,
        _op: ReduceOp,
        _input_ptr: u64,
        _output_ptr: u64,
        _shape: &[usize],
        _axis: usize,
    ) -> BackendResult<()> {
        Err(BackendError::DeviceError("Metal requires macOS".into()))
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_gemm(
        &self,
        _trans_a: BackendTranspose,
        _trans_b: BackendTranspose,
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f64,
        _a_ptr: u64,
        _lda: usize,
        _b_ptr: u64,
        _ldb: usize,
        _beta: f64,
        _c_ptr: u64,
        _ldc: usize,
    ) -> BackendResult<()> {
        Err(BackendError::DeviceError("Metal requires macOS".into()))
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch_batched_gemm(
        &self,
        _trans_a: BackendTranspose,
        _trans_b: BackendTranspose,
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f64,
        _a_ptr: u64,
        _lda: usize,
        _stride_a: usize,
        _b_ptr: u64,
        _ldb: usize,
        _stride_b: usize,
        _beta: f64,
        _c_ptr: u64,
        _ldc: usize,
        _stride_c: usize,
        _batch_count: usize,
    ) -> BackendResult<()> {
        Err(BackendError::DeviceError("Metal requires macOS".into()))
    }
    /// Half-precision GEMM: `C = alpha * A * B + beta * C` using FP16 storage.
    ///
    /// This is an inherent method (not on the `ComputeBackend` trait) since
    /// the trait operates on f32/f64 data. Element size is 2 bytes (half).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16(
        &self,
        _trans_a: BackendTranspose,
        _trans_b: BackendTranspose,
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f32,
        _a_ptr: u64,
        _lda: usize,
        _b_ptr: u64,
        _ldb: usize,
        _beta: f32,
        _c_ptr: u64,
        _ldc: usize,
    ) -> BackendResult<()> {
        self.check_init()?;
        Err(BackendError::DeviceError("Metal requires macOS".into()))
    }
}
