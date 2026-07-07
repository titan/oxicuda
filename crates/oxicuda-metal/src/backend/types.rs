//! `MetalBackend` struct, intrinsic helpers, and Metal-API dispatch helpers.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use oxicuda_backend::{BackendError, BackendResult, BackendTranspose, BinaryOp, ReduceOp, UnaryOp};

use crate::{device::MetalDevice, memory::MetalMemoryManager, pipeline::MetalComputePipeline};

#[cfg(target_os = "macos")]
use super::functions::next_power_of_2;

/// Validate that a GEMM request uses a layout the Metal reference kernels
/// actually honour.
///
/// The MSL GEMM kernels ([`crate::msl::gemm_msl`], [`crate::msl::batched_gemm_msl`],
/// [`crate::msl::gemm_msl_f16`]) index the operands row-major with the leading
/// dimension pinned to `k`/`n` (`a[row*k+i]`, `b[i*n+col]`, `c[row*n+col]`) and
/// have no notion of a transpose mode. Honouring the `trans_*`/`ld*` arguments
/// would require threading them into the shader; until that exists we must
/// **reject** any call that would otherwise be silently mis-computed rather than
/// return a wrong result. Only the contiguous, non-transposed natural layout
/// (`lda == k`, `ldb == n`, `ldc == n`) is accepted.
pub(super) fn validate_gemm_layout(
    trans_a: BackendTranspose,
    trans_b: BackendTranspose,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) -> BackendResult<()> {
    if trans_a != BackendTranspose::NoTrans || trans_b != BackendTranspose::NoTrans {
        return Err(BackendError::Unsupported(
            "Metal GEMM supports only NoTrans operands; transpose modes are not implemented".into(),
        ));
    }
    if lda != k || ldb != n || ldc != n {
        return Err(BackendError::Unsupported(format!(
            "Metal GEMM supports only contiguous natural leading dimensions \
             (expected lda={k}, ldb={n}, ldc={n}); got lda={lda}, ldb={ldb}, ldc={ldc}"
        )));
    }
    Ok(())
}

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

    /// Report whether `handle` refers to an imported (external) buffer.
    ///
    /// Returns `Some(true)` for a handle created by
    /// [`register_external`](Self::register_external) /
    /// [`import_buffer`](Self::import_buffer), `Some(false)` for an
    /// [`alloc`](oxicuda_backend::ComputeBackend::alloc)-owned handle, and
    /// `None` if the handle is unknown (e.g. already freed). Useful for
    /// asserting that a cache-backed buffer was imported (so `free` will not
    /// deallocate it).
    ///
    /// # Errors
    /// [`BackendError::NotInitialized`] if the backend is not initialised.
    pub fn is_imported(&self, handle: u64) -> BackendResult<Option<bool>> {
        self.check_init()?;
        self.memory()?
            .is_external(handle)
            .map_err(BackendError::from)
    }

    /// Copy `len_bytes` from device buffer `src` to device buffer `dst`
    /// **device-to-device**, with no host round-trip. Both handles may be
    /// [`alloc`](oxicuda_backend::ComputeBackend::alloc)-owned or imported, in
    /// any combination.
    ///
    /// Useful alongside [`register_external`](Self::register_external) for
    /// keeping data GPU-resident — e.g. copying a freshly computed result into a
    /// consumer's cached buffer without round-tripping through host memory.
    ///
    /// # Errors
    /// * [`BackendError::NotInitialized`] if the backend is not initialised.
    /// * [`BackendError::InvalidArgument`] for an unknown handle, if
    ///   `src == dst`, or if `len_bytes` exceeds either buffer's length.
    pub fn copy_dtod(&self, dst: u64, src: u64, len_bytes: usize) -> BackendResult<()> {
        self.check_init()?;
        if len_bytes == 0 {
            return Ok(());
        }
        self.memory()?
            .copy_device_to_device(dst, src, len_bytes)
            .map_err(BackendError::from)
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
    /// Register an **existing, externally owned** `metal::Buffer` as an oxicuda
    /// handle so compute ops (`gemm`, `copy_*`, custom kernels) can run on it
    /// **zero-copy**, with no host round-trip.
    ///
    /// This is the recommended import entry point for callers that keep their
    /// buffers in a cache (e.g. an `Arc<metal::Buffer>`): pass a reference and
    /// oxicuda takes its **own independent retain** for the lifetime of the
    /// handle. Ownership stays with the caller —
    /// [`free`](oxicuda_backend::ComputeBackend::free) and dropping the backend
    /// release only oxicuda's retain and **never deallocate the caller's
    /// buffer**.
    ///
    /// The returned `u64` handle is interchangeable with one from
    /// [`alloc`](oxicuda_backend::ComputeBackend::alloc) and may be mixed freely
    /// with oxicuda-owned handles in the same op (e.g. external A·B → owned C, or
    /// fully external A·B·C).
    ///
    /// `len_bytes` is the logical length the handle exposes (used to bound host
    /// copies); it must not exceed `buffer.length()`. The buffer **must** belong
    /// to the same `metal::Device` oxicuda initialised with.
    ///
    /// # Errors
    /// * [`BackendError::NotInitialized`] if [`init`](oxicuda_backend::ComputeBackend::init)
    ///   has not been called.
    /// * [`BackendError::InvalidArgument`] if `len_bytes` exceeds the buffer's
    ///   physical length.
    pub fn register_external(
        &self,
        buffer: &metal::Buffer,
        len_bytes: usize,
    ) -> BackendResult<u64> {
        self.check_init()?;
        self.memory()?
            .import_external(buffer, len_bytes)
            .map_err(BackendError::from)
    }

    /// Import an external `metal::Buffer` **by value**, returning a zero-copy
    /// handle. Convenience wrapper over [`register_external`](Self::register_external).
    ///
    /// oxicuda holds the buffer alive for the handle's lifetime via its own
    /// retain; the moved-in value is released once this call returns (its retain
    /// is balanced by the independent retain oxicuda takes). Callers that need to
    /// keep their own reference should clone first or use
    /// [`register_external`](Self::register_external) with a borrow. As with
    /// `register_external`, [`free`](oxicuda_backend::ComputeBackend::free) /
    /// backend drop never deallocate memory still referenced by the caller.
    ///
    /// # Errors
    /// Same as [`register_external`](Self::register_external).
    pub fn import_buffer(&self, buffer: metal::Buffer, len_bytes: usize) -> BackendResult<u64> {
        // Borrow for registration; `buffer` is dropped (its retain released) at
        // end of scope, leaving oxicuda's own independent retain in the map.
        self.register_external(&buffer, len_bytes)
    }

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
        let memory = self.memory()?;
        let msl = crate::msl::elementwise_msl(op_str);
        // Reuse a cached compiled pipeline (keyed on function name + MSL source
        // hash) instead of recompiling the shader + creating a command queue on
        // every call.
        let pipeline = self.custom_pipeline(&msl, "elementwise_f32")?;
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
        let memory = self.memory()?;
        let msl = crate::msl::binary_msl(op_str);
        // Reuse a cached compiled pipeline instead of recompiling per call.
        let pipeline = self.custom_pipeline(&msl, "binary_f32")?;
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
        // Reuse a cached compiled pipeline instead of recompiling per call.
        let pipeline = self.custom_pipeline(&msl, fn_name)?;
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
        let memory = self.memory()?;
        let msl = crate::msl::gemm_msl();
        // Reuse a cached compiled pipeline instead of recompiling per call.
        let pipeline = self.custom_pipeline(msl, "gemm_f32")?;
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
        let memory = self.memory()?;
        let msl = crate::msl::batched_gemm_msl();
        // Reuse a cached compiled pipeline instead of recompiling per call.
        let pipeline = self.custom_pipeline(msl, "batched_gemm_f32")?;
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
        trans_a: BackendTranspose,
        trans_b: BackendTranspose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a_ptr: u64,
        lda: usize,
        b_ptr: u64,
        ldb: usize,
        beta: f32,
        c_ptr: u64,
        ldc: usize,
    ) -> BackendResult<()> {
        self.check_init()?;
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        validate_gemm_layout(trans_a, trans_b, n, k, lda, ldb, ldc)?;
        let memory = self.memory()?;
        let msl = crate::msl::gemm_msl_f16();
        let pipeline = self.custom_pipeline(msl, "gemm_f16")?;
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

#[cfg(test)]
mod tests {
    use super::validate_gemm_layout;
    use oxicuda_backend::{BackendError, BackendTranspose};

    #[test]
    fn natural_layout_accepted() {
        // NoTrans/NoTrans with lda=k, ldb=n, ldc=n is the one supported layout.
        assert!(
            validate_gemm_layout(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                /*n*/ 4,
                /*k*/ 3,
                /*lda*/ 3,
                /*ldb*/ 4,
                /*ldc*/ 4,
            )
            .is_ok()
        );
    }

    #[test]
    fn transpose_b_rejected() {
        // The common `y = x @ W.T` (trans_b = Trans) op must be rejected loudly
        // rather than silently mis-computed.
        let err = validate_gemm_layout(
            BackendTranspose::NoTrans,
            BackendTranspose::Trans,
            4,
            3,
            3,
            4,
            4,
        )
        .unwrap_err();
        assert!(matches!(err, BackendError::Unsupported(_)));
    }

    #[test]
    fn transpose_a_rejected() {
        let err = validate_gemm_layout(
            BackendTranspose::Trans,
            BackendTranspose::NoTrans,
            4,
            3,
            3,
            4,
            4,
        )
        .unwrap_err();
        assert!(matches!(err, BackendError::Unsupported(_)));
    }

    #[test]
    fn strided_leading_dims_rejected() {
        // A strided sub-matrix view (lda > k) reads the wrong elements in the
        // row-major kernel, so it must be rejected.
        assert!(matches!(
            validate_gemm_layout(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                4,
                3,
                /*lda*/ 8,
                4,
                4,
            ),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            validate_gemm_layout(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                4,
                3,
                3,
                /*ldb*/ 9,
                4,
            ),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            validate_gemm_layout(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                4,
                3,
                3,
                4,
                /*ldc*/ 9,
            ),
            Err(BackendError::Unsupported(_))
        ));
    }
}
