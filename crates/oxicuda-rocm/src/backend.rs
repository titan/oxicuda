//! [`RocmBackend`] — the main entry point for the `oxicuda-rocm` crate.
//!
//! Implements the [`ComputeBackend`] trait from `oxicuda-backend` using
//! the AMD ROCm/HIP runtime for GPU compute on Linux.
//!
//! Compute operations (gemm, unary, binary, reduce, attention, conv2d)
//! currently execute on a **CPU fallback path**: the operands are staged
//! device→host, the math is performed on the host, and the result is copied
//! host→device. The HIP C++ kernel sources in [`crate::hip_kernels`] are the
//! intended `hiprtc`/`hipModuleLaunchKernel` implementation and are validated
//! by codegen tests, but they are not yet wired into these dispatch methods.
//! The fallback honours the column-major [`ComputeBackend`] contract so its
//! numerical results match the reference `CpuBackend`.

use std::sync::Arc;

use oxicuda_backend::{
    BackendError, BackendResult, BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp,
};

use crate::{device::RocmDevice, memory::RocmMemoryManager};

// ─── Backend struct ───────────────────────────────────────────────────────────

/// AMD ROCm/HIP GPU compute backend.
///
/// On Linux this loads `libamdhip64.so` at runtime, selects the first AMD
/// GPU device, and manages explicit device-memory allocations.
///
/// On non-Linux platforms (macOS, Windows) every operation returns
/// [`BackendError::DeviceError`] wrapping [`crate::error::RocmError::UnsupportedPlatform`].
///
/// # Lifecycle
///
/// 1. `RocmBackend::new()` — create an uninitialised backend.
/// 2. `init()` — load HIP runtime, select device, set up memory manager.
/// 3. Use `alloc`, `copy_htod`, compute ops, `copy_dtoh`, `free`.
/// 4. `synchronize()` — wait for all pending GPU work to complete.
#[derive(Debug)]
pub struct RocmBackend {
    device: Option<Arc<RocmDevice>>,
    memory: Option<Arc<RocmMemoryManager>>,
    initialized: bool,
}

impl RocmBackend {
    /// Create a new, uninitialised ROCm backend.
    pub fn new() -> Self {
        Self {
            device: None,
            memory: None,
            initialized: false,
        }
    }

    /// Return an error if the backend has not been initialised yet.
    fn check_init(&self) -> BackendResult<()> {
        if self.initialized {
            Ok(())
        } else {
            Err(BackendError::NotInitialized)
        }
    }

    /// Convenience accessor: return the memory manager or `NotInitialized`.
    fn memory_manager(&self) -> BackendResult<&Arc<RocmMemoryManager>> {
        self.memory.as_ref().ok_or(BackendError::NotInitialized)
    }
}

impl Default for RocmBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ─── ComputeBackend impl ──────────────────────────────────────────────────────

impl ComputeBackend for RocmBackend {
    fn name(&self) -> &str {
        "rocm"
    }

    fn init(&mut self) -> BackendResult<()> {
        if self.initialized {
            return Ok(());
        }
        match RocmDevice::new() {
            Ok(dev) => {
                let dev = Arc::new(dev);
                tracing::info!("ROCm backend initialised on: {}", dev.name());
                let memory = RocmMemoryManager::new(Arc::clone(&dev));
                self.device = Some(dev);
                self.memory = Some(Arc::new(memory));
                self.initialized = true;
                Ok(())
            }
            Err(e) => Err(BackendError::from(e)),
        }
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    // ── Compute operations ────────────────────────────────────────────────────

    fn gemm(
        &self,
        trans_a: BackendTranspose,
        trans_b: BackendTranspose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a_ptr: u64,
        lda: usize,
        b_ptr: u64,
        ldb: usize,
        beta: f64,
        c_ptr: u64,
        ldc: usize,
    ) -> BackendResult<()> {
        self.check_init()?;
        // Zero-dimension matrices are trivially complete.
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        self.dispatch_gemm(
            trans_a, trans_b, m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc,
        )
    }

    fn conv2d_forward(
        &self,
        input_ptr: u64,
        input_shape: &[usize],
        filter_ptr: u64,
        filter_shape: &[usize],
        output_ptr: u64,
        output_shape: &[usize],
        stride: &[usize],
        padding: &[usize],
    ) -> BackendResult<()> {
        self.check_init()?;

        if input_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(
                "input_shape must have 4 elements (NCHW)".into(),
            ));
        }
        if filter_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(
                "filter_shape must have 4 elements (KCFHFW)".into(),
            ));
        }
        if output_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(
                "output_shape must have 4 elements (NKOhOw)".into(),
            ));
        }
        if stride.len() != 2 {
            return Err(BackendError::InvalidArgument(
                "stride must have 2 elements [sh, sw]".into(),
            ));
        }
        if padding.len() != 2 {
            return Err(BackendError::InvalidArgument(
                "padding must have 2 elements [ph, pw]".into(),
            ));
        }

        self.dispatch_conv2d(
            input_ptr,
            input_shape,
            filter_ptr,
            filter_shape,
            output_ptr,
            output_shape,
            stride,
            padding,
        )
    }

    fn attention(
        &self,
        q_ptr: u64,
        k_ptr: u64,
        v_ptr: u64,
        o_ptr: u64,
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_kv: usize,
        head_dim: usize,
        scale: f64,
        causal: bool,
    ) -> BackendResult<()> {
        self.check_init()?;

        if seq_q == 0 || seq_kv == 0 || head_dim == 0 {
            return Err(BackendError::InvalidArgument(
                "seq_q, seq_kv, and head_dim must all be > 0".into(),
            ));
        }
        if scale <= 0.0 || !scale.is_finite() {
            return Err(BackendError::InvalidArgument(format!(
                "scale must be a positive finite number, got {scale}"
            )));
        }

        self.dispatch_attention(
            q_ptr, k_ptr, v_ptr, o_ptr, batch, heads, seq_q, seq_kv, head_dim, scale, causal,
        )
    }

    fn reduce(
        &self,
        op: ReduceOp,
        input_ptr: u64,
        output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()> {
        self.check_init()?;

        if shape.is_empty() {
            return Err(BackendError::InvalidArgument(
                "shape must not be empty".into(),
            ));
        }
        if axis >= shape.len() {
            return Err(BackendError::InvalidArgument(format!(
                "axis {axis} is out of bounds for shape of length {}",
                shape.len()
            )));
        }

        self.dispatch_reduce(op, input_ptr, output_ptr, shape, axis)
    }

    fn unary(&self, op: UnaryOp, input_ptr: u64, output_ptr: u64, n: usize) -> BackendResult<()> {
        self.check_init()?;
        if n == 0 {
            return Ok(());
        }
        self.dispatch_unary(op, input_ptr, output_ptr, n)
    }

    fn binary(
        &self,
        op: BinaryOp,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
        n: usize,
    ) -> BackendResult<()> {
        self.check_init()?;
        if n == 0 {
            return Ok(());
        }
        self.dispatch_binary(op, a_ptr, b_ptr, output_ptr, n)
    }

    #[allow(clippy::too_many_arguments)]
    fn batched_gemm(
        &self,
        trans_a: BackendTranspose,
        trans_b: BackendTranspose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a_ptr: u64,
        lda: usize,
        stride_a: usize,
        b_ptr: u64,
        ldb: usize,
        stride_b: usize,
        beta: f64,
        c_ptr: u64,
        ldc: usize,
        stride_c: usize,
        batch_count: usize,
    ) -> BackendResult<()> {
        self.check_init()?;
        if batch_count == 0 || m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        self.dispatch_batched_gemm(
            trans_a,
            trans_b,
            m,
            n,
            k,
            alpha,
            a_ptr,
            lda,
            stride_a,
            b_ptr,
            ldb,
            stride_b,
            beta,
            c_ptr,
            ldc,
            stride_c,
            batch_count,
        )
    }

    // ── Synchronisation ───────────────────────────────────────────────────────

    fn synchronize(&self) -> BackendResult<()> {
        self.check_init()?;
        #[cfg(target_os = "linux")]
        {
            if let Some(dev) = &self.device {
                // SAFETY: `hip_device_synchronize` is a valid fn ptr from the
                // loaded HIP library.  It blocks until all pending GPU work on
                // the current device completes.
                let rc = unsafe { (dev.api.hip_device_synchronize)() };
                if rc != crate::device::HIP_SUCCESS {
                    return Err(BackendError::DeviceError(format!(
                        "hipDeviceSynchronize failed (rc={rc})"
                    )));
                }
            }
        }
        Ok(())
    }

    // ── Memory management ─────────────────────────────────────────────────────

    fn alloc(&self, bytes: usize) -> BackendResult<u64> {
        self.check_init()?;
        if bytes == 0 {
            return Err(BackendError::InvalidArgument(
                "cannot allocate 0 bytes".into(),
            ));
        }
        self.memory_manager()?
            .alloc(bytes)
            .map_err(BackendError::from)
    }

    fn free(&self, ptr: u64) -> BackendResult<()> {
        self.check_init()?;
        self.memory_manager()?.free(ptr).map_err(BackendError::from)
    }

    fn copy_htod(&self, dst: u64, src: &[u8]) -> BackendResult<()> {
        self.check_init()?;
        if src.is_empty() {
            return Ok(());
        }
        self.memory_manager()?
            .copy_to_device(dst, src)
            .map_err(BackendError::from)
    }

    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> BackendResult<()> {
        self.check_init()?;
        if dst.is_empty() {
            return Ok(());
        }
        self.memory_manager()?
            .copy_from_device(dst, src)
            .map_err(BackendError::from)
    }
}

// ─── Dispatch helpers ─────────────────────────────────────────────────────────

// On Linux with HIP runtime available, the dispatch methods use hiprtc to compile
// HIP C++ kernel sources at runtime, then load and launch via hipModule API.
// On non-Linux, they return DeviceError since HIP is unavailable.

impl RocmBackend {
    #[allow(clippy::too_many_arguments)]
    fn dispatch_gemm(
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
        if _m == 0 || _n == 0 {
            return Ok(());
        }
        #[cfg(target_os = "linux")]
        {
            // CPU fallback path (see module docs): the HIP kernel launch is not
            // yet wired, so ensure the device is live and run the column-major
            // reference math on host-staged operands.
            self.device.as_ref().ok_or(BackendError::NotInitialized)?;
            let mm = self.memory_manager()?;

            let m = _m;
            let n = _n;
            let k = _k;
            let alpha = _alpha as f32;
            let beta = _beta as f32;

            // Leading-dimension-aware operand extents (column-major contract).
            let (a_rows, a_cols, b_rows, b_cols) =
                gemm_operand_extents(_trans_a, _trans_b, m, n, k);
            check_gemm_leading_dims(a_rows, b_rows, m, _lda, _ldb, _ldc)?;

            // Read A, B, C from device (sized by leading dimensions).
            let a_bytes = _lda * a_cols * 4;
            let b_bytes = _ldb * b_cols * 4;
            let c_bytes = _ldc * n * 4;

            let mut a_host = vec![0u8; a_bytes];
            let mut b_host = vec![0u8; b_bytes];
            let mut c_host = vec![0u8; c_bytes];

            mm.copy_from_device(&mut a_host, _a_ptr)
                .map_err(BackendError::from)?;
            mm.copy_from_device(&mut b_host, _b_ptr)
                .map_err(BackendError::from)?;
            mm.copy_from_device(&mut c_host, _c_ptr)
                .map_err(BackendError::from)?;

            // Reinterpret as f32 slices and run the column-major reference GEMM.
            let a_f32 = bytemuck_cast_f32(&a_host);
            let b_f32 = bytemuck_cast_f32(&b_host);
            let c_f32 = bytemuck_cast_f32_mut(&mut c_host);
            col_major_gemm_f32(
                _trans_a, _trans_b, m, n, k, alpha, a_f32, _lda, b_f32, _ldb, beta, c_f32, _ldc,
            );

            // Write C back to device (blocking hipMemcpy already synchronises).
            mm.copy_to_device(_c_ptr, &c_host)
                .map_err(BackendError::from)?;
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(BackendError::DeviceError(
                "ROCm compute requires Linux with HIP runtime".into(),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_batched_gemm(
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
        if _m == 0 || _n == 0 {
            return Ok(());
        }
        #[cfg(target_os = "linux")]
        {
            self.device.as_ref().ok_or(BackendError::NotInitialized)?;
            let mm = self.memory_manager()?;

            let m = _m;
            let n = _n;
            let k = _k;
            let alpha = _alpha as f32;
            let beta = _beta as f32;
            let batch_count = _batch_count;

            // Per-batch operand extents honouring the leading dimensions.
            let (a_rows, a_cols, b_rows, b_cols) =
                gemm_operand_extents(_trans_a, _trans_b, m, n, k);
            check_gemm_leading_dims(a_rows, b_rows, m, _lda, _ldb, _ldc)?;

            let mut a_host = vec![0u8; _lda * a_cols * 4];
            let mut b_host = vec![0u8; _ldb * b_cols * 4];
            let mut c_host = vec![0u8; _ldc * n * 4];

            for batch in 0..batch_count {
                let a_offset = _a_ptr + (batch * _stride_a * 4) as u64;
                let b_offset = _b_ptr + (batch * _stride_b * 4) as u64;
                let c_offset = _c_ptr + (batch * _stride_c * 4) as u64;

                mm.copy_from_device(&mut a_host, a_offset)
                    .map_err(BackendError::from)?;
                mm.copy_from_device(&mut b_host, b_offset)
                    .map_err(BackendError::from)?;
                mm.copy_from_device(&mut c_host, c_offset)
                    .map_err(BackendError::from)?;

                let a_f32 = bytemuck_cast_f32(&a_host);
                let b_f32 = bytemuck_cast_f32(&b_host);
                let c_f32 = bytemuck_cast_f32_mut(&mut c_host);
                col_major_gemm_f32(
                    _trans_a, _trans_b, m, n, k, alpha, a_f32, _lda, b_f32, _ldb, beta, c_f32, _ldc,
                );

                mm.copy_to_device(c_offset, &c_host)
                    .map_err(BackendError::from)?;
            }

            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(BackendError::DeviceError(
                "ROCm compute requires Linux with HIP runtime".into(),
            ))
        }
    }

    fn dispatch_unary(
        &self,
        op: UnaryOp,
        _input_ptr: u64,
        _output_ptr: u64,
        _n: usize,
    ) -> BackendResult<()> {
        #[cfg(target_os = "linux")]
        {
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
            let _src = crate::hip_kernels::elementwise_hip(op_str);

            let dev = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
            let mm = self.memory_manager()?;

            let n = _n;
            let byte_len = n * 4;
            let mut input_host = vec![0u8; byte_len];
            mm.copy_from_device(&mut input_host, _input_ptr)
                .map_err(BackendError::from)?;

            let in_f32 = bytemuck_cast_f32(&input_host);
            let mut out_host = vec![0u8; byte_len];
            let out_f32 = bytemuck_cast_f32_mut(&mut out_host);

            for i in 0..n {
                let x = in_f32[i];
                out_f32[i] = match op {
                    UnaryOp::Relu => x.max(0.0),
                    UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                    UnaryOp::Tanh => x.tanh(),
                    UnaryOp::Exp => x.exp(),
                    UnaryOp::Log => x.ln(),
                    UnaryOp::Sqrt => x.sqrt(),
                    UnaryOp::Abs => x.abs(),
                    UnaryOp::Neg => -x,
                };
            }

            mm.copy_to_device(_output_ptr, &out_host)
                .map_err(BackendError::from)?;

            let rc = unsafe { (dev.api.hip_device_synchronize)() };
            if rc != crate::device::HIP_SUCCESS {
                return Err(BackendError::DeviceError(format!(
                    "hipDeviceSynchronize failed (rc={rc})"
                )));
            }
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (op, _input_ptr, _output_ptr, _n);
            Err(BackendError::DeviceError(
                "ROCm compute requires Linux with HIP runtime".into(),
            ))
        }
    }

    fn dispatch_binary(
        &self,
        op: BinaryOp,
        _a_ptr: u64,
        _b_ptr: u64,
        _output_ptr: u64,
        _n: usize,
    ) -> BackendResult<()> {
        #[cfg(target_os = "linux")]
        {
            let op_str = match op {
                BinaryOp::Add => "add",
                BinaryOp::Sub => "sub",
                BinaryOp::Mul => "mul",
                BinaryOp::Div => "div",
                BinaryOp::Max => "max",
                BinaryOp::Min => "min",
            };
            let _src = crate::hip_kernels::binary_hip(op_str);

            let dev = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
            let mm = self.memory_manager()?;

            let n = _n;
            let byte_len = n * 4;
            let mut a_host = vec![0u8; byte_len];
            let mut b_host = vec![0u8; byte_len];
            mm.copy_from_device(&mut a_host, _a_ptr)
                .map_err(BackendError::from)?;
            mm.copy_from_device(&mut b_host, _b_ptr)
                .map_err(BackendError::from)?;

            let a_f32 = bytemuck_cast_f32(&a_host);
            let b_f32 = bytemuck_cast_f32(&b_host);
            let mut out_host = vec![0u8; byte_len];
            let out_f32 = bytemuck_cast_f32_mut(&mut out_host);

            for i in 0..n {
                let a = a_f32[i];
                let b = b_f32[i];
                out_f32[i] = match op {
                    BinaryOp::Add => a + b,
                    BinaryOp::Sub => a - b,
                    BinaryOp::Mul => a * b,
                    BinaryOp::Div => a / b,
                    BinaryOp::Max => a.max(b),
                    BinaryOp::Min => a.min(b),
                };
            }

            mm.copy_to_device(_output_ptr, &out_host)
                .map_err(BackendError::from)?;

            let rc = unsafe { (dev.api.hip_device_synchronize)() };
            if rc != crate::device::HIP_SUCCESS {
                return Err(BackendError::DeviceError(format!(
                    "hipDeviceSynchronize failed (rc={rc})"
                )));
            }
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (op, _a_ptr, _b_ptr, _output_ptr, _n);
            Err(BackendError::DeviceError(
                "ROCm compute requires Linux with HIP runtime".into(),
            ))
        }
    }

    fn dispatch_reduce(
        &self,
        op: ReduceOp,
        _input_ptr: u64,
        _output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()> {
        #[cfg(target_os = "linux")]
        {
            let op_str = match op {
                ReduceOp::Sum => "sum",
                ReduceOp::Max => "max",
                ReduceOp::Min => "min",
                ReduceOp::Mean => "mean",
            };
            let reduce_src = crate::hip_kernels::reduction_hip(op_str);
            if reduce_src.is_empty() {
                return Err(BackendError::Unsupported(format!(
                    "ROCm reduction op '{op_str}' not supported"
                )));
            }

            let dev = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
            let mm = self.memory_manager()?;

            let outer_size: usize = shape[..axis].iter().product::<usize>().max(1);
            let reduce_size = shape[axis];
            let inner_size: usize = shape[axis + 1..].iter().product::<usize>().max(1);
            let total_in: usize = shape.iter().product();
            let total_out = outer_size * inner_size;

            let in_bytes = total_in * 4;
            let out_bytes = total_out * 4;

            let mut input_host = vec![0u8; in_bytes];
            mm.copy_from_device(&mut input_host, _input_ptr)
                .map_err(BackendError::from)?;

            let in_f32 = bytemuck_cast_f32(&input_host);
            let mut out_host = vec![0u8; out_bytes];
            let out_f32 = bytemuck_cast_f32_mut(&mut out_host);

            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    let init = match op {
                        ReduceOp::Sum | ReduceOp::Mean => 0.0f32,
                        ReduceOp::Max => f32::NEG_INFINITY,
                        ReduceOp::Min => f32::INFINITY,
                    };
                    let mut acc = init;
                    for r in 0..reduce_size {
                        let idx = outer * (reduce_size * inner_size) + r * inner_size + inner;
                        let v = in_f32[idx];
                        acc = match op {
                            ReduceOp::Sum | ReduceOp::Mean => acc + v,
                            ReduceOp::Max => acc.max(v),
                            ReduceOp::Min => acc.min(v),
                        };
                    }
                    if matches!(op, ReduceOp::Mean) && reduce_size > 0 {
                        acc /= reduce_size as f32;
                    }
                    out_f32[outer * inner_size + inner] = acc;
                }
            }

            mm.copy_to_device(_output_ptr, &out_host)
                .map_err(BackendError::from)?;

            let rc = unsafe { (dev.api.hip_device_synchronize)() };
            if rc != crate::device::HIP_SUCCESS {
                return Err(BackendError::DeviceError(format!(
                    "hipDeviceSynchronize failed (rc={rc})"
                )));
            }
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (op, _input_ptr, _output_ptr, shape, axis);
            Err(BackendError::DeviceError(
                "ROCm compute requires Linux with HIP runtime".into(),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_attention(
        &self,
        _q_ptr: u64,
        _k_ptr: u64,
        _v_ptr: u64,
        _o_ptr: u64,
        _batch: usize,
        _heads: usize,
        _seq_q: usize,
        _seq_kv: usize,
        _head_dim: usize,
        _scale: f64,
        _causal: bool,
    ) -> BackendResult<()> {
        #[cfg(target_os = "linux")]
        {
            let _src = crate::hip_kernels::attention_hip();

            let dev = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
            let mm = self.memory_manager()?;

            let batch_heads = _batch * _heads;
            let q_elems = batch_heads * _seq_q * _head_dim;
            let k_elems = batch_heads * _seq_kv * _head_dim;
            let v_elems = k_elems;
            let o_elems = q_elems;

            let mut q_host = vec![0u8; q_elems * 4];
            let mut k_host = vec![0u8; k_elems * 4];
            let mut v_host = vec![0u8; v_elems * 4];

            mm.copy_from_device(&mut q_host, _q_ptr)
                .map_err(BackendError::from)?;
            mm.copy_from_device(&mut k_host, _k_ptr)
                .map_err(BackendError::from)?;
            mm.copy_from_device(&mut v_host, _v_ptr)
                .map_err(BackendError::from)?;

            let q_f32 = bytemuck_cast_f32(&q_host);
            let k_f32 = bytemuck_cast_f32(&k_host);
            let v_f32 = bytemuck_cast_f32(&v_host);
            let mut o_host = vec![0u8; o_elems * 4];
            let o_f32 = bytemuck_cast_f32_mut(&mut o_host);

            let scale = _scale as f32;
            for bh in 0..batch_heads {
                let q_off = bh * _seq_q * _head_dim;
                let k_off = bh * _seq_kv * _head_dim;
                let v_off = k_off;

                for sq in 0.._seq_q {
                    // Find max score for numerical stability
                    let mut max_score = f32::NEG_INFINITY;
                    let kv_limit = causal_kv_limit(_seq_q, _seq_kv, sq, _causal);
                    for sk in 0..kv_limit {
                        let mut dot = 0.0f32;
                        for dd in 0.._head_dim {
                            dot += q_f32[q_off + sq * _head_dim + dd]
                                * k_f32[k_off + sk * _head_dim + dd];
                        }
                        dot *= scale;
                        if dot > max_score {
                            max_score = dot;
                        }
                    }

                    // Compute softmax weights and accumulate output
                    let mut sum_exp = 0.0f32;
                    let mut acc = vec![0.0f32; _head_dim];
                    for sk in 0..kv_limit {
                        let mut dot = 0.0f32;
                        for dd in 0.._head_dim {
                            dot += q_f32[q_off + sq * _head_dim + dd]
                                * k_f32[k_off + sk * _head_dim + dd];
                        }
                        dot *= scale;
                        let w = (dot - max_score).exp();
                        sum_exp += w;
                        for dd in 0.._head_dim {
                            acc[dd] += w * v_f32[v_off + sk * _head_dim + dd];
                        }
                    }

                    for dd in 0.._head_dim {
                        let val = if sum_exp > 0.0 {
                            acc[dd] / sum_exp
                        } else {
                            0.0
                        };
                        o_f32[q_off + sq * _head_dim + dd] = val;
                    }
                }
            }

            mm.copy_to_device(_o_ptr, &o_host)
                .map_err(BackendError::from)?;

            let rc = unsafe { (dev.api.hip_device_synchronize)() };
            if rc != crate::device::HIP_SUCCESS {
                return Err(BackendError::DeviceError(format!(
                    "hipDeviceSynchronize failed (rc={rc})"
                )));
            }
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(BackendError::DeviceError(
                "ROCm compute requires Linux with HIP runtime".into(),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_conv2d(
        &self,
        _input_ptr: u64,
        input_shape: &[usize],
        _filter_ptr: u64,
        filter_shape: &[usize],
        _output_ptr: u64,
        output_shape: &[usize],
        stride: &[usize],
        padding: &[usize],
    ) -> BackendResult<()> {
        #[cfg(target_os = "linux")]
        {
            let _src = crate::hip_kernels::conv2d_forward_hip();

            let dev = self.device.as_ref().ok_or(BackendError::NotInitialized)?;
            let mm = self.memory_manager()?;

            let batch = input_shape[0];
            let c_in = input_shape[1];
            let h_in = input_shape[2];
            let w_in = input_shape[3];
            let k_out = filter_shape[0];
            let fh = filter_shape[2];
            let fw = filter_shape[3];
            let oh = output_shape[2];
            let ow = output_shape[3];
            let sh = stride[0];
            let sw = stride[1];
            let ph = padding[0];
            let pw = padding[1];

            let in_elems: usize = input_shape.iter().product();
            let f_elems: usize = filter_shape.iter().product();
            let o_elems: usize = output_shape.iter().product();

            let mut in_host = vec![0u8; in_elems * 4];
            let mut f_host = vec![0u8; f_elems * 4];
            mm.copy_from_device(&mut in_host, _input_ptr)
                .map_err(BackendError::from)?;
            mm.copy_from_device(&mut f_host, _filter_ptr)
                .map_err(BackendError::from)?;

            let in_f32 = bytemuck_cast_f32(&in_host);
            let f_f32 = bytemuck_cast_f32(&f_host);
            let mut out_host = vec![0u8; o_elems * 4];
            let out_f32 = bytemuck_cast_f32_mut(&mut out_host);

            for b in 0..batch {
                for kf in 0..k_out {
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let mut acc = 0.0f32;
                            for ci in 0..c_in {
                                for fy in 0..fh {
                                    for fx in 0..fw {
                                        let iy = (oy * sh + fy) as isize - ph as isize;
                                        let ix = (ox * sw + fx) as isize - pw as isize;
                                        if iy >= 0
                                            && (iy as usize) < h_in
                                            && ix >= 0
                                            && (ix as usize) < w_in
                                        {
                                            let in_idx = ((b * c_in + ci) * h_in + iy as usize)
                                                * w_in
                                                + ix as usize;
                                            let f_idx = ((kf * c_in + ci) * fh + fy) * fw + fx;
                                            acc += in_f32[in_idx] * f_f32[f_idx];
                                        }
                                    }
                                }
                            }
                            let o_idx = ((b * k_out + kf) * oh + oy) * ow + ox;
                            out_f32[o_idx] = acc;
                        }
                    }
                }
            }

            mm.copy_to_device(_output_ptr, &out_host)
                .map_err(BackendError::from)?;

            let rc = unsafe { (dev.api.hip_device_synchronize)() };
            if rc != crate::device::HIP_SUCCESS {
                return Err(BackendError::DeviceError(format!(
                    "hipDeviceSynchronize failed (rc={rc})"
                )));
            }
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (
                input_shape,
                _input_ptr,
                filter_shape,
                _filter_ptr,
                output_shape,
                _output_ptr,
                stride,
                padding,
            );
            Err(BackendError::DeviceError(
                "ROCm compute requires Linux with HIP runtime".into(),
            ))
        }
    }
}

// ─── FP16 / BF16 inherent methods ───────────────────────────────────────────

impl RocmBackend {
    /// Half-precision (FP16) GEMM: `C = alpha * A * B + beta * C`.
    ///
    /// All buffers use 2-byte `f16` elements.  Accumulation is performed in
    /// `f32` on the CPU fallback path.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16(
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
        self.check_init()?;
        if _m == 0 || _n == 0 || _k == 0 {
            return Ok(());
        }
        #[cfg(target_os = "linux")]
        {
            self.device.as_ref().ok_or(BackendError::NotInitialized)?;
            let mm = self.memory_manager()?;

            let m = _m;
            let n = _n;
            let k = _k;
            let alpha = _alpha as f32;
            let beta = _beta as f32;

            let (a_rows, a_cols, b_rows, b_cols) =
                gemm_operand_extents(_trans_a, _trans_b, m, n, k);
            check_gemm_leading_dims(a_rows, b_rows, m, _lda, _ldb, _ldc)?;

            let mut a_host = vec![0u8; _lda * a_cols * 2];
            let mut b_host = vec![0u8; _ldb * b_cols * 2];
            let mut c_host = vec![0u8; _ldc * n * 2];

            mm.copy_from_device(&mut a_host, _a_ptr)
                .map_err(BackendError::from)?;
            mm.copy_from_device(&mut b_host, _b_ptr)
                .map_err(BackendError::from)?;
            mm.copy_from_device(&mut c_host, _c_ptr)
                .map_err(BackendError::from)?;

            let a_f16 = bytemuck_cast_f16(&a_host);
            let b_f16 = bytemuck_cast_f16(&b_host);
            let c_f16 = bytemuck_cast_f16_mut(&mut c_host);
            col_major_gemm_f16(
                _trans_a, _trans_b, m, n, k, alpha, a_f16, _lda, b_f16, _ldb, beta, c_f16, _ldc,
            );

            mm.copy_to_device(_c_ptr, &c_host)
                .map_err(BackendError::from)?;
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(BackendError::DeviceError(
                "ROCm compute requires Linux with HIP runtime".into(),
            ))
        }
    }

    /// BFloat16 GEMM: `C = alpha * A * B + beta * C`.
    ///
    /// All buffers use 2-byte `bf16` elements.  Accumulation is performed in
    /// `f32` on the CPU fallback path.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bf16(
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
        self.check_init()?;
        if _m == 0 || _n == 0 || _k == 0 {
            return Ok(());
        }
        #[cfg(target_os = "linux")]
        {
            self.device.as_ref().ok_or(BackendError::NotInitialized)?;
            let mm = self.memory_manager()?;

            let m = _m;
            let n = _n;
            let k = _k;
            let alpha = _alpha as f32;
            let beta = _beta as f32;

            let (a_rows, a_cols, b_rows, b_cols) =
                gemm_operand_extents(_trans_a, _trans_b, m, n, k);
            check_gemm_leading_dims(a_rows, b_rows, m, _lda, _ldb, _ldc)?;

            let mut a_host = vec![0u8; _lda * a_cols * 2];
            let mut b_host = vec![0u8; _ldb * b_cols * 2];
            let mut c_host = vec![0u8; _ldc * n * 2];

            mm.copy_from_device(&mut a_host, _a_ptr)
                .map_err(BackendError::from)?;
            mm.copy_from_device(&mut b_host, _b_ptr)
                .map_err(BackendError::from)?;
            mm.copy_from_device(&mut c_host, _c_ptr)
                .map_err(BackendError::from)?;

            let a_bf16 = bytemuck_cast_bf16(&a_host);
            let b_bf16 = bytemuck_cast_bf16(&b_host);
            let c_bf16 = bytemuck_cast_bf16_mut(&mut c_host);
            col_major_gemm_bf16(
                _trans_a, _trans_b, m, n, k, alpha, a_bf16, _lda, b_bf16, _ldb, beta, c_bf16, _ldc,
            );

            mm.copy_to_device(_c_ptr, &c_host)
                .map_err(BackendError::from)?;
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(BackendError::DeviceError(
                "ROCm compute requires Linux with HIP runtime".into(),
            ))
        }
    }
}

// ─── Byte reinterpretation helpers ──────────────────────────────────────────

/// Reinterpret a `&[u8]` slice as `&[f32]`.
///
/// Uses [`bytemuck::cast_slice`], which validates both length divisibility and
/// pointer alignment (panicking loudly on a mismatch) rather than risking the
/// undefined behaviour of an unchecked `from_raw_parts` on an under-aligned
/// `Vec<u8>`. The staging buffers passed here come from a fresh `vec![0u8; _]`
/// whose allocation is suitably aligned on the supported targets.
///
/// # Panics
///
/// Panics if `bytes.len()` is not a multiple of 4 or the slice is not 4-aligned.
#[cfg(target_os = "linux")]
fn bytemuck_cast_f32(bytes: &[u8]) -> &[f32] {
    bytemuck::cast_slice(bytes)
}

/// Reinterpret a `&mut [u8]` slice as `&mut [f32]`. See [`bytemuck_cast_f32`].
#[cfg(target_os = "linux")]
fn bytemuck_cast_f32_mut(bytes: &mut [u8]) -> &mut [f32] {
    bytemuck::cast_slice_mut(bytes)
}

/// Reinterpret a `&[u8]` slice as `&[half::f16]`. See [`bytemuck_cast_f32`].
#[cfg(target_os = "linux")]
fn bytemuck_cast_f16(bytes: &[u8]) -> &[half::f16] {
    bytemuck::cast_slice(bytes)
}

/// Reinterpret a `&mut [u8]` slice as `&mut [half::f16]`. See [`bytemuck_cast_f32`].
#[cfg(target_os = "linux")]
fn bytemuck_cast_f16_mut(bytes: &mut [u8]) -> &mut [half::f16] {
    bytemuck::cast_slice_mut(bytes)
}

/// Reinterpret a `&[u8]` slice as `&[half::bf16]`. See [`bytemuck_cast_f32`].
#[cfg(target_os = "linux")]
fn bytemuck_cast_bf16(bytes: &[u8]) -> &[half::bf16] {
    bytemuck::cast_slice(bytes)
}

/// Reinterpret a `&mut [u8]` slice as `&mut [half::bf16]`. See [`bytemuck_cast_f32`].
#[cfg(target_os = "linux")]
fn bytemuck_cast_bf16_mut(bytes: &mut [u8]) -> &mut [half::bf16] {
    bytemuck::cast_slice_mut(bytes)
}

// ─── Column-major GEMM reference helpers ────────────────────────────────────

/// Row/column extents of the *stored* GEMM operands after honouring the
/// transpose flags, in the column-major contract shared with the CPU backend.
///
/// Returns `(a_rows, a_cols, b_rows, b_cols)`: the stored A buffer spans
/// `lda * a_cols` elements and the stored B buffer spans `ldb * b_cols`.
#[cfg(target_os = "linux")]
fn gemm_operand_extents(
    trans_a: BackendTranspose,
    trans_b: BackendTranspose,
    m: usize,
    n: usize,
    k: usize,
) -> (usize, usize, usize, usize) {
    let (a_rows, a_cols) = if trans_a == BackendTranspose::NoTrans {
        (m, k)
    } else {
        (k, m)
    };
    let (b_rows, b_cols) = if trans_b == BackendTranspose::NoTrans {
        (k, n)
    } else {
        (n, k)
    };
    (a_rows, a_cols, b_rows, b_cols)
}

/// Reject leading dimensions smaller than the stored matrix extent, matching
/// the CPU reference contract (`lda >= a_rows`, `ldb >= b_rows`, `ldc >= m`).
#[cfg(target_os = "linux")]
fn check_gemm_leading_dims(
    a_rows: usize,
    b_rows: usize,
    m: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) -> BackendResult<()> {
    if lda < a_rows || ldb < b_rows || ldc < m {
        return Err(BackendError::InvalidArgument(
            "leading dimension smaller than matrix extent".into(),
        ));
    }
    Ok(())
}

/// Column-major buffer index of `op(M)[r, c]` for a source matrix stored with
/// leading dimension `ld`. Mirrors `cpu.rs::at`:
/// `NoTrans → col_major(r, c, ld)`; `Trans/ConjTrans → col_major(c, r, ld)`.
#[cfg(target_os = "linux")]
#[inline]
fn op_index(trans: BackendTranspose, r: usize, c: usize, ld: usize) -> usize {
    match trans {
        BackendTranspose::NoTrans => c * ld + r,
        BackendTranspose::Trans | BackendTranspose::ConjTrans => r * ld + c,
    }
}

/// Number of keys the query at local index `sq` may attend to under a causal
/// mask.
///
/// With a KV cache the query at local index `sq` sits at absolute position
/// `seq_kv - seq_q + sq`, so it attends to `(seq_kv - seq_q) + sq + 1` keys
/// (clamped to `seq_kv`). This reduces to `sq + 1` only for full prefill
/// (`seq_q == seq_kv`). Non-causal attention sees all `seq_kv` keys.
#[cfg(target_os = "linux")]
#[inline]
fn causal_kv_limit(seq_q: usize, seq_kv: usize, sq: usize, causal: bool) -> usize {
    if causal {
        (seq_kv.saturating_sub(seq_q) + sq + 1).min(seq_kv)
    } else {
        seq_kv
    }
}

/// Column-major reference GEMM `C = alpha*op(A)*op(B) + beta*C` on `f32`.
#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn col_major_gemm_f32(
    trans_a: BackendTranspose,
    trans_b: BackendTranspose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += a[op_index(trans_a, row, i, lda)] * b[op_index(trans_b, i, col, ldb)];
            }
            let idx = col * ldc + row;
            c[idx] = alpha * acc + beta * c[idx];
        }
    }
}

/// Column-major reference GEMM on `f16` operands with `f32` accumulation.
#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn col_major_gemm_f16(
    trans_a: BackendTranspose,
    trans_b: BackendTranspose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[half::f16],
    lda: usize,
    b: &[half::f16],
    ldb: usize,
    beta: f32,
    c: &mut [half::f16],
    ldc: usize,
) {
    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += a[op_index(trans_a, row, i, lda)].to_f32()
                    * b[op_index(trans_b, i, col, ldb)].to_f32();
            }
            let idx = col * ldc + row;
            let cur = c[idx].to_f32();
            c[idx] = half::f16::from_f32(alpha * acc + beta * cur);
        }
    }
}

/// Column-major reference GEMM on `bf16` operands with `f32` accumulation.
#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn col_major_gemm_bf16(
    trans_a: BackendTranspose,
    trans_b: BackendTranspose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[half::bf16],
    lda: usize,
    b: &[half::bf16],
    ldb: usize,
    beta: f32,
    c: &mut [half::bf16],
    ldc: usize,
) {
    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += a[op_index(trans_a, row, i, lda)].to_f32()
                    * b[op_index(trans_b, i, col, ldb)].to_f32();
            }
            let idx = col * ldc + row;
            let cur = c[idx].to_f32();
            c[idx] = half::bf16::from_f32(alpha * acc + beta * cur);
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_backend::{BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp};

    // ── 1. Structural ─────────────────────────────────────────────────────────

    #[test]
    fn rocm_backend_new_uninitialized() {
        let b = RocmBackend::new();
        assert!(!b.is_initialized());
    }

    #[test]
    fn rocm_backend_name() {
        let b = RocmBackend::new();
        assert_eq!(b.name(), "rocm");
    }

    #[test]
    fn rocm_backend_default() {
        let b = RocmBackend::default();
        assert!(!b.is_initialized());
        assert_eq!(b.name(), "rocm");
    }

    #[test]
    fn backend_debug_impl() {
        let b = RocmBackend::new();
        let s = format!("{b:?}");
        assert!(s.contains("RocmBackend"));
    }

    #[test]
    fn backend_object_safe() {
        let b: Box<dyn ComputeBackend> = Box::new(RocmBackend::new());
        assert_eq!(b.name(), "rocm");
    }

    // ── 2. Not-initialized guards ─────────────────────────────────────────────

    #[test]
    fn backend_not_initialized_gemm() {
        let b = RocmBackend::new();
        let result = b.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            4,
            4,
            4,
            1.0,
            0,
            4,
            0,
            4,
            0.0,
            0,
            4,
        );
        assert_eq!(result, Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_alloc() {
        let b = RocmBackend::new();
        assert_eq!(b.alloc(1024), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_synchronize() {
        let b = RocmBackend::new();
        assert_eq!(b.synchronize(), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_free() {
        let b = RocmBackend::new();
        assert_eq!(b.free(1), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_copy_htod() {
        let b = RocmBackend::new();
        assert_eq!(b.copy_htod(1, b"hello"), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_copy_dtoh() {
        let b = RocmBackend::new();
        let mut buf = [0u8; 4];
        assert_eq!(b.copy_dtoh(&mut buf, 1), Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_batched_gemm() {
        let b = RocmBackend::new();
        let result = b.batched_gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            4,
            4,
            4,
            1.0,
            0,
            4,
            16,
            0,
            4,
            16,
            0.0,
            0,
            4,
            16,
            3,
        );
        assert_eq!(result, Err(BackendError::NotInitialized));
    }

    #[test]
    fn batched_gemm_zero_batch_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.batched_gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                4,
                4,
                4,
                1.0,
                0,
                4,
                16,
                0,
                4,
                16,
                0.0,
                0,
                4,
                16,
                0, // batch_count = 0
            ),
            Ok(())
        );
    }

    #[test]
    fn batched_gemm_zero_dims_noop() {
        let Some(b) = try_init() else {
            return;
        };
        // m == 0
        assert_eq!(
            b.batched_gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                0,
                4,
                4,
                1.0,
                0,
                4,
                0,
                0,
                4,
                0,
                0.0,
                0,
                4,
                0,
                2,
            ),
            Ok(())
        );
        // n == 0
        assert_eq!(
            b.batched_gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                4,
                0,
                4,
                1.0,
                0,
                4,
                0,
                0,
                4,
                0,
                0.0,
                0,
                4,
                0,
                2,
            ),
            Ok(())
        );
        // k == 0
        assert_eq!(
            b.batched_gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                4,
                4,
                0,
                1.0,
                0,
                4,
                0,
                0,
                4,
                0,
                0.0,
                0,
                4,
                0,
                2,
            ),
            Ok(())
        );
    }

    // ── 3. Graceful init failure ──────────────────────────────────────────────

    #[test]
    fn init_graceful_failure() {
        // Verify that init() returns a Result and never panics on any platform.
        let mut b = RocmBackend::new();
        let _result = b.init();
        // Both Ok(()) and Err(_) are acceptable.
    }

    // ── 4. Post-init tests (skip if no AMD GPU / not on Linux) ────────────────

    fn try_init() -> Option<RocmBackend> {
        let mut b = RocmBackend::new();
        match b.init() {
            Ok(()) => Some(b),
            Err(_) => None,
        }
    }

    #[test]
    fn alloc_zero_bytes_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.alloc(0),
            Err(BackendError::InvalidArgument(
                "cannot allocate 0 bytes".into()
            ))
        );
    }

    #[test]
    fn copy_htod_empty_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.copy_htod(0, &[]), Ok(()));
    }

    #[test]
    fn copy_dtoh_empty_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.copy_dtoh(&mut [], 0), Ok(()));
    }

    #[test]
    fn gemm_zero_dims_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                0,
                0,
                0,
                1.0,
                0,
                1,
                0,
                1,
                0.0,
                0,
                1
            ),
            Ok(())
        );
    }

    #[test]
    fn unary_zero_n_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.unary(UnaryOp::Relu, 0, 0, 0), Ok(()));
    }

    #[test]
    fn binary_zero_n_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.binary(BinaryOp::Add, 0, 0, 0, 0), Ok(()));
    }

    #[test]
    fn synchronize_after_init() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(b.synchronize(), Ok(()));
    }

    #[test]
    fn reduce_empty_shape_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.reduce(ReduceOp::Sum, 0, 0, &[], 0),
            Err(BackendError::InvalidArgument(
                "shape must not be empty".into()
            ))
        );
    }

    #[test]
    fn reduce_axis_out_of_bounds_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.reduce(ReduceOp::Sum, 0, 0, &[4, 4], 5),
            Err(BackendError::InvalidArgument(
                "axis 5 is out of bounds for shape of length 2".into()
            ))
        );
    }

    #[test]
    fn attention_zero_seq_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.attention(0, 0, 0, 0, 1, 1, 0, 8, 64, 0.125, false),
            Err(BackendError::InvalidArgument(
                "seq_q, seq_kv, and head_dim must all be > 0".into()
            ))
        );
    }

    #[test]
    fn attention_invalid_scale_error() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, 0.0, false),
            Err(BackendError::InvalidArgument(
                "scale must be a positive finite number, got 0".into()
            ))
        );
        assert_eq!(
            b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, -1.0, false),
            Err(BackendError::InvalidArgument(
                "scale must be a positive finite number, got -1".into()
            ))
        );
        assert!(
            b.attention(0, 0, 0, 0, 1, 1, 8, 8, 64, f64::INFINITY, false)
                .is_err()
        );
    }

    #[test]
    fn conv2d_wrong_input_rank() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.conv2d_forward(
                0,
                &[1, 3, 32],
                0,
                &[16, 3, 3, 3],
                0,
                &[1, 16, 30, 30],
                &[1, 1],
                &[0, 0]
            ),
            Err(BackendError::InvalidArgument(
                "input_shape must have 4 elements (NCHW)".into()
            ))
        );
    }

    #[test]
    fn conv2d_wrong_filter_rank() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.conv2d_forward(
                0,
                &[1, 3, 32, 32],
                0,
                &[16, 3, 3],
                0,
                &[1, 16, 30, 30],
                &[1, 1],
                &[0, 0]
            ),
            Err(BackendError::InvalidArgument(
                "filter_shape must have 4 elements (KCFHFW)".into()
            ))
        );
    }

    #[test]
    fn init_idempotent() {
        let Some(mut b) = try_init() else {
            return;
        };
        assert_eq!(b.init(), Ok(()));
        assert!(b.is_initialized());
    }

    #[test]
    fn alloc_copy_roundtrip() {
        let Some(b) = try_init() else {
            return;
        };
        let src: Vec<u8> = (0u8..64).collect();
        let handle = match b.alloc(src.len()) {
            Ok(h) => h,
            Err(_) => return,
        };
        b.copy_htod(handle, &src).expect("copy_htod");
        let mut dst = vec![0u8; src.len()];
        b.copy_dtoh(&mut dst, handle).expect("copy_dtoh");
        assert_eq!(src, dst);
        b.free(handle).expect("free");
    }

    #[test]
    fn double_init_is_noop() {
        let Some(mut b) = try_init() else {
            return;
        };
        assert!(b.is_initialized());
        assert_eq!(b.init(), Ok(()));
        assert!(b.is_initialized());
    }

    #[test]
    fn alloc_and_free_basic() {
        let Some(b) = try_init() else {
            return;
        };
        let handle = match b.alloc(512) {
            Ok(h) => h,
            Err(_) => return,
        };
        assert!(handle > 0);
        b.free(handle).expect("free should succeed");
    }

    // ── FP16 / BF16 backend methods ─────────────────────────────────────────

    #[test]
    fn backend_not_initialized_gemm_f16() {
        let b = RocmBackend::new();
        let result = b.gemm_f16(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            4,
            4,
            4,
            1.0,
            0,
            4,
            0,
            4,
            0.0,
            0,
            4,
        );
        assert_eq!(result, Err(BackendError::NotInitialized));
    }

    #[test]
    fn backend_not_initialized_gemm_bf16() {
        let b = RocmBackend::new();
        let result = b.gemm_bf16(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            4,
            4,
            4,
            1.0,
            0,
            4,
            0,
            4,
            0.0,
            0,
            4,
        );
        assert_eq!(result, Err(BackendError::NotInitialized));
    }

    #[test]
    fn gemm_f16_zero_dims_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.gemm_f16(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                0,
                0,
                0,
                1.0,
                0,
                1,
                0,
                1,
                0.0,
                0,
                1,
            ),
            Ok(())
        );
    }

    #[test]
    fn gemm_bf16_zero_dims_noop() {
        let Some(b) = try_init() else {
            return;
        };
        assert_eq!(
            b.gemm_bf16(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                0,
                0,
                0,
                1.0,
                0,
                1,
                0,
                1,
                0.0,
                0,
                1,
            ),
            Ok(())
        );
    }

    // ── Column-major GEMM contract (regression for the row-major bug) ──────────

    #[cfg(target_os = "linux")]
    #[test]
    fn gemm_operand_extents_are_transpose_aware() {
        use BackendTranspose::{NoTrans, Trans};
        // A: op(A) is m×k. NoTrans stored m×k, Trans stored k×m.
        assert_eq!(
            gemm_operand_extents(NoTrans, NoTrans, 8, 4, 2),
            (8, 2, 2, 4)
        );
        assert_eq!(gemm_operand_extents(Trans, Trans, 8, 4, 2), (2, 8, 4, 2));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn check_gemm_leading_dims_rejects_short_ld() {
        // a_rows=8, b_rows=2, m=8: lda>=8, ldb>=2, ldc>=8.
        assert!(check_gemm_leading_dims(8, 2, 8, 8, 2, 8).is_ok());
        assert!(matches!(
            check_gemm_leading_dims(8, 2, 8, 7, 2, 8),
            Err(BackendError::InvalidArgument(_))
        ));
        assert!(matches!(
            check_gemm_leading_dims(8, 2, 8, 8, 2, 7),
            Err(BackendError::InvalidArgument(_))
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn col_major_gemm_f32_matches_logical_product_all_transpose_combos() {
        use BackendTranspose::{NoTrans, Trans};
        // Non-symmetric logical operands A (m×k) and B (k×n).
        let (m, n, k) = (2usize, 3usize, 4usize);
        let a_logical = |r: usize, c: usize| (r * 10 + c + 1) as f32;
        let b_logical = |r: usize, c: usize| (r * 7 + c * 2 + 3) as f32;

        // Expected C = A·B laid out row-major (m×n) for comparison.
        let mut expected = vec![0.0f32; m * n];
        for (i, exp_row) in expected.chunks_exact_mut(n).enumerate() {
            for (j, slot) in exp_row.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a_logical(i, p) * b_logical(p, j);
                }
                *slot = acc;
            }
        }

        for &ta in &[NoTrans, Trans] {
            for &tb in &[NoTrans, Trans] {
                // Column-major storage with padded leading dimensions.
                let (a_rows, a_cols) = if ta == NoTrans { (m, k) } else { (k, m) };
                let lda = a_rows + 2;
                let mut a_buf = vec![0.0f32; lda * a_cols];
                for c in 0..a_cols {
                    for r in 0..a_rows {
                        // Stored (r,c) is logical A(r,c) for NoTrans, A(c,r) for Trans.
                        a_buf[c * lda + r] = if ta == NoTrans {
                            a_logical(r, c)
                        } else {
                            a_logical(c, r)
                        };
                    }
                }

                let (b_rows, b_cols) = if tb == NoTrans { (k, n) } else { (n, k) };
                let ldb = b_rows + 1;
                let mut b_buf = vec![0.0f32; ldb * b_cols];
                for c in 0..b_cols {
                    for r in 0..b_rows {
                        b_buf[c * ldb + r] = if tb == NoTrans {
                            b_logical(r, c)
                        } else {
                            b_logical(c, r)
                        };
                    }
                }

                let ldc = m + 3;
                let mut c_buf = vec![0.0f32; ldc * n];
                col_major_gemm_f32(
                    ta, tb, m, n, k, 1.0, &a_buf, lda, &b_buf, ldb, 0.0, &mut c_buf, ldc,
                );

                for i in 0..m {
                    for j in 0..n {
                        let got = c_buf[j * ldc + i];
                        let want = expected[i * n + j];
                        assert!(
                            (got - want).abs() < 1e-3,
                            "ta={ta:?} tb={tb:?} C[{i}][{j}] got {got} want {want}"
                        );
                    }
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn causal_kv_limit_offsets_by_cache_length() {
        // Incremental decode: 1 new query, 10 cached keys → must see all 10.
        assert_eq!(causal_kv_limit(1, 10, 0, true), 10);
        // Full prefill (seq_q == seq_kv): query sq attends to sq+1 keys.
        assert_eq!(causal_kv_limit(4, 4, 0, true), 1);
        assert_eq!(causal_kv_limit(4, 4, 3, true), 4);
        // Chunked decode: 2 new queries over a 10-key cache (offset 8).
        assert_eq!(causal_kv_limit(2, 10, 0, true), 9);
        assert_eq!(causal_kv_limit(2, 10, 1, true), 10);
        // Non-causal always sees the whole KV length.
        assert_eq!(causal_kv_limit(2, 10, 0, false), 10);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn col_major_gemm_f32_applies_alpha_beta() {
        use BackendTranspose::NoTrans;
        // 1×1: A=[2], B=[3], C0=[5], alpha=2, beta=4 → 2*(2*3)+4*5 = 32.
        let a = [2.0f32];
        let b = [3.0f32];
        let mut c = [5.0f32];
        col_major_gemm_f32(NoTrans, NoTrans, 1, 1, 1, 2.0, &a, 1, &b, 1, 4.0, &mut c, 1);
        assert!((c[0] - 32.0).abs() < 1e-6);
    }
}
