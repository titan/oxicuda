//! Abstract compute backend for GPU-accelerated operations.
//!
//! The [`ComputeBackend`] trait defines the interface for GPU computation,
//! allowing higher-level crates (SciRS2, oxionnx, ToRSh, TrustformeRS)
//! to use GPU acceleration without coupling to specific GPU APIs.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────┐
//! │  SciRS2 / ToRSh / oxionnx  │
//! │         (consumers)         │
//! └─────────────┬───────────────┘
//!               │  dyn ComputeBackend
//! ┌─────────────▼───────────────┐
//! │   BackendRegistry (select)  │
//! │       ComputeBackend        │
//! │     (trait definition)      │
//! └─────────────┬───────────────┘
//!               │
//! ┌─────────────▼───────────────┐
//! │  CudaBackend / MetalBackend │
//! │  CpuBackend / NullBackend   │
//! │    (concrete impls)         │
//! └─────────────────────────────┘
//! ```
//!
//! # Beyond the trait
//!
//! This crate also ships the host-side **control plane** that turns the bare
//! trait into a usable multi-backend abstraction layer:
//!
//! * [`BackendKind`] — names every backend and ranks them by preference.
//! * [`BackendRegistry`] — registers backends, probes availability, and does
//!   capability-based selection plus GPU→CPU [`fallback chains`](BackendRegistry::fallback_chain).
//! * [`Capabilities`] / [`DeviceInfo`] — per-backend feature and device
//!   discovery.
//! * [`CpuBackend`] — a genuinely-working pure-Rust reference backend (the
//!   always-available fallback and the numerical reference for conformance).
//! * [`NullBackend`] — a scaffold that refuses every op, for testing dispatch.
//!
//! None of the control-plane logic touches a GPU, so it is fully testable on
//! a machine with no accelerator.

mod backend_kind;
mod capabilities;
mod cpu;
mod error;
mod null;
mod ops;
mod precision;
mod registry;

use std::fmt;

pub use backend_kind::BackendKind;
pub use capabilities::{Capabilities, DeviceInfo, MemoryKind, TileShape, default_tile_for};
pub use cpu::CpuBackend;
pub use error::{BackendError, BackendResult};
pub use null::NullBackend;
pub use ops::{BackendTranspose, BinaryOp, MixedPrecision, ReduceOp, UnaryOp};
pub use precision::{round_to_bf16, round_to_f16};
pub use registry::{BackendEntry, BackendRegistry, OpClass, SelectionRequest};

// ─── ComputeBackend trait ───────────────────────────────────

/// Abstract compute backend trait.
///
/// Implementations provide GPU-accelerated compute operations.
/// All operations work with opaque device memory pointers (`u64`)
/// and explicit shape/stride information, making the trait
/// independent of any particular memory management scheme.
///
/// # Object Safety
///
/// This trait is object-safe and can be used as `Box<dyn ComputeBackend>`
/// or `&dyn ComputeBackend` for dynamic dispatch.
///
/// # Lifecycle
///
/// 1. Create the backend (`CudaBackend::new()`).
/// 2. Call [`init`](ComputeBackend::init) to select a device and create a context.
/// 3. Allocate memory with [`alloc`](ComputeBackend::alloc).
/// 4. Transfer data with [`copy_htod`](ComputeBackend::copy_htod).
/// 5. Run compute operations ([`gemm`](ComputeBackend::gemm), [`conv2d_forward`](ComputeBackend::conv2d_forward), etc.).
/// 6. Read results with [`copy_dtoh`](ComputeBackend::copy_dtoh).
/// 7. Free memory with [`free`](ComputeBackend::free).
pub trait ComputeBackend: Send + Sync + fmt::Debug {
    /// Backend name (e.g., `"cuda"`, `"rocm"`, `"metal"`).
    fn name(&self) -> &str;

    /// Initialize the backend (select device, create context).
    ///
    /// Must be called before any other operation. Calling `init` on an
    /// already-initialized backend is a no-op.
    fn init(&mut self) -> BackendResult<()>;

    /// Returns `true` if the backend is ready for operations.
    fn is_initialized(&self) -> bool;

    /// Report this backend's capabilities (precision support, Tensor Cores,
    /// unified memory, thread/shared-memory limits, …).
    ///
    /// The default is the conservative CPU profile; GPU backends override it
    /// with values read from their driver.
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    /// Enumerate the devices this backend exposes, in a backend-agnostic
    /// [`DeviceInfo`] shape.
    ///
    /// The default returns an empty list (a backend that cannot enumerate
    /// devices, e.g. a pure trait stub). Real backends override it; the
    /// [`CpuBackend`] reports a single host device.
    fn available_devices(&self) -> BackendResult<Vec<DeviceInfo>> {
        Ok(Vec::new())
    }

    /// Suggest a GEMM tile shape `(tile_m, tile_n, tile_k)` for the given
    /// problem dimensions, to seed an autotuner.
    ///
    /// The default heuristic scales the tile with the problem size and snaps
    /// to a WMMA-aligned tile when [`capabilities`](Self::capabilities)
    /// reports Tensor Cores. Backends may override with hardware-specific
    /// shapes.
    fn recommended_tile_for(&self, m: usize, n: usize, k: usize) -> TileShape {
        default_tile_for(m, n, k, &self.capabilities())
    }

    /// General matrix multiply: `C = alpha * op(A) * op(B) + beta * C`.
    ///
    /// # Arguments
    ///
    /// * `trans_a`, `trans_b` — transpose modes for A and B.
    /// * `m`, `n`, `k` — matrix dimensions (C is m×n, A is m×k, B is k×n after transpose).
    /// * `alpha`, `beta` — scaling factors.
    /// * `a_ptr`, `b_ptr`, `c_ptr` — device pointers to column-major f64 matrices.
    /// * `lda`, `ldb`, `ldc` — leading dimensions.
    #[allow(clippy::too_many_arguments)]
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
    ) -> BackendResult<()>;

    /// 2D convolution forward pass.
    ///
    /// # Arguments
    ///
    /// * `input_ptr` — device pointer to input tensor (NCHW layout).
    /// * `input_shape` — `[N, C, H, W]`.
    /// * `filter_ptr` — device pointer to filter tensor.
    /// * `filter_shape` — `[K, C, Fh, Fw]`.
    /// * `output_ptr` — device pointer to output tensor.
    /// * `output_shape` — `[N, K, Oh, Ow]`.
    /// * `stride` — `[sh, sw]`.
    /// * `padding` — `[ph, pw]`.
    #[allow(clippy::too_many_arguments)]
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
    ) -> BackendResult<()>;

    /// Scaled dot-product attention.
    ///
    /// Computes `softmax(Q * K^T / scale) * V` with optional causal masking.
    ///
    /// # Arguments
    ///
    /// * `q_ptr`, `k_ptr`, `v_ptr` — device pointers to query, key, value tensors.
    /// * `o_ptr` — device pointer to output tensor.
    /// * `batch`, `heads` — batch size and number of attention heads.
    /// * `seq_q`, `seq_kv` — query and key/value sequence lengths.
    /// * `head_dim` — dimension of each attention head.
    /// * `scale` — attention scale factor (typically `1 / sqrt(head_dim)`).
    /// * `causal` — if `true`, apply causal (lower-triangular) mask.
    #[allow(clippy::too_many_arguments)]
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
    ) -> BackendResult<()>;

    /// Reduction along an axis.
    ///
    /// Reduces `input` along `axis` using the specified `op` and writes to `output`.
    fn reduce(
        &self,
        op: ReduceOp,
        input_ptr: u64,
        output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()>;

    /// Element-wise unary operation.
    ///
    /// Applies `op` to each of the `n` elements at `input_ptr` and writes to `output_ptr`.
    fn unary(&self, op: UnaryOp, input_ptr: u64, output_ptr: u64, n: usize) -> BackendResult<()>;

    /// Element-wise binary operation.
    ///
    /// Applies `op` element-wise: `output[i] = op(a[i], b[i])` for `n` elements.
    fn binary(
        &self,
        op: BinaryOp,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
        n: usize,
    ) -> BackendResult<()>;

    /// Mixed-precision GEMM: `C = alpha * op(A) * op(B) + beta * C` where the
    /// `A`/`B` operands are **stored** in a reduced 16-bit format
    /// ([`MixedPrecision::F16`] or [`MixedPrecision::Bf16`]) but the dot
    /// products **accumulate in `f32`** — the Tensor-Core / WMMA contract.
    ///
    /// Unlike [`gemm`](Self::gemm) (column-major `f64`), this operates on
    /// column-major **`f32`** buffers, matching the `f32` storage every GPU
    /// uploads to a half/bfloat16 GEMM. The CPU reference emulates the 16-bit
    /// storage by rounding each input element to the target format
    /// (round-to-nearest, ties-to-even) before the f32 accumulation, so its
    /// output equals what a real reduced-precision kernel would produce.
    ///
    /// # Arguments
    ///
    /// * `prec` — input storage format (`f16` or `bf16`); the accumulator and
    ///   output `C` are always `f32`.
    /// * `trans_a`, `trans_b` — transpose modes for A and B.
    /// * `m`, `n`, `k` — matrix dimensions (C is m×n, A is m×k, B is k×n after transpose).
    /// * `alpha`, `beta` — scaling factors (applied in `f32`).
    /// * `a_ptr`, `b_ptr`, `c_ptr` — device pointers to column-major `f32` matrices.
    /// * `lda`, `ldb`, `ldc` — leading dimensions.
    ///
    /// The default implementation returns [`BackendError::Unsupported`]; the
    /// [`CpuBackend`] implements the reference math, and GPU backends override
    /// it with a Tensor-Core kernel.
    #[allow(clippy::too_many_arguments)]
    fn gemm_mixed_precision(
        &self,
        prec: MixedPrecision,
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
        let _ = (
            prec, trans_a, trans_b, m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc,
        );
        Err(BackendError::Unsupported(
            "gemm_mixed_precision not implemented by this backend".into(),
        ))
    }

    /// Backward pass of [`conv2d_forward`](Self::conv2d_forward) w.r.t. the
    /// **input** (data gradient): given the upstream gradient `grad_output`,
    /// produce `grad_input` of the same shape as the forward input.
    ///
    /// Mathematically this is the *full* convolution of `grad_output` with the
    /// spatially-flipped filter (equivalently, the transpose of the forward
    /// im2col matrix applied to the upstream gradient). All tensors are
    /// row-major `f32` in NCHW / KCHW layout, matching
    /// [`conv2d_forward`](Self::conv2d_forward).
    ///
    /// # Arguments
    ///
    /// * `grad_output_ptr` / `grad_output_shape` — upstream gradient, `[N, K, Oh, Ow]`.
    /// * `filter_ptr` / `filter_shape` — forward filter, `[K, C, Fh, Fw]`.
    /// * `grad_input_ptr` / `grad_input_shape` — output data gradient, `[N, C, H, W]`.
    /// * `stride` — `[sh, sw]`; `padding` — `[ph, pw]` (same as the forward pass).
    ///
    /// The default returns [`BackendError::Unsupported`]; the [`CpuBackend`]
    /// implements the reference math.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_backward_data(
        &self,
        grad_output_ptr: u64,
        grad_output_shape: &[usize],
        filter_ptr: u64,
        filter_shape: &[usize],
        grad_input_ptr: u64,
        grad_input_shape: &[usize],
        stride: &[usize],
        padding: &[usize],
    ) -> BackendResult<()> {
        let _ = (
            grad_output_ptr,
            grad_output_shape,
            filter_ptr,
            filter_shape,
            grad_input_ptr,
            grad_input_shape,
            stride,
            padding,
        );
        Err(BackendError::Unsupported(
            "conv2d_backward_data not implemented by this backend".into(),
        ))
    }

    /// Backward pass of [`conv2d_forward`](Self::conv2d_forward) w.r.t. the
    /// **filter** (weight gradient): given the forward input and the upstream
    /// gradient `grad_output`, produce `grad_filter` of the same shape as the
    /// forward filter.
    ///
    /// Mathematically this is the correlation of `input` with `grad_output`
    /// (the forward im2col matrix multiplied by the upstream gradient). All
    /// tensors are row-major `f32` in NCHW / KCHW layout, matching
    /// [`conv2d_forward`](Self::conv2d_forward).
    ///
    /// # Arguments
    ///
    /// * `input_ptr` / `input_shape` — forward input, `[N, C, H, W]`.
    /// * `grad_output_ptr` / `grad_output_shape` — upstream gradient, `[N, K, Oh, Ow]`.
    /// * `grad_filter_ptr` / `grad_filter_shape` — output weight gradient, `[K, C, Fh, Fw]`.
    /// * `stride` — `[sh, sw]`; `padding` — `[ph, pw]` (same as the forward pass).
    ///
    /// The default returns [`BackendError::Unsupported`]; the [`CpuBackend`]
    /// implements the reference math.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_backward_filter(
        &self,
        input_ptr: u64,
        input_shape: &[usize],
        grad_output_ptr: u64,
        grad_output_shape: &[usize],
        grad_filter_ptr: u64,
        grad_filter_shape: &[usize],
        stride: &[usize],
        padding: &[usize],
    ) -> BackendResult<()> {
        let _ = (
            input_ptr,
            input_shape,
            grad_output_ptr,
            grad_output_shape,
            grad_filter_ptr,
            grad_filter_shape,
            stride,
            padding,
        );
        Err(BackendError::Unsupported(
            "conv2d_backward_filter not implemented by this backend".into(),
        ))
    }

    /// Numerically-stable softmax along `axis` of the tensor described by
    /// `shape` (row-major, `f32`).
    ///
    /// The default implementation returns
    /// [`BackendError::Unsupported`];
    /// the [`CpuBackend`] implements it directly, and GPU backends override
    /// it with a fused kernel. Consumers that need softmax on a backend that
    /// does not provide it can still compose it from
    /// `reduce(Max) + unary(Exp) + reduce(Sum) + binary(Div)`.
    fn softmax(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()> {
        let _ = (input_ptr, output_ptr, shape, axis);
        Err(BackendError::Unsupported(
            "softmax not implemented by this backend".into(),
        ))
    }

    /// Row-gather: copy the rows named by `indices` out of a `rows × cols`
    /// (`f32`, row-major) table into a contiguous output of
    /// `indices.len() × cols`.
    ///
    /// Needed by embedding tables and MoE routing. The default returns
    /// [`BackendError::Unsupported`].
    fn gather(
        &self,
        input_ptr: u64,
        indices: &[usize],
        output_ptr: u64,
        rows: usize,
        cols: usize,
    ) -> BackendResult<()> {
        let _ = (input_ptr, indices, output_ptr, rows, cols);
        Err(BackendError::Unsupported(
            "gather not implemented by this backend".into(),
        ))
    }

    /// Row-scatter: write each input row (`indices.len() × cols`, `f32`) into
    /// `output` at the destination row given by `indices`, preserving
    /// unreferenced rows of the `rows × cols` output table.
    ///
    /// The inverse routing primitive to [`gather`](Self::gather). The default
    /// returns [`BackendError::Unsupported`].
    fn scatter(
        &self,
        input_ptr: u64,
        indices: &[usize],
        output_ptr: u64,
        rows: usize,
        cols: usize,
    ) -> BackendResult<()> {
        let _ = (input_ptr, indices, output_ptr, rows, cols);
        Err(BackendError::Unsupported(
            "scatter not implemented by this backend".into(),
        ))
    }

    /// Strided batched GEMM: for each batch `b` in `0..batch_count`,
    /// compute `C_b = alpha * op(A_b) * op(B_b) + beta * C_b`
    /// where `A_b` starts at `a_ptr + b * stride_a * 4` bytes (f32 elements), etc.
    ///
    /// # Arguments
    ///
    /// * `trans_a`, `trans_b` — transpose modes for A and B.
    /// * `m`, `n`, `k` — matrix dimensions (C is m×n).
    /// * `alpha`, `beta` — scaling factors.
    /// * `a_ptr`, `b_ptr`, `c_ptr` — device pointers to the first matrix in each batch.
    /// * `lda`, `ldb`, `ldc` — leading dimensions.
    /// * `stride_a`, `stride_b`, `stride_c` — element strides between consecutive matrices.
    /// * `batch_count` — number of GEMM operations in the batch.
    ///
    /// The default implementation dispatches `batch_count` individual
    /// [`gemm`](Self::gemm) calls with pointer offsets.
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
        // Default: loop over individual gemm calls with byte-offset pointers.
        // Backends should override with a single batched kernel for efficiency.
        let elem_bytes: u64 = 4; // f32
        for b in 0..batch_count {
            let b64 = b as u64;
            self.gemm(
                trans_a,
                trans_b,
                m,
                n,
                k,
                alpha,
                a_ptr + b64 * stride_a as u64 * elem_bytes,
                lda,
                b_ptr + b64 * stride_b as u64 * elem_bytes,
                ldb,
                beta,
                c_ptr + b64 * stride_c as u64 * elem_bytes,
                ldc,
            )?;
        }
        Ok(())
    }

    /// Synchronize all pending operations on this backend.
    ///
    /// Blocks the host until all previously submitted GPU work completes.
    fn synchronize(&self) -> BackendResult<()>;

    /// Allocate device memory.
    ///
    /// Returns an opaque device pointer. The caller is responsible for
    /// eventually calling [`free`](ComputeBackend::free).
    fn alloc(&self, bytes: usize) -> BackendResult<u64>;

    /// Free device memory previously allocated with [`alloc`](ComputeBackend::alloc).
    fn free(&self, ptr: u64) -> BackendResult<()>;

    /// Copy data from host memory to device memory.
    ///
    /// * `dst` — device pointer (destination).
    /// * `src` — host byte slice (source).
    fn copy_htod(&self, dst: u64, src: &[u8]) -> BackendResult<()>;

    /// Copy data from device memory to host memory.
    ///
    /// * `dst` — host byte slice (destination).
    /// * `src` — device pointer (source).
    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> BackendResult<()>;
}

// ─── Blanket impl for &mut T ─────────────────────────────────

/// Forward every [`ComputeBackend`] method through a mutable reference, so
/// callers holding `&mut dyn ComputeBackend` (or `&mut T`) can pass it where
/// a `ComputeBackend` is expected without re-boxing.
impl<T: ComputeBackend + ?Sized> ComputeBackend for &mut T {
    fn name(&self) -> &str {
        (**self).name()
    }

    fn init(&mut self) -> BackendResult<()> {
        (**self).init()
    }

    fn is_initialized(&self) -> bool {
        (**self).is_initialized()
    }

    fn capabilities(&self) -> Capabilities {
        (**self).capabilities()
    }

    fn available_devices(&self) -> BackendResult<Vec<DeviceInfo>> {
        (**self).available_devices()
    }

    fn recommended_tile_for(&self, m: usize, n: usize, k: usize) -> TileShape {
        (**self).recommended_tile_for(m, n, k)
    }

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
        (**self).gemm(
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
        (**self).conv2d_forward(
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
        (**self).attention(
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
        (**self).reduce(op, input_ptr, output_ptr, shape, axis)
    }

    fn unary(&self, op: UnaryOp, input_ptr: u64, output_ptr: u64, n: usize) -> BackendResult<()> {
        (**self).unary(op, input_ptr, output_ptr, n)
    }

    fn binary(
        &self,
        op: BinaryOp,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
        n: usize,
    ) -> BackendResult<()> {
        (**self).binary(op, a_ptr, b_ptr, output_ptr, n)
    }

    fn gemm_mixed_precision(
        &self,
        prec: MixedPrecision,
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
        (**self).gemm_mixed_precision(
            prec, trans_a, trans_b, m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc,
        )
    }

    fn conv2d_backward_data(
        &self,
        grad_output_ptr: u64,
        grad_output_shape: &[usize],
        filter_ptr: u64,
        filter_shape: &[usize],
        grad_input_ptr: u64,
        grad_input_shape: &[usize],
        stride: &[usize],
        padding: &[usize],
    ) -> BackendResult<()> {
        (**self).conv2d_backward_data(
            grad_output_ptr,
            grad_output_shape,
            filter_ptr,
            filter_shape,
            grad_input_ptr,
            grad_input_shape,
            stride,
            padding,
        )
    }

    fn conv2d_backward_filter(
        &self,
        input_ptr: u64,
        input_shape: &[usize],
        grad_output_ptr: u64,
        grad_output_shape: &[usize],
        grad_filter_ptr: u64,
        grad_filter_shape: &[usize],
        stride: &[usize],
        padding: &[usize],
    ) -> BackendResult<()> {
        (**self).conv2d_backward_filter(
            input_ptr,
            input_shape,
            grad_output_ptr,
            grad_output_shape,
            grad_filter_ptr,
            grad_filter_shape,
            stride,
            padding,
        )
    }

    fn softmax(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()> {
        (**self).softmax(input_ptr, output_ptr, shape, axis)
    }

    fn gather(
        &self,
        input_ptr: u64,
        indices: &[usize],
        output_ptr: u64,
        rows: usize,
        cols: usize,
    ) -> BackendResult<()> {
        (**self).gather(input_ptr, indices, output_ptr, rows, cols)
    }

    fn scatter(
        &self,
        input_ptr: u64,
        indices: &[usize],
        output_ptr: u64,
        rows: usize,
        cols: usize,
    ) -> BackendResult<()> {
        (**self).scatter(input_ptr, indices, output_ptr, rows, cols)
    }

    fn synchronize(&self) -> BackendResult<()> {
        (**self).synchronize()
    }

    fn alloc(&self, bytes: usize) -> BackendResult<u64> {
        (**self).alloc(bytes)
    }

    fn free(&self, ptr: u64) -> BackendResult<()> {
        (**self).free(ptr)
    }

    fn copy_htod(&self, dst: u64, src: &[u8]) -> BackendResult<()> {
        (**self).copy_htod(dst, src)
    }

    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> BackendResult<()> {
        (**self).copy_dtoh(dst, src)
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Mock backend that records every call, for testing default impls
    //    and consumer dispatch logic without a GPU. ──

    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A `(operation, byte/elem count)` record kept by [`MockBackend`].
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct CallRecord {
        op: &'static str,
        count: usize,
    }

    #[derive(Debug)]
    struct MockBackend {
        gemm_call_count: AtomicUsize,
        log: Mutex<Vec<CallRecord>>,
    }

    impl MockBackend {
        fn new() -> Self {
            Self {
                gemm_call_count: AtomicUsize::new(0),
                log: Mutex::new(Vec::new()),
            }
        }

        fn record(&self, op: &'static str, count: usize) {
            self.log.lock().unwrap().push(CallRecord { op, count });
        }

        fn calls(&self) -> Vec<CallRecord> {
            self.log.lock().unwrap().clone()
        }
    }

    impl ComputeBackend for MockBackend {
        fn name(&self) -> &str {
            "mock"
        }
        fn init(&mut self) -> BackendResult<()> {
            Ok(())
        }
        fn is_initialized(&self) -> bool {
            true
        }
        fn gemm(
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
            self.gemm_call_count.fetch_add(1, Ordering::Relaxed);
            self.record("gemm", 1);
            Ok(())
        }
        fn conv2d_forward(
            &self,
            _: u64,
            _: &[usize],
            _: u64,
            _: &[usize],
            _: u64,
            _: &[usize],
            _: &[usize],
            _: &[usize],
        ) -> BackendResult<()> {
            self.record("conv2d_forward", 1);
            Ok(())
        }
        fn attention(
            &self,
            _: u64,
            _: u64,
            _: u64,
            _: u64,
            _: usize,
            _: usize,
            _: usize,
            _: usize,
            _: usize,
            _: f64,
            _: bool,
        ) -> BackendResult<()> {
            self.record("attention", 1);
            Ok(())
        }
        fn reduce(&self, _: ReduceOp, _: u64, _: u64, _: &[usize], _: usize) -> BackendResult<()> {
            self.record("reduce", 1);
            Ok(())
        }
        fn unary(&self, _: UnaryOp, _: u64, _: u64, n: usize) -> BackendResult<()> {
            self.record("unary", n);
            Ok(())
        }
        fn binary(&self, _: BinaryOp, _: u64, _: u64, _: u64, n: usize) -> BackendResult<()> {
            self.record("binary", n);
            Ok(())
        }
        fn synchronize(&self) -> BackendResult<()> {
            Ok(())
        }
        fn alloc(&self, bytes: usize) -> BackendResult<u64> {
            self.record("alloc", bytes);
            Ok(0)
        }
        fn free(&self, _: u64) -> BackendResult<()> {
            Ok(())
        }
        fn copy_htod(&self, _: u64, src: &[u8]) -> BackendResult<()> {
            self.record("copy_htod", src.len());
            Ok(())
        }
        fn copy_dtoh(&self, dst: &mut [u8], _: u64) -> BackendResult<()> {
            self.record("copy_dtoh", dst.len());
            Ok(())
        }
    }

    #[test]
    fn batched_gemm_zero_batch_is_noop() {
        let backend = MockBackend::new();
        let result = backend.batched_gemm(
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
        );
        assert!(result.is_ok());
        assert_eq!(backend.gemm_call_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn batched_gemm_default_calls_gemm_n_times() {
        let backend = MockBackend::new();
        let batch_count = 7;
        let result = backend.batched_gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::Trans,
            8,
            8,
            8,
            1.0,
            1000,
            8,
            64,
            2000,
            8,
            64,
            0.0,
            3000,
            8,
            64,
            batch_count,
        );
        assert!(result.is_ok());
        assert_eq!(backend.gemm_call_count.load(Ordering::Relaxed), batch_count);
    }

    #[test]
    fn batched_gemm_single_batch() {
        let backend = MockBackend::new();
        let result = backend.batched_gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            16,
            16,
            16,
            1.0,
            0,
            16,
            256,
            0,
            16,
            256,
            1.0,
            0,
            16,
            256,
            1,
        );
        assert!(result.is_ok());
        assert_eq!(backend.gemm_call_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn mock_records_dispatch_for_consumer_tests() {
        let backend = MockBackend::new();
        backend.alloc(128).unwrap();
        backend.copy_htod(0, &[0u8; 32]).unwrap();
        backend.unary(UnaryOp::Relu, 0, 0, 64).unwrap();
        backend.binary(BinaryOp::Add, 0, 0, 0, 64).unwrap();
        let calls = backend.calls();
        assert_eq!(
            calls,
            vec![
                CallRecord {
                    op: "alloc",
                    count: 128
                },
                CallRecord {
                    op: "copy_htod",
                    count: 32
                },
                CallRecord {
                    op: "unary",
                    count: 64
                },
                CallRecord {
                    op: "binary",
                    count: 64
                },
            ]
        );
    }

    #[test]
    fn default_softmax_gather_scatter_are_unsupported() {
        let backend = MockBackend::new();
        assert!(matches!(
            backend.softmax(0, 0, &[2, 2], 1),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            backend.gather(0, &[0], 0, 1, 1),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            backend.scatter(0, &[0], 0, 1, 1),
            Err(BackendError::Unsupported(_))
        ));
    }

    #[test]
    fn default_mixed_precision_and_conv_backward_are_unsupported() {
        let backend = MockBackend::new();
        assert!(matches!(
            backend.gemm_mixed_precision(
                MixedPrecision::Bf16,
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                2,
                2,
                2,
                1.0,
                0,
                2,
                0,
                2,
                0.0,
                0,
                2,
            ),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            backend.conv2d_backward_data(
                0,
                &[1, 1, 2, 2],
                0,
                &[1, 1, 2, 2],
                0,
                &[1, 1, 3, 3],
                &[1, 1],
                &[0, 0],
            ),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            backend.conv2d_backward_filter(
                0,
                &[1, 1, 3, 3],
                0,
                &[1, 1, 2, 2],
                0,
                &[1, 1, 2, 2],
                &[1, 1],
                &[0, 0],
            ),
            Err(BackendError::Unsupported(_))
        ));
    }

    #[test]
    fn default_capabilities_and_tile_hint() {
        let backend = MockBackend::new();
        // Default capabilities are the conservative CPU profile.
        assert_eq!(backend.capabilities(), Capabilities::cpu());
        // Tile hint flows through the default heuristic.
        assert_eq!(
            backend.recommended_tile_for(32, 32, 32),
            TileShape::new(16, 16, 16)
        );
        assert!(backend.available_devices().unwrap().is_empty());
    }

    #[test]
    fn mut_ref_blanket_forwards() {
        // A generic helper that only accepts something implementing the
        // trait *by value*. Passing `&mut MockBackend` here only compiles
        // because of the blanket `impl ComputeBackend for &mut T`.
        fn run_one_gemm<B: ComputeBackend>(mut be: B) -> BackendResult<()> {
            be.init()?;
            be.gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                2,
                2,
                2,
                1.0,
                0,
                2,
                0,
                2,
                0.0,
                0,
                2,
            )
        }

        let mut backend = MockBackend::new();
        run_one_gemm(&mut backend).unwrap();
        assert_eq!(backend.name(), "mock");
        assert_eq!(backend.gemm_call_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn object_safety_vec_of_mixed_backends() {
        // A Vec of heterogeneous backends behind dyn proves object safety.
        let backends: Vec<Box<dyn ComputeBackend>> = vec![
            Box::new(MockBackend::new()),
            Box::new(CpuBackend::new()),
            Box::new(NullBackend::new()),
        ];
        let names: Vec<&str> = backends.iter().map(|b| b.name()).collect();
        assert_eq!(names, vec!["mock", "cpu", "null"]);
        // Every backend can be synchronized through the trait object.
        for b in &backends {
            assert!(b.synchronize().is_ok());
        }
    }
}

// ─── Cross-backend conformance (CPU reference) ───────────────

#[cfg(test)]
mod conformance {
    //! Conformance tests that pin the documented numerical contract of every
    //! op against the [`CpuBackend`] reference implementation. A real GPU
    //! backend can run these same property checks against its own output to
    //! prove agreement with the host reference (that cross-*hardware* run
    //! requires the device and lives in the concrete backend crates).

    use super::*;

    /// Tiny deterministic LCG producing values in `[0, 1)` using the full
    /// 32-bit range (÷2³², never ÷2³¹).
    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_u32(&mut self) -> u32 {
            // Numerical Recipes LCG constants.
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.state >> 32) as u32
        }
        fn next_unit(&mut self) -> f64 {
            f64::from(self.next_u32()) / f64::from(u32::MAX) // full-range, ÷(2³²-1)
        }
    }

    fn upload_f64(be: &CpuBackend, data: &[f64]) -> u64 {
        let ptr = be.alloc(data.len() * 8).unwrap();
        let mut bytes = Vec::with_capacity(data.len() * 8);
        for &v in data {
            bytes.extend_from_slice(&v.to_ne_bytes());
        }
        be.copy_htod(ptr, &bytes).unwrap();
        ptr
    }
    fn download_f64(be: &CpuBackend, ptr: u64, len: usize) -> Vec<f64> {
        let mut bytes = vec![0u8; len * 8];
        be.copy_dtoh(&mut bytes, ptr).unwrap();
        bytes
            .chunks_exact(8)
            .map(|c| {
                let mut b = [0u8; 8];
                b.copy_from_slice(c);
                f64::from_ne_bytes(b)
            })
            .collect()
    }

    /// Naive column-major reference GEMM used as ground truth.
    #[allow(clippy::too_many_arguments)]
    fn ref_gemm(
        ta: BackendTranspose,
        tb: BackendTranspose,
        m: usize,
        n: usize,
        k: usize,
        a: &[f64],
        b: &[f64],
    ) -> Vec<f64> {
        let at = |row: usize, col: usize| -> f64 {
            match ta {
                BackendTranspose::NoTrans => a[col * m + row],
                _ => a[row * k + col],
            }
        };
        let bt = |row: usize, col: usize| -> f64 {
            match tb {
                BackendTranspose::NoTrans => b[col * k + row],
                _ => b[row * n + col],
            }
        };
        let mut c = vec![0.0f64; m * n];
        for j in 0..n {
            for i in 0..m {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += at(i, p) * bt(p, j);
                }
                c[j * m + i] = acc;
            }
        }
        c
    }

    #[test]
    fn gemm_matches_reference_across_all_transpose_combos() {
        let be = CpuBackend::new();
        let (m, n, k) = (3, 4, 5);
        let mut rng = Lcg::new(0xC0FFEE);

        for &ta in &[
            BackendTranspose::NoTrans,
            BackendTranspose::Trans,
            BackendTranspose::ConjTrans,
        ] {
            for &tb in &[
                BackendTranspose::NoTrans,
                BackendTranspose::Trans,
                BackendTranspose::ConjTrans,
            ] {
                // A is (op rows × op cols) flattened col-major in its stored
                // orientation; for NoTrans that is m×k, else k×m.
                let a_elems = m * k;
                let b_elems = k * n;
                let a: Vec<f64> = (0..a_elems).map(|_| rng.next_unit()).collect();
                let b: Vec<f64> = (0..b_elems).map(|_| rng.next_unit()).collect();

                let (lda, a_cols) = if ta == BackendTranspose::NoTrans {
                    (m, k)
                } else {
                    (k, m)
                };
                let (ldb, b_cols) = if tb == BackendTranspose::NoTrans {
                    (k, n)
                } else {
                    (n, k)
                };
                // Reformat A/B into exactly lda*cols / ldb*cols buffers.
                assert_eq!(a.len(), lda * a_cols);
                assert_eq!(b.len(), ldb * b_cols);

                let a_ptr = upload_f64(&be, &a);
                let b_ptr = upload_f64(&be, &b);
                let c_ptr = upload_f64(&be, &vec![0.0f64; m * n]);

                be.gemm(ta, tb, m, n, k, 1.0, a_ptr, lda, b_ptr, ldb, 0.0, c_ptr, m)
                    .unwrap();
                let got = download_f64(&be, c_ptr, m * n);
                let want = ref_gemm(ta, tb, m, n, k, &a, &b);
                for (g, w) in got.iter().zip(want.iter()) {
                    assert!((g - w).abs() < 1e-9, "gemm({ta},{tb}) mismatch: {g} vs {w}");
                }

                be.free(a_ptr).unwrap();
                be.free(b_ptr).unwrap();
                be.free(c_ptr).unwrap();
            }
        }
    }

    #[test]
    fn reference_backend_is_always_available_via_registry() {
        // The end-to-end "device-absent" story: with only defaults, the
        // registry selects the CPU backend, which then really runs a gemm.
        let reg = BackendRegistry::with_defaults();
        let chosen = reg.select_best().unwrap();
        assert_eq!(chosen, BackendKind::Cpu);

        let be = CpuBackend::new();
        let a = upload_f64(&be, &[1.0, 0.0, 0.0, 1.0]);
        let b = upload_f64(&be, &[7.0, 8.0, 9.0, 10.0]);
        let c = upload_f64(&be, &[0.0; 4]);
        be.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            a,
            2,
            b,
            2,
            0.0,
            c,
            2,
        )
        .unwrap();
        // Identity * B = B.
        assert_eq!(download_f64(&be, c, 4), vec![7.0, 8.0, 9.0, 10.0]);
    }
}
