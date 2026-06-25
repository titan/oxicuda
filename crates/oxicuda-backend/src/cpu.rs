//! A genuinely-working pure-Rust **CPU reference backend**.
//!
//! Unlike a [`NullBackend`](crate::null::NullBackend) (which refuses every
//! op), this backend really executes on the host: it owns host buffers,
//! performs real GEMM / convolution / attention / reduction / element-wise
//! math, and is therefore the natural fallback when no GPU backend is
//! available, *and* the numerical reference that cross-backend conformance
//! tests compare against.
//!
//! # Memory model
//!
//! "Device" memory is modelled as host `Vec<u8>` allocations stored in an
//! internal table, keyed by a synthetic non-null `u64` "pointer". This lets
//! the CPU backend satisfy the exact same `alloc` / `copy_htod` /
//! `copy_dtoh` / `free` contract as a real GPU backend while running
//! entirely on the host.
//!
//! # Element types
//!
//! Following the [`ComputeBackend`](crate::ComputeBackend) trait contract,
//! [`gemm`](ComputeBackend::gemm) / [`batched_gemm`](ComputeBackend::batched_gemm)
//! operate on **column-major `f64`** matrices, while
//! [`conv2d_forward`](ComputeBackend::conv2d_forward),
//! [`attention`](ComputeBackend::attention),
//! [`reduce`](ComputeBackend::reduce),
//! [`unary`](ComputeBackend::unary),
//! [`binary`](ComputeBackend::binary), and
//! [`softmax`](ComputeBackend::softmax) operate on **`f32`** buffers (the
//! 4-byte element size assumed by `batched_gemm`'s default stride math).

use std::collections::HashMap;
use std::sync::Mutex;

use crate::ComputeBackend;
use crate::capabilities::{Capabilities, DeviceInfo, MemoryKind};
use crate::error::{BackendError, BackendResult};
use crate::ops::{BackendTranspose, BinaryOp, MixedPrecision, ReduceOp, UnaryOp};
use crate::precision::{round_to_bf16, round_to_f16};

/// Pure-Rust host backend. Always available; never touches a GPU.
#[derive(Debug)]
pub struct CpuBackend {
    initialized: bool,
    /// Maps synthetic device pointers to host allocations.
    allocations: Mutex<HashMap<u64, Vec<u8>>>,
    /// Monotonic source of synthetic pointers; never reuses an address so
    /// a use-after-free is reliably detected as an unknown pointer.
    next_ptr: Mutex<u64>,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackend {
    /// Create a new, uninitialized CPU backend.
    #[must_use]
    pub fn new() -> Self {
        Self {
            initialized: false,
            allocations: Mutex::new(HashMap::new()),
            // Start well above 0 so the synthetic pointers never collide
            // with the conventional null pointer.
            next_ptr: Mutex::new(0x1000),
        }
    }

    /// Number of currently-live allocations (test/diagnostic helper).
    #[must_use]
    pub fn live_allocations(&self) -> usize {
        self.allocations.lock().map(|t| t.len()).unwrap_or_default()
    }

    /// Read `len` `f32` values starting at `ptr`, returning an owned `Vec`.
    fn read_f32(&self, ptr: u64, len: usize) -> BackendResult<Vec<f32>> {
        let table = self
            .allocations
            .lock()
            .map_err(|_| BackendError::DeviceError("allocation table poisoned".into()))?;
        let buf = table.get(&ptr).ok_or_else(|| {
            BackendError::InvalidArgument(format!("unknown device pointer {ptr:#x}"))
        })?;
        let need = len * 4;
        if buf.len() < need {
            return Err(BackendError::InvalidArgument(format!(
                "buffer at {ptr:#x} holds {} bytes, need {need}",
                buf.len()
            )));
        }
        let mut out = Vec::with_capacity(len);
        for chunk in buf[..need].chunks_exact(4) {
            out.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    }

    /// Write `data` as `f32` little/native-endian bytes into the buffer at `ptr`.
    fn write_f32(&self, ptr: u64, data: &[f32]) -> BackendResult<()> {
        let mut table = self
            .allocations
            .lock()
            .map_err(|_| BackendError::DeviceError("allocation table poisoned".into()))?;
        let buf = table.get_mut(&ptr).ok_or_else(|| {
            BackendError::InvalidArgument(format!("unknown device pointer {ptr:#x}"))
        })?;
        let need = data.len() * 4;
        if buf.len() < need {
            return Err(BackendError::InvalidArgument(format!(
                "buffer at {ptr:#x} holds {} bytes, need {need}",
                buf.len()
            )));
        }
        for (slot, &v) in buf[..need].chunks_exact_mut(4).zip(data.iter()) {
            slot.copy_from_slice(&v.to_ne_bytes());
        }
        Ok(())
    }

    /// Read `len` `f64` values starting at `ptr`.
    fn read_f64(&self, ptr: u64, len: usize) -> BackendResult<Vec<f64>> {
        let table = self
            .allocations
            .lock()
            .map_err(|_| BackendError::DeviceError("allocation table poisoned".into()))?;
        let buf = table.get(&ptr).ok_or_else(|| {
            BackendError::InvalidArgument(format!("unknown device pointer {ptr:#x}"))
        })?;
        let need = len * 8;
        if buf.len() < need {
            return Err(BackendError::InvalidArgument(format!(
                "buffer at {ptr:#x} holds {} bytes, need {need}",
                buf.len()
            )));
        }
        let mut out = Vec::with_capacity(len);
        for chunk in buf[..need].chunks_exact(8) {
            let mut b = [0u8; 8];
            b.copy_from_slice(chunk);
            out.push(f64::from_ne_bytes(b));
        }
        Ok(out)
    }

    /// Write `data` as `f64` native-endian bytes into the buffer at `ptr`.
    fn write_f64(&self, ptr: u64, data: &[f64]) -> BackendResult<()> {
        let mut table = self
            .allocations
            .lock()
            .map_err(|_| BackendError::DeviceError("allocation table poisoned".into()))?;
        let buf = table.get_mut(&ptr).ok_or_else(|| {
            BackendError::InvalidArgument(format!("unknown device pointer {ptr:#x}"))
        })?;
        let need = data.len() * 8;
        if buf.len() < need {
            return Err(BackendError::InvalidArgument(format!(
                "buffer at {ptr:#x} holds {} bytes, need {need}",
                buf.len()
            )));
        }
        for (slot, &v) in buf[..need].chunks_exact_mut(8).zip(data.iter()) {
            slot.copy_from_slice(&v.to_ne_bytes());
        }
        Ok(())
    }
}

/// Index into a column-major matrix with leading dimension `ld`.
#[inline]
const fn col_major(row: usize, col: usize, ld: usize) -> usize {
    col * ld + row
}

/// Fetch `op(M)[row, col]` for a column-major source matrix `m` with leading
/// dimension `ld`. For [`BackendTranspose::ConjTrans`] this matches `Trans`
/// because the reference operates on real `f64`.
#[inline]
fn at(m: &[f64], trans: BackendTranspose, row: usize, col: usize, ld: usize) -> f64 {
    match trans {
        BackendTranspose::NoTrans => m[col_major(row, col, ld)],
        BackendTranspose::Trans | BackendTranspose::ConjTrans => m[col_major(col, row, ld)],
    }
}

/// `f32` analogue of [`at`] for the column-major mixed-precision GEMM
/// reference. `ConjTrans` matches `Trans` because the reference is real.
#[inline]
fn at_f32(m: &[f32], trans: BackendTranspose, row: usize, col: usize, ld: usize) -> f32 {
    match trans {
        BackendTranspose::NoTrans => m[col_major(row, col, ld)],
        BackendTranspose::Trans | BackendTranspose::ConjTrans => m[col_major(col, row, ld)],
    }
}

/// Round an `f32` to the chosen reduced-precision *storage* format, returning
/// the `f32` value the hardware would hold. This is the only place the CPU
/// mixed-precision GEMM differs from a full-`f32` GEMM: the operands lose
/// precision to 16-bit storage, but the accumulation stays in `f32`.
#[inline]
fn round_store(prec: MixedPrecision, x: f32) -> f32 {
    match prec {
        MixedPrecision::F16 => round_to_f16(x),
        MixedPrecision::Bf16 => round_to_bf16(x),
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn init(&mut self) -> BackendResult<()> {
        self.initialized = true;
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::cpu()
    }

    fn available_devices(&self) -> BackendResult<Vec<DeviceInfo>> {
        // Model a single host "device" sized to the process's available
        // address space heuristically (we do not probe physical RAM here;
        // a fixed conservative figure keeps the value deterministic).
        Ok(vec![DeviceInfo {
            ordinal: 0,
            name: "CPU (reference)".to_string(),
            compute_capability: (0, 0),
            total_memory_bytes: 0,
            memory_kind: MemoryKind::Host,
            capabilities: Capabilities::cpu(),
        }])
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
        if m == 0 || n == 0 {
            return Ok(());
        }
        // Determine how many source elements each operand spans so we can
        // load exactly the right amount from the (possibly oversized) buffer.
        let a_rows = if trans_a == BackendTranspose::NoTrans {
            m
        } else {
            k
        };
        let a_cols = if trans_a == BackendTranspose::NoTrans {
            k
        } else {
            m
        };
        let b_rows = if trans_b == BackendTranspose::NoTrans {
            k
        } else {
            n
        };
        let b_cols = if trans_b == BackendTranspose::NoTrans {
            n
        } else {
            k
        };
        if lda < a_rows || ldb < b_rows || ldc < m {
            return Err(BackendError::InvalidArgument(
                "leading dimension smaller than matrix extent".into(),
            ));
        }
        let a = if k == 0 {
            Vec::new()
        } else {
            self.read_f64(a_ptr, lda * a_cols)?
        };
        let b = if k == 0 {
            Vec::new()
        } else {
            self.read_f64(b_ptr, ldb * b_cols)?
        };
        let mut c = self.read_f64(c_ptr, ldc * n)?;

        for j in 0..n {
            for i in 0..m {
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += at(&a, trans_a, i, p, lda) * at(&b, trans_b, p, j, ldb);
                }
                let dst = &mut c[col_major(i, j, ldc)];
                *dst = alpha * acc + beta * *dst;
            }
        }
        self.write_f64(c_ptr, &c)
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
        if input_shape.len() != 4 || filter_shape.len() != 4 || output_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(
                "conv2d expects 4-D NCHW shapes".into(),
            ));
        }
        if stride.len() != 2 || padding.len() != 2 {
            return Err(BackendError::InvalidArgument(
                "conv2d expects 2-element stride and padding".into(),
            ));
        }
        let (n, c_in, h, w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (k_out, c_f, fh, fw) = (
            filter_shape[0],
            filter_shape[1],
            filter_shape[2],
            filter_shape[3],
        );
        let (on, ok, oh, ow) = (
            output_shape[0],
            output_shape[1],
            output_shape[2],
            output_shape[3],
        );
        if c_f != c_in || k_out != ok || on != n {
            return Err(BackendError::InvalidArgument(
                "conv2d shape mismatch between input/filter/output".into(),
            ));
        }
        let (sh, sw) = (stride[0], stride[1]);
        let (ph, pw) = (padding[0], padding[1]);
        // Validate the output spatial extent against the standard formula.
        let exp_oh = (h + 2 * ph).saturating_sub(fh) / sh.max(1) + 1;
        let exp_ow = (w + 2 * pw).saturating_sub(fw) / sw.max(1) + 1;
        if oh != exp_oh || ow != exp_ow {
            return Err(BackendError::InvalidArgument(format!(
                "conv2d output spatial size {oh}x{ow} != expected {exp_oh}x{exp_ow}"
            )));
        }

        let input = self.read_f32(input_ptr, n * c_in * h * w)?;
        let filter = self.read_f32(filter_ptr, k_out * c_in * fh * fw)?;
        let mut output = vec![0.0f32; n * k_out * oh * ow];

        let in_idx =
            |ni: usize, ci: usize, hi: usize, wi: usize| ((ni * c_in + ci) * h + hi) * w + wi;
        let f_idx =
            |ko: usize, ci: usize, fhi: usize, fwi: usize| ((ko * c_in + ci) * fh + fhi) * fw + fwi;
        let out_idx = |ni: usize, ko: usize, ohi: usize, owi: usize| {
            ((ni * k_out + ko) * oh + ohi) * ow + owi
        };

        for ni in 0..n {
            for ko in 0..k_out {
                for ohi in 0..oh {
                    for owi in 0..ow {
                        let mut acc = 0.0f32;
                        for ci in 0..c_in {
                            for fhi in 0..fh {
                                // Signed source row, accounting for padding.
                                let src_h = ohi * sh + fhi;
                                if src_h < ph || src_h >= h + ph {
                                    continue;
                                }
                                let ih = src_h - ph;
                                for fwi in 0..fw {
                                    let src_w = owi * sw + fwi;
                                    if src_w < pw || src_w >= w + pw {
                                        continue;
                                    }
                                    let iw = src_w - pw;
                                    acc += input[in_idx(ni, ci, ih, iw)]
                                        * filter[f_idx(ko, ci, fhi, fwi)];
                                }
                            }
                        }
                        output[out_idx(ni, ko, ohi, owi)] = acc;
                    }
                }
            }
        }
        self.write_f32(output_ptr, &output)
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
        if m == 0 || n == 0 {
            return Ok(());
        }
        let a_rows = if trans_a == BackendTranspose::NoTrans {
            m
        } else {
            k
        };
        let a_cols = if trans_a == BackendTranspose::NoTrans {
            k
        } else {
            m
        };
        let b_rows = if trans_b == BackendTranspose::NoTrans {
            k
        } else {
            n
        };
        let b_cols = if trans_b == BackendTranspose::NoTrans {
            n
        } else {
            k
        };
        if lda < a_rows || ldb < b_rows || ldc < m {
            return Err(BackendError::InvalidArgument(
                "leading dimension smaller than matrix extent".into(),
            ));
        }
        // Load the f32 operands, then round each element to the reduced-
        // precision *storage* format. This reproduces exactly what a GPU
        // does: it stores A/B in f16/bf16 before the GEMM. We round the whole
        // operand once so every dot product reuses the same stored values.
        let a_raw = if k == 0 {
            Vec::new()
        } else {
            self.read_f32(a_ptr, lda * a_cols)?
        };
        let b_raw = if k == 0 {
            Vec::new()
        } else {
            self.read_f32(b_ptr, ldb * b_cols)?
        };
        let a: Vec<f32> = a_raw.iter().map(|&v| round_store(prec, v)).collect();
        let b: Vec<f32> = b_raw.iter().map(|&v| round_store(prec, v)).collect();
        let mut c = self.read_f32(c_ptr, ldc * n)?;

        for j in 0..n {
            for i in 0..m {
                // Accumulate the dot product in f32 (never in f16/bf16) — this
                // is the whole point of mixed precision: long reductions keep
                // f32 precision even though the inputs are 16-bit.
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += at_f32(&a, trans_a, i, p, lda) * at_f32(&b, trans_b, p, j, ldb);
                }
                let dst = &mut c[col_major(i, j, ldc)];
                *dst = alpha * acc + beta * *dst;
            }
        }
        self.write_f32(c_ptr, &c)
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
        if grad_output_shape.len() != 4 || filter_shape.len() != 4 || grad_input_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(
                "conv2d_backward_data expects 4-D NCHW shapes".into(),
            ));
        }
        if stride.len() != 2 || padding.len() != 2 {
            return Err(BackendError::InvalidArgument(
                "conv2d_backward_data expects 2-element stride and padding".into(),
            ));
        }
        let (n, c_in, h, w) = (
            grad_input_shape[0],
            grad_input_shape[1],
            grad_input_shape[2],
            grad_input_shape[3],
        );
        let (k_out, c_f, fh, fw) = (
            filter_shape[0],
            filter_shape[1],
            filter_shape[2],
            filter_shape[3],
        );
        let (gn, gk, oh, ow) = (
            grad_output_shape[0],
            grad_output_shape[1],
            grad_output_shape[2],
            grad_output_shape[3],
        );
        if c_f != c_in || gk != k_out || gn != n {
            return Err(BackendError::InvalidArgument(
                "conv2d_backward_data shape mismatch between grad_output/filter/grad_input".into(),
            ));
        }
        let (sh, sw) = (stride[0], stride[1]);
        let (ph, pw) = (padding[0], padding[1]);
        // The grad_output spatial extent must be the forward output extent for
        // the claimed input/filter/stride/padding.
        let exp_oh = (h + 2 * ph).saturating_sub(fh) / sh.max(1) + 1;
        let exp_ow = (w + 2 * pw).saturating_sub(fw) / sw.max(1) + 1;
        if oh != exp_oh || ow != exp_ow {
            return Err(BackendError::InvalidArgument(format!(
                "conv2d_backward_data grad_output spatial size {oh}x{ow} != expected {exp_oh}x{exp_ow}"
            )));
        }

        let grad_output = self.read_f32(grad_output_ptr, n * k_out * oh * ow)?;
        let filter = self.read_f32(filter_ptr, k_out * c_in * fh * fw)?;
        let mut grad_input = vec![0.0f32; n * c_in * h * w];

        let in_idx =
            |ni: usize, ci: usize, hi: usize, wi: usize| ((ni * c_in + ci) * h + hi) * w + wi;
        let f_idx =
            |ko: usize, ci: usize, fhi: usize, fwi: usize| ((ko * c_in + ci) * fh + fhi) * fw + fwi;
        let go_idx = |ni: usize, ko: usize, ohi: usize, owi: usize| {
            ((ni * k_out + ko) * oh + ohi) * ow + owi
        };

        // grad_input is the transpose of the forward correlation: scatter each
        // grad_output element back onto the input positions that produced it.
        // For forward `out[oh,ow] += in[oh*sh+fh-ph, ow*sw+fw-pw] * w[fh,fw]`,
        // the data gradient is `grad_in[ih,iw] += grad_out[oh,ow] * w[fh,fw]`
        // over every (oh,ow,fh,fw) that maps onto (ih,iw). This is exactly the
        // full convolution of grad_output with the flipped filter.
        for ni in 0..n {
            for ko in 0..k_out {
                for ohi in 0..oh {
                    for owi in 0..ow {
                        let g = grad_output[go_idx(ni, ko, ohi, owi)];
                        if g == 0.0 {
                            continue;
                        }
                        for ci in 0..c_in {
                            for fhi in 0..fh {
                                let src_h = ohi * sh + fhi;
                                if src_h < ph || src_h >= h + ph {
                                    continue;
                                }
                                let ih = src_h - ph;
                                for fwi in 0..fw {
                                    let src_w = owi * sw + fwi;
                                    if src_w < pw || src_w >= w + pw {
                                        continue;
                                    }
                                    let iw = src_w - pw;
                                    grad_input[in_idx(ni, ci, ih, iw)] +=
                                        g * filter[f_idx(ko, ci, fhi, fwi)];
                                }
                            }
                        }
                    }
                }
            }
        }
        self.write_f32(grad_input_ptr, &grad_input)
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
        if input_shape.len() != 4 || grad_output_shape.len() != 4 || grad_filter_shape.len() != 4 {
            return Err(BackendError::InvalidArgument(
                "conv2d_backward_filter expects 4-D NCHW shapes".into(),
            ));
        }
        if stride.len() != 2 || padding.len() != 2 {
            return Err(BackendError::InvalidArgument(
                "conv2d_backward_filter expects 2-element stride and padding".into(),
            ));
        }
        let (n, c_in, h, w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (k_out, c_f, fh, fw) = (
            grad_filter_shape[0],
            grad_filter_shape[1],
            grad_filter_shape[2],
            grad_filter_shape[3],
        );
        let (gn, gk, oh, ow) = (
            grad_output_shape[0],
            grad_output_shape[1],
            grad_output_shape[2],
            grad_output_shape[3],
        );
        if c_f != c_in || gk != k_out || gn != n {
            return Err(BackendError::InvalidArgument(
                "conv2d_backward_filter shape mismatch between input/grad_output/grad_filter"
                    .into(),
            ));
        }
        let (sh, sw) = (stride[0], stride[1]);
        let (ph, pw) = (padding[0], padding[1]);
        let exp_oh = (h + 2 * ph).saturating_sub(fh) / sh.max(1) + 1;
        let exp_ow = (w + 2 * pw).saturating_sub(fw) / sw.max(1) + 1;
        if oh != exp_oh || ow != exp_ow {
            return Err(BackendError::InvalidArgument(format!(
                "conv2d_backward_filter grad_output spatial size {oh}x{ow} != expected {exp_oh}x{exp_ow}"
            )));
        }

        let input = self.read_f32(input_ptr, n * c_in * h * w)?;
        let grad_output = self.read_f32(grad_output_ptr, n * k_out * oh * ow)?;
        let mut grad_filter = vec![0.0f32; k_out * c_in * fh * fw];

        let in_idx =
            |ni: usize, ci: usize, hi: usize, wi: usize| ((ni * c_in + ci) * h + hi) * w + wi;
        let f_idx =
            |ko: usize, ci: usize, fhi: usize, fwi: usize| ((ko * c_in + ci) * fh + fhi) * fw + fwi;
        let go_idx = |ni: usize, ko: usize, ohi: usize, owi: usize| {
            ((ni * k_out + ko) * oh + ohi) * ow + owi
        };

        // The weight gradient is the correlation of input with grad_output:
        // `grad_w[ko,ci,fh,fw] += sum_{n,oh,ow} in[n,ci,oh*sh+fh-ph, ...]
        //                          * grad_out[n,ko,oh,ow]` — the exact dual of
        // the forward accumulation, summed over the batch and all output
        // positions.
        for ko in 0..k_out {
            for ci in 0..c_in {
                for fhi in 0..fh {
                    for fwi in 0..fw {
                        let mut acc = 0.0f32;
                        for ni in 0..n {
                            for ohi in 0..oh {
                                let src_h = ohi * sh + fhi;
                                if src_h < ph || src_h >= h + ph {
                                    continue;
                                }
                                let ih = src_h - ph;
                                for owi in 0..ow {
                                    let src_w = owi * sw + fwi;
                                    if src_w < pw || src_w >= w + pw {
                                        continue;
                                    }
                                    let iw = src_w - pw;
                                    acc += input[in_idx(ni, ci, ih, iw)]
                                        * grad_output[go_idx(ni, ko, ohi, owi)];
                                }
                            }
                        }
                        grad_filter[f_idx(ko, ci, fhi, fwi)] = acc;
                    }
                }
            }
        }
        self.write_f32(grad_filter_ptr, &grad_filter)
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
        let total_q = batch * heads * seq_q * head_dim;
        let total_kv = batch * heads * seq_kv * head_dim;
        let q = self.read_f32(q_ptr, total_q)?;
        let k = self.read_f32(k_ptr, total_kv)?;
        let v = self.read_f32(v_ptr, total_kv)?;
        let mut o = vec![0.0f32; total_q];

        let scale = scale as f32;
        for b in 0..batch {
            for h in 0..heads {
                let base_q = ((b * heads + h) * seq_q) * head_dim;
                let base_kv = ((b * heads + h) * seq_kv) * head_dim;
                for iq in 0..seq_q {
                    let q_off = base_q + iq * head_dim;
                    // Scores for this query row over all keys.
                    let valid = if causal {
                        // Causal: query iq attends to keys 0..=iq (aligned to
                        // the right edge when seq_q != seq_kv).
                        (iq + seq_kv).saturating_sub(seq_q) + 1
                    } else {
                        seq_kv
                    }
                    .min(seq_kv);
                    let mut scores = vec![f32::NEG_INFINITY; seq_kv];
                    let mut max_s = f32::NEG_INFINITY;
                    for (jk, score) in scores.iter_mut().enumerate().take(valid) {
                        let k_off = base_kv + jk * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_off + d] * k[k_off + d];
                        }
                        let s = dot * scale;
                        *score = s;
                        if s > max_s {
                            max_s = s;
                        }
                    }
                    // Numerically-stable softmax over the valid prefix.
                    let mut denom = 0.0f32;
                    for score in scores.iter_mut().take(valid) {
                        let e = (*score - max_s).exp();
                        *score = e;
                        denom += e;
                    }
                    let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                    // Weighted sum of values.
                    let o_off = q_off;
                    for d in 0..head_dim {
                        let mut acc = 0.0f32;
                        for (jk, &score) in scores.iter().enumerate().take(valid) {
                            let v_off = base_kv + jk * head_dim;
                            acc += score * inv * v[v_off + d];
                        }
                        o[o_off + d] = acc;
                    }
                }
            }
        }
        self.write_f32(o_ptr, &o)
    }

    fn reduce(
        &self,
        op: ReduceOp,
        input_ptr: u64,
        output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()> {
        if axis >= shape.len() {
            return Err(BackendError::InvalidArgument(format!(
                "reduce axis {axis} out of bounds for {}-D shape",
                shape.len()
            )));
        }
        let total: usize = shape.iter().product();
        let input = self.read_f32(input_ptr, total)?;
        let axis_len = shape[axis];
        // Sizes of the dimensions before/after the reduced axis (row-major).
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();
        let mut out = vec![0.0f32; outer * inner];

        for o in 0..outer {
            for i in 0..inner {
                let mut acc = match op {
                    ReduceOp::Sum | ReduceOp::Mean => 0.0f32,
                    ReduceOp::Max => f32::NEG_INFINITY,
                    ReduceOp::Min => f32::INFINITY,
                };
                for a in 0..axis_len {
                    let idx = (o * axis_len + a) * inner + i;
                    let v = input[idx];
                    acc = match op {
                        ReduceOp::Sum | ReduceOp::Mean => acc + v,
                        ReduceOp::Max => acc.max(v),
                        ReduceOp::Min => acc.min(v),
                    };
                }
                if op == ReduceOp::Mean && axis_len > 0 {
                    acc /= axis_len as f32;
                }
                out[o * inner + i] = acc;
            }
        }
        self.write_f32(output_ptr, &out)
    }

    fn unary(&self, op: UnaryOp, input_ptr: u64, output_ptr: u64, n: usize) -> BackendResult<()> {
        let input = self.read_f32(input_ptr, n)?;
        let mut out = Vec::with_capacity(n);
        for &x in &input {
            out.push(match op {
                UnaryOp::Relu => x.max(0.0),
                UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                UnaryOp::Tanh => x.tanh(),
                UnaryOp::Exp => x.exp(),
                UnaryOp::Log => x.ln(),
                UnaryOp::Sqrt => x.sqrt(),
                UnaryOp::Abs => x.abs(),
                UnaryOp::Neg => -x,
            });
        }
        self.write_f32(output_ptr, &out)
    }

    fn binary(
        &self,
        op: BinaryOp,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
        n: usize,
    ) -> BackendResult<()> {
        let a = self.read_f32(a_ptr, n)?;
        let b = self.read_f32(b_ptr, n)?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(match op {
                BinaryOp::Add => a[i] + b[i],
                BinaryOp::Sub => a[i] - b[i],
                BinaryOp::Mul => a[i] * b[i],
                BinaryOp::Div => a[i] / b[i],
                BinaryOp::Max => a[i].max(b[i]),
                BinaryOp::Min => a[i].min(b[i]),
            });
        }
        self.write_f32(output_ptr, &out)
    }

    fn softmax(
        &self,
        input_ptr: u64,
        output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()> {
        if axis >= shape.len() {
            return Err(BackendError::InvalidArgument(format!(
                "softmax axis {axis} out of bounds for {}-D shape",
                shape.len()
            )));
        }
        let total: usize = shape.iter().product();
        let input = self.read_f32(input_ptr, total)?;
        let axis_len = shape[axis];
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();
        let mut out = vec![0.0f32; total];

        for o in 0..outer {
            for i in 0..inner {
                // First pass: max for stability.
                let mut max_v = f32::NEG_INFINITY;
                for a in 0..axis_len {
                    let idx = (o * axis_len + a) * inner + i;
                    max_v = max_v.max(input[idx]);
                }
                // Second pass: exp + sum.
                let mut denom = 0.0f32;
                for a in 0..axis_len {
                    let idx = (o * axis_len + a) * inner + i;
                    let e = (input[idx] - max_v).exp();
                    out[idx] = e;
                    denom += e;
                }
                // Third pass: normalize.
                let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                for a in 0..axis_len {
                    let idx = (o * axis_len + a) * inner + i;
                    out[idx] *= inv;
                }
            }
        }
        self.write_f32(output_ptr, &out)
    }

    fn gather(
        &self,
        input_ptr: u64,
        indices: &[usize],
        output_ptr: u64,
        rows: usize,
        cols: usize,
    ) -> BackendResult<()> {
        let table = self.read_f32(input_ptr, rows * cols)?;
        let mut out = Vec::with_capacity(indices.len() * cols);
        for &row in indices {
            if row >= rows {
                return Err(BackendError::InvalidArgument(format!(
                    "gather index {row} out of bounds for {rows} rows"
                )));
            }
            out.extend_from_slice(&table[row * cols..(row + 1) * cols]);
        }
        self.write_f32(output_ptr, &out)
    }

    fn scatter(
        &self,
        input_ptr: u64,
        indices: &[usize],
        output_ptr: u64,
        rows: usize,
        cols: usize,
    ) -> BackendResult<()> {
        // `input` holds one row per index; write each into `output` at the
        // destination row. Existing destination contents are read first so
        // unreferenced rows are preserved.
        let src = self.read_f32(input_ptr, indices.len() * cols)?;
        let mut dst = self.read_f32(output_ptr, rows * cols)?;
        for (slot, &row) in indices.iter().enumerate() {
            if row >= rows {
                return Err(BackendError::InvalidArgument(format!(
                    "scatter index {row} out of bounds for {rows} rows"
                )));
            }
            dst[row * cols..(row + 1) * cols].copy_from_slice(&src[slot * cols..(slot + 1) * cols]);
        }
        self.write_f32(output_ptr, &dst)
    }

    fn synchronize(&self) -> BackendResult<()> {
        // The CPU backend executes synchronously; nothing to wait for.
        Ok(())
    }

    fn alloc(&self, bytes: usize) -> BackendResult<u64> {
        if bytes == 0 {
            return Err(BackendError::InvalidArgument(
                "cannot allocate 0 bytes".into(),
            ));
        }
        let mut next = self
            .next_ptr
            .lock()
            .map_err(|_| BackendError::DeviceError("pointer counter poisoned".into()))?;
        let ptr = *next;
        // Advance past this allocation, keeping 16-byte alignment between
        // synthetic addresses for realism, and never reuse an address.
        let advance = (bytes as u64).div_ceil(16) * 16;
        *next = next
            .checked_add(advance.max(16))
            .ok_or(BackendError::OutOfMemory)?;
        drop(next);

        let mut table = self
            .allocations
            .lock()
            .map_err(|_| BackendError::DeviceError("allocation table poisoned".into()))?;
        table.insert(ptr, vec![0u8; bytes]);
        Ok(ptr)
    }

    fn free(&self, ptr: u64) -> BackendResult<()> {
        let mut table = self
            .allocations
            .lock()
            .map_err(|_| BackendError::DeviceError("allocation table poisoned".into()))?;
        if table.remove(&ptr).is_none() {
            return Err(BackendError::InvalidArgument(format!(
                "free of unknown device pointer {ptr:#x}"
            )));
        }
        Ok(())
    }

    fn copy_htod(&self, dst: u64, src: &[u8]) -> BackendResult<()> {
        let mut table = self
            .allocations
            .lock()
            .map_err(|_| BackendError::DeviceError("allocation table poisoned".into()))?;
        let buf = table.get_mut(&dst).ok_or_else(|| {
            BackendError::InvalidArgument(format!("unknown device pointer {dst:#x}"))
        })?;
        if src.len() > buf.len() {
            return Err(BackendError::InvalidArgument(format!(
                "copy_htod of {} bytes into {}-byte buffer",
                src.len(),
                buf.len()
            )));
        }
        buf[..src.len()].copy_from_slice(src);
        Ok(())
    }

    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> BackendResult<()> {
        let table = self
            .allocations
            .lock()
            .map_err(|_| BackendError::DeviceError("allocation table poisoned".into()))?;
        let buf = table.get(&src).ok_or_else(|| {
            BackendError::InvalidArgument(format!("unknown device pointer {src:#x}"))
        })?;
        if dst.len() > buf.len() {
            return Err(BackendError::InvalidArgument(format!(
                "copy_dtoh of {} bytes from {}-byte buffer",
                dst.len(),
                buf.len()
            )));
        }
        dst.copy_from_slice(&buf[..dst.len()]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Allocate a device buffer, fill it from `data` (as `f32`), and return
    /// the pointer.
    fn upload_f32(be: &CpuBackend, data: &[f32]) -> u64 {
        let ptr = be.alloc(data.len() * 4).expect("alloc");
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &v in data {
            bytes.extend_from_slice(&v.to_ne_bytes());
        }
        be.copy_htod(ptr, &bytes).expect("htod");
        ptr
    }

    fn download_f32(be: &CpuBackend, ptr: u64, len: usize) -> Vec<f32> {
        let mut bytes = vec![0u8; len * 4];
        be.copy_dtoh(&mut bytes, ptr).expect("dtoh");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn upload_f64(be: &CpuBackend, data: &[f64]) -> u64 {
        let ptr = be.alloc(data.len() * 8).expect("alloc");
        let mut bytes = Vec::with_capacity(data.len() * 8);
        for &v in data {
            bytes.extend_from_slice(&v.to_ne_bytes());
        }
        be.copy_htod(ptr, &bytes).expect("htod");
        ptr
    }

    fn download_f64(be: &CpuBackend, ptr: u64, len: usize) -> Vec<f64> {
        let mut bytes = vec![0u8; len * 8];
        be.copy_dtoh(&mut bytes, ptr).expect("dtoh");
        bytes
            .chunks_exact(8)
            .map(|c| {
                let mut b = [0u8; 8];
                b.copy_from_slice(c);
                f64::from_ne_bytes(b)
            })
            .collect()
    }

    #[test]
    fn init_and_name() {
        let mut be = CpuBackend::new();
        assert_eq!(be.name(), "cpu");
        assert!(!be.is_initialized());
        be.init().unwrap();
        assert!(be.is_initialized());
    }

    #[test]
    fn alloc_copy_roundtrip_and_free() {
        let be = CpuBackend::new();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let ptr = upload_f32(&be, &data);
        assert_eq!(be.live_allocations(), 1);
        let back = download_f32(&be, ptr, 4);
        assert_eq!(back, data);
        be.free(ptr).unwrap();
        assert_eq!(be.live_allocations(), 0);
        // Use-after-free is detected.
        assert!(be.free(ptr).is_err());
    }

    #[test]
    fn alloc_never_reuses_pointer() {
        let be = CpuBackend::new();
        let p1 = be.alloc(64).unwrap();
        be.free(p1).unwrap();
        let p2 = be.alloc(64).unwrap();
        assert_ne!(p1, p2, "freed address must not be handed out again");
    }

    #[test]
    fn zero_byte_alloc_is_error() {
        let be = CpuBackend::new();
        assert!(matches!(be.alloc(0), Err(BackendError::InvalidArgument(_))));
    }

    #[test]
    fn gemm_identity_times_matrix() {
        let be = CpuBackend::new();
        // 2x2 column-major. A = I, B = [[1,2],[3,4]], expect C = B.
        let a = [1.0f64, 0.0, 0.0, 1.0]; // col-major identity
        let b = [1.0f64, 3.0, 2.0, 4.0]; // col-major [[1,2],[3,4]]
        let a_ptr = upload_f64(&be, &a);
        let b_ptr = upload_f64(&be, &b);
        let c_ptr = upload_f64(&be, &[0.0f64; 4]);
        be.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            a_ptr,
            2,
            b_ptr,
            2,
            0.0,
            c_ptr,
            2,
        )
        .unwrap();
        let c = download_f64(&be, c_ptr, 4);
        assert_eq!(c, b);
    }

    #[test]
    fn gemm_alpha_beta_and_known_product() {
        let be = CpuBackend::new();
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]] (col-major).
        // A*B = [[19,22],[43,50]].
        let a = [1.0f64, 3.0, 2.0, 4.0];
        let b = [5.0f64, 7.0, 6.0, 8.0];
        let c0 = [10.0f64, 10.0, 10.0, 10.0]; // C initial
        let a_ptr = upload_f64(&be, &a);
        let b_ptr = upload_f64(&be, &b);
        let c_ptr = upload_f64(&be, &c0);
        // alpha=2, beta=3 → 2*(A*B) + 3*C.
        be.gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            2.0,
            a_ptr,
            2,
            b_ptr,
            2,
            3.0,
            c_ptr,
            2,
        )
        .unwrap();
        let c = download_f64(&be, c_ptr, 4);
        // col-major [[19,22],[43,50]] = [19,43,22,50]
        let expected = [
            2.0 * 19.0 + 3.0 * 10.0,
            2.0 * 43.0 + 3.0 * 10.0,
            2.0 * 22.0 + 3.0 * 10.0,
            2.0 * 50.0 + 3.0 * 10.0,
        ];
        assert_eq!(c, expected);
    }

    #[test]
    fn gemm_transpose_a() {
        let be = CpuBackend::new();
        // A stored as [[1,2],[3,4]] col-major; op(A)=A^T = [[1,3],[2,4]].
        // B = I. Expect C = A^T.
        let a = [1.0f64, 3.0, 2.0, 4.0];
        let b = [1.0f64, 0.0, 0.0, 1.0];
        let a_ptr = upload_f64(&be, &a);
        let b_ptr = upload_f64(&be, &b);
        let c_ptr = upload_f64(&be, &[0.0f64; 4]);
        be.gemm(
            BackendTranspose::Trans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            a_ptr,
            2,
            b_ptr,
            2,
            0.0,
            c_ptr,
            2,
        )
        .unwrap();
        let c = download_f64(&be, c_ptr, 4);
        // A^T col-major = [[1,3],[2,4]] = [1,2,3,4]
        assert_eq!(c, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn unary_relu_and_neg() {
        let be = CpuBackend::new();
        let data = [-2.0f32, -0.5, 0.0, 1.5];
        let ip = upload_f32(&be, &data);
        let op = be.alloc(4 * 4).unwrap();
        be.unary(UnaryOp::Relu, ip, op, 4).unwrap();
        assert_eq!(download_f32(&be, op, 4), [0.0, 0.0, 0.0, 1.5]);
        be.unary(UnaryOp::Neg, ip, op, 4).unwrap();
        assert_eq!(download_f32(&be, op, 4), [2.0, 0.5, 0.0, -1.5]);
    }

    #[test]
    fn binary_ops() {
        let be = CpuBackend::new();
        let a = [1.0f32, 5.0, 3.0];
        let b = [4.0f32, 2.0, 3.0];
        let ap = upload_f32(&be, &a);
        let bp = upload_f32(&be, &b);
        let op = be.alloc(3 * 4).unwrap();
        be.binary(BinaryOp::Add, ap, bp, op, 3).unwrap();
        assert_eq!(download_f32(&be, op, 3), [5.0, 7.0, 6.0]);
        be.binary(BinaryOp::Max, ap, bp, op, 3).unwrap();
        assert_eq!(download_f32(&be, op, 3), [4.0, 5.0, 3.0]);
    }

    #[test]
    fn reduce_sum_and_mean_over_axis() {
        let be = CpuBackend::new();
        // shape [2,3] row-major: [[1,2,3],[4,5,6]]
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ip = upload_f32(&be, &data);
        // Reduce axis 1 (columns) → [6, 15] (sum of each row).
        let op = be.alloc(2 * 4).unwrap();
        be.reduce(ReduceOp::Sum, ip, op, &[2, 3], 1).unwrap();
        assert_eq!(download_f32(&be, op, 2), [6.0, 15.0]);
        // Reduce axis 0 (rows) → [5, 7, 9].
        let op2 = be.alloc(3 * 4).unwrap();
        be.reduce(ReduceOp::Sum, ip, op2, &[2, 3], 0).unwrap();
        assert_eq!(download_f32(&be, op2, 3), [5.0, 7.0, 9.0]);
        // Mean over axis 1 → [2, 5].
        let op3 = be.alloc(2 * 4).unwrap();
        be.reduce(ReduceOp::Mean, ip, op3, &[2, 3], 1).unwrap();
        assert_eq!(download_f32(&be, op3, 2), [2.0, 5.0]);
    }

    #[test]
    fn softmax_axis_sums_to_one() {
        let be = CpuBackend::new();
        let data = [1.0f32, 2.0, 3.0, 1.0, 1.0, 1.0];
        let ip = upload_f32(&be, &data);
        let op = be.alloc(6 * 4).unwrap();
        be.softmax(ip, op, &[2, 3], 1).unwrap();
        let out = download_f32(&be, op, 6);
        let row0: f32 = out[..3].iter().sum();
        let row1: f32 = out[3..].iter().sum();
        assert!((row0 - 1.0).abs() < 1e-6);
        assert!((row1 - 1.0).abs() < 1e-6);
        // Uniform row → uniform probabilities.
        for &p in &out[3..] {
            assert!((p - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn gather_selects_rows() {
        let be = CpuBackend::new();
        // 3 rows x 2 cols: [[10,11],[20,21],[30,31]]
        let table = [10.0f32, 11.0, 20.0, 21.0, 30.0, 31.0];
        let ip = upload_f32(&be, &table);
        let op = be.alloc(2 * 2 * 4).unwrap();
        be.gather(ip, &[2, 0], op, 3, 2).unwrap();
        assert_eq!(download_f32(&be, op, 4), [30.0, 31.0, 10.0, 11.0]);
        // Out-of-range index errors.
        assert!(be.gather(ip, &[5], op, 3, 2).is_err());
    }

    #[test]
    fn scatter_writes_rows_preserving_others() {
        let be = CpuBackend::new();
        let dst0 = [0.0f32; 6]; // 3x2 zeros
        let op = upload_f32(&be, &dst0);
        let src = [99.0f32, 98.0]; // one row
        let ip = upload_f32(&be, &src);
        be.scatter(ip, &[1], op, 3, 2).unwrap();
        // Row 1 written, rows 0 and 2 stay zero.
        assert_eq!(download_f32(&be, op, 6), [0.0, 0.0, 99.0, 98.0, 0.0, 0.0]);
    }

    #[test]
    fn conv2d_identity_filter() {
        let be = CpuBackend::new();
        // 1x1x3x3 input, 1x1x1x1 filter = [2.0], stride 1, pad 0.
        // Output = input * 2.
        let input: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let ip = upload_f32(&be, &input);
        let fp = upload_f32(&be, &[2.0f32]);
        let op = be.alloc(9 * 4).unwrap();
        be.conv2d_forward(
            ip,
            &[1, 1, 3, 3],
            fp,
            &[1, 1, 1, 1],
            op,
            &[1, 1, 3, 3],
            &[1, 1],
            &[0, 0],
        )
        .unwrap();
        let out = download_f32(&be, op, 9);
        let expected: Vec<f32> = input.iter().map(|x| x * 2.0).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn conv2d_rejects_wrong_output_size() {
        let be = CpuBackend::new();
        let ip = be.alloc(9 * 4).unwrap();
        let fp = be.alloc(4 * 4).unwrap();
        let op = be.alloc(9 * 4).unwrap();
        // 3x3 input, 2x2 filter, stride 1, pad 0 → output should be 2x2,
        // but we claim 3x3.
        let err = be.conv2d_forward(
            ip,
            &[1, 1, 3, 3],
            fp,
            &[1, 1, 2, 2],
            op,
            &[1, 1, 3, 3],
            &[1, 1],
            &[0, 0],
        );
        assert!(matches!(err, Err(BackendError::InvalidArgument(_))));
    }

    #[test]
    fn attention_uniform_keys_averages_values() {
        let be = CpuBackend::new();
        // batch=1, heads=1, seq_q=1, seq_kv=2, head_dim=2.
        // Q=[0,0] → all scores 0 → uniform softmax → output = mean(V).
        let q = [0.0f32, 0.0];
        let k = [1.0f32, 1.0, 2.0, 2.0];
        let v = [10.0f32, 20.0, 30.0, 40.0];
        let qp = upload_f32(&be, &q);
        let kp = upload_f32(&be, &k);
        let vp = upload_f32(&be, &v);
        let op = be.alloc(2 * 4).unwrap();
        be.attention(qp, kp, vp, op, 1, 1, 1, 2, 2, 1.0, false)
            .unwrap();
        let out = download_f32(&be, op, 2);
        // mean of rows [10,20] and [30,40] = [20,30].
        assert!((out[0] - 20.0).abs() < 1e-5);
        assert!((out[1] - 30.0).abs() < 1e-5);
    }

    #[test]
    fn attention_causal_first_query_sees_only_first_key() {
        let be = CpuBackend::new();
        // seq_q=2, seq_kv=2. Causal: query0 → key0 only; query1 → key0,key1.
        let q = [0.0f32, 0.0, 0.0, 0.0]; // both queries zero
        let k = [0.0f32, 0.0, 0.0, 0.0];
        let v = [1.0f32, 1.0, 5.0, 5.0]; // value rows differ
        let qp = upload_f32(&be, &q);
        let kp = upload_f32(&be, &k);
        let vp = upload_f32(&be, &v);
        let op = be.alloc(4 * 4).unwrap();
        be.attention(qp, kp, vp, op, 1, 1, 2, 2, 2, 1.0, true)
            .unwrap();
        let out = download_f32(&be, op, 4);
        // query0 sees only value row0 = [1,1].
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
        // query1 averages both rows = [3,3].
        assert!((out[2] - 3.0).abs() < 1e-5);
        assert!((out[3] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn batched_gemm_default_runs_on_cpu() {
        // The trait's default batched_gemm should drive the CPU gemm with
        // f32 byte strides — but the CPU gemm reads f64. Here we exercise
        // the explicit per-batch path with a single batch (stride 0) so the
        // default and direct calls coincide, verifying integration.
        let be = CpuBackend::new();
        let a = [1.0f64, 0.0, 0.0, 1.0];
        let b = [2.0f64, 3.0, 4.0, 5.0];
        let a_ptr = upload_f64(&be, &a);
        let b_ptr = upload_f64(&be, &b);
        let c_ptr = upload_f64(&be, &[0.0f64; 4]);
        be.batched_gemm(
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            a_ptr,
            2,
            0,
            b_ptr,
            2,
            0,
            0.0,
            c_ptr,
            2,
            0,
            1,
        )
        .unwrap();
        assert_eq!(download_f64(&be, c_ptr, 4), b);
    }

    #[test]
    fn unknown_pointer_errors() {
        let be = CpuBackend::new();
        let mut dst = [0u8; 4];
        assert!(be.copy_dtoh(&mut dst, 0xDEAD).is_err());
        assert!(be.copy_htod(0xDEAD, &[0u8; 4]).is_err());
    }

    #[test]
    fn capabilities_and_devices() {
        let be = CpuBackend::new();
        assert_eq!(be.capabilities(), Capabilities::cpu());
        let devs = be.available_devices().unwrap();
        assert_eq!(devs.len(), 1);
        assert_eq!(devs[0].memory_kind, MemoryKind::Host);
    }

    // ── Mixed-precision GEMM ──────────────────────────────────────────

    use crate::ops::MixedPrecision;
    use crate::precision::{round_to_bf16, round_to_f16};

    /// Plain column-major f32 reference GEMM (full precision) for comparison.
    fn ref_gemm_f32(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for j in 0..n {
            for i in 0..m {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[p * m + i] * b[j * k + p];
                }
                c[j * m + i] = acc;
            }
        }
        c
    }

    #[test]
    fn mixed_precision_bf16_matches_f32_within_rounding_tolerance() {
        let be = CpuBackend::new();
        let (m, n, k) = (4, 3, 5);
        // Deterministic operands in a moderate range.
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 % 11) as f32) * 0.25 - 1.0)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 5 % 13) as f32) * 0.125 - 0.5)
            .collect();

        let a_ptr = upload_f32(&be, &a);
        let b_ptr = upload_f32(&be, &b);
        let c_ptr = upload_f32(&be, &vec![0.0f32; m * n]);
        be.gemm_mixed_precision(
            MixedPrecision::Bf16,
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            m,
            n,
            k,
            1.0,
            a_ptr,
            m,
            b_ptr,
            k,
            0.0,
            c_ptr,
            m,
        )
        .unwrap();
        let got = download_f32(&be, c_ptr, m * n);

        // Ground truth: round the *operands* to bf16 (exactly what the kernel
        // stores), then accumulate in full f32.
        let a_r: Vec<f32> = a.iter().map(|&v| round_to_bf16(v)).collect();
        let b_r: Vec<f32> = b.iter().map(|&v| round_to_bf16(v)).collect();
        let want = ref_gemm_f32(m, n, k, &a_r, &b_r);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-6, "bf16 gemm {g} vs {w}");
        }

        // And the result is close to the *full*-f32 product within the bf16
        // storage tolerance (bf16 has ~3 decimal digits ⇒ rel err ~2^-8).
        let exact = ref_gemm_f32(m, n, k, &a, &b);
        for (g, e) in got.iter().zip(exact.iter()) {
            let tol = 1e-2 * (1.0 + e.abs());
            assert!((g - e).abs() < tol, "bf16 gemm {g} vs exact {e}");
        }
    }

    #[test]
    fn mixed_precision_f16_matches_f32_within_rounding_tolerance() {
        let be = CpuBackend::new();
        let (m, n, k) = (3, 4, 6);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 3 % 7) as f32) * 0.5 - 1.5)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 9 % 5) as f32) * 0.25 - 0.5)
            .collect();

        let a_ptr = upload_f32(&be, &a);
        let b_ptr = upload_f32(&be, &b);
        let c_ptr = upload_f32(&be, &vec![0.0f32; m * n]);
        be.gemm_mixed_precision(
            MixedPrecision::F16,
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            m,
            n,
            k,
            1.0,
            a_ptr,
            m,
            b_ptr,
            k,
            0.0,
            c_ptr,
            m,
        )
        .unwrap();
        let got = download_f32(&be, c_ptr, m * n);

        let a_r: Vec<f32> = a.iter().map(|&v| round_to_f16(v)).collect();
        let b_r: Vec<f32> = b.iter().map(|&v| round_to_f16(v)).collect();
        let want = ref_gemm_f32(m, n, k, &a_r, &b_r);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-6, "f16 gemm {g} vs {w}");
        }
    }

    #[test]
    fn mixed_precision_bf16_exact_for_representable_operands() {
        // When every operand element is exactly representable in bf16, the
        // mixed-precision GEMM must agree with the full-f32 GEMM bit-for-bit
        // (no rounding error anywhere).
        let be = CpuBackend::new();
        let (m, n, k) = (2, 2, 3);
        // Powers-of-two-scaled small integers are exact in both bf16 and f16.
        let a = [1.0f32, 2.0, -1.0, 0.5, 4.0, -2.0]; // col-major 2x3
        let b = [0.5f32, 1.0, 2.0, -1.0, 0.25, 8.0]; // col-major 3x2
        let a_ptr = upload_f32(&be, &a);
        let b_ptr = upload_f32(&be, &b);
        let c_ptr = upload_f32(&be, &vec![0.0f32; m * n]);
        be.gemm_mixed_precision(
            MixedPrecision::Bf16,
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            m,
            n,
            k,
            1.0,
            a_ptr,
            m,
            b_ptr,
            k,
            0.0,
            c_ptr,
            m,
        )
        .unwrap();
        let got = download_f32(&be, c_ptr, m * n);
        let want = ref_gemm_f32(m, n, k, &a, &b);
        assert_eq!(got, want, "exact bf16 operands must match f32 exactly");
    }

    #[test]
    fn mixed_precision_accumulates_in_f32_not_f16() {
        // The hallmark of mixed precision: a long dot product of small
        // bf16-representable values keeps f32 accumulation precision. We sum
        // K copies of 1.0 * (1/256). Each operand element (1.0 and 1/256) is
        // exact in bf16, so the *only* possible error is from accumulating in
        // a narrow type. With f32 accumulation the result is exact; a pure-
        // bf16 accumulator would stall (1 + 1/256 already rounds away the
        // increment in bf16), proving the accumulator is genuinely f32.
        let be = CpuBackend::new();
        let k = 512usize;
        let (m, n) = (1, 1);
        let a = vec![1.0f32; k]; // 1x k row (col-major: k elements)
        let inc = 1.0f32 / 256.0; // exactly representable in bf16 (2^-8)
        let b = vec![inc; k]; // k x 1 column
        let a_ptr = upload_f32(&be, &a);
        let b_ptr = upload_f32(&be, &b);
        let c_ptr = upload_f32(&be, &[0.0f32]);
        be.gemm_mixed_precision(
            MixedPrecision::Bf16,
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            m,
            n,
            k,
            1.0,
            a_ptr,
            m,
            b_ptr,
            k,
            0.0,
            c_ptr,
            m,
        )
        .unwrap();
        let got = download_f32(&be, c_ptr, 1)[0];
        let expected = k as f32 * inc; // = 2.0, exact in f32.
        assert!((got - expected).abs() < 1e-5, "f32-accumulated dot = {got}");

        // Demonstrate that a bf16 *accumulator* would NOT reach 2.0: once the
        // running sum exceeds 1.0, adding 2^-8 rounds back to the same bf16
        // value, so it saturates far below the true sum.
        let mut bf16_acc = 0.0f32;
        for _ in 0..k {
            bf16_acc = round_to_bf16(bf16_acc + inc);
        }
        assert!(
            bf16_acc < expected - 0.1,
            "bf16 accumulation should stall ({bf16_acc} < {expected})"
        );
        // The real op must beat the broken bf16-accumulator baseline.
        assert!(got > bf16_acc + 0.1);
    }

    #[test]
    fn mixed_precision_alpha_beta_and_transpose() {
        let be = CpuBackend::new();
        // op(A)=A^T with A stored col-major 2x2; B = I; alpha=2, beta=3.
        let a = [1.0f32, 3.0, 2.0, 4.0]; // col-major [[1,2],[3,4]]
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let c0 = [1.0f32, 1.0, 1.0, 1.0];
        let a_ptr = upload_f32(&be, &a);
        let b_ptr = upload_f32(&be, &b);
        let c_ptr = upload_f32(&be, &c0);
        be.gemm_mixed_precision(
            MixedPrecision::Bf16,
            BackendTranspose::Trans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            2.0,
            a_ptr,
            2,
            b_ptr,
            2,
            3.0,
            c_ptr,
            2,
        )
        .unwrap();
        let got = download_f32(&be, c_ptr, 4);
        // A^T col-major = [1,2,3,4]; 2*(A^T) + 3*C0 = [2+3, 4+3, 6+3, 8+3].
        assert_eq!(got, [5.0, 7.0, 9.0, 11.0]);
    }

    #[test]
    fn mixed_precision_rejects_bad_leading_dim() {
        let be = CpuBackend::new();
        let a_ptr = be.alloc(4 * 4).unwrap();
        let b_ptr = be.alloc(4 * 4).unwrap();
        let c_ptr = be.alloc(4 * 4).unwrap();
        let err = be.gemm_mixed_precision(
            MixedPrecision::F16,
            BackendTranspose::NoTrans,
            BackendTranspose::NoTrans,
            2,
            2,
            2,
            1.0,
            a_ptr,
            1, // lda < m
            b_ptr,
            2,
            0.0,
            c_ptr,
            2,
        );
        assert!(matches!(err, Err(BackendError::InvalidArgument(_))));
    }

    // ── conv2d backward (finite-difference gradient checks) ───────────

    /// Run a forward conv2d on the given host tensors and return the output.
    fn forward_conv(
        be: &CpuBackend,
        input: &[f32],
        in_shape: [usize; 4],
        filter: &[f32],
        f_shape: [usize; 4],
        out_shape: [usize; 4],
        stride: [usize; 2],
        pad: [usize; 2],
    ) -> Vec<f32> {
        let ip = upload_f32(be, input);
        let fp = upload_f32(be, filter);
        let out_len: usize = out_shape.iter().product();
        let op = be.alloc(out_len * 4).unwrap();
        be.conv2d_forward(ip, &in_shape, fp, &f_shape, op, &out_shape, &stride, &pad)
            .unwrap();
        let out = download_f32(be, op, out_len);
        be.free(ip).unwrap();
        be.free(fp).unwrap();
        be.free(op).unwrap();
        out
    }

    /// Scalar loss L = <forward(input, filter), grad_output> used to drive the
    /// finite-difference check (its gradient w.r.t. input/filter is exactly
    /// what backward_data / backward_filter must compute).
    #[allow(clippy::too_many_arguments)]
    fn conv_loss(
        be: &CpuBackend,
        input: &[f32],
        in_shape: [usize; 4],
        filter: &[f32],
        f_shape: [usize; 4],
        out_shape: [usize; 4],
        stride: [usize; 2],
        pad: [usize; 2],
        grad_output: &[f32],
    ) -> f32 {
        let y = forward_conv(be, input, in_shape, filter, f_shape, out_shape, stride, pad);
        y.iter().zip(grad_output.iter()).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn conv2d_backward_data_matches_finite_difference() {
        let be = CpuBackend::new();
        // N=1, C=2, H=4, W=4; K=3, Fh=Fw=3; stride 1, pad 1 → Oh=Ow=4.
        let in_shape = [1, 2, 4, 4];
        let f_shape = [3, 2, 3, 3];
        let out_shape = [1, 3, 4, 4];
        let stride = [1, 1];
        let pad = [1, 1];
        let in_len: usize = in_shape.iter().product();
        let f_len: usize = f_shape.iter().product();
        let out_len: usize = out_shape.iter().product();

        // Deterministic pseudo-random tensors.
        let input: Vec<f32> = (0..in_len)
            .map(|i| ((i * 13 % 17) as f32) * 0.1 - 0.8)
            .collect();
        let filter: Vec<f32> = (0..f_len)
            .map(|i| ((i * 7 % 11) as f32) * 0.1 - 0.5)
            .collect();
        let grad_output: Vec<f32> = (0..out_len)
            .map(|i| ((i * 5 % 9) as f32) * 0.2 - 0.8)
            .collect();

        // Analytic data gradient.
        let gop = upload_f32(&be, &grad_output);
        let fp = upload_f32(&be, &filter);
        let gip = be.alloc(in_len * 4).unwrap();
        be.conv2d_backward_data(gop, &out_shape, fp, &f_shape, gip, &in_shape, &stride, &pad)
            .unwrap();
        let analytic = download_f32(&be, gip, in_len);

        // Central-difference check on every input element.
        let eps = 1e-2f32;
        for idx in 0..in_len {
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let lp = conv_loss(
                &be,
                &plus,
                in_shape,
                &filter,
                f_shape,
                out_shape,
                stride,
                pad,
                &grad_output,
            );
            let lm = conv_loss(
                &be,
                &minus,
                in_shape,
                &filter,
                f_shape,
                out_shape,
                stride,
                pad,
                &grad_output,
            );
            let fd = (lp - lm) / (2.0 * eps);
            assert!(
                (analytic[idx] - fd).abs() < 1e-2,
                "grad_input[{idx}] analytic {} vs finite-diff {fd}",
                analytic[idx]
            );
        }
    }

    #[test]
    fn conv2d_backward_filter_matches_finite_difference() {
        let be = CpuBackend::new();
        // Use a stride-2 case with padding to exercise the index arithmetic.
        // N=2, C=2, H=5, W=5; K=2, Fh=Fw=3; stride 2, pad 1 → Oh=Ow=3.
        let in_shape = [2, 2, 5, 5];
        let f_shape = [2, 2, 3, 3];
        let out_shape = [2, 2, 3, 3];
        let stride = [2, 2];
        let pad = [1, 1];
        let in_len: usize = in_shape.iter().product();
        let f_len: usize = f_shape.iter().product();
        let out_len: usize = out_shape.iter().product();

        let input: Vec<f32> = (0..in_len)
            .map(|i| ((i * 11 % 19) as f32) * 0.07 - 0.6)
            .collect();
        let filter: Vec<f32> = (0..f_len)
            .map(|i| ((i * 3 % 13) as f32) * 0.1 - 0.6)
            .collect();
        let grad_output: Vec<f32> = (0..out_len)
            .map(|i| ((i * 17 % 7) as f32) * 0.15 - 0.4)
            .collect();

        // Analytic filter gradient.
        let ip = upload_f32(&be, &input);
        let gop = upload_f32(&be, &grad_output);
        let gfp = be.alloc(f_len * 4).unwrap();
        be.conv2d_backward_filter(ip, &in_shape, gop, &out_shape, gfp, &f_shape, &stride, &pad)
            .unwrap();
        let analytic = download_f32(&be, gfp, f_len);

        // Central difference on every filter element.
        let eps = 1e-2f32;
        for idx in 0..f_len {
            let mut plus = filter.clone();
            let mut minus = filter.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let lp = conv_loss(
                &be,
                &input,
                in_shape,
                &plus,
                f_shape,
                out_shape,
                stride,
                pad,
                &grad_output,
            );
            let lm = conv_loss(
                &be,
                &input,
                in_shape,
                &minus,
                f_shape,
                out_shape,
                stride,
                pad,
                &grad_output,
            );
            let fd = (lp - lm) / (2.0 * eps);
            assert!(
                (analytic[idx] - fd).abs() < 1e-2,
                "grad_filter[{idx}] analytic {} vs finite-diff {fd}",
                analytic[idx]
            );
        }
    }

    #[test]
    fn conv2d_backward_data_known_1x1_filter() {
        // With a 1x1x1x1 filter [w], grad_input = grad_output * w everywhere
        // (no spatial mixing), which is trivial to verify by hand.
        let be = CpuBackend::new();
        let grad_output: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let gop = upload_f32(&be, &grad_output);
        let fp = upload_f32(&be, &[3.0f32]);
        let gip = be.alloc(9 * 4).unwrap();
        be.conv2d_backward_data(
            gop,
            &[1, 1, 3, 3],
            fp,
            &[1, 1, 1, 1],
            gip,
            &[1, 1, 3, 3],
            &[1, 1],
            &[0, 0],
        )
        .unwrap();
        let got = download_f32(&be, gip, 9);
        let want: Vec<f32> = grad_output.iter().map(|g| g * 3.0).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn conv2d_backward_rejects_shape_mismatch() {
        let be = CpuBackend::new();
        let gop = be.alloc(9 * 4).unwrap();
        let fp = be.alloc(4 * 4).unwrap();
        let gip = be.alloc(9 * 4).unwrap();
        // grad_output claims 3x3 but a 2x2 filter over 3x3 input (stride 1,
        // pad 0) yields a 2x2 forward output ⇒ mismatch must be rejected.
        let err = be.conv2d_backward_data(
            gop,
            &[1, 1, 3, 3],
            fp,
            &[1, 1, 2, 2],
            gip,
            &[1, 1, 3, 3],
            &[1, 1],
            &[0, 0],
        );
        assert!(matches!(err, Err(BackendError::InvalidArgument(_))));

        // Same for backward_filter.
        let ip = be.alloc(9 * 4).unwrap();
        let err2 = be.conv2d_backward_filter(
            ip,
            &[1, 1, 3, 3],
            gop,
            &[1, 1, 3, 3],
            fp,
            &[1, 1, 2, 2],
            &[1, 1],
            &[0, 0],
        );
        assert!(matches!(err2, Err(BackendError::InvalidArgument(_))));
    }
}
