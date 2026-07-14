//! Common types shared across BLAS operations.
//!
//! This module defines enumerations for controlling BLAS behaviour,
//! such as math precision mode and scalar pointer location, as well as
//! the [`GpuFloat`] trait that abstracts over GPU-compatible floating-point
//! types, [`VectorDesc`] for describing strided vector layouts, and
//! [`MatrixDesc`] / [`MatrixDescMut`] for describing dense matrices on the
//! device.

use std::marker::PhantomData;

use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::ir::PtxType;

use crate::error::{BlasError, BlasResult};

// ---------------------------------------------------------------------------
// MathMode — precision / throughput trade-off
// ---------------------------------------------------------------------------

/// Controls whether Tensor-Core (reduced-precision) paths are used.
///
/// When set to [`TensorCore`](Self::TensorCore), GEMM and similar routines
/// may use FP16/BF16/TF32 Tensor-Core instructions for improved throughput,
/// at the cost of slightly reduced numerical precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MathMode {
    /// Use only standard FMA pipelines (FP32/FP64). This is the default.
    Default,
    /// Allow Tensor-Core instructions when the device supports them.
    TensorCore,
    /// Use lowest precision available for maximum throughput.
    MaxPerformance,
}

// ---------------------------------------------------------------------------
// PointerMode — where scalar arguments reside
// ---------------------------------------------------------------------------

/// Specifies where scalar arguments (alpha, beta) reside.
///
/// Most users should leave this at [`Host`](Self::Host). Switching to
/// [`Device`](Self::Device) avoids a host-device synchronisation barrier
/// when scalars are already computed on the GPU (e.g. in a training loop).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PointerMode {
    /// Scalars are passed from host memory (default).
    Host,
    /// Scalars reside in device memory.
    Device,
}

// ---------------------------------------------------------------------------
// Layout — memory ordering of a dense matrix
// ---------------------------------------------------------------------------

/// Memory layout of a dense matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Row-major (C-style): `A[i][j] = ptr[i * lda + j]`.
    RowMajor,
    /// Column-major (Fortran-style): `A[i][j] = ptr[j * lda + i]`.
    ColMajor,
}

// ---------------------------------------------------------------------------
// Transpose — matrix transposition mode
// ---------------------------------------------------------------------------

/// Transpose mode for a matrix operand.
///
/// This mirrors the classic BLAS `TRANSA` / `TRANSB` parameter and determines
/// whether a matrix is used as-is, transposed, or conjugate-transposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Transpose {
    /// Use the matrix as-is (no transposition).
    NoTrans,
    /// Use the transpose of the matrix (A^T).
    Trans,
    /// Use the conjugate-transpose (A^H). For real types this is identical
    /// to [`Trans`](Self::Trans).
    ConjTrans,
}

// ---------------------------------------------------------------------------
// FillMode — upper / lower triangle selection
// ---------------------------------------------------------------------------

/// Specifies which triangle of a symmetric or triangular matrix is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FillMode {
    /// The upper triangle is stored / referenced.
    Upper,
    /// The lower triangle is stored / referenced.
    Lower,
    /// Full matrix (both triangles).
    Full,
}

// ---------------------------------------------------------------------------
// Side — left/right operand position
// ---------------------------------------------------------------------------

/// Specifies on which side a special matrix (symmetric / triangular) appears
/// in a two-operand BLAS-3 operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    /// The special matrix is on the left: `op(A) * B`.
    Left,
    /// The special matrix is on the right: `B * op(A)`.
    Right,
}

// ---------------------------------------------------------------------------
// DiagType — unit / non-unit diagonal
// ---------------------------------------------------------------------------

/// Specifies whether a triangular matrix has an implicit unit diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiagType {
    /// The diagonal entries are stored explicitly and used as-is.
    NonUnit,
    /// The diagonal is implicitly all ones (unit diagonal).
    Unit,
}

// ---------------------------------------------------------------------------
// GpuFloat — trait for GPU-compatible floating-point types
// ---------------------------------------------------------------------------

/// Trait for floating-point types that can be used in GPU BLAS kernels.
///
/// Provides the mapping between Rust types and PTX register types, element
/// sizes, and bit-level representation for passing scalars as kernel parameters.
///
/// The trait bound is deliberately minimal so that half-precision types
/// (`half::f16`, `half::bf16`) and FP8 types can implement it.
pub trait GpuFloat: Copy + Send + Sync + 'static + std::fmt::Debug + PartialOrd {
    /// The PTX register type used for this precision (e.g. `PtxType::F32`).
    const PTX_TYPE: PtxType;

    /// Size of one element in bytes.
    const SIZE: usize;

    /// A short name used in generated kernel names (e.g. `"f32"`, `"f64"`).
    const NAME: &'static str;

    /// Whether this type is eligible for Tensor-Core acceleration.
    const TENSOR_CORE_ELIGIBLE: bool;

    /// The accumulator type used when this type feeds a Tensor-Core MMA.
    ///
    /// For f32/f64 this is `Self`; for f16/bf16/FP8 it is typically `f32`.
    type Accumulator: GpuFloat;

    /// Converts the scalar to its raw bit representation as a `u64`.
    ///
    /// For `f32`, the upper 32 bits are zero. For `f64`, all 64 bits are used.
    /// This is how scalar constants are passed to PTX kernels.
    fn to_bits_u64(self) -> u64;

    /// Converts the scalar into its [`Accumulator`](Self::Accumulator)
    /// precision and returns that value's raw bit representation as a `u64`.
    ///
    /// The generated GEMM kernels declare their `alpha`/`beta` parameters in
    /// the *accumulator* precision (e.g. `f32` for `f16`/`bf16`/FP8 inputs),
    /// not the input precision, because the epilogue `alpha * acc + beta * C`
    /// runs in the accumulator's registers. Passing the raw input-precision
    /// bits (via [`to_bits_u64`](Self::to_bits_u64)) would make the kernel
    /// reinterpret, say, a 16-bit `0x3C00` (`1.0_f16`) as an `f32`, yielding a
    /// near-zero denormal and silently zeroing the whole product. This method
    /// performs the widening so the scalar the kernel reads matches its
    /// declared parameter type.
    ///
    /// For `f32`/`f64` (where the accumulator is `Self`) it is identical to
    /// [`to_bits_u64`](Self::to_bits_u64).
    fn to_accumulator_bits(self) -> u64;

    /// Reconstructs a value from its raw bit representation stored in a `u64`.
    fn from_bits_u64(bits: u64) -> Self;

    /// The zero value for this type (additive identity).
    fn gpu_zero() -> Self;

    /// The one value for this type (multiplicative identity).
    fn gpu_one() -> Self;

    /// Size of one element in bytes, as `u32`.
    ///
    /// Convenience helper for PTX code-generation where `u32` strides are
    /// expected (e.g. `byte_offset_addr`).
    #[inline]
    fn size_u32() -> u32 {
        Self::SIZE as u32
    }
}

// -- f32 impl -----------------------------------------------------------------

impl GpuFloat for f32 {
    const PTX_TYPE: PtxType = PtxType::F32;
    const SIZE: usize = 4;
    const NAME: &'static str = "f32";
    const TENSOR_CORE_ELIGIBLE: bool = true;
    type Accumulator = f32;

    #[inline]
    fn to_bits_u64(self) -> u64 {
        u64::from(self.to_bits())
    }

    #[inline]
    fn to_accumulator_bits(self) -> u64 {
        // Accumulator == f32 == Self.
        self.to_bits_u64()
    }

    #[inline]
    fn from_bits_u64(bits: u64) -> Self {
        f32::from_bits(bits as u32)
    }

    #[inline]
    fn gpu_zero() -> Self {
        0.0
    }

    #[inline]
    fn gpu_one() -> Self {
        1.0
    }
}

// -- f64 impl -----------------------------------------------------------------

impl GpuFloat for f64 {
    const PTX_TYPE: PtxType = PtxType::F64;
    const SIZE: usize = 8;
    const NAME: &'static str = "f64";
    const TENSOR_CORE_ELIGIBLE: bool = true;
    type Accumulator = f64;

    #[inline]
    fn to_bits_u64(self) -> u64 {
        self.to_bits()
    }

    #[inline]
    fn to_accumulator_bits(self) -> u64 {
        // Accumulator == f64 == Self.
        self.to_bits_u64()
    }

    #[inline]
    fn from_bits_u64(bits: u64) -> Self {
        f64::from_bits(bits)
    }

    #[inline]
    fn gpu_zero() -> Self {
        0.0
    }

    #[inline]
    fn gpu_one() -> Self {
        1.0
    }
}

// -- half::f16 impl (feature-gated) ------------------------------------------

#[cfg(feature = "f16")]
impl GpuFloat for half::f16 {
    const PTX_TYPE: PtxType = PtxType::F16;
    const SIZE: usize = 2;
    const NAME: &'static str = "f16";
    const TENSOR_CORE_ELIGIBLE: bool = true;
    type Accumulator = f32;

    #[inline]
    fn to_bits_u64(self) -> u64 {
        u64::from(self.to_bits())
    }

    #[inline]
    fn to_accumulator_bits(self) -> u64 {
        // Accumulator is f32: widen the half to f32 before taking bits.
        u64::from(f32::from(self).to_bits())
    }

    #[inline]
    fn from_bits_u64(bits: u64) -> Self {
        half::f16::from_bits(bits as u16)
    }

    #[inline]
    fn gpu_zero() -> Self {
        half::f16::ZERO
    }

    #[inline]
    fn gpu_one() -> Self {
        half::f16::ONE
    }
}

// -- half::bf16 impl (feature-gated) -----------------------------------------

#[cfg(feature = "f16")]
impl GpuFloat for half::bf16 {
    const PTX_TYPE: PtxType = PtxType::BF16;
    const SIZE: usize = 2;
    const NAME: &'static str = "bf16";
    const TENSOR_CORE_ELIGIBLE: bool = true;
    type Accumulator = f32;

    #[inline]
    fn to_bits_u64(self) -> u64 {
        u64::from(self.to_bits())
    }

    #[inline]
    fn to_accumulator_bits(self) -> u64 {
        // Accumulator is f32: widen the bf16 to f32 before taking bits.
        u64::from(f32::from(self).to_bits())
    }

    #[inline]
    fn from_bits_u64(bits: u64) -> Self {
        half::bf16::from_bits(bits as u16)
    }

    #[inline]
    fn gpu_zero() -> Self {
        half::bf16::ZERO
    }

    #[inline]
    fn gpu_one() -> Self {
        half::bf16::ONE
    }
}

// ---------------------------------------------------------------------------
// FP8 types — Hopper+ (SM90) reduced-precision formats
// ---------------------------------------------------------------------------

/// FP8 E4M3 format (4-bit exponent, 3-bit mantissa).
///
/// Used primarily for inference on Hopper+ GPUs. The dynamic range is smaller
/// than E5M2 but the extra mantissa bit gives better precision for weights
/// and activations that stay within range.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct E4M3(pub u8);

/// FP8 E5M2 format (5-bit exponent, 2-bit mantissa).
///
/// Used primarily for training gradients on Hopper+ GPUs. The wider exponent
/// range accommodates the larger dynamic range of gradient values.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct E5M2(pub u8);

// SAFETY: E4M3 and E5M2 are `#[repr(transparent)]` wrappers around `u8`,
// which is trivially Send + Sync.
unsafe impl Send for E4M3 {}
unsafe impl Sync for E4M3 {}
unsafe impl Send for E5M2 {}
unsafe impl Sync for E5M2 {}

/// Decodes an OCP FP8 E4M3 (`e4m3fn`, finite -- no infinities) byte to `f32`.
///
/// Layout: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits. The single
/// pattern `S.1111.111` is NaN; every other `exp == 0b1111` value is a normal
/// number (max normal 448). `exp == 0` encodes zero / subnormals.
#[inline]
fn e4m3_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 != 0 { -1.0_f32 } else { 1.0_f32 };
    let exp = (bits >> 3) & 0x0F;
    let mant = bits & 0x07;
    if exp == 0 {
        // Zero or subnormal: 2^(1 - 7) * (mant / 8).
        sign * 2.0_f32.powi(-6) * (f32::from(mant) / 8.0)
    } else if exp == 0x0F && mant == 0x07 {
        f32::NAN
    } else {
        sign * 2.0_f32.powi(i32::from(exp) - 7) * (1.0 + f32::from(mant) / 8.0)
    }
}

/// Decodes an OCP FP8 E5M2 byte to `f32`.
///
/// Layout: 1 sign bit, 5 exponent bits (bias 15), 2 mantissa bits, with
/// IEEE-style specials: `exp == 0b11111` is infinity (mantissa 0) or NaN.
/// `exp == 0` encodes zero / subnormals.
#[inline]
fn e5m2_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 != 0 { -1.0_f32 } else { 1.0_f32 };
    let exp = (bits >> 2) & 0x1F;
    let mant = bits & 0x03;
    if exp == 0 {
        // Zero or subnormal: 2^(1 - 15) * (mant / 4).
        sign * 2.0_f32.powi(-14) * (f32::from(mant) / 4.0)
    } else if exp == 0x1F {
        if mant == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * 2.0_f32.powi(i32::from(exp) - 15) * (1.0 + f32::from(mant) / 4.0)
    }
}

impl GpuFloat for E4M3 {
    const PTX_TYPE: PtxType = PtxType::E4M3;
    const SIZE: usize = 1;
    const NAME: &'static str = "e4m3";
    const TENSOR_CORE_ELIGIBLE: bool = true;
    type Accumulator = f32;

    #[inline]
    fn to_bits_u64(self) -> u64 {
        u64::from(self.0)
    }

    #[inline]
    fn to_accumulator_bits(self) -> u64 {
        // Accumulator is f32: decode the FP8 byte to f32 before taking bits.
        u64::from(e4m3_to_f32(self.0).to_bits())
    }

    #[inline]
    fn from_bits_u64(bits: u64) -> Self {
        Self(bits as u8)
    }

    #[inline]
    fn gpu_zero() -> Self {
        Self(0x00)
    }

    #[inline]
    fn gpu_one() -> Self {
        Self(0x38)
    }
}

impl GpuFloat for E5M2 {
    const PTX_TYPE: PtxType = PtxType::E5M2;
    const SIZE: usize = 1;
    const NAME: &'static str = "e5m2";
    const TENSOR_CORE_ELIGIBLE: bool = true;
    type Accumulator = f32;

    #[inline]
    fn to_bits_u64(self) -> u64 {
        u64::from(self.0)
    }

    #[inline]
    fn to_accumulator_bits(self) -> u64 {
        // Accumulator is f32: decode the FP8 byte to f32 before taking bits.
        u64::from(e5m2_to_f32(self.0).to_bits())
    }

    #[inline]
    fn from_bits_u64(bits: u64) -> Self {
        Self(bits as u8)
    }

    #[inline]
    fn gpu_zero() -> Self {
        Self(0x00)
    }

    #[inline]
    fn gpu_one() -> Self {
        Self(0x3C)
    }
}

// ---------------------------------------------------------------------------
// VectorDesc — describes a strided vector on the device
// ---------------------------------------------------------------------------

/// Describes the layout of a vector stored in device memory.
///
/// BLAS Level 1 routines work on vectors that may be stored with a stride
/// (increment) between consecutive logical elements. This struct captures
/// the logical length, the stride, and the required buffer capacity.
#[derive(Debug, Clone, Copy)]
pub struct VectorDesc {
    /// Number of logical elements.
    pub n: u32,
    /// Stride (increment) between consecutive elements. Must be positive.
    pub inc: u32,
}

impl VectorDesc {
    /// Creates a new vector descriptor.
    ///
    /// # Arguments
    ///
    /// * `n` — number of logical elements.
    /// * `inc` — stride between elements (absolute value of the user-supplied
    ///   increment). Must be at least 1.
    #[must_use]
    pub fn new(n: u32, inc: u32) -> Self {
        Self { n, inc }
    }

    /// Returns the minimum number of elements the backing buffer must hold.
    ///
    /// For a vector of `n` elements with stride `inc`, the last element is at
    /// index `(n - 1) * inc`, so the buffer needs at least `1 + (n-1) * inc`
    /// elements.
    #[must_use]
    pub fn required_elements(&self) -> usize {
        if self.n == 0 {
            return 0;
        }
        1 + (self.n as usize - 1) * self.inc as usize
    }
}

// ---------------------------------------------------------------------------
// MatrixDesc — describes a dense matrix on the device (immutable view)
// ---------------------------------------------------------------------------

/// Describes a matrix stored in device memory.
///
/// This is an immutable (read-only) view. For an output matrix that will be
/// written to, use [`MatrixDescMut`].
///
/// All fields are `Copy`-sized, so `MatrixDesc` itself is `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct MatrixDesc<T: GpuFloat> {
    /// Device pointer to the matrix data.
    pub ptr: CUdeviceptr,
    /// Number of rows.
    pub rows: u32,
    /// Number of columns.
    pub cols: u32,
    /// Leading dimension (stride between rows/columns depending on layout).
    pub ld: u32,
    /// Memory layout.
    pub layout: Layout,
    _phantom: PhantomData<T>,
}

impl<T: GpuFloat> MatrixDesc<T> {
    /// Create a matrix descriptor from a [`DeviceBuffer`].
    ///
    /// Returns an error if the buffer is too small for the requested dimensions.
    pub fn from_buffer(
        buf: &DeviceBuffer<T>,
        rows: u32,
        cols: u32,
        layout: Layout,
    ) -> BlasResult<Self> {
        let required = rows as usize * cols as usize;
        if buf.len() < required {
            return Err(BlasError::BufferTooSmall {
                expected: required,
                actual: buf.len(),
            });
        }
        let ld = match layout {
            Layout::RowMajor => cols,
            Layout::ColMajor => rows,
        };
        Ok(Self {
            ptr: buf.as_device_ptr(),
            rows,
            cols,
            ld,
            layout,
            _phantom: PhantomData,
        })
    }

    /// Create with a raw device pointer (no size validation).
    pub fn from_raw(ptr: CUdeviceptr, rows: u32, cols: u32, ld: u32, layout: Layout) -> Self {
        Self {
            ptr,
            rows,
            cols,
            ld,
            layout,
            _phantom: PhantomData,
        }
    }

    /// Override the leading dimension.
    #[must_use]
    pub fn with_ld(mut self, ld: u32) -> Self {
        self.ld = ld;
        self
    }

    /// Total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.rows as usize * self.cols as usize
    }

    /// Storage bytes (full stride, including padding from leading dimension).
    #[must_use]
    pub fn storage_bytes(&self) -> usize {
        let major = match self.layout {
            Layout::RowMajor => self.rows,
            Layout::ColMajor => self.cols,
        };
        major as usize * self.ld as usize * T::SIZE
    }

    /// Effective dimensions after transpose.
    #[must_use]
    pub fn effective_dims(&self, trans: Transpose) -> (u32, u32) {
        match trans {
            Transpose::NoTrans => (self.rows, self.cols),
            Transpose::Trans | Transpose::ConjTrans => (self.cols, self.rows),
        }
    }
}

// ---------------------------------------------------------------------------
// MatrixDescMut — mutable matrix descriptor
// ---------------------------------------------------------------------------

/// Describes a mutable (output) matrix stored in device memory.
///
/// Identical to [`MatrixDesc`] but signals intent to write. This distinction
/// prevents accidentally passing an input buffer where an output is expected.
#[derive(Debug, Clone, Copy)]
pub struct MatrixDescMut<T: GpuFloat> {
    /// Device pointer to the matrix data.
    pub ptr: CUdeviceptr,
    /// Number of rows.
    pub rows: u32,
    /// Number of columns.
    pub cols: u32,
    /// Leading dimension (stride between rows/columns depending on layout).
    pub ld: u32,
    /// Memory layout.
    pub layout: Layout,
    _phantom: PhantomData<T>,
}

impl<T: GpuFloat> MatrixDescMut<T> {
    /// Create a mutable matrix descriptor from a [`DeviceBuffer`].
    ///
    /// Returns an error if the buffer is too small for the requested dimensions.
    pub fn from_buffer(
        buf: &mut DeviceBuffer<T>,
        rows: u32,
        cols: u32,
        layout: Layout,
    ) -> BlasResult<Self> {
        let required = rows as usize * cols as usize;
        if buf.len() < required {
            return Err(BlasError::BufferTooSmall {
                expected: required,
                actual: buf.len(),
            });
        }
        let ld = match layout {
            Layout::RowMajor => cols,
            Layout::ColMajor => rows,
        };
        Ok(Self {
            ptr: buf.as_device_ptr(),
            rows,
            cols,
            ld,
            layout,
            _phantom: PhantomData,
        })
    }

    /// Create with a raw device pointer (no size validation).
    pub fn from_raw(ptr: CUdeviceptr, rows: u32, cols: u32, ld: u32, layout: Layout) -> Self {
        Self {
            ptr,
            rows,
            cols,
            ld,
            layout,
            _phantom: PhantomData,
        }
    }

    /// Override the leading dimension.
    #[must_use]
    pub fn with_ld(mut self, ld: u32) -> Self {
        self.ld = ld;
        self
    }

    /// Total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.rows as usize * self.cols as usize
    }

    /// Storage bytes (full stride, including padding from leading dimension).
    #[must_use]
    pub fn storage_bytes(&self) -> usize {
        let major = match self.layout {
            Layout::RowMajor => self.rows,
            Layout::ColMajor => self.cols,
        };
        major as usize * self.ld as usize * T::SIZE
    }

    /// Effective dimensions after transpose.
    #[must_use]
    pub fn effective_dims(&self, trans: Transpose) -> (u32, u32) {
        match trans {
            Transpose::NoTrans => (self.rows, self.cols),
            Transpose::Trans | Transpose::ConjTrans => (self.cols, self.rows),
        }
    }

    /// Borrow as an immutable [`MatrixDesc`].
    #[must_use]
    pub fn as_immutable(&self) -> MatrixDesc<T> {
        MatrixDesc {
            ptr: self.ptr,
            rows: self.rows,
            cols: self.cols,
            ld: self.ld,
            layout: self.layout,
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod accumulator_bits_tests {
    use super::*;

    #[test]
    fn f32_accumulator_bits_equal_plain_bits() {
        // Accumulator == Self, so the two encodings agree bit-for-bit.
        for v in [0.0_f32, 1.0, -2.5, 128.0, f32::MIN_POSITIVE] {
            assert_eq!(v.to_accumulator_bits(), v.to_bits_u64());
        }
    }

    #[test]
    fn f64_accumulator_bits_equal_plain_bits() {
        for v in [0.0_f64, 1.0, -2.5, 128.0] {
            assert_eq!(v.to_accumulator_bits(), v.to_bits_u64());
        }
    }

    #[cfg(feature = "f16")]
    #[test]
    fn f16_accumulator_bits_widen_to_f32() {
        // The whole point of the fix: a half's accumulator bits are the *f32*
        // encoding, NOT the raw 16-bit pattern. Reading the raw pattern as an
        // f32 (the old bug) would have made 1.0 a near-zero denormal.
        let one = half::f16::ONE;
        assert_eq!(one.to_accumulator_bits(), 1.0_f32.to_bits() as u64);
        assert_ne!(one.to_accumulator_bits(), one.to_bits_u64());

        let two_five = half::f16::from_f32(2.5);
        assert_eq!(two_five.to_accumulator_bits(), 2.5_f32.to_bits() as u64);
        assert_eq!(
            half::f16::ZERO.to_accumulator_bits(),
            0.0_f32.to_bits() as u64
        );
    }

    #[cfg(feature = "f16")]
    #[test]
    fn bf16_accumulator_bits_widen_to_f32() {
        let one = half::bf16::ONE;
        assert_eq!(one.to_accumulator_bits(), 1.0_f32.to_bits() as u64);
        assert_ne!(one.to_accumulator_bits(), one.to_bits_u64());
        assert_eq!(
            half::bf16::from_f32(-4.0).to_accumulator_bits(),
            (-4.0_f32).to_bits() as u64
        );
    }

    #[test]
    fn fp8_accumulator_bits_decode_to_f32() {
        // `gpu_one()` must decode to exactly 1.0 in both FP8 formats, and zero
        // to 0.0 -- otherwise an FP8 GEMM would mis-scale by alpha/beta.
        assert_eq!(
            E4M3::gpu_one().to_accumulator_bits(),
            1.0_f32.to_bits() as u64
        );
        assert_eq!(
            E4M3::gpu_zero().to_accumulator_bits(),
            0.0_f32.to_bits() as u64
        );
        assert_eq!(
            E5M2::gpu_one().to_accumulator_bits(),
            1.0_f32.to_bits() as u64
        );
        assert_eq!(
            E5M2::gpu_zero().to_accumulator_bits(),
            0.0_f32.to_bits() as u64
        );

        // A representative non-unit value: E4M3 0x40 = +2.0
        // (exp = 0b1000 = 8, bias 7 -> 2^1; mantissa 0 -> 1.0).
        assert_eq!(E4M3(0x40).to_accumulator_bits(), 2.0_f32.to_bits() as u64);
        // E5M2 0x40 = +2.0 (exp = 0b10000 = 16, bias 15 -> 2^1; mantissa 0).
        assert_eq!(E5M2(0x40).to_accumulator_bits(), 2.0_f32.to_bits() as u64);
    }
}
