//! A backend that refuses every operation.
//!
//! [`NullBackend`] implements [`ComputeBackend`](crate::ComputeBackend) by
//! returning [`BackendError::Unsupported`]
//! from every compute/memory method. It is a **test scaffold**: a consumer
//! crate can compile and exercise its dispatch logic against
//! `Box<dyn ComputeBackend>` without pulling in any real GPU backend, and
//! can assert that the "no backend available" path is handled gracefully.

use crate::ComputeBackend;
use crate::capabilities::{Capabilities, DeviceInfo};
use crate::error::{BackendError, BackendResult};
use crate::ops::{BackendTranspose, BinaryOp, ReduceOp, UnaryOp};

/// A no-op backend whose operations all return
/// [`BackendError::Unsupported`].
#[derive(Debug, Default, Clone, Copy)]
pub struct NullBackend;

impl NullBackend {
    /// Create a new null backend.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Build the standard `Unsupported` error naming the offending op.
    fn deny(op: &str) -> BackendError {
        BackendError::Unsupported(format!("null backend cannot perform `{op}`"))
    }
}

impl ComputeBackend for NullBackend {
    fn name(&self) -> &str {
        "null"
    }

    fn init(&mut self) -> BackendResult<()> {
        // Initialization itself is allowed (so the type is constructible in
        // a pipeline); only actual work is refused.
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        true
    }

    fn capabilities(&self) -> Capabilities {
        // A null backend advertises no capabilities at all.
        Capabilities {
            max_threads_per_block: 0,
            max_shared_mem_per_block: 0,
            warp_size: 0,
            unified_memory: false,
            ..Capabilities::cpu()
        }
    }

    fn available_devices(&self) -> BackendResult<Vec<DeviceInfo>> {
        Ok(Vec::new())
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
        Err(Self::deny("gemm"))
    }

    fn conv2d_forward(
        &self,
        _input_ptr: u64,
        _input_shape: &[usize],
        _filter_ptr: u64,
        _filter_shape: &[usize],
        _output_ptr: u64,
        _output_shape: &[usize],
        _stride: &[usize],
        _padding: &[usize],
    ) -> BackendResult<()> {
        Err(Self::deny("conv2d_forward"))
    }

    fn attention(
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
        Err(Self::deny("attention"))
    }

    fn reduce(
        &self,
        _op: ReduceOp,
        _input_ptr: u64,
        _output_ptr: u64,
        _shape: &[usize],
        _axis: usize,
    ) -> BackendResult<()> {
        Err(Self::deny("reduce"))
    }

    fn unary(
        &self,
        _op: UnaryOp,
        _input_ptr: u64,
        _output_ptr: u64,
        _n: usize,
    ) -> BackendResult<()> {
        Err(Self::deny("unary"))
    }

    fn binary(
        &self,
        _op: BinaryOp,
        _a_ptr: u64,
        _b_ptr: u64,
        _output_ptr: u64,
        _n: usize,
    ) -> BackendResult<()> {
        Err(Self::deny("binary"))
    }

    fn synchronize(&self) -> BackendResult<()> {
        Ok(())
    }

    fn alloc(&self, _bytes: usize) -> BackendResult<u64> {
        Err(Self::deny("alloc"))
    }

    fn free(&self, _ptr: u64) -> BackendResult<()> {
        Err(Self::deny("free"))
    }

    fn copy_htod(&self, _dst: u64, _src: &[u8]) -> BackendResult<()> {
        Err(Self::deny("copy_htod"))
    }

    fn copy_dtoh(&self, _dst: &mut [u8], _src: u64) -> BackendResult<()> {
        Err(Self::deny("copy_dtoh"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_backend_name_and_init() {
        let mut be = NullBackend::new();
        assert_eq!(be.name(), "null");
        assert!(be.init().is_ok());
        assert!(be.is_initialized());
    }

    #[test]
    fn every_compute_op_is_unsupported() {
        let be = NullBackend::new();
        assert!(matches!(
            be.gemm(
                BackendTranspose::NoTrans,
                BackendTranspose::NoTrans,
                1,
                1,
                1,
                1.0,
                0,
                1,
                0,
                1,
                0.0,
                0,
                1
            ),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            be.unary(UnaryOp::Relu, 0, 0, 1),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            be.binary(BinaryOp::Add, 0, 0, 0, 1),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            be.reduce(ReduceOp::Sum, 0, 0, &[1], 0),
            Err(BackendError::Unsupported(_))
        ));
        assert!(matches!(be.alloc(16), Err(BackendError::Unsupported(_))));
        assert!(matches!(
            be.copy_htod(0, &[0u8; 4]),
            Err(BackendError::Unsupported(_))
        ));
    }

    #[test]
    fn synchronize_is_noop_ok() {
        assert!(NullBackend::new().synchronize().is_ok());
    }

    #[test]
    fn no_devices_and_no_capabilities() {
        let be = NullBackend::new();
        assert!(be.available_devices().unwrap().is_empty());
        let caps = be.capabilities();
        assert_eq!(caps.max_threads_per_block, 0);
        assert!(!caps.unified_memory);
    }

    #[test]
    fn error_message_names_the_op() {
        let be = NullBackend::new();
        let err = be.unary(UnaryOp::Exp, 0, 0, 1).unwrap_err();
        assert!(err.to_string().contains("unary"));
    }

    #[test]
    fn usable_as_boxed_trait_object() {
        let be: Box<dyn ComputeBackend> = Box::new(NullBackend::new());
        assert_eq!(be.name(), "null");
        assert!(be.synchronize().is_ok());
    }
}
