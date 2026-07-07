//! # MetalBackend - Trait Implementations
//!
//! This module contains trait implementations for `MetalBackend`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `ComputeBackend`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::Arc;

use oxicuda_backend::{
    BackendError, BackendResult, BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp,
};

use crate::{device::MetalDevice, memory::MetalMemoryManager};

use super::functions::{read_f32_le, write_f32_le};
use super::types::MetalBackend;

impl Default for MetalBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for MetalBackend {
    fn name(&self) -> &str {
        "metal"
    }
    fn init(&mut self) -> BackendResult<()> {
        if self.initialized {
            return Ok(());
        }
        match MetalDevice::new() {
            Ok(dev) => {
                let dev = Arc::new(dev);
                tracing::info!("Metal backend initialised on: {}", dev.name());
                let memory = MetalMemoryManager::new(Arc::clone(&dev));
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
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        super::types::validate_gemm_layout(trans_a, trans_b, n, k, lda, ldb, ldc)?;
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
        let n = input_shape[0];
        let c_in = input_shape[1];
        let h_in = input_shape[2];
        let w_in = input_shape[3];
        let k_out = filter_shape[0];
        let fh = filter_shape[2];
        let fw = filter_shape[3];
        let oh = output_shape[2];
        let ow = output_shape[3];
        let stride_h = stride[0];
        let stride_w = stride[1];
        let pad_h = padding[0];
        let pad_w = padding[1];
        let input_len = n * c_in * h_in * w_in;
        let filter_len = k_out * c_in * fh * fw;
        let output_len = n * k_out * oh * ow;
        let mut input_bytes = vec![0u8; input_len * 4];
        let mut filter_bytes = vec![0u8; filter_len * 4];
        self.copy_dtoh(&mut input_bytes, input_ptr)?;
        self.copy_dtoh(&mut filter_bytes, filter_ptr)?;
        let inp = read_f32_le(&input_bytes);
        let flt = read_f32_le(&filter_bytes);
        let mut out = vec![0.0f32; output_len];
        for b in 0..n {
            for kf in 0..k_out {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut acc = 0.0f32;
                        for ci in 0..c_in {
                            for fy in 0..fh {
                                for fx in 0..fw {
                                    let iy = (oy * stride_h + fy) as isize - pad_h as isize;
                                    let ix = (ox * stride_w + fx) as isize - pad_w as isize;
                                    if iy >= 0
                                        && (iy as usize) < h_in
                                        && ix >= 0
                                        && (ix as usize) < w_in
                                    {
                                        let iy = iy as usize;
                                        let ix = ix as usize;
                                        acc += inp[((b * c_in + ci) * h_in + iy) * w_in + ix]
                                            * flt[((kf * c_in + ci) * fh + fy) * fw + fx];
                                    }
                                }
                            }
                        }
                        out[((b * k_out + kf) * oh + oy) * ow + ox] = acc;
                    }
                }
            }
        }
        let out_bytes = write_f32_le(&out);
        self.copy_htod(output_ptr, &out_bytes)?;
        Ok(())
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
        let batch_heads = batch * heads;
        let q_len = batch_heads * seq_q * head_dim;
        let kv_len = batch_heads * seq_kv * head_dim;
        let o_len = batch_heads * seq_q * head_dim;
        let mut q_bytes = vec![0u8; q_len * 4];
        let mut k_bytes = vec![0u8; kv_len * 4];
        let mut v_bytes = vec![0u8; kv_len * 4];
        self.copy_dtoh(&mut q_bytes, q_ptr)?;
        self.copy_dtoh(&mut k_bytes, k_ptr)?;
        self.copy_dtoh(&mut v_bytes, v_ptr)?;
        let q = read_f32_le(&q_bytes);
        let k = read_f32_le(&k_bytes);
        let v = read_f32_le(&v_bytes);
        let mut o = vec![0.0f32; o_len];
        let scale_f = scale as f32;
        // Reusable per-(bh,sq) score buffer: the scaled Q·Kᵀ dot products are
        // computed once in the max pass and reused in the accumulate pass instead
        // of recomputing the O(head_dim) inner product a second time.
        let mut scores = vec![0.0f32; seq_kv];
        for bh in 0..batch_heads {
            for sq in 0..seq_q {
                let q_off = (bh * seq_q + sq) * head_dim;
                let mut max_score = f32::NEG_INFINITY;
                for (sk, score_slot) in scores.iter_mut().enumerate() {
                    if causal && sk > sq {
                        continue;
                    }
                    let k_off = (bh * seq_kv + sk) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_off + d] * k[k_off + d];
                    }
                    let score = dot * scale_f;
                    *score_slot = score;
                    if score > max_score {
                        max_score = score;
                    }
                }
                let mut sum_exp = 0.0f32;
                let mut acc = vec![0.0f32; head_dim];
                for (sk, &score) in scores.iter().enumerate() {
                    if causal && sk > sq {
                        continue;
                    }
                    // Reuse the scaled score cached in the max pass above.
                    let w = (score - max_score).exp();
                    sum_exp += w;
                    let v_off = (bh * seq_kv + sk) * head_dim;
                    for d in 0..head_dim {
                        acc[d] += w * v[v_off + d];
                    }
                }
                let o_off = (bh * seq_q + sq) * head_dim;
                if sum_exp > 0.0 {
                    for d in 0..head_dim {
                        o[o_off + d] = acc[d] / sum_exp;
                    }
                }
            }
        }
        let o_bytes = write_f32_le(&o);
        self.copy_htod(o_ptr, &o_bytes)?;
        Ok(())
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
        super::types::validate_gemm_layout(trans_a, trans_b, n, k, lda, ldb, ldc)?;
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
    fn synchronize(&self) -> BackendResult<()> {
        self.check_init()?;
        Ok(())
    }
    fn alloc(&self, bytes: usize) -> BackendResult<u64> {
        self.check_init()?;
        if bytes == 0 {
            return Err(BackendError::InvalidArgument(
                "cannot allocate 0 bytes".into(),
            ));
        }
        self.memory()?.alloc(bytes).map_err(BackendError::from)
    }
    fn free(&self, ptr: u64) -> BackendResult<()> {
        self.check_init()?;
        self.memory()?.free(ptr).map_err(BackendError::from)
    }
    fn copy_htod(&self, dst: u64, src: &[u8]) -> BackendResult<()> {
        self.check_init()?;
        if src.is_empty() {
            return Ok(());
        }
        self.memory()?
            .copy_to_device(dst, src)
            .map_err(BackendError::from)
    }
    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> BackendResult<()> {
        self.check_init()?;
        if dst.is_empty() {
            return Ok(());
        }
        self.memory()?
            .copy_from_device(dst, src)
            .map_err(BackendError::from)
    }
}
