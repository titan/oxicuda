//! [`WebGpuBackend`] — the main entry point for the oxicuda-webgpu crate.
//!
//! Implements the [`ComputeBackend`] trait from `oxicuda-backend` using
//! `wgpu` for cross-platform GPU compute (Vulkan, Metal, DX12, WebGPU).

use std::sync::Arc;

use oxicuda_backend::{
    BackendError, BackendResult, BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp,
};
use wgpu;

use crate::{device::WebGpuDevice, memory::WebGpuMemoryManager, shader};

// ─── Op-mapping helpers ──────────────────────────────────────────────────────

fn map_unary_op(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Relu => "relu",
        UnaryOp::Sigmoid => "sigmoid",
        UnaryOp::Tanh => "tanh",
        UnaryOp::Exp => "exp",
        UnaryOp::Log => "log",
        UnaryOp::Sqrt => "sqrt",
        UnaryOp::Abs => "abs",
        UnaryOp::Neg => "neg",
    }
}

fn map_binary_op(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Mul => "mul",
        BinaryOp::Div => "div",
        BinaryOp::Max => "max",
        BinaryOp::Min => "min",
    }
}

fn map_reduce_op(op: ReduceOp) -> &'static str {
    match op {
        ReduceOp::Sum => "sum",
        ReduceOp::Max => "max",
        ReduceOp::Min => "min",
        ReduceOp::Mean => "mean",
    }
}

// ─── Backend struct ──────────────────────────────────────────────────────────

/// Cross-platform GPU compute backend backed by `wgpu`.
///
/// # Lifecycle
///
/// 1. `WebGpuBackend::new()` — create an uninitialised backend.
/// 2. `init()` — select the best available adapter and create the device.
/// 3. Use `alloc`, `copy_htod`, compute ops, `copy_dtoh`, `free`.
/// 4. `synchronize()` — wait for all pending GPU work to finish.
#[derive(Debug)]
pub struct WebGpuBackend {
    device: Option<Arc<WebGpuDevice>>,
    memory: Option<Arc<WebGpuMemoryManager>>,
    initialized: bool,
}

impl WebGpuBackend {
    /// Create a new, uninitialised WebGPU backend.
    pub fn new() -> Self {
        Self {
            device: None,
            memory: None,
            initialized: false,
        }
    }

    /// Return an error if the backend is not yet initialised.
    fn check_init(&self) -> BackendResult<()> {
        if self.initialized {
            Ok(())
        } else {
            Err(BackendError::NotInitialized)
        }
    }

    /// Convenience accessor: get the memory manager or return `NotInitialized`.
    fn memory(&self) -> BackendResult<&Arc<WebGpuMemoryManager>> {
        self.memory.as_ref().ok_or(BackendError::NotInitialized)
    }

    /// Convenience accessor: get the device or return `NotInitialized`.
    fn device(&self) -> BackendResult<&Arc<WebGpuDevice>> {
        self.device.as_ref().ok_or(BackendError::NotInitialized)
    }

    /// Multi-dimensional reduce along a single axis.
    ///
    /// The tensor is logically reshaped to `[outer, dk, inner]`:
    /// * `outer` = product of dimensions before the reduce axis,
    /// * `dk`    = the reduce axis length,
    /// * `inner` = product of dimensions after the reduce axis.
    ///
    /// One workgroup of 256 threads is dispatched per `(o, j)` output slot.
    /// To stay within WebGPU's 65 535-per-axis dispatch limit a 2-D grid is
    /// used and the workgroup decodes its linear slot internally.
    ///
    /// `Mean` is handled inside the shader (divide by `dk`); the host does
    /// not need a post-pass.
    fn reduce_nd(
        &self,
        op: ReduceOp,
        input_ptr: u64,
        output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()> {
        // Caller (`reduce`) already validated `shape.is_empty()` and
        // `axis < shape.len()`; assert in debug to catch regressions but
        // recompute defensively in release as well.
        debug_assert!(!shape.is_empty());
        debug_assert!(axis < shape.len());

        // Output shape = shape with `axis` removed; length = outer * inner.
        let outer: usize = shape[..axis].iter().product();
        let dk: usize = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();

        // Empty tensor — nothing to do.
        if outer == 0 || dk == 0 || inner == 0 {
            return Ok(());
        }

        let total = outer.checked_mul(inner).ok_or_else(|| {
            BackendError::InvalidArgument("reduce: outer * inner overflows usize".into())
        })?;

        // Strides in elements: row-major (C order) layout.
        let inner_stride: usize = 1;
        let dk_stride: usize = inner;
        let outer_stride: usize = dk
            .checked_mul(inner)
            .ok_or_else(|| BackendError::InvalidArgument("reduce: dk * inner overflows".into()))?;

        // Cap each dispatch dimension below the WebGPU 65 535 limit.  We pick
        // grid_x = min(total, 32 768) so grid_y stays modest for huge tensors.
        const MAX_GRID_DIM: u32 = 32_768;
        let total_u32: u32 = total.try_into().map_err(|_| {
            BackendError::InvalidArgument(format!(
                "reduce: output element count {total} exceeds u32 range"
            ))
        })?;
        let grid_x: u32 = total_u32.clamp(1, MAX_GRID_DIM);
        let grid_y: u32 = total_u32.div_ceil(grid_x);

        let dev = self.device()?;
        let mem = self.memory()?;
        let op_str = map_reduce_op(op);

        let wgsl = shader::reduction_nd_wgsl(op_str);
        let shader_mod = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("oxicuda-reduce-nd"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("oxicuda-reduce-nd"),
                layout: None,
                module: &shader_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Build the uniform buffer: 8 × u32 = 32 bytes (16-byte aligned).
        let mut params_bytes = [0u8; 32];
        let outer_u32: u32 = outer
            .try_into()
            .map_err(|_| BackendError::InvalidArgument("reduce: outer exceeds u32 range".into()))?;
        let dk_u32: u32 = dk
            .try_into()
            .map_err(|_| BackendError::InvalidArgument("reduce: dk exceeds u32 range".into()))?;
        let inner_u32: u32 = inner
            .try_into()
            .map_err(|_| BackendError::InvalidArgument("reduce: inner exceeds u32 range".into()))?;
        let outer_stride_u32: u32 = outer_stride.try_into().map_err(|_| {
            BackendError::InvalidArgument("reduce: outer_stride exceeds u32 range".into())
        })?;
        let dk_stride_u32: u32 = dk_stride.try_into().map_err(|_| {
            BackendError::InvalidArgument("reduce: dk_stride exceeds u32 range".into())
        })?;
        let inner_stride_u32: u32 = inner_stride.try_into().map_err(|_| {
            BackendError::InvalidArgument("reduce: inner_stride exceeds u32 range".into())
        })?;
        params_bytes[0..4].copy_from_slice(&outer_u32.to_le_bytes());
        params_bytes[4..8].copy_from_slice(&dk_u32.to_le_bytes());
        params_bytes[8..12].copy_from_slice(&inner_u32.to_le_bytes());
        params_bytes[12..16].copy_from_slice(&outer_stride_u32.to_le_bytes());
        params_bytes[16..20].copy_from_slice(&dk_stride_u32.to_le_bytes());
        params_bytes[20..24].copy_from_slice(&inner_stride_u32.to_le_bytes());
        params_bytes[24..28].copy_from_slice(&grid_x.to_le_bytes());
        // bytes 28..32 are zero padding.

        let uniform_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-reduce-nd-params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&uniform_buf, 0, &params_bytes);

        let bgl = pipeline.get_bind_group_layout(0);
        let bind_group = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let in_info = buffers.get(&input_ptr).ok_or_else(|| {
                BackendError::InvalidArgument(format!("unknown handle {input_ptr}"))
            })?;
            let out_info = buffers.get(&output_ptr).ok_or_else(|| {
                BackendError::InvalidArgument(format!("unknown handle {output_ptr}"))
            })?;

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-reduce-nd"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: in_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buf.as_entire_binding(),
                    },
                ],
            })
        };

        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("oxicuda-reduce-nd"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-reduce-nd"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(grid_x, grid_y, 1);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());

        Ok(())
    }
}

impl WebGpuBackend {
    /// FP16 GEMM: `C = alpha * A * B + beta * C` with half-precision storage.
    ///
    /// This is an inherent method (not on `ComputeBackend`) because FP16
    /// support is WebGPU-specific and requires the `f16` WGSL extension.
    ///
    /// Buffers pointed to by `a_ptr`, `b_ptr`, `c_ptr` must contain `f16`
    /// elements (2 bytes each).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        a_ptr: u64,
        b_ptr: u64,
        beta: f64,
        c_ptr: u64,
    ) -> BackendResult<()> {
        self.check_init()?;
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }

        let dev = self.device()?;
        let mem = self.memory()?;

        let tile_size: u32 = 8;
        let wgsl = shader::gemm_wgsl_f16(tile_size);

        let shader_mod = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("oxicuda-gemm-f16"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });

        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("oxicuda-gemm-f16"),
                layout: None,
                module: &shader_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bgl = pipeline.get_bind_group_layout(0);

        // Build uniform buffer for GemmParams { m, n, k, alpha, beta }.
        let mut params_bytes = [0u8; 20];
        params_bytes[0..4].copy_from_slice(&(m as u32).to_le_bytes());
        params_bytes[4..8].copy_from_slice(&(n as u32).to_le_bytes());
        params_bytes[8..12].copy_from_slice(&(k as u32).to_le_bytes());
        params_bytes[12..16].copy_from_slice(&(alpha as f32).to_le_bytes());
        params_bytes[16..20].copy_from_slice(&(beta as f32).to_le_bytes());

        let uniform_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-gemm-f16-params"),
            size: 20,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&uniform_buf, 0, &params_bytes);

        let bind_group = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let a_info = buffers
                .get(&a_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {a_ptr}")))?;
            let b_info = buffers
                .get(&b_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {b_ptr}")))?;
            let c_info = buffers
                .get(&c_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {c_ptr}")))?;

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-gemm-f16"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: c_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buf.as_entire_binding(),
                    },
                ],
            })
        };

        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("oxicuda-gemm-f16"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-gemm-f16"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (n as u32).div_ceil(tile_size);
            let wg_y = (m as u32).div_ceil(tile_size);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());

        Ok(())
    }
}

impl Default for WebGpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ─── ComputeBackend impl ─────────────────────────────────────────────────────

impl ComputeBackend for WebGpuBackend {
    fn name(&self) -> &str {
        "webgpu"
    }

    fn init(&mut self) -> BackendResult<()> {
        if self.initialized {
            return Ok(());
        }

        match WebGpuDevice::new() {
            Ok(dev) => {
                let dev = Arc::new(dev);
                tracing::info!("WebGPU backend initialised on: {}", dev.adapter_name);
                let memory = WebGpuMemoryManager::new(Arc::clone(&dev));
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
        _lda: usize,
        b_ptr: u64,
        _ldb: usize,
        beta: f64,
        c_ptr: u64,
        _ldc: usize,
    ) -> BackendResult<()> {
        self.check_init()?;
        // Zero-dimension matrices are trivially done.
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }

        // Transpose not yet supported in the WGSL shader.
        if trans_a != BackendTranspose::NoTrans || trans_b != BackendTranspose::NoTrans {
            return Err(BackendError::Unsupported(
                "WebGPU GEMM does not yet support transposed inputs".into(),
            ));
        }

        let dev = self.device()?;
        let mem = self.memory()?;

        let tile_size: u32 = 8;
        let wgsl = shader::gemm_wgsl(tile_size);

        let shader_mod = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("oxicuda-gemm"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });

        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("oxicuda-gemm"),
                layout: None,
                module: &shader_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bgl = pipeline.get_bind_group_layout(0);

        // Build uniform buffer for GemmParams { m, n, k, alpha, beta }.
        let mut params_bytes = [0u8; 20];
        params_bytes[0..4].copy_from_slice(&(m as u32).to_le_bytes());
        params_bytes[4..8].copy_from_slice(&(n as u32).to_le_bytes());
        params_bytes[8..12].copy_from_slice(&(k as u32).to_le_bytes());
        params_bytes[12..16].copy_from_slice(&(alpha as f32).to_le_bytes());
        params_bytes[16..20].copy_from_slice(&(beta as f32).to_le_bytes());

        let uniform_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-gemm-params"),
            size: 20,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&uniform_buf, 0, &params_bytes);

        // Create bind group while holding the buffer lock.
        let bind_group = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let a_info = buffers
                .get(&a_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {a_ptr}")))?;
            let b_info = buffers
                .get(&b_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {b_ptr}")))?;
            let c_info = buffers
                .get(&c_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {c_ptr}")))?;

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-gemm"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: c_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buf.as_entire_binding(),
                    },
                ],
            })
        };

        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("oxicuda-gemm"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-gemm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (n as u32).div_ceil(tile_size);
            let wg_y = (m as u32).div_ceil(tile_size);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());

        Ok(())
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
        self.check_init()?;

        if batch_count == 0 || m == 0 || n == 0 || k == 0 {
            return Ok(());
        }

        if trans_a != BackendTranspose::NoTrans || trans_b != BackendTranspose::NoTrans {
            return Err(BackendError::Unsupported(
                "WebGPU batched GEMM does not yet support transposed inputs".into(),
            ));
        }

        let dev = self.device()?;
        let mem = self.memory()?;

        let tile_size: u32 = 8;
        let wgsl = shader::batched_gemm_wgsl(tile_size);

        let shader_mod = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("oxicuda-batched-gemm"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });

        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("oxicuda-batched-gemm"),
                layout: None,
                module: &shader_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bgl = pipeline.get_bind_group_layout(0);

        // BatchedGemmParams: m, n, k, alpha, beta, batch_count, stride_a, stride_b, stride_c
        // 9 fields: 5 x u32/f32 + 4 x u32 = 36 bytes total
        // But we need 16-byte alignment for uniform buffers. 36 rounds up to 48.
        // Actually: 3 u32 + 2 f32 + 1 u32 + 3 u32 = 9 x 4 = 36 bytes.
        // Pad to 48 for safety (16-byte aligned).
        let mut params_bytes = [0u8; 48];
        params_bytes[0..4].copy_from_slice(&(m as u32).to_le_bytes());
        params_bytes[4..8].copy_from_slice(&(n as u32).to_le_bytes());
        params_bytes[8..12].copy_from_slice(&(k as u32).to_le_bytes());
        params_bytes[12..16].copy_from_slice(&(alpha as f32).to_le_bytes());
        params_bytes[16..20].copy_from_slice(&(beta as f32).to_le_bytes());
        params_bytes[20..24].copy_from_slice(&(batch_count as u32).to_le_bytes());
        params_bytes[24..28].copy_from_slice(&(stride_a as u32).to_le_bytes());
        params_bytes[28..32].copy_from_slice(&(stride_b as u32).to_le_bytes());
        params_bytes[32..36].copy_from_slice(&(stride_c as u32).to_le_bytes());
        // bytes 36..48 are padding zeros

        let uniform_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-batched-gemm-params"),
            size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&uniform_buf, 0, &params_bytes);

        let bind_group = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let a_info = buffers
                .get(&a_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {a_ptr}")))?;
            let b_info = buffers
                .get(&b_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {b_ptr}")))?;
            let c_info = buffers
                .get(&c_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {c_ptr}")))?;

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-batched-gemm"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: c_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buf.as_entire_binding(),
                    },
                ],
            })
        };

        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("oxicuda-batched-gemm"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-batched-gemm"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (n as u32).div_ceil(tile_size);
            let wg_y = (m as u32).div_ceil(tile_size);
            pass.dispatch_workgroups(wg_x, wg_y, batch_count as u32);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());

        Ok(())
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

        let mem = self.memory()?;

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

        // CPU fallback: download input + filter, compute, upload output.
        let mut in_bytes = vec![0u8; in_elems * 4];
        let mut f_bytes = vec![0u8; f_elems * 4];
        mem.copy_from_device(&mut in_bytes, input_ptr)
            .map_err(BackendError::from)?;
        mem.copy_from_device(&mut f_bytes, filter_ptr)
            .map_err(BackendError::from)?;

        let in_f32 = bytes_to_f32_vec(&in_bytes);
        let f_f32 = bytes_to_f32_vec(&f_bytes);
        let mut out_f32 = vec![0.0f32; o_elems];

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
                                        let in_idx = ((b * c_in + ci) * h_in + iy as usize) * w_in
                                            + ix as usize;
                                        let f_idx = ((kf * c_in + ci) * fh + fy) * fw + fx;
                                        acc += in_f32[in_idx] * f_f32[f_idx];
                                    }
                                }
                            }
                        }
                        out_f32[((b * k_out + kf) * oh + oy) * ow + ox] = acc;
                    }
                }
            }
        }

        let out_bytes = f32_slice_to_bytes(&out_f32);
        mem.copy_to_device(output_ptr, &out_bytes)
            .map_err(BackendError::from)?;

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

        let mem = self.memory()?;

        let batch_heads = batch * heads;
        let q_elems = batch_heads * seq_q * head_dim;
        let kv_elems = batch_heads * seq_kv * head_dim;
        let o_elems = q_elems;

        // CPU fallback: download Q, K, V, compute attention, upload O.
        let mut q_bytes = vec![0u8; q_elems * 4];
        let mut k_bytes = vec![0u8; kv_elems * 4];
        let mut v_bytes = vec![0u8; kv_elems * 4];

        mem.copy_from_device(&mut q_bytes, q_ptr)
            .map_err(BackendError::from)?;
        mem.copy_from_device(&mut k_bytes, k_ptr)
            .map_err(BackendError::from)?;
        mem.copy_from_device(&mut v_bytes, v_ptr)
            .map_err(BackendError::from)?;

        let q_f32 = bytes_to_f32_vec(&q_bytes);
        let k_f32 = bytes_to_f32_vec(&k_bytes);
        let v_f32 = bytes_to_f32_vec(&v_bytes);
        let mut o_f32 = vec![0.0f32; o_elems];

        let scale_f32 = scale as f32;

        for bh in 0..batch_heads {
            let q_off = bh * seq_q * head_dim;
            let k_off = bh * seq_kv * head_dim;
            let v_off = k_off;

            for sq in 0..seq_q {
                let kv_limit = if causal { (sq + 1).min(seq_kv) } else { seq_kv };

                // Pass 1: find max score for numerical stability
                let mut max_score = f32::NEG_INFINITY;
                for sk in 0..kv_limit {
                    let mut dot = 0.0f32;
                    for dd in 0..head_dim {
                        dot +=
                            q_f32[q_off + sq * head_dim + dd] * k_f32[k_off + sk * head_dim + dd];
                    }
                    let s = dot * scale_f32;
                    if s > max_score {
                        max_score = s;
                    }
                }

                // Pass 2: exp(score - max), accumulate weighted V
                let mut sum_exp = 0.0f32;
                let mut acc = vec![0.0f32; head_dim];
                for sk in 0..kv_limit {
                    let mut dot = 0.0f32;
                    for dd in 0..head_dim {
                        dot +=
                            q_f32[q_off + sq * head_dim + dd] * k_f32[k_off + sk * head_dim + dd];
                    }
                    let w = (dot * scale_f32 - max_score).exp();
                    sum_exp += w;
                    for dd in 0..head_dim {
                        acc[dd] += w * v_f32[v_off + sk * head_dim + dd];
                    }
                }

                // Normalise
                let o_base = q_off + sq * head_dim;
                if sum_exp > 0.0 {
                    for dd in 0..head_dim {
                        o_f32[o_base + dd] = acc[dd] / sum_exp;
                    }
                }
            }
        }

        let o_bytes = f32_slice_to_bytes(&o_f32);
        mem.copy_to_device(o_ptr, &o_bytes)
            .map_err(BackendError::from)?;

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

        // 1-D shapes (or any shape that reduces to a single scalar) take the
        // optimised two-pass scalar path.  Higher-rank shapes go through the
        // batched N-D shader below.
        if shape.len() != 1 {
            return self.reduce_nd(op, input_ptr, output_ptr, shape, axis);
        }

        let n_elements = shape[0];
        if n_elements == 0 {
            return Ok(());
        }

        let dev = self.device()?;
        let mem = self.memory()?;
        let op_str = map_reduce_op(op);

        // ── Pass 1: per-workgroup reduction ─────────────────────────────────
        let wg_count = (n_elements as u32).div_ceil(256);

        let pass1_wgsl = shader::reduction_wgsl(op_str);
        let pass1_shader = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("oxicuda-reduce-pass1"),
                source: wgpu::ShaderSource::Wgsl(pass1_wgsl.into()),
            });
        let pass1_pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("oxicuda-reduce-pass1"),
                layout: None,
                module: &pass1_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Partial-sums buffer (temporary).
        let partial_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-reduce-partial"),
            size: (wg_count as u64) * 4, // f32 per workgroup
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Uniform for ReduceParams { n: u32 }.
        let mut p1_params = [0u8; 4];
        p1_params[0..4].copy_from_slice(&(n_elements as u32).to_le_bytes());
        let p1_uniform = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-reduce-p1-params"),
            size: 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&p1_uniform, 0, &p1_params);

        let bgl1 = pass1_pipeline.get_bind_group_layout(0);

        let bg1 = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let in_info = buffers.get(&input_ptr).ok_or_else(|| {
                BackendError::InvalidArgument(format!("unknown handle {input_ptr}"))
            })?;

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-reduce-pass1"),
                layout: &bgl1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: in_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: partial_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: p1_uniform.as_entire_binding(),
                    },
                ],
            })
        };

        // ── Pass 2: final reduction of partial sums ─────────────────────────
        let pass2_wgsl = shader::reduction_final_wgsl(op_str);
        let pass2_shader = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("oxicuda-reduce-pass2"),
                source: wgpu::ShaderSource::Wgsl(pass2_wgsl.into()),
            });
        let pass2_pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("oxicuda-reduce-pass2"),
                layout: None,
                module: &pass2_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // FinalReduceParams { num_groups: u32 }.
        let mut p2_params = [0u8; 4];
        p2_params[0..4].copy_from_slice(&wg_count.to_le_bytes());
        let p2_uniform = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oxicuda-reduce-p2-params"),
            size: 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&p2_uniform, 0, &p2_params);

        let bgl2 = pass2_pipeline.get_bind_group_layout(0);

        let bg2 = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let out_info = buffers.get(&output_ptr).ok_or_else(|| {
                BackendError::InvalidArgument(format!("unknown handle {output_ptr}"))
            })?;

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-reduce-pass2"),
                layout: &bgl2,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: partial_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: p2_uniform.as_entire_binding(),
                    },
                ],
            })
        };

        // ── Encode both passes into one command buffer ──────────────────────
        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("oxicuda-reduce"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-reduce-pass1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pass1_pipeline);
            pass.set_bind_group(0, &bg1, &[]);
            pass.dispatch_workgroups(wg_count, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-reduce-pass2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pass2_pipeline);
            pass.set_bind_group(0, &bg2, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());

        // For "mean", divide the result by N on the host side.
        if op == ReduceOp::Mean && n_elements > 1 {
            let mut buf = [0u8; 4];
            mem.copy_from_device(&mut buf, output_ptr)
                .map_err(BackendError::from)?;
            let val = f32::from_le_bytes(buf);
            let mean = val / (n_elements as f32);
            mem.copy_to_device(output_ptr, &mean.to_le_bytes())
                .map_err(BackendError::from)?;
        }

        Ok(())
    }

    fn unary(&self, op: UnaryOp, input_ptr: u64, output_ptr: u64, n: usize) -> BackendResult<()> {
        self.check_init()?;
        if n == 0 {
            return Ok(());
        }

        let dev = self.device()?;
        let mem = self.memory()?;

        let op_str = map_unary_op(op);
        let wgsl = shader::elementwise_wgsl(op_str);

        let shader_mod = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("oxicuda-unary"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });

        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("oxicuda-unary"),
                layout: None,
                module: &shader_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bgl = pipeline.get_bind_group_layout(0);

        let bind_group = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let in_info = buffers.get(&input_ptr).ok_or_else(|| {
                BackendError::InvalidArgument(format!("unknown handle {input_ptr}"))
            })?;
            let out_info = buffers.get(&output_ptr).ok_or_else(|| {
                BackendError::InvalidArgument(format!("unknown handle {output_ptr}"))
            })?;

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-unary"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: in_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_info.buffer.as_entire_binding(),
                    },
                ],
            })
        };

        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("oxicuda-unary"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-unary"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n as u32).div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());

        Ok(())
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

        let dev = self.device()?;
        let mem = self.memory()?;

        let op_str = map_binary_op(op);
        let wgsl = shader::binary_wgsl(op_str);

        let shader_mod = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("oxicuda-binary"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });

        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("oxicuda-binary"),
                layout: None,
                module: &shader_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bgl = pipeline.get_bind_group_layout(0);

        let bind_group = {
            let buffers = mem
                .lock_buffers()
                .map_err(|e| BackendError::DeviceError(e.to_string()))?;
            let a_info = buffers
                .get(&a_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {a_ptr}")))?;
            let b_info = buffers
                .get(&b_ptr)
                .ok_or_else(|| BackendError::InvalidArgument(format!("unknown handle {b_ptr}")))?;
            let out_info = buffers.get(&output_ptr).ok_or_else(|| {
                BackendError::InvalidArgument(format!("unknown handle {output_ptr}"))
            })?;

            dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oxicuda-binary"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_info.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_info.buffer.as_entire_binding(),
                    },
                ],
            })
        };

        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("oxicuda-binary"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("oxicuda-binary"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n as u32).div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        dev.queue.submit(std::iter::once(encoder.finish()));
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());

        Ok(())
    }

    // ── Synchronisation ───────────────────────────────────────────────────────

    fn synchronize(&self) -> BackendResult<()> {
        self.check_init()?;
        if let Some(dev) = &self.device {
            let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
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

// ─── Byte ↔ f32 helpers ──────────────────────────────────────────────────────

/// Convert a `&[u8]` (length must be a multiple of 4) to a `Vec<f32>`.
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Convert a `&[f32]` slice to its little-endian byte representation.
fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────
//
// The test module lives in a sibling file (`backend_tests.rs`) so the
// production code in this file stays under the 2 000-line refactoring policy.
#[cfg(test)]
#[path = "backend_tests.rs"]
mod tests;
