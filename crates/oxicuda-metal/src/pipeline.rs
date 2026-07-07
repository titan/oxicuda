//! Metal compute pipeline wrapper.
//!
//! A [`MetalComputePipeline`] compiles MSL source into a
//! `metal::ComputePipelineState` and owns the associated `metal::CommandQueue`.
//! On non-macOS platforms every constructor returns
//! [`MetalError::UnsupportedPlatform`].

use crate::{
    device::MetalDevice,
    error::{MetalError, MetalResult},
    memory::MetalMemoryManager,
};

// ─── MetalComputePipeline ─────────────────────────────────────────────────────

/// A compiled Metal compute pipeline together with its command queue.
///
/// Created by compiling an MSL source string through
/// [`MetalComputePipeline::new`].  The pipeline state and command queue are
/// kept together so that callers can dispatch work without needing to manage
/// them separately.
pub struct MetalComputePipeline {
    /// The compiled pipeline state — only present on macOS.
    /// Used by [`MetalComputePipeline::dispatch`].
    #[cfg(target_os = "macos")]
    pub(crate) pipeline_state: metal::ComputePipelineState,
    /// The command queue used to create command buffers — only present on macOS.
    /// Used by [`MetalComputePipeline::dispatch`].
    #[cfg(target_os = "macos")]
    pub(crate) command_queue: metal::CommandQueue,
    /// The MSL entry-point function name (kept for diagnostics).
    function_name: String,
}

impl MetalComputePipeline {
    /// Compile `msl_source` and look up `function_name` inside the resulting
    /// library, then create a compute pipeline state.
    ///
    /// Returns:
    /// * [`MetalError::ShaderCompilation`] if the MSL fails to compile.
    /// * [`MetalError::PipelineCreation`] if the PSO cannot be created.
    /// * [`MetalError::UnsupportedPlatform`] on non-macOS.
    pub fn new(device: &MetalDevice, msl_source: &str, function_name: &str) -> MetalResult<Self> {
        #[cfg(target_os = "macos")]
        {
            let opts = metal::CompileOptions::new();
            let library = device
                .device
                .new_library_with_source(msl_source, &opts)
                .map_err(|e| MetalError::ShaderCompilation(e.to_string()))?;

            let function = library
                .get_function(function_name, None)
                .map_err(|e| MetalError::ShaderCompilation(e.to_string()))?;

            let pipeline_state = device
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| MetalError::PipelineCreation(e.to_string()))?;

            let command_queue = device.device.new_command_queue();

            Ok(Self {
                pipeline_state,
                command_queue,
                function_name: function_name.to_string(),
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (device, msl_source, function_name);
            Err(MetalError::UnsupportedPlatform)
        }
    }

    /// The MSL function name this pipeline was compiled for.
    pub fn function_name(&self) -> &str {
        &self.function_name
    }

    /// Dispatch this compiled pipeline over `total_threads` GPU threads (1-D).
    ///
    /// Binding layout:
    /// * each handle in `handles` is resolved to its `metal::Buffer` through
    ///   `memory` and bound to `buffer(0)`, `buffer(1)`, … in order;
    /// * each blob in `scalar_bytes` is bound with `set_bytes` to the buffer
    ///   index immediately following the buffers — `buffer(handles.len())`,
    ///   `buffer(handles.len() + 1)`, … — so a kernel that declares `K` device
    ///   buffers followed by `S` `constant` scalars maps one-to-one.
    ///
    /// The threadgroup width is `min(max_total_threads_per_threadgroup, total_threads)`
    /// and the grid is rounded up to whole threadgroups, so the kernel **must**
    /// bounds-check its `thread_position_in_grid` against the element count.
    ///
    /// The call is synchronous: it commits the command buffer and waits for GPU
    /// completion before returning, matching the crate's other compute ops.
    ///
    /// Returns [`MetalError::UnsupportedPlatform`] on non-macOS, and
    /// [`MetalError::InvalidArgument`] for an unknown buffer handle or an empty
    /// `scalar_bytes` entry.
    pub fn dispatch(
        &self,
        memory: &MetalMemoryManager,
        handles: &[u64],
        scalar_bytes: &[&[u8]],
        total_threads: usize,
    ) -> MetalResult<()> {
        #[cfg(target_os = "macos")]
        {
            if total_threads == 0 {
                return Ok(());
            }
            for blob in scalar_bytes {
                if blob.is_empty() {
                    return Err(MetalError::InvalidArgument(
                        "scalar_bytes entries must be non-empty".into(),
                    ));
                }
            }
            // Resolve handles to independent buffer retains under the lock, then
            // release the lock *before* encoding + the blocking GPU wait. The
            // encoder (and command buffer) retain their bound resources until
            // completion, so a concurrent `free()` cannot invalidate them; this
            // keeps unrelated `alloc`/`free`/`copy_*` calls off the critical path
            // during the (potentially long) kernel run.
            let bound: Vec<metal::Buffer> = {
                let buffers = memory.lock_buffers()?;
                let mut v = Vec::with_capacity(handles.len());
                for handle in handles {
                    let info = buffers.get(handle).ok_or_else(|| {
                        MetalError::InvalidArgument(format!("unknown buffer handle {handle}"))
                    })?;
                    v.push(info.buffer.to_owned());
                }
                v
            };
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline_state);
            for (slot, buffer) in bound.iter().enumerate() {
                encoder.set_buffer(slot as u64, Some(buffer), 0);
            }
            let scalar_base = handles.len() as u64;
            for (offset, blob) in scalar_bytes.iter().enumerate() {
                encoder.set_bytes(
                    scalar_base + offset as u64,
                    blob.len() as u64,
                    blob.as_ptr() as *const std::ffi::c_void,
                );
            }
            let max_tg = self.pipeline_state.max_total_threads_per_threadgroup();
            let tg = max_tg.min(total_threads as u64).max(1);
            let groups = (total_threads as u64).div_ceil(tg);
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(groups, 1, 1),
                metal::MTLSize::new(tg, 1, 1),
            );
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            // Surface GPU-side failures (device lost, timeout/TDR, …) instead of
            // returning Ok on a command buffer that finished in an error state.
            match command_buffer.status() {
                metal::MTLCommandBufferStatus::Completed => Ok(()),
                status => Err(MetalError::CommandBufferError(format!(
                    "compute dispatch for '{}' finished with status {status:?}",
                    self.function_name
                ))),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (memory, handles, scalar_bytes, total_threads);
            Err(MetalError::UnsupportedPlatform)
        }
    }
}

impl std::fmt::Debug for MetalComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalComputePipeline(fn={})", self.function_name)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    #[cfg(target_os = "macos")]
    fn try_device() -> Option<MetalDevice> {
        MetalDevice::new().ok()
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn pipeline_compile_valid_msl() {
        let Some(dev) = try_device() else {
            return;
        };
        let src = crate::msl::gemm_msl();
        let p = MetalComputePipeline::new(&dev, src, "gemm_f32")
            .expect("pipeline creation from valid MSL should succeed");
        assert_eq!(p.function_name(), "gemm_f32");
        let dbg = format!("{p:?}");
        assert!(dbg.contains("gemm_f32"));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn pipeline_bad_msl_returns_shader_error() {
        let Some(dev) = try_device() else {
            return;
        };
        let bad_src = "this is not valid MSL !!!";
        let err = MetalComputePipeline::new(&dev, bad_src, "nope").unwrap_err();
        assert!(matches!(err, MetalError::ShaderCompilation(_)));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn pipeline_missing_function_returns_error() {
        let Some(dev) = try_device() else {
            return;
        };
        let src = crate::msl::gemm_msl();
        let err = MetalComputePipeline::new(&dev, src, "nonexistent_function").unwrap_err();
        assert!(matches!(err, MetalError::ShaderCompilation(_)));
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn pipeline_unsupported_on_non_macos() {
        // On non-macOS we can't even construct a MetalDevice, so just verify
        // the UnsupportedPlatform error is what MetalDevice returns.
        let result = MetalDevice::new();
        assert!(matches!(result, Err(MetalError::UnsupportedPlatform)));
    }
}
