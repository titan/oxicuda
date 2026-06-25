//! `VK_KHR_performance_query` host-side query-pool description.
//!
//! `VK_KHR_performance_query` exposes per-dispatch hardware counters (GPU
//! cycles, ALU utilisation, cache hits, …). Using it on a device requires:
//!
//! 1. enumerating the queue family's available counters
//!    (`vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR`),
//! 2. creating a `VkQueryPool` of type `PERFORMANCE_QUERY_KHR` with a
//!    `VkQueryPoolPerformanceCreateInfoKHR` listing the enabled counter indices,
//! 3. acquiring a profiling lock, recording `vkCmdBeginQuery`/`vkCmdEndQuery`
//!    around the dispatch, and reading back `VkPerformanceCounterResultKHR`.
//!
//! Steps 1–3 are inherently *device-gated*. What **is** CPU-testable — and what
//! this module provides — is the host-side bookkeeping: building the pool
//! description ([`PerformanceQueryPool`]), computing the required result-buffer
//! stride, mapping counter scopes/units, and interpreting a raw counter result
//! once the driver has filled it in.

/// The scope at which a performance counter is measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CounterScope {
    /// Counter accumulated over a whole command buffer.
    CommandBuffer,
    /// Counter accumulated over a render pass (unused for compute).
    RenderPass,
    /// Counter accumulated over a single command (`vkCmdDispatch`).
    Command,
}

/// The storage type the driver uses for a counter result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CounterStorage {
    /// Signed 32-bit integer.
    Int32,
    /// Signed 64-bit integer.
    Int64,
    /// Unsigned 32-bit integer.
    Uint32,
    /// Unsigned 64-bit integer (e.g. GPU cycle counts).
    Uint64,
    /// 32-bit float.
    Float32,
    /// 64-bit float.
    Float64,
}

impl CounterStorage {
    /// Size in bytes of one result of this storage type, as laid out in a
    /// `VkPerformanceCounterResultKHR` union (always 8-byte aligned).
    #[must_use]
    pub fn result_size(self) -> usize {
        match self {
            CounterStorage::Int32 | CounterStorage::Uint32 | CounterStorage::Float32 => 4,
            CounterStorage::Int64 | CounterStorage::Uint64 | CounterStorage::Float64 => 8,
        }
    }
}

/// The physical unit of a counter value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CounterUnit {
    /// Dimensionless count.
    Generic,
    /// A percentage in `[0, 100]`.
    Percentage,
    /// Nanoseconds.
    Nanoseconds,
    /// Bytes.
    Bytes,
    /// Bytes per second.
    BytesPerSecond,
    /// Hardware cycles.
    Cycles,
    /// Hertz.
    Hertz,
}

/// Description of a single enabled performance counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CounterDesc {
    /// Driver-assigned counter index (the value passed in the create-info).
    pub index: u32,
    /// Accumulation scope.
    pub scope: CounterScope,
    /// Result storage type.
    pub storage: CounterStorage,
    /// Physical unit.
    pub unit: CounterUnit,
}

/// A raw counter result as read back from the driver, plus its description.
#[derive(Debug, Clone, Copy)]
pub struct CounterResult {
    /// The counter this result belongs to.
    pub desc: CounterDesc,
    /// The raw 64-bit payload (reinterpreted per `desc.storage`).
    pub raw: u64,
}

impl CounterResult {
    /// Interpret the raw payload as an `f64`, applying the storage type.
    ///
    /// Float storage reinterprets the low bits; integer storage widens.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        match self.desc.storage {
            CounterStorage::Float32 => f32::from_bits(self.raw as u32) as f64,
            CounterStorage::Float64 => f64::from_bits(self.raw),
            CounterStorage::Int32 => (self.raw as u32 as i32) as f64,
            CounterStorage::Int64 => (self.raw as i64) as f64,
            CounterStorage::Uint32 => (self.raw as u32) as f64,
            CounterStorage::Uint64 => self.raw as f64,
        }
    }
}

/// Host-side description of a `VK_KHR_performance_query` query pool.
///
/// Models the data needed for `VkQueryPoolPerformanceCreateInfoKHR` and the
/// readback buffer, without owning any device handle.
#[derive(Debug, Clone)]
pub struct PerformanceQueryPool {
    queue_family_index: u32,
    counters: Vec<CounterDesc>,
    query_count: u32,
}

impl PerformanceQueryPool {
    /// Create a pool description for `queue_family_index` enabling `counters`,
    /// sized to hold `query_count` queries.
    ///
    /// Returns an error if `counters` is empty or `query_count` is zero (a
    /// driver would reject such a pool).
    pub fn new(
        queue_family_index: u32,
        counters: Vec<CounterDesc>,
        query_count: u32,
    ) -> Result<Self, crate::error::VulkanError> {
        if counters.is_empty() {
            return Err(crate::error::VulkanError::InvalidArgument(
                "performance query pool needs at least one counter".into(),
            ));
        }
        if query_count == 0 {
            return Err(crate::error::VulkanError::InvalidArgument(
                "performance query pool needs query_count >= 1".into(),
            ));
        }
        Ok(Self {
            queue_family_index,
            counters,
            query_count,
        })
    }

    /// The queue family the counters belong to.
    #[must_use]
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// The enabled counters.
    #[must_use]
    pub fn counters(&self) -> &[CounterDesc] {
        &self.counters
    }

    /// The counter indices, as required by `VkQueryPoolPerformanceCreateInfoKHR`.
    #[must_use]
    pub fn counter_indices(&self) -> Vec<u32> {
        self.counters.iter().map(|c| c.index).collect()
    }

    /// Number of queries the pool holds.
    #[must_use]
    pub fn query_count(&self) -> u32 {
        self.query_count
    }

    /// Size in bytes of the result buffer for **one** query: one
    /// `VkPerformanceCounterResultKHR` (8 bytes, union-aligned) per counter.
    #[must_use]
    pub fn per_query_stride(&self) -> usize {
        // Each VkPerformanceCounterResultKHR is an 8-byte union regardless of
        // the active member; the spec requires 8-byte alignment.
        self.counters.len() * 8
    }

    /// Total size in bytes of the readback buffer for all queries.
    #[must_use]
    pub fn result_buffer_size(&self) -> usize {
        self.per_query_stride() * self.query_count as usize
    }

    /// Whether any enabled counter is scoped to a single command (which forces
    /// the pass to be a "command" pass and disables some driver optimisations).
    #[must_use]
    pub fn requires_command_scope(&self) -> bool {
        self.counters
            .iter()
            .any(|c| c.scope == CounterScope::Command)
    }

    /// Parse a raw readback buffer for query `query_index` into per-counter
    /// results.
    ///
    /// `buffer` must contain at least `result_buffer_size()` bytes. Each counter
    /// occupies an 8-byte slot; the low bytes are reinterpreted per the
    /// counter's storage type.
    pub fn parse_query(
        &self,
        buffer: &[u8],
        query_index: u32,
    ) -> Result<Vec<CounterResult>, crate::error::VulkanError> {
        if query_index >= self.query_count {
            return Err(crate::error::VulkanError::InvalidArgument(format!(
                "query index {query_index} out of range (count {})",
                self.query_count
            )));
        }
        let stride = self.per_query_stride();
        let base = stride * query_index as usize;
        if buffer.len() < base + stride {
            return Err(crate::error::VulkanError::InvalidArgument(format!(
                "result buffer too small: have {}, need {}",
                buffer.len(),
                base + stride
            )));
        }
        let mut out = Vec::with_capacity(self.counters.len());
        for (i, desc) in self.counters.iter().enumerate() {
            let off = base + i * 8;
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&buffer[off..off + 8]);
            out.push(CounterResult {
                desc: *desc,
                raw: u64::from_le_bytes(bytes),
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cycles_counter(index: u32) -> CounterDesc {
        CounterDesc {
            index,
            scope: CounterScope::Command,
            storage: CounterStorage::Uint64,
            unit: CounterUnit::Cycles,
        }
    }

    fn util_counter(index: u32) -> CounterDesc {
        CounterDesc {
            index,
            scope: CounterScope::CommandBuffer,
            storage: CounterStorage::Float32,
            unit: CounterUnit::Percentage,
        }
    }

    #[test]
    fn empty_counters_rejected() {
        assert!(PerformanceQueryPool::new(0, vec![], 1).is_err());
    }

    #[test]
    fn zero_queries_rejected() {
        assert!(PerformanceQueryPool::new(0, vec![cycles_counter(0)], 0).is_err());
    }

    #[test]
    fn stride_and_buffer_size() {
        let pool =
            PerformanceQueryPool::new(0, vec![cycles_counter(3), util_counter(7)], 4).unwrap();
        assert_eq!(pool.per_query_stride(), 16); // 2 counters * 8
        assert_eq!(pool.result_buffer_size(), 64); // 16 * 4 queries
        assert_eq!(pool.counter_indices(), vec![3, 7]);
        assert!(pool.requires_command_scope());
    }

    #[test]
    fn command_scope_detection() {
        let only_cb = PerformanceQueryPool::new(0, vec![util_counter(1)], 1).unwrap();
        assert!(!only_cb.requires_command_scope());
    }

    #[test]
    fn parse_uint64_cycles() {
        let pool = PerformanceQueryPool::new(0, vec![cycles_counter(0)], 1).unwrap();
        let mut buf = vec![0u8; pool.result_buffer_size()];
        buf[0..8].copy_from_slice(&123_456_789u64.to_le_bytes());
        let results = pool.parse_query(&buf, 0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].raw, 123_456_789);
        assert_eq!(results[0].as_f64(), 123_456_789.0);
    }

    #[test]
    fn parse_float32_percentage() {
        let pool = PerformanceQueryPool::new(0, vec![util_counter(0)], 1).unwrap();
        let mut buf = vec![0u8; pool.result_buffer_size()];
        let bits = 87.5f32.to_bits() as u64;
        buf[0..8].copy_from_slice(&bits.to_le_bytes());
        let results = pool.parse_query(&buf, 0).unwrap();
        assert!((results[0].as_f64() - 87.5).abs() < 1e-6);
    }

    #[test]
    fn parse_second_query_offset() {
        let pool = PerformanceQueryPool::new(0, vec![cycles_counter(0)], 2).unwrap();
        let mut buf = vec![0u8; pool.result_buffer_size()];
        // Query 1 lives at byte 8 (stride 8 for a single counter).
        buf[8..16].copy_from_slice(&999u64.to_le_bytes());
        let q1 = pool.parse_query(&buf, 1).unwrap();
        assert_eq!(q1[0].raw, 999);
    }

    #[test]
    fn parse_out_of_range_query_rejected() {
        let pool = PerformanceQueryPool::new(0, vec![cycles_counter(0)], 1).unwrap();
        let buf = vec![0u8; pool.result_buffer_size()];
        assert!(pool.parse_query(&buf, 5).is_err());
    }

    #[test]
    fn parse_short_buffer_rejected() {
        let pool = PerformanceQueryPool::new(0, vec![cycles_counter(0)], 1).unwrap();
        assert!(pool.parse_query(&[0u8; 4], 0).is_err());
    }

    #[test]
    fn storage_result_sizes() {
        assert_eq!(CounterStorage::Uint32.result_size(), 4);
        assert_eq!(CounterStorage::Uint64.result_size(), 8);
        assert_eq!(CounterStorage::Float64.result_size(), 8);
    }

    #[test]
    fn negative_int32_interpretation() {
        let desc = CounterDesc {
            index: 0,
            scope: CounterScope::Command,
            storage: CounterStorage::Int32,
            unit: CounterUnit::Generic,
        };
        let r = CounterResult {
            desc,
            raw: (-5i32) as u32 as u64,
        };
        assert_eq!(r.as_f64(), -5.0);
    }
}
