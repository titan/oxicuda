//! Multi-node distributed training support (TCP/IP based).
//!
//! This module provides the multi-NODE coordination layer on top of the
//! single-node collective communication primitives in [`crate::collective`].
//! It implements PyTorch-style distributed primitives:
//!
//! - **`DistributedRuntime`** — manages multi-node initialization & barriers
//! - **`TcpStore`** / **`FileStore`** — key-value rendezvous stores
//! - **`GradientBucket`** — gradient bucketing for distributed data parallel
//! - **`DistributedOptimizer`** — gradient communication & ZeRO sharding
//!
//! All networking uses `std::net` (pure Rust, no external dependencies).
//!
//! On macOS the runtime returns simulated results since no NVIDIA GPU is
//! present.

use oxicuda_driver::{CudaError, CudaResult};
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::collective::{Communicator, ReduceOp, RingAllReduce, TreeAllReduce};

// ─── NodeId ─────────────────────────────────────────────────

/// Unique identifier for a node in the distributed cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Create a new node identifier.
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// The underlying integer value.
    pub fn value(&self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

// ─── NodeInfo ───────────────────────────────────────────────

/// Metadata about a single node in the cluster.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// This node's identifier.
    pub node_id: NodeId,
    /// Human-readable hostname.
    pub hostname: String,
    /// IP address (v4 or v6 string).
    pub ip_addr: String,
    /// Port the node listens on.
    pub port: u16,
    /// Number of GPUs available on this node.
    pub gpu_count: u32,
    /// Global rank assigned to this node.
    pub rank: u32,
}

impl NodeInfo {
    /// Create a new `NodeInfo`.
    pub fn new(
        node_id: NodeId,
        hostname: &str,
        ip_addr: &str,
        port: u16,
        gpu_count: u32,
        rank: u32,
    ) -> Self {
        Self {
            node_id,
            hostname: hostname.to_string(),
            ip_addr: ip_addr.to_string(),
            port,
            gpu_count,
            rank,
        }
    }
}

// ─── DistributedBackend ─────────────────────────────────────

/// Communication backend for distributed operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DistributedBackend {
    /// TCP/IP sockets (always available).
    #[default]
    Tcp,
    /// Shared-memory transport for intra-node communication.
    SharedMemory,
}

// ─── InitMethod ─────────────────────────────────────────────

/// How the distributed runtime discovers peers.
#[derive(Debug, Clone)]
pub enum InitMethod {
    /// TCP-based rendezvous through a master node.
    TcpRendezvous {
        /// Master node address.
        master_addr: String,
        /// Master node port.
        master_port: u16,
    },
    /// Read configuration from environment variables
    /// (`MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`).
    EnvVars,
    /// File-system rendezvous via a shared directory.
    FileStore(PathBuf),
}

// ─── DistributedConfig ──────────────────────────────────────

/// Configuration for initializing a distributed runtime.
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Total number of participating processes.
    pub world_size: u32,
    /// Rank of this process on its local node (GPU index).
    pub local_rank: u32,
    /// Globally unique rank across all nodes.
    pub global_rank: u32,
    /// Address of the master / rendezvous node.
    pub master_addr: String,
    /// Port on the master node.
    pub master_port: u16,
    /// Communication backend.
    pub backend: DistributedBackend,
}

impl DistributedConfig {
    /// Validate the configuration, returning an error for invalid values.
    pub fn validate(&self) -> CudaResult<()> {
        if self.world_size == 0 {
            return Err(CudaError::InvalidValue);
        }
        if self.global_rank >= self.world_size {
            return Err(CudaError::InvalidValue);
        }
        if self.master_addr.is_empty() {
            return Err(CudaError::InvalidValue);
        }
        if self.master_port == 0 {
            return Err(CudaError::InvalidValue);
        }
        Ok(())
    }
}

// ─── ProcessGroup ───────────────────────────────────────────

/// A subset of ranks that participate in collective operations together.
#[derive(Debug, Clone)]
pub struct ProcessGroup {
    /// Unique identifier for this group.
    pub group_id: u32,
    /// Ranks that belong to this group.
    pub ranks: Vec<u32>,
    /// Number of ranks in the group.
    pub size: u32,
}

impl ProcessGroup {
    /// Create a new process group.
    pub fn new(group_id: u32, ranks: Vec<u32>) -> CudaResult<Self> {
        if ranks.is_empty() {
            return Err(CudaError::InvalidValue);
        }
        let size = ranks.len() as u32;
        Ok(Self {
            group_id,
            ranks,
            size,
        })
    }

    /// Check whether a given rank belongs to this group.
    pub fn contains_rank(&self, rank: u32) -> bool {
        self.ranks.contains(&rank)
    }

    /// Return the local index of `rank` within this group, if present.
    pub fn local_rank(&self, rank: u32) -> Option<usize> {
        self.ranks.iter().position(|&r| r == rank)
    }
}

// ─── DistributedStatus ──────────────────────────────────────

/// Lifecycle status of the distributed runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributedStatus {
    /// Runtime is being set up.
    Initializing,
    /// Runtime is ready for collective operations.
    Ready,
    /// A synchronization primitive is in progress.
    Synchronizing,
    /// An error occurred.
    Error(String),
    /// Runtime has been shut down.
    Shutdown,
}

// ─── DistributedRuntime ─────────────────────────────────────

/// Multi-node distributed training coordinator.
///
/// Manages peer discovery, barriers, and lifecycle for a set of
/// processes spanning multiple machines.
pub struct DistributedRuntime {
    config: DistributedConfig,
    status: Arc<Mutex<DistributedStatus>>,
    /// Epoch counter used for barrier synchronization.
    barrier_epoch: Arc<Mutex<u64>>,
}

impl DistributedRuntime {
    /// Initialize a distributed runtime from an explicit configuration.
    pub fn init(config: DistributedConfig) -> CudaResult<Self> {
        config.validate()?;

        let rt = Self {
            config,
            status: Arc::new(Mutex::new(DistributedStatus::Ready)),
            barrier_epoch: Arc::new(Mutex::new(0)),
        };
        Ok(rt)
    }

    /// Initialize from environment variables.
    ///
    /// Reads `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, and
    /// optionally `LOCAL_RANK`.
    pub fn from_env() -> CudaResult<Self> {
        let master_addr = std::env::var("MASTER_ADDR").map_err(|_| CudaError::InvalidValue)?;
        let master_port: u16 = std::env::var("MASTER_PORT")
            .map_err(|_| CudaError::InvalidValue)?
            .parse()
            .map_err(|_| CudaError::InvalidValue)?;
        let rank: u32 = std::env::var("RANK")
            .map_err(|_| CudaError::InvalidValue)?
            .parse()
            .map_err(|_| CudaError::InvalidValue)?;
        let world_size: u32 = std::env::var("WORLD_SIZE")
            .map_err(|_| CudaError::InvalidValue)?
            .parse()
            .map_err(|_| CudaError::InvalidValue)?;
        let local_rank: u32 = std::env::var("LOCAL_RANK")
            .unwrap_or_else(|_| rank.to_string())
            .parse()
            .map_err(|_| CudaError::InvalidValue)?;

        let config = DistributedConfig {
            world_size,
            local_rank,
            global_rank: rank,
            master_addr,
            master_port,
            backend: DistributedBackend::Tcp,
        };

        Self::init(config)
    }

    /// Total number of processes in the distributed group.
    pub fn world_size(&self) -> u32 {
        self.config.world_size
    }

    /// Global rank of this process.
    pub fn rank(&self) -> u32 {
        self.config.global_rank
    }

    /// Local rank (GPU index on this node).
    pub fn local_rank(&self) -> u32 {
        self.config.local_rank
    }

    /// Whether this process is the master (rank 0).
    pub fn is_master(&self) -> bool {
        self.config.global_rank == 0
    }

    /// Execute a global barrier — all ranks must call this before any
    /// can proceed.
    ///
    /// In host simulation this increments an internal epoch counter.
    pub fn barrier(&self) -> CudaResult<()> {
        let mut status = self.status.lock().map_err(|_| CudaError::InvalidValue)?;
        if *status == DistributedStatus::Shutdown {
            return Err(CudaError::NotInitialized);
        }
        *status = DistributedStatus::Synchronizing;

        let mut epoch = self
            .barrier_epoch
            .lock()
            .map_err(|_| CudaError::InvalidValue)?;
        *epoch += 1;

        *status = DistributedStatus::Ready;
        Ok(())
    }

    /// Current lifecycle status.
    pub fn status(&self) -> DistributedStatus {
        self.status
            .lock()
            .map(|s| s.clone())
            .unwrap_or_else(|_| DistributedStatus::Error("lock poisoned".to_string()))
    }

    /// Shut down the distributed runtime.
    ///
    /// Idempotent — calling shutdown on an already-shut-down runtime is a
    /// no-op.
    pub fn shutdown(&self) -> CudaResult<()> {
        let mut status = self.status.lock().map_err(|_| CudaError::InvalidValue)?;
        *status = DistributedStatus::Shutdown;
        Ok(())
    }
}

// ─── TcpStore ───────────────────────────────────────────────

/// In-memory key-value store for distributed rendezvous.
///
/// Mirrors PyTorch's `TCPStore`. In this host simulation the store is
/// backed by a `HashMap` behind a `Mutex`; a real implementation would
/// run a TCP server on the master and proxy `set`/`get` over the wire.
pub struct TcpStore {
    /// Master address (informational).
    _master_addr: String,
    /// Port (informational).
    _port: u16,
    /// Expected number of workers.
    _world_size: u32,
    /// Whether this instance is the master.
    is_master: bool,
    /// The actual key-value data.
    data: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    /// Atomic counters for `add`.
    counters: Arc<Mutex<HashMap<String, i64>>>,
}

impl TcpStore {
    /// Create a new TCP store.
    ///
    /// On the master rank (`is_master = true`) the store is authoritative;
    /// workers connect to it for reads/writes.
    pub fn new(master_addr: &str, port: u16, world_size: u32, is_master: bool) -> CudaResult<Self> {
        if master_addr.is_empty() || world_size == 0 {
            return Err(CudaError::InvalidValue);
        }
        Ok(Self {
            _master_addr: master_addr.to_string(),
            _port: port,
            _world_size: world_size,
            is_master,
            data: Arc::new(Mutex::new(HashMap::new())),
            counters: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Whether this store instance is the master.
    pub fn is_master(&self) -> bool {
        self.is_master
    }

    /// Set a key to the given value.
    pub fn set(&self, key: &str, value: &[u8]) -> CudaResult<()> {
        let mut data = self.data.lock().map_err(|_| CudaError::InvalidValue)?;
        data.insert(key.to_string(), value.to_vec());
        Ok(())
    }

    /// Retrieve the value for `key`, or `CudaError::InvalidValue` if absent.
    pub fn get(&self, key: &str) -> CudaResult<Vec<u8>> {
        let data = self.data.lock().map_err(|_| CudaError::InvalidValue)?;
        data.get(key).cloned().ok_or(CudaError::InvalidValue)
    }

    /// Block until all specified keys exist in the store.
    ///
    /// In host simulation this succeeds immediately if all keys are
    /// present, otherwise returns an error (no actual blocking).
    pub fn wait(&self, keys: &[&str]) -> CudaResult<()> {
        let data = self.data.lock().map_err(|_| CudaError::InvalidValue)?;
        for &k in keys {
            if !data.contains_key(k) {
                return Err(CudaError::NotReady);
            }
        }
        Ok(())
    }

    /// Atomically add `amount` to the counter stored under `key`.
    ///
    /// If the key does not exist it is initialised to 0 before adding.
    /// Returns the new value.
    pub fn add(&self, key: &str, amount: i64) -> CudaResult<i64> {
        let mut counters = self.counters.lock().map_err(|_| CudaError::InvalidValue)?;
        let entry = counters.entry(key.to_string()).or_insert(0);
        *entry += amount;
        Ok(*entry)
    }
}

// ─── FileStore ──────────────────────────────────────────────

/// File-system based rendezvous store for shared filesystems (NFS, etc.).
///
/// Each key is stored as a separate file under the root directory.
pub struct FileStore {
    /// Root directory for the store.
    root: PathBuf,
}

impl FileStore {
    /// Create a new file store rooted at `path`.
    ///
    /// The directory is created if it does not exist.
    pub fn new(path: &Path) -> CudaResult<Self> {
        std::fs::create_dir_all(path).map_err(|_| CudaError::InvalidValue)?;
        Ok(Self {
            root: path.to_path_buf(),
        })
    }

    /// Sanitize a key to a safe filename component.
    fn key_path(&self, key: &str) -> PathBuf {
        let safe: String = key
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        self.root.join(safe)
    }

    /// Set a key to the given value.
    pub fn set(&self, key: &str, value: &[u8]) -> CudaResult<()> {
        std::fs::write(self.key_path(key), value).map_err(|_| CudaError::InvalidValue)
    }

    /// Retrieve the value for `key`.
    pub fn get(&self, key: &str) -> CudaResult<Vec<u8>> {
        std::fs::read(self.key_path(key)).map_err(|_| CudaError::InvalidValue)
    }

    /// Block until all keys exist on disk.
    ///
    /// In host simulation this is a single check (no polling).
    pub fn wait(&self, keys: &[&str]) -> CudaResult<()> {
        for &k in keys {
            if !self.key_path(k).exists() {
                return Err(CudaError::NotReady);
            }
        }
        Ok(())
    }

    /// Atomically add `amount` to a counter stored in a file.
    ///
    /// The read-modify-write cycle is guarded by a `FileLockGuard`: an
    /// advisory, cross-process mutex built from a `create_new` sentinel file
    /// next to the counter (rather than
    /// [`File::lock`](std::fs::File::lock), which requires a newer Rust than
    /// this crate's declared MSRV). The guard is held for the duration of
    /// the read, update, and write, so concurrent callers — even across
    /// separate processes sharing the same store directory — never lose an
    /// update or observe a torn read.
    pub fn add(&self, key: &str, amount: i64) -> CudaResult<i64> {
        use std::io::{Read, Seek, SeekFrom, Write};

        let path = self.key_path(key);
        let lock_path = lock_path_for(&path);
        let _lock_guard = FileLockGuard::acquire(&lock_path)?;

        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|_| CudaError::InvalidValue)?;

        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|_| CudaError::InvalidValue)?;
        let trimmed = contents.trim();
        let current: i64 = if trimmed.is_empty() {
            0
        } else {
            trimmed.parse().map_err(|_| CudaError::InvalidValue)?
        };
        let new_val = current + amount;

        file.set_len(0).map_err(|_| CudaError::InvalidValue)?;
        file.seek(SeekFrom::Start(0))
            .map_err(|_| CudaError::InvalidValue)?;
        file.write_all(new_val.to_string().as_bytes())
            .map_err(|_| CudaError::InvalidValue)?;
        file.flush().map_err(|_| CudaError::InvalidValue)?;

        // `_lock_guard` releases the advisory lock (removes the sentinel
        // file) here, on drop.
        Ok(new_val)
    }
}

/// Returns the sentinel lock-file path for the counter file at `path`.
///
/// Real counter files are produced by [`FileStore::key_path`], which
/// sanitizes the key to alphanumerics/`_`/`-`, so appending a literal `.lock`
/// extension can never collide with a real counter file.
fn lock_path_for(path: &Path) -> PathBuf {
    let mut file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_string();
    file_name.push_str(".lock");
    path.with_file_name(file_name)
}

/// RAII guard for an advisory, cross-process file lock.
///
/// The lock is a sentinel file created with
/// [`OpenOptions::create_new`](std::fs::OpenOptions::create_new), which is
/// atomic on every platform `std` supports: at most one caller can create it
/// successfully at a time, and every other concurrent caller (including in
/// other processes on a shared filesystem) observes
/// [`ErrorKind::AlreadyExists`](std::io::ErrorKind::AlreadyExists) and spins
/// until the holder drops its guard and removes the sentinel.
struct FileLockGuard {
    path: PathBuf,
}

impl FileLockGuard {
    /// Maximum number of acquisition attempts before giving up, bounding the
    /// wait so a crashed holder's stale lock file cannot hang callers
    /// forever (roughly two seconds at the `RETRY_DELAY` below).
    const MAX_ATTEMPTS: u32 = 10_000;
    /// Delay between acquisition attempts while the lock is contended.
    const RETRY_DELAY: std::time::Duration = std::time::Duration::from_micros(200);

    /// Spins until the sentinel lock file at `path` can be created
    /// exclusively, or returns an error once [`Self::MAX_ATTEMPTS`] is
    /// exhausted.
    fn acquire(path: &Path) -> CudaResult<Self> {
        for _ in 0..Self::MAX_ATTEMPTS {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
            {
                Ok(_) => {
                    return Ok(Self {
                        path: path.to_path_buf(),
                    });
                }
                Err(e) if Self::is_contention(&e) => {
                    std::thread::sleep(Self::RETRY_DELAY);
                }
                Err(_) => return Err(CudaError::InvalidValue),
            }
        }
        Err(CudaError::InvalidValue)
    }

    /// Whether `err` from the `create_new` attempt means "someone else holds
    /// the lock right now" rather than a genuine, non-retryable failure.
    ///
    /// The obvious signal is [`ErrorKind::AlreadyExists`]. Windows adds a
    /// second one: `DeleteFile` on a sentinel that still has an open handle
    /// only *marks* it for deletion, and until the last handle closes the name
    /// stays in the directory in a "delete pending" state. `CreateFileW` on a
    /// delete-pending name fails with `ERROR_ACCESS_DENIED` (5), surfacing as
    /// [`ErrorKind::PermissionDenied`] — so on Windows the releasing thread's
    /// own `remove_file` makes concurrent acquirers see `PermissionDenied` for
    /// a brief window. Treating that as fatal turned ordinary contention into
    /// spurious `InvalidValue` failures; it is retryable, exactly like
    /// `AlreadyExists`. A real permission problem still fails, just after the
    /// bounded retry loop rather than immediately.
    fn is_contention(err: &std::io::Error) -> bool {
        if err.kind() == std::io::ErrorKind::AlreadyExists {
            return true;
        }
        cfg!(windows) && err.kind() == std::io::ErrorKind::PermissionDenied
    }
}

impl Drop for FileLockGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

// ─── GradientBucket ─────────────────────────────────────────

/// A single bucket of gradients for communication overlap.
#[derive(Debug, Clone)]
pub struct Bucket {
    /// Parameter indices in this bucket.
    pub param_ids: Vec<usize>,
    /// Accumulated size in bytes.
    pub total_size: usize,
    /// Whether all gradients in this bucket have been computed.
    pub ready: bool,
    /// Flattened gradient values held by this rank for every parameter in the
    /// bucket, laid out in `param_ids` order.
    ///
    /// [`DistributedOptimizer::all_reduce_gradients`] reduces this buffer
    /// across all participating ranks and writes the result back in place, so
    /// after a successful all-reduce every bucket's `gradients` holds the
    /// summed (or averaged) values across the process group.
    pub gradients: Vec<f32>,
}

/// Gradient bucketing for distributed data parallel training.
///
/// Groups parameter gradients into fixed-size buckets so that
/// communication can overlap with backward-pass computation.
#[derive(Debug, Clone)]
pub struct GradientBucket {
    /// Maximum bucket size in bytes.
    bucket_size_bytes: usize,
    /// Current list of buckets.
    buckets: Vec<Bucket>,
}

impl GradientBucket {
    /// Create a new gradient bucketing scheme.
    ///
    /// `bucket_size_mb` is the target bucket capacity in megabytes.
    pub fn new(bucket_size_mb: usize) -> Self {
        Self {
            bucket_size_bytes: bucket_size_mb * 1024 * 1024,
            buckets: Vec::new(),
        }
    }

    /// Add a gradient for parameter `param_id` with `grad_size` bytes.
    ///
    /// If the current bucket cannot fit the gradient a new bucket is
    /// started. The gradient buffer is reserved as zero-filled `f32` storage
    /// (`grad_size / 4` elements); use [`add_gradient_data`](Self::add_gradient_data)
    /// to supply concrete gradient values for an all-reduce.
    pub fn add_gradient(&mut self, param_id: usize, grad_size: usize) {
        let elem_count = grad_size / std::mem::size_of::<f32>();
        self.insert(param_id, grad_size, vec![0.0f32; elem_count]);
    }

    /// Add a gradient for parameter `param_id` with concrete `f32` values.
    ///
    /// The gradient occupies `grad.len() * 4` bytes for bucket-capacity
    /// accounting. The values are stored so that
    /// [`DistributedOptimizer::all_reduce_gradients`] can reduce them across
    /// the process group.
    ///
    /// If the current bucket cannot fit the gradient a new bucket is started.
    pub fn add_gradient_data(&mut self, param_id: usize, grad: &[f32]) {
        let grad_size = std::mem::size_of_val(grad);
        self.insert(param_id, grad_size, grad.to_vec());
    }

    /// Shared insertion logic: place a gradient (size + data) into the trailing
    /// bucket, starting a new bucket when the capacity would be exceeded.
    fn insert(&mut self, param_id: usize, grad_size: usize, data: Vec<f32>) {
        let needs_new = self.buckets.is_empty()
            || self
                .buckets
                .last()
                .is_none_or(|b| b.total_size + grad_size > self.bucket_size_bytes);

        if needs_new {
            self.buckets.push(Bucket {
                param_ids: vec![param_id],
                total_size: grad_size,
                ready: false,
                gradients: data,
            });
        } else if let Some(last) = self.buckets.last_mut() {
            last.param_ids.push(param_id);
            last.total_size += grad_size;
            last.gradients.extend_from_slice(&data);
        }
    }

    /// Read-only access to the computed buckets.
    pub fn buckets(&self) -> &[Bucket] {
        &self.buckets
    }

    /// Mark a specific bucket as ready for communication.
    pub fn mark_ready(&mut self, bucket_idx: usize) -> CudaResult<()> {
        let bucket = self
            .buckets
            .get_mut(bucket_idx)
            .ok_or(CudaError::InvalidValue)?;
        bucket.ready = true;
        Ok(())
    }

    /// Number of buckets.
    pub fn num_buckets(&self) -> usize {
        self.buckets.len()
    }

    /// Target bucket capacity in bytes.
    pub fn bucket_capacity(&self) -> usize {
        self.bucket_size_bytes
    }
}

// ─── DataParallelConfig ─────────────────────────────────────

/// Configuration for distributed data-parallel training.
#[derive(Debug, Clone)]
pub struct DataParallelConfig {
    /// Target gradient bucket size in MB.
    pub gradient_bucket_size_mb: usize,
    /// Whether to overlap communication with backward computation.
    pub overlap_communication: bool,
    /// Whether to detect and skip unused parameters in each iteration.
    pub find_unused_parameters: bool,
}

impl Default for DataParallelConfig {
    fn default() -> Self {
        Self {
            gradient_bucket_size_mb: 25,
            overlap_communication: true,
            find_unused_parameters: false,
        }
    }
}

// ─── ModelParallelConfig ────────────────────────────────────

/// Configuration for model-parallel training.
#[derive(Debug, Clone)]
pub struct ModelParallelConfig {
    /// Number of GPUs across which each tensor is sharded.
    pub tensor_parallel_size: u32,
    /// Number of pipeline stages.
    pub pipeline_parallel_size: u32,
    /// Whether to enable sequence-parallelism (split along sequence dim).
    pub sequence_parallel: bool,
}

impl ModelParallelConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> CudaResult<()> {
        if self.tensor_parallel_size == 0 {
            return Err(CudaError::InvalidValue);
        }
        if self.pipeline_parallel_size == 0 {
            return Err(CudaError::InvalidValue);
        }
        Ok(())
    }

    /// Total number of GPUs required (tensor × pipeline parallelism).
    pub fn total_gpus_required(&self) -> u32 {
        self.tensor_parallel_size * self.pipeline_parallel_size
    }
}

impl Default for ModelParallelConfig {
    fn default() -> Self {
        Self {
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            sequence_parallel: false,
        }
    }
}

// ─── DistributedOptimizer ───────────────────────────────────

/// Wraps gradient communication for distributed training.
///
/// Provides a real ring all-reduce over gradient buckets (built on the
/// [`crate::collective`] primitives) and ZeRO-style optimizer-state
/// partitioning.
pub struct DistributedOptimizer;

impl DistributedOptimizer {
    /// Performs an all-reduce of every bucket's gradients across the process
    /// group described by `comm`.
    ///
    /// Each participating rank contributes its own copy of `bucket.gradients`;
    /// the reduction combines them element-wise with `op` and writes the
    /// converged result back into every bucket. After a successful call each
    /// bucket's `gradients` therefore holds the summed gradient (or the
    /// average, for [`ReduceOp::Avg`]) across all `comm.world_size()`
    /// participants — the value an optimizer step should consume.
    ///
    /// The reduction runs through the collective primitives in
    /// [`crate::collective`]: [`RingAllReduce`] — the bandwidth-optimal
    /// algorithm — for buckets whose element count is at least the world
    /// size, and the latency-optimal [`TreeAllReduce`] for smaller buckets
    /// (the ring algorithm chunks the buffer per rank and is only well-defined
    /// when each rank owns at least one element). This mirrors the
    /// NCCL `ncclAllReduce` data path. On a single host the other ranks'
    /// buffers are materialised from this rank's data, which is the standard
    /// host-simulation contract used throughout the collective module.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::NotReady`] if any bucket has not been marked
    /// ready, and [`CudaError::InvalidValue`] if the ring all-reduce rejects
    /// the per-rank buffers (e.g. inconsistent lengths).
    pub fn all_reduce_gradients(
        buckets: &mut [Bucket],
        comm: &Communicator,
        op: ReduceOp,
    ) -> CudaResult<()> {
        // Every bucket must have all its gradients computed before it can be
        // communicated — partial buckets would reduce stale data.
        for bucket in buckets.iter() {
            if !bucket.ready {
                return Err(CudaError::NotReady);
            }
        }

        let world_size = comm.world_size();
        for bucket in buckets.iter_mut() {
            if bucket.gradients.is_empty() {
                // An empty gradient buffer has nothing to reduce.
                continue;
            }

            if world_size < 2 {
                // A single participant: the all-reduce is the identity.
                // Averaging by a world size of 1 is also a no-op, so the
                // bucket gradients are already the correct result.
                continue;
            }

            // Materialise one buffer per rank. In this host simulation every
            // rank starts from the same locally-computed gradients; a real
            // multi-GPU run would pass each device's own buffer here.
            let mut per_rank: Vec<Vec<f32>> =
                (0..world_size).map(|_| bucket.gradients.clone()).collect();

            // The ring algorithm partitions each buffer into one chunk per
            // rank, so it requires at least `world_size` elements; fall back
            // to the tree algorithm for shorter gradient buffers.
            if bucket.gradients.len() >= world_size {
                RingAllReduce::execute(&mut per_rank, op).map_err(|_| CudaError::InvalidValue)?;
            } else {
                TreeAllReduce::execute(&mut per_rank, op).map_err(|_| CudaError::InvalidValue)?;
            }

            // After a ring all-reduce every rank holds the identical result;
            // adopt this rank's converged buffer as the bucket's gradients.
            if let Some(reduced) = per_rank.into_iter().next() {
                bucket.gradients = reduced;
            }
        }

        Ok(())
    }

    /// Convenience wrapper that all-reduces gradients with summation, the
    /// default convention for data-parallel training.
    ///
    /// Equivalent to [`all_reduce_gradients`](Self::all_reduce_gradients) with
    /// [`ReduceOp::Sum`].
    ///
    /// # Errors
    ///
    /// Same as [`all_reduce_gradients`](Self::all_reduce_gradients).
    pub fn all_reduce_gradients_sum(buckets: &mut [Bucket], comm: &Communicator) -> CudaResult<()> {
        Self::all_reduce_gradients(buckets, comm, ReduceOp::Sum)
    }

    /// Compute ZeRO-style parameter sharding ranges.
    ///
    /// Partitions `param_count` parameters evenly across `world_size`
    /// ranks, returning one `Range<usize>` per rank.
    pub fn zero_redundancy_partition(world_size: u32, param_count: usize) -> Vec<Range<usize>> {
        if world_size == 0 {
            return Vec::new();
        }
        let ws = world_size as usize;
        let base = param_count / ws;
        let remainder = param_count % ws;

        let mut ranges = Vec::with_capacity(ws);
        let mut start = 0;
        for i in 0..ws {
            let extra = if i < remainder { 1 } else { 0 };
            let end = start + base + extra;
            ranges.push(start..end);
            start = end;
        }
        ranges
    }
}

// ─── Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> DistributedConfig {
        DistributedConfig {
            world_size: 4,
            local_rank: 0,
            global_rank: 0,
            master_addr: "127.0.0.1".to_string(),
            master_port: 29500,
            backend: DistributedBackend::Tcp,
        }
    }

    // ── DistributedConfig creation and validation ───────────

    #[test]
    fn config_valid() {
        let cfg = sample_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_invalid_world_size_zero() {
        let mut cfg = sample_config();
        cfg.world_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_invalid_rank_exceeds_world() {
        let mut cfg = sample_config();
        cfg.global_rank = 10;
        assert!(cfg.validate().is_err());
    }

    // ── TcpStore set/get roundtrip ──────────────────────────

    #[test]
    fn tcp_store_set_get() {
        let store = TcpStore::new("127.0.0.1", 29500, 2, true).expect("create store");
        assert!(store.is_master());

        store.set("key1", b"hello").expect("set");
        let val = store.get("key1").expect("get");
        assert_eq!(val, b"hello");
    }

    #[test]
    fn tcp_store_get_missing_key() {
        let store = TcpStore::new("127.0.0.1", 29500, 1, true).expect("create store");
        assert!(store.get("nonexistent").is_err());
    }

    #[test]
    fn tcp_store_add_counter() {
        let store = TcpStore::new("127.0.0.1", 29500, 1, true).expect("create store");
        let v1 = store.add("counter", 5).expect("add");
        assert_eq!(v1, 5);
        let v2 = store.add("counter", 3).expect("add");
        assert_eq!(v2, 8);
    }

    #[test]
    fn tcp_store_wait_present() {
        let store = TcpStore::new("127.0.0.1", 29500, 1, true).expect("create store");
        store.set("a", b"1").expect("set");
        store.set("b", b"2").expect("set");
        assert!(store.wait(&["a", "b"]).is_ok());
    }

    #[test]
    fn tcp_store_wait_missing() {
        let store = TcpStore::new("127.0.0.1", 29500, 1, true).expect("create store");
        store.set("a", b"1").expect("set");
        assert!(store.wait(&["a", "missing"]).is_err());
    }

    // ── FileStore set/get roundtrip ─────────────────────────

    #[test]
    fn file_store_set_get() {
        let dir = std::env::temp_dir().join("oxicuda_test_filestore_setget");
        let _ = std::fs::remove_dir_all(&dir);
        let store = FileStore::new(&dir).expect("create file store");
        store.set("mykey", b"world").expect("set");
        let val = store.get("mykey").expect("get");
        assert_eq!(val, b"world");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_store_add_counter() {
        let dir = std::env::temp_dir().join("oxicuda_test_filestore_add");
        let _ = std::fs::remove_dir_all(&dir);
        let store = FileStore::new(&dir).expect("create file store");
        let v1 = store.add("ctr", 10).expect("add");
        assert_eq!(v1, 10);
        let v2 = store.add("ctr", -3).expect("add");
        assert_eq!(v2, 7);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Concurrent `add` calls on the same key, from multiple threads racing
    /// on the same on-disk counter file, must never lose an update: the
    /// final value must equal the sum of every increment. This guards
    /// against the read-modify-write race fixed by the advisory file lock
    /// in [`FileStore::add`].
    #[test]
    fn file_store_add_concurrent_no_lost_updates() {
        let dir = std::env::temp_dir().join("oxicuda_test_filestore_add_concurrent");
        let _ = std::fs::remove_dir_all(&dir);
        let store = Arc::new(FileStore::new(&dir).expect("create file store"));

        const THREADS: usize = 8;
        const INCREMENTS_PER_THREAD: i64 = 25;

        std::thread::scope(|scope| {
            for _ in 0..THREADS {
                let store_ref = Arc::clone(&store);
                scope.spawn(move || {
                    for _ in 0..INCREMENTS_PER_THREAD {
                        store_ref.add("race_ctr", 1).expect("add should succeed");
                    }
                });
            }
        });

        let final_val = store.add("race_ctr", 0).expect("final read via add");
        assert_eq!(
            final_val,
            THREADS as i64 * INCREMENTS_PER_THREAD,
            "concurrent add() calls lost updates"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_store_wait() {
        let dir = std::env::temp_dir().join("oxicuda_test_filestore_wait");
        let _ = std::fs::remove_dir_all(&dir);
        let store = FileStore::new(&dir).expect("create file store");
        store.set("x", b"1").expect("set");
        assert!(store.wait(&["x"]).is_ok());
        assert!(store.wait(&["x", "y"]).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── ProcessGroup creation ───────────────────────────────

    #[test]
    fn process_group_creation() {
        let pg = ProcessGroup::new(0, vec![0, 1, 2, 3]).expect("create pg");
        assert_eq!(pg.size, 4);
        assert_eq!(pg.group_id, 0);
        assert!(pg.contains_rank(2));
        assert!(!pg.contains_rank(5));
        assert_eq!(pg.local_rank(3), Some(3));
        assert_eq!(pg.local_rank(9), None);
    }

    #[test]
    fn process_group_empty_rejected() {
        assert!(ProcessGroup::new(0, vec![]).is_err());
    }

    // ── GradientBucket partitioning ─────────────────────────

    #[test]
    fn gradient_bucket_partitioning() {
        // 1 MB buckets
        let mut gb = GradientBucket::new(1);
        // Each gradient is 512 KB => 2 fit per bucket
        let half_mb = 512 * 1024;
        gb.add_gradient(0, half_mb);
        gb.add_gradient(1, half_mb);
        gb.add_gradient(2, half_mb);

        assert_eq!(gb.num_buckets(), 2);
        assert_eq!(gb.buckets()[0].param_ids, vec![0, 1]);
        assert_eq!(gb.buckets()[1].param_ids, vec![2]);
    }

    #[test]
    fn gradient_bucket_size_distribution() {
        let mut gb = GradientBucket::new(2); // 2 MB buckets
        let one_mb = 1024 * 1024;
        for i in 0..5 {
            gb.add_gradient(i, one_mb);
        }
        // Expect 3 buckets: [0,1], [2,3], [4]
        assert_eq!(gb.num_buckets(), 3);
        assert_eq!(gb.buckets()[0].total_size, 2 * one_mb);
        assert_eq!(gb.buckets()[2].param_ids.len(), 1);
    }

    // ── ZeRO parameter sharding ─────────────────────────────

    #[test]
    fn zero_sharding_even() {
        let ranges = DistributedOptimizer::zero_redundancy_partition(4, 100);
        assert_eq!(ranges.len(), 4);
        assert_eq!(ranges[0], 0..25);
        assert_eq!(ranges[1], 25..50);
        assert_eq!(ranges[2], 50..75);
        assert_eq!(ranges[3], 75..100);
    }

    #[test]
    fn zero_sharding_uneven() {
        let ranges = DistributedOptimizer::zero_redundancy_partition(3, 10);
        assert_eq!(ranges.len(), 3);
        // 10 / 3 = 3 remainder 1 => first rank gets 4
        assert_eq!(ranges[0], 0..4);
        assert_eq!(ranges[1], 4..7);
        assert_eq!(ranges[2], 7..10);
    }

    #[test]
    fn zero_sharding_zero_world() {
        let ranges = DistributedOptimizer::zero_redundancy_partition(0, 100);
        assert!(ranges.is_empty());
    }

    // ── Environment variable parsing ────────────────────────

    #[test]
    fn from_env_missing_vars() {
        // Without the env vars set, this should fail gracefully
        // (We don't set them here on purpose)
        // Note: if a CI sets these vars this test would behave differently,
        // so we only assert it doesn't panic.
        let _result = DistributedRuntime::from_env();
    }

    // ── Barrier logic ───────────────────────────────────────

    #[test]
    fn barrier_increments_epoch() {
        let rt = DistributedRuntime::init(sample_config()).expect("init");
        rt.barrier().expect("barrier 1");
        rt.barrier().expect("barrier 2");
        let epoch = rt.barrier_epoch.lock().expect("lock");
        assert_eq!(*epoch, 2);
    }

    // ── Master node detection ───────────────────────────────

    #[test]
    fn master_detection() {
        let cfg = sample_config(); // global_rank = 0
        let rt = DistributedRuntime::init(cfg).expect("init");
        assert!(rt.is_master());

        let mut cfg2 = sample_config();
        cfg2.global_rank = 2;
        let rt2 = DistributedRuntime::init(cfg2).expect("init");
        assert!(!rt2.is_master());
    }

    // ── World size / rank accessors ─────────────────────────

    #[test]
    fn world_size_rank_accessors() {
        let mut cfg = sample_config();
        cfg.world_size = 8;
        cfg.global_rank = 3;
        cfg.local_rank = 1;
        let rt = DistributedRuntime::init(cfg).expect("init");
        assert_eq!(rt.world_size(), 8);
        assert_eq!(rt.rank(), 3);
        assert_eq!(rt.local_rank(), 1);
    }

    // ── DataParallelConfig defaults ─────────────────────────

    #[test]
    fn data_parallel_config_defaults() {
        let dpc = DataParallelConfig::default();
        assert_eq!(dpc.gradient_bucket_size_mb, 25);
        assert!(dpc.overlap_communication);
        assert!(!dpc.find_unused_parameters);
    }

    // ── ModelParallelConfig validation ──────────────────────

    #[test]
    fn model_parallel_config_validation() {
        let mpc = ModelParallelConfig::default();
        assert!(mpc.validate().is_ok());
        assert_eq!(mpc.total_gpus_required(), 1);

        let bad = ModelParallelConfig {
            tensor_parallel_size: 0,
            pipeline_parallel_size: 4,
            sequence_parallel: false,
        };
        assert!(bad.validate().is_err());
    }

    // ── Status transitions ──────────────────────────────────

    #[test]
    fn status_transitions() {
        let rt = DistributedRuntime::init(sample_config()).expect("init");
        assert_eq!(rt.status(), DistributedStatus::Ready);

        rt.barrier().expect("barrier");
        assert_eq!(rt.status(), DistributedStatus::Ready);

        rt.shutdown().expect("shutdown");
        assert_eq!(rt.status(), DistributedStatus::Shutdown);
    }

    // ── Shutdown idempotency ────────────────────────────────

    #[test]
    fn shutdown_idempotent() {
        let rt = DistributedRuntime::init(sample_config()).expect("init");
        rt.shutdown().expect("shutdown 1");
        rt.shutdown().expect("shutdown 2");
        assert_eq!(rt.status(), DistributedStatus::Shutdown);
    }

    // ── Barrier after shutdown ──────────────────────────────

    #[test]
    fn barrier_after_shutdown_fails() {
        let rt = DistributedRuntime::init(sample_config()).expect("init");
        rt.shutdown().expect("shutdown");
        assert!(rt.barrier().is_err());
    }

    // ── AllReduce gradients ─────────────────────────────────

    #[test]
    fn all_reduce_gradients_ready() {
        let mut buckets = vec![
            Bucket {
                param_ids: vec![0, 1],
                total_size: 1024,
                ready: true,
                gradients: vec![1.0, 2.0, 3.0, 4.0],
            },
            Bucket {
                param_ids: vec![2],
                total_size: 512,
                ready: true,
                gradients: vec![5.0, 6.0],
            },
        ];
        let comm = Communicator::new(&[0, 1, 2, 3]).expect("comm");
        assert!(
            DistributedOptimizer::all_reduce_gradients(&mut buckets, &comm, ReduceOp::Sum).is_ok()
        );
    }

    #[test]
    fn all_reduce_gradients_not_ready() {
        let mut buckets = vec![
            Bucket {
                param_ids: vec![0],
                total_size: 1024,
                ready: true,
                gradients: vec![1.0; 4],
            },
            Bucket {
                param_ids: vec![1],
                total_size: 512,
                ready: false,
                gradients: vec![1.0; 2],
            },
        ];
        let comm = Communicator::new(&[0, 1]).expect("comm");
        assert!(
            DistributedOptimizer::all_reduce_gradients(&mut buckets, &comm, ReduceOp::Sum).is_err()
        );
    }

    #[test]
    fn all_reduce_gradients_sums_across_ranks() {
        // 4 ranks, each contributing the same [1, 2, 3, 4] => sum = [4, 8, 12, 16].
        let mut buckets = vec![Bucket {
            param_ids: vec![0],
            total_size: 16,
            ready: true,
            gradients: vec![1.0, 2.0, 3.0, 4.0],
        }];
        let comm = Communicator::new(&[0, 1, 2, 3]).expect("comm");
        DistributedOptimizer::all_reduce_gradients(&mut buckets, &comm, ReduceOp::Sum)
            .expect("all-reduce should succeed");
        let expected = [4.0f32, 8.0, 12.0, 16.0];
        for (got, want) in buckets[0].gradients.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
    }

    #[test]
    fn all_reduce_gradients_averages_across_ranks() {
        // 4 ranks of identical data, Avg => the mean equals the original data.
        let mut buckets = vec![Bucket {
            param_ids: vec![0, 1],
            total_size: 24,
            ready: true,
            gradients: vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        }];
        let comm = Communicator::new(&[0, 1, 2, 3]).expect("comm");
        DistributedOptimizer::all_reduce_gradients(&mut buckets, &comm, ReduceOp::Avg)
            .expect("all-reduce should succeed");
        let expected = [2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0];
        for (got, want) in buckets[0].gradients.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
        }
    }

    #[test]
    fn all_reduce_gradients_single_rank_is_identity() {
        // A world size of 1 leaves the gradients untouched.
        let mut buckets = vec![Bucket {
            param_ids: vec![0],
            total_size: 12,
            ready: true,
            gradients: vec![7.0, 8.0, 9.0],
        }];
        let comm = Communicator::new(&[0]).expect("comm");
        DistributedOptimizer::all_reduce_gradients(&mut buckets, &comm, ReduceOp::Sum)
            .expect("all-reduce should succeed");
        assert_eq!(buckets[0].gradients, vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn all_reduce_gradients_sum_wrapper() {
        let mut buckets = vec![Bucket {
            param_ids: vec![0],
            total_size: 8,
            ready: true,
            gradients: vec![1.0, 1.0],
        }];
        let comm = Communicator::new(&[0, 1]).expect("comm");
        DistributedOptimizer::all_reduce_gradients_sum(&mut buckets, &comm)
            .expect("sum all-reduce should succeed");
        // 2 ranks summed => each element doubled.
        assert_eq!(buckets[0].gradients, vec![2.0, 2.0]);
    }

    #[test]
    fn gradient_bucket_carries_real_gradient_data() {
        // add_gradient_data populates the bucket gradient buffer for all-reduce.
        let mut gb = GradientBucket::new(1);
        gb.add_gradient_data(0, &[1.0, 2.0]);
        gb.add_gradient_data(1, &[3.0, 4.0, 5.0]);
        assert_eq!(gb.num_buckets(), 1);
        assert_eq!(gb.buckets()[0].gradients, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut buckets = gb.buckets().to_vec();
        for b in &mut buckets {
            b.ready = true;
        }
        let comm = Communicator::new(&[0, 1]).expect("comm");
        DistributedOptimizer::all_reduce_gradients_sum(&mut buckets, &comm)
            .expect("all-reduce should succeed");
        // 2 ranks summed => every value doubled.
        assert_eq!(buckets[0].gradients, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    // ── NodeInfo ────────────────────────────────────────────

    #[test]
    fn node_info_creation() {
        let ni = NodeInfo::new(NodeId::new(0), "host0", "10.0.0.1", 8080, 4, 0);
        assert_eq!(ni.node_id, NodeId(0));
        assert_eq!(ni.hostname, "host0");
        assert_eq!(ni.gpu_count, 4);
    }

    // ── NodeId display ──────────────────────────────────────

    #[test]
    fn node_id_display() {
        let id = NodeId::new(42);
        assert_eq!(format!("{id}"), "Node(42)");
        assert_eq!(id.value(), 42);
    }
}
