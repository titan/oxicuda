//! Persistent storage for autotuning results.
//!
//! The [`ResultDb`] stores the best kernel configuration for each
//! (GPU, kernel, problem-size) triple, backed by a JSON file on disk.
//! This allows tuning results to persist across program runs, so the
//! autotuner does not need to re-benchmark configurations that were
//! already tested.
//!
//! # Default location
//!
//! On Unix systems the database lives at
//! `~/.cache/oxicuda/autotune/results.json`.  If the cache directory
//! cannot be determined, `$TMPDIR/oxicuda_autotune/results.json` is
//! used as a fallback.
//!
//! # Example
//!
//! ```rust
//! use oxicuda_autotune::{ResultDb, Config, BenchmarkResult};
//!
//! # fn example() -> Result<(), std::io::Error> {
//! let dir = std::env::temp_dir().join("oxicuda_test_db");
//! let path = dir.join("results.json");
//! std::fs::create_dir_all(&dir)?;
//!
//! let mut db = ResultDb::open_at(path)?;
//! let result = BenchmarkResult {
//!     config: Config::new(),
//!     median_us: 42.0,
//!     min_us: 40.0,
//!     max_us: 45.0,
//!     stddev_us: 1.5,
//!     gflops: Some(1200.0),
//!     efficiency: None,
//! };
//!
//! db.save("RTX 4090", "sgemm", "1024x1024x1024", result)?;
//!
//! let best = db.lookup("RTX 4090", "sgemm", "1024x1024x1024");
//! assert!(best.is_some());
//! # std::fs::remove_dir_all(&dir)?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::benchmark::BenchmarkResult;

/// Persistent database for autotuning results.
///
/// Stores the best configuration for each (GPU, kernel, problem) triple.
/// Backed by a JSON file that is read on open and written on every save.
pub struct ResultDb {
    /// Path to the backing JSON file.
    path: PathBuf,
    /// In-memory representation of the database.
    data: ResultDbData,
}

/// Serializable representation of the entire result database.
///
/// Structure: `gpu_name -> kernel_name -> problem_key -> BenchmarkResult`.
#[derive(Debug, Default, Serialize, Deserialize)]
struct ResultDbData {
    /// Nested map: GPU name -> kernel name -> problem key -> best result.
    entries: HashMap<String, HashMap<String, HashMap<String, BenchmarkResult>>>,
}

/// Key for looking up tuning results.
///
/// Combines a kernel name with a problem descriptor (e.g. matrix
/// dimensions) to form a unique lookup key within a GPU's result set.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ProblemKey {
    /// Name of the kernel (e.g. `"sgemm"`, `"conv2d_fwd"`).
    pub kernel_name: String,
    /// Problem descriptor (e.g. `"1024x1024x1024"`, `"3x224x224_64"`).
    pub problem_desc: String,
}

impl ResultDb {
    /// Opens or creates the result database at the default location.
    ///
    /// The default path is `~/.cache/oxicuda/autotune/results.json`.
    /// Falls back to `$TMPDIR/oxicuda_autotune/results.json` if the
    /// home directory cannot be determined.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the directory cannot be created or
    /// the existing file cannot be read.
    pub fn open() -> Result<Self, std::io::Error> {
        let base =
            home_cache_dir().unwrap_or_else(|| std::env::temp_dir().join("oxicuda_autotune"));
        let dir = base.join("oxicuda").join("autotune");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("results.json");
        Self::open_at(path)
    }

    /// Opens or creates the result database at the specified path.
    ///
    /// If the file exists, it is read and deserialized.  If it does
    /// not exist, an empty database is created (the file is written
    /// on the first [`save`](Self::save)).
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file exists but cannot be read,
    /// or if the JSON is malformed.
    pub fn open_at(path: PathBuf) -> Result<Self, std::io::Error> {
        let mut data = if path.exists() {
            let contents = std::fs::read_to_string(&path)?;
            if contents.trim().is_empty() {
                ResultDbData::default()
            } else {
                serde_json::from_str(&contents)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
            }
        } else {
            ResultDbData::default()
        };

        // Defense in depth: a hand-edited or corrupted `results.json` could
        // otherwise hand a physically-impossible `BenchmarkResult` (e.g. a
        // zero `block_size`, which would divide-by-zero in
        // `Config::estimated_registers_per_thread`, or a negative timing)
        // straight to callers such as `Dispatcher`. Drop insane entries
        // instead of erroring, so a partially corrupt file degrades to
        // re-tuning rather than bricking the whole database.
        let dropped = sanitize_entries(&mut data);
        if dropped > 0 {
            tracing::warn!(
                dropped,
                path = %path.display(),
                "dropped insane benchmark result entries while loading result DB",
            );
        }

        Ok(Self { path, data })
    }

    /// Looks up the best result for an exact (GPU, kernel, problem) match.
    ///
    /// Returns `None` if no result is stored for the given triple.
    #[must_use]
    pub fn lookup(
        &self,
        gpu_name: &str,
        kernel_name: &str,
        problem_key: &str,
    ) -> Option<&BenchmarkResult> {
        self.data
            .entries
            .get(gpu_name)?
            .get(kernel_name)?
            .get(problem_key)
    }

    /// Finds the nearest matching problem when an exact match is unavailable.
    ///
    /// Attempts an exact match first.  If none is found, searches all
    /// problem keys for the same (GPU, kernel) pair and returns the one
    /// whose key is lexicographically closest to the query.  This is a
    /// simple heuristic — callers should verify suitability.
    ///
    /// Returns `None` if no results exist for the (GPU, kernel) pair.
    #[must_use]
    pub fn lookup_nearest(
        &self,
        gpu_name: &str,
        kernel_name: &str,
        problem_key: &str,
    ) -> Option<&BenchmarkResult> {
        // Tier 1: exact match
        if let Some(result) = self.lookup(gpu_name, kernel_name, problem_key) {
            return Some(result);
        }

        // Tier 2: nearest neighbor by edit distance on problem key
        let kernel_map = self.data.entries.get(gpu_name)?.get(kernel_name)?;
        if kernel_map.is_empty() {
            return None;
        }

        // Find the key with minimum edit distance to the query.
        kernel_map
            .iter()
            .min_by_key(|(k, _)| edit_distance(k, problem_key))
            .map(|(_, v)| v)
    }

    /// Saves a benchmark result for the given (GPU, kernel, problem) triple.
    ///
    /// If a result already exists for the same triple, it is replaced
    /// only if the new result has a lower median execution time.
    /// The database is flushed to disk after every save.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be written.
    pub fn save(
        &mut self,
        gpu_name: &str,
        kernel_name: &str,
        problem_key: &str,
        result: BenchmarkResult,
    ) -> Result<(), std::io::Error> {
        let kernel_map = self
            .data
            .entries
            .entry(gpu_name.to_string())
            .or_default()
            .entry(kernel_name.to_string())
            .or_default();

        // Only replace if new result is faster (or no prior result).
        let should_insert = kernel_map
            .get(problem_key)
            .is_none_or(|existing| result.median_us < existing.median_us);

        if should_insert {
            kernel_map.insert(problem_key.to_string(), result);
            self.flush()?;
        }

        Ok(())
    }

    /// Unconditionally saves a benchmark result, replacing whatever is
    /// currently stored for the given (GPU, kernel, problem) triple
    /// regardless of `median_us`.
    ///
    /// Unlike [`save`](Self::save), which only replaces an existing entry
    /// when the new result is faster, this method always writes `result`
    /// verbatim. It exists so callers that need to force-overwrite a
    /// result (e.g. an "always replace" import policy) do not have to
    /// fabricate a sentinel value such as `median_us = 0.0` to defeat the
    /// "only replace if faster" guard in `save`.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be written.
    pub fn save_unconditional(
        &mut self,
        gpu_name: &str,
        kernel_name: &str,
        problem_key: &str,
        result: BenchmarkResult,
    ) -> Result<(), std::io::Error> {
        self.data
            .entries
            .entry(gpu_name.to_string())
            .or_default()
            .entry(kernel_name.to_string())
            .or_default()
            .insert(problem_key.to_string(), result);
        // Flush with this key marked as "forced": the read-merge-write
        // cycle in `flush_impl` normally keeps whichever of the in-memory
        // or on-disk value has the lower `median_us` (correct for
        // `save`'s "only replace if faster" contract), but that would
        // silently undo an unconditional override if a slower value
        // happens to already be on disk. Marking the key as forced makes
        // this specific entry win the merge unconditionally.
        self.flush_impl(true, Some((gpu_name, kernel_name, problem_key)))
    }

    /// Lists all entries stored for a specific GPU.
    ///
    /// Returns a vector of `(kernel_name, problem_key, result)` tuples.
    #[must_use]
    pub fn list_gpu(&self, gpu_name: &str) -> Vec<(&str, &str, &BenchmarkResult)> {
        let mut entries = Vec::new();
        if let Some(gpu_map) = self.data.entries.get(gpu_name) {
            for (kernel, problems) in gpu_map {
                for (problem, result) in problems {
                    entries.push((kernel.as_str(), problem.as_str(), result));
                }
            }
        }
        entries
    }

    /// Returns all GPU names that have stored results.
    #[must_use]
    pub fn list_gpus(&self) -> Vec<&str> {
        self.data.entries.keys().map(String::as_str).collect()
    }

    /// Returns the total number of stored results across all GPUs.
    #[must_use]
    pub fn total_entries(&self) -> usize {
        self.data
            .entries
            .values()
            .flat_map(|k| k.values())
            .map(HashMap::len)
            .sum()
    }

    /// Clears all stored results and flushes an empty database to disk.
    ///
    /// This intentionally bypasses the read-merge-write cycle used by
    /// `flush` — `clear` means "wipe everything", so it
    /// must not resurrect entries a concurrent writer left on disk.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be written.
    pub fn clear(&mut self) -> Result<(), std::io::Error> {
        self.data = ResultDbData::default();
        self.flush_impl(false, None)
    }

    /// Returns the path to the backing JSON file.
    #[must_use]
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    /// Writes the in-memory database to disk as pretty-printed JSON,
    /// merging in any concurrent writer's results first.
    fn flush(&mut self) -> Result<(), std::io::Error> {
        self.flush_impl(true, None)
    }

    /// Writes the in-memory database to disk.
    ///
    /// Two correctness properties are provided:
    ///
    /// 1. **Atomic replace** — the JSON is written to a unique temporary
    ///    file in the same directory and then renamed over the target
    ///    path. A crash or concurrent read mid-write can therefore never
    ///    observe a torn or truncated file.
    /// 2. **Lost-update avoidance** — while holding an exclusive advisory
    ///    lock (a sibling `.lock` file), the on-disk database is re-read
    ///    and merged with the in-memory state (keeping the lower
    ///    `median_us` per `(gpu, kernel, problem)` key) before writing
    ///    back, so a concurrent writer's results are not silently
    ///    clobbered by a last-writer-wins overwrite. When `merge_with_disk`
    ///    is `false` (used by [`clear`](Self::clear)), this merge step is
    ///    skipped so the wipe is not immediately undone.
    ///
    /// `force_key`, if given, identifies a `(gpu, kernel, problem)` triple
    /// whose in-memory value must win the merge unconditionally — used by
    /// [`save_unconditional`](Self::save_unconditional) so an intentional
    /// override is never discarded just because the on-disk copy has a
    /// lower `median_us`.
    fn flush_impl(
        &mut self,
        merge_with_disk: bool,
        force_key: Option<(&str, &str, &str)>,
    ) -> Result<(), std::io::Error> {
        // Ensure parent directory exists.
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let _lock = FileLockGuard::acquire(self.lock_path())?;

        if merge_with_disk && self.path.exists() {
            let contents = std::fs::read_to_string(&self.path)?;
            if !contents.trim().is_empty() {
                let on_disk: ResultDbData = serde_json::from_str(&contents)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                merge_keep_best(&mut self.data, on_disk, force_key);
            }
        }

        let json = serde_json::to_string_pretty(&self.data).map_err(std::io::Error::other)?;

        let tmp_path = self.tmp_path();
        std::fs::write(&tmp_path, &json)?;
        std::fs::rename(&tmp_path, &self.path)?;

        Ok(())
    }

    /// Returns the path to the sibling advisory lock file, e.g.
    /// `results.json.lock`.
    fn lock_path(&self) -> PathBuf {
        let file_name = self
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("results.json");
        self.path.with_file_name(format!("{file_name}.lock"))
    }

    /// Returns a unique sibling path for the atomic-replace temp file,
    /// e.g. `results.json.tmp.<pid>.<counter>`.
    fn tmp_path(&self) -> PathBuf {
        static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let file_name = self
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("results.json");
        self.path
            .with_file_name(format!("{file_name}.tmp.{}.{unique}", std::process::id()))
    }
}

/// RAII guard for a sibling `.lock` file, used as a simple, portable
/// advisory lock around the read-merge-write cycle in [`ResultDb::flush_impl`].
///
/// A dedicated lock file (rather than [`std::fs::File::lock`]) is used
/// because this workspace's `rust-version` is 1.85, while `File::lock`
/// was only stabilized in Rust 1.89.
struct FileLockGuard {
    lock_path: PathBuf,
}

impl FileLockGuard {
    /// Maximum time to wait for the lock before giving up.
    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
    /// Delay between lock-acquisition retries.
    const RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(10);

    /// Acquires the lock, retrying until [`Self::TIMEOUT`] elapses.
    fn acquire(lock_path: PathBuf) -> Result<Self, std::io::Error> {
        let start = std::time::Instant::now();
        loop {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
            {
                Ok(_file) => return Ok(Self { lock_path }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if start.elapsed() > Self::TIMEOUT {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            format!(
                                "timed out waiting for result-db lock file {}",
                                lock_path.display()
                            ),
                        ));
                    }
                    std::thread::sleep(Self::RETRY_DELAY);
                }
                Err(e) => return Err(e),
            }
        }
    }
}

impl Drop for FileLockGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.lock_path);
    }
}

/// Merges `other` into `data`, keeping the faster (lower `median_us`)
/// result whenever both sides have an entry for the same
/// `(gpu, kernel, problem)` key. Entries that exist only in `other` are
/// added to `data`. This implements the read-merge-write cycle so that
/// a concurrent writer's results are not lost to a last-writer-wins
/// overwrite.
///
/// `force_key`, if given, names a `(gpu, kernel, problem)` triple that is
/// exempt from the "keep the faster one" rule: `data`'s value for that
/// key is left untouched regardless of what `other` contains. This is
/// used by [`ResultDb::save_unconditional`] to guarantee an intentional
/// override always wins the merge.
fn merge_keep_best(
    data: &mut ResultDbData,
    other: ResultDbData,
    force_key: Option<(&str, &str, &str)>,
) {
    for (gpu, other_kernel_map) in other.entries {
        let dst_kernel_map = data.entries.entry(gpu.clone()).or_default();
        for (kernel, other_problem_map) in other_kernel_map {
            let dst_problem_map = dst_kernel_map.entry(kernel.clone()).or_default();
            for (problem, other_result) in other_problem_map {
                if let Some((force_gpu, force_kernel, force_problem)) = force_key {
                    if force_gpu == gpu && force_kernel == kernel && force_problem == problem {
                        // This key was just unconditionally set by this
                        // instance; the in-memory value is authoritative.
                        continue;
                    }
                }
                let take_other = dst_problem_map
                    .get(&problem)
                    .is_none_or(|existing| other_result.median_us < existing.median_us);
                if take_other {
                    dst_problem_map.insert(problem, other_result);
                }
            }
        }
    }
}

/// Returns `true` if a loaded benchmark result contains only sane,
/// physically plausible values.
///
/// Guards against a corrupted or hand-edited `results.json` poisoning
/// the database with degenerate entries — e.g. a zero `block_size`
/// would divide-by-zero in [`crate::config::Config::estimated_registers_per_thread`]
/// if served verbatim to a caller such as `Dispatcher`.
fn entry_is_sane(r: &BenchmarkResult) -> bool {
    r.median_us.is_finite()
        && r.min_us.is_finite()
        && r.max_us.is_finite()
        && r.stddev_us.is_finite()
        && r.median_us >= 0.0
        && r.min_us >= 0.0
        && r.max_us >= 0.0
        && r.stddev_us >= 0.0
        && r.min_us <= r.max_us
        && r.gflops.is_none_or(f64::is_finite)
        && (1..=1024).contains(&r.config.block_size)
        && r.config.tile_m > 0
        && r.config.tile_n > 0
        && r.config.tile_k > 0
}

/// Drops entries that fail [`entry_is_sane`] from `data`, in place, and
/// prunes now-empty kernel/GPU maps so [`ResultDb::list_gpus`] and
/// [`ResultDb::total_entries`] stay consistent. Returns the number of
/// entries removed.
fn sanitize_entries(data: &mut ResultDbData) -> usize {
    let mut dropped = 0usize;
    data.entries.retain(|_, kernel_map| {
        kernel_map.retain(|_, problem_map| {
            let before = problem_map.len();
            problem_map.retain(|_, result| entry_is_sane(result));
            dropped += before - problem_map.len();
            !problem_map.is_empty()
        });
        !kernel_map.is_empty()
    });
    dropped
}

/// Returns `~/.cache` on Unix systems, or `None` if `$HOME` is not set.
fn home_cache_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache"))
}

/// Computes the Levenshtein edit distance between two strings.
///
/// Used as a simple similarity metric for nearest-neighbor lookup
/// when an exact problem key match is unavailable.
fn edit_distance(a: &str, b: &str) -> usize {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let m = a_bytes.len();
    let n = b_bytes.len();

    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_bytes[i - 1] == b_bytes[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn make_result(median_us: f64) -> BenchmarkResult {
        BenchmarkResult {
            config: Config::new(),
            median_us,
            min_us: median_us - 1.0,
            max_us: median_us + 1.0,
            stddev_us: 0.5,
            gflops: None,
            efficiency: None,
        }
    }

    #[test]
    fn open_creates_empty_db() {
        let dir = std::env::temp_dir().join("oxicuda_test_open_creates");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("results.json");

        let db = ResultDb::open_at(path).expect("open_at");
        assert_eq!(db.total_entries(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_and_lookup_roundtrip() {
        let dir = std::env::temp_dir().join("oxicuda_test_save_lookup");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("results.json");

        let mut db = ResultDb::open_at(path.clone()).expect("open");
        db.save("GPU0", "sgemm", "1024x1024", make_result(42.0))
            .expect("save");

        assert_eq!(db.total_entries(), 1);
        let r = db.lookup("GPU0", "sgemm", "1024x1024");
        assert!(r.is_some());
        assert!((r.map_or(0.0, |r| r.median_us) - 42.0).abs() < 1e-9);

        // Re-open from disk and verify persistence.
        let db2 = ResultDb::open_at(path).expect("reopen");
        let r2 = db2.lookup("GPU0", "sgemm", "1024x1024");
        assert!(r2.is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_only_replaces_if_faster() {
        let dir = std::env::temp_dir().join("oxicuda_test_replace_if_faster");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("results.json");

        let mut db = ResultDb::open_at(path).expect("open");
        db.save("GPU0", "sgemm", "256x256", make_result(50.0))
            .expect("save1");

        // Slower result should NOT replace.
        db.save("GPU0", "sgemm", "256x256", make_result(60.0))
            .expect("save2");
        let r = db.lookup("GPU0", "sgemm", "256x256");
        assert!((r.map_or(0.0, |r| r.median_us) - 50.0).abs() < 1e-9);

        // Faster result should replace.
        db.save("GPU0", "sgemm", "256x256", make_result(30.0))
            .expect("save3");
        let r = db.lookup("GPU0", "sgemm", "256x256");
        assert!((r.map_or(0.0, |r| r.median_us) - 30.0).abs() < 1e-9);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lookup_nearest_finds_close_match() {
        let dir = std::env::temp_dir().join("oxicuda_test_nearest");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("results.json");

        let mut db = ResultDb::open_at(path).expect("open");
        db.save("GPU0", "sgemm", "1024x1024x1024", make_result(42.0))
            .expect("save");

        // Exact miss, nearest should find the stored entry.
        let r = db.lookup_nearest("GPU0", "sgemm", "1024x1024x512");
        assert!(r.is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clear_removes_all() {
        let dir = std::env::temp_dir().join("oxicuda_test_clear");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("results.json");

        let mut db = ResultDb::open_at(path).expect("open");
        db.save("GPU0", "sgemm", "1024x1024", make_result(42.0))
            .expect("save");
        assert_eq!(db.total_entries(), 1);

        db.clear().expect("clear");
        assert_eq!(db.total_entries(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_gpu_and_list_gpus() {
        let dir = std::env::temp_dir().join("oxicuda_test_list");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create dir");
        let path = dir.join("results.json");

        let mut db = ResultDb::open_at(path).expect("open");
        db.save("GPU0", "sgemm", "1024", make_result(10.0))
            .expect("s1");
        db.save("GPU0", "dgemm", "512", make_result(20.0))
            .expect("s2");
        db.save("GPU1", "sgemm", "2048", make_result(30.0))
            .expect("s3");

        let gpus = db.list_gpus();
        assert_eq!(gpus.len(), 2);

        let gpu0_entries = db.list_gpu("GPU0");
        assert_eq!(gpu0_entries.len(), 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn edit_distance_cases() {
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", "abc"), 0);
        assert_eq!(edit_distance("1024x1024", "1024x512"), 3);
    }

    // -----------------------------------------------------------------------
    // Round-trip persistence tests
    // -----------------------------------------------------------------------

    /// Compares two `BenchmarkResult` values field-by-field.
    ///
    /// `BenchmarkResult` does not implement `PartialEq`, so this helper
    /// performs an epsilon-tolerant comparison of all numeric fields.
    fn results_equal(a: &BenchmarkResult, b: &BenchmarkResult) -> bool {
        a.config == b.config
            && (a.median_us - b.median_us).abs() < 1e-9
            && (a.min_us - b.min_us).abs() < 1e-9
            && (a.max_us - b.max_us).abs() < 1e-9
            && (a.stddev_us - b.stddev_us).abs() < 1e-9
            && match (a.gflops, b.gflops) {
                (Some(x), Some(y)) => (x - y).abs() < 1e-9,
                (None, None) => true,
                _ => false,
            }
            && match (a.efficiency, b.efficiency) {
                (Some(x), Some(y)) => (x - y).abs() < 1e-9,
                (None, None) => true,
                _ => false,
            }
    }

    /// RAII cleanup guard: removes a directory on drop.
    struct CleanupGuard(std::path::PathBuf);

    impl Drop for CleanupGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Creates an isolated temp directory for a test and returns a cleanup
    /// guard that removes it automatically when the guard is dropped.
    fn tmp_dir(suffix: &str) -> (std::path::PathBuf, CleanupGuard) {
        let dir = std::env::temp_dir().join(format!("oxicuda_rtrip_{suffix}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let guard = CleanupGuard(dir.clone());
        (dir, guard)
    }

    // -----------------------------------------------------------------------

    /// Save one result, flush to disk, reload from the same path, verify
    /// the result is byte-for-byte equivalent.
    #[test]
    fn test_result_db_round_trip() {
        let (dir, _guard) = tmp_dir("round_trip");
        let path = dir.join("results.json");

        let result = BenchmarkResult {
            config: Config::new()
                .with_tile_m(64)
                .with_tile_n(128)
                .with_stages(3),
            median_us: 55.5,
            min_us: 50.0,
            max_us: 60.0,
            stddev_us: 2.5,
            gflops: Some(1800.0),
            efficiency: Some(0.75),
        };

        // Save and flush.
        let mut db = ResultDb::open_at(path.clone()).expect("open");
        db.save("A100", "sgemm", "2048x2048x2048", result.clone())
            .expect("save");

        // Reload from disk.
        let db2 = ResultDb::open_at(path).expect("reopen");
        let loaded = db2
            .lookup("A100", "sgemm", "2048x2048x2048")
            .expect("entry must be present after reload");

        assert!(
            results_equal(&result, loaded),
            "round-trip mismatch: original={result:?}, loaded={loaded:?}",
        );
    }

    /// Save results for multiple (GPU, kernel, problem) triples, reload,
    /// and verify all entries are restored correctly.
    #[test]
    fn test_result_db_multiple_results_round_trip() {
        let (dir, _guard) = tmp_dir("multi_round_trip");
        let path = dir.join("results.json");

        let entries = [
            ("RTX4090", "sgemm", "1024x1024x1024", 42.0_f64, None),
            ("RTX4090", "sgemm", "512x512x512", 12.0, Some(900.0)),
            ("RTX4090", "dgemm", "2048x2048x2048", 200.0, None),
            ("A100", "sgemm", "4096x4096x4096", 100.0, Some(3000.0)),
            ("A100", "conv2d", "64x3x224x224", 30.0, None),
        ];

        let mut db = ResultDb::open_at(path.clone()).expect("open");
        for (gpu, kernel, problem, med, gflops) in &entries {
            let r = BenchmarkResult {
                config: Config::new(),
                median_us: *med,
                min_us: med - 2.0,
                max_us: med + 2.0,
                stddev_us: 1.0,
                gflops: *gflops,
                efficiency: None,
            };
            db.save(gpu, kernel, problem, r).expect("save");
        }

        // Verify in-memory
        assert_eq!(db.total_entries(), entries.len());

        // Reload and verify each entry
        let db2 = ResultDb::open_at(path).expect("reopen");
        assert_eq!(
            db2.total_entries(),
            entries.len(),
            "total entries mismatch after reload"
        );

        for (gpu, kernel, problem, med, gflops) in &entries {
            let loaded = db2
                .lookup(gpu, kernel, problem)
                .unwrap_or_else(|| panic!("missing entry: {gpu}/{kernel}/{problem}"));

            assert!(
                (loaded.median_us - med).abs() < 1e-9,
                "median mismatch for {gpu}/{kernel}/{problem}: expected {med}, got {}",
                loaded.median_us,
            );
            match (loaded.gflops, gflops) {
                (Some(x), Some(y)) => assert!(
                    (x - y).abs() < 1e-9,
                    "gflops mismatch for {gpu}/{kernel}/{problem}",
                ),
                (None, None) => {}
                _ => panic!("gflops presence mismatch for {gpu}/{kernel}/{problem}"),
            }
        }
    }

    /// Save exactly three configs with distinct configs, reload, verify all
    /// three survive with byte-identical field values.
    ///
    /// This validates the JSON export schema: the file is written as
    /// human-readable pretty-printed JSON and must survive a
    /// serialize → write → read → deserialize cycle without data loss.
    #[test]
    fn test_result_db_three_configs_json_round_trip() {
        let (dir, _guard) = tmp_dir("three_json");
        let path = dir.join("results.json");

        let make = |tile_m: u32, med: f64, gflops: Option<f64>| BenchmarkResult {
            config: Config::new()
                .with_tile_m(tile_m)
                .with_tile_n(64)
                .with_tile_k(16)
                .with_stages(2)
                .with_block_size(128),
            median_us: med,
            min_us: med - 5.0,
            max_us: med + 5.0,
            stddev_us: 1.0,
            gflops,
            efficiency: None,
        };

        let r1 = make(64, 100.0, None);
        let r2 = make(128, 80.0, Some(1500.0));
        let r3 = make(256, 60.0, Some(2000.0));

        // Save all three to the same (gpu, kernel) namespace.
        let mut db = ResultDb::open_at(path.clone()).expect("open");
        db.save("RTX3090", "sgemm", "key_a", r1.clone())
            .expect("save r1");
        db.save("RTX3090", "sgemm", "key_b", r2.clone())
            .expect("save r2");
        db.save("RTX3090", "sgemm", "key_c", r3.clone())
            .expect("save r3");
        assert_eq!(db.total_entries(), 3);

        // Reload from disk.
        let db2 = ResultDb::open_at(path.clone()).expect("reopen");
        assert_eq!(
            db2.total_entries(),
            3,
            "all three entries must survive the JSON round-trip"
        );

        // Verify each config is identical to what was saved.
        for (key, original) in [("key_a", &r1), ("key_b", &r2), ("key_c", &r3)] {
            let loaded = db2
                .lookup("RTX3090", "sgemm", key)
                .unwrap_or_else(|| panic!("entry {key} missing after reload"));
            assert!(
                results_equal(original, loaded),
                "round-trip mismatch for {key}: \
                 original={original:?}, loaded={loaded:?}"
            );
        }

        // Verify the JSON file actually exists on disk.
        assert!(
            path.exists(),
            "JSON file must be written to disk at {:?}",
            path
        );
    }

    /// An empty ResultDb serializes and deserializes back to an empty db.
    #[test]
    fn test_result_db_empty_round_trip() {
        let (dir, _guard) = tmp_dir("empty_round_trip");
        let path = dir.join("results.json");

        // Write an empty db to disk.
        let db = ResultDb::open_at(path.clone()).expect("open");
        assert_eq!(db.total_entries(), 0);

        // Flush an empty file (the file does not exist yet — flush only on
        // save, so we trigger a save + clear cycle instead).
        let mut db2 = ResultDb::open_at(path.clone()).expect("reopen1");
        db2.save("GPU", "k", "p", make_result(1.0)).expect("save");
        db2.clear().expect("clear");
        assert_eq!(db2.total_entries(), 0);

        // Reload — must still be empty.
        let db3 = ResultDb::open_at(path).expect("reopen2");
        assert_eq!(db3.total_entries(), 0, "reloaded db should be empty");
        assert!(db3.list_gpus().is_empty(), "no GPUs expected in empty db");
    }
}
