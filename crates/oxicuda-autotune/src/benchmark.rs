//! GPU benchmark engine for measuring kernel execution time.
//!
//! The [`BenchmarkEngine`] uses CUDA events for precise GPU-side timing.
//! It performs warmup iterations to stabilize GPU clock frequencies and
//! caches, then collects multiple timed samples and computes robust
//! statistics (median, min, max, standard deviation).
//!
//! # Typical usage
//!
//! ```rust,no_run
//! use oxicuda_autotune::{BenchmarkEngine, BenchmarkConfig, WarmupStrategy, Config};
//! use oxicuda_driver::{Stream, Event};
//!
//! # fn example(stream: &Stream) -> Result<(), oxicuda_autotune::AutotuneError> {
//! let engine = BenchmarkEngine::with_config(BenchmarkConfig {
//!     warmup: WarmupStrategy::Fixed(3),
//!     benchmark_runs: 10,
//! });
//!
//! let config = Config::new();
//! let result = engine.benchmark(&stream, &config, Some(2.0e9), |s| {
//!     // Launch your kernel on stream `s` here.
//!     Ok(())
//! })?;
//!
//! println!("Median: {:.1} us, GFLOPS: {:.1}",
//!     result.median_us,
//!     result.gflops.unwrap_or(0.0));
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

use oxicuda_driver::{Event, Stream};
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::error::AutotuneError;

// ---------------------------------------------------------------------------
// WarmupStrategy
// ---------------------------------------------------------------------------

/// Controls how many warmup iterations the benchmark engine performs before
/// collecting timed samples.
///
/// # Variants
///
/// - [`WarmupStrategy::Fixed`] — Run exactly `n` warmup iterations.  This is
///   the default (`Fixed(5)`) and matches the original behaviour.
/// - [`WarmupStrategy::Adaptive`] — Run until two consecutive timing samples
///   agree within `tolerance` (relative delta), subject to `[min, max]`
///   iteration bounds.  Emits a `tracing::warn!` if the cap is reached
///   without convergence.
#[derive(Debug, Clone, PartialEq)]
pub enum WarmupStrategy {
    /// Run exactly `n` warmup iterations.
    Fixed(usize),
    /// Run until consecutive timing samples converge.
    Adaptive {
        /// Minimum iterations regardless of convergence.
        min_iterations: usize,
        /// Maximum iterations before giving up and emitting a warning.
        max_iterations: usize,
        /// Relative-delta convergence threshold (e.g. `0.05` = 5%).
        tolerance: f64,
    },
}

impl Default for WarmupStrategy {
    fn default() -> Self {
        WarmupStrategy::Fixed(5)
    }
}

/// Returns `true` when two consecutive warmup timing samples are considered
/// converged, i.e. the relative difference between them is less than
/// `tolerance`.
///
/// Returns `false` when `prev` is zero (avoids divide-by-zero).
pub(crate) fn warmup_converged(prev: Duration, curr: Duration, tolerance: f64) -> bool {
    let p = prev.as_secs_f64();
    if p <= 0.0 {
        return false;
    }
    let rel = (curr.as_secs_f64() - p).abs() / p;
    rel < tolerance
}

/// Result of benchmarking a single configuration.
///
/// All times are in microseconds.  GFLOPS and efficiency are
/// populated only when the caller provides a FLOP count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// The configuration that was benchmarked.
    pub config: Config,
    /// Median execution time in microseconds.
    pub median_us: f64,
    /// Minimum execution time in microseconds.
    pub min_us: f64,
    /// Maximum execution time in microseconds.
    pub max_us: f64,
    /// Standard deviation of execution times in microseconds.
    pub stddev_us: f64,
    /// Achieved GFLOPS (billions of floating-point operations per second).
    ///
    /// Populated only when the caller provides a FLOP count.
    pub gflops: Option<f64>,
    /// Efficiency vs. peak throughput (0.0–1.0).
    ///
    /// Populated only when the caller provides a FLOP count and the
    /// peak throughput is known.
    pub efficiency: Option<f64>,
}

/// Configuration for the benchmark engine.
///
/// Controls how many warmup and measurement iterations are performed.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Warmup strategy before measurement begins.
    ///
    /// Warmup stabilizes GPU clock frequencies and populates caches.
    /// Default: [`WarmupStrategy::Fixed`]`(5)`.
    pub warmup: WarmupStrategy,
    /// Number of timed measurement iterations.
    ///
    /// More iterations produce more stable statistics at the cost of
    /// longer tuning time.  Default: 20.
    pub benchmark_runs: u32,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup: WarmupStrategy::default(),
            benchmark_runs: 20,
        }
    }
}

/// GPU benchmark execution engine.
///
/// Measures kernel execution time using CUDA events for accurate
/// GPU-side timing (not wall-clock time).  The engine:
///
/// 1. Runs the kernel several times for warmup (not measured).
/// 2. Records CUDA events around each measurement iteration.
/// 3. Reads back elapsed times and computes statistics.
///
/// The `launch_fn` closure is responsible for enqueuing the kernel
/// onto the provided [`Stream`].  It will be called at least
/// `benchmark_runs` times in addition to the warmup invocations.
pub struct BenchmarkEngine {
    /// Benchmark configuration (warmup + measurement counts).
    config: BenchmarkConfig,
}

impl BenchmarkEngine {
    /// Creates a new benchmark engine with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
        }
    }

    /// Creates a new benchmark engine with the given configuration.
    #[must_use]
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Returns the current benchmark configuration.
    #[must_use]
    pub fn config(&self) -> &BenchmarkConfig {
        &self.config
    }

    /// Benchmarks a kernel launch function and returns timing statistics.
    ///
    /// # Arguments
    ///
    /// * `stream` — The CUDA stream on which the kernel will be launched.
    /// * `config` — The tuning configuration being evaluated.
    /// * `flops` — Optional total floating-point operation count for
    ///   computing GFLOPS.
    /// * `launch_fn` — A closure that launches the kernel onto the
    ///   provided stream.  It must **not** synchronize the stream itself.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::Cuda`] if any CUDA operation fails, or
    /// [`AutotuneError::BenchmarkFailed`] if no valid timing samples
    /// could be collected.
    pub fn benchmark<F>(
        &self,
        stream: &Stream,
        config: &Config,
        flops: Option<f64>,
        launch_fn: F,
    ) -> Result<BenchmarkResult, AutotuneError>
    where
        F: Fn(&Stream) -> Result<(), oxicuda_driver::CudaError>,
    {
        // Phase 1: Warmup — run the kernel without timing.
        match &self.config.warmup {
            WarmupStrategy::Fixed(n) => {
                for _ in 0..*n {
                    launch_fn(stream)?;
                }
                stream.synchronize()?;
            }
            WarmupStrategy::Adaptive {
                min_iterations,
                max_iterations,
                tolerance,
            } => {
                let mut prev: Option<Duration> = None;
                let mut converged = false;
                for i in 0..*max_iterations {
                    let start_ev = Event::new()?;
                    let end_ev = Event::new()?;
                    start_ev.record(stream)?;
                    launch_fn(stream)?;
                    end_ev.record(stream)?;
                    end_ev.synchronize()?;
                    let elapsed_ms = Event::elapsed_time(&start_ev, &end_ev)?;
                    let t = Duration::from_secs_f64(f64::from(elapsed_ms) * 1e-3);
                    if i + 1 >= *min_iterations {
                        if let Some(p) = prev {
                            if warmup_converged(p, t, *tolerance) {
                                converged = true;
                                break;
                            }
                        }
                    }
                    prev = Some(t);
                }
                if !converged {
                    tracing::warn!(
                        "adaptive warmup did not converge within {} iterations (tolerance={})",
                        max_iterations,
                        tolerance
                    );
                }
            }
        }

        // Phase 2: Timed measurement — record events around each launch.
        let num_runs = self.config.benchmark_runs;
        if num_runs == 0 {
            return Err(AutotuneError::BenchmarkFailed(
                "benchmark_runs must be > 0".to_string(),
            ));
        }

        let mut times_us = Vec::with_capacity(num_runs as usize);

        for i in 0..num_runs {
            let start_event = Event::new()?;
            let end_event = Event::new()?;

            start_event.record(stream)?;
            launch_fn(stream).map_err(|e| {
                AutotuneError::BenchmarkFailed(format!("launch failed on iteration {i}: {e}"))
            })?;
            end_event.record(stream)?;
            end_event.synchronize()?;

            let elapsed_ms = Event::elapsed_time(&start_event, &end_event)?;
            times_us.push(f64::from(elapsed_ms) * 1000.0);
        }

        if times_us.is_empty() {
            return Err(AutotuneError::BenchmarkFailed(
                "no timing samples collected".to_string(),
            ));
        }

        let (median, min, max, stddev) = compute_stats(&times_us);

        // Compute GFLOPS if FLOP count is provided.
        let gflops = flops.map(|f| f / (median * 1e-6) / 1e9);

        Ok(BenchmarkResult {
            config: config.clone(),
            median_us: median,
            min_us: min,
            max_us: max,
            stddev_us: stddev,
            gflops,
            efficiency: None, // Requires peak throughput knowledge
        })
    }

    /// Benchmarks a kernel using wall-clock timing (no CUDA events).
    ///
    /// This is useful when CUDA events are not available or when
    /// benchmarking host-side overhead.  Less precise than event-based
    /// timing but works without a GPU.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::BenchmarkFailed`] if no valid samples
    /// could be collected.
    pub fn benchmark_wallclock<F>(
        &self,
        config: &Config,
        flops: Option<f64>,
        run_fn: F,
    ) -> Result<BenchmarkResult, AutotuneError>
    where
        F: Fn() -> Result<(), AutotuneError>,
    {
        // Phase 1: Warmup
        match &self.config.warmup {
            WarmupStrategy::Fixed(n) => {
                for _ in 0..*n {
                    run_fn()?;
                }
            }
            WarmupStrategy::Adaptive {
                min_iterations,
                max_iterations,
                tolerance,
            } => {
                let mut prev: Option<Duration> = None;
                let mut converged = false;
                for i in 0..*max_iterations {
                    let start = std::time::Instant::now();
                    run_fn()?;
                    let t = start.elapsed();
                    if i + 1 >= *min_iterations {
                        if let Some(p) = prev {
                            if warmup_converged(p, t, *tolerance) {
                                converged = true;
                                break;
                            }
                        }
                    }
                    prev = Some(t);
                }
                if !converged {
                    tracing::warn!(
                        "adaptive warmup did not converge within {} iterations (tolerance={})",
                        max_iterations,
                        tolerance
                    );
                }
            }
        }

        // Phase 2: Timed measurement
        let num_runs = self.config.benchmark_runs;
        if num_runs == 0 {
            return Err(AutotuneError::BenchmarkFailed(
                "benchmark_runs must be > 0".to_string(),
            ));
        }

        let mut times_us = Vec::with_capacity(num_runs as usize);

        for _ in 0..num_runs {
            let start = std::time::Instant::now();
            run_fn()?;
            let elapsed = start.elapsed();
            times_us.push(elapsed.as_secs_f64() * 1_000_000.0);
        }

        if times_us.is_empty() {
            return Err(AutotuneError::BenchmarkFailed(
                "no timing samples collected".to_string(),
            ));
        }

        let (median, min, max, stddev) = compute_stats(&times_us);
        let gflops = flops.map(|f| f / (median * 1e-6) / 1e9);

        Ok(BenchmarkResult {
            config: config.clone(),
            median_us: median,
            min_us: min,
            max_us: max,
            stddev_us: stddev,
            gflops,
            efficiency: None,
        })
    }
}

impl Default for BenchmarkEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes median, minimum, maximum, and standard deviation from a
/// slice of timing samples.
///
/// The input slice must be non-empty.  Values are assumed to be in
/// microseconds but the function is unit-agnostic.
fn compute_stats(times: &[f64]) -> (f64, f64, f64, f64) {
    debug_assert!(!times.is_empty(), "compute_stats called with empty slice");

    let n = times.len() as f64;

    // Sort for median
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if sorted.len() % 2 == 0 {
        let mid = sorted.len() / 2;
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    let min = sorted.first().copied().unwrap_or(0.0);
    let max = sorted.last().copied().unwrap_or(0.0);

    // Standard deviation (population)
    let mean = times.iter().sum::<f64>() / n;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();

    (median, min, max, stddev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_stats_odd_count() {
        let times = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let (median, min, max, stddev) = compute_stats(&times);
        assert!((median - 30.0).abs() < 1e-9);
        assert!((min - 10.0).abs() < 1e-9);
        assert!((max - 50.0).abs() < 1e-9);
        // stddev of [10,20,30,40,50] = sqrt(200) ≈ 14.142
        assert!((stddev - 14.142_135_623_730_951).abs() < 1e-6);
    }

    #[test]
    fn compute_stats_even_count() {
        let times = vec![10.0, 20.0, 30.0, 40.0];
        let (median, min, max, _) = compute_stats(&times);
        assert!((median - 25.0).abs() < 1e-9);
        assert!((min - 10.0).abs() < 1e-9);
        assert!((max - 40.0).abs() < 1e-9);
    }

    #[test]
    fn compute_stats_single_value() {
        let times = vec![42.0];
        let (median, min, max, stddev) = compute_stats(&times);
        assert!((median - 42.0).abs() < 1e-9);
        assert!((min - 42.0).abs() < 1e-9);
        assert!((max - 42.0).abs() < 1e-9);
        assert!((stddev - 0.0).abs() < 1e-9);
    }

    #[test]
    fn benchmark_wallclock_smoke() {
        let engine = BenchmarkEngine::with_config(BenchmarkConfig {
            warmup: WarmupStrategy::Fixed(1),
            benchmark_runs: 3,
        });
        let cfg = Config::new();
        let result = engine
            .benchmark_wallclock(&cfg, Some(1e9), || Ok(()))
            .expect("wallclock benchmark should succeed");

        assert!(result.median_us >= 0.0);
        assert!(result.gflops.is_some());
    }

    #[test]
    fn benchmark_zero_runs_errors() {
        let engine = BenchmarkEngine::with_config(BenchmarkConfig {
            warmup: WarmupStrategy::Fixed(0),
            benchmark_runs: 0,
        });
        let cfg = Config::new();
        let result = engine.benchmark_wallclock(&cfg, None, || Ok(()));
        assert!(result.is_err());
    }

    #[test]
    fn test_variance_calculation_correctness() {
        // Identical values → stddev = 0, coefficient of variation = 0%
        let all_same = vec![1.0f64, 1.0, 1.0];
        let (median, _min, _max, stddev) = compute_stats(&all_same);
        assert!((median - 1.0).abs() < 1e-12);
        assert!(
            stddev.abs() < 1e-12,
            "stddev should be zero for identical values"
        );

        // [1.0, 1.1, 0.9]: mean=1.0, variance=((0)^2+(0.1)^2+(-0.1)^2)/3 = 0.02/3
        // stddev = sqrt(0.02/3) ≈ 0.08165
        let varying = vec![1.0f64, 1.1, 0.9];
        let (med2, _min2, _max2, stddev2) = compute_stats(&varying);
        assert!((med2 - 1.0).abs() < 1e-12);
        let expected_stddev = (0.02f64 / 3.0).sqrt();
        assert!(
            (stddev2 - expected_stddev).abs() < 1e-9,
            "stddev {stddev2} should ≈ {expected_stddev}"
        );

        // Five identical values of 10.0 → CV = 0%
        let uniform = vec![10.0f64, 10.0, 10.0, 10.0, 10.0];
        let (_med3, _min3, _max3, stddev3) = compute_stats(&uniform);
        assert!(stddev3.abs() < 1e-12, "CV should be 0% for uniform values");
    }

    #[test]
    fn test_gflops_formula() {
        // For M=N=K=1024: ops = 2 * 1024^3 = 2_147_483_648
        // If median_us = 1000 (1ms) → GFLOPS = 2_147_483_648 / 1000e-6 / 1e9 = 2147.48...
        let m: f64 = 1024.0;
        let n: f64 = 1024.0;
        let k: f64 = 1024.0;
        let flops: f64 = 2.0 * m * n * k;
        let median_us: f64 = 1000.0;

        // Replicate the formula used in BenchmarkEngine
        let gflops = flops / (median_us * 1e-6) / 1e9;
        assert!(
            (gflops - 2_147.483_648).abs() < 0.001,
            "gflops {gflops} should ≈ 2147.48"
        );

        // Verify via benchmark_wallclock with a mock that takes ~0 time.
        // We verify gflops is populated when flops is supplied.
        let engine = BenchmarkEngine::with_config(BenchmarkConfig {
            warmup: WarmupStrategy::Fixed(0),
            benchmark_runs: 5,
        });
        let cfg = Config::new();
        let result = engine
            .benchmark_wallclock(&cfg, Some(flops), || Ok(()))
            .expect("wallclock benchmark should succeed");
        assert!(
            result.gflops.is_some(),
            "gflops should be Some when flops is provided"
        );
        // The no-op run should produce some positive GFLOPS number
        assert!(
            result.gflops.unwrap_or(0.0) > 0.0,
            "gflops should be positive"
        );
    }

    // -----------------------------------------------------------------------
    // WarmupStrategy unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn warmup_strategy_default_is_fixed_5() {
        assert_eq!(WarmupStrategy::default(), WarmupStrategy::Fixed(5));
    }

    #[test]
    fn warmup_converged_pure_fn_stable() {
        let p = Duration::from_millis(10);
        let c = Duration::from_millis(10);
        assert!(
            warmup_converged(p, c, 0.05),
            "identical samples should converge"
        );
    }

    #[test]
    fn warmup_converged_pure_fn_noisy() {
        let p = Duration::from_millis(10);
        let c = Duration::from_millis(15); // 50% delta
        assert!(
            !warmup_converged(p, c, 0.05),
            "50% delta should not converge at 5% tolerance"
        );
    }

    #[test]
    fn warmup_converged_pure_fn_zero_prev() {
        let p = Duration::ZERO;
        let c = Duration::from_millis(10);
        assert!(
            !warmup_converged(p, c, 0.05),
            "zero prev should not converge (avoid div-by-zero)"
        );
    }

    #[test]
    fn warmup_strategy_fixed_runs_exact_count() {
        // Verify Fixed(7) runs exactly 7 warmup iterations + 1 measurement = 8 total.
        use std::sync::{Arc, Mutex};

        let total_calls = Arc::new(Mutex::new(0usize));
        let total_calls_c = Arc::clone(&total_calls);

        let engine = BenchmarkEngine::with_config(BenchmarkConfig {
            warmup: WarmupStrategy::Fixed(7),
            benchmark_runs: 1,
        });
        let cfg = Config::new();
        let _result = engine
            .benchmark_wallclock(&cfg, None, || {
                let mut g = total_calls_c.lock().expect("lock");
                *g += 1;
                Ok(())
            })
            .expect("benchmark should succeed");

        // Total calls = warmup(7) + measurement(1) = 8
        let total = *total_calls.lock().expect("lock");
        assert_eq!(
            total, 8,
            "Fixed(7) + 1 measurement run must produce 8 total calls"
        );
    }

    #[test]
    fn warmup_strategy_adaptive_converges_on_stable_input() {
        // All iterations return ~10ms — should converge right after min_iterations.
        let engine = BenchmarkEngine::with_config(BenchmarkConfig {
            warmup: WarmupStrategy::Adaptive {
                min_iterations: 2,
                max_iterations: 20,
                tolerance: 0.05,
            },
            benchmark_runs: 1,
        });
        let cfg = Config::new();
        // Use benchmark_wallclock; closure does effectively nothing so elapsed
        // will be near-zero and stable.  Convergence should happen quickly.
        let result = engine.benchmark_wallclock(&cfg, None, || Ok(()));
        assert!(result.is_ok(), "adaptive convergence should succeed");
    }

    #[test]
    fn warmup_strategy_adaptive_caps_at_max() {
        // Use the pure warmup_converged fn to verify cap logic: alternating
        // durations 10ms / 20ms never converge at 5% tolerance.
        let samples: Vec<Duration> = (0..6)
            .map(|i| {
                if i % 2 == 0 {
                    Duration::from_millis(10)
                } else {
                    Duration::from_millis(20)
                }
            })
            .collect();

        let tolerance = 0.05f64;
        let min_iterations = 2usize;
        let max_iterations = 5usize;

        let mut prev: Option<Duration> = None;
        let mut converged = false;
        let mut iters = 0usize;

        for (i, t) in samples.iter().copied().enumerate().take(max_iterations) {
            iters += 1;
            if i + 1 >= min_iterations {
                if let Some(p) = prev {
                    if warmup_converged(p, t, tolerance) {
                        converged = true;
                        break;
                    }
                }
            }
            prev = Some(t);
        }

        assert!(
            !converged,
            "alternating 10ms/20ms should not converge at 5% tolerance"
        );
        assert_eq!(
            iters, max_iterations,
            "should run exactly max_iterations iterations"
        );
    }

    #[test]
    fn warmup_strategy_adaptive_respects_min() {
        // Stable samples — would converge at i=1 if min were 1.
        // With min=3 it must not converge before iteration 3.
        let sample = Duration::from_millis(10);
        let tolerance = 0.05f64;
        let min_iterations = 3usize;
        let max_iterations = 20usize;

        let mut prev: Option<Duration> = None;
        let mut converged_at: Option<usize> = None;

        for i in 0..max_iterations {
            let t = sample;
            if i + 1 >= min_iterations {
                if let Some(p) = prev {
                    if warmup_converged(p, t, tolerance) {
                        converged_at = Some(i + 1);
                        break;
                    }
                }
            }
            prev = Some(t);
        }

        assert!(
            converged_at.is_some(),
            "stable input should converge eventually"
        );
        assert!(
            converged_at.unwrap_or(0) >= min_iterations,
            "convergence must not happen before min_iterations=3"
        );
    }

    #[test]
    fn warmup_strategy_adaptive_wallclock_integration() {
        // Verify the adaptive path in benchmark_wallclock runs without error.
        let engine = BenchmarkEngine::with_config(BenchmarkConfig {
            warmup: WarmupStrategy::Adaptive {
                min_iterations: 2,
                max_iterations: 10,
                tolerance: 0.10,
            },
            benchmark_runs: 3,
        });
        let cfg = Config::new();
        let result = engine.benchmark_wallclock(&cfg, Some(1e9), || Ok(()));
        assert!(
            result.is_ok(),
            "adaptive wallclock benchmark must not error"
        );
        let result = result.expect("already checked");
        assert!(result.median_us >= 0.0);
        assert!(result.gflops.is_some());
    }
}
