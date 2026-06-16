//! Roofline analytical performance model (Williams, Waterman & Patterson, 2009).
//!
//! The roofline model bounds the attainable performance of a kernel by two
//! ceilings:
//!
//! 1. **Compute ceiling** — the peak floating-point throughput of the device
//!    (`peak_flops`, in FLOP/s).  No kernel can exceed this regardless of how
//!    arithmetically intense it is.
//!
//! 2. **Memory ceiling(s)** — for each level of the memory hierarchy (e.g.
//!    L1, L2, DRAM) a bandwidth `bw` (in byte/s).  At a given *arithmetic
//!    intensity* `I` (FLOPs performed per byte moved through that level), the
//!    most FLOP/s sustainable by that level is `I * bw` — a line of slope `bw`
//!    through the origin in the (intensity, performance) plane.
//!
//! The attainable performance for a single memory ceiling is
//!
//! ```text
//! P(I) = min(peak_flops, I * bw)
//! ```
//!
//! When several memory ceilings apply, the *lowest* roof binds, so
//!
//! ```text
//! P(I) = min(peak_flops, min_level(I * bw_level))
//!      = min(peak_flops, I * min_level(bw_level))   (since I >= 0)
//! ```
//!
//! The two regimes meet at the **ridge point**
//!
//! ```text
//! I* = peak_flops / bw
//! ```
//!
//! Kernels with `I < I*` are *memory-bound* (the diagonal binds); kernels with
//! `I > I*` are *compute-bound* (the flat roof binds).  The ridge point is the
//! minimum intensity required to reach peak compute.
//!
//! The estimated wall-clock runtime for a kernel performing `total_flops`
//! floating-point operations is
//!
//! ```text
//! runtime = total_flops / P(I)
//! ```
//!
//! # Example
//!
//! ```rust
//! use oxicuda_autotune::cost_model::roofline::{Roofline, Bound};
//!
//! // 10 TFLOP/s peak, single 1 TB/s DRAM ceiling.
//! let model = Roofline::new(10.0e12, vec![1.0e12]).expect("valid roofline");
//!
//! // A kernel at I = 2 FLOP/byte is memory-bound (ridge point is 10 FLOP/byte).
//! let p = model.attainable(2.0);
//! assert!((p - 2.0e12).abs() < 1.0); // I * bw = 2 * 1 TB/s
//! assert_eq!(model.classify(2.0).bound, Bound::MemoryBound);
//! ```

use crate::error::AutotuneError;

/// Which ceiling limits a kernel's performance at a given arithmetic intensity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    /// The kernel is limited by memory bandwidth (it lies left of the ridge
    /// point; the sloped roof binds).
    MemoryBound,
    /// The kernel is limited by compute throughput (it lies right of the ridge
    /// point; the flat roof binds).
    ComputeBound,
}

/// A named memory-bandwidth ceiling (a level of the memory hierarchy).
#[derive(Debug, Clone, PartialEq)]
pub struct BandwidthCeiling {
    /// Human-readable label, e.g. `"L1"`, `"L2"`, `"DRAM"`.
    pub name: String,
    /// Sustainable bandwidth in bytes per second (must be strictly positive).
    pub bandwidth: f64,
}

impl BandwidthCeiling {
    /// Creates a new named bandwidth ceiling.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::BenchmarkFailed`] if `bandwidth` is not a
    /// finite, strictly-positive value.
    pub fn new(name: impl Into<String>, bandwidth: f64) -> Result<Self, AutotuneError> {
        if !bandwidth.is_finite() || bandwidth <= 0.0 {
            return Err(AutotuneError::BenchmarkFailed(format!(
                "bandwidth ceiling must be finite and positive, got {bandwidth}"
            )));
        }
        Ok(Self {
            name: name.into(),
            bandwidth,
        })
    }
}

/// The result of classifying a kernel of a given arithmetic intensity against
/// a [`Roofline`] model.
#[derive(Debug, Clone)]
pub struct RooflineClassification {
    /// Whether the kernel is memory- or compute-bound.
    pub bound: Bound,
    /// The attainable performance `P(I)` in FLOP/s.
    pub attainable_flops: f64,
    /// The name of the limiting resource: `"compute"` when compute-bound, or
    /// the name of the binding memory ceiling when memory-bound.
    pub limiting_resource: String,
    /// Fraction of peak compute attained, in `[0, 1]`.
    pub compute_utilization: f64,
}

/// A roofline analytical cost model with one compute ceiling and one or more
/// memory-bandwidth ceilings.
#[derive(Debug, Clone)]
pub struct Roofline {
    /// Peak compute throughput in FLOP/s (strictly positive).
    peak_flops: f64,
    /// Memory-bandwidth ceilings (at least one, each strictly positive).
    ceilings: Vec<BandwidthCeiling>,
}

impl Roofline {
    /// Builds a roofline model from a peak compute rate and a list of
    /// bandwidth ceilings (in byte/s).
    ///
    /// Each entry is given an anonymous level name `"mem{idx}"`.  Use
    /// [`Roofline::with_ceilings`] to supply named ceilings.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::BenchmarkFailed`] if `peak_flops` is not finite
    /// and positive, if `bandwidths` is empty, or if any bandwidth is not
    /// finite and positive.
    pub fn new(peak_flops: f64, bandwidths: Vec<f64>) -> Result<Self, AutotuneError> {
        let ceilings = bandwidths
            .into_iter()
            .enumerate()
            .map(|(idx, bw)| BandwidthCeiling::new(format!("mem{idx}"), bw))
            .collect::<Result<Vec<_>, _>>()?;
        Self::with_ceilings(peak_flops, ceilings)
    }

    /// Builds a roofline model from a peak compute rate and named ceilings.
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::BenchmarkFailed`] if `peak_flops` is not finite
    /// and positive, or if `ceilings` is empty.
    pub fn with_ceilings(
        peak_flops: f64,
        ceilings: Vec<BandwidthCeiling>,
    ) -> Result<Self, AutotuneError> {
        if !peak_flops.is_finite() || peak_flops <= 0.0 {
            return Err(AutotuneError::BenchmarkFailed(format!(
                "peak compute must be finite and positive, got {peak_flops}"
            )));
        }
        if ceilings.is_empty() {
            return Err(AutotuneError::BenchmarkFailed(
                "roofline requires at least one bandwidth ceiling".to_string(),
            ));
        }
        Ok(Self {
            peak_flops,
            ceilings,
        })
    }

    /// Returns the peak compute throughput in FLOP/s.
    #[must_use]
    pub fn peak_flops(&self) -> f64 {
        self.peak_flops
    }

    /// Returns the configured bandwidth ceilings.
    #[must_use]
    pub fn ceilings(&self) -> &[BandwidthCeiling] {
        &self.ceilings
    }

    /// Returns the binding (lowest-bandwidth) ceiling.  Because all bandwidths
    /// are strictly positive there is always at least one, so this never fails
    /// for a validly-constructed model.
    fn binding_ceiling(&self) -> &BandwidthCeiling {
        // `self.ceilings` is guaranteed non-empty by construction; fold over it
        // selecting the minimum bandwidth so we never index or unwrap.
        let mut iter = self.ceilings.iter();
        let first = match iter.next() {
            Some(c) => c,
            // Unreachable for a valid model, but handled without panicking.
            None => &self.ceilings[0],
        };
        iter.fold(
            first,
            |acc, c| {
                if c.bandwidth < acc.bandwidth { c } else { acc }
            },
        )
    }

    /// The effective (lowest) memory bandwidth across all ceilings.
    #[must_use]
    pub fn effective_bandwidth(&self) -> f64 {
        self.binding_ceiling().bandwidth
    }

    /// The ridge point `I* = peak_flops / bw`, the arithmetic intensity at which
    /// the memory and compute roofs meet, using the binding (lowest) bandwidth.
    ///
    /// Kernels below this intensity are memory-bound; above it, compute-bound.
    #[must_use]
    pub fn ridge_point(&self) -> f64 {
        self.peak_flops / self.effective_bandwidth()
    }

    /// The ridge point for a specific named ceiling, or `None` if no ceiling
    /// with that name exists.
    #[must_use]
    pub fn ridge_point_for(&self, name: &str) -> Option<f64> {
        self.ceilings
            .iter()
            .find(|c| c.name == name)
            .map(|c| self.peak_flops / c.bandwidth)
    }

    /// Attainable performance `P(I) = min(peak_flops, I * bw_min)` in FLOP/s,
    /// where `bw_min` is the lowest of the configured bandwidth ceilings.
    ///
    /// Negative or non-finite intensities are clamped to `0`, yielding `0`.
    #[must_use]
    pub fn attainable(&self, intensity: f64) -> f64 {
        let i = if intensity.is_finite() && intensity > 0.0 {
            intensity
        } else {
            0.0
        };
        let memory_roof = i * self.effective_bandwidth();
        memory_roof.min(self.peak_flops)
    }

    /// Classifies a kernel of arithmetic intensity `intensity`, reporting the
    /// regime, the attainable performance, and the limiting resource.
    ///
    /// The kernel is compute-bound when its intensity is at least the ridge
    /// point (the flat roof binds), and memory-bound otherwise.  At exactly the
    /// ridge point both roofs coincide; by convention this is reported as
    /// compute-bound (peak compute is reached).
    #[must_use]
    pub fn classify(&self, intensity: f64) -> RooflineClassification {
        let attainable = self.attainable(intensity);
        let binding = self.binding_ceiling();
        let i = if intensity.is_finite() && intensity > 0.0 {
            intensity
        } else {
            0.0
        };
        let ridge = self.ridge_point();
        let (bound, limiting_resource) = if i >= ridge {
            (Bound::ComputeBound, "compute".to_string())
        } else {
            (Bound::MemoryBound, binding.name.clone())
        };
        let compute_utilization = (attainable / self.peak_flops).clamp(0.0, 1.0);
        RooflineClassification {
            bound,
            attainable_flops: attainable,
            limiting_resource,
            compute_utilization,
        }
    }

    /// Estimates the wall-clock runtime (in seconds) of a kernel that performs
    /// `total_flops` floating-point operations at arithmetic intensity
    /// `intensity`:
    ///
    /// ```text
    /// runtime = total_flops / P(I)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`AutotuneError::BenchmarkFailed`] if `total_flops` is negative
    /// or non-finite, or if `P(I)` is zero (which happens only for a
    /// non-positive intensity).
    pub fn estimated_runtime(
        &self,
        total_flops: f64,
        intensity: f64,
    ) -> Result<f64, AutotuneError> {
        if !total_flops.is_finite() || total_flops < 0.0 {
            return Err(AutotuneError::BenchmarkFailed(format!(
                "total FLOPs must be finite and non-negative, got {total_flops}"
            )));
        }
        let p = self.attainable(intensity);
        if p <= 0.0 {
            return Err(AutotuneError::BenchmarkFailed(format!(
                "attainable performance is zero at intensity {intensity}; cannot estimate runtime"
            )));
        }
        Ok(total_flops / p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 10 TFLOP/s peak, single 1 TB/s ceiling -> ridge point I* = 10 FLOP/byte.
    fn single_ceiling_model() -> Roofline {
        Roofline::new(10.0e12, vec![1.0e12]).expect("valid model")
    }

    #[test]
    fn rejects_nonpositive_peak() {
        assert!(Roofline::new(0.0, vec![1.0e12]).is_err());
        assert!(Roofline::new(-1.0, vec![1.0e12]).is_err());
        assert!(Roofline::new(f64::NAN, vec![1.0e12]).is_err());
    }

    #[test]
    fn rejects_empty_or_bad_ceilings() {
        assert!(Roofline::new(1.0e12, vec![]).is_err());
        assert!(Roofline::new(1.0e12, vec![0.0]).is_err());
        assert!(Roofline::new(1.0e12, vec![-1.0]).is_err());
        assert!(BandwidthCeiling::new("x", f64::INFINITY).is_err());
    }

    // (a) Low intensity (I << I*) -> P == I * bw exactly, classified MemoryBound.
    #[test]
    fn low_intensity_is_memory_bound_and_linear() {
        let model = single_ceiling_model();
        let intensity = 0.5; // I* is 10, so this is deep in the memory-bound region.
        let p = model.attainable(intensity);
        let expected = intensity * 1.0e12; // I * bw
        assert!(
            (p - expected).abs() < 1.0,
            "P should equal I*bw exactly: got {p}, expected {expected}"
        );
        let cls = model.classify(intensity);
        assert_eq!(cls.bound, Bound::MemoryBound);
        assert_eq!(cls.limiting_resource, "mem0");
        assert!((cls.attainable_flops - expected).abs() < 1.0);
    }

    // (b) High intensity (I >> I*) -> P == peak_flops, classified ComputeBound.
    #[test]
    fn high_intensity_is_compute_bound_and_capped() {
        let model = single_ceiling_model();
        let intensity = 1000.0; // far past the ridge point of 10.
        let p = model.attainable(intensity);
        assert!(
            (p - model.peak_flops()).abs() < 1.0,
            "P should equal peak_flops: got {p}, peak {}",
            model.peak_flops()
        );
        let cls = model.classify(intensity);
        assert_eq!(cls.bound, Bound::ComputeBound);
        assert_eq!(cls.limiting_resource, "compute");
        assert!((cls.compute_utilization - 1.0).abs() < 1e-12);
    }

    // (c) Ridge point I* == peak/bw; at I == I* both roofs give equal P.
    #[test]
    fn ridge_point_value_and_roof_equality() {
        let model = single_ceiling_model();
        let ridge = model.ridge_point();
        assert!(
            (ridge - 10.0).abs() < 1e-9,
            "ridge should be peak/bw = 10, got {ridge}"
        );
        // At the ridge point, memory roof (I*bw) == compute roof (peak).
        let memory_roof = ridge * model.effective_bandwidth();
        let compute_roof = model.peak_flops();
        assert!(
            (memory_roof - compute_roof).abs() < 1.0,
            "roofs should be equal at ridge: mem={memory_roof}, compute={compute_roof}"
        );
        // attainable() at the ridge equals peak.
        assert!((model.attainable(ridge) - compute_roof).abs() < 1.0);
    }

    // (d) runtime == total_flops / P(I); doubling FLOPs at fixed I doubles runtime.
    #[test]
    fn runtime_matches_formula_and_scales_with_flops() {
        let model = single_ceiling_model();
        let intensity = 2.0; // memory-bound; P = 2e12.
        let total = 4.0e12;
        let p = model.attainable(intensity);
        let rt = model
            .estimated_runtime(total, intensity)
            .expect("runtime ok");
        assert!(
            (rt - total / p).abs() < 1e-6,
            "runtime should be total/P: got {rt}, expected {}",
            total / p
        );
        let rt2 = model
            .estimated_runtime(2.0 * total, intensity)
            .expect("runtime ok");
        assert!(
            (rt2 - 2.0 * rt).abs() < 1e-6,
            "doubling FLOPs should double runtime: rt={rt}, rt2={rt2}"
        );
    }

    #[test]
    fn runtime_rejects_bad_inputs() {
        let model = single_ceiling_model();
        assert!(model.estimated_runtime(-1.0, 2.0).is_err());
        assert!(model.estimated_runtime(f64::NAN, 2.0).is_err());
        // Non-positive intensity => P == 0 => error.
        assert!(model.estimated_runtime(1.0e12, 0.0).is_err());
    }

    // (e) P(I) is monotone non-decreasing in I and capped at peak_flops.
    #[test]
    fn attainable_is_monotone_and_capped() {
        let model = single_ceiling_model();
        let mut prev = model.attainable(0.0);
        let mut i = 0.0_f64;
        while i <= 50.0 {
            let p = model.attainable(i);
            assert!(
                p >= prev - 1e-3,
                "P must be non-decreasing: P({i})={p} < prev={prev}"
            );
            assert!(
                p <= model.peak_flops() + 1.0,
                "P must never exceed peak: P({i})={p}"
            );
            prev = p;
            i += 0.25;
        }
        // Well past the ridge it stays pinned at peak.
        assert!((model.attainable(10_000.0) - model.peak_flops()).abs() < 1.0);
    }

    // (f) With two ceilings, the effective roof is the lower one in the
    //     memory-bound region.
    #[test]
    fn two_ceilings_lower_roof_binds() {
        // Fast L2 (4 TB/s) and slow DRAM (1 TB/s). DRAM must bind.
        let model = Roofline::with_ceilings(
            10.0e12,
            vec![
                BandwidthCeiling::new("L2", 4.0e12).expect("ok"),
                BandwidthCeiling::new("DRAM", 1.0e12).expect("ok"),
            ],
        )
        .expect("valid model");

        assert!((model.effective_bandwidth() - 1.0e12).abs() < 1.0);
        // Memory-bound region: P should follow the DRAM (lower) roof, not L2.
        let intensity = 2.0;
        let p = model.attainable(intensity);
        let dram_roof = intensity * 1.0e12;
        let l2_roof = intensity * 4.0e12;
        assert!(
            (p - dram_roof).abs() < 1.0,
            "lower (DRAM) roof should bind: P={p}, dram={dram_roof}"
        );
        assert!(p < l2_roof, "P must be below the L2 roof");

        let cls = model.classify(intensity);
        assert_eq!(cls.bound, Bound::MemoryBound);
        assert_eq!(
            cls.limiting_resource, "DRAM",
            "the binding ceiling should be reported as DRAM"
        );

        // Ridge point uses the binding bandwidth: 10e12 / 1e12 = 10.
        assert!((model.ridge_point() - 10.0).abs() < 1e-9);
        // Per-level ridge points differ.
        assert!((model.ridge_point_for("L2").expect("ok") - 2.5).abs() < 1e-9);
        assert!((model.ridge_point_for("DRAM").expect("ok") - 10.0).abs() < 1e-9);
        assert!(model.ridge_point_for("missing").is_none());
    }
}
