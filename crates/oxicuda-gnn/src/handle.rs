//! Session handle for `oxicuda-gnn`.
//!
//! `GnnHandle` stores the compute device index, SM version (used by PTX
//! kernel generators to emit the correct `.target` directive), and a
//! deterministic LCG random number generator for sampling operations.

// ─── SmVersion ───────────────────────────────────────────────────────────────

/// SM (Streaming Multiprocessor) version encoded as `major*10 + minor`.
///
/// Examples: 80 = SM 8.0 (Ampere A100), 90 = SM 9.0 (Hopper H100),
/// 120 = SM 12.0 (Blackwell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmVersion(pub u32);

impl SmVersion {
    /// PTX `.version` directive string for this SM.
    ///
    /// PTX ISA 8.7 covers SM 12.x, 8.4 covers SM 9.x,
    /// 8.0 covers SM 8.x, 7.5 covers SM 7.x.
    pub fn ptx_version_str(self) -> &'static str {
        if self.0 >= 100 {
            "8.7"
        } else if self.0 >= 90 {
            "8.4"
        } else if self.0 >= 80 {
            "8.0"
        } else {
            "7.5"
        }
    }

    /// PTX `.target` string for this SM (e.g., `"sm_80"`).
    pub fn target_str(self) -> String {
        format!("sm_{}", self.0)
    }
}

impl std::fmt::Display for SmVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SM {}.{}", self.0 / 10, self.0 % 10)
    }
}

// ─── LcgRng ──────────────────────────────────────────────────────────────────

/// Fast, deterministic Linear Congruential Generator.
///
/// Uses the Knuth/MMIX multiplier:
/// `x_{n+1} = 6364136223846793005 * x_n + 1442695040888963407 (mod 2⁶⁴)`.
#[derive(Debug, Clone)]
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    /// Create a new LCG with the given seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Advance one step and return a `u32` in `[0, 2³¹)`.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 33) as u32
    }

    /// Return a `f32` in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }

    /// Return a `usize` in `[0, n)`.
    ///
    /// Requires `n > 0`.
    #[inline]
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }

    /// Return two independent `N(0, 1)` samples via the Box-Muller transform.
    ///
    /// Both returned values are standard-normal distributed and mutually
    /// independent.  The first uniform draw is clamped away from `0` so the
    /// logarithm stays finite.
    #[inline]
    pub fn next_normal_pair(&mut self) -> (f32, f32) {
        let u1 = (self.next_f32() + 1e-10).min(1.0 - 1e-10);
        let u2 = self.next_f32();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }

    /// Shuffle a slice in-place using Fisher-Yates.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = self.next_usize(i + 1);
            slice.swap(i, j);
        }
    }
}

// ─── GnnHandle ───────────────────────────────────────────────────────────────

/// Lightweight session descriptor for GNN operations.
///
/// A `GnnHandle` does **not** open a CUDA context; it merely records which
/// device and SM version are targeted so that PTX kernel generators can
/// emit architecture-appropriate code, and holds a seeded RNG for
/// deterministic graph sampling.
#[derive(Debug, Clone)]
pub struct GnnHandle {
    sm: SmVersion,
    device: u32,
    rng: LcgRng,
}

impl GnnHandle {
    /// Create a new handle for the given device, SM version, and RNG seed.
    #[must_use]
    pub fn new(device: u32, sm: SmVersion, seed: u64) -> Self {
        Self {
            sm,
            device,
            rng: LcgRng::new(seed),
        }
    }

    /// Convenience constructor: device 0, SM 8.0 (Ampere), seed 42.
    #[must_use]
    pub fn default_handle() -> Self {
        Self::new(0, SmVersion(80), 42)
    }

    /// SM version targeted by this handle.
    #[must_use]
    #[inline]
    pub fn sm_version(&self) -> SmVersion {
        self.sm
    }

    /// Device ordinal.
    #[must_use]
    #[inline]
    pub fn device(&self) -> u32 {
        self.device
    }

    /// Mutable access to the internal RNG.
    #[inline]
    pub fn rng_mut(&mut self) -> &mut LcgRng {
        &mut self.rng
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sm_version_ptx_strings() {
        assert_eq!(SmVersion(75).ptx_version_str(), "7.5");
        assert_eq!(SmVersion(80).ptx_version_str(), "8.0");
        assert_eq!(SmVersion(86).ptx_version_str(), "8.0");
        assert_eq!(SmVersion(90).ptx_version_str(), "8.4");
        assert_eq!(SmVersion(100).ptx_version_str(), "8.7");
        assert_eq!(SmVersion(120).ptx_version_str(), "8.7");
    }

    #[test]
    fn sm_version_target_str() {
        assert_eq!(SmVersion(80).target_str(), "sm_80");
        assert_eq!(SmVersion(90).target_str(), "sm_90");
        assert_eq!(SmVersion(120).target_str(), "sm_120");
    }

    #[test]
    fn sm_version_display() {
        assert_eq!(SmVersion(80).to_string(), "SM 8.0");
        assert_eq!(SmVersion(90).to_string(), "SM 9.0");
        assert_eq!(SmVersion(120).to_string(), "SM 12.0");
    }

    #[test]
    fn sm_version_ordering() {
        assert!(SmVersion(75) < SmVersion(80));
        assert!(SmVersion(80) < SmVersion(90));
        assert!(SmVersion(100) > SmVersion(90));
        assert_eq!(SmVersion(80), SmVersion(80));
    }

    #[test]
    fn gnn_handle_default() {
        let h = GnnHandle::default_handle();
        assert_eq!(h.device(), 0);
        assert_eq!(h.sm_version(), SmVersion(80));
    }

    #[test]
    fn gnn_handle_custom() {
        let h = GnnHandle::new(2, SmVersion(90), 123);
        assert_eq!(h.device(), 2);
        assert_eq!(h.sm_version(), SmVersion(90));
    }

    #[test]
    fn lcg_rng_deterministic() {
        let mut r1 = LcgRng::new(42);
        let mut r2 = LcgRng::new(42);
        for _ in 0..20 {
            assert_eq!(r1.next_u32(), r2.next_u32());
        }
    }

    #[test]
    fn lcg_rng_f32_range() {
        let mut rng = LcgRng::new(99);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn lcg_rng_usize_range() {
        let mut rng = LcgRng::new(7);
        for _ in 0..1000 {
            let v = rng.next_usize(10);
            assert!(v < 10);
        }
    }

    #[test]
    fn lcg_rng_shuffle_permutation() {
        let mut rng = LcgRng::new(5);
        let mut v: Vec<usize> = (0..8).collect();
        let original = v.clone();
        rng.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, original);
    }

    #[test]
    fn lcg_rng_normal_pair_finite() {
        let mut rng = LcgRng::new(13);
        for _ in 0..1000 {
            let (a, b) = rng.next_normal_pair();
            assert!(a.is_finite());
            assert!(b.is_finite());
        }
    }

    #[test]
    fn lcg_rng_normal_pair_spans_both_signs() {
        // Box-Muller output should produce both positive and negative samples
        // and stay within a sane magnitude window.
        let mut rng = LcgRng::new(2024);
        let mut saw_positive = false;
        let mut saw_negative = false;
        for _ in 0..5000 {
            let (a, _) = rng.next_normal_pair();
            if a > 0.0 {
                saw_positive = true;
            }
            if a < 0.0 {
                saw_negative = true;
            }
            assert!(a.abs() < 12.0, "unreasonable magnitude: {a}");
        }
        assert!(saw_positive && saw_negative);
    }

    #[test]
    fn lcg_rng_normal_pair_deterministic() {
        let mut r1 = LcgRng::new(321);
        let mut r2 = LcgRng::new(321);
        for _ in 0..50 {
            assert_eq!(r1.next_normal_pair(), r2.next_normal_pair());
        }
    }

    #[test]
    fn gnn_handle_rng_mut() {
        let mut h = GnnHandle::default_handle();
        let v1 = h.rng_mut().next_u32();
        let v2 = h.rng_mut().next_u32();
        // Two successive calls should differ
        assert_ne!(v1, v2);
    }
}
