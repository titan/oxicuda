//! Session handle for `oxicuda-vision`.
//!
//! `VisionHandle` stores the compute device index, the GPU SM version,
//! and a deterministic LCG random number generator for CPU-side operations
//! (buffer initialisation, test scaffolding).

// ─── SmVersion ───────────────────────────────────────────────────────────────

/// SM (Streaming Multiprocessor) version encoded as `major*10 + minor`.
///
/// Examples: 80 = SM 8.0 (Ampere A100), 90 = SM 9.0 (Hopper H100),
/// 120 = SM 12.0 (Blackwell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmVersion(pub u32);

impl SmVersion {
    /// Return the raw u32 version number.
    #[must_use]
    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }

    /// PTX `.version` directive string for this SM.
    ///
    /// PTX ISA 8.7 covers SM 12.x, 8.4 covers SM 9.x,
    /// 8.0 covers SM 8.x, 7.5 covers SM 7.x.
    #[must_use]
    pub fn ptx_version_str(self) -> &'static str {
        match self.0 {
            v if v >= 100 => "8.7",
            v if v >= 90 => "8.4",
            v if v >= 80 => "8.0",
            _ => "7.5",
        }
    }

    /// PTX `.target` string for this SM (e.g., `"sm_80"`).
    #[must_use]
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

/// Minimal LCG random number generator for deterministic CPU-side sampling
/// in tests and buffer initialisation.
///
/// Uses the Knuth MMIX 64-bit LCG multiplier:
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

    /// Advance one step and return a `u32` drawn from the high 32 bits.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 33) as u32
    }

    /// Return a `f32` uniformly distributed in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }

    /// Return a `usize` uniformly drawn from `[0, n)`.
    ///
    /// Returns 0 if `n == 0`.
    #[inline]
    pub fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u32() as usize) % n
    }

    /// Sample two independent N(0, 1) values via Box-Muller transform.
    pub fn next_normal_pair(&mut self) -> (f32, f32) {
        let u1 = (self.next_f32() + 1e-10).min(1.0 - 1e-10);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }

    /// Fill `buf` with N(0, 1) samples via Box-Muller.
    pub fn fill_normal(&mut self, buf: &mut [f32]) {
        let mut i = 0;
        while i + 1 < buf.len() {
            let (a, b) = self.next_normal_pair();
            buf[i] = a;
            buf[i + 1] = b;
            i += 2;
        }
        if i < buf.len() {
            let (a, _) = self.next_normal_pair();
            buf[i] = a;
        }
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

// ─── VisionHandle ────────────────────────────────────────────────────────────

/// Lightweight session descriptor for vision model operations.
///
/// A `VisionHandle` does **not** open a CUDA context; it merely records which
/// device and SM version are targeted so that PTX kernel generators can
/// emit architecture-appropriate code, and carries an LCG RNG for
/// deterministic CPU-side sampling and parameter initialisation.
#[derive(Debug, Clone)]
pub struct VisionHandle {
    sm: SmVersion,
    rng: LcgRng,
    device: u32,
}

impl VisionHandle {
    /// Create a new handle for the given device, SM version, and RNG seed.
    #[must_use]
    pub fn new(device: u32, sm: SmVersion, seed: u64) -> Self {
        Self {
            sm,
            rng: LcgRng::new(seed),
            device,
        }
    }

    /// Convenience constructor for unit-test / CPU environments
    /// (device 0, SM 8.0, seed 42).
    #[must_use]
    pub fn default_handle() -> Self {
        Self::new(0, SmVersion(80), 42)
    }

    /// Return the SM version.
    #[must_use]
    pub fn sm_version(&self) -> SmVersion {
        self.sm
    }

    /// Return the device ordinal.
    #[must_use]
    pub fn device(&self) -> u32 {
        self.device
    }

    /// Return a mutable reference to the internal RNG.
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
    }

    #[test]
    fn sm_version_ordering() {
        assert!(SmVersion(80) < SmVersion(90));
        assert!(SmVersion(100) > SmVersion(90));
        assert_eq!(SmVersion(80), SmVersion(80));
    }

    #[test]
    fn sm_version_as_u32() {
        assert_eq!(SmVersion(86).as_u32(), 86);
    }

    #[test]
    fn vision_handle_default() {
        let h = VisionHandle::default_handle();
        assert_eq!(h.device(), 0);
        assert_eq!(h.sm_version(), SmVersion(80));
    }

    #[test]
    fn vision_handle_custom() {
        let h = VisionHandle::new(2, SmVersion(90), 12345);
        assert_eq!(h.device(), 2);
        assert_eq!(h.sm_version(), SmVersion(90));
    }

    #[test]
    fn lcg_rng_determinism() {
        let mut a = LcgRng::new(42);
        let mut b = LcgRng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn lcg_rng_f32_in_range() {
        let mut rng = LcgRng::new(7);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn lcg_rng_usize_in_range() {
        let mut rng = LcgRng::new(99);
        for _ in 0..1000 {
            let v = rng.next_usize(7);
            assert!(v < 7, "out of range: {v}");
        }
    }

    #[test]
    fn lcg_rng_normal_fill_finite() {
        let mut rng = LcgRng::new(13);
        let mut buf = vec![0.0_f32; 64];
        rng.fill_normal(&mut buf);
        assert!(buf.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn lcg_rng_shuffle_permutes() {
        let mut rng = LcgRng::new(77);
        let mut v: Vec<usize> = (0..8).collect();
        rng.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn vision_handle_rng_mut() {
        let mut h = VisionHandle::default_handle();
        let v = h.rng_mut().next_f32();
        assert!((0.0..1.0).contains(&v));
    }
}
