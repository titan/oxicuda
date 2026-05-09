//! Session handle for `oxicuda-nerf`.
//!
//! Provides `LcgRng` (Knuth MMIX LCG), `SmVersion`, and `NerfHandle`.

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

/// Minimal LCG random number generator using Knuth MMIX multiplier.
///
/// `state = state * 6364136223846793005 + 1442695040888963407 (mod 2⁶⁴)`
#[derive(Debug, Clone)]
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    /// Create a new LCG RNG with the given seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1_442_695_040_888_963_407),
        }
    }

    /// Advance one step and return a mixed `u32`.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 33) ^ self.state) as u32
    }

    /// Return a `f32` uniformly in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    /// Return a `usize` uniformly drawn from `[0, n)`.
    #[inline]
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }

    /// Return a `f32` uniformly in `[lo, hi)`.
    #[inline]
    pub fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Sample two independent N(0, 1) values via Box-Muller transform.
    pub fn next_normal_pair(&mut self) -> (f32, f32) {
        let u1 = (self.next_f32() + 1e-10).min(1.0 - 1e-10);
        let u2 = self.next_f32();
        let r = (-2.0_f32 * u1.ln()).sqrt();
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
}

// ─── NerfHandle ──────────────────────────────────────────────────────────────

/// Lightweight session descriptor for NeRF / neural rendering operations.
#[derive(Debug)]
pub struct NerfHandle {
    /// SM version for PTX generation.
    pub sm: SmVersion,
    /// CUDA device ordinal.
    pub device: u32,
    /// Deterministic RNG for CPU-side operations.
    pub rng: LcgRng,
}

impl NerfHandle {
    /// Create a new handle.
    #[must_use]
    pub fn new(device: u32, sm: SmVersion, seed: u64) -> Self {
        Self {
            sm,
            device,
            rng: LcgRng::new(seed),
        }
    }

    /// Convenience constructor for unit-test / CPU environments
    /// (device 0, SM 8.0, seed 42).
    #[must_use]
    pub fn default_handle() -> Self {
        Self::new(0, SmVersion(80), 42)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_rng_deterministic() {
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
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn nerf_handle_default() {
        let h = NerfHandle::default_handle();
        assert_eq!(h.device, 0);
        assert_eq!(h.sm, SmVersion(80));
    }

    #[test]
    fn sm_version_target_str() {
        assert_eq!(SmVersion(80).target_str(), "sm_80");
        assert_eq!(SmVersion(90).target_str(), "sm_90");
        assert_eq!(SmVersion(120).target_str(), "sm_120");
    }
}
