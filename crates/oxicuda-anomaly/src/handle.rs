//! Session handle for `oxicuda-anomaly`.
//!
//! Provides `LcgRng` (Knuth MMIX LCG + Box-Muller) and `AnomalyHandle`
//! for deterministic CPU-side operations.

// ─── SmVersion ───────────────────────────────────────────────────────────────

/// GPU SM (Streaming Multiprocessor) version as a raw `u32` (major*10 + minor).
///
/// Examples: 75 = SM 7.5 (Turing), 80 = SM 8.0 (Ampere), 90 = SM 9.0 (Hopper),
/// 120 = SM 12.0 (Blackwell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmVersion(pub u32);

impl SmVersion {
    /// Raw numeric version.
    #[must_use]
    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }

    /// PTX `.version` string for this SM.
    #[must_use]
    pub fn ptx_version_str(self) -> &'static str {
        match self.0 {
            v if v >= 100 => "8.7",
            v if v >= 90 => "8.4",
            v if v >= 80 => "8.0",
            _ => "7.5",
        }
    }

    /// PTX `.target` string (e.g., `"sm_80"`).
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

/// Knuth MMIX 64-bit Linear Congruential Generator with Box-Muller normal sampling.
///
/// `x_{n+1} = 6364136223846793005 * x_n + 1442695040888963407 (mod 2⁶⁴)`
#[derive(Debug, Clone)]
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    /// Create a new generator from `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1_442_695_040_888_963_407),
        }
    }

    /// Advance one step; return a `u32` from the mixed state.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 33) ^ self.state) as u32
    }

    /// `f32` uniform in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    /// `usize` uniform in `[0, n)`.
    #[inline]
    pub fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u32() as usize) % n
    }

    /// Single N(0, 1) sample via Box-Muller transform.
    pub fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-12);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Two independent N(0, 1) samples (Box-Muller; more efficient in pairs).
    pub fn next_normal_pair(&mut self) -> (f32, f32) {
        let u1 = self.next_f32().max(1e-12);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }

    /// Fill `buf` with N(0, 1) samples.
    pub fn fill_normal(&mut self, buf: &mut [f32]) {
        let mut i = 0;
        while i + 1 < buf.len() {
            let (a, b) = self.next_normal_pair();
            buf[i] = a;
            buf[i + 1] = b;
            i += 2;
        }
        if i < buf.len() {
            buf[i] = self.next_normal();
        }
    }
}

// ─── AnomalyHandle ───────────────────────────────────────────────────────────

/// Lightweight session descriptor for anomaly detection operations.
#[derive(Debug, Clone)]
pub struct AnomalyHandle {
    /// SM version.
    pub sm: SmVersion,
    /// Device ordinal (0-indexed).
    pub device: u32,
    /// Deterministic RNG.
    pub rng: LcgRng,
}

impl AnomalyHandle {
    /// Construct with explicit parameters.
    #[must_use]
    pub fn new(device: u32, sm: SmVersion, seed: u64) -> Self {
        Self {
            sm,
            device,
            rng: LcgRng::new(seed),
        }
    }

    /// Default handle: SM 8.0, device 0, seed 42.
    #[must_use]
    pub fn default_handle() -> Self {
        Self::new(0, SmVersion(80), 42)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sm_version_ptx_strings() {
        assert_eq!(SmVersion(75).ptx_version_str(), "7.5");
        assert_eq!(SmVersion(80).ptx_version_str(), "8.0");
        assert_eq!(SmVersion(90).ptx_version_str(), "8.4");
        assert_eq!(SmVersion(100).ptx_version_str(), "8.7");
        assert_eq!(SmVersion(120).ptx_version_str(), "8.7");
    }

    #[test]
    fn lcg_determinism() {
        let mut a = LcgRng::new(7);
        let mut b = LcgRng::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn lcg_f32_in_range() {
        let mut rng = LcgRng::new(11);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "f32={v}");
        }
    }

    #[test]
    fn lcg_normal_finite() {
        let mut rng = LcgRng::new(13);
        let mut buf = vec![0.0_f32; 64];
        rng.fill_normal(&mut buf);
        assert!(buf.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn anomaly_handle_default() {
        let h = AnomalyHandle::default_handle();
        assert_eq!(h.device, 0);
        assert_eq!(h.sm, SmVersion(80));
    }
}
