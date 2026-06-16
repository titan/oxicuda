//! Handle types and LCG RNG for `oxicuda-moe`.

/// LCG pseudo-random number generator (Knuth MMIX parameters).
#[derive(Debug, Clone)]
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    /// Create a new LCG RNG from the given seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1_442_695_040_888_963_407),
        }
    }

    /// Generate a random u32.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 33) ^ self.state) as u32
    }

    /// Generate a random f32 in [0, 1).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    /// Generate a random usize in [0, n).
    #[inline]
    pub fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u32() as usize) % n
    }

    /// Generate a Box-Muller normal pair (mean=0, std=1).
    pub fn next_normal_pair(&mut self) -> (f32, f32) {
        let u1 = (self.next_f32() + 1e-10).min(1.0 - 1e-10);
        let u2 = self.next_f32();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        (radius * theta.cos(), radius * theta.sin())
    }

    /// Fill a buffer with N(0, std) samples.
    pub fn fill_normal_scaled(&mut self, buf: &mut [f32], std_dev: f32) {
        let mut idx = 0;
        while idx + 1 < buf.len() {
            let (a, b) = self.next_normal_pair();
            buf[idx] = a * std_dev;
            buf[idx + 1] = b * std_dev;
            idx += 2;
        }
        if idx < buf.len() {
            let (a, _) = self.next_normal_pair();
            buf[idx] = a * std_dev;
        }
    }

    /// Fill a buffer with N(0, 1) samples.
    pub fn fill_normal(&mut self, buf: &mut [f32]) {
        self.fill_normal_scaled(buf, 1.0);
    }
}

/// SM (Streaming Multiprocessor) version enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmVersion {
    Sm75,
    Sm80,
    Sm86,
    Sm90,
    Sm100,
    Sm120,
}

impl SmVersion {
    /// Return the numeric SM version.
    #[must_use]
    pub fn as_u32(self) -> u32 {
        match self {
            Self::Sm75 => 75,
            Self::Sm80 => 80,
            Self::Sm86 => 86,
            Self::Sm90 => 90,
            Self::Sm100 => 100,
            Self::Sm120 => 120,
        }
    }
}

/// Handle encapsulating SM version, device, and an LCG RNG.
pub struct MoeHandle {
    pub sm: SmVersion,
    pub device: u32,
    pub rng: LcgRng,
}

impl MoeHandle {
    /// Create a handle with SM80, device 0, and seed 42.
    #[must_use]
    pub fn default_handle() -> Self {
        Self {
            sm: SmVersion::Sm80,
            device: 0,
            rng: LcgRng::new(42),
        }
    }

    /// Create a handle with a custom SM version and device.
    #[must_use]
    pub fn new(sm: SmVersion, device: u32, seed: u64) -> Self {
        Self {
            sm,
            device,
            rng: LcgRng::new(seed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_rng_determinism() {
        let mut a = LcgRng::new(7);
        let mut b = LcgRng::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn lcg_rng_f32_in_range() {
        let mut rng = LcgRng::new(11);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn sm_version_as_u32() {
        assert_eq!(SmVersion::Sm75.as_u32(), 75);
        assert_eq!(SmVersion::Sm80.as_u32(), 80);
        assert_eq!(SmVersion::Sm86.as_u32(), 86);
        assert_eq!(SmVersion::Sm90.as_u32(), 90);
        assert_eq!(SmVersion::Sm100.as_u32(), 100);
        assert_eq!(SmVersion::Sm120.as_u32(), 120);
    }

    #[test]
    fn default_handle_fields() {
        let h = MoeHandle::default_handle();
        assert_eq!(h.device, 0);
        assert_eq!(h.sm, SmVersion::Sm80);
    }
}
