/// GPU SM (Streaming Multiprocessor) version as a raw `u32` (major*10 + minor).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmVersion(pub u32);

impl SmVersion {
    /// Return the raw SM version number.
    #[must_use]
    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }

    /// Map SM version to the matching PTX ISA version string.
    #[must_use]
    pub fn ptx_version_str(self) -> &'static str {
        match self.0 {
            v if v >= 100 => "8.7",
            v if v >= 90 => "8.4",
            v if v >= 80 => "8.0",
            _ => "7.5",
        }
    }

    /// Format as PTX target string, e.g. `"sm_80"`.
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

/// Knuth MMIX 64-bit Linear Congruential Generator with Box-Muller normal sampling.
#[derive(Debug, Clone)]
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    /// Seed the generator; the initial state is mixed to avoid trivial sequences.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1_442_695_040_888_963_407),
        }
    }

    /// Advance the state and return the next 32-bit pseudo-random integer.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 33) ^ self.state) as u32
    }

    /// Return a pseudo-random `f32` in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    /// Return a pseudo-random `u64` by combining two `u32` outputs.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let lo = self.next_u32() as u64;
        let hi = self.next_u32() as u64;
        (hi << 32) | lo
    }

    /// Return a sample from N(0,1) via Box-Muller transform.
    pub fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-12);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Return two independent N(0,1) samples at once (Box-Muller).
    pub fn next_normal_pair(&mut self) -> (f32, f32) {
        let u1 = self.next_f32().max(1e-12);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }

    /// Fill a buffer with N(0,1) samples using paired Box-Muller draws.
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

    /// Return a pseudo-random `usize` in `[0, n)`.
    #[inline]
    pub fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u32() as usize) % n
    }
}

/// Lightweight session descriptor for PEFT operations.
#[derive(Debug, Clone)]
pub struct PeftHandle {
    /// Streaming multiprocessor version for PTX selection.
    pub sm: SmVersion,
    /// Deterministic random number generator.
    pub rng: LcgRng,
}

impl PeftHandle {
    /// Create a new handle with the given SM version and RNG seed.
    #[must_use]
    pub fn new(sm_version: u32, seed: u64) -> Self {
        Self {
            sm: SmVersion(sm_version),
            rng: LcgRng::new(seed),
        }
    }

    /// Return the SM version.
    #[must_use]
    pub fn sm(&self) -> SmVersion {
        self.sm
    }

    /// Return a mutable reference to the internal RNG.
    #[must_use]
    pub fn rng_mut(&mut self) -> &mut LcgRng {
        &mut self.rng
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
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn peft_handle_new() {
        let h = PeftHandle::new(80, 42);
        assert_eq!(h.sm.as_u32(), 80);
    }
}
