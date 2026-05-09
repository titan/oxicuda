/// GPU SM (Streaming Multiprocessor) version as a raw `u32` (major*10 + minor).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmVersion(pub u32);

impl SmVersion {
    #[must_use]
    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }

    #[must_use]
    pub fn ptx_version_str(self) -> &'static str {
        match self.0 {
            v if v >= 100 => "8.7",
            v if v >= 90 => "8.4",
            v if v >= 80 => "8.0",
            _ => "7.5",
        }
    }

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
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1_442_695_040_888_963_407),
        }
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 33) ^ self.state) as u32
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let hi = self.next_u32() as u64;
        let lo = self.next_u32() as u64;
        (hi << 32) | lo
    }

    pub fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-12);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    pub fn fill_normal(&mut self, buf: &mut [f32]) {
        let mut i = 0;
        while i + 1 < buf.len() {
            let u1 = self.next_f32().max(1e-12_f32);
            let u2 = self.next_f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            buf[i] = r * theta.cos();
            buf[i + 1] = r * theta.sin();
            i += 2;
        }
        if i < buf.len() {
            buf[i] = self.next_normal();
        }
    }
}

/// Lightweight session descriptor for recommender system operations.
#[derive(Debug, Clone)]
pub struct RecsysHandle {
    pub sm: SmVersion,
    pub rng: LcgRng,
}

impl RecsysHandle {
    #[must_use]
    pub fn new(sm_version: u32, seed: u64) -> Self {
        Self {
            sm: SmVersion(sm_version),
            rng: LcgRng::new(seed),
        }
    }

    #[must_use]
    pub fn sm(&self) -> SmVersion {
        self.sm
    }

    #[must_use]
    pub fn rng(&self) -> &LcgRng {
        &self.rng
    }

    pub fn rng_mut(&mut self) -> &mut LcgRng {
        &mut self.rng
    }
}
