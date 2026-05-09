//! Session handle for `oxicuda-quantum`.

// ─── SmVersion ───────────────────────────────────────────────────────────────

/// GPU SM version as a raw `u32` (major*10 + minor).
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

// ─── LcgRng ──────────────────────────────────────────────────────────────────

/// Knuth MMIX 64-bit LCG with Box-Muller normal sampling.
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
    pub fn next_u64(&mut self) -> u64 {
        let lo = self.next_u32() as u64;
        let hi = self.next_u32() as u64;
        (hi << 32) | lo
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    pub fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-12);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    pub fn next_normal_pair(&mut self) -> (f32, f32) {
        let u1 = self.next_f32().max(1e-12);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

// ─── QuantumHandle ───────────────────────────────────────────────────────────

/// Lightweight session descriptor for quantum operations.
#[derive(Debug, Clone)]
pub struct QuantumHandle {
    pub sm: SmVersion,
    pub rng: LcgRng,
}

impl QuantumHandle {
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
    fn handle_new() {
        let h = QuantumHandle::new(80, 42);
        assert_eq!(h.sm(), SmVersion(80));
    }
}
