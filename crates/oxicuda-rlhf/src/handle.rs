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
        (self.state >> 33) as u32
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    #[inline]
    pub fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u32() as usize) % n
    }

    pub fn next_normal_pair(&mut self) -> (f32, f32) {
        let u1 = (self.next_f32() + 1e-10).min(1.0 - 1e-10);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }

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

#[derive(Debug, Clone)]
pub struct RlhfHandle {
    pub sm: SmVersion,
    pub device: u32,
    pub rng: LcgRng,
}

impl RlhfHandle {
    #[must_use]
    pub fn new(device: u32, sm: SmVersion, seed: u64) -> Self {
        Self {
            sm,
            device,
            rng: LcgRng::new(seed),
        }
    }

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
    fn default_handle_fields() {
        let h = RlhfHandle::default_handle();
        assert_eq!(h.device, 0);
        assert_eq!(h.sm, SmVersion(80));
    }
}
