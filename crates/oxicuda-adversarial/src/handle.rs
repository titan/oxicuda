//! Session handle for `oxicuda-adversarial`.

// ─── SmVersion ───────────────────────────────────────────────────────────────

/// SM version encoded as `major*10 + minor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmVersion(pub u32);

impl SmVersion {
    /// Raw u32 version.
    #[must_use]
    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }

    /// PTX `.version` directive string.
    #[must_use]
    pub fn ptx_version_str(self) -> &'static str {
        match self.0 {
            v if v >= 100 => "8.7",
            v if v >= 90 => "8.4",
            v if v >= 80 => "8.0",
            _ => "7.5",
        }
    }

    /// PTX `.target` string (e.g. "sm_80").
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

/// Knuth MMIX 64-bit LCG.
#[derive(Debug, Clone)]
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    /// New LCG with given seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Next u32 (high 32 bits of LCG state).
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 33) as u32
    }

    /// Uniform `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }

    /// Uniform `[0, n)`.
    #[inline]
    pub fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u32() as usize) % n
    }

    /// Box-Muller pair of N(0,1) samples.
    pub fn next_normal_pair(&mut self) -> (f32, f32) {
        let u1 = (self.next_f32() + 1e-10).min(1.0 - 1e-10);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }

    /// Fill with N(0,1) samples.
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

    /// Fisher-Yates in-place shuffle.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = self.next_usize(i + 1);
            slice.swap(i, j);
        }
    }
}

// ─── AdvHandle ───────────────────────────────────────────────────────────────

/// Session handle for adversarial operations.
#[derive(Debug, Clone)]
pub struct AdvHandle {
    sm: SmVersion,
    rng: LcgRng,
    device: u32,
}

impl AdvHandle {
    /// New handle.
    #[must_use]
    pub fn new(device: u32, sm: SmVersion, seed: u64) -> Self {
        Self {
            sm,
            rng: LcgRng::new(seed),
            device,
        }
    }

    /// Default test handle (device 0, SM 8.0, seed 42).
    #[must_use]
    pub fn default_handle() -> Self {
        Self::new(0, SmVersion(80), 42)
    }

    /// SM version.
    #[must_use]
    pub fn sm_version(&self) -> SmVersion {
        self.sm
    }

    /// Device ordinal.
    #[must_use]
    pub fn device(&self) -> u32 {
        self.device
    }

    /// Mutable RNG access.
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
    fn sm_version_target_str() {
        assert_eq!(SmVersion(86).target_str(), "sm_86");
    }

    #[test]
    fn sm_version_display() {
        assert_eq!(SmVersion(80).to_string(), "SM 8.0");
    }

    #[test]
    fn handle_default() {
        let h = AdvHandle::default_handle();
        assert_eq!(h.device(), 0);
        assert_eq!(h.sm_version(), SmVersion(80));
    }

    #[test]
    fn handle_custom() {
        let h = AdvHandle::new(1, SmVersion(90), 7);
        assert_eq!(h.device(), 1);
        assert_eq!(h.sm_version(), SmVersion(90));
    }

    #[test]
    fn lcg_determinism() {
        let mut a = LcgRng::new(1);
        let mut b = LcgRng::new(1);
        for _ in 0..50 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn lcg_f32_in_range() {
        let mut rng = LcgRng::new(0);
        for _ in 0..200 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn lcg_normal_finite() {
        let mut rng = LcgRng::new(0);
        let mut buf = vec![0.0_f32; 32];
        rng.fill_normal(&mut buf);
        assert!(buf.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn lcg_shuffle_preserves() {
        let mut rng = LcgRng::new(0);
        let mut v: Vec<usize> = (0..16).collect();
        rng.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn lcg_next_usize_in_range() {
        let mut rng = LcgRng::new(0);
        for _ in 0..100 {
            let v = rng.next_usize(7);
            assert!(v < 7);
        }
    }
}
