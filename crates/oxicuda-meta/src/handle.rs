#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SmVersion {
    Sm75,
    Sm80,
    Sm86,
    Sm90,
    Sm100,
    Sm120,
}

impl SmVersion {
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

#[derive(Debug, Clone)]
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
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
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }

    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = self.next_usize(i + 1);
            slice.swap(i, j);
        }
    }
}

pub struct MetaHandle {
    pub sm: SmVersion,
    pub device: u32,
    pub rng: LcgRng,
}

impl MetaHandle {
    pub fn default_handle() -> Self {
        Self {
            sm: SmVersion::Sm80,
            device: 0,
            rng: LcgRng::new(42),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_determinism() {
        let mut a = LcgRng::new(7);
        let mut b = LcgRng::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn lcg_f32_range() {
        let mut rng = LcgRng::new(11);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn default_handle_fields() {
        let h = MetaHandle::default_handle();
        assert_eq!(h.device, 0);
        assert_eq!(h.sm, SmVersion::Sm80);
    }
}
