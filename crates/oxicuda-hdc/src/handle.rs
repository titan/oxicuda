//! GPU SM version descriptor and LCG random number generator for HDC operations.

use crate::error::{HdcError, HdcResult};

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
            state: seed ^ 0x1234_5678_9ABC_DEF0,
        }
    }

    /// Advance the state and return the next 64-bit pseudo-random integer.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Return a pseudo-random `u32`.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Return a pseudo-random `f32` in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Return a pseudo-random boolean (50/50).
    /// Uses bit 32 of the state (high bits have better randomness than low bits in LCGs).
    #[inline]
    pub fn next_bool(&mut self) -> bool {
        (self.next_u64() >> 32) & 1 == 0
    }

    /// Fill a buffer with random binary values in {-1, +1}.
    pub fn fill_binary(&mut self, buf: &mut [i8]) {
        for v in buf.iter_mut() {
            *v = if self.next_bool() { 1 } else { -1 };
        }
    }

    /// Fill a buffer with uniform f32 values in [0, 1).
    pub fn fill_uniform_f32(&mut self, buf: &mut [f32]) {
        for v in buf.iter_mut() {
            *v = self.next_f32();
        }
    }

    /// Return two independent N(0,1) samples via Box-Muller transform.
    pub fn normal_pair_f32(&mut self) -> (f32, f32) {
        let u1 = (self.next_f32() as f64).max(f64::EPSILON);
        let u2 = self.next_f32() as f64;
        let r = (-2.0 * u1.ln()).sqrt() as f32;
        let theta = (std::f64::consts::TAU * u2) as f32;
        (r * theta.cos(), r * theta.sin())
    }

    /// Return a pseudo-random `usize` in `[0, n)`.
    #[inline]
    pub fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() as usize) % n
    }
}

/// Lightweight session descriptor for HDC operations.
#[derive(Debug, Clone)]
pub struct HdcHandle {
    /// Streaming multiprocessor version for PTX selection.
    pub sm: SmVersion,
    /// Deterministic random number generator.
    pub rng: LcgRng,
}

impl HdcHandle {
    /// Create a new handle with the given SM version and RNG seed.
    #[must_use]
    pub fn new(sm: u32, seed: u64) -> Self {
        Self {
            sm: SmVersion(sm),
            rng: LcgRng::new(seed),
        }
    }

    /// Return the SM version.
    #[must_use]
    pub fn sm(&self) -> SmVersion {
        self.sm
    }

    /// Generate a random binary hypervector of dimension D (values ±1).
    pub fn random_binary_hv(&mut self, dim: usize) -> HdcResult<Vec<i8>> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        let mut v = vec![0i8; dim];
        self.rng.fill_binary(&mut v);
        Ok(v)
    }

    /// Generate a random integer hypervector of dimension D (values in {-1, 0, +1}).
    pub fn random_integer_hv(&mut self, dim: usize) -> HdcResult<Vec<i32>> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        let mut v = vec![0i32; dim];
        for x in v.iter_mut() {
            *x = ((self.rng.next_u64() as i64).rem_euclid(3) - 1) as i32;
        }
        Ok(v)
    }

    /// Generate a random complex hypervector on unit circle.
    /// Stored as [re_0, im_0, re_1, im_1, ...] of length 2*dim.
    pub fn random_complex_hv(&mut self, dim: usize) -> HdcResult<Vec<f32>> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        let mut v = vec![0f32; 2 * dim];
        let mut i = 0;
        while i < dim {
            let theta = self.rng.next_f32() * std::f32::consts::TAU;
            v[2 * i] = theta.cos();
            v[2 * i + 1] = theta.sin();
            i += 1;
        }
        Ok(v)
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
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn hdc_handle_new() {
        let h = HdcHandle::new(80, 42);
        assert_eq!(h.sm.as_u32(), 80);
    }
}
