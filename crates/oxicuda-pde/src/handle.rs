//! Handle and RNG primitives for `oxicuda-pde`.
//!
//! Mirrors the pattern of `oxicuda-tn` / `oxicuda-tda` / `oxicuda-evol`.
//!
//! Provides:
//! - [`SmVersion`] tagging GPU compute capability.
//! - [`LcgRng`] a minimal LCG (MMIX variant) pseudo-random generator that uses bit 32
//!   for boolean sampling (LCG low bits have period defects).
//! - [`PdeHandle`] bundling SM version and a seeded RNG.

/// SM compute capability version (e.g. 75 for SM 7.5, 100 for Blackwell SM 10.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SmVersion(pub u32);

impl SmVersion {
    pub const SM_75: Self = Self(75);
    pub const SM_80: Self = Self(80);
    pub const SM_86: Self = Self(86);
    pub const SM_89: Self = Self(89);
    pub const SM_90: Self = Self(90);
    pub const SM_100: Self = Self(100);

    /// Return the numeric compute capability.
    #[must_use]
    pub fn value(self) -> u32 {
        self.0
    }
}

/// Minimal LCG (MMIX variant) pseudo-random number generator.
///
/// Uses the Knuth MMIX constants for a full-period 64-bit LCG.
/// CRITICAL: `next_bool()` uses bit 32, NOT bit 0 (bit 0 has period 2 in MMIX LCG).
#[derive(Debug, Clone)]
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    const MUL: u64 = 6_364_136_223_846_793_005;
    const ADD: u64 = 1_442_695_040_888_963_407;

    /// Create a new LCG seeded with `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1),
        }
    }

    /// Advance the state and return the next 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(Self::MUL).wrapping_add(Self::ADD);
        self.state
    }

    /// Return a uniform `f64` in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Return a random `bool`. Uses bit 32 (NOT bit 0) for better statistical quality.
    pub fn next_bool(&mut self) -> bool {
        (self.next_u64() >> 32) & 1 == 1
    }

    /// Return a random `usize` in `[0, n)`.
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n.max(1)
    }

    /// Return a uniform `f64` in `[lo, hi)`.
    pub fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Draw from a standard normal distribution via Box-Muller.
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1.0e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Top-level numerical PDE handle bundling SM version and a seeded RNG.
#[derive(Debug, Clone)]
pub struct PdeHandle {
    pub sm: SmVersion,
    pub rng: LcgRng,
}

impl PdeHandle {
    /// Create a new handle for the given SM version and RNG seed.
    #[must_use]
    pub fn new(sm: SmVersion, seed: u64) -> Self {
        Self {
            sm,
            rng: LcgRng::new(seed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sm_constants_correct() {
        assert_eq!(SmVersion::SM_75.value(), 75);
        assert_eq!(SmVersion::SM_100.value(), 100);
    }

    #[test]
    fn lcg_deterministic() {
        let mut r1 = LcgRng::new(42);
        let mut r2 = LcgRng::new(42);
        for _ in 0..16 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn lcg_unit_interval() {
        let mut r = LcgRng::new(7);
        for _ in 0..1000 {
            let v = r.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn lcg_bool_balanced() {
        let mut r = LcgRng::new(13);
        let mut trues = 0usize;
        for _ in 0..1000 {
            if r.next_bool() {
                trues += 1;
            }
        }
        assert!(trues > 400 && trues < 600);
    }

    #[test]
    fn lcg_normal_mean_zero() {
        let mut r = LcgRng::new(99);
        let n = 10_000;
        let mean: f64 = (0..n).map(|_| r.next_normal()).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.2);
    }

    #[test]
    fn handle_construction() {
        let h = PdeHandle::new(SmVersion::SM_90, 0);
        assert_eq!(h.sm.value(), 90);
    }
}
