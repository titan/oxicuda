//! TDA handle: SM version, LCG random number generator, and top-level TdaHandle.

/// SM compute capability version (e.g. 75 for SM 7.5, 86 for SM 8.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SmVersion(pub u32);

/// Minimal LCG (MMIX variant) pseudo-random number generator.
///
/// Uses the Knuth MMIX constants for a full-period 64-bit LCG.
/// CRITICAL: `next_bool()` uses bit 32, NOT bit 0 (bit 0 has period 2 in MMIX LCG).
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    const MUL: u64 = 6_364_136_223_846_793_005;
    const ADD: u64 = 1_442_695_040_888_963_407;

    /// Create a new LCG seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Advance the state and return the next 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(Self::MUL).wrapping_add(Self::ADD);
        self.state
    }

    /// Return a uniform f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Return a random bool.  Uses bit 32 (NOT bit 0) for better statistical quality.
    pub fn next_bool(&mut self) -> bool {
        (self.next_u64() >> 32) & 1 == 1
    }

    /// Return a random `usize` in `[0, n)`.
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

/// Top-level TDA computation handle bundling SM version and a seeded RNG.
pub struct TdaHandle {
    pub sm: SmVersion,
    pub rng: LcgRng,
}

impl TdaHandle {
    /// Create a new handle for the given SM version and RNG seed.
    pub fn new(sm: u32, seed: u64) -> Self {
        Self {
            sm: SmVersion(sm),
            rng: LcgRng::new(seed),
        }
    }
}
