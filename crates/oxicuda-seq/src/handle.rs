//! Handle and RNG primitives for `oxicuda-seq`.

/// SM compute capability version (e.g. 75 for SM 7.5, 86 for SM 8.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SmVersion(pub u32);

impl SmVersion {
    pub const SM_75: Self = Self(75);
    pub const SM_80: Self = Self(80);
    pub const SM_86: Self = Self(86);
    pub const SM_89: Self = Self(89);
    pub const SM_90: Self = Self(90);
    pub const SM_100: Self = Self(100);

    /// Get the numeric SM compute capability value.
    pub fn value(self) -> u32 {
        self.0
    }
}

/// Minimal LCG (MMIX variant) pseudo-random number generator.
///
/// Uses Knuth's MMIX constants for a full-period 64-bit LCG.
/// CRITICAL: `next_bool()` uses bit 32, NOT bit 0 (bit 0 has period 2 in MMIX LCG).
#[derive(Debug, Clone)]
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    const MUL: u64 = 6_364_136_223_846_793_005;
    const ADD: u64 = 1_442_695_040_888_963_407;

    /// Create a new LCG seeded with `seed`.
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

    /// Return a random bool. Uses bit 32 (NOT bit 0) for better statistical quality.
    pub fn next_bool(&mut self) -> bool {
        (self.next_u64() >> 32) & 1 == 1
    }

    /// Return a standard-normal sample via Box-Muller.
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Uniform sample on `[lo, hi)`.
    pub fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Sample a categorical index given a probability vector.  Probabilities are
    /// expected to sum to ~1; if the sum is short due to rounding, the last index
    /// is returned.
    pub fn sample_categorical(&mut self, probs: &[f64]) -> usize {
        let u = self.next_f64();
        let mut acc = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if u <= acc {
                return i;
            }
        }
        if probs.is_empty() { 0 } else { probs.len() - 1 }
    }

    /// Return a random `usize` in `[0, n)`.  Returns 0 if `n == 0`.
    pub fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() as usize) % n
        }
    }
}

/// Top-level sequence-model computation handle bundling SM version and a seeded RNG.
#[derive(Debug, Clone)]
pub struct SeqHandle {
    pub sm: SmVersion,
    pub rng: LcgRng,
}

impl SeqHandle {
    /// Create a new handle from an `SmVersion` and seed.
    pub fn new(sm: SmVersion, seed: u64) -> Self {
        Self {
            sm,
            rng: LcgRng::new(seed),
        }
    }

    /// Create a new handle from a raw SM numeric code and seed.
    pub fn from_sm_code(sm_code: u32, seed: u64) -> Self {
        Self::new(SmVersion(sm_code), seed)
    }
}
