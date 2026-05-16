//! Handle types for oxicuda-evol: SM version tag, LCG RNG, and the combined EvolHandle.

/// A type-safe SM version identifier (e.g. `SmVersion(80)` for sm_80).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SmVersion(pub u32);

/// A fast, stateless-ish LCG pseudo-random number generator using the MMIX multiplier.
///
/// This is intentionally a minimal, allocation-free RNG suitable for use inside GPU-style
/// simulation code. It is **not** cryptographically secure.
///
/// ## Bit-quality note
/// The low-order bits of a multiplicative LCG (MMIX) have very short periods. Bit 0 has
/// period 2. `next_bool()` therefore uses bit 32 (the most significant half of the 64-bit
/// output) to obtain a well-distributed boolean.
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    /// MMIX multiplier (Knuth).
    const MUL: u64 = 6_364_136_223_846_793_005;
    /// MMIX additive constant.
    const ADD: u64 = 1_442_695_040_888_963_407;

    /// Create a new `LcgRng` seeded from `seed`.
    pub fn new(seed: u64) -> Self {
        // +1 ensures seed=0 gives a non-trivial initial state.
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Advance the LCG and return a 64-bit pseudo-random value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(Self::MUL).wrapping_add(Self::ADD);
        self.state
    }

    /// Return a pseudo-random `f64` in `[0, 1)` using the 53 high-order bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Return a pseudo-random `bool` using bit 32 (avoids bit-0 period-2 defect).
    pub fn next_bool(&mut self) -> bool {
        (self.next_u64() >> 32) & 1 == 1
    }

    /// Return a pseudo-random `usize` uniformly in `[0, n)`.
    ///
    /// Uses simple modulo reduction — biased for non-power-of-two `n`, but adequate for EA use.
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }

    /// Return a pseudo-random standard-normal variate via the Box-Muller transform.
    ///
    /// `u1` is clamped away from zero to avoid `ln(0)`.
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// In-place Fisher-Yates shuffle of any slice.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        let n = v.len();
        for i in (1..n).rev() {
            let j = self.next_usize(i + 1);
            v.swap(i, j);
        }
    }
}

/// Combined handle carrying an SM version tag and a seeded LCG RNG.
pub struct EvolHandle {
    /// SM version of the target device (purely informational for PTX generation).
    pub sm: SmVersion,
    /// Deterministic LCG random number generator.
    pub rng: LcgRng,
}

impl EvolHandle {
    /// Create an `EvolHandle` for the given SM version and random seed.
    pub fn new(sm: u32, seed: u64) -> Self {
        Self {
            sm: SmVersion(sm),
            rng: LcgRng::new(seed),
        }
    }
}
