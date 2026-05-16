//! Session handle and LCG RNG for `oxicuda-privacy`.
//!
//! Provides `PrivacyHandle` (device/SM-version wrapper + RNG) and `LcgRng`
//! (deterministic 64-bit LCG for CPU-side noise generation).

use crate::error::{PrivacyError, PrivacyResult};

// ─── SmVersion ───────────────────────────────────────────────────────────────

/// SM (Streaming Multiprocessor) version encoded as `major*10 + minor`.
///
/// Examples: 75 = SM 7.5 (Turing), 80 = SM 8.0 (Ampere A100),
/// 90 = SM 9.0 (Hopper H100), 100 = SM 10.0 (Blackwell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmVersion(pub u32);

impl SmVersion {
    /// Return the raw u32 version number.
    #[must_use]
    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for SmVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SM {}.{}", self.0 / 10, self.0 % 10)
    }
}

// ─── LcgRng ──────────────────────────────────────────────────────────────────

/// Minimal 64-bit LCG random number generator for deterministic CPU-side
/// sampling, noise generation, and privacy mechanism simulation.
///
/// Uses the Knuth MMIX multiplier:
/// `x_{n+1} = 6364136223846793005·xₙ + 1442695040888963407 (mod 2⁶⁴)`.
#[derive(Debug, Clone)]
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    /// Create a new LCG with the given seed (XORed with a magic constant
    /// to avoid degenerate all-zero initial state).
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x1234_5678_9ABC_DEF0,
        }
    }

    /// Advance one step and return the raw 64-bit state.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Return a `f64` uniformly distributed in `[0, 1)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // Use top 53 bits for double precision
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Return a `f32` uniformly distributed in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }

    /// Fill a buffer with uniform `[0, 1)` f64 values.
    pub fn fill_uniform(&mut self, buf: &mut [f64]) {
        for v in buf.iter_mut() {
            *v = self.next_f64();
        }
    }

    /// Generate a pair of standard-normal samples via the Box-Muller transform.
    ///
    /// Returns `(z₁, z₂)` with z₁, z₂ ~ N(0,1) independently.
    /// Clamps u1 away from 0 to avoid `ln(0)`.
    pub fn normal_pair(&mut self) -> (f64, f64) {
        let u1 = self.next_f64().max(f64::EPSILON);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

// ─── PrivacyHandle ───────────────────────────────────────────────────────────

/// Privacy session handle wrapping the target SM version and an `LcgRng`.
///
/// All noise-generation methods are centralised here so callers can manage
/// a single handle rather than juggling separate RNG and SM state.
pub struct PrivacyHandle {
    pub sm: SmVersion,
    pub rng: LcgRng,
}

impl PrivacyHandle {
    /// Construct a new handle for the given SM version and RNG seed.
    #[must_use]
    pub fn new(sm: u32, seed: u64) -> Self {
        Self {
            sm: SmVersion(sm),
            rng: LcgRng::new(seed),
        }
    }

    /// Return the SM version of this handle.
    #[must_use]
    pub fn sm(&self) -> SmVersion {
        self.sm
    }

    /// Generate `n` independent N(0, σ²) samples via Box-Muller.
    ///
    /// # Errors
    /// Returns `InvalidParameter` if `sigma < 0`.
    pub fn generate_gaussian_noise(&mut self, sigma: f64, n: usize) -> PrivacyResult<Vec<f64>> {
        if sigma < 0.0 {
            return Err(PrivacyError::InvalidParameter("sigma must be ≥ 0".into()));
        }
        let mut out = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let (a, b) = self.rng.normal_pair();
            out.push(a * sigma);
            if i + 1 < n {
                out.push(b * sigma);
            }
            i += 2;
        }
        out.truncate(n);
        Ok(out)
    }

    /// Generate `n` independent Laplace(0, scale) samples via the inverse CDF.
    ///
    /// # Errors
    /// Returns `InvalidParameter` if `scale ≤ 0`.
    pub fn generate_laplace_noise(&mut self, scale: f64, n: usize) -> PrivacyResult<Vec<f64>> {
        if scale <= 0.0 {
            return Err(PrivacyError::InvalidParameter("scale must be > 0".into()));
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let u = self.rng.next_f64() - 0.5;
            // Inverse CDF of Laplace: -scale * sign(u) * ln(1 - 2|u|)
            out.push(-scale * u.signum() * (1.0 - 2.0 * u.abs()).ln());
        }
        Ok(out)
    }
}
