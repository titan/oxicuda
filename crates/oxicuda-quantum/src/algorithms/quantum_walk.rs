//! Discrete-time coined quantum walk on a cycle.
//!
//! References: Aharonov, Davidovich, Zagury, *"Quantum random walks"*,
//! Phys. Rev. A 48, 1687 (1993); Kempe, *"Quantum random walks – an introductory
//! overview"*, Contemp. Phys. 44, 307 (2003); Aharonov, Ambainis, Kempe,
//! Vazirani, *"Quantum walks on graphs"*, STOC 2001.
//!
//! # Model
//!
//! The walker lives on a cycle of `N` positions (with `N = 2^p` so positions fit
//! in `p` qubits) tensored with a two-level *coin*. One step of the walk is
//!
//! ```text
//! U = S · (I ⊗ C),
//! ```
//!
//! where `C` is the Hadamard coin acting on the coin qubit and `S` is the
//! coin-conditioned cyclic shift
//!
//! ```text
//! S |x, 0⟩ = |(x + 1) mod N, 0⟩,
//! S |x, 1⟩ = |(x − 1) mod N, 1⟩.
//! ```
//!
//! # State encoding (little-endian)
//!
//! Consistent with this crate's LSB-ordered [`StateVector`] (qubit `q` ↦ mask
//! `1 << q`), the coin occupies qubit `0` and the position occupies qubits
//! `1 ..= p`. A basis index `i` therefore decomposes as
//!
//! ```text
//! coin     = i & 1,
//! position = i >> 1,
//! ```
//!
//! so the global amplitude index is `i = coin + 2·position`.
//!
//! The Hadamard coin is applied with the shared [`apply_1q_inplace`] gate
//! machinery; the shift is the exact coin-conditioned cyclic permutation of the
//! amplitude vector (a genuine unitary, simulated without approximation).
//!
//! # Physics demonstrated by the tests
//!
//! The Hadamard walk spreads **ballistically**: the position standard deviation
//! grows *linearly* in the number of steps `T` (`σ ≈ 0.54·T` for the symmetric
//! coin), in stark contrast to the classical random walk's diffusive `σ = √T`.
//! The symmetric initial coin `(|0⟩ + i|1⟩)/√2` yields a left–right symmetric
//! distribution, and on the smallest cycles the walk is exactly periodic in time.

use crate::error::{QuantumError, QuantumResult};
use crate::gates::hadamard::gate_h;
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;
use num_complex::Complex;

type Complex32 = Complex<f32>;

/// A discrete-time coined quantum walk on a cycle of `n_positions = 2^p` sites.
#[derive(Debug, Clone)]
pub struct CoinedWalk {
    /// Number of positions on the cycle; must be a power of two.
    n_positions: usize,
    /// Number of position qubits `p` (so `n_positions == 1 << p`).
    position_qubits: usize,
}

/// The two coin basis states used to seed a localized walker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoinInit {
    /// Coin `|0⟩` — the "rightward-biased" Hadamard walk seed.
    Zero,
    /// Coin `|1⟩` — the "leftward-biased" Hadamard walk seed.
    One,
    /// Symmetric coin `(|0⟩ + i|1⟩)/√2`, which yields a left–right symmetric
    /// position distribution under the Hadamard walk.
    SymmetricI,
}

impl CoinedWalk {
    /// Create a walk on a cycle of `n_positions` sites.
    ///
    /// # Errors
    /// * [`QuantumError::InvalidParameter`] when `n_positions` is not a power of
    ///   two, is `< 2`, or would require more than 29 position qubits (leaving
    ///   room for the coin qubit within the simulator's 30-qubit limit).
    pub fn new(n_positions: usize) -> QuantumResult<Self> {
        if n_positions < 2 || !n_positions.is_power_of_two() {
            return Err(QuantumError::InvalidParameter {
                name: format!("n_positions={n_positions} must be a power of two ≥ 2"),
            });
        }
        let position_qubits = n_positions.trailing_zeros() as usize;
        if position_qubits + 1 > 29 {
            return Err(QuantumError::InvalidParameter {
                name: format!("cycle of {n_positions} sites needs too many qubits"),
            });
        }
        Ok(Self {
            n_positions,
            position_qubits,
        })
    }

    /// The number of cycle positions.
    #[must_use]
    pub fn n_positions(&self) -> usize {
        self.n_positions
    }

    /// Total qubit count: `position_qubits + 1` (coin).
    #[must_use]
    pub fn n_qubits(&self) -> usize {
        self.position_qubits + 1
    }

    /// Build the localized initial state `|x0⟩ ⊗ |coin⟩` over `n_qubits()` qubits.
    ///
    /// # Errors
    /// [`QuantumError::QubitIndexOutOfRange`] (reused as a range error) when
    /// `x0 >= n_positions`.
    pub fn initial_state(&self, x0: usize, coin: CoinInit) -> QuantumResult<StateVector> {
        if x0 >= self.n_positions {
            return Err(QuantumError::QubitIndexOutOfRange {
                index: x0,
                n_qubits: self.n_positions,
            });
        }
        let n_qubits = self.n_qubits();
        let dim = 1usize << n_qubits;
        let mut amps = vec![Complex32::new(0.0, 0.0); dim];
        let base = x0 << 1; // position in qubits 1.. , coin in qubit 0
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        match coin {
            CoinInit::Zero => amps[base] = Complex32::new(1.0, 0.0),
            CoinInit::One => amps[base | 1] = Complex32::new(1.0, 0.0),
            CoinInit::SymmetricI => {
                amps[base] = Complex32::new(inv_sqrt2, 0.0);
                amps[base | 1] = Complex32::new(0.0, inv_sqrt2);
            }
        }
        Ok(StateVector { amps, n_qubits })
    }

    /// Apply the coin-conditioned cyclic shift `S` in place.
    ///
    /// `S |x, 0⟩ = |(x+1) mod N, 0⟩` and `S |x, 1⟩ = |(x−1) mod N, 1⟩`. Implemented
    /// as the exact permutation of the amplitude vector that this unitary induces.
    fn apply_shift(&self, sv: &mut StateVector) -> QuantumResult<()> {
        let expected = self.n_qubits();
        if sv.n_qubits != expected {
            return Err(QuantumError::DimensionMismatch {
                expected: 1usize << expected,
                got: sv.amps.len(),
            });
        }
        let n = self.n_positions;
        let dim = sv.amps.len();
        let mut out = vec![Complex32::new(0.0, 0.0); dim];
        for (i, &a) in sv.amps.iter().enumerate() {
            let coin = i & 1;
            let pos = i >> 1;
            let new_pos = if coin == 0 {
                (pos + 1) % n
            } else {
                (pos + n - 1) % n
            };
            out[(new_pos << 1) | coin] = a;
        }
        sv.amps = out;
        Ok(())
    }

    /// Apply one full step `U = S · (I ⊗ C)` in place (coin first, then shift).
    ///
    /// # Errors
    /// [`QuantumError::DimensionMismatch`] if `sv` does not have `n_qubits()`
    /// qubits; propagates gate-application errors.
    pub fn step(&self, sv: &mut StateVector) -> QuantumResult<()> {
        apply_1q_inplace(sv, 0, &gate_h())?;
        self.apply_shift(sv)
    }

    /// Marginal position distribution of `sv`, summed over the coin.
    ///
    /// Returns a length-`n_positions` vector `p[x] = |⟨x,0|ψ⟩|² + |⟨x,1|ψ⟩|²`.
    ///
    /// # Errors
    /// [`QuantumError::DimensionMismatch`] if `sv` does not have `n_qubits()`
    /// qubits.
    pub fn position_distribution(&self, sv: &StateVector) -> QuantumResult<Vec<f32>> {
        let expected = self.n_qubits();
        if sv.n_qubits != expected {
            return Err(QuantumError::DimensionMismatch {
                expected: 1usize << expected,
                got: sv.amps.len(),
            });
        }
        let mut dist = vec![0.0_f32; self.n_positions];
        for (i, a) in sv.amps.iter().enumerate() {
            dist[i >> 1] += a.norm_sqr();
        }
        Ok(dist)
    }

    /// Evolve `t_steps` steps from a localized state `|x0, coin⟩` and return the
    /// final position distribution (summed over the coin).
    ///
    /// # Errors
    /// Propagates from [`Self::initial_state`] and [`Self::step`].
    pub fn evolve(&self, x0: usize, coin: CoinInit, t_steps: usize) -> QuantumResult<Vec<f32>> {
        let mut sv = self.initial_state(x0, coin)?;
        for _ in 0..t_steps {
            self.step(&mut sv)?;
        }
        self.position_distribution(&sv)
    }
}

/// Position mean and standard deviation of a distribution centered (for variance
/// purposes) on `x0`, with positions interpreted as signed offsets `x − x0` to
/// avoid the cycle's wrap-around discontinuity. The caller must ensure the
/// distribution has not wrapped around the cycle (`n_positions` large enough for
/// the number of steps), so that signed offsets are unambiguous in
/// `[-N/2, N/2)`.
#[must_use]
pub fn position_std_about(dist: &[f32], x0: usize) -> f64 {
    let n = dist.len() as i64;
    let mut mean = 0.0_f64;
    let mut total = 0.0_f64;
    for (x, &p) in dist.iter().enumerate() {
        let mut off = x as i64 - x0 as i64;
        // Wrap signed offset into [-N/2, N/2).
        if off >= n / 2 {
            off -= n;
        } else if off < -n / 2 {
            off += n;
        }
        mean += off as f64 * p as f64;
        total += p as f64;
    }
    if total > 0.0 {
        mean /= total;
    }
    let mut var = 0.0_f64;
    for (x, &p) in dist.iter().enumerate() {
        let mut off = x as i64 - x0 as i64;
        if off >= n / 2 {
            off -= n;
        } else if off < -n / 2 {
            off += n;
        }
        let d = off as f64 - mean;
        var += d * d * p as f64;
    }
    if total > 0.0 {
        var /= total;
    }
    var.max(0.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;
    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;

    // (a) One Hadamard-walk step from |x0, 0⟩ produces the known two-amplitude
    //     split: amplitude 1/√2 at (x0+1, coin 0) and 1/√2 at (x0−1, coin 1).
    #[test]
    fn single_step_two_amplitude_split() {
        let walk = CoinedWalk::new(8).expect("valid CoinedWalk with 8 positions");
        let x0 = 3usize;
        let mut sv = walk
            .initial_state(x0, CoinInit::Zero)
            .expect("valid initial state at x0");
        walk.step(&mut sv)
            .expect("step should succeed on valid state");

        // H|0⟩ = (|0⟩+|1⟩)/√2; shift sends coin0 → x0+1, coin1 → x0−1.
        let n = 8usize;
        let idx_right = ((x0 + 1) % n) << 1; // coin 0
        let idx_left = (((x0 + n - 1) % n) << 1) | 1; // coin 1
        for (i, a) in sv.amps.iter().enumerate() {
            if i == idx_right {
                assert!((a.re - INV_SQRT2).abs() < TOL, "right re={}", a.re);
                assert!(a.im.abs() < TOL, "right im={}", a.im);
            } else if i == idx_left {
                assert!((a.re - INV_SQRT2).abs() < TOL, "left re={}", a.re);
                assert!(a.im.abs() < TOL, "left im={}", a.im);
            } else {
                assert!(a.norm() < TOL, "spurious amp[{i}]={a:?}");
            }
        }
    }

    // (b) Ballistic spreading: σ(T) is linear in T (slope > 0, R² ≈ 1) and σ is
    //     much larger than the classical √T at large T.
    #[test]
    fn ballistic_spreading_linear_std() {
        let walk = CoinedWalk::new(128).expect("valid CoinedWalk with 128 positions");
        let x0 = 64usize;
        let t_values: Vec<usize> = (4..=24).collect();
        let sigmas: Vec<f64> = t_values
            .iter()
            .map(|&t| {
                let dist = walk
                    .evolve(x0, CoinInit::SymmetricI, t)
                    .expect("evolve should succeed");
                position_std_about(&dist, x0)
            })
            .collect();

        // Least-squares linear fit σ = a·T + b.
        let n = t_values.len() as f64;
        let sx: f64 = t_values.iter().map(|&t| t as f64).sum();
        let sy: f64 = sigmas.iter().sum();
        let sxx: f64 = t_values.iter().map(|&t| (t as f64) * (t as f64)).sum();
        let sxy: f64 = t_values
            .iter()
            .zip(&sigmas)
            .map(|(&t, &s)| (t as f64) * s)
            .sum();
        let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        let intercept = (sy - slope * sx) / n;

        // Coefficient of determination R².
        let mean_y = sy / n;
        let ss_tot: f64 = sigmas.iter().map(|&s| (s - mean_y).powi(2)).sum();
        let ss_res: f64 = t_values
            .iter()
            .zip(&sigmas)
            .map(|(&t, &s)| {
                let pred = slope * t as f64 + intercept;
                (s - pred).powi(2)
            })
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;

        assert!(slope > 0.3, "slope={slope} should be positive & ~0.54");
        assert!(r2 > 0.97, "R²={r2} should be ≈ 1 (linear growth)");

        // Quantum σ ≫ classical √T at the largest T.
        let t_max = *t_values.last().expect("t_values is non-empty");
        let sigma_max = *sigmas.last().expect("sigmas is non-empty");
        let classical = (t_max as f64).sqrt();
        assert!(
            sigma_max > 1.8 * classical,
            "quantum σ={sigma_max} not ≫ classical √T={classical}"
        );
    }

    // (c) Probability is conserved (Σ = 1) at every step.
    #[test]
    fn probability_conserved_each_step() {
        let walk = CoinedWalk::new(32).expect("valid CoinedWalk with 32 positions");
        let mut sv = walk
            .initial_state(10, CoinInit::Zero)
            .expect("valid initial state at position 10");
        for t in 0..30 {
            let dist = walk
                .position_distribution(&sv)
                .expect("position distribution should succeed on valid state");
            let total: f32 = dist.iter().sum();
            assert!((total - 1.0).abs() < 1e-4, "step {t}: Σp={total}");
            assert!((sv.norm_sq() - 1.0).abs() < 1e-4, "step {t}: norm");
            walk.step(&mut sv)
                .expect("step should succeed on valid state");
        }
    }

    // (d) The step operator U is unitary: it preserves inner products. Apply U to
    //     two distinct localized states and check ⟨Uψ|Uφ⟩ = ⟨ψ|φ⟩ (here both
    //     orthogonal → 0) and that each remains normalized; additionally verify on
    //     a generic superposition that the norm is preserved exactly.
    #[test]
    fn step_operator_is_unitary() {
        let walk = CoinedWalk::new(16).expect("valid CoinedWalk with 16 positions");
        // Orthogonal basis states stay orthonormal under U.
        let mut a = walk
            .initial_state(2, CoinInit::Zero)
            .expect("valid initial state at position 2");
        let mut b = walk
            .initial_state(5, CoinInit::One)
            .expect("valid initial state at position 5");
        let before = a
            .inner_product(&b)
            .expect("inner product between same-dimension states");
        walk.step(&mut a)
            .expect("step should succeed on valid state a");
        walk.step(&mut b)
            .expect("step should succeed on valid state b");
        let after = a
            .inner_product(&b)
            .expect("inner product between same-dimension states after step");
        assert!((before.norm()) < 1e-6, "states should start orthogonal");
        assert!(after.norm() < 1e-5, "U broke orthogonality: {after:?}");
        assert!((a.norm_sq() - 1.0).abs() < 1e-5);
        assert!((b.norm_sq() - 1.0).abs() < 1e-5);

        // Generic superposition: build a random normalized state on the walk's
        // register and confirm U preserves the norm over many steps.
        let mut rng = crate::handle::LcgRng::new(2026);
        let dim = 1usize << walk.n_qubits();
        let mut amps = Vec::with_capacity(dim);
        for _ in 0..dim {
            amps.push(Complex32::new(rng.next_normal(), rng.next_normal()));
        }
        let mut sv = StateVector {
            amps,
            n_qubits: walk.n_qubits(),
        };
        sv.normalize_inplace();
        for _ in 0..20 {
            walk.step(&mut sv)
                .expect("step should succeed on valid state");
            assert!((sv.norm_sq() - 1.0).abs() < 1e-4, "norm drift");
        }
    }

    // (e) The symmetric initial coin (|0⟩ + i|1⟩)/√2 yields a left–right symmetric
    //     distribution about x0.
    #[test]
    fn symmetric_coin_symmetric_distribution() {
        let walk = CoinedWalk::new(64).expect("valid CoinedWalk with 64 positions");
        let x0 = 32usize;
        let dist = walk
            .evolve(x0, CoinInit::SymmetricI, 18)
            .expect("evolve should succeed");
        let n = 64usize;
        for j in 1..20 {
            let right = dist[(x0 + j) % n];
            let left = dist[(x0 + n - j) % n];
            assert!(
                (right - left).abs() < 1e-5,
                "asymmetry at j={j}: right={right}, left={left}"
            );
        }
    }

    // (e-bis) The asymmetric coin |0⟩ produces a *biased* (non-symmetric)
    //         distribution — the contrast that makes (e) meaningful.
    #[test]
    fn zero_coin_is_biased() {
        let walk = CoinedWalk::new(64).expect("valid CoinedWalk with 64 positions");
        let x0 = 32usize;
        let dist = walk
            .evolve(x0, CoinInit::Zero, 18)
            .expect("evolve should succeed");
        let n = 64usize;
        // Mass to the right vs left of x0 should differ noticeably.
        let mut right_mass = 0.0_f32;
        let mut left_mass = 0.0_f32;
        for j in 1..30 {
            right_mass += dist[(x0 + j) % n];
            left_mass += dist[(x0 + n - j) % n];
        }
        assert!(
            (right_mass - left_mass).abs() > 0.1,
            "expected bias: right={right_mass}, left={left_mass}"
        );
    }

    // (f) On the smallest cycle the walk is periodic in time. For N = 2 both shift
    //     directions coincide (±1 mod 2 = NOT), so U = X_pos · H_coin and U² = I:
    //     the state — and hence the distribution — recurs exactly after 2 steps.
    #[test]
    fn small_cycle_periodic() {
        let walk = CoinedWalk::new(2).expect("valid CoinedWalk with 2 positions");
        let initial = walk
            .initial_state(0, CoinInit::Zero)
            .expect("valid initial state at position 0");
        let mut sv = initial.clone();
        walk.step(&mut sv).expect("first step should succeed");
        walk.step(&mut sv).expect("second step should succeed");
        // After 2 steps the full state returns to the initial state.
        for (a, b) in sv.amps.iter().zip(initial.amps.iter()) {
            assert!((a.re - b.re).abs() < 1e-5, "re {a:?} vs {b:?}");
            assert!((a.im - b.im).abs() < 1e-5, "im {a:?} vs {b:?}");
        }

        // The distribution is likewise periodic with period 2.
        let d0 = walk
            .position_distribution(&initial)
            .expect("position distribution of initial state should succeed");
        let d2 = walk
            .position_distribution(&sv)
            .expect("position distribution after 2 steps should succeed");
        for (p0, p2) in d0.iter().zip(d2.iter()) {
            assert!((p0 - p2).abs() < 1e-5, "dist not periodic: {p0} vs {p2}");
        }
    }

    // (f-bis) Cyclic translational covariance: walking from x0 equals walking from
    //         0 then cyclically shifting by x0 — the periodic/ring structure of the
    //         shift made explicit.
    #[test]
    fn cyclic_translation_covariance() {
        let walk = CoinedWalk::new(32).expect("valid CoinedWalk with 32 positions");
        let t = 10usize;
        let from_zero = walk
            .evolve(0, CoinInit::SymmetricI, t)
            .expect("evolve from position 0 should succeed");
        let x0 = 7usize;
        let from_x0 = walk
            .evolve(x0, CoinInit::SymmetricI, t)
            .expect("evolve from position x0 should succeed");
        let n = 32usize;
        for x in 0..n {
            let shifted = from_zero[(x + n - x0) % n];
            assert!(
                (from_x0[x] - shifted).abs() < 1e-5,
                "covariance broken at x={x}: {} vs {}",
                from_x0[x],
                shifted
            );
        }
    }

    // Construction validation.
    #[test]
    fn rejects_non_power_of_two() {
        assert!(CoinedWalk::new(6).is_err());
        assert!(CoinedWalk::new(0).is_err());
        assert!(CoinedWalk::new(1).is_err());
        assert!(CoinedWalk::new(8).is_ok());
    }

    #[test]
    fn rejects_out_of_range_start() {
        let walk = CoinedWalk::new(8).expect("valid CoinedWalk with 8 positions");
        assert!(walk.initial_state(8, CoinInit::Zero).is_err());
        assert!(walk.initial_state(0, CoinInit::Zero).is_ok());
    }
}
