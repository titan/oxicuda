//! Shor's period-finding subroutine.
//!
//! The quantum heart of Shor's factoring algorithm is *order finding*: given a
//! base `a` coprime to `N`, find the multiplicative order `r` — the least
//! positive integer with `a^r ≡ 1 (mod N)`. Quantum phase estimation on the
//! modular-multiplication unitary `U_a|y⟩ = |a·y mod N⟩` yields a phase
//! `φ ≈ k/r`, and the classical *continued-fraction* expansion of `φ` recovers
//! the denominator `r`. Once `r` is known (and even, with `a^{r/2} ≢ −1`), the
//! factors of `N` are `gcd(a^{r/2} ± 1, N)`.
//!
//! This module provides:
//! * [`mod_exp`] — fast modular exponentiation by repeated squaring.
//! * [`gcd`] — binary Euclidean greatest common divisor.
//! * [`continued_fraction_convergents`] — the convergents `p_i/q_i` of a real.
//! * [`order_finding`] — state-vector phase estimation of `U_a` plus classical
//!   continued-fraction post-processing.
//! * [`shor_factor`] — the end-to-end reduction from `r` to a non-trivial factor
//!   of `N`.
//!
//! The counting register is **little-endian** (qubit `i` carries weight `2^i`),
//! matching [`crate::fourier::qft`].

use crate::error::{QuantumError, QuantumResult};
use crate::fourier::qft::qft_inverse_inplace;
use crate::gates::hadamard::gate_h;
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;

/// Fast modular exponentiation `base^exp mod modulus` via repeated squaring.
///
/// Uses `u128` intermediates so products of two `u64`-range operands cannot
/// overflow. Returns `0` for `modulus == 0` (degenerate) and handles
/// `modulus == 1` (everything is `0`).
#[must_use]
pub fn mod_exp(base: u64, exp: u64, modulus: u64) -> u64 {
    if modulus <= 1 {
        return 0;
    }
    let mut result: u128 = 1;
    let mut b = (base % modulus) as u128;
    let m = modulus as u128;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = (result * b) % m;
        }
        b = (b * b) % m;
        e >>= 1;
    }
    result as u64
}

/// Binary-Euclidean greatest common divisor of `a` and `b`.
#[must_use]
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            break;
        }
    }
    a << shift
}

/// Compute the continued-fraction convergents `(p_i, q_i)` of `value ∈ [0, 1)`,
/// stopping once a denominator exceeds `max_denominator` or the expansion
/// terminates.
///
/// Each returned pair satisfies `p_i / q_i ≈ value`, with successively better
/// rational approximations. The denominators are the candidate periods in
/// Shor's post-processing.
#[must_use]
pub fn continued_fraction_convergents(value: f64, max_denominator: u64) -> Vec<(u64, u64)> {
    let mut convergents = Vec::new();
    let mut x = value;
    // Convergent recurrence h_n = a_n·h_{n-1} + h_{n-2}, k_n = a_n·k_{n-1} + k_{n-2}
    // seeded with h_{-1}=1, h_{-2}=0 and k_{-1}=0, k_{-2}=1.
    let mut p_minus2: i128 = 0; // h_{-2}
    let mut p_minus1: i128 = 1; // h_{-1}
    let mut q_minus2: i128 = 1; // k_{-2}
    let mut q_minus1: i128 = 0; // k_{-1}
    for _ in 0..64 {
        let a = x.floor();
        if !a.is_finite() {
            break;
        }
        let a_i = a as i128;
        let p_next = a_i * p_minus1 + p_minus2;
        let q_next = a_i * q_minus1 + q_minus2;
        if q_next <= 0 || q_next as u64 > max_denominator {
            break;
        }
        convergents.push((p_next.max(0) as u64, q_next as u64));
        p_minus2 = p_minus1;
        p_minus1 = p_next;
        q_minus2 = q_minus1;
        q_minus1 = q_next;
        let frac = x - a;
        if frac.abs() < 1e-12 {
            break;
        }
        x = 1.0 / frac;
    }
    convergents
}

/// Outcome of [`order_finding`].
#[derive(Debug, Clone)]
pub struct OrderFindingResult {
    /// The recovered period `r` such that `a^r ≡ 1 (mod N)` (or `0` if not found).
    pub order: u64,
    /// The raw phase `φ ≈ k/r ∈ [0, 1)` read from the counting register.
    pub phase: f64,
    /// The integer measured in the counting register (argmax readout).
    pub measured: usize,
    /// The number of counting qubits used.
    pub counting_qubits: usize,
}

/// Brute-force the multiplicative order of `a` modulo `n` (classical reference).
///
/// Returns the least `r ≥ 1` with `a^r ≡ 1 (mod n)`, or `0` if `gcd(a, n) ≠ 1`
/// or no order is found within `n` steps.
#[must_use]
pub fn classical_order(a: u64, n: u64) -> u64 {
    if n <= 1 || gcd(a, n) != 1 {
        return 0;
    }
    let mut cur = a % n;
    let mut r = 1u64;
    while cur != 1 {
        cur = (cur as u128 * (a % n) as u128 % n as u128) as u64;
        r += 1;
        if r > n {
            return 0;
        }
    }
    r
}

/// Quantum order finding for the unitary `U_a|y⟩ = |a·y mod N⟩`.
///
/// Runs phase estimation with `counting_qubits` in the counting register and a
/// `⌈log2 N⌉`-qubit work register initialised to `|1⟩`. The eigenphases of `U_a`
/// are `k/r`, so the argmax readout `φ ≈ k/r` is fed through
/// [`continued_fraction_convergents`] to extract the period `r` (verified by
/// `a^r ≡ 1`). The modular-multiplication ladder is applied **exactly** on the
/// amplitude array (controlled `U_a^{2^j}` = multiply by `a^{2^j} mod N`), so no
/// gate-level modular-arithmetic synthesis is required.
///
/// # Errors
/// * [`QuantumError::InvalidParameter`] if `modulus < 2`, `base % modulus` is
///   not coprime to `modulus`, or the registers exceed 30 qubits combined.
/// * [`QuantumError::InvalidQubitCount`] if `counting_qubits == 0`.
pub fn order_finding(
    base: u64,
    modulus: u64,
    counting_qubits: usize,
) -> QuantumResult<OrderFindingResult> {
    if modulus < 2 {
        return Err(QuantumError::InvalidParameter {
            name: format!("modulus {modulus} must be ≥ 2"),
        });
    }
    if counting_qubits == 0 {
        return Err(QuantumError::InvalidQubitCount { n: 0 });
    }
    let a = base % modulus;
    if gcd(a, modulus) != 1 {
        return Err(QuantumError::InvalidParameter {
            name: format!("base {base} not coprime to modulus {modulus}"),
        });
    }
    // Work register must index 0..modulus.
    let work_qubits = {
        let mut bits = 0usize;
        let mut cap = 1u64;
        while cap < modulus {
            cap <<= 1;
            bits += 1;
        }
        bits.max(1)
    };
    let total = counting_qubits + work_qubits;
    if total > 30 {
        return Err(QuantumError::InvalidParameter {
            name: format!("total qubits {total} exceeds 30"),
        });
    }

    let mut sv = StateVector::new_zero_state(total)?;
    // Initialise work register (qubits counting..total) to |1⟩.
    let work_one = 1usize << counting_qubits;
    sv.amps[0] = num_complex::Complex::new(0.0, 0.0);
    sv.amps[work_one] = num_complex::Complex::new(1.0, 0.0);

    // Hadamard the counting register (qubits 0..counting_qubits).
    for q in 0..counting_qubits {
        apply_1q_inplace(&mut sv, q, &gate_h())?;
    }

    // Controlled-U_a^{2^j}: for each counting qubit j, if it is set, multiply the
    // work register value by a^{2^j} mod N. Done exactly via amplitude permutation.
    let count_mask = (1usize << counting_qubits) - 1;
    for j in 0..counting_qubits {
        let power = mod_exp(a, 1u64 << j, modulus);
        if power == 1 {
            continue; // identity permutation
        }
        let ctrl_bit = 1usize << j;
        let dim = 1usize << total;
        let mut new_amps = vec![num_complex::Complex::<f32>::new(0.0, 0.0); dim];
        for (idx, amp) in sv.amps.iter().enumerate() {
            if amp.norm_sqr() == 0.0 {
                continue;
            }
            if idx & ctrl_bit == 0 {
                new_amps[idx] += *amp;
                continue;
            }
            let work_val = (idx >> counting_qubits) as u64;
            let count_part = idx & count_mask;
            let new_work = if work_val < modulus {
                (work_val as u128 * power as u128 % modulus as u128) as usize
            } else {
                work_val as usize // values ≥ N are inert (never populated)
            };
            let target = count_part | (new_work << counting_qubits);
            new_amps[target] += *amp;
        }
        sv.amps = new_amps;
    }

    // Inverse QFT on the counting register to read the phase.
    let counting_indices: Vec<usize> = (0..counting_qubits).collect();
    qft_inverse_inplace(&mut sv, &counting_indices)?;

    // Marginalise the work register to obtain the counting-register distribution.
    let dim_count = 1usize << counting_qubits;
    let mut marg = vec![0.0f32; dim_count];
    for (idx, amp) in sv.amps.iter().enumerate() {
        marg[idx & count_mask] += amp.norm_sqr();
    }

    // The textbook readout argmax: the single most probable basis index.
    let mut measured = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &p) in marg.iter().enumerate() {
        if p > best {
            best = p;
            measured = i;
        }
    }
    let phase = measured as f64 / dim_count as f64;

    // Continued-fraction post-processing. A single QPE shot returns a uniformly
    // random peak `≈ k·2ⁿ/r`; the useless `k = 0` peak (and any peak whose
    // convergents fail the `a^q ≡ 1` check) carries no period information. To
    // make the routine deterministic we scan the significant peaks in order of
    // decreasing probability and accept the first whose denominator is a genuine
    // order — exactly the value a repeated run would eventually sample.
    let threshold = 0.5f32 / dim_count as f32; // ignore numerical dust
    let mut peaks: Vec<(usize, f32)> = marg
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p > threshold)
        .map(|(i, &p)| (i, p))
        .collect();
    peaks.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    let mut order = 0u64;
    'scan: for &(idx, _) in &peaks {
        if idx == 0 {
            continue; // k = 0 ⇒ φ = 0 carries no period
        }
        let phi = idx as f64 / dim_count as f64;
        for (_, q) in continued_fraction_convergents(phi, modulus) {
            if q == 0 {
                continue;
            }
            if mod_exp(a, q, modulus) == 1 {
                order = q;
                break 'scan;
            }
        }
    }

    Ok(OrderFindingResult {
        order,
        phase,
        measured,
        counting_qubits,
    })
}

/// Outcome of [`shor_factor`].
#[derive(Debug, Clone)]
pub struct ShorResult {
    /// A non-trivial factor of `N` (with `1 < factor < N`), if one was found.
    pub factor: Option<u64>,
    /// The base `a` that succeeded (or the last one tried).
    pub base: u64,
    /// The order `r` of `a` modulo `N` that produced the factor.
    pub order: u64,
}

/// Classical reduction from a period `r` (order of `base` mod `n`) to a factor.
///
/// Implements the textbook step: if `r` is even and `a^{r/2} ≢ ±1 (mod n)`, then
/// `gcd(a^{r/2} − 1, n)` and `gcd(a^{r/2} + 1, n)` are non-trivial factors.
/// Returns the first non-trivial factor found, else `None`.
#[must_use]
pub fn factor_from_order(base: u64, n: u64, r: u64) -> Option<u64> {
    if r == 0 || !r.is_multiple_of(2) || n < 2 {
        return None;
    }
    let half = mod_exp(base, r / 2, n);
    if half == 1 || half == n - 1 {
        return None;
    }
    [gcd(half + 1, n), gcd(half.wrapping_sub(1), n)]
        .into_iter()
        .find(|&candidate| candidate > 1 && candidate < n)
}

/// End-to-end Shor factoring driver using quantum order finding.
///
/// Tries the supplied candidate bases in turn; for each coprime base it runs
/// [`order_finding`] and attempts [`factor_from_order`]. Returns as soon as a
/// non-trivial factor is found. Even `N` short-circuits to the factor `2`, and a
/// base sharing a common factor with `N` yields that factor directly via `gcd`.
///
/// # Errors
/// * [`QuantumError::InvalidParameter`] if `n < 2` or `bases` is empty.
pub fn shor_factor(n: u64, bases: &[u64], counting_qubits: usize) -> QuantumResult<ShorResult> {
    if n < 2 {
        return Err(QuantumError::InvalidParameter {
            name: format!("N {n} must be ≥ 2"),
        });
    }
    if bases.is_empty() {
        return Err(QuantumError::EmptyInput);
    }
    if n.is_multiple_of(2) {
        return Ok(ShorResult {
            factor: Some(2),
            base: 2,
            order: 0,
        });
    }
    let mut last_base = bases[0];
    for &a in bases {
        let a = a % n;
        if a < 2 {
            continue;
        }
        last_base = a;
        let g = gcd(a, n);
        if g > 1 {
            // Lucky: a shares a factor with N.
            return Ok(ShorResult {
                factor: Some(g),
                base: a,
                order: 0,
            });
        }
        let of = order_finding(a, n, counting_qubits)?;
        if of.order == 0 {
            continue;
        }
        if let Some(factor) = factor_from_order(a, n, of.order) {
            return Ok(ShorResult {
                factor: Some(factor),
                base: a,
                order: of.order,
            });
        }
    }
    Ok(ShorResult {
        factor: None,
        base: last_base,
        order: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mod_exp_basic() {
        assert_eq!(mod_exp(2, 10, 1000), 24); // 1024 mod 1000
        assert_eq!(mod_exp(7, 0, 13), 1);
        assert_eq!(mod_exp(3, 3, 7), 6); // 27 mod 7
    }

    #[test]
    fn mod_exp_large_no_overflow() {
        // 2^62 mod (2^61 - 1): exercise u128 path.
        let m = (1u64 << 61) - 1;
        let v = mod_exp(2, 62, m);
        assert!(v < m);
        // 2^62 = 2 * 2^61 ≡ 2 * 1 = 2 (mod 2^61 - 1).
        assert_eq!(v, 2);
    }

    #[test]
    fn gcd_basic() {
        assert_eq!(gcd(54, 24), 6);
        assert_eq!(gcd(17, 13), 1);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(8, 0), 8);
    }

    #[test]
    fn continued_fraction_recovers_denominator() {
        // φ = 1/4 ⇒ convergents include (1, 4).
        let cs = continued_fraction_convergents(0.25, 100);
        assert!(cs.iter().any(|&(_, q)| q == 4), "convergents={cs:?}");
    }

    #[test]
    fn continued_fraction_for_third() {
        let cs = continued_fraction_convergents(1.0 / 3.0, 100);
        assert!(cs.iter().any(|&(p, q)| p == 1 && q == 3), "cs={cs:?}");
    }

    #[test]
    fn classical_order_matches_definition() {
        // order of 2 mod 15: 2,4,8,16≡1 ⇒ r=4.
        assert_eq!(classical_order(2, 15), 4);
        // order of 7 mod 15: 7,49≡4,28≡13,91≡1 ⇒ r=4.
        assert_eq!(classical_order(7, 15), 4);
        // not coprime.
        assert_eq!(classical_order(6, 15), 0);
    }

    #[test]
    fn order_finding_period_of_2_mod_15() {
        // a=2, N=15, true order 4. 8 counting qubits resolves 1/4 exactly.
        let res = order_finding(2, 15, 8)
            .expect("a=2 coprime to N=15, 8 counting qubits within valid range");
        assert_eq!(
            res.order, 4,
            "phase={}, measured={}",
            res.phase, res.measured
        );
    }

    #[test]
    fn order_finding_period_of_4_mod_15() {
        // a=4, N=15: 4,16≡1 ⇒ order 2.
        let res = order_finding(4, 15, 8)
            .expect("a=4 coprime to N=15, 8 counting qubits within valid range");
        assert_eq!(res.order, 2);
    }

    #[test]
    fn order_finding_period_of_2_mod_21() {
        // order of 2 mod 21: 2,4,8,16,32≡11,22≡1 ⇒ r=6.
        let res = order_finding(2, 21, 9)
            .expect("a=2 coprime to N=21, 9 counting qubits within valid range");
        // Phase estimation may land on a convergent of k/6; accept any divisor
        // that genuinely satisfies a^r ≡ 1.
        assert!(res.order == 6 || (res.order > 0 && mod_exp(2, res.order, 21) == 1));
    }

    #[test]
    fn order_finding_rejects_non_coprime() {
        assert!(order_finding(3, 15, 6).is_err()); // gcd(3,15)=3
    }

    #[test]
    fn order_finding_rejects_bad_modulus() {
        assert!(order_finding(2, 1, 6).is_err());
        assert!(order_finding(2, 15, 0).is_err());
    }

    #[test]
    fn factor_from_order_factors_15() {
        // order of 7 mod 15 is 4; 7^2 = 49 ≡ 4 (mod 15); gcd(4±1,15) = {5,3}.
        let f = factor_from_order(7, 15, 4)
            .expect("order 4 of base 7 mod 15 is even and yields a non-trivial factor");
        assert!(f == 3 || f == 5, "f={f}");
    }

    #[test]
    fn factor_from_order_rejects_odd_order() {
        assert_eq!(factor_from_order(2, 7, 3), None);
    }

    #[test]
    fn shor_factor_even_shortcut() {
        let res = shor_factor(8, &[3, 5, 7], 6)
            .expect("N=8 is even so shor_factor short-circuits to factor 2");
        assert_eq!(res.factor, Some(2));
    }

    #[test]
    fn shor_factor_fifteen() {
        // N=15 should factor into 3 or 5 via order finding.
        let res = shor_factor(15, &[2, 7, 11, 13], 8)
            .expect("N=15 with coprime bases and 8 counting qubits should succeed");
        let f = res.factor.expect("should find a factor");
        assert!(f == 3 || f == 5, "factor={f}");
        assert_eq!(15 % f, 0);
    }

    #[test]
    fn shor_factor_lucky_gcd() {
        // base 5 shares factor with 15 ⇒ returns 5 immediately.
        let res = shor_factor(15, &[5], 6)
            .expect("base=5 shares factor 5 with N=15 so shor_factor succeeds immediately");
        assert_eq!(res.factor, Some(5));
    }

    #[test]
    fn shor_factor_empty_bases_errors() {
        assert!(shor_factor(15, &[], 6).is_err());
    }

    #[test]
    fn shor_factor_invalid_n_errors() {
        assert!(shor_factor(1, &[2], 6).is_err());
    }
}
