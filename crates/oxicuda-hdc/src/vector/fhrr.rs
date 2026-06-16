//! Phasor-only Fourier Holographic Reduced Representation (FHRR) hypervectors.
//!
//! The complex FHRR model in [`crate::vector::complex`] stores each component as an explicit
//! interleaved `(re, im)` pair and re-normalises after bundling. This module instead stores
//! the *phase angle only* (`Vec<f32>` of length `D`, each value in `[0, 2π)`), which makes the
//! unit-magnitude constraint exact and free by construction (Plate 2003; Frady, Kleyko &
//! Sommer 2018). All operations act directly on phases:
//!
//! - **Binding** (`fhrr_bind`) is element-wise phase addition modulo `2π`
//!   (complex multiplication of unit phasors): `θ_c = (θ_a + θ_b) mod 2π`.
//! - **Unbinding** (`fhrr_unbind`) is phase subtraction: `θ_c = (θ_a − θ_b) mod 2π`. Binding
//!   is its own inverse only up to the inverse element; `fhrr_unbind(bind(a, b), b) == a`.
//! - **Bundling** (`fhrr_bundle`) converts each phasor to Cartesian form, sums, and takes the
//!   resulting argument (circular mean of angles). The result remains a unit phasor per
//!   component, preserving the explicit magnitude constraint.
//! - **Permutation binding** is the cyclic shift of the phase vector (re-exported via the
//!   ordinary permutation ops on the underlying `f32` slice).
//! - **Similarity** (`fhrr_cosine`) is the mean cosine of phase differences,
//!   `(1/D) Σ cos(θ_a − θ_b)`, which equals the real part of the normalised Hermitian inner
//!   product and lies in `[−1, 1]`.
//!
//! The phase representation never accumulates magnitude drift, so repeated bind/unbind chains
//! stay exactly on the unit torus.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;

use std::f32::consts::TAU;

/// Wrap a phase angle into the canonical range `[0, 2π)`.
#[inline]
fn wrap_phase(theta: f32) -> f32 {
    let mut t = theta % TAU;
    if t < 0.0 {
        t += TAU;
    }
    // Guard the exact-`TAU` boundary produced by rounding.
    if t >= TAU {
        t -= TAU;
    }
    t
}

/// Generate a random phasor-only FHRR hypervector: `D` independent uniform phases in `[0, 2π)`.
///
/// # Errors
///
/// - [`HdcError::ZeroDimension`] if `dim == 0`.
pub fn random_fhrr(dim: usize, rng: &mut LcgRng) -> HdcResult<Vec<f32>> {
    if dim == 0 {
        return Err(HdcError::ZeroDimension);
    }
    let mut v = vec![0f32; dim];
    for p in v.iter_mut() {
        *p = rng.next_f32() * TAU;
    }
    Ok(v)
}

/// Validate that every phase lies in `[0, 2π)` (within a small tolerance) and is finite.
///
/// # Errors
///
/// - [`HdcError::EmptyInput`] if `phases` is empty.
/// - [`HdcError::InvalidProbability`] (reused) if any phase is non-finite or out of range.
pub fn validate_fhrr(phases: &[f32]) -> HdcResult<()> {
    if phases.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    for &p in phases {
        if !p.is_finite() || !(-1e-4..TAU + 1e-4).contains(&p) {
            return Err(HdcError::InvalidProbability(p as f64));
        }
    }
    Ok(())
}

/// Phasor-only binding: element-wise phase addition modulo `2π`.
///
/// # Errors
///
/// - [`HdcError::EmptyInput`] if either input is empty.
/// - [`HdcError::DimensionMismatch`] if lengths differ.
pub fn fhrr_bind(a: &[f32], b: &[f32]) -> HdcResult<Vec<f32>> {
    if a.is_empty() || b.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(&pa, &pb)| wrap_phase(pa + pb))
        .collect())
}

/// Phasor-only unbinding: element-wise phase subtraction modulo `2π`.
///
/// Satisfies `fhrr_unbind(fhrr_bind(a, b), b) == a` (exactly, up to phase wrapping).
///
/// # Errors
///
/// - [`HdcError::EmptyInput`] if either input is empty.
/// - [`HdcError::DimensionMismatch`] if lengths differ.
pub fn fhrr_unbind(bound: &[f32], b: &[f32]) -> HdcResult<Vec<f32>> {
    if bound.is_empty() || b.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    if bound.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: bound.len(),
            got: b.len(),
        });
    }
    Ok(bound
        .iter()
        .zip(b.iter())
        .map(|(&pc, &pb)| wrap_phase(pc - pb))
        .collect())
}

/// The inverse phasor of a hypervector under binding: negated phases.
///
/// `fhrr_bind(a, fhrr_inverse(a))` is the all-zero-phase identity vector.
///
/// # Errors
///
/// - [`HdcError::EmptyInput`] if `phases` is empty.
pub fn fhrr_inverse(phases: &[f32]) -> HdcResult<Vec<f32>> {
    if phases.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    Ok(phases.iter().map(|&p| wrap_phase(-p)).collect())
}

/// Phasor-only bundling: per-component circular mean of phases.
///
/// Each phasor is mapped to Cartesian coordinates, summed, and the argument of the resultant
/// is taken. If the resultant magnitude is (numerically) zero for some component, that phase
/// defaults to `0`. The output is always a valid unit-phasor hypervector.
///
/// # Errors
///
/// - [`HdcError::EmptyInput`] if `hvs` is empty.
/// - [`HdcError::DimensionMismatch`] if the inputs have differing dimensions.
pub fn fhrr_bundle(hvs: &[Vec<f32>]) -> HdcResult<Vec<f32>> {
    if hvs.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let dim = hvs[0].len();
    if dim == 0 {
        return Err(HdcError::ZeroDimension);
    }
    for hv in hvs.iter().skip(1) {
        if hv.len() != dim {
            return Err(HdcError::DimensionMismatch {
                expected: dim,
                got: hv.len(),
            });
        }
    }
    let mut re = vec![0f32; dim];
    let mut im = vec![0f32; dim];
    for hv in hvs {
        for ((r, i), &p) in re.iter_mut().zip(im.iter_mut()).zip(hv.iter()) {
            *r += p.cos();
            *i += p.sin();
        }
    }
    let mut out = vec![0f32; dim];
    for ((o, &r), &i) in out.iter_mut().zip(re.iter()).zip(im.iter()) {
        if r.abs() < f32::EPSILON && i.abs() < f32::EPSILON {
            *o = 0.0;
        } else {
            *o = wrap_phase(i.atan2(r));
        }
    }
    Ok(out)
}

/// Phasor-only similarity: mean cosine of phase differences, in `[−1, 1]`.
///
/// Equals `(1/D) Σ cos(θ_a − θ_b)`, the real part of the normalised Hermitian inner product
/// of the two unit-phasor vectors.
///
/// # Errors
///
/// - [`HdcError::EmptyInput`] if either input is empty.
/// - [`HdcError::DimensionMismatch`] if lengths differ.
pub fn fhrr_cosine(a: &[f32], b: &[f32]) -> HdcResult<f32> {
    if a.is_empty() || b.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let dim = a.len();
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&pa, &pb)| (pa - pb).cos())
        .sum();
    Ok(sum / dim as f32)
}

/// Convert a phasor-only hypervector to the interleaved `[re, im]` representation used by
/// [`crate::vector::complex`], for interoperability.
///
/// # Errors
///
/// - [`HdcError::EmptyInput`] if `phases` is empty.
pub fn fhrr_to_interleaved(phases: &[f32]) -> HdcResult<Vec<f32>> {
    if phases.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let mut out = vec![0f32; 2 * phases.len()];
    for (i, &p) in phases.iter().enumerate() {
        out[2 * i] = p.cos();
        out[2 * i + 1] = p.sin();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rng() -> LcgRng {
        LcgRng::new(0xFADE_F44D_0001)
    }

    #[test]
    fn random_fhrr_in_range() {
        let mut r = rng();
        let hv = random_fhrr(512, &mut r).expect("random");
        assert_eq!(hv.len(), 512);
        validate_fhrr(&hv).expect("validate");
        for &p in &hv {
            assert!((0.0..TAU).contains(&p), "phase {p} out of range");
        }
    }

    #[test]
    fn random_fhrr_zero_dim_errors() {
        let mut r = rng();
        assert!(matches!(
            random_fhrr(0, &mut r),
            Err(HdcError::ZeroDimension)
        ));
    }

    #[test]
    fn bind_unbind_recovers() {
        let mut r = rng();
        let a = random_fhrr(256, &mut r).expect("a");
        let b = random_fhrr(256, &mut r).expect("b");
        let bound = fhrr_bind(&a, &b).expect("bind");
        let recovered = fhrr_unbind(&bound, &b).expect("unbind");
        // Phase-exact recovery (within float rounding).
        for (&orig, &rec) in a.iter().zip(recovered.iter()) {
            let diff = (orig - rec).cos();
            assert!(diff > 0.9999, "phase mismatch: {orig} vs {rec}");
        }
        let sim = fhrr_cosine(&a, &recovered).expect("cosine");
        assert!(sim > 0.999, "sim={sim}");
    }

    #[test]
    fn inverse_yields_identity() {
        let mut r = rng();
        let a = random_fhrr(128, &mut r).expect("a");
        let inv = fhrr_inverse(&a).expect("inverse");
        let id = fhrr_bind(&a, &inv).expect("bind");
        for &p in &id {
            // identity has near-zero phase (cos ≈ 1).
            assert!(p.cos() > 0.9999, "non-identity phase {p}");
        }
    }

    #[test]
    fn self_similarity_is_one() {
        let mut r = rng();
        let a = random_fhrr(300, &mut r).expect("a");
        let sim = fhrr_cosine(&a, &a).expect("cosine");
        assert!((sim - 1.0).abs() < 1e-5, "sim={sim}");
    }

    #[test]
    fn orthogonal_random_near_zero() {
        let mut r = rng();
        let a = random_fhrr(8000, &mut r).expect("a");
        let b = random_fhrr(8000, &mut r).expect("b");
        let sim = fhrr_cosine(&a, &b).expect("cosine");
        assert!(
            sim.abs() < 0.1,
            "random phasors not quasi-orthogonal: {sim}"
        );
    }

    #[test]
    fn bundle_similar_to_inputs() {
        let mut r = rng();
        let a = random_fhrr(2000, &mut r).expect("a");
        let b = random_fhrr(2000, &mut r).expect("b");
        let c = random_fhrr(2000, &mut r).expect("c");
        let bundled = fhrr_bundle(&[a.clone(), b.clone(), c.clone()]).expect("bundle");
        let sa = fhrr_cosine(&bundled, &a).expect("sa");
        let sb = fhrr_cosine(&bundled, &b).expect("sb");
        let sc = fhrr_cosine(&bundled, &c).expect("sc");
        // Each member should be clearly more similar to the bundle than a random vector.
        assert!(sa > 0.2 && sb > 0.2 && sc > 0.2, "sa={sa} sb={sb} sc={sc}");
    }

    #[test]
    fn bundle_preserves_dim_and_unit() {
        let mut r = rng();
        let a = random_fhrr(64, &mut r).expect("a");
        let b = random_fhrr(64, &mut r).expect("b");
        let bundled = fhrr_bundle(&[a, b]).expect("bundle");
        assert_eq!(bundled.len(), 64);
        validate_fhrr(&bundled).expect("bundle is valid phasor");
    }

    #[test]
    fn bind_commutative() {
        let mut r = rng();
        let a = random_fhrr(200, &mut r).expect("a");
        let b = random_fhrr(200, &mut r).expect("b");
        let ab = fhrr_bind(&a, &b).expect("ab");
        let ba = fhrr_bind(&b, &a).expect("ba");
        for (&x, &y) in ab.iter().zip(ba.iter()) {
            assert!((x - y).abs() < 1e-4, "{x} != {y}");
        }
    }

    #[test]
    fn bind_dimension_mismatch_errors() {
        let mut r = rng();
        let a = random_fhrr(64, &mut r).expect("a");
        let b = random_fhrr(32, &mut r).expect("b");
        assert!(matches!(
            fhrr_bind(&a, &b),
            Err(HdcError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            fhrr_cosine(&a, &b),
            Err(HdcError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            fhrr_unbind(&a, &b),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn to_interleaved_unit_magnitude() {
        let mut r = rng();
        let a = random_fhrr(100, &mut r).expect("a");
        let inter = fhrr_to_interleaved(&a).expect("interleaved");
        assert_eq!(inter.len(), 200);
        for i in 0..100 {
            let mag = (inter[2 * i].powi(2) + inter[2 * i + 1].powi(2)).sqrt();
            assert!((mag - 1.0).abs() < 1e-5, "component {i} mag={mag}");
        }
    }

    #[test]
    fn wrap_phase_canonical_range() {
        // Every wrapped phase must lie in [0, TAU), regardless of input sign/magnitude.
        for &theta in &[0.0f32, -0.5, TAU + 0.1, 3.0 * TAU, -10.0 * TAU, 100.0] {
            let w = wrap_phase(theta);
            assert!(
                (0.0..TAU).contains(&w),
                "wrap_phase({theta}) = {w} out of range"
            );
        }
        // A small positive input is preserved exactly.
        assert!((wrap_phase(0.3) - 0.3).abs() < 1e-6);
        // -0.5 wraps to TAU - 0.5.
        assert!((wrap_phase(-0.5) - (TAU - 0.5)).abs() < 1e-4);
    }

    #[test]
    fn bundle_dimension_mismatch_errors() {
        let mut r = rng();
        let a = random_fhrr(64, &mut r).expect("a");
        let b = random_fhrr(32, &mut r).expect("b");
        assert!(matches!(
            fhrr_bundle(&[a, b]),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }
}
