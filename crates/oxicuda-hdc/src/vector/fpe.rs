//! Fractional Power Encoding (FPE) for continuous values.
//!
//! References: T. A. Plate, "Distributed Representations and Nested Compositional Structure"
//! (PhD thesis, 1994); E. P. Frady, D. Kleyko & F. T. Sommer, "Computing on Functions Using
//! Randomized Vector Representations" (2021); B. Komer & C. Eliasmith, "Spatial Semantic
//! Pointers" / fractional binding.
//!
//! FPE encodes a *continuous* scalar (or an N-dimensional vector) into a unit-magnitude
//! Fourier Holographic Reduced Representation (FHRR) hypervector by raising a fixed random
//! *base* hypervector to a fractional (real) power. In the FHRR / phasor domain each component
//! is a unit phasor `e^{iφ_k}`, and raising the whole vector to a power `v` multiplies every
//! phase by `v`:
//!
//! ```text
//! base_k        = e^{iφ_k}
//! encode(v)_k   = base_k^v = e^{i·v·φ_k}   ⇒  phase = v · φ_k
//! ```
//!
//! Binding (circular convolution) is phase addition, so `encode(v) = base^v` is a *fractional
//! binding* and `bind(encode(a), encode(b)) = encode(a + b)`. The similarity between two
//! encodings is a smooth, locality-preserving kernel of `|v₁ − v₂|`: it peaks at `1.0` when the
//! values coincide and decays (sinc-like) as the values separate.
//!
//! Hypervectors are returned in the crate's interleaved FHRR layout
//! `[re₀, im₀, re₁, im₁, …]` of length `2·D`, where `D` is the number of complex components,
//! matching [`crate::vector::fhrr::fhrr_to_interleaved`].

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;

use std::f32::consts::PI;

/// A fixed set of `D` base phases defining a fractional-power encoder.
///
/// Each stored phase `φ_k ∈ (−π, π]` is one phasor of the base hypervector. Encoding a value
/// `v` scales every phase by `v`.
#[derive(Debug, Clone)]
pub struct FpeBase {
    /// The `D` base phase angles, one per complex component.
    phases: Vec<f32>,
}

impl FpeBase {
    /// Draw a random base of `dim` complex components, with phases uniform in `(−π, π]`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::ZeroDimension`] if `dim == 0`.
    pub fn random(dim: usize, rng: &mut LcgRng) -> HdcResult<Self> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        let mut phases = vec![0f32; dim];
        for p in phases.iter_mut() {
            // next_f32() ∈ [0, 1) → map to (−π, π]: −π + u·2π lands in [−π, π); shift to (−π, π].
            *p = (rng.next_f32() * 2.0 - 1.0) * PI;
            if *p <= -PI {
                *p += 2.0 * PI;
            }
        }
        Ok(Self { phases })
    }

    /// The number of complex components `D`.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.phases.len()
    }

    /// The base phase angles.
    #[must_use]
    pub fn phases(&self) -> &[f32] {
        &self.phases
    }

    /// Encode a scalar `value` as an interleaved FHRR hypervector of length `2·D`.
    ///
    /// Component `k` is `(cos(value·φ_k), sin(value·φ_k))`, the base phasor raised to the
    /// `value` power.
    #[must_use]
    pub fn encode(&self, value: f32) -> Vec<f32> {
        let mut out = vec![0f32; 2 * self.phases.len()];
        for (k, &phi) in self.phases.iter().enumerate() {
            let theta = value * phi;
            out[2 * k] = theta.cos();
            out[2 * k + 1] = theta.sin();
        }
        out
    }

    /// Encode an N-dimensional `values` vector by binding the per-axis encodings.
    ///
    /// Each axis contributes its scaled base phases; binding is phase addition, so the output
    /// phase for component `k` is `Σ_axis values[axis] · bases[axis].φ_k`. All bases must share
    /// the same dimension `D`, and the result is an interleaved FHRR hypervector of length
    /// `2·D`.
    ///
    /// This is an associated function: it does not use any single base's phases directly but
    /// the supplied per-axis `bases`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::EmptyInput`] if `values` (or `bases`) is empty.
    /// - [`HdcError::DimensionMismatch`] if `values.len() != bases.len()`, or the bases do not
    ///   all share one dimension.
    pub fn encode_nd(values: &[f32], bases: &[FpeBase]) -> HdcResult<Vec<f32>> {
        if values.is_empty() || bases.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        if values.len() != bases.len() {
            return Err(HdcError::DimensionMismatch {
                expected: values.len(),
                got: bases.len(),
            });
        }
        let dim = bases[0].dim();
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        for base in bases.iter().skip(1) {
            if base.dim() != dim {
                return Err(HdcError::DimensionMismatch {
                    expected: dim,
                    got: base.dim(),
                });
            }
        }
        // Accumulate the combined phase per component, then convert to interleaved Cartesian.
        let mut combined = vec![0f64; dim];
        for (&v, base) in values.iter().zip(bases.iter()) {
            let vf = v as f64;
            for (slot, &phi) in combined.iter_mut().zip(base.phases.iter()) {
                *slot += vf * (phi as f64);
            }
        }
        let mut out = vec![0f32; 2 * dim];
        for (k, &theta) in combined.iter().enumerate() {
            out[2 * k] = theta.cos() as f32;
            out[2 * k + 1] = theta.sin() as f32;
        }
        Ok(out)
    }
}

/// Similarity between two interleaved FHRR hypervectors: `Re(Σ a · conj(b)) / D`.
///
/// For unit-magnitude phasors this is the mean cosine of phase differences, in `[−1, 1]`. Both
/// inputs must have the same even length `2·D`.
///
/// # Errors
///
/// - [`HdcError::EmptyInput`] if either input is empty.
/// - [`HdcError::DimensionMismatch`] if lengths differ, or the (shared) length is odd (the
///   interleaved layout requires an even length).
pub fn fpe_similarity(a: &[f32], b: &[f32]) -> HdcResult<f32> {
    if a.is_empty() || b.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if !a.len().is_multiple_of(2) {
        // Interleaved layout requires an even length.
        return Err(HdcError::DimensionMismatch {
            expected: a.len() - 1,
            got: a.len(),
        });
    }
    let dim = a.len() / 2;
    // Re(a · conj(b)) = Σ (a_re·b_re + a_im·b_im).
    let mut acc = 0f64;
    for k in 0..dim {
        let a_re = a[2 * k] as f64;
        let a_im = a[2 * k + 1] as f64;
        let b_re = b[2 * k] as f64;
        let b_im = b[2 * k + 1] as f64;
        acc += a_re * b_re + a_im * b_im;
    }
    Ok((acc / dim as f64) as f32)
}

/// Sample the FPE similarity kernel: `similarity(encode(0), encode(δ))` for each `δ`.
///
/// Useful for inspecting the locality-preserving bump: the profile peaks at `1.0` for `δ = 0`
/// and decays as `|δ|` grows.
#[must_use]
pub fn kernel_profile(base: &FpeBase, deltas: &[f32]) -> Vec<f32> {
    let origin = base.encode(0.0);
    deltas
        .iter()
        .map(|&delta| {
            let encoded = base.encode(delta);
            // Both vectors are well-formed unit phasors of equal length, so similarity is finite.
            fpe_similarity(&origin, &encoded).unwrap_or(0.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rng() -> LcgRng {
        LcgRng::new(0xF9E1_1234_5678)
    }

    #[test]
    fn random_base_phases_in_range() {
        let mut r = rng();
        let base = FpeBase::random(1024, &mut r).expect("base");
        assert_eq!(base.dim(), 1024);
        for &p in base.phases() {
            assert!(p > -PI - 1e-4 && p <= PI + 1e-4, "phase {p} out of (−π, π]");
        }
    }

    #[test]
    fn encode_components_unit_magnitude() {
        let mut r = rng();
        let base = FpeBase::random(256, &mut r).expect("base");
        let hv = base.encode(0.37);
        assert_eq!(hv.len(), 512);
        for k in 0..256 {
            let re = hv[2 * k];
            let im = hv[2 * k + 1];
            let mag = (re * re + im * im).sqrt();
            assert!((mag - 1.0).abs() < 1e-5, "component {k} mag={mag}");
        }
    }

    #[test]
    fn encode_zero_is_real_identity() {
        let mut r = rng();
        let base = FpeBase::random(128, &mut r).expect("base");
        let hv = base.encode(0.0);
        for k in 0..128 {
            assert!((hv[2 * k] - 1.0).abs() < 1e-6, "re[{k}]={}", hv[2 * k]);
            assert!(hv[2 * k + 1].abs() < 1e-6, "im[{k}]={}", hv[2 * k + 1]);
        }
    }

    #[test]
    fn self_similarity_is_one() {
        let mut r = rng();
        let base = FpeBase::random(1024, &mut r).expect("base");
        let hv = base.encode(2.5);
        let sim = fpe_similarity(&hv, &hv).expect("sim");
        assert!((sim - 1.0).abs() < 1e-4, "sim={sim}");
    }

    #[test]
    fn similarity_decreases_with_delta() {
        let mut r = rng();
        let base = FpeBase::random(1024, &mut r).expect("base");
        let origin = base.encode(0.0);
        let near = base.encode(0.1);
        let far = base.encode(1.0);
        let sim_near = fpe_similarity(&origin, &near).expect("near");
        let sim_far = fpe_similarity(&origin, &far).expect("far");
        assert!(
            sim_near > sim_far,
            "expected kernel to decay: near={sim_near} far={sim_far}"
        );
        assert!(sim_near > 0.9, "near similarity too low: {sim_near}");
    }

    #[test]
    fn similarity_symmetric() {
        let mut r = rng();
        let base = FpeBase::random(512, &mut r).expect("base");
        let ea = base.encode(0.8);
        let eb = base.encode(1.3);
        let ab = fpe_similarity(&ea, &eb).expect("ab");
        let ba = fpe_similarity(&eb, &ea).expect("ba");
        assert!((ab - ba).abs() < 1e-5, "asymmetric: {ab} vs {ba}");
    }

    /// Complex multiply two interleaved FHRR hypervectors (binding = phase addition).
    fn complex_bind(a: &[f32], b: &[f32]) -> Vec<f32> {
        let dim = a.len() / 2;
        let mut out = vec![0f32; a.len()];
        for k in 0..dim {
            let ar = a[2 * k];
            let ai = a[2 * k + 1];
            let br = b[2 * k];
            let bi = b[2 * k + 1];
            out[2 * k] = ar * br - ai * bi;
            out[2 * k + 1] = ar * bi + ai * br;
        }
        out
    }

    #[test]
    fn fractional_binding_additivity() {
        // encode(a) bound with encode(b) ≈ encode(a + b).
        let mut r = rng();
        let base = FpeBase::random(1024, &mut r).expect("base");
        let a = 0.6f32;
        let b = 1.1f32;
        let bound = complex_bind(&base.encode(a), &base.encode(b));
        let direct = base.encode(a + b);
        let sim = fpe_similarity(&bound, &direct).expect("sim");
        assert!(sim > 0.99, "additivity broke: sim={sim}");
    }

    #[test]
    fn encode_nd_shape_and_value() {
        let mut r = rng();
        let bases = vec![
            FpeBase::random(256, &mut r).expect("b0"),
            FpeBase::random(256, &mut r).expect("b1"),
        ];
        let hv = FpeBase::encode_nd(&[0.5, -0.5], &bases).expect("nd");
        assert_eq!(hv.len(), 512);
        // Each component is still a unit phasor.
        for k in 0..256 {
            let mag = (hv[2 * k] * hv[2 * k] + hv[2 * k + 1] * hv[2 * k + 1]).sqrt();
            assert!((mag - 1.0).abs() < 1e-5, "component {k} mag={mag}");
        }
    }

    #[test]
    fn encode_nd_matches_manual_bind() {
        // 2-D encode equals binding the two single-axis encodings.
        let mut r = rng();
        let b0 = FpeBase::random(512, &mut r).expect("b0");
        let b1 = FpeBase::random(512, &mut r).expect("b1");
        let v = [0.7f32, 1.2f32];
        let nd = FpeBase::encode_nd(&v, &[b0.clone(), b1.clone()]).expect("nd");
        let manual = complex_bind(&b0.encode(v[0]), &b1.encode(v[1]));
        let sim = fpe_similarity(&nd, &manual).expect("sim");
        assert!(sim > 0.999, "encode_nd != manual bind: sim={sim}");
    }

    #[test]
    fn encode_nd_rejects_mismatched_lengths() {
        let mut r = rng();
        let bases = vec![
            FpeBase::random(64, &mut r).expect("b0"),
            FpeBase::random(64, &mut r).expect("b1"),
        ];
        let res = FpeBase::encode_nd(&[0.1], &bases);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn encode_nd_rejects_mismatched_dims() {
        let mut r = rng();
        let bases = vec![
            FpeBase::random(64, &mut r).expect("b0"),
            FpeBase::random(128, &mut r).expect("b1"),
        ];
        let res = FpeBase::encode_nd(&[0.1, 0.2], &bases);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn encode_nd_rejects_empty() {
        let res = FpeBase::encode_nd(&[], &[]);
        assert!(matches!(res, Err(HdcError::EmptyInput)));
    }

    #[test]
    fn random_zero_dim_errors() {
        let mut r = rng();
        assert!(matches!(
            FpeBase::random(0, &mut r),
            Err(HdcError::ZeroDimension)
        ));
    }

    #[test]
    fn kernel_profile_peak_at_zero() {
        let mut r = rng();
        let base = FpeBase::random(1024, &mut r).expect("base");
        let deltas = [0.0f32, 0.25, 0.5, 1.0, 2.0];
        let profile = kernel_profile(&base, &deltas);
        assert_eq!(profile.len(), 5);
        assert!((profile[0] - 1.0).abs() < 1e-4, "peak={}", profile[0]);
        // The peak is the maximum of the profile.
        for &v in &profile[1..] {
            assert!(profile[0] >= v - 1e-5, "peak {} < {v}", profile[0]);
        }
    }

    #[test]
    fn fpe_similarity_dim_mismatch_errors() {
        let a = vec![1.0f32, 0.0, 1.0, 0.0];
        let b = vec![1.0f32, 0.0];
        assert!(matches!(
            fpe_similarity(&a, &b),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn fpe_similarity_empty_errors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert!(matches!(fpe_similarity(&a, &b), Err(HdcError::EmptyInput)));
    }

    #[test]
    fn determinism_same_seed_same_phases() {
        let mut r1 = LcgRng::new(0xABCDEF);
        let mut r2 = LcgRng::new(0xABCDEF);
        let b1 = FpeBase::random(256, &mut r1).expect("b1");
        let b2 = FpeBase::random(256, &mut r2).expect("b2");
        assert_eq!(b1.phases(), b2.phases());
        // And encodings agree.
        let h1 = b1.encode(1.7);
        let h2 = b2.encode(1.7);
        assert_eq!(h1, h2);
    }
}
