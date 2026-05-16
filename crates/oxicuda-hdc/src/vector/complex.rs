//! Complex unit hypervectors (FHRR/HRR model).
//!
//! Stored as `Vec<f32>` of length 2*dim interleaved: [re_0, im_0, re_1, im_1, ...].

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;

/// Check that a complex HV buffer has the expected length (must be even).
fn check_complex_len(hv: &[f32]) -> HdcResult<usize> {
    if hv.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    if !hv.len().is_multiple_of(2) {
        return Err(HdcError::DimensionMismatch {
            expected: hv.len() + 1,
            got: hv.len(),
        });
    }
    Ok(hv.len() / 2)
}

/// Generate random complex HV (uniform phases on unit circle).
pub fn random_complex(dim: usize, rng: &mut LcgRng) -> HdcResult<Vec<f32>> {
    if dim == 0 {
        return Err(HdcError::ZeroDimension);
    }
    let mut v = vec![0f32; 2 * dim];
    let mut i = 0;
    while i < dim {
        let theta = rng.next_f32() * std::f32::consts::TAU;
        v[2 * i] = theta.cos();
        v[2 * i + 1] = theta.sin();
        i += 1;
    }
    Ok(v)
}

/// Element-wise complex multiply (FHRR binding = phase addition).
/// (a_re + i*a_im)(b_re + i*b_im) = (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)
pub fn complex_bind(a: &[f32], b: &[f32]) -> HdcResult<Vec<f32>> {
    let dim_a = check_complex_len(a)?;
    let dim_b = check_complex_len(b)?;
    if dim_a != dim_b {
        return Err(HdcError::DimensionMismatch {
            expected: dim_a,
            got: dim_b,
        });
    }
    let mut out = vec![0f32; a.len()];
    let mut i = 0;
    while i < dim_a {
        let a_re = a[2 * i];
        let a_im = a[2 * i + 1];
        let b_re = b[2 * i];
        let b_im = b[2 * i + 1];
        out[2 * i] = a_re * b_re - a_im * b_im;
        out[2 * i + 1] = a_re * b_im + a_im * b_re;
        i += 1;
    }
    Ok(out)
}

/// Complex conjugate (FHRR inverse / unbinding): negate imaginary part.
pub fn complex_conjugate(hv: &[f32]) -> HdcResult<Vec<f32>> {
    let dim = check_complex_len(hv)?;
    let mut out = vec![0f32; hv.len()];
    let mut i = 0;
    while i < dim {
        out[2 * i] = hv[2 * i];
        out[2 * i + 1] = -hv[2 * i + 1];
        i += 1;
    }
    Ok(out)
}

/// Element-wise complex add (superposition / bundling) then normalize each component to unit circle.
pub fn complex_bundle(hvs: &[Vec<f32>]) -> HdcResult<Vec<f32>> {
    if hvs.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let dim = check_complex_len(&hvs[0])?;
    for hv in hvs.iter().skip(1) {
        let d = check_complex_len(hv)?;
        if d != dim {
            return Err(HdcError::DimensionMismatch {
                expected: dim,
                got: d,
            });
        }
    }
    let mut acc = vec![0f32; 2 * dim];
    for hv in hvs {
        for (a, &v) in acc.iter_mut().zip(hv.iter()) {
            *a += v;
        }
    }
    // Normalize each component back to unit circle
    let mut i = 0;
    while i < dim {
        let re = acc[2 * i];
        let im = acc[2 * i + 1];
        let mag = (re * re + im * im).sqrt();
        if mag > f32::EPSILON {
            acc[2 * i] = re / mag;
            acc[2 * i + 1] = im / mag;
        } else {
            // zero vector: default to angle 0
            acc[2 * i] = 1.0;
            acc[2 * i + 1] = 0.0;
        }
        i += 1;
    }
    Ok(acc)
}

/// Cosine similarity via real part of inner product (normalized).
/// Re(a · conj(b)) / dim
pub fn complex_cosine(a: &[f32], b: &[f32]) -> HdcResult<f32> {
    let dim_a = check_complex_len(a)?;
    let dim_b = check_complex_len(b)?;
    if dim_a != dim_b {
        return Err(HdcError::DimensionMismatch {
            expected: dim_a,
            got: dim_b,
        });
    }
    if dim_a == 0 {
        return Err(HdcError::ZeroDimension);
    }
    let mut re_sum = 0f32;
    let mut i = 0;
    while i < dim_a {
        let a_re = a[2 * i];
        let a_im = a[2 * i + 1];
        let b_re = b[2 * i];
        let b_im = b[2 * i + 1];
        // Re(a * conj(b)) = a_re*b_re + a_im*b_im
        re_sum += a_re * b_re + a_im * b_im;
        i += 1;
    }
    Ok(re_sum / dim_a as f32)
}

/// Re-normalize each component to lie exactly on unit circle.
pub fn complex_normalize(hv: &mut [f32]) -> HdcResult<()> {
    let dim = check_complex_len(hv)?;
    let mut i = 0;
    while i < dim {
        let re = hv[2 * i];
        let im = hv[2 * i + 1];
        let mag = (re * re + im * im).sqrt();
        if mag > f32::EPSILON {
            hv[2 * i] = re / mag;
            hv[2 * i + 1] = im / mag;
        } else {
            hv[2 * i] = 1.0;
            hv[2 * i + 1] = 0.0;
        }
        i += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn random_complex_unit_norms() {
        let mut rng = LcgRng::new(20);
        let hv = random_complex(100, &mut rng).expect("failed");
        assert_eq!(hv.len(), 200);
        for i in 0..100 {
            let mag = (hv[2 * i].powi(2) + hv[2 * i + 1].powi(2)).sqrt();
            assert!((mag - 1.0_f32).abs() < 1e-5);
        }
    }

    #[test]
    fn complex_bind_conjugate_roundtrip() {
        let mut rng = LcgRng::new(21);
        let a = random_complex(50, &mut rng).expect("failed");
        let b = random_complex(50, &mut rng).expect("failed");
        let bound = complex_bind(&a, &b).expect("bind");
        let conj_b = complex_conjugate(&b).expect("conj");
        let unbound = complex_bind(&bound, &conj_b).expect("unbind");
        let sim = complex_cosine(&a, &unbound).expect("cosine");
        assert!(sim > 0.99, "sim={sim}");
    }
}
