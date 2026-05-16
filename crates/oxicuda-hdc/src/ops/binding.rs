//! Binding operations: XOR (binary), element-wise multiply (integer), circular convolution (HRR).

use crate::error::{HdcError, HdcResult};

/// XOR binding for binary HVs (±1 domain): XOR = element-wise multiply.
/// For {-1,+1}: a ⊗ b = a * b (sign product).
pub fn binary_bind(a: &[i8], b: &[i8]) -> HdcResult<Vec<i8>> {
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let result: Vec<i8> = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            // Both must be ±1; product is ±1
            ai * bi
        })
        .collect();
    Ok(result)
}

/// Binary unbind: same as bind (self-inverse with ±1 XOR).
pub fn binary_unbind(bound: &[i8], b: &[i8]) -> HdcResult<Vec<i8>> {
    binary_bind(bound, b)
}

/// Integer bind (MAP): element-wise multiply.
pub fn integer_bind_op(a: &[i32], b: &[i32]) -> HdcResult<Vec<i32>> {
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let result: Vec<i32> = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).collect();
    Ok(result)
}

/// Circular convolution (HRR binding) for real-valued HVs.
/// Compute via direct O(n²) cyclic convolution.
/// `c[k] = Σ_{j=0..n-1} a[j] * b[(k - j + n) % n]`
#[allow(clippy::needless_range_loop)]
pub fn circular_convolution(a: &[f32], b: &[f32]) -> HdcResult<Vec<f32>> {
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let n = a.len();
    let mut c = vec![0f32; n];
    for k in 0..n {
        let mut sum = 0f32;
        for j in 0..n {
            let bk = (k + n - j) % n;
            sum += a[j] * b[bk];
        }
        c[k] = sum;
    }
    Ok(c)
}

/// Circular correlation (HRR unbinding): a ⋆ b = circular_convolution(a_flipped, b).
/// The "flip" reverses a: `a_flip[0] = a[0]`, `a_flip[k] = a[n-k]` for k > 0.
#[allow(clippy::needless_range_loop)]
pub fn circular_correlation(a: &[f32], b: &[f32]) -> HdcResult<Vec<f32>> {
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let n = a.len();
    // a_flip[0] = a[0]; a_flip[k] = a[n-k] for k in 1..n
    let mut a_flip = vec![0f32; n];
    a_flip[0] = a[0];
    for k in 1..n {
        a_flip[k] = a[n - k];
    }
    circular_convolution(&a_flip, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_bind_self_inverse() {
        let a: Vec<i8> = vec![1, -1, 1, -1, 1];
        let bound = binary_bind(&a, &a).expect("bind failed");
        // a * a = all 1s
        assert!(bound.iter().all(|&v| v == 1));
    }

    #[test]
    fn circular_conv_unit_impulse() {
        // Convolution with delta [1, 0, 0, ...] should return original.
        let n = 8;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut delta = vec![0f32; n];
        delta[0] = 1.0;
        let c = circular_convolution(&a, &delta).expect("conv");
        for i in 0..n {
            assert!(
                (c[i] - a[i]).abs() < 1e-5,
                "c[{i}]={} != a[{i}]={}",
                c[i],
                a[i]
            );
        }
    }
}
