//! Tensor Product Representation (TPR) binding — Smolensky 1990.
//!
//! Reference: P. Smolensky, "Tensor product variable binding and the representation of
//! symbolic structures in connectionist systems," *Artificial Intelligence* 46 (1990).
//!
//! A *role* vector and a *filler* vector are bound by their outer product, yielding a
//! rank-2 tensor (a `role_dim × filler_dim` matrix, stored flattened row-major). A
//! structure is represented by *superposing* (summing) several such bindings. A filler is
//! recovered by *contracting* the bound tensor with its role along the role axis and
//! dividing by `‖role‖²`:
//!
//! ```text
//! bind(r, f)[i][j] = r[i] * f[j]
//! unbind(M, r)[j]  = (Σ_i r[i] * M[i][j]) / (r · r)
//! ```
//!
//! For a unit-norm role this recovers the exact filler. When several bindings with
//! mutually **orthonormal** roles are bundled, contracting with one role recovers its
//! filler while the other terms project to ~0 (the cross-talk).

use crate::error::{HdcError, HdcResult};

/// Bind a `role` and a `filler` via their outer product, flattened row-major.
///
/// The result has length `role.len() * filler.len()`, with
/// `out[i * filler.len() + j] = role[i] * filler[j]`. Empty inputs yield an empty vector
/// (the `HdcResult`-returning entry points validate non-emptiness).
#[must_use]
pub fn tensor_product_bind(role: &[f32], filler: &[f32]) -> Vec<f32> {
    let filler_dim = filler.len();
    let mut out = vec![0f32; role.len() * filler_dim];
    for (i, &r) in role.iter().enumerate() {
        let row = &mut out[i * filler_dim..(i + 1) * filler_dim];
        for (slot, &f) in row.iter_mut().zip(filler.iter()) {
            *slot = r * f;
        }
    }
    out
}

/// Recover a filler by contracting the bound tensor `bound` (shape `role_dim × filler_dim`,
/// row-major) with `role` along the role axis, divided by `‖role‖²`.
///
/// `unbind(M, r)[j] = (Σ_i r[i] * M[i][j]) / (r · r)`.
///
/// # Errors
///
/// - [`HdcError::EmptyInput`] if `role_dim == 0` or `filler_dim == 0`.
/// - [`HdcError::DimensionMismatch`] if `role.len() != role_dim` or
///   `bound.len() != role_dim * filler_dim`.
/// - [`HdcError::DivisionByZero`] if `role` has (near-)zero norm.
pub fn tensor_product_unbind(
    bound: &[f32],
    role: &[f32],
    role_dim: usize,
    filler_dim: usize,
) -> HdcResult<Vec<f32>> {
    if role_dim == 0 || filler_dim == 0 {
        return Err(HdcError::EmptyInput);
    }
    if role.len() != role_dim {
        return Err(HdcError::DimensionMismatch {
            expected: role_dim,
            got: role.len(),
        });
    }
    let expected = role_dim * filler_dim;
    if bound.len() != expected {
        return Err(HdcError::DimensionMismatch {
            expected,
            got: bound.len(),
        });
    }
    let norm_sq: f64 = role.iter().map(|&r| (r as f64) * (r as f64)).sum();
    if norm_sq < f64::EPSILON {
        return Err(HdcError::DivisionByZero);
    }

    let mut filler = vec![0f32; filler_dim];
    for (i, &r) in role.iter().enumerate() {
        let ri = r as f64;
        let row = &bound[i * filler_dim..(i + 1) * filler_dim];
        for (slot, &m) in filler.iter_mut().zip(row.iter()) {
            *slot += (ri * (m as f64)) as f32;
        }
    }
    for slot in filler.iter_mut() {
        *slot = ((*slot as f64) / norm_sq) as f32;
    }
    Ok(filler)
}

/// Bundle (superpose) several equal-shaped bound tensors by element-wise summation.
///
/// # Errors
///
/// - [`HdcError::EmptyInput`] if `bindings` is empty or any binding is empty.
/// - [`HdcError::DimensionMismatch`] if the bindings are not all the same length.
pub fn tpr_bundle(bindings: &[Vec<f32>]) -> HdcResult<Vec<f32>> {
    if bindings.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let len = bindings[0].len();
    if len == 0 {
        return Err(HdcError::EmptyInput);
    }
    for binding in bindings.iter().skip(1) {
        if binding.len() != len {
            return Err(HdcError::DimensionMismatch {
                expected: len,
                got: binding.len(),
            });
        }
    }
    let mut out = vec![0f32; len];
    for binding in bindings {
        for (slot, &v) in out.iter_mut().zip(binding.iter()) {
            *slot += v;
        }
    }
    Ok(out)
}

/// Bind each `(role, filler)` pair and bundle them into a single TPR.
///
/// Equivalent to `tpr_bundle(&[bind(roles[0], fillers[0]), …])`. All roles must share one
/// dimension and all fillers another (so every binding has the same shape).
///
/// # Errors
///
/// - [`HdcError::EmptyInput`] if `roles` is empty, or any role or filler is empty.
/// - [`HdcError::DimensionMismatch`] if `roles.len() != fillers.len()`, or the role/filler
///   dimensions are not consistent across pairs.
pub fn tpr_encode(roles: &[Vec<f32>], fillers: &[Vec<f32>]) -> HdcResult<Vec<f32>> {
    if roles.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    if roles.len() != fillers.len() {
        return Err(HdcError::DimensionMismatch {
            expected: roles.len(),
            got: fillers.len(),
        });
    }
    let role_dim = roles[0].len();
    let filler_dim = fillers[0].len();
    if role_dim == 0 || filler_dim == 0 {
        return Err(HdcError::EmptyInput);
    }
    for (role, filler) in roles.iter().zip(fillers.iter()) {
        if role.len() != role_dim {
            return Err(HdcError::DimensionMismatch {
                expected: role_dim,
                got: role.len(),
            });
        }
        if filler.len() != filler_dim {
            return Err(HdcError::DimensionMismatch {
                expected: filler_dim,
                got: filler.len(),
            });
        }
    }
    let bindings: Vec<Vec<f32>> = roles
        .iter()
        .zip(fillers.iter())
        .map(|(r, f)| tensor_product_bind(r, f))
        .collect();
    tpr_bundle(&bindings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    /// Deterministic Gaussian vector via the LCG Box-Muller sampler.
    fn gaussian_vec(dim: usize, rng: &mut LcgRng) -> Vec<f32> {
        let mut v = Vec::with_capacity(dim);
        while v.len() < dim {
            let (a, b) = rng.normal_pair_f32();
            v.push(a);
            if v.len() < dim {
                v.push(b);
            }
        }
        v
    }

    fn l2_normalize(v: &mut [f32]) {
        let norm: f64 = v
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x = ((*x as f64) / norm) as f32;
            }
        }
    }

    /// Generate `n` orthonormal vectors of length `dim` via modified Gram-Schmidt.
    fn orthonormal_set(n: usize, dim: usize, rng: &mut LcgRng) -> Vec<Vec<f32>> {
        let mut basis: Vec<Vec<f32>> = Vec::new();
        while basis.len() < n {
            let mut candidate = gaussian_vec(dim, rng);
            for b in &basis {
                let dot: f64 = candidate
                    .iter()
                    .zip(b.iter())
                    .map(|(&c, &bv)| (c as f64) * (bv as f64))
                    .sum();
                for (c, &bv) in candidate.iter_mut().zip(b.iter()) {
                    *c = ((*c as f64) - dot * (bv as f64)) as f32;
                }
            }
            let norm: f64 = candidate
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum::<f64>()
                .sqrt();
            if norm > 1e-3 {
                l2_normalize(&mut candidate);
                basis.push(candidate);
            }
        }
        basis
    }

    #[test]
    fn bind_length_is_role_times_filler() {
        let role = vec![1.0f32, 2.0, 3.0];
        let filler = vec![4.0f32, 5.0];
        let bound = tensor_product_bind(&role, &filler);
        assert_eq!(bound.len(), 6);
    }

    #[test]
    fn bind_outer_product_values() {
        let role = vec![1.0f32, 2.0];
        let filler = vec![3.0f32, 4.0, 5.0];
        let bound = tensor_product_bind(&role, &filler);
        // Row 0: 1*[3,4,5]; row 1: 2*[3,4,5].
        assert_eq!(bound, vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn unbind_recovers_filler_unit_role() {
        // Unit-norm role → exact filler recovery.
        let mut rng = LcgRng::new(0x5EED);
        let role_dim = 8;
        let filler_dim = 5;
        let mut role = gaussian_vec(role_dim, &mut rng);
        l2_normalize(&mut role);
        let filler = gaussian_vec(filler_dim, &mut rng);
        let bound = tensor_product_bind(&role, &filler);
        let recovered = tensor_product_unbind(&bound, &role, role_dim, filler_dim)
            .expect("tensor_product_unbind should succeed");
        for (r, f) in recovered.iter().zip(filler.iter()) {
            assert!((r - f).abs() < 1e-5, "recovered {r} != filler {f}");
        }
    }

    #[test]
    fn unbind_divides_by_norm_squared_nonunit_role() {
        // Non-unit role: the /‖role‖² normalisation must still yield the exact filler.
        let role = vec![3.0f32, 0.0, 0.0]; // ‖role‖² = 9
        let filler = vec![2.0f32, -1.0];
        let bound = tensor_product_bind(&role, &filler);
        let recovered = tensor_product_unbind(&bound, &role, 3, 2)
            .expect("tensor_product_unbind should succeed");
        for (r, f) in recovered.iter().zip(filler.iter()) {
            assert!((r - f).abs() < 1e-5, "recovered {r} != filler {f}");
        }
    }

    #[test]
    fn unbind_general_nonunit_role_exact() {
        let mut rng = LcgRng::new(0xBEEF);
        let role_dim = 6;
        let filler_dim = 4;
        // Arbitrary (non-unit) role.
        let role = gaussian_vec(role_dim, &mut rng);
        let filler = gaussian_vec(filler_dim, &mut rng);
        let bound = tensor_product_bind(&role, &filler);
        let recovered = tensor_product_unbind(&bound, &role, role_dim, filler_dim)
            .expect("tensor_product_unbind should succeed");
        for (r, f) in recovered.iter().zip(filler.iter()) {
            assert!((r - f).abs() < 1e-4, "recovered {r} != filler {f}");
        }
    }

    #[test]
    fn bundle_then_unbind_recovers_each_filler_orthonormal_roles() {
        // The headline TPR property: orthonormal roles → unbind one recovers its filler.
        let mut rng = LcgRng::new(0xABCDEF01);
        let role_dim = 16;
        let filler_dim = 8;
        let n = 3;
        let roles = orthonormal_set(n, role_dim, &mut rng);
        let fillers: Vec<Vec<f32>> = (0..n).map(|_| gaussian_vec(filler_dim, &mut rng)).collect();
        let bundle = tpr_encode(&roles, &fillers).expect("tpr_encode should succeed");
        for i in 0..n {
            let recovered = tensor_product_unbind(&bundle, &roles[i], role_dim, filler_dim)
                .expect("tensor_product_unbind should succeed");
            for (r, f) in recovered.iter().zip(fillers[i].iter()) {
                assert!(
                    (r - f).abs() < 1e-4,
                    "role {i}: recovered {r} != filler {f}"
                );
            }
        }
    }

    #[test]
    fn tpr_bundle_sums_elementwise() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![10.0f32, 20.0, 30.0];
        let c = vec![100.0f32, 200.0, 300.0];
        let bundled = tpr_bundle(&[a, b, c]).expect("tpr_bundle should succeed");
        assert_eq!(bundled, vec![111.0, 222.0, 333.0]);
    }

    #[test]
    fn tpr_encode_equals_bind_each_plus_bundle() {
        let roles = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let fillers = vec![vec![2.0f32, 3.0], vec![4.0f32, 5.0]];
        let encoded = tpr_encode(&roles, &fillers).expect("tpr_encode should succeed");
        let manual = tpr_bundle(&[
            tensor_product_bind(&roles[0], &fillers[0]),
            tensor_product_bind(&roles[1], &fillers[1]),
        ])
        .expect("value should be present");
        assert_eq!(encoded, manual);
    }

    #[test]
    fn crosstalk_small_for_orthogonal_roles() {
        // Two orthonormal roles: unbinding role 0 should give ~filler0 (filler1 ~ 0 leakage).
        let mut rng = LcgRng::new(0x1357);
        let role_dim = 32;
        let filler_dim = 4;
        let roles = orthonormal_set(2, role_dim, &mut rng);
        let f0 = vec![1.0f32, 0.0, 0.0, 0.0];
        let f1 = vec![0.0f32, 1.0, 0.0, 0.0];
        let bundle =
            tpr_encode(&roles, &[f0.clone(), f1.clone()]).expect("value should be present");
        let rec0 = tensor_product_unbind(&bundle, &roles[0], role_dim, filler_dim)
            .expect("tensor_product_unbind should succeed");
        // rec0 ≈ f0: component 0 ≈ 1, all others ≈ 0 (cross-talk from f1 is negligible).
        assert!((rec0[0] - 1.0).abs() < 1e-4, "rec0[0]={}", rec0[0]);
        for &v in &rec0[1..] {
            assert!(v.abs() < 1e-4, "cross-talk leak {v}");
        }
    }

    #[test]
    fn bind_is_bilinear_in_filler() {
        // bind(r, f1 + f2) == bind(r, f1) + bind(r, f2).
        let role = vec![1.0f32, -2.0, 3.0];
        let f1 = vec![4.0f32, 5.0];
        let f2 = vec![-1.0f32, 2.0];
        let sum_filler: Vec<f32> = f1.iter().zip(f2.iter()).map(|(&a, &b)| a + b).collect();
        let lhs = tensor_product_bind(&role, &sum_filler);
        let rhs = tpr_bundle(&[
            tensor_product_bind(&role, &f1),
            tensor_product_bind(&role, &f2),
        ])
        .expect("value should be present");
        for (l, r) in lhs.iter().zip(rhs.iter()) {
            assert!((l - r).abs() < 1e-5, "bilinearity broke: {l} != {r}");
        }
    }

    #[test]
    fn bind_is_bilinear_in_role() {
        // bind(r1 + r2, f) == bind(r1, f) + bind(r2, f).
        let r1 = vec![1.0f32, 2.0];
        let r2 = vec![3.0f32, -1.0];
        let filler = vec![5.0f32, 6.0, 7.0];
        let sum_role: Vec<f32> = r1.iter().zip(r2.iter()).map(|(&a, &b)| a + b).collect();
        let lhs = tensor_product_bind(&sum_role, &filler);
        let rhs = tpr_bundle(&[
            tensor_product_bind(&r1, &filler),
            tensor_product_bind(&r2, &filler),
        ])
        .expect("value should be present");
        for (l, r) in lhs.iter().zip(rhs.iter()) {
            assert!((l - r).abs() < 1e-5, "bilinearity broke: {l} != {r}");
        }
    }

    #[test]
    fn err_unbind_dim_mismatch_bound() {
        // role_dim*filler_dim = 6, but bound has 5 elements.
        let role = vec![1.0f32, 0.0, 0.0];
        let res = tensor_product_unbind(&[0.0; 5], &role, 3, 2);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_unbind_dim_mismatch_role() {
        // role.len()=2 but role_dim=3.
        let res = tensor_product_unbind(&[0.0; 6], &[1.0, 0.0], 3, 2);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn err_unbind_zero_role_div_guard() {
        // All-zero role → ‖role‖² = 0 → DivisionByZero, not a panic/NaN.
        let res = tensor_product_unbind(&[0.0; 6], &[0.0, 0.0, 0.0], 3, 2);
        assert!(matches!(res, Err(HdcError::DivisionByZero)));
    }

    #[test]
    fn err_unbind_zero_dims() {
        assert!(matches!(
            tensor_product_unbind(&[], &[], 0, 3),
            Err(HdcError::EmptyInput)
        ));
        assert!(matches!(
            tensor_product_unbind(&[], &[1.0], 1, 0),
            Err(HdcError::EmptyInput)
        ));
    }

    #[test]
    fn err_bundle_empty() {
        let empty: Vec<Vec<f32>> = Vec::new();
        assert!(matches!(tpr_bundle(&empty), Err(HdcError::EmptyInput)));
    }

    #[test]
    fn err_bundle_mismatched_lengths() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 2.0]; // shorter
        assert!(matches!(
            tpr_bundle(&[a, b]),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_encode_roles_fillers_length_mismatch() {
        let roles = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let fillers = vec![vec![1.0f32, 2.0]]; // only 1 filler
        assert!(matches!(
            tpr_encode(&roles, &fillers),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_encode_empty_roles() {
        let roles: Vec<Vec<f32>> = Vec::new();
        let fillers: Vec<Vec<f32>> = Vec::new();
        assert!(matches!(
            tpr_encode(&roles, &fillers),
            Err(HdcError::EmptyInput)
        ));
    }

    #[test]
    fn err_encode_inconsistent_role_dims() {
        // Second role has a different dimension.
        let roles = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0, 0.0]];
        let fillers = vec![vec![1.0f32], vec![2.0f32]];
        assert!(matches!(
            tpr_encode(&roles, &fillers),
            Err(HdcError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn trivial_one_by_one() {
        // 1×1: bind is a scalar product, unbind divides it back out.
        let role = vec![2.0f32];
        let filler = vec![5.0f32];
        let bound = tensor_product_bind(&role, &filler);
        assert_eq!(bound, vec![10.0]);
        let recovered = tensor_product_unbind(&bound, &role, 1, 1)
            .expect("tensor_product_unbind should succeed");
        assert!((recovered[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn deterministic() {
        let role = vec![1.0f32, 2.0, 3.0];
        let filler = vec![4.0f32, 5.0];
        let a = tensor_product_bind(&role, &filler);
        let b = tensor_product_bind(&role, &filler);
        assert_eq!(a, b);
    }

    #[test]
    fn bundle_of_one_equals_that_binding() {
        let binding = tensor_product_bind(&[1.0, 2.0], &[3.0, 4.0]);
        let bundled = tpr_bundle(std::slice::from_ref(&binding)).expect("value should be present");
        assert_eq!(bundled, binding);
    }

    #[test]
    fn nonsquare_role_dim_ne_filler_dim() {
        // role_dim (5) != filler_dim (2): bind/unbind round-trips through a rectangular tensor.
        let mut rng = LcgRng::new(0xFACE);
        let role_dim = 5;
        let filler_dim = 2;
        let mut role = gaussian_vec(role_dim, &mut rng);
        l2_normalize(&mut role);
        let filler = gaussian_vec(filler_dim, &mut rng);
        let bound = tensor_product_bind(&role, &filler);
        assert_eq!(bound.len(), role_dim * filler_dim);
        let recovered = tensor_product_unbind(&bound, &role, role_dim, filler_dim)
            .expect("tensor_product_unbind should succeed");
        for (r, f) in recovered.iter().zip(filler.iter()) {
            assert!((r - f).abs() < 1e-5, "recovered {r} != filler {f}");
        }
    }
}
