//! Permutation-based binding (Plate's HRR realised with permutation matrices).
//!
//! In a Vector Symbolic Architecture (VSA) a *role* hypervector can be realised
//! as a fixed random permutation `ρ` of the coordinate axes — equivalently a
//! permutation matrix `P` with exactly one `1` per row and column. Binding a
//! *filler* hypervector `f` to such a role is then the application of the
//! permutation, `b = ρ(f)` (i.e. `b = P f`), and unbinding is the application
//! of the inverse permutation, `f = ρ⁻¹(b)` (i.e. `f = Pᵀ b`, because a
//! permutation matrix is orthogonal so `P⁻¹ = Pᵀ`).
//!
//! This realises the three properties Plate requires of a binding operator while
//! avoiding the spreading of magnitude that circular convolution produces:
//!
//! * **Invertibility** — `ρ⁻¹(ρ(f)) = f` exactly (a permutation is a bijection).
//! * **Dissimilarity** — for a non-identity permutation `ρ(f)` is, with high
//!   probability, near-orthogonal to `f`; the role hides the filler.
//! * **Distributivity over superposition** — permutation is linear, so binding
//!   distributes over the (element-wise) bundling sum.
//!
//! The last property is what makes the *role–filler* memory work. Given pairs
//! `(ρ_i, f_i)` we form the superposition
//!
//! ```text
//! s = Σ_i ρ_i(f_i)
//! ```
//!
//! and, presented with a probe role `ρ_j`, we recover the associated filler by
//! applying the inverse permutation,
//!
//! ```text
//! ρ_j⁻¹(s) = f_j + Σ_{i ≠ j} ρ_j⁻¹(ρ_i(f_i)) = f_j + noise,
//! ```
//!
//! where the cross terms `ρ_j⁻¹(ρ_i(f_i))` behave as pseudo-random noise that is
//! near-orthogonal to every clean filler. A clean-up / nearest-neighbour step
//! (here a cosine comparison against the original fillers) then resolves `f_j`.
//!
//! This is permutation binding in the sense of Gayler (2003) and Kanerva (2009),
//! and is *distinct* from the element-wise sign product implemented by
//! `binary_bind` and from circular-convolution binding (`bind_circular`).
//!
//! The operators here are built directly on top of the primitives in
//! [`crate::ops::permutation`]:
//! [`random_permutation`] (the
//! Fisher–Yates generator), [`random_permute`]
//! (forward application to a binary HV) and
//! [`inverse_permute`] (inverse
//! application). Only the integer-typed gather is added locally, since the
//! primitive module exposes the `i8` variants only.
//!
//! # References
//!
//! * R. W. Gayler, "Vector Symbolic Architectures answer Jackendoff's challenges
//!   for cognitive neuroscience," in *Proc. ICCS/ASCS Joint Int. Conf. on
//!   Cognitive Science*, 2003, pp. 133–138.
//! * P. Kanerva, "Hyperdimensional computing: An introduction to computing in
//!   distributed representation with high-dimensional random vectors,"
//!   *Cognitive Computation*, vol. 1, no. 2, pp. 139–159, 2009.
//! * T. A. Plate, "Holographic Reduced Representations," *IEEE Transactions on
//!   Neural Networks*, vol. 6, no. 3, pp. 623–641, 1995.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::ops::bundling::bundle_integer;
use crate::ops::permutation::{inverse_permute, random_permutation, random_permute};

/// Apply a permutation to an integer HV by index gather.
///
/// Mirrors [`random_permute`] for the
/// `i32` element type: `out[i] = hv[perm[i]]`. `perm[i]` is the *source* index
/// for output position `i`.
fn permute_i32(hv: &[i32], perm: &[usize]) -> HdcResult<Vec<i32>> {
    if hv.len() != perm.len() {
        return Err(HdcError::PermutationLengthMismatch {
            perm_len: perm.len(),
            dim: hv.len(),
        });
    }
    if hv.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let dim = hv.len();
    let mut out = vec![0i32; dim];
    for (i, &src) in perm.iter().enumerate() {
        if src >= dim {
            return Err(HdcError::FeatureIndexOutOfRange {
                feat: src,
                max: dim,
            });
        }
        out[i] = hv[src];
    }
    Ok(out)
}

/// Apply the inverse of a permutation to an integer HV.
///
/// Mirrors [`inverse_permute`] for the
/// `i32` element type. If `perm[i] = j` then the inverse maps `j` back to `i`,
/// so `permute_i32` and `inverse_permute_i32` compose to the identity.
fn inverse_permute_i32(hv: &[i32], perm: &[usize]) -> HdcResult<Vec<i32>> {
    if hv.len() != perm.len() {
        return Err(HdcError::PermutationLengthMismatch {
            perm_len: perm.len(),
            dim: hv.len(),
        });
    }
    if hv.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let dim = hv.len();
    let mut inv_perm = vec![0usize; dim];
    for (i, &p) in perm.iter().enumerate() {
        if p >= dim {
            return Err(HdcError::FeatureIndexOutOfRange { feat: p, max: dim });
        }
        inv_perm[p] = i;
    }
    let mut out = vec![0i32; dim];
    for i in 0..dim {
        out[i] = hv[inv_perm[i]];
    }
    Ok(out)
}

/// A binding *role* realised as a fixed permutation of the coordinate axes.
///
/// Conceptually this is a permutation matrix `P`; binding a filler `f` produces
/// `P f` ([`PermutationRole::bind`]) and unbinding produces `Pᵀ (P f) = f`
/// ([`PermutationRole::unbind`]). The same role can bind both binary (`i8`,
/// `{-1, +1}`) and integer (`i32`) hypervectors.
#[derive(Debug, Clone)]
pub struct PermutationRole {
    /// The permutation: `perm[i]` is the source index for output position `i`.
    /// It is always a genuine permutation of `0..dim`.
    perm: Vec<usize>,
    /// The hypervector dimension this role operates on (`perm.len()`).
    dim: usize,
}

impl PermutationRole {
    /// Construct a role from a fresh random permutation of `0..dim`.
    ///
    /// Uses [`random_permutation`]
    /// (Fisher–Yates) so the result is uniformly distributed over the symmetric
    /// group. Returns [`HdcError::ZeroDimension`] when `dim == 0`.
    pub fn random(dim: usize, rng: &mut LcgRng) -> HdcResult<Self> {
        let perm = random_permutation(dim, rng)?;
        Ok(Self { dim, perm })
    }

    /// Construct a role from an explicit permutation, validating it.
    ///
    /// The input must be a genuine permutation of `0..perm.len()`: every index
    /// in that range must appear exactly once. Validation uses a single
    /// seen-bitmap pass:
    ///
    /// * an entry `>= len` yields [`HdcError::FeatureIndexOutOfRange`];
    /// * a repeated entry (which, combined with the range check, also implies a
    ///   missing index) yields
    ///   [`HdcError::PermutationLengthMismatch`];
    /// * an empty input yields [`HdcError::EmptyInput`].
    pub fn from_perm(perm: Vec<usize>) -> HdcResult<Self> {
        let dim = perm.len();
        if dim == 0 {
            return Err(HdcError::EmptyInput);
        }
        let mut seen = vec![false; dim];
        for &p in &perm {
            if p >= dim {
                return Err(HdcError::FeatureIndexOutOfRange { feat: p, max: dim });
            }
            if seen[p] {
                // A duplicate within range guarantees some index of 0..dim is
                // missing, so the multiset is not a permutation of 0..dim.
                return Err(HdcError::PermutationLengthMismatch { perm_len: dim, dim });
            }
            seen[p] = true;
        }
        Ok(Self { dim, perm })
    }

    /// The hypervector dimension this role operates on.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Borrow the underlying permutation (`perm[i]` is the source index for
    /// output position `i`).
    #[must_use]
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// Bind a binary filler: apply the permutation, returning `ρ(f)`.
    ///
    /// Reuses [`random_permute`].
    /// Returns [`HdcError::PermutationLengthMismatch`] when
    /// `filler.len() != dim`.
    pub fn bind(&self, filler: &[i8]) -> HdcResult<Vec<i8>> {
        if filler.len() != self.dim {
            return Err(HdcError::PermutationLengthMismatch {
                perm_len: self.dim,
                dim: filler.len(),
            });
        }
        random_permute(filler, &self.perm)
    }

    /// Unbind a binary bound vector: apply the inverse permutation, returning
    /// `ρ⁻¹(b)`.
    ///
    /// Reuses [`inverse_permute`], so
    /// `unbind(bind(f)) == f` and `bind(unbind(b)) == b` exactly. Returns
    /// [`HdcError::PermutationLengthMismatch`] when `bound.len() != dim`.
    pub fn unbind(&self, bound: &[i8]) -> HdcResult<Vec<i8>> {
        if bound.len() != self.dim {
            return Err(HdcError::PermutationLengthMismatch {
                perm_len: self.dim,
                dim: bound.len(),
            });
        }
        inverse_permute(bound, &self.perm)
    }

    /// Bind an integer filler: apply the permutation, returning `ρ(f)`.
    ///
    /// The `i32` analogue of [`PermutationRole::bind`]. Returns
    /// [`HdcError::PermutationLengthMismatch`] when `filler.len() != dim`.
    pub fn bind_i32(&self, filler: &[i32]) -> HdcResult<Vec<i32>> {
        if filler.len() != self.dim {
            return Err(HdcError::PermutationLengthMismatch {
                perm_len: self.dim,
                dim: filler.len(),
            });
        }
        permute_i32(filler, &self.perm)
    }

    /// Unbind an integer bound vector: apply the inverse permutation.
    ///
    /// The `i32` analogue of [`PermutationRole::unbind`]; satisfies
    /// `unbind_i32(bind_i32(f)) == f` exactly. Returns
    /// [`HdcError::PermutationLengthMismatch`] when `bound.len() != dim`.
    pub fn unbind_i32(&self, bound: &[i32]) -> HdcResult<Vec<i32>> {
        if bound.len() != self.dim {
            return Err(HdcError::PermutationLengthMismatch {
                perm_len: self.dim,
                dim: bound.len(),
            });
        }
        inverse_permute_i32(bound, &self.perm)
    }
}

/// Bind every filler with its role and superpose the results.
///
/// Computes the integer superposition `s = Σ_i ρ_i(f_i)` by binding each
/// `fillers[i]` with `roles[i]` ([`PermutationRole::bind_i32`]) and summing with
/// [`bundle_integer`]. This is the role–filler memory record from which
/// [`recover_i32`] retrieves individual fillers.
///
/// # Errors
///
/// * [`HdcError::EmptyInput`] when `roles` (or `fillers`) is empty.
/// * [`HdcError::DimensionMismatch`] when `roles.len() != fillers.len()`.
/// * Any error propagated from per-pair binding (e.g. a length mismatch between
///   a role's dimension and its filler).
pub fn bind_superpose_i32(roles: &[PermutationRole], fillers: &[Vec<i32>]) -> HdcResult<Vec<i32>> {
    if roles.is_empty() || fillers.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    if roles.len() != fillers.len() {
        return Err(HdcError::DimensionMismatch {
            expected: roles.len(),
            got: fillers.len(),
        });
    }
    let mut bound: Vec<Vec<i32>> = Vec::with_capacity(roles.len());
    for (role, filler) in roles.iter().zip(fillers.iter()) {
        bound.push(role.bind_i32(filler)?);
    }
    bundle_integer(&bound)
}

/// Recover the filler associated with `role` from an integer superposition.
///
/// Applies the inverse permutation, `ρ⁻¹(s)` ([`PermutationRole::unbind_i32`]).
/// The result equals the clean filler plus near-orthogonal crossover noise from
/// the other bound pairs; a clean-up step (e.g. cosine nearest-neighbour against
/// the candidate fillers) recovers the exact symbol.
///
/// # Errors
///
/// Propagates [`HdcError::PermutationLengthMismatch`] when
/// `superposition.len() != role.dim()`.
pub fn recover_i32(role: &PermutationRole, superposition: &[i32]) -> HdcResult<Vec<i32>> {
    role.unbind_i32(superposition)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::cosine::{cosine_binary, cosine_integer};
    use crate::handle::LcgRng;

    /// Generate a random ±1 binary filler of length `dim`.
    fn random_binary(dim: usize, rng: &mut LcgRng) -> Vec<i8> {
        let mut buf = vec![0i8; dim];
        rng.fill_binary(&mut buf);
        buf
    }

    /// Generate a random small-integer filler (values in `{-1, +1}`) so that
    /// `cosine_integer` is well-conditioned (non-zero norm).
    fn random_integer(dim: usize, rng: &mut LcgRng) -> Vec<i32> {
        (0..dim)
            .map(|_| if rng.next_bool() { 1i32 } else { -1i32 })
            .collect()
    }

    #[test]
    fn binary_bind_unbind_roundtrip() {
        let mut rng = LcgRng::new(1);
        let dim = 1024;
        let role = PermutationRole::random(dim, &mut rng).expect("role");
        let filler = random_binary(dim, &mut rng);
        let bound = role.bind(&filler).expect("bind");
        let recovered = role.unbind(&bound).expect("unbind");
        assert_eq!(recovered, filler, "unbind∘bind must be identity (i8)");
        // bind∘unbind is also identity.
        let unbound = role.unbind(&filler).expect("unbind");
        let rebound = role.bind(&unbound).expect("bind");
        assert_eq!(rebound, filler, "bind∘unbind must be identity (i8)");
    }

    #[test]
    fn integer_bind_unbind_roundtrip() {
        let mut rng = LcgRng::new(2);
        let dim = 1024;
        let role = PermutationRole::random(dim, &mut rng).expect("role");
        let filler = random_integer(dim, &mut rng);
        let bound = role.bind_i32(&filler).expect("bind_i32");
        let recovered = role.unbind_i32(&bound).expect("unbind_i32");
        assert_eq!(recovered, filler, "unbind∘bind must be identity (i32)");
    }

    #[test]
    fn from_perm_accepts_genuine_permutation() {
        let perm = vec![3usize, 0, 2, 1];
        let role = PermutationRole::from_perm(perm.clone()).expect("from_perm");
        assert_eq!(role.dim(), 4);
        assert_eq!(role.perm(), perm.as_slice());
    }

    #[test]
    fn from_perm_rejects_duplicate_index() {
        // 2 appears twice, 1 is missing → not a permutation.
        let perm = vec![0usize, 2, 2, 3];
        let err = PermutationRole::from_perm(perm).expect_err("must reject");
        assert!(
            matches!(err, HdcError::PermutationLengthMismatch { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn from_perm_rejects_out_of_range_index() {
        // 4 is out of range for length 4 (valid indices 0..=3).
        let perm = vec![0usize, 1, 4, 3];
        let err = PermutationRole::from_perm(perm).expect_err("must reject");
        assert!(
            matches!(err, HdcError::FeatureIndexOutOfRange { feat: 4, max: 4 }),
            "got {err:?}"
        );
    }

    #[test]
    fn from_perm_rejects_empty() {
        let err = PermutationRole::from_perm(Vec::new()).expect_err("must reject");
        assert!(matches!(err, HdcError::EmptyInput), "got {err:?}");
    }

    #[test]
    fn binding_changes_vector_for_non_identity_role() {
        let mut rng = LcgRng::new(3);
        let dim = 2048;
        let role = PermutationRole::random(dim, &mut rng).expect("role");
        // A random permutation of 2048 elements is non-identity with
        // overwhelming probability; assert it explicitly for determinism.
        assert!(
            role.perm().iter().enumerate().any(|(i, &p)| i != p),
            "expected a non-identity permutation"
        );
        let filler = random_binary(dim, &mut rng);
        let bound = role.bind(&filler).expect("bind");
        let sim = cosine_binary(&filler, &bound).expect("cosine");
        // Role hides the filler: bound vector is near-orthogonal, well below 1.
        assert!(
            sim < 0.5,
            "permuted vector should differ from original, sim={sim}"
        );
    }

    #[test]
    fn two_roles_give_different_bindings() {
        let mut rng = LcgRng::new(4);
        let dim = 1024;
        let role_a = PermutationRole::random(dim, &mut rng).expect("role a");
        let role_b = PermutationRole::random(dim, &mut rng).expect("role b");
        assert_ne!(role_a.perm(), role_b.perm(), "roles must differ");
        let filler = random_binary(dim, &mut rng);
        let bound_a = role_a.bind(&filler).expect("bind a");
        let bound_b = role_b.bind(&filler).expect("bind b");
        assert_ne!(
            bound_a, bound_b,
            "different roles must bind the same filler differently"
        );
    }

    #[test]
    fn superpose_then_recover_each_filler() {
        let mut rng = LcgRng::new(5);
        let dim = 2048;
        let n_pairs = 5;
        let roles: Vec<PermutationRole> = (0..n_pairs)
            .map(|_| PermutationRole::random(dim, &mut rng).expect("role"))
            .collect();
        let fillers: Vec<Vec<i32>> = (0..n_pairs)
            .map(|_| random_integer(dim, &mut rng))
            .collect();

        let superposition = bind_superpose_i32(&roles, &fillers).expect("superpose");

        // An independent random reference vector for the noise baseline.
        let reference = random_integer(dim, &mut rng);

        for (i, role) in roles.iter().enumerate() {
            let recovered = recover_i32(role, &superposition).expect("recover");
            let sim_clean = cosine_integer(&recovered, &fillers[i]).expect("cosine clean");
            let sim_noise = cosine_integer(&recovered, &reference).expect("cosine noise");
            // The recovered vector is far closer to its true filler than to an
            // unrelated random vector.
            assert!(
                sim_clean > 0.2,
                "filler {i}: recovery too weak, sim_clean={sim_clean}"
            );
            assert!(
                sim_clean > sim_noise + 0.1,
                "filler {i}: signal {sim_clean} not clearly above noise {sim_noise}"
            );
        }
    }

    #[test]
    fn bind_length_mismatch_errors() {
        let mut rng = LcgRng::new(6);
        let dim = 512;
        let role = PermutationRole::random(dim, &mut rng).expect("role");
        let wrong = random_binary(dim - 1, &mut rng);
        let err = role.bind(&wrong).expect_err("must reject");
        assert!(
            matches!(
                err,
                HdcError::PermutationLengthMismatch {
                    perm_len: 512,
                    dim: 511
                }
            ),
            "got {err:?}"
        );

        let wrong_i32 = random_integer(dim + 3, &mut rng);
        let err_i32 = role.unbind_i32(&wrong_i32).expect_err("must reject");
        assert!(
            matches!(
                err_i32,
                HdcError::PermutationLengthMismatch {
                    perm_len: 512,
                    dim: 515
                }
            ),
            "got {err_i32:?}"
        );
    }

    #[test]
    fn superpose_empty_input_errors() {
        let empty_roles: Vec<PermutationRole> = Vec::new();
        let empty_fillers: Vec<Vec<i32>> = Vec::new();
        let err = bind_superpose_i32(&empty_roles, &empty_fillers).expect_err("must reject");
        assert!(matches!(err, HdcError::EmptyInput), "got {err:?}");
    }

    #[test]
    fn superpose_length_mismatch_errors() {
        let mut rng = LcgRng::new(7);
        let dim = 256;
        let roles: Vec<PermutationRole> = (0..3)
            .map(|_| PermutationRole::random(dim, &mut rng).expect("role"))
            .collect();
        let fillers: Vec<Vec<i32>> = (0..2).map(|_| random_integer(dim, &mut rng)).collect();
        let err = bind_superpose_i32(&roles, &fillers).expect_err("must reject");
        assert!(
            matches!(
                err,
                HdcError::DimensionMismatch {
                    expected: 3,
                    got: 2
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let dim = 512;
        let build = || {
            let mut rng = LcgRng::new(12_345);
            let role = PermutationRole::random(dim, &mut rng).expect("role");
            let filler = random_integer(dim, &mut rng);
            let bound = role.bind_i32(&filler).expect("bind");
            (role.perm().to_vec(), bound)
        };
        let (perm_a, bound_a) = build();
        let (perm_b, bound_b) = build();
        assert_eq!(perm_a, perm_b, "permutation must be deterministic");
        assert_eq!(bound_a, bound_b, "binding must be deterministic");
    }
}
