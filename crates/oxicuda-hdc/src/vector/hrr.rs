//! Holographic Reduced Representations (Plate 1995).
//!
//! Real-valued hypervectors in R^D with circular convolution binding.
//! Random HVs are sampled from N(0,1) per component and then L2-normalised
//! so that cosine similarity equals the inner product directly.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::ops::binding::{circular_convolution, circular_correlation};

// ── Low-level primitives ────────────────────────────────────────────────────

/// L2-normalise `v` in place.
///
/// Returns the pre-normalisation Euclidean norm.
///
/// # Errors
///
/// - `HdcError::ZeroDimension` if `v` is empty.
/// - `HdcError::DivisionByZero` if the norm is less than 1e-12 (zero or near-zero vector).
pub fn hrr_normalize(v: &mut [f32]) -> HdcResult<f32> {
    if v.is_empty() {
        return Err(HdcError::ZeroDimension);
    }
    let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm < 1e-12 {
        return Err(HdcError::DivisionByZero);
    }
    let inv = 1.0 / norm;
    for x in v.iter_mut() {
        *x *= inv;
    }
    Ok(norm)
}

/// Generate a random unit-norm HRR hypervector of dimension `dim`.
///
/// Components are drawn i.i.d. from N(0,1) using Box-Muller sampling,
/// then the whole vector is L2-normalised.
///
/// # Errors
///
/// Returns `HdcError::ZeroDimension` if `dim == 0`.
pub fn random_hrr(dim: usize, rng: &mut LcgRng) -> HdcResult<Vec<f32>> {
    if dim == 0 {
        return Err(HdcError::ZeroDimension);
    }
    let mut v = Vec::with_capacity(dim);
    let mut idx = 0usize;
    while idx + 1 < dim {
        let (z0, z1) = rng.normal_pair_f32();
        v.push(z0);
        v.push(z1);
        idx += 2;
    }
    if v.len() < dim {
        // dim is odd; generate one more pair and take the first sample.
        let (z0, _) = rng.normal_pair_f32();
        v.push(z0);
    }
    hrr_normalize(&mut v)?;
    Ok(v)
}

// ── Binding / unbinding ─────────────────────────────────────────────────────

/// HRR bind: circular convolution of `a` and `b`.
///
/// Reuses `crate::ops::binding::circular_convolution` directly.
///
/// # Errors
///
/// Propagates errors from `circular_convolution` (empty input or dim mismatch).
#[inline]
pub fn hrr_bind(a: &[f32], b: &[f32]) -> HdcResult<Vec<f32>> {
    circular_convolution(a, b)
}

/// HRR unbind: approximate inverse of bind.
///
/// Computes `circular_correlation(key, bound)`, which retrieves an approximation
/// of the "other" vector when `bound ≈ circular_convolution(key, other)`.
///
/// # Errors
///
/// Propagates errors from `circular_correlation`.
#[inline]
pub fn hrr_unbind(key: &[f32], bound: &[f32]) -> HdcResult<Vec<f32>> {
    circular_correlation(key, bound)
}

/// Bind a non-empty sequence of HVs left-associatively.
///
/// `bind_sequence([a, b, c]) = bind(bind(a, b), c)`
///
/// A single-element slice returns a clone of the sole element.
///
/// # Errors
///
/// - `HdcError::EmptyInput` if the slice is empty.
/// - Propagates dimension mismatches from `hrr_bind`.
pub fn hrr_bind_sequence(hvs: &[Vec<f32>]) -> HdcResult<Vec<f32>> {
    if hvs.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let mut acc = hvs[0].clone();
    for hv in &hvs[1..] {
        acc = hrr_bind(&acc, hv)?;
    }
    Ok(acc)
}

// ── Bundling ─────────────────────────────────────────────────────────────────

/// Add `src` into `acc` element-wise (in-place superposition).
///
/// # Errors
///
/// `HdcError::DimensionMismatch` if lengths differ.
pub fn hrr_bundle_add(acc: &mut [f32], src: &[f32]) -> HdcResult<()> {
    if acc.len() != src.len() {
        return Err(HdcError::DimensionMismatch {
            expected: acc.len(),
            got: src.len(),
        });
    }
    for (a, &s) in acc.iter_mut().zip(src.iter()) {
        *a += s;
    }
    Ok(())
}

/// Element-wise sum of all `hvs`, then L2-normalised.
///
/// # Errors
///
/// - `HdcError::EmptyInput` if `hvs` is empty.
/// - `HdcError::DimensionMismatch` if any HV has a different length to the first.
/// - `HdcError::DivisionByZero` if the summed vector has zero norm.
pub fn hrr_bundle(hvs: &[Vec<f32>]) -> HdcResult<Vec<f32>> {
    if hvs.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    let dim = hvs[0].len();
    let mut acc = vec![0f32; dim];
    for hv in hvs {
        hrr_bundle_add(&mut acc, hv)?;
    }
    hrr_normalize(&mut acc)?;
    Ok(acc)
}

// ── Similarity ───────────────────────────────────────────────────────────────

/// Cosine similarity between `a` and `b`.
///
/// Computes `(a · b) / (‖a‖ · ‖b‖)` for generality even when the inputs are
/// already unit-norm.
///
/// # Errors
///
/// - `HdcError::EmptyInput` if either slice is empty.
/// - `HdcError::DimensionMismatch` if lengths differ.
/// - `HdcError::DivisionByZero` if either vector has zero norm.
pub fn hrr_cosine(a: &[f32], b: &[f32]) -> HdcResult<f32> {
    if a.is_empty() || b.is_empty() {
        return Err(HdcError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(HdcError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return Err(HdcError::DivisionByZero);
    }
    Ok(dot / (norm_a * norm_b))
}

// ── HRR Item Memory ───────────────────────────────────────────────────────────

/// Item memory for unit-norm real-valued HRR hypervectors.
///
/// Associates integer IDs with unit-norm `Vec<f32>` hypervectors and supports
/// nearest-neighbour lookup by cosine similarity (dot product, since the stored
/// HVs are unit-norm).
#[derive(Debug, Clone)]
pub struct HrrItemMemory {
    dim: usize,
    items: Vec<(usize, Vec<f32>)>,
}

impl HrrItemMemory {
    /// Create an empty item memory for HVs of the given dimension.
    ///
    /// # Errors
    ///
    /// `HdcError::ZeroDimension` if `dim == 0`.
    pub fn new(dim: usize) -> HdcResult<Self> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        Ok(Self {
            dim,
            items: Vec::new(),
        })
    }

    /// Insert a unit-norm HV for the given ID.
    ///
    /// # Errors
    ///
    /// `HdcError::DimensionMismatch` if `hv.len() != self.dim`.
    pub fn insert(&mut self, id: usize, hv: Vec<f32>) -> HdcResult<()> {
        if hv.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: hv.len(),
            });
        }
        self.items.push((id, hv));
        Ok(())
    }

    /// Generate a random HRR HV and insert it for the given ID.
    ///
    /// # Errors
    ///
    /// Propagates errors from `random_hrr`.
    pub fn insert_random(&mut self, id: usize, rng: &mut LcgRng) -> HdcResult<()> {
        let hv = random_hrr(self.dim, rng)?;
        self.items.push((id, hv));
        Ok(())
    }

    /// Find the item whose HV has the maximum dot product with `probe`.
    ///
    /// Returns `(id, cosine_score)`.
    ///
    /// # Errors
    ///
    /// `HdcError::EmptyItemMemory` if no items have been inserted.
    pub fn query(&self, probe: &[f32]) -> HdcResult<(usize, f32)> {
        if self.items.is_empty() {
            return Err(HdcError::EmptyItemMemory);
        }
        let mut best_id = self.items[0].0;
        let mut best_score = f32::NEG_INFINITY;
        for (id, hv) in &self.items {
            let score: f32 = probe.iter().zip(hv.iter()).map(|(&p, &h)| p * h).sum();
            if score > best_score {
                best_score = score;
                best_id = *id;
            }
        }
        Ok((best_id, best_score))
    }

    /// Find the nearest item and return `(id, score, &hv)`.
    ///
    /// # Errors
    ///
    /// - `HdcError::EmptyItemMemory` if no items have been inserted.
    /// - `HdcError::DimensionMismatch` if `probe.len() != self.dim`.
    pub fn query_with_hv<'a>(&'a self, probe: &[f32]) -> HdcResult<(usize, f32, &'a [f32])> {
        if self.items.is_empty() {
            return Err(HdcError::EmptyItemMemory);
        }
        if probe.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: probe.len(),
            });
        }
        let mut best_id = self.items[0].0;
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = 0usize;
        for (idx, (id, hv)) in self.items.iter().enumerate() {
            let score: f32 = probe.iter().zip(hv.iter()).map(|(&p, &h)| p * h).sum();
            if score > best_score {
                best_score = score;
                best_id = *id;
                best_idx = idx;
            }
        }
        Ok((best_id, best_score, &self.items[best_idx].1))
    }

    /// Return the HV stored for `id`.
    ///
    /// # Errors
    ///
    /// `HdcError::ItemNotFound` if no item with that ID exists.
    pub fn get_hv(&self, id: usize) -> HdcResult<&[f32]> {
        for (sid, hv) in &self.items {
            if *sid == id {
                return Ok(hv.as_slice());
            }
        }
        Err(HdcError::ItemNotFound(id))
    }

    /// Number of items stored.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// True if no items have been stored.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The hypervector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rng() -> LcgRng {
        LcgRng::new(0xDEAD_BEEF_CAFE_1234)
    }

    // ── random_hrr ──────────────────────────────────────────────────────────

    #[test]
    fn random_hrr_length_equals_dim() {
        let mut rng = rng();
        let v = random_hrr(512, &mut rng).expect("random_hrr failed");
        assert_eq!(v.len(), 512);
    }

    #[test]
    fn random_hrr_unit_norm() {
        let mut rng = rng();
        let v = random_hrr(512, &mut rng).expect("random_hrr failed");
        let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm deviates from 1: {norm}");
    }

    #[test]
    fn random_hrr_odd_dim_length() {
        let mut rng = rng();
        let v = random_hrr(513, &mut rng).expect("random_hrr odd dim");
        assert_eq!(v.len(), 513);
    }

    #[test]
    fn random_hrr_zero_dim_error() {
        let mut rng = rng();
        let res = random_hrr(0, &mut rng);
        assert!(matches!(res, Err(HdcError::ZeroDimension)));
    }

    // ── hrr_normalize ───────────────────────────────────────────────────────

    #[test]
    fn hrr_normalize_produces_unit_norm() {
        let mut v = vec![3.0f32, 4.0f32];
        hrr_normalize(&mut v).expect("normalize failed");
        let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hrr_normalize_zero_vec_is_err() {
        let mut v = vec![0.0f32; 64];
        let res = hrr_normalize(&mut v);
        assert!(matches!(res, Err(HdcError::DivisionByZero)));
    }

    #[test]
    fn hrr_normalize_returns_prenorm() {
        let mut v = vec![3.0f32, 4.0f32]; // norm = 5
        let prenorm = hrr_normalize(&mut v).expect("normalize failed");
        assert!((prenorm - 5.0).abs() < 1e-6, "prenorm = {prenorm}");
    }

    // ── hrr_bind ────────────────────────────────────────────────────────────

    #[test]
    fn hrr_bind_output_length() {
        let mut rng = rng();
        let a = random_hrr(128, &mut rng).expect("a");
        let b = random_hrr(128, &mut rng).expect("b");
        let c = hrr_bind(&a, &b).expect("bind");
        assert_eq!(c.len(), 128);
    }

    #[test]
    fn hrr_bind_dim_mismatch_error() {
        let mut rng = rng();
        let a = random_hrr(128, &mut rng).expect("a");
        let b = random_hrr(256, &mut rng).expect("b");
        let res = hrr_bind(&a, &b);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    // ── hrr_unbind ──────────────────────────────────────────────────────────

    #[test]
    fn hrr_unbind_approximate_inverse() {
        // For random unit-norm a and b, correlation(a, conv(a,b)) ≈ b.
        // The cosine similarity should be noticeably positive (> 0.5) for
        // moderate D where HRR approximate inverse holds well.
        let mut rng = rng();
        let dim = 256;
        let a = random_hrr(dim, &mut rng).expect("a");
        let b = random_hrr(dim, &mut rng).expect("b");
        let bound = hrr_bind(&a, &b).expect("bind");
        let retrieved = hrr_unbind(&a, &bound).expect("unbind");
        let sim = hrr_cosine(&retrieved, &b).expect("cosine");
        assert!(
            sim > 0.5,
            "cosine similarity of retrieved vs b too low: {sim:.4}"
        );
    }

    // ── hrr_cosine ──────────────────────────────────────────────────────────

    #[test]
    fn hrr_cosine_self_is_one() {
        let mut rng = rng();
        let v = random_hrr(512, &mut rng).expect("v");
        let sim = hrr_cosine(&v, &v).expect("cosine");
        assert!((sim - 1.0).abs() < 1e-5, "self-cosine = {sim}");
    }

    #[test]
    fn hrr_cosine_empty_is_err() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let res = hrr_cosine(&a, &b);
        assert!(matches!(res, Err(HdcError::EmptyInput)));
    }

    #[test]
    fn hrr_cosine_dim_mismatch_error() {
        let a = vec![1.0f32, 0.0f32];
        let b = vec![1.0f32, 0.0f32, 0.0f32];
        let res = hrr_cosine(&a, &b);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    // ── hrr_bundle ──────────────────────────────────────────────────────────

    #[test]
    fn hrr_bundle_unit_norm() {
        let mut rng = rng();
        let hvs: Vec<Vec<f32>> = (0..8)
            .map(|_| random_hrr(256, &mut rng).expect("hv"))
            .collect();
        let bundled = hrr_bundle(&hvs).expect("bundle");
        let norm: f32 = bundled.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "bundle norm = {norm}");
    }

    #[test]
    fn hrr_bundle_single_element_equals_normalized() {
        let mut rng = rng();
        let mut v = random_hrr(128, &mut rng).expect("v");
        // Perturb to make it non-unit-norm.
        for x in v.iter_mut() {
            *x *= 3.7;
        }
        let bundled = hrr_bundle(&[v.clone()]).expect("bundle");
        let mut expected = v.clone();
        hrr_normalize(&mut expected).expect("normalize");
        for (a, b) in bundled.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn hrr_bundle_dim_mismatch_error() {
        let mut rng = rng();
        let a = random_hrr(128, &mut rng).expect("a");
        let b = random_hrr(256, &mut rng).expect("b");
        let res = hrr_bundle(&[a, b]);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn hrr_bundle_empty_error() {
        let res = hrr_bundle(&[]);
        assert!(matches!(res, Err(HdcError::EmptyInput)));
    }

    // ── hrr_bundle_add ──────────────────────────────────────────────────────

    #[test]
    fn hrr_bundle_add_correct_length() {
        let mut acc = vec![1.0f32; 64];
        let src = vec![2.0f32; 64];
        hrr_bundle_add(&mut acc, &src).expect("bundle_add");
        assert_eq!(acc.len(), 64);
        assert!((acc[0] - 3.0).abs() < 1e-7);
    }

    #[test]
    fn hrr_bundle_add_dim_mismatch_error() {
        let mut acc = vec![0.0f32; 64];
        let src = vec![0.0f32; 32];
        let res = hrr_bundle_add(&mut acc, &src);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    // ── hrr_bind_sequence ───────────────────────────────────────────────────

    #[test]
    fn hrr_bind_sequence_two_equals_bind() {
        let mut rng = rng();
        let a = random_hrr(64, &mut rng).expect("a");
        let b = random_hrr(64, &mut rng).expect("b");
        let direct = hrr_bind(&a, &b).expect("direct bind");
        let seq = hrr_bind_sequence(&[a, b]).expect("bind_sequence");
        for (d, s) in direct.iter().zip(seq.iter()) {
            assert!((d - s).abs() < 1e-6, "mismatch: {d} vs {s}");
        }
    }

    #[test]
    fn hrr_bind_sequence_empty_error() {
        let hvs: Vec<Vec<f32>> = vec![];
        let res = hrr_bind_sequence(&hvs);
        assert!(matches!(res, Err(HdcError::EmptyInput)));
    }

    // ── HrrItemMemory ────────────────────────────────────────────────────────

    #[test]
    fn hrr_item_memory_new_ok() {
        let mem = HrrItemMemory::new(512).expect("new");
        assert_eq!(mem.dim(), 512);
        assert!(mem.is_empty());
        assert_eq!(mem.len(), 0);
    }

    #[test]
    fn hrr_item_memory_zero_dim_error() {
        let res = HrrItemMemory::new(0);
        assert!(matches!(res, Err(HdcError::ZeroDimension)));
    }

    #[test]
    fn hrr_item_memory_insert_and_query_correct_id() {
        let mut rng = rng();
        let mut mem = HrrItemMemory::new(256).expect("new");
        for id in 0..5 {
            mem.insert_random(id, &mut rng).expect("insert_random");
        }
        // Exact match should return the correct id.
        let hv = mem.get_hv(2).expect("get_hv").to_vec();
        let (found_id, _score) = mem.query(&hv).expect("query");
        assert_eq!(found_id, 2);
    }

    #[test]
    fn hrr_item_memory_query_with_hv_hv_len() {
        let mut rng = rng();
        let mut mem = HrrItemMemory::new(128).expect("new");
        for id in 0..4 {
            mem.insert_random(id, &mut rng).expect("insert_random");
        }
        let probe = mem.get_hv(1).expect("probe").to_vec();
        let (_, _, hv_ref) = mem.query_with_hv(&probe).expect("query_with_hv");
        assert_eq!(hv_ref.len(), 128);
    }

    #[test]
    fn hrr_item_memory_insert_wrong_dim_error() {
        let mut mem = HrrItemMemory::new(128).expect("new");
        let wrong = vec![0.0f32; 64];
        let res = mem.insert(0, wrong);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    #[test]
    fn hrr_item_memory_empty_query_error() {
        let mem = HrrItemMemory::new(128).expect("new");
        let probe = vec![0.0f32; 128];
        let res = mem.query(&probe);
        assert!(matches!(res, Err(HdcError::EmptyItemMemory)));
    }

    #[test]
    fn hrr_item_memory_get_hv_not_found_error() {
        let mem = HrrItemMemory::new(64).expect("new");
        let res = mem.get_hv(99);
        assert!(matches!(res, Err(HdcError::ItemNotFound(99))));
    }
}
