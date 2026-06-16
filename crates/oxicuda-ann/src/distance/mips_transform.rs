//! Maximum-Inner-Product-Search (MIPS) to nearest-neighbour (L2) reduction.
//!
//! References:
//! - Shrivastava & Li, "Asymmetric LSH (ALSH) for Sublinear Time Maximum Inner
//!   Product Search (MIPS)", NeurIPS 2014.
//! - Bachrach et al., "Speeding Up the Xbox Recommender System Using a Euclidean
//!   Transformation for Inner-Product Spaces", RecSys 2014 (the "XBox" trick).
//!
//! MIPS asks for `argmax_x q·x`.  This is *not* directly a nearest-neighbour
//! problem because inner product is not a metric (it is unbounded and not
//! symmetric in the "self" term).  The transforms here map database vectors and
//! queries into a slightly higher-dimensional space such that **L2 nearest
//! neighbour in the transformed space ⇔ maximum inner product in the original
//! space**, which lets any off-the-shelf L2 index (IVF, HNSW, flat) answer MIPS
//! queries.
//!
//! ## XBox transform (norm-augmentation)
//!
//! Pick `m_factor ≥ max_x ‖x‖`.  Map each database vector to
//!
//! ```text
//! P(x) = [ x ;  sqrt(m_factor² − ‖x‖²) ]      (one extra coordinate)
//! ```
//!
//! and each query to
//!
//! ```text
//! Q(q) = [ q ;  0 ]                            (zero in the extra coordinate)
//! ```
//!
//! Then every transformed database vector has the **same** norm `m_factor`, so
//!
//! ```text
//! ‖Q(q) − P(x)‖² = ‖q‖² + m_factor² − 2 q·x.
//! ```
//!
//! Since `‖q‖²` and `m_factor²` are constant across the database,
//! `argmin_x ‖Q(q) − P(x)‖²  =  argmax_x q·x`.  One extra dimension, `O(1)`
//! query preprocessing.
//!
//! ## Shrivastava–Li transform (`m`-fold augmentation)
//!
//! The ALSH construction first scales the database so `max ‖x‖ ≤ U < 1`, then
//! appends `m` coordinates holding `‖x‖²`, `‖x‖⁴`, …, `‖x‖^{2^m}`, which drive
//! the augmented database norm to a near-constant as `m` grows.  Queries are
//! padded with `m` halves.  We expose this as an alternative for callers who
//! prefer the original ALSH formulation.
use crate::error::{AnnError, AnnResult};

/// A fitted XBox MIPS→L2 transform.
///
/// Stores the augmentation scale `m_factor` (an upper bound on the database
/// vector norms) and the original dimensionality.  Apply [`Self::transform_db`]
/// to database vectors and [`Self::transform_query`] to queries, then index /
/// search the results with any squared-L2 routine.
#[derive(Debug, Clone)]
pub struct XBoxTransform {
    m_factor: f32,
    dim: usize,
}

impl XBoxTransform {
    /// Build a transform whose augmentation scale is the maximum database norm.
    ///
    /// `data` is row-major `[n, dim]`.  The scale is set to `max_x ‖x‖` (times a
    /// small `1 + eps` slack so the extra coordinate stays real-valued under
    /// floating-point round-off).
    ///
    /// # Errors
    /// - [`AnnError::EmptyInput`] when `n == 0`.
    /// - [`AnnError::InvalidVectorDim`] when `dim == 0`.
    /// - [`AnnError::DimensionMismatch`] when `data.len() != n * dim`.
    pub fn fit(data: &[f32], n: usize, dim: usize) -> AnnResult<Self> {
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        if dim == 0 {
            return Err(AnnError::InvalidVectorDim { dim: 0 });
        }
        if data.len() != n * dim {
            return Err(AnnError::DimensionMismatch {
                expected: n * dim,
                got: data.len(),
            });
        }
        let mut max_norm_sq = 0.0_f32;
        for i in 0..n {
            let x = &data[i * dim..(i + 1) * dim];
            let ns: f32 = x.iter().map(|v| v * v).sum();
            if ns > max_norm_sq {
                max_norm_sq = ns;
            }
        }
        // 1 + eps slack guards sqrt(m² − ‖x‖²) against negative round-off.
        let m_factor = (max_norm_sq.sqrt() * 1.000_001).max(f32::MIN_POSITIVE);
        Ok(Self { m_factor, dim })
    }

    /// Build a transform with an explicit augmentation scale.
    ///
    /// `m_factor` must be ≥ every database vector norm; otherwise the extra
    /// coordinate is undefined (negative under the square root).
    ///
    /// # Errors
    /// - [`AnnError::InvalidVectorDim`] when `dim == 0`.
    /// - [`AnnError::Internal`] when `m_factor` is not finite and positive.
    pub fn with_scale(m_factor: f32, dim: usize) -> AnnResult<Self> {
        if dim == 0 {
            return Err(AnnError::InvalidVectorDim { dim: 0 });
        }
        if !m_factor.is_finite() || m_factor <= 0.0 {
            return Err(AnnError::Internal {
                msg: format!("xbox: m_factor must be finite and > 0, got {m_factor}"),
            });
        }
        Ok(Self { m_factor, dim })
    }

    /// Augmentation scale `m_factor` (the common norm of every transformed
    /// database vector).
    #[must_use]
    pub fn m_factor(&self) -> f32 {
        self.m_factor
    }

    /// Original (pre-transform) dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Dimensionality after transformation (`dim + 1`).
    #[must_use]
    pub fn transformed_dim(&self) -> usize {
        self.dim + 1
    }

    /// Transform a single database vector `x → [ x ; sqrt(m² − ‖x‖²) ]`.
    ///
    /// # Errors
    /// - [`AnnError::DimensionMismatch`] when `x.len() != dim`.
    /// - [`AnnError::Internal`] when `‖x‖ > m_factor` (negative under the root).
    pub fn transform_db(&self, x: &[f32]) -> AnnResult<Vec<f32>> {
        if x.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        let norm_sq: f32 = x.iter().map(|v| v * v).sum();
        let extra_sq = self.m_factor * self.m_factor - norm_sq;
        if extra_sq < -1e-3 {
            return Err(AnnError::Internal {
                msg: format!(
                    "xbox: ‖x‖²={norm_sq} exceeds m²={}; refit with a larger scale",
                    self.m_factor * self.m_factor
                ),
            });
        }
        let mut out = Vec::with_capacity(self.dim + 1);
        out.extend_from_slice(x);
        out.push(extra_sq.max(0.0).sqrt());
        Ok(out)
    }

    /// Transform `n` database vectors at once, returning `[n, dim + 1]`.
    ///
    /// # Errors
    /// Propagates [`Self::transform_db`] errors; also
    /// [`AnnError::DimensionMismatch`] when `data.len() != n * dim`.
    pub fn transform_db_batch(&self, data: &[f32], n: usize) -> AnnResult<Vec<f32>> {
        if data.len() != n * self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: n * self.dim,
                got: data.len(),
            });
        }
        let mut out = Vec::with_capacity(n * (self.dim + 1));
        for i in 0..n {
            let t = self.transform_db(&data[i * self.dim..(i + 1) * self.dim])?;
            out.extend_from_slice(&t);
        }
        Ok(out)
    }

    /// Transform a query `q → [ q ; 0 ]`.
    ///
    /// The query may optionally be L2-normalised first (set `normalize = true`);
    /// scaling the query does not change the `argmax` of `q·x`, but it makes the
    /// recovered transformed distances comparable across queries.
    ///
    /// # Errors
    /// [`AnnError::DimensionMismatch`] when `q.len() != dim`.
    pub fn transform_query(&self, q: &[f32], normalize: bool) -> AnnResult<Vec<f32>> {
        if q.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: q.len(),
            });
        }
        let mut out = Vec::with_capacity(self.dim + 1);
        if normalize {
            let norm: f32 = q.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > f32::MIN_POSITIVE {
                let inv = 1.0 / norm;
                out.extend(q.iter().map(|v| v * inv));
            } else {
                out.extend_from_slice(q);
            }
        } else {
            out.extend_from_slice(q);
        }
        out.push(0.0);
        Ok(out)
    }

    /// Recover the inner product `q·x` from the transformed squared-L2 distance.
    ///
    /// Given `d² = ‖Q(q) − P(x)‖²`, and the (un-normalised) query norm,
    /// `q·x = (‖q‖² + m² − d²) / 2`.  Useful for converting an L2 index's scores
    /// back into inner products for re-ranking.
    #[must_use]
    pub fn recover_inner_product(&self, transformed_dist_sq: f32, query_norm_sq: f32) -> f32 {
        (query_norm_sq + self.m_factor * self.m_factor - transformed_dist_sq) / 2.0
    }
}

/// Brute-force MIPS via the XBox transform, returning the top-`k` `(id, ip)`
/// pairs ordered by **descending** inner product.
///
/// This is the canonical correctness oracle: it transforms the database and the
/// query, runs an exact squared-L2 search, then converts the distances back to
/// inner products.  Real callers would replace the inner L2 search with an
/// approximate index.
///
/// # Errors
/// - [`AnnError::EmptyInput`] when `n == 0`.
/// - [`AnnError::InvalidK`] when `k == 0` or `k > n`.
/// - [`AnnError::DimensionMismatch`] when shapes disagree.
pub fn mips_search_xbox(
    transform: &XBoxTransform,
    data: &[f32],
    n: usize,
    query: &[f32],
    k: usize,
) -> AnnResult<Vec<(usize, f32)>> {
    if n == 0 {
        return Err(AnnError::EmptyInput);
    }
    if k == 0 || k > n {
        return Err(AnnError::InvalidK { k, n });
    }
    let dim = transform.dim();
    if data.len() != n * dim {
        return Err(AnnError::DimensionMismatch {
            expected: n * dim,
            got: data.len(),
        });
    }
    let q_t = transform.transform_query(query, false)?;
    let query_norm_sq: f32 = query.iter().map(|v| v * v).sum();

    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(n);
    for i in 0..n {
        let x = &data[i * dim..(i + 1) * dim];
        let x_t = transform.transform_db(x)?;
        let dist_sq: f32 = q_t
            .iter()
            .zip(x_t.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let ip = transform.recover_inner_product(dist_sq, query_norm_sq);
        scored.push((i, ip));
    }
    // Descending by inner product.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    Ok(scored)
}

/// A fitted Shrivastava–Li (ALSH) MIPS→L2 transform with `m`-fold augmentation.
///
/// Database vectors are first scaled by `scale` so the largest norm is `≤ u`,
/// then `m` coordinates `(‖x‖², ‖x‖⁴, …, ‖x‖^{2^m})` are appended.  Queries are
/// L2-normalised and padded with `m` halves.
#[derive(Debug, Clone)]
pub struct ShrivastavaLiTransform {
    scale: f32,
    m_aug: usize,
    dim: usize,
}

impl ShrivastavaLiTransform {
    /// Fit the transform: choose `scale = u / max‖x‖` so the scaled database has
    /// max norm `u`, with `m` appended power coordinates.
    ///
    /// `u` must lie in `(0, 1)` (the ALSH analysis requires `U < 1`); typical
    /// choices are `0.83` with `m = 3`.
    ///
    /// # Errors
    /// - [`AnnError::EmptyInput`] when `n == 0`.
    /// - [`AnnError::InvalidVectorDim`] when `dim == 0`.
    /// - [`AnnError::InvalidLayerCount`] when `m == 0`.
    /// - [`AnnError::DimensionMismatch`] when `data.len() != n * dim`.
    /// - [`AnnError::Internal`] when `u` is outside `(0, 1)`.
    pub fn fit(data: &[f32], n: usize, dim: usize, m: usize, u: f32) -> AnnResult<Self> {
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        if dim == 0 {
            return Err(AnnError::InvalidVectorDim { dim: 0 });
        }
        if m == 0 {
            return Err(AnnError::InvalidLayerCount { n: 0 });
        }
        if data.len() != n * dim {
            return Err(AnnError::DimensionMismatch {
                expected: n * dim,
                got: data.len(),
            });
        }
        if !(u.is_finite() && u > 0.0 && u < 1.0) {
            return Err(AnnError::Internal {
                msg: format!("alsh: U must be in (0, 1), got {u}"),
            });
        }
        let mut max_norm = 0.0_f32;
        for i in 0..n {
            let x = &data[i * dim..(i + 1) * dim];
            let ns: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            if ns > max_norm {
                max_norm = ns;
            }
        }
        let scale = if max_norm > f32::MIN_POSITIVE {
            u / max_norm
        } else {
            1.0
        };
        Ok(Self {
            scale,
            m_aug: m,
            dim,
        })
    }

    /// Number of appended power coordinates `m`.
    #[must_use]
    pub fn m_aug(&self) -> usize {
        self.m_aug
    }

    /// Transformed dimensionality `dim + m`.
    #[must_use]
    pub fn transformed_dim(&self) -> usize {
        self.dim + self.m_aug
    }

    /// Original dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Transform a database vector: scale, then append `‖Sx‖^{2^i}` for
    /// `i = 1..=m`.
    ///
    /// # Errors
    /// [`AnnError::DimensionMismatch`] when `x.len() != dim`.
    pub fn transform_db(&self, x: &[f32]) -> AnnResult<Vec<f32>> {
        if x.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        let mut out = Vec::with_capacity(self.dim + self.m_aug);
        out.extend(x.iter().map(|v| v * self.scale));
        let norm_sq: f32 = out.iter().map(|v| v * v).sum();
        // Append ‖x‖², ‖x‖⁴, ‖x‖⁸, … = (‖x‖²)^{2^{i-1}}.
        let mut power = norm_sq;
        for _ in 0..self.m_aug {
            out.push(power);
            power *= power;
        }
        Ok(out)
    }

    /// Transform a query: L2-normalise, then append `m` halves (the ALSH query
    /// map `Q(q) = [q/‖q‖ ; 1/2 ; … ; 1/2]`).
    ///
    /// # Errors
    /// [`AnnError::DimensionMismatch`] when `q.len() != dim`.
    pub fn transform_query(&self, q: &[f32]) -> AnnResult<Vec<f32>> {
        if q.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: q.len(),
            });
        }
        let norm: f32 = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mut out = Vec::with_capacity(self.dim + self.m_aug);
        if norm > f32::MIN_POSITIVE {
            let inv = 1.0 / norm;
            out.extend(q.iter().map(|v| v * inv));
        } else {
            out.extend_from_slice(q);
        }
        out.extend(std::iter::repeat_n(0.5_f32, self.m_aug));
        Ok(out)
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rand_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        (0..n * dim).map(|_| rng.next_f32() - 0.5).collect()
    }

    fn exact_mips(data: &[f32], n: usize, dim: usize, q: &[f32], k: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let x = &data[i * dim..(i + 1) * dim];
                let ip: f32 = q.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                (i, ip)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(i, _)| i).collect()
    }

    #[test]
    fn xbox_transformed_dim() {
        let data = rand_data(10, 6, 1);
        let t = XBoxTransform::fit(&data, 10, 6).expect("test invariant: should succeed");
        assert_eq!(t.transformed_dim(), 7);
        assert_eq!(t.dim(), 6);
    }

    #[test]
    fn xbox_all_db_norms_equal() {
        // Every transformed database vector must have norm == m_factor.
        let data = rand_data(20, 5, 2);
        let t = XBoxTransform::fit(&data, 20, 5).expect("test invariant: should succeed");
        let m2 = t.m_factor() * t.m_factor();
        for i in 0..20 {
            let xt = t
                .transform_db(&data[i * 5..(i + 1) * 5])
                .expect("test invariant: should succeed");
            let ns: f32 = xt.iter().map(|v| v * v).sum();
            assert!((ns - m2).abs() < 1e-3, "norm²={ns} m²={m2}");
        }
    }

    #[test]
    fn xbox_query_extra_coord_zero() {
        let data = rand_data(8, 4, 3);
        let t = XBoxTransform::fit(&data, 8, 4).expect("test invariant: should succeed");
        let qt = t
            .transform_query(&[1.0, 2.0, 3.0, 4.0], false)
            .expect("test invariant: should succeed");
        assert_eq!(qt.len(), 5);
        assert_eq!(qt[4], 0.0);
    }

    #[test]
    fn xbox_recover_inner_product_exact() {
        // ‖Q(q) − P(x)‖² should recover exactly q·x.
        let data = rand_data(12, 5, 4);
        let t = XBoxTransform::fit(&data, 12, 5).expect("test invariant: should succeed");
        let q = vec![0.3_f32, -0.2, 0.5, 0.1, -0.4];
        let qn2: f32 = q.iter().map(|v| v * v).sum();
        let qt = t
            .transform_query(&q, false)
            .expect("test invariant: should succeed");
        for i in 0..12 {
            let x = &data[i * 5..(i + 1) * 5];
            let xt = t.transform_db(x).expect("test invariant: should succeed");
            let d2: f32 = qt
                .iter()
                .zip(xt.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let recovered = t.recover_inner_product(d2, qn2);
            let exact: f32 = q.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            assert!(
                (recovered - exact).abs() < 1e-3,
                "rec={recovered} exact={exact}"
            );
        }
    }

    #[test]
    fn xbox_argmin_l2_equals_argmax_ip() {
        // The L2-nearest transformed db vector must be the inner-product max.
        let n = 50;
        let dim = 8;
        let data = rand_data(n, dim, 5);
        let t = XBoxTransform::fit(&data, n, dim).expect("test invariant: should succeed");
        let q = rand_data(1, dim, 99);
        let qt = t
            .transform_query(&q, false)
            .expect("test invariant: should succeed");

        let mut best_l2 = (0usize, f32::INFINITY);
        for i in 0..n {
            let xt = t
                .transform_db(&data[i * dim..(i + 1) * dim])
                .expect("test invariant: should succeed");
            let d2: f32 = qt
                .iter()
                .zip(xt.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if d2 < best_l2.1 {
                best_l2 = (i, d2);
            }
        }
        let exact = exact_mips(&data, n, dim, &q, 1);
        assert_eq!(best_l2.0, exact[0], "L2-argmin ≠ IP-argmax");
    }

    #[test]
    fn xbox_search_topk_matches_exact() {
        let n = 60;
        let dim = 8;
        let data = rand_data(n, dim, 6);
        let t = XBoxTransform::fit(&data, n, dim).expect("test invariant: should succeed");
        let q = rand_data(1, dim, 77);
        let res = mips_search_xbox(&t, &data, n, &q, 5).expect("test invariant: should succeed");
        let ids: Vec<usize> = res.iter().map(|(i, _)| *i).collect();
        let exact = exact_mips(&data, n, dim, &q, 5);
        assert_eq!(ids, exact, "top-5 MIPS mismatch");
    }

    #[test]
    fn xbox_search_descending_ip() {
        let n = 40;
        let dim = 6;
        let data = rand_data(n, dim, 7);
        let t = XBoxTransform::fit(&data, n, dim).expect("test invariant: should succeed");
        let q = rand_data(1, dim, 55);
        let res = mips_search_xbox(&t, &data, n, &q, 10).expect("test invariant: should succeed");
        for w in res.windows(2) {
            assert!(
                w[0].1 >= w[1].1 - 1e-5,
                "not descending: {} < {}",
                w[0].1,
                w[1].1
            );
        }
    }

    #[test]
    fn xbox_recovered_ip_matches_brute() {
        let n = 30;
        let dim = 5;
        let data = rand_data(n, dim, 8);
        let t = XBoxTransform::fit(&data, n, dim).expect("test invariant: should succeed");
        let q = rand_data(1, dim, 44);
        let res = mips_search_xbox(&t, &data, n, &q, n).expect("test invariant: should succeed");
        for (id, ip) in &res {
            let x = &data[id * dim..(id + 1) * dim];
            let exact: f32 = q.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            assert!((ip - exact).abs() < 1e-2, "id={id} ip={ip} exact={exact}");
        }
    }

    #[test]
    fn xbox_with_scale_constructor() {
        let t = XBoxTransform::with_scale(2.0, 4).expect("test invariant: should succeed");
        assert!((t.m_factor() - 2.0).abs() < 1e-6);
        assert_eq!(t.transformed_dim(), 5);
    }

    #[test]
    fn xbox_err_empty() {
        let err = XBoxTransform::fit(&[], 0, 4).unwrap_err();
        assert!(matches!(err, AnnError::EmptyInput));
    }

    #[test]
    fn xbox_err_zero_dim() {
        let err = XBoxTransform::fit(&[1.0], 1, 0).unwrap_err();
        assert!(matches!(err, AnnError::InvalidVectorDim { .. }));
    }

    #[test]
    fn xbox_err_dim_mismatch_transform() {
        let data = rand_data(5, 4, 1);
        let t = XBoxTransform::fit(&data, 5, 4).expect("test invariant: should succeed");
        let err = t.transform_db(&[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, AnnError::DimensionMismatch { .. }));
    }

    #[test]
    fn xbox_err_norm_exceeds_scale() {
        // A db vector larger than the configured scale must error.
        let t = XBoxTransform::with_scale(0.1, 3).expect("test invariant: should succeed");
        let err = t.transform_db(&[10.0, 10.0, 10.0]).unwrap_err();
        assert!(matches!(err, AnnError::Internal { .. }));
    }

    #[test]
    fn xbox_search_err_bad_k() {
        let data = rand_data(5, 4, 1);
        let t = XBoxTransform::fit(&data, 5, 4).expect("test invariant: should succeed");
        let err = mips_search_xbox(&t, &data, 5, &[1.0, 2.0, 3.0, 4.0], 0).unwrap_err();
        assert!(matches!(err, AnnError::InvalidK { .. }));
    }

    #[test]
    fn alsh_transformed_dim() {
        let data = rand_data(10, 6, 2);
        let t = ShrivastavaLiTransform::fit(&data, 10, 6, 3, 0.83)
            .expect("test invariant: should succeed");
        assert_eq!(t.transformed_dim(), 9);
        assert_eq!(t.m_aug(), 3);
    }

    #[test]
    fn alsh_db_norm_below_u() {
        // After scaling, the leading (original) block has norm ≤ U.
        let data = rand_data(20, 5, 3);
        let u = 0.83;
        let t = ShrivastavaLiTransform::fit(&data, 20, 5, 3, u)
            .expect("test invariant: should succeed");
        for i in 0..20 {
            let xt = t
                .transform_db(&data[i * 5..(i + 1) * 5])
                .expect("test invariant: should succeed");
            let lead_norm: f32 = xt[..5].iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(lead_norm <= u + 1e-4, "lead norm {lead_norm} > U {u}");
        }
    }

    #[test]
    fn alsh_query_halves_appended() {
        let data = rand_data(8, 4, 4);
        let t = ShrivastavaLiTransform::fit(&data, 8, 4, 2, 0.8)
            .expect("test invariant: should succeed");
        let qt = t
            .transform_query(&[1.0, 0.0, 0.0, 0.0])
            .expect("test invariant: should succeed");
        assert_eq!(qt.len(), 6);
        assert!((qt[4] - 0.5).abs() < 1e-6);
        assert!((qt[5] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn alsh_argmin_l2_recovers_top_ip() {
        // For unit-norm queries the ALSH transform's L2 argmin should agree with
        // the true MIPS winner on most queries; we require the top-1 to match.
        let n = 40;
        let dim = 8;
        let data = rand_data(n, dim, 5);
        let t = ShrivastavaLiTransform::fit(&data, n, dim, 3, 0.83)
            .expect("test invariant: should succeed");
        let q = rand_data(1, dim, 88);
        let qt = t
            .transform_query(&q)
            .expect("test invariant: should succeed");
        let mut best = (0usize, f32::INFINITY);
        for i in 0..n {
            let xt = t
                .transform_db(&data[i * dim..(i + 1) * dim])
                .expect("test invariant: should succeed");
            let d2: f32 = qt
                .iter()
                .zip(xt.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if d2 < best.1 {
                best = (i, d2);
            }
        }
        let exact = exact_mips(&data, n, dim, &q, 1);
        assert_eq!(best.0, exact[0], "ALSH top-1 ≠ exact MIPS");
    }

    #[test]
    fn alsh_err_u_out_of_range() {
        let data = rand_data(5, 4, 1);
        let err = ShrivastavaLiTransform::fit(&data, 5, 4, 2, 1.5).unwrap_err();
        assert!(matches!(err, AnnError::Internal { .. }));
    }

    #[test]
    fn alsh_err_zero_m() {
        let data = rand_data(5, 4, 1);
        let err = ShrivastavaLiTransform::fit(&data, 5, 4, 0, 0.8).unwrap_err();
        assert!(matches!(err, AnnError::InvalidLayerCount { .. }));
    }
}
