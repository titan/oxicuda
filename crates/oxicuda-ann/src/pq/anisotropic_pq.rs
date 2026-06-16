//! Anisotropic Product Quantization.
//!
//! Implements ScaNN-style score-aware quantisation from:
//! Guo et al., "Accelerating Large-Scale Inference with Anisotropic Vector Quantization",
//! ICML 2020.
//!
//! Standard PQ minimises `||x - x̂||²` uniformly in all directions.  For Maximum Inner
//! Product Search (MIPS) the error that matters is the component of the residual that is
//! *parallel* to the query direction.  The anisotropic loss down-weights the perpendicular
//! component:
//!
//!   L_aniso = r_∥² + η² · r_⊥²  =  ||r||² − (1−η²) · (r · q̂)²
//!
//! where r = x − x̂ is the reconstruction residual and q̂ is the unit-query direction.
use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;
use crate::pq::codebook::PqCodebook;

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Euclidean L2 distance squared between two same-length slices.
#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Dot product of two same-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a slice.
#[inline]
fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

// ─── k-means++ initialisation (subspace) ──────────────────────────────────────

fn kmeans_pp_init(data: &[f32], n: usize, d: usize, k: usize, rng: &mut LcgRng) -> Vec<f32> {
    let mut centers: Vec<f32> = Vec::with_capacity(k * d);
    let first = rng.next_u32() as usize % n;
    centers.extend_from_slice(&data[first * d..(first + 1) * d]);

    let mut min_dists = vec![f32::INFINITY; n];

    for _ in 1..k {
        let last = &centers[centers.len() - d..];
        let mut total = 0.0_f64;
        for (i, row) in data.chunks_exact(d).enumerate() {
            let dd = l2_sq(row, last);
            if dd < min_dists[i] {
                min_dists[i] = dd;
            }
            total += min_dists[i] as f64;
        }
        let threshold = rng.next_f32() as f64 * total;
        let mut cum = 0.0_f64;
        let mut chosen = n - 1;
        for (i, &dd) in min_dists.iter().enumerate() {
            cum += dd as f64;
            if cum >= threshold {
                chosen = i;
                break;
            }
        }
        centers.extend_from_slice(&data[chosen * d..(chosen + 1) * d]);
    }
    centers
}

/// Nearest centroid index (L2).
fn nearest_centroid(x: &[f32], centroids: &[f32], k: usize, d: usize) -> usize {
    let mut best = 0;
    let mut best_d = f32::INFINITY;
    for c in 0..k {
        let center = &centroids[c * d..(c + 1) * d];
        let dd = l2_sq(x, center);
        if dd < best_d {
            best_d = dd;
            best = c;
        }
    }
    best
}

/// Train a single subspace with anisotropic k-means.
///
/// The centroid update weights each member `x_i` by:
///   w_i = 1 + (1 − η²) · mean_{q̂ in Q_s} (q̂ · (x_i − c_j))²
///
/// This upweights points whose displacement from the centroid is large in query-relevant
/// directions, making the centroid gravitate towards the query-direction mean.
fn train_subspace_aniso(
    x_sub: &[f32], // [n × dsub]
    q_sub: &[f32], // [n_q × dsub]
    n: usize,
    n_q: usize,
    dsub: usize,
    k: usize,
    n_epochs: usize,
    eta: f32,
    rng: &mut LcgRng,
) -> Vec<f32> {
    let eta_sq = eta * eta;
    let aniso_coef = 1.0_f32 - eta_sq; // upweight factor for query-direction variance

    // Precompute normalised query sub-vectors  q̂_s = q_s / ||q_s||  (skip zero-norm queries)
    let mut q_hat: Vec<f32> = vec![0.0_f32; n_q * dsub];
    for qi in 0..n_q {
        let q = &q_sub[qi * dsub..(qi + 1) * dsub];
        let n_q_norm = norm(q);
        if n_q_norm > 1e-9 {
            let inv = 1.0 / n_q_norm;
            for d in 0..dsub {
                q_hat[qi * dsub + d] = q[d] * inv;
            }
        }
        // zero-norm queries contribute nothing to anisotropic weighting
    }

    // k-means++ init
    let mut centroids = kmeans_pp_init(x_sub, n, dsub, k, rng);
    let mut assignments = vec![0usize; n];

    for _epoch in 0..n_epochs {
        // ── Assignment step (standard L2) ──────────────────────────────────
        let mut changed = 0usize;
        for i in 0..n {
            let xi = &x_sub[i * dsub..(i + 1) * dsub];
            let c = nearest_centroid(xi, &centroids, k, dsub);
            if c != assignments[i] {
                changed += 1;
                assignments[i] = c;
            }
        }
        if _epoch > 0 && changed == 0 {
            break;
        }

        // ── Update step (anisotropic weighted centroid) ────────────────────
        let mut new_centroids = vec![0.0_f32; k * dsub];
        let mut weight_sum = vec![0.0_f32; k];

        for i in 0..n {
            let xi = &x_sub[i * dsub..(i + 1) * dsub];
            let ci = assignments[i];
            let centroid_ci = &centroids[ci * dsub..(ci + 1) * dsub];

            // Compute displacement from centroid
            let mut disp = vec![0.0_f32; dsub];
            for d in 0..dsub {
                disp[d] = xi[d] - centroid_ci[d];
            }

            // Anisotropic weight: 1 + (1-η²) * mean_q (q̂ · disp)²
            let mut query_var = 0.0_f32;
            if n_q > 0 {
                for qi in 0..n_q {
                    let qh = &q_hat[qi * dsub..(qi + 1) * dsub];
                    let proj = dot(qh, &disp);
                    query_var += proj * proj;
                }
                query_var /= n_q as f32;
            }
            let w = 1.0_f32 + aniso_coef * query_var;

            weight_sum[ci] += w;
            for d in 0..dsub {
                new_centroids[ci * dsub + d] += w * xi[d];
            }
        }

        // Normalise
        for c in 0..k {
            if weight_sum[c] > 0.0 {
                let inv = 1.0 / weight_sum[c];
                for d in 0..dsub {
                    centroids[c * dsub + d] = new_centroids[c * dsub + d] * inv;
                }
            } else {
                // Empty cluster: re-seed from a random point
                let ri = rng.next_u32() as usize % n;
                centroids[c * dsub..(c + 1) * dsub]
                    .copy_from_slice(&x_sub[ri * dsub..(ri + 1) * dsub]);
            }
        }
    }

    centroids
}

// ─── public API ──────────────────────────────────────────────────────────────

/// Weight for the perpendicular error component (η parameter).
///
/// - η = 0 → only parallel error matters (pure MIPS optimisation).
/// - η = 1 → isotropic (same as standard PQ loss).
/// - η ∈ (0, 1) → ScaNN-style intermediate weighting.
#[derive(Debug, Clone, Copy)]
pub struct AnisotropicWeight(pub f32);

impl AnisotropicWeight {
    /// Create a new weight; `eta` must be in `[0.0, 1.0]`.
    pub fn new(eta: f32) -> AnnResult<Self> {
        if !(0.0..=1.0).contains(&eta) {
            return Err(AnnError::Internal {
                msg: format!("AnisotropicWeight eta={eta} not in [0, 1]"),
            });
        }
        Ok(Self(eta))
    }

    /// η value.
    #[inline]
    pub fn eta(self) -> f32 {
        self.0
    }
}

/// Configuration for Anisotropic PQ.
#[derive(Debug, Clone)]
pub struct AnisotropicPqConfig {
    /// Number of PQ subspaces.
    pub m: usize,
    /// Number of codewords per subspace (≤ 256).
    pub ksub: usize,
    /// k-means epochs per subspace.
    pub n_epochs: usize,
    /// Perpendicular error weight η.
    pub weight: AnisotropicWeight,
}

impl AnisotropicPqConfig {
    /// Create configuration with explicit `eta ∈ [0, 1]`.
    pub fn new(m: usize, ksub: usize, eta: f32) -> AnnResult<Self> {
        if m == 0 {
            return Err(AnnError::InvalidNumSubspaces { m, dim: 0 });
        }
        if ksub == 0 || ksub > 256 {
            return Err(AnnError::InvalidK { k: ksub, n: 256 });
        }
        Ok(Self {
            m,
            ksub,
            n_epochs: 20,
            weight: AnisotropicWeight::new(eta)?,
        })
    }

    /// Isotropic configuration (η = 1.0, equivalent to standard PQ).
    pub fn isotropic(m: usize, ksub: usize) -> AnnResult<Self> {
        Self::new(m, ksub, 1.0)
    }
}

/// Anisotropic PQ trained model.
#[derive(Debug)]
pub struct AnisotropicPq {
    /// The trained codebook (each subspace optimised with anisotropic weighting).
    pub codebook: PqCodebook,
    /// Configuration used for training.
    pub config: AnisotropicPqConfig,
}

impl AnisotropicPq {
    /// Train anisotropic PQ on `n` data vectors using `n_queries` representative queries.
    pub fn train(
        data: &[f32], // [n × dim]
        n: usize,
        dim: usize,
        queries: &[f32], // [n_queries × dim]
        n_queries: usize,
        cfg: AnisotropicPqConfig,
        rng: &mut LcgRng,
    ) -> AnnResult<Self> {
        // ── validation ────────────────────────────────────────────────────────
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        if n_queries == 0 {
            return Err(AnnError::EmptyInput);
        }
        if dim == 0 || !dim.is_multiple_of(cfg.m) {
            return Err(AnnError::InvalidNumSubspaces { m: cfg.m, dim });
        }
        if data.len() != n * dim {
            return Err(AnnError::DimensionMismatch {
                expected: n * dim,
                got: data.len(),
            });
        }
        if queries.len() != n_queries * dim {
            return Err(AnnError::DimensionMismatch {
                expected: n_queries * dim,
                got: queries.len(),
            });
        }
        if cfg.ksub == 0 || cfg.ksub > n {
            return Err(AnnError::InvalidK { k: cfg.ksub, n });
        }

        let dsub = dim / cfg.m;
        let mut cb = PqCodebook::new(cfg.m, cfg.ksub, dsub);

        // Temporary buffers for subspace extraction
        let mut x_sub = vec![0.0_f32; n * dsub];
        let mut q_sub = vec![0.0_f32; n_queries * dsub];

        for s in 0..cfg.m {
            // Extract subspace s from data
            for i in 0..n {
                for d in 0..dsub {
                    x_sub[i * dsub + d] = data[i * dim + s * dsub + d];
                }
            }
            // Extract subspace s from queries
            for qi in 0..n_queries {
                for d in 0..dsub {
                    q_sub[qi * dsub + d] = queries[qi * dim + s * dsub + d];
                }
            }

            let centroids = train_subspace_aniso(
                &x_sub,
                &q_sub,
                n,
                n_queries,
                dsub,
                cfg.ksub,
                cfg.n_epochs,
                cfg.weight.eta(),
                rng,
            );

            // Store centroids in codebook
            for c in 0..cfg.ksub {
                let dst = cb.centroid_mut(s, c);
                dst.copy_from_slice(&centroids[c * dsub..(c + 1) * dsub]);
            }
        }

        Ok(AnisotropicPq {
            codebook: cb,
            config: cfg,
        })
    }

    /// Encode `n` vectors using standard PQ assignment (codebook trained anisotropically).
    pub fn encode(&self, data: &[f32], n: usize) -> AnnResult<Vec<u8>> {
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        let dim = self.codebook.m * self.codebook.dsub;
        if data.len() != n * dim {
            return Err(AnnError::DimensionMismatch {
                expected: n * dim,
                got: data.len(),
            });
        }
        let mut codes = Vec::with_capacity(n * self.codebook.m);
        for i in 0..n {
            let v = &data[i * dim..(i + 1) * dim];
            for s in 0..self.codebook.m {
                let sub = &v[s * self.codebook.dsub..(s + 1) * self.codebook.dsub];
                let c = nearest_centroid(
                    sub,
                    self.codebook.centroids_raw(),
                    self.codebook.ksub,
                    self.codebook.dsub,
                );
                // nearest_centroid searches all k centroids in the flat buffer starting at 0,
                // but we need to scope it to subspace s.
                // Redo with correct offset:
                let _ = c; // discard, recalculate below
                let sub_centroids =
                    &self.codebook.centroids_raw()[s * self.codebook.ksub * self.codebook.dsub
                        ..(s + 1) * self.codebook.ksub * self.codebook.dsub];
                let best =
                    nearest_centroid(sub, sub_centroids, self.codebook.ksub, self.codebook.dsub);
                codes.push(best as u8);
            }
        }
        Ok(codes)
    }

    /// Decode codes → reconstructed vectors.
    pub fn decode(&self, codes: &[u8], n: usize) -> AnnResult<Vec<f32>> {
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        let m = self.codebook.m;
        let dsub = self.codebook.dsub;
        let dim = m * dsub;
        if codes.len() != n * m {
            return Err(AnnError::DimensionMismatch {
                expected: n * m,
                got: codes.len(),
            });
        }
        let mut out = vec![0.0_f32; n * dim];
        for i in 0..n {
            for s in 0..m {
                let c = codes[i * m + s] as usize;
                let centroid = self.codebook.centroid(s, c);
                let base = i * dim + s * dsub;
                out[base..base + dsub].copy_from_slice(centroid);
            }
        }
        Ok(out)
    }

    /// Compute anisotropic quantisation loss over `n` data vectors and `n_queries` queries.
    ///
    /// L_aniso = (1/n) Σ_i \[ ||r_i||² − (1−η²) · (r_i · q̂_i)² \]
    ///
    /// where r_i = x_i − x̂_i and q̂_i = queries[i % n_queries] / ||queries[i % n_queries]||.
    pub fn anisotropic_loss(
        &self,
        data: &[f32],
        n: usize,
        queries: &[f32],
        n_queries: usize,
    ) -> AnnResult<f32> {
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        if n_queries == 0 {
            return Err(AnnError::EmptyInput);
        }
        let dim = self.codebook.m * self.codebook.dsub;
        if data.len() != n * dim {
            return Err(AnnError::DimensionMismatch {
                expected: n * dim,
                got: data.len(),
            });
        }
        if queries.len() != n_queries * dim {
            return Err(AnnError::DimensionMismatch {
                expected: n_queries * dim,
                got: queries.len(),
            });
        }

        let codes = self.encode(data, n)?;
        let recon = self.decode(&codes, n)?;

        let eta_sq = self.config.weight.eta() * self.config.weight.eta();
        let aniso_coef = 1.0_f32 - eta_sq;

        let mut total_loss = 0.0_f32;
        for i in 0..n {
            let xi = &data[i * dim..(i + 1) * dim];
            let xhati = &recon[i * dim..(i + 1) * dim];
            let qi = &queries[(i % n_queries) * dim..((i % n_queries) + 1) * dim];

            // Residual r = x - x̂
            let mut r = vec![0.0_f32; dim];
            for d in 0..dim {
                r[d] = xi[d] - xhati[d];
            }
            let r_sq: f32 = dot(&r, &r);

            // Unit query direction
            let q_norm = norm(qi);
            let parallel_sq = if q_norm > 1e-9 {
                let proj = dot(&r, qi) / q_norm;
                proj * proj
            } else {
                0.0_f32
            };

            total_loss += r_sq - aniso_coef * parallel_sq;
        }

        Ok(total_loss / n as f32)
    }

    /// Compute isotropic L2 reconstruction loss (for comparison with standard PQ).
    ///
    /// L_iso = (1/n) Σ_i ||x_i − x̂_i||²
    pub fn isotropic_loss(&self, data: &[f32], n: usize) -> AnnResult<f32> {
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        let dim = self.codebook.m * self.codebook.dsub;
        if data.len() != n * dim {
            return Err(AnnError::DimensionMismatch {
                expected: n * dim,
                got: data.len(),
            });
        }
        let codes = self.encode(data, n)?;
        let recon = self.decode(&codes, n)?;
        let total: f32 = data
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        Ok(total / n as f32)
    }

    /// Compare anisotropic vs isotropic loss ratio.
    ///
    /// Returns `anisotropic_loss / isotropic_loss`.  For η < 1 this should be ≤ 1.
    pub fn loss_ratio(
        &self,
        data: &[f32],
        n: usize,
        queries: &[f32],
        n_queries: usize,
    ) -> AnnResult<f32> {
        let aniso = self.anisotropic_loss(data, n, queries, n_queries)?;
        let iso = self.isotropic_loss(data, n)?;
        if iso.abs() < 1e-30 {
            // Both losses are essentially zero; ratio is 1 by convention
            return Ok(1.0_f32);
        }
        Ok(aniso / iso)
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng(seed: u64) -> LcgRng {
        LcgRng::new(seed)
    }

    fn normal_vecs(n: usize, dim: usize, rng: &mut LcgRng) -> Vec<f32> {
        let mut v = vec![0.0_f32; n * dim];
        rng.fill_normal(&mut v);
        v
    }

    fn rand_vecs(n: usize, dim: usize, rng: &mut LcgRng) -> Vec<f32> {
        (0..n * dim).map(|_| rng.next_f32()).collect()
    }

    // ── AnisotropicWeight ─────────────────────────────────────────────────────

    #[test]
    fn anisotropic_weight_new_valid() {
        assert!(AnisotropicWeight::new(0.0).is_ok());
        assert!(AnisotropicWeight::new(0.5).is_ok());
        assert!(AnisotropicWeight::new(1.0).is_ok());
    }

    #[test]
    fn anisotropic_weight_invalid_negative() {
        let res = AnisotropicWeight::new(-0.01);
        assert!(res.is_err(), "negative eta should fail");
    }

    #[test]
    fn anisotropic_weight_invalid_too_large() {
        let res = AnisotropicWeight::new(1.01);
        assert!(res.is_err(), "eta > 1 should fail");
    }

    // ── AnisotropicPqConfig ───────────────────────────────────────────────────

    #[test]
    fn anisotropic_config_isotropic() {
        let cfg = AnisotropicPqConfig::isotropic(2, 8).expect("isotropic config is valid");
        assert!((cfg.weight.eta() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn anisotropic_config_new_valid() {
        let cfg = AnisotropicPqConfig::new(4, 16, 0.5);
        assert!(cfg.is_ok(), "{:?}", cfg.err());
    }

    // ── training ──────────────────────────────────────────────────────────────

    #[test]
    fn anisotropic_train_basic() {
        let mut rng = make_rng(1);
        let n = 100;
        let dim = 8;
        let n_q = 10;
        let data = normal_vecs(n, dim, &mut rng);
        let queries = normal_vecs(n_q, dim, &mut rng);
        let mut cfg = AnisotropicPqConfig::new(2, 8, 0.5).expect("config parameters are valid");
        cfg.n_epochs = 5;
        let model = AnisotropicPq::train(&data, n, dim, &queries, n_q, cfg, &mut rng);
        assert!(model.is_ok(), "{:?}", model.err());
        let model = model.expect("training should succeed");
        assert_eq!(model.codebook.m, 2);
        assert_eq!(model.codebook.ksub, 8);
        assert_eq!(model.codebook.dsub, 4);
    }

    #[test]
    fn anisotropic_n_zero_error() {
        let mut rng = make_rng(2);
        let cfg = AnisotropicPqConfig::new(2, 8, 0.5).expect("config parameters are valid");
        let res = AnisotropicPq::train(&[], 0, 8, &[1.0], 1, cfg, &mut rng);
        assert!(matches!(res, Err(AnnError::EmptyInput)));
    }

    #[test]
    fn anisotropic_dim_not_divisible_error() {
        let mut rng = make_rng(3);
        let cfg = AnisotropicPqConfig::new(3, 4, 0.5).expect("config parameters are valid");
        let data = vec![0.5_f32; 50 * 8]; // dim=8, m=3 → 8 % 3 ≠ 0
        let queries = vec![0.5_f32; 5 * 8];
        let res = AnisotropicPq::train(&data, 50, 8, &queries, 5, cfg, &mut rng);
        assert!(res.is_err(), "should fail because 8 % 3 != 0");
    }

    #[test]
    fn anisotropic_wrong_query_count_error() {
        let mut rng = make_rng(4);
        let cfg = AnisotropicPqConfig::new(2, 4, 0.5).expect("config parameters are valid");
        let data = vec![0.5_f32; 50 * 8];
        // n_queries = 0
        let res = AnisotropicPq::train(&data, 50, 8, &[], 0, cfg, &mut rng);
        assert!(matches!(res, Err(AnnError::EmptyInput)));
    }

    // ── encode / decode ───────────────────────────────────────────────────────

    #[test]
    fn anisotropic_encode_shape() {
        let mut rng = make_rng(5);
        let n = 80;
        let dim = 8;
        let n_q = 10;
        let data = normal_vecs(n, dim, &mut rng);
        let queries = normal_vecs(n_q, dim, &mut rng);
        let mut cfg = AnisotropicPqConfig::new(2, 8, 0.5).expect("config parameters are valid");
        cfg.n_epochs = 5;
        let model = AnisotropicPq::train(&data, n, dim, &queries, n_q, cfg, &mut rng)
            .expect("training should succeed");
        let codes = model.encode(&data, n).expect("encode should succeed");
        assert_eq!(codes.len(), n * model.codebook.m);
    }

    #[test]
    fn anisotropic_decode_shape() {
        let mut rng = make_rng(6);
        let n = 80;
        let dim = 8;
        let n_q = 10;
        let data = normal_vecs(n, dim, &mut rng);
        let queries = normal_vecs(n_q, dim, &mut rng);
        let mut cfg = AnisotropicPqConfig::new(2, 8, 0.5).expect("config parameters are valid");
        cfg.n_epochs = 5;
        let model = AnisotropicPq::train(&data, n, dim, &queries, n_q, cfg, &mut rng)
            .expect("training should succeed");
        let codes = model.encode(&data, n).expect("encode should succeed");
        let decoded = model.decode(&codes, n).expect("decode should succeed");
        assert_eq!(decoded.len(), n * dim);
    }

    #[test]
    fn anisotropic_encode_decode_close() {
        let mut rng = make_rng(7);
        let n = 128;
        let dim = 4;
        let n_q = 16;
        let data = normal_vecs(n, dim, &mut rng);
        let queries = normal_vecs(n_q, dim, &mut rng);
        let mut cfg = AnisotropicPqConfig::new(2, 4, 0.5).expect("config parameters are valid");
        cfg.n_epochs = 10;
        let model = AnisotropicPq::train(&data, n, dim, &queries, n_q, cfg, &mut rng)
            .expect("training should succeed");
        let codes = model.encode(&data, n).expect("encode should succeed");
        let decoded = model.decode(&codes, n).expect("decode should succeed");
        let mse: f32 = data
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / (n * dim) as f32;
        assert!(mse.is_finite(), "MSE is not finite");
        assert!(mse < 10.0, "MSE too large: {mse}");
    }

    #[test]
    fn anisotropic_encode_wrong_data_len() {
        let mut rng = make_rng(8);
        let n = 64;
        let dim = 8;
        let n_q = 8;
        let data = normal_vecs(n, dim, &mut rng);
        let queries = normal_vecs(n_q, dim, &mut rng);
        let mut cfg = AnisotropicPqConfig::new(2, 8, 0.5).expect("config parameters are valid");
        cfg.n_epochs = 5;
        let model = AnisotropicPq::train(&data, n, dim, &queries, n_q, cfg, &mut rng)
            .expect("training should succeed");
        // Wrong data length for encode
        let bad = vec![0.0_f32; 3 * dim - 1];
        let res = model.encode(&bad, 3);
        assert!(res.is_err(), "wrong data length should fail");
    }

    // ── loss functions ────────────────────────────────────────────────────────

    #[test]
    fn anisotropic_isotropic_loss_nonneg() {
        let mut rng = make_rng(9);
        let n = 100;
        let dim = 8;
        let n_q = 10;
        let data = normal_vecs(n, dim, &mut rng);
        let queries = normal_vecs(n_q, dim, &mut rng);
        let mut cfg = AnisotropicPqConfig::new(2, 8, 0.5).expect("config parameters are valid");
        cfg.n_epochs = 5;
        let model = AnisotropicPq::train(&data, n, dim, &queries, n_q, cfg, &mut rng)
            .expect("training should succeed");
        let loss = model
            .isotropic_loss(&data, n)
            .expect("isotropic loss should succeed");
        assert!(loss >= 0.0, "isotropic loss={loss} should be ≥ 0");
        assert!(loss.is_finite(), "loss should be finite");
    }

    #[test]
    fn anisotropic_anisotropic_loss_nonneg() {
        let mut rng = make_rng(10);
        let n = 100;
        let dim = 8;
        let n_q = 10;
        let data = normal_vecs(n, dim, &mut rng);
        let queries = normal_vecs(n_q, dim, &mut rng);
        let mut cfg = AnisotropicPqConfig::new(2, 8, 0.5).expect("config parameters are valid");
        cfg.n_epochs = 5;
        let model = AnisotropicPq::train(&data, n, dim, &queries, n_q, cfg, &mut rng)
            .expect("training should succeed");
        let loss = model
            .anisotropic_loss(&data, n, &queries, n_q)
            .expect("anisotropic loss should succeed");
        assert!(loss >= 0.0, "anisotropic loss={loss} should be ≥ 0");
        assert!(loss.is_finite(), "loss should be finite");
    }

    #[test]
    fn anisotropic_loss_ratio_finite() {
        let mut rng = make_rng(11);
        let n = 100;
        let dim = 8;
        let n_q = 10;
        let data = normal_vecs(n, dim, &mut rng);
        let queries = normal_vecs(n_q, dim, &mut rng);
        let mut cfg = AnisotropicPqConfig::new(2, 8, 0.5).expect("config parameters are valid");
        cfg.n_epochs = 5;
        let model = AnisotropicPq::train(&data, n, dim, &queries, n_q, cfg, &mut rng)
            .expect("training should succeed");
        let ratio = model
            .loss_ratio(&data, n, &queries, n_q)
            .expect("loss ratio should succeed");
        assert!(ratio.is_finite(), "ratio={ratio} should be finite");
    }

    #[test]
    fn anisotropic_loss_ratio_not_negative() {
        let mut rng = make_rng(12);
        let n = 100;
        let dim = 8;
        let n_q = 10;
        let data = normal_vecs(n, dim, &mut rng);
        let queries = normal_vecs(n_q, dim, &mut rng);
        let mut cfg = AnisotropicPqConfig::new(2, 8, 0.5).expect("config parameters are valid");
        cfg.n_epochs = 5;
        let model = AnisotropicPq::train(&data, n, dim, &queries, n_q, cfg, &mut rng)
            .expect("training should succeed");
        let ratio = model
            .loss_ratio(&data, n, &queries, n_q)
            .expect("loss ratio should succeed");
        assert!(ratio >= 0.0, "loss ratio={ratio} should be ≥ 0");
    }

    #[test]
    fn anisotropic_eta1_matches_isotropic_loss() {
        // For η=1, anisotropic loss = isotropic loss (the (1-η²) term vanishes)
        let mut rng = make_rng(13);
        let n = 100;
        let dim = 8;
        let n_q = 10;
        let data = normal_vecs(n, dim, &mut rng);
        let queries = normal_vecs(n_q, dim, &mut rng);
        let mut cfg = AnisotropicPqConfig::isotropic(2, 8).expect("isotropic config is valid");
        cfg.n_epochs = 10;
        let model = AnisotropicPq::train(&data, n, dim, &queries, n_q, cfg, &mut rng)
            .expect("training should succeed");
        let iso = model
            .isotropic_loss(&data, n)
            .expect("isotropic loss should succeed");
        let aniso = model
            .anisotropic_loss(&data, n, &queries, n_q)
            .expect("anisotropic loss should succeed");
        assert!(
            (iso - aniso).abs() < 1e-5,
            "η=1: iso={iso:.6} ≠ aniso={aniso:.6}"
        );
    }

    #[test]
    fn anisotropic_eta0_parallel_dominant() {
        // For η=0, anisotropic loss should be ≤ isotropic loss because we subtract
        // (1-η²)·(r·q̂)² = (r·q̂)² ≥ 0 from the isotropic loss.
        let mut rng = make_rng(14);
        let n = 100;
        let dim = 8;
        let n_q = 10;
        let data = normal_vecs(n, dim, &mut rng);
        let queries = normal_vecs(n_q, dim, &mut rng);
        let mut cfg = AnisotropicPqConfig::new(2, 8, 0.0).expect("config parameters are valid");
        cfg.n_epochs = 10;
        let model = AnisotropicPq::train(&data, n, dim, &queries, n_q, cfg, &mut rng)
            .expect("training should succeed");
        let iso = model
            .isotropic_loss(&data, n)
            .expect("isotropic loss should succeed");
        let aniso = model
            .anisotropic_loss(&data, n, &queries, n_q)
            .expect("anisotropic loss should succeed");
        // aniso ≤ iso + tiny float tolerance
        assert!(
            aniso <= iso + 1e-5,
            "η=0: aniso={aniso:.6} should be ≤ iso={iso:.6}"
        );
    }

    // ── determinism ───────────────────────────────────────────────────────────

    #[test]
    fn anisotropic_deterministic() {
        let n = 80;
        let dim = 8;
        let n_q = 10;
        let mut rng0 = make_rng(99);
        let data = normal_vecs(n, dim, &mut rng0);
        let queries = normal_vecs(n_q, dim, &mut rng0);

        let mut rng_a = make_rng(42);
        let mut rng_b = make_rng(42);

        let mut cfg_a = AnisotropicPqConfig::new(2, 8, 0.5).expect("config parameters are valid");
        cfg_a.n_epochs = 5;
        let cfg_b = cfg_a.clone();

        let model_a = AnisotropicPq::train(&data, n, dim, &queries, n_q, cfg_a, &mut rng_a)
            .expect("training with cfg_a should succeed");
        let model_b = AnisotropicPq::train(&data, n, dim, &queries, n_q, cfg_b, &mut rng_b)
            .expect("training with cfg_b should succeed");

        // Same seed → identical codebook
        assert_eq!(
            model_a.codebook.centroids_raw(),
            model_b.codebook.centroids_raw(),
            "same seed should produce identical codebook"
        );
    }

    // ── rand_vecs variant ─────────────────────────────────────────────────────

    #[test]
    fn anisotropic_train_uniform_data() {
        let mut rng = make_rng(20);
        let n = 64;
        let dim = 4;
        let n_q = 8;
        let data = rand_vecs(n, dim, &mut rng);
        let queries = rand_vecs(n_q, dim, &mut rng);
        let mut cfg = AnisotropicPqConfig::new(2, 4, 0.3).expect("config parameters are valid");
        cfg.n_epochs = 5;
        let res = AnisotropicPq::train(&data, n, dim, &queries, n_q, cfg, &mut rng);
        assert!(res.is_ok(), "{:?}", res.err());
    }
}
