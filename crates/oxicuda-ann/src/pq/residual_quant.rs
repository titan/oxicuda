//! Residual Quantization (RQ) — multi-stage Vector Quantization.
//!
//! References:
//! - Chen, Y. et al., "Approximate Nearest Neighbor Search by Residual Vector
//!   Quantization", Sensors 2010.
//! - Liu, S. et al., "A Survey of Quantization Methods for Vector Search", 2019.
//!
//! Algorithm:
//! - Stage 0: train k-means with K centroids on the input samples X.
//! - For each subsequent stage m ∈ [1, M):
//!   * Compute the residual: r_i^{(m)} = r_i^{(m-1)} − c_{a_i^{(m-1)}}.
//!   * Train k-means with K centroids on the new residuals.
//! - The reconstruction of x_i is the sum of the M chosen centroids,
//!   x̂_i = Σ_m c_{m, a_i^{(m)}}.
//!
//! Unlike additive quantization (AQ) the encoding is **stage-wise greedy**:
//! at each stage we pick the nearest centroid in that stage's codebook to
//! the current residual.  This is suboptimal w.r.t. joint reconstruction
//! error but cheap to compute and gives the canonical monotone-refinement
//! guarantee: adding a stage cannot increase the reconstruction MSE.
use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;
use crate::kmeans::kmeans::KMeans;

/// Configuration for Residual Quantization training.
#[derive(Debug, Clone, Copy)]
pub struct RqConfig {
    /// Number of residual stages (M).  Must be ≥ 1.
    pub n_stages: usize,
    /// Number of centroids per stage (K).  Must be ≥ 1 and ≤ 65536 so codes
    /// fit in `u16` (we still store them as `u32` here for kernel friendliness).
    pub n_centroids_per_stage: usize,
    /// Dimension of each sample vector.  Must be ≥ 1.
    pub dim: usize,
    /// k-means epochs per stage.
    pub n_iter_kmeans: usize,
}

impl RqConfig {
    fn validate(&self) -> AnnResult<()> {
        if self.n_stages == 0 {
            return Err(AnnError::InvalidLayerCount { n: 0 });
        }
        if self.dim == 0 {
            return Err(AnnError::InvalidVectorDim { dim: 0 });
        }
        if self.n_centroids_per_stage == 0 {
            return Err(AnnError::InvalidK {
                k: 0,
                n: usize::MAX,
            });
        }
        if self.n_centroids_per_stage > 65_536 {
            return Err(AnnError::Internal {
                msg: format!(
                    "rq: n_centroids_per_stage = {} exceeds u16 capacity 65536",
                    self.n_centroids_per_stage
                ),
            });
        }
        Ok(())
    }
}

/// M stage codebooks; each stage stores `K * dim` row-major centroids.
#[derive(Debug, Clone)]
pub struct RqCodebooks {
    /// `stages[m]` is `K * dim` row-major centroids for stage `m`.
    pub stages: Vec<Vec<f32>>,
    pub dim: usize,
    pub n_centroids_per_stage: usize,
}

impl RqCodebooks {
    /// Number of stages (M).
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.stages.len()
    }

    fn centroid(&self, stage: usize, idx: usize) -> AnnResult<&[f32]> {
        let cb = self.stages.get(stage).ok_or(AnnError::Internal {
            msg: format!("rq: stage {stage} out of range"),
        })?;
        let k = self.n_centroids_per_stage;
        if idx >= k {
            return Err(AnnError::IdOutOfRange { id: idx, n: k });
        }
        let off = idx * self.dim;
        let end = off + self.dim;
        cb.get(off..end).ok_or(AnnError::Internal {
            msg: format!("rq: centroid slice [{off},{end}) out of range for stage {stage}"),
        })
    }
}

/// Per-sample per-stage centroid index codes.
#[derive(Debug, Clone)]
pub struct RqCodes {
    /// `codes[m]` is a length-N vector of centroid indices for stage `m`.
    pub codes: Vec<Vec<u32>>,
    pub n_samples: usize,
}

// ─── helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Find the nearest centroid of an arbitrary K-centroid codebook (row-major K×dim).
fn nearest_centroid(x: &[f32], codebook: &[f32], k: usize, dim: usize) -> AnnResult<usize> {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for c in 0..k {
        let off = c * dim;
        let end = off + dim;
        let centroid = codebook.get(off..end).ok_or(AnnError::Internal {
            msg: format!("rq: codebook slice [{off},{end}) out of range"),
        })?;
        let d = l2_sq(x, centroid);
        if d < best_d {
            best_d = d;
            best = c;
        }
    }
    Ok(best)
}

// ─── training ────────────────────────────────────────────────────────────────

/// Train M-stage residual quantizer codebooks on `samples`.
///
/// `samples` is `n_samples × dim` row-major; its length must be a positive
/// multiple of `cfg.dim`.
pub fn train(samples: &[f32], cfg: RqConfig, rng: &mut LcgRng) -> AnnResult<RqCodebooks> {
    cfg.validate()?;
    if samples.is_empty() {
        return Err(AnnError::EmptyInput);
    }
    if !samples.len().is_multiple_of(cfg.dim) {
        return Err(AnnError::DimensionMismatch {
            expected: cfg.dim,
            got: samples.len() % cfg.dim,
        });
    }
    let n_samples = samples.len() / cfg.dim;
    if cfg.n_centroids_per_stage > n_samples {
        return Err(AnnError::InvalidK {
            k: cfg.n_centroids_per_stage,
            n: n_samples,
        });
    }

    // working residual buffer (mutated stage-by-stage).
    let mut residual: Vec<f32> = samples.to_vec();
    let mut stages: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_stages);

    for _stage in 0..cfg.n_stages {
        // (a) k-means on current residual
        let km = KMeans::fit(
            &residual,
            n_samples,
            cfg.dim,
            cfg.n_centroids_per_stage,
            cfg.n_iter_kmeans,
            rng,
        )?;
        let centroids = km.centroids().to_vec();

        // (b) hard assign each residual to its nearest centroid (greedy)
        // (c) residual_i -= centroid[assignment_i]
        for i in 0..n_samples {
            let row_off = i * cfg.dim;
            let row_end = row_off + cfg.dim;
            let assignment = {
                let row = residual.get(row_off..row_end).ok_or(AnnError::Internal {
                    msg: format!("rq: residual row [{row_off},{row_end}) out of range"),
                })?;
                nearest_centroid(row, &centroids, cfg.n_centroids_per_stage, cfg.dim)?
            };
            let cen_off = assignment * cfg.dim;
            let cen_end = cen_off + cfg.dim;
            for d in 0..cfg.dim {
                let r = residual.get_mut(row_off + d).ok_or(AnnError::Internal {
                    msg: format!("rq: residual write {} out of range", row_off + d),
                })?;
                let cv = centroids
                    .get(cen_off + d)
                    .copied()
                    .ok_or(AnnError::Internal {
                        msg: format!(
                            "rq: centroid read {} out of range (slice [{cen_off},{cen_end}))",
                            cen_off + d
                        ),
                    })?;
                *r -= cv;
            }
        }

        stages.push(centroids);
    }

    Ok(RqCodebooks {
        stages,
        dim: cfg.dim,
        n_centroids_per_stage: cfg.n_centroids_per_stage,
    })
}

// ─── encoding / decoding ─────────────────────────────────────────────────────

/// Stage-wise greedy encode of `samples` against pre-trained codebooks.
///
/// At each stage we look up the nearest centroid to the *current residual*
/// (not the original sample); this is the canonical RQ encoding.
pub fn encode(samples: &[f32], codebooks: &RqCodebooks) -> AnnResult<RqCodes> {
    let dim = codebooks.dim;
    if dim == 0 {
        return Err(AnnError::InvalidVectorDim { dim: 0 });
    }
    if codebooks.stages.is_empty() {
        return Err(AnnError::InvalidLayerCount { n: 0 });
    }
    if samples.is_empty() {
        return Err(AnnError::EmptyInput);
    }
    if !samples.len().is_multiple_of(dim) {
        return Err(AnnError::DimensionMismatch {
            expected: dim,
            got: samples.len() % dim,
        });
    }
    let n_samples = samples.len() / dim;
    let k = codebooks.n_centroids_per_stage;
    let m = codebooks.stages.len();

    let mut residual: Vec<f32> = samples.to_vec();
    let mut codes: Vec<Vec<u32>> = vec![vec![0u32; n_samples]; m];

    for (stage_idx, stage_cb) in codebooks.stages.iter().enumerate() {
        for i in 0..n_samples {
            let row_off = i * dim;
            let row_end = row_off + dim;
            let assignment = {
                let row = residual.get(row_off..row_end).ok_or(AnnError::Internal {
                    msg: format!("rq: residual row [{row_off},{row_end}) out of range"),
                })?;
                nearest_centroid(row, stage_cb, k, dim)?
            };
            let cen_off = assignment * dim;
            for d in 0..dim {
                let r = residual.get_mut(row_off + d).ok_or(AnnError::Internal {
                    msg: format!("rq: residual write {} out of range", row_off + d),
                })?;
                let cv = stage_cb
                    .get(cen_off + d)
                    .copied()
                    .ok_or(AnnError::Internal {
                        msg: format!("rq: stage centroid read {} out of range", cen_off + d),
                    })?;
                *r -= cv;
            }
            let slot =
                codes
                    .get_mut(stage_idx)
                    .and_then(|c| c.get_mut(i))
                    .ok_or(AnnError::Internal {
                        msg: format!("rq: codes[{stage_idx}][{i}] out of range"),
                    })?;
            *slot = assignment as u32;
        }
    }

    Ok(RqCodes { codes, n_samples })
}

/// Decode codes into row-major `n_samples × dim` reconstructions
/// `x̂_i = Σ_m codebooks.stages[m][codes.codes[m][i]]`.
pub fn decode(codes: &RqCodes, codebooks: &RqCodebooks) -> AnnResult<Vec<f32>> {
    let m = codebooks.stages.len();
    if m == 0 {
        return Err(AnnError::InvalidLayerCount { n: 0 });
    }
    if codes.codes.len() != m {
        return Err(AnnError::DimensionMismatch {
            expected: m,
            got: codes.codes.len(),
        });
    }
    let dim = codebooks.dim;
    if dim == 0 {
        return Err(AnnError::InvalidVectorDim { dim: 0 });
    }
    let n = codes.n_samples;
    for (s, row) in codes.codes.iter().enumerate() {
        if row.len() != n {
            return Err(AnnError::DimensionMismatch {
                expected: n,
                got: row.len(),
            });
        }
        let _ = s;
    }

    let mut out = vec![0.0_f32; n * dim];
    for (stage_idx, code_row) in codes.codes.iter().enumerate() {
        for (i, &assignment) in code_row.iter().enumerate() {
            let cen = codebooks.centroid(stage_idx, assignment as usize)?;
            let row_off = i * dim;
            for d in 0..dim {
                let slot = out.get_mut(row_off + d).ok_or(AnnError::Internal {
                    msg: format!("rq: decode write {} out of range", row_off + d),
                })?;
                let cv = cen.get(d).copied().ok_or(AnnError::Internal {
                    msg: format!("rq: decode centroid index {d} out of range"),
                })?;
                *slot += cv;
            }
        }
    }
    Ok(out)
}

/// Mean squared reconstruction error per scalar entry over the entire
/// `n × dim` flat array.
pub fn reconstruction_error(samples: &[f32], decoded: &[f32]) -> AnnResult<f32> {
    if samples.is_empty() {
        return Err(AnnError::EmptyInput);
    }
    if samples.len() != decoded.len() {
        return Err(AnnError::DimensionMismatch {
            expected: samples.len(),
            got: decoded.len(),
        });
    }
    let mut acc = 0.0_f64;
    for (a, b) in samples.iter().zip(decoded.iter()) {
        let d = (a - b) as f64;
        acc += d * d;
    }
    Ok((acc / samples.len() as f64) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn gauss_samples(n: usize, dim: usize, rng: &mut LcgRng) -> Vec<f32> {
        let mut v = vec![0.0_f32; n * dim];
        rng.fill_normal(&mut v);
        v
    }

    fn default_cfg(dim: usize, m: usize, k: usize) -> RqConfig {
        RqConfig {
            n_stages: m,
            n_centroids_per_stage: k,
            dim,
            n_iter_kmeans: 20,
        }
    }

    // 1. M=1 → equivalent to plain k-means: stages[0] is the k-means fit on the
    //    samples themselves.  We verify each centroid equals the mean of its
    //    assigned samples.
    #[test]
    fn rq_m1_reduces_to_kmeans() {
        let mut rng = LcgRng::new(13);
        let n = 40;
        let dim = 4;
        let k = 5;
        let samples = gauss_samples(n, dim, &mut rng);

        let cfg = default_cfg(dim, 1, k);
        let mut rng2 = LcgRng::new(13);
        let cb = train(&samples, cfg, &mut rng2).expect("training should succeed");
        assert_eq!(cb.stages.len(), 1);

        // re-derive assignments
        let codes = encode(&samples, &cb).expect("encode should succeed");
        let assigns = &codes.codes[0];

        // each centroid = mean of its assigned samples
        let stage0 = &cb.stages[0];
        for c in 0..k {
            let members: Vec<usize> = (0..n).filter(|&i| assigns[i] as usize == c).collect();
            if members.is_empty() {
                continue;
            }
            for d in 0..dim {
                let mean: f32 = members.iter().map(|&i| samples[i * dim + d]).sum::<f32>()
                    / members.len() as f32;
                let cdv = stage0[c * dim + d];
                assert!((cdv - mean).abs() < 1e-4, "centroid drift {cdv} vs {mean}");
            }
        }
    }

    // 2. M=2 MSE ≤ M=1 MSE on the same data and seed (load-bearing monotone refinement).
    #[test]
    fn rq_monotone_refinement_m1_vs_m2() {
        let mut rng = LcgRng::new(7);
        let n = 80;
        let dim = 4;
        let k = 8;
        let samples = gauss_samples(n, dim, &mut rng);

        let cfg1 = default_cfg(dim, 1, k);
        let cfg2 = default_cfg(dim, 2, k);

        let mut rng1 = LcgRng::new(99);
        let cb1 = train(&samples, cfg1, &mut rng1).expect("training should succeed");
        let codes1 = encode(&samples, &cb1).expect("encode should succeed");
        let dec1 = decode(&codes1, &cb1).expect("decode should succeed");
        let mse1 =
            reconstruction_error(&samples, &dec1).expect("reconstruction error should succeed");

        let mut rng2 = LcgRng::new(99);
        let cb2 = train(&samples, cfg2, &mut rng2).expect("training should succeed");
        let codes2 = encode(&samples, &cb2).expect("encode should succeed");
        let dec2 = decode(&codes2, &cb2).expect("decode should succeed");
        let mse2 =
            reconstruction_error(&samples, &dec2).expect("reconstruction error should succeed");

        assert!(
            mse2 <= mse1 + 1e-6,
            "rq monotone refinement violated: m1={mse1} m2={mse2}"
        );
    }

    // 3. decode(encode(x)).len == samples.len
    #[test]
    fn rq_decode_encode_shape() {
        let mut rng = LcgRng::new(1);
        let n = 16;
        let dim = 3;
        let k = 4;
        let samples = gauss_samples(n, dim, &mut rng);
        let cfg = default_cfg(dim, 2, k);
        let cb = train(&samples, cfg, &mut rng).expect("training should succeed");
        let codes = encode(&samples, &cb).expect("encode should succeed");
        let dec = decode(&codes, &cb).expect("decode should succeed");
        assert_eq!(dec.len(), samples.len());
    }

    // 4. deterministic with same seed
    #[test]
    fn rq_deterministic() {
        let mut rng_src = LcgRng::new(101);
        let n = 24;
        let dim = 4;
        let samples = gauss_samples(n, dim, &mut rng_src);
        let cfg = default_cfg(dim, 3, 4);

        let mut a = LcgRng::new(33);
        let cba = train(&samples, cfg, &mut a).expect("training should succeed");
        let codes_a = encode(&samples, &cba).expect("encode should succeed");

        let mut b = LcgRng::new(33);
        let cbb = train(&samples, cfg, &mut b).expect("training should succeed");
        let codes_b = encode(&samples, &cbb).expect("encode should succeed");

        for s in 0..cba.stages.len() {
            for (ai, bi) in cba.stages[s].iter().zip(cbb.stages[s].iter()) {
                assert!((ai - bi).abs() < 1e-7);
            }
            assert_eq!(codes_a.codes[s], codes_b.codes[s]);
        }
    }

    // 5. K=1 → assignments all zero and stage 0 centroid is the global mean of samples
    #[test]
    fn rq_k1_global_mean() {
        let mut rng = LcgRng::new(2);
        let n = 12;
        let dim = 5;
        let samples = gauss_samples(n, dim, &mut rng);
        let cfg = default_cfg(dim, 1, 1);
        let mut rng2 = LcgRng::new(2);
        let cb = train(&samples, cfg, &mut rng2).expect("training should succeed");
        let codes = encode(&samples, &cb).expect("encode should succeed");
        for &c in &codes.codes[0] {
            assert_eq!(c, 0);
        }
        for d in 0..dim {
            let mean: f32 = (0..n).map(|i| samples[i * dim + d]).sum::<f32>() / n as f32;
            let c = cb.stages[0][d];
            assert!((c - mean).abs() < 1e-4, "global mean drift d={d}");
        }
    }

    // 6. all codes ∈ [0, K)
    #[test]
    fn rq_codes_within_k() {
        let mut rng = LcgRng::new(4);
        let n = 30;
        let dim = 4;
        let k = 7;
        let samples = gauss_samples(n, dim, &mut rng);
        let cfg = default_cfg(dim, 3, k);
        let cb = train(&samples, cfg, &mut rng).expect("training should succeed");
        let codes = encode(&samples, &cb).expect("encode should succeed");
        for stage_row in &codes.codes {
            assert_eq!(stage_row.len(), n);
            for &c in stage_row {
                assert!((c as usize) < k);
            }
        }
    }

    // 7. each stage's codebook centroid is the mean of its assigned residuals
    //    (within the same kmeans tolerance).
    #[test]
    fn rq_centroids_are_residual_means() {
        let mut rng = LcgRng::new(5);
        let n = 36;
        let dim = 3;
        let k = 4;
        let samples = gauss_samples(n, dim, &mut rng);
        let cfg = default_cfg(dim, 2, k);
        let mut rng2 = LcgRng::new(5);
        let cb = train(&samples, cfg, &mut rng2).expect("training should succeed");
        let codes = encode(&samples, &cb).expect("encode should succeed");

        // reconstruct what residual fed each stage by greedily subtracting prior centroids
        let mut residual = samples.clone();
        for (stage, code_row) in codes.codes.iter().enumerate() {
            // verify centroid c = mean of residual rows i with code_row[i] == c
            let stage_cb = &cb.stages[stage];
            for c in 0..k {
                let members: Vec<usize> = (0..n).filter(|&i| code_row[i] as usize == c).collect();
                if members.is_empty() {
                    continue;
                }
                for d in 0..dim {
                    let mean: f32 = members.iter().map(|&i| residual[i * dim + d]).sum::<f32>()
                        / members.len() as f32;
                    let cdv = stage_cb[c * dim + d];
                    assert!(
                        (cdv - mean).abs() < 5e-4,
                        "stage {stage} centroid drift {cdv} vs {mean}"
                    );
                }
            }
            // subtract assigned centroid to prepare residual for next stage
            for i in 0..n {
                let cen = code_row[i] as usize;
                for d in 0..dim {
                    residual[i * dim + d] -= stage_cb[cen * dim + d];
                }
            }
        }
    }

    // 8. M ≥ 1 always
    #[test]
    fn rq_err_m_zero() {
        let mut rng = LcgRng::new(0);
        let samples = vec![0.0_f32; 10];
        let cfg = default_cfg(2, 0, 2);
        let err = train(&samples, cfg, &mut rng);
        assert!(matches!(err, Err(AnnError::InvalidLayerCount { n: 0 })));
    }

    // 9. reconstruction MSE drops as M grows (4 stages on a small dataset).
    #[test]
    fn rq_mse_decreases_with_more_stages() {
        let mut rng_src = LcgRng::new(123);
        let n = 32;
        let dim = 4;
        let samples = gauss_samples(n, dim, &mut rng_src);
        let mut prev = f32::INFINITY;
        for m in 1..=4 {
            let cfg = default_cfg(dim, m, 8);
            let mut rng = LcgRng::new(123);
            let cb = train(&samples, cfg, &mut rng).expect("training should succeed");
            let codes = encode(&samples, &cb).expect("encode should succeed");
            let dec = decode(&codes, &cb).expect("decode should succeed");
            let mse =
                reconstruction_error(&samples, &dec).expect("reconstruction error should succeed");
            assert!(
                mse <= prev + 1e-5,
                "m={m} mse={mse} prev={prev} should be non-increasing"
            );
            prev = mse;
        }
    }

    // 10. err K=0
    #[test]
    fn rq_err_k_zero() {
        let mut rng = LcgRng::new(0);
        let samples = vec![0.0_f32; 10];
        let cfg = default_cfg(2, 1, 0);
        assert!(matches!(
            train(&samples, cfg, &mut rng),
            Err(AnnError::InvalidK { .. })
        ));
    }

    // 11. err dim=0
    #[test]
    fn rq_err_dim_zero() {
        let mut rng = LcgRng::new(0);
        let samples = vec![0.0_f32; 0];
        let cfg = default_cfg(0, 1, 2);
        assert!(matches!(
            train(&samples, cfg, &mut rng),
            Err(AnnError::InvalidVectorDim { dim: 0 })
        ));
    }

    // 12. err samples.len() not a multiple of dim
    #[test]
    fn rq_err_samples_not_multiple_of_dim() {
        let mut rng = LcgRng::new(0);
        let samples = vec![0.0_f32; 7];
        let cfg = default_cfg(2, 1, 2);
        let err = train(&samples, cfg, &mut rng);
        assert!(matches!(err, Err(AnnError::DimensionMismatch { .. })));
    }

    // 13. err empty samples
    #[test]
    fn rq_err_empty_samples() {
        let mut rng = LcgRng::new(0);
        let samples: Vec<f32> = Vec::new();
        let cfg = default_cfg(2, 1, 2);
        let err = train(&samples, cfg, &mut rng);
        assert!(matches!(err, Err(AnnError::EmptyInput)));
    }

    // 14. err decode with wrong-shape codes (M mismatch)
    #[test]
    fn rq_err_decode_codes_shape_mismatch() {
        let mut rng = LcgRng::new(0);
        let n = 8;
        let dim = 2;
        let samples = gauss_samples(n, dim, &mut rng);
        let cfg = default_cfg(dim, 3, 2);
        let cb = train(&samples, cfg, &mut rng).expect("training should succeed");
        let bogus = RqCodes {
            codes: vec![vec![0u32; n], vec![0u32; n]], // 2 stages but cb has 3
            n_samples: n,
        };
        let err = decode(&bogus, &cb);
        assert!(matches!(err, Err(AnnError::DimensionMismatch { .. })));
    }

    // 15. err decode with mismatched n_samples row length
    #[test]
    fn rq_err_decode_row_length_mismatch() {
        let mut rng = LcgRng::new(0);
        let n = 8;
        let dim = 2;
        let samples = gauss_samples(n, dim, &mut rng);
        let cfg = default_cfg(dim, 2, 2);
        let cb = train(&samples, cfg, &mut rng).expect("training should succeed");
        let bogus = RqCodes {
            codes: vec![vec![0u32; n], vec![0u32; n - 1]],
            n_samples: n,
        };
        let err = decode(&bogus, &cb);
        assert!(matches!(err, Err(AnnError::DimensionMismatch { .. })));
    }

    // 16. err reconstruction_error: input length mismatch
    #[test]
    fn rq_err_reconstruction_error_mismatch() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![1.0_f32, 2.0];
        let err = reconstruction_error(&a, &b);
        assert!(matches!(err, Err(AnnError::DimensionMismatch { .. })));
    }

    // 17. err reconstruction_error: empty
    #[test]
    fn rq_err_reconstruction_error_empty() {
        let a: Vec<f32> = Vec::new();
        let b: Vec<f32> = Vec::new();
        let err = reconstruction_error(&a, &b);
        assert!(matches!(err, Err(AnnError::EmptyInput)));
    }

    // 18. encoding round trip: decode(encode(X)) MSE ≤ a small bound when K is
    //     large enough to identify each point individually.
    #[test]
    fn rq_encode_decode_small_dataset_quality() {
        let mut rng = LcgRng::new(8);
        let n = 8;
        let dim = 3;
        let samples = gauss_samples(n, dim, &mut rng);
        // K=8, M=2 → 64 effective code combinations; trivially expressive.
        let cfg = default_cfg(dim, 2, 8);
        let cb = train(&samples, cfg, &mut rng).expect("training should succeed");
        let codes = encode(&samples, &cb).expect("encode should succeed");
        let dec = decode(&codes, &cb).expect("decode should succeed");
        let mse =
            reconstruction_error(&samples, &dec).expect("reconstruction error should succeed");
        // We can't assert exact zero (greedy stage encoding) but should be small.
        assert!(mse < 1.0, "mse={mse}");
    }
}
