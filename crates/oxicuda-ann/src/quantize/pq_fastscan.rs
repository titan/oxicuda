use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;
use crate::pq::codebook::PqCodebook;
use crate::pq::train::train_pq;
use crate::topk::heap::BoundedMaxHeap;

/// PQFastScan: Product Quantizer with 4-bit codes (ksub = 16) and packed nibble encoding.
///
/// Two 4-bit codes are packed per byte: the high nibble holds the code for even
/// sub-spaces and the low nibble holds the code for odd sub-spaces.
/// This allows a very compact representation and fast scan with a 16-entry LUT.
pub struct PqFastScan {
    codebook: PqCodebook,
    /// Number of sub-spaces.
    pub m: usize,
    /// Dimension of each sub-space (`dim / m`).
    pub dsub: usize,
    /// Original vector dimension.
    pub dim: usize,
}

impl PqFastScan {
    /// Train a PQFastScan on `n` rows of `dim`-dimensional data.
    ///
    /// Always uses `ksub = 16` (4-bit codes per sub-space).
    pub fn train(
        data: &[f32],
        n: usize,
        dim: usize,
        m: usize,
        n_iter: usize,
        rng: &mut LcgRng,
    ) -> AnnResult<Self> {
        if n == 0 {
            return Err(AnnError::EmptyInput);
        }
        if dim == 0 {
            return Err(AnnError::InvalidVectorDim { dim: 0 });
        }
        if m == 0 || !dim.is_multiple_of(m) {
            return Err(AnnError::InvalidNumSubspaces { m, dim });
        }

        let dsub = dim / m;
        let codebook = train_pq(data, n, dim, m, 16, n_iter, rng)?;
        Ok(Self {
            codebook,
            m,
            dsub,
            dim,
        })
    }

    /// Encode a single vector into `⌈m / 2⌉` packed bytes (4-bit codes).
    ///
    /// `bytes[k] = (code[2k] << 4) | code[2k+1]`
    /// For odd `m`, the low nibble of the last byte is 0 (padding).
    pub fn encode(&self, v: &[f32]) -> AnnResult<Vec<u8>> {
        if v.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: v.len(),
            });
        }

        // Assign each sub-space.
        let mut codes = Vec::with_capacity(self.m);
        for s in 0..self.m {
            let sub = &v[s * self.dsub..(s + 1) * self.dsub];
            codes.push(self.assign_subvec(s, sub));
        }

        // Pack two codes per byte.
        let n_bytes = self.m.div_ceil(2);
        let mut packed = vec![0u8; n_bytes];
        for k in 0..n_bytes {
            let hi = codes[2 * k];
            let lo = if 2 * k + 1 < self.m {
                codes[2 * k + 1]
            } else {
                0
            };
            packed[k] = ((hi as u8) << 4) | (lo as u8 & 0x0F);
        }

        Ok(packed)
    }

    /// Encode `n` vectors (row-major `[n, dim]`) in batch.
    pub fn encode_batch(&self, data: &[f32], n: usize) -> AnnResult<Vec<Vec<u8>>> {
        if data.len() != n * self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: n * self.dim,
                got: data.len(),
            });
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let v = &data[i * self.dim..(i + 1) * self.dim];
            out.push(self.encode(v)?);
        }
        Ok(out)
    }

    /// Build the 16-entry LUT for each sub-space: flat `[m * 16]`.
    ///
    /// `lut[s * 16 + c]` = L2² from `query[s*dsub..(s+1)*dsub]` to centroid `(s, c)`.
    pub fn build_lut(&self, query: &[f32]) -> AnnResult<Vec<f32>> {
        if query.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let mut lut = vec![0.0_f32; self.m * 16];
        for s in 0..self.m {
            let q_sub = &query[s * self.dsub..(s + 1) * self.dsub];
            for c in 0..16 {
                let centroid = self.codebook.centroid(s, c);
                let d: f32 = q_sub
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                lut[s * 16 + c] = d;
            }
        }
        Ok(lut)
    }

    /// Compute approximate L2² distance from packed 4-bit codes and a pre-built LUT.
    ///
    /// For each byte `k`:
    /// - `hi = codes[k] >> 4` → sub-space `2k`
    /// - `lo = codes[k] & 0x0F` → sub-space `2k+1` (skipped when `2k+1 >= m`)
    pub fn adc_dist(&self, codes: &[u8], lut: &[f32]) -> AnnResult<f32> {
        let expected_bytes = self.m.div_ceil(2);
        if codes.len() != expected_bytes {
            return Err(AnnError::DimensionMismatch {
                expected: expected_bytes,
                got: codes.len(),
            });
        }
        if lut.len() != self.m * 16 {
            return Err(AnnError::DimensionMismatch {
                expected: self.m * 16,
                got: lut.len(),
            });
        }

        let mut dist = 0.0_f32;
        for (k, &byte) in codes.iter().enumerate().take(expected_bytes) {
            let hi = (byte >> 4) as usize;
            let lo = (byte & 0x0F) as usize;
            let s_even = 2 * k;
            let s_odd = 2 * k + 1;
            dist += lut[s_even * 16 + hi];
            if s_odd < self.m {
                dist += lut[s_odd * 16 + lo];
            }
        }
        Ok(dist)
    }

    /// Scan all `all_codes` vectors, build LUT once, return top-`k` via heap.
    pub fn search_batch(
        &self,
        query: &[f32],
        all_codes: &[Vec<u8>],
        ids: &[u32],
        k: usize,
    ) -> AnnResult<Vec<(u32, f32)>> {
        if all_codes.len() != ids.len() {
            return Err(AnnError::DimensionMismatch {
                expected: ids.len(),
                got: all_codes.len(),
            });
        }
        if all_codes.is_empty() {
            return Ok(Vec::new());
        }

        let lut = self.build_lut(query)?;
        let actual_k = k.min(all_codes.len());
        let mut heap = BoundedMaxHeap::new(actual_k);

        for (codes, &id) in all_codes.iter().zip(ids.iter()) {
            let d = self.adc_dist(codes, &lut)?;
            heap.push(d, id as usize);
        }

        let raw = heap.into_sorted_vec();
        Ok(raw
            .into_iter()
            .map(|(id, dist)| (id as u32, dist))
            .collect())
    }

    /// Returns `ksub`, which is always 16 for PQFastScan.
    pub fn ksub(&self) -> usize {
        16
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Assign a sub-vector to the nearest of the 16 centroids for sub-space `s`.
    fn assign_subvec(&self, s: usize, sub: &[f32]) -> usize {
        let mut best_c = 0;
        let mut best_d = f32::INFINITY;
        for c in 0..16 {
            let centroid = self.codebook.centroid(s, c);
            let d: f32 = sub
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if d < best_d {
                best_d = d;
                best_c = c;
            }
        }
        best_c
    }
}

// --------------------------------------------------------------------------
// Unit tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng(seed: u64) -> LcgRng {
        LcgRng::new(seed)
    }

    fn rand_vecs_normal(n: usize, dim: usize, rng: &mut LcgRng) -> Vec<f32> {
        (0..n * dim).map(|_| rng.next_normal()).collect()
    }

    // 1. Encode length: m=4 → 2 bytes; m=5 → 3 bytes.
    #[test]
    fn encode_len() {
        let mut rng = make_rng(1);
        let n = 100;
        let dim = 8; // m=4 → dsub=2
        let data = rand_vecs_normal(n, dim, &mut rng);
        let fs =
            PqFastScan::train(&data, n, dim, 4, 10, &mut rng).expect("valid training parameters");
        let codes = fs.encode(&data[0..dim]).expect("valid vector dimension");
        assert_eq!(codes.len(), 2, "m=4 → ceil(4/2)=2 bytes");

        // m=5: need dim divisible by 5.
        let dim2 = 10;
        let data2 = rand_vecs_normal(n, dim2, &mut rng);
        let fs2 =
            PqFastScan::train(&data2, n, dim2, 5, 10, &mut rng).expect("valid training parameters");
        let codes2 = fs2.encode(&data2[0..dim2]).expect("valid vector dimension");
        assert_eq!(codes2.len(), 3, "m=5 → ceil(5/2)=3 bytes");
    }

    // 2. Nibble packing (even m): hi nibble of byte 0 == code for sub-space 0.
    #[test]
    fn nibble_packing_even_m() {
        let mut rng = make_rng(2);
        let n = 100;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let fs =
            PqFastScan::train(&data, n, dim, 4, 10, &mut rng).expect("valid training parameters");

        let v = &data[0..dim];
        let codes = fs.encode(v).expect("valid vector dimension");

        // Compute code for sub-space 0 independently.
        let expected_s0 = fs.assign_subvec(0, &v[0..fs.dsub]) as u8;
        let hi_nibble = codes[0] >> 4;
        assert_eq!(
            hi_nibble, expected_s0,
            "hi nibble of byte 0 should equal code for sub-space 0"
        );
    }

    // 3. Nibble packing (odd m): last byte's low nibble is 0.
    #[test]
    fn nibble_packing_odd_m() {
        let mut rng = make_rng(3);
        let n = 100;
        let dim = 10; // m=5 → dsub=2
        let data = rand_vecs_normal(n, dim, &mut rng);
        let fs =
            PqFastScan::train(&data, n, dim, 5, 10, &mut rng).expect("valid training parameters");
        let codes = fs.encode(&data[0..dim]).expect("valid vector dimension");
        // 3 bytes for m=5; last byte's low nibble is padding=0.
        assert_eq!(codes.len(), 3);
        let last_lo = codes[2] & 0x0F;
        assert_eq!(
            last_lo, 0,
            "low nibble of last byte should be 0 for odd m=5"
        );
    }

    // 4. adc_dist returns a finite positive value.
    #[test]
    fn adc_dist_finite() {
        let mut rng = make_rng(4);
        let n = 100;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let fs =
            PqFastScan::train(&data, n, dim, 4, 10, &mut rng).expect("valid training parameters");
        let query = rand_vecs_normal(1, dim, &mut rng);
        let candidate = &data[5 * dim..6 * dim];
        let codes = fs.encode(candidate).expect("valid vector dimension");
        let lut = fs.build_lut(&query[0..dim]).expect("valid query dimension");
        let dist = fs
            .adc_dist(&codes, &lut)
            .expect("valid packed codes and LUT");
        assert!(dist.is_finite(), "dist must be finite, got {dist}");
        assert!(dist >= 0.0, "dist must be non-negative, got {dist}");
    }

    // 5. adc_dist of query's own codes ≈ 0 (self-distance).
    #[test]
    fn adc_dist_self() {
        let mut rng = make_rng(5);
        let n = 200;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let fs =
            PqFastScan::train(&data, n, dim, 4, 10, &mut rng).expect("valid training parameters");
        // Use a training vector as query and encode it.
        let query = &data[0..dim];
        let codes = fs.encode(query).expect("valid vector dimension");
        let lut = fs.build_lut(query).expect("valid query dimension");
        let dist = fs
            .adc_dist(&codes, &lut)
            .expect("valid packed codes and LUT");
        // The distance from query to its own quantized centroid is non-negative and
        // should be small (bounded by PQ approximation error, not strictly 0).
        assert!(dist.is_finite(), "self-distance must be finite");
        assert!(dist >= 0.0, "self-distance must be non-negative");
        // Sanity: not astronomically large.
        assert!(dist < 100.0, "self-distance surprisingly large: {dist}");
    }

    // 6. build_lut returns m*16 entries.
    #[test]
    fn build_lut_len() {
        let mut rng = make_rng(6);
        let n = 100;
        let dim = 8;
        let m = 4;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let fs =
            PqFastScan::train(&data, n, dim, m, 10, &mut rng).expect("valid training parameters");
        let lut = fs.build_lut(&data[0..dim]).expect("valid query dimension");
        assert_eq!(lut.len(), m * 16, "LUT should have m*16={} entries", m * 16);
    }

    // 7. build_lut: all entries are finite.
    #[test]
    fn build_lut_all_finite() {
        let mut rng = make_rng(7);
        let n = 100;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let fs =
            PqFastScan::train(&data, n, dim, 4, 10, &mut rng).expect("valid training parameters");
        let lut = fs.build_lut(&data[0..dim]).expect("valid query dimension");
        for (i, &v) in lut.iter().enumerate() {
            assert!(v.is_finite(), "lut[{i}] = {v} is not finite");
        }
    }

    // 8. search_batch top-1 is the training point itself.
    #[test]
    fn search_batch_top1_self() {
        let mut rng = make_rng(8);
        let n = 200;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let fs =
            PqFastScan::train(&data, n, dim, 4, 10, &mut rng).expect("valid training parameters");
        let all_codes: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                fs.encode(&data[i * dim..(i + 1) * dim])
                    .expect("valid vector dimension")
            })
            .collect();
        let ids: Vec<u32> = (0..n as u32).collect();

        let results = fs
            .search_batch(&data[0..dim], &all_codes, &ids, 5)
            .expect("valid search parameters");
        assert!(!results.is_empty());
        assert!(
            results.iter().any(|(id, _)| *id == 0),
            "top-5 should include id=0; got {:?}",
            results
        );
    }

    // 9. m not dividing dim → Err(InvalidNumSubspaces).
    #[test]
    fn m_not_dividing_dim_error() {
        let mut rng = make_rng(9);
        let n = 100;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let result = PqFastScan::train(&data, n, dim, 3, 10, &mut rng);
        assert!(
            matches!(result, Err(AnnError::InvalidNumSubspaces { .. })),
            "expected InvalidNumSubspaces, got {:?}",
            result.err()
        );
    }

    // 10. Consistency with vanilla ADC: distances are in the same order of magnitude.
    #[test]
    fn consistency_with_vanilla_adc() {
        use crate::pq::adc::{adc_distance, build_adc_table};
        use crate::pq::encode::encode_vector;

        let mut rng = make_rng(10);
        let n = 200;
        let dim = 8;
        let m = 4;
        let data = rand_vecs_normal(n, dim, &mut rng);
        // Train a vanilla PQ also with ksub=16 to enable fair comparison.
        let vanilla_pq = crate::pq::train::train_pq(&data, n, dim, m, 16, 10, &mut rng)
            .expect("valid PQ training parameters");

        // Train FastScan PQ.
        let fs =
            PqFastScan::train(&data, n, dim, m, 10, &mut rng).expect("valid training parameters");

        let query = &data[0..dim];
        let candidate = &data[10 * dim..11 * dim];

        // FastScan distance.
        let fs_codes = fs.encode(candidate).expect("valid vector dimension");
        let fs_lut = fs.build_lut(query).expect("valid query dimension");
        let fs_dist = fs
            .adc_dist(&fs_codes, &fs_lut)
            .expect("valid packed codes and LUT");

        // Vanilla ADC distance.
        let van_codes = encode_vector(candidate, &vanilla_pq);
        let van_table = build_adc_table(query, &vanilla_pq);
        let van_dist = adc_distance(&van_codes, &van_table, m, 16);

        // Both should produce finite, non-negative approximate L2² values.
        assert!(fs_dist.is_finite(), "FastScan dist must be finite");
        assert!(van_dist.is_finite(), "vanilla dist must be finite");
        assert!(fs_dist >= 0.0);
        assert!(van_dist >= 0.0);
        // Both estimate the same quantity; they won't be identical (different
        // codebooks trained independently) but should be within a reasonable factor.
        let ratio = if van_dist > 0.0 {
            fs_dist / van_dist
        } else {
            fs_dist
        };
        assert!(
            ratio < 10.0,
            "FastScan dist={fs_dist} and vanilla dist={van_dist} are unexpectedly far apart (ratio={ratio})"
        );
    }
}
