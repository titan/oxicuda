use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;
use crate::kmeans::kmeans::KMeans;

/// Binary Product Quantizer: one bit per sub-space (k=2 centroids each).
///
/// Each sub-space dimension is projected to {0, 1} by assigning the sub-vector
/// to the nearer of two k-means centroids.  The resulting bits are packed into
/// `⌈m / 64⌉` `u64` words, enabling fast Hamming distance via XOR + popcount.
pub struct BinaryPq {
    /// Flat layout `[m * 2 * dsub]`: centroid of sub-space `s`, label `c` is at
    /// `centroids[(s * 2 + c) * dsub .. (s * 2 + c + 1) * dsub]`.
    centroids: Vec<f32>,
    /// Number of sub-spaces.
    pub m: usize,
    /// Dimension of each sub-space (`dim / m`).
    pub dsub: usize,
    /// Original vector dimension.
    pub dim: usize,
}

impl BinaryPq {
    /// Train a BinaryPq on `n` rows of `dim`-dimensional data.
    ///
    /// For each of the `m` sub-spaces, k-means with `k = 2` is run
    /// independently on the sub-space slice of the training data.
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
        // Need at least 2 points to form 2 centroids.
        if n < 2 {
            return Err(AnnError::InvalidK { k: 2, n });
        }

        let dsub = dim / m;
        let mut centroids = vec![0.0_f32; m * 2 * dsub];

        let mut sub_data = vec![0.0_f32; n * dsub];
        for s in 0..m {
            // Extract sub-space s.
            for i in 0..n {
                let src = &data[i * dim + s * dsub..i * dim + (s + 1) * dsub];
                sub_data[i * dsub..(i + 1) * dsub].copy_from_slice(src);
            }
            let km = KMeans::fit(&sub_data, n, dsub, 2, n_iter, rng)?;
            let km_cents = km.centroids();
            // Store centroid 0 and centroid 1.
            for c in 0..2 {
                let dst = &mut centroids[(s * 2 + c) * dsub..(s * 2 + c + 1) * dsub];
                dst.copy_from_slice(&km_cents[c * dsub..(c + 1) * dsub]);
            }
        }

        Ok(Self {
            centroids,
            m,
            dsub,
            dim,
        })
    }

    /// Encode a single vector into `⌈m / 64⌉` packed `u64` words.
    ///
    /// Bit `s` (counted from the LSB of word `s / 64`) is `1` if sub-space `s`
    /// is assigned to centroid 1, and `0` if assigned to centroid 0.
    pub fn encode(&self, v: &[f32]) -> AnnResult<Vec<u64>> {
        if v.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: v.len(),
            });
        }

        let n_words = self.m.div_ceil(64);
        let mut words = vec![0u64; n_words];

        for s in 0..self.m {
            let sub = &v[s * self.dsub..(s + 1) * self.dsub];
            let c = self.assign_subvec(s, sub);
            if c == 1 {
                let word_idx = s / 64;
                let bit_idx = s % 64;
                words[word_idx] |= 1u64 << bit_idx;
            }
        }

        Ok(words)
    }

    /// Encode `n` vectors (row-major `[n, dim]`) in batch.
    pub fn encode_batch(&self, data: &[f32], n: usize) -> AnnResult<Vec<Vec<u64>>> {
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

    /// Hamming distance between two encoded vectors (number of differing bits).
    pub fn hamming(&self, a: &[u64], b: &[u64]) -> AnnResult<u32> {
        let expected_len = self.m.div_ceil(64);
        if a.len() != expected_len {
            return Err(AnnError::DimensionMismatch {
                expected: expected_len,
                got: a.len(),
            });
        }
        if b.len() != expected_len {
            return Err(AnnError::DimensionMismatch {
                expected: expected_len,
                got: b.len(),
            });
        }
        let dist = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum();
        Ok(dist)
    }

    /// Reconstruct an approximate vector from packed bit codes.
    ///
    /// For each sub-space, returns the coordinates of the centroid indicated by
    /// the corresponding bit.
    pub fn reconstruct(&self, bits: &[u64]) -> AnnResult<Vec<f32>> {
        let expected_len = self.m.div_ceil(64);
        if bits.len() != expected_len {
            return Err(AnnError::DimensionMismatch {
                expected: expected_len,
                got: bits.len(),
            });
        }
        let mut out = vec![0.0_f32; self.dim];
        for s in 0..self.m {
            let word_idx = s / 64;
            let bit_idx = s % 64;
            let c = ((bits[word_idx] >> bit_idx) & 1) as usize;
            let cent = &self.centroids[(s * 2 + c) * self.dsub..(s * 2 + c + 1) * self.dsub];
            out[s * self.dsub..(s + 1) * self.dsub].copy_from_slice(cent);
        }
        Ok(out)
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Assign sub-vector `sub` in sub-space `s` to centroid 0 or 1.
    fn assign_subvec(&self, s: usize, sub: &[f32]) -> usize {
        let c0 = &self.centroids[s * 2 * self.dsub..(s * 2 + 1) * self.dsub];
        let c1 = &self.centroids[(s * 2 + 1) * self.dsub..(s * 2 + 2) * self.dsub];
        let d0: f32 = sub
            .iter()
            .zip(c0.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let d1: f32 = sub
            .iter()
            .zip(c1.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        if d0 <= d1 { 0 } else { 1 }
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

    // 1. Encode output length: m=64 → 1 u64, m=65 → 2 u64s.
    #[test]
    fn encode_output_len() {
        let mut rng = make_rng(1);
        let n = 128;
        let dim = 128; // m=64 ⇒ dsub=2
        let data = rand_vecs_normal(n, dim, &mut rng);
        let bpq =
            BinaryPq::train(&data, n, dim, 64, 10, &mut rng).expect("valid training parameters");
        let bits = bpq.encode(&data[0..dim]).expect("valid vector dimension");
        assert_eq!(bits.len(), 1, "m=64 should fit in 1 u64");

        // m=65: use dim=130.
        let dim2 = 130;
        let data2 = rand_vecs_normal(n, dim2, &mut rng);
        let bpq2 =
            BinaryPq::train(&data2, n, dim2, 65, 10, &mut rng).expect("valid training parameters");
        let bits2 = bpq2
            .encode(&data2[0..dim2])
            .expect("valid vector dimension");
        assert_eq!(bits2.len(), 2, "m=65 should require 2 u64s");
    }

    // 2. Hamming self = 0.
    #[test]
    fn hamming_self_zero() {
        let mut rng = make_rng(2);
        let n = 100;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let bpq =
            BinaryPq::train(&data, n, dim, 4, 10, &mut rng).expect("valid training parameters");
        let bits = bpq.encode(&data[0..dim]).expect("valid vector dimension");
        assert_eq!(
            bpq.hamming(&bits, &bits).expect("valid code word lengths"),
            0
        );
    }

    // 3. Hamming of all-flipped codes equals m.
    #[test]
    fn hamming_all_flipped_is_m() {
        let mut rng = make_rng(3);
        let m = 8;
        let dim = 8;
        let n = 200;
        // Build two perfectly separated clusters per sub-space.
        let zeros_block: Vec<f32> = vec![0.0_f32; dim * 100];
        let ones_block: Vec<f32> = vec![10.0_f32; dim * 100];
        let mut data = zeros_block.clone();
        data.extend_from_slice(&ones_block);
        let bpq =
            BinaryPq::train(&data, n, dim, m, 20, &mut rng).expect("valid training parameters");
        let bits_zero = bpq
            .encode(&zeros_block[0..dim])
            .expect("valid vector dimension");
        let bits_one = bpq
            .encode(&ones_block[0..dim])
            .expect("valid vector dimension");
        let dist = bpq
            .hamming(&bits_zero, &bits_one)
            .expect("valid code word lengths");
        assert_eq!(
            dist, m as u32,
            "perfectly opposite cluster assignments should give hamming == m"
        );
    }

    // 4. Hamming bounded by m.
    #[test]
    fn hamming_bounded_by_m() {
        let mut rng = make_rng(4);
        let n = 100;
        let dim = 8;
        let m = 4;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let bpq =
            BinaryPq::train(&data, n, dim, m, 10, &mut rng).expect("valid training parameters");
        for i in 0..10 {
            for j in 0..10 {
                let bi = bpq
                    .encode(&data[i * dim..(i + 1) * dim])
                    .expect("valid vector dimension");
                let bj = bpq
                    .encode(&data[j * dim..(j + 1) * dim])
                    .expect("valid vector dimension");
                let h = bpq.hamming(&bi, &bj).expect("valid code word lengths");
                assert!(h <= m as u32, "hamming={h} must be ≤ m={m}");
            }
        }
    }

    // 5. Train separates data — encode(zeros) ≠ encode(ones) in Hamming.
    #[test]
    fn train_separates_data() {
        let mut rng = make_rng(5);
        let n = 200;
        let dim = 8;
        let m = 4;
        let zeros: Vec<f32> = vec![0.0_f32; dim * 100];
        let ones: Vec<f32> = vec![10.0_f32; dim * 100];
        let mut data = zeros.clone();
        data.extend_from_slice(&ones);
        let bpq =
            BinaryPq::train(&data, n, dim, m, 20, &mut rng).expect("valid training parameters");
        let bz = bpq.encode(&zeros[0..dim]).expect("valid vector dimension");
        let bo = bpq.encode(&ones[0..dim]).expect("valid vector dimension");
        let h = bpq.hamming(&bz, &bo).expect("valid code word lengths");
        assert!(h > 0, "encode(zeros) should differ from encode(ones)");
    }

    // 6. Reconstruct shape — returns Vec<f32> of len dim.
    #[test]
    fn reconstruct_shape() {
        let mut rng = make_rng(6);
        let n = 100;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let bpq =
            BinaryPq::train(&data, n, dim, 4, 10, &mut rng).expect("valid training parameters");
        let bits = bpq.encode(&data[0..dim]).expect("valid vector dimension");
        let rec = bpq
            .reconstruct(&bits)
            .expect("valid packed bit code length");
        assert_eq!(rec.len(), dim, "reconstructed vector should have dim={dim}");
    }

    // 7. Reconstructed value is exactly one of the two centroids per sub-space.
    #[test]
    fn reconstruct_is_centroid() {
        let mut rng = make_rng(7);
        let n = 100;
        let dim = 8;
        let m = 4;
        let dsub = dim / m;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let bpq =
            BinaryPq::train(&data, n, dim, m, 10, &mut rng).expect("valid training parameters");
        let bits = bpq.encode(&data[0..dim]).expect("valid vector dimension");
        let rec = bpq
            .reconstruct(&bits)
            .expect("valid packed bit code length");

        // For each sub-space, the reconstructed slice must equal centroid 0 or centroid 1.
        for s in 0..m {
            let rec_sub = &rec[s * dsub..(s + 1) * dsub];
            let c0 = &bpq.centroids[s * 2 * dsub..(s * 2 + 1) * dsub];
            let c1 = &bpq.centroids[(s * 2 + 1) * dsub..(s * 2 + 2) * dsub];
            let is_c0 = rec_sub
                .iter()
                .zip(c0.iter())
                .all(|(a, b)| (a - b).abs() < 1e-6);
            let is_c1 = rec_sub
                .iter()
                .zip(c1.iter())
                .all(|(a, b)| (a - b).abs() < 1e-6);
            assert!(
                is_c0 || is_c1,
                "sub-space {s}: reconstructed slice is neither centroid 0 nor centroid 1"
            );
        }
    }

    // 8. m not dividing dim → Err(InvalidNumSubspaces).
    #[test]
    fn m_not_dividing_dim_error() {
        let mut rng = make_rng(8);
        let n = 100;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let result = BinaryPq::train(&data, n, dim, 3, 10, &mut rng);
        assert!(
            matches!(result, Err(AnnError::InvalidNumSubspaces { .. })),
            "expected InvalidNumSubspaces, got {:?}",
            result.err()
        );
    }

    // 9. encode_batch shape.
    #[test]
    fn encode_batch_shape() {
        let mut rng = make_rng(9);
        let n = 20;
        let dim = 8;
        let m = 4;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let bpq =
            BinaryPq::train(&data, n, dim, m, 10, &mut rng).expect("valid training parameters");
        let batch = bpq
            .encode_batch(&data, n)
            .expect("valid batch data dimensions");
        assert_eq!(batch.len(), n, "batch length should equal n");
        let expected_words = m.div_ceil(64);
        for (i, words) in batch.iter().enumerate() {
            assert_eq!(
                words.len(),
                expected_words,
                "row {i}: word length should be {expected_words}"
            );
        }
    }

    // 10. Hamming triangle inequality: hamming(a, c) ≤ hamming(a, b) + hamming(b, c).
    #[test]
    fn hamming_triangle_inequality() {
        let mut rng = make_rng(10);
        let n = 100;
        let dim = 8;
        let data = rand_vecs_normal(n, dim, &mut rng);
        let bpq =
            BinaryPq::train(&data, n, dim, 4, 10, &mut rng).expect("valid training parameters");

        for idx in 0..10 {
            let a = bpq
                .encode(&data[idx * dim..(idx + 1) * dim])
                .expect("valid vector dimension");
            let b = bpq
                .encode(&data[(idx + 1) * dim..(idx + 2) * dim])
                .expect("valid vector dimension");
            let c = bpq
                .encode(&data[(idx + 2) * dim..(idx + 3) * dim])
                .expect("valid vector dimension");
            let hab = bpq.hamming(&a, &b).expect("valid code word lengths");
            let hbc = bpq.hamming(&b, &c).expect("valid code word lengths");
            let hac = bpq.hamming(&a, &c).expect("valid code word lengths");
            assert!(
                hac <= hab + hbc,
                "triangle inequality violated: hamming(a,c)={hac} > hamming(a,b)={hab} + hamming(b,c)={hbc}"
            );
        }
    }
}
