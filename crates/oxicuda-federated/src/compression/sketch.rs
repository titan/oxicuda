//! Sketched gradient updates: Count-Sketch + Random Hadamard transform.
//!
//! **Count-Sketch** (Charikar, Chen & Farach-Colton, "Finding Frequent Items
//! in Data Streams", ICALP 2002): a linear sketch that maps a length-`n`
//! vector into a `depth × width` table using, per row, a bucket hash and a
//! ±1 sign hash.  Estimates are recovered as the median (over rows) of the
//! signed bucket values.  Because the sketch is linear, two sketches add
//! elementwise, which makes it directly usable for federated gradient
//! aggregation as in **FetchSGD** (Rothchild et al., "FetchSGD:
//! Communication-Efficient Federated Learning with Sketching", ICML 2020)
//! and the distributed-mean estimators of Suresh et al. ("Distributed Mean
//! Estimation with Limited Communication", ICML 2017).
//!
//! **Random Hadamard transform**: an orthonormal pre-conditioner `H · D`
//! where `D` is a random ±1 diagonal and `H` is the (normalised) Walsh-Hadamard
//! matrix.  Applied before quantization / sketching it spreads the energy of
//! the gradient evenly across coordinates so that no single entry dominates,
//! reducing the variance of subsequent compression.  The transform is
//! self-inverse on the zero-padded power-of-two space because both `H` (in its
//! orthonormal form) and `D` are involutions.

use crate::error::{FedError, FedResult};

// ─── Deterministic index hashing ───────────────────────────────────────────────

/// Odd 64-bit golden-ratio constant used to decorrelate the coordinate index.
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 finaliser — a high-quality avalanche over a 64-bit word.
///
/// This is a *pure* function of its argument (no RNG state), so the same
/// `(seed, row, index)` triple always maps to the same bucket and sign.
#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(GOLDEN);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Mix a `(seed, row, index)` triple into a 64-bit hash for the bucket choice.
#[inline]
fn mix_bucket(seed: u64, row: usize, index: usize) -> u64 {
    let r = row as u64;
    // Rotate the row so distinct rows decorrelate, fold in the index via the
    // golden ratio, then avalanche.
    splitmix64(seed ^ r.rotate_left(17) ^ (index as u64).wrapping_mul(GOLDEN))
}

/// Mix a `(seed, row, index)` triple into a 64-bit hash for the sign choice.
///
/// A different per-row salt keeps the sign hash independent of the bucket hash.
#[inline]
fn mix_sign(seed: u64, row: usize, index: usize) -> u64 {
    let r = (row as u64).wrapping_add(0xA5A5_5A5A_A5A5_5A5A);
    splitmix64(seed.rotate_left(31) ^ r.rotate_left(41) ^ (index as u64).wrapping_mul(GOLDEN))
}

// ─── Count-Sketch ──────────────────────────────────────────────────────────────

/// Configuration for a Count-Sketch.
#[derive(Debug, Clone, Copy)]
pub struct CountSketchConfig {
    /// Length of the dense vector being sketched.
    pub n_params: usize,
    /// Number of independent hash rows (more rows → tighter median estimate).
    pub depth: usize,
    /// Number of buckets per row (more buckets → fewer collisions).
    pub width: usize,
    /// Seed deriving all per-row bucket/sign hashes deterministically.
    pub seed: u64,
}

/// A linear Count-Sketch over a fixed-dimensional vector.
#[derive(Debug, Clone)]
pub struct CountSketch {
    config: CountSketchConfig,
}

impl CountSketch {
    /// Build a validated Count-Sketch.
    ///
    /// # Errors
    /// Returns [`FedError::Internal`] if `n_params`, `depth`, or `width` is zero.
    pub fn new(cfg: CountSketchConfig) -> FedResult<Self> {
        if cfg.n_params == 0 {
            return Err(FedError::Internal(
                "count-sketch: n_params must be >= 1".into(),
            ));
        }
        if cfg.depth == 0 {
            return Err(FedError::Internal(
                "count-sketch: depth must be >= 1".into(),
            ));
        }
        if cfg.width == 0 {
            return Err(FedError::Internal(
                "count-sketch: width must be >= 1".into(),
            ));
        }
        Ok(Self { config: cfg })
    }

    /// The sketch configuration.
    #[must_use]
    pub fn config(&self) -> CountSketchConfig {
        self.config
    }

    /// Total length of a sketch table (`depth * width`).
    #[must_use]
    pub fn table_len(&self) -> usize {
        self.config.depth * self.config.width
    }

    /// Bucket index `h_r(i)` for row `r`, coordinate `i`.
    #[inline]
    fn bucket(&self, row: usize, index: usize) -> usize {
        (mix_bucket(self.config.seed, row, index) % self.config.width as u64) as usize
    }

    /// Sign `s_r(i) ∈ {-1, +1}` for row `r`, coordinate `i`.
    #[inline]
    fn sign(&self, row: usize, index: usize) -> f32 {
        if mix_sign(self.config.seed, row, index) & 1 == 0 {
            1.0
        } else {
            -1.0
        }
    }

    /// Sketch a dense vector into a `depth × width` row-major table.
    ///
    /// `table[r*width + h_r(i)] += s_r(i) * g[i]`.
    ///
    /// # Errors
    /// Returns [`FedError::DimensionMismatch`] if `g.len() != n_params`.
    pub fn sketch(&self, g: &[f32]) -> FedResult<Vec<f32>> {
        if g.len() != self.config.n_params {
            return Err(FedError::DimensionMismatch {
                expected: self.config.n_params,
                got: g.len(),
            });
        }
        let width = self.config.width;
        let mut table = vec![0.0_f32; self.table_len()];
        for (i, &gi) in g.iter().enumerate() {
            for r in 0..self.config.depth {
                let b = self.bucket(r, i);
                table[r * width + b] += self.sign(r, i) * gi;
            }
        }
        Ok(table)
    }

    /// Estimate of coordinate `i` from a single row `r`.
    #[inline]
    fn row_estimate(&self, table: &[f32], row: usize, index: usize) -> f32 {
        let width = self.config.width;
        let b = self.bucket(row, index);
        self.sign(row, index) * table[row * width + b]
    }

    /// Median (across rows) estimate of coordinate `index` from a sketch table.
    fn median_estimate(&self, table: &[f32], index: usize) -> f32 {
        let depth = self.config.depth;
        let mut estimates = Vec::with_capacity(depth);
        for r in 0..depth {
            estimates.push(self.row_estimate(table, r, index));
        }
        estimates.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if depth % 2 == 1 {
            estimates[depth / 2]
        } else {
            // Average the two central order statistics for even depth.
            0.5 * (estimates[depth / 2 - 1] + estimates[depth / 2])
        }
    }

    /// Recover the full dense vector by taking, per coordinate, the median of
    /// the per-row signed bucket values.
    ///
    /// # Errors
    /// Returns [`FedError::DimensionMismatch`] if `table.len() != depth * width`.
    pub fn unsketch(&self, table: &[f32]) -> FedResult<Vec<f32>> {
        if table.len() != self.table_len() {
            return Err(FedError::DimensionMismatch {
                expected: self.table_len(),
                got: table.len(),
            });
        }
        let mut out = vec![0.0_f32; self.config.n_params];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.median_estimate(table, i);
        }
        Ok(out)
    }

    /// Merge two sketch tables of equal length by elementwise addition.
    ///
    /// Count-Sketch is linear, so `unsketch(merge(sketch(a), sketch(b)))`
    /// estimates `a + b`.
    ///
    /// # Errors
    /// Returns [`FedError::DimensionMismatch`] if the tables differ in length.
    pub fn merge(a: &[f32], b: &[f32]) -> FedResult<Vec<f32>> {
        if a.len() != b.len() {
            return Err(FedError::DimensionMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect())
    }

    /// Extract the `k` heavy hitters: the coordinates with the largest absolute
    /// median estimate.  Returns `(index, estimate)` pairs sorted by descending
    /// `|estimate|` (ties broken by ascending index).
    ///
    /// # Errors
    /// Returns [`FedError::DimensionMismatch`] if the table length is wrong, or
    /// if `k > n_params`.
    pub fn top_k_from_sketch(&self, table: &[f32], k: usize) -> FedResult<Vec<(usize, f32)>> {
        if table.len() != self.table_len() {
            return Err(FedError::DimensionMismatch {
                expected: self.table_len(),
                got: table.len(),
            });
        }
        if k > self.config.n_params {
            return Err(FedError::DimensionMismatch {
                expected: self.config.n_params,
                got: k,
            });
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut all: Vec<(usize, f32)> = (0..self.config.n_params)
            .map(|i| (i, self.median_estimate(table, i)))
            .collect();
        // Partial select: largest |estimate| first, ties by ascending index.
        let kth = k - 1;
        all.select_nth_unstable_by(kth, |&(ia, va), &(ib, vb)| {
            vb.abs()
                .partial_cmp(&va.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(ia.cmp(&ib))
        });
        let mut top: Vec<(usize, f32)> = all[..k].to_vec();
        top.sort_unstable_by(|&(ia, va), &(ib, vb)| {
            vb.abs()
                .partial_cmp(&va.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(ia.cmp(&ib))
        });
        Ok(top)
    }
}

// ─── Random Hadamard transform ──────────────────────────────────────────────────

/// Next power of two `>= n` (with `n >= 1`).
#[inline]
fn next_pow2(n: usize) -> usize {
    let mut p = 1_usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// Random Hadamard pre-conditioner `H · D` over a zero-padded power-of-two space.
#[derive(Debug, Clone)]
pub struct RandomHadamard {
    /// Logical input dimension.
    pub dim: usize,
    /// Padded dimension (next power of two `>= dim`).
    pub padded: usize,
    /// Seed deriving the ±1 diagonal `D`.
    seed: u64,
}

impl RandomHadamard {
    /// Build a validated Random Hadamard transform for vectors of length `dim`.
    ///
    /// # Errors
    /// Returns [`FedError::Internal`] if `dim == 0`.
    pub fn new(dim: usize, seed: u64) -> FedResult<Self> {
        if dim == 0 {
            return Err(FedError::Internal(
                "random-hadamard: dim must be >= 1".into(),
            ));
        }
        Ok(Self {
            dim,
            padded: next_pow2(dim),
            seed,
        })
    }

    /// Diagonal sign `D[i] ∈ {-1, +1}`, a pure function of `(seed, i)`.
    #[inline]
    fn diag_sign(&self, index: usize) -> f32 {
        if splitmix64(self.seed ^ (index as u64).wrapping_mul(GOLDEN)) & 1 == 0 {
            1.0
        } else {
            -1.0
        }
    }

    /// In-place orthonormal Fast Walsh-Hadamard transform on a power-of-two
    /// length buffer.  Each butterfly stage is scaled by `1/√2` so the overall
    /// transform is orthonormal (energy-preserving and self-inverse).
    fn fwht_orthonormal(buf: &mut [f32]) {
        let n = buf.len();
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let mut len = 1_usize;
        while len < n {
            let step = len << 1;
            let mut i = 0;
            while i < n {
                for j in i..i + len {
                    let a = buf[j];
                    let b = buf[j + len];
                    buf[j] = (a + b) * inv_sqrt2;
                    buf[j + len] = (a - b) * inv_sqrt2;
                }
                i += step;
            }
            len = step;
        }
    }

    /// Forward transform: orthonormal FWHT of `D ∘ pad(x)`.
    /// The returned vector has length `padded`.
    ///
    /// # Errors
    /// Returns [`FedError::DimensionMismatch`] if `x.len() != dim`.
    pub fn forward(&self, x: &[f32]) -> FedResult<Vec<f32>> {
        if x.len() != self.dim {
            return Err(FedError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        let mut buf = vec![0.0_f32; self.padded];
        for (i, &xi) in x.iter().enumerate() {
            buf[i] = self.diag_sign(i) * xi;
        }
        Self::fwht_orthonormal(&mut buf);
        Ok(buf)
    }

    /// Inverse transform → length `dim`.
    ///
    /// Since the orthonormal FWHT and the ±1 diagonal are both involutions on
    /// the padded space, applying FWHT then `D` and truncating to `dim`
    /// recovers the original input exactly (the padded tail being zero).
    ///
    /// # Errors
    /// Returns [`FedError::DimensionMismatch`] if `y.len() != padded`.
    pub fn inverse(&self, y: &[f32]) -> FedResult<Vec<f32>> {
        if y.len() != self.padded {
            return Err(FedError::DimensionMismatch {
                expected: self.padded,
                got: y.len(),
            });
        }
        let mut buf = y.to_vec();
        Self::fwht_orthonormal(&mut buf);
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot *= self.diag_sign(i);
        }
        buf.truncate(self.dim);
        Ok(buf)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sketch(n: usize, depth: usize, width: usize, seed: u64) -> CountSketch {
        CountSketch::new(CountSketchConfig {
            n_params: n,
            depth,
            width,
            seed,
        })
        .expect("test invariant: valid sketch config")
    }

    #[test]
    fn sketch_table_length() {
        let cs = sketch(100, 5, 64, 1);
        let g = vec![0.5_f32; 100];
        let table = cs.sketch(&g).expect("test invariant: valid sketch");
        assert_eq!(table.len(), 5 * 64);
        assert_eq!(table.len(), cs.table_len());
    }

    #[test]
    fn unsketch_length_matches_n_params() {
        let cs = sketch(37, 3, 128, 2);
        let g = vec![1.0_f32; 37];
        let table = cs.sketch(&g).expect("test invariant: valid sketch");
        let rec = cs.unsketch(&table).expect("test invariant: valid unsketch");
        assert_eq!(rec.len(), 37);
    }

    #[test]
    fn single_spike_recovered() {
        // A wide table with several rows: the lone nonzero rarely collides, so
        // the median estimate recovers ~5 at i and ~0 elsewhere.
        let n = 64;
        let cs = sketch(n, 5, 2048, 12345);
        let mut g = vec![0.0_f32; n];
        let spike_idx = 17;
        g[spike_idx] = 5.0;
        let table = cs.sketch(&g).expect("test invariant: valid sketch");
        let rec = cs.unsketch(&table).expect("test invariant: valid unsketch");
        assert!(
            (rec[spike_idx] - 5.0).abs() < 1e-3,
            "spike not recovered: {}",
            rec[spike_idx]
        );
        for (i, &v) in rec.iter().enumerate() {
            if i != spike_idx {
                assert!(v.abs() < 1e-3, "leakage at {i}: {v}");
            }
        }
    }

    #[test]
    fn sketch_is_linear() {
        let n = 50;
        let cs = sketch(n, 4, 256, 7);
        let mut a = vec![0.0_f32; n];
        let mut b = vec![0.0_f32; n];
        for i in 0..n {
            a[i] = (i as f32) * 0.1 - 2.0;
            b[i] = ((n - i) as f32) * 0.05;
        }
        let sum: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
        let sa = cs.sketch(&a).expect("test invariant: valid sketch");
        let sb = cs.sketch(&b).expect("test invariant: valid sketch");
        let ssum = cs.sketch(&sum).expect("test invariant: valid sketch");
        for j in 0..sa.len() {
            assert!(
                ((sa[j] + sb[j]) - ssum[j]).abs() < 1e-4,
                "linearity broken at {j}: {} vs {}",
                sa[j] + sb[j],
                ssum[j]
            );
        }
    }

    #[test]
    fn merge_then_unsketch_recovers_sum() {
        let n = 80;
        let cs = sketch(n, 5, 2048, 999);
        let mut a = vec![0.0_f32; n];
        let mut b = vec![0.0_f32; n];
        a[3] = 4.0;
        a[40] = -2.0;
        b[40] = 1.0;
        b[71] = 3.0;
        let sa = cs.sketch(&a).expect("test invariant: valid sketch");
        let sb = cs.sketch(&b).expect("test invariant: valid sketch");
        let merged = CountSketch::merge(&sa, &sb).expect("test invariant: valid merge");
        let rec = cs
            .unsketch(&merged)
            .expect("test invariant: valid unsketch");
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
        for i in 0..n {
            assert!(
                (rec[i] - expected[i]).abs() < 1e-3,
                "merge+unsketch mismatch at {i}: {} vs {}",
                rec[i],
                expected[i]
            );
        }
    }

    #[test]
    fn top_k_returns_largest_magnitude_coords() {
        let n = 100;
        let cs = sketch(n, 5, 4096, 31337);
        let mut g = vec![0.0_f32; n];
        g[10] = 9.0;
        g[20] = -7.0;
        g[30] = 5.0;
        g[40] = 1.0;
        let table = cs.sketch(&g).expect("test invariant: valid sketch");
        let top = cs
            .top_k_from_sketch(&table, 3)
            .expect("test invariant: valid top-k");
        assert_eq!(top.len(), 3);
        let idxs: Vec<usize> = top.iter().map(|&(i, _)| i).collect();
        assert_eq!(idxs, vec![10, 20, 30]);
        assert!((top[0].1 - 9.0).abs() < 1e-2);
    }

    #[test]
    fn top_k_zero_returns_empty() {
        let cs = sketch(10, 3, 64, 5);
        let g = vec![1.0_f32; 10];
        let table = cs.sketch(&g).expect("test invariant: valid sketch");
        let top = cs
            .top_k_from_sketch(&table, 0)
            .expect("test invariant: valid top-k");
        assert!(top.is_empty());
    }

    #[test]
    fn sketch_is_deterministic() {
        let cfg = CountSketchConfig {
            n_params: 64,
            depth: 4,
            width: 128,
            seed: 4242,
        };
        let cs_a = CountSketch::new(cfg).expect("test invariant: valid sketch");
        let cs_b = CountSketch::new(cfg).expect("test invariant: valid sketch");
        let mut g = vec![0.0_f32; 64];
        for (i, slot) in g.iter_mut().enumerate() {
            *slot = (i as f32).sin();
        }
        let ta = cs_a.sketch(&g).expect("test invariant: valid sketch");
        let tb = cs_b.sketch(&g).expect("test invariant: valid sketch");
        assert_eq!(ta, tb);
    }

    #[test]
    fn random_hadamard_padded_is_next_pow2() {
        assert_eq!(
            RandomHadamard::new(1, 0)
                .expect("test invariant: valid hadamard")
                .padded,
            1
        );
        assert_eq!(
            RandomHadamard::new(5, 0)
                .expect("test invariant: valid hadamard")
                .padded,
            8
        );
        assert_eq!(
            RandomHadamard::new(8, 0)
                .expect("test invariant: valid hadamard")
                .padded,
            8
        );
        assert_eq!(
            RandomHadamard::new(1000, 0)
                .expect("test invariant: valid hadamard")
                .padded,
            1024
        );
    }

    #[test]
    fn random_hadamard_forward_length_is_padded() {
        let h = RandomHadamard::new(5, 11).expect("test invariant: valid hadamard");
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let y = h.forward(&x).expect("test invariant: valid forward");
        assert_eq!(y.len(), 8);
    }

    #[test]
    fn random_hadamard_inverse_recovers_pow2_input() {
        let h = RandomHadamard::new(8, 77).expect("test invariant: valid hadamard");
        let x = vec![1.0_f32, -2.0, 3.5, 0.0, -1.5, 4.0, -0.25, 2.0];
        let y = h.forward(&x).expect("test invariant: valid forward");
        let rec = h.inverse(&y).expect("test invariant: valid inverse");
        assert_eq!(rec.len(), 8);
        for (a, &b) in rec.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-4, "recover mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn random_hadamard_inverse_recovers_non_pow2_input() {
        let h = RandomHadamard::new(5, 314).expect("test invariant: valid hadamard");
        let x = vec![3.0_f32, -1.0, 2.5, 0.5, -4.0];
        let y = h.forward(&x).expect("test invariant: valid forward");
        let rec = h.inverse(&y).expect("test invariant: valid inverse");
        assert_eq!(rec.len(), 5);
        for (a, &b) in rec.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-4, "recover mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn random_hadamard_preserves_energy_pow2() {
        let h = RandomHadamard::new(16, 2024).expect("test invariant: valid hadamard");
        let mut x = vec![0.0_f32; 16];
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = (i as f32) * 0.3 - 2.0;
        }
        let y = h.forward(&x).expect("test invariant: valid forward");
        let nx: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let ny: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((nx - ny).abs() < 1e-3, "energy not preserved: {nx} vs {ny}");
    }

    #[test]
    fn new_rejects_zero_n_params() {
        let cfg = CountSketchConfig {
            n_params: 0,
            depth: 3,
            width: 4,
            seed: 1,
        };
        assert!(matches!(CountSketch::new(cfg), Err(FedError::Internal(_))));
    }

    #[test]
    fn new_rejects_zero_depth() {
        let cfg = CountSketchConfig {
            n_params: 4,
            depth: 0,
            width: 4,
            seed: 1,
        };
        assert!(matches!(CountSketch::new(cfg), Err(FedError::Internal(_))));
    }

    #[test]
    fn new_rejects_zero_width() {
        let cfg = CountSketchConfig {
            n_params: 4,
            depth: 3,
            width: 0,
            seed: 1,
        };
        assert!(matches!(CountSketch::new(cfg), Err(FedError::Internal(_))));
    }

    #[test]
    fn sketch_rejects_wrong_input_length() {
        let cs = sketch(10, 3, 16, 1);
        let g = vec![1.0_f32; 9];
        assert!(matches!(
            cs.sketch(&g),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn unsketch_rejects_wrong_table_length() {
        let cs = sketch(10, 3, 16, 1);
        let table = vec![0.0_f32; 47];
        assert!(matches!(
            cs.unsketch(&table),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn top_k_rejects_k_too_large() {
        let cs = sketch(10, 3, 16, 1);
        let g = vec![1.0_f32; 10];
        let table = cs.sketch(&g).expect("test invariant: valid sketch");
        assert!(matches!(
            cs.top_k_from_sketch(&table, 11),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn random_hadamard_rejects_zero_dim() {
        assert!(matches!(
            RandomHadamard::new(0, 1),
            Err(FedError::Internal(_))
        ));
    }

    #[test]
    fn random_hadamard_forward_rejects_wrong_length() {
        let h = RandomHadamard::new(5, 1).expect("test invariant: valid hadamard");
        let x = vec![1.0_f32; 6];
        assert!(matches!(
            h.forward(&x),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn random_hadamard_inverse_rejects_wrong_length() {
        let h = RandomHadamard::new(5, 1).expect("test invariant: valid hadamard");
        let y = vec![1.0_f32; 5];
        assert!(matches!(
            h.inverse(&y),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn merge_rejects_length_mismatch() {
        let a = vec![1.0_f32; 4];
        let b = vec![1.0_f32; 5];
        assert!(matches!(
            CountSketch::merge(&a, &b),
            Err(FedError::DimensionMismatch { .. })
        ));
    }
}
