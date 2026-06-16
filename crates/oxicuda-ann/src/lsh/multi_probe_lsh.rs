//! Multi-probe LSH (Locality-Sensitive Hashing).
//!
//! Lv, Josephson, Wang, Charikar, Li, "Multi-Probe LSH: Efficient Indexing for
//! High-Dimensional Similarity Search", VLDB 2007.
//!
//! Standard E2LSH for L2 builds `L` independent hash tables; each table maps a
//! vector `x` to an integer key `g_i(x) ∈ Z^k`:
//!
//! ```text
//! g_i(x) = (⌊(a_{ij}·x + b_{ij}) / w⌋)_{j=1..k}
//! ```
//!
//! where `a_{ij} ~ N(0, I_d)` and `b_{ij} ~ U[0, w]`.
//!
//! For a query `q` the standard approach probes only the bucket `g_i(q)` in
//! each of the `L` tables — to compensate for the high false-negative rate one
//! needs a large `L`.  Multi-probe LSH reduces the required `L` by also
//! visiting a handful of buckets *close* to `g_i(q)`.  Closeness is measured by
//! the *expected* L2 distance between `q` and the buckets boundary it crossed.
//!
//! For component `j` with continuous projection `p_{ij}(q) = a_{ij}·q + b_{ij}`
//! and bucket key `⌊p_{ij}(q)/w⌋`, the score is `s_{ij}(q) = p_{ij}(q) − w·⌊…⌋`
//! and:
//!
//! - Perturbing component `j` by `−1` (next lower bucket) has expected distance
//!   `s_{ij}` to the boundary.
//! - Perturbing component `j` by `+1` (next higher bucket) has expected distance
//!   `w − s_{ij}` to the boundary.
//!
//! We sort all candidate single-component (and, when budget exceeds `k`,
//! pairwise) perturbations by their expected distance and probe in that order.
use crate::error::{AnnError, AnnResult};
use crate::handle::LcgRng;
use std::collections::HashMap;

/// Configuration for `MultiProbeLsh`.
#[derive(Debug, Clone, Copy)]
pub struct MultiProbeLshConfig {
    /// Number of independent hash tables (L).  Must be ≥ 1.
    pub n_tables: usize,
    /// Hash key dimension (k).  Must be ≥ 1.
    pub hash_dim: usize,
    /// Bucket width (w).  Must be > 0.
    pub bucket_width: f32,
    /// Max probe sequences per table including the zero perturbation.  ≥ 1.
    pub probe_budget: usize,
}

impl MultiProbeLshConfig {
    fn validate(&self) -> AnnResult<()> {
        if self.n_tables == 0 {
            return Err(AnnError::InvalidLayerCount { n: 0 });
        }
        if self.hash_dim == 0 {
            return Err(AnnError::InvalidVectorDim { dim: 0 });
        }
        if !(self.bucket_width.is_finite() && self.bucket_width > 0.0) {
            return Err(AnnError::Internal {
                msg: format!(
                    "multi_probe_lsh: bucket_width must be > 0, got {}",
                    self.bucket_width
                ),
            });
        }
        if self.probe_budget == 0 {
            return Err(AnnError::InvalidK {
                k: 0,
                n: usize::MAX,
            });
        }
        Ok(())
    }
}

/// Multi-probe LSH index with `L` tables of `k` hyperplane projections each.
#[derive(Debug)]
pub struct MultiProbeLsh {
    /// `tables[t]` maps `g_t(x)` (length-k i32 key) to a list of inserted ids.
    pub tables: Vec<HashMap<Vec<i32>, Vec<usize>>>,
    /// `projections[t]` is k*dim row-major Gaussian projections for table t.
    pub projections: Vec<Vec<f32>>,
    /// `biases[t]` is a k-vector with entries drawn ~ U[0, w].
    pub biases: Vec<Vec<f32>>,
    pub n_tables: usize,
    pub hash_dim: usize,
    pub bucket_width: f32,
    pub dim: usize,
    pub probe_budget: usize,
}

// ─── single-component perturbation candidate ─────────────────────────────────

/// A single delta on one of `k` components of the hash key (±1).
#[derive(Debug, Clone, Copy)]
struct ComponentDelta {
    component: usize,
    delta: i32,
    /// Expected perturbation distance to that boundary.
    distance: f32,
}

// ─── ordering helper used by probe-sequence generation ──────────────────────

/// A composite perturbation: a small set of single-component deltas.  We
/// represent it sparsely as a sorted Vec of (component, delta) pairs and a
/// pre-computed sum of `distance` over those deltas.
#[derive(Debug, Clone)]
struct ProbeSeq {
    /// Sorted by `component`.
    deltas: Vec<(usize, i32)>,
    sum_dist: f32,
}

// ─── construction ────────────────────────────────────────────────────────────

impl MultiProbeLsh {
    /// Initialise an empty multi-probe LSH index.
    pub fn new(cfg: MultiProbeLshConfig, dim: usize, rng: &mut LcgRng) -> AnnResult<Self> {
        cfg.validate()?;
        if dim == 0 {
            return Err(AnnError::InvalidVectorDim { dim: 0 });
        }

        let mut projections: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_tables);
        let mut biases: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_tables);
        let mut tables: Vec<HashMap<Vec<i32>, Vec<usize>>> = Vec::with_capacity(cfg.n_tables);

        for _ in 0..cfg.n_tables {
            let mut a = vec![0.0_f32; cfg.hash_dim * dim];
            rng.fill_normal(&mut a);
            let mut b = vec![0.0_f32; cfg.hash_dim];
            for slot in b.iter_mut() {
                *slot = rng.next_f32() * cfg.bucket_width;
            }
            projections.push(a);
            biases.push(b);
            tables.push(HashMap::new());
        }

        Ok(Self {
            tables,
            projections,
            biases,
            n_tables: cfg.n_tables,
            hash_dim: cfg.hash_dim,
            bucket_width: cfg.bucket_width,
            dim,
            probe_budget: cfg.probe_budget,
        })
    }

    /// Continuous projection vector `p_{ij} = a_{ij}·x + b_{ij}` for table `t`.
    fn continuous_projection(&self, t: usize, x: &[f32]) -> AnnResult<Vec<f32>> {
        let a = self.projections.get(t).ok_or(AnnError::Internal {
            msg: format!("multi_probe_lsh: projection table {t} out of range"),
        })?;
        let b = self.biases.get(t).ok_or(AnnError::Internal {
            msg: format!("multi_probe_lsh: bias table {t} out of range"),
        })?;
        let k = self.hash_dim;
        let d = self.dim;
        let mut out = vec![0.0_f32; k];
        for j in 0..k {
            let row_off = j * d;
            let row_end = row_off + d;
            let row = a.get(row_off..row_end).ok_or(AnnError::Internal {
                msg: format!(
                    "multi_probe_lsh: projection slice [{row_off},{row_end}) out of range"
                ),
            })?;
            let mut s = 0.0_f32;
            for (rv, xv) in row.iter().zip(x.iter()) {
                s += *rv * *xv;
            }
            let bj = b.get(j).copied().ok_or(AnnError::Internal {
                msg: format!("multi_probe_lsh: bias {j} out of range"),
            })?;
            let slot = out.get_mut(j).ok_or(AnnError::Internal {
                msg: format!("multi_probe_lsh: continuous output {j} out of range"),
            })?;
            *slot = s + bj;
        }
        Ok(out)
    }

    /// Bucket key `g_t(x) = ⌊p/w⌋`.
    fn key_from_projection(&self, projection: &[f32]) -> Vec<i32> {
        projection
            .iter()
            .map(|&p| (p / self.bucket_width).floor() as i32)
            .collect()
    }

    /// Bucket key for `x` in table `t`.
    pub fn hash_key(&self, t: usize, x: &[f32]) -> AnnResult<Vec<i32>> {
        if x.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        let p = self.continuous_projection(t, x)?;
        Ok(self.key_from_projection(&p))
    }

    /// Insert id `x_id` into the bucket corresponding to `g_t(x)` in every table.
    pub fn insert(&mut self, x_id: usize, x: &[f32]) -> AnnResult<()> {
        if x.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        for t in 0..self.n_tables {
            let key = self.hash_key(t, x)?;
            let bucket = self
                .tables
                .get_mut(t)
                .ok_or(AnnError::Internal {
                    msg: format!("multi_probe_lsh: table {t} out of range"),
                })?
                .entry(key)
                .or_default();
            if !bucket.contains(&x_id) {
                bucket.push(x_id);
            }
        }
        Ok(())
    }

    /// Generate the perturbation sequences for a query in a single table,
    /// sorted by non-decreasing expected distance.
    ///
    /// The first entry is always the zero perturbation (no deltas, distance 0).
    /// Up to `budget` sequences are returned.  Multi-component perturbations
    /// (`budget > k`) are emitted as the union of two distinct single-component
    /// deltas chosen in order of cumulative distance.
    fn probe_sequences(&self, scores: &[f32], budget: usize) -> Vec<ProbeSeq> {
        let k = self.hash_dim;
        let w = self.bucket_width;
        if budget == 0 {
            return Vec::new();
        }

        // 1. Single-component candidates: each component j has two boundary distances.
        let mut singles: Vec<ComponentDelta> = Vec::with_capacity(2 * k);
        for j in 0..k {
            let s_j = scores.get(j).copied().unwrap_or(0.0);
            singles.push(ComponentDelta {
                component: j,
                delta: -1,
                distance: s_j.max(0.0),
            });
            singles.push(ComponentDelta {
                component: j,
                delta: 1,
                distance: (w - s_j).max(0.0),
            });
        }
        singles.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 2. zero perturbation always first
        let mut out: Vec<ProbeSeq> = Vec::with_capacity(budget);
        out.push(ProbeSeq {
            deltas: Vec::new(),
            sum_dist: 0.0,
        });

        // 3. add single-component candidates in order
        for s in &singles {
            if out.len() >= budget {
                return out;
            }
            out.push(ProbeSeq {
                deltas: vec![(s.component, s.delta)],
                sum_dist: s.distance,
            });
        }

        // 4. If still room: pairwise perturbations.  For each pair of distinct
        //    single-component candidates touching DIFFERENT components, emit
        //    their merge with summed distance.  We collect the merges, sort by
        //    distance, and append in order.
        if out.len() < budget {
            let mut pairs: Vec<ProbeSeq> = Vec::new();
            for (i, a) in singles.iter().enumerate() {
                for b in singles.iter().skip(i + 1) {
                    if a.component == b.component {
                        continue;
                    }
                    let merged_dist = a.distance + b.distance;
                    let mut deltas = vec![(a.component, a.delta), (b.component, b.delta)];
                    deltas.sort_by_key(|d| d.0);
                    pairs.push(ProbeSeq {
                        deltas,
                        sum_dist: merged_dist,
                    });
                }
            }
            pairs.sort_by(|x, y| {
                x.sum_dist
                    .partial_cmp(&y.sum_dist)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for p in pairs {
                if out.len() >= budget {
                    break;
                }
                out.push(p);
            }
        }

        out
    }

    /// Apply a probe sequence to a base key.
    fn apply_seq(base: &[i32], seq: &ProbeSeq) -> Vec<i32> {
        let mut out = base.to_vec();
        for &(c, d) in &seq.deltas {
            if let Some(slot) = out.get_mut(c) {
                *slot += d;
            }
        }
        out
    }

    /// Query: return up to `max_candidates` deduped ids by probing each table
    /// with up to `probe_budget` perturbation keys in increasing-distance order.
    pub fn query(&self, q: &[f32], max_candidates: usize) -> AnnResult<Vec<usize>> {
        if q.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: q.len(),
            });
        }

        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut out: Vec<usize> = Vec::new();
        if max_candidates == 0 {
            return Ok(out);
        }

        for t in 0..self.n_tables {
            let proj = self.continuous_projection(t, q)?;
            let base_key = self.key_from_projection(&proj);
            // Continuous score = p - w*⌊p/w⌋
            let scores: Vec<f32> = proj
                .iter()
                .zip(base_key.iter())
                .map(|(p, k)| p - self.bucket_width * (*k as f32))
                .collect();

            let seqs = self.probe_sequences(&scores, self.probe_budget);
            let bucket_map = self.tables.get(t).ok_or(AnnError::Internal {
                msg: format!("multi_probe_lsh: table {t} out of range"),
            })?;
            for seq in &seqs {
                let key = Self::apply_seq(&base_key, seq);
                if let Some(bucket) = bucket_map.get(&key) {
                    for &id in bucket {
                        if seen.insert(id) {
                            out.push(id);
                            if out.len() >= max_candidates {
                                return Ok(out);
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Public introspection helper: ordered probe sequence distances for a
    /// single query in table `t`, useful for testing the ordering property.
    pub fn probe_distances_for(&self, t: usize, q: &[f32]) -> AnnResult<Vec<f32>> {
        if q.len() != self.dim {
            return Err(AnnError::DimensionMismatch {
                expected: self.dim,
                got: q.len(),
            });
        }
        if t >= self.n_tables {
            return Err(AnnError::IdOutOfRange {
                id: t,
                n: self.n_tables,
            });
        }
        let proj = self.continuous_projection(t, q)?;
        let base_key = self.key_from_projection(&proj);
        let scores: Vec<f32> = proj
            .iter()
            .zip(base_key.iter())
            .map(|(p, k)| p - self.bucket_width * (*k as f32))
            .collect();
        Ok(self
            .probe_sequences(&scores, self.probe_budget)
            .into_iter()
            .map(|s| s.sum_dist)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn rand_vec(dim: usize, rng: &mut LcgRng) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        rng.fill_normal(&mut v);
        v
    }

    fn small_index(seed: u64, n_tables: usize, probe_budget: usize) -> MultiProbeLsh {
        let cfg = MultiProbeLshConfig {
            n_tables,
            hash_dim: 4,
            bucket_width: 1.5,
            probe_budget,
        };
        let mut rng = LcgRng::new(seed);
        MultiProbeLsh::new(cfg, 8, &mut rng)
            .expect("valid LSH config should construct successfully")
    }

    // 1. determinism: same seed → identical projections, biases, hashes.
    #[test]
    fn mp_lsh_deterministic_construction() {
        let cfg = MultiProbeLshConfig {
            n_tables: 3,
            hash_dim: 4,
            bucket_width: 1.5,
            probe_budget: 4,
        };
        let mut r1 = LcgRng::new(7);
        let mut r2 = LcgRng::new(7);
        let a = MultiProbeLsh::new(cfg, 8, &mut r1)
            .expect("valid LSH config should construct successfully");
        let b = MultiProbeLsh::new(cfg, 8, &mut r2)
            .expect("valid LSH config should construct successfully");
        for t in 0..3 {
            assert_eq!(a.projections[t], b.projections[t]);
            assert_eq!(a.biases[t], b.biases[t]);
        }
        let v = vec![0.1_f32, -0.2, 0.3, 0.4, 0.5, -0.6, 0.7, -0.8];
        for t in 0..3 {
            assert_eq!(
                a.hash_key(t, &v).expect("valid table index and vector"),
                b.hash_key(t, &v).expect("valid table index and vector")
            );
        }
    }

    // 2. insert places x_id into the bucket corresponding to g_t(x) in every table
    #[test]
    fn mp_lsh_insert_places_in_correct_buckets() {
        let mut idx = small_index(11, 3, 4);
        let mut rng = LcgRng::new(99);
        let v = rand_vec(8, &mut rng);
        idx.insert(42, &v).expect("valid vector dimension");
        for t in 0..idx.n_tables {
            let key = idx.hash_key(t, &v).expect("valid table index and vector");
            let bucket = idx.tables[t]
                .get(&key)
                .expect("key must exist after insert");
            assert!(bucket.contains(&42));
        }
    }

    // 3. near-duplicate recovered as candidate with high probability
    #[test]
    fn mp_lsh_near_duplicate_recovered() {
        let cfg = MultiProbeLshConfig {
            n_tables: 6,
            hash_dim: 6,
            bucket_width: 1.0,
            probe_budget: 16,
        };
        let mut succ = 0usize;
        let n_trials = 8;
        for seed in 0..n_trials {
            let mut rng = LcgRng::new(seed as u64 + 7);
            let mut idx = MultiProbeLsh::new(cfg, 8, &mut rng)
                .expect("valid LSH config should construct successfully");
            let base = rand_vec(8, &mut rng);
            idx.insert(0, &base).expect("valid vector dimension");
            // small additive noise on top
            let mut noisy = base.clone();
            for s in &mut noisy {
                *s += (rng.next_f32() - 0.5) * 0.05;
            }
            let cands = idx.query(&noisy, 32).expect("valid query dimension");
            if cands.contains(&0) {
                succ += 1;
            }
        }
        assert!(succ >= n_trials / 2, "low recall: {succ}/{n_trials}");
    }

    // 4. irrelevant (far) query mostly returns different ids
    #[test]
    fn mp_lsh_far_query_mostly_different() {
        let cfg = MultiProbeLshConfig {
            n_tables: 2,
            hash_dim: 6,
            bucket_width: 0.5,
            probe_budget: 1,
        };
        let mut rng = LcgRng::new(33);
        let mut idx = MultiProbeLsh::new(cfg, 8, &mut rng)
            .expect("valid LSH config should construct successfully");
        let v = vec![0.0_f32; 8];
        idx.insert(0, &v).expect("valid vector dimension");
        // far query
        let q = vec![100.0_f32; 8];
        let cands = idx.query(&q, 8).expect("valid query dimension");
        assert!(!cands.contains(&0));
    }

    // 5. probe_budget=1 == single-bucket LSH probe (zero perturbation only)
    #[test]
    fn mp_lsh_budget_one_eq_single_bucket() {
        let cfg_one = MultiProbeLshConfig {
            n_tables: 2,
            hash_dim: 3,
            bucket_width: 2.0,
            probe_budget: 1,
        };
        let mut rng = LcgRng::new(55);
        let mut idx = MultiProbeLsh::new(cfg_one, 8, &mut rng)
            .expect("valid LSH config should construct successfully");
        // populate
        let mut r2 = LcgRng::new(77);
        let mut all_vecs: Vec<Vec<f32>> = Vec::new();
        for i in 0..16 {
            let v = rand_vec(8, &mut r2);
            idx.insert(i, &v).expect("valid vector dimension");
            all_vecs.push(v);
        }
        // query: candidates returned must exactly be union of base buckets across tables
        let q = rand_vec(8, &mut r2);
        let cands = idx.query(&q, 64).expect("valid query dimension");
        let mut expected: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for t in 0..idx.n_tables {
            let key = idx.hash_key(t, &q).expect("valid table index and vector");
            if let Some(b) = idx.tables[t].get(&key) {
                for id in b {
                    expected.insert(*id);
                }
            }
        }
        let got: std::collections::HashSet<usize> = cands.iter().copied().collect();
        assert_eq!(got, expected);
    }

    // 6. budget>1 increases recall on a hand-constructed boundary case
    #[test]
    fn mp_lsh_budget_increases_recall_boundary() {
        // Construct an index where the inserted point is exactly on a bucket
        // boundary so that a small perturbation of the query crosses out and
        // a single-bucket LSH would miss it but a 2-probe LSH finds it.
        // We use a fixed projection by hand-choosing a 1-d, k=1 index.
        let cfg_zero = MultiProbeLshConfig {
            n_tables: 1,
            hash_dim: 1,
            bucket_width: 1.0,
            probe_budget: 1,
        };
        let cfg_multi = MultiProbeLshConfig {
            n_tables: 1,
            hash_dim: 1,
            bucket_width: 1.0,
            probe_budget: 3,
        };
        let mut rng = LcgRng::new(1);
        let mut a = MultiProbeLsh::new(cfg_zero, 1, &mut rng)
            .expect("valid LSH config should construct successfully");
        // overwrite to a deterministic projection: a = [1], b = [0].
        a.projections[0] = vec![1.0_f32];
        a.biases[0] = vec![0.0_f32];
        // and rebuild bucket map from scratch
        a.tables[0].clear();
        a.insert(0, &[0.05_f32]).expect("valid vector dimension"); // bucket 0 (since 0.05/1 -> floor=0)

        let mut b_index = MultiProbeLsh::new(cfg_multi, 1, &mut rng)
            .expect("valid LSH config should construct successfully");
        b_index.projections[0] = vec![1.0_f32];
        b_index.biases[0] = vec![0.0_f32];
        b_index.tables[0].clear();
        b_index
            .insert(0, &[0.05_f32])
            .expect("valid vector dimension");

        // Query at 1.05 → bucket = floor(1.05) = 1, so single-bucket LSH misses point 0
        let q = &[1.05_f32];
        let cands_a = a.query(q, 4).expect("valid query dimension");
        let cands_b = b_index.query(q, 4).expect("valid query dimension");
        assert!(!cands_a.contains(&0));
        assert!(cands_b.contains(&0));
    }

    // 7. bucket key formula by hand on a fixed (a, b, w, x)
    #[test]
    fn mp_lsh_hash_formula_by_hand() {
        let cfg = MultiProbeLshConfig {
            n_tables: 1,
            hash_dim: 2,
            bucket_width: 2.0,
            probe_budget: 1,
        };
        let mut rng = LcgRng::new(0);
        let mut idx = MultiProbeLsh::new(cfg, 3, &mut rng)
            .expect("valid LSH config should construct successfully");
        // override projections & biases
        idx.projections[0] = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0]; // 2 rows x 3 cols
        idx.biases[0] = vec![0.0_f32, 1.0];
        let x = vec![3.0_f32, 5.5, 0.0];
        // p0 = 1*3 + 0*5.5 + 0*0 + 0 = 3   ; ⌊3/2⌋ = 1
        // p1 = 0*3 + 1*5.5 + 0*0 + 1 = 6.5 ; ⌊6.5/2⌋ = 3
        let key = idx.hash_key(0, &x).expect("valid table index and vector");
        assert_eq!(key, vec![1, 3]);
    }

    // 8. perturbation sequence ordered by non-decreasing expected distance
    #[test]
    fn mp_lsh_probe_seq_ordered_by_distance() {
        let cfg = MultiProbeLshConfig {
            n_tables: 1,
            hash_dim: 4,
            bucket_width: 1.0,
            probe_budget: 8,
        };
        let mut rng = LcgRng::new(2);
        let idx = MultiProbeLsh::new(cfg, 8, &mut rng)
            .expect("valid LSH config should construct successfully");
        let q = rand_vec(8, &mut LcgRng::new(123));
        let dists = idx
            .probe_distances_for(0, &q)
            .expect("valid table index and query dimension");
        assert!(dists.len() >= 2);
        for w in dists.windows(2) {
            assert!(w[0] <= w[1] + 1e-6, "probe order violated: {w:?}");
        }
    }

    // 9. empty index → empty candidates
    #[test]
    fn mp_lsh_empty_index_empty_query() {
        let idx = small_index(0, 2, 2);
        let q = vec![0.5_f32; 8];
        let cands = idx.query(&q, 10).expect("valid query dimension");
        assert!(cands.is_empty());
    }

    // 10. insertion order independent
    #[test]
    fn mp_lsh_insertion_order_independent() {
        let cfg = MultiProbeLshConfig {
            n_tables: 2,
            hash_dim: 3,
            bucket_width: 1.0,
            probe_budget: 1,
        };
        let mut r1 = LcgRng::new(5);
        let mut a = MultiProbeLsh::new(cfg, 4, &mut r1)
            .expect("valid LSH config should construct successfully");
        let mut r2 = LcgRng::new(5);
        let mut b = MultiProbeLsh::new(cfg, 4, &mut r2)
            .expect("valid LSH config should construct successfully");

        let mut data_rng = LcgRng::new(42);
        let data: Vec<Vec<f32>> = (0..8).map(|_| rand_vec(4, &mut data_rng)).collect();

        // order A: 0..n
        for (i, v) in data.iter().enumerate() {
            a.insert(i, v).expect("valid vector dimension");
        }
        // order B: reversed
        for (i, v) in data.iter().enumerate().rev() {
            b.insert(i, v).expect("valid vector dimension");
        }
        for t in 0..a.n_tables {
            let mut ka: Vec<&Vec<i32>> = a.tables[t].keys().collect();
            let mut kb: Vec<&Vec<i32>> = b.tables[t].keys().collect();
            ka.sort();
            kb.sort();
            assert_eq!(ka, kb);
            for k in ka {
                let mut va = a.tables[t]
                    .get(k)
                    .expect("key must exist in both tables")
                    .clone();
                let mut vb = b.tables[t]
                    .get(k)
                    .expect("key must exist in both tables")
                    .clone();
                va.sort();
                vb.sort();
                assert_eq!(va, vb);
            }
        }
    }

    // 11. duplicate insert of same id appears once per bucket
    #[test]
    fn mp_lsh_duplicate_insert_idempotent() {
        let mut idx = small_index(3, 2, 1);
        let mut rng = LcgRng::new(8);
        let v = rand_vec(8, &mut rng);
        idx.insert(99, &v).expect("valid vector dimension");
        idx.insert(99, &v).expect("valid vector dimension");
        idx.insert(99, &v).expect("valid vector dimension");
        for t in 0..idx.n_tables {
            let key = idx.hash_key(t, &v).expect("valid table index and vector");
            let bucket = idx.tables[t]
                .get(&key)
                .expect("key must exist after insert");
            let count = bucket.iter().filter(|id| **id == 99).count();
            assert_eq!(count, 1);
        }
    }

    // 12. err k=0
    #[test]
    fn mp_lsh_err_k_zero() {
        let cfg = MultiProbeLshConfig {
            n_tables: 1,
            hash_dim: 0,
            bucket_width: 1.0,
            probe_budget: 1,
        };
        let mut rng = LcgRng::new(0);
        let err = MultiProbeLsh::new(cfg, 4, &mut rng);
        assert!(matches!(err, Err(AnnError::InvalidVectorDim { dim: 0 })));
    }

    // 13. err L=0
    #[test]
    fn mp_lsh_err_l_zero() {
        let cfg = MultiProbeLshConfig {
            n_tables: 0,
            hash_dim: 2,
            bucket_width: 1.0,
            probe_budget: 1,
        };
        let mut rng = LcgRng::new(0);
        let err = MultiProbeLsh::new(cfg, 4, &mut rng);
        assert!(matches!(err, Err(AnnError::InvalidLayerCount { n: 0 })));
    }

    // 14. err w ≤ 0
    #[test]
    fn mp_lsh_err_w_non_positive() {
        let cfg = MultiProbeLshConfig {
            n_tables: 1,
            hash_dim: 2,
            bucket_width: 0.0,
            probe_budget: 1,
        };
        let mut rng = LcgRng::new(0);
        let err = MultiProbeLsh::new(cfg, 4, &mut rng);
        assert!(matches!(err, Err(AnnError::Internal { .. })));

        let cfg = MultiProbeLshConfig {
            n_tables: 1,
            hash_dim: 2,
            bucket_width: -1.0,
            probe_budget: 1,
        };
        let err = MultiProbeLsh::new(cfg, 4, &mut rng);
        assert!(matches!(err, Err(AnnError::Internal { .. })));
    }

    // 15. err probe_budget=0
    #[test]
    fn mp_lsh_err_budget_zero() {
        let cfg = MultiProbeLshConfig {
            n_tables: 1,
            hash_dim: 2,
            bucket_width: 1.0,
            probe_budget: 0,
        };
        let mut rng = LcgRng::new(0);
        let err = MultiProbeLsh::new(cfg, 4, &mut rng);
        assert!(matches!(err, Err(AnnError::InvalidK { k: 0, .. })));
    }

    // 16. err query dim mismatch
    #[test]
    fn mp_lsh_err_query_dim_mismatch() {
        let idx = small_index(0, 1, 1);
        let q = vec![0.0_f32; 5]; // index dim was 8
        let err = idx.query(&q, 4);
        assert!(matches!(err, Err(AnnError::DimensionMismatch { .. })));
    }

    // 17. err insert dim mismatch
    #[test]
    fn mp_lsh_err_insert_dim_mismatch() {
        let mut idx = small_index(0, 1, 1);
        let v = vec![0.0_f32; 7]; // expected 8
        let err = idx.insert(1, &v);
        assert!(matches!(err, Err(AnnError::DimensionMismatch { .. })));
    }

    // 18. err dim=0 on construction
    #[test]
    fn mp_lsh_err_index_dim_zero() {
        let cfg = MultiProbeLshConfig {
            n_tables: 1,
            hash_dim: 2,
            bucket_width: 1.0,
            probe_budget: 1,
        };
        let mut rng = LcgRng::new(0);
        let err = MultiProbeLsh::new(cfg, 0, &mut rng);
        assert!(matches!(err, Err(AnnError::InvalidVectorDim { dim: 0 })));
    }
}
