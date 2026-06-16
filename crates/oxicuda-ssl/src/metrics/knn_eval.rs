//! kNN classifier evaluation — the standard SSL evaluation protocol.
//!
//! Evaluates representation quality by doing **nearest-neighbour classification
//! on frozen features** without any parameter training.  This protocol is used
//! in essentially every SSL paper: He 2020 (MoCo v2), Chen 2020 (SimCLR), He
//! 2022 (MAE), Zhou 2021 (DINO), and many others.
//!
//! ## Protocol
//!
//! 1. L2-normalise every feature vector to the unit sphere.
//! 2. For each test query, compute cosine similarity with every train sample
//!    (= dot product of unit vectors).
//! 3. Retrieve the top-`k` most similar train samples.
//! 4. **Temperature-scaled softmax voting**: for each class `c`,
//!    `score_c = Σ_{neighbor i: label_i == c} exp(sim(q, kᵢ) / τ)`.
//! 5. Predict `argmax_c score_c`.
//! 6. Report top-1 and top-5 accuracy, as well as per-class top-1 accuracy.
//!
//! ## Memory
//!
//! Processing test samples in chunks of [`KnnEvalConfig::chunk_size`] keeps
//! peak memory to `O(chunk_size × n_train)` rather than `O(n_test × n_train)`.

use crate::error::{SslError, SslResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the kNN evaluation protocol.
#[derive(Debug, Clone)]
pub struct KnnEvalConfig {
    /// Number of nearest neighbours to retrieve.  Typical: 20 (SimCLR/MoCo).
    pub k: usize,
    /// Softmax temperature for voting.  Typical: 0.07.
    pub temperature: f64,
    /// `top_k_acc` — compute top-`k` accuracy (for `top5_accuracy`).
    /// Set to 5 to report top-5; automatically clamped to `n_classes`.
    pub top_k_acc: usize,
    /// Process this many test samples per batch to bound working memory.
    pub chunk_size: usize,
}

impl Default for KnnEvalConfig {
    fn default() -> Self {
        Self {
            k: 20,
            temperature: 0.07,
            top_k_acc: 5,
            chunk_size: 128,
        }
    }
}

// ─── Results ─────────────────────────────────────────────────────────────────

/// Aggregated output of one kNN evaluation run.
#[derive(Debug, Clone)]
pub struct KnnEvalResult {
    /// Fraction of test samples whose top-1 predicted class is correct.
    pub top1_accuracy: f64,
    /// Fraction of test samples whose true label appears among the top-5
    /// highest-scored classes (clamped to `n_classes`).
    pub top5_accuracy: f64,
    /// Per-class top-1 accuracy; `per_class_top1[c]` is `NaN` when class `c`
    /// has no test samples.
    pub per_class_top1: Vec<f64>,
    /// Total number of test samples evaluated.
    pub n_test: usize,
    /// Total number of train samples used as the memory bank.
    pub n_train: usize,
    /// Number of distinct classes.
    pub n_classes: usize,
    /// Number of nearest neighbours `k` actually used.
    pub k: usize,
}

// ─── Public entry point ───────────────────────────────────────────────────────

/// Evaluate kNN classification accuracy on frozen features.
///
/// Implements the standard SSL linear-eval alternative: classify every test
/// sample purely by the `k` closest training neighbours using temperature-
/// scaled softmax voting in representation space, without training any
/// parameters.
///
/// # Arguments
/// * `train_features` — row-major `[n_train × dim]` `f32` feature matrix.
/// * `train_labels`   — `[n_train]` class indices in `0..n_classes`.
/// * `test_features`  — row-major `[n_test × dim]` `f32` feature matrix.
/// * `test_labels`    — `[n_test]` class indices in `0..n_classes`.
/// * `n_train`        — number of training samples.
/// * `n_test`         — number of test samples.
/// * `dim`            — feature dimensionality (must be > 0).
/// * `n_classes`      — number of distinct classes (must be ≥ 2).
/// * `config`         — tunable hyper-parameters (k, τ, chunk size, …).
///
/// # Errors
/// * [`SslError::EmptyInput`] — `n_train == 0`, `n_test == 0`, or `dim == 0`.
/// * [`SslError::InvalidParameter`] — `k > n_train`, `n_classes < 2`,
///   non-positive temperature, or slice length mismatch.
pub fn knn_eval(
    train_features: &[f32],
    train_labels: &[usize],
    test_features: &[f32],
    test_labels: &[usize],
    n_train: usize,
    n_test: usize,
    dim: usize,
    n_classes: usize,
    config: &KnnEvalConfig,
) -> SslResult<KnnEvalResult> {
    // ── Input validation ──────────────────────────────────────────────────────
    if n_train == 0 || dim == 0 {
        return Err(SslError::EmptyInput);
    }
    if n_test == 0 {
        return Err(SslError::EmptyInput);
    }
    if n_classes < 2 {
        return Err(SslError::InvalidParameter {
            name: "n_classes".into(),
            reason: "must be >= 2".into(),
        });
    }
    if config.k == 0 || config.k > n_train {
        return Err(SslError::InvalidParameter {
            name: "k".into(),
            reason: format!("must be in [1, n_train={}]; got {}", n_train, config.k),
        });
    }
    if !config.temperature.is_finite() || config.temperature <= 0.0 {
        return Err(SslError::InvalidParameter {
            name: "temperature".into(),
            reason: "must be finite and > 0".into(),
        });
    }
    if train_features.len() != n_train * dim {
        return Err(SslError::InvalidParameter {
            name: "train_features".into(),
            reason: format!(
                "length {} != n_train({}) * dim({})",
                train_features.len(),
                n_train,
                dim
            ),
        });
    }
    if train_labels.len() != n_train {
        return Err(SslError::InvalidParameter {
            name: "train_labels".into(),
            reason: format!("length {} != n_train({})", train_labels.len(), n_train),
        });
    }
    if test_features.len() != n_test * dim {
        return Err(SslError::InvalidParameter {
            name: "test_features".into(),
            reason: format!(
                "length {} != n_test({}) * dim({})",
                test_features.len(),
                n_test,
                dim
            ),
        });
    }
    if test_labels.len() != n_test {
        return Err(SslError::InvalidParameter {
            name: "test_labels".into(),
            reason: format!("length {} != n_test({})", test_labels.len(), n_test),
        });
    }

    // Validate label ranges.
    for (idx, &lbl) in train_labels.iter().enumerate() {
        if lbl >= n_classes {
            return Err(SslError::InvalidParameter {
                name: "train_labels".into(),
                reason: format!("label {} at index {} >= n_classes {}", lbl, idx, n_classes),
            });
        }
    }
    for (idx, &lbl) in test_labels.iter().enumerate() {
        if lbl >= n_classes {
            return Err(SslError::InvalidParameter {
                name: "test_labels".into(),
                reason: format!("label {} at index {} >= n_classes {}", lbl, idx, n_classes),
            });
        }
    }

    // ── Normalise the entire training bank once ───────────────────────────────
    let mut train_norm = train_features.to_vec();
    l2_normalize_inplace(&mut train_norm, n_train, dim);

    // ── Chunk-based test evaluation ───────────────────────────────────────────
    let chunk_size = config.chunk_size.max(1);
    let effective_top_k = config.top_k_acc.min(n_classes);
    let k = config.k;

    let mut top1_correct: usize = 0;
    let mut top5_correct: usize = 0;
    // Per-class: (correct count, total count).
    let mut per_class_correct = vec![0_usize; n_classes];
    let mut per_class_total = vec![0_usize; n_classes];

    let mut chunk_start = 0;
    while chunk_start < n_test {
        let chunk_end = (chunk_start + chunk_size).min(n_test);
        let chunk_n = chunk_end - chunk_start;

        // Copy and normalise this chunk of test features.
        let chunk_feat_start = chunk_start * dim;
        let chunk_feat_end = chunk_end * dim;
        let mut chunk_feats = test_features[chunk_feat_start..chunk_feat_end].to_vec();
        l2_normalize_inplace(&mut chunk_feats, chunk_n, dim);

        // For each test sample in the chunk, compute full similarity vector,
        // find top-k neighbours, vote, and accumulate accuracy.
        for local_idx in 0..chunk_n {
            let global_idx = chunk_start + local_idx;
            let query = &chunk_feats[local_idx * dim..(local_idx + 1) * dim];
            let true_label = test_labels[global_idx];

            // Cosine similarities against all training samples.
            let sims = cosine_similarities(query, &train_norm, n_train, dim);

            // Retrieve top-k neighbour indices.
            let top_indices = top_k_indices(&sims, k);

            // Gather neighbour similarities and labels.
            let neighbor_sims: Vec<f32> = top_indices.iter().map(|&i| sims[i]).collect();
            let neighbor_labels: Vec<usize> =
                top_indices.iter().map(|&i| train_labels[i]).collect();

            // Temperature-scaled softmax voting.
            let scores = temperature_vote(
                &neighbor_sims,
                &neighbor_labels,
                k,
                n_classes,
                config.temperature,
            );

            // Top-1 prediction.
            let pred_top1 = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            if pred_top1 == true_label {
                top1_correct += 1;
            }

            // Top-K (top-5) accuracy.
            let top_classes = top_k_classes(&scores, effective_top_k);
            if top_classes.contains(&true_label) {
                top5_correct += 1;
            }

            // Per-class accumulation.
            per_class_total[true_label] += 1;
            if pred_top1 == true_label {
                per_class_correct[true_label] += 1;
            }
        }

        chunk_start = chunk_end;
    }

    // ── Aggregate ─────────────────────────────────────────────────────────────
    let top1_accuracy = top1_correct as f64 / n_test as f64;
    let top5_accuracy = top5_correct as f64 / n_test as f64;
    let per_class_top1: Vec<f64> = (0..n_classes)
        .map(|c| {
            if per_class_total[c] == 0 {
                f64::NAN
            } else {
                per_class_correct[c] as f64 / per_class_total[c] as f64
            }
        })
        .collect();

    Ok(KnnEvalResult {
        top1_accuracy,
        top5_accuracy,
        per_class_top1,
        n_test,
        n_train,
        n_classes,
        k,
    })
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// L2-normalise each row of `features` (row-major, `n` rows × `dim` cols)
/// in-place.  Rows with near-zero norm are left as zero vectors.
fn l2_normalize_inplace(features: &mut [f32], n: usize, dim: usize) {
    for i in 0..n {
        let row = &mut features[i * dim..(i + 1) * dim];
        let norm_sq: f32 = row.iter().map(|v| v * v).sum();
        let norm = norm_sq.sqrt();
        // Clamp to 1e-12 to avoid divide-by-zero or catastrophic amplification.
        if norm >= 1e-12 {
            let inv = 1.0 / norm;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }
}

/// Compute cosine similarity between a single unit-norm `query` (length `dim`)
/// and every row of `database` (already unit-norm, shape `[n_db × dim]`).
///
/// Returns `Vec<f32>` of length `n_db` where `result[j] = Σᵢ query[i]*db[j*dim+i]`.
fn cosine_similarities(query: &[f32], database: &[f32], n_db: usize, dim: usize) -> Vec<f32> {
    let mut sims = vec![0.0_f32; n_db];
    for j in 0..n_db {
        let db_row = &database[j * dim..(j + 1) * dim];
        let mut dot = 0.0_f32;
        // Manual loop — LLVM auto-vectorises this well without SIMD intrinsics.
        for (q, d) in query.iter().zip(db_row.iter()) {
            dot += q * d;
        }
        sims[j] = dot;
    }
    sims
}

/// Return the indices of the `k` largest elements in `sims` (ties broken by
/// index, smallest index wins).  Uses O(n·k) partial insertion sort.
///
/// # Panics
/// Panics in debug mode if `k == 0`.  Production callers always pass `k ≥ 1`.
fn top_k_indices(sims: &[f32], k: usize) -> Vec<usize> {
    debug_assert!(k > 0, "top_k_indices called with k=0");
    let n = sims.len();
    let k = k.min(n);

    // Maintain a buffer of (sim, index) for the current top-k.
    // We use insertion into a sorted buffer (ascending by sim) so the front is
    // the current minimum among top-k candidates.
    let mut top: Vec<(f32, usize)> = Vec::with_capacity(k + 1);

    for (i, &s) in sims.iter().enumerate() {
        if top.len() < k {
            // Buffer not full yet — insert in sorted order (ascending).
            let pos = top.partition_point(|(v, _)| *v < s);
            top.insert(pos, (s, i));
        } else if s > top[0].0 {
            // Better than the current minimum — replace and re-sort.
            top[0] = (s, i);
            // Bubble-up: find the correct position for top[0].
            let mut p = 0;
            while p + 1 < k && top[p].0 > top[p + 1].0 {
                top.swap(p, p + 1);
                p += 1;
            }
        }
    }

    // Return indices in descending order of similarity.
    top.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    top.iter().map(|&(_, idx)| idx).collect()
}

/// Temperature-scaled softmax voting over `k` neighbours.
///
/// For each class `c`, accumulates `Σ_{i: label_i == c} exp(sim_i / τ)`.
///
/// Returns a `Vec<f64>` of length `n_classes`.  Classes with zero neighbour
/// votes receive score `0.0`.
fn temperature_vote(
    neighbor_sims: &[f32],
    neighbor_labels: &[usize],
    k: usize,
    n_classes: usize,
    temperature: f64,
) -> Vec<f64> {
    let mut scores = vec![0.0_f64; n_classes];
    let inv_temp = 1.0 / temperature;
    let count = k.min(neighbor_sims.len());
    for i in 0..count {
        let sim = neighbor_sims[i] as f64;
        let lbl = neighbor_labels[i];
        scores[lbl] += (sim * inv_temp).exp();
    }
    scores
}

/// Return the indices of the `k` highest-scored classes in descending order.
/// Handles `k > n_classes` gracefully by clamping.
fn top_k_classes(scores: &[f64], k: usize) -> Vec<usize> {
    let n = scores.len();
    let k = k.min(n);
    // Build index list, sort descending by score.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(k);
    indices
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Deterministic LCG for generating test data without the `rand` crate.
    struct TestLcg {
        state: u64,
    }

    impl TestLcg {
        fn new(seed: u64) -> Self {
            Self {
                state: seed.wrapping_add(1),
            }
        }

        fn next_f32(&mut self) -> f32 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.state >> 33) as f32 / (u32::MAX as f32)
        }

        fn fill(&mut self, buf: &mut [f32]) {
            for v in buf.iter_mut() {
                *v = self.next_f32() * 2.0 - 1.0;
            }
        }
    }

    // ── 1. config_defaults ────────────────────────────────────────────────────

    #[test]
    fn config_defaults() {
        let cfg = KnnEvalConfig::default();
        assert_eq!(cfg.k, 20);
        assert!((cfg.temperature - 0.07).abs() < 1e-10);
        assert_eq!(cfg.top_k_acc, 5);
        assert_eq!(cfg.chunk_size, 128);
    }

    // ── 2. l2_normalize_unit_length ───────────────────────────────────────────

    #[test]
    fn l2_normalize_unit_length() {
        let n = 8_usize;
        let dim = 16_usize;
        let mut rng = TestLcg::new(42);
        let mut feats = vec![0.0_f32; n * dim];
        rng.fill(&mut feats);

        l2_normalize_inplace(&mut feats, n, dim);

        for i in 0..n {
            let row = &feats[i * dim..(i + 1) * dim];
            let norm_sq: f32 = row.iter().map(|v| v * v).sum();
            assert!(
                (norm_sq.sqrt() - 1.0).abs() < 1e-5,
                "row {i} norm = {} (expected 1.0)",
                norm_sq.sqrt()
            );
        }
    }

    // ── 3. perfect_classification ─────────────────────────────────────────────

    #[test]
    fn perfect_classification() {
        // Identical train and test features → every test sample is its own
        // nearest neighbour → top-1 accuracy must be 1.0.
        let n_classes = 5_usize;
        let dim = 32_usize;
        let samples_per_class = 4_usize;
        let n_train = n_classes * samples_per_class;

        let mut rng = TestLcg::new(1337);

        // Create well-separated class prototypes.
        let mut train_feats = vec![0.0_f32; n_train * dim];
        let mut train_labels = vec![0_usize; n_train];
        for c in 0..n_classes {
            for s in 0..samples_per_class {
                let idx = c * samples_per_class + s;
                train_labels[idx] = c;
                // Each class in a different "direction" with small noise.
                train_feats[idx * dim + c % dim] = 1.0;
                for d in 0..dim {
                    train_feats[idx * dim + d] += rng.next_f32() * 0.01;
                }
            }
        }

        let test_feats = train_feats.clone();
        let test_labels = train_labels.clone();

        let cfg = KnnEvalConfig {
            k: 1,
            ..Default::default()
        };

        let result = knn_eval(
            &train_feats,
            &train_labels,
            &test_feats,
            &test_labels,
            n_train,
            n_train,
            dim,
            n_classes,
            &cfg,
        )
        .expect("value should be present");

        assert_eq!(
            result.top1_accuracy, 1.0,
            "perfect scenario must have top1=1.0, got {}",
            result.top1_accuracy
        );
    }

    // ── 4. random_features_accuracy_range ────────────────────────────────────

    #[test]
    fn random_features_accuracy_range() {
        let n_classes = 4_usize;
        let dim = 64_usize;
        let n_train = 100_usize;
        let n_test = 50_usize;

        let mut rng = TestLcg::new(9999);
        let mut train_feats = vec![0.0_f32; n_train * dim];
        rng.fill(&mut train_feats);
        let mut test_feats = vec![0.0_f32; n_test * dim];
        rng.fill(&mut test_feats);

        let train_labels: Vec<usize> = (0..n_train).map(|i| i % n_classes).collect();
        let test_labels: Vec<usize> = (0..n_test).map(|i| i % n_classes).collect();

        let cfg = KnnEvalConfig {
            k: 5,
            ..Default::default()
        };

        let result = knn_eval(
            &train_feats,
            &train_labels,
            &test_feats,
            &test_labels,
            n_train,
            n_test,
            dim,
            n_classes,
            &cfg,
        )
        .expect("value should be present");

        assert!(
            (0.0..=1.0).contains(&result.top1_accuracy),
            "top1_accuracy={} not in [0,1]",
            result.top1_accuracy
        );
        assert!(
            (0.0..=1.0).contains(&result.top5_accuracy),
            "top5_accuracy={} not in [0,1]",
            result.top5_accuracy
        );
    }

    // ── 5. top5_geq_top1 ──────────────────────────────────────────────────────

    #[test]
    fn top5_geq_top1() {
        let n_classes = 10_usize;
        let dim = 32_usize;
        let n_train = 200_usize;
        let n_test = 50_usize;

        let mut rng = TestLcg::new(7654);
        let mut train_feats = vec![0.0_f32; n_train * dim];
        rng.fill(&mut train_feats);
        let mut test_feats = vec![0.0_f32; n_test * dim];
        rng.fill(&mut test_feats);

        let train_labels: Vec<usize> = (0..n_train).map(|i| i % n_classes).collect();
        let test_labels: Vec<usize> = (0..n_test).map(|i| i % n_classes).collect();

        let cfg = KnnEvalConfig {
            k: 10,
            ..Default::default()
        };

        let result = knn_eval(
            &train_feats,
            &train_labels,
            &test_feats,
            &test_labels,
            n_train,
            n_test,
            dim,
            n_classes,
            &cfg,
        )
        .expect("value should be present");

        assert!(
            result.top5_accuracy >= result.top1_accuracy - 1e-10,
            "top5={} < top1={}",
            result.top5_accuracy,
            result.top1_accuracy
        );
    }

    // ── 6. output_n_test_correct ──────────────────────────────────────────────

    #[test]
    fn output_n_test_correct() {
        let n_train = 20_usize;
        let n_test = 15_usize;
        let dim = 8_usize;
        let n_classes = 2_usize;

        let train_feats: Vec<f32> = (0..n_train * dim).map(|i| (i as f32) * 0.01).collect();
        let train_labels: Vec<usize> = (0..n_train).map(|i| i % n_classes).collect();
        let test_feats: Vec<f32> = (0..n_test * dim).map(|i| (i as f32) * 0.02).collect();
        let test_labels: Vec<usize> = (0..n_test).map(|i| i % n_classes).collect();

        let cfg = KnnEvalConfig {
            k: 3,
            ..Default::default()
        };

        let result = knn_eval(
            &train_feats,
            &train_labels,
            &test_feats,
            &test_labels,
            n_train,
            n_test,
            dim,
            n_classes,
            &cfg,
        )
        .expect("value should be present");

        assert_eq!(result.n_test, n_test);
        assert_eq!(result.n_train, n_train);
    }

    // ── 7. output_k_correct ───────────────────────────────────────────────────

    #[test]
    fn output_k_correct() {
        let n_train = 30_usize;
        let n_test = 10_usize;
        let dim = 8_usize;
        let n_classes = 3_usize;

        let train_feats: Vec<f32> = (0..n_train * dim).map(|i| (i as f32) * 0.01).collect();
        let train_labels: Vec<usize> = (0..n_train).map(|i| i % n_classes).collect();
        let test_feats: Vec<f32> = (0..n_test * dim).map(|i| (i as f32) * 0.015).collect();
        let test_labels: Vec<usize> = (0..n_test).map(|i| i % n_classes).collect();

        let cfg = KnnEvalConfig {
            k: 7,
            ..Default::default()
        };

        let result = knn_eval(
            &train_feats,
            &train_labels,
            &test_feats,
            &test_labels,
            n_train,
            n_test,
            dim,
            n_classes,
            &cfg,
        )
        .expect("value should be present");

        assert_eq!(result.k, 7);
    }

    // ── 8. per_class_length ───────────────────────────────────────────────────

    #[test]
    fn per_class_length() {
        let n_classes = 6_usize;
        let dim = 8_usize;
        let n_train = 24_usize;
        let n_test = 12_usize;

        let train_feats: Vec<f32> = (0..n_train * dim).map(|i| (i as f32) * 0.01).collect();
        let train_labels: Vec<usize> = (0..n_train).map(|i| i % n_classes).collect();
        let test_feats: Vec<f32> = (0..n_test * dim).map(|i| (i as f32) * 0.02).collect();
        let test_labels: Vec<usize> = (0..n_test).map(|i| i % n_classes).collect();

        let cfg = KnnEvalConfig {
            k: 3,
            ..Default::default()
        };

        let result = knn_eval(
            &train_feats,
            &train_labels,
            &test_feats,
            &test_labels,
            n_train,
            n_test,
            dim,
            n_classes,
            &cfg,
        )
        .expect("value should be present");

        assert_eq!(result.per_class_top1.len(), n_classes);
    }

    // ── 9. cosine_sim_self_is_one ─────────────────────────────────────────────

    #[test]
    fn cosine_sim_self_is_one() {
        let dim = 16_usize;
        let mut rng = TestLcg::new(111);
        let mut v = vec![0.0_f32; dim];
        rng.fill(&mut v);
        // Normalise manually.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in v.iter_mut() {
            *x /= norm;
        }
        // Dot with itself must be 1.
        let sims = cosine_similarities(&v, &v, 1, dim);
        assert!(
            (sims[0] - 1.0).abs() < 1e-5,
            "self cosine = {} (expected 1.0)",
            sims[0]
        );
    }

    // ── 10. top_k_indices_correct ─────────────────────────────────────────────

    #[test]
    fn top_k_indices_correct() {
        // sims = [3, 1, 4, 1, 5], top-2 should be indices 4 (sim=5) and 2 (sim=4).
        let sims = vec![3.0_f32, 1.0, 4.0, 1.0, 5.0];
        let top2 = top_k_indices(&sims, 2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0], 4, "first index should be 4 (sim=5)");
        assert_eq!(top2[1], 2, "second index should be 2 (sim=4)");
    }

    // ── 11. temperature_vote_correct_label ────────────────────────────────────

    #[test]
    fn temperature_vote_correct_label() {
        // Single neighbour of class 2 with sim=0.9; argmax should be 2.
        let neighbor_sims = vec![0.9_f32];
        let neighbor_labels = vec![2_usize];
        let n_classes = 5_usize;
        let scores = temperature_vote(&neighbor_sims, &neighbor_labels, 1, n_classes, 0.07);
        let pred = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("partial_cmp should succeed"))
            .map(|(i, _)| i)
            .expect("value should be present");
        assert_eq!(pred, 2, "predicted class should be 2, got {pred}");
        // All other classes must have score 0.
        for (c, &s) in scores.iter().enumerate() {
            if c != 2 {
                assert_eq!(s, 0.0, "class {c} should have score 0.0, got {s}");
            }
        }
    }

    // ── 12. empty_test_error ──────────────────────────────────────────────────

    #[test]
    fn empty_test_error() {
        let train_feats = vec![1.0_f32, 0.0, 0.0, 1.0];
        let train_labels = vec![0_usize, 1];
        let cfg = KnnEvalConfig {
            k: 1,
            ..Default::default()
        };

        let result = knn_eval(
            &train_feats,
            &train_labels,
            &[],
            &[],
            2,
            0, // n_test = 0
            2,
            2,
            &cfg,
        );
        assert!(
            matches!(result, Err(SslError::EmptyInput)),
            "expected EmptyInput, got {result:?}"
        );
    }

    // ── 13. k_larger_than_train_error ────────────────────────────────────────

    #[test]
    fn k_larger_than_train_error() {
        let train_feats = vec![1.0_f32, 0.0, 0.0, 1.0];
        let train_labels = vec![0_usize, 1];
        let test_feats = vec![1.0_f32, 0.0];
        let test_labels = vec![0_usize];

        let cfg = KnnEvalConfig {
            k: 10, // k > n_train = 2
            ..Default::default()
        };

        let result = knn_eval(
            &train_feats,
            &train_labels,
            &test_feats,
            &test_labels,
            2,
            1,
            2,
            2,
            &cfg,
        );
        assert!(
            matches!(result, Err(SslError::InvalidParameter { .. })),
            "expected InvalidParameter for k > n_train, got {result:?}"
        );
    }

    // ── 14. binary_classification ─────────────────────────────────────────────

    #[test]
    fn binary_classification() {
        // Two linearly separable classes in R^4.
        // Class 0 → all features near +e1, class 1 → all features near -e1.
        let n_classes = 2_usize;
        let dim = 4_usize;
        let n_train = 20_usize;
        let n_test = 10_usize;

        let mut rng = TestLcg::new(31415);

        let mut train_feats = vec![0.0_f32; n_train * dim];
        let mut train_labels = vec![0_usize; n_train];
        for i in 0..n_train {
            train_labels[i] = i % 2;
            let sign = if i % 2 == 0 { 1.0_f32 } else { -1.0_f32 };
            train_feats[i * dim] = sign * (1.0 + rng.next_f32() * 0.05);
            for d in 1..dim {
                train_feats[i * dim + d] = rng.next_f32() * 0.01;
            }
        }

        let mut test_feats = vec![0.0_f32; n_test * dim];
        let mut test_labels = vec![0_usize; n_test];
        for i in 0..n_test {
            test_labels[i] = i % 2;
            let sign = if i % 2 == 0 { 1.0_f32 } else { -1.0_f32 };
            test_feats[i * dim] = sign * (1.0 + rng.next_f32() * 0.05);
            for d in 1..dim {
                test_feats[i * dim + d] = rng.next_f32() * 0.01;
            }
        }

        let cfg = KnnEvalConfig {
            k: 5,
            temperature: 0.07,
            top_k_acc: 2, // top-2 for binary
            chunk_size: 4,
        };

        let result = knn_eval(
            &train_feats,
            &train_labels,
            &test_feats,
            &test_labels,
            n_train,
            n_test,
            dim,
            n_classes,
            &cfg,
        )
        .expect("value should be present");

        assert_eq!(
            result.top1_accuracy, 1.0,
            "perfectly separated binary classes must achieve top1=1.0, got {}",
            result.top1_accuracy
        );
        assert_eq!(result.n_classes, n_classes);
        assert_eq!(result.per_class_top1.len(), n_classes);
    }
}
