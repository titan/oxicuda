//! Continual-learning evaluation scenario generators.
//!
//! Pure-Rust synthetic constructors for the canonical continual-learning
//! benchmark *scenarios* used to evaluate forgetting-mitigation methods:
//!
//! - **Permuted MNIST** (Goodfellow 2013 / Kirkpatrick 2017): each task applies
//!   a fixed random permutation of the input pixels. The label space is shared,
//!   so this is a *domain-incremental* scenario (input distribution shifts, task
//!   identity is the permutation index).
//! - **Split MNIST / Split CIFAR** (Zenke 2017 / Lopez-Paz 2017): the global
//!   class set is partitioned into disjoint groups, one group per task. This is
//!   the classic *class-incremental* (or task-incremental when the head is
//!   given) scenario.
//! - **Rotated MNIST / CORe50-style** (Lomonaco 2017): each task rotates the
//!   input in a 2-D feature plane by a fixed angle, modelling a gradually
//!   shifting domain (a continuous *domain-incremental* stream).
//!
//! These generators produce synthetic feature vectors so the harness is fully
//! deterministic and dependency-free (driven by [`LcgRng`]); the structural
//! transformations (pixel permutation, class split, planar rotation) are exactly
//! the ones the real benchmarks apply, so a method's relative behaviour across
//! tasks is faithfully reproduced for unit-level evaluation.
//!
//! The output is a [`TaskStream`] (shared / per-task label spaces) or a
//! [`ClassIncStream`] (disjoint label spaces) that plugs directly into the rest
//! of the crate (EWC, replay, metrics, …).

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;
use crate::stream::class_stream::{ClassIncStream, class_inc_new};
use crate::stream::task_stream::{Task, TaskStream, task_stream_new};

/// Configuration shared by all scenario generators.
#[derive(Debug, Clone)]
pub struct ScenarioConfig {
    /// Number of tasks in the generated stream. Must be `>= 1`.
    pub n_tasks: usize,
    /// Dimensionality of each synthetic feature vector. Must be `>= 1`.
    pub feature_dim: usize,
    /// Number of samples generated per task. Must be `>= 1`.
    pub samples_per_task: usize,
    /// Total number of distinct classes across the whole benchmark.
    /// Must be `>= 2`. For split scenarios this is partitioned across tasks.
    pub n_classes: usize,
    /// Seed for the deterministic generator.
    pub seed: u64,
}

impl Default for ScenarioConfig {
    fn default() -> Self {
        Self {
            n_tasks: 5,
            feature_dim: 16,
            samples_per_task: 64,
            n_classes: 10,
            seed: 42,
        }
    }
}

impl ScenarioConfig {
    /// Create and validate a scenario configuration.
    pub fn new(
        n_tasks: usize,
        feature_dim: usize,
        samples_per_task: usize,
        n_classes: usize,
        seed: u64,
    ) -> ContinualResult<Self> {
        let cfg = Self {
            n_tasks,
            feature_dim,
            samples_per_task,
            n_classes,
            seed,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate the configuration fields.
    pub fn validate(&self) -> ContinualResult<()> {
        if self.n_tasks == 0 {
            return Err(ContinualError::NoTasksInStream);
        }
        if self.feature_dim == 0 || self.samples_per_task == 0 {
            return Err(ContinualError::EmptyInput);
        }
        if self.n_classes < 2 {
            return Err(ContinualError::Internal(
                "n_classes must be >= 2".to_string(),
            ));
        }
        Ok(())
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Draw a single deterministic class-conditional prototype feature.
///
/// Each class owns a fixed prototype (a deterministic vector). A sample of that
/// class is the prototype plus small Gaussian jitter, so classes are linearly
/// separable enough that retention / forgetting effects are measurable.
fn class_prototype(class: usize, dim: usize) -> Vec<f32> {
    // Deterministic, class-specific prototype on the unit hypercube corners
    // perturbed by a class-indexed frequency pattern.
    let mut proto = vec![0.0_f32; dim];
    for (i, p) in proto.iter_mut().enumerate() {
        // Distinct, bounded per-(class, feature) value.
        let phase = (class as f32 + 1.0) * (i as f32 + 1.0) * 0.6180339887_f32;
        *p = (phase.sin() + (class as f32 * 0.37).cos()) * 0.5;
    }
    proto
}

/// Generate `n` raw `(feature, label)` samples drawn from `classes`.
fn raw_samples(
    classes: &[usize],
    n: usize,
    dim: usize,
    rng: &mut LcgRng,
) -> Vec<(Vec<f32>, usize)> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let class = classes[rng.next_usize(classes.len())];
        let proto = class_prototype(class, dim);
        let mut feat = vec![0.0_f32; dim];
        let mut jitter = vec![0.0_f32; dim];
        rng.fill_normal(&mut jitter);
        for j in 0..dim {
            feat[j] = proto[j] + 0.1 * jitter[j];
        }
        out.push((feat, class));
    }
    out
}

/// Build a random permutation of `0..dim` via Fisher-Yates.
fn random_permutation(dim: usize, rng: &mut LcgRng) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..dim).collect();
    rng.shuffle(&mut perm);
    perm
}

/// Apply a permutation of feature indices in-place into a fresh vector.
fn permute_features(feat: &[f32], perm: &[usize]) -> Vec<f32> {
    perm.iter().map(|&p| feat[p]).collect()
}

/// Rotate a feature vector by `angle` radians in successive 2-D coordinate
/// planes `(0,1), (2,3), …`. A trailing odd coordinate is left unchanged.
fn rotate_features(feat: &[f32], angle: f32) -> Vec<f32> {
    let c = angle.cos();
    let s = angle.sin();
    let mut out = feat.to_vec();
    let mut i = 0;
    while i + 1 < out.len() {
        let x = out[i];
        let y = out[i + 1];
        out[i] = c * x - s * y;
        out[i + 1] = s * x + c * y;
        i += 2;
    }
    out
}

// ─── Permuted MNIST (domain-incremental) ──────────────────────────────────────

/// A scenario stream paired with any per-task transformation metadata that is
/// useful for evaluation (e.g. the pixel permutation that defines each task).
#[derive(Debug, Clone)]
pub struct PermutedScenario {
    /// The generated task-incremental stream (shared label space).
    pub stream: TaskStream,
    /// Per-task feature permutation (`permutations[t][j]` is the source index
    /// mapped to output position `j` for task `t`). Task 0 is the identity.
    pub permutations: Vec<Vec<usize>>,
}

/// Generate a **Permuted MNIST**-style domain-incremental scenario.
///
/// All tasks share the same `n_classes` label space; task `t > 0` applies a
/// fixed random permutation of the `feature_dim` input features (task 0 uses the
/// identity permutation as the reference domain). The resulting stream is ideal
/// for measuring catastrophic forgetting under input-distribution shift, where
/// the model output head is shared across tasks.
pub fn permuted_mnist(cfg: &ScenarioConfig) -> ContinualResult<PermutedScenario> {
    cfg.validate()?;
    let mut rng = LcgRng::new(cfg.seed);
    let all_classes: Vec<usize> = (0..cfg.n_classes).collect();

    let mut permutations: Vec<Vec<usize>> = Vec::with_capacity(cfg.n_tasks);
    let mut tasks: Vec<Task> = Vec::with_capacity(cfg.n_tasks);

    for t in 0..cfg.n_tasks {
        let perm = if t == 0 {
            (0..cfg.feature_dim).collect::<Vec<_>>()
        } else {
            random_permutation(cfg.feature_dim, &mut rng)
        };
        let raw = raw_samples(
            &all_classes,
            cfg.samples_per_task,
            cfg.feature_dim,
            &mut rng,
        );
        let data: Vec<(Vec<f32>, u32)> = raw
            .into_iter()
            .map(|(feat, label)| (permute_features(&feat, &perm), label as u32))
            .collect();
        tasks.push(Task::new(t, cfg.n_classes, data)?);
        permutations.push(perm);
    }

    let stream = task_stream_new(tasks)?;
    Ok(PermutedScenario {
        stream,
        permutations,
    })
}

// ─── Rotated MNIST / CORe50-style (domain-incremental) ─────────────────────────

/// A rotated-domain scenario with its per-task rotation angles.
#[derive(Debug, Clone)]
pub struct RotatedScenario {
    /// The generated task-incremental stream (shared label space).
    pub stream: TaskStream,
    /// Rotation angle (radians) applied to each task's features.
    pub angles: Vec<f32>,
}

/// Generate a **Rotated MNIST / CORe50**-style domain-incremental scenario.
///
/// Task `t` rotates the input features by `t * (max_angle / (n_tasks - 1))`
/// radians (so task 0 is the unrotated reference and the final task reaches
/// `max_angle`). This models a smoothly drifting domain — the continual analogue
/// of CORe50's object-pose sweep — while keeping the label space shared.
///
/// `max_angle` must be finite; angles are applied in successive 2-D coordinate
/// planes of the feature vector.
pub fn rotated_mnist(cfg: &ScenarioConfig, max_angle: f32) -> ContinualResult<RotatedScenario> {
    cfg.validate()?;
    if !max_angle.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "rotated_mnist:max_angle",
        });
    }
    let mut rng = LcgRng::new(cfg.seed);
    let all_classes: Vec<usize> = (0..cfg.n_classes).collect();

    let step = if cfg.n_tasks > 1 {
        max_angle / (cfg.n_tasks as f32 - 1.0)
    } else {
        0.0
    };

    let mut angles = Vec::with_capacity(cfg.n_tasks);
    let mut tasks = Vec::with_capacity(cfg.n_tasks);

    for t in 0..cfg.n_tasks {
        let angle = step * t as f32;
        let raw = raw_samples(
            &all_classes,
            cfg.samples_per_task,
            cfg.feature_dim,
            &mut rng,
        );
        let data: Vec<(Vec<f32>, u32)> = raw
            .into_iter()
            .map(|(feat, label)| (rotate_features(&feat, angle), label as u32))
            .collect();
        tasks.push(Task::new(t, cfg.n_classes, data)?);
        angles.push(angle);
    }

    let stream = task_stream_new(tasks)?;
    Ok(RotatedScenario { stream, angles })
}

// ─── Split MNIST / Split CIFAR (class-incremental) ─────────────────────────────

/// A class-split scenario paired with the per-task class partition.
#[derive(Debug, Clone)]
pub struct SplitScenario {
    /// The generated class-incremental stream (disjoint label spaces).
    pub stream: ClassIncStream,
    /// `class_groups[t]` lists the (globally-unique) classes introduced in
    /// task `t`.
    pub class_groups: Vec<Vec<usize>>,
}

/// Generate a **Split MNIST / Split CIFAR**-style class-incremental scenario.
///
/// The global `n_classes` are partitioned into `n_tasks` disjoint contiguous
/// groups (e.g. 10 classes / 5 tasks → `{0,1} {2,3} {4,5} {6,7} {8,9}`). Each
/// task draws samples only from its own classes, producing the canonical
/// disjoint-label class-incremental stream where the classifier must
/// distinguish all classes seen so far without a task oracle.
///
/// Requires `n_classes >= n_tasks` so every task receives at least one class.
pub fn split_classes(cfg: &ScenarioConfig) -> ContinualResult<SplitScenario> {
    cfg.validate()?;
    if cfg.n_classes < cfg.n_tasks {
        return Err(ContinualError::Internal(format!(
            "split scenario requires n_classes ({}) >= n_tasks ({})",
            cfg.n_classes, cfg.n_tasks
        )));
    }
    let mut rng = LcgRng::new(cfg.seed);

    // Partition classes into n_tasks contiguous groups, distributing the
    // remainder across the earliest tasks (balanced split).
    let base = cfg.n_classes / cfg.n_tasks;
    let rem = cfg.n_classes % cfg.n_tasks;
    let mut class_groups: Vec<Vec<usize>> = Vec::with_capacity(cfg.n_tasks);
    let mut next_class = 0usize;
    for t in 0..cfg.n_tasks {
        let count = base + usize::from(t < rem);
        let group: Vec<usize> = (next_class..next_class + count).collect();
        next_class += count;
        class_groups.push(group);
    }

    let mut tasks = Vec::with_capacity(cfg.n_tasks);
    for (t, group) in class_groups.iter().enumerate() {
        let raw = raw_samples(group, cfg.samples_per_task, cfg.feature_dim, &mut rng);
        let data: Vec<(Vec<f32>, u32)> = raw
            .into_iter()
            .map(|(feat, label)| (feat, label as u32))
            .collect();
        // n_classes is the global label-space size (labels are global ids).
        tasks.push(Task::new(t, cfg.n_classes, data)?);
    }

    let stream = class_inc_new(tasks)?;
    Ok(SplitScenario {
        stream,
        class_groups,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::class_stream::{init_class_inc, n_classes_seen};

    // ── Permuted MNIST ────────────────────────────────────────────────────────

    #[test]
    fn permuted_mnist_shapes_and_count() {
        let cfg = ScenarioConfig::new(4, 12, 32, 6, 7).expect("valid scenario config");
        let scn = permuted_mnist(&cfg).expect("permuted scenario should generate");
        assert_eq!(scn.stream.tasks.len(), 4);
        assert_eq!(scn.permutations.len(), 4);
        for task in &scn.stream.tasks {
            assert_eq!(task.data.len(), 32);
            assert_eq!(task.n_classes, 6);
            for (feat, label) in &task.data {
                assert_eq!(feat.len(), 12);
                assert!((*label as usize) < 6);
                assert!(feat.iter().all(|v| v.is_finite()));
            }
        }
    }

    #[test]
    fn permuted_mnist_task0_is_identity() {
        let cfg = ScenarioConfig::new(3, 8, 16, 4, 1).expect("valid scenario config");
        let scn = permuted_mnist(&cfg).expect("permuted scenario should generate");
        assert_eq!(scn.permutations[0], (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn permuted_mnist_permutation_is_a_bijection() {
        let cfg = ScenarioConfig::new(5, 20, 8, 4, 99).expect("valid scenario config");
        let scn = permuted_mnist(&cfg).expect("permuted scenario should generate");
        for perm in &scn.permutations {
            let mut sorted = perm.clone();
            sorted.sort_unstable();
            assert_eq!(
                sorted,
                (0..20).collect::<Vec<_>>(),
                "each permutation must be a bijection of 0..feature_dim"
            );
        }
        // A later task must actually permute (extremely unlikely to be identity
        // for dim=20 under a real shuffle).
        assert_ne!(scn.permutations[4], (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn permuted_mnist_preserves_multiset_of_features() {
        // Permuting features must conserve the multiset of values within a
        // sample (it is a pure reindexing), which we verify by comparing sorted
        // feature values of the same logical sample under identity vs permuted.
        let cfg = ScenarioConfig::new(2, 10, 4, 3, 2024).expect("valid scenario config");
        let scn = permuted_mnist(&cfg).expect("permuted scenario should generate");
        // Re-derive task-1 raw features with the same RNG stream is complex;
        // instead verify the permutation applied to an arbitrary vector keeps
        // its sorted contents.
        let v: Vec<f32> = (0..10).map(|i| i as f32 * 0.5).collect();
        let permuted = permute_features(&v, &scn.permutations[1]);
        let mut a = v.clone();
        let mut b = permuted.clone();
        a.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
        b.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
        assert_eq!(a, b);
    }

    #[test]
    fn permuted_mnist_deterministic() {
        let cfg = ScenarioConfig::new(3, 16, 20, 5, 555).expect("valid scenario config");
        let a = permuted_mnist(&cfg).expect("permuted scenario should generate");
        let b = permuted_mnist(&cfg).expect("permuted scenario should generate");
        assert_eq!(a.permutations, b.permutations);
        for (ta, tb) in a.stream.tasks.iter().zip(b.stream.tasks.iter()) {
            assert_eq!(ta.data, tb.data);
        }
    }

    // ── Rotated MNIST ─────────────────────────────────────────────────────────

    #[test]
    fn rotated_mnist_angles_monotone_and_bounded() {
        let cfg = ScenarioConfig::new(5, 8, 16, 4, 3).expect("valid scenario config");
        let max = std::f32::consts::FRAC_PI_2;
        let scn = rotated_mnist(&cfg, max).expect("rotated scenario should generate");
        assert_eq!(scn.angles.len(), 5);
        assert!((scn.angles[0]).abs() < 1e-7, "task 0 must be unrotated");
        assert!(
            (scn.angles[4] - max).abs() < 1e-5,
            "final task must reach max_angle"
        );
        for w in scn.angles.windows(2) {
            assert!(w[1] >= w[0], "angles must be non-decreasing");
        }
    }

    #[test]
    fn rotate_features_preserves_norm() {
        // A planar rotation is orthonormal, so the L2 norm is invariant.
        let v = vec![0.3_f32, -0.7, 1.2, 0.4, -0.9, 0.1];
        let norm0: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rotated = rotate_features(&v, 0.9);
        let norm1: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm0 - norm1).abs() < 1e-5,
            "rotation must preserve L2 norm ({norm0} vs {norm1})"
        );
    }

    #[test]
    fn rotated_mnist_single_task_zero_angle() {
        let cfg = ScenarioConfig::new(1, 8, 10, 3, 5).expect("valid scenario config");
        let scn = rotated_mnist(&cfg, 1.5).expect("rotated scenario should generate");
        assert_eq!(scn.angles, vec![0.0]);
    }

    #[test]
    fn rotated_mnist_rejects_nan_angle() {
        let cfg = ScenarioConfig::default();
        assert!(rotated_mnist(&cfg, f32::NAN).is_err());
    }

    // ── Split classes ─────────────────────────────────────────────────────────

    #[test]
    fn split_classes_partitions_disjointly() {
        let cfg = ScenarioConfig::new(5, 16, 32, 10, 11).expect("valid scenario config");
        let scn = split_classes(&cfg).expect("split scenario should generate");
        assert_eq!(scn.class_groups.len(), 5);
        // Even split: 2 classes each.
        for g in &scn.class_groups {
            assert_eq!(g.len(), 2);
        }
        // Union is exactly 0..10 and groups are pairwise disjoint.
        let mut all: Vec<usize> = scn.class_groups.iter().flatten().copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn split_classes_uneven_remainder_distributed() {
        // 7 classes / 3 tasks → {0,1,2} {3,4} {5,6}.
        let cfg = ScenarioConfig::new(3, 8, 16, 7, 1).expect("valid scenario config");
        let scn = split_classes(&cfg).expect("split scenario should generate");
        assert_eq!(scn.class_groups[0], vec![0, 1, 2]);
        assert_eq!(scn.class_groups[1], vec![3, 4]);
        assert_eq!(scn.class_groups[2], vec![5, 6]);
    }

    #[test]
    fn split_classes_stream_accumulates_classes() {
        let cfg = ScenarioConfig::new(5, 16, 24, 10, 7).expect("valid scenario config");
        let scn = split_classes(&cfg).expect("split scenario should generate");
        let mut stream = scn.stream;
        init_class_inc(&mut stream);
        assert_eq!(n_classes_seen(&stream), 2);
        for expected in [4, 6, 8, 10] {
            crate::stream::class_stream::advance_class_inc(&mut stream)
                .expect("advance should succeed");
            assert_eq!(n_classes_seen(&stream), expected);
        }
    }

    #[test]
    fn split_classes_samples_only_from_own_group() {
        let cfg = ScenarioConfig::new(5, 16, 40, 10, 321).expect("valid scenario config");
        let scn = split_classes(&cfg).expect("split scenario should generate");
        for (t, task) in scn.stream.tasks.iter().enumerate() {
            let group = &scn.class_groups[t];
            for (_, label) in &task.data {
                assert!(
                    group.contains(&(*label as usize)),
                    "task {t} produced label {label} outside its class group {group:?}"
                );
            }
        }
    }

    #[test]
    fn split_classes_rejects_too_few_classes() {
        // 3 classes cannot fill 5 tasks.
        let cfg = ScenarioConfig::new(5, 8, 16, 3, 1).expect("valid scenario config");
        assert!(split_classes(&cfg).is_err());
    }

    // ── Config validation ─────────────────────────────────────────────────────

    #[test]
    fn config_rejects_degenerate_values() {
        assert!(ScenarioConfig::new(0, 8, 16, 4, 1).is_err());
        assert!(ScenarioConfig::new(3, 0, 16, 4, 1).is_err());
        assert!(ScenarioConfig::new(3, 8, 0, 4, 1).is_err());
        assert!(ScenarioConfig::new(3, 8, 16, 1, 1).is_err());
    }

    #[test]
    fn class_prototypes_are_distinct_and_finite() {
        let p0 = class_prototype(0, 16);
        let p1 = class_prototype(1, 16);
        assert!(p0.iter().all(|v| v.is_finite()));
        assert!(p1.iter().all(|v| v.is_finite()));
        let diff: f32 = p0.iter().zip(p1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-3,
            "distinct classes must have distinct prototypes"
        );
    }
}
