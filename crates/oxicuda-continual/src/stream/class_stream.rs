//! Class-incremental data stream.
//!
//! Models the class-incremental scenario where:
//! - New classes are introduced at each task step.
//! - At test time, the task identity is NOT known (no task oracle).
//! - Each task introduces a disjoint set of new classes.
//! - The classifier must distinguish all seen classes simultaneously.

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;
use crate::stream::task_stream::Task;

/// Class-incremental stream: new classes per step, labels globally unique.
#[derive(Debug, Clone)]
pub struct ClassIncStream {
    /// All tasks (each introduces new classes).
    pub tasks: Vec<Task>,
    /// Per-task: list of class IDs introduced by that task.
    pub task_classes: Vec<Vec<usize>>,
    /// Accumulated classes seen so far (set of class IDs).
    pub seen_classes: Vec<usize>,
    /// Current task index.
    pub current: usize,
}

/// Create a new class-incremental stream.
///
/// Each task in `tasks` must have globally unique labels (no overlap).
/// The stream starts at task 0 with those classes marked as seen.
pub fn class_inc_new(tasks: Vec<Task>) -> ContinualResult<ClassIncStream> {
    if tasks.is_empty() {
        return Err(ContinualError::NoTasksInStream);
    }
    // Compute per-task class sets and verify disjointness
    let mut task_classes: Vec<Vec<usize>> = Vec::with_capacity(tasks.len());
    let mut all_classes: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for task in &tasks {
        let mut class_set: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (_, label) in &task.data {
            class_set.insert(*label as usize);
        }
        // Verify disjoint from previous tasks
        for &c in &class_set {
            if all_classes.contains(&c) {
                return Err(ContinualError::Internal(format!(
                    "Class {c} appears in multiple tasks (not disjoint)"
                )));
            }
            all_classes.insert(c);
        }
        let mut classes: Vec<usize> = class_set.into_iter().collect();
        classes.sort_unstable();
        task_classes.push(classes);
    }
    // Initially: no classes seen
    Ok(ClassIncStream {
        tasks,
        task_classes,
        seen_classes: vec![],
        current: 0,
    })
}

/// Advance to the next task, marking the new task's classes as seen.
///
/// After calling this, `n_classes_seen` will include the classes from the
/// newly-current task. If already at the last task, this is a no-op.
pub fn advance_class_inc(stream: &mut ClassIncStream) -> ContinualResult<()> {
    if stream.current + 1 >= stream.tasks.len() {
        // Already at last task; nothing to advance
        return Ok(());
    }
    stream.current += 1;
    // Mark the new current task's classes as seen
    let new_classes = stream.task_classes[stream.current].clone();
    for c in new_classes {
        if !stream.seen_classes.contains(&c) {
            stream.seen_classes.push(c);
        }
    }
    Ok(())
}

/// Initialize seen classes to the first task (call once at the start).
pub fn init_class_inc(stream: &mut ClassIncStream) {
    if let Some(classes) = stream.task_classes.first() {
        for &c in classes {
            if !stream.seen_classes.contains(&c) {
                stream.seen_classes.push(c);
            }
        }
    }
}

/// Return the total number of distinct classes seen so far.
#[must_use]
pub fn n_classes_seen(stream: &ClassIncStream) -> usize {
    stream.seen_classes.len()
}

/// Sample a mini-batch from a specific task with globally unique labels.
///
/// Returns `Err` if `task_id` is out of range or the task is empty.
pub fn class_inc_batch(
    stream: &ClassIncStream,
    task_id: usize,
    n: usize,
    rng: &mut LcgRng,
) -> ContinualResult<Vec<(Vec<f32>, u32)>> {
    if task_id >= stream.tasks.len() {
        return Err(ContinualError::TaskIndexOutOfRange {
            index: task_id,
            n_tasks: stream.tasks.len(),
        });
    }
    let task = &stream.tasks[task_id];
    if task.data.is_empty() {
        return Err(ContinualError::EmptyInput);
    }
    let batch_n = n.min(task.data.len());
    let mut indices: Vec<usize> = (0..task.data.len()).collect();
    for i in 0..batch_n {
        let j = i + rng.next_usize(task.data.len() - i);
        indices.swap(i, j);
    }
    let batch = indices[..batch_n]
        .iter()
        .map(|&i| task.data[i].clone())
        .collect();
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_task_cls(id: usize, n_samples: usize, class_offset: usize, n_classes: usize) -> Task {
        let data = (0..n_samples)
            .map(|i| {
                let label = (class_offset + (i % n_classes)) as u32;
                (vec![i as f32; 4], label)
            })
            .collect();
        Task {
            id,
            n_classes: class_offset + n_classes, // global n_classes
            data,
        }
    }

    #[test]
    fn class_inc_classes_accumulate() {
        let t0 = make_task_cls(0, 10, 0, 2); // classes 0, 1
        let t1 = make_task_cls(1, 10, 2, 3); // classes 2, 3, 4
        let mut stream = class_inc_new(vec![t0, t1])
            .expect("class-incremental stream should initialize with valid tasks");
        init_class_inc(&mut stream);
        assert_eq!(
            n_classes_seen(&stream),
            2,
            "Should see 2 classes after task 0"
        );
        advance_class_inc(&mut stream).expect("advancing class-incremental stream should succeed");
        assert_eq!(
            n_classes_seen(&stream),
            5,
            "Should see 5 classes after task 1"
        );
    }

    #[test]
    fn class_inc_batch_size_correct() {
        let t0 = make_task_cls(0, 20, 0, 2);
        let t1 = make_task_cls(1, 20, 2, 3);
        let mut stream = class_inc_new(vec![t0, t1])
            .expect("class-incremental stream should initialize with valid tasks");
        let mut rng = LcgRng::new(42);
        let batch = class_inc_batch(&stream, 0, 8, &mut rng)
            .expect("class-incremental batch sampling should succeed");
        assert_eq!(batch.len(), 8);
        // Advance and sample from task 1
        advance_class_inc(&mut stream).expect("advancing class-incremental stream should succeed");
        let batch1 = class_inc_batch(&stream, 1, 6, &mut rng)
            .expect("class-incremental batch sampling should succeed");
        assert_eq!(batch1.len(), 6);
    }

    #[test]
    fn class_inc_labels_disjoint() {
        let t0 = make_task_cls(0, 10, 0, 3); // classes 0, 1, 2
        let t1 = make_task_cls(1, 10, 3, 2); // classes 3, 4
        let stream = class_inc_new(vec![t0, t1])
            .expect("class-incremental stream should initialize with valid tasks");
        let mut rng = LcgRng::new(7);
        let b0 = class_inc_batch(&stream, 0, 10, &mut rng)
            .expect("class-incremental batch sampling should succeed");
        let b1 = class_inc_batch(&stream, 1, 10, &mut rng)
            .expect("class-incremental batch sampling should succeed");
        // Verify no label from t0 appears in t1
        let labels0: std::collections::HashSet<u32> = b0.iter().map(|(_, l)| *l).collect();
        let labels1: std::collections::HashSet<u32> = b1.iter().map(|(_, l)| *l).collect();
        let intersection: Vec<_> = labels0.intersection(&labels1).collect();
        assert!(
            intersection.is_empty(),
            "Labels across tasks should be disjoint, intersection: {intersection:?}"
        );
    }

    #[test]
    fn class_inc_overlapping_classes_returns_err() {
        // Both tasks use class 0 → should fail
        let t0 = make_task_cls(0, 5, 0, 2); // classes 0, 1
        let t1 = make_task_cls(1, 5, 0, 2); // also classes 0, 1 → overlap
        assert!(class_inc_new(vec![t0, t1]).is_err());
    }

    #[test]
    fn class_inc_empty_stream_returns_err() {
        assert!(class_inc_new(vec![]).is_err());
    }

    #[test]
    fn class_inc_batch_out_of_range_returns_err() {
        let t0 = make_task_cls(0, 10, 0, 2);
        let stream = class_inc_new(vec![t0])
            .expect("class-incremental stream should initialize with valid tasks");
        let mut rng = LcgRng::new(1);
        assert!(class_inc_batch(&stream, 5, 4, &mut rng).is_err());
    }

    #[test]
    fn class_inc_multiple_advances() {
        let tasks: Vec<Task> = (0..4).map(|i| make_task_cls(i, 5, i * 2, 2)).collect();
        let mut stream = class_inc_new(tasks)
            .expect("class-incremental stream should initialize with valid tasks");
        init_class_inc(&mut stream);
        assert_eq!(n_classes_seen(&stream), 2);
        for expected in [4, 6, 8] {
            advance_class_inc(&mut stream)
                .expect("advancing class-incremental stream should succeed");
            assert_eq!(
                n_classes_seen(&stream),
                expected,
                "Classes seen should be {expected}"
            );
        }
    }
}
