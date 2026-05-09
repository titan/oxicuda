//! Task-incremental data stream abstraction.
//!
//! Models the task-incremental scenario where:
//! - Tasks arrive sequentially.
//! - At test time, the task identity is known (task oracle).
//! - Each task has its own label space (disjoint or overlapping).

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

/// A labeled dataset for a single task.
#[derive(Debug, Clone)]
pub struct Task {
    /// Unique task identifier.
    pub id: usize,
    /// Number of distinct classes in this task.
    pub n_classes: usize,
    /// Samples: `(feature_vector, label)` pairs.
    pub data: Vec<(Vec<f32>, u32)>,
}

impl Task {
    /// Create a new task.
    pub fn new(id: usize, n_classes: usize, data: Vec<(Vec<f32>, u32)>) -> ContinualResult<Self> {
        // Validate labels are within range
        for (_, label) in &data {
            if *label as usize >= n_classes {
                return Err(ContinualError::TaskIndexOutOfRange {
                    index: *label as usize,
                    n_tasks: n_classes,
                });
            }
        }
        Ok(Self {
            id,
            n_classes,
            data,
        })
    }

    /// Number of samples in this task.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True if this task has no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Task-incremental data stream.
#[derive(Debug, Clone)]
pub struct TaskStream {
    /// All tasks in the stream.
    pub tasks: Vec<Task>,
    /// Index of the current task (0-indexed).
    pub current: usize,
}

/// Create a new task stream from a list of tasks.
pub fn task_stream_new(tasks: Vec<Task>) -> ContinualResult<TaskStream> {
    if tasks.is_empty() {
        return Err(ContinualError::NoTasksInStream);
    }
    Ok(TaskStream { tasks, current: 0 })
}

/// Advance to the next task.
///
/// Returns `Some(&Task)` for the new current task, or `None` if already at the end.
pub fn next_task(stream: &mut TaskStream) -> Option<&Task> {
    if stream.current + 1 >= stream.tasks.len() {
        return None;
    }
    stream.current += 1;
    Some(&stream.tasks[stream.current])
}

/// Return the current task (without advancing).
pub fn current_task(stream: &TaskStream) -> Option<&Task> {
    stream.tasks.get(stream.current)
}

/// Sample a random mini-batch from a task.
///
/// `batch_size` is clamped to `task.data.len()` if larger.
/// Returns the batch as a `Vec<(Vec<f32>, u32)>`.
pub fn task_batch(
    task: &Task,
    batch_size: usize,
    rng: &mut LcgRng,
) -> ContinualResult<Vec<(Vec<f32>, u32)>> {
    if task.data.is_empty() {
        return Err(ContinualError::EmptyInput);
    }
    let n = batch_size.min(task.data.len());
    let mut indices: Vec<usize> = (0..task.data.len()).collect();
    // Partial Fisher-Yates shuffle
    for i in 0..n {
        let j = i + rng.next_usize(task.data.len() - i);
        indices.swap(i, j);
    }
    let batch = indices[..n].iter().map(|&i| task.data[i].clone()).collect();
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_task(id: usize, n: usize) -> Task {
        let data = (0..n)
            .map(|i| (vec![i as f32; 4], (i % 3) as u32))
            .collect();
        Task {
            id,
            n_classes: 3,
            data,
        }
    }

    #[test]
    fn stream_advances_correctly() {
        let tasks = vec![make_task(0, 10), make_task(1, 10), make_task(2, 10)];
        let mut stream = task_stream_new(tasks).unwrap();
        assert_eq!(current_task(&stream).unwrap().id, 0);
        let t1 = next_task(&mut stream).unwrap();
        assert_eq!(t1.id, 1);
        let t2 = next_task(&mut stream).unwrap();
        assert_eq!(t2.id, 2);
        // At end: next_task returns None
        assert!(next_task(&mut stream).is_none());
    }

    #[test]
    fn task_batch_size_respected() {
        let mut rng = LcgRng::new(42);
        let task = make_task(0, 20);
        let batch = task_batch(&task, 8, &mut rng).unwrap();
        assert_eq!(batch.len(), 8, "Batch size should be respected");
    }

    #[test]
    fn task_batch_clamped_to_task_size() {
        let mut rng = LcgRng::new(7);
        let task = make_task(0, 5);
        let batch = task_batch(&task, 100, &mut rng).unwrap();
        assert_eq!(batch.len(), 5, "Batch should be clamped to task size");
    }

    #[test]
    fn task_batch_labels_valid() {
        let mut rng = LcgRng::new(13);
        let task = make_task(0, 20);
        let batch = task_batch(&task, 10, &mut rng).unwrap();
        for (_, label) in &batch {
            assert!(
                (*label as usize) < task.n_classes,
                "Label {label} out of range"
            );
        }
    }

    #[test]
    fn empty_stream_returns_err() {
        let result = task_stream_new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn task_new_validates_labels() {
        let data = vec![(vec![0.5_f32; 4], 5u32)]; // label 5 ≥ n_classes=3
        assert!(Task::new(0, 3, data).is_err());
    }

    #[test]
    fn task_empty_batch_returns_err() {
        let mut rng = LcgRng::new(1);
        let task = Task {
            id: 0,
            n_classes: 3,
            data: vec![],
        };
        assert!(task_batch(&task, 4, &mut rng).is_err());
    }

    #[test]
    fn current_task_returns_first() {
        let tasks = vec![make_task(0, 5), make_task(1, 5)];
        let stream = task_stream_new(tasks).unwrap();
        assert_eq!(current_task(&stream).unwrap().id, 0);
    }
}
