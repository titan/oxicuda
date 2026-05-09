//! Gradient Episodic Memory (GEM).
//!
//! Implements the method from:
//! Lopez-Paz & Ranzato. "Gradient Episodic Memory for Continual Learning."
//! NeurIPS 2017.
//!
//! GEM stores a small episodic memory for each task and projects the current
//! gradient so that it does not increase the loss on any previous task:
//! `min ||g' - g||² s.t. g'·g_k ≥ -margin ∀k`

use crate::error::{ContinualError, ContinualResult};

/// Configuration for GEM.
#[derive(Debug, Clone)]
pub struct GemConfig {
    /// Maximum number of tasks.
    pub n_tasks: usize,
    /// Number of memory samples per task.
    pub memory_per_task: usize,
    /// Margin for constraint violation: allow g·g_k ≥ -margin.
    pub margin: f32,
}

impl Default for GemConfig {
    fn default() -> Self {
        Self {
            n_tasks: 10,
            memory_per_task: 256,
            margin: 0.5,
        }
    }
}

/// Episodic memory for GEM.
#[derive(Debug, Clone, Default)]
pub struct GemMemory {
    /// Per-task stored feature vectors.
    pub per_task_data: Vec<Vec<Vec<f32>>>,
    /// Per-task stored labels.
    pub per_task_labels: Vec<Vec<u32>>,
}

impl GemMemory {
    /// Create an empty episodic memory.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of tasks stored.
    #[must_use]
    pub fn n_tasks(&self) -> usize {
        self.per_task_data.len()
    }

    /// Add a task's memory samples.
    pub fn add_task(&mut self, data: Vec<Vec<f32>>, labels: Vec<u32>) {
        self.per_task_data.push(data);
        self.per_task_labels.push(labels);
    }
}

/// Compute the dot product of two vectors.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Project the current gradient `g` so that `g · g_k ≥ -margin` for all
/// memory gradients `g_k`.
///
/// Algorithm:
/// 1. Check feasibility: if `g · g_k ≥ -margin` for all k, return `g` unchanged.
/// 2. Otherwise, find the most violated constraint (smallest `g · g_k`).
/// 3. Project `g` onto the half-space: `g' = g - (g·g_k / g_k·g_k) · g_k`.
/// 4. Repeat until feasible or no more violations.
///
/// This is a simplified single-step projection sufficient for GEM's online use.
pub fn gem_project_gradient(
    current_grad: &[f32],
    memory_grads: &[Vec<f32>],
    margin: f32,
) -> ContinualResult<Vec<f32>> {
    if current_grad.is_empty() {
        return Err(ContinualError::EmptyInput);
    }
    let d = current_grad.len();
    // Validate all memory gradient dimensions
    for (k, mg) in memory_grads.iter().enumerate() {
        if mg.len() != d {
            return Err(ContinualError::DimensionMismatch {
                expected: d,
                got: mg.len(),
            });
        }
        let _ = k;
    }

    if memory_grads.is_empty() {
        return Ok(current_grad.to_vec());
    }

    let mut g = current_grad.to_vec();

    // Iterative projection: up to n_constraints passes
    let n_constraints = memory_grads.len();
    for _pass in 0..n_constraints {
        // Find most violated constraint
        let mut worst_k = None;
        let mut worst_dot = f32::INFINITY;
        for (k, mg) in memory_grads.iter().enumerate() {
            let d_gm = dot(&g, mg);
            if d_gm < -margin && d_gm < worst_dot {
                worst_dot = d_gm;
                worst_k = Some(k);
            }
        }
        // If feasible, done
        let k = match worst_k {
            Some(k) => k,
            None => break,
        };
        // Project: g' = g - (g·g_k / g_k·g_k) · g_k
        let mg = &memory_grads[k];
        let dot_gm = dot(&g, mg);
        let dot_mm = dot(mg, mg);
        if dot_mm < 1e-30 {
            continue;
        }
        let scale = dot_gm / dot_mm;
        for i in 0..d {
            g[i] -= scale * mg[i];
        }
    }

    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gem_feasible_gradient_unchanged() {
        // g is aligned with memory grad → no projection needed
        let g = vec![1.0_f32, 0.0, 0.0, 0.0];
        let mem_grads = vec![vec![1.0_f32, 0.0, 0.0, 0.0]];
        let margin = 0.0;
        let g_proj = gem_project_gradient(&g, &mem_grads, margin).unwrap();
        // g · mem = 1.0 >= 0 = -margin, so no change
        for (a, b) in g.iter().zip(g_proj.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Feasible gradient should be unchanged"
            );
        }
    }

    #[test]
    fn gem_infeasible_gradient_projected() {
        // g is anti-aligned with memory grad → must be projected
        let g = vec![-1.0_f32, 0.0];
        let mem_grads = vec![vec![1.0_f32, 0.0]];
        let margin = 0.0;
        let g_proj = gem_project_gradient(&g, &mem_grads, margin).unwrap();
        // After projection, g · mem should be ≥ -margin
        let dot_after = dot(&g_proj, &mem_grads[0]);
        assert!(
            dot_after >= -margin - 1e-5,
            "Projected gradient must satisfy constraint, got {dot_after}"
        );
    }

    #[test]
    fn gem_projected_gradient_satisfies_constraint() {
        let g = vec![-2.0_f32, 1.0, -1.0];
        let mem_grads = vec![vec![1.0_f32, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let margin = 0.1;
        let g_proj = gem_project_gradient(&g, &mem_grads, margin).unwrap();
        for mg in &mem_grads {
            let d = dot(&g_proj, mg);
            assert!(
                d >= -margin - 1e-5,
                "Projected gradient must satisfy constraint g·g_k >= -margin, got {d}"
            );
        }
    }

    #[test]
    fn gem_empty_memory_returns_unchanged() {
        let g = vec![1.0_f32, 2.0, 3.0];
        let mem_grads: Vec<Vec<f32>> = vec![];
        let g_proj = gem_project_gradient(&g, &mem_grads, 0.0).unwrap();
        assert_eq!(g, g_proj);
    }

    #[test]
    fn gem_empty_gradient_returns_err() {
        let mem_grads = vec![vec![1.0_f32]];
        assert!(gem_project_gradient(&[], &mem_grads, 0.0).is_err());
    }

    #[test]
    fn gem_dimension_mismatch_returns_err() {
        let g = vec![1.0_f32; 4];
        let mem_grads = vec![vec![1.0_f32; 3]]; // wrong dim
        assert!(gem_project_gradient(&g, &mem_grads, 0.0).is_err());
    }

    #[test]
    fn gem_memory_add_task() {
        let mut mem = GemMemory::new();
        mem.add_task(vec![vec![0.5_f32]; 4], vec![0, 1, 0, 1]);
        assert_eq!(mem.n_tasks(), 1);
        mem.add_task(vec![vec![1.0_f32]; 4], vec![2, 3, 2, 3]);
        assert_eq!(mem.n_tasks(), 2);
    }
}
