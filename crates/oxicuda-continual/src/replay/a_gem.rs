//! Averaged-GEM (A-GEM): single averaged constraint gradient.
//!
//! Implements the method from:
//! Chaudhry et al. "Efficient Lifelong Learning with A-GEM." ICLR 2019.
//!
//! A-GEM simplifies GEM by using a single reference gradient averaged over
//! all episodic memory, reducing the quadratic programming to a simple
//! projection onto a single half-space.

use crate::error::{ContinualError, ContinualResult};

/// Configuration for A-GEM.
#[derive(Debug, Clone)]
pub struct AGemConfig {
    /// Margin for constraint: require g·g_ref ≥ -margin.
    pub margin: f32,
}

impl Default for AGemConfig {
    fn default() -> Self {
        Self { margin: 0.0 }
    }
}

/// Compute the dot product of two equal-length slices.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Project the current gradient `g` using the averaged memory gradient `g_ref`.
///
/// If `g · g_ref ≥ -margin`: return `g` unchanged (already feasible).
/// Otherwise: `g' = g - (g·g_ref + margin) / (g_ref·g_ref) · g_ref`
///
/// This ensures `g'·g_ref = g·g_ref - (g·g_ref + margin) = -margin`.
pub fn a_gem_project(
    current_grad: &[f32],
    memory_grad: &[f32],
    margin: f32,
) -> ContinualResult<Vec<f32>> {
    let d = current_grad.len();
    if d == 0 {
        return Err(ContinualError::EmptyInput);
    }
    if memory_grad.len() != d {
        return Err(ContinualError::DimensionMismatch {
            expected: d,
            got: memory_grad.len(),
        });
    }

    let dot_gm = dot(current_grad, memory_grad);

    // Feasibility check: g · g_ref >= -margin
    if dot_gm >= -margin {
        return Ok(current_grad.to_vec());
    }

    // Project: g' = g - (g·g_ref + margin) / (g_ref·g_ref) · g_ref
    let dot_mm = dot(memory_grad, memory_grad);
    if dot_mm < 1e-30 {
        // Memory gradient is near-zero; cannot project, return as-is
        return Ok(current_grad.to_vec());
    }

    let scale = (dot_gm + margin) / dot_mm;
    let result = current_grad
        .iter()
        .zip(memory_grad.iter())
        .map(|(&g, &m)| g - scale * m)
        .collect();
    Ok(result)
}

/// Compute the average gradient over a set of per-task gradients.
///
/// Returns `Err(EmptyInput)` if `grads` is empty.
pub fn average_gradients(grads: &[Vec<f32>]) -> ContinualResult<Vec<f32>> {
    if grads.is_empty() {
        return Err(ContinualError::EmptyInput);
    }
    let d = grads[0].len();
    let n = grads.len() as f32;
    let mut avg = vec![0.0_f32; d];
    for g in grads {
        if g.len() != d {
            return Err(ContinualError::DimensionMismatch {
                expected: d,
                got: g.len(),
            });
        }
        for (a, &v) in avg.iter_mut().zip(g.iter()) {
            *a += v;
        }
    }
    for a in &mut avg {
        *a /= n;
    }
    Ok(avg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gem_aligned_gradient_unchanged() {
        // g aligned with g_ref → g · g_ref > 0 ≥ -margin = 0 → no projection
        let g = vec![1.0_f32, 0.0, 0.0];
        let g_ref = vec![1.0_f32, 0.0, 0.0];
        let g_proj = a_gem_project(&g, &g_ref, 0.0).unwrap();
        for (a, b) in g.iter().zip(g_proj.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Aligned gradient should not be modified"
            );
        }
    }

    #[test]
    fn a_gem_anti_aligned_projected_to_near_orthogonal() {
        // g exactly anti-aligned: g = [-1, 0], g_ref = [1, 0]
        // g · g_ref = -1 < 0 = -margin → project
        // scale = (dot_gm + margin) / dot_mm = -1/1 = -1
        // g' = g - (-1)*g_ref = [-1,0] + [1,0] = [0, 0] → orthogonal
        let g = vec![-1.0_f32, 0.0];
        let g_ref = vec![1.0_f32, 0.0];
        let g_proj = a_gem_project(&g, &g_ref, 0.0).unwrap();
        let dot_after = dot(&g_proj, &g_ref);
        assert!(
            dot_after.abs() < 1e-5,
            "Projected anti-aligned gradient should be orthogonal, dot={dot_after}"
        );
    }

    #[test]
    fn a_gem_margin_enforcement() {
        // g · g_ref = -0.5, margin = 0.3 → -0.5 < -0.3 → must project
        // After projection: g'·g_ref = -margin = -0.3
        let g = vec![-0.5_f32, 1.0];
        let g_ref = vec![1.0_f32, 0.0]; // g · g_ref = -0.5
        let margin = 0.3;
        let g_proj = a_gem_project(&g, &g_ref, margin).unwrap();
        let dot_after = dot(&g_proj, &g_ref);
        assert!(
            dot_after >= -margin - 1e-5,
            "After projection, g'·g_ref should be >= -margin, got {dot_after}"
        );
    }

    #[test]
    fn a_gem_already_feasible_with_margin() {
        // g · g_ref = 0.5, margin = 1.0 → 0.5 >= -1.0 → no projection
        let g = vec![0.5_f32, 0.0];
        let g_ref = vec![1.0_f32, 0.0];
        let g_proj = a_gem_project(&g, &g_ref, 1.0).unwrap();
        assert!((g_proj[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_gem_empty_gradient_returns_err() {
        let g_ref = vec![1.0_f32];
        assert!(a_gem_project(&[], &g_ref, 0.0).is_err());
    }

    #[test]
    fn a_gem_dimension_mismatch_returns_err() {
        let g = vec![1.0_f32; 4];
        let g_ref = vec![1.0_f32; 3];
        assert!(a_gem_project(&g, &g_ref, 0.0).is_err());
    }

    #[test]
    fn average_gradients_correct() {
        let grads = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 4.0]];
        let avg = average_gradients(&grads).unwrap();
        assert!((avg[0] - 2.0).abs() < 1e-6);
        assert!((avg[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn average_gradients_empty_returns_err() {
        let grads: Vec<Vec<f32>> = vec![];
        assert!(average_gradients(&grads).is_err());
    }
}
