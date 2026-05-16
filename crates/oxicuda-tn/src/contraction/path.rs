//! Greedy contraction-path optimisation.
//!
//! Given a list of tensors with labels, find an ordering of binary contractions that
//! minimises the cumulative FLOPs. We use a textbook greedy: at every step pick the
//! pair that produces the smallest intermediate tensor.

use crate::contraction::einsum::{LabelledTensor, einsum_binary};
use crate::{TnError, TnResult};

/// A contraction path is a sequence of binary contractions specified as `(i, j)` index
/// pairs into the working list of tensors. After each step, the contracted result
/// replaces position `min(i, j)` and the larger index is removed.
#[derive(Debug, Clone)]
pub struct ContractionPath {
    pub steps: Vec<(usize, usize)>,
    pub total_cost: usize,
}

/// A single candidate contraction pair with the estimated cost, indices, and the
/// dimensions/labels of the resulting tensor.
struct BestPair {
    cost: usize,
    i: usize,
    j: usize,
    new_dims: Vec<usize>,
    new_labels: Vec<char>,
}

/// Compute a greedy contraction path for the given list of labelled tensors.
///
/// The cost of a contraction is approximated by the product of all involved dimensions
/// (FLOPs).
pub fn greedy_path(tensors: &[LabelledTensor]) -> TnResult<ContractionPath> {
    if tensors.is_empty() {
        return Err(TnError::EmptyInput);
    }
    if tensors.len() == 1 {
        return Ok(ContractionPath {
            steps: Vec::new(),
            total_cost: 0,
        });
    }
    // Working representation: vector of (dims, labels)
    let mut working: Vec<(Vec<usize>, Vec<char>)> = tensors
        .iter()
        .map(|t| (t.dims.clone(), t.labels.clone()))
        .collect();
    let mut steps = Vec::new();
    let mut total = 0usize;
    while working.len() > 1 {
        let mut best: Option<BestPair> = None;
        for i in 0..working.len() {
            for j in i + 1..working.len() {
                let (cost, new_dims, new_labels) = pair_cost(&working[i], &working[j]);
                if best.as_ref().map(|b| cost < b.cost).unwrap_or(true) {
                    best = Some(BestPair {
                        cost,
                        i,
                        j,
                        new_dims,
                        new_labels,
                    });
                }
            }
        }
        let bp = best.ok_or(TnError::ContractionPathInvalid("no pair found".into()))?;
        total = total.saturating_add(bp.cost);
        steps.push((bp.i, bp.j));
        working[bp.i] = (bp.new_dims, bp.new_labels);
        working.remove(bp.j);
    }
    Ok(ContractionPath {
        steps,
        total_cost: total,
    })
}

fn pair_cost(
    a: &(Vec<usize>, Vec<char>),
    b: &(Vec<usize>, Vec<char>),
) -> (usize, Vec<usize>, Vec<char>) {
    let mut all_labels: Vec<char> = a.1.clone();
    for &l in &b.1 {
        if !all_labels.contains(&l) {
            all_labels.push(l);
        }
    }
    let mut all_dims_map: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    for (l, d) in a.1.iter().zip(a.0.iter()) {
        all_dims_map.insert(*l, *d);
    }
    for (l, d) in b.1.iter().zip(b.0.iter()) {
        all_dims_map.insert(*l, *d);
    }
    // Cost ≈ product of all involved dims (shared + kept)
    let mut cost: usize = 1;
    for &l in &all_labels {
        let dim = *all_dims_map.get(&l).unwrap_or(&1);
        cost = cost.saturating_mul(dim);
    }
    // Output dims/labels: kept_a + kept_b, where kept = not shared
    let shared: Vec<char> = a.1.iter().filter(|l| b.1.contains(l)).copied().collect();
    let mut new_labels: Vec<char> =
        a.1.iter()
            .filter(|l| !shared.contains(l))
            .copied()
            .collect();
    new_labels.extend(b.1.iter().filter(|l| !shared.contains(l)).copied());
    let new_dims: Vec<usize> = new_labels.iter().map(|l| all_dims_map[l]).collect();
    (cost, new_dims, new_labels)
}

/// Execute a contraction path, returning the single resulting [`LabelledTensor`].
pub fn execute_path(
    tensors: Vec<LabelledTensor>,
    path: &ContractionPath,
) -> TnResult<LabelledTensor> {
    let mut working: Vec<LabelledTensor> = tensors;
    for &(i, j) in &path.steps {
        if i >= working.len() || j >= working.len() {
            return Err(TnError::ContractionPathInvalid(
                "step index out of range".into(),
            ));
        }
        let new_tensor = einsum_binary(&working[i], &working[j])?;
        working[i] = new_tensor;
        working.remove(j);
    }
    working.into_iter().next().ok_or(TnError::EmptyInput)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_path_two_tensors() {
        let a =
            LabelledTensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], vec!['i', 'j']).expect("ok");
        let b =
            LabelledTensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2], vec!['j', 'k']).expect("ok");
        let path = greedy_path(&[a.clone(), b.clone()]).expect("ok");
        assert_eq!(path.steps.len(), 1);
        let result = execute_path(vec![a, b], &path).expect("ok");
        assert_eq!(result.dims, vec![2, 2]);
    }

    #[test]
    fn greedy_path_chain_of_three() {
        let a = LabelledTensor::new(vec![1.0; 6], vec![2, 3], vec!['a', 'b']).expect("ok");
        let b = LabelledTensor::new(vec![1.0; 12], vec![3, 4], vec!['b', 'c']).expect("ok");
        let c = LabelledTensor::new(vec![1.0; 8], vec![4, 2], vec!['c', 'd']).expect("ok");
        let path = greedy_path(&[a.clone(), b.clone(), c.clone()]).expect("ok");
        assert_eq!(path.steps.len(), 2);
        let result = execute_path(vec![a, b, c], &path).expect("ok");
        assert_eq!(result.dims, vec![2, 2]);
    }
}
