use super::heap::BoundedMaxHeap;

/// O(n) top-K selection using `BoundedMaxHeap`.
/// Returns `(id, dist)` sorted ascending by distance.
pub fn select_topk(dists: &[(usize, f32)], k: usize) -> Vec<(usize, f32)> {
    if k == 0 || dists.is_empty() {
        return Vec::new();
    }
    let actual_k = k.min(dists.len());
    let mut heap = BoundedMaxHeap::new(actual_k);
    for &(id, d) in dists {
        heap.push(d, id);
    }
    heap.into_sorted_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_correct() {
        let input: Vec<(usize, f32)> = vec![(0, 9.0), (1, 1.0), (2, 5.0), (3, 2.0)];
        let res = select_topk(&input, 2);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].0, 1); // dist=1
        assert_eq!(res[1].0, 3); // dist=2
    }
}
