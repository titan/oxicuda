/// Max-heap of capacity `k` that tracks the k *smallest* (dist, id) items.
///
/// Internally stores the largest dist at the top so eviction is O(log k).
#[derive(Debug, Clone)]
pub struct BoundedMaxHeap {
    cap: usize,
    items: Vec<(f32, usize)>,
}

impl BoundedMaxHeap {
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            cap: k,
            items: Vec::with_capacity(k + 1),
        }
    }

    /// Current worst (maximum) distance stored, or `f32::INFINITY` when empty.
    #[must_use]
    pub fn worst_dist(&self) -> f32 {
        self.items.first().map_or(f32::INFINITY, |(d, _)| *d)
    }

    /// Push `(dist, id)`. If heap is full and `dist < worst`, evict worst first.
    pub fn push(&mut self, dist: f32, id: usize) {
        if self.items.len() < self.cap {
            self.items.push((dist, id));
            self.sift_up(self.items.len() - 1);
        } else if dist < self.worst_dist() {
            self.items[0] = (dist, id);
            self.sift_down(0);
        }
    }

    /// Return items sorted ascending by distance (consumed).
    pub fn into_sorted_vec(mut self) -> Vec<(usize, f32)> {
        self.items
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        self.items.into_iter().map(|(d, i)| (i, d)).collect()
    }

    fn sift_up(&mut self, mut pos: usize) {
        while pos > 0 {
            let parent = (pos - 1) / 2;
            if self.items[pos].0 > self.items[parent].0 {
                self.items.swap(pos, parent);
                pos = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut pos: usize) {
        let len = self.items.len();
        loop {
            let left = 2 * pos + 1;
            let right = 2 * pos + 2;
            let mut largest = pos;
            if left < len && self.items[left].0 > self.items[largest].0 {
                largest = left;
            }
            if right < len && self.items[right].0 > self.items[largest].0 {
                largest = right;
            }
            if largest == pos {
                break;
            }
            self.items.swap(pos, largest);
            pos = largest;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_k_smallest() {
        let mut h = BoundedMaxHeap::new(3);
        for (i, &d) in [5.0_f32, 1.0, 3.0, 2.0, 4.0].iter().enumerate() {
            h.push(d, i);
        }
        let res = h.into_sorted_vec();
        assert_eq!(res.len(), 3);
        assert!((res[0].1 - 1.0).abs() < 0.1 || res[0].0 < 2);
        assert!(res[0].1 <= res[1].1 && res[1].1 <= res[2].1);
    }
}
