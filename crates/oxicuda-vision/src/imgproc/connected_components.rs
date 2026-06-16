//! Connected-component labelling of binary images via union-find.
//!
//! Given a **single-channel** `[h × w]` row-major `f32` image, every pixel with
//! a strictly positive value (`> 0`) is treated as *foreground*; everything else
//! is *background*. [`connected_components`] groups foreground pixels into
//! maximal connected regions under either 4- or 8-connectivity and returns a
//! label image plus per-component statistics.
//!
//! The algorithm is the classical two-pass scan (Rosenfeld–Pfaltz) backed by a
//! union-find / disjoint-set forest with path halving and union by rank:
//!
//! 1. **First pass** — scan in raster order; each foreground pixel inherits the
//!    smallest label among its already-visited neighbours (or starts a new
//!    provisional label), unioning every neighbouring label together.
//! 2. **Resolve** — collapse equivalence classes to their representatives.
//! 3. **Second pass** — relabel every pixel to a compact `1..=num_components`
//!    identifier and accumulate component sizes.

use crate::error::{VisionError, VisionResult};

/// Pixel adjacency used when grouping foreground pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    /// 4-connectivity: only the N/S/E/W neighbours are adjacent.
    Four,
    /// 8-connectivity: the diagonal neighbours are adjacent as well.
    Eight,
}

/// Result of [`connected_components`].
#[derive(Debug, Clone)]
pub struct ComponentLabels {
    /// Row-major `[height × width]` label image; `0` is background and
    /// `1..=num_components` identifies each connected region.
    pub labels: Vec<u32>,
    /// Number of distinct foreground components.
    pub num_components: usize,
    /// Image height.
    pub height: usize,
    /// Image width.
    pub width: usize,
    /// Pixel count of each component; `sizes[label - 1]` is the size of `label`.
    sizes: Vec<usize>,
}

impl ComponentLabels {
    /// Per-component pixel counts, indexed by `label - 1`.
    #[must_use]
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// Pixel count of a specific `label` (`0` for background or an out-of-range
    /// label).
    #[must_use]
    pub fn size_of(&self, label: u32) -> usize {
        if label == 0 || (label as usize) > self.num_components {
            0
        } else {
            self.sizes[label as usize - 1]
        }
    }

    /// Axis-aligned bounding boxes of every component as inclusive
    /// `[x_min, y_min, x_max, y_max]`, indexed by `label - 1`.
    #[must_use]
    pub fn bounding_boxes(&self) -> Vec<[usize; 4]> {
        let mut boxes = vec![[usize::MAX, usize::MAX, 0usize, 0usize]; self.num_components];
        for y in 0..self.height {
            for x in 0..self.width {
                let label = self.labels[y * self.width + x];
                if label == 0 {
                    continue;
                }
                let bb = &mut boxes[label as usize - 1];
                bb[0] = bb[0].min(x);
                bb[1] = bb[1].min(y);
                bb[2] = bb[2].max(x);
                bb[3] = bb[3].max(y);
            }
        }
        boxes
    }
}

/// Disjoint-set forest over provisional labels (index `0` reserved for
/// background and never allocated as a set).
struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u32>,
}

impl UnionFind {
    fn new() -> Self {
        // Index 0 is the background sentinel.
        Self {
            parent: vec![0],
            rank: vec![0],
        }
    }

    /// Allocate a fresh singleton set and return its label.
    fn make_set(&mut self) -> u32 {
        let id = self.parent.len() as u32;
        self.parent.push(id);
        self.rank.push(0);
        id
    }

    /// Find the representative of `x` with path halving.
    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let grandparent = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = grandparent;
            x = grandparent;
        }
        x
    }

    /// Merge the sets containing `a` and `b` (union by rank).
    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        let (high, low) = if self.rank[ra as usize] < self.rank[rb as usize] {
            (rb, ra)
        } else {
            (ra, rb)
        };
        self.parent[low as usize] = high;
        if self.rank[high as usize] == self.rank[low as usize] {
            self.rank[high as usize] += 1;
        }
    }
}

/// Validate a single-channel image buffer.
#[inline]
fn validate_gray(img: &[f32], h: usize, w: usize) -> VisionResult<()> {
    if h == 0 || w == 0 {
        return Err(VisionError::InvalidImageSize {
            height: h,
            width: w,
            channels: 1,
        });
    }
    if img.len() != h * w {
        return Err(VisionError::DimensionMismatch {
            expected: h * w,
            got: img.len(),
        });
    }
    Ok(())
}

/// Label the connected foreground components of a binary image.
///
/// Foreground pixels are those with value `> 0`. Returns a [`ComponentLabels`]
/// holding the compact label image (`0` = background, `1..=num_components`),
/// component count, and per-component sizes.
///
/// # Errors
/// Returns [`VisionError::InvalidImageSize`] if `h == 0` or `w == 0`, and
/// [`VisionError::DimensionMismatch`] if `img.len() != h * w`.
pub fn connected_components(
    img: &[f32],
    h: usize,
    w: usize,
    connectivity: Connectivity,
) -> VisionResult<ComponentLabels> {
    validate_gray(img, h, w)?;

    let mut uf = UnionFind::new();
    let mut provisional = vec![0u32; h * w];

    // ── First pass: provisional labelling + equivalence recording ───────────
    let mut neighbours: Vec<u32> = Vec::with_capacity(4);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if img[idx] <= 0.0 {
                continue;
            }
            neighbours.clear();
            // West.
            if x > 0 && provisional[idx - 1] != 0 {
                neighbours.push(provisional[idx - 1]);
            }
            // North.
            if y > 0 && provisional[idx - w] != 0 {
                neighbours.push(provisional[idx - w]);
            }
            if connectivity == Connectivity::Eight {
                // North-west.
                if x > 0 && y > 0 && provisional[idx - w - 1] != 0 {
                    neighbours.push(provisional[idx - w - 1]);
                }
                // North-east.
                if x + 1 < w && y > 0 && provisional[idx - w + 1] != 0 {
                    neighbours.push(provisional[idx - w + 1]);
                }
            }

            match neighbours.split_first() {
                None => {
                    provisional[idx] = uf.make_set();
                }
                Some((&head, tail)) => {
                    let mut root = uf.find(head);
                    for &other in tail {
                        let r = uf.find(other);
                        if r < root {
                            root = r;
                        }
                    }
                    uf.union(root, head);
                    for &other in tail {
                        uf.union(root, other);
                    }
                    provisional[idx] = root;
                }
            }
        }
    }

    // ── Second pass: compact relabel + size accumulation ────────────────────
    let mut remap = vec![0u32; uf.parent.len()];
    let mut labels = vec![0u32; h * w];
    let mut sizes: Vec<usize> = Vec::new();
    let mut num_components = 0u32;
    for (idx, &prov) in provisional.iter().enumerate() {
        if prov == 0 {
            continue;
        }
        let root = uf.find(prov);
        let mut compact = remap[root as usize];
        if compact == 0 {
            num_components += 1;
            compact = num_components;
            remap[root as usize] = compact;
            sizes.push(0);
        }
        labels[idx] = compact;
        sizes[compact as usize - 1] += 1;
    }

    Ok(ComponentLabels {
        labels,
        num_components: num_components as usize,
        height: h,
        width: w,
        sizes,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn set_block(img: &mut [f32], w: usize, r0: usize, r1: usize, c0: usize, c1: usize) {
        for y in r0..r1 {
            for x in c0..c1 {
                img[y * w + x] = 1.0;
            }
        }
    }

    #[test]
    fn single_blob_one_component() {
        let mut img = vec![0.0_f32; 8 * 8];
        set_block(&mut img, 8, 2, 5, 2, 5); // 3×3 block
        let cc = connected_components(&img, 8, 8, Connectivity::Eight).expect("cc");
        assert_eq!(cc.num_components, 1);
        assert_eq!(cc.size_of(1), 9);
    }

    #[test]
    fn two_disjoint_blobs_two_components() {
        let mut img = vec![0.0_f32; 8 * 8];
        set_block(&mut img, 8, 0, 2, 0, 2);
        set_block(&mut img, 8, 5, 7, 5, 7);
        let cc = connected_components(&img, 8, 8, Connectivity::Eight).expect("cc");
        assert_eq!(cc.num_components, 2);
        assert_eq!(cc.size_of(1), 4);
        assert_eq!(cc.size_of(2), 4);
    }

    #[test]
    fn diagonal_pair_four_vs_eight_connectivity() {
        // Two pixels touching only at a corner.
        let mut img = vec![0.0_f32; 4 * 4];
        let at = |y: usize, x: usize| y * 4 + x;
        img[at(1, 1)] = 1.0;
        img[at(2, 2)] = 1.0;
        let cc4 = connected_components(&img, 4, 4, Connectivity::Four).expect("cc4");
        assert_eq!(cc4.num_components, 2, "4-connectivity splits the diagonal");
        let cc8 = connected_components(&img, 4, 4, Connectivity::Eight).expect("cc8");
        assert_eq!(cc8.num_components, 1, "8-connectivity joins the diagonal");
    }

    #[test]
    fn all_background_zero_components() {
        let img = vec![0.0_f32; 5 * 5];
        let cc = connected_components(&img, 5, 5, Connectivity::Eight).expect("cc");
        assert_eq!(cc.num_components, 0);
        assert!(cc.labels.iter().all(|&l| l == 0));
        assert!(cc.sizes().is_empty());
    }

    #[test]
    fn all_foreground_single_component() {
        let img = vec![1.0_f32; 5 * 5];
        let cc = connected_components(&img, 5, 5, Connectivity::Four).expect("cc");
        assert_eq!(cc.num_components, 1);
        assert_eq!(cc.size_of(1), 25);
    }

    #[test]
    fn u_shape_is_single_component() {
        // A U shape that requires equivalence merging across the two arms.
        let mut img = vec![0.0_f32; 5 * 5];
        // Left arm, right arm, and the bottom bar.
        for y in 0..5 {
            img[y * 5] = 1.0;
            img[y * 5 + 4] = 1.0;
        }
        for x in 0..5 {
            img[4 * 5 + x] = 1.0;
        }
        let cc = connected_components(&img, 5, 5, Connectivity::Four).expect("cc");
        assert_eq!(cc.num_components, 1, "the U is one connected region");
        assert_eq!(cc.size_of(1), 5 + 5 + 3); // two arms + bottom (corners shared)
    }

    #[test]
    fn sizes_sum_equals_foreground_count() {
        let mut img = vec![0.0_f32; 6 * 6];
        set_block(&mut img, 6, 0, 2, 0, 2);
        set_block(&mut img, 6, 3, 5, 3, 6);
        let foreground: usize = img.iter().filter(|&&v| v > 0.0).count();
        let cc = connected_components(&img, 6, 6, Connectivity::Eight).expect("cc");
        let total: usize = cc.sizes().iter().sum();
        assert_eq!(total, foreground);
    }

    #[test]
    fn bounding_boxes_match_blob_extent() {
        let mut img = vec![0.0_f32; 8 * 8];
        set_block(&mut img, 8, 2, 5, 3, 7); // rows 2..5, cols 3..7
        let cc = connected_components(&img, 8, 8, Connectivity::Eight).expect("cc");
        let boxes = cc.bounding_boxes();
        assert_eq!(boxes.len(), 1);
        // [x_min, y_min, x_max, y_max] inclusive.
        assert_eq!(boxes[0], [3, 2, 6, 4]);
    }

    #[test]
    fn size_of_out_of_range_is_zero() {
        let img = vec![1.0_f32; 4 * 4];
        let cc = connected_components(&img, 4, 4, Connectivity::Eight).expect("cc");
        assert_eq!(cc.size_of(0), 0);
        assert_eq!(cc.size_of(99), 0);
    }

    #[test]
    fn wrong_size_errors() {
        let img = vec![0.0_f32; 10];
        assert!(matches!(
            connected_components(&img, 8, 8, Connectivity::Four),
            Err(VisionError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn zero_dim_errors() {
        let img: Vec<f32> = vec![];
        assert!(matches!(
            connected_components(&img, 0, 4, Connectivity::Four),
            Err(VisionError::InvalidImageSize { .. })
        ));
    }

    #[test]
    fn checkerboard_four_connectivity_many_components() {
        // Each foreground pixel is isolated under 4-connectivity.
        let mut img = vec![0.0_f32; 4 * 4];
        let mut expected = 0;
        for y in 0..4 {
            for x in 0..4 {
                if (y + x) % 2 == 0 {
                    img[y * 4 + x] = 1.0;
                    expected += 1;
                }
            }
        }
        let cc = connected_components(&img, 4, 4, Connectivity::Four).expect("cc");
        assert_eq!(cc.num_components, expected);
        // Under 8-connectivity every foreground pixel touches diagonally.
        let cc8 = connected_components(&img, 4, 4, Connectivity::Eight).expect("cc8");
        assert_eq!(cc8.num_components, 1);
    }
}
