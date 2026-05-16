//! 2D R-tree with STR (Sort-Tile-Recursive) bulk loading and AABB queries.

use crate::primitives::aabb::Aabb;
use crate::primitives::point::Point;

const MAX_ENTRIES: usize = 8;

#[derive(Debug, Clone)]
enum RNode {
    Leaf {
        bbox: Aabb,
        entries: Vec<(Aabb, usize)>,
    },
    Internal {
        bbox: Aabb,
        children: Vec<RNode>,
    },
}

impl RNode {
    fn bbox(&self) -> Aabb {
        match self {
            RNode::Leaf { bbox, .. } => *bbox,
            RNode::Internal { bbox, .. } => *bbox,
        }
    }
}

/// 2D R-tree.
#[derive(Debug, Clone)]
pub struct Rtree2d {
    root: Option<RNode>,
    n: usize,
}

impl Rtree2d {
    /// Bulk load entries via STR.
    #[must_use]
    pub fn build(entries: Vec<(Aabb, usize)>) -> Self {
        let n = entries.len();
        if entries.is_empty() {
            return Self { root: None, n };
        }
        let leaves = build_leaves(entries);
        let root = build_levels(leaves);
        Self {
            root: Some(root),
            n,
        }
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Range query: collect entries whose bbox intersects `range`.
    #[must_use]
    pub fn search(&self, range: Aabb) -> Vec<usize> {
        let mut out = Vec::new();
        if let Some(root) = &self.root {
            search_rec(root, range, &mut out);
        }
        out
    }

    /// Query whether a point intersects any entry.
    #[must_use]
    pub fn point_query(&self, q: Point) -> Vec<usize> {
        self.search(Aabb::new(q, q))
    }
}

fn build_leaves(mut entries: Vec<(Aabb, usize)>) -> Vec<RNode> {
    entries.sort_by(|a, b| {
        a.0.center()
            .x
            .partial_cmp(&b.0.center().x)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let total = entries.len();
    let leaf_count = total.div_ceil(MAX_ENTRIES);
    let stripes = (leaf_count as f64).sqrt().ceil() as usize;
    let stripes = stripes.max(1);
    let per_stripe = total.div_ceil(stripes);
    let mut leaves: Vec<RNode> = Vec::new();
    for s in 0..stripes {
        let start = s * per_stripe;
        let end = (start + per_stripe).min(total);
        if start >= end {
            continue;
        }
        let mut stripe_entries: Vec<(Aabb, usize)> = entries[start..end].to_vec();
        stripe_entries.sort_by(|a, b| {
            a.0.center()
                .y
                .partial_cmp(&b.0.center().y)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        for chunk in stripe_entries.chunks(MAX_ENTRIES) {
            let mut bbox = chunk[0].0;
            for &(b, _) in chunk.iter().skip(1) {
                bbox = bbox.union(b);
            }
            leaves.push(RNode::Leaf {
                bbox,
                entries: chunk.to_vec(),
            });
        }
    }
    leaves
}

fn build_levels(mut nodes: Vec<RNode>) -> RNode {
    while nodes.len() > 1 {
        nodes.sort_by(|a, b| {
            a.bbox()
                .center()
                .x
                .partial_cmp(&b.bbox().center().x)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        let mut next: Vec<RNode> = Vec::new();
        for chunk in nodes.chunks(MAX_ENTRIES) {
            let mut bbox = chunk[0].bbox();
            for c in chunk.iter().skip(1) {
                bbox = bbox.union(c.bbox());
            }
            next.push(RNode::Internal {
                bbox,
                children: chunk.to_vec(),
            });
        }
        nodes = next;
    }
    nodes.remove(0)
}

fn search_rec(node: &RNode, range: Aabb, out: &mut Vec<usize>) {
    if !node.bbox().intersects(range) {
        return;
    }
    match node {
        RNode::Leaf { entries, .. } => {
            for &(b, idx) in entries {
                if b.intersects(range) {
                    out.push(idx);
                }
            }
        }
        RNode::Internal { children, .. } => {
            for c in children {
                search_rec(c, range, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_and_query() {
        let entries: Vec<(Aabb, usize)> = (0..16)
            .map(|i| {
                let x = (i % 4) as f64;
                let y = (i / 4) as f64;
                (Aabb::new(Point::new(x, y), Point::new(x + 0.5, y + 0.5)), i)
            })
            .collect();
        let rt = Rtree2d::build(entries);
        let found = rt.search(Aabb::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0)));
        assert!(!found.is_empty());
    }

    #[test]
    fn empty_tree() {
        let rt = Rtree2d::build(vec![]);
        assert!(rt.is_empty());
        let f = rt.search(Aabb::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0)));
        assert!(f.is_empty());
    }
}
