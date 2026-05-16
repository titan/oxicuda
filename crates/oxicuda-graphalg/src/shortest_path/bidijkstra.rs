//! Bidirectional Dijkstra (meet in the middle).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::weighted_graph::WeightedGraph;

#[derive(Debug, Clone)]
pub struct BiDijkstraOutput {
    pub dist: f64,
    pub path: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Item {
    dist: f64,
    node: usize,
}
impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Item {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.node.cmp(&self.node))
    }
}

pub fn bidirectional_dijkstra(
    g: &WeightedGraph,
    source: usize,
    target: usize,
) -> GraphalgResult<BiDijkstraOutput> {
    if source >= g.n || target >= g.n {
        return Err(GraphalgError::SourceOutOfRange {
            node: source.max(target),
            n: g.n,
        });
    }
    if source == target {
        return Ok(BiDijkstraOutput {
            dist: 0.0,
            path: vec![source],
        });
    }
    let rev = g.reverse();
    let mut df = vec![f64::INFINITY; g.n];
    let mut db = vec![f64::INFINITY; g.n];
    let mut pf = vec![usize::MAX; g.n];
    let mut pb = vec![usize::MAX; g.n];
    df[source] = 0.0;
    db[target] = 0.0;
    let mut hf: BinaryHeap<Item> = BinaryHeap::new();
    let mut hb: BinaryHeap<Item> = BinaryHeap::new();
    hf.push(Item {
        dist: 0.0,
        node: source,
    });
    hb.push(Item {
        dist: 0.0,
        node: target,
    });
    let mut best = f64::INFINITY;
    let mut meet = usize::MAX;
    while !hf.is_empty() || !hb.is_empty() {
        // expand smaller frontier
        let f_top = hf.peek().map(|i| i.dist).unwrap_or(f64::INFINITY);
        let b_top = hb.peek().map(|i| i.dist).unwrap_or(f64::INFINITY);
        if f_top + b_top >= best {
            break;
        }
        if f_top <= b_top {
            if let Some(Item { dist: d, node: u }) = hf.pop() {
                if d > df[u] {
                    continue;
                }
                for &(v, w) in g.neighbors(u)? {
                    if w < 0.0 {
                        return Err(GraphalgError::NegativeWeight {
                            edge: (u, v),
                            weight: w,
                        });
                    }
                    let nd = d + w;
                    if nd < df[v] {
                        df[v] = nd;
                        pf[v] = u;
                        hf.push(Item { dist: nd, node: v });
                        if db[v].is_finite() && df[v] + db[v] < best {
                            best = df[v] + db[v];
                            meet = v;
                        }
                    }
                }
            }
        } else if let Some(Item { dist: d, node: u }) = hb.pop() {
            if d > db[u] {
                continue;
            }
            for &(v, w) in rev.neighbors(u)? {
                if w < 0.0 {
                    return Err(GraphalgError::NegativeWeight {
                        edge: (v, u),
                        weight: w,
                    });
                }
                let nd = d + w;
                if nd < db[v] {
                    db[v] = nd;
                    pb[v] = u;
                    hb.push(Item { dist: nd, node: v });
                    if df[v].is_finite() && df[v] + db[v] < best {
                        best = df[v] + db[v];
                        meet = v;
                    }
                }
            }
        }
    }
    if !best.is_finite() {
        return Err(GraphalgError::NoSolution(format!(
            "no path from {source} to {target}"
        )));
    }
    let mut left = Vec::new();
    let mut cur = meet;
    while cur != source {
        left.push(cur);
        let p = pf[cur];
        if p == usize::MAX {
            return Err(GraphalgError::NumericalInstability(
                "forward parent missing".to_string(),
            ));
        }
        cur = p;
    }
    left.push(source);
    left.reverse();
    let mut cur = meet;
    while cur != target {
        let p = pb[cur];
        if p == usize::MAX {
            return Err(GraphalgError::NumericalInstability(
                "backward parent missing".to_string(),
            ));
        }
        cur = p;
        left.push(cur);
    }
    Ok(BiDijkstraOutput {
        dist: best,
        path: left,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bidir_dijkstra_path() {
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(0, 2, 4.0).expect("ok");
        g.add_edge(1, 2, 2.0).expect("ok");
        g.add_edge(1, 3, 5.0).expect("ok");
        g.add_edge(2, 3, 1.0).expect("ok");
        let out = bidirectional_dijkstra(&g, 0, 3).expect("ok");
        assert!((out.dist - 4.0).abs() < 1e-12);
    }
}
