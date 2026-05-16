//! Brandes 2001 algorithm for betweenness centrality (unweighted, O(VE)).

use std::collections::VecDeque;

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

pub fn betweenness_centrality(g: &AdjacencyList, normalized: bool) -> GraphalgResult<Vec<f64>> {
    let n = g.n;
    let mut cb = vec![0.0f64; n];
    for s in 0..n {
        let mut stack: Vec<usize> = Vec::new();
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut sigma = vec![0.0f64; n];
        sigma[s] = 1.0;
        let mut dist = vec![-1i64; n];
        dist[s] = 0;
        let mut q: VecDeque<usize> = VecDeque::new();
        q.push_back(s);
        while let Some(v) = q.pop_front() {
            stack.push(v);
            for &w in g.neighbors(v)? {
                if dist[w] < 0 {
                    dist[w] = dist[v] + 1;
                    q.push_back(w);
                }
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    preds[w].push(v);
                }
            }
        }
        let mut delta = vec![0.0f64; n];
        while let Some(w) = stack.pop() {
            for &v in &preds[w] {
                if sigma[w] > 0.0 {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
            }
            if w != s {
                cb[w] += delta[w];
            }
        }
    }
    if normalized && n > 2 {
        let scale = 1.0 / ((n - 1) as f64 * (n - 2) as f64);
        for v in cb.iter_mut() {
            *v *= scale;
        }
    }
    Ok(cb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn betweenness_line_middle() {
        // Path 0-1-2-3-4 -- middle node has highest betweenness.
        let mut g = AdjacencyList::new(5);
        for i in 0..4 {
            g.add_undirected_edge(i, i + 1).expect("ok");
        }
        let b = betweenness_centrality(&g, false).expect("ok");
        assert!(b[2] >= b[1]);
        assert!(b[2] >= b[3]);
    }
}
