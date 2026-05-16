//! Hopcroft-Karp bipartite matching, O(E * sqrt(V)).

use std::collections::VecDeque;

use crate::error::{GraphalgError, GraphalgResult};

/// Bipartite graph with left partition `l` and right partition `r`. Edges by left index.
#[derive(Debug, Clone)]
pub struct BipartiteGraph {
    pub n_left: usize,
    pub n_right: usize,
    pub adj_left: Vec<Vec<usize>>,
}

impl BipartiteGraph {
    pub fn new(n_left: usize, n_right: usize) -> Self {
        Self {
            n_left,
            n_right,
            adj_left: vec![Vec::new(); n_left],
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize) -> GraphalgResult<()> {
        if u >= self.n_left {
            return Err(GraphalgError::IndexOutOfBounds {
                index: u,
                len: self.n_left,
            });
        }
        if v >= self.n_right {
            return Err(GraphalgError::IndexOutOfBounds {
                index: v,
                len: self.n_right,
            });
        }
        self.adj_left[u].push(v);
        Ok(())
    }
}

const NIL: usize = usize::MAX;

/// Returns left->right matching (or NIL if unmatched) and the matching size.
pub fn hopcroft_karp_matching(g: &BipartiteGraph) -> GraphalgResult<(Vec<usize>, usize)> {
    let l = g.n_left;
    let r = g.n_right;
    let mut pair_l = vec![NIL; l];
    let mut pair_r = vec![NIL; r];
    let mut dist = vec![0i64; l + 1];
    let inf: i64 = i64::MAX;
    let mut matching = 0usize;
    loop {
        // BFS
        let mut q: VecDeque<usize> = VecDeque::new();
        for u in 0..l {
            if pair_l[u] == NIL {
                dist[u] = 0;
                q.push_back(u);
            } else {
                dist[u] = inf;
            }
        }
        let mut found = inf;
        while let Some(u) = q.pop_front() {
            if dist[u] < found {
                for &v in &g.adj_left[u] {
                    let pu = pair_r[v];
                    let d_pu = if pu == NIL { found } else { dist[pu] };
                    if d_pu == inf {
                        if pu == NIL {
                            found = dist[u] + 1;
                        } else {
                            dist[pu] = dist[u] + 1;
                            q.push_back(pu);
                        }
                    }
                }
            }
        }
        if found == inf {
            break;
        }
        for u in 0..l {
            if pair_l[u] == NIL && dfs_aug(u, g, &mut pair_l, &mut pair_r, &mut dist, found) {
                matching += 1;
            }
        }
    }
    Ok((pair_l, matching))
}

fn dfs_aug(
    u: usize,
    g: &BipartiteGraph,
    pair_l: &mut [usize],
    pair_r: &mut [usize],
    dist: &mut [i64],
    found_dist: i64,
) -> bool {
    let inf: i64 = i64::MAX;
    for i in 0..g.adj_left[u].len() {
        let v = g.adj_left[u][i];
        let pu = pair_r[v];
        let d_pu = if pu == NIL { found_dist } else { dist[pu] };
        if d_pu == dist[u] + 1 {
            if pu == NIL || dfs_aug(pu, g, pair_l, pair_r, dist, found_dist) {
                pair_r[v] = u;
                pair_l[u] = v;
                return true;
            }
        }
    }
    dist[u] = inf;
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hk_perfect_3x3() {
        let mut g = BipartiteGraph::new(3, 3);
        g.add_edge(0, 0).expect("ok");
        g.add_edge(0, 1).expect("ok");
        g.add_edge(1, 0).expect("ok");
        g.add_edge(1, 2).expect("ok");
        g.add_edge(2, 1).expect("ok");
        g.add_edge(2, 2).expect("ok");
        let (_pair, m) = hopcroft_karp_matching(&g).expect("ok");
        assert_eq!(m, 3);
    }

    #[test]
    fn hk_partial() {
        let mut g = BipartiteGraph::new(2, 2);
        g.add_edge(0, 0).expect("ok");
        g.add_edge(1, 0).expect("ok");
        let (_pair, m) = hopcroft_karp_matching(&g).expect("ok");
        assert_eq!(m, 1);
    }
}
