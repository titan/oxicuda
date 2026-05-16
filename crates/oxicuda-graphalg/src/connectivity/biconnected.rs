//! Biconnected components (edge-disjoint blocks of an undirected graph).

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

const UNVISITED: usize = usize::MAX;

pub fn biconnected_components(g: &AdjacencyList) -> GraphalgResult<Vec<Vec<(usize, usize)>>> {
    let n = g.n;
    let mut disc = vec![UNVISITED; n];
    let mut low = vec![0usize; n];
    let mut parent = vec![UNVISITED; n];
    let mut edge_stack: Vec<(usize, usize)> = Vec::new();
    let mut bcc: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut time = 0usize;
    for start in 0..n {
        if disc[start] != UNVISITED {
            continue;
        }
        disc[start] = time;
        low[start] = time;
        time += 1;
        parent[start] = start;
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
        while let Some(&(u, i)) = stack.last() {
            let adj = g.neighbors(u)?;
            if i < adj.len() {
                let v = adj[i];
                let last = stack.len() - 1;
                stack[last].1 = i + 1;
                if disc[v] == UNVISITED {
                    parent[v] = u;
                    disc[v] = time;
                    low[v] = time;
                    time += 1;
                    edge_stack.push((u.min(v), u.max(v)));
                    stack.push((v, 0));
                } else if v != parent[u] && disc[v] < disc[u] {
                    edge_stack.push((u.min(v), u.max(v)));
                    low[u] = low[u].min(disc[v]);
                }
            } else {
                let par = parent[u];
                if par != u {
                    low[par] = low[par].min(low[u]);
                    if low[u] >= disc[par] {
                        let mut cur = Vec::new();
                        while let Some(e) = edge_stack.pop() {
                            let target = (par.min(u), par.max(u));
                            cur.push(e);
                            if e == target {
                                break;
                            }
                        }
                        if !cur.is_empty() {
                            bcc.push(cur);
                        }
                    }
                }
                stack.pop();
            }
        }
    }
    Ok(bcc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bcc_single_triangle() {
        let mut g = AdjacencyList::new(3);
        g.add_undirected_edge(0, 1).expect("ok");
        g.add_undirected_edge(1, 2).expect("ok");
        g.add_undirected_edge(0, 2).expect("ok");
        let bcc = biconnected_components(&g).expect("ok");
        assert_eq!(bcc.len(), 1);
    }

    #[test]
    fn bcc_two_triangles_joined() {
        // 0-1-2-0 and 2-3-4-2 share vertex 2
        let mut g = AdjacencyList::new(5);
        g.add_undirected_edge(0, 1).expect("ok");
        g.add_undirected_edge(1, 2).expect("ok");
        g.add_undirected_edge(0, 2).expect("ok");
        g.add_undirected_edge(2, 3).expect("ok");
        g.add_undirected_edge(3, 4).expect("ok");
        g.add_undirected_edge(2, 4).expect("ok");
        let bcc = biconnected_components(&g).expect("ok");
        assert_eq!(bcc.len(), 2);
    }
}
