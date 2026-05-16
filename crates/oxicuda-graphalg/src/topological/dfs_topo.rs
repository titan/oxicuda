//! DFS-based topological sort (reverse post-order). Detects cycles via 3-color marking.

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::adjacency_list::AdjacencyList;

const WHITE: u8 = 0;
const GRAY: u8 = 1;
const BLACK: u8 = 2;

pub fn topo_sort_dfs(g: &AdjacencyList) -> GraphalgResult<Vec<usize>> {
    let mut color = vec![WHITE; g.n];
    let mut order: Vec<usize> = Vec::with_capacity(g.n);
    // Iterative DFS with state on stack: (node, child_index).
    for start in 0..g.n {
        if color[start] != WHITE {
            continue;
        }
        color[start] = GRAY;
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
        while let Some(&(u, i)) = stack.last() {
            let adj = g.neighbors(u)?;
            if i < adj.len() {
                let v = adj[i];
                let last = stack.len() - 1;
                stack[last].1 = i + 1;
                match color[v] {
                    WHITE => {
                        color[v] = GRAY;
                        stack.push((v, 0));
                    }
                    GRAY => {
                        return Err(GraphalgError::NotADag);
                    }
                    _ => {}
                }
            } else {
                color[u] = BLACK;
                order.push(u);
                stack.pop();
            }
        }
    }
    order.reverse();
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topo_dfs_chain() {
        let mut g = AdjacencyList::new(4);
        g.add_edge(0, 1).expect("ok");
        g.add_edge(1, 2).expect("ok");
        g.add_edge(2, 3).expect("ok");
        let o = topo_sort_dfs(&g).expect("ok");
        // Find positions of each
        let pos = |x: usize| o.iter().position(|&y| y == x).expect("ok");
        assert!(pos(0) < pos(1));
        assert!(pos(1) < pos(2));
        assert!(pos(2) < pos(3));
    }

    #[test]
    fn topo_dfs_cycle_err() {
        let mut g = AdjacencyList::new(3);
        g.add_edge(0, 1).expect("ok");
        g.add_edge(1, 2).expect("ok");
        g.add_edge(2, 0).expect("ok");
        assert!(topo_sort_dfs(&g).is_err());
    }
}
