//! Tarjan's strongly-connected components, O(V + E).

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

const UNVISITED: usize = usize::MAX;

pub fn scc_tarjan(g: &AdjacencyList) -> GraphalgResult<Vec<Vec<usize>>> {
    let n = g.n;
    let mut index = 0usize;
    let mut idx = vec![UNVISITED; n];
    let mut low = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut sccs: Vec<Vec<usize>> = Vec::new();
    // Iterative DFS with state (node, child_iter).
    for start in 0..n {
        if idx[start] != UNVISITED {
            continue;
        }
        let mut work: Vec<(usize, usize)> = Vec::new();
        idx[start] = index;
        low[start] = index;
        index += 1;
        stack.push(start);
        on_stack[start] = true;
        work.push((start, 0));
        while let Some(&(u, i)) = work.last() {
            let adj = g.neighbors(u)?;
            if i < adj.len() {
                let v = adj[i];
                let last = work.len() - 1;
                work[last].1 = i + 1;
                if idx[v] == UNVISITED {
                    idx[v] = index;
                    low[v] = index;
                    index += 1;
                    stack.push(v);
                    on_stack[v] = true;
                    work.push((v, 0));
                } else if on_stack[v] {
                    low[u] = low[u].min(idx[v]);
                }
            } else {
                if low[u] == idx[u] {
                    let mut comp = Vec::new();
                    loop {
                        let w = match stack.pop() {
                            Some(x) => x,
                            None => break,
                        };
                        on_stack[w] = false;
                        comp.push(w);
                        if w == u {
                            break;
                        }
                    }
                    sccs.push(comp);
                }
                work.pop();
                if let Some(&(parent, _)) = work.last() {
                    low[parent] = low[parent].min(low[u]);
                }
            }
        }
    }
    Ok(sccs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tarjan_two_components() {
        let mut g = AdjacencyList::new(4);
        g.add_edge(0, 1).expect("ok");
        g.add_edge(1, 2).expect("ok");
        g.add_edge(2, 0).expect("ok");
        g.add_edge(3, 1).expect("ok");
        let sccs = scc_tarjan(&g).expect("ok");
        assert_eq!(sccs.len(), 2);
        let sizes: Vec<usize> = sccs.iter().map(|c| c.len()).collect();
        assert!(sizes.contains(&3));
        assert!(sizes.contains(&1));
    }

    #[test]
    fn tarjan_self_loop() {
        let mut g = AdjacencyList::new(2);
        g.add_edge(0, 0).expect("ok");
        g.add_edge(0, 1).expect("ok");
        let sccs = scc_tarjan(&g).expect("ok");
        assert_eq!(sccs.len(), 2);
    }
}
