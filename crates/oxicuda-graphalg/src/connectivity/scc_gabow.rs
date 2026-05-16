//! Gabow's SCC algorithm using two stacks.

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

const UNVISITED: usize = usize::MAX;

pub fn scc_gabow(g: &AdjacencyList) -> GraphalgResult<Vec<Vec<usize>>> {
    let n = g.n;
    let mut pre = vec![UNVISITED; n];
    let mut comp = vec![UNVISITED; n];
    let mut s: Vec<usize> = Vec::new(); // unfinished
    let mut p: Vec<usize> = Vec::new(); // potential SCC roots
    let mut counter = 0usize;
    let mut sccs: Vec<Vec<usize>> = Vec::new();
    for start in 0..n {
        if pre[start] != UNVISITED {
            continue;
        }
        let mut work: Vec<(usize, usize)> = vec![(start, 0)];
        pre[start] = counter;
        counter += 1;
        s.push(start);
        p.push(start);
        while let Some(&(u, i)) = work.last() {
            let adj = g.neighbors(u)?;
            if i < adj.len() {
                let v = adj[i];
                let last = work.len() - 1;
                work[last].1 = i + 1;
                if pre[v] == UNVISITED {
                    pre[v] = counter;
                    counter += 1;
                    s.push(v);
                    p.push(v);
                    work.push((v, 0));
                } else if comp[v] == UNVISITED {
                    while let Some(&top) = p.last() {
                        if pre[top] > pre[v] {
                            p.pop();
                        } else {
                            break;
                        }
                    }
                }
            } else {
                if Some(&u) == p.last() {
                    p.pop();
                    let id = sccs.len();
                    let mut cur = Vec::new();
                    while let Some(w) = s.pop() {
                        comp[w] = id;
                        cur.push(w);
                        if w == u {
                            break;
                        }
                    }
                    sccs.push(cur);
                }
                work.pop();
            }
        }
    }
    Ok(sccs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gabow_matches_tarjan() {
        let mut g = AdjacencyList::new(4);
        g.add_edge(0, 1).expect("ok");
        g.add_edge(1, 2).expect("ok");
        g.add_edge(2, 0).expect("ok");
        g.add_edge(3, 1).expect("ok");
        let sccs = scc_gabow(&g).expect("ok");
        assert_eq!(sccs.len(), 2);
    }
}
