//! VF2 subgraph isomorphism (Cordella et al.). Returns the first found mapping.

use crate::error::GraphalgResult;
use crate::repr::adjacency_list::AdjacencyList;

/// Returns `Some(mapping)` such that `mapping[g1_node] = g2_node` is an isomorphism
/// from `g1` into `g2` (g1 must have at most as many nodes as g2).
/// Returns `None` if no isomorphism is found.
pub fn vf2_isomorphism(
    g1: &AdjacencyList,
    g2: &AdjacencyList,
) -> GraphalgResult<Option<Vec<usize>>> {
    if g1.n > g2.n {
        return Ok(None);
    }
    let mut mapping = vec![usize::MAX; g1.n];
    let mut inverse = vec![usize::MAX; g2.n];
    let found = match_recursive(g1, g2, &mut mapping, &mut inverse, 0)?;
    if found { Ok(Some(mapping)) } else { Ok(None) }
}

fn match_recursive(
    g1: &AdjacencyList,
    g2: &AdjacencyList,
    mapping: &mut [usize],
    inverse: &mut [usize],
    depth: usize,
) -> GraphalgResult<bool> {
    if depth == g1.n {
        return Ok(true);
    }
    let u = depth; // simple ordering: match nodes 0..n in order
    for v in 0..g2.n {
        if inverse[v] != usize::MAX {
            continue;
        }
        if feasible(g1, g2, u, v, mapping, inverse)? {
            mapping[u] = v;
            inverse[v] = u;
            if match_recursive(g1, g2, mapping, inverse, depth + 1)? {
                return Ok(true);
            }
            mapping[u] = usize::MAX;
            inverse[v] = usize::MAX;
        }
    }
    Ok(false)
}

fn feasible(
    g1: &AdjacencyList,
    g2: &AdjacencyList,
    u: usize,
    v: usize,
    mapping: &[usize],
    _inverse: &[usize],
) -> GraphalgResult<bool> {
    // For each previously-mapped neighbor u' of u, mapping[u'] must be a neighbor of v in g2.
    for &up in g1.neighbors(u)? {
        if up < u && mapping[up] != usize::MAX {
            let vp = mapping[up];
            if !g2.neighbors(v)?.contains(&vp) {
                return Ok(false);
            }
        }
    }
    // And the converse direction for symmetry.
    for &vp in g2.neighbors(v)? {
        // map back
        let mut found_up = usize::MAX;
        for (uu, &mv) in mapping.iter().enumerate() {
            if mv == vp {
                found_up = uu;
                break;
            }
        }
        if found_up != usize::MAX && found_up < u && !g1.neighbors(u)?.contains(&found_up) {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vf2_k3_eq_k3() {
        let mut g1 = AdjacencyList::new(3);
        g1.add_undirected_edge(0, 1).expect("ok");
        g1.add_undirected_edge(1, 2).expect("ok");
        g1.add_undirected_edge(0, 2).expect("ok");
        let g2 = g1.clone();
        let m = vf2_isomorphism(&g1, &g2).expect("ok");
        assert!(m.is_some());
    }

    #[test]
    fn vf2_no_match() {
        let mut g1 = AdjacencyList::new(3);
        g1.add_undirected_edge(0, 1).expect("ok");
        g1.add_undirected_edge(1, 2).expect("ok");
        let mut g2 = AdjacencyList::new(3); // empty
        let _ = &mut g2;
        let m = vf2_isomorphism(&g1, &g2).expect("ok");
        assert!(m.is_none());
    }
}
