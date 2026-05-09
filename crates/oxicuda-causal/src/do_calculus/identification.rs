use crate::dag::d_separation::d_separated;
use crate::dag::dag::Dag;
use std::collections::HashSet;

/// Backdoor criterion: Z satisfies backdoor relative to (X, Y) in dag.
/// 1. No element of Z is a descendant of X.
/// 2. Z blocks every path between X and Y that has an arrow INTO X.
///    Equivalent: in G_x_bar (remove outgoing arrows from X), Z d-separates X from Y.
pub fn backdoor_admissible(dag: &Dag, x: usize, y: usize, z: &[usize]) -> bool {
    let desc_x = dag.descendants(x);
    let desc_set: HashSet<usize> = desc_x.into_iter().collect();

    // Condition 1: no Z element is a descendant of X
    if z.iter().any(|&zv| desc_set.contains(&zv)) {
        return false;
    }

    // Condition 2: build G_x_bar = DAG with outgoing edges from X removed.
    // In G_x_bar, all paths from X to Y are blocked by removing X's children links.
    // Z must d-separate X and Y in G_x_bar — but since X has no outgoing edges,
    // the only paths remaining are backdoor paths (through X's parents).
    let n = dag.n;
    let mut g_x_bar = Dag::new(n);
    for from in 0..n {
        for &to in dag.children(from) {
            if from == x {
                continue; // remove outgoing edge from X
            }
            let _ = g_x_bar.add_edge(from, to);
        }
    }

    d_separated(&g_x_bar, x, y, z)
}

/// Frontdoor criterion: M satisfies frontdoor relative to (X, Y) in dag.
/// 1. M intercepts all directed paths from X to Y.
/// 2. No unblocked backdoor path from X to M.
/// 3. All backdoor paths from M to Y are blocked by X.
pub fn frontdoor_admissible(dag: &Dag, x: usize, y: usize, m: &[usize]) -> bool {
    if m.is_empty() {
        return false;
    }

    // Condition 1: every directed path from X to Y goes through some M element
    let m_set: HashSet<usize> = m.iter().copied().collect();
    if can_reach_avoiding(dag, x, y, &m_set) {
        return false;
    }

    // Condition 2: no backdoor from X to any M element (empty set blocks)
    for &mi in m {
        if !backdoor_admissible(dag, x, mi, &[]) {
            return false;
        }
    }

    // Condition 3: all backdoor paths from M to Y are blocked by X
    for &mi in m {
        if !backdoor_admissible(dag, mi, y, &[x]) {
            return false;
        }
    }

    true
}

fn can_reach_avoiding(dag: &Dag, start: usize, target: usize, avoid: &HashSet<usize>) -> bool {
    let mut visited = HashSet::new();
    let mut stack = vec![start];
    while let Some(cur) = stack.pop() {
        if cur == target {
            return true;
        }
        if !visited.insert(cur) {
            continue;
        }
        for &child in dag.children(cur) {
            if !avoid.contains(&child) && !visited.contains(&child) {
                stack.push(child);
            }
        }
    }
    false
}

/// Find a minimal backdoor adjustment set for (X, Y).
/// Returns None if no valid adjustment set found.
pub fn backdoor_adjustment(dag: &Dag, x: usize, y: usize) -> Option<Vec<usize>> {
    // Try empty set first
    if backdoor_admissible(dag, x, y, &[]) {
        return Some(vec![]);
    }

    // Try parents of X
    let parents_x = dag.parents(x).to_vec();
    if backdoor_admissible(dag, x, y, &parents_x) {
        return Some(parents_x);
    }

    // Try all non-descendant, non-x, non-y nodes
    let desc_x: HashSet<usize> = dag.descendants(x).into_iter().collect();
    let candidates: Vec<usize> = (0..dag.n)
        .filter(|&v| v != x && v != y && !desc_x.contains(&v))
        .collect();

    if backdoor_admissible(dag, x, y, &candidates) {
        return Some(candidates);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::dag::Dag;

    #[test]
    fn backdoor_chain() {
        // X -> Z -> Y: no confounders, empty set is valid
        let mut dag = Dag::new(3);
        dag.add_edge(0, 2).unwrap();
        dag.add_edge(2, 1).unwrap();
        assert!(backdoor_admissible(&dag, 0, 1, &[]));
    }

    #[test]
    fn backdoor_with_confounder() {
        // C -> X, C -> Y, X -> Y: need to adjust for C
        let mut dag = Dag::new(3);
        let c = 2;
        dag.add_edge(c, 0).unwrap(); // C -> X
        dag.add_edge(c, 1).unwrap(); // C -> Y
        dag.add_edge(0, 1).unwrap(); // X -> Y
        assert!(!backdoor_admissible(&dag, 0, 1, &[]));
        assert!(backdoor_admissible(&dag, 0, 1, &[c]));
    }
}
