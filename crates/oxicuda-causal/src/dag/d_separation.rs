use super::dag::Dag;
use std::collections::{HashSet, VecDeque};

/// Bayes-ball algorithm for d-separation.
/// Returns true if X and Y are d-separated given the set Z.
pub fn d_separated(dag: &Dag, x: usize, y: usize, z: &[usize]) -> bool {
    let z_set: HashSet<usize> = z.iter().copied().collect();

    // Precompute ancestors of Z (needed for v-structure active paths)
    let mut anc_z: HashSet<usize> = z_set.clone();
    for &zv in z {
        for anc in dag.ancestors(zv) {
            anc_z.insert(anc);
        }
    }

    // BFS state: (node, via_child)
    // via_child=true means we arrived from a child (traveling "up" toward parents)
    // via_child=false means we arrived from a parent (traveling "down" toward children)
    let mut visited: HashSet<(usize, bool)> = HashSet::new();
    let mut queue: VecDeque<(usize, bool)> = VecDeque::new();

    // Start from X going both directions
    queue.push_back((x, true)); // arrived via child (going up)
    queue.push_back((x, false)); // arrived via parent (going down)

    while let Some((node, via_child)) = queue.pop_front() {
        if visited.contains(&(node, via_child)) {
            continue;
        }
        visited.insert((node, via_child));

        if node == y {
            return false; // y is reachable, not d-separated
        }

        let is_observed = z_set.contains(&node);

        if via_child && !is_observed {
            // Active path going upward: propagate to parents
            for &parent in dag.parents(node) {
                let state = (parent, true);
                if !visited.contains(&state) {
                    queue.push_back(state);
                }
            }
            // Also propagate downward (chain/fork)
            for &child in dag.children(node) {
                let state = (child, false);
                if !visited.contains(&state) {
                    queue.push_back(state);
                }
            }
        } else if !via_child {
            // Active path going downward
            if !is_observed {
                // Propagate downward (chain)
                for &child in dag.children(node) {
                    let state = (child, false);
                    if !visited.contains(&state) {
                        queue.push_back(state);
                    }
                }
            }
            // V-structure: node is collider; active if node or descendant is observed
            if is_observed || anc_z.contains(&node) {
                for &parent in dag.parents(node) {
                    let state = (parent, true);
                    if !visited.contains(&state) {
                        queue.push_back(state);
                    }
                }
            }
        }
    }

    true // y not reachable => d-separated
}

#[cfg(test)]
mod tests {
    use super::super::dag::Dag;
    use super::*;

    #[test]
    fn chain_d_separation() {
        // X -> Z -> Y: X d-sep Y given Z
        let mut dag = Dag::new(3);
        dag.add_edge(0, 2).expect("add_edge should succeed");
        dag.add_edge(2, 1).expect("add_edge should succeed");
        assert!(d_separated(&dag, 0, 1, &[2]));
        assert!(!d_separated(&dag, 0, 1, &[]));
    }

    #[test]
    fn fork_d_separation() {
        // X <- Z -> Y: X d-sep Y given Z
        let mut dag = Dag::new(3);
        dag.add_edge(2, 0).expect("add_edge should succeed");
        dag.add_edge(2, 1).expect("add_edge should succeed");
        assert!(d_separated(&dag, 0, 1, &[2]));
        assert!(!d_separated(&dag, 0, 1, &[]));
    }

    #[test]
    fn collider_d_separation() {
        // X -> Z <- Y: X and Y independent, but dependent given Z
        let mut dag = Dag::new(3);
        dag.add_edge(0, 2).expect("add_edge should succeed");
        dag.add_edge(1, 2).expect("add_edge should succeed");
        assert!(d_separated(&dag, 0, 1, &[]));
        assert!(!d_separated(&dag, 0, 1, &[2]));
    }
}
