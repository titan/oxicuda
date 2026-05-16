//! End-to-end integration tests for `oxicuda-graphalg`.

use crate::arborescence::chu_liu_edmonds::chu_liu_edmonds;
use crate::centrality::betweenness_brandes::betweenness_centrality;
use crate::centrality::closeness::closeness_centrality;
use crate::centrality::eigenvector::eigenvector_centrality;
use crate::centrality::pagerank::pagerank;
use crate::coloring::dsatur::dsatur_coloring;
use crate::community::label_propagation::label_propagation;
use crate::community::louvain::louvain_communities;
use crate::connectivity::articulation_points::articulation_points;
use crate::connectivity::bridges_tarjan::bridges_tarjan;
use crate::connectivity::scc_kosaraju::scc_kosaraju;
use crate::connectivity::scc_tarjan::scc_tarjan;
use crate::eulerian::hierholzer::eulerian_circuit_hierholzer;
use crate::hamiltonian::held_karp_dp::held_karp_tsp;
use crate::isomorphism::vf2::vf2_isomorphism;
use crate::matching::hopcroft_karp::{BipartiteGraph, hopcroft_karp_matching};
use crate::matching::hungarian_munkres::hungarian_assignment;
use crate::max_flow::edmonds_karp::{FlowNetwork, edmonds_karp};
use crate::max_flow::min_cut::min_cut_from_max_flow;
use crate::metrics::metrics::{clustering_coefficient_global, diameter, radius};
use crate::mst::kruskal::kruskal_mst;
use crate::mst::prim::prim_mst;
use crate::ptx_kernels::{
    bfs_level_ptx, community_label_ptx, csr_spmv_bool_ptx, dijkstra_relax_ptx, fw_inner_ptx,
    pagerank_step_ptx, triangle_count_ptx,
};
use crate::repr::adjacency_list::AdjacencyList;
use crate::repr::weighted_graph::WeightedGraph;
use crate::shortest_path::a_star::a_star;
use crate::shortest_path::bellman_ford::bellman_ford;
use crate::shortest_path::dijkstra::dijkstra;
use crate::shortest_path::floyd_warshall::floyd_warshall;
use crate::topological::kahn::topo_sort_kahn;
use crate::traversal::bfs::bfs_levels;
use crate::traversal::dfs::dfs_iterative;
use crate::tsp::nearest_neighbor::nearest_neighbor_tour;

// 1. BFS on a line graph gives correct distances 0,1,2,...
#[test]
fn e2e_bfs_line() {
    let mut g = AdjacencyList::new(6);
    for i in 0..5 {
        g.add_undirected_edge(i, i + 1).expect("ok");
    }
    let lv = bfs_levels(&g, 0).expect("ok");
    assert_eq!(lv, vec![0, 1, 2, 3, 4, 5]);
}

// 2. DFS on a tree visits all nodes
#[test]
fn e2e_dfs_tree() {
    let mut g = AdjacencyList::new(7);
    for i in 0..6 {
        g.add_undirected_edge(i, i + 1).expect("ok");
    }
    let order = dfs_iterative(&g, 0).expect("ok");
    let mut s = order;
    s.sort_unstable();
    assert_eq!(s, vec![0, 1, 2, 3, 4, 5, 6]);
}

// 3. Dijkstra on a 4-node weighted graph recovers shortest path
#[test]
fn e2e_dijkstra_4node() {
    let mut g = WeightedGraph::new(4);
    g.add_edge(0, 1, 1.0).expect("ok");
    g.add_edge(0, 2, 4.0).expect("ok");
    g.add_edge(1, 2, 2.0).expect("ok");
    g.add_edge(1, 3, 5.0).expect("ok");
    g.add_edge(2, 3, 1.0).expect("ok");
    let out = dijkstra(&g, 0).expect("ok");
    assert!((out.dist[3] - 4.0).abs() < 1e-12);
}

// 4. Bellman-Ford detects negative cycles
#[test]
fn e2e_bellman_ford_neg_cycle() {
    let mut g = WeightedGraph::new(3);
    g.add_edge(0, 1, 1.0).expect("ok");
    g.add_edge(1, 2, -3.0).expect("ok");
    g.add_edge(2, 0, 1.0).expect("ok");
    assert!(bellman_ford(&g, 0).is_err());
}

// 5. Floyd-Warshall matches Dijkstra on positive graph
#[test]
fn e2e_fw_matches_dijkstra() {
    let mut g = WeightedGraph::new(4);
    g.add_undirected_edge(0, 1, 1.0).expect("ok");
    g.add_undirected_edge(1, 2, 2.0).expect("ok");
    g.add_undirected_edge(0, 2, 5.0).expect("ok");
    g.add_undirected_edge(2, 3, 1.0).expect("ok");
    let fw = floyd_warshall(&g).expect("ok");
    for src in 0..4 {
        let d = dijkstra(&g, src).expect("ok");
        for t in 0..4 {
            assert!((fw[src * 4 + t] - d.dist[t]).abs() < 1e-9);
        }
    }
}

// 6. MST Prim and Kruskal agree on total weight
#[test]
fn e2e_mst_prim_kruskal_agree() {
    let mut g = WeightedGraph::new(5);
    let edges = [
        (0, 1, 1.0),
        (0, 2, 4.0),
        (1, 2, 2.0),
        (1, 3, 5.0),
        (2, 3, 1.0),
        (3, 4, 3.0),
    ];
    for (u, v, w) in edges {
        g.add_undirected_edge(u, v, w).expect("ok");
    }
    let prim = prim_mst(&g, 0).expect("ok");
    let krus = kruskal_mst(&g).expect("ok");
    let p_total: f64 = prim.iter().map(|e| e.2).sum();
    let k_total: f64 = krus.iter().map(|e| e.2).sum();
    assert!((p_total - k_total).abs() < 1e-9);
}

// 7. Edmonds-Karp max-flow equals min-cut capacity
#[test]
fn e2e_max_flow_eq_min_cut() {
    let mut net = FlowNetwork::new(4);
    net.add_edge(0, 1, 3.0).expect("ok");
    net.add_edge(0, 2, 2.0).expect("ok");
    net.add_edge(1, 3, 3.0).expect("ok");
    net.add_edge(2, 3, 2.0).expect("ok");
    let f = edmonds_karp(&net, 0, 3).expect("ok");
    let mc = min_cut_from_max_flow(&net, 0, 3).expect("ok");
    let cap: f64 = mc.cut_edges.iter().map(|e| e.2).sum();
    assert!((f - cap).abs() < 1e-9);
}

// 8. Tarjan SCC on [0->1, 1->2, 2->0, 3->1] finds 2 SCCs: {0,1,2}, {3}
#[test]
fn e2e_tarjan_scc_two() {
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

// 9. PageRank converges and sum of values = 1
#[test]
fn e2e_pagerank_sums_to_one() {
    let mut g = AdjacencyList::new(5);
    g.add_edge(0, 1).expect("ok");
    g.add_edge(1, 2).expect("ok");
    g.add_edge(2, 0).expect("ok");
    g.add_edge(3, 0).expect("ok");
    g.add_edge(4, 1).expect("ok");
    let r = pagerank(&g, 0.85, 500, 1e-10).expect("ok");
    let s: f64 = r.iter().sum();
    assert!((s - 1.0).abs() < 1e-6);
}

// 10. Louvain modularity >= 0 on small community structure
#[test]
fn e2e_louvain_modularity_nonneg() {
    let mut g = WeightedGraph::new(6);
    for (u, v) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
        g.add_undirected_edge(u, v, 1.0).expect("ok");
    }
    let out = louvain_communities(&g).expect("ok");
    assert!(out.modularity >= 0.0);
}

// 11. A* with zero heuristic equals Dijkstra
#[test]
fn e2e_astar_zero_heuristic_eq_dijkstra() {
    let mut g = WeightedGraph::new(4);
    g.add_edge(0, 1, 1.0).expect("ok");
    g.add_edge(0, 2, 4.0).expect("ok");
    g.add_edge(1, 2, 2.0).expect("ok");
    g.add_edge(1, 3, 5.0).expect("ok");
    g.add_edge(2, 3, 1.0).expect("ok");
    let a = a_star(&g, 0, 3, |_| 0.0).expect("ok");
    let d = dijkstra(&g, 0).expect("ok");
    assert!((a.dist - d.dist[3]).abs() < 1e-12);
}

// 12. VF2 isomorphism detects K3 == K3
#[test]
fn e2e_vf2_k3_eq_k3() {
    let mut g1 = AdjacencyList::new(3);
    g1.add_undirected_edge(0, 1).expect("ok");
    g1.add_undirected_edge(1, 2).expect("ok");
    g1.add_undirected_edge(0, 2).expect("ok");
    let g2 = g1.clone();
    let m = vf2_isomorphism(&g1, &g2).expect("ok");
    assert!(m.is_some());
}

// 13. Hopcroft-Karp bipartite matching on 3-vs-3 perfect matching
#[test]
fn e2e_hopcroft_karp_perfect_3x3() {
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

// 14. Hungarian assignment minimum cost matches brute-force on 4x4
#[test]
fn e2e_hungarian_brute_4x4() {
    let c = vec![
        7.0, 2.0, 1.0, 9.0, 5.0, 6.0, 3.0, 4.0, 8.0, 1.0, 5.0, 3.0, 1.0, 9.0, 2.0, 7.0,
    ];
    let (_a, t) = hungarian_assignment(&c, 4).expect("ok");
    let perms = [
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [0, 2, 1, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        [0, 3, 2, 1],
        [1, 0, 2, 3],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 2, 3, 0],
        [1, 3, 0, 2],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 0, 3, 1],
        [2, 1, 0, 3],
        [2, 1, 3, 0],
        [2, 3, 0, 1],
        [2, 3, 1, 0],
        [3, 0, 1, 2],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
        [3, 1, 2, 0],
        [3, 2, 0, 1],
        [3, 2, 1, 0],
    ];
    let mut best = f64::INFINITY;
    for perm in perms {
        let s: f64 = (0..4).map(|i| c[i * 4 + perm[i]]).sum();
        if s < best {
            best = s;
        }
    }
    assert!((t - best).abs() < 1e-9);
}

// 15. Triangle counting on K4 returns 4
#[test]
fn e2e_triangles_k4() {
    let mut g = AdjacencyList::new(4);
    for u in 0..4 {
        for v in u + 1..4 {
            g.add_undirected_edge(u, v).expect("ok");
        }
    }
    let mut tri = 0u64;
    let mut nset: Vec<std::collections::BTreeSet<usize>> =
        vec![std::collections::BTreeSet::new(); 4];
    for u in 0..4 {
        for &v in g.neighbors(u).expect("ok") {
            if v != u {
                nset[u].insert(v);
            }
        }
    }
    for u in 0..4 {
        let neigh: Vec<usize> = nset[u].iter().copied().collect();
        for i in 0..neigh.len() {
            for j in i + 1..neigh.len() {
                if nset[neigh[i]].contains(&neigh[j]) {
                    tri += 1;
                }
            }
        }
    }
    // Each triangle counted 3 times once per vertex.
    assert_eq!(tri / 3, 4);
}

// 16. Topological sort gives valid order on a DAG
#[test]
fn e2e_topo_sort_dag() {
    let mut g = AdjacencyList::new(6);
    let edges = [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)];
    for (u, v) in edges {
        g.add_edge(u, v).expect("ok");
    }
    let o = topo_sort_kahn(&g).expect("ok");
    let pos = |x: usize| o.iter().position(|&y| y == x).expect("ok");
    for (u, v) in edges {
        assert!(pos(u) < pos(v));
    }
}

// 17. Kosaraju matches Tarjan on directed graph SCC count
#[test]
fn e2e_kosaraju_matches_tarjan() {
    let mut g = AdjacencyList::new(6);
    g.add_edge(0, 1).expect("ok");
    g.add_edge(1, 2).expect("ok");
    g.add_edge(2, 0).expect("ok");
    g.add_edge(3, 4).expect("ok");
    g.add_edge(4, 3).expect("ok");
    g.add_edge(5, 2).expect("ok");
    let a = scc_tarjan(&g).expect("ok");
    let b = scc_kosaraju(&g).expect("ok");
    assert_eq!(a.len(), b.len());
}

// 18. Bridge detection on a line
#[test]
fn e2e_bridges_line() {
    let mut g = AdjacencyList::new(4);
    for i in 0..3 {
        g.add_undirected_edge(i, i + 1).expect("ok");
    }
    let b = bridges_tarjan(&g).expect("ok");
    assert_eq!(b.len(), 3);
}

// 19. Articulation points middle of a line
#[test]
fn e2e_articulation_line() {
    let mut g = AdjacencyList::new(5);
    for i in 0..4 {
        g.add_undirected_edge(i, i + 1).expect("ok");
    }
    let ap = articulation_points(&g).expect("ok");
    assert_eq!(ap, vec![1, 2, 3]);
}

// 20. Eigenvector centrality on a star: center largest
#[test]
fn e2e_eigen_center_star() {
    let mut g = AdjacencyList::new(5);
    for v in 1..5 {
        g.add_undirected_edge(0, v).expect("ok");
    }
    let c = eigenvector_centrality(&g, 200, 1e-10).expect("ok");
    for v in 1..5 {
        assert!(c[0] >= c[v]);
    }
}

// 21. Closeness on a path
#[test]
fn e2e_closeness_path() {
    let mut g = AdjacencyList::new(3);
    g.add_undirected_edge(0, 1).expect("ok");
    g.add_undirected_edge(1, 2).expect("ok");
    let c = closeness_centrality(&g).expect("ok");
    assert!((c[1] - 1.0).abs() < 1e-9);
}

// 22. Betweenness on a path: middle highest
#[test]
fn e2e_betweenness_path_middle() {
    let mut g = AdjacencyList::new(5);
    for i in 0..4 {
        g.add_undirected_edge(i, i + 1).expect("ok");
    }
    let b = betweenness_centrality(&g, false).expect("ok");
    assert!(b[2] >= b[1] && b[2] >= b[3]);
}

// 23. Held-Karp returns optimal cost on unit square
#[test]
fn e2e_held_karp_square() {
    let pts: [(f64, f64); 4] = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    let n = 4;
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let dx = pts[i].0 - pts[j].0;
            let dy = pts[i].1 - pts[j].1;
            d[i * n + j] = (dx * dx + dy * dy).sqrt();
        }
    }
    let t = held_karp_tsp(&d, n).expect("ok");
    assert!((t.cost - 4.0).abs() < 1e-9);
}

// 24. Nearest-neighbor on square not worse than 1.5x optimal
#[test]
fn e2e_nn_tour_bound() {
    let pts: [(f64, f64); 4] = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    let n = 4;
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let dx = pts[i].0 - pts[j].0;
            let dy = pts[i].1 - pts[j].1;
            d[i * n + j] = (dx * dx + dy * dy).sqrt();
        }
    }
    let t = nearest_neighbor_tour(&d, n, 0).expect("ok");
    assert!(t.cost <= 4.0 * 1.5);
}

// 25. Eulerian circuit on triangle
#[test]
fn e2e_eulerian_triangle() {
    let mut g = AdjacencyList::new(3);
    g.add_undirected_edge(0, 1).expect("ok");
    g.add_undirected_edge(1, 2).expect("ok");
    g.add_undirected_edge(0, 2).expect("ok");
    let c = eulerian_circuit_hierholzer(&g).expect("ok");
    assert_eq!(c.len(), 4);
}

// 26. Chu-Liu-Edmonds arborescence on simple DAG
#[test]
fn e2e_cle_simple() {
    let mut g = WeightedGraph::new(4);
    g.add_edge(0, 1, 1.0).expect("ok");
    g.add_edge(0, 2, 1.0).expect("ok");
    g.add_edge(1, 3, 1.0).expect("ok");
    let p = chu_liu_edmonds(&g, 0).expect("ok");
    assert_eq!(p[1], 0);
    assert_eq!(p[2], 0);
    assert_eq!(p[3], 1);
}

// 27. Coloring proper on K4 uses 4 colors
#[test]
fn e2e_dsatur_k4() {
    let mut g = AdjacencyList::new(4);
    for u in 0..4 {
        for v in u + 1..4 {
            g.add_undirected_edge(u, v).expect("ok");
        }
    }
    let c = dsatur_coloring(&g).expect("ok");
    let used: std::collections::BTreeSet<usize> = c.iter().copied().collect();
    assert_eq!(used.len(), 4);
}

// 28. Label propagation on two triangles
#[test]
fn e2e_label_propagation_partitions() {
    let mut g = AdjacencyList::new(6);
    for (u, v) in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)] {
        g.add_undirected_edge(u, v).expect("ok");
    }
    let lbl = label_propagation(&g, 100).expect("ok");
    assert_eq!(lbl.len(), 6);
}

// 29. Metrics on a path
#[test]
fn e2e_metrics_path() {
    let mut g = AdjacencyList::new(5);
    for i in 0..4 {
        g.add_undirected_edge(i, i + 1).expect("ok");
    }
    assert_eq!(diameter(&g).expect("ok"), 4);
    assert_eq!(radius(&g).expect("ok"), 2);
    let cc = clustering_coefficient_global(&g).expect("ok");
    assert!(cc < 1e-9);
}

// 30. PTX kernels non-empty across SM versions
#[test]
fn e2e_ptx_all_sm() {
    let kernels: Vec<fn(u32) -> String> = vec![
        bfs_level_ptx,
        dijkstra_relax_ptx,
        pagerank_step_ptx,
        fw_inner_ptx,
        triangle_count_ptx,
        csr_spmv_bool_ptx,
        community_label_ptx,
    ];
    for sm in [75u32, 80, 86, 89, 90, 100] {
        for f in &kernels {
            let s = f(sm);
            assert!(!s.is_empty());
            assert!(s.contains(".visible .entry"));
        }
    }
}
