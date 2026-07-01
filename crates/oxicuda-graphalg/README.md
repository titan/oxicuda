# oxicuda-graphalg

Classical graph algorithms in pure Rust, paired with PTX kernels emitted at runtime for OxiCUDA.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) project.

## Overview

`oxicuda-graphalg` covers the canonical algorithmic graph-theory spectrum:
graph representations (adjacency list / matrix / edge list / CSR / weighted),
traversal, shortest paths, spanning trees, max-flow / min-cut, bipartite and
weighted matching, connectivity decompositions, centrality, community
detection, coloring, TSP heuristics, Eulerian and Hamiltonian circuits, and
basic descriptive metrics.

Every algorithm is implemented from scratch in safe Rust, with no external
graph or numerics libraries. Random sampling uses the workspace `LcgRng`
(MMIX-style LCG), and the crate enables `#![forbid(unsafe_code)]`. GPU
versions of the hot inner loops (BFS frontier expansion, Dijkstra relaxation,
SSSP, etc.) are emitted as PTX strings parametric in the device SM version.

The CPU implementations are the source of truth and double as a reference
oracle for the PTX kernels.

## Modules

| Module | Description |
|--------|-------------|
| `repr` | Graph representations: adjacency list, adjacency matrix, edge list, CSR, weighted graph |
| `traversal` | BFS, DFS (recursive + iterative), IDDFS, bidirectional BFS |
| `topological` | Topological sort (Kahn, DFS-based) |
| `shortest_path` | Dijkstra, Bellman-Ford, SPFA, Floyd-Warshall, Johnson, A\*, Yen K-shortest, bidirectional Dijkstra |
| `mst` | Minimum spanning tree: Prim, Kruskal, Borůvka, Union-Find |
| `max_flow` | Edmonds-Karp, Dinic, push-relabel, min-cut |
| `matching` | Hopcroft-Karp bipartite, Hungarian (Munkres), simplified blossom |
| `connectivity` | Tarjan / Kosaraju / Gabow SCC, bridges, articulation points, biconnected components |
| `centrality` | Degree, betweenness (Brandes), closeness, eigenvector, PageRank, Katz |
| `community` | Louvain, label propagation, Girvan-Newman |
| `arborescence` | Chu-Liu-Edmonds minimum spanning arborescence |
| `isomorphism` | VF2 subgraph isomorphism |
| `coloring` | Greedy, DSATUR, Welsh-Powell |
| `tsp` | Christofides approximation, nearest-neighbor, 2-opt |
| `eulerian` | Hierholzer's Eulerian circuit |
| `hamiltonian` | Held-Karp dynamic programming exact TSP |
| `metrics` | Diameter, radius, density, clustering coefficient, transitivity |
| `handle` | `GraphalgHandle`, `SmVersion`, `LcgRng` |
| `error` | `GraphalgError` / `GraphalgResult` |
| `ptx_kernels` | Runtime PTX strings for BFS / Dijkstra / SSSP per SM version |

## Quick Start

```rust,no_run
use oxicuda_graphalg::repr::weighted_graph::WeightedGraph;
use oxicuda_graphalg::shortest_path::dijkstra::dijkstra;
use oxicuda_graphalg::GraphalgResult;

fn main() -> GraphalgResult<()> {
    // Build a small weighted directed graph.
    let mut g = WeightedGraph::new(4);
    g.add_edge(0, 1, 1.0)?;
    g.add_edge(0, 2, 4.0)?;
    g.add_edge(1, 2, 2.0)?;
    g.add_edge(2, 3, 1.0)?;

    // Single-source shortest paths from node 0.
    let out = dijkstra(&g, 0)?;
    println!("dist = {:?}", out.dist);
    println!("parent = {:?}", out.parent);
    Ok(())
}
```

## Design Notes

- `#![forbid(unsafe_code)]` — every algorithm is implemented in safe Rust.
- No external graph or numerics dependencies — the `repr`, `mst::union_find`,
  binary-heap, and linear-algebra helpers used internally are all part of
  this crate or `std`.
- The CPU implementations are the reference oracle for the matching PTX
  kernels emitted by `ptx_kernels::*` — the GPU path is verified bit-wise
  against the CPU path in the end-to-end test suite.

## Status

**Alpha** — 11,913 SLoC, 358 passing tests. API may evolve before v1.0.

## License

Apache-2.0 — (C) 2026 COOLJAPAN OU (Team KitaSan)
