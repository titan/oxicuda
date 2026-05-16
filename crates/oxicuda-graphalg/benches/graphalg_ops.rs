#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::useless_vec)]

use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_graphalg::centrality::pagerank::pagerank;
use oxicuda_graphalg::connectivity::scc_tarjan::scc_tarjan;
use oxicuda_graphalg::handle::LcgRng;
use oxicuda_graphalg::matching::hungarian_munkres::hungarian_assignment;
use oxicuda_graphalg::max_flow::edmonds_karp::{FlowNetwork, edmonds_karp};
use oxicuda_graphalg::mst::kruskal::kruskal_mst;
use oxicuda_graphalg::ptx_kernels::{
    bfs_level_ptx, community_label_ptx, csr_spmv_bool_ptx, dijkstra_relax_ptx, fw_inner_ptx,
    pagerank_step_ptx, triangle_count_ptx,
};
use oxicuda_graphalg::repr::adjacency_list::AdjacencyList;
use oxicuda_graphalg::repr::weighted_graph::WeightedGraph;
use oxicuda_graphalg::shortest_path::dijkstra::dijkstra;
use oxicuda_graphalg::traversal::bfs::bfs_levels;

type KernelEntry = (&'static str, fn(u32) -> String);

fn bench_ptx(c: &mut Criterion) {
    let sm_versions = [75u32, 80, 89, 90];
    let kernels: &[KernelEntry] = &[
        ("bfs_level", bfs_level_ptx),
        ("dijkstra_relax", dijkstra_relax_ptx),
        ("pagerank_step", pagerank_step_ptx),
        ("fw_inner", fw_inner_ptx),
        ("triangle_count", triangle_count_ptx),
        ("csr_spmv_bool", csr_spmv_bool_ptx),
        ("community_label", community_label_ptx),
    ];
    for &sm in &sm_versions {
        for &(name, f) in kernels {
            c.bench_function(&format!("ptx_{name}_sm{sm}"), |b| b.iter(|| f(sm)));
        }
    }
}

fn random_graph(n: usize, density: f64, seed: u64) -> AdjacencyList {
    let mut rng = LcgRng::new(seed);
    let mut g = AdjacencyList::new(n);
    for u in 0..n {
        for v in (u + 1)..n {
            if rng.next_f64() < density {
                g.add_undirected_edge(u, v).expect("ok");
            }
        }
    }
    g
}

fn random_weighted(n: usize, density: f64, seed: u64) -> WeightedGraph {
    let mut rng = LcgRng::new(seed);
    let mut g = WeightedGraph::new(n);
    for u in 0..n {
        for v in (u + 1)..n {
            if rng.next_f64() < density {
                let w = 0.1 + rng.next_f64() * 9.9;
                g.add_undirected_edge(u, v, w).expect("ok");
            }
        }
    }
    g
}

fn bench_bfs(c: &mut Criterion) {
    let g = random_graph(64, 0.1, 11);
    c.bench_function("bfs_n64", |b| b.iter(|| bfs_levels(&g, 0).expect("ok")));
}

fn bench_dijkstra(c: &mut Criterion) {
    let g = random_weighted(64, 0.1, 13);
    c.bench_function("dijkstra_n64", |b| b.iter(|| dijkstra(&g, 0).expect("ok")));
}

fn bench_kruskal(c: &mut Criterion) {
    let g = random_weighted(64, 0.2, 17);
    c.bench_function("kruskal_n64", |b| b.iter(|| kruskal_mst(&g).expect("ok")));
}

fn bench_pagerank(c: &mut Criterion) {
    let g = random_graph(64, 0.1, 19);
    c.bench_function("pagerank_n64", |b| {
        b.iter(|| pagerank(&g, 0.85, 50, 1e-6).expect("ok"))
    });
}

fn bench_scc(c: &mut Criterion) {
    let g = random_graph(64, 0.05, 23);
    c.bench_function("scc_tarjan_n64", |b| b.iter(|| scc_tarjan(&g).expect("ok")));
}

fn bench_max_flow(c: &mut Criterion) {
    let mut net = FlowNetwork::new(16);
    let mut rng = LcgRng::new(29);
    for u in 0..16 {
        for v in 0..16 {
            if u != v && rng.next_f64() < 0.3 {
                net.add_edge(u, v, 1.0 + rng.next_f64() * 9.0).expect("ok");
            }
        }
    }
    c.bench_function("edmonds_karp_n16", |b| {
        b.iter(|| edmonds_karp(&net, 0, 15).expect("ok"))
    });
}

fn bench_hungarian(c: &mut Criterion) {
    let n = 12;
    let mut rng = LcgRng::new(31);
    let cost: Vec<f64> = (0..n * n).map(|_| rng.next_f64() * 10.0).collect();
    c.bench_function("hungarian_n12", |b| {
        b.iter(|| hungarian_assignment(&cost, n).expect("ok"))
    });
}

criterion_group!(
    benches,
    bench_ptx,
    bench_bfs,
    bench_dijkstra,
    bench_kruskal,
    bench_pagerank,
    bench_scc,
    bench_max_flow,
    bench_hungarian,
);
criterion_main!(benches);
