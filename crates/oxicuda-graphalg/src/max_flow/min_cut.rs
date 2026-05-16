//! Min cut via residual reachability from max flow.

use std::collections::VecDeque;

use crate::error::{GraphalgError, GraphalgResult};
use crate::max_flow::edmonds_karp::FlowNetwork;

#[derive(Debug, Clone)]
pub struct MinCut {
    pub value: f64,
    pub source_side: Vec<usize>,
    pub cut_edges: Vec<(usize, usize, f64)>,
}

/// Computes the s-t min cut by running Edmonds-Karp and then taking residual reachability from `source`.
pub fn min_cut_from_max_flow(
    net: &FlowNetwork,
    source: usize,
    sink: usize,
) -> GraphalgResult<MinCut> {
    if source >= net.n || sink >= net.n {
        return Err(GraphalgError::SourceOutOfRange {
            node: source.max(sink),
            n: net.n,
        });
    }
    let n = net.n;
    let mut residual = net.clone();
    let mut max_flow = 0.0;
    loop {
        let mut parent = vec![usize::MAX; n];
        parent[source] = source;
        let mut q: VecDeque<usize> = VecDeque::new();
        q.push_back(source);
        let mut found = false;
        while let Some(u) = q.pop_front() {
            if u == sink {
                found = true;
                break;
            }
            for v in 0..n {
                if parent[v] == usize::MAX && residual.c(u, v) > 1e-12 {
                    parent[v] = u;
                    q.push_back(v);
                }
            }
        }
        if !found {
            break;
        }
        let mut bottleneck = f64::INFINITY;
        let mut v = sink;
        while v != source {
            let u = parent[v];
            bottleneck = bottleneck.min(residual.c(u, v));
            v = u;
        }
        let mut v = sink;
        while v != source {
            let u = parent[v];
            *residual.c_mut(u, v) -= bottleneck;
            *residual.c_mut(v, u) += bottleneck;
            v = u;
        }
        max_flow += bottleneck;
    }
    // Find reachable set in residual graph from source
    let mut reach = vec![false; n];
    reach[source] = true;
    let mut q: VecDeque<usize> = VecDeque::new();
    q.push_back(source);
    while let Some(u) = q.pop_front() {
        for v in 0..n {
            if !reach[v] && residual.c(u, v) > 1e-12 {
                reach[v] = true;
                q.push_back(v);
            }
        }
    }
    let mut source_side = Vec::new();
    let mut cut_edges = Vec::new();
    for u in 0..n {
        if reach[u] {
            source_side.push(u);
            for v in 0..n {
                if !reach[v] && net.c(u, v) > 1e-12 {
                    cut_edges.push((u, v, net.c(u, v)));
                }
            }
        }
    }
    Ok(MinCut {
        value: max_flow,
        source_side,
        cut_edges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_cut_eq_max_flow() {
        let mut net = FlowNetwork::new(4);
        net.add_edge(0, 1, 3.0).expect("ok");
        net.add_edge(0, 2, 2.0).expect("ok");
        net.add_edge(1, 3, 3.0).expect("ok");
        net.add_edge(2, 3, 2.0).expect("ok");
        let mc = min_cut_from_max_flow(&net, 0, 3).expect("ok");
        let cut: f64 = mc.cut_edges.iter().map(|e| e.2).sum();
        assert!((mc.value - cut).abs() < 1e-9);
    }
}
