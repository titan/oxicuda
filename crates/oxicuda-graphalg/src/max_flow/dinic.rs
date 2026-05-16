//! Dinic's max flow: level-graph + blocking-flow DFS. O(V^2 E).

use std::collections::VecDeque;

use crate::error::{GraphalgError, GraphalgResult};
use crate::max_flow::edmonds_karp::FlowNetwork;

/// Dinic's algorithm using a dense residual matrix.
pub fn dinic_max_flow(net: &FlowNetwork, source: usize, sink: usize) -> GraphalgResult<f64> {
    if source >= net.n || sink >= net.n {
        return Err(GraphalgError::SourceOutOfRange {
            node: source.max(sink),
            n: net.n,
        });
    }
    let mut residual = net.clone();
    let mut max_flow = 0.0;
    let n = net.n;
    loop {
        // BFS to build level graph
        let mut level = vec![i64::MAX; n];
        level[source] = 0;
        let mut q: VecDeque<usize> = VecDeque::new();
        q.push_back(source);
        while let Some(u) = q.pop_front() {
            for v in 0..n {
                if level[v] == i64::MAX && residual.c(u, v) > 1e-12 {
                    level[v] = level[u] + 1;
                    q.push_back(v);
                }
            }
        }
        if level[sink] == i64::MAX {
            break;
        }
        // DFS to push blocking flow
        let mut next_idx = vec![0usize; n];
        loop {
            let pushed = dfs_push(
                &mut residual,
                source,
                sink,
                f64::INFINITY,
                &level,
                &mut next_idx,
            );
            if pushed <= 1e-15 {
                break;
            }
            max_flow += pushed;
        }
    }
    Ok(max_flow)
}

fn dfs_push(
    residual: &mut FlowNetwork,
    u: usize,
    sink: usize,
    pushed: f64,
    level: &[i64],
    next_idx: &mut [usize],
) -> f64 {
    if u == sink {
        return pushed;
    }
    let n = residual.n;
    while next_idx[u] < n {
        let v = next_idx[u];
        if residual.c(u, v) > 1e-12 && level[v] == level[u] + 1 {
            let bottleneck = pushed.min(residual.c(u, v));
            let res = dfs_push(residual, v, sink, bottleneck, level, next_idx);
            if res > 1e-15 {
                *residual.c_mut(u, v) -= res;
                *residual.c_mut(v, u) += res;
                return res;
            }
        }
        next_idx[u] += 1;
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dinic_simple() {
        let mut net = FlowNetwork::new(4);
        net.add_edge(0, 1, 3.0).expect("ok");
        net.add_edge(0, 2, 2.0).expect("ok");
        net.add_edge(1, 3, 3.0).expect("ok");
        net.add_edge(2, 3, 2.0).expect("ok");
        let f = dinic_max_flow(&net, 0, 3).expect("ok");
        assert!((f - 5.0).abs() < 1e-9);
    }
}
