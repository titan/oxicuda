//! Generic Push-Relabel max flow with FIFO selection of active vertices.

use std::collections::VecDeque;

use crate::error::{GraphalgError, GraphalgResult};
use crate::max_flow::edmonds_karp::FlowNetwork;

pub fn push_relabel_max_flow(net: &FlowNetwork, source: usize, sink: usize) -> GraphalgResult<f64> {
    if source >= net.n || sink >= net.n {
        return Err(GraphalgError::SourceOutOfRange {
            node: source.max(sink),
            n: net.n,
        });
    }
    let n = net.n;
    let mut residual = net.clone();
    let mut excess = vec![0.0f64; n];
    let mut height = vec![0usize; n];
    height[source] = n;
    let mut in_queue = vec![false; n];
    let mut queue: VecDeque<usize> = VecDeque::new();
    // Initial pre-flow saturate source edges
    for v in 0..n {
        let c = residual.c(source, v);
        if c > 1e-12 {
            excess[v] += c;
            excess[source] -= c;
            *residual.c_mut(source, v) -= c;
            *residual.c_mut(v, source) += c;
            if v != sink && v != source && !in_queue[v] {
                queue.push_back(v);
                in_queue[v] = true;
            }
        }
    }
    while let Some(u) = queue.pop_front() {
        in_queue[u] = false;
        while excess[u] > 1e-12 {
            // Try to push to a neighbor with height[u] = height[v] + 1
            let mut pushed = false;
            for v in 0..n {
                if residual.c(u, v) > 1e-12 && height[u] == height[v] + 1 {
                    let amount = excess[u].min(residual.c(u, v));
                    *residual.c_mut(u, v) -= amount;
                    *residual.c_mut(v, u) += amount;
                    excess[u] -= amount;
                    excess[v] += amount;
                    if v != source && v != sink && !in_queue[v] {
                        queue.push_back(v);
                        in_queue[v] = true;
                    }
                    pushed = true;
                    if excess[u] <= 1e-12 {
                        break;
                    }
                }
            }
            if !pushed {
                // Relabel
                let mut min_height = usize::MAX;
                for v in 0..n {
                    if residual.c(u, v) > 1e-12 && height[v] < min_height {
                        min_height = height[v];
                    }
                }
                if min_height == usize::MAX {
                    break;
                }
                height[u] = min_height + 1;
            }
        }
    }
    Ok(excess[sink])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pr_simple() {
        let mut net = FlowNetwork::new(4);
        net.add_edge(0, 1, 3.0).expect("ok");
        net.add_edge(0, 2, 2.0).expect("ok");
        net.add_edge(1, 3, 3.0).expect("ok");
        net.add_edge(2, 3, 2.0).expect("ok");
        let f = push_relabel_max_flow(&net, 0, 3).expect("ok");
        assert!((f - 5.0).abs() < 1e-6);
    }
}
