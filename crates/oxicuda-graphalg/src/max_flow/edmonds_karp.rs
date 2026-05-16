//! Edmonds-Karp max flow: BFS augmenting paths. O(VE^2).

use std::collections::VecDeque;

use crate::error::{GraphalgError, GraphalgResult};

/// Residual graph as an n x n capacity matrix (row-major).
#[derive(Debug, Clone)]
pub struct FlowNetwork {
    pub n: usize,
    pub cap: Vec<f64>,
}

impl FlowNetwork {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            cap: vec![0.0; n * n],
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize, c: f64) -> GraphalgResult<()> {
        if u >= self.n || v >= self.n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: u.max(v),
                len: self.n,
            });
        }
        if c < 0.0 {
            return Err(GraphalgError::InvalidEdgeWeight(format!(
                "negative capacity {c} on ({u},{v})"
            )));
        }
        self.cap[u * self.n + v] += c;
        Ok(())
    }

    pub fn c(&self, u: usize, v: usize) -> f64 {
        self.cap[u * self.n + v]
    }

    pub fn c_mut(&mut self, u: usize, v: usize) -> &mut f64 {
        &mut self.cap[u * self.n + v]
    }
}

pub fn edmonds_karp(net: &FlowNetwork, source: usize, sink: usize) -> GraphalgResult<f64> {
    if source >= net.n || sink >= net.n {
        return Err(GraphalgError::SourceOutOfRange {
            node: source.max(sink),
            n: net.n,
        });
    }
    let mut residual = net.clone();
    let mut max_flow = 0.0;
    loop {
        // BFS to find augmenting path
        let mut parent = vec![usize::MAX; net.n];
        parent[source] = source;
        let mut q: VecDeque<usize> = VecDeque::new();
        q.push_back(source);
        while let Some(u) = q.pop_front() {
            if u == sink {
                break;
            }
            for v in 0..net.n {
                if parent[v] == usize::MAX && residual.c(u, v) > 1e-12 {
                    parent[v] = u;
                    q.push_back(v);
                }
            }
        }
        if parent[sink] == usize::MAX {
            break;
        }
        // Find bottleneck
        let mut bottleneck = f64::INFINITY;
        let mut v = sink;
        while v != source {
            let u = parent[v];
            bottleneck = bottleneck.min(residual.c(u, v));
            v = u;
        }
        // Augment
        let mut v = sink;
        while v != source {
            let u = parent[v];
            *residual.c_mut(u, v) -= bottleneck;
            *residual.c_mut(v, u) += bottleneck;
            v = u;
        }
        max_flow += bottleneck;
    }
    Ok(max_flow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ek_simple() {
        // s -> a (3), s -> b (2), a -> t (3), b -> t (2)
        let mut net = FlowNetwork::new(4);
        net.add_edge(0, 1, 3.0).expect("ok");
        net.add_edge(0, 2, 2.0).expect("ok");
        net.add_edge(1, 3, 3.0).expect("ok");
        net.add_edge(2, 3, 2.0).expect("ok");
        let f = edmonds_karp(&net, 0, 3).expect("ok");
        assert!((f - 5.0).abs() < 1e-9);
    }

    #[test]
    fn ek_bottleneck() {
        let mut net = FlowNetwork::new(3);
        net.add_edge(0, 1, 10.0).expect("ok");
        net.add_edge(1, 2, 1.0).expect("ok");
        let f = edmonds_karp(&net, 0, 2).expect("ok");
        assert!((f - 1.0).abs() < 1e-9);
    }
}
