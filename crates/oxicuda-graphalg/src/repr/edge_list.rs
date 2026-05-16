//! Edge list representation.

use crate::error::{GraphalgError, GraphalgResult};

/// Edge list: list of `(u, v)` pairs.
#[derive(Debug, Clone)]
pub struct EdgeList {
    pub n: usize,
    pub edges: Vec<(usize, usize)>,
}

impl EdgeList {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: Vec::new(),
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize) -> GraphalgResult<()> {
        if u >= self.n || v >= self.n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: u.max(v),
                len: self.n,
            });
        }
        self.edges.push((u, v));
        Ok(())
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_edge_list() {
        let mut e = EdgeList::new(4);
        e.add_edge(0, 1).expect("ok");
        e.add_edge(1, 2).expect("ok");
        e.add_edge(2, 3).expect("ok");
        assert_eq!(e.num_edges(), 3);
    }
}
