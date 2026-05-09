use crate::error::{CausalError, CausalResult};
use std::collections::VecDeque;

pub struct Dag {
    pub n: usize,
    adj: Vec<Vec<usize>>,
    radj: Vec<Vec<usize>>,
}

impl Dag {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            adj: vec![vec![]; n],
            radj: vec![vec![]; n],
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize) -> CausalResult<()> {
        if from >= self.n || to >= self.n {
            return Err(CausalError::InvalidGraphSize { n: self.n });
        }
        if from == to {
            return Err(CausalError::CyclicGraph);
        }
        // Check if `to` is already an ancestor of `from` (would create cycle)
        if self.ancestors(from).contains(&to) {
            return Err(CausalError::CyclicGraph);
        }
        if !self.adj[from].contains(&to) {
            self.adj[from].push(to);
            self.radj[to].push(from);
        }
        Ok(())
    }

    pub fn remove_edge(&mut self, from: usize, to: usize) {
        if from < self.n && to < self.n {
            self.adj[from].retain(|&c| c != to);
            self.radj[to].retain(|&p| p != from);
        }
    }

    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        if from >= self.n || to >= self.n {
            return false;
        }
        self.adj[from].contains(&to)
    }

    pub fn parents(&self, node: usize) -> &[usize] {
        if node >= self.n {
            return &[];
        }
        &self.radj[node]
    }

    pub fn children(&self, node: usize) -> &[usize] {
        if node >= self.n {
            return &[];
        }
        &self.adj[node]
    }

    pub fn ancestors(&self, node: usize) -> Vec<usize> {
        if node >= self.n {
            return vec![];
        }
        let mut visited = vec![false; self.n];
        let mut queue = VecDeque::new();
        queue.push_back(node);
        visited[node] = true;
        let mut result = Vec::new();
        while let Some(cur) = queue.pop_front() {
            for &parent in &self.radj[cur] {
                if !visited[parent] {
                    visited[parent] = true;
                    result.push(parent);
                    queue.push_back(parent);
                }
            }
        }
        result
    }

    pub fn descendants(&self, node: usize) -> Vec<usize> {
        if node >= self.n {
            return vec![];
        }
        let mut visited = vec![false; self.n];
        let mut queue = VecDeque::new();
        queue.push_back(node);
        visited[node] = true;
        let mut result = Vec::new();
        while let Some(cur) = queue.pop_front() {
            for &child in &self.adj[cur] {
                if !visited[child] {
                    visited[child] = true;
                    result.push(child);
                    queue.push_back(child);
                }
            }
        }
        result
    }

    pub fn topo_sort(&self) -> CausalResult<Vec<usize>> {
        let mut in_degree = vec![0usize; self.n];
        for node in 0..self.n {
            for &child in &self.adj[node] {
                in_degree[child] += 1;
            }
        }
        let mut queue: VecDeque<usize> = (0..self.n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(self.n);
        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &child in &self.adj[node] {
                in_degree[child] -= 1;
                if in_degree[child] == 0 {
                    queue.push_back(child);
                }
            }
        }
        if order.len() == self.n {
            Ok(order)
        } else {
            Err(CausalError::CyclicGraph)
        }
    }

    pub fn is_dag(&self) -> bool {
        self.topo_sort().is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_dag_ops() {
        let mut dag = Dag::new(4);
        dag.add_edge(0, 1).unwrap();
        dag.add_edge(1, 2).unwrap();
        dag.add_edge(0, 3).unwrap();
        assert!(dag.has_edge(0, 1));
        assert!(!dag.has_edge(1, 0));
        let order = dag.topo_sort().unwrap();
        assert_eq!(order[0], 0);
    }

    #[test]
    fn cycle_detection() {
        let mut dag = Dag::new(3);
        dag.add_edge(0, 1).unwrap();
        dag.add_edge(1, 2).unwrap();
        let result = dag.add_edge(2, 0);
        assert!(result.is_err());
    }

    #[test]
    fn ancestors_descendants() {
        let mut dag = Dag::new(4);
        dag.add_edge(0, 1).unwrap();
        dag.add_edge(1, 2).unwrap();
        dag.add_edge(2, 3).unwrap();
        let anc = dag.ancestors(3);
        assert!(anc.contains(&0));
        assert!(anc.contains(&1));
        assert!(anc.contains(&2));
        let desc = dag.descendants(0);
        assert!(desc.contains(&1));
        assert!(desc.contains(&2));
        assert!(desc.contains(&3));
    }
}
