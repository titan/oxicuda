//! Disjoint Set Union (Union-Find) with path compression and union by rank.

#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u32>,
    size: Vec<usize>,
    num_sets: usize,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
            num_sets: n,
        }
    }

    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    pub fn find(&mut self, x: usize) -> usize {
        let mut r = x;
        while self.parent[r] != r {
            r = self.parent[r];
        }
        // Path compression
        let mut cur = x;
        while self.parent[cur] != r {
            let next = self.parent[cur];
            self.parent[cur] = r;
            cur = next;
        }
        r
    }

    /// Union two sets. Returns true if they were merged.
    pub fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => {
                self.parent[ra] = rb;
                self.size[rb] += self.size[ra];
            }
            std::cmp::Ordering::Greater => {
                self.parent[rb] = ra;
                self.size[ra] += self.size[rb];
            }
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
                self.size[ra] += self.size[rb];
            }
        }
        self.num_sets -= 1;
        true
    }

    pub fn set_size(&mut self, x: usize) -> usize {
        let r = self.find(x);
        self.size[r]
    }

    pub fn num_sets(&self) -> usize {
        self.num_sets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uf_basic() {
        let mut uf = UnionFind::new(5);
        assert_eq!(uf.num_sets(), 5);
        assert!(uf.union(0, 1));
        assert_eq!(uf.num_sets(), 4);
        assert!(!uf.union(0, 1));
        assert_eq!(uf.find(0), uf.find(1));
    }

    #[test]
    fn uf_set_size() {
        let mut uf = UnionFind::new(4);
        uf.union(0, 1);
        uf.union(2, 3);
        uf.union(1, 2);
        assert_eq!(uf.set_size(0), 4);
    }
}
