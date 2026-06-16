//! Block-sparse tensor operations.
//!
//! Many tensor networks carry a symmetry (U(1) charge, Z₂ parity, …) that forces
//! a *block structure*: the tensor is non-zero only on combinations of sector
//! labels whose charges are conserved. A **block-sparse tensor** stores only
//! those non-zero dense blocks, keyed by the sector tuple of each leg, which
//! saves both memory and arithmetic relative to a fully dense representation.
//!
//! This module provides a general, charge-agnostic block-sparse container:
//!
//! * [`BlockSparseTensor`] — a map from per-leg sector keys to dense blocks,
//!   with consistent per-leg sector→dimension tables.
//! * Element-wise [`BlockSparseTensor::add`] (sparse union of blocks).
//! * Matrix-style [`BlockSparseTensor::contract_matmul`] over a shared middle
//!   sector index (the workhorse for block-diagonal SVD / gauge steps).
//! * Dense round-trip via [`BlockSparseTensor::to_dense`] /
//!   [`BlockSparseTensor::from_dense_2d`].
//!
//! The container is deliberately independent of [`crate::mps::symmetric`] (which
//! hard-codes U(1) `Qn` quantum numbers): here a "sector" is just an opaque
//! `i64` label per leg, so the same machinery serves arbitrary abelian
//! symmetries.

use std::collections::BTreeMap;

use crate::{TnError, TnResult};

/// A block key: the sector label on each leg of the tensor.
pub type BlockKey = Vec<i64>;

/// A block-sparse tensor over `rank` legs.
///
/// Each leg has a *sector table* mapping a sector label to that sector's size.
/// A block is a dense buffer whose shape is the product of its legs' sector
/// sizes (row-major in leg order). Only explicitly inserted blocks are stored;
/// everything else is implicitly zero.
#[derive(Debug, Clone)]
pub struct BlockSparseTensor {
    /// Number of legs (tensor rank).
    pub rank: usize,
    /// `sectors[leg]` maps a sector label to its dimension on that leg.
    pub sectors: Vec<BTreeMap<i64, usize>>,
    /// Stored blocks, keyed by the per-leg sector tuple.
    pub blocks: BTreeMap<BlockKey, Vec<f64>>,
}

impl BlockSparseTensor {
    /// Create an empty block-sparse tensor of the given `rank` with the supplied
    /// per-leg sector tables.
    ///
    /// # Errors
    /// * [`TnError::InvalidConfiguration`] if `rank == 0` or `sectors.len() != rank`.
    pub fn new(rank: usize, sectors: Vec<BTreeMap<i64, usize>>) -> TnResult<Self> {
        if rank == 0 {
            return Err(TnError::InvalidConfiguration("rank must be ≥ 1".into()));
        }
        if sectors.len() != rank {
            return Err(TnError::InvalidConfiguration(format!(
                "expected {rank} sector tables, got {}",
                sectors.len()
            )));
        }
        Ok(Self {
            rank,
            sectors,
            blocks: BTreeMap::new(),
        })
    }

    /// Expected flat length of the block keyed by `key` (product of leg sizes).
    fn block_len(&self, key: &BlockKey) -> TnResult<usize> {
        if key.len() != self.rank {
            return Err(TnError::ShapeMismatch {
                expected: vec![self.rank],
                got: vec![key.len()],
            });
        }
        let mut len = 1usize;
        for (leg, &label) in key.iter().enumerate() {
            let dim = *self.sectors[leg].get(&label).ok_or_else(|| {
                TnError::InvalidConfiguration(format!("leg {leg} has no sector {label}"))
            })?;
            len *= dim;
        }
        Ok(len)
    }

    /// Insert (or overwrite) the dense block keyed by `key`.
    ///
    /// # Errors
    /// * [`TnError::ShapeMismatch`] if `key.len() != rank` or `data.len()` does
    ///   not match the product of the key's sector sizes.
    /// * [`TnError::InvalidConfiguration`] if a sector label is unknown.
    pub fn insert_block(&mut self, key: BlockKey, data: Vec<f64>) -> TnResult<()> {
        let expected = self.block_len(&key)?;
        if data.len() != expected {
            return Err(TnError::ShapeMismatch {
                expected: vec![expected],
                got: vec![data.len()],
            });
        }
        self.blocks.insert(key, data);
        Ok(())
    }

    /// Number of stored (non-zero) blocks.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Total number of stored scalar entries across all blocks.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.blocks.values().map(|b| b.len()).sum()
    }

    /// Frobenius norm `√(Σ x²)` over all stored entries.
    #[must_use]
    pub fn frobenius_norm(&self) -> f64 {
        self.blocks
            .values()
            .flat_map(|b| b.iter())
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt()
    }

    /// Element-wise sum `self + other` (sparse union of blocks).
    ///
    /// Blocks present in only one operand are copied; shared blocks are added.
    ///
    /// # Errors
    /// * [`TnError::DimensionMismatch`] if the ranks differ.
    /// * [`TnError::ShapeMismatch`] if a shared block's lengths disagree.
    pub fn add(&self, other: &BlockSparseTensor) -> TnResult<BlockSparseTensor> {
        if self.rank != other.rank {
            return Err(TnError::DimensionMismatch {
                a: self.rank,
                b: other.rank,
            });
        }
        let mut out = self.clone();
        for (key, blk) in &other.blocks {
            match out.blocks.get_mut(key) {
                Some(existing) => {
                    if existing.len() != blk.len() {
                        return Err(TnError::ShapeMismatch {
                            expected: vec![existing.len()],
                            got: vec![blk.len()],
                        });
                    }
                    for (e, v) in existing.iter_mut().zip(blk.iter()) {
                        *e += v;
                    }
                }
                None => {
                    // Merge sector tables for legs that only `other` knows about.
                    for (leg, table) in other.sectors.iter().enumerate() {
                        for (&label, &dim) in table {
                            out.sectors[leg].entry(label).or_insert(dim);
                        }
                    }
                    out.blocks.insert(key.clone(), blk.clone());
                }
            }
        }
        Ok(out)
    }

    /// Scale every entry in place by `alpha`.
    pub fn scale(&mut self, alpha: f64) {
        for blk in self.blocks.values_mut() {
            for x in blk.iter_mut() {
                *x *= alpha;
            }
        }
    }

    /// Contract two rank-2 block-sparse matrices over their shared middle index:
    /// `C[i, k] = Σ_j A[i, j] · B[j, k]`, block-diagonally over the sector `j`.
    ///
    /// `self` (leg order `[i, j]`) and `other` (leg order `[j, k]`) must share the
    /// `j` sector table (`self.sectors[1] == other.sectors[0]`). Only matching
    /// `j` sectors contribute, so the result is itself block-sparse with legs
    /// `[i, k]`.
    ///
    /// # Errors
    /// * [`TnError::InvalidConfiguration`] if either operand is not rank-2.
    /// * [`TnError::DimensionMismatch`] if the shared sector dimensions disagree.
    pub fn contract_matmul(&self, other: &BlockSparseTensor) -> TnResult<BlockSparseTensor> {
        if self.rank != 2 || other.rank != 2 {
            return Err(TnError::InvalidConfiguration(
                "contract_matmul requires two rank-2 tensors".into(),
            ));
        }
        let i_table = self.sectors[0].clone();
        let k_table = other.sectors[1].clone();
        let mut out = BlockSparseTensor::new(2, vec![i_table, k_table])?;

        // Accumulate C[i, k] += A[i, j] · B[j, k] over shared j sectors.
        for (a_key, a_blk) in &self.blocks {
            let i_label = a_key[0];
            let j_label = a_key[1];
            let i_dim = self.sectors[0][&i_label];
            let j_dim = self.sectors[1][&j_label];
            // Find the matching B block (j_label, k_label) for every k.
            for (b_key, b_blk) in &other.blocks {
                if b_key[0] != j_label {
                    continue;
                }
                let k_label = b_key[1];
                let j_dim_b = other.sectors[0][&j_label];
                if j_dim != j_dim_b {
                    return Err(TnError::DimensionMismatch {
                        a: j_dim,
                        b: j_dim_b,
                    });
                }
                let k_dim = other.sectors[1][&k_label];
                let prod = matmul(a_blk, b_blk, i_dim, j_dim, k_dim);
                let out_key = vec![i_label, k_label];
                match out.blocks.get_mut(&out_key) {
                    Some(existing) => {
                        for (e, v) in existing.iter_mut().zip(prod.iter()) {
                            *e += v;
                        }
                    }
                    None => {
                        out.blocks.insert(out_key, prod);
                    }
                }
            }
        }
        Ok(out)
    }

    /// Materialise a rank-2 block-sparse tensor into a dense row-major matrix.
    ///
    /// Sectors are laid out in ascending label order on each leg; the returned
    /// `(rows, cols, data)` gives the offsets implicitly (block of leg-0 sector
    /// `s` starts at the cumulative size of all smaller sectors).
    ///
    /// # Errors
    /// * [`TnError::InvalidConfiguration`] if the tensor is not rank-2.
    pub fn to_dense(&self) -> TnResult<(usize, usize, Vec<f64>)> {
        if self.rank != 2 {
            return Err(TnError::InvalidConfiguration(
                "to_dense supports rank-2 tensors".into(),
            ));
        }
        let (row_off, rows) = cumulative_offsets(&self.sectors[0]);
        let (col_off, cols) = cumulative_offsets(&self.sectors[1]);
        let mut dense = vec![0.0; rows * cols];
        for (key, blk) in &self.blocks {
            let r0 = row_off[&key[0]];
            let c0 = col_off[&key[1]];
            let r_dim = self.sectors[0][&key[0]];
            let c_dim = self.sectors[1][&key[1]];
            for a in 0..r_dim {
                for b in 0..c_dim {
                    dense[(r0 + a) * cols + (c0 + b)] = blk[a * c_dim + b];
                }
            }
        }
        Ok((rows, cols, dense))
    }

    /// Build a rank-2 block-sparse tensor from a single dense block in one
    /// `(row_sector, col_sector)` pair.
    ///
    /// This is the trivial single-sector embedding, handy for tests and for
    /// promoting a dense matrix into the block-sparse algebra.
    ///
    /// # Errors
    /// * [`TnError::ShapeMismatch`] if `data.len() != rows·cols`.
    pub fn from_dense_2d(
        data: &[f64],
        rows: usize,
        cols: usize,
        row_sector: i64,
        col_sector: i64,
    ) -> TnResult<BlockSparseTensor> {
        if data.len() != rows * cols {
            return Err(TnError::ShapeMismatch {
                expected: vec![rows * cols],
                got: vec![data.len()],
            });
        }
        let mut row_table = BTreeMap::new();
        row_table.insert(row_sector, rows);
        let mut col_table = BTreeMap::new();
        col_table.insert(col_sector, cols);
        let mut t = BlockSparseTensor::new(2, vec![row_table, col_table])?;
        t.insert_block(vec![row_sector, col_sector], data.to_vec())?;
        Ok(t)
    }
}

/// Ascending-label cumulative offsets + total size for one leg's sector table.
fn cumulative_offsets(table: &BTreeMap<i64, usize>) -> (BTreeMap<i64, usize>, usize) {
    let mut offsets = BTreeMap::new();
    let mut acc = 0usize;
    for (&label, &dim) in table {
        offsets.insert(label, acc);
        acc += dim;
    }
    (offsets, acc)
}

/// Multiply `a` (`m×k`) by `b` (`k×n`), both row-major, into `m×n`.
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            if aip == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += aip * b[p * n + j];
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(pairs: &[(i64, usize)]) -> BTreeMap<i64, usize> {
        pairs.iter().copied().collect()
    }

    #[test]
    fn new_rejects_bad_rank() {
        assert!(BlockSparseTensor::new(0, vec![]).is_err());
        assert!(BlockSparseTensor::new(2, vec![table(&[(0, 1)])]).is_err());
    }

    #[test]
    fn insert_block_validates_length() {
        let mut t = BlockSparseTensor::new(2, vec![table(&[(0, 2)]), table(&[(0, 3)])])
            .expect("value should be present");
        assert!(t.insert_block(vec![0, 0], vec![0.0; 6]).is_ok());
        assert!(t.insert_block(vec![0, 0], vec![0.0; 5]).is_err());
        assert!(t.insert_block(vec![0, 9], vec![0.0; 6]).is_err()); // unknown sector
    }

    #[test]
    fn num_blocks_and_nnz() {
        let mut t = BlockSparseTensor::new(2, vec![table(&[(0, 2), (1, 1)]), table(&[(0, 2)])])
            .expect("value should be present");
        t.insert_block(vec![0, 0], vec![1.0; 4])
            .expect("insert_block should succeed");
        t.insert_block(vec![1, 0], vec![2.0; 2])
            .expect("insert_block should succeed");
        assert_eq!(t.num_blocks(), 2);
        assert_eq!(t.nnz(), 6);
    }

    #[test]
    fn frobenius_norm_basic() {
        let mut t = BlockSparseTensor::new(2, vec![table(&[(0, 1)]), table(&[(0, 2)])])
            .expect("value should be present");
        t.insert_block(vec![0, 0], vec![3.0, 4.0])
            .expect("insert_block should succeed");
        assert!((t.frobenius_norm() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn add_disjoint_blocks() {
        let mut a = BlockSparseTensor::new(2, vec![table(&[(0, 1)]), table(&[(0, 1)])])
            .expect("value should be present");
        a.insert_block(vec![0, 0], vec![2.0])
            .expect("insert_block should succeed");
        let mut b = BlockSparseTensor::new(2, vec![table(&[(1, 1)]), table(&[(1, 1)])])
            .expect("value should be present");
        b.insert_block(vec![1, 1], vec![5.0])
            .expect("insert_block should succeed");
        let c = a.add(&b).expect("add should succeed");
        assert_eq!(c.num_blocks(), 2);
        assert_eq!(c.blocks[&vec![0, 0]], vec![2.0]);
        assert_eq!(c.blocks[&vec![1, 1]], vec![5.0]);
    }

    #[test]
    fn add_overlapping_blocks() {
        let mut a = BlockSparseTensor::new(2, vec![table(&[(0, 2)]), table(&[(0, 1)])])
            .expect("value should be present");
        a.insert_block(vec![0, 0], vec![1.0, 2.0])
            .expect("insert_block should succeed");
        let mut b = BlockSparseTensor::new(2, vec![table(&[(0, 2)]), table(&[(0, 1)])])
            .expect("value should be present");
        b.insert_block(vec![0, 0], vec![10.0, 20.0])
            .expect("insert_block should succeed");
        let c = a.add(&b).expect("add should succeed");
        assert_eq!(c.num_blocks(), 1);
        assert_eq!(c.blocks[&vec![0, 0]], vec![11.0, 22.0]);
    }

    #[test]
    fn add_rejects_rank_mismatch() {
        let a = BlockSparseTensor::new(2, vec![table(&[(0, 1)]), table(&[(0, 1)])])
            .expect("value should be present");
        let b = BlockSparseTensor::new(
            3,
            vec![table(&[(0, 1)]), table(&[(0, 1)]), table(&[(0, 1)])],
        )
        .expect("value should be present");
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn scale_multiplies_all_entries() {
        let mut t = BlockSparseTensor::new(2, vec![table(&[(0, 1)]), table(&[(0, 2)])])
            .expect("value should be present");
        t.insert_block(vec![0, 0], vec![1.0, -2.0])
            .expect("insert_block should succeed");
        t.scale(3.0);
        assert_eq!(t.blocks[&vec![0, 0]], vec![3.0, -6.0]);
    }

    #[test]
    fn contract_matmul_block_diagonal() {
        // A: legs [i, j], one block (i=0,j=0) 2×2. B: legs [j, k], block (0,0) 2×2.
        let mut a = BlockSparseTensor::new(2, vec![table(&[(0, 2)]), table(&[(0, 2)])])
            .expect("value should be present");
        a.insert_block(vec![0, 0], vec![1.0, 2.0, 3.0, 4.0])
            .expect("value should be present");
        let mut b = BlockSparseTensor::new(2, vec![table(&[(0, 2)]), table(&[(0, 2)])])
            .expect("value should be present");
        b.insert_block(vec![0, 0], vec![5.0, 6.0, 7.0, 8.0])
            .expect("value should be present");
        let c = a
            .contract_matmul(&b)
            .expect("contract_matmul should succeed");
        // [[1,2],[3,4]]·[[5,6],[7,8]] = [[19,22],[43,50]].
        assert_eq!(c.blocks[&vec![0, 0]], vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn contract_matmul_skips_nonmatching_sectors() {
        // A has j-sector 0; B only has j-sector 1 ⇒ no contraction ⇒ empty result.
        let mut a = BlockSparseTensor::new(2, vec![table(&[(0, 1)]), table(&[(0, 1)])])
            .expect("value should be present");
        a.insert_block(vec![0, 0], vec![2.0])
            .expect("insert_block should succeed");
        let mut b = BlockSparseTensor::new(2, vec![table(&[(1, 1)]), table(&[(0, 1)])])
            .expect("value should be present");
        b.insert_block(vec![1, 0], vec![3.0])
            .expect("insert_block should succeed");
        let c = a
            .contract_matmul(&b)
            .expect("contract_matmul should succeed");
        assert_eq!(c.num_blocks(), 0);
    }

    #[test]
    fn contract_matmul_two_sectors_stay_block_diagonal() {
        // Two independent j-sectors ⇒ two independent output blocks.
        let mut a =
            BlockSparseTensor::new(2, vec![table(&[(0, 1), (1, 1)]), table(&[(0, 1), (1, 1)])])
                .expect("value should be present");
        a.insert_block(vec![0, 0], vec![2.0])
            .expect("insert_block should succeed");
        a.insert_block(vec![1, 1], vec![3.0])
            .expect("insert_block should succeed");
        let mut b =
            BlockSparseTensor::new(2, vec![table(&[(0, 1), (1, 1)]), table(&[(0, 1), (1, 1)])])
                .expect("value should be present");
        b.insert_block(vec![0, 0], vec![5.0])
            .expect("insert_block should succeed");
        b.insert_block(vec![1, 1], vec![7.0])
            .expect("insert_block should succeed");
        let c = a
            .contract_matmul(&b)
            .expect("contract_matmul should succeed");
        assert_eq!(c.num_blocks(), 2);
        assert_eq!(c.blocks[&vec![0, 0]], vec![10.0]);
        assert_eq!(c.blocks[&vec![1, 1]], vec![21.0]);
    }

    #[test]
    fn contract_matmul_rejects_non_rank2() {
        let a = BlockSparseTensor::new(
            3,
            vec![table(&[(0, 1)]), table(&[(0, 1)]), table(&[(0, 1)])],
        )
        .expect("value should be present");
        let b = BlockSparseTensor::new(2, vec![table(&[(0, 1)]), table(&[(0, 1)])])
            .expect("value should be present");
        assert!(a.contract_matmul(&b).is_err());
    }

    #[test]
    fn to_dense_lays_out_sectors_in_order() {
        // Two row sectors (sizes 1, 2) and one col sector (size 2): 3×2 dense.
        let mut t = BlockSparseTensor::new(2, vec![table(&[(0, 1), (1, 2)]), table(&[(5, 2)])])
            .expect("value should be present");
        t.insert_block(vec![0, 5], vec![1.0, 2.0])
            .expect("insert_block should succeed");
        t.insert_block(vec![1, 5], vec![3.0, 4.0, 5.0, 6.0])
            .expect("value should be present");
        let (rows, cols, dense) = t.to_dense().expect("to_dense should succeed");
        assert_eq!((rows, cols), (3, 2));
        // Row 0 from sector 0, rows 1-2 from sector 1.
        assert_eq!(dense, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn from_dense_2d_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = BlockSparseTensor::from_dense_2d(&data, 2, 3, 0, 0)
            .expect("from_dense_2d should succeed");
        assert_eq!(t.num_blocks(), 1);
        let (rows, cols, dense) = t.to_dense().expect("to_dense should succeed");
        assert_eq!((rows, cols), (2, 3));
        assert_eq!(dense, data);
    }

    #[test]
    fn from_dense_2d_rejects_bad_length() {
        assert!(BlockSparseTensor::from_dense_2d(&[1.0, 2.0], 2, 3, 0, 0).is_err());
    }

    #[test]
    fn dense_matmul_matches_block_contract() {
        // Cross-check: block-diagonal contract == dense matmul of the embeddings.
        let a_data = vec![1.0, 2.0, 0.0, 3.0]; // 2×2
        let b_data = vec![4.0, 0.0, 1.0, 5.0]; // 2×2
        let a = BlockSparseTensor::from_dense_2d(&a_data, 2, 2, 0, 0)
            .expect("from_dense_2d should succeed");
        let b = BlockSparseTensor::from_dense_2d(&b_data, 2, 2, 0, 0)
            .expect("from_dense_2d should succeed");
        let c = a
            .contract_matmul(&b)
            .expect("contract_matmul should succeed");
        let (_, _, dense_c) = c.to_dense().expect("to_dense should succeed");
        let expect = matmul(&a_data, &b_data, 2, 2, 2);
        for (x, y) in dense_c.iter().zip(expect.iter()) {
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
    }
}
