//! Hierarchical / tree-structured encoding for hyperdimensional computing.
//!
//! A rooted tree (or general hierarchy) of leaf symbols is encoded into a single binary
//! hypervector so that the *structure* — the parent/child paths down to each leaf — is
//! preserved, not merely the multiset of leaf symbols. This is the canonical VSA construction
//! for recursive and tree-structured data (Kanerva 2009; Plate 2003 "Holographic Reduced
//! Representations"; Gayler 2003 "Vector Symbolic Architectures"): position within the tree is
//! represented by a position-dependent permutation applied along each edge, and the bundled
//! superposition of all permuted leaf hypervectors yields a single fixed-width hypervector.
//!
//! Concretely a *child-position permutation* `ρ_c` is the unit circular shift by `c + 1`, where
//! `c` is the index of a child among its parent's children. Two encodings are provided, both
//! producing `Vec<i8>` in `{−1, +1}` to match the crate-wide binary hypervector representation:
//!
//! - **Recursive bundle encoding** ([`TreeHdEncoder::encode`]). A leaf with symbol `s` encodes
//!   to its symbol hypervector `H(s)`. An internal node with children `[t₀, t₁, …, t_{m-1}]`
//!   encodes to the binary majority bundle `⨁_{c} ρ_c(encode(t_c))`, i.e. each child's encoding
//!   is circularly shifted by its child index plus one and the shifted children are superposed.
//!   Because `ρ_c(h)` is (nearly) orthogonal to `h` for `c ≠ 0`, reordering the children of a
//!   node changes which shifted copies are bundled and so changes the encoding; nesting the same
//!   leaves more deeply applies more shifts and likewise changes the encoding.
//!
//! - **Path-permutation encoding** ([`TreeHdEncoder::encode_paths`]). The equivalent
//!   "flattened" view: every leaf is enumerated together with the full sequence of child indices
//!   on the root→leaf path, the per-edge shifts are *composed* into a single circular shift by
//!   `Σ (child_index + 1)` over the path, that shift is applied to the leaf's symbol
//!   hypervector, and all leaf hypervectors are bundled. Because circular shifts compose
//!   additively (`ρ^{a} ∘ ρ^{b} = ρ^{a+b}`), this reproduces the recursion-by-bundle structure
//!   from the leaves' point of view and is, like [`TreeHdEncoder::encode`], sensitive to both
//!   child order and nesting depth.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::ops::bundling::bundle_binary;
use crate::ops::permutation::cyclic_shift;
use crate::vector::binary::random_binary;

/// A node of an input hierarchy supplied to a [`TreeHdEncoder`].
///
/// The tree is an owned recursive enum: a [`TreeNode::Leaf`] carries the index of a leaf symbol
/// (resolved against the encoder's symbol vocabulary), and a [`TreeNode::Internal`] carries its
/// ordered children. The child *order* is significant — it drives the per-edge position
/// permutation — so `Internal(vec![Leaf(0), Leaf(1)])` and `Internal(vec![Leaf(1), Leaf(0)])`
/// encode to different hypervectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeNode {
    /// A leaf carrying the index of its symbol in the encoder's vocabulary.
    Leaf(usize),
    /// An internal node carrying its ordered children.
    Internal(Vec<TreeNode>),
}

/// Encoder mapping a rooted tree of leaf symbols into a single binary hypervector.
///
/// The encoder owns a vocabulary of random leaf-symbol hypervectors (one per distinct leaf
/// symbol). Structure is injected purely through circular-shift permutations keyed by child
/// position, so the encoder itself is otherwise stateless across calls.
pub struct TreeHdEncoder {
    /// Hypervector dimension.
    dim: usize,
    /// Number of distinct leaf symbols.
    n_symbols: usize,
    /// Random hypervector per distinct leaf symbol.
    symbol_hvs: Vec<Vec<i8>>,
}

impl TreeHdEncoder {
    /// Create an encoder over `n_symbols` distinct leaf symbols for hypervectors of dimension
    /// `dim`, generating one random hypervector per symbol from `rng`.
    ///
    /// # Errors
    ///
    /// - [`HdcError::ZeroDimension`] if `dim == 0`.
    /// - [`HdcError::EmptyInput`] if `n_symbols == 0`.
    pub fn new(n_symbols: usize, dim: usize, rng: &mut LcgRng) -> HdcResult<Self> {
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        if n_symbols == 0 {
            return Err(HdcError::EmptyInput);
        }
        let mut symbol_hvs = Vec::with_capacity(n_symbols);
        for _ in 0..n_symbols {
            symbol_hvs.push(random_binary(dim, rng)?);
        }
        Ok(Self {
            dim,
            n_symbols,
            symbol_hvs,
        })
    }

    /// Hypervector dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of distinct leaf symbols.
    #[must_use]
    pub fn n_symbols(&self) -> usize {
        self.n_symbols
    }

    /// Hypervector for a distinct leaf symbol.
    ///
    /// # Errors
    ///
    /// - [`HdcError::FeatureIndexOutOfRange`] if `s >= n_symbols`.
    pub fn symbol_hv(&self, s: usize) -> HdcResult<&[i8]> {
        if s >= self.n_symbols {
            return Err(HdcError::FeatureIndexOutOfRange {
                feat: s,
                max: self.n_symbols,
            });
        }
        Ok(&self.symbol_hvs[s])
    }

    /// Encode a tree by recursion-and-bundle.
    ///
    /// A leaf encodes to its symbol hypervector; an internal node encodes to the binary majority
    /// bundle of its children's encodings, each circularly shifted by its child index plus one
    /// (`⨁_{c} ρ_{c}(encode(child_c))`). The position-dependent shift makes the encoding
    /// sensitive to child order, and the recursion makes it sensitive to nesting depth.
    ///
    /// `rng` is only consulted to break ties in the majority bundle (which occur solely when an
    /// even number of contributions cancel exactly at a component).
    ///
    /// # Errors
    ///
    /// - [`HdcError::FeatureIndexOutOfRange`] if any leaf symbol index is `>= n_symbols`.
    /// - [`HdcError::EmptyInput`] if any internal node has no children.
    pub fn encode(&self, tree: &TreeNode, rng: &mut LcgRng) -> HdcResult<Vec<i8>> {
        match tree {
            TreeNode::Leaf(s) => {
                let s = *s;
                if s >= self.n_symbols {
                    return Err(HdcError::FeatureIndexOutOfRange {
                        feat: s,
                        max: self.n_symbols,
                    });
                }
                Ok(self.symbol_hvs[s].clone())
            }
            TreeNode::Internal(children) => {
                if children.is_empty() {
                    return Err(HdcError::EmptyInput);
                }
                let mut shifted: Vec<Vec<i8>> = Vec::with_capacity(children.len());
                for (c, child) in children.iter().enumerate() {
                    let child_code = self.encode(child, rng)?;
                    // ρ_c is the circular shift by c + 1, so even the first child (c = 0) is
                    // shifted relative to its raw symbol — a depth-1 subtree never collides with
                    // the bare leaf at the same position.
                    shifted.push(cyclic_shift(&child_code, c + 1)?);
                }
                bundle_binary(&shifted, rng)
            }
        }
    }

    /// Encode a tree by the path-permutation (flattened-leaves) view.
    ///
    /// Each leaf is enumerated with the full child-index path from the root, the per-edge shifts
    /// are composed into a single circular shift by `Σ (child_index + 1)` over that path, the
    /// shift is applied to the leaf's symbol hypervector, and all leaf hypervectors are bundled.
    /// Because circular shifts compose additively, the *set* of shift amounts here mirrors the
    /// shifts that [`Self::encode`] applies along the way, so this encoding is likewise sensitive
    /// to child order and to nesting depth.
    ///
    /// `rng` is only consulted to break ties in the majority bundle.
    ///
    /// # Errors
    ///
    /// - [`HdcError::FeatureIndexOutOfRange`] if any leaf symbol index is `>= n_symbols`.
    /// - [`HdcError::EmptyInput`] if any internal node has no children (so the tree contributes no
    ///   leaves).
    pub fn encode_paths(&self, tree: &TreeNode, rng: &mut LcgRng) -> HdcResult<Vec<i8>> {
        let mut leaf_hvs: Vec<Vec<i8>> = Vec::new();
        // Accumulated shift along the current root→node path is Σ (child_index + 1); the root is
        // reached with an empty path, hence an accumulated shift of zero.
        self.collect_path_leaves(tree, 0, &mut leaf_hvs)?;
        if leaf_hvs.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        bundle_binary(&leaf_hvs, rng)
    }

    /// Depth-first walk collecting each leaf's symbol hypervector shifted by the composed path
    /// permutation `ρ^{path_shift}`, where `path_shift = Σ (child_index + 1)` along the root→leaf
    /// path. Internal nodes with no children abort with [`HdcError::EmptyInput`] to match the
    /// recursive encoder's treatment of empty subtrees.
    fn collect_path_leaves(
        &self,
        node: &TreeNode,
        path_shift: usize,
        out: &mut Vec<Vec<i8>>,
    ) -> HdcResult<()> {
        match node {
            TreeNode::Leaf(s) => {
                let s = *s;
                if s >= self.n_symbols {
                    return Err(HdcError::FeatureIndexOutOfRange {
                        feat: s,
                        max: self.n_symbols,
                    });
                }
                if path_shift == 0 {
                    // A bare leaf as the whole tree: no edge traversed, identity permutation.
                    out.push(self.symbol_hvs[s].clone());
                } else {
                    out.push(cyclic_shift(&self.symbol_hvs[s], path_shift)?);
                }
                Ok(())
            }
            TreeNode::Internal(children) => {
                if children.is_empty() {
                    return Err(HdcError::EmptyInput);
                }
                for (c, child) in children.iter().enumerate() {
                    self.collect_path_leaves(child, path_shift + c + 1, out)?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::cosine::cosine_binary;

    fn encoder(seed: u64, n_symbols: usize, dim: usize) -> TreeHdEncoder {
        let mut rng = LcgRng::new(seed);
        TreeHdEncoder::new(n_symbols, dim, &mut rng).expect("encoder")
    }

    #[test]
    fn new_rejects_bad_args() {
        // Zero dimension and zero symbol vocabulary are both rejected.
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            TreeHdEncoder::new(4, 0, &mut rng),
            Err(HdcError::ZeroDimension)
        ));
        assert!(matches!(
            TreeHdEncoder::new(0, 1024, &mut rng),
            Err(HdcError::EmptyInput)
        ));
    }

    #[test]
    fn single_leaf_encodes_to_symbol_hv() {
        // A bare leaf encodes exactly to its symbol hypervector (rng-independent).
        let dim = 1024;
        let enc = encoder(11, 4, dim);
        let tree = TreeNode::Leaf(2);
        let mut r1 = LcgRng::new(7);
        let mut r2 = LcgRng::new(123_456);
        let h1 = enc.encode(&tree, &mut r1).expect("h1");
        let h2 = enc.encode(&tree, &mut r2).expect("h2");
        assert_eq!(h1.len(), dim);
        assert_eq!(h1, h2, "single leaf must be rng-independent");
        assert_eq!(
            h1.as_slice(),
            enc.symbol_hv(2).expect("symbol"),
            "single leaf must equal its symbol HV"
        );
    }

    #[test]
    fn same_tree_same_encoding() {
        // Identical trees → identical encodings under same-seed bundle tie-breaks.
        let dim = 2048;
        let enc = encoder(22, 6, dim);
        let tree = TreeNode::Internal(vec![
            TreeNode::Leaf(0),
            TreeNode::Internal(vec![TreeNode::Leaf(1), TreeNode::Leaf(2)]),
            TreeNode::Leaf(3),
        ]);
        let mut r1 = LcgRng::new(5);
        let mut r2 = LcgRng::new(5);
        let h1 = enc.encode(&tree, &mut r1).expect("h1");
        let h2 = enc.encode(&tree, &mut r2).expect("h2");
        assert_eq!(h1, h2, "same tree must encode deterministically");
        assert_eq!(h1.len(), dim);
    }

    #[test]
    fn reordering_children_changes_encoding() {
        // Reordering the children of an internal node clearly changes the encoding.
        let dim = 4096;
        let enc = encoder(33, 4, dim);
        let tree = TreeNode::Internal(vec![
            TreeNode::Leaf(0),
            TreeNode::Leaf(1),
            TreeNode::Leaf(2),
        ]);
        let reordered = TreeNode::Internal(vec![
            TreeNode::Leaf(2),
            TreeNode::Leaf(0),
            TreeNode::Leaf(1),
        ]);
        let mut r1 = LcgRng::new(9);
        let mut r2 = LcgRng::new(9);
        let h = enc.encode(&tree, &mut r1).expect("h");
        let hr = enc.encode(&reordered, &mut r2).expect("hr");
        let sim = cosine_binary(&h, &hr).expect("cos");
        assert!(
            sim < 0.9,
            "child reordering should drop similarity clearly: sim={sim:.3}"
        );
    }

    #[test]
    fn deeper_nesting_differs_from_shallow() {
        // The same leaves arranged flat vs. nested encode differently.
        let dim = 4096;
        let enc = encoder(44, 4, dim);
        let shallow = TreeNode::Internal(vec![
            TreeNode::Leaf(0),
            TreeNode::Leaf(1),
            TreeNode::Leaf(2),
        ]);
        let nested = TreeNode::Internal(vec![
            TreeNode::Leaf(0),
            TreeNode::Internal(vec![TreeNode::Leaf(1), TreeNode::Leaf(2)]),
        ]);
        let mut r1 = LcgRng::new(3);
        let mut r2 = LcgRng::new(3);
        let hs = enc.encode(&shallow, &mut r1).expect("hs");
        let hn = enc.encode(&nested, &mut r2).expect("hn");
        let sim = cosine_binary(&hs, &hn).expect("cos");
        assert!(
            sim < 0.9,
            "nesting depth should change the encoding: sim={sim:.3}"
        );
    }

    #[test]
    fn out_of_range_symbol_errors() {
        // Leaf symbol indices >= n_symbols are rejected by both encoders and the accessor.
        let dim = 1024;
        let enc = encoder(55, 3, dim);
        let bad_leaf = TreeNode::Leaf(3);
        let bad_nested = TreeNode::Internal(vec![TreeNode::Leaf(0), TreeNode::Leaf(7)]);
        let mut r = LcgRng::new(1);
        assert!(matches!(
            enc.encode(&bad_leaf, &mut r),
            Err(HdcError::FeatureIndexOutOfRange { feat: 3, max: 3 })
        ));
        assert!(matches!(
            enc.encode(&bad_nested, &mut r),
            Err(HdcError::FeatureIndexOutOfRange { feat: 7, max: 3 })
        ));
        assert!(matches!(
            enc.encode_paths(&bad_leaf, &mut r),
            Err(HdcError::FeatureIndexOutOfRange { feat: 3, max: 3 })
        ));
        assert!(matches!(
            enc.symbol_hv(3),
            Err(HdcError::FeatureIndexOutOfRange { feat: 3, max: 3 })
        ));
    }

    #[test]
    fn empty_internal_node_errors() {
        // An internal node with no children is rejected by both encoders.
        let dim = 512;
        let enc = encoder(66, 4, dim);
        let empty = TreeNode::Internal(Vec::new());
        let nested_empty =
            TreeNode::Internal(vec![TreeNode::Leaf(0), TreeNode::Internal(Vec::new())]);
        let mut r = LcgRng::new(1);
        assert!(matches!(
            enc.encode(&empty, &mut r),
            Err(HdcError::EmptyInput)
        ));
        assert!(matches!(
            enc.encode(&nested_empty, &mut r),
            Err(HdcError::EmptyInput)
        ));
        assert!(matches!(
            enc.encode_paths(&empty, &mut r),
            Err(HdcError::EmptyInput)
        ));
        assert!(matches!(
            enc.encode_paths(&nested_empty, &mut r),
            Err(HdcError::EmptyInput)
        ));
    }

    #[test]
    fn encode_paths_is_order_sensitive() {
        // The leaf-path encoder is also sensitive to child order, and deterministic.
        let dim = 4096;
        let enc = encoder(77, 4, dim);
        let tree = TreeNode::Internal(vec![
            TreeNode::Leaf(0),
            TreeNode::Internal(vec![TreeNode::Leaf(1), TreeNode::Leaf(2)]),
            TreeNode::Leaf(3),
        ]);
        let reordered = TreeNode::Internal(vec![
            TreeNode::Leaf(3),
            TreeNode::Internal(vec![TreeNode::Leaf(2), TreeNode::Leaf(1)]),
            TreeNode::Leaf(0),
        ]);
        let mut r1 = LcgRng::new(2);
        let mut r2 = LcgRng::new(2);
        let h = enc.encode_paths(&tree, &mut r1).expect("h");
        let hr = enc.encode_paths(&reordered, &mut r2).expect("hr");
        assert_eq!(h.len(), dim);
        // Determinism: re-encoding the same tree gives the same result.
        let h_again = enc.encode_paths(&tree, &mut LcgRng::new(2)).expect("again");
        assert_eq!(h, h_again, "path encoding must be deterministic");
        let sim = cosine_binary(&h, &hr).expect("cos");
        assert!(
            sim < 0.9,
            "path encoding should be order sensitive: sim={sim:.3}"
        );
    }

    #[test]
    fn encode_paths_single_leaf_matches_symbol() {
        // For a bare leaf the path encoder applies the identity permutation → symbol HV.
        let dim = 1024;
        let enc = encoder(88, 5, dim);
        let tree = TreeNode::Leaf(4);
        let h = enc.encode_paths(&tree, &mut LcgRng::new(1)).expect("h");
        assert_eq!(h.as_slice(), enc.symbol_hv(4).expect("symbol"));
    }

    #[test]
    fn accessors_report_configuration() {
        let dim = 2048;
        let enc = encoder(99, 7, dim);
        assert_eq!(enc.dim(), dim);
        assert_eq!(enc.n_symbols(), 7);
        assert_eq!(enc.symbol_hv(0).expect("symbol").len(), dim);
        assert_eq!(enc.symbol_hv(6).expect("symbol").len(), dim);
    }
}
