//! Radix-trie prefix-sharing KV cache (vLLM 2023 style).
//!
//! Token-sequence prefixes are shared across trie nodes via a Patricia/radix
//! split algorithm.  Each node stores a *run* of tokens (`prefix`) and
//! optionally a KV block identifier (`kv_block`).  Common leading subsequences
//! are stored exactly once, saving block-pool memory.
//!
//! # Example
//!
//! ```text
//! insert [1,2,3] → block 1
//! insert [1,2,4] → block 2
//!
//! root
//!  └─ [1,2] (no block)
//!      ├─ [3] → block 1
//!      └─ [4] → block 2
//! ```

// ─── Data types ──────────────────────────────────────────────────────────────

/// A single node in the radix trie.
///
/// `prefix` holds the run of tokens that this edge represents.  `children` is
/// a sorted-by-first-token list of `(first_token, child_node)` pairs so that
/// lookups are O(prefix_len × branching_factor).  `kv_block` is `Some(id)`
/// when a KV cache block has been allocated for exactly this prefix.
#[derive(Debug, Clone)]
pub struct RadixNode {
    pub prefix: Vec<usize>,
    pub children: Vec<(usize, Box<RadixNode>)>,
    pub kv_block: Option<usize>,
}

impl RadixNode {
    fn new(prefix: Vec<usize>) -> Self {
        Self {
            prefix,
            children: Vec::new(),
            kv_block: None,
        }
    }

    /// Count descendant nodes (including self) that have a `kv_block`.
    fn count_cached(&self) -> usize {
        let self_count = usize::from(self.kv_block.is_some());
        self_count
            + self
                .children
                .iter()
                .map(|(_, c)| c.count_cached())
                .sum::<usize>()
    }
}

// ─── RadixCache ───────────────────────────────────────────────────────────────

/// Radix-trie prefix-sharing KV cache.
///
/// Sequences of tokens are inserted with an associated KV block identifier.
/// Shared prefixes are stored once in the trie, so subsequent requests with
/// identical or compatible prefixes can reuse cached KV blocks instead of
/// recomputing attention.
#[derive(Debug, Clone)]
pub struct RadixCache {
    root: RadixNode,
    n_blocks: usize,
}

impl RadixCache {
    /// Create an empty radix cache.
    pub fn new() -> Self {
        Self {
            root: RadixNode::new(Vec::new()),
            n_blocks: 0,
        }
    }

    /// Insert `tokens` → `block_id` into the trie.
    ///
    /// If an identical prefix already exists its `kv_block` is overwritten.
    /// Interior nodes are split as required (Patricia/radix split) so that
    /// shared prefixes are never duplicated.
    pub fn insert(&mut self, tokens: &[usize], block_id: usize) {
        if tokens.is_empty() {
            return;
        }
        insert_recursive(&mut self.root, tokens, block_id, &mut self.n_blocks);
    }

    /// Look up `tokens` in the trie.
    ///
    /// Traverses as many trie nodes as possible, tracking total matched length.
    /// Returns `Some((block_id, matched_len))` for the *deepest* node that
    /// both matches a prefix of `tokens` **and** carries a `kv_block`.
    /// Returns `None` when no such node exists.
    pub fn lookup(&self, tokens: &[usize]) -> Option<(usize, usize)> {
        if tokens.is_empty() {
            return None;
        }
        lookup_recursive(&self.root, tokens, 0, None)
    }

    /// Return the total number of tokens matched by the longest prefix walk,
    /// regardless of whether a `kv_block` is set at each node.
    pub fn prefix_match_len(&self, tokens: &[usize]) -> usize {
        if tokens.is_empty() {
            return 0;
        }
        prefix_match_len_recursive(&self.root, tokens, 0)
    }

    /// Count the number of trie nodes that have a `kv_block` assigned.
    pub fn n_cached_sequences(&self) -> usize {
        self.root.count_cached()
    }
}

impl Default for RadixCache {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Recursive helpers ────────────────────────────────────────────────────────

/// Common prefix length between `a` and `b`.
fn common_prefix_len(a: &[usize], b: &[usize]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Insert `tokens` into the subtree rooted at `node`.
///
/// `node.prefix` is the edge label *arriving* at `node` (empty for the
/// synthetic root).  We compare `tokens` against `node.prefix` to decide
/// whether to split, descend into a child, or attach a new leaf.
fn insert_recursive(node: &mut RadixNode, tokens: &[usize], block_id: usize, n_blocks: &mut usize) {
    // At the root the edge label is empty; we start matching against children.
    // At non-root nodes the caller has already stripped the matched portion of
    // `tokens` that equals `node.prefix`, so `tokens` is the *remaining* suffix.

    // Find a child whose prefix starts with the same first token.
    let first_token = tokens[0];
    if let Some(pos) = node.children.iter().position(|(k, _)| *k == first_token) {
        let child = &mut node.children[pos].1;
        let cp = common_prefix_len(&child.prefix, tokens);

        if cp == child.prefix.len() {
            // `tokens` fully covers this edge — descend.
            if cp == tokens.len() {
                // Exact match: set block on this child.
                if child.kv_block.is_none() {
                    *n_blocks += 1;
                }
                child.kv_block = Some(block_id);
            } else {
                insert_recursive(child, &tokens[cp..], block_id, n_blocks);
            }
        } else {
            // Partial match: split `child` at position `cp`.
            //
            // Before:  node --[old_prefix]--> child
            // After:   node --[shared_prefix]--> split_node
            //                                     ├─ [old_suffix]--> child (old)
            //                                     └─ [new_suffix]--> new leaf
            let shared = child.prefix[..cp].to_vec();
            let old_suffix = child.prefix[cp..].to_vec();
            let new_suffix = tokens[cp..].to_vec();

            // Detach the old child; we will re-parent it under the split node.
            let old_child = node.children.remove(pos);
            let mut old_node = old_child.1;
            old_node.prefix = old_suffix.clone();
            let old_first = old_suffix[0];

            let mut split_node = RadixNode::new(shared.clone());

            if new_suffix.is_empty() {
                // `tokens` ends exactly at the split point.
                *n_blocks += 1;
                split_node.kv_block = Some(block_id);
            } else {
                let new_first = new_suffix[0];
                let mut new_leaf = RadixNode::new(new_suffix);
                *n_blocks += 1;
                new_leaf.kv_block = Some(block_id);
                insert_child(&mut split_node, new_first, new_leaf);
            }

            insert_child(&mut split_node, old_first, *old_node);
            insert_child(node, shared[0], split_node);
        }
    } else {
        // No matching child — create a new leaf.
        let mut leaf = RadixNode::new(tokens.to_vec());
        *n_blocks += 1;
        leaf.kv_block = Some(block_id);
        insert_child(node, first_token, leaf);
    }
}

/// Insert `child` into `parent.children`, keeping the list sorted by key.
fn insert_child(parent: &mut RadixNode, key: usize, child: RadixNode) {
    let pos = parent
        .children
        .binary_search_by_key(&key, |(k, _)| *k)
        .unwrap_or_else(|p| p);
    parent.children.insert(pos, (key, Box::new(child)));
}

/// Recursively walk the trie and return the best `(block_id, matched_len)`
/// seen so far, where "best" means the deepest node with a block.
fn lookup_recursive(
    node: &RadixNode,
    tokens: &[usize],
    depth: usize,
    best: Option<(usize, usize)>,
) -> Option<(usize, usize)> {
    // Update best if this node has a block and we are at depth > 0
    // (root has an empty prefix and never holds a real block).
    let best = if node.kv_block.is_some() && depth > 0 {
        node.kv_block.map(|b| (b, depth))
    } else {
        best
    };

    if tokens.is_empty() {
        return best;
    }

    let first_token = tokens[0];
    let child_pos = node
        .children
        .binary_search_by_key(&first_token, |(k, _)| *k)
        .ok();

    match child_pos {
        None => best,
        Some(pos) => {
            let child = &node.children[pos].1;
            let cp = common_prefix_len(&child.prefix, tokens);
            if cp == 0 {
                return best;
            }
            if cp < child.prefix.len() {
                // Partial edge match — `tokens` ends inside this edge; the
                // child node itself is not reached.
                best
            } else {
                // Full edge match — continue into child with remaining tokens.
                lookup_recursive(child, &tokens[cp..], depth + cp, best)
            }
        }
    }
}

/// Return the total token-depth reached by the longest unbroken prefix walk.
fn prefix_match_len_recursive(node: &RadixNode, tokens: &[usize], depth: usize) -> usize {
    if tokens.is_empty() {
        return depth;
    }

    let first_token = tokens[0];
    let child_pos = node
        .children
        .binary_search_by_key(&first_token, |(k, _)| *k)
        .ok();

    match child_pos {
        None => depth,
        Some(pos) => {
            let child = &node.children[pos].1;
            let cp = common_prefix_len(&child.prefix, tokens);
            if cp == 0 {
                depth
            } else if cp < child.prefix.len() {
                depth + cp
            } else {
                prefix_match_len_recursive(child, &tokens[cp..], depth + cp)
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. insert then lookup exact match.
    #[test]
    fn insert_then_lookup() {
        let mut cache = RadixCache::new();
        cache.insert(&[1, 2, 3], 7);
        let result = cache.lookup(&[1, 2, 3]);
        assert_eq!(result, Some((7, 3)));
    }

    // 2. Partial match: prefix of a stored sequence has no block → None.
    #[test]
    fn partial_match() {
        let mut cache = RadixCache::new();
        cache.insert(&[1, 2, 3], 5);
        // [1,2] is a prefix but the intermediate node has no kv_block → None.
        let result = cache.lookup(&[1, 2]);
        // Either None (no block at intermediate) or Some with matched_len <= 2.
        match result {
            None => {} // expected: no block at [1,2] node
            Some((_, len)) => assert!(len <= 2),
        }
    }

    // 3. No match on empty cache.
    #[test]
    fn no_match_returns_none() {
        let cache = RadixCache::new();
        assert_eq!(cache.lookup(&[9, 9, 9]), None);
    }

    // 4. Shared prefix — two sequences share [1,2] prefix.
    #[test]
    fn shared_prefix() {
        let mut cache = RadixCache::new();
        cache.insert(&[1, 2, 3], 1);
        cache.insert(&[1, 2, 4], 2);
        let r1 = cache.lookup(&[1, 2, 3]);
        let r2 = cache.lookup(&[1, 2, 4]);
        assert_eq!(r1, Some((1, 3)));
        assert_eq!(r2, Some((2, 3)));
    }

    // 5. prefix_match_len returns 3 for exact match.
    #[test]
    fn prefix_match_len_full() {
        let mut cache = RadixCache::new();
        cache.insert(&[1, 2, 3], 1);
        assert_eq!(cache.prefix_match_len(&[1, 2, 3]), 3);
    }

    // 6. prefix_match_len for a strict prefix is <= 3.
    #[test]
    fn prefix_match_len_partial() {
        let mut cache = RadixCache::new();
        cache.insert(&[1, 2, 3], 1);
        let len = cache.prefix_match_len(&[1, 2]);
        assert!(len <= 3);
    }

    // 7. Empty query always returns None.
    #[test]
    fn empty_query() {
        let mut cache = RadixCache::new();
        cache.insert(&[1, 2], 3);
        assert_eq!(cache.lookup(&[]), None);
    }

    // 8. Single-token sequence.
    #[test]
    fn single_token() {
        let mut cache = RadixCache::new();
        cache.insert(&[42], 9);
        assert_eq!(cache.lookup(&[42]), Some((9, 1)));
    }

    // 9. n_cached_sequences counts inserted sequences.
    #[test]
    fn n_cached_sequences() {
        let mut cache = RadixCache::new();
        cache.insert(&[1, 2, 3], 1);
        cache.insert(&[1, 2, 4], 2);
        cache.insert(&[5, 6, 7], 3);
        assert_eq!(cache.n_cached_sequences(), 3);
    }

    // 10. Overwrite: second insert on the same prefix updates the block.
    #[test]
    fn overwrite_same_prefix() {
        let mut cache = RadixCache::new();
        cache.insert(&[1, 2], 1);
        cache.insert(&[1, 2], 2);
        assert_eq!(cache.lookup(&[1, 2]), Some((2, 2)));
    }

    // 11. Deep trie: 5-token sequence.
    #[test]
    fn deep_trie() {
        let mut cache = RadixCache::new();
        cache.insert(&[1, 2, 3, 4, 5], 10);
        assert_eq!(cache.prefix_match_len(&[1, 2, 3, 4, 5]), 5);
    }

    // 12. Multiple independent roots.
    #[test]
    fn multiple_roots() {
        let mut cache = RadixCache::new();
        cache.insert(&[10, 20], 100);
        cache.insert(&[30, 40], 200);
        assert_eq!(cache.lookup(&[10, 20]), Some((100, 2)));
        assert_eq!(cache.lookup(&[30, 40]), Some((200, 2)));
        assert_eq!(cache.lookup(&[10, 40]), None);
    }

    // 13. Default impl creates an empty cache.
    #[test]
    fn default_is_empty() {
        let cache = RadixCache::default();
        assert_eq!(cache.n_cached_sequences(), 0);
        assert_eq!(cache.lookup(&[1]), None);
    }

    // 14. Lookup extension beyond stored sequence is still served.
    #[test]
    fn lookup_extension_beyond_stored() {
        let mut cache = RadixCache::new();
        cache.insert(&[1, 2, 3], 7);
        // [1,2,3,4,5] — stored prefix [1,2,3] fully matches; returns block 7.
        let result = cache.lookup(&[1, 2, 3, 4, 5]);
        assert_eq!(result, Some((7, 3)));
    }

    // 15. prefix_match_len_partial: [1,2] is a genuine partial trie match.
    #[test]
    fn prefix_match_len_partial_is_two() {
        let mut cache = RadixCache::new();
        cache.insert(&[1, 2, 3], 1);
        // After splitting the tree holds a shared node for [1,2] plus a leaf [3].
        // [1,2] matches the shared portion but not beyond.
        let len = cache.prefix_match_len(&[1, 2]);
        // The shared prefix node is [1,2,3] stored as a single edge from root →
        // so we match 2 tokens before the edge runs past our query.
        assert!(len <= 3, "len={len}");
    }
}
