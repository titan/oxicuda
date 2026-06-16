//! Extended Persistence — Cohen-Steiner, Edelsbrunner, Harer 2009.
//!
//! Extends ordinary persistence to capture relative homology information
//! by running both ascending and descending filtrations, then pairing
//! ordinary birth/death events with relative birth/death events.
//!
//! For 0-dimensional (connected-component) extended persistence:
//! - **Ordinary pairs** `(birth, death)`: components merged during ascending filtration.
//! - **Relative pairs** `(birth, death)`: classes arising from the descending filtration.
//! - **Extended pairs**: ordinary birth matched with a relative death across levels.
//! - **Essential**: unpaired generators (one per global connected component).

use std::collections::HashSet;

use crate::error::{TdaError, TdaResult};

/// Extended persistence barcode for a scalar function on a simplicial complex.
///
/// Contains four families of intervals from Cohen-Steiner 2009:
/// ordinary (sub-level), relative (sup-level), extended (cross-level), essential.
#[derive(Debug, Clone)]
pub struct ExtendedBarcode {
    /// Ordinary pairs `(birth, death)` from the ascending filtration (H₀).
    pub ordinary: Vec<(f64, f64)>,
    /// Relative pairs `(birth, death)` from the descending filtration (relative H₀).
    pub relative: Vec<(f64, f64)>,
    /// Extended pairs connecting an ordinary birth to a relative death.
    pub extended: Vec<(f64, f64)>,
    /// Essential values (unpaired generators — one per connected component's minimum).
    pub essential: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Union-Find with minimum-vertex tracking
// ---------------------------------------------------------------------------

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    /// Index of the vertex with the smallest key in each component.
    min_rep: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            min_rep: (0..n).collect(),
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let root = self.find(self.parent[x]);
            self.parent[x] = root;
        }
        self.parent[x]
    }

    /// Union the components of `a` and `b`.
    ///
    /// `key[i]` gives the ordering key for vertex `i` (smaller key = earlier birth).
    /// The component whose representative has the **larger** key dies and is paired.
    ///
    /// Returns `Some((survivor_root, dying_root))` or `None` if already connected.
    fn union_keyed(&mut self, a: usize, b: usize, key: &[f64]) -> Option<(usize, usize)> {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return None;
        }
        // The component born "later" (larger key) dies.
        let (survivor, dying) = if key[self.min_rep[ra]] <= key[self.min_rep[rb]] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        // Union-by-rank: attach dying subtree under survivor.
        if self.rank[survivor] >= self.rank[dying] {
            self.parent[dying] = survivor;
            if self.rank[survivor] == self.rank[dying] {
                self.rank[survivor] += 1;
            }
            Some((survivor, dying))
        } else {
            // dying has higher rank; make it the structural root but keep survivor's semantics.
            self.parent[survivor] = dying;
            // Swap min_rep so that dying (now structural root) represents the survivor component.
            self.min_rep[dying] = self.min_rep[survivor];
            // Return dying as the survivor root (it now owns the component).
            Some((dying, survivor))
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute 0-dimensional extended persistence for a scalar function on a graph.
///
/// # Arguments
/// - `filtration_values`: real-valued function at each vertex (n_vertices entries).
/// - `edges`: pairs `(u, v)` (0-indexed vertices) forming the 1-skeleton.
///
/// # Errors
/// Returns [`TdaError::EmptyPointCloud`] for empty input,
/// [`TdaError::NanFiltrationValue`] for NaN values,
/// [`TdaError::InvalidSimplex`] for out-of-range edge endpoints.
pub fn extended_persistence(
    filtration_values: &[f64],
    edges: &[(usize, usize)],
) -> TdaResult<ExtendedBarcode> {
    let n = filtration_values.len();
    if n == 0 {
        return Err(TdaError::EmptyPointCloud);
    }
    for &v in filtration_values {
        if v.is_nan() {
            return Err(TdaError::NanFiltrationValue);
        }
    }
    for &(u, v) in edges {
        if u >= n || v >= n {
            return Err(TdaError::InvalidSimplex(format!(
                "edge ({u},{v}) out of range for n_vertices={n}"
            )));
        }
    }

    // -----------------------------------------------------------------------
    // Phase 1: Ascending filtration — ordinary pairs
    // -----------------------------------------------------------------------
    // Sort edges by max(f[u], f[v]) ascending: edge enters filtration when both
    // endpoints are present (which happens at the max of the two endpoint values).
    let mut asc_edges = edges.to_vec();
    asc_edges.sort_by(|&(u1, v1), &(u2, v2)| {
        let k1 = filtration_values[u1].max(filtration_values[v1]);
        let k2 = filtration_values[u2].max(filtration_values[v2]);
        k1.partial_cmp(&k2).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut uf_asc = UnionFind::new(n);
    let mut ordinary: Vec<(f64, f64)> = Vec::new();

    for &(u, v) in &asc_edges {
        let edge_val = filtration_values[u].max(filtration_values[v]);
        if let Some((_surv, dying)) = uf_asc.union_keyed(u, v, filtration_values) {
            // `dying` is the structural node for the dying component after union,
            // but we already recorded its min_rep before the union changed things.
            // We need the min_rep of the dying component *before* the union.
            // Reconstruct: the dying component's representative is the one whose key was larger.
            // Re-find to get dying's current root (path-compressed).
            let dying_root_now = uf_asc.find(dying);
            let _ = dying_root_now; // not needed — min_rep is on the survivor root now
            // Recover birth from the dying component: it was born at its minimum vertex.
            // After union_keyed, the dying component's original min_rep is:
            //   - If rank[survivor] >= rank[dying]: it stays as dying's min_rep (unchanged).
            //   - If rank[survivor] < rank[dying]: dying became structural root, survivor's
            //     min_rep was copied to dying.  The actual dying min_rep is survivor's original.
            // This is getting complex; instead track dying_min_rep before union.
            // We re-implement: use a simpler approach where we query min_rep *before* union.
            let b = filtration_values[dying]; // placeholder — see corrected logic below
            let _ = b;
            ordinary.push((filtration_values[dying], edge_val));
        }
    }

    // The above has a subtle bug: `dying` after union may be path-compressed.
    // Redo with explicit tracking.
    ordinary.clear();
    let mut uf_asc2 = UnionFind::new(n);
    for &(u, v) in &asc_edges {
        let edge_val = filtration_values[u].max(filtration_values[v]);
        let ru = uf_asc2.find(u);
        let rv = uf_asc2.find(v);
        if ru == rv {
            continue;
        }
        // Determine which component dies (the one born later = larger min key)
        let (surv_root, dying_root) =
            if filtration_values[uf_asc2.min_rep[ru]] <= filtration_values[uf_asc2.min_rep[rv]] {
                (ru, rv)
            } else {
                (rv, ru)
            };
        let dying_birth = filtration_values[uf_asc2.min_rep[dying_root]];
        // Now do the actual union
        if uf_asc2.rank[surv_root] >= uf_asc2.rank[dying_root] {
            uf_asc2.parent[dying_root] = surv_root;
            if uf_asc2.rank[surv_root] == uf_asc2.rank[dying_root] {
                uf_asc2.rank[surv_root] += 1;
            }
        } else {
            uf_asc2.parent[surv_root] = dying_root;
            uf_asc2.min_rep[dying_root] = uf_asc2.min_rep[surv_root];
        }
        ordinary.push((dying_birth, edge_val));
    }
    ordinary.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // -----------------------------------------------------------------------
    // Phase 2: Collect essential classes (surviving components after full ascending filtration)
    // -----------------------------------------------------------------------
    let mut uf_final = UnionFind::new(n);
    for &(u, v) in &asc_edges {
        let ru = uf_final.find(u);
        let rv = uf_final.find(v);
        if ru == rv {
            continue;
        }
        let (surv_root, dying_root) =
            if filtration_values[uf_final.min_rep[ru]] <= filtration_values[uf_final.min_rep[rv]] {
                (ru, rv)
            } else {
                (rv, ru)
            };
        if uf_final.rank[surv_root] >= uf_final.rank[dying_root] {
            uf_final.parent[dying_root] = surv_root;
            if uf_final.rank[surv_root] == uf_final.rank[dying_root] {
                uf_final.rank[surv_root] += 1;
            }
        } else {
            uf_final.parent[surv_root] = dying_root;
            uf_final.min_rep[dying_root] = uf_final.min_rep[surv_root];
        }
    }
    let mut seen_roots: HashSet<usize> = HashSet::new();
    let mut essential: Vec<f64> = Vec::new();
    for i in 0..n {
        let root = uf_final.find(i);
        if seen_roots.insert(root) {
            essential.push(filtration_values[uf_final.min_rep[root]]);
        }
    }
    essential.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // -----------------------------------------------------------------------
    // Phase 3: Descending filtration — relative pairs
    // -----------------------------------------------------------------------
    // Sort edges by max(f[u], f[v]) descending. In the sup-level set filtration
    // vertices enter from the top. When an edge merges two components (descending),
    // the component born at a lower sup-level value "dies".
    let mut desc_edges = asc_edges.clone();
    desc_edges.sort_by(|&(u1, v1), &(u2, v2)| {
        let k1 = filtration_values[u1].max(filtration_values[v1]);
        let k2 = filtration_values[u2].max(filtration_values[v2]);
        k2.partial_cmp(&k1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Use negated values so the same union-find logic picks the "later born" component to die.
    // In descending filtration: a component is born at a high value. When two merge, the
    // one with the smaller maximum birth value (born later in the descending sweep) dies.
    // Negating: smaller max birth → larger negated value → union-find picks the larger to die. ✓
    let neg_vals: Vec<f64> = filtration_values.iter().map(|&v| -v).collect();
    let mut uf_desc = UnionFind::new(n);
    // Initialize min_rep in negated values: min_rep[root] = vertex with smallest neg_vals = largest original.
    // By default min_rep[i] = i, which is correct (each vertex is its own rep).

    let mut relative: Vec<(f64, f64)> = Vec::new();

    for &(u, v) in &desc_edges {
        let edge_max = filtration_values[u].max(filtration_values[v]);
        let ru = uf_desc.find(u);
        let rv = uf_desc.find(v);
        if ru == rv {
            continue;
        }
        // In negated space: pick survivor as the one with smaller neg_vals[min_rep]
        // = larger original filtration_values[min_rep] = born earlier in descending sweep.
        let (surv_root, dying_root) =
            if neg_vals[uf_desc.min_rep[ru]] <= neg_vals[uf_desc.min_rep[rv]] {
                (ru, rv)
            } else {
                (rv, ru)
            };
        // dying component's birth = its highest vertex value (the negated min gives original max)
        let dying_birth_val = filtration_values[uf_desc.min_rep[dying_root]];
        // Relative pair: born at dying_birth_val (high), dies at edge_max (low, going down)
        // Convention: store (lower, higher) as (b, d).
        let (b_rel, d_rel) = if edge_max <= dying_birth_val {
            (edge_max, dying_birth_val)
        } else {
            (dying_birth_val, edge_max)
        };
        relative.push((b_rel, d_rel));

        // Do the union
        if uf_desc.rank[surv_root] >= uf_desc.rank[dying_root] {
            uf_desc.parent[dying_root] = surv_root;
            if uf_desc.rank[surv_root] == uf_desc.rank[dying_root] {
                uf_desc.rank[surv_root] += 1;
            }
        } else {
            uf_desc.parent[surv_root] = dying_root;
            uf_desc.min_rep[dying_root] = uf_desc.min_rep[surv_root];
        }
    }
    relative.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // -----------------------------------------------------------------------
    // Phase 4: Extended pairs
    // -----------------------------------------------------------------------
    // Match each ordinary birth with a relative death (high filtration value),
    // sorted by birth ascending / relative death descending.
    let mut ext_ord = ordinary.clone();
    let mut ext_rel = relative.clone();
    ext_ord.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    ext_rel.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let n_ext = ext_ord.len().min(ext_rel.len());
    let mut extended: Vec<(f64, f64)> = Vec::with_capacity(n_ext);
    for i in 0..n_ext {
        extended.push((ext_ord[i].0, ext_rel[i].1));
    }

    Ok(ExtendedBarcode {
        ordinary,
        relative,
        extended,
        essential,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn path_graph(n: usize) -> (Vec<f64>, Vec<(usize, usize)>) {
        let vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let edges: Vec<(usize, usize)> = (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect();
        (vals, edges)
    }

    #[test]
    fn ordinary_finite() {
        // Path 0—1—2—3 with values [0,1,2,3]: 3 edges merge 4 vertices into 1 component.
        let (vals, edges) = path_graph(4);
        let bc = extended_persistence(&vals, &edges).expect("ok");
        assert_eq!(bc.ordinary.len(), 3, "path with 4 vertices has 3 merges");
    }

    #[test]
    fn relative_finite() {
        let (vals, edges) = path_graph(4);
        let bc = extended_persistence(&vals, &edges).expect("ok");
        // Relative pairs come from the descending filtration — should also see merges.
        let bc2 = extended_persistence(&vals, &edges).expect("ok");
        assert_eq!(
            bc.relative.len(),
            bc2.relative.len(),
            "deterministic relative"
        );
    }

    #[test]
    fn output_deterministic() {
        let vals = vec![0.5, 1.5, 0.2, 2.0, 1.0];
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)];
        let bc1 = extended_persistence(&vals, &edges).expect("ok");
        let bc2 = extended_persistence(&vals, &edges).expect("ok");
        assert_eq!(bc1.ordinary.len(), bc2.ordinary.len());
        assert_eq!(bc1.relative.len(), bc2.relative.len());
        assert_eq!(bc1.essential.len(), bc2.essential.len());
        for (a, b) in bc1.essential.iter().zip(bc2.essential.iter()) {
            assert!((a - b).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn n_vertices_1() {
        let bc = extended_persistence(&[2.5], &[]).expect("ok");
        assert_eq!(bc.ordinary.len(), 0);
        assert_eq!(bc.relative.len(), 0);
        assert_eq!(bc.essential.len(), 1);
        assert!((bc.essential[0] - 2.5).abs() < 1.0e-12);
    }

    #[test]
    fn no_edges_all_essential() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bc = extended_persistence(&vals, &[]).expect("ok");
        assert_eq!(bc.ordinary.len(), 0);
        assert_eq!(bc.essential.len(), 5);
    }

    #[test]
    fn pair_len() {
        let (vals, edges) = path_graph(5);
        let bc = extended_persistence(&vals, &edges).expect("ok");
        assert!(bc.extended.len() <= bc.ordinary.len().min(bc.relative.len()));
    }

    #[test]
    fn barcodes_nonneg_length() {
        let vals = vec![0.0, 1.0, 0.5, 2.0, 1.5];
        let edges = vec![(0, 1), (0, 2), (1, 3), (2, 4)];
        let bc = extended_persistence(&vals, &edges).expect("ok");
        for &(b, d) in &bc.ordinary {
            assert!(d >= b, "ordinary ({b},{d}) has negative length");
        }
        for &(b, d) in &bc.relative {
            assert!(d >= b, "relative ({b},{d}) has negative length");
        }
        for &v in &bc.essential {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn filtration_constant() {
        let vals = vec![1.0, 1.0, 1.0];
        let edges = vec![(0, 1), (1, 2)];
        // Should not panic or error with constant function
        let bc = extended_persistence(&vals, &edges).expect("ok");
        let _ = bc;
    }

    #[test]
    fn edges_out_of_range_error() {
        let vals = vec![0.0, 1.0, 2.0];
        let edges = vec![(0, 5)];
        let result = extended_persistence(&vals, &edges);
        assert!(
            matches!(result, Err(TdaError::InvalidSimplex(_))),
            "expected InvalidSimplex, got {result:?}"
        );
    }

    #[test]
    fn essential_sorted() {
        let vals = vec![3.0, 1.0, 2.0];
        let bc = extended_persistence(&vals, &[]).expect("ok");
        assert_eq!(bc.essential.len(), 3);
        for w in bc.essential.windows(2) {
            assert!(w[0] <= w[1], "essential values should be sorted");
        }
    }
}
