//! Junction-tree (clique-tree) exact inference for discrete pairwise/higher-order
//! Markov Random Fields.
//!
//! Implements the Lauritzen-Spiegelhalter / Hugin algorithm:
//!
//! 1. **Moralise** the factor graph — connect every pair of variables that co-occur
//!    in the scope of some factor (this also marries co-parents).
//! 2. **Triangulate** the moral graph by eliminating variables in a heuristic order
//!    (min-fill, breaking ties by min-degree).  The set of a variable together with
//!    its still-living neighbours at elimination time is a candidate clique.
//! 3. Collect the **maximal cliques** (drop any candidate that is a subset of another).
//! 4. Build a **clique tree** as a maximum-weight spanning tree over the cliques,
//!    where the weight of an edge between two cliques is the size of their separator
//!    (their shared variables).  Maximising separator sizes guarantees the
//!    running-intersection property.
//! 5. **Assign** each input factor to a clique that contains its scope and multiply
//!    it into that clique's potential.
//! 6. **Calibrate** with a Hugin-style two-pass schedule (collect to a root, then
//!    distribute from the root) so that every clique holds the joint marginal over
//!    its variables (up to a global scale equal to the partition function `Z`).
//!
//! All potentials are stored in the **log domain** (`ln`-potentials, row-major over
//! the clique's joint configuration in increasing variable order) so that the
//! message passing never under/overflows.  A configuration value of `-inf`
//! represents a zero-probability entry.

use super::mrf::Mrf;
use crate::error::{SeqError, SeqResult};

/// Configuration for a junction tree: number of variables and their cardinalities.
#[derive(Debug, Clone)]
pub struct JunctionTreeConfig {
    /// Number of discrete random variables.
    pub n_vars: usize,
    /// Cardinality (number of states) of each variable; length must equal `n_vars`.
    pub cardinalities: Vec<usize>,
}

/// A clique of the junction tree.
///
/// * `vars` — the variables in the clique, stored in **strictly increasing** order.
/// * `potential` — the log-potential table over the clique's joint configuration,
///   row-major in `vars` order (the last variable varies fastest).
#[derive(Debug, Clone)]
pub struct Clique {
    /// Variables in the clique (sorted ascending).
    pub vars: Vec<usize>,
    /// Log-potential table over the clique's joint configuration (row-major).
    pub potential: Vec<f64>,
}

/// A separator between two adjacent cliques: the shared variables and a cached
/// log-potential table over them (used by Hugin message passing).
#[derive(Debug, Clone)]
struct Separator {
    /// Index of the first incident clique.
    clique_a: usize,
    /// Index of the second incident clique.
    clique_b: usize,
    /// Shared variables (sorted ascending).
    vars: Vec<usize>,
    /// Current log-potential table over the separator (row-major).
    potential: Vec<f64>,
}

/// A calibrated (or calibratable) junction tree for exact inference.
#[derive(Debug, Clone)]
pub struct JunctionTree {
    cfg: JunctionTreeConfig,
    cliques: Vec<Clique>,
    separators: Vec<Separator>,
    /// Adjacency list over cliques; each entry is `(neighbour_clique, separator_idx)`.
    adjacency: Vec<Vec<(usize, usize)>>,
    /// A root clique and an ordering of cliques such that parents precede children
    /// (breadth-first from the root over the clique tree).  Used for the two passes.
    bfs_order: Vec<usize>,
    /// Parent clique of each clique in the rooted tree (`usize::MAX` for the root).
    parent: Vec<usize>,
    /// Separator index connecting each clique to its parent (`usize::MAX` for root).
    parent_sep: Vec<usize>,
}

/// Compute the number of joint configurations of a set of variables.
fn config_count(vars: &[usize], cards: &[usize]) -> usize {
    let mut n = 1usize;
    for &v in vars {
        n = n.saturating_mul(cards[v]);
    }
    n
}

/// Decode a linear index into the per-variable state assignment (row-major, last
/// variable fastest) for `vars`.  `out` must have length `vars.len()`.
fn decode_index(mut idx: usize, vars: &[usize], cards: &[usize], out: &mut [usize]) {
    for k in (0..vars.len()).rev() {
        let c = cards[vars[k]];
        out[k] = idx % c;
        idx /= c;
    }
}

/// Given a full assignment over `super_vars` (as decoded states aligned to
/// `super_vars`), compute the linear index into a table over the `sub_vars`
/// (a subset of `super_vars`, both sorted ascending).
fn project_index(
    super_vars: &[usize],
    super_states: &[usize],
    sub_vars: &[usize],
    cards: &[usize],
) -> usize {
    let mut idx = 0usize;
    let mut sp = 0usize;
    for &sv in sub_vars {
        // Advance the super pointer to the matching variable (both sorted).
        while super_vars[sp] != sv {
            sp += 1;
        }
        idx = idx * cards[sv] + super_states[sp];
    }
    idx
}

/// Log-sum-exp of a slice, returning `-inf` for an all-`-inf` slice.
fn log_sum_exp(xs: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &x in xs {
        if x > m {
            m = x;
        }
    }
    if m == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let mut s = 0.0;
    for &x in xs {
        s += (x - m).exp();
    }
    m + s.ln()
}

impl JunctionTree {
    /// Build a junction tree from a list of factors.
    ///
    /// Each factor is `(vars, table)` where `vars` is the (unsorted is allowed)
    /// scope and `table` is a **linear-domain** non-negative potential table,
    /// row-major over the factor's joint configuration in the *given* `vars`
    /// order.  Tables are converted to the log domain internally.
    pub fn build(cfg: &JunctionTreeConfig, factors: &[(Vec<usize>, Vec<f64>)]) -> SeqResult<Self> {
        if cfg.n_vars == 0 {
            return Err(SeqError::InvalidConfiguration(
                "n_vars must be >= 1".to_string(),
            ));
        }
        if cfg.cardinalities.len() != cfg.n_vars {
            return Err(SeqError::ShapeMismatch {
                expected: cfg.n_vars,
                got: cfg.cardinalities.len(),
            });
        }
        for &c in &cfg.cardinalities {
            if c == 0 {
                return Err(SeqError::InvalidConfiguration(
                    "every cardinality must be >= 1".to_string(),
                ));
            }
        }
        for (vars, table) in factors {
            for &v in vars {
                if v >= cfg.n_vars {
                    return Err(SeqError::IndexOutOfBounds {
                        index: v,
                        len: cfg.n_vars,
                    });
                }
            }
            let expected = config_count(vars, &cfg.cardinalities);
            if table.len() != expected {
                return Err(SeqError::ShapeMismatch {
                    expected,
                    got: table.len(),
                });
            }
        }

        let cards = &cfg.cardinalities;
        let n = cfg.n_vars;

        // --- Step 1: moral graph adjacency (symmetric boolean matrix). ---
        let mut adj = vec![vec![false; n]; n];
        for (vars, _) in factors {
            for a in 0..vars.len() {
                for b in (a + 1)..vars.len() {
                    let (u, w) = (vars[a], vars[b]);
                    if u != w {
                        adj[u][w] = true;
                        adj[w][u] = true;
                    }
                }
            }
        }

        // --- Step 2+3: triangulation via heuristic elimination -> candidate cliques. ---
        let candidate_cliques = Self::eliminate_for_cliques(&adj, cards);

        // Keep only maximal cliques (drop subsets of another candidate).
        let maximal = Self::keep_maximal(candidate_cliques);

        // --- Step 4: build clique tree (maximum-weight spanning tree on |separator|). ---
        let (adjacency, separators) = Self::build_clique_tree(&maximal, cards);

        // Allocate clique potentials in the log domain, initialised to log(1)=0.
        let mut cliques: Vec<Clique> = maximal
            .into_iter()
            .map(|vars| {
                let len = config_count(&vars, cards);
                Clique {
                    vars,
                    potential: vec![0.0; len],
                }
            })
            .collect();

        // --- Step 5: assign each factor to a containing clique and multiply in. ---
        for (vars, table) in factors {
            let mut sorted = vars.clone();
            sorted.sort_unstable();
            sorted.dedup();
            let target = cliques
                .iter()
                .position(|c| sorted.iter().all(|v| c.vars.contains(v)));
            let target = match target {
                Some(t) => t,
                None => {
                    return Err(SeqError::GraphInvariantViolated(format!(
                        "factor scope {sorted:?} not contained in any clique"
                    )));
                }
            };
            Self::multiply_factor_into_clique(&mut cliques[target], vars, table, cards);
        }

        // Root the tree at clique 0 and compute a BFS order with parent pointers.
        let (bfs_order, parent, parent_sep) = Self::root_tree(cliques.len(), &adjacency);

        Ok(Self {
            cfg: cfg.clone(),
            cliques,
            separators,
            adjacency,
            bfs_order,
            parent,
            parent_sep,
        })
    }

    /// Build a junction tree directly from a pairwise [`Mrf`].  Unary terms become
    /// single-variable factors and each pairwise term a two-variable factor.  The
    /// [`Mrf`] stores **energies** (probability ∝ `exp(-energy)`), so tables are
    /// `exp(-unary)` and `exp(-pairwise)` respectively.
    pub fn from_mrf(mrf: &Mrf) -> SeqResult<Self> {
        let cfg = JunctionTreeConfig {
            n_vars: mrf.n_nodes,
            cardinalities: vec![mrf.n_labels; mrf.n_nodes],
        };
        let nl = mrf.n_labels;
        let l2 = nl * nl;
        let mut factors: Vec<(Vec<usize>, Vec<f64>)> = Vec::new();
        for i in 0..mrf.n_nodes {
            let mut table = vec![0.0; nl];
            for l in 0..nl {
                table[l] = (-mrf.unary[i * nl + l]).exp();
            }
            factors.push((vec![i], table));
        }
        for (e_idx, &(u, v)) in mrf.edges.iter().enumerate() {
            let (lo, hi) = if u < v { (u, v) } else { (v, u) };
            let mut table = vec![0.0; l2];
            // Table is row-major over (lo, hi); the Mrf stores pairwise as (u, v).
            for a in 0..nl {
                for b in 0..nl {
                    // a indexes `lo`, b indexes `hi`.
                    let (lu, lv) = if u == lo { (a, b) } else { (b, a) };
                    table[a * nl + b] = (-mrf.pairwise[e_idx * l2 + lu * nl + lv]).exp();
                }
            }
            factors.push((vec![lo, hi], table));
        }
        Self::build(&cfg, &factors)
    }

    /// Heuristic elimination (min-fill, ties broken by min-degree) returning the
    /// candidate cliques formed at each elimination step.
    fn eliminate_for_cliques(adj: &[Vec<bool>], cards: &[usize]) -> Vec<Vec<usize>> {
        let n = adj.len();
        // Working copy of adjacency we will fill in / remove from.
        let mut work = adj.to_vec();
        let mut alive = vec![true; n];
        let mut cliques: Vec<Vec<usize>> = Vec::new();

        for _ in 0..n {
            // Choose the next variable to eliminate.
            let mut best_var = usize::MAX;
            let mut best_fill = usize::MAX;
            let mut best_deg = usize::MAX;
            for v in 0..n {
                if !alive[v] {
                    continue;
                }
                let neighbours: Vec<usize> = (0..n)
                    .filter(|&u| alive[u] && u != v && work[v][u])
                    .collect();
                let deg = neighbours.len();
                // Count missing edges among neighbours (fill-in count).
                let mut fill = 0usize;
                for a in 0..neighbours.len() {
                    for b in (a + 1)..neighbours.len() {
                        if !work[neighbours[a]][neighbours[b]] {
                            fill += 1;
                        }
                    }
                }
                if fill < best_fill || (fill == best_fill && deg < best_deg) {
                    best_fill = fill;
                    best_deg = deg;
                    best_var = v;
                }
            }
            if best_var == usize::MAX {
                break;
            }

            // Form the candidate clique: best_var + its living neighbours.
            let neighbours: Vec<usize> = (0..n)
                .filter(|&u| alive[u] && u != best_var && work[best_var][u])
                .collect();
            let mut clique = Vec::with_capacity(neighbours.len() + 1);
            clique.push(best_var);
            clique.extend_from_slice(&neighbours);
            clique.sort_unstable();
            cliques.push(clique);

            // Add fill edges to make the neighbourhood a clique.
            for a in 0..neighbours.len() {
                for b in (a + 1)..neighbours.len() {
                    work[neighbours[a]][neighbours[b]] = true;
                    work[neighbours[b]][neighbours[a]] = true;
                }
            }
            // Eliminate the variable.
            alive[best_var] = false;
        }

        // Cardinalities are only needed implicitly; isolated variables (no factor,
        // no neighbour) still produce a singleton clique above, which is correct.
        let _ = cards;
        cliques
    }

    /// Drop any clique that is a subset of another, returning the maximal cliques.
    fn keep_maximal(mut cliques: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        // Sort by descending size so that supersets come first.
        cliques.sort_by_key(|c| std::cmp::Reverse(c.len()));
        let mut maximal: Vec<Vec<usize>> = Vec::new();
        for c in cliques {
            let is_subset = maximal.iter().any(|m| c.iter().all(|v| m.contains(v)));
            if !is_subset {
                maximal.push(c);
            }
        }
        maximal
    }

    /// Build a maximum-weight spanning forest over the cliques (weight = separator
    /// size) using Prim/Kruskal-style greedy selection; returns the adjacency list
    /// and the separators.  Disconnected groups of cliques form a forest, which the
    /// two-pass schedule handles by visiting each tree independently.
    fn build_clique_tree(
        cliques: &[Vec<usize>],
        cards: &[usize],
    ) -> (Vec<Vec<(usize, usize)>>, Vec<Separator>) {
        let m = cliques.len();
        let mut adjacency: Vec<Vec<(usize, usize)>> = vec![Vec::new(); m];
        let mut separators: Vec<Separator> = Vec::new();
        if m <= 1 {
            return (adjacency, separators);
        }

        // Candidate edges with their separator size; Kruskal on descending weight.
        let mut edges: Vec<(usize, usize, usize)> = Vec::new();
        for a in 0..m {
            for b in (a + 1)..m {
                let shared = shared_vars(&cliques[a], &cliques[b]);
                edges.push((shared.len(), a, b));
            }
        }
        edges.sort_by_key(|e| std::cmp::Reverse(e.0));

        // Union-find for cycle detection.
        let mut parent: Vec<usize> = (0..m).collect();
        fn find(parent: &mut [usize], x: usize) -> usize {
            let mut r = x;
            while parent[r] != r {
                r = parent[r];
            }
            // Path compression.
            let mut c = x;
            while parent[c] != r {
                let next = parent[c];
                parent[c] = r;
                c = next;
            }
            r
        }

        for (_w, a, b) in edges {
            let ra = find(&mut parent, a);
            let rb = find(&mut parent, b);
            if ra == rb {
                continue;
            }
            parent[ra] = rb;
            let shared = shared_vars(&cliques[a], &cliques[b]);
            let len = config_count(&shared, cards);
            let sep_idx = separators.len();
            separators.push(Separator {
                clique_a: a,
                clique_b: b,
                vars: shared,
                potential: vec![0.0; len],
            });
            adjacency[a].push((b, sep_idx));
            adjacency[b].push((a, sep_idx));
        }

        (adjacency, separators)
    }

    /// Multiply (add in the log domain) a linear-domain factor table into a clique
    /// potential.  `factor_vars` may be unsorted (matching the factor's table
    /// layout); the clique potential is row-major over the sorted `clique.vars`.
    fn multiply_factor_into_clique(
        clique: &mut Clique,
        factor_vars: &[usize],
        factor_table: &[f64],
        cards: &[usize],
    ) {
        let len = clique.potential.len();
        let mut states = vec![0usize; clique.vars.len()];
        // Pre-resolve each factor variable's position within the (sorted) clique
        // variables; the factor's scope is always a subset of the clique by
        // construction, so every lookup succeeds.
        let positions: Vec<usize> = factor_vars
            .iter()
            .filter_map(|fv| clique.vars.binary_search(fv).ok())
            .collect();
        if positions.len() != factor_vars.len() {
            // Defensive: scope not fully contained — leave the potential unchanged.
            return;
        }
        for idx in 0..len {
            decode_index(idx, &clique.vars, cards, &mut states);
            // Compute the factor's linear index for this clique configuration.
            let mut fidx = 0usize;
            for (k, &fv) in factor_vars.iter().enumerate() {
                fidx = fidx * cards[fv] + states[positions[k]];
            }
            let val = factor_table[fidx];
            clique.potential[idx] += if val > 0.0 {
                val.ln()
            } else {
                f64::NEG_INFINITY
            };
        }
    }

    /// Root the clique forest, returning a BFS visiting order (parents before
    /// children), the parent of each clique, and the separator connecting it to its
    /// parent.  Disconnected components are each rooted at their lowest-index clique.
    fn root_tree(
        m: usize,
        adjacency: &[Vec<(usize, usize)>],
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let mut visited = vec![false; m];
        let mut order = Vec::with_capacity(m);
        let mut parent = vec![usize::MAX; m];
        let mut parent_sep = vec![usize::MAX; m];
        for start in 0..m {
            if visited[start] {
                continue;
            }
            visited[start] = true;
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start);
            while let Some(c) = queue.pop_front() {
                order.push(c);
                for &(nbr, sep) in &adjacency[c] {
                    if !visited[nbr] {
                        visited[nbr] = true;
                        parent[nbr] = c;
                        parent_sep[nbr] = sep;
                        queue.push_back(nbr);
                    }
                }
            }
        }
        (order, parent, parent_sep)
    }

    /// Marginalise a clique's log-potential onto a separator's variables, returning
    /// a fresh log-potential table over the separator (row-major in `sep_vars`).
    fn marginalise_to_separator(&self, clique_idx: usize, sep_vars: &[usize]) -> Vec<f64> {
        let clique = &self.cliques[clique_idx];
        let cards = &self.cfg.cardinalities;
        let sep_len = config_count(sep_vars, cards);
        // Accumulate exp-domain mass per separator config via log-sum-exp.
        let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); sep_len];
        let mut states = vec![0usize; clique.vars.len()];
        for idx in 0..clique.potential.len() {
            decode_index(idx, &clique.vars, cards, &mut states);
            let sidx = project_index(&clique.vars, &states, sep_vars, cards);
            buckets[sidx].push(clique.potential[idx]);
        }
        let mut out = vec![f64::NEG_INFINITY; sep_len];
        for (s, bucket) in buckets.iter().enumerate() {
            out[s] = log_sum_exp(bucket);
        }
        out
    }

    /// Multiply a separator's log-message (`new - old`) into a clique potential,
    /// broadcasting over the clique configurations that share each separator config.
    fn absorb_message_into_clique(&mut self, clique_idx: usize, sep_idx: usize, delta: &[f64]) {
        let sep_vars = self.separators[sep_idx].vars.clone();
        let cards = self.cfg.cardinalities.clone();
        let clique_vars = self.cliques[clique_idx].vars.clone();
        let mut states = vec![0usize; clique_vars.len()];
        let len = self.cliques[clique_idx].potential.len();
        for idx in 0..len {
            decode_index(idx, &clique_vars, &cards, &mut states);
            let sidx = project_index(&clique_vars, &states, &sep_vars, &cards);
            self.cliques[clique_idx].potential[idx] += delta[sidx];
        }
    }

    /// Run Hugin two-pass message passing (collect to root, distribute from root)
    /// so that every clique stores its joint marginal (scaled by `Z`) and every
    /// separator is consistent with both incident cliques.
    pub fn calibrate(&mut self) -> SeqResult<()> {
        if self.cliques.is_empty() {
            return Ok(());
        }

        // Pass 1 — Collect: process cliques in reverse BFS order so that children
        // send messages to their parent before the parent is processed.
        let order = self.bfs_order.clone();
        for &c in order.iter().rev() {
            let p = self.parent[c];
            if p == usize::MAX {
                continue; // root has no parent to send to
            }
            let sep_idx = self.parent_sep[c];
            let sep_vars = self.separators[sep_idx].vars.clone();
            // New separator potential from the child clique.
            let new_sep = self.marginalise_to_separator(c, &sep_vars);
            // delta = new_sep - old_sep (log domain); update parent and separator.
            let old_sep = self.separators[sep_idx].potential.clone();
            let delta: Vec<f64> = new_sep
                .iter()
                .zip(old_sep.iter())
                .map(|(&a, &b)| safe_log_sub(a, b))
                .collect();
            self.absorb_message_into_clique(p, sep_idx, &delta);
            self.separators[sep_idx].potential = new_sep;
        }

        // Pass 2 — Distribute: process cliques in BFS order so that each parent
        // sends to its children after it has been fully updated.
        for &c in order.iter() {
            // For each child of c, send a message c -> child.
            let children: Vec<(usize, usize)> = self.adjacency[c]
                .iter()
                .filter(|&&(nbr, _)| self.parent[nbr] == c)
                .copied()
                .collect();
            for (child, sep_idx) in children {
                let sep_vars = self.separators[sep_idx].vars.clone();
                let new_sep = self.marginalise_to_separator(c, &sep_vars);
                let old_sep = self.separators[sep_idx].potential.clone();
                let delta: Vec<f64> = new_sep
                    .iter()
                    .zip(old_sep.iter())
                    .map(|(&a, &b)| safe_log_sub(a, b))
                    .collect();
                self.absorb_message_into_clique(child, sep_idx, &delta);
                self.separators[sep_idx].potential = new_sep;
            }
        }

        Ok(())
    }

    /// Marginal distribution over a single variable after calibration, normalised
    /// to sum to 1.
    pub fn marginal(&self, var: usize) -> SeqResult<Vec<f64>> {
        if var >= self.cfg.n_vars {
            return Err(SeqError::IndexOutOfBounds {
                index: var,
                len: self.cfg.n_vars,
            });
        }
        let card = self.cfg.cardinalities[var];
        // Find any clique containing the variable.
        let clique_idx = self
            .cliques
            .iter()
            .position(|c| c.vars.contains(&var))
            .ok_or_else(|| {
                SeqError::GraphInvariantViolated(format!(
                    "variable {var} not present in any clique"
                ))
            })?;
        let log_marg = self.marginalise_to_separator(clique_idx, &[var]);
        debug_assert_eq!(log_marg.len(), card);
        // Normalise in the log domain, then exponentiate.
        let logz = log_sum_exp(&log_marg);
        let mut out = vec![0.0; card];
        if logz == f64::NEG_INFINITY {
            // Degenerate (all-zero) potential — fall back to uniform.
            let u = 1.0 / card as f64;
            for v in out.iter_mut() {
                *v = u;
            }
            return Ok(out);
        }
        for l in 0..card {
            out[l] = (log_marg[l] - logz).exp();
        }
        Ok(out)
    }

    /// Joint marginal over the variables of clique `clique_idx`, normalised to sum
    /// to 1.  The output is row-major over the clique's (sorted) variables.
    pub fn clique_marginal(&self, clique_idx: usize) -> SeqResult<Vec<f64>> {
        if clique_idx >= self.cliques.len() {
            return Err(SeqError::IndexOutOfBounds {
                index: clique_idx,
                len: self.cliques.len(),
            });
        }
        let pot = &self.cliques[clique_idx].potential;
        let logz = log_sum_exp(pot);
        let mut out = vec![0.0; pot.len()];
        if logz == f64::NEG_INFINITY {
            let u = 1.0 / pot.len().max(1) as f64;
            for v in out.iter_mut() {
                *v = u;
            }
            return Ok(out);
        }
        for (o, &p) in out.iter_mut().zip(pot.iter()) {
            *o = (p - logz).exp();
        }
        Ok(out)
    }

    /// Log partition function `log Z` (log of the normalisation constant).
    ///
    /// After calibration every clique sums (in the linear domain) to `Z`, so we
    /// take the log-sum-exp of any clique's log-potential.  Before calibration this
    /// is generally **not** `Z`; callers should calibrate first.
    pub fn log_partition(&self) -> SeqResult<f64> {
        if self.cliques.is_empty() {
            return Err(SeqError::GraphInvariantViolated(
                "junction tree has no cliques".to_string(),
            ));
        }
        Ok(log_sum_exp(&self.cliques[0].potential))
    }

    /// Number of cliques in the tree.
    pub fn n_cliques(&self) -> usize {
        self.cliques.len()
    }

    /// Number of separators (tree edges) in the tree.
    pub fn n_separators(&self) -> usize {
        self.separators.len()
    }

    /// Read-only view of the cliques.
    pub fn cliques(&self) -> &[Clique] {
        &self.cliques
    }

    /// Variables of separator `sep_idx` (sorted); useful for tests checking the
    /// running-intersection property.
    pub fn separator_vars(&self, sep_idx: usize) -> SeqResult<&[usize]> {
        if sep_idx >= self.separators.len() {
            return Err(SeqError::IndexOutOfBounds {
                index: sep_idx,
                len: self.separators.len(),
            });
        }
        Ok(&self.separators[sep_idx].vars)
    }

    /// The `(clique_a, clique_b)` incident to separator `sep_idx`.
    pub fn separator_cliques(&self, sep_idx: usize) -> SeqResult<(usize, usize)> {
        if sep_idx >= self.separators.len() {
            return Err(SeqError::IndexOutOfBounds {
                index: sep_idx,
                len: self.separators.len(),
            });
        }
        Ok((
            self.separators[sep_idx].clique_a,
            self.separators[sep_idx].clique_b,
        ))
    }
}

/// Compute the sorted shared variables between two sorted variable lists.
fn shared_vars(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// Numerically-safe `log(exp(a) - exp(b))` for `a >= b`, with `-inf` handling.
fn safe_log_sub(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    // a - b is the log message; in the Hugin update we add (new_sep - old_sep) in
    // the log domain, which is exactly `a - b` (the *ratio* of separator
    // potentials), not `log(exp a - exp b)`.
    a - b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(cards: Vec<usize>) -> JunctionTreeConfig {
        JunctionTreeConfig {
            n_vars: cards.len(),
            cardinalities: cards,
        }
    }

    /// Brute-force marginal of `var` by enumerating the full joint of the factors.
    fn brute_force_marginal(
        cards: &[usize],
        factors: &[(Vec<usize>, Vec<f64>)],
        var: usize,
    ) -> Vec<f64> {
        let n = cards.len();
        let total: usize = cards.iter().product();
        let mut marg = vec![0.0; cards[var]];
        let mut states = vec![0usize; n];
        for joint in 0..total {
            let mut rem = joint;
            for k in (0..n).rev() {
                states[k] = rem % cards[k];
                rem /= cards[k];
            }
            let mut p = 1.0;
            for (vars, table) in factors {
                let mut idx = 0usize;
                for &v in vars {
                    idx = idx * cards[v] + states[v];
                }
                p *= table[idx];
            }
            marg[states[var]] += p;
        }
        let s: f64 = marg.iter().sum();
        if s > 0.0 {
            for m in marg.iter_mut() {
                *m /= s;
            }
        }
        marg
    }

    /// Brute-force log partition function over the full joint.
    fn brute_force_log_z(cards: &[usize], factors: &[(Vec<usize>, Vec<f64>)]) -> f64 {
        let n = cards.len();
        let total: usize = cards.iter().product();
        let mut z = 0.0;
        let mut states = vec![0usize; n];
        for joint in 0..total {
            let mut rem = joint;
            for k in (0..n).rev() {
                states[k] = rem % cards[k];
                rem /= cards[k];
            }
            let mut p = 1.0;
            for (vars, table) in factors {
                let mut idx = 0usize;
                for &v in vars {
                    idx = idx * cards[v] + states[v];
                }
                p *= table[idx];
            }
            z += p;
        }
        z.ln()
    }

    #[test]
    fn single_factor_one_clique() {
        let c = cfg(vec![2, 2]);
        let factors = vec![(vec![0, 1], vec![1.0, 2.0, 3.0, 4.0])];
        let jt = JunctionTree::build(&c, &factors).expect("build");
        assert_eq!(jt.n_cliques(), 1);
        assert_eq!(jt.n_separators(), 0);
    }

    #[test]
    fn single_var_factor_marginal_equals_normalised_potential() {
        let c = cfg(vec![3]);
        let factors = vec![(vec![0], vec![1.0, 2.0, 1.0])];
        let mut jt = JunctionTree::build(&c, &factors).expect("build");
        jt.calibrate().expect("cal");
        let m = jt.marginal(0).expect("marg");
        let expected = [0.25, 0.5, 0.25];
        for (a, b) in m.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn chain_marginals_match_brute_force() {
        // Chain X0 - X1 - X2 with pairwise factors.
        let c = cfg(vec![2, 2, 2]);
        let f01 = (vec![0, 1], vec![1.0, 0.3, 0.4, 2.0]);
        let f12 = (vec![1, 2], vec![1.5, 0.6, 0.2, 1.1]);
        let f0 = (vec![0], vec![0.7, 1.3]);
        let factors = vec![f0, f01, f12];
        let mut jt = JunctionTree::build(&c, &factors).expect("build");
        jt.calibrate().expect("cal");
        for var in 0..3 {
            let m = jt.marginal(var).expect("marg");
            let bf = brute_force_marginal(&c.cardinalities, &factors, var);
            for (a, b) in m.iter().zip(bf.iter()) {
                assert!((a - b).abs() < 1e-6, "var {var}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn chain_marginal_sums_to_one() {
        let c = cfg(vec![3, 2, 3]);
        let f01 = (vec![0, 1], vec![1.0, 0.3, 0.4, 2.0, 0.5, 1.2]);
        let f12 = (vec![1, 2], vec![1.5, 0.6, 0.2, 1.1, 0.9, 0.7]);
        let factors = vec![f01, f12];
        let mut jt = JunctionTree::build(&c, &factors).expect("build");
        jt.calibrate().expect("cal");
        for var in 0..3 {
            let m = jt.marginal(var).expect("marg");
            let s: f64 = m.iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "var {var} sum {s}");
        }
    }

    #[test]
    fn log_partition_matches_brute_force() {
        let c = cfg(vec![2, 3, 2]);
        let f01 = (vec![0, 1], vec![1.0, 0.3, 0.4, 2.0, 0.8, 0.5]);
        let f12 = (vec![1, 2], vec![1.5, 0.6, 0.2, 1.1, 0.9, 0.7]);
        let f2 = (vec![2], vec![1.2, 0.8]);
        let factors = vec![f01, f12, f2];
        let mut jt = JunctionTree::build(&c, &factors).expect("build");
        jt.calibrate().expect("cal");
        let lz = jt.log_partition().expect("logz");
        let bf = brute_force_log_z(&c.cardinalities, &factors);
        assert!((lz - bf).abs() < 1e-6, "logZ {lz} vs {bf}");
    }

    #[test]
    fn independent_variables_product_marginals() {
        // Two independent single-variable factors -> marginals are the normalised
        // single potentials, independent of each other.
        let c = cfg(vec![2, 3]);
        let f0 = (vec![0], vec![1.0, 3.0]);
        let f1 = (vec![1], vec![2.0, 2.0, 4.0]);
        let factors = vec![f0, f1];
        let mut jt = JunctionTree::build(&c, &factors).expect("build");
        jt.calibrate().expect("cal");
        let m0 = jt.marginal(0).expect("m0");
        let m1 = jt.marginal(1).expect("m1");
        assert!((m0[0] - 0.25).abs() < 1e-12);
        assert!((m0[1] - 0.75).abs() < 1e-12);
        assert!((m1[0] - 0.25).abs() < 1e-12);
        assert!((m1[1] - 0.25).abs() < 1e-12);
        assert!((m1[2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn disconnected_factors_handled() {
        // Factor over {0,1} and a separate factor over {2,3}: forest of two trees.
        let c = cfg(vec![2, 2, 2, 2]);
        let fa = (vec![0, 1], vec![1.0, 0.5, 0.5, 1.0]);
        let fb = (vec![2, 3], vec![2.0, 0.1, 0.1, 2.0]);
        let factors = vec![fa, fb];
        let mut jt = JunctionTree::build(&c, &factors).expect("build");
        jt.calibrate().expect("cal");
        for var in 0..4 {
            let m = jt.marginal(var).expect("marg");
            let bf = brute_force_marginal(&c.cardinalities, &factors, var);
            for (a, b) in m.iter().zip(bf.iter()) {
                assert!((a - b).abs() < 1e-6, "var {var}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn calibrate_is_idempotent() {
        let c = cfg(vec![2, 2, 2]);
        let f01 = (vec![0, 1], vec![1.0, 0.3, 0.4, 2.0]);
        let f12 = (vec![1, 2], vec![1.5, 0.6, 0.2, 1.1]);
        let factors = vec![f01, f12];
        let mut jt = JunctionTree::build(&c, &factors).expect("build");
        jt.calibrate().expect("cal1");
        let m_before: Vec<Vec<f64>> = (0..3).map(|v| jt.marginal(v).expect("m")).collect();
        jt.calibrate().expect("cal2");
        let m_after: Vec<Vec<f64>> = (0..3).map(|v| jt.marginal(v).expect("m")).collect();
        for (a, b) in m_before.iter().zip(m_after.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-9, "{x} vs {y}");
            }
        }
    }

    #[test]
    fn running_intersection_on_chain() {
        // Build a chain that induces cliques {0,1},{1,2},{2,3}; the variable 2
        // shared by cliques {1,2} and {2,3} must appear in their separator.
        let c = cfg(vec![2, 2, 2, 2]);
        let f01 = (vec![0, 1], vec![1.0, 0.5, 0.5, 1.0]);
        let f12 = (vec![1, 2], vec![1.0, 0.5, 0.5, 1.0]);
        let f23 = (vec![2, 3], vec![1.0, 0.5, 0.5, 1.0]);
        let factors = vec![f01, f12, f23];
        let jt = JunctionTree::build(&c, &factors).expect("build");
        // Every separator must be the intersection of its two incident cliques.
        for s in 0..jt.n_separators() {
            let (a, b) = jt.separator_cliques(s).expect("sep");
            let inter = shared_vars(&jt.cliques()[a].vars, &jt.cliques()[b].vars);
            assert_eq!(jt.separator_vars(s).expect("vars"), inter.as_slice());
            assert!(
                !inter.is_empty(),
                "separator should be non-empty on a chain"
            );
        }
    }

    #[test]
    fn n_cliques_sane_for_chain() {
        let c = cfg(vec![2, 2, 2, 2]);
        let f01 = (vec![0, 1], vec![1.0, 0.5, 0.5, 1.0]);
        let f12 = (vec![1, 2], vec![1.0, 0.5, 0.5, 1.0]);
        let f23 = (vec![2, 3], vec![1.0, 0.5, 0.5, 1.0]);
        let jt = JunctionTree::build(&c, &[f01, f12, f23]).expect("build");
        // A 4-node chain has exactly 3 maximal cliques of size 2.
        assert_eq!(jt.n_cliques(), 3);
        for cl in jt.cliques() {
            assert_eq!(cl.vars.len(), 2);
        }
    }

    #[test]
    fn ternary_cardinalities_match_brute_force() {
        let c = cfg(vec![3, 3]);
        let f = (
            vec![0, 1],
            vec![1.0, 0.2, 0.5, 0.3, 2.0, 0.4, 0.6, 0.1, 1.5],
        );
        let factors = vec![f];
        let mut jt = JunctionTree::build(&c, &factors).expect("build");
        jt.calibrate().expect("cal");
        for var in 0..2 {
            let m = jt.marginal(var).expect("marg");
            let bf = brute_force_marginal(&c.cardinalities, &factors, var);
            for (a, b) in m.iter().zip(bf.iter()) {
                assert!((a - b).abs() < 1e-9, "var {var}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn triangle_three_var_factor_match_brute_force() {
        // A loop 0-1-2-0 induces one triangle clique {0,1,2}.
        let c = cfg(vec![2, 2, 2]);
        let f01 = (vec![0, 1], vec![1.0, 0.5, 0.5, 1.0]);
        let f12 = (vec![1, 2], vec![1.2, 0.3, 0.4, 0.9]);
        let f02 = (vec![0, 2], vec![0.7, 1.1, 1.3, 0.6]);
        let factors = vec![f01, f12, f02];
        let mut jt = JunctionTree::build(&c, &factors).expect("build");
        jt.calibrate().expect("cal");
        for var in 0..3 {
            let m = jt.marginal(var).expect("marg");
            let bf = brute_force_marginal(&c.cardinalities, &factors, var);
            for (a, b) in m.iter().zip(bf.iter()) {
                assert!((a - b).abs() < 1e-6, "var {var}: {a} vs {b}");
            }
        }
        // The triangle collapses into a single 3-variable clique.
        assert_eq!(jt.n_cliques(), 1);
        assert_eq!(jt.cliques()[0].vars, vec![0, 1, 2]);
    }

    #[test]
    fn from_mrf_matches_direct_factors() {
        // Build an Mrf and check its junction-tree marginals against brute force on
        // the equivalent exp(-energy) factors.
        let m = Mrf::new(
            3,
            2,
            vec![(0, 1), (1, 2)],
            vec![0.1, 0.5, 0.2, 0.3, 0.0, 0.4],
            vec![0.0, 0.7, 0.7, 0.0, 0.0, 0.5, 0.5, 0.0],
        )
        .expect("mrf");
        let mut jt = JunctionTree::from_mrf(&m).expect("jt");
        jt.calibrate().expect("cal");
        // Equivalent factors.
        let nl = 2;
        let mut factors: Vec<(Vec<usize>, Vec<f64>)> = Vec::new();
        for i in 0..3 {
            let mut t = vec![0.0; nl];
            for l in 0..nl {
                t[l] = (-m.unary[i * nl + l]).exp();
            }
            factors.push((vec![i], t));
        }
        for (e, &(u, v)) in m.edges.iter().enumerate() {
            let mut t = vec![0.0; nl * nl];
            for a in 0..nl {
                for b in 0..nl {
                    t[a * nl + b] = (-m.pairwise[e * nl * nl + a * nl + b]).exp();
                }
            }
            factors.push((vec![u, v], t));
        }
        for var in 0..3 {
            let mm = jt.marginal(var).expect("marg");
            let bf = brute_force_marginal(&[nl; 3], &factors, var);
            for (a, b) in mm.iter().zip(bf.iter()) {
                assert!((a - b).abs() < 1e-6, "var {var}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn deterministic_build_and_calibrate() {
        let c = cfg(vec![2, 2, 2]);
        let f01 = (vec![0, 1], vec![1.0, 0.3, 0.4, 2.0]);
        let f12 = (vec![1, 2], vec![1.5, 0.6, 0.2, 1.1]);
        let factors = vec![f01, f12];
        let mut a = JunctionTree::build(&c, &factors).expect("a");
        let mut b = JunctionTree::build(&c, &factors).expect("b");
        a.calibrate().expect("ca");
        b.calibrate().expect("cb");
        for var in 0..3 {
            let ma = a.marginal(var).expect("ma");
            let mb = b.marginal(var).expect("mb");
            assert_eq!(ma, mb);
        }
    }

    #[test]
    fn err_cardinality_mismatch_with_factor_table() {
        let c = cfg(vec![2, 2]);
        // Factor over {0,1} should have a 4-entry table, give 3.
        let factors = vec![(vec![0, 1], vec![1.0, 2.0, 3.0])];
        let r = JunctionTree::build(&c, &factors);
        assert!(matches!(r, Err(SeqError::ShapeMismatch { .. })));
    }

    #[test]
    fn err_var_out_of_range_in_factor() {
        let c = cfg(vec![2, 2]);
        let factors = vec![(vec![0, 5], vec![1.0, 2.0, 3.0, 4.0])];
        let r = JunctionTree::build(&c, &factors);
        assert!(matches!(r, Err(SeqError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn err_empty_cardinalities_mismatch() {
        let c = JunctionTreeConfig {
            n_vars: 2,
            cardinalities: vec![2],
        };
        let r = JunctionTree::build(&c, &[]);
        assert!(matches!(r, Err(SeqError::ShapeMismatch { .. })));
    }

    #[test]
    fn err_n_vars_zero() {
        let c = JunctionTreeConfig {
            n_vars: 0,
            cardinalities: vec![],
        };
        let r = JunctionTree::build(&c, &[]);
        assert!(matches!(r, Err(SeqError::InvalidConfiguration(_))));
    }

    #[test]
    fn err_zero_cardinality() {
        let c = JunctionTreeConfig {
            n_vars: 2,
            cardinalities: vec![2, 0],
        };
        let r = JunctionTree::build(&c, &[]);
        assert!(matches!(r, Err(SeqError::InvalidConfiguration(_))));
    }

    #[test]
    fn err_marginal_var_out_of_range() {
        let c = cfg(vec![2, 2]);
        let factors = vec![(vec![0, 1], vec![1.0, 1.0, 1.0, 1.0])];
        let jt = JunctionTree::build(&c, &factors).expect("build");
        let r = jt.marginal(5);
        assert!(matches!(r, Err(SeqError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn binary_vs_ternary_isolated_factors() {
        // One binary variable, one ternary variable, single factor each.
        let c = cfg(vec![2, 3]);
        let f0 = (vec![0], vec![3.0, 1.0]);
        let f1 = (vec![1], vec![1.0, 1.0, 2.0]);
        let mut jt = JunctionTree::build(&c, &[f0, f1]).expect("build");
        jt.calibrate().expect("cal");
        let m0 = jt.marginal(0).expect("m0");
        let m1 = jt.marginal(1).expect("m1");
        assert_eq!(m0.len(), 2);
        assert_eq!(m1.len(), 3);
        assert!((m0[0] - 0.75).abs() < 1e-12);
        assert!((m1[2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn clique_marginal_normalises() {
        let c = cfg(vec![2, 2]);
        let factors = vec![(vec![0, 1], vec![1.0, 0.3, 0.4, 2.0])];
        let mut jt = JunctionTree::build(&c, &factors).expect("build");
        jt.calibrate().expect("cal");
        let cm = jt.clique_marginal(0).expect("cm");
        let s: f64 = cm.iter().sum();
        assert!((s - 1.0).abs() < 1e-12, "sum {s}");
    }

    #[test]
    fn no_factors_uniform_marginals() {
        // No factors at all: every clique is a singleton with uniform potential.
        let c = cfg(vec![2, 3]);
        let mut jt = JunctionTree::build(&c, &[]).expect("build");
        jt.calibrate().expect("cal");
        let m0 = jt.marginal(0).expect("m0");
        let m1 = jt.marginal(1).expect("m1");
        for v in &m0 {
            assert!((v - 0.5).abs() < 1e-12);
        }
        for v in &m1 {
            assert!((v - 1.0 / 3.0).abs() < 1e-12);
        }
    }
}
