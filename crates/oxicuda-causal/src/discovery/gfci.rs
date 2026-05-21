//! GFCI — Greedy Fast Causal Inference (Ogarrio, Spirtes & Ramsey 2016).
//!
//! Ogarrio JM, Spirtes P, Ramsey J. *A Hybrid Causal Search Algorithm for
//! Latent Variable Models.* JMLR Workshop & Conf. Proc. 52: 368–379 (2016)
//! (Proc. of the 8th Int. Conf. on Probabilistic Graphical Models, PGM-2016).
//!
//! GFCI is a *hybrid* score+constraint causal-discovery algorithm that
//! produces a PAG over the three marks `{Tail, Arrow, Circle}` and tolerates
//! latent confounders.
//!
//! # Two-phase design
//!
//! 1. **GES (score phase).** Run forward + backward greedy equivalence search
//!    with the Gaussian BIC. Result is a CPDAG; we convert it to a starting
//!    PAG by mapping every CPDAG edge to a `Circle ∘—∘` PAG edge and every
//!    CPDAG non-edge to "no PAG edge". The CPDAG also seeds the skeleton
//!    so that the FCI orientation step does *not* have to redo a quadratic
//!    independence search over all pairs.
//!
//! 2. **FCI orientation (constraint phase).** For every unshielded triple
//!    `(a, b, c)` in the current PAG (i.e. `a—b`, `b—c`, `a ⊥ c`), recompute
//!    a Fisher-Z separating set `Sep(a, c)`. If `b ∉ Sep(a, c)`, orient
//!    `a ∗→ b ←∗ c`. Apply Zhang 2008 rules R1–R4 iteratively until a
//!    fixed point or `max_orient_passes` is exhausted.
//!
//! The skeleton produced by GES is consistent under FGES assumptions; the FCI
//! refinement layer then injects the *latent-confounder-tolerant* causal
//! semantics of a PAG without paying the full FCI Possible-D-Sep cost.

use super::fci::{EdgeMark, Pag};
use super::fci_numeric::{fisher_z_dependent, partial_corr_f64, subsets_of_size};
use super::ges::Ges;
use crate::error::{CausalError, CausalResult};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PAG helpers — replicated from `rfci.rs` (the `Pag` convenience methods are
// private to `fci.rs`; we use only the public fields here).
// ---------------------------------------------------------------------------

#[inline]
fn pag_adj(pag: &Pag, i: usize, j: usize) -> bool {
    pag.adjacency[i * pag.n_vars + j]
}

#[inline]
fn pag_set_adj(pag: &mut Pag, i: usize, j: usize, value: bool) {
    let n = pag.n_vars;
    pag.adjacency[i * n + j] = value;
    pag.adjacency[j * n + i] = value;
}

#[inline]
fn pag_mark(pag: &Pag, i: usize, j: usize) -> EdgeMark {
    pag.marks[i * pag.n_vars + j][0]
}

#[inline]
fn pag_set_mark(pag: &mut Pag, i: usize, j: usize, mark: EdgeMark) {
    let n = pag.n_vars;
    pag.marks[i * n + j][0] = mark;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Configuration for [`Gfci`].
#[derive(Clone, Debug)]
pub struct GfciConfig {
    /// Multiplicative weight applied to the BIC complexity term. A value of
    /// `1.0` reproduces the classical Schwarz BIC; larger values prefer
    /// sparser graphs. Must be strictly positive.
    pub bic_penalty: f64,
    /// Two-sided Fisher-Z significance level for the conditional-independence
    /// tests run during phase 2. Must lie in `(0, 1)`.
    pub fisher_z_alpha: f64,
    /// Cap on iterations of the R1–R4 orientation loop. A value of `0`
    /// returns the graph immediately after the collider-orientation step,
    /// without any R1–R4 propagation.
    pub max_orient_passes: usize,
}

impl Default for GfciConfig {
    fn default() -> Self {
        Self {
            bic_penalty: 1.0,
            fisher_z_alpha: 0.05,
            max_orient_passes: 100,
        }
    }
}

/// Stateless namespace for the GFCI algorithm.
pub struct Gfci;

impl Gfci {
    /// Run GFCI on `data` and return the resulting PAG.
    ///
    /// `data` is column-oriented: `data[var]` is the sample vector for one
    /// variable. All columns must have the same length.
    pub fn discover(data: &[Vec<f64>], cfg: &GfciConfig) -> CausalResult<Pag> {
        validate_cfg(cfg)?;
        let d = data.len();
        if d == 0 {
            return Err(CausalError::EmptyInput);
        }
        if d == 1 {
            return Ok(Pag {
                n_vars: 1,
                marks: vec![[EdgeMark::Circle, EdgeMark::Circle]; 1],
                adjacency: vec![false; 1],
            });
        }
        let n = data[0].len();
        for col in data.iter() {
            if col.len() != n {
                return Err(CausalError::DimensionMismatch {
                    expected: n,
                    got: col.len(),
                });
            }
            for &v in col.iter() {
                if !v.is_finite() {
                    return Err(CausalError::IncompatibleData);
                }
            }
        }
        if n < 4 {
            return Err(CausalError::EmptyInput);
        }
        // GES requires n >= 4 and d >= 2 — already true.

        // --- Phase 1: GES ----------------------------------------------------
        // GES operates on f32 row-major data. We convert here.
        let mut row_major = vec![0.0_f32; n * d];
        for (j, col) in data.iter().enumerate() {
            for (i, &v) in col.iter().enumerate() {
                row_major[i * d + j] = v as f32;
            }
        }
        let cpdag = run_weighted_ges(&row_major, n, d, cfg.bic_penalty)?;

        // Convert CPDAG -> initial PAG (circle marks).
        let mut pag = empty_pag(d);
        for &(from, to, _directed) in cpdag.iter() {
            pag_set_adj(&mut pag, from, to, true);
        }

        // --- Phase 2a: re-compute sep-sets via Fisher-Z (using current edges).
        let mut sep_sets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        recompute_sep_sets(data, n, &pag, cfg.fisher_z_alpha, &mut sep_sets);

        // --- Phase 2b: orient colliders at unshielded triples.
        orient_unshielded_triples(&mut pag, &sep_sets);

        // --- Phase 2c: Zhang R1-R4 orientation rules.
        apply_zhang_rules(&mut pag, cfg.max_orient_passes);

        Ok(pag)
    }
}

fn validate_cfg(cfg: &GfciConfig) -> CausalResult<()> {
    if !(cfg.bic_penalty.is_finite() && cfg.bic_penalty > 0.0) {
        return Err(CausalError::IncompatibleData);
    }
    if !(cfg.fisher_z_alpha > 0.0 && cfg.fisher_z_alpha < 1.0) {
        return Err(CausalError::IncompatibleData);
    }
    Ok(())
}

/// Build an empty (no edges) PAG with all marks initialised to Circle.
fn empty_pag(n_vars: usize) -> Pag {
    Pag {
        n_vars,
        marks: vec![[EdgeMark::Circle, EdgeMark::Circle]; n_vars * n_vars],
        adjacency: vec![false; n_vars * n_vars],
    }
}

/// Phase 1. Run GES, optionally re-weighting the BIC penalty.
///
/// `Ges::run` uses the classical BIC with `k log(n) / 2`. When
/// `bic_penalty != 1.0` we re-derive the CPDAG by running our own
/// greedy forward/backward search with a re-weighted penalty. For the
/// default `bic_penalty = 1.0` we delegate to the existing `Ges`
/// implementation to avoid duplicating the (well-tested) baseline path.
fn run_weighted_ges(
    data: &[f32],
    n: usize,
    d: usize,
    bic_penalty: f64,
) -> CausalResult<Vec<(usize, usize, bool)>> {
    if (bic_penalty - 1.0).abs() < 1e-12 {
        let ges = Ges::run(data, n, d)?;
        return Ok(ges.cpdag);
    }
    weighted_greedy_search(data, n, d, bic_penalty as f32)
}

/// Forward+backward greedy equivalence search with a configurable BIC
/// penalty. Mirrors the structure of `Ges::run` but reuses the local
/// `bic_score` helper so that the complexity term can be re-weighted.
fn weighted_greedy_search(
    data: &[f32],
    n: usize,
    d: usize,
    bic_penalty: f32,
) -> CausalResult<Vec<(usize, usize, bool)>> {
    if data.is_empty() || n < 4 || d < 2 {
        return Err(CausalError::EmptyInput);
    }
    if data.len() != n * d {
        return Err(CausalError::DimensionMismatch {
            expected: n * d,
            got: data.len(),
        });
    }

    let mut parents: Vec<Vec<usize>> = vec![vec![]; d];
    let mut scores: Vec<f32> = (0..d)
        .map(|i| bic_score(residual_variance(data, i, &[], n, d), n, 0, bic_penalty))
        .collect();

    // forward
    let mut changed = true;
    while changed {
        changed = false;
        let mut best_delta = 0.0_f32;
        let mut best_edge: Option<(usize, usize)> = None;
        for from in 0..d {
            for to in 0..d {
                if from == to || parents[to].contains(&from) {
                    continue;
                }
                if would_create_cycle(&parents, from, to, d) {
                    continue;
                }
                let mut new_parents = parents[to].clone();
                new_parents.push(from);
                let nv = residual_variance(data, to, &new_parents, n, d);
                let ns = bic_score(nv, n, new_parents.len(), bic_penalty);
                let delta = ns - scores[to];
                if delta > best_delta {
                    best_delta = delta;
                    best_edge = Some((from, to));
                }
            }
        }
        if let Some((from, to)) = best_edge {
            parents[to].push(from);
            let nv = residual_variance(data, to, &parents[to], n, d);
            scores[to] = bic_score(nv, n, parents[to].len(), bic_penalty);
            changed = true;
        }
    }

    // backward
    changed = true;
    while changed {
        changed = false;
        let mut best_delta = 0.0_f32;
        let mut best_remove: Option<(usize, usize)> = None;
        for to in 0..d {
            for (idx, &from) in parents[to].iter().enumerate() {
                let mut new_parents = parents[to].clone();
                new_parents.remove(idx);
                let nv = residual_variance(data, to, &new_parents, n, d);
                let ns = bic_score(nv, n, new_parents.len(), bic_penalty);
                let delta = ns - scores[to];
                if delta > best_delta {
                    best_delta = delta;
                    best_remove = Some((from, to));
                }
            }
        }
        if let Some((from, to)) = best_remove {
            parents[to].retain(|&v| v != from);
            let nv = residual_variance(data, to, &parents[to], n, d);
            scores[to] = bic_score(nv, n, parents[to].len(), bic_penalty);
            changed = true;
        }
    }

    let mut cpdag = Vec::new();
    for (to, par) in parents.iter().enumerate() {
        for &from in par {
            cpdag.push((from, to, true));
        }
    }
    Ok(cpdag)
}

fn bic_score(residual_variance: f32, n: usize, n_parents: usize, penalty: f32) -> f32 {
    if residual_variance <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let log_n = (n as f32).ln();
    -0.5 * n as f32 * residual_variance.ln() - 0.5 * penalty * n_parents as f32 * log_n
}

fn residual_variance(data: &[f32], target: usize, parents: &[usize], n: usize, d: usize) -> f32 {
    let y: Vec<f32> = (0..n).map(|i| data[i * d + target]).collect();
    if parents.is_empty() {
        let mean = y.iter().sum::<f32>() / n as f32;
        return y.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    }
    let p = parents.len();
    let mut x_mat = vec![0.0_f32; n * p];
    for (col, &par) in parents.iter().enumerate() {
        for row in 0..n {
            x_mat[row * p + col] = data[row * d + par];
        }
    }
    let mut xtx = vec![0.0_f32; p * p];
    let mut xty = vec![0.0_f32; p];
    for row in 0..n {
        for i in 0..p {
            for j in 0..p {
                xtx[i * p + j] += x_mat[row * p + i] * x_mat[row * p + j];
            }
            xty[i] += x_mat[row * p + i] * y[row];
        }
    }
    for i in 0..p {
        xtx[i * p + i] += 1e-6;
    }
    let inv = match super::notears::gauss_jordan_inv(&xtx, p, 0.0) {
        Ok(m) => m,
        Err(_) => return f32::MAX,
    };
    let beta: Vec<f32> = (0..p)
        .map(|i| (0..p).map(|j| inv[i * p + j] * xty[j]).sum())
        .collect();
    let ss_res: f32 = (0..n)
        .map(|i| {
            let pred: f32 = (0..p).map(|j| x_mat[i * p + j] * beta[j]).sum();
            (y[i] - pred).powi(2)
        })
        .sum();
    ss_res / n as f32
}

fn would_create_cycle(parents: &[Vec<usize>], from: usize, to: usize, d: usize) -> bool {
    let mut visited = vec![false; d];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(to);
    visited[to] = true;
    while let Some(cur) = queue.pop_front() {
        if cur == from {
            return true;
        }
        for &p in &parents[cur] {
            if !visited[p] {
                visited[p] = true;
                queue.push_back(p);
            }
        }
    }
    false
}

/// Phase 2a. Recompute Fisher-Z separating sets for every *non-edge* in the
/// current PAG. We try conditioning subsets of size 0..=2 drawn from the
/// neighbours of `x` in the PAG. For every adjacent pair `(x, y)` we also
/// record a candidate sep-set so that the orientation rules have something
/// to inspect; for non-adjacent pairs we keep the smallest separator we find.
///
/// Sound and complete sep-set learning is delegated to the underlying GES
/// skeleton; this routine only adds the bookkeeping needed by the FCI
/// orientation step.
fn recompute_sep_sets(
    data: &[Vec<f64>],
    n_samples: usize,
    pag: &Pag,
    alpha: f64,
    sep_sets: &mut HashMap<(usize, usize), Vec<usize>>,
) {
    let cols = data.to_vec();
    let d = pag.n_vars;
    let max_cond: usize = 2; // matches FGES default
    for x in 0..d {
        for y in (x + 1)..d {
            if pag_adj(pag, x, y) {
                continue;
            }
            let neighbors: Vec<usize> = (0..d)
                .filter(|&v| v != x && v != y && pag_adj(pag, x, v))
                .collect();
            let cap = max_cond.min(neighbors.len());
            let mut best: Option<Vec<usize>> = None;
            'outer: for k in 0..=cap {
                for subset in subsets_of_size(&neighbors, k) {
                    let z_cols: Vec<&Vec<f64>> = subset.iter().map(|&i| &cols[i]).collect();
                    let r = partial_corr_f64(&cols[x], &cols[y], &z_cols, n_samples);
                    if !fisher_z_dependent(r, n_samples, subset.len(), alpha) {
                        best = Some(subset);
                        break 'outer;
                    }
                }
            }
            // Symmetric neighbour set on y for redundancy.
            if best.is_none() {
                let neighbors_y: Vec<usize> = (0..d)
                    .filter(|&v| v != x && v != y && pag_adj(pag, y, v))
                    .collect();
                let cap_y = max_cond.min(neighbors_y.len());
                'outer_y: for k in 0..=cap_y {
                    for subset in subsets_of_size(&neighbors_y, k) {
                        let z_cols: Vec<&Vec<f64>> = subset.iter().map(|&i| &cols[i]).collect();
                        let r = partial_corr_f64(&cols[x], &cols[y], &z_cols, n_samples);
                        if !fisher_z_dependent(r, n_samples, subset.len(), alpha) {
                            best = Some(subset);
                            break 'outer_y;
                        }
                    }
                }
            }
            let sep = best.unwrap_or_default();
            sep_sets.insert((x, y), sep.clone());
            sep_sets.insert((y, x), sep);
        }
    }
}

/// Phase 2b. For every unshielded triple `(a, b, c)` (a–b, b–c, a ⊥ c) with
/// `b ∉ Sep(a, c)`, orient `a ∗→ b ←∗ c`. Direct port of the corresponding
/// step in `rfci.rs`.
fn orient_unshielded_triples(pag: &mut Pag, sep_sets: &HashMap<(usize, usize), Vec<usize>>) {
    let d = pag.n_vars;
    let mut to_orient: Vec<(usize, usize, usize)> = Vec::new();
    for b in 0..d {
        for a in 0..d {
            if a == b || !pag_adj(pag, a, b) {
                continue;
            }
            for c in (a + 1)..d {
                if c == b || !pag_adj(pag, b, c) {
                    continue;
                }
                if pag_adj(pag, a, c) {
                    continue;
                }
                let sep = sep_sets
                    .get(&(a, c))
                    .cloned()
                    .unwrap_or_else(|| sep_sets.get(&(c, a)).cloned().unwrap_or_default());
                if !sep.contains(&b) {
                    to_orient.push((a, b, c));
                }
            }
        }
    }
    for (a, b, c) in to_orient {
        pag_set_mark(pag, a, b, EdgeMark::Arrow);
        pag_set_mark(pag, c, b, EdgeMark::Arrow);
    }
}

/// Phase 2c. Iterate Zhang R1–R4 until a fixed point or `max_passes`.
fn apply_zhang_rules(pag: &mut Pag, max_passes: usize) {
    let mut changed = true;
    let mut iters = 0_usize;
    while changed && iters < max_passes {
        changed = false;
        changed |= rule_r1(pag);
        changed |= rule_r2(pag);
        changed |= rule_r3(pag);
        changed |= rule_r4(pag);
        iters += 1;
    }
}

// Zhang R1
fn rule_r1(pag: &mut Pag) -> bool {
    let mut changed = false;
    let d = pag.n_vars;
    for alpha in 0..d {
        for beta in 0..d {
            if alpha == beta || !pag_adj(pag, alpha, beta) {
                continue;
            }
            if pag_mark(pag, alpha, beta) != EdgeMark::Arrow {
                continue;
            }
            for gamma in 0..d {
                if gamma == alpha || gamma == beta || !pag_adj(pag, beta, gamma) {
                    continue;
                }
                if pag_adj(pag, alpha, gamma)
                    || pag_mark(pag, gamma, beta) != EdgeMark::Circle
                    || pag_mark(pag, beta, gamma) == EdgeMark::Arrow
                {
                    continue;
                }
                pag_set_mark(pag, gamma, beta, EdgeMark::Tail);
                pag_set_mark(pag, beta, gamma, EdgeMark::Arrow);
                changed = true;
            }
        }
    }
    changed
}

// Zhang R2
fn rule_r2(pag: &mut Pag) -> bool {
    let mut changed = false;
    let d = pag.n_vars;
    for alpha in 0..d {
        for gamma in 0..d {
            if alpha == gamma || !pag_adj(pag, alpha, gamma) {
                continue;
            }
            if pag_mark(pag, alpha, gamma) != EdgeMark::Circle {
                continue;
            }
            for beta in 0..d {
                if beta == alpha || beta == gamma {
                    continue;
                }
                if !pag_adj(pag, alpha, beta) || !pag_adj(pag, beta, gamma) {
                    continue;
                }
                let a_to_b_arrow = pag_mark(pag, alpha, beta) == EdgeMark::Arrow;
                let b_tail_at_a = pag_mark(pag, beta, alpha) == EdgeMark::Tail;
                let b_to_g_arrow = pag_mark(pag, beta, gamma) == EdgeMark::Arrow;
                let g_tail_at_b = pag_mark(pag, gamma, beta) == EdgeMark::Tail;
                let pattern_a = a_to_b_arrow && g_tail_at_b && b_to_g_arrow;
                let pattern_b = b_tail_at_a && a_to_b_arrow && b_to_g_arrow;
                if pattern_a || pattern_b {
                    pag_set_mark(pag, alpha, gamma, EdgeMark::Arrow);
                    changed = true;
                    break;
                }
            }
        }
    }
    changed
}

// Zhang R3
fn rule_r3(pag: &mut Pag) -> bool {
    let mut changed = false;
    let d = pag.n_vars;
    for beta in 0..d {
        for delta in 0..d {
            if beta == delta || !pag_adj(pag, delta, beta) {
                continue;
            }
            if pag_mark(pag, delta, beta) != EdgeMark::Circle {
                continue;
            }
            for alpha in 0..d {
                if alpha == beta || alpha == delta {
                    continue;
                }
                if !pag_adj(pag, alpha, beta) || pag_mark(pag, alpha, beta) != EdgeMark::Arrow {
                    continue;
                }
                if !pag_adj(pag, alpha, delta) || pag_mark(pag, alpha, delta) != EdgeMark::Circle {
                    continue;
                }
                for gamma in 0..d {
                    if gamma == alpha || gamma == beta || gamma == delta {
                        continue;
                    }
                    if !pag_adj(pag, gamma, beta) || pag_mark(pag, gamma, beta) != EdgeMark::Arrow {
                        continue;
                    }
                    if !pag_adj(pag, gamma, delta)
                        || pag_mark(pag, gamma, delta) != EdgeMark::Circle
                    {
                        continue;
                    }
                    if pag_adj(pag, alpha, gamma) {
                        continue;
                    }
                    pag_set_mark(pag, delta, beta, EdgeMark::Arrow);
                    changed = true;
                    break;
                }
            }
        }
    }
    changed
}

// Zhang R4 (discriminating path, short form).
fn rule_r4(pag: &mut Pag) -> bool {
    let mut changed = false;
    let d = pag.n_vars;
    for beta in 0..d {
        for gamma in 0..d {
            if beta == gamma || !pag_adj(pag, beta, gamma) {
                continue;
            }
            if pag_mark(pag, beta, gamma) != EdgeMark::Circle {
                continue;
            }
            for alpha in 0..d {
                if alpha == beta || alpha == gamma || !pag_adj(pag, alpha, beta) {
                    continue;
                }
                if pag_mark(pag, alpha, beta) != EdgeMark::Arrow {
                    continue;
                }
                if !pag_adj(pag, alpha, gamma) {
                    continue;
                }
                if pag_mark(pag, alpha, gamma) != EdgeMark::Arrow
                    || pag_mark(pag, gamma, alpha) != EdgeMark::Tail
                {
                    continue;
                }
                if find_discriminating_origin(pag, alpha, beta, gamma).is_some() {
                    pag_set_mark(pag, beta, gamma, EdgeMark::Arrow);
                    pag_set_mark(pag, gamma, beta, EdgeMark::Arrow);
                    pag_set_mark(pag, alpha, beta, EdgeMark::Arrow);
                    pag_set_mark(pag, beta, alpha, EdgeMark::Arrow);
                    changed = true;
                }
            }
        }
    }
    changed
}

fn find_discriminating_origin(pag: &Pag, alpha: usize, beta: usize, gamma: usize) -> Option<usize> {
    let d = pag.n_vars;
    for theta in 0..d {
        if theta == alpha || theta == beta || theta == gamma {
            continue;
        }
        if pag_adj(pag, theta, gamma) || !pag_adj(pag, theta, alpha) {
            continue;
        }
        if pag_mark(pag, theta, alpha) != EdgeMark::Arrow
            || pag_mark(pag, alpha, gamma) != EdgeMark::Arrow
        {
            continue;
        }
        return Some(theta);
    }
    None
}
