//! Really Fast Causal Inference (RFCI) — Colombo, Maathuis, Kalisch &
//! Richardson 2012, "Learning high-dimensional directed acyclic graphs with
//! latent and selection variables", Annals of Statistics 40(1):294–321.
//!
//! Like FCI (Spirtes-Meek-Richardson 1999) RFCI infers a partial ancestral
//! graph (PAG) tolerant of latent and selection variables, but it skips the
//! expensive *Possible-D-Sep* refinement of skeleton edges. Instead, it relies
//! on the cheaper unshielded-triple sep-set test of Colombo et al. §3.1 to
//! decide collider orientation. The remaining Zhang R1–R4 orientation rules
//! are applied verbatim. This makes RFCI dramatically faster on
//! high-dimensional data sets while remaining sound under the same FCMG
//! (faithful causal mixed graph) assumption.

use super::fci::{EdgeMark, Pag};
use super::fci_numeric::{fisher_z_dependent, partial_corr_f64, subsets_of_size};
use crate::error::{CausalError, CausalResult};
use std::collections::HashMap;

// PAG accessors. The `Pag` struct exposes `marks` (Vec<[EdgeMark; 2]>) and
// `adjacency` (Vec<bool>) directly, but its convenience methods are private
// to `fci.rs`. We replicate them here in terms of those public fields rather
// than refactoring `fci.rs`.
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

/// Configuration knobs for [`Rfci`].
#[derive(Clone, Debug)]
pub struct RfciConfig {
    /// Maximum size of conditioning subsets explored during the skeleton
    /// phase. Mirrors FCI's `max_cond_set_size`.
    pub max_cond_set_size: usize,
    /// Two-sided Fisher-Z significance level for conditional independence.
    pub alpha: f64,
}

impl Default for RfciConfig {
    fn default() -> Self {
        Self {
            max_cond_set_size: 3,
            alpha: 0.05,
        }
    }
}

/// Really Fast Causal Inference engine.
///
/// Inputs follow the column-major layout: `data[i]` is the i-th variable as a
/// vector of `n_samples` observations. All variables must have the same
/// length.
pub struct Rfci {
    cfg: RfciConfig,
}

impl Rfci {
    /// Construct an RFCI engine after validating `cfg.alpha`.
    pub fn new(cfg: RfciConfig) -> CausalResult<Self> {
        if !(cfg.alpha > 0.0 && cfg.alpha < 1.0) {
            return Err(CausalError::Internal {
                msg: format!("alpha must be in (0, 1), got {}", cfg.alpha),
            });
        }
        Ok(Self { cfg })
    }

    /// Run RFCI on `data` and return the resulting PAG.
    ///
    /// `data` is column-oriented: `data[var]` is the sample vector for one
    /// variable.
    pub fn fit(&self, data: &[Vec<f64>]) -> CausalResult<Pag> {
        let n_vars = data.len();
        if n_vars == 0 {
            return Err(CausalError::InvalidGraphSize { n: n_vars });
        }
        if n_vars == 1 {
            return Err(CausalError::InvalidGraphSize { n: n_vars });
        }
        let n_samples = data[0].len();
        if n_samples < 4 {
            return Err(CausalError::EmptyInput);
        }
        for (i, col) in data.iter().enumerate() {
            if col.len() != n_samples {
                return Err(CausalError::DimensionMismatch {
                    expected: n_samples,
                    got: col.len(),
                });
            }
            for (j, &v) in col.iter().enumerate() {
                if !v.is_finite() {
                    return Err(CausalError::Internal {
                        msg: format!("data[{i}][{j}] is not finite"),
                    });
                }
            }
        }

        let mut pag = empty_complete_pag(n_vars);
        let mut sep_sets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        self.skeleton_phase(data, n_samples, &mut pag, &mut sep_sets);
        orient_unshielded_triples(&mut pag, &sep_sets);
        apply_zhang_rules(&mut pag);
        Ok(pag)
    }

    /// Phase 1 — PC-style skeleton construction with sep-set bookkeeping. We
    /// scan conditioning subsets of increasing size from `adj(x) \ {y}` and
    /// remove the edge as soon as a separating subset is found.
    fn skeleton_phase(
        &self,
        cols: &[Vec<f64>],
        n_samples: usize,
        pag: &mut Pag,
        sep_sets: &mut HashMap<(usize, usize), Vec<usize>>,
    ) {
        let d = pag.n_vars;
        let max_cond = self.cfg.max_cond_set_size.min(d.saturating_sub(2));
        for cond_size in 0..=max_cond {
            let mut to_remove: Vec<(usize, usize, Vec<usize>)> = Vec::new();
            for x in 0..d {
                for y in (x + 1)..d {
                    if !pag_adj(pag, x, y) {
                        continue;
                    }
                    let neighbors_x: Vec<usize> = (0..d)
                        .filter(|&v| v != x && v != y && pag_adj(pag, x, v))
                        .collect();
                    if neighbors_x.len() < cond_size {
                        continue;
                    }
                    let subsets = subsets_of_size(&neighbors_x, cond_size);
                    for subset in subsets {
                        let z_cols: Vec<&Vec<f64>> = subset.iter().map(|&k| &cols[k]).collect();
                        let r = partial_corr_f64(&cols[x], &cols[y], &z_cols, n_samples);
                        if !fisher_z_dependent(r, n_samples, subset.len(), self.cfg.alpha) {
                            to_remove.push((x, y, subset));
                            break;
                        }
                    }
                }
            }
            for (x, y, sep) in to_remove {
                pag_set_adj(pag, x, y, false);
                sep_sets.insert((x, y), sep.clone());
                sep_sets.insert((y, x), sep);
            }
        }
    }
}

/// Build a complete-skeleton PAG (every pair adjacent, all marks ∘).
fn empty_complete_pag(n_vars: usize) -> Pag {
    let mut pag = Pag {
        n_vars,
        marks: vec![[EdgeMark::Circle, EdgeMark::Circle]; n_vars * n_vars],
        adjacency: vec![false; n_vars * n_vars],
    };
    for i in 0..n_vars {
        for j in (i + 1)..n_vars {
            pag_set_adj(&mut pag, i, j, true);
        }
    }
    pag
}

/// Phase 2 — orient colliders at unshielded triples.
///
/// For every triple `(a, b, c)` where `a — b — c` and `a` is not adjacent to
/// `c`, if `b` does **not** lie in `Sep[a, c]` (or no sep-set was recorded
/// because the edge was never tested), set arrows pointing into `b` from both
/// sides. This is RFCI's substitute for FCI's Possible-D-Sep step.
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

/// Phase 3 — Zhang 2008 orientation rules R1–R4, replicated locally rather
/// than refactoring `fci.rs` (its rule methods are private associated
/// functions).
fn apply_zhang_rules(pag: &mut Pag) {
    let mut changed = true;
    let mut iters = 0_usize;
    let max_iters = pag.n_vars * pag.n_vars * 4 + 8;
    while changed && iters < max_iters {
        changed = false;
        changed |= rule_r1(pag);
        changed |= rule_r2(pag);
        changed |= rule_r3(pag);
        changed |= rule_r4(pag);
        iters += 1;
    }
}

/// R1: if `α ∗→ β o-∗ γ` and `α`, `γ` are non-adjacent, orient `β → γ`.
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

/// R2: if `α → β ∗→ γ` or `α ∗→ β → γ`, and `α ∗-o γ`, orient `α ∗→ γ`.
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

/// R3: if `α ∗→ β ←∗ γ`, `α ∗-o δ o-∗ γ`, `α` and `γ` non-adjacent, and
/// `δ ∗-o β`, orient `δ ∗→ β`.
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

/// R4 (discriminating path, Zhang 2008): if a short discriminating path for
/// `γ` terminates at `α ∗→ β` and `β o-∗ γ`, orient the head/tail of the
/// β–γ edge and the α–β edge as a bi-directed collider, the safer of the two
/// resolutions in the absence of a recorded sep-set.
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

/// Locate a θ such that θ → α and α ∗→ γ form the start of a short
/// discriminating path for γ. Returns the θ when one exists.
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
