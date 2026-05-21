//! Fast Causal Inference (FCI) — Spirtes, Meek & Richardson 1999.
//!
//! Constraint-based causal discovery extending the PC algorithm so that
//! latent confounders can be tolerated. The output is a *partial ancestral
//! graph* (PAG) over the three edge marks {Tail, Arrow, Circle}.
//!
//! Orientation rules R1–R4 follow Zhang 2008.

use super::fci_numeric::{extract_columns, fisher_z_dependent, partial_corr_f64, subsets_of_size};
use crate::error::{CausalError, CausalResult};
use std::collections::{BTreeSet, HashMap};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EdgeMark {
    Tail,
    Arrow,
    Circle,
}

/// Partial ancestral graph as produced by FCI.
///
/// `marks[i * n_vars + j]` is the mark *at endpoint `j`* of the edge between
/// `i` and `j`. The mark at endpoint `i` of the same edge lives in
/// `marks[j * n_vars + i]`. `adjacency` is symmetric.
#[derive(Clone, Debug)]
pub struct Pag {
    pub n_vars: usize,
    pub marks: Vec<[EdgeMark; 2]>,
    pub adjacency: Vec<bool>,
}

#[derive(Clone, Debug)]
pub struct FciConfig {
    pub alpha: f64,
    pub max_cond_set_size: usize,
}

impl Default for FciConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            max_cond_set_size: 3,
        }
    }
}

pub struct Fci {
    cfg: FciConfig,
}

impl Pag {
    fn empty(n_vars: usize) -> Self {
        Self {
            n_vars,
            marks: vec![[EdgeMark::Circle, EdgeMark::Circle]; n_vars * n_vars],
            adjacency: vec![false; n_vars * n_vars],
        }
    }

    fn adj(&self, i: usize, j: usize) -> bool {
        self.adjacency[i * self.n_vars + j]
    }

    fn set_adj(&mut self, i: usize, j: usize, v: bool) {
        let n = self.n_vars;
        self.adjacency[i * n + j] = v;
        self.adjacency[j * n + i] = v;
    }

    fn mark(&self, i: usize, j: usize) -> EdgeMark {
        self.marks[i * self.n_vars + j][0]
    }

    fn set_mark(&mut self, i: usize, j: usize, m: EdgeMark) {
        let n = self.n_vars;
        self.marks[i * n + j][0] = m;
    }

    fn neighbors(&self, i: usize) -> Vec<usize> {
        (0..self.n_vars)
            .filter(|&j| j != i && self.adj(i, j))
            .collect()
    }

    #[cfg(test)]
    pub(super) fn empty_pub(n_vars: usize) -> Self {
        Self::empty(n_vars)
    }
    #[cfg(test)]
    pub(super) fn adj_pub(&self, i: usize, j: usize) -> bool {
        self.adj(i, j)
    }
    #[cfg(test)]
    pub(super) fn set_adj_pub(&mut self, i: usize, j: usize, v: bool) {
        self.set_adj(i, j, v);
    }
    #[cfg(test)]
    pub(super) fn mark_pub(&self, i: usize, j: usize) -> EdgeMark {
        self.mark(i, j)
    }
    #[cfg(test)]
    pub(super) fn set_mark_pub(&mut self, i: usize, j: usize, m: EdgeMark) {
        self.set_mark(i, j, m);
    }
}

impl Fci {
    pub fn new(cfg: FciConfig) -> CausalResult<Self> {
        if !(cfg.alpha > 0.0 && cfg.alpha < 1.0) {
            return Err(CausalError::Internal {
                msg: format!("alpha must be in (0, 1), got {}", cfg.alpha),
            });
        }
        Ok(Self { cfg })
    }

    pub fn fit(&self, data: &[f64], n_samples: usize, n_vars: usize) -> CausalResult<Pag> {
        if n_vars == 0 {
            return Err(CausalError::InvalidGraphSize { n: n_vars });
        }
        if n_vars == 1 {
            return Ok(Pag::empty(1));
        }
        if n_samples < 4 {
            return Err(CausalError::EmptyInput);
        }
        if data.len() != n_samples * n_vars {
            return Err(CausalError::DimensionMismatch {
                expected: n_samples * n_vars,
                got: data.len(),
            });
        }

        let cols = extract_columns(data, n_samples, n_vars);
        let mut pag = Pag::empty(n_vars);
        for i in 0..n_vars {
            for j in (i + 1)..n_vars {
                pag.set_adj(i, j, true);
            }
        }

        let mut sep_sets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        self.skeleton_phase(&cols, n_samples, &mut pag, &mut sep_sets);
        self.possible_dsep_phase(&cols, n_samples, &mut pag, &mut sep_sets);
        Self::orient_collider_marks(&mut pag, &sep_sets);
        self.apply_orientation_rules(&mut pag);

        Ok(pag)
    }

    fn skeleton_phase(
        &self,
        cols: &[Vec<f64>],
        n: usize,
        pag: &mut Pag,
        sep_sets: &mut HashMap<(usize, usize), Vec<usize>>,
    ) {
        let d = pag.n_vars;
        let max_cond = self.cfg.max_cond_set_size.min(d.saturating_sub(2));
        for cond_size in 0..=max_cond {
            let mut to_remove: Vec<(usize, usize, Vec<usize>)> = Vec::new();
            for x in 0..d {
                for y in (x + 1)..d {
                    if !pag.adj(x, y) {
                        continue;
                    }
                    let neighbors_x: Vec<usize> = (0..d)
                        .filter(|&v| v != x && v != y && pag.adj(x, v))
                        .collect();
                    if neighbors_x.len() < cond_size {
                        continue;
                    }
                    let subsets = subsets_of_size(&neighbors_x, cond_size);
                    for subset in subsets {
                        let z_cols: Vec<&Vec<f64>> = subset.iter().map(|&k| &cols[k]).collect();
                        let r = partial_corr_f64(&cols[x], &cols[y], &z_cols, n);
                        if !fisher_z_dependent(r, n, subset.len(), self.cfg.alpha) {
                            to_remove.push((x, y, subset));
                            break;
                        }
                    }
                }
            }
            for (x, y, sep) in to_remove {
                pag.set_adj(x, y, false);
                sep_sets.insert((x, y), sep.clone());
                sep_sets.insert((y, x), sep);
            }
        }
    }

    /// FCI's distinctive step: re-check edges by conditioning on
    /// Possible-D-Sep(x, y) subsets. Per Spirtes-Meek-Richardson 1999, this
    /// catches edges that purely adjacency-based PC retains in the presence
    /// of latent confounders.
    fn possible_dsep_phase(
        &self,
        cols: &[Vec<f64>],
        n: usize,
        pag: &mut Pag,
        sep_sets: &mut HashMap<(usize, usize), Vec<usize>>,
    ) {
        let d = pag.n_vars;
        let snapshot = sep_sets.clone();
        let mut tentative = pag.clone();
        Self::orient_collider_marks(&mut tentative, &snapshot);

        let mut to_remove: Vec<(usize, usize, Vec<usize>)> = Vec::new();
        for x in 0..d {
            for y in (x + 1)..d {
                if !pag.adj(x, y) {
                    continue;
                }
                let pds = possible_d_sep(&tentative, x, y);
                let candidates: Vec<usize> =
                    pds.into_iter().filter(|&v| v != x && v != y).collect();
                if candidates.is_empty() {
                    continue;
                }
                let max_k = self.cfg.max_cond_set_size.min(candidates.len());
                let mut removed = false;
                for k in 0..=max_k {
                    for subset in subsets_of_size(&candidates, k) {
                        let z_cols: Vec<&Vec<f64>> = subset.iter().map(|&i| &cols[i]).collect();
                        let r = partial_corr_f64(&cols[x], &cols[y], &z_cols, n);
                        if !fisher_z_dependent(r, n, subset.len(), self.cfg.alpha) {
                            to_remove.push((x, y, subset));
                            removed = true;
                            break;
                        }
                    }
                    if removed {
                        break;
                    }
                }
            }
        }
        for (x, y, sep) in to_remove {
            pag.set_adj(x, y, false);
            sep_sets.insert((x, y), sep.clone());
            sep_sets.insert((y, x), sep);
            pag.set_mark(x, y, EdgeMark::Circle);
            pag.set_mark(y, x, EdgeMark::Circle);
        }
    }

    fn orient_collider_marks(pag: &mut Pag, sep_sets: &HashMap<(usize, usize), Vec<usize>>) {
        let d = pag.n_vars;
        for i in 0..d {
            for j in 0..d {
                if i == j || !pag.adj(i, j) {
                    continue;
                }
                for k in 0..d {
                    if k == i || k == j {
                        continue;
                    }
                    if !pag.adj(j, k) || pag.adj(i, k) || i >= k {
                        continue;
                    }
                    let sep = sep_sets.get(&(i, k)).cloned().unwrap_or_default();
                    if !sep.contains(&j) {
                        pag.set_mark(i, j, EdgeMark::Arrow);
                        pag.set_mark(k, j, EdgeMark::Arrow);
                    }
                }
            }
        }
    }

    fn apply_orientation_rules(&self, pag: &mut Pag) {
        let mut changed = true;
        let mut iters = 0_usize;
        let max_iters = pag.n_vars * pag.n_vars * 4 + 8;
        while changed && iters < max_iters {
            changed = false;
            changed |= Self::rule_r1(pag);
            changed |= Self::rule_r2(pag);
            changed |= Self::rule_r3(pag);
            changed |= Self::rule_r4(pag);
            iters += 1;
        }
    }

    /// R1: if alpha *-> beta o-* gamma and alpha and gamma are non-adjacent,
    /// orient beta -> gamma (tail at beta, arrow at gamma).
    fn rule_r1(pag: &mut Pag) -> bool {
        let mut changed = false;
        let d = pag.n_vars;
        for alpha in 0..d {
            for beta in 0..d {
                if alpha == beta || !pag.adj(alpha, beta) {
                    continue;
                }
                if pag.mark(alpha, beta) != EdgeMark::Arrow {
                    continue;
                }
                for gamma in 0..d {
                    if gamma == alpha || gamma == beta || !pag.adj(beta, gamma) {
                        continue;
                    }
                    if pag.adj(alpha, gamma)
                        || pag.mark(gamma, beta) != EdgeMark::Circle
                        || pag.mark(beta, gamma) == EdgeMark::Arrow
                    {
                        continue;
                    }
                    pag.set_mark(gamma, beta, EdgeMark::Tail);
                    pag.set_mark(beta, gamma, EdgeMark::Arrow);
                    changed = true;
                }
            }
        }
        changed
    }

    /// R2: if (alpha -> beta *-> gamma or alpha *-> beta -> gamma) and
    /// alpha *-o gamma, orient alpha *-> gamma.
    fn rule_r2(pag: &mut Pag) -> bool {
        let mut changed = false;
        let d = pag.n_vars;
        for alpha in 0..d {
            for gamma in 0..d {
                if alpha == gamma || !pag.adj(alpha, gamma) {
                    continue;
                }
                if pag.mark(alpha, gamma) != EdgeMark::Circle {
                    continue;
                }
                for beta in 0..d {
                    if beta == alpha || beta == gamma {
                        continue;
                    }
                    if !pag.adj(alpha, beta) || !pag.adj(beta, gamma) {
                        continue;
                    }
                    let a_to_b_arrow = pag.mark(alpha, beta) == EdgeMark::Arrow;
                    let b_tail_at_a = pag.mark(beta, alpha) == EdgeMark::Tail;
                    let b_to_g_arrow = pag.mark(beta, gamma) == EdgeMark::Arrow;
                    let g_tail_at_b = pag.mark(gamma, beta) == EdgeMark::Tail;
                    let pattern_a = a_to_b_arrow && g_tail_at_b && b_to_g_arrow;
                    let pattern_b = b_tail_at_a && a_to_b_arrow && b_to_g_arrow;
                    if pattern_a || pattern_b {
                        pag.set_mark(alpha, gamma, EdgeMark::Arrow);
                        changed = true;
                        break;
                    }
                }
            }
        }
        changed
    }

    /// R3: if alpha *-> beta <-* gamma, alpha *-o delta o-* gamma,
    /// alpha and gamma non-adjacent, delta *-o beta, orient delta *-> beta.
    fn rule_r3(pag: &mut Pag) -> bool {
        let mut changed = false;
        let d = pag.n_vars;
        for beta in 0..d {
            for delta in 0..d {
                if beta == delta || !pag.adj(delta, beta) {
                    continue;
                }
                if pag.mark(delta, beta) != EdgeMark::Circle {
                    continue;
                }
                for alpha in 0..d {
                    if alpha == beta || alpha == delta {
                        continue;
                    }
                    if !pag.adj(alpha, beta) || pag.mark(alpha, beta) != EdgeMark::Arrow {
                        continue;
                    }
                    if !pag.adj(alpha, delta) || pag.mark(alpha, delta) != EdgeMark::Circle {
                        continue;
                    }
                    for gamma in 0..d {
                        if gamma == alpha || gamma == beta || gamma == delta {
                            continue;
                        }
                        if !pag.adj(gamma, beta) || pag.mark(gamma, beta) != EdgeMark::Arrow {
                            continue;
                        }
                        if !pag.adj(gamma, delta) || pag.mark(gamma, delta) != EdgeMark::Circle {
                            continue;
                        }
                        if pag.adj(alpha, gamma) {
                            continue;
                        }
                        pag.set_mark(delta, beta, EdgeMark::Arrow);
                        changed = true;
                        break;
                    }
                }
            }
        }
        changed
    }

    /// R4 (discriminating path, Zhang 2008): when a short discriminating path
    /// exists for `gamma` ending at `alpha *-> beta` with `beta o-* gamma`,
    /// orient `alpha <-> beta <-> gamma` (the safer of the two options
    /// without a recorded sep-set).
    fn rule_r4(pag: &mut Pag) -> bool {
        let mut changed = false;
        let d = pag.n_vars;
        for beta in 0..d {
            for gamma in 0..d {
                if beta == gamma || !pag.adj(beta, gamma) {
                    continue;
                }
                if pag.mark(beta, gamma) != EdgeMark::Circle {
                    continue;
                }
                for alpha in 0..d {
                    if alpha == beta || alpha == gamma || !pag.adj(alpha, beta) {
                        continue;
                    }
                    if pag.mark(alpha, beta) != EdgeMark::Arrow {
                        continue;
                    }
                    if !pag.adj(alpha, gamma) {
                        continue;
                    }
                    if pag.mark(alpha, gamma) != EdgeMark::Arrow
                        || pag.mark(gamma, alpha) != EdgeMark::Tail
                    {
                        continue;
                    }
                    if find_discriminating_origin(pag, alpha, beta, gamma).is_some() {
                        pag.set_mark(beta, gamma, EdgeMark::Arrow);
                        pag.set_mark(gamma, beta, EdgeMark::Arrow);
                        pag.set_mark(alpha, beta, EdgeMark::Arrow);
                        pag.set_mark(beta, alpha, EdgeMark::Arrow);
                        changed = true;
                    }
                }
            }
        }
        changed
    }

    #[cfg(test)]
    pub(super) fn rule_r1_pub(pag: &mut Pag) -> bool {
        Self::rule_r1(pag)
    }
    #[cfg(test)]
    pub(super) fn rule_r2_pub(pag: &mut Pag) -> bool {
        Self::rule_r2(pag)
    }
    #[cfg(test)]
    pub(super) fn rule_r3_pub(pag: &mut Pag) -> bool {
        Self::rule_r3(pag)
    }
    #[cfg(test)]
    pub(super) fn rule_r4_pub(pag: &mut Pag) -> bool {
        Self::rule_r4(pag)
    }
}

fn possible_d_sep(pag: &Pag, x: usize, y: usize) -> BTreeSet<usize> {
    // Spirtes-Meek-Richardson 1999 Def. 11: v ∈ Possible-D-Sep(x, y) iff a
    // path between x and v exists such that every length-2 subpath <a, b, c>
    // has b a collider or forms a triangle.
    let mut result = BTreeSet::new();
    let mut frontier: Vec<(usize, Option<usize>)> = pag
        .neighbors(x)
        .into_iter()
        .filter(|&v| v != y)
        .map(|v| (v, Some(x)))
        .collect();
    let mut visited: BTreeSet<(usize, Option<usize>)> = frontier.iter().copied().collect();
    while let Some((v, prev)) = frontier.pop() {
        if v != x && v != y {
            result.insert(v);
        }
        for w in pag.neighbors(v) {
            if w == x || w == y || w == v {
                continue;
            }
            let ok = match prev {
                None => true,
                Some(p) => {
                    let collider_at_v =
                        pag.mark(p, v) == EdgeMark::Arrow && pag.mark(w, v) == EdgeMark::Arrow;
                    let triangle = pag.adj(p, w);
                    collider_at_v || triangle
                }
            };
            if !ok {
                continue;
            }
            let key = (w, Some(v));
            if visited.insert(key) {
                frontier.push((w, Some(v)));
            }
        }
    }
    result
}

fn find_discriminating_origin(pag: &Pag, alpha: usize, beta: usize, gamma: usize) -> Option<usize> {
    let d = pag.n_vars;
    for theta in 0..d {
        if theta == alpha || theta == beta || theta == gamma {
            continue;
        }
        if pag.adj(theta, gamma) || !pag.adj(theta, alpha) {
            continue;
        }
        if pag.mark(theta, alpha) != EdgeMark::Arrow || pag.mark(alpha, gamma) != EdgeMark::Arrow {
            continue;
        }
        return Some(theta);
    }
    None
}
