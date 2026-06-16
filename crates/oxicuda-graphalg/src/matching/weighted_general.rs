//! Maximum-weight matching on a general (non-bipartite) weighted graph.
//!
//! Implements Edmonds' weighted blossom algorithm in the `O(n^3)` primal-dual
//! array formulation of Galil (1986) — the same formulation underlying Van
//! Rantwijk's `mwmatching` / NetworkX `max_weight_matching`. Unlike the
//! cardinality-only [`blossom_v_simple`](super::blossom_v_simple), this finds a
//! matching of **maximum total weight** (which need not be a maximum-cardinality
//! matching), maintaining vertex potentials `y_v` and blossom potentials `z_B`
//! with the usual complementary-slackness invariants:
//!
//! * `slack(u,v) = y_u + y_v - 2 w(u,v) >= 0` for every edge,
//! * matched edges and full blossoms are tight (slack `0`),
//! * unmatched vertices keep `y_v >= 0` (so leaving a vertex unmatched is optimal
//!   exactly when `y_v = 0`).
//!
//! Alternating trees are grown from free vertices; odd cycles are contracted
//! into blossoms and expanded as the dual variables are adjusted. The public
//! entry point is [`WeightedGeneralMatching`].

use std::collections::HashMap;

use crate::error::{GraphalgError, GraphalgResult};
use crate::repr::weighted_graph::WeightedGraph;

/// Result of a maximum-weight general matching solve.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedMatchingResult {
    /// `mate[v] = Some(u)` iff `v` is matched to `u`; `None` if unmatched.
    pub mate: Vec<Option<usize>>,
    /// The matched edges as `(u, v, weight)` with `u < v`, sorted.
    pub matched_edges: Vec<(usize, usize, f64)>,
    /// Total weight of the matching.
    pub total_weight: f64,
}

/// Maximum-weight matching solver for a general undirected weighted graph.
#[derive(Debug, Clone)]
pub struct WeightedGeneralMatching {
    n: usize,
    /// Canonical `(min, max) -> weight` map (parallel edges keep the max weight,
    /// self-loops ignored).
    pair_weight: HashMap<(usize, usize), f64>,
}

impl WeightedGeneralMatching {
    /// Create an edgeless instance on `n` vertices.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            pair_weight: HashMap::new(),
        }
    }

    /// Add an undirected weighted edge `(u, v, w)`.
    ///
    /// Self-loops are ignored. Parallel edges are de-duplicated keeping the
    /// largest weight (only positive-contribution edges ever enter a maximum
    /// weight matching).
    pub fn add_edge(&mut self, u: usize, v: usize, w: f64) -> GraphalgResult<()> {
        if u >= self.n || v >= self.n {
            return Err(GraphalgError::IndexOutOfBounds {
                index: u.max(v),
                len: self.n,
            });
        }
        if !w.is_finite() {
            return Err(GraphalgError::InvalidEdgeWeight(format!(
                "edge ({u},{v}) weight not finite: {w}"
            )));
        }
        if u == v {
            return Ok(());
        }
        let key = if u < v { (u, v) } else { (v, u) };
        self.pair_weight
            .entry(key)
            .and_modify(|old| {
                if w > *old {
                    *old = w;
                }
            })
            .or_insert(w);
        Ok(())
    }

    /// Build a solver from a [`WeightedGraph`] (treated as undirected; the larger
    /// weight is kept for each pair).
    pub fn from_weighted_graph(g: &WeightedGraph) -> GraphalgResult<Self> {
        let mut m = Self::new(g.n);
        for (u, v, w) in g.all_edges() {
            m.add_edge(u, v, w)?;
        }
        Ok(m)
    }

    /// Number of vertices.
    pub fn num_nodes(&self) -> usize {
        self.n
    }

    /// The de-duplicated edge list as `(u, v, w)` with `u < v`, sorted.
    pub fn edges(&self) -> Vec<(usize, usize, f64)> {
        let mut e: Vec<(usize, usize, f64)> = self
            .pair_weight
            .iter()
            .map(|(&(u, v), &w)| (u, v, w))
            .collect();
        e.sort_by_key(|a| (a.0, a.1));
        e
    }

    /// Solve for a **maximum-weight** matching (not necessarily of maximum
    /// cardinality).
    pub fn solve(&self) -> GraphalgResult<WeightedMatchingResult> {
        self.solve_inner(false)
    }

    /// Solve for a maximum-weight matching **among the maximum-cardinality**
    /// matchings.
    pub fn solve_max_cardinality(&self) -> GraphalgResult<WeightedMatchingResult> {
        self.solve_inner(true)
    }

    fn solve_inner(&self, maxcardinality: bool) -> GraphalgResult<WeightedMatchingResult> {
        let edges = self.edges();
        let mate_vertices = if edges.is_empty() || self.n == 0 {
            vec![-1i64; self.n]
        } else {
            let mut matcher = Matcher::new(self.n, &edges, maxcardinality);
            matcher.run()?;
            matcher.mate
        };
        let mut mate = vec![None; self.n];
        let mut matched_edges = Vec::new();
        let mut total_weight = 0.0;
        for v in 0..self.n {
            if mate_vertices[v] >= 0 {
                let u = mate_vertices[v] as usize;
                mate[v] = Some(u);
                if v < u {
                    let key = (v, u);
                    let w = *self.pair_weight.get(&key).unwrap_or(&0.0);
                    matched_edges.push((v, u, w));
                    total_weight += w;
                }
            }
        }
        matched_edges.sort_by_key(|a| (a.0, a.1));
        Ok(WeightedMatchingResult {
            mate,
            matched_edges,
            total_weight,
        })
    }
}

/// Python-style modular index into a length-`len` ring (`len > 0`).
fn ring_index(idx: isize, len: isize) -> usize {
    (((idx % len) + len) % len) as usize
}

/// Internal blossom matcher (array formulation). Index conventions follow Galil
/// / Van Rantwijk: an *endpoint* `p` refers to `edges[p / 2]`, with `p ^ 1`
/// flipping between the two endpoints of an edge; `endpoint[p]` is the vertex at
/// endpoint `p`.
struct Matcher {
    nvertex: usize,
    edges: Vec<(usize, usize, f64)>,
    endpoint: Vec<usize>,
    neighbend: Vec<Vec<usize>>,
    /// `mate[v]` = matched endpoint (then converted to vertex), or `-1`.
    mate: Vec<i64>,
    /// Label: `-1` removed, `0` free, `1` S/outer, `2` T/inner, `5` scan-temp.
    label: Vec<i64>,
    labelend: Vec<i64>,
    inblossom: Vec<usize>,
    blossomparent: Vec<i64>,
    blossomchilds: Vec<Vec<usize>>,
    blossombase: Vec<i64>,
    blossomendps: Vec<Vec<usize>>,
    bestedge: Vec<i64>,
    blossombestedges: Vec<Option<Vec<usize>>>,
    unusedblossoms: Vec<usize>,
    dualvar: Vec<f64>,
    allowedge: Vec<bool>,
    queue: Vec<usize>,
    maxcardinality: bool,
}

impl Matcher {
    fn new(nvertex: usize, edges: &[(usize, usize, f64)], maxcardinality: bool) -> Self {
        let nedge = edges.len();
        let mut endpoint = vec![0usize; 2 * nedge];
        for (k, &(i, j, _)) in edges.iter().enumerate() {
            endpoint[2 * k] = i;
            endpoint[2 * k + 1] = j;
        }
        let mut neighbend = vec![Vec::new(); nvertex];
        for (k, &(i, j, _)) in edges.iter().enumerate() {
            neighbend[i].push(2 * k + 1);
            neighbend[j].push(2 * k);
        }
        let mut maxweight = 0.0f64;
        for &(_, _, w) in edges {
            if w > maxweight {
                maxweight = w;
            }
        }
        let mut dualvar = vec![0.0f64; 2 * nvertex];
        for d in dualvar.iter_mut().take(nvertex) {
            *d = maxweight;
        }
        Self {
            nvertex,
            edges: edges.to_vec(),
            endpoint,
            neighbend,
            mate: vec![-1; nvertex],
            label: vec![0; 2 * nvertex],
            labelend: vec![-1; 2 * nvertex],
            inblossom: (0..nvertex).collect(),
            blossomparent: vec![-1; 2 * nvertex],
            blossomchilds: vec![Vec::new(); 2 * nvertex],
            blossombase: (0..nvertex as i64)
                .chain(std::iter::repeat_n(-1, nvertex))
                .collect(),
            blossomendps: vec![Vec::new(); 2 * nvertex],
            bestedge: vec![-1; 2 * nvertex],
            blossombestedges: vec![None; 2 * nvertex],
            unusedblossoms: (nvertex..2 * nvertex).collect(),
            dualvar,
            allowedge: vec![false; nedge],
            queue: Vec::new(),
            maxcardinality,
        }
    }

    fn slack(&self, k: usize) -> f64 {
        let (i, j, w) = self.edges[k];
        self.dualvar[i] + self.dualvar[j] - 2.0 * w
    }

    fn blossom_leaves(&self, b: usize) -> Vec<usize> {
        if b < self.nvertex {
            return vec![b];
        }
        let mut out = Vec::new();
        for &t in &self.blossomchilds[b] {
            if t < self.nvertex {
                out.push(t);
            } else {
                out.extend(self.blossom_leaves(t));
            }
        }
        out
    }

    fn assign_label(&mut self, w: usize, t: i64, p: i64) {
        let b = self.inblossom[w];
        self.label[w] = t;
        self.label[b] = t;
        self.labelend[w] = p;
        self.labelend[b] = p;
        self.bestedge[w] = -1;
        self.bestedge[b] = -1;
        if t == 1 {
            let leaves = self.blossom_leaves(b);
            self.queue.extend(leaves);
        } else if t == 2 {
            let base = self.blossombase[b] as usize;
            let mb = self.mate[base];
            let endp = self.endpoint[mb as usize];
            self.assign_label(endp, 1, mb ^ 1);
        }
    }

    fn scan_blossom(&mut self, v: usize, w: usize) -> i64 {
        let mut path: Vec<usize> = Vec::new();
        let mut base: i64 = -1;
        let mut v: i64 = v as i64;
        let mut w: i64 = w as i64;
        while v != -1 || w != -1 {
            let b = self.inblossom[v as usize];
            if self.label[b] & 4 != 0 {
                base = self.blossombase[b];
                break;
            }
            path.push(b);
            self.label[b] = 5;
            if self.labelend[b] == -1 {
                v = -1;
            } else {
                v = self.endpoint[self.labelend[b] as usize] as i64;
                let b2 = self.inblossom[v as usize];
                v = self.endpoint[self.labelend[b2] as usize] as i64;
            }
            if w != -1 {
                std::mem::swap(&mut v, &mut w);
            }
        }
        for b in path {
            self.label[b] = 1;
        }
        base
    }

    fn add_blossom(&mut self, base: usize, k: usize) -> GraphalgResult<()> {
        let (v0, w0, _) = self.edges[k];
        let bb = self.inblossom[base];
        let b = self
            .unusedblossoms
            .pop()
            .ok_or_else(|| GraphalgError::NumericalInstability("blossom pool exhausted".into()))?;
        self.blossombase[b] = base as i64;
        self.blossomparent[b] = -1;
        self.blossomparent[bb] = b as i64;
        let mut path: Vec<usize> = Vec::new();
        let mut endps: Vec<usize> = Vec::new();
        let mut v = v0;
        let mut bv = self.inblossom[v0];
        while bv != bb {
            self.blossomparent[bv] = b as i64;
            path.push(bv);
            endps.push(self.labelend[bv] as usize);
            v = self.endpoint[self.labelend[bv] as usize];
            bv = self.inblossom[v];
        }
        path.push(bb);
        path.reverse();
        endps.reverse();
        endps.push(2 * k);
        let mut w = w0;
        let mut bw = self.inblossom[w0];
        while bw != bb {
            self.blossomparent[bw] = b as i64;
            path.push(bw);
            endps.push((self.labelend[bw] as usize) ^ 1);
            w = self.endpoint[self.labelend[bw] as usize];
            bw = self.inblossom[w];
        }
        let _ = (v, w);
        self.blossomchilds[b] = path;
        self.blossomendps[b] = endps;
        self.label[b] = 1;
        self.labelend[b] = self.labelend[bb];
        self.dualvar[b] = 0.0;
        for leaf in self.blossom_leaves(b) {
            if self.label[self.inblossom[leaf]] == 2 {
                self.queue.push(leaf);
            }
            self.inblossom[leaf] = b;
        }
        // Compute best edges from the new blossom to each external S-structure.
        let mut bestedgeto = vec![-1i64; 2 * self.nvertex];
        let childs = self.blossomchilds[b].clone();
        for &bv in &childs {
            let nblists: Vec<Vec<usize>> = match &self.blossombestedges[bv] {
                Some(list) => vec![list.clone()],
                None => self
                    .blossom_leaves(bv)
                    .into_iter()
                    .map(|leaf| {
                        self.neighbend[leaf]
                            .iter()
                            .map(|&p| p / 2)
                            .collect::<Vec<usize>>()
                    })
                    .collect(),
            };
            for nblist in nblists {
                for kk in nblist {
                    let (mut i, mut j, _) = self.edges[kk];
                    if self.inblossom[j] == b {
                        std::mem::swap(&mut i, &mut j);
                    }
                    let bj = self.inblossom[j];
                    if bj != b
                        && self.label[bj] == 1
                        && (bestedgeto[bj] == -1
                            || self.slack(kk) < self.slack(bestedgeto[bj] as usize))
                    {
                        bestedgeto[bj] = kk as i64;
                    }
                }
            }
            self.blossombestedges[bv] = None;
            self.bestedge[bv] = -1;
        }
        let collected: Vec<usize> = bestedgeto
            .iter()
            .filter(|&&kk| kk != -1)
            .map(|&kk| kk as usize)
            .collect();
        self.blossombestedges[b] = Some(collected.clone());
        self.bestedge[b] = -1;
        for kk in collected {
            if self.bestedge[b] == -1 || self.slack(kk) < self.slack(self.bestedge[b] as usize) {
                self.bestedge[b] = kk as i64;
            }
        }
        Ok(())
    }

    fn expand_blossom(&mut self, b: usize, endstage: bool) -> GraphalgResult<()> {
        let childs = self.blossomchilds[b].clone();
        for &s in &childs {
            self.blossomparent[s] = -1;
            if s < self.nvertex {
                self.inblossom[s] = s;
            } else if endstage && self.dualvar[s] == 0.0 {
                self.expand_blossom(s, endstage)?;
            } else {
                for leaf in self.blossom_leaves(s) {
                    self.inblossom[leaf] = s;
                }
            }
        }
        if !endstage && self.label[b] == 2 {
            let entry_ep = (self.labelend[b] as usize) ^ 1;
            let entrychild = self.inblossom[self.endpoint[entry_ep]];
            let childs_b = self.blossomchilds[b].clone();
            let endps_b = self.blossomendps[b].clone();
            let len = childs_b.len() as isize;
            let pos = childs_b
                .iter()
                .position(|&c| c == entrychild)
                .ok_or_else(|| GraphalgError::NumericalInstability("entrychild missing".into()))?;
            let mut j = pos as isize;
            let jstep: isize;
            let endptrick: isize;
            if j & 1 != 0 {
                j -= len;
                jstep = 1;
                endptrick = 0;
            } else {
                jstep = -1;
                endptrick = 1;
            }
            let mut p = self.labelend[b];
            while j != 0 {
                self.label[self.endpoint[(p as usize) ^ 1]] = 0;
                let e1 = endps_b[ring_index(j - endptrick, len)];
                self.label[self.endpoint[e1 ^ (endptrick as usize) ^ 1]] = 0;
                let endp = self.endpoint[(p as usize) ^ 1];
                self.assign_label(endp, 2, p);
                self.allowedge[e1 / 2] = true;
                j += jstep;
                let e2 = endps_b[ring_index(j - endptrick, len)];
                p = (e2 ^ (endptrick as usize)) as i64;
                self.allowedge[(p as usize) / 2] = true;
                j += jstep;
            }
            let bv = childs_b[ring_index(j, len)];
            let endp = self.endpoint[(p as usize) ^ 1];
            self.label[endp] = 2;
            self.label[bv] = 2;
            self.labelend[endp] = p;
            self.labelend[bv] = p;
            self.bestedge[bv] = -1;
            j += jstep;
            while childs_b[ring_index(j, len)] != entrychild {
                let bv = childs_b[ring_index(j, len)];
                if self.label[bv] == 1 {
                    j += jstep;
                    continue;
                }
                let mut vv = usize::MAX;
                for leaf in self.blossom_leaves(bv) {
                    if self.label[leaf] != 0 {
                        vv = leaf;
                        break;
                    }
                }
                if vv != usize::MAX {
                    self.label[vv] = 0;
                    let base_bv = self.blossombase[bv] as usize;
                    let mep = self.mate[base_bv];
                    self.label[self.endpoint[mep as usize]] = 0;
                    let le = self.labelend[vv];
                    self.assign_label(vv, 2, le);
                }
                j += jstep;
            }
        }
        self.label[b] = -1;
        self.labelend[b] = -1;
        self.blossomchilds[b] = Vec::new();
        self.blossomendps[b] = Vec::new();
        self.blossombase[b] = -1;
        self.blossombestedges[b] = None;
        self.bestedge[b] = -1;
        self.unusedblossoms.push(b);
        Ok(())
    }

    fn augment_blossom(&mut self, b: usize, v: usize) -> GraphalgResult<()> {
        let mut t = v;
        while self.blossomparent[t] != b as i64 {
            t = self.blossomparent[t] as usize;
        }
        if t >= self.nvertex {
            self.augment_blossom(t, v)?;
        }
        let childs_b = self.blossomchilds[b].clone();
        let endps_b = self.blossomendps[b].clone();
        let len = childs_b.len() as isize;
        let i =
            childs_b.iter().position(|&c| c == t).ok_or_else(|| {
                GraphalgError::NumericalInstability("augment child missing".into())
            })? as isize;
        let mut j = i;
        let jstep: isize;
        let endptrick: isize;
        if i & 1 != 0 {
            j -= len;
            jstep = 1;
            endptrick = 0;
        } else {
            jstep = -1;
            endptrick = 1;
        }
        while j != 0 {
            j += jstep;
            let t1 = childs_b[ring_index(j, len)];
            let p = endps_b[ring_index(j - endptrick, len)] ^ (endptrick as usize);
            if t1 >= self.nvertex {
                let ep = self.endpoint[p];
                self.augment_blossom(t1, ep)?;
            }
            j += jstep;
            let t2 = childs_b[ring_index(j, len)];
            if t2 >= self.nvertex {
                let ep = self.endpoint[p ^ 1];
                self.augment_blossom(t2, ep)?;
            }
            self.mate[self.endpoint[p]] = (p ^ 1) as i64;
            self.mate[self.endpoint[p ^ 1]] = p as i64;
        }
        let iu = i as usize;
        let new_childs: Vec<usize> = childs_b[iu..]
            .iter()
            .chain(childs_b[..iu].iter())
            .copied()
            .collect();
        let new_endps: Vec<usize> = endps_b[iu..]
            .iter()
            .chain(endps_b[..iu].iter())
            .copied()
            .collect();
        self.blossomchilds[b] = new_childs;
        self.blossomendps[b] = new_endps;
        self.blossombase[b] = self.blossombase[self.blossomchilds[b][0]];
        Ok(())
    }

    fn augment_matching(&mut self, k: usize) -> GraphalgResult<()> {
        let (v, w, _) = self.edges[k];
        let starts = [(v, (2 * k + 1) as i64), (w, (2 * k) as i64)];
        for &(s0, p0) in &starts {
            let mut s = s0;
            let mut p = p0;
            loop {
                let bs = self.inblossom[s];
                if bs >= self.nvertex {
                    self.augment_blossom(bs, s)?;
                }
                self.mate[s] = p;
                if self.labelend[bs] == -1 {
                    break;
                }
                let t = self.endpoint[self.labelend[bs] as usize];
                let bt = self.inblossom[t];
                s = self.endpoint[self.labelend[bt] as usize];
                let j = self.endpoint[(self.labelend[bt] as usize) ^ 1];
                if bt >= self.nvertex {
                    self.augment_blossom(bt, j)?;
                }
                self.mate[j] = self.labelend[bt];
                p = self.labelend[bt] ^ 1;
            }
        }
        Ok(())
    }

    fn min_vertex_dual(&self) -> f64 {
        let mut m = f64::INFINITY;
        for v in 0..self.nvertex {
            if self.dualvar[v] < m {
                m = self.dualvar[v];
            }
        }
        m
    }

    fn run(&mut self) -> GraphalgResult<()> {
        let guard_cap = 16 * self.nvertex * self.nvertex + 1000;
        for _phase in 0..self.nvertex {
            for x in self.label.iter_mut() {
                *x = 0;
            }
            for x in self.bestedge.iter_mut() {
                *x = -1;
            }
            for b in self.nvertex..2 * self.nvertex {
                self.blossombestedges[b] = None;
            }
            self.queue.clear();
            for x in self.allowedge.iter_mut() {
                *x = false;
            }
            for v in 0..self.nvertex {
                if self.mate[v] == -1 && self.label[self.inblossom[v]] == 0 {
                    self.assign_label(v, 1, -1);
                }
            }
            let mut augmented = false;
            let mut guard = 0usize;
            loop {
                guard += 1;
                if guard > guard_cap {
                    return Err(GraphalgError::NumericalInstability(
                        "weighted matching did not converge".into(),
                    ));
                }
                while !augmented {
                    let v = match self.queue.pop() {
                        Some(v) => v,
                        None => break,
                    };
                    let neigh = self.neighbend[v].clone();
                    for p in neigh {
                        let k = p / 2;
                        let w = self.endpoint[p];
                        if self.inblossom[v] == self.inblossom[w] {
                            continue;
                        }
                        if !self.allowedge[k] {
                            let kslack = self.slack(k);
                            if kslack <= 0.0 {
                                self.allowedge[k] = true;
                            }
                        }
                        if self.allowedge[k] {
                            if self.label[self.inblossom[w]] == 0 {
                                self.assign_label(w, 2, (p ^ 1) as i64);
                            } else if self.label[self.inblossom[w]] == 1 {
                                let base = self.scan_blossom(v, w);
                                if base >= 0 {
                                    self.add_blossom(base as usize, k)?;
                                } else {
                                    self.augment_matching(k)?;
                                    augmented = true;
                                    break;
                                }
                            } else if self.label[w] == 0 {
                                self.label[w] = 2;
                                self.labelend[w] = (p ^ 1) as i64;
                            }
                        } else if self.label[self.inblossom[w]] == 1 {
                            let bvv = self.inblossom[v];
                            if self.bestedge[bvv] == -1
                                || self.slack(k) < self.slack(self.bestedge[bvv] as usize)
                            {
                                self.bestedge[bvv] = k as i64;
                            }
                        } else if self.label[w] == 0
                            && (self.bestedge[w] == -1
                                || self.slack(k) < self.slack(self.bestedge[w] as usize))
                        {
                            self.bestedge[w] = k as i64;
                        }
                    }
                }
                if augmented {
                    break;
                }
                let mut deltatype: i32 = -1;
                let mut delta = 0.0f64;
                let mut deltaedge: i64 = -1;
                let mut deltablossom: i64 = -1;
                if !self.maxcardinality {
                    deltatype = 1;
                    delta = self.min_vertex_dual().max(0.0);
                }
                for v in 0..self.nvertex {
                    if self.label[self.inblossom[v]] == 0 && self.bestedge[v] != -1 {
                        let d = self.slack(self.bestedge[v] as usize);
                        if deltatype == -1 || d < delta {
                            delta = d;
                            deltatype = 2;
                            deltaedge = self.bestedge[v];
                        }
                    }
                }
                for b in 0..2 * self.nvertex {
                    if self.blossomparent[b] == -1 && self.label[b] == 1 && self.bestedge[b] != -1 {
                        let d = self.slack(self.bestedge[b] as usize) / 2.0;
                        if deltatype == -1 || d < delta {
                            delta = d;
                            deltatype = 3;
                            deltaedge = self.bestedge[b];
                        }
                    }
                }
                for b in self.nvertex..2 * self.nvertex {
                    if self.blossombase[b] >= 0
                        && self.blossomparent[b] == -1
                        && self.label[b] == 2
                        && (deltatype == -1 || self.dualvar[b] < delta)
                    {
                        delta = self.dualvar[b];
                        deltatype = 4;
                        deltablossom = b as i64;
                    }
                }
                if deltatype == -1 {
                    deltatype = 1;
                    delta = self.min_vertex_dual().max(0.0);
                }
                for v in 0..self.nvertex {
                    let lbl = self.label[self.inblossom[v]];
                    if lbl == 1 {
                        self.dualvar[v] -= delta;
                    } else if lbl == 2 {
                        self.dualvar[v] += delta;
                    }
                }
                for b in self.nvertex..2 * self.nvertex {
                    if self.blossombase[b] >= 0 && self.blossomparent[b] == -1 {
                        if self.label[b] == 1 {
                            self.dualvar[b] += delta;
                        } else if self.label[b] == 2 {
                            self.dualvar[b] -= delta;
                        }
                    }
                }
                if deltatype == 1 {
                    break;
                } else if deltatype == 2 {
                    let de = deltaedge as usize;
                    self.allowedge[de] = true;
                    let (mut i, j, _) = self.edges[de];
                    if self.label[self.inblossom[i]] == 0 {
                        i = j;
                    }
                    self.queue.push(i);
                } else if deltatype == 3 {
                    let de = deltaedge as usize;
                    self.allowedge[de] = true;
                    let (i, _, _) = self.edges[de];
                    self.queue.push(i);
                } else if deltatype == 4 {
                    self.expand_blossom(deltablossom as usize, false)?;
                }
            }
            if !augmented {
                break;
            }
            for b in self.nvertex..2 * self.nvertex {
                if self.blossomparent[b] == -1
                    && self.blossombase[b] >= 0
                    && self.label[b] == 1
                    && self.dualvar[b] == 0.0
                {
                    self.expand_blossom(b, true)?;
                }
            }
        }
        for v in 0..self.nvertex {
            if self.mate[v] >= 0 {
                self.mate[v] = self.endpoint[self.mate[v] as usize] as i64;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-7
    }

    /// Validate the structural invariants of a matching result.
    fn validate(res: &WeightedMatchingResult, edges: &[(usize, usize, f64)], n: usize) {
        // mate is symmetric, no shared vertices.
        let mut used = vec![false; n];
        for v in 0..n {
            if let Some(u) = res.mate[v] {
                assert_eq!(res.mate[u], Some(v), "mate not symmetric at {v}");
                assert!(!used[v], "vertex {v} shared by two matched edges");
                used[v] = true;
            }
        }
        // Reported total weight equals the summed weights of matched edges.
        let mut sum = 0.0;
        for &(u, v, w) in &res.matched_edges {
            assert_eq!(res.mate[u], Some(v));
            // weight matches an actual edge
            let found = edges
                .iter()
                .any(|&(a, b, ww)| ((a, b) == (u, v) || (a, b) == (v, u)) && approx(ww, w));
            assert!(found, "matched edge ({u},{v},{w}) not in graph");
            sum += w;
        }
        assert!(approx(sum, res.total_weight), "weight mismatch");
    }

    fn edge_weight(edges: &[(usize, usize, f64)], i: usize, j: usize) -> Option<f64> {
        let mut best: Option<f64> = None;
        for &(a, b, w) in edges {
            if (a, b) == (i, j) || (a, b) == (j, i) {
                best = Some(best.map_or(w, |x: f64| x.max(w)));
            }
        }
        best
    }

    /// Exact maximum-weight matching via subset DP (oracle), `n <= ~12`.
    fn brute_max_weight(n: usize, edges: &[(usize, usize, f64)]) -> f64 {
        let full = 1usize << n;
        let mut dp = vec![0.0f64; full];
        for mask in 1..full {
            let i = mask.trailing_zeros() as usize;
            let without_i = mask & !(1 << i);
            let mut best = dp[without_i];
            let mut rest = without_i;
            while rest != 0 {
                let jb = rest.trailing_zeros() as usize;
                if let Some(w) = edge_weight(edges, i, jb) {
                    let cand = w + dp[without_i & !(1 << jb)];
                    if cand > best {
                        best = cand;
                    }
                }
                rest &= rest - 1;
            }
            dp[mask] = best;
        }
        dp[full - 1]
    }

    #[test]
    fn triangle_picks_heaviest_edge() {
        let mut m = WeightedGeneralMatching::new(3);
        m.add_edge(0, 1, 3.0).expect("e");
        m.add_edge(1, 2, 5.0).expect("e");
        m.add_edge(0, 2, 4.0).expect("e");
        let res = m.solve().expect("solve");
        validate(&res, &m.edges(), 3);
        assert!(approx(res.total_weight, 5.0), "got {}", res.total_weight);
        assert_eq!(res.matched_edges, vec![(1, 2, 5.0)]);
    }

    #[test]
    fn empty_and_single_vertex() {
        let m0 = WeightedGeneralMatching::new(0);
        let r0 = m0.solve().expect("solve");
        assert!(r0.matched_edges.is_empty());
        assert!(approx(r0.total_weight, 0.0));

        let m1 = WeightedGeneralMatching::new(1);
        let r1 = m1.solve().expect("solve");
        assert_eq!(r1.mate, vec![None]);
        assert!(approx(r1.total_weight, 0.0));
    }

    #[test]
    fn path_optimal_alternating_set() {
        // Path 0-1-2-3-4 with weights 1,2,2,1.
        // Optimal: edges (1,2) and ... no, (1,2) blocks 0 and 3. Best alternating:
        // (0,1)+(2,3)=1+2=3, (1,2)+(3,4)=2+1=3, (0,1)+(3,4)=1+1=2 ... max = 3 via
        // (1,2)+(3,4) or (0,1)+(2,3). Verified against brute force below.
        let mut m = WeightedGeneralMatching::new(5);
        let edges = [(0, 1, 1.0), (1, 2, 2.0), (2, 3, 2.0), (3, 4, 1.0)];
        for &(u, v, w) in &edges {
            m.add_edge(u, v, w).expect("e");
        }
        let res = m.solve().expect("solve");
        validate(&res, &m.edges(), 5);
        let brute = brute_max_weight(5, &m.edges());
        assert!(
            approx(res.total_weight, brute),
            "got {} brute {brute}",
            res.total_weight
        );
        assert!(approx(res.total_weight, 3.0));
    }

    #[test]
    fn blossom_needed_odd_cycle() {
        // 5-cycle with a heavy chord forces an odd-cycle (blossom) decision.
        let mut m = WeightedGeneralMatching::new(5);
        let edges = [
            (0, 1, 6.0),
            (1, 2, 6.0),
            (2, 3, 6.0),
            (3, 4, 6.0),
            (4, 0, 6.0),
            (0, 2, 10.0),
        ];
        for &(u, v, w) in &edges {
            m.add_edge(u, v, w).expect("e");
        }
        let res = m.solve().expect("solve");
        validate(&res, &m.edges(), 5);
        let brute = brute_max_weight(5, &m.edges());
        assert!(
            approx(res.total_weight, brute),
            "got {} brute {brute}",
            res.total_weight
        );
    }

    #[test]
    fn matches_brute_force_on_random_graphs() {
        let mut rng = LcgRng::new(0xC0FFEE);
        for iter in 0..200 {
            let n = 2 + (rng.next_u64() as usize % 8); // 2..=9
            // alternate sparse/dense to exercise both isolated vertices and blossoms
            let prob = if iter % 2 == 0 { 50 } else { 80 };
            let mut m = WeightedGeneralMatching::new(n);
            let mut edges = Vec::new();
            for u in 0..n {
                for v in (u + 1)..n {
                    if rng.next_u64() % 100 < prob {
                        let w = 1.0 + (rng.next_u64() as usize % 9) as f64; // 1..=9
                        m.add_edge(u, v, w).expect("e");
                        edges.push((u, v, w));
                    }
                }
            }
            let res = m.solve().expect("solve");
            validate(&res, &m.edges(), n);
            let brute = brute_max_weight(n, &m.edges());
            assert!(
                approx(res.total_weight, brute),
                "n={n} edges={:?} got {} brute {brute}",
                edges,
                res.total_weight
            );
        }
    }

    #[test]
    fn complete_graph_even_perfect() {
        // K4 with weights; max weight matching is a perfect matching.
        let mut m = WeightedGeneralMatching::new(4);
        m.add_edge(0, 1, 1.0).expect("e");
        m.add_edge(2, 3, 1.0).expect("e");
        m.add_edge(0, 2, 5.0).expect("e");
        m.add_edge(1, 3, 5.0).expect("e");
        m.add_edge(0, 3, 2.0).expect("e");
        m.add_edge(1, 2, 2.0).expect("e");
        let res = m.solve().expect("solve");
        validate(&res, &m.edges(), 4);
        // Best: (0,2)+(1,3) = 10.
        assert!(approx(res.total_weight, 10.0), "got {}", res.total_weight);
        assert_eq!(res.matched_edges.len(), 2);
    }

    #[test]
    fn rejects_bad_edges() {
        let mut m = WeightedGeneralMatching::new(3);
        assert!(m.add_edge(0, 5, 1.0).is_err());
        assert!(m.add_edge(0, 1, f64::NAN).is_err());
        // self-loop ignored, not an error
        assert!(m.add_edge(1, 1, 3.0).is_ok());
        assert_eq!(m.edges().len(), 0);
    }

    #[test]
    fn parallel_edges_keep_max() {
        let mut m = WeightedGeneralMatching::new(2);
        m.add_edge(0, 1, 2.0).expect("e");
        m.add_edge(1, 0, 7.0).expect("e");
        let res = m.solve().expect("solve");
        assert!(approx(res.total_weight, 7.0));
    }
}
