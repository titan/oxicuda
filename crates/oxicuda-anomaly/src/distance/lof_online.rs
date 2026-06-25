//! Incremental / Online Local Outlier Factor (Pokrajac, Lazarevic & Latecki 2007).
//!
//! Maintains a k-nearest-neighbour graph together with per-point k-distances,
//! local reachability densities (`lrd`) and LOF scores, and updates them on every
//! point **insertion** (and **deletion**) by touching only the *affected*
//! neighbourhood rather than recomputing LOF for the whole data set.
//!
//! The static definitions match [`crate::distance::lof`] /
//! [`crate::distance::lof_kdtree`]:
//!
//! ```text
//! k-distance(p)      = distance from p to its k-th nearest neighbour
//! N_k(p)             = the k nearest neighbours of p (self excluded)
//! reach_dist_k(p,o)  = max(k-distance(o), dist(p, o))
//! lrd_k(p)           = k / Σ_{o ∈ N_k(p)} reach_dist_k(p, o)
//! LOF_k(p)           = (1/k) · Σ_{o ∈ N_k(p)} lrd_k(o) / lrd_k(p)
//! ```
//!
//! `LOF ≈ 1.0` → normal; `LOF ≫ 1.0` → strong outlier.
//!
//! # Incremental insertion (Pokrajac 2007)
//!
//! When a new point `pc` is inserted:
//! 1. compute `N_k(pc)`, `k-distance(pc)`, `lrd(pc)` and `LOF(pc)`;
//! 2. determine `S_update_k` = reverse-k-NN of `pc` (existing points `p` with
//!    `dist(p, pc) ≤ k-distance(p)`): for these, `pc` enters `N_k(p)` and may evict
//!    the previous k-th neighbour, lowering `k-distance(p)`;
//! 3. update `N_k` / `k-distance` for every `p ∈ S_update_k`;
//! 4. recompute `lrd` for every point whose reach-distances changed — that is
//!    `S_update_k` together with the reverse-neighbours of `S_update_k`
//!    (their `reach_dist(·, p)` depends on the changed `k-distance(p)`);
//! 5. recompute `LOF` for every point whose own `lrd` changed *or* one of whose
//!    neighbours' `lrd` changed.
//!
//! Only that affected subset is touched; there is **no** silent full refit.
//!
//! A reverse-neighbour adjacency (`rnn`) is maintained incrementally so the
//! affected-set propagation stays local — `O(k²)`-ish work per update in the
//! typical case.

use crate::error::{AnomalyError, AnomalyResult};

// ─── Config ────────────────────────────────────────────────────────────────────

/// Configuration for [`OnlineLof`].
#[derive(Debug, Clone)]
pub struct OnlineLofConfig {
    /// Number of nearest neighbours. Clamped to the number of *other* available
    /// points at query time, so a freshly seeded model still produces scores.
    pub k: usize,
}

impl Default for OnlineLofConfig {
    fn default() -> Self {
        Self { k: 5 }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────────

/// Squared Euclidean distance between two equal-length slices.
#[inline]
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Total-order comparison on `(distance, index)`.
///
/// Using the index as a tie-breaker makes neighbour selection deterministic, so
/// the incremental graph and a from-scratch recompute agree exactly even when
/// several candidates sit at the same distance.
#[inline]
fn cmp_dist_idx(a: &(f64, usize), b: &(f64, usize)) -> std::cmp::Ordering {
    a.0.partial_cmp(&b.0)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.1.cmp(&b.1))
}

// ─── OnlineLof ──────────────────────────────────────────────────────────────────

/// Incremental Local Outlier Factor detector.
///
/// Points are stored row-major in `OnlineLof::points`; deleted points are kept
/// as tombstones (`live[i] == false`) so that every other point keeps its index.
pub struct OnlineLof {
    /// Requested neighbour count.
    k: usize,
    /// Feature dimensionality.
    dim: usize,
    /// Flat `[capacity × dim]` row-major coordinates (tombstones included).
    points: Vec<f64>,
    /// `live[i]` is `false` for tombstoned (deleted) slots.
    live: Vec<bool>,
    /// Number of currently live points.
    n_live: usize,
    /// `neighbours[i]` — indices of `N_k(i)`, sorted by `(distance, index)`.
    neighbours: Vec<Vec<usize>>,
    /// `neigh_dist[i]` — distances aligned with `neighbours[i]`.
    neigh_dist: Vec<Vec<f64>>,
    /// `kdist[i]` — k-distance of point `i` (distance to its last neighbour).
    kdist: Vec<f64>,
    /// `lrd[i]` — local reachability density of point `i`.
    lrd: Vec<f64>,
    /// `lof[i]` — LOF score of point `i` (`NaN` for tombstones).
    lof: Vec<f64>,
    /// `rnn[o]` — reverse neighbours: points that currently hold `o` in their `N_k`.
    rnn: Vec<Vec<usize>>,
}

impl OnlineLof {
    /// Create an empty online-LOF model for `dim`-dimensional points.
    ///
    /// # Errors
    /// Returns [`AnomalyError::InvalidK`] when `k == 0` and
    /// [`AnomalyError::InvalidFeatureCount`] when `dim == 0`.
    pub fn new(k: usize, dim: usize) -> AnomalyResult<Self> {
        if k == 0 {
            return Err(AnomalyError::InvalidK { k: 0 });
        }
        if dim == 0 {
            return Err(AnomalyError::InvalidFeatureCount { n: 0 });
        }
        Ok(Self {
            k,
            dim,
            points: Vec::new(),
            live: Vec::new(),
            n_live: 0,
            neighbours: Vec::new(),
            neigh_dist: Vec::new(),
            kdist: Vec::new(),
            lrd: Vec::new(),
            lof: Vec::new(),
            rnn: Vec::new(),
        })
    }

    /// Create from a configuration.
    ///
    /// # Errors
    /// See [`OnlineLof::new`].
    pub fn from_config(cfg: &OnlineLofConfig, dim: usize) -> AnomalyResult<Self> {
        Self::new(cfg.k, dim)
    }

    /// Configured neighbour count.
    #[must_use]
    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Feature dimensionality.
    #[must_use]
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Total number of stored slots (live points plus tombstones).
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.live.len()
    }

    /// Number of currently live points.
    #[must_use]
    #[inline]
    pub fn n_live(&self) -> usize {
        self.n_live
    }

    /// `true` when no live points are stored.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_live == 0
    }

    /// Effective neighbour count given the current live population (excluding one
    /// query point): `min(k, n_live - 1)`.
    #[inline]
    fn effective_k(&self, exclude_self: bool) -> usize {
        let available = if exclude_self {
            self.n_live.saturating_sub(1)
        } else {
            self.n_live
        };
        self.k.min(available)
    }

    /// LOF scores for every slot (tombstones report `f64::NAN`).
    #[must_use]
    #[inline]
    pub fn lof_scores(&self) -> &[f64] {
        &self.lof
    }

    /// LOF score of slot `idx`.
    ///
    /// # Errors
    /// Returns [`AnomalyError::Internal`] when `idx` is out of range or tombstoned.
    pub fn score_of(&self, idx: usize) -> AnomalyResult<f64> {
        if idx >= self.live.len() || !self.live[idx] {
            return Err(AnomalyError::Internal {
                msg: format!("score_of: index {idx} is out of range or has been removed"),
            });
        }
        Ok(self.lof[idx])
    }

    /// Coordinates of slot `i`.
    #[inline]
    fn point(&self, i: usize) -> &[f64] {
        &self.points[i * self.dim..(i + 1) * self.dim]
    }

    /// Distance between slots `i` and `j`.
    #[inline]
    fn dist(&self, i: usize, j: usize) -> f64 {
        sq_dist(self.point(i), self.point(j)).sqrt()
    }

    // ── kNN over the live set ───────────────────────────────────────────────────

    /// Compute the `kk` nearest live neighbours of `query`, excluding `exclude`.
    ///
    /// Returns `(indices, distances)` ordered by `(distance, index)`.
    fn knn_live(&self, query: &[f64], kk: usize, exclude: Option<usize>) -> (Vec<usize>, Vec<f64>) {
        let mut cands: Vec<(f64, usize)> = Vec::with_capacity(self.n_live);
        for j in 0..self.live.len() {
            if !self.live[j] || Some(j) == exclude {
                continue;
            }
            cands.push((sq_dist(query, self.point(j)).sqrt(), j));
        }
        cands.sort_unstable_by(cmp_dist_idx);
        cands.truncate(kk);
        let idx = cands.iter().map(|&(_, i)| i).collect();
        let dst = cands.iter().map(|&(d, _)| d).collect();
        (idx, dst)
    }

    /// `lrd(i)` from `i`'s current neighbour list. `0.0` is returned when `i`
    /// has no neighbours (an isolated point).
    fn compute_lrd(&self, i: usize) -> f64 {
        let nbrs = &self.neighbours[i];
        if nbrs.is_empty() {
            return 0.0;
        }
        let mut sum_reach = 0.0_f64;
        for (slot, &o) in nbrs.iter().enumerate() {
            let d_io = self.neigh_dist[i][slot];
            let reach = self.kdist[o].max(d_io).max(0.0);
            sum_reach += reach;
        }
        if sum_reach < 1e-15 {
            f64::INFINITY
        } else {
            nbrs.len() as f64 / sum_reach
        }
    }

    /// `LOF(i)` from `i`'s current neighbours and the stored `lrd` values.
    fn compute_lof(&self, i: usize) -> f64 {
        let nbrs = &self.neighbours[i];
        if nbrs.is_empty() {
            return 1.0;
        }
        let lrd_i = self.lrd[i];
        // A point buried inside a perfectly dense cluster has infinite density;
        // mirror the static implementations and report LOF = 1.0 there.
        if lrd_i.is_infinite() {
            return 1.0;
        }
        if lrd_i <= 0.0 {
            return 1.0;
        }
        let acc: f64 = nbrs.iter().map(|&o| self.lrd[o] / lrd_i).sum();
        acc / nbrs.len() as f64
    }

    /// Recompute neighbour list, k-distance and `rnn` back-links for slot `i`
    /// from scratch over the current live set. Used for `pc` on insert and for
    /// points that lost a neighbour on delete.
    fn rebuild_neighbours(&mut self, i: usize) {
        // Detach old reverse links.
        let old: Vec<usize> = std::mem::take(&mut self.neighbours[i]);
        for o in old {
            self.rnn[o].retain(|&x| x != i);
        }
        self.neigh_dist[i].clear();

        let kk = self.effective_k(true);
        let query_owned = self.point(i).to_vec();
        let (idx, dst) = self.knn_live(&query_owned, kk, Some(i));
        self.kdist[i] = dst.last().copied().unwrap_or(0.0);
        for &o in &idx {
            self.rnn[o].push(i);
        }
        self.neighbours[i] = idx;
        self.neigh_dist[i] = dst;
    }

    // ── Insertion ───────────────────────────────────────────────────────────────

    /// Insert `point` and return its freshly computed LOF score.
    ///
    /// Maintains the kNN graph, k-distances, `lrd` and LOF incrementally: only the
    /// reverse-k-NN of the new point and the transitively affected neighbourhood
    /// are recomputed.
    ///
    /// # Errors
    /// Returns [`AnomalyError::DimensionMismatch`] when `point.len() != dim`.
    pub fn insert(&mut self, point: &[f64]) -> AnomalyResult<f64> {
        if point.len() != self.dim {
            return Err(AnomalyError::DimensionMismatch {
                expected: self.dim,
                got: point.len(),
            });
        }

        let pc = self.live.len();
        self.points.extend_from_slice(point);
        self.live.push(true);
        self.n_live += 1;
        self.neighbours.push(Vec::new());
        self.neigh_dist.push(Vec::new());
        self.kdist.push(0.0);
        self.lrd.push(0.0);
        self.lof.push(1.0);
        self.rnn.push(Vec::new());

        // The very first point cannot have neighbours yet.
        if self.n_live == 1 {
            self.kdist[pc] = 0.0;
            self.lrd[pc] = f64::INFINITY;
            self.lof[pc] = 1.0;
            return Ok(self.lof[pc]);
        }

        // ── Step 1: pc's own neighbours / k-distance ────────────────────────────
        self.rebuild_neighbours(pc);

        // ── Step 2: S_update_k — reverse-k-NN of pc ────────────────────────────
        // An existing live point p gains pc as a neighbour iff pc is at least as
        // close to p as p's current k-th neighbour. When p still has fewer than k
        // neighbours (early in the stream), pc is always added.
        let mut s_update_k: Vec<usize> = Vec::new();
        for p in 0..self.live.len() {
            if p == pc || !self.live[p] {
                continue;
            }
            let d = self.dist(p, pc);
            let full = self.neighbours[p].len() >= self.k;
            if !full || d <= self.kdist[p] {
                s_update_k.push(p);
            }
        }

        // ── Step 3: insert pc into each affected neighbour list, fix k-distance ─
        for &p in &s_update_k {
            self.insert_neighbour(p, pc);
        }

        // ── Step 4: lrd update set ──────────────────────────────────────────────
        // lrd(p) changes for every p in S_update_k (its own neighbourhood moved)
        // and for every reverse-neighbour of a point in S_update_k (its
        // reach_dist(·, p) depends on the changed k-distance(p)). Include pc.
        let mut lrd_dirty: Vec<bool> = vec![false; self.live.len()];
        lrd_dirty[pc] = true;
        for &p in &s_update_k {
            lrd_dirty[p] = true;
            // reverse-neighbours of p: their reach_dist to p uses kdist[p].
            let rev = self.rnn[p].clone();
            for r in rev {
                if r < lrd_dirty.len() {
                    lrd_dirty[r] = true;
                }
            }
        }

        let lrd_set: Vec<usize> = (0..self.live.len())
            .filter(|&i| self.live[i] && lrd_dirty[i])
            .collect();
        for &i in &lrd_set {
            self.lrd[i] = self.compute_lrd(i);
        }

        // ── Step 5: LOF update set ──────────────────────────────────────────────
        // LOF(i) changes if lrd(i) changed or any neighbour's lrd changed. So the
        // LOF set = lrd_set ∪ reverse-neighbours(lrd_set).
        let mut lof_dirty: Vec<bool> = vec![false; self.live.len()];
        for &i in &lrd_set {
            if i < lof_dirty.len() {
                lof_dirty[i] = true;
            }
            let rev = self.rnn[i].clone();
            for r in rev {
                if r < lof_dirty.len() {
                    lof_dirty[r] = true;
                }
            }
        }
        for i in (0..self.live.len()).filter(|&i| self.live[i] && lof_dirty[i]) {
            self.lof[i] = self.compute_lof(i);
        }

        Ok(self.lof[pc])
    }

    /// Insert candidate neighbour `cand` into `p`'s neighbour list, keeping it
    /// sorted and trimmed to `k`, and maintain the `rnn` back-links and
    /// `kdist[p]`. Does nothing if `cand` is already present.
    fn insert_neighbour(&mut self, p: usize, cand: usize) {
        if self.neighbours[p].contains(&cand) {
            return;
        }
        let d = self.dist(p, cand);

        // Find the sorted insertion position by (distance, index).
        let key = (d, cand);
        let pos = self.neighbours[p]
            .iter()
            .zip(self.neigh_dist[p].iter())
            .position(|(&idx, &dd)| cmp_dist_idx(&key, &(dd, idx)) == std::cmp::Ordering::Less)
            .unwrap_or(self.neighbours[p].len());

        self.neighbours[p].insert(pos, cand);
        self.neigh_dist[p].insert(pos, d);
        self.rnn[cand].push(p);

        // Trim to k, evicting the (now) farthest neighbour.
        while self.neighbours[p].len() > self.k {
            if let Some(evicted) = self.neighbours[p].pop() {
                self.neigh_dist[p].pop();
                self.rnn[evicted].retain(|&x| x != p);
            }
        }

        self.kdist[p] = self.neigh_dist[p].last().copied().unwrap_or(0.0);
    }

    // ── Deletion ────────────────────────────────────────────────────────────────

    /// Remove the point stored at slot `idx` (tombstone) and incrementally repair
    /// the kNN graph, k-distances, `lrd` and LOF of the affected neighbourhood.
    ///
    /// Indices of all other points are preserved. After removal,
    /// [`OnlineLof::score_of`] / [`OnlineLof::lof_scores`] report `NaN` for `idx`.
    ///
    /// # Errors
    /// Returns [`AnomalyError::Internal`] when `idx` is out of range or already
    /// removed.
    pub fn remove(&mut self, idx: usize) -> AnomalyResult<()> {
        if idx >= self.live.len() || !self.live[idx] {
            return Err(AnomalyError::Internal {
                msg: format!("remove: index {idx} is out of range or already removed"),
            });
        }

        // Points that currently hold `idx` as a neighbour lose it → their
        // neighbour list and k-distance must be rebuilt over the remaining set.
        let affected_holders: Vec<usize> = self.rnn[idx].clone();

        // Tombstone the slot first so it is excluded from neighbour searches.
        self.live[idx] = false;
        self.n_live -= 1;
        self.lof[idx] = f64::NAN;
        self.lrd[idx] = f64::NAN;
        self.kdist[idx] = f64::NAN;

        // Detach idx from its own neighbours' reverse lists and clear its state.
        let own = std::mem::take(&mut self.neighbours[idx]);
        for o in own {
            self.rnn[o].retain(|&x| x != idx);
        }
        self.neigh_dist[idx].clear();
        self.rnn[idx].clear();

        if self.n_live == 0 {
            return Ok(());
        }

        // Rebuild neighbour lists for every holder; this also refreshes their
        // k-distance and reverse links. k-distance can only grow here.
        for &p in &affected_holders {
            if self.live[p] {
                self.rebuild_neighbours(p);
            }
        }

        // lrd update set: the holders (own neighbourhood changed) plus the
        // reverse-neighbours of holders (their reach_dist to a holder uses the
        // holder's changed k-distance).
        let mut lrd_dirty: Vec<bool> = vec![false; self.live.len()];
        for &p in &affected_holders {
            if !self.live[p] {
                continue;
            }
            if p < lrd_dirty.len() {
                lrd_dirty[p] = true;
            }
            let rev = self.rnn[p].clone();
            for r in rev {
                if r < lrd_dirty.len() {
                    lrd_dirty[r] = true;
                }
            }
        }
        let lrd_set: Vec<usize> = (0..self.live.len())
            .filter(|&i| self.live[i] && lrd_dirty[i])
            .collect();
        for &i in &lrd_set {
            self.lrd[i] = self.compute_lrd(i);
        }

        // LOF update set: lrd_set plus reverse-neighbours of lrd_set.
        let mut lof_dirty: Vec<bool> = vec![false; self.live.len()];
        for &i in &lrd_set {
            if i < lof_dirty.len() {
                lof_dirty[i] = true;
            }
            let rev = self.rnn[i].clone();
            for r in rev {
                if r < lof_dirty.len() {
                    lof_dirty[r] = true;
                }
            }
        }
        for i in (0..self.live.len()).filter(|&i| self.live[i] && lof_dirty[i]) {
            self.lof[i] = self.compute_lof(i);
        }

        Ok(())
    }

    // ── Brute-force reference (tests only) ──────────────────────────────────────

    /// Full from-scratch LOF over the current live set, using the identical
    /// `(distance, index)` neighbour selection as the incremental path.
    ///
    /// Returns LOF aligned with live slots; tombstoned slots map to `f64::NAN`.
    /// Used exclusively by the test-suite to assert incremental ≡ batch.
    #[cfg(test)]
    fn brute_force_lof(&self) -> Vec<f64> {
        let n = self.live.len();
        let mut out = vec![f64::NAN; n];
        if self.n_live == 0 {
            return out;
        }

        // Neighbour lists + k-distances.
        let mut nbrs: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut nbr_d: Vec<Vec<f64>> = vec![Vec::new(); n];
        let mut kd = vec![0.0_f64; n];
        let kk = self.k.min(self.n_live.saturating_sub(1));
        for i in 0..n {
            if !self.live[i] {
                continue;
            }
            let (idx, dst) = self.knn_live(self.point(i), kk, Some(i));
            kd[i] = dst.last().copied().unwrap_or(0.0);
            nbrs[i] = idx;
            nbr_d[i] = dst;
        }

        // lrd.
        let mut lrd = vec![0.0_f64; n];
        for i in 0..n {
            if !self.live[i] {
                continue;
            }
            if nbrs[i].is_empty() {
                lrd[i] = 0.0;
                continue;
            }
            let mut sum_reach = 0.0_f64;
            for (slot, &o) in nbrs[i].iter().enumerate() {
                let reach = kd[o].max(nbr_d[i][slot]).max(0.0);
                sum_reach += reach;
            }
            lrd[i] = if sum_reach < 1e-15 {
                f64::INFINITY
            } else {
                nbrs[i].len() as f64 / sum_reach
            };
        }

        // LOF.
        for i in 0..n {
            if !self.live[i] {
                continue;
            }
            if nbrs[i].is_empty() {
                out[i] = 1.0;
                continue;
            }
            let lrd_i = lrd[i];
            if lrd_i.is_infinite() || lrd_i <= 0.0 {
                out[i] = 1.0;
                continue;
            }
            let acc: f64 = nbrs[i].iter().map(|&o| lrd[o] / lrd_i).sum();
            out[i] = acc / nbrs[i].len() as f64;
        }
        out
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert two LOF vectors agree on every live slot.
    fn assert_lof_close(got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            if w.is_nan() {
                assert!(g.is_nan(), "slot {i}: expected NaN tombstone, got {g}");
                continue;
            }
            assert!(g.is_finite(), "slot {i}: incremental LOF not finite: {g}");
            assert!(
                (g - w).abs() <= tol,
                "slot {i}: incremental {g} vs batch {w} (|Δ|={})",
                (g - w).abs()
            );
        }
    }

    // ── Key test: sequential insertion ≡ full batch LOF ─────────────────────────
    #[test]
    fn incremental_equals_batch_after_each_insert() {
        // A 2-D dataset with two clusters and a clear outlier, inserted one by one.
        let pts: Vec<[f64; 2]> = vec![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [0.05, 0.05],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
            [5.1, 5.1],
            [5.05, 5.05],
            [50.0, 50.0], // outlier
            [0.2, 0.05],
            [4.9, 5.0],
        ];
        let mut model = OnlineLof::new(3, 2).expect("construct online LOF");
        for (step, p) in pts.iter().enumerate() {
            let _lof_pc = model.insert(p).expect("insert should succeed");
            // After every insertion the whole live graph must match a from-scratch
            // batch recompute to floating-point tolerance.
            let batch = model.brute_force_lof();
            assert_lof_close(model.lof_scores(), &batch, 1e-9);
            assert_eq!(model.n_live(), step + 1);
        }
    }

    // ── Outlier scores ≫ 1, dense-cluster points ≈ 1 ────────────────────────────
    #[test]
    fn outlier_high_cluster_near_one() {
        let mut model = OnlineLof::new(4, 2).expect("construct");
        // Dense cluster around the origin.
        let mut idx_cluster = Vec::new();
        for i in 0..15_usize {
            let x = (i % 4) as f64 * 0.01;
            let y = (i / 4) as f64 * 0.01;
            model.insert(&[x, y]).expect("insert cluster point");
            idx_cluster.push(model.len() - 1);
        }
        // One blatant outlier far away.
        let out_lof = model.insert(&[100.0, 100.0]).expect("insert outlier");
        let out_idx = model.len() - 1;

        assert!(
            out_lof > 2.0,
            "outlier LOF should be well above 1, got {out_lof}"
        );
        // Interior cluster points should be ≈ 1.
        for &c in &idx_cluster {
            let s = model.score_of(c).expect("cluster score");
            assert!(s < 1.5, "cluster point {c} should have LOF ≈ 1, got {s}");
        }
        // And the outlier is the global maximum.
        let scores = model.lof_scores();
        let max_idx = scores
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite comparison"))
            .map(|(i, _)| i)
            .expect("at least one finite score");
        assert_eq!(max_idx, out_idx, "outlier should hold the max LOF");
    }

    // ── k clamped to available points ───────────────────────────────────────────
    #[test]
    fn k_clamped_to_available_points() {
        // k = 10 but we only ever have a handful of points.
        let mut model = OnlineLof::new(10, 1).expect("construct");
        let first = model.insert(&[0.0]).expect("first insert");
        assert_eq!(first, 1.0, "first ever point: LOF defaults to 1.0");
        model.insert(&[1.0]).expect("second insert");
        model.insert(&[2.0]).expect("third insert");
        model.insert(&[3.0]).expect("fourth insert");
        // Even with k > n−1, scores stay finite and match the clamped batch.
        let batch = model.brute_force_lof();
        assert_lof_close(model.lof_scores(), &batch, 1e-9);
        for s in model.lof_scores() {
            assert!(s.is_finite(), "clamped-k score not finite: {s}");
        }
    }

    // ── Single-point model ──────────────────────────────────────────────────────
    #[test]
    fn single_point_has_unit_lof() {
        let mut model = OnlineLof::new(3, 2).expect("construct");
        let s = model.insert(&[1.0, 2.0]).expect("insert");
        assert_eq!(s, 1.0);
        assert_eq!(model.n_live(), 1);
        assert_eq!(model.score_of(0).expect("score"), 1.0);
    }

    // ── Error path: k = 0 ───────────────────────────────────────────────────────
    #[test]
    fn error_k_zero() {
        let res = OnlineLof::new(0, 2);
        assert!(matches!(res, Err(AnomalyError::InvalidK { k: 0 })));
    }

    // ── Error path: dim = 0 ─────────────────────────────────────────────────────
    #[test]
    fn error_dim_zero() {
        let res = OnlineLof::new(3, 0);
        assert!(matches!(
            res,
            Err(AnomalyError::InvalidFeatureCount { n: 0 })
        ));
    }

    // ── Error path: dimension mismatch on insert ────────────────────────────────
    #[test]
    fn error_dim_mismatch_on_insert() {
        let mut model = OnlineLof::new(3, 2).expect("construct");
        let res = model.insert(&[1.0, 2.0, 3.0]); // 3-D into a 2-D model
        assert!(matches!(
            res,
            Err(AnomalyError::DimensionMismatch {
                expected: 2,
                got: 3
            })
        ));
    }

    // ── Error path: score_of out-of-range / removed ─────────────────────────────
    #[test]
    fn error_score_of_invalid_index() {
        let mut model = OnlineLof::new(2, 1).expect("construct");
        model.insert(&[0.0]).expect("insert a");
        model.insert(&[1.0]).expect("insert b");
        model.insert(&[2.0]).expect("insert c");
        assert!(model.score_of(99).is_err(), "out-of-range index errors");
        model.remove(1).expect("remove b");
        assert!(model.score_of(1).is_err(), "removed index errors");
    }

    // ── Deletion repairs the graph: incremental ≡ batch afterwards ──────────────
    #[test]
    fn remove_then_matches_batch() {
        let mut model = OnlineLof::new(3, 2).expect("construct");
        let pts: Vec<[f64; 2]> = vec![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [0.05, 0.05],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
            [40.0, 40.0],
            [0.2, 0.2],
        ];
        for p in &pts {
            model.insert(p).expect("insert");
        }
        // Remove the far outlier, then a dense-cluster member.
        model.remove(8).expect("remove outlier");
        let batch = model.brute_force_lof();
        assert_lof_close(model.lof_scores(), &batch, 1e-9);

        model.remove(4).expect("remove cluster member");
        let batch2 = model.brute_force_lof();
        assert_lof_close(model.lof_scores(), &batch2, 1e-9);
        assert_eq!(model.n_live(), pts.len() - 2);
    }

    // ── Interleaved insert / remove keeps the graph consistent ──────────────────
    #[test]
    fn interleaved_insert_remove_consistent() {
        let mut model = OnlineLof::new(4, 2).expect("construct");
        for i in 0..20_usize {
            let x = (i as f64).sin();
            let y = (i as f64 * 0.7).cos();
            model.insert(&[x, y]).expect("insert");
            if i.is_multiple_of(5) && i > 0 {
                // Remove an early, still-live slot.
                let target = i / 5;
                if target < model.len() && model.score_of(target).is_ok() {
                    model.remove(target).expect("remove");
                }
            }
            let batch = model.brute_force_lof();
            assert_lof_close(model.lof_scores(), &batch, 1e-9);
        }
    }

    // ── Returned LOF from insert equals the stored score ────────────────────────
    #[test]
    fn insert_returns_stored_score() {
        let mut model = OnlineLof::new(3, 2).expect("construct");
        let mut last_idx = 0;
        let mut last_returned = 1.0;
        for i in 0..12_usize {
            let p = [i as f64 * 0.3, (i as f64 * 0.3).fract()];
            last_returned = model.insert(&p).expect("insert");
            last_idx = model.len() - 1;
        }
        let stored = model.score_of(last_idx).expect("score");
        assert!(
            (stored - last_returned).abs() <= 1e-12,
            "insert returned {last_returned} but stored {stored}"
        );
    }

    // ── 1-D stream with a clear gap: the isolated point scores highest ──────────
    #[test]
    fn one_dim_gap_outlier() {
        let mut model = OnlineLof::new(3, 1).expect("construct");
        for i in 0..20_usize {
            model.insert(&[i as f64 * 0.1]).expect("dense insert");
        }
        let gap_lof = model.insert(&[100.0]).expect("gap insert");
        assert!(
            gap_lof > 2.0,
            "gapped point LOF should be high, got {gap_lof}"
        );
        let batch = model.brute_force_lof();
        assert_lof_close(model.lof_scores(), &batch, 1e-9);
    }
}
