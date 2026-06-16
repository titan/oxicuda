//! GraphRec — Graph Neural Network for Social Recommendation (item-aggregation
//! core, opinion-aware dual aggregation).
//!
//! Reference: Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang,
//! Dawei Yin, "Graph Neural Networks for Social Recommendation", WWW 2019.
//!
//! # Architecture
//!
//! GraphRec models a user and an item with two symmetric *aggregation* paths
//! that fuse interactions with the **opinion** (rating) attached to each edge:
//!
//! * **User modelling — item-space aggregation.** A user `u` is summarised
//!   from the items they interacted with. Each interacted item `i_t` (with
//!   rating `r_t`) is turned into an *opinion-aware* representation
//!   `x_t = W_v · [q_{i_t} ‖ e_{r_t}] + b_v`, where `q` is the item-embedding
//!   table and `e` the rating/opinion table. The contributions are combined
//!   with an **attention** distribution `α_t = softmax_t( (x_t · p_u)/√d )`
//!   that scores each opinion-aware item against the user's base embedding
//!   `p_u`. The item-space user factor is the residual aggregate
//!   `h_u = p_u + Σ_t α_t · x_t`.
//! * **Item modelling — user-space aggregation.** Symmetrically, an item `i`
//!   is summarised from the users who rated it. Each rater `u_s` (rating
//!   `r_s`) becomes `f_s = W_u · [p_{u_s} ‖ e_{r_s}] + b_u`, attention
//!   `β_s = softmax_s( (f_s · q_i)/√d )`, and the user-space item factor is
//!   `z_i = q_i + Σ_s β_s · f_s`.
//!
//! The predicted rating is the inner product of the two factors plus a global
//! bias `r̂_{ui} = h_u · z_i + b_0`. A user with **no** interactions falls back
//! to its base embedding (`h_u = p_u`); likewise an item with no raters uses
//! `z_i = q_i`.
//!
//! # Training
//!
//! [`GraphRec::train_step`] performs one full-batch gradient-descent step on
//! the mean-squared rating error over a list of `(user, item, rating)`
//! triples, with **manually-derived** gradients flowing through the dual
//! aggregation, the two opinion-fusion linear maps, the embedding tables and
//! the global bias. The attention coefficients `α`, `β` are treated as
//! constants during back-propagation (a stop-gradient on the softmax weights);
//! this is the standard "attention-as-gating" simplification and keeps the
//! aggregation linear in the value vectors so the gradient is exact for every
//! updated parameter.

use crate::error::{RecsysError, RecsysResult};
use crate::handle::LcgRng;

/// Per-node adjacency: for each node, a list of `(neighbour_id, rating)` pairs.
type Adjacency = Vec<Vec<(usize, usize)>>;

/// Inner product of two equal-length slices (zips, so it never indexes out of
/// bounds).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Numerically-stable soft-max returning a fresh vector. An empty input maps
/// to an empty output.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let inv = 1.0 / (sum + 1e-10);
    for e in exps.iter_mut() {
        *e *= inv;
    }
    exps
}

/// Build per-user and per-item adjacency lists `(neighbour_id, rating)` from a
/// triple stream. Out-of-range ids are skipped (callers validate upstream).
fn build_adjacency(
    triples: &[(usize, usize, usize)],
    n_users: usize,
    n_items: usize,
) -> (Adjacency, Adjacency) {
    let mut user_items: Adjacency = vec![Vec::new(); n_users];
    let mut item_users: Adjacency = vec![Vec::new(); n_items];
    for &(u, i, r) in triples {
        if u < n_users && i < n_items {
            user_items[u].push((i, r));
            item_users[i].push((u, r));
        }
    }
    (user_items, item_users)
}

/// Result of one aggregation path. `cs` holds, per neighbour, the
/// `[left ‖ rating]` concatenation (length `2·dim`) reused by the backward
/// pass.
struct AggResult {
    /// Aggregated latent factor (length `dim`).
    h: Vec<f32>,
    /// Soft-max attention weights over the neighbours (sum ≈ 1, empty when
    /// there are no neighbours).
    alpha: Vec<f32>,
    /// Per-neighbour `[left ‖ rating]` concatenations (each length `2·dim`).
    cs: Vec<Vec<f32>>,
}

/// GraphRec opinion-aware social recommender.
pub struct GraphRec {
    /// Number of users.
    pub n_users: usize,
    /// Number of items.
    pub n_items: usize,
    /// Number of distinct rating levels (opinion vocabulary).
    pub n_ratings: usize,
    /// Embedding / factor width.
    pub dim: usize,
    /// User base-embedding table `p`: `n_users × dim` (row-major).
    pub user_emb: Vec<f32>,
    /// Item base-embedding table `q`: `n_items × dim` (row-major).
    pub item_emb: Vec<f32>,
    /// Rating / opinion embedding table `e`: `n_ratings × dim` (row-major).
    pub rating_emb: Vec<f32>,
    /// Item-side opinion fusion `W_v`: `dim × (2·dim)` (row-major).
    pub w_item_fuse: Vec<f32>,
    /// Item-side opinion fusion bias `b_v`: `dim`.
    pub b_item_fuse: Vec<f32>,
    /// User-side opinion fusion `W_u`: `dim × (2·dim)` (row-major).
    pub w_user_fuse: Vec<f32>,
    /// User-side opinion fusion bias `b_u`: `dim`.
    pub b_user_fuse: Vec<f32>,
    /// Global rating bias `b_0`.
    pub b_global: f32,
    /// Stored per-user adjacency `(item, rating)` (set by
    /// [`GraphRec::set_interactions`] / [`GraphRec::fit`]).
    user_items: Adjacency,
    /// Stored per-item adjacency `(user, rating)`.
    item_users: Adjacency,
    /// Whether interactions have been registered (gates [`GraphRec::score`]).
    fitted: bool,
}

impl GraphRec {
    /// Construct a GraphRec with Kaiming-style normal initialisation.
    ///
    /// # Errors
    /// - [`RecsysError::InvalidNumUsers`] when `n_users == 0`.
    /// - [`RecsysError::InvalidNumItems`] when `n_items == 0`.
    /// - [`RecsysError::InvalidConfig`] when `n_ratings == 0`.
    /// - [`RecsysError::InvalidEmbeddingDim`] when `dim == 0`.
    pub fn new(
        n_users: usize,
        n_items: usize,
        n_ratings: usize,
        dim: usize,
        rng: &mut LcgRng,
    ) -> RecsysResult<Self> {
        if n_users == 0 {
            return Err(RecsysError::InvalidNumUsers { n: n_users });
        }
        if n_items == 0 {
            return Err(RecsysError::InvalidNumItems { n: n_items });
        }
        if n_ratings == 0 {
            return Err(RecsysError::InvalidConfig {
                msg: "n_ratings must be >= 1".into(),
            });
        }
        if dim == 0 {
            return Err(RecsysError::InvalidEmbeddingDim { d: dim });
        }

        let emb_scale = (1.0 / dim as f32).sqrt();
        let fuse_scale = (1.0 / (2 * dim) as f32).sqrt();
        let user_emb: Vec<f32> = (0..n_users * dim)
            .map(|_| rng.next_normal() * emb_scale)
            .collect();
        let item_emb: Vec<f32> = (0..n_items * dim)
            .map(|_| rng.next_normal() * emb_scale)
            .collect();
        let rating_emb: Vec<f32> = (0..n_ratings * dim)
            .map(|_| rng.next_normal() * emb_scale)
            .collect();
        let w_item_fuse: Vec<f32> = (0..dim * 2 * dim)
            .map(|_| rng.next_normal() * fuse_scale)
            .collect();
        let w_user_fuse: Vec<f32> = (0..dim * 2 * dim)
            .map(|_| rng.next_normal() * fuse_scale)
            .collect();

        Ok(Self {
            n_users,
            n_items,
            n_ratings,
            dim,
            user_emb,
            item_emb,
            rating_emb,
            w_item_fuse,
            b_item_fuse: vec![0.0_f32; dim],
            w_user_fuse,
            b_user_fuse: vec![0.0_f32; dim],
            b_global: 0.0,
            user_items: vec![Vec::new(); n_users],
            item_users: vec![Vec::new(); n_items],
            fitted: false,
        })
    }

    fn check_user(&self, u: usize) -> RecsysResult<()> {
        if u >= self.n_users {
            return Err(RecsysError::UnknownUser { id: u });
        }
        Ok(())
    }

    fn check_item(&self, i: usize) -> RecsysResult<()> {
        if i >= self.n_items {
            return Err(RecsysError::UnknownItem { id: i });
        }
        Ok(())
    }

    fn check_rating(&self, r: usize) -> RecsysResult<()> {
        if r >= self.n_ratings {
            return Err(RecsysError::ItemOutOfBounds {
                idx: r,
                n: self.n_ratings,
            });
        }
        Ok(())
    }

    /// One opinion-aware aggregation path. `base` is the node's own embedding
    /// (used both as the attention query and as the residual). `neighbours`
    /// lists `(left_id, rating)` where `left_emb` is the table the left part
    /// of each opinion concatenation is drawn from, fused by `w_fuse`/`b_fuse`.
    fn aggregate(
        &self,
        base: &[f32],
        neighbours: &[(usize, usize)],
        left_emb: &[f32],
        w_fuse: &[f32],
        b_fuse: &[f32],
    ) -> RecsysResult<AggResult> {
        let d = self.dim;
        if neighbours.is_empty() {
            return Ok(AggResult {
                h: base.to_vec(),
                alpha: Vec::new(),
                cs: Vec::new(),
            });
        }

        let inv_sqrt = 1.0 / (d as f32).sqrt();
        let mut xs: Vec<Vec<f32>> = Vec::with_capacity(neighbours.len());
        let mut cs: Vec<Vec<f32>> = Vec::with_capacity(neighbours.len());
        let mut logits: Vec<f32> = Vec::with_capacity(neighbours.len());

        for &(lid, r) in neighbours {
            let left = left_emb
                .get(lid * d..(lid + 1) * d)
                .ok_or(RecsysError::Internal {
                    msg: "aggregate: left id out of range".into(),
                })?;
            let rate = self
                .rating_emb
                .get(r * d..(r + 1) * d)
                .ok_or(RecsysError::Internal {
                    msg: "aggregate: rating id out of range".into(),
                })?;
            let mut c = Vec::with_capacity(2 * d);
            c.extend_from_slice(left);
            c.extend_from_slice(rate);

            // x = W_fuse · c + b_fuse
            let mut x = b_fuse.to_vec();
            for (o, xo) in x.iter_mut().enumerate() {
                let row = w_fuse
                    .get(o * 2 * d..(o + 1) * 2 * d)
                    .ok_or(RecsysError::Internal {
                        msg: "aggregate: fuse row out of range".into(),
                    })?;
                *xo += dot(row, &c);
            }
            logits.push(dot(&x, base) * inv_sqrt);
            xs.push(x);
            cs.push(c);
        }

        let alpha = softmax(&logits);
        let mut h = base.to_vec();
        for (a, x) in alpha.iter().zip(xs.iter()) {
            for (hv, &xi) in h.iter_mut().zip(x.iter()) {
                *hv += a * xi;
            }
        }
        Ok(AggResult { h, alpha, cs })
    }

    /// Validate explicit neighbour slices and assemble `(id, rating)` pairs.
    fn neighbours_from(
        &self,
        ids: &[usize],
        ratings: &[usize],
        id_is_item: bool,
    ) -> RecsysResult<Vec<(usize, usize)>> {
        if ids.len() != ratings.len() {
            return Err(RecsysError::DimensionMismatch {
                expected: ids.len(),
                got: ratings.len(),
            });
        }
        for (&id, &r) in ids.iter().zip(ratings.iter()) {
            if id_is_item {
                self.check_item(id)?;
            } else {
                self.check_user(id)?;
            }
            self.check_rating(r)?;
        }
        Ok(ids.iter().copied().zip(ratings.iter().copied()).collect())
    }

    /// Item-space **user factor** `h_u` from a user's interacted items and the
    /// ratings attached to them. With an empty interaction list it falls back
    /// to the base user embedding `p_u`.
    ///
    /// # Errors
    /// - [`RecsysError::UnknownUser`] when `u >= n_users`.
    /// - [`RecsysError::UnknownItem`] when any interacted id `>= n_items`.
    /// - [`RecsysError::ItemOutOfBounds`] when any rating `>= n_ratings`.
    /// - [`RecsysError::DimensionMismatch`] when the two slices differ in
    ///   length.
    pub fn user_factor(
        &self,
        u: usize,
        interacted_items: &[usize],
        ratings: &[usize],
    ) -> RecsysResult<Vec<f32>> {
        self.check_user(u)?;
        let d = self.dim;
        let base = self
            .user_emb
            .get(u * d..(u + 1) * d)
            .ok_or(RecsysError::UnknownUser { id: u })?;
        let neighbours = self.neighbours_from(interacted_items, ratings, true)?;
        Ok(self
            .aggregate(
                base,
                &neighbours,
                &self.item_emb,
                &self.w_item_fuse,
                &self.b_item_fuse,
            )?
            .h)
    }

    /// User-space **item factor** `z_i` from the users who rated an item.
    /// Falls back to the base item embedding `q_i` when the rater list is
    /// empty.
    ///
    /// # Errors
    /// Mirrors [`GraphRec::user_factor`] with user/item roles swapped.
    pub fn item_factor(
        &self,
        i: usize,
        rater_users: &[usize],
        ratings: &[usize],
    ) -> RecsysResult<Vec<f32>> {
        self.check_item(i)?;
        let d = self.dim;
        let base = self
            .item_emb
            .get(i * d..(i + 1) * d)
            .ok_or(RecsysError::UnknownItem { id: i })?;
        let neighbours = self.neighbours_from(rater_users, ratings, false)?;
        Ok(self
            .aggregate(
                base,
                &neighbours,
                &self.user_emb,
                &self.w_user_fuse,
                &self.b_user_fuse,
            )?
            .h)
    }

    /// Soft-max attention weights a user assigns to each interacted item
    /// during item-space aggregation. The returned vector sums to ≈ 1 (and is
    /// empty for an empty interaction list).
    ///
    /// # Errors
    /// Mirrors [`GraphRec::user_factor`].
    pub fn attention_weights(
        &self,
        u: usize,
        interacted_items: &[usize],
        ratings: &[usize],
    ) -> RecsysResult<Vec<f32>> {
        self.check_user(u)?;
        let d = self.dim;
        let base = self
            .user_emb
            .get(u * d..(u + 1) * d)
            .ok_or(RecsysError::UnknownUser { id: u })?;
        let neighbours = self.neighbours_from(interacted_items, ratings, true)?;
        Ok(self
            .aggregate(
                base,
                &neighbours,
                &self.item_emb,
                &self.w_item_fuse,
                &self.b_item_fuse,
            )?
            .alpha)
    }

    /// Predict `r̂_{ui}` from explicit user/item neighbourhoods.
    ///
    /// # Errors
    /// Validation as in [`GraphRec::user_factor`] / [`GraphRec::item_factor`].
    pub fn score_with(
        &self,
        u: usize,
        i: usize,
        user_items: &[usize],
        user_ratings: &[usize],
        item_users: &[usize],
        item_ratings: &[usize],
    ) -> RecsysResult<f32> {
        let h = self.user_factor(u, user_items, user_ratings)?;
        let z = self.item_factor(i, item_users, item_ratings)?;
        Ok(dot(&h, &z) + self.b_global)
    }

    /// Predict `r̂_{ui}` using the interactions registered via
    /// [`GraphRec::set_interactions`] / [`GraphRec::fit`].
    ///
    /// # Errors
    /// - [`RecsysError::NotFitted`] when no interactions were registered.
    /// - [`RecsysError::UnknownUser`] / [`RecsysError::UnknownItem`] for bad ids.
    pub fn score(&self, u: usize, i: usize) -> RecsysResult<f32> {
        if !self.fitted {
            return Err(RecsysError::NotFitted);
        }
        self.check_user(u)?;
        self.check_item(i)?;
        let d = self.dim;
        let p_u = self
            .user_emb
            .get(u * d..(u + 1) * d)
            .ok_or(RecsysError::UnknownUser { id: u })?;
        let q_i = self
            .item_emb
            .get(i * d..(i + 1) * d)
            .ok_or(RecsysError::UnknownItem { id: i })?;
        let nu = self
            .user_items
            .get(u)
            .ok_or(RecsysError::UnknownUser { id: u })?;
        let ni = self
            .item_users
            .get(i)
            .ok_or(RecsysError::UnknownItem { id: i })?;
        let h = self
            .aggregate(
                p_u,
                nu,
                &self.item_emb,
                &self.w_item_fuse,
                &self.b_item_fuse,
            )?
            .h;
        let z = self
            .aggregate(
                q_i,
                ni,
                &self.user_emb,
                &self.w_user_fuse,
                &self.b_user_fuse,
            )?
            .h;
        Ok(dot(&h, &z) + self.b_global)
    }

    /// Register the `(user, item, rating)` interactions used by
    /// [`GraphRec::score`].
    ///
    /// # Errors
    /// - [`RecsysError::UnknownUser`] / [`RecsysError::UnknownItem`] /
    ///   [`RecsysError::ItemOutOfBounds`] for out-of-range entries.
    pub fn set_interactions(&mut self, triples: &[(usize, usize, usize)]) -> RecsysResult<()> {
        for &(u, i, r) in triples {
            self.check_user(u)?;
            self.check_item(i)?;
            self.check_rating(r)?;
        }
        let (user_items, item_users) = build_adjacency(triples, self.n_users, self.n_items);
        self.user_items = user_items;
        self.item_users = item_users;
        self.fitted = true;
        Ok(())
    }

    /// One full-batch gradient-descent step on the mean-squared rating error
    /// over `triples`. Returns the **pre-update** mean-squared error so a fit
    /// loop can verify the loss decreases. An empty batch is a no-op (returns
    /// `0.0`).
    ///
    /// # Errors
    /// - [`RecsysError::UnknownUser`] / [`RecsysError::UnknownItem`] /
    ///   [`RecsysError::ItemOutOfBounds`] for out-of-range triples.
    pub fn train_step(&mut self, triples: &[(usize, usize, usize)], lr: f32) -> RecsysResult<f32> {
        if triples.is_empty() {
            return Ok(0.0);
        }
        let d = self.dim;
        for &(u, i, r) in triples {
            self.check_user(u)?;
            self.check_item(i)?;
            self.check_rating(r)?;
        }
        let (user_items, item_users) = build_adjacency(triples, self.n_users, self.n_items);

        let mut g_user = vec![0.0_f32; self.n_users * d];
        let mut g_item = vec![0.0_f32; self.n_items * d];
        let mut g_rating = vec![0.0_f32; self.n_ratings * d];
        let mut g_wi = vec![0.0_f32; d * 2 * d];
        let mut g_bi = vec![0.0_f32; d];
        let mut g_wu = vec![0.0_f32; d * 2 * d];
        let mut g_bu = vec![0.0_f32; d];
        let mut g_bg = 0.0_f32;
        let mut total_se = 0.0_f32;

        for &(u, i, r) in triples {
            let p_u = self.user_emb[u * d..(u + 1) * d].to_vec();
            let q_i = self.item_emb[i * d..(i + 1) * d].to_vec();
            let nu = &user_items[u];
            let ni = &item_users[i];
            let au = self.aggregate(
                &p_u,
                nu,
                &self.item_emb,
                &self.w_item_fuse,
                &self.b_item_fuse,
            )?;
            let ai = self.aggregate(
                &q_i,
                ni,
                &self.user_emb,
                &self.w_user_fuse,
                &self.b_user_fuse,
            )?;
            let h = &au.h;
            let z = &ai.h;

            let pred = dot(h, z) + self.b_global;
            let diff = pred - r as f32;
            total_se += diff * diff;
            let delta = 2.0 * diff;
            g_bg += delta;

            // user side: h = p_u + Σ_t α_t x_t  (x_t = W_v c_t + b_v).
            for k in 0..d {
                g_user[u * d + k] += delta * z[k];
            }
            for (t, &(lid, rt)) in nu.iter().enumerate() {
                let a_t = au.alpha[t];
                let c_t = &au.cs[t];
                for (o, (&zo, gbi)) in z.iter().zip(g_bi.iter_mut()).enumerate() {
                    let gxo = a_t * delta * zo;
                    *gbi += gxo;
                    let base = o * 2 * d;
                    for (k, &ck) in c_t.iter().enumerate() {
                        g_wi[base + k] += gxo * ck;
                    }
                }
                for k in 0..2 * d {
                    let mut acc = 0.0_f32;
                    for (o, &zo) in z.iter().enumerate() {
                        acc += self.w_item_fuse[o * 2 * d + k] * (a_t * delta * zo);
                    }
                    if k < d {
                        g_item[lid * d + k] += acc;
                    } else {
                        g_rating[rt * d + (k - d)] += acc;
                    }
                }
            }

            // item side: z = q_i + Σ_s β_s f_s  (f_s = W_u c_s + b_u).
            for k in 0..d {
                g_item[i * d + k] += delta * h[k];
            }
            for (s, &(lid, rs)) in ni.iter().enumerate() {
                let b_s = ai.alpha[s];
                let c_s = &ai.cs[s];
                for (o, (&ho, gbu)) in h.iter().zip(g_bu.iter_mut()).enumerate() {
                    let gfo = b_s * delta * ho;
                    *gbu += gfo;
                    let base = o * 2 * d;
                    for (k, &ck) in c_s.iter().enumerate() {
                        g_wu[base + k] += gfo * ck;
                    }
                }
                for k in 0..2 * d {
                    let mut acc = 0.0_f32;
                    for (o, &ho) in h.iter().enumerate() {
                        acc += self.w_user_fuse[o * 2 * d + k] * (b_s * delta * ho);
                    }
                    if k < d {
                        g_user[lid * d + k] += acc;
                    } else {
                        g_rating[rs * d + (k - d)] += acc;
                    }
                }
            }
        }

        let step = lr / triples.len() as f32;
        apply_grad(&mut self.user_emb, &g_user, step);
        apply_grad(&mut self.item_emb, &g_item, step);
        apply_grad(&mut self.rating_emb, &g_rating, step);
        apply_grad(&mut self.w_item_fuse, &g_wi, step);
        apply_grad(&mut self.b_item_fuse, &g_bi, step);
        apply_grad(&mut self.w_user_fuse, &g_wu, step);
        apply_grad(&mut self.b_user_fuse, &g_bu, step);
        self.b_global -= step * g_bg;

        Ok(total_se / triples.len() as f32)
    }

    /// Register `triples` then run `epochs` full-batch [`GraphRec::train_step`]
    /// updates with learning rate `lr`. Returns the per-epoch (pre-update) MSE
    /// history; on a well-posed problem the sequence decreases.
    ///
    /// # Errors
    /// - [`RecsysError::EmptyInteraction`] when `triples` is empty.
    /// - Validation errors propagated from [`GraphRec::train_step`].
    pub fn fit(
        &mut self,
        triples: &[(usize, usize, usize)],
        epochs: usize,
        lr: f32,
    ) -> RecsysResult<Vec<f32>> {
        if triples.is_empty() {
            return Err(RecsysError::EmptyInteraction);
        }
        self.set_interactions(triples)?;
        let mut history = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            history.push(self.train_step(triples, lr)?);
        }
        Ok(history)
    }

    /// Total number of learnable parameters.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.user_emb.len()
            + self.item_emb.len()
            + self.rating_emb.len()
            + self.w_item_fuse.len()
            + self.b_item_fuse.len()
            + self.w_user_fuse.len()
            + self.b_user_fuse.len()
            + 1
    }
}

/// In-place SGD update `param -= step · grad`.
fn apply_grad(param: &mut [f32], grad: &[f32], step: f32) {
    for (w, &g) in param.iter_mut().zip(grad.iter()) {
        *w -= step * g;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    #[test]
    fn factor_shapes_and_finite() {
        let mut rng = make_rng();
        let model = GraphRec::new(4, 5, 6, 4, &mut rng).expect("new should succeed");
        let h = model
            .user_factor(0, &[0, 1, 2], &[5, 4, 3])
            .expect("user_factor should succeed");
        let z = model
            .item_factor(1, &[0, 2], &[5, 4])
            .expect("item_factor should succeed");
        assert_eq!(h.len(), 4);
        assert_eq!(z.len(), 4);
        assert!(h.iter().all(|v| v.is_finite()));
        assert!(z.iter().all(|v| v.is_finite()));
        let s = model
            .score_with(0, 1, &[0, 1, 2], &[5, 4, 3], &[0, 2], &[5, 4])
            .expect("value should be present");
        assert!(s.is_finite(), "score must be finite, got {s}");
    }

    #[test]
    fn attention_sums_to_one() {
        let mut rng = make_rng();
        let model = GraphRec::new(3, 5, 6, 4, &mut rng).expect("new should succeed");
        let alpha = model
            .attention_weights(0, &[0, 1, 2, 3], &[5, 4, 3, 2])
            .expect("value should be present");
        assert_eq!(alpha.len(), 4);
        let sum: f32 = alpha.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "attention must sum to 1, got {sum}"
        );
        assert!(alpha.iter().all(|&a| (0.0..=1.0).contains(&a)));
    }

    #[test]
    fn similar_item_scored_higher_and_mse_decreases() {
        // Construct controlled embeddings: items 0,1 (user 0's interactions)
        // and item 2 share the +x direction; item 3 is the opposite. The
        // item-side fusion is pinned to identity-on-item / zero-on-rating and
        // the rating table is zeroed so the opinion-aware item rep equals the
        // raw item embedding. The user-space item factor of an unseen item
        // equals its base embedding (no raters), so the structural ordering is
        // analytic.
        let mut rng = LcgRng::new(7);
        let mut model = GraphRec::new(1, 4, 6, 2, &mut rng).expect("new should succeed");
        model.item_emb = vec![
            1.0, 0.0, // item 0 (+x)
            0.9, 0.1, // item 1 (+x)
            1.0, 0.0, // item 2 (+x, unseen, similar)
            -1.0, 0.0, // item 3 (-x, unseen, dissimilar)
        ];
        model.user_emb = vec![1.0, 0.0]; // user 0 base query along +x
        model.rating_emb = vec![0.0_f32; 6 * 2];
        // W_v = [I | 0]  →  x_t = q_{i_t}
        model.w_item_fuse = vec![
            1.0, 0.0, 0.0, 0.0, // out 0: pick item dim 0
            0.0, 1.0, 0.0, 0.0, // out 1: pick item dim 1
        ];
        model.b_item_fuse = vec![0.0, 0.0];
        model.b_user_fuse = vec![0.0, 0.0];
        model.b_global = 0.0;
        model
            .set_interactions(&[(0, 0, 5), (0, 1, 5)])
            .expect("value should be present");

        let s_similar = model.score(0, 2).expect("score should succeed");
        let s_dissimilar = model.score(0, 3).expect("score should succeed");
        assert!(
            s_similar > s_dissimilar,
            "similar item ({s_similar}) should outscore dissimilar item ({s_dissimilar})"
        );

        // Train on a fresh random model and confirm MSE decreases.
        let mut rng2 = LcgRng::new(11);
        let mut trainable = GraphRec::new(2, 4, 6, 4, &mut rng2).expect("new should succeed");
        let triples = vec![(0, 0, 5), (0, 1, 4), (1, 2, 1), (1, 3, 2), (0, 2, 5)];
        let history = trainable
            .fit(&triples, 40, 0.05)
            .expect("fit should succeed");
        let first = history.first().copied().expect("copied should succeed");
        let last = history.last().copied().expect("copied should succeed");
        assert!(
            last < first,
            "train MSE should decrease: first {first}, last {last}"
        );
    }

    #[test]
    fn empty_interactions_fall_back_to_base() {
        let mut rng = make_rng();
        let model = GraphRec::new(2, 3, 4, 4, &mut rng).expect("new should succeed");
        let h = model
            .user_factor(0, &[], &[])
            .expect("user_factor should succeed");
        let base = &model.user_emb[0..4];
        for (a, b) in h.iter().zip(base.iter()) {
            assert!(
                (a - b).abs() < 1e-7,
                "empty interaction must return base embedding"
            );
        }
        let z = model
            .item_factor(1, &[], &[])
            .expect("item_factor should succeed");
        let item_base = &model.item_emb[4..8];
        for (a, b) in z.iter().zip(item_base.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn err_unknown_user_and_item() {
        let mut rng = make_rng();
        let model = GraphRec::new(2, 3, 4, 4, &mut rng).expect("new should succeed");
        assert!(matches!(
            model.user_factor(9, &[0], &[1]),
            Err(RecsysError::UnknownUser { id: 9 })
        ));
        assert!(matches!(
            model.user_factor(0, &[9], &[1]),
            Err(RecsysError::UnknownItem { id: 9 })
        ));
        assert!(matches!(
            model.item_factor(9, &[0], &[1]),
            Err(RecsysError::UnknownItem { id: 9 })
        ));
        assert!(matches!(model.score(0, 0), Err(RecsysError::NotFitted)));
    }

    #[test]
    fn err_dim_and_rating_validation() {
        let mut rng = make_rng();
        let model = GraphRec::new(2, 3, 4, 4, &mut rng).expect("new should succeed");
        assert!(matches!(
            model.user_factor(0, &[0, 1], &[1]),
            Err(RecsysError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            model.user_factor(0, &[0], &[99]),
            Err(RecsysError::ItemOutOfBounds { .. })
        ));
        let mut rng2 = make_rng();
        assert!(matches!(
            GraphRec::new(0, 3, 4, 4, &mut rng2),
            Err(RecsysError::InvalidNumUsers { .. })
        ));
        assert!(matches!(
            GraphRec::new(2, 0, 4, 4, &mut rng2),
            Err(RecsysError::InvalidNumItems { .. })
        ));
        assert!(matches!(
            GraphRec::new(2, 3, 0, 4, &mut rng2),
            Err(RecsysError::InvalidConfig { .. })
        ));
        assert!(matches!(
            GraphRec::new(2, 3, 4, 0, &mut rng2),
            Err(RecsysError::InvalidEmbeddingDim { d: 0 })
        ));
    }

    #[test]
    fn deterministic_given_seed() {
        let mut ra = LcgRng::new(2026);
        let mut rb = LcgRng::new(2026);
        let a = GraphRec::new(3, 4, 5, 4, &mut ra).expect("new should succeed");
        let b = GraphRec::new(3, 4, 5, 4, &mut rb).expect("new should succeed");
        assert_eq!(a.user_emb, b.user_emb);
        assert_eq!(a.item_emb, b.item_emb);
        assert_eq!(a.w_item_fuse, b.w_item_fuse);
        assert_eq!(a.n_params(), b.n_params());
    }

    #[test]
    fn n_params_closed_form() {
        let mut rng = make_rng();
        let (nu, ni, nr, d) = (3_usize, 4_usize, 5_usize, 4_usize);
        let model = GraphRec::new(nu, ni, nr, d, &mut rng).expect("new should succeed");
        let expected = nu * d + ni * d + nr * d + d * 2 * d + d + d * 2 * d + d + 1;
        assert_eq!(model.n_params(), expected);
    }

    #[test]
    fn single_neighbour_attention_is_one() {
        let mut rng = make_rng();
        let model = GraphRec::new(2, 3, 4, 4, &mut rng).expect("new should succeed");
        let alpha = model
            .attention_weights(0, &[1], &[2])
            .expect("attention_weights should succeed");
        assert_eq!(alpha.len(), 1);
        assert!(
            (alpha[0] - 1.0).abs() < 1e-6,
            "single neighbour weight must be 1"
        );
    }
}
