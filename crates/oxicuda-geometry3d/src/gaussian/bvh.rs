//! BVH-accelerated ray / 3D-Gaussian intersection.
//!
//! Builds a bounding-volume hierarchy over a set of 3D Gaussians (each bounded
//! by its `kσ` axis-aligned box) and answers ray queries that report which
//! Gaussians the ray meaningfully passes through and the peak Gaussian response
//! along the ray. This is the spatial-acceleration primitive underlying
//! ray-traced Gaussian splatting (e.g. ray-Gaussian intersection in 3DGRT).
//!
//! # Peak response along a ray
//!
//! A ray `x(t) = o + t·d` (`t >= 0`) passing a Gaussian with mean `μ` and
//! covariance `Σ` accumulates the un-normalised density
//! `ρ(t) = exp(-½ · m(t))`, `m(t) = (x(t) - μ)ᵀ Σ⁻¹ (x(t) - μ)`. Since `m(t)`
//! is a convex quadratic in `t`, its minimiser
//! `t* = (dᵀ Σ⁻¹ (μ - o)) / (dᵀ Σ⁻¹ d)` is closed-form; clamping `t*` to the
//! valid ray interval gives the point of maximal response. The opacity-weighted
//! peak `α · exp(-½ m(t*))` is what each query returns per hit.
//!
//! The BVH is a median-split binary tree over Gaussian-centroid coordinates,
//! identical in spirit to the kd-tree already in the crate but storing AABBs so
//! the slab test can prune whole sub-trees.

use crate::error::{Geom3dError, Geom3dResult};
use crate::gaussian::gaussian::Gaussian3d;

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
struct Aabb {
    min: [f32; 3],
    max: [f32; 3],
}

impl Aabb {
    fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
        }
    }

    fn expand(&mut self, other: &Aabb) {
        for k in 0..3 {
            self.min[k] = self.min[k].min(other.min[k]);
            self.max[k] = self.max[k].max(other.max[k]);
        }
    }

    fn centroid(&self) -> [f32; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    /// Slab test. Returns the entry `t` if the ray hits the box within
    /// `[t_min, t_max]`, else `None`. `inv_dir` is `1/dir` componentwise.
    fn ray_hits(&self, origin: &[f32; 3], inv_dir: &[f32; 3], t_min: f32, t_max: f32) -> bool {
        let mut tmin = t_min;
        let mut tmax = t_max;
        for k in 0..3 {
            let t1 = (self.min[k] - origin[k]) * inv_dir[k];
            let t2 = (self.max[k] - origin[k]) * inv_dir[k];
            let (lo, hi) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
            tmin = tmin.max(lo);
            tmax = tmax.min(hi);
            if tmax < tmin {
                return false;
            }
        }
        true
    }
}

/// A single ray-Gaussian hit.
#[derive(Debug, Clone, Copy)]
pub struct GaussianHit {
    /// Index of the Gaussian (into the array passed to [`GaussianBvh::build`]).
    pub index: usize,
    /// Ray parameter `t*` of maximal response (clamped to the ray interval).
    pub t: f32,
    /// Opacity-weighted peak response `α · exp(-½ · m(t*)) ∈ [0, α]`.
    pub response: f32,
}

enum Node {
    Leaf {
        bbox: Aabb,
        first: usize,
        count: usize,
    },
    Inner {
        bbox: Aabb,
        left: usize,
        right: usize,
    },
}

/// A bounding-volume hierarchy over 3D Gaussians for ray queries.
pub struct GaussianBvh {
    nodes: Vec<Node>,
    /// Permutation of Gaussian indices grouped by leaf.
    prim_indices: Vec<usize>,
    /// Per-Gaussian inverse covariance (row-major 3×3).
    inv_cov: Vec<[f32; 9]>,
    /// Per-Gaussian opacity (post-sigmoid).
    opacity: Vec<f32>,
    /// Per-Gaussian mean.
    mean: Vec<[f32; 3]>,
    root: usize,
    sigma_k: f32,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Invert a symmetric positive-definite 3×3 matrix (row-major); `None` if
/// (near-)singular.
fn invert3x3(m: &[f32; 9]) -> Option<[f32; 9]> {
    let a = m[4] * m[8] - m[5] * m[7];
    let b = m[5] * m[6] - m[3] * m[8];
    let c = m[3] * m[7] - m[4] * m[6];
    let det = m[0] * a + m[1] * b + m[2] * c;
    if det.abs() < 1e-20 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        a * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,
        b * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,
        c * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det,
    ])
}

fn mat3_vec(m: &[f32; 9], v: &[f32; 3]) -> [f32; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

fn dot3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl GaussianBvh {
    /// Build a BVH over `gaussians`, bounding each by its `sigma_k`-σ box.
    ///
    /// `sigma_k` is the number of standard deviations used for each Gaussian's
    /// AABB (a typical value is `3.0`). Gaussians whose covariance is singular
    /// are skipped (they never produce hits).
    ///
    /// # Errors
    ///
    /// Returns [`Geom3dError::InvalidRadius`] if `sigma_k <= 0` and
    /// [`Geom3dError::EmptyPointCloud`] if there are no Gaussians.
    pub fn build(gaussians: &[Gaussian3d], sigma_k: f32) -> Geom3dResult<Self> {
        if gaussians.is_empty() {
            return Err(Geom3dError::EmptyPointCloud);
        }
        if !(sigma_k > 0.0 && sigma_k.is_finite()) {
            return Err(Geom3dError::InvalidRadius { radius: sigma_k });
        }

        let n = gaussians.len();
        let mut inv_cov = vec![[0.0_f32; 9]; n];
        let mut opacity = vec![0.0_f32; n];
        let mut mean = vec![[0.0_f32; 3]; n];
        let mut boxes = vec![Aabb::empty(); n];
        let mut valid = vec![false; n];

        for (i, g) in gaussians.iter().enumerate() {
            mean[i] = g.pos;
            opacity[i] = sigmoid(g.opacity);
            let cov = match g.covariance3d() {
                Ok(c) => c,
                Err(_) => continue,
            };
            let inv = match invert3x3(&cov) {
                Some(m) => m,
                None => continue,
            };
            inv_cov[i] = inv;
            valid[i] = true;
            // Tight AABB of a kσ ellipsoid: half-extent along axis k is
            // sigma_k · sqrt(Σ_kk).
            let hx = sigma_k * cov[0].max(0.0).sqrt();
            let hy = sigma_k * cov[4].max(0.0).sqrt();
            let hz = sigma_k * cov[8].max(0.0).sqrt();
            boxes[i] = Aabb {
                min: [g.pos[0] - hx, g.pos[1] - hy, g.pos[2] - hz],
                max: [g.pos[0] + hx, g.pos[1] + hy, g.pos[2] + hz],
            };
        }

        let mut prim_indices: Vec<usize> = (0..n).filter(|&i| valid[i]).collect();
        if prim_indices.is_empty() {
            // No buildable Gaussians: a single empty leaf.
            let nodes = vec![Node::Leaf {
                bbox: Aabb {
                    min: [0.0; 3],
                    max: [0.0; 3],
                },
                first: 0,
                count: 0,
            }];
            return Ok(Self {
                nodes,
                prim_indices,
                inv_cov,
                opacity,
                mean,
                root: 0,
                sigma_k,
            });
        }

        let mut nodes: Vec<Node> = Vec::new();
        let len = prim_indices.len();
        let root = build_recursive(&mut nodes, &mut prim_indices, &boxes, 0, len);

        Ok(Self {
            nodes,
            prim_indices,
            inv_cov,
            opacity,
            mean,
            root,
            sigma_k,
        })
    }

    /// Number of σ used for each Gaussian's bounding box.
    #[must_use]
    pub fn sigma_k(&self) -> f32 {
        self.sigma_k
    }

    /// Compute the clamped peak response of Gaussian `idx` along the ray.
    fn response(
        &self,
        idx: usize,
        origin: &[f32; 3],
        dir: &[f32; 3],
        t_min: f32,
        t_max: f32,
    ) -> Option<GaussianHit> {
        let inv = &self.inv_cov[idx];
        let mu = &self.mean[idx];
        // m(t) = (o + t d - μ)ᵀ Σ⁻¹ (o + t d - μ).
        // Let e = o - μ. m(t) = (e + t d)ᵀ A (e + t d),
        //   = eᵀAe + 2 t dᵀAe + t² dᵀAd.
        let e = [origin[0] - mu[0], origin[1] - mu[1], origin[2] - mu[2]];
        let ae = mat3_vec(inv, &e);
        let ad = mat3_vec(inv, dir);
        let q_a = dot3(dir, &ad); // dᵀ A d  (>0 for PD A and d≠0)
        if q_a <= 1e-20 {
            return None;
        }
        let q_b = 2.0 * dot3(dir, &ae); // 2 dᵀ A e
        let t_star = (-q_b / (2.0 * q_a)).clamp(t_min, t_max);
        // m(t*) = eᵀAe + q_b·t* + q_a·t*²
        let eae = dot3(&e, &ae);
        let m = eae + q_b * t_star + q_a * t_star * t_star;
        let resp = self.opacity[idx] * (-0.5 * m).exp();
        Some(GaussianHit {
            index: idx,
            t: t_star,
            response: resp,
        })
    }

    /// Query the ray `origin + t·dir`, returning every Gaussian whose peak
    /// response (within `[0, t_max]`) is at least `min_response`, sorted by
    /// ascending `t`.
    ///
    /// `dir` need not be normalised; `t` is expressed in units of `dir`'s
    /// length. `t_max` bounds the search distance (use `f32::INFINITY` for an
    /// unbounded ray).
    ///
    /// # Errors
    ///
    /// Returns [`Geom3dError::InvalidRadius`] if `dir` is the zero vector.
    pub fn ray_query(
        &self,
        origin: &[f32; 3],
        dir: &[f32; 3],
        t_max: f32,
        min_response: f32,
    ) -> Geom3dResult<Vec<GaussianHit>> {
        let len2 = dot3(dir, dir);
        if len2 <= 1e-30 {
            return Err(Geom3dError::InvalidRadius { radius: 0.0 });
        }
        let inv_dir = [
            1.0 / safe_dir(dir[0]),
            1.0 / safe_dir(dir[1]),
            1.0 / safe_dir(dir[2]),
        ];
        let mut hits = Vec::new();
        let mut stack = vec![self.root];
        while let Some(ni) = stack.pop() {
            match &self.nodes[ni] {
                Node::Leaf { bbox, first, count } => {
                    if *count == 0 {
                        continue;
                    }
                    if !bbox.ray_hits(origin, &inv_dir, 0.0, t_max) {
                        continue;
                    }
                    for &idx in &self.prim_indices[*first..*first + *count] {
                        if let Some(hit) = self.response(idx, origin, dir, 0.0, t_max) {
                            if hit.response >= min_response {
                                hits.push(hit);
                            }
                        }
                    }
                }
                Node::Inner { bbox, left, right } => {
                    if bbox.ray_hits(origin, &inv_dir, 0.0, t_max) {
                        stack.push(*left);
                        stack.push(*right);
                    }
                }
            }
        }
        hits.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
        Ok(hits)
    }
}

/// Avoid division by exactly zero in the slab test (axis-parallel rays).
fn safe_dir(v: f32) -> f32 {
    if v.abs() < 1e-20 {
        if v < 0.0 { -1e-20 } else { 1e-20 }
    } else {
        v
    }
}

/// Maximum primitives per BVH leaf.
const MAX_LEAF: usize = 4;

fn aabb_of(indices: &[usize], boxes: &[Aabb], lo: usize, hi: usize) -> Aabb {
    let mut b = Aabb::empty();
    for &i in &indices[lo..hi] {
        b.expand(&boxes[i]);
    }
    b
}

fn build_recursive(
    nodes: &mut Vec<Node>,
    indices: &mut [usize],
    boxes: &[Aabb],
    lo: usize,
    hi: usize,
) -> usize {
    let bbox = aabb_of(indices, boxes, lo, hi);
    let count = hi - lo;
    if count <= MAX_LEAF {
        let id = nodes.len();
        nodes.push(Node::Leaf {
            bbox,
            first: lo,
            count,
        });
        return id;
    }
    // Split on the centroid AABB's longest axis at the median.
    let mut cb = Aabb::empty();
    for &i in &indices[lo..hi] {
        let c = boxes[i].centroid();
        cb.expand(&Aabb { min: c, max: c });
    }
    let ext = [
        cb.max[0] - cb.min[0],
        cb.max[1] - cb.min[1],
        cb.max[2] - cb.min[2],
    ];
    let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] {
        0
    } else if ext[1] >= ext[2] {
        1
    } else {
        2
    };
    let mid = lo + count / 2;
    indices[lo..hi].sort_by(|&a, &b| {
        boxes[a].centroid()[axis]
            .partial_cmp(&boxes[b].centroid()[axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Reserve this inner node's slot, then build children.
    let id = nodes.len();
    nodes.push(Node::Leaf {
        bbox,
        first: lo,
        count,
    }); // placeholder
    let left = build_recursive(nodes, indices, boxes, lo, mid);
    let right = build_recursive(nodes, indices, boxes, mid, hi);
    nodes[id] = Node::Inner { bbox, left, right };
    id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn iso_gaussian(pos: [f32; 3], log_scale: f32, opacity: f32) -> Gaussian3d {
        Gaussian3d {
            pos,
            rot: [1.0, 0.0, 0.0, 0.0],
            scale: [log_scale, log_scale, log_scale],
            opacity,
            sh: vec![0.0; 27],
        }
    }

    /// Brute-force reference: peak response for each Gaussian along the ray.
    fn brute_force(
        gaussians: &[Gaussian3d],
        origin: &[f32; 3],
        dir: &[f32; 3],
        t_max: f32,
        min_response: f32,
        sigma_k: f32,
    ) -> Vec<GaussianHit> {
        let bvh = GaussianBvh::build(gaussians, sigma_k).expect("build should succeed");
        let mut out = Vec::new();
        for idx in 0..gaussians.len() {
            if !bvh.prim_indices.contains(&idx) {
                continue;
            }
            if let Some(h) = bvh.response(idx, origin, dir, 0.0, t_max) {
                if h.response >= min_response {
                    out.push(h);
                }
            }
        }
        out.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    #[test]
    fn ray_through_single_gaussian_center() {
        // Unit Gaussian at (0,0,5); ray along +z from origin passes through μ.
        let g = iso_gaussian([0.0, 0.0, 5.0], 0.0, 0.0); // sigmoid(0)=0.5
        let bvh = GaussianBvh::build(&[g], 3.0).expect("build should succeed");
        let hits = bvh
            .ray_query(&[0.0, 0.0, 0.0], &[0.0, 0.0, 1.0], f32::INFINITY, 0.0)
            .expect("query should succeed");
        assert_eq!(hits.len(), 1);
        // Peak at t=5 (μ.z), response = α·exp(0) = 0.5.
        assert!((hits[0].t - 5.0).abs() < 1e-3, "t={}", hits[0].t);
        assert!(
            (hits[0].response - 0.5).abs() < 1e-4,
            "r={}",
            hits[0].response
        );
    }

    #[test]
    fn ray_missing_gaussian_low_response() {
        // Gaussian at (0,0,5) with tiny scale; ray far to the side misses it.
        let g = iso_gaussian([0.0, 0.0, 5.0], (0.05_f32).ln(), 2.0);
        let bvh = GaussianBvh::build(&[g], 3.0).expect("build should succeed");
        let hits = bvh
            .ray_query(&[10.0, 0.0, 0.0], &[0.0, 0.0, 1.0], f32::INFINITY, 0.1)
            .expect("query should succeed");
        assert!(hits.is_empty(), "far ray should not register a strong hit");
    }

    #[test]
    fn bvh_matches_brute_force_random() {
        let mut rng = LcgRng::new(2024);
        let mut gs = Vec::new();
        for _ in 0..120 {
            let x = (rng.next_u32() as f32 / 4_294_967_296.0) * 10.0 - 5.0;
            let y = (rng.next_u32() as f32 / 4_294_967_296.0) * 10.0 - 5.0;
            let z = (rng.next_u32() as f32 / 4_294_967_296.0) * 10.0;
            gs.push(iso_gaussian([x, y, z], (0.4_f32).ln(), 1.0));
        }
        let bvh = GaussianBvh::build(&gs, 3.0).expect("build should succeed");

        // Several rays.
        let rays = [
            ([0.0_f32, 0.0, -1.0], [0.0_f32, 0.1, 1.0]),
            ([-5.0, -5.0, 0.0], [1.0, 1.0, 0.5]),
            ([2.0, -3.0, 0.0], [-0.3, 0.6, 1.0]),
        ];
        for (o, d) in &rays {
            let q = bvh
                .ray_query(o, d, f32::INFINITY, 0.05)
                .expect("query should succeed");
            let bf = brute_force(&gs, o, d, f32::INFINITY, 0.05, 3.0);
            assert_eq!(q.len(), bf.len(), "hit count mismatch for ray {o:?}->{d:?}");
            // Same hit set (compare sorted index lists).
            let mut qi: Vec<usize> = q.iter().map(|h| h.index).collect();
            let mut bi: Vec<usize> = bf.iter().map(|h| h.index).collect();
            qi.sort_unstable();
            bi.sort_unstable();
            assert_eq!(qi, bi, "hit indices mismatch");
            for (a, b) in q.iter().zip(bf.iter()) {
                assert!((a.t - b.t).abs() < 1e-3);
                assert!((a.response - b.response).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn empty_input_errors() {
        assert!(GaussianBvh::build(&[], 3.0).is_err());
    }

    #[test]
    fn invalid_sigma_errors() {
        let g = iso_gaussian([0.0, 0.0, 1.0], 0.0, 0.0);
        assert!(GaussianBvh::build(&[g], 0.0).is_err());
    }

    #[test]
    fn zero_direction_errors() {
        let g = iso_gaussian([0.0, 0.0, 1.0], 0.0, 0.0);
        let bvh = GaussianBvh::build(&[g], 3.0).expect("build should succeed");
        assert!(
            bvh.ray_query(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], 10.0, 0.0)
                .is_err()
        );
    }

    #[test]
    fn t_max_bounds_search() {
        // Gaussian at z=5; with t_max=2 the clamped peak is at t=2, low response.
        let g = iso_gaussian([0.0, 0.0, 5.0], 0.0, 5.0);
        let bvh = GaussianBvh::build(&[g], 3.0).expect("build should succeed");
        let unbounded = bvh
            .ray_query(&[0.0, 0.0, 0.0], &[0.0, 0.0, 1.0], f32::INFINITY, 0.0)
            .expect("query should succeed");
        let bounded = bvh
            .ray_query(&[0.0, 0.0, 0.0], &[0.0, 0.0, 1.0], 2.0, 0.0)
            .expect("query should succeed");
        assert!(bounded[0].t <= 2.0 + 1e-6);
        assert!(
            bounded[0].response < unbounded[0].response,
            "clamped peak must be weaker"
        );
    }
}
