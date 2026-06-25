//! Human-NeRF / InstantAvatar: skeleton-driven canonical-space mapping.
//!
//! Weng et al. (2022) "HumanNeRF" and Jiang et al. (2023) "InstantAvatar".
//!
//! An animatable human is modelled as a NeRF defined in a *canonical* pose. To
//! render the body in an arbitrary *observed* pose, observation-space query
//! points are warped **back** into canonical space by inverting a Linear Blend
//! Skinning (LBS) deformation driven by a kinematic skeleton, and the canonical
//! radiance field is evaluated there:
//!
//! ```text
//! x_obs  ──(inverse LBS)──▶  x_can  ──(canonical NeRF)──▶ (σ, rgb)
//! ```
//!
//! # Skeleton & forward kinematics
//!
//! A [`Skeleton`] holds `J` joints, each with a rest-pose world position and a
//! parent index (`-1`/`None` for the root). A *pose* assigns every joint a local
//! rotation (axis-angle). Forward kinematics composes these into per-joint world
//! rigid transforms `G_j = [R_j | t_j] ∈ SE(3)`, expressed relative to the rest
//! pose so that the rest pose maps to the identity warp (every canonical point
//! stays put when the observed pose equals the rest pose).
//!
//! # Linear Blend Skinning (forward)
//!
//! For a canonical point `x_c` with skinning weights `w_j(x_c)` (Σ_j w_j = 1):
//! ```text
//! x_obs = ( Σ_j w_j · G_j ) · x_c        (homogeneous 3×4 blend).
//! ```
//!
//! # Inverse skinning (observation → canonical)
//!
//! Following InstantAvatar's iterative root-finding, we evaluate the blend
//! transform `B(x) = Σ_j w_j(x)·G_j`, invert the affine `B`, and iterate
//! `x_c ← B(x̂_c)⁻¹ · x_obs`, recomputing the weights at the current canonical
//! estimate until convergence. Because weights are bounded and `B` is a smooth
//! convex blend of rigid transforms, a handful of fixed-point iterations
//! converges for points near the body surface.
//!
//! Skinning weights use a distance-to-bone heat kernel
//! `w_j ∝ exp(-d_j² / (2τ²))` (`d_j` = distance from the point to joint `j`),
//! a standard differentiable approximation to learned skinning weights.
//!
//! All randomness uses the crate [`LcgRng`]; no external crates.

use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;
use crate::network::tiny_nerf::TinyNerf;

// ─── Rigid transform (SE(3), stored as 3×4) ────────────────────────────────────

/// A rigid body transform `[R | t]`: a 3×3 rotation `r` (row-major) and a
/// translation `t`.
#[derive(Debug, Clone, Copy)]
pub struct Rigid {
    /// Row-major 3×3 rotation.
    pub r: [[f32; 3]; 3],
    /// Translation.
    pub t: [f32; 3],
}

impl Rigid {
    /// The identity transform.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t: [0.0, 0.0, 0.0],
        }
    }

    /// Build from an axis-angle rotation (`axis * angle`) and translation.
    ///
    /// Uses Rodrigues' formula.
    #[must_use]
    pub fn from_axis_angle(axis_angle: [f32; 3], t: [f32; 3]) -> Self {
        let theta = (axis_angle[0] * axis_angle[0]
            + axis_angle[1] * axis_angle[1]
            + axis_angle[2] * axis_angle[2])
            .sqrt();
        if theta < 1.0e-8 {
            return Self {
                r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                t,
            };
        }
        let (kx, ky, kz) = (
            axis_angle[0] / theta,
            axis_angle[1] / theta,
            axis_angle[2] / theta,
        );
        let c = theta.cos();
        let s = theta.sin();
        let one_c = 1.0 - c;
        // R = I·c + (1-c)·k kᵀ + sin·[k]_×
        let r = [
            [
                c + kx * kx * one_c,
                kx * ky * one_c - kz * s,
                kx * kz * one_c + ky * s,
            ],
            [
                ky * kx * one_c + kz * s,
                c + ky * ky * one_c,
                ky * kz * one_c - kx * s,
            ],
            [
                kz * kx * one_c - ky * s,
                kz * ky * one_c + kx * s,
                c + kz * kz * one_c,
            ],
        ];
        Self { r, t }
    }

    /// Apply to a point: `R·p + t`.
    #[must_use]
    pub fn apply(&self, p: [f32; 3]) -> [f32; 3] {
        [
            self.r[0][0] * p[0] + self.r[0][1] * p[1] + self.r[0][2] * p[2] + self.t[0],
            self.r[1][0] * p[0] + self.r[1][1] * p[1] + self.r[1][2] * p[2] + self.t[1],
            self.r[2][0] * p[0] + self.r[2][1] * p[1] + self.r[2][2] * p[2] + self.t[2],
        ]
    }

    /// Compose: `self ∘ other` (apply `other` first, then `self`).
    #[must_use]
    pub fn compose(&self, other: &Rigid) -> Rigid {
        let mut r = [[0.0_f32; 3]; 3];
        for (i, ri) in r.iter_mut().enumerate() {
            for (j, rij) in ri.iter_mut().enumerate() {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += self.r[i][k] * other.r[k][j];
                }
                *rij = acc;
            }
        }
        let t = self.apply(other.t);
        Rigid { r, t }
    }
}

// ─── Affine blend (3×4) and its inversion ──────────────────────────────────────

/// A general affine map `y = A·x + b` (3×3 `a`, translation `b`). The LBS blend
/// of several rigid transforms is affine but **not** generally rigid, so it is
/// inverted as a full 3×3 affine.
#[derive(Debug, Clone, Copy)]
struct Affine {
    a: [[f32; 3]; 3],
    b: [f32; 3],
}

impl Affine {
    fn apply(&self, p: [f32; 3]) -> [f32; 3] {
        [
            self.a[0][0] * p[0] + self.a[0][1] * p[1] + self.a[0][2] * p[2] + self.b[0],
            self.a[1][0] * p[0] + self.a[1][1] * p[1] + self.a[1][2] * p[2] + self.b[1],
            self.a[2][0] * p[0] + self.a[2][1] * p[1] + self.a[2][2] * p[2] + self.b[2],
        ]
    }

    /// Solve `A·x = (p - b)` for `x` (the inverse map at point `p`).
    ///
    /// Returns `None` if `A` is singular.
    fn solve_inverse(&self, p: [f32; 3]) -> Option<[f32; 3]> {
        let m = self.a;
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        if det.abs() < 1.0e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        // Cofactor / adjugate inverse.
        let inv = [
            [
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
            ],
            [
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
            ],
            [
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
            ],
        ];
        let d = [p[0] - self.b[0], p[1] - self.b[1], p[2] - self.b[2]];
        Some([
            inv[0][0] * d[0] + inv[0][1] * d[1] + inv[0][2] * d[2],
            inv[1][0] * d[0] + inv[1][1] * d[1] + inv[1][2] * d[2],
            inv[2][0] * d[0] + inv[2][1] * d[1] + inv[2][2] * d[2],
        ])
    }
}

// ─── Skeleton ──────────────────────────────────────────────────────────────────

/// A kinematic skeleton: rest-pose joint positions and parent topology.
#[derive(Debug, Clone)]
pub struct Skeleton {
    /// Rest-pose world position of each joint.
    rest_positions: Vec<[f32; 3]>,
    /// Parent joint index, or `usize::MAX` for a root.
    parents: Vec<usize>,
}

/// Sentinel parent index marking a root joint.
pub const NO_PARENT: usize = usize::MAX;

impl Skeleton {
    /// Build a skeleton from rest positions and parent indices.
    ///
    /// # Errors
    ///
    /// - [`NerfError::EmptyInput`] if there are no joints.
    /// - [`NerfError::DimensionMismatch`] if the two slices differ in length.
    /// - [`NerfError::InvalidEmbeddingConfig`] if a parent index is out of range
    ///   or a joint is its own parent.
    pub fn new(rest_positions: Vec<[f32; 3]>, parents: Vec<usize>) -> NerfResult<Self> {
        if rest_positions.is_empty() {
            return Err(NerfError::EmptyInput);
        }
        if rest_positions.len() != parents.len() {
            return Err(NerfError::DimensionMismatch {
                expected: rest_positions.len(),
                got: parents.len(),
            });
        }
        let j = rest_positions.len();
        for (i, &p) in parents.iter().enumerate() {
            if p != NO_PARENT && (p >= j || p == i) {
                return Err(NerfError::InvalidEmbeddingConfig {
                    msg: format!("invalid parent {p} for joint {i}"),
                });
            }
        }
        Ok(Self {
            rest_positions,
            parents,
        })
    }

    /// Number of joints.
    #[must_use]
    pub fn n_joints(&self) -> usize {
        self.rest_positions.len()
    }

    /// Rest-pose joint positions.
    #[must_use]
    pub fn rest_positions(&self) -> &[[f32; 3]] {
        &self.rest_positions
    }

    /// Forward kinematics: compose per-joint local rotations into world rigid
    /// transforms that map *rest* coordinates to *posed* coordinates.
    ///
    /// `local_rotations[j]` is the axis-angle rotation of joint `j` about its
    /// rest position, applied relative to its parent. The returned transform of
    /// joint `j` satisfies `G_j(rest_j) = posed_j` and, crucially, equals the
    /// identity when all rotations are zero (rest pose ⇒ identity warp).
    ///
    /// # Errors
    ///
    /// [`NerfError::DimensionMismatch`] if `local_rotations.len() != n_joints`.
    pub fn forward_kinematics(&self, local_rotations: &[[f32; 3]]) -> NerfResult<Vec<Rigid>> {
        let j = self.n_joints();
        if local_rotations.len() != j {
            return Err(NerfError::DimensionMismatch {
                expected: j,
                got: local_rotations.len(),
            });
        }
        // Resolve a topological order (parents before children) by repeated
        // passes; skeletons are tiny so this is inexpensive.
        let mut world: Vec<Option<Rigid>> = vec![None; j];
        let mut remaining = j;
        let mut guard = 0usize;
        while remaining > 0 {
            guard += 1;
            if guard > j + 2 {
                return Err(NerfError::InvalidEmbeddingConfig {
                    msg: "skeleton has a cycle or detached joint".into(),
                });
            }
            for i in 0..j {
                if world[i].is_some() {
                    continue;
                }
                let parent = self.parents[i];
                // A joint's local transform rotates space about its rest pivot.
                let pivot = self.rest_positions[i];
                let rot = Rigid::from_axis_angle(local_rotations[i], [0.0, 0.0, 0.0]);
                // Conjugate by the pivot: T(pivot)·R·T(-pivot).
                let local = conjugate_about_pivot(&rot, pivot);

                if parent == NO_PARENT {
                    world[i] = Some(local);
                    remaining -= 1;
                } else if let Some(pw) = world[parent] {
                    world[i] = Some(pw.compose(&local));
                    remaining -= 1;
                }
            }
        }
        let mut out = Vec::with_capacity(j);
        for w in world {
            out.push(w.unwrap_or_else(Rigid::identity));
        }
        Ok(out)
    }
}

/// Conjugate a rotation by a translation pivot: `T(pivot)·R·T(-pivot)`.
///
/// The result rotates space about `pivot` rather than the origin.
fn conjugate_about_pivot(rot: &Rigid, pivot: [f32; 3]) -> Rigid {
    let neg = [-pivot[0], -pivot[1], -pivot[2]];
    let to_origin = Rigid {
        r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        t: neg,
    };
    let back = Rigid {
        r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        t: pivot,
    };
    back.compose(&rot.compose(&to_origin))
}

// ─── Config ────────────────────────────────────────────────────────────────────

/// Configuration for the human-NeRF avatar.
#[derive(Debug, Clone)]
pub struct HumanNerfConfig {
    /// Frequency levels for the canonical positional encoding.
    pub pos_freq: usize,
    /// Hidden width of the canonical NeRF MLP.
    pub hidden_dim: usize,
    /// Skinning-weight bandwidth `τ` (heat kernel std-dev).
    pub skin_bandwidth: f32,
    /// Number of inverse-skinning fixed-point iterations.
    pub inverse_iters: usize,
}

impl Default for HumanNerfConfig {
    fn default() -> Self {
        Self {
            pos_freq: 6,
            hidden_dim: 64,
            skin_bandwidth: 0.5,
            inverse_iters: 8,
        }
    }
}

// ─── HumanNerf ─────────────────────────────────────────────────────────────────

/// Skeleton-driven canonical-space NeRF (HumanNeRF / InstantAvatar core).
#[derive(Debug, Clone)]
pub struct HumanNerf {
    skeleton: Skeleton,
    canonical: TinyNerf,
    cfg: HumanNerfConfig,
    pos_cfg: crate::encoding::positional::PosEncConfig,
}

impl HumanNerf {
    /// Build an avatar over a skeleton with a randomly-initialised canonical NeRF.
    ///
    /// # Errors
    ///
    /// - [`NerfError::InvalidFreqLevels`] if `pos_freq == 0`.
    /// - [`NerfError::InvalidFeatureDim`] if `hidden_dim == 0`.
    /// - [`NerfError::InvalidEmbeddingConfig`] if `skin_bandwidth <= 0` or
    ///   `inverse_iters == 0`.
    pub fn new(skeleton: Skeleton, cfg: HumanNerfConfig, rng: &mut LcgRng) -> NerfResult<Self> {
        if cfg.pos_freq == 0 {
            return Err(NerfError::InvalidFreqLevels { levels: 0 });
        }
        if cfg.hidden_dim == 0 {
            return Err(NerfError::InvalidFeatureDim { dim: 0 });
        }
        if !cfg.skin_bandwidth.is_finite() || cfg.skin_bandwidth <= 0.0 {
            return Err(NerfError::InvalidEmbeddingConfig {
                msg: "skin_bandwidth must be positive".into(),
            });
        }
        if cfg.inverse_iters == 0 {
            return Err(NerfError::InvalidEmbeddingConfig {
                msg: "inverse_iters must be > 0".into(),
            });
        }
        let pos_cfg = crate::encoding::positional::PosEncConfig {
            n_freq: cfg.pos_freq,
            include_input: true,
            input_dim: 3,
        };
        let canonical = TinyNerf::new(pos_cfg.output_dim(), cfg.hidden_dim, rng);
        Ok(Self {
            skeleton,
            canonical,
            cfg,
            pos_cfg,
        })
    }

    /// Borrow the skeleton.
    #[must_use]
    pub fn skeleton(&self) -> &Skeleton {
        &self.skeleton
    }

    /// Skinning weights of a point relative to each joint via the distance heat
    /// kernel `w_j ∝ exp(-‖p − joint_j‖² / (2τ²))`, normalised to sum to 1.
    fn skin_weights(&self, p: [f32; 3]) -> Vec<f32> {
        let tau2 = self.cfg.skin_bandwidth * self.cfg.skin_bandwidth;
        let mut w = vec![0.0_f32; self.skeleton.n_joints()];
        let mut max_logit = f32::NEG_INFINITY;
        // Compute logits first for a numerically-stable softmax-style normaliser.
        for (j, jp) in self.skeleton.rest_positions.iter().enumerate() {
            let dx = p[0] - jp[0];
            let dy = p[1] - jp[1];
            let dz = p[2] - jp[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            let logit = -d2 / (2.0 * tau2);
            w[j] = logit;
            if logit > max_logit {
                max_logit = logit;
            }
        }
        let mut sum = 0.0_f32;
        for wi in w.iter_mut() {
            *wi = (*wi - max_logit).exp();
            sum += *wi;
        }
        if sum > 0.0 {
            for wi in w.iter_mut() {
                *wi /= sum;
            }
        } else {
            let u = 1.0 / w.len() as f32;
            for wi in w.iter_mut() {
                *wi = u;
            }
        }
        w
    }

    /// Build the blended affine `B(x) = Σ_j w_j(x) · G_j` at canonical point `x`.
    fn blend_affine(&self, x: [f32; 3], bones: &[Rigid]) -> Affine {
        let w = self.skin_weights(x);
        let mut a = [[0.0_f32; 3]; 3];
        let mut b = [0.0_f32; 3];
        for (j, g) in bones.iter().enumerate() {
            let wj = w[j];
            for (i, ai) in a.iter_mut().enumerate() {
                for (k, aik) in ai.iter_mut().enumerate() {
                    *aik += wj * g.r[i][k];
                }
                b[i] += wj * g.t[i];
            }
        }
        Affine { a, b }
    }

    /// Forward LBS: warp a canonical point into observation space under `bones`.
    ///
    /// # Errors
    ///
    /// [`NerfError::DimensionMismatch`] if `bones.len() != n_joints`.
    pub fn skin_forward(&self, x_can: [f32; 3], bones: &[Rigid]) -> NerfResult<[f32; 3]> {
        if bones.len() != self.skeleton.n_joints() {
            return Err(NerfError::DimensionMismatch {
                expected: self.skeleton.n_joints(),
                got: bones.len(),
            });
        }
        let aff = self.blend_affine(x_can, bones);
        Ok(aff.apply(x_can))
    }

    /// Inverse LBS: recover the canonical point of an observation-space query by
    /// iterative root-finding (InstantAvatar-style).
    ///
    /// Returns `None` if the blend transform is singular at some iterate.
    ///
    /// # Errors
    ///
    /// [`NerfError::DimensionMismatch`] if `bones.len() != n_joints`.
    pub fn skin_inverse(&self, x_obs: [f32; 3], bones: &[Rigid]) -> NerfResult<Option<[f32; 3]>> {
        if bones.len() != self.skeleton.n_joints() {
            return Err(NerfError::DimensionMismatch {
                expected: self.skeleton.n_joints(),
                got: bones.len(),
            });
        }
        // Initialise the canonical estimate at the observation point.
        let mut x_can = x_obs;
        for _ in 0..self.cfg.inverse_iters {
            let aff = self.blend_affine(x_can, bones);
            match aff.solve_inverse(x_obs) {
                Some(next) => {
                    let dx = next[0] - x_can[0];
                    let dy = next[1] - x_can[1];
                    let dz = next[2] - x_can[2];
                    x_can = next;
                    if dx * dx + dy * dy + dz * dz < 1.0e-12 {
                        break;
                    }
                }
                None => return Ok(None),
            }
        }
        Ok(Some(x_can))
    }

    /// Query the avatar at an observation-space point under a given pose.
    ///
    /// Warps the point into canonical space, positionally encodes it, and
    /// evaluates the canonical NeRF, returning `(sigma, rgb)`. If the inverse
    /// warp is singular, returns zero density (empty space).
    ///
    /// # Errors
    ///
    /// Propagates shape and encoding errors.
    pub fn query_observed(&self, x_obs: [f32; 3], bones: &[Rigid]) -> NerfResult<(f32, [f32; 3])> {
        let x_can = match self.skin_inverse(x_obs, bones)? {
            Some(c) => c,
            None => return Ok((0.0, [0.0, 0.0, 0.0])),
        };
        let pe = crate::encoding::positional::positional_encode(&x_can, &self.pos_cfg)?;
        self.canonical.forward(&pe)
    }

    /// Convenience: query the canonical field directly (no skinning).
    ///
    /// # Errors
    ///
    /// Propagates encoding errors.
    pub fn query_canonical(&self, x_can: [f32; 3]) -> NerfResult<(f32, [f32; 3])> {
        let pe = crate::encoding::positional::positional_encode(&x_can, &self.pos_cfg)?;
        self.canonical.forward(&pe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple 3-joint chain along +x: root at origin, then x=1, then x=2.
    fn chain_skeleton() -> Skeleton {
        Skeleton::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            vec![NO_PARENT, 0, 1],
        )
        .expect("skeleton")
    }

    #[test]
    fn rigid_axis_angle_identity() {
        let r = Rigid::from_axis_angle([0.0, 0.0, 0.0], [0.5, -0.2, 0.1]);
        let p = r.apply([1.0, 2.0, 3.0]);
        assert!((p[0] - 1.5).abs() < 1e-6);
        assert!((p[1] - 1.8).abs() < 1e-6);
        assert!((p[2] - 3.1).abs() < 1e-6);
    }

    #[test]
    fn rigid_rotation_preserves_length() {
        let r = Rigid::from_axis_angle([0.0, 0.0, std::f32::consts::FRAC_PI_2], [0.0, 0.0, 0.0]);
        let p = [1.0, 0.0, 0.0];
        let q = r.apply(p);
        // 90° about z maps (1,0,0) → (0,1,0).
        assert!((q[0]).abs() < 1e-6, "qx={}", q[0]);
        assert!((q[1] - 1.0).abs() < 1e-6, "qy={}", q[1]);
        let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fk_rest_pose_is_identity() {
        let skel = chain_skeleton();
        let zero = vec![[0.0_f32; 3]; 3];
        let bones = skel.forward_kinematics(&zero).expect("fk");
        for g in &bones {
            // Each world transform should map any point to itself at rest.
            let p = g.apply([0.3, -0.7, 0.9]);
            assert!((p[0] - 0.3).abs() < 1e-5);
            assert!((p[1] + 0.7).abs() < 1e-5);
            assert!((p[2] - 0.9).abs() < 1e-5);
        }
    }

    #[test]
    fn fk_child_follows_parent_rotation() {
        let skel = chain_skeleton();
        // Rotate the root 90° about z; children inherit it.
        let mut rots = vec![[0.0_f32; 3]; 3];
        rots[0] = [0.0, 0.0, std::f32::consts::FRAC_PI_2];
        let bones = skel.forward_kinematics(&rots).expect("fk");
        // Joint 1 rest position (1,0,0) rotates about root pivot (0,0,0) → (0,1,0).
        let j1 = bones[1].apply([1.0, 0.0, 0.0]);
        assert!((j1[0]).abs() < 1e-4, "j1x={}", j1[0]);
        assert!((j1[1] - 1.0).abs() < 1e-4, "j1y={}", j1[1]);
    }

    #[test]
    fn skin_weights_normalised() {
        let mut rng = LcgRng::new(1);
        let avatar =
            HumanNerf::new(chain_skeleton(), HumanNerfConfig::default(), &mut rng).expect("new");
        let w = avatar.skin_weights([0.9, 0.0, 0.0]);
        let s: f32 = w.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "weights sum={s}");
        // Nearest joint (joint 1 at x=1) should dominate.
        assert!(w[1] > w[0] && w[1] > w[2], "weights={w:?}");
    }

    #[test]
    fn rest_pose_warp_is_identity() {
        let mut rng = LcgRng::new(2);
        let avatar =
            HumanNerf::new(chain_skeleton(), HumanNerfConfig::default(), &mut rng).expect("new");
        let zero = vec![[0.0_f32; 3]; 3];
        let bones = avatar.skeleton().forward_kinematics(&zero).expect("fk");
        let x = [0.7, 0.1, -0.2];
        let obs = avatar.skin_forward(x, &bones).expect("forward");
        assert!((obs[0] - x[0]).abs() < 1e-5);
        assert!((obs[1] - x[1]).abs() < 1e-5);
        assert!((obs[2] - x[2]).abs() < 1e-5);
    }

    #[test]
    fn inverse_recovers_canonical_point() {
        let mut rng = LcgRng::new(3);
        let cfg = HumanNerfConfig {
            inverse_iters: 16,
            ..Default::default()
        };
        let avatar = HumanNerf::new(chain_skeleton(), cfg, &mut rng).expect("new");
        // Pose: rotate root modestly about z.
        let mut rots = vec![[0.0_f32; 3]; 3];
        rots[0] = [0.0, 0.0, 0.3];
        let bones = avatar.skeleton().forward_kinematics(&rots).expect("fk");

        let x_can = [0.6, 0.15, 0.0];
        let x_obs = avatar.skin_forward(x_can, &bones).expect("forward");
        let recovered = avatar
            .skin_inverse(x_obs, &bones)
            .expect("inverse")
            .expect("non-singular");
        let err = ((recovered[0] - x_can[0]).powi(2)
            + (recovered[1] - x_can[1]).powi(2)
            + (recovered[2] - x_can[2]).powi(2))
        .sqrt();
        assert!(err < 1e-2, "inverse error {err}, recovered={recovered:?}");
    }

    #[test]
    fn query_observed_finite_and_valid() {
        let mut rng = LcgRng::new(5);
        let avatar =
            HumanNerf::new(chain_skeleton(), HumanNerfConfig::default(), &mut rng).expect("new");
        let mut rots = vec![[0.0_f32; 3]; 3];
        rots[1] = [0.0, 0.0, 0.2];
        let bones = avatar.skeleton().forward_kinematics(&rots).expect("fk");
        let (sigma, rgb) = avatar
            .query_observed([1.1, 0.05, 0.0], &bones)
            .expect("query");
        assert!(sigma.is_finite() && sigma >= 0.0);
        for c in rgb {
            assert!(c.is_finite() && (0.0..=1.0).contains(&c));
        }
    }

    #[test]
    fn query_observed_deterministic() {
        let mut rng = LcgRng::new(9);
        let avatar =
            HumanNerf::new(chain_skeleton(), HumanNerfConfig::default(), &mut rng).expect("new");
        let zero = vec![[0.0_f32; 3]; 3];
        let bones = avatar.skeleton().forward_kinematics(&zero).expect("fk");
        let a = avatar.query_observed([0.5, 0.0, 0.0], &bones).expect("q1");
        let b = avatar.query_observed([0.5, 0.0, 0.0], &bones).expect("q2");
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
    }

    #[test]
    fn skeleton_validation() {
        assert!(Skeleton::new(vec![], vec![]).is_err());
        assert!(Skeleton::new(vec![[0.0; 3]], vec![NO_PARENT, NO_PARENT]).is_err());
        // self-parent
        assert!(Skeleton::new(vec![[0.0; 3]], vec![0]).is_err());
        // out-of-range parent
        assert!(Skeleton::new(vec![[0.0; 3], [1.0, 0.0, 0.0]], vec![NO_PARENT, 5]).is_err());
    }

    #[test]
    fn config_validation() {
        let mut rng = LcgRng::new(7);
        let bad = HumanNerfConfig {
            skin_bandwidth: 0.0,
            ..Default::default()
        };
        assert!(HumanNerf::new(chain_skeleton(), bad, &mut rng).is_err());
        let bad2 = HumanNerfConfig {
            inverse_iters: 0,
            ..Default::default()
        };
        assert!(HumanNerf::new(chain_skeleton(), bad2, &mut rng).is_err());
    }
}
