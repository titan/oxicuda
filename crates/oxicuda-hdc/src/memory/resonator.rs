//! VSA Resonator Network (Frady-Kent-Olshausen-Sommer, NeurIPS 2020).
//!
//! Given a composite HV `s = Σ_i bind(fillers[i], roles[i])` (element-wise sum of HRR
//! bindings), known role HVs, and per-role codebooks of candidate fillers, the Resonator
//! iteratively refines estimates of each filler until all role assignments converge.
//!
//! Each iteration:
//! ```text
//! for i in 0..n_roles:
//!     residual = s - Σ_{j ≠ i} bind(estimates[j], roles[j])
//!     probe    = unbind(roles[i], residual)   // circular_correlation(roles[i], residual)
//!     estimates[i] = codebook[i].query_with_hv(probe)
//! ```
//! Converged when all filler IDs are stable across two consecutive iterations.

use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::vector::hrr::{HrrItemMemory, hrr_bind, hrr_unbind};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for a `ResonatorNetwork` decomposition run.
#[derive(Debug, Clone)]
pub struct ResonatorConfig {
    /// Number of roles (and fillers) in the composite HV.
    pub n_roles: usize,
    /// Maximum number of fixed-point iterations.
    pub max_iter: usize,
    /// If `true`, initialise estimates with random HVs drawn from the corresponding
    /// codebooks. If `false`, the caller must supply `init_estimates`.
    pub init_random: bool,
}

impl Default for ResonatorConfig {
    fn default() -> Self {
        Self {
            n_roles: 2,
            max_iter: 100,
            init_random: true,
        }
    }
}

// ── Result ────────────────────────────────────────────────────────────────────

/// Result of a `ResonatorNetwork::decompose` call.
#[derive(Debug, Clone)]
pub struct ResonatorResult {
    /// Filler IDs (one per role), in role order.
    pub filler_ids: Vec<usize>,
    /// Filler HVs (one per role), in role order.
    pub filler_hvs: Vec<Vec<f32>>,
    /// Number of iterations actually executed.
    pub n_iter: usize,
    /// Whether the network converged (all IDs stable) before `max_iter` was reached.
    pub converged: bool,
}

// ── Resonator Network ─────────────────────────────────────────────────────────

/// Stateless resonator network — all state is local to each `decompose` call.
pub struct ResonatorNetwork;

impl ResonatorNetwork {
    /// Decompose a composite HV into role-filler pairs using fixed-point iteration.
    ///
    /// # Parameters
    ///
    /// - `composite`: the superposition HV `Σ bind(filler_i, role_i)`, length = `dim`.
    /// - `roles`: `n_roles` unit-norm role HVs, each of length `dim`.
    /// - `codebooks`: `n_roles` item memories; `codebooks[i]` contains candidates for role `i`.
    /// - `cfg`: configuration (n_roles, max_iter, init_random).
    /// - `init_estimates`: if `Some`, must supply `n_roles` initial filler estimates.
    /// - `rng`: used only when `cfg.init_random == true` and `init_estimates == None`.
    ///
    /// # Errors
    ///
    /// - `HdcError::DimensionMismatch` if any role or estimate HV has wrong length.
    /// - `HdcError::EmptyInput` if `n_roles == 0`.
    /// - `HdcError::EmptyItemMemory` if any codebook is empty.
    pub fn decompose(
        composite: &[f32],
        roles: &[Vec<f32>],
        codebooks: &[HrrItemMemory],
        cfg: &ResonatorConfig,
        init_estimates: Option<&[Vec<f32>]>,
        rng: &mut LcgRng,
    ) -> HdcResult<ResonatorResult> {
        // ── Validate ──────────────────────────────────────────────────────────
        if cfg.n_roles == 0 {
            return Err(HdcError::EmptyInput);
        }
        if composite.is_empty() {
            return Err(HdcError::ZeroDimension);
        }
        let dim = composite.len();

        if roles.len() != cfg.n_roles {
            return Err(HdcError::DimensionMismatch {
                expected: cfg.n_roles,
                got: roles.len(),
            });
        }
        if codebooks.len() != cfg.n_roles {
            return Err(HdcError::DimensionMismatch {
                expected: cfg.n_roles,
                got: codebooks.len(),
            });
        }
        for (i, role) in roles.iter().enumerate() {
            if role.len() != dim {
                return Err(HdcError::DimensionMismatch {
                    expected: dim,
                    got: role.len(),
                });
            }
            if codebooks[i].is_empty() {
                return Err(HdcError::EmptyItemMemory);
            }
        }

        // ── Initialise estimates ──────────────────────────────────────────────
        let mut estimates: Vec<Vec<f32>> = match init_estimates {
            Some(provided) => {
                if provided.len() != cfg.n_roles {
                    return Err(HdcError::DimensionMismatch {
                        expected: cfg.n_roles,
                        got: provided.len(),
                    });
                }
                for (i, est) in provided.iter().enumerate() {
                    if est.len() != dim {
                        return Err(HdcError::DimensionMismatch {
                            expected: dim,
                            got: est.len(),
                        });
                    }
                    // Validate ID exists for at least partial sanity — we just
                    // check dim; the codebook may map any HV.
                    let _ = i;
                }
                provided.to_vec()
            }
            None => {
                // Random initialisation: pick a random item from each codebook.
                let mut init = Vec::with_capacity(cfg.n_roles);
                for cb in codebooks.iter() {
                    let n = cb.len();
                    let pick = rng.next_usize(n);
                    // Walk through items to find the one at index `pick`.
                    // We use query_with_hv on a zero probe to get any HV, but that
                    // is biased; instead generate a uniform random probe and query.
                    let probe = Self::random_probe(dim, rng);
                    let (_, _, hv_ref) = cb.query_with_hv(&probe)?;
                    let _ = pick; // pick is unused after the random-probe strategy
                    init.push(hv_ref.to_vec());
                }
                init
            }
        };

        // ── Initial ID snapshot ───────────────────────────────────────────────
        let mut prev_ids: Vec<usize> = Vec::with_capacity(cfg.n_roles);
        for i in 0..cfg.n_roles {
            let probe = Self::residual_probe(composite, i, roles, &estimates)?;
            let (id, _, hv_ref) = codebooks[i].query_with_hv(&probe)?;
            estimates[i] = hv_ref.to_vec();
            prev_ids.push(id);
        }

        // Early exit if max_iter == 0.
        if cfg.max_iter == 0 {
            let filler_hvs = estimates;
            return Ok(ResonatorResult {
                filler_ids: prev_ids,
                filler_hvs,
                n_iter: 0,
                converged: false,
            });
        }

        // ── Fixed-point iteration ─────────────────────────────────────────────
        let mut n_iter = 0usize;
        let mut converged = false;

        for _iter in 0..cfg.max_iter {
            let mut new_ids = Vec::with_capacity(cfg.n_roles);

            for i in 0..cfg.n_roles {
                let (id, hv) =
                    Self::update_role(composite, i, &roles[i], roles, &estimates, &codebooks[i])?;
                estimates[i] = hv;
                new_ids.push(id);
            }

            n_iter += 1;

            if Self::estimates_converged(&prev_ids, &new_ids) {
                converged = true;
                prev_ids = new_ids;
                break;
            }
            prev_ids = new_ids;
        }

        Ok(ResonatorResult {
            filler_ids: prev_ids,
            filler_hvs: estimates,
            n_iter,
            converged,
        })
    }

    /// Compute the residual probe for role `role_idx` and query its codebook.
    ///
    /// Returns `(id, score, hv)` for the winning filler.
    ///
    /// # Errors
    ///
    /// - `HdcError::DimensionMismatch` if any HV has the wrong length.
    /// - `HdcError::EmptyItemMemory` if the codebook is empty.
    pub fn update_role(
        composite: &[f32],
        role_idx: usize,
        role: &[f32],
        all_roles: &[Vec<f32>],
        current_estimates: &[Vec<f32>],
        codebook: &HrrItemMemory,
    ) -> HdcResult<(usize, Vec<f32>)> {
        let probe = Self::residual_probe(composite, role_idx, all_roles, current_estimates)?;
        let unbound = hrr_unbind(role, &probe)?;
        let (id, _, hv_ref) = codebook.query_with_hv(&unbound)?;
        Ok((id, hv_ref.to_vec()))
    }

    /// Check whether two ID vectors are identical (convergence criterion).
    pub fn estimates_converged(ids_a: &[usize], ids_b: &[usize]) -> bool {
        ids_a.len() == ids_b.len() && ids_a.iter().zip(ids_b.iter()).all(|(a, b)| a == b)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Compute `residual = composite - Σ_{j ≠ role_idx} bind(estimates[j], roles[j])`.
    fn residual_probe(
        composite: &[f32],
        role_idx: usize,
        all_roles: &[Vec<f32>],
        estimates: &[Vec<f32>],
    ) -> HdcResult<Vec<f32>> {
        let dim = composite.len();
        let mut residual = composite.to_vec();

        for (j, (est, role)) in estimates.iter().zip(all_roles.iter()).enumerate() {
            if j == role_idx {
                continue;
            }
            if est.len() != dim || role.len() != dim {
                return Err(HdcError::DimensionMismatch {
                    expected: dim,
                    got: if est.len() != dim {
                        est.len()
                    } else {
                        role.len()
                    },
                });
            }
            let contribution = hrr_bind(est, role)?;
            for (r, c) in residual.iter_mut().zip(contribution.iter()) {
                *r -= c;
            }
        }
        Ok(residual)
    }

    /// Generate a random probe vector (uniform components) for random initialisation.
    fn random_probe(dim: usize, rng: &mut LcgRng) -> Vec<f32> {
        let mut probe = Vec::with_capacity(dim);
        for _ in 0..dim {
            probe.push(rng.next_f32() * 2.0 - 1.0);
        }
        probe
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::vector::hrr::{HrrItemMemory, hrr_bind, random_hrr};

    fn rng() -> LcgRng {
        LcgRng::new(0x1234_5678_ABCD_EF01)
    }

    /// Build a composite HV: s = Σ bind(filler_i, role_i).
    fn build_composite(fillers: &[Vec<f32>], roles: &[Vec<f32>]) -> HdcResult<Vec<f32>> {
        assert!(!fillers.is_empty());
        let dim = fillers[0].len();
        let mut acc = vec![0f32; dim];
        for (f, r) in fillers.iter().zip(roles.iter()) {
            let bound = hrr_bind(f, r)?;
            for (a, b) in acc.iter_mut().zip(bound.iter()) {
                *a += b;
            }
        }
        Ok(acc)
    }

    // ── ResonatorConfig ──────────────────────────────────────────────────────

    #[test]
    fn resonator_config_default() {
        let cfg = ResonatorConfig::default();
        assert_eq!(cfg.n_roles, 2);
        assert_eq!(cfg.max_iter, 100);
        assert!(cfg.init_random);
    }

    // ── 1-role (trivial) decomposition ──────────────────────────────────────

    #[test]
    fn decompose_one_role_trivial() {
        let mut rng = rng();
        let dim = 64;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler = random_hrr(dim, &mut rng).expect("filler");

        let composite = hrr_bind(&filler, &role).expect("bind");

        let mut codebook = HrrItemMemory::new(dim).expect("codebook");
        codebook.insert(42, filler.clone()).expect("insert");

        let cfg = ResonatorConfig {
            n_roles: 1,
            max_iter: 50,
            init_random: true,
        };
        let mut rng2 = LcgRng::new(99);
        let result =
            ResonatorNetwork::decompose(&composite, &[role], &[codebook], &cfg, None, &mut rng2)
                .expect("decompose");

        assert_eq!(result.filler_ids.len(), 1);
        assert_eq!(result.filler_ids[0], 42);
    }

    // ── 2-role decomposition recovers both fillers ──────────────────────────

    #[test]
    fn decompose_two_roles_recovers_fillers() {
        let mut rng = rng();
        let dim = 128;

        let role0 = random_hrr(dim, &mut rng).expect("role0");
        let role1 = random_hrr(dim, &mut rng).expect("role1");
        let filler0 = random_hrr(dim, &mut rng).expect("f0");
        let filler1 = random_hrr(dim, &mut rng).expect("f1");

        let composite = build_composite(
            &[filler0.clone(), filler1.clone()],
            &[role0.clone(), role1.clone()],
        )
        .expect("composite");

        let mut cb0 = HrrItemMemory::new(dim).expect("cb0");
        cb0.insert(0, filler0).expect("insert f0");

        let mut cb1 = HrrItemMemory::new(dim).expect("cb1");
        cb1.insert(1, filler1).expect("insert f1");

        let cfg = ResonatorConfig {
            n_roles: 2,
            max_iter: 100,
            init_random: true,
        };
        let mut rng2 = LcgRng::new(7);
        let result = ResonatorNetwork::decompose(
            &composite,
            &[role0, role1],
            &[cb0, cb1],
            &cfg,
            None,
            &mut rng2,
        )
        .expect("decompose");

        assert_eq!(result.filler_ids[0], 0);
        assert_eq!(result.filler_ids[1], 1);
    }

    // ── Convergence flag ────────────────────────────────────────────────────

    #[test]
    fn decompose_two_roles_converged_true() {
        let mut rng = rng();
        let dim = 128;

        let role0 = random_hrr(dim, &mut rng).expect("role0");
        let role1 = random_hrr(dim, &mut rng).expect("role1");
        let filler0 = random_hrr(dim, &mut rng).expect("f0");
        let filler1 = random_hrr(dim, &mut rng).expect("f1");

        let composite = build_composite(
            &[filler0.clone(), filler1.clone()],
            &[role0.clone(), role1.clone()],
        )
        .expect("composite");

        let mut cb0 = HrrItemMemory::new(dim).expect("cb0");
        cb0.insert(0, filler0).expect("insert f0");
        let mut cb1 = HrrItemMemory::new(dim).expect("cb1");
        cb1.insert(1, filler1).expect("insert f1");

        let cfg = ResonatorConfig {
            n_roles: 2,
            max_iter: 100,
            init_random: true,
        };
        let mut rng2 = LcgRng::new(13);
        let result = ResonatorNetwork::decompose(
            &composite,
            &[role0, role1],
            &[cb0, cb1],
            &cfg,
            None,
            &mut rng2,
        )
        .expect("decompose");

        assert!(result.converged);
    }

    // ── n_iter ≤ max_iter ───────────────────────────────────────────────────

    #[test]
    fn decompose_n_iter_le_max_iter() {
        let mut rng = rng();
        let dim = 64;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler = random_hrr(dim, &mut rng).expect("filler");
        let composite = hrr_bind(&filler, &role).expect("bind");

        let mut cb = HrrItemMemory::new(dim).expect("cb");
        cb.insert(0, filler).expect("insert");

        let cfg = ResonatorConfig {
            n_roles: 1,
            max_iter: 30,
            init_random: true,
        };
        let mut rng2 = LcgRng::new(1);
        let result = ResonatorNetwork::decompose(&composite, &[role], &[cb], &cfg, None, &mut rng2)
            .expect("decompose");

        assert!(result.n_iter <= 30);
    }

    // ── Error: n_roles mismatch (roles.len()) ───────────────────────────────

    #[test]
    fn decompose_roles_len_mismatch_error() {
        let mut rng = rng();
        let dim = 64;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler = random_hrr(dim, &mut rng).expect("filler");
        let composite = hrr_bind(&filler, &role).expect("bind");

        let mut cb = HrrItemMemory::new(dim).expect("cb");
        cb.insert(0, filler).expect("insert");

        let cfg = ResonatorConfig {
            n_roles: 2,
            max_iter: 10,
            init_random: true,
        }; // mismatch
        let mut rng2 = LcgRng::new(2);
        let res = ResonatorNetwork::decompose(
            &composite,
            &[role], // only 1 role, but n_roles=2
            &[cb],
            &cfg,
            None,
            &mut rng2,
        );
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    // ── Error: codebooks.len() mismatch ─────────────────────────────────────

    #[test]
    fn decompose_codebooks_len_mismatch_error() {
        let mut rng = rng();
        let dim = 64;

        let role0 = random_hrr(dim, &mut rng).expect("role0");
        let role1 = random_hrr(dim, &mut rng).expect("role1");
        let filler = random_hrr(dim, &mut rng).expect("filler");
        let composite = hrr_bind(&filler, &role0).expect("bind");

        let mut cb = HrrItemMemory::new(dim).expect("cb");
        cb.insert(0, filler).expect("insert");

        let cfg = ResonatorConfig {
            n_roles: 2,
            max_iter: 10,
            init_random: true,
        };
        let mut rng2 = LcgRng::new(3);
        let res = ResonatorNetwork::decompose(
            &composite,
            &[role0, role1],
            &[cb], // only 1 codebook, but n_roles=2
            &cfg,
            None,
            &mut rng2,
        );
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    // ── Error: empty codebook ───────────────────────────────────────────────

    #[test]
    fn decompose_empty_codebook_error() {
        let mut rng = rng();
        let dim = 64;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler = random_hrr(dim, &mut rng).expect("filler");
        let composite = hrr_bind(&filler, &role).expect("bind");

        let empty_cb = HrrItemMemory::new(dim).expect("empty cb");

        let cfg = ResonatorConfig {
            n_roles: 1,
            max_iter: 10,
            init_random: true,
        };
        let mut rng2 = LcgRng::new(4);
        let res =
            ResonatorNetwork::decompose(&composite, &[role], &[empty_cb], &cfg, None, &mut rng2);
        assert!(matches!(res, Err(HdcError::EmptyItemMemory)));
    }

    // ── Error: wrong composite dim ──────────────────────────────────────────

    #[test]
    fn decompose_wrong_composite_dim_error() {
        let mut rng = rng();
        let dim = 64;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler = random_hrr(dim, &mut rng).expect("filler");

        let mut cb = HrrItemMemory::new(dim).expect("cb");
        cb.insert(0, filler).expect("insert");

        let composite_wrong = vec![0.0f32; 128]; // wrong dim

        let cfg = ResonatorConfig {
            n_roles: 1,
            max_iter: 10,
            init_random: true,
        };
        let mut rng2 = LcgRng::new(5);
        let res =
            ResonatorNetwork::decompose(&composite_wrong, &[role], &[cb], &cfg, None, &mut rng2);
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    // ── update_role output id in codebook ───────────────────────────────────

    #[test]
    fn update_role_id_exists_in_codebook() {
        let mut rng = rng();
        let dim = 64;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler = random_hrr(dim, &mut rng).expect("filler");
        let composite = hrr_bind(&filler, &role).expect("bind");

        let mut cb = HrrItemMemory::new(dim).expect("cb");
        cb.insert(7, filler.clone()).expect("insert");
        let estimate = vec![filler.clone()];

        let (id, _hv) = ResonatorNetwork::update_role(
            &composite,
            0,
            &role,
            std::slice::from_ref(&role),
            &estimate,
            &cb,
        )
        .expect("update_role");

        assert_eq!(id, 7);
    }

    // ── estimates_converged ─────────────────────────────────────────────────

    #[test]
    fn estimates_converged_equal_ids() {
        assert!(ResonatorNetwork::estimates_converged(&[0, 1], &[0, 1]));
    }

    #[test]
    fn estimates_converged_different_ids() {
        assert!(!ResonatorNetwork::estimates_converged(&[0, 1], &[1, 0]));
    }

    // ── 3-role decomposition ─────────────────────────────────────────────────

    #[test]
    fn decompose_three_roles_recovers_fillers() {
        let mut rng = LcgRng::new(0xABCDEF);
        let dim = 256;

        let roles: Vec<Vec<f32>> = (0..3)
            .map(|_| random_hrr(dim, &mut rng).expect("role"))
            .collect();
        let fillers: Vec<Vec<f32>> = (0..3)
            .map(|_| random_hrr(dim, &mut rng).expect("filler"))
            .collect();

        let composite = build_composite(&fillers, &roles).expect("composite");

        let codebooks: Vec<HrrItemMemory> = (0..3)
            .map(|i| {
                let mut cb = HrrItemMemory::new(dim).expect("cb");
                cb.insert(i, fillers[i].clone()).expect("insert");
                cb
            })
            .collect();

        let cfg = ResonatorConfig {
            n_roles: 3,
            max_iter: 200,
            init_random: true,
        };
        let mut rng2 = LcgRng::new(0x9999);
        let result =
            ResonatorNetwork::decompose(&composite, &roles, &codebooks, &cfg, None, &mut rng2)
                .expect("decompose");

        for i in 0..3 {
            assert_eq!(
                result.filler_ids[i], i,
                "role {i}: expected id {i}, got {}",
                result.filler_ids[i]
            );
        }
    }

    // ── max_iter = 0 returns n_iter = 0 ─────────────────────────────────────

    #[test]
    fn decompose_max_iter_zero_returns_early() {
        let mut rng = rng();
        let dim = 64;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler = random_hrr(dim, &mut rng).expect("filler");
        let composite = hrr_bind(&filler, &role).expect("bind");

        let mut cb = HrrItemMemory::new(dim).expect("cb");
        cb.insert(0, filler).expect("insert");

        let cfg = ResonatorConfig {
            n_roles: 1,
            max_iter: 0,
            init_random: true,
        };
        let mut rng2 = LcgRng::new(6);
        let result = ResonatorNetwork::decompose(&composite, &[role], &[cb], &cfg, None, &mut rng2)
            .expect("decompose");

        assert_eq!(result.n_iter, 0);
        assert!(!result.converged);
    }

    // ── Single-item codebook always wins ─────────────────────────────────────

    #[test]
    fn single_item_codebook_always_wins() {
        let mut rng = rng();
        let dim = 64;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler = random_hrr(dim, &mut rng).expect("filler");
        let composite = hrr_bind(&filler, &role).expect("bind");

        let mut cb = HrrItemMemory::new(dim).expect("cb");
        cb.insert(42, filler).expect("insert");

        let cfg = ResonatorConfig {
            n_roles: 1,
            max_iter: 50,
            init_random: true,
        };
        let mut rng2 = LcgRng::new(7);
        let result = ResonatorNetwork::decompose(&composite, &[role], &[cb], &cfg, None, &mut rng2)
            .expect("decompose");

        assert_eq!(result.filler_ids[0], 42);
    }

    // ── init_estimates = Some with correct shape ─────────────────────────────

    #[test]
    fn decompose_init_estimates_some_correct_shape() {
        let mut rng = rng();
        let dim = 64;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler = random_hrr(dim, &mut rng).expect("filler");
        let composite = hrr_bind(&filler, &role).expect("bind");

        let mut cb = HrrItemMemory::new(dim).expect("cb");
        cb.insert(5, filler.clone()).expect("insert");

        let init = vec![filler.clone()];
        let cfg = ResonatorConfig {
            n_roles: 1,
            max_iter: 50,
            init_random: false,
        };
        let mut rng2 = LcgRng::new(8);
        let result =
            ResonatorNetwork::decompose(&composite, &[role], &[cb], &cfg, Some(&init), &mut rng2)
                .expect("decompose");

        assert_eq!(result.filler_ids[0], 5);
    }

    // ── init_estimates = Some with wrong shape ───────────────────────────────

    #[test]
    fn decompose_init_estimates_wrong_shape_error() {
        let mut rng = rng();
        let dim = 64;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler = random_hrr(dim, &mut rng).expect("filler");
        let composite = hrr_bind(&filler, &role).expect("bind");

        let mut cb = HrrItemMemory::new(dim).expect("cb");
        cb.insert(0, filler).expect("insert");

        // Wrong dim in estimate.
        let bad_init = vec![vec![0.0f32; 32]]; // 32 != 64
        let cfg = ResonatorConfig {
            n_roles: 1,
            max_iter: 50,
            init_random: false,
        };
        let mut rng2 = LcgRng::new(9);
        let res = ResonatorNetwork::decompose(
            &composite,
            &[role],
            &[cb],
            &cfg,
            Some(&bad_init),
            &mut rng2,
        );
        assert!(matches!(res, Err(HdcError::DimensionMismatch { .. })));
    }

    // ── Minimum even HRR dimension (dim=4) ───────────────────────────────────

    #[test]
    fn decompose_dim4_minimum_even() {
        let mut rng = LcgRng::new(0xFEED);
        let dim = 4;

        let role = random_hrr(dim, &mut rng).expect("role");
        let filler = random_hrr(dim, &mut rng).expect("filler");
        let composite = hrr_bind(&filler, &role).expect("bind");

        let mut cb = HrrItemMemory::new(dim).expect("cb");
        cb.insert(0, filler).expect("insert");

        let cfg = ResonatorConfig {
            n_roles: 1,
            max_iter: 50,
            init_random: true,
        };
        let mut rng2 = LcgRng::new(11);
        let result = ResonatorNetwork::decompose(&composite, &[role], &[cb], &cfg, None, &mut rng2)
            .expect("decompose dim=4");

        assert_eq!(result.filler_ids.len(), 1);
    }
}
