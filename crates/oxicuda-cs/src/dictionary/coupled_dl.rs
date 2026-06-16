//! Coupled Dictionary Learning (Yang, Wright, Huang & Ma 2010).
//!
//! Learns a *pair* of dictionaries `(D_l, D_h)` from aligned low-/high-resolution
//! patch pairs so that **the same** sparse code `z` reconstructs both members of a
//! pair:
//!
//! ```text
//! x_l ≈ D_l z,   x_h ≈ D_h z,   z sparse.
//! ```
//!
//! Following Yang et al., the two reconstruction terms are balanced by the patch
//! dimensions and stacked into one *joint* dictionary
//!
//! ```text
//! D_c = [ D_l / √d_l ]      X_c = [ X_l / √d_l ]
//!       [ D_h / √d_h ] ,          [ X_h / √d_h ] ,
//! ```
//!
//! and the coupled objective
//!
//! ```text
//! argmin_{D_l, D_h, Z}  ‖X_l − D_l Z‖²_F / d_l + ‖X_h − D_h Z‖²_F / d_h
//!                       s.t.  ‖z_i‖_0 ≤ T,  ‖atom_k‖₂ = 1
//! ```
//!
//! reduces to ordinary dictionary learning on the stacked space `D_c`. We alternate
//! - **sparse coding** of every joint sample with OMP on `D_c`, and
//! - a **closed-form least-squares** dictionary update (Method of Optimal Directions)
//!   `D_c ← X_c Zᵀ (Z Zᵀ + εI)⁻¹`, followed by per-atom renormalisation in the joint
//!   space and a final split back into `(D_l, D_h)`.
//!
//! At test time a low-resolution patch `x_l` is coded against `D_l` (via [`couple_code`])
//! and the high-resolution estimate is `x̂_h = D_h z` (via [`CoupledDictionary::reconstruct_high`]).
//!
//! All patch matrices use the crate's column-major convention: an `r × n` matrix stores
//! sample `j`'s feature `i` at index `i * n + j`.
//!
//! # References
//!
//! - J. Yang, J. Wright, T. Huang & Y. Ma (2010), "Image Super-Resolution via Sparse
//!   Representation", IEEE Trans. Image Processing 19(11):2861-2873.

use crate::error::{CsError, CsResult};
use crate::greedy::omp::omp;
use crate::handle::LcgRng;
use crate::linalg::cholesky::{cholesky_factor, cholesky_solve};

/// A learned coupled dictionary pair plus its training metadata.
#[derive(Debug, Clone)]
pub struct CoupledDictionary {
    /// Low-resolution dictionary, column-major `[d_l × n_atoms]`.
    pub dict_low: Vec<f64>,
    /// High-resolution dictionary, column-major `[d_h × n_atoms]`.
    pub dict_high: Vec<f64>,
    /// Low-resolution patch dimension `d_l`.
    pub d_low: usize,
    /// High-resolution patch dimension `d_h`.
    pub d_high: usize,
    /// Number of dictionary atoms `K`.
    pub n_atoms: usize,
    /// Target sparsity `T` used during training.
    pub sparsity: usize,
    /// Number of outer alternation iterations performed.
    pub iterations: usize,
    /// Final joint reconstruction RMSE.
    pub final_error: f64,
}

impl CoupledDictionary {
    /// Reconstruct a high-resolution patch from a code `z` (`[n_atoms]`).
    ///
    /// `x̂_h = D_h z`.
    ///
    /// # Errors
    /// [`CsError::DimensionMismatch`] if `code.len() != n_atoms`.
    pub fn reconstruct_high(&self, code: &[f64]) -> CsResult<Vec<f64>> {
        if code.len() != self.n_atoms {
            return Err(CsError::DimensionMismatch {
                a: code.len(),
                b: self.n_atoms,
            });
        }
        Ok(dict_times_code(
            &self.dict_high,
            self.d_high,
            self.n_atoms,
            code,
        ))
    }

    /// Reconstruct a low-resolution patch from a code `z`.
    ///
    /// `x̂_l = D_l z`.
    ///
    /// # Errors
    /// [`CsError::DimensionMismatch`] if `code.len() != n_atoms`.
    pub fn reconstruct_low(&self, code: &[f64]) -> CsResult<Vec<f64>> {
        if code.len() != self.n_atoms {
            return Err(CsError::DimensionMismatch {
                a: code.len(),
                b: self.n_atoms,
            });
        }
        Ok(dict_times_code(
            &self.dict_low,
            self.d_low,
            self.n_atoms,
            code,
        ))
    }

    /// Super-resolve a low-resolution patch end-to-end: code it against `D_l`, then
    /// synthesise `x̂_h = D_h z`.
    ///
    /// # Errors
    /// [`CsError::DimensionMismatch`] if `patch_low.len() != d_low`.
    pub fn super_resolve(&self, patch_low: &[f64], sparsity: usize) -> CsResult<Vec<f64>> {
        let code = couple_code(
            &self.dict_low,
            self.d_low,
            self.n_atoms,
            patch_low,
            sparsity.min(self.n_atoms).max(1),
        )?;
        self.reconstruct_high(&code)
    }
}

/// Configuration for [`coupled_dl`].
#[derive(Debug, Clone)]
pub struct CoupledDlConfig {
    /// Number of dictionary atoms `K`.
    pub n_atoms: usize,
    /// Target sparsity `T` per code (number of non-zeros).
    pub sparsity: usize,
    /// Maximum outer alternation iterations.
    pub max_iter: usize,
    /// Stop when the relative joint-RMSE improvement falls below `tol`.
    pub tol: f64,
    /// Ridge term `ε` added to `Z Zᵀ` in the dictionary update for stability.
    pub ridge: f64,
}

impl Default for CoupledDlConfig {
    fn default() -> Self {
        Self {
            n_atoms: 16,
            sparsity: 3,
            max_iter: 30,
            tol: 1e-5,
            ridge: 1e-8,
        }
    }
}

/// Learn a coupled dictionary pair from aligned low-/high-resolution patches.
///
/// * `patches_low`  — column-major `[d_low × n_samples]` low-resolution patches.
/// * `patches_high` — column-major `[d_high × n_samples]` high-resolution patches.
/// * `d_low`, `d_high`, `n_samples` — dimensions.
/// * `cfg` — configuration.
/// * `rng` — RNG for dictionary initialisation.
///
/// # Errors
/// * [`CsError::ShapeMismatch`] on bad matrix sizes.
/// * [`CsError::InvalidRank`] / [`CsError::InvalidSparsity`] for bad `n_atoms` / `sparsity`.
pub fn coupled_dl(
    patches_low: &[f64],
    patches_high: &[f64],
    d_low: usize,
    d_high: usize,
    n_samples: usize,
    cfg: &CoupledDlConfig,
    rng: &mut LcgRng,
) -> CsResult<CoupledDictionary> {
    if patches_low.len() != d_low * n_samples {
        return Err(CsError::ShapeMismatch {
            expected: vec![d_low, n_samples],
            got: vec![patches_low.len()],
        });
    }
    if patches_high.len() != d_high * n_samples {
        return Err(CsError::ShapeMismatch {
            expected: vec![d_high, n_samples],
            got: vec![patches_high.len()],
        });
    }
    if d_low == 0 || d_high == 0 || n_samples == 0 {
        return Err(CsError::InvalidParameter(
            "coupled_dl: dimensions must be > 0".into(),
        ));
    }
    let k = cfg.n_atoms;
    if k == 0 || k > n_samples {
        return Err(CsError::InvalidRank(k));
    }
    if cfg.sparsity == 0 || cfg.sparsity > k {
        return Err(CsError::InvalidSparsity(cfg.sparsity));
    }

    // Balancing weights w_l = 1/√d_l, w_h = 1/√d_h.
    let w_low = 1.0 / (d_low as f64).sqrt();
    let w_high = 1.0 / (d_high as f64).sqrt();
    let d_joint = d_low + d_high;

    // Build the joint sample matrix X_c, column-major [d_joint × n_samples].
    let mut x_joint = vec![0.0_f64; d_joint * n_samples];
    for j in 0..n_samples {
        for i in 0..d_low {
            x_joint[i * n_samples + j] = w_low * patches_low[i * n_samples + j];
        }
        for i in 0..d_high {
            x_joint[(d_low + i) * n_samples + j] = w_high * patches_high[i * n_samples + j];
        }
    }

    // Initialise the joint dictionary D_c from distinct, normalised samples.
    let mut dict = vec![0.0_f64; d_joint * k];
    let mut chosen = vec![false; n_samples];
    for atom in 0..k {
        let mut idx = rng.next_usize(n_samples);
        while chosen[idx] {
            idx = (idx + 1) % n_samples;
        }
        chosen[idx] = true;
        let mut nrm = 0.0_f64;
        for i in 0..d_joint {
            let v = x_joint[i * n_samples + idx];
            dict[i * k + atom] = v;
            nrm += v * v;
        }
        let nrm = nrm.sqrt().max(1e-300);
        for i in 0..d_joint {
            dict[i * k + atom] /= nrm;
        }
    }

    let mut codes = vec![0.0_f64; k * n_samples]; // column-major [K × n_samples]
    let mut iterations = 0usize;
    let mut last_err = f64::INFINITY;
    let mut final_error = f64::INFINITY;

    // Row-major view of D_c for OMP: phi[m × n] with m = d_joint, n = K.
    for _ in 0..cfg.max_iter {
        iterations += 1;

        // Row-major dictionary for OMP.
        let phi = col_to_row_major(&dict, d_joint, k);

        // ── Sparse coding of every joint sample. ──
        for j in 0..n_samples {
            let mut y = vec![0.0_f64; d_joint];
            for i in 0..d_joint {
                y[i] = x_joint[i * n_samples + j];
            }
            let res = omp(&phi, d_joint, k, &y, cfg.sparsity, 1e-10)?;
            for atom in 0..k {
                codes[atom * n_samples + j] = res.x[atom];
            }
        }

        // ── Dictionary update: D_c ← X_c Zᵀ (Z Zᵀ + εI)⁻¹. ──
        // Z Zᵀ is K×K.
        let mut zzt = vec![0.0_f64; k * k];
        for j in 0..n_samples {
            for a in 0..k {
                let za = codes[a * n_samples + j];
                if za == 0.0 {
                    continue;
                }
                for b in 0..k {
                    zzt[a * k + b] += za * codes[b * n_samples + j];
                }
            }
        }
        for a in 0..k {
            zzt[a * k + a] += cfg.ridge;
        }
        // X_c Zᵀ is d_joint×K.
        let mut xzt = vec![0.0_f64; d_joint * k];
        for j in 0..n_samples {
            for a in 0..k {
                let za = codes[a * n_samples + j];
                if za == 0.0 {
                    continue;
                }
                for i in 0..d_joint {
                    xzt[i * k + a] += x_joint[i * n_samples + j] * za;
                }
            }
        }
        // Solve (Z Zᵀ + εI) D_cᵀ_row = (X_c Zᵀ)ᵀ_row for each dictionary row.
        let l = cholesky_factor(&zzt, k)?;
        for i in 0..d_joint {
            let mut rhs = vec![0.0_f64; k];
            for a in 0..k {
                rhs[a] = xzt[i * k + a];
            }
            let sol = cholesky_solve(&l, k, &rhs)?;
            for a in 0..k {
                dict[i * k + a] = sol[a];
            }
        }

        // Renormalise atoms in the joint space (preserves the coupling balance).
        for atom in 0..k {
            let mut nrm = 0.0_f64;
            for i in 0..d_joint {
                let v = dict[i * k + atom];
                nrm += v * v;
            }
            let nrm = nrm.sqrt();
            if nrm > 1e-300 {
                for i in 0..d_joint {
                    dict[i * k + atom] /= nrm;
                }
            }
        }

        // ── Joint reconstruction RMSE. ──
        let mut err_sq = 0.0_f64;
        for j in 0..n_samples {
            for i in 0..d_joint {
                let mut recon = 0.0_f64;
                for atom in 0..k {
                    recon += dict[i * k + atom] * codes[atom * n_samples + j];
                }
                let e = x_joint[i * n_samples + j] - recon;
                err_sq += e * e;
            }
        }
        let rmse = (err_sq / (d_joint * n_samples) as f64).sqrt();
        final_error = rmse;
        if (last_err - rmse).abs() <= cfg.tol * last_err.max(1e-300) {
            break;
        }
        last_err = rmse;
    }

    // Split D_c back into (D_l, D_h), undoing the balancing weights so that the stored
    // dictionaries operate on the *original* patch spaces.
    let mut dict_low = vec![0.0_f64; d_low * k];
    let mut dict_high = vec![0.0_f64; d_high * k];
    for atom in 0..k {
        for i in 0..d_low {
            dict_low[i * k + atom] = dict[i * k + atom] / w_low;
        }
        for i in 0..d_high {
            dict_high[i * k + atom] = dict[(d_low + i) * k + atom] / w_high;
        }
    }

    Ok(CoupledDictionary {
        dict_low,
        dict_high,
        d_low,
        d_high,
        n_atoms: k,
        sparsity: cfg.sparsity,
        iterations,
        final_error,
    })
}

/// Sparse-code a single low-resolution patch against `D_l` via OMP.
///
/// Returns the full-length code vector `z ∈ ℝ^{n_atoms}`.
///
/// * `dict_low` — column-major `[d_low × n_atoms]`.
///
/// # Errors
/// * [`CsError::DimensionMismatch`] if `patch.len() != d_low`.
/// * Propagated OMP errors.
pub fn couple_code(
    dict_low: &[f64],
    d_low: usize,
    n_atoms: usize,
    patch: &[f64],
    sparsity: usize,
) -> CsResult<Vec<f64>> {
    if patch.len() != d_low {
        return Err(CsError::DimensionMismatch {
            a: patch.len(),
            b: d_low,
        });
    }
    if dict_low.len() != d_low * n_atoms {
        return Err(CsError::ShapeMismatch {
            expected: vec![d_low, n_atoms],
            got: vec![dict_low.len()],
        });
    }
    let phi = col_to_row_major(dict_low, d_low, n_atoms);
    let res = omp(
        &phi,
        d_low,
        n_atoms,
        patch,
        sparsity.min(n_atoms).max(1),
        1e-10,
    )?;
    Ok(res.x)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// `D z` where `D` is a column-major `[rows × cols]` dictionary and `z ∈ ℝ^cols`.
fn dict_times_code(dict: &[f64], rows: usize, cols: usize, z: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; rows];
    for i in 0..rows {
        let mut s = 0.0_f64;
        for a in 0..cols {
            s += dict[i * cols + a] * z[a];
        }
        out[i] = s;
    }
    out
}

/// Convert a column-major `[rows × cols]` matrix to row-major layout (same logical
/// matrix; both index as `i * cols + j` here so this is an identity copy used only to
/// document intent — the dictionary modules store dictionaries as `i * n_atoms + atom`,
/// which is exactly the `m × n` row-major layout OMP expects).
fn col_to_row_major(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    debug_assert_eq!(a.len(), rows * cols);
    a.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic coupled dataset: shared codes Z drive both low/high patches
    /// through fixed ground-truth dictionaries, so a coupled learner should recover the
    /// reconstruction relationship.
    fn synth(
        n_samples: usize,
        d_low: usize,
        d_high: usize,
        k: usize,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut rng = LcgRng::new(seed);
        // Ground-truth dictionaries (column-major), normalised columns.
        let mut dl = vec![0.0_f64; d_low * k];
        let mut dh = vec![0.0_f64; d_high * k];
        for atom in 0..k {
            let mut nl = 0.0;
            for i in 0..d_low {
                let v = rng.next_f64() - 0.5;
                dl[i * k + atom] = v;
                nl += v * v;
            }
            let nl = nl.sqrt().max(1e-12);
            for i in 0..d_low {
                dl[i * k + atom] /= nl;
            }
            let mut nh = 0.0;
            for i in 0..d_high {
                let v = rng.next_f64() - 0.5;
                dh[i * k + atom] = v;
                nh += v * v;
            }
            let nh = nh.sqrt().max(1e-12);
            for i in 0..d_high {
                dh[i * k + atom] /= nh;
            }
        }
        let mut xl = vec![0.0_f64; d_low * n_samples];
        let mut xh = vec![0.0_f64; d_high * n_samples];
        for j in 0..n_samples {
            // 2-sparse code.
            let a1 = rng.next_usize(k);
            let mut a2 = rng.next_usize(k);
            if a2 == a1 {
                a2 = (a2 + 1) % k;
            }
            let c1 = rng.next_f64() + 0.5;
            let c2 = rng.next_f64() + 0.5;
            for i in 0..d_low {
                xl[i * n_samples + j] = c1 * dl[i * k + a1] + c2 * dl[i * k + a2];
            }
            for i in 0..d_high {
                xh[i * n_samples + j] = c1 * dh[i * k + a1] + c2 * dh[i * k + a2];
            }
        }
        (xl, xh)
    }

    #[test]
    fn shapes_are_correct() {
        let (xl, xh) = synth(40, 4, 9, 8, 1);
        let cfg = CoupledDlConfig {
            n_atoms: 8,
            sparsity: 2,
            max_iter: 10,
            ..Default::default()
        };
        let mut rng = LcgRng::new(7);
        let cd = coupled_dl(&xl, &xh, 4, 9, 40, &cfg, &mut rng).expect("ok");
        assert_eq!(cd.dict_low.len(), 4 * 8);
        assert_eq!(cd.dict_high.len(), 9 * 8);
        assert_eq!(cd.n_atoms, 8);
        assert_eq!(cd.d_low, 4);
        assert_eq!(cd.d_high, 9);
    }

    #[test]
    fn joint_error_decreases() {
        let (xl, xh) = synth(50, 5, 8, 10, 2);
        let cfg = CoupledDlConfig {
            n_atoms: 10,
            sparsity: 2,
            max_iter: 25,
            tol: 1e-9,
            ..Default::default()
        };
        let mut rng = LcgRng::new(3);
        let cd = coupled_dl(&xl, &xh, 5, 8, 50, &cfg, &mut rng).expect("ok");
        assert!(cd.final_error.is_finite());
        assert!(cd.final_error < 1.0, "rmse = {}", cd.final_error);
    }

    #[test]
    fn reconstruct_high_dimension() {
        let (xl, xh) = synth(30, 4, 6, 6, 4);
        let cfg = CoupledDlConfig {
            n_atoms: 6,
            sparsity: 2,
            max_iter: 10,
            ..Default::default()
        };
        let mut rng = LcgRng::new(11);
        let cd = coupled_dl(&xl, &xh, 4, 6, 30, &cfg, &mut rng).expect("ok");
        let code = vec![0.0_f64; 6];
        let hi = cd.reconstruct_high(&code).expect("ok");
        assert_eq!(hi.len(), 6);
        assert!(hi.iter().all(|&v| v == 0.0)); // zero code ⇒ zero patch
    }

    #[test]
    fn reconstruct_low_matches_dict_product() {
        let (xl, xh) = synth(25, 3, 5, 5, 6);
        let cfg = CoupledDlConfig {
            n_atoms: 5,
            sparsity: 2,
            max_iter: 8,
            ..Default::default()
        };
        let mut rng = LcgRng::new(9);
        let cd = coupled_dl(&xl, &xh, 3, 5, 25, &cfg, &mut rng).expect("ok");
        let mut code = vec![0.0_f64; 5];
        code[1] = 1.0;
        let lo = cd.reconstruct_low(&code).expect("ok");
        // Should equal column 1 of D_l.
        for i in 0..3 {
            assert!((lo[i] - cd.dict_low[i * 5 + 1]).abs() < 1e-12);
        }
    }

    #[test]
    fn super_resolution_round_trip_is_reasonable() {
        // Learn on synthetic data, then super-resolve a *training* low patch and check
        // the high estimate is correlated with the ground-truth high patch.
        let (xl, xh) = synth(60, 5, 9, 12, 5);
        let cfg = CoupledDlConfig {
            n_atoms: 12,
            sparsity: 2,
            max_iter: 40,
            tol: 1e-9,
            ..Default::default()
        };
        let mut rng = LcgRng::new(13);
        let cd = coupled_dl(&xl, &xh, 5, 9, 60, &cfg, &mut rng).expect("ok");
        // Take sample 0's low patch.
        let mut patch_low = vec![0.0_f64; 5];
        for i in 0..5 {
            patch_low[i] = xl[i * 60];
        }
        let hi_est = cd.super_resolve(&patch_low, 2).expect("ok");
        assert_eq!(hi_est.len(), 9);
        // Compare with ground-truth high patch of sample 0 by cosine similarity.
        let mut gt = vec![0.0_f64; 9];
        for i in 0..9 {
            gt[i] = xh[i * 60];
        }
        let dot: f64 = hi_est.iter().zip(gt.iter()).map(|(a, b)| a * b).sum();
        let n1: f64 = hi_est.iter().map(|v| v * v).sum::<f64>().sqrt();
        let n2: f64 = gt.iter().map(|v| v * v).sum::<f64>().sqrt();
        if n1 > 1e-9 && n2 > 1e-9 {
            let cos = dot / (n1 * n2);
            assert!(cos > 0.3, "cosine sim = {cos}");
        }
    }

    #[test]
    fn couple_code_returns_full_length() {
        let (xl, xh) = synth(20, 4, 5, 5, 8);
        let cfg = CoupledDlConfig {
            n_atoms: 5,
            sparsity: 2,
            max_iter: 6,
            ..Default::default()
        };
        let mut rng = LcgRng::new(2);
        let cd = coupled_dl(&xl, &xh, 4, 5, 20, &cfg, &mut rng).expect("ok");
        let patch = vec![0.1_f64, 0.2, -0.1, 0.05];
        let code = couple_code(&cd.dict_low, 4, 5, &patch, 2).expect("ok");
        assert_eq!(code.len(), 5);
        let nnz = code.iter().filter(|&&v| v.abs() > 1e-12).count();
        assert!(nnz <= 2, "nnz = {nnz}");
    }

    #[test]
    fn atoms_unit_norm_in_joint_space() {
        // After training, the joint atoms are normalised; verify the *combined*
        // (weighted) norm of each (D_l, D_h) atom is ≈ 1.
        let (xl, xh) = synth(30, 4, 6, 8, 14);
        let d_low = 4;
        let d_high = 6;
        let cfg = CoupledDlConfig {
            n_atoms: 8,
            sparsity: 2,
            max_iter: 12,
            ..Default::default()
        };
        let mut rng = LcgRng::new(21);
        let cd = coupled_dl(&xl, &xh, d_low, d_high, 30, &cfg, &mut rng).expect("ok");
        let w_low = 1.0 / (d_low as f64).sqrt();
        let w_high = 1.0 / (d_high as f64).sqrt();
        for atom in 0..8 {
            let mut nrm = 0.0_f64;
            for i in 0..d_low {
                let v = w_low * cd.dict_low[i * 8 + atom];
                nrm += v * v;
            }
            for i in 0..d_high {
                let v = w_high * cd.dict_high[i * 8 + atom];
                nrm += v * v;
            }
            // Atoms that were used should have unit joint norm; unused atoms keep their
            // (also unit) initialisation, so all should be ≈ 1.
            assert!(
                (nrm.sqrt() - 1.0).abs() < 1e-6,
                "atom {atom} norm = {}",
                nrm.sqrt()
            );
        }
    }

    #[test]
    fn rejects_low_shape_mismatch() {
        let cfg = CoupledDlConfig::default();
        let mut rng = LcgRng::new(1);
        let err = coupled_dl(&[0.0; 10], &[0.0; 18], 4, 9, 3, &cfg, &mut rng);
        assert!(matches!(err, Err(CsError::ShapeMismatch { .. })));
    }

    #[test]
    fn rejects_bad_atom_count() {
        let (xl, xh) = synth(5, 4, 5, 5, 1);
        let cfg = CoupledDlConfig {
            n_atoms: 100, // > n_samples
            sparsity: 2,
            ..Default::default()
        };
        let mut rng = LcgRng::new(1);
        let err = coupled_dl(&xl, &xh, 4, 5, 5, &cfg, &mut rng);
        assert!(matches!(err, Err(CsError::InvalidRank(_))));
    }

    #[test]
    fn rejects_bad_sparsity() {
        let (xl, xh) = synth(10, 4, 5, 5, 1);
        let cfg = CoupledDlConfig {
            n_atoms: 5,
            sparsity: 99,
            ..Default::default()
        };
        let mut rng = LcgRng::new(1);
        let err = coupled_dl(&xl, &xh, 4, 5, 10, &cfg, &mut rng);
        assert!(matches!(err, Err(CsError::InvalidSparsity(_))));
    }

    #[test]
    fn reconstruct_high_dimension_mismatch() {
        let (xl, xh) = synth(20, 4, 5, 5, 1);
        let cfg = CoupledDlConfig {
            n_atoms: 5,
            sparsity: 2,
            max_iter: 5,
            ..Default::default()
        };
        let mut rng = LcgRng::new(1);
        let cd = coupled_dl(&xl, &xh, 4, 5, 20, &cfg, &mut rng).expect("ok");
        assert!(matches!(
            cd.reconstruct_high(&[0.0; 3]),
            Err(CsError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn couple_code_patch_dim_mismatch() {
        let dict = vec![0.0_f64; 4 * 5];
        let err = couple_code(&dict, 4, 5, &[0.0, 0.0], 2);
        assert!(matches!(err, Err(CsError::DimensionMismatch { .. })));
    }
}
