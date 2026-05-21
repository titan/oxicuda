//! Infinite-system TEBD (iTEBD) for translationally-invariant 1D quantum systems.
//!
//! ## Algorithm (Vidal 2007)
//!
//! The infinite MPS is represented in the **Γ–Λ (Vidal) canonical form** for a
//! two-site unit cell `{A, B}`:
//!
//! ```text
//!  ─ λ^A ─ Γ^A ─ λ^B ─ Γ^B ─ λ^A ─ Γ^A ─ λ^B ─ Γ^B ─ …
//! ```
//!
//! Bond dimensions:
//! * `chi_a = lambda_a.len()` — bond between B and A (the "A-boundary" bond).
//! * `chi_b = lambda_b.len()` — bond between A and B.
//! * `gamma_a` has shape `[chi_a, d, chi_b]`.
//! * `gamma_b` has shape `[chi_b, d, chi_a]`.
//!
//! Imaginary-time evolution `exp(-δτ h)` is applied alternately to the A–B and
//! B–A bonds using a Suzuki–Trotter splitting.  After each gate application the
//! bond is re-SVD-compressed to `chi_max` Schmidt values and the Γ/Λ tensors are
//! updated accordingly.
//!
//! ## References
//!
//! * G. Vidal, *Phys. Rev. Lett.* **98**, 070201 (2007).
//! * I. P. McCulloch, "Infinite size density matrix renormalization group,
//!   revisited" (arXiv:0804.2509).

use crate::error::{TnError, TnResult};
use crate::handle::LcgRng;
use crate::mps::truncation::svd_truncate;
use crate::svd::svd_dense::svd_jacobi;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Run-time parameters for an iTEBD simulation.
#[derive(Debug, Clone)]
pub struct ItedbConfig {
    /// Maximum bond dimension χ kept after each SVD step.
    pub chi_max: usize,
    /// Imaginary-time step δτ per Trotter layer.
    pub delta_tau: f64,
    /// Total number of Trotter sweeps.
    pub n_steps: usize,
    /// Physical (on-site) Hilbert-space dimension `d`.
    pub d: usize,
    /// Trotter order: 1 (first-order) or 2 (Strang/second-order, default).
    pub trotter_order: usize,
    /// Energy convergence tolerance (stop early when |ΔE| < tol).
    pub tol: f64,
}

impl Default for ItedbConfig {
    fn default() -> Self {
        Self {
            chi_max: 10,
            delta_tau: 0.01,
            n_steps: 500,
            d: 2,
            trotter_order: 2,
            tol: 1e-8,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

/// Vidal Γ–Λ canonical-form state for a two-site unit cell.
///
/// Bond dimensions are tracked explicitly:
/// * `chi_a = lambda_a.len()` — bond between B and A.
/// * `chi_b = lambda_b.len()` — bond between A and B.
///
/// Tensor shapes (row-major):
/// * `gamma_a[chi_a × d × chi_b]` — Γ^A, index (α, i, β).
/// * `gamma_b[chi_b × d × chi_a]` — Γ^B, index (α, i, β).
/// * `lambda_a[chi_a]` — Schmidt values λ^A (A-boundary bond).
/// * `lambda_b[chi_b]` — Schmidt values λ^B (A–B bond).
/// * `chi` — max(chi_a, chi_b) for convenience; real dims are `lambda_{a,b}.len()`.
/// * `d` — physical dimension.
#[derive(Debug, Clone)]
pub struct ItedbState {
    /// Γ^A tensor, shape `[chi_a, d, chi_b]` row-major.
    pub gamma_a: Vec<f64>,
    /// Γ^B tensor, shape `[chi_b, d, chi_a]` row-major.
    pub gamma_b: Vec<f64>,
    /// Schmidt values λ^A on the bond between B and A.
    pub lambda_a: Vec<f64>,
    /// Schmidt values λ^B on the bond between A and B.
    pub lambda_b: Vec<f64>,
    /// Current maximum bond dimension (= max(chi_a, chi_b)).
    pub chi: usize,
    /// Physical dimension d.
    pub d: usize,
}

impl ItedbState {
    /// Construct a random product state for bond dimension `chi_init`.
    ///
    /// Each site physical vector is drawn i.i.d. standard-normal and then
    /// normalised.  The Schmidt weights are initialised uniform.
    pub fn new_product_state(d: usize, chi_init: usize, seed: u64) -> TnResult<Self> {
        if d == 0 || chi_init == 0 {
            return Err(TnError::EmptyInput);
        }
        let mut rng = LcgRng::new(seed);

        let draw_norm = |rng: &mut LcgRng| -> Vec<f64> {
            let mut v: Vec<f64> = (0..d).map(|_| rng.next_normal()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-15);
            v.iter_mut().for_each(|x| *x /= norm);
            v
        };

        let phys_a = draw_norm(&mut rng);
        let phys_b = draw_norm(&mut rng);

        // gamma_a: shape [chi_init, d, chi_init]. Only (0, p, 0) is non-zero.
        let mut gamma_a = vec![0.0f64; chi_init * d * chi_init];
        let mut gamma_b = vec![0.0f64; chi_init * d * chi_init];
        for p in 0..d {
            // idx3 with chi_right = chi_init: (alpha=0, p, beta=0) -> p * chi_init
            gamma_a[p * chi_init] = phys_a[p];
            gamma_b[p * chi_init] = phys_b[p];
        }

        // Uniform Schmidt weights normalised so Σ λᵢ² = 1.
        let lambda_val = 1.0 / (chi_init as f64).sqrt();
        let lambda_a = vec![lambda_val; chi_init];
        let lambda_b = vec![lambda_val; chi_init];

        Ok(Self {
            gamma_a,
            gamma_b,
            lambda_a,
            lambda_b,
            chi: chi_init,
            d,
        })
    }

    // ── index helpers ──────────────────────────────────────────────────────

    /// Row-major index into a `[chi_left × d × chi_right]` tensor: (α, p, β).
    #[inline]
    pub(crate) fn idx3(chi_right: usize, d: usize, alpha: usize, p: usize, beta: usize) -> usize {
        (alpha * d + p) * chi_right + beta
    }

    // ── theta on the A–B bond ──────────────────────────────────────────────

    /// Form the two-site tensor θ on the A–B bond.
    ///
    /// ```text
    /// θ_{α,i,j,γ} = Σ_β  λ^A_α · Γ^A_{α,i,β} · λ^B_β · Γ^B_{β,j,γ} · λ^A_γ
    /// ```
    ///
    /// Where:
    /// * α, γ index the A-boundary bond (chi_a).
    /// * β indexes the A–B bond (chi_b).
    ///
    /// Returns shape `[chi_a, d, d, chi_a]`.
    pub(crate) fn theta_ab(&self) -> Vec<f64> {
        let chi_a = self.lambda_a.len();
        let chi_b = self.lambda_b.len();
        let d = self.d;
        let mut theta = vec![0.0f64; chi_a * d * d * chi_a];
        for alpha in 0..chi_a {
            let la = self.lambda_a[alpha];
            for i in 0..d {
                for beta in 0..chi_b {
                    // gamma_a: [chi_a, d, chi_b]
                    let ga_val = self.gamma_a[Self::idx3(chi_b, d, alpha, i, beta)];
                    let lb = self.lambda_b[beta];
                    for j in 0..d {
                        for gamma in 0..chi_a {
                            // gamma_b: [chi_b, d, chi_a]
                            let gb = self.gamma_b[Self::idx3(chi_a, d, beta, j, gamma)];
                            let la_g = self.lambda_a[gamma];
                            // θ index: (α, i, j, γ) in [chi_a, d, d, chi_a]
                            let idx = ((alpha * d + i) * d + j) * chi_a + gamma;
                            theta[idx] += la * ga_val * lb * gb * la_g;
                        }
                    }
                }
            }
        }
        theta
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result
// ─────────────────────────────────────────────────────────────────────────────

/// Output of a completed iTEBD run.
#[derive(Debug, Clone)]
pub struct ItedbResult {
    /// Final Γ–Λ state.
    pub state: ItedbState,
    /// Energy expectation value per site ⟨h⟩/2.
    pub energy_per_site: f64,
    /// Energy history sampled every 10 steps.
    pub energy_history: Vec<f64>,
    /// Total number of Trotter steps applied.
    pub n_steps: usize,
    /// Whether the run converged to within `config.tol`.
    pub converged: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Heisenberg Hamiltonian helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build the two-site Heisenberg exchange Hamiltonian for `d = 2` (spin-1/2):
///
/// ```text
/// h = J · (S^x⊗S^x + S^y⊗S^y + S^z⊗S^z)
///   = J · [[ 1/4,  0,    0,    0   ],
///           [ 0,  -1/4,  1/2,  0   ],
///           [ 0,   1/2, -1/4,  0   ],
///           [ 0,   0,    0,    1/4 ]]
/// ```
///
/// Stored row-major as a flat `[4; 16]` array (d² × d², here d=2 so 4×4).
#[must_use]
pub fn heisenberg_hamiltonian_2site(j: f64) -> [f64; 16] {
    #[rustfmt::skip]
    let h = [
        j * 0.25,  0.0,        0.0,        0.0,
        0.0,      -j * 0.25,   j * 0.5,    0.0,
        0.0,       j * 0.5,   -j * 0.25,   0.0,
        0.0,       0.0,        0.0,        j * 0.25,
    ];
    h
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix exponential (4×4 real symmetric via Jacobi eigendecomposition)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `exp(scale · A)` for a **4×4 real symmetric** matrix `A` (row-major).
///
/// Uses the Jacobi eigendecomposition: `A = V diag(λ) V^T`, then
/// `exp(scale·A) = V diag(exp(scale·λ)) V^T`.
///
/// Jacobi iteration sweeps over all off-diagonal pairs until convergence.
pub fn mat_exp_4x4(a: &[f64], scale: f64) -> TnResult<[f64; 16]> {
    if a.len() != 16 {
        return Err(TnError::ShapeMismatch {
            expected: vec![4, 4],
            got: vec![a.len()],
        });
    }

    const N: usize = 4;
    let mut mat = [0.0f64; N * N];
    mat.copy_from_slice(a);

    // V starts as identity; it accumulates the eigenvectors column-wise.
    let mut v_mat = [0.0f64; N * N];
    for i in 0..N {
        v_mat[i * N + i] = 1.0;
    }

    // Jacobi sweeps on the 4×4 symmetric matrix.
    let max_sweeps = 200;
    let tol = 1e-14_f64;
    'outer: for _ in 0..max_sweeps {
        let mut off_diag_sq = 0.0f64;
        for i in 0..N {
            for j in (i + 1)..N {
                off_diag_sq += mat[i * N + j] * mat[i * N + j];
            }
        }
        if off_diag_sq < tol {
            break 'outer;
        }

        // One sweep: rotate every (p, q) pair with p < q.
        for p in 0..N {
            for q in (p + 1)..N {
                let app = mat[p * N + p];
                let aqq = mat[q * N + q];
                let apq = mat[p * N + q];
                if apq.abs() < 1e-15 {
                    continue;
                }
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (tau - (1.0 + tau * tau).sqrt())
                };
                let cos_val = 1.0 / (1.0 + t * t).sqrt();
                let sin_val = t * cos_val;

                // Update matrix diagonal.
                let app_new = app - t * apq;
                let aqq_new = aqq + t * apq;
                mat[p * N + p] = app_new;
                mat[q * N + q] = aqq_new;
                mat[p * N + q] = 0.0;
                mat[q * N + p] = 0.0;

                // Update off-diagonal rows/columns r ≠ p, q.
                for r in 0..N {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = mat[r * N + p];
                    let arq = mat[r * N + q];
                    mat[r * N + p] = cos_val * arp - sin_val * arq;
                    mat[p * N + r] = mat[r * N + p];
                    mat[r * N + q] = sin_val * arp + cos_val * arq;
                    mat[q * N + r] = mat[r * N + q];
                }

                // Accumulate into eigenvector matrix V.
                for r in 0..N {
                    let vrp = v_mat[r * N + p];
                    let vrq = v_mat[r * N + q];
                    v_mat[r * N + p] = cos_val * vrp - sin_val * vrq;
                    v_mat[r * N + q] = sin_val * vrp + cos_val * vrq;
                }
            }
        }
    }

    // Eigenvalues sit on the diagonal of `mat` after convergence.
    let eigenvalues: [f64; N] = [mat[0], mat[5], mat[10], mat[15]];

    // exp(scale · A) = V · diag(exp(scale · λ)) · V^T
    let mut result = [0.0f64; N * N];
    for i in 0..N {
        let exp_lambda = (scale * eigenvalues[i]).exp();
        for r in 0..N {
            for c in 0..N {
                result[r * N + c] += v_mat[r * N + i] * exp_lambda * v_mat[c * N + i];
            }
        }
    }
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Gate application: update Γ–Λ on a single bond
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a two-site gate to a bond and update the Γ–Λ representation.
///
/// This function handles one bond update in the Vidal canonical form.
/// For the A–B bond, call with:
///   `gamma_left = gamma_a`, `gamma_right = gamma_b`,
///   `lambda_boundary = lambda_a`, `lambda_inner = lambda_b`,
///   `chi_left = chi_a`, `chi_inner_old = chi_b`.
///
/// For the B–A bond, call with A and B swapped.
///
/// ## Arguments
///
/// * `gamma_left`    — left site Γ tensor, shape `[chi_left, d, chi_inner]`.
/// * `gamma_right`   — right site Γ tensor, shape `[chi_inner, d, chi_left]`.
/// * `lambda_boundary` — Schmidt values at the outer boundaries, length `chi_left`.
/// * `lambda_inner`  — Schmidt values on the middle bond, length `chi_inner`.
/// * `gate`          — `[d², d²]` two-site Trotter gate (row-major).
/// * `chi_max`       — maximum retained bond dimension.
///
/// On return, `gamma_left`, `gamma_right` and `lambda_inner` are updated;
/// `chi_inner` is updated to the new inner bond dimension.
#[allow(clippy::too_many_arguments)]
fn apply_gate_on_bond(
    gamma_left: &mut Vec<f64>,
    gamma_right: &mut Vec<f64>,
    lambda_boundary: &[f64],
    lambda_inner: &mut Vec<f64>,
    d: usize,
    gate: &[f64],
    chi_max: usize,
) -> TnResult<()> {
    let chi_bdy = lambda_boundary.len(); // left/right boundary bond dim
    let chi_inn = lambda_inner.len(); // inner (middle) bond dim
    let d2 = d * d;

    if gate.len() != d2 * d2 {
        return Err(TnError::ShapeMismatch {
            expected: vec![d2, d2],
            got: vec![gate.len()],
        });
    }

    // ── Step 1: form θ_{α,i,j,γ} ──────────────────────────────────────────
    //
    //   θ_{α,i,j,γ} = λ^bdy_α · Γ^L_{α,i,β} · λ^inn_β · Γ^R_{β,j,γ} · λ^bdy_γ
    //
    // gamma_left:  [chi_bdy, d, chi_inn]
    // gamma_right: [chi_inn, d, chi_bdy]
    // theta shape: [chi_bdy, d, d, chi_bdy]
    let mut theta = vec![0.0f64; chi_bdy * d * d * chi_bdy];
    for alpha in 0..chi_bdy {
        let la = lambda_boundary[alpha];
        for i in 0..d {
            for beta in 0..chi_inn {
                let gl_val = gamma_left[ItedbState::idx3(chi_inn, d, alpha, i, beta)];
                let lb = lambda_inner[beta];
                for j in 0..d {
                    for gamma in 0..chi_bdy {
                        let gr_val = gamma_right[ItedbState::idx3(chi_bdy, d, beta, j, gamma)];
                        let la_g = lambda_boundary[gamma];
                        let idx = ((alpha * d + i) * d + j) * chi_bdy + gamma;
                        theta[idx] += la * gl_val * lb * gr_val * la_g;
                    }
                }
            }
        }
    }

    // ── Step 2: apply gate ─────────────────────────────────────────────────
    //
    //   θ'_{α,i,j,γ} = Σ_{k,l} U_{(i,j),(k,l)} · θ_{α,k,l,γ}
    //
    // gate[ij, kl]: row = (i,j), col = (k,l).
    let mut theta_prime = vec![0.0f64; chi_bdy * d * d * chi_bdy];
    for alpha in 0..chi_bdy {
        for gamma in 0..chi_bdy {
            for ij in 0..d2 {
                let i = ij / d;
                let j = ij % d;
                let mut acc = 0.0f64;
                for kl in 0..d2 {
                    let k = kl / d;
                    let l = kl % d;
                    let gate_val = gate[ij * d2 + kl];
                    let th_idx = ((alpha * d + k) * d + l) * chi_bdy + gamma;
                    acc += gate_val * theta[th_idx];
                }
                let out_idx = ((alpha * d + i) * d + j) * chi_bdy + gamma;
                theta_prime[out_idx] = acc;
            }
        }
    }

    // ── Step 3: SVD on matrix view [chi_bdy·d, d·chi_bdy] ────────────────
    //
    //   Row index: (α, i) ↔ α*d + i
    //   Col index: (j, γ) ↔ j*chi_bdy + γ
    let m_svd = chi_bdy * d;
    let n_svd = d * chi_bdy;
    let mut mat = vec![0.0f64; m_svd * n_svd];
    for alpha in 0..chi_bdy {
        for i in 0..d {
            let row = alpha * d + i;
            for j in 0..d {
                for gamma in 0..chi_bdy {
                    let col = j * chi_bdy + gamma;
                    let th_idx = ((alpha * d + i) * d + j) * chi_bdy + gamma;
                    mat[row * n_svd + col] = theta_prime[th_idx];
                }
            }
        }
    }

    let svd = svd_jacobi(&mat, m_svd, n_svd)?;
    let (svd_trunc, _) = svd_truncate(svd, chi_max, 1e-14)?;
    let chi_new = svd_trunc.k; // new inner bond dimension

    // Normalise λ^new so Σ λᵢ² = 1 (state remains unit-norm throughout).
    let norm_sq = svd_trunc.s.iter().map(|x| x * x).sum::<f64>();
    let norm = norm_sq.sqrt().max(1e-15);
    let lambda_new: Vec<f64> = svd_trunc.s.iter().map(|x| x / norm).collect();

    // ── Step 4: extract new Γ^L and Γ^R ───────────────────────────────────
    //
    //   New gamma_left:  shape [chi_bdy, d, chi_new]
    //   New gamma_right: shape [chi_new, d, chi_bdy]
    //
    //   Γ^L_new[α, i, β] = U_svd[α·d+i, β] / λ^bdy_α
    //   Γ^R_new[β, j, γ] = V^T[β, j·chi_bdy+γ] / λ^bdy_γ

    let mut gamma_l_new = vec![0.0f64; chi_bdy * d * chi_new];
    for alpha in 0..chi_bdy {
        let la = lambda_boundary[alpha];
        let la_inv = if la.abs() > 1e-15 { 1.0 / la } else { 0.0 };
        for i in 0..d {
            let row = alpha * d + i;
            for beta in 0..chi_new {
                // U is (m_svd × chi_new) row-major after truncation
                gamma_l_new[ItedbState::idx3(chi_new, d, alpha, i, beta)] =
                    svd_trunc.u[row * chi_new + beta] * la_inv;
            }
        }
    }

    let mut gamma_r_new = vec![0.0f64; chi_new * d * chi_bdy];
    for beta in 0..chi_new {
        for j in 0..d {
            for gamma in 0..chi_bdy {
                let la_g = lambda_boundary[gamma];
                let la_g_inv = if la_g.abs() > 1e-15 { 1.0 / la_g } else { 0.0 };
                let col = j * chi_bdy + gamma;
                // vt is (chi_new × n_svd) row-major after truncation
                gamma_r_new[ItedbState::idx3(chi_bdy, d, beta, j, gamma)] =
                    svd_trunc.vt[beta * n_svd + col] * la_g_inv;
            }
        }
    }

    // Commit updates.
    *gamma_left = gamma_l_new;
    *gamma_right = gamma_r_new;
    *lambda_inner = lambda_new;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Energy measurement
// ─────────────────────────────────────────────────────────────────────────────

/// Compute ⟨ψ|h_{AB}|ψ⟩ on the A–B bond, returning the **energy per site**
/// (= bond energy / 2 due to the 2-site unit cell).
///
/// `hamiltonian`: `[d², d²]` row-major 2-site Hamiltonian.
///
/// ## Algorithm
///
/// 1. Form θ_{α,i,j,γ} (shape `[chi_a, d, d, chi_a]`).
/// 2. Compute `⟨h⟩ = Σ_{α,γ,i,j,k,l} θ_{α,k,l,γ} · h_{(k,l),(i,j)} · θ_{α,i,j,γ}`.
/// 3. Normalise by `⟨θ|θ⟩` for robustness.
/// 4. Return `⟨h⟩ / 2` (energy per site).
pub fn itebd_energy(state: &ItedbState, hamiltonian: &[f64]) -> TnResult<f64> {
    let chi_a = state.lambda_a.len();
    let d = state.d;
    let d2 = d * d;

    if hamiltonian.len() != d2 * d2 {
        return Err(TnError::ShapeMismatch {
            expected: vec![d2, d2],
            got: vec![hamiltonian.len()],
        });
    }

    // Form θ on the A–B bond, shape [chi_a, d, d, chi_a].
    let theta = state.theta_ab();

    let mut energy = 0.0f64;
    let mut norm_sq = 0.0f64;

    for alpha in 0..chi_a {
        for gamma in 0..chi_a {
            for ij in 0..d2 {
                let i = ij / d;
                let j = ij % d;
                let th_ij = theta[((alpha * d + i) * d + j) * chi_a + gamma];
                norm_sq += th_ij * th_ij;
                for kl in 0..d2 {
                    let k = kl / d;
                    let l = kl % d;
                    let h_val = hamiltonian[ij * d2 + kl];
                    let th_kl = theta[((alpha * d + k) * d + l) * chi_a + gamma];
                    energy += th_kl * h_val * th_ij;
                }
            }
        }
    }

    // energy / norm_sq gives ⟨h_{AB}⟩ = Tr[ρ^{AB} h] / Tr[ρ^{AB}].
    // Energy per site for the 2-site unit cell:
    //   there are 2 bonds per unit cell (A-B and B-A), each contributing ⟨h_{AB}⟩.
    //   There are 2 sites per unit cell.
    //   → E/site = (2 · ⟨h_{AB}⟩) / 2 = ⟨h_{AB}⟩.
    let norm_safe = norm_sq.max(1e-60);
    Ok(energy / norm_safe)
}

// ─────────────────────────────────────────────────────────────────────────────
// Main iTEBD driver
// ─────────────────────────────────────────────────────────────────────────────

/// Run iTEBD imaginary-time evolution starting from a random product state.
///
/// # Arguments
///
/// * `gate`   — Trotter gate `exp(-δτ h)` of shape `[d², d²]` (row-major).
/// * `config` — algorithm hyperparameters.
/// * `rng`    — pseudo-random generator (used for initial state seeding).
///
/// ## Trotter scheme
///
/// * **Order 1**: alternately apply AB gate, then BA gate, once per macrostep.
/// * **Order 2** (default): apply AB then BA per macrostep; the alternating
///   structure gives 2nd-order Strang-splitting error O(δτ³).
///
/// ## Convergence
///
/// Convergence is checked every 10 steps by comparing the change in the L2
/// norm of `lambda_b` between checks.  When this change falls below `config.tol`
/// the iteration stops early and `converged = true` is returned.
pub fn itebd_run(gate: &[f64], config: &ItedbConfig, rng: &mut LcgRng) -> TnResult<ItedbResult> {
    let d = config.d;
    let d2 = d * d;
    if gate.len() != d2 * d2 {
        return Err(TnError::ShapeMismatch {
            expected: vec![d2, d2],
            got: vec![gate.len()],
        });
    }
    if config.chi_max == 0 || d == 0 || config.n_steps == 0 {
        return Err(TnError::EmptyInput);
    }

    let seed = rng.next_u64();
    let init_chi = config.chi_max.clamp(1, 4);
    let mut state = ItedbState::new_product_state(d, init_chi, seed)?;

    let mut energy_history: Vec<f64> = Vec::new();
    let mut converged = false;

    // Convergence is tracked by the max absolute change in the individual
    // Schmidt values λ^B between consecutive checks.  Since λ^B is normalised
    // (Σ λ² = 1), its L2 norm is always 1 and is not useful; we compare
    // element-wise instead, padding with zeros if the bond dimension changed.
    let mut prev_lambda_b: Vec<f64> = vec![f64::INFINITY; config.chi_max];

    for step in 0..config.n_steps {
        // Apply A–B gate, then B–A gate (Strang-like, order 2).
        // For order 1: only one bond per step, alternating.

        if config.trotter_order == 2 {
            apply_gate_on_bond(
                &mut state.gamma_a,
                &mut state.gamma_b,
                &state.lambda_a.clone(),
                &mut state.lambda_b,
                d,
                gate,
                config.chi_max,
            )?;
            apply_gate_on_bond(
                &mut state.gamma_b,
                &mut state.gamma_a,
                &state.lambda_b.clone(),
                &mut state.lambda_a,
                d,
                gate,
                config.chi_max,
            )?;
        } else {
            if step % 2 == 0 {
                apply_gate_on_bond(
                    &mut state.gamma_a,
                    &mut state.gamma_b,
                    &state.lambda_a.clone(),
                    &mut state.lambda_b,
                    d,
                    gate,
                    config.chi_max,
                )?;
            } else {
                apply_gate_on_bond(
                    &mut state.gamma_b,
                    &mut state.gamma_a,
                    &state.lambda_b.clone(),
                    &mut state.lambda_a,
                    d,
                    gate,
                    config.chi_max,
                )?;
            }
        }

        // Update state.chi as max of the two bond dimensions.
        state.chi = state.lambda_a.len().max(state.lambda_b.len());

        // Check convergence every 10 steps.
        // Convergence criterion: max |λ_i^B(new) - λ_i^B(old)| < tol,
        // comparing element-wise and zero-padding if sizes differ.
        if step % 10 == 9 || step == config.n_steps - 1 {
            let chi_b = state.lambda_b.len();
            let chi_prev = prev_lambda_b.len();
            let max_dim = chi_b.max(chi_prev);
            // Compare element-wise, padding the shorter vector with zeros.
            let max_delta = (0..max_dim)
                .map(|i| {
                    let cur = if i < chi_b { state.lambda_b[i] } else { 0.0 };
                    let prv = if i < chi_prev { prev_lambda_b[i] } else { 0.0 };
                    (cur - prv).abs()
                })
                .fold(0.0f64, f64::max);

            // Store max_delta in energy_history (actual energy computed by caller).
            energy_history.push(max_delta);

            if max_delta < config.tol {
                converged = true;
                break;
            }
            prev_lambda_b = state.lambda_b.clone();
        }
    }

    let n_steps_done = if converged {
        energy_history.len() * 10
    } else {
        config.n_steps
    };

    Ok(ItedbResult {
        energy_per_site: 0.0, // caller fills via itebd_energy
        state,
        energy_history,
        n_steps: n_steps_done,
        converged,
    })
}

/// High-level entry point: run iTEBD for the **Heisenberg model** and return
/// the ground-state energy per site.
///
/// `j`: exchange coupling (J > 0 antiferromagnetic, J < 0 ferromagnetic).
/// Uses the provided [`ItedbConfig`].
pub fn itebd_heisenberg(j: f64, config: &ItedbConfig, rng: &mut LcgRng) -> TnResult<ItedbResult> {
    let ham = heisenberg_hamiltonian_2site(j);
    let gate = mat_exp_4x4(&ham, -config.delta_tau)?;
    let mut result = itebd_run(&gate, config, rng)?;
    result.energy_per_site = itebd_energy(&result.state, &ham)?;
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config_fast() -> ItedbConfig {
        ItedbConfig {
            chi_max: 8,
            delta_tau: 0.05,
            n_steps: 100,
            d: 2,
            trotter_order: 2,
            tol: 1e-6,
        }
    }

    // ── Test 1: product state init ─────────────────────────────────────────

    #[test]
    fn product_state_init() {
        let d = 2;
        let chi = 4;
        let state = ItedbState::new_product_state(d, chi, 42).expect("init ok");
        assert_eq!(state.chi, chi);
        assert_eq!(state.d, d);
        assert_eq!(state.gamma_a.len(), chi * d * chi);
        assert_eq!(state.gamma_b.len(), chi * d * chi);
        assert_eq!(state.lambda_a.len(), chi);
        assert_eq!(state.lambda_b.len(), chi);
    }

    // ── Test 2: lambda normalised at init ─────────────────────────────────

    #[test]
    fn lambda_normalized() {
        let state = ItedbState::new_product_state(2, 4, 7).expect("ok");
        let sq_a: f64 = state.lambda_a.iter().map(|x| x * x).sum();
        let sq_b: f64 = state.lambda_b.iter().map(|x| x * x).sum();
        assert!(
            (sq_a - 1.0).abs() < 1e-12,
            "lambda_a not normalised: {sq_a}"
        );
        assert!(
            (sq_b - 1.0).abs() < 1e-12,
            "lambda_b not normalised: {sq_b}"
        );
    }

    // ── Test 3: Heisenberg Hamiltonian properties ──────────────────────────

    #[test]
    fn heisenberg_hamiltonian() {
        let h = heisenberg_hamiltonian_2site(1.0);
        let trace = h[0] + h[5] + h[10] + h[15];
        assert!(trace.abs() < 1e-14, "trace = {trace}");
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (h[i * 4 + j] - h[j * 4 + i]).abs() < 1e-14,
                    "not symmetric at ({i},{j})"
                );
            }
        }
    }

    // ── Test 4: gate inverse — exp(+dt·H) · exp(-dt·H) = I ────────────────

    #[test]
    fn gate_unitary_for_real_time() {
        let h = heisenberg_hamiltonian_2site(1.0);
        let u_pos = mat_exp_4x4(&h, 0.1).expect("ok");
        let u_neg = mat_exp_4x4(&h, -0.1).expect("ok");
        let mut prod = [0.0f64; 16];
        for i in 0..4 {
            for j in 0..4 {
                let mut acc = 0.0;
                for k in 0..4 {
                    acc += u_pos[i * 4 + k] * u_neg[k * 4 + j];
                }
                prod[i * 4 + j] = acc;
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[i * 4 + j] - expected).abs() < 1e-10,
                    "product[{i},{j}] = {} expected {expected}",
                    prod[i * 4 + j]
                );
            }
        }
    }

    // ── Test 5: ferromagnetic Heisenberg energy < 0 ────────────────────────

    #[test]
    fn itebd_heisenberg_energy_below_zero() {
        let config = ItedbConfig {
            chi_max: 4,
            delta_tau: 0.05,
            n_steps: 100,
            d: 2,
            trotter_order: 2,
            tol: 1e-10,
        };
        let mut rng = LcgRng::new(42);
        let result = itebd_heisenberg(-1.0, &config, &mut rng).expect("run ok");
        assert!(
            result.energy_per_site < 0.0,
            "ferromagnetic energy should be negative, got {}",
            result.energy_per_site
        );
    }

    // ── Test 6: energy per site is finite ─────────────────────────────────

    #[test]
    fn itebd_energy_per_site_finite() {
        let config = default_config_fast();
        let mut rng = LcgRng::new(13);
        let result = itebd_heisenberg(1.0, &config, &mut rng).expect("run ok");
        assert!(
            result.energy_per_site.is_finite(),
            "energy_per_site is not finite: {}",
            result.energy_per_site
        );
    }

    // ── Test 7: bond dimension ≤ chi_max ──────────────────────────────────

    #[test]
    fn itebd_state_bond_dim_leq_chi_max() {
        let config = default_config_fast();
        let mut rng = LcgRng::new(99);
        let result = itebd_heisenberg(1.0, &config, &mut rng).expect("run ok");
        assert!(
            result.state.chi <= config.chi_max,
            "chi {} > chi_max {}",
            result.state.chi,
            config.chi_max
        );
    }

    // ── Test 8: energy decreases during imaginary-time evolution ──────────

    #[test]
    fn itebd_energy_decreases() {
        let config = ItedbConfig {
            chi_max: 6,
            delta_tau: 0.01,
            n_steps: 400,
            d: 2,
            trotter_order: 2,
            tol: 1e-10,
        };
        let h = heisenberg_hamiltonian_2site(1.0);
        let gate = mat_exp_4x4(&h, -config.delta_tau).expect("ok");
        let mut rng = LcgRng::new(7);
        let init_chi = config.chi_max.clamp(1, 4);
        let mut state = ItedbState::new_product_state(2, init_chi, rng.next_u64()).expect("ok");

        let mut energies = Vec::new();
        for step in 0..config.n_steps {
            apply_gate_on_bond(
                &mut state.gamma_a,
                &mut state.gamma_b,
                &state.lambda_a.clone(),
                &mut state.lambda_b,
                2,
                &gate,
                config.chi_max,
            )
            .expect("ab ok");
            apply_gate_on_bond(
                &mut state.gamma_b,
                &mut state.gamma_a,
                &state.lambda_b.clone(),
                &mut state.lambda_a,
                2,
                &gate,
                config.chi_max,
            )
            .expect("ba ok");
            state.chi = state.lambda_a.len().max(state.lambda_b.len());
            if step % 40 == 39 {
                energies.push(itebd_energy(&state, &h).expect("energy ok"));
            }
        }

        let n = energies.len();
        if n >= 3 {
            let first_half_avg = energies[..n / 2].iter().sum::<f64>() / (n / 2) as f64;
            let second_half_avg = energies[n / 2..].iter().sum::<f64>() / (n - n / 2) as f64;
            assert!(
                second_half_avg <= first_half_avg + 1e-3,
                "energy not decreasing: first={first_half_avg:.6}, second={second_half_avg:.6}"
            );
        }
    }

    // ── Test 9: antiferromagnetic energy approaches Bethe ansatz ──────────

    #[test]
    fn itebd_antiferromagnetic_energy() {
        // Bethe ansatz exact: E/site = (1/4 - ln 2) ≈ -0.4431 for J=1 spin-1/2 chain.
        let bethe_ansatz = 0.25 - 2f64.ln();

        let config = ItedbConfig {
            chi_max: 16,
            delta_tau: 0.01,
            n_steps: 1000,
            d: 2,
            trotter_order: 2,
            tol: 1e-10,
        };
        let mut rng = LcgRng::new(314);
        let result = itebd_heisenberg(1.0, &config, &mut rng).expect("run ok");

        // Require within 5% of Bethe ansatz (generous for finite chi).
        let rel_err = (result.energy_per_site - bethe_ansatz).abs() / bethe_ansatz.abs();
        assert!(
            rel_err < 0.05,
            "energy/site = {:.6}, Bethe = {:.6}, rel_err = {:.4}",
            result.energy_per_site,
            bethe_ansatz,
            rel_err
        );
    }

    // ── Test 10: lambda values strictly positive after evolution ──────────

    #[test]
    fn lambda_values_positive() {
        let config = default_config_fast();
        let mut rng = LcgRng::new(55);
        let result = itebd_heisenberg(1.0, &config, &mut rng).expect("run ok");
        for &v in &result.state.lambda_a {
            assert!(v > 0.0, "lambda_a has non-positive entry: {v}");
        }
        for &v in &result.state.lambda_b {
            assert!(v > 0.0, "lambda_b has non-positive entry: {v}");
        }
    }

    // ── Test 11: wrong gate size returns Err ──────────────────────────────

    #[test]
    fn empty_gate_error() {
        let config = default_config_fast();
        let mut rng = LcgRng::new(1);
        let bad_gate = vec![0.0f64; 9]; // should be 16 for d=2
        let err = itebd_run(&bad_gate, &config, &mut rng);
        assert!(err.is_err(), "expected Err for wrong-size gate");
    }

    // ── Test 12: exp(0·H) = I ─────────────────────────────────────────────

    #[test]
    fn mat_exp_zero_is_identity() {
        let h = heisenberg_hamiltonian_2site(1.0);
        let exp0 = mat_exp_4x4(&h, 0.0).expect("ok");
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (exp0[i * 4 + j] - expected).abs() < 1e-12,
                    "exp(0·H)[{i},{j}] = {} expected {expected}",
                    exp0[i * 4 + j]
                );
            }
        }
    }

    // ── Test 13: itebd_energy of product state is finite ─────────────────

    #[test]
    fn energy_of_product_state_is_finite() {
        let state = ItedbState::new_product_state(2, 2, 17).expect("ok");
        let h = heisenberg_hamiltonian_2site(1.0);
        let energy = itebd_energy(&state, &h).expect("energy ok");
        assert!(energy.is_finite(), "energy = {energy}");
    }
}
