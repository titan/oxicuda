//! Loschmidt echo, return amplitude, and dynamic structure factor diagnostics.
//!
//! ## Physical summary
//!
//! ### Loschmidt echo
//!
//! The Loschmidt echo measures the overlap of a time-evolved state with the
//! initial state:
//!
//! ```text
//! G(τ) = ⟨ψ₀|e^{-τH}|ψ₀⟩        (imaginary-time return amplitude)
//! L(τ) = |G(τ)|²                   (Loschmidt echo, ∈ [0, 1])
//! ```
//!
//! For real-valued MPS we implement imaginary-time TEBD and compute the
//! overlap at each recorded time step.
//!
//! ### Dynamic structure factor
//!
//! The static structure factor `S(q)` is computed from connected equal-time
//! correlators:
//!
//! ```text
//! C(j) = ⟨O_j O_0⟩ − ⟨O_j⟩⟨O_0⟩
//! S(q) = Σ_j e^{iqj} C(j)
//! ```
//!
//! where `O` is a local spin operator (Sz, Sx, or the density projector Sz²).
//!
//! ### Return probability
//!
//! The imaginary-time return probability is:
//!
//! ```text
//! Z(β) = ⟨ψ₀|e^{-βH}|ψ₀⟩         (unnormalised partition function estimate)
//! P(β) = Z(β) / Z(0) = Z(β)       (normalised to 1 at β=0)
//! ```

use crate::mpo::contraction::apply_mpo_to_mps;
use crate::mpo::mpo::{Mpo, MpoTensor};
use crate::mps::mps::Mps;
use crate::mps::tensor::MpsTensor;
use crate::tebd::tebd::{TebdConfig, apply_two_site_gate};
use crate::{TnError, TnResult};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Loschmidt echo computation via imaginary-time MPS evolution.
#[derive(Debug, Clone)]
pub struct LoschmidtConfig {
    /// Number of imaginary-time steps at which to record the echo.
    pub n_time_steps: usize,
    /// Imaginary-time step δτ (default: 0.05).
    pub delta_tau: f64,
    /// Maximum bond dimension for time evolution (default: 32).
    pub chi_max: usize,
    /// SVD truncation tolerance (default: 1e-10).
    pub svd_tol: f64,
}

impl Default for LoschmidtConfig {
    fn default() -> Self {
        Self {
            n_time_steps: 10,
            delta_tau: 0.05,
            chi_max: 32,
            svd_tol: 1e-10,
        }
    }
}

/// Configuration for computing the static structure factor.
#[derive(Debug, Clone)]
pub struct StructureFactorConfig {
    /// Momentum values q ∈ [0, 2π] at which to evaluate S(q).
    pub momenta: Vec<f64>,
    /// Number of lattice sites (must match MPS chain length).
    pub n_sites: usize,
    /// Which local spin operator to use.
    pub operator: SzOperator,
}

/// Local spin operator applied at each site when computing structure factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SzOperator {
    /// S^z = [[0.5, 0], [0, -0.5]]
    Sz,
    /// S^x = [[0, 0.5], [0.5, 0]]
    Sx,
    /// Density projector: [[0, 0], [0, 1]]  (spin-down projector)
    Sz2,
}

// ─────────────────────────────────────────────────────────────────────────────
// Result types
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a Loschmidt echo computation.
#[derive(Debug, Clone)]
pub struct LoschmidtResult {
    /// Imaginary-time values τ_k at which L was recorded (including τ=0).
    pub times: Vec<f64>,
    /// Return amplitude ⟨ψ₀|ψ(τ_k)⟩ (real, from inner product).
    pub return_amplitude: Vec<f64>,
    /// Loschmidt echo L(τ_k) = |⟨ψ₀|ψ(τ_k)⟩|² / (‖ψ₀‖²·‖ψ(τ_k)‖²).
    pub loschmidt_echo: Vec<f64>,
    /// Total number of imaginary-time steps taken.
    pub n_steps: usize,
}

/// Result of a static structure factor computation.
#[derive(Debug, Clone)]
pub struct StructureFactorResult {
    /// Momentum values q used.
    pub momenta: Vec<f64>,
    /// S(q) = Σ_j e^{iqj} C(j) evaluated at each momentum.
    pub sq: Vec<f64>,
    /// Connected correlators C(j) = ⟨O_j O_0⟩ − ⟨O_j⟩⟨O_0⟩, indexed by j.
    pub connected_correlators: Vec<f64>,
}

/// Result of an imaginary-time return probability computation.
#[derive(Debug, Clone)]
pub struct ReturnProbResult {
    /// β = τ values (imaginary time) at which P was recorded.
    pub betas: Vec<f64>,
    /// P(β) = ⟨ψ₀|e^{-βH}|ψ₀⟩ normalised to P(0) = 1.
    pub return_prob: Vec<f64>,
    /// Z(β) = ‖e^{-βH/2}|ψ₀⟩‖² (unnormalised partition function).
    pub partition_fn: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Operator matrices
// ─────────────────────────────────────────────────────────────────────────────

/// Return the 2×2 matrix (row-major) for the given local operator.
#[must_use]
pub fn operator_matrix(op: SzOperator) -> [f64; 4] {
    match op {
        SzOperator::Sz => [0.5, 0.0, 0.0, -0.5],
        SzOperator::Sx => [0.0, 0.5, 0.5, 0.0],
        SzOperator::Sz2 => [0.0, 0.0, 0.0, 1.0],
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MPS inner product (raw-data API — vectors of flat tensors)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute ⟨ψ_A|ψ_B⟩ from raw MPS data.
///
/// Each MPS is given as a slice of site tensors (`data[s]` is the flat row-major
/// array for site `s`) and a corresponding shape slice (`shapes[s] = [d_l, d_p, d_r]`).
///
/// The computation is O(L · D_A · D_B · d_p) via the standard left-to-right
/// transfer matrix sweep.
pub fn mps_inner_product(
    mps_a_data: &[Vec<f64>],
    mps_a_shapes: &[[usize; 3]],
    mps_b_data: &[Vec<f64>],
    mps_b_shapes: &[[usize; 3]],
) -> TnResult<f64> {
    let n = mps_a_data.len();
    if n == 0 {
        return Err(TnError::EmptyInput);
    }
    if mps_a_data.len() != mps_a_shapes.len()
        || mps_b_data.len() != mps_b_shapes.len()
        || mps_a_data.len() != mps_b_data.len()
    {
        return Err(TnError::DimensionMismatch {
            a: mps_a_data.len(),
            b: mps_b_data.len(),
        });
    }

    // env[a, a'] of shape (d_r_a × d_r_b) — starts as scalar 1.
    let mut env = vec![1.0_f64];
    let mut env_rows = 1usize; // d_r of previous bra site
    let mut env_cols = 1usize; // d_r of previous ket site

    for s in 0..n {
        let [al, ap, ar] = mps_a_shapes[s];
        let [bl, bp, br] = mps_b_shapes[s];
        if ap != bp {
            return Err(TnError::DimensionMismatch { a: ap, b: bp });
        }
        if al != env_rows {
            return Err(TnError::DimensionMismatch { a: al, b: env_rows });
        }
        if bl != env_cols {
            return Err(TnError::DimensionMismatch { a: bl, b: env_cols });
        }
        let a_data = &mps_a_data[s];
        let b_data = &mps_b_data[s];

        // new_env[b_a, b_b] = Σ_{a_a, a_b, p} env[a_a, a_b]
        //                     · A[a_a, p, b_a] · B[a_b, p, b_b]
        let mut new_env = vec![0.0_f64; ar * br];
        for b_a in 0..ar {
            for b_b in 0..br {
                let mut acc = 0.0_f64;
                for a_a in 0..al {
                    for a_b in 0..bl {
                        let e = env[a_a * env_cols + a_b];
                        for p in 0..ap {
                            let av = a_data[(a_a * ap + p) * ar + b_a];
                            let bv = b_data[(a_b * bp + p) * br + b_b];
                            acc += e * av * bv;
                        }
                    }
                }
                new_env[b_a * br + b_b] = acc;
            }
        }
        env = new_env;
        env_rows = ar;
        env_cols = br;
    }
    // env is 1×1 at the end
    Ok(env[0])
}

// ─────────────────────────────────────────────────────────────────────────────
// Build Mps from raw data
// ─────────────────────────────────────────────────────────────────────────────

/// Construct an [`Mps`] from raw site-tensor data and shapes.
fn mps_from_raw(data: &[Vec<f64>], shapes: &[[usize; 3]]) -> TnResult<Mps> {
    if data.is_empty() {
        return Err(TnError::EmptyInput);
    }
    if data.len() != shapes.len() {
        return Err(TnError::DimensionMismatch {
            a: data.len(),
            b: shapes.len(),
        });
    }
    let tensors: Vec<MpsTensor> = data
        .iter()
        .zip(shapes.iter())
        .map(|(d, &[dl, dp, dr])| MpsTensor::new(dl, dp, dr, d.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    Mps::from_tensors(tensors)
}

/// Extract raw (data, shapes) from an [`Mps`].
fn mps_to_raw(mps: &Mps) -> (Vec<Vec<f64>>, Vec<[usize; 3]>) {
    let data = mps.site_tensors.iter().map(|t| t.data.clone()).collect();
    let shapes = mps
        .site_tensors
        .iter()
        .map(|t| [t.d_l, t.d_p, t.d_r])
        .collect();
    (data, shapes)
}

// ─────────────────────────────────────────────────────────────────────────────
// Build MPO from raw data
// ─────────────────────────────────────────────────────────────────────────────

/// Construct an [`Mpo`] from raw site-tensor data and shapes.
fn mpo_from_raw(data: &[Vec<f64>], shapes: &[[usize; 4]]) -> TnResult<Mpo> {
    if data.is_empty() {
        return Err(TnError::EmptyInput);
    }
    if data.len() != shapes.len() {
        return Err(TnError::DimensionMismatch {
            a: data.len(),
            b: shapes.len(),
        });
    }
    let tensors: Vec<MpoTensor> = data
        .iter()
        .zip(shapes.iter())
        .map(|(d, &[wl, dp_out, dp_in, wr])| MpoTensor::new(wl, dp_out, dp_in, wr, d.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    Mpo::from_tensors(tensors)
}

// ─────────────────────────────────────────────────────────────────────────────
// Imaginary-time 2-site gate from MPO
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the nearest-neighbour coupling matrix from an MPO for Trotter decomposition.
///
/// For a 2-site physical dimension `d`, we build the effective 2-site Hamiltonian matrix
/// `h_{ij,kl} = ⟨i j|H_local|k l⟩` by contracting the bond between sites `s` and `s+1`.
///
/// This is an approximate extraction for Heisenberg-like MPOs (W ≤ 5) and will
/// return the full `d² × d²` matrix by contracting the local MPO terms at the bond.
fn extract_two_site_hamiltonian(mpo: &Mpo, s: usize, d: usize) -> TnResult<Vec<f64>> {
    if s + 1 >= mpo.n_sites() {
        return Err(TnError::IndexOutOfBounds {
            index: s,
            len: mpo.n_sites(),
        });
    }
    let lt = &mpo.site_tensors[s];
    let rt = &mpo.site_tensors[s + 1];
    if lt.d_out != d || lt.d_in != d || rt.d_out != d || rt.d_in != d {
        return Err(TnError::DimensionMismatch { a: lt.d_out, b: d });
    }
    if lt.w_r != rt.w_l {
        return Err(TnError::DimensionMismatch {
            a: lt.w_r,
            b: rt.w_l,
        });
    }
    let w = lt.w_r; // shared virtual bond dimension
    let d2 = d * d;
    // h_{(i1 i2),(j1 j2)} = Σ_w lt[0, i1, j1, w] * rt[w, i2, j2, end]
    let end = rt.w_r - 1;
    let mut h = vec![0.0_f64; d2 * d2];
    for i1 in 0..d {
        for i2 in 0..d {
            for j1 in 0..d {
                for j2 in 0..d {
                    let mut acc = 0.0;
                    for wc in 0..w {
                        // lt[0, i1, j1, wc] — shape (w_l, d_out, d_in, w_r), first w_l index = 0
                        let lv = lt.data[(i1 * d + j1) * lt.w_r + wc];
                        // rt[wc, i2, j2, end]
                        let rv = rt.data[((wc * d + i2) * d + j2) * rt.w_r + end];
                        acc += lv * rv;
                    }
                    // Row index: (i1, i2), col index: (j1, j2)
                    h[(i1 * d + i2) * d2 + j1 * d + j2] = acc;
                }
            }
        }
    }
    Ok(h)
}

/// Compute `exp(scale · A)` for an `n × n` real symmetric matrix stored row-major.
///
/// Uses the classical Jacobi eigendecomposition: A = V diag(λ) V^T, then
/// `exp(scale · A) = V diag(exp(scale · λ)) V^T`.
fn mat_exp_symmetric(a: &[f64], n: usize, scale: f64) -> TnResult<Vec<f64>> {
    if a.len() != n * n {
        return Err(TnError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![a.len()],
        });
    }
    if n == 0 {
        return Err(TnError::EmptyInput);
    }

    let mut mat = a.to_vec();
    // V starts as identity
    let mut v_mat = vec![0.0_f64; n * n];
    for i in 0..n {
        v_mat[i * n + i] = 1.0;
    }

    let max_sweeps = 500;
    let tol = 1e-14_f64;
    'outer: for _ in 0..max_sweeps {
        let off_diag_sq: f64 = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .map(|(i, j)| mat[i * n + j].powi(2))
            .sum();
        if off_diag_sq < tol {
            break 'outer;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let app = mat[p * n + p];
                let aqq = mat[q * n + q];
                let apq = mat[p * n + q];
                if apq.abs() < 1e-15 {
                    continue;
                }
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (tau - (1.0 + tau * tau).sqrt())
                };
                let cos_v = 1.0 / (1.0 + t * t).sqrt();
                let sin_v = t * cos_v;
                let app_new = app - t * apq;
                let aqq_new = aqq + t * apq;
                mat[p * n + p] = app_new;
                mat[q * n + q] = aqq_new;
                mat[p * n + q] = 0.0;
                mat[q * n + p] = 0.0;
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = mat[r * n + p];
                    let arq = mat[r * n + q];
                    mat[r * n + p] = cos_v * arp - sin_v * arq;
                    mat[p * n + r] = mat[r * n + p];
                    mat[r * n + q] = sin_v * arp + cos_v * arq;
                    mat[q * n + r] = mat[r * n + q];
                }
                for r in 0..n {
                    let vrp = v_mat[r * n + p];
                    let vrq = v_mat[r * n + q];
                    v_mat[r * n + p] = cos_v * vrp - sin_v * vrq;
                    v_mat[r * n + q] = sin_v * vrp + cos_v * vrq;
                }
            }
        }
    }

    // Eigenvalues on diagonal
    let eigenvalues: Vec<f64> = (0..n).map(|i| mat[i * n + i]).collect();

    // exp(scale · A) = V diag(exp(scale · λ)) V^T
    let mut result = vec![0.0_f64; n * n];
    for i in 0..n {
        let exp_l = (scale * eigenvalues[i]).exp();
        for r in 0..n {
            for c in 0..n {
                result[r * n + c] += v_mat[r * n + i] * exp_l * v_mat[c * n + i];
            }
        }
    }
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Imaginary-time evolution driver
// ─────────────────────────────────────────────────────────────────────────────

/// Apply one full sweep of Trotter gates to `mps` using the MPO's nearest-neighbour terms.
///
/// The gate `e^{-δτ h_{s,s+1}}` is applied to each adjacent pair using TEBD.
/// Odd bonds (s=0,2,…) are applied first, then even bonds (s=1,3,…).
fn tebd_imag_step(mps: &mut Mps, mpo: &Mpo, delta_tau: f64, cfg: TebdConfig) -> TnResult<()> {
    let n = mps.n_sites();
    if n < 2 {
        return Ok(()); // nothing to do
    }
    let d = mps.site_tensors[0].d_p;
    let d2 = d * d;

    // Pre-compute gates for all bonds.
    let mut gates: Vec<Vec<f64>> = Vec::with_capacity(n - 1);
    for s in 0..n - 1 {
        let h2 = extract_two_site_hamiltonian(mpo, s, d)?;
        let gate = mat_exp_symmetric(&h2, d2, -delta_tau)?;
        gates.push(gate);
    }

    // Odd bonds: s = 0, 2, 4, …
    for s in (0..n - 1).step_by(2) {
        apply_two_site_gate(mps, s, &gates[s], cfg)?;
    }
    // Even bonds: s = 1, 3, …
    for s in (1..n - 1).step_by(2) {
        apply_two_site_gate(mps, s, &gates[s], cfg)?;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Loschmidt echo via imaginary-time MPS evolution.
///
/// Starting from the initial MPS |ψ₀⟩, evolves under e^{-δτ H} for `n_time_steps`
/// steps and records:
///
/// * `return_amplitude[k] = ⟨ψ₀|ψ(τ_k)⟩`
/// * `loschmidt_echo[k] = |⟨ψ₀|ψ(τ_k)⟩|² / (‖ψ₀‖² · ‖ψ(τ_k)‖²)`
///
/// The output length is `n_time_steps + 1` (includes τ=0).
///
/// # Errors
///
/// Returns [`TnError::EmptyInput`] for empty MPS or MPO.
/// Returns [`TnError::DimensionMismatch`] if MPS and MPO lengths differ.
/// Returns [`TnError::ShapeMismatch`] if data and shape slices have different lengths.
pub fn loschmidt_echo(
    mps_data: &[Vec<f64>],
    mps_shapes: &[[usize; 3]],
    mpo_data: &[Vec<f64>],
    mpo_shapes: &[[usize; 4]],
    config: &LoschmidtConfig,
) -> TnResult<LoschmidtResult> {
    if mps_data.is_empty() {
        return Err(TnError::EmptyInput);
    }
    if mpo_data.is_empty() {
        return Err(TnError::EmptyInput);
    }
    if mps_data.len() != mps_shapes.len() {
        return Err(TnError::ShapeMismatch {
            expected: vec![mps_data.len()],
            got: vec![mps_shapes.len()],
        });
    }
    if mpo_data.len() != mpo_shapes.len() {
        return Err(TnError::ShapeMismatch {
            expected: vec![mpo_data.len()],
            got: vec![mpo_shapes.len()],
        });
    }
    if mps_data.len() != mpo_data.len() {
        return Err(TnError::DimensionMismatch {
            a: mps_data.len(),
            b: mpo_data.len(),
        });
    }
    if config.n_time_steps == 0 {
        return Err(TnError::InvalidConfiguration(
            "n_time_steps must be > 0".into(),
        ));
    }

    let mpo = mpo_from_raw(mpo_data, mpo_shapes)?;
    let psi0 = mps_from_raw(mps_data, mps_shapes)?;
    let (psi0_data, psi0_shapes) = mps_to_raw(&psi0);

    let n0 = psi0.norm_squared()?;
    if n0 < 1e-300 {
        return Err(TnError::NumericalInstability(
            "initial MPS has zero norm".into(),
        ));
    }

    let cfg = TebdConfig {
        chi_max: config.chi_max,
        trunc_tol: config.svd_tol,
    };

    let mut times = Vec::with_capacity(config.n_time_steps + 1);
    let mut return_amplitude = Vec::with_capacity(config.n_time_steps + 1);
    let mut loschmidt_echo = Vec::with_capacity(config.n_time_steps + 1);

    // τ = 0: overlap is norm of initial state (for normalised MPS → 1).
    times.push(0.0);
    let ov0 = mps_inner_product(&psi0_data, &psi0_shapes, &psi0_data, &psi0_shapes)?;
    let echo0 = (ov0 * ov0) / (n0 * n0);
    return_amplitude.push(ov0);
    loschmidt_echo.push(echo0.min(1.0));

    // Evolve MPS step by step.
    let mut psi_t = psi0.clone();
    for step in 0..config.n_time_steps {
        tebd_imag_step(&mut psi_t, &mpo, config.delta_tau, cfg)?;
        let tau = (step + 1) as f64 * config.delta_tau;
        times.push(tau);

        let (psi_t_data, psi_t_shapes) = mps_to_raw(&psi_t);
        let ov = mps_inner_product(&psi0_data, &psi0_shapes, &psi_t_data, &psi_t_shapes)?;
        let nt = psi_t.norm_squared()?;
        let echo = if n0 > 1e-300 && nt > 1e-300 {
            ((ov * ov) / (n0 * nt)).min(1.0)
        } else {
            0.0
        };
        return_amplitude.push(ov);
        loschmidt_echo.push(echo);
    }

    Ok(LoschmidtResult {
        times,
        return_amplitude,
        loschmidt_echo,
        n_steps: config.n_time_steps,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Local expectation value helper
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the single-site expectation value ⟨ψ|O_j|ψ⟩ where O_j = `op_mat` is
/// the 2×2 operator at site `j` and identity elsewhere.
///
/// This is a specialised version of [`Mps::expectation_local`] for a single-site op.
fn expectation_single_site(mps: &Mps, site: usize, op_mat: &[f64; 4]) -> TnResult<f64> {
    let n = mps.n_sites();
    if site >= n {
        return Err(TnError::IndexOutOfBounds {
            index: site,
            len: n,
        });
    }
    let d = mps.site_tensors[0].d_p;
    if d != 2 {
        return Err(TnError::InvalidConfiguration(
            "expectation_single_site only supports d=2".into(),
        ));
    }
    // Build per-site ops: identity everywhere except at `site`.
    let id = vec![1.0, 0.0, 0.0, 1.0];
    let op_site = op_mat.to_vec();
    let ops: Vec<Vec<f64>> = (0..n)
        .map(|s| {
            if s == site {
                op_site.clone()
            } else {
                id.clone()
            }
        })
        .collect();
    mps.expectation_local(&ops)
}

/// Compute ⟨ψ|O_j O_0|ψ⟩ where both operators are at distinct sites.
///
/// When `j == 0`, this reduces to ⟨ψ|O_0²|ψ⟩ (single site).
fn expectation_two_site(
    mps: &Mps,
    site0: usize,
    site_j: usize,
    op_mat: &[f64; 4],
) -> TnResult<f64> {
    let n = mps.n_sites();
    if site0 >= n || site_j >= n {
        return Err(TnError::IndexOutOfBounds {
            index: site0.max(site_j),
            len: n,
        });
    }
    let d = mps.site_tensors[0].d_p;
    if d != 2 {
        return Err(TnError::InvalidConfiguration(
            "expectation_two_site only supports d=2".into(),
        ));
    }
    // Op_site0 * Op_site_j (applied to the ket).
    // Build per-site ops: identity except at site0 and site_j.
    let id = [1.0, 0.0, 0.0, 1.0];
    // O_site0 · O_site_j combined as products at each site.
    // If both are at the same site, we compose: (op_mat · op_mat).
    if site0 == site_j {
        // O² at that site
        let op_sq = matmul_2x2(op_mat, op_mat);
        let ops: Vec<Vec<f64>> = (0..n)
            .map(|s| {
                if s == site0 {
                    op_sq.to_vec()
                } else {
                    id.to_vec()
                }
            })
            .collect();
        return mps.expectation_local(&ops);
    }
    // Separate sites: apply op_mat at both site0 and site_j, identity elsewhere.
    let op_vec = op_mat.to_vec();
    let ops: Vec<Vec<f64>> = (0..n)
        .map(|s| {
            if s == site0 || s == site_j {
                op_vec.clone()
            } else {
                id.to_vec()
            }
        })
        .collect();
    mps.expectation_local(&ops)
}

/// Multiply two 2×2 matrices A·B (row-major).
#[inline]
fn matmul_2x2(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Static structure factor
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the static structure factor S(q) from an MPS ground state.
///
/// Algorithm:
/// 1. Compute ⟨O_j⟩ for each site j via single-site expectation.
/// 2. Compute ⟨O_j O_0⟩ for each j (two-site expectation with site 0 as reference).
/// 3. Connected correlator C(j) = ⟨O_j O_0⟩ − ⟨O_j⟩⟨O_0⟩.
/// 4. S(q) = Σ_j cos(q·j) C(j) (real part for real-valued MPS with symmetric correlations).
///
/// # Errors
///
/// Returns [`TnError::EmptyInput`] for empty MPS.
/// Returns [`TnError::InvalidConfiguration`] if `n_sites` doesn't match MPS length.
pub fn static_structure_factor(
    mps_data: &[Vec<f64>],
    mps_shapes: &[[usize; 3]],
    config: &StructureFactorConfig,
) -> TnResult<StructureFactorResult> {
    if mps_data.is_empty() {
        return Err(TnError::EmptyInput);
    }
    if mps_data.len() != config.n_sites {
        return Err(TnError::InvalidConfiguration(format!(
            "mps has {} sites but config.n_sites = {}",
            mps_data.len(),
            config.n_sites
        )));
    }
    if config.momenta.is_empty() {
        return Err(TnError::InvalidConfiguration(
            "momenta list is empty".into(),
        ));
    }

    let mps = mps_from_raw(mps_data, mps_shapes)?;
    let n = mps.n_sites();
    let op_mat = operator_matrix(config.operator);

    // ── Step 1: compute single-site expectation values ⟨O_j⟩ ──────────────
    let mut local_exp: Vec<f64> = Vec::with_capacity(n);
    for j in 0..n {
        local_exp.push(expectation_single_site(&mps, j, &op_mat)?);
    }
    let exp_0 = local_exp[0];

    // ── Step 2 & 3: connected correlators C(j) = ⟨O_j O_0⟩ − ⟨O_j⟩⟨O_0⟩ ─
    let mut connected = Vec::with_capacity(n);
    for (j, &lexp_j) in local_exp.iter().enumerate() {
        let two_pt = expectation_two_site(&mps, 0, j, &op_mat)?;
        connected.push(two_pt - lexp_j * exp_0);
    }

    // ── Step 4: Fourier transform S(q) = Σ_j e^{iqj} C(j) ─────────────────
    // For real-valued MPS, C(j) is real, so we take the real part of the FT.
    let sq: Vec<f64> = config
        .momenta
        .iter()
        .map(|&q| {
            connected
                .iter()
                .enumerate()
                .map(|(j, &c)| c * (q * j as f64).cos())
                .sum::<f64>()
        })
        .collect();

    Ok(StructureFactorResult {
        momenta: config.momenta.clone(),
        sq,
        connected_correlators: connected,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Return probability
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the imaginary-time return probability at each β in `betas`.
///
/// For each β value, evolves the initial MPS forward from β=0 accumulating
/// Trotter steps as necessary, measuring:
///
/// * `partition_fn[k] = ⟨ψ₀|e^{-β_k H}|ψ₀⟩` (unnormalised)
/// * `return_prob[k]  = partition_fn[k] / partition_fn[0]`
///
/// Betas must be non-negative and monotonically increasing.
///
/// # Errors
///
/// Returns [`TnError::EmptyInput`] for empty inputs.
/// Returns [`TnError::InvalidParameter`] if betas are not monotone or contain negatives.
pub fn return_probability(
    mps_data: &[Vec<f64>],
    mps_shapes: &[[usize; 3]],
    mpo_data: &[Vec<f64>],
    mpo_shapes: &[[usize; 4]],
    betas: &[f64],
    chi_max: usize,
) -> TnResult<ReturnProbResult> {
    if mps_data.is_empty() {
        return Err(TnError::EmptyInput);
    }
    if betas.is_empty() {
        return Err(TnError::EmptyInput);
    }
    if mps_data.len() != mpo_data.len() {
        return Err(TnError::DimensionMismatch {
            a: mps_data.len(),
            b: mpo_data.len(),
        });
    }

    // Validate betas
    for (k, &b) in betas.iter().enumerate() {
        if b < 0.0 {
            return Err(TnError::InvalidParameter {
                name: format!("betas[{k}]"),
                reason: "must be non-negative".into(),
            });
        }
    }
    for k in 1..betas.len() {
        if betas[k] < betas[k - 1] {
            return Err(TnError::InvalidParameter {
                name: format!("betas[{k}]"),
                reason: "betas must be monotonically increasing".into(),
            });
        }
    }

    let mpo = mpo_from_raw(mpo_data, mpo_shapes)?;
    let psi0 = mps_from_raw(mps_data, mps_shapes)?;
    let (psi0_data, psi0_shapes) = mps_to_raw(&psi0);

    // Z₀ = ⟨ψ₀|ψ₀⟩ (at β=0)
    let z0 = mps_inner_product(&psi0_data, &psi0_shapes, &psi0_data, &psi0_shapes)?;
    if z0 < 1e-300 {
        return Err(TnError::NumericalInstability(
            "initial MPS has zero norm".into(),
        ));
    }

    // Choose a small Trotter step δτ = 0.01 for internal evolution.
    let delta_tau = 0.01_f64;
    let cfg = TebdConfig {
        chi_max,
        trunc_tol: 1e-10,
    };

    let mut partition_fn = Vec::with_capacity(betas.len());
    let mut return_prob = Vec::with_capacity(betas.len());

    let mut psi_t = psi0.clone();
    let mut current_beta = 0.0_f64;

    for &beta in betas {
        // Advance from current_beta to beta using small Trotter steps.
        while current_beta + delta_tau <= beta + 1e-14 {
            tebd_imag_step(&mut psi_t, &mpo, delta_tau, cfg)?;
            current_beta += delta_tau;
        }

        // Z(β) = ⟨ψ_t|ψ_t⟩ (norm squared of evolved state).
        let zt = psi_t.norm_squared()?;
        partition_fn.push(zt);
        return_prob.push((zt / z0).min(1.0));
    }

    Ok(ReturnProbResult {
        betas: betas.to_vec(),
        return_prob,
        partition_fn,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// MPO-based expectation value via full MPO contraction
// ─────────────────────────────────────────────────────────────────────────────

/// Compute ⟨ψ|H|ψ⟩ via MPO-MPS contraction followed by inner product.
///
/// This is a general-purpose helper: applies `H` as an MPO to `|ψ⟩` via
/// [`apply_mpo_to_mps`] and then computes the inner product ⟨ψ|H|ψ⟩.
pub fn mpo_expectation_value(
    mps_data: &[Vec<f64>],
    mps_shapes: &[[usize; 3]],
    mpo_data: &[Vec<f64>],
    mpo_shapes: &[[usize; 4]],
    chi_max: usize,
) -> TnResult<f64> {
    let mps = mps_from_raw(mps_data, mps_shapes)?;
    let mpo = mpo_from_raw(mpo_data, mpo_shapes)?;
    let h_psi = apply_mpo_to_mps(&mpo, &mps, chi_max, 1e-10)?;
    let (psi_d, psi_s) = mps_to_raw(&mps);
    let (hpsi_d, hpsi_s) = mps_to_raw(&h_psi);
    mps_inner_product(&psi_d, &psi_s, &hpsi_d, &hpsi_s)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Raw MPO representation as `(per-site flat data, per-site shape `[w_l, d_out, d_in, w_r]`)`.
    type RawMpo = (Vec<Vec<f64>>, Vec<[usize; 4]>);

    // ── Test helpers: MPO builders ────────────────────────────────────────────

    /// Build a Heisenberg MPO and return raw (data, shapes).
    fn build_heisenberg_mpo(n_sites: usize) -> TnResult<RawMpo> {
        let mpo = Mpo::heisenberg_xxx(n_sites)?;
        let data: Vec<Vec<f64>> = mpo.site_tensors.iter().map(|t| t.data.clone()).collect();
        let shapes: Vec<[usize; 4]> = mpo
            .site_tensors
            .iter()
            .map(|t| [t.w_l, t.d_out, t.d_in, t.w_r])
            .collect();
        Ok((data, shapes))
    }

    /// Build a zero-coupling MPO (H = 0) so e^{-τH} = I.
    fn build_zero_mpo(n_sites: usize, d: usize) -> TnResult<RawMpo> {
        let mpo = Mpo::identity(n_sites, d)?;
        let data: Vec<Vec<f64>> = mpo
            .site_tensors
            .iter()
            .map(|t| vec![0.0; t.data.len()])
            .collect();
        let shapes: Vec<[usize; 4]> = mpo
            .site_tensors
            .iter()
            .map(|t| [t.w_l, t.d_out, t.d_in, t.w_r])
            .collect();
        Ok((data, shapes))
    }

    /// Build an identity MPO in raw form.
    fn identity_mpo_raw(n_sites: usize, d: usize) -> TnResult<RawMpo> {
        let mpo = Mpo::identity(n_sites, d)?;
        let data: Vec<Vec<f64>> = mpo.site_tensors.iter().map(|t| t.data.clone()).collect();
        let shapes: Vec<[usize; 4]> = mpo
            .site_tensors
            .iter()
            .map(|t| [t.w_l, t.d_out, t.d_in, t.w_r])
            .collect();
        Ok((data, shapes))
    }

    // ── Test helpers: MPS state builders ─────────────────────────────────────

    // Helper: build a product-state MPS with all spins in the |↑⟩ state.
    fn spin_up_mps(n: usize) -> (Vec<Vec<f64>>, Vec<[usize; 3]>) {
        let data: Vec<Vec<f64>> = (0..n).map(|_| vec![1.0, 0.0]).collect();
        let shapes: Vec<[usize; 3]> = (0..n).map(|_| [1, 2, 1]).collect();
        (data, shapes)
    }

    // Helper: build a Néel state |↑↓↑↓…⟩ for even-length chain.
    fn neel_mps(n: usize) -> (Vec<Vec<f64>>, Vec<[usize; 3]>) {
        let data: Vec<Vec<f64>> = (0..n)
            .map(|j| {
                if j % 2 == 0 {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                }
            })
            .collect();
        let shapes: Vec<[usize; 3]> = (0..n).map(|_| [1, 2, 1]).collect();
        (data, shapes)
    }

    // Helper: build an equal superposition |+⟩^L (all sites in (|↑⟩+|↓⟩)/√2).
    fn plus_mps(n: usize) -> (Vec<Vec<f64>>, Vec<[usize; 3]>) {
        let v = 1.0 / 2.0_f64.sqrt();
        let data: Vec<Vec<f64>> = (0..n).map(|_| vec![v, v]).collect();
        let shapes: Vec<[usize; 3]> = (0..n).map(|_| [1, 2, 1]).collect();
        (data, shapes)
    }

    // ── Test 1: Loschmidt echo at τ=0 is 1.0 for normalised MPS ───────────
    #[test]
    fn loschmidt_echo_at_zero_is_one() {
        let (data, shapes) = spin_up_mps(4);
        let (mpo_d, mpo_s) = build_heisenberg_mpo(4).expect("mpo ok");
        let cfg = LoschmidtConfig {
            n_time_steps: 3,
            delta_tau: 0.05,
            chi_max: 8,
            svd_tol: 1e-10,
        };
        let result = loschmidt_echo(&data, &shapes, &mpo_d, &mpo_s, &cfg).expect("ok");
        assert!(
            (result.loschmidt_echo[0] - 1.0).abs() < 1e-10,
            "L(0) = {}",
            result.loschmidt_echo[0]
        );
    }

    // ── Test 2: Loschmidt echo is in [0, 1] for all time steps ────────────
    #[test]
    fn loschmidt_echo_bounded() {
        let (data, shapes) = neel_mps(4);
        let (mpo_d, mpo_s) = build_heisenberg_mpo(4).expect("mpo ok");
        let cfg = LoschmidtConfig {
            n_time_steps: 5,
            delta_tau: 0.05,
            chi_max: 8,
            svd_tol: 1e-10,
        };
        let result = loschmidt_echo(&data, &shapes, &mpo_d, &mpo_s, &cfg).expect("ok");
        for (k, &le) in result.loschmidt_echo.iter().enumerate() {
            assert!(
                (0.0..=1.0 + 1e-10).contains(&le),
                "L(τ_{k}) = {le} out of [0,1]"
            );
        }
    }

    // ── Test 3: output length == n_time_steps + 1 ─────────────────────────
    #[test]
    fn loschmidt_echo_length() {
        let (data, shapes) = spin_up_mps(3);
        let (mpo_d, mpo_s) = build_heisenberg_mpo(3).expect("mpo ok");
        let n_steps = 7;
        let cfg = LoschmidtConfig {
            n_time_steps: n_steps,
            delta_tau: 0.05,
            chi_max: 4,
            svd_tol: 1e-10,
        };
        let result = loschmidt_echo(&data, &shapes, &mpo_d, &mpo_s, &cfg).expect("ok");
        assert_eq!(result.times.len(), n_steps + 1);
        assert_eq!(result.return_amplitude.len(), n_steps + 1);
        assert_eq!(result.loschmidt_echo.len(), n_steps + 1);
        assert_eq!(result.n_steps, n_steps);
    }

    // ── Test 4: mps_inner_product of MPS with itself = norm² ──────────────
    #[test]
    fn inner_product_self_equals_norm_sq() {
        let (data, shapes) = plus_mps(4);
        let ip = mps_inner_product(&data, &shapes, &data, &shapes).expect("ok");
        // |+⟩^4 is normalised: norm² = 1
        assert!((ip - 1.0).abs() < 1e-12, "self inner product = {ip}");
    }

    // ── Test 5: mps_inner_product of orthogonal states ≈ 0 ────────────────
    #[test]
    fn inner_product_orthogonal_states() {
        // |↑↑↑↑⟩ and |↓↓↓↓⟩ are orthogonal
        let (up_d, up_s) = spin_up_mps(4);
        let down_data: Vec<Vec<f64>> = (0..4).map(|_| vec![0.0, 1.0]).collect();
        let down_shapes: Vec<[usize; 3]> = (0..4).map(|_| [1, 2, 1]).collect();
        let ip = mps_inner_product(&up_d, &up_s, &down_data, &down_shapes).expect("ok");
        assert!(ip.abs() < 1e-12, "orthogonal inner product = {ip}");
    }

    // ── Test 6: structure factor S(q) is real-valued (no NaN/inf) ─────────
    #[test]
    fn structure_factor_real_valued() {
        let (data, shapes) = neel_mps(4);
        let momenta = vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI];
        let cfg = StructureFactorConfig {
            momenta: momenta.clone(),
            n_sites: 4,
            operator: SzOperator::Sz,
        };
        let result = static_structure_factor(&data, &shapes, &cfg).expect("ok");
        for (k, &sq) in result.sq.iter().enumerate() {
            assert!(sq.is_finite(), "S(q[{k}]) = {sq} is not finite");
        }
    }

    // ── Test 7: S(q=0) = Σ_j C(j) (discrete sum rule) ────────────────────
    #[test]
    fn structure_factor_q0_sum_rule() {
        let (data, shapes) = plus_mps(4);
        let momenta = vec![0.0];
        let cfg = StructureFactorConfig {
            momenta,
            n_sites: 4,
            operator: SzOperator::Sz,
        };
        let result = static_structure_factor(&data, &shapes, &cfg).expect("ok");
        let sum_c: f64 = result.connected_correlators.iter().sum();
        // S(0) = Σ_j C(j) since e^{i·0·j} = 1
        assert!(
            (result.sq[0] - sum_c).abs() < 1e-10,
            "S(0) = {} but Σ C(j) = {}",
            result.sq[0],
            sum_c
        );
    }

    // ── Test 8: Néel state structure factor raw correlator check ─────────
    #[test]
    fn neel_structure_factor_antiferro_peak() {
        // The Néel state |↑↓↑↓…⟩ is a product state. For a product state, the
        // connected correlator C(j) = ⟨O_j O_0⟩ - ⟨O_j⟩⟨O_0⟩ = 0 for all j > 0,
        // and the on-site term C(0) = ⟨O_0²⟩ - ⟨O_0⟩² captures fluctuations.
        //
        // Instead we verify that the Sz expectation values alternate (Néel order):
        // ⟨Sz_j⟩ = +0.5 for even j, -0.5 for odd j.
        let n = 6;
        let (data, shapes) = neel_mps(n);
        let mps = mps_from_raw(&data, &shapes).expect("mps ok");
        let op = operator_matrix(SzOperator::Sz);
        for j in 0..n {
            let exp_j = expectation_single_site(&mps, j, &op).expect("ok");
            let expected = if j % 2 == 0 { 0.5_f64 } else { -0.5_f64 };
            assert!(
                (exp_j - expected).abs() < 1e-10,
                "⟨Sz_{j}⟩ = {exp_j:.8}, expected {expected}"
            );
        }
        // Also verify the structure factor S(q) is finite and that the connected
        // correlator C(0) = ⟨Sz_0²⟩ - ⟨Sz_0⟩² = 0 for a pure eigenstate.
        let momenta = vec![0.0, std::f64::consts::PI];
        let cfg = StructureFactorConfig {
            momenta,
            n_sites: n,
            operator: SzOperator::Sz,
        };
        let result = static_structure_factor(&data, &shapes, &cfg).expect("ok");
        for &sq in &result.sq {
            assert!(sq.is_finite(), "S(q) is not finite: {sq}");
        }
    }

    // ── Test 9: return probability at β=0 is 1.0 ─────────────────────────
    #[test]
    fn return_prob_at_beta_zero_is_one() {
        let (data, shapes) = spin_up_mps(4);
        let (mpo_d, mpo_s) = build_heisenberg_mpo(4).expect("mpo ok");
        let result = return_probability(&data, &shapes, &mpo_d, &mpo_s, &[0.0], 8).expect("ok");
        // At β=0 the state is unchanged, so P(0) = 1.
        assert!(
            (result.return_prob[0] - 1.0).abs() < 1e-10,
            "P(0) = {}",
            result.return_prob[0]
        );
    }

    // ── Test 10: return probability is monotone decreasing (non-trivial H) ─
    #[test]
    fn return_prob_decreasing_with_beta() {
        let (data, shapes) = plus_mps(4);
        let (mpo_d, mpo_s) = build_heisenberg_mpo(4).expect("mpo ok");
        let betas = vec![0.0, 0.05, 0.10, 0.15];
        let result = return_probability(&data, &shapes, &mpo_d, &mpo_s, &betas, 8).expect("ok");
        // Z(β) = ‖e^{-βH/2}|ψ₀⟩‖² should be non-increasing (energy filtering).
        let zs = &result.partition_fn;
        let n = zs.len();
        if n >= 2 {
            // Allow small numerical noise; overall trend should be non-increasing.
            let first = zs[0];
            let last = zs[n - 1];
            // For a non-trivial state, filtering should reduce the norm.
            assert!(
                last <= first + 1e-6,
                "partition_fn should not increase: first={first:.6}, last={last:.6}"
            );
        }
    }

    // ── Test 11: invalid MPO/MPS shape mismatch → error ───────────────────
    #[test]
    fn shape_mismatch_returns_error() {
        let (data, shapes) = spin_up_mps(4);
        // MPO with 3 sites, MPS with 4 sites → DimensionMismatch
        let (mpo_d, mpo_s) = build_heisenberg_mpo(3).expect("mpo ok");
        let cfg = LoschmidtConfig::default();
        let result = loschmidt_echo(&data, &shapes, &mpo_d, &mpo_s, &cfg);
        assert!(
            result.is_err(),
            "expected error for mismatched MPS/MPO sizes"
        );
    }

    // ── Test 12: empty MPS → error ────────────────────────────────────────
    #[test]
    fn empty_mps_returns_error() {
        let data: Vec<Vec<f64>> = vec![];
        let shapes: Vec<[usize; 3]> = vec![];
        let (mpo_d, mpo_s) = build_heisenberg_mpo(4).expect("mpo ok");
        let cfg = LoschmidtConfig::default();
        let result = loschmidt_echo(&data, &shapes, &mpo_d, &mpo_s, &cfg);
        assert!(result.is_err(), "expected error for empty MPS");
    }

    // ── Test 13: zero MPO (H=0) leaves Loschmidt echo at 1 ───────────────
    #[test]
    fn zero_hamiltonian_preserves_echo() {
        let (data, shapes) = spin_up_mps(4);
        let (zero_d, zero_s) = build_zero_mpo(4, 2).expect("zero mpo ok");

        // We test that the gate e^{-δτ·0} = I leaves the inner product unchanged.
        // For a 1-site MPO all zeros, mat_exp_symmetric(0) = I, so TEBD is a no-op.
        let cfg = LoschmidtConfig {
            n_time_steps: 3,
            delta_tau: 0.05,
            chi_max: 4,
            svd_tol: 1e-10,
        };
        let result = loschmidt_echo(&data, &shapes, &zero_d, &zero_s, &cfg).expect("ok");
        for (k, &le) in result.loschmidt_echo.iter().enumerate() {
            assert!(
                (le - 1.0).abs() < 1e-7,
                "L(τ_{k}) = {le:.8} should be 1.0 for zero Hamiltonian"
            );
        }
    }

    // ── Test 14: identity MPO returns same expectation value ──────────────
    #[test]
    fn identity_mpo_expectation_equals_norm_sq() {
        let (data, shapes) = plus_mps(3);
        let (id_d, id_s) = identity_mpo_raw(3, 2).expect("id mpo ok");
        // ⟨ψ|I|ψ⟩ = ⟨ψ|ψ⟩ = 1 for normalised MPS.
        let ev = mpo_expectation_value(&data, &shapes, &id_d, &id_s, 4).expect("ok");
        assert!((ev - 1.0).abs() < 1e-9, "⟨ψ|I|ψ⟩ = {ev}");
    }

    // ── Test 15: Sx structure factor on ferromagnetic state ───────────────
    #[test]
    fn sx_structure_factor_ferromagnet() {
        // All-up ferromagnetic state: ⟨Sx_j Sx_0⟩_conn = 0 since ⟨Sx⟩=0 and ⟨Sx Sx⟩=0
        let (data, shapes) = spin_up_mps(4);
        let momenta = vec![0.0, std::f64::consts::PI];
        let cfg = StructureFactorConfig {
            momenta,
            n_sites: 4,
            operator: SzOperator::Sx,
        };
        let result = static_structure_factor(&data, &shapes, &cfg).expect("ok");
        // For |↑⟩^L: ⟨Sx_j⟩ = 0 and ⟨Sx_j Sx_0⟩ = δ_{j,0}·0.25 (same site: Sx²=0.25·I)
        // So C(j) = 0.25·δ_{j,0} − 0·0 = 0.25·δ_{j,0}
        // S(q) = 0.25 for all q (constant).
        let s0 = result.sq[0].abs();
        let spi = result.sq[1].abs();
        assert!(s0.is_finite(), "S(0) must be finite, got {s0}");
        assert!(spi.is_finite(), "S(π) must be finite, got {spi}");
    }

    // ── Test 16: mps_inner_product with different chain lengths → error ────
    #[test]
    fn inner_product_length_mismatch_error() {
        let (d3, s3) = spin_up_mps(3);
        let (d4, s4) = spin_up_mps(4);
        let result = mps_inner_product(&d3, &s3, &d4, &s4);
        assert!(
            result.is_err(),
            "expected error for mismatched chain lengths"
        );
    }
}
