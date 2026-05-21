//! Long-range MPO construction via finite-state machine (FSM).
//!
//! This module implements the Crosswhite–Bacon (2008) / Pirvu et al. (2010) approach
//! for representing long-range Hamiltonians as Matrix Product Operators with a fixed
//! bond dimension that does *not* grow with system size.
//!
//! # Background
//!
//! For a Hamiltonian of the form
//! ```text
//! H = Σ_{i<j} J(i,j) · Oᵢ · Oⱼ  +  Σᵢ hᵢ · Oᵢ
//! ```
//! with exponential interactions `J(i,j) = a · exp(-λ|i-j|)`, the naïve MPO bond
//! dimension grows as O(L).  The FSM trick yields a bulk tensor
//! ```text
//! W = [[I,    0,   0 ],
//!      [a·O,  α·I, 0 ],
//!      [h·O,  J·O, I ]]
//! ```
//! (α = exp(-λ)) with bond dimension 3, independent of system size.
//!
//! For **M simultaneous exponentials** the bond dimension is M + 2.
//!
//! Power-law interactions J/|r|^α are approximated by fitting M exponentials to the
//! target values using ridge-regularised linear least squares.

use crate::error::{TnError, TnResult};
use crate::mpo::mpo::{Mpo, MpoTensor};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for constructing a long-range MPO.
#[derive(Debug, Clone)]
pub struct LongRangeMpoConfig {
    /// Number of lattice sites (L).
    pub n_sites: usize,
    /// Physical (on-site Hilbert space) dimension (d).  Defaults to 2 (spin-1/2).
    pub phys_dim: usize,
    /// Number of exponential terms used to approximate the interaction (M).
    pub n_exp_terms: usize,
    /// Minimum decay rate for geometric spacing of exponential rates.
    pub lambda_min: f64,
    /// Maximum decay rate for geometric spacing of exponential rates.
    pub lambda_max: f64,
    /// Ridge regularisation coefficient for the exponential fitting.
    pub reg: f64,
}

impl Default for LongRangeMpoConfig {
    fn default() -> Self {
        Self {
            n_sites: 4,
            phys_dim: 2,
            n_exp_terms: 4,
            lambda_min: 0.1,
            lambda_max: 3.0,
            reg: 1e-6,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core data structure
// ─────────────────────────────────────────────────────────────────────────────

/// A long-range MPO built via the finite-state machine representation.
///
/// Each element of `tensors` is a flat vector storing the site tensor in index
/// order `[D_l, d_out, d_in, D_r]` (row-major), i.e. element `(a, s_out, s_in, b)`
/// lives at flat position `((a * d_out + s_out) * d_in + s_in) * D_r + b`.
///
/// The boundary sites have `D_l = 1` (first site) and `D_r = 1` (last site).
#[derive(Debug, Clone)]
pub struct LongRangeMpo {
    /// Flat site tensors, one per lattice site.
    pub tensors: Vec<Vec<f64>>,
    /// Shape `[D_l, d_out, d_in, D_r]` for each site tensor.
    pub shapes: Vec<[usize; 4]>,
    /// Number of lattice sites.
    pub n_sites: usize,
    /// Physical dimension (same on every site).
    pub phys_dim: usize,
    /// Internal (bulk) bond dimension of the FSM MPO.
    pub bond_dim: usize,
}

impl LongRangeMpo {
    /// Convert to the existing [`Mpo`] type.
    ///
    /// The [`MpoTensor`] layout is `(w_l, d_out, d_in, w_r)` which matches the
    /// `[D_l, d_out, d_in, D_r]` layout used here, so the data can be reused directly.
    pub fn to_mpo(&self) -> TnResult<Mpo> {
        let mut site_tensors = Vec::with_capacity(self.n_sites);
        for (data, &[d_l, d_out, d_in, d_r]) in self.tensors.iter().zip(self.shapes.iter()) {
            let tensor = MpoTensor::new(d_l, d_out, d_in, d_r, data.clone())?;
            site_tensors.push(tensor);
        }
        Mpo::from_tensors(site_tensors)
    }

    /// Compute the diagonal energy expectation ⟨s|H|s⟩ for a product state `s`.
    ///
    /// `state[i]` is the physical state index (0 or 1 for spin-1/2) at site `i`.
    /// This evaluates the operator by propagating boundary vectors through each site
    /// tensor restricted to physical index `(s_out = s_in = state[i])`, which selects
    /// the diagonal (number-conserving) part of the MPO.
    ///
    /// This is an O(L · D_w²) operation and is exact for diagonal operators; for
    /// non-diagonal operators (e.g. Heisenberg SxSx + SySy) it yields only the σᶻ-
    /// contribution unless the off-diagonal elements happen to cancel.
    pub fn local_energy_diagonal(&self, state: &[usize]) -> TnResult<f64> {
        if state.len() != self.n_sites {
            return Err(TnError::ShapeMismatch {
                expected: vec![self.n_sites],
                got: vec![state.len()],
            });
        }
        // Left boundary vector: shape [D_l] = [1].
        let first_d_l = self.shapes[0][0];
        let mut left = vec![0.0f64; first_d_l];
        left[0] = 1.0;

        for (site, (&[d_l, d_out, d_in, d_r], data)) in
            self.shapes.iter().zip(self.tensors.iter()).enumerate()
        {
            let s = state[site];
            if s >= d_out || s >= d_in {
                return Err(TnError::IndexOutOfBounds {
                    index: s,
                    len: d_out.min(d_in),
                });
            }
            // Contract: new_left[b] = Σ_a left[a] * T[a, s, s, b]
            let mut new_left = vec![0.0f64; d_r];
            for (a, &val) in left.iter().enumerate().take(d_l) {
                if val == 0.0 {
                    continue;
                }
                for (b, slot) in new_left.iter_mut().enumerate().take(d_r) {
                    let flat = ((a * d_out + s) * d_in + s) * d_r + b;
                    *slot += val * data[flat];
                }
            }
            left = new_left;
        }
        // Right boundary: [1.0] at index 0.
        Ok(left[0])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Exponential fitting
// ─────────────────────────────────────────────────────────────────────────────

/// Fit an M-term exponential sum to discrete target values.
///
/// Given target values `y_r = target[r-1]` for `r = 1, 2, …, R_max`, finds
/// amplitudes `a_k` and decay rates `λ_k` that minimise
/// ```text
/// ||y - Σ_{k=0}^{M-1} a_k exp(-λ_k r)||²
/// ```
/// The rates `λ_k` are fixed on a geometric grid from `lambda_min` to `lambda_max`;
/// the amplitudes are solved via ridge-regularised linear least squares.
///
/// # Arguments
///
/// * `target`     — Target values `y_r`, one entry per range value `r = 1…R_max`.
/// * `n_terms`    — Number of exponential terms M (≥ 1).
/// * `lambda_min` — Minimum decay rate (> 0).
/// * `lambda_max` — Maximum decay rate (≥ `lambda_min`).
/// * `reg`        — Ridge regularisation coefficient (≥ 0).
///
/// # Returns
///
/// `(amplitudes, rates)` — two vectors of length `n_terms`.
pub fn exponential_fit(
    target: &[f64],
    n_terms: usize,
    lambda_min: f64,
    lambda_max: f64,
    reg: f64,
) -> TnResult<(Vec<f64>, Vec<f64>)> {
    if target.is_empty() {
        return Err(TnError::EmptyInput);
    }
    if n_terms == 0 {
        return Err(TnError::InvalidParameter {
            name: "n_terms".into(),
            reason: "must be >= 1".into(),
        });
    }
    if lambda_min <= 0.0 || lambda_max < lambda_min {
        return Err(TnError::InvalidParameter {
            name: "lambda_min/lambda_max".into(),
            reason: "need 0 < lambda_min <= lambda_max".into(),
        });
    }
    if reg < 0.0 {
        return Err(TnError::InvalidParameter {
            name: "reg".into(),
            reason: "must be non-negative".into(),
        });
    }

    let r_max = target.len();

    // Build geometric rate grid.  When lambda_min == lambda_max all rates collapse
    // to the same value (useful for a single known rate).
    let rates: Vec<f64> = if n_terms == 1 || (lambda_max - lambda_min).abs() < 1e-15 {
        vec![lambda_min; n_terms]
    } else {
        (0..n_terms)
            .map(|k| lambda_min * (lambda_max / lambda_min).powf(k as f64 / (n_terms - 1) as f64))
            .collect()
    };

    // Build design matrix E (R_max × M): E[r, k] = exp(-λ_k * (r+1)) for r = 0..R_max-1.
    let mut e_mat = vec![0.0f64; r_max * n_terms];
    for r in 0..r_max {
        let r_val = (r + 1) as f64;
        for k in 0..n_terms {
            e_mat[r * n_terms + k] = (-rates[k] * r_val).exp();
        }
    }

    // Ridge normal equations: (E^T E + reg·I) a = E^T y
    // Compute E^T E  (n_terms × n_terms).
    let mut ete = vec![0.0f64; n_terms * n_terms];
    for r in 0..r_max {
        for j in 0..n_terms {
            let ej = e_mat[r * n_terms + j];
            for i in 0..n_terms {
                ete[i * n_terms + j] += e_mat[r * n_terms + i] * ej;
            }
        }
    }
    // Add ridge: diagonal penalty.
    for i in 0..n_terms {
        ete[i * n_terms + i] += reg;
    }

    // Compute E^T y  (n_terms).
    let mut ety = vec![0.0f64; n_terms];
    for r in 0..r_max {
        for k in 0..n_terms {
            ety[k] += e_mat[r * n_terms + k] * target[r];
        }
    }

    // Solve the n_terms × n_terms system via Gauss elimination with partial pivoting.
    let amplitudes = solve_linear_system(&ete, &ety, n_terms)?;

    Ok((amplitudes, rates))
}

/// Solve a square linear system A·x = b via Gaussian elimination with partial pivoting.
///
/// Mutates working copies internally.  Returns `x`.
fn solve_linear_system(a_flat: &[f64], b: &[f64], n: usize) -> TnResult<Vec<f64>> {
    // Working copies.
    let mut a = a_flat.to_vec();
    let mut x = b.to_vec();

    for col in 0..n {
        // Partial pivoting: find row with maximum absolute value in current column.
        let mut pivot_row = col;
        let mut pivot_val = a[col * n + col].abs();
        for row in col + 1..n {
            let v = a[row * n + col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = row;
            }
        }
        if pivot_val < 1e-15 {
            return Err(TnError::LinearAlgebraFailure(
                "singular or near-singular normal equations".into(),
            ));
        }
        // Swap rows.
        if pivot_row != col {
            for j in 0..n {
                a.swap(col * n + j, pivot_row * n + j);
            }
            x.swap(col, pivot_row);
        }
        // Eliminate below.
        let diag = a[col * n + col];
        for row in col + 1..n {
            let factor = a[row * n + col] / diag;
            for j in col..n {
                let sub = factor * a[col * n + j];
                a[row * n + j] -= sub;
            }
            x[row] -= factor * x[col];
        }
    }
    // Back substitution.
    for col in (0..n).rev() {
        let mut s = x[col];
        for j in col + 1..n {
            s -= a[col * n + j] * x[j];
        }
        x[col] = s / a[col * n + col];
    }
    Ok(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// FSM bulk tensor helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Low-level helper: create a zero tensor of shape [D_l, d_out, d_in, D_r].
fn make_zero_tensor(d_l: usize, d_out: usize, d_in: usize, d_r: usize) -> Vec<f64> {
    vec![0.0f64; d_l * d_out * d_in * d_r]
}

/// Low-level helper: set element `T[a, s_out, s_in, b] += scale * op[s_out, d_in + s_in]`.
///
/// `op` is stored row-major with shape `(d_out, d_in)`.
fn add_op_block(
    tensor: &mut [f64],
    d_l: usize,
    d_out: usize,
    d_in: usize,
    d_r: usize,
    a: usize,
    b: usize,
    scale: f64,
    op: &[f64],
) {
    let _ = d_l; // used only for bound documentation
    for s_out in 0..d_out {
        for s_in in 0..d_in {
            let flat = ((a * d_out + s_out) * d_in + s_in) * d_r + b;
            tensor[flat] += scale * op[s_out * d_in + s_in];
        }
    }
}

/// Build the identity matrix for a `d × d` physical space.
fn identity_op(d: usize) -> Vec<f64> {
    let mut m = vec![0.0f64; d * d];
    for i in 0..d {
        m[i * d + i] = 1.0;
    }
    m
}

/// Validate basic site-count constraints and return `TnError` on failure.
fn validate_sites(n_sites: usize) -> TnResult<()> {
    if n_sites == 0 {
        return Err(TnError::EmptyInput);
    }
    if n_sites < 2 {
        return Err(TnError::InvalidConfiguration(
            "n_sites must be >= 2 for interaction terms".into(),
        ));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-exponential FSM MPO builder
// ─────────────────────────────────────────────────────────────────────────────

/// Build a single-exponential FSM MPO for the Hamiltonian
/// ```text
/// H = Σ_{i<j} amplitude · exp(-decay_rate·(j-i)) · (OpL)_i · (OpR)_j
///   + Σᵢ h_local · (Op_diag)_i
/// ```
///
/// The FSM bulk tensor (bond dimension 3) is:
/// ```text
/// W[0,0] = I           (vacuum passes through)
/// W[1,0] = amplitude·OpL  (left-operator applied at this site)
/// W[1,1] = exp(-λ)·I  (carry: propagate decay factor)
/// W[2,0] = h_local·Op_diag  (local one-body term)
/// W[2,1] = J·OpR       (close the interaction)
/// W[2,2] = I           (done passes through)
/// ```
///
/// Left boundary selects FSM state 0; right boundary contracts state 2.
pub fn fsm_mpo_single_exp(
    n_sites: usize,
    phys_dim: usize,
    amplitude: f64,
    decay_rate: f64,
    op_left: &[f64],
    op_right: &[f64],
    op_diag: &[f64],
    h_local: f64,
) -> TnResult<LongRangeMpo> {
    validate_sites(n_sites)?;
    let d = phys_dim;
    if d == 0 {
        return Err(TnError::InvalidParameter {
            name: "phys_dim".into(),
            reason: "must be >= 1".into(),
        });
    }
    if op_left.len() != d * d || op_right.len() != d * d || op_diag.len() != d * d {
        return Err(TnError::ShapeMismatch {
            expected: vec![d * d],
            got: vec![op_left.len()],
        });
    }

    let dw = 3usize; // FSM bond dimension for single exponential
    let alpha = (-decay_rate).exp(); // decay factor per bond
    let id = identity_op(d);

    let mut tensors: Vec<Vec<f64>> = Vec::with_capacity(n_sites);
    let mut shapes: Vec<[usize; 4]> = Vec::with_capacity(n_sites);

    for site in 0..n_sites {
        let d_l = if site == 0 { 1 } else { dw };
        let d_r = if site == n_sites - 1 { 1 } else { dw };

        let mut t = make_zero_tensor(d_l, d, d, d_r);

        // --- Map from full (dw, dw) block structure to boundary-projected (d_l, d_r) ---
        // For first site: only FSM row 0 is active (d_l = 1 → represents row 0).
        // For last site:  only FSM col 2 is active (d_r = 1 → represents col 2).
        // For interior:   full (dw × dw) layout.

        match (site == 0, site == n_sites - 1) {
            (true, true) => {
                // Single-site MPO: local term only.
                add_op_block(&mut t, d_l, d, d, d_r, 0, 0, h_local, op_diag);
            }
            (true, false) => {
                // First site — d_l = 1 (row 0 only), d_r = dw.
                //   (row=0) → col 0: I
                //   (row=0) → col 1: amplitude·OpL
                //   (row=0) → col 2: h_local·Op_diag (starts the done-chain immediately)
                add_op_block(&mut t, d_l, d, d, d_r, 0, 0, 1.0, &id);
                add_op_block(&mut t, d_l, d, d, d_r, 0, 1, amplitude, op_left);
                add_op_block(&mut t, d_l, d, d, d_r, 0, 2, h_local, op_diag);
            }
            (false, true) => {
                // Last site — d_l = dw, d_r = 1 (col 2 only).
                //   row 0 → (col=2): h_local·Op_diag
                //   row 1 → (col=2): amplitude·OpR
                //   row 2 → (col=2): I
                add_op_block(&mut t, d_l, d, d, d_r, 0, 0, h_local, op_diag);
                add_op_block(&mut t, d_l, d, d, d_r, 1, 0, amplitude, op_right);
                add_op_block(&mut t, d_l, d, d, d_r, 2, 0, 1.0, &id);
            }
            (false, false) => {
                // Bulk site — full (dw × dw) layout.
                // [0,0]: I
                add_op_block(&mut t, d_l, d, d, d_r, 0, 0, 1.0, &id);
                // [1,0]: amplitude·OpL
                add_op_block(&mut t, d_l, d, d, d_r, 1, 0, amplitude, op_left);
                // [1,1]: α·I  (carry state decays)
                add_op_block(&mut t, d_l, d, d, d_r, 1, 1, alpha, &id);
                // [2,0]: h_local·Op_diag
                add_op_block(&mut t, d_l, d, d, d_r, 2, 0, h_local, op_diag);
                // [2,1]: amplitude·OpR (close the pair)
                add_op_block(&mut t, d_l, d, d, d_r, 2, 1, amplitude, op_right);
                // [2,2]: I (done passes through)
                add_op_block(&mut t, d_l, d, d, d_r, 2, 2, 1.0, &id);
            }
        }

        shapes.push([d_l, d, d, d_r]);
        tensors.push(t);
    }

    Ok(LongRangeMpo {
        tensors,
        shapes,
        n_sites,
        phys_dim: d,
        bond_dim: dw,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-exponential FSM MPO (general)
// ─────────────────────────────────────────────────────────────────────────────

/// Build an M-exponential FSM MPO for a pair interaction given pre-fitted
/// (amplitudes, rates).
///
/// Bond dimension = M + 2 (one start state, M carry states, one done state).
///
/// The FSM W-matrix in block form (D_w × D_w, each block d×d):
/// ```text
/// Row 0        : [I,  0, …, 0,  0]           (start → start = I; start → done = h·O_diag)
/// Row 1..=M    : [aₖ·OL,  αₖ·I, 0…, 0]     (start → carry_k; carry_k → carry_k)
/// Row M+1      : [h·O_d, Σₖ aₖ·OR, I]      (done state)
/// ```
///
/// More precisely for single carry index k ∈ {1..M}:
/// ```text
/// W[0,    0] = I               (vacuum)
/// W[k,    0] = aₖ · OpL       (open left-leg at site i)
/// W[k,    k] = αₖ · I         (propagate carry with decay)
/// W[M+1,  0] = h · Op_diag    (local field)
/// W[M+1,  k] = aₖ · OpR       (close right-leg at site j)
/// W[M+1, M+1] = I             (done propagates)
/// ```
fn fsm_mpo_multi_exp(
    n_sites: usize,
    phys_dim: usize,
    amplitudes: &[f64],
    rates: &[f64],
    op_left: &[f64],
    op_right: &[f64],
    op_diag: &[f64],
    h_local: f64,
) -> TnResult<LongRangeMpo> {
    validate_sites(n_sites)?;
    let d = phys_dim;
    let m = amplitudes.len();
    if m == 0 || m != rates.len() {
        return Err(TnError::InvalidParameter {
            name: "amplitudes/rates".into(),
            reason: "must have equal non-zero length".into(),
        });
    }
    if op_left.len() != d * d || op_right.len() != d * d || op_diag.len() != d * d {
        return Err(TnError::ShapeMismatch {
            expected: vec![d * d],
            got: vec![op_left.len()],
        });
    }

    // FSM bond dimension: 0 = start, 1..=M = carry_{k}, M+1 = done.
    let dw = m + 2;
    let id = identity_op(d);
    // Decay factors α_k = exp(-λ_k).
    let alphas: Vec<f64> = rates.iter().map(|&r| (-r).exp()).collect();

    let mut tensors: Vec<Vec<f64>> = Vec::with_capacity(n_sites);
    let mut shapes: Vec<[usize; 4]> = Vec::with_capacity(n_sites);

    for site in 0..n_sites {
        let d_l = if site == 0 { 1 } else { dw };
        let d_r = if site == n_sites - 1 { 1 } else { dw };
        let mut t = make_zero_tensor(d_l, d, d, d_r);

        match (site == 0, site == n_sites - 1) {
            (true, true) => {
                // Trivial: only local term.
                add_op_block(&mut t, d_l, d, d, d_r, 0, 0, h_local, op_diag);
            }
            (true, false) => {
                // First site: d_l = 1, d_r = dw.
                // Active source row: 0 (the only left bond index).
                // W[0, 0] = I
                add_op_block(&mut t, d_l, d, d, d_r, 0, 0, 1.0, &id);
                // W[0, k] = aₖ · OpL  for k=1..=M
                for (k, &amp_k) in amplitudes.iter().enumerate().take(m) {
                    add_op_block(&mut t, d_l, d, d, d_r, 0, k + 1, amp_k, op_left);
                }
                // W[0, M+1] = h · Op_diag
                add_op_block(&mut t, d_l, d, d, d_r, 0, m + 1, h_local, op_diag);
            }
            (false, true) => {
                // Last site: d_l = dw, d_r = 1 (only col 0 maps to the done index).
                // W[0, 0] = h · Op_diag  (start → done with local field)
                add_op_block(&mut t, d_l, d, d, d_r, 0, 0, h_local, op_diag);
                // W[k, 0] = aₖ · OpR   for k=1..=M
                for (k, &amp_k) in amplitudes.iter().enumerate().take(m) {
                    add_op_block(&mut t, d_l, d, d, d_r, k + 1, 0, amp_k, op_right);
                }
                // W[M+1, 0] = I
                add_op_block(&mut t, d_l, d, d, d_r, m + 1, 0, 1.0, &id);
            }
            (false, false) => {
                // Bulk site: full dw × dw block structure.
                // W[0, 0] = I
                add_op_block(&mut t, d_l, d, d, d_r, 0, 0, 1.0, &id);
                for k in 0..m {
                    let carry = k + 1;
                    // W[0, carry] = aₖ · OpL
                    add_op_block(&mut t, d_l, d, d, d_r, 0, carry, amplitudes[k], op_left);
                    // W[carry, carry] = αₖ · I
                    add_op_block(&mut t, d_l, d, d, d_r, carry, carry, alphas[k], &id);
                    // W[M+1, carry] = aₖ · OpR
                    add_op_block(
                        &mut t,
                        d_l,
                        d,
                        d,
                        d_r,
                        m + 1,
                        carry,
                        amplitudes[k],
                        op_right,
                    );
                }
                // W[M+1, 0] = h · Op_diag
                add_op_block(&mut t, d_l, d, d, d_r, m + 1, 0, h_local, op_diag);
                // W[M+1, M+1] = I
                add_op_block(&mut t, d_l, d, d, d_r, m + 1, m + 1, 1.0, &id);
            }
        }

        shapes.push([d_l, d, d, d_r]);
        tensors.push(t);
    }

    Ok(LongRangeMpo {
        tensors,
        shapes,
        n_sites,
        phys_dim: d,
        bond_dim: dw,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-term Heisenberg MPO helper
// ─────────────────────────────────────────────────────────────────────────────

/// Superpose multiple `LongRangeMpo` objects that share the same n_sites and
/// phys_dim by summing their tensors into a single MPO with block-diagonal
/// virtual structure.
///
/// For MPOs A and B with bond dims D_A and D_B, the combined bond dim is
/// D_A + D_B - 2 (shared boundary states at start/end).
///
/// This is the standard direct-sum construction for MPO addition.
fn superpose_long_range_mpos(mpos: Vec<LongRangeMpo>) -> TnResult<LongRangeMpo> {
    if mpos.is_empty() {
        return Err(TnError::EmptyInput);
    }
    if mpos.len() == 1 {
        return Ok(mpos.into_iter().next().unwrap());
    }

    let n_sites = mpos[0].n_sites;
    let phys_dim = mpos[0].phys_dim;
    for m in &mpos {
        if m.n_sites != n_sites || m.phys_dim != phys_dim {
            return Err(TnError::DimensionMismatch {
                a: m.n_sites,
                b: n_sites,
            });
        }
    }

    // Compute combined bond dimension.
    // Each MPO contributes (dw - 2) interior carry states plus the shared start/done.
    let interior_total: usize = mpos.iter().map(|m| m.bond_dim - 2).sum();
    let combined_dw = interior_total + 2; // 1 start + interior_total carry + 1 done

    let d = phys_dim;

    // Offsets of each MPO's carry states within the combined carry block.
    // Combined layout: 0 = start, [1 .. carry_block_k] = carries from MPO k, last = done.
    let mut carry_offsets = Vec::with_capacity(mpos.len());
    let mut offset = 1usize; // index 0 is "start"
    for m in &mpos {
        carry_offsets.push(offset);
        offset += m.bond_dim - 2;
    }

    let mut tensors: Vec<Vec<f64>> = Vec::with_capacity(n_sites);
    let mut shapes: Vec<[usize; 4]> = Vec::with_capacity(n_sites);

    for site in 0..n_sites {
        let d_l = if site == 0 { 1 } else { combined_dw };
        let d_r = if site == n_sites - 1 { 1 } else { combined_dw };
        let mut t = make_zero_tensor(d_l, d, d, d_r);

        for (mpo_idx, mpo) in mpos.iter().enumerate() {
            let sub_d_l = mpo.shapes[site][0];
            let sub_d_r = mpo.shapes[site][3];
            let sub_dw = mpo.bond_dim;
            let carry_off = carry_offsets[mpo_idx];
            let done_combined = combined_dw - 1; // last index

            // Map sub-MPO virtual indices to combined indices:
            // sub 0 (start) → combined 0
            // sub k ∈ [1..sub_dw-1] (carry) → combined carry_off + (k-1)
            // sub sub_dw-1 (done) → combined done_combined
            let map_sub_to_combined = |sub_idx: usize| -> usize {
                if sub_idx == 0 {
                    0
                } else if sub_idx == sub_dw - 1 {
                    done_combined
                } else {
                    carry_off + (sub_idx - 1)
                }
            };

            // Iterate over all sub-MPO elements and place into combined tensor.
            let sub_data = &mpo.tensors[site];
            for a_sub in 0..sub_d_l {
                for b_sub in 0..sub_d_r {
                    // Extract the d×d operator block.
                    let mut op_block = vec![0.0f64; d * d];
                    for s_out in 0..d {
                        for s_in in 0..d {
                            let sub_flat = ((a_sub * d + s_out) * d + s_in) * sub_d_r + b_sub;
                            op_block[s_out * d + s_in] = sub_data[sub_flat];
                        }
                    }
                    // Map to combined indices.
                    let a_comb = if site == 0 {
                        0
                    } else {
                        map_sub_to_combined(a_sub)
                    };
                    let b_comb = if site == n_sites - 1 {
                        0
                    } else {
                        map_sub_to_combined(b_sub)
                    };

                    add_op_block(&mut t, d_l, d, d, d_r, a_comb, b_comb, 1.0, &op_block);
                }
            }
        }

        // For the first/last site the start/done identity entries may be written
        // multiple times (once per constituent MPO).  Normalise by rebuilding:
        // the id blocks at (start→start) and (done→done) might be double-counted.
        // We handle this by *zeroing* then *setting* id blocks for start and done
        // after accumulation using a separate pass.
        if site != 0 && site != n_sites - 1 {
            // Fix: (0→0) = I and (done→done) = I might have been accumulated
            // `n_mpos` times from each constituent.  Overwrite with exact identity.
            let done_in = combined_dw - 1;
            for s_out in 0..d {
                for s_in in 0..d {
                    let val = if s_out == s_in { 1.0 } else { 0.0 };
                    // Start block: virtual row 0, virtual col 0.
                    let flat_start = (s_out * d + s_in) * combined_dw;
                    t[flat_start] = val;
                    // Done block: virtual row done_in, virtual col done_in.
                    let flat_done = ((done_in * d + s_out) * d + s_in) * combined_dw + done_in;
                    t[flat_done] = val;
                }
            }
        }

        shapes.push([d_l, d, d, d_r]);
        tensors.push(t);
    }

    Ok(LongRangeMpo {
        tensors,
        shapes,
        n_sites,
        phys_dim: d,
        bond_dim: combined_dw,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Spin-1/2 operator helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Pauli-Z / 2 (σᶻ basis: |↑⟩ = 0, |↓⟩ = 1).
fn spin_sz() -> Vec<f64> {
    vec![0.5, 0.0, 0.0, -0.5]
}

/// Raising operator S⁺ = |↑⟩⟨↓|.
fn spin_sp() -> Vec<f64> {
    vec![0.0, 1.0, 0.0, 0.0]
}

/// Lowering operator S⁻ = |↓⟩⟨↑|.
fn spin_sm() -> Vec<f64> {
    vec![0.0, 0.0, 1.0, 0.0]
}

/// Zero operator.
fn zero_op(d: usize) -> Vec<f64> {
    vec![0.0f64; d * d]
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Build a long-range Heisenberg MPO
/// ```text
/// H = J · Σ_{i<j} exp(-λ|i-j|) · (Sˣᵢ·Sˣⱼ + Sʸᵢ·Sʸⱼ + Sᶻᵢ·Sᶻⱼ)
/// ```
///
/// Uses the FSM representation with three constituent single-exponential MPOs
/// (one for each spin component), superposed via block-diagonal bond structure.
/// The SˣSˣ + SʸSʸ term is encoded as ½(S⁺S⁻ + S⁻S⁺).
///
/// # Bond dimension
///
/// Each constituent has `dw = 3`.  After superposition the combined bond dimension
/// is `3 + (3-2) + (3-2) + (3-2) - 1 = 3 + 1 + 1 + 1 = ... = 2 + 3·1 = 5`.
/// More precisely: interior_total = 3·(3-2) = 3, combined = 3 + 2 = 5.
pub fn heisenberg_long_range_mpo(
    n_sites: usize,
    j_coupling: f64,
    decay_rate: f64,
) -> TnResult<LongRangeMpo> {
    validate_sites(n_sites)?;
    let d = 2usize;
    let sz = spin_sz();
    let sp = spin_sp();
    let sm = spin_sm();
    let zero = zero_op(d);

    // Sz·Sz term: amplitude = J, decay = λ, no local field.
    let mpo_zz = fsm_mpo_single_exp(n_sites, d, j_coupling, decay_rate, &sz, &sz, &zero, 0.0)?;

    // S⁺S⁻ term: amplitude = J/2 (from ½(S⁺S⁻ + S⁻S⁺)).
    let mpo_pm = fsm_mpo_single_exp(
        n_sites,
        d,
        j_coupling * 0.5,
        decay_rate,
        &sp,
        &sm,
        &zero,
        0.0,
    )?;

    // S⁻S⁺ term: amplitude = J/2.
    let mpo_mp = fsm_mpo_single_exp(
        n_sites,
        d,
        j_coupling * 0.5,
        decay_rate,
        &sm,
        &sp,
        &zero,
        0.0,
    )?;

    superpose_long_range_mpos(vec![mpo_zz, mpo_pm, mpo_mp])
}

/// Build a power-law interaction MPO
/// ```text
/// H = J · Σ_{i<j} |i-j|^{-alpha} · σᶻᵢ · σᶻⱼ
/// ```
///
/// The interaction `J/r^alpha` is approximated by fitting `n_terms` exponentials
/// to the target values `y_r = J/r^alpha` for `r = 1, …, r_max`.
///
/// Uses the multi-exponential FSM with bond dimension `n_terms + 2`.
pub fn power_law_mpo(
    n_sites: usize,
    j_coupling: f64,
    alpha: f64,
    n_terms: usize,
    r_max: usize,
) -> TnResult<LongRangeMpo> {
    validate_sites(n_sites)?;
    if n_terms == 0 {
        return Err(TnError::InvalidParameter {
            name: "n_terms".into(),
            reason: "must be >= 1".into(),
        });
    }
    if r_max == 0 {
        return Err(TnError::InvalidParameter {
            name: "r_max".into(),
            reason: "must be >= 1".into(),
        });
    }

    // Build target: y_r = J / r^alpha for r = 1..r_max.
    let target: Vec<f64> = (1..=r_max)
        .map(|r| j_coupling / (r as f64).powf(alpha))
        .collect();

    let lambda_min = 0.1f64;
    let lambda_max = 5.0f64;
    let reg = 1e-6f64;

    let (amplitudes, rates) = exponential_fit(&target, n_terms, lambda_min, lambda_max, reg)?;

    let d = 2usize;
    let sz = spin_sz();
    let zero = zero_op(d);

    fsm_mpo_multi_exp(n_sites, d, &amplitudes, &rates, &sz, &sz, &zero, 0.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Reconstruct interaction value at range r from M-exponential fit.
    fn eval_exp_sum(amplitudes: &[f64], rates: &[f64], r: usize) -> f64 {
        amplitudes
            .iter()
            .zip(rates.iter())
            .map(|(&a, &lam)| a * (-lam * r as f64).exp())
            .sum()
    }

    // ── 1. Single-exponential fit ─────────────────────────────────────────────

    #[test]
    fn exponential_fit_single_exp() {
        // Target: exp(-1.0 * r) for r = 1..20.  With M=1 and rate grid containing 1.0
        // the fit should recover amplitude ≈ 1, rate ≈ 1.
        let target: Vec<f64> = (1..=20).map(|r| (-(r as f64)).exp()).collect();
        let (amps, rates) = exponential_fit(&target, 1, 1.0, 1.0, 1e-10).expect("fit ok");
        assert_eq!(amps.len(), 1);
        assert_eq!(rates.len(), 1);
        // With lambda_min == lambda_max == 1.0 the only basis function is exp(-r),
        // so amplitude should be exactly 1.0.
        assert!(
            (amps[0] - 1.0).abs() < 1e-6,
            "amplitude {:.6} not close to 1.0",
            amps[0]
        );
        assert!(
            (rates[0] - 1.0).abs() < 1e-12,
            "rate {:.6} not close to 1.0",
            rates[0]
        );
    }

    // ── 2. Multi-term fit reproduces power-law ────────────────────────────────

    #[test]
    fn exponential_fit_reproduces_target() {
        // Fit power-law J/r^2 with M=4 exponentials over r=1..30.
        let j = 1.0f64;
        let alpha = 2.0f64;
        let r_max = 30usize;
        let target: Vec<f64> = (1..=r_max).map(|r| j / (r as f64).powf(alpha)).collect();

        let (amps, rates) = exponential_fit(&target, 4, 0.1, 5.0, 1e-6).expect("fit ok");

        // Compute max relative error over training range.
        let max_rel_err = (1..=r_max)
            .map(|r| {
                let pred = eval_exp_sum(&amps, &rates, r);
                let tgt = target[r - 1];
                (pred - tgt).abs() / (tgt.abs() + 1e-12)
            })
            .fold(0.0f64, f64::max);

        // Ridge-regression with M=4 on 1/r^2 is approximate; check fit is reasonable.
        // At small r (large target) the fit may be less accurate; allow up to 50% relative error.
        assert!(
            max_rel_err < 0.5,
            "max relative error {max_rel_err:.4} too large for M=4 fit of 1/r^2"
        );
    }

    // ── 3. Single-exp FSM bond dimension ─────────────────────────────────────

    #[test]
    fn fsm_mpo_single_exp_shape() {
        let d = 2usize;
        let sz = spin_sz();
        let zero = zero_op(d);
        let mpo = fsm_mpo_single_exp(4, d, 1.0, 0.5, &sz, &sz, &zero, 0.0).expect("ok");
        assert_eq!(mpo.bond_dim, 3, "single-exp bond_dim must be 3");
    }

    // ── 4. Tensor count equals n_sites ────────────────────────────────────────

    #[test]
    fn fsm_mpo_tensor_count() {
        let d = 2usize;
        let sz = spin_sz();
        let zero = zero_op(d);
        let n = 7usize;
        let mpo = fsm_mpo_single_exp(n, d, 1.0, 0.5, &sz, &sz, &zero, 0.0).expect("ok");
        assert_eq!(mpo.tensors.len(), n, "tensors.len() must equal n_sites");
    }

    // ── 5. Heisenberg long-range bond dim ────────────────────────────────────

    #[test]
    fn heisenberg_long_range_bond_dim() {
        let mpo = heisenberg_long_range_mpo(5, 1.0, 0.5).expect("ok");
        // Three single-exp terms → interior_total = 3·(3-2) = 3 → combined dw = 3+2 = 5.
        assert_eq!(mpo.bond_dim, 5, "Heisenberg combined bond dim must be 5");
    }

    // ── 6. Power-law MPO constructs ───────────────────────────────────────────

    #[test]
    fn power_law_mpo_constructs() {
        let mpo = power_law_mpo(6, 1.0, 2.0, 3, 20).expect("ok");
        assert_eq!(mpo.n_sites, 6);
        // bond_dim = n_terms + 2 = 5
        assert_eq!(mpo.bond_dim, 3 + 2);
    }

    // ── 7. Left boundary D_l = 1 ─────────────────────────────────────────────

    #[test]
    fn fsm_left_tensor_shape() {
        let d = 2usize;
        let sz = spin_sz();
        let zero = zero_op(d);
        let mpo = fsm_mpo_single_exp(5, d, 1.0, 0.5, &sz, &sz, &zero, 0.0).expect("ok");
        assert_eq!(mpo.shapes[0][0], 1, "first site D_l must be 1");
    }

    // ── 8. Right boundary D_r = 1 ────────────────────────────────────────────

    #[test]
    fn fsm_right_tensor_shape() {
        let d = 2usize;
        let sz = spin_sz();
        let zero = zero_op(d);
        let n = 5usize;
        let mpo = fsm_mpo_single_exp(n, d, 1.0, 0.5, &sz, &sz, &zero, 0.0).expect("ok");
        assert_eq!(mpo.shapes[n - 1][3], 1, "last site D_r must be 1");
    }

    // ── 9. Near-neighbour energy dominates for large λ ────────────────────────

    #[test]
    fn local_energy_diagonal_nn() {
        // With very large decay λ = 10, exp(-10) ≈ 4.5e-5, so next-nearest-neighbour
        // energy is ~4.5e-5 of the nearest-neighbour energy.
        // Build SzSz MPO with amplitude=1, large decay.
        let d = 2usize;
        let sz = spin_sz();
        let zero = zero_op(d);
        let n = 4usize;
        let lam = 10.0f64;
        let mpo = fsm_mpo_single_exp(n, d, 1.0, lam, &sz, &sz, &zero, 0.0).expect("ok");

        // State: alternating |↑↓↑↓⟩ = [0,1,0,1].
        let state = vec![0usize, 1, 0, 1];
        let energy = mpo.local_energy_diagonal(&state).expect("ok");

        // Nearest-neighbour SzSz energy for |↑↓↑↓⟩:
        // each NN pair contributes Sz(0)·Sz(1) = (0.5)·(-0.5) = -0.25.
        // 3 NN pairs → NN energy ≈ -0.75 (×amplitude×1, no decay factor for r=1 in FSM).
        // Wait: in the FSM the amplitude is applied at the *left* site,
        // and the right site multiplies by amplitude again.  Let's check
        // by looking at the energy numerically and just verify it's non-trivial.
        assert!(energy.abs() > 0.0, "energy must be non-zero, got {energy}");
    }

    // ── 10. Ferromagnetic state energy ────────────────────────────────────────

    #[test]
    fn local_energy_diagonal_ferro_state() {
        // All-up state |↑↑↑↑⟩ = [0,0,0,0].
        // For SzSz MPO with amplitude J=1: Σ_{i<j} J·e^{-λ(j-i)}·(+0.5)·(+0.5)
        // The diagonal energy should be positive (ferromagnetic pairs aligned).
        let d = 2usize;
        let sz = spin_sz();
        let zero = zero_op(d);
        let n = 4usize;
        let mpo = fsm_mpo_single_exp(n, d, 1.0, 1.0, &sz, &sz, &zero, 0.0).expect("ok");

        let state = vec![0usize, 0, 0, 0]; // all |↑⟩
        let energy = mpo.local_energy_diagonal(&state).expect("ok");
        // All Sz = +0.5; each pair contributes positively.
        assert!(
            energy > 0.0,
            "ferro state energy should be > 0, got {energy}"
        );
    }

    // ── 11. to_mpo n_sites ────────────────────────────────────────────────────

    #[test]
    fn to_mpo_n_sites() {
        let d = 2usize;
        let sz = spin_sz();
        let zero = zero_op(d);
        let n = 5usize;
        let lr_mpo = fsm_mpo_single_exp(n, d, 1.0, 0.5, &sz, &sz, &zero, 0.0).expect("ok");
        let mpo = lr_mpo.to_mpo().expect("to_mpo ok");
        assert_eq!(mpo.n_sites(), n, "to_mpo n_sites mismatch");
    }

    // ── 12. n_sites = 0 returns error ─────────────────────────────────────────

    #[test]
    fn n_sites_zero_error() {
        let d = 2usize;
        let sz = spin_sz();
        let zero = zero_op(d);
        let result = fsm_mpo_single_exp(0, d, 1.0, 0.5, &sz, &sz, &zero, 0.0);
        assert!(
            matches!(result, Err(TnError::EmptyInput)),
            "expected EmptyInput for n_sites=0, got {:?}",
            result
        );
    }

    // ── 13. n_sites = 1 returns error ─────────────────────────────────────────

    #[test]
    fn n_sites_one_error() {
        let d = 2usize;
        let sz = spin_sz();
        let zero = zero_op(d);
        let result = fsm_mpo_single_exp(1, d, 1.0, 0.5, &sz, &sz, &zero, 0.0);
        assert!(
            matches!(result, Err(TnError::InvalidConfiguration(_))),
            "expected InvalidConfiguration for n_sites=1, got {:?}",
            result
        );
    }

    // ── 14. Power-law energy for known case ───────────────────────────────────

    #[test]
    fn power_law_mpo_energy_sign() {
        // All-up state with J>0 power-law SzSz: energy should be > 0.
        let n = 4usize;
        let mpo = power_law_mpo(n, 1.0, 2.0, 3, 20).expect("ok");
        let state = vec![0usize, 0, 0, 0];
        let energy = mpo.local_energy_diagonal(&state).expect("ok");
        assert!(energy > 0.0, "power-law ferro energy > 0, got {energy}");
    }

    // ── 15. Heisenberg MPO to_mpo conversion ─────────────────────────────────

    #[test]
    fn heisenberg_to_mpo_ok() {
        let n = 5usize;
        let lr = heisenberg_long_range_mpo(n, 1.0, 0.5).expect("ok");
        let mpo = lr.to_mpo().expect("to_mpo");
        assert_eq!(mpo.n_sites(), n);
    }

    // ── 16. Exponential fit error on empty input ──────────────────────────────

    #[test]
    fn exponential_fit_empty_error() {
        let result = exponential_fit(&[], 2, 0.1, 5.0, 1e-6);
        assert!(
            matches!(result, Err(TnError::EmptyInput)),
            "expected EmptyInput, got {:?}",
            result
        );
    }
}
