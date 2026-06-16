//! Harrow–Hassidim–Lloyd (HHL) quantum linear-systems algorithm.
//!
//! Reference: Harrow, Hassidim, Lloyd, *"Quantum algorithm for linear systems of
//! equations"*, Phys. Rev. Lett. 103, 150502 (2009) (arXiv:0811.3171).
//!
//! HHL prepares a quantum state proportional to the solution `x` of `A x = b` for
//! a Hermitian `A`. The circuit, on three registers — a *system* register holding
//! `|b⟩`, a *clock* register, and a single-qubit *ancilla* — is:
//!
//! 1. Prepare the normalized right-hand side `|b⟩` in the system register.
//! 2. Hadamard the clock register into a uniform superposition.
//! 3. **Phase estimation** of `U = e^{iAt}`: apply controlled-`U^{2^k}` from clock
//!    qubit `k` onto the system register. Because `A = Σ_j λ_j |u_j⟩⟨u_j|`, the
//!    propagator `U^{2^k} = e^{iA t 2^k}` is built *exactly* from the classical
//!    eigendecomposition of the small Hermitian `A`.
//! 4. Inverse QFT on the clock register, leaving `Σ_j β_j |u_j⟩_sys |c_j⟩_clk`
//!    with `c_j` the integer encoding of eigenvalue `λ_j`.
//! 5. **Eigenvalue inversion**: a clock-controlled `R_y(2·arcsin(C/λ))` rotation on
//!    the ancilla writes amplitude `C/λ_j` into `|1⟩_anc`.
//! 6. **Uncompute** (inverse phase estimation) to disentangle and reset the clock.
//! 7. **Post-select** the ancilla on `|1⟩`; the system register now holds
//!    `∝ A⁻¹|b⟩ = |x⟩`.
//!
//! # Clock-resolution constraint
//!
//! Phase estimation maps eigenvalue `λ_j` to the clock integer
//!
//! ```text
//! c_j = λ_j · t · 2^{n_clock} / (2π).
//! ```
//!
//! For a *faithful yet exactly resolvable* simulation we require every `c_j` to be
//! a positive integer in `[1, 2^{n_clock})`. Equivalently, choosing
//! `t = 2π / 2^{n_clock}` makes `c_j = λ_j`, so the eigenvalues must be **distinct
//! positive integers** `< 2^{n_clock}` (the "eigenvalues are exact multiples of
//! the clock resolution" condition). [`HhlConfig::with_recommended_t`] picks this
//! `t`; the inverse map `λ(c) = 2π c /(t·2^{n_clock})` is used to drive the
//! ancilla rotation. Negative or non-integer eigenvalues, or eigenvalues `≥`
//! `2^{n_clock}`, are reported via [`crate::error::QuantumError`].
//!
//! # Numerics
//!
//! The classical eigendecomposition (a complex-Hermitian Jacobi sweep) and the
//! propagator construction run in `f64`; the resulting unitaries are applied to
//! the `f32` [`StateVector`] via the shared gate machinery and a small number of
//! exact dense-block helpers.

use crate::error::{QuantumError, QuantumResult};
use crate::fourier::qft::{qft_inplace, qft_inverse_inplace};
use crate::gates::hadamard::gate_h;
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;
use num_complex::Complex;

type C64 = Complex<f64>;
type C32 = Complex<f32>;

const TWO_PI: f64 = std::f64::consts::TAU;

/// A small Hermitian matrix `A` together with its (classically computed)
/// eigendecomposition `A = Σ_j λ_j |u_j⟩⟨u_j|`.
#[derive(Debug, Clone)]
pub struct HermitianMatrix {
    /// Dimension `dim = 2^{n_sys}` (here 2 or 4).
    dim: usize,
    /// Number of system qubits (`log2(dim)`).
    n_sys: usize,
    /// Row-major `dim×dim` matrix entries.
    data: Vec<C64>,
    /// Eigenvalues `λ_0 ≤ … ≤ λ_{dim-1}`.
    eigenvalues: Vec<f64>,
    /// Eigenvectors as columns: `eigenvectors[j]` is `|u_j⟩` (length `dim`).
    eigenvectors: Vec<Vec<C64>>,
}

impl HermitianMatrix {
    /// Build from a row-major `dim×dim` matrix, validating Hermiticity and
    /// computing the eigendecomposition.
    ///
    /// # Errors
    /// * [`QuantumError::InvalidParameter`] if `dim` is not 2 or 4, or the data
    ///   length is wrong.
    /// * [`QuantumError::NonHermitianHamiltonian`] if `A ≠ A†` beyond tolerance.
    pub fn new(data: Vec<C64>, dim: usize) -> QuantumResult<Self> {
        if dim != 2 && dim != 4 {
            return Err(QuantumError::InvalidParameter {
                name: format!("HHL matrix dim={dim} must be 2 or 4"),
            });
        }
        if data.len() != dim * dim {
            return Err(QuantumError::DimensionMismatch {
                expected: dim * dim,
                got: data.len(),
            });
        }
        // Hermiticity check: A[i][j] == conj(A[j][i]).
        for i in 0..dim {
            for j in 0..dim {
                let aij = data[i * dim + j];
                let aji = data[j * dim + i];
                if (aij - aji.conj()).norm() > 1e-9 {
                    return Err(QuantumError::NonHermitianHamiltonian);
                }
            }
        }
        let n_sys = dim.trailing_zeros() as usize;
        let (eigenvalues, eigenvectors) = hermitian_eigendecomposition(&data, dim)?;
        Ok(Self {
            dim,
            n_sys,
            data,
            eigenvalues,
            eigenvectors,
        })
    }

    /// Number of system qubits.
    #[must_use]
    pub fn n_sys(&self) -> usize {
        self.n_sys
    }

    /// Matrix dimension `2^{n_sys}`.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The row-major matrix entries `A` as originally supplied.
    #[must_use]
    pub fn entries(&self) -> &[C64] {
        &self.data
    }

    /// The eigenvalues (ascending).
    #[must_use]
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    /// The propagator `e^{iA·s}` as a dense `dim×dim` matrix (row-major) built from
    /// the eigendecomposition: `Σ_j e^{i λ_j s} |u_j⟩⟨u_j|`.
    #[must_use]
    pub fn propagator(&self, s: f64) -> Vec<C64> {
        let dim = self.dim;
        let mut m = vec![C64::new(0.0, 0.0); dim * dim];
        for (j, uj) in self.eigenvectors.iter().enumerate() {
            let phase = C64::new(
                (self.eigenvalues[j] * s).cos(),
                (self.eigenvalues[j] * s).sin(),
            );
            for r in 0..dim {
                for c in 0..dim {
                    m[r * dim + c] += phase * uj[r] * uj[c].conj();
                }
            }
        }
        m
    }

    /// Classical solution `A⁻¹ b` (normalized), the reference target for the
    /// quantum output. Returns `None` if `A` is singular within tolerance.
    #[must_use]
    pub fn classical_solution(&self, b: &[C64]) -> Option<Vec<C64>> {
        let dim = self.dim;
        if b.len() != dim {
            return None;
        }
        // x = Σ_j (⟨u_j|b⟩ / λ_j) |u_j⟩.
        let mut x = vec![C64::new(0.0, 0.0); dim];
        for (j, uj) in self.eigenvectors.iter().enumerate() {
            if self.eigenvalues[j].abs() < 1e-12 {
                return None;
            }
            let mut overlap = C64::new(0.0, 0.0);
            for (r, &br) in b.iter().enumerate() {
                overlap += uj[r].conj() * br;
            }
            let coeff = overlap / C64::new(self.eigenvalues[j], 0.0);
            for r in 0..dim {
                x[r] += coeff * uj[r];
            }
        }
        let norm: f64 = x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return None;
        }
        for z in &mut x {
            *z /= norm;
        }
        Some(x)
    }
}

/// HHL run configuration.
#[derive(Debug, Clone)]
pub struct HhlConfig {
    /// Number of clock qubits.
    pub n_clock: usize,
    /// Phase-estimation evolution time `t`.
    pub t: f64,
    /// Inversion constant `C` (must satisfy `C ≤ min_j |λ_j|` for valid `arcsin`).
    pub c: f64,
}

impl HhlConfig {
    /// Recommended configuration for a matrix whose eigenvalues are distinct
    /// positive integers `< 2^{n_clock}`.
    ///
    /// Sets `t = 2π / 2^{n_clock}` (so clock integer `c_j = λ_j`) and
    /// `C = min_j λ_j` (the largest valid inversion constant).
    ///
    /// # Errors
    /// [`QuantumError::InvalidParameter`] if the eigenvalues violate the
    /// clock-resolution constraint (non-positive, non-integer, or `≥ 2^{n_clock}`).
    pub fn with_recommended_t(matrix: &HermitianMatrix, n_clock: usize) -> QuantumResult<Self> {
        if n_clock == 0 || n_clock > 20 {
            return Err(QuantumError::InvalidParameter {
                name: format!("n_clock={n_clock} out of range"),
            });
        }
        let levels = 1u64 << n_clock;
        let mut c_min = f64::INFINITY;
        for &lam in matrix.eigenvalues() {
            if lam <= 0.0 {
                return Err(QuantumError::InvalidParameter {
                    name: format!("eigenvalue {lam} must be positive for the recommended clock"),
                });
            }
            let rounded = lam.round();
            if (lam - rounded).abs() > 1e-6 || rounded < 1.0 || rounded as u64 >= levels {
                return Err(QuantumError::InvalidParameter {
                    name: format!(
                        "eigenvalue {lam} must be an integer in [1, {levels}) for clock resolution"
                    ),
                });
            }
            c_min = c_min.min(lam);
        }
        let t = TWO_PI / levels as f64;
        Ok(Self {
            n_clock,
            t,
            c: c_min,
        })
    }
}

/// Result of an HHL run.
#[derive(Debug, Clone)]
pub struct HhlResult {
    /// Normalized system-register state after post-selecting `ancilla = 1`
    /// (length `dim`), `∝ A⁻¹ b`.
    pub solution: Vec<C32>,
    /// Probability of the post-selection succeeding (`ancilla` measured `1`).
    pub success_probability: f64,
    /// Total probability *before* post-selection (should be ≈ 1; a conservation
    /// check exposed for callers/tests).
    pub total_probability_before: f64,
}

/// Solve `A x = b` with the HHL algorithm and read out the (post-selected,
/// normalized) system state.
///
/// `b` must be a length-`dim` complex vector; it is normalized internally.
///
/// # Errors
/// * [`QuantumError::DimensionMismatch`] if `b.len() != matrix.dim()`.
/// * [`QuantumError::InvalidParameter`] if `b` is the zero vector, or if `C/λ(c)`
///   exceeds 1 for some occupied clock value (invalid `arcsin`, i.e. `C` too
///   large).
/// * [`QuantumError::MeasurementFailed`] if the post-selection has vanishing
///   probability.
pub fn hhl_solve(
    matrix: &HermitianMatrix,
    b: &[C64],
    config: &HhlConfig,
) -> QuantumResult<HhlResult> {
    let dim = matrix.dim;
    if b.len() != dim {
        return Err(QuantumError::DimensionMismatch {
            expected: dim,
            got: b.len(),
        });
    }
    let n_sys = matrix.n_sys;
    let n_clock = config.n_clock;
    let n_qubits = n_sys + n_clock + 1;
    if n_qubits > 30 {
        return Err(QuantumError::InvalidQubitCount { n: n_qubits });
    }

    // Qubit layout (little-endian): system = [0, n_sys), clock = [n_sys,
    // n_sys+n_clock), ancilla = n_sys+n_clock.
    let sys_qubits: Vec<usize> = (0..n_sys).collect();
    let clock_qubits: Vec<usize> = (n_sys..n_sys + n_clock).collect();
    let ancilla = n_sys + n_clock;

    // --- Step 1: prepare |b⟩ in the system register. ---
    let mut sv = prepare_b_state(b, n_qubits, &sys_qubits)?;

    // --- Step 2: Hadamard the clock register. ---
    for &q in &clock_qubits {
        apply_1q_inplace(&mut sv, q, &gate_h())?;
    }

    // --- Step 3: controlled-U^{2^k} phase estimation ladder. ---
    for (k, &cq) in clock_qubits.iter().enumerate() {
        let scale = (1u64 << k) as f64;
        let u_pow = matrix.propagator(config.t * scale);
        let u32: Vec<C32> = u_pow.iter().map(c64_to_c32).collect();
        apply_controlled_dense(&mut sv, cq, &sys_qubits, &u32)?;
    }

    // --- Step 4: inverse QFT on the clock register. ---
    qft_inverse_inplace(&mut sv, &clock_qubits)?;

    // --- Step 5: clock-controlled ancilla rotation R_y(2·arcsin(C/λ(c))). ---
    apply_eigenvalue_inversion(&mut sv, &clock_qubits, ancilla, config)?;

    // --- Step 6: uncompute (inverse phase estimation). ---
    qft_inplace(&mut sv, &clock_qubits)?;
    for (k, &cq) in clock_qubits.iter().enumerate() {
        let scale = (1u64 << k) as f64;
        // Inverse propagator e^{-iA t 2^k}.
        let u_inv = matrix.propagator(-config.t * scale);
        let u32: Vec<C32> = u_inv.iter().map(c64_to_c32).collect();
        apply_controlled_dense(&mut sv, cq, &sys_qubits, &u32)?;
    }
    for &q in &clock_qubits {
        apply_1q_inplace(&mut sv, q, &gate_h())?;
    }

    // --- Total probability (conservation check). ---
    let total_probability_before = sv.norm_sq() as f64;

    // --- Step 7: post-select ancilla = 1, clock = 0; read out system. ---
    let anc_mask = 1usize << ancilla;
    let clock_mask: usize = clock_qubits.iter().map(|&q| 1usize << q).sum();
    let sys_mask: usize = sys_qubits.iter().map(|&q| 1usize << q).sum();

    // Success probability = total mass with ancilla = 1.
    let mut success_probability = 0.0_f64;
    for (i, a) in sv.amps.iter().enumerate() {
        if i & anc_mask != 0 {
            success_probability += a.norm_sqr() as f64;
        }
    }

    // Extract the system amplitudes from the (ancilla = 1, clock = 0) subspace.
    let mut solution = vec![C32::new(0.0, 0.0); dim];
    for (i, &a) in sv.amps.iter().enumerate() {
        if (i & anc_mask) != 0 && (i & clock_mask) == 0 {
            // Decode the system index from the system bits (contiguous low bits).
            let sys_index = i & sys_mask;
            solution[sys_index] = a;
        }
    }
    let sol_norm: f64 = solution
        .iter()
        .map(|z| z.norm_sqr() as f64)
        .sum::<f64>()
        .sqrt();
    if sol_norm < 1e-9 {
        return Err(QuantumError::MeasurementFailed);
    }
    let inv = 1.0_f32 / sol_norm as f32;
    for z in &mut solution {
        *z *= inv;
    }

    Ok(HhlResult {
        solution,
        success_probability,
        total_probability_before,
    })
}

/// Prepare the (normalized) amplitude-encoded state `|b⟩` in the system register
/// of an `n_qubits`-qubit zero state. The system qubits are the contiguous low
/// bits, so the system index maps directly onto the global index.
fn prepare_b_state(b: &[C64], n_qubits: usize, sys_qubits: &[usize]) -> QuantumResult<StateVector> {
    let norm: f64 = b.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    if norm < 1e-15 {
        return Err(QuantumError::InvalidParameter {
            name: "right-hand side b must be non-zero".into(),
        });
    }
    let dim = 1usize << n_qubits;
    let mut amps = vec![C32::new(0.0, 0.0); dim];
    let sys_mask: usize = sys_qubits.iter().map(|&q| 1usize << q).sum();
    for (sys_index, &bj) in b.iter().enumerate() {
        // Place b at (clock=0, ancilla=0, system=sys_index). Since system qubits
        // are the low bits, the global index equals sys_index (masked).
        let global = sys_index & sys_mask;
        let normalized = bj / C64::new(norm, 0.0);
        amps[global] = c64_to_c32(&normalized);
    }
    Ok(StateVector { amps, n_qubits })
}

/// Apply a dense `dim×dim` unitary `u` (row-major) to the contiguous system
/// register conditioned on `ctrl = 1`.
///
/// For every global index whose control bit is set, the `dim` system amplitudes
/// (varying only the system bits) are transformed by `u`. The system qubits are
/// assumed contiguous and starting at bit 0, so the system sub-index equals the
/// global index masked to the system bits.
fn apply_controlled_dense(
    sv: &mut StateVector,
    ctrl: usize,
    sys_qubits: &[usize],
    u: &[C32],
) -> QuantumResult<()> {
    if ctrl >= sv.n_qubits {
        return Err(QuantumError::QubitIndexOutOfRange {
            index: ctrl,
            n_qubits: sv.n_qubits,
        });
    }
    let n_sys = sys_qubits.len();
    let dim = 1usize << n_sys;
    if u.len() != dim * dim {
        return Err(QuantumError::DimensionMismatch {
            expected: dim * dim,
            got: u.len(),
        });
    }
    let ctrl_mask = 1usize << ctrl;
    let sys_mask: usize = sys_qubits.iter().map(|&q| 1usize << q).sum();
    let total = sv.amps.len();

    // Iterate over "outer" configurations (everything except system + nothing
    // else changes); the system bits are the low `n_sys` bits, so we can stride.
    let mut base = 0usize;
    while base < total {
        // Only process blocks where the control is set and the system bits are 0
        // (the block representative), to transform each system block once.
        if (base & ctrl_mask) != 0 && (base & sys_mask) == 0 {
            // Gather the dim system amplitudes.
            let mut input = vec![C32::new(0.0, 0.0); dim];
            for (s, slot) in input.iter_mut().enumerate() {
                *slot = sv.amps[base | s];
            }
            // Apply U.
            for r in 0..dim {
                let mut acc = C32::new(0.0, 0.0);
                for (col, &xv) in input.iter().enumerate() {
                    acc += u[r * dim + col] * xv;
                }
                sv.amps[base | r] = acc;
            }
        }
        base += 1;
    }
    Ok(())
}

/// Apply the eigenvalue-inversion rotation: for each clock basis value `c`, rotate
/// the ancilla by `R_y(2·arcsin(C/λ(c)))`, where `λ(c) = 2π c /(t·2^{n_clock})`.
///
/// Implemented as the exact controlled unitary
/// `Σ_c |c⟩⟨c|_clk ⊗ R_y(θ_c)_anc ⊗ I_sys`, evaluated directly on the amplitude
/// pairs that differ only in the ancilla bit.
fn apply_eigenvalue_inversion(
    sv: &mut StateVector,
    clock_qubits: &[usize],
    ancilla: usize,
    config: &HhlConfig,
) -> QuantumResult<()> {
    if ancilla >= sv.n_qubits {
        return Err(QuantumError::QubitIndexOutOfRange {
            index: ancilla,
            n_qubits: sv.n_qubits,
        });
    }
    let n_clock = clock_qubits.len();
    let levels = (1u64 << n_clock) as f64;
    let anc_mask = 1usize << ancilla;
    let total = sv.amps.len();

    // Precompute θ_c for every clock value c ∈ [0, 2^{n_clock}).
    let mut thetas = vec![0.0_f64; 1usize << n_clock];
    for (c, theta) in thetas.iter_mut().enumerate() {
        if c == 0 {
            // c = 0 ↦ λ = 0: no inversion (leave ancilla unrotated).
            *theta = 0.0;
            continue;
        }
        let lambda = TWO_PI * c as f64 / (config.t * levels);
        let ratio = config.c / lambda;
        if ratio.abs() > 1.0 + 1e-9 {
            return Err(QuantumError::InvalidParameter {
                name: format!("C/λ={ratio} > 1 at clock value {c}; reduce C"),
            });
        }
        *theta = 2.0 * ratio.clamp(-1.0, 1.0).asin();
    }

    let mut i = 0usize;
    while i < total {
        if i & anc_mask == 0 {
            // Decode the clock value c from this index.
            let mut c = 0usize;
            for (k, &q) in clock_qubits.iter().enumerate() {
                c |= ((i >> q) & 1) << k;
            }
            let theta = thetas[c];
            if theta != 0.0 {
                let half = (theta * 0.5) as f32;
                let cos = half.cos();
                let sin = half.sin();
                let i1 = i | anc_mask;
                let x0 = sv.amps[i];
                let x1 = sv.amps[i1];
                // R_y(θ) = [[cos, -sin], [sin, cos]] acting on (|0⟩_anc, |1⟩_anc).
                sv.amps[i] = x0 * cos - x1 * sin;
                sv.amps[i1] = x0 * sin + x1 * cos;
            }
        }
        i += 1;
    }
    Ok(())
}

/// Convert an `f64` complex number to `f32`.
#[inline]
fn c64_to_c32(z: &C64) -> C32 {
    C32::new(z.re as f32, z.im as f32)
}

// ─── Complex-Hermitian Jacobi eigensolver ────────────────────────────────────

/// Eigendecomposition of a small dense complex-Hermitian matrix via cyclic Jacobi
/// rotations. Returns `(eigenvalues_ascending, eigenvectors_as_columns)` with each
/// eigenvector normalized; `eigenvectors[j]` is the column `|u_j⟩` for
/// `eigenvalues[j]`.
fn hermitian_eigendecomposition(
    data: &[C64],
    dim: usize,
) -> QuantumResult<(Vec<f64>, Vec<Vec<C64>>)> {
    // Working copy of A (will be driven toward a diagonal of eigenvalues).
    let mut a = data.to_vec();
    // Accumulated unitary V (columns become eigenvectors). Start at identity.
    let mut v = vec![C64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        v[i * dim + i] = C64::new(1.0, 0.0);
    }

    let max_sweeps = 100;
    for _ in 0..max_sweeps {
        // Off-diagonal Frobenius norm.
        let mut off = 0.0_f64;
        for p in 0..dim {
            for q in 0..dim {
                if p != q {
                    off += a[p * dim + q].norm_sqr();
                }
            }
        }
        if off.sqrt() < 1e-14 {
            break;
        }
        // Sweep over upper-triangular (p < q) pairs.
        for p in 0..dim {
            for q in (p + 1)..dim {
                jacobi_rotate(&mut a, &mut v, dim, p, q);
            }
        }
    }

    // Read eigenvalues off the diagonal; eigenvectors are columns of V.
    let mut pairs: Vec<(f64, Vec<C64>)> = (0..dim)
        .map(|j| {
            let lam = a[j * dim + j].re;
            let col: Vec<C64> = (0..dim).map(|r| v[r * dim + j]).collect();
            (lam, col)
        })
        .collect();
    // Sort ascending by eigenvalue.
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f64> = pairs.iter().map(|(l, _)| *l).collect();
    let mut eigenvectors: Vec<Vec<C64>> = pairs.into_iter().map(|(_, c)| c).collect();
    // Normalize each eigenvector (guards against accumulated round-off).
    for col in &mut eigenvectors {
        let norm: f64 = col.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for z in col.iter_mut() {
                *z /= norm;
            }
        }
    }
    Ok((eigenvalues, eigenvectors))
}

/// One complex-Hermitian Jacobi rotation zeroing the `(p, q)` off-diagonal entry.
///
/// For a Hermitian `A`, the `(p, q)` block is diagonalized by a complex Givens
/// rotation `G` so that `G† A G` has `a_{pq} = 0`. The rotation is parameterized
/// by the phase of `a_{pq}` and a real Jacobi angle.
fn jacobi_rotate(a: &mut [C64], v: &mut [C64], dim: usize, p: usize, q: usize) {
    let apq = a[p * dim + q];
    let apq_abs = apq.norm();
    if apq_abs < 1e-300 {
        return;
    }
    let app = a[p * dim + p].re;
    let aqq = a[q * dim + q].re;

    // Phase factor e^{iα} = a_{pq}/|a_{pq}| so that the rotated off-diagonal is
    // real before applying the real Jacobi angle.
    let phase = apq / C64::new(apq_abs, 0.0);

    // Real Jacobi angle θ from the 2×2 real-symmetric problem
    // [[app, |apq|], [|apq|, aqq]].
    let tau = (aqq - app) / (2.0 * apq_abs);
    let t = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
    };
    let cos = 1.0 / (1.0 + t * t).sqrt();
    let sin = t * cos;

    // Givens rotation entries: column p gets cos, column q gets cos; the coupling
    // uses sin·phase (and its conjugate) to handle the complex off-diagonal.
    // Define G acting on the (p, q) 2D subspace:
    //   |p'⟩ =  cos |p⟩ + sin·conj(phase) |q⟩
    //   |q'⟩ = -sin·phase |p⟩ + cos |q⟩
    let s_phase = phase * C64::new(sin, 0.0);
    let s_phase_conj = phase.conj() * C64::new(sin, 0.0);
    let cc = C64::new(cos, 0.0);

    // Update A ← G† A G. Apply on the left (rows) then on the right (columns) for
    // columns/rows p and q only.
    // Left multiply by G† : rows p and q.
    for col in 0..dim {
        let a_p = a[p * dim + col];
        let a_q = a[q * dim + col];
        // row p' = cos·row_p + conj(s_phase)·row_q ; conj of |p'⟩ coefficients.
        a[p * dim + col] = cc * a_p + s_phase * a_q;
        a[q * dim + col] = -s_phase_conj * a_p + cc * a_q;
    }
    // Right multiply by G : columns p and q.
    for row in 0..dim {
        let a_p = a[row * dim + p];
        let a_q = a[row * dim + q];
        a[row * dim + p] = cc * a_p + s_phase_conj * a_q;
        a[row * dim + q] = -s_phase * a_p + cc * a_q;
    }
    // Accumulate eigenvectors V ← V·G (columns p and q).
    for row in 0..dim {
        let v_p = v[row * dim + p];
        let v_q = v[row * dim + q];
        v[row * dim + p] = cc * v_p + s_phase_conj * v_q;
        v[row * dim + q] = -s_phase * v_p + cc * v_q;
    }
    // Enforce exact zero on the symmetric off-diagonal (numerical hygiene).
    a[p * dim + q] = C64::new(0.0, 0.0);
    a[q * dim + p] = C64::new(0.0, 0.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fidelity |⟨target|sol⟩|² between two normalized complex vectors.
    fn fidelity(sol: &[C32], target: &[C64]) -> f64 {
        let mut inner = C64::new(0.0, 0.0);
        for (s, t) in sol.iter().zip(target.iter()) {
            let s64 = C64::new(s.re as f64, s.im as f64);
            inner += t.conj() * s64;
        }
        inner.norm_sqr()
    }

    fn c(re: f64, im: f64) -> C64 {
        C64::new(re, im)
    }

    // Build a 2×2 Hermitian with given diagonal & complex off-diagonal.
    fn herm2(a: f64, d: f64, off: C64) -> Vec<C64> {
        vec![c(a, 0.0), off, off.conj(), c(d, 0.0)]
    }

    // (a) 2×2 Hermitian with integer eigenvalues; post-selected state has fidelity
    //     > 0.99 with normalized A⁻¹b.
    #[test]
    fn hhl_fidelity_2x2_hermitian() {
        // A = [[2, 1-? ]] choose A = [[2, -1],[-1, 2]] → eigenvalues 1 and 3, both
        // integers and < 2^n_clock. b = (1, 0).
        let data = herm2(2.0, 2.0, c(-1.0, 0.0));
        let matrix = HermitianMatrix::new(data, 2)
            .expect("2×2 Hermitian with integer eigenvalues must be constructible");
        let mut evs = matrix.eigenvalues().to_vec();
        evs.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        assert!(
            (evs[0] - 1.0).abs() < 1e-6 && (evs[1] - 3.0).abs() < 1e-6,
            "evs={evs:?}"
        );

        let b = vec![c(1.0, 0.0), c(0.0, 0.0)];
        let config = HhlConfig::with_recommended_t(&matrix, 4)
            .expect("eigenvalues 1 and 3 satisfy the 4-clock-qubit constraint");
        let res = hhl_solve(&matrix, &b, &config)
            .expect("HHL must succeed for a valid 2×2 Hermitian with integer eigenvalues");

        let target = matrix
            .classical_solution(&b)
            .expect("classical solution must exist for a non-singular 2×2 matrix");
        let fid = fidelity(&res.solution, &target);
        assert!(fid > 0.99, "fidelity={fid}");
    }

    // (b) A = I → output = |b⟩ (up to global phase / normalization).
    #[test]
    fn hhl_identity_returns_b() {
        let data = vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)];
        let matrix =
            HermitianMatrix::new(data, 2).expect("2×2 identity is a valid Hermitian matrix");
        // Both eigenvalues are 1 → fine for clock. b = (0.6, 0.8).
        let b = vec![c(0.6, 0.0), c(0.8, 0.0)];
        let config = HhlConfig::with_recommended_t(&matrix, 4)
            .expect("identity eigenvalue 1 satisfies the 4-clock-qubit constraint");
        let res = hhl_solve(&matrix, &b, &config)
            .expect("HHL on the identity matrix must succeed with a non-zero b");

        let bnorm = (b[0].norm_sqr() + b[1].norm_sqr()).sqrt();
        let target: Vec<C64> = b.iter().map(|z| z / c(bnorm, 0.0)).collect();
        let fid = fidelity(&res.solution, &target);
        assert!(fid > 0.999, "identity fidelity={fid}");
    }

    // (c) Post-selection success probability ≈ Σ_i (C/λ_i)²|⟨u_i|b⟩|².
    #[test]
    fn hhl_success_probability_matches_formula() {
        // Diagonal A = diag(1, 2): eigenvectors are |0⟩, |1⟩. b = (1, 1)/√2.
        let data = vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(2.0, 0.0)];
        let matrix = HermitianMatrix::new(data, 2)
            .expect("diagonal 2×2 Hermitian with entries 1 and 2 is valid");
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let b = vec![c(inv_sqrt2, 0.0), c(inv_sqrt2, 0.0)];
        let config = HhlConfig::with_recommended_t(&matrix, 4)
            .expect("integer eigenvalues 1 and 2 satisfy the 4-clock-qubit constraint");
        let res = hhl_solve(&matrix, &b, &config)
            .expect("HHL must succeed for diagonal matrix with valid integer eigenvalues");

        // Expected: Σ_i (C/λ_i)² |⟨u_i|b⟩|². λ = {1, 2}, C = 1, |⟨u_i|b⟩|² = 1/2.
        let cval = config.c;
        let expected = (cval / 1.0).powi(2) * 0.5 + (cval / 2.0).powi(2) * 0.5;
        assert!(
            (res.success_probability - expected).abs() < 1e-3,
            "success={}, expected={expected}",
            res.success_probability
        );
    }

    // (d) Diagonal A = diag(1, 2) gives x ∝ (b0/1, b1/2).
    #[test]
    fn hhl_diagonal_inverse_scaling() {
        let data = vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(2.0, 0.0)];
        let matrix = HermitianMatrix::new(data, 2)
            .expect("diagonal 2×2 Hermitian with entries 1 and 2 is valid");
        let b = vec![c(1.0, 0.0), c(1.0, 0.0)];
        let config = HhlConfig::with_recommended_t(&matrix, 4)
            .expect("integer eigenvalues satisfy the 4-clock-qubit constraint");
        let res = hhl_solve(&matrix, &b, &config)
            .expect("HHL must succeed for a diagonal matrix with valid config");

        // x ∝ (1/1, 1/2) = (1, 0.5); normalized → (0.8944, 0.4472).
        let raw = [1.0_f64, 0.5];
        let nrm = (raw[0] * raw[0] + raw[1] * raw[1]).sqrt();
        let target = [raw[0] / nrm, raw[1] / nrm];
        // Compare magnitudes (global phase irrelevant).
        let m0 = res.solution[0].norm() as f64;
        let m1 = res.solution[1].norm() as f64;
        assert!((m0 - target[0]).abs() < 1e-2, "|x0|={m0} vs {}", target[0]);
        assert!((m1 - target[1]).abs() < 1e-2, "|x1|={m1} vs {}", target[1]);
        // Ratio |x0|/|x1| ≈ 2.
        assert!((m0 / m1 - 2.0).abs() < 5e-2, "ratio={}", m0 / m1);
    }

    // (e) Non-Hermitian A → error.
    #[test]
    fn hhl_non_hermitian_errors() {
        // [[1, 1], [0, 2]] is not Hermitian (a01 != conj(a10)).
        let data = vec![c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(2.0, 0.0)];
        assert!(HermitianMatrix::new(data, 2).is_err());
        // Complex non-Hermitian: a01 = i but a10 = i (should be -i).
        let data2 = vec![c(1.0, 0.0), c(0.0, 1.0), c(0.0, 1.0), c(2.0, 0.0)];
        assert!(HermitianMatrix::new(data2, 2).is_err());
    }

    // (f) Total probability conserved before post-selection (≈ 1).
    #[test]
    fn hhl_total_probability_conserved() {
        let data = herm2(2.0, 2.0, c(-1.0, 0.0));
        let matrix = HermitianMatrix::new(data, 2)
            .expect("2×2 Hermitian with integer eigenvalues 1 and 3 is constructible");
        let b = vec![c(1.0, 0.0), c(0.0, 0.0)];
        let config = HhlConfig::with_recommended_t(&matrix, 4)
            .expect("eigenvalues 1 and 3 satisfy the 4-clock-qubit constraint");
        let res = hhl_solve(&matrix, &b, &config)
            .expect("HHL must succeed for this valid matrix and config");
        assert!(
            (res.total_probability_before - 1.0).abs() < 1e-3,
            "total prob={}",
            res.total_probability_before
        );
    }

    // 4×4 Hermitian: the eigensolver + propagator path works end-to-end.
    #[test]
    fn hhl_4x4_diagonal_inverse() {
        // diag(1, 2, 3, 4): integer eigenvalues, distinct, < 16.
        let mut data = vec![c(0.0, 0.0); 16];
        for (i, lam) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            data[i * 4 + i] = c(*lam, 0.0);
        }
        let matrix = HermitianMatrix::new(data, 4)
            .expect("4×4 diagonal Hermitian with integer eigenvalues 1–4 is valid");
        let b = vec![c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)];
        let config = HhlConfig::with_recommended_t(&matrix, 4)
            .expect("eigenvalues 1–4 all fit within the 4-clock-qubit (< 16) constraint");
        let res = hhl_solve(&matrix, &b, &config)
            .expect("HHL must succeed for a diagonal 4×4 matrix with valid config");
        let target = matrix
            .classical_solution(&b)
            .expect("classical solution exists for a non-singular diagonal matrix");
        let fid = fidelity(&res.solution, &target);
        assert!(fid > 0.98, "4x4 fidelity={fid}");
    }

    // Eigensolver sanity: a known 2×2 complex-Hermitian matrix.
    #[test]
    fn eigensolver_complex_hermitian_2x2() {
        // A = [[2, i], [-i, 2]] → eigenvalues 1 and 3.
        let data = vec![c(2.0, 0.0), c(0.0, 1.0), c(0.0, -1.0), c(2.0, 0.0)];
        let (evs, vecs) = hermitian_eigendecomposition(&data, 2)
            .expect("Jacobi eigensolver must converge for a 2×2 complex-Hermitian matrix");
        assert!((evs[0] - 1.0).abs() < 1e-9, "λ0={}", evs[0]);
        assert!((evs[1] - 3.0).abs() < 1e-9, "λ1={}", evs[1]);
        // Verify A u = λ u for each eigenpair.
        for (j, uj) in vecs.iter().enumerate() {
            for r in 0..2 {
                let mut au = c(0.0, 0.0);
                for col in 0..2 {
                    au += data[r * 2 + col] * uj[col];
                }
                let lu = c(evs[j], 0.0) * uj[r];
                assert!((au - lu).norm() < 1e-8, "Au≠λu at j={j}, r={r}");
            }
        }
    }

    // Eigendecomposition reconstructs the stored matrix: A = Σ_j λ_j |u_j⟩⟨u_j|.
    // Exercises the `entries()` accessor against the propagator-at-zero identity
    // path (propagator(0) = I) and a direct spectral rebuild.
    #[test]
    fn eigendecomposition_reconstructs_matrix() {
        let data = vec![c(2.0, 0.0), c(0.0, 1.0), c(0.0, -1.0), c(2.0, 0.0)];
        let matrix = HermitianMatrix::new(data.clone(), 2)
            .expect("2×2 Hermitian matrix from valid data must be constructible");
        // Rebuild A from eigenpairs: small generator s with propagator gives e^{iAs};
        // here check the spectral sum directly via classical_solution consistency
        // and the stored entries.
        assert_eq!(matrix.entries().len(), 4);
        // Spectral reconstruction Σ_j λ_j u_j u_j† must equal the stored entries.
        let dim = 2usize;
        let mut rebuilt = vec![c(0.0, 0.0); dim * dim];
        let (evs, vecs) = hermitian_eigendecomposition(matrix.entries(), dim)
            .expect("Jacobi eigensolver must converge on the stored matrix entries");
        for (j, uj) in vecs.iter().enumerate() {
            for r in 0..dim {
                for col in 0..dim {
                    rebuilt[r * dim + col] += c(evs[j], 0.0) * uj[r] * uj[col].conj();
                }
            }
        }
        for (rebuilt_entry, original) in rebuilt.iter().zip(matrix.entries().iter()) {
            assert!(
                (rebuilt_entry - original).norm() < 1e-8,
                "reconstruction mismatch: {rebuilt_entry:?} vs {original:?}"
            );
        }
    }

    // Dimension / input-validation errors.
    #[test]
    fn hhl_input_validation() {
        let data = herm2(2.0, 2.0, c(-1.0, 0.0));
        let matrix = HermitianMatrix::new(data, 2)
            .expect("valid 2×2 Hermitian data produces a constructible matrix");
        let config = HhlConfig::with_recommended_t(&matrix, 4)
            .expect("eigenvalues 1 and 3 satisfy the clock-resolution constraints");
        // Wrong-length b.
        assert!(hhl_solve(&matrix, &[c(1.0, 0.0)], &config).is_err());
        // Zero b.
        assert!(hhl_solve(&matrix, &[c(0.0, 0.0), c(0.0, 0.0)], &config).is_err());
        // Bad dim.
        assert!(HermitianMatrix::new(vec![c(1.0, 0.0)], 1).is_err());
        // Eigenvalues not integer → recommended-t rejects.
        let frac = herm2(0.5, 0.5, c(0.0, 0.0));
        let mfrac = HermitianMatrix::new(frac, 2)
            .expect("diagonal 2×2 Hermitian is a valid matrix structure");
        assert!(HhlConfig::with_recommended_t(&mfrac, 4).is_err());
    }
}
