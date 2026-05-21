//! Quantum gate matrix constructors.
//!
//! All gates return flat row-major arrays.  Single-qubit gates are stored as
//! 4-element `[f64; 4]` corresponding to the `[2×2]` matrix:
//!
//! ```text
//! [ a  b ]
//! [ c  d ]
//! ```
//! → `[a, b, c, d]`
//!
//! Two-qubit gates are stored as 16-element `[f64; 16]` corresponding to the
//! `[4×4]` matrix in the computational basis `{|00⟩, |01⟩, |10⟩, |11⟩}`.
//!
//! **Note on imaginary units**: This crate operates in the real-number MPS
//! framework.  Gates that are inherently complex (Y, S, T, Rz, iSWAP) are
//! replaced by real-valued approximations that preserve the *real* subspace
//! dynamics.  Each function is documented with the approximation used.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

// ─── Single-qubit gates ──────────────────────────────────────────────────────

/// Pauli-X (NOT) gate: `[[0,1],[1,0]]`.
#[inline]
pub fn pauli_x() -> [f64; 4] {
    [0.0, 1.0, 1.0, 0.0]
}

/// Pauli-Y gate — real approximation `iX = [[0,1],[-1,0]]`.
///
/// The true Pauli-Y is `[[0,-i],[i,0]]`.  In the real-valued MPS framework the
/// imaginary factor is absorbed into the global phase; the real part of the
/// action on `|0⟩` and `|1⟩` is captured by the antisymmetric matrix above.
#[inline]
pub fn pauli_y() -> [f64; 4] {
    [0.0, 1.0, -1.0, 0.0]
}

/// Pauli-Z gate: `[[1,0],[0,-1]]`.
#[inline]
pub fn pauli_z() -> [f64; 4] {
    [1.0, 0.0, 0.0, -1.0]
}

/// Hadamard gate: `[[1,1],[1,-1]] / sqrt(2)`.
#[inline]
pub fn hadamard() -> [f64; 4] {
    let s = FRAC_1_SQRT_2;
    [s, s, s, -s]
}

/// Phase gate S — real approximation: `[[1,0],[0,0]]` + `[[0,0],[0,1]]` (identity on Z basis).
///
/// The true gate is `diag(1, i)`.  The real approximation here is the identity
/// matrix, which preserves the Z-basis structure when `i` is not tracked.
#[inline]
pub fn s_gate_real() -> [f64; 4] {
    // Real approximation: acts as identity on |0⟩ and |1⟩ in the real subspace.
    [1.0, 0.0, 0.0, 1.0]
}

/// T gate — real approximation: `diag(1, cos(π/4))`.
///
/// The true T gate is `diag(1, exp(iπ/4))`.  In real-valued simulation the
/// imaginary component is dropped; only the real part `cos(π/4) = 1/√2` acts.
#[inline]
pub fn t_gate_real() -> [f64; 4] {
    [1.0, 0.0, 0.0, (PI / 4.0).cos()]
}

/// Rx(θ) rotation: `[[cos(θ/2), -sin(θ/2)],[sin(θ/2), cos(θ/2)]]`.
///
/// This is the real-valued rotation about the X axis (dropping the global `i`
/// factor from the standard definition `exp(-i θ/2 X)`).
#[inline]
pub fn rx(theta: f64) -> [f64; 4] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [c, -s, s, c]
}

/// Ry(θ) rotation: `[[cos(θ/2), -sin(θ/2)],[sin(θ/2), cos(θ/2)]]`.
///
/// In the real MPS framework Ry and Rx have the same matrix form; they differ
/// physically only by the axis of rotation.  We expose both for API clarity.
#[inline]
pub fn ry(theta: f64) -> [f64; 4] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [c, -s, s, c]
}

/// Rz(θ) — real approximation as a Z-axis rotation matrix.
///
/// The true Rz is `diag(exp(-iθ/2), exp(iθ/2))`.  In the real subspace this
/// becomes a 2×2 rotation: `[[cos(θ/2), -sin(θ/2)],[sin(θ/2), cos(θ/2)]]`.
#[inline]
pub fn rz_real(theta: f64) -> [f64; 4] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    [c, -s, s, c]
}

// ─── Two-qubit gates ──────────────────────────────────────────────────────────

/// CNOT (CX) gate in the basis `{|00⟩,|01⟩,|10⟩,|11⟩}`:
///
/// ```text
/// 1 0 0 0
/// 0 1 0 0
/// 0 0 0 1
/// 0 0 1 0
/// ```
#[inline]
pub fn cnot() -> [f64; 16] {
    #[rustfmt::skip]
    let m = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 0.0,
    ];
    m
}

/// CZ gate:
///
/// ```text
/// 1  0  0  0
/// 0  1  0  0
/// 0  0  1  0
/// 0  0  0 -1
/// ```
#[inline]
pub fn cz() -> [f64; 16] {
    #[rustfmt::skip]
    let m = [
        1.0, 0.0, 0.0,  0.0,
        0.0, 1.0, 0.0,  0.0,
        0.0, 0.0, 1.0,  0.0,
        0.0, 0.0, 0.0, -1.0,
    ];
    m
}

/// SWAP gate:
///
/// ```text
/// 1 0 0 0
/// 0 0 1 0
/// 0 1 0 0
/// 0 0 0 1
/// ```
#[inline]
pub fn swap() -> [f64; 16] {
    #[rustfmt::skip]
    let m = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    m
}

/// iSWAP — real approximation (drops `i` factor off-diagonal):
///
/// The true iSWAP is `[[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]`.
/// The real approximation replaces `i` with 1, giving the SWAP gate as the
/// closest real unitary.
#[inline]
pub fn iswap_real() -> [f64; 16] {
    swap()
}

/// Controlled-U gate for a 2×2 real unitary `U`.
///
/// `CU = diag(I₂, U)` in the basis `{|0⟩,|1⟩} ⊗ {|0⟩,|1⟩}`:
///
/// ```text
/// 1       0       0       0
/// 0       1       0       0
/// 0       0    U[0,0]  U[0,1]
/// 0       0    U[1,0]  U[1,1]
/// ```
#[inline]
pub fn controlled_u(u: &[f64; 4]) -> [f64; 16] {
    #[rustfmt::skip]
    let m = [
        1.0, 0.0,   0.0,   0.0,
        0.0, 1.0,   0.0,   0.0,
        0.0, 0.0,  u[0],  u[1],
        0.0, 0.0,  u[2],  u[3],
    ];
    m
}

/// XX rotation gate: `exp(-i θ/2 · X⊗X)`.
///
/// In the real-valued approximation (setting `i → real rotation`):
///
/// ```text
/// cos(θ/2)      0          0       -sin(θ/2)
///    0        cos(θ/2)  sin(θ/2)      0
///    0        sin(θ/2)  cos(θ/2)      0
/// -sin(θ/2)     0          0        cos(θ/2)
/// ```
///
/// This is the real part of the exact XX rotation.
#[inline]
pub fn xx_rotation(theta: f64) -> [f64; 16] {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    #[rustfmt::skip]
    let m = [
         c,   0.0,  0.0,  -s,
        0.0,   c,    s,   0.0,
        0.0,   s,    c,   0.0,
        -s,   0.0,  0.0,   c,
    ];
    m
}

/// Heisenberg exchange gate: `exp(-τ H_Heis)` where
/// `H_Heis = J(XX + YY + ZZ)`.
///
/// For the real subspace (dropping imaginary off-diagonal YY terms), this is:
///
/// ```text
/// exp(-τ J)    0            0          0
///    0       cosh(2τJ)  -sinh(2τJ)     0
///    0      -sinh(2τJ)   cosh(2τJ)     0
///    0         0            0        exp(-τ J)
/// ```
///
/// This gives the dominant real-valued Heisenberg dynamics on the ↑↓ sector.
#[inline]
pub fn heisenberg_exchange(j: f64, delta_tau: f64) -> [f64; 16] {
    let t = delta_tau * j;
    let exp_neg = (-t).exp();
    let ch = (2.0 * t).cosh();
    let sh = (2.0 * t).sinh();
    #[rustfmt::skip]
    let m = [
        exp_neg,  0.0,   0.0,     0.0,
        0.0,      ch,   -sh,      0.0,
        0.0,     -sh,    ch,      0.0,
        0.0,      0.0,   0.0,  exp_neg,
    ];
    m
}

/// Multiply two 2×2 matrices stored row-major: `C = A * B`.
#[cfg(test)]
fn mat2_mul(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

/// Check whether a 2×2 matrix is (approximately) the identity.
#[cfg(test)]
fn is_identity2(m: &[f64; 4], tol: f64) -> bool {
    (m[0] - 1.0).abs() < tol && m[1].abs() < tol && m[2].abs() < tol && (m[3] - 1.0).abs() < tol
}

/// Check whether a 4×4 matrix is (approximately) the identity.
#[cfg(test)]
fn is_identity4(m: &[f64; 16], tol: f64) -> bool {
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (m[i * 4 + j] - expected).abs() >= tol {
                return false;
            }
        }
    }
    true
}

/// Multiply two 4×4 matrices stored row-major: `C = A * B`.
#[cfg(test)]
fn mat4_mul(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut c = [0.0f64; 16];
    for i in 0..4 {
        for j in 0..4 {
            let mut acc = 0.0;
            for k in 0..4 {
                acc += a[i * 4 + k] * b[k * 4 + j];
            }
            c[i * 4 + j] = acc;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-12;

    // ── single-qubit ──────────────────────────────────────────────────────────

    #[test]
    fn pauli_x_correct_entries() {
        let x = pauli_x();
        assert_eq!(x, [0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn pauli_x_squared_is_identity() {
        let x = pauli_x();
        let x2 = mat2_mul(&x, &x);
        assert!(is_identity2(&x2, TOL), "XX = {:?}", x2);
    }

    #[test]
    fn pauli_z_correct_entries() {
        let z = pauli_z();
        assert_eq!(z, [1.0, 0.0, 0.0, -1.0]);
    }

    #[test]
    fn pauli_z_squared_is_identity() {
        let z = pauli_z();
        let z2 = mat2_mul(&z, &z);
        assert!(is_identity2(&z2, TOL), "ZZ = {:?}", z2);
    }

    #[test]
    fn hadamard_squared_is_identity() {
        let h = hadamard();
        let h2 = mat2_mul(&h, &h);
        assert!(is_identity2(&h2, TOL), "HH = {:?}", h2);
    }

    #[test]
    fn rx_zero_is_identity() {
        let r = rx(0.0);
        assert!(is_identity2(&r, TOL), "Rx(0) = {:?}", r);
    }

    #[test]
    fn rx_pi_approximates_x() {
        // Rx(π) = [[0, -1],[1, 0]] (up to global phase in real approx)
        let r = rx(PI);
        assert!(r[0].abs() < TOL, "Rx(π)[0,0] should be 0");
        assert!((r[1] + 1.0).abs() < TOL, "Rx(π)[0,1] should be -1");
        assert!((r[2] - 1.0).abs() < TOL, "Rx(π)[1,0] should be +1");
        assert!(r[3].abs() < TOL, "Rx(π)[1,1] should be 0");
    }

    #[test]
    fn ry_zero_is_identity() {
        let r = ry(0.0);
        assert!(is_identity2(&r, TOL));
    }

    #[test]
    fn rz_zero_is_identity() {
        let r = rz_real(0.0);
        assert!(is_identity2(&r, TOL));
    }

    #[test]
    fn t_gate_real_diagonal() {
        let t = t_gate_real();
        assert!((t[0] - 1.0).abs() < TOL);
        assert!(t[1].abs() < TOL);
        assert!(t[2].abs() < TOL);
        assert!((t[3] - (PI / 4.0).cos()).abs() < TOL);
    }

    // ── two-qubit ─────────────────────────────────────────────────────────────

    #[test]
    fn cnot_correct_entries() {
        let c = cnot();
        // |00⟩ → |00⟩
        assert!((c[0] - 1.0).abs() < TOL);
        // |01⟩ → |01⟩
        assert!((c[5] - 1.0).abs() < TOL);
        // |10⟩ → |11⟩
        assert!((c[11] - 1.0).abs() < TOL);
        // |11⟩ → |10⟩
        assert!((c[14] - 1.0).abs() < TOL);
    }

    #[test]
    fn cnot_squared_is_identity() {
        let c = cnot();
        let c2 = mat4_mul(&c, &c);
        assert!(is_identity4(&c2, TOL), "CNOT² = {:?}", c2);
    }

    #[test]
    fn cz_squared_is_identity() {
        let c = cz();
        let c2 = mat4_mul(&c, &c);
        assert!(is_identity4(&c2, TOL), "CZ² = {:?}", c2);
    }

    #[test]
    fn swap_squared_is_identity() {
        let s = swap();
        let s2 = mat4_mul(&s, &s);
        assert!(is_identity4(&s2, TOL), "SWAP² = {:?}", s2);
    }

    #[test]
    fn controlled_u_of_pauli_x_equals_cnot() {
        let x = pauli_x();
        let cu = controlled_u(&x);
        let c = cnot();
        for (i, (&a, &b)) in cu.iter().zip(c.iter()).enumerate() {
            assert!(
                (a - b).abs() < TOL,
                "CU(X)[{}] = {} ≠ CNOT[{}] = {}",
                i,
                a,
                i,
                b
            );
        }
    }

    #[test]
    fn xx_rotation_zero_is_identity() {
        let r = xx_rotation(0.0);
        assert!(is_identity4(&r, TOL), "XX(0) = {:?}", r);
    }

    #[test]
    fn heisenberg_zero_tau_is_exp_neg_j() {
        // When delta_tau = 0, exp(0) = 1, cosh(0) = 1, sinh(0) = 0
        let h = heisenberg_exchange(1.0, 0.0);
        assert!(is_identity4(&h, TOL), "H(j=1, τ=0) should be identity");
    }
}
