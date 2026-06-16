//! Binary Multi-scale Entanglement Renormalisation Ansatz (MERA) primitives.
//!
//! A binary-MERA layer coarse-grains a 1D lattice in two steps: a **disentangler**
//! `u` (a two-site unitary) removes short-range entanglement across a block
//! boundary, and an **isometry** `w` (a two-into-one isometric map) merges the
//! resulting block into a single coarse site. Iterating the layer realises an
//! entanglement-renormalisation RG flow (Vidal 2007).
//!
//! This module implements the two core super-operators of one MERA layer acting
//! on a two-site block of dimension `d`:
//!
//! - the **ascending** (Schrödinger) channel `A(ρ) = w† u ρ u† w`, which lifts a
//!   fine-lattice density matrix to the coarse lattice, and
//! - its **adjoint / descending** (Heisenberg) channel `A*(O) = u† w O w† u`,
//!   which pulls a coarse-lattice operator back to the fine lattice.
//!
//! These obey the duality `tr[A(ρ) · O] = tr[ρ · A*(O)]`, the defining property of
//! a quantum channel and its adjoint, which the tests verify directly. All
//! tensors are real `f64` (so the adjoint is the transpose); complex MERAs reduce
//! to this case for real Hamiltonians such as the transverse-field Ising and
//! Heisenberg chains used in benchmarks.
//!
//! # References
//! - Vidal, G. (2007). "Entanglement renormalization". *Phys. Rev. Lett.* 99,
//!   220405.
//! - Evenbly, G. & Vidal, G. (2009). "Algorithms for entanglement
//!   renormalization". *Phys. Rev. B* 79, 144108.

use crate::svd::svd_jacobi;
use crate::{TnError, TnResult};

/// One binary-MERA layer acting on a two-site block of local dimension `d`.
///
/// - `disentangler`: a `(d² × d²)` real orthogonal matrix `u` (row-major).
/// - `isometry`: a `(d² × chi)` real matrix `w` whose columns are orthonormal
///   (`wᵀ w = I_χ`); it maps the `d²`-dimensional two-site block to a single
///   coarse site of dimension `chi`.
#[derive(Debug, Clone)]
pub struct MeraLayer {
    /// Local (fine) site dimension.
    pub d: usize,
    /// Coarse site dimension `χ ≤ d²`.
    pub chi: usize,
    /// `(d²×d²)` orthogonal disentangler, row-major.
    pub disentangler: Vec<f64>,
    /// `(d²×chi)` column-orthonormal isometry, row-major.
    pub isometry: Vec<f64>,
}

// ─── Small dense linear-algebra helpers (row-major) ─────────────────────────

/// `C(m×p) = A(m×k) · B(k×p)`.
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, p: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * p];
    for i in 0..m {
        for kk in 0..k {
            let aik = a[i * k + kk];
            if aik == 0.0 {
                continue;
            }
            for j in 0..p {
                c[i * p + j] += aik * b[kk * p + j];
            }
        }
    }
    c
}

/// Transpose of an `r×c` matrix.
fn transpose(a: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut t = vec![0.0_f64; r * c];
    for i in 0..r {
        for j in 0..c {
            t[j * r + i] = a[i * c + j];
        }
    }
    t
}

/// `tr(A·B)` for two `n×n` matrices, computed without forming the product.
fn trace_product(a: &[f64], b: &[f64], n: usize) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            s += a[i * n + j] * b[j * n + i];
        }
    }
    s
}

/// Orthonormalise the columns of an `(m×k)` real matrix by a thin SVD
/// (`A = U Σ Vᵀ` ⇒ the orthonormal factor is `U Vᵀ`, the nearest isometry).
fn nearest_isometry(a: &[f64], m: usize, k: usize) -> TnResult<Vec<f64>> {
    let svd = svd_jacobi(a, m, k)?;
    // U is (m × r), Vt is (r × k); the polar factor U·Vᵀ→ we want (m×k) isometry
    // = U(:, :r) · Vt(:r, :). Since r = min(m,k) = k here (m ≥ k expected).
    let r = svd.k;
    let mut w = vec![0.0_f64; m * k];
    for i in 0..m {
        for j in 0..k {
            let mut acc = 0.0_f64;
            for t in 0..r {
                acc += svd.u[i * r + t] * svd.vt[t * k + j];
            }
            w[i * k + j] = acc;
        }
    }
    Ok(w)
}

/// Deterministic pseudo-random matrix generator for building random unitaries /
/// isometries (centred in `[-0.5, 0.5)`).
struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }
    fn next(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 33) as f64) / (1u64 << 31) as f64 - 0.5
    }
}

impl MeraLayer {
    /// Build a MERA layer with a random orthogonal disentangler and a random
    /// isometry of coarse dimension `chi`.
    ///
    /// # Errors
    /// - [`TnError::InvalidConfiguration`] if `d < 2` or `chi == 0` or `chi > d²`.
    /// - propagates SVD failures from the orthonormalisation step.
    pub fn random(d: usize, chi: usize, seed: u64) -> TnResult<Self> {
        if d < 2 || chi == 0 || chi > d * d {
            return Err(TnError::InvalidConfiguration(format!(
                "MERA layer requires d≥2 and 1≤χ≤d² (got d={d}, χ={chi})"
            )));
        }
        let dd = d * d;
        let mut rng = Lcg::new(seed);

        // Disentangler: orthonormalise a random (dd×dd) matrix → orthogonal.
        let mut raw_u = vec![0.0_f64; dd * dd];
        for v in raw_u.iter_mut() {
            *v = rng.next();
        }
        let disentangler = nearest_isometry(&raw_u, dd, dd)?;

        // Isometry: orthonormalise a random (dd×chi) matrix → column-orthonormal.
        let mut raw_w = vec![0.0_f64; dd * chi];
        for v in raw_w.iter_mut() {
            *v = rng.next();
        }
        let isometry = nearest_isometry(&raw_w, dd, chi)?;

        Ok(Self {
            d,
            chi,
            disentangler,
            isometry,
        })
    }

    /// Construct a layer from explicit tensors (validated for shape only).
    ///
    /// # Errors
    /// - [`TnError::ShapeMismatch`] if either buffer has the wrong length.
    /// - [`TnError::InvalidConfiguration`] if `d < 2` or `chi` is out of range.
    pub fn from_tensors(
        d: usize,
        chi: usize,
        disentangler: Vec<f64>,
        isometry: Vec<f64>,
    ) -> TnResult<Self> {
        if d < 2 || chi == 0 || chi > d * d {
            return Err(TnError::InvalidConfiguration(format!(
                "MERA layer requires d≥2 and 1≤χ≤d² (got d={d}, χ={chi})"
            )));
        }
        let dd = d * d;
        if disentangler.len() != dd * dd {
            return Err(TnError::ShapeMismatch {
                expected: vec![dd, dd],
                got: vec![disentangler.len()],
            });
        }
        if isometry.len() != dd * chi {
            return Err(TnError::ShapeMismatch {
                expected: vec![dd, chi],
                got: vec![isometry.len()],
            });
        }
        Ok(Self {
            d,
            chi,
            disentangler,
            isometry,
        })
    }

    /// Maximum deviation of `uᵀu` from the identity (disentangler unitarity error).
    pub fn disentangler_error(&self) -> f64 {
        let dd = self.d * self.d;
        let ut = transpose(&self.disentangler, dd, dd);
        let prod = matmul(&ut, &self.disentangler, dd, dd, dd);
        let mut err = 0.0_f64;
        for i in 0..dd {
            for j in 0..dd {
                let expected = if i == j { 1.0 } else { 0.0 };
                err = err.max((prod[i * dd + j] - expected).abs());
            }
        }
        err
    }

    /// Maximum deviation of `wᵀw` from the identity (isometry condition error).
    pub fn isometry_error(&self) -> f64 {
        let dd = self.d * self.d;
        let wt = transpose(&self.isometry, dd, self.chi);
        let prod = matmul(&wt, &self.isometry, self.chi, dd, self.chi);
        let mut err = 0.0_f64;
        for i in 0..self.chi {
            for j in 0..self.chi {
                let expected = if i == j { 1.0 } else { 0.0 };
                err = err.max((prod[i * self.chi + j] - expected).abs());
            }
        }
        err
    }

    /// Ascending (Schrödinger) channel `A(ρ) = wᵀ u ρ uᵀ w`.
    ///
    /// Lifts a fine two-site density matrix `ρ` (shape `d²×d²`) to a coarse
    /// single-site density matrix of shape `chi×chi`.
    ///
    /// # Errors
    /// - [`TnError::ShapeMismatch`] if `rho.len() != (d²)²`.
    pub fn ascend_density(&self, rho: &[f64]) -> TnResult<Vec<f64>> {
        let dd = self.d * self.d;
        if rho.len() != dd * dd {
            return Err(TnError::ShapeMismatch {
                expected: vec![dd, dd],
                got: vec![rho.len()],
            });
        }
        // tmp = u ρ uᵀ  (dd×dd)
        let ut = transpose(&self.disentangler, dd, dd);
        let u_rho = matmul(&self.disentangler, rho, dd, dd, dd);
        let u_rho_ut = matmul(&u_rho, &ut, dd, dd, dd);
        // coarse = wᵀ (u ρ uᵀ) w   (chi×chi)
        let wt = transpose(&self.isometry, dd, self.chi);
        let left = matmul(&wt, &u_rho_ut, self.chi, dd, dd); // chi×dd
        let coarse = matmul(&left, &self.isometry, self.chi, dd, self.chi);
        Ok(coarse)
    }

    /// Descending / adjoint (Heisenberg) channel `A*(O) = uᵀ w O wᵀ u`.
    ///
    /// Pulls a coarse single-site operator `O` (shape `chi×chi`) back to a fine
    /// two-site operator of shape `d²×d²`.
    ///
    /// # Errors
    /// - [`TnError::ShapeMismatch`] if `op.len() != chi²`.
    pub fn descend_operator(&self, op: &[f64]) -> TnResult<Vec<f64>> {
        let dd = self.d * self.d;
        if op.len() != self.chi * self.chi {
            return Err(TnError::ShapeMismatch {
                expected: vec![self.chi, self.chi],
                got: vec![op.len()],
            });
        }
        // tmp = w O wᵀ   (dd×dd)
        let wt = transpose(&self.isometry, dd, self.chi);
        let w_op = matmul(&self.isometry, op, dd, self.chi, self.chi); // dd×chi
        let w_op_wt = matmul(&w_op, &wt, dd, self.chi, dd); // dd×dd
        // fine = uᵀ (w O wᵀ) u   (dd×dd)
        let ut = transpose(&self.disentangler, dd, dd);
        let left = matmul(&ut, &w_op_wt, dd, dd, dd);
        let fine = matmul(&left, &self.disentangler, dd, dd, dd);
        Ok(fine)
    }

    /// Expectation value of a coarse operator `O` in a coarse density `σ`,
    /// `tr(σ O)`.
    ///
    /// # Errors
    /// - [`TnError::ShapeMismatch`] if either matrix is not `chi×chi`.
    pub fn coarse_expectation(&self, sigma: &[f64], op: &[f64]) -> TnResult<f64> {
        if sigma.len() != self.chi * self.chi || op.len() != self.chi * self.chi {
            return Err(TnError::ShapeMismatch {
                expected: vec![self.chi, self.chi],
                got: vec![sigma.len().max(op.len())],
            });
        }
        Ok(trace_product(sigma, op, self.chi))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trace of an `n×n` matrix.
    fn trace(a: &[f64], n: usize) -> f64 {
        (0..n).map(|i| a[i * n + i]).sum()
    }

    /// Identity matrix of size `n` (row-major).
    fn eye(n: usize) -> Vec<f64> {
        let mut e = vec![0.0_f64; n * n];
        for i in 0..n {
            e[i * n + i] = 1.0;
        }
        e
    }

    /// A deterministic symmetric "density-like" matrix ρ ≽ 0 with tr ρ = 1.
    fn random_density(n: usize, seed: u64) -> Vec<f64> {
        // ρ = M Mᵀ / tr(M Mᵀ) with random M.
        let mut rng = Lcg::new(seed);
        let mut m = vec![0.0_f64; n * n];
        for v in m.iter_mut() {
            *v = rng.next();
        }
        let mt = transpose(&m, n, n);
        let mut rho = matmul(&m, &mt, n, n, n);
        let tr = trace(&rho, n);
        for v in rho.iter_mut() {
            *v /= tr;
        }
        rho
    }

    fn random_symmetric(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Lcg::new(seed);
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in i..n {
                let v = rng.next();
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        a
    }

    #[test]
    fn random_layer_rejects_bad_config() {
        assert!(matches!(
            MeraLayer::random(1, 2, 0),
            Err(TnError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            MeraLayer::random(2, 0, 0),
            Err(TnError::InvalidConfiguration(_))
        ));
        // chi > d² is invalid.
        assert!(matches!(
            MeraLayer::random(2, 5, 0),
            Err(TnError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn from_tensors_validates_shapes() {
        assert!(matches!(
            MeraLayer::from_tensors(2, 2, vec![0.0; 10], vec![0.0; 8]),
            Err(TnError::ShapeMismatch { .. })
        ));
        let ok = MeraLayer::from_tensors(2, 2, eye(4), {
            // (4×2) isometry: first two columns of I4.
            let mut w = vec![0.0; 8];
            w[0] = 1.0;
            w[3] = 1.0;
            w
        });
        assert!(ok.is_ok());
    }

    #[test]
    fn disentangler_is_orthogonal() {
        let layer = MeraLayer::random(2, 4, 42).expect("layer");
        assert!(
            layer.disentangler_error() < 1e-9,
            "uᵀu error = {}",
            layer.disentangler_error()
        );
    }

    #[test]
    fn isometry_is_orthonormal() {
        let layer = MeraLayer::random(2, 2, 99).expect("layer");
        assert!(
            layer.isometry_error() < 1e-9,
            "wᵀw error = {}",
            layer.isometry_error()
        );
    }

    #[test]
    fn ascend_density_preserves_trace_when_chi_full() {
        // With χ = d² the isometry is a full orthogonal matrix, so the ascending
        // channel is trace-preserving: tr A(ρ) = tr ρ = 1.
        let layer = MeraLayer::random(2, 4, 7).expect("layer");
        let rho = random_density(4, 123);
        let coarse = layer.ascend_density(&rho).expect("ascend");
        assert!(
            (trace(&coarse, 4) - 1.0).abs() < 1e-9,
            "tr = {}",
            trace(&coarse, 4)
        );
    }

    #[test]
    fn ascend_density_trace_nonincreasing_when_truncating() {
        // With χ < d² the channel can only lose weight: 0 ≤ tr A(ρ) ≤ 1.
        let layer = MeraLayer::random(2, 2, 5).expect("layer");
        let rho = random_density(4, 321);
        let coarse = layer.ascend_density(&rho).expect("ascend");
        let t = trace(&coarse, 2);
        assert!(t > 0.0 && t <= 1.0 + 1e-9, "trace {t} out of (0, 1]");
    }

    #[test]
    fn descend_operator_shape_and_identity() {
        // A* maps a chi×chi operator to a d²×d² operator.
        let layer = MeraLayer::random(2, 2, 11).expect("layer");
        let id_coarse = eye(2);
        let fine = layer.descend_operator(&id_coarse).expect("descend");
        assert_eq!(fine.len(), 16);
        // A*(I_χ) = uᵀ w wᵀ u is a projector of rank χ ⇒ trace = χ.
        assert!(
            (trace(&fine, 4) - 2.0).abs() < 1e-9,
            "tr A*(I)={}",
            trace(&fine, 4)
        );
    }

    #[test]
    fn ascend_descend_adjoint_duality() {
        // The defining property: tr[A(ρ) · O] = tr[ρ · A*(O)] for all ρ, O.
        let layer = MeraLayer::random(2, 2, 2024).expect("layer");
        let rho = random_density(4, 1);
        let op = random_symmetric(2, 2);
        let coarse = layer.ascend_density(&rho).expect("ascend");
        let lhs = trace_product(&coarse, &op, 2);
        let fine_op = layer.descend_operator(&op).expect("descend");
        let rhs = trace_product(&rho, &fine_op, 4);
        assert!(
            (lhs - rhs).abs() < 1e-9,
            "adjoint duality violated: lhs={lhs}, rhs={rhs}"
        );
    }

    #[test]
    fn adjoint_duality_full_chi() {
        // Same duality but with χ = d² (full orthogonal isometry).
        let layer = MeraLayer::random(2, 4, 77).expect("layer");
        let rho = random_density(4, 9);
        let op = random_symmetric(4, 8);
        let coarse = layer.ascend_density(&rho).expect("ascend");
        let lhs = trace_product(&coarse, &op, 4);
        let fine_op = layer.descend_operator(&op).expect("descend");
        let rhs = trace_product(&rho, &fine_op, 4);
        assert!((lhs - rhs).abs() < 1e-9, "lhs={lhs}, rhs={rhs}");
    }

    #[test]
    fn ascend_density_wrong_shape_errors() {
        let layer = MeraLayer::random(2, 2, 3).expect("layer");
        assert!(matches!(
            layer.ascend_density(&[0.0; 9]),
            Err(TnError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn descend_operator_wrong_shape_errors() {
        let layer = MeraLayer::random(2, 2, 3).expect("layer");
        assert!(matches!(
            layer.descend_operator(&[0.0; 9]),
            Err(TnError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn coarse_expectation_matches_trace_product() {
        let layer = MeraLayer::random(2, 2, 13).expect("layer");
        let sigma = random_density(2, 4);
        let op = random_symmetric(2, 5);
        let e = layer.coarse_expectation(&sigma, &op).expect("exp");
        assert!((e - trace_product(&sigma, &op, 2)).abs() < 1e-12);
    }

    #[test]
    fn ascend_symmetric_density_stays_symmetric() {
        // A(ρ) of a symmetric ρ must be symmetric (Hermitian for real case).
        let layer = MeraLayer::random(2, 3, 31).expect("layer");
        let rho = random_density(4, 6);
        let coarse = layer.ascend_density(&rho).expect("ascend");
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (coarse[i * 3 + j] - coarse[j * 3 + i]).abs() < 1e-9,
                    "coarse density not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn d3_layer_builds_and_duality_holds() {
        // Qutrits (d=3, d²=9), χ=4.
        let layer = MeraLayer::random(3, 4, 314).expect("layer");
        assert!(layer.disentangler_error() < 1e-8);
        assert!(layer.isometry_error() < 1e-8);
        let rho = random_density(9, 2);
        let op = random_symmetric(4, 3);
        let coarse = layer.ascend_density(&rho).expect("ascend");
        let lhs = trace_product(&coarse, &op, 4);
        let fine_op = layer.descend_operator(&op).expect("descend");
        let rhs = trace_product(&rho, &fine_op, 9);
        assert!((lhs - rhs).abs() < 1e-8, "lhs={lhs}, rhs={rhs}");
    }
}
