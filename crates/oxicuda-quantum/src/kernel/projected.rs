//! Projected quantum kernels (PQK) beyond global overlap fidelity.
//!
//! Reference: Huang, Broughton, Mohseni, Babbush, Boixo, Neven & McClean,
//! "Power of data in quantum machine learning", Nat. Commun. 12, 2631 (2021),
//! §"Projected quantum kernels".
//!
//! The global overlap (fidelity) kernel `k(x,y) = |⟨ψ(x)|ψ(y)⟩|²` concentrates
//! exponentially: for many qubits, off-diagonal entries collapse to ≈0 and the
//! Gram matrix approaches the identity, destroying generalization. The
//! **projected** quantum kernel instead measures only **local** (reduced)
//! information about the embedded state and compares data points in that
//! low-dimensional classical feature space.
//!
//! Concretely, for each data point `x` we embed `|ψ(x)⟩` and record the
//! single-qubit Pauli expectation vector
//!
//! ```text
//! ρ₁(x) = ( ⟨X_q⟩, ⟨Y_q⟩, ⟨Z_q⟩ )_{q=0..n-1}  ∈ ℝ^{3n},
//! ```
//!
//! (equivalent to the Bloch vectors of all single-qubit reduced density
//! matrices). The PQK is the Gaussian (RBF) kernel over these classical feature
//! vectors:
//!
//! ```text
//! k_PQK(x,y) = exp( -γ · Σ_q Σ_{P∈{X,Y,Z}} ( ⟨P_q⟩_x − ⟨P_q⟩_y )² ).
//! ```
//!
//! This is positive-semidefinite by construction (an RBF kernel on real feature
//! vectors) and provably avoids the exponential concentration of the fidelity
//! kernel.

use crate::error::{QuantumError, QuantumResult};
use crate::pauli::expval::expectation_value;
use crate::pauli::hamiltonian::Hamiltonian;
use crate::pauli::pauli_string::PauliOp;
use crate::statevec::state::StateVector;

/// How a data vector is embedded into a quantum state before its local
/// observables are read out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PqkEmbedding {
    /// Angle embedding: `Ry(x_i)` on qubit `i`.
    Angle,
    /// ZZ feature map (Havlíček 2019) with the given number of repetitions.
    ZzFeatureMap { reps: usize },
}

/// Configuration for a projected quantum kernel.
#[derive(Debug, Clone)]
pub struct ProjectedKernelConfig {
    /// RBF bandwidth `γ > 0`.
    pub gamma: f32,
    /// Embedding used to map classical data into quantum states.
    pub embedding: PqkEmbedding,
}

impl ProjectedKernelConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns [`QuantumError::InvalidParameter`] if `gamma <= 0`.
    pub fn new(gamma: f32, embedding: PqkEmbedding) -> QuantumResult<Self> {
        if gamma.is_nan() || gamma <= 0.0 || !gamma.is_finite() {
            return Err(QuantumError::InvalidParameter {
                name: "gamma".into(),
            });
        }
        Ok(Self { gamma, embedding })
    }
}

/// Embed `data` and return the `3·n` vector of single-qubit Pauli expectations
/// `(⟨X_0⟩,⟨Y_0⟩,⟨Z_0⟩, ⟨X_1⟩,…)` — the concatenated Bloch vectors.
///
/// # Errors
/// Propagates embedding / expectation-value errors.
pub fn local_pauli_features(data: &[f32], embedding: PqkEmbedding) -> QuantumResult<Vec<f32>> {
    if data.is_empty() {
        return Err(QuantumError::EmptyInput);
    }
    let sv: StateVector = match embedding {
        PqkEmbedding::Angle => crate::embedding::angle::angle_embedding(data)?,
        PqkEmbedding::ZzFeatureMap { reps } => {
            crate::embedding::zz_feature::zz_feature_map(data, reps.max(1))?
        }
    };
    let n = sv.n_qubits;
    let mut feats = Vec::with_capacity(3 * n);
    for q in 0..n {
        for p in [PauliOp::X, PauliOp::Y, PauliOp::Z] {
            let mut ops = vec![PauliOp::I; n];
            ops[q] = p;
            let mut ham = Hamiltonian::new();
            ham.add_term(1.0, ops);
            let ev = expectation_value(&sv, &ham)?;
            feats.push(ev);
        }
    }
    Ok(feats)
}

/// Compute the projected quantum kernel value `k_PQK(x, y)`.
///
/// # Errors
/// Returns [`QuantumError::DimensionMismatch`] if `x` and `y` differ in length,
/// or propagates embedding errors.
pub fn projected_kernel(x: &[f32], y: &[f32], cfg: &ProjectedKernelConfig) -> QuantumResult<f32> {
    if x.len() != y.len() {
        return Err(QuantumError::DimensionMismatch {
            expected: x.len(),
            got: y.len(),
        });
    }
    let fx = local_pauli_features(x, cfg.embedding)?;
    let fy = local_pauli_features(y, cfg.embedding)?;
    let sq_dist: f32 = fx
        .iter()
        .zip(fy.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum();
    Ok((-cfg.gamma * sq_dist).exp())
}

/// Build the full PQK Gram matrix `K[i][j] = k_PQK(xs[i], xs[j])`.
///
/// Features are computed once per data point and cached, so the cost is
/// `O(m · embed) + O(m²·3n)` rather than `O(m²·embed)`.
///
/// # Errors
/// Returns [`QuantumError::EmptyInput`] for an empty data set; propagates
/// embedding errors.
pub fn projected_kernel_matrix(
    xs: &[Vec<f32>],
    cfg: &ProjectedKernelConfig,
) -> QuantumResult<Vec<Vec<f32>>> {
    if xs.is_empty() {
        return Err(QuantumError::EmptyInput);
    }
    let m = xs.len();
    // Precompute features.
    let mut feats = Vec::with_capacity(m);
    for x in xs {
        feats.push(local_pauli_features(x, cfg.embedding)?);
    }
    let mut mat = vec![vec![0.0_f32; m]; m];
    for i in 0..m {
        for j in i..m {
            let sq_dist: f32 = feats[i]
                .iter()
                .zip(feats[j].iter())
                .map(|(a, b)| {
                    let d = a - b;
                    d * d
                })
                .sum();
            let k = (-cfg.gamma * sq_dist).exp();
            mat[i][j] = k;
            mat[j][i] = k;
        }
    }
    Ok(mat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_bad_gamma() {
        assert!(ProjectedKernelConfig::new(0.0, PqkEmbedding::Angle).is_err());
        assert!(ProjectedKernelConfig::new(-1.0, PqkEmbedding::Angle).is_err());
        assert!(ProjectedKernelConfig::new(1.0, PqkEmbedding::Angle).is_ok());
    }

    #[test]
    fn self_kernel_is_one() {
        let cfg = ProjectedKernelConfig::new(1.0, PqkEmbedding::Angle).expect("cfg");
        let x = vec![0.5_f32, 1.2, 0.3];
        let k = projected_kernel(&x, &x, &cfg).expect("kernel");
        assert!((k - 1.0).abs() < 1e-6, "k={k}");
    }

    #[test]
    fn kernel_in_unit_interval_and_decreases_with_distance() {
        let cfg = ProjectedKernelConfig::new(2.0, PqkEmbedding::Angle).expect("cfg");
        let x = vec![0.1_f32, 0.2];
        let near = vec![0.15_f32, 0.22];
        let far = vec![3.0_f32, -2.5];
        let k_near = projected_kernel(&x, &near, &cfg).expect("k_near");
        let k_far = projected_kernel(&x, &far, &cfg).expect("k_far");
        assert!((0.0..=1.0).contains(&k_near), "k_near={k_near}");
        assert!((0.0..=1.0).contains(&k_far), "k_far={k_far}");
        assert!(k_near > k_far, "near {k_near} should exceed far {k_far}");
    }

    #[test]
    fn gram_matrix_is_symmetric_psd_diag_one() {
        let cfg = ProjectedKernelConfig::new(1.5, PqkEmbedding::Angle).expect("cfg");
        let xs = vec![vec![0.3_f32, 0.7], vec![1.0_f32, -0.5], vec![-0.2_f32, 0.9]];
        let mat = projected_kernel_matrix(&xs, &cfg).expect("matrix");
        // Diagonal == 1, symmetric.
        for (i, row) in mat.iter().enumerate() {
            assert!((row[i] - 1.0).abs() < 1e-6);
            for (j, &v) in row.iter().enumerate() {
                assert!((v - mat[j][i]).abs() < 1e-6);
            }
        }
        // PSD check via a few random test vectors v: vᵀ K v ≥ 0.
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(9);
        for _ in 0..20 {
            let v: Vec<f32> = (0..3).map(|_| rng.next_normal()).collect();
            let mut quad = 0.0_f32;
            for i in 0..3 {
                for j in 0..3 {
                    quad += v[i] * mat[i][j] * v[j];
                }
            }
            assert!(quad >= -1e-4, "quadratic form negative: {quad}");
        }
    }

    #[test]
    fn zz_embedding_variant_works() {
        let cfg =
            ProjectedKernelConfig::new(1.0, PqkEmbedding::ZzFeatureMap { reps: 1 }).expect("cfg");
        let x = vec![0.4_f32, 0.8];
        let y = vec![0.5_f32, 0.7];
        let k = projected_kernel(&x, &y, &cfg).expect("k");
        assert!((0.0..=1.0).contains(&k), "k={k}");
    }
}
