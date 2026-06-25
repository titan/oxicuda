//! Trainable quantum (embedding) kernels optimized by gradient ascent.
//!
//! Reference: Hubregtsen, Wierichs, Gil-Fuster, Derks, Faehrmann & Meyer,
//! "Training quantum embedding kernels on near-term quantum computers",
//! Phys. Rev. A 106, 042431 (2022).
//!
//! A *quantum embedding kernel* is the fidelity kernel of a **parametrized**
//! feature map `|ψ_θ(x)⟩`:
//!
//! ```text
//! k_θ(x, y) = |⟨ψ_θ(x) | ψ_θ(y)⟩|².
//! ```
//!
//! The embedding parameters `θ` are trained (classically, here) to make the
//! kernel useful for a labelled data set by maximizing the **kernel-target
//! alignment** with the ideal label kernel `Y_ij = y_i y_j` (`y_i ∈ {-1,+1}`):
//!
//! ```text
//! A(θ) = ⟨K_θ, Y⟩_F / ( ‖K_θ‖_F · ‖Y‖_F ),
//! ```
//!
//! the Frobenius inner product of the (centered-by-labels) Gram matrix and the
//! outer product of the labels. We optimize `A(θ)` by gradient ascent; gradients
//! are obtained with the parameter-shift rule applied to every kernel entry.
//!
//! ## Trainable embedding
//! The feature map is a single hardware-efficient layer with a per-feature
//! trainable scale: qubit `i` receives `Ry(θ_i · x_i)`, followed by a CNOT
//! entangling ladder. The trainable scales `θ` let the optimizer adapt the
//! data-dependent rotation magnitude (the dominant knob controlling the kernel's
//! effective bandwidth and expressivity).

use crate::error::{QuantumError, QuantumResult};
use crate::gates::controlled::apply_cnot;
use crate::gates::parametric::gate_ry;
use crate::statevec::apply_1q::apply_1q_inplace;
use crate::statevec::state::StateVector;

/// Configuration for kernel-alignment training.
#[derive(Debug, Clone)]
pub struct TrainableKernelConfig {
    /// Number of qubits / features.
    pub n_features: usize,
    /// Gradient-ascent iterations.
    pub iters: usize,
    /// Learning rate for the alignment ascent.
    pub lr: f32,
}

impl TrainableKernelConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// Returns [`QuantumError::InvalidParameter`] for a zero feature count, zero
    /// iterations, or a non-finite/non-positive learning rate.
    pub fn new(n_features: usize, iters: usize, lr: f32) -> QuantumResult<Self> {
        if n_features == 0 {
            return Err(QuantumError::InvalidParameter {
                name: "n_features".into(),
            });
        }
        if iters == 0 {
            return Err(QuantumError::InvalidParameter {
                name: "iters".into(),
            });
        }
        if !lr.is_finite() || lr <= 0.0 {
            return Err(QuantumError::InvalidParameter { name: "lr".into() });
        }
        Ok(Self {
            n_features,
            iters,
            lr,
        })
    }
}

/// A trainable quantum embedding kernel with per-feature rotation scales `θ`.
#[derive(Debug, Clone)]
pub struct TrainableKernel {
    n_features: usize,
    /// Trainable per-feature scale parameters.
    pub theta: Vec<f32>,
}

impl TrainableKernel {
    /// Construct with all scales initialized to `1.0` (plain angle embedding).
    #[must_use]
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            theta: vec![1.0_f32; n_features],
        }
    }

    /// Construct with explicit initial scales.
    ///
    /// # Errors
    /// Returns [`QuantumError::DimensionMismatch`] if `theta.len() != n_features`.
    pub fn with_theta(n_features: usize, theta: Vec<f32>) -> QuantumResult<Self> {
        if theta.len() != n_features {
            return Err(QuantumError::DimensionMismatch {
                expected: n_features,
                got: theta.len(),
            });
        }
        Ok(Self { n_features, theta })
    }

    /// Number of features / qubits.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Embed `x` into `|ψ_θ(x)⟩` with the given scales.
    fn embed_with(&self, x: &[f32], theta: &[f32]) -> QuantumResult<StateVector> {
        if x.len() != self.n_features {
            return Err(QuantumError::DimensionMismatch {
                expected: self.n_features,
                got: x.len(),
            });
        }
        let n = self.n_features;
        let mut sv = StateVector::new_zero_state(n)?;
        for q in 0..n {
            apply_1q_inplace(&mut sv, q, &gate_ry(theta[q] * x[q]))?;
        }
        for q in 0..n.saturating_sub(1) {
            apply_cnot(&mut sv, q, q + 1)?;
        }
        Ok(sv)
    }

    /// Kernel value `k_θ(x,y)` using the **current** `self.theta`.
    ///
    /// # Errors
    /// Propagates embedding errors.
    pub fn kernel(&self, x: &[f32], y: &[f32]) -> QuantumResult<f32> {
        self.kernel_with(x, y, &self.theta)
    }

    /// Kernel value with an explicit scale vector (used for parameter shifts).
    fn kernel_with(&self, x: &[f32], y: &[f32], theta: &[f32]) -> QuantumResult<f32> {
        let psi_x = self.embed_with(x, theta)?;
        let psi_y = self.embed_with(y, theta)?;
        let ip = psi_x.inner_product(&psi_y)?;
        Ok(ip.norm_sqr())
    }

    /// Full Gram matrix for `xs` using the current scales.
    ///
    /// # Errors
    /// Returns [`QuantumError::EmptyInput`] for empty data; propagates embedding
    /// errors.
    pub fn gram(&self, xs: &[Vec<f32>]) -> QuantumResult<Vec<Vec<f32>>> {
        if xs.is_empty() {
            return Err(QuantumError::EmptyInput);
        }
        let m = xs.len();
        let mut mat = vec![vec![0.0_f32; m]; m];
        for i in 0..m {
            for j in i..m {
                let k = self.kernel(&xs[i], &xs[j])?;
                mat[i][j] = k;
                mat[j][i] = k;
            }
        }
        Ok(mat)
    }

    /// Kernel-target alignment `A = ⟨K, Y⟩_F / (‖K‖_F ‖Y‖_F)` for `±1` labels,
    /// evaluated with an explicit `theta`.
    fn alignment_with(&self, xs: &[Vec<f32>], labels: &[f32], theta: &[f32]) -> QuantumResult<f32> {
        let m = xs.len();
        let mut k_dot_y = 0.0_f32;
        let mut k_norm_sq = 0.0_f32;
        let mut y_norm_sq = 0.0_f32;
        for i in 0..m {
            for j in 0..m {
                let kij = self.kernel_with(&xs[i], &xs[j], theta)?;
                let yij = labels[i] * labels[j];
                k_dot_y += kij * yij;
                k_norm_sq += kij * kij;
                y_norm_sq += yij * yij;
            }
        }
        let denom = (k_norm_sq.sqrt() * y_norm_sq.sqrt()).max(1e-12);
        Ok(k_dot_y / denom)
    }

    /// Kernel-target alignment with the current scales.
    ///
    /// # Errors
    /// Returns [`QuantumError::DimensionMismatch`] if `labels.len() != xs.len()`,
    /// [`QuantumError::EmptyInput`] for empty data, and propagates embedding errors.
    pub fn alignment(&self, xs: &[Vec<f32>], labels: &[f32]) -> QuantumResult<f32> {
        if xs.is_empty() {
            return Err(QuantumError::EmptyInput);
        }
        if labels.len() != xs.len() {
            return Err(QuantumError::DimensionMismatch {
                expected: xs.len(),
                got: labels.len(),
            });
        }
        self.alignment_with(xs, labels, &self.theta)
    }

    /// Train the scales `θ` by gradient ascent on the kernel-target alignment.
    ///
    /// Gradients use the parameter-shift rule on each scale component (the
    /// alignment is a smooth function of the scales). Returns the alignment value
    /// at every iteration (length `iters + 1`, including the initial alignment).
    ///
    /// # Errors
    /// Returns [`QuantumError::DimensionMismatch`] on a label/data length
    /// mismatch and propagates embedding errors.
    pub fn train_alignment(
        &mut self,
        xs: &[Vec<f32>],
        labels: &[f32],
        cfg: &TrainableKernelConfig,
    ) -> QuantumResult<Vec<f32>> {
        if xs.is_empty() {
            return Err(QuantumError::EmptyInput);
        }
        if labels.len() != xs.len() {
            return Err(QuantumError::DimensionMismatch {
                expected: xs.len(),
                got: labels.len(),
            });
        }
        let shift = std::f32::consts::FRAC_PI_2;
        let mut history = Vec::with_capacity(cfg.iters + 1);
        history.push(self.alignment_with(xs, labels, &self.theta)?);

        for _ in 0..cfg.iters {
            let mut grad = vec![0.0_f32; self.n_features];
            for p in 0..self.n_features {
                let mut t_plus = self.theta.clone();
                let mut t_minus = self.theta.clone();
                t_plus[p] += shift;
                t_minus[p] -= shift;
                let a_plus = self.alignment_with(xs, labels, &t_plus)?;
                let a_minus = self.alignment_with(xs, labels, &t_minus)?;
                grad[p] = 0.5 * (a_plus - a_minus);
            }
            // Ascent (maximize alignment).
            for (t, g) in self.theta.iter_mut().zip(grad.iter()) {
                *t += cfg.lr * g;
            }
            history.push(self.alignment_with(xs, labels, &self.theta)?);
        }
        Ok(history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validation() {
        assert!(TrainableKernelConfig::new(0, 5, 0.1).is_err());
        assert!(TrainableKernelConfig::new(2, 0, 0.1).is_err());
        assert!(TrainableKernelConfig::new(2, 5, 0.0).is_err());
        assert!(TrainableKernelConfig::new(2, 5, 0.1).is_ok());
    }

    #[test]
    fn self_kernel_is_one() {
        let tk = TrainableKernel::new(2);
        let x = vec![0.5_f32, 1.0];
        let k = tk.kernel(&x, &x).expect("kernel");
        assert!((k - 1.0).abs() < 1e-5, "k={k}");
    }

    #[test]
    fn gram_symmetric_diag_one() {
        let tk = TrainableKernel::new(2);
        let xs = vec![vec![0.3_f32, 0.7], vec![1.0_f32, -0.5]];
        let g = tk.gram(&xs).expect("gram");
        assert!((g[0][0] - 1.0).abs() < 1e-5);
        assert!((g[1][1] - 1.0).abs() < 1e-5);
        assert!((g[0][1] - g[1][0]).abs() < 1e-6);
    }

    #[test]
    fn alignment_in_unit_interval() {
        let tk = TrainableKernel::new(2);
        let xs = vec![
            vec![0.2_f32, 0.1],
            vec![0.3_f32, 0.2],
            vec![2.0_f32, 1.8],
            vec![2.1_f32, 1.9],
        ];
        let labels = vec![1.0_f32, 1.0, -1.0, -1.0];
        let a = tk.alignment(&xs, &labels).expect("alignment");
        assert!((-1.0..=1.0).contains(&a), "alignment={a}");
    }

    #[test]
    fn training_increases_alignment() {
        // Two well-separated clusters with opposite labels; training the scales
        // should not decrease (and generally increases) the alignment.
        let xs = vec![
            vec![0.1_f32, 0.0],
            vec![0.0_f32, 0.1],
            vec![1.5_f32, 1.6],
            vec![1.6_f32, 1.5],
        ];
        let labels = vec![1.0_f32, 1.0, -1.0, -1.0];
        let cfg = TrainableKernelConfig::new(2, 25, 0.4).expect("cfg");
        let mut tk = TrainableKernel::new(2);
        let history = tk.train_alignment(&xs, &labels, &cfg).expect("train");
        let first = history[0];
        let last = *history.last().expect("non-empty history");
        assert!(last >= first - 1e-3, "alignment dropped: {first} → {last}");
        assert!(last.is_finite());
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let tk = TrainableKernel::new(3);
        let x = vec![0.1_f32, 0.2]; // wrong length
        let y = vec![0.1_f32, 0.2, 0.3];
        assert!(tk.kernel(&x, &y).is_err());
    }
}
