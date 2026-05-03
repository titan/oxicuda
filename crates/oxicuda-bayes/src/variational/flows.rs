//! Normalizing flows: PlanarFlow and RadialFlow.
//!
//! Both flows implement invertible transformations with tractable log-determinants
//! of the Jacobian, enabling flexible variational posteriors.

use crate::error::{BayesError, BayesResult};
use crate::handle::LcgRng;

// ─── PlanarFlow ───────────────────────────────────────────────────────────────

/// Planar normalizing flow: `z' = z + u * tanh(w^T z + b)`.
///
/// The invertibility constraint requires `w^T u_hat >= -1` where
/// `u_hat = u + (m(w^T u) - w^T u) * w / ||w||²` and `m(x) = -1 + softplus(x)`.
#[derive(Debug, Clone)]
pub struct PlanarFlow {
    /// Weight vector `w` of dimension D.
    pub w: Vec<f32>,
    /// Unconstrained vector `u` of dimension D (used to compute `u_hat`).
    pub u: Vec<f32>,
    /// Bias scalar `b`.
    pub b: f32,
}

impl PlanarFlow {
    /// Create a new planar flow with random initialization.
    #[must_use]
    pub fn new(dim: usize, rng: &mut LcgRng) -> Self {
        let scale = (1.0_f32 / dim as f32).sqrt();
        let mut w = vec![0.0_f32; dim];
        let mut u = vec![0.0_f32; dim];
        for i in 0..dim {
            // Small random init
            let (a, b) = rng.next_normal_pair();
            w[i] = a * scale;
            u[i] = b * scale;
        }
        Self { w, u, b: 0.0 }
    }

    /// Compute `u_hat` that satisfies the invertibility constraint.
    ///
    /// `u_hat = u + (m(alpha) - alpha) * w / ||w||²`
    /// where `alpha = w^T u` and `m(x) = -1 + softplus(x)`.
    #[must_use]
    pub fn hat_u(&self) -> Vec<f32> {
        let alpha = dot(&self.w, &self.u);
        // m(alpha) = -1 + softplus(alpha) = -1 + ln(1 + exp(alpha))
        let m_alpha = -1.0 + softplus(alpha);
        let correction = m_alpha - alpha;
        let w_norm_sq = dot(&self.w, &self.w).max(1e-10_f32);
        let scale = correction / w_norm_sq;
        self.u
            .iter()
            .zip(self.w.iter())
            .map(|(&ui, &wi)| ui + scale * wi)
            .collect()
    }

    /// Apply the planar flow transformation.
    ///
    /// Returns `(z', log|det J|)` where:
    /// - `z' = z + u_hat * tanh(w^T z + b)`
    /// - `log|det J| = log|1 + u_hat^T * (1 - tanh²(lin)) * w|`
    ///
    /// # Errors
    /// Returns `BayesError::FlowDimensionMismatch` if `z` has wrong dimension,
    /// or `BayesError::NanEncountered` if result is not finite.
    pub fn forward(&self, z: &[f32]) -> BayesResult<(Vec<f32>, f32)> {
        let dim = self.w.len();
        if z.len() != dim {
            return Err(BayesError::FlowDimensionMismatch);
        }
        let u_hat = self.hat_u();
        let lin = dot(&self.w, z) + self.b; // w^T z + b
        let tanh_lin = lin.tanh();
        let tanh_sq = tanh_lin * tanh_lin;
        let psi = 1.0 - tanh_sq; // tanh'(lin)

        // z' = z + u_hat * tanh(lin)
        let z_prime: Vec<f32> = z
            .iter()
            .zip(u_hat.iter())
            .map(|(&zi, &ui)| zi + ui * tanh_lin)
            .collect();

        // log|det J| = log|1 + u_hat^T * psi * w|
        let u_hat_dot_w = dot(&u_hat, &self.w);
        let det_factor = 1.0 + u_hat_dot_w * psi;
        let log_det = det_factor.abs().ln();

        if !log_det.is_finite() {
            return Err(BayesError::NanEncountered {
                location: "PlanarFlow::forward: non-finite log_det",
            });
        }

        Ok((z_prime, log_det))
    }
}

// ─── RadialFlow ───────────────────────────────────────────────────────────────

/// Radial normalizing flow: `z' = z + β_hat * (z - z0) / (α + ||z - z0||)`.
///
/// Invertibility is guaranteed by restricting `β_hat >= -1`.
#[derive(Debug, Clone)]
pub struct RadialFlow {
    /// Reference point `z0` of dimension D.
    pub z0: Vec<f32>,
    /// Scale parameter `α > 0`.
    pub alpha: f32,
    /// Constrained parameter `β_hat ∈ [-1, +∞)`.
    pub beta_hat: f32,
}

impl RadialFlow {
    /// Create a new radial flow with random initialization.
    ///
    /// `z0` is sampled from N(0, 0.1), `alpha` = 1.0, `beta_hat` = 0.0.
    #[must_use]
    pub fn new(dim: usize, rng: &mut LcgRng) -> Self {
        let scale = 0.1_f32;
        let mut z0 = vec![0.0_f32; dim];
        rng.fill_normal(&mut z0);
        for v in z0.iter_mut() {
            *v *= scale;
        }
        Self {
            z0,
            alpha: 1.0,
            beta_hat: 0.0,
        }
    }

    /// Apply the radial flow transformation.
    ///
    /// Returns `(z', log|det J|)` where:
    /// - `r = ||z - z0||`
    /// - `h = 1 / (α + r)`,  `h' = -h²`
    /// - `z' = z + β_hat * (z - z0) * h`
    /// - `log|det J| = (D-1) * ln(1 + β_hat*h) + ln(1 + β_hat*h + β_hat*h'*r)`
    ///
    /// # Errors
    /// Returns `BayesError::FlowDimensionMismatch` if `z` has wrong dimension,
    /// or `BayesError::NanEncountered` if result is not finite.
    pub fn forward(&self, z: &[f32]) -> BayesResult<(Vec<f32>, f32)> {
        let dim = self.z0.len();
        if z.len() != dim {
            return Err(BayesError::FlowDimensionMismatch);
        }

        let diff: Vec<f32> = z
            .iter()
            .zip(self.z0.iter())
            .map(|(&zi, &z0i)| zi - z0i)
            .collect();
        let r = diff
            .iter()
            .map(|d| d * d)
            .sum::<f32>()
            .sqrt()
            .max(1e-10_f32);
        let h = 1.0 / (self.alpha + r);
        let h_prime = -(h * h);

        let scale = self.beta_hat * h;

        // z' = z + beta_hat * h * (z - z0)
        let z_prime: Vec<f32> = z
            .iter()
            .zip(diff.iter())
            .map(|(&zi, &di)| zi + scale * di)
            .collect();

        // log|det J|: D-dimensional Jacobian is:
        // (1 + beta_hat*h + beta_hat*h'*r) * (1 + beta_hat*h)^(D-1)
        let factor1 = 1.0 + self.beta_hat * h + self.beta_hat * h_prime * r;
        let factor2 = 1.0 + self.beta_hat * h;
        let log_det = factor1.abs().ln() + (dim as f32 - 1.0) * factor2.abs().ln();

        if !log_det.is_finite() {
            return Err(BayesError::NanEncountered {
                location: "RadialFlow::forward: non-finite log_det",
            });
        }

        Ok((z_prime, log_det))
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// Numerically stable softplus: `ln(1 + exp(x))`.
#[must_use]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planar_flow_forward_shape() {
        let mut rng = LcgRng::new(42);
        let flow = PlanarFlow::new(4, &mut rng);
        let z = vec![1.0_f32, -1.0, 0.5, 0.0];
        let (z_prime, _) = flow
            .forward(&z)
            .expect("test invariant: forward must succeed");
        assert_eq!(z_prime.len(), 4);
    }

    #[test]
    fn planar_flow_dim_mismatch() {
        let mut rng = LcgRng::new(1);
        let flow = PlanarFlow::new(4, &mut rng);
        let z = vec![1.0_f32; 3];
        assert!(flow.forward(&z).is_err());
    }

    #[test]
    fn planar_flow_log_det_finite() {
        let mut rng = LcgRng::new(5);
        let flow = PlanarFlow::new(8, &mut rng);
        let z = vec![0.5_f32; 8];
        let (_, log_det) = flow
            .forward(&z)
            .expect("test invariant: forward must succeed");
        assert!(log_det.is_finite());
    }

    #[test]
    fn radial_flow_forward_shape() {
        let mut rng = LcgRng::new(7);
        let flow = RadialFlow::new(6, &mut rng);
        let z = vec![0.0_f32; 6];
        let (z_prime, _) = flow
            .forward(&z)
            .expect("test invariant: forward must succeed");
        assert_eq!(z_prime.len(), 6);
    }

    #[test]
    fn radial_flow_dim_mismatch() {
        let mut rng = LcgRng::new(2);
        let flow = RadialFlow::new(4, &mut rng);
        let z = vec![1.0_f32; 3];
        assert!(flow.forward(&z).is_err());
    }

    #[test]
    fn softplus_large_input() {
        // For x > 20, softplus(x) ≈ x
        assert!((softplus(100.0) - 100.0).abs() < 1e-3);
    }

    #[test]
    fn softplus_small_input() {
        // softplus(0) = ln(2)
        assert!((softplus(0.0) - 2.0_f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn hat_u_invertibility() {
        let mut rng = LcgRng::new(13);
        let flow = PlanarFlow::new(4, &mut rng);
        let u_hat = flow.hat_u();
        let w_dot_uhat = dot(&flow.w, &u_hat);
        // Invertibility requires w^T u_hat >= -1
        assert!(w_dot_uhat >= -1.0 - 1e-5, "w^T u_hat = {w_dot_uhat}");
    }
}
