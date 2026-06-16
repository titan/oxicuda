//! X-PINN: eXtended Physics-Informed Neural Networks with domain decomposition.
//!
//! Jagtap & Karniadakis (2020) "Extended Physics-Informed Neural Networks
//! (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning
//! Framework for Nonlinear Partial Differential Equations", Communications in
//! Computational Physics, vol. 28(5), pp. 2002-2041.
//!
//! XPINNs partition the domain into **non-overlapping** subdomains, each solved by
//! its own neural network `u_j`. Adjacent subdomains share a common **interface**
//! on which two extra conditions are enforced so that the patched global solution
//! is consistent:
//!
//! 1. **Solution continuity** (average / interface condition): the two networks
//!    must agree on the shared interface,
//!    `u_i(x) = u_j(x)` for `x ∈ Γ_{ij}`. XPINN penalises the deviation of each
//!    network from the interface average `ū = ½(u_i + u_j)`.
//! 2. **Residual continuity**: the PDE residuals computed from each side must also
//!    agree, `R_i(x) = R_j(x)` on `Γ_{ij}`. This couples the physics across the
//!    interface (the generalisation of XPINN over the earlier conservative PINN
//!    flux condition).
//!
//! This module provides the **interface-coupling loss machinery**. A subdomain is
//! a 1-D interval; the decomposition is a contiguous chain of intervals sharing
//! single-point interfaces. Given each subdomain network's value and residual at
//! the shared interface points, the assembler computes the average-continuity and
//! residual-continuity penalties that are added to the per-subdomain interior
//! residual losses. The subdomain networks themselves are plain [`Mlp`]s, so the
//! caller supplies their interface evaluations.

use crate::error::{PinnError, PinnResult};
use crate::handle::LcgRng;
use crate::network::mlp::{Activation, Mlp, MlpConfig};

/// A single 1-D subdomain `[lo, hi]` owned by one network.
#[derive(Debug, Clone, Copy)]
pub struct XPinnSubdomain {
    /// Lower bound of the subdomain interval.
    pub lo: f32,
    /// Upper bound of the subdomain interval.
    pub hi: f32,
}

impl XPinnSubdomain {
    /// Whether `x` lies within `[lo, hi]`.
    #[must_use]
    pub fn contains(&self, x: f32) -> bool {
        x >= self.lo && x <= self.hi
    }

    /// Map `x ∈ [lo, hi]` to the local normalised coordinate `[-1, 1]`.
    #[must_use]
    pub fn normalize(&self, x: f32) -> f32 {
        let w = self.hi - self.lo;
        if w <= 0.0 {
            return 0.0;
        }
        2.0 * (x - self.lo) / w - 1.0
    }
}

/// Configuration for an XPINN decomposition of a 1-D domain.
#[derive(Debug, Clone)]
pub struct XPinnConfig {
    /// Lower bound of the global domain.
    pub domain_lo: f32,
    /// Upper bound of the global domain.
    pub domain_hi: f32,
    /// Number of non-overlapping subdomains (`>= 1`).
    pub n_subdomains: usize,
    /// Hidden width of each subdomain MLP.
    pub hidden: usize,
    /// Weight `w_u` on the interface solution-continuity (average) penalty.
    pub w_interface: f32,
    /// Weight `w_r` on the interface residual-continuity penalty.
    pub w_residual: f32,
}

impl Default for XPinnConfig {
    fn default() -> Self {
        Self {
            domain_lo: 0.0,
            domain_hi: 1.0,
            n_subdomains: 3,
            hidden: 16,
            w_interface: 20.0,
            w_residual: 1.0,
        }
    }
}

/// Breakdown of the XPINN interface loss into its physical terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XPinnInterfaceLoss {
    /// Solution-continuity (average) penalty `Σ_Γ (u_L − ū)² + (u_R − ū)²`.
    pub interface_term: f32,
    /// Residual-continuity penalty `Σ_Γ (R_L − R_R)²`.
    pub residual_term: f32,
    /// Weighted total `w_u · interface_term + w_r · residual_term`.
    pub total: f32,
}

/// Extended PINN over a chain of non-overlapping 1-D subdomains.
pub struct XPinn {
    /// One subdomain interval per network, left-to-right.
    pub subdomains: Vec<XPinnSubdomain>,
    /// One MLP per subdomain (`1 → hidden → 1`, `tanh` activation).
    pub networks: Vec<Mlp>,
    config: XPinnConfig,
}

impl XPinn {
    /// Construct an XPINN by splitting `[domain_lo, domain_hi]` into
    /// `n_subdomains` equal contiguous intervals, one `tanh` MLP each.
    ///
    /// # Errors
    /// - [`PinnError::InvalidTimeInterval`] if `domain_hi <= domain_lo`.
    /// - [`PinnError::InvalidGridResolution`] if `n_subdomains == 0`.
    /// - [`PinnError::InvalidLayerWidth`] if `hidden == 0`.
    /// - [`PinnError::InvalidWeight`] if either interface weight is negative / non-finite.
    pub fn new(config: XPinnConfig, rng: &mut LcgRng) -> PinnResult<Self> {
        if !config.domain_lo.is_finite()
            || !config.domain_hi.is_finite()
            || config.domain_hi <= config.domain_lo
        {
            return Err(PinnError::InvalidTimeInterval {
                t0: config.domain_lo,
                t1: config.domain_hi,
            });
        }
        if config.n_subdomains == 0 {
            return Err(PinnError::InvalidGridResolution {
                n: config.n_subdomains,
            });
        }
        if config.hidden == 0 {
            return Err(PinnError::InvalidLayerWidth);
        }
        for w in [config.w_interface, config.w_residual] {
            if !w.is_finite() || w < 0.0 {
                return Err(PinnError::InvalidWeight { weight: w });
            }
        }

        let h = (config.domain_hi - config.domain_lo) / config.n_subdomains as f32;
        let mut subdomains = Vec::with_capacity(config.n_subdomains);
        let mut networks = Vec::with_capacity(config.n_subdomains);
        for j in 0..config.n_subdomains {
            let lo = config.domain_lo + j as f32 * h;
            subdomains.push(XPinnSubdomain { lo, hi: lo + h });
            let cfg = MlpConfig {
                layer_widths: vec![1, config.hidden, 1],
                activation: Activation::Tanh,
                omega_0: 1.0,
            };
            networks.push(Mlp::new(cfg, rng)?);
        }

        Ok(Self {
            subdomains,
            networks,
            config,
        })
    }

    /// Number of subdomains.
    #[must_use]
    pub fn n_subdomains(&self) -> usize {
        self.subdomains.len()
    }

    /// The `n_subdomains − 1` interior interface coordinates (shared endpoints).
    #[must_use]
    pub fn interface_points(&self) -> Vec<f32> {
        (1..self.subdomains.len())
            .map(|j| self.subdomains[j].lo)
            .collect()
    }

    /// Index of the subdomain that owns coordinate `x` (first interval whose
    /// `[lo, hi]` contains it; the last subdomain absorbs the right endpoint).
    #[must_use]
    pub fn locate(&self, x: f32) -> Option<usize> {
        self.subdomains.iter().position(|s| s.contains(x))
    }

    /// Forward pass of the global patched solution at `x`.
    ///
    /// Evaluates the owning subdomain's MLP on the locally-normalised coordinate.
    /// At an interior interface, the average of the two adjacent networks is
    /// returned (well-defined once the interface penalty has been minimised).
    ///
    /// # Errors
    /// - [`PinnError::EmptyInput`] if `x` is outside the global domain.
    /// - Propagates MLP forward errors.
    pub fn forward(&self, x: f32) -> PinnResult<f32> {
        // Detect an interior interface: average the two neighbours.
        for j in 1..self.subdomains.len() {
            if (x - self.subdomains[j].lo).abs() < 1e-7 {
                let left = self.eval_subdomain(j - 1, x)?;
                let right = self.eval_subdomain(j, x)?;
                return Ok(0.5 * (left + right));
            }
        }
        let idx = self.locate(x).ok_or(PinnError::EmptyInput)?;
        self.eval_subdomain(idx, x)
    }

    /// Evaluate subdomain `idx`'s network at physical coordinate `x`.
    ///
    /// # Errors
    /// - [`PinnError::TapeIndexOutOfRange`] (re-used) if `idx` is out of range.
    /// - Propagates MLP forward errors.
    pub fn eval_subdomain(&self, idx: usize, x: f32) -> PinnResult<f32> {
        let sub = self
            .subdomains
            .get(idx)
            .ok_or(PinnError::TapeIndexOutOfRange {
                idx,
                size: self.subdomains.len(),
            })?;
        let xn = sub.normalize(x);
        let out = self.networks[idx].forward(&[xn])?;
        Ok(out[0])
    }

    /// Assemble the XPINN interface loss from per-interface evaluations.
    ///
    /// For each of the `n_subdomains − 1` interfaces, the caller supplies:
    /// - `u_left[i]`  : the left network's value at interface `i`,
    /// - `u_right[i]` : the right network's value at interface `i`,
    /// - `r_left[i]`  : the left network's PDE residual at interface `i`,
    /// - `r_right[i]` : the right network's PDE residual at interface `i`.
    ///
    /// The solution-continuity penalty pushes both sides toward the interface
    /// average `ū = ½(u_L + u_R)`:
    /// `interface_term = Σ_i (u_L − ū)² + (u_R − ū)² = Σ_i ½ (u_L − u_R)²`.
    /// The residual-continuity penalty is `Σ_i (R_L − R_R)²`.
    ///
    /// # Errors
    /// - [`PinnError::DimensionMismatch`] if the four slices are not all length
    ///   `n_subdomains − 1`.
    /// - [`PinnError::NanEncountered`] on non-finite inputs / results.
    pub fn interface_loss(
        &self,
        u_left: &[f32],
        u_right: &[f32],
        r_left: &[f32],
        r_right: &[f32],
    ) -> PinnResult<XPinnInterfaceLoss> {
        let n_iface = self.subdomains.len().saturating_sub(1);
        for (slice, len) in [
            (u_left.len(), n_iface),
            (u_right.len(), n_iface),
            (r_left.len(), n_iface),
            (r_right.len(), n_iface),
        ] {
            if slice != len {
                return Err(PinnError::DimensionMismatch {
                    expected: len,
                    got: slice,
                });
            }
        }
        if n_iface == 0 {
            return Ok(XPinnInterfaceLoss {
                interface_term: 0.0,
                residual_term: 0.0,
                total: 0.0,
            });
        }
        if [u_left, u_right, r_left, r_right]
            .iter()
            .any(|s| s.iter().any(|v| !v.is_finite()))
        {
            return Err(PinnError::NanEncountered {
                location: "xpinn::interface_loss(input)",
            });
        }

        let mut interface_term = 0.0_f32;
        let mut residual_term = 0.0_f32;
        for i in 0..n_iface {
            let avg = 0.5 * (u_left[i] + u_right[i]);
            interface_term += (u_left[i] - avg).powi(2) + (u_right[i] - avg).powi(2);
            residual_term += (r_left[i] - r_right[i]).powi(2);
        }
        interface_term /= n_iface as f32;
        residual_term /= n_iface as f32;

        let total =
            self.config.w_interface * interface_term + self.config.w_residual * residual_term;
        if !total.is_finite() {
            return Err(PinnError::NanEncountered {
                location: "xpinn::interface_loss(output)",
            });
        }
        Ok(XPinnInterfaceLoss {
            interface_term,
            residual_term,
            total,
        })
    }

    /// Evaluate the interface loss directly from the subdomain networks
    /// (solution-continuity only) at every interface point.
    ///
    /// `residual_fn(idx, x)` returns the PDE residual of subdomain `idx` at `x`;
    /// pass a closure wrapping finite-difference or autodiff derivatives.
    ///
    /// # Errors
    /// Propagates evaluation errors.
    pub fn interface_loss_from_networks<F>(&self, residual_fn: F) -> PinnResult<XPinnInterfaceLoss>
    where
        F: Fn(usize, f32) -> PinnResult<f32>,
    {
        let ifaces = self.interface_points();
        let mut u_left = Vec::with_capacity(ifaces.len());
        let mut u_right = Vec::with_capacity(ifaces.len());
        let mut r_left = Vec::with_capacity(ifaces.len());
        let mut r_right = Vec::with_capacity(ifaces.len());
        for (i, &x) in ifaces.iter().enumerate() {
            // interface i lies between subdomain i and i+1
            u_left.push(self.eval_subdomain(i, x)?);
            u_right.push(self.eval_subdomain(i + 1, x)?);
            r_left.push(residual_fn(i, x)?);
            r_right.push(residual_fn(i + 1, x)?);
        }
        self.interface_loss(&u_left, &u_right, &r_left, &r_right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn construct_partitions_domain() {
        let mut rng = LcgRng::new(1);
        let cfg = XPinnConfig {
            domain_lo: 0.0,
            domain_hi: 3.0,
            n_subdomains: 3,
            ..Default::default()
        };
        let xp =
            XPinn::new(cfg, &mut rng).expect("XPinn construction with valid config should succeed");
        assert_eq!(xp.n_subdomains(), 3);
        assert!(approx(xp.subdomains[0].lo, 0.0, 1e-6));
        assert!(approx(xp.subdomains[0].hi, 1.0, 1e-6));
        assert!(approx(xp.subdomains[2].hi, 3.0, 1e-6));
    }

    #[test]
    fn interface_points_are_shared_endpoints() {
        let mut rng = LcgRng::new(2);
        let cfg = XPinnConfig {
            domain_lo: 0.0,
            domain_hi: 4.0,
            n_subdomains: 4,
            ..Default::default()
        };
        let xp =
            XPinn::new(cfg, &mut rng).expect("XPinn construction with valid config should succeed");
        let ifaces = xp.interface_points();
        assert_eq!(ifaces.len(), 3);
        assert!(approx(ifaces[0], 1.0, 1e-6));
        assert!(approx(ifaces[1], 2.0, 1e-6));
        assert!(approx(ifaces[2], 3.0, 1e-6));
    }

    #[test]
    fn locate_returns_owning_subdomain() {
        let mut rng = LcgRng::new(3);
        let xp = XPinn::new(XPinnConfig::default(), &mut rng)
            .expect("XPinn construction with valid config should succeed");
        // default: [0,1] split into 3 → widths 1/3
        assert_eq!(xp.locate(0.1), Some(0));
        assert_eq!(xp.locate(0.5), Some(1));
        assert_eq!(xp.locate(0.9), Some(2));
        assert_eq!(xp.locate(2.0), None);
    }

    #[test]
    fn normalize_maps_to_unit_interval() {
        let sub = XPinnSubdomain { lo: 2.0, hi: 4.0 };
        assert!(approx(sub.normalize(2.0), -1.0, 1e-6));
        assert!(approx(sub.normalize(3.0), 0.0, 1e-6));
        assert!(approx(sub.normalize(4.0), 1.0, 1e-6));
    }

    #[test]
    fn forward_evaluates_and_is_finite() {
        let mut rng = LcgRng::new(4);
        let xp = XPinn::new(XPinnConfig::default(), &mut rng)
            .expect("XPinn construction with valid config should succeed");
        for i in 0..10 {
            let x = i as f32 / 10.0;
            let u = xp
                .forward(x)
                .expect("forward pass should succeed for valid input");
            assert!(u.is_finite(), "forward({x}) = {u}");
        }
    }

    #[test]
    fn forward_outside_domain_errors() {
        let mut rng = LcgRng::new(5);
        let xp = XPinn::new(XPinnConfig::default(), &mut rng)
            .expect("XPinn construction with valid config should succeed");
        assert!(xp.forward(5.0).is_err());
    }

    #[test]
    fn interface_loss_zero_when_consistent() {
        let mut rng = LcgRng::new(6);
        let cfg = XPinnConfig {
            n_subdomains: 3,
            ..Default::default()
        };
        let xp =
            XPinn::new(cfg, &mut rng).expect("XPinn construction with valid config should succeed");
        // 2 interfaces; both sides agree and residuals match → zero loss.
        let u_l = vec![0.3, -0.2];
        let u_r = vec![0.3, -0.2];
        let r_l = vec![1.0, 2.0];
        let r_r = vec![1.0, 2.0];
        let loss = xp
            .interface_loss(&u_l, &u_r, &r_l, &r_r)
            .expect("interface loss computation should succeed for valid input");
        assert!(loss.interface_term < 1e-9);
        assert!(loss.residual_term < 1e-9);
        assert!(loss.total < 1e-9);
    }

    #[test]
    fn interface_loss_positive_when_inconsistent() {
        let mut rng = LcgRng::new(7);
        let cfg = XPinnConfig {
            n_subdomains: 2,
            w_interface: 10.0,
            w_residual: 2.0,
            ..Default::default()
        };
        let xp =
            XPinn::new(cfg, &mut rng).expect("XPinn construction with valid config should succeed");
        let u_l = vec![1.0];
        let u_r = vec![0.0];
        let r_l = vec![3.0];
        let r_r = vec![1.0];
        let loss = xp
            .interface_loss(&u_l, &u_r, &r_l, &r_r)
            .expect("interface loss computation should succeed for valid input");
        // interface_term = ½(u_L − u_R)² = ½·1 = 0.5
        assert!(approx(loss.interface_term, 0.5, 1e-5));
        // residual_term = (3−1)² = 4
        assert!(approx(loss.residual_term, 4.0, 1e-5));
        // total = 10·0.5 + 2·4 = 13
        assert!(approx(loss.total, 13.0, 1e-4));
    }

    #[test]
    fn interface_loss_dim_mismatch_errors() {
        let mut rng = LcgRng::new(8);
        let cfg = XPinnConfig {
            n_subdomains: 3,
            ..Default::default()
        };
        let xp =
            XPinn::new(cfg, &mut rng).expect("XPinn construction with valid config should succeed");
        // needs length 2, give 1
        assert!(matches!(
            xp.interface_loss(&[0.0], &[0.0, 0.0], &[0.0, 0.0], &[0.0, 0.0]),
            Err(PinnError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn interface_loss_from_networks_finite() {
        let mut rng = LcgRng::new(9);
        let cfg = XPinnConfig {
            n_subdomains: 3,
            ..Default::default()
        };
        let xp =
            XPinn::new(cfg, &mut rng).expect("XPinn construction with valid config should succeed");
        // zero residual closure → residual_term must be 0, interface_term >= 0
        let loss = xp
            .interface_loss_from_networks(|_idx, _x| Ok(0.0_f32))
            .expect("interface loss from networks should succeed for valid input");
        assert!(loss.residual_term < 1e-9);
        assert!(loss.interface_term >= 0.0);
        assert!(loss.total.is_finite());
    }

    #[test]
    fn single_subdomain_has_no_interface() {
        let mut rng = LcgRng::new(10);
        let cfg = XPinnConfig {
            n_subdomains: 1,
            ..Default::default()
        };
        let xp =
            XPinn::new(cfg, &mut rng).expect("XPinn construction with valid config should succeed");
        assert_eq!(xp.interface_points().len(), 0);
        let loss = xp
            .interface_loss(&[], &[], &[], &[])
            .expect("interface loss computation should succeed for valid input");
        assert_eq!(loss.total, 0.0);
    }

    #[test]
    fn invalid_config_errors() {
        let mut rng = LcgRng::new(11);
        assert!(
            XPinn::new(
                XPinnConfig {
                    domain_lo: 1.0,
                    domain_hi: 0.0,
                    ..Default::default()
                },
                &mut rng
            )
            .is_err()
        );
        assert!(
            XPinn::new(
                XPinnConfig {
                    n_subdomains: 0,
                    ..Default::default()
                },
                &mut rng
            )
            .is_err()
        );
        assert!(
            XPinn::new(
                XPinnConfig {
                    w_interface: -1.0,
                    ..Default::default()
                },
                &mut rng
            )
            .is_err()
        );
    }

    #[test]
    fn nan_input_errors() {
        let mut rng = LcgRng::new(12);
        let cfg = XPinnConfig {
            n_subdomains: 2,
            ..Default::default()
        };
        let xp =
            XPinn::new(cfg, &mut rng).expect("XPinn construction with valid config should succeed");
        assert!(matches!(
            xp.interface_loss(&[f32::NAN], &[0.0], &[0.0], &[0.0]),
            Err(PinnError::NanEncountered { .. })
        ));
    }
}
