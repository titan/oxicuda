//! Top-level [`Mps`] container and constructors.
//!
//! An [`Mps`] is an ordered sequence of [`MpsTensor`]s whose neighbouring bond
//! dimensions must match (`tensors[s].d_r == tensors[s+1].d_l`).
//!
//! The boundary bonds are conventionally `d_l = 1` for the first site and
//! `d_r = 1` for the last site, so contracting the entire chain yields a single
//! amplitude per physical configuration.

use crate::handle::LcgRng;
use crate::mps::tensor::MpsTensor;
use crate::{TnError, TnResult};

/// Matrix Product State for `L` sites with arbitrary bond dimensions.
#[derive(Debug, Clone)]
pub struct Mps {
    /// One rank-3 tensor per site.
    pub site_tensors: Vec<MpsTensor>,
}

impl Mps {
    /// Construct an MPS from site tensors. Validates that neighbouring bonds match
    /// and that the boundary bonds are 1.
    pub fn from_tensors(site_tensors: Vec<MpsTensor>) -> TnResult<Self> {
        if site_tensors.is_empty() {
            return Err(TnError::EmptyInput);
        }
        if site_tensors[0].d_l != 1 {
            return Err(TnError::InvalidBondDimension(site_tensors[0].d_l));
        }
        let last_dr = site_tensors.last().ok_or(TnError::EmptyInput)?.d_r;
        if last_dr != 1 {
            return Err(TnError::InvalidBondDimension(last_dr));
        }
        for i in 0..(site_tensors.len() - 1) {
            if site_tensors[i].d_r != site_tensors[i + 1].d_l {
                return Err(TnError::DimensionMismatch {
                    a: site_tensors[i].d_r,
                    b: site_tensors[i + 1].d_l,
                });
            }
        }
        Ok(Self { site_tensors })
    }

    /// Build an MPS from a product state `|σ_0 σ_1 ... σ_{L-1}>` with local physical
    /// dimension `d`.
    ///
    /// Each `local_states[s]` is a length-`d` vector giving the amplitudes of site `s`.
    pub fn from_product_state(local_states: &[Vec<f64>]) -> TnResult<Self> {
        if local_states.is_empty() {
            return Err(TnError::EmptyInput);
        }
        let d = local_states[0].len();
        if d == 0 {
            return Err(TnError::EmptyInput);
        }
        let mut tensors = Vec::with_capacity(local_states.len());
        for state in local_states {
            if state.len() != d {
                return Err(TnError::ShapeMismatch {
                    expected: vec![d],
                    got: vec![state.len()],
                });
            }
            // Each site tensor has shape (1, d, 1) with values state[p].
            tensors.push(MpsTensor::new(1, d, 1, state.clone())?);
        }
        Self::from_tensors(tensors)
    }

    /// Generate a random MPS with bond dimension `chi` and physical dimension `d`
    /// for `n_sites` sites. Tensors are drawn i.i.d. standard normal.
    pub fn random_mps(n_sites: usize, d: usize, chi: usize, rng: &mut LcgRng) -> TnResult<Self> {
        if n_sites == 0 || d == 0 || chi == 0 {
            return Err(TnError::EmptyInput);
        }
        let mut tensors = Vec::with_capacity(n_sites);
        for s in 0..n_sites {
            let d_l = if s == 0 { 1 } else { chi };
            let d_r = if s + 1 == n_sites { 1 } else { chi };
            let n_el = d_l * d * d_r;
            let data: Vec<f64> = (0..n_el).map(|_| rng.next_normal()).collect();
            tensors.push(MpsTensor::new(d_l, d, d_r, data)?);
        }
        Self::from_tensors(tensors)
    }

    /// Number of sites.
    pub fn n_sites(&self) -> usize {
        self.site_tensors.len()
    }

    /// Physical dimension at site `s`.
    pub fn physical_dim(&self, s: usize) -> TnResult<usize> {
        self.site_tensors
            .get(s)
            .map(|t| t.d_p)
            .ok_or(TnError::IndexOutOfBounds {
                index: s,
                len: self.site_tensors.len(),
            })
    }

    /// Bond dimension at the bond *between* sites `s` and `s+1`.
    pub fn bond_dim(&self, s: usize) -> TnResult<usize> {
        if s + 1 >= self.site_tensors.len() {
            return Err(TnError::IndexOutOfBounds {
                index: s,
                len: self.site_tensors.len(),
            });
        }
        Ok(self.site_tensors[s].d_r)
    }

    /// Compute `<ψ|ψ>` by contracting the chain with itself.
    ///
    /// Time complexity is `O(L * D^3 * d)` (negligible for unit-test sizes).
    pub fn norm_squared(&self) -> TnResult<f64> {
        // E will be a `(D_l, D_l)` matrix that accumulates left environments
        // across sites: E_{a, a'} = sum over already-contracted sites.
        let mut env = vec![1.0_f64]; // 1×1 environment initially
        let mut env_rows = 1usize;
        for site in &self.site_tensors {
            let dl = site.d_l;
            let d = site.d_p;
            let dr = site.d_r;
            if dl != env_rows {
                return Err(TnError::DimensionMismatch { a: dl, b: env_rows });
            }
            // new_env[b, b'] = sum_{a, a', p} env[a, a'] * M[a, p, b] * M[a', p, b']
            let mut new_env = vec![0.0_f64; dr * dr];
            for b in 0..dr {
                for bp in 0..dr {
                    let mut acc = 0.0;
                    for a in 0..dl {
                        for ap in 0..dl {
                            let eaa = env[a * dl + ap];
                            for p in 0..d {
                                let m1 = site.data[(a * d + p) * dr + b];
                                let m2 = site.data[(ap * d + p) * dr + bp];
                                acc += eaa * m1 * m2;
                            }
                        }
                    }
                    new_env[b * dr + bp] = acc;
                }
            }
            env = new_env;
            env_rows = dr;
        }
        // env is 1×1 at the end
        Ok(env[0])
    }

    /// Compute `<ψ|ψ>^{1/2}`.
    pub fn norm(&self) -> TnResult<f64> {
        Ok(self.norm_squared()?.sqrt())
    }

    /// Compute the expectation value of a per-site operator `op[s]` (each `op[s]` of shape
    /// `(d, d)` in row-major, applied to site `s`). All ops are inserted simultaneously.
    pub fn expectation_local(&self, op: &[Vec<f64>]) -> TnResult<f64> {
        if op.len() != self.n_sites() {
            return Err(TnError::ShapeMismatch {
                expected: vec![self.n_sites()],
                got: vec![op.len()],
            });
        }
        let mut env = vec![1.0_f64];
        let mut env_rows = 1usize;
        for (s, site) in self.site_tensors.iter().enumerate() {
            let dl = site.d_l;
            let d = site.d_p;
            let dr = site.d_r;
            let opmat = &op[s];
            if opmat.len() != d * d {
                return Err(TnError::ShapeMismatch {
                    expected: vec![d, d],
                    got: vec![opmat.len()],
                });
            }
            if dl != env_rows {
                return Err(TnError::DimensionMismatch { a: dl, b: env_rows });
            }
            let mut new_env = vec![0.0_f64; dr * dr];
            for b in 0..dr {
                for bp in 0..dr {
                    let mut acc = 0.0;
                    for a in 0..dl {
                        for ap in 0..dl {
                            let eaa = env[a * dl + ap];
                            for p in 0..d {
                                for pp in 0..d {
                                    let m1 = site.data[(a * d + p) * dr + b];
                                    let m2 = site.data[(ap * d + pp) * dr + bp];
                                    let opv = opmat[p * d + pp];
                                    acc += eaa * m1 * opv * m2;
                                }
                            }
                        }
                    }
                    new_env[b * dr + bp] = acc;
                }
            }
            env = new_env;
            env_rows = dr;
        }
        Ok(env[0])
    }

    /// Scale every site tensor at site 0 by `factor` (the easiest way to renormalise).
    pub fn rescale(&mut self, factor: f64) -> TnResult<()> {
        if self.site_tensors.is_empty() {
            return Err(TnError::EmptyInput);
        }
        for x in &mut self.site_tensors[0].data {
            *x *= factor;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_state_norm() {
        // |0> on each of 3 sites: amplitudes [1, 0]
        let local = vec![vec![1.0, 0.0]; 3];
        let mps = Mps::from_product_state(&local).expect("ok");
        let n2 = mps.norm_squared().expect("ok");
        assert!((n2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn random_mps_constructs() {
        let mut rng = LcgRng::new(42);
        let mps = Mps::random_mps(4, 2, 3, &mut rng).expect("ok");
        assert_eq!(mps.n_sites(), 4);
        assert_eq!(mps.physical_dim(0).expect("ok"), 2);
        // Inner bonds match chi=3
        assert_eq!(mps.bond_dim(0).expect("ok"), 3);
    }

    #[test]
    fn product_state_local_expectation() {
        // Apply identity operators: <psi|I^L|psi> = norm^2 = 1
        let local = vec![vec![1.0, 0.0]; 3];
        let mps = Mps::from_product_state(&local).expect("ok");
        let id: Vec<Vec<f64>> = (0..3).map(|_| vec![1.0, 0.0, 0.0, 1.0]).collect();
        let ev = mps.expectation_local(&id).expect("ok");
        assert!((ev - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mismatched_bond_rejected() {
        let t1 = MpsTensor::zeros(1, 2, 3).expect("ok");
        let t2 = MpsTensor::zeros(4, 2, 1).expect("ok");
        assert!(Mps::from_tensors(vec![t1, t2]).is_err());
    }

    #[test]
    fn rescale_changes_norm() {
        let local = vec![vec![1.0, 0.0]; 2];
        let mut mps = Mps::from_product_state(&local).expect("ok");
        mps.rescale(2.0).expect("ok");
        let n2 = mps.norm_squared().expect("ok");
        assert!((n2 - 4.0).abs() < 1e-12);
    }
}
