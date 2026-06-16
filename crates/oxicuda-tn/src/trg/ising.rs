//! Construction of the 2D classical Ising-model partition-function tensor and
//! Onsager's exact free energy, used as physical test oracles for [`super`].
//!
//! The square-lattice Ising partition function `Z = Σ_{s} exp(βJ Σ_{⟨ij⟩} s_i
//! s_j)` can be written as a uniform tensor network. Placing a rank-4 tensor on
//! each *bond-crossing* (dual-lattice) site, the local Boltzmann weight on a
//! bond between aligned/anti-aligned spins is the `2×2` matrix
//! `W = [[e^{βJ}, e^{−βJ}], [e^{−βJ}, e^{βJ}]]`. Factoring `W = Q Qᵀ` (matrix
//! square root) and absorbing one `Q` into each adjacent site yields the
//! symmetric rank-4 site tensor
//! `T[u,l,d,r] = Σ_s Q[s,u] Q[s,l] Q[s,d] Q[s,r]`.

use super::LatticeTensor;
use crate::TnResult;

/// Build the bond dimension-2 Ising site tensor at inverse temperature `beta`
/// with coupling `j` (ferromagnetic for `j > 0`).
///
/// Uses the symmetric `Q = W^{1/2}` factorisation so the resulting tensor is
/// invariant under leg permutations (isotropic), the form TRG expects.
///
/// # Errors
/// Propagated from [`LatticeTensor::new`].
pub fn ising_tensor(beta: f64, j: f64) -> TnResult<LatticeTensor> {
    let a = (beta * j).exp(); // aligned weight
    let b = (-beta * j).exp(); // anti-aligned weight
    // Eigendecomposition of the symmetric 2×2 W = [[a,b],[b,a]]:
    //   eigenvalues a+b (vector (1,1)/√2) and a−b (vector (1,−1)/√2).
    let lam_plus = (a + b).max(0.0);
    let lam_minus = (a - b).max(0.0);
    let sp = lam_plus.sqrt();
    let sm = lam_minus.sqrt();
    // Q = V · diag(√λ) · Vᵀ (symmetric square root). With V columns the
    // eigenvectors above, Q has entries:
    //   Q[0,0] = Q[1,1] = (sp + sm)/2 ; Q[0,1] = Q[1,0] = (sp − sm)/2
    let q = [
        (sp + sm) * 0.5,
        (sp - sm) * 0.5,
        (sp - sm) * 0.5,
        (sp + sm) * 0.5,
    ];
    // T[u,l,d,r] = Σ_{s∈{0,1}} Q[s,u] Q[s,l] Q[s,d] Q[s,r].
    let mut data = vec![0.0; 16];
    for u in 0..2 {
        for l in 0..2 {
            for d in 0..2 {
                for r in 0..2 {
                    let mut acc = 0.0;
                    for s in 0..2 {
                        acc += q[s * 2 + u] * q[s * 2 + l] * q[s * 2 + d] * q[s * 2 + r];
                    }
                    data[((u * 2 + l) * 2 + d) * 2 + r] = acc;
                }
            }
        }
    }
    LatticeTensor::new(2, 2, 2, 2, data)
}

/// Onsager's exact free energy `ln Z / N` (i.e. `−βf`) of the isotropic 2D Ising
/// model on the infinite square lattice at inverse temperature `beta`, coupling
/// `j`.
///
/// Starting from the exact double-integral form (with `K = βJ`)
/// `ln Z/N = ln 2 + (1/8π²) ∫₀^{2π}∫₀^{2π} ln[cosh²(2K) − sinh(2K)(cos θ₁ + cos θ₂)] dθ₁ dθ₂`
/// and performing the inner angular integral analytically via the identity
/// `(1/2π) ∫₀^{2π} ln(A − B cos φ) dφ = ln[(A + √(A² − B²)) / 2]` (for `A > |B|`)
/// collapses it to the single quadrature
/// `ln Z/N = ½ ln 2 + (1/2π) ∫₀^π ln[ A(θ) + √(A(θ)² − sinh²(2K)) ] dθ`,
/// where `A(θ) = cosh²(2K) − sinh(2K) cos θ`. The integrand is singular only at
/// the critical coupling `sinh(2K) = 1` (the band touches zero), where the model
/// is gapless. This is the reference value TRG converges to.
#[must_use]
pub fn onsager_log_z_per_site(beta: f64, j: f64) -> f64 {
    let k = beta * j;
    let sinh2 = (2.0 * k).sinh();
    let cosh2 = (2.0 * k).cosh();
    if sinh2.abs() < 1e-300 {
        // β → 0: Z/N → 2 (free spins). ln 2.
        return std::f64::consts::LN_2;
    }
    let cosh2_sq = cosh2 * cosh2;
    let sinh2_sq = sinh2 * sinh2;
    // Composite trapezoidal rule for (1/2π) ∫₀^π ln[ A + √(A² − sinh²2K) ] dθ
    // with A(θ) = cosh²(2K) − sinh(2K) cos θ.
    let n = 20_000usize;
    let mut integral = 0.0f64;
    for i in 0..=n {
        let theta = std::f64::consts::PI * i as f64 / n as f64;
        let a = cosh2_sq - sinh2 * theta.cos();
        let rad = (a * a - sinh2_sq).max(0.0);
        let val = (a + rad.sqrt()).max(1e-300).ln();
        let w = if i == 0 || i == n { 0.5 } else { 1.0 };
        integral += w * val;
    }
    integral *= std::f64::consts::PI / n as f64;
    0.5 * std::f64::consts::LN_2 + integral / (2.0 * std::f64::consts::PI)
}

#[cfg(test)]
mod tests {
    use super::super::trg_partition_log;
    use super::*;

    #[test]
    fn ising_tensor_is_symmetric_under_leg_swap() {
        let t = ising_tensor(0.4, 1.0).expect("ising_tensor should succeed");
        // T[u,l,d,r] == T[l,u,d,r] (isotropic from symmetric Q).
        for u in 0..2 {
            for l in 0..2 {
                for d in 0..2 {
                    for r in 0..2 {
                        let a = t.data[((u * 2 + l) * 2 + d) * 2 + r];
                        let b = t.data[((l * 2 + u) * 2 + d) * 2 + r];
                        assert!((a - b).abs() < 1e-12, "asymmetry at {u}{l}{d}{r}");
                    }
                }
            }
        }
    }

    #[test]
    fn ising_tensor_all_positive() {
        let t = ising_tensor(0.5, 1.0).expect("ising_tensor should succeed");
        assert!(t.data.iter().all(|&x| x >= -1e-12));
    }

    #[test]
    fn onsager_high_temperature_limit() {
        // β → 0 ⇒ ln Z/N → ln 2 (free spins, 2 states each).
        let v = onsager_log_z_per_site(1e-6, 1.0);
        assert!((v - std::f64::consts::LN_2).abs() < 1e-3, "v={v}");
    }

    #[test]
    fn onsager_is_finite_across_temperatures() {
        for &beta in &[0.1, 0.3, 0.440_686_8, 0.6, 1.0] {
            let v = onsager_log_z_per_site(beta, 1.0);
            assert!(v.is_finite() && v > 0.0, "beta={beta} v={v}");
        }
    }

    #[test]
    fn onsager_matches_double_integral_reference() {
        // Ground-truth ln Z/N from the exact 2D double integral
        //   ln 2 + (1/8π²) ∫∫ ln[cosh²2K − sinh2K(cos θ₁ + cos θ₂)] dθ₁ dθ₂,
        // computed independently. The collapsed single-integral form must agree
        // at every temperature, not only at criticality (sinh 2K = 1).
        let refs = [
            (0.2_f64, 0.734_531_f64),
            (0.3, 0.790_559),
            (0.440_686_8, 0.929_696),
            (0.5, 1.025_793),
            (0.6, 1.210_132),
            (0.8, 1.601_811),
            (1.0, 2.000_348),
        ];
        for &(beta, expected) in &refs {
            let v = onsager_log_z_per_site(beta, 1.0);
            assert!(
                (v - expected).abs() < 1e-4,
                "beta={beta}: got {v}, expected {expected}, diff {}",
                (v - expected).abs()
            );
        }
    }

    #[test]
    fn trg_matches_onsager_high_temperature() {
        // At high T (small β) TRG with modest χ should track Onsager closely.
        // β = 0.2 is deep in the disordered phase (β_c ≈ 0.4407): the correlation
        // length is short, so a small bond dimension already converges.
        let beta = 0.2;
        let t = ising_tensor(beta, 1.0).expect("ising_tensor should succeed");
        let trg = trg_partition_log(&t, 10, 1e-12, 10).expect("trg_partition_log should succeed");
        let exact = onsager_log_z_per_site(beta, 1.0);
        assert!(
            (trg - exact).abs() < 5e-3,
            "TRG={trg}, Onsager={exact}, |diff|={}",
            (trg - exact).abs()
        );
    }

    #[test]
    fn trg_matches_onsager_low_temperature() {
        // At low T (large β) the ordered phase: TRG should still track Onsager.
        // β = 0.6 sits in the gapped ordered phase (β_c ≈ 0.4407), so — as at high
        // T — a modest bond dimension converges; the costly regime is criticality.
        let beta = 0.6;
        let t = ising_tensor(beta, 1.0).expect("ising_tensor should succeed");
        let trg = trg_partition_log(&t, 10, 1e-12, 12).expect("trg_partition_log should succeed");
        let exact = onsager_log_z_per_site(beta, 1.0);
        assert!(
            (trg - exact).abs() < 5e-3,
            "TRG={trg}, Onsager={exact}, |diff|={}",
            (trg - exact).abs()
        );
    }

    #[test]
    fn trg_partition_log_positive_for_ising() {
        let t = ising_tensor(0.44, 1.0).expect("ising_tensor should succeed");
        let log = trg_partition_log(&t, 10, 1e-12, 8).expect("trg_partition_log should succeed");
        assert!(log > 0.0 && log.is_finite(), "log={log}");
    }
}
