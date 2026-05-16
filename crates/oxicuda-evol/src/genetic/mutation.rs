//! Mutation operators: Gaussian, polynomial, swap.

use crate::handle::LcgRng;

/// Gaussian mutation: each gene is perturbed by `N(0, sigma)` with probability `p_mut`.
/// The resulting gene is clamped to `[bounds.0, bounds.1]`.
pub fn gaussian_mutate(
    genome: &mut [f64],
    sigma: f64,
    p_mut: f64,
    bounds: (f64, f64),
    rng: &mut LcgRng,
) {
    let (lb, ub) = bounds;
    for gene in genome.iter_mut() {
        if rng.next_f64() < p_mut {
            let delta = rng.next_normal() * sigma;
            *gene = (*gene + delta).clamp(lb, ub);
        }
    }
}

/// Polynomial mutation (Deb & Goyal, 1995): used in NSGA-II.
///
/// The mutation polynomial distribution is defined by the distribution index `eta_m`.
/// Larger `eta_m` → mutations closer to the parent. Typical value: 20.0.
pub fn polynomial_mutate(
    genome: &mut [f64],
    eta_m: f64,
    p_mut: f64,
    bounds: (f64, f64),
    rng: &mut LcgRng,
) {
    let (lb, ub) = bounds;
    let range = ub - lb;
    for gene in genome.iter_mut() {
        if rng.next_f64() < p_mut {
            let delta1 = (*gene - lb) / range;
            let delta2 = (ub - *gene) / range;
            let u = rng.next_f64();
            let delta_q = if u <= 0.5 {
                let val = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta1).powf(eta_m + 1.0);
                val.powf(1.0 / (eta_m + 1.0)) - 1.0
            } else {
                let val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta2).powf(eta_m + 1.0);
                1.0 - val.powf(1.0 / (eta_m + 1.0))
            };
            *gene = (*gene + delta_q * range).clamp(lb, ub);
        }
    }
}

/// Swap mutation: select two random distinct positions and exchange their gene values.
/// Does nothing if the genome has fewer than 2 genes.
pub fn swap_mutate(genome: &mut [f64], rng: &mut LcgRng) {
    let n = genome.len();
    if n < 2 {
        return;
    }
    let i = rng.next_usize(n);
    let mut j = rng.next_usize(n - 1);
    if j >= i {
        j += 1;
    }
    genome.swap(i, j);
}
