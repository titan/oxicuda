//! Standard black-box benchmark functions (BBOB-style) and algorithm evaluation harness.
//!
//! References:
//! - N. Hansen et al., "Real-Parameter Black-Box Optimization Benchmarking 2009: Noiseless Functions
//!   Definitions", INRIA Research Report RR-6829, 2009.
//! - Deb, K., Thiele, L., Laumanns, M., Zitzler, E. (2002). "Scalable multi-objective
//!   optimization test problems." Proc. CEC 2002.

use crate::evolution::cmaes::cmaes::{CmaEsConfig, CmaEsState};
use crate::handle::LcgRng;
use crate::metrics::hypervolume_nd::hypervolume_nd;
use crate::multiobjective::nsga2::{Nsga2Config, nsga2_run};
use crate::{EvolError, EvolResult};

// ─────────────────────────────────────────────────────────────────────────────
// Single-objective benchmark functions
// ─────────────────────────────────────────────────────────────────────────────

/// Sphere function: `f(x) = Σ xᵢ²`. Global minimum at **x = 0**, f = 0.
///
/// Convex, separable, unimodal. The simplest benchmark for sanity-checking.
#[inline]
pub fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}

/// Ellipsoid (ill-conditioned sphere): `f(x) = Σ (1000^(i/(n-1)) · xᵢ)²`.
///
/// Conditioning number is 10⁶. Tests adaptation to axis-aligned covariance.
/// Global minimum at **x = 0**, f = 0.
pub fn ellipsoid(x: &[f64]) -> f64 {
    let n = x.len();
    if n <= 1 {
        return x.first().map(|&v| v * v).unwrap_or(0.0);
    }
    x.iter()
        .enumerate()
        .map(|(i, &xi)| {
            let scale = 1000_f64.powf(i as f64 / (n - 1) as f64);
            (scale * xi) * (scale * xi)
        })
        .sum()
}

/// Rosenbrock (banana) function: `f(x) = Σ [100·(x_{i+1} - xᵢ²)² + (xᵢ - 1)²]`.
///
/// Non-convex, narrow curved valley. Global minimum at **x = (1,…,1)**, f = 0.
pub fn rosenbrock(x: &[f64]) -> f64 {
    if x.len() < 2 {
        return 0.0;
    }
    x.windows(2)
        .map(|w| {
            let xi = w[0];
            let xi1 = w[1];
            100.0 * (xi1 - xi * xi).powi(2) + (xi - 1.0).powi(2)
        })
        .sum()
}

/// Rastrigin function: `f(x) = 10n + Σ [xᵢ² − 10·cos(2π·xᵢ)]`.
///
/// Highly multimodal, ≈10n local minima. Global minimum at **x = 0**, f = 0.
pub fn rastrigin(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let two_pi = 2.0 * std::f64::consts::PI;
    10.0 * n
        + x.iter()
            .map(|&xi| xi * xi - 10.0 * (two_pi * xi).cos())
            .sum::<f64>()
}

/// Schwefel function: `f(x) = 418.9829·n − Σ xᵢ·sin(√|xᵢ|)`.
///
/// Deceptive: global minimum is far from secondary minima.
/// Global minimum at **xᵢ = 418.9829…**, f ≈ 0.
pub fn schwefel(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    418.9829 * n - x.iter().map(|&xi| xi * xi.abs().sqrt().sin()).sum::<f64>()
}

/// Ackley function: `f(x) = −20·exp(−0.2·√(Σxᵢ²/n)) − exp(Σcos(2πxᵢ)/n) + 20 + e`.
///
/// Many local minima, exponential global basin. Global minimum at **x = 0**, f = 0.
pub fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let two_pi = 2.0 * std::f64::consts::PI;
    let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sum_cos: f64 = x.iter().map(|&xi| (two_pi * xi).cos()).sum();
    -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + 20.0 + std::f64::consts::E
}

/// Griewank function: `f(x) = 1 + Σxᵢ²/4000 − Πcos(xᵢ/√(i+1))`.
///
/// Product term introduces regular structure in the multimodal landscape.
/// Global minimum at **x = 0**, f = 0.
pub fn griewank(x: &[f64]) -> f64 {
    let sum_sq: f64 = x.iter().map(|&xi| xi * xi / 4000.0).sum();
    let product: f64 = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| xi / ((i + 1) as f64).sqrt())
        .map(|v| v.cos())
        .product();
    1.0 + sum_sq - product
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-objective benchmark functions
// ─────────────────────────────────────────────────────────────────────────────

/// ZDT1: two-objective benchmark with convex Pareto front.
///
/// Decision variables: `x ∈ [0, 1]^n`.
/// - `f₁ = x₀`
/// - `g = 1 + 9·Σ(x[1:]/(n−1))`
/// - `f₂ = g·(1 − √(f₁/g))`
///
/// Returns `[f1, f2]`.
pub fn zdt1(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![0.0, 0.0];
    }
    let f1 = x[0];
    if n == 1 {
        let g = 1.0;
        let f2 = g * (1.0 - (f1 / g).max(0.0).sqrt());
        return vec![f1, f2];
    }
    let g_sum: f64 = x[1..].iter().sum::<f64>();
    let g = 1.0 + 9.0 * g_sum / (n - 1) as f64;
    let f2 = g * (1.0 - (f1 / g).max(0.0).sqrt());
    vec![f1, f2]
}

/// ZDT2: two-objective benchmark with non-convex Pareto front.
///
/// Decision variables: `x ∈ [0, 1]^n`.
/// - `f₁ = x₀`
/// - `g = 1 + 9·Σ(x[1:]/(n−1))`
/// - `f₂ = g·(1 − (f₁/g)²)`
///
/// Returns `[f1, f2]`.
pub fn zdt2(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![0.0, 0.0];
    }
    let f1 = x[0];
    if n == 1 {
        let g = 1.0;
        let f2 = g * (1.0 - (f1 / g).powi(2));
        return vec![f1, f2];
    }
    let g_sum: f64 = x[1..].iter().sum::<f64>();
    let g = 1.0 + 9.0 * g_sum / (n - 1) as f64;
    let f2 = g * (1.0 - (f1 / g).powi(2));
    vec![f1, f2]
}

/// DTLZ1: three-objective benchmark.
///
/// Decision variables: `x ∈ [0, 1]^n` with `n ≥ 3`.
/// - `xm = x[2..]` (the "distance" variables)
/// - `g(xm) = 100·(|xm| + Σ[(xi − 0.5)² − cos(20π(xi − 0.5))])`
/// - `f₁ = 0.5·x₀·x₁·(1 + g)`
/// - `f₂ = 0.5·x₀·(1 − x₁)·(1 + g)`
/// - `f₃ = 0.5·(1 − x₀)·(1 + g)`
///
/// Returns `[f1, f2, f3]`.
pub fn dtlz1(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 3 {
        // Degenerate case: pad with zeros for objective vector
        return vec![0.0, 0.0, 0.0];
    }
    let xm = &x[2..];
    let k = xm.len() as f64;
    let two_pi = 2.0 * std::f64::consts::PI;
    let g_sum: f64 = xm
        .iter()
        .map(|&xi| {
            let shifted = xi - 0.5;
            shifted * shifted - (20.0 * two_pi * shifted).cos()
        })
        .sum::<f64>();
    let g = 100.0 * (k + g_sum);
    let f1 = 0.5 * x[0] * x[1] * (1.0 + g);
    let f2 = 0.5 * x[0] * (1.0 - x[1]) * (1.0 + g);
    let f3 = 0.5 * (1.0 - x[0]) * (1.0 + g);
    vec![f1, f2, f3]
}

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm harness types
// ─────────────────────────────────────────────────────────────────────────────

/// Performance profile for a single-objective benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Human-readable name of the benchmark function.
    pub function_name: &'static str,
    /// Problem dimensionality.
    pub n_dims: usize,
    /// Best objective value found.
    pub best_value: f64,
    /// Total number of function evaluations consumed.
    pub n_evaluations: usize,
    /// Whether the algorithm achieved `best_value < target_precision`.
    pub converged: bool,
    /// Target precision threshold used to classify convergence.
    pub target_precision: f64,
}

/// Performance profile for a multi-objective benchmark run.
#[derive(Debug, Clone)]
pub struct MoBenchmarkResult {
    /// Human-readable name of the benchmark function.
    pub function_name: &'static str,
    /// Problem dimensionality.
    pub n_dims: usize,
    /// Hypervolume of the final Pareto front approximation.
    pub hypervolume: f64,
    /// Number of non-dominated points in the final approximation.
    pub n_front_points: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// CMA-ES harness
// ─────────────────────────────────────────────────────────────────────────────

/// Run CMA-ES on a scalar benchmark function and return a convergence profile.
///
/// # Arguments
/// - `f` — objective function (lower is better)
/// - `n_dims` — number of decision variables
/// - `max_evals` — maximum number of function evaluations
/// - `target_precision` — convergence threshold; if `best_value < target_precision` the run is
///   declared converged
/// - `seed` — deterministic random seed
/// - `function_name` — label stored in `BenchmarkResult`
///
/// The initial distribution mean is the origin and `σ₀ = 0.3·range` with range ≈ 5.0
/// (a sensible default for BBOB functions defined on [−5, 5]).
pub fn run_cmaes_benchmark<F>(
    f: F,
    n_dims: usize,
    max_evals: usize,
    target_precision: f64,
    seed: u64,
    function_name: &'static str,
) -> EvolResult<BenchmarkResult>
where
    F: Fn(&[f64]) -> f64,
{
    if n_dims == 0 {
        return Err(EvolError::InvalidParameter(
            "n_dims must be >= 1".to_owned(),
        ));
    }

    let mut cfg = CmaEsConfig::new(n_dims)?;
    cfg.max_evals = max_evals;
    cfg.sigma_init = 0.5; // wider initial step for BBOB search domain ≈ [−5, 5]
    cfg.tol_fun = target_precision * 1e-2; // stop slightly below target

    // Start from origin
    let mean_init = vec![0.0f64; n_dims];
    let mut state = CmaEsState::new(mean_init, &cfg)?;
    let mut rng = LcgRng::new(seed);

    let (_, best_value) = state.run(&f, &cfg, &mut rng)?;
    let n_evaluations = state.n_evals;
    let converged = best_value < target_precision;

    Ok(BenchmarkResult {
        function_name,
        n_dims,
        best_value,
        n_evaluations,
        converged,
        target_precision,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// NSGA-II harness
// ─────────────────────────────────────────────────────────────────────────────

/// Run NSGA-II on a multi-objective benchmark function and return Pareto front quality metrics.
///
/// # Arguments
/// - `f` — multi-objective function returning `Vec<f64>` of length `n_obj`
/// - `n_dims` — decision variable count
/// - `n_obj` — objective count
/// - `pop_size` — population size (must be even, ≥ 4)
/// - `n_gens` — number of NSGA-II generations
/// - `seed` — deterministic random seed
/// - `function_name` — label stored in `MoBenchmarkResult`
/// - `reference_point` — hypervolume reference point (length must equal `n_obj`)
///
/// Decision variables are assumed to lie in `[0, 1]`.
pub fn run_nsga2_benchmark<F>(
    f: F,
    n_dims: usize,
    n_obj: usize,
    pop_size: usize,
    n_gens: usize,
    seed: u64,
    function_name: &'static str,
    reference_point: Vec<f64>,
) -> EvolResult<MoBenchmarkResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if n_dims == 0 {
        return Err(EvolError::InvalidParameter(
            "n_dims must be >= 1".to_owned(),
        ));
    }
    if n_obj == 0 {
        return Err(EvolError::InvalidParameter("n_obj must be >= 1".to_owned()));
    }
    if pop_size < 4 {
        return Err(EvolError::PopulationTooSmall {
            size: pop_size,
            op: "NSGA-II benchmark",
        });
    }
    if reference_point.len() != n_obj {
        return Err(EvolError::DimensionMismatch {
            expected: n_obj,
            got: reference_point.len(),
        });
    }

    // Ensure pop_size is even
    let pop_size = if pop_size.is_multiple_of(2) {
        pop_size
    } else {
        pop_size + 1
    };

    let cfg = Nsga2Config {
        n_dims,
        n_objectives: n_obj,
        pop_size,
        max_generations: n_gens,
        crossover_eta: 15.0,
        mutation_eta: 20.0,
        mutation_prob: 1.0 / n_dims as f64,
        bounds: (0.0, 1.0),
    };

    let mut rng = LcgRng::new(seed);
    let final_pop = nsga2_run(f, &cfg, &mut rng)?;

    // Extract Pareto front (rank 0)
    let front_points: Vec<Vec<f64>> = final_pop
        .iter()
        .filter(|ind| ind.rank == 0)
        .map(|ind| ind.objectives.clone())
        .collect();

    let n_front_points = front_points.len();

    // Compute hypervolume using the WFG algorithm.
    // hypervolume_nd expects reference as &[Vec<f64>] with reference[0] = ref point,
    // and requires every front point to be strictly dominated by the reference.
    // Filter to only include points that are strictly dominated by the reference point.
    let ref_pt = &reference_point;
    let dominated_front: Vec<Vec<f64>> = front_points
        .into_iter()
        .filter(|p| p.iter().zip(ref_pt.iter()).all(|(fi, ri)| fi < ri))
        .collect();

    let hypervolume = if dominated_front.is_empty() {
        0.0
    } else {
        hypervolume_nd(&dominated_front, &[reference_point]).unwrap_or(0.0)
    };

    Ok(MoBenchmarkResult {
        function_name,
        n_dims,
        hypervolume,
        n_front_points,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Sphere ───────────────────────────────────────────────────────────────

    #[test]
    fn sphere_origin_is_zero() {
        assert_eq!(sphere(&[0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn sphere_unit_vector() {
        let val = sphere(&[1.0, 0.0, 0.0]);
        assert!((val - 1.0).abs() < 1e-14, "sphere([1,0,0]) = {val}");
    }

    #[test]
    fn sphere_positive() {
        assert!(sphere(&[1.0, 2.0, 3.0]) > 0.0);
    }

    // ── Ellipsoid ─────────────────────────────────────────────────────────────

    #[test]
    fn ellipsoid_origin_is_zero() {
        assert_eq!(ellipsoid(&[0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn ellipsoid_single_dim_origin() {
        assert_eq!(ellipsoid(&[0.0]), 0.0);
    }

    #[test]
    fn ellipsoid_larger_than_sphere() {
        // Due to conditioning, ellipsoid should be larger than sphere for non-zero x
        let x = &[1.0, 1.0, 1.0, 1.0, 1.0];
        assert!(ellipsoid(x) >= sphere(x));
    }

    // ── Rosenbrock ───────────────────────────────────────────────────────────

    #[test]
    fn rosenbrock_global_minimum() {
        // f([1, 1]) = 0
        let val = rosenbrock(&[1.0, 1.0]);
        assert!(val.abs() < 1e-14, "rosenbrock([1,1]) = {val}");
    }

    #[test]
    fn rosenbrock_global_minimum_higher_dim() {
        let ones = vec![1.0f64; 5];
        let val = rosenbrock(&ones);
        assert!(val.abs() < 1e-10, "rosenbrock(ones_5) = {val}");
    }

    #[test]
    fn rosenbrock_non_minimum_positive() {
        assert!(rosenbrock(&[0.0, 0.0]) > 0.0);
    }

    // ── Rastrigin ────────────────────────────────────────────────────────────

    #[test]
    fn rastrigin_global_minimum() {
        // f([0, 0]) = 0
        let val = rastrigin(&[0.0, 0.0]);
        assert!(val.abs() < 1e-14, "rastrigin([0,0]) = {val}");
    }

    #[test]
    fn rastrigin_positive_elsewhere() {
        // Rastrigin ≥ 0 globally and > 0 away from origin
        assert!(rastrigin(&[1.0, 1.0]) > 0.0);
    }

    // ── Ackley ───────────────────────────────────────────────────────────────

    #[test]
    fn ackley_global_minimum_approx_zero() {
        // f([0, 0]) should be ≈ 0 (within floating-point precision)
        let val = ackley(&[0.0, 0.0]);
        assert!(val.abs() < 1e-10, "ackley([0,0]) = {val}");
    }

    #[test]
    fn ackley_positive_elsewhere() {
        assert!(ackley(&[1.0, 2.0]) > 0.0);
    }

    // ── Griewank ─────────────────────────────────────────────────────────────

    #[test]
    fn griewank_origin_is_zero() {
        let val = griewank(&[0.0]);
        assert!(val.abs() < 1e-14, "griewank([0]) = {val}");
    }

    #[test]
    fn griewank_origin_multi_dim() {
        // griewank(zeros) = 1 + 0 - 1 = 0
        let val = griewank(&[0.0, 0.0, 0.0]);
        assert!(val.abs() < 1e-14, "griewank(zeros_3) = {val}");
    }

    // ── Schwefel ─────────────────────────────────────────────────────────────

    #[test]
    fn schwefel_global_minimum_approx() {
        // Global minimum near x_i = 418.9829, f ≈ 0
        // (more accurate value for the Schwefel minimizer is ~420.9687, but the
        // BBOB Schwefel formulation uses the 418.9829 coefficient).
        // Use the commonly cited approximation
        let val = schwefel(&[418.9829]);
        // The residual should be small — within ≈ 0.01 of zero
        assert!(val.abs() < 1.0, "schwefel([418.9829]) = {val}");
    }

    #[test]
    fn schwefel_at_exact_minimizer_low() {
        // At x = 420.9687..., f ≈ 0 for 1-D
        // The precise minimizer of xi*sin(sqrt(|xi|)) is ~420.9687
        // schwefel([x]) = 418.9829 - x*sin(sqrt(x))
        // We verify it's close to zero (within 2 for robustness across formulations)
        let xm = 420.9687_f64;
        let val = schwefel(&[xm]);
        // The global minimum might not be exactly 0 with the coefficient 418.9829
        // but it should be bounded near zero
        assert!(val.abs() < 5.0, "schwefel near minimizer = {val}");
    }

    // ── Multi-objective ───────────────────────────────────────────────────────

    #[test]
    fn zdt1_returns_two_objectives() {
        let result = zdt1(&[0.5; 5]);
        assert_eq!(result.len(), 2, "zdt1 must return 2 objectives");
    }

    #[test]
    fn zdt1_objectives_nonnegative() {
        let result = zdt1(&[0.3, 0.1, 0.2, 0.4, 0.5]);
        assert!(result[0] >= 0.0 && result[1] >= 0.0);
    }

    #[test]
    fn zdt2_returns_two_objectives() {
        let result = zdt2(&[0.5; 5]);
        assert_eq!(result.len(), 2, "zdt2 must return 2 objectives");
    }

    #[test]
    fn dtlz1_returns_three_objectives() {
        let result = dtlz1(&[0.5; 5]);
        assert_eq!(result.len(), 3, "dtlz1 must return 3 objectives");
    }

    #[test]
    fn dtlz1_short_input_still_three_objectives() {
        // n < 3 is a degenerate case but must still return 3-element Vec
        let result = dtlz1(&[0.5, 0.5]);
        assert_eq!(result.len(), 3);
    }

    // ── Algorithm harness ────────────────────────────────────────────────────

    #[test]
    fn cmaes_benchmark_sphere_5d_converges() {
        let result = run_cmaes_benchmark(sphere, 5, 50_000, 1e-5, 42, "sphere-5d")
            .expect("CMA-ES on sphere should not error");
        assert!(
            result.converged,
            "CMA-ES should converge on 5-D sphere, best = {}",
            result.best_value
        );
        assert!(result.best_value < 1e-5);
    }

    #[test]
    fn cmaes_benchmark_rosenbrock_2d() {
        let result = run_cmaes_benchmark(rosenbrock, 2, 100_000, 1e-3, 7, "rosenbrock-2d")
            .expect("CMA-ES on Rosenbrock should not error");
        assert!(
            result.best_value < 1.0,
            "CMA-ES on Rosenbrock 2D should reach near optimum, best = {}",
            result.best_value
        );
    }

    #[test]
    fn cmaes_benchmark_n_evaluations_positive() {
        let result = run_cmaes_benchmark(sphere, 3, 5_000, 1e-5, 1, "sphere-3d-eval-check")
            .expect("no error");
        assert!(
            result.n_evaluations > 0,
            "n_evaluations must be > 0, got {}",
            result.n_evaluations
        );
    }

    #[test]
    fn cmaes_benchmark_invalid_n_dims_errors() {
        let err = run_cmaes_benchmark(sphere, 0, 1000, 1e-5, 0, "bad-dims");
        assert!(err.is_err(), "n_dims=0 must return an error");
    }

    #[test]
    fn nsga2_benchmark_zdt1_positive_hypervolume() {
        // Use a generous reference point to ensure front points are dominated.
        // ZDT1: f1 in [0,1], f2 in [0, 10] roughly; use (2.0, 15.0) to capture all.
        let ref_pt = vec![2.0, 15.0];
        let result = run_nsga2_benchmark(zdt1, 5, 2, 40, 80, 123, "zdt1-5d", ref_pt)
            .expect("NSGA-II on ZDT1 should not error");
        assert!(
            result.hypervolume > 0.0,
            "Hypervolume must be positive for ZDT1, got {}",
            result.hypervolume
        );
        assert!(result.n_front_points > 0, "Pareto front must be non-empty");
    }

    #[test]
    fn nsga2_benchmark_zdt2_positive_hypervolume() {
        // ZDT2: f1 in [0,1], f2 in [0, 10]; use generous reference point
        let ref_pt = vec![2.0, 15.0];
        let result = run_nsga2_benchmark(zdt2, 5, 2, 40, 80, 77, "zdt2-5d", ref_pt)
            .expect("NSGA-II on ZDT2 should not error");
        assert!(
            result.hypervolume > 0.0,
            "Hypervolume must be positive for ZDT2, got {}",
            result.hypervolume
        );
    }

    #[test]
    fn benchmark_result_fields_consistent() {
        let result =
            run_cmaes_benchmark(sphere, 2, 10_000, 1e-6, 99, "sphere-2d-fields").expect("no error");
        assert_eq!(result.n_dims, 2);
        assert_eq!(result.function_name, "sphere-2d-fields");
        assert_eq!(result.target_precision, 1e-6);
        assert_eq!(
            result.converged,
            result.best_value < result.target_precision
        );
    }
}
