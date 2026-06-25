//! Discrete (categorical) conditional-independence test for the PC algorithm.
//!
//! The production PC skeleton/orientation in [`super::pc`] ships with a
//! linear-Gaussian (Fisher-Z partial-correlation) conditional-independence
//! test, which assumes the data are jointly normal. That test is meaningless on
//! *categorical* data (a Bayesian network over discrete variables). This module
//! supplies the standard non-parametric alternative used by `bnlearn`,
//! `pcalg::disCItest` and Tetrad's `g2`: a contingency-table test of
//!
//! ```text
//!     X ⫫ Y | Z          (X, Y, Z all categorical)
//! ```
//!
//! computed by stratifying the sample on the joint configuration of the
//! conditioning set `Z` and summing a per-stratum 2-way independence statistic
//! over the `r_x × r_y` contingency tables. Two statistics are provided, both
//! asymptotically `χ²(df)` under the independence null:
//!
//! * [`DiscreteStatistic::ChiSquare`] — Pearson's `Σ (O − E)² / E`.
//! * [`DiscreteStatistic::GTest`] — the likelihood-ratio / G statistic
//!   `2 Σ O · ln(O / E)`, which equals `2 N · Î(X; Y | Z)` (twice the sample
//!   conditional mutual information in nats); hence the "CMI" test.
//!
//! Degrees of freedom follow the adaptive rule of `bnlearn`/Tetrad: within each
//! stratum only the *non-empty* rows and columns count,
//! `df_z = (r_x⁺ − 1)(r_y⁺ − 1)`, and a stratum with fewer than two non-empty
//! rows *or* columns contributes nothing (it cannot exhibit association). The
//! `p`-value is the upper tail `P(χ²(df) > stat)` evaluated with the
//! regularized upper incomplete gamma function implemented below
//! (Numerical Recipes §6.2, series + continued fraction, Lanczos `lnΓ`).

use crate::discovery::pc::ConditionalIndependenceTest;
use crate::error::{CausalError, CausalResult};
use std::collections::HashMap;

/// Which discrete independence statistic to accumulate over the strata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscreteStatistic {
    /// Pearson chi-square: `Σ (O − E)² / E`.
    ChiSquare,
    /// Likelihood-ratio "G" / conditional-mutual-information: `2 Σ O · ln(O/E)`.
    GTest,
}

/// Discrete conditional-independence test over a categorical design matrix.
///
/// The data are row-major category codes: `data[row * d + var]` is the integer
/// level (`0 ≤ code < n_levels[var]`) of variable `var` in sample `row`. The
/// test implements [`ConditionalIndependenceTest`], so it can be dropped into
/// [`super::pc::PcAlgorithm::run_with_test`] in place of the Fisher-Z test.
#[derive(Debug, Clone)]
pub struct DiscreteCiTest {
    data: Vec<usize>,
    n: usize,
    d: usize,
    n_levels: Vec<usize>,
    alpha: f64,
    statistic: DiscreteStatistic,
}

impl DiscreteCiTest {
    /// Build a Pearson chi-square test from categorical data.
    ///
    /// `data` is `n × d` row-major category codes, `n_levels[j]` is the number
    /// of categories (cardinality) of variable `j`. Every code must satisfy
    /// `0 ≤ data[i*d + j] < n_levels[j]`.
    pub fn new(
        data: &[usize],
        n: usize,
        d: usize,
        n_levels: &[usize],
        alpha: f32,
    ) -> CausalResult<Self> {
        Self::with_statistic(data, n, d, n_levels, alpha, DiscreteStatistic::ChiSquare)
    }

    /// Build the test with an explicit [`DiscreteStatistic`] choice.
    pub fn with_statistic(
        data: &[usize],
        n: usize,
        d: usize,
        n_levels: &[usize],
        alpha: f32,
        statistic: DiscreteStatistic,
    ) -> CausalResult<Self> {
        if data.is_empty() || n < 4 || d < 2 {
            return Err(CausalError::EmptyInput);
        }
        if data.len() != n * d {
            return Err(CausalError::DimensionMismatch {
                expected: n * d,
                got: data.len(),
            });
        }
        if n_levels.len() != d {
            return Err(CausalError::DimensionMismatch {
                expected: d,
                got: n_levels.len(),
            });
        }
        if n_levels.contains(&0) {
            return Err(CausalError::InvalidParameter {
                reason: "every variable must have at least one category".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(CausalError::InvalidParameter {
                reason: "significance level alpha must lie in [0, 1]".to_string(),
            });
        }
        // Bounds-validate every code so contingency indexing can never overflow.
        for row in 0..n {
            for var in 0..d {
                if data[row * d + var] >= n_levels[var] {
                    return Err(CausalError::InvalidParameter {
                        reason: format!(
                            "category code {} of variable {var} exceeds its cardinality {}",
                            data[row * d + var],
                            n_levels[var]
                        ),
                    });
                }
            }
        }
        Ok(Self {
            data: data.to_vec(),
            n,
            d,
            n_levels: n_levels.to_vec(),
            alpha: alpha as f64,
            statistic,
        })
    }

    #[inline]
    fn code(&self, row: usize, var: usize) -> usize {
        self.data[row * self.d + var]
    }

    /// Mixed-radix key for the joint configuration of the conditioning set `z`
    /// in `row`. Two rows share a key iff they agree on every variable in `z`,
    /// so the keys partition the sample into the conditioning strata. With the
    /// PC conditioning-set cap (`|z| ≤ 3`) and small cardinalities the radix
    /// stays tiny, so the encoding is exact.
    fn stratum_key(&self, row: usize, z: &[usize]) -> u64 {
        let mut key = 0_u64;
        let mut radix = 1_u64;
        for &zk in z {
            let code = self.code(row, zk) as u64;
            key = key.wrapping_add(code.wrapping_mul(radix));
            radix = radix.saturating_mul(self.n_levels[zk] as u64);
        }
        key
    }

    /// Compute `(statistic, df, p_value)` for the hypothesis `X ⫫ Y | Z`.
    ///
    /// `df == 0` means the data carry no testable association (e.g. a constant
    /// variable, or every stratum degenerate); the `p`-value is then `1.0`.
    pub fn test(&self, x: usize, y: usize, z: &[usize]) -> (f64, usize, f64) {
        let r_x = self.n_levels[x];
        let r_y = self.n_levels[y];
        // Build one r_x × r_y contingency table per conditioning stratum.
        let mut strata: HashMap<u64, Vec<u32>> = HashMap::new();
        for row in 0..self.n {
            let key = self.stratum_key(row, z);
            let xi = self.code(row, x);
            let yi = self.code(row, y);
            let table = strata.entry(key).or_insert_with(|| vec![0_u32; r_x * r_y]);
            table[xi * r_y + yi] += 1;
        }
        let mut stat = 0.0_f64;
        let mut df = 0_usize;
        for table in strata.values() {
            accumulate_table(table, r_x, r_y, self.statistic, &mut stat, &mut df);
        }
        let p = if df == 0 {
            1.0
        } else {
            chi_square_sf(stat, df)
        };
        (stat, df, p)
    }
}

impl ConditionalIndependenceTest for DiscreteCiTest {
    fn num_vars(&self) -> usize {
        self.d
    }

    fn dependent(&self, x: usize, y: usize, z: &[usize]) -> bool {
        let (_stat, df, p) = self.test(x, y, z);
        if df == 0 {
            return false;
        }
        p < self.alpha
    }
}

/// Add a single stratum's contribution to the running statistic and degrees of
/// freedom. Uses the adaptive non-empty-rows/columns rule, summing only over
/// cells with positive row *and* column marginals (so `E > 0` always).
fn accumulate_table(
    counts: &[u32],
    r_x: usize,
    r_y: usize,
    kind: DiscreteStatistic,
    stat: &mut f64,
    df: &mut usize,
) {
    let mut row_sum = vec![0_u64; r_x];
    let mut col_sum = vec![0_u64; r_y];
    let mut total = 0_u64;
    for i in 0..r_x {
        for j in 0..r_y {
            let c = counts[i * r_y + j] as u64;
            row_sum[i] += c;
            col_sum[j] += c;
            total += c;
        }
    }
    if total == 0 {
        return;
    }
    let nz_rows = row_sum.iter().filter(|&&v| v > 0).count();
    let nz_cols = col_sum.iter().filter(|&&v| v > 0).count();
    if nz_rows < 2 || nz_cols < 2 {
        return; // degenerate stratum: no association possible.
    }
    *df += (nz_rows - 1) * (nz_cols - 1);
    let total_f = total as f64;
    for i in 0..r_x {
        if row_sum[i] == 0 {
            continue;
        }
        for j in 0..r_y {
            if col_sum[j] == 0 {
                continue;
            }
            let observed = f64::from(counts[i * r_y + j]);
            let expected = row_sum[i] as f64 * col_sum[j] as f64 / total_f;
            match kind {
                DiscreteStatistic::ChiSquare => {
                    let diff = observed - expected;
                    *stat += diff * diff / expected;
                }
                DiscreteStatistic::GTest => {
                    if observed > 0.0 {
                        *stat += 2.0 * observed * (observed / expected).ln();
                    }
                }
            }
        }
    }
}

// --- Regularized upper incomplete gamma → chi-square survival function -------
//
// `P(χ²(k) > x) = Q(k/2, x/2)` where `Q(a, x) = Γ(a, x) / Γ(a)` is the
// regularized *upper* incomplete gamma function. Following Numerical Recipes
// §6.2 we evaluate the *lower* regularized `P(a, x)` by its power series for
// `x < a + 1` and the *upper* `Q(a, x)` by Lentz's continued fraction
// otherwise, switching at the crossover for numerical stability. `lnΓ` is the
// Lanczos approximation (g = 5, 6 coefficients), accurate to ~1e-10 for x > 0.

const GAMMA_ITMAX: usize = 300;
const GAMMA_EPS: f64 = 1e-13;
const GAMMA_FPMIN: f64 = 1e-300;

/// Lanczos approximation of `ln Γ(x)` for `x > 0`.
fn ln_gamma(x: f64) -> f64 {
    const COEF: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.001_208_650_973_866_179,
        -0.000_005_395_239_384_953,
    ];
    let mut y = x;
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut ser = 1.000_000_000_190_015_f64;
    for &c in &COEF {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.506_628_274_631_000_5_f64 * ser / x).ln()
}

/// Lower regularized incomplete gamma `P(a, x)` via its power series.
fn gamma_p_series(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let gln = ln_gamma(a);
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..GAMMA_ITMAX {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * GAMMA_EPS {
            break;
        }
    }
    sum * (-x + a * x.ln() - gln).exp()
}

/// Upper regularized incomplete gamma `Q(a, x)` via Lentz's continued fraction.
fn gamma_q_cf(a: f64, x: f64) -> f64 {
    let gln = ln_gamma(a);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / GAMMA_FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=GAMMA_ITMAX {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < GAMMA_FPMIN {
            d = GAMMA_FPMIN;
        }
        c = b + an / c;
        if c.abs() < GAMMA_FPMIN {
            c = GAMMA_FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < GAMMA_EPS {
            break;
        }
    }
    (-x + a * x.ln() - gln).exp() * h
}

/// Regularized upper incomplete gamma `Q(a, x) = Γ(a, x) / Γ(a)`.
fn reg_gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if a <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        (1.0 - gamma_p_series(a, x)).clamp(0.0, 1.0)
    } else {
        gamma_q_cf(a, x).clamp(0.0, 1.0)
    }
}

/// Chi-square survival function `P(χ²(df) > stat)`.
pub fn chi_square_sf(stat: f64, df: usize) -> f64 {
    if df == 0 {
        return 1.0;
    }
    if !stat.is_finite() {
        return if stat > 0.0 { 0.0 } else { 1.0 };
    }
    if stat <= 0.0 {
        return 1.0;
    }
    reg_gamma_q(df as f64 / 2.0, stat / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::pc::PcAlgorithm;
    use crate::handle::LcgRng;

    /// Single-table statistic + df, for known-answer verification.
    fn table_statistic(
        counts: &[u32],
        r_x: usize,
        r_y: usize,
        kind: DiscreteStatistic,
    ) -> (f64, usize) {
        let mut stat = 0.0;
        let mut df = 0;
        accumulate_table(counts, r_x, r_y, kind, &mut stat, &mut df);
        (stat, df)
    }

    // ----- numerical correctness of the incomplete-gamma / chi-square SF -----

    #[test]
    fn chi_square_sf_matches_known_percentiles() {
        // Upper 5% critical values of χ²(df): qchisq(0.95, df).
        assert!((chi_square_sf(3.841_459, 1) - 0.05).abs() < 1e-3);
        assert!((chi_square_sf(5.991_465, 2) - 0.05).abs() < 1e-3);
        assert!((chi_square_sf(7.814_728, 3) - 0.05).abs() < 1e-3);
        assert!((chi_square_sf(9.487_729, 4) - 0.05).abs() < 1e-3);
        // Upper 1% critical value of χ²(1): qchisq(0.99, 1) = 6.6349.
        assert!((chi_square_sf(6.634_897, 1) - 0.01).abs() < 1e-3);
        // Median of χ²(2) is 2·ln 2 ≈ 1.386294 → survival 0.5.
        assert!((chi_square_sf(1.386_294, 2) - 0.5).abs() < 1e-3);
        // Tails.
        assert!((chi_square_sf(0.0, 3) - 1.0).abs() < 1e-12);
        assert!(chi_square_sf(200.0, 3) < 1e-30);
        assert_eq!(chi_square_sf(5.0, 0), 1.0);
    }

    #[test]
    fn pearson_statistic_known_table() {
        // 2×2 table [[10,20],[30,40]]. Marginals 30/70 × 40/60, N=100.
        // E = [[12,18],[28,42]]; χ² = 4/12 + 4/18 + 4/28 + 4/42 = 0.793651.
        let (stat, df) = table_statistic(&[10, 20, 30, 40], 2, 2, DiscreteStatistic::ChiSquare);
        assert_eq!(df, 1);
        assert!((stat - 0.793_651).abs() < 1e-4, "chi2 = {stat}");
    }

    #[test]
    fn g_statistic_known_table() {
        // Same table; G = 2·Σ O ln(O/E) = 0.804380.
        let (stat, df) = table_statistic(&[10, 20, 30, 40], 2, 2, DiscreteStatistic::GTest);
        assert_eq!(df, 1);
        assert!((stat - 0.804_380).abs() < 1e-4, "G = {stat}");
    }

    #[test]
    fn degenerate_stratum_has_zero_df() {
        // A table with an all-zero second row cannot show association.
        let (stat, df) = table_statistic(&[10, 20, 0, 0], 2, 2, DiscreteStatistic::ChiSquare);
        assert_eq!(df, 0);
        assert_eq!(stat, 0.0);
    }

    #[test]
    fn three_by_three_independence_df() {
        // Full 3×3 table → df = (3-1)(3-1) = 4.
        let counts = [5_u32, 7, 9, 6, 4, 8, 3, 11, 2];
        let (stat, df) = table_statistic(&counts, 3, 3, DiscreteStatistic::ChiSquare);
        assert_eq!(df, 4);
        assert!(stat.is_finite() && stat >= 0.0);
    }

    // ----- behaviour of the test on generated categorical data ---------------

    /// Independent binary X, Y (no relationship at all).
    fn independent_data(n: usize, seed: u64) -> Vec<usize> {
        let mut rng = LcgRng::new(seed);
        let mut data = vec![0_usize; n * 2];
        for i in 0..n {
            data[i * 2] = rng.next_usize(2);
            data[i * 2 + 1] = rng.next_usize(2);
        }
        data
    }

    /// Strongly dependent binary X, Y (Y tracks X with 90% fidelity).
    fn dependent_data(n: usize, seed: u64) -> Vec<usize> {
        let mut rng = LcgRng::new(seed);
        let mut data = vec![0_usize; n * 2];
        for i in 0..n {
            let x = rng.next_usize(2);
            let y = if rng.next_f32() < 0.9 { x } else { 1 - x };
            data[i * 2] = x;
            data[i * 2 + 1] = y;
        }
        data
    }

    #[test]
    fn declares_independence_for_independent_vars() {
        let n = 4000;
        let data = independent_data(n, 0xC0FFEE);
        let test =
            DiscreteCiTest::new(&data, n, 2, &[2, 2], 0.05).expect("discrete CI test should build");
        let (stat, df, p) = test.test(0, 1, &[]);
        assert_eq!(df, 1);
        assert!(
            p > 0.05,
            "independent vars wrongly flagged dependent (p = {p}, stat = {stat})"
        );
        assert!(!test.dependent(0, 1, &[]));
    }

    #[test]
    fn declares_dependence_for_dependent_vars() {
        let n = 4000;
        let data = dependent_data(n, 0xBADF00D);
        let test =
            DiscreteCiTest::new(&data, n, 2, &[2, 2], 0.05).expect("discrete CI test should build");
        let (_stat, df, p) = test.test(0, 1, &[]);
        assert_eq!(df, 1);
        assert!(p < 1e-6, "dependent vars not detected (p = {p})");
        assert!(test.dependent(0, 1, &[]));
    }

    /// Collider X → Z ← Y with independent binary parents and a *faithful*
    /// additive noisy CPD for the collider: `P(Z = 1 | X, Y)` rises with
    /// `X + Y` (0.1, 0.5, 0.9 for sums 0, 1, 2). Node layout 0 = X, 1 = Z
    /// (collider), 2 = Y. This makes each parent marginally **dependent** with
    /// the collider (`P(Z=1|X=0)=0.3`, `P(Z=1|X=1)=0.7`) yet the two parents
    /// marginally **independent** of each other, and conditionally dependent
    /// given Z (the classic "explaining away"). A pure XOR collider is *not*
    /// usable here: parity makes every pair marginally independent, violating
    /// faithfulness, so PC would (correctly) erase the whole skeleton.
    fn collider_data(n: usize, seed: u64) -> Vec<usize> {
        let mut rng = LcgRng::new(seed);
        let mut data = vec![0_usize; n * 3];
        for i in 0..n {
            let x = rng.next_usize(2);
            let y = rng.next_usize(2);
            let p_z = match x + y {
                0 => 0.1_f32,
                1 => 0.5_f32,
                _ => 0.9_f32,
            };
            let z = usize::from(rng.next_f32() < p_z);
            data[i * 3] = x;
            data[i * 3 + 1] = z;
            data[i * 3 + 2] = y;
        }
        data
    }

    #[test]
    fn collider_marginal_independent_but_conditionally_dependent() {
        let n = 4000;
        let data = collider_data(n, 0x5EED);
        let test = DiscreteCiTest::new(&data, n, 3, &[2, 2, 2], 0.05)
            .expect("discrete CI test should build");
        // X ⫫ Y marginally.
        let (_s_m, _df_m, p_marg) = test.test(0, 2, &[]);
        assert!(
            p_marg > 0.05,
            "collider parents X,Y wrongly dependent (p = {p_marg})"
        );
        assert!(!test.dependent(0, 2, &[]));
        // X ̸⫫ Y | Z (explaining away).
        let (_s_c, _df_c, p_cond) = test.test(0, 2, &[1]);
        assert!(p_cond < 1e-6, "explaining-away not detected (p = {p_cond})");
        assert!(test.dependent(0, 2, &[1]));
        // Each parent is dependent with the collider.
        assert!(test.dependent(0, 1, &[]));
        assert!(test.dependent(2, 1, &[]));
        // The G-test must reach the same qualitative verdicts.
        let g =
            DiscreteCiTest::with_statistic(&data, n, 3, &[2, 2, 2], 0.05, DiscreteStatistic::GTest)
                .expect("G-test should build");
        assert!(!g.dependent(0, 2, &[]));
        assert!(g.dependent(0, 2, &[1]));
    }

    // ----- end-to-end PC recovery on synthetic discrete networks -------------

    #[test]
    fn pc_recovers_collider_skeleton_and_v_structure() {
        let n = 4000;
        let data = collider_data(n, 0x1234_ABCD);
        // Stricter level controls false edges; the XOR signal is very strong.
        let pc = PcAlgorithm::run_discrete(&data, n, 3, &[2, 2, 2], 0.01)
            .expect("PC on discrete collider should succeed");
        // Skeleton must be exactly {X–Z, Y–Z} = {(0,1), (1,2)} (no X–Y edge).
        let mut skel = pc.skeleton.clone();
        skel.sort_unstable();
        assert_eq!(
            skel,
            vec![(0, 1), (1, 2)],
            "wrong skeleton: {:?}",
            pc.skeleton
        );
        // The unshielded triple 0 – 1 – 2 must be oriented as the collider
        // 0 → 1 ← 2, i.e. both directed edges point INTO the collider node 1.
        let directed: Vec<(usize, usize)> = pc
            .cpdag
            .iter()
            .filter(|&&(_, _, dir)| dir)
            .map(|&(a, b, _)| (a, b))
            .collect();
        assert!(
            directed.contains(&(0, 1)) && directed.contains(&(2, 1)),
            "collider not oriented X→Z←Y, cpdag = {:?}",
            pc.cpdag
        );
    }

    /// Chain X → Y → Z (0 → 1 → 2): Y tracks X, Z tracks Y, both 90% fidelity.
    fn chain_data(n: usize, seed: u64) -> Vec<usize> {
        let mut rng = LcgRng::new(seed);
        let mut data = vec![0_usize; n * 3];
        for i in 0..n {
            let x = rng.next_usize(2);
            let y = if rng.next_f32() < 0.9 { x } else { 1 - x };
            let z = if rng.next_f32() < 0.9 { y } else { 1 - y };
            data[i * 3] = x;
            data[i * 3 + 1] = y;
            data[i * 3 + 2] = z;
        }
        data
    }

    #[test]
    fn pc_recovers_chain_skeleton() {
        let n = 4000;
        let data = chain_data(n, 0x0FED_CBA9);
        let pc = PcAlgorithm::run_discrete(&data, n, 3, &[2, 2, 2], 0.01)
            .expect("PC on discrete chain should succeed");
        // Skeleton must be exactly {X–Y, Y–Z}; X–Z is removed by X ⫫ Z | Y.
        let mut skel = pc.skeleton.clone();
        skel.sort_unstable();
        assert_eq!(
            skel,
            vec![(0, 1), (1, 2)],
            "wrong skeleton: {:?}",
            pc.skeleton
        );
        // A chain is Markov-equivalent to its reverses, so the unshielded
        // triple 0 – 1 – 2 is NOT a collider and must stay unoriented.
        let directed = pc.cpdag.iter().filter(|&&(_, _, dir)| dir).count();
        assert_eq!(directed, 0, "chain edges wrongly oriented: {:?}", pc.cpdag);
    }

    #[test]
    fn rejects_out_of_range_codes() {
        let data = vec![0_usize, 1, 2, 0, 1, 0, 0, 1]; // a "2" with cardinality 2
        let err = DiscreteCiTest::new(&data, 4, 2, &[2, 2], 0.05);
        assert!(err.is_err());
    }
}
