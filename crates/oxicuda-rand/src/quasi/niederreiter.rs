//! Niederreiter base-2 low-discrepancy quasi-random sequence generator.
//!
//! The Niederreiter sequence (Niederreiter, 1988) is a *digital
//! `(t, m, s)`-net / `(t, s)`-sequence* in base 2.  Like the Sobol sequence
//! it is a low-discrepancy sequence whose points fill the unit hypercube far
//! more evenly than independent pseudo-random draws, accelerating the
//! convergence of quasi-Monte-Carlo integration.
//!
//! ## Construction
//!
//! For every dimension `j` (1-indexed) a distinct monic *irreducible*
//! polynomial `p_j(x)` over `GF(2)` of degree `e_j` is chosen.  The
//! generating matrix `C^(j)` is an `nbits x nbits` binary matrix whose
//! entries are read from the formal Laurent-series expansion of rational
//! functions `x^u / p_j(x)^(Q+1)` over `GF(2)` (Niederreiter's construction
//! based on formal Laurent series of rationals `1 / p_j(x)`).  Concretely the
//! running power `b(x) = p_j(x)^Q` is maintained and, each time the digit
//! index crosses a multiple of `e_j`, the next block of Laurent coefficients
//! `v(0), v(1), ...` is produced by the linear recurrence implied by `b(x)`;
//! the matrix bit `c^(j)_{r,k}` is simply `v(r + u)` for the appropriate
//! offset `u`.  Each matrix row is then packed MSB-first into a single
//! 32-bit column word `cj[j][r]`.
//!
//! For index `n` the `k`-th coordinate is the XOR-accumulation of the
//! generating-matrix columns selected by the bits of the *Gray code*
//! `g(n) = n ^ (n >> 1)`, interpreted as a binary fraction in `[0, 1)`.
//! Successive Gray codes differ in exactly one bit, so advancing the
//! sequence costs a single XOR per dimension (the Gray-code optimisation).
//!
//! This is a faithful, self-contained reimplementation of the base-2 path of
//! Bratley, Fox & Niederreiter's ACM TOMS Algorithm 738 (`CALCC2` / `CALCV2`
//! / `GOLO2`).  It is **not** a wrapper around Sobol and uses **no**
//! random numbers -- the entire sequence is deterministic.
//!
//! ## Example
//!
//! ```rust
//! use oxicuda_rand::quasi::Niederreiter;
//!
//! # fn main() -> oxicuda_rand::RandResult<()> {
//! let mut seq = Niederreiter::new(3)?;
//! let p0 = seq.next_point();   // the origin: [0, 0, 0]
//! assert!(p0.iter().all(|&x| x == 0.0));
//! let p1 = seq.next_point();   // all coordinates in [0, 1)
//! assert!(p1.iter().all(|&x| (0.0..1.0).contains(&x)));
//! # Ok(())
//! # }
//! ```

use crate::error::{RandError, RandResult};

// ---------------------------------------------------------------------------
// Constants (base-2 BFN Algorithm 738 parameters)
// ---------------------------------------------------------------------------

/// Number of fixed-point bits used for each coordinate (matches the classic
/// base-2 Niederreiter reference; coordinates are integers scaled by 2^-31).
const NBITS: usize = 31;

/// Highest degree among the tabulated irreducible polynomials.
const MAX_E: usize = 6;

/// Length of the Laurent-coefficient working vector `v` (`NBITS + MAX_E`).
const MAXV: usize = NBITS + MAX_E;

/// Reciprocal scale `2^-NBITS` converting the integer coordinate to `[0, 1)`.
const RECIP: f64 = 1.0 / ((1u64 << NBITS) as f64);

/// Maximum supported dimension (size of the irreducible-polynomial table).
pub const MAX_NIEDERREITER_DIMENSION: usize = 20;

// ---------------------------------------------------------------------------
// Irreducible polynomials over GF(2)
// ---------------------------------------------------------------------------
//
// Each entry is `[degree, c_0, c_1, ..., c_degree]` -- the coefficients of a
// distinct monic irreducible polynomial over GF(2), in ascending order of
// power, with the leading 1 included.  Trailing slots are padded with 0 so
// every row has the same length (`MAX_E + 2`).  These are exactly the 20
// polynomials of Bratley-Fox-Niederreiter (degrees 1 through 6).

/// Number of stored coefficient slots per polynomial row
/// (`1` degree field `+` up to `MAX_E + 1` coefficients).
const POLY_ROW: usize = MAX_E + 2;

/// Table of 20 distinct irreducible polynomials over GF(2).
static IRRED: [[u8; POLY_ROW]; MAX_NIEDERREITER_DIMENSION] = [
    // degree 1: x + 1            -> 1 + x
    [1, 1, 1, 0, 0, 0, 0, 0],
    // degree 1: x                -> x        (irreducible, the "free" factor)
    [1, 0, 1, 0, 0, 0, 0, 0],
    // degree 2: x^2 + x + 1
    [2, 1, 1, 1, 0, 0, 0, 0],
    // degree 3: x^3 + x + 1
    [3, 1, 1, 0, 1, 0, 0, 0],
    // degree 3: x^3 + x^2 + 1
    [3, 1, 0, 1, 1, 0, 0, 0],
    // degree 4: x^4 + x + 1
    [4, 1, 1, 0, 0, 1, 0, 0],
    // degree 4: x^4 + x^3 + 1
    [4, 1, 0, 0, 1, 1, 0, 0],
    // degree 4: x^4 + x^3 + x^2 + x + 1
    [4, 1, 1, 1, 1, 1, 0, 0],
    // degree 5: x^5 + x^2 + 1
    [5, 1, 0, 1, 0, 0, 1, 0],
    // degree 5: x^5 + x^3 + 1
    [5, 1, 0, 0, 1, 0, 1, 0],
    // degree 5: x^5 + x^3 + x^2 + x + 1
    [5, 1, 1, 1, 1, 0, 1, 0],
    // degree 5: x^5 + x^4 + x^2 + x + 1
    [5, 1, 1, 1, 0, 1, 1, 0],
    // degree 5: x^5 + x^4 + x^3 + x + 1
    [5, 1, 1, 0, 1, 1, 1, 0],
    // degree 5: x^5 + x^4 + x^3 + x^2 + 1
    [5, 1, 0, 1, 1, 1, 1, 0],
    // degree 6: x^6 + x + 1
    [6, 1, 1, 0, 0, 0, 0, 1],
    // degree 6: x^6 + x^5 + 1
    [6, 1, 0, 0, 0, 0, 1, 1],
    // degree 6: x^6 + x^5 + x^2 + x + 1
    [6, 1, 1, 1, 0, 0, 1, 1],
    // degree 6: x^6 + x^5 + x^3 + x^2 + 1
    [6, 1, 0, 1, 1, 0, 1, 1],
    // degree 6: x^6 + x^5 + x^4 + x + 1
    [6, 1, 1, 0, 0, 1, 1, 1],
    // degree 6: x^6 + x^5 + x^4 + x^2 + 1
    [6, 1, 0, 1, 0, 1, 1, 1],
];

// ---------------------------------------------------------------------------
// Polynomial arithmetic over GF(2) (coefficient vectors, ascending powers)
// ---------------------------------------------------------------------------

/// A polynomial over GF(2) stored as ascending-power coefficient bits with an
/// explicit degree.  `degree == NONE_DEGREE` marks the zero polynomial.
#[derive(Clone)]
struct GfPoly {
    /// Degree of the polynomial, or [`GfPoly::NONE_DEGREE`] for the zero
    /// polynomial.
    degree: isize,
    /// Coefficients `c[k]` for power `x^k` (only `0..=degree` are meaningful).
    coeff: Vec<u8>,
}

impl GfPoly {
    /// Sentinel degree representing the zero polynomial.
    const NONE_DEGREE: isize = -1;

    /// Builds the constant polynomial `1`.
    fn one(capacity: usize) -> Self {
        let mut coeff = vec![0u8; capacity.max(1)];
        coeff[0] = 1;
        Self { degree: 0, coeff }
    }

    /// Builds a polynomial from a `[degree, c_0, ..]` table row.
    fn from_row(row: &[u8; POLY_ROW]) -> Self {
        let degree = row[0] as isize;
        let mut coeff = vec![0u8; (degree as usize) + 1];
        for (k, slot) in coeff.iter_mut().enumerate() {
            *slot = row[1 + k] & 1;
        }
        Self { degree, coeff }
    }

    /// Multiplies `self` by `other` in place, growing the buffer as needed.
    ///
    /// All coefficient products and sums are reduced mod 2 (`AND` / `XOR`).
    fn mul_assign(&mut self, other: &GfPoly) {
        if self.degree == Self::NONE_DEGREE || other.degree == Self::NONE_DEGREE {
            self.degree = Self::NONE_DEGREE;
            return;
        }
        let new_degree = self.degree + other.degree;
        let mut result = vec![0u8; (new_degree as usize) + 1];
        for i in 0..=(self.degree as usize) {
            if self.coeff[i] == 0 {
                continue;
            }
            for j in 0..=(other.degree as usize) {
                result[i + j] ^= self.coeff[i] & other.coeff[j];
            }
        }
        self.degree = new_degree;
        self.coeff = result;
    }
}

// ---------------------------------------------------------------------------
// CALCV2 -- next block of Laurent-series coefficients
// ---------------------------------------------------------------------------

/// Computes the next block of the Laurent-series coefficient vector `v` for a
/// dimension, advancing the running power `b(x) <- b(x) * px(x)`.
///
/// This is the base-2 specialisation of BFN's `CALCV2`.  Over `GF(2)`,
/// addition and subtraction are XOR and multiplication is AND; the
/// "arbitrary" and "non-zero" field elements are both `1`.  On entry `b`
/// holds `px^(Q-1)`; on exit it holds `px^Q` and `v[0..MAXV]` carries the
/// coefficients needed for the next `e` columns of the generating matrix.
fn calc_v2(px: &GfPoly, b: &mut GfPoly, v: &mut [u8; MAXV]) {
    // h <- b (the polynomial *before* multiplying by px), bigm = deg(h).
    let h = b.clone();
    let bigm = h.degree;

    // b <- b * px ; m = deg(b).
    b.mul_assign(px);
    let m = b.degree;

    // Kj choice (base-2 reference uses kj = bigm).
    let kj = bigm;

    // Initialise v[0..=kj]: zeros then a leading 1.
    for slot in v.iter_mut().take(kj as usize) {
        *slot = 0;
    }
    v[kj as usize] = 1;

    if kj < bigm {
        // term = -h(kj) = h(kj)  (mod 2)
        let mut term = h.coeff[kj as usize] & 1;
        let mut r = kj + 1;
        while r < bigm {
            // arbitrary element -> 1
            v[r as usize] = 1;
            // term <- term - h(r) * v(r)   (XOR of AND, mod 2)
            term ^= h.coeff[r as usize] & v[r as usize];
            r += 1;
        }
        // v(bigm) = nonzer + term = 1 ^ term
        v[bigm as usize] = 1 ^ term;
        // remaining slots up to m-1 are arbitrary -> 1
        let mut r = bigm + 1;
        while r < m {
            v[r as usize] = 1;
            r += 1;
        }
    } else {
        // kj == bigm : slots (kj+1 .. m-1) are arbitrary -> 1
        let mut r = kj + 1;
        while r < m {
            v[r as usize] = 1;
            r += 1;
        }
    }

    // Linear recurrence (mod 2) generated by b(x):
    //   v(r+m) = - sum_{i=0}^{m-1} b(i) * v(r+i)
    // which over GF(2) is the XOR of the selected earlier coefficients.
    let m_usize = m as usize;
    let mut r = 0usize;
    while r + m_usize < MAXV {
        let mut term = 0u8;
        for i in 0..m_usize {
            term ^= b.coeff[i] & v[r + i];
        }
        v[r + m_usize] = term;
        r += 1;
    }
}

// ---------------------------------------------------------------------------
// CALCC2 -- generating-matrix columns for every dimension
// ---------------------------------------------------------------------------

/// Computes the packed generating-matrix columns `cj[dim][r]` for the first
/// `dimension` dimensions.
///
/// `cj[i][r]` is the `r`-th column of dimension `i`'s generating matrix,
/// packed MSB-first into a 32-bit word.  This is the base-2 `CALCC2`.
fn calc_c2(dimension: usize) -> Vec<[u32; NBITS]> {
    let mut cj = vec![[0u32; NBITS]; dimension];

    for (i, cj_dim) in cj.iter_mut().enumerate() {
        let px = GfPoly::from_row(&IRRED[i]);
        let e = px.degree;

        // Running power b(x), starting at the constant polynomial 1.
        let mut b = GfPoly::one(MAXV + 1);
        let mut v = [0u8; MAXV];

        // ci[j][r] : bit of row j (digit index, 1..=NBITS) in column r.
        let mut ci = [[0u8; NBITS]; NBITS];

        let mut u: isize = 0;
        for ci_row in ci.iter_mut() {
            if u == 0 {
                calc_v2(&px, &mut b, &mut v);
            }
            for (r, slot) in ci_row.iter_mut().enumerate() {
                *slot = v[r + u as usize];
            }
            u += 1;
            if u == e {
                u = 0;
            }
        }

        // Pack: cj[r] gets bit ci[j][r] with j=1 the most significant bit.
        for (r, col) in cj_dim.iter_mut().enumerate() {
            let mut term: u32 = 0;
            for ci_row in ci.iter() {
                term = (term << 1) | u32::from(ci_row[r]);
            }
            *col = term;
        }
    }

    cj
}

// ---------------------------------------------------------------------------
// Niederreiter sequence generator
// ---------------------------------------------------------------------------

/// Base-2 Niederreiter low-discrepancy quasi-random sequence generator.
///
/// Produces deterministic multi-dimensional points in the half-open unit
/// hypercube `[0, 1)^dim`.  The first point is the origin by convention.
///
/// Up to [`MAX_NIEDERREITER_DIMENSION`] dimensions are supported (limited by
/// the built-in table of irreducible polynomials over `GF(2)`).
///
/// # Example
///
/// ```rust
/// use oxicuda_rand::quasi::Niederreiter;
///
/// # fn main() -> oxicuda_rand::RandResult<()> {
/// let mut seq = Niederreiter::new(2)?;
/// let pts = seq.points(16);
/// assert_eq!(pts.len(), 16);
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Niederreiter {
    /// Number of dimensions.
    dimension: usize,
    /// Packed generating-matrix columns, one `[u32; NBITS]` per dimension.
    cj: Vec<[u32; NBITS]>,
    /// Running XOR-accumulated integer coordinate per dimension.
    nextq: Vec<u32>,
    /// Index of the *next* point to emit (`0` => origin).
    seed: u32,
}

impl Niederreiter {
    /// Creates a new base-2 Niederreiter generator for `dimension` dimensions.
    ///
    /// The generating matrices are constructed eagerly from the built-in
    /// irreducible-polynomial table.
    ///
    /// # Errors
    ///
    /// Returns [`RandError::InvalidParameter`] if `dimension` is `0` or
    /// exceeds [`MAX_NIEDERREITER_DIMENSION`].
    pub fn new(dimension: usize) -> RandResult<Self> {
        if dimension == 0 {
            return Err(RandError::InvalidParameter(
                "Niederreiter dimension must be >= 1".to_string(),
            ));
        }
        if dimension > MAX_NIEDERREITER_DIMENSION {
            return Err(RandError::InvalidParameter(format!(
                "Niederreiter dimension must be 1..={MAX_NIEDERREITER_DIMENSION}, got {dimension}"
            )));
        }

        let cj = calc_c2(dimension);

        Ok(Self {
            dimension,
            cj,
            nextq: vec![0u32; dimension],
            seed: 0,
        })
    }

    /// Returns the number of dimensions.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the index of the next point that will be produced.
    pub fn position(&self) -> u32 {
        self.seed
    }

    /// Generates the next quasi-random point (Gray-code incremental update).
    ///
    /// The returned vector has `dimension` coordinates, each lying in
    /// `[0, 1)`.  The very first call (index `0`) returns the origin.
    pub fn next_point(&mut self) -> Vec<f64> {
        // Emit the coordinate corresponding to the current accumulator.
        let mut point = Vec::with_capacity(self.dimension);
        for &q in &self.nextq {
            point.push(f64::from(q) * RECIP);
        }

        // Advance: flip the single generating-matrix column at the position
        // of the rightmost zero bit of `seed` (Gray-code transition).
        let r = (!self.seed).trailing_zeros() as usize;
        if r < NBITS {
            for (q, col) in self.nextq.iter_mut().zip(self.cj.iter()) {
                *q ^= col[r];
            }
        }
        self.seed = self.seed.wrapping_add(1);

        point
    }

    /// Skips the next `n` points, advancing the sequence without producing
    /// output.
    ///
    /// After `skip(n)` the generator is positioned exactly as if
    /// [`Niederreiter::next_point`] had been called `n` times.
    pub fn skip(&mut self, n: u32) {
        if n == 0 {
            return;
        }
        // Jump directly using the Gray code of the target index: the
        // accumulator for index `m` is the XOR of all columns whose bit is
        // set in `gray(m) = m ^ (m >> 1)`.
        let target = self.seed.wrapping_add(n);
        self.set_position(target);
    }

    /// Repositions the generator so the next point produced is index
    /// `target`, rebuilding the XOR accumulator from the Gray code.
    fn set_position(&mut self, target: u32) {
        for q in self.nextq.iter_mut() {
            *q = 0;
        }
        let mut gray = target ^ (target >> 1);
        let mut r = 0usize;
        while gray != 0 {
            if gray & 1 == 1 && r < NBITS {
                for (q, col) in self.nextq.iter_mut().zip(self.cj.iter()) {
                    *q ^= col[r];
                }
            }
            gray >>= 1;
            r += 1;
        }
        self.seed = target;
    }

    /// Resets the generator to the start of the sequence (the origin).
    pub fn reset(&mut self) {
        for q in self.nextq.iter_mut() {
            *q = 0;
        }
        self.seed = 0;
    }

    /// Generates `count` points and returns them as a vector of coordinate
    /// vectors (row-major: outer index is the point, inner index the
    /// dimension).
    pub fn points(&mut self, count: usize) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            out.push(self.next_point());
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal 64-bit linear-congruential generator used *only* to produce a
    /// pseudo-random baseline for the discrepancy comparison.  Constants are
    /// Knuth's MMIX LCG.
    struct LcgRng {
        state: u64,
    }

    impl LcgRng {
        fn new(seed: u64) -> Self {
            Self {
                state: seed.wrapping_mul(6364136223846793005).wrapping_add(1),
            }
        }

        fn next_u32(&mut self) -> u32 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.state >> 32) as u32
        }

        fn next_f64(&mut self) -> f64 {
            f64::from(self.next_u32()) / ((1u64 << 32) as f64)
        }
    }

    #[test]
    fn rejects_zero_dimension() {
        let err = Niederreiter::new(0);
        assert!(err.is_err());
        if let Err(e) = err {
            assert!(matches!(e, RandError::InvalidParameter(_)));
        }
    }

    #[test]
    fn rejects_dimension_over_table_limit() {
        let err = Niederreiter::new(MAX_NIEDERREITER_DIMENSION + 1);
        assert!(err.is_err());
        // The maximum supported dimension is exactly the table size.
        assert!(Niederreiter::new(MAX_NIEDERREITER_DIMENSION).is_ok());
    }

    #[test]
    fn first_point_is_origin_and_all_in_unit_cube() {
        let mut seq = Niederreiter::new(5).expect("dim 5 valid");
        let first = seq.next_point();
        assert_eq!(first.len(), 5);
        for &x in &first {
            assert_eq!(x, 0.0, "first point must be the origin");
        }
        // All subsequent coordinates must lie in [0, 1).
        for _ in 0..2000 {
            let p = seq.next_point();
            for &x in &p {
                assert!((0.0..1.0).contains(&x), "coordinate {x} escaped [0,1)");
            }
        }
    }

    #[test]
    fn reproducible_same_dimension_identical_sequence() {
        let mut a = Niederreiter::new(4).expect("dim 4 valid");
        let mut b = Niederreiter::new(4).expect("dim 4 valid");
        let pa = a.points(512);
        let pb = b.points(512);
        assert_eq!(pa, pb, "same dimension must yield identical sequence");
    }

    #[test]
    fn skip_matches_sequential_advance() {
        // skip(k) then next must equal the (k)-th sequential point.
        let mut sequential = Niederreiter::new(3).expect("dim 3 valid");
        let all = sequential.points(300);

        for k in [0u32, 1, 2, 7, 64, 100, 255, 299] {
            let mut jumped = Niederreiter::new(3).expect("dim 3 valid");
            jumped.skip(k);
            assert_eq!(jumped.position(), k);
            let p = jumped.next_point();
            assert_eq!(p, all[k as usize], "skip({k}) mismatch");
        }
    }

    #[test]
    fn skip_in_steps_equivalent_to_one_jump() {
        let mut step = Niederreiter::new(2).expect("dim 2 valid");
        step.skip(50);
        step.skip(50);
        let mut once = Niederreiter::new(2).expect("dim 2 valid");
        once.skip(100);
        assert_eq!(step.position(), once.position());
        assert_eq!(step.next_point(), once.next_point());
    }

    #[test]
    fn reset_returns_to_origin() {
        let mut seq = Niederreiter::new(3).expect("dim 3 valid");
        let _ = seq.points(40);
        seq.reset();
        assert_eq!(seq.position(), 0);
        let p = seq.next_point();
        assert!(p.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn one_dimensional_projection_is_dyadic_permutation() {
        // For N = 2^m the 1-D values must be exactly the dyadic grid
        // {0, 1/N, ..., (N-1)/N} in some order (van der Corput property).
        let m = 8u32;
        let n = 1usize << m;
        let mut seq = Niederreiter::new(1).expect("dim 1 valid");
        let pts = seq.points(n);

        let mut buckets = vec![false; n];
        let scale = (1u64 << m) as f64;
        for p in &pts {
            let idx = (p[0] * scale).round() as usize;
            assert!(idx < n, "value {} mapped out of grid", p[0]);
            // Each grid cell must be hit exactly once (a permutation).
            assert!(
                !buckets[idx],
                "dyadic value {idx} repeated -> not a permutation"
            );
            buckets[idx] = true;
        }
        assert!(buckets.iter().all(|&b| b), "not all dyadic values covered");
    }

    #[test]
    fn tms_net_dyadic_box_balance() {
        // (t, m, s)-net balance: for N = 2^m points each elementary dyadic
        // box of volume 2^-m must contain close to its expected share.  We
        // split the unit square into 2^a x 2^b boxes with a + b = m so each
        // box has expected count 1.
        let m = 10u32;
        let n = 1usize << m;
        let a = 5u32; // 32 columns
        let b = 5u32; // 32 rows  -> 1024 boxes, expected 1 point each
        let cols = 1usize << a;
        let rows = 1usize << b;

        let mut seq = Niederreiter::new(2).expect("dim 2 valid");
        let pts = seq.points(n);

        let mut counts = vec![0u32; cols * rows];
        for p in &pts {
            let cx = ((p[0] * cols as f64) as usize).min(cols - 1);
            let cy = ((p[1] * rows as f64) as usize).min(rows - 1);
            counts[cy * cols + cx] += 1;
        }

        // A genuine digital net keeps every elementary box very close to its
        // expected single point; allow only a small slack for the net's
        // quality parameter t.
        let max_count = counts.iter().copied().max().unwrap_or(0);
        let min_count = counts.iter().copied().min().unwrap_or(0);
        assert!(
            max_count <= 4,
            "elementary box overfilled (max {max_count}); net balance broken"
        );
        // No fully empty regions in such a coarse grid for a good net.
        assert!(
            min_count >= 1,
            "found an empty elementary box (min {min_count})"
        );
    }

    #[test]
    fn low_discrepancy_beats_uniform_box_counting() {
        // Box-counting equidistribution on the unit square: the Niederreiter
        // net's worst-case cell deviation from the expected count must beat
        // i.i.d. uniform (LcgRng) for the same N.
        let m = 12u32;
        let n = 1usize << m; // 4096 points
        let grid = 16usize; // 16 x 16 = 256 cells, expected 16 points/cell
        let expected = n as f64 / (grid * grid) as f64;

        // Niederreiter points.
        let mut seq = Niederreiter::new(2).expect("dim 2 valid");
        let qmc = seq.points(n);
        let mut qmc_cells = vec![0u32; grid * grid];
        for p in &qmc {
            let cx = ((p[0] * grid as f64) as usize).min(grid - 1);
            let cy = ((p[1] * grid as f64) as usize).min(grid - 1);
            qmc_cells[cy * grid + cx] += 1;
        }
        let qmc_dev = qmc_cells
            .iter()
            .map(|&c| (f64::from(c) - expected).abs())
            .fold(0.0_f64, f64::max);

        // i.i.d. uniform points from the LCG baseline.
        let mut rng = LcgRng::new(0x1234_5678_9abc_def0);
        let mut unif_cells = vec![0u32; grid * grid];
        for _ in 0..n {
            let x = rng.next_f64();
            let y = rng.next_f64();
            let cx = ((x * grid as f64) as usize).min(grid - 1);
            let cy = ((y * grid as f64) as usize).min(grid - 1);
            unif_cells[cy * grid + cx] += 1;
        }
        let unif_dev = unif_cells
            .iter()
            .map(|&c| (f64::from(c) - expected).abs())
            .fold(0.0_f64, f64::max);

        assert!(
            qmc_dev < unif_dev,
            "Niederreiter max cell deviation {qmc_dev} should beat uniform {unif_dev}"
        );
    }

    #[test]
    fn high_dimension_generating_matrices_are_distinct() {
        // Use the full table; each dimension must have a distinct generating
        // matrix (distinct polynomials) and produce decorrelated coordinates.
        let dim = MAX_NIEDERREITER_DIMENSION;
        let mut seq = Niederreiter::new(dim).expect("max dim valid");
        // Full generating matrices must differ across dimensions.  (Low-degree
        // polynomials can legitimately share leading columns -- e.g. both
        // degree-1 irreducibles give column 0 == 2^30 -- so the whole matrix,
        // not a single column, is what distinguishes the dimensions.)
        for i in 0..dim {
            for j in (i + 1)..dim {
                assert_ne!(
                    seq.cj[i], seq.cj[j],
                    "dimensions {i} and {j} share an identical generating matrix"
                );
            }
        }
        // Sanity: points stay in range across all dimensions.
        let pts = seq.points(1024);
        for p in &pts {
            assert_eq!(p.len(), dim);
            for &x in p {
                assert!((0.0..1.0).contains(&x));
            }
        }
    }

    #[test]
    fn two_d_mean_converges_near_half() {
        // Quasi-Monte-Carlo estimate of the mean of each coordinate over a
        // power-of-two sample should be very close to 0.5.
        let n = 1usize << 12;
        let mut seq = Niederreiter::new(2).expect("dim 2 valid");
        let pts = seq.points(n);
        let mut sx = 0.0;
        let mut sy = 0.0;
        for p in &pts {
            sx += p[0];
            sy += p[1];
        }
        let mx = sx / n as f64;
        let my = sy / n as f64;
        assert!((mx - 0.5).abs() < 0.01, "mean x {mx} far from 0.5");
        assert!((my - 0.5).abs() < 0.01, "mean y {my} far from 0.5");
    }

    #[test]
    fn points_helper_length_and_consistency() {
        let mut a = Niederreiter::new(3).expect("dim 3 valid");
        let bulk = a.points(100);
        assert_eq!(bulk.len(), 100);

        // Calling next_point 100 times must match points(100).
        let mut b = Niederreiter::new(3).expect("dim 3 valid");
        for (k, expected) in bulk.iter().enumerate() {
            let got = b.next_point();
            assert_eq!(&got, expected, "point {k} mismatch between APIs");
        }
    }
}
