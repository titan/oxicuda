//! Nussinov (1978) RNA secondary-structure folding.
//!
//! The Nussinov–Jacobson dynamic program maximises the number of
//! complementary base pairs in an RNA sequence subject to two structural
//! constraints:
//!
//! 1. **No pseudoknots** — pairings must be properly nested.
//! 2. **Minimum hairpin loop length** — a pair `(i, k)` is only admissible
//!    when at least `min_loop` unpaired bases lie strictly between the paired
//!    positions, i.e. `k - i - 1 >= min_loop`.
//!
//! # Recurrence
//!
//! For the inclusive substring `[i..=j]` (0-based) with `M[i][j]` the maximum
//! number of pairs:
//!
//! ```text
//! M[i][j] = 0                                            if i >= j
//! M[i][j] = max(
//!     M[i+1][j],                                         // i unpaired
//!     max over k in (i+1)..=j with can_pair(seq[i], seq[k])
//!          and (k - i - 1) >= min_loop of
//!          M[i+1][k-1] + M[k+1][j] + 1                   // i pairs with k
//! )
//! ```
//!
//! The table is filled by increasing substring length, and the optimum pair
//! count for the whole sequence is `M[0][n-1]`.
//!
//! A standard explicit-stack traceback recovers a dot-bracket string and the
//! explicit list of paired indices. The bracket string is guaranteed to be
//! properly nested and balanced.
//!
//! # Reference
//!
//! Nussinov, R., Pieczenik, G., Griggs, J. R., & Kleitman, D. J. (1978).
//! *Algorithms for loop matchings.* SIAM Journal on Applied Mathematics,
//! 35(1), 68–82.

use crate::error::{SeqError, SeqResult};

/// Configuration for the Nussinov folding algorithm.
#[derive(Debug, Clone, Copy)]
pub struct NussinovConfig {
    /// Minimum number of unpaired bases strictly between a paired position
    /// `(i, k)`, i.e. the requirement `k - i - 1 >= min_loop`. The biological
    /// default is `3`.
    pub min_loop: usize,
    /// When `true`, the non-canonical G–U / U–G wobble pair is permitted in
    /// addition to the Watson–Crick pairs A–U and G–C.
    pub allow_wobble: bool,
}

impl Default for NussinovConfig {
    fn default() -> Self {
        Self {
            min_loop: 3,
            allow_wobble: false,
        }
    }
}

impl NussinovConfig {
    /// Construct a new configuration.
    ///
    /// `min_loop` is the minimum hairpin loop length (`>= 0`); `allow_wobble`
    /// toggles non-canonical G–U pairing. The configuration is always valid for
    /// any `usize` `min_loop`, so this constructor is infallible.
    pub fn new(min_loop: usize, allow_wobble: bool) -> Self {
        Self {
            min_loop,
            allow_wobble,
        }
    }
}

/// Result of folding an RNA sequence with [`nussinov_fold`].
#[derive(Debug, Clone)]
pub struct NussinovResult {
    /// The maximum number of base pairs found, equal to `M[0][n-1]`.
    pub pairs: usize,
    /// Dot-bracket representation of the structure, of length `n`. Each byte is
    /// one of `'('`, `')'`, or `'.'`; the bracket string is properly nested.
    pub structure: String,
    /// Explicit list of paired index positions `(i, k)` with `i < k`.
    pub pair_list: Vec<(usize, usize)>,
}

/// Normalise a nucleotide byte to a canonical upper-case base.
///
/// Both cases of `A`, `C`, `G`, and `U` map to themselves (upper-cased), and
/// `T`/`t` is treated as `U` for DNA tolerance. Any other byte maps to `0`,
/// which never pairs.
#[inline]
fn canonical_base(byte: u8) -> u8 {
    match byte {
        b'A' | b'a' => b'A',
        b'C' | b'c' => b'C',
        b'G' | b'g' => b'G',
        b'U' | b'u' | b'T' | b't' => b'U',
        _ => 0,
    }
}

/// Test whether two nucleotide bytes form an admissible base pair.
///
/// Returns `true` for the Watson–Crick complementary pairs A–U and G–C in
/// either order, and additionally for G–U / U–G when `allow_wobble` is `true`.
/// Comparison is case-insensitive and `T` is treated as `U`. Non-nucleotide
/// bytes never pair.
pub fn can_pair(a: u8, b: u8, allow_wobble: bool) -> bool {
    let x = canonical_base(a);
    let y = canonical_base(b);
    if x == 0 || y == 0 {
        return false;
    }
    match (x, y) {
        (b'A', b'U') | (b'U', b'A') => true,
        (b'G', b'C') | (b'C', b'G') => true,
        (b'G', b'U') | (b'U', b'G') => allow_wobble,
        _ => false,
    }
}

/// Fold an RNA sequence with the Nussinov maximum-base-pairing algorithm.
///
/// Accepts a raw byte sequence over the alphabet `{A, C, G, U}` (case
/// insensitive, with `T` treated as `U`). Bases outside this alphabet are
/// retained in the output structure as unpaired `'.'` positions and never
/// participate in a pair.
///
/// # Errors
///
/// Returns [`SeqError::EmptyInput`] when `seq` is empty.
pub fn nussinov_fold(seq: &[u8], config: &NussinovConfig) -> SeqResult<NussinovResult> {
    let n = seq.len();
    if n == 0 {
        return Err(SeqError::EmptyInput);
    }

    // Flat DP table `dp[i * n + j] = M[i][j]`, the maximum number of pairs in
    // the inclusive substring `[i..=j]`. Entries with `i >= j` stay 0.
    let mut dp = vec![0usize; n * n];
    let min_loop = config.min_loop;
    let allow_wobble = config.allow_wobble;

    // Fill by increasing substring length `span = j - i`.
    for span in 1..n {
        for i in 0..(n - span) {
            let j = i + span;

            // Case (1): position `i` is left unpaired, value `M[i+1][j]`. Here
            // `j = i + span` with `span >= 1`, so `i < j` and the range
            // `[i+1..=j]` is always non-empty.
            let mut best = dp[(i + 1) * n + j];

            // Case (2): position `i` pairs with some `k` in `(i+1)..=j`.
            for k in (i + 1)..=j {
                if k - i - 1 < min_loop {
                    // Too few unpaired bases between i and k for a hairpin.
                    continue;
                }
                if !can_pair(seq[i], seq[k], allow_wobble) {
                    continue;
                }
                // Inner range `[i+1..=k-1]`: empty (0) when `i + 1 > k - 1`,
                // equivalently when `i + 1 >= k`.
                let inner = if i + 1 < k {
                    dp[(i + 1) * n + (k - 1)]
                } else {
                    0
                };
                // Outer range `[k+1..=j]`: empty (0) when `k + 1 > j`,
                // equivalently when `k >= j`.
                let outer = if k < j { dp[(k + 1) * n + j] } else { 0 };
                let candidate = inner + outer + 1;
                if candidate > best {
                    best = candidate;
                }
            }

            dp[i * n + j] = best;
        }
    }

    let pairs = dp[n - 1]; // dp[0 * n + (n - 1)] == M[0][n-1]

    // Explicit-stack traceback over the DP table.
    let mut structure = vec![b'.'; n];
    let mut pair_list: Vec<(usize, usize)> = Vec::new();
    let mut stack: Vec<(usize, usize)> = vec![(0, n - 1)];

    while let Some((i, j)) = stack.pop() {
        if i >= j {
            continue;
        }
        // If leaving `i` unpaired already realises the optimum, do so.
        // The range `[i+1..=j]` is non-empty exactly when `i < j` (true here,
        // since `i >= j` was handled above).
        let unpaired = dp[(i + 1) * n + j];
        if dp[i * n + j] == unpaired {
            stack.push((i + 1, j));
            continue;
        }
        // Otherwise find the `k` that realised the pairing of `i`.
        let mut matched = false;
        for k in (i + 1)..=j {
            if k - i - 1 < min_loop {
                continue;
            }
            if !can_pair(seq[i], seq[k], allow_wobble) {
                continue;
            }
            let inner = if i + 1 < k {
                dp[(i + 1) * n + (k - 1)]
            } else {
                0
            };
            let outer = if k < j { dp[(k + 1) * n + j] } else { 0 };
            if dp[i * n + j] == inner + outer + 1 {
                structure[i] = b'(';
                structure[k] = b')';
                pair_list.push((i, k));
                // `k >= i + 1 >= 1`, so `k - 1` cannot underflow.
                stack.push((i + 1, k - 1));
                stack.push((k + 1, j));
                matched = true;
                break;
            }
        }
        if !matched {
            // Defensive fallback: should be unreachable because the DP value
            // was strictly greater than the `i`-unpaired option, so a pairing
            // `k` must exist. Treat `i` as unpaired to keep the traceback
            // total without panicking.
            stack.push((i + 1, j));
        }
    }

    pair_list.sort_unstable();

    // `structure` is built only from the ASCII bytes '(', ')', '.', so this
    // conversion cannot fail; map any theoretical error to a configuration
    // error rather than unwrapping.
    let structure = String::from_utf8(structure)
        .map_err(|e| SeqError::InvalidConfiguration(format!("non-utf8 structure: {e}")))?;

    Ok(NussinovResult {
        pairs,
        structure,
        pair_list,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scan a dot-bracket string and confirm it is properly nested: the running
    /// `'(' = +1`, `')' = -1` counter never dips below zero and ends at zero.
    fn is_balanced(structure: &str) -> bool {
        let mut depth: i64 = 0;
        for ch in structure.chars() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth < 0 {
                        return false;
                    }
                }
                '.' => {}
                _ => return false,
            }
        }
        depth == 0
    }

    #[test]
    fn can_pair_watson_crick() {
        assert!(can_pair(b'A', b'U', false));
        assert!(can_pair(b'U', b'A', false));
        assert!(can_pair(b'G', b'C', false));
        assert!(can_pair(b'C', b'G', false));
        // Lower-case and DNA T tolerance.
        assert!(can_pair(b'a', b'u', false));
        assert!(can_pair(b'A', b'T', false));
        assert!(can_pair(b't', b'a', false));
        // Non-complementary and non-nucleotide.
        assert!(!can_pair(b'A', b'G', false));
        assert!(!can_pair(b'A', b'C', false));
        assert!(!can_pair(b'N', b'U', false));
    }

    #[test]
    fn can_pair_wobble_toggle() {
        assert!(!can_pair(b'G', b'U', false));
        assert!(!can_pair(b'U', b'G', false));
        assert!(can_pair(b'G', b'U', true));
        assert!(can_pair(b'U', b'G', true));
        // Wobble flag never enables a non-pair.
        assert!(!can_pair(b'A', b'A', true));
    }

    #[test]
    fn empty_sequence_errors() {
        let cfg = NussinovConfig::default();
        let err = nussinov_fold(b"", &cfg);
        assert!(matches!(err, Err(SeqError::EmptyInput)));
    }

    // (a) Classic stem with default config.
    #[test]
    fn gggaaaccc_full_stem() {
        let cfg = NussinovConfig::default();
        let r = nussinov_fold(b"GGGAAACCC", &cfg).expect("fold ok");
        assert_eq!(r.pairs, 3, "expected 3 pairs");
        assert_eq!(r.structure, "(((...)))", "exact dot-bracket");
        // (b) nested check on this result.
        assert!(is_balanced(&r.structure), "structure must be nested");
        // (f) consistency: pair_list length, bracket counts.
        assert_eq!(r.pair_list.len(), r.pairs);
        assert_eq!(r.structure.matches('(').count(), r.pairs);
        assert_eq!(r.structure.matches(')').count(), r.pairs);
    }

    // (b) Additional nesting checks on several inputs.
    #[test]
    fn structures_are_nested() {
        let cfg = NussinovConfig::default();
        for seq in [
            &b"GGGAAACCC"[..],
            &b"GGGGGAAACCCCC"[..],
            &b"AAAA"[..],
            &b"GCGCGCAAAUGCGCGC"[..],
            &b"AUGCAUGCAUGC"[..],
        ] {
            let r = nussinov_fold(seq, &cfg).expect("fold ok");
            assert_eq!(r.structure.len(), seq.len());
            assert!(
                is_balanced(&r.structure),
                "not nested for {:?}: {}",
                core::str::from_utf8(seq),
                r.structure
            );
            // Each emitted pair respects the loop constraint.
            // `k - i > min_loop` is equivalent to `k - i - 1 >= min_loop`.
            for &(i, k) in &r.pair_list {
                assert!(i < k);
                assert!(k - i > cfg.min_loop, "pair ({i},{k}) violates min_loop");
            }
        }
    }

    // (c) Minimum-loop constraint.
    #[test]
    fn min_loop_constraint_respected() {
        let cfg = NussinovConfig::default();

        let r = nussinov_fold(b"GGGGCCCC", &cfg).expect("fold ok");
        for &(i, k) in &r.pair_list {
            assert!(i < k);
            // `k - i > 3` is equivalent to loop length `k - i - 1 >= 3`.
            assert!(k - i > 3, "pair ({i},{k}) loop length {} < 3", k - i - 1);
        }

        // GCGC: closest G-C across the 4-mer is (0,3) with loop length 2 < 3.
        let r4 = nussinov_fold(b"GCGC", &cfg).expect("fold ok");
        assert_eq!(r4.pairs, 0, "no admissible pair in GCGC");
        assert_eq!(r4.structure, "....");

        // GC: loop length 0 < 3.
        let r2 = nussinov_fold(b"GC", &cfg).expect("fold ok");
        assert_eq!(r2.pairs, 0);
        assert_eq!(r2.structure, "..");
    }

    // (d) No possible pairs.
    #[test]
    fn aaaa_no_pairs() {
        let cfg = NussinovConfig::default();
        let r = nussinov_fold(b"AAAA", &cfg).expect("fold ok");
        assert_eq!(r.pairs, 0);
        assert_eq!(r.structure, "....");
        assert!(r.pair_list.is_empty());
    }

    // (e) Perfect palindromic stem with a legal innermost hairpin loop.
    #[test]
    fn full_five_pair_stem() {
        let cfg = NussinovConfig::default();
        // 5×G, 3×A, 5×C, n = 13. Innermost pair (4,8) has loop length 3.
        let r = nussinov_fold(b"GGGGGAAACCCCC", &cfg).expect("fold ok");
        assert_eq!(r.pairs, 5, "expected a full 5-pair stem");
        assert_eq!(r.structure, "(((((...)))))");
        assert!(is_balanced(&r.structure));
        // Innermost emitted pair must still leave >= 3 unpaired bases.
        let innermost = r
            .pair_list
            .iter()
            .max_by_key(|&&(i, _)| i)
            .copied()
            .expect("at least one pair");
        assert!(innermost.0 < innermost.1);
        // `hi - lo > 3` is equivalent to loop length `hi - lo - 1 >= 3`.
        assert!(innermost.1 - innermost.0 > 3, "innermost loop too short");
    }

    // (f) Traceback consistency: pair_list length and bracket counts agree.
    #[test]
    fn traceback_consistency() {
        let cfg = NussinovConfig::default();
        for seq in [
            &b"GGGAAACCC"[..],
            &b"GGGGGAAACCCCC"[..],
            &b"GCGCGCAAAUGCGCGC"[..],
        ] {
            let r = nussinov_fold(seq, &cfg).expect("fold ok");
            assert_eq!(r.pair_list.len(), r.pairs, "pair_list len == pairs");
            assert_eq!(
                r.structure.matches('(').count(),
                r.pairs,
                "open bracket count == pairs"
            );
            assert_eq!(
                r.structure.matches(')').count(),
                r.pairs,
                "close bracket count == pairs"
            );
        }
    }

    // (g) Wobble strictly increases the achievable pair count.
    #[test]
    fn wobble_increases_pairs() {
        let no_wobble = NussinovConfig::new(3, false);
        let wobble = NussinovConfig::new(3, true);

        let seq = b"GGGUUU";
        let r_no = nussinov_fold(seq, &no_wobble).expect("fold ok");
        let r_yes = nussinov_fold(seq, &wobble).expect("fold ok");

        assert_eq!(r_no.pairs, 0, "no WC/AU/GC pairs without wobble");
        assert!(
            r_yes.pairs > r_no.pairs,
            "wobble must add pairs: {} > {}",
            r_yes.pairs,
            r_no.pairs
        );
        // The wobble structure must still be nested and respect min_loop.
        assert!(is_balanced(&r_yes.structure));
        for &(i, k) in &r_yes.pair_list {
            assert!(i < k);
            // `k - i > 3` is equivalent to loop length `k - i - 1 >= 3`.
            assert!(k - i > 3);
        }
    }

    #[test]
    fn config_defaults() {
        let cfg = NussinovConfig::default();
        assert_eq!(cfg.min_loop, 3);
        assert!(!cfg.allow_wobble);
        let cfg2 = NussinovConfig::new(5, true);
        assert_eq!(cfg2.min_loop, 5);
        assert!(cfg2.allow_wobble);
    }
}
