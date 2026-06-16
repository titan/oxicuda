//! BLAST-style seed-and-extend local alignment.
//!
//! Classic heuristic local aligner: index every exact `k`-mer ("seed") shared
//! between the query and the subject, then extend each seed in both directions
//! with an ungapped *X-drop* extension to grow high-scoring segment pairs
//! (HSPs). This trades the optimality of Smith–Waterman for near-linear speed
//! on long sequences, finding only the alignments anchored by an exact seed.

use crate::error::{SeqError, SeqResult};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Configuration for the seed-and-extend aligner.
#[derive(Debug, Clone, Copy)]
pub struct BlastConfig {
    /// Length of the exact-match seed (`k`-mer). Must be `> 0`.
    pub kmer_len: usize,
    /// Score awarded for each matching position.
    pub match_score: i32,
    /// Score (penalty, typically negative) for each mismatching position.
    pub mismatch_score: i32,
    /// X-drop threshold: extension stops once the running score falls more than
    /// this far below the best score seen so far. Must be `> 0`.
    pub x_drop: i32,
}

impl Default for BlastConfig {
    fn default() -> Self {
        Self {
            kmer_len: 3,
            match_score: 2,
            mismatch_score: -1,
            x_drop: 5,
        }
    }
}

/// A high-scoring segment pair (ungapped local alignment) found by extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hsp {
    /// Start offset of the segment in the query.
    pub query_start: usize,
    /// Start offset of the segment in the subject.
    pub subject_start: usize,
    /// Length of the aligned (ungapped) segment.
    pub length: usize,
    /// Alignment score of the segment.
    pub score: i32,
}

/// Seed-and-extend aligner with a validated configuration.
#[derive(Debug, Clone)]
pub struct BlastAligner {
    config: BlastConfig,
}

impl BlastAligner {
    /// Construct a new aligner, validating the configuration.
    ///
    /// # Errors
    /// * [`SeqError::InvalidConfiguration`] if `kmer_len == 0`.
    /// * [`SeqError::InvalidParameter`] if `x_drop <= 0`.
    pub fn new(config: BlastConfig) -> SeqResult<Self> {
        if config.kmer_len == 0 {
            return Err(SeqError::InvalidConfiguration(
                "kmer_len must be > 0".to_string(),
            ));
        }
        if config.x_drop <= 0 {
            return Err(SeqError::InvalidParameter {
                name: "x_drop".to_string(),
                value: f64::from(config.x_drop),
            });
        }
        Ok(Self { config })
    }

    /// Borrow the validated configuration.
    pub fn config(&self) -> &BlastConfig {
        &self.config
    }

    /// Search `subject` for high-scoring segment pairs anchored by exact
    /// `k`-mer seeds shared with `query`.
    ///
    /// Returned HSPs are deduplicated (segments subsumed by a longer HSP on the
    /// same diagonal are dropped) and sorted by descending score, then by
    /// position for determinism.
    ///
    /// # Errors
    /// [`SeqError::EmptyInput`] if either sequence is empty. Non-empty
    /// sequences shorter than one seed simply yield an empty result.
    pub fn search<T: Eq + Hash>(&self, query: &[T], subject: &[T]) -> SeqResult<Vec<Hsp>> {
        if query.is_empty() || subject.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        let k = self.config.kmer_len;
        // Sequences shorter than one seed cannot share a k-mer.
        if query.len() < k || subject.len() < k {
            return Ok(Vec::new());
        }

        // Index every k-mer of the subject → list of start positions.
        let mut index: HashMap<&[T], Vec<usize>> = HashMap::new();
        for sp in 0..=(subject.len() - k) {
            index.entry(&subject[sp..sp + k]).or_default().push(sp);
        }

        // Every (query k-mer, subject hit) pair is a seed; extend each one.
        let mut seen: HashSet<(usize, usize, usize)> = HashSet::new();
        let mut hsps: Vec<Hsp> = Vec::new();
        for qp in 0..=(query.len() - k) {
            let Some(hits) = index.get(&query[qp..qp + k]) else {
                continue;
            };
            for &sp in hits {
                let hsp = self.extend_seed(query, subject, qp, sp);
                if seen.insert((hsp.query_start, hsp.subject_start, hsp.length)) {
                    hsps.push(hsp);
                }
            }
        }

        // Drop HSPs whose span is contained in a longer HSP on the same diagonal.
        let mut kept = dedup_subsumed(hsps);
        // Deterministic ordering: best score first, then by position.
        kept.sort_by(|x, y| {
            y.score
                .cmp(&x.score)
                .then(x.query_start.cmp(&y.query_start))
                .then(x.subject_start.cmp(&y.subject_start))
        });
        Ok(kept)
    }

    /// Ungapped X-drop extension of an exact `k`-mer seed anchored at
    /// query offset `qp` and subject offset `sp`.
    fn extend_seed<T: Eq>(&self, query: &[T], subject: &[T], qp: usize, sp: usize) -> Hsp {
        let k = self.config.kmer_len;
        let mtch = self.config.match_score;
        let mis = self.config.mismatch_score;
        let x_drop = self.config.x_drop;

        // Right extension beyond the seed.
        let mut right_gain = 0i32;
        let mut right_len = 0usize;
        {
            let mut run = 0i32;
            let mut ext = 0usize;
            while qp + k + ext < query.len() && sp + k + ext < subject.len() {
                let step = if query[qp + k + ext] == subject[sp + k + ext] {
                    mtch
                } else {
                    mis
                };
                run = run.saturating_add(step);
                ext += 1;
                if run > right_gain {
                    right_gain = run;
                    right_len = ext;
                }
                if right_gain - run > x_drop {
                    break;
                }
            }
        }

        // Left extension before the seed.
        let mut left_gain = 0i32;
        let mut left_len = 0usize;
        {
            let mut run = 0i32;
            let mut ext = 0usize;
            while qp > ext && sp > ext {
                let step = if query[qp - 1 - ext] == subject[sp - 1 - ext] {
                    mtch
                } else {
                    mis
                };
                run = run.saturating_add(step);
                ext += 1;
                if run > left_gain {
                    left_gain = run;
                    left_len = ext;
                }
                if left_gain - run > x_drop {
                    break;
                }
            }
        }

        let seed_score = (k as i32).saturating_mul(mtch);
        Hsp {
            query_start: qp - left_len,
            subject_start: sp - left_len,
            length: left_len + k + right_len,
            score: seed_score
                .saturating_add(left_gain)
                .saturating_add(right_gain),
        }
    }
}

/// Remove HSPs whose span is fully contained within a strictly longer HSP on
/// the same diagonal (`subject_start − query_start`).
fn dedup_subsumed(hsps: Vec<Hsp>) -> Vec<Hsp> {
    let n = hsps.len();
    let mut keep = vec![true; n];
    for i in 0..n {
        for j in 0..n {
            if i == j || !keep[j] {
                continue;
            }
            let di = hsps[i].subject_start as isize - hsps[i].query_start as isize;
            let dj = hsps[j].subject_start as isize - hsps[j].query_start as isize;
            if di != dj {
                continue;
            }
            let i_start = hsps[i].query_start;
            let i_end = i_start + hsps[i].length;
            let j_start = hsps[j].query_start;
            let j_end = j_start + hsps[j].length;
            if j_start <= i_start && i_end <= j_end && hsps[j].length > hsps[i].length {
                keep[i] = false;
                break;
            }
        }
    }
    hsps.into_iter()
        .zip(keep)
        .filter_map(|(h, k)| k.then_some(h))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_substring_one_full_hsp() {
        let cfg = BlastConfig {
            kmer_len: 3,
            match_score: 2,
            mismatch_score: -1,
            x_drop: 5,
        };
        let aligner = BlastAligner::new(cfg).expect("cfg");
        let query = b"GATTACA";
        let subject = b"AAGATTACAGG";
        let hsps = aligner.search(query, subject).expect("search");
        assert_eq!(hsps.len(), 1, "expected exactly one HSP, got {hsps:?}");
        let h = hsps[0];
        assert_eq!(h.length, query.len());
        assert_eq!(h.score, query.len() as i32 * cfg.match_score);
        assert_eq!(h.query_start, 0);
        assert_eq!(h.subject_start, 2);
    }

    #[test]
    fn no_shared_kmer_yields_empty() {
        let aligner = BlastAligner::new(BlastConfig::default()).expect("cfg");
        let query = b"AAAAAA";
        let subject = b"CCCCCC";
        let hsps = aligner.search(query, subject).expect("search");
        assert!(hsps.is_empty());
    }

    #[test]
    fn longer_kmer_reduces_sensitivity() {
        // A divergent pair: short seeds still hit, a long seed finds nothing.
        let query = b"ACGTACGTAC";
        let subject = b"TGCATGCATG";
        let short = BlastAligner::new(BlastConfig {
            kmer_len: 1,
            ..BlastConfig::default()
        })
        .expect("cfg");
        let long = BlastAligner::new(BlastConfig {
            kmer_len: 6,
            ..BlastConfig::default()
        })
        .expect("cfg");
        let short_hits = short.search(query, subject).expect("short");
        let long_hits = long.search(query, subject).expect("long");
        assert!(long_hits.len() <= short_hits.len());
        assert!(long_hits.is_empty());
    }

    #[test]
    fn xdrop_stops_at_mismatch_run() {
        let cfg = BlastConfig {
            kmer_len: 5,
            match_score: 1,
            mismatch_score: -2,
            x_drop: 1,
        };
        let aligner = BlastAligner::new(cfg).expect("cfg");
        let query = b"TTTTTGGGGG";
        let subject = b"TTTTTAAAAA";
        let hsps = aligner.search(query, subject).expect("search");
        assert_eq!(hsps.len(), 1);
        // Only the "TTTTT" seed survives; the mismatched tail is X-dropped.
        assert_eq!(hsps[0].length, 5);
        assert_eq!(hsps[0].score, 5);
        assert_eq!(hsps[0].query_start, 0);
        assert_eq!(hsps[0].subject_start, 0);
    }

    #[test]
    fn invalid_config_and_empty_input_error() {
        // k = 0 rejected at construction.
        assert!(
            BlastAligner::new(BlastConfig {
                kmer_len: 0,
                ..BlastConfig::default()
            })
            .is_err()
        );
        // Non-positive x_drop rejected at construction.
        assert!(
            BlastAligner::new(BlastConfig {
                x_drop: 0,
                ..BlastConfig::default()
            })
            .is_err()
        );
        // Empty input rejected at search time.
        let aligner = BlastAligner::new(BlastConfig::default()).expect("cfg");
        let empty: &[u8] = b"";
        assert!(aligner.search(empty, b"ACGT").is_err());
        assert!(aligner.search(b"ACGT", empty).is_err());
    }
}
