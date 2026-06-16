//! WFA — the Wavefront Alignment algorithm for gap-affine **global** alignment.
//!
//! This is a faithful implementation of the algorithm of Marco-Sola, Moure,
//! Moreto & Espinosa, *"Fast gap-affine pairwise alignment using the wavefront
//! algorithm"*, Bioinformatics 37(4):456–463, 2021.
//!
//! # Model
//!
//! WFA **minimizes** an alignment *penalty* under the gap-affine model
//!
//! ```text
//! match     → 0
//! mismatch  → x          (x > 0)
//! gap run   → o + k·e    (gap-open o paid once per run, plus e per gap symbol
//!                          INCLUDING the first; o ≥ 0, e > 0)
//! ```
//!
//! Instead of filling an `(m+1)·(n+1)` dynamic-programming matrix, WFA tracks,
//! for each increasing penalty `s`, the *furthest-reaching* point on every
//! diagonal `k = i − j` (the "wavefront"). On similar sequences only a narrow
//! band of diagonals is ever touched, which is what makes the algorithm fast.
//!
//! Three wavefront components are maintained per penalty `s`:
//!
//! * `M` — the match / substitution path (the alignment proper),
//! * `I` — the *insertion* path (a gap in `a`, consuming a character of `b`),
//! * `D` — the *deletion* path (a gap in `b`, consuming a character of `a`).
//!
//! # Convention
//!
//! We align `a` along the rows (index `i`, length `m`) and `b` along the
//! columns (index `j`, length `n`). A diagonal is `k = i − j`. The *offset*
//! stored on a diagonal is `i`, the number of characters of `a` consumed, so
//! the matching column is `j = i − k`. The optimum is reached when the `M`
//! wavefront on the final diagonal `k_final = m − n` attains offset `m`
//! (equivalently `i = m`, `j = n`).
//!
//! * [`WfaOp::Ins`] is a gap in `a`: it consumes one character of `b` only.
//! * [`WfaOp::Del`] is a gap in `b`: it consumes one character of `a` only.
//!
//! # Cross-check with Gotoh
//!
//! [`crate::alignment::gotoh::gotoh_align`] solves the *same* problem but
//! **maximizes** a score. Given a [`GotohScoring`] `(M, mis, go, ge)` we derive
//! the (×2-scaled, integral) WFA penalties
//!
//! ```text
//! x = 2·(M − mis)        // mismatch penalty
//! o = 2·(ge − go)        // gap-open penalty
//! e = M − 2·ge           // gap-extend penalty
//! ```
//!
//! Run WFA to obtain the minimum penalty `P`; the equivalent Gotoh maximum
//! score is then
//!
//! ```text
//! gotoh_score = ((m + n)·M − P) / 2
//! ```
//!
//! which is exact (the division is always even). [`WfaAlignment`] reports both
//! the raw `penalty` and the converted `score`.

use crate::alignment::gotoh::GotohScoring;
use crate::error::{SeqError, SeqResult};

/// A single edit operation of a WFA alignment, in left-to-right order.
///
/// The convention is:
///
/// * [`WfaOp::Match`] / [`WfaOp::Mismatch`] consume one character of *both* `a`
///   and `b`.
/// * [`WfaOp::Ins`] is a gap in `a`; it consumes one character of `b` only.
/// * [`WfaOp::Del`] is a gap in `b`; it consumes one character of `a` only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WfaOp {
    /// Aligned, equal characters (`a[i] == b[j]`).
    Match,
    /// Aligned, unequal characters (`a[i] != b[j]`).
    Mismatch,
    /// Insertion: gap in `a`, consumes `b[j]`.
    Ins,
    /// Deletion: gap in `b`, consumes `a[i]`.
    Del,
}

/// Result of a WFA gap-affine global alignment.
#[derive(Debug, Clone)]
pub struct WfaAlignment {
    /// The equivalent Gotoh **maximum** score (`((m+n)·M − penalty) / 2`).
    pub score: i32,
    /// The raw, ×2-scaled WFA **minimum** penalty.
    pub penalty: i32,
    /// The optimal alignment as a left-to-right list of edit operations.
    pub cigar: Vec<WfaOp>,
}

/// The ×2-scaled, integral gap-affine penalties derived from a [`GotohScoring`].
#[derive(Debug, Clone, Copy)]
struct WfaPenalties {
    /// Mismatch penalty `x`.
    x: i32,
    /// Gap-open penalty `o` (paid once per gap run).
    o: i32,
    /// Gap-extend penalty `e` (paid per gap symbol, including the first).
    e: i32,
}

impl WfaPenalties {
    /// Derive the ×2-scaled WFA penalties from a Gotoh scoring scheme.
    ///
    /// Returns [`SeqError::InvalidConfiguration`] when the resulting affine
    /// model is degenerate (the algorithm needs `x > 0`, `o ≥ 0`, `e > 0`).
    fn from_gotoh(sc: &GotohScoring) -> SeqResult<Self> {
        let x = 2 * (sc.match_score - sc.mismatch);
        let o = 2 * (sc.gap_extend - sc.gap_open);
        let e = sc.match_score - 2 * sc.gap_extend;
        if x <= 0 {
            return Err(SeqError::InvalidConfiguration(format!(
                "WFA requires a positive mismatch penalty (match_score must exceed mismatch); \
                 derived x = {x}"
            )));
        }
        if o < 0 {
            return Err(SeqError::InvalidConfiguration(format!(
                "WFA requires a non-negative gap-open penalty (gap_extend must be >= gap_open); \
                 derived o = {o}"
            )));
        }
        if e <= 0 {
            return Err(SeqError::InvalidConfiguration(format!(
                "WFA requires a positive gap-extend penalty (match_score must exceed 2*gap_extend); \
                 derived e = {e}"
            )));
        }
        Ok(Self { x, o, e })
    }
}

/// Sentinel marking an unreachable diagonal/offset.
const NIL: i32 = i32::MIN;

/// A single wavefront component (`M`, `I` or `D`) at one penalty.
///
/// Offsets are stored densely over the inclusive diagonal range
/// `[lo, hi]`; unreachable diagonals hold [`NIL`].
#[derive(Debug, Clone)]
struct Wavefront {
    /// Lowest diagonal index covered (inclusive).
    lo: i32,
    /// Highest diagonal index covered (inclusive).
    hi: i32,
    /// Dense offsets indexed by `k - lo`.
    offsets: Vec<i32>,
}

impl Wavefront {
    /// An empty (all-[`NIL`]) wavefront covering `[lo, hi]`.
    fn new(lo: i32, hi: i32) -> Self {
        let len = if hi >= lo { (hi - lo + 1) as usize } else { 0 };
        Self {
            lo,
            hi,
            offsets: vec![NIL; len],
        }
    }

    /// The offset on diagonal `k`, or [`NIL`] if out of range / unreachable.
    #[inline]
    fn get(&self, k: i32) -> i32 {
        if k < self.lo || k > self.hi {
            NIL
        } else {
            self.offsets[(k - self.lo) as usize]
        }
    }

    /// Set the offset on diagonal `k` (no-op if `k` is out of range).
    #[inline]
    fn set(&mut self, k: i32, v: i32) {
        if k >= self.lo && k <= self.hi {
            self.offsets[(k - self.lo) as usize] = v;
        }
    }
}

/// The three wavefront components recorded at a single penalty `s`.
#[derive(Debug, Clone)]
struct WfSet {
    m: Wavefront,
    i: Wavefront,
    d: Wavefront,
}

/// Run the WFA gap-affine **global** alignment of `a` against `b`.
///
/// `sc` is interpreted exactly as in [`crate::alignment::gotoh::gotoh_align`];
/// the returned [`WfaAlignment::score`] is guaranteed to equal that function's
/// score on the same inputs.
///
/// # Errors
///
/// * [`SeqError::EmptyInput`] if either sequence is empty (mirroring Gotoh).
/// * [`SeqError::InvalidConfiguration`] if the derived affine model is
///   degenerate (see `WfaPenalties::from_gotoh`).
pub fn wfa_align(a: &[u8], b: &[u8], sc: &GotohScoring) -> SeqResult<WfaAlignment> {
    let m = a.len();
    let n = b.len();
    if m == 0 || n == 0 {
        return Err(SeqError::EmptyInput);
    }
    let pen = WfaPenalties::from_gotoh(sc)?;

    let m_i = m as i32;
    let n_i = n as i32;
    let k_final = m_i - n_i;
    let a_off_max = m_i; // offset == i, so the final offset is m.

    // History of wavefront sets, indexed by penalty s.
    let mut history: Vec<WfSet> = Vec::new();

    // s = 0: only M[0] = 0, then extend.
    {
        let mut m_wf = Wavefront::new(0, 0);
        m_wf.set(0, 0);
        extend(&mut m_wf, a, b);
        let set = WfSet {
            m: m_wf,
            i: Wavefront::new(0, -1),
            d: Wavefront::new(0, -1),
        };
        if reached(&set.m, k_final, a_off_max) {
            let cigar = traceback(&history, &set, &pen, k_final);
            return finish(0, m, n, sc, cigar);
        }
        history.push(set);
    }

    // A generous upper bound on the optimal penalty: the cost of aligning
    // everything as gaps. We grow up to (and including) this value.
    let max_pen = (m_i + n_i) * (pen.x + pen.o + pen.e) + pen.o + pen.e;

    let mut s = 1i32;
    loop {
        if s > max_pen {
            // Unreachable for valid positive penalties, but keep the loop total.
            return Err(SeqError::NumericalInstability(
                "WFA failed to reach the alignment endpoint within the penalty bound".into(),
            ));
        }
        let set = compute_next(&history, s, &pen, k_final, a, b);
        if reached(&set.m, k_final, a_off_max) {
            let cigar = traceback(&history, &set, &pen, k_final);
            return finish(s, m, n, sc, cigar);
        }
        history.push(set);
        s += 1;
    }
}

/// Advance the `M` wavefront along every diagonal while characters match.
fn extend(m_wf: &mut Wavefront, a: &[u8], b: &[u8]) {
    let m = a.len() as i32;
    let n = b.len() as i32;
    for k in m_wf.lo..=m_wf.hi {
        let mut off = m_wf.get(k);
        if off == NIL {
            continue;
        }
        // offset == i, j = i - k.
        loop {
            let i = off;
            let j = off - k;
            if i < m && j >= 0 && j < n && a[i as usize] == b[j as usize] {
                off += 1;
            } else {
                break;
            }
        }
        m_wf.set(k, off);
    }
}

/// Has the `M` wavefront reached the bottom-right corner?
#[inline]
fn reached(m_wf: &Wavefront, k_final: i32, a_off_max: i32) -> bool {
    m_wf.get(k_final) >= a_off_max
}

/// Compute the wavefront set at penalty `s` from the recorded history, then
/// extend its `M` component.
fn compute_next(
    history: &[WfSet],
    s: i32,
    pen: &WfaPenalties,
    k_final: i32,
    a: &[u8],
    b: &[u8],
) -> WfSet {
    // Predecessor penalties.
    let s_x = s - pen.x; // mismatch
    let s_o_e = s - pen.o - pen.e; // gap open (+ first extend)
    let s_e = s - pen.e; // gap extend

    // Diagonal range of the new wavefront: union of predecessor ranges, grown
    // by one on each side to allow opening fresh gaps, and always covering the
    // final diagonal so the endpoint can be detected.
    let mut lo = k_final;
    let mut hi = k_final;
    for &(sp, grow) in &[(s_x, 0i32), (s_o_e, 1), (s_e, 1)] {
        if sp >= 0 {
            if let Some(set) = history.get(sp as usize) {
                lo = lo.min(set.m.lo - grow);
                hi = hi.max(set.m.hi + grow);
                lo = lo.min(set.i.lo - grow);
                hi = hi.max(set.i.hi + grow);
                lo = lo.min(set.d.lo - grow);
                hi = hi.max(set.d.hi + grow);
            }
        }
    }

    let mut i_wf = Wavefront::new(lo, hi);
    let mut d_wf = Wavefront::new(lo, hi);
    let mut m_wf = Wavefront::new(lo, hi);

    let m_open = history.get_at(s_o_e, |set| &set.m);
    let i_ext = history.get_at(s_e, |set| &set.i);
    let d_ext = history.get_at(s_e, |set| &set.d);
    let m_mis = history.get_at(s_x, |set| &set.m);

    for k in lo..=hi {
        // I[k]: gap in a (consumes b → j+1, i unchanged). offset == i unchanged.
        // Predecessor lives on diagonal k+1 (since k = i - j, j+1 ⇒ k-1; thus a
        // cell on diagonal k is reached from a cell on diagonal k+1).
        let i_from_open = opt_get(m_open, k + 1);
        let i_from_ext = opt_get(i_ext, k + 1);
        let i_val = max2(i_from_open, i_from_ext);
        i_wf.set(k, i_val);

        // D[k]: gap in b (consumes a → i+1, j unchanged). offset == i, so +1.
        // Predecessor lives on diagonal k-1.
        let d_open = opt_get(m_open, k - 1);
        let d_ext_v = opt_get(d_ext, k - 1);
        let d_pred = max2(d_open, d_ext_v);
        let d_val = if d_pred == NIL { NIL } else { d_pred + 1 };
        d_wf.set(k, d_val);

        // M[k]: mismatch (i+1, j+1, offset+1) OR fold in I[k] / D[k].
        let m_sub = {
            let v = opt_get(m_mis, k);
            if v == NIL { NIL } else { v + 1 }
        };
        let m_val = max3(m_sub, i_val, d_val);
        m_wf.set(k, m_val);
    }

    extend(&mut m_wf, a, b);
    WfSet {
        m: m_wf,
        i: i_wf,
        d: d_wf,
    }
}

/// `max` of two offsets, [`NIL`]-aware.
#[inline]
fn max2(a: i32, b: i32) -> i32 {
    if a == NIL {
        b
    } else if b == NIL {
        a
    } else {
        a.max(b)
    }
}

/// `max` of three offsets, [`NIL`]-aware.
#[inline]
fn max3(a: i32, b: i32, c: i32) -> i32 {
    max2(max2(a, b), c)
}

/// Fetch the offset on diagonal `k` from an optional wavefront reference.
#[inline]
fn opt_get(wf: Option<&Wavefront>, k: i32) -> i32 {
    match wf {
        Some(w) => w.get(k),
        None => NIL,
    }
}

/// Small helper trait letting us index the history with a (possibly negative)
/// penalty and project to one wavefront component.
trait HistoryExt {
    fn get_at<'a, F>(&'a self, s: i32, f: F) -> Option<&'a Wavefront>
    where
        F: Fn(&'a WfSet) -> &'a Wavefront;
}

impl HistoryExt for [WfSet] {
    #[inline]
    fn get_at<'a, F>(&'a self, s: i32, f: F) -> Option<&'a Wavefront>
    where
        F: Fn(&'a WfSet) -> &'a Wavefront,
    {
        if s < 0 {
            None
        } else {
            self.get(s as usize).map(f)
        }
    }
}

/// Convert a raw penalty to a [`WfaAlignment`].
fn finish(
    penalty: i32,
    m: usize,
    n: usize,
    sc: &GotohScoring,
    cigar: Vec<WfaOp>,
) -> SeqResult<WfaAlignment> {
    let score = ((m as i32 + n as i32) * sc.match_score - penalty) / 2;
    Ok(WfaAlignment {
        score,
        penalty,
        cigar,
    })
}

/// Which wavefront component a traceback cursor currently sits in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Comp {
    M,
    I,
    D,
}

/// What explains an `M` cell `(s, k, off)` once its trailing matches are peeled.
enum MOrigin {
    /// The cell sits at the origin `(0, 0)`; emit leading matches and stop.
    Start,
    /// The cell is reached by a run of matches down to `target_off`.
    Match { target_off: i32 },
    /// The cell is reached by a mismatch from `(prev_s, k, prev_off)`.
    Mismatch { prev_s: i32, prev_off: i32 },
    /// The cell coincides with the `I` component at the same `(s, k)`.
    FromI,
    /// The cell coincides with the `D` component at the same `(s, k)`.
    FromD,
}

/// Read-only context for walking the recorded wavefronts back to the origin.
struct Tracer<'a> {
    history: &'a [WfSet],
    final_set: &'a WfSet,
    pen: WfaPenalties,
    /// The optimal penalty, i.e. the index of `final_set`.
    s_opt: i32,
}

impl<'a> Tracer<'a> {
    fn new(history: &'a [WfSet], final_set: &'a WfSet, pen: WfaPenalties) -> Self {
        Self {
            history,
            final_set,
            pen,
            s_opt: history.len() as i32,
        }
    }

    /// Borrow the wavefront set recorded at penalty `sp`, treating the optimal
    /// penalty as the (non-recorded) `final_set`.
    fn get_set(&self, sp: i32) -> Option<&'a WfSet> {
        if sp == self.s_opt {
            Some(self.final_set)
        } else if sp >= 0 {
            self.history.get(sp as usize)
        } else {
            None
        }
    }

    /// Determine how the `M` cell at `(s, k, off)` was produced.
    fn m_origin(&self, s: i32, k: i32, off: i32) -> MOrigin {
        let s_x = s - self.pen.x;
        let mis_pred = self.get_set(s_x).map(|set| set.m.get(k)).unwrap_or(NIL);
        let mis_bare = if mis_pred == NIL { NIL } else { mis_pred + 1 };
        let i_here = self.get_set(s).map(|set| set.i.get(k)).unwrap_or(NIL);
        let d_here = self.get_set(s).map(|set| set.d.get(k)).unwrap_or(NIL);

        // Pick the largest bare offset not exceeding `off`; the gap up to `off`
        // is the matched run that `extend` appended. Any predecessor whose bare
        // offset equals the true value is a valid traceback choice.
        let mut best_bare = NIL;
        let mut kind = 0u8; // 1=mismatch 2=I 3=D
        for (cand, kd) in [(mis_bare, 1u8), (i_here, 2), (d_here, 3)] {
            if cand != NIL && cand <= off && cand > best_bare {
                best_bare = cand;
                kind = kd;
            }
        }

        if best_bare == NIL {
            return MOrigin::Start;
        }
        if best_bare < off {
            return MOrigin::Match {
                target_off: best_bare,
            };
        }
        match kind {
            1 => MOrigin::Mismatch {
                prev_s: s_x,
                prev_off: mis_pred,
            },
            2 => MOrigin::FromI,
            _ => MOrigin::FromD,
        }
    }

    /// Walk back from the terminal cell on diagonal `k_final`, emitting ops in
    /// reverse, then reverse to left-to-right order.
    fn run(&self, k_final: i32) -> Vec<WfaOp> {
        let mut ops: Vec<WfaOp> = Vec::new();
        let mut s = self.s_opt;
        let mut k = k_final;
        let mut comp = Comp::M;
        let mut off = self.final_set.m.get(k);

        loop {
            match comp {
                Comp::M => match self.m_origin(s, k, off) {
                    MOrigin::Start => {
                        // Leading matches down to the origin (0, 0).
                        for _ in 0..off.max(0) {
                            ops.push(WfaOp::Match);
                        }
                        break;
                    }
                    MOrigin::Match { target_off } => {
                        let mut cur = off;
                        while cur > target_off {
                            ops.push(WfaOp::Match);
                            cur -= 1;
                        }
                        off = target_off;
                    }
                    MOrigin::Mismatch { prev_s, prev_off } => {
                        ops.push(WfaOp::Mismatch);
                        s = prev_s;
                        off = prev_off;
                    }
                    MOrigin::FromI => {
                        comp = Comp::I;
                        if let Some(set) = self.get_set(s) {
                            off = set.i.get(k);
                        }
                    }
                    MOrigin::FromD => {
                        comp = Comp::D;
                        if let Some(set) = self.get_set(s) {
                            off = set.d.get(k);
                        }
                    }
                },
                Comp::I => {
                    // I[k] ← M[k+1] @ s-o-e (open) or I[k+1] @ s-e (extend);
                    // offset is preserved. Emit one Ins, move to diagonal k+1.
                    ops.push(WfaOp::Ins);
                    let s_o_e = s - self.pen.o - self.pen.e;
                    let s_e = s - self.pen.e;
                    let open = self
                        .get_set(s_o_e)
                        .map(|set| set.m.get(k + 1))
                        .unwrap_or(NIL);
                    let ext = self.get_set(s_e).map(|set| set.i.get(k + 1)).unwrap_or(NIL);
                    if ext != NIL && ext == off {
                        s = s_e;
                        k += 1;
                        comp = Comp::I;
                    } else if open != NIL && open == off {
                        s = s_o_e;
                        k += 1;
                        comp = Comp::M;
                    } else if open != NIL {
                        s = s_o_e;
                        k += 1;
                        comp = Comp::M;
                        off = open;
                    } else if ext != NIL {
                        s = s_e;
                        k += 1;
                        comp = Comp::I;
                        off = ext;
                    } else {
                        break;
                    }
                }
                Comp::D => {
                    // D[k] ← M[k-1] @ s-o-e (open) or D[k-1] @ s-e (extend),
                    // offset+1. Emit one Del, move to diagonal k-1, offset−1.
                    ops.push(WfaOp::Del);
                    let s_o_e = s - self.pen.o - self.pen.e;
                    let s_e = s - self.pen.e;
                    let open = self
                        .get_set(s_o_e)
                        .map(|set| set.m.get(k - 1))
                        .unwrap_or(NIL);
                    let ext = self.get_set(s_e).map(|set| set.d.get(k - 1)).unwrap_or(NIL);
                    let pred_off = off - 1;
                    if ext != NIL && ext == pred_off {
                        s = s_e;
                        k -= 1;
                        off = pred_off;
                        comp = Comp::D;
                    } else if open != NIL && open == pred_off {
                        s = s_o_e;
                        k -= 1;
                        off = pred_off;
                        comp = Comp::M;
                    } else if open != NIL {
                        s = s_o_e;
                        k -= 1;
                        off = open;
                        comp = Comp::M;
                    } else if ext != NIL {
                        s = s_e;
                        k -= 1;
                        off = ext;
                        comp = Comp::D;
                    } else {
                        break;
                    }
                }
            }
        }

        ops.reverse();
        ops
    }
}

/// Reconstruct the optimal alignment by walking the recorded wavefronts back
/// from the terminal cell to the origin.
///
/// `final_set` is the freshly-computed wavefront set at the optimal penalty
/// `s = history.len()` (it is *not* part of `history`).
fn traceback(history: &[WfSet], final_set: &WfSet, pen: &WfaPenalties, k_final: i32) -> Vec<WfaOp> {
    Tracer::new(history, final_set, *pen).run(k_final)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::gotoh::gotoh_align;

    fn default_sc() -> GotohScoring {
        GotohScoring::default()
    }

    fn custom_sc() -> GotohScoring {
        GotohScoring {
            match_score: 3,
            mismatch: -2,
            gap_open: -6,
            gap_extend: -2,
        }
    }

    /// Independent re-scorer: walk a CIGAR applying Gotoh's scoring rules.
    fn score_cigar(a: &[u8], b: &[u8], cigar: &[WfaOp], sc: &GotohScoring) -> i32 {
        let mut score = 0i32;
        let mut i = 0usize;
        let mut j = 0usize;
        // Track whether the previous op was the same kind of gap (for affine).
        let mut prev: Option<WfaOp> = None;
        for &op in cigar {
            match op {
                WfaOp::Match => {
                    score += sc.match_score;
                    i += 1;
                    j += 1;
                }
                WfaOp::Mismatch => {
                    score += sc.mismatch;
                    i += 1;
                    j += 1;
                }
                WfaOp::Ins => {
                    // gap in a, consumes b.
                    if prev == Some(WfaOp::Ins) {
                        score += sc.gap_extend;
                    } else {
                        score += sc.gap_open;
                    }
                    j += 1;
                }
                WfaOp::Del => {
                    // gap in b, consumes a.
                    if prev == Some(WfaOp::Del) {
                        score += sc.gap_extend;
                    } else {
                        score += sc.gap_open;
                    }
                    i += 1;
                }
            }
            prev = Some(op);
        }
        assert_eq!(i, a.len(), "cigar must consume all of a");
        assert_eq!(j, b.len(), "cigar must consume all of b");
        score
    }

    fn check_consumption(a: &[u8], b: &[u8], cigar: &[WfaOp]) {
        let consumes_a = cigar
            .iter()
            .filter(|o| matches!(o, WfaOp::Match | WfaOp::Mismatch | WfaOp::Del))
            .count();
        let consumes_b = cigar
            .iter()
            .filter(|o| matches!(o, WfaOp::Match | WfaOp::Mismatch | WfaOp::Ins))
            .count();
        assert_eq!(consumes_a, a.len(), "Match+Mismatch+Del must consume a");
        assert_eq!(consumes_b, b.len(), "Match+Mismatch+Ins must consume b");
    }

    // (a) CENTRAL cross-check: WFA converted score == Gotoh score.
    #[test]
    fn central_cross_check_matches_gotoh() {
        let pairs: &[(&[u8], &[u8])] = &[
            (b"GATTACA", b"GCATGCU"),
            (b"ACGTACGT", b"ACGTTCGT"),
            (b"AAAA", b"AAAAGGGGAAAA"),
            (b"ACGT", b"TGCA"),
            (b"AGGGCT", b"AGGCT"),
            (b"HELLOWORLD", b"HELOWRLD"),
        ];
        for sc in [default_sc(), custom_sc()] {
            for &(a, b) in pairs {
                let w = wfa_align(a, b, &sc).expect("wfa ok");
                let g = gotoh_align(a, b, &sc).expect("gotoh ok");
                assert_eq!(
                    w.score,
                    g.score,
                    "score mismatch on {:?} vs {:?} with {:?}",
                    std::str::from_utf8(a),
                    std::str::from_utf8(b),
                    sc
                );
                // The CIGAR must itself reproduce the score.
                check_consumption(a, b, &w.cigar);
                assert_eq!(
                    score_cigar(a, b, &w.cigar, &sc),
                    w.score,
                    "cigar re-score mismatch on {:?} vs {:?}",
                    std::str::from_utf8(a),
                    std::str::from_utf8(b),
                );
            }
        }
    }

    // (b) identical sequences.
    #[test]
    fn identical_sequences() {
        let a = b"ACGTACGT";
        let sc = default_sc();
        let w = wfa_align(a, a, &sc).expect("ok");
        assert_eq!(w.penalty, 0);
        assert_eq!(w.score, sc.match_score * a.len() as i32);
        assert!(w.cigar.iter().all(|o| *o == WfaOp::Match));
        assert_eq!(w.cigar.len(), a.len());
    }

    // (c) traceback validity (consumption + re-score) on a tricky pair.
    #[test]
    fn traceback_validity() {
        let sc = default_sc();
        let cases: &[(&[u8], &[u8])] = &[
            (b"GATTACA", b"GCATGCU"),
            (b"ACGTACGTACGT", b"ACGTTTACGT"),
            (b"BANANA", b"ANANAS"),
        ];
        for &(a, b) in cases {
            let w = wfa_align(a, b, &sc).expect("ok");
            check_consumption(a, b, &w.cigar);
            assert_eq!(score_cigar(a, b, &w.cigar, &sc), w.score);
            let g = gotoh_align(a, b, &sc).expect("ok");
            assert_eq!(w.score, g.score);
        }
    }

    // (d) affine: one contiguous length-4 gap stays a single run.
    #[test]
    fn affine_single_long_gap() {
        let sc = default_sc();
        let a = b"ACGTACGT";
        // Insert 4 contiguous characters into the middle of `a` to make `b`.
        let b = b"ACGTGGGGACGT";
        let w = wfa_align(a, b, &sc).expect("ok");
        let g = gotoh_align(a, b, &sc).expect("ok");
        assert_eq!(w.score, g.score);
        // There must be exactly one Ins run of length 4 and no Del.
        let ins = w.cigar.iter().filter(|o| **o == WfaOp::Ins).count();
        let del = w.cigar.iter().filter(|o| **o == WfaOp::Del).count();
        assert_eq!(ins, 4, "expected 4 inserted symbols, cigar = {:?}", w.cigar);
        assert_eq!(del, 0);
        // And they must be contiguous (exactly one maximal Ins run).
        let runs = count_runs(&w.cigar, WfaOp::Ins);
        assert_eq!(runs, 1, "Ins must form a single run, cigar = {:?}", w.cigar);
    }

    fn count_runs(cigar: &[WfaOp], op: WfaOp) -> usize {
        let mut runs = 0;
        let mut in_run = false;
        for &c in cigar {
            if c == op {
                if !in_run {
                    runs += 1;
                    in_run = true;
                }
            } else {
                in_run = false;
            }
        }
        runs
    }

    // (e) single mismatch.
    #[test]
    fn single_mismatch_cost() {
        let sc = default_sc();
        let a = b"ACGTACGT";
        let b = b"ACGTTCGT"; // differs at index 4 (A vs T).
        let w = wfa_align(a, b, &sc).expect("ok");
        let g = gotoh_align(a, b, &sc).expect("ok");
        assert_eq!(w.score, g.score);
        let len = a.len() as i32;
        assert_eq!(w.score, sc.match_score * (len - 1) + sc.mismatch);
        // Penalty is exactly one mismatch unit x = 2*(M - mis).
        assert_eq!(w.penalty, 2 * (sc.match_score - sc.mismatch));
    }

    // (f) empty-sequence handling mirrors gotoh.
    #[test]
    fn empty_sequence_errors() {
        let sc = default_sc();
        assert!(matches!(
            wfa_align(b"", b"ACGT", &sc),
            Err(SeqError::EmptyInput)
        ));
        assert!(matches!(
            wfa_align(b"ACGT", b"", &sc),
            Err(SeqError::EmptyInput)
        ));
        // Confirm gotoh errors the same way.
        assert!(matches!(
            gotoh_align(b"", b"ACGT", &sc),
            Err(SeqError::EmptyInput)
        ));
        assert!(matches!(
            gotoh_align(b"ACGT", b"", &sc),
            Err(SeqError::EmptyInput)
        ));
    }

    // (g) match extension across a long identical run.
    #[test]
    fn long_match_extension() {
        let sc = default_sc();
        let prefix = vec![b'A'; 50];
        let mut a = prefix.clone();
        a.extend_from_slice(b"CGTACG");
        let mut b = prefix.clone();
        b.extend_from_slice(b"CTTACG"); // diverges within the suffix.
        let w = wfa_align(&a, &b, &sc).expect("ok");
        let g = gotoh_align(&a, &b, &sc).expect("ok");
        assert_eq!(w.score, g.score);
        assert!(w.penalty > 0);
        check_consumption(&a, &b, &w.cigar);
        assert_eq!(score_cigar(&a, &b, &w.cigar, &sc), w.score);
    }

    // Degenerate scoring → InvalidConfiguration.
    #[test]
    fn degenerate_scoring_rejected() {
        // match_score <= mismatch ⇒ x <= 0.
        let bad = GotohScoring {
            match_score: 1,
            mismatch: 1,
            gap_open: -5,
            gap_extend: -1,
        };
        assert!(matches!(
            wfa_align(b"AC", b"AG", &bad),
            Err(SeqError::InvalidConfiguration(_))
        ));
        // gap_extend < gap_open ⇒ o < 0.
        let bad_open = GotohScoring {
            match_score: 2,
            mismatch: -1,
            gap_open: -1,
            gap_extend: -5,
        };
        assert!(matches!(
            wfa_align(b"AC", b"AG", &bad_open),
            Err(SeqError::InvalidConfiguration(_))
        ));
    }

    // Heavy randomized cross-check: the converted WFA score must equal Gotoh on
    // hundreds of random pairs across several valid scoring schemes, and the
    // reconstructed CIGAR must independently reproduce that score.
    #[test]
    fn randomized_cross_check_matches_gotoh() {
        use crate::handle::LcgRng;

        let alphabet = b"ACGT";
        // A handful of valid (positive-penalty) affine scoring schemes.
        let schemes = [
            GotohScoring::default(),
            GotohScoring {
                match_score: 3,
                mismatch: -2,
                gap_open: -6,
                gap_extend: -2,
            },
            GotohScoring {
                match_score: 1,
                mismatch: -1,
                gap_open: -2,
                gap_extend: -1,
            },
            GotohScoring {
                match_score: 4,
                mismatch: -3,
                gap_open: -8,
                gap_extend: -1,
            },
            GotohScoring {
                match_score: 2,
                mismatch: 0,
                gap_open: -4,
                gap_extend: -1,
            },
        ];

        let mut rng = LcgRng::new(0x5EED_1234_ABCD);
        for sc in schemes {
            // Sanity: every scheme must derive valid positive penalties.
            assert!(WfaPenalties::from_gotoh(&sc).is_ok());
            for _ in 0..120 {
                let la = 1 + rng.next_usize(14);
                let lb = 1 + rng.next_usize(14);
                let a: Vec<u8> = (0..la).map(|_| alphabet[rng.next_usize(4)]).collect();
                let b: Vec<u8> = (0..lb).map(|_| alphabet[rng.next_usize(4)]).collect();
                let w = wfa_align(&a, &b, &sc).expect("wfa ok");
                let g = gotoh_align(&a, &b, &sc).expect("gotoh ok");
                assert_eq!(
                    w.score,
                    g.score,
                    "score mismatch: a={:?} b={:?} sc={:?} (wfa={} gotoh={})",
                    std::str::from_utf8(&a),
                    std::str::from_utf8(&b),
                    sc,
                    w.score,
                    g.score,
                );
                check_consumption(&a, &b, &w.cigar);
                assert_eq!(
                    score_cigar(&a, &b, &w.cigar, &sc),
                    w.score,
                    "cigar re-score mismatch: a={:?} b={:?}",
                    std::str::from_utf8(&a),
                    std::str::from_utf8(&b),
                );
            }
        }
    }

    // Asymmetric long gaps in both directions (deletion-heavy and
    // insertion-heavy) must still match Gotoh exactly.
    #[test]
    fn asymmetric_gaps_match_gotoh() {
        let sc = custom_sc();
        let cases: &[(&[u8], &[u8])] = &[
            (b"AAAAGGGGAAAA", b"AAAA"),      // deletion-heavy
            (b"AAAA", b"AAAAGGGGAAAA"),      // insertion-heavy
            (b"ACGTACGTACGT", b"ACGT"),      // big deletion
            (b"ACGT", b"ACGTACGTACGT"),      // big insertion
            (b"TTTTACGTTTTT", b"ACGT"),      // flanking deletions
            (b"GATTACAGATTACA", b"GATTACA"), // tandem deletion
        ];
        for &(a, b) in cases {
            let w = wfa_align(a, b, &sc).expect("ok");
            let g = gotoh_align(a, b, &sc).expect("ok");
            assert_eq!(
                w.score,
                g.score,
                "mismatch on {:?} vs {:?}",
                std::str::from_utf8(a),
                std::str::from_utf8(b),
            );
            check_consumption(a, b, &w.cigar);
            assert_eq!(score_cigar(a, b, &w.cigar, &sc), w.score);
        }
    }
}
