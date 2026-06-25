//! # Grammar / Regex Finite-State-Machine Constrained Decoding
//!
//! General guided decoding over an arbitrary **deterministic finite automaton**
//! (DFA) on the *byte* alphabet, in the spirit of Outlines (Willard & Louf,
//! 2023) and `lm-format-enforcer`. Whereas
//! [`crate::sampling::json_constrained`] hard-codes the JSON grammar, this
//! module accepts a caller-supplied DFA and masks, at every decode step,
//! **exactly** the vocabulary tokens whose byte expansion cannot extend the
//! current automaton state toward an accepting run.
//!
//! ## Model
//!
//! * A token is a short byte string (its detokenised surface form).
//! * The constraint is a DFA: `transition[state][byte] -> next_state`, with a
//!   set of accepting states. A `None` transition means the byte is rejected
//!   in that state (a dead end).
//! * A token id `t` is **allowed** in DFA state `s` iff feeding the bytes of
//!   token `t` one at a time from `s` never hits a dead end. The resulting
//!   state becomes the new constraint state once that token is committed.
//!
//! Because the per-token feasibility test is a pure DFA walk, the mask is exact:
//! it lets through precisely the tokens that keep at least one accepting
//! continuation reachable from the next state (a state from which no accepting
//! state is reachable is itself treated as dead via [`Dfa::is_live`]).
//!
//! ## Building a DFA
//!
//! [`Dfa::from_literal`] builds the trivial automaton accepting one fixed
//! string; [`Dfa::char_star`] accepts any sequence over a byte set (a Kleene
//! star); [`Dfa::builder`] exposes manual construction for hand-written or
//! compiled grammars. A regex compiler is intentionally *out of scope* — the
//! DFA is the stable interface every front-end targets.

use crate::error::{InferError, InferResult};
use std::collections::{HashMap, VecDeque};

// ─── Dfa ─────────────────────────────────────────────────────────────────────

/// A deterministic finite automaton over the byte alphabet `0..=255`.
///
/// State `0` is always the start state. Transitions are stored sparsely: a
/// missing `(state, byte)` entry is a dead transition (rejection).
#[derive(Debug, Clone)]
pub struct Dfa {
    /// `transitions[state]` maps an input byte to the destination state.
    transitions: Vec<HashMap<u8, usize>>,
    /// Whether each state is accepting.
    accepting: Vec<bool>,
}

impl Dfa {
    /// Number of states.
    #[must_use]
    pub fn n_states(&self) -> usize {
        self.transitions.len()
    }

    /// Is `state` an accepting (final) state?
    #[must_use]
    pub fn is_accepting(&self, state: usize) -> bool {
        self.accepting.get(state).copied().unwrap_or(false)
    }

    /// Step a single byte. Returns the next state, or `None` for a dead
    /// transition (the byte is not permitted from `state`).
    #[must_use]
    pub fn step(&self, state: usize, byte: u8) -> Option<usize> {
        self.transitions.get(state)?.get(&byte).copied()
    }

    /// Is any accepting state reachable from `state`?
    ///
    /// A state failing this test is a *dead* state: no continuation can ever be
    /// accepted, so committing to it is forbidden. Computed by a forward BFS.
    #[must_use]
    pub fn is_live(&self, state: usize) -> bool {
        if state >= self.transitions.len() {
            return false;
        }
        let mut seen = vec![false; self.transitions.len()];
        let mut queue = VecDeque::new();
        seen[state] = true;
        queue.push_back(state);
        while let Some(s) = queue.pop_front() {
            if self.accepting[s] {
                return true;
            }
            for &dst in self.transitions[s].values() {
                if !seen[dst] {
                    seen[dst] = true;
                    queue.push_back(dst);
                }
            }
        }
        false
    }

    /// Feed all the bytes of one token from `state`. Returns the resulting state
    /// if the whole token is accepted *and* leaves the automaton live, or `None`
    /// if any byte dead-ends or the final state is dead.
    #[must_use]
    pub fn feed_token(&self, state: usize, token_bytes: &[u8]) -> Option<usize> {
        let mut s = state;
        for &byte in token_bytes {
            s = self.step(s, byte)?;
        }
        if self.is_live(s) { Some(s) } else { None }
    }

    // ── Constructors ─────────────────────────────────────────────────────────

    /// DFA accepting exactly the byte string `s` (and nothing else).
    #[must_use]
    pub fn from_literal(s: &[u8]) -> Self {
        let n = s.len();
        let mut transitions = vec![HashMap::new(); n + 1];
        for (i, &byte) in s.iter().enumerate() {
            transitions[i].insert(byte, i + 1);
        }
        let mut accepting = vec![false; n + 1];
        accepting[n] = true;
        Self {
            transitions,
            accepting,
        }
    }

    /// DFA accepting any sequence (including empty) over the byte set `alphabet`
    /// (Kleene star). The single state is both start and accepting.
    #[must_use]
    pub fn char_star(alphabet: &[u8]) -> Self {
        let mut t = HashMap::new();
        for &byte in alphabet {
            t.insert(byte, 0_usize);
        }
        Self {
            transitions: vec![t],
            accepting: vec![true],
        }
    }

    /// Begin manual DFA construction with `n_states` states (all non-accepting,
    /// no transitions). State `0` is the start state.
    #[must_use]
    pub fn builder(n_states: usize) -> DfaBuilder {
        DfaBuilder {
            transitions: vec![HashMap::new(); n_states.max(1)],
            accepting: vec![false; n_states.max(1)],
        }
    }
}

// ─── DfaBuilder ──────────────────────────────────────────────────────────────

/// Fluent builder for hand-constructed DFAs.
#[derive(Debug, Clone)]
pub struct DfaBuilder {
    transitions: Vec<HashMap<u8, usize>>,
    accepting: Vec<bool>,
}

impl DfaBuilder {
    /// Add a transition `from --byte--> to`.
    ///
    /// # Errors
    /// * [`InferError::InvalidConfig`] if a state index is out of range.
    pub fn transition(mut self, from: usize, byte: u8, to: usize) -> InferResult<Self> {
        let n = self.transitions.len();
        if from >= n || to >= n {
            return Err(InferError::InvalidConfig(
                "DFA transition state out of range",
            ));
        }
        self.transitions[from].insert(byte, to);
        Ok(self)
    }

    /// Add the same destination for a whole set of bytes (a character class).
    ///
    /// # Errors
    /// * [`InferError::InvalidConfig`] if a state index is out of range.
    pub fn transition_class(mut self, from: usize, bytes: &[u8], to: usize) -> InferResult<Self> {
        let n = self.transitions.len();
        if from >= n || to >= n {
            return Err(InferError::InvalidConfig(
                "DFA transition state out of range",
            ));
        }
        for &byte in bytes {
            self.transitions[from].insert(byte, to);
        }
        Ok(self)
    }

    /// Mark `state` as accepting.
    ///
    /// # Errors
    /// * [`InferError::InvalidConfig`] if `state` is out of range.
    pub fn accept(mut self, state: usize) -> InferResult<Self> {
        if state >= self.accepting.len() {
            return Err(InferError::InvalidConfig("DFA accept state out of range"));
        }
        self.accepting[state] = true;
        Ok(self)
    }

    /// Finalise the DFA.
    #[must_use]
    pub fn build(self) -> Dfa {
        Dfa {
            transitions: self.transitions,
            accepting: self.accepting,
        }
    }
}

// ─── GrammarConstraint ───────────────────────────────────────────────────────

/// A live constrained-decoding session: a [`Dfa`] plus the current state.
///
/// Holds a borrowed token vocabulary (each entry is a token's surface bytes) so
/// it can compute the per-step allow-mask without recomputing it from scratch.
#[derive(Debug, Clone)]
pub struct GrammarConstraint {
    dfa: Dfa,
    state: usize,
    /// Detokenised surface form (bytes) of each vocabulary token id.
    vocab: Vec<Vec<u8>>,
}

impl GrammarConstraint {
    /// Create a constraint over `dfa` for the given `vocab` (token id → bytes).
    ///
    /// # Errors
    /// * [`InferError::InvalidConfig`] if the DFA's start state is already dead
    ///   (no accepting state is reachable), making decoding impossible.
    pub fn new(dfa: Dfa, vocab: Vec<Vec<u8>>) -> InferResult<Self> {
        if !dfa.is_live(0) {
            return Err(InferError::InvalidConfig(
                "grammar DFA start state has no reachable accepting state",
            ));
        }
        Ok(Self {
            dfa,
            state: 0,
            vocab,
        })
    }

    /// Current automaton state.
    #[must_use]
    pub fn state(&self) -> usize {
        self.state
    }

    /// Is the current state accepting (a complete utterance is permitted here)?
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.dfa.is_accepting(self.state)
    }

    /// Indices (token ids) that are legal from the current state.
    ///
    /// A token is legal iff feeding its bytes from the current state never
    /// dead-ends and leaves the automaton live.
    #[must_use]
    pub fn allowed_tokens(&self) -> Vec<usize> {
        (0..self.vocab.len())
            .filter(|&t| self.dfa.feed_token(self.state, &self.vocab[t]).is_some())
            .collect()
    }

    /// Apply the constraint to `logits` in place: every *illegal* token id is set
    /// to `NEG_INFINITY`, masking it out of sampling. Legal tokens are untouched.
    ///
    /// # Errors
    /// * [`InferError::DimensionMismatch`] if `logits.len() != vocab length`.
    /// * [`InferError::SamplingError`] if **no** token is legal (a dead end the
    ///   grammar cannot escape), so the caller can surface a hard failure rather
    ///   than sample garbage.
    pub fn mask_logits(&self, logits: &mut [f32]) -> InferResult<()> {
        if logits.len() != self.vocab.len() {
            return Err(InferError::DimensionMismatch {
                expected: self.vocab.len(),
                got: logits.len(),
            });
        }
        let mut any = false;
        for (t, logit) in logits.iter_mut().enumerate() {
            if self.dfa.feed_token(self.state, &self.vocab[t]).is_some() {
                any = true;
            } else {
                *logit = f32::NEG_INFINITY;
            }
        }
        if !any {
            return Err(InferError::SamplingError(
                "grammar constraint masked every token: no legal continuation".into(),
            ));
        }
        Ok(())
    }

    /// Commit token id `t` (the sampled token), advancing the automaton state.
    ///
    /// # Errors
    /// * [`InferError::InvalidConfig`] if `t` is out of vocabulary range.
    /// * [`InferError::SamplingError`] if `t` is *not* a legal continuation from
    ///   the current state (the caller violated the mask).
    pub fn commit(&mut self, t: usize) -> InferResult<()> {
        let bytes = self.vocab.get(t).ok_or(InferError::InvalidConfig(
            "grammar commit: token out of range",
        ))?;
        match self.dfa.feed_token(self.state, bytes) {
            Some(next) => {
                self.state = next;
                Ok(())
            }
            None => Err(InferError::SamplingError(format!(
                "grammar constraint: token {t} is not a legal continuation"
            ))),
        }
    }

    /// Reset the automaton to its start state (begin a fresh generation).
    pub fn reset(&mut self) {
        self.state = 0;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Vocab: 0="ab", 1="a", 2="b", 3="c", 4="abc".
    fn small_vocab() -> Vec<Vec<u8>> {
        vec![
            b"ab".to_vec(),
            b"a".to_vec(),
            b"b".to_vec(),
            b"c".to_vec(),
            b"abc".to_vec(),
        ]
    }

    #[test]
    fn literal_dfa_accepts_exact_string() {
        let dfa = Dfa::from_literal(b"abc");
        // walk a-b-c
        let s = dfa.step(0, b'a').expect("a ok");
        let s = dfa.step(s, b'b').expect("b ok");
        let s = dfa.step(s, b'c').expect("c ok");
        assert!(dfa.is_accepting(s));
        // wrong byte dead-ends
        assert!(dfa.step(0, b'z').is_none());
    }

    #[test]
    fn is_live_detects_dead_states() {
        // Two states, start has a transition to state 1, neither accepting.
        let dfa = Dfa::builder(2)
            .transition(0, b'x', 1)
            .expect("valid")
            .build();
        // No accepting state anywhere → start is dead.
        assert!(!dfa.is_live(0));
    }

    #[test]
    fn feed_token_rejects_dead_end() {
        let dfa = Dfa::from_literal(b"abc");
        // token "ab" leaves state 2, from which "c" still reaches accept → live.
        assert_eq!(dfa.feed_token(0, b"ab"), Some(2));
        // token "ac" dead-ends at the second byte.
        assert_eq!(dfa.feed_token(0, b"ac"), None);
        // token "abc" reaches the accepting state.
        assert_eq!(dfa.feed_token(0, b"abc"), Some(3));
    }

    #[test]
    fn allowed_tokens_exact_for_literal() {
        let dfa = Dfa::from_literal(b"abc");
        let g = GrammarConstraint::new(dfa, small_vocab()).expect("live start");
        // From the start of "abc", legal tokens are those whose bytes are a
        // prefix path that stays live: "ab"(0), "a"(1), "abc"(4).
        // "b"(2) and "c"(3) dead-end immediately.
        let allowed = g.allowed_tokens();
        assert_eq!(allowed, vec![0, 1, 4]);
    }

    #[test]
    fn mask_logits_blocks_exactly_illegal() {
        let dfa = Dfa::from_literal(b"abc");
        let g = GrammarConstraint::new(dfa, small_vocab()).expect("live start");
        let mut logits = vec![1.0_f32; 5];
        g.mask_logits(&mut logits)
            .expect("at least one legal token");
        // legal: 0,1,4 untouched; illegal: 2,3 → -inf
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 1.0);
        assert_eq!(logits[4], 1.0);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn commit_advances_state_and_completes() {
        let dfa = Dfa::from_literal(b"abc");
        let mut g = GrammarConstraint::new(dfa, small_vocab()).expect("live");
        assert!(!g.is_complete());
        g.commit(0).expect("commit 'ab' legal"); // now at state 2
        assert!(!g.is_complete());
        g.commit(3).expect("commit 'c' legal"); // 'c' from state 2 → accept
        assert!(g.is_complete(), "abc fully formed → accepting");
    }

    #[test]
    fn commit_illegal_token_errors() {
        let dfa = Dfa::from_literal(b"abc");
        let mut g = GrammarConstraint::new(dfa, small_vocab()).expect("live");
        // 'c'(3) is illegal from the start of "abc".
        assert!(matches!(g.commit(3), Err(InferError::SamplingError(_))));
    }

    #[test]
    fn char_star_accepts_any_alphabet_seq() {
        let dfa = Dfa::char_star(b"ab");
        let g = GrammarConstraint::new(dfa, small_vocab()).expect("live");
        // Every token consisting solely of {a,b} is legal; "c"(3),"abc"(4) are not.
        let allowed = g.allowed_tokens();
        assert_eq!(allowed, vec![0, 1, 2]);
        assert!(g.is_complete(), "Kleene star start is accepting");
    }

    #[test]
    fn dead_start_rejected_at_construction() {
        let dfa = Dfa::builder(1).build(); // no accepting state
        let r = GrammarConstraint::new(dfa, small_vocab());
        assert!(matches!(r, Err(InferError::InvalidConfig(_))));
    }

    #[test]
    fn mask_all_illegal_is_error() {
        // DFA accepting only "z"; vocab has no token producing 'z' as a legal path.
        let dfa = Dfa::from_literal(b"z");
        let g = GrammarConstraint::new(dfa, small_vocab()).expect("live start");
        let mut logits = vec![0.0_f32; 5];
        // none of a/ab/b/c/abc start with 'z' → all masked → error.
        assert!(matches!(
            g.mask_logits(&mut logits),
            Err(InferError::SamplingError(_))
        ));
    }

    #[test]
    fn dimension_mismatch_on_mask() {
        let dfa = Dfa::from_literal(b"abc");
        let g = GrammarConstraint::new(dfa, small_vocab()).expect("live");
        let mut logits = vec![0.0_f32; 3]; // wrong length
        assert!(matches!(
            g.mask_logits(&mut logits),
            Err(InferError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn class_transition_builds_digit_run() {
        // State machine accepting one or more ASCII digits.
        let digits: Vec<u8> = (b'0'..=b'9').collect();
        let dfa = Dfa::builder(2)
            .transition_class(0, &digits, 1)
            .expect("valid 0->1")
            .transition_class(1, &digits, 1)
            .expect("valid 1->1")
            .accept(1)
            .expect("valid accept")
            .build();
        let vocab = vec![b"12".to_vec(), b"a".to_vec(), b"3".to_vec()];
        let g = GrammarConstraint::new(dfa, vocab).expect("live");
        // "12"(0) and "3"(2) legal; "a"(1) illegal.
        assert_eq!(g.allowed_tokens(), vec![0, 2]);
    }

    #[test]
    fn reset_returns_to_start() {
        let dfa = Dfa::from_literal(b"abc");
        let mut g = GrammarConstraint::new(dfa, small_vocab()).expect("live");
        g.commit(0).expect("commit ab");
        assert_eq!(g.state(), 2);
        g.reset();
        assert_eq!(g.state(), 0);
    }
}
