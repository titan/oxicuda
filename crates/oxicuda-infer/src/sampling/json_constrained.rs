//! # JSON-Constrained Sampling
//!
//! Constrained / grammar-guided decoding for structured output.  Implements a
//! **character-level pushdown automaton** that validates JSON structure and
//! masks the next-token logits so that only structurally-valid continuations
//! are allowed.
//!
//! ## Concept
//!
//! When forcing a language model to emit syntactically valid JSON, we drive a
//! parser alongside generation.  At every decode step the parser exposes the
//! set of characters that may legally come next.  Any vocabulary entry whose
//! leading character is not in that set has its logit set to
//! `f32::NEG_INFINITY`, so the sampler can never select a token that would
//! break the grammar.
//!
//! ## Automaton
//!
//! A JSON value is recognised by a stack of *contexts* (`JsonContext`) — one
//! pushed for every open `{` (object) or `[` (array) — combined with a
//! fine-grained per-position `JsonState`.  The grammar follows RFC 8259:
//!
//! * a **value** is one of `{` `[` `"` a number (`-`/digit) or a literal
//!   (`true` / `false` / `null`);
//! * inside a **string** any character is accepted until an unescaped `"`,
//!   with `\` introducing an escape;
//! * a **number** is `-? int (. frac)? ([eE] [+-]? exp)?`;
//! * inside an **object**, a key string is followed by `:` then a value, and
//!   members are separated by `,`;
//! * inside an **array**, values are separated by `,`;
//! * insignificant whitespace (space, tab, newline, carriage return) is
//!   permitted between structural tokens.
//!
//! The automaton is *online*: [`JsonConstraint::step`] advances one character
//! at a time and returns [`InferError`] the moment a character violates the
//! grammar, while [`JsonConstraint::is_complete`] reports when a full top-level
//! value has been consumed and the context stack is empty.

use crate::error::{InferError, InferResult};

// ─── JsonToken ─────────────────────────────────────────────────────────────────

/// Structural token classes recognised by the JSON automaton.
///
/// These name the syntactic landmarks of the grammar; they are returned by
/// [`JsonConstraint::last_token`] for inspection / debugging and used
/// internally to reason about transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonToken {
    /// `{` — beginning of an object.
    ObjectStart,
    /// `}` — end of an object.
    ObjectEnd,
    /// `[` — beginning of an array.
    ArrayStart,
    /// `]` — end of an array.
    ArrayEnd,
    /// `:` — separates a key from its value inside an object.
    Colon,
    /// `,` — separates members of an object or elements of an array.
    Comma,
    /// `"` — beginning of a string literal.
    StringStart,
    /// `"` — end of a string literal.
    StringEnd,
    /// A character consumed inside a string literal.
    StringChar,
    /// A character consumed inside a numeric literal.
    NumberChar,
    /// A character consumed inside a `true` / `false` / `null` literal.
    LiteralChar,
    /// Insignificant whitespace between tokens.
    Whitespace,
}

// ─── JsonContext ───────────────────────────────────────────────────────────────

/// A single open structural context on the automaton stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonContext {
    /// Inside an object — between `{` and its matching `}`.
    Object,
    /// Inside an array — between `[` and its matching `]`.
    Array,
}

// ─── JsonState ─────────────────────────────────────────────────────────────────

/// Fine-grained per-position state of the automaton.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonState {
    /// Expecting the start of a JSON value (top level, after `[`, after `:`,
    /// or after a `,` inside an array).
    ExpectValue,
    /// Expecting an object key string (`"`) or the closing `}` of an object.
    /// `after_comma` records whether a `,` already committed to a further
    /// member, in which case `}` is *not* permitted.
    ExpectKeyOrEnd { after_comma: bool },
    /// A key string just finished; expecting the `:` separator.
    ExpectColon,
    /// Inside a string literal.  `is_key` distinguishes object keys from
    /// string values; `escape` is set immediately after a `\`.
    InString { is_key: bool, escape: bool },
    /// Inside a numeric literal.  The phase tracks which grammar productions
    /// are still reachable so a trailing `.`/`e` is rejected correctly.
    InNumber { phase: NumberPhase },
    /// Inside a `true` / `false` / `null` literal; `kind` selects the word and
    /// `pos` is the number of characters already matched.
    InLiteral { kind: LiteralKind, pos: usize },
    /// A complete value has just been consumed; expecting a separator
    /// (`,`), a closing delimiter (`}` / `]`), or end-of-input at top level.
    AfterValue,
    /// A complete top-level value has been consumed and the stack is empty.
    Done,
}

/// Sub-state of numeric-literal recognition (RFC 8259 number grammar).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumberPhase {
    /// Just consumed the leading `-`; a digit (or `0`) must follow.
    AfterSign,
    /// Consumed a leading `0`; only `.`, `e`/`E`, or a terminator may follow
    /// (no further integer digits — JSON forbids leading zeros).
    AfterLeadingZero,
    /// Inside the integer part after a non-zero leading digit.
    IntDigits,
    /// Consumed the `.`; at least one fraction digit must follow.
    AfterDot,
    /// Inside the fractional digits.
    FracDigits,
    /// Consumed `e`/`E`; an optional sign or a digit must follow.
    AfterExp,
    /// Consumed the exponent sign; a digit must follow.
    AfterExpSign,
    /// Inside the exponent digits.
    ExpDigits,
}

/// Which literal word is being matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiteralKind {
    /// `true`
    True,
    /// `false`
    False,
    /// `null`
    Null,
}

impl LiteralKind {
    /// The full word for this literal.
    fn word(self) -> &'static str {
        match self {
            LiteralKind::True => "true",
            LiteralKind::False => "false",
            LiteralKind::Null => "null",
        }
    }
}

// ─── JsonConstraint ────────────────────────────────────────────────────────────

/// Online JSON validator and logit mask.
///
/// Drive it one character at a time with [`step`](JsonConstraint::step); query
/// the legal next characters with [`allowed_chars`](JsonConstraint::allowed_chars)
/// or mask a logit vector directly with
/// [`mask_logits`](JsonConstraint::mask_logits).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonConstraint {
    /// Stack of open structural contexts (object / array).
    stack: Vec<JsonContext>,
    /// Current fine-grained automaton state.
    state: JsonState,
    /// Token class of the most recently consumed character (for inspection).
    last_token: Option<JsonToken>,
}

impl JsonConstraint {
    /// Create a fresh constraint expecting a single top-level JSON value.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            state: JsonState::ExpectValue,
            last_token: None,
        }
    }

    /// Token class of the most recently consumed character, if any.
    #[must_use]
    pub fn last_token(&self) -> Option<JsonToken> {
        self.last_token
    }

    /// Current nesting depth (number of open objects/arrays).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Whether a full valid JSON value has been consumed and the stack is empty.
    ///
    /// A top-level number has no explicit terminator, so a bare value such as
    /// `42` or `1e5` is complete as soon as its automaton reaches a terminal
    /// numeric phase with an empty context stack (in addition to the explicit
    /// `JsonState::Done` reached by objects, arrays, strings, and literals).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        if !self.stack.is_empty() {
            return false;
        }
        match self.state {
            JsonState::Done => true,
            JsonState::InNumber { phase } => Self::number_phase_terminal(phase),
            _ => false,
        }
    }

    /// Advance the automaton by one character.
    ///
    /// # Errors
    ///
    /// Returns [`InferError::SamplingError`] when `ch` violates the JSON
    /// grammar in the current state (e.g. a structural delimiter in the wrong
    /// place, an illegal character inside a number, or any input once a
    /// top-level value is already complete).
    pub fn step(&mut self, ch: char) -> InferResult<()> {
        match self.state {
            JsonState::ExpectValue => self.step_expect_value(ch),
            JsonState::ExpectKeyOrEnd { after_comma } => {
                self.step_expect_key_or_end(ch, after_comma)
            }
            JsonState::ExpectColon => self.step_expect_colon(ch),
            JsonState::InString { is_key, escape } => self.step_in_string(ch, is_key, escape),
            JsonState::InNumber { phase } => self.step_in_number(ch, phase),
            JsonState::InLiteral { kind, pos } => self.step_in_literal(ch, kind, pos),
            JsonState::AfterValue => self.step_after_value(ch),
            JsonState::Done => Err(Self::reject(ch, "input after complete top-level value")),
        }
    }

    /// A representative set of characters that may legally come next.
    ///
    /// The set contains the structural delimiters that are valid right now plus
    /// one representative of each permitted character *class*: the digit `0`
    /// stands for any digit `0-9`, and the literal lead letters `t` / `f` / `n`
    /// stand for the `true` / `false` / `null` keywords.  When inside a string
    /// the representative `'x'` denotes "any character", and when mid-token the
    /// concrete legal continuations are listed.
    #[must_use]
    pub fn allowed_chars(&self) -> Vec<char> {
        match self.state {
            JsonState::ExpectValue => Self::value_start_chars(),
            JsonState::ExpectKeyOrEnd { after_comma } => {
                if after_comma {
                    vec!['"']
                } else {
                    vec!['"', '}']
                }
            }
            JsonState::ExpectColon => vec![':'],
            JsonState::InString { escape, .. } => {
                if escape {
                    // After a backslash only the JSON escape characters are legal.
                    vec!['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']
                } else {
                    // Any character except an unescaped control char; `"` closes,
                    // `\` escapes.  `'x'` is a representative "ordinary char".
                    vec!['x', '"', '\\']
                }
            }
            JsonState::InNumber { phase } => Self::number_next_chars(phase, &self.stack),
            JsonState::InLiteral { kind, pos } => kind
                .word()
                .chars()
                .nth(pos)
                .map_or_else(Vec::new, |c| vec![c]),
            JsonState::AfterValue => self.after_value_chars(),
            JsonState::Done => Vec::new(),
        }
    }

    /// Mask `logits` so that vocabulary entries which cannot legally continue
    /// the JSON are set to `f32::NEG_INFINITY`.
    ///
    /// A vocabulary entry is *allowed* iff its leading character is in
    /// [`allowed_chars`](JsonConstraint::allowed_chars) — interpreting the
    /// class representatives (`'0'` ⇒ any digit, `'x'` ⇒ any character while in
    /// a string) appropriately.  Empty vocabulary strings are always masked.
    ///
    /// # Errors
    ///
    /// Returns [`InferError::DimensionMismatch`] if `logits.len()` differs from
    /// `vocab.len()`.
    pub fn mask_logits(&self, logits: &mut [f32], vocab: &[String]) -> InferResult<()> {
        if logits.len() != vocab.len() {
            return Err(InferError::DimensionMismatch {
                expected: vocab.len(),
                got: logits.len(),
            });
        }
        let allowed = self.allowed_chars();
        for (logit, token) in logits.iter_mut().zip(vocab.iter()) {
            let permitted = match token.chars().next() {
                Some(first) => Self::char_allowed(first, &allowed),
                None => false,
            };
            if !permitted {
                *logit = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }

    // ─── transition helpers ─────────────────────────────────────────────────

    /// Characters that may begin a JSON value.
    fn value_start_chars() -> Vec<char> {
        vec![
            '{', '[', '"', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 't', 'f', 'n',
        ]
    }

    /// Decide whether a concrete first character `first` is permitted given the
    /// representative `allowed` set returned by [`allowed_chars`].
    fn char_allowed(first: char, allowed: &[char]) -> bool {
        for &a in allowed {
            let hit = match a {
                // `'0'` represents the whole digit class.
                '0' => first.is_ascii_digit(),
                // `'x'` (only emitted while in a string) represents any
                // ordinary character — i.e. anything that is not the closing
                // quote or the escape backslash, which are listed explicitly.
                'x' => first != '"' && first != '\\',
                other => first == other,
            };
            if hit {
                return true;
            }
        }
        false
    }

    /// Whitespace test (RFC 8259 insignificant whitespace).
    fn is_ws(ch: char) -> bool {
        matches!(ch, ' ' | '\t' | '\n' | '\r')
    }

    /// Build a rejection error for `ch` with a contextual `reason`.
    fn reject(ch: char, reason: &str) -> InferError {
        InferError::SamplingError(format!("invalid JSON: unexpected '{ch}' ({reason})"))
    }

    /// Begin a value given the leading character (already known to be a value
    /// starter or whitespace handled by the caller).
    fn begin_value(&mut self, ch: char) -> InferResult<()> {
        match ch {
            '{' => {
                self.stack.push(JsonContext::Object);
                self.state = JsonState::ExpectKeyOrEnd { after_comma: false };
                self.last_token = Some(JsonToken::ObjectStart);
                Ok(())
            }
            '[' => {
                self.stack.push(JsonContext::Array);
                self.state = JsonState::ExpectValue;
                self.last_token = Some(JsonToken::ArrayStart);
                Ok(())
            }
            '"' => {
                self.state = JsonState::InString {
                    is_key: false,
                    escape: false,
                };
                self.last_token = Some(JsonToken::StringStart);
                Ok(())
            }
            '-' => {
                self.state = JsonState::InNumber {
                    phase: NumberPhase::AfterSign,
                };
                self.last_token = Some(JsonToken::NumberChar);
                Ok(())
            }
            '0' => {
                self.state = JsonState::InNumber {
                    phase: NumberPhase::AfterLeadingZero,
                };
                self.last_token = Some(JsonToken::NumberChar);
                Ok(())
            }
            '1'..='9' => {
                self.state = JsonState::InNumber {
                    phase: NumberPhase::IntDigits,
                };
                self.last_token = Some(JsonToken::NumberChar);
                Ok(())
            }
            't' => self.begin_literal(LiteralKind::True),
            'f' => self.begin_literal(LiteralKind::False),
            'n' => self.begin_literal(LiteralKind::Null),
            _ => Err(Self::reject(ch, "expected a JSON value")),
        }
    }

    /// Begin a literal whose first character `t`/`f`/`n` was just consumed.
    fn begin_literal(&mut self, kind: LiteralKind) -> InferResult<()> {
        // The first character is already matched, so advance to position 1.
        let word = kind.word();
        if word.len() == 1 {
            // Not reachable for the three JSON literals, but keep total.
            self.complete_value();
        } else {
            self.state = JsonState::InLiteral { kind, pos: 1 };
        }
        self.last_token = Some(JsonToken::LiteralChar);
        Ok(())
    }

    /// Transition after a *complete* value: either pop into the enclosing
    /// context's "after value" position or, at top level, mark Done.
    fn complete_value(&mut self) {
        if self.stack.is_empty() {
            self.state = JsonState::Done;
        } else {
            self.state = JsonState::AfterValue;
        }
    }

    fn step_expect_value(&mut self, ch: char) -> InferResult<()> {
        if Self::is_ws(ch) {
            self.last_token = Some(JsonToken::Whitespace);
            return Ok(());
        }
        // Special case: an empty array `[]` — `]` allowed right after `[`.
        if ch == ']' {
            if matches!(self.stack.last(), Some(JsonContext::Array)) {
                self.stack.pop();
                self.last_token = Some(JsonToken::ArrayEnd);
                self.complete_value();
                return Ok(());
            }
            return Err(Self::reject(ch, "']' without a matching '['"));
        }
        self.begin_value(ch)
    }

    fn step_expect_key_or_end(&mut self, ch: char, after_comma: bool) -> InferResult<()> {
        if Self::is_ws(ch) {
            self.last_token = Some(JsonToken::Whitespace);
            return Ok(());
        }
        match ch {
            '"' => {
                self.state = JsonState::InString {
                    is_key: true,
                    escape: false,
                };
                self.last_token = Some(JsonToken::StringStart);
                Ok(())
            }
            '}' if !after_comma => {
                // Close an (possibly empty) object.
                if matches!(self.stack.last(), Some(JsonContext::Object)) {
                    self.stack.pop();
                    self.last_token = Some(JsonToken::ObjectEnd);
                    self.complete_value();
                    Ok(())
                } else {
                    Err(Self::reject(ch, "'}' without a matching '{'"))
                }
            }
            _ => Err(Self::reject(ch, "expected an object key string")),
        }
    }

    fn step_expect_colon(&mut self, ch: char) -> InferResult<()> {
        if Self::is_ws(ch) {
            self.last_token = Some(JsonToken::Whitespace);
            return Ok(());
        }
        if ch == ':' {
            self.state = JsonState::ExpectValue;
            self.last_token = Some(JsonToken::Colon);
            Ok(())
        } else {
            Err(Self::reject(ch, "expected ':' after object key"))
        }
    }

    fn step_in_string(&mut self, ch: char, is_key: bool, escape: bool) -> InferResult<()> {
        if escape {
            // Only the standard JSON escapes are valid after a backslash.
            // `\u` introduces a four-hex-digit sequence; we accept it
            // permissively (the four digits are validated as ordinary chars).
            match ch {
                '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u' => {
                    self.state = JsonState::InString {
                        is_key,
                        escape: false,
                    };
                    self.last_token = Some(JsonToken::StringChar);
                    Ok(())
                }
                _ => Err(Self::reject(ch, "invalid escape sequence in string")),
            }
        } else {
            match ch {
                '"' => {
                    // Close the string.
                    if is_key {
                        self.state = JsonState::ExpectColon;
                    } else {
                        self.complete_value();
                    }
                    self.last_token = Some(JsonToken::StringEnd);
                    Ok(())
                }
                '\\' => {
                    self.state = JsonState::InString {
                        is_key,
                        escape: true,
                    };
                    self.last_token = Some(JsonToken::StringChar);
                    Ok(())
                }
                c if (c as u32) < 0x20 => {
                    Err(Self::reject(ch, "unescaped control character in string"))
                }
                _ => {
                    self.last_token = Some(JsonToken::StringChar);
                    Ok(())
                }
            }
        }
    }

    fn step_in_number(&mut self, ch: char, phase: NumberPhase) -> InferResult<()> {
        // A number is terminated by any character that cannot extend it; in
        // that case we finish the number and re-dispatch `ch` from the
        // post-value state.
        if let Some(next) = Self::number_advance(phase, ch) {
            self.state = JsonState::InNumber { phase: next };
            self.last_token = Some(JsonToken::NumberChar);
            return Ok(());
        }
        // The number cannot continue.  It is only well-formed if the current
        // phase is a valid terminal phase.
        if !Self::number_phase_terminal(phase) {
            return Err(Self::reject(ch, "incomplete numeric literal"));
        }
        self.complete_value();
        // Re-process `ch` as a post-value character.
        self.step(ch)
    }

    /// Given the current number phase and the next char, return the next phase
    /// if the char extends the number, else `None` (number ends here).
    fn number_advance(phase: NumberPhase, ch: char) -> Option<NumberPhase> {
        match phase {
            NumberPhase::AfterSign => match ch {
                '0' => Some(NumberPhase::AfterLeadingZero),
                '1'..='9' => Some(NumberPhase::IntDigits),
                _ => None,
            },
            NumberPhase::AfterLeadingZero => match ch {
                '.' => Some(NumberPhase::AfterDot),
                'e' | 'E' => Some(NumberPhase::AfterExp),
                _ => None,
            },
            NumberPhase::IntDigits => match ch {
                '0'..='9' => Some(NumberPhase::IntDigits),
                '.' => Some(NumberPhase::AfterDot),
                'e' | 'E' => Some(NumberPhase::AfterExp),
                _ => None,
            },
            NumberPhase::AfterDot => match ch {
                '0'..='9' => Some(NumberPhase::FracDigits),
                _ => None,
            },
            NumberPhase::FracDigits => match ch {
                '0'..='9' => Some(NumberPhase::FracDigits),
                'e' | 'E' => Some(NumberPhase::AfterExp),
                _ => None,
            },
            NumberPhase::AfterExp => match ch {
                '+' | '-' => Some(NumberPhase::AfterExpSign),
                '0'..='9' => Some(NumberPhase::ExpDigits),
                _ => None,
            },
            NumberPhase::AfterExpSign => match ch {
                '0'..='9' => Some(NumberPhase::ExpDigits),
                _ => None,
            },
            NumberPhase::ExpDigits => match ch {
                '0'..='9' => Some(NumberPhase::ExpDigits),
                _ => None,
            },
        }
    }

    /// Whether a number phase is a valid place to *end* the number.
    fn number_phase_terminal(phase: NumberPhase) -> bool {
        matches!(
            phase,
            NumberPhase::AfterLeadingZero
                | NumberPhase::IntDigits
                | NumberPhase::FracDigits
                | NumberPhase::ExpDigits
        )
    }

    /// Characters that may extend a number in `phase`, plus — if the phase is
    /// terminal — the separators/closers permitted by the enclosing context.
    fn number_next_chars(phase: NumberPhase, stack: &[JsonContext]) -> Vec<char> {
        let mut out = Vec::new();
        // Continuation characters.
        match phase {
            NumberPhase::AfterSign => {
                out.extend_from_slice(&['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            }
            NumberPhase::AfterLeadingZero => out.extend_from_slice(&['.', 'e', 'E']),
            NumberPhase::IntDigits => out.extend_from_slice(&[
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 'e', 'E',
            ]),
            NumberPhase::AfterDot => {
                out.extend_from_slice(&['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            }
            NumberPhase::FracDigits => {
                out.extend_from_slice(&['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'e', 'E'])
            }
            NumberPhase::AfterExp => {
                out.extend_from_slice(&['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            }
            NumberPhase::AfterExpSign => {
                out.extend_from_slice(&['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            }
            NumberPhase::ExpDigits => {
                out.extend_from_slice(&['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            }
        }
        // If the number can end here, the post-value separators are also legal.
        if Self::number_phase_terminal(phase) {
            out.extend(Self::after_value_separators(stack));
        }
        out
    }

    fn step_in_literal(&mut self, ch: char, kind: LiteralKind, pos: usize) -> InferResult<()> {
        let word = kind.word();
        match word.chars().nth(pos) {
            Some(expected) if expected == ch => {
                self.last_token = Some(JsonToken::LiteralChar);
                let next_pos = pos + 1;
                if next_pos == word.len() {
                    self.complete_value();
                } else {
                    self.state = JsonState::InLiteral {
                        kind,
                        pos: next_pos,
                    };
                }
                Ok(())
            }
            _ => Err(Self::reject(
                ch,
                "invalid literal (expected true/false/null)",
            )),
        }
    }

    fn step_after_value(&mut self, ch: char) -> InferResult<()> {
        if Self::is_ws(ch) {
            self.last_token = Some(JsonToken::Whitespace);
            return Ok(());
        }
        match self.stack.last().copied() {
            Some(JsonContext::Object) => match ch {
                ',' => {
                    self.state = JsonState::ExpectKeyOrEnd { after_comma: true };
                    self.last_token = Some(JsonToken::Comma);
                    Ok(())
                }
                '}' => {
                    self.stack.pop();
                    self.last_token = Some(JsonToken::ObjectEnd);
                    self.complete_value();
                    Ok(())
                }
                _ => Err(Self::reject(ch, "expected ',' or '}' inside object")),
            },
            Some(JsonContext::Array) => match ch {
                ',' => {
                    self.state = JsonState::ExpectValue;
                    self.last_token = Some(JsonToken::Comma);
                    Ok(())
                }
                ']' => {
                    self.stack.pop();
                    self.last_token = Some(JsonToken::ArrayEnd);
                    self.complete_value();
                    Ok(())
                }
                _ => Err(Self::reject(ch, "expected ',' or ']' inside array")),
            },
            None => Err(Self::reject(ch, "input after complete top-level value")),
        }
    }

    /// Separators / closers permitted in the current enclosing context once a
    /// value has just completed (used by both `AfterValue` and terminal
    /// numbers, which can end implicitly).
    fn after_value_separators(stack: &[JsonContext]) -> Vec<char> {
        match stack.last() {
            Some(JsonContext::Object) => vec![',', '}'],
            Some(JsonContext::Array) => vec![',', ']'],
            None => Vec::new(),
        }
    }

    fn after_value_chars(&self) -> Vec<char> {
        Self::after_value_separators(&self.stack)
    }
}

impl Default for JsonConstraint {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed a whole string into a fresh constraint, returning the result of the
    /// first failing step (or `Ok(())` if all succeed).
    fn run(s: &str) -> (JsonConstraint, InferResult<()>) {
        let mut c = JsonConstraint::new();
        for ch in s.chars() {
            if let Err(e) = c.step(ch) {
                return (c, Err(e));
            }
        }
        (c, Ok(()))
    }

    #[test]
    fn simple_object_completes() {
        let (c, r) = run("{\"a\":1}");
        assert!(r.is_ok(), "stepping a valid object should succeed");
        assert!(c.is_complete(), "object should be complete");
        assert_eq!(c.depth(), 0);
    }

    #[test]
    fn bare_number_completes() {
        let (c, r) = run("42");
        assert!(r.is_ok());
        assert!(c.is_complete(), "bare number should complete after digits");
    }

    #[test]
    fn nested_arrays_complete() {
        let (c, r) = run("[[]]");
        assert!(r.is_ok(), "nested arrays via the stack should parse");
        assert!(c.is_complete());
        assert_eq!(c.depth(), 0);
    }

    #[test]
    fn invalid_leading_char_errors() {
        let (_, r) = run("}");
        assert!(matches!(r, Err(InferError::SamplingError(_))));
    }

    #[test]
    fn comma_without_key_errors() {
        // `{,}` — a comma where a key string is expected.
        let (_, r) = run("{,}");
        assert!(matches!(r, Err(InferError::SamplingError(_))));
    }

    #[test]
    fn close_without_open_errors() {
        let (_, r) = run("]");
        assert!(matches!(r, Err(InferError::SamplingError(_))));
    }

    #[test]
    fn string_with_escaped_quote() {
        let (c, r) = run("\"a\\\"b\"");
        assert!(r.is_ok(), "escaped quote inside a string should be handled");
        assert!(c.is_complete());
    }

    #[test]
    fn incomplete_object_not_complete() {
        let (c, r) = run("{");
        assert!(r.is_ok());
        assert!(!c.is_complete(), "lone '{{' is not a complete value");
        assert_eq!(c.depth(), 1);
    }

    #[test]
    fn allowed_chars_at_start() {
        let c = JsonConstraint::new();
        let allowed = c.allowed_chars();
        for expected in ['{', '[', '"', '-', '0', 't', 'f', 'n'] {
            assert!(
                allowed.contains(&expected),
                "start should allow {expected:?}, got {allowed:?}"
            );
        }
    }

    #[test]
    fn allowed_chars_after_object_open() {
        let mut c = JsonConstraint::new();
        c.step('{').expect("'{' starts an object");
        let allowed = c.allowed_chars();
        assert!(allowed.contains(&'"'), "after '{{' a key string is allowed");
        assert!(allowed.contains(&'}'), "empty object close is allowed");
    }

    #[test]
    fn mask_logits_blocks_disallowed_token() {
        let c = JsonConstraint::new(); // expecting a value
        let vocab = vec![
            "{".to_owned(),  // allowed (object start)
            "}".to_owned(),  // disallowed at start
            "42".to_owned(), // allowed (digit)
        ];
        let mut logits = vec![1.0_f32, 2.0, 3.0];
        c.mask_logits(&mut logits, &vocab)
            .expect("matching logits and vocab lengths");
        assert!(logits[0].is_finite(), "'{{' should remain finite");
        assert!(
            logits[1].is_infinite() && logits[1] < 0.0,
            "'}}' should be -inf"
        );
        assert!(logits[2].is_finite(), "digit token should remain finite");
    }

    #[test]
    fn mask_logits_length_mismatch_errors() {
        let c = JsonConstraint::new();
        let vocab = vec!["a".to_owned(), "b".to_owned()];
        let mut logits = vec![1.0_f32]; // wrong length
        assert!(matches!(
            c.mask_logits(&mut logits, &vocab),
            Err(InferError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn whitespace_between_tokens_accepted() {
        let (c, r) = run("{ \"a\" : 1 , \"b\" : 2 }");
        assert!(r.is_ok(), "whitespace between tokens should be accepted");
        assert!(c.is_complete());
    }

    #[test]
    fn empty_object_and_array_complete() {
        let (co, ro) = run("{}");
        assert!(ro.is_ok());
        assert!(co.is_complete(), "empty object should complete");

        let (ca, ra) = run("[]");
        assert!(ra.is_ok());
        assert!(ca.is_complete(), "empty array should complete");
    }

    #[test]
    fn number_with_exponent() {
        let (c, r) = run("1e5");
        assert!(r.is_ok(), "exponent number should parse");
        assert!(c.is_complete());

        let (c2, r2) = run("-3.14e+10");
        assert!(r2.is_ok(), "signed fractional exponent should parse");
        assert!(c2.is_complete());
    }

    #[test]
    fn deterministic_repeated_runs() {
        let input = "[1,2,{\"k\":true},null,\"s\"]";
        let (c1, r1) = run(input);
        let (c2, r2) = run(input);
        assert!(r1.is_ok() && r2.is_ok());
        assert_eq!(c1, c2, "identical input must yield identical state");
        assert_eq!(c1.is_complete(), c2.is_complete());
        assert!(c1.is_complete());
    }

    #[test]
    fn literals_true_false_null() {
        for lit in ["true", "false", "null"] {
            let (c, r) = run(lit);
            assert!(r.is_ok(), "{lit} should parse");
            assert!(c.is_complete(), "{lit} should be complete");
        }
    }

    #[test]
    fn invalid_literal_errors() {
        let (_, r) = run("trux");
        assert!(matches!(r, Err(InferError::SamplingError(_))));
    }

    #[test]
    fn leading_zero_then_int_digit_errors() {
        // JSON forbids leading zeros: `01` must be rejected when `1` arrives.
        // After `0` and `1`, the `1` cannot extend the number and `1` is not a
        // valid post-value char at top level → Done then re-dispatch errors.
        let (_, r) = run("01");
        assert!(matches!(r, Err(InferError::SamplingError(_))));
    }

    #[test]
    fn trailing_dot_number_errors() {
        // `1.` then end is incomplete, but feeding a closer reveals the error.
        let (_, r) = run("[1.]");
        assert!(matches!(r, Err(InferError::SamplingError(_))));
    }

    #[test]
    fn object_value_then_comma_then_key() {
        let (c, r) = run("{\"a\":[1,2],\"b\":null}");
        assert!(
            r.is_ok(),
            "object with array value and second key should parse"
        );
        assert!(c.is_complete());
    }

    #[test]
    fn double_close_array_errors() {
        let (_, r) = run("[]]");
        assert!(matches!(r, Err(InferError::SamplingError(_))));
    }

    #[test]
    fn mask_logits_in_string_allows_ordinary_char() {
        // Inside a string, an ordinary token like "x" is allowed, a closing
        // quote token is allowed, but a structural token like "{" is NOT
        // (its leading char `{` is ordinary inside a string, so it IS allowed).
        let mut c = JsonConstraint::new();
        c.step('"').expect("'\"' starts a string value");
        let vocab = vec!["a".to_owned(), "\"".to_owned(), "\\n".to_owned()];
        let mut logits = vec![1.0_f32, 1.0, 1.0];
        c.mask_logits(&mut logits, &vocab)
            .expect("matching lengths");
        assert!(logits[0].is_finite(), "ordinary char allowed in string");
        assert!(logits[1].is_finite(), "closing quote allowed in string");
        assert!(logits[2].is_finite(), "backslash escape allowed in string");
    }

    #[test]
    fn last_token_tracks_structure() {
        let mut c = JsonConstraint::new();
        c.step('[').expect("array start");
        assert_eq!(c.last_token(), Some(JsonToken::ArrayStart));
        c.step('1').expect("number");
        assert_eq!(c.last_token(), Some(JsonToken::NumberChar));
        c.step(']').expect("array end");
        assert_eq!(c.last_token(), Some(JsonToken::ArrayEnd));
        assert!(c.is_complete());
    }
}
