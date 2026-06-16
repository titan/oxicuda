//! Streaming RNN-Transducer greedy decoder.
//!
//! This is the **decoder** counterpart to the RNN-T *loss* in [`crate::ctc::rnnt`].
//! It performs the standard frame-synchronous greedy transducer decode (Graves
//! 2012; Graves et al. 2013): the joint network produces, for each encoder
//! frame `t` and prediction-network label-state `u`, a distribution over the
//! `vocab` symbols (the blank plus the real labels). The decoder greedily picks
//! the argmax symbol:
//!
//! - **blank** → advance to the next encoder frame (`t += 1`), keep the label
//!   state `u`;
//! - **non-blank** → emit that label, advance the label state (`u += 1`), and
//!   stay on the current frame so the prediction network can propose another
//!   symbol.
//!
//! Because a degenerate joint network could keep emitting non-blank symbols
//! forever on a single frame, a **max-symbols-per-frame** guard bounds the
//! number of emissions per frame and forces a frame advance once reached. The
//! decode therefore always terminates in at most `t_frames · max_symbols`
//! steps.
//!
//! Two entry points are provided:
//! - [`TransducerGreedyDecoder::decode_with`] — streaming: a caller-supplied
//!   joint closure fills a `[vocab]` log-prob buffer for each `(t, u)`. This
//!   matches the real online setting where the joint output depends on the
//!   running prediction-network state.
//! - [`TransducerGreedyDecoder::decode`] — convenience: a precomputed
//!   `[t_frames, u_cap, vocab]` joint log-prob tensor; the label-state index is
//!   clamped to `u_cap − 1`, so the final row can act as an absorbing
//!   blank-dominant state.
//!
//! ## References
//! - Graves, A. (2012). "Sequence Transduction with Recurrent Neural Networks."
//!   *ICML Workshop on Representation Learning*.
//! - Graves, A., Mohamed, A., Hinton, G. (2013). "Speech Recognition with Deep
//!   Recurrent Neural Networks." *ICASSP*.

use crate::error::{AudioError, AudioResult};

// ─── Public type ─────────────────────────────────────────────────────────────

/// Greedy frame-synchronous RNN-Transducer decoder.
#[derive(Debug, Clone, Copy)]
pub struct TransducerGreedyDecoder {
    blank: usize,
    max_symbols_per_frame: usize,
}

impl TransducerGreedyDecoder {
    /// Build a greedy transducer decoder.
    ///
    /// `blank` is the blank-symbol index (validated against `vocab` at decode
    /// time). `max_symbols_per_frame` bounds the non-blank emissions on any one
    /// frame and must be `≥ 1`.
    ///
    /// # Errors
    /// - [`AudioError::Internal`] if `max_symbols_per_frame == 0`.
    pub fn new(blank: usize, max_symbols_per_frame: usize) -> AudioResult<Self> {
        if max_symbols_per_frame == 0 {
            return Err(AudioError::Internal(
                "transducer_decode: max_symbols_per_frame must be ≥ 1".into(),
            ));
        }
        Ok(Self {
            blank,
            max_symbols_per_frame,
        })
    }

    /// The blank-symbol index.
    #[must_use]
    pub fn blank(&self) -> usize {
        self.blank
    }

    /// The per-frame emission cap.
    #[must_use]
    pub fn max_symbols_per_frame(&self) -> usize {
        self.max_symbols_per_frame
    }

    /// Streaming greedy decode driven by a joint closure.
    ///
    /// `joint(t, u, buf)` must fill `buf` (length `vocab`) with the joint
    /// log-probabilities for encoder frame `t` and label-state `u`. The closure
    /// may return an error, which aborts the decode.
    ///
    /// Returns the emitted (non-blank) label sequence.
    ///
    /// # Errors
    /// - [`AudioError::EmptyInput`] if `t_frames == 0`.
    /// - [`AudioError::InvalidVocabSize`] if `vocab == 0`.
    /// - [`AudioError::BlankOutOfRange`] if `blank ≥ vocab`.
    /// - Any error returned by the `joint` closure.
    pub fn decode_with<F>(
        &self,
        t_frames: usize,
        vocab: usize,
        mut joint: F,
    ) -> AudioResult<Vec<usize>>
    where
        F: FnMut(usize, usize, &mut [f32]) -> AudioResult<()>,
    {
        if t_frames == 0 {
            return Err(AudioError::EmptyInput {
                msg: "transducer_decode: t_frames == 0".into(),
            });
        }
        if vocab == 0 {
            return Err(AudioError::InvalidVocabSize(vocab));
        }
        if self.blank >= vocab {
            return Err(AudioError::BlankOutOfRange {
                blank: self.blank,
                vocab,
            });
        }

        let mut emitted: Vec<usize> = Vec::new();
        let mut buf = vec![0.0_f32; vocab];
        let mut u = 0_usize; // label-state = number of labels emitted so far
        let mut t = 0_usize;

        while t < t_frames {
            let mut symbols = 0_usize;
            loop {
                joint(t, u, &mut buf)?;
                let k = argmax(&buf);
                if k == self.blank {
                    t += 1;
                    break;
                }
                emitted.push(k);
                u += 1;
                symbols += 1;
                if symbols >= self.max_symbols_per_frame {
                    // Guard: force a frame advance to bound emissions.
                    t += 1;
                    break;
                }
            }
        }

        Ok(emitted)
    }

    /// Convenience greedy decode over a precomputed joint log-prob tensor.
    ///
    /// `joint_logprobs` is row-major `[t_frames, u_cap, vocab]`; element
    /// `(t, u, v)` lives at `(t · u_cap + u) · vocab + v`. The label-state index
    /// is clamped to `u_cap − 1`.
    ///
    /// # Errors
    /// - [`AudioError::EmptyInput`] if `t_frames == 0`.
    /// - [`AudioError::InvalidVocabSize`] if `vocab == 0`.
    /// - [`AudioError::InvalidSequenceLength`] if `u_cap == 0`.
    /// - [`AudioError::BlankOutOfRange`] if `blank ≥ vocab`.
    /// - [`AudioError::DimensionMismatch`] if
    ///   `joint_logprobs.len() != t_frames · u_cap · vocab`.
    pub fn decode(
        &self,
        joint_logprobs: &[f32],
        t_frames: usize,
        u_cap: usize,
        vocab: usize,
    ) -> AudioResult<Vec<usize>> {
        if t_frames == 0 {
            return Err(AudioError::EmptyInput {
                msg: "transducer_decode: t_frames == 0".into(),
            });
        }
        if vocab == 0 {
            return Err(AudioError::InvalidVocabSize(vocab));
        }
        if u_cap == 0 {
            return Err(AudioError::InvalidSequenceLength(u_cap));
        }
        if self.blank >= vocab {
            return Err(AudioError::BlankOutOfRange {
                blank: self.blank,
                vocab,
            });
        }
        let expected = t_frames * u_cap * vocab;
        if joint_logprobs.len() != expected {
            return Err(AudioError::DimensionMismatch {
                expected,
                got: joint_logprobs.len(),
            });
        }

        self.decode_with(t_frames, vocab, |t, u, buf| {
            let uu = u.min(u_cap - 1);
            let base = (t * u_cap + uu) * vocab;
            buf.copy_from_slice(&joint_logprobs[base..base + vocab]);
            Ok(())
        })
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Index of the first maximal element (deterministic on ties).
fn argmax(buf: &[f32]) -> usize {
    let mut best = 0_usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in buf.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best = i;
        }
    }
    best
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `[t_frames, u_cap, vocab]` joint tensor from an explicit
    /// `argmax[(t, u)]` choice table. The chosen symbol gets a high log-prob.
    fn build_joint(
        t_frames: usize,
        u_cap: usize,
        vocab: usize,
        choose: impl Fn(usize, usize) -> usize,
    ) -> Vec<f32> {
        let mut lp = vec![-10.0_f32; t_frames * u_cap * vocab];
        for t in 0..t_frames {
            for u in 0..u_cap {
                let k = choose(t, u);
                lp[(t * u_cap + u) * vocab + k] = 5.0;
            }
        }
        lp
    }

    #[test]
    fn all_blank_yields_empty() {
        let t_frames = 6;
        let u_cap = 2;
        let vocab = 4;
        let blank = 3;
        // Every (t, u) argmax is blank.
        let joint = build_joint(t_frames, u_cap, vocab, |_, _| blank);
        let dec = TransducerGreedyDecoder::new(blank, 4).expect("new");
        let out = dec.decode(&joint, t_frames, u_cap, vocab).expect("decode");
        assert!(out.is_empty(), "all-blank decode must be empty: {out:?}");
    }

    #[test]
    fn recovers_known_sequence() {
        // Expect [1, 2]. blank=0, vocab=3, u-states 0,1,2.
        let t_frames = 4;
        let u_cap = 3;
        let vocab = 3;
        let blank = 0;
        let joint = build_joint(t_frames, u_cap, vocab, |t, u| {
            if t == 0 {
                match u {
                    0 => 1, // emit label 1
                    1 => 2, // emit label 2
                    _ => blank,
                }
            } else {
                blank // remaining frames are blank → advance to the end
            }
        });
        let dec = TransducerGreedyDecoder::new(blank, 5).expect("new");
        let out = dec.decode(&joint, t_frames, u_cap, vocab).expect("decode");
        assert_eq!(
            out,
            vec![1, 2],
            "should recover the constructed label sequence"
        );
    }

    #[test]
    fn recovers_sequence_across_frames() {
        // One emission per frame: frame t at u=t emits label t+1, then blank.
        let t_frames = 3;
        let u_cap = 4;
        let vocab = 5; // labels 1..=3, blank 0
        let blank = 0;
        let joint = build_joint(t_frames, u_cap, vocab, |t, u| {
            // At (t, u==t) emit label t+1; otherwise blank.
            if u == t { t + 1 } else { blank }
        });
        let dec = TransducerGreedyDecoder::new(blank, 4).expect("new");
        let out = dec.decode(&joint, t_frames, u_cap, vocab).expect("decode");
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn max_symbols_guard_bounds_runaway() {
        // Degenerate joint that NEVER emits blank (argmax is always label 1).
        let t_frames = 3;
        let u_cap = 2;
        let vocab = 3;
        let blank = 0;
        let max_symbols = 2;
        let joint = build_joint(t_frames, u_cap, vocab, |_, _| 1); // never blank
        let dec = TransducerGreedyDecoder::new(blank, max_symbols).expect("new");
        let out = dec.decode(&joint, t_frames, u_cap, vocab).expect("decode");
        // Exactly max_symbols emissions per frame, then forced advance.
        assert_eq!(
            out.len(),
            t_frames * max_symbols,
            "guard must bound emissions"
        );
        assert!(out.iter().all(|&k| k == 1));
    }

    #[test]
    fn blank_out_of_range_errors() {
        let joint = vec![0.0_f32; 6]; // [t=2, u=1, v=3]
        let dec = TransducerGreedyDecoder::new(5, 4).expect("new"); // blank 5 ≥ vocab 3
        assert!(matches!(
            dec.decode(&joint, 2, 1, 3).unwrap_err(),
            AudioError::BlankOutOfRange { blank: 5, vocab: 3 }
        ));
    }

    #[test]
    fn empty_input_errors() {
        let dec = TransducerGreedyDecoder::new(0, 4).expect("new");
        assert!(matches!(
            dec.decode(&[], 0, 1, 3).unwrap_err(),
            AudioError::EmptyInput { .. }
        ));
    }

    #[test]
    fn dim_mismatch_errors() {
        let dec = TransducerGreedyDecoder::new(0, 4).expect("new");
        // Correct size is 2*2*3 = 12; provide 5.
        let joint = vec![0.0_f32; 5];
        assert!(matches!(
            dec.decode(&joint, 2, 2, 3).unwrap_err(),
            AudioError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn zero_max_symbols_errors() {
        assert!(matches!(
            TransducerGreedyDecoder::new(0, 0).unwrap_err(),
            AudioError::Internal(_)
        ));
    }

    #[test]
    fn zero_ucap_errors() {
        let dec = TransducerGreedyDecoder::new(0, 4).expect("new");
        let joint = vec![0.0_f32; 0];
        assert!(matches!(
            dec.decode(&joint, 2, 0, 3).unwrap_err(),
            AudioError::InvalidSequenceLength(0)
        ));
    }

    #[test]
    fn decode_with_streaming_closure() {
        // Streaming joint: emit label (u+1) up to 2 labels, then blank.
        let t_frames = 5;
        let vocab = 4; // blank 0, labels 1..=3
        let blank = 0;
        let dec = TransducerGreedyDecoder::new(blank, 8).expect("new");
        let out = dec
            .decode_with(t_frames, vocab, |t, u, buf| {
                buf.fill(-10.0);
                // On the first frame emit labels 1 then 2, then blank forever.
                let k = if t == 0 && u < 2 { u + 1 } else { blank };
                buf[k] = 3.0;
                Ok(())
            })
            .expect("decode_with");
        assert_eq!(out, vec![1, 2]);
    }

    #[test]
    fn decode_with_propagates_closure_error() {
        let dec = TransducerGreedyDecoder::new(0, 4).expect("new");
        let err = dec
            .decode_with(3, 3, |_, _, _| {
                Err(AudioError::NonFinite { msg: "boom".into() })
            })
            .unwrap_err();
        assert!(matches!(err, AudioError::NonFinite { .. }));
    }

    #[test]
    fn u_state_clamped_to_cap() {
        // u_cap=1 means the only available state is u=0; emitting clamps u to 0.
        // With a never-blank joint and max_symbols=3, each frame emits 3 labels.
        let t_frames = 2;
        let u_cap = 1;
        let vocab = 2; // blank 0, label 1
        let blank = 0;
        let joint = build_joint(t_frames, u_cap, vocab, |_, _| 1);
        let dec = TransducerGreedyDecoder::new(blank, 3).expect("new");
        let out = dec.decode(&joint, t_frames, u_cap, vocab).expect("decode");
        assert_eq!(out.len(), t_frames * 3);
    }
}
