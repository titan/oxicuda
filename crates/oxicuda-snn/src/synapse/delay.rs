//! Synaptic / axonal delay lines.
//!
//! Reference: Izhikevich, "Polychronization: Computation with Spikes",
//! *Neural Computation* 18(2), 245–282 (2006), which shows that *axonal
//! conduction delays* — heterogeneous transmission latencies between neurons —
//! give rise to reproducible *polychronous* spike groups and a combinatorially
//! large memory capacity. Such delays are also the substrate exploited by the
//! Liquid State Machine of Maass, Natschläger & Markram (2002). Faithful
//! simulation of these phenomena requires delaying each presynaptic spike by an
//! integer number of timesteps before it reaches the postsynaptic target.
//!
//! This module implements an integer-step delay as a ring buffer: a value
//! pushed at time `t` re-emerges exactly `delay_steps` steps later. A
//! population variant [`DelayBank`] holds one independent line per synapse and
//! supports *heterogeneous* per-synapse delays, matching the biological
//! diversity of conduction latencies.
//!
//! All buffers operate on `f64` host values to match [`crate::synapse`]
//! conventions; `dt` is expressed in milliseconds by default.

use std::collections::VecDeque;

use crate::error::{SnnError, SnnResult};

/// Configuration for an axonal/synaptic delay line.
#[derive(Debug, Clone, Copy)]
pub struct DelayConfig {
    /// Transmission delay in integer timesteps (`0` = pass-through).
    pub delay_steps: usize,
    /// Integration step `dt` in ms; used only to report the delay in ms.
    pub dt: f64,
}

impl Default for DelayConfig {
    /// Minimal one-step delay with `dt = 1 ms`.
    fn default() -> Self {
        Self {
            delay_steps: 1,
            dt: 1.0,
        }
    }
}

impl DelayConfig {
    /// Physical delay in milliseconds, `delay_steps · dt`.
    #[must_use]
    pub fn delay_ms(&self) -> f64 {
        self.delay_steps as f64 * self.dt
    }
}

/// A single integer-step delay line backed by a FIFO ring buffer.
///
/// A value pushed now re-emerges `delay_steps` pushes later. Before the buffer
/// has filled, the emitted value is `0.0`. With `delay_steps == 0` the line is
/// a transparent pass-through that returns its input immediately.
#[derive(Debug, Clone)]
pub struct DelayLine {
    /// Pending values; the front is the value about to be emitted. Empty for a
    /// zero-delay pass-through line.
    buffer: VecDeque<f64>,
    /// Configured delay in timesteps.
    delay_steps: usize,
}

impl DelayLine {
    /// Construct a delay line of `delay_steps` timesteps.
    ///
    /// `delay_steps == 0` yields a pass-through line. The internal buffer is
    /// pre-filled with `delay_steps` zeros so the line emits `0.0` until real
    /// data has propagated through.
    pub fn new(delay_steps: usize) -> SnnResult<Self> {
        let buffer = VecDeque::from(vec![0.0_f64; delay_steps]);
        Ok(Self {
            buffer,
            delay_steps,
        })
    }

    /// Configured delay in timesteps.
    #[must_use]
    pub fn delay_steps(&self) -> usize {
        self.delay_steps
    }

    /// Push `x` into the line and return the value emerging from `delay_steps`
    /// ago (`0.0` before the buffer has filled). For a pass-through line
    /// (`delay_steps == 0`) this returns `x` unchanged.
    pub fn push(&mut self, x: f64) -> f64 {
        if self.delay_steps == 0 {
            return x;
        }
        self.buffer.push_back(x);
        // The buffer length is held at exactly `delay_steps` entries, so the
        // front is always the value inserted `delay_steps` steps ago.
        self.buffer.pop_front().unwrap_or(0.0)
    }

    /// Peek at the value that the next [`push`](Self::push) will emit without
    /// consuming it. Returns `0.0` for an empty buffer.
    #[must_use]
    pub fn peek(&self) -> f64 {
        self.buffer.front().copied().unwrap_or(0.0)
    }

    /// Reset the line to its initial all-zero state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        for _ in 0..self.delay_steps {
            self.buffer.push_back(0.0);
        }
    }
}

/// A population of independent delay lines, one per synapse.
///
/// Lines may share a homogeneous delay ([`DelayBank::new`]) or carry
/// heterogeneous per-synapse delays ([`DelayBank::with_delays`]), the latter
/// being the configuration that produces polychronous dynamics
/// (Izhikevich 2006).
#[derive(Debug, Clone)]
pub struct DelayBank {
    /// One delay line per synapse.
    lines: Vec<DelayLine>,
}

impl DelayBank {
    /// Construct `n` delay lines that all share `delay_steps`.
    pub fn new(n: usize, delay_steps: usize) -> SnnResult<Self> {
        let mut lines = Vec::with_capacity(n);
        for _ in 0..n {
            lines.push(DelayLine::new(delay_steps)?);
        }
        Ok(Self { lines })
    }

    /// Construct one delay line per entry of `delays`, with heterogeneous
    /// per-synapse latencies.
    pub fn with_delays(delays: &[usize]) -> SnnResult<Self> {
        let mut lines = Vec::with_capacity(delays.len());
        for &d in delays {
            lines.push(DelayLine::new(d)?);
        }
        Ok(Self { lines })
    }

    /// Number of delay lines in the bank.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    /// Whether the bank holds no lines.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Push one input per line and write each delayed output into `out`.
    ///
    /// `inputs` and `out` must both have length equal to the number of lines.
    pub fn step(&mut self, inputs: &[f64], out: &mut [f64]) -> SnnResult<()> {
        if inputs.len() != self.lines.len() {
            return Err(SnnError::IncompatibleLength {
                a: self.lines.len(),
                b: inputs.len(),
            });
        }
        if out.len() != self.lines.len() {
            return Err(SnnError::IncompatibleLength {
                a: self.lines.len(),
                b: out.len(),
            });
        }
        for ((line, &x), o) in self.lines.iter_mut().zip(inputs.iter()).zip(out.iter_mut()) {
            *o = line.push(x);
        }
        Ok(())
    }

    /// Reset every line to its initial all-zero state.
    pub fn reset(&mut self) {
        for line in &mut self.lines {
            line.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn delay_zero_passes_through_immediately() {
        let mut line = DelayLine::new(0).expect("line");
        assert!((line.push(3.0) - 3.0).abs() < EPS);
        assert!((line.push(-1.5) - (-1.5)).abs() < EPS);
    }

    #[test]
    fn delay_one_returns_previous_input() {
        let mut line = DelayLine::new(1).expect("line");
        // First push emits the pre-filled zero; afterwards it emits the prior input.
        assert!(line.push(5.0).abs() < EPS);
        assert!((line.push(7.0) - 5.0).abs() < EPS);
        assert!((line.push(9.0) - 7.0).abs() < EPS);
    }

    #[test]
    fn delay_three_emits_spike_three_steps_later() {
        let mut line = DelayLine::new(3).expect("line");
        let out0 = line.push(1.0); // spike enters
        let out1 = line.push(0.0);
        let out2 = line.push(0.0);
        let out3 = line.push(0.0); // spike should emerge here
        assert!(out0.abs() < EPS);
        assert!(out1.abs() < EPS);
        assert!(out2.abs() < EPS);
        assert!((out3 - 1.0).abs() < EPS, "out3={out3}");
    }

    #[test]
    fn buffer_emits_zero_before_filled() {
        let mut line = DelayLine::new(4).expect("line");
        for _ in 0..4 {
            assert!(line.push(2.0).abs() < EPS, "expected 0 before fill");
        }
        // Fifth push finally emits the first real value.
        assert!((line.push(0.0) - 2.0).abs() < EPS);
    }

    #[test]
    fn reset_clears_buffer() {
        let mut line = DelayLine::new(2).expect("line");
        let _ = line.push(1.0);
        let _ = line.push(2.0);
        line.reset();
        // After reset the line behaves as freshly constructed (delay 2):
        // the buffer holds two zeros, so a value needs two pushes to emerge.
        assert!(line.peek().abs() < EPS);
        assert!(line.push(9.0).abs() < EPS);
        assert!(line.push(0.0).abs() < EPS);
        assert!((line.push(0.0) - 9.0).abs() < EPS);
    }

    #[test]
    fn impulse_response_is_shifted_impulse() {
        let delay = 5usize;
        let mut line = DelayLine::new(delay).expect("line");
        let mut outputs = Vec::new();
        // Feed a unit impulse at t=0 then zeros.
        for t in 0..12 {
            let x = if t == 0 { 1.0 } else { 0.0 };
            outputs.push(line.push(x));
        }
        for (t, &o) in outputs.iter().enumerate() {
            if t == delay {
                assert!((o - 1.0).abs() < EPS, "impulse should appear at t={delay}");
            } else {
                assert!(o.abs() < EPS, "unexpected nonzero at t={t}: {o}");
            }
        }
    }

    #[test]
    fn bank_homogeneous_delays_all_equal() {
        let mut bank = DelayBank::new(3, 2).expect("bank");
        assert_eq!(bank.len(), 3);
        let mut out = vec![0.0_f64; 3];
        bank.step(&[1.0, 2.0, 3.0], &mut out).expect("step");
        assert_eq!(out, vec![0.0, 0.0, 0.0]); // not filled yet
        bank.step(&[0.0; 3], &mut out).expect("step");
        assert_eq!(out, vec![0.0, 0.0, 0.0]);
        bank.step(&[0.0; 3], &mut out).expect("step");
        // Now the first inputs emerge for every (equal-delay) line.
        assert!((out[0] - 1.0).abs() < EPS);
        assert!((out[1] - 2.0).abs() < EPS);
        assert!((out[2] - 3.0).abs() < EPS);
    }

    #[test]
    fn bank_heterogeneous_each_line_independent() {
        // Delays 0, 1, 2 on three lines: the impulse appears at different steps.
        let mut bank = DelayBank::with_delays(&[0, 1, 2]).expect("bank");
        let mut out = vec![0.0_f64; 3];
        bank.step(&[1.0, 1.0, 1.0], &mut out).expect("step");
        // Line 0 (delay 0) emits immediately; lines 1, 2 still zero.
        assert!((out[0] - 1.0).abs() < EPS);
        assert!(out[1].abs() < EPS);
        assert!(out[2].abs() < EPS);
        bank.step(&[0.0; 3], &mut out).expect("step");
        // Line 1 (delay 1) now emits its impulse.
        assert!(out[0].abs() < EPS);
        assert!((out[1] - 1.0).abs() < EPS);
        assert!(out[2].abs() < EPS);
        bank.step(&[0.0; 3], &mut out).expect("step");
        // Line 2 (delay 2) emits last.
        assert!((out[2] - 1.0).abs() < EPS);
    }

    #[test]
    fn bank_step_length_mismatch_rejected() {
        let mut bank = DelayBank::new(3, 1).expect("bank");
        let mut out = vec![0.0_f64; 3];
        // inputs length mismatch
        let err = bank.step(&[1.0, 2.0], &mut out);
        assert!(matches!(err, Err(SnnError::IncompatibleLength { .. })));
        // out length mismatch
        let mut bad_out = vec![0.0_f64; 4];
        let err2 = bank.step(&[1.0; 3], &mut bad_out);
        assert!(matches!(err2, Err(SnnError::IncompatibleLength { .. })));
    }

    #[test]
    fn with_delays_builds_correct_count() {
        let bank = DelayBank::with_delays(&[1, 2, 3, 4, 5]).expect("bank");
        assert_eq!(bank.len(), 5);
        assert!(!bank.is_empty());
        let empty = DelayBank::with_delays(&[]).expect("bank");
        assert!(empty.is_empty());
    }

    #[test]
    fn continuous_stream_reproduced_shifted() {
        let delay = 3usize;
        let mut line = DelayLine::new(delay).expect("line");
        let stream = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = Vec::new();
        for &x in &stream {
            out.push(line.push(x));
        }
        // out[t] == stream[t - delay] for t >= delay, else 0.
        for (t, &o) in out.iter().enumerate() {
            if t < delay {
                assert!(o.abs() < EPS, "t={t} should be 0");
            } else {
                assert!(
                    (o - stream[t - delay]).abs() < EPS,
                    "t={t} out={o} expected={}",
                    stream[t - delay]
                );
            }
        }
    }

    #[test]
    fn peek_does_not_consume() {
        let mut line = DelayLine::new(2).expect("line");
        let _ = line.push(4.0);
        let _ = line.push(8.0);
        // peek twice — value must not change, and the next push still emits it.
        let p1 = line.peek();
        let p2 = line.peek();
        assert!((p1 - p2).abs() < EPS);
        assert!((p1 - 4.0).abs() < EPS, "peek={p1}");
        let emitted = line.push(0.0);
        assert!((emitted - 4.0).abs() < EPS);
    }

    #[test]
    fn delay_ms_reports_physical_delay() {
        let cfg = DelayConfig {
            delay_steps: 4,
            dt: 0.25,
        };
        assert!((cfg.delay_ms() - 1.0).abs() < EPS);
        let bank = DelayBank::new(2, 4).expect("bank");
        assert_eq!(bank.len(), 2);
    }
}
