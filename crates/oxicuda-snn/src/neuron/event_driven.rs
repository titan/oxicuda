//! Event-driven LIF simulation backend for very sparse spiking regimes.
//!
//! A clock-stepped simulation updates **every** neuron at **every** time step,
//! costing `Θ(t_steps · n)` work regardless of how few spikes occur. When the
//! spiking activity is sparse, almost all of that work decays an otherwise
//! untouched membrane potential. The event-driven backend instead advances
//! time from one spike event to the next and only ever touches a neuron when an
//! event actually arrives for it.
//!
//! # Exact lazy decay
//!
//! The continuous-time LIF membrane obeys `dv/dt = −v / τ_m` between input
//! events, whose exact solution over an interval `Δ` is
//!
//! ```text
//! v(t + Δ) = v(t) · exp(−Δ / τ_m).
//! ```
//!
//! Each neuron stores its membrane potential together with the time it was last
//! updated. When an event for neuron `i` is dequeued at time `t`, the membrane
//! is first decayed **analytically** from `last_update[i]` to `t` using the
//! closed-form factor above (never an Euler approximation), then the synaptic
//! weight jump is applied, and a spike + reset is emitted if the threshold is
//! crossed. This is exact, not approximate.
//!
//! # Correspondence with the clock-stepped model
//!
//! The discrete LIF step in [`crate::neuron::lif`] uses `β = exp(−dt / τ_m)`
//! and applies `v ← β · v + I` once per step. Over `k` consecutive empty steps
//! the membrane therefore decays by `βᵏ = exp(−k · dt / τ_m)`, which is exactly
//! the analytic factor `exp(−Δ / τ_m)` with `Δ = k · dt`. Consequently, when
//! input events are placed on the integration grid (times that are integer
//! multiples of `dt`) and synapses deliver an instantaneous additive jump, the
//! event-driven trajectory reproduces the clock-stepped trajectory — identical
//! spike times and membrane values up to floating-point rounding — while
//! performing far fewer membrane updates in the sparse regime.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::error::{SnnError, SnnResult};
use crate::neuron::lif::{LifConfig, ResetMode};

/// A synaptic event delivering `weight` of current to neuron `target` at `time`.
///
/// External stimulation is injected as input events; threshold crossings push
/// further events onto downstream neurons via the recurrent weight matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SynapticEvent {
    /// Delivery time (same units as [`LifConfig::dt`]).
    pub time: f32,
    /// Index of the neuron receiving the current jump.
    pub target: usize,
    /// Magnitude of the instantaneous membrane jump.
    pub weight: f32,
}

/// An emitted output spike: neuron `neuron` crossed threshold at `time`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpikeRecord {
    /// Time at which the threshold crossing occurred.
    pub time: f32,
    /// Index of the neuron that spiked.
    pub neuron: usize,
}

/// Internal heap entry: a `SynapticEvent` plus a monotone sequence number that
/// breaks ties between simultaneous events deterministically (FIFO order).
#[derive(Debug, Clone, Copy)]
struct QueuedEvent {
    event: SynapticEvent,
    seq: u64,
}

impl PartialEq for QueuedEvent {
    fn eq(&self, other: &Self) -> bool {
        self.event.time.total_cmp(&other.event.time) == Ordering::Equal && self.seq == other.seq
    }
}

impl Eq for QueuedEvent {}

impl Ord for QueuedEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` is a max-heap; invert so the *earliest* time pops first.
        // Ties are ordered by ascending sequence number (also inverted), giving
        // deterministic, insertion-order processing of simultaneous events.
        other
            .event
            .time
            .total_cmp(&self.event.time)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl PartialOrd for QueuedEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Event-driven LIF network over `n` neurons with a recurrent weight matrix.
///
/// The weight matrix `w_rec` is `[n × n]` row-major in `(post, pre)` order:
/// `w_rec[post * n + pre]` is the jump delivered to `post` when `pre` spikes.
/// An empty (all-zero or absent) matrix models a feed-forward population that
/// only responds to external input events.
#[derive(Debug, Clone)]
pub struct EventDrivenLif {
    /// Number of neurons.
    pub n: usize,
    /// LIF parameters (`tau_m`, `v_th`, `v_rest`, `reset`; `dt` only sets the grid).
    pub cfg: LifConfig,
    /// Recurrent weights `[n × n]`, row-major `(post, pre)`; `None` ⇒ feed-forward.
    pub w_rec: Option<Vec<f32>>,
    /// Synaptic transmission delay added to downstream event times (`≥ 0`).
    pub delay: f32,
    /// Membrane potential per neuron.
    v: Vec<f32>,
    /// Time at which each neuron was last decayed/updated.
    last_update: Vec<f32>,
    /// Pending event queue ordered by `(time, seq)`.
    queue: BinaryHeap<QueuedEvent>,
    /// Monotone counter giving each queued event a unique tie-break key.
    next_seq: u64,
    /// Total number of neuron-state updates performed (cost accounting).
    updates: usize,
}

impl EventDrivenLif {
    /// Create a feed-forward (no recurrence) event-driven population.
    ///
    /// Errors: [`SnnError::BadDim`] for `n == 0`, plus the usual
    /// [`LifConfig`] validation ([`SnnError::BadTau`] / [`SnnError::BadDt`] /
    /// [`SnnError::BadThreshold`]).
    pub fn new(n: usize, cfg: LifConfig) -> SnnResult<Self> {
        Self::with_recurrence(n, cfg, None, 0.0)
    }

    /// Create an event-driven population with an optional recurrent matrix and
    /// synaptic delay.
    ///
    /// Errors as [`EventDrivenLif::new`], plus [`SnnError::BadShape`] when a
    /// recurrent matrix is supplied with the wrong length and
    /// [`SnnError::OutOfRange`] for a negative / non-finite `delay`.
    pub fn with_recurrence(
        n: usize,
        cfg: LifConfig,
        w_rec: Option<Vec<f32>>,
        delay: f32,
    ) -> SnnResult<Self> {
        if n == 0 {
            return Err(SnnError::BadDim { got: n });
        }
        if cfg.tau_m <= 0.0 || !cfg.tau_m.is_finite() {
            return Err(SnnError::BadTau { tau: cfg.tau_m });
        }
        if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
            return Err(SnnError::BadDt { dt: cfg.dt });
        }
        if !cfg.v_th.is_finite() {
            return Err(SnnError::BadThreshold { v_th: cfg.v_th });
        }
        if !delay.is_finite() || delay < 0.0 {
            return Err(SnnError::OutOfRange {
                name: "delay".into(),
                val: delay,
            });
        }
        if let Some(w) = &w_rec
            && w.len() != n * n
        {
            return Err(SnnError::BadShape {
                expected: n * n,
                got: w.len(),
            });
        }
        Ok(Self {
            n,
            cfg,
            w_rec,
            delay,
            v: vec![cfg.v_rest; n],
            last_update: vec![0.0_f32; n],
            queue: BinaryHeap::new(),
            next_seq: 0,
            updates: 0,
        })
    }

    /// Number of membrane-state updates performed so far.
    ///
    /// In a sparse regime this is far below the `t_steps · n` updates a
    /// clock-stepped simulation would perform.
    #[must_use]
    pub fn update_count(&self) -> usize {
        self.updates
    }

    /// Membrane potential of neuron `i` as last computed (not decayed to "now").
    #[must_use]
    pub fn membrane(&self, i: usize) -> f32 {
        self.v[i]
    }

    /// Push a single external stimulation event into the queue.
    ///
    /// Errors: [`SnnError::OutOfRange`] for a non-finite `time`,
    /// [`SnnError::LayerOutOfRange`] when `target >= n`.
    pub fn push_event(&mut self, event: SynapticEvent) -> SnnResult<()> {
        if !event.time.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "event.time".into(),
                val: event.time,
            });
        }
        if event.target >= self.n {
            return Err(SnnError::LayerOutOfRange {
                idx: event.target,
                num_layers: self.n,
            });
        }
        self.queue.push(QueuedEvent {
            event,
            seq: self.next_seq,
        });
        self.next_seq += 1;
        Ok(())
    }

    /// Decay neuron `i` analytically from its last update time to `time`.
    ///
    /// Applies the exact factor `exp(−Δ / τ_m)` and advances `last_update[i]`.
    /// Events must be processed in non-decreasing time order, so `Δ ≥ 0`.
    fn decay_to(&mut self, i: usize, time: f32) {
        let dt = time - self.last_update[i];
        if dt > 0.0 {
            let factor = (-dt / self.cfg.tau_m).exp();
            self.v[i] = (self.v[i] - self.cfg.v_rest) * factor + self.cfg.v_rest;
            self.last_update[i] = time;
            self.updates += 1;
        }
    }

    /// Apply a spike reset to neuron `i` (membrane already at/above threshold).
    fn apply_reset(&mut self, i: usize) {
        match self.cfg.reset {
            ResetMode::Hard => self.v[i] = self.cfg.v_rest,
            ResetMode::Soft => self.v[i] -= self.cfg.v_th,
        }
    }

    /// Run the event-driven simulation until the queue is empty or `time > t_end`.
    ///
    /// Returns the recorded output spikes in time order. Threshold crossings
    /// push recurrent events (delayed by `self.delay`) onto downstream neurons,
    /// which may in turn spike, so the simulation terminates naturally once all
    /// activity within `[0, t_end]` has been consumed.
    ///
    /// Errors: [`SnnError::OutOfRange`] for a non-finite `t_end`.
    pub fn run(&mut self, t_end: f32) -> SnnResult<Vec<SpikeRecord>> {
        if !t_end.is_finite() {
            return Err(SnnError::OutOfRange {
                name: "t_end".into(),
                val: t_end,
            });
        }
        let mut spikes = Vec::new();
        while let Some(top) = self.queue.peek().copied() {
            if top.event.time > t_end {
                break;
            }
            let qe = self.queue.pop().expect("peeked element exists");
            let SynapticEvent {
                time,
                target,
                weight,
            } = qe.event;

            // 1. Lazily decay the target to the event time (exact analytic decay).
            self.decay_to(target, time);
            // 2. Apply the synaptic current jump.
            self.v[target] += weight;

            // 3. Threshold crossing → emit spike, reset, propagate downstream.
            if self.v[target] >= self.cfg.v_th {
                spikes.push(SpikeRecord {
                    time,
                    neuron: target,
                });
                self.apply_reset(target);
                self.propagate(target, time);
            }
        }
        Ok(spikes)
    }

    /// Schedule recurrent events for every post-synaptic neuron of `pre`.
    fn propagate(&mut self, pre: usize, time: f32) {
        let Some(w_rec) = self.w_rec.clone() else {
            return;
        };
        let fire_time = time + self.delay;
        for post in 0..self.n {
            let w = w_rec[post * self.n + pre];
            if w != 0.0 && w.is_finite() {
                self.queue.push(QueuedEvent {
                    event: SynapticEvent {
                        time: fire_time,
                        target: post,
                        weight: w,
                    },
                    seq: self.next_seq,
                });
                self.next_seq += 1;
            }
        }
    }
}

/// Clock-stepped reference: simulate a single feed-forward LIF neuron driven by
/// per-step input current, returning the spike times (integer step × `dt`).
///
/// Used to validate that [`EventDrivenLif`] reproduces the dense trajectory.
/// `currents[t]` is the input delivered at step `t`; the membrane updates as
/// `v ← β · v + I` with hard/soft reset per `cfg.reset`.
///
/// Errors: [`SnnError::BadTau`] / [`SnnError::BadDt`] for an invalid config.
pub fn clock_stepped_spike_times(currents: &[f32], cfg: &LifConfig) -> SnnResult<Vec<f32>> {
    if cfg.tau_m <= 0.0 || !cfg.tau_m.is_finite() {
        return Err(SnnError::BadTau { tau: cfg.tau_m });
    }
    if cfg.dt <= 0.0 || !cfg.dt.is_finite() {
        return Err(SnnError::BadDt { dt: cfg.dt });
    }
    let beta = (-cfg.dt / cfg.tau_m).exp();
    let mut v = cfg.v_rest;
    let mut times = Vec::new();
    for (t, &i_in) in currents.iter().enumerate() {
        v = beta * (v - cfg.v_rest) + cfg.v_rest + i_in;
        if v >= cfg.v_th {
            // Spike registered at the end of step `t` ⇒ time (t + 1) · dt,
            // matching an input event placed at that grid time.
            times.push((t as f32 + 1.0) * cfg.dt);
            match cfg.reset {
                ResetMode::Hard => v = cfg.v_rest,
                ResetMode::Soft => v -= cfg.v_th,
            }
        }
    }
    Ok(times)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_cfg() -> LifConfig {
        LifConfig {
            tau_m: 20.0,
            v_th: 1.0,
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Hard,
        }
    }

    #[test]
    fn analytic_decay_matches_powers_of_beta() {
        // A single sub-threshold input then pure decay: event-driven membrane
        // must equal β^k decay of the clock-stepped model.
        let cfg = LifConfig {
            v_th: 100.0,
            ..base_cfg()
        }; // never fires
        let mut ed = EventDrivenLif::new(1, cfg).expect("ctor");
        // inject 0.5 at t=0
        ed.push_event(SynapticEvent {
            time: 0.0,
            target: 0,
            weight: 0.5,
        })
        .expect("push");
        // dummy zero-weight event at t=10 forces a lazy decay-to to be observed
        ed.push_event(SynapticEvent {
            time: 10.0,
            target: 0,
            weight: 0.0,
        })
        .expect("push");
        ed.run(10.0).expect("run");
        let beta = (-cfg.dt / cfg.tau_m).exp();
        let expected = 0.5 * beta.powi(10);
        assert!(
            (ed.membrane(0) - expected).abs() < 1e-6,
            "v={} expected={}",
            ed.membrane(0),
            expected
        );
    }

    #[test]
    fn single_neuron_matches_clock_stepped_hard_reset() {
        // Constant per-step drive of 0.1 for 100 steps; compare spike times.
        let cfg = base_cfg();
        let steps = 100usize;
        let drive = 0.1_f32;
        let currents = vec![drive; steps];
        let ref_times = clock_stepped_spike_times(&currents, &cfg).expect("ref");

        // Event-driven: one input event of `drive` at every grid time 1..=steps.
        let mut ed = EventDrivenLif::new(1, cfg).expect("ctor");
        for t in 1..=steps {
            ed.push_event(SynapticEvent {
                time: t as f32 * cfg.dt,
                target: 0,
                weight: drive,
            })
            .expect("push");
        }
        let ed_spikes = ed.run(steps as f32 * cfg.dt).expect("run");
        let ed_times: Vec<f32> = ed_spikes.iter().map(|s| s.time).collect();

        assert_eq!(
            ed_times.len(),
            ref_times.len(),
            "spike count mismatch: ed={ed_times:?} ref={ref_times:?}"
        );
        for (a, b) in ed_times.iter().zip(ref_times.iter()) {
            assert!((a - b).abs() < 1e-4, "spike time {a} vs {b}");
        }
        assert!(!ref_times.is_empty(), "test should produce spikes");
    }

    #[test]
    fn single_neuron_matches_clock_stepped_soft_reset() {
        let cfg = LifConfig {
            reset: ResetMode::Soft,
            ..base_cfg()
        };
        let steps = 80usize;
        let drive = 0.2_f32;
        let currents = vec![drive; steps];
        let ref_times = clock_stepped_spike_times(&currents, &cfg).expect("ref");

        let mut ed = EventDrivenLif::new(1, cfg).expect("ctor");
        for t in 1..=steps {
            ed.push_event(SynapticEvent {
                time: t as f32 * cfg.dt,
                target: 0,
                weight: drive,
            })
            .expect("push");
        }
        let ed_times: Vec<f32> = ed
            .run(steps as f32 * cfg.dt)
            .expect("run")
            .iter()
            .map(|s| s.time)
            .collect();

        assert_eq!(ed_times.len(), ref_times.len());
        for (a, b) in ed_times.iter().zip(ref_times.iter()) {
            assert!((a - b).abs() < 1e-4, "spike time {a} vs {b}");
        }
    }

    #[test]
    fn sparse_regime_processes_far_fewer_updates() {
        // 1 neuron, but a long horizon with input only on a few grid points.
        // A clock-stepped sim would do t_steps updates; event-driven does ~#events.
        let cfg = LifConfig {
            v_th: 100.0,
            ..base_cfg()
        };
        let t_steps = 10_000usize;
        let mut ed = EventDrivenLif::new(1, cfg).expect("ctor");
        // only 3 input events across the whole 10k-step horizon
        for &t in &[10.0_f32, 5_000.0, 9_999.0] {
            ed.push_event(SynapticEvent {
                time: t,
                target: 0,
                weight: 0.3,
            })
            .expect("push");
        }
        ed.run(t_steps as f32).expect("run");
        // far fewer than t_steps updates (at most one decay per event).
        assert!(
            ed.update_count() <= 3,
            "updates={} should be tiny vs {t_steps}",
            ed.update_count()
        );
    }

    #[test]
    fn recurrent_chain_propagates_spike() {
        // 2 neurons: neuron 0 → neuron 1 with a supra-threshold weight.
        // Driving 0 to fire should make 1 fire one delay later.
        let cfg = LifConfig {
            tau_m: 1e9, // effectively no leak
            v_th: 1.0,
            v_rest: 0.0,
            dt: 1.0,
            reset: ResetMode::Hard,
        };
        let n = 2;
        // w_rec[post*n + pre]; post=1, pre=0 ⇒ index 1*2 + 0 = 2
        let mut w = vec![0.0_f32; n * n];
        w[n] = 1.5; // post=1, pre=0 ⇒ 0 → 1, supra-threshold
        let delay = 1.0_f32;
        let mut ed = EventDrivenLif::with_recurrence(n, cfg, Some(w), delay).expect("ctor");
        // make neuron 0 fire at t=1
        ed.push_event(SynapticEvent {
            time: 1.0,
            target: 0,
            weight: 1.5,
        })
        .expect("push");
        let spikes = ed.run(10.0).expect("run");
        assert_eq!(spikes.len(), 2, "both neurons should fire: {spikes:?}");
        assert_eq!(spikes[0].neuron, 0);
        assert!((spikes[0].time - 1.0).abs() < 1e-6);
        assert_eq!(spikes[1].neuron, 1);
        // neuron 1 fires one synaptic delay after neuron 0
        assert!((spikes[1].time - (1.0 + delay)).abs() < 1e-6);
    }

    #[test]
    fn simultaneous_events_accumulate() {
        // Two sub-threshold events at the same time on the same neuron should
        // sum and cross threshold (deterministic tie-break, both applied).
        let cfg = LifConfig {
            tau_m: 1e9,
            v_th: 1.0,
            ..base_cfg()
        };
        let mut ed = EventDrivenLif::new(1, cfg).expect("ctor");
        ed.push_event(SynapticEvent {
            time: 1.0,
            target: 0,
            weight: 0.6,
        })
        .expect("push");
        ed.push_event(SynapticEvent {
            time: 1.0,
            target: 0,
            weight: 0.6,
        })
        .expect("push");
        let spikes = ed.run(2.0).expect("run");
        // first event → 0.6 (no spike); second → 1.2 (spike at t=1)
        assert_eq!(spikes.len(), 1);
        assert!((spikes[0].time - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_invalid_arguments() {
        assert!(matches!(
            EventDrivenLif::new(0, base_cfg()),
            Err(SnnError::BadDim { .. })
        ));
        let bad_tau = LifConfig {
            tau_m: 0.0,
            ..base_cfg()
        };
        assert!(matches!(
            EventDrivenLif::new(2, bad_tau),
            Err(SnnError::BadTau { .. })
        ));
        assert!(matches!(
            EventDrivenLif::with_recurrence(2, base_cfg(), Some(vec![0.0; 3]), 0.0),
            Err(SnnError::BadShape { .. })
        ));
        assert!(matches!(
            EventDrivenLif::with_recurrence(2, base_cfg(), None, -1.0),
            Err(SnnError::OutOfRange { .. })
        ));
        let mut ed = EventDrivenLif::new(2, base_cfg()).expect("ctor");
        assert!(matches!(
            ed.push_event(SynapticEvent {
                time: 0.0,
                target: 5,
                weight: 1.0
            }),
            Err(SnnError::LayerOutOfRange { .. })
        ));
    }
}
