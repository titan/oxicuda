//! Quantum-circuit interface on top of MPS / TEBD.
//!
//! This module provides a high-level API for building quantum circuits that
//! compile down to TEBD-style MPS gate application (TEBD = Time-Evolving Block
//! Decimation).
//!
//! # Architecture
//!
//! ```text
//! Circuit  ──► apply_to_mps ──► for each gate:
//!                                  Single: 1-site contraction
//!                                  Two   : TEBD SVD (adjacent bonds only)
//! ```
//!
//! Gates are accumulated in a `Vec<CircuitGate>` and applied sequentially when
//! [`Circuit::apply_to_mps`] is called.  All arithmetic is real-valued; complex
//! gates (Y, S, T, Rz, iSWAP) use the documented real approximations from
//! [`gates`].
//!
//! # Example
//!
//! ```rust
//! use oxicuda_tn::circuits::{Circuit, CircuitConfig};
//!
//! let mut circ = Circuit::new(2);
//! circ.h(0).unwrap();
//! circ.cnot(0, 1).unwrap();
//!
//! // Product state |00⟩: each site has amplitude [1, 0] for |0⟩.
//! let mps_data = vec![vec![1.0_f64, 0.0], vec![1.0_f64, 0.0]];
//! let mps_shapes = [[1usize, 2, 1], [1, 2, 1]];
//! let cfg = CircuitConfig::default();
//!
//! let (_new_data, _new_shapes) = circ
//!     .apply_to_mps(&mps_data, &mps_shapes, &cfg)
//!     .unwrap();
//! ```

pub mod gates;
pub(crate) mod mod_impl;
pub mod tebd_compile;

pub use mod_impl::{Circuit, CircuitConfig, CircuitGate};
pub use tebd_compile::compile_circuit_to_tebd_gates;
