//! Application-level PDE solvers built atop the core discretisations.
//!
//! * [`advection_diffusion`] — linear advection–diffusion (upwind + central FDM);
//! * [`cahn_hilliard`] — 4th-order Cahn–Hilliard phase-field equation (spectral,
//!   stabilised semi-implicit convex splitting);
//! * [`immersed_boundary`] — Peskin immersed-boundary spread/interpolate operators
//!   with direct forcing for no-slip boundaries;
//! * [`maxwell`] — Maxwell's equations by the Yee-grid FDTD method (1-D & 2-D TM);
//! * [`wave_equation`] — second-order wave equation (explicit leapfrog);
//! * [`weno`] — 5th-order Jiang–Shu WENO finite-volume advection (SSP-RK3).

pub mod advection_diffusion;
pub mod cahn_hilliard;
pub mod immersed_boundary;
pub mod maxwell;
pub mod wave_equation;
pub mod weno;

pub use advection_diffusion::{
    AdvDiffBoundary, AdvDiffBoundary2d, AdvectionDiffusion1d, AdvectionDiffusion2d,
};
pub use cahn_hilliard::{CahnHilliard, CahnHilliard2d, DEFAULT_STABILIZATION};
pub use immersed_boundary::{DeltaKernel, ImmersedBoundary};
pub use maxwell::{
    Maxwell1d, Maxwell2dTm, MaxwellBoundary1d, MaxwellBoundary2d, MaxwellState1d, MaxwellState2dTm,
};
pub use wave_equation::{WaveBoundary, WaveEquation, WaveState};
pub use weno::{
    WENO5_EPS, WENO5_IDEAL_WEIGHTS, Weno5Advection, weno5_reconstruct_left, weno5_weights,
};
