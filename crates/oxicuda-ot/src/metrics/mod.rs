//! Diagnostic metrics for transport plans: marginal violation, transport
//! cost, plan entropy, and information-theoretic divergences (KL and
//! Jensen-Shannon).
//!
//! These are pure-Python-style "report cards" used to validate solver output,
//! benchmark convergence, and guide hyper-parameter tuning. None of them
//! mutate the inputs.

/// Diagnostic metrics for transport plans.
pub mod metrics;
