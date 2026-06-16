//! ε-budget tracker for stateful attack pipelines.
//!
//! Some adaptive attacks (e.g. AutoAttack) consume a fraction of the total
//! perturbation budget per inner loop. This helper makes spending visible and
//! enforceable: a single `EpsilonBudget` aggregates spend across calls and
//! errors as soon as the total exceeds the cap.

use crate::error::{AdvError, AdvResult};

/// Tracks cumulative epsilon spend up to a cap.
#[derive(Debug, Clone)]
pub struct EpsilonBudget {
    /// Maximum total budget.
    pub max: f32,
    /// Currently spent (always `<= max`).
    pub spent: f32,
}

impl EpsilonBudget {
    /// New tracker with the given total cap.
    ///
    /// # Errors
    /// [`AdvError::InvalidEpsilon`] if `max <= 0` or non-finite.
    pub fn new(max: f32) -> AdvResult<Self> {
        if !(max.is_finite() && max > 0.0) {
            return Err(AdvError::InvalidEpsilon { eps: max });
        }
        Ok(Self { max, spent: 0.0 })
    }

    /// Remaining budget.
    #[must_use]
    pub fn remaining(&self) -> f32 {
        (self.max - self.spent).max(0.0)
    }

    /// Spend `amount` from the budget.
    ///
    /// # Errors
    /// - [`AdvError::InvalidEpsilon`] if `amount` is non-finite or negative.
    /// - [`AdvError::BudgetExceeded`] if the spend would exceed `max`.
    pub fn spend(&mut self, amount: f32) -> AdvResult<()> {
        if !(amount.is_finite() && amount >= 0.0) {
            return Err(AdvError::InvalidEpsilon { eps: amount });
        }
        let new_spent = self.spent + amount;
        if new_spent > self.max + 1e-6 {
            return Err(AdvError::BudgetExceeded {
                spent: new_spent,
                max: self.max,
            });
        }
        self.spent = new_spent;
        Ok(())
    }

    /// Reset the spent counter to zero.
    pub fn reset(&mut self) {
        self.spent = 0.0;
    }

    /// True if budget is fully consumed.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.spent + 1e-6 >= self.max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_new_validates() {
        assert!(EpsilonBudget::new(0.0).is_err());
        assert!(EpsilonBudget::new(-0.1).is_err());
        assert!(EpsilonBudget::new(f32::NAN).is_err());
        assert!(EpsilonBudget::new(1.0).is_ok());
    }

    #[test]
    fn budget_spend_tracks() {
        let mut b = EpsilonBudget::new(1.0).expect("new should succeed");
        b.spend(0.3).expect("spend should succeed");
        assert!((b.spent - 0.3).abs() < 1e-6);
        assert!((b.remaining() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn budget_exhausted_after_full_spend() {
        let mut b = EpsilonBudget::new(0.5).expect("new should succeed");
        b.spend(0.5).expect("spend should succeed");
        assert!(b.is_exhausted());
    }

    #[test]
    fn budget_overspend_errors() {
        let mut b = EpsilonBudget::new(0.5).expect("new should succeed");
        let r = b.spend(0.6);
        assert!(r.is_err());
    }

    #[test]
    fn budget_reset_clears_spent() {
        let mut b = EpsilonBudget::new(0.5).expect("new should succeed");
        b.spend(0.5).expect("spend should succeed");
        b.reset();
        assert!((b.spent - 0.0).abs() < 1e-6);
        assert!(!b.is_exhausted());
    }

    #[test]
    fn budget_rejects_negative_spend() {
        let mut b = EpsilonBudget::new(1.0).expect("new should succeed");
        assert!(b.spend(-0.1).is_err());
    }
}
