//! Discrete-time life-table (actuarial) estimator.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// Life-table row for an interval `[a_i, a_{i+1})`.
#[derive(Debug, Clone)]
pub struct LifeTableRow {
    pub interval_start: f64,
    pub interval_end: f64,
    pub n_entering: f64,
    pub n_censored: f64,
    pub n_effective_at_risk: f64,
    pub n_events: f64,
    pub conditional_survival: f64,
    pub cumulative_survival: f64,
}

/// Output of a life-table computation.
#[derive(Debug, Clone)]
pub struct LifeTable {
    pub rows: Vec<LifeTableRow>,
}

impl LifeTable {
    /// Number of intervals.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Vector of cumulative survival values aligned with intervals.
    #[must_use]
    pub fn cumulative_survival(&self) -> Vec<f64> {
        self.rows.iter().map(|r| r.cumulative_survival).collect()
    }
}

/// Compute the actuarial life table given user-defined interval breakpoints (ascending).
/// Censored observations within an interval contribute `0.5` to the effective at-risk count.
pub fn life_table(data: &Dataset, breakpoints: &[f64]) -> SurvivalResult<LifeTable> {
    if breakpoints.len() < 2 {
        return Err(SurvivalError::InvalidParameter(
            "need at least 2 breakpoints".to_string(),
        ));
    }
    for w in breakpoints.windows(2) {
        if w[1] <= w[0] {
            return Err(SurvivalError::InvalidParameter(
                "breakpoints must be strictly increasing".to_string(),
            ));
        }
    }
    let k = breakpoints.len() - 1;
    let mut events = vec![0.0_f64; k];
    let mut censored = vec![0.0_f64; k];
    let mut entering = vec![0.0_f64; k];
    entering[0] = data.len() as f64;
    let mut interval_indices: Vec<Option<usize>> = vec![None; data.len()];
    for (i, o) in data.observations.iter().enumerate() {
        // find interval index
        let mut idx: Option<usize> = None;
        for j in 0..k {
            if o.time >= breakpoints[j] && o.time < breakpoints[j + 1] {
                idx = Some(j);
                break;
            }
        }
        if idx.is_none() && o.time >= breakpoints[k] {
            // beyond table; treat as censored at the last interval boundary
            idx = Some(k - 1);
        }
        interval_indices[i] = idx;
        if let Some(j) = idx {
            if o.event {
                events[j] += 1.0;
            } else {
                censored[j] += 1.0;
            }
        }
    }
    for j in 0..k {
        if j > 0 {
            entering[j] = entering[j - 1] - events[j - 1] - censored[j - 1];
        }
    }
    let mut rows = Vec::with_capacity(k);
    let mut cum = 1.0_f64;
    for j in 0..k {
        let effective = entering[j] - 0.5 * censored[j];
        let cond = if effective > 0.0 {
            1.0 - events[j] / effective
        } else {
            1.0
        };
        cum *= cond.max(0.0);
        rows.push(LifeTableRow {
            interval_start: breakpoints[j],
            interval_end: breakpoints[j + 1],
            n_entering: entering[j],
            n_censored: censored[j],
            n_effective_at_risk: effective,
            n_events: events[j],
            conditional_survival: cond,
            cumulative_survival: cum,
        });
    }
    Ok(LifeTable { rows })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn life_table_simple() {
        // 10 subjects in 5 intervals [0,2), [2,4), [4,6), [6,8), [8,10)
        let times = vec![1.0, 1.5, 3.0, 3.5, 5.0, 5.5, 7.0, 7.5, 9.0, 9.5];
        let events = vec![
            true, false, true, true, false, true, false, true, true, false,
        ];
        let d = Dataset::from_arrays(&times, &events).expect("ok");
        let bp = vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0];
        let lt = life_table(&d, &bp).expect("ok");
        assert_eq!(lt.len(), 5);
        // First interval: 1 event, 1 censored, 10 entering, effective = 10 - 0.5 = 9.5
        assert!((lt.rows[0].n_effective_at_risk - 9.5).abs() < 1.0e-12);
        assert!((lt.rows[0].conditional_survival - (1.0 - 1.0 / 9.5)).abs() < 1.0e-12);
    }

    #[test]
    fn life_table_rejects_few_breakpoints() {
        let d = Dataset::from_arrays(&[1.0], &[true]).expect("ok");
        assert!(life_table(&d, &[0.0]).is_err());
    }

    #[test]
    fn life_table_rejects_non_monotone() {
        let d = Dataset::from_arrays(&[1.0], &[true]).expect("ok");
        assert!(life_table(&d, &[0.0, 1.0, 0.5]).is_err());
    }
}
