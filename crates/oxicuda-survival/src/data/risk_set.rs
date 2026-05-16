//! Risk-set helpers: ordered unique event times with associated at-risk counts.

use crate::data::dataset::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

/// A "risk set" summary at each unique observed time:
/// `(t, d, n)` — time, number of events at t, number at risk just before t.
#[derive(Debug, Clone)]
pub struct RiskSet {
    pub times: Vec<f64>,
    pub deaths: Vec<f64>,
    pub at_risk: Vec<f64>,
}

impl RiskSet {
    /// Build a risk set from a `Dataset`. Times are sorted ascending; tied times are merged.
    /// At-risk counts include all subjects with `time >= t_i`.
    pub fn from_dataset(d: &Dataset) -> SurvivalResult<Self> {
        if d.is_empty() {
            return Err(SurvivalError::EmptyDataset);
        }
        let mut order = d.order_by_time();
        order.sort_by(|&a, &b| {
            d.observations[a]
                .time
                .partial_cmp(&d.observations[b].time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut times = Vec::new();
        let mut deaths = Vec::new();
        let mut at_risk = Vec::new();
        let n = d.len();
        let mut i = 0usize;
        while i < order.len() {
            let t = d.observations[order[i]].time;
            let mut dd = 0.0_f64;
            let mut total = 0.0_f64;
            let mut j = i;
            while j < order.len() && d.observations[order[j]].time == t {
                total += 1.0;
                if d.observations[order[j]].event {
                    dd += 1.0;
                }
                j += 1;
            }
            // n at risk just before this time = remaining (i to end) before consuming
            let nrisk = (n - i) as f64;
            times.push(t);
            deaths.push(dd);
            at_risk.push(nrisk);
            i = j;
            let _ = total;
        }
        Ok(Self {
            times,
            deaths,
            at_risk,
        })
    }

    /// Number of distinct times.
    #[must_use]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Whether the risk set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Iterate as `(t, d, n)` triples.
    pub fn iter(&self) -> impl Iterator<Item = (f64, f64, f64)> + '_ {
        self.times
            .iter()
            .zip(self.deaths.iter())
            .zip(self.at_risk.iter())
            .map(|((t, d), n)| (*t, *d, *n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn risk_set_basic() {
        let d = Dataset::from_arrays(&[1.0, 2.0, 3.0], &[true, false, true]).expect("ok");
        let rs = RiskSet::from_dataset(&d).expect("ok");
        assert_eq!(rs.len(), 3);
        assert_eq!(rs.at_risk, vec![3.0, 2.0, 1.0]);
        assert_eq!(rs.deaths, vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn risk_set_ties_merged() {
        let d =
            Dataset::from_arrays(&[1.0, 1.0, 2.0, 2.0], &[true, true, false, true]).expect("ok");
        let rs = RiskSet::from_dataset(&d).expect("ok");
        assert_eq!(rs.len(), 2);
        assert_eq!(rs.deaths[0], 2.0);
        assert_eq!(rs.deaths[1], 1.0);
        assert_eq!(rs.at_risk[0], 4.0);
        assert_eq!(rs.at_risk[1], 2.0);
    }
}
