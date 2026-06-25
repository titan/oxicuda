//! Compression-ratio / effective-rank dashboards.
//!
//! Aggregates the per-adapter primitives in [`crate::metrics::efficiency`] into a single
//! report over a whole *set* of adapters (one per layer / module). Where
//! [`efficiency`](crate::metrics::efficiency) answers "what is the compression ratio of *this*
//! adapter", a [`crate::metrics::dashboard::PeftDashboard`] answers the model-wide questions a practitioner actually asks:
//!
//! - How many trainable parameters did PEFT add across every layer, and what fraction of the
//!   full model is that (overall compression ratio)?
//! - What is the mean / min / max effective rank of the adapters, i.e. how much of the budgeted
//!   rank is actually being used (energy-based effective rank vs. the nominal rank)?
//! - Per-layer rows for rendering a table.
//!
//! Everything is computed deterministically from caller-supplied counts and singular-value
//! spectra; there is no RNG and no I/O.

use crate::error::{PeftError, PeftResult};
use crate::metrics::efficiency::{compression_ratio, effective_rank, param_efficiency_ratio};

/// One row of the dashboard: the efficiency profile of a single adapter (typically one layer).
#[derive(Debug, Clone)]
pub struct AdapterReport {
    /// Human-readable identifier for the layer / module this adapter targets.
    pub name: String,
    /// Number of parameters the dense base layer would have (`in_dim · out_dim`).
    pub base_params: usize,
    /// Number of trainable parameters the adapter introduces.
    pub trainable_params: usize,
    /// Nominal (budgeted) rank of the adapter (`0` if rank-free).
    pub nominal_rank: usize,
    /// Energy-based effective rank of the adapter's `ΔW` spectrum.
    pub effective_rank: f32,
}

impl AdapterReport {
    /// Build a row from a singular-value spectrum, deriving the effective rank.
    ///
    /// `singular_values` are the singular values of the adapter's `ΔW` (or of `B·A`); the
    /// effective rank is `(Σσ)² / Σσ²`.
    #[must_use]
    pub fn from_spectrum(
        name: impl Into<String>,
        base_params: usize,
        trainable_params: usize,
        nominal_rank: usize,
        singular_values: &[f32],
    ) -> Self {
        Self {
            name: name.into(),
            base_params,
            trainable_params,
            nominal_rank,
            effective_rank: effective_rank(singular_values),
        }
    }

    /// Per-row compression ratio `base_params / trainable_params` (`0.0` if no trainables).
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        compression_ratio(self.base_params, self.trainable_params)
    }

    /// Fraction of the budgeted rank actually exercised: `effective_rank / nominal_rank`.
    ///
    /// Returns `0.0` when `nominal_rank == 0`. A value near `1.0` means the adapter is using its
    /// full rank budget; a small value means the spectrum is dominated by a few directions and
    /// the rank could likely be reduced.
    #[must_use]
    pub fn rank_utilization(&self) -> f32 {
        if self.nominal_rank == 0 {
            return 0.0;
        }
        self.effective_rank / self.nominal_rank as f32
    }
}

/// Aggregated efficiency dashboard over a collection of [`AdapterReport`] rows.
#[derive(Debug, Clone, Default)]
pub struct PeftDashboard {
    rows: Vec<AdapterReport>,
}

impl PeftDashboard {
    /// Construct an empty dashboard.
    #[must_use]
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    /// Append a report row, returning `self` for chaining.
    #[must_use]
    pub fn with_row(mut self, row: AdapterReport) -> Self {
        self.rows.push(row);
        self
    }

    /// Append a report row in place.
    pub fn push(&mut self, row: AdapterReport) {
        self.rows.push(row);
    }

    /// Borrow the underlying rows.
    #[must_use]
    pub fn rows(&self) -> &[AdapterReport] {
        &self.rows
    }

    /// Number of adapters tracked.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the dashboard holds no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Total trainable parameters summed across every adapter.
    #[must_use]
    pub fn total_trainable_params(&self) -> usize {
        self.rows.iter().map(|r| r.trainable_params).sum()
    }

    /// Total dense base parameters summed across every adapter's target layer.
    #[must_use]
    pub fn total_base_params(&self) -> usize {
        self.rows.iter().map(|r| r.base_params).sum()
    }

    /// Overall compression ratio `Σ base / Σ trainable` across the whole model.
    ///
    /// Returns `0.0` when there are no trainable parameters.
    #[must_use]
    pub fn overall_compression_ratio(&self) -> f32 {
        compression_ratio(self.total_base_params(), self.total_trainable_params())
    }

    /// Overall trainable fraction `Σ trainable / Σ base` (the inverse view of compression).
    ///
    /// Returns `0.0` when there are no base parameters.
    #[must_use]
    pub fn overall_trainable_fraction(&self) -> f32 {
        param_efficiency_ratio(self.total_trainable_params(), self.total_base_params())
    }

    /// Mean effective rank across every adapter.
    ///
    /// # Errors
    ///
    /// [`PeftError::EmptyInput`] when the dashboard has no rows.
    pub fn mean_effective_rank(&self) -> PeftResult<f32> {
        if self.rows.is_empty() {
            return Err(PeftError::EmptyInput);
        }
        let sum: f32 = self.rows.iter().map(|r| r.effective_rank).sum();
        Ok(sum / self.rows.len() as f32)
    }

    /// Minimum and maximum effective rank across every adapter.
    ///
    /// # Errors
    ///
    /// [`PeftError::EmptyInput`] when the dashboard has no rows.
    pub fn effective_rank_range(&self) -> PeftResult<(f32, f32)> {
        if self.rows.is_empty() {
            return Err(PeftError::EmptyInput);
        }
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for r in &self.rows {
            lo = lo.min(r.effective_rank);
            hi = hi.max(r.effective_rank);
        }
        Ok((lo, hi))
    }

    /// Mean rank utilization (`effective / nominal`) across rows that *have* a nominal rank.
    ///
    /// Rank-free adapters (`nominal_rank == 0`) are excluded from the average.
    ///
    /// # Errors
    ///
    /// [`PeftError::EmptyInput`] when no row carries a non-zero nominal rank.
    pub fn mean_rank_utilization(&self) -> PeftResult<f32> {
        let mut sum = 0.0_f32;
        let mut n = 0usize;
        for r in &self.rows {
            if r.nominal_rank > 0 {
                sum += r.rank_utilization();
                n += 1;
            }
        }
        if n == 0 {
            return Err(PeftError::EmptyInput);
        }
        Ok(sum / n as f32)
    }

    /// Render a fixed-width, human-readable table of every row plus the aggregate summary.
    ///
    /// The output is deterministic and dependency-free (no `tabwriter`); it is intended for CLI
    /// dashboards and test snapshots.
    #[must_use]
    pub fn render_table(&self) -> String {
        let mut s = String::new();
        s.push_str(
            "name                 base_params  train_params  comp_ratio  nom_rank  eff_rank  util\n",
        );
        s.push_str(
            "-------------------------------------------------------------------------------------\n",
        );
        for r in &self.rows {
            let name = if r.name.len() > 20 {
                &r.name[..20]
            } else {
                &r.name
            };
            s.push_str(&format!(
                "{name:<20} {:>11} {:>13} {:>11.2} {:>9} {:>9.3} {:>5.3}\n",
                r.base_params,
                r.trainable_params,
                r.compression_ratio(),
                r.nominal_rank,
                r.effective_rank,
                r.rank_utilization(),
            ));
        }
        s.push_str(
            "-------------------------------------------------------------------------------------\n",
        );
        s.push_str(&format!(
            "TOTAL                {:>11} {:>13} {:>11.2}\n",
            self.total_base_params(),
            self.total_trainable_params(),
            self.overall_compression_ratio(),
        ));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-layer LoRA: 1024×1024 base, rank 8 ⇒ 8·(1024+1024)=16384 trainable per layer.
    fn two_layer_dashboard() -> PeftDashboard {
        // Adapter 1: a rank-8 spectrum using ~4 strong directions (eff_rank ≈ 4).
        let spec1 = [4.0_f32, 4.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        // Adapter 2: a rank-8 spectrum using all directions equally (eff_rank = 8).
        let spec2 = [1.0_f32; 8];
        PeftDashboard::new()
            .with_row(AdapterReport::from_spectrum(
                "layer.0.attn",
                1024 * 1024,
                8 * (1024 + 1024),
                8,
                &spec1,
            ))
            .with_row(AdapterReport::from_spectrum(
                "layer.1.attn",
                1024 * 1024,
                8 * (1024 + 1024),
                8,
                &spec2,
            ))
    }

    #[test]
    fn effective_rank_matches_energy_formula() {
        // Four equal non-zero singular values → effective rank exactly 4.
        let row = AdapterReport::from_spectrum("x", 100, 10, 8, &[2.0, 2.0, 2.0, 2.0, 0.0, 0.0]);
        assert!((row.effective_rank - 4.0).abs() < 1e-5);
        // All eight equal → effective rank 8 → utilization 1.0.
        let row2 = AdapterReport::from_spectrum("y", 100, 10, 8, &[1.0_f32; 8]);
        assert!((row2.effective_rank - 8.0).abs() < 1e-5);
        assert!((row2.rank_utilization() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn aggregate_totals_and_compression() {
        let d = two_layer_dashboard();
        assert_eq!(d.len(), 2);
        assert_eq!(d.total_base_params(), 2 * 1024 * 1024);
        assert_eq!(d.total_trainable_params(), 2 * 16384);
        // overall compression = (2·1024²) / (2·16384) = 1024²/16384 = 64.0
        assert!((d.overall_compression_ratio() - 64.0).abs() < 1e-3);
        // trainable fraction is the reciprocal: 1/64.
        assert!((d.overall_trainable_fraction() - 1.0 / 64.0).abs() < 1e-6);
    }

    #[test]
    fn mean_and_range_of_effective_rank() {
        let d = two_layer_dashboard();
        // eff ranks ≈ 4 and 8 → mean 6, range (4, 8).
        let mean = d.mean_effective_rank().expect("non-empty");
        assert!((mean - 6.0).abs() < 1e-4, "mean eff rank {mean}");
        let (lo, hi) = d.effective_rank_range().expect("non-empty");
        assert!((lo - 4.0).abs() < 1e-4 && (hi - 8.0).abs() < 1e-4);
    }

    #[test]
    fn mean_rank_utilization_excludes_rank_free() {
        let mut d = two_layer_dashboard();
        // util = 4/8=0.5 and 8/8=1.0 → mean 0.75.
        let util = d.mean_rank_utilization().expect("has ranked rows");
        assert!((util - 0.75).abs() < 1e-4, "util {util}");
        // Add a rank-free adapter (e.g. BitFit) — it must NOT drag the mean.
        d.push(AdapterReport::from_spectrum("bias", 1000, 1000, 0, &[]));
        let util2 = d.mean_rank_utilization().expect("still has ranked rows");
        assert!((util2 - 0.75).abs() < 1e-4, "rank-free row leaked: {util2}");
    }

    #[test]
    fn empty_dashboard_errors_on_stats() {
        let d = PeftDashboard::new();
        assert!(d.is_empty());
        assert_eq!(d.overall_compression_ratio(), 0.0);
        assert!(matches!(
            d.mean_effective_rank(),
            Err(PeftError::EmptyInput)
        ));
        assert!(matches!(
            d.effective_rank_range(),
            Err(PeftError::EmptyInput)
        ));
        assert!(matches!(
            d.mean_rank_utilization(),
            Err(PeftError::EmptyInput)
        ));
    }

    #[test]
    fn all_rank_free_has_no_utilization() {
        let d = PeftDashboard::new()
            .with_row(AdapterReport::from_spectrum("b0", 100, 100, 0, &[]))
            .with_row(AdapterReport::from_spectrum("b1", 100, 100, 0, &[]));
        assert!(matches!(
            d.mean_rank_utilization(),
            Err(PeftError::EmptyInput)
        ));
    }

    #[test]
    fn render_table_contains_rows_and_total() {
        let table = two_layer_dashboard().render_table();
        assert!(table.contains("layer.0.attn"));
        assert!(table.contains("layer.1.attn"));
        assert!(table.contains("TOTAL"));
        // Header present.
        assert!(table.contains("comp_ratio"));
    }

    #[test]
    fn long_name_is_truncated_safely_on_char_boundary() {
        // 25-char ASCII name → truncated to 20 without panicking.
        let d = PeftDashboard::new().with_row(AdapterReport::from_spectrum(
            "abcdefghijklmnopqrstuvwxy",
            10,
            10,
            4,
            &[1.0; 4],
        ));
        let _ = d.render_table();
    }
}
