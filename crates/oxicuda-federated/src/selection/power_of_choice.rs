//! Power-of-Choice and related adaptive client-selection strategies.
//!
//! Cho, Wang & Joshi, "Client Selection in Federated Learning: Convergence
//! Analysis and Power-of-Choice Selection Strategies" (2020).
//!
//! The Power-of-Choice strategy first samples a candidate set of `d` clients
//! with probability proportional to their local data size (data-size weighting
//! is the natural FedAvg aggregation weight), and then selects the `m` clients
//! with the highest current local loss from that candidate set.  Biasing the
//! selection toward high-loss clients provably accelerates convergence on
//! heterogeneous data compared with uniform sampling.
//!
//! Weighted sampling without replacement is performed via the
//! Efraimidis-Spirakis A-Res reservoir scheme (Efraimidis & Spirakis,
//! "Weighted random sampling with a reservoir", Information Processing
//! Letters 2006): each item draws a key `u^{1/w}` and the items with the
//! largest keys form the sample.

use crate::error::{FedError, FedResult};
use crate::handle::LcgRng;

/// Configuration for adaptive client selection.
#[derive(Debug, Clone, Copy)]
pub struct PowerOfChoiceConfig {
    /// Total number of clients in the population.
    pub n_clients: usize,
    /// Number of clients to select for the round.
    pub n_select: usize,
    /// Candidate-set multiplier: the candidate pool size is
    /// `min(candidate_factor * n_select, n_clients)`.
    pub candidate_factor: usize,
}

/// Which selection strategy to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Power-of-Choice: weighted candidate sampling, then top-loss selection.
    PowerOfChoice,
    /// Pure loss-based: globally select the highest-loss clients.
    LossBased,
    /// Power-of-Choice restricted to the currently-available clients.
    AvailabilityAware,
    /// Uniform sampling without replacement (data-size agnostic).
    Random,
}

/// Adaptive client selection entry point.
pub struct PowerOfChoice;

impl PowerOfChoice {
    /// Validate the shared inputs common to every strategy.
    fn validate(
        data_sizes: &[usize],
        losses: &[f32],
        availability: Option<&[bool]>,
        cfg: &PowerOfChoiceConfig,
    ) -> FedResult<()> {
        if cfg.n_clients == 0 {
            return Err(FedError::EmptyClientList);
        }
        if cfg.n_select == 0 {
            return Err(FedError::Internal(
                "power-of-choice: n_select must be >= 1".into(),
            ));
        }
        if cfg.n_select > cfg.n_clients {
            return Err(FedError::InsufficientClients {
                min: cfg.n_select,
                got: cfg.n_clients,
            });
        }
        if cfg.candidate_factor == 0 {
            return Err(FedError::Internal(
                "power-of-choice: candidate_factor must be >= 1".into(),
            ));
        }
        if data_sizes.len() != cfg.n_clients {
            return Err(FedError::DimensionMismatch {
                expected: cfg.n_clients,
                got: data_sizes.len(),
            });
        }
        if losses.len() != cfg.n_clients {
            return Err(FedError::DimensionMismatch {
                expected: cfg.n_clients,
                got: losses.len(),
            });
        }
        if let Some(avail) = availability {
            if avail.len() != cfg.n_clients {
                return Err(FedError::DimensionMismatch {
                    expected: cfg.n_clients,
                    got: avail.len(),
                });
            }
        }
        Ok(())
    }

    /// Select `n_select` clients according to `strategy`.
    ///
    /// - `data_sizes` — per-client sample counts; selection weights are
    ///   proportional to these sizes.
    /// - `losses` — current per-client local loss (higher loss is preferred).
    /// - `availability` — optional per-client availability mask (used by
    ///   [`SelectionStrategy::AvailabilityAware`]).
    /// - `cfg` — selection configuration.
    /// - `rng` — deterministic RNG used for the weighted candidate sampling.
    ///
    /// # Errors
    /// Returns an error if validation fails (see [`PowerOfChoiceConfig`]), if
    /// the total positive weight is zero, or — for
    /// [`SelectionStrategy::AvailabilityAware`] — if fewer than `n_select`
    /// clients are available.
    pub fn select(
        strategy: SelectionStrategy,
        data_sizes: &[usize],
        losses: &[f32],
        availability: Option<&[bool]>,
        cfg: &PowerOfChoiceConfig,
        rng: &mut LcgRng,
    ) -> FedResult<Vec<usize>> {
        Self::validate(data_sizes, losses, availability, cfg)?;
        match strategy {
            SelectionStrategy::PowerOfChoice => {
                Self::select_power_of_choice(data_sizes, losses, cfg, rng)
            }
            SelectionStrategy::LossBased => Self::select_loss_based(losses, cfg.n_select),
            SelectionStrategy::AvailabilityAware => {
                Self::select_availability_aware(data_sizes, losses, availability, cfg, rng)
            }
            SelectionStrategy::Random => Self::select_random(cfg.n_clients, cfg.n_select, rng),
        }
    }

    /// Power-of-Choice over the full population: candidate pool by data size,
    /// then top `n_select` by loss.
    fn select_power_of_choice(
        data_sizes: &[usize],
        losses: &[f32],
        cfg: &PowerOfChoiceConfig,
        rng: &mut LcgRng,
    ) -> FedResult<Vec<usize>> {
        let d = (cfg.candidate_factor.saturating_mul(cfg.n_select)).min(cfg.n_clients);
        let candidates = Self::sample_candidates(data_sizes, d, rng)?;
        Ok(Self::top_loss_from_candidates(
            &candidates,
            losses,
            cfg.n_select,
        ))
    }

    /// Sort `candidates` by descending loss (stable; ties broken by ascending
    /// index) and keep the first `n_select`.
    fn top_loss_from_candidates(
        candidates: &[usize],
        losses: &[f32],
        n_select: usize,
    ) -> Vec<usize> {
        let mut cand = candidates.to_vec();
        cand.sort_by(|&a, &b| {
            losses[b]
                .partial_cmp(&losses[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        cand.truncate(n_select);
        cand
    }

    /// Globally select the `n_select` highest-loss clients (ties by index).
    fn select_loss_based(losses: &[f32], n_select: usize) -> FedResult<Vec<usize>> {
        let mut idx: Vec<usize> = (0..losses.len()).collect();
        idx.sort_by(|&a, &b| {
            losses[b]
                .partial_cmp(&losses[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        idx.truncate(n_select);
        Ok(idx)
    }

    /// Power-of-Choice restricted to available clients.
    fn select_availability_aware(
        data_sizes: &[usize],
        losses: &[f32],
        availability: Option<&[bool]>,
        cfg: &PowerOfChoiceConfig,
        rng: &mut LcgRng,
    ) -> FedResult<Vec<usize>> {
        let avail = availability.unwrap_or(&[]);
        let available: Vec<usize> = (0..cfg.n_clients)
            .filter(|&i| avail.get(i).copied().unwrap_or(false))
            .collect();
        if available.len() < cfg.n_select {
            return Err(FedError::InsufficientClients {
                min: cfg.n_select,
                got: available.len(),
            });
        }
        // Restrict data sizes / losses to the available sub-population, run the
        // weighted candidate sampler over it, then re-map back to global ids.
        let sub_sizes: Vec<usize> = available.iter().map(|&i| data_sizes[i]).collect();
        let d = (cfg.candidate_factor.saturating_mul(cfg.n_select)).min(available.len());
        let local_candidates = Self::sample_candidates(&sub_sizes, d, rng)?;
        let global_candidates: Vec<usize> =
            local_candidates.iter().map(|&l| available[l]).collect();
        Ok(Self::top_loss_from_candidates(
            &global_candidates,
            losses,
            cfg.n_select,
        ))
    }

    /// Uniform sampling without replacement (equal weights → A-Res reduces to
    /// a uniform reservoir).
    fn select_random(n_clients: usize, n_select: usize, rng: &mut LcgRng) -> FedResult<Vec<usize>> {
        let equal = vec![1_usize; n_clients];
        Self::sample_candidates(&equal, n_select, rng)
    }

    /// Sample `d` distinct clients with probability proportional to
    /// `data_sizes` using the Efraimidis-Spirakis A-Res key scheme.
    ///
    /// Each client `i` draws `key_i = u_i^{1/w_i}` with `u_i ∈ (0, 1)` and
    /// `w_i = data_sizes[i]`; the `d` largest keys are selected.  Clients with
    /// zero weight receive key `0` and are only chosen to backfill when there
    /// are fewer than `d` positive-weight clients (deterministically by index).
    ///
    /// # Errors
    /// Returns [`FedError::EmptyClientList`] if `data_sizes` is empty,
    /// [`FedError::InsufficientClients`] if `d` exceeds the client count, and
    /// [`FedError::InvalidWeight`] if the total positive weight is zero.
    pub fn sample_candidates(
        data_sizes: &[usize],
        d: usize,
        rng: &mut LcgRng,
    ) -> FedResult<Vec<usize>> {
        let n = data_sizes.len();
        if n == 0 {
            return Err(FedError::EmptyClientList);
        }
        if d > n {
            return Err(FedError::InsufficientClients { min: d, got: n });
        }
        let positive = data_sizes.iter().filter(|&&w| w > 0).count();
        if positive == 0 {
            return Err(FedError::InvalidWeight { weight: 0.0 });
        }
        if d == 0 {
            return Ok(Vec::new());
        }

        // Compute an A-Res key per client.  Draw u for every client (in index
        // order) so the result is deterministic for a given seeded RNG.
        let mut keys: Vec<(usize, f32)> = Vec::with_capacity(n);
        for (i, &w) in data_sizes.iter().enumerate() {
            let u = rng.next_f32().max(1e-12);
            let key = if w == 0 {
                // Zero weight ⇒ key 0 ⇒ only used to backfill.
                0.0_f32
            } else {
                // key = u^{1/w} = exp(ln(u) / w).
                (u.ln() / w as f32).exp()
            };
            keys.push((i, key));
        }

        // Largest keys first; ties broken by ascending index for determinism.
        // This also places the zero-weight (key == 0) clients last, so they
        // only enter the sample when fewer than d positive-weight clients exist.
        let kth = d - 1;
        keys.select_nth_unstable_by(kth, |&(ia, ka), &(ib, kb)| {
            kb.partial_cmp(&ka)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(ia.cmp(&ib))
        });
        let mut chosen: Vec<usize> = keys[..d].iter().map(|&(i, _)| i).collect();
        chosen.sort_unstable();
        Ok(chosen)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n_clients: usize, n_select: usize, candidate_factor: usize) -> PowerOfChoiceConfig {
        PowerOfChoiceConfig {
            n_clients,
            n_select,
            candidate_factor,
        }
    }

    #[test]
    fn loss_based_picks_highest_loss_clients() {
        let mut rng = LcgRng::new(1);
        let data = vec![1_usize; 5];
        let losses = vec![0.1_f32, 0.9, 0.3, 0.7, 0.5];
        let c = cfg(5, 2, 1);
        let sel = PowerOfChoice::select(
            SelectionStrategy::LossBased,
            &data,
            &losses,
            None,
            &c,
            &mut rng,
        )
        .expect("test invariant: valid selection");
        // Highest losses are at indices 1 (0.9) and 3 (0.7).
        assert_eq!(sel, vec![1, 3]);
    }

    #[test]
    fn power_of_choice_returns_distinct_in_range() {
        let mut rng = LcgRng::new(42);
        let data = vec![10_usize, 20, 30, 40, 50, 60, 70, 80];
        let losses = vec![0.5_f32; 8];
        let c = cfg(8, 3, 2);
        let sel = PowerOfChoice::select(
            SelectionStrategy::PowerOfChoice,
            &data,
            &losses,
            None,
            &c,
            &mut rng,
        )
        .expect("test invariant: valid selection");
        assert_eq!(sel.len(), 3);
        let mut sorted = sel.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "selected ids must be distinct");
        for &i in &sel {
            assert!(i < 8, "id {i} out of range");
        }
    }

    #[test]
    fn power_of_choice_full_candidate_pool_equals_loss_based() {
        // candidate_factor * n_select >= n_clients ⇒ d == n_clients ⇒ the
        // candidate set is everyone ⇒ Power-of-Choice == LossBased.
        let mut rng_a = LcgRng::new(7);
        let mut rng_b = LcgRng::new(7);
        let data = vec![3_usize, 1, 4, 1, 5, 9];
        let losses = vec![0.2_f32, 0.8, 0.1, 0.9, 0.5, 0.3];
        let c = cfg(6, 3, 6);
        let poc = PowerOfChoice::select(
            SelectionStrategy::PowerOfChoice,
            &data,
            &losses,
            None,
            &c,
            &mut rng_a,
        )
        .expect("test invariant: valid selection");
        let lb = PowerOfChoice::select(
            SelectionStrategy::LossBased,
            &data,
            &losses,
            None,
            &c,
            &mut rng_b,
        )
        .expect("test invariant: valid selection");
        assert_eq!(poc, lb);
    }

    #[test]
    fn availability_aware_never_selects_unavailable() {
        let mut rng = LcgRng::new(123);
        let data = vec![5_usize; 8];
        let losses = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let avail = vec![false, true, false, true, true, false, true, false];
        let c = cfg(8, 2, 2);
        let sel = PowerOfChoice::select(
            SelectionStrategy::AvailabilityAware,
            &data,
            &losses,
            Some(&avail),
            &c,
            &mut rng,
        )
        .expect("test invariant: valid selection");
        assert_eq!(sel.len(), 2);
        for &i in &sel {
            assert!(avail[i], "selected unavailable client {i}");
        }
    }

    #[test]
    fn availability_aware_prefers_high_loss_available() {
        let mut rng = LcgRng::new(9);
        let data = vec![1_usize; 6];
        let losses = vec![0.9_f32, 0.1, 0.8, 0.2, 0.7, 0.05];
        // Only odd indices available: losses 0.1, 0.2, 0.05 → top two are 3,1.
        let avail = vec![false, true, false, true, false, true];
        let c = cfg(6, 2, 6); // full candidate pool over the available set
        let sel = PowerOfChoice::select(
            SelectionStrategy::AvailabilityAware,
            &data,
            &losses,
            Some(&avail),
            &c,
            &mut rng,
        )
        .expect("test invariant: valid selection");
        assert_eq!(sel, vec![3, 1]);
    }

    #[test]
    fn random_returns_distinct() {
        let mut rng = LcgRng::new(55);
        let data = vec![1_usize; 10];
        let losses = vec![0.0_f32; 10];
        let c = cfg(10, 4, 1);
        let sel = PowerOfChoice::select(
            SelectionStrategy::Random,
            &data,
            &losses,
            None,
            &c,
            &mut rng,
        )
        .expect("test invariant: valid selection");
        assert_eq!(sel.len(), 4);
        let mut sorted = sel.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 4);
    }

    #[test]
    fn sample_candidates_returns_d_distinct_in_range() {
        let mut rng = LcgRng::new(2024);
        let data = vec![10_usize, 20, 30, 40, 50];
        let cand = PowerOfChoice::sample_candidates(&data, 3, &mut rng)
            .expect("test invariant: valid candidate sampling");
        assert_eq!(cand.len(), 3);
        let mut sorted = cand.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 3);
        for &i in &cand {
            assert!(i < 5);
        }
    }

    #[test]
    fn weighted_sampling_biases_toward_large_data() {
        // One client (index 0) has 100x the data: over many seeded single
        // draws it should be picked far more often than uniform (1/5).
        let data = vec![100_usize, 1, 1, 1, 1];
        let mut count_zero = 0;
        let draws = 200;
        for s in 0..draws {
            let mut rng = LcgRng::new(s as u64 + 1);
            let cand = PowerOfChoice::sample_candidates(&data, 1, &mut rng)
                .expect("test invariant: valid candidate sampling");
            if cand == vec![0] {
                count_zero += 1;
            }
        }
        // With weight 100 vs 4*1, P(pick 0) ≈ 100/104 ≈ 0.96. Require a clear
        // majority to keep the test robust to RNG quirks.
        assert!(
            count_zero > draws / 2,
            "expected heavy bias toward client 0, got {count_zero}/{draws}"
        );
    }

    #[test]
    fn n_select_equals_n_clients_is_permutation() {
        let mut rng = LcgRng::new(321);
        let data = vec![3_usize, 7, 2, 9, 4];
        let losses = vec![0.5_f32, 0.1, 0.9, 0.2, 0.7];
        let c = cfg(5, 5, 2);
        let sel = PowerOfChoice::select(
            SelectionStrategy::PowerOfChoice,
            &data,
            &losses,
            None,
            &c,
            &mut rng,
        )
        .expect("test invariant: valid selection");
        let mut sorted = sel.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn selection_is_deterministic_given_seed() {
        let data = vec![5_usize, 1, 8, 3, 6, 2, 7, 4];
        let losses = vec![0.3_f32, 0.6, 0.1, 0.9, 0.2, 0.8, 0.4, 0.5];
        let c = cfg(8, 3, 2);
        let mut rng_a = LcgRng::new(98765);
        let mut rng_b = LcgRng::new(98765);
        let a = PowerOfChoice::select(
            SelectionStrategy::PowerOfChoice,
            &data,
            &losses,
            None,
            &c,
            &mut rng_a,
        )
        .expect("test invariant: valid selection");
        let b = PowerOfChoice::select(
            SelectionStrategy::PowerOfChoice,
            &data,
            &losses,
            None,
            &c,
            &mut rng_b,
        )
        .expect("test invariant: valid selection");
        assert_eq!(a, b);
    }

    #[test]
    fn err_n_select_exceeds_n_clients() {
        let mut rng = LcgRng::new(1);
        let data = vec![1_usize; 4];
        let losses = vec![0.0_f32; 4];
        let c = cfg(4, 5, 1);
        assert!(matches!(
            PowerOfChoice::select(
                SelectionStrategy::LossBased,
                &data,
                &losses,
                None,
                &c,
                &mut rng
            ),
            Err(FedError::InsufficientClients { .. })
        ));
    }

    #[test]
    fn err_n_select_zero() {
        let mut rng = LcgRng::new(1);
        let data = vec![1_usize; 4];
        let losses = vec![0.0_f32; 4];
        let c = cfg(4, 0, 1);
        assert!(matches!(
            PowerOfChoice::select(
                SelectionStrategy::LossBased,
                &data,
                &losses,
                None,
                &c,
                &mut rng
            ),
            Err(FedError::Internal(_))
        ));
    }

    #[test]
    fn err_n_clients_zero() {
        let mut rng = LcgRng::new(1);
        let data: Vec<usize> = Vec::new();
        let losses: Vec<f32> = Vec::new();
        let c = cfg(0, 1, 1);
        assert!(matches!(
            PowerOfChoice::select(
                SelectionStrategy::Random,
                &data,
                &losses,
                None,
                &c,
                &mut rng
            ),
            Err(FedError::EmptyClientList)
        ));
    }

    #[test]
    fn err_data_loss_length_mismatch() {
        let mut rng = LcgRng::new(1);
        let data = vec![1_usize; 4];
        let losses = vec![0.0_f32; 3];
        let c = cfg(4, 2, 1);
        assert!(matches!(
            PowerOfChoice::select(
                SelectionStrategy::LossBased,
                &data,
                &losses,
                None,
                &c,
                &mut rng
            ),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_availability_length_mismatch() {
        let mut rng = LcgRng::new(1);
        let data = vec![1_usize; 4];
        let losses = vec![0.0_f32; 4];
        let avail = vec![true, false, true];
        let c = cfg(4, 2, 1);
        assert!(matches!(
            PowerOfChoice::select(
                SelectionStrategy::AvailabilityAware,
                &data,
                &losses,
                Some(&avail),
                &c,
                &mut rng
            ),
            Err(FedError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_fewer_than_n_select_available() {
        let mut rng = LcgRng::new(1);
        let data = vec![1_usize; 5];
        let losses = vec![0.0_f32; 5];
        let avail = vec![true, false, false, true, false];
        let c = cfg(5, 3, 1);
        assert!(matches!(
            PowerOfChoice::select(
                SelectionStrategy::AvailabilityAware,
                &data,
                &losses,
                Some(&avail),
                &c,
                &mut rng
            ),
            Err(FedError::InsufficientClients { .. })
        ));
    }

    #[test]
    fn err_candidate_factor_zero() {
        let mut rng = LcgRng::new(1);
        let data = vec![1_usize; 4];
        let losses = vec![0.0_f32; 4];
        let c = cfg(4, 2, 0);
        assert!(matches!(
            PowerOfChoice::select(
                SelectionStrategy::PowerOfChoice,
                &data,
                &losses,
                None,
                &c,
                &mut rng
            ),
            Err(FedError::Internal(_))
        ));
    }

    #[test]
    fn err_all_zero_data_sizes() {
        let mut rng = LcgRng::new(1);
        let data = vec![0_usize; 4];
        let losses = vec![0.0_f32; 4];
        let c = cfg(4, 2, 1);
        assert!(matches!(
            PowerOfChoice::select(
                SelectionStrategy::PowerOfChoice,
                &data,
                &losses,
                None,
                &c,
                &mut rng
            ),
            Err(FedError::InvalidWeight { .. })
        ));
    }

    #[test]
    fn sample_candidates_d_zero_returns_empty() {
        let mut rng = LcgRng::new(1);
        let data = vec![1_usize, 2, 3];
        let cand = PowerOfChoice::sample_candidates(&data, 0, &mut rng)
            .expect("test invariant: valid candidate sampling");
        assert!(cand.is_empty());
    }

    #[test]
    fn sample_candidates_d_exceeds_n_errors() {
        let mut rng = LcgRng::new(1);
        let data = vec![1_usize, 2, 3];
        assert!(matches!(
            PowerOfChoice::sample_candidates(&data, 4, &mut rng),
            Err(FedError::InsufficientClients { .. })
        ));
    }

    #[test]
    fn sample_candidates_empty_errors() {
        let mut rng = LcgRng::new(1);
        let data: Vec<usize> = Vec::new();
        assert!(matches!(
            PowerOfChoice::sample_candidates(&data, 0, &mut rng),
            Err(FedError::EmptyClientList)
        ));
    }
}
