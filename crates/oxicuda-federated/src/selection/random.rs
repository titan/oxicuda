//! Random and stratified client selection for federated learning.

use crate::error::{FedError, FedResult};
use crate::handle::LcgRng;

/// Select `k` clients uniformly at random from `[0, n_clients)`.
///
/// Uses Fisher-Yates partial shuffle to select without replacement.
///
/// # Errors
/// Returns `InsufficientClients` if `k > n_clients` or `EmptyClientList` if
/// `n_clients == 0`.
pub fn random_select(n_clients: usize, k: usize, rng: &mut LcgRng) -> FedResult<Vec<usize>> {
    if n_clients == 0 {
        return Err(FedError::EmptyClientList);
    }
    if k > n_clients {
        return Err(FedError::InsufficientClients {
            min: k,
            got: n_clients,
        });
    }

    // Build full index list and partial Fisher-Yates shuffle
    let mut indices: Vec<usize> = (0..n_clients).collect();
    for i in 0..k {
        let j = i + rng.next_usize(n_clients - i);
        indices.swap(i, j);
    }
    Ok(indices[..k].to_vec())
}

/// Select clients using stratified sampling.
///
/// Clients are divided into `n_strata` strata of approximately equal size.
/// Exactly `k_per_stratum` clients are selected from each stratum.
///
/// # Arguments
/// - `n_clients` — total number of clients
/// - `n_strata` — number of strata
/// - `k_per_stratum` — clients selected per stratum
/// - `rng` — random number generator
///
/// # Errors
/// Returns `EmptyClientList` if `n_clients == 0`, `InsufficientClients` if
/// any stratum has fewer clients than `k_per_stratum`.
pub fn stratified_select(
    n_clients: usize,
    n_strata: usize,
    k_per_stratum: usize,
    rng: &mut LcgRng,
) -> FedResult<Vec<usize>> {
    if n_clients == 0 {
        return Err(FedError::EmptyClientList);
    }
    if n_strata == 0 || k_per_stratum == 0 {
        return Ok(Vec::new());
    }

    let stratum_size = n_clients / n_strata;
    let remainder = n_clients % n_strata;
    let mut selected = Vec::new();

    let mut start = 0;
    for s in 0..n_strata {
        // Strata may be unequal in size when n_clients is not divisible by n_strata
        let sz = stratum_size + if s < remainder { 1 } else { 0 };
        if sz < k_per_stratum {
            return Err(FedError::InsufficientClients {
                min: k_per_stratum,
                got: sz,
            });
        }
        // Randomly select k_per_stratum from this stratum
        let mut stratum_indices: Vec<usize> = (start..start + sz).collect();
        for i in 0..k_per_stratum {
            let j = i + rng.next_usize(sz - i);
            stratum_indices.swap(i, j);
        }
        selected.extend_from_slice(&stratum_indices[..k_per_stratum]);
        start += sz;
    }

    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_select_count() {
        let mut rng = LcgRng::new(42);
        let selected = random_select(20, 5, &mut rng).expect("test invariant: valid random select");
        assert_eq!(selected.len(), 5);
    }

    #[test]
    fn random_select_unique() {
        let mut rng = LcgRng::new(7);
        let selected =
            random_select(100, 30, &mut rng).expect("test invariant: valid random select");
        let mut sorted = selected.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 30, "all selected IDs should be unique");
    }

    #[test]
    fn random_select_in_range() {
        let mut rng = LcgRng::new(99);
        let n = 50;
        let selected = random_select(n, 10, &mut rng).expect("test invariant: valid random select");
        for &id in &selected {
            assert!(id < n, "selected ID {id} out of range [0, {n})");
        }
    }

    #[test]
    fn random_select_empty_error() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            random_select(0, 1, &mut rng),
            Err(FedError::EmptyClientList)
        ));
    }

    #[test]
    fn random_select_k_exceeds_n() {
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            random_select(5, 10, &mut rng),
            Err(FedError::InsufficientClients { .. })
        ));
    }

    #[test]
    fn stratified_select_count() {
        let mut rng = LcgRng::new(42);
        let selected =
            stratified_select(20, 4, 2, &mut rng).expect("test invariant: valid stratified select");
        assert_eq!(selected.len(), 4 * 2);
    }

    #[test]
    fn stratified_select_zero_k_per_stratum() {
        let mut rng = LcgRng::new(1);
        let selected = stratified_select(20, 4, 0, &mut rng)
            .expect("test invariant: valid zero k_per_stratum");
        assert!(selected.is_empty());
    }
}
