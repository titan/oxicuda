//! Sharded experience-replay buffer for multi-device continual learning.
//!
//! In a data-parallel / multi-GPU setting the global replay reservoir is split
//! across `n_shards` independent shards, one per device, so that buffer storage
//! and reservoir-sampling work are distributed instead of duplicated. This
//! module implements the **device-agnostic sharding algorithm** — partitioning,
//! per-shard reservoir maintenance and balanced cross-shard batch retrieval —
//! which is the part that is fully testable on the CPU. Each shard carries a
//! `device` ordinal as placement metadata so a launcher (`oxicuda-launch`) can
//! map shard `s` onto its GPU; the actual on-device kernels and NCCL-style
//! all-gather are out of scope here (and require GPU hardware).
//!
//! ## Algorithm
//!
//! A new sample is routed to exactly one shard by a *shard-assignment policy*:
//!
//! - [`ShardPolicy::RoundRobin`]: shard = `n_routed mod n_shards`. Gives an
//!   exactly balanced load independent of the data, the standard choice for a
//!   uniformly-shuffled stream.
//! - [`ShardPolicy::HashLabel`]: shard = `hash(label) mod n_shards`. Keeps every
//!   instance of a class on the same device (label-affinity), which matters when
//!   downstream gradients are reduced per-class.
//!
//! Within its shard the sample is admitted by **Vitter's reservoir algorithm R**
//! (identical to [`crate::replay::er::er_add`]): once a shard's local reservoir
//! of capacity `shard_capacity` is full, a new arrival evicts a uniformly random
//! slot with probability `shard_capacity / n_seen_local`. Because the union of
//! independent uniform reservoirs over disjoint sub-streams is *not* in general a
//! uniform reservoir of the whole stream, balanced retrieval draws an equal
//! quota from each non-empty shard, which keeps device utilisation even and the
//! replayed mini-batch representative across shards.

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

/// Policy deciding which shard a new sample is routed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardPolicy {
    /// Assign shards cyclically by arrival order (exactly balanced load).
    RoundRobin,
    /// Assign shards by `hash(label) mod n_shards` (per-class affinity).
    HashLabel,
}

/// One per-device shard: an independent reservoir of fixed capacity.
#[derive(Debug, Clone)]
pub struct ReplayShard {
    /// Device ordinal this shard is placed on (placement metadata).
    pub device: u32,
    /// Stored feature vectors (length `<= capacity`).
    pub data: Vec<Vec<f32>>,
    /// Corresponding labels.
    pub labels: Vec<u32>,
    /// Per-shard reservoir capacity.
    pub capacity: usize,
    /// Number of samples routed to this shard so far (Vitter `n`).
    pub n_seen: usize,
}

impl ReplayShard {
    /// Current number of stored items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True if the shard holds no items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Configuration for a [`ShardedReplayBuffer`].
#[derive(Debug, Clone)]
pub struct ShardedReplayConfig {
    /// Number of shards / devices. Must be `>= 1`.
    pub n_shards: usize,
    /// Total capacity across all shards. Must be `>= n_shards`. The per-shard
    /// capacity is `total_capacity / n_shards`, remainder given to low shards.
    pub total_capacity: usize,
    /// Shard-assignment policy.
    pub policy: ShardPolicy,
}

impl ShardedReplayConfig {
    /// Create and validate the configuration.
    pub fn new(
        n_shards: usize,
        total_capacity: usize,
        policy: ShardPolicy,
    ) -> ContinualResult<Self> {
        let cfg = Self {
            n_shards,
            total_capacity,
            policy,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate the configuration fields.
    pub fn validate(&self) -> ContinualResult<()> {
        if self.n_shards == 0 {
            return Err(ContinualError::Internal(
                "n_shards must be >= 1".to_string(),
            ));
        }
        if self.total_capacity < self.n_shards {
            return Err(ContinualError::BufferCapacityTooSmall);
        }
        Ok(())
    }
}

/// Hash a label into a shard index (FNV-1a over the 4 label bytes).
fn hash_label(label: u32) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in label.to_le_bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A replay buffer partitioned into independent per-device reservoir shards.
#[derive(Debug, Clone)]
pub struct ShardedReplayBuffer {
    /// The shards (length `n_shards`).
    pub shards: Vec<ReplayShard>,
    /// Configuration.
    pub cfg: ShardedReplayConfig,
    /// Total number of samples routed across all shards.
    pub n_routed: usize,
}

/// Create a new sharded replay buffer.
///
/// Per-shard capacity is `total_capacity / n_shards`; the first
/// `total_capacity % n_shards` shards receive one extra slot so that the sum of
/// shard capacities is exactly `total_capacity`. Shard `s` is assigned device
/// ordinal `base_device + s`.
pub fn sharded_buffer_new(
    cfg: ShardedReplayConfig,
    base_device: u32,
) -> ContinualResult<ShardedReplayBuffer> {
    cfg.validate()?;
    let base = cfg.total_capacity / cfg.n_shards;
    let rem = cfg.total_capacity % cfg.n_shards;
    let mut shards = Vec::with_capacity(cfg.n_shards);
    for s in 0..cfg.n_shards {
        let capacity = base + usize::from(s < rem);
        shards.push(ReplayShard {
            device: base_device + s as u32,
            data: Vec::with_capacity(capacity),
            labels: Vec::with_capacity(capacity),
            capacity,
            n_seen: 0,
        });
    }
    Ok(ShardedReplayBuffer {
        shards,
        cfg,
        n_routed: 0,
    })
}

/// Resolve the destination shard index for a sample under the active policy.
fn shard_for(buf: &ShardedReplayBuffer, label: u32) -> usize {
    match buf.cfg.policy {
        ShardPolicy::RoundRobin => buf.n_routed % buf.cfg.n_shards,
        ShardPolicy::HashLabel => (hash_label(label) % buf.cfg.n_shards as u64) as usize,
    }
}

/// Route a sample into the appropriate shard and apply reservoir sampling there.
///
/// Returns the index of the shard the sample was routed to.
pub fn sharded_add(
    buf: &mut ShardedReplayBuffer,
    sample: Vec<f32>,
    label: u32,
    rng: &mut LcgRng,
) -> usize {
    let s = shard_for(buf, label);
    let shard = &mut buf.shards[s];
    let n = shard.n_seen;
    if n < shard.capacity {
        shard.data.push(sample);
        shard.labels.push(label);
    } else {
        // Vitter algorithm R: evict slot r with prob capacity/(n+1).
        let r = rng.next_usize(n + 1);
        if r < shard.capacity {
            shard.data[r] = sample;
            shard.labels[r] = label;
        }
    }
    shard.n_seen += 1;
    buf.n_routed += 1;
    s
}

/// Total number of items currently stored across all shards.
#[must_use]
pub fn sharded_len(buf: &ShardedReplayBuffer) -> usize {
    buf.shards.iter().map(ReplayShard::len).sum()
}

/// Sample a balanced mini-batch by drawing an equal quota from each non-empty
/// shard (round-robin top-up for any remainder), without replacement *within* a
/// shard.
///
/// Returns `(features, labels, shard_of_each)` where `shard_of_each[i]` is the
/// shard that item `i` came from — useful for routing the replayed gradient back
/// to the originating device. Returns [`ContinualError::BufferEmpty`] if every
/// shard is empty, or [`ContinualError::BatchExceedsBuffer`] if `n` exceeds the
/// total stored count.
#[allow(clippy::type_complexity)]
pub fn sharded_sample_balanced(
    buf: &ShardedReplayBuffer,
    n: usize,
    rng: &mut LcgRng,
) -> ContinualResult<(Vec<Vec<f32>>, Vec<u32>, Vec<usize>)> {
    let total = sharded_len(buf);
    if total == 0 {
        return Err(ContinualError::BufferEmpty);
    }
    if n > total {
        return Err(ContinualError::BatchExceedsBuffer {
            requested: n,
            available: total,
        });
    }

    // Non-empty shard indices.
    let live: Vec<usize> = (0..buf.shards.len())
        .filter(|&s| !buf.shards[s].is_empty())
        .collect();

    // Pre-shuffle each live shard's indices to sample without replacement.
    let mut perms: Vec<Vec<usize>> = Vec::with_capacity(buf.shards.len());
    for shard in &buf.shards {
        let mut idx: Vec<usize> = (0..shard.len()).collect();
        rng.shuffle(&mut idx);
        perms.push(idx);
    }
    // How many already taken from each shard.
    let mut taken = vec![0usize; buf.shards.len()];

    let mut feats = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    let mut origin = Vec::with_capacity(n);

    // Balanced round-robin over live shards: each pass takes one item from every
    // shard that still has capacity, which equalises the per-shard quota and
    // gracefully drains smaller shards.
    let mut cursor = 0usize;
    while feats.len() < n {
        let s = live[cursor % live.len()];
        cursor += 1;
        let shard = &buf.shards[s];
        if taken[s] < shard.len() {
            let local = perms[s][taken[s]];
            taken[s] += 1;
            feats.push(shard.data[local].clone());
            labels.push(shard.labels[local]);
            origin.push(s);
        }
        // If we have cycled a full round and pulled nothing new, every live
        // shard is exhausted (cannot happen while feats.len() < n <= total, but
        // guard against an infinite loop defensively).
        if cursor % live.len() == 0 {
            let remaining: usize = live.iter().map(|&s| buf.shards[s].len() - taken[s]).sum();
            if remaining == 0 {
                break;
            }
        }
    }

    Ok((feats, labels, origin))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_rr(n_shards: usize, cap: usize) -> ShardedReplayConfig {
        ShardedReplayConfig::new(n_shards, cap, ShardPolicy::RoundRobin)
            .expect("valid sharded config")
    }

    #[test]
    fn capacities_sum_to_total() {
        // 17 / 4 = 4 r1 → capacities {5,4,4,4} = 17.
        let buf = sharded_buffer_new(cfg_rr(4, 17), 0).expect("buffer should build");
        let sum: usize = buf.shards.iter().map(|s| s.capacity).sum();
        assert_eq!(sum, 17);
        assert_eq!(buf.shards[0].capacity, 5);
        assert_eq!(buf.shards[1].capacity, 4);
    }

    #[test]
    fn devices_are_assigned_sequentially() {
        let buf = sharded_buffer_new(cfg_rr(3, 12), 4).expect("buffer should build");
        let devices: Vec<u32> = buf.shards.iter().map(|s| s.device).collect();
        assert_eq!(devices, vec![4, 5, 6]);
    }

    #[test]
    fn each_shard_bounded_by_its_capacity() {
        let mut rng = LcgRng::new(42);
        let mut buf = sharded_buffer_new(cfg_rr(4, 16), 0).expect("buffer should build");
        for i in 0..1000usize {
            sharded_add(&mut buf, vec![i as f32; 3], (i % 7) as u32, &mut rng);
        }
        for shard in &buf.shards {
            assert!(
                shard.len() <= shard.capacity,
                "shard exceeded its capacity ({} > {})",
                shard.len(),
                shard.capacity
            );
        }
        assert_eq!(buf.n_routed, 1000);
    }

    #[test]
    fn round_robin_balances_n_seen_exactly() {
        let mut rng = LcgRng::new(7);
        let mut buf = sharded_buffer_new(cfg_rr(4, 40), 0).expect("buffer should build");
        for i in 0..400usize {
            sharded_add(&mut buf, vec![i as f32], (i % 10) as u32, &mut rng);
        }
        // 400 routed round-robin over 4 shards → exactly 100 each.
        for shard in &buf.shards {
            assert_eq!(shard.n_seen, 100);
        }
    }

    #[test]
    fn hash_label_keeps_class_on_one_shard() {
        let cfg =
            ShardedReplayConfig::new(4, 64, ShardPolicy::HashLabel).expect("valid sharded config");
        let mut rng = LcgRng::new(1);
        let mut buf = sharded_buffer_new(cfg, 0).expect("buffer should build");
        // Track which shard each label lands on.
        let mut label_shard: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for i in 0..500usize {
            let label = (i % 8) as u32;
            let s = sharded_add(&mut buf, vec![i as f32], label, &mut rng);
            match label_shard.get(&label) {
                Some(&prev) => assert_eq!(prev, s, "label {label} jumped shards"),
                None => {
                    label_shard.insert(label, s);
                }
            }
        }
        // Every stored label must live on its assigned shard.
        for (s, shard) in buf.shards.iter().enumerate() {
            for &label in &shard.labels {
                assert_eq!(*label_shard.get(&label).expect("seen"), s);
            }
        }
    }

    #[test]
    fn balanced_sample_size_and_no_intra_shard_duplicates() {
        let mut rng = LcgRng::new(99);
        let mut buf = sharded_buffer_new(cfg_rr(4, 40), 0).expect("buffer should build");
        for i in 0..200usize {
            sharded_add(&mut buf, vec![i as f32], (i % 5) as u32, &mut rng);
        }
        let (feats, labels, origin) =
            sharded_sample_balanced(&buf, 20, &mut rng).expect("sampling should succeed");
        assert_eq!(feats.len(), 20);
        assert_eq!(labels.len(), 20);
        assert_eq!(origin.len(), 20);
        // No duplicate (shard, feature-id) pair: items from the same shard must
        // be distinct stored vectors.
        let mut per_shard: std::collections::HashMap<usize, Vec<f32>> =
            std::collections::HashMap::new();
        for (f, &s) in feats.iter().zip(origin.iter()) {
            let key = f[0];
            let entry = per_shard.entry(s).or_default();
            assert!(!entry.contains(&key), "duplicate item {key} from shard {s}");
            entry.push(key);
        }
    }

    #[test]
    fn balanced_sample_is_spread_across_shards() {
        let mut rng = LcgRng::new(123);
        let mut buf = sharded_buffer_new(cfg_rr(4, 80), 0).expect("buffer should build");
        for i in 0..800usize {
            sharded_add(&mut buf, vec![i as f32], (i % 6) as u32, &mut rng);
        }
        let (_, _, origin) =
            sharded_sample_balanced(&buf, 16, &mut rng).expect("sampling should succeed");
        let distinct: std::collections::HashSet<usize> = origin.iter().copied().collect();
        // With 4 well-filled shards and 16 items drawn round-robin, all 4 shards
        // must be represented (exactly 4 each).
        assert_eq!(distinct.len(), 4, "balanced draw must touch every shard");
        for s in 0..4 {
            let c = origin.iter().filter(|&&o| o == s).count();
            assert_eq!(c, 4, "shard {s} should contribute 4 of 16");
        }
    }

    #[test]
    fn balanced_sample_drains_partial_shards() {
        // One shard much smaller than the others: balanced draw must still
        // succeed by pulling the remainder from the larger shards.
        let mut rng = LcgRng::new(55);
        let cfg =
            ShardedReplayConfig::new(2, 60, ShardPolicy::HashLabel).expect("valid sharded config");
        let mut buf = sharded_buffer_new(cfg, 0).expect("buffer should build");
        // Use labels that mostly hash to one shard by adding many of one class.
        for i in 0..3usize {
            sharded_add(&mut buf, vec![i as f32], 100, &mut rng); // few of class 100
        }
        for i in 0..50usize {
            sharded_add(&mut buf, vec![1000.0 + i as f32], 200, &mut rng); // many of class 200
        }
        let total = sharded_len(&buf);
        let (feats, _, _) =
            sharded_sample_balanced(&buf, total, &mut rng).expect("sampling should succeed");
        assert_eq!(
            feats.len(),
            total,
            "drawing all items must return all items"
        );
    }

    #[test]
    fn empty_buffer_sample_errors() {
        let mut rng = LcgRng::new(1);
        let buf = sharded_buffer_new(cfg_rr(3, 12), 0).expect("buffer should build");
        assert!(sharded_sample_balanced(&buf, 1, &mut rng).is_err());
    }

    #[test]
    fn sample_exceeding_total_errors() {
        let mut rng = LcgRng::new(2);
        let mut buf = sharded_buffer_new(cfg_rr(2, 8), 0).expect("buffer should build");
        for i in 0..4usize {
            sharded_add(&mut buf, vec![i as f32], 0, &mut rng);
        }
        assert!(sharded_sample_balanced(&buf, 100, &mut rng).is_err());
    }

    #[test]
    fn config_validation() {
        assert!(ShardedReplayConfig::new(0, 10, ShardPolicy::RoundRobin).is_err());
        assert!(ShardedReplayConfig::new(8, 4, ShardPolicy::RoundRobin).is_err());
        assert!(ShardedReplayConfig::new(4, 4, ShardPolicy::RoundRobin).is_ok());
    }

    #[test]
    fn hash_label_is_deterministic() {
        assert_eq!(hash_label(7), hash_label(7));
        assert_ne!(hash_label(7), hash_label(8));
    }
}
