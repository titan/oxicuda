//! HAT — Hardware-Aware Transformers (multi-objective transformer NAS).
//!
//! Reference: Wang, Wu, Liu, Cai, Zhu, Gan & Han, "HAT: Hardware-Aware
//! Transformers for Efficient Natural Language Processing", ACL 2020.
//!
//! HAT searches a weight-shared transformer *SuperTransformer* for a
//! *SubTransformer* that is Pareto-optimal in `(loss-proxy, latency)` for one
//! target device. The two ingredients HAT contributes over a generic
//! evolutionary search are:
//!
//! 1. a **per-device latency model** — HAT trains one latency predictor per
//!    hardware target (CPU, GPU, mobile) because the same architecture has very
//!    different latencies across devices. Here that is a transformer-block
//!    latency LUT, [`BlockLatencyLut`], keyed on `(embed_dim, num_heads,
//!    mlp_ratio, n_experts, moe_top_k)` with a default fall-back; and
//!
//! 2. a **multi-objective evolutionary loop** over the elastic
//!    [`TransformerSearchSpace`]: a population is sorted into Pareto fronts, the
//!    non-dominated front is bred (crossover + per-axis mutation), and the run
//!    returns the final Pareto front so a deployer can pick the
//!    accuracy/latency trade-off they want.
//!
//! The accuracy side is supplied by the caller as a proxy closure (a trained
//! predictor, a zero-cost proxy, or `-flops` for a pure cost study), since the
//! true SuperTransformer accuracy needs GPU training. Latency comes from the LUT
//! — a *recorded* per-device measurement table, never a number this host
//! fabricates.

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;
use crate::ops::transformer_nas::{BlockSpec, TransformerArch, TransformerSearchSpace};
use std::collections::HashMap;

// ─── BlockLatencyLut ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BlockLatencyKey {
    embed_dim: usize,
    num_heads: usize,
    mlp_ratio: usize,
    n_experts: usize,
    moe_top_k: usize,
}

impl BlockLatencyKey {
    fn from_spec(b: &BlockSpec) -> Self {
        Self {
            embed_dim: b.embed_dim,
            num_heads: b.num_heads,
            mlp_ratio: b.mlp_ratio,
            n_experts: b.n_experts,
            moe_top_k: b.moe_top_k,
        }
    }
}

/// Per-device transformer-block latency lookup table (one HAT latency model).
///
/// A whole-architecture latency is the sum of its blocks' looked-up latencies,
/// modelling the (largely additive) per-layer cost of an encoder stack.
#[derive(Debug, Clone, Default)]
pub struct BlockLatencyLut {
    table: HashMap<BlockLatencyKey, f32>,
    /// Latency charged for a block configuration absent from the table.
    pub default_latency: f32,
}

impl BlockLatencyLut {
    /// Empty LUT with `default_latency = 0`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a measured latency (seconds) for one block configuration.
    pub fn insert(&mut self, block: &BlockSpec, latency_seconds: f32) {
        self.table
            .insert(BlockLatencyKey::from_spec(block), latency_seconds);
    }

    /// Look up one block, falling back to `default_latency`.
    #[must_use]
    pub fn lookup(&self, block: &BlockSpec) -> f32 {
        self.table
            .get(&BlockLatencyKey::from_spec(block))
            .copied()
            .unwrap_or(self.default_latency)
    }

    /// Total latency of an architecture: sum over its blocks.
    #[must_use]
    pub fn predict(&self, arch: &TransformerArch) -> f32 {
        arch.blocks.iter().map(|b| self.lookup(b)).sum()
    }

    /// Number of recorded block configurations.
    #[must_use]
    pub fn n_entries(&self) -> usize {
        self.table.len()
    }
}

// ─── Candidate ───────────────────────────────────────────────────────────────

/// A scored architecture inside the HAT search: the architecture plus its two
/// objectives (both *minimised*: a loss-like accuracy proxy and latency).
#[derive(Debug, Clone)]
pub struct Candidate {
    /// The transformer architecture.
    pub arch: TransformerArch,
    /// Accuracy-side objective (minimised; e.g. `1 - accuracy` or `-proxy`).
    pub loss: f32,
    /// Predicted latency (seconds, minimised).
    pub latency: f32,
}

impl Candidate {
    /// Pareto dominance on `(loss, latency)`: `self` dominates `other` iff it is
    /// no worse on both objectives and strictly better on at least one.
    #[must_use]
    pub fn dominates(&self, other: &Candidate) -> bool {
        let leq = self.loss <= other.loss && self.latency <= other.latency;
        let lt = self.loss < other.loss || self.latency < other.latency;
        leq && lt
    }
}

/// Extract the non-dominated (Pareto) subset of a candidate slice, preserving
/// input order. A candidate enters the front unless some other candidate
/// strictly dominates it.
#[must_use]
pub fn pareto_front(candidates: &[Candidate]) -> Vec<Candidate> {
    let mut front = Vec::new();
    for (i, c) in candidates.iter().enumerate() {
        let dominated = candidates
            .iter()
            .enumerate()
            .any(|(j, o)| j != i && o.dominates(c));
        if !dominated {
            front.push(c.clone());
        }
    }
    front
}

// ─── HatConfig ───────────────────────────────────────────────────────────────

/// Configuration for [`HatSearcher`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HatConfig {
    /// Population size (number of candidates per generation). Must be `>= 2`.
    pub population: usize,
    /// Number of evolutionary generations. `0` returns the initial front.
    pub generations: usize,
    /// Number of children bred from the Pareto parents each generation. Must be
    /// `>= 1`.
    pub n_children: usize,
}

impl HatConfig {
    /// Construct and validate a configuration.
    ///
    /// # Errors
    /// [`NasError::PopulationTooSmall`] if `population < 2`;
    /// [`NasError::InvalidNumOps`] if `n_children == 0`.
    pub fn new(population: usize, generations: usize, n_children: usize) -> NasResult<Self> {
        if population < 2 {
            return Err(NasError::PopulationTooSmall {
                min: 2,
                got: population,
            });
        }
        if n_children == 0 {
            return Err(NasError::InvalidNumOps);
        }
        Ok(Self {
            population,
            generations,
            n_children,
        })
    }
}

/// Outcome of a HAT search.
#[derive(Debug, Clone)]
pub struct HatResult {
    /// Final Pareto front over `(loss, latency)`, sorted by ascending latency.
    pub front: Vec<Candidate>,
    /// Number of generations actually executed.
    pub generations: usize,
    /// Total number of architecture evaluations spent.
    pub evals: usize,
}

impl HatResult {
    /// The lowest-loss candidate on the front (the accuracy-optimal trade-off).
    #[must_use]
    pub fn best_accuracy(&self) -> Option<&Candidate> {
        self.front.iter().min_by(|a, b| {
            a.loss
                .partial_cmp(&b.loss)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// The lowest-latency candidate on the front (the speed-optimal trade-off).
    #[must_use]
    pub fn best_latency(&self) -> Option<&Candidate> {
        self.front.iter().min_by(|a, b| {
            a.latency
                .partial_cmp(&b.latency)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

// ─── HatSearcher ───────────────────────────────────────────────────────────────

/// Evolutionary Pareto-front search over a transformer space with a per-device
/// latency LUT (HAT, Wang et al. 2020).
#[derive(Debug, Clone)]
pub struct HatSearcher {
    config: HatConfig,
}

impl HatSearcher {
    /// Build a searcher from its configuration.
    #[must_use]
    pub fn new(config: HatConfig) -> Self {
        Self { config }
    }

    /// Borrow the configuration.
    #[must_use]
    pub fn config(&self) -> &HatConfig {
        &self.config
    }

    /// Run the multi-objective search.
    ///
    /// `space` defines the elastic transformer search space, `latency_lut` is
    /// the (per-device) latency model, and `loss_proxy(arch)` returns the
    /// accuracy-side objective to *minimise* (e.g. `1 - val_acc`, or a negated
    /// zero-cost proxy). The returned [`HatResult`] holds the final Pareto front.
    ///
    /// # Errors
    /// Propagates sampling / construction errors from `space`.
    pub fn search<F>(
        &self,
        space: &TransformerSearchSpace,
        latency_lut: &BlockLatencyLut,
        mut loss_proxy: F,
        rng: &mut LcgRng,
    ) -> NasResult<HatResult>
    where
        F: FnMut(&TransformerArch) -> f32,
    {
        let mut evals = 0usize;
        let score = |arch: TransformerArch, loss_proxy: &mut F, evals: &mut usize| -> Candidate {
            let loss = loss_proxy(&arch);
            let latency = latency_lut.predict(&arch);
            *evals += 1;
            Candidate {
                arch,
                loss,
                latency,
            }
        };

        // Initial random population.
        let mut population: Vec<Candidate> = Vec::with_capacity(self.config.population);
        for _ in 0..self.config.population {
            let arch = space.sample(rng)?;
            population.push(score(arch, &mut loss_proxy, &mut evals));
        }

        let mut gens_done = 0usize;
        for _ in 0..self.config.generations {
            let parents = pareto_front(&population);
            // Breed children from the Pareto parents.
            let mut children = Vec::with_capacity(self.config.n_children);
            for _ in 0..self.config.n_children {
                let arch = if parents.len() >= 2 {
                    let i = rng.next_usize(parents.len());
                    let mut j = rng.next_usize(parents.len());
                    if j == i {
                        j = (j + 1) % parents.len();
                    }
                    let child = crossover(&parents[i].arch, &parents[j].arch, space, rng)?;
                    mutate(&child, space, rng)?
                } else if let Some(p) = parents.first() {
                    mutate(&p.arch, space, rng)?
                } else {
                    space.sample(rng)?
                };
                children.push(score(arch, &mut loss_proxy, &mut evals));
            }

            // Environmental selection: merge, keep the Pareto front, then top up
            // with the best-by-latency remainder to the population size so the
            // search keeps a fixed-size, diverse pool.
            population.extend(children);
            population = truncate_population(population, self.config.population);
            gens_done += 1;
        }

        let mut front = pareto_front(&population);
        front.sort_by(|a, b| {
            a.latency
                .partial_cmp(&b.latency)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(HatResult {
            front,
            generations: gens_done,
            evals,
        })
    }
}

// ─── Genetic operators over a transformer architecture ──────────────────────────

/// Per-block uniform crossover of two architectures (depth taken from the
/// shorter parent so every child block has both parents' blocks available).
fn crossover(
    a: &TransformerArch,
    b: &TransformerArch,
    space: &TransformerSearchSpace,
    rng: &mut LcgRng,
) -> NasResult<TransformerArch> {
    // Embedding dim is a trunk-wide choice: inherit from one parent.
    let embed_dim = if rng.next_f32() < 0.5 {
        a.embed_dim
    } else {
        b.embed_dim
    };
    let depth = a.depth().min(b.depth()).max(1);
    let mut blocks = Vec::with_capacity(depth);
    for i in 0..depth {
        let src = if rng.next_f32() < 0.5 {
            &a.blocks[i]
        } else {
            &b.blocks[i]
        };
        // Re-home the inherited block onto the chosen embed_dim; if heads no
        // longer divide, fall back to the smallest valid head choice.
        blocks.push(rehome_block(src, embed_dim, space)?);
    }
    TransformerArch::new(embed_dim, blocks, a.seq_len)
}

/// Mutate one randomly-chosen axis of one randomly-chosen block (or the depth)
/// to another admissible value.
fn mutate(
    arch: &TransformerArch,
    space: &TransformerSearchSpace,
    rng: &mut LcgRng,
) -> NasResult<TransformerArch> {
    let mut blocks = arch.blocks.clone();
    let mut embed_dim = arch.embed_dim;
    // 4 block-level axes + depth + embed = 6 mutation kinds.
    match rng.next_usize(6) {
        0 => {
            // Mutate embed_dim (trunk-wide), re-homing all blocks.
            embed_dim = space.embed_dims[rng.next_usize(space.embed_dims.len())];
            for b in &mut blocks {
                *b = rehome_block(b, embed_dim, space)?;
            }
        }
        1 if !blocks.is_empty() => {
            // Mutate one block's head count.
            let idx = rng.next_usize(blocks.len());
            let heads = space.head_choices[rng.next_usize(space.head_choices.len())];
            if embed_dim % heads == 0 {
                let b = blocks[idx];
                blocks[idx] =
                    BlockSpec::new(embed_dim, heads, b.mlp_ratio, b.n_experts, b.moe_top_k)?;
            }
        }
        2 if !blocks.is_empty() => {
            // Mutate one block's MLP ratio.
            let idx = rng.next_usize(blocks.len());
            let mlp = space.mlp_ratio_choices[rng.next_usize(space.mlp_ratio_choices.len())];
            let b = blocks[idx];
            blocks[idx] = BlockSpec::new(embed_dim, b.num_heads, mlp, b.n_experts, b.moe_top_k)?;
        }
        3 if !blocks.is_empty() => {
            // Mutate one block's expert layout.
            let idx = rng.next_usize(blocks.len());
            let experts = space.expert_choices[rng.next_usize(space.expert_choices.len())];
            let top_k = space.moe_top_k.min(experts);
            let b = blocks[idx];
            blocks[idx] = BlockSpec::new(embed_dim, b.num_heads, b.mlp_ratio, experts, top_k)?;
        }
        4 => {
            // Mutate depth: grow by cloning the last block, or shrink by one.
            let target = space.depth_choices[rng.next_usize(space.depth_choices.len())];
            if target > blocks.len() {
                if let Some(&last) = blocks.last() {
                    while blocks.len() < target {
                        blocks.push(last);
                    }
                }
            } else if target >= 1 {
                blocks.truncate(target);
            }
        }
        _ => {
            // No-op mutation (acts as elitism for this child).
        }
    }
    if blocks.is_empty() {
        blocks.push(rehome_block(&arch.blocks[0], embed_dim, space)?);
    }
    TransformerArch::new(embed_dim, blocks, arch.seq_len)
}

/// Re-home a block onto a (possibly new) `embed_dim`, repairing the head count
/// if it no longer divides the dimension.
fn rehome_block(
    src: &BlockSpec,
    embed_dim: usize,
    space: &TransformerSearchSpace,
) -> NasResult<BlockSpec> {
    let heads = if embed_dim % src.num_heads == 0 {
        src.num_heads
    } else {
        // Pick the smallest admissible head count that divides embed_dim.
        space
            .head_choices
            .iter()
            .copied()
            .filter(|&h| embed_dim % h == 0)
            .min()
            .unwrap_or(1)
    };
    BlockSpec::new(
        embed_dim,
        heads,
        src.mlp_ratio,
        src.n_experts,
        src.moe_top_k,
    )
}

/// Keep the Pareto front plus the best-by-latency remainder up to `cap`.
fn truncate_population(pop: Vec<Candidate>, cap: usize) -> Vec<Candidate> {
    if pop.len() <= cap {
        return pop;
    }
    let front = pareto_front(&pop);
    if front.len() >= cap {
        // Too many non-dominated: keep the `cap` with the smallest latency for
        // a fixed-size pool (a crowding proxy along the latency axis).
        let mut f = front;
        f.sort_by(|a, b| {
            a.latency
                .partial_cmp(&b.latency)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        f.truncate(cap);
        return f;
    }
    // Take the whole front, then the best-loss dominated remainder.
    let mut keep = front.clone();
    let mut remainder: Vec<Candidate> = pop
        .into_iter()
        .filter(|c| !front.iter().any(|f| candidate_eq(f, c)))
        .collect();
    remainder.sort_by(|a, b| {
        a.loss
            .partial_cmp(&b.loss)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for c in remainder {
        if keep.len() >= cap {
            break;
        }
        keep.push(c);
    }
    keep
}

fn candidate_eq(a: &Candidate, b: &Candidate) -> bool {
    a.arch == b.arch
        && a.loss.to_bits() == b.loss.to_bits()
        && a.latency.to_bits() == b.latency.to_bits()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn space() -> TransformerSearchSpace {
        TransformerSearchSpace::new(
            vec![192, 384],
            vec![3, 4, 6],
            vec![3, 4],
            vec![1, 4],
            2,
            vec![4, 6, 8],
            197,
        )
        .expect("space")
    }

    fn calibrated_lut(space: &TransformerSearchSpace) -> BlockLatencyLut {
        // A monotone synthetic per-device cost: latency grows with hidden width
        // and expert count. Insert one entry per (embed,head,mlp,expert) combo.
        let mut lut = BlockLatencyLut::new();
        for &e in &space.embed_dims {
            for &h in &space.head_choices {
                if e % h != 0 {
                    continue;
                }
                for &m in &space.mlp_ratio_choices {
                    for &x in &space.expert_choices {
                        let top_k = space.moe_top_k.min(x);
                        let b = BlockSpec::new(e, h, m, x, top_k).expect("block");
                        let lat = 1e-5 * (e * m) as f32 + 1e-4 * x as f32;
                        lut.insert(&b, lat);
                    }
                }
            }
        }
        lut
    }

    #[test]
    fn lut_predict_sums_blocks() {
        let s = space();
        let lut = calibrated_lut(&s);
        let b = BlockSpec::new(192, 6, 4, 1, 1).expect("block");
        let arch = TransformerArch::new(192, vec![b; 3], 197).expect("arch");
        let per = lut.lookup(&b);
        assert!((lut.predict(&arch) - per * 3.0).abs() < 1e-9);
    }

    #[test]
    fn lut_default_for_unknown() {
        let mut lut = BlockLatencyLut::new();
        lut.default_latency = 0.5;
        let b = BlockSpec::new(192, 6, 4, 1, 1).expect("block");
        assert_eq!(lut.lookup(&b), 0.5);
    }

    #[test]
    fn dominance_is_correct() {
        let b = BlockSpec::new(192, 6, 4, 1, 1).expect("b");
        let arch = TransformerArch::new(192, vec![b], 197).expect("a");
        let c1 = Candidate {
            arch: arch.clone(),
            loss: 0.2,
            latency: 0.1,
        };
        let c2 = Candidate {
            arch: arch.clone(),
            loss: 0.3,
            latency: 0.2,
        };
        let c3 = Candidate {
            arch,
            loss: 0.1,
            latency: 0.3,
        };
        assert!(c1.dominates(&c2));
        assert!(!c2.dominates(&c1));
        // c1 and c3 are mutually non-dominated (trade-off).
        assert!(!c1.dominates(&c3));
        assert!(!c3.dominates(&c1));
    }

    #[test]
    fn pareto_front_excludes_dominated() {
        let s = space();
        let b = BlockSpec::new(192, 6, 4, 1, 1).expect("b");
        let arch = TransformerArch::new(192, vec![b], 197).expect("a");
        let mk = |loss, latency| Candidate {
            arch: arch.clone(),
            loss,
            latency,
        };
        let pop = vec![
            mk(0.2, 0.1),  // non-dominated
            mk(0.3, 0.3),  // dominated by (0.2,0.1)
            mk(0.1, 0.4),  // non-dominated (best loss)
            mk(0.5, 0.05), // non-dominated (best latency)
        ];
        let _ = s;
        let front = pareto_front(&pop);
        // The dominated (0.3,0.3) must be gone; the 3 trade-off points remain.
        assert_eq!(front.len(), 3);
        assert!(front.iter().all(|c| !(c.loss == 0.3 && c.latency == 0.3)));
    }

    #[test]
    fn config_rejects_small_population() {
        assert_eq!(
            HatConfig::new(1, 5, 3),
            Err(NasError::PopulationTooSmall { min: 2, got: 1 })
        );
        assert_eq!(HatConfig::new(8, 5, 0), Err(NasError::InvalidNumOps));
    }

    #[test]
    fn search_returns_valid_pareto_front() {
        let s = space();
        let lut = calibrated_lut(&s);
        // Loss proxy: prefer more parameters (negate params, normalised) so the
        // accuracy objective is in genuine tension with the latency objective.
        let loss = |a: &TransformerArch| -(a.total_params() as f32) / 1e8;
        let searcher = HatSearcher::new(HatConfig::new(12, 8, 6).expect("cfg"));
        let mut rng = LcgRng::new(2024);
        let res = searcher.search(&s, &lut, loss, &mut rng).expect("search");
        assert_eq!(res.generations, 8);
        assert!(!res.front.is_empty());
        // The returned front must be internally non-dominated.
        for (i, c) in res.front.iter().enumerate() {
            for (j, o) in res.front.iter().enumerate() {
                if i != j {
                    assert!(!o.dominates(c), "front contains a dominated member");
                }
            }
        }
        // Front sorted by ascending latency.
        for w in res.front.windows(2) {
            assert!(w[0].latency <= w[1].latency + 1e-9);
        }
        // Endpoints exist and the speed-optimal is no slower than acc-optimal.
        let fast = res.best_latency().expect("fast");
        let acc = res.best_accuracy().expect("acc");
        assert!(fast.latency <= acc.latency + 1e-9);
    }

    #[test]
    fn search_improves_latency_over_random_under_latency_only_objective() {
        // With a constant accuracy proxy the search collapses to single-objective
        // latency minimisation; the final front's best latency should beat the
        // mean latency of a fresh random population.
        let s = space();
        let lut = calibrated_lut(&s);
        let mut rng = LcgRng::new(7);
        // Baseline: mean latency of 30 random archs.
        let mut sum = 0.0_f32;
        let n = 30;
        for _ in 0..n {
            let a = s.sample(&mut rng).expect("sample");
            sum += lut.predict(&a);
        }
        let random_mean = sum / n as f32;

        let searcher = HatSearcher::new(HatConfig::new(16, 12, 8).expect("cfg"));
        let mut rng2 = LcgRng::new(7);
        let res = searcher
            .search(&s, &lut, |_| 0.0, &mut rng2)
            .expect("search");
        let best = res.best_latency().expect("best").latency;
        assert!(
            best < random_mean,
            "evolved best latency {best} should beat random mean {random_mean}"
        );
    }

    #[test]
    fn zero_generations_returns_initial_front() {
        let s = space();
        let lut = calibrated_lut(&s);
        let searcher = HatSearcher::new(HatConfig::new(10, 0, 4).expect("cfg"));
        let mut rng = LcgRng::new(1);
        let res = searcher
            .search(&s, &lut, |a| a.total_macs() as f32 / 1e9, &mut rng)
            .expect("search");
        assert_eq!(res.generations, 0);
        assert_eq!(res.evals, 10);
        assert!(!res.front.is_empty());
    }

    #[test]
    fn search_is_deterministic_given_seed() {
        let s = space();
        let lut = calibrated_lut(&s);
        let searcher = HatSearcher::new(HatConfig::new(10, 6, 5).expect("cfg"));
        let loss = |a: &TransformerArch| a.total_macs() as f32 / 1e9;
        let mut ra = LcgRng::new(555);
        let mut rb = LcgRng::new(555);
        let res_a = searcher.search(&s, &lut, loss, &mut ra).expect("a");
        let res_b = searcher.search(&s, &lut, loss, &mut rb).expect("b");
        assert_eq!(res_a.front.len(), res_b.front.len());
        for (x, y) in res_a.front.iter().zip(res_b.front.iter()) {
            assert_eq!(x.arch, y.arch);
            assert_eq!(x.loss.to_bits(), y.loss.to_bits());
            assert_eq!(x.latency.to_bits(), y.latency.to_bits());
        }
    }
}
