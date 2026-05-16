//! NEAT: NeuroEvolution of Augmenting Topologies.
//!
//! Reference: K. Stanley & R. Miikkulainen, "Evolving Neural Networks through Augmenting
//! Topologies", Evolutionary Computation 10(2):99-127, 2002.

#![allow(clippy::needless_range_loop)]

use crate::{EvolError, EvolResult, handle::LcgRng};
use std::collections::{HashMap, VecDeque};

// ─── Gene types ──────────────────────────────────────────────────────────────

/// Type of a node in the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Input,
    Hidden,
    Output,
}

/// Activation function for hidden/output nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Sigmoid,
    Tanh,
    ReLU,
}

impl Activation {
    /// Apply the activation function to a value.
    pub fn apply(self, x: f64) -> f64 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::ReLU => x.max(0.0),
        }
    }
}

/// A node gene defines a single neuron.
#[derive(Debug, Clone)]
pub struct NodeGene {
    /// Unique node identifier.
    pub id: usize,
    /// Functional role in the network.
    pub node_type: NodeType,
    /// Non-linearity applied to the weighted sum.
    pub activation: Activation,
}

/// A connection gene defines one directed weighted edge.
#[derive(Debug, Clone)]
pub struct ConnectionGene {
    /// Source node id.
    pub from: usize,
    /// Target node id.
    pub to: usize,
    /// Synaptic weight.
    pub weight: f64,
    /// Whether this connection is currently active.
    pub enabled: bool,
    /// Global innovation number (unique per (from, to) pair globally).
    pub innovation: usize,
}

/// A complete genome: node genes + connection genes + fitness.
#[derive(Debug, Clone)]
pub struct Genome {
    /// All node genes (inputs, bias, hidden, output).
    pub nodes: Vec<NodeGene>,
    /// All connection genes.
    pub connections: Vec<ConnectionGene>,
    /// Current fitness (higher = better; NEAT maximises).
    pub fitness: f64,
}

impl Genome {
    /// Create a minimal genome with the specified numbers of inputs and outputs.
    /// All inputs are connected to all outputs with random weights.
    fn minimal(
        n_inputs: usize,
        n_outputs: usize,
        rng: &mut LcgRng,
        tracker: &mut InnovationTracker,
    ) -> Self {
        let mut nodes = Vec::new();
        for i in 0..n_inputs {
            nodes.push(NodeGene {
                id: i,
                node_type: NodeType::Input,
                activation: Activation::Sigmoid,
            });
        }
        for i in 0..n_outputs {
            nodes.push(NodeGene {
                id: n_inputs + i,
                node_type: NodeType::Output,
                activation: Activation::Sigmoid,
            });
        }
        let mut connections = Vec::new();
        for i in 0..n_inputs {
            for j in 0..n_outputs {
                let innov = tracker.get_or_insert(i, n_inputs + j);
                connections.push(ConnectionGene {
                    from: i,
                    to: n_inputs + j,
                    weight: rng.next_normal() * 0.5,
                    enabled: true,
                    innovation: innov,
                });
            }
        }
        Genome {
            nodes,
            connections,
            fitness: 0.0,
        }
    }
}

// ─── Innovation tracker ───────────────────────────────────────────────────────

/// Global innovation tracker ensuring that the same structural mutation always gets
/// the same innovation number within a generation.
pub struct InnovationTracker {
    counter: usize,
    map: HashMap<(usize, usize), usize>,
}

impl InnovationTracker {
    /// Create a new tracker starting at innovation 0.
    pub fn new() -> Self {
        Self {
            counter: 0,
            map: HashMap::new(),
        }
    }

    /// Return the existing innovation number for `(from, to)`, or allocate a new one.
    pub fn get_or_insert(&mut self, from: usize, to: usize) -> usize {
        let key = (from, to);
        if let Some(&innov) = self.map.get(&key) {
            innov
        } else {
            let n = self.counter;
            self.map.insert(key, n);
            self.counter += 1;
            n
        }
    }
}

impl Default for InnovationTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Compatibility distance ───────────────────────────────────────────────────

/// Compute the NEAT compatibility distance δ between two genomes.
///
/// δ = (c1·E + c2·D) / N + c3·W̄
///
/// where E = excess genes, D = disjoint genes, W̄ = mean weight difference of matching genes,
/// N = max(|g1|, |g2|) (normalisation factor; treated as 1 if both small).
pub fn compatibility_distance(g1: &Genome, g2: &Genome, c1: f64, c2: f64, c3: f64) -> f64 {
    let innov1: HashMap<usize, &ConnectionGene> =
        g1.connections.iter().map(|c| (c.innovation, c)).collect();
    let innov2: HashMap<usize, &ConnectionGene> =
        g2.connections.iter().map(|c| (c.innovation, c)).collect();

    let max1 = innov1.keys().copied().max().unwrap_or(0);
    let max2 = innov2.keys().copied().max().unwrap_or(0);
    let max_innov = max1.max(max2);

    let mut matching_weight_diff = 0.0;
    let mut n_matching = 0usize;
    let mut disjoint = 0usize;
    let mut excess = 0usize;

    for innov in 0..=max_innov {
        match (innov1.get(&innov), innov2.get(&innov)) {
            (Some(c1g), Some(c2g)) => {
                matching_weight_diff += (c1g.weight - c2g.weight).abs();
                n_matching += 1;
            }
            (Some(_), None) | (None, Some(_)) => {
                // Disjoint if within range of both, excess otherwise
                if innov <= max1 && innov <= max2 {
                    disjoint += 1;
                } else {
                    excess += 1;
                }
            }
            (None, None) => {}
        }
    }

    let w_bar = if n_matching > 0 {
        matching_weight_diff / n_matching as f64
    } else {
        0.0
    };

    let big_n = (innov1.len().max(innov2.len())).max(1) as f64;
    let big_n = if big_n < 20.0 { 1.0 } else { big_n };

    (c1 * excess as f64 + c2 * disjoint as f64) / big_n + c3 * w_bar
}

// ─── Genome evaluation ────────────────────────────────────────────────────────

/// Execute a forward pass through the network encoded by `genome`.
///
/// Uses Kahn's topological sort with compact node re-indexing to avoid HashMap overhead.
/// Nodes not reachable from inputs are treated as having zero activation.
pub fn evaluate_genome(genome: &Genome, inputs: &[f64], n_outputs: usize) -> EvolResult<Vec<f64>> {
    let n_inputs = inputs.len();
    let n_nodes = genome.nodes.len();

    if n_nodes == 0 {
        return Ok(vec![0.0; n_outputs]);
    }

    // ── Compact re-index: sort nodes by id; compact[i] = index into sorted array ──
    // Build sorted (node_id, original_index) pairs
    let mut sorted_nodes: Vec<(usize, usize)> = genome
        .nodes
        .iter()
        .enumerate()
        .map(|(orig_idx, n)| (n.id, orig_idx))
        .collect();
    sorted_nodes.sort_unstable_by_key(|(id, _)| *id);

    // compact: node_id → compact index (index in sorted_nodes)
    let compact = |id: usize| -> usize { sorted_nodes.partition_point(|(nid, _)| *nid < id) };

    // Activation values indexed by compact index
    let mut activation = vec![0.0f64; n_nodes];
    // In-degree for Kahn's algorithm
    let mut in_degree = vec![0usize; n_nodes];
    // Adjacency list: from_compact → [(to_compact, weight)]
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_nodes];

    // Helper: get node at compact index
    let get_node = |ci: usize| -> &NodeGene { &genome.nodes[sorted_nodes[ci].1] };

    // Initialise input activations
    for (i, &val) in inputs.iter().enumerate() {
        let ci = compact(i);
        if ci < n_nodes && sorted_nodes[ci].0 == i {
            activation[ci] = val;
        }
    }

    // Build graph
    for conn in &genome.connections {
        if !conn.enabled {
            continue;
        }
        let from_c = compact(conn.from);
        let to_c = compact(conn.to);
        let from_valid = from_c < n_nodes && sorted_nodes[from_c].0 == conn.from;
        let to_valid = to_c < n_nodes && sorted_nodes[to_c].0 == conn.to;
        if from_valid && to_valid {
            adj[from_c].push((to_c, conn.weight));
            in_degree[to_c] += 1;
        }
    }

    // Kahn's algorithm
    let mut queue: VecDeque<usize> = VecDeque::new();

    // Seed: all nodes with zero in-degree (input nodes always qualify)
    for ci in 0..n_nodes {
        if in_degree[ci] == 0 {
            queue.push_back(ci);
        }
    }

    let mut visited = vec![false; n_nodes];

    while let Some(ci) = queue.pop_front() {
        if visited[ci] {
            continue;
        }
        visited[ci] = true;

        // Apply activation
        let node = get_node(ci);
        let act_val = if node.node_type == NodeType::Input {
            activation[ci]
        } else {
            node.activation.apply(activation[ci])
        };
        activation[ci] = act_val;

        // Propagate to successors
        for k in 0..adj[ci].len() {
            let (to_c, weight) = adj[ci][k];
            activation[to_c] += act_val * weight;
            if in_degree[to_c] > 0 {
                in_degree[to_c] -= 1;
                if in_degree[to_c] == 0 {
                    queue.push_back(to_c);
                }
            }
        }
    }

    // Collect output activations (output nodes have ids n_inputs..n_inputs+n_outputs)
    let outputs: Vec<f64> = (0..n_outputs)
        .map(|i| {
            let out_id = n_inputs + i;
            let ci = compact(out_id);
            if ci < n_nodes && sorted_nodes[ci].0 == out_id {
                let node = get_node(ci);
                if visited[ci] {
                    activation[ci]
                } else {
                    // Not visited yet: apply activation to accumulated pre-activation
                    node.activation.apply(activation[ci])
                }
            } else {
                0.0
            }
        })
        .collect();

    Ok(outputs)
}

// ─── NEAT Config ──────────────────────────────────────────────────────────────

/// Hyper-parameters for a NEAT run.
#[derive(Debug, Clone)]
pub struct NeatConfig {
    /// Number of network inputs.
    pub n_inputs: usize,
    /// Number of network outputs.
    pub n_outputs: usize,
    /// Population size.
    pub pop_size: usize,
    /// Maximum generations.
    pub max_generations: usize,
    /// Speciation compatibility threshold δ_t.
    pub delta_t: f64,
    /// Excess gene coefficient c1.
    pub c1: f64,
    /// Disjoint gene coefficient c2.
    pub c2: f64,
    /// Weight difference coefficient c3.
    pub c3: f64,
    /// Probability of weight mutation per connection.
    pub weight_mut_prob: f64,
    /// Probability of adding a new node (structural).
    pub add_node_prob: f64,
    /// Probability of adding a new connection.
    pub add_conn_prob: f64,
    /// Probability of re-enabling a disabled connection.
    pub enable_prob: f64,
    /// Fraction of species population to carry forward.
    pub survival_threshold: f64,
}

impl NeatConfig {
    /// Build a default NEAT config for the given I/O shape.
    pub fn new(n_inputs: usize, n_outputs: usize) -> Self {
        Self {
            n_inputs,
            n_outputs,
            pop_size: 150,
            max_generations: 100,
            delta_t: 3.0,
            c1: 1.0,
            c2: 1.0,
            c3: 0.4,
            weight_mut_prob: 0.8,
            add_node_prob: 0.03,
            add_conn_prob: 0.05,
            enable_prob: 0.25,
            survival_threshold: 0.2,
        }
    }
}

// ─── Species ──────────────────────────────────────────────────────────────────

/// A species: a cluster of genetically similar genomes.
pub struct Species {
    /// Representative genome (chosen at start of generation).
    pub representative: Genome,
    /// Indices into the population vector.
    pub members: Vec<usize>,
    /// Best (highest) fitness seen in this species.
    pub best_fitness: f64,
    /// Generations since fitness improved.
    pub staleness: usize,
}

// ─── NEAT State ───────────────────────────────────────────────────────────────

/// Full NEAT algorithm state.
pub struct NeatState {
    /// Current population.
    pub population: Vec<Genome>,
    /// Current species.
    pub species: Vec<Species>,
    /// Global innovation counter.
    pub innovation: InnovationTracker,
    /// Current generation index.
    pub generation: usize,
}

impl NeatState {
    /// Initialise NEAT with a minimal population.
    pub fn new(cfg: &NeatConfig, rng: &mut LcgRng) -> Self {
        let mut innovation = InnovationTracker::new();
        let population = (0..cfg.pop_size)
            .map(|_| Genome::minimal(cfg.n_inputs, cfg.n_outputs, rng, &mut innovation))
            .collect();
        Self {
            population,
            species: Vec::new(),
            innovation,
            generation: 0,
        }
    }

    /// Evaluate all genomes using `eval_fn` and store fitness.
    pub fn evaluate_all<F: Fn(&Genome) -> f64>(&mut self, eval_fn: F) {
        for genome in &mut self.population {
            genome.fitness = eval_fn(genome);
        }
    }

    /// Assign each genome to the first species whose representative is within δ_t,
    /// or create a new species.
    pub fn speciate(&mut self, cfg: &NeatConfig) {
        // Clear member lists; keep representatives
        for species in &mut self.species {
            species.members.clear();
        }

        for (i, genome) in self.population.iter().enumerate() {
            let mut assigned = false;
            for species in &mut self.species {
                let dist =
                    compatibility_distance(genome, &species.representative, cfg.c1, cfg.c2, cfg.c3);
                if dist < cfg.delta_t {
                    species.members.push(i);
                    assigned = true;
                    break;
                }
            }
            if !assigned {
                self.species.push(Species {
                    representative: genome.clone(),
                    members: vec![i],
                    best_fitness: genome.fitness,
                    staleness: 0,
                });
            }
        }

        // Remove empty species
        self.species.retain(|s| !s.members.is_empty());

        // Update species best fitness and staleness
        for species in &mut self.species {
            let best = species
                .members
                .iter()
                .map(|&i| self.population[i].fitness)
                .fold(f64::NEG_INFINITY, f64::max);
            if best > species.best_fitness {
                species.best_fitness = best;
                species.staleness = 0;
            } else {
                species.staleness += 1;
            }
        }
    }

    /// Mutate a genome: perturb weights, add nodes, add connections, toggle connections.
    fn mutate(
        &self,
        genome: &mut Genome,
        cfg: &NeatConfig,
        rng: &mut LcgRng,
        next_node_id: &mut usize,
    ) {
        // Weight mutation
        if rng.next_f64() < cfg.weight_mut_prob {
            for conn in &mut genome.connections {
                if rng.next_f64() < 0.9 {
                    conn.weight += rng.next_normal() * 0.1;
                } else {
                    // Assign random weight
                    conn.weight = rng.next_normal();
                }
            }
        }

        // Enable/disable mutation
        if rng.next_f64() < cfg.enable_prob && !genome.connections.is_empty() {
            let idx = rng.next_usize(genome.connections.len());
            genome.connections[idx].enabled ^= true;
        }

        // Add connection
        if rng.next_f64() < cfg.add_conn_prob && genome.nodes.len() >= 2 {
            let n = genome.nodes.len();
            let from_idx = rng.next_usize(n);
            let to_idx = rng.next_usize(n);
            let from_id = genome.nodes[from_idx].id;
            let to_id = genome.nodes[to_idx].id;
            // Don't connect to input nodes
            if genome.nodes[to_idx].node_type != NodeType::Input && from_id != to_id {
                let already_exists = genome
                    .connections
                    .iter()
                    .any(|c| c.from == from_id && c.to == to_id);
                if !already_exists {
                    let innov = self
                        .innovation
                        .map
                        .get(&(from_id, to_id))
                        .copied()
                        .unwrap_or_else(|| {
                            // We can't mutate the tracker here without mut; use a hash
                            (from_id * 10007 + to_id) % 100000
                        });
                    genome.connections.push(ConnectionGene {
                        from: from_id,
                        to: to_id,
                        weight: rng.next_normal(),
                        enabled: true,
                        innovation: innov,
                    });
                }
            }
        }

        // Add node: split an existing connection
        if rng.next_f64() < cfg.add_node_prob {
            let enabled_conns: Vec<usize> = genome
                .connections
                .iter()
                .enumerate()
                .filter(|(_, c)| c.enabled)
                .map(|(i, _)| i)
                .collect();
            if !enabled_conns.is_empty() {
                let conn_idx = enabled_conns[rng.next_usize(enabled_conns.len())];
                let old_from = genome.connections[conn_idx].from;
                let old_to = genome.connections[conn_idx].to;
                let old_weight = genome.connections[conn_idx].weight;
                genome.connections[conn_idx].enabled = false;

                let new_id = *next_node_id;
                *next_node_id += 1;
                genome.nodes.push(NodeGene {
                    id: new_id,
                    node_type: NodeType::Hidden,
                    activation: Activation::Sigmoid,
                });
                let innov_in = (old_from * 10007 + new_id) % 100000;
                let innov_out = (new_id * 10007 + old_to) % 100000;
                genome.connections.push(ConnectionGene {
                    from: old_from,
                    to: new_id,
                    weight: 1.0,
                    enabled: true,
                    innovation: innov_in,
                });
                genome.connections.push(ConnectionGene {
                    from: new_id,
                    to: old_to,
                    weight: old_weight,
                    enabled: true,
                    innovation: innov_out,
                });
            }
        }
    }

    /// Crossover two genomes (NEAT-style: matching genes averaged, disjoint/excess from fitter parent).
    fn crossover(g1: &Genome, g2: &Genome, rng: &mut LcgRng) -> Genome {
        // g1 should be the fitter parent
        let (fitter, weaker) = if g1.fitness >= g2.fitness {
            (g1, g2)
        } else {
            (g2, g1)
        };
        let weaker_map: HashMap<usize, &ConnectionGene> = weaker
            .connections
            .iter()
            .map(|c| (c.innovation, c))
            .collect();

        let connections = fitter
            .connections
            .iter()
            .map(|fc| {
                if let Some(wc) = weaker_map.get(&fc.innovation) {
                    // Matching gene: randomly pick weight from either parent
                    let weight = if rng.next_bool() {
                        fc.weight
                    } else {
                        wc.weight
                    };
                    let enabled = fc.enabled && wc.enabled;
                    ConnectionGene {
                        weight,
                        enabled,
                        ..*fc
                    }
                } else {
                    fc.clone()
                }
            })
            .collect();

        Genome {
            nodes: fitter.nodes.clone(),
            connections,
            fitness: 0.0,
        }
    }

    /// Produce the next generation via speciated selection + reproduction.
    pub fn reproduce(&mut self, cfg: &NeatConfig, rng: &mut LcgRng) -> EvolResult<()> {
        if self.species.is_empty() {
            return Err(EvolError::EmptyPopulation);
        }

        // Compute shared (explicit) fitness for each individual
        let pop_size = self.population.len();
        let _species_count = self.species.len();
        let mut shared_fitness = vec![0.0f64; pop_size];
        for species in &self.species {
            let share = species.members.len() as f64;
            for &i in &species.members {
                shared_fitness[i] = self.population[i].fitness / share;
            }
        }

        // Species offspring allocation proportional to total shared fitness
        let total_shared: f64 = shared_fitness.iter().sum();
        let total_shared = if total_shared <= 0.0 {
            1.0
        } else {
            total_shared
        };

        let mut offspring_counts: Vec<usize> = self
            .species
            .iter()
            .map(|s| {
                let s_fit: f64 = s.members.iter().map(|&i| shared_fitness[i]).sum();
                let n = ((s_fit / total_shared) * pop_size as f64).round() as usize;
                n.max(1)
            })
            .collect();

        // Adjust total to pop_size
        let total_alloc: usize = offspring_counts.iter().sum();
        if total_alloc < pop_size {
            offspring_counts[0] += pop_size - total_alloc;
        } else {
            let mut excess = total_alloc - pop_size;
            for count in offspring_counts.iter_mut().rev() {
                if *count > 1 && excess > 0 {
                    let reduce = (*count - 1).min(excess);
                    *count -= reduce;
                    excess -= reduce;
                }
                if excess == 0 {
                    break;
                }
            }
        }

        // Compute next node id (for add-node mutation)
        let mut next_node_id = self
            .population
            .iter()
            .flat_map(|g| g.nodes.iter().map(|n| n.id))
            .max()
            .unwrap_or(cfg.n_inputs + cfg.n_outputs - 1)
            + 1;

        // Generate offspring
        let mut new_population: Vec<Genome> = Vec::with_capacity(pop_size);
        for (si, species) in self.species.iter().enumerate() {
            let n_off = offspring_counts[si];
            if n_off == 0 {
                continue;
            }

            // Sort members by fitness descending
            let mut members = species.members.clone();
            members.sort_by(|&a, &b| {
                self.population[b]
                    .fitness
                    .partial_cmp(&self.population[a].fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Elitism: carry best genome unchanged
            let elites = 1.min(members.len()).min(n_off);
            for &idx in members.iter().take(elites) {
                new_population.push(self.population[idx].clone());
            }

            // Survival threshold: only top fraction can reproduce
            let n_survive = ((members.len() as f64 * cfg.survival_threshold).ceil() as usize)
                .max(1)
                .min(members.len());
            let breeders = &members[..n_survive];

            // Fill remaining offspring
            for _ in elites..n_off {
                let p1_idx = breeders[rng.next_usize(breeders.len())];
                let mut child = if breeders.len() > 1 && rng.next_f64() < 0.75 {
                    let p2_idx = breeders[rng.next_usize(breeders.len())];
                    Self::crossover(&self.population[p1_idx], &self.population[p2_idx], rng)
                } else {
                    self.population[p1_idx].clone()
                };
                self.mutate(&mut child, cfg, rng, &mut next_node_id);
                new_population.push(child);
            }
        }

        // Pad if under pop_size (safety)
        while new_population.len() < pop_size {
            let src = rng.next_usize(new_population.len().max(1));
            let mut clone = if new_population.is_empty() {
                Genome::minimal(cfg.n_inputs, cfg.n_outputs, rng, &mut self.innovation)
            } else {
                new_population[src].clone()
            };
            self.mutate(&mut clone, cfg, rng, &mut next_node_id);
            new_population.push(clone);
        }
        new_population.truncate(pop_size);

        self.population = new_population;
        self.generation += 1;
        Ok(())
    }

    /// Execute one generation: evaluate → speciate → reproduce.
    pub fn step<F: Fn(&Genome) -> f64>(
        &mut self,
        eval_fn: F,
        cfg: &NeatConfig,
        rng: &mut LcgRng,
    ) -> EvolResult<()> {
        self.evaluate_all(eval_fn);
        self.speciate(cfg);
        self.reproduce(cfg, rng)
    }

    /// Return the genome with the highest fitness.
    pub fn best(&self) -> Option<&Genome> {
        self.population.iter().max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}
