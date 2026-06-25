//! Neuroevolution: NEAT, HyperNEAT, ES-HyperNEAT, and NEAT + Novelty Search.

pub mod es_hyperneat;
pub mod hyperneat;
pub mod neat;
pub mod neat_novelty;

pub use es_hyperneat::{
    EsHyperNeatConfig, EsHyperNeatState, EsSubstrate, discover_hidden_nodes,
    discover_hidden_nodes_quadtree, es_hyperneat_discover_substrate,
    es_hyperneat_discover_substrate_cfg, es_hyperneat_forward, es_hyperneat_run,
};
pub use hyperneat::{
    CppnActivation, CppnConfig, CppnWeights, HyperNeatConfig, HyperNeatState, Substrate,
    hyperneat_forward, hyperneat_query_weights, hyperneat_run,
};
pub use neat::{
    Activation, ConnectionGene, Genome, InnovationTracker, NeatConfig, NeatState, NodeGene,
    NodeType, Species, compatibility_distance, evaluate_genome,
};
pub use neat_novelty::{
    NeatNovelty, NeatNoveltyConfig, NoveltyArchive, compute_novelty, compute_population_novelty,
};
