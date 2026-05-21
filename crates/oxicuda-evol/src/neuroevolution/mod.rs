//! Neuroevolution: NEAT, HyperNEAT, and ES-HyperNEAT.

pub mod es_hyperneat;
pub mod hyperneat;
pub mod neat;

pub use es_hyperneat::{
    EsHyperNeatConfig, EsHyperNeatState, EsSubstrate, es_hyperneat_discover_substrate,
    es_hyperneat_forward, es_hyperneat_run,
};
pub use hyperneat::{
    CppnActivation, CppnConfig, CppnWeights, HyperNeatConfig, HyperNeatState, Substrate,
    hyperneat_forward, hyperneat_query_weights, hyperneat_run,
};
pub use neat::{
    Activation, ConnectionGene, Genome, InnovationTracker, NeatConfig, NeatState, NodeGene,
    NodeType, Species, compatibility_distance, evaluate_genome,
};
