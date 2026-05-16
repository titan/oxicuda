//! Neuroevolution: NEAT (NeuroEvolution of Augmenting Topologies).

pub mod neat;

pub use neat::{
    Activation, ConnectionGene, Genome, InnovationTracker, NeatConfig, NeatState, NodeGene,
    NodeType, Species, compatibility_distance, evaluate_genome,
};
