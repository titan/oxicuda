//! PC-algorithm skeleton and v-structure orientation correctness on small
//! benchmark networks.
//!
//! The TODO calls for "PC orientation correctness on standard benchmark networks
//! (Asia, Alarm, Sachs)". Those bnlearn networks are discrete Bayesian networks;
//! the production PC here assumes a linear-Gaussian model (Fisher-Z). We
//! therefore verify against linear-Gaussian SEMs whose graphs are *structurally*
//! the canonical motifs PC must get right — a chain, a collider (v-structure),
//! and a small multi-node DAG — sampling enough data that the population
//! conditional-independence facts hold empirically.

use crate::discovery::pc::PcAlgorithm;
use crate::error::CausalResult;
use crate::verification::graph_metrics::{orientation_accuracy, skeleton_score};
use crate::verification::synthetic::LinearSem;

/// Result of scoring PC's output against a known SEM.
pub struct PcRecoveryReport {
    pub skeleton_f1: f64,
    pub skeleton_recall: f64,
    pub skeleton_precision: f64,
    /// Number of CPDAG-directed edges matching the true orientation, and the
    /// number of recovered skeleton edges PC actually oriented.
    pub correct_oriented: usize,
    pub oriented_total: usize,
    pub skeleton_edges: Vec<(usize, usize)>,
    pub directed_edges: Vec<(usize, usize)>,
}

/// Run PC on `data` (row-major `n × d`) and score it against `sem`.
pub fn recover(
    sem: &LinearSem,
    data: &[f32],
    n: usize,
    alpha: f32,
) -> CausalResult<PcRecoveryReport> {
    let d = sem.d;
    let pc = PcAlgorithm::run(data, n, d, alpha)?;
    let learned_skel: Vec<(usize, usize)> = pc.skeleton.clone();
    let score = skeleton_score(&learned_skel, &sem.true_skeleton());
    // Directed edges from the CPDAG (third element = oriented flag).
    let directed: Vec<(usize, usize)> = pc
        .cpdag
        .iter()
        .filter(|&&(_, _, oriented)| oriented)
        .map(|&(a, b, _)| (a, b))
        .collect();
    let (correct, total) = orientation_accuracy(&directed, &sem.true_edges());
    Ok(PcRecoveryReport {
        skeleton_f1: score.f1,
        skeleton_recall: score.recall,
        skeleton_precision: score.precision,
        correct_oriented: correct,
        oriented_total: total,
        skeleton_edges: learned_skel,
        directed_edges: directed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::verification::synthetic::{chain_sem, collider_sem, random_dag_sem};

    #[test]
    fn recovers_chain_skeleton_exactly() {
        // X0 -> X1 -> X2: PC must find exactly the two adjacent edges and drop
        // the (0,2) edge thanks to X0 ⟂ X2 | X1.
        let sem = chain_sem(3, 0.9);
        let mut rng = LcgRng::new(101);
        let n = 1500;
        let data = sem.sample(n, 0.3, &mut rng);
        let rep = recover(&sem, &data, n, 0.01).expect("pc recover");
        assert!(
            (rep.skeleton_recall - 1.0).abs() < 1e-9,
            "chain recall {}; edges {:?}",
            rep.skeleton_recall,
            rep.skeleton_edges
        );
        assert!(
            rep.skeleton_precision >= 0.9,
            "chain precision {}; edges {:?}",
            rep.skeleton_precision,
            rep.skeleton_edges
        );
    }

    #[test]
    fn orients_collider_v_structure() {
        // 0 -> 2 <- 1 (with 2 -> 3). PC must orient the collider at 2 because
        // 0 ⟂ 1 (empty set) but 0 ⊥̸ 1 | 2, so 2 is NOT in the sepset of (0,1).
        let sem = collider_sem(1.0);
        let mut rng = LcgRng::new(202);
        let n = 2000;
        let data = sem.sample(n, 0.4, &mut rng);
        let rep = recover(&sem, &data, n, 0.01).expect("pc recover");
        // Skeleton: edges {0-2, 1-2, 2-3}; (0,1) must be absent.
        assert!(
            rep.skeleton_recall >= 0.66,
            "collider skeleton recall {}; edges {:?}",
            rep.skeleton_recall,
            rep.skeleton_edges
        );
        assert!(
            !rep.skeleton_edges.contains(&(0, 1)),
            "0-1 should be non-adjacent; edges {:?}",
            rep.skeleton_edges
        );
        // The collider arrows 0->2 and 1->2 should be among the oriented edges.
        let has_0_to_2 = rep.directed_edges.contains(&(0, 2));
        let has_1_to_2 = rep.directed_edges.contains(&(1, 2));
        assert!(
            has_0_to_2 && has_1_to_2,
            "collider not oriented: directed {:?}",
            rep.directed_edges
        );
    }

    #[test]
    fn recovers_small_random_dag_skeleton() {
        // A 5-node DAG: PC skeleton F1 should be high with abundant data.
        let mut rng = LcgRng::new(303);
        let sem = random_dag_sem(5, 0.35, &mut rng);
        let n = 2500;
        let data = sem.sample(n, 0.3, &mut rng);
        let rep = recover(&sem, &data, n, 0.01).expect("pc recover");
        assert!(
            rep.skeleton_f1 >= 0.6,
            "random-DAG skeleton F1 {}; learned {:?} truth {:?}",
            rep.skeleton_f1,
            rep.skeleton_edges,
            sem.true_skeleton()
        );
    }

    #[test]
    fn no_false_edges_on_independent_columns() {
        // Three mutually-independent Gaussians: PC should return ~no edges.
        let sem = LinearSem {
            d: 4,
            w: vec![0.0_f32; 16],
            topo: (0..4).collect(),
        };
        let mut rng = LcgRng::new(404);
        let n = 1500;
        let data = sem.sample(n, 1.0, &mut rng);
        let rep = recover(&sem, &data, n, 0.01).expect("pc recover");
        assert!(
            rep.skeleton_edges.len() <= 1,
            "spurious edges on independent data: {:?}",
            rep.skeleton_edges
        );
    }
}
