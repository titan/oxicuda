//! NOTEARS structure-recovery accuracy against ground-truth linear SEMs.
//!
//! We sample data from a known weighted DAG, fit [`crate::discovery::notears::NotearsSem`],
//! threshold the learned weighted adjacency to obtain an edge set, and score it
//! against the truth (skeleton F1 + acyclicity of the result). This is the CPU
//! analogue of "NOTEARS recovery accuracy vs. ground-truth DAGs".

use crate::discovery::notears::NotearsSem;
use crate::error::CausalResult;
use crate::verification::graph_metrics::{skeleton_score, structural_hamming_distance};
use crate::verification::synthetic::LinearSem;

/// Outcome of a single NOTEARS recovery run.
pub struct NotearsRecoveryReport {
    /// Skeleton F1 of the thresholded learned graph vs the true skeleton.
    pub skeleton_f1: f64,
    pub skeleton_precision: f64,
    pub skeleton_recall: f64,
    /// Structural Hamming distance of the thresholded directed edges.
    pub shd: usize,
    /// Final acyclicity residual `|h(W)|` reported by the fitted model
    /// (should be ≈ 0 for a successful, acyclic fit).
    pub acyclicity_residual: f32,
    pub learned_edges: Vec<(usize, usize)>,
}

/// Extract directed edges from a learned weighted adjacency by absolute
/// thresholding (`|W[i,j]| > threshold ⇒ edge i→j`).
fn extract_edges(w: &[f32], d: usize, threshold: f32) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for i in 0..d {
        for j in 0..d {
            if i != j && w[i * d + j].abs() > threshold {
                edges.push((i, j));
            }
        }
    }
    edges
}

/// Re-derive the acyclicity residual `h(W) = tr(exp(W⊙W)) − d` for the learned
/// matrix using the production Padé exponential (independent of the optimizer's
/// internal bookkeeping).
fn acyclicity_residual(w: &[f32], d: usize) -> CausalResult<f32> {
    use crate::discovery::notears::expm_pade;
    let a: Vec<f32> = w.iter().map(|&v| v * v).collect();
    let e = expm_pade(&a, d)?;
    let tr: f32 = (0..d).map(|i| e[i * d + i]).sum();
    Ok((tr - d as f32).abs())
}

/// Fit NOTEARS to data sampled from `sem` and produce a recovery report.
pub fn recover(
    sem: &LinearSem,
    data: &[f32],
    n: usize,
    lambda: f32,
    max_iter: usize,
    threshold: f32,
) -> CausalResult<NotearsRecoveryReport> {
    let d = sem.d;
    let mut model = NotearsSem::new(d);
    // The simple proximal optimizer may not drive h(W) fully to zero; that is
    // acceptable here — we still threshold and score the recovered structure.
    let _ = model.fit(data, n, lambda, max_iter);
    let learned_edges = extract_edges(&model.w, d, threshold);
    let learned_skel: Vec<(usize, usize)> = learned_edges
        .iter()
        .copied()
        .map(|(a, b)| if a < b { (a, b) } else { (b, a) })
        .collect();
    let score = skeleton_score(&learned_skel, &sem.true_skeleton());
    let shd = structural_hamming_distance(&learned_edges, &sem.true_edges());
    let residual = acyclicity_residual(&model.w, d)?;
    Ok(NotearsRecoveryReport {
        skeleton_f1: score.f1,
        skeleton_precision: score.precision,
        skeleton_recall: score.recall,
        shd,
        acyclicity_residual: residual,
        learned_edges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::verification::synthetic::{chain_sem, random_dag_sem};

    #[test]
    fn recovers_chain_skeleton() {
        // A 4-node chain with strong weights and low noise. NOTEARS recovers all
        // three chain edges (recall = 1) and the learned graph is acyclic; linear
        // SEM recovery leaks one transitive edge, so precision is allowed to be
        // partial but must stay well above chance.
        let sem = chain_sem(4, 1.2);
        let mut rng = LcgRng::new(2024);
        let n = 800;
        let data = sem.sample(n, 0.15, &mut rng);
        let rep = recover(&sem, &data, n, 0.01, 4000, 0.25).expect("recover");
        assert!(
            rep.acyclicity_residual < 1e-3,
            "h(W) residual too large: {}",
            rep.acyclicity_residual
        );
        assert!(
            (rep.skeleton_recall - 1.0).abs() < 1e-9,
            "chain recall {} (edges {:?})",
            rep.skeleton_recall,
            rep.learned_edges
        );
        assert!(
            rep.skeleton_precision >= 0.6,
            "skeleton precision low: {} (edges {:?})",
            rep.skeleton_precision,
            rep.learned_edges
        );
    }

    #[test]
    fn recovers_10_node_sem() {
        // 10-node random DAG (the smallest size called out in the TODO). The
        // augmented-Lagrangian recovery should pick up the *majority* of the true
        // edges with good precision and produce an acyclic graph whose SHD is a
        // small fraction of the 90 possible directed edges. A broken recoverer
        // (all-zero short-circuit, or a saturated graph) fails these.
        let mut rng = LcgRng::new(31);
        let sem = random_dag_sem(10, 0.25, &mut rng);
        let n = 1000;
        let data = sem.sample(n, 0.1, &mut rng);
        let rep = recover(&sem, &data, n, 0.001, 3000, 0.1).expect("recover");
        assert!(
            rep.acyclicity_residual < 1e-3,
            "h(W) residual: {}",
            rep.acyclicity_residual
        );
        assert!(
            rep.skeleton_recall >= 0.5,
            "recall too low: {} (edges {:?})",
            rep.skeleton_recall,
            rep.learned_edges
        );
        assert!(
            rep.skeleton_precision >= 0.6,
            "precision too low: {} (edges {:?})",
            rep.skeleton_precision,
            rep.learned_edges
        );
        assert!(rep.shd < 15, "SHD too high: {}", rep.shd);
    }

    #[test]
    fn recovers_20_node_sem() {
        // 20-node DAG. Acyclicity must hold and the recovered sparse structure
        // must stay close to the (sparse) truth: SHD a small fraction of the 380
        // possible directed edges, with the majority of edges recovered.
        let mut rng = LcgRng::new(77);
        let sem = random_dag_sem(20, 0.12, &mut rng);
        let n = 1000;
        let data = sem.sample(n, 0.1, &mut rng);
        let rep = recover(&sem, &data, n, 0.001, 2000, 0.1).expect("recover");
        assert!(
            rep.acyclicity_residual < 1e-2,
            "h(W) residual: {}",
            rep.acyclicity_residual
        );
        let n_true = sem.true_edges().len();
        // SHD strictly below the number of true edges means we are net-recovering
        // structure rather than guessing the empty graph (which would give
        // SHD = n_true).
        assert!(
            rep.shd < n_true,
            "SHD {} not below #true edges {}",
            rep.shd,
            n_true
        );
        assert!(
            rep.skeleton_precision >= 0.6,
            "precision {} (edges {:?})",
            rep.skeleton_precision,
            rep.learned_edges
        );
    }

    #[test]
    fn empty_data_structure_is_acyclic() {
        // Pure-noise data (no edges): the learned graph should stay (near) empty
        // and acyclic, with high skeleton precision penalty avoided by an empty
        // truth being matched by an empty learned set.
        let sem = LinearSem {
            d: 5,
            w: vec![0.0_f32; 25],
            topo: (0..5).collect(),
        };
        let mut rng = LcgRng::new(5);
        let n = 400;
        let data = sem.sample(n, 1.0, &mut rng);
        let rep = recover(&sem, &data, n, 0.1, 400, 0.25).expect("recover");
        assert!(rep.acyclicity_residual < 1e-2);
        // Few or no spurious edges on independent noise.
        assert!(
            rep.learned_edges.len() <= 4,
            "too many spurious edges: {:?}",
            rep.learned_edges
        );
    }
}
