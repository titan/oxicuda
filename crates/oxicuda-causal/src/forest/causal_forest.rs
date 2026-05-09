use crate::error::{CausalError, CausalResult};
use crate::handle::LcgRng;

struct TreeNode {
    feature: usize,
    threshold: f32,
    left_child: Option<usize>,
    right_child: Option<usize>,
    cate: f32,
}

struct CausalTree {
    nodes: Vec<TreeNode>,
}

pub struct CausalForest {
    pub n_trees: usize,
    trees: Vec<CausalTree>,
    pub n_features: usize,
    min_samples: usize,
}

fn leaf_cate(indices: &[usize], t: &[f32], y: &[f32]) -> f32 {
    let treated: Vec<usize> = indices.iter().copied().filter(|&i| t[i] >= 0.5).collect();
    let control: Vec<usize> = indices.iter().copied().filter(|&i| t[i] < 0.5).collect();
    if treated.is_empty() || control.is_empty() {
        return 0.0;
    }
    let mean1 = treated.iter().map(|&i| y[i]).sum::<f32>() / treated.len() as f32;
    let mean0 = control.iter().map(|&i| y[i]).sum::<f32>() / control.len() as f32;
    mean1 - mean0
}

fn split_score(left: &[usize], right: &[usize], t: &[f32], y: &[f32]) -> f32 {
    if left.is_empty() || right.is_empty() {
        return f32::NEG_INFINITY;
    }
    let tau_l = leaf_cate(left, t, y);
    let tau_r = leaf_cate(right, t, y);
    let n = (left.len() + right.len()) as f32;
    let n_l = left.len() as f32;
    let n_r = right.len() as f32;
    (tau_l - tau_r).powi(2) * n_l * n_r / n
}

fn build_tree(
    build_idx: &[usize],
    est_idx: &[usize],
    x: &[f32],
    t: &[f32],
    y: &[f32],
    n_features: usize,
    min_samples: usize,
    feat_subset: &[usize],
    nodes: &mut Vec<TreeNode>,
) -> usize {
    let node_id = nodes.len();
    let cate = leaf_cate(est_idx, t, y);
    nodes.push(TreeNode {
        feature: 0,
        threshold: 0.0,
        left_child: None,
        right_child: None,
        cate,
    });

    if build_idx.len() < min_samples * 2 || est_idx.len() < min_samples * 2 {
        return node_id;
    }

    let mut best_score = f32::NEG_INFINITY;
    let mut best_feat = 0;
    let mut best_thresh = 0.0_f32;

    for &feat in feat_subset {
        // Try each unique threshold in sorted values of this feature on build set
        let mut vals: Vec<f32> = build_idx
            .iter()
            .map(|&i| x[i * n_features + feat])
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vals.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        if vals.len() < 2 {
            continue;
        }

        for w in vals.windows(2) {
            let thresh = (w[0] + w[1]) * 0.5;
            let left_b: Vec<usize> = build_idx
                .iter()
                .copied()
                .filter(|&i| x[i * n_features + feat] < thresh)
                .collect();
            let right_b: Vec<usize> = build_idx
                .iter()
                .copied()
                .filter(|&i| x[i * n_features + feat] >= thresh)
                .collect();
            let score = split_score(&left_b, &right_b, t, y);
            if score > best_score {
                best_score = score;
                best_feat = feat;
                best_thresh = thresh;
            }
        }
    }

    if best_score <= 0.0 {
        return node_id;
    }

    let left_build: Vec<usize> = build_idx
        .iter()
        .copied()
        .filter(|&i| x[i * n_features + best_feat] < best_thresh)
        .collect();
    let right_build: Vec<usize> = build_idx
        .iter()
        .copied()
        .filter(|&i| x[i * n_features + best_feat] >= best_thresh)
        .collect();
    let left_est: Vec<usize> = est_idx
        .iter()
        .copied()
        .filter(|&i| x[i * n_features + best_feat] < best_thresh)
        .collect();
    let right_est: Vec<usize> = est_idx
        .iter()
        .copied()
        .filter(|&i| x[i * n_features + best_feat] >= best_thresh)
        .collect();

    if left_build.len() < min_samples
        || right_build.len() < min_samples
        || left_est.is_empty()
        || right_est.is_empty()
    {
        return node_id;
    }

    let left_id = build_tree(
        &left_build,
        &left_est,
        x,
        t,
        y,
        n_features,
        min_samples,
        feat_subset,
        nodes,
    );
    let right_id = build_tree(
        &right_build,
        &right_est,
        x,
        t,
        y,
        n_features,
        min_samples,
        feat_subset,
        nodes,
    );

    nodes[node_id].feature = best_feat;
    nodes[node_id].threshold = best_thresh;
    nodes[node_id].left_child = Some(left_id);
    nodes[node_id].right_child = Some(right_id);

    node_id
}

fn predict_tree(tree: &CausalTree, x: &[f32], _n_features: usize) -> f32 {
    let mut node_id = 0;
    loop {
        let node = &tree.nodes[node_id];
        match (node.left_child, node.right_child) {
            (Some(left), Some(right)) => {
                if x[node.feature] < node.threshold {
                    node_id = left;
                } else {
                    node_id = right;
                }
            }
            _ => return node.cate,
        }
    }
}

impl CausalForest {
    pub fn new(n_trees: usize, n_features: usize, min_samples: usize, _rng: &mut LcgRng) -> Self {
        Self {
            n_trees,
            trees: Vec::new(),
            n_features,
            min_samples,
        }
    }

    pub fn fit(&mut self, x: &[f32], t: &[f32], y: &[f32], n: usize) -> CausalResult<()> {
        if x.is_empty() || n == 0 {
            return Err(CausalError::EmptyInput);
        }
        if x.len() != n * self.n_features || t.len() != n || y.len() != n {
            return Err(CausalError::IncompatibleData);
        }

        let mut rng = LcgRng::new(54321);
        let feat_sub_size = ((self.n_features as f32).sqrt() as usize).max(1);
        let sub_size = (n as f32 * 0.75) as usize;

        self.trees.clear();
        for _ in 0..self.n_trees {
            // Random subsample (with replacement)
            let sample_idx: Vec<usize> = (0..sub_size).map(|_| rng.next_usize(n)).collect();

            // Split into build and estimate sets (50/50 from subsample)
            let half = sub_size / 2;
            let build_idx = sample_idx[..half].to_vec();
            let est_idx = sample_idx[half..].to_vec();

            // Random feature subset
            let mut feat_subset: Vec<usize> = (0..self.n_features).collect();
            // Shuffle by partial random swaps
            for i in (1..self.n_features).rev() {
                let j = rng.next_usize(i + 1);
                feat_subset.swap(i, j);
            }
            let feat_subset = &feat_subset[..feat_sub_size];

            let mut nodes = Vec::new();
            build_tree(
                &build_idx,
                &est_idx,
                x,
                t,
                y,
                self.n_features,
                self.min_samples,
                feat_subset,
                &mut nodes,
            );
            self.trees.push(CausalTree { nodes });
        }
        Ok(())
    }

    pub fn predict(&self, x: &[f32], n: usize) -> CausalResult<Vec<f32>> {
        if self.trees.is_empty() {
            return Err(CausalError::NotFitted);
        }
        if x.len() != n * self.n_features {
            return Err(CausalError::DimensionMismatch {
                expected: n * self.n_features,
                got: x.len(),
            });
        }
        Ok((0..n)
            .map(|i| {
                let xi = &x[i * self.n_features..(i + 1) * self.n_features];
                let sum: f32 = self
                    .trees
                    .iter()
                    .map(|tree| predict_tree(tree, xi, self.n_features))
                    .sum();
                sum / self.trees.len() as f32
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_forest_fit_predict() {
        let mut rng = LcgRng::new(100);
        let n = 40;
        let d = 3;
        let x: Vec<f32> = (0..n * d).map(|i| i as f32 / (n * d) as f32).collect();
        let t: Vec<f32> = (0..n).map(|i| if i < n / 2 { 1.0 } else { 0.0 }).collect();
        let y: Vec<f32> = (0..n).map(|i| x[i * d] + t[i]).collect();
        let mut forest = CausalForest::new(5, d, 3, &mut rng);
        forest.fit(&x, &t, &y, n).unwrap();
        let preds = forest.predict(&x, n).unwrap();
        assert_eq!(preds.len(), n);
    }
}
