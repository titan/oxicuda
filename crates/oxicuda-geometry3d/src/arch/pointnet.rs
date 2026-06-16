//! PointNet architecture for 3D point cloud classification.

use crate::error::{Geom3dError, Geom3dResult};
use crate::handle::LcgRng;

// ─── Helper: linear layer ────────────────────────────────────────────────────

/// Dense linear layer: `out[i] = relu(Σ_j w[i,j] * in[j] + b[i])` or without relu.
fn linear_relu(
    input: &[f32],
    weights: &[f32],
    biases: &[f32],
    in_dim: usize,
    out_dim: usize,
    use_relu: bool,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; out_dim];
    for i in 0..out_dim {
        let mut acc = biases[i];
        for j in 0..in_dim {
            acc += weights[i * in_dim + j] * input[j];
        }
        out[i] = if use_relu { acc.max(0.0) } else { acc };
    }
    out
}

/// Per-point shared MLP (no bias): `out[p, i] = relu(Σ_j w[i,j] * in[p,j] + b[i])`.
fn shared_mlp(
    points: &[f32],
    n: usize,
    in_dim: usize,
    out_dim: usize,
    weights: &[f32],
    biases: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0_f32; n * out_dim];
    for p in 0..n {
        let feat = &points[p * in_dim..(p + 1) * in_dim];
        for i in 0..out_dim {
            let mut acc = biases[i];
            for j in 0..in_dim {
                acc += weights[i * in_dim + j] * feat[j];
            }
            out[p * out_dim + i] = acc.max(0.0);
        }
    }
    out
}

/// Global max pool over N points with C features.
fn global_max_pool(features: &[f32], n: usize, c: usize) -> Vec<f32> {
    let mut pooled = vec![f32::NEG_INFINITY; c];
    for p in 0..n {
        for ch in 0..c {
            let v = features[p * c + ch];
            if v > pooled[ch] {
                pooled[ch] = v;
            }
        }
    }
    // Replace any remaining -inf with 0.0 (empty N case guarded by caller)
    for v in &mut pooled {
        if *v == f32::NEG_INFINITY {
            *v = 0.0;
        }
    }
    pooled
}

// ─── T-Net (Input Transform) ─────────────────────────────────────────────────

struct TNet {
    // 3→64→128→1024 per-point shared MLPs
    w1: Vec<f32>, // [64×3]
    b1: Vec<f32>, // [64]
    w2: Vec<f32>, // [128×64]
    b2: Vec<f32>, // [128]
    w3: Vec<f32>, // [1024×128]
    b3: Vec<f32>, // [1024]
    // FC: 1024→512→256→9
    fc1w: Vec<f32>, // [512×1024]
    fc1b: Vec<f32>, // [512]
    fc2w: Vec<f32>, // [256×512]
    fc2b: Vec<f32>, // [256]
    fc3w: Vec<f32>, // [9×256]
    fc3b: Vec<f32>, // [9] = identity flatten [1,0,0,0,1,0,0,0,1]
}

impl TNet {
    fn new(rng: &mut LcgRng) -> Self {
        let mut w1 = vec![0.0_f32; 64 * 3];
        rng.fill_xavier_uniform(&mut w1, 3, 64);
        let b1 = vec![0.0_f32; 64];

        let mut w2 = vec![0.0_f32; 128 * 64];
        rng.fill_xavier_uniform(&mut w2, 64, 128);
        let b2 = vec![0.0_f32; 128];

        let mut w3 = vec![0.0_f32; 1024 * 128];
        rng.fill_xavier_uniform(&mut w3, 128, 1024);
        let b3 = vec![0.0_f32; 1024];

        let mut fc1w = vec![0.0_f32; 512 * 1024];
        rng.fill_xavier_uniform(&mut fc1w, 1024, 512);
        let fc1b = vec![0.0_f32; 512];

        let mut fc2w = vec![0.0_f32; 256 * 512];
        rng.fill_xavier_uniform(&mut fc2w, 512, 256);
        let fc2b = vec![0.0_f32; 256];

        // Last layer: zero weights, identity bias
        let fc3w = vec![0.0_f32; 9 * 256];
        let fc3b = vec![
            1.0_f32, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, // row 1
            0.0, 0.0, 1.0, // row 2
        ];

        Self {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            fc1w,
            fc1b,
            fc2w,
            fc2b,
            fc3w,
            fc3b,
        }
    }

    /// Forward: input [n×3] → 3×3 transform matrix (row-major [9]).
    fn forward(&self, points: &[f32], n: usize) -> Vec<f32> {
        // Per-point shared MLP
        let h1 = shared_mlp(points, n, 3, 64, &self.w1, &self.b1);
        let h2 = shared_mlp(&h1, n, 64, 128, &self.w2, &self.b2);
        let h3 = shared_mlp(&h2, n, 128, 1024, &self.w3, &self.b3);

        // Global max pool
        let global = global_max_pool(&h3, n, 1024);

        // FC layers
        let fc1 = linear_relu(&global, &self.fc1w, &self.fc1b, 1024, 512, true);
        let fc2 = linear_relu(&fc1, &self.fc2w, &self.fc2b, 512, 256, true);
        linear_relu(&fc2, &self.fc3w, &self.fc3b, 256, 9, false)
    }

    /// Apply 3×3 transform to N×3 points.
    fn apply_transform(&self, mat: &[f32], points: &[f32], n: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; n * 3];
        for p in 0..n {
            for i in 0..3 {
                let mut v = 0.0_f32;
                for j in 0..3 {
                    v += mat[i * 3 + j] * points[p * 3 + j];
                }
                out[p * 3 + i] = v;
            }
        }
        out
    }
}

// ─── PointNet ─────────────────────────────────────────────────────────────────

/// Configuration for PointNet.
#[derive(Debug, Clone)]
pub struct PointNetConfig {
    pub n_points: usize,
    pub n_classes: usize,
}

/// PointNet for classification: input \[N×3\] → logits \[n_classes\].
///
/// Architecture:
/// - T-Net (3×3 input transform)
/// - Shared MLP: 3→64→128→1024 per point
/// - Global max pool
/// - FC: 1024→512→256→n_classes
pub struct PointNet {
    config: PointNetConfig,
    tnet: TNet,
    // Shared MLP weights: [n×3] → [n×64] → [n×128] → [n×1024]
    mlp1_w: Vec<f32>, // [64×3]
    mlp1_b: Vec<f32>, // [64]
    mlp2_w: Vec<f32>, // [128×64]
    mlp2_b: Vec<f32>, // [128]
    mlp3_w: Vec<f32>, // [1024×128]
    mlp3_b: Vec<f32>, // [1024]
    // FC classifier
    fc1_w: Vec<f32>, // [512×1024]
    fc1_b: Vec<f32>, // [512]
    fc2_w: Vec<f32>, // [256×512]
    fc2_b: Vec<f32>, // [256]
    fc3_w: Vec<f32>, // [n_classes×256]
    fc3_b: Vec<f32>, // [n_classes]
}

impl PointNet {
    /// Create a new PointNet with Xavier-uniform weight initialization.
    pub fn new(config: PointNetConfig, rng: &mut LcgRng) -> Self {
        let nc = config.n_classes;

        let tnet = TNet::new(rng);

        let mut mlp1_w = vec![0.0_f32; 64 * 3];
        rng.fill_xavier_uniform(&mut mlp1_w, 3, 64);
        let mlp1_b = vec![0.0_f32; 64];

        let mut mlp2_w = vec![0.0_f32; 128 * 64];
        rng.fill_xavier_uniform(&mut mlp2_w, 64, 128);
        let mlp2_b = vec![0.0_f32; 128];

        let mut mlp3_w = vec![0.0_f32; 1024 * 128];
        rng.fill_xavier_uniform(&mut mlp3_w, 128, 1024);
        let mlp3_b = vec![0.0_f32; 1024];

        let mut fc1_w = vec![0.0_f32; 512 * 1024];
        rng.fill_xavier_uniform(&mut fc1_w, 1024, 512);
        let fc1_b = vec![0.0_f32; 512];

        let mut fc2_w = vec![0.0_f32; 256 * 512];
        rng.fill_xavier_uniform(&mut fc2_w, 512, 256);
        let fc2_b = vec![0.0_f32; 256];

        let mut fc3_w = vec![0.0_f32; nc * 256];
        rng.fill_xavier_uniform(&mut fc3_w, 256, nc);
        let fc3_b = vec![0.0_f32; nc];

        Self {
            config,
            tnet,
            mlp1_w,
            mlp1_b,
            mlp2_w,
            mlp2_b,
            mlp3_w,
            mlp3_b,
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
            fc3_w,
            fc3_b,
        }
    }

    /// Forward pass: `points [N×3]` → class logits `[n_classes]`.
    pub fn forward(&self, points: &[f32]) -> Geom3dResult<Vec<f32>> {
        let n = self.config.n_points;
        let nc = self.config.n_classes;

        if points.len() != n * 3 {
            return Err(Geom3dError::DimensionMismatch {
                expected: n * 3,
                got: points.len(),
            });
        }
        if n == 0 {
            return Err(Geom3dError::EmptyPointCloud);
        }

        // T-Net input transform
        let tmat = self.tnet.forward(points, n);
        let transformed = self.tnet.apply_transform(&tmat, points, n);

        // Shared MLP feature extraction
        let h1 = shared_mlp(&transformed, n, 3, 64, &self.mlp1_w, &self.mlp1_b);
        let h2 = shared_mlp(&h1, n, 64, 128, &self.mlp2_w, &self.mlp2_b);
        let h3 = shared_mlp(&h2, n, 128, 1024, &self.mlp3_w, &self.mlp3_b);

        // Global max pool
        let global = global_max_pool(&h3, n, 1024);

        // Classifier
        let fc1 = linear_relu(&global, &self.fc1_w, &self.fc1_b, 1024, 512, true);
        let fc2 = linear_relu(&fc1, &self.fc2_w, &self.fc2_b, 512, 256, true);
        let logits = linear_relu(&fc2, &self.fc3_w, &self.fc3_b, 256, nc, false);

        Ok(logits)
    }

    /// Returns the predicted class index (argmax of logits).
    pub fn classify(&self, points: &[f32]) -> Geom3dResult<usize> {
        let logits = self.forward(points)?;
        let best = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or(Geom3dError::EmptyPointCloud)?;
        Ok(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_points(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut pts = vec![0.0_f32; n * 3];
        for v in &mut pts {
            *v = rng.next_f32() * 2.0 - 1.0;
        }
        pts
    }

    #[test]
    fn pointnet_forward_output_shape() {
        let cfg = PointNetConfig {
            n_points: 16,
            n_classes: 10,
        };
        let mut rng = LcgRng::new(42);
        let net = PointNet::new(cfg, &mut rng);
        let pts = make_points(16, 1);
        let logits = net.forward(&pts).expect("forward should succeed");
        assert_eq!(logits.len(), 10);
    }

    #[test]
    fn pointnet_forward_finite() {
        let cfg = PointNetConfig {
            n_points: 8,
            n_classes: 5,
        };
        let mut rng = LcgRng::new(42);
        let net = PointNet::new(cfg, &mut rng);
        let pts = make_points(8, 1);
        let logits = net.forward(&pts).expect("forward should succeed");
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "Logits must be finite"
        );
    }

    #[test]
    fn pointnet_classify_in_range() {
        let n_classes = 8;
        let cfg = PointNetConfig {
            n_points: 16,
            n_classes,
        };
        let mut rng = LcgRng::new(42);
        let net = PointNet::new(cfg, &mut rng);
        let pts = make_points(16, 2);
        let cls = net.classify(&pts).expect("classify should succeed");
        assert!(cls < n_classes, "Class must be in [0, n_classes)");
    }

    #[test]
    fn pointnet_deterministic_same_seed() {
        let cfg = PointNetConfig {
            n_points: 8,
            n_classes: 4,
        };
        let pts = make_points(8, 5);

        let mut rng1 = LcgRng::new(99);
        let net1 = PointNet::new(cfg.clone(), &mut rng1);

        let mut rng2 = LcgRng::new(99);
        let net2 = PointNet::new(cfg, &mut rng2);

        let l1 = net1.forward(&pts).expect("forward should succeed");
        let l2 = net2.forward(&pts).expect("forward should succeed");
        assert_eq!(l1, l2, "Same seed must produce identical output");
    }

    #[test]
    fn tnet_output_near_identity_at_init() {
        // With zero last-layer weights and identity bias, T-Net should output
        // a near-identity matrix (no input-dependent transformation)
        let mut rng = LcgRng::new(42);
        let tnet = TNet::new(&mut rng);
        let pts = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mat = tnet.forward(&pts, 3);
        // Should be close to [1,0,0,0,1,0,0,0,1]
        let identity = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for (a, &b) in mat.iter().zip(identity.iter()) {
            assert!(
                (a - b).abs() < 0.5,
                "T-Net should be near identity at init: got {:?}",
                mat
            );
        }
    }

    #[test]
    fn pointnet_dim_mismatch_error() {
        let cfg = PointNetConfig {
            n_points: 16,
            n_classes: 5,
        };
        let mut rng = LcgRng::new(42);
        let net = PointNet::new(cfg, &mut rng);
        // Pass wrong size
        let pts = vec![0.0_f32; 12]; // should be 16*3=48
        assert!(net.forward(&pts).is_err());
    }
}
