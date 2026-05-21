//! Principal Neighbourhood Aggregation (PNA) layer — Corso et al. 2020 (NeurIPS).
//!
//! PNA combines multiple aggregation functions with multiple degree-dependent scalers,
//! then projects the concatenated result through a two-layer MLP.

use crate::error::{GnnError, GnnResult};
use crate::graph::csr::CsrGraph;

// ─── Aggregators ─────────────────────────────────────────────────────────────

/// Neighbourhood aggregation functions for PNA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PnaAggregator {
    /// Sum of neighbor features.
    Sum,
    /// Mean of neighbor features.
    Mean,
    /// Component-wise maximum.
    Max,
    /// Component-wise minimum.
    Min,
    /// Standard deviation of neighbor features.
    Std,
}

// ─── Scalers ──────────────────────────────────────────────────────────────────

/// Degree-dependent scalers for PNA (modulate aggregated features).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PnaScaler {
    /// No scaling (multiply by 1).
    Identity,
    /// Amplification: scale by log(d+1) / log(δ+1+ε) — amplifies high-degree nodes.
    Amplification,
    /// Attenuation: scale by log(δ+1+ε) / log(d+1+ε) — attenuates high-degree nodes.
    Attenuation,
    /// Linear amplification per Corso 2020.
    LinearAmplification,
    /// Linear attenuation (inverse of linear amplification).
    LinearAttenuation,
}

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for a PNA layer.
#[derive(Debug, Clone)]
pub struct PnaConfig {
    /// Input feature dimension.
    pub in_features: usize,
    /// Hidden dimension of the two-layer MLP (applied to concatenated aggregator outputs).
    pub hidden_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// List of aggregators to use.
    pub aggregators: Vec<PnaAggregator>,
    /// List of scalers to combine with each aggregator.
    pub scalers: Vec<PnaScaler>,
    /// Average degree δ used for scaler computation.
    pub delta: f32,
}

impl Default for PnaConfig {
    fn default() -> Self {
        Self {
            in_features: 16,
            hidden_features: 64,
            out_features: 16,
            aggregators: vec![
                PnaAggregator::Sum,
                PnaAggregator::Mean,
                PnaAggregator::Max,
                PnaAggregator::Std,
            ],
            scalers: vec![
                PnaScaler::Identity,
                PnaScaler::Amplification,
                PnaScaler::Attenuation,
            ],
            delta: 1.0,
        }
    }
}

// ─── Layer ────────────────────────────────────────────────────────────────────

/// A PNA layer (inference only — weight parameters are passed externally).
#[derive(Debug, Clone)]
pub struct PnaLayer {
    /// Layer configuration.
    pub config: PnaConfig,
}

impl PnaLayer {
    /// Construct a PNA layer from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `GnnError::InvalidLayerConfig` if any dimension is zero, the
    /// aggregator/scaler lists are empty, or `delta` is not positive.
    pub fn new(config: PnaConfig) -> GnnResult<Self> {
        if config.in_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "in_features must be > 0".to_string(),
            ));
        }
        if config.hidden_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "hidden_features must be > 0".to_string(),
            ));
        }
        if config.out_features == 0 {
            return Err(GnnError::InvalidLayerConfig(
                "out_features must be > 0".to_string(),
            ));
        }
        if config.aggregators.is_empty() {
            return Err(GnnError::InvalidLayerConfig(
                "aggregators must be non-empty".to_string(),
            ));
        }
        if config.scalers.is_empty() {
            return Err(GnnError::InvalidLayerConfig(
                "scalers must be non-empty".to_string(),
            ));
        }
        if config.delta <= 0.0 {
            return Err(GnnError::InvalidLayerConfig(
                "delta must be > 0".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Dimensionality of the concatenated aggregator input to the MLP.
    ///
    /// = `in_features * aggregators.len() * scalers.len()`
    #[inline]
    pub fn mlp_in_dim(&self) -> usize {
        self.config.in_features * self.config.aggregators.len() * self.config.scalers.len()
    }

    /// PNA forward pass.
    ///
    /// For each node v:
    /// 1. For each aggregator A: compute `agg_A(h_u : u ∈ N(v))` → `[in_features]`
    /// 2. For each `(A, S)` pair: scale using the degree-based scaler `S(d_v)`
    /// 3. Concatenate all `|aggregators| × |scalers|` scaled vectors
    /// 4. Apply two-layer MLP: Linear → ReLU → Linear
    ///
    /// The concatenated vector is ordered `[agg_0_scale_0, agg_0_scale_1, ..., agg_A_scale_S]`
    /// (inner loop over scalers).
    ///
    /// # Arguments
    /// - `graph`: CSR graph (out-neighbors of node v = N(v))
    /// - `x`: node features `[n_nodes × in_features]`, row-major
    /// - `w1`: MLP weight matrix `[hidden × mlp_in_dim]`, row-major
    /// - `b1`: MLP bias `[hidden]`
    /// - `w2`: MLP weight matrix `[out_features × hidden]`, row-major
    /// - `b2`: MLP bias `[out_features]`
    ///
    /// # Returns
    /// Updated node features `[n_nodes × out_features]`
    pub fn forward(
        &self,
        graph: &CsrGraph,
        x: &[f32],
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: &[f32],
    ) -> GnnResult<Vec<f32>> {
        let n = graph.n_nodes();
        let in_f = self.config.in_features;
        let hid = self.config.hidden_features;
        let out_f = self.config.out_features;
        let mlp_in = self.mlp_in_dim();
        let delta = self.config.delta;

        // Validate inputs
        if x.len() != n * in_f {
            return Err(GnnError::NodeFeatureMismatch(n, x.len() / in_f.max(1)));
        }
        if w1.len() != hid * mlp_in {
            return Err(GnnError::WeightShapeMismatch {
                r: hid,
                c: mlp_in,
                d: mlp_in,
            });
        }
        if b1.len() != hid {
            return Err(GnnError::DimensionMismatch {
                expected: hid,
                got: b1.len(),
            });
        }
        if w2.len() != out_f * hid {
            return Err(GnnError::WeightShapeMismatch {
                r: out_f,
                c: hid,
                d: hid,
            });
        }
        if b2.len() != out_f {
            return Err(GnnError::DimensionMismatch {
                expected: out_f,
                got: b2.len(),
            });
        }

        // Build output: [n × out_f]
        let mut output = vec![0.0_f32; n * out_f];

        for v in 0..n {
            let neighbors = graph.neighbors(v)?;
            let d = neighbors.len();

            // Gather neighbor feature rows as a flat slice [n_nbrs × in_f]
            let mut nbr_feats: Vec<f32> = Vec::with_capacity(d * in_f);
            for &u in neighbors {
                nbr_feats.extend_from_slice(&x[u * in_f..(u + 1) * in_f]);
            }

            // Build the concatenated aggregation vector [mlp_in]
            // Order: agg_0_scale_0, agg_0_scale_1, ..., agg_A_scale_S
            let mut concat = Vec::with_capacity(mlp_in);
            for &agg_kind in &self.config.aggregators {
                let agg_vec = aggregate(agg_kind, &nbr_feats, d, in_f);
                for &scaler_kind in &self.config.scalers {
                    let scaled = scale(scaler_kind, &agg_vec, d, delta);
                    concat.extend_from_slice(&scaled);
                }
            }

            // Two-layer MLP: [mlp_in] → [hid] → [out_f]
            // Layer 1: h = ReLU(W1 @ concat + b1)
            let mut h = vec![0.0_f32; hid];
            for k in 0..hid {
                let mut acc = b1[k];
                for j in 0..mlp_in {
                    acc += w1[k * mlp_in + j] * concat[j];
                }
                h[k] = acc.max(0.0); // ReLU
            }

            // Layer 2: out = W2 @ h + b2
            for k in 0..out_f {
                let mut acc = b2[k];
                for j in 0..hid {
                    acc += w2[k * hid + j] * h[j];
                }
                output[v * out_f + k] = acc;
            }
        }

        Ok(output)
    }
}

// ─── Aggregation helpers ──────────────────────────────────────────────────────

/// Apply a single aggregator to a set of neighbor feature vectors.
///
/// `neighbors` is a flat `[n_nbrs × in_features]` slice in row-major order.
/// Returns an aggregated vector of length `in_features`.
///
/// For empty neighborhoods: Sum→zeros, Mean→zeros, Max→zeros, Min→zeros, Std→zeros.
pub fn aggregate(
    aggregator: PnaAggregator,
    neighbors: &[f32], // n_nbrs × in_features, row-major; may be 0 rows
    n_nbrs: usize,
    in_features: usize,
) -> Vec<f32> {
    if n_nbrs == 0 || in_features == 0 {
        return vec![0.0_f32; in_features];
    }

    match aggregator {
        PnaAggregator::Sum => {
            let mut out = vec![0.0_f32; in_features];
            for i in 0..n_nbrs {
                for k in 0..in_features {
                    out[k] += neighbors[i * in_features + k];
                }
            }
            out
        }

        PnaAggregator::Mean => {
            let mut out = vec![0.0_f32; in_features];
            for i in 0..n_nbrs {
                for k in 0..in_features {
                    out[k] += neighbors[i * in_features + k];
                }
            }
            let inv_n = 1.0_f32 / n_nbrs as f32;
            for v in &mut out {
                *v *= inv_n;
            }
            out
        }

        PnaAggregator::Max => {
            let mut out = vec![f32::NEG_INFINITY; in_features];
            for i in 0..n_nbrs {
                for k in 0..in_features {
                    let v = neighbors[i * in_features + k];
                    if v > out[k] {
                        out[k] = v;
                    }
                }
            }
            // Clamp -inf to 0 for isolated nodes (already handled by early return above,
            // but guard against all-negative feature edge cases)
            for v in &mut out {
                if v.is_infinite() {
                    *v = 0.0;
                }
            }
            out
        }

        PnaAggregator::Min => {
            let mut out = vec![f32::INFINITY; in_features];
            for i in 0..n_nbrs {
                for k in 0..in_features {
                    let v = neighbors[i * in_features + k];
                    if v < out[k] {
                        out[k] = v;
                    }
                }
            }
            for v in &mut out {
                if v.is_infinite() {
                    *v = 0.0;
                }
            }
            out
        }

        PnaAggregator::Std => {
            // Two-pass: first mean, then variance
            if n_nbrs < 2 {
                return vec![0.0_f32; in_features];
            }
            let eps = 1e-7_f32;
            let inv_n = 1.0_f32 / n_nbrs as f32;

            // Pass 1: mean per feature
            let mut mean = vec![0.0_f32; in_features];
            for i in 0..n_nbrs {
                for k in 0..in_features {
                    mean[k] += neighbors[i * in_features + k];
                }
            }
            for m in &mut mean {
                *m *= inv_n;
            }

            // Pass 2: variance per feature
            let mut var = vec![0.0_f32; in_features];
            for i in 0..n_nbrs {
                for k in 0..in_features {
                    let diff = neighbors[i * in_features + k] - mean[k];
                    var[k] += diff * diff;
                }
            }
            // std = sqrt(var/n + eps)
            let mut out = vec![0.0_f32; in_features];
            for k in 0..in_features {
                out[k] = (var[k] * inv_n + eps).sqrt();
            }
            out
        }
    }
}

// ─── Scaling helpers ──────────────────────────────────────────────────────────

/// Apply a degree-dependent scaler to an aggregated vector.
///
/// `d`: degree of the current node, `delta`: average degree.
/// Returns a scaled vector of the same length as `agg`.
pub fn scale(scaler: PnaScaler, agg: &[f32], d: usize, delta: f32) -> Vec<f32> {
    let factor = compute_scale_factor(scaler, d, delta);
    agg.iter().map(|&v| v * factor).collect()
}

/// Compute the scalar multiplier for a given scaler, node degree, and average degree.
fn compute_scale_factor(scaler: PnaScaler, d: usize, delta: f32) -> f32 {
    const EPS: f32 = 1e-7;
    match scaler {
        PnaScaler::Identity => 1.0,

        PnaScaler::Amplification => {
            let log_d = (d as f32 + 1.0).ln();
            let log_delta = (delta + 1.0 + EPS).ln();
            log_d / log_delta
        }

        PnaScaler::Attenuation => {
            let log_d = (d as f32 + 1.0 + EPS).ln();
            let log_delta = (delta + 1.0 + EPS).ln();
            log_delta / log_d
        }

        PnaScaler::LinearAmplification => {
            if delta > 1.0 {
                (d as f32 - 1.0).max(0.0) / (delta - 1.0)
            } else {
                1.0
            }
        }

        PnaScaler::LinearAttenuation => {
            if delta > 1.0 {
                (delta - 1.0) / (d as f32 - 1.0).max(1.0)
            } else {
                1.0
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 3-node triangle: 0↔1↔2↔0 (bidirectional).
    fn triangle_graph() -> CsrGraph {
        CsrGraph::from_edges(3, &[(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])
            .expect("test invariant: valid graph")
    }

    /// Minimal valid weights for `in=2, hid=4, out=2` with `n_agg=1, n_scl=1`.
    fn minimal_weights(
        mlp_in: usize,
        hid: usize,
        out: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let w1 = vec![0.1_f32; hid * mlp_in];
        let b1 = vec![0.0_f32; hid];
        let w2 = vec![0.1_f32; out * hid];
        let b2 = vec![0.0_f32; out];
        (w1, b1, w2, b2)
    }

    // ─── aggregate tests ─────────────────────────────────────────────────────

    #[test]
    fn aggregate_sum_basic() {
        // 2 neighbors, in_features=2: [1,2] and [3,4] → sum = [4,6]
        let nbrs = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = aggregate(PnaAggregator::Sum, &nbrs, 2, 2);
        assert_eq!(out.len(), 2);
        assert!(
            (out[0] - 4.0).abs() < 1e-6,
            "sum[0] should be 4, got {}",
            out[0]
        );
        assert!(
            (out[1] - 6.0).abs() < 1e-6,
            "sum[1] should be 6, got {}",
            out[1]
        );
    }

    #[test]
    fn aggregate_mean_basic() {
        // 2 neighbors: [1,2] and [3,4] → mean = [2,3]
        let nbrs = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = aggregate(PnaAggregator::Mean, &nbrs, 2, 2);
        assert_eq!(out.len(), 2);
        assert!(
            (out[0] - 2.0).abs() < 1e-6,
            "mean[0] should be 2, got {}",
            out[0]
        );
        assert!(
            (out[1] - 3.0).abs() < 1e-6,
            "mean[1] should be 3, got {}",
            out[1]
        );
    }

    #[test]
    fn aggregate_max_basic() {
        // 2 neighbors: [1,4] and [3,2] → max = [3,4]
        let nbrs = vec![1.0_f32, 4.0, 3.0, 2.0];
        let out = aggregate(PnaAggregator::Max, &nbrs, 2, 2);
        assert_eq!(out.len(), 2);
        assert!(
            (out[0] - 3.0).abs() < 1e-6,
            "max[0] should be 3, got {}",
            out[0]
        );
        assert!(
            (out[1] - 4.0).abs() < 1e-6,
            "max[1] should be 4, got {}",
            out[1]
        );
    }

    #[test]
    fn aggregate_min_basic() {
        // 2 neighbors: [1,4] and [3,2] → min = [1,2]
        let nbrs = vec![1.0_f32, 4.0, 3.0, 2.0];
        let out = aggregate(PnaAggregator::Min, &nbrs, 2, 2);
        assert_eq!(out.len(), 2);
        assert!(
            (out[0] - 1.0).abs() < 1e-6,
            "min[0] should be 1, got {}",
            out[0]
        );
        assert!(
            (out[1] - 2.0).abs() < 1e-6,
            "min[1] should be 2, got {}",
            out[1]
        );
    }

    #[test]
    fn aggregate_std_identical_values_is_zero() {
        // n=2, in=1, both values identical → std = 0 (before eps sqrt)
        let nbrs = vec![5.0_f32, 5.0];
        let out = aggregate(PnaAggregator::Std, &nbrs, 2, 1);
        assert_eq!(out.len(), 1);
        // std = sqrt(0 + 1e-7) ≈ 3.16e-4, very close to 0
        assert!(
            out[0] < 1e-3,
            "std of identical values should be ~0, got {}",
            out[0]
        );
    }

    #[test]
    fn aggregate_std_known_values() {
        // n=2, in=1: values [1, 3] → mean=2, var=((1-2)²+(3-2)²)/2=1 → std≈1
        let nbrs = vec![1.0_f32, 3.0];
        let out = aggregate(PnaAggregator::Std, &nbrs, 2, 1);
        assert_eq!(out.len(), 1);
        // sqrt(1.0 + 1e-7) ≈ 1.0
        assert!(
            (out[0] - 1.0).abs() < 1e-3,
            "std should be ~1.0, got {}",
            out[0]
        );
    }

    #[test]
    fn aggregate_empty_neighbors_all_zeros() {
        for &agg in &[
            PnaAggregator::Sum,
            PnaAggregator::Mean,
            PnaAggregator::Max,
            PnaAggregator::Min,
            PnaAggregator::Std,
        ] {
            let out = aggregate(agg, &[], 0, 3);
            assert_eq!(
                out,
                vec![0.0_f32; 3],
                "empty neighbors with {agg:?} should return zeros"
            );
        }
    }

    #[test]
    fn aggregate_std_single_neighbor_is_zero() {
        // d < 2 → Std returns zeros
        let nbrs = vec![7.0_f32, 8.0]; // 1 neighbor, in_f=2
        let out = aggregate(PnaAggregator::Std, &nbrs, 1, 2);
        assert_eq!(out, vec![0.0_f32; 2]);
    }

    // ─── scale tests ─────────────────────────────────────────────────────────

    #[test]
    fn scale_identity_unchanged() {
        let agg = vec![1.0_f32, 2.0];
        let out = scale(PnaScaler::Identity, &agg, 4, 2.0);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn scale_amplification_amplifies_high_degree() {
        // d=10, delta=1.0 → log(11)/log(2+1e-7) ≈ 2.4/1.1 > 1
        let agg = vec![1.0_f32];
        let out = scale(PnaScaler::Amplification, &agg, 10, 1.0);
        assert!(
            out[0] > 1.0,
            "Amplification at d=10, delta=1 should give factor > 1, got {}",
            out[0]
        );
    }

    #[test]
    fn scale_attenuation_attenuates_high_degree() {
        // d=10, delta=1.0 → log(2+eps)/log(11+eps) < 1
        let agg = vec![1.0_f32];
        let out = scale(PnaScaler::Attenuation, &agg, 10, 1.0);
        assert!(
            out[0] < 1.0,
            "Attenuation at d=10, delta=1 should give factor < 1, got {}",
            out[0]
        );
    }

    #[test]
    fn scale_delta_equal_d_identity_like() {
        // When d == delta == 2, Amplification factor = log(3)/log(3+eps) ≈ 1
        let agg = vec![3.0_f32];
        let amp_out = scale(PnaScaler::Amplification, &agg, 2, 2.0);
        assert!(
            (amp_out[0] - 3.0).abs() < 0.1,
            "Amplification with d=delta should be ~identity, got {}",
            amp_out[0]
        );
    }

    #[test]
    fn scale_linear_amplification_zero_degree() {
        // d=0, delta=5 → max(0-1, 0)/(5-1) = 0
        let agg = vec![1.0_f32, 2.0];
        let out = scale(PnaScaler::LinearAmplification, &agg, 0, 5.0);
        assert!(
            (out[0]).abs() < 1e-6,
            "LinearAmplification at d=0 should give 0, got {}",
            out[0]
        );
    }

    #[test]
    fn scale_linear_attenuation_delta_le_1() {
        // delta <= 1 → factor = 1 (identity)
        let agg = vec![4.0_f32];
        let out = scale(PnaScaler::LinearAttenuation, &agg, 10, 0.5);
        assert!(
            (out[0] - 4.0).abs() < 1e-6,
            "LinearAttenuation with delta<=1 should be identity, got {}",
            out[0]
        );
    }

    // ─── PnaLayer tests ───────────────────────────────────────────────────────

    #[test]
    fn pna_new_invalid_in_features() {
        let cfg = PnaConfig {
            in_features: 0,
            ..Default::default()
        };
        assert!(PnaLayer::new(cfg).is_err());
    }

    #[test]
    fn pna_new_invalid_hidden_features() {
        let cfg = PnaConfig {
            hidden_features: 0,
            ..Default::default()
        };
        assert!(PnaLayer::new(cfg).is_err());
    }

    #[test]
    fn pna_new_invalid_out_features() {
        let cfg = PnaConfig {
            out_features: 0,
            ..Default::default()
        };
        assert!(PnaLayer::new(cfg).is_err());
    }

    #[test]
    fn pna_new_empty_aggregators() {
        let cfg = PnaConfig {
            aggregators: vec![],
            ..Default::default()
        };
        assert!(PnaLayer::new(cfg).is_err());
    }

    #[test]
    fn pna_new_empty_scalers() {
        let cfg = PnaConfig {
            scalers: vec![],
            ..Default::default()
        };
        assert!(PnaLayer::new(cfg).is_err());
    }

    #[test]
    fn pna_new_invalid_delta() {
        let cfg = PnaConfig {
            delta: 0.0,
            ..Default::default()
        };
        assert!(PnaLayer::new(cfg).is_err());
    }

    #[test]
    fn pna_mlp_in_dim_correct() {
        let cfg = PnaConfig {
            in_features: 4,
            hidden_features: 8,
            out_features: 4,
            aggregators: vec![PnaAggregator::Sum, PnaAggregator::Mean, PnaAggregator::Max],
            scalers: vec![PnaScaler::Identity, PnaScaler::Amplification],
            delta: 1.0,
        };
        let layer = PnaLayer::new(cfg).expect("valid config");
        // 4 * 3 * 2 = 24
        assert_eq!(layer.mlp_in_dim(), 24);
    }

    #[test]
    fn pna_mlp_in_dim_all_aggs_all_scalers() {
        let cfg = PnaConfig {
            in_features: 3,
            hidden_features: 8,
            out_features: 2,
            aggregators: vec![
                PnaAggregator::Sum,
                PnaAggregator::Mean,
                PnaAggregator::Max,
                PnaAggregator::Min,
                PnaAggregator::Std,
            ],
            scalers: vec![
                PnaScaler::Identity,
                PnaScaler::Amplification,
                PnaScaler::Attenuation,
                PnaScaler::LinearAmplification,
                PnaScaler::LinearAttenuation,
            ],
            delta: 2.0,
        };
        let layer = PnaLayer::new(cfg).expect("valid config");
        // 3 * 5 * 5 = 75
        assert_eq!(layer.mlp_in_dim(), 75);
    }

    #[test]
    fn pna_forward_output_shape() {
        let graph = triangle_graph();
        let cfg = PnaConfig {
            in_features: 2,
            hidden_features: 4,
            out_features: 3,
            aggregators: vec![PnaAggregator::Sum, PnaAggregator::Mean],
            scalers: vec![PnaScaler::Identity],
            delta: 2.0,
        };
        let layer = PnaLayer::new(cfg.clone()).expect("valid config");
        let mlp_in = layer.mlp_in_dim(); // 2*2*1 = 4
        let (w1, b1, w2, b2) = minimal_weights(mlp_in, cfg.hidden_features, cfg.out_features);
        let x = vec![0.5_f32; 3 * cfg.in_features];
        let out = layer
            .forward(&graph, &x, &w1, &b1, &w2, &b2)
            .expect("forward should succeed");
        assert_eq!(out.len(), 3 * cfg.out_features);
    }

    #[test]
    fn pna_forward_output_finite() {
        let graph = triangle_graph();
        let cfg = PnaConfig::default();
        let layer = PnaLayer::new(cfg.clone()).expect("valid config");
        let mlp_in = layer.mlp_in_dim();
        let (w1, b1, w2, b2) = minimal_weights(mlp_in, cfg.hidden_features, cfg.out_features);
        let x: Vec<f32> = (0..3 * cfg.in_features).map(|i| i as f32 * 0.1).collect();
        let out = layer
            .forward(&graph, &x, &w1, &b1, &w2, &b2)
            .expect("forward should succeed");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "all outputs should be finite"
        );
    }

    #[test]
    fn pna_forward_isolated_node_works() {
        // Node 2 has no outgoing edges → isolated
        let graph = CsrGraph::from_edges(3, &[(0, 1), (1, 0)]).expect("valid graph");
        let cfg = PnaConfig {
            in_features: 2,
            hidden_features: 4,
            out_features: 2,
            aggregators: vec![PnaAggregator::Sum],
            scalers: vec![PnaScaler::Identity],
            delta: 1.0,
        };
        let layer = PnaLayer::new(cfg.clone()).expect("valid config");
        let mlp_in = layer.mlp_in_dim();
        let (w1, b1, w2, b2) = minimal_weights(mlp_in, cfg.hidden_features, cfg.out_features);
        let x = vec![1.0_f32; 3 * cfg.in_features];
        let out = layer
            .forward(&graph, &x, &w1, &b1, &w2, &b2)
            .expect("forward should succeed for isolated node");
        assert_eq!(out.len(), 3 * cfg.out_features);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "isolated node output should be finite"
        );
    }

    #[test]
    fn pna_forward_all_same_features_shape_correct() {
        let graph = triangle_graph();
        let cfg = PnaConfig {
            in_features: 4,
            hidden_features: 8,
            out_features: 4,
            aggregators: vec![PnaAggregator::Sum, PnaAggregator::Mean, PnaAggregator::Max],
            scalers: vec![PnaScaler::Identity, PnaScaler::Amplification],
            delta: 2.0,
        };
        let layer = PnaLayer::new(cfg.clone()).expect("valid config");
        let mlp_in = layer.mlp_in_dim();
        let (w1, b1, w2, b2) = minimal_weights(mlp_in, cfg.hidden_features, cfg.out_features);
        let x = vec![1.0_f32; 3 * cfg.in_features];
        let out = layer
            .forward(&graph, &x, &w1, &b1, &w2, &b2)
            .expect("forward should succeed");
        assert_eq!(out.len(), 3 * cfg.out_features);
    }

    #[test]
    fn pna_forward_w1_wrong_shape_error() {
        let graph = triangle_graph();
        let cfg = PnaConfig {
            in_features: 2,
            hidden_features: 4,
            out_features: 2,
            aggregators: vec![PnaAggregator::Sum],
            scalers: vec![PnaScaler::Identity],
            delta: 1.0,
        };
        let layer = PnaLayer::new(cfg.clone()).expect("valid config");
        let x = vec![0.0_f32; 3 * cfg.in_features];
        // Wrong w1 size
        let w1 = vec![0.1_f32; 3]; // wrong
        let b1 = vec![0.0_f32; cfg.hidden_features];
        let w2 = vec![0.1_f32; cfg.out_features * cfg.hidden_features];
        let b2 = vec![0.0_f32; cfg.out_features];
        assert!(layer.forward(&graph, &x, &w1, &b1, &w2, &b2).is_err());
    }

    #[test]
    fn pna_forward_w2_wrong_shape_error() {
        let graph = triangle_graph();
        let cfg = PnaConfig {
            in_features: 2,
            hidden_features: 4,
            out_features: 2,
            aggregators: vec![PnaAggregator::Sum],
            scalers: vec![PnaScaler::Identity],
            delta: 1.0,
        };
        let layer = PnaLayer::new(cfg.clone()).expect("valid config");
        let mlp_in = layer.mlp_in_dim();
        let x = vec![0.0_f32; 3 * cfg.in_features];
        let w1 = vec![0.1_f32; cfg.hidden_features * mlp_in];
        let b1 = vec![0.0_f32; cfg.hidden_features];
        let w2 = vec![0.1_f32; 5]; // wrong
        let b2 = vec![0.0_f32; cfg.out_features];
        assert!(layer.forward(&graph, &x, &w1, &b1, &w2, &b2).is_err());
    }

    #[test]
    fn pna_amplification_attenuation_uniform_delta() {
        // When d == 2 and delta == 2.0:
        // Amplification: log(3)/log(3+eps) ≈ 1.0
        // Attenuation:   log(3+eps)/log(3+eps) ≈ 1.0 (since d==delta)
        let agg = vec![2.0_f32];
        let amp = scale(PnaScaler::Amplification, &agg, 2, 2.0);
        let att = scale(PnaScaler::Attenuation, &agg, 2, 2.0);
        assert!(
            (amp[0] - 2.0).abs() < 0.05,
            "amplification factor at d=delta=2 should be ~1, got factor={}",
            amp[0] / 2.0
        );
        assert!(
            (att[0] - 2.0).abs() < 0.05,
            "attenuation factor at d=delta=2 should be ~1, got factor={}",
            att[0] / 2.0
        );
    }

    #[test]
    fn pna_default_config_valid() {
        let cfg = PnaConfig::default();
        assert!(PnaLayer::new(cfg).is_ok());
    }
}
