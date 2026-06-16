//! Progressive Neural Networks: lateral connections + frozen columns.
//!
//! Implements the method from:
//! Rusu et al. "Progressive Neural Networks." arXiv 2016.
//!
//! Each task trains a new column of layers. Previous columns are frozen.
//! Lateral connections transfer knowledge from all previous columns to the
//! current column at each layer:
//! `h_k^l = relu(W_k^l · h_k^{l-1} + Σ_{j<k} U_{j→k}^l · h_j^{l-1})`

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

/// ReLU activation in-place.
#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Dense matrix-vector product: `out = W · x + bias`.
///
/// `W` is stored row-major with shape `[out_dim × in_dim]`.
fn matvec(w: &[f32], x: &[f32], bias: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; out_dim];
    for row in 0..out_dim {
        let mut acc = bias[row];
        let base = row * in_dim;
        for col in 0..in_dim {
            acc += w[base + col] * x[col];
        }
        out[row] = acc;
    }
    out
}

/// One column of a Progressive Neural Network.
///
/// Contains `n_layers` dense layers each with weight matrix `[d_hidden × d_hidden]`
/// stored row-major, plus biases.
#[derive(Debug, Clone)]
pub struct ProgNnColumn {
    /// Weight matrices per layer, each `[d_hidden × d_hidden]` (row-major).
    pub weights: Vec<Vec<f32>>,
    /// Bias vectors per layer.
    pub biases: Vec<Vec<f32>>,
    /// Number of hidden layers.
    pub n_layers: usize,
    /// Hidden dimension.
    pub d_hidden: usize,
}

impl ProgNnColumn {
    /// Construct a new column with random weights (N(0,1) / sqrt(d_hidden)).
    pub fn random_init(
        n_layers: usize,
        d_hidden: usize,
        rng: &mut LcgRng,
    ) -> ContinualResult<Self> {
        if n_layers == 0 {
            return Err(ContinualError::InvalidNumLayers);
        }
        let scale = 1.0 / (d_hidden as f32).sqrt();
        let mut weights = Vec::with_capacity(n_layers);
        let mut biases = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let mut w = vec![0.0_f32; d_hidden * d_hidden];
            rng.fill_normal(&mut w);
            for v in &mut w {
                *v *= scale;
            }
            weights.push(w);
            biases.push(vec![0.0_f32; d_hidden]);
        }
        Ok(Self {
            weights,
            biases,
            n_layers,
            d_hidden,
        })
    }
}

/// Lateral connection adapter: maps from a previous column's hidden state
/// to the current column's pre-activation at a given layer.
#[derive(Debug, Clone)]
pub struct LateralConnection {
    /// Adapter weight matrix `[d_hidden × d_hidden]` (row-major).
    pub weights: Vec<Vec<f32>>,
    /// Per-layer biases for lateral adapters.
    pub biases: Vec<Vec<f32>>,
    /// Number of layers (must match ProgNnColumn.n_layers).
    pub n_layers: usize,
    /// Hidden dimension.
    pub d_hidden: usize,
}

impl LateralConnection {
    /// Initialise a lateral connection adapter with small random weights.
    pub fn random_init(
        n_layers: usize,
        d_hidden: usize,
        rng: &mut LcgRng,
    ) -> ContinualResult<Self> {
        if n_layers == 0 {
            return Err(ContinualError::InvalidNumLayers);
        }
        let scale = 0.01_f32 / (d_hidden as f32).sqrt();
        let mut weights = Vec::with_capacity(n_layers);
        let mut biases = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let mut w = vec![0.0_f32; d_hidden * d_hidden];
            rng.fill_normal(&mut w);
            for v in &mut w {
                *v *= scale;
            }
            weights.push(w);
            biases.push(vec![0.0_f32; d_hidden]);
        }
        Ok(Self {
            weights,
            biases,
            n_layers,
            d_hidden,
        })
    }
}

/// Progressive Neural Network containing multiple task columns.
///
/// `laterals[k][j]` is the lateral adapter from column `j` to column `k+1`
/// at all layers (shape: outer vec = new columns, inner vec = source columns).
#[derive(Debug, Clone, Default)]
pub struct ProgNnNetwork {
    /// Task columns, one per task.
    pub columns: Vec<ProgNnColumn>,
    /// `laterals[k]` = adapters for column `k+1` from all previous columns `0..k`.
    pub laterals: Vec<Vec<LateralConnection>>,
}

impl ProgNnNetwork {
    /// Create an empty network.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of columns (= number of tasks added so far).
    #[must_use]
    pub fn n_columns(&self) -> usize {
        self.columns.len()
    }
}

/// Add a new task column to the network.
///
/// Creates a new `ProgNnColumn` and lateral connections from all existing
/// columns to the new one. Previous columns remain frozen (the caller must
/// not update their weights).
pub fn add_column(
    net: &mut ProgNnNetwork,
    d_hidden: usize,
    n_layers: usize,
    rng: &mut LcgRng,
) -> ContinualResult<()> {
    let new_col = ProgNnColumn::random_init(n_layers, d_hidden, rng)?;
    let n_prev = net.columns.len();
    let mut new_laterals = Vec::with_capacity(n_prev);
    for _ in 0..n_prev {
        let lat = LateralConnection::random_init(n_layers, d_hidden, rng)?;
        new_laterals.push(lat);
    }
    net.columns.push(new_col);
    net.laterals.push(new_laterals);
    Ok(())
}

/// Run forward pass through column `col_idx` with lateral inputs from
/// all previous columns.
///
/// `h_k^l = relu(W_k^l · h_k^{l-1} + Σ_{j<k} U_{j→k}^l · h_j^{l-1})`
///
/// `input` must have length `d_hidden` (first layer input).
/// Returns the final-layer hidden state of shape `[d_hidden]`.
pub fn prog_forward(
    net: &ProgNnNetwork,
    input: &[f32],
    col_idx: usize,
) -> ContinualResult<Vec<f32>> {
    if net.columns.is_empty() {
        return Err(ContinualError::NoTasksInStream);
    }
    if col_idx >= net.columns.len() {
        return Err(ContinualError::ColumnIndexOutOfRange {
            index: col_idx,
            n_columns: net.columns.len(),
        });
    }
    let col = &net.columns[col_idx];
    let d = col.d_hidden;
    let n_layers = col.n_layers;

    if input.len() != d {
        return Err(ContinualError::DimensionMismatch {
            expected: d,
            got: input.len(),
        });
    }

    // Run all previous columns forward to get their hidden states per layer.
    // prev_hiddens[j][l] = hidden state of column j at layer l (0=input).
    let mut prev_hiddens: Vec<Vec<Vec<f32>>> = Vec::with_capacity(col_idx);
    for j in 0..col_idx {
        let prev_col = &net.columns[j];
        let mut h = input.to_vec();
        let mut layer_states = vec![h.clone()]; // layer 0 = input
        for l in 0..prev_col.n_layers {
            let pre = matvec(&prev_col.weights[l], &h, &prev_col.biases[l], d, d);
            h = pre.iter().map(|&v| relu(v)).collect();
            layer_states.push(h.clone());
        }
        prev_hiddens.push(layer_states);
    }

    // Now forward through col_idx with lateral inputs.
    let mut h = input.to_vec();
    for l in 0..n_layers {
        // Main column pre-activation
        let mut pre = matvec(&col.weights[l], &h, &col.biases[l], d, d);

        // Add lateral contributions from each previous column at layer l
        // net.laterals[col_idx] has adapters for column col_idx from columns 0..col_idx
        if col_idx < net.laterals.len() {
            let adapters = &net.laterals[col_idx];
            for (j, adapter) in adapters.iter().enumerate() {
                if j < prev_hiddens.len() && l < adapter.n_layers {
                    // Use hidden state of column j at layer l (before ReLU)
                    let h_prev = &prev_hiddens[j][l];
                    let lat_out = matvec(&adapter.weights[l], h_prev, &adapter.biases[l], d, d);
                    for (p, &lat) in pre.iter_mut().zip(lat_out.iter()) {
                        *p += lat;
                    }
                }
            }
        }
        h = pre.iter().map(|&v| relu(v)).collect();
    }

    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_column_no_laterals_forward() {
        let mut rng = LcgRng::new(42);
        let mut net = ProgNnNetwork::new();
        add_column(&mut net, 8, 2, &mut rng).expect("adding a progressive column should succeed");
        let input = vec![0.5_f32; 8];
        let out = prog_forward(&net, &input, 0)
            .expect("progressive forward pass should succeed for valid column");
        assert_eq!(out.len(), 8, "Output shape should match d_hidden");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "All outputs must be finite"
        );
        // ReLU: all non-negative
        assert!(
            out.iter().all(|&v| v >= 0.0),
            "ReLU output must be non-negative"
        );
    }

    #[test]
    fn multi_column_forward_shape_correct() {
        let mut rng = LcgRng::new(7);
        let mut net = ProgNnNetwork::new();
        add_column(&mut net, 4, 2, &mut rng).expect("adding a progressive column should succeed");
        add_column(&mut net, 4, 2, &mut rng).expect("adding a progressive column should succeed");
        assert_eq!(net.n_columns(), 2);
        let input = vec![1.0_f32; 4];
        let out0 = prog_forward(&net, &input, 0)
            .expect("progressive forward pass should succeed for valid column");
        let out1 = prog_forward(&net, &input, 1)
            .expect("progressive forward pass should succeed for valid column");
        assert_eq!(out0.len(), 4);
        assert_eq!(out1.len(), 4);
        assert!(out0.iter().all(|v| v.is_finite()));
        assert!(out1.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn frozen_columns_unchanged_after_add() {
        let mut rng = LcgRng::new(11);
        let mut net = ProgNnNetwork::new();
        add_column(&mut net, 4, 2, &mut rng).expect("adding a progressive column should succeed");
        // Snapshot column 0 weights before adding column 1
        let w0_before = net.columns[0].weights.clone();
        add_column(&mut net, 4, 2, &mut rng).expect("adding a progressive column should succeed");
        // Column 0 should be unchanged
        assert_eq!(
            net.columns[0].weights, w0_before,
            "Frozen column should not change"
        );
    }

    #[test]
    fn add_column_invalid_layers_returns_err() {
        let mut rng = LcgRng::new(42);
        let mut net = ProgNnNetwork::new();
        assert!(add_column(&mut net, 4, 0, &mut rng).is_err());
    }

    #[test]
    fn prog_forward_column_out_of_range() {
        let mut rng = LcgRng::new(42);
        let mut net = ProgNnNetwork::new();
        add_column(&mut net, 4, 2, &mut rng).expect("adding a progressive column should succeed");
        let input = vec![1.0_f32; 4];
        assert!(prog_forward(&net, &input, 5).is_err());
    }

    #[test]
    fn prog_forward_dimension_mismatch() {
        let mut rng = LcgRng::new(42);
        let mut net = ProgNnNetwork::new();
        add_column(&mut net, 4, 2, &mut rng).expect("adding a progressive column should succeed");
        let bad_input = vec![1.0_f32; 3]; // wrong dim
        assert!(prog_forward(&net, &bad_input, 0).is_err());
    }

    #[test]
    fn prog_forward_empty_network_returns_err() {
        let net = ProgNnNetwork::new();
        let input = vec![1.0_f32; 4];
        assert!(prog_forward(&net, &input, 0).is_err());
    }

    #[test]
    fn three_columns_all_forward_correctly() {
        let mut rng = LcgRng::new(99);
        let mut net = ProgNnNetwork::new();
        for _ in 0..3 {
            add_column(&mut net, 4, 2, &mut rng)
                .expect("adding a progressive column should succeed");
        }
        let input = vec![0.1_f32; 4];
        for col in 0..3 {
            let out = prog_forward(&net, &input, col)
                .expect("progressive forward pass should succeed for valid column");
            assert_eq!(out.len(), 4);
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }
}
