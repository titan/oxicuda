//! Batched PointNet forward pass (`B × N × 3` point clouds).
//!
//! [`crate::arch::pointnet::PointNet`] classifies a single `N × 3` cloud. In
//! practice point clouds arrive in mini-batches; this module runs the same
//! network over a batch of `B` independent clouds and stacks the logits into a
//! row-major `[B × n_classes]` matrix. Each sample is processed independently
//! (PointNet is permutation-invariant per cloud and has no cross-sample
//! coupling), so the batched result is exactly the per-sample result.

use crate::arch::pointnet::PointNet;
use crate::error::{Geom3dError, Geom3dResult};

/// Run [`PointNet`] over a batch of point clouds.
///
/// * `net` — the (shared-weight) PointNet to evaluate.
/// * `points` — row-major `[B × N × 3]`, i.e. `B · N · 3` floats; sample `b`
///   occupies `points[b·N·3 .. (b+1)·N·3]`.
/// * `batch` — number of clouds `B`.
///
/// Returns the stacked logits `[B × n_classes]` (row-major).
///
/// # Errors
///
/// * [`Geom3dError::BatchSizeMismatch`] if `batch == 0`.
/// * [`Geom3dError::DimensionMismatch`] if `points.len() != B · N · 3`.
pub fn pointnet_forward_batched(
    net: &PointNet,
    points: &[f32],
    batch: usize,
    n_per_cloud: usize,
) -> Geom3dResult<Vec<f32>> {
    if batch == 0 {
        return Err(Geom3dError::BatchSizeMismatch { lhs: 0, rhs: 1 });
    }
    let stride = n_per_cloud * 3;
    if points.len() != batch * stride {
        return Err(Geom3dError::DimensionMismatch {
            expected: batch * stride,
            got: points.len(),
        });
    }

    // Probe one sample to learn the class count, then size the output.
    let first = net.forward(&points[0..stride])?;
    let n_classes = first.len();
    let mut out = vec![0.0_f32; batch * n_classes];
    out[0..n_classes].copy_from_slice(&first);

    for b in 1..batch {
        let logits = net.forward(&points[b * stride..(b + 1) * stride])?;
        if logits.len() != n_classes {
            return Err(Geom3dError::BatchSizeMismatch {
                lhs: n_classes,
                rhs: logits.len(),
            });
        }
        out[b * n_classes..(b + 1) * n_classes].copy_from_slice(&logits);
    }

    Ok(out)
}

/// Batched classification: argmax of the logits for every sample.
///
/// Returns a `Vec<usize>` of length `batch` with the predicted class index per
/// cloud.
///
/// # Errors
///
/// Same as [`pointnet_forward_batched`].
pub fn pointnet_classify_batched(
    net: &PointNet,
    points: &[f32],
    batch: usize,
    n_per_cloud: usize,
) -> Geom3dResult<Vec<usize>> {
    let logits = pointnet_forward_batched(net, points, batch, n_per_cloud)?;
    let n_classes = logits.len() / batch;
    let mut preds = vec![0usize; batch];
    for b in 0..batch {
        let row = &logits[b * n_classes..(b + 1) * n_classes];
        let best = row
            .iter()
            .enumerate()
            .max_by(|a, c| a.1.partial_cmp(c.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or(Geom3dError::EmptyPointCloud)?;
        preds[b] = best;
    }
    Ok(preds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::pointnet::PointNetConfig;
    use crate::handle::LcgRng;

    fn make_cloud(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut pts = vec![0.0_f32; n * 3];
        for v in &mut pts {
            *v = rng.next_u32() as f32 / 4_294_967_296.0 * 2.0 - 1.0;
        }
        pts
    }

    fn build_net(n: usize, nc: usize) -> PointNet {
        let mut rng = LcgRng::new(7);
        PointNet::new(
            PointNetConfig {
                n_points: n,
                n_classes: nc,
            },
            &mut rng,
        )
    }

    #[test]
    fn batched_output_shape() {
        let (n, nc, b) = (16, 5, 4);
        let net = build_net(n, nc);
        let mut batch = Vec::new();
        for s in 0..b {
            batch.extend_from_slice(&make_cloud(n, 100 + s as u64));
        }
        let out = pointnet_forward_batched(&net, &batch, b, n).expect("forward should succeed");
        assert_eq!(out.len(), b * nc);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn batched_equals_per_sample() {
        let (n, nc, b) = (12, 4, 3);
        let net = build_net(n, nc);
        let mut clouds = Vec::new();
        let mut flat = Vec::new();
        for s in 0..b {
            let c = make_cloud(n, 7 + s as u64);
            flat.extend_from_slice(&c);
            clouds.push(c);
        }
        let batched = pointnet_forward_batched(&net, &flat, b, n).expect("forward should succeed");
        for (s, cloud) in clouds.iter().enumerate() {
            let single = net.forward(cloud).expect("forward should succeed");
            for (j, &v) in single.iter().enumerate() {
                assert!(
                    (batched[s * nc + j] - v).abs() < 1e-6,
                    "batched row {s} must equal single forward"
                );
            }
        }
    }

    #[test]
    fn classify_batched_in_range() {
        let (n, nc, b) = (16, 6, 5);
        let net = build_net(n, nc);
        let mut flat = Vec::new();
        for s in 0..b {
            flat.extend_from_slice(&make_cloud(n, 200 + s as u64));
        }
        let preds = pointnet_classify_batched(&net, &flat, b, n).expect("classify should succeed");
        assert_eq!(preds.len(), b);
        assert!(preds.iter().all(|&c| c < nc));
    }

    #[test]
    fn zero_batch_errors() {
        let net = build_net(8, 3);
        assert!(pointnet_forward_batched(&net, &[], 0, 8).is_err());
    }

    #[test]
    fn wrong_length_errors() {
        let net = build_net(8, 3);
        let bad = vec![0.0_f32; 8 * 3 * 2 - 1]; // not a multiple of stride
        assert!(pointnet_forward_batched(&net, &bad, 2, 8).is_err());
    }

    #[test]
    fn deterministic_across_calls() {
        let (n, nc, b) = (10, 4, 2);
        let net = build_net(n, nc);
        let mut flat = Vec::new();
        for s in 0..b {
            flat.extend_from_slice(&make_cloud(n, 5 + s as u64));
        }
        let _ = nc;
        let a = pointnet_forward_batched(&net, &flat, b, n).expect("forward should succeed");
        let c = pointnet_forward_batched(&net, &flat, b, n).expect("forward should succeed");
        assert_eq!(a, c);
    }
}
