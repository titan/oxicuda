//! Latency surrogate predictors for hardware-aware NAS.
//!
//! Two complementary models:
//! - [`LatencyLut`] — measurement lookup table indexed by `(OpKind, in_ch, out_ch, h, w)`.
//!   Returns calibrated latency in seconds. Use for known device-specific
//!   benchmarks where the search space is small enough to memoise.
//! - [`LatencyMlp`] — small two-layer ReLU MLP that consumes
//!   [`ArchFeatures`] and predicts a scalar.
//!   Use for generalising across unseen `(op, shape)` combinations from a
//!   profiled training set.
//!
//! Both models can be calibrated against measured data with
//! [`LatencyLut::insert`] / [`LatencyMlp::fit`], and queried with `predict`.

use std::collections::HashMap;

use crate::error::{NasError, NasResult};
use crate::handle::LcgRng;
use crate::ops::OpKind;
use crate::predictor::predictor_io::{ArchFeatures, LayerSpec};

// ─── LatencyLut serialisation constants ──────────────────────────────────────

/// Magic bytes that open every serialised [`LatencyLut`] buffer.
const LUT_MAGIC: [u8; 4] = *b"LLUT";
/// Binary encoding version; increment whenever the layout changes incompatibly.
const LUT_VERSION: u8 = 1;

// ─── LatencyLut ──────────────────────────────────────────────────────────────

/// Lookup-table latency model.
#[derive(Debug, Default, Clone)]
pub struct LatencyLut {
    table: HashMap<LatencyKey, f32>,
    /// Default latency returned for unknown layers.
    pub default_latency: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LatencyKey {
    op: OpKind,
    cin: usize,
    cout: usize,
    h: usize,
    w: usize,
}

impl LatencyLut {
    /// Create an empty LUT with default latency `0.0`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a measured latency for a `(op, cin, cout, h, w)` configuration.
    pub fn insert(&mut self, layer: &LayerSpec, latency_seconds: f32) {
        self.table.insert(
            LatencyKey {
                op: layer.op,
                cin: layer.in_channels,
                cout: layer.out_channels,
                h: layer.h,
                w: layer.w,
            },
            latency_seconds,
        );
    }

    /// Look up a single layer's latency. Falls back to `default_latency` if absent.
    #[must_use]
    pub fn lookup(&self, layer: &LayerSpec) -> f32 {
        let k = LatencyKey {
            op: layer.op,
            cin: layer.in_channels,
            cout: layer.out_channels,
            h: layer.h,
            w: layer.w,
        };
        self.table.get(&k).copied().unwrap_or(self.default_latency)
    }

    /// Sum the latencies of an architecture's layers.
    ///
    /// # Errors
    /// [`NasError::EmptySearchSpace`] when `layers.is_empty()`.
    pub fn predict(&self, layers: &[LayerSpec]) -> NasResult<f32> {
        if layers.is_empty() {
            return Err(NasError::EmptySearchSpace);
        }
        let mut total = 0.0_f32;
        for layer in layers {
            total += self.lookup(layer);
        }
        Ok(total)
    }

    /// Number of measured `(op, shape)` entries in the table.
    #[must_use]
    pub fn n_entries(&self) -> usize {
        self.table.len()
    }

    /// Serialise the LUT to a deterministic, dependency-free little-endian byte
    /// buffer suitable for file persistence or network transfer.
    ///
    /// # Layout (all multi-byte integers are little-endian)
    ///
    /// | Bytes          | Content                                           |
    /// |----------------|---------------------------------------------------|
    /// | `[0..4)`       | magic `b"LLUT"` (4 bytes)                        |
    /// | `[4]`          | format version byte (currently `1`)               |
    /// | `[5..9)`       | `default_latency` as `f32` bit-pattern            |
    /// | `[9..17)`      | entry count `n` as `u64`                          |
    /// | `[17..17+37n)` | `n` entries × 37 bytes each (see below)           |
    ///
    /// Per-entry layout (37 bytes, emitted in stable sort order):
    ///
    /// | Offset | Size | Content                                             |
    /// |--------|------|-----------------------------------------------------|
    /// | 0      | 1    | `OpKind` index 0–7 (per [`OpKind::all`] order)    |
    /// | 1      | 8    | `cin` as `u64`                                     |
    /// | 9      | 8    | `cout` as `u64`                                    |
    /// | 17     | 8    | `h` as `u64`                                       |
    /// | 25     | 8    | `w` as `u64`                                       |
    /// | 33     | 4    | latency as `f32` bit-pattern                       |
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let n = self.table.len();
        // Header: 4 magic + 1 version + 4 default_latency + 8 entry_count = 17 bytes.
        // Each entry: 1 disc + 8 cin + 8 cout + 8 h + 8 w + 4 lat = 37 bytes.
        let mut buf = Vec::with_capacity(17 + n * 37);
        buf.extend_from_slice(&LUT_MAGIC);
        buf.push(LUT_VERSION);
        buf.extend_from_slice(&self.default_latency.to_bits().to_le_bytes());
        buf.extend_from_slice(&(n as u64).to_le_bytes());
        // Collect into an owned Vec so we can sort for deterministic output.
        let mut entries: Vec<(LatencyKey, f32)> =
            self.table.iter().map(|(&k, &v)| (k, v)).collect();
        entries.sort_unstable_by_key(|item| {
            (
                op_kind_to_disc(item.0.op),
                item.0.cin,
                item.0.cout,
                item.0.h,
                item.0.w,
            )
        });
        for (key, latency) in &entries {
            buf.push(op_kind_to_disc(key.op));
            buf.extend_from_slice(&(key.cin as u64).to_le_bytes());
            buf.extend_from_slice(&(key.cout as u64).to_le_bytes());
            buf.extend_from_slice(&(key.h as u64).to_le_bytes());
            buf.extend_from_slice(&(key.w as u64).to_le_bytes());
            buf.extend_from_slice(&latency.to_bits().to_le_bytes());
        }
        buf
    }

    /// Deserialise a [`LatencyLut`] from a buffer previously produced by
    /// [`LatencyLut::to_bytes`].
    ///
    /// # Errors
    ///
    /// Returns [`NasError::Internal`] when the buffer is truncated, begins
    /// with wrong magic bytes, carries an unrecognised format version, or
    /// contains an out-of-range `OpKind` discriminant.  The method **never
    /// panics** on malformed input.
    pub fn from_bytes(bytes: &[u8]) -> NasResult<Self> {
        let mut cur = 0usize;

        // 1. Magic
        let magic = lut_read4(bytes, &mut cur)
            .ok_or_else(|| NasError::Internal("lut: truncated header (magic)".into()))?;
        if magic != LUT_MAGIC {
            return Err(NasError::Internal(
                "lut: invalid magic — buffer is not a serialised LatencyLut".into(),
            ));
        }

        // 2. Format version
        let version = lut_read1(bytes, &mut cur)
            .ok_or_else(|| NasError::Internal("lut: truncated header (version)".into()))?;
        if version != LUT_VERSION {
            return Err(NasError::Internal(format!(
                "lut: unsupported format version {version} (expected {LUT_VERSION})"
            )));
        }

        // 3. default_latency
        let def_raw = lut_read4(bytes, &mut cur)
            .ok_or_else(|| NasError::Internal("lut: truncated default_latency field".into()))?;
        let default_latency = f32::from_bits(u32::from_le_bytes(def_raw));

        // 4. Entry count
        let n_raw = lut_read8(bytes, &mut cur)
            .ok_or_else(|| NasError::Internal("lut: truncated entry count field".into()))?;
        let n = u64::from_le_bytes(n_raw) as usize;

        // 5. Entries
        let mut table = HashMap::with_capacity(n);
        for idx in 0..n {
            let disc = lut_read1(bytes, &mut cur).ok_or_else(|| {
                NasError::Internal(format!("lut: truncated entry {idx} (op discriminant)"))
            })?;
            let op = disc_to_op_kind(disc).ok_or_else(|| {
                NasError::Internal(format!(
                    "lut: invalid OpKind discriminant {disc} in entry {idx}"
                ))
            })?;

            let cin_raw = lut_read8(bytes, &mut cur)
                .ok_or_else(|| NasError::Internal(format!("lut: truncated entry {idx} (cin)")))?;
            let cin = u64::from_le_bytes(cin_raw) as usize;

            let cout_raw = lut_read8(bytes, &mut cur)
                .ok_or_else(|| NasError::Internal(format!("lut: truncated entry {idx} (cout)")))?;
            let cout = u64::from_le_bytes(cout_raw) as usize;

            let h_raw = lut_read8(bytes, &mut cur)
                .ok_or_else(|| NasError::Internal(format!("lut: truncated entry {idx} (h)")))?;
            let h = u64::from_le_bytes(h_raw) as usize;

            let w_raw = lut_read8(bytes, &mut cur)
                .ok_or_else(|| NasError::Internal(format!("lut: truncated entry {idx} (w)")))?;
            let w = u64::from_le_bytes(w_raw) as usize;

            let lat_raw = lut_read4(bytes, &mut cur).ok_or_else(|| {
                NasError::Internal(format!("lut: truncated entry {idx} (latency)"))
            })?;
            let latency = f32::from_bits(u32::from_le_bytes(lat_raw));

            table.insert(
                LatencyKey {
                    op,
                    cin,
                    cout,
                    h,
                    w,
                },
                latency,
            );
        }

        Ok(Self {
            table,
            default_latency,
        })
    }
}

// ─── LatencyLut serialisation helpers ────────────────────────────────────────

/// Map an [`OpKind`] to its stable one-byte serialisation discriminant.
///
/// The mapping is the binary format contract and **must never be reordered**.
/// It mirrors the order of [`OpKind::all`].
fn op_kind_to_disc(op: OpKind) -> u8 {
    match op {
        OpKind::Zero => 0,
        OpKind::Identity => 1,
        OpKind::SepConv3x3 => 2,
        OpKind::SepConv5x5 => 3,
        OpKind::DilConv3x3 => 4,
        OpKind::DilConv5x5 => 5,
        OpKind::MaxPool3x3 => 6,
        OpKind::AvgPool3x3 => 7,
    }
}

/// Reverse of [`op_kind_to_disc`]; returns `None` for unrecognised discriminants.
fn disc_to_op_kind(disc: u8) -> Option<OpKind> {
    match disc {
        0 => Some(OpKind::Zero),
        1 => Some(OpKind::Identity),
        2 => Some(OpKind::SepConv3x3),
        3 => Some(OpKind::SepConv5x5),
        4 => Some(OpKind::DilConv3x3),
        5 => Some(OpKind::DilConv5x5),
        6 => Some(OpKind::MaxPool3x3),
        7 => Some(OpKind::AvgPool3x3),
        _ => None,
    }
}

/// Read one byte from `bytes` at `*cur`, advancing the cursor.
/// Returns `None` on underflow.
fn lut_read1(bytes: &[u8], cur: &mut usize) -> Option<u8> {
    let v = bytes.get(*cur).copied()?;
    *cur += 1;
    Some(v)
}

/// Read four bytes from `bytes` at `*cur`, advancing the cursor.
/// Returns `None` on underflow.
fn lut_read4(bytes: &[u8], cur: &mut usize) -> Option<[u8; 4]> {
    let start = *cur;
    let end = start.checked_add(4)?;
    if end > bytes.len() {
        return None;
    }
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&bytes[start..end]);
    *cur = end;
    Some(arr)
}

/// Read eight bytes from `bytes` at `*cur`, advancing the cursor.
/// Returns `None` on underflow.
fn lut_read8(bytes: &[u8], cur: &mut usize) -> Option<[u8; 8]> {
    let start = *cur;
    let end = start.checked_add(8)?;
    if end > bytes.len() {
        return None;
    }
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[start..end]);
    *cur = end;
    Some(arr)
}

// ─── LatencyMlp ──────────────────────────────────────────────────────────────

/// Small MLP latency surrogate.
#[derive(Debug, Clone)]
pub struct LatencyMlp {
    /// Hidden layer weights `[hidden_dim × in_dim]` (row-major).
    pub w1: Vec<f32>,
    /// Hidden layer bias `[hidden_dim]`.
    pub b1: Vec<f32>,
    /// Output weights `[hidden_dim]`.
    pub w2: Vec<f32>,
    /// Output bias scalar.
    pub b2: f32,
    /// Input dimension (must match `ArchFeatures::dim()` of fitted samples).
    pub in_dim: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// True once [`LatencyMlp::fit`] has run successfully.
    pub fitted: bool,
}

impl LatencyMlp {
    /// Create an unfitted MLP with Kaiming-initialised hidden layer.
    #[must_use]
    pub fn new(in_dim: usize, hidden_dim: usize, rng: &mut LcgRng) -> Self {
        let scale = (2.0 / in_dim as f32).sqrt();
        let mut w1 = vec![0.0_f32; hidden_dim * in_dim];
        rng.fill_normal(&mut w1);
        for v in w1.iter_mut() {
            *v *= scale;
        }
        let b1 = vec![0.0_f32; hidden_dim];
        let mut w2 = vec![0.0_f32; hidden_dim];
        rng.fill_normal(&mut w2);
        for v in w2.iter_mut() {
            *v *= (2.0_f32 / hidden_dim as f32).sqrt();
        }
        Self {
            w1,
            b1,
            w2,
            b2: 0.0,
            in_dim,
            hidden_dim,
            fitted: false,
        }
    }

    /// One forward pass: returns scalar prediction.
    fn forward(&self, x: &[f32]) -> NasResult<f32> {
        if x.len() != self.in_dim {
            return Err(NasError::DimensionMismatch {
                expected: self.in_dim,
                got: x.len(),
            });
        }
        let mut h = vec![0.0_f32; self.hidden_dim];
        for ((hj, b), row) in h
            .iter_mut()
            .zip(self.b1.iter())
            .zip(self.w1.chunks(self.in_dim))
        {
            let mut acc = *b;
            for (w, &xi) in row.iter().zip(x.iter()) {
                acc += w * xi;
            }
            *hj = acc.max(0.0); // ReLU
        }
        let mut y = self.b2;
        for (wi, &hi) in self.w2.iter().zip(h.iter()) {
            y += wi * hi;
        }
        Ok(y)
    }

    /// Predict the latency of an architecture.
    ///
    /// # Errors
    /// - [`NasError::LatencyModelNotFitted`] if `fit` has not been called.
    /// - [`NasError::DimensionMismatch`] if features don't match `in_dim`.
    pub fn predict(&self, layers: &[LayerSpec]) -> NasResult<f32> {
        if !self.fitted {
            return Err(NasError::LatencyModelNotFitted);
        }
        let f = ArchFeatures::from_layers(layers)?;
        self.forward(&f.data)
    }

    /// Fit the MLP via simple per-sample gradient descent on MSE.
    ///
    /// `samples` is a list of `(features, latency)` pairs; all features must
    /// have the same length equal to `self.in_dim`.
    ///
    /// # Errors
    /// - [`NasError::EmptySearchSpace`] if `samples.is_empty()`.
    /// - [`NasError::DimensionMismatch`] if any feature length disagrees with `in_dim`.
    pub fn fit(
        &mut self,
        samples: &[(Vec<f32>, f32)],
        epochs: usize,
        learning_rate: f32,
    ) -> NasResult<f32> {
        if samples.is_empty() {
            return Err(NasError::EmptySearchSpace);
        }
        for (x, _) in samples {
            if x.len() != self.in_dim {
                return Err(NasError::DimensionMismatch {
                    expected: self.in_dim,
                    got: x.len(),
                });
            }
        }
        let mut last_loss = f32::INFINITY;
        for _ in 0..epochs {
            let mut total_loss = 0.0_f64;
            for (x, target) in samples {
                let target = *target;
                let mut h_pre = vec![0.0_f32; self.hidden_dim];
                let mut h = vec![0.0_f32; self.hidden_dim];
                for ((((hp, hh), b), row), _) in h_pre
                    .iter_mut()
                    .zip(h.iter_mut())
                    .zip(self.b1.iter())
                    .zip(self.w1.chunks(self.in_dim))
                    .zip(0..self.hidden_dim)
                {
                    let mut acc = *b;
                    for (w, &xi) in row.iter().zip(x.iter()) {
                        acc += w * xi;
                    }
                    *hp = acc;
                    *hh = acc.max(0.0);
                }
                let mut y = self.b2;
                for (wi, &hi) in self.w2.iter().zip(h.iter()) {
                    y += wi * hi;
                }
                let err = y - target;
                total_loss += (err * err) as f64;
                // Gradients
                let dy = 2.0 * err;
                // Output bias and weights
                self.b2 -= learning_rate * dy;
                for (wi, &hi) in self.w2.iter_mut().zip(h.iter()) {
                    *wi -= learning_rate * dy * hi;
                }
                // Hidden gradients
                for (((hp, b), row), w2) in h_pre
                    .iter()
                    .zip(self.b1.iter_mut())
                    .zip(self.w1.chunks_mut(self.in_dim))
                    .zip(self.w2.iter())
                {
                    if *hp <= 0.0 {
                        continue;
                    }
                    let dh = dy * w2;
                    *b -= learning_rate * dh;
                    for (w, &xi) in row.iter_mut().zip(x.iter()) {
                        *w -= learning_rate * dh * xi;
                    }
                }
            }
            last_loss = (total_loss / samples.len() as f64) as f32;
            if !last_loss.is_finite() {
                return Err(NasError::Internal(
                    "non-finite loss during latency MLP fit".into(),
                ));
            }
        }
        self.fitted = true;
        Ok(last_loss)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── existing LUT tests ────────────────────────────────────────────────────

    #[test]
    fn lut_returns_default_for_unknown() {
        let mut lut = LatencyLut::new();
        lut.default_latency = 1e-3;
        let layer = LayerSpec::new(OpKind::SepConv3x3, 4, 4, 8, 8);
        assert!((lut.lookup(&layer) - 1e-3).abs() < 1e-9);
    }

    #[test]
    fn lut_returns_inserted_value() {
        let mut lut = LatencyLut::new();
        let layer = LayerSpec::new(OpKind::SepConv3x3, 4, 4, 8, 8);
        lut.insert(&layer, 0.005);
        assert!((lut.lookup(&layer) - 0.005).abs() < 1e-7);
    }

    #[test]
    fn lut_predict_sums() {
        let mut lut = LatencyLut::new();
        let l1 = LayerSpec::new(OpKind::SepConv3x3, 4, 4, 8, 8);
        let l2 = LayerSpec::new(OpKind::AvgPool3x3, 4, 4, 8, 8);
        lut.insert(&l1, 0.001);
        lut.insert(&l2, 0.0001);
        let total = lut.predict(&[l1, l2]).expect("predict should succeed");
        assert!((total - 0.0011).abs() < 1e-6);
    }

    #[test]
    fn lut_predict_rejects_empty() {
        let lut = LatencyLut::new();
        let r = lut.predict(&[]);
        assert!(r.is_err());
    }

    // ── LatencyLut serialisation tests ───────────────────────────────────────

    /// Round-trip serialise+deserialise a LUT with one entry per `OpKind`
    /// variant, a non-default `default_latency`, and verify every field is
    /// bit-exact after the round-trip.
    #[test]
    fn lut_serde_roundtrip_all_ops() {
        let mut lut = LatencyLut::new();
        lut.default_latency = 9.99e-4_f32;

        // (op, cin, cout, h, w, latency)
        let specs: &[(OpKind, usize, usize, usize, usize, f32)] = &[
            (OpKind::Zero, 3, 64, 16, 16, 1.23e-3),
            (OpKind::Identity, 64, 64, 8, 8, 4.56e-4),
            (OpKind::SepConv3x3, 32, 64, 14, 14, 2.11e-3),
            (OpKind::SepConv5x5, 16, 32, 7, 7, 3.00e-3),
            (OpKind::DilConv3x3, 64, 128, 8, 8, 5.55e-3),
            (OpKind::DilConv5x5, 128, 256, 4, 4, 7.77e-3),
            (OpKind::MaxPool3x3, 64, 64, 8, 8, 1.10e-4),
            (OpKind::AvgPool3x3, 64, 64, 4, 4, 9.00e-5),
        ];
        for &(op, cin, cout, h, w, lat) in specs {
            lut.insert(&LayerSpec::new(op, cin, cout, h, w), lat);
        }

        let bytes = lut.to_bytes();
        let lut2 = LatencyLut::from_bytes(&bytes).expect("round-trip deserialization");

        // default_latency must be bit-exact.
        assert_eq!(
            lut2.default_latency.to_bits(),
            lut.default_latency.to_bits(),
            "default_latency changed after round-trip"
        );
        // Entry count must match.
        assert_eq!(lut2.n_entries(), lut.n_entries(), "entry count mismatch");
        // Each inserted entry must reproduce bit-exactly.
        for &(op, cin, cout, h, w, lat) in specs {
            let spec = LayerSpec::new(op, cin, cout, h, w);
            assert_eq!(
                lut2.lookup(&spec).to_bits(),
                lat.to_bits(),
                "lookup mismatch for {op:?} ({cin},{cout},{h},{w})"
            );
        }
    }

    /// `predict()` must return the same bit-exact value before and after a
    /// round-trip, including for a key that falls back to `default_latency`.
    #[test]
    fn lut_serde_predict_identical_before_after() {
        let mut lut = LatencyLut::new();
        lut.default_latency = 1.5e-3_f32;

        let l1 = LayerSpec::new(OpKind::SepConv3x3, 32, 64, 8, 8);
        let l2 = LayerSpec::new(OpKind::MaxPool3x3, 64, 64, 8, 8);
        // l_fallback is never inserted → falls back to default_latency.
        let l_fallback = LayerSpec::new(OpKind::DilConv5x5, 999, 999, 99, 99);

        lut.insert(&l1, 2.0e-3_f32);
        lut.insert(&l2, 3.0e-4_f32);

        let pred_before = lut
            .predict(&[l1, l2, l_fallback])
            .expect("predict before serialisation");

        let bytes = lut.to_bytes();
        let lut2 = LatencyLut::from_bytes(&bytes).expect("round-trip");

        let pred_after = lut2
            .predict(&[l1, l2, l_fallback])
            .expect("predict after deserialisation");

        assert_eq!(
            pred_before.to_bits(),
            pred_after.to_bits(),
            "predict() result changed after round-trip"
        );
    }

    /// An empty LUT with a custom `default_latency` must survive a round-trip.
    #[test]
    fn lut_serde_roundtrip_empty_lut() {
        let mut lut = LatencyLut::new();
        lut.default_latency = 7.77e-5_f32;
        let bytes = lut.to_bytes();
        let lut2 = LatencyLut::from_bytes(&bytes).expect("empty lut round-trip");
        assert_eq!(
            lut2.default_latency.to_bits(),
            lut.default_latency.to_bits(),
            "default_latency mismatch for empty LUT"
        );
        assert_eq!(lut2.n_entries(), 0, "empty LUT should have 0 entries");
    }

    /// Every strict prefix of a valid buffer must return `Err`, never panic.
    #[test]
    fn lut_serde_truncated_bytes_returns_err() {
        let mut lut = LatencyLut::new();
        lut.insert(&LayerSpec::new(OpKind::Identity, 4, 4, 8, 8), 1e-3_f32);
        let bytes = lut.to_bytes();

        for truncate_at in 0..bytes.len() {
            let result = LatencyLut::from_bytes(&bytes[..truncate_at]);
            assert!(
                result.is_err(),
                "expected Err for truncation at {truncate_at} bytes (full buf = {})",
                bytes.len()
            );
        }
    }

    /// Completely unrelated bytes must produce `Err`, not panic.
    #[test]
    fn lut_serde_garbage_bytes_returns_err() {
        let garbage = b"this is not a valid LatencyLut binary blob at all!!";
        assert!(
            LatencyLut::from_bytes(garbage).is_err(),
            "expected Err for garbage input"
        );
    }

    /// A buffer whose magic bytes have been corrupted must be rejected.
    #[test]
    fn lut_serde_wrong_magic_returns_err() {
        let lut = LatencyLut::new();
        let mut bytes = lut.to_bytes();
        bytes[0] ^= 0xFF; // flip bits of the first magic byte
        assert!(
            LatencyLut::from_bytes(&bytes).is_err(),
            "expected Err for corrupted magic"
        );
    }

    /// An out-of-range `OpKind` discriminant inside the entry region must
    /// produce `Err`.  Header ends at byte 17; byte 17 is the first entry's
    /// discriminant.
    #[test]
    fn lut_serde_invalid_op_discriminant_returns_err() {
        let mut lut = LatencyLut::new();
        lut.insert(&LayerSpec::new(OpKind::Identity, 4, 4, 8, 8), 1e-3_f32);
        let mut bytes = lut.to_bytes();
        // Header: 4 magic + 1 version + 4 default_latency + 8 entry_count = 17 bytes.
        // Byte 17 is the OpKind discriminant of the first entry.
        bytes[17] = 0xFF; // 0xFF is not a valid discriminant (0–7 are)
        assert!(
            LatencyLut::from_bytes(&bytes).is_err(),
            "expected Err for invalid OpKind discriminant"
        );
    }

    // ── existing MLP tests ────────────────────────────────────────────────────

    #[test]
    fn mlp_predict_before_fit_errors() {
        let mut rng = LcgRng::new(0);
        let mlp = LatencyMlp::new(ArchFeatures::PER_LAYER_DIM, 16, &mut rng);
        let layer = LayerSpec::new(OpKind::Identity, 4, 4, 8, 8);
        let r = mlp.predict(&[layer]);
        assert!(r.is_err());
    }

    #[test]
    fn mlp_fit_reduces_loss_on_constant_target() {
        let mut rng = LcgRng::new(0);
        let in_dim = ArchFeatures::PER_LAYER_DIM;
        let mut mlp = LatencyMlp::new(in_dim, 16, &mut rng);
        // Synthetic samples with target = 1.0
        let layer = LayerSpec::new(OpKind::Identity, 4, 4, 8, 8);
        let f = ArchFeatures::from_layers(&[layer]).expect("from_layers should succeed");
        let samples = vec![(f.data.clone(), 1.0_f32); 16];
        let loss0 = mlp.fit(&samples, 1, 1e-4).expect("fit should succeed");
        let loss1 = mlp.fit(&samples, 200, 1e-4).expect("fit should succeed");
        assert!(
            loss1 <= loss0 + 1e-3,
            "loss did not decrease: {loss0} -> {loss1}"
        );
        assert!(mlp.fitted);
        let pred = mlp.predict(&[layer]).expect("predict should succeed");
        assert!(pred.is_finite());
    }

    #[test]
    fn mlp_fit_rejects_empty() {
        let mut rng = LcgRng::new(0);
        let mut mlp = LatencyMlp::new(4, 4, &mut rng);
        assert!(mlp.fit(&[], 1, 1e-3).is_err());
    }

    #[test]
    fn mlp_fit_rejects_wrong_in_dim() {
        let mut rng = LcgRng::new(0);
        let mut mlp = LatencyMlp::new(4, 4, &mut rng);
        let r = mlp.fit(&[(vec![0.0_f32; 5], 1.0)], 1, 1e-3);
        assert!(r.is_err());
    }
}
