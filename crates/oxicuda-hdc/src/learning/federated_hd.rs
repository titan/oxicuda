//! Federated HD learning — gradient-free, DP-compatible single-round aggregation.
//!
//! Hyperdimensional computing (HDC) is naturally suited to federated learning:
//! a model is nothing more than one *integer prototype accumulator* per class
//! (the element-wise sum of every training hypervector seen for that class).
//! Because the accumulator is a plain additive statistic, the global model can
//! be reconstructed from local models by **summing the per-class accumulators
//! element-wise** — no gradients, no learning rate, no multi-round optimisation.
//!
//! This module implements that scheme:
//!
//! 1. Each [`ClientModel`] independently bundles its local training data into
//!    per-class `i32` accumulators (`add_example`).
//! 2. The [`FederatedServer`] performs a single gradient-free aggregation round
//!    (`aggregate`) that sums all client accumulators, then thresholds each
//!    aggregated accumulator to a `±1` binary prototype (`build_prototypes`).
//! 3. Inference (`classify`) is the usual argmax cosine similarity against the
//!    binary class prototypes.
//!
//! Because integer addition is associative and commutative, the aggregated
//! server accumulators are **bit-for-bit identical** to those of a centralized
//! model trained on the union of all client data. The federated split therefore
//! incurs *zero* accuracy loss relative to centralized HD training (modulo the
//! optional privacy perturbations described below).
//!
//! # Differential-privacy compatibility
//!
//! Two optional, privacy-oriented operations are provided per client and are
//! intended to be applied *before* the client accumulator leaves the device:
//!
//! * [`ClientModel::clip`] bounds the L-infinity norm of each accumulator entry,
//!   limiting the contribution (sensitivity) of any single client/coordinate.
//! * [`ClientModel::add_dp_noise`] adds bounded, uniformly distributed integer
//!   noise to every accumulator entry.
//!
//! **Honesty note:** the noise mechanism here is a *simple bounded-uniform
//! discrete perturbation* drawn from the crate's deterministic [`LcgRng`](crate::handle::LcgRng). It is
//! a DP-*style* mechanism — `scale` trades privacy against utility — but it is
//! **not** a calibrated Laplace/Gaussian mechanism and provides **no formal
//! `(ε, δ)`-differential-privacy guarantee**. For a provable guarantee one must
//! substitute a properly calibrated noise distribution and a cryptographically
//! secure RNG; the structure here (clip-then-noise-then-sum) is the correct
//! pipeline into which such a mechanism would drop.
//!
//! # References
//!
//! * M. Imani et al., "A Framework for Collaborative Learning in Secure
//!   High-Dimensional Space" (FedHD / distributed HD), 2019.
//! * Hsieh et al., "FL-HDC: Hyperdimensional Computing Design for the
//!   Application of Federated Learning", 2021.
//!
//! This is intentionally distinct from the centralized
//! [`crate::classifier::hd_classifier::HdClassifier`]: it models the
//! client/server split and the privacy-preserving aggregation pipeline.

use crate::distance::cosine::cosine_binary;
use crate::error::{HdcError, HdcResult};
use crate::handle::LcgRng;
use crate::vector::binary::threshold_binary;

/// A single federated client's local HD model.
///
/// The client maintains one `i32` accumulator per class — the running
/// element-wise sum of every training hypervector added for that class — plus a
/// per-class example count. The accumulators are the *only* quantity that is
/// shared with the [`FederatedServer`]; they may optionally be clipped and/or
/// perturbed for privacy before being aggregated.
#[derive(Debug, Clone)]
pub struct ClientModel {
    /// Number of distinct classes this client models.
    n_classes: usize,
    /// Hypervector dimensionality.
    dim: usize,
    /// Per-class `i32` accumulators (`n_classes` rows, each of length `dim`).
    accumulators: Vec<Vec<i32>>,
    /// Per-class count of training examples bundled so far.
    counts: Vec<usize>,
}

impl ClientModel {
    /// Create a new, empty client model.
    ///
    /// # Errors
    ///
    /// * [`HdcError::EmptyInput`] if `n_classes == 0`.
    /// * [`HdcError::ZeroDimension`] if `dim == 0`.
    pub fn new(n_classes: usize, dim: usize) -> HdcResult<Self> {
        if n_classes == 0 {
            return Err(HdcError::EmptyInput);
        }
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        Ok(Self {
            n_classes,
            dim,
            accumulators: vec![vec![0i32; dim]; n_classes],
            counts: vec![0usize; n_classes],
        })
    }

    /// Bundle one local training example (a `±1` hypervector) into the
    /// accumulator of the given class.
    ///
    /// The hypervector is added element-wise as `i32` and the class count is
    /// incremented. This is the local equivalent of HD prototype bundling.
    ///
    /// # Errors
    ///
    /// * [`HdcError::ClassNotFound`] if `class >= n_classes`.
    /// * [`HdcError::DimensionMismatch`] if `hv.len() != dim`.
    pub fn add_example(&mut self, class: usize, hv: &[i8]) -> HdcResult<()> {
        if class >= self.n_classes {
            return Err(HdcError::ClassNotFound(class));
        }
        if hv.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: hv.len(),
            });
        }
        for (a, &v) in self.accumulators[class].iter_mut().zip(hv.iter()) {
            *a += v as i32;
        }
        self.counts[class] += 1;
        Ok(())
    }

    /// Clip every accumulator entry to the closed interval `[-bound, bound]`.
    ///
    /// This bounds the L-infinity norm of each coordinate of each per-class
    /// accumulator, limiting the *sensitivity* of the client's contribution —
    /// a prerequisite for any bounded-sensitivity differential-privacy
    /// mechanism. It is intended to be applied immediately before the
    /// accumulators are shared with the server (optionally followed by
    /// [`ClientModel::add_dp_noise`]).
    ///
    /// Clipping is lossy: it discards magnitude information beyond `bound`, so
    /// `bound` trades utility against bounded sensitivity.
    ///
    /// # Errors
    ///
    /// * [`HdcError::InvalidProbability`] if `bound <= 0` (the bound is reused as
    ///   the carrier error variant; the crate intentionally defines no dedicated
    ///   "invalid bound" variant).
    pub fn clip(&mut self, bound: i32) -> HdcResult<()> {
        if bound <= 0 {
            return Err(HdcError::InvalidProbability(bound as f64));
        }
        for row in self.accumulators.iter_mut() {
            for a in row.iter_mut() {
                *a = (*a).clamp(-bound, bound);
            }
        }
        Ok(())
    }

    /// Add bounded uniform integer noise in `[-scale, scale]` to every
    /// accumulator entry.
    ///
    /// For each coordinate of each per-class accumulator a fresh integer is
    /// drawn from the deterministic [`LcgRng`] as
    /// `rng.next_usize(2 * scale + 1) as i32 - scale`, yielding a value in the
    /// closed interval `[-scale, scale]`, and added to the accumulator.
    ///
    /// **Privacy honesty:** this is a *bounded-uniform discrete perturbation*,
    /// not a calibrated Laplace/Gaussian mechanism. `scale` controls the
    /// privacy/utility trade-off (larger `scale` ⇒ more noise ⇒ more privacy,
    /// less accuracy), but this mechanism provides **no formal `(ε, δ)`-DP
    /// guarantee** and the RNG is not cryptographically secure. It exists to
    /// exercise — and to provide the correct insertion point for — a
    /// DP-compatible aggregation pipeline. A `scale` of `0` is a valid no-op
    /// (the noise interval collapses to `{0}`).
    ///
    /// # Errors
    ///
    /// * [`HdcError::InvalidProbability`] if `scale < 0`.
    pub fn add_dp_noise(&mut self, scale: i32, rng: &mut LcgRng) -> HdcResult<()> {
        if scale < 0 {
            return Err(HdcError::InvalidProbability(scale as f64));
        }
        if scale == 0 {
            return Ok(());
        }
        // `scale > 0`, so `2 * scale + 1` is a positive range for `next_usize`.
        let span = 2usize * scale as usize + 1;
        for row in self.accumulators.iter_mut() {
            for a in row.iter_mut() {
                let noise = rng.next_usize(span) as i32 - scale;
                *a += noise;
            }
        }
        Ok(())
    }

    /// Number of classes modelled by this client.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Hypervector dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Borrow the `i32` accumulator for the given class.
    ///
    /// # Errors
    ///
    /// * [`HdcError::ClassNotFound`] if `class >= n_classes`.
    pub fn accumulator(&self, class: usize) -> HdcResult<&[i32]> {
        if class >= self.n_classes {
            return Err(HdcError::ClassNotFound(class));
        }
        Ok(&self.accumulators[class])
    }

    /// Number of training examples bundled for the given class.
    ///
    /// # Errors
    ///
    /// * [`HdcError::ClassNotFound`] if `class >= n_classes`.
    pub fn count(&self, class: usize) -> HdcResult<usize> {
        if class >= self.n_classes {
            return Err(HdcError::ClassNotFound(class));
        }
        Ok(self.counts[class])
    }
}

/// The federated aggregation server.
///
/// The server holds the aggregated per-class `i32` accumulators (the sum over
/// all participating clients) together with the `±1` binary prototypes built
/// from them. Aggregation is a single gradient-free round (additive over all
/// clients); inference is argmax cosine similarity.
#[derive(Debug, Clone)]
pub struct FederatedServer {
    /// Number of classes (must match every client).
    n_classes: usize,
    /// Hypervector dimensionality (must match every client).
    dim: usize,
    /// Aggregated per-class accumulators (sum of all client accumulators).
    accumulators: Vec<Vec<i32>>,
    /// `±1` binary prototypes built from the aggregated accumulators.
    prototypes: Vec<Vec<i8>>,
    /// Whether [`FederatedServer::build_prototypes`] has produced valid
    /// prototypes for the current aggregated state.
    prototypes_built: bool,
}

impl FederatedServer {
    /// Create a new server with zeroed accumulators and no prototypes yet.
    ///
    /// # Errors
    ///
    /// * [`HdcError::EmptyInput`] if `n_classes == 0`.
    /// * [`HdcError::ZeroDimension`] if `dim == 0`.
    pub fn new(n_classes: usize, dim: usize) -> HdcResult<Self> {
        if n_classes == 0 {
            return Err(HdcError::EmptyInput);
        }
        if dim == 0 {
            return Err(HdcError::ZeroDimension);
        }
        Ok(Self {
            n_classes,
            dim,
            accumulators: vec![vec![0i32; dim]; n_classes],
            prototypes: Vec::new(),
            prototypes_built: false,
        })
    }

    /// Gradient-free FedAvg-style aggregation: **sum** every client's per-class
    /// accumulators element-wise into the server's accumulators.
    ///
    /// This is the single federated round. It is additive over the round: the
    /// server accumulators are incremented by the contribution of each client,
    /// so the result is the element-wise sum of all client accumulators (which,
    /// by associativity of integer addition, equals the accumulators of a
    /// centralized model trained on the union of the clients' data).
    ///
    /// Aggregating invalidates any previously built prototypes; call
    /// [`FederatedServer::build_prototypes`] again afterwards.
    ///
    /// # Errors
    ///
    /// * [`HdcError::EmptyInput`] if `clients` is empty.
    /// * [`HdcError::DimensionMismatch`] if any client's `n_classes` or `dim`
    ///   differs from the server's.
    pub fn aggregate(&mut self, clients: &[ClientModel]) -> HdcResult<()> {
        if clients.is_empty() {
            return Err(HdcError::EmptyInput);
        }
        for client in clients {
            if client.n_classes != self.n_classes {
                return Err(HdcError::DimensionMismatch {
                    expected: self.n_classes,
                    got: client.n_classes,
                });
            }
            if client.dim != self.dim {
                return Err(HdcError::DimensionMismatch {
                    expected: self.dim,
                    got: client.dim,
                });
            }
        }
        for client in clients {
            for c in 0..self.n_classes {
                for (a, &v) in self.accumulators[c]
                    .iter_mut()
                    .zip(client.accumulators[c].iter())
                {
                    *a += v;
                }
            }
        }
        self.prototypes_built = false;
        Ok(())
    }

    /// Threshold each aggregated accumulator to a `±1` binary prototype.
    ///
    /// Uses [`threshold_binary`], which maps positive sums to `+1`, negative
    /// sums to `-1`, and breaks ties (sum `== 0`) randomly via `rng`.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`threshold_binary`] (e.g.
    /// [`HdcError::EmptyInput`] for a zero-length accumulator).
    pub fn build_prototypes(&mut self, rng: &mut LcgRng) -> HdcResult<()> {
        let mut prototypes = Vec::with_capacity(self.n_classes);
        for c in 0..self.n_classes {
            prototypes.push(threshold_binary(&self.accumulators[c], rng)?);
        }
        self.prototypes = prototypes;
        self.prototypes_built = true;
        Ok(())
    }

    /// Classify a `±1` query hypervector by argmax cosine similarity against the
    /// per-class binary prototypes.
    ///
    /// # Errors
    ///
    /// * [`HdcError::EmptyInput`] if prototypes have not been built (call
    ///   [`FederatedServer::build_prototypes`] first).
    /// * [`HdcError::DimensionMismatch`] if `query.len() != dim`.
    /// * Any error propagated from [`cosine_binary`].
    pub fn classify(&self, query: &[i8]) -> HdcResult<usize> {
        if !self.prototypes_built {
            return Err(HdcError::EmptyInput);
        }
        if query.len() != self.dim {
            return Err(HdcError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let mut best_class = 0usize;
        let mut best_sim = f32::NEG_INFINITY;
        for c in 0..self.n_classes {
            let sim = cosine_binary(&self.prototypes[c], query)?;
            if sim > best_sim {
                best_sim = sim;
                best_class = c;
            }
        }
        Ok(best_class)
    }

    /// Number of classes modelled by the server.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Hypervector dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Whether valid prototypes have been built for the current aggregated
    /// state.
    pub fn is_built(&self) -> bool {
        self.prototypes_built
    }

    /// Borrow the aggregated `i32` accumulator for the given class.
    ///
    /// # Errors
    ///
    /// * [`HdcError::ClassNotFound`] if `class >= n_classes`.
    pub fn accumulator(&self, class: usize) -> HdcResult<&[i32]> {
        if class >= self.n_classes {
            return Err(HdcError::ClassNotFound(class));
        }
        Ok(&self.accumulators[class])
    }

    /// Borrow the `±1` binary prototype for the given class.
    ///
    /// # Errors
    ///
    /// * [`HdcError::EmptyInput`] if prototypes have not been built yet.
    /// * [`HdcError::ClassNotFound`] if `class >= n_classes`.
    pub fn prototype(&self, class: usize) -> HdcResult<&[i8]> {
        if !self.prototypes_built {
            return Err(HdcError::EmptyInput);
        }
        if class >= self.n_classes {
            return Err(HdcError::ClassNotFound(class));
        }
        Ok(&self.prototypes[class])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;
    use crate::vector::binary::random_binary;

    /// Build a deterministic noisy variant of `base` by flipping `n_flips`
    /// coordinates chosen via `rng`. Result stays in `±1`.
    fn noisy_variant(base: &[i8], n_flips: usize, rng: &mut LcgRng) -> Vec<i8> {
        let mut v = base.to_vec();
        for _ in 0..n_flips {
            let idx = rng.next_usize(base.len());
            v[idx] = -v[idx];
        }
        v
    }

    #[test]
    fn client_construction_validation() {
        assert!(matches!(
            ClientModel::new(0, 512),
            Err(HdcError::EmptyInput)
        ));
        assert!(matches!(
            ClientModel::new(3, 0),
            Err(HdcError::ZeroDimension)
        ));
        let client = ClientModel::new(4, 1024).expect("valid client");
        assert_eq!(client.n_classes(), 4);
        assert_eq!(client.dim(), 1024);
        assert_eq!(client.count(0).expect("count"), 0);
        assert_eq!(client.accumulator(3).expect("acc").len(), 1024);
        assert!(matches!(
            client.accumulator(4),
            Err(HdcError::ClassNotFound(4))
        ));
    }

    #[test]
    fn server_construction_validation() {
        assert!(matches!(
            FederatedServer::new(0, 512),
            Err(HdcError::EmptyInput)
        ));
        assert!(matches!(
            FederatedServer::new(2, 0),
            Err(HdcError::ZeroDimension)
        ));
        let server = FederatedServer::new(3, 768).expect("valid server");
        assert_eq!(server.n_classes(), 3);
        assert_eq!(server.dim(), 768);
        assert!(!server.is_built());
        assert_eq!(server.accumulator(0).expect("acc").len(), 768);
    }

    #[test]
    fn add_example_out_of_range_errors() {
        let dim = 512;
        let mut client = ClientModel::new(2, dim).expect("client");
        let hv = vec![1i8; dim];
        assert!(matches!(
            client.add_example(2, &hv),
            Err(HdcError::ClassNotFound(2))
        ));
        let short = vec![1i8; dim - 1];
        assert!(matches!(
            client.add_example(0, &short),
            Err(HdcError::DimensionMismatch {
                expected,
                got,
            }) if expected == dim && got == dim - 1
        ));
    }

    #[test]
    fn federated_equals_centralized_accumulators_and_classification() {
        // CORE equivalence test: a centralized model trained on ALL data must
        // produce accumulators bit-for-bit identical to summing three clients
        // that partition the SAME data — because aggregation is integer sum.
        let mut rng = LcgRng::new(12345);
        let dim = 2048;
        let n_classes = 3;

        // Per-class base prototypes.
        let bases: Vec<Vec<i8>> = (0..n_classes)
            .map(|_| random_binary(dim, &mut rng).expect("base"))
            .collect();

        // Generate a labelled training set (noisy variants of the bases).
        let mut training: Vec<(usize, Vec<i8>)> = Vec::new();
        for round in 0..15 {
            for (class, base) in bases.iter().enumerate() {
                let flips = (round + class) % 40; // modest noise
                training.push((class, noisy_variant(base, flips, &mut rng)));
            }
        }

        // Centralized model: everything in ONE client.
        let mut central = ClientModel::new(n_classes, dim).expect("central");
        for (class, hv) in &training {
            central.add_example(*class, hv).expect("central add");
        }

        // Federated: round-robin the SAME data across 3 clients.
        let mut clients: Vec<ClientModel> = (0..3)
            .map(|_| ClientModel::new(n_classes, dim).expect("client"))
            .collect();
        for (i, (class, hv)) in training.iter().enumerate() {
            clients[i % 3].add_example(*class, hv).expect("client add");
        }

        let mut server = FederatedServer::new(n_classes, dim).expect("server");
        server.aggregate(&clients).expect("aggregate");

        // EXACT equality of accumulators (sum is associative & commutative).
        for c in 0..n_classes {
            let central_acc = central.accumulator(c).expect("central acc");
            let server_acc = server.accumulator(c).expect("server acc");
            assert_eq!(
                central_acc, server_acc,
                "federated aggregated accumulator must EXACTLY equal centralized for class {c}"
            );
        }

        // Identical prototypes (use independent but identically-seeded RNGs so
        // tie-breaks coincide) and identical classification on a test set.
        let mut central_clf =
            crate::classifier::hd_classifier::HdClassifier::new(n_classes, dim).expect("clf");
        for (class, hv) in &training {
            central_clf.add_example(*class, hv).expect("clf add");
        }
        let mut rng_a = LcgRng::new(999);
        let mut rng_b = LcgRng::new(999);
        central_clf.build_prototypes(&mut rng_a).expect("clf build");
        server.build_prototypes(&mut rng_b).expect("server build");

        for c in 0..n_classes {
            assert_eq!(
                central_clf.prototype(c).expect("clf proto"),
                server.prototype(c).expect("server proto"),
                "prototypes must match for class {c}"
            );
        }

        // Same predictions on a fresh test set.
        let mut test_rng = LcgRng::new(777);
        for _ in 0..30 {
            for (class, base) in bases.iter().enumerate() {
                let q = noisy_variant(base, class % 20, &mut test_rng);
                let central_pred = central_clf.classify(&q).expect("central classify");
                let server_pred = server.classify(&q).expect("server classify");
                assert_eq!(central_pred, server_pred);
                assert_eq!(server_pred, class, "should recover the true class");
            }
        }
    }

    #[test]
    fn clip_bounds_values() {
        let dim = 512;
        let mut client = ClientModel::new(2, dim).expect("client");
        // Drive accumulators well beyond the clip bound in both directions.
        let pos = vec![1i8; dim];
        let neg = vec![-1i8; dim];
        for _ in 0..50 {
            client.add_example(0, &pos).expect("add pos");
            client.add_example(1, &neg).expect("add neg");
        }
        let bound = 10i32;
        client.clip(bound).expect("clip");
        for c in 0..2 {
            for &v in client.accumulator(c).expect("acc") {
                assert!(
                    (-bound..=bound).contains(&v),
                    "value {v} not clipped into [-{bound}, {bound}]"
                );
            }
        }
        // Class 0 saturates at +bound, class 1 at -bound.
        assert!(
            client
                .accumulator(0)
                .expect("acc0")
                .iter()
                .all(|&v| v == bound)
        );
        assert!(
            client
                .accumulator(1)
                .expect("acc1")
                .iter()
                .all(|&v| v == -bound)
        );

        // Invalid bound rejected.
        assert!(matches!(
            client.clip(0),
            Err(HdcError::InvalidProbability(_))
        ));
        assert!(matches!(
            client.clip(-5),
            Err(HdcError::InvalidProbability(_))
        ));
    }

    #[test]
    fn dp_noise_bounded_and_deterministic() {
        let dim = 512;
        let scale = 7i32;

        // Reference accumulators (no noise).
        let mut base = ClientModel::new(2, dim).expect("base");
        let hv = vec![1i8; dim];
        for _ in 0..3 {
            base.add_example(0, &hv).expect("add");
            base.add_example(1, &hv).expect("add");
        }

        // Two noised copies with the SAME seed must be identical (determinism).
        let mut a = base.clone();
        let mut b = base.clone();
        let mut rng_a = LcgRng::new(42);
        let mut rng_b = LcgRng::new(42);
        a.add_dp_noise(scale, &mut rng_a).expect("noise a");
        b.add_dp_noise(scale, &mut rng_b).expect("noise b");
        for c in 0..2 {
            assert_eq!(
                a.accumulator(c).expect("a acc"),
                b.accumulator(c).expect("b acc"),
                "DP noise must be deterministic for a fixed seed (class {c})"
            );
            // Every noised entry must stay within reference ± scale.
            let ref_acc = base.accumulator(c).expect("ref acc");
            let noised = a.accumulator(c).expect("noised acc");
            for (&r, &n) in ref_acc.iter().zip(noised.iter()) {
                assert!(
                    (r - scale..=r + scale).contains(&n),
                    "noised value {n} outside [{}, {}]",
                    r - scale,
                    r + scale
                );
            }
        }

        // scale == 0 is a valid no-op.
        let mut z = base.clone();
        let mut rng_z = LcgRng::new(1);
        z.add_dp_noise(0, &mut rng_z).expect("zero noise ok");
        assert_eq!(
            z.accumulator(0).expect("z acc"),
            base.accumulator(0).expect("base acc")
        );

        // Negative scale rejected.
        let mut neg = base.clone();
        let mut rng_n = LcgRng::new(1);
        assert!(matches!(
            neg.add_dp_noise(-1, &mut rng_n),
            Err(HdcError::InvalidProbability(_))
        ));
    }

    #[test]
    fn aggregate_validation_errors() {
        let dim = 512;
        let mut server = FederatedServer::new(2, dim).expect("server");

        // Empty client list.
        assert!(matches!(server.aggregate(&[]), Err(HdcError::EmptyInput)));

        // Class-count mismatch.
        let wrong_classes = ClientModel::new(3, dim).expect("wrong classes");
        assert!(matches!(
            server.aggregate(&[wrong_classes]),
            Err(HdcError::DimensionMismatch {
                expected: 2,
                got: 3,
            })
        ));

        // Dimension mismatch.
        let wrong_dim = ClientModel::new(2, dim + 1).expect("wrong dim");
        assert!(matches!(
            server.aggregate(&[wrong_dim]),
            Err(HdcError::DimensionMismatch {
                expected,
                got,
            }) if expected == dim && got == dim + 1
        ));
    }

    #[test]
    fn classify_before_build_errors() {
        let dim = 512;
        let server = FederatedServer::new(2, dim).expect("server");
        let q = vec![1i8; dim];
        assert!(matches!(server.classify(&q), Err(HdcError::EmptyInput)));
        assert!(matches!(server.prototype(0), Err(HdcError::EmptyInput)));
    }

    #[test]
    fn two_class_federated_end_to_end() {
        let mut rng = LcgRng::new(2024);
        let dim = 1024;

        let base0 = random_binary(dim, &mut rng).expect("base0");
        let base1 = random_binary(dim, &mut rng).expect("base1");

        // Two clients each hold data for both classes.
        let mut c0 = ClientModel::new(2, dim).expect("c0");
        let mut c1 = ClientModel::new(2, dim).expect("c1");
        for _ in 0..8 {
            c0.add_example(0, &noisy_variant(&base0, 30, &mut rng))
                .expect("c0 add0");
            c0.add_example(1, &noisy_variant(&base1, 30, &mut rng))
                .expect("c0 add1");
            c1.add_example(0, &noisy_variant(&base0, 30, &mut rng))
                .expect("c1 add0");
            c1.add_example(1, &noisy_variant(&base1, 30, &mut rng))
                .expect("c1 add1");
        }

        let mut server = FederatedServer::new(2, dim).expect("server");
        server.aggregate(&[c0, c1]).expect("aggregate");
        server.build_prototypes(&mut rng).expect("build");
        assert!(server.is_built());

        // Clean prototypes classify to their own class.
        assert_eq!(server.classify(&base0).expect("c0"), 0);
        assert_eq!(server.classify(&base1).expect("c1"), 1);

        // Noisy queries too.
        let mut test_rng = LcgRng::new(55);
        for _ in 0..20 {
            let q0 = noisy_variant(&base0, 40, &mut test_rng);
            let q1 = noisy_variant(&base1, 40, &mut test_rng);
            assert_eq!(server.classify(&q0).expect("q0"), 0);
            assert_eq!(server.classify(&q1).expect("q1"), 1);
        }
    }

    #[test]
    fn dim_mismatch_on_classify() {
        let dim = 512;
        let mut rng = LcgRng::new(3);
        let mut client = ClientModel::new(2, dim).expect("client");
        let hv = vec![1i8; dim];
        client.add_example(0, &hv).expect("add0");
        client.add_example(1, &vec![-1i8; dim]).expect("add1");
        let mut server = FederatedServer::new(2, dim).expect("server");
        server.aggregate(&[client]).expect("aggregate");
        server.build_prototypes(&mut rng).expect("build");
        let bad = vec![1i8; dim + 3];
        assert!(matches!(
            server.classify(&bad),
            Err(HdcError::DimensionMismatch {
                expected,
                got,
            }) if expected == dim && got == dim + 3
        ));
    }
}
