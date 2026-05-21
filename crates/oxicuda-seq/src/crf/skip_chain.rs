//! Skip-chain Conditional Random Fields.
//!
//! A skip-chain CRF (Sutton & McCallum 2004; Galley 2006) augments a linear-chain
//! CRF with long-range "skip" edges that connect non-adjacent positions believed to
//! be related (for example, repeated tokens that should receive the same label).
//! The skip edges turn the chain into a loopy graph, so exact forward-backward no
//! longer applies; inference is performed with **loopy belief propagation**
//! (sum-product for marginals, max-product for decoding).
//!
//! ## Parameterisation (score / log-potential space)
//!
//! To match the rest of the `crf` module, every potential is a **log-potential**
//! (additive score) and probabilities are proportional to `exp(score)`:
//!
//! * `unary[t * n_labels + l]` — log-potential of label `l` at position `t`.
//! * `transition[prev * n_labels + cur]` — chain edge `(t, t+1)` log-potential.
//! * `skip_potential[l_i * n_labels + l_j]` — skip edge `(i, j)` log-potential
//!   (with `i < j`; the first index addresses position `i`, the second `j`).
//!
//! Messages are kept in the log domain and normalised every iteration (max
//! subtracted) so they neither under- nor overflow on long, high-score chains.
//! With **no** skip edges the factor graph is a tree and loopy BP reduces to exact
//! forward-backward (sum-product) / Viterbi (max-product).

use crate::error::{SeqError, SeqResult};

/// Configuration for a skip-chain CRF.
#[derive(Debug, Clone)]
pub struct SkipChainConfig {
    /// Number of labels per position.
    pub n_labels: usize,
    /// Maximum number of loopy-BP iterations.
    pub max_bp_iters: usize,
    /// Convergence tolerance on the max absolute message change (log domain).
    pub bp_tol: f64,
}

/// A skip-chain CRF holding the (shared) chain and skip-edge log-potentials.
#[derive(Debug, Clone)]
pub struct SkipChainCrf {
    cfg: SkipChainConfig,
    /// Chain transition log-potentials, `n_labels × n_labels` row-major.
    transition: Vec<f64>,
    /// Skip-edge log-potentials, `n_labels × n_labels` row-major.
    skip_potential: Vec<f64>,
    /// Damping factor for message updates (fixed, in `(0, 1]`).
    damping: f64,
}

/// Internal description of an undirected pairwise edge in the factor graph.
#[derive(Debug, Clone, Copy)]
struct Edge {
    /// Lower-indexed endpoint position.
    u: usize,
    /// Higher-indexed endpoint position.
    v: usize,
    /// `true` if this is a chain edge (uses `transition`); `false` for a skip edge.
    is_chain: bool,
}

/// Log-sum-exp of a slice (`-inf` for an all-`-inf` slice).
fn log_sum_exp(xs: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &x in xs {
        if x > m {
            m = x;
        }
    }
    if m == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let mut s = 0.0;
    for &x in xs {
        s += (x - m).exp();
    }
    m + s.ln()
}

/// Maximum of a slice (`-inf` for empty).
fn max_of(xs: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &x in xs {
        if x > m {
            m = x;
        }
    }
    m
}

impl SkipChainCrf {
    /// Construct a skip-chain CRF, validating the potential shapes.
    pub fn new(
        cfg: SkipChainConfig,
        transition: Vec<f64>,
        skip_potential: Vec<f64>,
    ) -> SeqResult<Self> {
        if cfg.n_labels == 0 {
            return Err(SeqError::InvalidConfiguration(
                "n_labels must be >= 1".to_string(),
            ));
        }
        if cfg.max_bp_iters == 0 {
            return Err(SeqError::InvalidConfiguration(
                "max_bp_iters must be >= 1".to_string(),
            ));
        }
        if cfg.bp_tol <= 0.0 || cfg.bp_tol.is_nan() {
            return Err(SeqError::InvalidParameter {
                name: "bp_tol".to_string(),
                value: cfg.bp_tol,
            });
        }
        let l2 = cfg.n_labels * cfg.n_labels;
        if transition.len() != l2 {
            return Err(SeqError::ShapeMismatch {
                expected: l2,
                got: transition.len(),
            });
        }
        if skip_potential.len() != l2 {
            return Err(SeqError::ShapeMismatch {
                expected: l2,
                got: skip_potential.len(),
            });
        }
        Ok(Self {
            cfg,
            transition,
            skip_potential,
            damping: 0.5,
        })
    }

    /// Override the message-passing damping factor (must be in `(0, 1]`).
    pub fn with_damping(mut self, damping: f64) -> SeqResult<Self> {
        if damping <= 0.0 || damping > 1.0 || damping.is_nan() {
            return Err(SeqError::InvalidParameter {
                name: "damping".to_string(),
                value: damping,
            });
        }
        self.damping = damping;
        Ok(self)
    }

    /// Number of labels.
    pub fn n_labels(&self) -> usize {
        self.cfg.n_labels
    }

    /// Validate the inference inputs and assemble the edge list (chain + skip).
    fn prepare_edges(
        &self,
        unary: &[f64],
        seq_len: usize,
        skip_edges: &[(usize, usize)],
    ) -> SeqResult<Vec<Edge>> {
        let nl = self.cfg.n_labels;
        if seq_len == 0 {
            return Err(SeqError::EmptyInput);
        }
        if unary.len() != seq_len * nl {
            return Err(SeqError::ShapeMismatch {
                expected: seq_len * nl,
                got: unary.len(),
            });
        }
        let mut edges: Vec<Edge> = Vec::with_capacity(seq_len.saturating_sub(1) + skip_edges.len());
        for t in 0..seq_len.saturating_sub(1) {
            edges.push(Edge {
                u: t,
                v: t + 1,
                is_chain: true,
            });
        }
        for &(i, j) in skip_edges {
            if i >= seq_len || j >= seq_len {
                return Err(SeqError::IndexOutOfBounds {
                    index: i.max(j),
                    len: seq_len,
                });
            }
            if i >= j {
                return Err(SeqError::GraphInvariantViolated(format!(
                    "skip edge ({i}, {j}) must have i < j"
                )));
            }
            edges.push(Edge {
                u: i,
                v: j,
                is_chain: false,
            });
        }
        Ok(edges)
    }

    /// The oriented pairwise log-potential for `edge` between source label `l_src`
    /// (at `src`) and destination label `l_dst` (at `dst`).
    #[inline]
    fn edge_log_potential(&self, edge: &Edge, src: usize, l_src: usize, l_dst: usize) -> f64 {
        let nl = self.cfg.n_labels;
        let table = if edge.is_chain {
            &self.transition
        } else {
            &self.skip_potential
        };
        // Table is row-major over (position u, position v); orient by source.
        if src == edge.u {
            table[l_src * nl + l_dst]
        } else {
            table[l_dst * nl + l_src]
        }
    }

    /// Run loopy BP with the supplied combine operator (`log_sum_exp` for
    /// sum-product, `max_of` for max-product) and return the converged log-domain
    /// directed messages plus the per-edge directions.
    ///
    /// Directed message layout: edge `e` direction `u→v` lives at slot `2*e`,
    /// direction `v→u` at slot `2*e+1`; each slot holds `n_labels` log values.
    fn run_bp(
        &self,
        unary: &[f64],
        seq_len: usize,
        edges: &[Edge],
        combine: fn(&[f64]) -> f64,
    ) -> (Vec<f64>, usize, bool) {
        let nl = self.cfg.n_labels;
        let n_slots = edges.len() * 2;
        let mut log_msg = vec![0.0; n_slots * nl];
        let mut new_log_msg = log_msg.clone();

        // Precompute, for each position, the list of (edge_idx, incoming slot) so
        // that gathering neighbour messages is cheap.
        let mut incoming: Vec<Vec<(usize, usize)>> = vec![Vec::new(); seq_len];
        for (e_idx, e) in edges.iter().enumerate() {
            // Message arriving at u from v is stored in slot 2*e+1 (v→u).
            incoming[e.u].push((e_idx, e_idx * 2 + 1));
            // Message arriving at v from u is stored in slot 2*e (u→v).
            incoming[e.v].push((e_idx, e_idx * 2));
        }

        let mut iters = 0;
        let mut converged = false;
        let mut terms = vec![0.0; nl];

        for it in 0..self.cfg.max_bp_iters {
            iters = it + 1;
            for (e_idx, e) in edges.iter().enumerate() {
                // Two directions: src=u→dst=v (slot 2*e) and src=v→dst=u (slot 2*e+1).
                for &(src, dst, out_slot) in &[(e.u, e.v, e_idx * 2), (e.v, e.u, e_idx * 2 + 1)] {
                    let _ = dst;
                    let mut out = vec![f64::NEG_INFINITY; nl];
                    for l_dst in 0..nl {
                        for l_src in 0..nl {
                            // Unary at src + oriented pairwise + product of all
                            // incoming messages to src except along this edge.
                            let mut acc = unary[src * nl + l_src]
                                + self.edge_log_potential(e, src, l_src, l_dst);
                            for &(k_edge, slot) in &incoming[src] {
                                if k_edge == e_idx {
                                    continue;
                                }
                                acc += log_msg[slot * nl + l_src];
                            }
                            terms[l_src] = acc;
                        }
                        out[l_dst] = combine(&terms);
                    }
                    // Normalise (subtract max) for stability.
                    let m = max_of(&out);
                    if m != f64::NEG_INFINITY {
                        for v in out.iter_mut() {
                            *v -= m;
                        }
                    }
                    // Damped write.
                    for l in 0..nl {
                        new_log_msg[out_slot * nl + l] = (1.0 - self.damping)
                            * log_msg[out_slot * nl + l]
                            + self.damping * out[l];
                    }
                }
            }
            // Convergence on max absolute message change.
            let mut max_diff = 0.0_f64;
            for k in 0..log_msg.len() {
                let d = (new_log_msg[k] - log_msg[k]).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
            log_msg.copy_from_slice(&new_log_msg);
            if max_diff < self.cfg.bp_tol {
                converged = true;
                break;
            }
        }
        (log_msg, iters, converged)
    }

    /// Compute per-position belief = unary + sum of all incoming messages (log
    /// domain) for position `pos`.
    fn position_belief(
        &self,
        unary: &[f64],
        edges: &[Edge],
        log_msg: &[f64],
        incoming: &[Vec<(usize, usize)>],
        pos: usize,
    ) -> Vec<f64> {
        let nl = self.cfg.n_labels;
        let mut belief = vec![0.0; nl];
        for l in 0..nl {
            belief[l] = unary[pos * nl + l];
        }
        for &(_e_idx, slot) in &incoming[pos] {
            for l in 0..nl {
                belief[l] += log_msg[slot * nl + l];
            }
        }
        // `edges` is unused directly here but kept for signature symmetry.
        let _ = edges;
        belief
    }

    /// Build the per-position incoming-message index used by belief readout.
    fn build_incoming(seq_len: usize, edges: &[Edge]) -> Vec<Vec<(usize, usize)>> {
        let mut incoming: Vec<Vec<(usize, usize)>> = vec![Vec::new(); seq_len];
        for (e_idx, e) in edges.iter().enumerate() {
            incoming[e.u].push((e_idx, e_idx * 2 + 1));
            incoming[e.v].push((e_idx, e_idx * 2));
        }
        incoming
    }

    /// Loopy sum-product BP returning per-position marginals (`seq_len × n_labels`,
    /// each position normalised to sum to 1).
    pub fn infer_marginals(
        &self,
        unary: &[f64],
        seq_len: usize,
        skip_edges: &[(usize, usize)],
    ) -> SeqResult<Vec<f64>> {
        let nl = self.cfg.n_labels;
        let edges = self.prepare_edges(unary, seq_len, skip_edges)?;
        let (log_msg, _iters, _converged) = self.run_bp(unary, seq_len, &edges, log_sum_exp);
        let incoming = Self::build_incoming(seq_len, &edges);

        let mut marginals = vec![0.0; seq_len * nl];
        for pos in 0..seq_len {
            let belief = self.position_belief(unary, &edges, &log_msg, &incoming, pos);
            let logz = log_sum_exp(&belief);
            if logz == f64::NEG_INFINITY {
                let u = 1.0 / nl as f64;
                for l in 0..nl {
                    marginals[pos * nl + l] = u;
                }
            } else {
                for l in 0..nl {
                    marginals[pos * nl + l] = (belief[l] - logz).exp();
                }
            }
        }
        Ok(marginals)
    }

    /// Loopy sum-product BP returning per-position marginals together with whether
    /// the message passing converged and the iteration count.
    pub fn infer_marginals_with_status(
        &self,
        unary: &[f64],
        seq_len: usize,
        skip_edges: &[(usize, usize)],
    ) -> SeqResult<(Vec<f64>, usize, bool)> {
        let nl = self.cfg.n_labels;
        let edges = self.prepare_edges(unary, seq_len, skip_edges)?;
        let (log_msg, iters, converged) = self.run_bp(unary, seq_len, &edges, log_sum_exp);
        let incoming = Self::build_incoming(seq_len, &edges);

        let mut marginals = vec![0.0; seq_len * nl];
        for pos in 0..seq_len {
            let belief = self.position_belief(unary, &edges, &log_msg, &incoming, pos);
            let logz = log_sum_exp(&belief);
            if logz == f64::NEG_INFINITY {
                let u = 1.0 / nl as f64;
                for l in 0..nl {
                    marginals[pos * nl + l] = u;
                }
            } else {
                for l in 0..nl {
                    marginals[pos * nl + l] = (belief[l] - logz).exp();
                }
            }
        }
        Ok((marginals, iters, converged))
    }

    /// Loopy max-product BP decoding, returning the best label per position.
    pub fn decode(
        &self,
        unary: &[f64],
        seq_len: usize,
        skip_edges: &[(usize, usize)],
    ) -> SeqResult<Vec<usize>> {
        let nl = self.cfg.n_labels;
        let edges = self.prepare_edges(unary, seq_len, skip_edges)?;
        let (log_msg, _iters, _converged) = self.run_bp(unary, seq_len, &edges, max_of);
        let incoming = Self::build_incoming(seq_len, &edges);

        let mut labels = vec![0usize; seq_len];
        for pos in 0..seq_len {
            let belief = self.position_belief(unary, &edges, &log_msg, &incoming, pos);
            let mut best_l = 0usize;
            let mut best_v = f64::NEG_INFINITY;
            for l in 0..nl {
                if belief[l] > best_v {
                    best_v = belief[l];
                    best_l = l;
                }
            }
            labels[pos] = best_l;
        }
        Ok(labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crf::linear_chain_crf::LinearChainCrf;
    use crate::crf::viterbi_decode::viterbi_decode;
    use crate::hmm::forward_backward::logsumexp;

    fn cfg(n_labels: usize) -> SkipChainConfig {
        SkipChainConfig {
            n_labels,
            max_bp_iters: 200,
            bp_tol: 1e-10,
        }
    }

    /// Exact linear-chain forward-backward marginals in score space, given per
    /// position emission log-potentials `emit` (seq_len × n_labels) and a transition
    /// matrix.  Mirrors `crf_train::forward_scores`/`backward_scores`.
    fn exact_chain_marginals(emit: &[f64], transition: &[f64], n: usize, t_max: usize) -> Vec<f64> {
        let mut alpha = vec![f64::NEG_INFINITY; t_max * n];
        alpha[..n].copy_from_slice(&emit[..n]);
        let mut tmp = vec![0.0; n];
        for t in 1..t_max {
            for j in 0..n {
                for i in 0..n {
                    tmp[i] = alpha[(t - 1) * n + i] + transition[i * n + j];
                }
                alpha[t * n + j] = logsumexp(&tmp) + emit[t * n + j];
            }
        }
        let mut beta = vec![0.0; t_max * n];
        for t in (0..t_max - 1).rev() {
            for i in 0..n {
                for j in 0..n {
                    tmp[j] = transition[i * n + j] + emit[(t + 1) * n + j] + beta[(t + 1) * n + j];
                }
                beta[t * n + i] = logsumexp(&tmp);
            }
        }
        let log_z = logsumexp(&alpha[(t_max - 1) * n..]);
        let mut marg = vec![0.0; t_max * n];
        for t in 0..t_max {
            for j in 0..n {
                marg[t * n + j] = (alpha[t * n + j] + beta[t * n + j] - log_z).exp();
            }
        }
        marg
    }

    #[test]
    fn marginals_shape() {
        let crf = SkipChainCrf::new(cfg(3), vec![0.0; 9], vec![0.0; 9]).expect("new");
        let unary = vec![0.0; 4 * 3];
        let m = crf.infer_marginals(&unary, 4, &[]).expect("marg");
        assert_eq!(m.len(), 4 * 3);
    }

    #[test]
    fn marginals_each_position_sums_to_one() {
        let transition = vec![0.5, -0.2, 0.1, 0.3];
        let crf = SkipChainCrf::new(cfg(2), transition, vec![0.0, 0.0, 0.0, 0.0]).expect("new");
        let unary = vec![1.0, -0.5, 0.2, 0.7, -0.3, 0.4];
        let m = crf.infer_marginals(&unary, 3, &[(0, 2)]).expect("marg");
        for t in 0..3 {
            let s: f64 = m[t * 2..t * 2 + 2].iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "pos {t} sum {s}");
        }
    }

    #[test]
    fn no_skip_marginals_equal_forward_backward() {
        let n = 3;
        let transition = vec![0.4, -0.1, 0.2, -0.3, 0.5, 0.0, 0.1, -0.2, 0.6];
        let crf = SkipChainCrf::new(cfg(n), transition.clone(), vec![0.0; 9]).expect("new");
        let t_max = 5;
        // Arbitrary deterministic unary log-potentials.
        let mut unary = vec![0.0; t_max * n];
        for t in 0..t_max {
            for l in 0..n {
                unary[t * n + l] = ((t * 7 + l * 3) as f64 % 5.0) - 2.0 + 0.1 * (t as f64);
            }
        }
        let bp = crf.infer_marginals(&unary, t_max, &[]).expect("bp");
        let exact = exact_chain_marginals(&unary, &transition, n, t_max);
        for k in 0..t_max * n {
            assert!(
                (bp[k] - exact[k]).abs() < 1e-5,
                "idx {k}: bp={} exact={}",
                bp[k],
                exact[k]
            );
        }
    }

    #[test]
    fn no_skip_marginals_equal_brute_force_short() {
        // Brute-force enumeration on a length-3 chain, n_labels=2.
        let n = 2;
        let transition = vec![0.3, -0.4, 0.2, 0.5];
        let crf = SkipChainCrf::new(cfg(n), transition.clone(), vec![0.0; 4]).expect("new");
        let t_max = 3;
        let unary = vec![0.5, -0.2, 0.1, 0.7, -0.3, 0.4];
        let bp = crf.infer_marginals(&unary, t_max, &[]).expect("bp");
        // Enumerate all 2^3 = 8 sequences.
        let mut marg = vec![0.0; t_max * n];
        let mut z = 0.0;
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    let y = [a, b, c];
                    let mut score = 0.0;
                    for (t, &yt) in y.iter().enumerate() {
                        score += unary[t * n + yt];
                        if t > 0 {
                            score += transition[y[t - 1] * n + yt];
                        }
                    }
                    let p = score.exp();
                    z += p;
                    for (t, &yt) in y.iter().enumerate() {
                        marg[t * n + yt] += p;
                    }
                }
            }
        }
        for v in marg.iter_mut() {
            *v /= z;
        }
        for k in 0..t_max * n {
            assert!(
                (bp[k] - marg[k]).abs() < 1e-6,
                "idx {k}: {} vs {}",
                bp[k],
                marg[k]
            );
        }
    }

    #[test]
    fn no_skip_decode_equals_viterbi() {
        // Build an equivalent LinearChainCrf whose emissions reproduce the unary.
        let n = 3;
        let k = n; // one-hot features per position -> emission = unary
        let transition = vec![0.4, -0.1, 0.2, -0.3, 0.5, 0.0, 0.1, -0.2, 0.6];
        let crf = SkipChainCrf::new(cfg(n), transition.clone(), vec![0.0; 9]).expect("new");
        let t_max = 6;
        let mut unary = vec![0.0; t_max * n];
        for t in 0..t_max {
            for l in 0..n {
                unary[t * n + l] = ((t * 5 + l * 11) as f64 % 7.0) - 3.0;
            }
        }
        let bp_labels = crf.decode(&unary, t_max, &[]).expect("decode");

        // LinearChainCrf with identity emission weights and one-hot features so that
        // emit_score(l, x_t) = unary[t, l].
        let mut lc = LinearChainCrf::zeros(n, k).expect("lc");
        lc.transitions = transition;
        // emissions[l*k + f] = 1 if l==f else 0.
        for l in 0..n {
            for f in 0..k {
                lc.emissions[l * k + f] = if l == f { 1.0 } else { 0.0 };
            }
        }
        let mut x = vec![0.0; t_max * k];
        for t in 0..t_max {
            for f in 0..k {
                x[t * k + f] = unary[t * n + f];
            }
        }
        let vit = viterbi_decode(&lc, &x).expect("viterbi");
        assert_eq!(bp_labels, vit);
    }

    #[test]
    fn skip_edge_pulls_marginals_to_agreement() {
        // Two positions with identical evidence and a strongly-attractive skip edge
        // should agree more than without the edge.  Use a diagonal skip potential.
        let n = 2;
        let transition = vec![0.0, 0.0, 0.0, 0.0];
        // Attractive: large on equal labels, small on disagreement.
        let skip = vec![2.0, -2.0, -2.0, 2.0];
        let crf = SkipChainCrf::new(cfg(n), transition, skip).expect("new");
        let t_max = 3;
        // Position 0 prefers label 0; position 2 prefers label 1 (conflicting),
        // position 1 neutral.
        let unary = vec![1.0, -1.0, 0.0, 0.0, -1.0, 1.0];
        let no_skip = crf.infer_marginals(&unary, t_max, &[]).expect("ns");
        let with_skip = crf.infer_marginals(&unary, t_max, &[(0, 2)]).expect("ws");
        // Without the skip edge, position 0 favours label 0 and position 2 favours
        // label 1, so they disagree.  The attractive skip edge should bring the
        // marginals of position 0 and 2 closer together.
        let dist_no = (no_skip[0] - no_skip[2 * n]).abs();
        let dist_ws = (with_skip[0] - with_skip[2 * n]).abs();
        assert!(
            dist_ws < dist_no,
            "skip edge should reduce disagreement: no={dist_no} ws={dist_ws}"
        );
    }

    #[test]
    fn decode_returns_valid_labels() {
        let n = 4;
        let crf = SkipChainCrf::new(cfg(n), vec![0.1; 16], vec![0.0; 16]).expect("new");
        let t_max = 5;
        let mut unary = vec![0.0; t_max * n];
        for (i, v) in unary.iter_mut().enumerate() {
            *v = (i as f64 % 3.0) - 1.0;
        }
        let labels = crf.decode(&unary, t_max, &[(0, 3), (1, 4)]).expect("dec");
        assert_eq!(labels.len(), t_max);
        for &l in &labels {
            assert!(l < n);
        }
    }

    #[test]
    fn bp_converges_on_short_sequence() {
        let n = 2;
        let transition = vec![0.5, -0.2, 0.1, 0.3];
        let skip = vec![0.4, -0.1, -0.1, 0.4];
        let crf = SkipChainCrf::new(cfg(n), transition, skip).expect("new");
        let unary = vec![0.3, -0.2, 0.1, 0.4, -0.3, 0.2, 0.0, 0.1];
        let (_m, iters, converged) = crf
            .infer_marginals_with_status(&unary, 4, &[(0, 3)])
            .expect("bp");
        assert!(converged, "BP should converge");
        assert!(iters <= 200);
    }

    #[test]
    fn uniform_unary_uniform_potentials_uniform_marginals() {
        let n = 3;
        let crf = SkipChainCrf::new(cfg(n), vec![0.0; 9], vec![0.0; 9]).expect("new");
        let t_max = 4;
        let unary = vec![0.0; t_max * n];
        let m = crf.infer_marginals(&unary, t_max, &[(0, 2)]).expect("m");
        for t in 0..t_max {
            for l in 0..n {
                assert!(
                    (m[t * n + l] - 1.0 / n as f64).abs() < 1e-9,
                    "pos {t} label {l}: {}",
                    m[t * n + l]
                );
            }
        }
    }

    #[test]
    fn deterministic_inference() {
        let n = 2;
        let transition = vec![0.5, -0.2, 0.1, 0.3];
        let skip = vec![0.4, -0.1, -0.1, 0.4];
        let crf = SkipChainCrf::new(cfg(n), transition, skip).expect("new");
        let unary = vec![0.3, -0.2, 0.1, 0.4, -0.3, 0.2];
        let a = crf.infer_marginals(&unary, 3, &[(0, 2)]).expect("a");
        let b = crf.infer_marginals(&unary, 3, &[(0, 2)]).expect("b");
        assert_eq!(a, b);
        let da = crf.decode(&unary, 3, &[(0, 2)]).expect("da");
        let db = crf.decode(&unary, 3, &[(0, 2)]).expect("db");
        assert_eq!(da, db);
    }

    #[test]
    fn seq_len_one_marginal_is_softmax() {
        let n = 3;
        let crf = SkipChainCrf::new(cfg(n), vec![0.0; 9], vec![0.0; 9]).expect("new");
        let unary = vec![1.0, 0.0, -1.0];
        let m = crf.infer_marginals(&unary, 1, &[]).expect("m");
        // Softmax of unary.
        let mx = unary.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = unary.iter().map(|&u| (u - mx).exp()).collect();
        let s: f64 = exps.iter().sum();
        for l in 0..n {
            assert!((m[l] - exps[l] / s).abs() < 1e-12, "label {l}");
        }
    }

    #[test]
    fn single_label_trivial() {
        let crf = SkipChainCrf::new(cfg(1), vec![0.0], vec![0.0]).expect("new");
        let unary = vec![3.0, -1.0, 0.5];
        let m = crf.infer_marginals(&unary, 3, &[(0, 2)]).expect("m");
        for v in &m {
            assert!((v - 1.0).abs() < 1e-12);
        }
        let labels = crf.decode(&unary, 3, &[(0, 2)]).expect("dec");
        assert_eq!(labels, vec![0, 0, 0]);
    }

    #[test]
    fn err_transition_wrong_length() {
        let r = SkipChainCrf::new(cfg(2), vec![0.0; 3], vec![0.0; 4]);
        assert!(matches!(r, Err(SeqError::ShapeMismatch { .. })));
    }

    #[test]
    fn err_skip_potential_wrong_length() {
        let r = SkipChainCrf::new(cfg(2), vec![0.0; 4], vec![0.0; 5]);
        assert!(matches!(r, Err(SeqError::ShapeMismatch { .. })));
    }

    #[test]
    fn err_unary_wrong_length() {
        let crf = SkipChainCrf::new(cfg(2), vec![0.0; 4], vec![0.0; 4]).expect("new");
        let r = crf.infer_marginals(&[0.0, 0.0, 0.0], 3, &[]);
        assert!(matches!(r, Err(SeqError::ShapeMismatch { .. })));
    }

    #[test]
    fn err_skip_edge_out_of_range() {
        let crf = SkipChainCrf::new(cfg(2), vec![0.0; 4], vec![0.0; 4]).expect("new");
        let unary = vec![0.0; 6];
        let r = crf.infer_marginals(&unary, 3, &[(0, 9)]);
        assert!(matches!(r, Err(SeqError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn err_skip_edge_i_ge_j() {
        let crf = SkipChainCrf::new(cfg(2), vec![0.0; 4], vec![0.0; 4]).expect("new");
        let unary = vec![0.0; 6];
        let r = crf.infer_marginals(&unary, 3, &[(2, 1)]);
        assert!(matches!(r, Err(SeqError::GraphInvariantViolated(_))));
        let r2 = crf.infer_marginals(&unary, 3, &[(1, 1)]);
        assert!(matches!(r2, Err(SeqError::GraphInvariantViolated(_))));
    }

    #[test]
    fn err_n_labels_zero() {
        let r = SkipChainCrf::new(cfg(0), vec![], vec![]);
        assert!(matches!(r, Err(SeqError::InvalidConfiguration(_))));
    }

    #[test]
    fn err_max_bp_iters_zero() {
        let c = SkipChainConfig {
            n_labels: 2,
            max_bp_iters: 0,
            bp_tol: 1e-6,
        };
        let r = SkipChainCrf::new(c, vec![0.0; 4], vec![0.0; 4]);
        assert!(matches!(r, Err(SeqError::InvalidConfiguration(_))));
    }

    #[test]
    fn err_bp_tol_non_positive() {
        let c = SkipChainConfig {
            n_labels: 2,
            max_bp_iters: 10,
            bp_tol: 0.0,
        };
        let r = SkipChainCrf::new(c, vec![0.0; 4], vec![0.0; 4]);
        assert!(matches!(r, Err(SeqError::InvalidParameter { .. })));
        let c2 = SkipChainConfig {
            n_labels: 2,
            max_bp_iters: 10,
            bp_tol: -1.0,
        };
        let r2 = SkipChainCrf::new(c2, vec![0.0; 4], vec![0.0; 4]);
        assert!(matches!(r2, Err(SeqError::InvalidParameter { .. })));
    }

    #[test]
    fn err_empty_input_seq_len_zero() {
        let crf = SkipChainCrf::new(cfg(2), vec![0.0; 4], vec![0.0; 4]).expect("new");
        let r = crf.infer_marginals(&[], 0, &[]);
        assert!(matches!(r, Err(SeqError::EmptyInput)));
    }

    #[test]
    fn n_labels_accessor() {
        let crf = SkipChainCrf::new(cfg(5), vec![0.0; 25], vec![0.0; 25]).expect("new");
        assert_eq!(crf.n_labels(), 5);
    }

    #[test]
    fn with_damping_validates() {
        let crf = SkipChainCrf::new(cfg(2), vec![0.0; 4], vec![0.0; 4]).expect("new");
        assert!(crf.clone().with_damping(0.3).is_ok());
        assert!(crf.clone().with_damping(1.0).is_ok());
        assert!(crf.clone().with_damping(0.0).is_err());
        assert!(crf.with_damping(1.5).is_err());
    }

    #[test]
    fn no_skip_decode_equals_viterbi_two_labels() {
        // A second decode/Viterbi agreement test with a unique (non-degenerate) MAP
        // sequence 0,1,1,0 induced by asymmetric unary log-potentials.
        let n = 2;
        let transition = vec![0.8, -0.5, -0.3, 0.6];
        let crf = SkipChainCrf::new(cfg(n), transition.clone(), vec![0.0; 4]).expect("new");
        let t_max = 4;
        let unary = vec![3.0, -1.0, -1.0, 3.0, -2.0, 2.0, 2.5, -1.5];
        let bp_labels = crf.decode(&unary, t_max, &[]).expect("dec");
        let mut lc = LinearChainCrf::zeros(n, n).expect("lc");
        lc.transitions = transition;
        for l in 0..n {
            lc.emissions[l * n + l] = 1.0;
        }
        let x = unary.clone();
        let vit = viterbi_decode(&lc, &x).expect("vit");
        assert_eq!(bp_labels, vit);
    }
}
