//! Zigzag persistent homology for sequences of simplicial complexes related by
//! inclusions (the *inclusion zigzag*).
//!
//! Reference: Gunnar Carlsson and Vin de Silva, "Zigzag Persistence", Foundations
//! of Computational Mathematics 10 (2010), 367–405; and Carlsson, de Silva,
//! Morozov, "Zigzag persistent homology and real-valued functions", SoCG 2009,
//! which describes the elementary add/remove update used here.
//!
//! # Model
//!
//! The input is a sequence of simplicial complexes
//!
//! ```text
//!   X_0  ─a_0─  X_1  ─a_1─  X_2  ─ … ─  X_K
//! ```
//!
//! where each arrow `a_i` is a *single-direction inclusion*:
//!
//! * [`ZigzagArrow::Forward`]  — `X_i ⊆ X_{i+1}` (simplices are **added**);
//! * [`ZigzagArrow::Backward`] — `X_{i+1} ⊆ X_i` (simplices are **removed**).
//!
//! Zigzag homology assigns to this diagram a multiset of *intervals* `[b, d]`
//! (closed, in complex-index units): an interval `[b, d]` in dimension `k` means a
//! `k`-dimensional homology class that is alive in exactly the complexes
//! `X_b, …, X_d`.
//!
//! # Algorithm (elementary, representative-based)
//!
//! Each inclusion is decomposed into a tape of **elementary** single-simplex
//! additions / removals (faces before cofaces on the way up, cofaces before faces
//! on the way down).  We then maintain, over Z₂:
//!
//! * a reduced basis of **alive cycle representatives** (one per alive homology
//!   class), each tagged with the *complex index* at which it was born and a
//!   strictly-increasing *operation counter* used to apply the elder rule;
//! * a reduced basis of **boundary generators** per dimension, each a cycle that is
//!   a boundary together with an explicit *filling* (a chain whose boundary is that
//!   cycle).
//!
//! On an **addition** of σ (dimension `d`) we reduce `∂σ` first against the boundary
//! basis, then against the alive `(d−1)`-cycles.  If `∂σ` was already a boundary, σ
//! creates a `d`-class (`rep = σ +` the fillings used); otherwise σ destroys the
//! youngest `(d−1)`-class in the expansion of `[∂σ]`, and `∂σ` becomes a new
//! boundary generator filled by σ.
//!
//! On a **removal** of a maximal σ (dimension `d`):
//!
//! * if σ appears in some alive `d`-cycle representatives, removing it **destroys**
//!   the youngest such class (after XOR-transferring that representative into the
//!   others so they remain valid cycles);
//! * otherwise σ is a filler, and removing it **resurrects** the `(d−1)`-cycle it
//!   used to bound — a class **born by deletion**, the characteristic effect of
//!   zigzag persistence.
//!
//! Births recorded on a forward arrow `i → i+1` use the right index `i+1`; deaths
//! use the left index `i`.  A class born by deletion on a backward arrow `i → i+1`
//! is born at `i+1`; a class destroyed by a backward arrow dies at `i`.  Classes
//! still alive at the end close at the last complex index.

use crate::complex::simplex::Simplex;
use crate::error::{TdaError, TdaResult};
use std::collections::HashMap;

// ─── Public types ───────────────────────────────────────────────────────────────

/// Direction of a single inclusion arrow in a zigzag sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZigzagArrow {
    /// `X_i ⊆ X_{i+1}`: the simplices in `X_{i+1} \ X_i` are **added**.
    Forward,
    /// `X_{i+1} ⊆ X_i`: the simplices in `X_i \ X_{i+1}` are **removed**.
    Backward,
}

/// One simplicial complex in a zigzag sequence (a face-closed set of simplices).
#[derive(Debug, Clone, Default)]
pub struct ZigzagComplex {
    /// The simplices of this complex.
    pub simplices: Vec<Simplex>,
}

/// A zigzag input: a sequence of complexes joined by single-direction arrows.
///
/// Invariant: `complexes.len() == arrows.len() + 1`.
#[derive(Debug, Clone, Default)]
pub struct ZigzagInput {
    /// The complexes `X_0, …, X_K`.
    pub complexes: Vec<ZigzagComplex>,
    /// The arrows `a_0, …, a_{K-1}` joining consecutive complexes.
    pub arrows: Vec<ZigzagArrow>,
}

/// A single zigzag interval (bar): a homology class alive in `X_birth … X_death`.
#[derive(Debug, Clone)]
pub struct ZigzagBar {
    /// Homological dimension of the class.
    pub dim: usize,
    /// First complex index at which the class is alive.
    pub birth: usize,
    /// Last complex index at which the class is alive.
    pub death: usize,
}

impl ZigzagBar {
    /// Number of complexes the class spans, `death − birth + 1`.
    pub fn length(&self) -> usize {
        self.death - self.birth + 1
    }
}

/// The collection of zigzag intervals produced by [`zigzag_persistence`].
#[derive(Debug, Clone, Default)]
pub struct ZigzagBarcode {
    /// All intervals, in no particular order.
    pub bars: Vec<ZigzagBar>,
}

impl ZigzagBarcode {
    /// All bars of a given homological dimension.
    pub fn bars_in_dim(&self, dim: usize) -> Vec<&ZigzagBar> {
        self.bars.iter().filter(|b| b.dim == dim).collect()
    }

    /// The Betti number `β_dim(X_{complex_index})`: the number of intervals of
    /// dimension `dim` that are alive at `complex_index` (i.e. `birth ≤ i ≤ death`).
    pub fn betti(&self, complex_index: usize, dim: usize) -> usize {
        self.bars
            .iter()
            .filter(|b| b.dim == dim && b.birth <= complex_index && complex_index <= b.death)
            .count()
    }
}

// ─── Internal representation ────────────────────────────────────────────────────

/// An alive homology class with an explicit cycle representative.
#[derive(Debug, Clone)]
struct AliveClass {
    dim: usize,
    /// Complex index at which the class was born.
    birth_complex: usize,
    /// Inclusion-block index at birth (0 = initial complex, then one per arrow).
    ///
    /// A class created and destroyed within the *same* block is never alive at any
    /// `X_k` (it is born and dies strictly *between* complexes), so its interval is
    /// empty and must be dropped.
    birth_block: usize,
    /// Strictly increasing operation counter at birth (for the elder rule).
    birth_op: usize,
    /// Cycle representative: sorted ids of `dim`-simplices (a Z₂ chain with ∂ = 0).
    rep: Vec<usize>,
}

/// A boundary generator: a cycle that bounds, together with a filling chain.
#[derive(Debug, Clone)]
struct BoundaryGen {
    /// The `dim`-cycle (sorted simplex ids) that is a boundary.
    chain: Vec<usize>,
    /// A `(dim+1)`-chain (sorted simplex ids) whose boundary equals `chain`.
    filling: Vec<usize>,
}

/// One elementary tape operation.
#[derive(Debug, Clone)]
enum ElemOp {
    /// Add a single simplex.
    Add {
        verts: Vec<usize>,
        /// Inclusion-block index (0 = initial complex, then one per arrow).
        block: usize,
        /// Complex index to record as the birth of a class created here.
        create_index: usize,
        /// Complex index to record as the death of a class destroyed here.
        destroy_index: usize,
    },
    /// Remove a single simplex.
    Remove {
        verts: Vec<usize>,
        /// Inclusion-block index.
        block: usize,
        /// Complex index to record as the birth of a class born by deletion here.
        create_index: usize,
        /// Complex index to record as the death of a class destroyed here.
        destroy_index: usize,
    },
}

// ─── Z₂ chain helpers ───────────────────────────────────────────────────────────

/// Symmetric difference (XOR over Z₂) of two sorted id lists, written into `target`.
fn xor_into(target: &mut Vec<usize>, source: &[usize]) {
    let tgt = std::mem::take(target);
    let mut result = Vec::with_capacity(tgt.len() + source.len());
    let mut ti = 0usize;
    let mut si = 0usize;
    while ti < tgt.len() && si < source.len() {
        match tgt[ti].cmp(&source[si]) {
            std::cmp::Ordering::Less => {
                result.push(tgt[ti]);
                ti += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(source[si]);
                si += 1;
            }
            std::cmp::Ordering::Equal => {
                ti += 1;
                si += 1;
            }
        }
    }
    result.extend_from_slice(&tgt[ti..]);
    result.extend_from_slice(&source[si..]);
    *target = result;
}

/// The pivot (lowest / maximum id) of a chain, or `None` if it is zero.
#[inline]
fn pivot(chain: &[usize]) -> Option<usize> {
    chain.last().copied()
}

// ─── The zigzag engine ──────────────────────────────────────────────────────────

/// Mutable state of the elementary zigzag computation.
struct ZigzagEngine {
    /// Registry: id → vertex list of the simplex (grows, never shrinks).
    id_verts: Vec<Vec<usize>>,
    /// Registry: id → dimension.
    id_dim: Vec<usize>,
    /// Currently present simplices: vertices → id.
    present: HashMap<Vec<usize>, usize>,
    /// Alive homology classes.
    alive: Vec<AliveClass>,
    /// Boundary generators per dimension (`boundaries[d]` holds `d`-cycles).
    boundaries: Vec<Vec<BoundaryGen>>,
    /// Completed (and to-be-completed) bars.
    bars: Vec<ZigzagBar>,
    /// Strictly increasing operation counter (elder rule).
    op_counter: usize,
}

impl ZigzagEngine {
    fn new(max_dim: usize) -> Self {
        Self {
            id_verts: Vec::new(),
            id_dim: Vec::new(),
            present: HashMap::new(),
            alive: Vec::new(),
            boundaries: vec![Vec::new(); max_dim + 2],
            bars: Vec::new(),
            op_counter: 0,
        }
    }

    /// Register a freshly added simplex, returning its new id.
    fn register(&mut self, verts: &[usize], dim: usize) -> usize {
        let id = self.id_verts.len();
        self.id_verts.push(verts.to_vec());
        self.id_dim.push(dim);
        self.present.insert(verts.to_vec(), id);
        id
    }

    /// Boundary of `σ` (given by `verts`, dimension `d`) as a sorted chain of the
    /// ids of its `(d−1)`-faces.  All faces must currently be present.
    fn boundary_chain(&self, verts: &[usize], d: usize) -> TdaResult<Vec<usize>> {
        if d == 0 {
            return Ok(Vec::new());
        }
        let simplex = Simplex::new(verts.to_vec())?;
        let mut chain: Vec<usize> = Vec::with_capacity(d + 1);
        for face in simplex.faces() {
            match self.present.get(&face.vertices) {
                Some(&fid) => chain.push(fid),
                None => {
                    return Err(TdaError::ClosureViolation(format!(
                        "face {:?} of {:?} absent when taking boundary",
                        face.vertices, verts
                    )));
                }
            }
        }
        chain.sort_unstable();
        Ok(chain)
    }

    /// Reduce `chain` against the alive classes of dimension `dim` by pivot,
    /// returning the indices (into `self.alive`) of every class XOR-ed in.
    ///
    /// After the call, `chain`'s pivot (if any) is not shared by any alive
    /// `dim`-class — the alive `dim`-classes are kept pivot-reduced, so this both
    /// classifies `chain` and leaves it in reduced form.
    fn reduce_against_alive(&self, chain: &mut Vec<usize>, dim: usize) -> Vec<usize> {
        let mut used = Vec::new();
        while let Some(piv) = pivot(chain) {
            let matched = self
                .alive
                .iter()
                .position(|cls| cls.dim == dim && pivot(&cls.rep) == Some(piv));
            match matched {
                Some(idx) => {
                    let rep = self.alive[idx].rep.clone();
                    xor_into(chain, &rep);
                    used.push(idx);
                }
                None => break,
            }
        }
        used
    }

    /// Reduce `chain` (a `dim`-cycle) against the boundary basis of dimension `dim`,
    /// simultaneously accumulating the corresponding fillings into `fill`.
    fn reduce_against_boundaries(&self, chain: &mut Vec<usize>, fill: &mut Vec<usize>, dim: usize) {
        while let Some(piv) = pivot(chain) {
            let matched = self.boundaries[dim]
                .iter()
                .position(|bgen| pivot(&bgen.chain) == Some(piv));
            match matched {
                Some(idx) => {
                    let gchain = self.boundaries[dim][idx].chain.clone();
                    let gfill = self.boundaries[dim][idx].filling.clone();
                    xor_into(chain, &gchain);
                    xor_into(fill, &gfill);
                }
                None => break,
            }
        }
    }

    /// Insert a new alive class, keeping the alive `dim`-classes pivot-reduced.
    ///
    /// The new representative `rep` is first reduced against existing alive
    /// `dim`-classes (preserving its homology class up to the reduced basis, which
    /// is irrelevant for the interval indices).  `birth_complex` / `birth_op` are
    /// recorded as given.
    fn insert_alive(
        &mut self,
        dim: usize,
        mut rep: Vec<usize>,
        birth_complex: usize,
        birth_block: usize,
        birth_op: usize,
    ) {
        // Reduce against existing alive classes so the pivot is unique.
        let _ = self.reduce_against_alive(&mut rep, dim);
        self.alive.push(AliveClass {
            dim,
            birth_complex,
            birth_block,
            birth_op,
            rep,
        });
    }

    /// Emit a closing bar for `cls` dying at `death` during block `block`, unless the
    /// interval is empty (created and destroyed within the same inclusion block, in
    /// which case the class was never alive at any complex).
    fn close(&mut self, cls: &AliveClass, death: usize, block: usize) {
        if cls.birth_block == block {
            return; // empty interval — never alive at any X_k.
        }
        self.bars.push(ZigzagBar {
            dim: cls.dim,
            birth: cls.birth_complex,
            death,
        });
    }

    /// Process one elementary addition.
    fn do_add(
        &mut self,
        verts: &[usize],
        block: usize,
        create_index: usize,
        destroy_index: usize,
    ) -> TdaResult<()> {
        self.op_counter += 1;
        let op = self.op_counter;
        let simplex = Simplex::new(verts.to_vec())?;
        let d = simplex.dim();

        // Boundary chain (faces must be present) and its filling (σ itself).
        let bnd = self.boundary_chain(verts, d)?;
        let id = self.register(verts, d);

        let mut chain = bnd;
        let mut fill = vec![id];

        // Reduce against the boundary basis of dimension d-1.
        if d >= 1 {
            self.reduce_against_boundaries(&mut chain, &mut fill, d - 1);
        }

        if chain.is_empty() {
            // ∂σ was already a boundary ⇒ σ creates a d-class with rep = fill.
            self.insert_alive(d, fill, create_index, block, op);
            return Ok(());
        }

        // Otherwise σ is a destroyer: record the boundary-reduced chain/filling as a
        // new boundary generator (fresh pivot), then find and kill the youngest
        // (d-1)-class in the expansion of [∂σ].
        let new_gen = BoundaryGen {
            chain: chain.clone(),
            filling: fill,
        };

        // Expand [∂σ] in the alive (d-1)-cycle basis.
        let mut expand = chain.clone();
        let used = self.reduce_against_alive(&mut expand, d.saturating_sub(1));

        if d >= 1 {
            self.boundaries[d - 1].push(new_gen);
        }

        if let Some(victim) = used
            .iter()
            .max_by_key(|&&idx| self.alive[idx].birth_op)
            .copied()
        {
            let cls = self.alive.remove(victim);
            self.close(&cls, destroy_index, block);
        }
        Ok(())
    }

    /// Process one elementary removal of a maximal simplex.
    fn do_remove(
        &mut self,
        verts: &[usize],
        block: usize,
        create_index: usize,
        destroy_index: usize,
    ) -> TdaResult<()> {
        self.op_counter += 1;
        let op = self.op_counter;
        let id = match self.present.get(verts) {
            Some(&id) => id,
            None => {
                return Err(TdaError::ClosureViolation(format!(
                    "removal of absent simplex {verts:?}"
                )));
            }
        };
        let d = self.id_dim[id];

        // Which alive d-classes contain σ?
        let containing: Vec<usize> = self
            .alive
            .iter()
            .enumerate()
            .filter(|(_, cls)| cls.dim == d && cls.rep.binary_search(&id).is_ok())
            .map(|(idx, _)| idx)
            .collect();

        if !containing.is_empty() {
            // Case 1 (destroy): consolidate so only the youngest contains σ.
            let pivot_cls = containing
                .iter()
                .max_by_key(|&&idx| self.alive[idx].birth_op)
                .copied()
                .unwrap_or(containing[0]);
            let pivot_rep = self.alive[pivot_cls].rep.clone();
            for &idx in &containing {
                if idx != pivot_cls {
                    xor_into(&mut self.alive[idx].rep, &pivot_rep);
                }
            }
            // Clear σ from any boundary fillings of dimension d (∂(d-cycle)=0, so the
            // recorded boundary chains are unaffected).
            for bgen in &mut self.boundaries[d] {
                if bgen.filling.binary_search(&id).is_ok() {
                    xor_into(&mut bgen.filling, &pivot_rep);
                }
            }
            let cls = self.alive.remove(pivot_cls);
            self.close(&cls, destroy_index, block);
        } else if d >= 1 {
            // Case 2 (born by deletion): σ is a filler; resurrect the (d-1)-cycle it
            // bounded.  Find the boundary generator of dimension d-1 filled by σ.
            let gen_idx = self.boundaries[d - 1]
                .iter()
                .position(|bgen| bgen.filling.binary_search(&id).is_ok());
            if let Some(gi) = gen_idx {
                // Consolidate fillings so only this generator is filled by σ.
                let pivot_chain = self.boundaries[d - 1][gi].chain.clone();
                let pivot_fill = self.boundaries[d - 1][gi].filling.clone();
                let len = self.boundaries[d - 1].len();
                for j in 0..len {
                    if j != gi && self.boundaries[d - 1][j].filling.binary_search(&id).is_ok() {
                        xor_into(&mut self.boundaries[d - 1][j].chain, &pivot_chain);
                        xor_into(&mut self.boundaries[d - 1][j].filling, &pivot_fill);
                    }
                }
                let resurrected = self.boundaries[d - 1][gi].chain.clone();
                self.boundaries[d - 1].remove(gi);
                // The resurrected (d-1)-cycle is now a genuine class, born at create_index.
                self.insert_alive(d - 1, resurrected, create_index, block, op);
            }
            // If no generator is found the homology is unchanged (e.g. a redundant
            // removal that cannot occur for valid face-closed inclusion inputs).
        }

        // Retire σ from the present set (its id stays in the registry).
        self.present.remove(verts);
        Ok(())
    }

    /// Close all still-alive classes at the final complex index.
    fn finalize(&mut self, last_index: usize) {
        for cls in self.alive.drain(..) {
            self.bars.push(ZigzagBar {
                dim: cls.dim,
                birth: cls.birth_complex,
                death: last_index,
            });
        }
    }
}

// ─── Validation & tape construction ─────────────────────────────────────────────

/// Verify that a complex is face-closed (every face of every simplex is present).
fn check_face_closed(complex: &ZigzagComplex) -> TdaResult<()> {
    let present: std::collections::HashSet<&Vec<usize>> =
        complex.simplices.iter().map(|s| &s.vertices).collect();
    for s in &complex.simplices {
        for face in s.faces() {
            if !present.contains(&face.vertices) {
                return Err(TdaError::ClosureViolation(format!(
                    "complex not face-closed: face {:?} of {:?} missing",
                    face.vertices, s.vertices
                )));
            }
        }
    }
    Ok(())
}

/// Build a `set` of the vertex lists of a complex for inclusion tests.
fn vertex_set(complex: &ZigzagComplex) -> std::collections::HashSet<Vec<usize>> {
    complex
        .simplices
        .iter()
        .map(|s| s.vertices.clone())
        .collect()
}

/// Order a set of simplices by **increasing** dimension, then lexicographically
/// (faces before cofaces) — the order in which they are *added* on a forward arrow.
fn order_increasing(simplices: &[Simplex]) -> Vec<Vec<usize>> {
    let mut v: Vec<Vec<usize>> = simplices.iter().map(|s| s.vertices.clone()).collect();
    v.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
    v
}

/// Order a set of simplices by **decreasing** dimension, then reverse-lexicographically
/// (cofaces before faces) — the order in which they are *removed* on a backward arrow.
fn order_decreasing(simplices: &[Simplex]) -> Vec<Vec<usize>> {
    let mut v: Vec<Vec<usize>> = simplices.iter().map(|s| s.vertices.clone()).collect();
    v.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| b.cmp(a)));
    v
}

/// Compute zigzag persistence of an inclusion-zigzag sequence.
///
/// See the module documentation for the model, the index conventions, and the
/// algorithm.
///
/// # Errors
///
/// * [`TdaError::EmptyComplex`] — the input has no complexes.
/// * [`TdaError::DimensionMismatch`] — `complexes.len() != arrows.len() + 1`.
/// * [`TdaError::DimensionTooLarge`] — some simplex has dimension `> 6`.
/// * [`TdaError::InvalidSimplex`] — a simplex has an invalid vertex list.
/// * [`TdaError::ClosureViolation`] — a complex is not face-closed, or a declared
///   inclusion does not actually hold (the added/removed set is not on the correct
///   side of the arrow).
pub fn zigzag_persistence(input: &ZigzagInput) -> TdaResult<ZigzagBarcode> {
    // (A) Validate the input.
    if input.complexes.is_empty() {
        return Err(TdaError::EmptyComplex);
    }
    if input.complexes.len() != input.arrows.len() + 1 {
        return Err(TdaError::DimensionMismatch {
            expected: input.arrows.len() + 1,
            got: input.complexes.len(),
        });
    }

    let mut max_dim = 0usize;
    for complex in &input.complexes {
        for s in &complex.simplices {
            // Validate the simplex (sorted, no duplicates, non-empty).
            let checked = Simplex::new(s.vertices.clone())?;
            if checked.dim() > 6 {
                return Err(TdaError::DimensionTooLarge(checked.dim()));
            }
            if checked.dim() > max_dim {
                max_dim = checked.dim();
            }
        }
        check_face_closed(complex)?;
    }

    // Validate each arrow's claimed inclusion.
    for (i, arrow) in input.arrows.iter().enumerate() {
        let left = vertex_set(&input.complexes[i]);
        let right = vertex_set(&input.complexes[i + 1]);
        match arrow {
            ZigzagArrow::Forward => {
                // X_i ⊆ X_{i+1}: every simplex of the left is in the right.
                if !left.is_subset(&right) {
                    return Err(TdaError::ClosureViolation(format!(
                        "forward arrow {i}: X_{i} is not a subset of X_{}",
                        i + 1
                    )));
                }
            }
            ZigzagArrow::Backward => {
                // X_{i+1} ⊆ X_i: every simplex of the right is in the left.
                if !right.is_subset(&left) {
                    return Err(TdaError::ClosureViolation(format!(
                        "backward arrow {i}: X_{} is not a subset of X_{i}",
                        i + 1
                    )));
                }
            }
        }
    }

    // (A.2) Build the elementary tape.
    let mut tape: Vec<ElemOp> = Vec::new();

    // Initial complex X_0: add everything, born at complex index 0 (block 0).
    for verts in order_increasing(&input.complexes[0].simplices) {
        tape.push(ElemOp::Add {
            verts,
            block: 0,
            create_index: 0,
            destroy_index: 0,
        });
    }

    for (i, arrow) in input.arrows.iter().enumerate() {
        let left = vertex_set(&input.complexes[i]);
        let right = vertex_set(&input.complexes[i + 1]);
        match arrow {
            ZigzagArrow::Forward => {
                // Added simplices = X_{i+1} \ X_i, faces before cofaces.
                let added: Vec<Simplex> = input.complexes[i + 1]
                    .simplices
                    .iter()
                    .filter(|s| !left.contains(&s.vertices))
                    .cloned()
                    .collect();
                for verts in order_increasing(&added) {
                    tape.push(ElemOp::Add {
                        verts,
                        block: i + 1,
                        create_index: i + 1,
                        destroy_index: i,
                    });
                }
            }
            ZigzagArrow::Backward => {
                // Removed simplices = X_i \ X_{i+1}, cofaces before faces.
                let removed: Vec<Simplex> = input.complexes[i]
                    .simplices
                    .iter()
                    .filter(|s| !right.contains(&s.vertices))
                    .cloned()
                    .collect();
                for verts in order_decreasing(&removed) {
                    tape.push(ElemOp::Remove {
                        verts,
                        block: i + 1,
                        create_index: i + 1,
                        destroy_index: i,
                    });
                }
            }
        }
    }

    // (B) Run the engine over the tape.
    let mut engine = ZigzagEngine::new(max_dim);
    for op in &tape {
        match op {
            ElemOp::Add {
                verts,
                block,
                create_index,
                destroy_index,
            } => engine.do_add(verts, *block, *create_index, *destroy_index)?,
            ElemOp::Remove {
                verts,
                block,
                create_index,
                destroy_index,
            } => engine.do_remove(verts, *block, *create_index, *destroy_index)?,
        }
    }

    // (C) Close survivors at the last complex index and emit the barcode.
    let last_index = input.complexes.len() - 1;
    engine.finalize(last_index);
    Ok(ZigzagBarcode { bars: engine.bars })
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::filtration::{FilteredSimplex, Filtration};
    use crate::homology::boundary::BoundaryMatrix;
    use crate::homology::persistent::extract_persistence_pairs;
    use crate::homology::reduction::reduce_boundary_matrix;

    fn s(verts: &[usize]) -> Simplex {
        Simplex::new(verts.to_vec()).expect("valid simplex")
    }

    fn complex(simplices: &[&[usize]]) -> ZigzagComplex {
        ZigzagComplex {
            simplices: simplices.iter().map(|v| s(v)).collect(),
        }
    }

    #[test]
    fn worked_example_build_and_tear() {
        // X0 = {0},{1},{2}
        // X1 = + edge {0,1}              (Forward)
        // X2 = + {0,2} + {1,2}           (Forward, triangle boundary = 1-cycle)
        // X3 = + {0,1,2}                 (Forward, fills the loop)
        // X4 = − {0,1,2}                 (Backward, loop reborn)
        let x0 = complex(&[&[0], &[1], &[2]]);
        let x1 = complex(&[&[0], &[1], &[2], &[0, 1]]);
        let x2 = complex(&[&[0], &[1], &[2], &[0, 1], &[0, 2], &[1, 2]]);
        let x3 = complex(&[&[0], &[1], &[2], &[0, 1], &[0, 2], &[1, 2], &[0, 1, 2]]);
        let x4 = x2.clone();

        let input = ZigzagInput {
            complexes: vec![x0, x1, x2, x3, x4],
            arrows: vec![
                ZigzagArrow::Forward,
                ZigzagArrow::Forward,
                ZigzagArrow::Forward,
                ZigzagArrow::Backward,
            ],
        };
        let bc = zigzag_persistence(&input).expect("zigzag");

        // Two H1 bars: born-by-creation [2,2] and born-by-deletion [4,4].
        let h1 = bc.bars_in_dim(1);
        assert_eq!(h1.len(), 2, "expected two H1 bars, got {}", h1.len());
        assert!(
            h1.iter().any(|b| b.birth == 2 && b.death == 2),
            "missing H1 bar [2,2] (born by creation, killed by fill)"
        );
        assert!(
            h1.iter().any(|b| b.birth == 4 && b.death == 4),
            "missing H1 bar [4,4] (born by deletion, survives to end)"
        );

        // Betti numbers across the sequence.
        assert_eq!(bc.betti(0, 0), 3, "β0(X0)");
        assert_eq!(bc.betti(1, 0), 2, "β0(X1)");
        assert_eq!(bc.betti(2, 0), 1, "β0(X2)");
        assert_eq!(bc.betti(2, 1), 1, "β1(X2)");
        assert_eq!(bc.betti(3, 1), 0, "β1(X3)");
        assert_eq!(bc.betti(4, 1), 1, "β1(X4)");
    }

    #[test]
    fn single_forward_edge() {
        // X0 = {0},{1}; X1 = + {0,1}.
        let x0 = complex(&[&[0], &[1]]);
        let x1 = complex(&[&[0], &[1], &[0, 1]]);
        let input = ZigzagInput {
            complexes: vec![x0, x1],
            arrows: vec![ZigzagArrow::Forward],
        };
        let bc = zigzag_persistence(&input).expect("zigzag");

        assert_eq!(bc.betti(0, 0), 2, "β0(X0) = 2 components");
        assert_eq!(bc.betti(1, 0), 1, "β0(X1) = 1 component");

        let h0 = bc.bars_in_dim(0);
        assert_eq!(h0.len(), 2, "two H0 bars");
        // One essential (birth 0, death 1) and one finite (birth 0, death 0).
        assert!(
            h0.iter().any(|b| b.birth == 0 && b.death == 1),
            "missing essential H0 bar [0,1]"
        );
        assert!(
            h0.iter().any(|b| b.birth == 0 && b.death == 0),
            "missing finite H0 bar [0,0]"
        );
    }

    #[test]
    fn pure_growth_matches_ordinary_persistence() {
        // An all-forward zigzag that adds exactly ONE simplex per arrow, building a
        // filled triangle.  Complex index == filtration position, so we can compare
        // the (dim, birth_idx, death_idx) multiset against ordinary persistence.
        //
        // Filtration order (one simplex per step):
        //   0:{0} 1:{1} 2:{2} 3:{0,1} 4:{0,2} 5:{1,2} 6:{0,1,2}
        let steps: Vec<&[usize]> = vec![&[0], &[1], &[2], &[0, 1], &[0, 2], &[1, 2], &[0, 1, 2]];

        // Build the zigzag: X_k = first k+1 simplices (cumulative).
        let mut complexes = Vec::new();
        let mut acc: Vec<&[usize]> = Vec::new();
        for st in &steps {
            acc.push(st);
            complexes.push(complex(&acc));
        }
        let arrows = vec![ZigzagArrow::Forward; complexes.len() - 1];
        let input = ZigzagInput { complexes, arrows };
        let bc = zigzag_persistence(&input).expect("zigzag");

        // Ordinary persistence on the same filtration (value == position index).
        let filt_simplices: Vec<FilteredSimplex> = steps
            .iter()
            .enumerate()
            .map(|(i, v)| FilteredSimplex {
                simplex: s(v),
                value: i as f64,
            })
            .collect();
        let filt = Filtration::new(filt_simplices).expect("filtration");
        let mut bm = BoundaryMatrix::from_filtration(&filt).expect("bm");
        reduce_boundary_matrix(&mut bm);
        let pairs = extract_persistence_pairs(&bm, &filt).expect("pairs");

        // Build the ordinary multiset (dim, birth_idx, death_idx).  A *finite* death
        // at ordinary filtration index j corresponds, in the cumulative zigzag, to a
        // death crossing arrow (j−1)→j and is recorded at the LEFT index j−1, so we
        // shift finite deaths down by one.  An *essential* class (death = None) is
        // closed at the last complex index, matching the zigzag finalisation.
        let last = steps.len() - 1;
        let mut ordinary: Vec<(usize, usize, usize)> = pairs
            .iter()
            .map(|p| {
                let birth = p.birth.round() as usize;
                let death = match p.death {
                    Some(d) => (d.round() as usize).saturating_sub(1),
                    None => last,
                };
                (p.dim, birth, death)
            })
            .collect();

        let mut zz: Vec<(usize, usize, usize)> =
            bc.bars.iter().map(|b| (b.dim, b.birth, b.death)).collect();

        ordinary.sort_unstable();
        zz.sort_unstable();
        assert_eq!(
            zz, ordinary,
            "zigzag multiset {zz:?} != ordinary persistence {ordinary:?}"
        );
    }

    #[test]
    fn minimal_born_by_deletion() {
        // X0 = filled triangle; X1 = remove the 2-simplex. The loop is reborn.
        let full = complex(&[&[0], &[1], &[2], &[0, 1], &[0, 2], &[1, 2], &[0, 1, 2]]);
        let without = complex(&[&[0], &[1], &[2], &[0, 1], &[0, 2], &[1, 2]]);
        let input = ZigzagInput {
            complexes: vec![full, without],
            arrows: vec![ZigzagArrow::Backward],
        };
        let bc = zigzag_persistence(&input).expect("zigzag");
        let h1 = bc.bars_in_dim(1);
        assert_eq!(h1.len(), 1, "expected exactly one H1 bar");
        assert_eq!(h1[0].birth, 1, "born by deletion at index 1");
        assert_eq!(h1[0].death, 1, "survives to the last index 1");
        assert_eq!(bc.betti(0, 1), 0, "no loop before deletion");
        assert_eq!(bc.betti(1, 1), 1, "loop present after deletion");
    }

    #[test]
    fn closure_violation_errors() {
        // An edge with a missing vertex.
        let bad = ZigzagComplex {
            simplices: vec![s(&[0]), s(&[0, 1])], // vertex {1} missing
        };
        let input = ZigzagInput {
            complexes: vec![bad],
            arrows: vec![],
        };
        assert!(matches!(
            zigzag_persistence(&input),
            Err(TdaError::ClosureViolation(_))
        ));
    }

    #[test]
    fn bad_inclusion_errors() {
        // Forward arrow declared, but X1 does not contain all of X0.
        let x0 = complex(&[&[0], &[1]]);
        let x1 = complex(&[&[0]]); // missing {1}
        let input = ZigzagInput {
            complexes: vec![x0, x1],
            arrows: vec![ZigzagArrow::Forward],
        };
        assert!(matches!(
            zigzag_persistence(&input),
            Err(TdaError::ClosureViolation(_))
        ));
    }

    #[test]
    fn length_mismatch_errors() {
        let x0 = complex(&[&[0]]);
        let x1 = complex(&[&[0]]);
        // Two complexes but two arrows (should be one).
        let input = ZigzagInput {
            complexes: vec![x0, x1],
            arrows: vec![ZigzagArrow::Forward, ZigzagArrow::Forward],
        };
        assert!(matches!(
            zigzag_persistence(&input),
            Err(TdaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn empty_input_errors() {
        let input = ZigzagInput {
            complexes: vec![],
            arrows: vec![],
        };
        assert!(matches!(
            zigzag_persistence(&input),
            Err(TdaError::EmptyComplex)
        ));
    }

    #[test]
    fn bar_length_and_betti_helpers() {
        let bar = ZigzagBar {
            dim: 1,
            birth: 2,
            death: 5,
        };
        assert_eq!(bar.length(), 4);
        let bc = ZigzagBarcode {
            bars: vec![
                ZigzagBar {
                    dim: 0,
                    birth: 0,
                    death: 3,
                },
                ZigzagBar {
                    dim: 1,
                    birth: 1,
                    death: 1,
                },
            ],
        };
        assert_eq!(bc.betti(0, 0), 1);
        assert_eq!(bc.betti(1, 1), 1);
        assert_eq!(bc.betti(2, 1), 0);
        assert_eq!(bc.bars_in_dim(0).len(), 1);
    }
}
