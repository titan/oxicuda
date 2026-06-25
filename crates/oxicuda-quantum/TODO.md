# oxicuda-quantum TODO

GPU-accelerated quantum simulation and Quantum Machine Learning (QML) primitives,
serving as a pure-Rust complement to NVIDIA cuQuantum / Qiskit-Aer style backends.
Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.38).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** ~21,300 total lines (87 files)
- **Coverage:** state-vector simulation (dense + sparse + FP16/BF16-packed),
  standard + parametric gates, Pauli/Hamiltonian expectation values, Trotter-Suzuki
  1/2/4-order evolution (cross-validated vs exact `expm`), Lindblad master equation
  (first-order Euler + RK4) and Monte-Carlo wave-function (MCWF) quantum-trajectory
  unraveling, VQE with parameter-shift / SPSA / natural-gradient / layer-wise
  warm-start optimizers, QAOA Max-Cut/Ising (+ warm-start), density matrices with
  partial-trace + quantum-information metrics, Kraus channels (noise models),
  feature-map embeddings (angle/amplitude/ZZ), overlap + projected (PQK) + trainable
  quantum kernels for QML, QCNN and qGAN generative models, stabilizer/Clifford and
  MPS and tensor-network-contraction back-ends, QFT/QPE, mid-circuit measurement,
  surface-code MWPM decoder, high-level `QuantumCircuit` DSL, and PTX kernel-string
  generation for 6 SM tiers.

### Completed

#### Core Infrastructure
- [x] error.rs — `QuantumError`, `QuantumResult<T>`
- [x] handle.rs — `LcgRng` deterministic PRNG, `SmVersion` PTX target descriptor

#### State Vector & Gate Application (statevec/)
- [x] state.rs — `StateVector`, `new_zero_state`, `norm_sq`, complex `amps` storage
- [x] apply_1q.rs — `apply_1q_inplace` bit-mask in-place single-qubit application
- [x] apply_2q.rs — `apply_2q_inplace` 4×4 complex matmul, `apply_1q_controlled`

#### Quantum Gates (gates/)
- [x] pauli.rs — `gate_{i,x,y,z}` Pauli matrices
- [x] hadamard.rs — `gate_h`, `gate_{s,t,sdg,tdg}` phase/T family
- [x] controlled.rs — `apply_{cnot,cz,swap,ccx}` multi-qubit controlled forms
- [x] parametric.rs — `gate_{rx,ry,rz,u3,phase}` parametric rotations

#### Pauli Operators & Hamiltonians (pauli/)
- [x] pauli_string.rs — `PauliOp`, `PauliString` (tensor-product strings)
- [x] hamiltonian.rs — `Hamiltonian { Σ_k coeff_k · P_k }`, term assembly
- [x] expval.rs — `expectation_value` via basis-rotation + parity counting

#### Time Evolution (trotter/)
- [x] trotter.rs — `TrotterStep` 1st / 2nd / 4th-order Suzuki-Yoshida product formulas
- [x] lindblad.rs — `LindbladOp`, `lindblad_step` density-matrix master-equation step

#### Variational & Hybrid Algorithms
- [x] vqe/ansatz.rs — `HardwareEfficientAnsatz` parametric Ry/Rz + CNOT layers
- [x] vqe/vqe.rs — `VqeOptimizer` parameter-shift gradient descent
- [x] qaoa/qaoa.rs — `QaoaCircuit` p-layer cost+mixer Max-Cut energy evaluation

#### Density Matrices & Channels (density/, channel/)
- [x] density/density.rs — `DensityMatrix`, `from_pure_state`, `trace`
- [x] density/partial_trace.rs — index-folding partial trace over subsystems
- [x] density/metrics.rs — `purity`, `fidelity`, `von_neumann_entropy`
- [x] channel/kraus.rs — `KrausChannel`, completeness-checked `apply`
- [x] channel/noise.rs — `depolarizing_channel`, `amplitude_damping_channel`,
  `phase_damping_channel`

#### Feature Maps & Quantum Kernels (embedding/, kernel/)
- [x] embedding/angle.rs — `angle_embedding` Ry-rotation encoding
- [x] embedding/amplitude.rs — `amplitude_embedding` normalized amplitude loading
- [x] embedding/zz_feature.rs — `zz_feature_map` Havlíček depth-2 ZZ entangler
- [x] kernel/quantum_kernel.rs — `overlap_kernel` K(x,y)=|⟨ψ(x)|ψ(y)⟩|²,
  `kernel_matrix` for QML

#### High-Level Circuit DSL (circuit/)
- [x] circuit/circuit.rs — `QuantumCircuit`, `GateOp` enum, `exec_on_state`

#### PTX Kernel Generation (ptx_kernels.rs)
- [x] 8 kernel string generators × 6 SM versions (sm_75/80/86/89/90/100):
  `statevec_apply_1q`, `statevec_apply_2q`, `statevec_apply_cnot`,
  `expval_pauli`, `partial_trace`, `trotter_step`, `measure_prob`,
  `qft_butterfly`

#### Tests & Benchmarks
- [x] 17 end-to-end tests in `lib.rs::e2e_tests` (norm, superposition, Bell state,
  Pauli-Z eigenvalues, mixed Hamiltonian, VQE convergence, QAOA round-trip,
  pure-state purity, depolarizing reduces purity, amplitude damping trace,
  circuit-DSL Bell, PTX non-empty × all SM versions, QFT uniform, QFT round-trip
  identity, QPE recovers φ=1/4)
- [x] Benchmarks (`benches/quantum_ops.rs`) — 7 PTX kernel groups × 4 SM versions
  + 5 algorithm benches (H, Bell, ZZ feature map, VQE energy, QAOA)
- **Tests:** 482 passing

### Future Enhancements

#### P0 — Hardware Verification
- [ ] All 7 PTX kernels validated on actual NVIDIA hardware (currently PTX-string
  generation tested only; runtime launch path pending Linux+NVIDIA host)
- [ ] State-vector benchmarks measured on real GPU (currently CPU-side PTX
  emission benches only)

#### P1 — Algorithm Coverage Extensions
- [x] Clifford+T circuit decomposition pass (rotation → discrete gate-set synthesis)
- [x] Stabilizer formalism back-end for Clifford-only circuits (poly-time simulation) (stabilizer/tableau.rs -- Aaronson-Gottesman 2004 CHP tableau; H/S/CNOT/measure with rowsum phase bookkeeping, poly-time Clifford simulation)
- [x] Matrix Product State (MPS) simulator for low-entanglement circuits (mps/simulator.rs -- site tensors + adjacent 2q SVD truncation to χ + self-contained complex SVD; full-χ == statevector)
- [x] Tensor-network contraction back-end for shallow wide circuits (tensor/contraction.rs -- `Tensor` with named legs + general pairwise `tensordot` contraction, greedy `TensorNetwork` contractor, `amplitude()` evaluates ⟨bitstring|U|0…0⟩ via wire-graph contraction; exact match vs statevector on Bell + random deep circuits, unitarity prob-sum verified)
- [x] Open-system quantum trajectories (Monte-Carlo unraveling of Lindblad) (trotter/trajectory.rs -- `QuantumTrajectory`/`TrajectoryConfig` MCWF: effective non-Hermitian H_eff = H - (i/2)Σγ_k L_k†L_k propagation + stochastic quantum jumps weighted by δp_k; ensemble-averaged ρ reproduces master equation; trace preserved, dephasing purity-loss verified)
- [x] Mid-circuit measurement + classical feed-forward conditional gates (midcircuit/measurement.rs -- statevector measure+collapse+renormalize, classical register, predicate-conditioned gates, run executor)
- [x] Quantum Fourier Transform (QFT) + inverse (fourier/qft.rs -- little-endian H + controlled-phase R_k ladder + bit-reversal; DFT-matrix verified)

#### P1 — Variational Algorithms
- [x] SPSA optimizer for VQE (stochastic perturbation, fewer evaluations than
  parameter-shift)
- [x] Adam / RMSProp optimizers for variational parameter updates (vqe/adam.rs -- standard Adam (β1,β2,ε bias-corrected moments) + RMSProp (decay·v+(1-decay)·g²) optimizers with reset_state)
- [x] Natural-gradient VQE via quantum Fisher information matrix (vqe/qfim.rs -- Stokes 2020; Fubini-Study metric via finite-diff statevector derivatives + (F+reg·I)δ=grad natural-gradient solve; reuses HardwareEfficientAnsatz + StateVector)
- [x] Layer-wise warm-start initialization to mitigate barren plateaus (vqe/layerwise.rs -- Grant 2019: grows HardwareEfficientAnsatz depth 0→target one layer at a time, zero-initializing each new RY layer (identity block) so each stage starts at the previous optimum; parameter-shift GD per stage; monotone non-increasing stage energies + single-Z ground-state recovery verified)
- [x] QAOA warm-start from classical GW / Goemans-Williamson relaxations (qaoa/warm_start.rs -- Egger 2021; projected-gradient continuous MaxCut relaxation c∈[0,1]ⁿ + θ_i=2·arcsin(√c_i) Ry-init angle mapping + zero-init (γ,β))
- [x] Quantum Phase Estimation (QPE) primitive (fourier/qpe.rs -- Hadamard counting register + controlled-U^{2^k} repeated-squaring ladder + inverse-QFT readout; little-endian; recovers φ=j/2^n exactly for phase-gate / Rz eigenstates)

#### P2 — QML Extensions
- [x] Surface code logical qubit (`error_correction/surface_code.rs`) — Fowler 2012: distance-d rotated surface code; ALREADY EXISTS (863-line real impl): d²−1 checkerboard stabilizers, syndrome extraction (parity of stabilizer-support ∩ error), MWPM decoder via exact bitmask DP over the defect set (same minimum-weight matching as Blossom-V), corrects all weight-≤⌊(d−1)/2⌋ errors; `SurfaceCode`/`SurfaceCodeConfig`/`Stabilizer`/`PauliError`/`Syndrome`
- [x] Full Lindblad RK4 integrator (`trotter/lindblad_rk4.rs`) — Breuer-Petruccione 2002: classical RK4 (k1+2k2+2k3+k4)/6 of the full Lindblad superoperator L[ρ] for arbitrary Pauli Hamiltonian + collapse operators (O(dt⁵) local error, distinct from first-order `lindblad_step`); `LindbladRk4`; verified vs closed-form dephasing ρ01(t)=½e^{-2γt} and shown more accurate than Euler at equal dt
- [x] Projected quantum kernels (PQK) beyond overlap fidelity (kernel/projected.rs -- Huang 2021: per-qubit single-qubit-RDM Bloch features (⟨X⟩,⟨Y⟩,⟨Z⟩) + RBF kernel over them, avoiding fidelity concentration; PSD Gram matrix verified; angle + ZZ embeddings; `ProjectedKernelConfig`/`projected_kernel`/`projected_kernel_matrix`)
- [x] Trainable quantum kernels with gradient descent over feature-map params (kernel/trainable.rs -- Hubregtsen 2022: per-feature trainable rotation scales `θ` in a hardware-efficient embedding, kernel-target alignment objective maximized by parameter-shift gradient ascent; `TrainableKernel`/`TrainableKernelConfig`)
- [x] Quantum convolutional neural networks (QCNN) translation-invariant layers (qml/qcnn.rs -- Cong-Choi-Lukin 2019: weight-shared parametrized 2q conv blocks over a brick-wall + pooling layers halving the active-qubit set n→n/2→…→1, ⟨Z⟩ readout classifier, parameter-shift training; `Qcnn`; loss-reduction on separable data verified)
- [x] Quantum GAN / Variational Quantum Eigensolver-Inspired Generative models (qml/qgan.rs -- Zoufal 2019 + Liu-Wang Born-machine: parametrized generator state, exact basis-measurement distribution fit to a target via Gaussian-mixture MMD² (proper IPM, MMD=0⟺p=q) with parameter-shift gradient descent; `QuantumGenerator`; learns delta + bimodal targets)

#### P2 — Performance & Memory
- [ ] GPU shared-memory amplitude caching for high-locality gate sequences (requires GPU hardware)
- [x] Sparse state-vector representation for low-occupancy circuits (statevec/sparse.rs -- `SparseStateVector` stores only nonzero basis amplitudes in a HashMap, single-qubit + CNOT/CCX gates operate on the support with auto-prune; permutation circuits keep occupancy O(1); exact dense round-trip + Bell-vs-dense verified)
- [ ] Multi-GPU state-vector partitioning via amplitude-index sharding (requires GPU hardware)
- [x] FP16/BF16 amplitude storage for memory-bound large-qubit simulation (statevec/fp16.rs -- pure-Rust dependency-free IEEE-754 binary16 (subnormals/overflow/round-to-nearest-even) + bfloat16 pack/unpack, `HalfStateVector`/`HalfFormat` halving amplitude footprint; known bit-patterns 1.0→0x3c00/0x3f80, overflow-to-inf, norm-preserving round-trip verified)

## Dependencies

| Dependency | Purpose | Pure Rust? |
|------------|---------|------------|
| thiserror | Error derive macros | Yes |
| num-complex | Complex amplitude arithmetic | Yes |
| criterion (dev) | Benchmark harness | Yes |

No CUDA-SDK / nvcc / cuQuantum dependency — PTX kernels are emitted as strings.
No oxicuda-driver / -memory / -launch dependency at this layer — runtime wiring
is delegated to higher-level integrators.

## Quality Status

- Warnings: 0 (clippy clean, workspace lints inherited)
- Tests: 482 passing (state vector, gates, Pauli, Trotter, VQE, QAOA, density,
  channels, kernels, circuit, stabilizer, MPS, mid-circuit, SPSA, QFIM/natural
  gradient, QFT/QPE Fourier, PTX × 6 SM, MCWF quantum trajectories, Lindblad RK4,
  tensor-network contraction, layer-wise warm-start VQE, projected + trainable
  quantum kernels, QCNN, qGAN, sparse + FP16/BF16 state vectors, Trotter-vs-expm
  accuracy + trajectory-vs-master-equation + Gram-PSD cross-checks)
- unwrap() calls: 0 in production code
- macOS: compiles but returns `UnsupportedPlatform` at runtime when actual launch
  is attempted (PTX emission still works on every host)
- Refactoring policy: every source file is well under 2,000 lines

## Performance Targets

| Workload | Target |
|----------|--------|
| State-vector single-qubit gate, n=20 qubits | ≥ 90% of cuQuantum throughput |
| State-vector CNOT, n=20 qubits | ≥ 85% of cuQuantum throughput |
| Pauli expectation value (k-local terms) | ≥ 80% of cuQuantum throughput |
| VQE energy evaluation (4-qubit, depth-2) | ≥ 85% of reference |

Performance harnesses are CPU-side today; GPU-side numbers will be filled in once
the Linux+NVIDIA verification run is executed.

## Benchmark Coverage

- [x] Criterion benchmarks (`benches/quantum_ops.rs`) — 7 PTX kernel groups × 4 SM
  + 5 algorithm benches (H gate, Bell state, ZZ feature map, VQE energy, QAOA)

---

## Notes

- All amplitudes are `num_complex::Complex32` (FP32). FP64 is a future option.
- The `LcgRng` is intentionally a simple linear-congruential generator — reproducible
  but not cryptographic; used for shot sampling and stochastic initialization.
- Gate-matrix conventions match Qiskit / pyquil (little-endian qubit ordering).
- Trotter-Suzuki 4th order uses the Yoshida triple-product coefficient
  `s = 1/(2 - 2^(1/3))`.

---

## Architecture-Specific Deepening Opportunities

### Turing (sm_75)
- [x] PTX target string emitted (`statevec_apply_1q`, ..., `measure_prob`)
- [ ] WMMA m16n16k16 paths for batched gate-tile application

### Ampere (sm_80) / Ada (sm_89)
- [x] PTX target string emitted
- [ ] `cp.async` global→shared prefetch for amplitude pairs
- [ ] Shared-memory bank-conflict-free amplitude tile layout

### Hopper (sm_90)
- [x] PTX target string emitted
- [ ] TMA-based amplitude bulk transfer for n≥24 qubit states
- [ ] WGMMA-based dense gate-tile application
- [ ] Warp-specialized producer/consumer pipeline for Trotter steps

### Blackwell (sm_100)
- [x] PTX target string emitted
- [ ] Native FP4/FP6 amplitude storage exploration for very large state vectors

---

## Deepening Opportunities

> Items marked `[x]` above represent API surface coverage and PTX-string generation.
> These items represent the gap between the current depth and full
> production-grade GPU quantum simulation.

### Verification Gaps
- [ ] Bell-state circuit verified on real GPU (currently CPU-only)
- [ ] VQE 4-qubit H₂-style toy Hamiltonian converges on GPU
- [ ] Density-matrix Lindblad evolution numerical agreement with QuTiP reference (requires external QuTiP/Python reference; note: RK4 Lindblad evolution is already cross-validated against a closed-form analytic dephasing solution in trotter/lindblad_rk4.rs)

### Implementation Deepening
- [x] SPSA / natural-gradient VQE optimizers (vqe/spsa.rs SPSA + 2SPSA Hessian; vqe/qfim.rs Fubini-Study natural gradient)
- [x] Stabilizer / Clifford fast-track simulator (stabilizer/tableau.rs Aaronson-Gottesman CHP tableau; circuit/clifford_t.rs Clifford+T decomposition)
- [x] MPS back-end for low-entanglement regimes (mps/simulator.rs site tensors + adjacent 2q SVD truncation)
- [x] Tensor-network contraction back-end for low-entanglement regimes (tensor/contraction.rs -- see P1; exact amplitude evaluation via wire-graph `tensordot`)
- [x] Mid-circuit measurement + classical conditional control (midcircuit/measurement.rs measure+collapse+renormalize + predicate-conditioned gates)

### Numerical Accuracy
- [x] Trotter-error analysis vs. exact `expm` for XX-Ising (trotter/trotter.rs::accuracy_tests -- dense Taylor-series matrix-exponential reference `expm`; Trotter error decreases with step count and 4th-order ≪ 2nd ≪ 1st for the transverse-field XX-Ising propagator)
- [x] Lindblad-trajectory ensemble vs. master-equation cross-check (trotter/trajectory.rs::tests::trajectory_matches_rk4_master_equation -- MCWF ensemble-averaged ρ agrees with the RK4 Lindblad master-equation ρ for a dephased qubit within Monte-Carlo tolerance)
- [x] Quantum-kernel matrix positive-semidefiniteness verified for Gram matrices (kernel/quantum_kernel.rs::tests::gram_matrix_is_psd -- overlap-kernel Gram matrix has all-non-negative quadratic forms vᵀKv ≥ 0; PQK Gram PSD also verified in kernel/projected.rs)

## Performance Verification Harness Status (2026-05-16)

- **State-vector kernels:** harnesses at `benches/quantum_ops.rs::quantum_ptx`;
  CPU-side PTX-emission timings landed, GPU launch path awaiting Linux+NVIDIA run.
- **Algorithm benches:** `bench_statevec` (H gate, Bell, ZZ map, VQE, QAOA)
  exercised on CPU; GPU-side throughput numbers pending.
