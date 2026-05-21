# oxicuda-quantum TODO

GPU-accelerated quantum simulation and Quantum Machine Learning (QML) primitives,
serving as a pure-Rust complement to NVIDIA cuQuantum / Qiskit-Aer style backends.
Part of [OxiCUDA](https://github.com/cool-japan/oxicuda) (Vol.38).

(C) 2026 COOLJAPAN OU (Team KitaSan) -- Pure Rust, no C/Fortran, no CUDA SDK, no nvcc.

## Implementation Status

- **Actual SLoC:** 3,611 total lines (2,983 code, 40 files)
- **Coverage:** state-vector simulation, standard + parametric gates, Pauli/Hamiltonian
  expectation values, Trotter-Suzuki 1/2/4-order evolution, Lindblad master equation,
  VQE with parameter-shift gradients, QAOA Max-Cut/Ising, density matrices with
  partial-trace + quantum-information metrics, Kraus channels (noise models),
  feature-map embeddings (angle/amplitude/ZZ), overlap quantum kernel for QML,
  high-level `QuantumCircuit` DSL, and PTX kernel-string generation for 6 SM tiers.

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
- [x] 7 kernel string generators × 6 SM versions (sm_75/80/86/89/90/100):
  `statevec_apply_1q`, `statevec_apply_2q`, `statevec_apply_cnot`,
  `expval_pauli`, `partial_trace`, `trotter_step`, `measure_prob`

#### Tests & Benchmarks
- [x] 14 end-to-end tests in `lib.rs::e2e_tests` (norm, superposition, Bell state,
  Pauli-Z eigenvalues, mixed Hamiltonian, VQE convergence, QAOA round-trip,
  pure-state purity, depolarizing reduces purity, amplitude damping trace,
  circuit-DSL Bell, PTX non-empty × all SM versions)
- [x] Benchmarks (`benches/quantum_ops.rs`) — 7 PTX kernel groups × 4 SM versions
  + 5 algorithm benches (H, Bell, ZZ feature map, VQE energy, QAOA)
- **Tests:** 61 passing

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
- [ ] Tensor-network contraction back-end for shallow wide circuits
- [ ] Open-system quantum trajectories (Monte-Carlo unraveling of Lindblad)
- [x] Mid-circuit measurement + classical feed-forward conditional gates (midcircuit/measurement.rs -- statevector measure+collapse+renormalize, classical register, predicate-conditioned gates, run executor)

#### P1 — Variational Algorithms
- [x] SPSA optimizer for VQE (stochastic perturbation, fewer evaluations than
  parameter-shift)
- [x] Adam / RMSProp optimizers for variational parameter updates (vqe/adam.rs -- standard Adam (β1,β2,ε bias-corrected moments) + RMSProp (decay·v+(1-decay)·g²) optimizers with reset_state)
- [x] Natural-gradient VQE via quantum Fisher information matrix (vqe/qfim.rs -- Stokes 2020; Fubini-Study metric via finite-diff statevector derivatives + (F+reg·I)δ=grad natural-gradient solve; reuses HardwareEfficientAnsatz + StateVector)
- [ ] Layer-wise warm-start initialization to mitigate barren plateaus
- [x] QAOA warm-start from classical GW / Goemans-Williamson relaxations (qaoa/warm_start.rs -- Egger 2021; projected-gradient continuous MaxCut relaxation c∈[0,1]ⁿ + θ_i=2·arcsin(√c_i) Ry-init angle mapping + zero-init (γ,β))
- [ ] Quantum Phase Estimation (QPE) primitive

#### P2 — QML Extensions
- [ ] Projected quantum kernels (PQK) beyond overlap fidelity
- [ ] Trainable quantum kernels with gradient descent over feature-map params
- [ ] Quantum convolutional neural networks (QCNN) translation-invariant layers
- [ ] Quantum GAN / Variational Quantum Eigensolver-Inspired Generative models

#### P2 — Performance & Memory
- [ ] GPU shared-memory amplitude caching for high-locality gate sequences
- [ ] Sparse state-vector representation for low-occupancy circuits
- [ ] Multi-GPU state-vector partitioning via amplitude-index sharding
- [ ] FP16/BF16 amplitude storage for memory-bound large-qubit simulation

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
- Tests: 61 passing (state vector, gates, Pauli, Trotter, VQE, QAOA, density,
  channels, kernels, circuit, PTX × 6 SM)
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
- [ ] Density-matrix Lindblad evolution numerical agreement with QuTiP reference

### Implementation Deepening
- [ ] SPSA / natural-gradient VQE optimizers
- [ ] Stabilizer / Clifford fast-track simulator
- [ ] MPS / tensor-network back-ends for low-entanglement regimes
- [ ] Mid-circuit measurement + classical conditional control

### Numerical Accuracy
- [ ] Trotter-error analysis vs. exact `expm` for 4-qubit XX-Ising
- [ ] Lindblad-trajectory ensemble vs. master-equation cross-check
- [ ] Quantum-kernel matrix positive-semidefiniteness verified for Gram matrices

## Performance Verification Harness Status (2026-05-16)

- **State-vector kernels:** harnesses at `benches/quantum_ops.rs::quantum_ptx`;
  CPU-side PTX-emission timings landed, GPU launch path awaiting Linux+NVIDIA run.
- **Algorithm benches:** `bench_statevec` (H gate, Bell, ZZ map, VQE, QAOA)
  exercised on CPU; GPU-side throughput numbers pending.
