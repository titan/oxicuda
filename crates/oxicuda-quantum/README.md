# oxicuda-quantum

Quantum simulation primitives for OxiCUDA — state-vector simulation, Pauli expectation values, VQE/QAOA, density matrices, Trotter-Suzuki, quantum kernels.

Part of the [OxiCUDA](https://github.com/cool-japan/oxicuda) ecosystem — Pure Rust CUDA replacement.

## Features

- **State-vector simulation**: Arbitrary-qubit state vectors with single- and two-qubit gate application; Hadamard, Pauli-X/Y/Z, CNOT, and custom gates
- **Quantum algorithms**: VQE (Variational Quantum Eigensolver) with hardware-efficient ansatz and gradient-descent optimizer; QAOA (Quantum Approximate Optimization Algorithm) for combinatorial problems
- **Open quantum systems**: Density matrix representation, purity computation, depolarizing and amplitude-damping noise channels, Trotter-Suzuki time evolution
- **Hamiltonians and expectation values**: Pauli string Hamiltonians with real coefficients and exact expectation value evaluation
- **Quantum circuits**: High-level circuit API supporting gate sequences and execution on state vectors
- **PTX kernels**: 7 GPU kernels (statevec 1q/2q/CNOT apply, Pauli expval, partial trace, Trotter step, measurement probability) × 6 SM versions

## Usage

```rust
use oxicuda_quantum::{
    statevec::{apply_1q::apply_1q_inplace, state::StateVector},
    gates::{hadamard::gate_h, controlled::apply_cnot},
    pauli::{hamiltonian::Hamiltonian, expval::expectation_value, pauli_string::PauliOp},
};

// Create a 2-qubit Bell state: H on qubit 0, then CNOT(0,1)
let mut sv = StateVector::new_zero_state(2).unwrap();
apply_1q_inplace(&mut sv, 0, &gate_h()).unwrap();
apply_cnot(&mut sv, 0, 1).unwrap();

// Measure <ZZ> expectation
let mut ham = Hamiltonian::new();
ham.add_term(1.0, vec![PauliOp::Z, PauliOp::Z]);
let ev = expectation_value(&sv, &ham).unwrap();
println!("<ZZ> on Bell state: {ev}"); // should be 1.0
```

## Documentation

- [API Documentation](https://docs.rs/oxicuda-quantum)
- [OxiCUDA Project](https://github.com/cool-japan/oxicuda)

## License

Apache-2.0 — Copyright 2026 COOLJAPAN OU (Team Kitasan)
