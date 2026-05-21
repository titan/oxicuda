//! Quantum simulation primitives for OxiCUDA.
//!
//! Provides state-vector simulation, Pauli expectation values, VQE/QAOA,
//! density matrices, Trotter-Suzuki time evolution, quantum kernels, and
//! PTX kernel code generation for GPU execution.

pub mod channel;
pub mod circuit;
pub mod density;
pub mod embedding;
pub mod error;
pub mod gates;
pub mod handle;
pub mod kernel;
pub mod midcircuit;
pub mod mps;
pub mod pauli;
pub mod ptx_kernels;
pub mod qaoa;
pub mod stabilizer;
pub mod statevec;
pub mod trotter;
pub mod vqe;

pub use error::{QuantumError, QuantumResult};
pub use midcircuit::{
    ClassicalRegister, MidCircuitOp, apply_if, measure_and_collapse, measure_deterministic,
    run as run_midcircuit,
};
pub use mps::{MatrixProductState, MpsConfig};

#[cfg(test)]
mod e2e_tests {
    use crate::channel::noise::{amplitude_damping_channel, depolarizing_channel};
    use crate::circuit::circuit::{GateOp, QuantumCircuit};
    use crate::density::density::DensityMatrix;
    use crate::density::metrics::purity;
    use crate::gates::controlled::apply_cnot;
    use crate::gates::hadamard::gate_h;
    use crate::gates::pauli::gate_x;
    use crate::handle::LcgRng;
    use crate::pauli::expval::expectation_value;
    use crate::pauli::hamiltonian::Hamiltonian;
    use crate::pauli::pauli_string::PauliOp;
    use crate::ptx_kernels::{
        expval_pauli_ptx, measure_prob_ptx, partial_trace_ptx, statevec_apply_1q_ptx,
        statevec_apply_2q_ptx, statevec_apply_cnot_ptx, trotter_step_ptx,
    };
    use crate::qaoa::qaoa::QaoaCircuit;
    use crate::statevec::apply_1q::apply_1q_inplace;
    use crate::statevec::state::StateVector;
    use crate::vqe::ansatz::HardwareEfficientAnsatz;
    use crate::vqe::vqe::VqeOptimizer;

    // Test 1: |0⟩ state has norm 1
    #[test]
    fn e2e_01_zero_state_norm_is_one() {
        let sv = StateVector::new_zero_state(3).unwrap();
        let norm = sv.norm_sq();
        assert!((norm - 1.0).abs() < 1e-6, "norm={norm}");
    }

    // Test 2: H gate on |0⟩ gives equal superposition
    #[test]
    fn e2e_02_hadamard_creates_superposition() {
        let mut sv = StateVector::new_zero_state(1).unwrap();
        apply_1q_inplace(&mut sv, 0, &gate_h()).unwrap();
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (sv.amps[0].re - inv_sqrt2).abs() < 1e-5,
            "amp0={:?}",
            sv.amps[0]
        );
        assert!(
            (sv.amps[1].re - inv_sqrt2).abs() < 1e-5,
            "amp1={:?}",
            sv.amps[1]
        );
    }

    // Test 3: H·H = I (round-trip)
    #[test]
    fn e2e_03_hadamard_round_trip() {
        let mut sv = StateVector::new_zero_state(1).unwrap();
        apply_1q_inplace(&mut sv, 0, &gate_h()).unwrap();
        apply_1q_inplace(&mut sv, 0, &gate_h()).unwrap();
        assert!((sv.amps[0].re - 1.0).abs() < 1e-5, "amp0={:?}", sv.amps[0]);
        assert!(sv.amps[1].norm() < 1e-5, "amp1={:?}", sv.amps[1]);
    }

    // Test 4: CNOT creates Bell state |00⟩+|11⟩/√2
    #[test]
    fn e2e_04_cnot_creates_bell_state() {
        let mut sv = StateVector::new_zero_state(2).unwrap();
        apply_1q_inplace(&mut sv, 0, &gate_h()).unwrap();
        apply_cnot(&mut sv, 0, 1).unwrap();
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        // |00⟩ at index 0, |11⟩ at index 3
        assert!((sv.amps[0].re - inv_sqrt2).abs() < 1e-5);
        assert!((sv.amps[3].re - inv_sqrt2).abs() < 1e-5);
        assert!(sv.amps[1].norm() < 1e-5);
        assert!(sv.amps[2].norm() < 1e-5);
    }

    // Test 5: Pauli-Z expectation of |0⟩ = +1
    #[test]
    fn e2e_05_pauli_z_expval_zero_state() {
        let sv = StateVector::new_zero_state(1).unwrap();
        let mut ham = Hamiltonian::new();
        ham.add_term(1.0, vec![PauliOp::Z]);
        let ev = expectation_value(&sv, &ham).unwrap();
        assert!((ev - 1.0).abs() < 1e-5, "ev={ev}");
    }

    // Test 6: Pauli-Z expectation of |1⟩ = -1
    #[test]
    fn e2e_06_pauli_z_expval_one_state() {
        let mut sv = StateVector::new_zero_state(1).unwrap();
        apply_1q_inplace(&mut sv, 0, &gate_x()).unwrap(); // flip to |1⟩
        let mut ham = Hamiltonian::new();
        ham.add_term(1.0, vec![PauliOp::Z]);
        let ev = expectation_value(&sv, &ham).unwrap();
        assert!((ev - (-1.0)).abs() < 1e-5, "ev={ev}");
    }

    // Test 7: Mixed Hamiltonian expectation value is finite
    #[test]
    fn e2e_07_mixed_hamiltonian_expval_finite() {
        let mut sv = StateVector::new_zero_state(2).unwrap();
        apply_1q_inplace(&mut sv, 0, &gate_h()).unwrap();
        apply_cnot(&mut sv, 0, 1).unwrap();
        let mut ham = Hamiltonian::new();
        ham.add_term(0.5, vec![PauliOp::Z, PauliOp::Z]);
        ham.add_term(0.3, vec![PauliOp::X, PauliOp::I]);
        ham.add_term(0.2, vec![PauliOp::I, PauliOp::Y]);
        let ev = expectation_value(&sv, &ham).unwrap();
        assert!(ev.is_finite(), "ev={ev}");
    }

    // Test 8: VQE energy decreases over iterations
    #[test]
    fn e2e_08_vqe_energy_decreases() {
        let ans = HardwareEfficientAnsatz::new(2, 1);
        let mut ham = Hamiltonian::new();
        ham.add_term(1.0, vec![PauliOp::Z, PauliOp::Z]);
        ham.add_term(-0.5, vec![PauliOp::X, PauliOp::I]);
        let mut rng = LcgRng::new(42);
        let mut opt = VqeOptimizer::new(ans, ham, &mut rng);
        let e_initial = opt.energy(&opt.params.clone()).unwrap();
        let (e_final, _) = opt.optimize(5, 0.1).unwrap();
        assert!(
            e_final <= e_initial + 1e-3,
            "e_initial={e_initial}, e_final={e_final}"
        );
    }

    // Test 9: QAOA circuit runs without error
    #[test]
    fn e2e_09_qaoa_runs_without_error() {
        let circuit = QaoaCircuit::new(4, 2, vec![0.3, 0.5], vec![0.7, 0.2]).unwrap();
        let graph = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        let sv = circuit.run(&graph).unwrap();
        let norm = sv.norm_sq();
        assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
    }

    // Test 10: Density matrix from pure state has purity 1
    #[test]
    fn e2e_10_pure_state_purity_is_one() {
        let sv = StateVector::new_zero_state(2).unwrap();
        let dm = DensityMatrix::from_pure_state(&sv);
        let p = purity(&dm);
        assert!((p - 1.0).abs() < 1e-5, "purity={p}");
    }

    // Test 11: Depolarizing channel reduces purity
    #[test]
    fn e2e_11_depolarizing_channel_reduces_purity() {
        let sv = StateVector::new_zero_state(1).unwrap();
        let dm = DensityMatrix::from_pure_state(&sv);
        let p_initial = purity(&dm);

        let ch = depolarizing_channel(0.5, 2).unwrap();
        let dm_out = ch.apply(&dm).unwrap();
        let p_final = purity(&dm_out);

        assert!(
            p_final < p_initial,
            "purity did not decrease: {p_initial} → {p_final}"
        );
    }

    // Test 12: PTX kernels generate non-empty strings for all 6 SM versions
    #[test]
    fn e2e_12_ptx_kernels_non_empty_all_sm() {
        for sm in [75u32, 80, 86, 89, 90, 100] {
            let s1 = statevec_apply_1q_ptx(sm);
            let s2 = statevec_apply_2q_ptx(sm);
            let s3 = statevec_apply_cnot_ptx(sm);
            let s4 = expval_pauli_ptx(sm);
            let s5 = partial_trace_ptx(sm);
            let s6 = trotter_step_ptx(sm);
            let s7 = measure_prob_ptx(sm);
            assert!(!s1.is_empty(), "statevec_apply_1q sm={sm}");
            assert!(!s2.is_empty(), "statevec_apply_2q sm={sm}");
            assert!(!s3.is_empty(), "statevec_apply_cnot sm={sm}");
            assert!(!s4.is_empty(), "expval_pauli sm={sm}");
            assert!(!s5.is_empty(), "partial_trace sm={sm}");
            assert!(!s6.is_empty(), "trotter_step sm={sm}");
            assert!(!s7.is_empty(), "measure_prob sm={sm}");
        }
    }

    // Supplementary: amplitude damping channel test
    #[test]
    fn e2e_13_amplitude_damping_channel() {
        let sv = StateVector::new_zero_state(1).unwrap();
        let dm = DensityMatrix::from_pure_state(&sv);
        let ch = amplitude_damping_channel(0.3).unwrap();
        let dm_out = ch.apply(&dm).unwrap();
        let tr = dm_out.trace();
        assert!((tr.re - 1.0).abs() < 1e-5, "trace={}", tr.re);
    }

    // Supplementary: circuit with explicit CNOTs
    #[test]
    fn e2e_14_circuit_bell_state() {
        let mut circ = QuantumCircuit::new(2);
        circ.add_gate(GateOp::H);
        circ.add_gate(GateOp::Cnot { ctrl: 0, tgt: 1 });
        let sv = StateVector::new_zero_state(2).unwrap();
        let mut rng = LcgRng::new(7);
        let out = circ.exec_on_state(&sv, &mut rng).unwrap();
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        assert!((out.amps[0].re - inv_sqrt2).abs() < 1e-5);
        assert!((out.amps[3].re - inv_sqrt2).abs() < 1e-5);
    }
}
