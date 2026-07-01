//! Quantum simulation primitives for OxiCUDA.
//!
//! Provides state-vector simulation, Pauli expectation values, VQE/QAOA,
//! density matrices, Trotter-Suzuki time evolution, quantum kernels, and
//! PTX kernel code generation for GPU execution.

pub mod algorithm;
pub mod algorithms;
pub mod channel;
pub mod circuit;
pub mod density;
pub mod embedding;
pub mod error;
pub mod error_correction;
pub mod fourier;
pub mod gates;
pub mod handle;
pub mod kernel;
pub mod linear_systems;
pub mod midcircuit;
pub mod mps;
pub mod pauli;
pub mod ptx_kernels;
pub mod qaoa;
pub mod qml;
pub mod stabilizer;
pub mod statevec;
pub mod tensor;
pub mod trotter;
pub mod vqe;

pub use algorithm::{
    BernsteinVaziraniResult, DeutschJozsaResult, FunctionKind, GroverResult, OrderFindingResult,
    ShorResult, SimonResult, SuperdenseResult, TeleportResult, bernstein_vazirani,
    bit_flip_correct, bit_flip_encode, classical_order, continued_fraction_convergents,
    deutsch_jozsa, factor_from_order, gcd, grover_search, hadamard_test, marked_probability,
    mod_exp, optimal_iterations, order_finding, phase_flip_correct, phase_flip_encode, prepare_ghz,
    prepare_w, recover_period_from_constraints, shor_factor, simon, superdense_decode, swap_test,
    teleport,
};
pub use algorithms::{
    AmplitudeEstimationResult, CoinInit, CoinedWalk, IterativeQpeResult, LcuOperator, Mat2, Pauli,
    PauliTerm, StatePreparation, VqlsResult, VqlsSolver, amplitude_estimation,
    chebyshev_qsp_angles, chebyshev_t, iterative_phase_estimation, position_std_about,
    qsp_top_left, qsp_unitary, signal_operator,
};
pub use channel::pauli_channel::{
    bit_flip_channel, bit_phase_flip_channel, pauli_channel, pauli_twirl, phase_flip_channel,
};
pub use error::{QuantumError, QuantumResult};
pub use error_correction::{
    PauliError, StabKind, Stabilizer, SurfaceCode, SurfaceCodeConfig, Syndrome,
};
pub use fourier::{PhaseEstimationResult, phase_estimation, qft_inplace, qft_inverse_inplace};
pub use linear_systems::{HermitianMatrix, HhlConfig, HhlResult, hhl_solve};
pub use midcircuit::{
    ClassicalRegister, MidCircuitOp, apply_if, measure_and_collapse, measure_deterministic,
    run as run_midcircuit,
};
pub use mps::{MatrixProductState, MpsConfig};

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests;

#[cfg(test)]
mod e2e_tests {
    use crate::channel::noise::{amplitude_damping_channel, depolarizing_channel};
    use crate::circuit::circuit::{GateOp, QuantumCircuit};
    use crate::density::density::DensityMatrix;
    use crate::density::metrics::purity;
    use crate::fourier::{phase_estimation, qft_inplace, qft_inverse_inplace};
    use crate::gates::controlled::apply_cnot;
    use crate::gates::hadamard::gate_h;
    use crate::gates::hadamard::gate_s;
    use crate::gates::pauli::gate_x;
    use crate::handle::LcgRng;
    use crate::pauli::expval::expectation_value;
    use crate::pauli::hamiltonian::Hamiltonian;
    use crate::pauli::pauli_string::PauliOp;
    use crate::ptx_kernels::{
        expval_pauli_ptx, measure_prob_ptx, partial_trace_ptx, qft_butterfly_ptx,
        statevec_apply_1q_ptx, statevec_apply_2q_ptx, statevec_apply_cnot_ptx, trotter_step_ptx,
    };
    use crate::qaoa::qaoa::QaoaCircuit;
    use crate::statevec::apply_1q::apply_1q_inplace;
    use crate::statevec::state::StateVector;
    use crate::vqe::ansatz::HardwareEfficientAnsatz;
    use crate::vqe::vqe::VqeOptimizer;

    // Test 1: |0⟩ state has norm 1
    #[test]
    fn e2e_01_zero_state_norm_is_one() {
        let sv = StateVector::new_zero_state(3).expect("valid 3-qubit zero state");
        let norm = sv.norm_sq();
        assert!((norm - 1.0).abs() < 1e-6, "norm={norm}");
    }

    // Test 2: H gate on |0⟩ gives equal superposition
    #[test]
    fn e2e_02_hadamard_creates_superposition() {
        let mut sv = StateVector::new_zero_state(1).expect("valid 1-qubit zero state");
        apply_1q_inplace(&mut sv, 0, &gate_h()).expect("valid qubit index for H gate");
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
        let mut sv = StateVector::new_zero_state(1).expect("valid 1-qubit zero state");
        apply_1q_inplace(&mut sv, 0, &gate_h()).expect("valid qubit index for first H gate");
        apply_1q_inplace(&mut sv, 0, &gate_h()).expect("valid qubit index for second H gate");
        assert!((sv.amps[0].re - 1.0).abs() < 1e-5, "amp0={:?}", sv.amps[0]);
        assert!(sv.amps[1].norm() < 1e-5, "amp1={:?}", sv.amps[1]);
    }

    // Test 4: CNOT creates Bell state |00⟩+|11⟩/√2
    #[test]
    fn e2e_04_cnot_creates_bell_state() {
        let mut sv = StateVector::new_zero_state(2).expect("valid 2-qubit zero state");
        apply_1q_inplace(&mut sv, 0, &gate_h()).expect("valid qubit index for H gate");
        apply_cnot(&mut sv, 0, 1).expect("valid CNOT qubit indices");
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
        let sv = StateVector::new_zero_state(1).expect("valid 1-qubit zero state");
        let mut ham = Hamiltonian::new();
        ham.add_term(1.0, vec![PauliOp::Z]);
        let ev = expectation_value(&sv, &ham).expect("valid Hamiltonian expectation value");
        assert!((ev - 1.0).abs() < 1e-5, "ev={ev}");
    }

    // Test 6: Pauli-Z expectation of |1⟩ = -1
    #[test]
    fn e2e_06_pauli_z_expval_one_state() {
        let mut sv = StateVector::new_zero_state(1).expect("valid 1-qubit zero state");
        apply_1q_inplace(&mut sv, 0, &gate_x()).expect("valid qubit index for X gate"); // flip to |1⟩
        let mut ham = Hamiltonian::new();
        ham.add_term(1.0, vec![PauliOp::Z]);
        let ev = expectation_value(&sv, &ham).expect("valid Hamiltonian expectation value");
        assert!((ev - (-1.0)).abs() < 1e-5, "ev={ev}");
    }

    // Test 7: Mixed Hamiltonian expectation value is finite
    #[test]
    fn e2e_07_mixed_hamiltonian_expval_finite() {
        let mut sv = StateVector::new_zero_state(2).expect("valid 2-qubit zero state");
        apply_1q_inplace(&mut sv, 0, &gate_h()).expect("valid qubit index for H gate");
        apply_cnot(&mut sv, 0, 1).expect("valid CNOT qubit indices");
        let mut ham = Hamiltonian::new();
        ham.add_term(0.5, vec![PauliOp::Z, PauliOp::Z]);
        ham.add_term(0.3, vec![PauliOp::X, PauliOp::I]);
        ham.add_term(0.2, vec![PauliOp::I, PauliOp::Y]);
        let ev = expectation_value(&sv, &ham).expect("valid Hamiltonian expectation value");
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
        let e_initial = opt
            .energy(&opt.params.clone())
            .expect("valid energy computation");
        let (e_final, _) = opt.optimize(5, 0.1).expect("valid VQE optimization run");
        assert!(
            e_final <= e_initial + 1e-3,
            "e_initial={e_initial}, e_final={e_final}"
        );
    }

    // Test 9: QAOA circuit runs without error
    #[test]
    fn e2e_09_qaoa_runs_without_error() {
        let circuit = QaoaCircuit::new(4, 2, vec![0.3, 0.5], vec![0.7, 0.2])
            .expect("valid QAOA circuit parameters");
        let graph = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        let sv = circuit.run(&graph).expect("valid QAOA circuit run");
        let norm = sv.norm_sq();
        assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
    }

    // Test 10: Density matrix from pure state has purity 1
    #[test]
    fn e2e_10_pure_state_purity_is_one() {
        let sv = StateVector::new_zero_state(2).expect("valid 2-qubit zero state");
        let dm = DensityMatrix::from_pure_state(&sv);
        let p = purity(&dm);
        assert!((p - 1.0).abs() < 1e-5, "purity={p}");
    }

    // Test 11: Depolarizing channel reduces purity
    #[test]
    fn e2e_11_depolarizing_channel_reduces_purity() {
        let sv = StateVector::new_zero_state(1).expect("valid 1-qubit zero state");
        let dm = DensityMatrix::from_pure_state(&sv);
        let p_initial = purity(&dm);

        let ch = depolarizing_channel(0.5, 2).expect("valid depolarizing channel");
        let dm_out = ch.apply(&dm).expect("valid channel application");
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
            let s8 = qft_butterfly_ptx(sm);
            assert!(!s1.is_empty(), "statevec_apply_1q sm={sm}");
            assert!(!s2.is_empty(), "statevec_apply_2q sm={sm}");
            assert!(!s3.is_empty(), "statevec_apply_cnot sm={sm}");
            assert!(!s4.is_empty(), "expval_pauli sm={sm}");
            assert!(!s5.is_empty(), "partial_trace sm={sm}");
            assert!(!s6.is_empty(), "trotter_step sm={sm}");
            assert!(!s7.is_empty(), "measure_prob sm={sm}");
            assert!(!s8.is_empty(), "qft_butterfly sm={sm}");
        }
    }

    // Supplementary: amplitude damping channel test
    #[test]
    fn e2e_13_amplitude_damping_channel() {
        let sv = StateVector::new_zero_state(1).expect("valid 1-qubit zero state");
        let dm = DensityMatrix::from_pure_state(&sv);
        let ch = amplitude_damping_channel(0.3).expect("valid amplitude damping channel");
        let dm_out = ch.apply(&dm).expect("valid channel application");
        let tr = dm_out.trace();
        assert!((tr.re - 1.0).abs() < 1e-5, "trace={}", tr.re);
    }

    // Supplementary: circuit with explicit CNOTs
    #[test]
    fn e2e_14_circuit_bell_state() {
        let mut circ = QuantumCircuit::new(2);
        circ.add_gate(GateOp::H);
        circ.add_gate(GateOp::Cnot { ctrl: 0, tgt: 1 });
        let sv = StateVector::new_zero_state(2).expect("valid 2-qubit zero state");
        let mut rng = LcgRng::new(7);
        let out = circ
            .exec_on_state(&sv, &mut rng)
            .expect("valid circuit execution");
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        assert!((out.amps[0].re - inv_sqrt2).abs() < 1e-5);
        assert!((out.amps[3].re - inv_sqrt2).abs() < 1e-5);
    }

    // Test 15: QFT on |0…0⟩ yields the uniform superposition
    #[test]
    fn e2e_15_qft_uniform_from_zero_state() {
        let mut sv = StateVector::new_zero_state(3).expect("valid 3-qubit zero state");
        qft_inplace(&mut sv, &[0, 1, 2]).expect("valid QFT qubit indices");
        let expected = 1.0 / (8.0_f32).sqrt();
        for a in &sv.amps {
            assert!((a.re - expected).abs() < 1e-5, "re={}", a.re);
            assert!(a.im.abs() < 1e-5, "im={}", a.im);
        }
    }

    // Test 16: QFT followed by inverse QFT is the identity
    #[test]
    fn e2e_16_qft_roundtrip_identity() {
        let mut rng = LcgRng::new(2024);
        let mut amps = Vec::with_capacity(8);
        for _ in 0..8 {
            amps.push(num_complex::Complex::new(
                rng.next_normal(),
                rng.next_normal(),
            ));
        }
        let mut sv = StateVector { amps, n_qubits: 3 };
        sv.normalize_inplace();
        let original = sv.amps.clone();
        qft_inplace(&mut sv, &[0, 1, 2]).expect("valid QFT qubit indices");
        qft_inverse_inplace(&mut sv, &[0, 1, 2]).expect("valid inverse QFT qubit indices");
        for (a, b) in sv.amps.iter().zip(original.iter()) {
            assert!((a.re - b.re).abs() < 1e-5, "re {a:?} vs {b:?}");
            assert!((a.im - b.im).abs() < 1e-5, "im {a:?} vs {b:?}");
        }
    }

    // Test 17: QPE recovers φ = 1/4 (integer 1) for U = S on eigenstate |1⟩
    #[test]
    fn e2e_17_qpe_recovers_quarter_phase() {
        let mut sv = StateVector::new_zero_state(3).expect("valid 3-qubit zero state");
        apply_1q_inplace(&mut sv, 2, &gate_x())
            .expect("valid qubit index for X gate on target qubit"); // target qubit → |1⟩
        let res =
            phase_estimation(&mut sv, &[0, 1], 2, &gate_s()).expect("valid QPE configuration");
        assert_eq!(res.integer, 1, "integer={}", res.integer);
        assert!((res.phase - 0.25).abs() < 1e-6, "phase={}", res.phase);
        assert!(res.probability > 0.999, "prob={}", res.probability);
    }
}
