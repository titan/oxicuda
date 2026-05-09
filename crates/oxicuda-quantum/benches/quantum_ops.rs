use criterion::{Criterion, criterion_group, criterion_main};
use oxicuda_quantum::gates::hadamard::gate_h;
use oxicuda_quantum::statevec::apply_1q::apply_1q_inplace;
use oxicuda_quantum::statevec::state::StateVector;

fn bench_ptx_kernels(c: &mut Criterion) {
    let mut g = c.benchmark_group("quantum_ptx");
    for sm in [75u32, 80, 89, 100] {
        g.bench_function(format!("statevec_apply_1q_sm{sm}"), |b| {
            b.iter(|| oxicuda_quantum::ptx_kernels::statevec_apply_1q_ptx(sm))
        });
        g.bench_function(format!("statevec_apply_2q_sm{sm}"), |b| {
            b.iter(|| oxicuda_quantum::ptx_kernels::statevec_apply_2q_ptx(sm))
        });
        g.bench_function(format!("statevec_apply_cnot_sm{sm}"), |b| {
            b.iter(|| oxicuda_quantum::ptx_kernels::statevec_apply_cnot_ptx(sm))
        });
        g.bench_function(format!("expval_pauli_sm{sm}"), |b| {
            b.iter(|| oxicuda_quantum::ptx_kernels::expval_pauli_ptx(sm))
        });
        g.bench_function(format!("partial_trace_sm{sm}"), |b| {
            b.iter(|| oxicuda_quantum::ptx_kernels::partial_trace_ptx(sm))
        });
        g.bench_function(format!("trotter_step_sm{sm}"), |b| {
            b.iter(|| oxicuda_quantum::ptx_kernels::trotter_step_ptx(sm))
        });
        g.bench_function(format!("measure_prob_sm{sm}"), |b| {
            b.iter(|| oxicuda_quantum::ptx_kernels::measure_prob_ptx(sm))
        });
    }
    g.finish();
}

fn bench_statevec(c: &mut Criterion) {
    let mut g = c.benchmark_group("quantum_algo");

    g.bench_function("h_gate_10qubits", |b| {
        b.iter(|| {
            let mut sv = StateVector::new_zero_state(10).unwrap();
            for q in 0..10 {
                apply_1q_inplace(&mut sv, q, &gate_h()).unwrap();
            }
            sv
        })
    });

    g.bench_function("bell_state_2q", |b| {
        use oxicuda_quantum::gates::controlled::apply_cnot;
        b.iter(|| {
            let mut sv = StateVector::new_zero_state(2).unwrap();
            apply_1q_inplace(&mut sv, 0, &gate_h()).unwrap();
            apply_cnot(&mut sv, 0, 1).unwrap();
            sv
        })
    });

    g.bench_function("zz_feature_map_3q_depth2", |b| {
        use oxicuda_quantum::embedding::zz_feature::zz_feature_map;
        let data = [0.5_f32, 1.0, -0.3];
        b.iter(|| zz_feature_map(&data, 2).unwrap())
    });

    g.bench_function("vqe_energy_2q", |b| {
        use oxicuda_quantum::handle::LcgRng;
        use oxicuda_quantum::pauli::hamiltonian::Hamiltonian;
        use oxicuda_quantum::pauli::pauli_string::PauliOp;
        use oxicuda_quantum::vqe::ansatz::HardwareEfficientAnsatz;
        use oxicuda_quantum::vqe::vqe::VqeOptimizer;
        let ans = HardwareEfficientAnsatz::new(2, 1);
        let mut ham = Hamiltonian::new();
        ham.add_term(1.0, vec![PauliOp::Z, PauliOp::Z]);
        let mut rng = LcgRng::new(42);
        let opt = VqeOptimizer::new(ans, ham, &mut rng);
        let params = opt.params.clone();
        b.iter(|| opt.energy(&params).unwrap())
    });

    g.bench_function("qaoa_4q_p1", |b| {
        use oxicuda_quantum::qaoa::qaoa::QaoaCircuit;
        let circuit = QaoaCircuit::new(4, 1, vec![0.3], vec![0.7]).unwrap();
        let graph = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        b.iter(|| circuit.run(&graph).unwrap())
    });

    g.finish();
}

criterion_group!(benches, bench_ptx_kernels, bench_statevec);
criterion_main!(benches);
