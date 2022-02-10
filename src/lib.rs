use eyre::Result as EyreResult;
use log::Level;
use plonky2::{
    field::{field_types::Field, goldilocks_field::GoldilocksField},
    gadgets::arithmetic_u32::U32Target,
    gates::arithmetic_u32::U32ArithmeticGate,
    iop::{
        target::Target,
        witness::{PartialWitness, Witness},
    },
    plonk::{
        circuit_builder::CircuitBuilder,
        circuit_data::CircuitConfig,
        config::{GenericConfig, KeccakGoldilocksConfig, PoseidonGoldilocksConfig},
    },
};
use rand::prelude::*;
use std::sync::Arc;
use structopt::StructOpt;
use tokio::sync::broadcast;
use tracing::{info, trace};

#[derive(Clone, Debug, PartialEq, StructOpt)]
pub struct Options {}

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
// type C = KeccakGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;
type Builder = CircuitBuilder<F, D>;

// https://arxiv.org/pdf/1509.09308.pdf
// https://en.wikipedia.org/wiki/Freivalds%27_algorithm ?

fn to_field(value: i32) -> F {
    let value = value >> 24;
    assert!(value.abs() <= 128);
    if value >= 0 {
        F::from_canonical_u32(value as u32)
    } else {
        -F::from_canonical_u32(-value as u32)
    }
}

/// Compute the inner product of `coefficients` and `input`
fn dot(builder: &mut Builder, coefficients: &[i32], input: &[Target]) -> Target {
    // TODO: Compare this accumulator approach against a batch sum.
    assert_eq!(coefficients.len(), input.len());
    builder.push_context(Level::Info, "dot");
    let mut sum = builder.zero();
    for (&coefficient, &input) in coefficients.iter().zip(input) {
        let coefficient = to_field(coefficient);
        sum = builder.mul_const_add(coefficient, input, sum);
    }
    builder.pop_context();
    sum
}

fn full(builder: &mut Builder, coefficients: &[i32], input: &[Target]) -> Vec<Target> {
    let input_size = input.len();
    let output_size = coefficients.len() / input_size;
    assert_eq!(coefficients.len(), input_size * output_size);

    builder.push_context(Level::Info, "full");
    let mut output = Vec::with_capacity(output_size);
    for coefficients in coefficients.chunks_exact(input_size) {
        output.push(dot(builder, coefficients, input));
    }
    builder.pop_context();
    output
}

pub async fn main(options: Options) -> EyreResult<()> {
    let config = CircuitConfig {
        // num_wires: 300, // 30s, 317785B
        // num_wires: 400, // 34s, 348272B
        num_wires: 450, // 18s, 349340B, 12
        // num_wires: 500, // 20s, 359772B
        // num_wires: 600, // 23s, 382665B
        // num_wires: 1000, // 34s,  478825B
        num_routed_wires: 400,
        constant_gate_size: 10,
        ..CircuitConfig::default()
    };
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let input_size = 14688;
    // let output_size = 100;  // 18s, 349340B, 12
    // let output_size = 200; // 40s
    // let output_size = 400; // 86.4806s
    // let output_size = 800; // 259.6348s, 380660B
    // let output_size = 1000; //
    // let output_size = 1600; //  (crash)

    let output_size = 1600; // 44s

    // Coefficients
    let coefficients: Vec<i32> = (0..input_size * output_size).map(|_| random()).collect();

    /// Circuit
    // Inputs
    builder.push_context(Level::Info, "Inputs");
    let inputs = builder.add_virtual_targets(input_size);
    inputs
        .iter()
        .for_each(|target| builder.register_public_input(*target));
    builder.pop_context();

    // Circuit
    let output = full(&mut builder, &coefficients, &inputs);
    output
        .iter()
        .for_each(|target| builder.register_public_input(*target));

    // Log circuit size
    builder.print_gate_counts(0);
    info!("Building circuit");
    let data = builder.build::<C>();

    /// Proof

    // Set witness for proof
    info!("Proving");
    let input_values = (0..input_size as i32).into_iter().map(|_| random());
    let mut pw = PartialWitness::new();
    for (&target, value) in inputs.iter().zip(input_values) {
        let value = F::from_canonical_u32(value);
        pw.set_target(target, value);
    }
    let proof = data.prove(pw).unwrap();
    dbg!(proof.public_inputs.len());
    let compressed = proof.clone().compress(&data.common).unwrap();
    let bytes = compressed.to_bytes().unwrap();
    dbg!(bytes.len());

    // Verifying
    info!("Verifying");
    data.verify(proof);

    Ok(())
}

#[cfg(test)]
pub mod test {
    use super::*;
    use pretty_assertions::assert_eq;
    use proptest::proptest;
    use tracing::{error, warn};
    use tracing_test::traced_test;

    #[test]
    #[allow(clippy::eq_op)]
    fn test_with_proptest() {
        proptest!(|(a in 0..5, b in 0..5)| {
            assert_eq!(a + b, b + a);
        });
    }

    #[test]
    #[traced_test]
    fn test_with_log_output() {
        error!("logged on the error level");
        assert!(logs_contain("logged on the error level"));
    }

    #[tokio::test]
    #[traced_test]
    #[allow(clippy::semicolon_if_nothing_returned)] // False positive
    async fn async_test_with_log() {
        // Local log
        info!("This is being logged on the info level");

        // Log from a spawned task (which runs in a separate thread)
        tokio::spawn(async {
            warn!("This is being logged on the warn level from a spawned task");
        })
        .await
        .unwrap();

        // Ensure that `logs_contain` works as intended
        assert!(logs_contain("logged on the info level"));
        assert!(logs_contain("logged on the warn level"));
        assert!(!logs_contain("logged on the error level"));
    }
}

#[cfg(feature = "bench")]
pub mod bench {
    use criterion::{black_box, BatchSize, Criterion};
    use proptest::{
        strategy::{Strategy, ValueTree},
        test_runner::TestRunner,
    };
    use std::time::Duration;
    use tokio::runtime;

    pub fn group(criterion: &mut Criterion) {
        bench_example_proptest(criterion);
        bench_example_async(criterion);
    }

    /// Constructs an executor for async tests
    pub(crate) fn runtime() -> runtime::Runtime {
        runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    /// Example proptest benchmark
    /// Uses proptest to randomize the benchmark input
    fn bench_example_proptest(criterion: &mut Criterion) {
        let input = (0..5, 0..5);
        let mut runner = TestRunner::deterministic();
        // Note: benchmarks need to have proper identifiers as names for
        // the CI to pick them up correctly.
        criterion.bench_function("example_proptest", move |bencher| {
            bencher.iter_batched(
                || input.new_tree(&mut runner).unwrap().current(),
                |(a, b)| {
                    // Benchmark number addition
                    black_box(a + b)
                },
                BatchSize::LargeInput,
            );
        });
    }

    /// Example async benchmark
    /// See <https://bheisler.github.io/criterion.rs/book/user_guide/benchmarking_async.html>
    fn bench_example_async(criterion: &mut Criterion) {
        let duration = Duration::from_micros(1);
        criterion.bench_function("example_async", move |bencher| {
            bencher.to_async(runtime()).iter(|| async {
                tokio::time::sleep(duration).await;
            });
        });
    }
}
