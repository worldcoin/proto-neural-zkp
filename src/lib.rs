// #![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]
// Stabilized soon: https://github.com/rust-lang/rust/pull/93827

// Disable globally while still iterating on design
#![allow(clippy::missing_panics_doc)]
// TODO
#![allow(unreadable_literal)]
// benchmarking
#![feature(test)]

mod allocator;
mod anyhow;

pub mod layers;
pub mod nn;
pub mod serialize;

use self::{allocator::Allocator, anyhow::MapAny as _};
use bytesize::ByteSize;
use eyre::{eyre, Result as EyreResult};
use log::Level;
use plonky2::{
    field::types::Field,
    iop::{
        target::Target,
        witness::{PartialWitness, Witness},
    },
    plonk::{
        circuit_builder::CircuitBuilder,
        circuit_data::{CircuitConfig, CircuitData},
        config::{GenericConfig, KeccakGoldilocksConfig, PoseidonGoldilocksConfig},
        proof::{CompressedProofWithPublicInputs, ProofWithPublicInputs},
    },
};
use rand::Rng as _;
use std::{iter::once, sync::atomic::Ordering, time::Instant};
use structopt::StructOpt;
use tracing::{info, trace};

type Rng = rand_pcg::Mcg128Xsl64;

#[cfg(not(feature = "mimalloc"))]
#[global_allocator]
pub static ALLOCATOR: Allocator<allocator::StdAlloc> = allocator::new_std();

#[cfg(feature = "mimalloc")]
#[global_allocator]
pub static ALLOCATOR: Allocator<allocator::MiMalloc> = allocator::new_mimalloc();

#[derive(Clone, Debug, PartialEq, StructOpt)]
pub struct Options {
    /// Bench over increasing output sizes
    #[structopt(long)]
    pub bench: bool,

    /// The size of the input layer
    #[structopt(long, default_value = "1000")]
    pub input_size: usize,

    /// The size of the output layer
    #[structopt(long, default_value = "1000")]
    pub output_size: usize,

    /// Coefficient bits
    #[structopt(long, default_value = "16")]
    pub coefficient_bits: usize,

    /// Number of wire columns
    #[structopt(long, default_value = "400")]
    pub num_wires: usize,

    /// Number of routed wire columns
    #[structopt(long, default_value = "400")]
    pub num_routed_wires: usize,

    /// Number of constants per constant gate
    #[structopt(long, default_value = "90")]
    pub constant_gate_size: usize,
}

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
// type C = KeccakGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;
type Builder = CircuitBuilder<F, D>;
type Proof = ProofWithPublicInputs<F, C, D>;

// https://arxiv.org/pdf/1509.09308.pdf
// https://en.wikipedia.org/wiki/Freivalds%27_algorithm ?

fn to_field(value: i32) -> F {
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
    // builder.push_context(Level::Info, "dot");
    let mut sum = builder.zero();
    // iterates over array coefficients and input and performs the dot product:
    // sum = co[0] * in[0] + co[1] * in[1] .... + co[len(co)] * in[len(in)] <=>
    // len(co) = len (in) each coefficient needs to be a Goldilocks field
    // element (modular arithmetic inside a field element) CircuitBuilder.
    // mul_const_add()
    // sum = co[0] * in[0] + co[1] * in[1] .... + co[len(co)] * in[len(in)] <=>
    // len(co) = len (in) each coefficient needs to be a Goldilocks field
    // element (modular arithmetic inside a field element) CircuitBuilder.
    // mul_const_add()
    for (&coefficient, &input) in coefficients.iter().zip(input) {
        let coefficient = to_field(coefficient);
        sum = builder.mul_const_add(coefficient, input, sum);
    }
    // builder.pop_context();
    sum
}

// len(co) = k * len(in); k is a constant that belongs to N; k > 1 (ref 1)
fn full(builder: &mut Builder, coefficients: &[i32], input: &[Target]) -> Vec<Target> {
    let input_size = input.len();
    // len(output_size) = k
    let output_size = coefficients.len() / input_size;
    // enforces (ref 1)
    assert_eq!(coefficients.len(), input_size * output_size);

    // TODO: read docs CircuitBuilder.push_context(), Level
    builder.push_context(Level::Info, "full");

    // output is a vector that contains dot products of coefficients and inputs,
    // len(output) = k
    // output is a vector that contains dot products of coefficients and inputs,
    // len(output) = k
    let mut output = Vec::with_capacity(output_size);
    // &[i32].chunks_exact() creates an iterator over k arrays of len(input)
    for coefficients in coefficients.chunks_exact(input_size) {
        // builder enforces Goldilock fields modular arithmetic
        output.push(dot(builder, coefficients, input));
    }
    builder.pop_context();
    output
}

// Plonky2 circuit
struct Circuit {
    inputs:  Vec<Target>,
    outputs: Vec<Target>,
    data:    CircuitData<F, C, D>,
}

impl Circuit {
    // options are inputs that can be changed from cli and have default values
    fn build(options: &Options, coefficients: &[i32]) -> Circuit {
        assert_eq!(coefficients.len(), options.input_size * options.output_size);
        info!(
            "Building circuit for for {}x{} matrix-vector multiplication",
            options.input_size, options.output_size
        );

        let config = CircuitConfig {
            num_wires: options.num_wires,
            num_routed_wires: options.num_routed_wires,
            // constant_gate_size: options.constant_gate_size,
            ..CircuitConfig::default()
        };
        let mut builder = CircuitBuilder::<F, D>::new(config);

        // Inputs
        builder.push_context(Level::Info, "Inputs");
        // TODO: Look at CircuitBuilder.add_virtual_targets()
        let inputs = builder.add_virtual_targets(options.input_size);
        inputs
            .iter()
            .for_each(|target| builder.register_public_input(*target));
        builder.pop_context();

        // Circuit
        let outputs = full(&mut builder, &coefficients, &inputs);
        outputs
            .iter()
            .for_each(|target| builder.register_public_input(*target));

        // Log circuit size
        builder.print_gate_counts(0);
        let data = builder.build::<C>();

        Self {
            inputs,
            outputs,
            data,
        }
    }

    fn prove(&self, input: &[i32]) -> EyreResult<Proof> {
        info!("Proving {} size input", input.len());
        // TODO: Look into plonky2::iop::PartialWitness
        // What's the difference between PW and W
        let mut pw = PartialWitness::new();
        for (&target, &value) in self.inputs.iter().zip(input) {
            pw.set_target(target, to_field(value));
        }
        let proof = self.data.prove(pw).map_any()?;
        // let compressed = proof.clone().compress(&self.data.common).map_any()?;
        let proof_size = ByteSize(proof.to_bytes().map_any()?.len() as u64);
        info!("Proof size: {proof_size}");
        Ok(proof)
    }

    fn verify(&self, proof: &Proof) -> EyreResult<()> {
        info!(
            "Verifying proof with {} public inputs",
            proof.public_inputs.len()
        );
        // Why proof.clone()
        self.data.verify(proof.clone()).map_any()
    }
}

pub async fn main(mut rng: Rng, mut options: Options) -> EyreResult<()> {
    info!(
        "Computing proof for {}x{} matrix-vector multiplication",
        options.input_size, options.output_size
    );

    println!(
        "input_size,output_size,build_time_s,proof_time_s,proof_mem_b,proof_size_b,verify_time_s"
    );
    // TODO: What does this line mean?
    let output_sizes: Box<dyn Iterator<Item = usize>> = if options.bench {
        Box::new((1..).map(|n| n * 1000))
    } else {
        // what is once
        Box::new(once(options.output_size))
    };

    for output_size in output_sizes {
        options.output_size = output_size;

        // Coefficients
        // what is quantizing?
        let quantize_coeff = |c: i32| c % (1 << options.coefficient_bits);
        let coefficients: Vec<i32> = (0..options.input_size * options.output_size)
            .map(|_| quantize_coeff(rng.gen()))
            .collect();
        let now = Instant::now();
        let circuit = Circuit::build(&options, &coefficients);
        let circuit_build_time = now.elapsed();

        // Set witness for proof
        ALLOCATOR.peak_allocated.store(0, Ordering::Release);
        let input_values = (0..options.input_size as i32)
            .into_iter()
            .map(|_| rng.gen())
            .collect::<Vec<_>>();
        let now = Instant::now();
        let proof = circuit.prove(&input_values)?;
        let proof_mem = ALLOCATOR.peak_allocated.load(Ordering::Acquire);
        let proof_time = now.elapsed();
        let proof_size = proof.to_bytes().map_any()?.len() as u64;
        info!("Prover memory usage: {}", ByteSize(proof_mem as u64));

        // Verifying
        let now = Instant::now();
        circuit.verify(&proof)?;
        let verify_time = now.elapsed();

        println!(
            "{},{},{},{},{},{},{}",
            options.input_size,
            options.output_size,
            circuit_build_time.as_secs_f64(),
            proof_time.as_secs_f64(),
            proof_mem,
            proof_size,
            verify_time.as_secs_f64()
        );
    }

    Ok(())
}

#[cfg(test)]
pub mod test {
    use super::*;
    use proptest::proptest;
    use tracing::{error, warn};
    use tracing_test::traced_test;

    #[allow(clippy::eq_op)]
    // fn test_with_proptest() {
    //     proptest!(|(a in 0..5, b in 0..5)| {
    //         assert_eq!(a + b, b + a);
    //     });
    // }
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
