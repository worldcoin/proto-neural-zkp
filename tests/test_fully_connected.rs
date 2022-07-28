use ndarray::{Array1, Array2};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use neural_zkp::fully_connected::*;
use rand_isaac::isaac64::Isaac64Rng;

#[test]
fn fc_test() {
    let seed = 694201337;
    let mut rng = Isaac64Rng::seed_from_u64(seed);

    let input = Array1::random_using(14688, Uniform::<f32>::new(-10.0, 10.0), &mut rng);
    let weights = Array2::random_using((14688, 1000), Uniform::<f32>::new(-10.0, 10.0), &mut rng);
    let biases = Array1::random_using(1000, Uniform::<f32>::new(-10.0, 10.0), &mut rng);

    let FCLayer::<f32> {
        output: x,
        n_params,
        n_multiplications,
        name,
    } = fully_connected(input, weights, biases);

    println!(
        "
        {} 
        # of parameters: {}
        output dim: {}x1
        # of ops: {}
        output:\n{}",
        name,
        x.dim(),
        n_params,
        n_multiplications,
        x
    );
}
