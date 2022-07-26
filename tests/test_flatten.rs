use ndarray::Array3;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use neural_zkp::flatten::*;
use rand_isaac::isaac64::Isaac64Rng;

#[test]
fn flatten_test() {
    let seed = 694201337;
    let mut rng = Isaac64Rng::seed_from_u64(seed);

    let input = Array3::random_using((27, 17, 32), Uniform::<f32>::new(-5.0, 5.0), &mut rng);

    let (x, n_params, n_multiplications, name) = flatten_layer(input);

    assert_eq!(x.len(), 14688);

    println!(
        "
        {} \n
        # of parameters: {}\n
        output dim: {} \n
        # of ops: {}\n
        output:\n
        {}",
        name,
        n_params,
        x.len(),
        n_multiplications,
        x
    );
}
