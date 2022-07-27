use ndarray::Array3;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use neural_zkp::maxpool::*;
use rand_isaac::isaac64::Isaac64Rng;

#[test]
fn maxpool_test() {
    let seed = 694201337;
    let mut rng = Isaac64Rng::seed_from_u64(seed);

    let input = Array3::random_using((116, 76, 32), Uniform::<f32>::new(-5.0, 5.0), &mut rng);
    let s = 2;

    let MaxPool::<f32> {
        output: x,
        n_params,
        n_multiplications,
        name,
    } = max_pooling_layer(input, s);

    assert_eq!(x.dim(), (58, 38, 32));

    let (dim_x, dim_y, dim_z) = x.dim();

    println!(
        "
        {} \n
        # of parameters: {}\n
        output dim: {}x{}x{}\n
        # of ops: {}\n
        output:\n
        {}",
        name, n_params, dim_x, dim_y, dim_z, n_multiplications, x
    );
}
