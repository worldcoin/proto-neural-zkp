use neural_zkp::conv::*;

use ndarray::{Array3, Array4};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use rand_isaac::isaac64::Isaac64Rng;

#[test]
fn conv_test() {
    // man of culture fixed randomness seed
    let seed = 694201337;
    let mut rng = Isaac64Rng::seed_from_u64(seed);

    let input = Array3::random_using((120, 80, 3), Uniform::<f32>::new(-5.0, 5.0), &mut rng);
    let kernel = Array4::random_using((32, 5, 5, 3), Uniform::<f32>::new(-10.0, 10.0), &mut rng);

    // x is the result of  conv(input, kernel)
    let (x, n_params, name) = convolution(input, kernel);

    assert_eq!(x.dim(), (116, 76, 32));

    let (dim_x, dim_y, dim_z) = x.dim();

    println!(
        "# of parameters: {}\n
    output dim: {}x{}x{}\n
    {} output:\n
    {}",
        n_params, dim_x, dim_y, dim_z, name, x
    );
}
