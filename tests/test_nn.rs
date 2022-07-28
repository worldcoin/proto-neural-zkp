use ndarray::{Array1, Array2, Array3};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use neural_zkp::{conv, flatten, maxpool, relu};
use rand_isaac::isaac64::Isaac64Rng;

#[test]
fn nn_test() {
    let seed = 694201337;
    let mut rng = Isaac64Rng::seed_from_u64(seed);

    println!(
        "{:<20} | {:<15} | {:<15} | {:<15}",
        "layer", "output shape", "#parameters", "#ops"
    );
    println!("{:-<77}", "");
}
