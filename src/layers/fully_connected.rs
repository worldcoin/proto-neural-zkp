#![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]
use ndarray::{Array1, Array2};

pub struct FCLayer<T> {
    pub output:            Array1<T>,
    pub n_params:          i32,
    pub n_multiplications: i32,
    pub name:              String,
}

pub fn fully_connected(
    input: &Array1<f32>,
    weights: &Array2<f32>,
    biases: &Array1<f32>,
) -> FCLayer<f32> {
    assert!(input.ndim() == 1, "Input must be a flattenened array!");
    assert!(
        weights.shape()[1] == input.shape()[0],
        "Input shapes must match (for the dot product to work)!"
    );
    assert!(
        weights.shape()[0] == biases.shape()[0],
        "Output shapes must match!"
    );

    let output = weights.dot(input) + biases;

    let n_params = (weights.len() + biases.len()) as i32;

    let n_multiplications = weights.len() as i32;

    FCLayer {
        output,
        n_params,
        n_multiplications,
        name: String::from("fully_connected"),
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
    use rand::rngs::StdRng;

    #[test]
    fn fully_connected_test() {
        let seed = 694201337;
        let mut rng = StdRng::seed_from_u64(seed);

        let input = Array1::random_using(14688, Uniform::<f32>::new(-10.0, 10.0), &mut rng);
        let weights =
            Array2::random_using((1000, 14688), Uniform::<f32>::new(-10.0, 10.0), &mut rng);
        let biases = Array1::random_using(1000, Uniform::<f32>::new(-10.0, 10.0), &mut rng);

        let FCLayer::<f32> {
            output: x,
            n_params,
            n_multiplications,
            name,
        } = fully_connected(&input, &weights, &biases);

        println!(
            "
        {} 
        # of parameters: {}
        output dim: {}x1
        # of ops: {}
        output:\n{}",
            name,
            n_params,
            x.dim(),
            n_multiplications,
            x
        );
    }
}
