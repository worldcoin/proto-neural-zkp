use ndarray::{Array1, Array2, ArrayD, ArrayViewD, Ix1};

use super::Layer;

pub struct FCLayer<T> {
    pub output:            Array1<T>,
    pub n_params:          i32,
    pub n_multiplications: i32,
    pub name:              String,
}

pub struct FullyConnected {
    weights: Array2<f32>,
    biases:  Array1<f32>,
    name:    String,
}

impl FullyConnected {
    #[must_use]
    pub fn new(name: String, weights: Array2<f32>, biases: Array1<f32>) -> FullyConnected {
        FullyConnected {
            weights,
            biases,
            name,
        }
    }
}

impl Layer for FullyConnected {
    fn apply(&self, input: &ArrayViewD<f32>) -> ArrayD<f32> {
        assert!(input.ndim() == 1, "Input must be a flattenened array!");
        assert!(
            self.weights.shape()[1] == input.shape()[0],
            "Input shapes must match (for the dot product to work)!"
        );
        assert!(
            self.weights.shape()[0] == self.biases.shape()[0],
            "Output shapes must match!"
        );

        let output = self
            .weights
            .dot(&input.clone().into_dimensionality::<Ix1>().unwrap())
            + self.biases.clone();

        output.into_dyn()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn num_params(&self) -> usize {
        self.weights.len() + self.biases.len()
    }

    fn num_muls(&self, _input: &ArrayViewD<f32>) -> usize {
        self.weights.len()
    }

    fn output_shape(&self, _input: &ArrayViewD<f32>, dim: usize) -> Option<Vec<usize>> {
        if dim == 1 {
            let dim = self.biases.dim();

            Some(vec![dim])
        } else {
            None
        }
    }
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

    let output = weights.dot(&input.clone().into_dimensionality::<Ix1>().unwrap()) + biases;

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

        let fully_connected = FullyConnected::new("fully_connected".into(), weights, biases);

        let output = fully_connected.apply(&input.clone().into_dyn().view());

        let n_params = fully_connected.num_params();

        let n_multiplications = fully_connected.num_muls(&input.into_dyn().view());

        // let FCLayer::<f32> {
        //     output: x,
        //     n_params,
        //     n_multiplications,
        //     name,
        // } = fully_connected(&input, &weights, &biases);

        println!(
            "
        {}
        # of parameters: {}
        output dim: {}x1
        # of ops: {}
        output:\n{}",
            fully_connected.name,
            n_params,
            output.len(),
            n_multiplications,
            output
        );
    }
}
