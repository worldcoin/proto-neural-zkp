use ndarray::{Array1, Array2, ArrayD, ArrayViewD, Ix1};
use serde::Serialize;

use super::{Layer, LayerJson};

#[derive(Clone, Serialize)]
pub struct FullyConnected {
    weights: Array2<f32>,
    biases:  Array1<f32>,
    name:    String,
}

impl FullyConnected {
    #[must_use]
    pub fn new(weights: Array2<f32>, biases: Array1<f32>) -> FullyConnected {
        FullyConnected {
            weights,
            biases,
            name: "full".into(),
        }
    }
}

impl Layer for FullyConnected {
    fn box_clone(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

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

    fn num_muls(&self) -> usize {
        self.weights.len()
    }

    fn output_shape(&self) -> Vec<usize> {
        assert!(
            self.weights.shape()[0] == self.biases.shape()[0],
            "Output shapes must match!"
        );

        vec![self.weights.shape()[0]]
    }

    fn input_shape(&self) -> Vec<usize> {
        vec![self.weights.shape()[1]]
    }

    fn to_json(&self) -> LayerJson {
        LayerJson::FullyConnected {
            weights: self.weights.clone().into(),
            biases:  self.biases.clone().into(),
        }
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

        let fully_connected = FullyConnected::new(weights, biases);

        let output = fully_connected.apply(&input.into_dyn().view());

        let n_params = fully_connected.num_params();

        let n_multiplications = fully_connected.num_muls();

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
