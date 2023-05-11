use ndarray::{Array1, ArrayD, ArrayViewD};
use serde::Serialize;

use super::{Layer, LayerJson};

#[derive(Clone, Serialize)]
pub struct Flatten {
    name:        String,
    input_shape: Vec<usize>,
}

impl Flatten {
    #[must_use]
    pub fn new(input_shape: Vec<usize>) -> Flatten {
        Flatten {
            name: "flatten".into(),
            input_shape,
        }
    }
}

impl Layer for Flatten {
    fn box_clone(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

    fn apply(&self, input: &ArrayViewD<f32>) -> ArrayD<f32> {
        Array1::from_iter(input.iter().copied()).into_dyn()
    }

    fn input_shape(&self) -> Vec<usize> {
        self.input_shape.clone()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn num_params(&self) -> usize {
        0
    }

    fn num_muls(&self) -> usize {
        0
    }

    fn output_shape(&self) -> Vec<usize> {
        let mut output_shape = 1;

        for i in self.input_shape() {
            output_shape *= i;
        }

        vec![output_shape]
    }

    fn to_json(&self) -> LayerJson {
        LayerJson::Flatten {
            input_shape: self.input_shape(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::Array3;
    use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
    use rand::rngs::StdRng;

    #[test]
    fn flatten_test() {
        let seed = 694201337;
        let mut rng = StdRng::seed_from_u64(seed);

        let input = Array3::random_using((27, 17, 32), Uniform::<f32>::new(-5.0, 5.0), &mut rng);

        let flat = Flatten::new(vec![27, 17, 32]);

        let output = flat.apply(&input.into_dyn().view());

        let n_multiplications = flat.num_muls();

        let n_params = flat.num_params();

        assert_eq!(output.len(), 14688);

        println!(
            "
        {} \n
        # of parameters: {}\n
        output dim: {} \n
        # of ops: {}\n
        output:\n
        {}",
            flat.name,
            n_params,
            output.len(),
            n_multiplications,
            output
        );
    }
}
