#![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]
use ndarray::{ArrayD, ArrayViewD};
use serde::Serialize;

use super::{Layer, LayerJson};

#[derive(Clone, Serialize)]
pub struct Relu {
    name:        String,
    input_shape: Vec<usize>,
}

impl Relu {
    #[must_use]
    pub fn new(input_shape: Vec<usize>) -> Self {
        Self {
            name: "relu".into(),
            input_shape,
        }
    }
}

impl Layer for Relu {
    fn box_clone(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
    fn apply(&self, input: &ArrayViewD<f32>) -> ArrayD<f32> {
        input.mapv(|x| f32::max(0.0, x))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn num_params(&self) -> usize {
        let mut params = 1;

        for i in self.input_shape() {
            params *= i;
        }

        params
    }

    fn num_muls(&self) -> usize {
        0
    }

    fn output_shape(&self) -> Vec<usize> {
        self.input_shape.clone()
    }

    fn input_shape(&self) -> Vec<usize> {
        self.input_shape.clone()
    }

    fn to_json(&self) -> LayerJson {
        LayerJson::Relu {
            input_shape: self.input_shape(),
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use ndarray::{arr1, arr3, Ix1, Ix3};

    // Array3 ReLU
    #[test]
    fn relu_test3() {
        let input = arr3(&[[[1.2, -4.3], [-2.1, 4.3]], [[5.2, 6.1], [7.6, -1.8]], [
            [9.3, 0.0],
            [1.2, 3.4],
        ]]);

        let relu = Relu::new(vec![3, 2, 2]);

        let output = relu.apply(&input.into_dyn().view());

        let n_params = relu.num_params();

        let n_multiplications = relu.num_muls();

        assert_eq!(
            output,
            arr3(&[[[1.2, 0.0], [0.0, 4.3]], [[5.2, 6.1], [7.6, 0.0]], [
                [9.3, 0.0],
                [1.2, 3.4],
            ]])
            .into_dyn()
        );

        let size: (usize, usize, usize) = (3, 2, 2);

        assert_eq!(
            &output.clone().into_dimensionality::<Ix3>().unwrap().dim(),
            &size
        );

        println!(
            "
        {} 
        # of parameters: {}
        output dim: {}x1
        # of ops: {}
        output:\n{}",
            relu.name(),
            n_params,
            n_params,
            n_multiplications,
            output
        );
    }

    // Array1 ReLU
    #[test]
    fn relu_test1() {
        let input = arr1(&[-4., -3.4, 6., 7., 1., -3.]).into_dyn();

        let relu = Relu::new(vec![6]);

        let output = relu.apply(&input.into_dyn().view());

        let n_params = relu.num_params();

        let n_multiplications = relu.num_muls();

        assert_eq!(output, arr1(&[0., 0., 6., 7., 1., 0.]).into_dyn());

        assert_eq!(
            output.clone().into_dimensionality::<Ix1>().unwrap().len(),
            6
        );

        println!(
            "
        {} 
        # of parameters: {}
        output dim: {}x1
        # of ops: {}
        output:\n{}",
            relu.name(),
            n_params,
            n_params,
            n_multiplications,
            output
        );
    }
}
