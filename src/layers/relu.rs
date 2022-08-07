#![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]
use ndarray::{ArrayD, ArrayViewD};

use super::Layer;

// pub struct ReLU<T> {
//     pub output:            ArrayD<T>,
//     pub n_params:          i32,
//     pub n_multiplications: i32,
//     pub name:              String,
// }

pub struct Relu {
    name:   String,
    params: usize,
}

impl Relu {
    #[must_use]
    pub const fn new(name: String) -> Self {
        Self { name, params: 0 }
    }

    pub fn update_params(&mut self, output: &ArrayViewD<f32>) {
        self.params = output.len();
    }
}

impl Layer for Relu {
    fn apply(&self, input: &ArrayViewD<f32>) -> ArrayD<f32> {
        input.mapv(|x| f32::max(0.0, x))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn num_params(&self) -> usize {
        self.params
    }

    fn num_muls(&self, _input: &ArrayViewD<f32>) -> usize {
        0
    }
}

// pub fn relu_layer(input: &ArrayD<f32>) -> ReLU<f32> {
//     let output = input.mapv(|x| f32::max(0.0, x));
//     let n_params = output.len() as i32;
//     let n_multiplications = 0;
//
//     ReLU {
//         output,
//         n_params,
//         n_multiplications,
//         name: String::from("ReLU"),
//     }
// }

#[cfg(test)]
pub mod test {
    use super::*;
    use ndarray::{arr1, arr3, Ix1, Ix3};

    #[test]
    fn relu_test3() {
        // Array3 ReLU
        let input = arr3(&[[[1.2, -4.3], [-2.1, 4.3]], [[5.2, 6.1], [7.6, -1.8]], [
            [9.3, 0.0],
            [1.2, 3.4],
        ]]);

        let mut relu = Relu::new("ReLU".into());

        let output = relu.apply(&input.clone().into_dyn().view());

        relu.update_params(&output.view());

        let n_params = relu.num_params();

        let n_multiplications = relu.num_muls(&input.into_dyn().view());

        //  let ReLU::<f32> {
        //      output: x,
        //      n_params,
        //      n_multiplications,
        //      name,
        //  } = relu_layer(&input);

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

    #[test]
    fn relu_test1() {
        // Array1 ReLU

        let input = arr1(&[-4., -3.4, 6., 7., 1., -3.]).into_dyn();

        let mut relu = Relu::new("ReLU".into());

        let output = relu.apply(&input.clone().into_dyn().view());

        relu.update_params(&output.view());

        let n_params = relu.num_params();

        let n_multiplications = relu.num_muls(&input.into_dyn().view());

        // let ReLU::<f32> {
        //     output: x,
        //     n_params,
        //     n_multiplications,
        //     name,
        // } = relu_layer(&input);

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
