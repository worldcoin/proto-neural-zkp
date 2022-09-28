use ndarray::{s, Array3, ArrayD, ArrayViewD, Ix3};
use ndarray_stats::QuantileExt;
use serde::Serialize;

use super::{Layer, LayerJson};

#[derive(Clone, Serialize)]
pub struct MaxPool {
    kernel_side: usize,
    name:        String,
    input_shape: Vec<usize>,
}

impl MaxPool {
    #[must_use]
    pub fn new(kernel_side: usize, input_shape: Vec<usize>) -> MaxPool {
        MaxPool {
            name: "max-pool".into(),
            kernel_side,
            input_shape,
        }
    }
}

impl Layer for MaxPool {
    fn box_clone(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

    fn apply(&self, input: &ArrayViewD<f32>) -> ArrayD<f32> {
        let input = input.clone().into_dimensionality::<Ix3>().unwrap();
        let (h, w, c) = input.dim();

        assert!(h % self.kernel_side == 0, "Height must be divisible by s!");
        assert!(w % self.kernel_side == 0, "Width must be divisible by s!");

        let mut output = Array3::<f32>::zeros((h / self.kernel_side, w / self.kernel_side, c));

        // TODO: turn loops into iterators and parallelize with rayon or
        // ndarray::parallel
        // let h_iter = (0..h).into_par_iter().filter(|x| x % s == 0);
        for i in (0..h).step_by(self.kernel_side) {
            for j in (0..w).step_by(self.kernel_side) {
                for k in 0..c {
                    let a = input.slice(s![i..i + self.kernel_side, j..j + self.kernel_side, k]);
                    // https://docs.rs/ndarray-stats/latest/ndarray_stats/trait.QuantileExt.html#tymethod.max
                    output[[i / self.kernel_side, j / self.kernel_side, k]] = *a.max().unwrap();
                }
            }
        }

        output.into_dyn()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn num_params(&self) -> usize {
        0
    }

    fn num_muls(&self) -> usize {
        let mut muls = 1;

        for i in self.input_shape() {
            muls *= i;
        }

        muls
    }

    fn output_shape(&self) -> Vec<usize> {
        let input_shape = self.input_shape();

        let h = input_shape[0];
        let w = input_shape[1];
        let c = input_shape[2];

        assert!(h % self.kernel_side == 0, "Height must be divisible by s!");
        assert!(w % self.kernel_side == 0, "Width must be divisible by s!");

        vec![w / self.kernel_side, h / self.kernel_side, c]
    }

    fn input_shape(&self) -> Vec<usize> {
        self.input_shape.clone()
    }

    fn to_json(&self) -> LayerJson {
        LayerJson::MaxPool {
            window:      self.kernel_side,
            input_shape: self.input_shape(),
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
    use rand::rngs::StdRng;

    #[test]
    fn maxpool_test() {
        let seed = 694201337;
        let mut rng = StdRng::seed_from_u64(seed);

        let input = Array3::random_using((116, 76, 32), Uniform::<f32>::new(-5.0, 5.0), &mut rng);

        let maxpool = MaxPool::new(2, vec![126, 76, 32]);

        let output = maxpool
            .apply(&input.into_dyn().view())
            .into_dimensionality::<Ix3>()
            .unwrap();

        let n_params = maxpool.num_params();

        let n_multiplications = maxpool.num_muls();

        assert_eq!(output.dim(), (58, 38, 32));

        let (dim_x, dim_y, dim_z) = output.dim();

        println!(
            "
        {} \n
        # of parameters: {}\n
        output dim: {}x{}x{}\n
        # of ops: {}\n
        output:\n
        {}",
            maxpool.name, n_params, dim_x, dim_y, dim_z, n_multiplications, output
        );
    }
}
