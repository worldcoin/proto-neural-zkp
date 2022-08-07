use ndarray::{s, Array3, ArrayD, ArrayViewD, Ix3};
use ndarray_stats::QuantileExt;

use super::Layer;

pub struct MaxPool<T> {
    pub output:            Array3<T>,
    pub n_params:          i32,
    pub n_multiplications: usize,
    pub name:              String,
}

pub struct MaxPooling {
    kernel_side: usize,
    name:        String,
}

impl MaxPooling {
    #[must_use]
    pub fn new(name: String, kernel_side: usize) -> MaxPooling {
        MaxPooling { name, kernel_side }
    }
}

impl Layer for MaxPooling {
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

    fn num_muls(&self, input: &ArrayViewD<f32>) -> usize {
        input.len()
    }
}

// TODO: Generalize for more dimensions and number types
// @param input: h, w, c (3D Array)
// h - height
// w - width
// c - channels
// @param s - square filter side length (bigger -> more downsampling -> less
// definition) output: Array3 -> Downsampled input where biggest value in filter
// prevails
pub fn max_pooling_layer(input: &Array3<f32>, s: usize) -> MaxPool<f32> {
    let (h, w, c) = input.dim();

    assert!(h % s == 0, "Height must be divisible by s!");
    assert!(w % s == 0, "Width must be divisible by s!");

    let mut output = Array3::<f32>::zeros((h / s, w / s, c));

    // TODO: turn loops into iterators and parallelize with rayon or
    // ndarray::parallel
    // let h_iter = (0..h).into_par_iter().filter(|x| x % s == 0);
    for i in (0..h).step_by(s) {
        for j in (0..w).step_by(s) {
            for k in 0..c {
                let a = input.slice(s![i..i + s, j..j + s, k]);
                // https://docs.rs/ndarray-stats/latest/ndarray_stats/trait.QuantileExt.html#tymethod.max
                output[[i / s, j / s, k]] = *a.max().unwrap();
            }
        }
    }

    let n_params = 0;
    let n_multiplications = input.len();

    MaxPool {
        output,
        n_params,
        n_multiplications,
        name: String::from("max-pool"),
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

        let maxpool = MaxPooling::new("max-pool".into(), 2);

        let output = maxpool
            .apply(&input.clone().into_dyn().view())
            .into_dimensionality::<Ix3>()
            .unwrap();

        let n_params = maxpool.num_params();

        let n_multiplications = maxpool.num_muls(&input.into_dyn().view());

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
