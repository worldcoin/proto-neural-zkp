#![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]
use ndarray::{s, Array3};
use ndarray_stats::QuantileExt;

pub struct MaxPool<T> {
    pub output:            Array3<T>,
    pub n_params:          i32,
    pub n_multiplications: usize,
    pub name:              String,
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
        let s = 2;

        let MaxPool::<f32> {
            output: x,
            n_params,
            n_multiplications,
            name,
        } = max_pooling_layer(&input, s);

        assert_eq!(x.dim(), (58, 38, 32));

        let (dim_x, dim_y, dim_z) = x.dim();

        println!(
            "
        {} \n
        # of parameters: {}\n
        output dim: {}x{}x{}\n
        # of ops: {}\n
        output:\n
        {}",
            name, n_params, dim_x, dim_y, dim_z, n_multiplications, x
        );
    }
}
