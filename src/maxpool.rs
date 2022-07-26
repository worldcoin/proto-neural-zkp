use ndarray::{s, Array3};
// use ndarray::parallel::prelude::*;
// required for finding element-wise maximum in array a
use ndarray_stats::QuantileExt;

// @param input: h, w, c
// h - height
// w - width
// c - channels
// @param s - square filter side length (bigger -> more downsampling -> less
// definition) output: Array3 -> Downsampled input where biggest value in filter
// prevails
pub fn max_pooling_layer(input: Array3<f32>, s: usize) -> (Array3<f32>, i32, usize, String) {
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

    return (
        output,
        n_params,
        n_multiplications,
        String::from("max-pool"),
    );
}
