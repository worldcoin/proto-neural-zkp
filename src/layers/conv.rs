use ndarray::{s, Array, Array3, Array4};

#[allow(clippy::module_name_repetitions)]
pub struct Conv2D<T> {
    pub output:            Array3<T>,
    pub n_params:          usize,
    pub n_multiplications: usize,
    pub name:              String,
}

#[must_use]
pub fn convolution(input: &Array3<f32>, kernels: &Array4<f32>) -> Conv2D<f32> {
    // height, width, channels
    let (h, w, c) = input.dim();

    // output channels, kernel height, kernel width, input channels
    let (c_out, hf, wf, c_in) = kernel.dim();

    // input channels must match
    assert_eq!(c, c_in);

    // height and width of kernel must be an uneven number
    assert!(hf % 2 == 1);
    assert!(wf % 2 == 1);

    let dh = hf / 2;
    let dw = wf / 2;

    let mut output = Array3::zeros((h - 2 * dh, w - 2 * dw, c_out));
    let mut alen = 0;

    // run convolution
    // go over image height - kernel padding (2*dh)
    for i in dh..h - dh {
        //  go over image width - kernel padding (2*dw)
        for j in dw..w - dw {
            // kernel slice
            let a = &input.slice(s![i - dh..i + dh + 1, j - dw..j + dw + 1, ..]);
            alen = a.len();
            // for output channels (number of kernels)
            for k in 0..c_out {
                // filter channel
                let b = &kernel.slice(s![k, .., .., ..]);
                // apply filter on kernel slice
                output[[i - dh, j - dw, k]] = (a * b).sum();
            }
        }
    }

    let n_params = kernel.len() as i32;
    let name = String::from(format!("conv {}x{}x{}x{}", c_out, hf, wf, c_in));
    let n_multiplications = alen * c_out * (w - 2 * dw) * (h - 2 * dh);

    Conv2D {
        output,
        n_params,
        n_multiplications,
        name,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{array, stack};
    use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
    use rand::rngs::StdRng;

    #[test]
    fn test_small() {
        let input = array![
            [
                [0.51682377_f32],
                [-2.3552072],
                [-0.120499134],
                [2.3132505],
                [-3.470844]
            ],
            [[-1.1741579], [3.4295654], [-1.2318683], [-1.9749749], [
                -0.8161392
            ]],
            [[4.7562046], [-2.8918338], [2.308525], [2.6111293], [
                -1.0765815
            ]],
            [[-4.1224194], [3.022316], [-4.5339823], [4.2970715], [
                2.6773367
            ]],
            [[-4.289216], [-3.3795083], [-2.651745], [-1.1392272], [
                3.9378529
            ]]
        ];
        let kernel: Array4<f32> = array![
            [[1.0336475_f32], [-4.7104144], [-0.24099827]],
            [[4.626501], [-6.941688], [-2.3483157]],
            [[6.859131], [-2.4637365], [-3.9499497]]
        ]
        .into_shape((1, 3, 3, 1))
        .unwrap();
        let expected = array![
            [[15.940444], [-9.205237], [13.396301]],
            [[1.7727833], [-10.784569], [-48.952152]],
            [[-22.043327], [8.725433], [-97.68271]]
        ];

        let result = convolution(&input, &kernel);

        let delta = (result.output - &expected);
        let max_error = delta.into_iter().map(f32::abs).fold(0.0, f32::max);
        dbg!(max_error);
        assert!(max_error < 10.0 * f32::EPSILON);
    }

    #[test]
    fn conv_test() {
        // man of culture fixed randomness seed
        let seed = 694201337;
        let mut rng = StdRng::seed_from_u64(seed);

        let input = Array3::random_using((120, 80, 3), Uniform::<f32>::new(-5., 5.), &mut rng);
        let kernel = Array4::random_using((32, 5, 5, 3), Uniform::<f32>::new(-10., 10.), &mut rng);

        // x is the result of  conv(input, kernel)
        let Conv2D::<f32> {
            output: x,
            n_params,
            n_multiplications,
            name,
        } = convolution(&input, &kernel);

        assert_eq!(x.dim(), (116, 76, 32));

        let (dim_x, dim_y, dim_z) = x.dim();

        println!(
            "# of parameters: {}\n
    output dim: {}x{}x{}\n
    # of multiplications: {}
    {} output:\n
    {}",
            n_params, dim_x, dim_y, dim_z, n_multiplications, name, x
        );
    }
}
