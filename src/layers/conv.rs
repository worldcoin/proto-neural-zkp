use ndarray::{s, Array, Array3, Array4, ArrayD, ArrayViewD, Ix3};
use serde::Serialize;

use super::{Layer, LayerJson};

#[derive(Clone, Serialize)]
pub struct Convolution {
    kernel:      Array4<f32>,
    name:        String,
    input_shape: Vec<usize>,
}

impl Convolution {
    #[must_use]
    pub fn new(kernel: Array4<f32>, input_shape: Vec<usize>) -> Convolution {
        let (c_out, hf, wf, c_in) = kernel.dim();
        let name = format!("conv {}x{}x{}x{}", c_out, hf, wf, c_in);

        Convolution {
            kernel,
            name,
            input_shape,
        }
    }
}

impl Layer for Convolution {
    fn box_clone(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
    
    fn apply(&self, input: &ArrayViewD<f32>) -> ArrayD<f32> {
        // height, width, channels
        let input = input.clone().into_dimensionality::<Ix3>().unwrap();
        let (h, w, c) = input.dim();

        // output channels, kernel height, kernel width, input channels
        let (c_out, hf, wf, c_in) = self.kernel.dim();

        assert_eq!(c, c_in, "input channels must match");

        assert!(hf % 2 == 1, "height of the kernel must be an odd number");
        assert!(wf % 2 == 1, "width of the kernel must be an odd number");

        let window_dim = (hf, wf, c_in);
        let output_shape = (h - hf + 1, w - wf + 1);
        let mut output = Array3::zeros((output_shape.0, output_shape.1, c_out));

        for i in 0..c_out {
            let mut output_mut = output.slice_mut(s![.., .., i]);
            let kernel = self.kernel.slice(s![i, .., .., ..]);
            let values = input
                .windows(window_dim)
                .into_iter()
                .map(|w| (&w * &kernel).sum());
            let values = Array::from_iter(values)
                .into_shape(output_shape)
                .expect("Kernel result dimensions mismatch");
            output_mut.assign(&values);
        }
        output.into_dyn()
    }

    fn input_shape(&self) -> Vec<usize> {
        self.input_shape.clone()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn num_params(&self) -> usize {
        self.kernel.len()
    }

    fn num_muls(&self) -> usize {
        // output channels, kernel height, kernel width, input channels
        let (c_out, hf, wf, _) = self.kernel.dim();

        let output_shape = self.output_shape();

        output_shape[0] * output_shape[1] * c_out * hf * wf
    }

    fn output_shape(&self) -> Vec<usize> {
        let input_shape = self.input_shape();

        let h = input_shape[0];
        let w = input_shape[1];
        let c = input_shape[2];

        // output channels, kernel height, kernel width, input channels
        let (c_out, hf, wf, c_in) = self.kernel.dim();

        assert_eq!(c, c_in, "input channels must match");

        assert!(hf % 2 == 1, "height of the kernel must be an odd number");
        assert!(wf % 2 == 1, "width of the kernel must be an odd number");

        vec![h - hf + 1, w - wf + 1, c_out]
    }

    fn to_json(&self) -> LayerJson {
        LayerJson::Convolution {
            kernel:      self.kernel.clone().into(),
            input_shape: self.input_shape(),
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::*;

    #[test]
    fn test_small() {
        use ndarray::array;
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

        let conv = Convolution::new(kernel, vec![1, 5, 5, 1]);

        let result = conv.apply(&input.into_dyn().view());

        let delta = result - &expected;
        let max_error = delta.into_iter().map(f32::abs).fold(0.0, f32::max);
        dbg!(max_error);
        assert!(max_error < 10.0 * f32::EPSILON);
    }

    #[test]
    fn conv_test() {
        use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
        use rand::rngs::StdRng;
        // man of culture fixed randomness seed
        let seed = 694201337;
        let mut rng = StdRng::seed_from_u64(seed);

        let input = Array3::random_using((120, 80, 3), Uniform::<f32>::new(-5., 5.), &mut rng);
        let kernel = Array4::random_using((32, 5, 5, 3), Uniform::<f32>::new(-10., 10.), &mut rng);

        let conv = Convolution::new(kernel, vec![120, 80, 3]);

        let result = conv
            .apply(&input.into_dyn().view())
            .into_dimensionality::<Ix3>()
            .unwrap();

        assert_eq!(conv.output_shape(), vec![116, 76, 32]);

        let (dim_x, dim_y, dim_z) = result.dim();

        println!(
            "# of parameters: {}\n
    output dim: {}x{}x{}\n
    # of multiplications: {}
    {} output:\n
    {}",
            conv.num_params(),
            dim_x,
            dim_y,
            dim_z,
            conv.num_muls(),
            conv.name(),
            result
        );
    }
}
