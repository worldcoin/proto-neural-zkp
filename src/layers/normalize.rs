use super::LayerJson;
use ndarray::{ArrayD, ArrayViewD};
use serde::Serialize;

use super::Layer;

#[derive(Clone, Serialize)]
pub struct Normalize {
    name:        String,
    input_shape: Vec<usize>,
}

impl Normalize {
    #[must_use]
    pub fn new(input_shape: Vec<usize>) -> Normalize {
        Normalize {
            name: "normalize".into(),
            input_shape,
        }
    }
}

impl Layer for Normalize {
    fn box_clone(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }

    fn apply(&self, input: &ArrayViewD<f32>) -> ArrayD<f32> {
        let input = input.clone().mapv(|x| x as i128);
        let norm = f32::sqrt(input.mapv(|x| x.pow(2)).sum() as f32);
        input.mapv(|x| x as f32 / norm).into_dyn()
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
        let mut muls = 1;

        for i in self.input_shape() {
            muls *= i;
        }

        1 + muls
    }

    fn output_shape(&self) -> Vec<usize> {
        self.input_shape.clone()
    }

    fn to_json(&self) -> LayerJson {
        LayerJson::Normalize {
            input_shape: self.input_shape(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{arr1, array};

    #[test]
    fn normalize_test() {
        let input = arr1(&[
            -6276474000.,
            8343393300.,
            8266027500.,
            -7525360600.,
            7814137000.,
        ]);

        let normalize = Normalize::new(vec![5]);

        let output = normalize.apply(&input.into_dyn().view());

        let expected = array![-0.36541474, 0.4857503, 0.48124605, -0.43812463, 0.4549371];

        let delta = &output - &expected;

        let max_error = delta.into_iter().map(f32::abs).fold(0.0, f32::max);

        assert!(max_error < 10.0 * f32::EPSILON);

        println!("{}", output);
    }
}
