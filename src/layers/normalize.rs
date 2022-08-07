use ndarray::{Array1, ArrayD, ArrayViewD};

use super::Layer;

pub struct Normalize<T> {
    pub output:            Array1<T>,
    pub n_params:          i32,
    pub n_multiplications: i32,
    pub name:              String,
}

pub struct Normalization {
    name: String,
}

impl Normalization {
    #[must_use]
    pub fn new(name: String) -> Normalization {
        Normalization { name }
    }
}

impl Layer for Normalization {
    fn apply(&self, input: &ArrayViewD<f32>) -> ArrayD<f32> {
        let input = input.clone().mapv(|x| x as i128);
        let norm = f32::sqrt(input.mapv(|x| x.pow(2)).sum() as f32);
        input.mapv(|x| x as f32 / norm).into_dyn()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn num_muls(&self, input: &ArrayViewD<f32>) -> usize {
        1 + input.len()
    }

    fn num_params(&self) -> usize {
        0
    }
}

pub fn normalize(input: &Array1<i128>) -> Normalize<f64> {
    let n_params = 0;
    let n_multiplications = 1 + input.len() as i32;
    let norm = f64::sqrt(input.mapv(|x| x.pow(2)).sum() as f64);
    let output = input.mapv(|x| x as f64 / norm);

    Normalize {
        output,
        n_params,
        n_multiplications,
        name: String::from("normalize"),
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

        let normalize = Normalization::new("normalize".into());

        let output = normalize.apply(&input.into_dyn().view());

        let expected = array![-0.36541474, 0.4857503, 0.48124605, -0.43812463, 0.4549371];

        let delta = &output - &expected;

        let max_error = delta.into_iter().map(f32::abs).fold(0.0, f32::max);

        assert!(max_error < 10.0 * f32::EPSILON);

        // let Normalize::<f64> {
        //     output: x,
        //     n_params,
        //     n_multiplications,
        //     name,
        // } = normalize(&input);

        println!("{}", output);
    }
}
