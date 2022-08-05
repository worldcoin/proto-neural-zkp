use ndarray::{Array1, ArrayD, ArrayViewD};

use super::Layer;

pub struct Flatten<T> {
    pub output:            Array1<T>,
    pub n_params:          i32,
    pub n_multiplications: i32,
    pub name:              String,
}

pub struct Flat {
    name: String,
}

impl Flat {
    #[must_use]
    pub fn new(name: String) -> Flat {
        Flat { name }
    }
}

impl Layer for Flat {
    fn apply(&self, input: &ArrayViewD<f32>) -> ArrayD<f32> {
        let output = Array1::from_iter(input.iter().map(|&x| x));
        output.into_dyn()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn num_muls(&self, input: &ArrayViewD<f32>) -> usize {
        0
    }

    fn num_params(&self) -> usize {
        0
    }
}

pub fn flatten_layer(input: &ArrayViewD<f32>) -> Flatten<f32> {
    let n_params = 0;
    let n_multiplications = 0;

    let output = Array1::from_iter(input.iter().map(|&x| x));

    Flatten {
        output,
        n_params,
        n_multiplications,
        name: String::from("flatten"),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::Array3;
    use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
    use rand::rngs::StdRng;

    #[test]
    fn flatten_test() {
        let seed = 694201337;
        let mut rng = StdRng::seed_from_u64(seed);

        let input = Array3::random_using((27, 17, 32), Uniform::<f32>::new(-5.0, 5.0), &mut rng);

        let flat = Flat::new("Flatten".into());

        let output = flat.apply(&input.clone().into_dyn().view());

        let n_multiplications = flat.num_muls(&input.clone().into_dyn().view());

        let n_params = flat.num_params();

        // let Flatten::<f32> {
        //     output: x,
        //     n_params,
        //     n_multiplications,
        //     name,
        // } = flatten_layer(&input.into_dyn().view());

        assert_eq!(output.len(), 14688);

        println!(
            "
        {} \n
        # of parameters: {}\n
        output dim: {} \n
        # of ops: {}\n
        output:\n
        {}",
            flat.name,
            n_params,
            output.len(),
            n_multiplications,
            output
        );
    }
}
