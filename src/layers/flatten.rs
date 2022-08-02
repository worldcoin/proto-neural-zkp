use ndarray::{Array1, Array3};

pub struct Flatten<T> {
    pub output:            Array1<T>,
    pub n_params:          i32,
    pub n_multiplications: i32,
    pub name:              String,
}

pub fn flatten_layer(input: Array3<f32>) -> Flatten<f32> {
    let n_params = 0;
    let n_multiplications = 0;
    let output = input.into_iter().collect();

    Flatten {
        output,
        n_params,
        n_multiplications,
        name: String::from("flatten"),
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
    use rand::rngs::StdRng;

    #[test]
    fn flatten_test() {
        let seed = 694201337;
        let mut rng = StdRng::seed_from_u64(seed);

        let input = Array3::random_using((27, 17, 32), Uniform::<f32>::new(-5.0, 5.0), &mut rng);

        let Flatten::<f32> {
            output: x,
            n_params,
            n_multiplications,
            name,
        } = flatten_layer(input);

        assert_eq!(x.len(), 14688);

        println!(
            "
        {} \n
        # of parameters: {}\n
        output dim: {} \n
        # of ops: {}\n
        output:\n
        {}",
            name,
            n_params,
            x.len(),
            n_multiplications,
            x
        );
    }
}
