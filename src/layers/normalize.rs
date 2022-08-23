#![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]
use ndarray::Array1;

pub struct Normalize<T> {
    pub output:            Array1<T>,
    pub n_params:          i32,
    pub n_multiplications: i32,
    pub name:              String,
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
pub mod test {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn normalize_test() {
        let input = arr1(&[-6276474000, 8343393300, 8266027500, -7525360600, 7814137000]);

        let Normalize::<f64> {
            output: x,
            n_params,
            n_multiplications,
            name,
        } = normalize(&input);

        println!("{}", x);
    }
}
