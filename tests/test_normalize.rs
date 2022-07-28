use ndarray::arr1;
use neural_zkp::normalize::*;

#[test]
fn normalize_test() {
    let input = arr1(&[-6276474000, 8343393300, 8266027500, -7525360600, 7814137000]);

    let Normalize::<f64> {
        output: x,
        n_params,
        n_multiplications,
        name,
    } = normalize(input);

    println!("{}", x);
}
