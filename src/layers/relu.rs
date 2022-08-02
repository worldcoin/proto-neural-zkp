use ndarray::ArrayD;

pub struct ReLU<T> {
    pub output:            ArrayD<T>,
    pub n_params:          i32,
    pub n_multiplications: i32,
    pub name:              String,
}

fn relu(v: f32) -> f32 {
    if v >= 0.0 {
        v
    } else {
        0.0
    }
}

pub fn relu_layer(input: ArrayD<f32>) -> ReLU<f32> {
    let output = input.mapv(relu);
    let n_params = output.len() as i32;
    let n_multiplications = 0;

    ReLU {
        output,
        n_params,
        n_multiplications,
        name: String::from("ReLU"),
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use ndarray::{arr1, arr3, Ix1, Ix3};

    #[test]

    fn relu_test3() {
        // Array3 ReLU
        let input = arr3(&[[[1.2, -4.3], [-2.1, 4.3]], [[5.2, 6.1], [7.6, -1.8]], [
            [9.3, 0.0],
            [1.2, 3.4],
        ]])
        .into_dyn();

        let ReLU::<f32> {
            output: x,
            n_params,
            n_multiplications,
            name,
        } = relu_layer(input);

        let result = x.clone();

        assert_eq!(
            result,
            arr3(&[[[1.2, 0.0], [0.0, 4.3]], [[5.2, 6.1], [7.6, 0.0]], [
                [9.3, 0.0],
                [1.2, 3.4],
            ]])
            .into_dyn()
        );

        let size: (usize, usize, usize) = (3, 2, 2);

        assert_eq!(result.into_dimensionality::<Ix3>().unwrap().dim(), size);

        println!(
            "
        {} 
        # of parameters: {}
        output dim: {}x1
        # of ops: {}
        output:\n{}",
            name, n_params, n_params, n_multiplications, x
        );
    }

    #[test]
    fn relu_test1() {
        // Array1 ReLU

        let input = arr1(&[-4., -3.4, 6., 7., 1., -3.]).into_dyn();

        let ReLU::<f32> {
            output: x,
            n_params,
            n_multiplications,
            name,
        } = relu_layer(input);

        let result = x.clone();

        assert_eq!(result, arr1(&[0., 0., 6., 7., 1., 0.]).into_dyn());

        assert_eq!(result.into_dimensionality::<Ix1>().unwrap().len(), 6);

        println!(
            "
        {} 
        # of parameters: {}
        output dim: {}x1
        # of ops: {}
        output:\n{}",
            name, n_params, n_params, n_multiplications, x
        );
    }
}
