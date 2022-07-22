use ndarray::{Array, Array3, ArrayBase, Data, Ix1, Ix2, Ix3, Ix4, s, Array4};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand_isaac::isaac64::Isaac64Rng;
use ndarray_rand::rand::SeedableRng;

fn convolution (input: Array3<i32>, kernel: Array4<i32>) -> Array3<i32>
{

// height, width, channels
let (h, w, c) = input.dim();

// output channels, kernel height, kernel width, input channels
let (c_out, hf, wf, c_in) = kernel.dim();

// input channels must match
assert_eq!(c, c_in);

// height and width of kernel must be an uneven number
assert!(hf % 2 == 1);
assert!(wf % 2 == 1);

let dh = hf/2;
let dw = wf/2;

let mut output = Array::zeros((h - 2*dh, w - 2*dw, c_out));

// run convolution
// for height of kernel
for i in dh..h-dh {
    // for width of kernel
    for j in dw..w-dw {
        let a = input.slice(s![i-dh..i+dh+1, j-dw..j+dw+1, ..]);
        for k in 0..c_out {
            let b = kernel.slice(s![k, .., .., ..]);
            output[[i-dh, j-dw, k]] = (a*b).sum();
        }
    }
}

return output;
}


fn main() {
    // man of culture fixed randomness seed
    let seed = 694201337;
    let mut rng = Isaac64Rng::seed_from_u64(seed);

    //let arr1 = Array::<f64, _>::ones((3, 4, 5));
    //let shape = arr1.shape();
    let input = Array3::random((120, 80, 3), Uniform::<i32>::new(-5, 5));
    let kernel = Array4::random((32, 5, 5, 3), Uniform::<i32>::new(-10, 10));

    let conv = convolution(input, kernel);
    
    // assert_eq!(conv, Array3::<i32>::zeros((116, 76, 32)));
    // assert_eq!(shape, &[3,4,5]);
}
