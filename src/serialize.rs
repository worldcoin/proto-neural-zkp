#[cfg(test)]
pub mod tests {
    use ndarray::{arr2, ArcArray, IxDyn};
    use serde_json;
    use std::fs;

    #[test]
    fn serde_array() {
        let a = arr2(&[[3., 1., 2.2], [3.1, 4., 7.]]);
        let serial = serde_json::to_string(&a).unwrap();
        println!("Serde encode {:?} => {:?}", a, serial);
        let res = serde_json::from_str::<ArcArray<f32, _>>(&serial);
        println!("{:?}", res);
        assert_eq!(a, res.unwrap());
        let text = r##"{"v":1,"dim":[2,3],"data":[3,1,2.2,3.1,4,7]}"##;
        let b = serde_json::from_str::<ArcArray<f32, _>>(text);
        assert_eq!(a, b.unwrap());
    }

    #[test]
    fn serde_json_file() {
        let data = fs::read_to_string("./src/json/test.json").expect("Unable to read file");
        println!("{}", data);

        let res = serde_json::from_str::<ArcArray<i32, IxDyn>>(&data);

        println!("{:?}", res.unwrap().view());
    }
}
