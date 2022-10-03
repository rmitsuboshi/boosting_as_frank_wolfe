mod cv;


use crate::cv::cross_validation;


const PATH: &str = "./fold_dataset";
fn main() {
    cross_validation(PATH);
}


