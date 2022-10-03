mod lpb_bad_classifier;
mod lpb_bad_base_learner;

use polars::prelude::*;
use lycaon::prelude::*;


use mlpboost::{Switch, FW, MLPBoost};

use lpb_bad_base_learner::BadBaseLearner;

fn main() {
    let tolerance = 1e-2;
    let (data, target) = dummy_sample_of_size(64);


    let mut booster = MLPBoost::init(&data)
        .switch(Switch::Edge)
        .strategy(FW::ShortStep)
        .nu(1.0);

    let bbl = BadBaseLearner::init(&data)
        .finish();


    let f = booster.run(&bbl, &data, &target, tolerance);

    println!("{f:?}");
}


fn dummy_sample_of_size(m: usize) -> (DataFrame, Series) {
    let one = vec![1_i64; m];
    let data = DataFrame::new(
        vec![ Series::new("dummy", &one) ]
    ).unwrap();

    let target = Series::new("class", &one);

    (data, target)
}
