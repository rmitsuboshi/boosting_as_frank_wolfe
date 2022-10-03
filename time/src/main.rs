use polars::prelude::*;
use mlpboost::{FW, Switch, MLPBoost};

use lycaon::{
    Booster,
    Classifier,
    LPBoost,
    ERLPBoost,
    DTree,
    Criterion,
};


// Tolerance parameter.
const TOLERANCE: f64 = 0.01;

fn main() {
    let args = std::env::args().collect::<Vec<String>>();

    if args.len() < 4 {
        panic!(
            "[USAGE] ./time [input] \
            [lpb | erlpb | mlpb | mlpb(ss) | \
             mlpb(ss_only) | mlpb(pfw) | mlpb(pfw_only)] \
            [cap_ratio]"
        );
    }


    let input = &args[1];
    let booster = &args[2];

    let cap_ratio = args[3].trim().parse::<f64>()
        .expect(
            "Capping ratio should be an floating number.\n\
            [USAGE] ./time \
            [lpb | erlpb | mlpb | mlpb(ss) | \
             mlpb(ss_only) | mlpb(pfw) | mlpb(pfw_only)] \
            [cap_ratio]"
        );

    run_once(input, booster, cap_ratio);
}


fn run_once(input: &str, booster: &str, cap_ratio: f64) {

    // Get the training data.
    let mut df = CsvReader::from_path(input)
        .unwrap()
        .has_header(true)
        .finish()
        .unwrap();

    let train_size = df.shape().0 as f64;

    let target = df.drop_in_place("class").unwrap();
    let data = df;


    let dtree = DTree::init(&data)
        .criterion(Criterion::Edge)
        .max_depth(2);
    if booster == "lpb" {
        let mut booster = LPBoost::init(&data)
            .nu(cap_ratio * train_size);

        let _ = booster.run(&dtree, &data, &target, TOLERANCE);

    } else if booster == "erlpb" {
        let mut booster = ERLPBoost::init(&data)
            .nu(cap_ratio * train_size);

        let f = booster.run(&dtree, &data, &target, TOLERANCE);

        // DEBUG
        let err = f.predict_all(&data)
            .into_iter()
            .zip(target.i64().unwrap())
            .map(|(p, y)| if p != y.unwrap() { 1.0 } else { 0.0 })
            .sum::<f64>() / train_size;

        println!("train err: {err}");
    } else if booster == "mlpb(ss)" {
        let mut booster = MLPBoost::init(&data)
            .nu(cap_ratio * train_size)
            .switch(Switch::Edge)
            .strategy(FW::ShortStep);

        let f = booster.run(&dtree, &data, &target, TOLERANCE);

        let err = f.predict_all(&data)
            .into_iter()
            .zip(target.i64().unwrap())
            .map(|(p, y)| if p != y.unwrap() { 1.0 } else { 0.0 })
            .sum::<f64>() / train_size;

        println!("train err: {err}");
    } else if booster == "mlpb(ss_only)" {
        let mut booster = MLPBoost::init(&data)
            .nu(cap_ratio * train_size)
            .fw_only(true)
            .strategy(FW::ShortStep);

        let f = booster.run(&dtree, &data, &target, TOLERANCE);

        let err = f.predict_all(&data)
            .into_iter()
            .zip(target.i64().unwrap())
            .map(|(p, y)| if p != y.unwrap() { 1.0 } else { 0.0 })
            .sum::<f64>() / train_size;

        println!("train err: {err}");
    } else if booster == "mlpb(pfw)" {
        let mut booster = MLPBoost::init(&data)
            .nu(cap_ratio * train_size)
            .strategy(FW::PairWise);

        let f = booster.run(&dtree, &data, &target, TOLERANCE);

        let err = f.predict_all(&data)
            .into_iter()
            .zip(target.i64().unwrap())
            .map(|(p, y)| if p != y.unwrap() { 1.0 } else { 0.0 })
            .sum::<f64>() / train_size;

        println!("train err: {err}");
    } else if booster == "mlpb(pfw_only)" {
        let mut booster = MLPBoost::init(&data)
            .nu(cap_ratio * train_size)
            .fw_only(true)
            .strategy(FW::PairWise);

        let f = booster.run(&dtree, &data, &target, TOLERANCE);

        let err = f.predict_all(&data)
            .into_iter()
            .zip(target.i64().unwrap())
            .map(|(p, y)| if p != y.unwrap() { 1.0 } else { 0.0 })
            .sum::<f64>() / train_size;

        println!("train err: {err}");
    } else {
        panic!(
            "Invalid booster name.\n\
            [USAGE] ./time [input] \
            [lpb | erlpb | mlpb | mlpb(ss) | \
             mlpb(ss_only) | mlpb(pfw) | mlpb(pfw_only)] \
            [cap_ratio]"
        );
    }
}


