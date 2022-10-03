use polars::prelude::*;
use lycaon::prelude::*;

use mlpboost::MLPBoost;

use std::fs::File;
use std::io::prelude::*;


// -------------------------------------------------------------
// Constant parameters.

/// Boosting parameters
const TOLERANCE: f64 = 0.01;

/// CAP_RATIOS.len()
const CAP_SIZE: usize = 5;

/// Capping parameters
const CAP_RATIOS: [f64; CAP_SIZE] = [0.1, 0.2, 0.3, 0.4, 0.5];


/// Number of fold size
const FOLD_SIZE: usize = 5;


/// Number of benchmark dataset
const SIZE: usize = 13;

/// File names for benchmark dataset
const BENCHMARKS: [&str; SIZE] = [
    // 20 folds, 2 datasets
    "image",
    "splice",

    // 100 folds, 11 datasets
    "banana",
    "breast_cancer",
    "diabetis",
    "flare_solar",
    "german",
    "heart",
    "ringnorm",
    "thyroid",
    "titanic",
    "twonorm",
    "waveform",
];
// -------------------------------------------------------------

fn df(train: String, test: String)
    -> (DataFrame, Series, DataFrame, Series)
{
    let mut data = CsvReader::from_path(train)
        .unwrap()
        .has_header(true)
        .finish()
        .unwrap();

    let train_y = data.drop_in_place("class")
        .unwrap();
    let train_x = data;


    let mut data = CsvReader::from_path(test)
        .unwrap()
        .has_header(true)
        .finish()
        .unwrap();

    let test_y = data.drop_in_place("class")
        .unwrap();
    let test_x = data;

    (train_x, train_y, test_x, test_y)
}


/// Run the cross validations,
pub(crate) fn cross_validation(path: &str) {
    let mut lpb_test   = Vec::with_capacity(SIZE);
    let mut mlpb_test  = Vec::with_capacity(SIZE);
    let mut erlpb_test = Vec::with_capacity(SIZE);
    for file in BENCHMARKS {
        println!("DATASET: {file}");
        let prefix = format!("{path}/{file}");




        println!("  Measuring train/test errors ...");

        let (lpb_r, mlpb_r, erlpb_r) = find_best_param(prefix.clone());

        println!(
            "\
            Best capping ratios over 0.1, 0.2, ..., 0.5:\n
            \tLPB:   {lpb_r:.2}\n\
            \tERLPB: {erlpb_r:.2}\n\
            \tMLPB:  {mlpb_r:.2}"
        );

        let train_name = format!("{prefix}_train.csv");
        let test_name = format!("{prefix}_test.csv");

        let (train_x, train_y, test_x, test_y) = df(train_name, test_name);

        let train_size = train_x.shape().0;

        let dtree = DTree::init(&train_x)
            .criterion(Criterion::Edge)
            .max_depth(2);

        // ---------------------------------------------------
        // LPBoost
        let mut lpb = LPBoost::init(&train_x)
            .nu(lpb_r * train_size as f64);

        let test_loss = test(
            &mut lpb, &dtree, &train_x, &train_y, &test_x, &test_y
        );

        lpb_test.push(test_loss);


        // ---------------------------------------------------
        // Our work
        let mut mlpb = MLPBoost::init(&train_x)
            .capping(mlpb_r * train_size as f64);

        let test_loss = test(
            &mut mlpb, &dtree, &train_x, &train_y, &test_x, &test_y
        );

        mlpb_test.push(test_loss);


        // ---------------------------------------------------
        // ERLPBoost
        let mut erlpb = ERLPBoost::init(&train_x)
            .nu(erlpb_r * train_size as f64);

        let test_loss = test(
            &mut erlpb, &dtree, &train_x, &train_y, &test_x, &test_y
        );

        erlpb_test.push(test_loss);
        println!("  done.");
    }
    println!("Writing result ...");
    save_to_csv("test_errors", lpb_test, mlpb_test, erlpb_test);
    println!("done!");
}


/// Train a boosting algorithm and measure the test error.
fn test<B>(booster: &mut B,
           dtree:   &DTree,
           train_x: &DataFrame,
           train_y: &Series,
           test_x:  &DataFrame,
           test_y:  &Series)
    -> f64
    where B: Booster<DTreeClassifier>,
{
    let f = booster.run(dtree, train_x, train_y, TOLERANCE);


    // ---------------------------------------------------------
    // Test loss
    // ---------------------------------------------------------
    let test_size = test_x.shape().0;


    let predictions = f.predict_all(test_x);


    test_y.i64().unwrap()
        .into_iter()
        .zip(predictions)
        .map(|(t, p)| if t.unwrap() != p { 1.0 } else { 0.0 })
        .sum::<f64>() / test_size as f64
}


fn save_to_csv(file:      &str,
               lpb_test:   Vec<f64>,
               mlpb_test:  Vec<f64>,
               erlpb_test: Vec<f64>)
{
    let size = lpb_test.len();

    let output = format!("./{file}.csv");
    let mut file = File::create(output).unwrap();

    let header = "lpb_test,erlpb_test,mlpb_test\n";
    file.write_all(header.as_bytes()).unwrap();

    for i in 0..size {
        let mut line = format!(
            "{lpb_ts},{erlpb_ts},{mlpb_ts}",
            lpb_ts   = lpb_test[i],

            erlpb_ts = erlpb_test[i],
            mlpb_ts  = mlpb_test[i],
        );
        if i + 1 != size {
            line = format!("{line}\n");
        }

        file.write_all(line.as_bytes()).unwrap();
    }
}

fn find_best_param(prefix: String) -> (f64, f64, f64) {
    let mut test_lpb   = Vec::with_capacity(FOLD_SIZE);
    let mut test_erlpb = Vec::with_capacity(FOLD_SIZE);
    let mut test_mlpb  = Vec::with_capacity(FOLD_SIZE);


    for ratio in CAP_RATIOS {
        let mut lpb_loss   = 0.0;
        let mut mlpb_loss  = 0.0;
        let mut erlpb_loss = 0.0;

        for k in 1..=FOLD_SIZE {
            let train_name = format!("{prefix}_fold{k}_train.csv");
            let test_name = format!("{prefix}_fold{k}_test.csv");


            let (train_x, train_y, test_x, test_y) = df(train_name, test_name);

            let train_size = train_x.shape().0;


            let dtree = DTree::init(&train_x)
                .criterion(Criterion::Edge)
                .max_depth(2);


            // ---------------------------------------------------
            // LPBoost
            let mut lpb = LPBoost::init(&train_x)
                .nu(ratio * train_size as f64);

            lpb_loss += test(
                &mut lpb, &dtree, &train_x, &train_y, &test_x, &test_y
            );


            // ---------------------------------------------------
            // Our work
            let mut mlpb = MLPBoost::init(&train_x)
                .capping(ratio * train_size as f64);

            mlpb_loss += test(
                &mut mlpb, &dtree, &train_x, &train_y, &test_x, &test_y
            );


            // ---------------------------------------------------
            // ERLPBoost
            let mut erlpb = ERLPBoost::init(&train_x)
                .nu(ratio * train_size as f64);

            erlpb_loss += test(
                &mut erlpb, &dtree, &train_x, &train_y, &test_x, &test_y
            );
        }

        test_lpb.push(lpb_loss / FOLD_SIZE as f64);
        test_mlpb.push(mlpb_loss / FOLD_SIZE as f64);
        test_erlpb.push(erlpb_loss / FOLD_SIZE as f64);
    }


    println!(
        "\n\
        -------------------------------\n\
         RATIO |   LPB | ERLPB |  MLPB \n\
        -------------------------------\
        "
    );
    for (i, ratio) in CAP_RATIOS.iter().enumerate() {

        let ratio = format!("{:>.1}", ratio);
        let lpb = format!("{:>.2}", test_lpb[i]);
        let erlpb = format!("{:>.2}", test_erlpb[i]);
        let mlpb = format!("{:>.2}", test_mlpb[i]);
        println!(
            " {ratio:>width$} | {lpb:>width$} | {erlpb:>width$} | {mlpb:>width$}",
            width = 5
        );
    }
    println!("-------------------------------");


    let lpb_best_ratio = test_lpb.into_iter()
        .zip(CAP_RATIOS)
        .min_by(|(a, _), (b, _)| a.partial_cmp(&b).unwrap())
        .unwrap().1;
    let mlpb_best_ratio = test_mlpb.into_iter()
        .zip(CAP_RATIOS)
        .min_by(|(a, _), (b, _)| a.partial_cmp(&b).unwrap())
        .unwrap().1;
    let erlpb_best_ratio = test_erlpb.into_iter()
        .zip(CAP_RATIOS)
        .min_by(|(a, _), (b, _)| a.partial_cmp(&b).unwrap())
        .unwrap().1;


    (lpb_best_ratio, mlpb_best_ratio, erlpb_best_ratio)
}



