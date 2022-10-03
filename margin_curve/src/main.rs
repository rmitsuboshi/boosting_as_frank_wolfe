mod benchmarks_io;

use rayon::prelude::*;
use lycaon::prelude::*;

// use mlpboost::{MLPBoost, FW};

use benchmarks_io::Benchmark;

use std::fs::File;
use std::io::prelude::*;


const TOLERANCE: f64 = 0.01;
const PATH: &str = "/home/mitsuboshi/dataset/benchmarks";

const BENCHMARKS: [&str; 13] = [
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

const IX: usize = 0;

fn main() {
    // for dataset in BENCHMARKS {
    BENCHMARKS.par_iter()
        .for_each(|dataset| {
            let file = format!("{PATH}/{dataset}");
            println!(
                "-----\n\
                 Running {file}"
            );

            let bm = Benchmark::new(&file);

            let (train_x, train_y) = bm.train_at(IX);
            let ( test_x,  test_y) = bm.test_at(IX);


            let size = train_x.shape().0 as f64;

            let nu = size * 0.1;


            let dtree = DTree::init(&train_x)
                .criterion(Criterion::Edge)
                .max_depth(2);


            // // ---------------------------------------------------
            // // MLPBoost (SS)
            // {
            //     let mut booster = MLPBoost::init(&train_x)
            //         .nu(nu)
            //         .strategy(FW::ShortStep);

            //     let (objvals, tests, times, lpb_calls, total_iter) = booster.monitor(
            //         &dtree, &train_x, &train_y, &test_x, &test_y, TOLERANCE
            //     );


            //     println!("[MLPB. ] Terminated at {total_iter}");

            //     let file = format!("./mlpb(ss)_{dataset}.csv");
            //     mlpb_save_to_csv(&file, objvals, tests, times, lpb_calls);
            // }

            // // ---------------------------------------------------
            // // MLPBoost (SS only)
            // {
            //     let mut booster = MLPBoost::init(&train_x)
            //         .nu(nu)
            //         .fw_only(true)
            //         .strategy(FW::ShortStep);

            //     let (objvals, tests, times, lpb_calls, total_iter) = booster.monitor(
            //         &dtree, &train_x, &train_y, &test_x, &test_y, TOLERANCE
            //     );


            //     println!("[MLPB. ] Terminated at {total_iter}");

            //     let file = format!("./mlpb(ss_only)_{dataset}.csv");
            //     mlpb_save_to_csv(&file, objvals, tests, times, lpb_calls);
            // }

            // // ---------------------------------------------------
            // // MLPBoost (PFW)
            // {
            //     let mut booster = MLPBoost::init(&train_x)
            //         .nu(nu)
            //         .strategy(FW::PairWise);

            //     let (objvals, tests, times, lpb_calls, total_iter) = booster.monitor(
            //         &dtree, &train_x, &train_y, &test_x, &test_y, TOLERANCE
            //     );


            //     println!("[MLPB. ] Terminated at {total_iter}");

            //     let file = format!("./mlpb(pfw)_{dataset}.csv");
            //     mlpb_save_to_csv(&file, objvals, tests, times, lpb_calls);
            // }

            // // ---------------------------------------------------
            // // MLPBoost (SS only)
            // {
            //     let mut booster = MLPBoost::init(&train_x)
            //         .nu(nu)
            //         .fw_only(true)
            //         .strategy(FW::PairWise);

            //     let (objvals, tests, times, lpb_calls, total_iter) = booster.monitor(
            //         &dtree, &train_x, &train_y, &test_x, &test_y, TOLERANCE
            //     );


            //     println!("[MLPB. ] Terminated at {total_iter}");

            //     let file = format!("./mlpb(pfw_only)_{dataset}.csv");
            //     mlpb_save_to_csv(&file, objvals, tests, times, lpb_calls);
            // }


            // ---------------------------------------------------
            // LPBoost
            {
                let mut booster = LPBoost::init(&train_x)
                    .nu(nu);

                let (objvals, tests, times, total_iter) = booster.monitor(
                    &dtree, &train_x, &train_y, &test_x, &test_y, TOLERANCE
                );


                println!("[LPB.  ] Terminated at {total_iter}");

                let file = format!("./lpb_{dataset}.csv");
                save_to_csv(&file, objvals, tests, times);
            }


            // ---------------------------------------------------
            // ERLPBoost
            {
                let mut booster = ERLPBoost::init(&train_x)
                    .nu(nu);

                let (objvals, tests, times, total_iter) = booster.monitor(
                    &dtree, &train_x, &train_y, &test_x, &test_y, TOLERANCE
                );


                println!("[ERLPB.] Terminated at {total_iter}");

                let file = format!("./erlpb_{dataset}.csv");
                save_to_csv(&file, objvals, tests, times);
            }
        })
    // }
}


fn save_to_csv(file: &str,
               objvals: Vec<f64>,
               tests: Vec<f64>,
               times: Vec<u128>)
{
    assert!(objvals.len() == times.len());
    let mut file = File::create(file).unwrap();

    let header = "objval,test_loss,time\n";
    file.write_all(header.as_bytes()).unwrap();

    let size = objvals.len();
    for i in 0..size {
        let mut line = format!(
            "{objval},{test},{time}",
            objval = objvals[i],
            test = tests[i],
            time = times[i],
        );

        if i + 1 != size {
            line = format!("{line}\n");
        }

        file.write_all(line.as_bytes()).unwrap();
    }
}


// fn mlpb_save_to_csv(file: &str,
//                     objvals: Vec<f64>,
//                     tests: Vec<f64>,
//                     times: Vec<u128>,
//                     lpb_calls: Vec<usize>)
// {
//     assert!(
//         objvals.len() == times.len() && times.len() == lpb_calls.len()
//     );
//     let mut file = File::create(file).unwrap();
// 
//     let header = "objval,test_loss,time,lpb_call\n";
//     file.write_all(header.as_bytes()).unwrap();
// 
//     let size = objvals.len();
//     for i in 0..size {
//         let mut line = format!(
//             "{objval},{test},{time},{lpb_call}",
//             objval = objvals[i],
//             test = tests[i],
//             time = times[i],
//             lpb_call = lpb_calls[i]
//         );
// 
//         if i + 1 != size {
//             line = format!("{line}\n");
//         }
// 
//         file.write_all(line.as_bytes()).unwrap();
//     }
// }
