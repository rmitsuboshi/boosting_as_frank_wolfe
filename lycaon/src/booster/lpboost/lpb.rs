//! This file defines `LPBoost` based on the paper
//! "Boosting algorithms for Maximizing the Soft Margin"
//! by Warmuth et al.
//! 
use polars::prelude::*;
use grb::prelude::*;
// use rayon::prelude::*;

use super::lp_model::LPModel;

use crate::{Classifier, CombinedClassifier};
use crate::BaseLearner;
use crate::Booster;


use std::cell::RefCell;



/// LPBoost struct. See [this paper](https://proceedings.neurips.cc/paper/2007/file/cfbce4c1d7c425baf21d6b6f2babe6be-Paper.pdf).
pub struct LPBoost {
    // Distribution over examples
    dist: Vec<f64>,

    // min-max edge of the new hypothesis
    gamma_hat: f64,

    // Tolerance parameter
    tolerance: f64,


    // Number of examples
    size: usize,


    // Capping parameter
    nu: f64,


    // GRBModel.
    lp_model: Option<RefCell<LPModel>>,


    terminated: usize,
}


impl LPBoost {
    /// Initialize the `LPBoost`.
    pub fn init(df: &DataFrame) -> Self {
        let (size, _) = df.shape();
        assert!(size != 0);


        let uni = 1.0 / size as f64;
        LPBoost {
            dist:      vec![uni; size],
            gamma_hat: 1.0,
            tolerance: uni,
            size,
            nu:        1.0,
            lp_model: None,

            terminated: 0_usize,
        }
    }


    /// This method updates the capping parameter.
    /// This parameter must be in `[1, sample_size]`.
    pub fn nu(mut self, nu: f64) -> Self {
        assert!(1.0 <= nu && nu <= self.size as f64);
        self.nu = nu;

        self
    }


    fn init_solver(&mut self) {
        let upper_bound = 1.0 / self.nu;

        assert!((0.0..=1.0).contains(&upper_bound));

        let lp_model = RefCell::new(LPModel::init(self.size, upper_bound));


        self.lp_model = Some(lp_model);
    }


    /// Set the tolerance parameter.
    /// Default is `1.0 / sample_size`/
    #[inline(always)]
    fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }


    /// Returns the terminated iteration.
    /// This method returns `0` before the boosting step.
    #[inline(always)]
    pub fn terminated(&self) -> usize {
        self.terminated
    }


    /// This method updates `self.dist` and `self.gamma_hat`
    /// by solving a linear program
    /// over the hypotheses obtained in past steps.
    #[inline(always)]
    fn update_distribution_mut<C>(&self,
                                  data: &DataFrame,
                                  target: &Series,
                                  h: &C)
        -> f64
        where C: Classifier
    {
        self.lp_model.as_ref()
            .unwrap()
            .borrow_mut()
            .update(data, target, h)
    }
}


impl<C> Booster<C> for LPBoost
    where C: Classifier,
{
    fn run<B>(&mut self,
              base_learner: &B,
              data: &DataFrame,
              target: &Series,
              tolerance: f64)
        -> CombinedClassifier<C>
        where B: BaseLearner<Clf = C>,
    {
        self.set_tolerance(tolerance);

        self.init_solver();

        let mut classifiers = Vec::new();

        self.terminated = usize::MAX;

        // Since the LPBoost does not have non-trivial iteration,
        // we run this until the stopping criterion is satisfied.
        loop {
            let h = base_learner.produce(data, target, &self.dist);

            // Each element in `margins` is the product of
            // the predicted vector and the correct vector

            let ghat = target.i64()
                .expect("The target class is not a dtype of i64")
                .into_iter()
                .enumerate()
                .map(|(i, y)| y.unwrap() as f64 * h.confidence(data, i))
                .zip(self.dist.iter())
                .map(|(yh, &d)| d * yh)
                .sum::<f64>();

            self.gamma_hat = ghat.min(self.gamma_hat);


            let gamma_star = self.update_distribution_mut(
                data, target, &h
            );


            classifiers.push(h);

            if gamma_star >= self.gamma_hat - self.tolerance {
                println!("Break loop at: {t}", t = classifiers.len());
                self.terminated = classifiers.len();
                break;
            }

            // Update the distribution over the training examples.
            self.dist = self.lp_model.as_ref()
                .unwrap()
                .borrow()
                .distribution();
        }


        let clfs = self.lp_model.as_ref()
            .unwrap()
            .borrow()
            .weight()
            .zip(classifiers)
            .filter(|(w, _)| *w != 0.0)
            .collect::<Vec<(f64, C)>>();


        CombinedClassifier::from(clfs)
    }
}




use std::time::Instant;
impl LPBoost {
    /// Returns the quadruple of 
    /// - A vector of objective values,
    /// - A vector of test errors
    /// - A vector of times,
    /// - Total number of iterations
    pub fn monitor<B, C>(&mut self,
                         base_learner: &B,
                         train_x: &DataFrame,
                         train_y: &Series,
                         test_x: &DataFrame,
                         test_y: &Series,
                         tolerance: f64)
        -> (Vec<f64>, Vec<f64>, Vec<u128>, usize)
        where B: BaseLearner<Clf = C>,
              C: Classifier + PartialEq + std::fmt::Debug,
    {
        self.set_tolerance(tolerance);
        self.init_solver();

        let mut classifiers = Vec::new();

        self.terminated = usize::MAX;


        let mut objvals = vec![-1.0_f64];
        let mut tests = vec![1.0_f64];
        let mut times = vec![0_u128];



        // Since the LPBoost does not have non-trivial iteration,
        // we run this until the stopping criterion is satisfied.
        loop {
            let now = Instant::now();
            let h = base_learner.produce(train_x, train_y, &self.dist);

            // Each element in `margins` is the product of
            // the predicted vector and the correct vector
            let ghat = train_y.i64()
                .expect("The target class is not a dtype of i64")
                .into_iter()
                .enumerate()
                .map(|(i, y)| y.unwrap() as f64 * h.confidence(train_x, i))
                .zip(self.dist.iter())
                .map(|(yh, &d)| d * yh)
                .sum::<f64>();


            self.gamma_hat = ghat.min(self.gamma_hat);


            let gamma_star = self.update_distribution_mut(
                train_x, train_y, &h
            );


            classifiers.push(h);

            if gamma_star >= self.gamma_hat - self.tolerance {
                println!("Break loop at: {t}", t = classifiers.len());
                self.terminated = classifiers.len();
                break;
            }

            // Update the distribution over the training examples.
            self.dist = self.lp_model.as_ref()
                .unwrap()
                .borrow_mut()
                .distribution();


            let diff = now.elapsed().as_millis() + times.last().unwrap_or(&0);
            times.push(diff);

            let weights = self.lp_model.as_ref()
                .unwrap()
                .borrow()
                .weight()
                .collect::<Vec<f64>>();


            objvals.push(lp_objval(
                train_x, train_y, &weights[..], &classifiers[..], self.nu
            ));

            tests.push(
                loss(test_x, test_y, &classifiers[..], &weights[..])
            );
        }

        (objvals, tests, times, self.terminated)
    }
}


fn lp_objval<C>(data: &DataFrame,
                target: &Series,
                weights: &[f64],
                classifiers: &[C],
                nu: f64)
    -> f64
    where C: Classifier
{
    let m = data.shape().0;

    let arr = target.i64().unwrap();

    let margins = arr.into_iter()
        .enumerate()
        .map(|(i, y)| {
            let y = y.unwrap() as f64;
            let p = prediction(i, data, classifiers, weights);
            y * p
        })
        .collect::<Vec<_>>();

    let mut indices = (0..m).into_iter().collect::<Vec<_>>();
    indices.sort_by(|&i, &j| margins[i].partial_cmp(&margins[j]).unwrap());

    let mut total = 1.0_f64;
    let max_weight = 1.0 / nu;

    let mut dist = vec![0.0_f64; m];

    for i in indices {
        if max_weight <= total {
            dist[i] = max_weight;
            total -= max_weight;
        } else {
            dist[i] = total;
            break;
        }
    }

    assert!(dist.iter().all(|d| (0.0..=1.0).contains(d)));
    assert!(
        (1.0 - dist.iter().sum::<f64>()).abs() < 1e-6
    );


    let objval = dist.into_iter()
        .zip(margins)
        .map(|(d, yh)| d * yh)
        .sum::<f64>();


    let checker = solve_primal(data, target, classifiers, nu);
    assert!((objval - checker).abs() < 1e-6);

    objval
}


fn solve_primal<C>(data: &DataFrame,
                   target: &Series,
                   classifiers: &[C],
                   nu: f64)
    -> f64
    where C: Classifier
{
    let mut env = Env::new("primal").unwrap();
    env.set(param::OutputFlag, 0).unwrap();

    let mut model = Model::with_env("primal_model", env).unwrap();


    let rho = add_ctsvar!(model, name: "rho", bounds: ..).unwrap();


    let n = classifiers.len();
    let weights = (0..n).map(|j| {
            let name = format!("w[{j}]");
            add_ctsvar!(model, name: &name, bounds: 0.0..)
                .unwrap()
        }).collect::<Vec<Var>>();

    let m = data.shape().0;
    let xi = (0..m).map(|i| {
            let name = format!("xi[{i}]");
            add_ctsvar!(model, name: &name, bounds: 0.0..)
                .unwrap()
        }).collect::<Vec<Var>>();


    // Set a constraint
    model.add_constr(&"sum_is_1", c!(weights.iter().grb_sum() == 1.0))
        .unwrap();

    target.i64()
        .unwrap()
        .into_iter()
        .zip(xi.iter().copied())
        .enumerate()
        .for_each(|(i, (y, x))| {
            let y = y.unwrap();

            let confidence = weights.iter()
                .copied()
                .zip(classifiers)
                .map(|(w, h)| w * h.confidence(data, i))
                .grb_sum();

            let margin = y * confidence;

            let name = format!("[{i}] th margin");

            model.add_constr(&name, c!(margin >= rho - x)).unwrap();
        });

    model.update().unwrap();

    let objective = rho - ((1.0 / nu) * xi.iter().grb_sum());
    // Set objective function
    model.set_objective(objective, Maximize).unwrap();
    model.update().unwrap();


    model.optimize().unwrap();


    let status = model.status().unwrap();
    if status != Status::Optimal {
        panic!("Status is {status:?}. Something wrong.");
    }


    model.get_attr(attr::ObjVal).unwrap()
}


fn prediction<C>(i: usize,
                 data: &DataFrame,
                 classifiers: &[C],
                 weights: &[f64])
    -> f64
    where C: Classifier
{
    classifiers.iter()
        .zip(weights)
        .map(|(h, w)| w * h.confidence(data, i))
        .sum::<f64>()
}


fn loss<C>(data: &DataFrame,
           target: &Series,
           classifiers: &[C],
           weights: &[f64])
    -> f64
    where C: Classifier
{
    let size = data.shape().0 as f64;

    target.i64()
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(i, y)| {
            let y = y.unwrap() as f64;
            let p = prediction(i, data, classifiers, weights).signum();
            if y * p > 0.0 { 0.0 } else { 1.0 }
        })
        .sum::<f64>()
        / size
}
