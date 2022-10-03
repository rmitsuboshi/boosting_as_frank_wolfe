//! This file defines `ERLPBoost` based on the paper
//! "Entropy Regularized LPBoost"
//! by Warmuth et al.
//! 
use polars::prelude::*;
use grb::prelude::*;
// use rayon::prelude::*;


use crate::{Classifier, CombinedClassifier};
use crate::BaseLearner;
use crate::Booster;
use super::qp_model::QPModel;


use std::cell::RefCell;



/// ERLPBoost struct. 
/// See [this paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.141.1759&rep=rep1&type=pdf).
pub struct ERLPBoost {
    dist: Vec<f64>,

    // `gamma_hat` corresponds to $\min_{q=1, .., t} P^q (d^{q-1})$
    gamma_hat: f64,

    // `gamma_star` corresponds to $P^{t-1} (d^{t-1})$
    gamma_star: f64,
    // regularization parameter defined in the paper
    eta: f64,

    tolerance: f64,

    qp_model: Option<RefCell<QPModel>>,


    // an accuracy parameter for the sub-problems
    size: usize,
    nu: f64,


    terminated: usize,
}


impl ERLPBoost {
    /// Initialize the `ERLPBoost`.
    /// Use `data` for argument.
    /// This method does not care 
    /// whether the label is included in `data` or not.
    pub fn init(df: &DataFrame) -> Self {
        let size = df.shape().0;
        assert!(size != 0);


        // Set uni as an uniform weight
        let uni = 1.0 / size as f64;

        // Compute $\ln(size)$ in advance
        let ln_size = (size as f64).ln();


        // Set tolerance
        let tolerance = uni / 2.0;


        // Set regularization parameter
        let eta = 0.5_f64.max(2.0_f64 * ln_size / tolerance);

        // Set gamma_hat and gamma_star
        let gamma_hat  = 1.0;
        let gamma_star = f64::MIN;


        ERLPBoost {
            dist: vec![uni; size],
            gamma_hat,
            gamma_star,
            eta,
            tolerance,
            qp_model: None,
            size,
            nu: 1.0,

            terminated: 0_usize,
        }
    }


    fn init_solver(&mut self) {
        let upper_bound = 1.0 / self.nu;

        assert!((0.0..=1.0).contains(&upper_bound));


        let qp_model = RefCell::new(QPModel::init(
            self.eta, self.size, upper_bound
        ));

        self.qp_model = Some(qp_model);
    }


    /// Updates the capping parameter.
    pub fn nu(mut self, nu: f64) -> Self {
        assert!(1.0 <= nu && nu <= self.size as f64);
        self.nu = nu;
        self.regularization_param();

        self
    }


    /// Returns the break iteration.
    /// This method returns `0` before the `.run()` call.
    #[inline(always)]
    pub fn terminated(&self) -> usize {
        self.terminated
    }


    /// Setter method of `self.tolerance`
    #[inline(always)]
    fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance / 2.0;
    }


    /// Setter method of `self.eta`
    #[inline(always)]
    fn regularization_param(&mut self) {
        let ln_size = (self.size as f64 / self.nu).ln();


        self.eta = 0.5_f64.max(ln_size / self.tolerance);
    }



    /// Set `gamma_hat` and `gamma_star`.
    #[inline]
    fn set_gamma(&mut self) {
        self.gamma_hat = 1.0;
        self.gamma_star = -1.0;
    }


    /// Set all parameters in ERLPBoost.
    #[inline]
    fn init_params(&mut self, tolerance: f64) {
        self.set_tolerance(tolerance);

        self.regularization_param();

        self.set_gamma();
    }


    /// `max_loop` returns the maximum iteration
    /// of the Adaboost to find a combined hypothesis
    /// that has error at most `tolerance`.
    fn max_loop(&mut self) -> u64 {
        let size = self.size as f64;

        let mut max_iter = 4.0 / self.tolerance;


        let ln_size = (size / self.nu).ln();
        let temp = 8.0 * ln_size / self.tolerance.powi(2);


        max_iter = max_iter.max(temp);

        max_iter.ceil() as u64
    }
}


impl ERLPBoost {
    /// Update `self.gamma_hat`.
    /// `self.gamma_hat` holds the minimum value of the objective value.
    #[inline]
    fn update_gamma_hat_mut<C>(&mut self,
                               h: &C,
                               data: &DataFrame,
                               target: &Series)
        where C: Classifier,
    {
        let edge = target.i64()
            .expect("The target class is not a dtype i64")
            .into_iter()
            .zip(self.dist.iter().copied())
            .enumerate()
            .map(|(i, (y, d))| d * y.unwrap() as f64 * h.confidence(data, i))
            .sum::<f64>();


        let m = self.dist.len() as f64;
        let entropy = self.dist.iter()
            .copied()
            .map(|d| if d == 0.0 { 0.0 } else { d * d.ln() })
            .sum::<f64>() + m.ln();


        let obj_val = edge + (entropy / self.eta);

        self.gamma_hat = self.gamma_hat.min(obj_val);
    }


    /// Update `self.gamma_star`.
    /// `self.gamma_star` holds the current optimal value.
    fn update_gamma_star_mut<C>(&mut self,
                                classifiers: &[C],
                                data: &DataFrame,
                                target: &Series)
        where C: Classifier,
    {
        let max_edge = classifiers.iter()
            .map(|h|
                target.i64()
                    .expect("The target class is not a dtype i64")
                    .into_iter()
                    .zip(self.dist.iter().copied())
                    .enumerate()
                    .map(|(i, (y, d))|
                        d * y.unwrap() as f64 * h.confidence(data, i)
                    )
                    .sum::<f64>()
            )
            .reduce(f64::max)
            .unwrap();


        let entropy = self.dist.iter()
            .copied()
            .map(|d| d * d.ln())
            .sum::<f64>();


        let m = self.dist.len() as f64;
        self.gamma_star = max_edge + (entropy + m.ln()) / self.eta;
    }


    /// Updates `self.dist`
    /// This method repeatedly minimizes the quadratic approximation of 
    /// ERLPB. objective around current distribution `self.dist`.
    /// Then update `self.dist` as the optimal solution of 
    /// the approximate problem. 
    /// This method continues minimizing the quadratic objective 
    /// while the decrease of the optimal value is 
    /// greater than `self.sub_tolerance`.
    fn update_distribution_mut<C>(&mut self,
                                  data: &DataFrame,
                                  target: &Series,
                                  clf: &C)
        where C: Classifier,
    {
        self.qp_model.as_ref()
            .unwrap()
            .borrow_mut()
            .update(data, target, &mut self.dist[..], clf);

        self.dist = self.qp_model.as_ref()
            .unwrap()
            .borrow()
            .distribution();
    }
}


impl<C> Booster<C> for ERLPBoost
    where C: Classifier
{
    fn run<B>(&mut self,
              base_learner: &B,
              data: &DataFrame,
              target: &Series,
              tolerance: f64)
        -> CombinedClassifier<C>
        where B: BaseLearner<Clf = C>,
    {
        // Initialize all parameters
        self.init_params(tolerance);


        self.init_solver();


        // Get max iteration.
        let max_iter = self.max_loop();

        self.terminated = max_iter as usize;


        // This vector holds the classifiers
        // obtained from the `base_learner`.
        let mut classifiers = Vec::new();

        for step in 1..=max_iter {
            // Receive a hypothesis from the base learner
            let h = base_learner.produce(data, target, &self.dist[..]);


            // update `self.gamma_hat`
            self.update_gamma_hat_mut(&h, data, target);


            // Check the stopping criterion
            let diff = self.gamma_hat - self.gamma_star;
            if diff <= self.tolerance {
                println!("Break loop at: {step}");
                self.terminated = step as usize;
                break;
            }

            // At this point, the stopping criterion is not satisfied.

            // Update the parameters
            self.update_distribution_mut(data, target, &h);


            // Append a new hypothesis to `clfs`.
            classifiers.push(h);


            // update `self.gamma_star`.
            self.update_gamma_star_mut(&classifiers, data, target);
        }

        // Set the weights on the hypotheses
        // by solving a linear program
        let clfs = self.qp_model.as_ref()
            .unwrap()
            .borrow_mut()
            .weight()
            .zip(classifiers)
            .filter(|(w, _)| *w != 0.0)
            .collect::<Vec<(f64, C)>>();


        CombinedClassifier::from(clfs)
    }
}


use std::time::Instant;
impl ERLPBoost {
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
        // Initialize all parameters
        self.init_params(tolerance);

        self.init_solver();


        // Get max iteration.
        let max_iter = self.max_loop();

        self.terminated = max_iter as usize;


        // This vector holds the classifiers
        // obtained from the `base_learner`.
        let mut classifiers = Vec::new();


        let mut objvals = vec![-1.0_f64];
        let mut tests = vec![1.0_f64];
        let mut times = vec![0_u128];


        for step in 1..=max_iter {
            let now = Instant::now();
            // Receive a hypothesis from the base learner
            let h = base_learner.produce(train_x, train_y, &self.dist[..]);


            // update `self.gamma_hat`
            self.update_gamma_hat_mut(&h, train_x, train_y);


            // Check the stopping criterion
            let diff = self.gamma_hat - self.gamma_star;
            if diff <= self.tolerance {
                println!("Break loop at: {step}");
                self.terminated = step as usize;
                break;
            }


            // At this point, the stopping criterion is not satisfied.
            // Append a new hypothesis to `clfs`.


            // Update the parameters
            self.update_distribution_mut(train_x, train_y, &h);


            classifiers.push(h);


            // update `self.gamma_star`.
            self.update_gamma_star_mut(&classifiers, train_x, train_y);


            let diff = now.elapsed().as_millis() + times.last().unwrap_or(&0);
            times.push(diff);

            let weights = self.qp_model.as_ref()
                .unwrap()
                .borrow_mut()
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
            add_ctsvar!(model, name: &name, bounds: 0.0..1.0)
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


