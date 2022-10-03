//! Our proposed algorithm
use polars::prelude::*;

use lycaon::{
    Classifier,
    CombinedClassifier,
    BaseLearner,
    Booster
};

use crate::lp_model::*;
use std::cell::RefCell;


#[derive(Clone, Copy)]
pub enum FW {
    Classic,
    ShortStep,
    LineSearch,
    AwayStep,
    PairWise,
}


#[derive(Clone, Copy)]
pub enum Switch {
    Edge,
    ObjVal,
}


/// Defines `MLPBoost`.
pub struct MLPBoost {
    // Regularization parameter.
    eta: f64,

    // number of examples
    size: usize,

    // Tolerance parameter
    tolerance: f64,


    // Capping parameter.
    // This parameter must be in the interval `[1.0, size]`.
    // Each `dist[i]` is at most `1.0 / nu`.
    nu: f64,


    switch: Switch,


    strategy: FW,


    // GRBModel
    lp_model: Option<RefCell<LPModel>>,


    fw_only: bool,

    pub(self) terminated: usize,
    pub(self) lpb_call: usize,
}


impl MLPBoost {
    /// Initialize the `MLPBoost`.
    pub fn init(df: &DataFrame) -> Self {
        let size = df.shape().0;
        assert!(size != 0);
        let uni = 1.0 / size as f64;


        let eta = 2.0 * (size as f64).ln() / uni;


        MLPBoost {
            eta,
            size,

            tolerance: uni,
            nu: 1.0,

            switch: Switch::ObjVal,

            strategy: FW::ShortStep,

            lp_model: None,

            fw_only: false,

            terminated: 0_usize,
            lpb_call: 0_usize,
        }
    }


    /// Set the FW stragety.
    pub fn strategy(mut self, strategy: FW) -> Self {
        self.strategy = strategy;

        self
    }


    pub fn switch(mut self, switch: Switch) -> Self {
        self.switch = switch;

        self
    }


    /// This method updates the capping parameter.
    pub fn nu(mut self, nu: f64) -> Self {
        assert!(1.0 <= nu && nu <= self.size as f64);
        self.nu = nu;

        self.regularization_param();

        self
    }


    /// Setter method for `fw_only`.
    pub fn fw_only(mut self, fw_only: bool) -> Self {
        self.fw_only = fw_only;

        self
    }


    /// Update set_tolerance parameter `tolerance`.
    #[inline(always)]
    fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance / 2.0;
        self.regularization_param();
    }


    /// Update regularization parameter.
    /// (the regularization parameter on
    ///  `self.tolerance` and `self.nu`.)
    #[inline(always)]
    fn regularization_param(&mut self) {
        let ln_part = (self.size as f64 / self.nu).ln();
        self.eta = ln_part / self.tolerance;
    }


    /// returns the maximum iteration of the CERLPBoost
    /// to find a combined hypothesis that has error at most `tolerance`.
    pub fn max_loop(&mut self, tolerance: f64) -> usize {
        if 2.0 * self.tolerance != tolerance {
            self.set_tolerance(tolerance);
        }

        let m = self.size as f64;


        let max_iter = 8.0 * (m / self.nu).ln()
            / self.tolerance.powi(2);

        max_iter.ceil() as usize
    }


    /// Returns the iteration that this algorithm terminated.
    pub fn terminated(&self) -> usize {
        self.terminated
    }


    /// Returns the number of lp update.
    pub fn lpb_call(&self) -> usize {
        self.lpb_call
    }


    /// Returns the logarithmic distribution for the given weights.
    fn log_dist_at<C>(&self,
                      data: &DataFrame,
                      target: &Series,
                      classifiers: &[C],
                      weights: &[f64])
        -> Vec<f64>
        where C: Classifier
    {
        // Assign the logarithmic distribution in `dist`.
        target.i64()
            .expect("The target is not a dtype i64")
            .into_iter()
            .enumerate()
            .map(|(i, y)| {
                let p = prediction(i, data, classifiers, weights);
                - self.eta * y.unwrap() as f64 * p
            })
            .collect::<Vec<_>>()
    }


    /// Project the logarithmic distribution `dist`
    /// onto the capped simplex.
    fn projection(&self, dist: &[f64]) -> Vec<f64> {
        // Sort the indices over `dist` in non-increasing order.
        let mut ix = (0..self.size).collect::<Vec<_>>();
        ix.sort_by(|&i, &j| dist[j].partial_cmp(&dist[i]).unwrap());


        let mut dist = dist.to_vec();


        let mut logsums: Vec<f64> = Vec::with_capacity(self.size);
        ix.iter().rev()
            .copied()
            .for_each(|i| {
                let logsum = logsums.last()
                    .map(|&v| {
                        let small = v.min(dist[i]);
                        let large = v.max(dist[i]);
                        large + (1.0 + (small - large).exp()).ln()
                    })
                    .unwrap_or(dist[i]);
                logsums.push(logsum);
            });

        let logsums = logsums.into_iter().rev();


        let ub = 1.0 / self.nu;
        let log_cap = self.nu.ln();

        let mut ix_with_logsum = ix.into_iter().zip(logsums).enumerate();

        while let Some((i, (i_sorted, logsum))) = ix_with_logsum.next() {
            let log_xi = (1.0 - ub * i as f64).ln() - logsum;
            // TODO replace this line into `get_unchecked`
            let d = dist[i_sorted];

            // Stopping criterion of this while loop
            if log_xi + d + log_cap <= 0.0 {
                dist[i_sorted] = (log_xi + d).exp();
                while let Some((_, (ii, _))) = ix_with_logsum.next() {
                    dist[ii] = (log_xi + dist[ii]).exp();
                }
                break;
            }

            dist[i_sorted] = ub;
        }
        dist
    }


    /// Compute the distribution `d` at current weights `w = weights` 
    /// according to the following equation:
    /// `d = arg min_{d \in \Delta_{m, nu}} d^\top A w + (h(d) / eta)`
    /// This computation takes `O(m ln(m))`, 
    /// where m is the number of examples.
    #[inline(always)]
    fn distribution_at<C>(&self,
                          data: &DataFrame,
                          target: &Series,
                          classifiers: &[C],
                          weights: &[f64])
        -> Vec<f64>
        where C: Classifier
    {
        let dist = self.log_dist_at(data, target, classifiers, weights);

        self.projection(&dist[..])
    }


    /// Returns the weights on hypotheses
    /// based on the soft margin optimization
    fn lpb_weight<C>(&mut self,
                     data: &DataFrame,
                     target: &Series,
                     opt_h: Option<&C>)
        -> Vec<f64>
        where C: Classifier,
    {
        self.lp_model.as_ref()
            .unwrap()
            .borrow_mut()
            .update(data, target, opt_h)
    }


    /// Update the weights on hypotheses
    /// by taking a interior_point.
    fn classic_weight(&self,
                      index: usize,
                      lambda: f64,
                      mut weights: Vec<f64>)
        -> Vec<f64>
    {
        weights.iter_mut()
            .enumerate()
            .for_each(|(i, w)| {
                let e = if index == i { 1.0 } else { 0.0 };
                *w = lambda * e + (1.0 - lambda) * *w;
            });


        weights
    }


    /// Update with short step strategy.
    fn shortstep_weight<C>(&self,
                           data: &DataFrame,
                           target: &Series,
                           dist: &[f64],
                           index: usize,
                           classifiers: &[C],
                           mut weights: Vec<f64>)
        -> Vec<f64>
        where C: Classifier
    {
        if classifiers.len() == 1 {
            return vec![1.0];
        }
        let target = target.i64()
            .expect("The target is not a dtype i64");


        let new_h: &C = &classifiers[index];

        let mut numer: f64 = 0.0;
        let mut denom: f64 = f64::MIN;

        target.into_iter()
            .zip(dist)
            .enumerate()
            .for_each(|(i, (y, &d))| {
                let y = y.unwrap() as f64;
                let new_p = new_h.confidence(data, i);
                let old_p = prediction(i, data, classifiers, &weights[..]);

                let gap = y * (new_p - old_p);
                numer += d * gap;
                denom = denom.max(gap.abs());
            });

        let step = numer / (self.eta * denom.powi(2));

        let lambda = (step.max(0.0_f64)).min(1.0_f64);


        weights.iter_mut()
            .enumerate()
            .for_each(|(i, w)| {
                let e = if index == i { 1.0 } else { 0.0 };
                *w = lambda * e + (1.0 - lambda) * *w;
            });


        weights
    }


    /// Update with short step strategy.
    fn linesearch_weight<C>(&self,
                            data: &DataFrame,
                            target: &Series,
                            base: Vec<f64>,
                            dir: Vec<f64>,
                            classifiers: &[C],
                            max_stepsize: f64)
        -> Vec<f64>
        where C: Classifier
    {
        let mut lb = 0.0_f64;
        let mut ub = max_stepsize;

        {
            let tmp = base.iter()
                .zip(dir.iter())
                .map(|(b, d)| b + max_stepsize * d)
                .collect::<Vec<f64>>();

            let dist = self.distribution_at(
                data, target, classifiers, &tmp[..]
            );


            let edge = edge_of(data, target, &dist[..], classifiers, &dir[..]);

            if edge >= 0.0 {
                // println!("[step size]: {max_stepsize}");
                let weights = base.into_iter()
                    .zip(dir)
                    .map(|(b, d)| b + max_stepsize * d)
                    .collect::<Vec<f64>>();
                return weights
            }
        }


        while 10.0 * (ub - lb) > self.tolerance {
            let stepsize = (lb + ub) / 2.0;

            let tmp = base.iter()
                .zip(dir.iter())
                .map(|(b, d)| b + stepsize * d)
                .collect::<Vec<f64>>();


            let dist = self.distribution_at(
                data, target, classifiers, &tmp[..]
            );


            let edge = edge_of(data, target, &dist[..], classifiers, &dir[..]);

            if edge > 0.0 {
                lb = stepsize;
            } else if edge < 0.0 {
                ub = stepsize;
            } else {
                break;
            }
        }


        let stepsize = (lb + ub) / 2.0;

        // println!("[step size]: {stepsize}");


        base.into_iter()
            .zip(dir)
            .map(|(b, d)| b + stepsize * d)
            .collect::<Vec<f64>>()
    }


    /// Update with away step strategy.
    fn awaystep_weight<C>(&self,
                          data: &DataFrame,
                          target: &Series,
                          dist: &[f64],
                          index: usize,
                          classifiers: &[C],
                          weights: Vec<f64>,
                          fw_edge: f64)
        -> Vec<f64>
        where C: Classifier
    {
        let arr = target.i64()
            .expect("The target is not a dtype i64");

        let (away_ix, away_edge) = classifiers.into_iter()
            .enumerate()
            .filter(|(i, _)| weights[*i] > 0.0)
            .map(|(i, h)| {
                let e = arr.into_iter()
                    .zip(dist)
                    .enumerate()
                    .map(|(i, (y, &d))| {
                        let y = y.unwrap() as f64;
                        d * y * h.confidence(data, i)
                    })
                    .sum::<f64>();
                (i, e)
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let edge = edge_of(data, target, dist, classifiers, &weights[..]);

        let mut direction;
        let max_stepsize;
        if fw_edge - edge >= edge - away_edge {
            direction = weights.clone();
            direction.iter_mut()
                .for_each(|d| { *d *= -1.0; });
            direction[index] += 1.0;

            max_stepsize = 1.0;
        } else {
            let alpha = weights[away_ix];
            direction = weights.clone();
            direction[away_ix] = 0.0;

            // max_stepsize = if alpha == 1.0 {
            //     1.0_f64
            // } else {
            //     alpha / (1.0 - alpha)
            // };
            max_stepsize = alpha / (1.0 - alpha);
        }
        self.linesearch_weight(data,
                               target,
                               weights,
                               direction,
                               classifiers,
                               max_stepsize)
    }


    /// Update with pairwise strategy.
    fn pairwise_weight<C>(&self,
                          data: &DataFrame,
                          target: &Series,
                          dist: &[f64],
                          index: usize,
                          classifiers: &[C],
                          weights: Vec<f64>)
        -> Vec<f64>
        where C: Classifier
    {
        let arr = target.i64()
            .expect("The target is not a dtype i64");

        let away_ix = classifiers.into_iter()
            .enumerate()
            .filter(|(j, _)| weights[*j] > 0.0)
            .map(|(j, h)| {
                let e = arr.into_iter()
                    .zip(dist)
                    .enumerate()
                    .map(|(i, (y, &d))| {
                        let y = y.unwrap() as f64;
                        d * y * h.confidence(data, i)
                    })
                    .sum::<f64>();
                (j, e)
            })
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0;

        let max_stepsize = weights[away_ix];

        self.pairwise_linesearch(
            data, target, weights, index, away_ix, classifiers, max_stepsize
        )
    }


    fn pairwise_linesearch<C>(&self,
                              data: &DataFrame,
                              target: &Series,
                              mut base: Vec<f64>,
                              fw_ix: usize,
                              aw_ix: usize,
                              classifiers: &[C],
                              max_stepsize: f64)
        -> Vec<f64>
        where C: Classifier
    {
        let arr = target.i64()
            .expect("The target is not a dtype i64");
        let fw_h = &classifiers[fw_ix];
        let aw_h = &classifiers[aw_ix];

        let delta = |dist: &[f64]| {
            dist.iter()
                .copied()
                .zip(arr)
                .enumerate()
                .map(|(i, (d, y))| {
                    let y = y.unwrap() as f64;
                    let fw_p = fw_h.confidence(data, i);
                    let aw_p = aw_h.confidence(data, i);
                    d * y * (fw_p - aw_p)
                })
                .sum::<f64>()
        };



        base[fw_ix] += max_stepsize;
        base[aw_ix] -= max_stepsize;

        let mut log_dist = self.log_dist_at(
            data, target, classifiers, &base[..]
        );

        let dist = self.projection(&log_dist[..]);
        // let dist = self.distribution_at(data, target, classifiers, &base[..]);


        let gap_vec = arr.into_iter()
            .enumerate()
            .map(|(i, y)| {
                let y = y.unwrap() as f64;
                let fw_p = fw_h.confidence(data, i);
                let aw_p = aw_h.confidence(data, i);

                y * (fw_p - aw_p)
            })
            .collect::<Vec<_>>();


        let df = delta(&dist[..]);

        // TODO
        // If the max step size is the best one,
        // remove the variable 
        // and the corresponding hypothesis and constraint.
        // To remove a constraint, see
        // https://docs.rs/grb/latest/grb/struct.Model.html#method.remove
        // https://doc.rust-lang.org/std/vec/struct.Vec.html#method.swap_remove
        if df >= 0.0 {
            return base;
        }


        let mut lb = 0.0;
        let mut ub = max_stepsize;

        let mut old_stepsize = max_stepsize;
        while ub - lb > 1e-9 {
        // while 10.0 * (ub - lb) > self.tolerance {
            let stepsize = (lb + ub) / 2.0;

            base[fw_ix] += stepsize - old_stepsize;
            base[aw_ix] -= stepsize - old_stepsize;

            log_dist.iter_mut()
                .zip(gap_vec.iter())
                .for_each(|(d, g)| {
                    *d -= self.eta * (stepsize - old_stepsize) * g;
                });

            let dist = self.projection(&log_dist[..]);


            old_stepsize = stepsize;


            let df = delta(&dist[..]);

            if df > 0.0 {
                lb = stepsize;
            } else if df < 0.0 {
                ub = stepsize;
            } else {
                break;
            }
        }


        let stepsize = (lb + ub) / 2.0;
        base[fw_ix] += stepsize - old_stepsize;
        base[aw_ix] -= stepsize - old_stepsize;

        base
    }


    /// Returns the objective value 
    /// `- \tilde{f}^\star (-Aw)` at the current weighting `w = weights`.
    fn objval<C>(&self,
                 data: &DataFrame,
                 target: &Series,
                 classifiers: &[C],
                 weights: &[f64])
        -> f64
        where C: Classifier
    {
        let dist = self.distribution_at(data, target, classifiers, weights);


        let margin = edge_of(data, target, &dist[..], classifiers, weights);


        let entropy = dist.iter()
            .copied()
            .map(|d| if d == 0.0 { 0.0 } else { d * d.ln() })
            .sum::<f64>();

        margin + (entropy + (self.size as f64).ln()) / self.eta
    }


    fn init_solver(&mut self) {
        let upper_bound = 1.0 / self.nu;

        assert!((0.0..=1.0).contains(&upper_bound));

        self.lp_model = Some(RefCell::new(
            LPModel::init(self.size, upper_bound)
        ));
    }


    fn fw_weight<C>(&self,
                    data: &DataFrame,
                    target: &Series,
                    dist: &[f64],
                    position: usize,
                    classifiers: &[C],
                    weights: Vec<f64>,
                    edge_of_h: f64,
                    step: usize)
        -> Vec<f64>
        where C: Classifier
    {
        match self.strategy {
            FW::Classic =>
                self.classic_weight(position,
                                    2.0 / (step as f64 + 2.0),
                                    weights),
            FW::ShortStep =>
                self.shortstep_weight(data,
                                      target,
                                      dist,
                                      position,
                                      classifiers,
                                      weights),
            FW::LineSearch => {
                let mut direction = weights.clone();
                direction.iter_mut()
                    .for_each(|d| { *d *= -1.0; });
                direction[position] += 1.0;
                self.linesearch_weight(data,
                                      target,
                                      weights,
                                      direction,
                                      classifiers,
                                      1.0)
            },
            FW::AwayStep =>
                self.awaystep_weight(data,
                                     target,
                                     dist,
                                     position,
                                     classifiers,
                                     weights,
                                     edge_of_h),
            FW::PairWise =>
                self.pairwise_weight(data,
                                     target,
                                     dist,
                                     position,
                                     classifiers,
                                     weights),
        }
    }
}


impl<C> Booster<C> for MLPBoost
    where C: Classifier + PartialEq + std::fmt::Debug,
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


        let max_iter = self.max_loop(tolerance);

        self.terminated = max_iter;


        // Obtain a hypothesis for the uniform distribution.
        // let dist = self.distribution_at(
        //     data, target, &classifiers[..], &weights[..]
        // );


        let h = base_learner.produce(
            data, target, &vec![1.0 / self.size as f64; self.size]
        );

        let mut classifiers = vec![h];
        let mut weights = self.lpb_weight(data, target, classifiers.last());


        self.lpb_call = 1_usize;


        let mut gamma: f64 = 1.0;


        // Since the MLPBoost does not have non-trivial iteration,
        // we run this until the stopping criterion is satisfied.
        for step in 0..max_iter {
            let dist = self.distribution_at(
                data, target, &classifiers[..], &weights[..]
            );


            let h = base_learner.produce(data, target, &dist[..]);


            // let edge = edge_of(
            //     data, target, &dist[..], &classifiers[..], &weights[..]
            // );


            let edge_of_h = target.i64()
                .expect("The target is not a dtype i64")
                .into_iter()
                .zip(dist.iter().copied())
                .enumerate()
                .map(|(i, (y, d))|
                    d * y.unwrap() as f64 * h.confidence(data, i)
                )
                .sum::<f64>();

            gamma = gamma.min(edge_of_h);


            let objval = self.objval(
                data, target, &classifiers[..], &weights[..]
            );


            if gamma - objval <= self.tolerance {
                println!("Break loop at: {step}");
                self.terminated = step + 1;
                break;
            }


            let mut opt_h = None;
            let position = classifiers.iter()
                .position(|f| *f == h)
                .unwrap_or(classifiers.len());


            if position == classifiers.len() {
                classifiers.push(h);
                weights.push(0.0);
                opt_h = classifiers.last();
            }


            // let lambda = 2.0 / (step + 2) as f64;


            let fw_w = self.fw_weight(data,
                                      target,
                                      &dist[..],
                                      position,
                                      &classifiers[..],
                                      weights,
                                      edge_of_h,
                                      step);

            if self.fw_only {
                weights = fw_w;
            } else {
                let lpb_w = self.lpb_weight(data, target, opt_h);

                match self.switch {
                    Switch::Edge => {
                        let lpb_edge = edge_of(data,
                                               target,
                                               &dist[..],
                                               &classifiers[..],
                                               &lpb_w[..]);
                        let fw_edge = edge_of(data,
                                              target,
                                              &dist[..],
                                              &classifiers[..],
                                              &fw_w[..]);
                        if lpb_edge > fw_edge {
                            self.lpb_call += 1;
                            weights = lpb_w;
                        } else {
                            weights = fw_w;
                        }
                    },
                    Switch::ObjVal => {
                            let lpb_objval = self.objval(
                                data, target, &classifiers[..], &lpb_w[..]
                            );
                            let fw_objval = self.objval(
                                data, target, &classifiers[..], &fw_w[..]
                            );


                            if lpb_objval >= fw_objval {
                                self.lpb_call += 1;
                                weights = lpb_w;
                            } else {
                                weights = fw_w;
                            }
                        },
                }
            }
        }


        let clfs = weights.into_iter()
            .zip(classifiers)
            .filter(|(w, _)| *w > 0.0)
            .collect::<Vec<_>>();


        CombinedClassifier::from(clfs)
    }
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


fn edge_of<C>(data: &DataFrame,
              target: &Series,
              dist: &[f64],
              classifiers: &[C],
              weights: &[f64])
    -> f64
    where C: Classifier
{
    let target = target.i64()
        .expect("The target is not a dtype i64");

    dist.iter()
        .copied()
        .zip(target)
        .enumerate()
        .map(|(i, (d, y))| {
            let y = y.unwrap() as f64;
            let p = prediction(i, data, classifiers, weights);
            d * y * p
        })
        .sum::<f64>()
}


use std::time::Instant;
impl MLPBoost {
    /// Returns the quadruple of 
    /// - A vector of objective values,
    /// - A vector of times,
    /// - A vector whether the LPB update (1) occurs or not (0), and
    /// - Total number of iterations
    pub fn monitor<B, C>(&mut self,
                         base_learner: &B,
                         train_x: &DataFrame,
                         train_y: &Series,
                         test_x: &DataFrame,
                         test_y: &Series,
                         tolerance: f64,
                         time_bound: u128)
        -> (Vec<f64>, Vec<f64>, Vec<u128>, Vec<usize>, usize)
        where B: BaseLearner<Clf = C>,
              C: Classifier + PartialEq + std::fmt::Debug,
    {
        // ------------------------------------------
        // Set parameters
        self.set_tolerance(tolerance);
        self.init_solver();


        let max_iter = self.max_loop(tolerance);
        self.terminated = max_iter;


        let mut gamma: f64 = 1.0;

        let mut objvals = vec![-1.0_f64];
        let mut tests = vec![1.0_f64];
        let mut times = vec![0_u128];
        let mut lpb_updates = vec![0_usize];
        // ------------------------------------------


        let now = Instant::now();

        let h = base_learner.produce(
            train_x, train_y, &vec![1.0 / self.size as f64; self.size]
        );

        let mut classifiers = vec![h];
        let mut weights = self.lpb_weight(train_x, train_y, classifiers.last());

        times.push(now.elapsed().as_millis());
        lpb_updates.push(1_usize);

        objvals.push(lp_objval(
            train_x, train_y, &weights[..], &classifiers[..], self.nu
        ));
        tests.push(
            loss(test_x, test_y, &classifiers[..], &weights[..])
        );


        // Since the MLPBoost does not have non-trivial iteration,
        // we run this until the stopping criterion is satisfied.
        for step in 1..=max_iter {
            let now = Instant::now();

            let dist = self.distribution_at(
                train_x, train_y, &classifiers[..], &weights[..]
            );


            let h = base_learner.produce(train_x, train_y, &dist[..]);


            // let edge = edge_of(
            //     data, target, &dist[..], &classifiers[..], &weights[..]
            // );


            let edge_of_h = train_y.i64()
                .expect("The target is not a dtype i64")
                .into_iter()
                .zip(dist.iter().copied())
                .enumerate()
                .map(|(i, (y, d))|
                    d * y.unwrap() as f64 * h.confidence(train_x, i)
                )
                .sum::<f64>();

            gamma = gamma.min(edge_of_h);


            let objval = self.objval(
                train_x, train_y, &classifiers[..], &weights[..]
            );


            if gamma - objval <= self.tolerance {
                println!("Break loop at: {step}");
                self.terminated = step + 1;
                break;
            }


            let mut opt_h = None;
            let position = classifiers.iter()
                .position(|f| *f == h)
                .unwrap_or(classifiers.len());


            if position == classifiers.len() {
                classifiers.push(h);
                weights.push(0.0);
                opt_h = classifiers.last();
            }


            // let lambda = 2.0 / (step + 2) as f64;


            let fw_w = self.fw_weight(train_x,
                                      train_y,
                                      &dist[..],
                                      position,
                                      &classifiers[..],
                                      weights,
                                      edge_of_h,
                                      step);
            if self.fw_only {
                lpb_updates.push(0_usize);
                weights = fw_w;
            } else {
                let lpb_w = self.lpb_weight(train_x, train_y, opt_h);

                match self.switch {
                    Switch::Edge => {
                        let lpb_edge = edge_of(train_x,
                                               train_y,
                                               &dist[..],
                                               &classifiers[..],
                                               &lpb_w[..]);
                        let fw_edge = edge_of(train_x,
                                              train_y,
                                              &dist[..],
                                              &classifiers[..],
                                              &fw_w[..]);
                        if lpb_edge > fw_edge {
                            lpb_updates.push(1_usize);
                            weights = lpb_w;
                        } else {
                            lpb_updates.push(0_usize);
                            weights = fw_w;
                        }
                    },
                    Switch::ObjVal => {
                            let lpb_objval = self.objval(
                                train_x, train_y, &classifiers[..], &lpb_w[..]
                            );
                            let fw_objval = self.objval(
                                train_x, train_y, &classifiers[..], &fw_w[..]
                            );


                            if lpb_objval >= fw_objval {
                                lpb_updates.push(1_usize);
                                weights = lpb_w;
                            } else {
                                lpb_updates.push(0_usize);
                                weights = fw_w;
                            }
                        },
                }
            }


            let diff = now.elapsed().as_millis() + times.last().unwrap();

            times.push(diff);

            objvals.push(lp_objval(
                train_x, train_y, &weights[..], &classifiers[..], self.nu
            ));

            tests.push(
                loss(test_x, test_y, &classifiers[..], &weights[..])
            );

            if diff > time_bound {
                break;
            }
        }


        (objvals, tests, times, lpb_updates, self.terminated)
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


    dist.into_iter()
        .zip(margins)
        .map(|(d, yh)| d * yh)
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
// // DEBUG CODE
// fn print_log<C>(dist: &[f64],
//                 data: &DataFrame,
//                 target: &Series,
//                 h: &C,
//                 classifiers: &[C],
//                 weights: &[f64])
//     where C: Classifier + std::fmt::Debug
// {
//     println!("Got {h:?}");
//     println!(
//         "# of nonzero in dist: {} / {}",
//         dist.iter().filter(|&&d| d > 0.0).count(),
//         dist.len()
//     );
//     println!(
//         "\tneg sum: {}\n\
//          \tpos sum: {}",
//         target.i64()
//             .expect("The target is not a dtype i64")
//             .into_iter()
//             .zip(dist.iter().copied())
//             .filter_map(|(y, d)|
//                 if y.unwrap() > 0 { Some(d) } else { None }
//                 // if y.unwrap() > 0 { Some(d.exp()) } else { None }
//             )
//             .sum::<f64>(),
//         target.i64()
//             .expect("The target is not a dtype i64")
//             .into_iter()
//             .zip(dist.iter().copied())
//             .filter_map(|(y, d)|
//                 if y.unwrap() < 0 { Some(d) } else { None }
//                 // if y.unwrap() < 0 { Some(d.exp()) } else { None }
//             )
//             .sum::<f64>(),
//     );
//     let wrong = target.i64()
//         .expect("The target is not a dtype i64")
//         .into_iter()
//         .enumerate()
//         .map(|(i, y)| {
//             let y = y.unwrap();
//             let p = h.predict(data, i) as f64;
// 
//             (y, p)
//         })
//         .collect::<Vec<_>>();
// 
//     println!(
//         "# of misclassification (new): {:>4}/{:>4}",
//         wrong.iter().filter(|(y, p)| *y as f64 * p < 0.0).count(),
//         wrong.len()
//     );
// 
//     println!(
//         "\t(+): {:>4}/{:>4}\n\
//          \t(-): {:>4}/{:>4}",
//          wrong.iter().filter(|(y, p)| *y as f64 * p < 0.0 && *y == 1_i64).count(),
//          wrong.iter().filter(|(y, _)| *y == 1_i64).count(),
//          wrong.iter().filter(|(y, p)| *y as f64 * p < 0.0 && *y == -1_i64).count(),
//          wrong.iter().filter(|(y, _)| *y == -1_i64).count(),
//     );
// 
// 
//     let wrong = target.i64()
//         .expect("The target is not a dtype i64")
//         .into_iter()
//         .enumerate()
//         .map(|(i, y)| {
//             let y = y.unwrap();
//             let p = prediction(i, data, &classifiers[..], &weights[..]);
// 
//             (y, p)
//         })
//         .collect::<Vec<_>>();
// 
//     println!(
//         "# of misclassification (now): {:>4}/{:>4}",
//         wrong.iter().filter(|(y, p)| *y as f64 * p <= 0.0).count(),
//         wrong.len()
//     );
// 
//     println!(
//         "\t(+): ({:>4} + {:>4})/{:>4}\n\
//          \t(-): ({:>4} + {:>4})/{:>4}",
//          wrong.iter().filter(|(y, p)| *y as f64 * p < 0.0 && *y == 1_i64).count(),
//          wrong.iter().filter(|(y, p)| *y as f64 * p == 0.0 && *y == 1_i64).count(),
//          wrong.iter().filter(|(y, _)| *y == 1_i64).count(),
//          wrong.iter().filter(|(y, p)| *y as f64 * p < 0.0 && *y == -1_i64).count(),
//          wrong.iter().filter(|(y, p)| *y as f64 * p == 0.0 && *y == -1_i64).count(),
//          wrong.iter().filter(|(y, _)| *y == -1_i64).count(),
//     );
// }
