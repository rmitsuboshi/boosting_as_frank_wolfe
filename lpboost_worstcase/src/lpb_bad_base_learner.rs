use polars::prelude::*;

use lycaon::{
    BaseLearner,
    Classifier,
};

use crate::lpb_bad_classifier::BadClassifier;


pub struct BadBaseLearner {
    // # of examples
    size: usize,

    // Tolerance parameter
    eps: f64,

    pub classifiers: Vec<BadClassifier>
}


impl BadBaseLearner {
    pub fn init(df: &DataFrame) -> Self {
        let size = df.shape().0;

        let classifiers = Vec::with_capacity(size + 1);

        Self {
            size,

            eps: 1e-9,

            classifiers,
        }
    }


    pub fn finish(mut self) -> Self {
        self.eps = 1e-6;
        let mut gap = 3.0_f64;
        let half = (self.size + 1) / 2;
        self.classifiers.push(
            BadClassifier::new(true, false, half, gap, self.eps)
        );

        for k in 0..half-1 {
            let index = half + k;
            gap += 2.0_f64;
            self.classifiers.push(
                BadClassifier::new(false, false, index, gap, self.eps)
            );
        }

        if self.eps * gap >= 1.0 {
            panic!(
                "Too big sample. The predictions become greater than 1"
            );
        }


        gap = 3.0_f64;
        self.classifiers.push(
            BadClassifier::new(false, true, half, gap, self.eps)
        );

        self
    }
}


impl BaseLearner for BadBaseLearner {
    type Clf = BadClassifier;

    fn produce(&self, data: &DataFrame, target: &Series, dist: &[f64])
        -> Self::Clf
    {
        self.classifiers.iter()
            .map(|h| {
                let edge = edge_of(data, target, dist, h);
                (edge, h)
            })
            .max_by(|(e1, _), (e2, _)| e1.partial_cmp(e2).unwrap())
            .unwrap().1
            .clone()
    }
}



fn edge_of<C>(data: &DataFrame,
              target: &Series,
              dist: &[f64],
              clf: &C)
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
            let p = clf.confidence(data, i);
            d * y * p
        })
        .sum::<f64>()
}
