use polars::prelude::*;
use lycaon::Classifier;


#[derive(Debug, Clone, PartialEq)]
pub struct BadClassifier {
    is_first: bool,
    is_last: bool,
    index: usize,
    gap: f64,
    eps: f64,
}


impl BadClassifier {
    pub fn new(is_first: bool,
               is_last: bool,
               index: usize,
               gap: f64,
               eps: f64)
        -> Self
    {
        Self {
            is_first,
            is_last,
            index,
            gap,
            eps,
        }
    }
}


impl Classifier for BadClassifier {
    fn confidence(&self, _data: &DataFrame, row: usize) -> f64 {
        if self.is_first {
            if row < self.index {
                1.0_f64
            } else if row == self.index {
                -1.0_f64 + 2.0_f64 * self.eps
            } else {
                -1.0_f64 + 3.0_f64 * self.eps
            }
        } else if self.is_last {
            if row < self.index {
                -1.0_f64 + self.eps
            } else {
                1.0_f64 - self.eps
            }
        } else {
            if row == self.index {
                1.0_f64
            } else if row == self.index + 1_usize {
                -1.0_f64 + (self.gap - 1.0_f64) * self.eps
            } else {
                -1.0_f64 + self.gap * self.eps
            }
        }
    }
}
