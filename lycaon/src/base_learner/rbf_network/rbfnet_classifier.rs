//! Provides RBF network class.
//! RBF network is 3-layers neural network.

use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize};


/// Defines the weights on each layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Weights {
    matrix:    Vec<Vec<f64>>,
    intercept: f64,
}


impl From<(Vec<Vec<f64>>, f64)> for Weights {
    #[inline]
    fn from((matrix, intercept): (Vec<Vec<f64>>, f64)) -> Self {
        Self { matrix, intercept }
    }
}


/// Classifier output from `RBFNet`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RBFNetClassifier {
    layer_1: Weights,
    layer_2: Weights,
}



