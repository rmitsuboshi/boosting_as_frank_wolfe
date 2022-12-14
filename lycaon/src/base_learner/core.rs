//! The core library for the base learner in the boosting protocol.
//! 
//! The base learner in the general boosting setting is as follows:
//! 
//! Given a distribution over training examples,
//! the base learner returns a hypothesis that is slightly better than
//! the random guessing, where the **edge** is the affine transformation of
//! the weighted training error.
//! 
//! In this code, we assume that the base learner returns a hypothesis
//! that **maximizes** the edge for a given distribution.
//! This assumption is stronger than the previous one, but the resulting
//! combined hypothesis becomes much stronger.
//! 
//! I'm planning to implement the code for the general base learner setting.
//! 
use polars::prelude::*;


/// An interface that returns a function that implements 
/// the `Classifier` trait.
pub trait BaseLearner {
    /// Returned hypothesis generated by `self`.
    type Clf;

    /// Outputs an instance of `Classifier` trait
    /// that achieves high accuracy
    /// on the given distribution `dist`.
    fn produce(&self, data: &DataFrame, target: &Series, dist: &[f64])
        -> Self::Clf;
}

