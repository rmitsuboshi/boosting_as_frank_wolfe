use rand::prelude::*;
use polars::prelude::*;


use crate::BaseLearner;


use super::dtree_classifier::DTreeClassifier;
use super::node::*;
use super::train_node::*;


use std::rc::Rc;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::collections::HashMap;


pub struct RTree {
    max_depth: Option<usize>,
    size: usize,
}


impl RTree {
    /// Initialize `DTree`.
    #[inline]
    pub fn init(df: &DataFrame) -> Self
    {
        let max_depth = None;
        let size = df.shape().0;

        Self {
            max_depth,
            size,
        }
    }
}


impl BaseLearner for RTree {
    type Clf = RTreeClassifier<O, L>;
    fn produce(&self,
               sample: &DataFrame,
               target: &Series,
               res: &[f64])
        -> Self::Clf
    {
        let mut indices = (0..self.size).into_iter()
            .collect::<Vec<usize>>();


        let mut tree = full_tree(
            sample, distribution, train_indices, self.max_depth,
        );


        let root = match Rc::try_unwrap(tree) {
            Ok(train_node) => Node::from(train_node.into_inner()),
            Err(_) => panic!("Root node has reference counter >= 1")
        };

        DTreeClassifier::from(root)
    }
}


#[inline]
fn full_tree(data: &DataFrame,
             target: &Series,
             res: &[f64],
             indices: Vec<usize>,
             max_depth: Option<usize>)
    -> Rc<RefCell<TrainNode>>
{
    let (label, train_node_err) = calc_train_err(sample, dist, &train[..]);

    let test_node_err = calc_test_err(sample, dist, &test[..], &label);


    let node_err = NodeError::from((train_node_err, test_node_err));


    // Compute the node impurity
    let (mut best_split, mut best_decrease) = find_best_split(
        sample, &dist[..], &train[..], 0
    );


    let mut best_index = 0_usize;


    let dim = sample.dim();
    for j in 1..dim {

        let (split, decrease) = find_best_split(
            sample, &dist[..], &train[..], j
        );


        if decrease <= best_decrease {
            best_index    = j;
            best_split    = split;
            best_decrease = decrease;
        }
    }


    let (ltrain, rtrain) = train.into_iter()
        .fold((Vec::new(), Vec::new()), |(mut l, mut r), i| {
            let (x, _) = &sample[i];
            match rule.split(x) {
                LR::Left  => { l.push(i); },
                LR::Right => { r.push(i); }
            }
            (l, r)
        });
    let (ltest, rtest) = test.into_iter()
        .fold((Vec::new(), Vec::new()), |(mut l, mut r), i| {
            let (x, _) = &sample[i];
            match rule.split(x) {
                LR::Left  => { l.push(i); },
                LR::Right => { r.push(i); }
            }
            (l, r)
        });


    // If the split has no meaning, construct a leaf node.
    if ltrain.is_empty() || rtrain.is_empty() {
        let leaf = TrainNode::leaf(label, node_err);
        return Rc::new(RefCell::new(leaf));
    }


    let left = stump_fulltree(sample, dist, ltrain, ltest, criterion);
    let right = stump_fulltree(sample, dist, rtrain, rtest, criterion);


    Rc::new(RefCell::new(TrainNode::branch(
        rule, left, right, label, node_err
    )))
}


/// Returns the best split
/// that maximizes the decrease of impurity.
#[inline]
fn find_best_split(data: &Series,
                   target: &Series,
                   res: &[f64],
                   indices: &[usize])
    -> (f64, Impurity)
{
    todo!()
}
