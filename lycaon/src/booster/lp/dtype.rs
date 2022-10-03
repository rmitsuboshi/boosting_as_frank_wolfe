pub mod constraint;

use constraint::*;


/// The enum `Status` represents the status after the optimization.
/// 
/// - `Optimal`:    This implies that the optimization is successed.
/// - `Infeasible`: This implies that there is no feasible solution.
/// - `Unbounded`:  This implies the optimal value is `f64::MIN` or `f64::MAX`.
/// - `NoSolution`: This implies that there is no optimal solution.
///                 The `noSolution` happens when there is no $\min$ or $\max$
///                 value but exists some $\inf$ or $\sup$ value.
#[derive(Debug, PartialEq, Eq)]
pub enum Status {
    Optimal,
    Infeasible,
    Unbounded,
    NoSolution
}






/// This `Model` holds the linear programming problem of the standard form:
/// 
///     $min_{x} c \cdot x \text{ s.t. } A x = b, x \geq 0$
/// 
/// where $c$ are $m$-dimensional real-valued vectors and
///       $A$ is an $n \times m$ real-valued matrix and
///       $b, 0$ is an $n$-dimensional real vector.
/// 
/// Here, the parameters are corresponds to the variables:
///     - the objective function `objective` corresponds to $c$.
///     - the `matrix` corresponds to the matrix $A$
///     - the `vector` corresponds to the vector $b$
/// 
/// Since we only use this sub-crate for boosting,
/// we assume that the variables are real-valued.
pub(crate) struct Model {
    objective: Option<Vec<f64>>,
    constraints: Option<Vec<Constraint>>,
    // matrix:    Option<Vec<Vec<f64>>>,
    // vector:    Option<Vec<f64>>,
    // var_size:  usize,


    optimal_solution: Option<Vec<f64>>,
    optimal_value:    Option<f64>,
    optimal_status:   Option<Status>,
}


impl Model {
    pub fn new() -> Self {
        Model {
            objective:   None,
            constraints: None,
            // matrix:    None,
            // vector:    None,

            optimal_solution: None,
            optimal_value:    None,
            optimal_status:   None,
        }
    }


    /// Set the coefficient vector of the objective function.
    pub fn objective(&mut self, objective: Vec<f64>) {
        self.objective = Some(objective);
    }


    pub fn add_constraint(&mut self,
                          lhs: mut Vec<f64>,
                          sense: Sense,
                          rhs: mut f64)
    {
        let constraint = Constraint::new(lhs, sense, rhs);

        match self.constraints {
            None => {
                self.constraints = Some(vec![constraint]);
            },
            Some(c) => {
                c.push(constraint);
            }
        }
    }


    fn missing_param_exist(&self) -> bool {
        todo!();
    }


    /// Solve the given optimization problem.
    pub fn optimize(&mut self) -> std::io::Result<()> {
    }
}







