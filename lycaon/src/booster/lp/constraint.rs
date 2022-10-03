
/// Sense of the optimization problem
#[derive(Debug, PartialEq, Eq)]
pub enum Sense {
    L,
    G,
    E,
}


pub struct Constraint {
    lhs:   Vec<f64>,
    sense: Sense,
    rhs:   f64,
}


impl Constraint {
    pub fn new(lhs: mut Vec<f64>, sense: Sense, rhs: mut f64) -> Constraint {
        if sense == Sense::G {
            for l in lhs.iter_mut() {
                *l *= -1.0_f64;
            }
            rhs *= -1.0;
        }

        Constraint {
            lhs,
            sense,
            rhs,
        }
    }
}
