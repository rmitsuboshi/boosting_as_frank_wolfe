# Boosting as Frank-Wolfe
This repository is the code for 
[Boosting as Frank-Wolfe](https://arxiv.org/abs/2209.10831). 
One can reproduce the results in this paper on your computer. 

# PROGRAM DESCRIPTION
A programs that yields the result in this paper.

## Requirements
This program is implemented in [rust 1.63.0](https://www.rust-lang.org/) 
so you need to install it.

You can get the benchmarks dataset from [here](http://theoval.cmp.uea.ac.uk/~gcc/matlab/default.html#benchmarks). 
Or, you can get the dataset from [this repository](https://github.com/tdiethe/gunnar_raetsch_benchmark_datasets). 


## Usage
Here is the usage of our program. 
One can reproduce the experiment section with these commands.

### Test error table
```bash
$ cd cv5fold
$ cargo build --release
$ cd target/release
$ ./cv5fold
```

You need to specify the `PATH` to the benchmarks dataset 
in `into_5foldcv.py`.


### Running time table
```bash
$ cd time
$ cargo build --release
$ cp ./time.sh target/release
$ cd target/release
$ ./time.sh
```

You need to specify the path to the benchmarks dataset 
`path` in `time/time.sh`.


### Worst case for LPBoost
```bash
$ cd lpboost_worstcase
$ cargo build --release
$ cd target/release
$ ./lpboost_worstcase
```

The number `64` of line 14 in `main.rs` is 
the number of training examples.
```rust
let (data, target) = dummy_sample_of_size(64);
```

In this case, LPBoost terminates after 32 iterations, 
while the other algorithms terminates in 2 iterations. 


### Plots
The following command yields CSV files for the curves of 
objective value, test error, and the number of secondary update call. 
```bash
$ cd margin_curve
$ cargo build --release
$ cd target/release
$ ./margin_curve
```


