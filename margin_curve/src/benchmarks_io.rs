use polars::prelude::*;


use std::fs::File;
use std::io::prelude::*;


pub struct Benchmark {
    file: String,
    train: Vec<Vec<usize>>,
    test: Vec<Vec<usize>>,
}


impl Benchmark {
    /// Construct a new instance of `Benchmark`
    /// Note that `dataset_{train,test}.csv` holds 
    /// the row indices over `dataset.csv` (1-indexed).
    pub fn new(path: &str) -> Self {
        let file = format!("{path}.csv");

        let train = format!("{path}_train.csv");
        let mut train = File::open(train).unwrap();

        let mut contents = String::new();
        train.read_to_string(&mut contents).unwrap();
        let train = contents.lines()
            .map(|line| {
                line.split(',')
                    // Convert to 0-indexed indices
                    .map(|word| word.trim().parse::<usize>().unwrap() - 1)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<Vec<usize>>>();
        let test = format!("{path}_test.csv");
        let mut test = File::open(test).unwrap();

        let mut contents = String::new();
        test.read_to_string(&mut contents).unwrap();
        let test = contents.lines()
            .map(|line| {
                line.split(',')
                    // Convert to 0-indexed indices
                    .map(|word| word.trim().parse::<usize>().unwrap() - 1)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<Vec<usize>>>();

        Self { file, train, test }
    }


    pub fn train_at(&self, k: usize) -> (DataFrame, Series) {

        let df = CsvReader::from_path(&self.file)
            .unwrap()
            .has_header(true)
            .finish()
            .unwrap();

        let rows = self.train[k].iter()
            .copied()
            .map(|i| df.get_row(i))
            .collect::<Vec<_>>();

        let mut data = DataFrame::from_rows(&rows).unwrap();
        data.set_column_names(&df.get_column_names()).unwrap();
        let target = data.drop_in_place("class")
            .unwrap();

        (data, target)
    }


    pub fn test_at(&self, k: usize) -> (DataFrame, Series) {

        let df = CsvReader::from_path(&self.file)
            .unwrap()
            .has_header(true)
            .finish()
            .unwrap();

        let rows = self.test[k].iter()
            .copied()
            .map(|i| df.get_row(i))
            .collect::<Vec<_>>();

        let mut data = DataFrame::from_rows(&rows).unwrap();
        data.set_column_names(&df.get_column_names()).unwrap();
        let target = data.drop_in_place("class")
            .unwrap();

        (data, target)
    }
}


