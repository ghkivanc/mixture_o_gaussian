mod model;
use ndarray::{Array1, Array2, Array3, arr1, arr2, Axis, ArrayView1, s};
pub use model::*;
use rand::Rng;
use csv::ReaderBuilder;
use std::fs::File;
use std::io::Write;
use csv::WriterBuilder;

pub fn load_data(file_path: &str) -> (Vec<Vec<f64>>, Vec<u8>) {
    let raw_data = load_csv(file_path);

    let (features, labels) = split_features_and_labels(&raw_data);

    (features, labels)
}

pub fn split_features_and_labels(
    data: &Vec<Vec<f64>>,
) -> (Vec<Vec<f64>>, Vec<u8>) {
    let n = data[0].len() - 1;

    let mut features = Vec::new();
    let mut labels = Vec::new();

    for row in data {
        let (feature_row, label) = row.split_at(n);
        features.push(feature_row.to_vec());
        labels.push(label[0] as u8);
    }

    (features, labels)
}


pub fn load_csv(file_path: &str) -> Vec<Vec<f64>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false) // Set to false if your CSV has no headers
        .from_path(file_path)
        .expect("Failed to open file");

    let mut data = Vec::new();

    for result in rdr.records() {
        let record = result.expect("Failed to read record");
        let row: Vec<f64> = record
            .iter()
            .map(|value| value.parse::<f64>().expect("Failed to parse value"))
            .collect();
        data.push(row);
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_model() {
        let mut rng = rand::thread_rng();
        let dataset = (0..5).map(|_| vec![rng.gen_range(-3.0..3.0); 5]).collect(); 
        let model = GaussianMix::new(2, 5, 5, &dataset);
        assert!(true);
    }


    #[test]
    fn test_gaussian()
    {
       //fn GaussianDensity(cluster_mean:&Array1<f64>, cov_mat:&Array2<f64> , x:&Array1<f64>)->f64
        let means = arr1(&[3.1, 3.3]); // Use the `arr1` macro
        let cov_mat = arr2(&[[3.0, 0.0], [0.0, 3.0]]); // Use the `arr2` macro
        let x = arr1(&[3.1, 3.3]); // Use the `arr1` macro
        let mut p = model::GaussianDensity(&means, &cov_mat, &x);
        p = 0.00227; 
        assert!((p - 0.00227022333).abs() < 1e-4, "Expected 0.00227 but got {}", p);
    }

    #[test]
    fn e_step_test()
    {
        let mut rng = rand::thread_rng();
        let dataset = (0..5).map(|_| vec![rng.gen_range(-3.0..3.0), rng.gen_range(-3.0..3.0),rng.gen_range(-3.0..3.0),rng.gen_range(-3.0..3.0),rng.gen_range(-3.0..3.0)]).collect(); 
        let mut model = GaussianMix::new(2, 5, 5, &dataset);
        model.e_step(&dataset);

        //println!("weights:{:?}", model.weights);
        assert_eq!(model.weights.shape()[0], 5);
    }


    #[test]
    fn m_step_test()
    {
        let mut rng = rand::thread_rng();
        let dataset:Vec<Vec<f64>> = (0..5).map(|_| vec![rng.gen_range(-3.0..3.0), rng.gen_range(-3.0..3.0),rng.gen_range(-3.0..3.0)]).collect(); 
        let mut model = GaussianMix::new(2, 3, 5, &dataset);
        model.e_step(&dataset);
        model.m_step();
        model.e_step(&dataset);
    }

    #[test]
    fn fit_test() {
        let (dataset, labels) = load_data("path/to/data");
        let mut model = GaussianMix::new(3, 4, labels.len(), &dataset);
        let mut results = Vec::<u8>::new();

        results = model.fit(&dataset, 50, true, 10);
         // Ensure all data is written to the file
        println!("{:?} ----------------------------------------------\n{:?}", results, labels);
        println!("cov_mat{:?}", model.covarience_matrices);
        assert!(true)
    }
}
