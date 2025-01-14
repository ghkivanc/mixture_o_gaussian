use ndarray::{Array1, Array2, Array3, ArrayView1, Axis, s};
use ndarray_linalg::Eig;
use ndarray_linalg::solve::Inverse;
use rand::Rng;
use std::fmt;
use nalgebra::{DMatrix, Cholesky};

#[derive(Debug)]
pub struct GaussianMix
{
    pub cluster_means:Array2<f64>,
    pub curr_cluster_dist:Array1<f64>,
    pub covarience_matrices:Array3<f64>,
    pub weights:Array2<f64>,
    pub dataset:Array2<f64>
}

impl GaussianMix {
    pub fn e_step(&mut self, dataset: &Vec<Vec<f64>>) {
        for (idx, instance) in dataset.iter().enumerate() {
            let instance_array = Array1::from_vec(instance.clone());
            let mut log_probs = Vec::new();
            
            // Compute log probabilities
            for i in 0..self.curr_cluster_dist.len() {
                let gaussian_prob = GaussianDensity(
                    &self.cluster_means.slice(s![i, ..]).to_owned(),
                    &self.covarience_matrices.slice(s![i, .., ..]).to_owned(),
                    &instance_array,
                );
                
                // Work in log space
                log_probs.push(gaussian_prob.ln() + self.curr_cluster_dist[i].ln());
            }
            
            // Log-sum-exp trick for numerical stability
            let max_log_prob = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum = log_probs.iter()
                .map(|&x| (x - max_log_prob).exp())
                .sum::<f64>();
            let log_sum = max_log_prob + sum.ln();
            
            // Convert back to probabilities
            for (i, &log_prob) in log_probs.iter().enumerate() {
                self.weights.slice_mut(s![idx, ..])[i] = (log_prob - log_sum).exp();
            }
        }
    }
    pub fn m_step(&mut self) {
        let dataset_nd = &self.dataset;
        let n_samples = dataset_nd.shape()[0];
        
        // Update cluster weights (curr_cluster_dist)
        for k in 0..self.curr_cluster_dist.len() {
            // Sum of responsibilities for cluster k
            let n_k: f64 = self.weights.column(k).sum();
            self.curr_cluster_dist[k] = n_k / n_samples as f64;
            
            if n_k > 0.0 {  // Avoid division by zero
                // Update means
                let mut new_mean = Array1::zeros(dataset_nd.shape()[1]);
                for i in 0..n_samples {
                    let row_weight = self.weights[[i, k]] * &dataset_nd.row(i).to_owned();
                    new_mean = new_mean + row_weight;
                }
                new_mean = new_mean / n_k;
                self.cluster_means.row_mut(k).assign(&new_mean);
                
                // Update covariance
                let mut new_cov = Array2::zeros((dataset_nd.shape()[1], dataset_nd.shape()[1]));
                for i in 0..n_samples {
                    let diff = dataset_nd.row(i).to_owned() - self.cluster_means.row(k).to_owned();
                    let diff_matrix = diff.clone().insert_axis(Axis(1));
                    let outer_product = diff_matrix.dot(&diff_matrix.t());
                    new_cov = new_cov + (outer_product * self.weights[[i, k]]);
                }
                new_cov = new_cov / n_k;
                
                // Add regularization
                for i in 0..new_cov.shape()[0] {
                    new_cov[[i, i]] = new_cov[[i, i]] + 1e-6;
                }
                
                self.covarience_matrices.slice_mut(s![k, .., ..]).assign(&new_cov);
            }
        }
    }


    pub fn fit(&mut self, dataset: &Vec<Vec<f64>>, num_iterations:u64, debug:bool, report_step:u64)->Vec<u8>
    {
        let mut results = Vec::<u8>::new();

        if debug {
            println!("\nInitial state:");
            println!("Means:\n{:?}", self.cluster_means);
            println!("Covariances:\n{:?}", self.covarience_matrices);
        }

        for i in 1..num_iterations{
            self.e_step(&dataset);
            self.m_step();

            if debug && i % report_step == 0 {
                println!("\nIteration {}:", i);
                println!("Means:\n{:?}", self.cluster_means);
                println!("Cluster distributions:\n{:?}", self.curr_cluster_dist);

                // Print first few weights to see cluster assignments
                println!("Sample weights (first 5 points):");
                for j in 0..5 {
                    println!("Point {:?} weights: {:?}",
                            dataset[j],
                            self.weights.slice(s![j, ..]));
                }
            }
        }
        for row in self.weights.rows()
        {
            results.push(argmax(&row).unwrap());
        }

        results
    }

    pub fn new(num_clusters: usize, num_features: usize, num_instances: usize, dataset: &Vec<Vec<f64>>) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_vec(
            (num_instances, num_clusters),
            vec![1.0 / num_clusters as f64; num_clusters * num_instances],
        );
        let curr_cluster_dist = Array1::from_vec(
            (0..num_clusters).map(|_| 1.0 / num_clusters as f64).collect(),
        );
        
        // Initialize covariance matrices with small random noise
        let mut cov_matrices = Array3::zeros((num_clusters, num_features, num_features));
        for j in 0..num_clusters {
            for i in 0..num_features {
                cov_matrices[[j, i, i]] = 1.0 + rng.gen::<f64>() * 0.1;  // Add small random noise
            }
        }
        
        // Initialize means by randomly selecting points from dataset
        let dataset_nd = Array2::from_shape_vec(
            (dataset.len(), dataset[0].len()),
            dataset.iter().flat_map(|v| v.iter()).cloned().collect(),
        ).unwrap();
        
        // K-means++ initialization for means
        let mut means = Array2::zeros((num_clusters, num_features));
        
        // Choose first centroid randomly
        let first_idx = rng.gen_range(0..dataset.len());
        means.slice_mut(s![0, ..]).assign(&dataset_nd.slice(s![first_idx, ..]));
        
        // Choose remaining centroids
        for k in 1..num_clusters {
            let mut distances = Vec::with_capacity(dataset.len());
            
            // Calculate distances to nearest existing centroid for each point
            for i in 0..dataset.len() {
                let point = dataset_nd.slice(s![i, ..]);
                let min_dist = (0..k).map(|j| {
                    let centroid = means.slice(s![j, ..]);
                    let diff = &point - &centroid;
                    diff.dot(&diff)
                }).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                distances.push(min_dist);
            }
            
            // Choose next centroid with probability proportional to distance squared
            let total_dist: f64 = distances.iter().sum();
            let mut r = rng.gen::<f64>() * total_dist;
            let mut chosen_idx = 0;
            
            for (idx, dist) in distances.iter().enumerate() {
                r -= dist;
                if r <= 0.0 {
                    chosen_idx = idx;
                    break;
                }
            }
            
            means.slice_mut(s![k, ..]).assign(&dataset_nd.slice(s![chosen_idx, ..]));
        }

        Self {
            cluster_means: means,
            curr_cluster_dist,
            covarience_matrices: cov_matrices,
            weights: weights.unwrap(),
            dataset: dataset_nd,
        }
    }
}

pub fn argmax(arr: &ArrayView1<f64>) -> Option<u8> {
    if arr.is_empty() {
        return None;
    }

    let mut max_value = arr[0];
    let mut max_index:u8= 0;

    for (idx, &value) in arr.iter().enumerate() {
        if value > max_value {
            max_value = value;
            max_index = idx as u8;
        }
    }

    Some(max_index)
}

pub fn GaussianDensity(cluster_mean: &Array1<f64>, cov_mat: &Array2<f64>, x: &Array1<f64>) -> f64 {
    let cov_mat_na = DMatrix::from_row_slice(
        cov_mat.nrows(),
        cov_mat.ncols(),
        cov_mat.as_slice().unwrap(),
    );

    // Add small regularization term to ensure positive definiteness
    let epsilon = 1e-6;
    let n = cov_mat_na.nrows();
    let regularization = DMatrix::identity(n, n) * epsilon;
    let regularized_cov = cov_mat_na + regularization;

    let cholesky = match Cholesky::new(regularized_cov) {
        Some(c) => c,
        None => {
            // If Cholesky decomposition fails, return very small probability
            return f64::MIN_POSITIVE;
        }
    };

    let inv = cholesky.inverse();
    let inv = Array2::from_shape_vec(
        (inv.nrows(), inv.ncols()),
        inv.iter().cloned().collect(),
    ).unwrap();

    let det = cholesky.determinant();
    let diff = x - cluster_mean;

    let dim = x.len() as f64;
    let normalization = (2.0 * std::f64::consts::PI).powf(dim / 2.0) * det.sqrt();
    let exponent = -0.5 * diff.dot(&inv.dot(&diff));

    // Clamp exponent to avoid overflow
    let exponent_clamped = exponent.max(-709.0);  // ln(f64::MIN_POSITIVE) ≈ -709.0
    (1.0 / normalization) * exponent_clamped.exp()
}
impl fmt::Display for GaussianMix{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(
            f,
            "Mixture of Gaussians Model:\n\n
            Means: {:?}\n\n\
            Covariance Matrices (per cluster):{:?}\n\n
            Cluster Distribution:{:?}\n\n",
            self.cluster_means,
            self.covarience_matrices,
            self.curr_cluster_dist
        )?;

        for i in 0..self.cluster_means.len() {
            writeln!(f, "Cluster {}:\n", i)?;
            let cov_matrix = self.covarience_matrices.slice(s![i, .., ..]);
            writeln!(f, "Covariance Matrix:\n")?;
            for row in cov_matrix.rows() {
                writeln!(f, "{:?}", row)?;
            }
            writeln!(f)?;
        }

        write!(f, "Cluster Distribution: {:?}", self.curr_cluster_dist)
    }
}
