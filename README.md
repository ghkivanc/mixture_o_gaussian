# Mixture Of Gaussian Models in Rust

This is a pet project I made. It uses crates such as `nalgebra` and `ndarray` for linear algebra. It's poorly optimized, if optimized at all.

## How to Use It

### 1. Creating a Model
Calling `GaussianMixture::new(..)` with appropriate parameters will do. Hopefully, I named the parameters sensibly, so I would like to think it's easy to understand what they mean.

### 2. Loading Data
The code provides a function called `load_data(..)`; you can just specify a file path to get your **unlabeled** data. It only works with tabular data that has numerical values, specifically `f64` values. The CSV should also contain no headers.

### 3. Clustering
Calling `model.fit(..)` with appropriate parameters will do. This will return a `Vec<u8>` of labels, where each element corresponds to which cluster that instance belongs to.

---

**I used a lot of help from LLMs throughout the project. Mostly for debugging.**  
Feel free to fork/contribute if you want to.
