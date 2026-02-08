use crate::layer::*;

pub trait Optimizer {
    fn update_params<const M: usize, const I: usize, const N: usize>(
        &self,
        layer: &mut Layer<M, I, N>,
        num: usize,
    );
}

pub struct Optimizer_SGD;
impl Optimizer for Optimizer_SGD {
    fn update_params<const M: usize, const I: usize, const N: usize>(
        &self,
        layer: &mut Layer<M, I, N>,
        num: usize,
    ) {
        let learning_rate = 1.0;
        let learning_decay = 0.001;
        let momentum = 0.9;

        let learning_rate = learning_rate * (1.0 / (1. + learning_decay * (num as f64)));

        let weights_update = momentum * *layer.weights_momentum - *layer.dweights * learning_rate;
        let biases_update = momentum * *layer.biases_momentum - *layer.dbiases * learning_rate;
        layer.weights = Box::new(*layer.weights + weights_update);
        layer.biases = Box::new(*layer.biases + biases_update);
        layer.weights_momentum = Box::new(weights_update);
        layer.biases_momentum = Box::new(biases_update);
    }
}
pub struct Optimizer_Adagrad;
impl Optimizer for Optimizer_Adagrad {
    fn update_params<const M: usize, const I: usize, const N: usize>(
        &self,
        layer: &mut Layer<M, I, N>,
        num: usize,
    ) {
        let learning_rate = 1.0;
        let learning_decay = 1e-4;
        let epsilon = 1e-7;

        let learning_rate = learning_rate * (1.0 / (1. + learning_decay * (num as f64)));
        layer.weights_cache = Box::new(*layer.weights_cache + layer.dweights.map(|aa| aa * aa));

        layer.biases_cache = Box::new(*layer.biases_cache + layer.dbiases.map(|aa| aa * aa));

        layer.weights = Box::new(
            *layer.weights
                - learning_rate
                    * (layer
                        .dweights
                        .zip_map(&*layer.weights_cache, |dweights, weights_cache| {
                            dweights / (weights_cache.sqrt() + epsilon)
                        })),
        );
        layer.biases = Box::new(
            *layer.biases
                - learning_rate
                    * (layer
                        .dbiases
                        .zip_map(&*layer.biases_cache, |dbiases, biases_cache| {
                            dbiases / (biases_cache.sqrt() + epsilon)
                        })),
        );
    }
}

pub struct Optimizer_RMS;
impl Optimizer for Optimizer_RMS {
    fn update_params<const M: usize, const I: usize, const N: usize>(
        &self,
        layer: &mut Layer<M, I, N>,
        num: usize,
    ) {
        let rho = 0.999;
        let learning_rate = 0.02;
        let learning_decay = 1e-5;
        let epsilon = 1e-7;

        let learning_rate = learning_rate * (1.0 / (1. + learning_decay * (num as f64)));

        layer.weights_cache =
            Box::new(*layer.weights_cache * rho + (1.0 - rho) * layer.dweights.map(|aa| aa * aa));

        layer.biases_cache =
            Box::new(*layer.biases_cache * rho + (1.0 - rho) * layer.dbiases.map(|aa| aa * aa));

        layer.weights = Box::new(
            *layer.weights
                - learning_rate
                    * (layer
                        .dweights
                        .zip_map(&*layer.weights_cache, |dweights, weights_cache| {
                            dweights / (weights_cache.sqrt() + epsilon)
                        })),
        );
        layer.biases = Box::new(
            *layer.biases
                - learning_rate
                    * (layer
                        .dbiases
                        .zip_map(&*layer.biases_cache, |dbiases, biases_cache| {
                            dbiases / (biases_cache.sqrt() + epsilon)
                        })),
        );
    }
}
pub struct Optimizer_Adam;
impl Optimizer for Optimizer_Adam {
    fn update_params<const M: usize, const I: usize, const N: usize>(
        &self,
        layer: &mut Layer<M, I, N>,
        num: usize,
    ) {
        let beta_1 = 0.9;
        let beta_2 = 0.999;
        let learning_rate = 0.05;
        let learning_decay = 5e-7;
        let epsilon = 1e-7;

        let learning_rate = learning_rate * (1.0 / (1. + learning_decay * (num as f64)));

        let weights_update = beta_1 * *layer.weights_momentum + *layer.dweights * (1.0 - beta_1);
        let biases_update = beta_1 * *layer.biases_momentum + *layer.dbiases * (1.0 - beta_1);

        layer.weights_momentum = Box::new(weights_update);
        layer.biases_momentum = Box::new(biases_update);

        let weight_momentums_corrected = &layer
            .weights_momentum
            .map(|aa| aa / (1.0 - beta_1.powi(num as i32 + 1)));

        let biases_momentums_corrected = &layer
            .biases_momentum
            .map(|aa| aa / (1.0 - beta_1.powi(num as i32 + 1)));

        layer.weights_cache = Box::new(
            *layer.weights_cache * beta_2 + (1.0 - beta_2) * layer.dweights.map(|aa| aa * aa),
        );

        layer.biases_cache = Box::new(
            *layer.biases_cache * beta_2 + (1.0 - beta_2) * layer.dbiases.map(|aa| aa * aa),
        );

        let weight_cache_corrected = &layer
            .weights_cache
            .map(|aa| aa / (1.0 - beta_2.powi(num as i32 + 1)));

        let biases_cache_corrected = &layer
            .biases_cache
            .map(|aa| aa / (1.0 - beta_2.powi(num as i32 + 1)));

        layer.weights = Box::new(
            *layer.weights
                - learning_rate
                    * (weight_momentums_corrected
                        .zip_map(weight_cache_corrected, |momentum, weights_cache| {
                            momentum / (weights_cache.sqrt() + epsilon)
                        })),
        );
        layer.biases = Box::new(
            *layer.biases
                - learning_rate
                    * (biases_momentums_corrected
                        .zip_map(biases_cache_corrected, |momentum, biases_cache| {
                            momentum / (biases_cache.sqrt() + epsilon)
                        })),
        );
    }
}
