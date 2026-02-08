// use nalgebra::zero;
use nalgebra::{
    ArrayStorage, Const, Matrix,
    /* , Matrix3, Matrix3x4, OMatrix,*/ Vector, /*  matrix, vector*/
};
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
//use std::thread::{self, JoinHandle};

use plotters::style::full_palette::PURPLE;
use rand::Rng;
// use rand_distr::num_traits::abs;

// use std::iter::Enumerate;
// use std::default;

// use std::process::Output;
use rand_distr::Distribution;
type MatrixNM<const N: usize, const M: usize> =
    Matrix<f64, Const<N>, Const<M>, ArrayStorage<f64, N, M>>;
type VectorN<const N: usize> = Vector<f64, Const<N>, ArrayStorage<f64, N, 1>>;

struct LossCategoricalCrossentropy<const N: usize, const M: usize> {
    dinputs: MatrixNM<N, M>,
}

impl<const N: usize, const M: usize> LossCategoricalCrossentropy<N, M> {
    pub fn new() -> Self {
        let dinput = MatrixNM::<N, M>::zeros();
        Self { dinputs: dinput }
    }

    fn forward<const T: usize>(&self, y_pred: MatrixNM<N, M>, y_true: MatrixNM<T, 1>) -> f64 {
        //correct_cofidences=y_pred[range(N),y_true]
        //let mut correct_cofidences = Vec::<f64>::new();

        let mut mean = 0.0;
        for i in 0..y_pred.nrows() {
            let row = y_pred.row(i);
            let row_clip = row[(0, y_true[i] as usize)].max(1e-7).min(1. - 1e-7);
            mean -= row_clip.ln();
        }
        mean = mean / y_pred.nrows() as f64;
        mean
    }

    fn backward(&mut self, dvalues: MatrixNM<N, M>, y_true: VectorN<N>) {
        let samples = N;
        //let lables = M;
        let mut y_true_c = MatrixNM::<N, M>::zeros();
        for ii in 0..samples {
            let mut ans = VectorN::<M>::zeros();
            ans[(y_true[ii]) as usize] = 1.0;
            y_true_c.set_row(ii, &ans.transpose());
        }
        for ii in 0..N {
            for jj in 0..M {
                self.dinputs[(ii, jj)] =
                    (-y_true_c[(ii, jj)] / dvalues[(ii, jj)]) / (samples as f64);
            }
        }
    }
}

fn accuracy<const N: usize, const M: usize, const T: usize>(
    y_pred: MatrixNM<N, M>,
    y_true: MatrixNM<T, 1>,
) -> f64 {
    let mut acc = 0.0;
    for i in 0..y_pred.nrows() {
        let row = y_pred.row(i);
        let (idx, _) = row.transpose().argmax();
        if idx == y_true[i] as usize {
            acc += 1.0;
        }
    }
    let acc = acc / y_pred.nrows() as f64;
    acc
}

fn vertical_data<const SAMPLES: usize, const CLASSES: usize, const TOTAL: usize>()
-> (MatrixNM<TOTAL, 2>, VectorN<TOTAL>) {
    assert_eq!(TOTAL, SAMPLES * CLASSES);
    let mut x = MatrixNM::<TOTAL, 2>::zeros();
    let mut y = VectorN::zeros();
    let mut rng = rand::rng();

    for class_number in 0..CLASSES {
        let start = SAMPLES * class_number;

        for i in 0..SAMPLES {
            let row = start + i;

            let x_offset =
                rng.sample::<f64, _>(rand_distr::StandardNormal) * 0.1 + (class_number as f64) / 3.;
            let y_offset = rng.sample::<f64, _>(rand_distr::StandardNormal) * 0.1 + 0.5;

            x[(row, 0)] = x_offset;
            x[(row, 1)] = y_offset;
            y[row] = class_number as f64;
        }
    }

    (x, y)
}

fn spiral_data<const SAMPLES: usize, const CLASSES: usize, const TOTAL: usize>()
-> (MatrixNM<TOTAL, 2>, VectorN<TOTAL>) {
    assert_eq!(TOTAL, SAMPLES * CLASSES);
    let mut x = MatrixNM::<TOTAL, 2>::zeros();
    let mut y = VectorN::zeros();
    let normal = rand_distr::Normal::new(0., 0.2).unwrap();
    let mut rng = rand::rng();

    for class_number in 0..CLASSES {
        for i in 0..SAMPLES {
            let ix = class_number * SAMPLES + i;
            let r = i as f64 / (SAMPLES as f64 - 1.);
            let t = class_number as f64 * 4.
                + (i as f64 / (SAMPLES as f64 - 1.)) * 4.
                + normal.sample(&mut rng);
            x[(ix, 0)] = r * (t * 2.5).sin();
            x[(ix, 1)] = r * (t * 2.5).cos();
            y[ix] = class_number as f64;
        }
    }
    (x, y)
}

struct ActivationReLu<const N: usize, const M: usize> {
    inputs: MatrixNM<N, M>,
    dinputs: MatrixNM<N, M>,
}
impl<const N: usize, const M: usize> ActivationReLu<N, M> {
    pub fn new() -> Self {
        let input = MatrixNM::<N, M>::zeros();

        let dinput = MatrixNM::<N, M>::zeros();

        Self {
            inputs: input,

            dinputs: dinput,
        }
    }
    pub fn forward(&mut self, input: MatrixNM<N, M>) -> MatrixNM<N, M> {
        self.inputs = input;
        let mut output = input;
        for i in 0..N {
            for j in 0..M {
                if input[(i, j)] > 0.0 {
                    output[(i, j)] = input[(i, j)];
                } else {
                    output[(i, j)] = 0.0;
                }
            }
        }
        output
        //
    }
    pub fn backward(&mut self, dvalues: MatrixNM<N, M>) -> MatrixNM<N, M> {
        self.dinputs = dvalues.clone();
        for ii in 0..N {
            for jj in 0..M {
                if self.inputs[(ii, jj)] <= 0.0 {
                    self.dinputs[(ii, jj)] = 0.0;
                }
            }
        }
        return self.dinputs;
    }
}

struct ActivationSoftmax<const I: usize, const J: usize> {
    inputs: MatrixNM<I, J>,
    outputs: MatrixNM<I, J>,
    dinputs: MatrixNM<I, J>,
}
impl<const I: usize, const J: usize> ActivationSoftmax<I, J> {
    pub fn new() -> Self {
        let input = MatrixNM::<I, J>::zeros();
        let output = MatrixNM::<I, J>::zeros();
        let dinput = MatrixNM::<I, J>::zeros();

        Self {
            inputs: input,
            outputs: output,
            dinputs: dinput,
        }
    }
    pub fn forward(&mut self, input: MatrixNM<I, J>) -> MatrixNM<I, J> {
        self.inputs = input;
        let mut result = input;
        for i in 0..input.nrows() {
            let row = input.row(i);
            let max_val = row.max();
            let exp_row = row.map(|v| (v - max_val).exp());
            let sum_exp = exp_row.sum();
            let softmax_row = exp_row / sum_exp;
            result.set_row(i, &softmax_row);
        }
        self.outputs = result.clone();
        result
    }
    // pub fn backward(&mut self, dvalues: MatrixNM<I, J>) {
    //     let temp = MatrixNM::<I, J>::zeros();
    //     self.dinputs = temp;
    //     //let single_output = MatrixNM::<I, J>::zeros();

    //     //let single_dvalues = MatrixNM::<I, J>::zeros();
    //     for (index, (single_output, single_dvalues)) in
    //         zip(self.outputs.row_iter(), dvalues.row_iter()).enumerate()
    //     {
    //         let single_output = single_output.transpose();
    //         let jacobian_matrix =
    //             MatrixNM::from_diagonal(&single_output) - single_output * single_output.transpose();
    //         let temp1 = jacobian_matrix * single_dvalues.transpose();
    //         self.dinputs.set_row(index, &temp1.transpose());
    //     }
    // }
}
#[derive(Clone)]
struct Layer<const M: usize, const I: usize, const N: usize> {
    inputs: Box<MatrixNM<M, I>>,
    weights: Box<MatrixNM<I, N>>,
    biases: Box<VectorN<N>>,
    dinputs: Box<MatrixNM<M, I>>,
    dweights: Box<MatrixNM<I, N>>,
    dbiases: Box<VectorN<N>>,

    weights_momentum: Box<MatrixNM<I, N>>,
    biases_momentum: Box<VectorN<N>>,
    weights_cache: Box<MatrixNM<I, N>>,
    biases_cache: Box<VectorN<N>>,

    weights_l1: f64,
    weights_l2: f64,
    biases_l1: f64,
    biases_l2: f64,
}
impl<const M: usize, const I: usize, const N: usize> Layer<M, I, N> {
    pub fn new(weights_l1: f64, weights_l2: f64, biases_l1: f64, biases_l2: f64) -> Self {
        let input = MatrixNM::<M, I>::zeros();
        //let weights = 0.01 * MatrixNM::<I, N>::new_random();
        let mut rng = rand::rng();
        let dist = rand_distr::Normal::new(0., 0.01).unwrap();
        let weights = MatrixNM::<I, N>::from_fn(|_, _| dist.sample(&mut rng));
        let biases = VectorN::<N>::zeros();
        let dinput = MatrixNM::<M, I>::zeros();
        let dweights = MatrixNM::<I, N>::zeros();
        let dbiases = VectorN::<N>::zeros();
        let weights_momentum = MatrixNM::<I, N>::zeros();
        let biases_momentum = VectorN::<N>::zeros();
        let weights_cache = MatrixNM::<I, N>::zeros();
        let biases_cache = VectorN::<N>::zeros();

        Self {
            inputs: Box::new(input),
            weights: Box::new(weights),
            biases: Box::new(biases),
            dinputs: Box::new(dinput),
            dweights: Box::new(dweights),
            dbiases: Box::new(dbiases),
            weights_momentum: Box::new(weights_momentum),
            biases_momentum: Box::new(biases_momentum),
            weights_cache: Box::new(weights_cache),
            biases_cache: Box::new(biases_cache),
            weights_l1: weights_l1,
            weights_l2: weights_l2,
            biases_l1: biases_l1,
            biases_l2: biases_l2,
        }
    }
    //M 입력의수
    pub fn forward(&mut self, inputs: MatrixNM<M, I>) -> MatrixNM<M, N> {
        //let weights = self.weights.transpose();
        self.inputs = Box::new(inputs);
        let m: [_; M] = std::array::from_fn(|_| *self.biases);
        let m = MatrixNM::<N, M>::from_columns(&m);
        //let m = Matrix3::from_columns(&[self.biases, self.biases, self.biases]);
        //let m = m.transpose();
        inputs * *self.weights + m.transpose()
    }

    pub fn backward(&mut self, dvalues: MatrixNM<M, N>) -> MatrixNM<M, I> {
        self.dweights = Box::new(self.inputs.transpose() * dvalues);
        let mut sum = MatrixNM::<N, 1>::zeros();
        self.dweights = Box::new(
            *self.dweights + self.weights_l1 * self.weights.map(|w| if w >= 0. { 1. } else { -1. }),
        );
        self.dweights = Box::new(*self.dweights + 2. * self.weights_l2 * *self.weights);
        for ii in dvalues.row_iter() {
            for jj in 0..ii.len() {
                sum[jj] += ii[jj];
            }
        }
        self.dbiases = Box::new(sum);
        self.dbiases = Box::new(
            *self.dbiases + self.biases_l1 * self.biases.map(|w| if w >= 0. { 1. } else { -1. }),
        );
        self.dbiases = Box::new(*self.dbiases + 2. * self.biases_l2 * *self.biases);
        self.dinputs = Box::new(dvalues * self.weights.transpose());
        return *self.dinputs;
    }
    pub fn l1<const A: usize, const B: usize>(&self, weights: &MatrixNM<A, B>) -> f64 {
        let mut aa = 0.0;
        for ii in 0..weights.nrows() {
            let row = weights.row(ii);
            for jj in row {
                aa += jj.abs();
            }
        }
        return aa;
    }
    pub fn l2<const A: usize, const B: usize>(&self, weights: &MatrixNM<A, B>) -> f64 {
        let mut aa = 0.0;
        for ii in 0..weights.nrows() {
            let row = weights.row(ii);
            for jj in row {
                aa += jj.powi(2);
            }
        }
        return aa;
    }
    pub fn regulation(&self) -> f64 {
        let sum = self.weights_l1 * self.l1(&self.weights)
            + self.biases_l1 * self.l1(&self.biases)
            + self.weights_l2 * self.l2(&self.weights)
            + self.biases_l2 * self.l2(&self.biases);
        sum
    }
}

struct DropoutLayer<const I: usize, const J: usize> {
    binary_mask: MatrixNM<I, J>,
    p: f64,
}
impl<const I: usize, const J: usize> DropoutLayer<I, J> {
    pub fn new(p: f64) -> Self {
        let binary_mask = MatrixNM::<I, J>::zeros();
        Self {
            p: p,
            binary_mask: binary_mask,
        }
    }

    fn forward(&mut self, input: MatrixNM<I, J>) -> MatrixNM<I, J> {
        for ii in 0..I {
            for jj in 0..J {
                if !rand::random_bool(self.p) {
                    self.binary_mask[(ii, jj)] = 1.0 / (1.0 - self.p);
                }
            }
        }
        return input.zip_map(&self.binary_mask, |input, mask| input * mask);
    }
    fn backward(&self, dvalue: MatrixNM<I, J>) -> MatrixNM<I, J> {
        return dvalue.zip_map(&self.binary_mask, |dvalue, mask| dvalue * mask);
    }
}

struct Activation_Softmax_Loss_CategoricalCrossentropy<const I: usize, const J: usize> {
    activation: ActivationSoftmax<I, J>,
    loss: LossCategoricalCrossentropy<I, J>,
    output: MatrixNM<I, J>,
    dinputs: MatrixNM<I, J>,
}
impl<const I: usize, const J: usize> Activation_Softmax_Loss_CategoricalCrossentropy<I, J> {
    fn new() -> Self {
        let activation1 = ActivationSoftmax::<I, J>::new();
        let loss1 = LossCategoricalCrossentropy::<I, J>::new();
        let output1 = MatrixNM::<I, J>::zeros();
        let dinput1 = MatrixNM::<I, J>::zeros();

        Self {
            activation: activation1,
            loss: loss1,
            output: output1,
            dinputs: dinput1,
        }
    }
    fn forward(&mut self, inputs: MatrixNM<I, J>, y_true: MatrixNM<I, 1>) -> f64 {
        self.output = self.activation.forward(inputs);
        return self.loss.forward(self.output, y_true).clone();
    }

    fn backward(&mut self, dvalues: MatrixNM<I, J>, y_true: MatrixNM<I, 1>) -> MatrixNM<I, J> {
        let samples = dvalues.nrows();
        self.dinputs = dvalues.clone();
        for ii in 0..I {
            self.dinputs[(ii, y_true[(ii, 0)] as usize)] += -1.0;
        }
        self.dinputs = self.dinputs / samples as f64;
        return self.dinputs;
    }
}

trait Optimizer {
    fn update_params<const M: usize, const I: usize, const N: usize>(
        &self,
        layer: &mut Layer<M, I, N>,
        num: usize,
    );
}

struct Optimizer_SGD;
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
struct Optimizer_Adagrad;
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

struct Optimizer_RMS;
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
struct Optimizer_Adam;
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
fn graph(
    pointx: MatrixNM<300, 2>,
    what: MatrixNM<300, 1>,
    testx: MatrixNM<300, 2>,
    what_t: MatrixNM<300, 1>,
    mut dense1: Vec<Vec<(Layer<300, 2, 64>, Layer<300, 64, 3>)>>,
    lossnacc: Vec<Vec<(f64, f64)>>,
    lossnacc_t: Vec<Vec<(f64, f64)>>,
) -> Result<(), Box<dyn std::error::Error>> {
    for ii in 0..=10 {
        let p = format!("image/{}.png", ii);

        let root = BitMapBackend::new(&p, (1000, 1920)).into_drawing_area();
        root.fill(&WHITE)?;
        let (areas) = root.split_evenly((4, 1));
        for (index, root) in areas.iter().enumerate() {
            let (left, right) = root.split_horizontally(600);
            let mut chart = ChartBuilder::on(&left)
                // .x_label_area_size(40)
                // .y_label_area_size(40)
                .build_cartesian_2d(-1f64..1f64, -1f64..1f64)?;
            let px = pointx.column(0);
            let py = pointx.column(1);
            let mut pxy = Vec::new();
            predict(
                &chart,
                &mut dense1[index][ii].0.clone(),
                &mut dense1[index][ii].1.clone(),
            );
            for ii in 0..pointx.nrows() {
                let mut color = RGBColor(0, 0, 0);
                if what[ii] as usize == 0 {
                    color = RGBColor(255, 0, 0);
                }
                if what[ii] as usize == 1 {
                    color = RGBColor(0, 255, 0);
                }
                if what[ii] as usize == 2 {
                    color = RGBColor(0, 0, 255);
                }
                pxy.push((px[ii], py[ii], color));
            }
            let px = testx.column(0);
            let py = testx.column(1);
            let mut pxy_t = Vec::new();
            for ii in 0..testx.nrows() {
                let mut color = RGBColor(0, 0, 0);
                if what_t[ii] as usize == 0 {
                    color = RGBColor(255, 0, 0);
                }
                if what_t[ii] as usize == 1 {
                    color = RGBColor(0, 255, 0);
                }
                if what_t[ii] as usize == 2 {
                    color = RGBColor(0, 0, 255);
                }
                pxy_t.push((px[ii], py[ii], color));
            }
            chart
                .configure_mesh()
                .disable_axes()
                .disable_mesh()
                .draw()?;
            chart.draw_series(
                pxy.iter()
                    .map(|(x, y, c)| Circle::new((*x, *y), 2, c.filled())),
            )?;
            chart.draw_series(
                pxy_t
                    .iter()
                    .map(|(x, y, c)| Circle::new((*x, *y), 2, c.filled())),
            )?;
            // let loss_max = lossnacc[index]
            //     .iter()
            //     .map(|x| x.0)
            //     .max_by(|a, b| a.total_cmp(b))
            //     .unwrap();

            let (up, down) = right.split_vertically(240);

            let mut loss_chart = ChartBuilder::on(&up)
                .x_label_area_size(10)
                .y_label_area_size(10)
                .build_cartesian_2d(0..10000usize, 0f64..1.0)?;
            let mut acc_chart = ChartBuilder::on(&down)
                .x_label_area_size(10)
                .y_label_area_size(10)
                .build_cartesian_2d(0..10000usize, 0f64..1f64)?;
            loss_chart.configure_mesh().draw()?;
            loss_chart.draw_series(LineSeries::new(
                lossnacc[index]
                    .iter()
                    .enumerate()
                    .map(|loss| (loss.0, loss.1.0)),
                &RED,
            ))?;
            loss_chart.draw_series(LineSeries::new(
                lossnacc_t[index]
                    .iter()
                    .enumerate()
                    .map(|loss| (loss.0, loss.1.0)),
                &PURPLE,
            ))?;

            acc_chart.configure_mesh().draw()?;
            acc_chart.draw_series(LineSeries::new(
                lossnacc[index]
                    .iter()
                    .enumerate()
                    .map(|loss| (loss.0, loss.1.1)),
                &RED,
            ))?;
            acc_chart.draw_series(LineSeries::new(
                lossnacc_t[index]
                    .iter()
                    .enumerate()
                    .map(|loss| (loss.0, loss.1.1)),
                &PURPLE,
            ))?;
        }
    }
    Ok(())
}
fn predict(
    chart: &ChartContext<'_, BitMapBackend<'_>, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    dense1: &mut Layer<300, 2, 64>,
    dense2: &mut Layer<300, 64, 3>,
) {
    let mut activation1 = ActivationReLu::<300, 64>::new();

    let mut activation2 = ActivationSoftmax::<300, 3>::new();

    for ii in -50..50 {
        for jj in -50..50 {
            let mut mt = MatrixNM::<300, 2>::zeros();
            mt[(0, 0)] = ii as f64 / 50 as f64;
            mt[(0, 1)] = jj as f64 / 50 as f64;

            let aa = dense1.forward(mt);
            let bb = activation1.forward(aa);

            let cc = dense2.forward(bb);
            let result = activation2.forward(cc);

            let result = result.row(0).transpose();
            let color = RGBAColor(
                (result[0] * 255f64) as u8,
                (result[1] * 255 as f64) as u8,
                (result[2] * 255.0) as u8,
                0.3,
            );
            //let mut color = [RED, GREEN, BLUE][result].to_rgba();
            //color.3 = (result.1) * 0.1;

            let circle = Circle::new((mt[(0, 0)], mt[(0, 1)]), 4, color.filled());
            chart.plotting_area().draw(&circle).expect("error");
        }
    }
}

fn forwarding(
    dense1: &mut Layer<300, 2, 64>,
    dense2: &mut Layer<300, 64, 3>,
    act1: &mut ActivationReLu<300, 64>,
    act2: &mut ActivationSoftmax<300, 3>,
    drop1: &mut DropoutLayer<300, 64>,
    x: MatrixNM<300, 2>,
    y: MatrixNM<300, 1>,
    optimizer: &impl Optimizer,
    iteration: usize,
) -> (f64, f64) {
    let mut loss_activation =
        Box::new(Activation_Softmax_Loss_CategoricalCrossentropy::<300, 3>::new());
    // let mut dense1 = dense1.clone();
    // let mut dense2 = dense2.clone();
    let aa = dense1.forward(x);
    let bb = act1.forward(aa);
    let dl = drop1.forward(bb);
    let cc = dense2.forward(dl);
    let result = act2.forward(cc);

    let loss = loss_activation.forward(cc, y) + dense1.regulation() + dense2.regulation();

    let acc = accuracy(result, y);
    let dinputs = loss_activation.backward(result, y);
    let d2b = dense2.backward(dinputs);
    let dlb = drop1.backward(d2b);
    let a1b = act1.backward(dlb);
    dense1.backward(a1b);
    optimizer.update_params::<300, 2, 64>(dense1, iteration);
    optimizer.update_params::<300, 64, 3>(dense2, iteration);
    //println!("{loss:.3},{acc:.3}");
    return (loss, acc);
}

fn test_forwarding(
    dense1: &mut Layer<300, 2, 64>,
    dense2: &mut Layer<300, 64, 3>,
    act1: &mut ActivationReLu<300, 64>,
    act2: &mut ActivationSoftmax<300, 3>,
    x: MatrixNM<300, 2>,
    y: MatrixNM<300, 1>,
) -> (f64, f64) {
    let mut loss_activation =
        Box::new(Activation_Softmax_Loss_CategoricalCrossentropy::<300, 3>::new());
    let aa = dense1.forward(x);
    let bb = act1.forward(aa);
    let cc = dense2.forward(bb);
    let result = act2.forward(cc);
    let loss = loss_activation.forward(cc, y);
    let acc = accuracy(result, y);
    return (loss, acc);
}

fn randlearn(
    mut dense1: Layer<300, 2, 64>,
    mut dense2: Layer<300, 64, 3>,

    x: MatrixNM<300, 2>,
    y: MatrixNM<300, 1>,
    x_test: MatrixNM<300, 2>,
    y_test: MatrixNM<300, 1>,
    optimizer1: impl Optimizer,
) -> (
    Vec<(f64, f64)>,
    Vec<(f64, f64)>,
    Vec<(Layer<300, 2, 64>, Layer<300, 64, 3>)>,
) {
    let mut activation1 = Box::new(ActivationReLu::<300, 64>::new());
    let mut activation2 = Box::new(ActivationSoftmax::<300, 3>::new());
    let mut drop1 = DropoutLayer::new(0.1);
    // let mut loss_activation =
    //     Box::new(Activation_Softmax_Loss_CategoricalCrossentropy::<300, 3>::new());
    //let mut th = Vec::new();
    let mut lns_series = Vec::<(f64, f64)>::new();
    let mut lns_series_t = Vec::<(f64, f64)>::new();
    let mut ds_needed = Vec::<(Layer<300, 2, 64>, Layer<300, 64, 3>)>::new();
    //let mut lossnacc_s = Vec::<(f64, f64)>::new();
    for iteration in 0..=10000 {
        //let optimizer2 = Optimizer_Adam;
        let (loss, acc) = forwarding(
            &mut dense1,
            &mut dense2,
            &mut activation1,
            &mut activation2,
            &mut drop1,
            x,
            y,
            &optimizer1,
            iteration,
        );
        println!("loss:{loss},acc: {acc}");
        lns_series.push((loss.clone(), acc.clone()));

        let (loss, acc) = test_forwarding(
            &mut dense1,
            &mut dense2,
            &mut activation1,
            &mut activation2,
            x_test,
            y_test,
        );
        lns_series_t.push((loss.clone(), acc.clone()));
        if iteration % 1000 == 0 {
            ds_needed.push((dense1.clone(), dense2.clone()));
        }
    }
    return (lns_series, lns_series_t, ds_needed);
    // for th in th {
    //     th.join().unwrap();
    // }
}

fn main() {
    let (x, y) = spiral_data::<100, 3, 300>();
    let (x_test, y_test) = spiral_data::<100, 3, 300>();
    let mut dense1 = Layer::<300, 2, 64>::new(0.0, 5e-4, 0.0, 5e-4);
    //let activation1 = ActivationReLu::<300, 3>::new();
    let mut dense2 = Layer::<300, 64, 3>::new(0.0, 0.0, 0.0, 0.0);

    let (lnss1, lnss1_t, ds1) = randlearn(
        dense1.clone(),
        dense2.clone(),
        x,
        y,
        x_test,
        y_test,
        Optimizer_SGD,
    );
    let (lnss2, lnss2_t, ds2) = randlearn(
        dense1.clone(),
        dense2.clone(),
        x,
        y,
        x_test,
        y_test,
        Optimizer_Adagrad,
    );
    let (lnss3, lnss3_t, ds3) = randlearn(
        dense1.clone(),
        dense2.clone(),
        x,
        y,
        x_test,
        y_test,
        Optimizer_RMS,
    );
    let (lnss4, lnss4_t, ds4) = randlearn(
        dense1.clone(),
        dense2.clone(),
        x,
        y,
        x_test,
        y_test,
        Optimizer_Adam,
    );
    let lnss = vec![lnss1, lnss2, lnss3, lnss4];
    let lnss_t = vec![lnss1_t, lnss2_t, lnss3_t, lnss4_t];
    let ds = vec![ds1, ds2, ds3, ds4];
    graph(x, y, x_test, y_test, ds, lnss, lnss_t).unwrap();
}
