// use nalgebra::zero;
use nalgebra::{
    ArrayStorage, Const, Matrix,
    /* , Matrix3, Matrix3x4, OMatrix,*/ Vector, /*  matrix, vector*/
};
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use std::thread::{self, JoinHandle};

use rand::{Rng, random};
// use rand_distr::num_traits::abs;

// use std::iter::Enumerate;
// use std::default;
use std::iter::zip;

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
    pub fn backward(&mut self, dvalues: MatrixNM<I, J>) {
        let temp = MatrixNM::<I, J>::zeros();
        self.dinputs = temp;
        //let single_output = MatrixNM::<I, J>::zeros();

        //let single_dvalues = MatrixNM::<I, J>::zeros();
        for (index, (single_output, single_dvalues)) in
            zip(self.outputs.row_iter(), dvalues.row_iter()).enumerate()
        {
            let single_output = single_output.transpose();
            let jacobian_matrix =
                MatrixNM::from_diagonal(&single_output) - single_output * single_output.transpose();
            let temp1 = jacobian_matrix * single_dvalues.transpose();
            self.dinputs.set_row(index, &temp1.transpose());
        }
    }
}
#[derive(Clone)]
struct Layer<const M: usize, const I: usize, const N: usize> {
    inputs: MatrixNM<M, I>,
    weights: MatrixNM<I, N>,
    biases: VectorN<N>,
    dinputs: MatrixNM<M, I>,
    dweights: MatrixNM<I, N>,
    dbiases: VectorN<N>,
}
impl<const M: usize, const I: usize, const N: usize> Layer<M, I, N> {
    pub fn new() -> Self {
        let input = MatrixNM::<M, I>::zeros();
        let weights = 0.01 * MatrixNM::<I, N>::new_random();
        let biases = VectorN::<N>::zeros();
        let dinput = MatrixNM::<M, I>::zeros();
        let dweights = MatrixNM::<I, N>::zeros();
        let dbiases = VectorN::<N>::zeros();

        Self {
            inputs: input,
            weights: weights,
            biases: biases,
            dinputs: dinput,
            dweights: dweights,
            dbiases: dbiases,
        }
    }
    //M 입력의수
    pub fn forward(&mut self, inputs: MatrixNM<M, I>) -> MatrixNM<M, N> {
        //let weights = self.weights.transpose();
        self.inputs = inputs;
        let m: [_; M] = std::array::from_fn(|_| self.biases.clone());
        let m = MatrixNM::<N, M>::from_columns(&m);
        //let m = Matrix3::from_columns(&[self.biases, self.biases, self.biases]);
        //let m = m.transpose();
        inputs * self.weights + m.transpose()
    }

    pub fn backward(&mut self, dvalues: MatrixNM<M, N>) -> MatrixNM<M, I> {
        self.dweights = self.inputs.transpose() * dvalues;
        let mut sum = MatrixNM::<N, 1>::zeros();
        for ii in dvalues.row_iter() {
            for jj in 0..ii.len() {
                sum[jj] += ii[jj];
            }
        }
        self.dbiases = sum;
        self.dinputs = dvalues * self.weights.transpose();
        return self.dinputs;
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

struct Optimizer_SGD<const M: usize, const I: usize, const N: usize>;
impl<const M: usize, const I: usize, const N: usize> Optimizer_SGD<M, I, N> {
    fn update_params(&self, layer: &mut Layer<M, I, N>) {
        layer.weights += -layer.dweights;
        layer.biases += -layer.dbiases;
    }
}

fn graph(
    pointx: MatrixNM<300, 2>,
    what: MatrixNM<300, 1>,
    dense1: Layer<300, 2, 3>,
    dense2: Layer<300, 3, 3>,
    num: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let p = format!("image/{}.png", num);
    println!("{}", num);
    let root = BitMapBackend::new(&p, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-1f64..1f64, -1f64..1f64)?;
    let px = pointx.column(0);
    let py = pointx.column(1);
    let mut pxy = Vec::new();
    predict(&chart, dense1, dense2);
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

    chart.configure_mesh().draw()?;
    chart.draw_series(
        pxy.iter()
            .map(|(x, y, c)| Circle::new((*x, *y), 2, c.filled())),
    )?;

    //chart.configure_mesh().draw()?;
    root.present()?;
    Ok(())
}
fn predict(
    chart: &ChartContext<'_, BitMapBackend<'_>, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    mut dense1: Layer<300, 2, 3>,
    mut dense2: Layer<300, 3, 3>,
) {
    let mut activation1 = ActivationReLu::<300, 3>::new();

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

            let result = result.row(0).transpose().argmax();
            let mut color = [RED, GREEN, BLUE][result.0].to_rgba();
            color.3 = 0.1;

            let circle = Circle::new((mt[(0, 0)], mt[(0, 1)]), 4, color.filled());
            chart.plotting_area().draw(&circle).expect("error");
        }
    }
}
fn randlearn(
    mut dense1: Layer<300, 2, 3>,
    mut dense2: Layer<300, 3, 3>,
    x: MatrixNM<300, 2>,
    y: MatrixNM<300, 1>,
) {
    // let mut loss_lowest = 99999.0;
    // let mut best_dense1_w = dense1.weights.clone();
    // let mut best_dense1_b = dense1.biases.clone();
    // let mut best_dense2_w = dense2.weights.clone();
    // let mut best_dense2_b = dense2.biases.clone();
    let mut activation1 = ActivationReLu::<300, 3>::new();
    let mut activation2 = ActivationSoftmax::<300, 3>::new();
    let mut loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy::<300, 3>::new();
    let mut th = Vec::new();
    for iteration in 0..=10000 {
        let loss_func = LossCategoricalCrossentropy::<300, 3>::new();
        // dense1.weights += 0.05 * MatrixNM::new_random();
        // dense1.biases += 0.05 * MatrixNM::new_random();
        // dense2.weights += 0.05 * MatrixNM::new_random();
        // dense2.biases += 0.05 * MatrixNM::new_random();

        let aa = dense1.forward(x);
        let bb = activation1.forward(aa);
        let cc = dense2.forward(bb);
        let result = activation2.forward(cc);
        let loss = loss_activation.forward(result, y);
        let acc = accuracy(result, y);
        //println!("{},{}", result, y);

        let dinputs = loss_activation.backward(result, y);
        let d2b = dense2.backward(dinputs);
        let a1b = activation1.backward(d2b);
        let d1b = dense1.backward(a1b);
        let optimizer1 = Optimizer_SGD::<300, 2, 3>;
        let optimizer2 = Optimizer_SGD::<300, 3, 3>;
        optimizer1.update_params(&mut dense1);
        optimizer2.update_params(&mut dense2);
        println!("{loss:.3},{acc:.3}");

        if iteration % 100 == 0 {
            let d1c = dense1.clone();
            let d2c = dense2.clone();
            th.push(std::thread::spawn(move || {
                graph(x, y, d1c, d2c, iteration).expect("error")
            }));
        }
    }
    for th in th {
        th.join().unwrap();
    }
}

fn main() {
    // let inputs1 = matrix![
    //     1.0, 2.0;
    //     2.0,5.0;
    //     4.0,3.0;
    // ];
    // let layer = Layer::<4, 4>::new();
    // println!("{}", layer.forward(inputs));
    //let (x, y) = vertical_data::<100, 3, 300>();

    let (x, y) = spiral_data::<100, 3, 300>();

    let dense1 = Layer::<300, 2, 3>::new();
    //let activation1 = ActivationReLu::<300, 3>::new();
    let dense2 = Layer::<300, 3, 3>::new();
    //let activation2 = ActivationSoftmax::<300, 3>::new();
    randlearn(dense1, dense2, x, y);
    // let aa = dense1.forward(x);
    // let bb = activation1.forward(aa);
    // let cc = dense2.forward(bb);
    // let result = activation2.forward(cc);
    // let loss = LossCategoricalCrossentropy::<300, 3>::new();

    // let loss = loss.forward(result, y);
    // let acc = accuracy(result, y);
    // println!("{loss},{acc}");
    //println!("{result} , {y}");
}
