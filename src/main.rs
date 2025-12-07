use nalgebra::zero;
use nalgebra::{
    ArrayStorage, Const, Matrix,
    /* , Matrix3, Matrix3x4, OMatrix,*/ Vector, /*  matrix, vector*/
};
use plotters::prelude::*;
use rand::Rng;
use std::iter::Enumerate;
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

    fn forward<const T: usize>(self, y_pred: MatrixNM<N, M>, y_true: MatrixNM<T, 1>) -> f64 {
        //correct_cofidences=y_pred[range(N),y_true]
        //let mut correct_cofidences = Vec::<f64>::new();
        let mut mean = 0.0;
        for i in 0..y_pred.nrows() {
            let row = y_pred.row(i);
            mean -= row[(0, y_true[i] as usize)].ln();
        }
        mean = mean / y_pred.nrows() as f64;
        mean
    }

    fn backward(mut self, dvalues: MatrixNM<N, M>, y_true: VectorN<N>) {
        let samples = N;
        let lables = M;
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
struct Layer<const M: usize, const I: usize, const N: usize> {
    inputs: MatrixNM<M, I>,
    weights: MatrixNM<I, N>,
    biases: VectorN<N>,
    dinputs: MatrixNM<M, I>,
    dweights: MatrixNM<I, N>,
    dbiases: VectorN<N>,
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
    pub fn forward(mut self, input: MatrixNM<N, M>) -> MatrixNM<N, M> {
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
    pub fn backward(mut self, dvalues: MatrixNM<N, M>) {
        self.dinputs = dvalues.clone();
        for ii in 0..N {
            for jj in 0..M {
                if self.inputs[(ii, jj)] <= 0.0 {
                    self.dinputs[(ii, jj)] = 0.0;
                }
            }
        }
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
    pub fn forward(mut self, input: MatrixNM<I, J>) -> MatrixNM<I, J> {
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
    pub fn backward(mut self, dvalues: MatrixNM<I, J>) {
        let temp = MatrixNM::<I, J>::zeros();
        self.dinputs = temp;
        let single_output = MatrixNM::<I, J>::zeros();

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

impl<const M: usize, const I: usize, const N: usize> Layer<M, I, N> {
    pub fn new() -> Self {
        let input = MatrixNM::<M, I>::new_random();
        let weights = MatrixNM::<I, N>::new_random();
        let biases = VectorN::<N>::new_random();
        let dinput = MatrixNM::<M, I>::new_random();
        let dweights = MatrixNM::<I, N>::new_random();
        let dbiases = VectorN::<N>::new_random();

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

    pub fn backward(mut self, dvalues: MatrixNM<M, N>) {
        self.dweights = self.inputs.transpose() * dvalues;
        let mut sum = MatrixNM::<N, 1>::zeros();
        for ii in dvalues.row_iter() {
            for jj in 0..ii.len() {
                sum[jj] += ii[jj];
            }
        }
        self.dbiases = sum;
        self.dinputs = dvalues * self.weights.transpose();
    }
}

// fn graph() -> Result<(), Box<dyn std::error::Error>>{
//     let root = BitMapBackend::new("plotters-doc-data/0.png", (640, 480)).into_drawing_area();
//     root.fill(&WHITE)?;
//     let mut chart = ChartBuilder::on(&root);

//     Ok(())
// }

fn main() {
    // let inputs1 = matrix![
    //     1.0, 2.0;
    //     2.0,5.0;
    //     4.0,3.0;
    // ];
    // let layer = Layer::<4, 4>::new();
    // println!("{}", layer.forward(inputs));
    let (x, y) = spiral_data::<100, 3, 300>();
    let mut dense1 = Layer::<300, 2, 3>::new();
    let activation1 = ActivationReLu::<300, 3>::new();
    let mut dense2 = Layer::<300, 3, 3>::new();
    let activation2 = ActivationSoftmax::<300, 3>::new();
    let aa = dense1.forward(x);
    let bb = activation1.forward(aa);
    let cc = dense2.forward(bb);
    let result = activation2.forward(cc);
    let loss = LossCategoricalCrossentropy::<300, 3>::new();
    let loss = loss.forward(result, y);
    let acc = accuracy(result, y);
    println!("{loss},{acc}");
    //println!("{result} , {y}");
}
