use nalgebra::{ArrayStorage, Const, Matrix, Matrix3, Matrix3x4, OMatrix, Vector, matrix, vector};
// use std::default;
// use std::iter::zip;
// use std::process::Output;
use rand_distr::Distribution;
type MatrixNM<const N: usize, const M: usize> =
    Matrix<f64, Const<N>, Const<M>, ArrayStorage<f64, N, M>>;
type VectorN<const N: usize> = Vector<f64, Const<N>, ArrayStorage<f64, N, 1>>;

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
struct Layer<const I: usize, const N: usize> {
    //inputs: Matrix3<f64>,
    weights: MatrixNM<I, N>,
    biases: VectorN<N>,
}
struct Activation_ReLu;
impl Activation_ReLu {
    pub fn forward<const N: usize, const M: usize>(self, input: MatrixNM<N, M>) -> MatrixNM<N, M> {
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
}

struct Activation_Softmax;
impl Activation_Softmax {
    pub fn forward<const I: usize, const J: usize>(self, input: MatrixNM<I, J>) -> MatrixNM<I, J> {
        let mut result = input;
        for i in 0..input.nrows() {
            let row = input.row(i);
            let max_val = row.max();
            let exp_row = row.map(|v| (v - max_val).exp());
            let sum_exp = exp_row.sum();
            let softmax_row = exp_row / sum_exp;
            result.set_row(i, &softmax_row);
        }
        result
    }
}

impl<const I: usize, const N: usize> Layer<I, N> {
    pub fn new() -> Self {
        let weights = MatrixNM::<I, N>::new_random();
        let biases = VectorN::<N>::new_random();

        Self {
            weights: weights,
            biases: biases,
        }
    }
    //M 입력의수
    pub fn forward<const M: usize>(self, inputs: MatrixNM<M, I>) -> MatrixNM<M, N> {
        //let weights = self.weights.transpose();
        let m: [_; M] = std::array::from_fn(|_| self.biases.clone());
        let m = MatrixNM::<N, M>::from_columns(&m);
        //let m = Matrix3::from_columns(&[self.biases, self.biases, self.biases]);
        //let m = m.transpose();
        inputs * self.weights + m.transpose()
    }
}

fn main() {
    let inputs1 = matrix![
        1.0, 2.0;
        2.0,5.0;
        4.0,3.0;
    ];
    // let layer = Layer::<4, 4>::new();
    // println!("{}", layer.forward(inputs));
    let (x, y) = spiral_data::<100, 3, 300>();
    let dense1 = Layer::<2, 3>::new();
    let activation1 = Activation_ReLu;
    let dense2 = Layer::<3, 3>::new();
    let activation2 = Activation_Softmax;
    let aa = dense1.forward(x);
    let bb = activation1.forward(aa);
    let cc = dense2.forward(bb);
    let result = activation2.forward(cc);
    println!("{result} , {y}");
}
