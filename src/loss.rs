use crate::activation::*;
use crate::types::*;
pub struct LossCategoricalCrossentropy<const N: usize, const M: usize> {
    dinputs: MatrixNM<N, M>,
}

impl<const N: usize, const M: usize> LossCategoricalCrossentropy<N, M> {
    pub fn new() -> Self {
        let dinput = MatrixNM::<N, M>::zeros();
        Self { dinputs: dinput }
    }

    pub fn forward<const T: usize>(&self, y_pred: MatrixNM<N, M>, y_true: MatrixNM<T, 1>) -> f64 {
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

    pub fn backward(&mut self, dvalues: MatrixNM<N, M>, y_true: VectorN<N>) {
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

pub fn accuracy<const N: usize, const M: usize, const T: usize>(
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
pub struct Activation_Softmax_Loss_CategoricalCrossentropy<const I: usize, const J: usize> {
    activation: ActivationSoftmax<I, J>,
    loss: LossCategoricalCrossentropy<I, J>,
    output: MatrixNM<I, J>,
    dinputs: MatrixNM<I, J>,
}
impl<const I: usize, const J: usize> Activation_Softmax_Loss_CategoricalCrossentropy<I, J> {
    pub fn new() -> Self {
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
    pub fn forward(&mut self, inputs: MatrixNM<I, J>, y_true: MatrixNM<I, 1>) -> f64 {
        self.output = self.activation.forward(inputs);
        return self.loss.forward(self.output, y_true).clone();
    }

    pub fn backward(&mut self, dvalues: MatrixNM<I, J>, y_true: MatrixNM<I, 1>) -> MatrixNM<I, J> {
        let samples = dvalues.nrows();
        self.dinputs = dvalues.clone();
        for ii in 0..I {
            self.dinputs[(ii, y_true[(ii, 0)] as usize)] += -1.0;
        }
        self.dinputs = self.dinputs / samples as f64;
        return self.dinputs;
    }
}
