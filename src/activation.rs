use crate::types::*;
pub struct ActivationReLu<const N: usize, const M: usize> {
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

pub struct ActivationSoftmax<const I: usize, const J: usize> {
    inputs: MatrixNM<I, J>,
    outputs: MatrixNM<I, J>,
    //dinputs: MatrixNM<I, J>,
}
impl<const I: usize, const J: usize> ActivationSoftmax<I, J> {
    pub fn new() -> Self {
        let input = MatrixNM::<I, J>::zeros();
        let output = MatrixNM::<I, J>::zeros();
        //let dinput = MatrixNM::<I, J>::zeros();

        Self {
            inputs: input,
            outputs: output,
            //dinputs: dinput,
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
}
