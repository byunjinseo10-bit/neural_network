use crate::types::MatrixNM;
use crate::types::VectorN;
use rand_distr::Distribution;
#[derive(Clone)]
pub struct Layer<const M: usize, const I: usize, const N: usize> {
    inputs: Box<MatrixNM<M, I>>,
    pub weights: Box<MatrixNM<I, N>>,
    pub biases: Box<VectorN<N>>,
    pub dinputs: Box<MatrixNM<M, I>>,
    pub dweights: Box<MatrixNM<I, N>>,
    pub dbiases: Box<VectorN<N>>,

    pub weights_momentum: Box<MatrixNM<I, N>>,
    pub biases_momentum: Box<VectorN<N>>,
    pub weights_cache: Box<MatrixNM<I, N>>,
    pub biases_cache: Box<VectorN<N>>,

    pub weights_l1: f64,
    pub weights_l2: f64,
    pub biases_l1: f64,
    pub biases_l2: f64,
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

pub struct DropoutLayer<const I: usize, const J: usize> {
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

    pub fn forward(&mut self, input: MatrixNM<I, J>) -> MatrixNM<I, J> {
        for ii in 0..I {
            for jj in 0..J {
                if !rand::random_bool(self.p) {
                    self.binary_mask[(ii, jj)] = 1.0 / (1.0 - self.p);
                }
            }
        }
        return input.zip_map(&self.binary_mask, |input, mask| input * mask);
    }
    pub fn backward(&self, dvalue: MatrixNM<I, J>) -> MatrixNM<I, J> {
        return dvalue.zip_map(&self.binary_mask, |dvalue, mask| dvalue * mask);
    }
}
