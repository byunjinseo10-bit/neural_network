use nalgebra::{
    ArrayStorage, Const, Matrix,
    /* , Matrix3, Matrix3x4, OMatrix,*/ Vector, /*  matrix, vector*/
};

pub type MatrixNM<const N: usize, const M: usize> =
    Matrix<f64, Const<N>, Const<M>, ArrayStorage<f64, N, M>>;
pub type VectorN<const N: usize> = Vector<f64, Const<N>, ArrayStorage<f64, N, 1>>;
