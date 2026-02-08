mod activation;
mod data;
mod graph;
mod layer;
mod loss;
mod model;
mod optimizer;
mod types;
use crate::data::*;
use crate::graph::*;
use crate::layer::*;
use crate::model::*;
use crate::optimizer::*;
// use nalgebra::zero;

fn main() {
    let (x, y) = spiral_data::<100, 3, 300>();
    let (x_test, y_test) = spiral_data::<100, 3, 300>();
    let dense1 = Layer::<300, 2, 64>::new(0.0, 5e-4, 0.0, 5e-4);
    //let activation1 = ActivationReLu::<300, 3>::new();
    let dense2 = Layer::<300, 64, 3>::new(0.0, 0.0, 0.0, 0.0);

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
