use crate::activation::*;
use crate::layer::*;
use crate::loss::*;
use crate::optimizer::*;
use crate::types::*;
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

pub fn randlearn(
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
