use crate::types::*;
use rand::Rng;
use rand_distr::Distribution;
pub fn vertical_data<const SAMPLES: usize, const CLASSES: usize, const TOTAL: usize>()
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

pub fn spiral_data<const SAMPLES: usize, const CLASSES: usize, const TOTAL: usize>()
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
