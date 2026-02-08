use crate::activation::*;
use crate::layer::*;
use crate::types::*;
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use plotters::style::full_palette::PURPLE;
pub fn graph(
    pointx: MatrixNM<300, 2>,
    what: MatrixNM<300, 1>,
    testx: MatrixNM<300, 2>,
    what_t: MatrixNM<300, 1>,
    dense1: Vec<Vec<(Layer<300, 2, 64>, Layer<300, 64, 3>)>>,
    lossnacc: Vec<Vec<(f64, f64)>>,
    lossnacc_t: Vec<Vec<(f64, f64)>>,
) -> Result<(), Box<dyn std::error::Error>> {
    for ii in 0..=10 {
        let p = format!("image/{}.png", ii);

        let root = BitMapBackend::new(&p, (1000, 1920)).into_drawing_area();
        root.fill(&WHITE)?;
        let areas= root.split_evenly((4, 1));
        for (index, root) in areas.iter().enumerate() {
            let (left, right) = root.split_horizontally(600);
            let mut chart = ChartBuilder::on(&left)
                // .x_label_area_size(40)
                // .y_label_area_size(40)
                .build_cartesian_2d(-1f64..1f64, -1f64..1f64)?;
            let px = pointx.column(0);
            let py = pointx.column(1);
            let mut pxy = Vec::new();
            predict(
                &chart,
                &mut dense1[index][ii].0.clone(),
                &mut dense1[index][ii].1.clone(),
            );
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
            let px = testx.column(0);
            let py = testx.column(1);
            let mut pxy_t = Vec::new();
            for ii in 0..testx.nrows() {
                let mut color = RGBColor(0, 0, 0);
                if what_t[ii] as usize == 0 {
                    color = RGBColor(255, 0, 0);
                }
                if what_t[ii] as usize == 1 {
                    color = RGBColor(0, 255, 0);
                }
                if what_t[ii] as usize == 2 {
                    color = RGBColor(0, 0, 255);
                }
                pxy_t.push((px[ii], py[ii], color));
            }
            chart
                .configure_mesh()
                .disable_axes()
                .disable_mesh()
                .draw()?;
            chart.draw_series(
                pxy.iter()
                    .map(|(x, y, c)| Circle::new((*x, *y), 2, c.filled())),
            )?;
            chart.draw_series(
                pxy_t
                    .iter()
                    .map(|(x, y, c)| Circle::new((*x, *y), 2, c.filled())),
            )?;
            // let loss_max = lossnacc[index]
            //     .iter()
            //     .map(|x| x.0)
            //     .max_by(|a, b| a.total_cmp(b))
            //     .unwrap();

            let (up, down) = right.split_vertically(240);

            let mut loss_chart = ChartBuilder::on(&up)
                .x_label_area_size(10)
                .y_label_area_size(10)
                .build_cartesian_2d(0..10000usize, 0f64..1.0)?;
            let mut acc_chart = ChartBuilder::on(&down)
                .x_label_area_size(10)
                .y_label_area_size(10)
                .build_cartesian_2d(0..10000usize, 0f64..1f64)?;
            loss_chart.configure_mesh().draw()?;
            loss_chart.draw_series(LineSeries::new(
                lossnacc[index]
                    .iter()
                    .enumerate()
                    .map(|loss| (loss.0, loss.1.0)),
                &RED,
            ))?;
            loss_chart.draw_series(LineSeries::new(
                lossnacc_t[index]
                    .iter()
                    .enumerate()
                    .map(|loss| (loss.0, loss.1.0)),
                &PURPLE,
            ))?;

            acc_chart.configure_mesh().draw()?;
            acc_chart.draw_series(LineSeries::new(
                lossnacc[index]
                    .iter()
                    .enumerate()
                    .map(|loss| (loss.0, loss.1.1)),
                &RED,
            ))?;
            acc_chart.draw_series(LineSeries::new(
                lossnacc_t[index]
                    .iter()
                    .enumerate()
                    .map(|loss| (loss.0, loss.1.1)),
                &PURPLE,
            ))?;
        }
    }
    Ok(())
}

fn predict(
    chart: &ChartContext<'_, BitMapBackend<'_>, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    dense1: &mut Layer<300, 2, 64>,
    dense2: &mut Layer<300, 64, 3>,
) {
    let mut activation1 = ActivationReLu::<300, 64>::new();

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

            let result = result.row(0).transpose();
            let color = RGBAColor(
                (result[0] * 255f64) as u8,
                (result[1] * 255 as f64) as u8,
                (result[2] * 255.0) as u8,
                0.3,
            );
            //let mut color = [RED, GREEN, BLUE][result].to_rgba();
            //color.3 = (result.1) * 0.1;

            let circle = Circle::new((mt[(0, 0)], mt[(0, 1)]), 4, color.filled());
            chart.plotting_area().draw(&circle).expect("error");
        }
    }
}
