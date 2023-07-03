use libm::exp;
use rand::Rng;

const TRAIN: [[f64; 3]; 4] = [[0.0; 3], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0; 3]];
const TRAIN_SIZE: usize = TRAIN.len();

fn sigmoid(x: f64) -> f64 {
    return 1.0 / (1.0 + exp(-x));
}

fn cost(w1: f64, w2: f64, b: f64) -> f64 {
    let mut result: f64 = 0.0;
    for i in 0..TRAIN_SIZE {
        let x1: f64 = TRAIN[i][0];
        let x2: f64 = TRAIN[i][1];
        let y: f64 = sigmoid((x1 * w1) + (x2 * w2) + b);
        let d: f64 = y - TRAIN[i][2];
        result += d * d;
    }
    result /= TRAIN_SIZE as f64;
    result
}

const EPS: f64 = 1e-3;
const RATE: f64 = 1e-1;

fn main() {
    let mut rng = rand::thread_rng();
    let mut w1: f64 = rng.gen::<f64>();
    let mut w2: f64 = rng.gen::<f64>();
    let mut b: f64 = rng.gen::<f64>();
    for _ in 0..1000 * 10001 {
        let c: f64 = cost(w1, w2, b);
        // println!("w1 = {w1:.5}, w2 = {w2:.5}, b = {b:.5} cost = {c}");
        let dw1: f64 = (cost(w1 + EPS, w2, b) - c) / EPS;
        let dw2: f64 = (cost(w1, w2 + EPS, b) - c) / EPS;
        let db: f64 = (cost(w1, w2, b + EPS) - c) / EPS;
        w1 -= RATE * dw1;
        w2 -= RATE * dw2;
        b -= RATE * db;
    }
    for i in 0..2 {
        for j in 0..2 {
            println!(
                "{} | {} = {}",
                i,
                j,
                sigmoid(i as f64 * w1 + j as f64 * w2 + b)
            );
        }
    }
}
