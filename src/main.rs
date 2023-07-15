pub mod tensor;

use tensor::Tensor;
fn main() {
    let t1:Tensor<f32> = Tensor::new(3, 3, 1.0);
    let t2:Tensor<f32> = Tensor::new(3, 3, 2.0);
    let t3 = t1 + t2;
    println!("{:?}", t3);

    let v = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let t4 = Tensor::from(v);
    println!("{:?}", t4);

    let n1 = t4.get(10, 0);
    println!("{:?}", n1);

    // let t5 = t1.matmul(&t2);

    // println!("{:?}", t5);
    println!("Hello, world!");
}
