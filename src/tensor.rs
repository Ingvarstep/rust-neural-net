use std::ops::{Add, Mul, Sub, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    pub data: Vec<Vec<T>>,
    pub nrows: usize,
    pub ncols: usize,
}

impl <T: Clone + Copy + Mul<Output = T> + AddAssign> Tensor<T> {
    pub fn from(data: Vec<Vec<T>>) -> Self {
        let nrows = data.len();
        let ncols = data[0].len();
        Self { data, nrows, ncols }
    }

    pub fn new(nrows:usize, ncols:usize, initial:T) -> Self {
        let data = vec![vec![initial; ncols]; nrows];
        Tensor {
            data,
            nrows,
            ncols
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.nrows && col<self.ncols {
            Some(&self.data[row][col])
        }
        else {
            None
        }
    }

    // matmul it cant bultiply T by T
    pub fn matmul(&self, other: &Tensor<T>) -> Option<Tensor<T>> {
        if self.ncols!=other.nrows {
            None
        }
        else {
            let mut result = Tensor::new(self.nrows, self.ncols, self.data[0][0]);

            for r in 0..self.nrows {
                for c in 0..other.ncols {
                    for i in 0..self.ncols {
                        result.data[r][c] += self.data[r][i] * other.data[i][c];
                    }
                }
            }
            Some(result)
        }
    }
}


impl <T: Add<Output=T> + Copy + Clone + Mul<Output=T> + AddAssign> Add for Tensor<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let initial = *self.get(0, 0).unwrap();
        let mut result = Tensor::new(self.nrows, self.ncols, initial);
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        result
    }
}

impl <T: Sub<Output=T> + Copy + Clone + Mul<Output=T> + AddAssign> Sub for Tensor<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut result = Tensor::new(self.nrows, self.ncols, self.data[0][0]);
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                result.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        result
    }
}

impl <T: Sub<Output=T> + Copy + Clone + Mul<Output=T> + AddAssign> Mul for Tensor<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut result = Tensor::new(self.nrows, self.ncols, self.data[0][0]);
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                result.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }
        result
    }
}

impl <T: Sub<Output=T> + Copy + Clone + Mul<Output=T> + Div<Output=T>+ AddAssign> Div for Tensor<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let mut result = Tensor::new(self.nrows, self.ncols, self.data[0][0]);
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                result.data[i][j] = self.data[i][j] / other.data[i][j];
            }
        }
        result
    }
}

impl <T: Sub<Output=T> + Copy + Clone + Mul<Output=T> + AddAssign + Neg<Output=T>> Neg for Tensor<T> {
    type Output = Self;

    fn neg(self) -> Self {
        let mut result = Tensor::new(self.nrows, self.ncols, self.data[0][0]);
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                result.data[i][j] = -self.data[i][j];
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let t1:Tensor<f32> = Tensor::new(3, 3, 1.0);
        let t2:Tensor<f32> = Tensor::new(3, 3, 2.0);
        let t3 = t1 + t2;
        assert_eq!(t3.data[0][0], 3.0);
    }

    #[test]
    fn test_sub() {
        let t1:Tensor<f32> = Tensor::new(3, 3, 1.0);
        let t2:Tensor<f32> = Tensor::new(3, 3, 2.0);
        let t3 = t1 - t2;
        assert_eq!(t3.data[0][0], -1.0);
    }

    #[test]
    fn test_mul() {
        let t1:Tensor<f32> = Tensor::new(3, 3, 1.0);
        let t2:Tensor<f32> = Tensor::new(3, 3, 2.0);
        let t3 = t1 * t2;
        assert_eq!(t3.data[0][0], 2.0);
    }

    #[test]
    fn test_div() {
        let t1:Tensor<f32> = Tensor::new(3, 3, 1.0);
        let t2:Tensor<f32> = Tensor::new(3, 3, 2.0);
        let t3 = t1 / t2;
        assert_eq!(t3.data[0][0], 0.5);
    }

    #[test]
    fn test_neg() {
        let t1:Tensor<f32> = Tensor::new(3, 3, 1.0);
        let t2 = -t1;
        assert_eq!(t2.data[0][0], -1.0);
    }

    #[test]
    fn test_get() {
        let t1:Tensor<f32> = Tensor::new(3, 3, 1.0);
        let n1 = t1.get(0, 0);
        assert_eq!(n1, Some(&1.0));
    }

    #[test]
    fn test_from() {
        let v = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let t1 = Tensor::from(v);
        assert_eq!(t1.data[0][0], 1.0);
    }
}
