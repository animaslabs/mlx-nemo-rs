//! Helpers for reading MLX array data on the host.

use mlx_rs::{Array, ArrayElement};

use crate::core::error::{Error, Result};

/// Host-side access to array data.
pub trait ArrayExt {
    /// Copy the array's elements into a `Vec` in row-major order.
    ///
    /// Unlike [`Array::as_slice`], this is correct for non-contiguous views
    /// (slices, transposes, broadcasts): it materializes a row-major array
    /// first, which shares the existing buffer when it is already suitable.
    fn to_vec_row_major<T: ArrayElement + Clone>(&self) -> Result<Vec<T>>;
}

impl ArrayExt for Array {
    fn to_vec_row_major<T: ArrayElement + Clone>(&self) -> Result<Vec<T>> {
        self.contiguous()?
            .to_vec_exact::<T>()
            .map_err(|e| Error::model(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::indexing::IndexOp;

    #[test]
    fn test_to_vec_row_major_strided_view() {
        // A column slice of a row-major matrix is a non-contiguous view.
        let a = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], &[3, 2]);
        let col = a.index((.., 1));
        assert_eq!(col.to_vec_row_major::<f32>().unwrap(), vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_to_vec_row_major_transpose() {
        let a = Array::from_slice(&[0i32, 1, 2, 3, 4, 5], &[2, 3]);
        let t = a.t();
        assert_eq!(t.to_vec_row_major::<i32>().unwrap(), vec![0, 3, 1, 4, 2, 5]);
    }
}
