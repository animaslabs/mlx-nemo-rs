//! Mathematical operations and utilities.

use thiserror::Error;

/// Errors that can occur during mathematical operations.
#[derive(Error, Debug, PartialEq)]
pub enum MathError {
    #[error("Division by zero")]
    DivisionByZero,
    #[error("Overflow error")]
    Overflow,
    #[error("Underflow error")]
    Underflow,
}

/// Adds two numbers together.
///
/// # Examples
///
/// ```
/// use mlx_nemo_rs::math::add;
///
/// let result = add(2, 3);
/// assert_eq!(result, 5);
/// ```
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Subtracts two numbers.
///
/// # Examples
///
/// ```
/// use mlx_nemo_rs::math::subtract;
///
/// let result = subtract(5, 3);
/// assert_eq!(result, 2);
/// ```
pub fn subtract(a: i32, b: i32) -> i32 {
    a - b
}

/// Multiplies two numbers.
///
/// # Examples
///
/// ```
/// use mlx_nemo_rs::math::multiply;
///
/// let result = multiply(4, 3);
/// assert_eq!(result, 12);
/// ```
pub fn multiply(a: i32, b: i32) -> i32 {
    a * b
}

/// Divides two numbers with error handling for division by zero.
///
/// # Examples
///
/// ```
/// use mlx_nemo_rs::math::{divide, MathError};
///
/// let result = divide(12, 3)?;
/// assert_eq!(result, 4);
///
/// let error = divide(12, 0);
/// assert_eq!(error, Err(MathError::DivisionByZero));
/// # Ok::<(), MathError>(())
/// ```
pub fn divide(a: i32, b: i32) -> Result<i32, MathError> {
    if b == 0 {
        Err(MathError::DivisionByZero)
    } else {
        Ok(a / b)
    }
}

/// Calculates the factorial of a number.
///
/// # Examples
///
/// ```
/// use mlx_nemo_rs::math::factorial;
///
/// let result = factorial(5);
/// assert_eq!(result, 120);
/// ```
pub fn factorial(n: u32) -> u64 {
    match n {
        0 | 1 => 1,
        _ => n as u64 * factorial(n - 1),
    }
}

/// Checks if a number is prime.
///
/// # Examples
///
/// ```
/// use mlx_nemo_rs::math::is_prime;
///
/// assert!(is_prime(7));
/// assert!(!is_prime(6));
/// ```
pub fn is_prime(n: u32) -> bool {
    if n <= 1 {
        return false;
    }
    if n <= 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }

    let mut i = 5;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-1, 1), 0);
    }

    #[test]
    fn test_subtract() {
        assert_eq!(subtract(5, 3), 2);
        assert_eq!(subtract(3, 5), -2);
    }

    #[test]
    fn test_multiply() {
        assert_eq!(multiply(4, 3), 12);
        assert_eq!(multiply(-2, 3), -6);
    }

    #[test]
    fn test_divide() {
        assert_eq!(divide(12, 3), Ok(4));
        assert_eq!(divide(-6, 2), Ok(-3));
        assert_eq!(divide(12, 0), Err(MathError::DivisionByZero));
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
        assert_eq!(factorial(10), 3628800);
    }

    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(!is_prime(6));
        assert!(is_prime(7));
        assert!(!is_prime(8));
        assert!(!is_prime(9));
        assert!(is_prime(11));
        assert!(is_prime(13));
        assert!(is_prime(17));
        assert!(is_prime(19));
        assert!(is_prime(23));
    }
}
