//! Exact rational number type
//!
//! A simple rational number implementation using BigInt for numerator and denominator.

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{Zero, One, Signed};
use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg};

/// Exact rational number (numerator / denominator)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rational {
    pub numerator: BigInt,
    pub denominator: BigInt,
}

impl Rational {
    /// Create a new rational number from numerator and denominator
    pub fn new(num: BigInt, den: BigInt) -> Self {
        let mut r = Self {
            numerator: num,
            denominator: den,
        };
        r.reduce();
        r
    }

    /// Create a rational from an integer
    pub fn from_int<T: Into<BigInt>>(n: T) -> Self {
        Self {
            numerator: n.into(),
            denominator: BigInt::one(),
        }
    }

    /// Create a rational from a BigInt (alias for from_int)
    pub fn from_bigint(n: BigInt) -> Self {
        Self::from_int(n)
    }

    /// Create zero
    pub fn zero() -> Self {
        Self {
            numerator: BigInt::zero(),
            denominator: BigInt::one(),
        }
    }

    /// Create one
    pub fn one() -> Self {
        Self {
            numerator: BigInt::one(),
            denominator: BigInt::one(),
        }
    }

    /// Check if this rational is zero
    pub fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    /// Reduce to lowest terms
    fn reduce(&mut self) {
        if self.numerator.is_zero() {
            self.denominator = BigInt::one();
            return;
        }

        let g = self.numerator.gcd(&self.denominator);
        self.numerator = &self.numerator / &g;
        self.denominator = &self.denominator / &g;

        // Ensure denominator is positive
        if self.denominator.is_negative() {
            self.numerator = -&self.numerator;
            self.denominator = -&self.denominator;
        }
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == BigInt::one() {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

impl From<i64> for Rational {
    fn from(n: i64) -> Self {
        Self::from_int(n)
    }
}

impl From<BigInt> for Rational {
    fn from(n: BigInt) -> Self {
        Self::from_int(n)
    }
}

impl Add for Rational {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let num = &self.numerator * &other.denominator + &other.numerator * &self.denominator;
        let den = &self.denominator * &other.denominator;
        Self::new(num, den)
    }
}

impl Add for &Rational {
    type Output = Rational;

    fn add(self, other: Self) -> Rational {
        let num = &self.numerator * &other.denominator + &other.numerator * &self.denominator;
        let den = &self.denominator * &other.denominator;
        Rational::new(num, den)
    }
}

impl Sub for Rational {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let num = &self.numerator * &other.denominator - &other.numerator * &self.denominator;
        let den = &self.denominator * &other.denominator;
        Self::new(num, den)
    }
}

impl Mul for Rational {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let num = &self.numerator * &other.numerator;
        let den = &self.denominator * &other.denominator;
        Self::new(num, den)
    }
}

impl Mul for &Rational {
    type Output = Rational;

    fn mul(self, other: Self) -> Rational {
        let num = &self.numerator * &other.numerator;
        let den = &self.denominator * &other.denominator;
        Rational::new(num, den)
    }
}

impl Div for Rational {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let num = &self.numerator * &other.denominator;
        let den = &self.denominator * &other.numerator;
        Self::new(num, den)
    }
}

impl Neg for Rational {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            numerator: -self.numerator,
            denominator: self.denominator,
        }
    }
}

impl Neg for &Rational {
    type Output = Rational;

    fn neg(self) -> Rational {
        Rational {
            numerator: -&self.numerator,
            denominator: self.denominator.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational_arithmetic() {
        let a = Rational::new(BigInt::from(1), BigInt::from(2));
        let b = Rational::new(BigInt::from(1), BigInt::from(3));

        let sum = a.clone() + b.clone();
        assert_eq!(sum, Rational::new(BigInt::from(5), BigInt::from(6)));

        let prod = a.clone() * b.clone();
        assert_eq!(prod, Rational::new(BigInt::from(1), BigInt::from(6)));
    }

    #[test]
    fn test_rational_reduction() {
        let r = Rational::new(BigInt::from(4), BigInt::from(8));
        assert_eq!(r, Rational::new(BigInt::from(1), BigInt::from(2)));
    }
}
