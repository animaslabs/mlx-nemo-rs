//! Utility functions and constants.

/// Mathematical constants
pub mod constants {
    /// Pi (π) - ratio of a circle's circumference to its diameter
    pub const PI: f64 = 3.14159265358979323846264338327950288_f64;

    /// Euler's number (e) - base of natural logarithms
    pub const E: f64 = 2.71828182845904523536028747135266250_f64;

    /// Golden ratio (φ)
    pub const GOLDEN_RATIO: f64 = 1.61803398874989484820458683436563812_f64;

    /// Square root of 2
    pub const SQRT_2: f64 = 1.41421356237309504880168872420969808_f64;

    /// Square root of 3
    pub const SQRT_3: f64 = 1.73205080756887729352744634150587236_f64;
}

/// String manipulation utilities
pub mod strings {
    /// Reverses a string.
    ///
    /// # Examples
    ///
    /// ```
    /// use mlx_nemo_rs::utils::strings::reverse;
    ///
    /// assert_eq!(reverse("hello"), "olleh");
    /// assert_eq!(reverse(""), "");
    /// ```
    pub fn reverse(s: &str) -> String {
        s.chars().rev().collect()
    }

    /// Checks if a string is a palindrome.
    ///
    /// # Examples
    ///
    /// ```
    /// use mlx_nemo_rs::utils::strings::is_palindrome;
    ///
    /// assert!(is_palindrome("racecar"));
    /// assert!(is_palindrome("A man a plan a canal Panama"));
    /// assert!(!is_palindrome("hello"));
    /// ```
    pub fn is_palindrome(s: &str) -> bool {
        let cleaned: String = s
            .chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();

        cleaned == cleaned.chars().rev().collect::<String>()
    }

    /// Counts the number of words in a string.
    ///
    /// # Examples
    ///
    /// ```
    /// use mlx_nemo_rs::utils::strings::word_count;
    ///
    /// assert_eq!(word_count("hello world"), 2);
    /// assert_eq!(word_count("  hello   world  "), 2);
    /// assert_eq!(word_count(""), 0);
    /// ```
    pub fn word_count(s: &str) -> usize {
        s.split_whitespace().count()
    }
}

/// Collection utilities
pub mod collections {
    /// Finds the maximum value in a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use mlx_nemo_rs::utils::collections::max_in_slice;
    ///
    /// let numbers = [1, 5, 3, 9, 2];
    /// assert_eq!(max_in_slice(&numbers), Some(&9));
    ///
    /// let empty: [i32; 0] = [];
    /// assert_eq!(max_in_slice(&empty), None);
    /// ```
    pub fn max_in_slice<T: Ord>(slice: &[T]) -> Option<&T> {
        if slice.is_empty() {
            None
        } else {
            slice.iter().max()
        }
    }

    /// Checks if all elements in a slice are unique.
    ///
    /// # Examples
    ///
    /// ```
    /// use mlx_nemo_rs::utils::collections::all_unique;
    ///
    /// let unique = [1, 2, 3, 4];
    /// assert!(all_unique(&unique));
    ///
    /// let not_unique = [1, 2, 2, 3];
    /// assert!(!all_unique(&not_unique));
    /// ```
    pub fn all_unique<T: Eq + std::hash::Hash>(slice: &[T]) -> bool {
        let mut seen = std::collections::HashSet::new();
        for item in slice {
            if !seen.insert(item) {
                return false;
            }
        }
        true
    }

    /// Removes duplicates from a vector while preserving order.
    ///
    /// # Examples
    ///
    /// ```
    /// use mlx_nemo_rs::utils::collections::remove_duplicates;
    ///
    /// let mut vec = vec![1, 2, 2, 3, 1, 4];
    /// remove_duplicates(&mut vec);
    /// assert_eq!(vec, vec![1, 2, 3, 4]);
    /// ```
    pub fn remove_duplicates<T: Eq + std::hash::Hash>(vec: &mut Vec<T>) {
        let mut seen = std::collections::HashSet::new();
        vec.retain(|item| seen.insert(item.clone()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_reverse() {
        assert_eq!(strings::reverse("hello"), "olleh");
        assert_eq!(strings::reverse("racecar"), "racecar");
        assert_eq!(strings::reverse(""), "");
    }

    #[test]
    fn test_is_palindrome() {
        assert!(strings::is_palindrome("racecar"));
        assert!(strings::is_palindrome("A man a plan a canal Panama"));
        assert!(strings::is_palindrome("Madam"));
        assert!(!strings::is_palindrome("hello"));
        assert!(!strings::is_palindrome("world"));
    }

    #[test]
    fn test_word_count() {
        assert_eq!(strings::word_count("hello world"), 2);
        assert_eq!(strings::word_count("  hello   world  "), 2);
        assert_eq!(strings::word_count(""), 0);
        assert_eq!(strings::word_count("one two three four"), 4);
    }

    #[test]
    fn test_max_in_slice() {
        let numbers = [1, 5, 3, 9, 2];
        assert_eq!(collections::max_in_slice(&numbers), Some(&9));

        let single = [42];
        assert_eq!(collections::max_in_slice(&single), Some(&42));

        let empty: [i32; 0] = [];
        assert_eq!(collections::max_in_slice(&empty), None);
    }

    #[test]
    fn test_all_unique() {
        let unique = [1, 2, 3, 4];
        assert!(collections::all_unique(&unique));

        let not_unique = [1, 2, 2, 3];
        assert!(!collections::all_unique(&not_unique));

        let empty: [i32; 0] = [];
        assert!(collections::all_unique(&empty));
    }

    #[test]
    fn test_remove_duplicates() {
        let mut vec = vec![1, 2, 2, 3, 1, 4];
        collections::remove_duplicates(&mut vec);
        assert_eq!(vec, vec![1, 2, 3, 4]);

        let mut vec2 = vec!["a", "b", "a", "c"];
        collections::remove_duplicates(&mut vec2);
        assert_eq!(vec2, vec!["a", "b", "c"]);
    }
}
