//! Token merging for streaming ASR.
//!
//! Implements the Longest Common Subsequence (LCS) merge algorithm for
//! aligning and deduplicating overlapping token sequences from streaming
//! inference chunks.

/// Result of LCS alignment containing merge indices.
#[derive(Debug, Clone, Copy)]
pub struct LcsAlignmentResult {
    /// Start index in the old buffer (relative to search window).
    pub i: usize,
    /// Start index in the new data.
    pub j: usize,
    /// Length of the longest common subsequence.
    pub length: usize,
}

/// Compute the Longest Common Subsequence alignment matrix.
///
/// # Arguments
/// * `x` - Previous token buffer (typically sliced to the overlap region)
/// * `y` - New token data
///
/// # Returns
/// * LCS alignment result with indices and length
/// * The LCS suffix table (for debugging)
pub fn longest_common_subsequence(
    x: &[usize],
    y: &[usize],
) -> (LcsAlignmentResult, Vec<Vec<usize>>) {
    let m = x.len();
    let n = y.len();

    // LCSuff[i][j] = length of longest common suffix ending at x[i-1], y[j-1]
    let mut lcs_suff = vec![vec![0usize; n + 1]; m + 1];

    let mut result = LcsAlignmentResult {
        i: 0,
        j: 0,
        length: 0,
    };

    // Build LCS suffix table
    for i in 1..=m {
        for j in 1..=n {
            if x[i - 1] == y[j - 1] {
                lcs_suff[i][j] = lcs_suff[i - 1][j - 1] + 1;

                // Track the longest subsequence (prefer later matches)
                if lcs_suff[i][j] >= result.length {
                    result.length = lcs_suff[i][j];
                    result.i = i;
                    result.j = j;
                }
            } else {
                lcs_suff[i][j] = 0;
            }
        }
    }

    // Check if we found a complete merge (alignment extends to end of old buffer)
    let is_complete = result.i == m;

    if is_complete {
        // Backtrack to find the start of the alignment
        let mut i = result.i;
        let mut j = result.j;
        let mut length = result.length;

        while length > 0 && i > 0 && j > 0 {
            if lcs_suff[i - 1][j - 1] > 0 {
                length -= 1;
                i -= 1;
                j -= 1;
            } else {
                // End of alignment
                i = i.saturating_sub(1);
                j = j.saturating_sub(1);
                break;
            }
        }

        result.i = i;
        result.j = j;
    } else {
        // Partial alignment: find leftmost LCS for robustness
        // This handles cases where the alignment breaks in the middle

        let mut max_len = 0;
        let mut max_j_idx = n;

        // Backward search for leftmost j with longest subsequence
        for i_idx in (0..=m).rev() {
            for j_idx in 0..=n {
                if lcs_suff[i_idx][j_idx] > max_len && j_idx <= max_j_idx {
                    max_len = lcs_suff[i_idx][j_idx];
                    max_j_idx = j_idx;
                    result.i = i_idx;
                    result.j = j_idx;
                }
            }
        }

        // Backtrack to find start of slice
        let mut i = result.i;
        let mut j = result.j;
        let mut slice_count = 0;

        while i > 0 && j > 0 {
            if lcs_suff[i][j] == 0 {
                j = j.saturating_sub(1);
            }
            if j > 0 {
                slice_count += 1;
                i -= 1;
                j -= 1;
            }
        }

        result.i = i;
        result.j = j;
        result.length = slice_count;
    }

    (result, lcs_suff)
}

/// Merge new tokens with previous buffer using LCS alignment.
///
/// This implements NeMo's `lcs_alignment_merge_buffer` function.
///
/// # Arguments
/// * `buffer` - Previous token buffer
/// * `data` - New tokens to merge
/// * `delay` - Number of delay timesteps (determines overlap search size)
/// * `max_steps_per_timestep` - Maximum tokens per acoustic frame (default 5)
/// * `min_lcs_length` - Minimum LCS length for deduplication (default 1)
///
/// # Returns
/// Merged token buffer
pub fn lcs_merge_buffer(
    buffer: &[usize],
    data: &[usize],
    delay: usize,
    max_steps_per_timestep: usize,
    min_lcs_length: usize,
) -> Vec<usize> {
    // If no delay or empty buffer, just concatenate
    if delay < 1 || buffer.is_empty() {
        let mut result = buffer.to_vec();
        result.extend_from_slice(data);
        return result;
    }

    // Only search within the overlap region
    let search_size = delay * max_steps_per_timestep;
    let buffer_start = buffer.len().saturating_sub(search_size);
    let buffer_slice = &buffer[buffer_start..];

    let (lcs_result, _) = longest_common_subsequence(buffer_slice, data);

    // If LCS is too short, don't deduplicate
    if lcs_result.length < min_lcs_length {
        let mut result = buffer.to_vec();
        result.extend_from_slice(data);
        return result;
    }

    // Slice off the duplicated portion from new data
    let slice_idx = lcs_result.j + lcs_result.length;
    let new_data = if slice_idx < data.len() {
        &data[slice_idx..]
    } else {
        &[]
    };

    let mut result = buffer.to_vec();
    result.extend_from_slice(new_data);
    result
}

/// Simple greedy merge for CTC output.
///
/// Collapses repeated tokens and removes blanks.
///
/// # Arguments
/// * `tokens` - Raw CTC token sequence
/// * `blank_id` - ID of the blank token
///
/// # Returns
/// Collapsed token sequence
pub fn greedy_ctc_merge(tokens: &[usize], blank_id: usize) -> Vec<usize> {
    let mut result = Vec::new();
    let mut previous = blank_id;

    for &token in tokens {
        if (token != previous || previous == blank_id) && token != blank_id {
            result.push(token);
        }
        previous = token;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcs_identical_sequences() {
        let x = vec![1, 2, 3, 4, 5];
        let y = vec![1, 2, 3, 4, 5];
        let (result, _) = longest_common_subsequence(&x, &y);

        assert_eq!(result.length, 5);
    }

    #[test]
    fn test_lcs_partial_overlap() {
        let x = vec![1, 2, 3, 4, 5];
        let y = vec![4, 5, 6, 7, 8];
        let (result, _) = longest_common_subsequence(&x, &y);

        assert!(result.length >= 2, "Should find overlap of [4, 5]");
    }

    #[test]
    fn test_lcs_no_overlap() {
        let x = vec![1, 2, 3];
        let y = vec![4, 5, 6];
        let (result, _) = longest_common_subsequence(&x, &y);

        assert_eq!(result.length, 0);
    }

    #[test]
    fn test_merge_buffer_no_delay() {
        let buffer = vec![1, 2, 3];
        let data = vec![4, 5, 6];

        let result = lcs_merge_buffer(&buffer, &data, 0, 5, 1);
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_merge_buffer_with_overlap() {
        let buffer = vec![1, 2, 3, 4, 5];
        let data = vec![4, 5, 6, 7, 8];

        let result = lcs_merge_buffer(&buffer, &data, 3, 5, 1);

        // Should merge [4, 5] and only add [6, 7, 8]
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_merge_buffer_empty_buffer() {
        let buffer: Vec<usize> = vec![];
        let data = vec![1, 2, 3];

        let result = lcs_merge_buffer(&buffer, &data, 2, 5, 1);
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_greedy_ctc_merge_basic() {
        let tokens = vec![1, 1, 0, 2, 2, 2, 0, 3];
        let blank_id = 0;

        let result = greedy_ctc_merge(&tokens, blank_id);
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_greedy_ctc_merge_all_blanks() {
        let tokens = vec![0, 0, 0, 0];
        let blank_id = 0;

        let result = greedy_ctc_merge(&tokens, blank_id);
        assert!(result.is_empty());
    }

    #[test]
    fn test_greedy_ctc_merge_repeated_with_blank() {
        // [1, 0, 1] should give [1, 1] (blank separates repeated tokens)
        let tokens = vec![1, 0, 1];
        let blank_id = 0;

        let result = greedy_ctc_merge(&tokens, blank_id);
        assert_eq!(result, vec![1, 1]);
    }
}
