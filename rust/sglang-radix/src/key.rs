//! Query-side radix key: a view over raw token ids plus the namespace
//! (`extra_key` / `cache_salt`) and the bigram flag.
//!
//! This mirrors `RadixKey` from
//! `python/sglang/srt/mem_cache/radix_cache.py`. The tree itself stores
//! keys *flattened* into logical units (one element per token, or two
//! elements per bigram), so all comparisons happen on contiguous `i64`
//! slices — no per-token boxing, no slice-chaining views.

use std::borrow::Cow;

/// A query key for one tree operation.
///
/// `tokens` holds raw token ids. In bigram mode the key represents
/// `raw_len - 1` logical units (pairs `(t[i], t[i+1])`); in plain mode the
/// number of logical units equals the raw length. `limit` caps the raw
/// token count without an O(n) copy, exactly like the Python `limit`.
#[derive(Debug, Clone, Copy)]
pub struct RadixKey<'a> {
    /// Raw token ids (in bigram mode: N+1 raw tokens for N logical bigrams).
    pub tokens: &'a [i64],
    /// Caller-defined namespace; different `extra_key`s never share nodes.
    pub extra_key: Option<&'a str>,
    /// Distinct namespace salt; namespaced so it cannot collide with
    /// `extra_key`.
    pub cache_salt: Option<&'a str>,
    /// Bigram view flag. The tree's `is_eagle` may force this on
    /// (equivalent to Python `maybe_to_bigram_view`).
    pub is_bigram: bool,
    /// Optional cap on raw tokens (behaves as if sliced to `tokens[:limit]`).
    pub limit: Option<usize>,
}

impl<'a> RadixKey<'a> {
    pub fn new(tokens: &'a [i64]) -> Self {
        Self {
            tokens,
            extra_key: None,
            cache_salt: None,
            is_bigram: false,
            limit: None,
        }
    }

    /// Raw token count honoring `limit`.
    pub fn raw_len(&self) -> usize {
        let n = self.tokens.len();
        match self.limit {
            Some(l) if l < n => l,
            _ => n,
        }
    }

    /// Logical unit count (bigrams: `max(0, raw - 1)`).
    pub fn logical_len(&self) -> usize {
        let n = self.raw_len();
        if self.is_bigram {
            n.saturating_sub(1)
        } else {
            n
        }
    }

    /// Flattened logical units, truncated to a multiple of `page_size`.
    ///
    /// Zero-copy (`Cow::Borrowed`) for plain keys without a cap or
    /// truncation; bigram keys materialize the `(t[i], t[i+1])` pairs.
    /// This is the representation the tree stores and compares.
    pub fn flatten_page_aligned(&self, page_size: usize) -> Cow<'a, [i64]> {
        let raw = self.raw_len();
        let logical = self.logical_len();
        let aligned = logical / page_size * page_size;
        if aligned == 0 {
            return Cow::Borrowed(&[]);
        }
        if !self.is_bigram {
            let end = aligned.min(raw);
            Cow::Borrowed(&self.tokens[..end])
        } else {
            // Bigrams [0, aligned) span raw tokens [0, aligned + 1).
            let src = &self.tokens[..(aligned + 1).min(raw)];
            let mut v = Vec::with_capacity(2 * aligned.min(src.len().saturating_sub(1)));
            for j in 0..aligned.min(src.len().saturating_sub(1)) {
                v.push(src[j]);
                v.push(src[j + 1]);
            }
            Cow::Owned(v)
        }
    }
}

/// Number of shared leading logical units between two flattened unit
/// sequences, using exponential (gallop) window compares followed by a
/// binary refinement — the same strategy as Python `RadixKey.match`,
/// without per-token work on long shared prefixes.
pub fn common_prefix_len(a: &[i64], b: &[i64]) -> usize {
    let n = a.len().min(b.len());
    let mut lo = 0;
    let mut step = 1;
    while lo < n {
        let mut hi = (lo + step).min(n);
        if a[lo..hi] != b[lo..hi] {
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if a[lo..mid] == b[lo..mid] {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return lo;
        }
        lo = hi;
        step *= 2;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_len_and_align() {
        let t: Vec<i64> = (0..7).collect();
        let k = RadixKey::new(&t);
        assert_eq!(k.raw_len(), 7);
        assert_eq!(k.logical_len(), 7);
        let f = k.flatten_page_aligned(4);
        assert_eq!(f.as_ref(), [0, 1, 2, 3]);
        let f1 = k.flatten_page_aligned(1);
        assert!(matches!(f1, Cow::Borrowed(_)));
        assert_eq!(f1.as_ref(), [0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn limit_caps_raw() {
        let t: Vec<i64> = (0..10).collect();
        let k = RadixKey {
            tokens: &t,
            limit: Some(3),
            ..RadixKey::new(&t)
        };
        assert_eq!(k.raw_len(), 3);
        assert_eq!(k.logical_len(), 3);
        assert_eq!(k.flatten_page_aligned(1).as_ref(), [0, 1, 2]);
    }

    #[test]
    fn bigram_len_and_flatten() {
        // raw [1,2,3,4,5] -> 4 bigrams: (1,2),(2,3),(3,4),(4,5)
        let t = vec![1, 2, 3, 4, 5];
        let k = RadixKey {
            tokens: &t,
            is_bigram: true,
            ..RadixKey::new(&t)
        };
        assert_eq!(k.raw_len(), 5);
        assert_eq!(k.logical_len(), 4);
        let f = k.flatten_page_aligned(1);
        assert_eq!(f.as_ref(), [1, 2, 2, 3, 3, 4, 4, 5]);
        // page_size 2 -> 4 aligned units (already aligned)
        let f2 = k.flatten_page_aligned(2);
        assert_eq!(f2.len(), 8);
        // raw [1,2] -> 1 bigram; page_size 2 -> aligned 0
        let t2 = vec![1, 2];
        let k2 = RadixKey {
            tokens: &t2,
            is_bigram: true,
            ..RadixKey::new(&t2)
        };
        assert_eq!(k2.logical_len(), 1);
        assert_eq!(k2.flatten_page_aligned(2).len(), 0);
    }

    #[test]
    fn common_prefix() {
        let a = [1, 2, 3, 4, 5, 6];
        let b = [1, 2, 3, 9, 5, 6];
        assert_eq!(common_prefix_len(&a, &b), 3);
        assert_eq!(common_prefix_len(&a, &[1, 2]), 2);
        assert_eq!(common_prefix_len(&[], &a), 0);
        // long shared prefix, divergence near the end (gallop path)
        let long: Vec<i64> = (0..10_000).collect();
        let mut other = long.clone();
        other[9_999] = -1;
        assert_eq!(common_prefix_len(&long, &other), 9_999);
        assert_eq!(common_prefix_len(&long, &long), 10_000);
    }
}
