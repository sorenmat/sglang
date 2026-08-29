//! String-based stop decisions, next to the Rust detokenizer that owns the
//! decoded text.
//!
//! These are the Rust twins of `Req._check_str_based_finish` /
//! `_locate_str_stop_finished_len` / `check_match_stop_str_prefix`
//! (schedule_batch.py), which run per decode step on the Python scheduler
//! thread with the HF tokenizer. The detokenizer shard decodes the very same
//! tail, so the decisions move here as pure functions over the decoded text
//! and the token window; the `decode_prefix` callback is where the shard's
//! tokenizer plugs in. A differential test drives both sides over the same
//! fake token map (see `test/registered/rust/test_rust_str_stop_parity.py`).
//!
//! The regex branch assumes ingress-validated patterns: `stop_regex` is
//! admitted only through `utils::regex::validate`'s portable subset, so a
//! compile failure here is a real fault (mirrored as
//! [`StrStopDecision::InvalidRegex`], the seatbelt Python's `re.error`
//! handler plays — fail this request, not the loop).

use regex::Regex;

/// A string-stop decision for one decode step — `None` means "no string stop
/// matched this step" and the caller falls through to the token-based checks,
/// exactly like Python's `_check_str_based_finish` returning `False`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StrStopDecision {
    /// A stop string matched. `matched` is the stop that ended the request
    /// (Python's `FINISH_MATCHED_STR(matched=stop_str)`). `finished_len` is
    /// `Some` only when the match was found in THIS step's tail (the length to
    /// trim at); a match that lives only in the previously-decoded text leaves
    /// it `None` (Python sets no `finished_len` in that case).
    MatchedStr {
        matched: String,
        finished_len: Option<usize>,
    },
    /// A stop regex matched (Python's `FINISHED_MATCHED_REGEX`); the match is
    /// always in the tail, so `finished_len` is always `Some`.
    MatchedRegex {
        matched: String,
        finished_len: usize,
    },
    /// A stop regex that cannot be compiled: Python's `re.error` seatbelt
    /// aborts THIS request with a 400 instead of escaping into the loop.
    InvalidRegex { pattern: String, message: String },
}

/// The request-side state the check reads: the decoded tail window, the text
/// decoded in earlier steps, the token window the tail was decoded from, and
/// the sampling params' stop lists.
pub struct StrStopState<'a> {
    /// `tokenizer.decode(output_ids[-tail_len:])` for this step's window.
    pub tail_text: &'a str,
    /// The text decoded in all earlier steps (a stop matched only here still
    /// finishes the request, without a trim position).
    pub previously_decoded: &'a str,
    /// `output_ids[start..]` — the token window `tail_text` covers.
    pub token_window: &'a [i32],
    /// `len(output_ids) - tail_len`.
    pub window_start: usize,
    /// Tokens accepted this step (speculative decoding may accept several).
    pub new_accepted_len: usize,
    /// `len(output_ids)`.
    pub output_len: usize,
    pub stop_strs: &'a [String],
    pub stop_regex_strs: &'a [String],
    /// Decode `window[..count]` for the stop-location scan (the shard's
    /// tokenizer). Only called with counts the scan actually needs.
    pub decode_prefix: &'a dyn Fn(&[i32]) -> String,
}

/// `Req._stop_match_tail_len`: how many trailing tokens the stop match may
/// reach. Covers `max(stop_str_max_len, stop_regex_max_len) + 1` chars plus
/// the extra newly-accepted tokens, clamped to the output length.
pub fn stop_match_tail_len(
    stop_str_max_len: usize,
    stop_regex_max_len: usize,
    new_accepted_len: usize,
    output_len: usize,
) -> usize {
    let max_len_tail_str = stop_str_max_len
        .saturating_add(1)
        .max(stop_regex_max_len.saturating_add(1));
    // `new_accepted_len - 1` floors at 0 (Python: `max(new_accepted_len - 1, 0)`).
    (max_len_tail_str + new_accepted_len.saturating_sub(1)).min(output_len)
}

/// `Req.check_match_stop_str_prefix`: true when this step's tail already
/// contains a stop string, or ENDS with the prefix of one (the stream-interval
/// gate holds back emission while a stop is one token away from completing).
pub fn check_match_stop_str_prefix(tail_text: &str, stop_strs: &[String]) -> bool {
    if stop_strs.is_empty() || tail_text.is_empty() {
        return false;
    }
    for stop_str in stop_strs {
        if stop_str.is_empty() {
            continue;
        }
        if tail_text.contains(stop_str.as_str()) {
            return true;
        }
        // Tail suffix vs stop prefix: the longest common tail/head decides.
        let min_len = tail_text.len().min(stop_str.len());
        let mut i = 1usize;
        while i <= min_len {
            if tail_text[tail_text.len() - i..] == stop_str[..i] {
                return true;
            }
            i += 1;
        }
    }
    false
}

/// `Req._locate_str_stop_finished_len`: the first token-count of the window
/// whose decoded text matches, counted from the window's start. Older prefixes
/// were already checked in the previous step, so the scan starts at
/// `max(1, len - new_accepted_len + 1)`; when the scan is empty the whole
/// window is already known to match by the caller.
pub fn locate_str_stop_finished_len(
    window: &[i32],
    window_start: usize,
    new_accepted_len: usize,
    output_len: usize,
    decode_prefix: &dyn Fn(&[i32]) -> String,
    is_stop: &dyn Fn(&str) -> bool,
) -> usize {
    let n = window.len();
    // Python: `max(1, len(token_window) - new_accepted_len + 1)` over Python
    // ints, which floor below zero — the i64 mirror keeps that floor.
    let a = 1_i64.max(n as i64 - new_accepted_len as i64 + 1) as usize;
    for token_count in a..n {
        if is_stop(&decode_prefix(&window[..token_count])) {
            return window_start + token_count;
        }
    }
    // The full tail window is already known to match by the caller.
    output_len
}

/// `Req._check_str_based_finish`: stop strings first (in param order), then
/// stop regexes. `None` = nothing string-based matched this step.
pub fn check_str_based_finish(s: &StrStopState) -> Option<StrStopDecision> {
    if s.stop_strs.is_empty() && s.stop_regex_strs.is_empty() {
        return None;
    }
    if !s.stop_strs.is_empty() {
        for stop_str in s.stop_strs {
            let in_tail = s.tail_text.contains(stop_str.as_str());
            // Python: `stop_str_in_tail or stop_str in self.decoded_text`.
            if in_tail || s.previously_decoded.contains(stop_str.as_str()) {
                let finished_len = in_tail.then(|| {
                    locate_str_stop_finished_len(
                        s.token_window,
                        s.window_start,
                        s.new_accepted_len,
                        s.output_len,
                        s.decode_prefix,
                        &|text: &str| text.contains(stop_str.as_str()),
                    )
                });
                return Some(StrStopDecision::MatchedStr {
                    matched: stop_str.clone(),
                    finished_len,
                });
            }
        }
    }
    if !s.stop_regex_strs.is_empty() {
        for stop_regex in s.stop_regex_strs {
            // Seatbelt, not validation: patterns are checked at ingress
            // (`utils::regex::validate`), and this runs per decode step — a
            // compile failure fails the request instead of escaping the loop.
            let Ok(re) = Regex::new(stop_regex) else {
                return Some(StrStopDecision::InvalidRegex {
                    pattern: stop_regex.clone(),
                    message: format!("invalid stop_regex {stop_regex:?}"),
                });
            };
            if re.is_match(s.tail_text) {
                let finished_len = locate_str_stop_finished_len(
                    s.token_window,
                    s.window_start,
                    s.new_accepted_len,
                    s.output_len,
                    s.decode_prefix,
                    &|text: &str| re.is_match(text),
                );
                return Some(StrStopDecision::MatchedRegex {
                    matched: stop_regex.clone(),
                    finished_len,
                });
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One token id = one char, id 0 is the NUL char — a deterministic stand-in
    /// for a tokenizer the location scan can be checked by hand.
    fn decode(ids: &[i32]) -> String {
        ids.iter().map(|&i| i as u8 as char).collect()
    }

    fn toks(s: &str) -> Vec<i32> {
        s.bytes().map(|b| b as i32).collect()
    }

    /// The state borrows all of its inputs, so each test builds it against its
    /// own locals.
    fn state<'a>(
        window: &'a [i32],
        tail_text: &'a str,
        window_start: usize,
        new_accepted_len: usize,
        stop_strs: &'a [String],
        stop_regex_strs: &'a [String],
        previously_decoded: &'a str,
    ) -> StrStopState<'a> {
        StrStopState {
            tail_text,
            previously_decoded,
            token_window: window,
            window_start,
            new_accepted_len,
            output_len: window_start + window.len(),
            stop_strs,
            stop_regex_strs,
            decode_prefix: &decode,
        }
    }

    #[test]
    fn tail_len_covers_newly_accepted_tokens() {
        // max_len 5: 6-char window, 3 new tokens → the match may start 6 + 2 = 8 back…
        assert_eq!(stop_match_tail_len(5, 0, 3, 100), 6 + 2);
        // …and never past the output.
        assert_eq!(stop_match_tail_len(5, 0, 3, 4), 4);
        // `new_accepted_len - 1` floors at 0.
        assert_eq!(stop_match_tail_len(5, 0, 0, 100), 6);
        // The regex bound wins over the string bound.
        assert_eq!(stop_match_tail_len(5, 10, 1, 100), 11);
    }

    #[test]
    fn prefix_gate_matches_python() {
        let stop = "STOP".to_string();
        let empty = String::new();
        assert!(!check_match_stop_str_prefix(
            "",
            std::slice::from_ref(&stop)
        ));
        assert!(!check_match_stop_str_prefix("hello", &[]));
        assert!(check_match_stop_str_prefix(
            "hello STOP",
            std::slice::from_ref(&stop)
        ));
        // Tail ends with the prefix of the stop ("ST" of "STOP").
        assert!(check_match_stop_str_prefix(
            "hello ST",
            std::slice::from_ref(&stop)
        ));
        // Empty stop strings are skipped (Python's `if not stop_str: continue`).
        assert!(!check_match_stop_str_prefix("hello ST", &[empty]));
        // A prefix that only matches in the MIDDLE of the tail is not a prefix.
        assert!(!check_match_stop_str_prefix("hello SX hello", &[stop]));
    }

    #[test]
    fn locate_finds_the_first_matching_prefix() {
        // Window "abcSTOP" from start 10, 1 new token… the scan starts at
        // max(1, 7 - 1 + 1) = 7: only the full window is scanned, and the
        // caller already knows it matches → finished_len = output_len = 17.
        let w = toks("abcSTOP");
        assert_eq!(
            locate_str_stop_finished_len(&w, 10, 1, 17, &decode, &|t: &str| t.contains("STOP"),),
            17
        );
        // With 3 new tokens the scan starts at 7 - 3 + 1 = 5: "abcST" misses,
        // "abcSTOP" hits → 10 + 7 = 17 again; shorten the match to "ST":
        // count 5 decodes to "abcST" → contains "ST" → 10 + 5 = 15.
        assert_eq!(
            locate_str_stop_finished_len(&w, 10, 3, 17, &decode, &|t: &str| t.contains("ST")),
            15
        );
    }

    #[test]
    fn stop_str_in_tail_finishes_with_trim_position() {
        let w = toks("STOP");
        let stop = "STOP".to_string();
        let tail = decode(&w);
        let stops = vec![stop.clone()];
        let st = state(&w, &tail, 0, 4, &stops, &[], "");
        assert_eq!(
            check_str_based_finish(&st),
            Some(StrStopDecision::MatchedStr {
                matched: "STOP".into(),
                finished_len: Some(4),
            })
        );
    }

    /// A stop matched only in the EARLIER text still finishes, without a trim
    /// position (Python leaves `finished_len` unset).
    #[test]
    fn stop_str_in_earlier_text_finishes_without_len() {
        let w = toks("x");
        let stop = "STOP".to_string();
        let tail = decode(&w);
        let stops = vec![stop];
        let st = state(&w, &tail, 5, 1, &stops, &[], "the STOP word was here");
        assert_eq!(
            check_str_based_finish(&st),
            Some(StrStopDecision::MatchedStr {
                matched: "STOP".into(),
                finished_len: None,
            })
        );
    }

    /// No match: the caller falls through to the token-based checks.
    #[test]
    fn no_match_is_none() {
        let w = toks("hi");
        let stop = "STOP".to_string();
        let tail = decode(&w);
        let stops = vec![stop];
        let st = state(&w, &tail, 0, 2, &stops, &[], "");
        assert_eq!(check_str_based_finish(&st), None);
    }

    #[test]
    fn regex_match_finishes_with_trim_position() {
        // "STOP" over window "xxSTOPyy" (8 tokens from start 3): the first
        // prefix containing a whole "STOP" is count 6 → 3 + 6 = 9.
        let w = toks("xxSTOPyy");
        let pat = "STOP".to_string();
        let tail = decode(&w);
        let pats = vec![pat];
        let st = state(&w, &tail, 3, 8, &[], &pats, "");
        assert_eq!(
            check_str_based_finish(&st),
            Some(StrStopDecision::MatchedRegex {
                matched: "STOP".into(),
                finished_len: 9,
            })
        );
    }

    #[test]
    fn uncompileable_regex_aborts_the_request() {
        let w = toks("a");
        let pat = "(".to_string();
        let tail = decode(&w);
        let pats = vec![pat];
        let st = state(&w, &tail, 0, 1, &[], &pats, "");
        assert!(matches!(
            check_str_based_finish(&st),
            Some(StrStopDecision::InvalidRegex { .. })
        ));
    }

    /// The stop-string branch shadows the regex branch: with both active and
    /// the string matched, the string wins (Python checks strings first and
    /// returns on the hit).
    #[test]
    fn stop_str_beats_regex() {
        let w = toks("STOP");
        let stop = "STOP".to_string();
        let pat = r".+".to_string();
        let tail = decode(&w);
        let stops = vec![stop];
        let pats = vec![pat];
        let st = state(&w, &tail, 0, 4, &stops, &pats, "");
        assert_eq!(
            check_str_based_finish(&st),
            Some(StrStopDecision::MatchedStr {
                matched: "STOP".into(),
                finished_len: Some(4),
            })
        );
    }

    /// An empty stop string matches everything (Python's `"" in text`) and
    /// finishes immediately on the first scannable prefix — the degenerate but
    /// real behavior, kept for parity.
    #[test]
    fn empty_stop_str_matches_everything() {
        let w = toks("z");
        let empty = String::new();
        let tail = decode(&w);
        let stops = vec![empty.clone()];
        let st = state(&w, &tail, 0, 1, &stops, &[], "");
        assert_eq!(
            check_str_based_finish(&st),
            Some(StrStopDecision::MatchedStr {
                matched: String::new(),
                finished_len: Some(1),
            })
        );
    }
}
