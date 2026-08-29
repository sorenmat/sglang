//! The Python↔Rust boundary for the generation egress frame
//! ([`crate::stream`]) and the string-stop decisions
//! ([`crate::tokenizer_manager::stop_check`]).
//!
//! `build_generation_frame` takes the same per-request columns
//! `RustServer.push_generation` reads off the `BatchTokenIDOutput` and
//! returns the fully framed wire bytes — the msgpack header plus the raw
//! little-endian data columns — so the scheduler's GIL-held work shrinks to
//! the per-request collection (and the C-speed `array("i")` id flatten),
//! with the header encoding, column packing and ring push running here.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyBytes, PyDict, PyFloat, PyInt, PyIterator, PyList, PyString};
use rmpv::Value;

use crate::stream::{
    FinishCell, FlatCell, HiddenRow, RaggedCell, StreamColumns, stream_frame_bytes,
};
use crate::tokenizer_manager::stop_check::{
    StrStopDecision, StrStopState, check_match_stop_str_prefix, check_str_based_finish,
    locate_str_stop_finished_len, stop_match_tail_len,
};

fn bad_column(name: &str, e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("stream column {name}: {e}"))
}

fn iter_seq<'py>(name: &str, v: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyIterator>> {
    v.try_iter().map_err(|e| bad_column(name, e))
}

fn extract_strs(name: &str, v: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let mut out = Vec::new();
    for item in iter_seq(name, v)? {
        let item = item.map_err(|e| bad_column(name, e))?;
        out.push(item.extract::<String>().map_err(|e| bad_column(name, e))?);
    }
    Ok(out)
}

fn extract_u32s(name: &str, v: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    let mut out = Vec::new();
    for item in iter_seq(name, v)? {
        let item = item.map_err(|e| bad_column(name, e))?;
        out.push(item.extract::<u32>().map_err(|e| bad_column(name, e))?);
    }
    Ok(out)
}

/// One msgpack value from the Python objects a finish-reason dict holds
/// (str / int / float / None / list / dict), in the dict's own order.
fn extract_finish_value(name: &str, v: Bound<'_, PyAny>) -> PyResult<Value> {
    if v.is_none() {
        return Ok(Value::Nil);
    }
    if v.cast::<PyString>().is_ok() {
        return v
            .extract::<String>()
            .map(|s| Value::String(rmpv::Utf8String::from(s)))
            .map_err(|e| bad_column(name, e));
    }
    // Bool before int: Python bool is an int subclass and the int cast
    // would accept it.
    if v.cast::<pyo3::types::PyBool>().is_ok() {
        return v
            .extract::<bool>()
            .map(Value::from)
            .map_err(|e| bad_column(name, e));
    }
    if v.cast::<PyInt>().is_ok() {
        return v
            .extract::<i64>()
            .map(Value::from)
            .map_err(|e| bad_column(name, e));
    }
    if v.cast::<PyFloat>().is_ok() {
        return v
            .extract::<f64>()
            .map(Value::F64)
            .map_err(|e| bad_column(name, e));
    }
    if let Ok(l) = v.cast::<PyList>() {
        let items = l
            .iter()
            .map(|x| extract_finish_value(name, x))
            .collect::<PyResult<_>>()?;
        return Ok(Value::Array(items));
    }
    if let Ok(d) = v.cast::<PyDict>() {
        let mut m = Vec::new();
        for (k, val) in d.iter() {
            let k = k.extract::<String>().map_err(|e| bad_column(name, e))?;
            m.push((
                Value::String(rmpv::Utf8String::from(k)),
                extract_finish_value(name, val)?,
            ));
        }
        return Ok(Value::Map(m));
    }
    Err(bad_column(
        name,
        format!(
            "unsupported finish-reason value of type {:?}",
            v.get_type()
                .name()
                .map(|s| s.to_string())
                .unwrap_or("<untyped>".into())
        ),
    ))
}

fn extract_finish_cells(name: &str, v: &Bound<'_, PyAny>) -> PyResult<Vec<Option<FinishCell>>> {
    let mut out = Vec::new();
    for item in iter_seq(name, v)? {
        let item = item.map_err(|e| bad_column(name, e))?;
        out.push(if item.is_none() {
            None
        } else {
            // Python finish reasons are dicts; the msgpack map ships in the
            // dict's insertion order, so walk it directly.
            let d = item
                .cast::<PyDict>()
                .map_err(|_| bad_column(name, format!("{name} finish reason is not a dict")))?;
            let mut cell = Vec::new();
            for (k, val) in d.iter() {
                let k = k.extract::<String>().map_err(|e| bad_column(name, e))?;
                cell.push((k, extract_finish_value(name, val)?));
            }
            Some(cell)
        });
    }
    Ok(out)
}

fn extract_i32s(name: &str, v: &Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
    let mut out = Vec::new();
    for item in iter_seq(name, v)? {
        let item = item.map_err(|e| bad_column(name, e))?;
        out.push(item.extract::<i32>().map_err(|e| bad_column(name, e))?);
    }
    Ok(out)
}

/// Per-request flat cells: `list[list[float | None]]` — a `None` cell is the
/// empty cell; a `None` element is the leading sentinel only (the builder
/// enforces that). `None` argument = family not requested at all.
fn extract_flat_cells(name: &str, v: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<FlatCell>>> {
    let Some(v) = v else {
        return Ok(None);
    };
    let mut out = Vec::new();
    for cell in iter_seq(name, v)? {
        let cell = cell.map_err(|e| bad_column(name, e))?;
        if cell.is_none() {
            out.push(Vec::new());
            continue;
        }
        let mut row = Vec::new();
        for x in iter_seq(name, &cell)? {
            let x = x.map_err(|e| bad_column(name, e))?;
            row.push(if x.is_none() {
                None
            } else {
                Some(x.extract::<f32>().map_err(|e| bad_column(name, e))?)
            });
        }
        out.push(row);
    }
    Ok(Some(out))
}

/// Per-request flat idx cells: `list[list[int]]`.
fn extract_i32_cells(name: &str, v: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<Vec<i32>>>> {
    let Some(v) = v else {
        return Ok(None);
    };
    let mut out = Vec::new();
    for cell in iter_seq(name, v)? {
        let cell = cell.map_err(|e| bad_column(name, e))?;
        out.push(if cell.is_none() {
            Vec::new()
        } else {
            extract_i32s(name, &cell)?
        });
    }
    Ok(Some(out))
}

/// Per-request ragged val cells: `list[list[list[float] | None]]`.
fn extract_ragged_cells(
    name: &str,
    v: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<RaggedCell>>> {
    let Some(v) = v else {
        return Ok(None);
    };
    let mut out = Vec::new();
    for cell in iter_seq(name, v)? {
        let cell = cell.map_err(|e| bad_column(name, e))?;
        if cell.is_none() {
            out.push(Vec::new());
            continue;
        }
        let mut row = Vec::new();
        for pos in iter_seq(name, &cell)? {
            let pos = pos.map_err(|e| bad_column(name, e))?;
            row.push(if pos.is_none() {
                None
            } else {
                Some(
                    iter_seq(name, &pos)?
                        .map(|x| {
                            x.map_err(|e| bad_column(name, e))?
                                .extract::<f32>()
                                .map_err(|e| bad_column(name, e))
                        })
                        .collect::<PyResult<Vec<_>>>()?,
                )
            });
        }
        out.push(row);
    }
    Ok(Some(out))
}

/// Per-request ragged idx cells: `list[list[list[int] | None]]`.
type RaggedIdxCell = Vec<Option<Vec<i32>>>;

fn extract_ragged_i32_cells(
    name: &str,
    v: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<RaggedIdxCell>>> {
    let Some(v) = v else {
        return Ok(None);
    };
    let mut out = Vec::new();
    for cell in iter_seq(name, v)? {
        let cell = cell.map_err(|e| bad_column(name, e))?;
        if cell.is_none() {
            out.push(Vec::new());
            continue;
        }
        let mut row = Vec::new();
        for pos in iter_seq(name, &cell)? {
            let pos = pos.map_err(|e| bad_column(name, e))?;
            row.push(if pos.is_none() {
                None
            } else {
                Some(extract_i32s(name, &pos)?)
            });
        }
        out.push(row);
    }
    Ok(Some(out))
}

/// Per-request hidden rows: `list[list[float | int | list[...]]]`, the
/// recursive float union `_flatten_floats` walks.
fn extract_hidden_rows(
    name: &str,
    v: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<Vec<HiddenRow>>>> {
    fn one(name: &str, x: Bound<'_, PyAny>) -> PyResult<HiddenRow> {
        if x.cast::<PyFloat>().is_ok() {
            return x
                .extract::<f64>()
                .map(|f| HiddenRow::F(f as f32))
                .map_err(|e| bad_column(name, e));
        }
        if x.cast::<pyo3::types::PyBool>().is_ok() {
            // bool is an int subclass; Python's `isinstance(x, (int, float))`
            // admits it, so keep parity.
            return x
                .extract::<bool>()
                .map(|b| HiddenRow::F(f32::from(b)))
                .map_err(|e| bad_column(name, e));
        }
        if x.cast::<PyInt>().is_ok() {
            return x
                .extract::<i64>()
                .map(|i| HiddenRow::F(i as f32))
                .map_err(|e| bad_column(name, e));
        }
        if let Ok(l) = x.cast::<PyList>() {
            return Ok(HiddenRow::L(
                l.iter()
                    .map(|e| one(name, e))
                    .collect::<PyResult<Vec<_>>>()?,
            ));
        }
        Err(bad_column(
            name,
            format!(
                "hidden cell has a non-numeric/non-list element of type {:?}",
                x.get_type()
                    .name()
                    .map(|s| s.to_string())
                    .unwrap_or("<untyped>".into())
            ),
        ))
    }
    let Some(v) = v else {
        return Ok(None);
    };
    let mut out = Vec::new();
    for cell in iter_seq(name, v)? {
        let cell = cell.map_err(|e| bad_column(name, e))?;
        if cell.is_none() {
            out.push(Vec::new());
            continue;
        }
        out.push(
            iter_seq(name, &cell)?
                .map(|row| {
                    row.map_err(|e| bad_column(name, e))
                        .and_then(|row| one(name, row))
                })
                .collect::<PyResult<Vec<_>>>()?,
        );
    }
    Ok(Some(out))
}

/// Assemble [`StreamColumns`] from the Python objects `push_generation`
/// reads. `ids` is the batch-flattened i32 token stream (little-endian, as
/// `array("i").tobytes()` produces it); each optional family arg is either a
/// per-request container or `None` (nobody in the batch asked for it).
/// Crate-visible: `Server.push_generation_frame` in `lib.rs` shares this
/// extraction with the free `build_generation_frame`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn columns_from(
    rids: &Bound<'_, PyAny>,
    finished_reasons: &Bound<'_, PyAny>,
    prompt_tokens: &Bound<'_, PyAny>,
    tok_lens: &Bound<'_, PyAny>,
    ids: PyBackedBytes,
    out_lp_val: Option<&Bound<'_, PyAny>>,
    out_lp_idx: Option<&Bound<'_, PyAny>>,
    in_lp_val: Option<&Bound<'_, PyAny>>,
    in_lp_idx: Option<&Bound<'_, PyAny>>,
    out_top_val: Option<&Bound<'_, PyAny>>,
    out_top_idx: Option<&Bound<'_, PyAny>>,
    in_top_val: Option<&Bound<'_, PyAny>>,
    in_top_idx: Option<&Bound<'_, PyAny>>,
    out_tid_val: Option<&Bound<'_, PyAny>>,
    out_tid_idx: Option<&Bound<'_, PyAny>>,
    in_tid_val: Option<&Bound<'_, PyAny>>,
    in_tid_idx: Option<&Bound<'_, PyAny>>,
    hidden_rows: Option<&Bound<'_, PyAny>>,
) -> PyResult<StreamColumns> {
    if !ids.len().is_multiple_of(4) {
        return Err(PyValueError::new_err(format!(
            "stream column ids: {} bytes is not a whole number of i32 tokens",
            ids.len()
        )));
    }
    let toks = extract_u32s("tok_lens", tok_lens)?;
    let mut ids_vec: Vec<i32> = Vec::with_capacity(ids.len() / 4);
    for w in ids.as_ref().chunks_exact(4) {
        ids_vec.push(i32::from_le_bytes([w[0], w[1], w[2], w[3]]));
    }
    // An empty container is the "nobody asked" spelling; both spellings (None
    // and []) land on `None` here — the builder's `has_extra` gate then decides.
    fn empty_to_none<T>(v: Vec<T>) -> Option<Vec<T>> {
        (!v.is_empty()).then_some(v)
    }
    Ok(StreamColumns {
        rids: extract_strs("rids", rids)?,
        finish_reasons: extract_finish_cells("finished_reasons", finished_reasons)?,
        prompt_tokens: extract_u32s("prompt_tokens", prompt_tokens)?,
        tok_lens: toks,
        output_ids: ids_vec,
        out_lp_vals: extract_flat_cells("output_token_logprobs_val", out_lp_val)?
            .and_then(empty_to_none),
        out_lp_idxs: extract_i32_cells("output_token_logprobs_idx", out_lp_idx)?
            .and_then(empty_to_none),
        in_lp_vals: extract_flat_cells("input_token_logprobs_val", in_lp_val)?
            .and_then(empty_to_none),
        in_lp_idxs: extract_i32_cells("input_token_logprobs_idx", in_lp_idx)?
            .and_then(empty_to_none),
        out_top_vals: extract_ragged_cells("output_top_logprobs_val", out_top_val)?
            .and_then(empty_to_none),
        out_top_idxs: extract_ragged_i32_cells("output_top_logprobs_idx", out_top_idx)?
            .and_then(empty_to_none),
        in_top_vals: extract_ragged_cells("input_top_logprobs_val", in_top_val)?
            .and_then(empty_to_none),
        in_top_idxs: extract_ragged_i32_cells("input_top_logprobs_idx", in_top_idx)?
            .and_then(empty_to_none),
        out_tid_vals: extract_ragged_cells("output_token_ids_logprobs_val", out_tid_val)?
            .and_then(empty_to_none),
        out_tid_idxs: extract_ragged_i32_cells("output_token_ids_logprobs_idx", out_tid_idx)?
            .and_then(empty_to_none),
        in_tid_vals: extract_ragged_cells("input_token_ids_logprobs_val", in_tid_val)?
            .and_then(empty_to_none),
        in_tid_idxs: extract_ragged_i32_cells("input_token_ids_logprobs_idx", in_tid_idx)?
            .and_then(empty_to_none),
        hidden_rows: extract_hidden_rows("output_hidden_states", hidden_rows)?
            .and_then(empty_to_none),
    })
}

/// Build the full egress frame (`[BATCH][u32 len][header][cols…]`) from the
/// scheduler's per-request columns. The frame decodes byte-for-byte through
/// `for_each_chunk`; see `crate::stream` for the column contract.
#[pyfunction]
#[pyo3(
    signature = (
        rids,
        finished_reasons,
        prompt_tokens,
        tok_lens,
        ids,
        out_lp_val = None,
        out_lp_idx = None,
        in_lp_val = None,
        in_lp_idx = None,
        out_top_val = None,
        out_top_idx = None,
        in_top_val = None,
        in_top_idx = None,
        out_tid_val = None,
        out_tid_idx = None,
        in_tid_val = None,
        in_tid_idx = None,
        hidden_rows = None,
    )
)]
// The wide arg list IS the column surface — one per `BatchTokenIDOutput`
// field — not a call-site ergonomics problem.
#[allow(clippy::too_many_arguments)]
pub fn build_generation_frame(
    py: Python<'_>,
    rids: &Bound<'_, PyAny>,
    finished_reasons: &Bound<'_, PyAny>,
    prompt_tokens: &Bound<'_, PyAny>,
    tok_lens: &Bound<'_, PyAny>,
    ids: PyBackedBytes,
    out_lp_val: Option<&Bound<'_, PyAny>>,
    out_lp_idx: Option<&Bound<'_, PyAny>>,
    in_lp_val: Option<&Bound<'_, PyAny>>,
    in_lp_idx: Option<&Bound<'_, PyAny>>,
    out_top_val: Option<&Bound<'_, PyAny>>,
    out_top_idx: Option<&Bound<'_, PyAny>>,
    in_top_val: Option<&Bound<'_, PyAny>>,
    in_top_idx: Option<&Bound<'_, PyAny>>,
    out_tid_val: Option<&Bound<'_, PyAny>>,
    out_tid_idx: Option<&Bound<'_, PyAny>>,
    in_tid_val: Option<&Bound<'_, PyAny>>,
    in_tid_idx: Option<&Bound<'_, PyAny>>,
    hidden_rows: Option<&Bound<'_, PyAny>>,
) -> PyResult<pyo3::Py<PyBytes>> {
    let c = columns_from(
        rids,
        finished_reasons,
        prompt_tokens,
        tok_lens,
        ids,
        out_lp_val,
        out_lp_idx,
        in_lp_val,
        in_lp_idx,
        out_top_val,
        out_top_idx,
        in_top_val,
        in_top_idx,
        out_tid_val,
        out_tid_idx,
        in_tid_val,
        in_tid_idx,
        hidden_rows,
    )?;
    Ok(PyBytes::new(py, &stream_frame_bytes(&c)).unbind())
}

/// `Req._stop_match_tail_len`: how many trailing tokens a stop match may
/// reach, for the decoded-tail window.
#[pyfunction(name = "stop_match_tail_len")]
pub fn py_stop_match_tail_len(
    stop_str_max_len: usize,
    stop_regex_max_len: usize,
    new_accepted_len: usize,
    output_len: usize,
) -> usize {
    stop_match_tail_len(
        stop_str_max_len,
        stop_regex_max_len,
        new_accepted_len,
        output_len,
    )
}

/// `Req.check_match_stop_str_prefix`: the stream-interval gate.
#[pyfunction]
pub fn stop_prefix_match(tail_text: &str, stop_strs: Vec<String>) -> bool {
    check_match_stop_str_prefix(tail_text, &stop_strs)
}

/// `Req._locate_str_stop_finished_len`: the first scannable prefix of the
/// window whose decoded text matches. `prefix_texts[i]` is the decoded
/// `token_window[:i+1]` (the caller's tokenizer); only the needed range is
/// read.
#[pyfunction]
#[pyo3(signature = (token_window, window_start, new_accepted_len, output_len, prefix_texts, stop_str))]
pub fn locate_str_stop_len(
    token_window: Vec<i32>,
    window_start: usize,
    new_accepted_len: usize,
    output_len: usize,
    prefix_texts: Vec<String>,
    stop_str: &str,
) -> PyResult<usize> {
    if prefix_texts.len() != token_window.len() {
        return Err(PyValueError::new_err(format!(
            "locate_str_stop_len: {} prefix texts for {} window tokens",
            prefix_texts.len(),
            token_window.len()
        )));
    }
    let decode = |ids: &[i32]| prefix_texts[ids.len() - 1].clone();
    Ok(locate_str_stop_finished_len(
        &token_window,
        window_start,
        new_accepted_len,
        output_len,
        &decode,
        &|text| text.contains(stop_str),
    ))
}

/// `Req._check_str_based_finish`: the whole string/regex stop decision.
/// Returns `(kind, matched, finished_len)` with kind one of
/// `"str"` / `"regex"` / `"invalid_regex"` / `"none"`.
#[pyfunction]
#[pyo3(
    signature = (
        tail_text,
        previously_decoded,
        stop_strs,
        stop_regex_strs,
        token_window,
        window_start,
        new_accepted_len,
        output_len,
        prefix_texts,
    )
)]
// Same wide surface as the columns — one arg per request-side input.
#[allow(clippy::too_many_arguments)]
pub fn check_str_stop(
    tail_text: &str,
    previously_decoded: &str,
    stop_strs: Vec<String>,
    stop_regex_strs: Vec<String>,
    token_window: Vec<i32>,
    window_start: usize,
    new_accepted_len: usize,
    output_len: usize,
    prefix_texts: Vec<String>,
) -> PyResult<(String, Option<String>, Option<usize>)> {
    if prefix_texts.len() != token_window.len() {
        return Err(PyValueError::new_err(format!(
            "check_str_stop: {} prefix texts for {} window tokens",
            prefix_texts.len(),
            token_window.len()
        )));
    }
    let decode = |ids: &[i32]| prefix_texts[ids.len() - 1].clone();
    let s = StrStopState {
        tail_text,
        previously_decoded,
        token_window: &token_window,
        window_start,
        new_accepted_len,
        output_len,
        stop_strs: &stop_strs,
        stop_regex_strs: &stop_regex_strs,
        decode_prefix: &decode,
    };
    match check_str_based_finish(&s) {
        None => Ok(("none".into(), None, None)),
        Some(StrStopDecision::MatchedStr {
            matched,
            finished_len,
        }) => Ok(("str".into(), Some(matched), finished_len)),
        Some(StrStopDecision::MatchedRegex {
            matched,
            finished_len,
        }) => Ok(("regex".into(), Some(matched), Some(finished_len))),
        Some(StrStopDecision::InvalidRegex { pattern, .. }) => {
            Ok(("invalid_regex".into(), Some(pattern), None))
        }
    }
}

// `Server.push_generation_frame` (the egress-ring push with the frame built
// in Rust) lives in `lib.rs`, next to the other `Server` methods — pyo3
// allows one `#[pymethods]` block per type. It shares this module's
// `columns_from` with the free `build_generation_frame`.
