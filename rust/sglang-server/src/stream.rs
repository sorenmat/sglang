//! The per-iteration generation-egress frame, built in Rust.
//!
//! One decode step's whole batch becomes ONE frame on the from-scheduler ring:
//! a msgpack [`BatchHeader`] positional array plus the concatenated raw
//! little-endian data columns, in exactly the order
//! [`for_each_chunk`](crate::message::response::for_each_chunk) reads them.
//!
//! Until the Rust scheduler core owns the per-request output state itself,
//! Python's `RustServer.push_generation` still *collects* the columns from the
//! `Req`s; this module takes over everything from there — the ragged
//! logprob/hidden flattens, the header encoding, the column concatenation —
//! so the GIL-held work shrinks to the per-request collection and the ring
//! push runs detached. The wire bytes are the producer side of the contract
//! `for_each_chunk` decodes: a frame built here decodes back to the same
//! per-request columns, byte for byte.

use rmpv::Value;

/// One per-request cell of a flat val/idx family (per-token logprob values).
/// Elements are logprob values; `None` is only meaningful as the *leading*
/// element of an input-logprob cell (the first-prompt-token sentinel, which
/// maps to NaN on the wire — see [`StreamColumns::in_lp_vals`]).
pub type FlatCell = Vec<Option<f32>>;

/// One per-request cell of a ragged val/idx family (top-k / token-ids
/// logprobs): one entry per output position. `None` = a null position (0 on
/// the wire); a `Some` position holds that position's real values.
pub type RaggedCell = Vec<Option<Vec<f32>>>;

/// A hidden-state row: the `(float | list)` union Python's
/// `_flatten_floats` walks, one row per output position.
#[derive(Debug, Clone, PartialEq)]
pub enum HiddenRow {
    F(f32),
    L(Vec<HiddenRow>),
}

/// A finish-reason cell, exactly as Python's `BaseFinishReason.to_json()` puts
/// it on the wire: an ordered key/value map (insertion order is the byte
/// order). `None` entries in [`StreamColumns::finish_reasons`] encode as msgpack
/// `Nil`. The builder re-emits the values it is handed verbatim — it never
/// reshapes a reason, so any future Python-side key cannot drift.
pub type FinishCell = Vec<(String, Value)>;

/// One batch's generation columns — the same data `push_generation` reads off
/// the `BatchTokenIDOutput`, pre-extracted into Rust-owned columns.
///
/// `output_ids` is the batch-flattened token stream (one entry per new token,
/// concatenated per request) and `tok_lens` its per-request lengths; the pair
/// replaces Python's `array("i", chain.from_iterable(...))` + `map(len, ...)`.
/// Every per-request column is either empty (nobody in the batch asked for it)
/// or exactly `rids.len()` entries long — the all-or-nothing invariant the
/// decoder's `per_req_ok` check relies on.
#[derive(Debug, Clone, Default)]
pub struct StreamColumns {
    pub rids: Vec<String>,
    /// Per-request finish reason; `None` while streaming.
    pub finish_reasons: Vec<Option<FinishCell>>,
    pub prompt_tokens: Vec<u32>,
    /// Per-request lengths into `output_ids`.
    pub tok_lens: Vec<u32>,
    /// The batch-flattened new token ids, little-endian i32 elements.
    pub output_ids: Vec<i32>,

    /// The seven optional families, in header order. `None` (or an empty
    /// container) means no request in the batch asked for the family: with no
    /// family active the frame carries only the four core columns.
    pub out_lp_vals: Option<Vec<FlatCell>>,
    pub out_lp_idxs: Option<Vec<Vec<i32>>>,
    /// `first_none_to_nan` applies to this family only: a leading `None`
    /// element is the first-prompt-token sentinel and ships as NaN.
    pub in_lp_vals: Option<Vec<FlatCell>>,
    pub in_lp_idxs: Option<Vec<Vec<i32>>>,
    pub out_top_vals: Option<Vec<RaggedCell>>,
    pub out_top_idxs: Option<Vec<Vec<Option<Vec<i32>>>>>,
    pub in_top_vals: Option<Vec<RaggedCell>>,
    pub in_top_idxs: Option<Vec<Vec<Option<Vec<i32>>>>>,
    pub out_tid_vals: Option<Vec<RaggedCell>>,
    pub out_tid_idxs: Option<Vec<Vec<Option<Vec<i32>>>>>,
    pub in_tid_vals: Option<Vec<RaggedCell>>,
    pub in_tid_idxs: Option<Vec<Vec<Option<Vec<i32>>>>>,
    /// Per-request hidden rows (one row per output position).
    pub hidden_rows: Option<Vec<Vec<HiddenRow>>>,
}

/// The built frame: the msgpack header bytes plus the raw data columns, each
/// column already in `for_each_chunk`'s read order.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamFrame {
    pub header: Vec<u8>,
    pub cols: Vec<Vec<u8>>,
}

/// True when any of the seven optional families is populated for this batch —
/// the gate that decides whether the frame carries the 12 extra header
/// columns and 14 extra data buffers at all.
fn has_extra(c: &StreamColumns) -> bool {
    c.out_lp_vals.as_ref().is_some_and(|v| !v.is_empty())
        || c.in_lp_vals.as_ref().is_some_and(|v| !v.is_empty())
        || c.out_top_vals.as_ref().is_some_and(|v| !v.is_empty())
        || c.in_top_vals.as_ref().is_some_and(|v| !v.is_empty())
        || c.out_tid_vals.as_ref().is_some_and(|v| !v.is_empty())
        || c.in_tid_vals.as_ref().is_some_and(|v| !v.is_empty())
        || c.hidden_rows.as_ref().is_some_and(|v| !v.is_empty())
}

/// Check one per-request column's invariant: empty, or one entry per request.
fn check_per_req(name: &str, n: usize, batch_size: usize) {
    assert!(
        n == 0 || n == batch_size,
        "stream column {name}: {n} entries for a batch of {batch_size}"
    );
}

fn int_col(xs: &[u32]) -> Value {
    Value::Array(xs.iter().map(|&x| Value::from(i64::from(x))).collect())
}

fn str_col(xs: &[String]) -> Value {
    Value::Array(
        xs.iter()
            .map(|s| Value::String(rmpv::Utf8String::from(s.clone())))
            .collect(),
    )
}

fn finish_col(xs: &[Option<FinishCell>]) -> Value {
    Value::Array(
        xs.iter()
            .map(|f| match f {
                Some(m) => Value::Map(
                    m.iter()
                        .map(|(k, v)| (Value::String(rmpv::Utf8String::from(k.clone())), v.clone()))
                        .collect(),
                ),
                None => Value::Nil,
            })
            .collect(),
    )
}

/// Flatten one active flat family (mirrors `FlatPairColumns`): per-request
/// element counts plus the concatenated f32/i32 buffers. An inactive family
/// (empty container) contributes empty columns — present in place, never
/// omitted, so the header arity the decoder expects is unchanged.
fn flat_family(
    vals: &Option<Vec<FlatCell>>,
    idxs: &Option<Vec<Vec<i32>>>,
    name: &str,
    first_none_to_nan: bool,
    batch_size: usize,
) -> (Vec<u32>, Vec<f32>, Vec<i32>) {
    assert!(
        vals.is_some() == idxs.is_some(),
        "stream family {name}: val/idx containers must be both present or both absent"
    );
    let Some(vals) = vals else {
        return (Vec::new(), Vec::new(), Vec::new());
    };
    let idxs = idxs.as_ref().expect("checked above");
    check_per_req(&format!("{name}_val"), vals.len(), batch_size);
    check_per_req(&format!("{name}_idx"), idxs.len(), batch_size);
    if vals.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let mut lens = Vec::with_capacity(batch_size);
    let mut v = Vec::new();
    let mut i = Vec::new();
    for (j, (vv, ii)) in vals.iter().zip(idxs.iter()).enumerate() {
        // Parity with `FlatPairColumns.accept`: only `len(vv)` is recorded, so a
        // longer idx column would shift every later column's offset on the wire.
        assert_eq!(
            ii.len(),
            vv.len(),
            "stream family {name}: request {j} has {} idx entries but {} vals",
            ii.len(),
            vv.len()
        );
        if first_none_to_nan && vv.first().is_some_and(|x| x.is_none()) {
            v.push(f32::NAN);
            for x in &vv[1..] {
                v.push(x.expect("only the leading input-logprob value may be None"));
            }
        } else {
            for x in vv {
                v.push(x.expect("logprob cell has an interior None (Python would crash here)"));
            }
        }
        i.extend_from_slice(ii);
        lens.push(vv.len() as u32);
    }
    (lens, v, i)
}

/// Flatten one ragged family (mirrors `RaggedPairColumns`): per-request
/// position counts, the flat per-position lengths, and the concatenated
/// f32/i32 buffers. A null position (falsy val) ships as length 0.
fn ragged_family(
    vals: &Option<Vec<RaggedCell>>,
    idxs: &Option<Vec<Vec<Option<Vec<i32>>>>>,
    name: &str,
    batch_size: usize,
) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<i32>) {
    assert!(
        vals.is_some() == idxs.is_some(),
        "stream family {name}: val/idx containers must be both present or both absent"
    );
    let Some(vals) = vals else {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    };
    let idxs = idxs.as_ref().expect("checked above");
    check_per_req(&format!("{name}_val"), vals.len(), batch_size);
    check_per_req(&format!("{name}_idx"), idxs.len(), batch_size);
    if vals.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }
    let mut reqlens = Vec::with_capacity(batch_size);
    let mut poslens = Vec::new();
    let mut v = Vec::new();
    let mut i = Vec::new();
    for (j, (cell, icell)) in vals.iter().zip(idxs.iter()).enumerate() {
        if cell.is_empty() {
            assert!(
                icell.is_empty(),
                "stream family {name}: request {j} has {icell_len} idx positions but the val cell is empty",
                icell_len = icell.len()
            );
            reqlens.push(0);
            continue;
        }
        assert_eq!(
            icell.len(),
            cell.len(),
            "stream family {name}: request {j} has {} idx positions but {} val positions",
            icell.len(),
            cell.len()
        );
        for (p, (pv, pi)) in cell.iter().zip(icell.iter()).enumerate() {
            let pv_empty = pv.as_ref().is_none_or(Vec::is_empty);
            let pi_empty = pi.as_ref().is_none_or(Vec::is_empty);
            if pv_empty {
                assert!(
                    pi_empty,
                    "stream family {name}: request {j} position {p}: idx has entries but val is empty"
                );
                poslens.push(0);
            } else {
                let (pv, pi) = (
                    pv.as_ref().expect("pv_empty checked"),
                    pi.as_ref()
                        .expect("a truthy val position always has idx in the payload"),
                );
                assert_eq!(
                    pi.len(),
                    pv.len(),
                    "stream family {name}: request {j} position {p}: idx len {} != val len {}",
                    pi.len(),
                    pv.len()
                );
                v.extend_from_slice(pv);
                i.extend_from_slice(pi);
                poslens.push(pv.len() as u32);
            }
        }
        reqlens.push(cell.len() as u32);
    }
    (reqlens, poslens, v, i)
}

/// Recursively flatten one hidden row (mirrors `_flatten_floats`).
fn flatten_hidden_row(row: &HiddenRow, out: &mut Vec<f32>) {
    match row {
        HiddenRow::F(f) => out.push(*f),
        HiddenRow::L(es) => {
            for e in es {
                flatten_hidden_row(e, out);
            }
        }
    }
}

/// Flatten the hidden family (mirrors `NestedRowColumns`): per-request row
/// counts, the per-row lengths, and one concatenated f32 buffer.
fn hidden_family(
    rows: &Option<Vec<Vec<HiddenRow>>>,
    batch_size: usize,
) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let Some(rows) = rows else {
        return (Vec::new(), Vec::new(), Vec::new());
    };
    check_per_req("output_hidden_states", rows.len(), batch_size);
    if rows.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let mut reqlens = Vec::with_capacity(batch_size);
    let mut poslens = Vec::new();
    let mut v = Vec::new();
    for cell in rows {
        for row in cell {
            let before = v.len();
            flatten_hidden_row(row, &mut v);
            poslens.push((v.len() - before) as u32);
        }
        reqlens.push(cell.len() as u32);
    }
    (reqlens, poslens, v)
}

fn f32_buf(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i32_buf(xs: &[i32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Build the frame (see [`StreamFrame`]). Panics on a violated column
/// invariant (the producer is in-process; a mismatched frame would decode as
/// another column's bytes, so fail loud instead).
pub fn build_stream_frame(c: &StreamColumns) -> StreamFrame {
    let n = c.rids.len();
    assert_eq!(
        c.finish_reasons.len(),
        n,
        "stream frame: {} finish reasons for {} rids",
        c.finish_reasons.len(),
        n
    );
    assert_eq!(
        c.prompt_tokens.len(),
        n,
        "stream frame: {} prompt_tokens for {} rids",
        c.prompt_tokens.len(),
        n
    );
    assert_eq!(
        c.tok_lens.len(),
        n,
        "stream frame: {} tok_lens for {} rids",
        c.tok_lens.len(),
        n
    );
    assert_eq!(
        c.tok_lens.iter().map(|&v| v as usize).sum::<usize>(),
        c.output_ids.len(),
        "stream frame: tok_lens sum {} != {} flattened ids",
        c.tok_lens.iter().sum::<u32>(),
        c.output_ids.len()
    );

    let extra = has_extra(c);

    let (mut header_cols, mut data_cols): (Vec<Value>, Vec<Vec<u8>>) =
        (Vec::with_capacity(4 + 12), Vec::with_capacity(1 + 14));
    header_cols.push(str_col(&c.rids));
    header_cols.push(finish_col(&c.finish_reasons));
    header_cols.push(int_col(&c.prompt_tokens));
    header_cols.push(int_col(&c.tok_lens));
    data_cols.push(i32_buf(&c.output_ids));

    if extra {
        let (out_lp_lens, out_lp_v, out_lp_i) = flat_family(
            &c.out_lp_vals,
            &c.out_lp_idxs,
            "output_token_logprobs",
            false,
            n,
        );
        let (in_lp_lens, in_lp_v, in_lp_i) = flat_family(
            &c.in_lp_vals,
            &c.in_lp_idxs,
            "input_token_logprobs",
            true,
            n,
        );
        let (out_top_req, out_top_pos, out_top_v, out_top_i) =
            ragged_family(&c.out_top_vals, &c.out_top_idxs, "output_top_logprobs", n);
        let (in_top_req, in_top_pos, in_top_v, in_top_i) =
            ragged_family(&c.in_top_vals, &c.in_top_idxs, "input_top_logprobs", n);
        let (out_tid_req, out_tid_pos, out_tid_v, out_tid_i) = ragged_family(
            &c.out_tid_vals,
            &c.out_tid_idxs,
            "output_token_ids_logprobs",
            n,
        );
        let (in_tid_req, in_tid_pos, in_tid_v, in_tid_i) = ragged_family(
            &c.in_tid_vals,
            &c.in_tid_idxs,
            "input_token_ids_logprobs",
            n,
        );
        let (hidden_req, hidden_pos, hidden_v) = hidden_family(&c.hidden_rows, n);

        header_cols.push(int_col(&out_lp_lens));
        header_cols.push(int_col(&in_lp_lens));
        header_cols.push(int_col(&out_top_req));
        header_cols.push(int_col(&out_top_pos));
        header_cols.push(int_col(&in_top_req));
        header_cols.push(int_col(&in_top_pos));
        header_cols.push(int_col(&out_tid_req));
        header_cols.push(int_col(&out_tid_pos));
        header_cols.push(int_col(&in_tid_req));
        header_cols.push(int_col(&in_tid_pos));
        header_cols.push(int_col(&hidden_req));
        header_cols.push(int_col(&hidden_pos));

        data_cols.push(f32_buf(&out_lp_v));
        data_cols.push(i32_buf(&out_lp_i));
        data_cols.push(f32_buf(&in_lp_v));
        data_cols.push(i32_buf(&in_lp_i));
        data_cols.push(f32_buf(&out_top_v));
        data_cols.push(i32_buf(&out_top_i));
        data_cols.push(f32_buf(&in_top_v));
        data_cols.push(i32_buf(&in_top_i));
        data_cols.push(f32_buf(&out_tid_v));
        data_cols.push(i32_buf(&out_tid_i));
        data_cols.push(f32_buf(&in_tid_v));
        data_cols.push(i32_buf(&in_tid_i));
        data_cols.push(f32_buf(&hidden_v));
    }

    let mut header = Vec::new();
    rmpv::encode::write_value(&mut header, &Value::Array(header_cols))
        .expect("msgpack encoding of owned values cannot fail");

    StreamFrame {
        header,
        cols: data_cols,
    }
}

/// The full wire frame: `[DISPATCH_TAG_BATCH][u32 LE header len][header][cols…]`
/// — what lands on the from-scheduler ring (see
/// [`frame_decode_batch_cols`](crate::message::response::frame_decode_batch_cols)).
pub fn stream_frame_bytes(c: &StreamColumns) -> bytes::Bytes {
    let f = build_stream_frame(c);
    let cols: Vec<&[u8]> = f.cols.iter().map(|c| c.as_slice()).collect();
    crate::message::response::frame_decode_batch_cols(&f.header, &cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::response::{for_each_chunk, frame_decode_batch_cols};

    fn decode(frame: &[u8]) -> (bool, Vec<crate::message::response::ChunkEvent>) {
        let mut events = Vec::new();
        let d = for_each_chunk(frame, |ev| events.push(ev));
        (d.ok, events)
    }

    fn stop(matched: i64) -> FinishCell {
        vec![
            ("type".into(), Value::from("stop")),
            ("matched".into(), Value::from(matched)),
        ]
    }

    /// The common frame (no family active): exactly four header columns and one
    /// data column, byte-pinned — a producer that later "helpfully" emits zero
    /// columns for inactive families would break this.
    #[test]
    fn common_frame_is_four_columns_plus_ids() {
        let c = StreamColumns {
            rids: vec!["r0".into(), "r1".into()],
            finish_reasons: vec![None, Some(stop(7))],
            prompt_tokens: vec![4, 5],
            tok_lens: vec![2, 0],
            output_ids: vec![10, 11],
            ..Default::default()
        };
        let f = build_stream_frame(&c);
        assert_eq!(f.cols.len(), 1, "no extras: only the ids column");
        assert_eq!(
            f.cols[0],
            [10i32, 11]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>()
        );
        // The header is a 4-element array; the first element is the rid column.
        assert_eq!(f.header[0], 0x94, "fixarray(4)");
        assert_eq!(
            &f.header[1..5],
            &[0x92, 0xa2, b'r', b'0'],
            "fixarray(2) + fixstr \"r0\""
        );

        let framed = frame_decode_batch_cols(&f.header, &[&f.cols[0]]);
        let (ok, evs) = decode(&framed[1..]);
        assert!(ok);
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].token_ids, vec![10, 11]);
        assert_eq!(evs[0].prompt_tokens, 4);
        assert!(evs[0].finish_reason.is_none());
        assert_eq!(evs[1].token_ids, Vec::<i32>::new());
        assert!(evs[1].finish_reason.is_some());
    }

    /// The B=0 idle batch: four empty header columns, empty ids column, and the
    /// frame still decodes (the decoder's `per_req_ok` admits the empty form).
    #[test]
    fn empty_batch_frame_round_trips() {
        let c = StreamColumns::default();
        let framed = stream_frame_bytes(&c);
        let (ok, evs) = decode(&framed[1..]);
        assert!(ok);
        assert!(evs.is_empty());
    }

    /// Every family active, each with a DISTINCT shape: flat with a leading
    /// None sentinel (input logprobs), ragged with a null position, and a
    /// nested hidden row. Round-trips through the real decoder, and the
    /// inactive-vs-active spelling rule holds: an inactive family in an
    /// extras frame contributes empty columns in place.
    #[test]
    fn full_extras_frame_round_trips() {
        let f32_nan = f32::NAN.to_le_bytes();
        let c = StreamColumns {
            rids: vec!["a".into(), "b".into()],
            finish_reasons: vec![None, None],
            prompt_tokens: vec![3, 4],
            tok_lens: vec![1, 1],
            output_ids: vec![100, 200],
            out_lp_vals: Some(vec![vec![Some(-1.0), Some(-2.0)], vec![]]),
            out_lp_idxs: Some(vec![vec![11, 12], vec![]]),
            // Leading None = the first-prompt-token sentinel → NaN on the wire.
            in_lp_vals: Some(vec![vec![None, Some(0.5)], vec![Some(0.25)]]),
            in_lp_idxs: Some(vec![vec![1, 2], vec![3]]),
            out_top_vals: Some(vec![
                vec![Some(vec![0.1, 0.2]), None],
                vec![Some(vec![-9.0])],
            ]),
            out_top_idxs: Some(vec![vec![Some(vec![5, 6]), None], vec![Some(vec![7])]]),
            in_top_vals: Some(vec![vec![], vec![]]),
            in_top_idxs: Some(vec![vec![], vec![]]),
            out_tid_vals: Some(vec![vec![], vec![Some(vec![0.3])]]),
            out_tid_idxs: Some(vec![vec![], vec![Some(vec![42])]]),
            in_tid_vals: Some(vec![vec![None], vec![None]]),
            in_tid_idxs: Some(vec![vec![None], vec![None]]),
            hidden_rows: Some(vec![
                vec![HiddenRow::L(vec![HiddenRow::F(1.0), HiddenRow::F(2.0)])],
                vec![HiddenRow::F(0.5)],
            ]),
        };
        let f = build_stream_frame(&c);
        let mut cursor = &f.header[..];
        let hdr: Value = rmpv::decode::read_value(&mut cursor).expect("header decodes");
        assert_eq!(
            hdr.as_array().unwrap().len(),
            16,
            "all 16 header columns present"
        );
        assert_eq!(f.cols.len(), 14, "ids + 13 family buffers");

        // The leading None shipped as the x86 NaN bit pattern CPython writes.
        assert!(f.cols[3].windows(4).any(|w| w == f32_nan.as_slice()));

        let col_refs: Vec<&[u8]> = f.cols.iter().map(|c| c.as_slice()).collect();
        let framed = frame_decode_batch_cols(&f.header, &col_refs);
        let (ok, evs) = decode(&framed[1..]);
        assert!(ok);
        assert_eq!(evs.len(), 2);

        let ex0 = evs[0].extras.as_deref().expect("req0 carries extras");
        assert_eq!(ex0.out_lp_val, vec![-1.0, -2.0]);
        assert_eq!(ex0.out_lp_idx, vec![11, 12]);
        assert_eq!(ex0.in_lp_val.len(), 2);
        assert!(ex0.in_lp_val[0].is_nan(), "leading None → NaN");
        assert_eq!(ex0.in_lp_val[1], 0.5);
        assert_eq!(ex0.out_top_lens, vec![2, 0], "null position keeps length 0");
        assert_eq!(ex0.out_top_val, vec![0.1, 0.2]);
        assert!(
            ex0.in_top_lens.is_empty(),
            "inactive in_top family: empty column"
        );
        assert!(ex0.out_tid_lens.is_empty());
        assert_eq!(ex0.hidden_val, vec![1.0, 2.0]);
        assert_eq!(ex0.hidden_lens, vec![2]);

        let ex1 = evs[1].extras.as_deref().expect("req1 carries extras");
        assert!(ex1.out_lp_val.is_empty());
        assert_eq!(ex1.in_lp_val, vec![0.25]);
        assert_eq!(ex1.out_top_val, vec![-9.0]);
        assert_eq!(ex1.out_top_lens, vec![1]);
        assert_eq!(ex1.out_tid_val, vec![0.3]);
        assert_eq!(ex1.out_tid_idx, vec![42]);
        assert!(ex1.in_tid_val.is_empty());
        assert_eq!(ex1.hidden_val, vec![0.5]);
        assert_eq!(ex1.hidden_lens, vec![1]);
    }

    /// A family present but with NO populated cell is inactive: empty columns,
    /// exactly Python's `active` gating. And with no family active at all the
    /// extra columns don't exist.
    #[test]
    fn inactive_families_emit_empty_columns() {
        let c = StreamColumns {
            rids: vec!["a".into()],
            finish_reasons: vec![None],
            prompt_tokens: vec![1],
            tok_lens: vec![1],
            output_ids: vec![9],
            // out_lp active, every other family an empty container:
            out_lp_vals: Some(vec![vec![Some(0.25)]]),
            out_lp_idxs: Some(vec![vec![4]]),
            in_lp_vals: Some(vec![vec![]]),
            in_lp_idxs: Some(vec![vec![]]),
            out_top_vals: Some(vec![vec![]]),
            out_top_idxs: Some(vec![vec![]]),
            in_top_vals: Some(vec![vec![]]),
            in_top_idxs: Some(vec![vec![]]),
            out_tid_vals: Some(vec![vec![]]),
            out_tid_idxs: Some(vec![vec![]]),
            in_tid_vals: Some(vec![vec![]]),
            in_tid_idxs: Some(vec![vec![]]),
            hidden_rows: Some(vec![vec![]]),
        };
        let f = build_stream_frame(&c);
        assert_eq!(
            f.cols.len(),
            14,
            "an extras frame keeps all 14 data columns"
        );

        // Without any active family the frame is back to the 4-column form.
        let plain = StreamColumns {
            rids: vec!["a".into()],
            finish_reasons: vec![None],
            prompt_tokens: vec![1],
            tok_lens: vec![1],
            output_ids: vec![9],
            ..Default::default()
        };
        assert_eq!(build_stream_frame(&plain).cols.len(), 1);

        let framed = stream_frame_bytes(&c);
        let (ok, evs) = decode(&framed[1..]);
        assert!(ok);
        let ex = evs[0].extras.as_deref().expect("out_lp is active");
        assert_eq!(ex.out_lp_val, vec![0.25]);
        assert!(ex.in_lp_val.is_empty());
        assert!(ex.out_top_lens.is_empty());
        assert!(ex.hidden_lens.is_empty());
    }

    /// The parity asserts fire on the shapes Python would have crashed on: a
    /// longer idx column, a val position with no idx, an idx position where the
    /// val is empty, and a ragged idx/val length mismatch.
    #[test]
    #[should_panic(expected = "idx len 2 != val len 1")]
    fn ragged_idx_longer_than_val_panics() {
        let c = StreamColumns {
            rids: vec!["a".into()],
            finish_reasons: vec![None],
            prompt_tokens: vec![0],
            tok_lens: vec![1],
            output_ids: vec![1],
            out_top_vals: Some(vec![vec![Some(vec![0.5])]]),
            out_top_idxs: Some(vec![vec![Some(vec![1, 2])]]),
            ..Default::default()
        };
        build_stream_frame(&c);
    }

    #[test]
    #[should_panic(expected = "idx has entries but val is empty")]
    fn ragged_idx_where_val_empty_panics() {
        let c = StreamColumns {
            rids: vec!["a".into()],
            finish_reasons: vec![None],
            prompt_tokens: vec![0],
            tok_lens: vec![1],
            output_ids: vec![1],
            out_top_vals: Some(vec![vec![None]]),
            out_top_idxs: Some(vec![vec![Some(vec![1])]]),
            ..Default::default()
        };
        build_stream_frame(&c);
    }

    #[test]
    #[should_panic(expected = "8 finish reasons for 4 rids")]
    fn per_req_length_mismatch_panics() {
        let c = StreamColumns {
            rids: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            finish_reasons: vec![None; 8],
            prompt_tokens: vec![0, 0, 0, 0],
            tok_lens: vec![0, 0, 0, 0],
            ..Default::default()
        };
        build_stream_frame(&c);
    }

    #[test]
    #[should_panic(expected = "tok_lens sum 3 != 2 flattened ids")]
    fn tok_lens_sum_mismatch_panics() {
        let c = StreamColumns {
            rids: vec!["a".into(), "b".into()],
            finish_reasons: vec![None, None],
            prompt_tokens: vec![0, 0],
            tok_lens: vec![2, 1],
            output_ids: vec![1, 2],
            ..Default::default()
        };
        build_stream_frame(&c);
    }

    /// A 32-char rid encodes as str8 (msgpack has no fixstr that wide); a
    /// 200-char rid as str16 — pinning the minimal-form rule the byte parity
    /// against msgspec relies on.
    #[test]
    fn string_columns_use_minimal_msgpack_forms() {
        let long = "x".repeat(200);
        let c = StreamColumns {
            rids: vec!["0123456789abcdef0123456789abcdef".into(), long],
            finish_reasons: vec![None, None],
            prompt_tokens: vec![0, 0],
            tok_lens: vec![0, 0],
            ..Default::default()
        };
        let f = build_stream_frame(&c);
        // The exact header the builder must emit, built independently.
        let header = Value::Array(vec![
            Value::Array(vec![
                Value::from("0123456789abcdef0123456789abcdef"),
                Value::from("x".repeat(200)),
            ]),
            Value::Array(vec![Value::Nil, Value::Nil]),
            Value::Array(vec![Value::from(0i64), Value::from(0i64)]),
            Value::Array(vec![Value::from(0i64), Value::from(0i64)]),
        ]);
        let mut expected = Vec::new();
        rmpv::encode::write_value(&mut expected, &header).unwrap();
        // Both rids are short enough for the str8 marker: 32 and 200 chars.
        assert!(f.header.windows(2).any(|w| w == [0xd9, 32]));
        assert!(f.header.windows(2).any(|w| w == [0xd9, 0xc8]));
        assert_eq!(f.header, expected);
    }
}
