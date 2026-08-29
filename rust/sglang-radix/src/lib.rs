//! `sglang-radix` — the SGLang KV-cache radix tree core in Rust.
//!
//! A faithful port of the base `RadixCache`
//! (`python/sglang/srt/mem_cache/radix_cache.py`): page-aligned keys,
//! `extra_key`/`cache_salt` namespacing, the EAGLE bigram view, LRU/LFU/
//! FIFO/MRU/FILO/Priority/SLRU eviction, lock-ref protection with
//! incremental size bookkeeping, and match/insert node splitting.
//!
//! The tree is pure CPU and torch-free: node "values" are plain `i64` KV
//! cache indices. The caller (scheduler planner or the PyO3 facade)
//! releases evicted runs through the actual allocator.

mod key;
mod policy;
mod swa;
mod tree;

pub use key::{common_prefix_len, RadixKey};
pub use policy::{EvictionPolicy, Prio};
pub use swa::{
    FreeOps, LRUList, Lru, SWADecResult, SWAEvictResult, SWAInsertResult, SWAMatchResult,
    SWANode, SWARecover, SWARadixTree,
};
pub use tree::{ChildKey, EvictResult, Head, InsertResult, MatchResult, NodeId, RawNode, RadixTree, ROOT};
