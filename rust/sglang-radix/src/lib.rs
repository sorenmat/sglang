//! `sglang-radix` — the SGLang KV-cache radix tree core in Rust.
//!
//! Faithful ports of the SGLang KV-cache radix trees:
//! - `RadixTree` — the base `RadixCache`
//!   (`python/sglang/srt/mem_cache/radix_cache.py`): page-aligned keys,
//!   `extra_key`/`cache_salt` namespacing, the EAGLE bigram view,
//!   LRU/LFU/FIFO/MRU/FILO/Priority/SLRU eviction, lock-ref protection
//!   with incremental size bookkeeping, and match/insert node
//!   splitting;
//! - `SWARadixTree` — the sliding-window-attention dual-counter cache
//!   (`swa_radix_cache.py`);
//! - `MambaRadixTree` — the Mamba/GDN hybrid cache
//!   (`mamba_radix_cache.py`);
//! - `HiRadixTree` — the host-tier (HiCache) cache
//!   (`hiradix_cache.py`): device + host values, write-through
//!   backup/load-back, host eviction, and the write_back primitives.
//!
//! The tree is pure CPU and torch-free: node "values" are plain `i64` KV
//! cache indices. The caller (scheduler planner or the PyO3 facade)
//! releases evicted runs through the actual allocator.

mod hiradix;
mod key;
mod lru;
mod mamba;
mod policy;
mod swa;
mod tree;
mod unified;

pub use hiradix::{
    DropSubtreeResult, HiEvictResult, HiHostEvictResult, HiInsertResult, HiMatchResult, HiPolicy,
    HiRadixNode, HiRadixTree, LoadBackPlan,
};
pub use key::{common_prefix_len, RadixKey};
pub use lru::{Lru, LRUList};
pub use mamba::{
    MambaEvictResult, MambaFreeOps, MambaInsertResult, MambaMatchResult, MambaNode,
    MambaRadixTree,
};
pub use policy::{EvictionPolicy, Prio};
pub use swa::{
    FreeOps, SWADecResult, SWAEvictResult, SWAInsertResult, SWAMatchResult, SWANode, SWARecover,
    SWARadixTree,
};
pub use tree::{ChildKey, EvictResult, Head, InsertResult, MatchResult, NodeId, RawNode, RadixTree, ROOT};
pub use unified::{
    UCacheAction, UChildKey, UConfig, UHead, UIncLockResult, ULoadBackSpec, UNode, UNodeDump,
    UStepResult, UStorageBackupSpec, UTransfer, UWalkResult, UDecLockParams, UnifiedRadixTree,
    UIntsertResult, UMatchResult, UInsertParams, UEvictOutcome, UEvictStep, UrkvEvent,
    PHASE_BACKUP_HOST, PHASE_BACKUP_STORAGE, PHASE_LOAD_BACK, PHASE_PREFETCH,
    CT_BASE, CT_C128, CT_FULL, CT_MAMBA, CT_SWA, NUM_CT,
};
