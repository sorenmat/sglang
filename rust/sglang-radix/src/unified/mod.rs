//! The unified multi-pool radix tree core — a faithful port of
//! `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py` plus the
//! tree-level hooks of its component drivers (`full_component.py`,
//! `swa_component.py`, `mamba_component.py`).
//!
//! One tree, up to three component layers per node:
//! - FULL (`CT_FULL`) — the Full-attention KV backbone; device + host values,
//!   path locks, heap-ordered eviction, leaf-set driven demote/delete;
//! - SWA (`CT_SWA`) — sliding-window-attention pool; window-bounded LRU
//!   refresh, tombstoning on internal eviction (frees the FULL value),
//!   uuid window locks;
//! - MAMBA (`CT_MAMBA`) — per-leaf SSM state; leaf-only data, single-node
//!   locks, excess-path-state capping.
//!
//! The tree is pure CPU and torch-free: component "values" are plain `i64`
//! pool indices. The caller (the PyO3 facade wrapper) applies the emitted
//! [`UCacheAction`]s against the real allocators and returns pool effects.
//! Page/event hashes stay caller-side (C++ native hashing); the core stores
//! whatever hash lists it is given and splits them on node splits.
//!
//! Deliberately out of scope for this port: the session-radix-cache
//! machinery (the constructor rejects `enable_session_radix_cache`).

mod evict;
mod hicache;
mod insert;
mod locks;
mod sanity;
mod tree;

pub use hicache::{
    ULoadBackSpec, UStorageBackupSpec, PHASE_BACKUP_HOST, PHASE_BACKUP_STORAGE, PHASE_LOAD_BACK,
    PHASE_PREFETCH,
};
pub use insert::UInsertParams;
pub use tree::{
    UChildKey, UMatchResult, UNode, UnifiedRadixTree, UStepResult, UWalkResult, UIntsertResult,
};

use crate::policy::EvictionPolicy;

/// The first `page_size` logical units of a key (child-map key head).
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum UHead {
    /// One token (page_size == 1, non-bigram).
    Token(i64),
    /// One bigram (page_size == 1, EAGLE bigram view).
    Bigram(i64, i64),
    /// A page of raw tokens (page_size > 1; 2*page values in bigram mode).
    Tokens(Vec<i64>),
}

/// Component type ids — mirror `ComponentType` (FULL=0, SWA=1, MAMBA=2, C128=3).
pub const CT_FULL: u8 = 0;
pub const CT_SWA: u8 = 1;
pub const CT_MAMBA: u8 = 2;
pub const CT_C128: u8 = 3;
/// `_NUM_COMPONENT_TYPES` — array sizing for per-component slots.
pub const NUM_CT: usize = 4;
/// `BASE_COMPONENT_TYPE`.
pub const CT_BASE: u8 = CT_FULL;

/// A component layer to evict. 1 = device, 2 = host, 3 = both (Python `EvictLayer`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    Device,
    Host,
    All,
}

impl Layer {
    pub const fn device(self) -> bool {
        matches!(self, Layer::Device | Layer::All)
    }
    pub const fn host(self) -> bool {
        matches!(self, Layer::Host | Layer::All)
    }
}

/// Construction-time configuration. Components are implied: SWA when
/// `sliding_window_size > 0`, MAMBA when `mamba_checkpoint_grid > 0`.
#[derive(Debug, Clone)]
pub struct UConfig {
    pub page_size: u32,
    /// EAGLE bigram view over keys (`params.is_eagle and MAMBA not in components`).
    pub is_eagle: bool,
    pub sliding_window_size: i64,
    /// `mamba_checkpoint_grid(page_size)`; 0 disables Mamba.
    pub mamba_checkpoint_grid: i64,
    /// `mamba_max_states_per_path`; < 0 disables the cap.
    pub mamba_max_states_per_path: i64,
    pub eviction_policy: EvictionPolicy,
    pub write_through_threshold: i64,
    pub is_write_back: bool,
    /// `has_swa_host_pool` (HiCache with an SWA host pool).
    pub has_swa_host_pool: bool,
    /// `--enable-session-radix-cache`; rejected (not supported) in this port.
    pub enable_session_radix_cache: bool,
}

impl UConfig {
    pub fn has_swa(&self) -> bool {
        self.sliding_window_size > 0
    }
    pub fn has_mamba(&self) -> bool {
        self.mamba_checkpoint_grid > 0
    }
    /// Smallest page-aligned size that still covers the sliding window
    /// (Python `SWAComponent.full_window_pages * page_size`).
    pub fn swa_tail_size(&self) -> i64 {
        let page = i64::from(self.page_size);
        (self.sliding_window_size + page - 1) / page * page
    }
}

/// Tree-emitted actions the caller applies against allocators/controllers.
/// Mirrors `cache_action.py` (CacheAction + ComponentAction) flattened.
#[derive(Debug, Clone, PartialEq)]
pub enum UCacheAction {
    /// `ReplaceWriteThroughOnNodeSplit`
    ReplaceWT {
        ack_id: i64,
        old_node: u32,
        new_node: u32,
        new_child_node: u32,
    },
    /// `FreeDeviceKV` — duplicate FULL KV slices the insert did not consume.
    FreeDeviceKV { chunks: Vec<Vec<i64>> },
    /// `FreeDeviceKVFullOnly`
    FreeDeviceKVFullOnly { chunks: Vec<Vec<i64>> },
    /// `BackupKV` — node ids, root-first (write-through chain).
    BackupKV { node_ids: Vec<u32> },
    /// `FreeComponentDeviceSlot`
    FreeComponentDeviceSlot { ct: u8, chunks: Vec<Vec<i64>> },
    /// `FreeComponentHostSlot`
    FreeComponentHostSlot { ct: u8, chunks: Vec<Vec<i64>> },
    /// `MambaEvictExcessPathStates` — the core defers it; the facade routes it
    /// back to `evict_excess_path_states` at the next barrier.
    MambaEvictExcess { tail_node: u32 },
    /// `RebuildFullToSWAMapping`
    RebuildFullToSWAMapping {
        full_indices: Vec<Vec<i64>>,
        swa_indices: Vec<Vec<i64>>,
    },
    /// `RecoverSWAWithLockedFull`
    RecoverSWAWithLockedFull {
        node: u32,
        kept_full: Vec<i64>,
        incoming_full: Vec<i64>,
    },
    /// `SWARebuild`
    SWARebuild { node: u32, source_value: Vec<i64> },
}

impl UCacheAction {
    /// Python `_is_deferrable_action`: fire-and-forget until the next barrier.
    pub fn is_deferrable(&self) -> bool {
        matches!(
            self,
            UCacheAction::FreeDeviceKV { .. }
                | UCacheAction::FreeDeviceKVFullOnly { .. }
                | UCacheAction::ReplaceWT { .. }
        )
    }
}

/// One HiCache pool transfer, mirroring `hicache_storage.PoolTransfer`'s
/// tree-visible fields. The caller converts to/from `PoolTransfer`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UTransfer {
    /// PoolName: 0 = KV, 1 = SWA, 2 = MAMBA.
    pub pool: u8,
    pub device_indices: Option<Vec<i64>>,
    pub host_indices: Option<Vec<i64>>,
    pub keys: Option<Vec<String>>,
    pub nodes_to_load: Vec<u32>,
    /// PoolHitPolicy: 0 = exact, 1 = trailing pages.
    pub hit_policy: u8,
}

/// `IncLockRefResult` (NodeId-flavored).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UIncLockResult {
    pub delta: i64,
    pub swa_uuid_for_lock: Option<i64>,
    pub swa_uuid_for_host_lock: Option<i64>,
    /// `skip_lock_node_ids`: (component_type, node ids).
    pub skip_lock_node_ids: Vec<(u8, Vec<u32>)>,
}

impl UIncLockResult {
    pub fn skip_ids(&self, ct: u8) -> &[u32] {
        self.skip_lock_node_ids
            .iter()
            .find(|(c, _)| *c == ct)
            .map(|(_, v)| v.as_slice())
            .unwrap_or(&[])
    }
}

/// `DecLockRefParams`.
#[derive(Debug, Clone, Default)]
pub struct UDecLockParams {
    pub swa_uuid_for_lock: Option<i64>,
    pub swa_uuid_for_host_lock: Option<i64>,
    pub skip_lock_node_ids: Vec<(u8, Vec<u32>)>,
}

impl UDecLockParams {
    pub fn skip_ids(&self, ct: u8) -> &[u32] {
        self.skip_lock_node_ids
            .iter()
            .find(|(c, _)| *c == ct)
            .map(|(_, v)| v.as_slice())
            .unwrap_or(&[])
    }
}

/// One-step result of a component device-eviction walk
/// (Python `EvictDeviceNextNodeResult`). `tracker` holds this step's deltas.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UEvictStep {
    pub node_id: Option<u32>,
    pub made_progress: bool,
    pub tracker: Vec<(u8, i64)>,
    pub device_frees: Vec<(u8, Vec<Vec<i64>>)>,
    pub host_frees: Vec<(u8, Vec<Vec<i64>>)>,
}

/// Result of `drop_subtree_no_host` / `demote` / host-eviction drives.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UEvictOutcome {
    pub tracker: Vec<(u8, i64)>,
    pub device_frees: Vec<(u8, Vec<Vec<i64>>)>,
    pub host_frees: Vec<(u8, Vec<Vec<i64>>)>,
    /// drop_subtree_no_host only.
    pub is_dropped: bool,
    /// evict_device_leaf only: BackupKV the facade must execute, then demote.
    pub backup_kv: Option<Vec<u32>>,
}

/// Raw KV-event log entry for the caller's event recorder
/// (`StorageMedium`: 1 = GPU, 2 = CPU).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UrkvEvent {
    pub op: u8,
    pub node: u32,
    pub medium: u8,
}

/// Snapshot of one live node, for caller-side dumps/parity.
#[derive(Debug, Clone, PartialEq)]
pub struct UNodeDump {
    pub id: u32,
    pub key: Vec<i64>,
    pub last_access: f64,
    pub creation: f64,
    pub hit_count: i64,
    pub priority: i64,
    pub full_value: Option<Vec<i64>>,
    pub full_host_value: Option<Vec<i64>>,
    pub swa_value: Option<Vec<i64>>,
    pub swa_host_value: Option<Vec<i64>>,
    pub mamba_value: Option<Vec<i64>>,
    pub mamba_host_value: Option<Vec<i64>>,
    pub lock_refs: [i32; NUM_CT],
    pub host_lock_refs: [i32; NUM_CT],
    pub swa_uuid: Option<i64>,
    pub swa_host_uuid: Option<i64>,
    pub write_through_pending: Option<i64>,
    pub load_back_pending: Option<u32>,
    pub in_device_leaves: bool,
    pub in_host_leaves: bool,
    pub is_duplicate_tracked: bool,
}
