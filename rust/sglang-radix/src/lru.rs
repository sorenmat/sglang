//! Intrusive doubly-linked LRU list shared by the radix tree variants
//! (base, SWA, Mamba).
//!
//! This is a structural port of the Python `LRUList` (dummy head/tail
//! sentinels, move-to-MRU on every match/insert walk). Order is
//! maintained structurally, so the eviction order is deterministic
//! without a wall clock; each tree keeps its own per-node
//! last-access tick for the sanity-check role the Python float counter
//! plays.
//!
//! The list itself only knows `NodeId`s. Variant-specific predicates
//! (which lock ref gates eviction, which nodes are members, whether a
//! node is a leaf) live in the tree that owns the list — the free
//! walks in the Python trees differ per variant in exactly those
//! predicates, nothing else.

use crate::tree::NodeId;

/// A list position: a real node or one of the two dummy sentinels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lru {
    /// Dummy head, the most-recently-used side.
    Head,
    /// Dummy tail, the least-recently-used side.
    Tail,
    /// A real tree node.
    Node(NodeId),
}

impl Lru {
    pub fn node(self) -> Option<NodeId> {
        match self {
            Lru::Node(id) => Some(id),
            _ => None,
        }
    }
}

/// Node ids index the `prev`/`next` arrays; the dummy head/tail live in
/// `head_next`/`tail_prev`.
#[derive(Debug, Default, Clone)]
pub struct LRUList {
    /// Predecessor of node `id`; `None` = not in the list.
    prev: Vec<Option<Lru>>,
    /// Successor of node `id`; `None` = not in the list.
    next: Vec<Option<Lru>>,
    /// First real node after the head; `None` = empty list.
    head_next: Option<NodeId>,
    /// Last real node before the tail; `None` = empty list.
    tail_prev: Option<NodeId>,
}

impl LRUList {
    /// Grow the per-node arrays to `len` slots (new slots start
    /// unlinked).
    pub fn grow(&mut self, len: usize) {
        self.prev.resize_with(len, || None);
        self.next.resize_with(len, || None);
    }

    pub fn in_list(&self, id: NodeId) -> bool {
        self.prev.get(id as usize).is_some_and(Option::is_some)
    }

    /// The more-recently-used neighbor of `l` (port of
    /// `getattr(x, self.prv)`).
    pub fn predecessor(&self, l: Lru) -> Lru {
        match l {
            Lru::Head => unreachable!("no predecessor of the head"),
            Lru::Tail => self.tail_prev.map(Lru::Node).unwrap_or(Lru::Head),
            Lru::Node(id) => self.prev[id as usize].unwrap_or(Lru::Head),
        }
    }

    /// The less-recently-used neighbor of `l` (port of
    /// `getattr(x, self.nxt)`).
    pub fn successor(&self, l: Lru) -> Lru {
        match l {
            Lru::Head => self.head_next.map(Lru::Node).unwrap_or(Lru::Tail),
            Lru::Tail => unreachable!("no successor of the tail"),
            Lru::Node(id) => self.next[id as usize].unwrap_or(Lru::Tail),
        }
    }

    /// Insert `id` right after `old` (port of `_add_node_after`).
    pub fn add_after(&mut self, old: Lru, id: NodeId) {
        let old_next = self.successor(old);
        match old {
            Lru::Head => self.head_next = Some(id),
            Lru::Node(o) => self.next[o as usize] = Some(Lru::Node(id)),
            Lru::Tail => unreachable!(),
        }
        match old_next {
            Lru::Tail => self.tail_prev = Some(id),
            Lru::Node(n) => self.prev[n as usize] = Some(Lru::Node(id)),
            Lru::Head => unreachable!(),
        }
        self.prev[id as usize] = Some(old);
        self.next[id as usize] = Some(old_next);
    }

    /// Insert `id` as the most-recently-used node (port of
    /// `insert_mru`; the caller enforces the "not already in the
    /// list" invariant).
    pub fn insert_mru(&mut self, id: NodeId) {
        debug_assert!(
            !self.in_list(id),
            "insert_mru: node {id} already in the list"
        );
        self.add_after(Lru::Head, id);
    }

    /// Remove `id` from the list (port of `remove_node` without the
    /// id-cache bookkeeping).
    pub fn remove(&mut self, id: NodeId) {
        let p = self
            .prev
            .get(id as usize)
            .copied()
            .flatten()
            .expect("remove: node not in list");
        let n = self
            .next
            .get(id as usize)
            .copied()
            .flatten()
            .expect("remove: node not in list");
        match p {
            Lru::Head => self.head_next = n.node(),
            Lru::Node(x) => self.next[x as usize] = Some(n),
            Lru::Tail => unreachable!(),
        }
        match n {
            Lru::Tail => self.tail_prev = p.node(),
            Lru::Node(x) => self.prev[x as usize] = Some(p),
            Lru::Head => unreachable!(),
        }
        self.prev[id as usize] = None;
        self.next[id as usize] = None;
    }

    /// Move an existing node to the MRU position (port of
    /// `reset_node_mru`). No-op when the node is not in the list.
    pub fn reset_mru(&mut self, id: NodeId) {
        if !self.in_list(id) {
            return;
        }
        self.remove(id);
        self.insert_mru(id);
    }
}
