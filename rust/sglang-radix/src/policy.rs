//! Eviction strategies — port of `python/sglang/srt/mem_cache/evict_policy.py`
//! plus the `get_eviction_strategy` registry in `mem_cache/utils.py`.
//!
//! The heap orders by ascending `Prio` (smallest evicted first), matching
//! Python's `heapq` min-heap over `strategy.get_priority(node)`. Python
//! tie-breaks equal priorities with `TreeNode.__lt__` (last_access_time)
//! over a set-ordered leaf list; ties here fall back to node id, which is
//! total and deterministic (Python's tie order is itself run-dependent).

use crate::tree::RawNode;

/// One eviction-policy priority. `Prio` is the ascending eviction order
/// (smaller first); components mirror the Python priority tuples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prio {
    /// Python primary component.
    pub a: i64,
    /// Python secondary component.
    pub b: i64,
    /// Final tie-break: node id (total order).
    pub c: u32,
}

impl Prio {
    pub(crate) fn key(&self) -> (i64, i64, u32) {
        (self.a, self.b, self.c)
    }
}

impl PartialOrd for Prio {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Prio {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key().cmp(&other.key())
    }
}

fn clamp_u64(v: u64) -> i64 {
    v.min(i64::MAX as u64) as i64
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EvictionPolicy {
    #[default]
    Lru,
    Lfu,
    Fifo,
    Mru,
    Filo,
    Priority,
    Slru { threshold: u64 },
}

impl EvictionPolicy {
    /// Parse a policy name the way `get_eviction_strategy` does
    /// (`lower()`, exact names `lru|lfu|fifo|mru|filo|priority|slru`).
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.to_ascii_lowercase().as_str() {
            "lru" => Ok(Self::Lru),
            "lfu" => Ok(Self::Lfu),
            "fifo" => Ok(Self::Fifo),
            "mru" => Ok(Self::Mru),
            "filo" => Ok(Self::Filo),
            "priority" => Ok(Self::Priority),
            "slru" => Ok(Self::Slru { threshold: 2 }),
            other => Err(format!(
                "unknown eviction policy {other:?} (expected lru|lfu|fifo|mru|filo|priority|slru)"
            )),
        }
    }

    /// Eviction priority; smaller values are evicted first.
    ///
    /// Node fields: `last_access` (u64 walk clock), `hit_count`,
    /// `priority`, `id` (creation order).
    pub fn prio(&self, node: &RawNode) -> Prio {
        let id = node.id;
        let la = clamp_u64(node.last_access);
        match self {
            // `last_access_time`
            Self::Lru => Prio { a: la, b: 0, c: id },
            // `(hit_count, last_access_time)`
            Self::Lfu => Prio {
                a: clamp_u64(node.hit_count),
                b: la,
                c: id,
            },
            // `creation_time` — node id is creation order.
            Self::Fifo => Prio { a: 0, b: 0, c: id },
            // `-last_access_time`
            Self::Mru => Prio {
                a: i64::MAX - la,
                b: 0,
                c: id,
            },
            // `-creation_time`
            Self::Filo => Prio {
                a: i64::MAX - (id as i64),
                b: 0,
                c: la as u32,
            },
            // `(priority, last_access_time)`
            Self::Priority => Prio {
                a: node.priority as i64,
                b: la,
                c: id,
            },
            // `(0/1 segment, last_access_time)`
            Self::Slru { threshold } => Prio {
                a: i64::from(node.hit_count >= *threshold),
                b: la,
                c: id,
            },
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_all() {
        assert_eq!(EvictionPolicy::parse("lru"), Ok(EvictionPolicy::Lru));
        assert_eq!(EvictionPolicy::parse("LRU"), Ok(EvictionPolicy::Lru));
        assert_eq!(
            EvictionPolicy::parse("slru"),
            Ok(EvictionPolicy::Slru { threshold: 2 })
        );
        assert!(EvictionPolicy::parse("arc").is_err());
    }

    #[test]
    fn lru_orders_by_last_access_then_id() {
        let p = EvictionPolicy::Lru;
        let old = RawNode::test_node(1, 5, 0, 0);
        let new = RawNode::test_node(2, 9, 0, 0);
        let tie = RawNode::test_node(3, 5, 0, 0);
        let a = p.prio(&old);
        let b = p.prio(&new);
        let c = p.prio(&tie);
        assert!(a < b);
        // same last_access, different id -> id tie-break
        assert!(c > a);
    }

    #[test]
    fn lfu_orders_by_hits_then_age() {
        let p = EvictionPolicy::Lfu;
        let hot = RawNode::test_node(1, 1, 10, 0);
        let cold = RawNode::test_node(2, 1, 0, 0);
        assert!(p.prio(&cold) < p.prio(&hot));
    }

    #[test]
    fn slru_probation_first() {
        let p = EvictionPolicy::Slru { threshold: 2 };
        let probation = RawNode::test_node(1, 1, 1, 0);
        let protected = RawNode::test_node(2, 1, 2, 0);
        assert!(p.prio(&probation) < p.prio(&protected));
    }
}
