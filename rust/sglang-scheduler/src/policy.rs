//! Port of `SchedulePolicy.calc_priority` (`schedule_policy.py`) — the
//! waiting-queue ordering step that runs every iteration.
//!
//! LPM scoring (`num_matched_prefix_tokens`) and the in-batch dedup set are
//! computed by the caller and passed in: in shadow mode Python computes
//! them against its own tree (the planner stays token-free); in core mode
//! the core computes them against the Rust tree. The ordering + degradation
//! rules here are bit-identical to the Python statics.

use crate::types::{Config, PlanReq, Policy};

/// Port of `_determine_active_policy`: LPM turns off the expensive prefix
/// matching/sorting when the queue grows past the threshold.
pub fn active_policy(cfg: &Config, waiting: &[PlanReq]) -> Policy {
    let p = cfg.active_policy();
    if p == Policy::Lpm && waiting.len() as u32 > cfg.lpm_queue_degrade_at {
        Policy::Fcfs
    } else {
        p
    }
}

/// Port of `calc_priority`: returns the admission order as waiting-queue
/// indices. `scores` are `num_matched_prefix_tokens` per waiting req
/// (meaningful for LPM); `deprioritized` is the in-batch dedup set
/// (LPM only). `iter` drives the deterministic random shuffle.
///
/// For `DfsWeight` the caller must use [`order_dfs`] with node handles —
/// this function degrades it to the arrival-stable order (shadow mode has
/// no tree to walk).
pub fn order_waiting(
    cfg: &Config,
    waiting: &[PlanReq],
    running: &[PlanReq],
    scores: &[u32],
    deprioritized: &[bool],
    iter: u64,
) -> Vec<u32> {
    let n = waiting.len();
    let policy = active_policy(cfg, waiting);
    match policy {
        Policy::Fcfs => order_fcfs(cfg, waiting),
        Policy::Lpm => order_lpm(n, scores, deprioritized),
        Policy::Lof => order_lof(cfg, waiting),
        Policy::Random => order_random(cfg.random_seed, iter, n),
        Policy::RoutingKey => order_routing_key(waiting, running),
        // No tree available to weight: keep the (priority, arrival) order.
        Policy::DfsWeight => order_fcfs(cfg, waiting),
    }
}

/// Port of `_sort_by_priority_and_fcfs`: `(priority * sign, arrival)`,
/// stable; a no-op without priority scheduling (Python also does not sort,
/// so the queue keeps its current order).
fn order_fcfs(cfg: &Config, waiting: &[PlanReq]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..waiting.len() as u32).collect();
    if cfg.priority_scheduling {
        let sign = cfg.priority_sign();
        order.sort_by_key(|&i| {
            (
                waiting[i as usize].priority * sign,
                waiting[i as usize].arrival_seq,
            )
        });
    }
    order
}

/// Port of `_sort_by_longest_prefix`: stable sort by
/// `-num_matched_prefix_tokens`, deprioritized reqs to the tail.
fn order_lpm(n: usize, scores: &[u32], deprioritized: &[bool]) -> Vec<u32> {
    debug_assert_eq!(n, scores.len());
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|&i| {
        if deprioritized[i as usize] {
            i64::MAX
        } else {
            -(scores[i as usize] as i64)
        }
    });
    order
}

/// Port of `_sort_by_longest_output`: `(priority * sign, -max_new)` or
/// just `-max_new`, stable.
fn order_lof(cfg: &Config, waiting: &[PlanReq]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..waiting.len() as u32).collect();
    if cfg.priority_scheduling {
        let sign = cfg.priority_sign();
        order.sort_by_key(|&i| {
            (
                waiting[i as usize].priority * sign,
                -(waiting[i as usize].max_new_tokens as i64),
            )
        });
    } else {
        order.sort_by_key(|&i| -(waiting[i as usize].max_new_tokens as i64));
    }
    order
}

/// Deterministic shuffle (SplitMix64 Fisher–Yates). Python's
/// `random.shuffle` is Mersenne-Twister and not seed-parity-able across
/// the boundary, so the `random` policy is the one documented deviation
/// from plan-for-plan parity (the policy is inherently order-opaque).
fn order_random(seed: u64, iter: u64, n: usize) -> Vec<u32> {
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut state = seed.wrapping_add(iter.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut next = || -> u64 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49DB_FF1C_E165);
        z ^ (z >> 31)
    };
    for i in (1..n).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    order
}

/// Port of `_sort_by_routing_key`: running-batch routing-key frequency.
/// Stable; unknown keys and keys absent from the running batch sort after
/// the known ones.
fn order_routing_key(waiting: &[PlanReq], running: &[PlanReq]) -> Vec<u32> {
    // Counter over the running batch (routing_key == 0 means "none" and is
    // excluded, mirroring `Counter(r.routing_key for r in ... if r.routing_key)`).
    let mut counts: std::collections::HashMap<u64, i64> = std::collections::HashMap::new();
    for r in running {
        if r.routing_key != 0 {
            *counts.entry(r.routing_key).or_insert(0) += 1;
        }
    }
    let mut order: Vec<u32> = (0..waiting.len() as u32).collect();
    if !counts.is_empty() {
        order.sort_by_key(|&i| {
            let key = waiting[i as usize].routing_key;
            match counts.get(&key) {
                Some(count) => (0u8, -count, key),
                None => (1u8, 0i64, key),
            }
        });
    }
    order
}

/// Port of `_sort_by_dfs_weight` for the core engine (tree-backed).
///
/// Weights are derived from the waiting queue itself: a node's direct
/// weight is the number of waiting reqs whose `last_node` is that node,
/// and the DFS walk accumulates subtree weights post-order — exactly
/// `_calc_weight`. Ties between equal-weight siblings break on node id
/// (Python iterates the parent's children dict in insertion order, which is
/// not reproducible across languages; documented deviation, opt-in policy
/// only). Waiting reqs hold their prefix lock, so their `last_node` is
/// always reachable from the root.
///
/// Returns waiting-queue indices in DFS-priority order.
pub fn order_dfs(waiting: &[PlanReq], children_of: &dyn Fn(u32) -> Vec<u32>) -> Vec<u32> {
    use std::collections::HashMap;

    let mut last_node_to_reqs: HashMap<u32, Vec<u32>> = HashMap::new();
    for (i, r) in waiting.iter().enumerate() {
        last_node_to_reqs.entry(r.last_node).or_default().push(i as u32);
    }

    // Deterministic child order: sorted node ids.
    let sorted_children = |node: u32| -> Vec<u32> {
        let mut c = children_of(node);
        c.sort();
        c
    };

    // Post-order subtree weights (iterative; the Python walk is recursive).
    let mut weight: HashMap<u32, i64> = HashMap::new();
    let mut frames: Vec<(u32, Vec<u32>, usize)> = vec![(0, sorted_children(0), 0)];
    while let Some(f) = frames.last_mut() {
        let node = f.0;
        if f.2 < f.1.len() {
            let child = f.1[f.2];
            f.2 += 1;
            frames.push((child, sorted_children(child), 0));
        } else {
            let mut w = last_node_to_reqs.get(&node).map(|reqs| reqs.len() as i64).unwrap_or(0);
            for &child in &f.1 {
                w += weight.get(&child).copied().unwrap_or(0);
            }
            weight.insert(node, w);
            frames.pop();
        }
    }

    // DFS: children by -weight (ties by node id), node's own reqs after all
    // descendants.
    let weighted_children = |node: u32| -> Vec<u32> {
        let mut c = sorted_children(node);
        c.sort_by_key(|&child| -weight.get(&child).copied().unwrap_or(0));
        c
    };
    let mut out: Vec<u32> = Vec::with_capacity(waiting.len());
    let mut frames: Vec<(u32, Vec<u32>, usize)> = vec![(0, weighted_children(0), 0)];
    while let Some(f) = frames.last_mut() {
        if f.2 < f.1.len() {
            let child = f.1[f.2];
            f.2 += 1;
            frames.push((child, weighted_children(child), 0));
        } else {
            if let Some(reqs) = last_node_to_reqs.get(&f.0) {
                out.extend(reqs.iter().copied());
            }
            frames.pop();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(prio: i32, arrival: u64, max_new: u32, key: u64) -> PlanReq {
        PlanReq {
            priority: prio,
            arrival_seq: arrival,
            max_new_tokens: max_new,
            routing_key: key,
            ..Default::default()
        }
    }

    fn cfg(policy: Policy, prio: bool) -> Config {
        Config {
            policy,
            priority_scheduling: prio,
            ..Default::default()
        }
    }

    #[test]
    fn fcfs_no_priority_keeps_order() {
        let w = vec![req(0, 3, 1, 0), req(0, 1, 1, 0), req(0, 2, 1, 0)];
        assert_eq!(order_waiting(&cfg(Policy::Fcfs, false), &w, &[], &[], &[], 0), vec![0, 1, 2]);
    }

    #[test]
    fn fcfs_priority_then_arrival() {
        // Low values first: lower priority scheduled earlier (sign +1).
        let w = vec![
            req(5, 1, 1, 0),
            req(1, 9, 1, 0),
            req(1, 2, 1, 0),
            req(5, 0, 1, 0),
        ];
        let c = Config {
            low_priority_values_first: true,
            ..cfg(Policy::Fcfs, true)
        };
        // Keys: 0:(5,1) 1:(1,9) 2:(1,2) 3:(5,0) -> (1,2) (1,9) (5,0) (5,1).
        assert_eq!(
            order_waiting(&c, &w, &[], &[], &[], 0),
            vec![2, 1, 3, 0]
        );
    }

    #[test]
    fn lpm_scores_and_deprioritize() {
        let w = vec![req(0, 0, 1, 0); 4];
        let scores = [100u32, 50, 50, 200];
        let deprio = [false, true, false, false];
        // Order: 3 (200), 0 (100), 2 (50, not deprio), 1 (50, deprio -> tail).
        assert_eq!(
            order_waiting(&cfg(Policy::Lpm, false), &w, &[], &scores, &deprio, 0),
            vec![3, 0, 2, 1]
        );
    }

    #[test]
    fn lpm_degrades_to_fcfs_past_128() {
        let w = vec![req(0, 0, 1, 0); 129];
        let scores: Vec<u32> = (0..129).map(|i| i * 1000).collect(); // would change order
        let c = cfg(Policy::Lpm, false);
        let got = order_waiting(&c, &w, &[], &scores, &[false; 129], 0);
        assert_eq!(got, (0..129u32).collect::<Vec<_>>()); // FCFS: unchanged
    }

    #[test]
    fn lof_and_routing_key() {
        let w = vec![req(0, 0, 5, 7), req(0, 0, 9, 7), req(0, 0, 7, 0), req(0, 0, 8, 9)];
        assert_eq!(order_lof(&cfg(Policy::Lof, false), &w), vec![1, 3, 2, 0]);

        let running = vec![
            req(0, 0, 0, 7),
            req(0, 0, 0, 7),
            req(0, 0, 0, 9),
            req(0, 0, 0, 0), // no key: excluded from the counter
        ];
        // 7 has count 2, 9 has count 1: (0,-2,7) < (0,-1,9) < unknown.
        assert_eq!(
            order_routing_key(&w, &running),
            vec![0, 1, 3, 2]
        );
    }

    #[test]
    fn random_is_deterministic_and_a_permutation() {
        let a = order_random(42, 7, 32);
        let b = order_random(42, 7, 32);
        let c = order_random(42, 8, 32);
        assert_eq!(a, b);
        assert_ne!(a, c);
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(sorted, (0..32u32).collect::<Vec<_>>());
    }

    #[test]
    fn dfs_weight_orders_by_subtree_weight() {
        // Tree: 0 -> 1, 0 -> 2; 1 -> 3. Waiting: [node3, node1, node2].
        // Direct weights: node1=1, node2=1, node3=1; accumulated:
        // node1 = 2 (self + node3 subtree), node2 = 1.
        // DFS: child 1 (weight 2) before child 2 (weight 1); within child 1,
        // descendant node3's reqs come before node1's own.
        let waiting: Vec<PlanReq> = vec![
            PlanReq {
                last_node: 3,
                ..Default::default()
            },
            PlanReq {
                last_node: 1,
                ..Default::default()
            },
            PlanReq {
                last_node: 2,
                ..Default::default()
            },
        ];
        let children = |node: u32| -> Vec<u32> {
            match node {
                0 => vec![1, 2],
                1 => vec![3],
                _ => vec![],
            }
        };
        assert_eq!(order_dfs(&waiting, &children), vec![0, 1, 2]);

        // A root-level req (empty prefix) comes after all subtrees.
        let waiting = vec![
            waiting[0],
            PlanReq {
                last_node: 0,
                ..Default::default()
            },
            waiting[1],
            waiting[2],
        ];
        assert_eq!(order_dfs(&waiting, &children), vec![0, 2, 3, 1]);
    }
}
