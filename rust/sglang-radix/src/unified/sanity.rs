//! Tree invariant verification — port of `UnifiedTreeCore.sanity_check`
//! (five parts: structure, node state machine + leaf qualification,
//! tracking structures, size accounting, ongoing operations).
//!
//! Python raises `AssertionError` with the collected messages; the Rust core
//! returns the message list (empty = invariants hold).

use std::collections::HashSet;

use crate::unified::tree::{PARENT_NONE, UnifiedRadixTree};
use crate::unified::{CT_BASE, CT_FULL, CT_MAMBA, CT_SWA, NUM_CT};

fn collect_all_nodes(t: &UnifiedRadixTree) -> Vec<u32> {
    let mut out = Vec::new();
    let mut stack = vec![t.root];
    while let Some(node) = stack.pop() {
        out.push(node);
        stack.extend(t.nodes[node as usize].children.values());
    }
    out
}

fn check_lru_linked_list(
    t: &UnifiedRadixTree,
    slot: usize,
    ct: u8,
    label: &str,
    errors: &mut Vec<String>,
) {
    let mut visited: HashSet<u32> = HashSet::new();
    let mut x = t.nodes[t.lru_head[slot] as usize].lru_next[slot];
    let mut prev = t.lru_head[slot];
    while x != t.lru_tail[slot] {
        if t.nodes[x as usize].lru_prev[slot] != prev {
            errors.push(format!("[{label}][ct{ct}] broken prev at node {x}"));
        }
        if !t.lru_in(slot, x) {
            errors.push(format!("[{label}][ct{ct}] node {x} in list not in_lru"));
        }
        if !visited.insert(x) {
            errors.push(format!("[{label}][ct{ct}] cycle at node {x}"));
            break;
        }
        prev = x;
        x = t.nodes[x as usize].lru_next[slot];
        // Cycle guard: the list can only be longer than the arena.
        if visited.len() as u64 > t.nodes.len() as u64 {
            errors.push(format!("[{label}][ct{ct}] unbounded walk"));
            break;
        }
    }
    if x == t.lru_tail[slot] && visited.len() as u64 > t.nodes.len() as u64 {
        // unreachable; keep borrowck simple
    }
}

/// `sanity_check` — returns the violation messages (empty when clean).
pub fn sanity_check(
    t: &UnifiedRadixTree,
    ongoing_write_through: &[(i64, u32)],
    ongoing_load_back: &[(i64, u32)],
) -> Vec<String> {
    let mut errors: Vec<String> = Vec::new();
    let all_nodes = collect_all_nodes(t);
    let all_node_set: HashSet<u32> = all_nodes.iter().copied().collect();
    let root = t.root;

    // ── PART 1: Tree structure ──
    {
        let r = &t.nodes[root as usize];
        if r.value[CT_BASE as usize].is_none() {
            errors.push("[Root] root missing Full device value".into());
        }
        if r.lock_ref[CT_BASE as usize] <= 0 {
            errors.push(format!(
                "[Root] root Full lock_ref={}",
                r.lock_ref[CT_BASE as usize]
            ));
        }
        if r.parent != PARENT_NONE {
            errors.push("[Root] root has a parent pointer".into());
        }
    }
    for &node in &all_nodes {
        let n = &t.nodes[node as usize];
        for &child in n.children.values() {
            let c = &t.nodes[child as usize];
            if c.parent != node {
                let pid = if c.parent == PARENT_NONE {
                    "None".to_string()
                } else {
                    c.parent.to_string()
                };
                errors.push(format!("[Tree] child {child} parent={pid}, expected {node}"));
            }
            if c.key.is_empty() {
                errors.push(format!("[Tree] node {child} has no key"));
            }
        }
    }

    // ── PART 2: per-node state machine + leaf qualification ──
    let mut expected_dev_leaves: HashSet<u32> = HashSet::new();
    let mut expected_hst_leaves: HashSet<u32> = HashSet::new();
    let mut expected_duplicates: HashSet<u32> = HashSet::new();

    for &node in &all_nodes {
        if node == root {
            continue;
        }
        let nid = node;
        let n = &t.nodes[node as usize];
        let full_dev = n.value[CT_BASE as usize].is_some();
        let full_hst = n.host_value[CT_BASE as usize].is_some();

        for &ct in t.active_cts().iter() {
            if ct == CT_FULL {
                continue;
            }
            if n.value[ct as usize].is_some() && !full_dev {
                errors.push(format!("node {nid} ct{ct} device present but Full.value=None"));
            }
            if n.host_value[ct as usize].is_some()
                && !full_hst
                && !(t.is_write_back() && full_dev)
            {
                errors.push(format!(
                    "node {nid} ct{ct} host present but Full.host_value=None"
                ));
            }
        }

        if !full_dev && !full_hst {
            errors.push(format!("node {nid} dead: no Full device and no Full host"));
        }

        if n.parent != PARENT_NONE && n.parent != root {
            let p = &t.nodes[n.parent as usize];
            let p_dev = p.value[CT_BASE as usize].is_some();
            let p_hst = p.host_value[CT_BASE as usize].is_some();
            if full_dev && !p_dev {
                errors.push(format!(
                    "node {nid} device present but parent {} evicted",
                    n.parent
                ));
            }
            if full_hst && !p_hst && !t.is_write_back() {
                errors.push(format!(
                    "node {nid} backed up but parent {} not backed up",
                    n.parent
                ));
            }
        }

        let fl = n.lock_ref[CT_BASE as usize];
        for &ct in t.active_cts().iter() {
            if n.lock_ref[ct as usize] < 0 {
                errors.push(format!(
                    "node {nid} ct{ct} lock_ref={}",
                    n.lock_ref[ct as usize]
                ));
            }
            if n.host_lock_ref[ct as usize] < 0 {
                errors.push(format!(
                    "node {nid} ct{ct} host_lock_ref={}",
                    n.host_lock_ref[ct as usize]
                ));
            }
            if ct != CT_FULL && fl < n.lock_ref[ct as usize] {
                errors.push(format!(
                    "node {nid} full_lock={fl} < ct{ct}_lock={}",
                    n.lock_ref[ct as usize]
                ));
            }
            if n.value[ct as usize].is_none() && n.lock_ref[ct as usize] > 0 {
                errors.push(format!(
                    "node {nid} ct{ct} evicted but lock_ref={}",
                    n.lock_ref[ct as usize]
                ));
            }
        }

        if t.is_device_leaf(node) {
            expected_dev_leaves.insert(node);
        }
        if t.is_host_leaf(node) {
            expected_hst_leaves.insert(node);
        }
        if t.is_settled_duplicate(node) {
            expected_duplicates.insert(node);
        }
    }

    // ── PART 3: tracking structures ──
    let d_extra: Vec<u32> = t
        .d_leaves
        .iter()
        .filter(|id| !expected_dev_leaves.contains(id))
        .copied()
        .take(5)
        .collect();
    let d_missing: Vec<u32> = expected_dev_leaves
        .iter()
        .filter(|id| !t.d_leaves.contains(id))
        .copied()
        .take(5)
        .collect();
    if !d_extra.is_empty() {
        errors.push(format!("D-leaf extra: {d_extra:?}"));
    }
    if !d_missing.is_empty() {
        errors.push(format!("D-leaf missing: {d_missing:?}"));
    }

    let h_extra: Vec<u32> = t
        .h_leaves
        .iter()
        .filter(|id| !expected_hst_leaves.contains(id))
        .copied()
        .take(5)
        .collect();
    let h_missing: Vec<u32> = expected_hst_leaves
        .iter()
        .filter(|id| !t.h_leaves.contains(id))
        .copied()
        .take(5)
        .collect();
    if !h_extra.is_empty() {
        errors.push(format!("H-leaf extra: {h_extra:?}"));
    }
    if !h_missing.is_empty() {
        errors.push(format!("H-leaf missing: {h_missing:?}"));
    }

    // Lazy deregistration: settled duplicates must be tracked; no ghosts.
    let dup_missing: Vec<u32> = expected_duplicates
        .iter()
        .filter(|id| !t.dup_set.contains(id))
        .copied()
        .take(5)
        .collect();
    if !dup_missing.is_empty() {
        errors.push(format!("Duplicate missing: {dup_missing:?}"));
    }
    let ghost: Vec<u32> = t
        .dup_set
        .iter()
        .filter(|id| !all_node_set.contains(id))
        .copied()
        .take(5)
        .collect();
    if !ghost.is_empty() {
        errors.push(format!("Duplicate ghosts: {ghost:?}"));
    }

    let overlap: Vec<u32> = t
        .d_leaves
        .iter()
        .filter(|id| t.h_leaves.contains(id))
        .copied()
        .take(5)
        .collect();
    if !overlap.is_empty() {
        errors.push(format!("[Leaf] {} in both sets: {overlap:?}", overlap.len()));
    }

    // Stale nodes: leaf sets only contain tree-reachable nodes.
    let stale_d: Vec<u32> = t
        .d_leaves
        .iter()
        .filter(|id| !all_node_set.contains(id))
        .copied()
        .take(5)
        .collect();
    if !stale_d.is_empty() {
        errors.push(format!(
            "{} stale nodes in device_leaves: {stale_d:?}",
            stale_d.len()
        ));
    }
    let stale_h: Vec<u32> = t
        .h_leaves
        .iter()
        .filter(|id| !all_node_set.contains(id))
        .copied()
        .take(5)
        .collect();
    if !stale_h.is_empty() {
        errors.push(format!(
            "{} stale nodes in host_leaves: {stale_h:?}",
            stale_h.len()
        ));
    }

    // Per-component LRU tracking.
    for &ct in t.active_cts().iter() {
        let dev_slot = UnifiedRadixTree::lru_slot_public(ct, 0);
        let host_slot = UnifiedRadixTree::lru_slot_public(ct, 1);
        if ct == CT_FULL {
            if !t.lru_order(ct, 0).is_empty() {
                errors.push(format!(
                    "Full device LRU not empty: {}",
                    t.lru_order(ct, 0).len()
                ));
            }
            if !t.lru_order(ct, 1).is_empty() {
                errors.push(format!(
                    "Full host LRU not empty: {}",
                    t.lru_order(ct, 1).len()
                ));
            }
        } else {
            let tree_ids: HashSet<u32> = all_nodes
                .iter()
                .filter(|&&n| {
                    n != root && t.nodes[n as usize].value[ct as usize].is_some()
                })
                .copied()
                .collect();
            let lru_ids: HashSet<u32> = t
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.in_lru & (1u16 << dev_slot) != 0)
                .map(|(i, _)| i as u32)
                .collect();
            if tree_ids != lru_ids {
                let in_tree: Vec<u32> = tree_ids.difference(&lru_ids).copied().collect();
                let in_lru: Vec<u32> = lru_ids.difference(&tree_ids).copied().collect();
                errors.push(format!(
                    "ct{ct} device LRU: +tree={in_tree:?}, +lru={in_lru:?}"
                ));
            }
            let s3_ids: HashSet<u32> = all_nodes
                .iter()
                .filter(|&&n| {
                    n != root
                        && t.nodes[n as usize].value[ct as usize].is_none()
                        && t.nodes[n as usize].host_value[ct as usize].is_some()
                })
                .copied()
                .collect();
            let host_lru_ids: HashSet<u32> = t
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.in_lru & (1u16 << host_slot) != 0)
                .map(|(i, _)| i as u32)
                .collect();
            if s3_ids != host_lru_ids {
                let in_tree: Vec<u32> = s3_ids.difference(&host_lru_ids).copied().collect();
                let in_lru: Vec<u32> = host_lru_ids.difference(&s3_ids).copied().collect();
                errors.push(format!(
                    "ct{ct} host LRU: +S3={in_tree:?}, +lru={in_lru:?}"
                ));
            }
            let both: Vec<u32> = lru_ids.intersection(&host_lru_ids).copied().collect();
            if !both.is_empty() {
                errors.push(format!("ct{ct} in both device and host LRU: {both:?}"));
            }
            check_lru_linked_list(t, dev_slot, ct, "device", &mut errors);
            check_lru_linked_list(t, host_slot, ct, "host", &mut errors);
        }
    }

    // ── PART 4: size accounting ──
    for &ct in t.active_cts().iter() {
        let mut evictable = 0i64;
        let mut protected = 0i64;
        for &n in &all_nodes {
            if n == root {
                continue;
            }
            let cd = &t.nodes[n as usize];
            if let Some(v) = &cd.value[ct as usize] {
                let toks = v.len() as i64;
                if cd.lock_ref[ct as usize] > 0 {
                    protected += toks;
                } else {
                    evictable += toks;
                }
            }
        }
        if t.evictable_size[ct as usize] != evictable {
            errors.push(format!(
                "[Size] ct{ct} evictable={} != recomputed={evictable}",
                t.evictable_size[ct as usize]
            ));
        }
        if t.protected_size[ct as usize] != protected {
            errors.push(format!(
                "[Size] ct{ct} protected={} != recomputed={protected}",
                t.protected_size[ct as usize]
            ));
        }
    }

    // ── PART 5: ongoing operations ──
    for &(nid, node_id) in ongoing_write_through {
        match t.nodes.get(node_id as usize) {
            Some(n) if all_node_set.contains(&node_id) => {
                if n.lock_ref[CT_BASE as usize] <= 0 {
                    errors.push(format!(
                        "[Ongoing] write_through node {nid} lock_ref={}",
                        n.lock_ref[CT_BASE as usize]
                    ));
                }
            }
            _ => {
                errors.push(format!("[Ongoing] write_through node {nid} not in tree"));
            }
        }
    }
    for &(nid, node_id) in ongoing_load_back {
        match t.nodes.get(node_id as usize) {
            Some(n) if all_node_set.contains(&node_id) => {
                if n.lock_ref[CT_BASE as usize] <= 0 {
                    errors.push(format!(
                        "[Ongoing] load_back node {nid} lock_ref={}",
                        n.lock_ref[CT_BASE as usize]
                    ));
                }
            }
            _ => {
                errors.push(format!("[Ongoing] load_back node {nid} not in tree"));
            }
        }
    }
    let ongoing_load_ids: HashSet<u32> =
        ongoing_load_back.iter().map(|&(_, n)| n).collect();
    for &node in &all_nodes {
        if let Some(anchor) = t.nodes[node as usize].lb_pending
            && !ongoing_load_ids.contains(&anchor)
        {
            errors.push(format!(
                "[Ongoing] node {node} load_back_pending_id={anchor} has no live load-back"
            ));
        }
    }

    // Keep the NUM_CT/CT_* imports alive for parity with the Python CTS.
    let _ = (CT_SWA, CT_MAMBA, NUM_CT);
    errors
}
