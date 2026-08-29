//! Lock ref counting — port of the tree-level `inc_lock_ref` / `dec_lock_ref`
//! / `dec_swa_lock_only` / `inc_host_lock_ref` / `dec_host_lock_ref` plus the
//! per-component `acquire_component_lock` / `release_component_lock` /
//! `release_window_lock` hooks of `full_component.py`, `swa_component.py`,
//! `mamba_component.py`.

use crate::unified::tree::UnifiedRadixTree;
use crate::unified::{UDecLockParams, UIncLockResult, UEvictOutcome, CT_BASE, CT_FULL, CT_MAMBA, CT_SWA};

impl UnifiedRadixTree {
    /// `inc_lock_ref` (device path-locks).
    pub fn inc_lock_ref(&mut self, node: u32, skip_lock_components: &[u8]) -> UIncLockResult {
        let mut result = UIncLockResult::default();
        for &ct in self.active_cts().iter() {
            if skip_lock_components.contains(&ct) {
                if node != self.root {
                    Self::add_skip(&mut result, ct, node);
                }
                continue;
            }
            match ct {
                CT_FULL => self.full_acquire_lock(node, &mut result, false),
                CT_SWA => self.swa_acquire_lock(node, &mut result, false),
                CT_MAMBA => self.mamba_acquire_lock(node, &mut result, false),
                _ => {}
            }
        }
        self.update_leaf_sets(node);
        result
    }

    /// `dec_lock_ref`.
    pub fn dec_lock_ref(&mut self, node: u32, params: &UDecLockParams, skip_swa: bool) {
        for &ct in self.active_cts().iter() {
            if skip_swa && ct == CT_SWA {
                continue;
            }
            match ct {
                CT_FULL => self.full_release_lock(node, params, false),
                CT_SWA => self.swa_release_lock(node, params, false),
                CT_MAMBA => self.mamba_release_lock(node, params, false),
                _ => {}
            }
        }
        self.update_leaf_sets(node);
    }

    /// `dec_swa_lock_only`.
    pub fn dec_swa_lock_only(
        &mut self,
        node: u32,
        swa_uuid_for_lock: Option<i64>,
        skip_lock_node_ids: &[(u8, Vec<u32>)],
    ) -> UEvictOutcome {
        let mut result = UEvictOutcome::default();
        if !self.has_swa {
            return result;
        }
        self.swa_release_window_lock(node, swa_uuid_for_lock, &mut result);

        // Drop strictly-lower-priority locks (Mamba: 0 < 1).
        let swa_priority = 1; // SWA internal priority
        let dec_params = UDecLockParams {
            swa_uuid_for_lock,
            swa_uuid_for_host_lock: None,
            skip_lock_node_ids: skip_lock_node_ids.to_vec(),
        };
        if self.has_mamba && 0 < swa_priority {
            self.mamba_release_lock(node, &dec_params, false);
        }
        result
    }

    /// `inc_host_lock_ref`.
    pub fn inc_host_lock_ref(&mut self, node: u32) -> UIncLockResult {
        let mut result = UIncLockResult::default();
        for &ct in self.active_cts().iter() {
            match ct {
                CT_FULL => self.full_acquire_lock(node, &mut result, true),
                CT_SWA => self.swa_acquire_lock(node, &mut result, true),
                CT_MAMBA => self.mamba_acquire_lock(node, &mut result, true),
                _ => {}
            }
        }
        self.update_leaf_sets(node);
        result
    }

    /// `dec_host_lock_ref`.
    pub fn dec_host_lock_ref(&mut self, node: u32, params: &UDecLockParams) {
        for &ct in self.active_cts().iter() {
            match ct {
                CT_FULL => self.full_release_lock(node, params, true),
                CT_SWA => self.swa_release_lock(node, params, true),
                CT_MAMBA => self.mamba_release_lock(node, params, true),
                _ => {}
            }
        }
        self.update_leaf_sets(node);
    }

    fn add_skip(result: &mut UIncLockResult, ct: u8, node: u32) {
        if let Some(entry) = result.skip_lock_node_ids.iter_mut().find(|(c, _)| *c == ct) {
            entry.1.push(node);
        } else {
            result.skip_lock_node_ids.push((ct, vec![node]));
        }
    }

    // ---- FULL ----

    fn full_acquire_lock(&mut self, node: u32, result: &mut UIncLockResult, lock_host: bool) {
        if lock_host {
            // Only the last host node needs to be protected.
            let cd = &self.nodes[node as usize];
            if cd.host_value[CT_BASE as usize].is_none() && !self.cfg.is_write_back {
                return;
            }
            self.nodes[node as usize].host_lock_ref[CT_BASE as usize] += 1;
            self.update_leaf_sets(node);
            return;
        }
        let mut cur = node;
        // Skip the bottom evicted segment.
        while cur != self.root && self.nodes[cur as usize].value[CT_BASE as usize].is_none() {
            Self::add_skip(result, CT_FULL, cur);
            cur = self.nodes[cur as usize].parent;
        }
        let mut delta = 0i64;
        while cur != self.root {
            let cd = &self.nodes[cur as usize];
            debug_assert!(
                cd.value[CT_BASE as usize].is_some(),
                "FULL invariant broken: evicted ancestor above device-on segment"
            );
            if cd.lock_ref[CT_BASE as usize] == 0 {
                let key_len = cd
                    .value[CT_BASE as usize]
                    .as_ref()
                    .map(|v| v.len() as i64)
                    .unwrap_or(0);
                self.evictable_size[CT_BASE as usize] -= key_len;
                self.protected_size[CT_BASE as usize] += key_len;
                delta += key_len;
            }
            self.nodes[cur as usize].lock_ref[CT_BASE as usize] += 1;
            self.d_leaves.remove(&cur);
            cur = self.nodes[cur as usize].parent;
        }
        result.delta = delta;
    }

    fn full_release_lock(&mut self, node: u32, params: &UDecLockParams, lock_host: bool) {
        if lock_host {
            let cd = &self.nodes[node as usize];
            if cd.host_lock_ref[CT_BASE as usize] == 0 {
                return;
            }
            if cd.host_value[CT_BASE as usize].is_none() && !self.cfg.is_write_back {
                return;
            }
            self.nodes[node as usize].host_lock_ref[CT_BASE as usize] -= 1;
            self.update_leaf_sets(node);
            return;
        }
        let skip = params.skip_ids(CT_FULL);
        let mut cur = node;
        while cur != self.root {
            if skip.contains(&cur) {
                cur = self.nodes[cur as usize].parent;
                continue;
            }
            let cd = &self.nodes[cur as usize];
            debug_assert!(
                cd.value[CT_BASE as usize].is_some(),
                "FULL release: evicted node on lock path"
            );
            debug_assert!(cd.lock_ref[CT_BASE as usize] > 0);
            if cd.lock_ref[CT_BASE as usize] == 1 {
                let key_len = cd
                    .value[CT_BASE as usize]
                    .as_ref()
                    .map(|v| v.len() as i64)
                    .unwrap_or(0);
                self.evictable_size[CT_BASE as usize] += key_len;
                self.protected_size[CT_BASE as usize] -= key_len;
            }
            self.nodes[cur as usize].lock_ref[CT_BASE as usize] -= 1;
            if self.nodes[cur as usize].lock_ref[CT_BASE as usize] == 0 {
                self.update_leaf_sets(cur);
            }
            cur = self.nodes[cur as usize].parent;
        }
    }

    // ---- SWA ----

    fn swa_acquire_lock(&mut self, node: u32, result: &mut UIncLockResult, lock_host: bool) {
        let window = self.cfg.sliding_window_size;
        let mut swa_lock_size = 0i64;
        let mut uuid: Option<i64> = None;
        let dev_slot = Self::lru_slot_public(CT_SWA, 0);
        let host_slot = Self::lru_slot_public(CT_SWA, 1);
        let mut cur = node;
        while cur != self.root && swa_lock_size < window {
            let value_len: i64;
            {
                let cd = &self.nodes[cur as usize];
                let present = if lock_host {
                    cd.host_value[CT_SWA as usize].is_some()
                } else {
                    cd.value[CT_SWA as usize].is_some()
                };
                if !present {
                    Self::add_skip(result, CT_SWA, cur);
                    cur = cd.parent;
                    continue;
                }
                value_len = if lock_host {
                    cd.host_value[CT_SWA as usize].as_ref().unwrap().len() as i64
                } else {
                    cd.value[CT_SWA as usize].as_ref().unwrap().len() as i64
                };
            }
            if lock_host {
                let first = self.nodes[cur as usize].host_lock_ref[CT_SWA as usize] == 0;
                if first && self.lru_in(host_slot, cur) {
                    self.lru_remove(host_slot, cur);
                }
                self.nodes[cur as usize].host_lock_ref[CT_SWA as usize] += 1;
            } else {
                let first = self.nodes[cur as usize].lock_ref[CT_SWA as usize] == 0;
                if first {
                    let key_len = self.key_len(&self.nodes[cur as usize].key);
                    self.evictable_size[CT_SWA as usize] -= key_len;
                    self.protected_size[CT_SWA as usize] += key_len;
                }
                self.nodes[cur as usize].lock_ref[CT_SWA as usize] += 1;
            }
            swa_lock_size += value_len;
            if swa_lock_size >= window {
                let have = if lock_host {
                    self.nodes[cur as usize].swa_host_uuid.is_some()
                } else {
                    self.nodes[cur as usize].swa_uuid.is_some()
                };
                if !have {
                    let u = self.next_uuid();
                    if lock_host {
                        self.nodes[cur as usize].swa_host_uuid = Some(u);
                    } else {
                        self.nodes[cur as usize].swa_uuid = Some(u);
                    }
                }
                uuid = if lock_host {
                    self.nodes[cur as usize].swa_host_uuid
                } else {
                    self.nodes[cur as usize].swa_uuid
                };
            }
            cur = self.nodes[cur as usize].parent;
        }
        if lock_host {
            result.swa_uuid_for_host_lock = uuid;
        } else {
            result.swa_uuid_for_lock = uuid;
        }
        let _ = dev_slot;
    }

    fn swa_release_lock(&mut self, node: u32, params: &UDecLockParams, lock_host: bool) {
        let uuid = if lock_host {
            params.swa_uuid_for_host_lock
        } else {
            params.swa_uuid_for_lock
        };
        let skip = params.skip_ids(CT_SWA);
        let host_slot = Self::lru_slot_public(CT_SWA, 1);
        let mut cur = node;
        let mut dec_swa = true;
        while cur != self.root && dec_swa {
            if skip.contains(&cur) {
                cur = self.nodes[cur as usize].parent;
                continue;
            }
            let ref_now = if lock_host {
                self.nodes[cur as usize].host_lock_ref[CT_SWA as usize]
            } else {
                self.nodes[cur as usize].lock_ref[CT_SWA as usize]
            };
            if ref_now == 0 {
                cur = self.nodes[cur as usize].parent;
                continue;
            }
            if ref_now == 1 {
                if lock_host {
                    let n = &self.nodes[cur as usize];
                    if n.value[CT_SWA as usize].is_none()
                        && n.host_value[CT_SWA as usize].is_some()
                        && !self.lru_in(host_slot, cur)
                    {
                        self.lru_insert_mru(host_slot, cur);
                    }
                } else {
                    let key_len = self
                        .nodes[cur as usize]
                        .value[CT_SWA as usize]
                        .as_ref()
                        .map(|v| v.len() as i64)
                        .unwrap_or(0);
                    self.evictable_size[CT_SWA as usize] += key_len;
                    self.protected_size[CT_SWA as usize] -= key_len;
                }
            }
            if lock_host {
                self.nodes[cur as usize].host_lock_ref[CT_SWA as usize] = ref_now - 1;
            } else {
                self.nodes[cur as usize].lock_ref[CT_SWA as usize] = ref_now - 1;
            }
            let matches = if lock_host {
                self.nodes[cur as usize].swa_host_uuid
            } else {
                self.nodes[cur as usize].swa_uuid
            };
            if uuid.is_some() && matches == uuid {
                dec_swa = false;
            }
            cur = self.nodes[cur as usize].parent;
        }
    }

    /// `SWAComponent.release_window_lock` (device).
    fn swa_release_window_lock(
        &mut self,
        node: u32,
        swa_uuid_for_lock: Option<i64>,
        result: &mut UEvictOutcome,
    ) {
        let mut cur = node;
        while cur != self.root {
            let (val_present, lock_ref) = {
                let n = &self.nodes[cur as usize];
                (
                    n.value[CT_SWA as usize].is_some(),
                    n.lock_ref[CT_SWA as usize],
                )
            };
            let uuid = self.nodes[cur as usize].swa_uuid;
            if !val_present || lock_ref == 0 {
                if swa_uuid_for_lock == uuid && swa_uuid_for_lock.is_some() {
                    break;
                }
                cur = self.nodes[cur as usize].parent;
                continue;
            }
            self.nodes[cur as usize].lock_ref[CT_SWA as usize] -= 1;
            if self.nodes[cur as usize].lock_ref[CT_SWA as usize] == 0 {
                let key_len = self.key_len(&self.nodes[cur as usize].key);
                self.protected_size[CT_SWA as usize] -= key_len;
                self.evictable_size[CT_SWA as usize] += key_len;
                if self.is_device_leaf(cur) {
                    self.evict_component_and_detach_lru(
                        cur,
                        CT_SWA,
                        crate::unified::Layer::Device,
                        None,
                        &mut result.device_frees,
                        &mut result.host_frees,
                    );
                }
            }
            if swa_uuid_for_lock == uuid && swa_uuid_for_lock.is_some() {
                break;
            }
            cur = self.nodes[cur as usize].parent;
        }
    }

    // ---- Mamba ----

    fn mamba_acquire_lock(&mut self, node: u32, result: &mut UIncLockResult, lock_host: bool) {
        if node == self.root {
            return;
        }
        let value_present = if lock_host {
            self.nodes[node as usize].host_value[CT_MAMBA as usize].is_some()
        } else {
            self.nodes[node as usize].value[CT_MAMBA as usize].is_some()
        };
        if !value_present {
            Self::add_skip(result, CT_MAMBA, node);
            return;
        }
        if lock_host {
            let host_slot = Self::lru_slot_public(CT_MAMBA, 1);
            if self.nodes[node as usize].host_lock_ref[CT_MAMBA as usize] == 0
                && self.lru_in(host_slot, node)
            {
                self.lru_remove(host_slot, node);
            }
            self.nodes[node as usize].host_lock_ref[CT_MAMBA as usize] += 1;
        } else {
            let dev_slot = Self::lru_slot_public(CT_MAMBA, 0);
            let _ = dev_slot;
            if self.nodes[node as usize].lock_ref[CT_MAMBA as usize] == 0 {
                let vlen = self.nodes[node as usize]
                    .value[CT_MAMBA as usize]
                    .as_ref()
                    .unwrap()
                    .len() as i64;
                self.evictable_size[CT_MAMBA as usize] -= vlen;
                self.protected_size[CT_MAMBA as usize] += vlen;
            }
            self.nodes[node as usize].lock_ref[CT_MAMBA as usize] += 1;
        }
    }

    fn mamba_release_lock(&mut self, node: u32, params: &UDecLockParams, lock_host: bool) {
        if node == self.root {
            return;
        }
        let skip = params.skip_ids(CT_MAMBA);
        if skip.contains(&node) {
            return;
        }
        if lock_host {
            self.nodes[node as usize].host_lock_ref[CT_MAMBA as usize] -= 1;
            let cd = &self.nodes[node as usize];
            if cd.host_lock_ref[CT_MAMBA as usize] == 0
                && cd.value[CT_MAMBA as usize].is_none()
                && cd.host_value[CT_MAMBA as usize].is_some()
            {
                let host_slot = Self::lru_slot_public(CT_MAMBA, 1);
                if !self.lru_in(host_slot, node) {
                    self.lru_insert_mru(host_slot, node);
                }
            }
            return;
        }
        let ref_now = self.nodes[node as usize].lock_ref[CT_MAMBA as usize];
        if ref_now > 0 {
            if ref_now == 1 {
                let vlen = self.nodes[node as usize]
                    .value[CT_MAMBA as usize]
                    .as_ref()
                    .map(|v| v.len() as i64)
                    .unwrap_or(0);
                self.evictable_size[CT_MAMBA as usize] += vlen;
                self.protected_size[CT_MAMBA as usize] -= vlen;
            }
            self.nodes[node as usize].lock_ref[CT_MAMBA as usize] = ref_now - 1;
        }
    }
}
