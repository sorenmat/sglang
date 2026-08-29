//! Port of `managers/scheduler_components/new_token_ratio_tracker.py`.
//!
//! Float math mirrors Python bit-for-bit: every operation is the same
//! `f64` op in the same order (Python floats are C doubles).

use crate::types::Config;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ntr {
    init: f64,
    min: f64,
    decay: f64,
    current: f64,
}

impl Ntr {
    /// `NewTokenRatioTracker.from_config()` with the env values folded into
    /// the config.
    pub fn from_config(cfg: &Config) -> Self {
        let init = (cfg.ntr_init_raw * cfg.schedule_conservativeness).min(1.0);
        let min = (init * cfg.ntr_min_factor).min(1.0);
        let decay = (init - min) / cfg.ntr_decay_steps.max(1) as f64;
        Self {
            init,
            min,
            decay,
            current: init,
        }
    }

    pub fn decay_step(&mut self) {
        self.current = (self.current - self.decay).max(self.min);
    }

    pub fn reset(&mut self) {
        self.current = self.init;
    }

    pub fn current(&self) -> f64 {
        self.current
    }

    /// `new_token_ratio_tracker.current = new_token_ratio` (the post-retract
    /// estimate, applied by `update_running_batch` on the OOM path).
    pub fn set_current(&mut self, v: f64) {
        self.current = v;
    }

    /// The value [`Self::decay_step`] would produce, without mutating.
    pub fn next_after_decay(&self) -> f64 {
        (self.current - self.decay).max(self.min)
    }

    /// `estimate_new_token_ratio_after_retract(reqs)`:
    /// `(sum(out) + RETRACT_STEPS * n) / (sum(max_new) + 1)`, capped at 1.0.
    pub fn estimate_after_retract(
        out_lens: &[u32],
        max_news: &[u32],
        retract_steps: u32,
    ) -> f64 {
        debug_assert_eq!(out_lens.len(), max_news.len());
        let decoded: i64 = out_lens.iter().map(|l| *l as i64).sum();
        let max_new: i64 = max_news.iter().map(|l| *l as i64).sum();
        let ratio =
            (decoded as f64 + retract_steps as f64 * out_lens.len() as f64) / (max_new as f64 + 1.0);
        ratio.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        // Defaults: init = min(0.7 * 1.0, 1.0) = 0.7,
        // min = min(0.7 * 0.1, 1.0) = 0.07, decay = 0.63 / 600.
        Config::default()
    }

    #[test]
    fn from_config_values() {
        let mut n = Ntr::from_config(&cfg());
        assert!((n.current() - 0.7).abs() < 1e-18);
        for _ in 0..10 {
            n.decay_step();
        }
        // 0.7 - 10 * (0.63 / 600)
        let expected = 0.7 - 10.0 * (0.63 / 600.0);
        assert!((n.current() - expected).abs() < 1e-18);
    }

    #[test]
    fn decay_never_below_min() {
        let mut n = Ntr::from_config(&cfg());
        // Python: `min(init * factor, 1.0)` — compare against the same f64
        // op, not the literal 0.07 (the product rounds differently).
        let min_v = 0.7 * 0.1;
        for _ in 0..10_000 {
            n.decay_step();
        }
        // Pinned exactly at the floor: (current - decay).max(min) == min.
        assert_eq!(n.current(), min_v);
        let before = n.current();
        n.decay_step();
        assert_eq!(n.current(), before);
    }

    #[test]
    fn estimate_matches_python_formula() {
        // Python: (1 + 3 + 20*2) / (100 + 200 + 1), capped at 1.0.
        let got = Ntr::estimate_after_retract(&[1, 3], &[100, 200], 20);
        assert!((got - 44.0 / 301.0).abs() < 1e-12);
        // Saturates at 1.0: (4 + 40) / (10 + 20 + 1) > 1.
        assert_eq!(Ntr::estimate_after_retract(&[1, 3], &[10, 20], 20), 1.0);
        assert_eq!(Ntr::estimate_after_retract(&[900, 900], &[10, 10], 20), 1.0);
        // Zero outputs: (0 + 20*2) / (0 + 1) = 40 -> capped at 1.0.
        assert_eq!(Ntr::estimate_after_retract(&[0, 0], &[0, 0], 20), 1.0);
        // One req, no outputs beyond the steps: (5 + 20) / (1000 + 1).
        assert!((Ntr::estimate_after_retract(&[5], &[1000], 20) - 25.0 / 1001.0).abs() < 1e-12);
    }
}
