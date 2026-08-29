//! `sglang-scheduler` — the SGLang CPU control plane in Rust.
//!
//! Two front ends over the same pure decision engine ([`planner::plan_next_batch`]):
//!
//! - **Shadow planner** (`SGLANG_RUST_SCHEDULER=planner`): Python keeps the
//!   queues/tree/allocator and passes compact snapshots in; the returned
//!   [`types::BatchPlan`] is diffed against Python's own decision (trace
//!   capture) before (eventually) being applied.
//! - **Core** (`SGLANG_RUST_SCHEDULER=core`, [`core::SchedulerCore`]): the
//!   engine owns the queues, the [`sglang_radix::RadixTree`] and the
//!   new-token-ratio tracker, and folds result bookkeeping into each step.
//!
//! Scope: base MHA, single-node, non-PP/DP, non-overlap — the fastest path
//! for the Qwen3-27B NVFP4 target. Spec-v2 accept-run bookkeeping (plan §9)
//! is ported in [`spec`]; the draft/verify GPU shapes stay in Python.

pub mod adder;
pub mod core;
pub mod ntr;
pub mod planner;
pub mod policy;
pub mod spec;
pub mod types;

#[cfg(feature = "python")]
mod pybind;
#[cfg(feature = "python")]
mod unified;

pub use planner::plan_next_batch;
pub use spec::{SpecCounters, SpecError, SpecResolution, SpecRow, SpecRun, resolve_spec_runs};
pub use types::{BatchPlan, Config, PlanReq, StepEnv};
