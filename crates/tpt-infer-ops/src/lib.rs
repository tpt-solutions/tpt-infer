//! Operators and backends.
#![cfg_attr(not(test), no_std)]
pub mod backend;
pub mod naive;
pub mod dispatch;
pub use backend::Backend;
