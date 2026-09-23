//! Zero-allocation inference runtime.
#![cfg_attr(not(test), no_std)]
#[cfg(feature = "alloc")]
extern crate alloc;
pub mod exec;
pub use exec::execute;
