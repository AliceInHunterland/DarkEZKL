#![allow(clippy::needless_return)]

//! Minimal `ezkl` library target.
//!
//! The upstream project contains substantial functionality, but in this kata
//! repository snapshot only a subset of files is present.
//!
//! This repository's Docker build uses `maturin build --features python-bindings`
//! and expects the produced shared object to be a real CPython extension module
//! exporting a `PyInit_*` symbol.

/// Returns the crate version from Cargo metadata.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(all(feature = "python-bindings", not(target_arch = "wasm32")))]
use pyo3::prelude::*;

#[cfg(all(feature = "python-bindings", not(target_arch = "wasm32")))]
use pyo3::types::PyModuleMethods;

// Ensure the bindings module is included when python-bindings are enabled
#[cfg(all(feature = "python-bindings", not(target_arch = "wasm32")))]
#[path = "bindings/python.rs"]
mod python_bindings;

#[cfg(all(feature = "python-bindings", not(target_arch = "wasm32")))]
#[pyfunction]
#[pyo3(name = "version")]
fn version_py() -> &'static str {
    version()
}

#[cfg(all(feature = "python-bindings", not(target_arch = "wasm32")))]
#[pymodule]
#[pyo3(name = "ezkl")]
pub fn ezkl(module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Keep a conventional attribute for Python callers/tests.
    module.add("__version__", version())?;
    // Provide a tiny callable API so import-time sanity checks have something stable.
    module.add_function(wrap_pyfunction!(version_py, module)?)?;

    // Register the custom bindings (including gen_settings/calibrate_settings)
    python_bindings::register(module.py(), module)?;

    Ok(())
}
