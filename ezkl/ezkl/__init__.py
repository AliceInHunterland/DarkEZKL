"""
ezkl Python package shim.

The compiled Rust extension is built as the submodule `ezkl.ezkl` (file:
`ezkl/ezkl.*.so`). Re-export its public API at the package level so callers can:

    import ezkl
    ezkl.version()
"""

from .ezkl import *  # noqa: F401,F403
