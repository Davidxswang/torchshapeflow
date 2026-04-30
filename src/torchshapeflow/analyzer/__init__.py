"""Analyzer subpackage entry: re-exports the public API.

The expression-evaluation walker is split across three modules to keep each
file under the 600-line ceiling:

- ``expressions``: ``eval_expr`` dispatcher + ``apply_module_spec`` and the
  signature/module-alias helpers.
- ``calls``: ``eval_call`` for ``ast.Call`` nodes + size/constructor/interpolate
  helpers.
- ``tensor_methods``: ``eval_tensor_method`` for ``x.method(...)`` calls +
  ``infer_expand``.

Cross-module recursion is broken with a single late import inside
``expressions.eval_expr`` (which pulls in ``calls.eval_call``); every other
edge is a top-level import.
"""

from __future__ import annotations

from torchshapeflow.analyzer.entry import analyze_path, analyze_source

__all__ = ["analyze_path", "analyze_source"]
