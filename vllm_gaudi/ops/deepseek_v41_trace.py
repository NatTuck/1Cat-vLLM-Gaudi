# SPDX-License-Identifier: Apache-2.0
"""On-demand prefill scopes; never add synchronization or compiled nodes."""
import contextlib
from functools import wraps
import json
import os
import threading
import time

import torch

_active = False


def enable(enabled):
    global _active
    _active = bool(enabled)


def prefill_scope(name):

    def decorate(function):

        @wraps(function)
        def call(self, *args, **kwargs):
            if torch.compiler.is_compiling() or not _active:
                return function(self, *args, **kwargs)
            # C1 native replay has its own stage/segment markers. These host
            # scopes are for the large-M path, not per-token replay callbacks.
            if args and isinstance(args[0], torch.Tensor) and args[0].shape[0] <= 6:
                return function(self, *args, **kwargs)
            label = f"v41::prefill::{name}::L{getattr(self, 'layer', 'unknown')}"
            with torch.profiler.record_function(label):
                return function(self, *args, **kwargs)

        return call

    return decorate


_SPAN_LOG = os.environ.get("VLLM_HPU_DSV41_STEP_SPANS") or None
_span_stream = None
_span_lock = threading.Lock()


class _JsonlSpan:
    """Record one span as a JSON line, with no profiler and no SDK involvement at all.

    Three attempts to capture this through the Kineto profiler failed on this build (read-only
    $HOME/.habana, then /stop_profile -> 500 "cancelled" with an empty trace and no worker traces),
    and a profiler-enabled build also costs +15-17 % on the bar. A plain append to a JSONL file
    depends on nothing but Python, so it cannot fail that way and cannot perturb the step time.

    Timestamps are `time.perf_counter_ns()`, the same clock the batch runner already writes into
    request-trace/rank<N>.jsonl as `start_ns`/`end_ns`, so spans align to steps with no extra
    instrumentation on the step side.
    """

    __slots__ = ("name", "start")

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, *exception):
        global _span_stream
        end = time.perf_counter_ns()
        with _span_lock:
            if _span_stream is None:
                _span_stream = open(_SPAN_LOG, "a", buffering=1)
            _span_stream.write(json.dumps({
                "name": self.name, "start_ns": self.start, "end_ns": end,
                "dur_us": (end - self.start) / 1000.0,
                "pid": os.getpid(), "tid": threading.get_native_id(),
            }) + "\n")
        return False


def host_span(name):
    """A host-side profiler annotation that costs ~0.6 us unless a profiler is running.

    `torch.profiler.record_function` is already a semantic no-op when nothing is profiling, but it
    still constructs a context manager on every call, and the two spans that use this sit on the
    decode step's critical path -- the very region under measurement. `torch.autograd
    ._profiler_enabled()` is the predicate the profiler itself uses, so guarding on it makes the
    idle cost to a measured 596 ns per span (two spans per decode step is ~1.2 us against a 66 ms
    step, 0.002 %), while keeping the annotation in the trace whenever a profiler is active.

    Safe by construction: `record_function` is a HOST-side annotation, so it cannot add a node to a
    compiled graph. Neither call site is inside a torch.compile region -- one is the batch runner's
    host pre-submission path, the other wraps the loop that *calls* the compiled chunks.
    """
    if torch.autograd._profiler_enabled():
        return torch.profiler.record_function(name)
    if _SPAN_LOG:
        return _JsonlSpan(name)
    return contextlib.nullcontext()
