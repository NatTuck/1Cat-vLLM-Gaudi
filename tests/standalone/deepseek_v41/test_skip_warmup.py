# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the VLLM_SKIP_WARMUP startup mode.

These drive the runner and worker code paths directly: no HPU allocation, no
service, no device context.
"""
import logging
from types import SimpleNamespace

import pytest
import torch

import vllm_gaudi.ops.tp2_prepared_plan as prepared_plan
import vllm_gaudi.ops.deepseek_v41_prefill_plan as prefill_plan
import vllm_gaudi.v1.worker.deepseek_v41_runner as runner_module
from vllm_gaudi.v1.worker import hpu_worker as worker_module


def _fake_worker(monkeypatch, *, graphs=0, replays=0, fallbacks=0, tp=4, native=True):
    """A worker whose deferred diagnostics can be driven without a device."""
    worker = worker_module.HPUWorker.__new__(worker_module.HPUWorker)
    worker.rank = 0
    worker.parallel_config = SimpleNamespace(tensor_parallel_size=tp)
    ordinary = SimpleNamespace(audit={"calls": 0})
    model = SimpleNamespace(ordinary=ordinary,
                            compiled_input_calls=0,
                            program=SimpleNamespace(replay_owner=object() if native else None))
    worker.model_runner = SimpleNamespace(audit={
        "prefill_steps": 0,
        "decode_steps": 0
    },
                                          model=model)
    stats = {
        "native_graphs": graphs,
        "native_entry_replays": replays,
        "tp4_direct_group_replays": fallbacks,
    }
    monkeypatch.setattr(prepared_plan, "prepared_group_stats", lambda: dict(stats))
    return worker, model, stats


def _errors(caplog):
    return [record for record in caplog.records if record.levelno >= logging.ERROR]


def _step(worker, *, prefill=0, decode=0, scheduled=1):
    """Run one post-execution diagnostic step with the given audit deltas."""
    before = worker._v41_audit_counters()
    worker.model_runner.audit["prefill_steps"] += prefill
    worker.model_runner.audit["decode_steps"] += decode
    worker._v41_deferred_diagnostics(SimpleNamespace(num_scheduled_tokens={"req": scheduled}), before)


def test_deferred_decode_check_is_quiet_on_the_first_decode_of_a_cold_geometry(monkeypatch, caplog):
    """A cold group prepares, captures, then replays: nothing is wrong on steps 1-2."""
    caplog.set_level(logging.INFO)
    worker, _, stats = _fake_worker(monkeypatch, graphs=0, replays=0)
    worker._v41_deferred_native_decode_check = True
    worker._v41_deferred_prefill_plan_check = False

    _step(worker, decode=1)
    assert not _errors(caplog), "the first real decode step must not be reported as a failure"
    assert worker._v41_deferred_native_decode_check is True

    stats["native_graphs"] = 1
    _step(worker, decode=1)
    assert not _errors(caplog), "a capture without a replay yet must not be reported as a failure"
    assert worker._v41_deferred_native_decode_check is True

    stats["native_entry_replays"] = 1
    _step(worker, decode=1)
    assert not _errors(caplog)
    assert worker._v41_deferred_native_decode_check is False, "the diagnostic must disarm once it is satisfied"
    assert any("native decode stage executed" in record.getMessage() for record in caplog.records)


def test_deferred_decode_check_reports_a_missing_capture_and_disarms(monkeypatch, caplog):
    """From the second real decode step a capture must exist. Never raises."""
    caplog.set_level(logging.INFO)
    worker, _, _ = _fake_worker(monkeypatch, graphs=0, replays=0)
    worker._v41_deferred_native_decode_check = True
    worker._v41_deferred_prefill_plan_check = False

    for _ in range(worker._V41_DEFERRED_DECODE_WATCH_STEPS):
        _step(worker, decode=1)

    assert worker._v41_deferred_native_decode_check is False
    assert len(_errors(caplog)) == 1
    assert "without capturing a native decode graph" in _errors(caplog)[0].getMessage()


def test_deferred_decode_check_reports_a_missing_replay_and_disarms(monkeypatch, caplog):
    """From the third real decode step a replay must exist. Never raises."""
    caplog.set_level(logging.INFO)
    worker, _, _ = _fake_worker(monkeypatch, graphs=1, replays=0)
    worker._v41_deferred_native_decode_check = True
    worker._v41_deferred_prefill_plan_check = False

    for _ in range(worker._V41_DEFERRED_DECODE_WATCH_STEPS):
        _step(worker, decode=1)

    assert worker._v41_deferred_native_decode_check is False
    assert len(_errors(caplog)) == 1
    assert "without a complete native decode stage" in _errors(caplog)[0].getMessage()


def test_deferred_prefill_plan_check_reports_and_disarms_without_raising(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    worker, _, _ = _fake_worker(monkeypatch)
    monkeypatch.setattr(prefill_plan, "prefill_plan_stats",
                        lambda: {"executed": {"largest_token_bucket": 64}})
    worker._v41_deferred_prefill_plan_check = True
    worker._v41_deferred_native_decode_check = False

    _step(worker, prefill=1, scheduled=8192)

    assert worker._v41_deferred_prefill_plan_check is False
    assert len(_errors(caplog)) == 1
    assert "largest executed bucket 64" in _errors(caplog)[0].getMessage()


def test_short_chunked_prefill_tail_is_not_treated_as_a_decode_step(monkeypatch, caplog):
    """A 1-6 token prompt tail is a prefill step; only audit decode deltas count."""
    caplog.set_level(logging.INFO)
    worker, _, stats = _fake_worker(monkeypatch, graphs=0, replays=0)
    worker._v41_deferred_native_decode_check = True
    worker._v41_deferred_prefill_plan_check = True
    monkeypatch.setattr(prefill_plan, "prefill_plan_stats",
                        lambda: {"executed": {"largest_token_bucket": 1}})

    for _ in range(worker._V41_DEFERRED_DECODE_WATCH_STEPS + 2):
        _step(worker, prefill=1, scheduled=3)

    assert getattr(worker, "_v41_deferred_decode_steps", 0) == 0
    assert worker._v41_deferred_native_decode_check is True, "a prefill tail must not consume the decode watch window"
    assert not _errors(caplog), "a short prefill must not be reported as a decode failure"
    # The prefill-plan check is also gated on a >=128-token prefill.
    assert worker._v41_deferred_prefill_plan_check is True


def test_deferred_diagnostics_survive_a_raising_counter(monkeypatch, caplog):
    """The post-execution path must never propagate: peers would wait on a collective."""
    caplog.set_level(logging.INFO)
    worker, _, _ = _fake_worker(monkeypatch, graphs=0, replays=0)
    worker._v41_deferred_native_decode_check = True
    worker._v41_deferred_prefill_plan_check = True

    def explode():
        raise RuntimeError("counter unavailable")

    monkeypatch.setattr(prepared_plan, "prepared_group_stats", explode)
    _step(worker, decode=1)

    assert worker._v41_deferred_native_decode_check is False
    assert worker._v41_deferred_prefill_plan_check is False


def _skip_warmup_runner(monkeypatch, frozen_calls):
    runner = runner_module.V41ModelRunner.__new__(runner_module.V41ModelRunner)
    ready, targets, prefills = [], [], []
    program = SimpleNamespace(length=65536,
                              runtime_indexer=False,
                              replay_owner=SimpleNamespace(require_ready=lambda c, *, search: ready.append((c, search))))
    runner.model = SimpleNamespace(native=True, program=program, last_aux=torch.zeros(6, 3), pp_rank=0, tp_rank=0)
    runner.vllm_config = SimpleNamespace(additional_config={})
    runner.positions = torch.arange(8192)
    runner.state = runner_module.PagedStageState.__new__(runner_module.PagedStageState)
    runner.state.clear = lambda: None
    runner.pp = SimpleNamespace(group=SimpleNamespace(is_last_rank=True, barrier=lambda: None))
    runner.request_batches = None
    runner.graphed_buckets = set()
    runner.use_dspark = False
    runner.prefix_checkpoints = object()
    runner.prefill_capacity = 32768
    runner.model_config = SimpleNamespace(max_model_len=65536)
    runner.active_request = None
    runner._dummy_run = lambda count, **kw: (prefills if kw.get("prefill") else targets).append((count, kw))
    runner._warm_device_round_inputs = lambda: None
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: None)
    monkeypatch.setattr(runner_module.envs, "VLLM_HPU_DSV41_DSPARK", False)
    monkeypatch.setattr(runner_module.envs, "VLLM_HPU_DSV41_VERIFY_TIMING", False)
    import vllm_gaudi.ops.deepseek_v41_prefill_regions as regions

    monkeypatch.setattr(regions, "freeze_prefill_regions", lambda: frozen_calls.append(True))
    return runner, ready, targets, prefills


def _mock_bank():
    """Just enough BatchState for the warmed serving-prefill sweep."""
    slot = SimpleNamespace(index=0)
    return SimpleNamespace(pages=torch.zeros(4, 8, dtype=torch.int32),
                           acquire=lambda owner: slot,
                           publish_pages=lambda *args: None,
                           bind_prefill=lambda *args: None,
                           bind_single=lambda *args: None,
                           restore_single_bindings=lambda: None,
                           warmup_single_handoff=lambda blocks: None,
                           release=lambda *args: None)


@pytest.mark.parametrize("mode", (True, False))
def test_warmup_model_skip_mode_compiles_nothing_and_leaves_regions_unfrozen(monkeypatch, caplog, mode):
    caplog.set_level(logging.INFO)
    frozen = []
    runner, ready, targets, prefills = _skip_warmup_runner(monkeypatch, frozen)
    if not mode:
        # The warmed path exercises the serving-prefill sweep, which needs a bank.
        runner.prefill_capacity = 8192
        runner.model.batch_state = _mock_bank()
        runner.state.blocks = 8
    monkeypatch.setattr(runner_module, "get_config", lambda: SimpleNamespace(skip_warmup=mode))

    runner.warmup_model()

    summary = [record.getMessage() for record in caplog.records if "warmup summary" in record.getMessage()]
    assert len(summary) == 1, "one summary line per rank"
    if mode:
        assert (targets, prefills, ready) == ([], [], []), "no warmup shape may execute"
        assert runner.graphed_buckets == set(), "skipped buckets must not be reported as graphed"
        assert frozen == [], "prefill regions must stay unfrozen so shapes compile on first use"
        assert "freeze_prefill_regions" in summary[0]
        assert "skipped=[" in summary[0] and "C1_search_buckets" in summary[0]
    else:
        assert targets and ready, "the default path still warms the decode shapes"
        assert runner.graphed_buckets == {1}
        assert frozen == [True], "the default path freezes the warmed regions"
