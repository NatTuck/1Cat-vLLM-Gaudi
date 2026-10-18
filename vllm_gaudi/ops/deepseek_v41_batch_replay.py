# SPDX-License-Identifier: Apache-2.0
"""Finite ordinary-request batch graphs, with request-owned state capture."""
from dataclasses import replace
import os

import torch

from vllm_gaudi.models.deepseek_v41_batch_program import CompiledBatchStage
from vllm_gaudi.ops.deepseek_v41_replay import _Metadata
from vllm_gaudi.ops.deepseek_v41_reindex_compact import bounded_reindex_bucket
from vllm_gaudi.ops.tp2_model_adapter import DEEPSEEK_V41_PP0_INPUT, DEEPSEEK_V41_PP1


class BatchSnapshot:
    """Save only rows this C1/request transaction can overwrite during capture."""

    def __init__(self, program, positions, slots, pages):
        from vllm_gaudi.ops.deepseek_v41_batch_attention import batch_physical_rows
        self.rows = []
        active = (slots >= 0) & (positions >= 0)

        def save(value, indices):
            indices = indices.detach().cpu().unique()
            indices = indices[(indices >= 0) & (indices < value.shape[0])].to(value.device).long()
            if indices.numel():
                self.rows.append((value, indices, value.index_select(0, indices).clone()))

        for layer in program.layers:
            state = layer.attention.batch_state
            save(state.swa, torch.where(active, slots * 256 + positions.remainder(256), -1))
            if hasattr(state, "kv_history"):
                row = torch.where(active, slots * 8 + positions.remainder(8), -1)
                save(state.kv_history, row)
                save(state.score_history, row)
        for cache in program.shared.sources.values():
            row = batch_physical_rows((positions // cache.ratio)[:, None], pages, cache.ratio).flatten()
            visible = active & ((positions + 1).remainder(cache.ratio) == 0)
            row = torch.where(visible, row, -1)
            save(cache.main, row)
            save(cache.index, row)
        self.bytes = sum(value.numel() * value.element_size() for _, _, value in self.rows)

    def restore(self):
        for destination, indices, saved in self.rows:
            destination.index_copy_(0, indices, saved)


class BatchStageVariant(torch.nn.Module):
    """Reuse the native joined-recipe executor for one static request bucket."""

    def __init__(self, program, hidden, pre, positions, ids, engram, slots, pages):
        super().__init__()
        if program.dspark or not program.runtime_indexer or program.length <= 512:
            raise ValueError("Batched native replay requires paged ordinary runtime-position CSA2")
        self.program = program
        self.bucket = positions.numel()
        if self.bucket not in (1, 2, 4, 8, 16, 32, 64):
            raise ValueError("Unsupported native request bucket")
        base = DEEPSEEK_V41_PP0_INPUT if program.pp_rank == 0 else DEEPSEEK_V41_PP1
        # The module constants carry a five-group topology (20 layers). This model
        # has 40, so CompiledBatchStage builds ten chunks and _flush never sees
        # len(plans) == full_groups: it replays prepared groups but never captures,
        # so require_ready can never pass. Derive the group count from the stage,
        # as the ordinary C1 owner already does (deepseek_v41_replay.py:266).
        layers = len(program.layers)
        if layers % 4:
            raise ValueError("Native V4.1 stage requires complete four-layer groups")
        self.adapter = replace(base,
                               name=f"deepseek_v41_pp{program.pp_rank}_request_batch",
                               group_layers=(4,) * (layers // 4),
                               extra_collectives=base.extra_collectives +
                               sum(2 for layer in program.layers if layer.attention.owns_index))
        self.compiled = CompiledBatchStage(program, text_input=program.pp_rank == 0)
        self.fixed = tuple(value.clone() for value in (hidden, pre, positions, ids))
        self.attention_inputs = tuple(value.clone() for value in (*engram, slots, pages))
        self.engram_count = len(engram)
        self.empty_selections = tuple(
            torch.full((self.bucket, 512), -1, dtype=torch.int32, device=positions.device) for _ in program.shared.topk)
        self.empty_candidates = torch.full((self.bucket, 2048), -1, dtype=torch.int32, device=positions.device)
        self.empty_ready = tuple(
            torch.zeros(self.bucket, dtype=torch.int32, device=positions.device) for _ in program.shared.sources)
        self.states = (tuple(value for layer in program.layers for value in layer.attention.batch_state.buffers()) +
                       tuple(value for cache in program.shared.sources.values() for value in (cache.main, cache.index)))
        self.metadata, self.capture_bytes = _Metadata(), 0
        self.reindex_ratios = tuple(
            layer.attention.ratio for layer in program.layers
            if bounded_reindex_bucket(self.bucket) and layer.attention.batch_reindex_mme and layer.attention.
            bounded_reindex and layer.attention.owns_index and layer.layer > layer.attention.candidate_source)
        if len(set(self.reindex_ratios)) > 1:
            raise ValueError("Bounded Reindex tactics require one compression ratio per stage")

    def snapshot(self):
        result = BatchSnapshot(self.program, self.fixed[2], *self.attention_inputs[-2:])
        self.capture_bytes = result.bytes
        return result

    def forward(self, hidden, pre, positions, ids, engram, slots, pages, *, host_positions=None):
        from vllm_gaudi.ops.tp2_prepared_plan import (
            collect_prepared_group_replays,
            record_native_decoder_outputs,
            replay_native_decoder,
        )
        if os.environ.get("VLLM_HPU_DSV41_BATCH_PROBE") == "1" and self.bucket >= 2:
            # Direct observation, not inference: two identical prompts in one
            # batch have identical positions and identically cleared slots, so
            # any output difference between them is produced by something keyed
            # on the slot index. Print what the graph is actually handed.
            if getattr(self, "_probe_count", 0) < 40:
                self._probe_count = getattr(self, "_probe_count", 0) + 1
                print(f"[batch-probe] bucket={self.bucket} "
                      f"slots={slots.tolist()} positions={positions.reshape(-1).tolist()} "
                      f"pages_rows={[row[:4] for row in pages.tolist()]} "
                      f"pages_identical={pages[0].tolist() == pages[1].tolist()}",
                      flush=True)
        # Pin the shapes the compiler sees. The batch path hands positions and
        # ids in from a host-filled metadata copy, and an operand whose shape is
        # not statically declared reaches the swa index_copy_ unranked; Synapse
        # then rejects the node with "Invalid Node Params" naming nothing useful.
        # The bucket is fixed per variant, so reshaping is free and exact.
        positions = positions.reshape(self.bucket)
        ids = ids.reshape(self.bucket)
        roots = dict(hidden_states=hidden,
                     pre_mix=pre,
                     positions=positions,
                     input_ids=ids,
                     attention_inputs=(*engram, slots, pages),
                     metadata=self.metadata,
                     state_generation=(self.program.generation, self.program.precision_fingerprint),
                     state_tensors=self.states)
        if bounded_reindex_bucket(self.bucket) and self.reindex_ratios:
            from vllm_gaudi.ops.deepseek_v41_reindex_compact import unique_pool_tile_bound
            if host_positions is None or len(host_positions) != self.bucket:
                raise RuntimeError("Bounded Reindex requires scheduler-owned positions for every padded row")
            roots["reindex_tile_bound"] = max(
                unique_pool_tile_bound(host_positions, ratio) for ratio in self.reindex_ratios)
        output = replay_native_decoder(self, **roots)
        if output is not None:
            return output
        for destination, source in zip(self.fixed, (hidden, pre, positions, ids), strict=True):
            destination.copy_(source)
        for destination, source in zip(self.attention_inputs, (*engram, slots, pages), strict=True):
            destination.copy_(source)
        h, p, positions, ids = self.fixed
        slots, pages = self.attention_inputs[-2:]
        selections, candidates = self.empty_selections, self.empty_candidates
        main_ready = index_ready = self.empty_ready
        fixed_roots = dict(roots,
                           hidden_states=h,
                           pre_mix=p,
                           positions=positions,
                           input_ids=ids,
                           attention_inputs=self.attention_inputs)
        with collect_prepared_group_replays(owner=self, adapter=self.adapter, snapshot=self.snapshot,
                                            **fixed_roots) as context:
            for index, chunk in enumerate(self.compiled.chunks):
                context["group_index"] = index
                h, p, selections, candidates, main_ready, index_ready = chunk(h, p, positions, ids,
                                                                              self.attention_inputs[:self.engram_count],
                                                                              slots, pages, selections, candidates,
                                                                              main_ready, index_ready)
            record_native_decoder_outputs(h, p, None)
        if os.environ.get("VLLM_HPU_DSV41_BATCH_PROBE") == "1":
            # Quantify the numerical difference the batch geometry introduces,
            # rather than only detecting that an argmax diverged. `h` is the
            # stage output, so this measures the whole forward, and bucket 1
            # gives the serial reference for the same request.
            if getattr(self, "_hprobe_count", 0) < 40:
                self._hprobe_count = getattr(self, "_hprobe_count", 0) + 1
                hf = h.float()
                peak = float(hf.abs().max())
                if hf.shape[0] >= 2:
                    delta = float((hf[0] - hf[1]).abs().max())
                    print(f"[h-probe] bucket={self.bucket} peak={peak:.6g} "
                          f"row0_vs_row1_maxabs={delta:.6g} rel={delta / peak if peak else 0:.3g}",
                          flush=True)
                else:
                    print(f"[h-probe] bucket={self.bucket} peak={peak:.6g} "
                          f"checksum={float(hf.sum()):.8g}", flush=True)
        return h, p, None


class BatchStageReplay:

    def __init__(self, program):
        self.program, self.variants = program, {}
        self.generation = (program.generation, program.precision_fingerprint)

    def __call__(self, hidden, pre, positions, ids, engram, slots, pages, *, host_positions=None):
        identity = (self.program.generation, self.program.precision_fingerprint)
        if identity != self.generation:
            self.close()
            self.generation = identity
        bucket = positions.numel()
        if bucket not in self.variants:
            self.variants[bucket] = BatchStageVariant(self.program, hidden, pre, positions, ids, engram, slots, pages)
        return self.variants[bucket](hidden, pre, positions, ids, engram, slots, pages, host_positions=host_positions)

    def require_ready(self, bucket):
        from vllm_gaudi.ops.tp2_prepared_plan import _native_entries
        if self.variants.get(bucket) not in _native_entries:
            raise RuntimeError(f"Native request bucket {bucket} has not completed preparation")

    def close(self):
        from vllm_gaudi.ops.tp2_prepared_plan import invalidate_prepared_group_plans
        for variant in self.variants.values():
            invalidate_prepared_group_plans(owner=variant, reason="batch_stage_close")
        self.variants.clear()
