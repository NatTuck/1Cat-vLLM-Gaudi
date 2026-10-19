# SPDX-License-Identifier: Apache-2.0
"""Request-batch execution owned by the normal V4.1 model runner."""
import json
from contextlib import contextmanager
import os
from pathlib import Path
import time

import torch

from vllm_gaudi.ops.deepseek_v41_trace import host_span

from vllm_gaudi.ops.deepseek_v41_batch_input import fill_request_metadata
import torch.distributed as dist

from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()

from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import ModelRunnerOutput


class BatchExecution:
    """One generation per stage; all buffers have a last-consumer owner."""

    def __init__(self, runner, capacity):
        self.runner, self.capacity = runner, capacity
        self.model, self.device = runner.model, runner.device
        self.bank = self.model.batch_state
        self.buckets = tuple(b for b in (1, 2, 4, 8, 16, 32, 64) if b <= 1 << (capacity - 1).bit_length())
        self.buffers, self.samplers = {}, {}
        # The batch path was written for a two-stage pipeline. Record the real
        # shape so a single-stage (PP1) deployment samples locally instead of
        # indexing ranks[1], which does not exist.
        _pp = self.runner.pp.group
        self._pp_stages = len(_pp.ranks)
        # get_pp_group().ranks holds *global* ranks, and a PP1 worker's group is
        # just itself, while model.pp_rank is the PP index. Comparing the two held
        # only on global rank 0, so ranks 1-3 built no sampler and died with
        # KeyError on the first batch. Use the predicate the rest of the runner
        # already uses for "this stage owns the output head".
        self._last_stage = bool(getattr(_pp, "is_last_rank", True))
        for b in self.buckets:
            host = torch.zeros(3, b, dtype=torch.int32).pin_memory("hpu")
            meta = torch.zeros_like(host, device=self.device)
            pages = torch.zeros(b, self.bank.pages.shape[1], dtype=torch.int32, device=self.device)
            packet = torch.empty(b * 10244, dtype=torch.int32, device=self.device)
            hidden = packet[:b * 10240].view(torch.bfloat16).view(b, 4, 5120)
            pre = packet[b * 10240:].view(torch.float32).view(b, 4)
            token = torch.zeros(b, 1, dtype=torch.int32, device=self.device)
            token_host = torch.zeros(b, 1, dtype=torch.int32).pin_memory("hpu")
            self.buffers[b] = (host, meta, pages, packet, hidden, pre, token, token_host)
            if self._last_stage:
                self.samplers[b] = torch.compile(self.model.program.sample_greedy_token,
                                                 backend="hpu_backend",
                                                 fullgraph=True,
                                                 dynamic=False)
        self.generation = 0
        self.last_done = None
        from vllm_gaudi import envs
        if envs.VLLM_HPU_DSV41_PP_MICROBATCHES not in (1, 2):
            raise ValueError("Ordinary PP decode supports one batch or two owned microbatches")
        self.pipeline = None
        if envs.VLLM_HPU_DSV41_PP_MICROBATCHES == 2:
            from vllm_gaudi.v1.worker.deepseek_v41_pipeline_batch import TwoMicrobatchPipeline
            self.pipeline = TwoMicrobatchPipeline(self)

    def execute(self, requests):
        requests = tuple(requests)
        if not getattr(self.runner, "trace_enabled", False):
            return self._execute(requests)
        with self._trace_requests(requests, "batch"):
            return self._execute(requests)

    @contextmanager
    def trace_single(self, request):
        with self._trace_requests((request, ), "native_single"):
            self.generation += 1
            self.runner.audit["native_single_steps"] = self.runner.audit.get("native_single_steps", 0) + 1
            yield

    @contextmanager
    def _trace_requests(self, requests, execution_kind):
        if not getattr(self.runner, "trace_enabled", False):
            yield
            return
        if not hasattr(self, "_request_trace"):
            directory = Path(os.environ["DSV41_RUN_EVIDENCE"]) / "request-trace"
            directory.mkdir(exist_ok=True)
            self._request_trace = (directory / f"rank{dist.get_rank()}.jsonl").open("a", buffering=1)
        record = {
            "start_ns":
            time.perf_counter_ns(),
            "start_raw_ns":
            time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
            "pp_rank":
            self.model.pp_rank,
            "rank":
            dist.get_rank(),
            "execution_kind":
            execution_kind,
            "requests": [{
                "request_id": request.req_id,
                "position": request.num_computed_tokens,
                "output_count": len(request.output),
                "lane": index // ((len(requests) + 1) // 2) if len(requests) >= 32 else 0
            } for index, request in enumerate(requests)],
        }
        if execution_kind == "native_single" and getattr(self.runner, "_step_trace_start", None) is not None:
            # V2 may submit the next native prefix before entering the base
            # runner. Include that producer in this actual request period.
            record["start_ns"], record["start_raw_ns"] = self.runner._step_trace_start
        label = (f"v41::request_batch::PP{self.model.pp_rank}::generation{self.generation + 1}"
                 f"::requests{len(requests)}")
        try:
            with torch.profiler.record_function(label):
                yield
        finally:
            record["end_ns"] = time.perf_counter_ns()
            record["end_raw_ns"] = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            record["generation"] = self.generation
            self._request_trace.write(json.dumps(record, separators=(",", ":")) + "\n")

    @staticmethod
    def _needs_sampling(requests):
        """True when any request asks for non-greedy sampling.

        Warmup requests are built with sampling_params=None, so this must not
        dereference the attribute blindly.
        """
        for request in requests:
            params = getattr(request, "sampling_params", None)
            if params is not None and params.temperature != 0:
                return True
        return False

    _debug_split_warned = False

    @staticmethod
    def _debug_split(label):
        """Diagnostic only: close the region at a named point.

        Bisection instrument. The batch path once failed to compile at bucket 2
        while bucket 1 and prefill succeeded, and the exception surfaced at the
        first D2H copy of the step, naming no node. A synchronise after each
        stage turns that into "which stage owns the bad node".

        Costs four device synchronisations per decode step when enabled, so it
        is a diagnostic and never a setting to leave on: warn once, loudly.
        """
        if os.environ.get("VLLM_HPU_DSV41_BATCH_DEBUG_SPLIT") != "1":
            return
        if not BatchExecution._debug_split_warned:
            BatchExecution._debug_split_warned = True
            logger.warning("V4.1 BATCH_DEBUG_SPLIT is enabled: four device synchronisations per decode step. "
                           "This is a diagnostic and will wreck throughput; unset it to serve.")
        torch.hpu.synchronize()
        logger.info("V4.1 batch debug split completed after %s", label)

    def _execute(self, requests):
        if not requests or len(requests) > self.capacity:
            raise ValueError("Invalid ordinary decode request batch")
        if self.pipeline is not None and len(requests) >= 32:
            return self.pipeline.execute(requests)
        if self.last_done is not None:
            self.last_done.synchronize()
        owners, spans = [], []
        for request in requests:
            start = request.num_computed_tokens
            if start >= request.token_count or start >= self.model.program.length or len(request.block_ids) != 1:
                raise RuntimeError("Request decode batch lacks its committed input or scheduler pages")
            owner = self.bank.acquire(request.req_id)
            self.bank.publish_pages(owner, request.block_ids[0], self.runner.state.blocks)
            token = request.token_at(start)
            spans.append((request.req_id, start, [token], [token in (129264, 129265)]))
            owners.append(owner)
        generation = self.bank.begin(owners)
        self.generation += 1
        b = 1 << (len(requests) - 1).bit_length()
        if getattr(self.runner, "trace_enabled", False):
            mapping = ",".join(f"{owner.index}.{owner.generation}:{req.num_computed_tokens}"
                               for owner, req in zip(owners, requests))
            with torch.profiler.record_function(f"v41::batch_slots::generation{self.generation}::{mapping}"):
                pass
        # HOST span 1 of 2 (see notes/step-head-attribution.md). Everything after the
        # last_done.synchronize() above and before the first chunk submission: the buffer unpack,
        # fill_request_metadata, the meta H2D, the pages index_select. A no-op unless a profiler is
        # active, and outside any compiled region.
        with host_span("v41::batch_step::pre_submit"):
            host, meta, pages, packet, hidden, pre, token, token_host = self.buffers[b]
            fill_request_metadata(host, requests, owners, input_ids=(span[2][0] for span in spans))
            meta.copy_(host, non_blocking=True)
            ids, positions, slots = meta.unbind(0)
            torch.index_select(self.bank.pages, 0, slots.clamp_min(0).long(), out=pages)
            pp = self.runner.pp.group
            single_stage = self._pp_stages == 1
            sent = None
        if self.model.pp_rank == 0:
            values = self.model.forward_request_batch(ids, positions, slots, pages, spans)
            self._debug_split("forward_request_batch")
            if not single_stage:
                # The packet exists only to carry the stage output across a
                # pipeline boundary (dist.isend below). A single-stage
                # deployment samples the forward's own tensors, so writing them
                # into the packet was pure work: two copies into offset views of
                # one int32 buffer on every decode step, whose destinations are
                # never read. Skip them there.
                #
                # This also unblocks batched decode: with these two copies in
                # place the b=2 graph was rejected with "Invalid Node Params"
                # while b=1 compiled. Removing both fixes it. Which of the two
                # copies Synapse rejected, and why b=2 and not b=1, is NOT
                # established -- the offset-view shape of `pre` is equally true
                # at b=1, so it cannot by itself explain the asymmetry. Do not
                # treat this as a compiler workaround with a known mechanism.
                hidden.copy_(values["hidden_states"])
                pre.copy_(values["pre_mix"])
                self._debug_split("batch_io")
            if single_stage:
                # No second stage to sample: do it here, exactly as the last
                # stage would. forward_request_batch returns an
                # IntermediateTensors record on pp_rank 0 rather than the bare
                # hidden tensor a PP2 head receives, so unwrap it first.
                head = values["hidden_states"]
                if self._needs_sampling(requests):
                    token.copy_(self.runner._sample_requests(head, requests))
                else:
                    token.copy_(self.samplers[b](head))
                self._debug_split("sampler_graph")
            else:
                sent = dist.isend(packet, dst=pp.ranks[1], group=pp.device_group)
                self.runner.pp.sends = getattr(self.runner.pp, "sends", 0) + 1
        else:
            received = dist.irecv(packet, src=pp.ranks[0], group=pp.device_group)
            received.wait()
            self.runner.pp.receives = getattr(self.runner.pp, "receives", 0) + 1
            values = self.model.forward_request_batch(ids, positions, slots, pages, spans,
                                                      IntermediateTensors({
                                                          "hidden_states": hidden,
                                                          "pre_mix": pre
                                                      }))
            if self._needs_sampling(requests):
                token.copy_(self.runner._sample_requests(values, requests))
            else:
                # `values` is a bare hidden tensor here, not IntermediateTensors:
                # forward_request_batch returns the record only on pp_rank 0 and
                # the bare tensor on the last stage (models/deepseek_v41.py:391).
                token.copy_(self.samplers[b](values))
        if not single_stage:
            pp.broadcast(token, src=1)
        self._debug_split("sampler")
        token_host.copy_(token, non_blocking=True)
        done = torch.hpu.Event()
        done.record()
        done.synchronize()
        if sent is not None:
            sent.wait()
        outputs = token_host[:len(requests)].tolist()
        self.model.complete_request_batch([1] * len(requests))
        self.bank.finish(generation, done)
        self.last_done = done
        self.runner.pp.commits = getattr(self.runner.pp, "commits", 0) + 1
        for request, output in zip(requests, outputs):
            request.output.extend(output)
        self.runner.audit["batch_decode_steps"] = self.runner.audit.get("batch_decode_steps", 0) + 1
        self.runner.audit["decode_steps"] += 1
        self.runner.audit["target_steps"] += 1
        self.runner.audit["target_tokens"] += len(requests)
        buckets = self.runner.audit.setdefault("batch_bucket_calls", {})
        buckets[b] = buckets.get(b, 0) + 1
        self.runner.audit["batch_padding_rows"] = self.runner.audit.get("batch_padding_rows", 0) + b - len(requests)
        if not single_stage:
            # Only count a PP payload when one is actually sent; a single-stage
            # deployment neither fills nor transmits the packet.
            self.runner.audit["batch_pp_payload_bytes"] = (self.runner.audit.get("batch_pp_payload_bytes", 0) +
                                                           packet.numel() * packet.element_size())
        ids = [request.req_id for request in requests]
        return (ModelRunnerOutput(
            req_ids=ids, req_id_to_index={
                name: i
                for i, name in enumerate(ids)
            }, sampled_token_ids=outputs) if self._last_stage else None)

    def warmup(self):
        from vllm_gaudi.v1.worker.deepseek_v41_runner import RequestState
        for b in self.buckets:
            # Mark the bucket boundary in the log so a compile failure can be
            # attributed to the graph that produced it: bucket 1 succeeds and
            # bucket 2 fails, and the graph dumps carry no bucket identity.
            logger.info("V4.1 PP%d batch warmup starting bucket %d (capacity %d)", self.model.pp_rank, b,
                        self.capacity)
            # Every row gets the SAME prompt but a DISTINCT page allocation. The
            # graph only needs to compile, and identical inputs on distinct slots
            # are an exact instrument: rows 0..n-1 should agree, so any
            # max|row0-rowi| is batching numerics and nothing else. With one
            # token per row that comparison was unavailable from outside, because
            # real serving never puts two requests at the same position.
            requests = [
                RequestState(f"__batch_warm_{i}", [1], [], None, ([1 + i], )) for i in range(min(b, self.capacity))
            ]
            for _ in range(4):
                self.execute(requests)
                for request in requests:
                    request.num_computed_tokens += 1
            if self.pipeline is not None and b >= 32:
                self.pipeline.require_ready(b // 2)
            else:
                self.model.batch_replay.require_ready(b)
            for request in requests:
                self.bank.release(request.req_id)
                if self.model.engram_host is not None:
                    self.model.engram_host.release_request(request.req_id)
            self.runner.graphed_buckets.add(b)
        if self._pp_stages > 1:
            self.runner.pp.group.barrier()
