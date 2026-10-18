# SPDX-License-Identifier: Apache-2.0
"""A single-stage batch step samples the forward, never the PP packet."""
from types import SimpleNamespace

import pytest
import torch

from vllm_gaudi.v1.worker import deepseek_v41_batch_runner
from vllm_gaudi.v1.worker.deepseek_v41_batch_runner import BatchExecution
from vllm_gaudi.v1.worker.deepseek_v41_runner import RequestState

# Distinct per-owner values, so a packet view still holding the previous step
# cannot be mistaken for a write of this step's forward output.
PACKET_HIDDEN, PACKET_PRE = 3.0, 5.0
FORWARD_HIDDEN, FORWARD_PRE = 7.0, 9.0
SAMPLED_TOKEN = 17


@pytest.fixture(autouse=True)
def host_events(monkeypatch):
    """The step only records and joins one completion event; keep it off the device."""

    class Event:

        def record(self, *args, **kwargs):
            pass

        def synchronize(self):
            pass

    if not hasattr(torch, "hpu"):
        monkeypatch.setattr(torch, "hpu", SimpleNamespace(), raising=False)
    monkeypatch.setattr(torch.hpu, "Event", Event, raising=False)


def buffers(buckets):
    """The production packet layout, on host memory."""
    result = {}
    for b in buckets:
        host = torch.zeros(3, b, dtype=torch.int32)
        meta = torch.zeros_like(host)
        pages = torch.zeros(b, 4, dtype=torch.int32)
        packet = torch.empty(b * 10244, dtype=torch.int32)
        hidden = packet[:b * 10240].view(torch.bfloat16).view(b, 4, 5120)
        pre = packet[b * 10240:].view(torch.float32).view(b, 4)
        token = torch.zeros(b, 1, dtype=torch.int32)
        token_host = torch.zeros(b, 1, dtype=torch.int32)
        result[b] = (host, meta, pages, packet, hidden, pre, token, token_host)
    return result


class Bank:
    """The page table and slot lifecycle the metadata frame indexes into."""

    def __init__(self, rows=8):
        self.pages = torch.zeros(rows, 4, dtype=torch.int32)
        self.owners = []

    def acquire(self, req_id):
        owner = SimpleNamespace(index=len(self.owners), generation=1, req_id=req_id, blocks=None)
        self.owners.append(owner)
        return owner

    def publish_pages(self, owner, blocks, capacity):
        assert capacity == 32
        owner.blocks = list(blocks)

    def begin(self, owners):
        return len(owners)

    def finish(self, generation, done):
        pass


def request(req_id, temperature=0.0):
    """One request with a single committed input and one scheduler page group."""
    return RequestState(req_id, [100, 101], [], SimpleNamespace(temperature=temperature), ([1, 2], ), 1)


def execution(*, stages=1, buckets=(1, 2)):
    """A BatchExecution holding only what one _execute step reads."""
    sampled = []
    model = SimpleNamespace(pp_rank=0,
                            program=SimpleNamespace(length=1048576),
                            returned=None,
                            complete_request_batch=lambda counts: None)

    def forward(ids, positions, slots, pages, spans):
        rows = len(positions)
        model.returned = {
            "hidden_states": torch.full((rows, 4, 5120), FORWARD_HIDDEN, dtype=torch.bfloat16),
            "pre_mix": torch.full((rows, 4), FORWARD_PRE, dtype=torch.float32),
        }
        return model.returned

    def sample(value, requests=None):
        sampled.append(value)
        return torch.full((value.shape[0], 1), SAMPLED_TOKEN, dtype=torch.int32)

    model.forward_request_batch = forward
    bank = Bank()
    model.batch_state = bank
    group = SimpleNamespace(ranks=list(range(stages)), is_last_rank=stages == 1, device_group=object())
    runner = SimpleNamespace(pp=SimpleNamespace(group=group, commits=0),
                             state=SimpleNamespace(blocks=32),
                             audit={
                                 "decode_steps": 0,
                                 "target_steps": 0,
                                 "target_tokens": 0
                             },
                             trace_enabled=False,
                             _sample_requests=sample)
    batch = object.__new__(BatchExecution)
    batch.runner, batch.model, batch.bank = runner, model, bank
    batch.capacity = 32
    batch.buffers = buffers(buckets)
    batch.samplers = dict.fromkeys(buckets, sample)
    batch._pp_stages = stages
    batch._last_stage = stages == 1
    batch.pipeline = None
    batch.generation = 0
    batch.last_done = None
    return batch, sampled


def forbid_transport(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("a single stage sent or received a PP packet")

    monkeypatch.setattr(deepseek_v41_batch_runner, "dist", SimpleNamespace(isend=fail, irecv=fail))


@pytest.mark.parametrize("bucket", [1, 2])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_single_stage_batch_step_skips_the_packet_copies(bucket, temperature, monkeypatch):
    batch, sampled = execution(stages=1, buckets=(bucket, ))
    forbid_transport(monkeypatch)
    host, meta, pages, packet, hidden, pre, token, token_host = batch.buffers[bucket]
    hidden.fill_(PACKET_HIDDEN)
    pre.fill_(PACKET_PRE)
    packet_before = packet.clone()
    requests = [request(name, temperature) for name in ("a", "b")[:bucket]]
    result = batch._execute(requests)
    # Packing the stage output into offset views of the transport packet is
    # dead work without a pipeline peer: nothing reads those views, and the two
    # copies rejected the b=2 graph. They must not happen at all.
    assert torch.equal(packet, packet_before)
    assert torch.all(hidden == PACKET_HIDDEN) and torch.all(pre == PACKET_PRE)
    # The token is still taken from the forward's own output tensor, not a view
    # of the packet, and no PP payload is packed or accounted.
    assert len(sampled) == 1 and sampled[0] is batch.model.returned["hidden_states"]
    assert sampled[0].data_ptr() != packet.data_ptr()
    assert "batch_pp_payload_bytes" not in batch.runner.audit
    assert result.sampled_token_ids == [[SAMPLED_TOKEN]] * bucket
    assert [r.output for r in requests] == [[SAMPLED_TOKEN]] * bucket
    assert batch.runner.audit["batch_bucket_calls"] == {bucket: 1}


def test_two_stage_batch_step_still_packs_and_sends_the_packet(monkeypatch):
    """The guard is a predicate on the deployment, not a removed copy."""
    batch, sampled = execution(stages=2, buckets=(1, ))
    sent, broadcast = [], []
    monkeypatch.setattr(
        deepseek_v41_batch_runner, "dist",
        SimpleNamespace(isend=lambda packet, dst, group: sent.append(dst) or SimpleNamespace(wait=lambda: None)))
    batch.runner.pp.group.broadcast = lambda token, src: broadcast.append(src)
    host, meta, pages, packet, hidden, pre, token, token_host = batch.buffers[1]
    hidden.fill_(PACKET_HIDDEN)
    pre.fill_(PACKET_PRE)
    batch._execute([request("a")])
    assert torch.all(hidden == FORWARD_HIDDEN) and torch.all(pre == FORWARD_PRE)
    assert sent == [1] and broadcast == [1]
    assert batch.runner.audit["batch_pp_payload_bytes"] == packet.numel() * packet.element_size()
    # Sampling stays on the stage that receives the packet.
    assert not sampled
