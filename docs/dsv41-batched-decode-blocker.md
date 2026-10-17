# Batched decode is unavailable on the current TP4 build

Status: **confirmed by measurement**, 2026-10-17. Blocks throughput scaling beyond one
request at a time; does not block correctness.

## What we wanted

`--max-num-seqs 8` at 262144 context, so eight concurrent requests decode together.

## What actually happens

With `VLLM_HPU_DSV41_BATCH_DECODE=1` the fork fails during warmup, in two successive places.

**1. TP2 peer exchange bound.**

```
RuntimeError: Peer exchange requires a bounded BF16 row and the bound TP group size
```

`_tp_peer_allgather_impl` (`vllm_gaudi/distributed/tp2_fused_ar_norm.py:563`) allows a row of at most
32768 elements unless `VLLM_HPU_DSV41_DSPARK_LARGE_HCCL=1`, which raises the ceiling to 387840 for a
four-rank group. Batched decode gathers `batch_size x hidden = 8 x 5120 = 40960` elements, so it
exceeds the default bound. Setting `DSPARK_LARGE_HCCL=1` gets past this.

**2. TP4 reduction contract.**

```
RuntimeError: TP4 reduction requires bounded activations with distinct storage
```

With the bound raised, warmup proceeds and then fails here. The batched-decode activations do not
satisfy the TP4 fused AR/norm reduction contract. This is a code-level incompatibility, not another
missing flag: the batch path would need the reduction to accept a multi-row, non-distinct-storage
activation, or to be bypassed for batch decode.

## Consequence

`--max-num-seqs 8` still works, and results are **correct**: we verified BS 1, 3 and 8 all produce
token-identical output to the same requests run serially (BS 3 is deliberately odd, since decode
buckets are powers of two and a batch of 3 exercises the padded path). But the engine time-slices
rather than batches, so per-request latency scales as ~1/N and aggregate throughput does not improve.
Concurrency is a capability, not a throughput multiplier, until the reduction contract is extended.

## Related, also unusable with the V2 runner

`VLLM_HPU_DSV41_BATCHED_INPUT_STAGING=1` raises
`Batched input staging requires fused stage I/O` at construction
(`deepseek_v41_runner.py:1045`), because it requires `FUSED_STAGE_IO=1`, and
`deepseek_v41_config.py:85` rejects `FUSED_STAGE_IO` together with `V2_SEGMENTED_PREFIX`, which the
V2 contract requires. So on the V2 path there is currently no way to enable batched input staging at
all.

## Suggested direction

Either extend the TP4 reduction to accept batched activations (and document the row bound that
`DSPARK_LARGE_HCCL` raises), or provide a batch-decode mode that routes around the fused reduction.
An end-to-end test that issues 8 concurrent requests and compares against the serial run would catch
both failure modes; ours does, but only after the flags are set.
