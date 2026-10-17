# Bug report draft — fused dense-attention output op is wrong when compiled

Status: **confirmed on our hardware**, fix is a workaround (two flags), not a root-cause fix.
Audience: 1Cat devs (vLLM-Gaudi fork).

## Symptom

Serving DeepSeek-V4.1-Flash, TP4 x PP1 on 4x Gaudi2 (24 GB cards, modules 0-3), vLLM
`0.28.1rc1.dev392+ge47aa780b`, every served completion was non-finite: all 20480 values of the
sampled row were NaN/Inf, so generation produced EOS immediately or degenerate text. Prefill numbers
looked fine and every eager component test passed.

Diagnostic that localised it: a probe on `V41V2ModelRunner._forward` printed, per request,
`tokens=<n> start=<pos> finite= nan= inf= numel= absmax=`. Warmup `_dummy_run` forwards
(`tokens>=128`, `native=False`) were finite. Only the **one-token** forwards (`tokens=1`,
`native=True`) were non-finite, and `numel=5120` with `absmax=nan` means one bad element had already
spread across the whole row by the time it was observed.

## Localisation

- Correlates exactly with `attention.compress_ratio == 0`, i.e. **layers 0 and 1 only** (of 40).
  Ratio>0 layers have `decoded == True`, so `_can_fuse_mla_woa` is true and they take
  `mla_selected_woa_wob_fp8_roundtrip_gaudi2`; they never enter `_finish_output` and are always finite.
- The construct is `custom_deepseek_v41_rope_woa_wob_roundtrip_fp8_gaudi2`, the count-1-only fused
  **output** handoff in `ops/deepseek_v41_paged_attention.py:1636-1644`, recipe
  `hpu_dsv41_woa_dense_quant_pt2.cpp:85-104`. At `shape[0] == 1` three output ops collapse into it:
  the inverse `_rope`, `project_output`, and `project_output_consumer`.
- **Correct eagerly, wrong compiled.** Every eager component gate we ran on this path came back
  clean, which is why the hunt took as long as it did. This is the important part of the report: an
  eager test of this op does not detect the defect.
- Layer 0 is first, so one bad element reaches all 5120 columns and then all 40 layers.

## Workaround that works

```bash
VLLM_HPU_DSV41_WOA_OUTPUT_ROUNDTRIP=0
VLLM_HPU_DSV41_PREFILL_OUTPUT_PROJECTION=0   # its validator requires the roundtrip
```

Both flags touch only the dense (`compress_ratio == 0`) path, so 2 of 40 layers lose the fusion.
Measured after the change at 32768 max model len / 8192 max batched tokens:

- 6040/6040 one-token forwards finite (was 20480/20480 non-finite).
- 1273-word Roman history, 1633 tokens, sustained 43.2 tok/s decode, ITL p50 22.9 ms, zero repetition.
- Needle retrieval from a 10,044-token prompt; prefill 2157 tok/s.
- Two greedy runs byte- and token-id-identical.

## What we could not determine

- Which of the three fused ops is at fault, and whether the fault is in the compiled recipe, in the
  fusion's buffer/scale handoff, or in the count-1 specialisation of the recipe. We only proved the
  fused construct is the carrier: disabling it (and nothing else) makes the output finite.
- Whether the failure is present at `shape[0] == 1` for other dtypes/scales on the same recipe.

## Suggested fix direction

Either make the count-1 fused path eager-equivalent, or refuse the fusion at count 1 until it is
fixed — the latter is what our two flags do, and it costs little because it affects 2 of 40 layers.
An eager-vs-compiled differential test for this op would have caught it; we recommend adding one.
