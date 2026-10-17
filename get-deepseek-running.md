# Running DeepSeek V4.1 Flash on 1Cat-vLLM-Gaudi

Target: serve `DeepSeek-V4.1-Flash` on **4×Gaudi2 HL-225**, profile **TP4×PP1**,
one request, greedy ordinary-C1 decode (the currently quality-qualified profile,
default 512-token context, configurable up to 1,048,576 tokens).

This repo is the plugin + tooling. It does **not** run the model out of the box;
DeepSeek V4.1 needs a separately-built engine, native replay runtime, prepared
weights and FP8 sidecars. Everything below is built from the pinned public
commits recorded in this repo's patch manifests.

> Status caveat: per `docs/1cat_gaudi_guide.md`, the installable V4.1 engine lock
> is not fully archived. This is a research/experimental path built from public
> pinned commits; ABI match does not imply compiler or numerical equivalence.

## Prerequisites already present

| Requirement | Status |
|---|---|
| 4× Gaudi2 HL-225, driver 1.24.1 / fw 1.24.0 | present |
| Gaudi PyTorch 2.11 + `habana_frameworks` 1.24.1 | `~/Envs/onecat/venv` |
| Source HF checkpoint `deepseek-ai/DeepSeek-V4.1-Flash` | `~/.cache/huggingface/hub` (475 GiB) |
| vLLM engine (patched) | to build |
| V4.1 native kernels / host gather | to build |
| Patched Bridge / Synapse / HCL + TP2 bridge extension | to build |
| Prepared shards + Engram manifests | to prepare |
| FP8 sidecars (`wo_a_fp8`, `attention_dense_fp8`) | to prepare |
| Runtime profile | to record |

## Artifact layout

| Role | Path | FS / size |
|---|---|---|
| Repo | `/home/nat/Envs/onecat/1Cat-vLLM-Gaudi` | /home |
| External source checkouts | `/home/nat/Envs/onecat/src/{vllm-dsv41,gaudi-pytorch-bridge,synapse,hcl}` | /home, ~5–15 GiB |
| TP2 bridge extension | `/home/nat/Envs/onecat/src/tp2-ext` | /home, small |
| Native build trees + kernel libs | `/home/nat/Envs/onecat/build/dsv41-native/{build,host,lib}` | /home, ~10–50 GiB |
| Runtime profile | `/home/nat/Envs/onecat/build/dsv41-native/verified-runtime-profile.json` | /home, tiny |
| Prepared rank files + sidecars | `~/Models/1Cat/DeepSeek-V4.1-Flash-prepared` | /data, 342 GiB (+~4) |
| N256 runtime cache (optional, 1M ctx) | `~/Models/1Cat/DeepSeek-V4.1-Flash-n256` | /data, 307 GiB |
| Source checkpoint (read-only) | `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a…` | /data, 475 GiB |
| HPU recipe cache | `~/.cache/habana/recipe-cache` (`PT_HPU_RECIPE_CACHE_CONFIG`) | /data |
| Torch/Inductor/Triton/pip caches | `~/.cache/{torch,torchinductor,triton,pip}` | /data |
| Run evidence / device locks | `~/.cache/1cat/{evidence,locks}` | /data |
| In-repo (gitignored) | repo `build/`, `vllm_gaudi/lib` | /home, small |

`~/Models` → `/data/nat/Models`; `~/.cache` → `/data/nat/.cache`. All large or
transient data therefore lands on `/data/nat` (2.4 TiB free). Code and build
trees stay on `/home` (838 GiB free).

## Environment

Work from the repo directory; activate the parent venv.

```bash
cd ~/Envs/onecat/1Cat-vLLM-Gaudi
source ~/Envs/onecat/venv/bin/activate
source ~/Envs/onecat/vendor/habanalabs-env.sh

# keep every cache off /home
export HF_HOME=~/.cache/huggingface
export XDG_CACHE_HOME=~/.cache
export TORCHINDUCTOR_CACHE_DIR=~/.cache/torchinductor
export TRITON_CACHE_DIR=~/.cache/triton
export PT_HPU_RECIPE_CACHE_CONFIG=~/.cache/habana/recipe-cache
```

## Phase 0.5 — Disk-budget preflight (before any generation)

- `/data/nat` free ≥ prepared payload + 8 GiB reserve. Real budget: **342 GiB**
  prepared; **307 GiB** more only if doing the optional N256 long-context cache.
- `/home` free ≥ ~50 GiB for `../src` + `../build`.
- Source snapshot complete: 48 `model-*.safetensors`, index reports 96085
  tensors / 48 shards.
- 4 modules free (`hl-smi`); `torch.hpu.device_count() == 4`.
- **Never clear `~/.cache/huggingface`**: Engram host tables are mmap'd from the
  source shards at serve time even though the path is a cache.

## Phase 1 — Patch & install the vLLM engine

```bash
git clone https://github.com/vllm-project/vllm.git ~/Envs/onecat/src/vllm-dsv41
git -C ~/Envs/onecat/src/vllm-dsv41 checkout e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba
python tools/prepare_deepseek_v41_engine.py ~/Envs/onecat/src/vllm-dsv41 --check
python tools/prepare_deepseek_v41_engine.py ~/Envs/onecat/src/vllm-dsv41      # apply serving delta
git -C ~/Envs/onecat/src/vllm-dsv41 apply tools/communication/patches/dsv41-v2-platform.patch

VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e ~/Envs/onecat/src/vllm-dsv41
pip install --no-build-isolation -e .
python -c "import vllm, vllm_gaudi; print(vllm.__version__)"
```

## Phase 2 — Native replay runtime + TP2 bridge

Per `tools/communication/patches/native-runtime/README.md`, using clean
checkouts at the pinned commits (Bridge `5176b1b2…`, Synapse `6de1f66a…`,
HCL `11f9114b…`):

```bash
python tools/communication/apply_native_runtime.py bridge  --source ~/Envs/onecat/src/gaudi-pytorch-bridge --apply
python tools/communication/apply_native_runtime.py synapse --source ~/Envs/onecat/src/synapse --apply
python tools/communication/apply_native_runtime.py hcl     --source ~/Envs/onecat/src/hcl --apply
python tools/communication/apply_native_program_window.py --source ~/Envs/onecat/src/synapse --apply
```

Build HCL, then Synapse (Release, tests off), then Bridge; build the isolated
eager-concat cache-key adapter if not rebuilding the frontend. Then build the
TP2 fused AR/Norm bridge extension with `build_tp2_fused_ar_norm_bridge.py`
(producing its `.abi.json` sidecar).

## Phase 3 — V4.1 kernels + host gather

```bash
export GAUDI_PYTORCH_BRIDGE_ROOT=~/Envs/onecat/src/gaudi-pytorch-bridge
export GAUDI_PYTORCH_BRIDGE_BUILD_ROOT=~/Envs/onecat/build/gaudi-pytorch-bridge
python tools/build_deepseek_v41.py \
  --build-root ~/Envs/onecat/build/dsv41-native/build \
  --output-dir ~/Envs/onecat/build/dsv41-native/lib \
  --jobs "$(nproc)"
export VLLM_HPU_DSV41_NATIVE_LIBRARY_DIR=~/Envs/onecat/build/dsv41-native/lib
```

Expect `libdeepseek_v4_gaudi2_kernels.so`, `hpu_dsv4_sparse_attn_pt2*.so`,
`dsv41_host_gather*.so`, `deepseek_v4_build.json`, `deepseek_v41_build.json`.

## Phase 4 — Checkpoint audit & prepared shards

```bash
SRC=~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
python tools/audit_deepseek_v41_upstream.py ~/.cache/1cat/evidence/upstream
python tools/sync_deepseek_v41_checkpoint.py "$SRC" --evidence ~/.cache/1cat/evidence/checkpoint

python tools/prepare_deepseek_v41_shards.py "$SRC" ~/Models/1Cat/DeepSeek-V4.1-Flash-prepared \
  --checkpoint-audit ~/.cache/1cat/evidence/checkpoint/complete.json \
  --upstream-lock ~/.cache/1cat/evidence/upstream/upstream-lock.json \
  --tensor-parallel-size 4 --pipeline-parallel-size 1 --plan-only

python tools/prepare_deepseek_v41_shards.py "$SRC" ~/Models/1Cat/DeepSeek-V4.1-Flash-prepared \
  --checkpoint-audit ~/.cache/1cat/evidence/checkpoint/complete.json \
  --upstream-lock ~/.cache/1cat/evidence/upstream/upstream-lock.json \
  --tensor-parallel-size 4 --pipeline-parallel-size 1
```

Keep the original Engram source files; rank files do not contain the full host
tables. A published prepared checkpoint is immutable.

## Phase 5 — Mandatory FP8 sidecars

```bash
python tools/prepare_deepseek_v41_woa_fp8.py   ~/Models/1Cat/DeepSeek-V4.1-Flash-prepared
python tools/prepare_deepseek_v41_dense_fp8.py ~/Models/1Cat/DeepSeek-V4.1-Flash-prepared
```

Outputs default to `…-prepared/sidecars/{wo_a_fp8,attention_dense_fp8}`. The
entrypoint validates and discovers both before model loading.

## Phase 6 — Runtime profile & machine settings

Record the verified environment and fingerprint every library/config, then
generate `~/Envs/onecat/build/dsv41-native/verified-runtime-profile.json` per
`docs/1cat_gaudi_guide.md#runtime-profile`. Adapt
`docs/configuration/examples/deepseek_v41_tp4_machine.json` to this host:
`HABANA_VISIBLE_MODULES=0,1,2,3` and CPU pools mapped to the two NUMA nodes.

## Phase 7 — Launch & validate

```bash
python tools/run_deepseek_v41.py \
  --lock-dir ~/.cache/1cat/locks \
  --runtime-profile ~/Envs/onecat/build/dsv41-native/verified-runtime-profile.json \
  --modules 0,1,2,3 \
  ~/.cache/1cat/evidence/run \
  -- \
  python -m vllm_gaudi.entrypoints.deepseek_v41 \
    ~/Models/1Cat/DeepSeek-V4.1-Flash-prepared \
    --checkpoint-audit ~/.cache/1cat/evidence/checkpoint \
    --host 127.0.0.1 --port 8000
```

Defaults: TP4×PP1, 512 context, greedy C1, V2 async, numeric fastpaths, DSpark off.

```bash
curl --fail --show-error http://127.0.0.1:8000/health
curl --fail --show-error --no-buffer http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"Hello"}],"temperature":0,"max_tokens":64}'
python -m pytest -q tests/unit_tests/test_deepseek_v41_entrypoint.py
```

Long context (capacity only, not quality-qualified), optional:

```bash
python tools/prepare_deepseek_v41_n256.py \
  ~/Models/1Cat/DeepSeek-V4.1-Flash-prepared \
  ~/Models/1Cat/DeepSeek-V4.1-Flash-n256
# append to the deepseek_v41 command:
# --n256-prepared-dir ~/Models/1Cat/DeepSeek-V4.1-Flash-n256
# --max-model-len 1048576 --max-num-batched-tokens 8192 --max-num-seqs 1
# --block-size 128 --num-gpu-blocks-override 8193
```

Optionally package a clean deployment with
`tools/install_deepseek_v41_runtime.py` + `tools/serve_deepseek_v41.py`.

## Risks / notes

- Source checkpoint must stay present and readable (Engram mmap); never delete it.
- `MODEL_DIR` is the read-only HF snapshot; verify the sync/audit step adopts the
  already-complete files instead of re-downloading 475 GiB.
- Build roots must be overridden to `../build/…`; do not let native trees land on
  `/home` outside `/home/nat/Envs/onecat`.
- All `tools/` scripts import `vllm` / `vllm_gaudi`; always run them from the
  activated Gaudi venv with the patched engine + plugin installed.
- Preparing shards requires payload space + an 8 GiB reserve; preparation uses a
  2 GiB per-process temporary workspace.
