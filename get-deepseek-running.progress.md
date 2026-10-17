# DeepSeek V4.1 build — progress notes and deviations

Executed on 2026-10-01, host `nat` (2× Xeon 6338N, 128 threads, 4× Gaudi2 HL-225).
Repo checkout: `/home/nat/Envs/onecat/1Cat-vLLM-Gaudi` (commit 25b87d5).
Venv: `/home/nat/Envs/onecat/venv` (Python 3.12.3, torch 2.11.0a0+git009b5f6,
habana_frameworks 1.24.1, driver 1.24.1 / fw 1.24.0).

## Environment fixes
- `PYTHONPATH` contained a bogus `/root` entry which made pip crash scanning
  `/root`. Cleaned to `PYTHONPATH=/usr/lib/habanalabs/` for all pip/build steps.
- `tpc-clang` lives at `/opt/habanalabs/bin/tpc-clang` (not on default PATH).
- Installed CMake 3.31.6 into the venv for the Synapse tree; the system CMake
  4.4.0 cannot configure the bundled protobuf-3.9.x. Use
  `/home/nat/Envs/onecat/venv/bin/cmake` for native builds.

## Phase 1 (engine + plugin) — DONE
- vLLM at `e47aa780…` ("[Model] Support DeepSeek-V4.1-Flash");
  `tools/prepare_deepseek_v41_engine.py` serving delta applied.
- `VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .`
  → vllm 0.28.1rc1.dev392+ge47aa780b.
- Plugin editable install: vllm_gaudi 0.1.dev973+g25b87d5e3.
- HPU plugin registers; `torch.hpu.device_count()==4`; HPU still healthy.

### Deviation: skipped `dsv41-v2-platform.patch`
The serving-engine delta already contains the full V2 platform hook
(`Platform.supports_v2_without_triton` + its call site). The standalone patch
has stale parent hashes and does not apply. The plugin overrides the hook to
True at `vllm_gaudi/platform.py:622`, gated on `uses_v2(config)`. Correct.

## Phase 2a (HCL + Synapse) — DONE
Sources at pinned commits under `/home/nat/Envs/onecat/src/`:
bridge `5176b1b2…`, synapse `6de1f66a…`, hcl `11f9114b…`. Native-runtime patches
applied via `apply_native_runtime.py` (all three) plus
`apply_bounded_reindex_runtime.py` and `apply_native_program_window.py`.

### Deviation: skipped `synapse-tpc-fuser-state.patch`
`synapse.patch` already implements the same per-invocation TPC-fuser state fix
(passes `nodesToAdd`/`totalOriginalNodesSet` as parameters plus an atomic
cluster counter). The standalone patch has stale context (its "before" hash is
the pristine HEAD) and does not apply. Functionally superseded.

### Build procedure
Order: hl_logger, hl_gcfg, scal(SDK), mme, synapse.
```
cd src/synapse
source ./habana_set_env_val.sh ./ /home/nat/Envs/onecat/build/synapse
export PATH=/home/nat/Envs/onecat/venv/bin:$PATH
build_hl_logger -r ; build_hl_gcfg -r     # -> build/synapse/latest/libhl_*.so
cmake -S $SYNAPSE_ROOT -B build/synapse/synapse-release \
  -DCMAKE_BUILD_TYPE=Release -DTESTS_ENABLED=OFF -DAUTO_GENERATED_TESTS=OFF \
  -DSYNAPSE_DFA_REGISTER_DUMP=OFF
make -j$(nproc)
```
HCL built from patched source:
```
HCL_SRC_PKG_DIR=/home/nat/Envs/onecat/src/hcl HCL_LIB_DIR=/usr/lib/habanalabs \
  cmake -S src/hcl/hcl/src -B build/hcl -DCMAKE_BUILD_TYPE=Release && make -j
```
`build/synapse/latest/` must contain: libhcl.so (patched, built), libhl_logger.so,
libhl_gcfg.so (built), plus SDK libscal.so / libTpcElfReader.so / libtpcsim_shared.so
/ libhl-thunk.so and the built libSynapseMme.so.

### Deviation: `SYNAPSE_DFA_REGISTER_DUMP=OFF`
The public snapshot ships 0-byte `specs/{gaudi,gaudi2}/all_regs.txt` and
`specs_h9/gaudi3/all_regs.txt`. The CMake option is designed for research builds
and generates empty register-dump stubs. DFA register scans are device-failure
analysis only; no inference math depends on them.

### Deviation: HCL include set
Synapse compiles against the SDK stub headers under `src/synapse/hcl/include`
(they carry `internal/hccl_api_wrapper.inl`, absent from the public HCL repo).
`libhcl.so` itself is built from the patched `src/hcl` tree (adds
`hccl_tp2_native_graph.h` etc.). The Synapse/HCL ABI is therefore a mixed
patched-lib + SDK-header build, not the archived internal combination.

Outputs: `build/synapse/synapse-release/lib/libSynapse.so`,
`lib/libGraphCompilerPlugin.so`, `build/hcl/lib/libhcl.so`.

## Phase 2b (Gaudi PyTorch Bridge) — BLOCKED: SDK/bridge header-version skew

Bridge source at commit `5176b1b2d` builds ~840 targets but cannot be linked
into a consistent `libhabana_pytorch_backend.so`. Two incompatible header sets
are available and neither matches the bridge commit exactly:

1. Installed SDK headers `/usr/include/habanalabs` (1.24.1). With these the
   bridge compiles fully, but the link fails with undefined references:
   - `hl_logger::v12_inline_fmt_compile::moduleLoggerData<HlLogger::LoggerType>`
     / `::levels` / `getLogger`
   - `towl::impl::TowlEnabled::flag`, `towl::impl::emitCopyFinished`,
     `emitCopyMultipleFinished`
   - `Logger::formatStatusMsg(synStatus)`
   The installed `/usr/lib/habanalabs/libhl_logger.so` only exports the older
   `hl_logger::v1_0` API, and the backend target does not link the libs
   (`libSynapse.so`, `libscal.so`, `libswtools_sdk.so`) that do define the
   `v12_inline_fmt_compile`/`towl` symbols.

2. Tree snapshot headers (`src/synapse/{swtools_sdk,synapse,hcl,specs_external}`).
   These `hl_logger` headers are older (no v12 path) but then the build fails
   earlier:
   - `hl_logger/hllog.hpp` wants `magic_enum-0.8.1/include/...` while the bridge
     fetches magic_enum 0.9.7
   - `syn_sl_api.h` is not in the tree
   - `pytorch_helpers/synapse_shim/hccl_impl.cpp` needs the newer `hccl_api_t`
     signatures (`InternalStreamHandle*`), which the tree's stub HCL headers lack

### Root cause
The bridge commit expects the *internal* Gaudi SDK source/build matrix used by
the archived runtime. The public Intel_Gaudi3_Software snapshot and the
installed 1.24.1 SDK package are each mismatched with it along a different
axis. This is the "matching native runtime" prerequisite the docs say is not
publicly archived (`docs/1cat_gaudi_guide.md:107`).

### What is needed to unblock
- The exact internal SDK/source trees the bridge commit was built against
  (hl_logger with the magic_enum-0.8.1 include layout + newer hccl headers +
  matching synapse headers), OR
- A pre-built, ABI-qualified `libhabana_pytorch_backend.so` and matching bridge
  build tree matching SDK 1.24.1 + torch 2.11, OR
- A bridge commit known to pair with the installed 1.24.1 SDK packaging.

Everything downstream (Phase 2c TP2 extension, Phase 3 plugin kernels, Phases
4-7) depends on a successful bridge build, so they remain blocked.
