# Handoff: why eBay run6 is only ~1.5× over run1

**For:** worker with the GPU machine.
**TL;DR:** The wall is not decode, resolution, or GPU. Offline `llm.chat` HF-preprocesses all N videos **one at a time, single-threaded, in the front-end process, before the GPU does anything**. `renderer_num_workers` does **not** touch that path. The parallel path exists but only the API server uses it. Everything below is the evidence + what to verify on the machine.

---

## 1. The data we analyzed (eBay, N=1000, your numbers)

| step | change | wall | Δ | read |
|---|---|---:|---:|---|
| run1→run2 | `max_pixels` cap | 880.6→865.8 | **−1.7%** | no-op (vs −35% on benchmark ⇒ eBay video already ≤ cap, low-res) |
| run2→run3 | fixed `num_frames=16` | 865.8→792.0 | −8.5% | small |
| run3→run4b | ViT compile/cudagraph | 792.0→795.7 | +0.5% | noise — GPU is not the bottleneck |
| run4b→run5 | `renderer_num_workers=2`, cache off | 795.7→**1211.3** | **+52%** | regression — UNEXPLAINED by source (see §4) |
| run5→run6 | custom PyAV keyframe extract | 1211.3→573.0 | −53% | decode removed |

Net run1→run6 = 880.6/573.0 = **1.54×**.

**TTFT signature (run6):** mean 268 ≈ wall/2, max 572 ≈ wall, mean≈median; E2E mean 0.39 s ≪ TTFT.
⇒ requests complete in a **linear ramp** → throughput-bound by a **serial per-request stage**; GPU (E2E ~0.39 s) is idle ~the entire run. `run3→run6 −28%` bounds *decode* at ≤28% of wall — decode was never the main cost (low-res = cheap decode).

---

## 2. Root cause — GROUNDED in vLLM source (v0.19.1 tag + confirmed on `main`)

Offline `infer.py:180` `llm.chat(...)` path:

1. `LLM.chat` → `vllm/entrypoints/llm.py:918` (v0.19.1) / `:923` (main) → `renderer.render_chat(...)` — **synchronous**.
2. `vllm/renderers/base.py:892-894` (v0.19.1) / `main:992-993`:
   ```python
   eng_prompts = [self.process_for_engine(prompt, arrival_time) for prompt in tok_prompts]
   ```
   Plain serial list comprehension. The full list is built before `render_chat` returns — i.e. **all N videos preprocessed before the engine/GPU gets anything**.
3. `process_for_engine` (`base.py:797`) → `_process_tokens` → `base.py:667` calls `self._process_multimodal(...)` **directly** (sync; the async one at `base.py:720` is unused here).
4. `_process_multimodal` → `base.py:649-650`:
   ```python
   with set_default_torch_num_threads():           # torch intra-op threads clamped
       mm_inputs = mm_processor.apply(...)          # HF Qwen2.5-VL: decode + smart_resize + normalize + patchify
   ```

**`renderer_num_workers`** sizes a thread pool at `base.py:89-90`, wired only to `_process_multimodal_async` (`base.py:104-105`), used **only** by `*_async` paths (`render_chat_async` gather, `base.py:926-928` / `main:1027-1032`) — the **OpenAI API server** path. **Offline `llm.chat` never submits to it.**

Confirmed identical on `main` (not a 0.19.1 artifact). `LLM` docstring (`main llm.py:195`) explicitly says use `AsyncLLMEngine` for serving — offline `LLM` is the simple/sync path by design.

---

## 3. The fix options (ranked)

1. **Use the async path — zero code change.** `vllm serve <model> --renderer-num-workers N` + a **concurrent** client over the eBay set. That path uses `render_chat_async` → the pool. Work is GIL-bound numpy/torch, so also scale **process-level** (`--api-server-count`, data-parallel front-ends), not just threads. *Projected* big win — must be measured (§5.3).
2. **Precompute + inject** (vLLM PR #39502, merged **after** 0.19.1 — backport or do manually): run `mm_processor.apply` in your own `multiprocessing.Pool` offline, persist `MultiModalKwargs`, feed pre-processed inputs. Sidesteps the serial loop entirely.
3. **Fewer frames** (frame sparsification, vLLM #31803 — ideal for static eBay product video). Only lever that helps the pure-serial offline path without changing entrypoint.

---

## 4. Honest caveats / corrections (read before trusting this)

- **Provenance:** I read the released **v0.19.1 tag** and `main`. The machine's build is **patched** (local PR #38997 shim) and the repo `.venv` was empty, so the *exact installed bytes were never read*. Verify §2 against the actual installed `vllm` on the machine.
- **The run5 +52% regression is NOT explained by the source.** An earlier hypothesis (renderer thread-pool contention / #36557 "Already borrowed") is **contradicted**: the offline sync path doesn't use the pool, and #36557's fix is already in the v0.19.1 tag (`base.py:110-122` deep-copies the tokenizer). The tag predicts `renderer_num_workers` is ~inert offline and gives no +52% mechanism. **This is an open question for the machine to resolve (§5.6).**
- All speedup projections here are **reasoned from source, not benchmarked.**

---

## 5. VERIFY ON THE MACHINE (high-value — these are the experiments we couldn't run)

1. **Confirm the no-op knob.** Read the *installed* renderer on the box:
   `python -c "import vllm.renderers.base as b, inspect; print(inspect.getsource(b.BaseRenderer.render_chat))"`
   — confirm the serial list comp is in THEIR build (it's patched). Then `infer.py --preset run6 --workers 1` vs `--workers 8` on identical data: if wall ≈ unchanged ⇒ `renderer_num_workers` confirmed inert for offline `llm.chat`.
2. **Confirm GPU starvation.** `nvidia-smi dmon -s u` during an offline run6 → expect GPU util ≈ near-zero most of the run, one CPU core pinned ~100%.
3. **Measure the real fix (§3.1).** `vllm serve` + `--renderer-num-workers 8` (also try `--api-server-count >1` / data-parallel) driven by a concurrent client (`vllm bench serve` or asyncio client) over the same eBay set. Compare wall/throughput vs offline run6. **This is the decision-maker.**
4. **Profile one `mm_processor.apply`** (py-spy / cProfile on the front-end) → split resize/normalize/rearrange vs demux/keyframe-seek. Resolves which sub-cost dominates the serial step.
5. **Attribute run5's +52%:** re-run run4b→run5 isolating `mm_processor_cache_gb=0` alone vs `renderer_num_workers=2` alone, on the machine, to find what actually caused it (source says it shouldn't).
6. Optional: per-phase decode-vs-HF-processor split in engine logs (decode already bounded ≤28% by run3→run6).

---

## 6. Actions / decision tree

- §5.3 shows big speedup → **switch pipeline to `vllm serve` + concurrent client**; likely no PR needed.
- File an upstream **ISSUE** (not a drive-by PR): *"`renderer_num_workers` is a no-op for offline `LLM.generate`/`chat`"* — with `main` file:line (§2) + the §5.1 repro. Ask maintainers if it's intended **before** any PR. (My upstream search was limited keyword queries — check for an existing tracking issue, don't assume none.)
- If maintainers are receptive → PR adds **bounded** concurrency to the sync render path (reuse `self._executor`, cap in-flight) — **not** a naive `gather` (both paths already materialize all N `MultiModalKwargs` in RAM before generation; unbounded concurrency can OOM large video batches). Note: even `render_chat_async` doesn't pipeline preprocessing with the GPU — the deeper overlap fix is RFC-scale.
- Independent of upstream: pursue §3.2 (precompute+inject).

## 7. Relevant upstream refs

- **#39502** (merged, post-0.19.1) — externally-processed `mm_kwargs` + cache injection → the offline bypass (§3.2).
- **#36557** (merged ~2026-03) — "Already borrowed" VLM-throughput fix; already in v0.19.1 tag (`base.py:110-122`). Confirm in machine build.
- **#38418** (merged) — disallows `renderer_num_workers>1` with mm cache (the guard run5 tripped → `mm_processor_cache_gb=0`).
- **#31803 / #32845** — video frame sparsification for Qwen2.5-VL (§3.3) — best fit for static eBay video.
- **#30839** (RFC) — PyNvVideoCodec zero-copy GPU decode; **its own benchmark = only 2–3% throughput** ⇒ independent proof decode is *not* the bottleneck (corroborates §1).
- **#22070** (open) — HF processing on GPU.
- **#24519** (merged) — InternVL limits CPU threads for image transforms; corroborates the `set_default_torch_num_threads()` clamp at `base.py:649`.
