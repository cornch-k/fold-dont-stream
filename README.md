# Fold, Don't Stream

Artifacts for the preprint:

> **Fold, Don't Stream: Recurrent Depth as a Storage-Residency Optimization for Consumer-Device LLM Inference**
> Gigyeom Ahn, 2026 — preprint draft v3 (adds Metal execution of folded models and a measured ceiling on training-free folded quality; revised after a four-lens adversarial review)

Zenodo record: [doi:10.5281/zenodo.21534339](https://doi.org/10.5281/zenodo.21534339) (concept DOI — always resolves to the latest version). Version DOIs: v3 [10.5281/zenodo.21618186](https://doi.org/10.5281/zenodo.21618186), v2.2 [10.5281/zenodo.21534340](https://doi.org/10.5281/zenodo.21534340). The English PDF is authoritative; the Korean PDF is a translation provided for accessibility.

A dense 70B model that exceeds a 32 GB laptop's RAM decodes at ~0.02 tokens/s because every token re-reads 40–44 GB of weights from SSD. A ~300-line graph-folding patch to llama.cpp (`LLAMA_FOLD_K`) makes a looped model's *unique* weights resident and removes that cliff on the same laptop (~120× vs the untuned lazy-mmap streaming baseline, dense 70B). Controlled 15M-parameter studies (30.4 MB fp16 resident) measure what looping costs in quality at equal resident weights and equal training tokens, and a *flattened twin* of a production looped model (Nanbeige4.2-3B) isolates residency as the sole cause of a ~150× speed gap. v3 adds two results. Folded models now execute on Metal: a 118B-A8B checkpoint **decodes** at 34.4 tokens/s with 25.2 GB wired, closing the loader-buffer-scoping item v2.2 left open. (Batched prefill in that configuration returns NaN; §4.2 reports the fold-factor ordering behind it as an ordering, not a diagnosis.) And a measured ceiling on what folding an *untrained* checkpoint can be worth: per-token least squares over the physical layer's entire 256-expert set — a strict lower bound on any gating-side repair — leaves 0.93–0.95 of the target residual against a 0.957 chance floor, so its expert basis is barely more useful for its neighbour's routed function than a random subspace, and at the deep layers not usefully better at all.

## Repository layout

```
paper/      HTML sources (en/ko), figures, figure generator, PDF builds
patches/    llama.cpp patches (see below)
training/   §3 quality studies: 15M/42M TinyStories code, tests, training logs,
            Colab notebooks (T4 cross-check, MoE, multi-seed, R-sweep)
systems/    §4 measurement harness: run scripts, mlock ballast, GGUF flattener,
            watchdog, protocol notes, and raw benchmark logs
```

## Patches

| File | What | Base |
|---|---|---|
| `llama.cpp-e2-fold.patch` | The **expert-only** fold of §4.1 and everything §4.5 and Table 3's Metal row depend on: `LLAMA_FOLD_EXPERTS=R` / `LLAMA_FOLD_MAP` fold the FFN block only (router, routed experts, shared expert, post-attention norm) while the mixer keeps its logical layer, plus `LLAMA_FOLD_SHEXP_LOGICAL`, `LLAMA_FOLD_GAMMA[_FILE]`, `LLAMA_FOLD_NORM_LOGICAL`, and the two loader knobs that make Metal execution possible (`LLAMA_FOLD_SKIP_LOAD`, `LLAMA_MMAP_GPU_COPY`). Includes the new `src/models/fold.h`. laguna + qwen35moe architectures. | local stack (see header) |
| `llama.cpp-fold.patch` | The graph-folding patch of §4.1: `LLAMA_FOLD_K=K` folds weight reads to physical layer `il % K` while KV cache, positions, and callbacks stay logical (llama + laguna architectures, SWA-phase assert). | local commit `0966377` (readable excerpt; apply via the full stack below) |
| `llama.cpp-local-stack.patch` | Full local stack: the fold patch plus the §1 "exhausted toolbox" harness (explicit MoE expert cache, pread read-through, prefill expert reduction, gate-ratio skipping, m1max benchmark scripts). Applies to upstream [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at `67b9b0e7f6ce45d929a4411907d3c48ec719e81c`. | upstream `67b9b0e` |
| `nanbeige-flatten-norm.patch` | Loop-boundary RMSNorm applied at a configurable physical stride, restoring exact function equivalence between the folded and flattened forms of Nanbeige4.2-3B (§4.4). Applies to [Nanbeige/llama.cpp](https://github.com/Nanbeige/llama.cpp) at `03327d628ad6db847d28a330636ffbd845030e29`. | Nanbeige fork `03327d6` |

## Reproducing

Exact hardware, OS, compiler, and pinned package versions for every arm are in
[ENVIRONMENT.md](ENVIRONMENT.md).

**Systems (§4).**

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 67b9b0e7f6ce45d929a4411907d3c48ec719e81c
git apply ../patches/llama.cpp-local-stack.patch
cmake -B build && cmake --build build -j
LLAMA_FOLD_K=20 ./build/bin/llama-cli -m <model.gguf> --no-repack ...   # see systems/e1_battery_v6.sh
```

`systems/e1_battery_v6.sh` is the exact battery that produced Table 3's CPU rows (it explicitly excludes Metal); the Metal row's commands and raw output are in `systems/v3-evidence/` (n=3 repetitions, swap guard, page-in accounting, peak-RSS capture). `systems/twin_battery.sh` + `systems/ballast.py` + `systems/flatten_gguf.py` reproduce the flattened-twin experiment of §4.4 (protocol in `systems/flattened-twin-protocol.md`). Checkpoints are pulled from Hugging Face (Llama-3.3-70B-Instruct Q4_K_M; Laguna-S-2.1 UD-IQ4_XS; Nanbeige4.2-3B) and are not redistributed here.

**Quality (§3).**

```sh
cd training/rdepth
pip install -r requirements.txt
python -m pytest tests        # fresh clone: 14 passed, 2 skipped
python prepare_data.py        # TinyStories, 4k BPE
python train.py --run small|loop|large|... [--seed N] [--dtype fp32]
```

`training/rdepth/logs/` holds the per-run CSV/stdout logs behind Tables 1–2. The four notebooks reproduce the Colab T4/fp16 arms: cross-check, MoE policy comparison, n=3 multi-seed replication, and the R-sweep.

Absolute paths inside scripts and logs refer to the author's machine; adjust before running. Raw logs are published unedited for verifiability.

## License

- Code, patches, scripts: [MIT](LICENSE) (llama.cpp itself is MIT).
- Paper content (`paper/`: HTML, PDFs, figures): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

```bibtex
@misc{ahn2026fold,
  author       = {Ahn, Gigyeom},
  title        = {Fold, Don't Stream: Recurrent Depth as a Storage-Residency
                  Optimization for Consumer-Device LLM Inference},
  year         = {2026},
  howpublished = {Preprint, Zenodo},
  version      = {3},
  doi          = {10.5281/zenodo.21534339},
  url          = {https://github.com/cornch-k/fold-dont-stream}
}
```
