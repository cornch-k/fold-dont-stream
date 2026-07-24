# Fold, Don't Stream

Artifacts for the preprint:

> **Fold, Don't Stream: Recurrent Depth as a Storage-Residency Optimization for Consumer-Device LLM Inference**
> Gigyeom Ahn, 2026 — preprint draft v2.2 (revised after adversarial review, cross-session literature and log-consistency checks)

Zenodo record: *(DOI to be added at upload)*. The English PDF is authoritative; the Korean PDF is a translation provided for accessibility.

A dense 70B model that exceeds a 32 GB laptop's RAM decodes at ~0.02 tokens/s because every token re-reads 40–44 GB of weights from SSD. A ~300-line graph-folding patch to llama.cpp (`LLAMA_FOLD_K`) makes a looped model's *unique* weights resident and removes that cliff on the same laptop (~125× vs the untuned lazy-mmap streaming baseline, dense 70B). Controlled 30M-parameter studies measure what looping costs in quality at equal resident weights and equal training tokens, and a *flattened twin* of a production looped model (Nanbeige4.2-3B) isolates residency as the sole cause of a 151× speed gap.

## Repository layout

```
paper/      HTML sources (en/ko), figures, figure generator, PDF builds (v2.2)
patches/    llama.cpp patches (see below)
training/   §3 quality studies: 30M/42M TinyStories code, tests, training logs,
            Colab notebooks (T4 cross-check, MoE, multi-seed, R-sweep)
systems/    §4 measurement harness: run scripts, mlock ballast, GGUF flattener,
            watchdog, protocol notes, and raw benchmark logs
```

## Patches

| File | What | Base |
|---|---|---|
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

`systems/e1_battery_v6.sh` is the exact battery that produced Table 3 (n=3 repetitions, swap guard, page-in accounting, peak-RSS capture). `systems/twin_battery.sh` + `systems/ballast.py` + `systems/flatten_gguf.py` reproduce the flattened-twin experiment of §4.4 (protocol in `systems/flattened-twin-protocol.md`). Checkpoints are pulled from Hugging Face (Llama-3.3-70B-Instruct Q4_K_M; Laguna-S-2.1 UD-IQ4_XS; Nanbeige4.2-3B) and are not redistributed here.

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
  howpublished = {Preprint},
  doi          = {TBD (Zenodo concept DOI)},
  url          = {https://github.com/cornch-k/fold-dont-stream}
}
```
