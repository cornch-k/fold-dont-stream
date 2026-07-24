# Environment

Exact environment behind every number in the paper. Two stacks were used: the
local M1 Max machine (all systems measurements, dense fp32 training arm) and
Google Colab (T4/fp16 cross-check, MoE, multi-seed, R-sweep arms).

## Local machine (all §4 measurements; §3 M1 Max/fp32 arm)

| Item | Value |
|---|---|
| Hardware | MacBook Pro, Apple M1 Max, 32 GB unified memory, NVMe SSD (~2 GB/s sustained reads observed) |
| OS | macOS 26.5.2 (build 25F84, Darwin 25.5.0) |
| Shell | zsh (battery scripts use zsh-specific syntax; run with `zsh`, not `bash`) |
| Python | 3.13.5 (Anaconda distribution) |

Python packages (exact versions used):

| Package | Version | Used by |
|---|---|---|
| torch | 2.9.1 | `training/rdepth/` (CPU/MPS not used for timing — training only) |
| numpy | 1.26.4 | training, data prep, `systems/flatten_gguf.py` |
| tokenizers | 0.22.2 | 4k BPE tokenizer (`prepare_data.py`) |
| huggingface_hub | 1.24.0 | TinyStories download (`prepare_data.py`) |
| pytest | 8.3.4 | `training/rdepth/tests/` |
| gguf | 0.19.0 | `systems/flatten_gguf.py` (flattened-twin GGUF construction) |
| matplotlib | 3.10.0 | `paper/gen_figs.py` (figure regeneration) |

## llama.cpp build (§4)

| Item | Value |
|---|---|
| Upstream base | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) commit `67b9b0e7f6ce45d929a4411907d3c48ec719e81c` |
| Patch | `patches/llama.cpp-local-stack.patch` (verified to `git apply --check` cleanly on that commit) |
| Compiler | Apple clang 21.0.0 (clang-2100.1.1.101) |
| CMake | 4.2.3 |
| Backend | CPU only for all reported timings (`-ngl 0`; Metal excluded — see paper Appendix A(4)); `--no-repack` required (Appendix A(1)) |
| Twin experiment base | [Nanbeige/llama.cpp](https://github.com/Nanbeige/llama.cpp) commit `03327d628ad6db847d28a330636ffbd845030e29` + `patches/nanbeige-flatten-norm.patch` |

## Colab arm (§3 T4/fp16 cross-check, MoE, multi-seed, R-sweep)

NVIDIA T4, fp16 autocast. The notebooks in `training/notebooks/` install their
dependencies at runtime (`pip install` cells); package versions were whatever
Colab resolved at run time (July 2026) and were not pinned — the cross-stack
agreement reported in Table 1 is itself the evidence that the results are not
sensitive to this.

## Verifying this repository

```sh
cd training/rdepth
pip install -r requirements.txt
python -m pytest tests          # fresh clone: 14 passed, 2 skipped
                                # (data tests skip until prepare_data.py is run;
                                #  with data present: 16 passed)
python ../../paper/gen_figs.py  # regenerates fig1–fig4 next to the script
```

Model checkpoints (Llama-3.3-70B-Instruct Q4_K_M, Laguna-S-2.1 UD-IQ4_XS,
Nanbeige4.2-3B) are downloaded from Hugging Face and not redistributed here.
Absolute paths inside `systems/` scripts and logs refer to the author's
machine; adjust before running.
