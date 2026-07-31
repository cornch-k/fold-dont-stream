#!/bin/bash
set -u
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
m=(); for ((il=0; il<48; il++)); do
  if (( il >= 44 )); then m[il]=$il; else g=$(( il * 6 / 44 )); m[il]=$(( (g * 44 + 5) / 6 )); fi
done
IFS=,; MAP6="${m[*]}"; unset IFS
rm -rf heal/ref && mkdir -p heal/ref
env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
    LLAMA_FOLD_MAP="$MAP6" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=exp/g46.txt \
    LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
    LLAMA_DEBUG_SAVE_DIR=heal/ref \
    nanfix/build/bin/llama-debug -m models/qwen35-v2a-hot.gguf \
    -p "The quick brown fox jumps over the lazy dog" -ngl 99 --no-repack -c 512 -b 512 \
    --override-kv qwen35moe.expert_used_count=int:6 \
    --tensor-filter 'linear_attn_out|attn_residual|model.input_embed'
