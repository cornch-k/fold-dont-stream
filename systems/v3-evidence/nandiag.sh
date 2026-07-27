#!/bin/bash
set -u
S=/private/tmp/claude-501/-Users-angigyeom-Desktop-DEV-optillama/525ff5d7-3e78-403d-9ac0-15e30e1a2ad7/scratchpad
D=/Users/angigyeom/Desktop/DEV/optillama/llama.cpp
LAG=/Users/angigyeom/Desktop/DEV/optillama/models/Laguna-S-2.1-GGUF/UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf
L=48; m=(); m[0]=0
for ((il=1; il<L; il++)); do m[il]=$(( 1 + ((il-1)/3)*3 )); done
for v in 45 46 47; do m[$v]=$v; done
IFS=,; MAP="${m[*]}"; unset IFS
CFG="LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA=0.5 LLAMA_FOLD_MAP=$MAP"
GPU="LLAMA_MMAP_GPU_COPY=1 LLAMA_FOLD_SKIP_LOAD=1"
A="-f $S/wikitext-2-raw/wiki.test.raw --chunks 2 -c 512 --no-repack"

echo "=== ppl 배치 스윕 (Metal, -ngl 99) — CPU 기준 27.5888 ==="
for bu in "512 512" "512 128" "128 128" "64 64" "1 1"; do
  set -- $bu
  printf '  -b %-4s -ub %-4s ' "$1" "$2"
  env $CFG $GPU "$D/build-fold/bin/llama-perplexity" -m "$LAG" $A -b $1 -ub $2 -ngl 99 2>&1 \
    | grep -aoE '\[2\][0-9.]+|\[2\]nan|error[^\n]{0,40}' | tail -1; echo
done
echo
echo "=== 부분 오프로드 (-b 512) ==="
for g in 24 40; do
  printf '  -ngl %-3s ' "$g"
  env $CFG $GPU "$D/build-fold/bin/llama-perplexity" -m "$LAG" $A -b 512 -ngl $g 2>&1 \
    | grep -aoE '\[2\][0-9.]+|\[2\]nan|error[^\n]{0,40}' | tail -1; echo
done
