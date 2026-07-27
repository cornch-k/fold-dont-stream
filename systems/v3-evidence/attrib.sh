#!/bin/bash
set -u
S=/private/tmp/claude-501/-Users-angigyeom-Desktop-DEV-optillama/525ff5d7-3e78-403d-9ac0-15e30e1a2ad7/scratchpad
D=/Users/angigyeom/Desktop/DEV/optillama/llama.cpp
LAG=/Users/angigyeom/Desktop/DEV/optillama/models/Laguna-S-2.1-GGUF/UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf
NB=/Users/angigyeom/Desktop/DEV/optillama/models/Nanbeige4.2-3B-GGUF/nanbeige4.2-3b-Q4_K_M.gguf
P="$D/build-fold/bin/llama-perplexity"
A="-f $S/wikitext-2-raw/wiki.test.raw --chunks 2 -c 512 --no-repack"
BASE="LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999"
r(){ printf '%-48s ' "$1"; shift; env "$@" 2>&1 | grep -aoE '\[2\][0-9.]+|\[2\]nan|Insufficient Memory' | tail -1; echo; }

echo "=== 복구 손잡이 없이, 가장 단순한 접기 (Metal -ngl 99 -b 512) ==="
r "EXPERTS=2, 손잡이 없음"   $BASE LLAMA_FOLD_EXPERTS=2 LLAMA_MMAP_GPU_COPY=1 LLAMA_FOLD_SKIP_LOAD=1 "$P" -m "$LAG" $A -b 512 -ngl 99
r "EXPERTS=4, 손잡이 없음"   $BASE LLAMA_FOLD_EXPERTS=4 LLAMA_MMAP_GPU_COPY=1 LLAMA_FOLD_SKIP_LOAD=1 "$P" -m "$LAG" $A -b 512 -ngl 99
echo
echo "=== 대조: 접기·해킹 없는 소형 모델 Metal 배치 ==="
r "Nanbeige 3B, ngl 99, b 512"  $BASE "$P" -m "$NB" $A -b 512 -ngl 99
r "Nanbeige 3B, ngl 0 (CPU)"    $BASE "$P" -m "$NB" $A -b 512 -ngl 0 --device none
