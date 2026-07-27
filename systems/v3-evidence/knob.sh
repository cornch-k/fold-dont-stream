#!/bin/bash
set -u
S=/private/tmp/claude-501/-Users-angigyeom-Desktop-DEV-optillama/525ff5d7-3e78-403d-9ac0-15e30e1a2ad7/scratchpad
D=/Users/angigyeom/Desktop/DEV/optillama/llama.cpp
LAG=/Users/angigyeom/Desktop/DEV/optillama/models/Laguna-S-2.1-GGUF/UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf
L=48; m=(); m[0]=0
for ((il=1; il<L; il++)); do m[il]=$(( 1 + ((il-1)/3)*3 )); done
for v in 45 46 47; do m[$v]=$v; done
IFS=,; MAP="${m[*]}"; unset IFS
P="$D/build-fold/bin/llama-perplexity"
A="-f $S/wikitext-2-raw/wiki.test.raw --chunks 2 -c 512 -b 512 --no-repack -ngl 99"
H="LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 LLAMA_MMAP_GPU_COPY=1 LLAMA_FOLD_SKIP_LOAD=1"
r(){ printf '%-46s ' "$1"; shift; env "$@" 2>&1 | grep -aoE '\[2\][0-9.]+|\[2\]nan|Insufficient Memory' | tail -1; echo; }

echo "손잡이 하나씩 (Metal, -ngl 99, -b 512).  CPU 기준: 42292 / 46.46 / 27.59"
r "MAP만 (고정 3층, 나머지 없음)"        $H LLAMA_FOLD_MAP="$MAP" "$P" -m "$LAG" $A
r "MAP + SHEXP_LOGICAL"                  $H LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 "$P" -m "$LAG" $A
r "MAP + SHEXP + GAMMA=0.5"              $H LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA=0.5 "$P" -m "$LAG" $A
r "EXPERTS=3 + SHEXP (MAP 없음)"         $H LLAMA_FOLD_EXPERTS=3 LLAMA_FOLD_SHEXP_LOGICAL=1 "$P" -m "$LAG" $A
r "EXPERTS=4 + SHEXP"                    $H LLAMA_FOLD_EXPERTS=4 LLAMA_FOLD_SHEXP_LOGICAL=1 "$P" -m "$LAG" $A
