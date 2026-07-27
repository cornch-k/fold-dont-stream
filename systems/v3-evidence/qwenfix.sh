#!/bin/bash
set -u
S=/private/tmp/claude-501/-Users-angigyeom-Desktop-DEV-optillama/525ff5d7-3e78-403d-9ac0-15e30e1a2ad7/scratchpad
D=/Users/angigyeom/Desktop/DEV/optillama/llama.cpp
LAG=/Users/angigyeom/Desktop/DEV/optillama/models/Laguna-S-2.1-GGUF/UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf
QW=/Users/angigyeom/Desktop/DEV/optillama/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf
P="$D/build-cpu/bin/llama-perplexity"
A="-f $S/wikitext-2-raw/wiki.test.raw --chunks 2 -c 512 -b 512 -t 8 -ngl 0 --no-repack"
L=48; m=(); m[0]=0
for ((il=1; il<L; il++)); do m[il]=$(( 1 + ((il-1)/3)*3 )); done
for v in 45 46 47; do m[$v]=$v; done
IFS=,; MAP="${m[*]}"; unset IFS
r(){ printf '%-46s ' "$1"; shift; local mdl="$1"; shift
     env LLAMA_MMAP_NO_PREFETCH=1 "$@" "$P" -m "$mdl" $A 2>&1 \
       | grep -aoE '\[2\][0-9.]+|\[2\]nan|error[^\n]{0,40}' | tail -1; echo; }

echo "### A. Laguna 회귀 — 27.5888 유지되어야 함 ###"
r "laguna MAP+shexp+g0.5" "$LAG" LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA=0.5

echo "### B. Qwen3.5 — 두 값이 달라야 수정이 먹은 것 ###"
r "qwen R=4, shexp 물리(기본)" "$QW" LLAMA_FOLD_EXPERTS=4
r "qwen R=4, shexp 논리"       "$QW" LLAMA_FOLD_EXPERTS=4 LLAMA_FOLD_SHEXP_LOGICAL=1
