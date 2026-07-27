#!/bin/bash
# γ 수정 후 Metal 재측정. CPU 기준값(같은 그래프) = 27.5888
set -u
S=/private/tmp/claude-501/-Users-angigyeom-Desktop-DEV-optillama/525ff5d7-3e78-403d-9ac0-15e30e1a2ad7/scratchpad
D=/Users/angigyeom/Desktop/DEV/optillama/llama.cpp
LAG=/Users/angigyeom/Desktop/DEV/optillama/models/Laguna-S-2.1-GGUF/UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf
L=48; m=(); m[0]=0
for ((il=1; il<L; il++)); do m[il]=$(( 1 + ((il-1)/3)*3 )); done
for v in 45 46 47; do m[$v]=$v; done
IFS=,; MAP="${m[*]}"; unset IFS

CFG="LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA=0.5"
GPU="LLAMA_MMAP_GPU_COPY=1 LLAMA_FOLD_SKIP_LOAD=1"
PPLARGS="-f $S/wikitext-2-raw/wiki.test.raw --chunks 2 -c 512 -b 512 --no-repack"
w() { vm_stat | awk '/wired/{gsub(/\./,"",$4); printf "%.1f", $4*16384/1e9}'; }

echo "맵 R=3 + 층45/46/47 고정 + shexp논리 + γ=0.5 | CPU 기준값 27.5888"
echo "시작 wired $(w) GB"; echo

printf '%-44s ' "(a) Metal빌드 ppl, --device none (CPU경로)"
env $CFG LLAMA_FOLD_MAP="$MAP" "$D/build-fold/bin/llama-perplexity" -m "$LAG" $PPLARGS -ngl 0 --device none 2>&1 \
  | grep -aoE '\[2\][0-9.]+|\[2\]nan|error[^\n]{0,40}' | tail -1; echo

printf '%-44s ' "(b) Metal빌드 ppl, -ngl 99 + 로더스킵"
env $CFG $GPU LLAMA_FOLD_MAP="$MAP" "$D/build-fold/bin/llama-perplexity" -m "$LAG" $PPLARGS -ngl 99 2>&1 \
  | grep -aoE '\[2\][0-9.]+|\[2\]nan|error[^\n]{0,40}' | tail -1; echo
echo "  ppl 후 wired $(w) GB"; echo

echo "(c) 속도 재확인 (Metal, -n 32, r=3)"
env $CFG $GPU LLAMA_FOLD_MAP="$MAP" "$D/build-fold/bin/llama-bench" -m "$LAG" -p 0 -n 32 -r 3 -ngl 99 2>&1 \
  | grep -E "tg32|error|failed|out of" | tail -2
echo "종료 wired $(w) GB"
