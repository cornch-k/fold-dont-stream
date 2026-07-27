#!/bin/bash
set -u
S=/private/tmp/claude-501/-Users-angigyeom-Desktop-DEV-optillama/525ff5d7-3e78-403d-9ac0-15e30e1a2ad7/scratchpad
D=/Users/angigyeom/Desktop/DEV/optillama/llama.cpp
LAG=/Users/angigyeom/Desktop/DEV/optillama/models/Laguna-S-2.1-GGUF/UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf
L=48; m=(); m[0]=0
for ((il=1; il<L; il++)); do m[il]=$(( 1 + ((il-1)/3)*3 )); done
for v in 45 46 47; do m[$v]=$v; done
IFS=,; MAP="${m[*]}"; unset IFS
P="$D/build-cpu/bin/llama-perplexity"
A="-f $S/wikitext-2-raw/wiki.test.raw --chunks 2 -c 512 -b 512 -t 8 -ngl 0 --no-repack"
C="LLAMA_MMAP_NO_PREFETCH=1 LLAMA_FOLD_MAP=$MAP LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA=0.5"
r(){ printf '%-40s ' "$1"; shift; env "$@" "$P" -m "$LAG" $A 2>&1 | grep -aoE '\[2\][0-9.]+|\[2\]nan' | tail -1; echo; }
echo "회귀: 헤드라인 27.5888 이 유지되어야 한다"
r "코드 수정 후 (기준 재현)"        $C
r "SHEXP_LOGICAL=0 (이제 '꺼짐')"   LLAMA_MMAP_NO_PREFETCH=1 LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=0 LLAMA_FOLD_GAMMA=0.5
