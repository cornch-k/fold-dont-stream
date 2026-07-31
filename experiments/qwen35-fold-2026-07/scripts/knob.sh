#!/bin/bash
# 노브 스윕 실행기. 스펙 파일의 각 줄을 순차로 ppl 측정한다.
# 스펙 형식: 라벨<TAB>맵태그<TAB>추가 ENV (예: "LLAMA_FOLD_HOT_SCALE=1.3")
# 메모리 안전은 상위 safe.sh 가 담당하고, 여기서는 실행 간 정착만 한다.
set -u
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
EXP=exp
SPEC=$1
CHUNKS=${2:-16}
M=${3:-models/qwen35-v2a-hot.gguf}
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
settle() { local s=0 W; while (( s < 40 )); do W=$(wired_gb); awk -v w="$W" 'BEGIN{exit !(w<8)}' && return; sleep 5; s=$((s+1)); done; }

while IFS=$'\t' read -r LBL TAG EXTRA K; do
  K=${K:-6}
  case "$LBL" in ''|'#'*) continue;; esac
  MAP=$(python3 -c "import json;print(json.load(open('$EXP/maps58.json'))['$TAG'])")
  GAM=$EXP/g58-$TAG.txt
  settle
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] knob $LBL" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$GAM \
      LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 $EXTRA \
      nanfix/build/bin/llama-perplexity -m $M \
      -f nanfix/scripts/wikitext-2-raw/wiki.test.raw --chunks $CHUNKS -c 512 --no-repack -b 512 -ngl 99 \
      --override-kv qwen35moe.expert_used_count=int:$K > $EXP/k-$LBL.log 2>&1
  echo "RESULT_KNOB $LBL ppl=$(grep -a 'Final estimate' $EXP/k-$LBL.log | tail -1 | sed 's/.*PPL = //')  $(python3 $EXP/memcalc58.py $EXP/k-$LBL.log $M 2>/dev/null)"
done < "$SPEC"
echo "노브 스윕 완료: $SPEC"
