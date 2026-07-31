#!/bin/bash
# EXP-14a γ=0 MoE 스킵 — ppl 회귀. 기준 23.9036 ± 1.09049 (스킵 전)
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
Q3=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-q3ks.gguf
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); G=7
for ((il=0; il<48; il++)); do
  if (( il >= 45 )); then m[il]=$il
  else g=$(( il * G / 45 )); m[il]=$(( (g * 45 + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-14a 스킵 ppl 회귀" >> $EXP/timeline.log
env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
    LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g0-half_front.txt \
    LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
    /Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity \
    -m "$Q3" -f /Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/scripts/wikitext-2-raw/wiki.test.raw \
    --chunks 16 -c 512 --no-repack -b 512 -ngl 99 > $EXP/sk-ppl.log 2>&1 &
PID=$!; PEAK=0; i=0
while kill -0 $PID 2>/dev/null && (( i < 60 )); do
  W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
  awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill"; kill -9 $PID; break; }
  sleep 5; i=$((i+1))
done
wait $PID 2>/dev/null
echo "스킵 후 ppl: $(grep -a 'Final estimate' $EXP/sk-ppl.log | tail -1 | sed 's/.*PPL = //')   peak ${PEAK} GB"
echo "스킵 전 기준: 23.9036 +/- 1.09049   peak 17.04 GB"
