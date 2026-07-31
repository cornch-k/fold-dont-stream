#!/bin/bash
# EXP-16 층 통째 드롭(DROP_ZERO). 기준: MoE만 스킵 → ppl 23.9036, tg32 28.10
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
Q3=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-q3ks.gguf
B=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); G=7
for ((il=0; il<48; il++)); do
  if (( il >= 45 )); then m[il]=$il
  else g=$(( il * G / 45 )); m[il]=$(( (g * 45 + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS
BASE="LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 LLAMA_FOLD_MAP=$MAP LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g0-half_front.txt LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1"

watch_run() { local PID=$1 MAXI=$2 PEAK=0 W i=0
  while kill -0 $PID 2>/dev/null && (( i < MAXI )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  wait $PID 2>/dev/null; echo "$PEAK"; }

echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-16 DROP_ZERO" >> $EXP/timeline.log
echo "=== ppl (기준 23.9036 ± 1.09049) ==="
env $BASE LLAMA_FOLD_DROP_ZERO=1 $B/llama-perplexity -m "$Q3" \
  -f $B/../../scripts/wikitext-2-raw/wiki.test.raw --chunks 16 -c 512 --no-repack -b 512 -ngl 99 \
  > $EXP/dr-ppl.log 2>&1 & PK=$(watch_run $! 60)
echo "  DROP: $(grep -a 'Final estimate' $EXP/dr-ppl.log | tail -1 | sed 's/.*PPL = //')  peak ${PK} GB"
grep -aiE "assert|abort|error" $EXP/dr-ppl.log | head -2

echo
echo "=== 속도 (기준 pp512 314.81 / tg32 28.10) ==="
env $BASE LLAMA_FOLD_DROP_ZERO=1 $B/llama-bench -m "$Q3" -ngl 99 -p 512 -n 32 -r 3 \
  > $EXP/dr-bench.log 2>&1 & PK=$(watch_run $! 60)
echo "  pp512: $(grep -a pp512 $EXP/dr-bench.log | grep -aoE '[0-9]+\.[0-9]+ ± +[0-9]+\.[0-9]+' | tail -1)"
echo "  tg32 : $(grep -a tg32  $EXP/dr-bench.log | grep -aoE '[0-9]+\.[0-9]+ ± +[0-9]+\.[0-9]+' | tail -1)   peak ${PK} GB"
