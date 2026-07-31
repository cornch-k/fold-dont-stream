#!/bin/bash
# EXP-23 ppl 게이트: (a) shexp까지 스킵 (b) linear-attn 5층 드롭 (c) 둘 다
# 기준 23.9036 ± 1.09049
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
Q3=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-q3ks.gguf
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
WIKI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/scripts/wikitext-2-raw/wiki.test.raw
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); G=7
for ((il=0; il<48; il++)); do
  if (( il >= 45 )); then m[il]=$il
  else g=$(( il * G / 45 )); m[il]=$(( (g * 45 + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS

run() { # $1=extra env  $2=태그
  local LOG=$EXP/g2-$2.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-23 $2 : $1" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g0-half_front.txt \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 $1 \
      "$BIN" -m "$Q3" -f "$WIKI" --chunks 16 -c 512 --no-repack -b 512 -ngl 99 > "$LOG" 2>&1 &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < 60 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  wait $PID 2>/dev/null
  printf '%-14s %-24s %s GB\n' "$2" "$(grep -a 'Final estimate' "$LOG" | tail -1 | sed 's/.*PPL = //')" "$PEAK"
  sleep 3
}
printf '%-14s %-24s\n' "기준" "23.9036 +/- 1.09049"
run "LLAMA_FOLD_SHEXP_SKIP_ZERO=1" "shexp-skip"
run "LLAMA_FOLD_DROP_LAYERS=2,5,9,16,21" "drop5"
run "LLAMA_FOLD_SHEXP_SKIP_ZERO=1 LLAMA_FOLD_DROP_LAYERS=2,5,9,16,21" "both"
