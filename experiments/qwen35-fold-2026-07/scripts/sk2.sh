#!/bin/bash
# EXP-14b γ=0 MoE 스킵 속도. 대조: 균일 γ=0.3 (스킵 없음)
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
Q3=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-q3ks.gguf
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-bench
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); G=7
for ((il=0; il<48; il++)); do
  if (( il >= 45 )); then m[il]=$il
  else g=$(( il * G / 45 )); m[il]=$(( (g * 45 + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS

echo "구성            pp512             tg32              peak"
for cfg in "GAMMA_FILE=$EXP/g0-half_front.txt|skip19" "GAMMA=0.3|uniform"; do
  KV="${cfg%%|*}"; TAG="${cfg##*|}"
  LOG=$EXP/skb-$TAG.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-14b $TAG" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 "LLAMA_FOLD_$KV" \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$BIN" -m "$Q3" -ngl 99 -p 512 -n 32 -r 3 > "$LOG" 2>&1 &
  PID=$!; PEAK=0; i=0
  while kill -0 $PID 2>/dev/null && (( i < 60 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  wait $PID 2>/dev/null
  PP=$(grep -a "pp512" "$LOG" | grep -aoE "[0-9]+\.[0-9]+ ± +[0-9]+\.[0-9]+" | tail -1)
  TG=$(grep -a "tg32"  "$LOG" | grep -aoE "[0-9]+\.[0-9]+ ± +[0-9]+\.[0-9]+" | tail -1)
  printf '%-15s %-17s %-17s %s GB\n' "$TAG" "${PP:-FAIL}" "${TG:-FAIL}" "$PEAK"
  sleep 3
done
