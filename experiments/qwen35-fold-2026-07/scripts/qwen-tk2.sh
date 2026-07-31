#!/bin/bash
# EXP-11 top-k 축소 속도 (Q3_K_S 물리층 10개). 기준 top-k=8 → 26.4 t/s
set -u
KILL_GB=22
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-cli
Q3=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-q3ks.gguf
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }

m=(); G=7
for ((il=0; il<48; il++)); do
  if (( il >= 45 )); then m[il]=$il
  else g=$(( il * G / 45 )); m[il]=$(( (g * 45 + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS

P='Tokyo became the capital of Japan in 1868, when Emperor Meiji moved the court from Kyoto.

Q: In what year did Tokyo become the capital of Japan?
A:'

echo "top-k  속도              답변            peak"
for K in 8 6 4 2; do
  LOG=$EXP/tk2-$K.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-11 top-k=$K" >> $EXP/timeline.log
  OV=""; [ "$K" != "8" ] && OV="--override-kv qwen35moe.expert_used_count=int:$K"
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA=0.3 \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$BIN" -m "$Q3" -p "$P" -n 120 -ngl 99 --no-repack -no-cnv -st --no-warmup \
      -c 512 --seed 1337 --no-jinja --temp 0 --logit-bias 248068-20 $OV > "$LOG" 2>&1 < /dev/null &
  PID=$!; PEAK=0; i=0
  while kill -0 $PID 2>/dev/null && (( i < 20 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! OOM 위험 kill"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  SP=$(grep -ao 'Generation: [0-9.]* t/s' "$LOG" | tail -1 | sed 's/Generation: //')
  AN=$(awk '/become the capital of Japan\?$/{f=1;next} f' "$LOG" | grep -av "^\[ Prompt" | grep -av "^$" | head -2 | tr '\n' ' ' | cut -c1-30)
  ERR=$(grep -aoE "error[^\n]{0,30}" "$LOG" | head -1)
  printf '%-6s %-17s %-15s %s GB %s\n' "$K" "${SP:-—}" "${AN:-—}" "$PEAK" "$ERR"
  sleep 3
done
