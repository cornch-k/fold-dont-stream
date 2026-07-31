#!/bin/bash
# EXP-20 ngram 투기 디코딩 스윕. 기준 real-nomtp 25.8 t/s. draft 모델 불필요.
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
Q3=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-q3ks.gguf
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-cli
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); G=7
for ((il=0; il<48; il++)); do
  if (( il >= 45 )); then m[il]=$il
  else g=$(( il * G / 45 )); m[il]=$(( (g * 45 + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS

P='The history of computing began with mechanical calculators. Charles Babbage designed the Analytical Engine in the 1830s, and Ada Lovelace wrote algorithms for it. In the twentieth century, electronic computers replaced mechanical ones. The story continues:'

run() { # $1=spec-type $2=태그
  local LOG=$EXP/ng-$2.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-20 $2" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g0-half_front.txt \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$BIN" -m "$Q3" -p "$P" -n 200 --ignore-eos -ngl 99 --no-repack -no-cnv -st --no-warmup \
      -c 1024 --seed 1337 --no-jinja --temp 0 --repeat-penalty 1.15 --logit-bias 248068-20 \
      $1 > "$LOG" 2>&1 < /dev/null &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < 70 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  local ERR=$(grep -aoE "error[^\n]{0,40}" "$LOG" | head -1)
  printf '%-18s %-28s peak %s GB  %s\n' "$2" "$(grep -ao 'Generation: [0-9.]* t/s' "$LOG" | tail -1)" "$PEAK" "$ERR"
  sleep 2
}

run "--spec-type ngram-simple --spec-draft-n-max 8 --override-kv qwen35moe.expert_used_count=int:4" "ng-n8-k4"
run "--spec-type ngram-simple --spec-draft-n-max 24 --override-kv qwen35moe.expert_used_count=int:4" "ng-n24-k4"
run "--spec-type ngram-map-k4v --spec-draft-n-max 16 --override-kv qwen35moe.expert_used_count=int:4" "ngmap-n16-k4"
run "--spec-type ngram-simple --spec-draft-n-max 16 --override-kv qwen35moe.expert_used_count=int:6" "ng-n16-k6"
