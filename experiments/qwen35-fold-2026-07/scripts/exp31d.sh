#!/bin/bash
# EXP-31d v2a 12층 — 생성 peak(예산 판정) + 속도 + QA
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
V2=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a.gguf
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-cli
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); G=7; F=43
for ((il=0; il<48; il++)); do
  if (( il >= F )); then m[il]=$il
  else g=$(( il * G / F )); m[il]=$(( (g * F + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS

STORY='The history of computing began with mechanical calculators. Charles Babbage designed the Analytical Engine in the 1830s, and Ada Lovelace wrote algorithms for it. In the twentieth century, electronic computers replaced mechanical ones. The story continues:'
QA='Tokyo became the capital of Japan in 1868, when Emperor Meiji moved the court from Kyoto.

Q: In what year did Tokyo become the capital of Japan?
A:'

run() { # $1=프롬프트 $2=n $3=ig $4=태그
  local LOG=$EXP/e31d-$4.log IG=""
  [ "$3" = "1" ] && IG="--ignore-eos"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-31d $4" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g30-B.txt \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$BIN" -m "$V2" -p "$1" -n $2 $IG -ngl 99 --no-repack -no-cnv -st --no-warmup \
      -c 1024 --seed 1337 --no-jinja --temp 0 --repeat-penalty 1.15 --logit-bias 248068-20 \
      --spec-type ngram-simple --spec-draft-n-max 12 \
      --override-kv qwen35moe.expert_used_count=int:6 > "$LOG" 2>&1 < /dev/null &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < 80 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  printf '%-8s %-26s peak %s GB (llama≈%.1f)\n' "$4" "$(grep -ao 'Generation: [0-9.]* t/s' "$LOG" | tail -1)" "$PEAK" "$(echo "$PEAK - 2.2" | bc)"
  sleep 2
}

run "$STORY" 300 1 "speed"
run "$QA" 100 0 "qa"
echo "── QA 출력 ──"
awk '/become the capital of Japan\?$/{f=1;next} f' $EXP/e31d-qa.log | grep -av "Prompt:" | head -6
