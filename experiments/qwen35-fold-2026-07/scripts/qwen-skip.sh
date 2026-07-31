#!/bin/bash
# EXP-14 γ=0 MoE 스킵 검증. 회귀 기준: half_front(γ=0 19층) = 23.9036 ± 1.09049
set -u
KILL_GB=22
PPL=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
CLI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-cli
Q3=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-q3ks.gguf
WIKI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/scripts/wikitext-2-raw/wiki.test.raw
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }

m=(); G=7
for ((il=0; il<48; il++)); do
  if (( il >= 45 )); then m[il]=$il
  else g=$(( il * G / 45 )); m[il]=$(( (g * 45 + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS

run() { # $1=bin $2=옵션 $3=로그 $4=최대반복
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 $5 \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      $1 -m "$Q3" $2 > "$3" 2>&1 < /dev/null &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < $4 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! OOM 위험 kill"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  echo "$PEAK"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-14 γ=0 스킵 회귀+속도" >> $EXP/timeline.log

echo "=== 1) ppl 회귀 (γ=0 19층, 기준 23.9036) ==="
PK=$(run "$PPL" "-f $WIKI --chunks 16 -c 512 --no-repack -b 512 -ngl 99" "$EXP/sk-ppl.log" 60 "LLAMA_FOLD_GAMMA_FILE=$EXP/g0-half_front.txt")
echo "  $(grep -a 'Final estimate' $EXP/sk-ppl.log | tail -1 | sed 's/.*PPL = //')   peak ${PK} GB"

echo
echo "=== 2) 생성 속도 (γ=0 19층 vs 균일 0.3) ==="
P='Tokyo became the capital of Japan in 1868, when Emperor Meiji moved the court from Kyoto.

Q: In what year did Tokyo become the capital of Japan?
A:'
for cfg in "LLAMA_FOLD_GAMMA_FILE=$EXP/g0-half_front.txt:skip19" "LLAMA_FOLD_GAMMA=0.3:uniform"; do
  IFS=: read -r ENV TAG <<< "$cfg"
  PK=$(run "$CLI" "-p \"\" -n 120 -ngl 99 --no-repack -no-cnv -st --no-warmup -c 512 --seed 1337 --no-jinja --temp 0 --logit-bias 248068-20" "$EXP/sk-$TAG.log" 20 "$ENV")
  :
done
