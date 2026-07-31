#!/bin/bash
# EXP-39 think 모드 부활 게이트: 18.99 품질에서 사고 사슬이 완결되는가.
# 이전(22.6 품질): <think> 진입 즉시 루프 → 차단이 유일한 해법이었음.
# 열리면: 생성형 CoT 평가로 벤치 대폭 상승 여지 (추론 모델의 본래 사용법).
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
V2=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a.gguf
CLI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-cli
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); G=7; F=44
for ((il=0; il<48; il++)); do
  if (( il >= 44 )); then m[il]=$il
  else g=$(( il * G / 44 )); m[il]=$(( (g * 44 + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS

gen() { # $1=프롬프트 $2=태그 $3=추가옵션
  local LOG=$EXP/e39-$2.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-39 $2" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g32.txt \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$CLI" -m "$V2" -p "$1" -n 400 -ngl 99 --no-repack -no-cnv -st --no-warmup \
      -c 1024 --seed 1337 --no-jinja --temp 0 --repeat-penalty 1.15 $3 \
      --override-kv qwen35moe.expert_used_count=int:6 > "$LOG" 2>&1 < /dev/null &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < 40 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  echo "══ $2 (peak ${PEAK} GB) ══"
  awk '/^A:$|^A: |Answer:/{f=1} f' "$LOG" | grep -av "Prompt:" | head -14
  echo "   [$(grep -ao 'Generation: [0-9.]* t/s' "$LOG" | tail -1)]"
}

P1='Q: A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?
A:'
P2='Q: Which planet has more moons, Mars or Jupiter? Think briefly, then answer.
A:'
gen "$P1" "think-math" ""
gen "$P2" "think-fact" ""
