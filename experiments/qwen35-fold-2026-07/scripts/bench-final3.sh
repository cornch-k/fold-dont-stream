#!/bin/bash
# EXP-35 최종 벤치: A열=확정 C구성(접힌 122B v2a 11층), C열=라이벌(Qwen3-30B-A3B native)
# 같은 시드 1337 → 동일 태스크 부분집합. 우도 채점이라 spec/샘플링 무관.
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
V2=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a.gguf
RIVAL=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/rival/Qwen3-30B-A3B-Instruct-2507-UD-Q3_K_XL.gguf
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); G=7; F=44
for ((il=0; il<48; il++)); do
  if (( il >= F )); then m[il]=$il
  else g=$(( il * G / F )); m[il]=$(( (g * F + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS
FOLD_ENV="LLAMA_FOLD_MAP=$MAP LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g32.txt LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1"

run() { # $1=모델 $2=fold환경("-"이면 없음) $3=벤치인자 $4=k-override("-"이면 없음) $5=태그 $6=최대반복
  local LOG=$EXP/bf-$5.log ENV1="" KV=""
  [ "$2" != "-" ] && ENV1="$2"
  [ "$4" != "-" ] && KV="--override-kv qwen35moe.expert_used_count=int:6"
  # 사전 점검: 직전 kill 의 Metal wired 회수가 끝날 때까지 대기 (최대 2분)
  local s W0
  for ((s=0; s<24; s++)); do
    W0=$(wired_gb)
    awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break
    sleep 5
  done
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-35 $5 시작 (사전 wired ${W0}GB)" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 $ENV1 \
      "$BIN" -m "$1" -ngl 99 --no-repack -c 2048 -b 512 -ub 512 --seed 1337 $KV \
      $3 > "$LOG" 2>&1 &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < $6 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill $5"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-35 $5 종료 peak=${PEAK}GB" >> $EXP/timeline.log
  echo "══ $5 (peak ${PEAK} GB) ══"
  tail -40 "$LOG" | grep -aE "Final|acc = |score|%" | tail -3
done_marker=1
}

for MODEL_TAG in "folded" "rival"; do
  if [ "$MODEL_TAG" = "folded" ]; then MDL=$V2; FE="$FOLD_ENV"; KO="k"; else MDL=$RIVAL; FE="-"; KO="-"; fi
  if [ "$MODEL_TAG" != "folded" ]; then run "$MDL" "$FE" "--multiple-choice --multiple-choice-tasks 2000 -f $EXP/mmlu-test.bin" "$KO" "$MODEL_TAG-mmlu" 700; fi
  run "$MDL" "$FE" "--hellaswag --hellaswag-tasks 400 -f $EXP/hellaswag_val_full.txt" "$KO" "$MODEL_TAG-hswag" 500
  run "$MDL" "$FE" "--winogrande --winogrande-tasks 2000 -f $EXP/winogrande-debiased-eval.csv" "$KO" "$MODEL_TAG-wino" 500
  run "$MDL" "$FE" "--multiple-choice -f $EXP/arc-challenge-validation.bin" "$KO" "$MODEL_TAG-arc" 400
done
echo "벤치 스위트 전체 완료"
