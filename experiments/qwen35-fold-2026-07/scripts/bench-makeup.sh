#!/bin/bash
# EXP-36 보충 + 라이벌2(동세대 35B-A3B)
#   접힌: hswag/wino (-c 1024 -b 128, 스파이크 억제), arc (-np 8)
#   라이벌30B: arc (-np 8)
#   라이벌2 35B: mmlu / hswag / wino / arc(-np 8)
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
V2=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a.gguf
R1=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/rival/Qwen3-30B-A3B-Instruct-2507-UD-Q3_K_XL.gguf
R2=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/rival/Qwen3.5-35B-A3B-UD-IQ3_S.gguf
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); G=7; F=44
for ((il=0; il<48; il++)); do
  if (( il >= F )); then m[il]=$il
  else g=$(( il * G / F )); m[il]=$(( (g * F + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS
FOLD_ENV="LLAMA_FOLD_MAP=$MAP LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g32.txt LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1"

run() { # $1=모델 $2=fold("-"=없음) $3=k6("-"=없음) $4=ctx/batch/np인자 $5=벤치인자 $6=태그 $7=최대반복
  local LOG=$EXP/mk-$6.log ENV1="" KV=""
  [ "$2" != "-" ] && ENV1="$2"
  [ "$3" != "-" ] && KV="--override-kv qwen35moe.expert_used_count=int:6"
  local s W0
  for ((s=0; s<24; s++)); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; done
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-36 $6 시작 (wired ${W0}GB)" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 $ENV1 \
      "$BIN" -m "$1" -ngl 99 --no-repack --seed 1337 $KV $4 $5 > "$LOG" 2>&1 &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < $7 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill $6"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  echo "══ $6 (peak ${PEAK} GB) ══"
  tail -40 "$LOG" | grep -aE "Final|acc = |score|%" | tail -3
}

echo "── 보충: 접힌 ──"
run "$V2" "$FOLD_ENV" "k" "-c 1024 -b 128 -ub 128" "--hellaswag --hellaswag-tasks 400 -f $EXP/hellaswag_val_full.txt" "folded-hswag" 700
run "$V2" "$FOLD_ENV" "k" "-c 1024 -b 128 -ub 128" "--winogrande --winogrande-tasks 2000 -f $EXP/winogrande-debiased-eval.csv" "folded-wino" 700
run "$V2" "$FOLD_ENV" "k" "-c 2048 -b 512 -ub 512 -np 8" "--multiple-choice -f $EXP/arc-challenge-validation.bin" "folded-arc" 500
echo "── 보충: 라이벌 30B ──"
run "$R1" "-" "-" "-c 2048 -b 512 -ub 512 -np 8" "--multiple-choice -f $EXP/arc-challenge-validation.bin" "rival-arc" 400
echo "── 라이벌2: Qwen3.5-35B-A3B (동세대) ──"
run "$R2" "-" "-" "-c 2048 -b 512 -ub 512" "--multiple-choice --multiple-choice-tasks 2000 -f $EXP/mmlu-test.bin" "r2-mmlu" 700
run "$R2" "-" "-" "-c 2048 -b 512 -ub 512" "--hellaswag --hellaswag-tasks 400 -f $EXP/hellaswag_val_full.txt" "r2-hswag" 500
run "$R2" "-" "-" "-c 2048 -b 512 -ub 512" "--winogrande --winogrande-tasks 2000 -f $EXP/winogrande-debiased-eval.csv" "r2-wino" 500
run "$R2" "-" "-" "-c 2048 -b 512 -ub 512 -np 8" "--multiple-choice -f $EXP/arc-challenge-validation.bin" "r2-arc" 400
echo "보충 스위트 전체 완료"
