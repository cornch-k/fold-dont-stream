#!/bin/bash
# EXP-37 벤치 지표로 구성 재검증 — ppl로 "공짜" 판정했던 결정들이 벤치에서 유죄인지.
# MMLU 500태스크(동일 시드=동일 부분집합), 4팔:
#   A 현재구성(γ파일 front-0, k6)   B γ0.3균일, k6   C γ파일, k8   D γ0.3균일, k8
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
V2=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a.gguf
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); G=7; F=44
for ((il=0; il<48; il++)); do
  if (( il >= F )); then m[il]=$il
  else g=$(( il * G / F )); m[il]=$(( (g * F + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS

# 보충 스위트 완료 대기 (GPU 직렬화, 최대 90분)
for ((w=0; w<1080; w++)); do
  grep -q "보충 스위트 전체 완료" $EXP/bench-makeup.out 2>/dev/null && break
  pgrep -f bench-makeup.sh >/dev/null || break
  sleep 5
done

run() { # $1=γ스펙(file:<path>|const:<v>) $2=k $3=태그
  local LOG=$EXP/e37-$3.log GAMMA_ENV KV=""
  case "$1" in
    file:*) GAMMA_ENV="LLAMA_FOLD_GAMMA_FILE=${1#file:}" ;;
    const:*) GAMMA_ENV="LLAMA_FOLD_GAMMA=${1#const:}" ;;
  esac
  [ "$2" != "8" ] && KV="--override-kv qwen35moe.expert_used_count=int:$2"
  local s W0
  for ((s=0; s<24; s++)); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; done
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-37 $3" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 $GAMMA_ENV \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$BIN" -m "$V2" -ngl 99 --no-repack -c 2048 -b 512 -ub 512 --seed 1337 $KV \
      --multiple-choice --multiple-choice-tasks 500 -f $EXP/mmlu-test.bin > "$LOG" 2>&1 &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < 400 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill $3"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  echo "══ $3 (peak ${PEAK} GB) ══"
  grep -a "Final result" "$LOG" | tail -1
}

run "file:$EXP/g32.txt" 6 "A-gfile-k6"
run "const:0.3"         6 "B-g03-k6"
run "file:$EXP/g32.txt" 8 "C-gfile-k8"
run "const:0.3"         8 "D-g03-k8"
echo "EXP-37 전체 완료"
