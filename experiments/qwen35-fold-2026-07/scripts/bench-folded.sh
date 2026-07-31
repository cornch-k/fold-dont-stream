#!/bin/bash
# EXP-29 접힌 최종 구성 벤치마크 4종 (기준선 완료 대기 후 순차 실행)
# 구성 = 최종 배포: Q3_K_S 물리층10 + skip19 γ파일 + k6. 우도 채점이라 spec/logit-bias 불필요.
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
Q3=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-q3ks.gguf
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); G=7
for ((il=0; il<48; il++)); do
  if (( il >= 45 )); then m[il]=$il
  else g=$(( il * G / 45 )); m[il]=$(( (g * 45 + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS

# 기준선(무접기 ppl) 완료 대기 — 최대 50분
for ((w=0; w<600; w++)); do
  grep -aq "Final estimate" $EXP/baseline.log 2>/dev/null && break
  pgrep -f "device none" >/dev/null || break
  sleep 5
done
echo "기준선: $(grep -a 'Final estimate' $EXP/baseline.log | tail -1 | sed 's/.*PPL = //')"

run() { # $1=벤치 인자 $2=태그 $3=최대반복(5초단위)
  local LOG=$EXP/bench-$2.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-29 $2 시작" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g0-half_front.txt \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$BIN" -m "$Q3" -ngl 99 --no-repack -c 2048 -b 2048 -ub 512 --seed 1337 \
      --override-kv qwen35moe.expert_used_count=int:6 \
      $1 > "$LOG" 2>&1 &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < $3 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill $2"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-29 $2 종료 peak=${PEAK}GB" >> $EXP/timeline.log
  echo "══ $2 (peak ${PEAK} GB) ══"
  tail -30 "$LOG" | grep -aE "Final|acc|score|[0-9]+\.[0-9]+%" | tail -4
}

run "--multiple-choice --multiple-choice-tasks 2000 -f $EXP/mmlu-test.bin" "mmlu2000" 480
run "--hellaswag --hellaswag-tasks 400 -f $EXP/hellaswag_val_full.txt" "hellaswag400" 360
run "--winogrande --winogrande-tasks 2000 -f $EXP/winogrande-debiased-eval.csv" "winogrande" 360
run "--multiple-choice -f $EXP/arc-challenge-validation.bin" "arc-challenge" 300
echo "전체 완료"
