#!/bin/bash
# EXP-53 최종 표준 A 의 기록급 벤치 4종 (MMLU-2000, HSwag400, Wino, ARC -np8)
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
HOT=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a-hot.gguf
PPL=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); for ((il=0; il<48; il++)); do
  if (( il >= 44 )); then m[il]=$il
  else g=$(( il * 6 / 44 )); m[il]=$(( (g * 44 + 5) / 6 )); fi
done
IFS=,; MAP6="${m[*]}"; unset IFS
run() { # $1=벤치인자 $2=ctx/batch $3=태그 $4=MAXI
  local LOG=$EXP/e53-$3.log
  local s W0
  for ((s=0; s<24; s++)); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; done
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-53 $3" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP6" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g46.txt \
      LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$PPL" -m "$HOT" -ngl 99 --no-repack $2 --seed 1337 \
      --override-kv qwen35moe.expert_used_count=int:6 $1 > "$LOG" 2>&1 &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < $4 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill $3"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  echo "$3: $(tail -30 "$LOG" | grep -aE 'Final|score' | tail -1)  peak ${PEAK} GB"
}
run "--multiple-choice --multiple-choice-tasks 2000 -f $EXP/mmlu-test.bin" "-c 2048 -b 512 -ub 512" "mmlu2000" 700
run "--hellaswag --hellaswag-tasks 400 -f $EXP/hellaswag_val_full.txt" "-c 1024 -b 128 -ub 128" "hswag" 700
run "--winogrande --winogrande-tasks 2000 -f $EXP/winogrande-debiased-eval.csv" "-c 1024 -b 128 -ub 128" "wino" 700
run "--multiple-choice -f $EXP/arc-challenge-validation.bin" "-c 2048 -b 512 -ub 512 -np 8" "arc" 500
echo "EXP-53 완료"
