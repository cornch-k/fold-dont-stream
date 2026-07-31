#!/bin/bash
# EXP-47 꼬리 프로브 재시도 @ 6-rep 맵 (예산 내): hot16 vs hot64 MMLU-500 정면 비교
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
H16=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a-hot.gguf
H64=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a-hot64.gguf
PPL=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
# 6-rep 맵: G=6, F=44 → reps 0,8,15,22,29,37 + 고정 44-47 (물리층 10)
m=(); for ((il=0; il<48; il++)); do
  if (( il >= 44 )); then m[il]=$il
  else g=$(( il * 5 / 44 )); m[il]=$(( (g * 44 + 4) / 5 )); fi
done
IFS=,; MAP6="${m[*]}"; unset IFS
# γ: 대표/고정 0.3, 접힌 il<23 0, 그 외 0.3
w=(${MAP6//,/ }); : > $EXP/g47.txt
for ((il=0; il<48; il++)); do
  if [ "${w[il]}" = "$il" ]; then echo "0.3"
  elif (( il < 23 )); then echo "0.0"
  else echo "0.3"; fi
done >> $EXP/g47.txt
echo "6-rep 유일층: $(echo "$MAP6" | tr ',' '\n' | sort -un | tr '\n' ' ')"

run() { # $1=모델 $2=태그 $3=벤치인자 $4=MAXI
  local LOG=$EXP/e47-$2.log
  local s W0
  for ((s=0; s<24; s++)); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; done
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-47 $2" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP6" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g47.txt \
      LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$PPL" -m "$1" -ngl 99 --no-repack -c 2048 -b 512 -ub 512 --seed 1337 \
      --override-kv qwen35moe.expert_used_count=int:6 $3 > "$LOG" 2>&1 &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < $4 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill $2"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  echo "$2: $(grep -a 'Final result' "$LOG" | tail -1 | sed 's/.*result: //')  peak ${PEAK} GB"
}
run "$H16" "h16-mmlu" "--multiple-choice --multiple-choice-tasks 500 -f $EXP/mmlu-test.bin" 400
run "$H64" "h64-mmlu" "--multiple-choice --multiple-choice-tasks 500 -f $EXP/mmlu-test.bin" 400
echo "EXP-47 완료"
