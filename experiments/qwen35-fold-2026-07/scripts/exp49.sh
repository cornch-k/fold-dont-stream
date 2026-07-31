#!/bin/bash
# EXP-49 꼬리 프로브 최종: 6-rep 맵, hot16(32.0 측정됨) vs hot32(뒤6층 질량~30%)
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
H32=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a-hot32.gguf
PPL=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); for ((il=0; il<48; il++)); do
  if (( il >= 44 )); then m[il]=$il
  else g=$(( il * 6 / 44 )); m[il]=$(( (g * 44 + 5) / 6 )); fi
done
IFS=,; MAP6="${m[*]}"; unset IFS
LOG=$EXP/e49-h32.log
s=0; while (( s < 24 )); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; s=$((s+1)); done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-49 h32 MMLU" >> $EXP/timeline.log
env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
    LLAMA_FOLD_MAP="$MAP6" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g46.txt \
    LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
    "$PPL" -m "$H32" -ngl 99 --no-repack -c 2048 -b 512 -ub 512 --seed 1337 \
    --override-kv qwen35moe.expert_used_count=int:6 \
    --multiple-choice --multiple-choice-tasks 500 -f $EXP/mmlu-test.bin > "$LOG" 2>&1 &
PID=$!; PEAK=0; i=0
while kill -0 $PID 2>/dev/null && (( i < 400 )); do
  W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
  awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill"; kill -9 $PID; break; }
  sleep 5; i=$((i+1))
done
kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
echo "h32-mmlu: $(grep -a 'Final result' "$LOG" | tail -1 | sed 's/.*result: //')  peak ${PEAK} GB  (h16@6rep 32.0)"
:
# h64 도 이어서 (top-16-of-64)
H64=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a-hot64.gguf
LOG=$EXP/e49-h64.log
s=0; while (( s < 24 )); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; s=$((s+1)); done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-49 h64 MMLU" >> $EXP/timeline.log
env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
    LLAMA_FOLD_MAP="$MAP6" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g46.txt \
    LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
    "$PPL" -m "$H64" -ngl 99 --no-repack -c 2048 -b 512 -ub 512 --seed 1337 \
    --override-kv qwen35moe.expert_used_count=int:6 \
    --multiple-choice --multiple-choice-tasks 500 -f $EXP/mmlu-test.bin > "$LOG" 2>&1 &
PID=$!; PEAK=0; i=0
while kill -0 $PID 2>/dev/null && (( i < 400 )); do
  W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
  awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill h64"; kill -9 $PID; break; }
  sleep 5; i=$((i+1))
done
kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
echo "h64c-mmlu: $(grep -a 'Final result' "$LOG" | tail -1 | sed 's/.*result: //')  peak ${PEAK} GB"
echo "EXP-49b 완료"
