#!/bin/bash
# EXP-51b front-hot ppl: A구성(6-rep) + front6/back18 hot. 기준 A = 14.6569
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
HF=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a-hotF.gguf
PPL=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
WIKI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/scripts/wikitext-2-raw/wiki.test.raw
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
for ((w=0; w<240; w++)); do grep -q "완료:" $EXP/surgeryF.log 2>/dev/null && break; sleep 5; done
grep -q "완료:" $EXP/surgeryF.log || { echo "수술 미완"; exit 1; }
m=(); for ((il=0; il<48; il++)); do
  if (( il >= 44 )); then m[il]=$il
  else g=$(( il * 6 / 44 )); m[il]=$(( (g * 44 + 5) / 6 )); fi
done
IFS=,; MAP6="${m[*]}"; unset IFS
LOG=$EXP/e51-ppl.log
s=0; while (( s < 24 )); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; s=$((s+1)); done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-51b front-hot ppl" >> $EXP/timeline.log
env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
    LLAMA_FOLD_MAP="$MAP6" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g46.txt \
    LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
    "$PPL" -m "$HF" -f "$WIKI" --chunks 16 -c 512 --no-repack -b 512 -ngl 99 \
    --override-kv qwen35moe.expert_used_count=int:6 > "$LOG" 2>&1 &
PID=$!; PEAK=0; i=0
while kill -0 $PID 2>/dev/null && (( i < 60 )); do
  W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
  awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill"; kill -9 $PID; break; }
  sleep 5; i=$((i+1))
done
kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
echo "front-hot ppl: $(grep -a 'Final estimate' "$LOG" | tail -1 | sed 's/.*PPL = //')  peak ${PEAK} GB  (A 14.6569)"
echo "EXP-51 완료"
