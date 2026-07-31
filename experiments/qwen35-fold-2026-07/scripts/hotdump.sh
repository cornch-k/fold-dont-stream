#!/bin/bash
# EXP-43a 자기 라우터 확률 덤프 (활성 접힌 층 18개, ROUTER_LOGICAL=1)
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
V2=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a.gguf
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-debug
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); for ((il=0; il<48; il++)); do
  if (( il >= 44 )); then m[il]=$il
  else g=$(( il * 7 / 44 )); m[il]=$(( (g * 44 + 6) / 7 )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS
FILT=""
for il in 23 24 25 27 28 29 30 31 33 34 35 36 37 39 40 41 42 43; do
  FILT="$FILT --tensor-filter ffn_moe_probs-${il}\$"
done
rm -rf $EXP/hotprobs; mkdir -p $EXP/hotprobs
s=0; while (( s < 24 )); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; s=$((s+1)); done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-43a hot probs 덤프" >> $EXP/timeline.log
env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
    LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g32.txt \
    LLAMA_FOLD_ROUTER_LOGICAL=1 \
    LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 LLAMA_DEBUG_SAVE_DIR=$EXP/hotprobs \
    "$BIN" -m "$V2" -f $EXP/calib8k.txt -ngl 99 --no-repack -c 8192 -b 8192 -ub 512 \
    $FILT --override-kv qwen35moe.expert_used_count=int:6 > $EXP/hotdump.log 2>&1 &
PID=$!; PEAK=0; i=0
while kill -0 $PID 2>/dev/null && (( i < 60 )); do
  W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
  awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill"; kill -9 $PID; break; }
  sleep 5; i=$((i+1))
done
kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
echo "peak ${PEAK} GB, 파일 $(ls $EXP/hotprobs | wc -l | tr -d ' ')개"
