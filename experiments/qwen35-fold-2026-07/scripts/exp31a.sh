#!/bin/bash
# EXP-31a 접힌 그래프 imatrix 수집 (map B: 7그룹+고정43-47, k6, 배포 γ)
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
Q3=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-q3ks.gguf
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-imatrix
TRAIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/scripts/wikitext-2-raw/wiki.train.raw
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }

m=(); G=7; F=43
for ((il=0; il<48; il++)); do
  if (( il >= F )); then m[il]=$il
  else g=$(( il * G / F )); m[il]=$(( (g * F + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS

echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-31a imatrix 수집 시작 (60청크)" >> $EXP/timeline.log
env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
    LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g30-B.txt \
    LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
    "$BIN" -m "$Q3" -f "$TRAIN" --chunks 60 -c 512 -b 512 -ngl 99 --no-repack \
    --override-kv qwen35moe.expert_used_count=int:6 \
    -o $EXP/qwen35-folded.imatrix > $EXP/e31a.log 2>&1 &
PID=$!; PEAK=0; i=0
while kill -0 $PID 2>/dev/null && (( i < 240 )); do
  W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
  awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill"; kill -9 $PID; break; }
  sleep 5; i=$((i+1))
done
wait $PID 2>/dev/null
echo "peak ${PEAK} GB"
ls -la $EXP/qwen35-folded.imatrix 2>/dev/null || echo "imatrix 파일 없음!"
tail -3 $EXP/e31a.log | head -3
