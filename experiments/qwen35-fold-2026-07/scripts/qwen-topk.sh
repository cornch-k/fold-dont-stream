#!/bin/bash
# EXP-06 top-k 축소로 속도. 16GB 구성(물리층 5개). 기본 expert_used_count=8.
set -u
KILL_GB=22
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-bench
QWEN=/Users/angigyeom/Desktop/DEV/optillama/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }

m=()
for ((il=0; il<48; il++)); do
  if   (( il >= 45 )); then m[il]=$il
  elif (( il <  23 )); then m[il]=0
  else m[il]=23; fi
done
IFS=,; MAP="${m[*]}"; unset IFS

echo "top-k   pp512            tg32             peak"
for K in 8 6 4 2; do
  LOG=$EXP/tk-$K.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-06 top-k=$K" >> $EXP/timeline.log
  OV=""; [ "$K" != "8" ] && OV="--override-kv qwen35moe.expert_used_count=int:$K"
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA=0.3 \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$BIN" -m "$QWEN" -ngl 99 -p 512 -n 32 -r 2 --no-warmup $OV > "$LOG" 2>&1 &
  PID=$!; PEAK=0; i=0
  while kill -0 $PID 2>/dev/null && (( i < 40 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! OOM 위험 kill"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  wait $PID 2>/dev/null
  PP=$(grep -a "pp512" "$LOG" | grep -aoE "[0-9]+\.[0-9]+ ± +[0-9]+\.[0-9]+" | tail -1)
  TG=$(grep -a "tg32"  "$LOG" | grep -aoE "[0-9]+\.[0-9]+ ± +[0-9]+\.[0-9]+" | tail -1)
  printf '%-7s %-16s %-16s %s GB\n' "$K" "${PP:-FAIL}" "${TG:-FAIL}" "$PEAK"
  sleep 3
done
