#!/bin/bash
# EXP-09 Q3_K_S(3.47bpw) 재양자화 모델. 이중 양자화 손실 + 물리층 확장 효과.
# 대조: 원본 Q4_K_XL(4.955bpw) 물리층 8개 = 28.5475 ± 1.32174 (19.6 GB)
set -u
KILL_GB=22
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
Q3=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-q3ks.gguf
WIKI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/scripts/wikitext-2-raw/wiki.test.raw
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }

mk() { local G=$1 il g; local -a m=()
  for ((il=0; il<48; il++)); do
    if (( il >= 45 )); then m[il]=$il
    else g=$(( il * G / 45 )); m[il]=$(( (g * 45 + G - 1) / G )); fi
  done
  local IFS=,; echo "${m[*]}"; }

echo "물리층  MAP유일층                          PPL                    peak"
for G in 5 7; do
  MAP=$(mk $G)
  UNIQ=$(echo "$MAP" | tr ',' '\n' | sort -un | tr '\n' ',' | sed 's/,$//')
  NP=$(echo "$MAP" | tr ',' '\n' | sort -u | wc -l | tr -d ' ')
  LOG=$EXP/q3-np$NP.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-09 Q3_K_S 물리층$NP" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA=0.3 \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$BIN" -m "$Q3" -f "$WIKI" --chunks 16 -c 512 --no-repack -b 512 -ngl 99 > "$LOG" 2>&1 &
  PID=$!; PEAK=0; i=0
  while kill -0 $PID 2>/dev/null && (( i < 60 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! OOM 위험 kill"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  wait $PID 2>/dev/null
  printf '%-7s %-34s %-22s %s GB\n' "$NP" "$UNIQ" "$(grep -a 'Final estimate' "$LOG" | tail -1 | sed 's/.*PPL = //')" "$PEAK"
  sleep 3
done
