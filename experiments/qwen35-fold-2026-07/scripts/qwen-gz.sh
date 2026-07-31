#!/bin/bash
# EXP-12 γ=0 층 도입. 가설: γ=0 층은 MoE 기여가 0이므로 계산 스킵 가능(→속도).
# 먼저 코드수정 없이 품질만 확인. 기준 균일 γ=0.3 = 22.6446 ± 1.02794
set -u
KILL_GB=22
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
Q3=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-q3ks.gguf
WIKI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/scripts/wikitext-2-raw/wiki.test.raw
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }

m=(); G=7
for ((il=0; il<48; il++)); do
  if (( il >= 45 )); then m[il]=$il
  else g=$(( il * G / 45 )); m[il]=$(( (g * 45 + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS

echo "프로파일     γ=0층  PPL                    peak"
printf '%-12s %-6s %-22s %s\n' "균일0.3" "0" "22.6446 +/- 1.02794" "17.04 GB"
for P in gz-30 gz-36 gz-42 gz-45; do
  LOG=$EXP/z-$P.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-12 $P" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/$P.txt \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$BIN" -m "$Q3" -f "$WIKI" --chunks 16 -c 512 --no-repack -b 512 -ngl 99 > "$LOG" 2>&1 &
  PID=$!; PEAK=0; i=0
  while kill -0 $PID 2>/dev/null && (( i < 60 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! OOM 위험 kill"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  wait $PID 2>/dev/null
  Z=$(grep -c "^0.0000" $EXP/$P.txt)
  printf '%-12s %-6s %-22s %s GB\n' "$P" "$Z" "$(grep -a 'Final estimate' "$LOG" | tail -1 | sed 's/.*PPL = //')" "$PEAK"
  sleep 3
done
