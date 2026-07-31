#!/bin/bash
# EXP-43d δ 미세 × γ 상호작용. 기준 δ=1.0/γ=0.3 → 13.4929
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
HOT=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a-hot.gguf
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
WIKI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/scripts/wikitext-2-raw/wiki.test.raw
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); for ((il=0; il<48; il++)); do
  if (( il >= 44 )); then m[il]=$il
  else g=$(( il * 7 / 44 )); m[il]=$(( (g * 44 + 6) / 7 )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS

mkg() { # $1=γ  (접힌 il<23 = 0 유지)
  local G=$1 il; local -a w=(${MAP//,/ })
  for ((il=0; il<48; il++)); do
    if [ "${w[il]}" = "$il" ]; then echo "$G"
    elif (( il < 23 )); then echo "0.0"
    else echo "$G"; fi
  done
}

run() { # $1=δ $2=γ
  mkg $2 > $EXP/g43-$2.txt
  local LOG=$EXP/e43d-$1-$2.log
  local s W0
  for ((s=0; s<24; s++)); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; done
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-43d δ=$1 γ=$2" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g43-$2.txt \
      LLAMA_FOLD_HOT_SCALE=$1 \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$BIN" -m "$HOT" -f "$WIKI" --chunks 16 -c 512 --no-repack -b 512 -ngl 99 \
      --override-kv qwen35moe.expert_used_count=int:6 > "$LOG" 2>&1 &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < 60 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  wait $PID 2>/dev/null
  printf 'δ=%-5s γ=%-5s %-22s %s GB\n' "$1" "$2" "$(grep -a 'Final estimate' "$LOG" | tail -1 | sed 's/.*PPL = //')" "$PEAK"
}
echo "δ=1.0  γ=0.3  13.4929 +/- 0.55534  (기준)"
run 0.8 0.3
run 1.2 0.3
run 1.0 0.2
run 1.0 0.4
run 1.2 0.2
echo "EXP-43d 완료"
