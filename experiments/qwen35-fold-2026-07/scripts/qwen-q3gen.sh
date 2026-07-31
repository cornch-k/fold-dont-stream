#!/bin/bash
# EXP-10 Q3_K_S 물리층 10개 (ppl 22.64, 14.9 GB) 생성 품질.
set -u
KILL_GB=22
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-cli
Q3=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-q3ks.gguf
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }

m=(); G=7
for ((il=0; il<48; il++)); do
  if (( il >= 45 )); then m[il]=$il
  else g=$(( il * G / 45 )); m[il]=$(( (g * 45 + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS
echo "물리층: $(echo "$MAP" | tr ',' '\n' | sort -un | tr '\n' ' ')"

P='Tokyo became the capital of Japan in 1868, when Emperor Meiji moved the court from Kyoto.

Q: In what year did Tokyo become the capital of Japan?
A:'

gen() { # $1=옵션 $2=태그
  local LOG=$EXP/q3g-$2.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-10 $2 : $1" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA=0.3 \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$BIN" -m "$Q3" -p "$P" -n 150 -ngl 99 --no-repack -no-cnv -st --no-warmup \
      -c 1024 --seed 1337 --no-jinja --temp 0 $1 > "$LOG" 2>&1 < /dev/null &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < 20 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! OOM 위험 kill"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  echo "══ $2  (peak ${PEAK} GB) ══"
  awk '/become the capital of Japan\?$/{f=1;next} f' "$LOG" | grep -av "^\[ Prompt" | head -8
  echo "   [$(grep -ao 'Generation: [0-9.]* t/s' "$LOG" | tail -1)]"; echo
  sleep 2
}

gen "" "bare"
gen "--logit-bias 248068-20 --logit-bias 248046+2" "think-eos2"
gen "--logit-bias 248068-20" "thinkonly"
gen "--logit-bias 248046+3" "eos3"
