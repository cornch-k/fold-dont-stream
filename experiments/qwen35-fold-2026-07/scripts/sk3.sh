#!/bin/bash
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
Q3=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-q3ks.gguf
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); G=7
for ((il=0; il<48; il++)); do
  if (( il >= 45 )); then m[il]=$il
  else g=$(( il * G / 45 )); m[il]=$(( (g * 45 + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS
P='Tokyo became the capital of Japan in 1868, when Emperor Meiji moved the court from Kyoto.

Q: In what year did Tokyo become the capital of Japan?
A:'
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-14c 스킵 후 생성 품질" >> $EXP/timeline.log
env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
    LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g0-half_front.txt \
    LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
    /Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-cli \
    -m "$Q3" -p "$P" -n 120 -ngl 99 --no-repack -no-cnv -st --no-warmup \
    -c 512 --seed 1337 --no-jinja --temp 0 --logit-bias 248068-20 > $EXP/sk-gen.log 2>&1 &
PID=$!; PEAK=0; i=0
while kill -0 $PID 2>/dev/null && (( i < 20 )); do
  W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
  awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill"; kill -9 $PID; break; }
  sleep 5; i=$((i+1))
done
kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
echo "══ γ=0 스킵 19층, 생성 (peak ${PEAK} GB) ══"
awk '/become the capital of Japan\?$/{f=1;next} f' $EXP/sk-gen.log | grep -av "^\[ Prompt" | head -6
echo "   [$(grep -ao 'Generation: [0-9.]* t/s' $EXP/sk-gen.log | tail -1)]"
