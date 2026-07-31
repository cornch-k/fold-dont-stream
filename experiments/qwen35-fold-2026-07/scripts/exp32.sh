#!/bin/bash
# EXP-32 v2a 11층 (7그룹 + 고정 44-47): 예산·속도 회복 + ppl 손실 확인
# 비교: v2a 12층 ppl 19.1161 / 49.6 t/s / llama 16.3
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
V2=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a.gguf
PPL=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
CLI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-cli
WIKI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/scripts/wikitext-2-raw/wiki.test.raw
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); G=7; F=44
for ((il=0; il<48; il++)); do
  if (( il >= F )); then m[il]=$il
  else g=$(( il * G / F )); m[il]=$(( (g * F + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS
echo "유일층: $(echo "$MAP" | tr ',' '\n' | sort -un | tr '\n' ' ')"
# γ 파일: 대표/고정 0.3, 접힌 il<23 0.0, 그 외 0.3
w=(${MAP//,/ }); : > $EXP/g32.txt
for ((il=0; il<48; il++)); do
  if [ "${w[il]}" = "$il" ]; then echo "0.3"
  elif (( il < 23 )); then echo "0.0"
  else echo "0.3"; fi
done >> $EXP/g32.txt

echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-32 11층 ppl" >> $EXP/timeline.log
env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
    LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g32.txt \
    LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
    "$PPL" -m "$V2" -f "$WIKI" --chunks 16 -c 512 --no-repack -b 512 -ngl 99 \
    --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e32-ppl.log 2>&1 &
PID=$!; PEAK=0; i=0
while kill -0 $PID 2>/dev/null && (( i < 60 )); do
  W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
  awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { kill -9 $PID; break; }
  sleep 5; i=$((i+1))
done
wait $PID 2>/dev/null
echo "11층 ppl: $(grep -a 'Final estimate' $EXP/e32-ppl.log | tail -1 | sed 's/.*PPL = //')"

STORY='The history of computing began with mechanical calculators. Charles Babbage designed the Analytical Engine in the 1830s, and Ada Lovelace wrote algorithms for it. In the twentieth century, electronic computers replaced mechanical ones. The story continues:'
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-32 11층 속도" >> $EXP/timeline.log
env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
    LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g32.txt \
    LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
    "$CLI" -m "$V2" -p "$STORY" -n 300 --ignore-eos -ngl 99 --no-repack -no-cnv -st --no-warmup \
    -c 1024 --seed 1337 --no-jinja --temp 0 --repeat-penalty 1.15 --logit-bias 248068-20 \
    --spec-type ngram-simple --spec-draft-n-max 12 \
    --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e32-speed.log 2>&1 < /dev/null &
PID=$!; PEAK=0; i=0
while kill -0 $PID 2>/dev/null && (( i < 80 )); do
  W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
  awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { kill -9 $PID; break; }
  sleep 5; i=$((i+1))
done
kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
printf '11층 속도: %s   peak %s GB (llama≈%.1f)\n' "$(grep -ao 'Generation: [0-9.]* t/s' $EXP/e32-speed.log | tail -1)" "$PEAK" "$(echo "$PEAK - 2.2" | bc)"
