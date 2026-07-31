#!/bin/bash
# EXP-34 최종 후보 격자:
#   (i)  C(11층) + lean사이드카 + n-max 4/5  — 속도 상한 확인 (예산은 초과 상태)
#   (ii) D(10층 v2a) + lean사이드카 + n3     — 16GB 진입 여부 + ppl + 속도
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
V2=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a.gguf
MTP=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a-mtp.gguf
CLI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-cli
PPL=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
WIKI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/scripts/wikitext-2-raw/wiki.test.raw
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }

mkmap() { local G=$1 F=$2 il g; local -a m=()
  for ((il=0; il<48; il++)); do
    if (( il >= F )); then m[il]=$il
    else g=$(( il * G / F )); m[il]=$(( (g * F + G - 1) / G )); fi
  done; local IFS=,; echo "${m[*]}"; }
mkgam() { local MAP=$1; local -a w=(${MAP//,/ }); local il
  for ((il=0; il<48; il++)); do
    if [ "${w[il]}" = "$il" ]; then echo "0.3"
    elif (( il < 23 )); then echo "0.0"
    else echo "0.3"; fi
  done; }

M11=$(mkmap 7 44); mkgam "$M11" > $EXP/g34-11.txt
M10=$(mkmap 7 45); mkgam "$M10" > $EXP/g34-10.txt

STORY='The history of computing began with mechanical calculators. Charles Babbage designed the Analytical Engine in the 1830s, and Ada Lovelace wrote algorithms for it. In the twentieth century, electronic computers replaced mechanical ones. The story continues:'

speed() { # $1=MAP $2=γ $3=spec옵션 $4=태그
  local LOG=$EXP/e34-$4.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-34 $4" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$1" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$2 \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$CLI" -m "$V2" -p "$STORY" -n 300 --ignore-eos -ngl 99 --no-repack -no-cnv -st --no-warmup \
      -c 1024 --seed 1337 --no-jinja --temp 0 --repeat-penalty 1.15 --logit-bias 248068-20 \
      --override-kv qwen35moe.expert_used_count=int:6 $3 > "$LOG" 2>&1 < /dev/null &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < 80 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  printf '%-14s %-26s peak %s GB (llama≈%.1f)\n' "$4" "$(grep -ao 'Generation: [0-9.]* t/s' "$LOG" | tail -1)" "$PEAK" "$(echo "$PEAK - 2.2" | bc)"
  sleep 2
}

echo "── (i) 11층 + MTP n-max 스윕 ──"
speed "$M11" $EXP/g34-11.txt "-md $MTP --spec-type draft-mtp -ngld 99 --spec-draft-n-max 4" "11-n4"
speed "$M11" $EXP/g34-11.txt "-md $MTP --spec-type draft-mtp -ngld 99 --spec-draft-n-max 5" "11-n5"
echo "── (ii) 10층 v2a + MTP n3 ──"
speed "$M10" $EXP/g34-10.txt "-md $MTP --spec-type draft-mtp -ngld 99 --spec-draft-n-max 3" "10-n3"

echo "── 10층 v2a ppl ──"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-34 10층 ppl" >> $EXP/timeline.log
env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
    LLAMA_FOLD_MAP="$M10" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g34-10.txt \
    LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
    "$PPL" -m "$V2" -f "$WIKI" --chunks 16 -c 512 --no-repack -b 512 -ngl 99 \
    --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e34-ppl10.log 2>&1 &
PID=$!; PEAK=0; i=0
while kill -0 $PID 2>/dev/null && (( i < 60 )); do
  W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
  awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { kill -9 $PID; break; }
  sleep 5; i=$((i+1))
done
wait $PID 2>/dev/null
echo "10층 v2a ppl: $(grep -a 'Final estimate' $EXP/e34-ppl10.log | tail -1 | sed 's/.*PPL = //')   (11층 18.9896)"
