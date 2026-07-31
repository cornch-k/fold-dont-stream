#!/bin/bash
# EXP-17 MTP 투기적 디코딩. 기준: 물리층10 + skip19 = 31.2 t/s (품질 "1868" 정상)
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
Q3=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-q3ks.gguf
MTP=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-q3ks-mtp.gguf
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-cli
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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-17 MTP spec decode" >> $EXP/timeline.log
env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
    LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g0-half_front.txt \
    LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
    "$BIN" -m "$Q3" -md "$MTP" --spec-type draft-mtp -ngld 99 \
    -p "$P" -n 150 -ngl 99 --no-repack -no-cnv -st --no-warmup \
    -c 1024 --seed 1337 --no-jinja --temp 0 --logit-bias 248068-20 \
    > $EXP/mtp1.log 2>&1 < /dev/null &
PID=$!; PEAK=0; i=0
while kill -0 $PID 2>/dev/null && (( i < 70 )); do
  W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
  awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! OOM 위험 kill"; kill -9 $PID; break; }
  sleep 5; i=$((i+1))
done
kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
echo "peak wired: ${PEAK} GB"
echo "── 출력 ──"
awk '/become the capital of Japan\?$/{f=1;next} f' $EXP/mtp1.log | grep -av "^\[ Prompt" | head -8
echo "── 속도/수용률 ──"
grep -aiE "Generation:|accept|draft|spec" $EXP/mtp1.log | grep -av "^load" | tail -8
echo "── 오류 ──"
grep -aiE "error|assert|abort|failed" $EXP/mtp1.log | head -4
