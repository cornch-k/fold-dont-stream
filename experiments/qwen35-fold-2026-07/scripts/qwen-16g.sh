#!/bin/bash
# EXP-04 목표 구성(16GB, 물리층 5개) + EOS bias 검증.
# 대조: 물리층 8개(19.6GB)에서 EOS+5 가 정상 종료 확인됨.
set -u
KILL_GB=22
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-cli
QWEN=/Users/angigyeom/Desktop/DEV/optillama/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }

# 물리층 5개: 0,23,45,46,47  (16 GB 목표 구성)
m=()
for ((il=0; il<48; il++)); do
  if   (( il >= 45 )); then m[il]=$il
  elif (( il <  23 )); then m[il]=0
  else m[il]=23; fi
done
IFS=,; MAP="${m[*]}"; unset IFS

gen() { # $1=bias $2=태그 $3=프롬프트
  local LOG=$EXP/g16-$2.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-04 $2 bias=$1" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA=0.3 \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$BIN" -m "$QWEN" -p "$3" -n 200 -ngl 99 --no-repack -no-cnv -st --no-warmup \
      -c 1024 --seed 1337 --no-jinja --temp 0 --logit-bias 248046+$1 > "$LOG" 2>&1 < /dev/null &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < 20 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! OOM 위험 kill"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  echo "══ $2  bias+$1  (peak ${PEAK} GB) ══"
  awk '/^A:$|^A: /{f=1} f' "$LOG" | grep -av "^\[ Prompt" | head -10
  echo "   [$(grep -ao 'Generation: [0-9.]* t/s' "$LOG" | tail -1)]"; echo
  sleep 2
}

P1='Tokyo became the capital of Japan in 1868, when Emperor Meiji moved the court from Kyoto.

Q: In what year did Tokyo become the capital of Japan?
A:'
P2='Q: What is the largest planet in our solar system?
A:'

gen 5 "p1-b5"  "$P1"
gen 4 "p1-b4"  "$P1"
gen 6 "p1-b6"  "$P1"
gen 5 "p2-b5"  "$P2"
