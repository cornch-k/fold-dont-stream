#!/bin/bash
# EXP-18 MTP 속도 정밀 측정. --ignore-eos 로 200토큰 강제 생성, MTP off/on × n-max 스윕.
# 동일 시드/그리디 → 두 팔이 같은 텍스트를 생성 → 공정 A/B.
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

run() { # $1=추가옵션 $2=태그
  local LOG=$EXP/mtp2-$2.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-18 $2 : $1" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g0-half_front.txt \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$BIN" -m "$Q3" -p "$P" -n 200 --ignore-eos -ngl 99 --no-repack -no-cnv -st --no-warmup \
      -c 1024 --seed 1337 --no-jinja --temp 0 --logit-bias 248068-20 $1 \
      > "$LOG" 2>&1 < /dev/null &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < 70 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  printf '%-14s %-28s peak %s GB\n' "$2" "$(grep -ao 'Generation: [0-9.]* t/s' "$LOG" | tail -1)" "$PEAK"
  sleep 2
}

run "" "no-mtp"
run "-md $MTP --spec-type draft-mtp -ngld 99" "mtp-n3"
run "-md $MTP --spec-type draft-mtp -ngld 99 --spec-draft-n-max 1" "mtp-n1"
run "-md $MTP --spec-type draft-mtp -ngld 99 --spec-draft-n-max 6" "mtp-n6"
