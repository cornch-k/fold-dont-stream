#!/bin/bash
# EXP-33 MTP 재시험 @ C구성 (v2a 11층, ppl 18.99)
# 가설: 접힌 모델 품질이 오르면(23.4→19.0) 무접기를 예측하는 MTP draft 수용률이 회복된다.
# 기준: C + ngram = 25.8 t/s. (이전 MTP는 ppl 22.6 구성에서 실텍스트 20.3으로 역효과)
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
V2=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a.gguf
MTP=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-q3ks-mtp.gguf
CLI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-cli
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); G=7; F=44
for ((il=0; il<48; il++)); do
  if (( il >= F )); then m[il]=$il
  else g=$(( il * G / F )); m[il]=$(( (g * F + G - 1) / G )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS

STORY='The history of computing began with mechanical calculators. Charles Babbage designed the Analytical Engine in the 1830s, and Ada Lovelace wrote algorithms for it. In the twentieth century, electronic computers replaced mechanical ones. The story continues:'

run() { # $1=spec옵션 $2=태그
  local LOG=$EXP/e33-$2.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-33 $2" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g32.txt \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$CLI" -m "$V2" -p "$STORY" -n 300 --ignore-eos -ngl 99 --no-repack -no-cnv -st --no-warmup \
      -c 1024 --seed 1337 --no-jinja --temp 0 --repeat-penalty 1.15 --logit-bias 248068-20 \
      --override-kv qwen35moe.expert_used_count=int:6 $1 > "$LOG" 2>&1 < /dev/null &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < 80 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  printf '%-12s %-26s peak %s GB\n' "$2" "$(grep -ao 'Generation: [0-9.]* t/s' "$LOG" | tail -1)" "$PEAK"
  sleep 2
}

printf '%-12s %-26s\n' "ngram(기준)" "25.8 t/s"
run "-md $MTP --spec-type draft-mtp -ngld 99 --spec-draft-n-max 2" "mtp-n2"
run "-md $MTP --spec-type draft-mtp -ngld 99 --spec-draft-n-max 3" "mtp-n3"
run "-md $MTP --spec-type draft-mtp -ngld 99 --spec-draft-n-max 3 --spec-type draft-mtp,ngram-simple" "mtp+ngram"
