#!/bin/bash
# EXP-41 층별 접기 손상 센서스: ||ffn_out(원본) − ffn_out(접힘)|| / ||원본||, 37개 접힌 층 전수.
#   run0 = 배포 맵에서 전 층 ffn_out 덤프 (1회)
#   run_il = map[il]=il 로 층 il 만 안 접고 ffn_out-il 덤프 (37회)
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
V2=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a.gguf
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-debug
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }

mkmap() { local U=$1 il g; local -a m=()
  for ((il=0; il<48; il++)); do
    if (( il >= 44 )); then m[il]=$il
    else g=$(( il * 7 / 44 )); m[il]=$(( (g * 44 + 6) / 7 )); fi
  done
  [ "$U" != "-" ] && m[$U]=$U
  local IFS=,; echo "${m[*]}"; }

run() { # $1=맵 $2=디렉터리 $3=필터인자들 $4=태그
  rm -rf "$2"; mkdir -p "$2"
  local s W0
  for ((s=0; s<24; s++)); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; done
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$1" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g32.txt \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 LLAMA_DEBUG_SAVE_DIR="$2" \
      "$BIN" -m "$V2" -f $EXP/calib8k.txt -ngl 99 --no-repack -c 8192 -b 8192 -ub 512 \
      $3 --override-kv qwen35moe.expert_used_count=int:6 > $EXP/cen-$4.log 2>&1 &
  local PID=$! W i=0
  while kill -0 $PID 2>/dev/null && (( i < 60 )); do
    W=$(wired_gb)
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill $4"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  echo "[$(date '+%H:%M:%S')] $4 완료 ($(ls "$2" | wc -l | tr -d ' ')파일)"
}

BASE=$(mkmap -)
FOLDED=$(echo "$BASE" | tr ',' '\n' | awk '{if ($1 != NR-1) print NR-1}')
FILT=""
for il in $FOLDED; do FILT="$FILT --tensor-filter ffn_out-${il}\$"; done
echo "접힌 층: $(echo $FOLDED | tr '\n' ' ')"
run "$BASE" "$EXP/cen-base" "$FILT" "base"

for il in $FOLDED; do
  run "$(mkmap $il)" "$EXP/cen-$il" "--tensor-filter ffn_out-${il}\$" "u$il"
done
echo "센서스 덤프 전체 완료"
