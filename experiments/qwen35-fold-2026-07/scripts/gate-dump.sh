#!/bin/bash
# EXP-38a 게이트 덤프: 층 30의 (입력, 접힌 출력, 원본 출력) 수집
#   run1 = 배포 맵(층30 접힘, wl=26)   run2 = 층30만 안 접은 맵
#   층 0-29 는 두 맵에서 동일 → 층30 입력 동일 → R = y_orig − y_fold 성립
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
V2=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a.gguf
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-debug
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }

mkmap() { # $1=unfold_il ("-"=없음)
  local U=$1 il g; local -a m=()
  for ((il=0; il<48; il++)); do
    if (( il >= 44 )); then m[il]=$il
    else g=$(( il * 7 / 44 )); m[il]=$(( (g * 44 + 6) / 7 )); fi
  done
  [ "$U" != "-" ] && m[$U]=$U
  local IFS=,; echo "${m[*]}"
}

run() { # $1=맵 $2=덤프디렉터리 $3=태그
  rm -rf "$2"; mkdir -p "$2"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-38a $3" >> $EXP/timeline.log
  local s W0
  for ((s=0; s<24; s++)); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; done
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$1" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g32.txt \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      LLAMA_DEBUG_SAVE_DIR="$2" \
      "$BIN" -m "$V2" -f $EXP/calib8k.txt -ngl 99 --no-repack -c 8192 -b 8192 -ub 512 \
      --tensor-filter 'attn_residual-30$' --tensor-filter 'ffn_out-30$' \
      --override-kv qwen35moe.expert_used_count=int:6 > $EXP/gd-$3.log 2>&1 &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < 120 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill $3"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  echo "$3: peak ${PEAK} GB, 파일 $(ls "$2" | wc -l | tr -d ' ')개, $(du -sh "$2" | cut -f1)"
}

run "$(mkmap -)"  "$EXP/acts-fold" "fold"
run "$(mkmap 30)" "$EXP/acts-orig" "orig30"
