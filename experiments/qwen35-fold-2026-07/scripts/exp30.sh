#!/bin/bash
# EXP-30 물리층 10→12: 남은 예산 2GB를 층에 투자. 두 배치안 비교.
#   A) 접힌 9그룹 + 고정 45,46,47   (12층)
#   B) 접힌 7그룹 + 고정 43~47      (12층)
# 기준: 물리층 10 (7그룹+3고정) = 22.6446 ± 1.02794 (γ=0.3 균일) / 23.9036 (skip19)
# 주의: γ파일은 맵에 맞춰 재생성 (대표층은 γ=0.3, 접힌 il<23 은 0.0)
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
Q3=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-q3ks.gguf
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
WIKI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/scripts/wikitext-2-raw/wiki.test.raw
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }

mkmap() { # $1=그룹수 $2=고정시작
  local G=$1 F=$2 il g; local -a m=()
  for ((il=0; il<48; il++)); do
    if (( il >= F )); then m[il]=$il
    else g=$(( il * G / F )); m[il]=$(( (g * F + G - 1) / G )); fi
  done
  local IFS=,; echo "${m[*]}"
}
mkgam() { # $1=MAP → stdout γ파일 (대표/고정=0.3, 접힌 il<23=0, 접힌 il>=23=0.3)
  local MAP=$1; local -a w=(${MAP//,/ }); local il
  for ((il=0; il<48; il++)); do
    if [ "${w[il]}" = "$il" ]; then echo "0.3"
    elif (( il < 23 )); then echo "0.0"
    else echo "0.3"; fi
  done
}

run() { # $1=MAP $2=γ파일 $3=태그
  local LOG=$EXP/e30-$3.log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-30 $3" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$1" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$2 \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$BIN" -m "$Q3" -f "$WIKI" --chunks 16 -c 512 --no-repack -b 512 -ngl 99 \
      --override-kv qwen35moe.expert_used_count=int:6 > "$LOG" 2>&1 &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < 60 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  wait $PID 2>/dev/null
  printf '%-10s %-24s peak %s GB\n' "$3" "$(grep -a 'Final estimate' "$LOG" | tail -1 | sed 's/.*PPL = //')" "$PEAK"
  sleep 3
}

MA=$(mkmap 9 45); mkgam "$MA" > $EXP/g30-A.txt
MB=$(mkmap 7 43); mkgam "$MB" > $EXP/g30-B.txt
echo "A 유일층: $(echo "$MA" | tr ',' '\n' | sort -un | tr '\n' ' ')"
echo "B 유일층: $(echo "$MB" | tr ',' '\n' | sort -un | tr '\n' ' ')"
printf '%-10s %-24s\n' "기준(10층,k6)" "23.4146 +/- 1.05924"
run "$MA" $EXP/g30-A.txt "A-9g3f"
run "$MB" $EXP/g30-B.txt "B-7g5f"
