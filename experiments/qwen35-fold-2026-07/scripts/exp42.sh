#!/bin/bash
# EXP-42: (a) E0 논리 라우터 ppl  (b) damage7 맵 ppl. 기준 = 현행 11층 18.9896
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
V2=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a.gguf
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
WIKI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/scripts/wikitext-2-raw/wiki.test.raw
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); G=7; F=44
for ((il=0; il<48; il++)); do
  if (( il >= F )); then m[il]=$il
  else g=$(( il * G / F )); m[il]=$(( (g * F + G - 1) / G )); fi
done
IFS=,; BASEMAP="${m[*]}"; unset IFS
DMGMAP="4,4,4,4,4,5,6,6,6,6,6,6,6,6,6,15,16,17,17,17,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,44,45,46,47"
# damage7 γ: 대표/고정 0.3, 접힌 il<23 0, 그 외 0.3
python3 - <<'PY'
dmg = [int(x) for x in "4,4,4,4,4,5,6,6,6,6,6,6,6,6,6,15,16,17,17,17,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,44,45,46,47".split(',')]
with open('/Users/angigyeom/Desktop/DEV/fun/accelerator/exp/g42-dmg.txt','w') as f:
    for il in range(48):
        if dmg[il] == il: f.write("0.3\n")
        elif il < 23:     f.write("0.0\n")
        else:             f.write("0.3\n")
PY

run() { # $1=맵 $2=γ파일 $3=추가env $4=태그
  local LOG=$EXP/e42-$4.log
  local s W0
  for ((s=0; s<24; s++)); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; done
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-42 $4" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP="$1" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$2 $3 \
      LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      "$BIN" -m "$V2" -f "$WIKI" --chunks 16 -c 512 --no-repack -b 512 -ngl 99 \
      --override-kv qwen35moe.expert_used_count=int:6 > "$LOG" 2>&1 &
  local PID=$! W PEAK=0 i=0
  while kill -0 $PID 2>/dev/null && (( i < 60 )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill $4"; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  wait $PID 2>/dev/null
  printf '%-14s %-24s peak %s GB\n' "$4" "$(grep -a 'Final estimate' "$LOG" | tail -1 | sed 's/.*PPL = //')" "$PEAK"
}
printf '%-14s %-24s\n' "기준" "18.9896 +/- 0.82529"
run "$BASEMAP" "$EXP/g32.txt" "LLAMA_FOLD_ROUTER_LOGICAL=1" "E0-router"
run "$DMGMAP" "$EXP/g42-dmg.txt" "" "damage7"
