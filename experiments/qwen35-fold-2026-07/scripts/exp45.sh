#!/bin/bash
# EXP-45 꼬리 프로브: 뒤 6층 핫64 (질량 ~39%) — MMLU 가 움직이는가
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
H64=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a-hot64.gguf
PPL=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
WIKI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/scripts/wikitext-2-raw/wiki.test.raw
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
# 수술 완료 대기
for ((w=0; w<240; w++)); do grep -q "완료:" $EXP/surgery64.log 2>/dev/null && break; sleep 5; done
grep -q "완료:" $EXP/surgery64.log || { echo "수술 미완 — 중단"; exit 1; }
echo "수술 확인: $(tail -1 $EXP/surgery64.log)"

m=(); for ((il=0; il<48; il++)); do
  if (( il >= 44 )); then m[il]=$il
  else g=$(( il * 7 / 44 )); m[il]=$(( (g * 44 + 6) / 7 )); fi
done
IFS=,; MAP="${m[*]}"; unset IFS
ENVS="LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 LLAMA_FOLD_MAP=$MAP LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g32.txt LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1"
watch() { local PID=$1 MAXI=$2 PEAK=0 W i=0
  while kill -0 $PID 2>/dev/null && (( i < MAXI )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done; kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null; echo "$PEAK"; }

s=0; while (( s < 24 )); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; s=$((s+1)); done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-45 MMLU (hot64)" >> $EXP/timeline.log
env $ENVS "$PPL" -m "$H64" -ngl 99 --no-repack -c 2048 -b 512 -ub 512 --seed 1337 \
    --override-kv qwen35moe.expert_used_count=int:6 \
    --multiple-choice --multiple-choice-tasks 500 -f $EXP/mmlu-test.bin > $EXP/e45-mmlu.log 2>&1 &
PK=$(watch $! 400)
echo "hot64 MMLU-500: $(grep -a 'Final result' $EXP/e45-mmlu.log | tail -1 | sed 's/.*result: //')  peak ${PK} GB  (hot16 31.4 / 기준 31.6)"

s=0; while (( s < 24 )); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; s=$((s+1)); done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-45 ppl (hot64)" >> $EXP/timeline.log
env $ENVS "$PPL" -m "$H64" -f "$WIKI" --chunks 16 -c 512 --no-repack -b 512 -ngl 99 \
    --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e45-ppl.log 2>&1 &
PK=$(watch $! 60)
echo "hot64 ppl: $(grep -a 'Final estimate' $EXP/e45-ppl.log | tail -1 | sed 's/.*PPL = //')  peak ${PK} GB  (hot16 13.4929)"
echo "EXP-45 완료"
