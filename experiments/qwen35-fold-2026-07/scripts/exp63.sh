#!/bin/bash
# EXP-63 상위 구성 확정: 40청크로 재측정해 16청크 잡음(±0.2)을 걷어낸다
set -u
KILL_GB=24
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
EXP=exp
PPL=nanfix/build/bin/llama-perplexity
WIKI=nanfix/scripts/wikitext-2-raw/wiki.test.raw
M=models/qwen35-v2a-hot.gguf
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
settle() { local s=0 W; while (( s < 40 )); do W=$(wired_gb); awk -v w="$W" 'BEGIN{exit !(w<9)}' && return; sleep 5; s=$((s+1)); done; }
for TAG in Q2 F Q1; do
  MAP=$(python3 -c "import json;print(json.load(open('$EXP/maps58.json'))['$TAG'])")
  settle
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-63 $TAG 40청크" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP=$MAP LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g58-$TAG.txt \
      LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      $PPL -m $M -f $WIKI --chunks 40 -c 512 --no-repack -b 512 -ngl 99 \
      --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e63-$TAG.log 2>&1 &
  PID=$!; i=0
  while kill -0 $PID 2>/dev/null && (( i < 120 )); do
    W=$(wired_gb); awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  echo "RESULT63 $TAG 40청크 ppl=$(grep -a 'Final estimate' $EXP/e63-$TAG.log | tail -1 | sed 's/.*PPL = //')"
done

# 순수 디코드(추측 없음) 비교: 기존 표준 A vs 신구성 — 33.4 t/s 가 ngram 덕임을 정량화
CLI=nanfix/build/bin/llama-cli
P='The history of computing began with mechanical calculators. Charles Babbage designed the Analytical Engine in the 1830s, and Ada Lovelace wrote algorithms for it. In the twentieth century, electronic computers replaced mechanical ones. The story continues:'
for TAG in A Q3; do
  MAP=$(python3 -c "import json;print(json.load(open('$EXP/maps58.json'))['$TAG'])")
  settle
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP=$MAP LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g58-$TAG.txt \
      LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      $CLI -m $M -p "$P" -n 200 --ignore-eos -ngl 99 --no-repack -no-cnv -st --no-warmup \
      -c 1024 --seed 1337 --no-jinja --temp 0 \
      --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e63-$TAG-nospec.log 2>&1 < /dev/null &
  PID=$!; i=0
  while kill -0 $PID 2>/dev/null && (( i < 90 )); do
    W=$(wired_gb); awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done
  kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
  echo "RESULT63 $TAG 순수디코드 $(grep -ao 'Generation: [0-9.]* t/s' $EXP/e63-$TAG-nospec.log | tail -1)"
done
echo "EXP-63 완료"
