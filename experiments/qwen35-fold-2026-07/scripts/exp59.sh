#!/bin/bash
# EXP-58 전문가 예산 재배치: 앞쪽 1:1 대표(0,8,15 = 2.90 GiB)를 뒤쪽 해상도로 바꾼다.
set -u
KILL_GB=24
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
EXP=exp
PPL=nanfix/build/bin/llama-perplexity
CLI=nanfix/build/bin/llama-cli
WIKI=nanfix/scripts/wikitext-2-raw/wiki.test.raw
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
settle() { local s=0 W; while (( s < 40 )); do W=$(wired_gb); awk -v w="$W" 'BEGIN{exit !(w<9)}' && return; sleep 5; s=$((s+1)); done; }
watch() { local PID=$1 MAXI=$2 i=0 PW=0 W
  while kill -0 $PID 2>/dev/null && (( i < MAXI )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PW" 'BEGIN{exit !(w>p)}' && PW=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done; kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null; echo "$PW"; }
P='The history of computing began with mechanical calculators. Charles Babbage designed the Analytical Engine in the 1830s, and Ada Lovelace wrote algorithms for it. In the twentieth century, electronic computers replaced mechanical ones. The story continues:'

run() { # $1=tag $2=model
  local TAG=$1 M=$2
  local MAP=$(python3 -c "import json;print(json.load(open('$EXP/maps58.json'))['$TAG'])")
  local E="LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 LLAMA_FOLD_MAP=$MAP LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g58-$TAG.txt LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1"
  settle
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-58 $TAG ppl" >> $EXP/timeline.log
  env $E $PPL -m $M -f $WIKI --chunks 16 -c 512 --no-repack -b 512 -ngl 99 \
      --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e58-$TAG-ppl.log 2>&1 &
  local PW=$(watch $! 80)
  local PPLV=$(grep -a 'Final estimate' $EXP/e58-$TAG-ppl.log | tail -1 | sed 's/.*PPL = //')
  settle
  env $E $CLI -m $M -p "$P" -n 300 --ignore-eos -ngl 99 --no-repack -no-cnv -st --no-warmup \
      -c 1024 --seed 1337 --no-jinja --temp 0 --repeat-penalty 1.15 --logit-bias 248068-20 \
      --spec-type ngram-simple --spec-draft-n-max 12 \
      --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e58-$TAG-spd.log 2>&1 < /dev/null &
  local PW2=$(watch $! 90)
  local SPD=$(grep -ao 'Generation: [0-9.]* t/s' $EXP/e58-$TAG-spd.log | tail -1 | sed 's/Generation: //')
  local MEM=$(python3 $EXP/memcalc58.py $EXP/e58-$TAG-ppl.log "$M")
  echo "RESULT $TAG  ppl=$PPLV  속도=$SPD  $MEM  wired=$PW2"
}

run F4 models/qwen35-v2a-hot.gguf
run F5 models/qwen35-v2a-hot.gguf
run Ffront models/qwen35-v2a-hot.gguf
run Fpos models/qwen35-v2a-hot.gguf


echo "EXP-59 완료"
