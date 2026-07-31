#!/bin/bash
# EXP-60 F 구성 기록급 벤치 + 추측 없는 순수 디코드 속도(보장용 하한)
set -u
KILL_GB=24
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
EXP=exp
M=models/qwen35-v2a-hot.gguf
PPL=nanfix/build/bin/llama-perplexity
CLI=nanfix/build/bin/llama-cli
MMLU=$EXP/mmlu-test.bin
HS=$EXP/hellaswag_val_full.txt
WG=$EXP/winogrande-debiased-eval.csv
ARC=$EXP/arc-challenge-validation.bin
TAG=${1:-F}
MAP=$(python3 -c "import json;print(json.load(open('$EXP/maps58.json'))['$TAG'])")
GAM=$EXP/g58-$TAG.txt
E="LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 LLAMA_FOLD_MAP=$MAP LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$GAM LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1"
K="--override-kv qwen35moe.expert_used_count=int:6"
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
settle() { local s=0 W; while (( s < 40 )); do W=$(wired_gb); awk -v w="$W" 'BEGIN{exit !(w<9)}' && return; sleep 5; s=$((s+1)); done; }
watch() { local PID=$1 MAXI=$2 i=0 PW=0 W
  while kill -0 $PID 2>/dev/null && (( i < MAXI )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PW" 'BEGIN{exit !(w>p)}' && PW=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!!GUARD" >&2; kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done; kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null; echo "$PW"; }

P='The history of computing began with mechanical calculators. Charles Babbage designed the Analytical Engine in the 1830s, and Ada Lovelace wrote algorithms for it. In the twentieth century, electronic computers replaced mechanical ones. The story continues:'
settle
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-60 F 순수 디코드 속도" >> $EXP/timeline.log
env $E $CLI -m $M -p "$P" -n 200 --ignore-eos -ngl 99 --no-repack -no-cnv -st --no-warmup \
    -c 1024 --seed 1337 --no-jinja --temp 0 $K > $EXP/e60-$TAG-spd-nospec.log 2>&1 < /dev/null &
PW=$(watch $! 90)
echo "RESULT 순수디코드(추측없음): $(grep -ao 'Generation: [0-9.]* t/s' $EXP/e60-$TAG-spd-nospec.log | tail -1)  wired=$PW"

settle
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-60 F MMLU-2000" >> $EXP/timeline.log
env $E $PPL -m $M --multiple-choice -f $MMLU --multiple-choice-tasks 2000 -c 2048 -b 512 -ngl 99 \
    --no-repack --seed 1337 $K > $EXP/e60-$TAG-mmlu.log 2>&1 &
PW=$(watch $! 400)
echo "RESULT MMLU-2000: $(grep -a 'Final result' $EXP/e60-$TAG-mmlu.log | tail -1 | sed 's/.*Final result: //')  wired=$PW"

settle
env $E $PPL -m $M --hellaswag -f $HS --hellaswag-tasks 400 -c 1024 -b 128 -ub 128 -ngl 99 \
    --no-repack --seed 1337 $K > $EXP/e60-$TAG-hswag.log 2>&1 &
PW=$(watch $! 300)
echo "RESULT HellaSwag-400: $(grep -aE '^(399|400)\s' $EXP/e60-$TAG-hswag.log | tail -1)  wired=$PW"

settle
env $E $PPL -m $M --winogrande -f $WG -c 1024 -b 128 -ub 128 -ngl 99 \
    --no-repack --seed 1337 $K > $EXP/e60-$TAG-wino.log 2>&1 &
PW=$(watch $! 400)
echo "RESULT Winogrande: $(grep -a 'Final Winogrande score' $EXP/e60-$TAG-wino.log | tail -1 | sed 's/.*score//')  wired=$PW"

settle
env $E $PPL -m $M --multiple-choice -f $ARC -c 2048 -b 512 -np 8 -ngl 99 \
    --no-repack --seed 1337 $K > $EXP/e60-$TAG-arc.log 2>&1 &
PW=$(watch $! 400)
echo "RESULT ARC-C: $(grep -a 'Final result' $EXP/e60-$TAG-arc.log | tail -1 | sed 's/.*Final result: //')  wired=$PW"
echo "EXP-60 완료 (구성 $TAG)"
