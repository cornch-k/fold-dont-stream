#!/bin/bash
# EXP-44: 핫 구성의 (a) MMLU-500 (기준 31.6) (b) 지속 생성 속도 (기준 25.8 t/s)
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
HOT=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a-hot.gguf
PPL=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
CLI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-cli
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
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
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-44 MMLU" >> $EXP/timeline.log
env $ENVS "$PPL" -m "$HOT" -ngl 99 --no-repack -c 2048 -b 512 -ub 512 --seed 1337 \
    --override-kv qwen35moe.expert_used_count=int:6 \
    --multiple-choice --multiple-choice-tasks 500 -f $EXP/mmlu-test.bin > $EXP/e44-mmlu.log 2>&1 &
PK=$(watch $! 400)
echo "MMLU-500: $(grep -a 'Final result' $EXP/e44-mmlu.log | tail -1 | sed 's/.*result: //')   peak ${PK} GB   (기준 31.6 ± 2.1)"

P='The history of computing began with mechanical calculators. Charles Babbage designed the Analytical Engine in the 1830s, and Ada Lovelace wrote algorithms for it. In the twentieth century, electronic computers replaced mechanical ones. The story continues:'
s=0; while (( s < 24 )); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; s=$((s+1)); done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-44 속도" >> $EXP/timeline.log
env $ENVS "$CLI" -m "$HOT" -p "$P" -n 300 --ignore-eos -ngl 99 --no-repack -no-cnv -st --no-warmup \
    -c 1024 --seed 1337 --no-jinja --temp 0 --repeat-penalty 1.15 --logit-bias 248068-20 \
    --spec-type ngram-simple --spec-draft-n-max 12 \
    --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e44-speed.log 2>&1 < /dev/null &
PK=$(watch $! 80)
echo "속도: $(grep -ao 'Generation: [0-9.]* t/s' $EXP/e44-speed.log | tail -1)   peak ${PK} GB   (기준 25.8 t/s)"
echo "EXP-44 완료"
