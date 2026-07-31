#!/bin/bash
# EXP-55 pool32 예산 정합: 5-rep 맵 ppl + 속도. 비교: pool32@6rep 13.35(초과) / A 14.66(예산내)
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
HP=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a-hotP.gguf
PPL=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
CLI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-cli
WIKI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/scripts/wikitext-2-raw/wiki.test.raw
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); for ((il=0; il<48; il++)); do
  if (( il >= 44 )); then m[il]=$il
  else g=$(( il * 5 / 44 )); m[il]=$(( (g * 44 + 4) / 5 )); fi
done
IFS=,; MAP5="${m[*]}"; unset IFS
# γ 파일 (맵 의존)
w=(${MAP5//,/ }); : > $EXP/g55.txt
for ((il=0; il<48; il++)); do
  if [ "${w[il]}" = "$il" ]; then echo "0.3"
  elif (( il < 23 )); then echo "0.0"
  else echo "0.3"; fi
done >> $EXP/g55.txt
echo "5-rep 유일층: $(echo "$MAP5" | tr ',' '\n' | sort -un | tr '\n' ' ')"
ENVS="LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 LLAMA_FOLD_MAP=$MAP5 LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g55.txt LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1"
watch() { local PID=$1 MAXI=$2 PEAK=0 W i=0
  while kill -0 $PID 2>/dev/null && (( i < MAXI )); do
    W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
    awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { kill -9 $PID; break; }
    sleep 5; i=$((i+1))
  done; kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null; echo "$PEAK"; }

s=0; while (( s < 24 )); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; s=$((s+1)); done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-55 ppl" >> $EXP/timeline.log
env $ENVS "$PPL" -m "$HP" -f "$WIKI" --chunks 16 -c 512 --no-repack -b 512 -ngl 99 \
    --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e55-ppl.log 2>&1 &
PK=$(watch $! 60)
echo "p32@5rep ppl: $(grep -a 'Final estimate' $EXP/e55-ppl.log | tail -1 | sed 's/.*PPL = //')  peak ${PK} GB"

P='The history of computing began with mechanical calculators. Charles Babbage designed the Analytical Engine in the 1830s, and Ada Lovelace wrote algorithms for it. In the twentieth century, electronic computers replaced mechanical ones. The story continues:'
s=0; while (( s < 24 )); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; s=$((s+1)); done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-55 속도" >> $EXP/timeline.log
env $ENVS "$CLI" -m "$HP" -p "$P" -n 300 --ignore-eos -ngl 99 --no-repack -no-cnv -st --no-warmup \
    -c 1024 --seed 1337 --no-jinja --temp 0 --repeat-penalty 1.15 --logit-bias 248068-20 \
    --spec-type ngram-simple --spec-draft-n-max 12 \
    --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e55-speed.log 2>&1 < /dev/null &
PK=$(watch $! 80)
echo "p32@5rep 속도: $(grep -ao 'Generation: [0-9.]* t/s' $EXP/e55-speed.log | tail -1)  peak ${PK} GB"
echo "EXP-55 완료"
