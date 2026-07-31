#!/bin/bash
# EXP-70 k8 이 벤치를 움직이는가. ppl 이득은 0.058(잡음)이지만 축이 다를 수 있다.
# 가장 반응이 컸던 HellaSwag(400) + ARC(299) 로 확인. 같은 문항이므로 대응 검정 가능.
set -u
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
EXP=exp
MAP=$(python3 -c "import json;print(json.load(open('$EXP/maps58.json'))['Q3'])")
E="LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 LLAMA_FOLD_MAP=$MAP LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g58-Q3.txt LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1"
M=models/qwen35-v2a-hot.gguf
P=nanfix/build/bin/llama-perplexity
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
settle() { local s=0 W; while (( s < 40 )); do W=$(wired_gb); awk -v w="$W" 'BEGIN{exit !(w<8)}' && return; sleep 5; s=$((s+1)); done; }

settle
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-70 k8 HellaSwag" >> $EXP/timeline.log
env $E $P -m $M --hellaswag -f $EXP/hellaswag_val_full.txt --hellaswag-tasks 400 \
    -c 1024 -b 128 -ub 128 -ngl 99 --no-repack --seed 1337 \
    --override-kv qwen35moe.expert_used_count=int:8 > $EXP/e70-k8-hswag.log 2>&1
echo "RESULT70 k8 HellaSwag-400: $(grep -aE '^400\s' $EXP/e70-k8-hswag.log | tail -1)"

settle
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-70 k8 ARC" >> $EXP/timeline.log
env $E $P -m $M --multiple-choice -f $EXP/arc-challenge-validation.bin \
    -c 2048 -b 512 -np 8 -ngl 99 --no-repack --seed 1337 \
    --override-kv qwen35moe.expert_used_count=int:8 > $EXP/e70-k8-arc.log 2>&1
echo "RESULT70 k8 ARC-C: $(grep -a 'Final result' $EXP/e70-k8-arc.log | tail -1 | sed 's/.*Final result: //')"
echo "EXP-70 완료"
