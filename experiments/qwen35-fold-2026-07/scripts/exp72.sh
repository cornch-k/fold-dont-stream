#!/bin/bash
# EXP-72 미접힘 MMLU(우도) 기준선 — 지식 축의 접기 비용을 처음 정량화.
# 생성형은 CPU 에서 문항당 10분이라 불가. 우도는 생성이 없어 감당 가능.
set -u
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
EXP=exp
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-72 미접힘 MMLU-200" >> $EXP/timeline.log
env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 \
    nanfix/build/bin/llama-perplexity -m models/qwen35-v2a.gguf \
    --multiple-choice -f $EXP/mmlu-test.bin --multiple-choice-tasks 200 \
    -c 2048 -b 64 -ub 64 --device none -ngl 0 --no-repack --seed 1337 \
    --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e72-mmlu.log 2>&1
echo "RESULT72 미접힘 MMLU: $(tr -d '\0' < $EXP/e72-mmlu.log | grep -aE '^[0-9]+\s' | tail -1)"
echo "EXP-72 완료"
