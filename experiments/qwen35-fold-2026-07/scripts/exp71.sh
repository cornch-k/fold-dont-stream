#!/bin/bash
# EXP-71 미접힘 122B 기준선 — 캠페인 최대 설계 누락. 접기 손실의 진짜 크기를 처음 측정.
# 78.64GB 를 16GiB 에 못 올리므로 -ngl 로 일부만 GPU, 나머지는 CPU/mmap. 느리지만 소수 태스크는 가능.
set -u
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
EXP=exp
M=models/qwen35-v2a.gguf
P=nanfix/build/bin/llama-perplexity
# 접기 없음(LLAMA_FOLD_MAP 미설정) + GPU 층 제한으로 메모리 상한 유지
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-71 미접힘 HellaSwag-40" >> $EXP/timeline.log
env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 \
    $P -m $M --hellaswag -f $EXP/hellaswag_val_full.txt --hellaswag-tasks 40 \
    -c 1024 -b 64 -ub 64 --device none -ngl 0 --no-repack --seed 1337 \
    --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e71-hswag.log 2>&1
echo "RESULT71 미접힘 HellaSwag-40: $(grep -aE '^(39|40)\s' $EXP/e71-hswag.log | tail -1)"
echo "EXP-71 완료"
