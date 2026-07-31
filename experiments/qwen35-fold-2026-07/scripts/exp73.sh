#!/bin/bash
# EXP-73 산출물 질적 검증: 최종 구성 Q3 가 실제로 정상 답변을 내는가.
# 벤치 점수만 쟀고 실제 생성물을 확인한 적이 없다. 원 요구사항은 "정상 구동"이었다.
set -u
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
EXP=exp
MAP=$(python3 -c "import json;print(json.load(open('$EXP/maps58.json'))['Q3'])")
E="LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 LLAMA_FOLD_MAP=$MAP LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g58-Q3.txt LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1"
K="--override-kv qwen35moe.expert_used_count=int:6"
for Q in "What is the capital of France? Answer in one sentence." "파이썬으로 리스트를 정렬하는 코드를 한 줄로 써줘." "2+2는 얼마인가?"; do
  echo "═══ 질문: $Q"
  env $E nanfix/build/bin/llama-cli -m models/qwen35-v2a-hot.gguf \
      -p "<|im_start|>user
$Q<|im_end|>
<|im_start|>assistant
<think>

</think>

" -n 120 -ngl 99 --no-repack -no-cnv -st --no-warmup -c 1024 --seed 1337 \
      --no-jinja --temp 0.6 --top-p 0.95 $K 2>/dev/null | tail -n +2
  echo
done
echo "RESULT73 질적 검증 완료"
