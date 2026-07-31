#!/bin/bash
# EXP-66 유효한 생성형 MMLU (thinking 켬, 채팅 템플릿 사용). 접힌 Q3 구성.
set -u
KILL_GB=22
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
EXP=exp
M=models/qwen35-v2a-hot.gguf
SRV=nanfix/build/bin/llama-server
TAG=Q3
MAP=$(python3 -c "import json;print(json.load(open('$EXP/maps58.json'))['$TAG'])")
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
settle() { local s=0 W; while (( s < 48 )); do W=$(wired_gb); awk -v w="$W" 'BEGIN{exit !(w<9)}' && return; sleep 5; s=$((s+1)); done; }
settle
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-66 생성형 MMLU (thinking)" >> $EXP/timeline.log
# 채팅 템플릿 사용(--no-jinja 제거), thinking 을 위해 컨텍스트 넉넉히
env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
    LLAMA_FOLD_MAP=$MAP LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g58-$TAG.txt \
    LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
    $SRV -m $M -ngl 99 --no-repack -c 4096 -b 512 -ub 512 --port 8087 \
    --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e66-srv.log 2>&1 &
SPID=$!
for ((w=0; w<90; w++)); do
  curl -s http://127.0.0.1:8087/health 2>/dev/null | grep -q ok && break
  kill -0 $SPID 2>/dev/null || { echo "!! 서버 사망"; tail -5 $EXP/e66-srv.log; exit 1; }
  W=$(wired_gb); awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { kill -9 $SPID; echo "!! 가드"; exit 1; }
  sleep 5
done
echo "서버 준비 (wired $(wired_gb) GB, 채팅 템플릿 사용)"
( while kill -0 $SPID 2>/dev/null; do
    W=$(wired_gb); awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && kill -9 $SPID
    sleep 5
  done ) & WPID=$!
echo "── 추론 예산 강제 (600 토큰) ──"
python3 exp/genmc3.py --n 60 --think 600
echo "── 대조: 추론 없이 즉답 ──"
python3 exp/genmc3.py --n 60 --no-think
kill $SPID $WPID 2>/dev/null; wait 2>/dev/null
echo "EXP-66 완료"
