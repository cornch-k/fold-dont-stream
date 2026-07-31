#!/bin/bash
# EXP-56 생성형 MMLU 프로브 실행: 서버(A 구성) 기동 → 100문항 → 종료
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
HOT=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a-hot.gguf
SRV=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-server
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); for ((il=0; il<48; il++)); do
  if (( il >= 44 )); then m[il]=$il
  else g=$(( il * 6 / 44 )); m[il]=$(( (g * 44 + 5) / 6 )); fi
done
IFS=,; MAP6="${m[*]}"; unset IFS
s=0; while (( s < 24 )); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; s=$((s+1)); done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-56 생성형 MC 프로브" >> $EXP/timeline.log
env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
    LLAMA_FOLD_MAP="$MAP6" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g46.txt \
    LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
    "$SRV" -m "$HOT" -ngl 99 --no-repack -c 2048 -b 512 -ub 512 --port 8087 --no-jinja \
    --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e56-srv.log 2>&1 &
SPID=$!
# 서버 준비 대기 (모델 로드 ~1분)
for ((w=0; w<60; w++)); do
  curl -s http://127.0.0.1:8087/health 2>/dev/null | grep -q "ok" && break
  kill -0 $SPID 2>/dev/null || { echo "서버 사망"; exit 1; }
  W=$(wired_gb); awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { kill -9 $SPID; echo "!! kill"; exit 1; }
  sleep 5
done
echo "서버 준비 완료 (wired $(wired_gb) GB)"
# 감시 병행하며 프로브 실행
( while kill -0 $SPID 2>/dev/null; do
    W=$(wired_gb); awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && kill -9 $SPID
    sleep 5
  done ) &
WPID=$!
python3 exp/genmc.py
kill $SPID $WPID 2>/dev/null; wait 2>/dev/null
echo "EXP-56 완료"
