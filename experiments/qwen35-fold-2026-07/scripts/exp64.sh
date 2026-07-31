#!/bin/bash
# EXP-64 비트폭↔용량 거래: 전문가 q3_K→q2_K 로 물리층 10→13.
# 1) R13 맵으로 fold-aware imatrix 수집 (기존 imatrix 는 11층 맵용이라 새 물리층 미커버)
# 2) 전문가만 q2_K 재양자화 (나머지 비트폭 보존)
# 3) 측정: R13@q2_K(13층) vs Q3@q2_K(10층, 비트폭 비용 분리)
set -u
KILL_GB=22
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
EXP=exp
BIN=nanfix/build/bin
SRC=models/qwen35-v2a-hot.gguf
DST=models/qwen35-e2k.gguf
IMAT=$EXP/r13.imatrix
TRAIN=nanfix/scripts/wikitext-2-raw/wiki.train.raw
WIKI=nanfix/scripts/wikitext-2-raw/wiki.test.raw
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
settle() { local s=0 W; while (( s < 48 )); do W=$(wired_gb); awk -v w="$W" 'BEGIN{exit !(w<9)}' && return; sleep 5; s=$((s+1)); done; }
guard() { local PID=$1 MAXI=$2 i=0 W
  while kill -0 $PID 2>/dev/null && (( i < MAXI )); do
    W=$(wired_gb); awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! 메모리 가드 $W"; kill -9 $PID; return 1; }
    sleep 5; i=$((i+1))
  done; kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null; return 0; }

MAP13=$(python3 -c "import json;print(json.load(open('$EXP/maps58.json'))['R13'])")
E13="LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 LLAMA_FOLD_MAP=$MAP13 LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g58-R13.txt LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1"
K="--override-kv qwen35moe.expert_used_count=int:6"

# ── 1) imatrix (R13 맵) ──
if [ ! -f "$IMAT" ]; then
  settle
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-64 imatrix (R13)" >> $EXP/timeline.log
  env $E13 $BIN/llama-imatrix -m $SRC -f $TRAIN -o $IMAT --chunks 40 -c 512 -b 512 -ngl 99 \
      --no-repack $K > $EXP/e64-imat.log 2>&1 &
  guard $! 300 || exit 1
  [ -f "$IMAT" ] || { echo "!! imatrix 생성 실패"; tail -5 $EXP/e64-imat.log; exit 1; }
fi
python3 - <<'PY'
import sys, re; sys.path.insert(0,'nanfix/gguf-py')
from gguf import GGUFReader
n = [t.name for t in GGUFReader('exp/r13.imatrix').tensors]
exps = sorted({int(m.group(1)) for x in n if 'exps' in x for m in [re.search(r'blk\.(\d+)\.', x)] if m})
print(f"RESULT64 imatrix 전문가 커버 층: {exps}")
PY

# ── 2) 전문가만 q2_K 로 (나머지 현재 비트폭 보존) ──
if [ ! -f "$DST" ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-64 q2_K 재양자화" >> $EXP/timeline.log
  $BIN/llama-quantize --allow-requantize --imatrix $IMAT \
    --tensor-type ffn_gate_exps=q2_K --tensor-type ffn_up_exps=q2_K --tensor-type ffn_down_exps=q2_K \
    --tensor-type ffn_gate_hexps=q3_K --tensor-type ffn_up_hexps=q3_K --tensor-type ffn_down_hexps=q3_K \
    --tensor-type ffn_gate_shexp=q6_K --tensor-type ffn_up_shexp=q6_K --tensor-type ffn_down_shexp=q6_K \
    --tensor-type attn_qkv=q4_K --tensor-type attn_output=q4_K --tensor-type ssm_out=q4_K \
    --output-tensor-type q6_K --token-embedding-type q4_K \
    $SRC $DST Q3_K_S 6 > $EXP/e64-quant.log 2>&1
  [ -f "$DST" ] || { echo "!! 재양자화 실패"; tail -6 $EXP/e64-quant.log; exit 1; }
fi
python3 - <<'PY'
import sys, re, collections; sys.path.insert(0,'nanfix/gguf-py')
from gguf import GGUFReader
NM={0:'F32',10:'Q2_K',11:'Q3_K',12:'Q4_K',14:'Q6_K'}
g=collections.defaultdict(collections.Counter)
for t in GGUFReader('models/qwen35-e2k.gguf').tensors:
    g[re.sub(r'^blk\.\d+\.','',t.name)][NM.get(int(t.tensor_type),str(t.tensor_type))]+=1
for k in ('ffn_gate_exps.weight','ffn_gate_hexps.weight','ffn_gate_shexp.weight','attn_qkv.weight','token_embd.weight'):
    if k in g: print(f"RESULT64 비트폭 {k:26s} {dict(g[k])}")
PY

# ── 3) 측정 ──
run() { # $1=tag(맵) $2=라벨
  local TAG=$1 LBL=$2
  local MAP=$(python3 -c "import json;print(json.load(open('$EXP/maps58.json'))['$TAG'])")
  settle
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-64 $LBL ppl" >> $EXP/timeline.log
  env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
      LLAMA_FOLD_MAP=$MAP LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g58-$TAG.txt \
      LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
      $BIN/llama-perplexity -m $DST -f $WIKI --chunks 16 -c 512 --no-repack -b 512 -ngl 99 $K \
      > $EXP/e64-$TAG-ppl.log 2>&1 &
  guard $! 100
  local MEM=$(python3 $EXP/memcalc58.py $EXP/e64-$TAG-ppl.log $DST)
  echo "RESULT64 $LBL ppl=$(grep -a 'Final estimate' $EXP/e64-$TAG-ppl.log | tail -1 | sed 's/.*PPL = //')  $MEM"
}
run Q3 "q2_K@10층(비트폭 비용)"
run R13 "q2_K@13층(거래 결과)"
echo "EXP-64 완료"
