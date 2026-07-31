#!/bin/bash
# EXP-69 핫 재선정 (검증된 hotdump.sh 방식 그대로 — ROUTER_LOGICAL=1 필수, 베이스 모델 사용)
set -u
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
EXP=exp
MAP=$(python3 -c "import json;print(json.load(open('exp/maps58.json'))['Q3'])")
FILT=""
for il in 23 24 25 27 28 29 31 33 34 35 36 37 39 40 42 43; do FILT="$FILT --tensor-filter ffn_moe_probs-${il}\$"; done
if [ -s exp/hotQ3_ids.txt ]; then
  echo "RESULT69 이미 완료 — 재덤프 생략 (겹침 13.2/16 로 수술 불필요 판정)"; exit 0
fi
rm -rf $EXP/hotprobs2; mkdir -p $EXP/hotprobs2
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-69 재덤프 (ROUTER_LOGICAL)" >> $EXP/timeline.log
env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
    LLAMA_FOLD_MAP="$MAP" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g58-Q3.txt \
    LLAMA_FOLD_ROUTER_LOGICAL=1 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
    LLAMA_DEBUG_SAVE_DIR=$EXP/hotprobs2 \
    nanfix/build/bin/llama-debug -m models/qwen35-v2a.gguf -f $EXP/calib8k.txt \
    -ngl 99 --no-repack -c 8192 -b 8192 -ub 512 $FILT \
    --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e69-dump.log 2>&1
n=$(ls $EXP/hotprobs2 2>/dev/null | wc -l | tr -d ' ')
echo "RESULT69 덤프 텐서 $n 개"
[ "$n" -gt 0 ] || { echo "!! 덤프 실패"; tail -3 $EXP/e69-dump.log; exit 1; }
python3 - <<'PY'
import numpy as np, glob, re, json
maps=json.load(open("exp/maps58.json")); wl=[int(x) for x in maps["Q3"].split(",")]
gam=[float(x) for x in open("exp/g58-Q3.txt")]
need=[il for il in range(48) if wl[il]!=il and gam[il]>0]
old={}
for line in open("exp/hot_ids.txt"):
    p=line.split(); old[int(p[0])]=set(map(int,p[1:]))
out={}; ov=[]
for il in need:
    fs=sorted(glob.glob(f"exp/hotprobs2/ffn_moe_probs-{il}_*.bin"))
    if not fs: continue
    a=np.concatenate([np.fromfile(f,dtype=np.float32).reshape(-1,256) for f in fs])
    ids=sorted(np.argsort(-a.mean(0))[:16].tolist()); out[il]=ids
    if il in old: ov.append(len(set(ids)&old[il]))
with open("exp/hotQ3_ids.txt","w") as f:
    for il in sorted(out): f.write(f"{il} "+" ".join(map(str,out[il]))+"\n")
print(f"RESULT69 재선정 {len(out)}층, 기존 대비 평균 겹침 {np.mean(ov):.1f}/16" if ov else f"RESULT69 재선정 {len(out)}층 (비교 대상 없음)")
PY
echo "EXP-69 완료"
