#!/bin/bash
# EXP-54 pool-32(top16 캡) 전층 확장: 덤프→선정→수술→ppl. 기준 A = 14.6569
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
bash $EXP/hotdump.sh >> $EXP/e54.log 2>&1   # 자기 라우터 확률 재덤프 (18층)
python3 - <<'PY'
import numpy as np, glob
with open("exp/hot32all_ids.txt","w") as f:
    for il in [23,24,25,27,28,29,30,31,33,34,35,36,37,39,40,41,42,43]:
        fs = sorted(glob.glob(f"exp/hotprobs/ffn_moe_probs-{il}_*.bin"))
        a = np.concatenate([np.fromfile(x, dtype=np.float32).reshape(-1,256) for x in fs])
        top = np.argsort(-a.mean(0))[:32]
        f.write(f"{il} " + " ".join(map(str, sorted(top.tolist()))) + "\n")
print("ids ok")
PY
sed -e "s|exp/hot_ids.txt|exp/hot32all_ids.txt|" -e "s|qwen35-v2a-hot.gguf|qwen35-v2a-hotP.gguf|" $EXP/hot-surgery.py > $EXP/hot-surgeryP.py
python3 $EXP/hot-surgeryP.py >> $EXP/e54.log 2>&1
grep -q "완료:" $EXP/e54.log || { echo "수술 실패"; exit 1; }
rm -rf $EXP/hotprobs

HP=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-v2a-hotP.gguf
PPL=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
WIKI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/scripts/wikitext-2-raw/wiki.test.raw
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
m=(); for ((il=0; il<48; il++)); do
  if (( il >= 44 )); then m[il]=$il
  else g=$(( il * 6 / 44 )); m[il]=$(( (g * 44 + 5) / 6 )); fi
done
IFS=,; MAP6="${m[*]}"; unset IFS
s=0; while (( s < 24 )); do W0=$(wired_gb); awk -v w="$W0" 'BEGIN{exit !(w<8)}' && break; sleep 5; s=$((s+1)); done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-54 pool32 ppl" >> $EXP/timeline.log
env LLAMA_ARG_REPACK=false LLAMA_MMAP_NO_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=999999 \
    LLAMA_FOLD_MAP="$MAP6" LLAMA_FOLD_SHEXP_LOGICAL=1 LLAMA_FOLD_GAMMA_FILE=$EXP/g46.txt \
    LLAMA_FOLD_HOT_SCALE=1.0 LLAMA_FOLD_SKIP_LOAD=1 LLAMA_MMAP_GPU_COPY=1 \
    "$PPL" -m "$HP" -f "$WIKI" --chunks 16 -c 512 --no-repack -b 512 -ngl 99 \
    --override-kv qwen35moe.expert_used_count=int:6 > $EXP/e54-ppl.log 2>&1 &
PID=$!; PEAK=0; i=0
while kill -0 $PID 2>/dev/null && (( i < 60 )); do
  W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
  awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill"; kill -9 $PID; break; }
  sleep 5; i=$((i+1))
done
kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
echo "pool32 ppl: $(grep -a 'Final estimate' $EXP/e54-ppl.log | tail -1 | sed 's/.*PPL = //')  peak ${PEAK} GB  (A 14.6569)"
echo "EXP-54 완료"
