#!/bin/bash
# EXP-62 종료 대기 → F/Q1/Q2/Q3 중 최저 ppl 구성 선택 → 벤치 4종 + 순수 디코드 속도
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
while pgrep -f "exp61.sh|exp62.sh|chain62.sh" >/dev/null 2>&1; do sleep 20; done
BEST=$(python3 - <<'PY'
import re, glob
best, bt = None, None
for tag in ("F","Q1","Q2","Q3"):
    for f in glob.glob(f"exp/e58-{tag}-ppl.log"):
        for line in open(f, 'rb'):
            m = re.search(rb'Final estimate: PPL = ([\d.]+)', line)
            if m:
                v = float(m.group(1))
                if best is None or v < best: best, bt = v, tag
print(bt or "F")
PY
)
echo "선택된 구성: $BEST" 
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-60 벤치 시작 (구성 $BEST)" >> exp/timeline.log
bash exp/exp60.sh "$BEST" > exp/exp60.out 2>&1
