#!/bin/bash
# 순차 실행: EXP-62(Q 스윕) → 최저 ppl 구성 벤치. 완료 마커는 출력 파일 문자열로만 판정.
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
echo $$ > exp/runchain.pid
bash exp/exp62.sh > exp/exp62.out 2>&1
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
echo "선택된 구성: $BEST (최저 ppl)" > exp/chain60.out
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-60 벤치 시작 (구성 $BEST)" >> exp/timeline.log
bash exp/exp60.sh "$BEST" > exp/exp60.out 2>&1
echo "체인 전체 종료" >> exp/chain60.out
rm -f exp/runchain.pid
