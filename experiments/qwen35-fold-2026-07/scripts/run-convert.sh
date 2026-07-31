#!/bin/bash
set -u
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
KILL_GB=24
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
echo "[$(date '+%Y-%m-%d %H:%M:%S')] MLX 변환 시작" >> exp/timeline.log
python3 heal/convert.py > heal/convert.log 2>&1 &
PID=$!
PEAK=0
while kill -0 $PID 2>/dev/null; do
  W=$(wired_gb)
  awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
  awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! 메모리 가드 발동 ${W}GB"; kill -9 $PID; break; }
  sleep 5
done
wait $PID 2>/dev/null
echo "peak ${PEAK} GB"
tail -2 heal/convert.log
echo "변환 종료"
