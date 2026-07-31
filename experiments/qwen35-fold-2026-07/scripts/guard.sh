#!/bin/bash
# 인자: KILL_GB LOGFILE -- 명령...
set -u
KILL_GB=$1; LOG=$2; shift 3
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
s=0; while (( s < 36 )); do W=$(wired_gb); awk -v w="$W" 'BEGIN{exit !(w<10)}' && break; sleep 5; s=$((s+1)); done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] $* (시작 wired $(wired_gb) GB)" >> exp/timeline.log
"$@" > "$LOG" 2>&1 &
PID=$!; PEAK=0
while kill -0 $PID 2>/dev/null; do
  W=$(wired_gb)
  awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
  awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! 메모리 가드 ${W}GB — 중단"; kill -9 $PID; break; }
  sleep 5
done
wait $PID 2>/dev/null
echo "peak ${PEAK} GB"
tail -6 "$LOG"
echo "GUARD 종료"
