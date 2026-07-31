#!/bin/bash
# EXP-28 무접기 Q3_K_S 기준선 ppl — CPU mmap 스트리밍 (fold 환경변수 전부 없음)
set -u
KILL_GB=22
EXP=/Users/angigyeom/Desktop/DEV/fun/accelerator/exp
Q3=/Users/angigyeom/Desktop/DEV/fun/accelerator/models/qwen35-q3ks.gguf
BIN=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/build/bin/llama-perplexity
WIKI=/Users/angigyeom/Desktop/DEV/fun/accelerator/nanfix/scripts/wikitext-2-raw/wiki.test.raw
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-28 무접기 기준선 (CPU 스트리밍) 시작" >> $EXP/timeline.log
"$BIN" -m "$Q3" -f "$WIKI" --chunks 16 -c 512 -b 512 --device none -ngl 0 --no-repack \
  > $EXP/baseline.log 2>&1 &
PID=$!; PEAK=0; i=0
while kill -0 $PID 2>/dev/null && (( i < 720 )); do   # 최대 1시간
  W=$(wired_gb); awk -v w="$W" -v p="$PEAK" 'BEGIN{exit !(w>p)}' && PEAK=$W
  awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! kill"; kill -9 $PID; break; }
  sleep 5; i=$((i+1))
done
kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-28 종료 peak=${PEAK}GB" >> $EXP/timeline.log
echo "무접기 Q3_K_S: $(grep -a 'Final estimate' $EXP/baseline.log | tail -1 | sed 's/.*PPL = //')  peak ${PEAK} GB"
