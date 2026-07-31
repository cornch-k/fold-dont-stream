#!/bin/bash
# EXP-65 라이벌 ppl: 접기가 이기는 축이 있는지 판정하는 마지막 빈칸.
# 동일 프로토콜(wikitext-2 test, 16청크, c512 b512) 로 네이티브 35B 를 잰다.
set -u
KILL_GB=22
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
EXP=exp
PPL=nanfix/build/bin/llama-perplexity
WIKI=nanfix/scripts/wikitext-2-raw/wiki.test.raw
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
settle() { local s=0 W; while (( s < 48 )); do W=$(wired_gb); awk -v w="$W" 'BEGIN{exit !(w<9)}' && return; sleep 5; s=$((s+1)); done; }
guard() { local PID=$1 MAXI=$2 i=0 W
  while kill -0 $PID 2>/dev/null && (( i < MAXI )); do
    W=$(wired_gb); awk -v w="$W" -v k="$KILL_GB" 'BEGIN{exit !(w>k)}' && { echo "!! 가드 $W"; kill -9 $PID; return 1; }
    sleep 5; i=$((i+1))
  done; kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null; }

for M in models/rival/Qwen3.5-35B-A3B-UD-IQ3_S.gguf models/rival/Qwen3-30B-A3B-Instruct-2507-UD-Q3_K_XL.gguf; do
  B=$(basename $M .gguf)
  settle
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-65 $B ppl" >> $EXP/timeline.log
  $PPL -m $M -f $WIKI --chunks 16 -c 512 --no-repack -b 512 -ngl 99 > $EXP/e65-$B.log 2>&1 &
  guard $! 120
  echo "RESULT65 $B ppl=$(grep -a 'Final estimate' $EXP/e65-$B.log | tail -1 | sed 's/.*PPL = //')  파일 $(du -h $M | cut -f1)"
done
echo "EXP-65 완료"
