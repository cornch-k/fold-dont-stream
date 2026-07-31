#!/bin/bash
# EXP-68 결정적 대조: 네이티브 35B 를 접힌 모델과 **완전히 동일한** 생성형 하네스로 측정.
# 35B 도 30%대면 하네스 결함, 80%대면 접기 손실. 이 구분 없이는 EXP-66 을 해석할 수 없다.
set -u
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
EXP=exp
M=models/rival/Qwen3.5-35B-A3B-UD-IQ3_S.gguf
SRV=nanfix/build/bin/llama-server
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
settle() { local s=0 W; while (( s < 40 )); do W=$(wired_gb); awk -v w="$W" 'BEGIN{exit !(w<8)}' && return; sleep 5; s=$((s+1)); done; }
settle
echo "[$(date '+%Y-%m-%d %H:%M:%S')] EXP-68 35B 생성형 MMLU" >> $EXP/timeline.log
$SRV -m $M -ngl 99 --no-repack -c 4096 -b 512 -ub 512 --port 8087 > $EXP/e68-srv.log 2>&1 &
SPID=$!
for ((w=0; w<90; w++)); do
  curl -s http://127.0.0.1:8087/health 2>/dev/null | grep -q ok && break
  kill -0 $SPID 2>/dev/null || { echo "!! 서버 사망"; tail -4 $EXP/e68-srv.log; exit 1; }
  sleep 5
done
echo "35B 서버 준비 (wired $(wired_gb) GB)"
echo "── 35B 추론 예산 강제 (600) ──"
python3 exp/genmc3.py --n 60 --think 600
echo "── 35B 즉답 대조 ──"
python3 exp/genmc3.py --n 60 --no-think
kill $SPID 2>/dev/null; wait 2>/dev/null
echo "EXP-68 완료"
