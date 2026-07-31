#!/bin/bash
# 단일 순차 큐 실행기. 이 프로세스 하나만 존재하므로 경쟁이 구조적으로 불가능하다.
# (이전에는 체인 스크립트 6개가 각자 조건을 감시하다 둘이 동시에 터져 커널 패닉이 났다.)
#
# 큐 파일 형식: 한 줄에 "타임아웃초<TAB>라벨<TAB>명령..."  (# 으로 시작하면 주석)
set -u
cd /Users/angigyeom/Desktop/DEV/fun/accelerator
Q=${1:-exp/queue.txt}
echo $$ > exp/queue.pid
trap 'rm -f exp/queue.pid' EXIT

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(stamp)] 큐 시작: $Q"

n=0
while IFS=$'\t' read -r TO LABEL CMD; do
  case "$TO" in ''|'#'*) continue;; esac
  n=$((n+1))
  echo "════ [$(stamp)] ($n) $LABEL ════"
  echo "[$(stamp)] QUEUE 시작 $LABEL" >> exp/timeline.log
  # < /dev/null 필수: 루프 안 명령이 큐 파일 fd(stdin)를 소비하면 read 위치가 깨진다
  bash exp/safe.sh "$TO" "exp/q-$LABEL.log" -- bash -c "$CMD" < /dev/null
  rc=$?
  echo "[$(stamp)] ($n) $LABEL 종료 rc=$rc"
  # 결과 줄이 있으면 즉시 표면화
  grep -ahE "^RESULT" "exp/q-$LABEL.log" 2>/dev/null | sed 's/^/    /'
done < "$Q"
echo "[$(stamp)] 큐 전체 완료 ($n개)"
