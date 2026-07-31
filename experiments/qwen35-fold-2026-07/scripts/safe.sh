#!/bin/bash
# 실험 실행의 유일한 관문. 2026-07-31 00:07 커널 패닉(watchdog timeout) 재발 방지.
#
# 그날의 원인: 체인 두 개가 같은 조건을 기다려 1초 차이로 동시에 시작 → 15GiB 모델 2개
# 동시 적재 → 스와핑에 커널이 묶여 94초 무응답 → 워치독 강제 재부팅.
# 가드는 "자기 프로세스만" 감시했고 실험 간 상호 배제가 아예 없었다.
#
# 이 스크립트가 강제하는 것:
#   1) 전역 단일 잠금(shlock) — 잠금 획득 실패 시 그냥 종료. 동시 실행이 구조적으로 불가.
#   2) 사전 점검 — 여유 메모리가 회복될 때까지 시작하지 않는다.
#   3) 시스템 전체 기준 감시(2초) — 프로세스별 wired 가 아니라 여유율/스왑을 본다.
#      스왑 증가가 패닉의 선행 신호이므로 스왑을 1차 지표로 쓴다.
#   4) 타임아웃 — 무한 실행 방지.
set -u
cd /Users/angigyeom/Desktop/DEV/fun/accelerator

LOCK=/tmp/accel-gpu.lock
FREE_MIN=12          # 시스템 여유율 하한 (%)
SWAP_MAX_MB=1536     # 스왑 사용 상한 (MB) — 패닉 선행 신호
WIRED_MAX_GB=26      # 최후 방어선
POLL=2               # 감시 주기(초)
PREFLIGHT_FREE=55    # 시작 전 요구 여유율 (%)

usage() { echo "사용법: safe.sh <타임아웃초> <로그> -- <명령...>"; exit 2; }
[ $# -ge 4 ] || usage
TIMEOUT=$1; LOG=$2; shift 2
[ "$1" = "--" ] || usage
shift

free_pct() { memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{gsub(/%/,"",$2); print $2+0; exit}'; }
swap_mb()  { sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); print $6+0}'; }
wired_gb() { vm_stat | awk '/Pages wired/{gsub(/\./,"",$4); printf "%.2f", $4*16384/1073741824}'; }
stamp()    { date '+%Y-%m-%d %H:%M:%S'; }

# ── 1) 전역 잠금 ──
if ! shlock -f "$LOCK" -p $$; then
  echo "[$(stamp)] 잠금 보유자 PID $(cat $LOCK 2>/dev/null) 실행 중 — 건너뜀: $*"
  exit 3
fi
trap 'rm -f "$LOCK"' EXIT

# ── 1.5) 디스크 안전선 ── (양자화/수술 작업이 디스크를 가득 채우면 시스템이 불안정해진다)
DF_GB=$(df -g . | awk 'NR==2{print $4}')
if (( DF_GB < 25 )); then
  echo "[$(stamp)] 디스크 여유 ${DF_GB}GiB < 25GiB — 중단: $*"
  exit 4
fi

# ── 2) 사전 점검: 여유 메모리 회복 대기 ──
for ((i=0; i<180; i++)); do
  F=$(free_pct); S=$(swap_mb)
  [ -n "$F" ] || F=0
  if (( ${F%.*} >= PREFLIGHT_FREE )) && (( ${S%.*} <= 256 )); then break; fi
  sleep 5
done
F=$(free_pct); S=$(swap_mb)
echo "[$(stamp)] 시작 (여유 ${F}% 스왑 ${S}MB wired $(wired_gb)GB): $*"

# ── 3) 실행 + 시스템 전체 감시 ──
nice -n 5 "$@" > "$LOG" 2>&1 &
PID=$!
KILLED=""
PEAK_W=0; PEAK_S=0; MIN_F=100
for ((t=0; t<TIMEOUT; t+=POLL)); do
  kill -0 $PID 2>/dev/null || break
  F=$(free_pct); S=$(swap_mb); W=$(wired_gb)
  [ -n "$F" ] || F=100
  (( ${F%.*} < MIN_F )) && MIN_F=${F%.*}
  (( ${S%.*} > PEAK_S )) && PEAK_S=${S%.*}
  awk -v w="$W" -v p="$PEAK_W" 'BEGIN{exit !(w>p)}' && PEAK_W=$W
  if (( ${S%.*} > SWAP_MAX_MB )); then KILLED="스왑 ${S}MB > ${SWAP_MAX_MB}MB"; break; fi
  if (( ${F%.*} < FREE_MIN ));    then KILLED="여유 ${F}% < ${FREE_MIN}%"; break; fi
  awk -v w="$W" -v k="$WIRED_MAX_GB" 'BEGIN{exit !(w>k)}' && { KILLED="wired ${W}GB > ${WIRED_MAX_GB}GB"; break; }
  sleep $POLL
done
if kill -0 $PID 2>/dev/null; then
  [ -n "$KILLED" ] || KILLED="타임아웃 ${TIMEOUT}초"
  # 자식까지 정리
  pkill -9 -P $PID 2>/dev/null
  kill -9 $PID 2>/dev/null
fi
wait $PID 2>/dev/null
RC=$?

echo "[$(stamp)] 종료 ${KILLED:+(중단: $KILLED)} 최저여유 ${MIN_F}% 최대스왑 ${PEAK_S}MB 최대wired ${PEAK_W}GB"

# ── 4) 정착 대기: 다음 작업이 잔여 메모리를 물려받지 않게 ──
for ((i=0; i<60; i++)); do
  F=$(free_pct); [ -n "$F" ] || F=100
  (( ${F%.*} >= PREFLIGHT_FREE )) && break
  sleep 5
done
exit $RC
